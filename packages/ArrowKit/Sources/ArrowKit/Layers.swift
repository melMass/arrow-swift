import Foundation
import MLX
import MLXFast
import MLXNN

/// timm's LayerScale: a learned per-channel gain.
final class LayerScale: Module, UnaryLayer {
    @ParameterInfo(key: "gamma") var gamma: MLXArray

    init(_ dim: Int) {
        _gamma.wrappedValue = MLXArray.ones([dim])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { x * gamma }
}

/// timm's Mlp: Linear, exact GELU, Linear.
final class Mlp: Module, UnaryLayer {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(_ dim: Int, hidden: Int) {
        _fc1.wrappedValue = Linear(dim, hidden)
        _fc2.wrappedValue = Linear(hidden, dim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(gelu(fc1(x))) }
}

/// DINOv2's fused SwiGLU: one projection to both halves, SiLU on the first.
final class SwiGLU: Module, UnaryLayer {
    @ModuleInfo(key: "w12") var w12: Linear
    @ModuleInfo(key: "w3") var w3: Linear

    /// `hidden` before DINOv2's 2/3 shrink, rounded up to a multiple of 8.
    init(_ dim: Int, hidden: Int) {
        let shrunk = (hidden * 2 / 3 + 7) / 8 * 8
        _w12.wrappedValue = Linear(dim, 2 * shrunk)
        _w3.wrappedValue = Linear(shrunk, dim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let halves = w12(x).split(parts: 2, axis: -1)
        return w3(silu(halves[0]) * halves[1])
    }
}

/// 2D rotary embedding (VGGT's): the head's first half rotates by the token's row,
/// the second half by its column, each with `frequency`-based angles on its own half.
struct RotaryTable {
    /// [tokens, half] each: cos and sin of the row (`y`) and column (`x`) angles.
    let cosY: MLXArray, sinY: MLXArray, cosX: MLXArray, sinX: MLXArray

    /// `positions`: (row, column) per token.
    init(positions: [(Int, Int)], headDim: Int, frequency: Float = 100) {
        let half = headDim / 2
        let inverse = stride(from: 0, to: half, by: 2).map { 1 / Foundation.pow(frequency, Float($0) / Float(half)) }
        func table(_ position: ((Int, Int)) -> Int) -> (MLXArray, MLXArray) {
            var cosines = [Float](), sines = [Float]()
            cosines.reserveCapacity(positions.count * half)
            sines.reserveCapacity(positions.count * half)
            for p in positions {
                let t = Float(position(p))
                let angles = inverse.map { t * $0 }
                for a in angles + angles {
                    cosines.append(Foundation.cos(a))
                    sines.append(Foundation.sin(a))
                }
            }
            return (MLXArray(cosines, [positions.count, half]), MLXArray(sines, [positions.count, half]))
        }
        (cosY, sinY) = table { $0.0 }
        (cosX, sinX) = table { $0.1 }
    }

    /// `x`: [batch, heads, tokens, headDim].
    func apply(_ x: MLXArray) -> MLXArray {
        let parts = x.split(parts: 2, axis: -1)
        return concatenated([Self.rotate(parts[0], cosY, sinY), Self.rotate(parts[1], cosX, sinX)], axis: -1)
    }

    private static func rotate(_ x: MLXArray, _ cos: MLXArray, _ sin: MLXArray) -> MLXArray {
        let halves = x.split(parts: 2, axis: -1)
        let turned = concatenated([-halves[1], halves[0]], axis: -1)
        return x * cos.asType(x.dtype) + turned * sin.asType(x.dtype)
    }
}

/// DINOv2 self-attention, with optional QK LayerNorm and 2D RoPE (DA3's later blocks).
final class SelfAttention: Module {
    let heads: Int
    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var proj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: LayerNorm?
    @ModuleInfo(key: "k_norm") var kNorm: LayerNorm?

    init(_ dim: Int, heads: Int, qkNorm: Bool) {
        self.heads = heads
        _qkv.wrappedValue = Linear(dim, 3 * dim, bias: true)
        _proj.wrappedValue = Linear(dim, dim)
        _qNorm.wrappedValue = qkNorm ? LayerNorm(dimensions: dim / heads) : nil
        _kNorm.wrappedValue = qkNorm ? LayerNorm(dimensions: dim / heads) : nil
    }

    func callAsFunction(_ x: MLXArray, rope: RotaryTable?) -> MLXArray {
        let (batch, count, channels) = (x.dim(0), x.dim(1), x.dim(2))
        let headDim = channels / heads
        let parts = qkv(x).reshaped(batch, count, 3, heads, headDim).transposed(2, 0, 3, 1, 4)
        var q = parts[0], k = parts[1]
        let v = parts[2]
        if let qNorm, let kNorm {
            q = qNorm(q)
            k = kNorm(k)
        }
        if let rope {
            q = rope.apply(q)
            k = rope.apply(k)
        }
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1 / Float(headDim).squareRoot(), mask: nil)
        return proj(out.transposed(0, 2, 1, 3).reshaped(batch, count, channels))
    }
}

/// A DINOv2 block: pre-norm attention and SwiGLU, each scaled by LayerScale.
final class EncoderBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attn") var attn: SelfAttention
    @ModuleInfo(key: "ls1") var ls1: LayerScale
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: SwiGLU
    @ModuleInfo(key: "ls2") var ls2: LayerScale

    init(_ dim: Int, heads: Int, qkNorm: Bool) {
        _norm1.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)
        _attn.wrappedValue = SelfAttention(dim, heads: heads, qkNorm: qkNorm)
        _ls1.wrappedValue = LayerScale(dim)
        _norm2.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)
        _mlp.wrappedValue = SwiGLU(dim, hidden: 4 * dim)
        _ls2.wrappedValue = LayerScale(dim)
    }

    func callAsFunction(_ x: MLXArray, rope: RotaryTable?) -> MLXArray {
        let h = x + ls1(attn(norm1(x), rope: rope))
        return h + ls2(mlp(norm2(h)))
    }
}

/// The keys and values a decoder block attends to: a function of the encoded
/// frames alone, so computed once and shared by every chunk of queries.
struct CrossMemory {
    /// [batch, heads, tokens, headDim] each.
    let keys: MLXArray
    let values: MLXArray
}

final class CrossAttention: Module {
    let heads: Int
    @ModuleInfo(key: "q_layer") var qLayer: Linear
    @ModuleInfo(key: "kv_layer") var kvLayer: Linear
    @ModuleInfo(key: "q_norm") var qNorm: LayerNorm
    @ModuleInfo(key: "k_norm") var kNorm: LayerNorm
    @ModuleInfo(key: "proj") var proj: Linear

    init(_ dim: Int, heads: Int) {
        self.heads = heads
        _qLayer.wrappedValue = Linear(dim, dim)
        _kvLayer.wrappedValue = Linear(dim, 2 * dim)
        _qNorm.wrappedValue = LayerNorm(dimensions: dim / heads)
        _kNorm.wrappedValue = LayerNorm(dimensions: dim / heads)
        _proj.wrappedValue = Linear(dim, dim)
    }

    /// `context` already normed by the block.
    func memory(_ context: MLXArray) -> CrossMemory {
        let (batch, count, channels) = (context.dim(0), context.dim(1), context.dim(2))
        let kv = kvLayer(context).reshaped(batch, count, 2, heads, channels / heads).transposed(2, 0, 3, 1, 4)
        return CrossMemory(keys: kNorm(kv[0]), values: kv[1])
    }

    func callAsFunction(_ query: MLXArray, memory: CrossMemory) -> MLXArray {
        let (batch, count, channels) = (query.dim(0), query.dim(1), query.dim(2))
        let headDim = channels / heads
        let q = qNorm(qLayer(query).reshaped(batch, count, heads, headDim).transposed(0, 2, 1, 3))
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: memory.keys, values: memory.values, scale: 1 / Float(headDim).squareRoot(), mask: nil)
        return proj(out.transposed(0, 2, 1, 3).reshaped(batch, count, channels))
    }
}

/// A decoder block: queries attend to the frames' patches (one LayerNorm for both),
/// then an MLP, each scaled by LayerScale.
final class CrossBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attn") var attn: CrossAttention
    @ModuleInfo(key: "ls1") var ls1: LayerScale
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: Mlp
    @ModuleInfo(key: "ls2") var ls2: LayerScale

    init(_ dim: Int, heads: Int) {
        _norm1.wrappedValue = LayerNorm(dimensions: dim)
        _attn.wrappedValue = CrossAttention(dim, heads: heads)
        _ls1.wrappedValue = LayerScale(dim)
        _norm2.wrappedValue = LayerNorm(dimensions: dim)
        _mlp.wrappedValue = Mlp(dim, hidden: 4 * dim)
        _ls2.wrappedValue = LayerScale(dim)
    }

    func memory(_ context: MLXArray) -> CrossMemory { attn.memory(norm1(context)) }

    func callAsFunction(_ query: MLXArray, memory: CrossMemory) -> MLXArray {
        let h = query + ls1(attn(norm1(query), memory: memory))
        return h + ls2(mlp(norm2(h)))
    }
}

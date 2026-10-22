import Foundation
import MLX
import MLXNN

/// Questions for the decoder, one per entry: the 3D point seen at pixel `xy` of
/// `source`, located in `target`, expressed in `reference`'s camera.
public struct Queries: Sendable {
    /// Pixel centers (x, y) in model-image pixels; fractions are kept.
    public var xy: [SIMD2<Float>]
    public var source: [Int32]
    public var target: [Int32]
    public var reference: [Int32]

    public init(xy: [SIMD2<Float>], source: [Int32], target: [Int32], reference: [Int32]) {
        precondition(source.count == xy.count && target.count == xy.count && reference.count == xy.count)
        self.xy = xy
        self.source = source
        self.target = target
        self.reference = reference
    }

    public var count: Int { xy.count }

    public static func + (a: Queries, b: Queries) -> Queries {
        var all = a
        all.append(b)
        return all
    }

    public mutating func append(_ other: Queries) {
        xy += other.xy; source += other.source; target += other.target; reference += other.reference
    }

    func slice(_ range: Range<Int>) -> Queries {
        Queries(xy: Array(xy[range]), source: Array(source[range]), target: Array(target[range]), reference: Array(reference[range]))
    }
}

/// The decoder's answers, one row per query, in float32.
public struct Predictions: Sendable {
    /// In the reference camera: x right, y down, z forward; relative scale.
    public var points: [SIMD3<Float>]
    /// ≥ 1; higher is surer.
    public var confidence: [Float]
    /// Where the point is in the target frame, normalized: pixel = uv · (W, H) − 0.5.
    public var uv: [SIMD2<Float>]
    /// > 0: visible in the target frame.
    public var visibility: [Float]
    /// Unit surface normals, reference camera.
    public var normals: [SIMD3<Float>]
    public var displacement: [SIMD3<Float>]

    public init() {
        points = []; confidence = []; uv = []; visibility = []; normals = []; displacement = []
    }

    public var count: Int { points.count }

    mutating func append(_ other: Predictions) {
        points += other.points; confidence += other.confidence; uv += other.uv
        visibility += other.visibility; normals += other.normals; displacement += other.displacement
    }
}

/// Encoded frames: what every query batch attends to.
public final class Encoded: @unchecked Sendable {
    /// [1, frames, 3, H, W] in [0, 1].
    let images: MLXArray
    /// [1, frames, 3072].
    let ids: MLXArray
    /// One per decoder block.
    let memories: [CrossMemory]
    public let frames: Int
    public let height: Int
    public let width: Int
    /// The frame DA3 put first, nil under three frames.
    public let reference: Int?

    init(images: MLXArray, ids: MLXArray, memories: [CrossMemory], reference: Int?) {
        self.images = images
        self.ids = ids
        self.memories = memories
        self.reference = reference
        frames = images.dim(1)
        height = images.dim(3)
        width = images.dim(4)
    }
}

/// The query token: a 9 × 9 RGB patch around the pixel, a Fourier code of its position,
/// and the three frames' identities.
final class QueryEmbedding: Module {
    static let patch = 9
    let dim: Int

    @ModuleInfo(key: "rgb_patch_embed") var rgbPatchEmbed: TwoLayer
    @ModuleInfo(key: "uv_fourier_proj") var uvFourierProj: TwoLayer
    @ModuleInfo(key: "camera_token_norm") var cameraTokenNorm: LayerNorm
    @ModuleInfo(key: "camera_relation_embed") var cameraRelationEmbed: TwoLayer

    init(dim: Int) {
        self.dim = dim
        _rgbPatchEmbed.wrappedValue = TwoLayer(3 * Self.patch * Self.patch, dim)
        _uvFourierProj.wrappedValue = TwoLayer(dim, dim)
        _cameraTokenNorm.wrappedValue = LayerNorm(dimensions: 2 * Encoder.dim)
        _cameraRelationEmbed.wrappedValue = TwoLayer(6 * Encoder.dim, dim)
    }

    func callAsFunction(_ queries: Queries, encoded: Encoded, padded: MLXArray, dtype: DType) -> MLXArray {
        let frames = encoded.frames, height = encoded.height, width = encoded.width
        let source = queries.source.map { min(max($0, 0), Int32(frames - 1)) }
        let target = queries.target.map { min(max($0, 0), Int32(frames - 1)) }
        let reference = queries.reference.map { min(max($0, 0), Int32(frames - 1)) }

        let rgb = rgbPatchEmbed(Self.samplePatches(padded, queries.xy, source: source, height: height, width: width).asType(dtype))

        var uv = [Float]()
        uv.reserveCapacity(queries.count * 2)
        for p in queries.xy {
            uv.append((p.x + 0.5) / Float(width) * 2 - 1)
            uv.append((p.y + 0.5) / Float(height) * 2 - 1)
        }
        let position = uvFourierProj(fourier(MLXArray(uv, [1, queries.count, 2])).asType(dtype))

        let identities = [source, target, reference].map { frame in
            cameraTokenNorm(encoded.ids.take(MLXArray(frame), axis: 1))
        }
        let camera = cameraRelationEmbed(concatenated(identities, axis: -1))
        return rgb + position + camera
    }

    /// [sin u, cos u, sin v, cos v] at `dim / 4` frequencies 100^(−k / (dim / 4)).
    func fourier(_ uv: MLXArray) -> MLXArray {
        let perAxis = dim / 4
        let omega = MLXArray((0 ..< perAxis).map { 1 / Foundation.pow(100, Float($0) / Float(perAxis)) })
        let u = uv[0..., 0..., 0 ..< 1] * omega, v = uv[0..., 0..., 1 ..< 2] * omega
        return concatenated([sin(u), cos(u), sin(v), cos(v)], axis: -1)
    }

    /// The model images edge-padded by 4: [frames, H + 8, W + 8, 3], flattened to rows of RGB.
    static func padImages(_ images: MLXArray) -> MLXArray {
        let pad = patch / 2
        let hwc = images[0].transposed(0, 2, 3, 1)
        let padded = MLX.padded(hwc, widths: [.init(0), .init(pad), .init(pad), .init(0)], mode: .edge)
        return padded.reshaped(-1, 3)
    }

    /// Each query's 9 × 9 patch around its rounded pixel, channel-major: [1, queries, 243].
    static func samplePatches(_ padded: MLXArray, _ xy: [SIMD2<Float>], source: [Int32], height: Int, width: Int) -> MLXArray {
        let pad = patch / 2
        let rowLength = width + 2 * pad, frameSize = (height + 2 * pad) * rowLength
        var index = [Int32]()
        index.reserveCapacity(xy.count * patch * patch)
        for (p, frame) in zip(xy, source) {
            // torch.round: halves to even.
            let y = min(max(Int(p.y.rounded(.toNearestOrEven)), 0), height - 1)
            let x = min(max(Int(p.x.rounded(.toNearestOrEven)), 0), width - 1)
            let base = Int(frame) * frameSize + y * rowLength + x
            for dy in 0 ..< patch {
                for dx in 0 ..< patch { index.append(Int32(base + dy * rowLength + dx)) }
            }
        }
        let rgb = padded.take(MLXArray(index), axis: 0).reshaped(xy.count, patch * patch, 3)
        return rgb.transposed(0, 2, 1).reshaped(1, xy.count, 3 * patch * patch)
    }
}

/// Linear, exact GELU, Linear: a torch Sequential's 0 and 2, renamed at load.
final class TwoLayer: Module, UnaryLayer {
    @ModuleInfo(key: "first") var first: Linear
    @ModuleInfo(key: "second") var second: Linear

    init(_ input: Int, _ dim: Int) {
        _first.wrappedValue = Linear(input, dim)
        _second.wrappedValue = Linear(dim, dim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { second(gelu(first(x))) }
}

/// An output head: LayerNorm, Linear to half width, GELU, Linear (a torch Sequential's
/// 0, 1 and 3, renamed at load).
final class Head: Module, UnaryLayer {
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "hidden") var hidden: Linear
    @ModuleInfo(key: "out") var out: Linear

    init(_ dim: Int, _ outputs: Int) {
        _norm.wrappedValue = LayerNorm(dimensions: dim)
        _hidden.wrappedValue = Linear(dim, dim / 2)
        _out.wrappedValue = Linear(dim / 2, outputs)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { out(gelu(hidden(norm(x)))) }
}

final class Heads: Module {
    @ModuleInfo(key: "points") var points: Head
    @ModuleInfo(key: "point_confidence") var pointConfidence: Head
    @ModuleInfo(key: "uv") var uv: Head
    @ModuleInfo(key: "visibility_logits") var visibility: Head
    @ModuleInfo(key: "normals") var normals: Head
    @ModuleInfo(key: "displacement") var displacement: Head

    init(_ dim: Int) {
        _points.wrappedValue = Head(dim, 3)
        _pointConfidence.wrappedValue = Head(dim, 1)
        _uv.wrappedValue = Head(dim, 2)
        _visibility.wrappedValue = Head(dim, 1)
        _normals.wrappedValue = Head(dim, 3)
        _displacement.wrappedValue = Head(dim, 3)
    }
}

final class Decoder: Module {
    @ModuleInfo(key: "blocks") var blocks: [CrossBlock]

    init(dim: Int, depth: Int, heads: Int) {
        _blocks.wrappedValue = (0 ..< depth).map { _ in CrossBlock(dim, heads: heads) }
    }
}

/// ARROW's network: the DA3 encoder, the query embedding, eight cross-attention blocks
/// and six heads. Weights keep torch's names.
final class ArrowNetwork: Module {
    static let decoderDim = 1280

    @ModuleInfo(key: "encoder") var encoder: Encoder
    @ModuleInfo(key: "query_embed") var queryEmbed: QueryEmbedding
    @ModuleInfo(key: "context_proj") var contextProj: Linear
    @ModuleInfo(key: "decoder") var decoder: Decoder
    @ModuleInfo(key: "heads") var heads: Heads

    override init() {
        _encoder.wrappedValue = Encoder()
        _queryEmbed.wrappedValue = QueryEmbedding(dim: Self.decoderDim)
        _contextProj.wrappedValue = Linear(2 * Encoder.dim, Self.decoderDim)
        _decoder.wrappedValue = Decoder(dim: Self.decoderDim, depth: 8, heads: 16)
        _heads.wrappedValue = Heads(Self.decoderDim)
    }

    /// The decoder blocks' keys and values over `patches` [1, frames, patches, 3072].
    func memories(_ patches: MLXArray) -> [CrossMemory] {
        let flat = patches.reshaped(1, -1, 2 * Encoder.dim)
        let context = contextProj(flat)
        return decoder.blocks.map { block in
            let memory = block.memory(context)
            eval(memory.keys, memory.values)
            return memory
        }
    }

    /// Raw head outputs for one batch of queries.
    func decode(_ queries: Queries, encoded: Encoded, padded: MLXArray, dtype: DType) -> [String: MLXArray] {
        var tokens = queryEmbed(queries, encoded: encoded, padded: padded, dtype: dtype)
        for (block, memory) in zip(decoder.blocks, encoded.memories) {
            tokens = block(tokens, memory: memory)
        }
        let f = { (head: Head) in head(tokens).asType(.float32) }
        let normals = f(heads.normals)
        let length = sqrt((normals * normals).sum(axis: -1, keepDims: true))
        return [
            "points": f(heads.points),
            "point_confidence": 1 + exp(f(heads.pointConfidence)),
            "uv": f(heads.uv),
            "visibility_logits": f(heads.visibility),
            "normals": normals / maximum(length, 1e-6),
            "displacement": f(heads.displacement),
        ]
    }
}

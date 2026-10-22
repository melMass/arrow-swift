import Foundation
import MLX
import MLXNN

/// Depth Anything 3's Giant backbone (DINOv2-g), as ARROW uses it.
///
/// Blocks 0–12 attend within each frame. From block 13 on, QK-normed attention with 2D
/// RoPE alternates: odd blocks attend across every frame (all patches at one shared
/// position), even ones within a frame (patches at their grid positions). Before block
/// 12 the frames are reordered so a reference frame comes first; before block 13 each
/// frame's CLS token becomes a camera token, the reference's own and a shared one for
/// the rest. Output per patch: the last within-frame features beside the normed last
/// across-frame features (3072), and per frame its first token, the frame's identity.
final class Encoder: Module {
    static let dim = 1536
    static let heads = 24
    static let depth = 40
    static let patchSize = 14
    /// The pretraining grid, 37 × 37 patches (518 px).
    static let gridSide = 37

    @ModuleInfo(key: "patch_embed") var patchEmbed: PatchEmbed
    @ParameterInfo(key: "cls_token") var clsToken: MLXArray
    @ParameterInfo(key: "camera_token") var cameraToken: MLXArray
    @ParameterInfo(key: "pos_embed") var posEmbed: MLXArray
    @ModuleInfo(key: "blocks") var blocks: [EncoderBlock]
    @ModuleInfo(key: "norm") var norm: LayerNorm

    private var positionCache: [Int: MLXArray] = [:]

    override init() {
        _patchEmbed.wrappedValue = PatchEmbed()
        _clsToken.wrappedValue = MLXArray.zeros([1, 1, Self.dim])
        _cameraToken.wrappedValue = MLXArray.zeros([1, 2, Self.dim])
        _posEmbed.wrappedValue = MLXArray.zeros([1, 1 + Self.gridSide * Self.gridSide, Self.dim])
        _blocks.wrappedValue = (0 ..< Self.depth).map { EncoderBlock(Self.dim, heads: Self.heads, qkNorm: $0 >= 13) }
        _norm.wrappedValue = LayerNorm(dimensions: Self.dim)
    }

    /// ImageNet's RGB mean and standard deviation.
    static var mean: MLXArray { MLXArray([0.485, 0.456, 0.406] as [Float]).reshaped(1, 1, 3, 1, 1) }
    static var std: MLXArray { MLXArray([0.229, 0.224, 0.225] as [Float]).reshaped(1, 1, 3, 1, 1) }

    /// Whether block `i` attends across frames.
    static func isGlobal(_ i: Int) -> Bool { i >= 13 && i % 2 == 1 }

    /// `images`: [1, frames, 3, H, W] in [0, 1], H and W multiples of 14. Returns patch
    /// features [1, frames, patches, 3072], ID tokens [1, frames, 3072] and the reference
    /// frame chosen before block 12 (nil with fewer than three frames).
    func callAsFunction(_ images: MLXArray, dtype: DType) -> (patches: MLXArray, ids: MLXArray, reference: Int?) {
        let frames = images.dim(1), height = images.dim(3), width = images.dim(4)
        let rows = height / Self.patchSize, columns = width / Self.patchSize
        let tokens = 1 + rows * columns

        let normalized = ((images - Self.mean) / Self.std).asType(dtype)
        var x = patchEmbed(normalized.reshaped(frames, 3, height, width))
        let cls = broadcast(clsToken.asType(dtype), to: [frames, 1, Self.dim])
        x = concatenated([cls, x], axis: 1) + positions(rows: rows, columns: columns).asType(dtype)

        let local = RotaryTable(positions: [(0, 0)] + (0 ..< rows * columns).map { ($0 / columns + 1, $0 % columns + 1) },
                                headDim: Self.dim / Self.heads)
        // Across frames every patch sits at (1, 1) and the special token at (0, 0).
        let global = RotaryTable(positions: (0 ..< frames).flatMap { _ in [(0, 0)] + Array(repeating: (1, 1), count: rows * columns) },
                                 headDim: Self.dim / Self.heads)

        var withinFrame = x
        var reference: Int?
        for (i, block) in blocks.enumerated() {
            if i == 12, frames >= 3 {
                let chosen = Self.referenceFrame(x[0..., 0])
                reference = chosen
                let order = MLXArray(Self.referenceFirst(chosen, frames: frames).map(Int32.init))
                x = x.take(order, axis: 0)
                withinFrame = withinFrame.take(order, axis: 0)
            }
            if i == 13 {
                let camera = cameraToken.asType(dtype)
                let first = camera[0..., 0 ..< 1]
                let rest = broadcast(camera[0..., 1 ..< 2], to: [frames - 1, 1, Self.dim])
                x = concatenated([concatenated([first, rest], axis: 0), x[0..., 1...]], axis: 1)
            }
            if Self.isGlobal(i) {
                x = block(x.reshaped(1, frames * tokens, Self.dim), rope: global).reshaped(frames, tokens, Self.dim)
            } else {
                x = block(x, rope: i >= 13 ? local : nil)
                withinFrame = x
            }
            // One block's graph at a time: forty unevaluated blocks hold every activation.
            eval(x, withinFrame)
        }

        var joined = concatenated([withinFrame, x], axis: -1)
        if let reference {
            let restore = MLXArray(Self.originalOrder(reference, frames: frames).map(Int32.init))
            joined = joined.take(restore, axis: 0)
        }
        let ids = joined[0..., 0]
        let patches = concatenated([joined[0..., 1..., ..<Self.dim], norm(joined[0..., 1..., Self.dim...])], axis: -1)
        return (patches.expandedDimensions(axis: 0), ids.expandedDimensions(axis: 0), reference)
    }

    /// DA3's choice: the frame whose CLS token is most average in mutual similarity, norm
    /// and variance. `cls`: [frames, dim].
    static func referenceFrame(_ cls: MLXArray) -> Int {
        let features = cls.asType(.float32)
        let norms = sqrt((features * features).sum(axis: -1))
        let unit = features / norms.expandedDimensions(axis: -1)
        let frames = features.dim(0)
        let similarity = matmul(unit, unit.transposed()) - MLXArray.eye(frames)
        let score = similarity.sum(axis: -1) / Float(frames - 1)
        // torch.var: unbiased.
        let variance = unit.variance(axis: -1, ddof: 1)
        func normalize(_ metric: MLXArray) -> MLXArray {
            let low = metric.min(), high = metric.max()
            return (metric - low) / (high - low + 1e-8)
        }
        let balance = abs(normalize(score) - 0.5) + abs(normalize(norms) - 0.5) + abs(normalize(variance) - 0.5)
        return balance.argMin().item(Int.self)
    }

    /// Input frame at each position once `reference` is moved to the front.
    static func referenceFirst(_ reference: Int, frames: Int) -> [Int] {
        [reference] + (0 ..< frames).filter { $0 != reference }
    }

    /// Position of each input frame in the reordered sequence.
    static func originalOrder(_ reference: Int, frames: Int) -> [Int] {
        (0 ..< frames).map { $0 < reference ? $0 + 1 : ($0 == reference ? 0 : $0) }
    }

    /// The position embedding resized to the grid, cached per size.
    private func positions(rows: Int, columns: Int) -> MLXArray {
        let key = rows << 16 | columns
        if let cached = positionCache[key] { return cached }
        let resized = Self.resizePositions(posEmbed, rows: rows, columns: columns)
        positionCache[key] = resized
        return resized
    }

    /// DINOv2's `interpolate_pos_encoding`: the 37 × 37 table resized by bicubic
    /// interpolation (A = −0.75, no antialiasing) with scale factors (n + 0.1) / 37, which
    /// PyTorch uses as given to map output to input coordinates. The CLS entry is kept.
    static func resizePositions(_ table: MLXArray, rows: Int, columns: Int) -> MLXArray {
        let side = gridSide
        if rows == side, columns == side { return table }
        let values = table.asType(.float32).asArray(Float.self)
        let dim = table.dim(2)
        let rowWeights = bicubicWeights(output: rows, input: side, scale: (Float(rows) + 0.1) / Float(side))
        let columnWeights = bicubicWeights(output: columns, input: side, scale: (Float(columns) + 0.1) / Float(side))
        var out = [Float](repeating: 0, count: (1 + rows * columns) * dim)
        out.replaceSubrange(0 ..< dim, with: values[0 ..< dim])
        for r in 0 ..< rows {
            for c in 0 ..< columns {
                let target = (1 + r * columns + c) * dim
                for (sr, wr) in rowWeights[r] {
                    for (sc, wc) in columnWeights[c] {
                        let weight = wr * wc
                        let source = (1 + sr * side + sc) * dim
                        for d in 0 ..< dim { out[target + d] += weight * values[source + d] }
                    }
                }
            }
        }
        return MLXArray(out, [1, 1 + rows * columns, dim])
    }

    /// Source indices (clamped to the border) and cubic weights for each output index.
    static func bicubicWeights(output: Int, input: Int, scale: Float) -> [[(Int, Float)]] {
        let a: Float = -0.75
        func near(_ x: Float) -> Float { ((a + 2) * x - (a + 3)) * x * x + 1 }
        func far(_ x: Float) -> Float { ((a * x - 5 * a) * x + 8 * a) * x - 4 * a }
        return (0 ..< output).map { o in
            let real = (Float(o) + 0.5) / scale - 0.5
            let base = Int(real.rounded(.down))
            let t = real - Float(base)
            let weights = [far(t + 1), near(t), near(1 - t), far(2 - t)]
            return (0 ..< 4).map { (min(max(base - 1 + $0, 0), input - 1), weights[$0]) }
        }
    }
}

/// timm's PatchEmbed: a 14 × 14 stride-14 convolution, written as a matmul over
/// unfolded patches (channel-major, as the convolution's weight is laid out).
final class PatchEmbed: Module {
    @ModuleInfo(key: "proj") var proj: Linear

    override init() {
        _proj.wrappedValue = Linear(3 * Encoder.patchSize * Encoder.patchSize, Encoder.dim)
    }

    /// `x`: [images, 3, H, W] → [images, patches, dim].
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let p = Encoder.patchSize
        let (count, height, width) = (x.dim(0), x.dim(2), x.dim(3))
        let rows = height / p, columns = width / p
        let unfolded = x.reshaped(count, 3, rows, p, columns, p).transposed(0, 2, 4, 1, 3, 5)
            .reshaped(count, rows * columns, 3 * p * p)
        return proj(unfolded)
    }
}

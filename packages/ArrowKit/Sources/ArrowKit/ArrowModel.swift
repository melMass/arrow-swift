import Foundation
import MLX
import MLXNN

/// The arithmetic the network runs in. Weights are stored in fp32; they are cast once
/// at load.
public enum ArrowPrecision: String, Sendable, CaseIterable {
    case float16, bfloat16, float32

    var dtype: DType {
        switch self {
        case .float16: .float16
        case .bfloat16: .bfloat16
        case .float32: .float32
        }
    }
}

/// ARROW (ifradlin/arrow) on MLX: encode a set of frames once, then answer any number
/// of point queries against the encoding.
public final class ArrowModel: @unchecked Sendable {
    let network: ArrowNetwork
    public let precision: ArrowPrecision

    init(network: ArrowNetwork, precision: ArrowPrecision) {
        self.network = network
        self.precision = precision
    }

    /// Loads `model.safetensors` from `directory`, casting its weights to `precision`.
    public static func load(from directory: URL, precision: ArrowPrecision = .float16) throws -> ArrowModel {
        let network = ArrowNetwork()
        var weights = try loadArrays(url: directory.appending(path: "model.safetensors"))
        weights["encoder.image_mean"] = nil
        weights["encoder.image_std"] = nil
        if let conv = weights["encoder.patch_embed.proj.weight"] {
            weights["encoder.patch_embed.proj.weight"] = conv.reshaped(conv.dim(0), -1)
        }
        let dtype = precision.dtype
        var renamed: [String: MLXArray] = [:]
        for (key, value) in weights {
            // The positional table is resized in float32 per image size; it stays float32.
            renamed[Self.moduleKey(key)] = key == "encoder.pos_embed" ? value : value.asType(dtype)
        }
        weights = renamed
        try network.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        eval(network.parameters())
        return ArrowModel(network: network, precision: precision)
    }

    /// torch Sequential indices as the layers' names: MLX reads a numeric key as an
    /// array index. Heads are (0 norm, 1 hidden, 3 out); two-layer MLPs (0 first, 2 second).
    static func moduleKey(_ key: String) -> String {
        var parts = key.split(separator: ".").map(String.init)
        guard parts.count >= 4 else { return key }
        let names: [String: String]
        if parts[0] == "heads" {
            names = ["0": "norm", "1": "hidden", "3": "out"]
        } else if parts[0] == "query_embed", ["rgb_patch_embed", "uv_fourier_proj", "camera_relation_embed"].contains(parts[1]) {
            names = ["0": "first", "2": "second"]
        } else {
            return key
        }
        if let name = names[parts[2]] { parts[2] = name }
        return parts.joined(separator: ".")
    }

    /// Encodes `images` [1, frames, 3, H, W] in [0, 1] (H and W multiples of 14).
    ///
    /// The decoder's keys and values (8 blocks × 2 × tokens × 1280) are computed once and
    /// held when they fit `memoryBudget` bytes: about 30 MB per frame at 504 × 280 in
    /// fp16. Past it, only the projected patches are held, a sixteenth of that, and each
    /// query batch recomputes the keys and values block by block: the same answers for
    /// some time per batch.
    public func encode(_ images: MLXArray, memoryBudget: Int = 2 << 30) -> Encoded {
        let dtype = precision.dtype
        let (patches, ids, reference) = network.encoder(images, dtype: dtype)
        eval(patches, ids)
        let context = network.context(patches)
        eval(context)
        let tokens = context.dim(1), blocks = network.decoder.blocks.count
        let memoryBytes = blocks * 2 * tokens * ArrowNetwork.decoderDim * dtype.size
        let memories = memoryBytes <= memoryBudget ? network.memories(context) : nil
        return Encoded(images: images.asType(.float32), ids: ids, context: memories == nil ? context : nil,
                       memories: memories, reference: reference)
    }

    /// Answers `queries` in chunks of `chunk`, reporting the fraction done after each.
    /// Cancellation is checked between chunks.
    public func decode(
        _ queries: Queries, encoded: Encoded, chunk: Int = 16_384,
        progress: (Double) -> Void = { _ in }
    ) throws -> Predictions {
        let padded = QueryEmbedding.padImages(encoded.images)
        var result = Predictions()
        result.points.reserveCapacity(queries.count)
        var start = 0
        while start < queries.count {
            try Task.checkCancellation()
            let end = min(start + chunk, queries.count)
            let raw = network.decode(queries.slice(start ..< end), encoded: encoded, padded: padded, dtype: precision.dtype)
            eval(Array(raw.values))
            result.append(Self.predictions(raw))
            start = end
            progress(Double(end) / Double(queries.count))
        }
        return result
    }

    static func predictions(_ raw: [String: MLXArray]) -> Predictions {
        func vec3(_ key: String) -> [SIMD3<Float>] {
            let v = raw[key]!.asArray(Float.self)
            return stride(from: 0, to: v.count, by: 3).map { SIMD3(v[$0], v[$0 + 1], v[$0 + 2]) }
        }
        let uv = raw["uv"]!.asArray(Float.self)
        var p = Predictions()
        p.points = vec3("points")
        p.confidence = raw["point_confidence"]!.asArray(Float.self)
        p.uv = stride(from: 0, to: uv.count, by: 2).map { SIMD2(uv[$0], uv[$0 + 1]) }
        p.visibility = raw["visibility_logits"]!.asArray(Float.self)
        p.normals = vec3("normals")
        p.displacement = vec3("displacement")
        return p
    }
}

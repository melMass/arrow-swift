import Foundation
import MLX
@testable import ArrowKit

/// One of `Fixtures/*.safetensors` (written by `Fixtures/capture.py`): arrays by key, and
/// the case that produced them.
struct Fixture {
    static let directory = URL(filePath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appending(path: "Fixtures")

    let arrays: [String: MLXArray]
    let spec: [String: Any]

    init(_ name: String) throws {
        let url = Self.directory.appending(path: "\(name).safetensors")
        let (arrays, metadata) = try loadArraysAndMetadata(url: url)
        self.arrays = arrays
        spec = try JSONSerialization.jsonObject(with: Data((metadata["case"] ?? "{}").utf8)) as? [String: Any] ?? [:]
    }

    subscript(key: String) -> MLXArray {
        guard let array = arrays[key] else { fatalError("fixture has no \(key)") }
        return array
    }

    func queries() -> Queries {
        let xy = self["xy"].asType(.float32).reshaped(-1, 2).asArray(Float.self)
        func frames(_ key: String) -> [Int32] { self[key].asType(.int32).reshaped(-1).asArray(Int32.self) }
        return Queries(
            xy: stride(from: 0, to: xy.count, by: 2).map { SIMD2(xy[$0], xy[$0 + 1]) },
            source: frames("source_frame"), target: frames("target_frame"), reference: frames("reference_frame"))
    }
}

/// The weights in the Hugging Face cache, or `ARROW_WEIGHTS`.
enum Weights {
    static var directory: URL? {
        if let path = ProcessInfo.processInfo.environment["ARROW_WEIGHTS"] { return URL(filePath: path) }
        return ArrowStore().cachedSnapshot()
    }

    static var enabled: Bool { ProcessInfo.processInfo.environment["ARROW_MODEL_TESTS"] == "1" && directory != nil }
}

/// Largest absolute difference, and cosine similarity of the flattened arrays.
func compare(_ a: MLXArray, _ b: MLXArray) -> (maxError: Float, cosine: Float) {
    let x = a.asType(.float32).reshaped(-1), y = b.asType(.float32).reshaped(-1)
    let maxError = abs(x - y).max().item(Float.self)
    let cosine = ((x * y).sum() / (sqrt((x * x).sum()) * sqrt((y * y).sum()))).item(Float.self)
    return (maxError, cosine)
}

import CoreML
import Foundation
import HuggingFace

/// Which pixels of each frame are sky, at model-image size.
public struct SkyMask: Sendable {
    public let width: Int
    public let height: Int
    /// Frame-major, row-major: frame f's pixel (x, y) at `f · width · height + y · width + x`.
    public let sky: [Bool]

    public init(width: Int, height: Int, sky: [Bool]) {
        self.width = width
        self.height = height
        self.sky = sky
    }

    public var frames: Int { sky.count / max(width * height, 1) }

    /// Whether a query's source pixel is sky: its rounded, clamped pixel (upstream's
    /// `sky_queries`).
    public func contains(_ xy: SIMD2<Float>, frame: Int) -> Bool {
        let x = min(max(Int(xy.x.rounded(.toNearestOrEven)), 0), width - 1)
        let y = min(max(Int(xy.y.rounded(.toNearestOrEven)), 0), height - 1)
        return sky[(frame * height + y) * width + x]
    }

    /// The share of all pixels that are sky.
    public var fraction: Double { Double(sky.lazy.filter { $0 }.count) / Double(max(sky.count, 1)) }

    /// Per query, whether its source pixel is sky.
    public func skipped(_ queries: Queries) -> [Bool] {
        zip(queries.xy, queries.source).map { contains($0, frame: Int($1)) }
    }
}

/// Upstream's sky segmenter (JianyuanWang/skyseg, a U²-Net) in Core ML, run on the Neural
/// Engine where it can: each frame at 320 × 320 with ImageNet normalization, the score map
/// stretched to [0, 1] per frame and quantized to 8 bits, resized back, sky at ≥ 32/255
/// (upstream's `segment_sky`). ARROW skips queries whose source pixel is sky.
public final class SkySegmenter: @unchecked Sendable {
    public static let repo = "melmass/skyseg-coreml"
    static let package = "SkySegmentation.mlpackage"
    /// The package's files, named one by one: a folder pattern also matches the folders
    /// themselves, which the hub cannot serve.
    static let files = ["Manifest.json", "Data/com.apple.CoreML/model.mlmodel", "Data/com.apple.CoreML/weights/weight.bin"]
    public static let downloadSize = "84 MB"
    static let side = 320
    static let threshold: Float = 32 / 255
    static let mean: [Float] = [0.485, 0.456, 0.406]
    static let deviation: [Float] = [0.229, 0.224, 0.225]

    let model: MLModel

    init(model: MLModel) {
        self.model = model
    }

    /// Downloads the package into the Hugging Face cache if needed, compiles it once
    /// (kept beside the snapshot) and loads it.
    public static func load(
        computeUnits: MLComputeUnits = .all, cache: HubCache = .default,
        progress: @MainActor @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> SkySegmenter {
        guard let id = Repo.ID(rawValue: repo) else { throw CocoaError(.fileNoSuchFile) }
        let snapshot: URL
        if let manifest = cache.cachedFilePath(repo: id, kind: .model, revision: "main", filename: "\(package)/Manifest.json") {
            snapshot = manifest.deletingLastPathComponent().deletingLastPathComponent()
        } else {
            snapshot = try await HubClient(cache: cache).downloadSnapshot(
                of: id, kind: .model, revision: "main", matching: files.map { "\(package)/\($0)" },
                progressHandler: progress)
        }
        let compiled = try compiledModel(snapshot.appending(path: package))
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        return SkySegmenter(model: try MLModel(contentsOf: compiled, configuration: configuration))
    }

    /// The compiled model, in Caches keyed by the package's revision folder, compiled on
    /// first use.
    static func compiledModel(_ package: URL) throws -> URL {
        let revision = package.deletingLastPathComponent().lastPathComponent
        let folder = try FileManager.default.url(for: .cachesDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
            .appending(path: "arrow-swift/skyseg-\(revision)")
        let compiled = folder.appending(path: "SkySegmentation.mlmodelc")
        if FileManager.default.fileExists(atPath: compiled.path) { return compiled }
        // The hub cache holds each file as a symlink to a blob, which the Core ML compiler
        // does not follow inside a package: compile a plain copy.
        let plain = FileManager.default.temporaryDirectory.appending(path: "skyseg-\(UUID().uuidString)/\(Self.package)")
        defer { try? FileManager.default.removeItem(at: plain.deletingLastPathComponent()) }
        for file in files {
            let destination = plain.appending(path: file)
            try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            try FileManager.default.copyItem(at: package.appending(path: file).resolvingSymlinksInPath(), to: destination)
        }
        let temporary = try MLModel.compileModel(at: plain)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try? FileManager.default.removeItem(at: compiled)
        try FileManager.default.moveItem(at: temporary, to: compiled)
        return compiled
    }

    /// The sky in every frame of `frames`.
    public func mask(_ frames: FrameSet) throws -> SkyMask {
        let (width, height) = (frames.width, frames.height)
        var sky: [Bool] = []
        sky.reserveCapacity(frames.images.count * width * height)
        for image in frames.images {
            try Task.checkCancellation()
            let rgba = FrameSet.rgba(image, width: width, height: height)
            sky += try mask(rgba: rgba, width: width, height: height)
        }
        return SkyMask(width: width, height: height, sky: sky)
    }

    /// One frame's sky, RGBA bytes in, a `width × height` mask out.
    func mask(rgba: [UInt8], width: Int, height: Int) throws -> [Bool] {
        let side = Self.side
        let input = try MLMultiArray(shape: [1, 3, NSNumber(value: side), NSNumber(value: side)], dataType: .float32)
        let pointer = input.dataPointer.assumingMemoryBound(to: Float.self)
        for channel in 0 ..< 3 {
            let plane = (0 ..< width * height).map { Float(rgba[$0 * 4 + channel]) / 255 }
            let resized = Self.bilinear(plane, from: (width, height), to: (side, side))
            let base = channel * side * side
            for i in 0 ..< side * side {
                pointer[base + i] = (resized[i] - Self.mean[channel]) / Self.deviation[channel]
            }
        }
        let output = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["image": input]))
        guard let score = output.featureValue(for: "score")?.multiArrayValue else { throw CocoaError(.coderReadCorrupt) }
        var scores = (0 ..< side * side).map { score[$0].floatValue }
        let low = scores.min() ?? 0, high = scores.max() ?? 1
        let span = high - low
        scores = scores.map { span > 0 ? Foundation.floor((($0 - low) / span) * 255) / 255 : 0 }
        return Self.bilinear(scores, from: (side, side), to: (width, height)).map { $0 >= Self.threshold }
    }

    /// PyTorch's bilinear `interpolate` (align_corners false, no antialiasing) of one
    /// row-major plane.
    static func bilinear(_ plane: [Float], from source: (Int, Int), to target: (Int, Int)) -> [Float] {
        func taps(_ input: Int, _ output: Int) -> [(Int, Int, Float)] {
            let scale = Float(input) / Float(output)
            return (0 ..< output).map { o in
                let position = max((Float(o) + 0.5) * scale - 0.5, 0)
                let low = min(Int(position), input - 1)
                let high = min(low + 1, input - 1)
                return (low, high, position - Float(low))
            }
        }
        let xs = taps(source.0, target.0), ys = taps(source.1, target.1)
        var out = [Float](repeating: 0, count: target.0 * target.1)
        for (row, (y0, y1, ly)) in ys.enumerated() {
            for (column, (x0, x1, lx)) in xs.enumerated() {
                let top = plane[y0 * source.0 + x0] * (1 - lx) + plane[y0 * source.0 + x1] * lx
                let bottom = plane[y1 * source.0 + x0] * (1 - lx) + plane[y1 * source.0 + x1] * lx
                out[row * target.0 + column] = top * (1 - ly) + bottom * ly
            }
        }
        return out
    }
}

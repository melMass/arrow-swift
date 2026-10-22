import CoreGraphics
import Foundation
import ImageIO
import simd
import UniformTypeIdentifiers

/// A result saved whole, to reopen without the model: the prepared frames (PNG), every
/// query and prediction (packed little-endian float32 / int32), the task groups, the
/// cameras and the timings, in one binary property list. `sources` are the videos or
/// folders the frames came from, one per camera.
public enum ResultArchive {
    public static let fileExtension = "quiver"
    static let version = 1

    public enum Failure: LocalizedError {
        case unreadable(String)

        public var errorDescription: String? {
            switch self {
            case let .unreadable(reason): "Not a Quiver result: \(reason)."
            }
        }
    }

    public static func write(_ result: ArrowResult, sources: [URL] = [], to url: URL) throws {
        try data(result, sources: sources).write(to: url, options: .atomic)
    }

    public static func read(_ url: URL) throws -> (result: ArrowResult, sources: [URL]) {
        try result(from: Data(contentsOf: url))
    }

    public static func data(_ result: ArrowResult, sources: [URL] = []) throws -> Data {
        let p = result.predictions, q = result.queries
        var archive: [String: Any] = [
            "version": version,
            "width": result.frames.width,
            "height": result.frames.height,
            "views": result.frames.views,
            "times": result.frames.times,
            "frames": try result.frames.images.map(png),
            "xy": pack(q.xy.flatMap { [$0.x, $0.y] }),
            "source": pack(q.source),
            "target": pack(q.target),
            "reference": pack(q.reference),
            "points": pack(p.points.flatMap { [$0.x, $0.y, $0.z] }),
            "confidence": pack(p.confidence),
            "uv": pack(p.uv.flatMap { [$0.x, $0.y] }),
            "visibility": pack(p.visibility),
            "normals": pack(p.normals.flatMap { [$0.x, $0.y, $0.z] }),
            "displacement": pack(p.displacement.flatMap { [$0.x, $0.y, $0.z] }),
            "groups": result.groups.map { group -> [String: Any] in
                var entry: [String: Any] = ["task": group.task.rawValue, "start": group.start, "count": group.count, "perFrame": group.perFrame]
                if let reference = group.reference { entry["reference"] = reference }
                return entry
            },
            "encodeSeconds": result.encodeSeconds,
            "decodeSeconds": result.decodeSeconds,
        ]
        if let cameras = result.cameras {
            archive["cameras"] = cameras.map { camera -> [String: Any] in
                let m = camera.cameraToReference
                var entry: [String: Any] = [
                    "pose": pack([m.columns.0, m.columns.1, m.columns.2, m.columns.3].flatMap { [$0.x, $0.y, $0.z, $0.w] }),
                    "valid": camera.valid,
                ]
                if let k = camera.intrinsics { entry["intrinsics"] = pack([k.x, k.y, k.z, k.w]) }
                return entry
            }
        }
        archive["sourcePaths"] = sources.map(\.path)
        return try PropertyListSerialization.data(fromPropertyList: archive, format: .binary, options: 0)
    }

    public static func result(from data: Data) throws -> (result: ArrowResult, sources: [URL]) {
        guard let archive = try PropertyListSerialization.propertyList(from: data, format: nil) as? [String: Any] else {
            throw Failure.unreadable("not a property list")
        }
        guard (archive["version"] as? Int) == version,
              let width = archive["width"] as? Int, let height = archive["height"] as? Int,
              let times = archive["times"] as? [Double], let pngs = archive["frames"] as? [Data]
        else { throw Failure.unreadable("missing frames") }
        let views = archive["views"] as? Int ?? 1
        guard views >= 1, pngs.count % views == 0 else { throw Failure.unreadable("frames are not whole timesteps") }
        func floats(_ key: String) throws -> [Float] {
            guard let data = archive[key] as? Data else { throw Failure.unreadable("missing \(key)") }
            return unpack(data)
        }
        func ints(_ key: String) throws -> [Int32] {
            guard let data = archive[key] as? Data else { throw Failure.unreadable("missing \(key)") }
            return unpack(data)
        }
        func pairs(_ v: [Float]) -> [SIMD2<Float>] { stride(from: 0, to: v.count, by: 2).map { SIMD2(v[$0], v[$0 + 1]) } }
        func triples(_ v: [Float]) -> [SIMD3<Float>] { stride(from: 0, to: v.count, by: 3).map { SIMD3(v[$0], v[$0 + 1], v[$0 + 2]) } }

        let images = try pngs.map(image)
        let queries = Queries(
            xy: pairs(try floats("xy")), source: try ints("source"), target: try ints("target"), reference: try ints("reference"))
        var predictions = Predictions()
        predictions.points = triples(try floats("points"))
        predictions.confidence = try floats("confidence")
        predictions.uv = pairs(try floats("uv"))
        predictions.visibility = try floats("visibility")
        predictions.normals = triples(try floats("normals"))
        predictions.displacement = triples(try floats("displacement"))
        guard predictions.points.count == queries.count, predictions.confidence.count == queries.count else {
            throw Failure.unreadable("predictions and queries differ in length")
        }
        let groups = try (archive["groups"] as? [[String: Any]] ?? []).map { entry -> QueryGroup in
            guard let name = entry["task"] as? String, let task = ArrowTask(rawValue: name),
                  let start = entry["start"] as? Int, let count = entry["count"] as? Int, let perFrame = entry["perFrame"] as? Int,
                  start >= 0, start + count <= queries.count
            else { throw Failure.unreadable("a bad task group") }
            return QueryGroup(task: task, start: start, count: count, perFrame: perFrame, reference: entry["reference"] as? Int)
        }
        let cameras = (archive["cameras"] as? [[String: Any]])?.map { entry -> CameraEstimate in
            let m: [Float] = (entry["pose"] as? Data).map(unpack) ?? []
            let pose = m.count == 16
                ? simd_float4x4(SIMD4(m[0], m[1], m[2], m[3]), SIMD4(m[4], m[5], m[6], m[7]),
                                SIMD4(m[8], m[9], m[10], m[11]), SIMD4(m[12], m[13], m[14], m[15]))
                : matrix_identity_float4x4
            let k: [Float] = (entry["intrinsics"] as? Data).map(unpack) ?? []
            return CameraEstimate(cameraToReference: pose, valid: entry["valid"] as? Bool ?? false,
                                  intrinsics: k.count == 4 ? SIMD4(k[0], k[1], k[2], k[3]) : nil)
        }
        let result = ArrowResult(
            frames: FrameSet(images: images, times: times, width: width, height: height, views: views),
            queries: queries, predictions: predictions, groups: groups, cameras: cameras,
            encodeSeconds: archive["encodeSeconds"] as? Double ?? 0, decodeSeconds: archive["decodeSeconds"] as? Double ?? 0)
        return (result, (archive["sourcePaths"] as? [String] ?? []).map { URL(filePath: $0) })
    }

    static func pack<T>(_ values: [T]) -> Data { values.withUnsafeBytes { Data($0) } }

    static func unpack<T>(_ data: Data) -> [T] {
        data.withUnsafeBytes { Array($0.bindMemory(to: T.self)) }
    }

    static func png(_ image: CGImage) throws -> Data {
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil) else {
            throw Failure.unreadable("could not encode a frame")
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { throw Failure.unreadable("could not encode a frame") }
        return data as Data
    }

    static func image(_ data: Data) throws -> CGImage {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else { throw Failure.unreadable("a frame does not decode") }
        return image
    }
}

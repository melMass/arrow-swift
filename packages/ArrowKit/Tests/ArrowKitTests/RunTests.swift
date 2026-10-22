import CoreGraphics
import Foundation
import Synchronization
import Testing
@testable import ArrowKit

/// A whole run on a real clip, timed: `ARROW_CLIP=/path/to/video.mp4` (upstream's
/// assets/videos/railway.mp4) with `ARROW_MODEL_TESTS=1`.
@Suite("End to end", .serialized, .enabled(if: Weights.enabled && ProcessInfo.processInfo.environment["ARROW_CLIP"] != nil))
struct RunTests {
    @Test("Tracks, a point cloud and cameras from a video")
    func clip() async throws {
        let clip = URL(filePath: ProcessInfo.processInfo.environment["ARROW_CLIP"]!)
        let frameCount = Int(ProcessInfo.processInfo.environment["ARROW_FRAMES"] ?? "") ?? 8
        let side = Int(ProcessInfo.processInfo.environment["ARROW_SIDE"] ?? "") ?? 504
        let started = Date()
        let frames = try await FrameSet.load(.video(clip), count: frameCount, maxSide: side)
        print("frames", frames.images.count, frames.width, "x", frames.height, "in", Date().timeIntervalSince(started))
        #expect(frames.images.count == frameCount)
        #expect(frames.width % 14 == 0 && frames.height % 14 == 0)

        let runner = ArrowRunner(model: try ArrowModel.load(from: Weights.directory!))
        var settings = TaskSettings()
        settings.trackStride = 16
        settings.reconstructionStride = 8
        settings.poseGrid = 32
        // The streamed chunks, assembled as a viewer would, end up as the finished predictions.
        let streamed = Streamed()
        let result = try await runner.run(frames, tasks: [.track, .reconstruct, .pose], settings: settings) { streamed.take($0) }
        print("queries", result.queries.count, "encode", result.encodeSeconds, "decode", result.decodeSeconds)
        let assembled = try #require(streamed.result)
        print("streamed chunks", streamed.chunks)
        #expect(streamed.chunks > 1)
        #expect(assembled.predictions.points == result.predictions.points)
        #expect(assembled.predictions.visibility == result.predictions.visibility)
        for group in result.groups { print(group.task, group.count, "per frame", group.perFrame) }

        let cameras = try #require(result.cameras)
        print("valid cameras", cameras.filter(\.valid).count, "of", cameras.count)
        for (i, c) in cameras.enumerated() { print(i, c.valid, c.cameraToReference.columns.3, c.intrinsics as Any) }
        #expect(cameras.filter(\.valid).count == frameCount)
        // In the tracks' frame: frame 0 is the identity.
        let first = cameras[0].cameraToReference
        #expect(abs(first.columns.3.x) < 1e-5 && abs(first.columns.0.x - 1) < 1e-5)

        let track = try #require(result.group(.track))
        let visible = (track.start ..< track.start + track.count).filter { result.predictions.visibility[$0] > 0 }.count
        print("track visible", visible, "of", track.count)
        #expect(visible > track.count / 2)
        // A seed asked in its own frame lands back on its own pixel.
        var error: Float = 0
        for k in 0 ..< track.perFrame {
            let i = track.start + k
            let pixel = result.predictions.uv[i] * SIMD2(Float(frames.width), Float(frames.height)) - 0.5
            error = max(error, simd_distance(pixel, result.queries.xy[i]))
        }
        print("source-frame reprojection, max px", error)
        #expect(error < 3)

        // The outputs, on real predictions.
        let keep = result.recommendedKeep
        print("recommended keep", keep)
        #expect(keep > 0.5 && keep <= 1)
        let folder = FileManager.default.temporaryDirectory.appending(path: "arrow-run-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        // `ARROW_USD_OUT` keeps the stage, to look at.
        let usd = ProcessInfo.processInfo.environment["ARROW_USD_OUT"].map { URL(filePath: $0) }
            ?? folder.appending(path: "railway.usda")
        try USDExport.write(result, to: usd, options: .init(keep: keep))
        let checker = Process()
        checker.executableURL = URL(filePath: "/usr/bin/usdchecker")
        checker.arguments = [usd.path]
        try checker.run()
        checker.waitUntilExit()
        let size = try FileManager.default.attributesOfItem(atPath: usd.path)[.size] as? Int ?? 0
        print("usda", size / 1024, "KB, usdchecker", checker.terminationStatus)
        #expect(checker.terminationStatus == 0)

        let archive = folder.appending(path: "railway.quiver")
        try ResultArchive.write(result, sources: [clip], to: archive)
        let (back, sources) = try ResultArchive.read(archive)
        #expect(sources == [clip])
        #expect(back.predictions.points == result.predictions.points)
        #expect(back.cameras?.map(\.cameraToReference) == cameras.map(\.cameraToReference))
    }

    /// Two overlapping crops of the clip stand in for two synchronized cameras: a seed
    /// placed in one should land on the same 3D point when asked through the other.
    @Test("Two synchronized cameras: tracks seeded in both, cameras fitted for each")
    func twoCameras() async throws {
        let clip = URL(filePath: ProcessInfo.processInfo.environment["ARROW_CLIP"]!)
        let source = try await FrameSet.load(.video(clip), count: 4, maxSide: 336)
        func crop(_ image: CGImage, from x: Int) -> CGImage {
            image.cropping(to: CGRect(x: x, y: 0, width: image.width * 7 / 10, height: image.height))!
        }
        var images: [CGImage] = [], times: [Double] = []
        for (image, time) in zip(source.images, source.times) {
            images += [crop(image, from: 0), crop(image, from: image.width * 3 / 10)]
            times += [time, time]
        }
        let (width, height) = FrameSet.inputSize(width: images[0].width, height: images[0].height, maxSide: 252)
        let frames = FrameSet(
            images: images.map { FrameSet.place($0, size: (width, height), canvas: (width, height)) },
            times: times, width: width, height: height, views: 2)

        let runner = ArrowRunner(model: try ArrowModel.load(from: Weights.directory!))
        var settings = TaskSettings()
        settings.trackStride = 28
        settings.poseGrid = 24
        let result = try await runner.run(frames, tasks: [.track, .pose], settings: settings)
        let group = try #require(result.group(.track))
        let grid = QueryGrid.spread(width: width, height: height, stride: 28).count
        #expect(group.perFrame == 2 * grid)
        #expect(Set(result.queries.source[group.start ..< group.start + group.perFrame]) == [0, 1])
        let cameras = try #require(result.cameras)
        print("valid cameras", cameras.filter(\.valid).count, "of", cameras.count)
        #expect(cameras.filter(\.valid).count == 8)

        // Where both cameras see a seed, their answers agree: the median against each.
        let median = TrackSet(result, group, fusion: .median)
        let seeded = TrackSet(result, group, fusion: .source)
        var gaps: [Float] = []
        for k in 0 ..< median.points.count where median.visibility[k] > 0 && TrackSet.finite(seeded.points[k]) {
            gaps.append(simd_distance(median.points[k], seeded.points[k]))
        }
        gaps.sort()
        let gap = gaps[gaps.count / 2]
        print("median vs source camera, median gap", gap, "scene", result.sceneDiagonal)
        #expect(gap < result.sceneDiagonal * 0.05)

        let usd = FileManager.default.temporaryDirectory.appending(path: "arrow-views-\(UUID().uuidString).usda")
        defer { try? FileManager.default.removeItem(at: usd) }
        try USDExport.write(result, to: usd)
        let stage = try String(contentsOf: usd, encoding: .utf8)
        #expect(stage.contains("def Camera \"Camera_1\"") && stage.contains("endTimeCode = 3"))
    }

    @Test("All-pixel tracking: the source frame's whole lattice, then what later frames add")
    func allPixels() async throws {
        let clip = URL(filePath: ProcessInfo.processInfo.environment["ARROW_CLIP"]!)
        let frames = try await FrameSet.load(.video(clip), count: 4, maxSide: 252)
        let runner = ArrowRunner(model: try ArrowModel.load(from: Weights.directory!))
        var settings = TaskSettings()
        settings.trackAllPixels = true
        settings.allPixelsStride = 6
        settings.trackSource = 1
        settings.reconstructionStride = 14
        let streamed = Streamed()
        let result = try await runner.run(frames, tasks: [.track, .reconstruct], settings: settings, chunk: 2048) { streamed.take($0) }
        print("all pixels: decode", result.decodeSeconds)
        let group = try #require(result.group(.track))
        let reconstruct = try #require(result.group(.reconstruct))
        #expect(group.start == reconstruct.count && group.count == group.perFrame * 4)

        // Seeded in order: frame 1's whole lattice, then 2, 3 and 0 add what is still uncovered.
        let sources = result.queries.source[group.start ..< group.start + group.perFrame].map(Int.init)
        let rank = PixelCoverage.order(frames: 4, source: 1)
        #expect(zip(sources, sources.dropFirst()).allSatisfy { rank.firstIndex(of: $0)! <= rank.firstIndex(of: $1)! })
        let lattice = PixelCoverage(width: frames.width, height: frames.height, frames: 1, stride: 6).latticeCount
        let perFrame = (0 ..< 4).map { f in sources.filter { $0 == f }.count }
        print("seeds per frame", perFrame, "lattice", lattice)
        #expect(perFrame[1] == lattice)
        #expect(perFrame[0] < lattice / 2 && perFrame[2] < lattice / 2)

        let assembled = try #require(streamed.result)
        #expect(assembled.queries.xy == result.queries.xy)
        #expect(assembled.predictions.points == result.predictions.points)
        #expect(assembled.predictions.visibility == result.predictions.visibility)
    }
}

/// A run's `started` (or latest `grew`) result with each chunk of answers written into it.
final class Streamed: Sendable {
    private let state = Mutex<(result: ArrowResult?, chunks: Int)>((nil, 0))

    var result: ArrowResult? { state.withLock { $0.result } }
    var chunks: Int { state.withLock { $0.chunks } }

    func take(_ event: ArrowEvent) {
        state.withLock { state in
            switch event {
            case let .started(skeleton): state.result = skeleton
            case let .grew(result): state.result = result
            case let .answered(start, answers):
                state.result?.predictions.fill(answers, from: start)
                state.chunks += 1
            case let .placed(positions, answers):
                state.result?.predictions.scatter(answers, at: positions)
                state.chunks += 1
            default: break
            }
        }
    }
}

import simd

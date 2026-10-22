import CoreGraphics
import Foundation
import Testing
@testable import ArrowKit

@Suite("Frame selection")
struct FrameSelectionTests {
    @Test("A range reads every step-th frame from start, before end, up to the limit")
    func range() {
        #expect(FrameSelection.range(start: 0, end: nil, step: 1, limit: 300).indices(of: 10) == Array(0 ..< 10))
        #expect(FrameSelection.range(start: 2, end: 9, step: 3, limit: 300).indices(of: 10) == [2, 5, 8])
        #expect(FrameSelection.range(start: 0, end: nil, step: 1, limit: 4).indices(of: 10) == [0, 1, 2, 3])
        #expect(FrameSelection.range(start: 0, end: 50, step: 6, limit: 300).indices(of: 24) == [0, 6, 12, 18])
    }

    @Test("A spread keeps the first and last frame")
    func spread() {
        #expect(FrameSelection.spread(3).indices(of: 11) == [0, 5, 10])
        #expect(FrameSelection.spread(20).indices(of: 4) == [0, 1, 2, 3])
    }

    @Test("Images read at the selected indices, cameras interleaved by timestep")
    func interleavedViews() async throws {
        let folder = FileManager.default.temporaryDirectory.appending(path: "arrow-views-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        // Camera a's frames are wider than b's, so a frame's width tells its camera.
        func frames(_ name: String, width: Int) throws -> [URL] {
            try (0 ..< 6).map { i in
                let url = folder.appending(path: "\(name)\(i).png")
                try ResultArchive.png(solid(width: width, height: 40, gray: CGFloat(i) / 10))
                    .write(to: url)
                return url
            }
        }
        let a = try frames("a", width: 80), b = try frames("b", width: 60)
        let set = try await FrameSet.load(
            views: [.images(a), .images(b)], selection: .range(start: 1, end: nil, step: 2, limit: 300), maxSide: 84)
        #expect(set.views == 2 && set.timesteps == 3)
        #expect(set.times == [1, 1, 3, 3, 5, 5])
        #expect(set.index(timestep: 2, view: 1) == 5)
    }

    func solid(width: Int, height: Int, gray: CGFloat) -> CGImage {
        let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                                space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        context.setFillColor(CGColor(gray: gray, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        return context.makeImage()!
    }
}

@Suite("Multi-view tracks")
struct MultiViewTrackTests {
    @Test("Without a drawn region, every camera's frame at the seed timestep gets the grid; picks stay on theirs")
    func seedsEveryCamera() {
        var settings = TaskSettings()
        settings.trackStride = 14
        settings.trackSource = 3 // timestep 1, camera 1 of 2
        settings.trackPicks = [SIMD2(3.5, 4.5)]
        let grid = QueryGrid.spread(width: 28, height: 28, stride: 14)
        let seeds = settings.trackSeeds(width: 28, height: 28, views: 2)
        #expect(seeds.sources == Array(repeating: 2, count: grid.count) + Array(repeating: 3, count: grid.count + 1))
        #expect(seeds.xy.last == SIMD2(3.5, 4.5))
    }

    @Test("A drawn region seeds only the camera it was drawn on")
    func regionSeedsOneCamera() {
        var settings = TaskSettings()
        settings.trackStride = 7
        settings.trackSource = 1
        settings.trackRegion = SIMD4(0, 0, 0.5, 0.5)
        let seeds = settings.trackSeeds(width: 28, height: 28, views: 2)
        #expect(Set(seeds.sources) == [1])
    }

    /// Two cameras, two timesteps, one seed placed in camera 1: frame t·2+v, so the
    /// decoded rows are (t0 v0), (t0 v1), (t1 v0), (t1 v1).
    func result(points: [SIMD3<Float>], visibility: [Float]) -> ArrowResult {
        let image = FrameSelectionTests().solid(width: 14, height: 14, gray: 0.5)
        let frames = FrameSet(images: Array(repeating: image, count: 4), times: [0, 0, 1, 1], width: 14, height: 14, views: 2)
        let queries = QueryBuilder.track([SIMD2(3, 3)], sources: [1], targets: [0, 1, 2, 3], reference: 0)
        var predictions = Predictions()
        predictions.points = points
        predictions.visibility = visibility
        predictions.confidence = [2, 4, 6, 8]
        predictions.uv = [SIMD2(0.1, 0.1), SIMD2(0.2, 0.2), SIMD2(0.3, 0.3), SIMD2(0.4, 0.4)]
        predictions.normals = Array(repeating: .zero, count: 4)
        predictions.displacement = Array(repeating: .zero, count: 4)
        return ArrowResult(frames: frames, queries: queries, predictions: predictions,
                           groups: [QueryGroup(task: .track, start: 0, count: 4, perFrame: 1, reference: 0)])
    }

    @Test("Camera median uses the cameras that see the point; source camera uses the seed's")
    func fusion() throws {
        let r = result(
            points: [SIMD3(0, 0, 1), SIMD3(0, 0, 3), SIMD3(1, 0, 1), SIMD3(5, 0, 1)],
            visibility: [2, 3, -4, 1])
        let group = try #require(r.group(.track))
        let median = TrackSet(r, group, fusion: .median, camera: 0)
        #expect(median.timesteps == 2 && median.seeds == 1)
        #expect(median.sourceView == [1])
        #expect(median.points[0] == SIMD3(0, 0, 2))
        // At t1 only camera 1 sees it.
        #expect(median.points[1] == SIMD3(5, 0, 1))
        #expect(median.visibility == [3, 1])
        #expect(median.confidence == [3, 7])
        #expect(median.uv == [SIMD2(0.1, 0.1), SIMD2(0.3, 0.3)])

        let source = TrackSet(r, group, fusion: .source, camera: 1)
        #expect(source.points == [SIMD3(0, 0, 3), SIMD3(5, 0, 1)])
        #expect(source.uv == [SIMD2(0.2, 0.2), SIMD2(0.4, 0.4)])
    }

    @Test("A single camera's tracks are the group as decoded")
    func singleView() throws {
        let image = FrameSelectionTests().solid(width: 14, height: 14, gray: 0.5)
        let frames = FrameSet(images: [image, image], times: [0, 1], width: 14, height: 14)
        var predictions = Predictions()
        predictions.points = [SIMD3(1, 2, 3), SIMD3(4, 5, 6)]
        predictions.visibility = [1, -1]
        predictions.confidence = [2, 3]
        predictions.uv = [.zero, .zero]
        let r = ArrowResult(frames: frames, queries: QueryBuilder.track([SIMD2(1, 1)], source: 0, targets: [0, 1], reference: 0),
                            predictions: predictions, groups: [QueryGroup(task: .track, start: 0, count: 2, perFrame: 1, reference: 0)])
        let tracks = TrackSet(r, try #require(r.group(.track)))
        #expect(tracks.points == predictions.points && tracks.visibility == predictions.visibility)
    }
}

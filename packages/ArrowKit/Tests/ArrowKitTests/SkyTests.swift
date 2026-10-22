import CoreML
import Foundation
import Testing
@testable import ArrowKit

@Suite("Sky")
struct SkyTests {
    @Test("Bilinear resizing matches PyTorch's interpolate (align_corners false)")
    func bilinear() {
        // torch.nn.functional.interpolate(torch.arange(12.).view(1, 1, 3, 4), (2, 6), mode="bilinear")
        let plane = (0 ..< 12).map(Float.init)
        let out = SkySegmenter.bilinear(plane, from: (4, 3), to: (6, 2))
        let expected: [Float] = [1, 1.5, 2.1666667, 2.8333333, 3.5, 4, 7, 7.5, 8.166667, 8.833333, 9.5, 10]
        #expect(zip(out, expected).allSatisfy { abs($0 - $1) < 1e-5 }, "\(out)")
    }

    @Test("A query is sky by its rounded, clamped source pixel")
    func maskLookup() {
        var sky = [Bool](repeating: false, count: 2 * 4 * 3)
        sky[(1 * 3 + 2) * 4 + 3] = true // frame 1, (3, 2)
        let mask = SkyMask(width: 4, height: 3, sky: sky)
        #expect(mask.contains(SIMD2(3.4, 1.6), frame: 1))
        #expect(mask.contains(SIMD2(9, 9), frame: 1))
        #expect(!mask.contains(SIMD2(3, 2), frame: 0))
        let queries = Queries(xy: [SIMD2(3, 2), SIMD2(0, 0)], source: [1, 1], target: [0, 0], reference: [0, 0])
        #expect(mask.skipped(queries) == [true, false])
    }

    /// Upstream's `segment_sky` on real outdoor frames against this port: `ARROW_SKY_DIR`
    /// holds `sky<i>.png` (at model size) and `sky<i>.mask` (upstream's mask, one byte
    /// per pixel), with `ARROW_MODEL_TESTS=1` for the download.
    /// A run on an outdoor clip (`ARROW_SKY_CLIP`) with the sky skipped: exactly the
    /// queries on sky pixels come back NaN.
    @Test("A run skips its sky queries", .enabled(if: Weights.enabled && ProcessInfo.processInfo.environment["ARROW_SKY_CLIP"] != nil))
    func runSkipsSky() async throws {
        let clip = URL(filePath: ProcessInfo.processInfo.environment["ARROW_SKY_CLIP"]!)
        let frames = try await FrameSet.load(.video(clip), count: 4, maxSide: 336)
        let mask = try await SkySegmenter.load().mask(frames)
        let runner = ArrowRunner(model: try ArrowModel.load(from: Weights.directory!))
        var settings = TaskSettings()
        settings.reconstructionStride = 8
        let result = try await runner.run(frames, tasks: [.reconstruct], settings: settings, sky: mask)
        let skipped = mask.skipped(result.queries)
        let unanswered = result.predictions.points.map { !TrackSet.finite($0) }
        print("sky", mask.fraction, "queries skipped", Double(skipped.filter { $0 }.count) / Double(skipped.count))
        #expect(skipped.contains(true) && skipped.contains(false))
        #expect(skipped == unanswered)
    }

    @Test("Masks match upstream's segment_sky", .enabled(if: ProcessInfo.processInfo.environment["ARROW_SKY_DIR"] != nil))
    func upstreamMasks() async throws {
        let folder = URL(filePath: ProcessInfo.processInfo.environment["ARROW_SKY_DIR"]!)
        let segmenter = try await SkySegmenter.load()
        var i = 0
        while FileManager.default.fileExists(atPath: folder.appending(path: "sky\(i).png").path) {
            let frames = try await FrameSet.load(.images([folder.appending(path: "sky\(i).png")]), count: 1)
            let mask = try segmenter.mask(frames)
            let reference = try Data(contentsOf: folder.appending(path: "sky\(i).mask")).map { $0 != 0 }
            #expect(reference.count == mask.sky.count)
            let agree = Double(zip(mask.sky, reference).filter { $0 == $1 }.count) / Double(reference.count)
            print("frame", i, "sky", mask.fraction, "agreement", agree)
            #expect(agree > 0.995)
            i += 1
        }
        #expect(i > 0)
    }
}

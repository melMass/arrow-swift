@testable import ArrowKit
import Foundation
import simd
import Testing
@testable import QuiverKit

@Suite("Saved results")
struct ArchiveTests {
    @Test("A result survives a save and reopen: frames, queries, predictions, groups, cameras")
    func roundTrip() throws {
        let result = PreviewFixtures.result()
        let data = try ResultArchive.data(result, sources: [URL(filePath: "/Movies/railway.mp4")])
        let (back, sources) = try ResultArchive.result(from: data)
        #expect(sources.map(\.path) == ["/Movies/railway.mp4"])
        #expect(back.frames.views == 1)
        #expect(back.frames.images.count == result.frames.images.count)
        #expect(back.frames.width == result.frames.width && back.frames.height == result.frames.height)
        #expect(back.queries.xy == result.queries.xy)
        #expect(back.queries.source == result.queries.source)
        #expect(back.predictions.points == result.predictions.points)
        #expect(back.predictions.confidence == result.predictions.confidence)
        #expect(back.predictions.uv == result.predictions.uv)
        #expect(back.groups.map(\.task) == result.groups.map(\.task))
        #expect(back.groups.map(\.perFrame) == result.groups.map(\.perFrame))
        #expect(back.cameras?.map(\.cameraToReference) == result.cameras?.map(\.cameraToReference))
        #expect(back.cameras?.map(\.intrinsics) == result.cameras?.map(\.intrinsics))
        // Lossless frames: the same RGB (the skipped fourth byte is not kept).
        let a = result.frames.images[3].dataProvider!.data! as Data, b = back.frames.images[3].dataProvider!.data! as Data
        #expect(a.count == b.count)
        let rgb = (0 ..< a.count / 4).allSatisfy { i in (0 ..< 3).allSatisfy { a[i * 4 + $0] == b[i * 4 + $0] } }
        #expect(rgb)
    }

    @Test("Something else is refused with a reason")
    func refusesOtherData() {
        #expect(throws: ResultArchive.Failure.self) { try ResultArchive.result(from: Data("hello".utf8)) }
    }
}

@Suite("Confidence")
struct ConfidenceTests {
    @Test("Recommended keep drops a low tail and keeps a clean distribution whole")
    func recommended() {
        // A tight cluster plus 5% far below: the tail goes.
        let cluster = (0 ..< 950).map { Float(20 + ($0 % 10)) }
        let tail = (0 ..< 50).map { _ in Float(1.01) }
        let keep = Confidence.recommendedKeep(cluster + tail)
        #expect(abs(keep - 0.95) < 0.011)
        #expect(Confidence.recommendedKeep([5, 5, 5]) == 1)
        #expect(Confidence.recommendedKeep([]) == 1)
    }

    @Test("Still tracks stray less than moving ones")
    func motion() throws {
        let result = PreviewFixtures.result()
        let group = try #require(result.group(.track))
        let statistics = TrackStatistics(result, group)
        #expect(statistics.radius.count == group.perFrame)
        // The fixture's points are fixed in the scene: every track is still.
        #expect(statistics.radius.allSatisfy { $0 < 1e-4 })
        #expect(result.sceneDiagonal > 1)
    }
}

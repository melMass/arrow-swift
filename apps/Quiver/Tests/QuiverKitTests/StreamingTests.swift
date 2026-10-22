@testable import ArrowKit
import Foundation
import Testing
@testable import QuiverKit

@MainActor
@Suite("Streamed runs")
struct StreamingTests {
    /// The fixture result replayed as a run sends it: its queries, then chunks of answers.
    @discardableResult
    func replay(into session: Session, chunk: Int, from first: Int = 0, until stop: Int? = nil) -> ArrowResult {
        let result = PreviewFixtures.result()
        if first == 0 {
            var skeleton = result
            skeleton.predictions = .unanswered(result.queries.count)
            session.show(.started(skeleton))
        }
        var start = first
        while start < min(stop ?? result.queries.count, result.queries.count) {
            let end = min(start + chunk, result.queries.count)
            var answers = Predictions()
            answers.points = Array(result.predictions.points[start ..< end])
            answers.confidence = Array(result.predictions.confidence[start ..< end])
            answers.uv = Array(result.predictions.uv[start ..< end])
            answers.visibility = Array(result.predictions.visibility[start ..< end])
            answers.normals = Array(result.predictions.normals[start ..< end])
            answers.displacement = Array(result.predictions.displacement[start ..< end])
            session.show(.answered(start: start, answers))
            start = end
        }
        return result
    }

    @Test("The scene fills in chunk by chunk, framed once, and the finished result keeps that framing")
    func fillsIn() throws {
        let session = Session.fixture(frame: 0, result: false)
        let before = session.sceneID
        let full = PreviewFixtures.result()
        let cloud = try #require(full.group(.reconstruct))
        let first = cloud.perFrame * 2
        replay(into: session, chunk: cloud.perFrame, until: first)
        #expect(session.streaming)
        let partial = session.geometry.points.count
        #expect(partial > 0)
        #expect(session.sceneID == before + 1)

        let result = replay(into: session, chunk: 500, from: first)
        #expect(session.geometry.points.count > partial)
        session.show(result)
        #expect(!session.streaming)
        #expect(session.sceneID == before + 1)
        #expect(session.result?.predictions.points == result.predictions.points)
    }

    @Test("While tracks arrive, the playhead follows the last timestep they reach")
    func followsTracks() throws {
        let session = Session.fixture(frame: 0, result: false)
        let full = PreviewFixtures.result()
        let tracks = try #require(full.group(.track))
        replay(into: session, chunk: tracks.perFrame, until: tracks.start + tracks.perFrame * 4)
        #expect(session.frame == 3)
        // Answered tracks are drawn at that timestep; later timesteps are still empty.
        #expect(session.tracks.map { TrackSet.finite($0.points[$0.index(3, 0)]) } == true)
        #expect(session.tracks.map { TrackSet.finite($0.points[$0.index(4, 0)]) } == false)
    }
}

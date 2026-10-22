@testable import ArrowKit
import Foundation
import Testing
@testable import QuiverKit

@MainActor
@Suite("Seeds keep their frame")
struct SeedFrameTests {
    @Test("Tracks start on the frame the picks were clicked on, wherever the scrubber is")
    func runsFromSeedFrame() throws {
        let session = Session.fixture(frame: 1, result: false)
        session.editSeeds { $0.picks.append(SIMD2(0.5, 0.5)) }
        session.frame = 6
        let frames = try #require(session.frames)
        #expect(session.seedFrame == 1)
        #expect(session.runSettings(for: frames).trackSource == 1)
    }

    @Test("Seeding another frame starts the seeds over on it")
    func editOnOtherFrameRestarts() {
        let session = Session.fixture(frame: 1, result: false)
        session.editSeeds {
            $0.picks.append(SIMD2(0.2, 0.2))
            $0.region = SIMD4(0.1, 0.1, 0.6, 0.6)
        }
        session.frame = 4
        session.editSeeds { $0.picks.append(SIMD2(0.8, 0.8)) }
        #expect(session.seedFrame == 4)
        #expect(session.picks == [SIMD2(0.8, 0.8)])
        #expect(session.region == Session.fullFrame)
    }

    @Test("With two cameras, seeds belong to the camera they were clicked on; the grid goes to both")
    func twoCameras() throws {
        let session = Session.fixture(frame: 0, result: false)
        let single = try #require(session.frames)
        // The fixture's frames read as 4 timesteps of 2 cameras.
        session.frames = FrameSet(images: Array(single.images.prefix(8)), times: Array(single.times.prefix(8)),
                                  width: single.width, height: single.height, views: 2)
        session.frame = 2
        session.camera = 1
        #expect(session.displayFrame == 5)
        session.editSeeds { $0.picks.append(SIMD2(0.5, 0.5)) }
        #expect(session.seedFrame == 5)
        #expect(session.frameName(5) == "frame 3, camera 2")
        session.frame = 0
        session.camera = 0
        session.showTrackSource()
        #expect(session.frame == 2 && session.camera == 1)
        let frames = try #require(session.frames)
        let settings = session.runSettings(for: frames)
        let seeds = settings.trackSeeds(width: frames.width, height: frames.height, views: 2)
        #expect(Set(seeds.sources) == [4, 5])
        #expect(session.seedTotal == seeds.xy.count)
    }

    @Test("With no picks and the full frame, tracks start on the frame shown")
    func clearedSeedsFollowScrubber() throws {
        let session = Session.fixture(frame: 2, result: false)
        session.editSeeds { $0.picks.append(SIMD2(0.5, 0.5)) }
        session.picks = []
        session.frame = 5
        let frames = try #require(session.frames)
        #expect(session.seedFrame == nil)
        #expect(session.runSettings(for: frames).trackSource == 5)
    }
}

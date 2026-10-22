import ArrowKit
import Testing
@testable import QuiverKit

@Suite("Scene building")
struct SceneBuilderTests {
    let result = PreviewFixtures.result()

    @Test("The confidence filter keeps the asked fraction, all of it at 1")
    func keep() {
        #expect(Confidence.threshold([1, 2, 3, 4], keep: 1) == -.infinity)
        #expect(Confidence.threshold([1, 2, 3, 4], keep: 0.5) == 3)
        #expect([1, 2, 3, 4].filter { $0 >= Confidence.threshold([1, 2, 3, 4], keep: 0.25) }.count == 1)
    }

    @Test("The cloud has a point per kept reconstruction query, flipped into SceneKit's frame")
    func cloud() throws {
        let group = try #require(result.group(.reconstruct))
        let colors = result.colors(group)
        var settings = DisplaySettings()
        settings.keep = 1
        let all = SceneBuilder.cloud(result, colors: colors, settings: settings)
        #expect(all.points.count == group.count)
        // Reference camera z forward is SceneKit −z.
        #expect(all.points.allSatisfy { $0.z < 0 })
        settings.keep = 0.3
        #expect(SceneBuilder.cloud(result, colors: colors, settings: settings).points.count < group.count)
    }

    @Test("Hidden track points can be shown, dimmed or dropped; trails reach back `trail` frames")
    func tracks() throws {
        let group = try #require(result.group(.track))
        let colors = result.colors(group)
        let tracks = TrackSet(result, group)
        let statistics = TrackStatistics(tracks)
        var settings = DisplaySettings()
        settings.keep = 1
        settings.trail = 3
        settings.hidden = .show
        let shown = SceneBuilder.tracks(result, tracks, timestep: 7, colors: colors, statistics: statistics, diagonal: 4, settings: settings)
        #expect(shown.points.count == group.perFrame)
        #expect(shown.lines.count == group.perFrame * 3 * 2)
        settings.hidden = .hide
        let visible = (0 ..< group.perFrame).filter { result.predictions.visibility[group.start + 7 * group.perFrame + $0] > 0 }.count
        #expect(SceneBuilder.tracks(result, tracks, timestep: 7, colors: colors, statistics: statistics, diagonal: 4, settings: settings).points.count == visible)
        #expect(visible < group.perFrame)
    }

    @Test("Still tracks collapse to one point each, without trails")
    func collapse() throws {
        let group = try #require(result.group(.track))
        var settings = DisplaySettings()
        settings.keep = 1
        settings.separateMotion = true
        let tracks = TrackSet(result, group)
        let geometry = SceneBuilder.tracks(result, tracks, timestep: 5, colors: result.colors(group), statistics: TrackStatistics(tracks),
                                           diagonal: result.sceneDiagonal, settings: settings)
        // The fixture's points never move.
        #expect(geometry.points.count == group.perFrame)
        #expect(geometry.lines.isEmpty)
    }

    @Test("Each valid camera is a frustum of eight segments, joined by a path, the current one with its frame")
    func cameras() {
        var settings = DisplaySettings()
        let geometry = SceneBuilder.cameras(result, frame: 2, depth: 0.2, settings: settings)
        let frames = result.frames.images.count
        #expect(geometry.lines.count == 8 * 2 * frames + 2 * (frames - 1))
        #expect(geometry.plane != nil)
        settings.cameraPath = false
        settings.imagePlane = 0
        let bare = SceneBuilder.cameras(result, frame: 2, depth: 0.2, settings: settings)
        #expect(bare.lines.count == 8 * 2 * frames)
        #expect(bare.plane == nil)
    }

    @Test("Estimated normals of a flat wall face the camera")
    func estimatedNormals() throws {
        let group = try #require(result.group(.reconstruct))
        let normals = SceneBuilder.estimatedNormals(result, group)
        // A wall pixel far from the box's edges, in frame 0.
        let xy = result.queries.xy
        let wall = try #require((group.start ..< group.start + group.perFrame).first { xy[$0].x > 20 && xy[$0].x < 60 && xy[$0].y > 20 && xy[$0].y < 60 })
        let n = normals[wall - group.start]
        #expect(n.z < -0.99)
    }

    @Test("The palettes run dark to bright")
    func palettes() {
        #expect(SceneBuilder.viridis(0).z > SceneBuilder.viridis(0).x)
        #expect(SceneBuilder.viridis(1).x > 0.9)
        #expect(SceneBuilder.turbo(0.5).y > 0.8)
    }
}

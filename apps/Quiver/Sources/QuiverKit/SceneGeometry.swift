import ArrowKit
import CoreGraphics
import Foundation
import simd

/// How points and tracks are drawn; everything here leaves the predictions untouched.
/// Defaults and colour rules follow upstream's viewer.
struct DisplaySettings: Equatable {
    enum PointColoring: String, CaseIterable, Identifiable {
        case rgb = "RGB", confidence = "Confidence", visibility = "Visibility"
        case normals = "Predicted normals", estimatedNormals = "Estimated normals"
        var id: Self { self }
    }

    enum TrailColoring: String, CaseIterable, Identifiable {
        case position = "Seed position", rgb = "RGB", time = "Time", confidence = "Log confidence"
        var id: Self { self }
    }

    /// What happens to a track point the model calls hidden in the current frame.
    enum Hidden: String, CaseIterable, Identifiable {
        case show = "Show", dim = "Dim", hide = "Hide"
        var id: Self { self }
    }

    var showCloud = true
    var showTracks = true
    var showCameras = true
    var showDepth = false
    var cloudColoring: PointColoring = .rgb
    var trackColoring: TrailColoring = .position
    /// The most confident fraction of points and tracks kept (tracks by mean confidence).
    var keep: Float = 1
    var pointSize: Float = 3
    /// Frames of trail behind each track point.
    var trail = 8
    var hidden: Hidden = .dim
    /// A track point counts as seen when sigmoid(visibility) reaches this.
    var visibleAt: Float = 0.5
    /// Split tracks into still and moving.
    var separateMotion = false
    /// A track is still when it strays at most this % of the scene's diagonal from its mean.
    var stillPercent: Float = 0.25
    /// Draw still tracks once, at their median position, without a trail.
    var collapseStill = true
    /// Still tracks blue, moving ones orange.
    var tintMotion = true
    /// How several cameras' predictions of a track become one position.
    var fusion: TrackSet.Fusion = .median
    var cameraPath = true
    /// The current frame drawn on its camera's frustum; 0 hides it.
    var imagePlane: Float = 0.85

    var seenLogit: Float { visibleAt <= 0 ? -.infinity : visibleAt >= 1 ? .infinity : log(visibleAt / (1 - visibleAt)) }
}

/// Points and line segments in SceneKit's frame (y up, looking down −z), built from a
/// result in the reference camera's (x right, y down, z forward).
struct Geometry {
    var points: [SIMD3<Float>] = []
    var colors: [SIMD4<Float>] = []
    /// Pairs of endpoints.
    var lines: [SIMD3<Float>] = []
    var lineColors: [SIMD4<Float>] = []
    /// A frame drawn on a camera's far rectangle: corners top-left, top-right,
    /// bottom-right, bottom-left.
    var plane: (corners: [SIMD3<Float>], image: CGImage, opacity: Float)?

    var isEmpty: Bool { points.isEmpty && lines.isEmpty }

    mutating func append(_ other: Geometry) {
        points += other.points; colors += other.colors
        lines += other.lines; lineColors += other.lineColors
        if let plane = other.plane { self.plane = plane }
    }

    static func scene(_ p: SIMD3<Float>) -> SIMD3<Float> { SIMD3(p.x, -p.y, -p.z) }
}

enum SceneBuilder {
    static let missing = SIMD3<Float>(repeating: 110 / 255)

    /// The reconstruction's point cloud: every frame's pixel grid in the reference camera,
    /// the least confident dropped.
    static func cloud(_ result: ArrowResult, colors: [SIMD3<Float>], settings: DisplaySettings) -> Geometry {
        guard let group = result.group(.reconstruct) else { return Geometry() }
        let range = group.start ..< group.start + group.count
        let tints = pointColors(result, group, rgb: colors, coloring: settings.cloudColoring, keep: settings.keep)
        let cut = Confidence.threshold(range.map { result.predictions.confidence[$0] }, keep: settings.keep)
        var geometry = Geometry()
        for (k, i) in range.enumerated() where result.predictions.confidence[i] >= cut {
            let p = result.predictions.points[i]
            guard p.x.isFinite, p.y.isFinite, p.z.isFinite else { continue }
            geometry.points.append(Geometry.scene(p))
            geometry.colors.append(SIMD4(tints[k], 1))
        }
        return geometry
    }

    /// One frame's depth: its own pixels in its own camera.
    static func depth(_ result: ArrowResult, frame: Int, colors: [SIMD3<Float>], settings: DisplaySettings) -> Geometry {
        guard let group = result.group(.depth), frame < result.frames.images.count else { return Geometry() }
        let tints = pointColors(result, group, rgb: colors, coloring: settings.cloudColoring, keep: settings.keep)
        let range = group.start + frame * group.perFrame ..< group.start + (frame + 1) * group.perFrame
        let cut = Confidence.threshold(range.map { result.predictions.confidence[$0] }, keep: settings.keep)
        var geometry = Geometry()
        for i in range where result.predictions.confidence[i] >= cut {
            geometry.points.append(Geometry.scene(result.predictions.points[i]))
            geometry.colors.append(SIMD4(tints[i - group.start], 1))
        }
        return geometry
    }

    /// Track points at `timestep` and their trails over the timesteps before it. A hidden
    /// point in "hide" mode also breaks its trail.
    static func tracks(
        _ result: ArrowResult, _ tracks: TrackSet, timestep frame: Int, colors: [SIMD3<Float>], statistics: TrackStatistics,
        diagonal: Float, settings: DisplaySettings
    ) -> Geometry {
        guard tracks.seeds > 0, frame < tracks.timesteps else { return Geometry() }
        let frames = tracks.timesteps
        let kept = statistics.kept(settings.keep)
        let still = stillTracks(statistics, diagonal: diagonal, settings: settings)
        let seen = settings.seenLogit
        var geometry = Geometry()
        for seed in 0 ..< tracks.seeds where kept[seed] {
            let base = trackColor(result, tracks, seed: seed, colors: colors, coloring: settings.trackColoring, timestep: frame)
            let color = settings.separateMotion && settings.tintMotion
                ? (still[seed] ? SIMD3<Float>(0.35, 0.6, 1) : SIMD3<Float>(1, 0.55, 0.2)) : base
            if settings.separateMotion && settings.collapseStill && still[seed] {
                let median = statistics.median[seed]
                guard median.x.isFinite else { continue }
                geometry.points.append(Geometry.scene(median))
                geometry.colors.append(SIMD4(color, 1))
                continue
            }
            let i = tracks.index(frame, seed)
            let isSeen = tracks.visibility[i] >= seen
            if (isSeen || settings.hidden != .hide), TrackSet.finite(tracks.points[i]) {
                geometry.points.append(Geometry.scene(tracks.points[i]))
                geometry.colors.append(SIMD4(isSeen || settings.hidden == .show ? color : color * 0.35, 1))
            }
            let first = max(0, frame - settings.trail)
            guard first < frame else { continue }
            for t in first ..< min(frame, frames - 1) {
                let a = tracks.index(t, seed), b = tracks.index(t + 1, seed)
                if settings.hidden == .hide, tracks.visibility[a] < seen || tracks.visibility[b] < seen { continue }
                guard TrackSet.finite(tracks.points[a]), TrackSet.finite(tracks.points[b]) else { continue }
                // Fades in over the oldest 30 % of the trail.
                let fade = min(1, Float(t - first + 1) / Float(frame - first) / 0.3)
                let segment = settings.trackColoring == .time || settings.trackColoring == .confidence
                    ? trackColor(result, tracks, seed: seed, colors: colors, coloring: settings.trackColoring, timestep: t + 1)
                    : color
                geometry.lines += [Geometry.scene(tracks.points[a]), Geometry.scene(tracks.points[b])]
                geometry.lineColors += Array(repeating: SIMD4(segment * fade, 1), count: 2)
            }
        }
        return geometry
    }

    /// Whether each track stays within `stillPercent` % of the scene's diagonal of its mean.
    static func stillTracks(_ statistics: TrackStatistics, diagonal: Float, settings: DisplaySettings) -> [Bool] {
        guard settings.separateMotion else { return statistics.radius.map { _ in false } }
        let limit = settings.stillPercent / 100 * diagonal
        return statistics.radius.map { $0 <= limit }
    }

    /// Each fitted camera as a frustum (apex at its center, a rectangle at `depth`), the
    /// current one brighter and the rest fading with their distance in time; the path
    /// through their centers (one path per camera of a multi-view set); the current frame
    /// on its frustum.
    static func cameras(_ result: ArrowResult, frame: Int, depth: Float, settings: DisplaySettings) -> Geometry {
        guard let cameras = result.cameras else { return Geometry() }
        let width = Float(result.frames.width), height = Float(result.frames.height)
        let views = result.frames.views
        var geometry = Geometry()
        var centers = [[SIMD3<Float>]](repeating: [], count: views)
        for (index, camera) in cameras.enumerated() where camera.valid {
            let k = camera.intrinsics ?? SIMD4(max(width, height), max(width, height), width / 2, height / 2)
            let corners = [(-0.5, -0.5), (width - 0.5, -0.5), (width - 0.5, height - 0.5), (-0.5, height - 0.5)]
                .map { (x: Float, y: Float) -> SIMD3<Float> in SIMD3((x - k.z) / k.x * depth, (y - k.w) / k.y * depth, depth) }
            func world(_ p: SIMD3<Float>) -> SIMD3<Float> {
                let w = camera.cameraToReference * SIMD4(p, 1)
                return Geometry.scene(SIMD3(w.x, w.y, w.z))
            }
            let apex = world(.zero)
            centers[index % views].append(apex)
            let rim = corners.map(world)
            let color = index == frame ? SIMD4(cameraColor(0), 1) : SIMD4(cameraColor(index + 1) * max(0.16, 0.42 * exp(-Float(abs(index - frame)) / 14)) * 2, 1)
            for c in 0 ..< 4 {
                geometry.lines += [apex, rim[c], rim[c], rim[(c + 1) % 4]]
                geometry.lineColors += Array(repeating: color, count: 4)
            }
            if index == frame, settings.imagePlane > 0, frame < result.frames.images.count {
                geometry.plane = (rim, result.frames.images[frame], settings.imagePlane)
            }
        }
        if settings.cameraPath {
            for path in centers where path.count > 1 {
                for i in 0 ..< path.count - 1 {
                    geometry.lines += [path[i], path[i + 1]]
                    geometry.lineColors += Array(repeating: SIMD4(cameraColor(0) * 0.55, 1), count: 2)
                }
            }
        }
        return geometry
    }

    /// Upstream's camera hues: the current camera at 218°, others stepped by the golden angle.
    static func cameraColor(_ index: Int) -> SIMD3<Float> {
        hsl((218 + Float(index) * 137.507764).truncatingRemainder(dividingBy: 360) / 360, 0.72, 0.62)
    }

    /// Colours for every query of a point group.
    static func pointColors(
        _ result: ArrowResult, _ group: QueryGroup, rgb: [SIMD3<Float>], coloring: DisplaySettings.PointColoring, keep: Float
    ) -> [SIMD3<Float>] {
        let range = group.start ..< group.start + group.count
        let p = result.predictions
        switch coloring {
        case .rgb:
            return rgb
        case .confidence:
            // Viridis by rank among the points kept.
            let order = range.indices.sorted { p.confidence[range.lowerBound + $0] < p.confidence[range.lowerBound + $1] }
            let drop = Int((Float(order.count) * (1 - keep)).rounded())
            var colors = [SIMD3<Float>](repeating: viridis(0), count: order.count)
            let span = Float(max(1, order.count - drop - 1))
            for (rank, k) in order.enumerated() { colors[k] = viridis(Float(rank - drop) / span) }
            return colors
        case .visibility:
            return range.map { viridis(1 / (1 + exp(-p.visibility[$0]))) }
        case .normals:
            return range.map { normalColor(p.normals[$0]) }
        case .estimatedNormals:
            return estimatedNormals(result, group).map(normalColor)
        }
    }

    /// x red, y up green, facing the camera blue (upstream's byte encoding); grey when unknown.
    static func normalColor(_ n: SIMD3<Float>) -> SIMD3<Float> {
        guard n.x.isFinite, n.y.isFinite, n.z.isFinite, simd_length(n) > 0 else { return missing }
        return simd_clamp(n * SIMD3(0.5, -0.5, -0.5) + 0.5, SIMD3(repeating: 0), SIMD3(repeating: 1))
    }

    /// Normals from each frame's grid of points: the cross product of the pointmap's
    /// central-difference gradients down and across (np.gradient; one-sided at edges),
    /// facing the camera. NaN where a neighbour is missing.
    static func estimatedNormals(_ result: ArrowResult, _ group: QueryGroup) -> [SIMD3<Float>] {
        let xy = result.queries.xy
        let firstRow = xy[group.start].y
        let columns = (group.start ..< group.start + group.perFrame).prefix(while: { xy[$0].y == firstRow }).count
        guard columns > 1, group.perFrame % columns == 0, group.perFrame / columns > 1 else {
            return Array(repeating: SIMD3(repeating: .nan), count: group.count)
        }
        let rows = group.perFrame / columns, frames = group.count / group.perFrame
        var normals = [SIMD3<Float>](repeating: SIMD3(repeating: .nan), count: group.count)
        let p = result.predictions.points
        for f in 0 ..< frames {
            let base = group.start + f * group.perFrame
            func at(_ r: Int, _ c: Int) -> SIMD3<Float> { p[base + r * columns + c] }
            for r in 0 ..< rows {
                for c in 0 ..< columns {
                    let down = r == 0 ? at(1, c) - at(0, c) : r == rows - 1 ? at(r, c) - at(r - 1, c) : (at(r + 1, c) - at(r - 1, c)) / 2
                    let across = c == 0 ? at(r, 1) - at(r, 0) : c == columns - 1 ? at(r, c) - at(r, c - 1) : (at(r, c + 1) - at(r, c - 1)) / 2
                    let n = simd_cross(down, across)
                    let length = simd_length(n)
                    if length.isFinite, length > 0 { normals[base - group.start + r * columns + c] = n / length }
                }
            }
        }
        return normals
    }

    static func trackColor(
        _ result: ArrowResult, _ tracks: TrackSet, seed: Int, colors: [SIMD3<Float>],
        coloring: DisplaySettings.TrailColoring, timestep: Int
    ) -> SIMD3<Float> {
        switch coloring {
        case .rgb: return colors[seed]
        case .time:
            return turbo(Float(timestep) / Float(max(1, tracks.timesteps - 1)))
        case .confidence:
            let c = tracks.confidence[tracks.index(max(timestep - 1, 0), seed)]
            return turbo(min(1, log1p(max(0, c - 1)) / 4))
        case .position:
            guard let group = result.group(.track) else { return missing }
            let xy = result.queries.xy[group.start + seed]
            return turbo((xy.x + 0.5) / Float(result.frames.width))
        }
    }

    /// Google's Turbo, the 5th-degree polynomial fit upstream uses.
    static func turbo(_ t: Float) -> SIMD3<Float> {
        polynomial(min(max(t, 0), 1), [
            [0.13572138, 4.6153926, -42.66032258, 132.13108234, -152.94239396, 59.28637943],
            [0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604],
            [0.1066733, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973],
        ])
    }

    /// Viridis, the 6th-degree fit upstream uses.
    static func viridis(_ t: Float) -> SIMD3<Float> {
        polynomial(min(max(t, 0), 1), [
            [0.2777273272, 0.1050930431, -0.3308618287, -4.634230499, 6.228269936, 4.776384998, -5.435455856],
            [0.005407344545, 1.404613530, 0.2148475595, -5.799100973, 14.17993337, -13.74514538, 4.645852612],
            [0.3340998053, 1.384590163, 0.09509516303, -19.33244096, 56.69055260, -65.35303263, 26.31243525],
        ])
    }

    static func polynomial(_ t: Float, _ coefficients: [[Float]]) -> SIMD3<Float> {
        let channels = coefficients.map { c in c.reversed().reduce(Float(0)) { $0 * t + $1 } }
        return simd_clamp(SIMD3(channels[0], channels[1], channels[2]), SIMD3(repeating: 0), SIMD3(repeating: 1))
    }

    static func hsl(_ h: Float, _ s: Float, _ l: Float) -> SIMD3<Float> {
        let a = s * min(l, 1 - l)
        func f(_ n: Float) -> Float {
            let k = (n + h * 12).truncatingRemainder(dividingBy: 12)
            return l - a * max(-1, min(k - 3, 9 - k, 1))
        }
        return SIMD3(f(0), f(8), f(4))
    }

    /// A point to orbit around: the median of the cloud (or the tracks).
    static func center(_ geometry: Geometry) -> SIMD3<Float> {
        guard !geometry.points.isEmpty else { return SIMD3(0, 0, -1) }
        func median(_ axis: Int) -> Float {
            let values = geometry.points.map { $0[axis] }.sorted()
            return values[values.count / 2]
        }
        return SIMD3(median(0), median(1), median(2))
    }
}

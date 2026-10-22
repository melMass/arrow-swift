import Foundation
import simd

/// A track group read back per timestep: seed s at timestep t is entry `t · seeds + s`.
///
/// With one camera this is the group as decoded. With several (`FrameSet.views`), each
/// seed was asked in every camera's frame at every timestep, and the cameras' answers are
/// folded onto the timestep the way upstream's viewer does (`_multiview_tracks`).
public struct TrackSet: Sendable {
    /// How several cameras' predictions of one track at one timestep become one.
    public enum Fusion: String, Sendable, CaseIterable, Identifiable {
        /// Per-axis median of the cameras that see the point (of every camera with an
        /// answer when none does); visible when any camera sees it.
        case median = "Camera median"
        /// The prediction made in the camera the track was seeded in.
        case source = "Source camera"
        public var id: Self { self }
    }

    public let timesteps: Int
    public let seeds: Int
    /// Positions in the reference camera; NaN where no camera gave one.
    public var points: [SIMD3<Float>]
    /// Visibility logits.
    public var visibility: [Float]
    /// Mean over the cameras with an answer.
    public var confidence: [Float]
    /// Where the track is in `camera`'s frame at each timestep, normalized like
    /// `Predictions.uv`.
    public var uv: [SIMD2<Float>]
    /// Visibility logits in `camera`'s frame alone.
    public var cameraVisibility: [Float]
    /// The camera each seed was placed in.
    public var sourceView: [Int]

    public init(_ result: ArrowResult, _ group: QueryGroup, fusion: Fusion = .median, camera: Int = 0) {
        let views = result.frames.views, seeds = group.perFrame
        let timesteps = seeds == 0 ? 0 : group.count / seeds / views
        self.timesteps = timesteps
        self.seeds = seeds
        let p = result.predictions
        let camera = min(max(camera, 0), views - 1)
        sourceView = (0 ..< seeds).map { Int(result.queries.source[group.start + $0]) % views }
        func raw(_ t: Int, _ view: Int, _ s: Int) -> Int { group.start + (t * views + view) * seeds + s }

        guard views > 1 else {
            let range = group.start ..< group.start + group.count
            points = Array(p.points[range]); visibility = Array(p.visibility[range])
            confidence = Array(p.confidence[range]); uv = Array(p.uv[range])
            cameraVisibility = visibility
            return
        }
        let count = timesteps * seeds
        points = Array(repeating: SIMD3(repeating: .nan), count: count)
        visibility = Array(repeating: -100, count: count)
        confidence = Array(repeating: 0, count: count)
        uv = Array(repeating: SIMD2(repeating: .nan), count: count)
        cameraVisibility = Array(repeating: -100, count: count)
        for t in 0 ..< timesteps {
            for s in 0 ..< seeds {
                let k = t * seeds + s
                uv[k] = p.uv[raw(t, camera, s)]
                cameraVisibility[k] = p.visibility[raw(t, camera, s)]
                let answers = (0 ..< views).map { raw(t, $0, s) }.filter { Self.finite(p.points[$0]) }
                let confident = answers.filter { p.confidence[$0].isFinite }
                if !confident.isEmpty { confidence[k] = confident.map { p.confidence[$0] }.reduce(0, +) / Float(confident.count) }
                switch fusion {
                case .source:
                    let i = raw(t, sourceView[s], s)
                    points[k] = p.points[i]
                    if p.visibility[i].isFinite { visibility[k] = p.visibility[i] }
                case .median:
                    let seen = answers.filter { p.visibility[$0] > 0 }
                    let used = seen.isEmpty ? answers : seen
                    guard !used.isEmpty else { continue }
                    points[k] = Self.median(used.map { p.points[$0] })
                    visibility[k] = answers.map { p.visibility[$0] }.filter(\.isFinite).max() ?? -100
                }
            }
        }
    }

    public func index(_ timestep: Int, _ seed: Int) -> Int { timestep * seeds + seed }

    public static func finite(_ p: SIMD3<Float>) -> Bool { p.x.isFinite && p.y.isFinite && p.z.isFinite }

    /// Per-axis median, the mean of the middle two for an even count.
    static func median(_ values: [SIMD3<Float>]) -> SIMD3<Float> {
        SIMD3((0 ..< 3).map { axis -> Float in
            let v = values.map { $0[axis] }.sorted(), n = v.count
            return n % 2 == 1 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2
        })
    }
}

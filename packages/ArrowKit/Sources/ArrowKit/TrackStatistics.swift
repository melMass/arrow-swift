import Foundation
import simd

/// Per-track figures over every timestep of a `TrackSet`.
public struct TrackStatistics: Sendable {
    /// Mean confidence over every timestep.
    public var meanConfidence: [Float]
    /// The farthest a track's point strays from its own mean position.
    public var radius: [Float]
    /// Per-axis median position, for drawing a still track as one point.
    public var median: [SIMD3<Float>]

    public init(_ result: ArrowResult, _ group: QueryGroup) {
        self.init(TrackSet(result, group))
    }

    public init(_ tracks: TrackSet) {
        let seeds = tracks.seeds, steps = tracks.timesteps
        meanConfidence = []; radius = []; median = []
        meanConfidence.reserveCapacity(seeds)
        for s in 0 ..< seeds {
            let samples = (0 ..< steps).map { tracks.points[tracks.index($0, s)] }.filter(TrackSet.finite)
            let confidence = (0 ..< steps).map { tracks.confidence[tracks.index($0, s)] }
            meanConfidence.append(confidence.reduce(0, +) / Float(max(steps, 1)))
            guard !samples.isEmpty else {
                radius.append(.infinity); median.append(SIMD3(repeating: .nan)); continue
            }
            let mean = samples.reduce(SIMD3<Float>(), +) / Float(samples.count)
            radius.append(samples.map { simd_distance($0, mean) }.max()!)
            median.append(TrackSet.median(samples))
        }
    }

    /// Which tracks survive keeping the `keep` most confident, by mean confidence.
    public func kept(_ keep: Float) -> [Bool] {
        let cut = Confidence.threshold(meanConfidence, keep: keep)
        return meanConfidence.map { $0 >= cut }
    }
}

extension ArrowResult {
    /// The span of the scene: the diagonal of the box between the 1st and 99th
    /// percentiles of every predicted point on each axis (upstream's viewer measure).
    public var sceneDiagonal: Float {
        let stride = max(1, predictions.points.count / 100_000)
        let points = Swift.stride(from: 0, to: predictions.points.count, by: stride).map { predictions.points[$0] }
            .filter { $0.x.isFinite && $0.y.isFinite && $0.z.isFinite }
        guard points.count > 1 else { return 1 }
        let extent = (0 ..< 3).map { axis -> Float in
            let v = points.map { $0[axis] }.sorted()
            return v[Int(Float(v.count - 1) * 0.99)] - v[Int(Float(v.count - 1) * 0.01)]
        }
        return max(0.02, (extent[0] * extent[0] + extent[1] * extent[1] + extent[2] * extent[2]).squareRoot())
    }

    /// The recommended keep fraction over the cloud's points and the tracks' means.
    public var recommendedKeep: Float {
        var values: [Float] = []
        if let cloud = group(.reconstruct) { values += predictions.confidence[cloud.start ..< cloud.start + cloud.count] }
        if let tracks = group(.track) { values += TrackStatistics(self, tracks).meanConfidence }
        return Confidence.recommendedKeep(values)
    }
}

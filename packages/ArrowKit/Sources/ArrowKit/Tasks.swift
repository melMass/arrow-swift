import Foundation
import simd

/// What a run asks the model. Each task is a block of queries; upstream's `make_queries`.
public enum ArrowTask: String, Sendable, CaseIterable, Codable {
    /// Every frame's pixel grid, in the reference frame's camera: one shared point cloud.
    case reconstruct
    /// Every frame's pixel grid in its own camera: per-frame depth.
    case depth
    /// Seeds in a source frame followed through every frame.
    case track
    /// A grid per frame asked in its own and the reference camera, then a rigid fit per frame.
    case pose
}

/// A task's block of queries within a run, and how to read it back.
public struct QueryGroup: Sendable {
    public var task: ArrowTask
    /// Position of the block's first query in the run.
    public var start: Int
    public var count: Int
    /// Queries per frame (per target frame for tracks; per frame and camera for poses).
    public var perFrame: Int
    public var reference: Int?

    public init(task: ArrowTask, start: Int, count: Int, perFrame: Int, reference: Int?) {
        self.task = task
        self.start = start
        self.count = count
        self.perFrame = perFrame
        self.reference = reference
    }
}

public enum QueryGrid {
    /// Integer (x, y) centers: every `stride` pixels from the top-left, row-major.
    public static func strided(width: Int, height: Int, stride step: Int) -> [SIMD2<Float>] {
        stride(from: 0, to: height, by: step).flatMap { y in
            stride(from: 0, to: width, by: step).map { SIMD2(Float($0), Float(y)) }
        }
    }

    /// `size` × `size` integer centers spread edge to edge (torch.linspace, rounded half to even).
    public static func sized(width: Int, height: Int, size: Int) -> [SIMD2<Float>] {
        func axis(_ length: Int) -> [Float] {
            guard size > 1 else { return [0] }
            return (0 ..< size).map { (Float($0) * Float(length - 1) / Float(size - 1)).rounded(.toNearestOrEven) }
        }
        let xs = axis(width)
        return axis(height).flatMap { y in xs.map { SIMD2($0, y) } }
    }

    /// Centers about `stride` px apart spanning `region` (fractions of the image, x0 y0 x1
    /// y1) edge to edge, its first and last pixel included (upstream's `spread_grid`).
    /// Stride 0 is no grid.
    public static func spread(
        width: Int, height: Int, stride step: Int, region: SIMD4<Float> = SIMD4(0, 0, 1, 1)
    ) -> [SIMD2<Float>] {
        let xs = spreadAxis(width, low: region.x, high: region.z, stride: step)
        return spreadAxis(height, low: region.y, high: region.w, stride: step).flatMap { y in
            xs.map { SIMD2(Float($0), Float(y)) }
        }
    }

    /// Pixel centers along one axis of `[low, high]` (fractions of `size`).
    public static func spreadAxis(_ size: Int, low: Float, high: Float, stride step: Int) -> [Int] {
        let first = max(0, Int((low * Float(size) - 0.5).rounded(.up)))
        let last = min(size - 1, Int((high * Float(size) - 0.5).rounded(.down)))
        guard step > 0, last >= first else { return [] }
        let span = last - first
        let count = (2 * span + step) / (2 * step) + 1
        if count == 1 { return [first + span / 2] }
        return (0 ..< count).map { first + ($0 * span + (count - 1) / 2) / (count - 1) }
    }

    /// Grid seeds over `region` plus clicked points (model pixels, fractional), a click that
    /// lands exactly on a grid node or repeats another counted once (upstream's
    /// `track_pixels`).
    public static func seeds(
        width: Int, height: Int, stride step: Int, region: SIMD4<Float>, picks: [SIMD2<Float>]
    ) -> [SIMD2<Float>] {
        let xs = Set(spreadAxis(width, low: region.x, high: region.z, stride: step).map(Float.init))
        let ys = Set(spreadAxis(height, low: region.y, high: region.w, stride: step).map(Float.init))
        // torch.unique: distinct, sorted by x then y.
        let extra = Set(picks.map {
            SIMD2(min(max($0.x, 0), Float(width - 1)), min(max($0.y, 0), Float(height - 1)))
        }).filter { !(xs.contains($0.x) && ys.contains($0.y)) }
            .sorted { ($0.x, $0.y) < ($1.x, $1.y) }
        return spread(width: width, height: height, stride: step, region: region) + extra
    }
}

public enum QueryBuilder {
    /// Each frame asked about its own pixels, frame-major; `reference` nil means each
    /// frame's own camera.
    public static func perFrame(_ xy: [SIMD2<Float>], frames: Int, reference: Int?) -> Queries {
        var queries = Queries(xy: [], source: [], target: [], reference: [])
        let rounded = xy.map { $0.rounded(.toNearestOrEven) }
        for frame in 0 ..< frames {
            let f = Int32(frame)
            queries.append(Queries(
                xy: rounded,
                source: Array(repeating: f, count: xy.count), target: Array(repeating: f, count: xy.count),
                reference: Array(repeating: reference.map(Int32.init) ?? f, count: xy.count)))
        }
        return queries
    }

    /// Seeds in `source` asked in each of `targets`, target-major.
    public static func track(_ xy: [SIMD2<Float>], source: Int, targets: [Int], reference: Int) -> Queries {
        track(xy, sources: Array(repeating: source, count: xy.count), targets: targets, reference: reference)
    }

    /// Seeds, each in its own source frame, asked in each of `targets`, target-major.
    public static func track(_ xy: [SIMD2<Float>], sources: [Int], targets: [Int], reference: Int) -> Queries {
        precondition(sources.count == xy.count)
        var queries = Queries(xy: [], source: [], target: [], reference: [])
        let source = sources.map(Int32.init)
        for target in targets {
            queries.append(Queries(
                xy: xy, source: source,
                target: Array(repeating: Int32(target), count: xy.count),
                reference: Array(repeating: Int32(reference), count: xy.count)))
        }
        return queries
    }
}

/// Settings for one run's queries; upstream's inference defaults.
public struct TaskSettings: Sendable {
    public var reconstructionStride = 2
    public var depthStride = 2
    /// Grid spacing for track seeds; 0 is no grid (clicked points only).
    public var trackStride = 2
    /// The frame the region and picks are on.
    public var trackSource = 0
    /// Where the seed grid spans: x0, y0, x1, y1 as fractions of the frame.
    public var trackRegion = SIMD4<Float>(0, 0, 1, 1)
    /// Clicked seeds in model pixels, tracked besides the grid.
    public var trackPicks: [SIMD2<Float>] = []
    public var poseGrid = 64
    /// The camera tracks and reconstruction are expressed in.
    public var reference = 0

    public init() {}

    /// The track seeds for a `width` × `height` frame: the region's grid and the picks.
    public func trackSeeds(width: Int, height: Int) -> [SIMD2<Float>] {
        QueryGrid.seeds(width: width, height: height, stride: trackStride, region: trackRegion, picks: trackPicks)
    }

    /// Every track seed and its source frame. With several cameras and the whole frame as
    /// the region, each camera's frame at `trackSource`'s timestep gets the grid, as
    /// upstream seeds a rig; a drawn region bounds the grid on `trackSource` only, the
    /// camera it was drawn on. Picks stay on `trackSource`.
    public func trackSeeds(width: Int, height: Int, views: Int) -> (xy: [SIMD2<Float>], sources: [Int]) {
        let own = trackSeeds(width: width, height: height)
        guard views > 1, trackRegion == SIMD4(0, 0, 1, 1), trackStride > 0 else {
            return (own, Array(repeating: trackSource, count: own.count))
        }
        let grid = QueryGrid.spread(width: width, height: height, stride: trackStride)
        let first = trackSource / views * views
        var xy: [SIMD2<Float>] = [], sources: [Int] = []
        for frame in first ..< first + views {
            let seeds = frame == trackSource ? own : grid
            xy += seeds
            sources += Array(repeating: frame, count: seeds.count)
        }
        return (xy, sources)
    }

    /// The queries for `tasks` in order, and where each task's block sits. `views` is the
    /// number of synchronized cameras (`FrameSet.views`).
    public func queries(_ tasks: [ArrowTask], frames: Int, views: Int = 1, width: Int, height: Int) -> (Queries, [QueryGroup]) {
        var all = Queries(xy: [], source: [], target: [], reference: [])
        var groups: [QueryGroup] = []
        for task in tasks {
            let block: Queries
            let perFrame: Int
            var reference: Int? = reference
            switch task {
            case .reconstruct:
                let grid = QueryGrid.strided(width: width, height: height, stride: reconstructionStride)
                block = QueryBuilder.perFrame(grid, frames: frames, reference: self.reference)
                perFrame = grid.count
            case .depth:
                let grid = QueryGrid.strided(width: width, height: height, stride: depthStride)
                block = QueryBuilder.perFrame(grid, frames: frames, reference: nil)
                perFrame = grid.count
                reference = nil
            case .track:
                let seeds = trackSeeds(width: width, height: height, views: views)
                block = QueryBuilder.track(seeds.xy, sources: seeds.sources, targets: Array(0 ..< frames), reference: self.reference)
                perFrame = seeds.xy.count
            case .pose:
                let grid = QueryGrid.sized(width: width, height: height, size: poseGrid)
                let middle = PoseFit.middle(frames)
                block = QueryBuilder.perFrame(grid, frames: frames, reference: nil)
                    + QueryBuilder.perFrame(grid, frames: frames, reference: middle)
                perFrame = grid.count
                reference = middle
            }
            groups.append(QueryGroup(task: task, start: all.count, count: block.count, perFrame: perFrame, reference: reference))
            all.append(block)
        }
        return (all, groups)
    }
}

/// A camera: pose in the reference camera's frame and pinhole intrinsics, nil where the
/// fit failed.
public struct CameraEstimate: Sendable {
    /// Maps this frame's camera coordinates to the reference camera's.
    public var cameraToReference: simd_float4x4
    public var valid: Bool
    /// fx, fy, cx, cy in model pixels.
    public var intrinsics: SIMD4<Float>?

    public init(cameraToReference: simd_float4x4, valid: Bool, intrinsics: SIMD4<Float>?) {
        self.cameraToReference = cameraToReference
        self.valid = valid
        self.intrinsics = intrinsics
    }
}

/// Upstream's `estimate_poses`: each frame's points in its own camera against the same
/// pixels' points in the reference camera, a weighted rigid fit between them, and
/// least-squares pinhole intrinsics from its own pointmap.
public enum PoseFit {
    public static let confidenceKeep: Float = 0.5
    public static let weightPower: Float = 2
    public static let minimumPoints = 8
    public static let visibilityThreshold: Float = 0

    public static func middle(_ frames: Int) -> Int { frames / 2 }

    /// `predictions` and `xy` hold the pose group: per frame, its own camera block then
    /// (after all frames) the reference camera block.
    public static func cameras(
        _ predictions: Predictions, xy: [SIMD2<Float>], frames: Int, perFrame: Int, reference: Int
    ) -> [CameraEstimate] {
        (0 ..< frames).map { frame in
            var estimate = CameraEstimate(cameraToReference: matrix_identity_float4x4, valid: frame == reference, intrinsics: nil)
            let own = frame * perFrame, shared = (frames + frame) * perFrame
            func usable(_ i: Int) -> Bool {
                let uv = predictions.uv[i]
                return uv.x.isFinite && uv.y.isFinite && uv.x >= 0 && uv.y >= 0 && uv.x <= 1 && uv.y <= 1
                    && predictions.visibility[i] > visibilityThreshold
            }
            var index: [Int] = []
            var weight = [Float](repeating: 0, count: perFrame)
            for k in 0 ..< perFrame {
                let a = own + k, b = shared + k
                let w = predictions.confidence[a] * predictions.confidence[b]
                weight[k] = w
                let finite = isFinite(predictions.points[a]) && isFinite(predictions.points[b])
                if finite, w.isFinite, w > 0, usable(a), usable(b) { index.append(k) }
            }
            guard index.count >= minimumPoints else { return estimate }
            if confidenceKeep < 1 {
                let cut = quantile(index.map { weight[$0] }, 1 - confidenceKeep)
                index = index.filter { weight[$0] >= cut }
            }
            guard index.count >= minimumPoints else { return estimate }
            if frame != reference,
               let rigid = weightedRigid(
                   target: index.map { predictions.points[shared + $0] },
                   source: index.map { predictions.points[own + $0] },
                   weight: index.map { Foundation.pow(weight[$0], weightPower) }) {
                estimate.cameraToReference = rigid
                estimate.valid = true
            }
            let positive = index.filter { predictions.points[own + $0].z > 0 }
            if positive.count >= minimumPoints {
                estimate.intrinsics = pinhole(
                    points: positive.map { predictions.points[own + $0] }, xy: positive.map { xy[own + $0] })
            }
            return estimate
        }
    }

    /// Poses re-expressed in `frame`'s camera (upstream's viewer: inv(c[frame]) · c), so
    /// they share the coordinates of tracks and reconstruction asked in that frame. Left as
    /// they are when `frame` has no valid pose.
    public static func rebased(_ cameras: [CameraEstimate], onto frame: Int) -> [CameraEstimate] {
        guard frame < cameras.count, cameras[frame].valid else { return cameras }
        let inverse = cameras[frame].cameraToReference.inverse
        return cameras.map { camera in
            var rebased = camera
            if camera.valid { rebased.cameraToReference = inverse * camera.cameraToReference }
            return rebased
        }
    }

    /// torch.quantile's linear interpolation.
    static func quantile(_ values: [Float], _ q: Float) -> Float {
        let sorted = values.sorted()
        let position = q * Float(sorted.count - 1)
        let low = Int(position.rounded(.down)), high = min(low + 1, sorted.count - 1)
        return sorted[low] + (sorted[high] - sorted[low]) * (position - Float(low))
    }

    /// The rotation and translation minimizing Σ w‖R·source + t − target‖², by Horn's
    /// quaternion method (the same optimum as the reflection-corrected SVD fit).
    static func weightedRigid(target: [SIMD3<Float>], source: [SIMD3<Float>], weight: [Float]) -> simd_float4x4? {
        guard target.count >= 3 else { return nil }
        let total = weight.reduce(0, +)
        guard total.isFinite, total > 0 else { return nil }
        var sourceMean = SIMD3<Double>(), targetMean = SIMD3<Double>()
        for i in 0 ..< target.count {
            let w = Double(weight[i] / total)
            sourceMean += w * SIMD3<Double>(source[i])
            targetMean += w * SIMD3<Double>(target[i])
        }
        // Cross-covariance S = Σ w (s − s̄)(t − t̄)ᵀ.
        var s = [[Double]](repeating: [0, 0, 0], count: 3)
        for i in 0 ..< target.count {
            let w = Double(weight[i] / total)
            let a = SIMD3<Double>(source[i]) - sourceMean, b = SIMD3<Double>(target[i]) - targetMean
            for r in 0 ..< 3 { for c in 0 ..< 3 { s[r][c] += w * a[r] * b[c] } }
        }
        let (xx, xy, xz) = (s[0][0], s[0][1], s[0][2])
        let (yx, yy, yz) = (s[1][0], s[1][1], s[1][2])
        let (zx, zy, zz) = (s[2][0], s[2][1], s[2][2])
        let n: [[Double]] = [
            [xx + yy + zz, yz - zy, zx - xz, xy - yx],
            [yz - zy, xx - yy - zz, xy + yx, zx + xz],
            [zx - xz, xy + yx, -xx + yy - zz, yz + zy],
            [xy - yx, zx + xz, yz + zy, -xx - yy + zz],
        ]
        let q = largestEigenvector(n)
        let quaternion = simd_quatd(ix: q[1], iy: q[2], iz: q[3], r: q[0]).normalized
        let rotation = simd_double3x3(quaternion)
        let translation = targetMean - rotation * sourceMean
        guard translation.x.isFinite, translation.y.isFinite, translation.z.isFinite else { return nil }
        func column(_ c: SIMD3<Double>, _ w: Float) -> SIMD4<Float> { SIMD4(SIMD3<Float>(c), w) }
        return simd_float4x4(
            column(rotation.columns.0, 0), column(rotation.columns.1, 0), column(rotation.columns.2, 0),
            column(translation, 1))
    }

    static func isFinite(_ v: SIMD3<Float>) -> Bool { v.x.isFinite && v.y.isFinite && v.z.isFinite }

    /// The eigenvector of the largest eigenvalue of a symmetric 4 × 4 matrix (Jacobi sweeps).
    static func largestEigenvector(_ input: [[Double]]) -> [Double] {
        var a = input
        var v: [[Double]] = (0 ..< 4).map { r in (0 ..< 4).map { $0 == r ? 1 : 0 } }
        for _ in 0 ..< 64 {
            var off = 0.0
            for p in 0 ..< 4 { for q in p + 1 ..< 4 { off += a[p][q] * a[p][q] } }
            if off < 1e-30 { break }
            for p in 0 ..< 4 {
                for q in p + 1 ..< 4 where abs(a[p][q]) > 1e-300 {
                    let theta = (a[q][q] - a[p][p]) / (2 * a[p][q])
                    let t = (theta >= 0 ? 1 : -1) / (abs(theta) + (theta * theta + 1).squareRoot())
                    let c = 1 / (t * t + 1).squareRoot(), s = t * c
                    for k in 0 ..< 4 {
                        let akp = a[k][p], akq = a[k][q]
                        a[k][p] = c * akp - s * akq
                        a[k][q] = s * akp + c * akq
                    }
                    for k in 0 ..< 4 {
                        let apk = a[p][k], aqk = a[q][k]
                        a[p][k] = c * apk - s * aqk
                        a[q][k] = s * apk + c * aqk
                    }
                    for k in 0 ..< 4 {
                        let vkp = v[k][p], vkq = v[k][q]
                        v[k][p] = c * vkp - s * vkq
                        v[k][q] = s * vkp + c * vkq
                    }
                }
            }
        }
        let best = (0 ..< 4).max { a[$0][$0] < a[$1][$1] }!
        return (0 ..< 4).map { v[$0][best] }
    }

    /// Pixel-center intrinsics with xy = f · (X, Y) / Z + c, least squares per axis.
    static func pinhole(points: [SIMD3<Float>], xy: [SIMD2<Float>]) -> SIMD4<Float>? {
        var result = SIMD4<Float>()
        for axis in 0 ..< 2 {
            var sr = 0.0, srr = 0.0, sp = 0.0, srp = 0.0
            for (point, pixel) in zip(points, xy) {
                let r = Double(point[axis] / point.z), p = Double(pixel[axis])
                sr += r; srr += r * r; sp += p; srp += r * p
            }
            let n = Double(points.count)
            let det = n * srr - sr * sr
            guard det != 0 else { return nil }
            let focal = (n * srp - sr * sp) / det
            let center = (sp - focal * sr) / n
            guard focal > 0 else { return nil }
            result[axis] = Float(focal)
            result[axis + 2] = Float(center)
        }
        return result
    }
}

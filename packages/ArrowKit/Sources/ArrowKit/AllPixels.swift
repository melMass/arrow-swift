import Foundation

/// Which pixels of the clip a track already covers, for all-pixel tracking (D4RT's
/// Algorithm 1, upstream's `track_all_pixels`): every `stride`-th pixel is tracked once.
/// Frames are seeded in `order`; the first seeds its whole lattice, each later one only the
/// lattice pixels no earlier track was predicted visible at, within `radius` pixels. Sky
/// pixels start covered, so they are never seeded.
public struct PixelCoverage: Sendable {
    public let width: Int
    public let height: Int
    public let frames: Int
    public let stride: Int
    /// Frame-major, row-major, like `SkyMask.sky`.
    public private(set) var covered: [Bool]

    public init(width: Int, height: Int, frames: Int, stride: Int, sky: SkyMask? = nil) {
        precondition(stride > 0)
        precondition(sky.map { $0.width == width && $0.height == height && $0.frames == frames } ?? true)
        self.width = width
        self.height = height
        self.frames = frames
        self.stride = stride
        covered = sky?.sky ?? Array(repeating: false, count: frames * width * height)
    }

    /// How far around a track's predicted pixel counts as covered.
    public var radius: Int { max(1, (stride + 1) / 2) }

    /// Lattice pixels in one frame.
    public var latticeCount: Int { ((width + stride - 1) / stride) * ((height + stride - 1) / stride) }

    /// The seeding order: `source` to the last frame, then back from `source − 1` to the first.
    public static func order(frames: Int, source: Int) -> [Int] {
        Array(source ..< frames) + Array((0 ..< source).reversed())
    }

    /// The lattice pixels of `frame` not covered yet, row-major, now marked covered.
    public mutating func seed(_ frame: Int) -> [SIMD2<Float>] {
        var seeds: [SIMD2<Float>] = []
        for y in Swift.stride(from: 0, to: height, by: stride) {
            for x in Swift.stride(from: 0, to: width, by: stride) {
                let i = (frame * height + y) * width + x
                guard !covered[i] else { continue }
                covered[i] = true
                seeds.append(SIMD2(Float(x), Float(y)))
            }
        }
        return seeds
    }

    /// Marks every pixel, in every frame but `source`, that a block of `seeds` tracks from
    /// `source` was predicted visible at. `predictions` are target-major over all frames.
    public mutating func mark(_ predictions: Predictions, seeds: Int, source: Int) {
        precondition(predictions.count == seeds * frames)
        for frame in 0 ..< frames where frame != source {
            for s in 0 ..< seeds {
                let i = frame * seeds + s
                let uv = predictions.uv[i]
                let fx = (uv.x * Float(width) - 0.5).rounded(.toNearestOrEven)
                let fy = (uv.y * Float(height) - 0.5).rounded(.toNearestOrEven)
                // Also rejects NaN; a point far off the frame covers nothing.
                guard predictions.visibility[i] > 0,
                      abs(fx) < Float(width + radius + 1), abs(fy) < Float(height + radius + 1) else { continue }
                let x = Int(fx), y = Int(fy)
                let x0 = max(x - radius, 0), x1 = min(x + radius, width - 1)
                let y0 = max(y - radius, 0), y1 = min(y + radius, height - 1)
                guard x0 <= x1, y0 <= y1 else { continue }
                for py in y0 ... y1 {
                    for px in x0 ... x1 { covered[(frame * height + py) * width + px] = true }
                }
            }
        }
    }
}

/// Track blocks seeded frame by frame, each target-major over all frames, read as one
/// target-major track group: at each target, every block's seeds in seeding order.
struct TrackBlocks {
    let frames: Int
    private(set) var queries = Queries(xy: [], source: [], target: [], reference: [])
    private(set) var predictions = Predictions()
    /// Seeds per block, in seeding order.
    private(set) var seeds: [Int] = []

    init(frames: Int) {
        self.frames = frames
    }

    var total: Int { seeds.reduce(0, +) }

    mutating func append(_ block: Queries, seeds count: Int) {
        precondition(block.count == count * frames)
        queries.append(block)
        predictions.append(.unanswered(block.count))
        seeds.append(count)
    }

    /// Where each query of block `b` sits in the merged group.
    func positions(_ b: Int) -> [Int] {
        let n = seeds[b], offset = seeds[..<b].reduce(0, +), total = total
        return (0 ..< n * frames).map { ($0 / n) * total + offset + $0 % n }
    }

    mutating func answer(_ b: Int, _ answers: Predictions) {
        let start = seeds[..<b].reduce(0, +) * frames
        predictions.fill(answers, from: start)
    }

    /// The blocks laid out as one target-major group.
    func merged() -> (Queries, Predictions) {
        var order = Array(repeating: 0, count: queries.count)
        var start = 0
        for b in seeds.indices {
            for (k, position) in positions(b).enumerated() { order[position] = start + k }
            start += seeds[b] * frames
        }
        return (queries.select(order), predictions.select(order))
    }
}

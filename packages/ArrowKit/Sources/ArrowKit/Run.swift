import CoreGraphics
import Foundation
import MLX

/// One run: the frames, what was asked, what came back.
public struct ArrowResult: Sendable {
    public var frames: FrameSet
    public var queries: Queries
    public var predictions: Predictions
    public var groups: [QueryGroup]
    /// From the pose task, else nil. Expressed in the camera tracks and reconstruction use
    /// (`TaskSettings.reference`), not the pose task's own middle frame.
    public var cameras: [CameraEstimate]?
    public var encodeSeconds: Double
    public var decodeSeconds: Double
    /// The sky the run skipped: its queries were not decoded and are NaN. Not saved in
    /// archives; the NaN answers are.
    public var sky: SkyMask?

    public init(
        frames: FrameSet, queries: Queries, predictions: Predictions, groups: [QueryGroup],
        cameras: [CameraEstimate]? = nil, encodeSeconds: Double = 0, decodeSeconds: Double = 0
    ) {
        self.frames = frames
        self.queries = queries
        self.predictions = predictions
        self.groups = groups
        self.cameras = cameras
        self.encodeSeconds = encodeSeconds
        self.decodeSeconds = decodeSeconds
    }

    public func group(_ task: ArrowTask) -> QueryGroup? { groups.first { $0.task == task } }

    /// Each query's colour at its source pixel, 0...1.
    public func colors(_ group: QueryGroup) -> [SIMD3<Float>] {
        let pixels = frames.images.map { FrameSet.rgba($0, width: frames.width, height: frames.height) }
        return (group.start ..< group.start + group.count).map { i in
            let p = queries.xy[i]
            let x = min(max(Int(p.x.rounded()), 0), frames.width - 1), y = min(max(Int(p.y.rounded()), 0), frames.height - 1)
            let rgba = pixels[Int(queries.source[i])]
            let o = (y * frames.width + x) * 4
            return SIMD3(Float(rgba[o]), Float(rgba[o + 1]), Float(rgba[o + 2])) / 255
        }
    }
}

public enum ArrowEvent: Sendable {
    case downloading(Double)
    case loading
    case encoding
    /// Fraction of the queries answered.
    case decoding(Double)
    /// The run's frames, queries and groups, with every prediction still unanswered
    /// (`Predictions.unanswered`): what `answered` chunks fill in.
    case started(ArrowResult)
    /// A chunk's answers, for the queries from `start` on. Cameras come only with the
    /// finished result.
    case answered(start: Int, Predictions)
    /// All-pixel tracking seeded another frame: the run so far, laid out anew with that
    /// frame's tracks unanswered. Positions in later chunks refer to this layout.
    case grew(ArrowResult)
    /// A chunk's answers for the queries at these positions.
    case placed([Int], Predictions)
}

/// The model, loaded once, and the last encoding, kept for further questions about the
/// same frames.
public actor ArrowRunner {
    public let model: ArrowModel
    private var encoded: (frames: [CGImage], encoding: Encoded)?

    public init(model: ArrowModel) {
        self.model = model
    }

    /// Downloads the weights if needed and loads them.
    public static func load(
        precision: ArrowPrecision = .float16, store: ArrowStore = ArrowStore(),
        progress: @Sendable @escaping (ArrowEvent) -> Void = { _ in }
    ) async throws -> ArrowRunner {
        let folder = try await store.snapshot { progress(.downloading($0.fractionCompleted)) }
        progress(.loading)
        let model = try ArrowModel.load(from: folder, precision: precision)
        return ArrowRunner(model: model)
    }

    /// Runs `tasks` on `frames`, encoding them unless the last run had the same frames.
    /// Queries whose source pixel is in `sky` (`SkySegmenter.mask`) are skipped.
    public func run(
        _ frames: FrameSet, tasks: [ArrowTask], settings: TaskSettings = TaskSettings(), chunk: Int = 8192,
        sky: SkyMask? = nil, progress: @Sendable @escaping (ArrowEvent) -> Void = { _ in }
    ) throws -> ArrowResult {
        precondition(sky.map { $0.width == frames.width && $0.height == frames.height && $0.frames == frames.images.count } ?? true)
        let cacheLimit = Memory.cacheLimit
        Memory.cacheLimit = 1 << 30
        defer { Memory.cacheLimit = cacheLimit }

        let started = Date()
        let encoding: Encoded
        if let previous = encoded, previous.frames.count == frames.images.count,
           zip(previous.frames, frames.images).allSatisfy({ $0 === $1 }) {
            encoding = previous.encoding
        } else {
            encoded = nil
            Memory.clearCache()
            progress(.encoding)
            encoding = model.encode(frames.tensor())
            encoded = (frames.images, encoding)
        }
        let encodeSeconds = Date().timeIntervalSince(started)

        let (queries, groups) = settings.queries(tasks, frames: frames.images.count, views: frames.views, width: frames.width, height: frames.height)
        let allPixels = settings.trackAllPixels && tasks.contains(.track)
        let decodeStart = Date()
        var skeleton = ArrowResult(
            frames: frames, queries: queries, predictions: .unanswered(queries.count), groups: groups,
            encodeSeconds: encodeSeconds)
        skeleton.sky = sky
        progress(.started(skeleton))
        progress(.decoding(0))
        // All-pixel tracking's size is known only as it goes; its first frame's block stands in.
        let estimate = allPixels
            ? PixelCoverage(width: frames.width, height: frames.height, frames: 1, stride: settings.allPixelsStride).latticeCount
                * frames.images.count
            : 0
        let share = Double(queries.count) / Double(max(queries.count + estimate, 1))
        let predictions = try model.decode(
            queries, encoded: encoding, chunk: chunk, skipped: sky?.skipped(queries),
            answered: { progress(.answered(start: $0, $1)) },
            progress: { progress(.decoding($0 * share)) })
        var result = ArrowResult(
            frames: frames, queries: queries, predictions: predictions, groups: groups, cameras: nil,
            encodeSeconds: encodeSeconds)
        result.sky = sky
        if allPixels {
            try trackAllPixels(
                &result, encoding: encoding, settings: settings, chunk: chunk, sky: sky,
                progress: { progress($0) }, fraction: { share + $0 * (1 - share) })
        }
        result.decodeSeconds = Date().timeIntervalSince(decodeStart)
        if let pose = result.group(.pose), let reference = pose.reference {
            let range = pose.start ..< pose.start + pose.count
            let fitted = PoseFit.cameras(
                predictions.select(Array(range)), xy: Array(queries.xy[range]), frames: frames.images.count,
                perFrame: pose.perFrame, reference: reference)
            result.cameras = PoseFit.rebased(fitted, onto: settings.reference)
        }
        Memory.clearCache()
        return result
    }

    /// Upstream's `track_all_pixels` after the other tasks: each frame in
    /// `PixelCoverage.order` seeds what is still uncovered, its tracks are decoded through
    /// every frame, and where they are seen covers those pixels for the frames after it.
    /// The tracks are appended to `result` as one target-major track group. Sky never seeds
    /// (`PixelCoverage`), so no query is skipped.
    private func trackAllPixels(
        _ result: inout ArrowResult, encoding: Encoded, settings: TaskSettings, chunk: Int, sky: SkyMask?,
        progress: (ArrowEvent) -> Void, fraction: (Double) -> Double
    ) throws {
        let frames = result.frames, count = frames.images.count
        let fixed = (queries: result.queries, predictions: result.predictions, groups: result.groups)
        var coverage = PixelCoverage(
            width: frames.width, height: frames.height, frames: count, stride: settings.allPixelsStride, sky: sky)
        var blocks = TrackBlocks(frames: count)

        func layout() -> ArrowResult {
            var current = result
            let (queries, predictions) = blocks.merged()
            current.queries = fixed.queries + queries
            current.predictions = fixed.predictions
            current.predictions.append(predictions)
            current.groups = fixed.groups + [QueryGroup(
                task: .track, start: fixed.queries.count, count: queries.count, perFrame: blocks.total,
                reference: settings.reference)]
            return current
        }

        let order = PixelCoverage.order(frames: count, source: min(max(settings.trackSource, 0), count - 1))
        for (step, frame) in order.enumerated() {
            let seeds = coverage.seed(frame)
            guard !seeds.isEmpty else { continue }
            let block = QueryBuilder.track(seeds, source: frame, targets: Array(0 ..< count), reference: settings.reference)
            blocks.append(block, seeds: seeds.count)
            let b = blocks.seeds.count - 1
            progress(.grew(layout()))
            let positions = blocks.positions(b).map { $0 + fixed.queries.count }
            let answers = try model.decode(
                block, encoded: encoding, chunk: chunk,
                answered: { progress(.placed(Array(positions[$0 ..< $0 + $1.count]), $1)) },
                progress: { progress(.decoding(fraction((Double(step) + $0) / Double(order.count)))) })
            blocks.answer(b, answers)
            coverage.mark(answers, seeds: seeds.count, source: frame)
        }
        if !blocks.seeds.isEmpty { result = layout() }
    }
}

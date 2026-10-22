import ArrowKit
import Foundation
import Observation

/// One window's work: the input, the frames taken from it, the model, the last result,
/// and where playback is.
@MainActor @Observable
final class Session {
    enum Phase: Equatable {
        case idle
        case reading
        case downloading(Double)
        case loading
        case encoding
        case decoding(Double)
        case failed(String)

        var isBusy: Bool {
            switch self {
            case .idle, .failed: false
            default: true
            }
        }

        var label: String {
            switch self {
            case .idle: "Ready"
            case .reading: "Reading frames…"
            case let .downloading(f): "Downloading ARROW (\(ArrowStore.downloadSize))… \(Int(f * 100))%"
            case .loading: "Loading the model…"
            case .encoding: "Encoding frames…"
            case let .decoding(f): "Answering queries… \(Int(f * 100))%"
            case let .failed(message): message
            }
        }
    }

    /// What a drag or click on the frame does.
    enum SeedTool: String, CaseIterable, Identifiable {
        case region = "Region", points = "Points"
        var id: Self { self }
    }

    /// How frames are picked from the input.
    enum Sampling: String, CaseIterable, Identifiable {
        case spread = "Spread", range = "Range"
        var id: Self { self }
    }

    /// The most timesteps a run takes. Encoding attends across every frame and each query
    /// reads all of them, so memory and time grow with the count.
    static let frameLimit = 300

    /// One video or image folder per camera; several are synchronized views of one scene,
    /// aligned frame for frame.
    var sources: [URL] = []
    var source: URL? { sources.first }
    /// Frames in the shortest source, once read.
    private(set) var sourceFrames: Int?
    var frames: FrameSet?
    var result: ArrowResult? {
        didSet {
            sceneDiagonal = result?.sceneDiagonal ?? 1
            readTracks()
        }
    }
    var phase: Phase = .idle

    var sampling: Sampling = .spread
    /// Frames taken evenly across the input.
    var frameCount = 8
    /// Every `rangeStep`-th frame from `rangeStart` to `rangeEnd` (inclusive), 0-based.
    var rangeStart = 0
    var rangeEnd = 0
    var rangeStep = 1
    /// The long side the model sees, in pixels (a multiple of 14).
    var maxSide = 504
    var tasks: Set<ArrowTask> = [.track, .reconstruct, .pose]
    var settings: TaskSettings = {
        var s = TaskSettings()
        s.trackStride = 12
        s.reconstructionStride = 4
        s.depthStride = 4
        s.poseGrid = 32
        return s
    }()

    var seedTool: SeedTool = .region
    /// Seed region, x0 y0 x1 y1 as fractions of the frame.
    var region = Session.fullFrame { didSet { releaseSeeds() } }
    /// Clicked seeds as fractions of the frame (u, v).
    var picks: [SIMD2<Float>] = [] { didSet { releaseSeeds() } }
    /// The frame (camera at timestep) the region and picks were placed on, which tracks
    /// then start from: they mark what is under them in that frame only. Nil while there
    /// are none.
    private(set) var seedFrame: Int?
    static let fullFrame = SIMD4<Float>(0, 0, 1, 1)
    /// Show the seeds on the frame even once there are tracks.
    var showSeeds = false

    var display = DisplaySettings() {
        didSet { if display.fusion != oldValue.fusion { readTracks() } }
    }
    /// The keep fraction upstream's heuristic recommends for the current result.
    private(set) var recommendedKeep: Float = 1
    /// The timestep shown.
    var frame = 0
    /// The camera whose frame is shown, with several.
    var camera = 0 { didSet { if camera != oldValue { readTracks() } } }
    var isPlaying = false

    /// Colours at each query's source pixel, per task, read once per result.
    private(set) var colors: [ArrowTask: [SIMD3<Float>]] = [:]
    /// The track group per timestep, cameras folded as `display.fusion` says.
    private(set) var tracks: TrackSet?
    private(set) var statistics: TrackStatistics?
    private(set) var sceneDiagonal: Float = 1

    private var runner: ArrowRunner?
    private var job: Task<Void, Never>?
    private var player: Task<Void, Never>?

    init() {}

    /// Timesteps to scrub through.
    var frameTotal: Int { frames?.timesteps ?? 0 }
    var views: Int { frames?.views ?? max(sources.count, 1) }
    /// The frame shown: `camera` at timestep `frame`.
    var displayFrame: Int { frame * views + min(camera, views - 1) }

    /// The frames the sampling settings select from the shortest source.
    var selection: FrameSelection {
        switch sampling {
        case .spread: .spread(frameCount)
        case .range: .range(start: rangeStart, end: rangeEnd + 1, step: max(rangeStep, 1), limit: Self.frameLimit)
        }
    }

    var selectedCount: Int? { sourceFrames.map { selection.indices(of: $0).count } }

    func open(_ url: URL) {
        if url.pathExtension == ResultArchive.fileExtension {
            openResult(url)
            return
        }
        sources = [url]
        sourcesChanged()
    }

    /// Adds a synchronized camera: another video or folder of the same scene, aligned
    /// frame for frame with the others.
    func addCamera(_ url: URL) {
        sources.append(url)
        sourcesChanged()
    }

    func removeCamera(at index: Int) {
        guard sources.indices.contains(index), sources.count > 1 else { return }
        sources.remove(at: index)
        sourcesChanged()
    }

    private func sourcesChanged() {
        frames = nil
        result = nil
        sourceFrames = nil
        picks = []
        region = Session.fullFrame
        frame = 0
        camera = 0
        readFrames()
    }

    /// Applies a seed edit made on the frame shown. Seeds left on another frame are
    /// cleared first: a spot on one frame is not the same thing on another.
    func editSeeds(_ edit: (Session) -> Void) {
        if let seedFrame, seedFrame != displayFrame {
            picks = []
            region = Session.fullFrame
        }
        edit(self)
        if picks.isEmpty && region == Session.fullFrame { return }
        seedFrame = displayFrame
    }

    private func releaseSeeds() {
        if picks.isEmpty && region == Session.fullFrame { seedFrame = nil }
    }

    /// Where tracks start: the seeds' frame, else the one shown.
    var trackSource: Int { seedFrame ?? displayFrame }

    /// "frame 3", or "frame 3, camera 2" with several cameras, 1-based.
    func frameName(_ index: Int) -> String {
        views > 1 ? "frame \(index / views + 1), camera \(index % views + 1)" : "frame \(index + 1)"
    }

    /// Shows the frame tracks start on.
    func showTrackSource() {
        let views = self.views
        frame = trackSource / views
        camera = trackSource % views
    }

    /// Re-reads the frames after the sources, sampling or size changed.
    func readFrames() {
        guard !sources.isEmpty else { return }
        job?.cancel()
        let sources = self.sources, side = maxSide
        let sampling = self.sampling
        phase = .reading
        job = Task {
            do {
                let inputs = try sources.map(FrameSource.at)
                if self.sourceFrames == nil {
                    var shortest = Int.max
                    for input in inputs { shortest = min(shortest, try await input.frameCount()) }
                    self.sourceFrames = shortest
                }
                // A range starts out as the first `frameLimit` frames.
                if sampling == .range, let shortest = self.sourceFrames, self.rangeEnd == 0 || self.rangeEnd >= shortest {
                    self.rangeEnd = min(shortest, Self.frameLimit) - 1
                }
                let frames = try await FrameSet.load(views: inputs, selection: self.selection, maxSide: side)
                try Task.checkCancellation()
                // Other frames sample other moments: frame k is no longer the seeds' frame.
                if frames.times != self.frames?.times {
                    self.picks = []
                    self.region = Session.fullFrame
                }
                self.frames = frames
                self.result = nil
                self.frame = min(self.frame, frames.timesteps - 1)
                self.camera = min(self.camera, frames.views - 1)
                self.phase = .idle
            } catch is CancellationError {
            } catch {
                self.phase = .failed(error.localizedDescription)
            }
        }
    }

    /// The run's settings with the seeds as drawn, in model pixels.
    func runSettings(for frames: FrameSet) -> TaskSettings {
        var settings = self.settings
        settings.trackSource = min(trackSource, frames.images.count - 1)
        settings.trackRegion = region
        settings.trackPicks = picks.map { SIMD2($0.x * Float(frames.width) - 0.5, $0.y * Float(frames.height) - 0.5) }
        return settings
    }

    /// Track seeds on the seeds' frame as they would be sent now, in model pixels.
    var seeds: [SIMD2<Float>] {
        guard let frames else { return [] }
        return runSettings(for: frames).trackSeeds(width: frames.width, height: frames.height)
    }

    /// Every track seed, in every camera it is placed in.
    var seedTotal: Int {
        guard let frames else { return 0 }
        return runSettings(for: frames).trackSeeds(width: frames.width, height: frames.height, views: frames.views).xy.count
    }

    var canRun: Bool {
        guard frames != nil, !tasks.isEmpty, !phase.isBusy else { return false }
        return !tasks.contains(.track) || settings.trackStride > 0 || !picks.isEmpty
    }

    func run() {
        guard let frames, canRun else { return }
        job?.cancel()
        let ordered = ArrowTask.allCases.filter(tasks.contains)
        let settings = runSettings(for: frames)
        job = Task {
            do {
                let progress: @Sendable (ArrowEvent) -> Void = { event in
                    Task { @MainActor in self.show(event) }
                }
                if runner == nil { runner = try await ArrowRunner.load(progress: progress) }
                guard let runner else { return }
                let result = try await runner.run(frames, tasks: ordered, settings: settings, progress: progress)
                try Task.checkCancellation()
                self.show(result)
                self.phase = .idle
            } catch is CancellationError {
                self.phase = .idle
            } catch {
                self.phase = .failed(error.localizedDescription)
            }
        }
    }

    func cancel() {
        job?.cancel()
    }

    func show(_ result: ArrowResult) {
        var colors: [ArrowTask: [SIMD3<Float>]] = [:]
        for group in result.groups { colors[group.task] = result.colors(group) }
        self.colors = colors
        self.result = result
        recommendedKeep = result.recommendedKeep
        display.keep = recommendedKeep
        showSeeds = false
    }

    private func readTracks() {
        guard let result, let group = result.group(.track) else {
            tracks = nil
            statistics = nil
            return
        }
        let tracks = TrackSet(result, group, fusion: display.fusion, camera: camera)
        self.tracks = tracks
        statistics = TrackStatistics(tracks)
    }

    private func show(_ event: ArrowEvent) {
        switch event {
        case let .downloading(f): phase = .downloading(f)
        case .loading: phase = .loading
        case .encoding: phase = .encoding
        case let .decoding(f): phase = .decoding(f)
        }
    }

    // MARK: Saving

    func save(to url: URL) {
        guard let result else { return }
        do {
            try ResultArchive.write(result, sources: sources, to: url)
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    /// Writes what is shown: the confidence filter applies.
    func exportUSD(to url: URL) {
        guard let result else { return }
        do {
            try USDExport.write(result, to: url, options: .init(keep: display.keep, pointWidth: cameraDepth / 40))
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    func openResult(_ url: URL) {
        job?.cancel()
        do {
            let (result, sources) = try ResultArchive.read(url)
            self.sources = sources.isEmpty ? [url] : sources
            sourceFrames = nil
            frames = result.frames
            sampling = .spread
            frameCount = result.frames.timesteps
            frame = 0
            camera = 0
            show(result)
            phase = .idle
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    func togglePlay() {
        isPlaying.toggle()
        player?.cancel()
        guard isPlaying else { return }
        player = Task {
            while !Task.isCancelled, self.isPlaying, self.frameTotal > 1 {
                try? await Task.sleep(for: .milliseconds(160))
                self.frame = (self.frame + 1) % self.frameTotal
            }
        }
    }

    /// Everything the viewport draws at the current timestep.
    var geometry: Geometry {
        guard let result else { return Geometry() }
        var geometry = Geometry()
        if display.showCloud, let colors = colors[.reconstruct] {
            geometry.append(SceneBuilder.cloud(result, colors: colors, settings: display))
        }
        if display.showDepth, let colors = colors[.depth] {
            geometry.append(SceneBuilder.depth(result, frame: displayFrame, colors: colors, settings: display))
        }
        if display.showTracks, let colors = colors[.track], let tracks, let statistics {
            geometry.append(SceneBuilder.tracks(
                result, tracks, timestep: frame, colors: colors, statistics: statistics, diagonal: sceneDiagonal, settings: display))
        }
        if display.showCameras {
            geometry.append(SceneBuilder.cameras(result, frame: displayFrame, depth: cameraDepth, settings: display))
        }
        return geometry
    }

    /// Still and moving track counts, when they are split.
    var motionCounts: (still: Int, moving: Int)? {
        guard display.separateMotion, let statistics else { return nil }
        let still = SceneBuilder.stillTracks(statistics, diagonal: sceneDiagonal, settings: display)
        let kept = statistics.kept(display.keep)
        let stillCount = zip(still, kept).filter { $0 && $1 }.count
        return (stillCount, kept.filter { $0 }.count - stillCount)
    }

    /// A frustum's depth: 12 % of half the scene's diagonal (upstream's default size).
    var cameraDepth: Float {
        guard result != nil else { return 0.1 }
        return max(sceneDiagonal / 2 * 0.12, 0.005)
    }
}

import ArrowKit
import SwiftUI

/// Input, run, display and output controls.
struct Sidebar: View {
    @Bindable var session: Session
    @State private var inspecting = false
    @State private var reading: Task<Void, Never>?

    var body: some View {
        Form {
            Section("Input") {
                if session.sources.isEmpty {
                    LabeledContent("Source") { Text("None").foregroundStyle(.secondary) }
                }
                ForEach(Array(session.sources.enumerated()), id: \.offset) { index, url in
                    LabeledContent(session.sources.count > 1 ? "Camera \(index + 1)" : "Source") {
                        HStack(spacing: 4) {
                            Text(url.lastPathComponent).lineLimit(1).truncationMode(.middle)
                            if session.sources.count > 1 {
                                Button("Remove", systemImage: "minus.circle") { session.removeCamera(at: index) }
                                    .labelStyle(.iconOnly).buttonStyle(.borderless)
                            }
                        }
                    }
                }
                HStack {
                    Button("Open…", systemImage: "folder", action: QuiverApp.openPanel(session))
                    Button("Add Camera…", systemImage: "plus", action: QuiverApp.addCameraPanel(session))
                        .disabled(session.sources.isEmpty)
                        .help("Another synchronized video of the same scene: multi-view tracking and reconstruction")
                }
                sampling
                Picker("Long side", selection: $session.maxSide) {
                    ForEach([252, 378, 504], id: \.self) { Text("\($0) px").tag($0) }
                }
                .onChange(of: session.maxSide) { session.readFrames() }
                if let frames = session.frames {
                    LabeledContent("Model size", value: "\(frames.width) × \(frames.height)")
                }
            }

            Section("Tasks") {
                task(.track, "3D tracking", detail: "Seeds in the current frame, followed through every frame")
                if session.tasks.contains(.track) { seeds }
                task(.reconstruct, "3D reconstruction", detail: "Every frame's pixels in the first frame's camera")
                if session.tasks.contains(.reconstruct) {
                    Stepper("Pixel spacing \(session.settings.reconstructionStride) px", value: $session.settings.reconstructionStride, in: 1 ... 16)
                }
                task(.pose, "Camera poses", detail: "A rigid fit per frame, and its intrinsics")
                task(.depth, "Video depth", detail: "Each frame in its own camera")
                Toggle(isOn: $session.skipSky) {
                    Text("Skip sky")
                    Text("Sky pixels are not asked about, as upstream does; the segmenter (\(SkySegmenter.downloadSize)) runs on the Neural Engine")
                }
                if session.tasks.contains(.depth) {
                    Stepper("Pixel spacing \(session.settings.depthStride) px", value: $session.settings.depthStride, in: 1 ... 16)
                }
                HStack {
                    if session.phase.isBusy {
                        ProgressView().controlSize(.small)
                        Button("Cancel", role: .cancel, action: session.cancel)
                    } else {
                        Button("Run", systemImage: "play.fill", action: session.run)
                            .keyboardShortcut(.return, modifiers: .command)
                            .disabled(!session.canRun)
                    }
                }
                Text(session.phase.label)
                    .font(.caption)
                    .foregroundStyle(phaseColor)
                    .fixedSize(horizontal: false, vertical: true)
                if let result = session.result {
                    Text(String(format: "%@ queries · encoded in %.1f s · answered in %.1f s",
                                result.queries.count.formatted(), result.encodeSeconds, result.decodeSeconds))
                        .font(.caption).foregroundStyle(.secondary)
                    if let sky = result.sky {
                        Text(String(format: "Sky: %.1f %% of pixels, their queries skipped", sky.fraction * 100))
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
            }

            Section("Display") {
                Toggle("Point cloud", isOn: $session.display.showCloud)
                Toggle("Tracks", isOn: $session.display.showTracks)
                Toggle("Cameras", isOn: $session.display.showCameras)
                Toggle("Depth of this frame", isOn: $session.display.showDepth)
                Picker("Point colour", selection: $session.display.cloudColoring) {
                    ForEach(DisplaySettings.PointColoring.allCases) { Text($0.rawValue).tag($0) }
                }
                Picker("Track colour", selection: $session.display.trackColoring) {
                    ForEach(DisplaySettings.TrailColoring.allCases) { Text($0.rawValue).tag($0) }
                }
                if session.views > 1 {
                    Picker("3D tracks from", selection: $session.display.fusion) {
                        ForEach(TrackSet.Fusion.allCases) { Text($0.rawValue).tag($0) }
                    }
                    .help(session.display.fusion == .median
                          ? "The median of the cameras that see the point at each timestep"
                          : "Each track's predictions from the camera it was seeded in")
                }
                confidence
                LabeledContent("Point size") {
                    Slider(value: $session.display.pointSize, in: 1 ... 12).labelsHidden()
                }
                Stepper("Trail \(session.display.trail) frames", value: $session.display.trail, in: 0 ... 48)
            }

            Section("Visibility and motion") {
                LabeledContent("Visible at") {
                    HStack {
                        Slider(value: $session.display.visibleAt, in: 0 ... 1).labelsHidden()
                        Text(String(format: "%.2f", session.display.visibleAt)).monospacedDigit().font(.caption)
                    }
                }
                .help("A track point counts as seen when the model's visibility probability reaches this")
                Picker("Hidden points", selection: $session.display.hidden) {
                    ForEach(DisplaySettings.Hidden.allCases) { Text($0.rawValue).tag($0) }
                }
                Toggle("Separate still and moving tracks", isOn: $session.display.separateMotion)
                if session.display.separateMotion {
                    LabeledContent("Still within") {
                        HStack {
                            Slider(value: $session.display.stillPercent, in: 0 ... 20, step: 0.05).labelsHidden()
                            Text(String(format: "%.2f %%", session.display.stillPercent)).monospacedDigit().font(.caption)
                        }
                    }
                    .help("A track is still when it strays at most this share of the scene's diagonal from its mean")
                    Toggle("Collapse still tracks", isOn: $session.display.collapseStill)
                    Toggle("Tint still blue, moving orange", isOn: $session.display.tintMotion)
                    if let counts = session.motionCounts {
                        Text("\(counts.still.formatted()) still · \(counts.moving.formatted()) moving")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
            }

            Section("Cameras") {
                Toggle("Path", isOn: $session.display.cameraPath)
                LabeledContent("Image plane") {
                    Slider(value: $session.display.imagePlane, in: 0 ... 1).labelsHidden()
                }
                .help("The current frame drawn on its camera's frustum; all the way left hides it")
            }

            Section("Output") {
                Button("Save Result…", systemImage: "square.and.arrow.down", action: QuiverApp.savePanel(session))
                    .disabled(session.result == nil || session.streaming)
                Button("Export USD…", systemImage: "cube", action: QuiverApp.exportPanel(session))
                    .disabled(session.result == nil || session.streaming)
                    .help("Points, tracks over time, trajectories and cameras as a .usda stage, filtered as shown")
                Button("Clear Scene", systemImage: "xmark.circle", role: .destructive, action: session.clear)
                    .disabled(session.result == nil)
                    .help("Empty the 3D view, stopping a run in progress; the frames and seeds stay")
            }
        }
        .formStyle(.grouped)
        .sheet(isPresented: $inspecting) { QueryInspector(session: session) }
    }

    /// Frames spread over the input, or every step-th frame of a range, up to
    /// `Session.frameLimit` timesteps.
    @ViewBuilder private var sampling: some View {
        Picker("Frames", selection: $session.sampling) {
            ForEach(Session.Sampling.allCases) { Text($0.rawValue).tag($0) }
        }
        .pickerStyle(.segmented)
        .onChange(of: session.sampling) { session.readFrames() }
        let last = max((session.sourceFrames ?? 1) - 1, 0)
        switch session.sampling {
        case .spread:
            Stepper("\(session.frameCount) frames, evenly spread", value: $session.frameCount, in: 2 ... Session.frameLimit,
                    step: session.frameCount >= 48 ? 8 : session.frameCount >= 24 ? 4 : 1)
                .onChange(of: session.frameCount) { readSoon() }
        case .range:
            Stepper("From frame \(session.rangeStart + 1)", value: $session.rangeStart, in: 0 ... last)
                .onChange(of: session.rangeStart) { readSoon() }
            Stepper("To frame \(session.rangeEnd + 1)", value: $session.rangeEnd, in: 0 ... last)
                .onChange(of: session.rangeEnd) { readSoon() }
            Stepper(session.rangeStep == 1 ? "Every frame" : "Every \(session.rangeStep) frames",
                    value: $session.rangeStep, in: 1 ... 60)
                .onChange(of: session.rangeStep) { readSoon() }
        }
        if let total = session.sourceFrames, let count = session.selectedCount {
            let views = max(session.sources.count, 1)
            Text("\(count) of \(total) frames\(views > 1 ? " × \(views) cameras = \(count * views) images" : "")"
                 + (session.sampling == .range && count == Session.frameLimit ? ", the first \(Session.frameLimit)" : ""))
                .font(.caption).foregroundStyle(.secondary)
            if count * views > 48 {
                Text("Encoding attends across every frame and each query reads all of them: long sets take far more memory and time. A smaller long side helps.")
                    .font(.caption).foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    /// Re-reads after a stepper settles, not on every click of it.
    private func readSoon() {
        reading?.cancel()
        reading = Task {
            try? await Task.sleep(for: .milliseconds(400))
            guard !Task.isCancelled else { return }
            session.readFrames()
        }
    }

    @ViewBuilder private var seeds: some View {
        Picker("Seeds", selection: $session.seedTool) {
            ForEach(Session.SeedTool.allCases) { Text($0.rawValue).tag($0) }
        }
        .pickerStyle(.segmented)
        .help("Region: drag on the frame to bound the grid. Points: click to add seeds.")
        Stepper(session.settings.trackStride == 0 ? "Grid off" : "Grid every \(session.settings.trackStride) px",
                value: $session.settings.trackStride, in: 0 ... 64, step: 2)
        HStack {
            Button("Full Frame") { session.region = Session.fullFrame }
                .disabled(session.region == Session.fullFrame)
            Button("Clear \(session.picks.count)") { session.picks = [] }
                .disabled(session.picks.isEmpty)
            Spacer()
            Button("Inspect…") { inspecting = true }
                .disabled(session.frames == nil)
        }
        .controlSize(.small)
        if let seedFrame = session.seedFrame {
            HStack {
                Text("Tracks start on \(session.frameName(seedFrame))").font(.caption)
                Spacer()
                Button("Go There", action: session.showTrackSource)
                    .controlSize(.small)
                    .disabled(session.displayFrame == seedFrame)
            }
            .help("The region and clicked points mark what is under them on this frame. Editing seeds on another frame starts them over.")
        }
        Toggle("Show seeds on the frame", isOn: $session.showSeeds)
            .disabled(session.result?.group(.track) == nil)
        Text(SeedSummary.text(session)).font(.caption).foregroundStyle(.secondary)
    }

    private var confidence: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Confidence")
                Spacer()
                Text(String(format: "keep %.1f %%", session.display.keep * 100)).monospacedDigit().font(.caption)
            }
            Slider(value: $session.display.keep, in: 0.01 ... 1).labelsHidden()
            HStack {
                Spacer()
                Button(String(format: "Recommended (%.1f %%)", session.recommendedKeep * 100)) {
                    session.display.keep = session.recommendedKeep
                }
                .controlSize(.small)
                .disabled(session.result == nil || session.display.keep == session.recommendedKeep)
                .help("Upstream's heuristic: drop the low tail of log confidence past 3 robust deviations")
            }
        }
    }

    private var phaseColor: Color {
        if case .failed = session.phase { return .red }
        return .secondary
    }

    private func task(_ task: ArrowTask, _ title: String, detail: String) -> some View {
        Toggle(isOn: Binding(
            get: { session.tasks.contains(task) },
            set: { on in if on { session.tasks.insert(task) } else { session.tasks.remove(task) } }
        )) {
            Text(title)
            Text(detail)
        }
    }
}

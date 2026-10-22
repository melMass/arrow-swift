import AppKit
import ArrowKit
import SwiftUI
import UniformTypeIdentifiers

extension UTType {
    /// A saved result (`ResultArchive`), declared in macos-plist-extras.xml.
    static let quiverResult = UTType(exportedAs: "com.mtb.quiver.result", conformingTo: .data)
    static let usda = UTType(filenameExtension: "usda") ?? .data
}

public struct QuiverApp: App {
    @NSApplicationDelegateAdaptor private var delegate: AppDelegate
    @State private var session = Session()

    public init() {}

    public var body: some Scene {
        Window("Quiver", id: "main") {
            ContentView(session: session)
                .frame(minWidth: 1000, minHeight: 640)
                .task {
                    delegate.session = session
                    Control.startIfRequested(session: session)
                }
        }
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("Open…", action: Self.openPanel(session)).keyboardShortcut("o")
            }
            CommandGroup(replacing: .saveItem) {
                Button("Save Result…", action: Self.savePanel(session)).keyboardShortcut("s")
                    .disabled(session.result == nil || session.streaming)
                Button("Export USD…", action: Self.exportPanel(session)).keyboardShortcut("e")
                    .disabled(session.result == nil || session.streaming)
            }
            CommandMenu("Playback") {
                Button(session.isPlaying ? "Pause" : "Play", action: session.togglePlay)
                    .keyboardShortcut(.space, modifiers: [])
                Button("Next Frame") { session.frame = min(session.frame + 1, max(session.frameTotal - 1, 0)) }
                    .keyboardShortcut(.rightArrow, modifiers: [])
                Button("Previous Frame") { session.frame = max(session.frame - 1, 0) }
                    .keyboardShortcut(.leftArrow, modifiers: [])
            }
        }
    }

    /// A panel for a video, a folder of images, an image, or a saved result.
    @MainActor static func openPanel(_ session: Session) -> () -> Void {
        {
            let panel = NSOpenPanel()
            panel.canChooseDirectories = true
            panel.allowedContentTypes = [.movie, .image, .folder, .quiverResult]
            panel.message = "A video, a folder of images, an image, or a saved result"
            if panel.runModal() == .OK, let url = panel.url { session.open(url) }
        }
    }

    /// A panel for another camera's video or image folder of the same scene.
    @MainActor static func addCameraPanel(_ session: Session) -> () -> Void {
        {
            let panel = NSOpenPanel()
            panel.canChooseDirectories = true
            panel.allowedContentTypes = [.movie, .folder]
            panel.message = "Another camera's video or image folder, synchronized frame for frame with the others"
            if panel.runModal() == .OK, let url = panel.url { session.addCamera(url) }
        }
    }

    @MainActor static func savePanel(_ session: Session) -> () -> Void {
        {
            let panel = NSSavePanel()
            panel.allowedContentTypes = [.quiverResult]
            panel.nameFieldStringValue = (session.source?.deletingPathExtension().lastPathComponent ?? "Result") + ".quiver"
            panel.message = "Frames, queries and every prediction, to reopen without the model"
            if panel.runModal() == .OK, let url = panel.url { session.save(to: url) }
        }
    }

    @MainActor static func exportPanel(_ session: Session) -> () -> Void {
        {
            let panel = NSSavePanel()
            panel.allowedContentTypes = [.usda]
            panel.nameFieldStringValue = (session.source?.deletingPathExtension().lastPathComponent ?? "Result") + ".usda"
            panel.message = "Points, tracks, trajectories and cameras as shown (the confidence filter applies)"
            if panel.runModal() == .OK, let url = panel.url { session.exportUSD(to: url) }
        }
    }
}

/// Opens files handed to the app (Finder, `open -a`), once the window's session exists.
final class AppDelegate: NSObject, NSApplicationDelegate {
    @MainActor var session: Session? {
        didSet { if let session { pending.forEach(session.open); pending = [] } }
    }
    @MainActor private var pending: [URL] = []

    @MainActor func application(_ application: NSApplication, open urls: [URL]) {
        guard let url = urls.first else { return }
        if let session { session.open(url) } else { pending = [url] }
    }
}

struct ContentView: View {
    @Bindable var session: Session

    enum Panel: String, CaseIterable, Identifiable {
        case frame = "Frame", depth = "Depth"
        var id: Self { self }
    }

    @State private var panel: Panel = .frame

    var body: some View {
        HSplitView {
            Sidebar(session: session)
                .frame(minWidth: 280, idealWidth: 300, maxWidth: 380)
            VSplitView {
                ZStack(alignment: .topLeading) {
                    SceneViewport(geometry: session.geometry, pointSize: session.display.pointSize, sceneID: session.sceneID)
                    if session.result == nil {
                        Text(session.frames == nil ? "Open a video, then Run." : "Run to see the scene in 3D.")
                            .font(.callout).foregroundStyle(.secondary)
                            .padding(10)
                    }
                }
                .frame(minHeight: 300)
                VStack(spacing: 6) {
                    if session.result?.group(.depth) != nil {
                        Picker("Panel", selection: $panel) {
                            ForEach(Panel.allCases) { Text($0.rawValue).tag($0) }
                        }
                        .pickerStyle(.segmented).labelsHidden().frame(width: 180)
                    }
                    Group {
                        if panel == .depth, session.result?.group(.depth) != nil {
                            DepthView(session: session)
                        } else {
                            FramePreview(session: session)
                        }
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    Transport(session: session)
                }
                .padding(8)
                .frame(minHeight: 180, idealHeight: 260)
            }
        }
        .dropDestination(for: URL.self) { urls, _ in
            guard let url = urls.first else { return false }
            session.open(url)
            return true
        }
    }
}

/// Play, and a scrubber over the frames.
struct Transport: View {
    @Bindable var session: Session

    var body: some View {
        HStack(spacing: 10) {
            Button(session.isPlaying ? "Pause" : "Play", systemImage: session.isPlaying ? "pause.fill" : "play.fill",
                   action: session.togglePlay)
                .labelStyle(.iconOnly).buttonStyle(.borderless)
                .disabled(session.frameTotal < 2)
            if session.frameTotal > 1 {
                Slider(value: Binding(get: { Double(session.frame) }, set: { session.frame = Int($0.rounded()) }),
                       in: 0 ... Double(session.frameTotal - 1), step: 1)
            } else {
                Spacer()
            }
            if session.views > 1 {
                Picker("Camera", selection: $session.camera) {
                    ForEach(0 ..< session.views, id: \.self) { Text("Camera \($0 + 1)").tag($0) }
                }
                .labelsHidden().fixedSize()
                .help("The camera whose frame is shown below; the scene holds them all")
            }
            Text(label)
                .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                .frame(minWidth: 110, alignment: .trailing)
        }
    }

    private var label: String {
        guard let frames = session.frames, session.displayFrame < frames.times.count else { return "–" }
        return String(format: "frame %d of %d · %.2f s", session.frame + 1, frames.timesteps, frames.times[session.displayFrame])
    }
}

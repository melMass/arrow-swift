import ArrowKit
import SwiftUI

/// The current input frame as the model saw it. Before a run, and on request after one,
/// it shows the track seeds and takes a drawn region or clicked points; with tracks, each
/// kept track sits where the model puts it in this frame (its predicted uv) with a trail.
struct FramePreview: View {
    @Bindable var session: Session
    /// Seed editing: a region drag or clicks, as the session's tool says.
    var editable = true
    @State private var dragStart: CGPoint?
    @State private var dragRect: CGRect?

    var body: some View {
        if let frames = session.frames, session.displayFrame < frames.images.count {
            Image(decorative: frames.images[session.displayFrame], scale: 1)
                .resizable()
                .interpolation(.none)
                .aspectRatio(contentMode: .fit)
                .overlay {
                    GeometryReader { proxy in
                        Canvas { context, size in
                            if showsSeeds { drawSeeds(in: &context, size: size) } else { drawTracks(in: &context, size: size) }
                            if let dragRect {
                                context.stroke(Path(dragRect), with: .color(.white), style: StrokeStyle(lineWidth: 1, dash: [4, 3]))
                            }
                        }
                        .contentShape(Rectangle())
                        .gesture(seedGesture(size: proxy.size), including: editing ? .all : .none)
                    }
                }
        } else {
            ContentUnavailableView("No frames", systemImage: "film.stack", description: Text("Open a video or a folder of images."))
        }
    }

    private var editing: Bool {
        editable && session.tasks.contains(.track) && !session.settings.trackAllPixels && !session.phase.isBusy
    }
    private var showsSeeds: Bool { session.result?.group(.track) == nil || session.showSeeds }

    private func seedGesture(size: CGSize) -> some Gesture {
        DragGesture(minimumDistance: 0)
            .onChanged { value in
                guard session.seedTool == .region else { return }
                let start = dragStart ?? value.startLocation
                dragStart = start
                dragRect = CGRect(x: min(start.x, value.location.x), y: min(start.y, value.location.y),
                                  width: abs(value.location.x - start.x), height: abs(value.location.y - start.y))
            }
            .onEnded { value in
                defer { dragStart = nil; dragRect = nil }
                func fraction(_ p: CGPoint) -> SIMD2<Float> {
                    SIMD2(Float(min(max(p.x / size.width, 0), 1)), Float(min(max(p.y / size.height, 0), 1)))
                }
                switch session.seedTool {
                case .points:
                    session.editSeeds { $0.picks.append(fraction(value.location)) }
                case .region:
                    let a = fraction(value.startLocation), b = fraction(value.location)
                    // Too small to mean a region: a click.
                    guard abs(a.x - b.x) > 0.02, abs(a.y - b.y) > 0.02 else { return }
                    session.editSeeds { $0.region = SIMD4(min(a.x, b.x), min(a.y, b.y), max(a.x, b.x), max(a.y, b.y)) }
                }
                session.showSeeds = true
            }
    }

    /// The region's outline, its grid as dots (a wash when they would crowd closer than
    /// 3 px), and the clicked points. With several cameras and no drawn region, every
    /// camera's frame at the seeds' timestep gets the grid; picks stay on their frame.
    private func drawSeeds(in context: inout GraphicsContext, size: CGSize) {
        guard let frames = session.frames, session.tasks.contains(.track) else { return }
        let onSeedFrame = session.trackSource == session.displayFrame
        if session.settings.trackAllPixels {
            // Only the start frame's grid is known before the run.
            let label = onSeedFrame ? "Every \(session.settings.allPixelsStride) px, then what other frames add" : "Starts on \(session.frameName(session.trackSource))"
            if onSeedFrame { context.fill(Path(CGRect(origin: .zero, size: size)), with: .color(.accentColor.opacity(0.2))) }
            context.draw(Text(label).font(.caption).foregroundStyle(.white), at: CGPoint(x: 8, y: 8), anchor: .topLeading)
            return
        }
        let gridOnEveryCamera = frames.views > 1 && session.region == Session.fullFrame && session.settings.trackStride > 0
        guard onSeedFrame || (gridOnEveryCamera && session.trackSource / frames.views == session.frame) else {
            context.draw(Text("Seeds on \(session.frameName(session.trackSource))").font(.caption).foregroundStyle(.white),
                         at: CGPoint(x: 8, y: 8), anchor: .topLeading)
            return
        }
        let r = session.region
        let rect = CGRect(x: CGFloat(r.x) * size.width, y: CGFloat(r.y) * size.height,
                          width: CGFloat(r.z - r.x) * size.width, height: CGFloat(r.w - r.y) * size.height)
        let scale = size.width / CGFloat(frames.width)
        if session.settings.trackStride > 0 {
            if CGFloat(session.settings.trackStride) * scale < 3 {
                context.fill(Path(rect), with: .color(.accentColor.opacity(0.2)))
            } else {
                let grid = QueryGrid.spread(width: frames.width, height: frames.height,
                                            stride: session.settings.trackStride, region: r)
                for p in grid {
                    let c = CGPoint(x: (CGFloat(p.x) + 0.5) * scale, y: (CGFloat(p.y) + 0.5) * scale)
                    let dot = Path(ellipseIn: CGRect(x: c.x - 1.5, y: c.y - 1.5, width: 3, height: 3))
                    context.fill(dot, with: .color(.white))
                    context.stroke(dot, with: .color(.black.opacity(0.7)), lineWidth: 0.5)
                }
            }
        }
        context.stroke(Path(rect), with: .color(.white), lineWidth: 1)
        let radius = max(size.width, size.height) / 110
        for p in session.picks where onSeedFrame {
            let c = CGPoint(x: CGFloat(p.x) * size.width, y: CGFloat(p.y) * size.height)
            context.fill(Path(ellipseIn: CGRect(x: c.x - radius, y: c.y - radius, width: 2 * radius, height: 2 * radius)),
                         with: .color(.accentColor))
        }
    }

    private func drawTracks(in context: inout GraphicsContext, size: CGSize) {
        guard session.display.showTracks, let result = session.result, let tracks = session.tracks,
              let colors = session.colors[.track], let statistics = session.statistics,
              session.frame < tracks.timesteps else { return }
        let frame = session.frame
        let kept = statistics.kept(session.display.keep)
        let seen = session.display.seenLogit
        func point(_ i: Int) -> CGPoint {
            CGPoint(x: CGFloat(tracks.uv[i].x) * size.width, y: CGFloat(tracks.uv[i].y) * size.height)
        }
        let first = max(0, frame - session.display.trail)
        for seed in 0 ..< tracks.seeds where kept[seed] {
            let i = tracks.index(frame, seed)
            // Not answered yet, while a run streams in.
            guard tracks.uv[i].x.isFinite else { continue }
            // Seen in this camera's frame, whatever the others see.
            let isSeen = tracks.cameraVisibility[i] >= seen
            if !isSeen && session.display.hidden == .hide { continue }
            let c = SceneBuilder.trackColor(result, tracks, seed: seed, colors: colors, coloring: session.display.trackColoring, timestep: frame)
            let color = Color(red: Double(c.x), green: Double(c.y), blue: Double(c.z))
                .opacity(isSeen || session.display.hidden == .show ? 1 : 0.35)
            if first < frame {
                var trail = Path()
                trail.move(to: point(tracks.index(first, seed)))
                for t in first + 1 ... frame { trail.addLine(to: point(tracks.index(t, seed))) }
                context.stroke(trail, with: .color(color.opacity(0.6)), lineWidth: 1)
            }
            let center = point(i)
            context.fill(Path(ellipseIn: CGRect(x: center.x - 2, y: center.y - 2, width: 4, height: 4)), with: .color(color))
        }
    }
}

/// The start frame enlarged, to check the exact query pixels (upstream's "Inspect
/// queries"): zoom 25–1600 %, the grid as dots, picks and region editable.
struct QueryInspector: View {
    @Bindable var session: Session
    @Environment(\.dismiss) private var dismiss
    @State private var zoom: Double = 4

    var body: some View {
        VStack(spacing: 10) {
            HStack {
                Text("Inspect queries").font(.headline)
                Spacer()
                Picker("Tool", selection: $session.seedTool) {
                    ForEach(Session.SeedTool.allCases) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented).labelsHidden().frame(width: 160)
                Button("Zoom Out", systemImage: "minus.magnifyingglass") { zoom = max(0.25, zoom / 2) }
                    .labelStyle(.iconOnly)
                Text("\(Int(zoom * 100)) %").monospacedDigit().frame(width: 60)
                Button("Zoom In", systemImage: "plus.magnifyingglass") { zoom = min(16, zoom * 2) }
                    .labelStyle(.iconOnly)
                Button("Done") { dismiss() }.keyboardShortcut(.defaultAction)
            }
            if let frames = session.frames {
                ScrollView([.horizontal, .vertical]) {
                    FramePreview(session: session)
                        .frame(width: CGFloat(frames.width) * zoom, height: CGFloat(frames.height) * zoom)
                }
                .background(.black)
            }
            Text(SeedSummary.text(session)).font(.caption).foregroundStyle(.secondary)
        }
        .padding()
        .frame(minWidth: 760, minHeight: 560)
        .onAppear {
            session.showSeeds = true
            session.showTrackSource()
        }
    }
}

enum SeedSummary {
    /// "N × M = K grid points · P picked points".
    @MainActor static func text(_ session: Session) -> String {
        guard let frames = session.frames else { return "" }
        if session.settings.trackAllPixels {
            let lattice = PixelCoverage(width: frames.width, height: frames.height, frames: 1, stride: session.settings.allPixelsStride).latticeCount
            return "\(lattice.formatted()) seeds on \(session.frameName(session.trackSource)), then each other frame's uncovered pixels"
        }
        let r = session.region, stride = session.settings.trackStride
        let columns = QueryGrid.spreadAxis(frames.width, low: r.x, high: r.z, stride: stride).count
        let rows = QueryGrid.spreadAxis(frames.height, low: r.y, high: r.w, stride: stride).count
        let grid = stride > 0 ? "\(columns) × \(rows) = \((columns * rows).formatted()) grid points" : "Grid off"
        let total = frames.views > 1 ? " · \(session.seedTotal.formatted()) over \(frames.views) cameras" : ""
        return "\(grid) · \(session.picks.count) picked · \(session.seeds.count.formatted()) seeds, overlaps counted once\(total)"
    }
}

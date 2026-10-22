#if DEBUG
import SwiftUI

/// Named views on the synthetic scene of `PreviewFixtures`, for a host that renders each
/// in a real window to check them by eye. Debug builds only.
public enum Gallery {
    public typealias Emit = (_ event: String, _ payload: Any?) -> Void
    public typealias Sheet = (name: String, make: (_ params: [String: Any], _ emit: @escaping Emit) -> AnyView)

    @MainActor public static var sheets: [Sheet] {
        [
            ("window", { _, _ in AnyView(ContentView(session: .fixture()).frame(width: 1200, height: 780)) }),
            ("window-empty", { _, _ in AnyView(ContentView(session: Session()).frame(width: 1100, height: 700)) }),
            ("window-frames", { _, _ in AnyView(ContentView(session: seeded()).frame(width: 1100, height: 700)) }),
            ("window-motion", { _, _ in
                let session = Session.fixture()
                session.display.separateMotion = true
                session.display.cloudColoring = .estimatedNormals
                return AnyView(ContentView(session: session).frame(width: 1200, height: 780))
            }),
            ("sidebar", { _, _ in AnyView(Sidebar(session: .fixture()).frame(width: 320, height: 1500)) }),
            ("sidebar-running", { _, _ in AnyView(Sidebar(session: .fixture(phase: .decoding(0.42))).frame(width: 320, height: 900)) }),
            ("sidebar-downloading", { _, _ in AnyView(Sidebar(session: .fixture(phase: .downloading(0.17), result: false)).frame(width: 320, height: 900)) }),
            ("sidebar-failed", { _, _ in
                AnyView(Sidebar(session: .fixture(phase: .failed("Could not read /Movies/broken.mov."), result: false)).frame(width: 320, height: 900))
            }),
            ("preview", { _, _ in AnyView(FramePreview(session: .fixture()).frame(width: 672, height: 392)) }),
            ("seeds", { _, _ in AnyView(FramePreview(session: seeded()).frame(width: 672, height: 392)) }),
            ("inspector", { _, _ in AnyView(QueryInspector(session: seeded()).frame(width: 900, height: 640)) }),
            ("depth", { _, _ in AnyView(DepthView(session: .fixture()).frame(width: 700, height: 440)) }),
        ]
    }

    /// Frames read, nothing run: a region and three clicked seeds.
    @MainActor static func seeded() -> Session {
        let session = Session.fixture(frame: 0, result: false)
        session.region = SIMD4(0.3, 0.15, 0.75, 0.85)
        session.picks = [SIMD2(0.1, 0.2), SIMD2(0.12, 0.8), SIMD2(0.9, 0.5)]
        session.settings.trackStride = 16
        return session
    }
}
#endif

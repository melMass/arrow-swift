#if CONTROL_SOCKET
import AppKit
import ArrowKit
import MCPKit

/// The app's control socket (`~/.cache/mcpkit/quiver.sock`), for an MCP client driving
/// the running app: MCPKit's window tools, plus opening a clip, running the model as the
/// Run button does, and moving the playhead. Listens only when launched with
/// `QUIVER_CONTROL=1`, and is built only when MCPKit is linked (see Package.swift).
@MainActor
enum Control {
    static func startIfRequested(session: Session, name: String = "quiver") {
        ControlServer.startIfRequested(name: name, variable: "QUIVER_CONTROL") { tools(session) }
    }

    private static func tools(_ session: Session) -> [Tool] {
        [
            Tool("open", "Open a video, a folder of images or an image, and wait for its frames.", [
                .string("path", "The file or folder"),
                .integer("frames", "Frames to take across it (default: the sidebar's)", required: false),
                .integer("side", "The model's long side in pixels, a multiple of 14 (default: the sidebar's)", required: false),
            ]) { arguments in
                if let frames = arguments["frames"] as? Int { session.frameCount = frames }
                if let side = arguments["side"] as? Int { session.maxSide = side }
                session.open(URL(filePath: (try arguments.string("path") as NSString).expandingTildeInPath))
                try await settle(session)
                return state(session)
            },
            Tool("add_camera", "Add another synchronized camera's video or image folder, and wait for its frames.", [
                .string("path", "The file or folder"),
            ]) { arguments in
                session.addCamera(URL(filePath: (try arguments.string("path") as NSString).expandingTildeInPath))
                try await settle(session)
                return state(session)
            },
            Tool("run", "Run the model as the Run button does and wait for the result.", [
                .string("tasks", "Comma-separated: track, reconstruct, pose, depth (default: the sidebar's)", required: false),
            ]) { arguments in
                if let names = arguments["tasks"] as? String {
                    session.tasks = Set(names.split(separator: ",").compactMap {
                        ArrowTask(rawValue: $0.trimmingCharacters(in: .whitespaces))
                    })
                }
                session.run()
                try await settle(session)
                return state(session)
            },
            Tool("frame", "Move the playhead to a timestep (0-based), and pick the camera shown.", [
                .integer("index", "The timestep", required: true),
                .integer("camera", "The camera, with several (0-based)", required: false),
            ]) { arguments in
                guard let index = arguments["index"] as? Int else { throw ToolError("index is required") }
                session.frame = min(max(index, 0), max(session.frameTotal - 1, 0))
                if let camera = arguments["camera"] as? Int { session.camera = min(max(camera, 0), session.views - 1) }
                return state(session)
            },
            Tool("state", "The input, the phase, and the last result's size and timings.") { _ in
                state(session)
            },
        ]
    }

    /// Waits until the session is no longer reading or running, failing with its message.
    private static func settle(_ session: Session) async throws {
        try await Task.sleep(for: .milliseconds(50))
        while session.phase.isBusy {
            try await Task.sleep(for: .milliseconds(200))
        }
        if case let .failed(message) = session.phase { throw ToolError(message) }
    }

    private static func state(_ session: Session) -> [String: Any] {
        var state: [String: Any] = [
            "sources": session.sources.map(\.path),
            "phase": session.phase.label,
            "frame": session.frame,
            "frames": session.frameTotal,
            "camera": session.camera,
            "cameras": session.views,
        ]
        if let frames = session.frames { state["size"] = "\(frames.width)x\(frames.height)" }
        if let result = session.result {
            state["queries"] = result.queries.count
            state["encodeSeconds"] = result.encodeSeconds
            state["decodeSeconds"] = result.decodeSeconds
            state["groups"] = result.groups.map { "\($0.task.rawValue): \($0.count)" }
            state["validCameras"] = result.cameras?.filter(\.valid).count ?? 0
        }
        return state
    }
}
#else
@MainActor
enum Control {
    static func startIfRequested(session: Session, name: String = "quiver") {}
}
#endif

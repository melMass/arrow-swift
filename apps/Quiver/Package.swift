// swift-tools-version: 6.0
import Foundation
import PackageDescription

/// Optional: an MCPKit checkout (`MCPKIT_DIR`, or `$SWIFTKIT_DIR/packages/MCPKit`) adds a
/// control socket an MCP client can drive the app through (`Support/Control.swift`,
/// `QUIVER_CONTROL=1`). Without one the app builds without it.
let mcpkit = Context.environment["MCPKIT_DIR"]
    ?? Context.environment["SWIFTKIT_DIR"].map { "\($0)/packages/MCPKit" }
let linksMCPKit = mcpkit.map { FileManager.default.fileExists(atPath: "\($0)/Package.swift") } ?? false

let package = Package(
    name: "Quiver",
    platforms: [.macOS(.v15)],
    products: [
        .executable(name: "Quiver", targets: ["Quiver"]),
        .library(name: "QuiverKit", targets: ["QuiverKit"]),
    ],
    dependencies: [
        .package(path: "../../packages/ArrowKit"),
    ] + (linksMCPKit ? [.package(path: mcpkit!)] : []),
    targets: [
        .executableTarget(name: "Quiver", dependencies: ["QuiverKit"]),
        .target(
            name: "QuiverKit",
            dependencies: [
                .product(name: "ArrowKit", package: "ArrowKit"),
            ] + (linksMCPKit ? [.product(name: "MCPKit", package: "MCPKit")] : []),
            swiftSettings: linksMCPKit ? [.define("CONTROL_SOCKET")] : []),
        .testTarget(name: "QuiverKitTests", dependencies: ["QuiverKit"]),
    ]
)

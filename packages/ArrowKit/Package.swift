// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "ArrowKit",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "ArrowKit", targets: ["ArrowKit"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.31.6")),
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "0.10.0"),
    ],
    targets: [
        .target(name: "ArrowKit", dependencies: [
            .product(name: "MLX", package: "mlx-swift"),
            .product(name: "MLXNN", package: "mlx-swift"),
            .product(name: "MLXFast", package: "mlx-swift"),
            .product(name: "HuggingFace", package: "swift-huggingface"),
        ]),
        .testTarget(name: "ArrowKitTests", dependencies: ["ArrowKit"]),
    ]
)

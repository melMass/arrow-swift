import ArrowKit
import CoreGraphics
import SwiftUI

/// Upstream's depth view: the current frame beside its depth (Turbo of inverse depth
/// between the 2nd and 98th percentiles), predicted normals and visibility, each a
/// panel at the depth task's grid resolution.
struct DepthView: View {
    let session: Session

    var body: some View {
        if let result = session.result, let panels = DepthPanels(result, frame: session.displayFrame) {
            Grid(horizontalSpacing: 6, verticalSpacing: 6) {
                GridRow {
                    panel("Input", result.frames.images[session.displayFrame])
                    panel(String(format: "Depth · near %.2f · far %.2f", panels.near, panels.far), panels.depth)
                }
                GridRow {
                    panel("Predicted normals", panels.normals)
                    panel("Visibility", panels.visibility)
                }
            }
        } else {
            ContentUnavailableView("No depth", systemImage: "square.3.layers.3d",
                                   description: Text("Run Video depth to see each frame's depth."))
        }
    }

    private func panel(_ title: String, _ image: CGImage) -> some View {
        VStack(spacing: 2) {
            Image(decorative: image, scale: 1).resizable().interpolation(.none).aspectRatio(contentMode: .fit)
            Text(title).font(.caption2).foregroundStyle(.secondary)
        }
    }
}

struct DepthPanels {
    let depth: CGImage
    let normals: CGImage
    let visibility: CGImage
    let near: Float
    let far: Float

    init?(_ result: ArrowResult, frame: Int) {
        guard let group = result.group(.depth), frame < result.frames.images.count else { return nil }
        let xy = result.queries.xy
        let firstRow = xy[group.start].y
        let columns = (group.start ..< group.start + group.perFrame).prefix(while: { xy[$0].y == firstRow }).count
        guard columns > 0, group.perFrame % columns == 0 else { return nil }
        let rows = group.perFrame / columns
        let range = group.start + frame * group.perFrame ..< group.start + (frame + 1) * group.perFrame
        let p = result.predictions
        let depths = range.map { p.points[$0].z }
        let valid = depths.filter { $0.isFinite && $0 > 0 }.sorted()
        guard !valid.isEmpty else { return nil }
        near = valid[Int(Float(valid.count - 1) * 0.02)]
        far = valid[Int(Float(valid.count - 1) * 0.98)]
        let (near, far) = (near, far)
        func image(_ colors: [SIMD3<Float>]) -> CGImage { Self.image(colors, width: columns, height: rows) }
        depth = image(depths.map { d in
            guard d.isFinite, d > 0, far > near else { return .zero }
            return SceneBuilder.turbo((1 / d - 1 / far) / (1 / near - 1 / far))
        })
        normals = image(range.map { SceneBuilder.normalColor(p.normals[$0]) })
        visibility = image(range.map { SceneBuilder.viridis(1 / (1 + exp(-p.visibility[$0]))) })
    }

    static func image(_ colors: [SIMD3<Float>], width: Int, height: Int) -> CGImage {
        var bytes = [UInt8](repeating: 255, count: width * height * 4)
        for (i, c) in colors.enumerated() {
            bytes[i * 4] = UInt8(c.x * 255); bytes[i * 4 + 1] = UInt8(c.y * 255); bytes[i * 4 + 2] = UInt8(c.z * 255)
        }
        let provider = CGDataProvider(data: Data(bytes) as CFData)!
        return CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
                       space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
    }
}

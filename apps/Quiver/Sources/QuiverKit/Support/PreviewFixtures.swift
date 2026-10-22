#if DEBUG
import ArrowKit
import CoreGraphics
import Foundation
import simd

/// A result made without the model: a camera dollying right past a textured wall with a
/// box in front of it, seen in `frames` frames. Reconstruction, tracks and cameras are
/// exact projections of that scene, so the views can be checked by eye.
enum PreviewFixtures {
    static let width = 336, height = 196
    static let focal: Float = 260

    static func result(frames: Int = 8) -> ArrowResult {
        let images = (0 ..< frames).map { image(frame: $0) }
        let frameSet = FrameSet(images: images, times: (0 ..< frames).map { Double($0) / 8 }, width: width, height: height)
        var settings = TaskSettings()
        settings.trackStride = 24
        settings.reconstructionStride = 6
        settings.depthStride = 4
        settings.poseGrid = 4
        let (queries, groups) = settings.queries([.track, .reconstruct, .depth], frames: frames, width: width, height: height)
        var predictions = Predictions()
        for i in 0 ..< queries.count {
            let source = Int(queries.source[i]), target = Int(queries.target[i])
            let world = point(at: queries.xy[i], frame: source)
            // In the reference (frame 0) camera; uv where frame `target` sees it.
            predictions.points.append(world)
            let seen = project(world, frame: target)
            predictions.uv.append(SIMD2((seen.x + 0.5) / Float(width), (seen.y + 0.5) / Float(height)))
            predictions.visibility.append(seen.x >= 0 && seen.x < Float(width) ? 2 : -2)
            // Surer on the box than the wall, and toward the image's left.
            predictions.confidence.append(5 + 20 * (3 - world.z) + 4 * (1 - queries.xy[i].x / Float(width)))
            predictions.normals.append(world.z < 2.5 ? SIMD3(0.3, -0.2, -0.93) : SIMD3(0, 0, -1))
            predictions.displacement.append(.zero)
        }
        let cameras = (0 ..< frames).map { f in
            var pose = matrix_identity_float4x4
            pose.columns.3 = SIMD4(offset(f), 0, 0, 1)
            return CameraEstimate(cameraToReference: pose, valid: true,
                                  intrinsics: SIMD4(focal, focal, Float(width) / 2, Float(height) / 2))
        }
        return ArrowResult(frames: frameSet, queries: queries, predictions: predictions, groups: groups,
                           cameras: cameras, encodeSeconds: 3.2, decodeSeconds: 1.4)
    }

    static func offset(_ frame: Int) -> Float { Float(frame) * 0.06 }

    /// Depth along a pixel's ray in `frame`: the box (nearer) over the middle, else the wall.
    static func depth(x: Float, y: Float) -> Float {
        (x > 120 && x < 220 && y > 70 && y < 170) ? 2 : 3
    }

    static func point(at xy: SIMD2<Float>, frame: Int) -> SIMD3<Float> {
        let shifted = xy.x + offset(frame) * focal / 2.5
        let z = depth(x: shifted, y: xy.y)
        return SIMD3((xy.x - Float(width) / 2) / focal * z + offset(frame), (xy.y - Float(height) / 2) / focal * z, z)
    }

    static func project(_ p: SIMD3<Float>, frame: Int) -> SIMD2<Float> {
        let x = p.x - offset(frame)
        return SIMD2(x / p.z * focal + Float(width) / 2, p.y / p.z * focal + Float(height) / 2)
    }

    /// Stripes on the wall, a checkered box, shifting with the camera.
    static func image(frame: Int) -> CGImage {
        let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                                space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        let shift = CGFloat(offset(frame) * focal / 3)
        for stripe in -2 ..< 20 {
            let hue = CGFloat(stripe % 6) / 6
            context.setFillColor(CGColor(red: 0.35 + 0.4 * hue, green: 0.45, blue: 0.75 - 0.4 * hue, alpha: 1))
            context.fill(CGRect(x: CGFloat(stripe) * 24 - shift, y: 0, width: 24, height: CGFloat(height)))
        }
        let box = CGRect(x: 120 - shift * 1.5, y: 30, width: 100, height: 100)
        for row in 0 ..< 4 {
            for column in 0 ..< 4 {
                context.setFillColor((row + column) % 2 == 0 ? CGColor(gray: 0.95, alpha: 1) : CGColor(red: 0.9, green: 0.35, blue: 0.2, alpha: 1))
                context.fill(CGRect(x: box.minX + CGFloat(column) * 25, y: box.minY + CGFloat(row) * 25, width: 25, height: 25))
            }
        }
        return context.makeImage()!
    }
}

extension Session {
    /// A session holding the fixture result, at `frame`.
    static func fixture(frame: Int = 5, phase: Phase = .idle, result: Bool = true) -> Session {
        let session = Session()
        let fixture = PreviewFixtures.result()
        session.sources = [URL(filePath: "/Movies/railway.mp4")]
        session.frames = fixture.frames
        if result { session.show(fixture) }
        session.frame = frame
        session.phase = phase
        return session
    }
}
#endif

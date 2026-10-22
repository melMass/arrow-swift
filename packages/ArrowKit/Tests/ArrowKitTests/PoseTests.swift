import Foundation
import MLX
import simd
import Testing
@testable import ArrowKit

/// The pose task against upstream's on the same predictions (`pose.safetensors`): no
/// weights needed, the decoder's answers come from the fixture.
@Suite("Pose task")
struct PoseTests {
    @Test("Pose queries are upstream's: a 16 × 16 grid per frame, own camera then the middle frame's")
    func queries() throws {
        let fixture = try Fixture("pose")
        let frames = fixture.spec["frames"] as! Int
        var settings = TaskSettings()
        settings.poseGrid = 16
        let (queries, groups) = settings.queries(
            [.pose], frames: frames, width: fixture.spec["width"] as! Int, height: fixture.spec["height"] as! Int)
        let expected = fixture.queries()
        #expect(queries.xy == expected.xy)
        #expect(queries.source == expected.source)
        #expect(queries.target == expected.target)
        #expect(queries.reference == expected.reference)
        #expect(groups.first?.perFrame == 256)
        #expect(groups.first?.reference == 1)
    }

    @Test("Cameras fitted from upstream's predictions match its poses and intrinsics")
    func cameras() throws {
        let fixture = try Fixture("pose")
        let frames = fixture.spec["frames"] as! Int
        let raw = ["points", "point_confidence", "uv", "visibility_logits", "normals", "displacement"]
            .reduce(into: [String: MLXArray]()) { $0[$1] = fixture["predictions.\($1)"] }
        let predictions = ArrowModel.predictions(raw)
        let cameras = PoseFit.cameras(
            predictions, xy: fixture.queries().xy, frames: frames, perFrame: 256, reference: PoseFit.middle(frames))

        let poses = fixture["camera_to_reference"].asArray(Float.self)
        let valid = fixture["valid"].asArray(Bool.self)
        let intrinsics = fixture["intrinsics"].asArray(Float.self)
        for (frame, camera) in cameras.enumerated() {
            #expect(camera.valid == valid[frame], "frame \(frame) validity")
            // Row-major 4 × 4 in the fixture.
            let expected = (0 ..< 16).map { poses[frame * 16 + $0] }
            for r in 0 ..< 3 {
                for c in 0 ..< 4 {
                    let value = camera.cameraToReference[c][r]
                    #expect(abs(value - expected[r * 4 + c]) < 1e-3, "frame \(frame) [\(r), \(c)] \(value) vs \(expected[r * 4 + c])")
                }
            }
            let k = (0 ..< 9).map { intrinsics[frame * 9 + $0] }
            if k[0].isNaN {
                #expect(camera.intrinsics == nil)
            } else if let mine = camera.intrinsics {
                let theirs = SIMD4(k[0], k[4], k[2], k[5])
                let error = (mine - theirs).indices.map { abs(mine[$0] - theirs[$0]) }.max()!
                #expect(error < 1e-2 * theirs.max(), "frame \(frame) \(mine) vs \(theirs)")
            } else {
                Issue.record("frame \(frame): no intrinsics, upstream has \(k)")
            }
        }
    }

    @Test("A known rotation and translation are recovered from weighted points")
    func rigid() {
        let rotation = simd_float3x3(simd_quatf(angle: 0.7, axis: simd_normalize(SIMD3(1, 2, 3))))
        let translation = SIMD3<Float>(0.3, -1.2, 2)
        let source = (0 ..< 40).map { i in SIMD3<Float>(sin(Float(i)), cos(Float(i) * 1.3), Float(i % 7) * 0.2) }
        let target = source.map { rotation * $0 + translation }
        let fit = PoseFit.weightedRigid(target: target, source: source, weight: source.map { _ in 1 })!
        for (s, t) in zip(source, target) {
            let mapped = fit * SIMD4(s, 1)
            #expect(simd_distance(SIMD3(mapped.x, mapped.y, mapped.z), t) < 1e-4)
        }
    }
}

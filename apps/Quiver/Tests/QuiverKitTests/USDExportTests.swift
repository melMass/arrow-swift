@testable import ArrowKit
import Foundation
import Testing
@testable import QuiverKit

@Suite("USD export")
struct USDExportTests {
    let result = PreviewFixtures.result()

    @Test("The stage passes usdchecker and holds the cloud, tracks, trajectories and cameras")
    func checker() throws {
        let url = FileManager.default.temporaryDirectory.appending(path: "quiver-\(UUID().uuidString).usda")
        defer { try? FileManager.default.removeItem(at: url) }
        try USDExport.write(result, to: url)

        let checker = Process()
        checker.executableURL = URL(filePath: "/usr/bin/usdchecker")
        checker.arguments = [url.path]
        let pipe = Pipe()
        checker.standardOutput = pipe
        checker.standardError = pipe
        try checker.run()
        checker.waitUntilExit()
        let report = String(decoding: pipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
        #expect(checker.terminationStatus == 0, "\(report)")

        let tree = Process()
        tree.executableURL = URL(filePath: "/usr/bin/usdtree")
        tree.arguments = [url.path]
        let out = Pipe()
        tree.standardOutput = out
        try tree.run()
        tree.waitUntilExit()
        let prims = String(decoding: out.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
        for name in ["Cloud", "Tracks", "Trajectories", "Cameras", "Frame_0", "Camera"] {
            #expect(prims.contains(name), "\(name) in \(prims)")
        }
    }

    @Test("Points flip into USD's frame and the keep fraction thins the cloud")
    func content() throws {
        let all = USDExport.stage(result)
        let kept = USDExport.stage(result, options: .init(keep: 0.25))
        #expect(kept.count < all.count)
        // The fixture's wall is 3 units ahead: z forward becomes −z.
        #expect(all.contains("-3)"))
        #expect(all.contains("endTimeCode = 7"))
    }

    @Test("A camera pose round-trips through the axis flip: the reference camera is the identity")
    func identityCamera() {
        #expect(USDExport.matrix(matrix_identity_float4x4) == "( (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0) )")
    }
}

import simd

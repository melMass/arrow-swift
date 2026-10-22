import Foundation
import simd

/// A result as a USD stage (text `.usda`), Y up, one time code per timestep (per frame
/// for a single camera):
///
/// - `/Arrow/Cloud`: the reconstruction, `Points` with each point's colour at its source
///   pixel and `primvars:confidence`;
/// - `/Arrow/Depth/Frame_k`: a frame's depth points, visible only at its timestep;
/// - `/Arrow/Tracks`: one point per track, its position sampled at every timestep, with
///   `primvars:visible` (1 where the model sees it then); several cameras are merged by
///   their median (`TrackSet.Fusion.median`);
/// - `/Arrow/Trajectories`: each track's whole path as a linear curve;
/// - `/Arrow/Cameras/Frame_k` and the animated `/Arrow/Cameras/Camera` (`Camera_v` per
///   camera of a multi-view set): the fitted poses and intrinsics (aperture 36 wide,
///   focal length scaled to match).
///
/// ARROW's reference camera (x right, y down, z forward) becomes USD's (x right, y up,
/// z back). Scale is the model's, relative, not metres.
public enum USDExport {
    public struct Options: Sendable {
        /// The most confident fraction of cloud points and tracks written (tracks ranked by
        /// mean confidence).
        public var keep: Float = 1
        public var framesPerSecond: Double = 8
        /// Point width, in scene units.
        public var pointWidth: Float = 0.004

        public init(keep: Float = 1, framesPerSecond: Double = 8, pointWidth: Float = 0.004) {
            self.keep = keep
            self.framesPerSecond = framesPerSecond
            self.pointWidth = pointWidth
        }
    }

    public static func write(_ result: ArrowResult, to url: URL, options: Options = Options()) throws {
        try stage(result, options: options).write(to: url, atomically: true, encoding: .utf8)
    }

    public static func stage(_ result: ArrowResult, options: Options = Options()) -> String {
        let steps = result.frames.timesteps
        var out = """
        #usda 1.0
        (
            defaultPrim = "Arrow"
            upAxis = "Y"
            metersPerUnit = 1
            startTimeCode = 0
            endTimeCode = \(max(steps - 1, 0))
            timeCodesPerSecond = \(options.framesPerSecond)
            doc = "ARROW (ifradlin/arrow) predictions, exported by Quiver. Relative scale."
        )

        def Xform "Arrow"
        {

        """
        if let group = result.group(.reconstruct) { out += cloud(result, group, options) }
        if let group = result.group(.depth) { out += depth(result, group, options) }
        if let group = result.group(.track) { out += tracks(result, group, options) }
        if let cameras = result.cameras { out += self.cameras(cameras, result: result) }
        out += "}\n"
        return out
    }

    /// ARROW camera coordinates to USD's.
    static func usd(_ p: SIMD3<Float>) -> SIMD3<Float> { SIMD3(p.x, -p.y, -p.z) }

    static func cloud(_ result: ArrowResult, _ group: QueryGroup, _ options: Options) -> String {
        let range = group.start ..< group.start + group.count
        let colors = result.colors(group)
        let cut = threshold(range.map { result.predictions.confidence[$0] }, keep: options.keep)
        var points: [SIMD3<Float>] = [], tints: [SIMD3<Float>] = [], confidence: [Float] = []
        for (k, i) in range.enumerated() where result.predictions.confidence[i] >= cut && finite(result.predictions.points[i]) {
            points.append(usd(result.predictions.points[i]))
            tints.append(colors[k])
            confidence.append(result.predictions.confidence[i])
        }
        return pointsPrim("Cloud", points: points, colors: tints, width: options.pointWidth, extra: [
            "float[] primvars:confidence = \(list(confidence)) (\n            interpolation = \"vertex\"\n        )",
        ])
    }

    static func depth(_ result: ArrowResult, _ group: QueryGroup, _ options: Options) -> String {
        let frames = result.frames.images.count, views = result.frames.views
        let colors = result.colors(group)
        var out = "    def Scope \"Depth\"\n    {\n"
        for frame in 0 ..< frames {
            let range = group.start + frame * group.perFrame ..< group.start + (frame + 1) * group.perFrame
            let cut = threshold(range.map { result.predictions.confidence[$0] }, keep: options.keep)
            var points: [SIMD3<Float>] = [], tints: [SIMD3<Float>] = []
            for i in range where result.predictions.confidence[i] >= cut && finite(result.predictions.points[i]) {
                points.append(usd(result.predictions.points[i]))
                tints.append(colors[i - group.start])
            }
            let shown = (0 ..< result.frames.timesteps)
                .map { "\($0): \"\($0 == frame / views ? "inherited" : "invisible")\"" }.joined(separator: ", ")
            out += indent(pointsPrim("Frame_\(frame)", points: points, colors: tints, width: options.pointWidth, extra: [
                "token visibility.timeSamples = { \(shown) }",
            ]))
        }
        return out + "    }\n\n"
    }

    static func tracks(_ result: ArrowResult, _ group: QueryGroup, _ options: Options) -> String {
        let tracks = TrackSet(result, group)
        let seeds = tracks.seeds, frames = tracks.timesteps
        let statistics = TrackStatistics(tracks)
        let mean = statistics.meanConfidence
        let confident = statistics.kept(options.keep)
        let kept = (0 ..< seeds).filter { seed in
            confident[seed] && (0 ..< frames).allSatisfy { finite(tracks.points[tracks.index($0, seed)]) }
        }
        let colors = result.colors(group)
        let tints = kept.map { colors[$0] }

        var positions: [String] = [], visible: [String] = []
        for t in 0 ..< frames {
            positions.append("\(t): \(list(kept.map { usd(tracks.points[tracks.index(t, $0)]) }))")
            visible.append("\(t): \(list(kept.map { Float(tracks.visibility[tracks.index(t, $0)] > 0 ? 1 : 0) }))")
        }
        var out = """
            def Points "Tracks"
            {
                point3f[] points.timeSamples = {
                    \(positions.joined(separator: ",\n            "))
                }
                float[] widths = [\(options.pointWidth * 2)] (
                    interpolation = "constant"
                )
                color3f[] primvars:displayColor = \(list(tints)) (
                    interpolation = "vertex"
                )
                float[] primvars:visible.timeSamples = {
                    \(visible.joined(separator: ",\n            "))
                }
                float[] primvars:meanConfidence = \(list(kept.map { mean[$0] })) (
                    interpolation = "vertex"
                )
            }


        """
        let path = kept.flatMap { s in (0 ..< frames).map { usd(tracks.points[tracks.index($0, s)]) } }
        out += """
            def BasisCurves "Trajectories"
            {
                uniform token type = "linear"
                int[] curveVertexCounts = \(list(kept.map { _ in frames }))
                point3f[] points = \(list(path))
                float[] widths = [\(options.pointWidth / 2)] (
                    interpolation = "constant"
                )
                color3f[] primvars:displayColor = \(list(tints)) (
                    interpolation = "uniform"
                )
            }


        """
        return out
    }

    static func cameras(_ cameras: [CameraEstimate], result: ArrowResult) -> String {
        let width = Float(result.frames.width), height = Float(result.frames.height)
        let aperture: Float = 36
        func lens(_ camera: CameraEstimate) -> String {
            let k = camera.intrinsics ?? SIMD4(max(width, height), max(width, height), width / 2, height / 2)
            // USD's default near plane is 1 unit; this scene's scale is relative, often ~1.
            return """
                    float2 clippingRange = (0.0001, 100000)
                    float focalLength = \(aperture * k.x / width)
                    float horizontalAperture = \(aperture)
                    float verticalAperture = \(aperture * height / width * k.x / k.y)
                    float horizontalApertureOffset = \(aperture * (k.z - width / 2) / width)
                    float verticalApertureOffset = \(-aperture * (k.w - height / 2) / width * k.x / k.y)
            """
        }
        var out = "    def Scope \"Cameras\"\n    {\n"
        for (frame, camera) in cameras.enumerated() where camera.valid {
            out += """
                    def Camera "Frame_\(frame)"
                    {
                        matrix4d xformOp:transform = \(matrix(camera.cameraToReference))
                        uniform token[] xformOpOrder = ["xformOp:transform"]
                \(lens(camera))
                    }


            """
        }
        let views = result.frames.views
        for view in 0 ..< views {
            let valid = cameras.enumerated().filter { $0.element.valid && $0.offset % views == view }
            guard let first = valid.first?.element else { continue }
            let samples = valid.map { "\($0.offset / views): \(matrix($0.element.cameraToReference))" }
                .joined(separator: ",\n                ")
            out += """
                    def Camera "\(views == 1 ? "Camera" : "Camera_\(view)")"
                    {
                        matrix4d xformOp:transform.timeSamples = {
                            \(samples)
                        }
                        uniform token[] xformOpOrder = ["xformOp:transform"]
                \(lens(first))
                    }

            """
        }
        return out + "    }\n"
    }

    /// A camera-to-reference pose as a USD camera's local-to-world: USD cameras look down
    /// −z with +y up, so the flip S = diag(1, −1, −1) goes on both sides (S·M·S), and USD
    /// writes row vectors, so each row is a column of the simd matrix.
    static func matrix(_ m: simd_float4x4) -> String {
        let flip = simd_float4x4(diagonal: SIMD4(1, -1, -1, 1))
        let w = flip * m * flip
        let rows = (0 ..< 4).map { c in "(\(w[c].x), \(w[c].y), \(w[c].z), \(w[c].w))" }
        return "( \(rows.joined(separator: ", ")) )"
    }

    static func pointsPrim(_ name: String, points: [SIMD3<Float>], colors: [SIMD3<Float>], width: Float, extra: [String]) -> String {
        """
            def Points "\(name)"
            {
                point3f[] points = \(list(points))
                float[] widths = [\(width)] (
                    interpolation = "constant"
                )
                color3f[] primvars:displayColor = \(list(colors)) (
                    interpolation = "vertex"
                )
                \(extra.joined(separator: "\n        "))
            }


        """
    }

    static func indent(_ text: String) -> String {
        text.split(separator: "\n", omittingEmptySubsequences: false).map { $0.isEmpty ? "" : "    " + $0 }.joined(separator: "\n")
    }

    static func list(_ values: [Float]) -> String { "[" + values.map { format($0) }.joined(separator: ", ") + "]" }
    static func list(_ values: [Int]) -> String { "[" + values.map(String.init).joined(separator: ", ") + "]" }
    static func list(_ values: [SIMD3<Float>]) -> String {
        "[" + values.map { "(\(format($0.x)), \(format($0.y)), \(format($0.z)))" }.joined(separator: ", ") + "]"
    }

    /// Six significant digits: plenty for a relative-scale scene, a third of the text.
    static func format(_ value: Float) -> String { String(format: "%.6g", value) }

    static func finite(_ p: SIMD3<Float>) -> Bool { p.x.isFinite && p.y.isFinite && p.z.isFinite }

    static func threshold(_ values: [Float], keep: Float) -> Float { Confidence.threshold(values, keep: keep) }
}

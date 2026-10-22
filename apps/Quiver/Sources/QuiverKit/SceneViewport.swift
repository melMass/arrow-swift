import AppKit
import SceneKit
import SwiftUI

/// The result in 3D: points and line segments, orbited with a drag and zoomed with a
/// scroll or pinch. A new scene puts the camera back behind the reference camera, looking
/// at the scene's middle.
struct SceneViewport: NSViewRepresentable {
    let geometry: Geometry
    let pointSize: Float
    /// Changes when a new result arrives, so the camera resets once per result.
    let sceneID: Int

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> SCNView {
        let view = SCNView()
        view.scene = context.coordinator.scene
        view.pointOfView = context.coordinator.camera
        view.allowsCameraControl = true
        view.defaultCameraController.interactionMode = .orbitTurntable
        view.backgroundColor = NSColor(white: 0.09, alpha: 1)
        view.antialiasingMode = .multisampling4X
        view.rendersContinuously = false
        return view
    }

    func updateNSView(_ view: SCNView, context: Context) {
        let coordinator = context.coordinator
        coordinator.show(geometry, pointSize: pointSize)
        if coordinator.sceneID != sceneID {
            coordinator.sceneID = sceneID
            let center = SceneBuilder.center(geometry)
            // Behind and above the reference camera, so its frustum and the others are in view.
            let distance = max(center.z.magnitude, 0.1)
            coordinator.camera.simdPosition = SIMD3(0, 0.35 * distance, 0.6 * distance)
            coordinator.camera.simdLook(at: center, up: SIMD3(0, 1, 0), localFront: SIMD3(0, 0, -1))
            view.defaultCameraController.target = SCNVector3(center)
            view.pointOfView = coordinator.camera
        }
    }

    @MainActor
    final class Coordinator {
        let scene = SCNScene()
        let camera = SCNNode()
        let points = SCNNode()
        let lines = SCNNode()
        let plane = SCNNode()
        var sceneID = -1

        init() {
            let lens = SCNCamera()
            lens.zNear = 0.001
            lens.zFar = 1000
            lens.fieldOfView = 60
            camera.camera = lens
            scene.rootNode.addChildNode(camera)
            scene.rootNode.addChildNode(points)
            scene.rootNode.addChildNode(lines)
            scene.rootNode.addChildNode(plane)
        }

        func show(_ geometry: Geometry, pointSize: Float) {
            points.geometry = Self.make(geometry.points, geometry.colors, primitive: .point, pointSize: CGFloat(pointSize))
            lines.geometry = Self.make(geometry.lines, geometry.lineColors, primitive: .line, pointSize: 1)
            plane.geometry = geometry.plane.map { Self.quad($0.corners, image: $0.image, opacity: $0.opacity) }
        }

        /// An image on a quad: corners top-left, top-right, bottom-right, bottom-left;
        /// SceneKit's texture origin is the image's top-left.
        static func quad(_ corners: [SIMD3<Float>], image: CGImage, opacity: Float) -> SCNGeometry {
            let vertices = SCNGeometrySource(vertices: corners.map { SCNVector3($0) })
            let uv = SCNGeometrySource(textureCoordinates: [CGPoint(x: 0, y: 0), CGPoint(x: 1, y: 0), CGPoint(x: 1, y: 1), CGPoint(x: 0, y: 1)])
            let element = SCNGeometryElement(indices: [Int32(0), 1, 2, 0, 2, 3], primitiveType: .triangles)
            let geometry = SCNGeometry(sources: [vertices, uv], elements: [element])
            let material = SCNMaterial()
            material.lightingModel = .constant
            material.diffuse.contents = image
            material.isDoubleSided = true
            material.transparency = CGFloat(opacity)
            material.writesToDepthBuffer = false
            geometry.materials = [material]
            return geometry
        }

        /// Vertices and per-vertex colours as one unlit SceneKit geometry.
        static func make(_ positions: [SIMD3<Float>], _ colors: [SIMD4<Float>], primitive: SCNGeometryPrimitiveType, pointSize: CGFloat) -> SCNGeometry? {
            guard !positions.isEmpty else { return nil }
            let vertices = positions.withUnsafeBytes { Data($0) }
            let tints = colors.withUnsafeBytes { Data($0) }
            let vertexSource = SCNGeometrySource(
                data: vertices, semantic: .vertex, vectorCount: positions.count, usesFloatComponents: true,
                componentsPerVector: 3, bytesPerComponent: 4, dataOffset: 0, dataStride: MemoryLayout<SIMD3<Float>>.stride)
            let colorSource = SCNGeometrySource(
                data: tints, semantic: .color, vectorCount: colors.count, usesFloatComponents: true,
                componentsPerVector: 4, bytesPerComponent: 4, dataOffset: 0, dataStride: MemoryLayout<SIMD4<Float>>.stride)
            let count = primitive == .line ? positions.count / 2 : positions.count
            let indices = (0 ..< Int32(positions.count)).map { $0 }
            let element = SCNGeometryElement(
                data: indices.withUnsafeBytes { Data($0) }, primitiveType: primitive, primitiveCount: count, bytesPerIndex: 4)
            if primitive == .point {
                element.pointSize = pointSize
                element.minimumPointScreenSpaceRadius = pointSize / 2
                element.maximumPointScreenSpaceRadius = pointSize * 2
            }
            let geometry = SCNGeometry(sources: [vertexSource, colorSource], elements: [element])
            let material = SCNMaterial()
            material.lightingModel = .constant
            material.diffuse.contents = NSColor.white
            geometry.materials = [material]
            return geometry
        }
    }
}

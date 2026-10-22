@preconcurrency import AVFoundation
import CoreGraphics
import Foundation
import ImageIO
import MLX
import UniformTypeIdentifiers

/// Where frames come from: one video, or still images in order.
public enum FrameSource: Sendable, Equatable {
    case video(URL)
    case images([URL])

    /// A video file, or a folder's images in natural name order.
    public static func at(_ url: URL) throws -> FrameSource {
        var directory: ObjCBool = false
        if FileManager.default.fileExists(atPath: url.path, isDirectory: &directory), directory.boolValue {
            let images = try FileManager.default.contentsOfDirectory(at: url, includingPropertiesForKeys: nil)
                .filter { isImage($0) }
                .sorted { $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending }
            guard !images.isEmpty else { throw FramesError.noFrames(url.path) }
            return .images(images)
        }
        return isImage(url) ? .images([url]) : .video(url)
    }

    static func isImage(_ url: URL) -> Bool {
        UTType(filenameExtension: url.pathExtension)?.conforms(to: .image) ?? false
    }

    /// How many frames there are to select from: a video's frames at its nominal rate, or
    /// the images.
    public func frameCount() async throws -> Int {
        switch self {
        case let .images(urls): return urls.count
        case let .video(url): return try await VideoInfo.read(url).frames
        }
    }
}

/// Which frames of a source to read, as indices into its frames.
public enum FrameSelection: Sendable, Equatable {
    /// `count` frames spread evenly over the whole source, first and last included.
    case spread(Int)
    /// Every `step`-th frame from `start`, before `end` (nil: the last frame), at most
    /// `limit` of them: upstream's start, stride, end and max frames.
    case range(start: Int, end: Int?, step: Int, limit: Int)

    /// The selected indices among `total` frames.
    public func indices(of total: Int) -> [Int] {
        switch self {
        case let .spread(count):
            return FrameSet.spread(total, count)
        case let .range(start, end, step, limit):
            let stop = min(end ?? total, total)
            guard start < stop, step > 0, limit > 0 else { return [] }
            return Array(stride(from: max(0, start), to: stop, by: step).prefix(limit))
        }
    }
}

public enum FramesError: LocalizedError {
    case noFrames(String)
    case unreadable(String)
    case unevenViews

    public var errorDescription: String? {
        switch self {
        case let .noFrames(path): "No frames in \(path)."
        case let .unreadable(path): "Could not read \(path)."
        case .unevenViews: "Every camera needs a frame at each selected index; one of them is shorter."
        }
    }
}

/// Model-ready frames: resized so the long side is `maxSide`, each side rounded to a
/// multiple of 14, centered on a white canvas the size of the largest (upstream's
/// `prepare_images`).
///
/// Several synchronized cameras (multi-view) are one set too: frame `t · views + v` is
/// camera v at timestep t, the order upstream's `load_multiview_input` uses. The model
/// itself has no notion of cameras or time; only reading the results back does.
public struct FrameSet: @unchecked Sendable {
    /// Each frame as prepared, for display: RGBA, `width` × `height`.
    public let images: [CGImage]
    /// Seconds into the video, or the image's index.
    public let times: [Double]
    public let width: Int
    public let height: Int
    /// Synchronized cameras; 1 for a single video or photo set.
    public let views: Int

    /// Frames already at model size (`width` × `height`, multiples of 14).
    public init(images: [CGImage], times: [Double], width: Int, height: Int, views: Int = 1) {
        precondition(views >= 1 && images.count % views == 0, "frames must be whole timesteps")
        self.images = images
        self.times = times
        self.width = width
        self.height = height
        self.views = views
    }

    public var timesteps: Int { images.count / views }

    /// The frame of camera `view` at `timestep`.
    public func index(timestep: Int, view: Int) -> Int { timestep * views + view }

    /// [1, frames, 3, H, W] in [0, 1].
    public func tensor() -> MLXArray {
        var values = [Float](repeating: 1, count: images.count * 3 * height * width)
        let plane = height * width
        for (f, image) in images.enumerated() {
            let pixels = Self.rgba(image, width: width, height: height)
            let base = f * 3 * plane
            for i in 0 ..< plane {
                values[base + i] = Float(pixels[i * 4]) / 255
                values[base + plane + i] = Float(pixels[i * 4 + 1]) / 255
                values[base + 2 * plane + i] = Float(pixels[i * 4 + 2]) / 255
            }
        }
        return MLXArray(values, [1, images.count, 3, height, width])
    }

    static func rgba(_ image: CGImage, width: Int, height: Int) -> [UInt8] {
        var pixels = [UInt8](repeating: 255, count: width * height * 4)
        pixels.withUnsafeMutableBytes { buffer in
            let context = CGContext(
                data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
            context?.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        }
        return pixels
    }

    /// The model size for an image: long side scaled to `maxSide`, each side rounded to
    /// the nearest multiple of 14 (at least 14).
    public static func inputSize(width: Int, height: Int, maxSide: Int) -> (Int, Int) {
        let scale = Double(maxSide) / Double(max(width, height))
        func side(_ s: Int) -> Int { max(14, (max(1, Int((Double(s) * scale).rounded(.toNearestOrEven))) + 7) / 14 * 14) }
        return (side(width), side(height))
    }

    /// Reads `count` frames spread evenly over `source` (all of a shorter image list) and
    /// prepares them at `maxSide`.
    public static func load(_ source: FrameSource, count: Int, maxSide: Int = 504) async throws -> FrameSet {
        try await load(source, selection: .spread(count), maxSide: maxSide)
    }

    public static func load(_ source: FrameSource, selection: FrameSelection, maxSide: Int = 504) async throws -> FrameSet {
        try await load(views: [source], selection: selection, maxSide: maxSide)
    }

    /// Reads the same selected frame indices from each synchronized camera and interleaves
    /// them by timestep. The streams must already be aligned frame for frame.
    public static func load(views sources: [FrameSource], selection: FrameSelection, maxSide: Int = 504) async throws -> FrameSet {
        guard !sources.isEmpty else { throw FramesError.noFrames("no sources") }
        var totals: [Int] = []
        for source in sources { totals.append(try await source.frameCount()) }
        // Cameras of different lengths are read over the shortest.
        let indices = selection.indices(of: totals.min()!)

        var streams: [(images: [CGImage], times: [Double])] = []
        for source in sources {
            let read: ([CGImage], [Double])
            switch source {
            case let .video(url): read = try await videoFrames(url, indices: indices, maxSide: maxSide)
            case let .images(urls):
                read = (try indices.map { try still(urls[$0], maxSide: maxSide) }, indices.map(Double.init))
            }
            guard !read.0.isEmpty else { throw FramesError.noFrames("\(source)") }
            streams.append(read)
        }
        let timesteps = streams.map(\.images.count).min()!
        guard streams.allSatisfy({ $0.images.count == timesteps }) else { throw FramesError.unevenViews }

        var raw: [CGImage] = [], times: [Double] = []
        for t in 0 ..< timesteps {
            for stream in streams {
                raw.append(stream.images[t])
                times.append(stream.times[t])
            }
        }
        let sizes = raw.map { inputSize(width: $0.width, height: $0.height, maxSide: maxSide) }
        let width = sizes.map(\.0).max()!, height = sizes.map(\.1).max()!
        let prepared = zip(raw, sizes).map { image, size in
            place(image, size: size, canvas: (width, height))
        }
        return FrameSet(images: prepared, times: times, width: width, height: height, views: sources.count)
    }

    /// `count` indices spread evenly over `total`, first and last included.
    static func spread(_ total: Int, _ count: Int) -> [Int] {
        guard count < total else { return Array(0 ..< total) }
        guard count > 1 else { return [0] }
        return (0 ..< count).map { Int((Double($0) * Double(total - 1) / Double(count - 1)).rounded()) }
    }

    /// `image` resized to `size` (high-quality interpolation) and centered on a white canvas.
    static func place(_ image: CGImage, size: (Int, Int), canvas: (Int, Int)) -> CGImage {
        let context = CGContext(
            data: nil, width: canvas.0, height: canvas.1, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        context.setFillColor(CGColor(gray: 1, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: canvas.0, height: canvas.1))
        context.interpolationQuality = .high
        let left = (canvas.0 - size.0) / 2, top = (canvas.1 - size.1) / 2
        // Core Graphics' origin is the bottom-left.
        context.draw(image, in: CGRect(x: left, y: canvas.1 - top - size.1, width: size.0, height: size.1))
        return context.makeImage()!
    }

    /// An image upright as its EXIF orientation says, decoded no larger than twice the
    /// model size: hundreds of full-resolution photos would not fit in memory.
    static func still(_ url: URL, maxSide: Int) throws -> CGImage {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let width = properties[kCGImagePropertyPixelWidth] as? Int,
              let height = properties[kCGImagePropertyPixelHeight] as? Int,
              let image = CGImageSourceCreateThumbnailAtIndex(source, 0, [
                  kCGImageSourceCreateThumbnailFromImageAlways: true,
                  kCGImageSourceCreateThumbnailWithTransform: true,
                  kCGImageSourceThumbnailMaxPixelSize: min(max(width, height), 2 * maxSide),
              ] as CFDictionary)
        else { throw FramesError.unreadable(url.path) }
        return image
    }

    /// The frames at `indices` (at the video's nominal rate), upright as players show them,
    /// decoded no larger than twice the model size.
    static func videoFrames(_ url: URL, indices: [Int], maxSide: Int) async throws -> ([CGImage], [Double]) {
        let info = try await VideoInfo.read(url)
        let generator = AVAssetImageGenerator(asset: info.asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        generator.maximumSize = CGSize(width: 2 * maxSide, height: 2 * maxSide)
        let requested = indices.map { CMTime(seconds: Double($0) / info.rate, preferredTimescale: 600) }
        var frames: [(requested: Double, image: CGImage, actual: Double)] = []
        // Requested together, the generator decodes runs of nearby frames once.
        for await result in generator.images(for: requested) {
            frames.append((result.requestedTime.seconds, try result.image, try result.actualTime.seconds))
        }
        frames.sort { $0.requested < $1.requested }
        return (frames.map(\.image), frames.map(\.actual))
    }
}

/// A video's frame count at its nominal rate.
struct VideoInfo {
    let asset: AVURLAsset
    let rate: Double
    let frames: Int

    static func read(_ url: URL) async throws -> VideoInfo {
        let asset = AVURLAsset(url: url)
        let duration = try await asset.load(.duration).seconds
        guard let track = try await asset.loadTracks(withMediaType: .video).first else { throw FramesError.noFrames(url.path) }
        let rate = max(Double(try await track.load(.nominalFrameRate)), 1)
        return VideoInfo(asset: asset, rate: rate, frames: max(1, Int((duration * rate).rounded(.down))))
    }
}

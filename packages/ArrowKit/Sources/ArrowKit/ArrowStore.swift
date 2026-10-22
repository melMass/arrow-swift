import Foundation
import HuggingFace

/// ARROW's weights in the Hugging Face cache (the layout huggingface_hub uses, so a cache
/// filled by the Python tools serves this one and the reverse).
public struct ArrowStore: Sendable {
    public static let repo = "ifradlin/arrow"
    /// The one file the model reads, 5.3 GB.
    public static let weightsFile = "model.safetensors"
    public static let downloadSize = "5.3 GB"

    public let cache: HubCache

    public init(cache: HubCache = .default) {
        self.cache = cache
    }

    /// The snapshot folder when the weights are already cached, else nil. No network.
    public func cachedSnapshot() -> URL? {
        guard let id = Repo.ID(rawValue: Self.repo),
              let file = cache.cachedFilePath(repo: id, kind: .model, revision: "main", filename: Self.weightsFile)
        else { return nil }
        return file.deletingLastPathComponent()
    }

    /// The snapshot folder, downloading the weights first if needed.
    public func snapshot(progress: @MainActor @Sendable @escaping (Progress) -> Void = { _ in }) async throws -> URL {
        if let cached = cachedSnapshot() { return cached }
        guard let id = Repo.ID(rawValue: Self.repo) else { throw CocoaError(.fileNoSuchFile) }
        return try await HubClient(cache: cache).downloadSnapshot(
            of: id, kind: .model, revision: "main", matching: [Self.weightsFile, "config.json"],
            progressHandler: progress)
    }
}

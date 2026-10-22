import Foundation
import MLX
import Testing
@testable import ArrowKit

/// The network against the PyTorch model's own outputs on the same inputs
/// (`Fixtures/capture.py`, CPU float32). Needs the weights and `ARROW_MODEL_TESTS=1`;
/// run through `test.sh`, which puts MLX's shaders where it looks.
@Suite("Parity with PyTorch", .serialized, .enabled(if: Weights.enabled))
struct ParityTests {
    static let float32 = Result { try ArrowModel.load(from: Weights.directory!, precision: .float32) }

    @Test("Encoder: patches and ID tokens match in float32")
    func encoder() throws {
        let model = try Self.float32.get()
        let fixture = try Fixture("encode")
        let encoded = model.network.encoder(fixture["images"], dtype: .float32)
        let patches = compare(encoded.patches.reshaped(1, -1, 3072), fixture["patches"])
        let ids = compare(encoded.ids, fixture["id_tokens"])
        print("patches", patches, "ids", ids, "reference", encoded.reference as Any)
        #expect(encoded.reference == fixture["reference"].item(Int.self))
        #expect(patches.cosine > 0.9999)
        #expect(ids.cosine > 0.9999)
    }

    @Test("Encoder: a clip DA3 reorders (reference frame 3) comes back in input order")
    func reordered() throws {
        let model = try Self.float32.get()
        let fixture = try Fixture("reordered")
        let encoded = model.network.encoder(fixture["images"], dtype: .float32)
        let patches = compare(encoded.patches.reshaped(1, -1, 3072), fixture["patches"])
        let ids = compare(encoded.ids, fixture["id_tokens"])
        print("reordered patches", patches, "ids", ids)
        #expect(encoded.reference == 3)
        #expect(encoded.reference == fixture["reference"].item(Int.self))
        #expect(patches.cosine > 0.9999)
        #expect(ids.cosine > 0.9999)
    }

    @Test("Half precision: fp16 and bf16 against the float32 reference", arguments: [ArrowPrecision.float16, .bfloat16])
    func halfPrecision(_ precision: ArrowPrecision) throws {
        let model = try ArrowModel.load(from: Weights.directory!, precision: precision)
        let encode = try Fixture("encode"), decode = try Fixture("decode")
        let encoded = model.network.encoder(encode["images"], dtype: precision.dtype)
        let patches = compare(encoded.patches.reshaped(1, -1, 3072), encode["patches"])
        let predictions = model.network.decode(
            decode.queries(), encoded: model.encode(encode["images"]),
            padded: QueryEmbedding.padImages(encode["images"]), dtype: precision.dtype)
        let points = compare(predictions["points"]!.reshaped(-1), decode["predictions.points"].reshaped(-1))
        let uv = compare(predictions["uv"]!.reshaped(-1), decode["predictions.uv"].reshaped(-1))
        print(precision, "patches", patches, "points", points, "uv", uv)
        #expect(patches.cosine > 0.999)
        #expect(points.cosine > 0.999)
        #expect(uv.cosine > 0.999)
    }

    @Test("Decoder: query tokens and every head match in float32")
    func decoder() throws {
        let model = try Self.float32.get()
        let encode = try Fixture("encode"), decode = try Fixture("decode")
        let encoded = model.encode(encode["images"])
        let queries = decode.queries()
        let tokens = model.network.queryEmbed(
            queries, encoded: encoded, padded: QueryEmbedding.padImages(encoded.images), dtype: .float32)
        let tokenError = compare(tokens, decode["tokens"])
        print("tokens", tokenError)
        #expect(tokenError.cosine > 0.99999)

        let raw = model.network.decode(
            queries, encoded: encoded, padded: QueryEmbedding.padImages(encoded.images), dtype: .float32)
        for (key, value) in raw {
            let error = compare(value.reshaped(-1), decode["predictions.\(key)"].reshaped(-1))
            print(key, error)
            #expect(error.cosine > 0.9999, "\(key)")
        }
    }
}

import Testing
@testable import ArrowKit

@Suite("All-pixel tracking")
struct AllPixelsTests {
    @Test("Frames are seeded from the source forwards, then backwards")
    func order() {
        #expect(PixelCoverage.order(frames: 5, source: 2) == [2, 3, 4, 1, 0])
        #expect(PixelCoverage.order(frames: 3, source: 0) == [0, 1, 2])
    }

    @Test("A frame seeds its uncovered lattice once, row-major, and never its sky")
    func seeds() {
        var sky = Array(repeating: false, count: 2 * 6 * 4)
        sky[0] = true  // frame 0, pixel (0, 0)
        var coverage = PixelCoverage(width: 6, height: 4, frames: 2, stride: 2, sky: SkyMask(width: 6, height: 4, sky: sky))
        #expect(coverage.latticeCount == 6)
        #expect(coverage.seed(0) == [SIMD2(2, 0), SIMD2(4, 0), SIMD2(0, 2), SIMD2(2, 2), SIMD2(4, 2)])
        #expect(coverage.seed(0).isEmpty)
        #expect(coverage.seed(1).count == 6)
    }

    @Test("Where a track is seen covers the pixels within the radius, in every frame but its source")
    func mark() {
        let width = 12, height = 8
        var coverage = PixelCoverage(width: width, height: height, frames: 3, stride: 2)
        #expect(coverage.radius == 1)
        // One seed from frame 0: seen at pixel (4, 4) in frames 0 and 1, hidden in frame 2.
        var answers = Predictions.unanswered(3)
        let uv = SIMD2<Float>((4 + 0.5) / Float(width), (4 + 0.5) / Float(height))
        answers.uv = [uv, uv, uv]
        answers.visibility = [5, 5, -5]
        coverage.mark(answers, seeds: 1, source: 0)
        func covered(_ frame: Int) -> Set<SIMD2<Int>> {
            Set((0 ..< width * height).filter { coverage.covered[frame * width * height + $0] }.map { SIMD2($0 % width, $0 / width) })
        }
        #expect(covered(0).isEmpty)
        #expect(covered(1) == Set((3 ... 5).flatMap { y in (3 ... 5).map { SIMD2($0, y) } }))
        #expect(covered(2).isEmpty)
        // Frame 1's lattice loses the one point inside: (4, 4).
        let seeds = coverage.seed(1)
        #expect(seeds.count == coverage.latticeCount - 1 && !seeds.contains(SIMD2(4, 4)))
    }

    @Test("A track seen off the frame covers only what its radius reaches inside")
    func markEdges() {
        var coverage = PixelCoverage(width: 4, height: 4, frames: 2, stride: 2)
        var answers = Predictions.unanswered(4)
        // Seeds 0 and 1 from frame 0; in frame 1, seed 0 is just past the left edge, seed 1 far away.
        answers.uv = [.zero, .zero, SIMD2(-0.2, 0.5), SIMD2(40, 40)]
        answers.visibility = [1, 1, 1, 1]
        coverage.mark(answers, seeds: 2, source: 0)
        let row = (0 ..< 16).filter { coverage.covered[16 + $0] }
        // Pixel (-1, 2) reaches column 0 of rows 1–3.
        #expect(row == [4, 8, 12])
    }

    @Test("Blocks seeded frame by frame read back as one target-major track group")
    func blocks() {
        var blocks = TrackBlocks(frames: 3)
        blocks.append(QueryBuilder.track([SIMD2(0, 0), SIMD2(2, 0)], source: 0, targets: [0, 1, 2], reference: 0), seeds: 2)
        blocks.append(QueryBuilder.track([SIMD2(4, 4)], source: 1, targets: [0, 1, 2], reference: 0), seeds: 1)
        #expect(blocks.total == 3)
        #expect(blocks.positions(0) == [0, 1, 3, 4, 6, 7])
        #expect(blocks.positions(1) == [2, 5, 8])
        var answers = Predictions.unanswered(3)
        answers.visibility = [10, 11, 12]
        blocks.answer(1, answers)
        let (queries, predictions) = blocks.merged()
        #expect(queries.target == [0, 0, 0, 1, 1, 1, 2, 2, 2])
        #expect(queries.source == [0, 0, 1, 0, 0, 1, 0, 0, 1])
        #expect(queries.xy[5] == SIMD2(4, 4))
        #expect(predictions.visibility[2] == 10 && predictions.visibility[8] == 12 && predictions.visibility[0].isNaN)
    }

    @Test("The track task is left to the runner when it tracks all pixels")
    func queries() {
        var settings = TaskSettings()
        settings.trackAllPixels = true
        let (queries, groups) = settings.queries([.track, .depth], frames: 2, width: 8, height: 8)
        #expect(groups.map(\.task) == [.depth])
        #expect(queries.count == groups[0].count)
    }
}

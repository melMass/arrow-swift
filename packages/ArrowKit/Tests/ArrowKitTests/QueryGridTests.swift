import Testing
@testable import ArrowKit

/// Values from upstream's `grid_axis` and `track_pixels` (arrow 6fca6f8).
@Suite("Seed grids")
struct QueryGridTests {
    @Test("Region axes match grid_axis: first and last pixel in, the rest spread evenly")
    func axes() {
        #expect(QueryGrid.spreadAxis(10, low: 0.2, high: 0.8, stride: 3) == [2, 5, 7])
        let wide = QueryGrid.spreadAxis(504, low: 0.1, high: 0.55, stride: 12)
        #expect(Array(wide.prefix(5)) == [50, 62, 74, 86, 98])
        #expect(wide.count == 20)
        #expect(QueryGrid.spreadAxis(280, low: 0.3, high: 0.31, stride: 8) == [85])
        #expect(QueryGrid.spreadAxis(50, low: 0, high: 1, stride: 0) == [])
    }

    @Test("Clicks join the grid as track_pixels does: clamped, deduplicated, sorted, nodes dropped")
    func seeds() {
        // Clicks given in model pixels: the centre twice, a grid node, a fractional point
        // and one past the bottom edge.
        let seeds = QueryGrid.seeds(
            width: 200, height: 100, stride: 10, region: SIMD4(0.1, 0.2, 0.5, 0.6),
            picks: [SIMD2(99.5, 49.5), SIMD2(99.5, 49.5), SIMD2(30, 30), SIMD2(30, 25), SIMD2(0, 119.5)])
        #expect(Array(seeds.suffix(4)) == [SIMD2(99, 59), SIMD2(0, 99), SIMD2(30, 25), SIMD2(99.5, 49.5)])
    }
}

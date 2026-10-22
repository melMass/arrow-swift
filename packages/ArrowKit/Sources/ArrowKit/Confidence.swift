import Foundation

/// Keeping the most confident points: what the viewers and the exports share.
public enum Confidence {
    /// The value at the `1 − keep` quantile: keeping values ≥ it keeps the top `keep`.
    /// −∞ (keep everything) at `keep` ≥ 1 or with no finite values.
    public static func threshold(_ values: [Float], keep: Float) -> Float {
        guard keep < 1 else { return -.infinity }
        let sorted = values.filter(\.isFinite).sorted()
        guard !sorted.isEmpty else { return -.infinity }
        return sorted[min(sorted.count - 1, Int(Float(sorted.count) * (1 - keep)))]
    }

    /// Upstream viewer's "Recommended" keep fraction (`viewer/confidence.py`): on log
    /// confidence, drop the lower tail past median − 3 · 1.4826 · MAD. When that drops
    /// nothing, split by Otsu's threshold over 128 bins instead, if the split is balanced
    /// (10–90 %) and explains ≥ 90 % of the variance. 1 when there is nothing to cut.
    public static func recommendedKeep(_ confidence: [Float]) -> Float {
        let values = confidence.filter { $0.isFinite && $0 > 0 }.map { Double(log($0)) }.sorted()
        guard let low = values.first, let high = values.last, high > low else { return 1 }
        func median(_ sorted: [Double]) -> Double {
            let n = sorted.count
            return n % 2 == 1 ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2
        }
        let center = median(values)
        let mad = median(values.map { abs($0 - center) }.sorted())
        let cut = center - 3 * 1.4826 * mad
        var keep = Double(values.filter { $0 >= cut }.count) / Double(values.count)
        if keep >= 1, let split = otsu(values, low: low, high: high) {
            keep = Double(values.filter { $0 >= split }.count) / Double(values.count)
        }
        return Float(keep)
    }

    /// Otsu's threshold over 128 bins, or nil when the best split is lopsided or weak.
    static func otsu(_ values: [Double], low: Double, high: Double) -> Double? {
        let bins = 128
        let width = (high - low) / Double(bins)
        var counts = [Double](repeating: 0, count: bins)
        for v in values { counts[min(bins - 1, Int((v - low) / width))] += 1 }
        let total = Double(values.count)
        let centers = (0 ..< bins).map { low + (Double($0) + 0.5) * width }
        let mean = zip(counts, centers).map { $0 * $1 }.reduce(0, +) / total
        let variance = zip(counts, centers).map { $0 * ($1 - mean) * ($1 - mean) }.reduce(0, +) / total
        guard variance > 0 else { return nil }
        var weight = 0.0, moment = 0.0
        var best = (between: -1.0, index: 0, weight: 0.0)
        for i in 0 ..< bins - 1 {
            weight += counts[i] / total
            moment += counts[i] / total * centers[i]
            guard weight > 0, weight < 1 else { continue }
            let between = (mean * weight - moment) * (mean * weight - moment) / (weight * (1 - weight))
            if between > best.between { best = (between, i, weight) }
        }
        guard best.weight >= 0.1, best.weight <= 0.9, best.between >= 0.9 * variance else { return nil }
        return low + Double(best.index + 1) * width
    }
}

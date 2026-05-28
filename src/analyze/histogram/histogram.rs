//! The [`Histogram`] type — a typed container for per-channel bin counts.
//!
//! `Histogram<S, V>` pairs a [`BinningStrategy<V>`] instance with the
//! counts it produced. Storing the strategy is deliberate: for data-
//! carrying strategies (`LinearBins`, `CustomBins`) the strategy *is*
//! the histogram's interpretive contract, and queries such as
//! `count_for(value)` or `bin_range(i)` are only meaningful relative
//! to it. See ADR-0040 §3 for the rationale.
//!
//! The second type parameter `V` is a phantom witness for the channel
//! value type the histogram was built from. It is never stored: it
//! exists only so that the type system can distinguish, for example,
//! `Histogram<NaturalBins, u8>` (built from `Indexed8`) from
//! `Histogram<NaturalBins, Saturating<u8>>` (built from `Mono8` /
//! `Rgb8` / …). Without `V`, the two would be the same type and
//! `bin_range(i)` would be ambiguous after ADR-0046.
//!

use core::marker::PhantomData;

use super::strategy::{BinIndex, BinningStrategy};

// ═══════════════════════════════════════════════════════════════════════════════
// Histogram<S, V>
// ═══════════════════════════════════════════════════════════════════════════════

/// A histogram of channel-value counts, parameterised by its binning
/// strategy `S` and the channel value type `V` it was built from.
///
/// # Type parameters
///
/// - `S` — the [`BinningStrategy`] used to classify values into bins.
///   The strategy instance is stored by value: see ADR-0040 §3.
/// - `V` — the channel value type. A phantom witness; never stored.
///   Distinguishes histograms built from `u8` (e.g. `Indexed8`) from
///   those built from `Saturating<u8>` (e.g. `Mono8`, `Rgb8`).
///
/// # Counters
///
/// Bin counts are `u64`. A 64 Mpx image with every pixel in one bin
/// produces 6.4 × 10⁷ — well within `u64` and an overflow of `u32`.
///
/// `nan_count`, `underflow_count`, and `overflow_count` are surfaced as
/// public fields rather than hidden by `count_for`. Philosophy §8
/// ("surface information, don't decide"): the library reports each
/// out-of-bin category and lets the caller decide what to do with them.
///
/// # Construction
///
/// `Histogram` is constructed only by the histogram engine. The
/// constructor `Histogram::new` is `pub(crate)` and lives in this
/// module; user code obtains a `Histogram` via the top-level
/// `histogram()` function (added in M4).
///
/// # Example
///
/// ```ignore
/// // (Engine entry point lands in M4; this is the M3 surface.)
/// use irys_cv::analyze::histogram::histogram::Histogram;
/// use irys_cv::analyze::histogram::strategy::NaturalBins;
///
/// let h: &Histogram<NaturalBins, u8> = /* … */;
/// assert_eq!(h.bins().len(), 256);
/// ```
#[derive(Debug, Clone)]
pub struct Histogram<S, V> {
    /// The strategy used to classify values into bins.
    strategy: S,

    /// In-range bin counts. `bins.len() == strategy.bin_count()` after
    /// construction.
    bins: Vec<u64>,

    /// Pixels whose value was `NaN`. Always 0 for integer strategies.
    pub nan_count: u64,

    /// Pixels strictly below the strategy's configured minimum. Always
    /// 0 for `NaturalBins`.
    pub underflow_count: u64,

    /// Pixels strictly above the strategy's configured maximum. Always
    /// 0 for `NaturalBins`.
    pub overflow_count: u64,

    /// Total pixels processed:
    /// `bins.iter().sum::<u64>() + nan_count + underflow_count + overflow_count`.
    pub total_count: u64,

    /// Phantom witness for the channel value type. Never stored —
    /// only inhabits the type system. The `fn() -> V` form keeps
    /// `Histogram<S, V>` covariant in `V` without imposing
    /// `Send` / `Sync` constraints derived from a stored `V`.
    _value: PhantomData<fn() -> V>,
}

impl<S, V> Histogram<S, V> {
    /// Constructs a histogram from a strategy and pre-tallied counters.
    ///
    /// `total_count` is computed from the supplied counters; callers do
    /// not pass it.
    ///
    /// This constructor is `pub(crate)` because the only valid producers
    /// are the histogram engine and the test code in this module. User
    /// code goes through the top-level `histogram()` function.
    #[allow(dead_code)] // M3: consumed by the engine in M4.
    pub(crate) fn new(
        strategy: S,
        bins: Vec<u64>,
        nan_count: u64,
        underflow_count: u64,
        overflow_count: u64,
    ) -> Self {
        let total_count =
            bins.iter().copied().sum::<u64>() + nan_count + underflow_count + overflow_count;

        Self {
            strategy,
            bins,
            nan_count,
            underflow_count,
            overflow_count,
            total_count,
            _value: PhantomData,
        }
    }
}

// ── Strategy-independent queries ────────────────────────────────────────────
//
// These methods only inspect `bins` and the stored strategy's identity;
// they do not require `S: BinningStrategy<V>`. Splitting them out keeps
// the trait bounds minimal: e.g. `bins()` and `cumulative()` are usable
// on a `Histogram<UserStrategy, V>` even before the user supplies the
// matching `BinningStrategy<V>` impl.

impl<S, V> Histogram<S, V> {
    /// Raw bin counts, one entry per in-range bin.
    #[inline]
    pub fn bins(&self) -> &[u64] {
        &self.bins
    }

    /// The count for the bin at position `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.bins().len()` (Tier 3 — programmer
    /// bug, per ADR-0025).
    #[inline]
    pub fn count_at_bin(&self, index: usize) -> u64 {
        assert!(
            index < self.bins.len(),
            "Histogram::count_at_bin: index {} out of range (bin_count = {})",
            index,
            self.bins.len()
        );
        self.bins[index]
    }

    /// The binning strategy used to construct this histogram.
    #[inline]
    pub fn strategy(&self) -> &S {
        &self.strategy
    }

    /// Computes the cumulative histogram.
    ///
    /// `result[i]` is the number of pixels whose bin index is `≤ i`.
    /// NaN, underflow, and overflow counts are *not* included — they
    /// are not in-range bins. Callers that need a true total reach for
    /// `total_count` directly.
    pub fn cumulative(&self) -> Vec<u64> {
        let mut out = self.bins.clone();
        for i in 1..out.len() {
            out[i] += out[i - 1];
        }
        out
    }
}

// ── Strategy-aware queries ──────────────────────────────────────────────────
//
// These methods delegate to `BinningStrategy<V>` and therefore require
// the impl to exist. They are the user-facing way to round-trip a
// channel value through the histogram or to recover a bin's edges.

impl<S, V> Histogram<S, V>
where
    V: Copy,
    S: BinningStrategy<V>,
{
    /// Count for the bin that `value` maps to.
    ///
    /// Returns `Some(count)` only when the value lies in an in-range
    /// bin (Tier 1 absence per ADR-0025). Returns `None` for `NaN`,
    /// underflow, and overflow — callers that need the precise
    /// category read [`nan_count`](Self::nan_count),
    /// [`underflow_count`](Self::underflow_count), or
    /// [`overflow_count`](Self::overflow_count) directly.
    #[inline]
    pub fn count_for(&self, value: V) -> Option<u64> {
        match self.strategy.bin_index(value) {
            BinIndex::In(i) => Some(self.bins[i]),
            BinIndex::Underflow | BinIndex::Overflow | BinIndex::Nan => None,
        }
    }

    /// Value range `[lower, upper]` covered by bin `index`.
    ///
    /// Delegates to [`BinningStrategy::bin_range`]. The return type is
    /// the strategy's [`Range`](BinningStrategy::Range), which may
    /// differ from `V` (e.g. `LinearBins` reports `f64` edges even for
    /// `Saturating<u16>` channels).
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.bins().len()` (Tier 3 — programmer
    /// bug, per ADR-0025).
    #[inline]
    pub fn bin_range(&self, index: usize) -> (S::Range, S::Range) {
        self.strategy.bin_range(index)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyze::histogram::strategy::{CustomBins, LinearBins, NaturalBins};
    use std::num::Saturating;

    // ── Construction ────────────────────────────────────────────────────────

    #[test]
    fn new_computes_total_count_from_counters() {
        let h: Histogram<NaturalBins, u8> = Histogram::new(NaturalBins, vec![1, 2, 3, 4], 5, 6, 7);
        assert_eq!(h.total_count, 1 + 2 + 3 + 4 + 5 + 6 + 7);
    }

    #[test]
    fn new_stores_supplied_counters_verbatim() {
        let h: Histogram<NaturalBins, u8> = Histogram::new(NaturalBins, vec![10, 20], 1, 2, 3);
        assert_eq!(h.nan_count, 1);
        assert_eq!(h.underflow_count, 2);
        assert_eq!(h.overflow_count, 3);
    }

    #[test]
    fn new_with_empty_bins_has_only_outlier_total() {
        let h: Histogram<NaturalBins, u8> = Histogram::new(NaturalBins, vec![], 4, 5, 6);
        assert_eq!(h.total_count, 15);
        assert!(h.bins().is_empty());
    }

    // ── bins() / count_at_bin() ─────────────────────────────────────────────

    #[test]
    fn bins_returns_supplied_slice() {
        let h: Histogram<NaturalBins, u8> = Histogram::new(NaturalBins, vec![7, 8, 9], 0, 0, 0);
        assert_eq!(h.bins(), &[7, 8, 9]);
    }

    #[test]
    fn count_at_bin_returns_value_at_index() {
        let h: Histogram<NaturalBins, u8> = Histogram::new(NaturalBins, vec![7, 8, 9], 0, 0, 0);
        assert_eq!(h.count_at_bin(0), 7);
        assert_eq!(h.count_at_bin(1), 8);
        assert_eq!(h.count_at_bin(2), 9);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn count_at_bin_panics_when_out_of_range() {
        let h: Histogram<NaturalBins, u8> = Histogram::new(NaturalBins, vec![7, 8, 9], 0, 0, 0);
        let _ = h.count_at_bin(3);
    }

    // ── strategy() ──────────────────────────────────────────────────────────

    #[test]
    fn strategy_returns_supplied_strategy() {
        let s = LinearBins {
            min: 0.0,
            max: 1.0,
            bin_count: 4,
        };
        let h: Histogram<LinearBins, f32> = Histogram::new(s, vec![0; 4], 0, 0, 0);
        assert_eq!(
            *h.strategy(),
            LinearBins {
                min: 0.0,
                max: 1.0,
                bin_count: 4
            }
        );
    }

    // ── cumulative() ────────────────────────────────────────────────────────

    #[test]
    fn cumulative_on_empty_bins_is_empty() {
        let h: Histogram<NaturalBins, u8> = Histogram::new(NaturalBins, vec![], 0, 0, 0);
        assert!(h.cumulative().is_empty());
    }

    #[test]
    fn cumulative_on_single_bin_is_identity() {
        let h: Histogram<NaturalBins, u8> = Histogram::new(NaturalBins, vec![42], 0, 0, 0);
        assert_eq!(h.cumulative(), vec![42]);
    }

    #[test]
    fn cumulative_on_multiple_bins_is_running_sum() {
        let h: Histogram<NaturalBins, u8> = Histogram::new(NaturalBins, vec![1, 2, 3, 4], 0, 0, 0);
        assert_eq!(h.cumulative(), vec![1, 3, 6, 10]);
    }

    #[test]
    fn cumulative_excludes_outlier_counters() {
        // 99 NaN / underflow / overflow pixels do not appear in
        // `cumulative()` because they are not in-range bins.
        let h: Histogram<NaturalBins, u8> = Histogram::new(NaturalBins, vec![1, 2, 3], 99, 99, 99);
        assert_eq!(h.cumulative(), vec![1, 3, 6]);
    }

    // ── count_for() (strategy-aware) ────────────────────────────────────────

    #[test]
    fn count_for_natural_u8_returns_some_for_in_range() {
        // bins[42] = 7 means seven pixels had value 42.
        let mut bins = vec![0u64; 256];
        bins[42] = 7;
        let h: Histogram<NaturalBins, u8> = Histogram::new(NaturalBins, bins, 0, 0, 0);
        assert_eq!(h.count_for(42_u8), Some(7));
        assert_eq!(h.count_for(0_u8), Some(0));
    }

    #[test]
    fn count_for_natural_satu8_dispatches_through_wrapper_impl() {
        let mut bins = vec![0u64; 256];
        bins[200] = 11;
        let h: Histogram<NaturalBins, Saturating<u8>> = Histogram::new(NaturalBins, bins, 0, 0, 0);
        assert_eq!(h.count_for(Saturating(200_u8)), Some(11));
    }

    #[test]
    fn count_for_returns_none_for_underflow_overflow_nan() {
        let s = LinearBins {
            min: 0.0,
            max: 1.0,
            bin_count: 4,
        };
        let h: Histogram<LinearBins, f32> = Histogram::new(s, vec![1, 1, 1, 1], 0, 0, 0);
        assert_eq!(h.count_for(-0.5_f32), None); // underflow
        assert_eq!(h.count_for(1.5_f32), None); // overflow
        assert_eq!(h.count_for(f32::NAN), None); // NaN
    }

    // ── bin_range() (strategy-aware) ────────────────────────────────────────

    #[test]
    fn bin_range_delegates_to_strategy_natural() {
        let h: Histogram<NaturalBins, u8> = Histogram::new(NaturalBins, vec![0; 256], 0, 0, 0);
        assert_eq!(h.bin_range(17), (17_u8, 17_u8));
    }

    #[test]
    fn bin_range_delegates_to_strategy_linear() {
        let s = LinearBins {
            min: 0.0,
            max: 1.0,
            bin_count: 4,
        };
        let h: Histogram<LinearBins, f32> = Histogram::new(s, vec![0; 4], 0, 0, 0);
        let (lo, hi) = h.bin_range(0);
        assert_eq!(lo, 0.0);
        assert_eq!(hi, 0.25);
        let (_, hi_last) = h.bin_range(3);
        assert_eq!(hi_last, 1.0);
    }

    #[test]
    fn bin_range_delegates_to_strategy_custom() {
        let s = CustomBins {
            edges: vec![0.0, 0.5, 1.0, 4.0],
        };
        let h: Histogram<CustomBins, f64> = Histogram::new(s, vec![0; 3], 0, 0, 0);
        assert_eq!(h.bin_range(0), (0.0, 0.5));
        assert_eq!(h.bin_range(2), (1.0, 4.0));
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn bin_range_panics_when_strategy_panics() {
        let s = LinearBins {
            min: 0.0,
            max: 1.0,
            bin_count: 4,
        };
        let h: Histogram<LinearBins, f32> = Histogram::new(s, vec![0; 4], 0, 0, 0);
        let _ = h.bin_range(4);
    }

    // ── Type-level: V distinguishes histograms with the same S ─────────────

    #[test]
    fn v_phantom_distinguishes_u8_and_satu8_histograms() {
        // This test exists to *document* a compile-time invariant: the
        // two histograms below have different types even though they
        // share the same strategy `S = NaturalBins`. If that ever
        // stops being true, the assertion in the doc above is wrong
        // and ADR-0040 §3 needs revisiting.
        let h_idx: Histogram<NaturalBins, u8> = Histogram::new(NaturalBins, vec![0; 256], 0, 0, 0);
        let h_mono: Histogram<NaturalBins, Saturating<u8>> =
            Histogram::new(NaturalBins, vec![0; 256], 0, 0, 0);

        // We can call the strategy-aware query on each with its own
        // value type, but we *cannot* mix them — that's the point.
        assert_eq!(h_idx.count_for(0_u8), Some(0));
        assert_eq!(h_mono.count_for(Saturating(0_u8)), Some(0));
    }
}

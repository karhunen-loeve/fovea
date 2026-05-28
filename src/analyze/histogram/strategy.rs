//! Binning strategies for histograms.
//!
//! This module implements the [`BinningStrategy`] trait, the [`BinIndex`]
//! classification enum, and the three concrete strategies — [`NaturalBins`],
//! [`LinearBins`], and [`CustomBins`] — described in ADR-0040 and in
//! `HISTOGRAM_PLAN.md` §3–§5.
//!
//! ## Pixel-model context
//!
//! Strategies bind on channel values, never on pixels. The trait bound on
//! `V` is just `Copy`; pixel-role traits (`PlainPixel`, `LinearPixel`,
//! `LinearSpace`) are deliberately *not* required. After ADR-0044 / 0046,
//! `f32` / `f64` are valid channels (e.g. for `MonoF32`, `RgbF32`) but not
//! pixels — and `BinningStrategy<f32>` / `BinningStrategy<f64>` impls
//! exist for exactly that reason.
//!
//! See ADR-0040 §1 for the trait shape and ADR-0040 §2 for each concrete
//! strategy's intended semantics.

use std::num::Saturating;

use crate::Error;
use crate::pixel::Mono;

// ═══════════════════════════════════════════════════════════════════════════════
// BinIndex
// ═══════════════════════════════════════════════════════════════════════════════

/// Outcome of classifying a channel value into a strategy's bins.
///
/// Strategies own classification logic; the histogram engine never
/// re-checks ranges or NaN. The four-way enum makes every outcome a
/// distinct, named case in the type system, in line with Philosophy §1
/// ("types are the spec") and §8 ("surface information, don't decide").
///
/// # Variants
///
/// - [`In(i)`](BinIndex::In) — value falls inside bin `i`, where
///   `i < strategy.bin_count()`.
/// - [`Underflow`](BinIndex::Underflow) — value lies below the strategy's
///   configured range.
/// - [`Overflow`](BinIndex::Overflow) — value lies above the strategy's
///   configured range.
/// - [`Nan`](BinIndex::Nan) — value is `NaN`. Only float-valued strategy
///   impls ever produce this; integer impls never do.
///
/// # Example
///
/// ```
/// use irys_cv::analyze::histogram::strategy::{BinIndex, BinningStrategy, NaturalBins};
///
/// assert_eq!(NaturalBins.bin_index(17_u8), BinIndex::In(17));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinIndex {
    /// Value falls inside an in-range bin.
    In(usize),
    /// Value is below the strategy's configured minimum.
    Underflow,
    /// Value is above the strategy's configured maximum.
    Overflow,
    /// Value is `NaN`. Only float-valued impls produce this.
    Nan,
}

// ═══════════════════════════════════════════════════════════════════════════════
// BinningStrategy
// ═══════════════════════════════════════════════════════════════════════════════

/// A scheme that maps channel values to bin indices.
///
/// `BinningStrategy` is the analytic engine of a histogram: given a channel
/// value of type `V`, it answers "which bin?" and reports the value range
/// covered by each bin. The histogram itself ([`super::histogram::Histogram`])
/// stores counts and the strategy instance; the strategy is the only
/// component that knows how to classify a value or reproduce a bin's edges.
///
/// # Trait bounds
///
/// - `V: Copy` — values are passed by value into the per-pixel hot loop.
///   No pixel-role traits are required: the strategy is independent of
///   pixel layout, gamma, or linear-space membership.
/// - [`Range`](BinningStrategy::Range) is the type returned by
///   [`bin_range`](BinningStrategy::bin_range). For [`NaturalBins`] this is
///   `V` itself; for [`LinearBins`] and [`CustomBins`] it is `f64`, since
///   bin edges are mathematically real numbers and must not be silently
///   rounded into a `Saturating<u16>` / `Mono<BITS>` value.
///
/// # Default `validate`
///
/// Strategies whose configuration cannot be invalid (e.g. [`NaturalBins`])
/// inherit the default `Ok(())`. Data-carrying strategies override and
/// return [`Error::InvalidBinningStrategy`] on bad input. The `histogram()`
/// engine calls `validate()` exactly once before any per-pixel work.
///
/// See ADR-0025 for the three-tier error convention this trait participates
/// in (`validate` is Tier 2; `bin_range` out-of-range is Tier 3).
pub trait BinningStrategy<V: Copy> {
    /// The numeric type used to report bin edges via
    /// [`bin_range`](BinningStrategy::bin_range).
    ///
    /// Equal to `V` for natural integer bins; equal to `f64` for the
    /// linear and custom strategies, whose edges are real-valued.
    type Range: Copy;

    /// The number of in-range bins the strategy produces.
    ///
    /// Must be `> 0` after `validate()` succeeds.
    fn bin_count(&self) -> usize;

    /// Classifies a channel value into one of the four [`BinIndex`]
    /// outcomes.
    fn bin_index(&self, value: V) -> BinIndex;

    /// Returns the `[lower, upper]` value range covered by bin `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.bin_count()` (Tier 3 — programmer bug,
    /// per ADR-0025).
    fn bin_range(&self, index: usize) -> (Self::Range, Self::Range);

    /// Validates the strategy's configuration.
    ///
    /// The default impl returns `Ok(())` — appropriate for parameter-free
    /// strategies such as [`NaturalBins`]. Data-carrying strategies override
    /// to reject invalid configurations (non-finite bounds, empty bin
    /// counts, non-monotonic edges, …) with
    /// [`Error::InvalidBinningStrategy`].
    fn validate(&self) -> Result<(), Error> {
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// NaturalBins
// ═══════════════════════════════════════════════════════════════════════════════

/// One bin per integer value of an 8-bit channel.
///
/// `NaturalBins` is a zero-sized strategy that produces 256 bins, with
/// `bin_index(v)` mapping a channel value `v` directly to bin `v`. It is
/// the canonical strategy for `Mono8` / `Rgb8` / `Indexed8` / similar
/// 8-bit pixel families.
///
/// # Implemented value types
///
/// `NaturalBins` implements [`BinningStrategy`] for exactly two channel
/// types in M2:
///
/// - [`u8`] — used by `Indexed8` and bare `u8` images.
/// - [`Saturating<u8>`] — used by `Mono8`, `Rgb8`, `Rgba8`, `Bgr8`, `Bgra8`,
///   etc.
///
/// Wider integer channels (`u16`, `u32`, `Saturating<u16>`,
/// `Saturating<u32>`, …) are *intentionally not implemented*: a 65 536-bin
/// or 4 G-bin histogram is rarely what the caller wants, so the compile
/// error forces them to choose [`LinearBins`] or [`CustomBins`] explicitly.
/// The same reasoning excludes `Mono<10>` / `Mono<12>` / `Mono<14>` here:
/// their valid-range invariant is a pixel-level concern, not visible from
/// the shared `Saturating<u16>` channel type alone.
///
/// # Example
///
/// ```
/// use irys_cv::analyze::histogram::strategy::{BinIndex, BinningStrategy, NaturalBins};
///
/// assert_eq!(<NaturalBins as BinningStrategy<u8>>::bin_count(&NaturalBins), 256);
/// assert_eq!(NaturalBins.bin_index(0_u8), BinIndex::In(0));
/// assert_eq!(NaturalBins.bin_index(255_u8), BinIndex::In(255));
/// assert_eq!(
///     <NaturalBins as BinningStrategy<u8>>::bin_range(&NaturalBins, 42),
///     (42_u8, 42_u8)
/// );
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NaturalBins;

impl BinningStrategy<u8> for NaturalBins {
    type Range = u8;

    #[inline]
    fn bin_count(&self) -> usize {
        256
    }

    #[inline]
    fn bin_index(&self, value: u8) -> BinIndex {
        BinIndex::In(value as usize)
    }

    #[inline]
    fn bin_range(&self, index: usize) -> (u8, u8) {
        assert!(
            index < 256,
            "NaturalBins::bin_range: index {} out of range (bin_count = 256)",
            index
        );
        let v = index as u8;
        (v, v)
    }
}

impl BinningStrategy<Saturating<u8>> for NaturalBins {
    type Range = Saturating<u8>;

    #[inline]
    fn bin_count(&self) -> usize {
        256
    }

    #[inline]
    fn bin_index(&self, value: Saturating<u8>) -> BinIndex {
        BinIndex::In(value.0 as usize)
    }

    #[inline]
    fn bin_range(&self, index: usize) -> (Saturating<u8>, Saturating<u8>) {
        assert!(
            index < 256,
            "NaturalBins::bin_range: index {} out of range (bin_count = 256)",
            index
        );
        let v = Saturating(index as u8);
        (v, v)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// LinearBins
// ═══════════════════════════════════════════════════════════════════════════════

/// Uniformly-spaced bins over a caller-specified `[min, max]` range.
///
/// Each bin covers an interval of width `(max - min) / bin_count`. Values
/// at exactly `max` are placed in the last bin; values strictly below
/// `min` produce [`BinIndex::Underflow`]; values strictly above `max`
/// produce [`BinIndex::Overflow`]; `NaN` produces [`BinIndex::Nan`] for
/// float impls.
///
/// # Why `f64` for `min` / `max`
///
/// `f32`'s 24-bit mantissa silently coarsens `u32` (and edge-case `u16`)
/// histograms — a Philosophy §4 violation, since the discretisation would
/// happen without the caller naming it. `f64` represents every `u32` and
/// every `f32` value exactly. The cost is 8 bytes per strategy instance.
///
/// # Implemented value types
///
/// - `f32`, `f64`
/// - `u8`, `u16`, `u32`
/// - `Saturating<u8>`, `Saturating<u16>`, `Saturating<u32>`
/// - `Mono<BITS>` (covers `Rgb<BITS>` / `Rgba<BITS>` / `Bgr<BITS>` /
///   `Bgra<BITS>` channels, 10/12/14 bits)
///
/// `u64` and `Saturating<u64>` are intentionally not implemented: `f64`
/// cannot represent every `u64` exactly. Add a wider strategy later if
/// the use case appears.
///
/// # Example
///
/// ```
/// use irys_cv::analyze::histogram::strategy::{BinIndex, BinningStrategy, LinearBins};
///
/// let s = LinearBins { min: 0.0, max: 1.0, bin_count: 4 };
/// assert_eq!(s.bin_index(0.0_f32), BinIndex::In(0));
/// assert_eq!(s.bin_index(0.25_f32), BinIndex::In(1));
/// assert_eq!(s.bin_index(1.0_f32), BinIndex::In(3));   // max → last bin
/// assert_eq!(s.bin_index(-0.1_f32), BinIndex::Underflow);
/// assert_eq!(s.bin_index(1.1_f32), BinIndex::Overflow);
/// assert_eq!(s.bin_index(f32::NAN), BinIndex::Nan);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearBins {
    /// Inclusive lower edge of the histogram range.
    pub min: f64,
    /// Inclusive upper edge of the histogram range.
    pub max: f64,
    /// Number of uniformly-spaced bins.
    pub bin_count: usize,
}

impl LinearBins {
    /// Shared validation used by every `BinningStrategy<V>` impl.
    fn validate_self(&self) -> Result<(), Error> {
        if !self.min.is_finite() {
            return Err(Error::InvalidBinningStrategy(format!(
                "LinearBins: min is not finite ({})",
                self.min
            )));
        }
        if !self.max.is_finite() {
            return Err(Error::InvalidBinningStrategy(format!(
                "LinearBins: max is not finite ({})",
                self.max
            )));
        }
        if self.min >= self.max {
            return Err(Error::InvalidBinningStrategy(format!(
                "LinearBins: min ({}) must be strictly less than max ({})",
                self.min, self.max
            )));
        }
        if self.bin_count == 0 {
            return Err(Error::InvalidBinningStrategy(
                "LinearBins: bin_count must be > 0".to_string(),
            ));
        }
        Ok(())
    }

    /// Shared `bin_range(i)` math; panics if `index >= bin_count`.
    #[inline]
    fn linear_bin_range(&self, index: usize) -> (f64, f64) {
        assert!(
            index < self.bin_count,
            "LinearBins::bin_range: index {} out of range (bin_count = {})",
            index,
            self.bin_count
        );
        let step = (self.max - self.min) / self.bin_count as f64;
        let lower = self.min + index as f64 * step;
        // Pin the last bin's upper edge exactly to `max` to avoid
        // accumulated FP drift.
        let upper = if index + 1 == self.bin_count {
            self.max
        } else {
            self.min + (index + 1) as f64 * step
        };
        (lower, upper)
    }
}

/// Shared classifier for `LinearBins` impls. Treats `NaN` as `Nan`,
/// values strictly outside `[min, max]` as `Underflow` / `Overflow`,
/// and `value == max` as the last bin.
#[inline]
fn classify_linear(value: f64, min: f64, max: f64, bin_count: usize) -> BinIndex {
    // `NaN` fails every comparison, so the explicit check is necessary.
    if value.is_nan() {
        return BinIndex::Nan;
    }
    if value < min {
        return BinIndex::Underflow;
    }
    if value > max {
        return BinIndex::Overflow;
    }
    if value == max {
        return BinIndex::In(bin_count - 1);
    }
    let t = (value - min) / (max - min);
    let i = (t * bin_count as f64).floor() as usize;
    BinIndex::In(i.min(bin_count - 1))
}

// Internal trait: widen each supported channel type to `f64` for
// classification. This keeps the per-impl bodies one-liners and
// guarantees every channel type widens through exactly one path.
trait WidenF64 {
    fn widen_f64(self) -> f64;
}

impl WidenF64 for f32 {
    #[inline]
    fn widen_f64(self) -> f64 {
        self as f64
    }
}
impl WidenF64 for f64 {
    #[inline]
    fn widen_f64(self) -> f64 {
        self
    }
}
impl WidenF64 for u8 {
    #[inline]
    fn widen_f64(self) -> f64 {
        self as f64
    }
}
impl WidenF64 for u16 {
    #[inline]
    fn widen_f64(self) -> f64 {
        self as f64
    }
}
impl WidenF64 for u32 {
    #[inline]
    fn widen_f64(self) -> f64 {
        self as f64
    }
}
impl WidenF64 for Saturating<u8> {
    #[inline]
    fn widen_f64(self) -> f64 {
        self.0 as f64
    }
}
impl WidenF64 for Saturating<u16> {
    #[inline]
    fn widen_f64(self) -> f64 {
        self.0 as f64
    }
}
impl WidenF64 for Saturating<u32> {
    #[inline]
    fn widen_f64(self) -> f64 {
        self.0 as f64
    }
}
impl<const BITS: usize> WidenF64 for Mono<BITS> {
    #[inline]
    fn widen_f64(self) -> f64 {
        self.value() as f64
    }
}

// `LinearBins` impls. The body of every impl is identical modulo the
// channel type — they all widen to `f64` and call `classify_linear`.
// A blanket `impl<V: WidenF64 + Copy> BinningStrategy<V> for LinearBins`
// would be cleaner, but it would also lock the strategy's value-type
// inventory behind a public-ish trait. The explicit list keeps the
// supported channel types visible and reviewable.
macro_rules! impl_linear_for {
    ($t:ty) => {
        impl BinningStrategy<$t> for LinearBins {
            type Range = f64;

            #[inline]
            fn bin_count(&self) -> usize {
                self.bin_count
            }

            #[inline]
            fn bin_index(&self, value: $t) -> BinIndex {
                classify_linear(value.widen_f64(), self.min, self.max, self.bin_count)
            }

            #[inline]
            fn bin_range(&self, index: usize) -> (f64, f64) {
                self.linear_bin_range(index)
            }

            fn validate(&self) -> Result<(), Error> {
                self.validate_self()
            }
        }
    };
}

impl_linear_for!(f32);
impl_linear_for!(f64);
impl_linear_for!(u8);
impl_linear_for!(u16);
impl_linear_for!(u32);
impl_linear_for!(Saturating<u8>);
impl_linear_for!(Saturating<u16>);
impl_linear_for!(Saturating<u32>);

impl<const BITS: usize> BinningStrategy<Mono<BITS>> for LinearBins {
    type Range = f64;

    #[inline]
    fn bin_count(&self) -> usize {
        self.bin_count
    }

    #[inline]
    fn bin_index(&self, value: Mono<BITS>) -> BinIndex {
        classify_linear(value.widen_f64(), self.min, self.max, self.bin_count)
    }

    #[inline]
    fn bin_range(&self, index: usize) -> (f64, f64) {
        self.linear_bin_range(index)
    }

    fn validate(&self) -> Result<(), Error> {
        self.validate_self()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CustomBins
// ═══════════════════════════════════════════════════════════════════════════════

/// Non-uniform bins specified by an explicit, strictly-increasing edge
/// vector.
///
/// `edges.len() - 1` bins are produced; bin `i` covers
/// `[edges[i], edges[i + 1])`, except the final bin, which is closed on
/// both sides so that values exactly equal to `edges.last()` land in it.
///
/// # Implemented value types
///
/// Same matrix as [`LinearBins`]: `f32`, `f64`, `u8`, `u16`, `u32`,
/// `Saturating<u8>`, `Saturating<u16>`, `Saturating<u32>`, `Mono<BITS>`.
/// `u64` / `Saturating<u64>` are excluded for the same `f64`-precision
/// reason.
///
/// # Example
///
/// ```
/// use irys_cv::analyze::histogram::strategy::{BinIndex, BinningStrategy, CustomBins};
///
/// let s = CustomBins { edges: vec![0.0, 0.5, 1.0, 4.0] };
/// assert_eq!(<CustomBins as BinningStrategy<f32>>::bin_count(&s), 3);
/// assert_eq!(s.bin_index(0.25_f32), BinIndex::In(0));
/// assert_eq!(s.bin_index(0.5_f32), BinIndex::In(1));
/// assert_eq!(s.bin_index(4.0_f32), BinIndex::In(2));   // last edge → last bin
/// assert_eq!(s.bin_index(-1.0_f32), BinIndex::Underflow);
/// assert_eq!(s.bin_index(5.0_f32), BinIndex::Overflow);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct CustomBins {
    /// Strictly-increasing, finite bin edges. Must contain at least two
    /// elements after [`validate`](CustomBins::validate).
    pub edges: Vec<f64>,
}

impl CustomBins {
    /// Shared validation used by every `BinningStrategy<V>` impl.
    fn validate_self(&self) -> Result<(), Error> {
        if self.edges.len() < 2 {
            return Err(Error::InvalidBinningStrategy(format!(
                "CustomBins: need at least 2 edges, got {}",
                self.edges.len()
            )));
        }
        for (i, &e) in self.edges.iter().enumerate() {
            if !e.is_finite() {
                return Err(Error::InvalidBinningStrategy(format!(
                    "CustomBins: edge {} is not finite ({})",
                    i, e
                )));
            }
        }
        for w in self.edges.windows(2) {
            if w[0] >= w[1] {
                return Err(Error::InvalidBinningStrategy(format!(
                    "CustomBins: edges must be strictly increasing, found {} >= {}",
                    w[0], w[1]
                )));
            }
        }
        Ok(())
    }

    #[inline]
    fn custom_bin_range(&self, index: usize) -> (f64, f64) {
        let last_bin = self.edges.len() - 1;
        assert!(
            index < last_bin,
            "CustomBins::bin_range: index {} out of range (bin_count = {})",
            index,
            last_bin
        );
        (self.edges[index], self.edges[index + 1])
    }
}

#[inline]
fn classify_custom(value: f64, edges: &[f64]) -> BinIndex {
    if value.is_nan() {
        return BinIndex::Nan;
    }
    let last = edges.len() - 1;
    if value < edges[0] {
        return BinIndex::Underflow;
    }
    if value > edges[last] {
        return BinIndex::Overflow;
    }
    if value == edges[last] {
        return BinIndex::In(last - 1);
    }
    // `partition_point` returns the count of edges satisfying the
    // predicate. With `e <= value` that count, minus one, is the largest
    // index whose edge is `<= value` — i.e. the bin index.
    let i = edges.partition_point(|&e| e <= value) - 1;
    BinIndex::In(i)
}

macro_rules! impl_custom_for {
    ($t:ty) => {
        impl BinningStrategy<$t> for CustomBins {
            type Range = f64;

            #[inline]
            fn bin_count(&self) -> usize {
                self.edges.len() - 1
            }

            #[inline]
            fn bin_index(&self, value: $t) -> BinIndex {
                classify_custom(value.widen_f64(), &self.edges)
            }

            #[inline]
            fn bin_range(&self, index: usize) -> (f64, f64) {
                self.custom_bin_range(index)
            }

            fn validate(&self) -> Result<(), Error> {
                self.validate_self()
            }
        }
    };
}

impl_custom_for!(f32);
impl_custom_for!(f64);
impl_custom_for!(u8);
impl_custom_for!(u16);
impl_custom_for!(u32);
impl_custom_for!(Saturating<u8>);
impl_custom_for!(Saturating<u16>);
impl_custom_for!(Saturating<u32>);

impl<const BITS: usize> BinningStrategy<Mono<BITS>> for CustomBins {
    type Range = f64;

    #[inline]
    fn bin_count(&self) -> usize {
        self.edges.len() - 1
    }

    #[inline]
    fn bin_index(&self, value: Mono<BITS>) -> BinIndex {
        classify_custom(value.widen_f64(), &self.edges)
    }

    #[inline]
    fn bin_range(&self, index: usize) -> (f64, f64) {
        self.custom_bin_range(index)
    }

    fn validate(&self) -> Result<(), Error> {
        self.validate_self()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── BinIndex ────────────────────────────────────────────────────────────

    #[test]
    fn bin_index_variants_are_distinct() {
        assert_ne!(BinIndex::In(0), BinIndex::In(1));
        assert_ne!(BinIndex::In(0), BinIndex::Underflow);
        assert_ne!(BinIndex::Underflow, BinIndex::Overflow);
        assert_ne!(BinIndex::Overflow, BinIndex::Nan);
    }

    // ── NaturalBins on u8 ───────────────────────────────────────────────────

    #[test]
    fn natural_bins_u8_count_is_256() {
        assert_eq!(
            <NaturalBins as BinningStrategy<u8>>::bin_count(&NaturalBins),
            256
        );
    }

    #[test]
    fn natural_bins_u8_maps_every_value_to_its_index() {
        for v in 0u8..=255 {
            assert_eq!(
                <NaturalBins as BinningStrategy<u8>>::bin_index(&NaturalBins, v),
                BinIndex::In(v as usize)
            );
        }
    }

    #[test]
    fn natural_bins_u8_bin_range_matches_index() {
        assert_eq!(
            <NaturalBins as BinningStrategy<u8>>::bin_range(&NaturalBins, 0),
            (0_u8, 0_u8)
        );
        assert_eq!(
            <NaturalBins as BinningStrategy<u8>>::bin_range(&NaturalBins, 255),
            (255_u8, 255_u8)
        );
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn natural_bins_u8_bin_range_panics_at_256() {
        let _ = <NaturalBins as BinningStrategy<u8>>::bin_range(&NaturalBins, 256);
    }

    #[test]
    fn natural_bins_u8_validate_is_ok() {
        assert!(<NaturalBins as BinningStrategy<u8>>::validate(&NaturalBins).is_ok());
    }

    // ── NaturalBins on Saturating<u8> ───────────────────────────────────────

    #[test]
    fn natural_bins_satu8_maps_every_value_to_its_index() {
        for v in 0u8..=255 {
            let sv = Saturating(v);
            assert_eq!(
                <NaturalBins as BinningStrategy<Saturating<u8>>>::bin_index(&NaturalBins, sv),
                BinIndex::In(v as usize)
            );
        }
    }

    #[test]
    fn natural_bins_satu8_bin_range_matches_index() {
        assert_eq!(
            <NaturalBins as BinningStrategy<Saturating<u8>>>::bin_range(&NaturalBins, 0),
            (Saturating(0_u8), Saturating(0_u8))
        );
        assert_eq!(
            <NaturalBins as BinningStrategy<Saturating<u8>>>::bin_range(&NaturalBins, 255),
            (Saturating(255_u8), Saturating(255_u8))
        );
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn natural_bins_satu8_bin_range_panics_at_256() {
        let _ = <NaturalBins as BinningStrategy<Saturating<u8>>>::bin_range(&NaturalBins, 256);
    }

    // ── LinearBins: validation ──────────────────────────────────────────────

    fn lb(min: f64, max: f64, bin_count: usize) -> LinearBins {
        LinearBins {
            min,
            max,
            bin_count,
        }
    }

    #[test]
    fn linear_validate_accepts_well_formed() {
        assert!(<LinearBins as BinningStrategy<f32>>::validate(&lb(0.0, 1.0, 4)).is_ok());
    }

    #[test]
    fn linear_validate_rejects_non_finite_min() {
        let err =
            <LinearBins as BinningStrategy<f32>>::validate(&lb(f64::NAN, 1.0, 4)).unwrap_err();
        match err {
            Error::InvalidBinningStrategy(msg) => assert!(msg.contains("min")),
            _ => panic!("expected InvalidBinningStrategy"),
        }
    }

    #[test]
    fn linear_validate_rejects_non_finite_max() {
        let err =
            <LinearBins as BinningStrategy<f32>>::validate(&lb(0.0, f64::INFINITY, 4)).unwrap_err();
        match err {
            Error::InvalidBinningStrategy(msg) => assert!(msg.contains("max")),
            _ => panic!("expected InvalidBinningStrategy"),
        }
    }

    #[test]
    fn linear_validate_rejects_min_eq_max() {
        let err = <LinearBins as BinningStrategy<f32>>::validate(&lb(1.0, 1.0, 4)).unwrap_err();
        assert!(matches!(err, Error::InvalidBinningStrategy(_)));
    }

    #[test]
    fn linear_validate_rejects_min_gt_max() {
        let err = <LinearBins as BinningStrategy<f32>>::validate(&lb(2.0, 1.0, 4)).unwrap_err();
        assert!(matches!(err, Error::InvalidBinningStrategy(_)));
    }

    #[test]
    fn linear_validate_rejects_zero_bin_count() {
        let err = <LinearBins as BinningStrategy<f32>>::validate(&lb(0.0, 1.0, 0)).unwrap_err();
        match err {
            Error::InvalidBinningStrategy(msg) => assert!(msg.contains("bin_count")),
            _ => panic!("expected InvalidBinningStrategy"),
        }
    }

    // ── LinearBins: classification on f32 ───────────────────────────────────

    #[test]
    fn linear_f32_min_maps_to_first_bin() {
        let s = lb(0.0, 1.0, 4);
        assert_eq!(s.bin_index(0.0_f32), BinIndex::In(0));
    }

    #[test]
    fn linear_f32_max_maps_to_last_bin() {
        let s = lb(0.0, 1.0, 4);
        assert_eq!(s.bin_index(1.0_f32), BinIndex::In(3));
    }

    #[test]
    fn linear_f32_midpoint_maps_to_expected_bin() {
        let s = lb(0.0, 1.0, 4);
        // Bin 0 = [0.00, 0.25), Bin 1 = [0.25, 0.50), …
        assert_eq!(s.bin_index(0.0_f32), BinIndex::In(0));
        assert_eq!(s.bin_index(0.249_f32), BinIndex::In(0));
        assert_eq!(s.bin_index(0.25_f32), BinIndex::In(1));
        assert_eq!(s.bin_index(0.5_f32), BinIndex::In(2));
        assert_eq!(s.bin_index(0.75_f32), BinIndex::In(3));
    }

    #[test]
    fn linear_f32_below_range_is_underflow() {
        let s = lb(0.0, 1.0, 4);
        assert_eq!(s.bin_index(-0.001_f32), BinIndex::Underflow);
        assert_eq!(s.bin_index(f32::NEG_INFINITY), BinIndex::Underflow);
    }

    #[test]
    fn linear_f32_above_range_is_overflow() {
        let s = lb(0.0, 1.0, 4);
        assert_eq!(s.bin_index(1.001_f32), BinIndex::Overflow);
        assert_eq!(s.bin_index(f32::INFINITY), BinIndex::Overflow);
    }

    #[test]
    fn linear_f32_nan_is_nan() {
        let s = lb(0.0, 1.0, 4);
        assert_eq!(s.bin_index(f32::NAN), BinIndex::Nan);
    }

    #[test]
    fn linear_f64_classification_matches_f32() {
        let s = lb(0.0, 1.0, 4);
        assert_eq!(s.bin_index(0.5_f64), BinIndex::In(2));
        assert_eq!(s.bin_index(f64::NAN), BinIndex::Nan);
        assert_eq!(s.bin_index(-1.0_f64), BinIndex::Underflow);
        assert_eq!(s.bin_index(2.0_f64), BinIndex::Overflow);
    }

    // ── LinearBins: integer / wrapper channel impls ─────────────────────────

    #[test]
    fn linear_u8_classification() {
        let s = lb(0.0, 255.0, 4);
        assert_eq!(s.bin_index(0_u8), BinIndex::In(0));
        assert_eq!(s.bin_index(127_u8), BinIndex::In(1));
        assert_eq!(s.bin_index(255_u8), BinIndex::In(3));
    }

    #[test]
    fn linear_u16_classification() {
        let s = lb(0.0, 65535.0, 8);
        assert_eq!(s.bin_index(0_u16), BinIndex::In(0));
        assert_eq!(s.bin_index(65535_u16), BinIndex::In(7));
    }

    #[test]
    fn linear_u32_classification() {
        let s = lb(0.0, 4_294_967_295.0, 4);
        assert_eq!(s.bin_index(0_u32), BinIndex::In(0));
        assert_eq!(s.bin_index(u32::MAX), BinIndex::In(3));
    }

    #[test]
    fn linear_satu8_unwraps_correctly() {
        let s = lb(0.0, 255.0, 4);
        assert_eq!(s.bin_index(Saturating(0_u8)), BinIndex::In(0));
        assert_eq!(s.bin_index(Saturating(255_u8)), BinIndex::In(3));
    }

    #[test]
    fn linear_satu16_unwraps_correctly() {
        let s = lb(0.0, 65535.0, 4);
        assert_eq!(s.bin_index(Saturating(0_u16)), BinIndex::In(0));
        assert_eq!(s.bin_index(Saturating(65535_u16)), BinIndex::In(3));
    }

    #[test]
    fn linear_satu32_unwraps_correctly() {
        let s = lb(0.0, 4_294_967_295.0, 4);
        assert_eq!(s.bin_index(Saturating(0_u32)), BinIndex::In(0));
        assert_eq!(s.bin_index(Saturating(u32::MAX)), BinIndex::In(3));
    }

    #[test]
    fn linear_mono10_uses_value() {
        let s = lb(0.0, 1023.0, 4);
        assert_eq!(s.bin_index(Mono::<10>::new(0)), BinIndex::In(0));
        assert_eq!(s.bin_index(Mono::<10>::new(1023)), BinIndex::In(3));
    }

    #[test]
    fn linear_mono12_uses_value() {
        let s = lb(0.0, 4095.0, 4);
        assert_eq!(s.bin_index(Mono::<12>::new(0)), BinIndex::In(0));
        assert_eq!(s.bin_index(Mono::<12>::new(4095)), BinIndex::In(3));
    }

    #[test]
    fn linear_mono14_uses_value() {
        let s = lb(0.0, 16383.0, 4);
        assert_eq!(s.bin_index(Mono::<14>::new(0)), BinIndex::In(0));
        assert_eq!(s.bin_index(Mono::<14>::new(16383)), BinIndex::In(3));
    }

    // ── LinearBins: bin_range ───────────────────────────────────────────────

    #[test]
    fn linear_bin_range_returns_f64_edges() {
        let s = lb(0.0, 1.0, 4);
        let (lo, hi) = <LinearBins as BinningStrategy<f32>>::bin_range(&s, 0);
        assert_eq!(lo, 0.0);
        assert_eq!(hi, 0.25);
    }

    #[test]
    fn linear_bin_range_last_pinned_to_max() {
        let s = lb(0.0, 1.0, 3);
        let (_, hi) = <LinearBins as BinningStrategy<f32>>::bin_range(&s, 2);
        assert_eq!(hi, 1.0);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn linear_bin_range_panics_when_out_of_range() {
        let s = lb(0.0, 1.0, 4);
        let _ = <LinearBins as BinningStrategy<f32>>::bin_range(&s, 4);
    }

    // ── CustomBins: validation ──────────────────────────────────────────────

    fn cb(edges: &[f64]) -> CustomBins {
        CustomBins {
            edges: edges.to_vec(),
        }
    }

    #[test]
    fn custom_validate_accepts_well_formed() {
        assert!(<CustomBins as BinningStrategy<f32>>::validate(&cb(&[0.0, 1.0, 2.0])).is_ok());
    }

    #[test]
    fn custom_validate_rejects_zero_or_one_edge() {
        let err = <CustomBins as BinningStrategy<f32>>::validate(&cb(&[])).unwrap_err();
        assert!(matches!(err, Error::InvalidBinningStrategy(_)));

        let err = <CustomBins as BinningStrategy<f32>>::validate(&cb(&[1.0])).unwrap_err();
        assert!(matches!(err, Error::InvalidBinningStrategy(_)));
    }

    #[test]
    fn custom_validate_rejects_non_finite_edge() {
        let err =
            <CustomBins as BinningStrategy<f32>>::validate(&cb(&[0.0, f64::NAN, 1.0])).unwrap_err();
        match err {
            Error::InvalidBinningStrategy(msg) => assert!(msg.contains("finite")),
            _ => panic!("expected InvalidBinningStrategy"),
        }
    }

    #[test]
    fn custom_validate_rejects_non_increasing_edges() {
        let err =
            <CustomBins as BinningStrategy<f32>>::validate(&cb(&[0.0, 1.0, 1.0])).unwrap_err();
        assert!(matches!(err, Error::InvalidBinningStrategy(_)));

        let err =
            <CustomBins as BinningStrategy<f32>>::validate(&cb(&[0.0, 2.0, 1.0])).unwrap_err();
        assert!(matches!(err, Error::InvalidBinningStrategy(_)));
    }

    // ── CustomBins: classification ──────────────────────────────────────────

    #[test]
    fn custom_lower_edge_maps_to_that_bin() {
        let s = cb(&[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(s.bin_index(0.0_f32), BinIndex::In(0));
        assert_eq!(s.bin_index(1.0_f32), BinIndex::In(1));
        assert_eq!(s.bin_index(2.0_f32), BinIndex::In(2));
    }

    #[test]
    fn custom_final_edge_maps_to_last_bin() {
        let s = cb(&[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(s.bin_index(3.0_f32), BinIndex::In(2));
    }

    #[test]
    fn custom_below_first_edge_is_underflow() {
        let s = cb(&[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(s.bin_index(-0.001_f32), BinIndex::Underflow);
    }

    #[test]
    fn custom_above_final_edge_is_overflow() {
        let s = cb(&[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(s.bin_index(3.001_f32), BinIndex::Overflow);
    }

    #[test]
    fn custom_nan_is_nan_for_floats() {
        let s = cb(&[0.0, 1.0, 2.0]);
        assert_eq!(s.bin_index(f32::NAN), BinIndex::Nan);
        assert_eq!(s.bin_index(f64::NAN), BinIndex::Nan);
    }

    #[test]
    fn custom_interior_value_is_classified_by_partition() {
        let s = cb(&[0.0, 0.5, 1.0, 4.0]);
        assert_eq!(s.bin_index(0.25_f64), BinIndex::In(0));
        assert_eq!(s.bin_index(0.75_f64), BinIndex::In(1));
        assert_eq!(s.bin_index(2.5_f64), BinIndex::In(2));
    }

    #[test]
    fn custom_bin_count_matches_edges_minus_one() {
        let s = cb(&[0.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(<CustomBins as BinningStrategy<f32>>::bin_count(&s), 4);
    }

    #[test]
    fn custom_bin_range_returns_adjacent_edges() {
        let s = cb(&[0.0, 0.5, 1.0, 4.0]);
        assert_eq!(
            <CustomBins as BinningStrategy<f32>>::bin_range(&s, 0),
            (0.0, 0.5)
        );
        assert_eq!(
            <CustomBins as BinningStrategy<f32>>::bin_range(&s, 1),
            (0.5, 1.0)
        );
        assert_eq!(
            <CustomBins as BinningStrategy<f32>>::bin_range(&s, 2),
            (1.0, 4.0)
        );
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn custom_bin_range_panics_when_out_of_range() {
        let s = cb(&[0.0, 1.0, 2.0]);
        let _ = <CustomBins as BinningStrategy<f32>>::bin_range(&s, 2);
    }

    #[test]
    fn custom_mono_channels_use_value() {
        let s = cb(&[0.0, 512.0, 1024.0]);
        assert_eq!(s.bin_index(Mono::<10>::new(0)), BinIndex::In(0));
        assert_eq!(s.bin_index(Mono::<10>::new(600)), BinIndex::In(1));
        assert_eq!(s.bin_index(Mono::<10>::new(1023)), BinIndex::In(1));
    }

    #[test]
    fn custom_int_wrapper_channels_classify() {
        let s = cb(&[0.0, 100.0, 200.0]);
        assert_eq!(s.bin_index(Saturating(50_u8)), BinIndex::In(0));
        assert_eq!(s.bin_index(Saturating(150_u8)), BinIndex::In(1));
        assert_eq!(s.bin_index(Saturating(200_u8)), BinIndex::In(1)); // last edge
        assert_eq!(s.bin_index(Saturating(50_u16)), BinIndex::In(0));
        assert_eq!(s.bin_index(Saturating(150_u32)), BinIndex::In(1));
    }
}

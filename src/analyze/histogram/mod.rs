//! Per-channel value histograms.
//!
//! This module provides the concrete histogram API for per-channel image
//! analysis.
//!
//! # Surface
//!
//! - [`strategy`] — the [`BinningStrategy`] trait, the [`BinIndex`]
//!   classification enum, and the concrete [`NaturalBins`], [`LinearBins`],
//!   and [`CustomBins`] strategies.
//! - [`engine`] — the [`Histogram`] type itself.
//! - [`HistogramOutput`] — caller-driven output shape (single histogram,
//!   `Vec`, or fixed-size array).
//! - [`histogram`](self::histogram()) — the top-level entry point.
//!
//! # Why this lives under `analyze`
//!
//! Histograms produce data *about* an image, not a new image, so they
//! belong under [`crate::analyze`] rather than under `transform/`. See
//! the module-level docs of [`crate::analyze`] for the broader rationale.
//!
//! # Pixel-model context
//!
//! Histograms operate on [`HomogeneousPixel::Channel`] values, not on
//! pixels. That distinction matters: `f32` and `f64` are valid channel
//! types (e.g. for `MonoF32`, `RgbF32`)
//! but are not pixel types. Integer storage channels are typically
//! `Saturating<T>`, not bare `T`. The strategy trait binds on the channel
//! value directly, with no `LinearPixel` / `LinearSpace` requirement —
//! gamma-encoded sRGB and `Indexed8` images are valid histogram inputs.
//!
//! # Examples
//!
//! Single-channel natural histogram on `Mono8`:
//!
//! ```
//! use fovea::analyze::histogram::{histogram, Histogram, NaturalBins};
//! use fovea::image::Image;
//! use fovea::pixel::Mono8;
//!
//! let img = Image::from_vec(2, 2, vec![
//!     Mono8::new(0),
//!     Mono8::new(1),
//!     Mono8::new(1),
//!     Mono8::new(255),
//! ])?;
//!
//! let hist: Histogram<NaturalBins, _> = histogram(&img, &NaturalBins)?;
//! assert_eq!(hist.count_at_bin(1), 2);
//! # Ok::<(), fovea::Error>(())
//! ```
//!
//! RGB image as a fixed-size array of per-channel histograms:
//!
//! ```
//! use fovea::analyze::histogram::{histogram, Histogram, NaturalBins};
//! use fovea::image::Image;
//! use fovea::pixel::Rgb8;
//!
//! let img = Image::from_vec(1, 1, vec![Rgb8::new(10, 20, 30)])?;
//!
//! let [r, g, b]: [Histogram<NaturalBins, _>; 3] = histogram(&img, &NaturalBins)?;
//! assert_eq!(r.count_at_bin(10), 1);
//! assert_eq!(g.count_at_bin(20), 1);
//! assert_eq!(b.count_at_bin(30), 1);
//! # Ok::<(), fovea::Error>(())
//! ```
//!
//! Float-channel histogram with NaN counted but not silently lost:
//!
//! ```
//! use fovea::analyze::histogram::{histogram, Histogram, LinearBins};
//! use fovea::image::Image;
//! use fovea::pixel::MonoF32;
//!
//! let img = Image::from_vec(2, 1, vec![
//!     MonoF32::new(0.25),
//!     MonoF32::new(f32::NAN),
//! ])?;
//!
//! let strategy = LinearBins { min: 0.0, max: 1.0, bin_count: 4 };
//! let hist: Histogram<LinearBins, _> = histogram(&img, &strategy)?;
//!
//! assert_eq!(hist.count_at_bin(1), 1);
//! assert_eq!(hist.nan_count, 1);
//! # Ok::<(), fovea::Error>(())
//! ```
//!
//! [`HomogeneousPixel::Channel`]: crate::pixel::HomogeneousPixel::Channel
//! [`BinningStrategy`]: strategy::BinningStrategy
//! [`BinIndex`]: strategy::BinIndex
//! [`NaturalBins`]: strategy::NaturalBins
//! [`LinearBins`]: strategy::LinearBins
//! [`CustomBins`]: strategy::CustomBins
//! [`Histogram`]: engine::Histogram
//!
//! # Consumers
//!
//! Operations built on top of the histogram engine live alongside it:
//!
//! - [`otsu_threshold`] / [`otsu_binary_mask`] — Otsu's automatic
//!   threshold from a 256-bin histogram.
//! - [`equalization_lut`] / [`equalize_image`] /
//!   [`equalize_image_into`] — per-channel histogram equalization.

pub mod engine;
pub mod strategy;

pub mod equalize;
pub mod otsu;

pub use engine::Histogram;
pub use strategy::{BinIndex, BinningStrategy, CustomBins, LinearBins, NaturalBins};

pub use equalize::{equalization_lut, equalize_image, equalize_image_into};
pub use otsu::{otsu_binary_mask, otsu_threshold};

use crate::Error;
use crate::image::RasterImage;
use crate::pixel::HomogeneousPixel;

// ═══════════════════════════════════════════════════════════════════════════════
// HistogramOutput
// ═══════════════════════════════════════════════════════════════════════════════

/// Caller-driven output shape for [`histogram()`].
///
/// `histogram()` is generic over the output type; the same call site can
/// produce a single histogram, a `Vec`, or a fixed-size array depending
/// on what the caller asks for. This keeps multi-channel and
/// single-channel use cases on the same entry point.
///
/// # Implementations
///
/// - [`Histogram<S, V>`] — exactly one channel. Panics if the pixel type
///   has more than one channel (Tier 3 — programmer bug).
/// - [`Vec<Histogram<S, V>>`] — any channel count.
/// - [`[Histogram<S, V>; N]`] — exactly `N` channels. Panics on mismatch
///   (Tier 3).
///
/// # Why panics, not `Result`
///
/// The pixel type's channel count is a compile-time property of `P`. If
/// the caller asks for `[Histogram<_, _>; 4]` from an `Rgb8` image, that
/// mismatch is a programmer bug, not data-dependent failure, so it uses
/// `panic!` rather than `Result`.
pub trait HistogramOutput<S, V>: Sized {
    /// Builds the output shape by invoking `compute(channel_index)`
    /// once per channel.
    ///
    /// `channel_count` is `P::CHANNEL_COUNT` from the input image's
    /// pixel type. Implementations decide whether they accept the
    /// supplied channel count and panic otherwise.
    fn collect(
        channel_count: usize,
        compute: impl FnMut(usize) -> Result<Histogram<S, V>, Error>,
    ) -> Result<Self, Error>;
}

impl<S, V> HistogramOutput<S, V> for Histogram<S, V> {
    fn collect(
        channel_count: usize,
        mut compute: impl FnMut(usize) -> Result<Histogram<S, V>, Error>,
    ) -> Result<Self, Error> {
        assert_eq!(
            channel_count, 1,
            "histogram() called with output type `Histogram<S, V>` on a pixel with {} \
             channels; use `Vec<Histogram<S, V>>` or `[Histogram<S, V>; N]` instead",
            channel_count
        );
        compute(0)
    }
}

impl<S, V> HistogramOutput<S, V> for Vec<Histogram<S, V>> {
    fn collect(
        channel_count: usize,
        mut compute: impl FnMut(usize) -> Result<Histogram<S, V>, Error>,
    ) -> Result<Self, Error> {
        let mut out = Vec::with_capacity(channel_count);
        for c in 0..channel_count {
            out.push(compute(c)?);
        }
        Ok(out)
    }
}

impl<S, V, const N: usize> HistogramOutput<S, V> for [Histogram<S, V>; N] {
    fn collect(
        channel_count: usize,
        mut compute: impl FnMut(usize) -> Result<Histogram<S, V>, Error>,
    ) -> Result<Self, Error> {
        assert_eq!(
            channel_count, N,
            "histogram() called with output type `[Histogram<S, V>; {}]` on a pixel with \
             {} channels",
            N, channel_count
        );

        // Allocate a small `Vec` and fall back to `try_into` for the
        // array conversion. The allocation is bounded by the pixel's
        // channel count (≤ 4 in practice) and is not on the per-pixel
        // hot path.
        let mut tmp: Vec<Histogram<S, V>> = Vec::with_capacity(N);
        for c in 0..N {
            tmp.push(compute(c)?);
        }
        Ok(tmp.try_into().unwrap_or_else(|_| {
            // Unreachable: we just pushed exactly N elements.
            unreachable!("internal error: vec length != N after collect")
        }))
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// histogram() entry point
// ═══════════════════════════════════════════════════════════════════════════════

/// Computes a per-channel value histogram of an image.
///
/// The output shape is chosen by the caller via the [`HistogramOutput`]
/// trait — annotate the binding with `Histogram<S, _>`,
/// `Vec<Histogram<S, _>>`, or `[Histogram<S, _>; N]` and the engine
/// dispatches accordingly.
///
/// # Trait bounds
///
/// - `P: HomogeneousPixel` — pixels expose a uniform channel type and a
///   `CHANNEL_COUNT`. `LinearPixel` / `LinearSpace` are *not* required:
///   gamma-encoded sRGB and indexed pixels are legitimate histogram
///   inputs.
/// - `S: BinningStrategy<P::Channel> + Clone` — the strategy must
///   accept the pixel's channel type. `Clone` is required because the
///   engine stores the strategy in every produced [`Histogram`] (one
///   per channel).
/// - The image is bound on [`RasterImage`] (not `ImageView`) so the
///   engine can iterate row slices for cache-friendly access. Strided
///   ROIs (`SubView`) satisfy this and work as inputs.
///
/// # Errors
///
/// Returns [`Error::InvalidBinningStrategy`] if `strategy.validate()`
/// rejects the strategy's own configuration (Tier 2).
/// Validation runs once before any per-pixel work.
///
/// # Panics
///
/// - The selected [`HistogramOutput`] impl may panic on shape mismatch
///   (e.g. asking for a single `Histogram<_, _>` from a 3-channel
///   pixel). See the impl docs.
///
/// # Examples
///
/// See the module-level documentation for end-to-end examples.
pub fn histogram<P, S, O>(image: &impl RasterImage<Pixel = P>, strategy: &S) -> Result<O, Error>
where
    P: HomogeneousPixel,
    S: BinningStrategy<P::Channel> + Clone,
    O: HistogramOutput<S, P::Channel>,
{
    strategy.validate()?;

    O::collect(P::CHANNEL_COUNT, |channel| {
        Ok(compute_channel_histogram(image, channel, strategy))
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// Engine
// ═══════════════════════════════════════════════════════════════════════════════

/// Computes a single-channel histogram by iterating row slices.
///
/// Invariants assumed by the body:
///
/// - `strategy.validate()` already returned `Ok(())` — `bin_count() > 0`
///   for `LinearBins` / `CustomBins`.
/// - `channel < P::CHANNEL_COUNT` — guaranteed by the engine driver.
fn compute_channel_histogram<P, S>(
    image: &impl RasterImage<Pixel = P>,
    channel: usize,
    strategy: &S,
) -> Histogram<S, P::Channel>
where
    P: HomogeneousPixel,
    S: BinningStrategy<P::Channel> + Clone,
{
    let bin_count = strategy.bin_count();
    let mut bins = vec![0u64; bin_count];
    let mut nan_count: u64 = 0;
    let mut underflow_count: u64 = 0;
    let mut overflow_count: u64 = 0;

    for y in 0..image.height() {
        let row = image.row(y);
        for px in row {
            let value = px.channel(channel);
            match strategy.bin_index(value) {
                BinIndex::In(i) => {
                    debug_assert!(
                        i < bins.len(),
                        "BinningStrategy::bin_index returned In({}) but bin_count = {}",
                        i,
                        bins.len()
                    );
                    bins[i] += 1;
                }
                BinIndex::Underflow => underflow_count += 1,
                BinIndex::Overflow => overflow_count += 1,
                BinIndex::Nan => nan_count += 1,
            }
        }
    }

    Histogram::new(
        strategy.clone(),
        bins,
        nan_count,
        underflow_count,
        overflow_count,
    )
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{Image, SubView};
    use crate::pixel::{Indexed8, Mono8, MonoF32, Rgb8};
    use std::num::Saturating;

    // Bring the trait into scope so `img.roi(..)` resolves.
    #[allow(unused_imports)]
    use crate::image::SubView as _;

    // ── Single-channel: NaturalBins on Mono8 ────────────────────────────────

    #[test]
    fn mono8_natural_single_histogram() {
        let img = Image::from_vec(
            2,
            2,
            vec![Mono8::new(0), Mono8::new(1), Mono8::new(1), Mono8::new(255)],
        )
        .unwrap();
        let h: Histogram<NaturalBins, Saturating<u8>> = histogram(&img, &NaturalBins).unwrap();

        assert_eq!(h.count_at_bin(0), 1);
        assert_eq!(h.count_at_bin(1), 2);
        assert_eq!(h.count_at_bin(255), 1);
        assert_eq!(h.total_count, 4);
        assert_eq!(h.nan_count, 0);
        assert_eq!(h.underflow_count, 0);
        assert_eq!(h.overflow_count, 0);
    }

    // ── Single-channel: NaturalBins on Indexed8 (channel = u8) ──────────────

    #[test]
    fn indexed8_natural_uses_u8_channel_type() {
        let img = Image::from_vec(
            2,
            2,
            vec![Indexed8(3), Indexed8(3), Indexed8(7), Indexed8(200)],
        )
        .unwrap();
        let h: Histogram<NaturalBins, u8> = histogram(&img, &NaturalBins).unwrap();

        assert_eq!(h.count_at_bin(3), 2);
        assert_eq!(h.count_at_bin(7), 1);
        assert_eq!(h.count_at_bin(200), 1);
        assert_eq!(h.total_count, 4);
    }

    // ── RGB: array output ───────────────────────────────────────────────────

    #[test]
    fn rgb8_natural_array_output() {
        let img = Image::from_vec(1, 1, vec![Rgb8::new(10, 20, 30)]).unwrap();
        let [r, g, b]: [Histogram<NaturalBins, Saturating<u8>>; 3] =
            histogram(&img, &NaturalBins).unwrap();

        assert_eq!(r.count_at_bin(10), 1);
        assert_eq!(g.count_at_bin(20), 1);
        assert_eq!(b.count_at_bin(30), 1);
    }

    #[test]
    fn rgb8_natural_vec_output() {
        let img =
            Image::from_vec(1, 2, vec![Rgb8::new(10, 20, 30), Rgb8::new(10, 21, 30)]).unwrap();
        let v: Vec<Histogram<NaturalBins, Saturating<u8>>> = histogram(&img, &NaturalBins).unwrap();

        assert_eq!(v.len(), 3);
        assert_eq!(v[0].count_at_bin(10), 2); // R
        assert_eq!(v[1].count_at_bin(20), 1); // G[0]
        assert_eq!(v[1].count_at_bin(21), 1); // G[1]
        assert_eq!(v[2].count_at_bin(30), 2); // B
    }

    #[test]
    #[should_panic(expected = "channels")]
    fn rgb8_requested_as_single_histogram_panics() {
        let img = Image::from_vec(1, 1, vec![Rgb8::new(0, 0, 0)]).unwrap();
        let _: Histogram<NaturalBins, Saturating<u8>> = histogram(&img, &NaturalBins).unwrap();
    }

    #[test]
    #[should_panic(expected = "channels")]
    fn rgb8_requested_as_4_array_panics() {
        let img = Image::from_vec(1, 1, vec![Rgb8::new(0, 0, 0)]).unwrap();
        let _: [Histogram<NaturalBins, Saturating<u8>>; 4] = histogram(&img, &NaturalBins).unwrap();
    }

    // ── Float channels: LinearBins on MonoF32 ───────────────────────────────

    #[test]
    fn monof32_linear_counts_in_range_nan_underflow_overflow() {
        let img = Image::from_vec(
            5,
            1,
            vec![
                MonoF32::new(0.0),
                MonoF32::new(0.25),
                MonoF32::new(f32::NAN),
                MonoF32::new(-0.5),
                MonoF32::new(2.0),
            ],
        )
        .unwrap();

        let s = LinearBins {
            min: 0.0,
            max: 1.0,
            bin_count: 4,
        };
        let h: Histogram<LinearBins, f32> = histogram(&img, &s).unwrap();

        assert_eq!(h.count_at_bin(0), 1); // 0.0 → bin 0
        assert_eq!(h.count_at_bin(1), 1); // 0.25 → bin 1
        assert_eq!(h.nan_count, 1);
        assert_eq!(h.underflow_count, 1);
        assert_eq!(h.overflow_count, 1);
        assert_eq!(h.total_count, 5);
    }

    // ── Strategy validation: error tier ─────────────────────────────────────

    #[test]
    fn invalid_linear_returns_error() {
        let img = Image::from_vec(1, 1, vec![MonoF32::new(0.0)]).unwrap();
        let bad = LinearBins {
            min: 1.0,
            max: 0.0,
            bin_count: 4,
        };
        let r: Result<Histogram<LinearBins, f32>, _> = histogram(&img, &bad);
        match r {
            Err(Error::InvalidBinningStrategy(_)) => {}
            other => panic!("expected InvalidBinningStrategy, got {:?}", other),
        }
    }

    #[test]
    fn invalid_custom_returns_error() {
        let img = Image::from_vec(1, 1, vec![MonoF32::new(0.0)]).unwrap();
        let bad = CustomBins { edges: vec![1.0] };
        let r: Result<Histogram<CustomBins, f32>, _> = histogram(&img, &bad);
        assert!(matches!(r, Err(Error::InvalidBinningStrategy(_))));
    }

    // ── ROI: SubView counts only the selected region ────────────────────────

    #[test]
    fn subview_histogram_only_counts_selected_region() {
        // 4×1 image with values [10, 20, 30, 40]; ROI selects the
        // middle two pixels.
        let img = Image::from_vec(
            4,
            1,
            vec![
                Mono8::new(10),
                Mono8::new(20),
                Mono8::new(30),
                Mono8::new(40),
            ],
        )
        .unwrap();
        let roi = SubView::roi(
            &img,
            crate::Rectangle::new(crate::Coordinate::new(1, 0), crate::Size::new(2, 1)),
        )
        .unwrap();

        let h: Histogram<NaturalBins, Saturating<u8>> = histogram(&roi, &NaturalBins).unwrap();
        assert_eq!(h.total_count, 2);
        assert_eq!(h.count_at_bin(20), 1);
        assert_eq!(h.count_at_bin(30), 1);
        assert_eq!(h.count_at_bin(10), 0);
        assert_eq!(h.count_at_bin(40), 0);
    }

    // ── Engine: total_count is correct across many pixels ───────────────────

    #[test]
    fn engine_total_count_matches_pixel_count() {
        let pixels: Vec<Mono8> = (0..100u16).map(|v| Mono8::new((v % 256) as u8)).collect();
        let img = Image::from_vec(10, 10, pixels).unwrap();

        let h: Histogram<NaturalBins, Saturating<u8>> = histogram(&img, &NaturalBins).unwrap();
        assert_eq!(h.total_count, 100);
        assert_eq!(h.bins().iter().sum::<u64>(), 100);
    }
}

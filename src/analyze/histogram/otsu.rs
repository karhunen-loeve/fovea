//! Otsu's automatic threshold — a histogram consumer.
//!
//! Two entry points:
//!
//! - [`otsu_threshold`] — pure algorithm on a 256-bin
//!   [`Histogram<NaturalBins, V>`](super::Histogram); returns
//!   `Option<u8>` (Tier 1 per ADR-0025: `None` for degenerate inputs).
//! - [`otsu_binary_mask`] — convenience that runs the histogram engine,
//!   picks the threshold, and applies
//!   [`BinaryMask`](crate::transform::BinaryMask) to produce a
//!   [`BinaryImage`].
//!
//! See `HISTOGRAM_CONSUMERS_PLAN.md` and ADR-0040 for the design
//! rationale.

use crate::Error;
use crate::analyze::histogram::strategy::BinningStrategy;
use crate::analyze::histogram::{Histogram, NaturalBins, histogram};
use crate::image::{BinaryImage, RasterImage};
use crate::pixel::HomogeneousPixel;
use crate::transform::{BinaryMask, convert_image};

/// Compute Otsu's optimal threshold from a 256-bin histogram.
///
/// Returns `None` (Tier 1 per ADR-0025) for degenerate histograms
/// where no meaningful split exists — either there are no in-range
/// pixels at all, or every pixel falls in a single bin. Otherwise
/// returns `Some(t)` where pixels with channel value `> t` belong to
/// the bright class, matching
/// [`BinaryMask`](crate::transform::BinaryMask)'s
/// `value > thresh => true` convention.
///
/// Generic over `V` so both `u8` (e.g. [`Indexed8`](crate::pixel::Indexed8))
/// and [`Saturating<u8>`](std::num::Saturating) (e.g.
/// [`Mono8`](crate::pixel::Mono8), [`Rgb8`](crate::pixel::Rgb8), …)
/// channel types are accepted via the existing
/// `NaturalBins: BinningStrategy<V>` impls.
///
/// NaN, underflow, and overflow counters of the input histogram are
/// ignored: by construction of [`NaturalBins`] those counters are
/// always zero, but documenting the behaviour pins it for any future
/// 256-bin strategy that might reuse this function.
///
/// # Algorithm
///
/// Standard one-pass Otsu: scan `t` from 0 to 255 maintaining the
/// running class-0 weight `w0`, class-0 weighted sum `sum_b`, and the
/// resulting between-class variance
/// `w0 * w1 * (mu0 - mu1)^2`. Return the `t` that maximises this
/// quantity.
///
/// # Examples
///
/// ```
/// use fovea::analyze::histogram::{Histogram, NaturalBins, histogram, otsu_threshold};
/// use fovea::image::Image;
/// use fovea::pixel::Mono8;
///
/// // 4 dark pixels and 4 bright ones — a clean bimodal split.
/// let img = Image::from_vec(
///     4, 2,
///     vec![
///         Mono8::new(20), Mono8::new(20), Mono8::new(20), Mono8::new(20),
///         Mono8::new(220), Mono8::new(220), Mono8::new(220), Mono8::new(220),
///     ],
/// ).unwrap();
/// let h: Histogram<NaturalBins, _> = histogram(&img, &NaturalBins).unwrap();
/// let t = otsu_threshold(&h).unwrap();
/// assert!((20..220).contains(&(t as u32)));
/// ```
pub fn otsu_threshold<V: Copy>(hist: &Histogram<NaturalBins, V>) -> Option<u8>
where
    NaturalBins: BinningStrategy<V>,
{
    let bins = hist.bins();
    debug_assert_eq!(
        bins.len(),
        256,
        "otsu_threshold: NaturalBins must yield exactly 256 bins"
    );

    let total: u64 = bins.iter().sum();
    if total == 0 {
        return None;
    }

    // Σ i * h[i] — fits in u64 because i ≤ 255 and Σ h[i] ≤ u64::MAX / 255
    // for any realistic image. Document the budget for future widenings.
    let sum_total: u64 = bins.iter().enumerate().map(|(i, &c)| (i as u64) * c).sum();

    let mut w0: u64 = 0;
    let mut sum_b: u64 = 0;
    let mut best_var: f64 = -1.0;
    let mut best_t: Option<u8> = None;

    for (t, &c) in bins.iter().enumerate() {
        w0 += c;
        if w0 == 0 {
            continue;
        }
        let w1 = total - w0;
        if w1 == 0 {
            // Remaining bins are all empty: no further split possible.
            break;
        }
        sum_b += (t as u64) * c;

        let mu0 = sum_b as f64 / w0 as f64;
        let mu1 = (sum_total - sum_b) as f64 / w1 as f64;
        let diff = mu0 - mu1;
        let var_between = (w0 as f64) * (w1 as f64) * diff * diff;
        if var_between > best_var {
            best_var = var_between;
            best_t = Some(t as u8);
        }
    }
    best_t
}

/// Compute Otsu's threshold on a single-channel image and return both
/// the chosen threshold and the corresponding [`BinaryImage`].
///
/// On a degenerate histogram ([`otsu_threshold`] returns `None`) this
/// falls back to a threshold of `0` — matching the "no contrast →
/// everything bright" behaviour of OpenCV's `threshold(…, OTSU)` for
/// flat inputs.
///
/// # Errors
///
/// Returns [`Error::InvalidBinningStrategy`] (Tier 2) only if the
/// internal histogram engine ever does. [`NaturalBins`] has no
/// configuration to validate, but the `Result` return shape leaves
/// room for 16-bit overloads that take a caller-supplied
/// [`LinearBins`](crate::analyze::histogram::LinearBins).
///
/// # Panics
///
/// Panics if `P::CHANNEL_COUNT != 1` (Tier 3 per ADR-0025; callers
/// with multi-channel images must convert to single channel first).
///
/// # Examples
///
/// ```
/// use fovea::analyze::histogram::otsu_binary_mask;
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::Mono8;
///
/// let img = Image::from_vec(
///     2,
///     2,
///     vec![Mono8::new(0), Mono8::new(0), Mono8::new(255), Mono8::new(255)],
/// ).unwrap();
/// let (t, mask) = otsu_binary_mask(&img).unwrap();
/// assert!(t < 255);
/// assert!(!mask.pixel_at(0, 0));
/// assert!(mask.pixel_at(0, 1));
/// ```
pub fn otsu_binary_mask<I, P>(image: &I) -> Result<(u8, BinaryImage), Error>
where
    I: RasterImage<Pixel = P>,
    P: HomogeneousPixel + From<u8>,
    P::Channel: Ord,
    NaturalBins: BinningStrategy<P::Channel>,
{
    assert_eq!(
        P::CHANNEL_COUNT,
        1,
        "otsu_binary_mask: requires a single-channel pixel; got CHANNEL_COUNT = {}",
        P::CHANNEL_COUNT
    );

    let h: Histogram<NaturalBins, P::Channel> = histogram(image, &NaturalBins)?;
    let t = otsu_threshold(&h).unwrap_or(0);
    let thresh = P::from(t);
    let mask: BinaryImage = convert_image(image, BinaryMask { thresh });
    Ok((t, mask))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{Image, ImageView};
    use crate::pixel::{Indexed8, Mono8};
    use std::num::Saturating;

    fn hist_from_bins_u8(bins: Vec<u64>) -> Histogram<NaturalBins, u8> {
        assert_eq!(bins.len(), 256);
        Histogram::new(NaturalBins, bins, 0, 0, 0)
    }

    // ── otsu_threshold ──────────────────────────────────────────────────────

    #[test]
    fn otsu_threshold_bimodal_picks_valley() {
        // Two overlapping gaussians at μ=64 and μ=192 with σ=20. Wide
        // enough to produce a real valley around the midpoint (128)
        // — narrower modes leave a flat empty plateau, which is also
        // an optimum for Otsu but degenerate to test by the "near
        // midpoint" criterion (the algorithm picks the leftmost
        // plateau point on ties).
        let mut bins = vec![0u64; 256];
        for (i, bin) in bins.iter_mut().enumerate() {
            let d0 = (i as i32 - 64) as f64;
            let d1 = (i as i32 - 192) as f64;
            let g = (-d0 * d0 / (2.0 * 20.0 * 20.0)).exp() + (-d1 * d1 / (2.0 * 20.0 * 20.0)).exp();
            *bin = (g * 1000.0) as u64;
        }
        let h = hist_from_bins_u8(bins);
        let t = otsu_threshold(&h).expect("bimodal histogram must yield a threshold");
        assert!(
            (118..=138).contains(&(t as u32)),
            "expected ~128 between the modes; got {t}",
        );
    }

    #[test]
    fn otsu_threshold_bimodal_non_overlapping_modes_pick_between() {
        // Narrow non-overlapping gaussians at μ=64 and μ=192. The
        // optimum is the empty valley between them; the algorithm
        // settles on the left edge of that valley but the chosen `t`
        // must still be strictly between the two modes.
        let mut bins = vec![0u64; 256];
        for (i, bin) in bins.iter_mut().enumerate() {
            let d0 = (i as i32 - 64) as f64;
            let d1 = (i as i32 - 192) as f64;
            let g = (-d0 * d0 / (2.0 * 10.0 * 10.0)).exp() + (-d1 * d1 / (2.0 * 10.0 * 10.0)).exp();
            *bin = (g * 1000.0) as u64;
        }
        let h = hist_from_bins_u8(bins);
        let t = otsu_threshold(&h).expect("bimodal histogram must yield a threshold");
        assert!(
            (64..192).contains(&(t as u32)),
            "expected t strictly between the two modes; got {t}",
        );
    }

    #[test]
    fn otsu_threshold_uniform_returns_central_value() {
        // Flat histogram: every t in (0, 255) yields a finite positive
        // variance; the maximum lies near the middle of the range.
        let bins = vec![1u64; 256];
        let h = hist_from_bins_u8(bins);
        let t = otsu_threshold(&h).expect("uniform histogram has a meaningful split");
        assert!(
            (50..=200).contains(&(t as u32)),
            "uniform histogram should pick a central threshold; got {t}",
        );
    }

    #[test]
    fn otsu_threshold_all_in_one_bin_returns_none() {
        let mut bins = vec![0u64; 256];
        bins[100] = 50;
        let h = hist_from_bins_u8(bins);
        assert_eq!(otsu_threshold(&h), None);
    }

    #[test]
    fn otsu_threshold_all_in_first_bin_returns_none() {
        let mut bins = vec![0u64; 256];
        bins[0] = 42;
        let h = hist_from_bins_u8(bins);
        assert_eq!(otsu_threshold(&h), None);
    }

    #[test]
    fn otsu_threshold_all_in_last_bin_returns_none() {
        let mut bins = vec![0u64; 256];
        bins[255] = 7;
        let h = hist_from_bins_u8(bins);
        assert_eq!(otsu_threshold(&h), None);
    }

    #[test]
    fn otsu_threshold_empty_total_returns_none() {
        let bins = vec![0u64; 256];
        let h = hist_from_bins_u8(bins);
        assert_eq!(otsu_threshold(&h), None);
    }

    #[test]
    fn otsu_threshold_two_pixels_split() {
        // h[10] = 1, h[200] = 1: variance is constant on the plateau
        // [10, 199]; the first index in that plateau wins.
        let mut bins = vec![0u64; 256];
        bins[10] = 1;
        bins[200] = 1;
        let h = hist_from_bins_u8(bins);
        let t = otsu_threshold(&h).expect("two distinct pixels must yield a split");
        assert!(
            (10..200).contains(&(t as u32)),
            "expected t in [10, 200); got {t}",
        );
    }

    #[test]
    fn otsu_threshold_matches_known_reference() {
        // h[50]=5, h[150]=5. Variance is constant at 250_000 across
        // t ∈ [50, 149]; the first index wins → 50.
        let mut bins = vec![0u64; 256];
        bins[50] = 5;
        bins[150] = 5;
        let h = hist_from_bins_u8(bins);
        assert_eq!(otsu_threshold(&h), Some(50));
    }

    #[test]
    fn otsu_threshold_unequal_classes() {
        // Asymmetric counts at the two peaks must still pick a
        // threshold between them.
        let mut bins = vec![0u64; 256];
        bins[40] = 100;
        bins[210] = 5;
        let h = hist_from_bins_u8(bins);
        let t = otsu_threshold(&h).expect("bimodal split exists");
        assert!((40..210).contains(&(t as u32)));
    }

    #[test]
    fn otsu_threshold_accepts_saturating_u8_channel() {
        // Compile-level + behavioural check: NaturalBins also
        // implements BinningStrategy<Saturating<u8>>.
        let mut bins = vec![0u64; 256];
        bins[10] = 1;
        bins[240] = 1;
        let h: Histogram<NaturalBins, Saturating<u8>> = Histogram::new(NaturalBins, bins, 0, 0, 0);
        let t = otsu_threshold(&h).unwrap();
        assert!((10..240).contains(&(t as u32)));
    }

    #[test]
    fn otsu_threshold_ignores_outlier_counters() {
        // NaN / underflow / overflow are never produced by NaturalBins
        // but documenting: even if a histogram somehow carries them,
        // the in-range result is unchanged.
        let mut bins = vec![0u64; 256];
        bins[50] = 5;
        bins[150] = 5;
        let h: Histogram<NaturalBins, u8> = Histogram::new(NaturalBins, bins, 999, 999, 999);
        assert_eq!(otsu_threshold(&h), Some(50));
    }

    // ── otsu_binary_mask ────────────────────────────────────────────────────

    #[test]
    fn otsu_binary_mask_mono8_separates_black_and_white() {
        let img = Image::from_vec(
            2,
            2,
            vec![
                Mono8::new(0),
                Mono8::new(0),
                Mono8::new(255),
                Mono8::new(255),
            ],
        )
        .unwrap();
        let (t, mask) = otsu_binary_mask(&img).unwrap();
        assert!(
            t < 255,
            "threshold must leave the bright class non-empty; got {t}",
        );
        assert!(!mask.pixel_at(0, 0));
        assert!(!mask.pixel_at(1, 0));
        assert!(mask.pixel_at(0, 1));
        assert!(mask.pixel_at(1, 1));
    }

    #[test]
    fn otsu_binary_mask_indexed8_runs_on_u8_channel() {
        // Indexed8 channel type is bare u8; covers the second
        // NaturalBins impl. Indexed8 implements From<u8> as of this
        // crate's `pixel::indexed` module.
        let img = Image::from_vec(
            2,
            2,
            vec![Indexed8(10), Indexed8(10), Indexed8(200), Indexed8(200)],
        )
        .unwrap();
        let (t, mask) = otsu_binary_mask(&img).unwrap();
        assert!((10..200).contains(&(t as u32)));
        assert!(!mask.pixel_at(0, 0));
        assert!(mask.pixel_at(0, 1));
    }

    #[test]
    fn otsu_binary_mask_flat_image_falls_back_to_zero_threshold() {
        let img: Image<Mono8> = Image::fill(3, 3, Mono8::new(42));
        let (t, mask) = otsu_binary_mask(&img).unwrap();
        assert_eq!(t, 0);
        // 42 > 0 → every pixel is "foreground".
        for y in 0..mask.height() {
            for x in 0..mask.width() {
                assert!(mask.pixel_at(x, y));
            }
        }
    }

    #[test]
    fn otsu_binary_mask_preserves_image_size() {
        let img: Image<Mono8> =
            Image::generate(7, 5, |x, y| Mono8::new(((x * 30 + y * 50) % 256) as u8));
        let (_, mask) = otsu_binary_mask(&img).unwrap();
        assert_eq!(mask.size(), img.size());
    }

    #[test]
    fn otsu_binary_mask_threshold_matches_standalone_otsu() {
        // The convenience wrapper must agree with the pure-algorithm
        // path applied to the same image.
        let img: Image<Mono8> = Image::generate(16, 16, |x, y| {
            Mono8::new(if (x + y) % 2 == 0 { 30 } else { 200 })
        });
        let h: Histogram<NaturalBins, _> = histogram(&img, &NaturalBins).unwrap();
        let direct = otsu_threshold(&h).unwrap();
        let (via_mask, _) = otsu_binary_mask(&img).unwrap();
        assert_eq!(via_mask, direct);
    }
}

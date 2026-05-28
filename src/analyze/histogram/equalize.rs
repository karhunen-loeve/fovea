//! Histogram equalization — a histogram consumer.
//!
//! Three entry points:
//!
//! - [`equalization_lut`] — build a 256-entry per-channel LUT from a
//!   single-channel [`Histogram<NaturalBins, V>`](super::Histogram).
//! - [`equalize_image`] — apply the LUT (one per channel) to an entire
//!   image, returning a new owned image.
//! - [`equalize_image_into`] — `_into` companion that writes into a
//!   caller-supplied output.
//!
//! See `HISTOGRAM_CONSUMERS_PLAN.md` and ADR-0040 for the design
//! rationale.

use std::num::Saturating;

use crate::Error;
use crate::analyze::histogram::strategy::BinningStrategy;
use crate::analyze::histogram::{Histogram, NaturalBins, histogram};
use crate::image::{Image, RasterImage, RasterImageMut};
use crate::pixel::{Array, HomogeneousPixel, ZeroablePixel};
use crate::transform::ChannelLut;

/// Build a 256-entry per-channel equalization LUT from a single-channel
/// histogram (or one channel of a multi-channel histogram).
///
/// The result can be applied with
/// [`convert_image`](crate::transform::convert_image) /
/// [`convert_image_into`](crate::transform::convert_image_into) using
/// the existing [`ChannelLut`] strategy, or queried point-wise via
/// [`ChannelLut::lookup`].
///
/// # Algorithm
///
/// Standard textbook formula:
///
/// ```text
/// lut[i] = round( 255 * (cdf[i] - cdf_min) / (total - cdf_min) )
/// ```
///
/// where `cdf_min` is the smallest non-zero entry of the cumulative
/// histogram. Matches OpenCV's `equalizeHist` for 8-bit grayscale.
///
/// Degenerate inputs (`total - cdf_min == 0`, i.e. every in-range
/// pixel falls in a single bin, or no in-range pixels at all) yield
/// the identity LUT — there is no contrast to stretch.
///
/// # Examples
///
/// ```
/// use fovea::analyze::histogram::{Histogram, NaturalBins, equalization_lut, histogram};
/// use fovea::image::Image;
/// use fovea::pixel::Mono8;
///
/// // Image with the full 0..=255 range — LUT should be (nearly) identity.
/// let img: Image<Mono8> = Image::from_vec(
///     256, 1, (0u8..=255).map(Mono8::new).collect(),
/// ).unwrap();
/// let h: Histogram<NaturalBins, _> = histogram(&img, &NaturalBins).unwrap();
/// let lut = equalization_lut(&h);
/// assert_eq!(lut.lookup(0), 0);
/// assert_eq!(lut.lookup(255), 255);
/// ```
pub fn equalization_lut<V: Copy>(hist: &Histogram<NaturalBins, V>) -> ChannelLut
where
    NaturalBins: BinningStrategy<V>,
{
    let cdf = hist.cumulative();
    debug_assert_eq!(cdf.len(), 256);

    let total: u64 = *cdf.last().expect("NaturalBins yields 256 bins");
    let cdf_min: u64 = cdf.iter().copied().find(|&c| c != 0).unwrap_or(0);

    if total <= cdf_min {
        // Degenerate: all in-range pixels in a single bin (or no
        // in-range pixels at all). Identity preserves data.
        return ChannelLut::from_fn(|i| i);
    }

    let denom = (total - cdf_min) as f64;
    let mut table = [0u8; 256];
    for i in 0..256 {
        let num = cdf[i].saturating_sub(cdf_min) as f64;
        let v = (num * 255.0 / denom).round();
        table[i] = v.clamp(0.0, 255.0) as u8;
    }
    ChannelLut::new(table)
}

/// Equalize the histogram of every channel of an 8-bit image.
///
/// Each channel is equalized independently. For color images this does
/// **not** preserve hue — convert to a luminance/chrominance space
/// (e.g. YCbCr) first if hue preservation is required. The library
/// surfaces information, it does not silently colour-space-convert
/// (Philosophy §8).
///
/// The `Channel = Saturating<u8>` bound deliberately rejects
/// [`Indexed8`](crate::pixel::Indexed8): equalising palette indices is
/// meaningless (ADR-0010, Philosophy §1). It also rejects 16-bit and
/// float channel types — a wider equalization can be added later
/// without breaking changes (Philosophy §10 — extension by addition).
///
/// # Errors
///
/// Returns [`Error::InvalidBinningStrategy`] (Tier 2) only if the
/// internal histogram engine ever does. [`NaturalBins`] has no
/// configuration to validate, but the `Result` return shape is kept
/// for future strategy overloads.
///
/// # Examples
///
/// ```
/// use fovea::analyze::histogram::equalize_image;
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::Mono8;
///
/// // Pixels in [100, 110] — equalization stretches to [0, 255].
/// let img: Image<Mono8> = Image::generate(11, 1, |x, _| Mono8::new(100 + x as u8));
/// let out = equalize_image(&img).unwrap();
/// assert_eq!(u8::from(out.pixel_at(0, 0)), 0);
/// assert_eq!(u8::from(out.pixel_at(10, 0)), 255);
/// ```
pub fn equalize_image<I, P>(image: &I) -> Result<Image<P>, Error>
where
    I: RasterImage<Pixel = P>,
    P: HomogeneousPixel<Channel = Saturating<u8>> + ZeroablePixel,
    NaturalBins: BinningStrategy<P::Channel>,
{
    let mut out = Image::<P>::zero(image.width(), image.height());
    equalize_image_into(image, &mut out)?;
    Ok(out)
}

/// `_into` companion of [`equalize_image`].
///
/// # Panics
///
/// Panics if `image.size() != out.size()` (Tier 3 per ADR-0025).
pub fn equalize_image_into<I, O, P>(image: &I, out: &mut O) -> Result<(), Error>
where
    I: RasterImage<Pixel = P>,
    O: RasterImageMut<Pixel = P>,
    P: HomogeneousPixel<Channel = Saturating<u8>>,
    NaturalBins: BinningStrategy<P::Channel>,
{
    assert_eq!(
        image.size(),
        out.size(),
        "equalize_image_into: input size {:?} does not match output size {:?}",
        image.size(),
        out.size()
    );

    let hists: Vec<Histogram<NaturalBins, P::Channel>> = histogram(image, &NaturalBins)?;
    let luts: Vec<ChannelLut> = hists.iter().map(equalization_lut).collect();
    debug_assert_eq!(luts.len(), P::CHANNEL_COUNT);

    for y in 0..image.height() {
        let in_row = image.row(y);
        let out_row = out.row_mut(y);
        for (src, dst) in in_row.iter().zip(out_row.iter_mut()) {
            let channels = <P::Channels as Array<P::Channel>>::from_fn(|c| {
                let v: Saturating<u8> = src.channel(c);
                Saturating(luts[c].lookup(v.0))
            });
            *dst = P::from_channels(channels.as_ref());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{Image, ImageView};
    use crate::pixel::{Mono8, Rgb8};
    use crate::transform::ConvertPixel;
    use std::num::Saturating as S;

    fn hist_from_bins(bins: Vec<u64>) -> Histogram<NaturalBins, S<u8>> {
        assert_eq!(bins.len(), 256);
        Histogram::new(NaturalBins, bins, 0, 0, 0)
    }

    // ── equalization_lut ────────────────────────────────────────────────────

    #[test]
    fn equalization_lut_uniform_is_identity() {
        // h[i] = 1 for all i ⇒ cdf[i] = i+1, cdf_min = 1, denom = 255.
        // lut[i] = round((i+1 - 1) * 255 / 255) = i.
        let h = hist_from_bins(vec![1u64; 256]);
        let lut = equalization_lut(&h);
        for i in 0..=255u8 {
            assert_eq!(lut.lookup(i), i, "expected identity at i={i}");
        }
    }

    #[test]
    fn equalization_lut_single_bin_is_identity() {
        let mut bins = vec![0u64; 256];
        bins[100] = 50;
        let h = hist_from_bins(bins);
        let lut = equalization_lut(&h);
        for i in [0u8, 50, 100, 200, 255] {
            assert_eq!(lut.lookup(i), i);
        }
    }

    #[test]
    fn equalization_lut_empty_histogram_is_identity() {
        let h = hist_from_bins(vec![0u64; 256]);
        let lut = equalization_lut(&h);
        for i in [0u8, 64, 200, 255] {
            assert_eq!(lut.lookup(i), i);
        }
    }

    #[test]
    fn equalization_lut_known_reference() {
        // h[0]=2, h[1]=3, h[3]=5; total=10.
        // cdf = [2,5,5,10,10,...]. cdf_min=2, denom=8, scale=31.875.
        //   lut[0] = round(0*31.875)   = 0
        //   lut[1] = round(3*31.875)   = 96  (95.625 → 96)
        //   lut[2] = round(3*31.875)   = 96
        //   lut[3] = round(8*31.875)   = 255
        //   lut[i>=3] = 255
        let mut bins = vec![0u64; 256];
        bins[0] = 2;
        bins[1] = 3;
        bins[3] = 5;
        let h = hist_from_bins(bins);
        let lut = equalization_lut(&h);
        assert_eq!(lut.lookup(0), 0);
        assert_eq!(lut.lookup(1), 96);
        assert_eq!(lut.lookup(2), 96);
        assert_eq!(lut.lookup(3), 255);
        assert_eq!(lut.lookup(50), 255);
        assert_eq!(lut.lookup(255), 255);
    }

    #[test]
    fn equalization_lut_endpoints_are_zero_and_255() {
        // Any non-degenerate histogram must map the smallest non-zero
        // bin index to 0 and the largest non-zero bin index to 255.
        let mut bins = vec![0u64; 256];
        bins[10] = 7;
        bins[20] = 3;
        bins[200] = 9;
        let h = hist_from_bins(bins);
        let lut = equalization_lut(&h);
        assert_eq!(lut.lookup(10), 0);
        assert_eq!(lut.lookup(200), 255);
    }

    #[test]
    fn equalization_lut_is_monotonic_non_decreasing() {
        // The CDF is monotonic non-decreasing, so the LUT must be too.
        let mut bins = vec![0u64; 256];
        for (i, c) in [10u8, 30, 80, 120, 200]
            .iter()
            .zip([5u64, 9, 13, 8, 3].iter())
        {
            bins[*i as usize] = *c;
        }
        let h = hist_from_bins(bins);
        let lut = equalization_lut(&h);
        for i in 1..=255u8 {
            assert!(
                lut.lookup(i) >= lut.lookup(i - 1),
                "non-monotonic at i={i}: {} < {}",
                lut.lookup(i),
                lut.lookup(i - 1)
            );
        }
    }

    // ── equalize_image / equalize_image_into ────────────────────────────────

    #[test]
    fn equalize_image_mono8_preserves_size_and_dtype() {
        let img: Image<Mono8> = Image::generate(8, 2, |x, _| Mono8::new((x * 30) as u8));
        let out = equalize_image(&img).unwrap();
        assert_eq!(out.size(), img.size());
    }

    #[test]
    fn equalize_image_mono8_uniform_image_is_unchanged() {
        // Every pixel identical → degenerate histogram → identity LUT
        // → output equal to input.
        let img: Image<Mono8> = Image::fill(4, 3, Mono8::new(123));
        let out = equalize_image(&img).unwrap();
        for y in 0..img.height() {
            for x in 0..img.width() {
                assert_eq!(out.pixel_at(x, y), img.pixel_at(x, y));
            }
        }
    }

    #[test]
    fn equalize_image_mono8_stretches_low_contrast() {
        // Pixels in [100, 110]; equalization should stretch the
        // extremes to 0 and 255.
        let img: Image<Mono8> = Image::generate(11, 1, |x, _| Mono8::new(100 + x as u8));
        let out = equalize_image(&img).unwrap();
        assert_eq!(u8::from(out.pixel_at(0, 0)), 0);
        assert_eq!(u8::from(out.pixel_at(10, 0)), 255);
    }

    #[test]
    fn equalize_image_mono8_is_idempotent_on_already_uniform_histogram() {
        // After equalization, the histogram is approximately flat;
        // a second pass shouldn't change values meaningfully. With
        // many distinct input values, the LUT is close to identity.
        let img: Image<Mono8> = Image::generate(256, 1, |x, _| Mono8::new(x as u8));
        let once = equalize_image(&img).unwrap();
        let twice = equalize_image(&once).unwrap();
        for x in 0..256 {
            assert_eq!(once.pixel_at(x, 0), twice.pixel_at(x, 0));
        }
    }

    #[test]
    fn equalize_image_rgb8_independent_per_channel() {
        // Each channel has a distinct distribution; the output of
        // `equalize_image` must equal the result of applying each
        // channel's own LUT independently.
        let w = 16usize;
        let img: Image<Rgb8> = Image::generate(w, 1, |x, _| {
            Rgb8::new(
                (x as u8) * 4,
                ((x as u8) * 2) + 50,
                if x < w / 2 { 10 } else { 240 },
            )
        });
        let out = equalize_image(&img).unwrap();

        let hists: Vec<Histogram<NaturalBins, S<u8>>> = histogram(&img, &NaturalBins).unwrap();
        let lut_r = equalization_lut(&hists[0]);
        let lut_g = equalization_lut(&hists[1]);
        let lut_b = equalization_lut(&hists[2]);

        for x in 0..w {
            let p_in = img.pixel_at(x, 0);
            let p_out = out.pixel_at(x, 0);
            let expected = Rgb8::new(
                lut_r.lookup(p_in.r.0),
                lut_g.lookup(p_in.g.0),
                lut_b.lookup(p_in.b.0),
            );
            assert_eq!(p_out, expected, "mismatch at x={x}");
        }
    }

    #[test]
    fn equalize_image_rgb8_uses_per_channel_distinct_luts() {
        // Sanity: confirm we do NOT collapse to a single LUT applied
        // to every channel (which would lose information).
        let img: Image<Rgb8> = Image::generate(8, 1, |x, _| {
            Rgb8::new((x * 30) as u8, ((x * 3) + 10) as u8, 255 - (x * 30) as u8)
        });
        let out = equalize_image(&img).unwrap();
        let hists: Vec<Histogram<NaturalBins, S<u8>>> = histogram(&img, &NaturalBins).unwrap();
        let lut_r = equalization_lut(&hists[0]);

        // If equalize_image incorrectly applied `lut_r` to every
        // channel, the comparison below would always be equal.
        let mut any_disagree = false;
        for x in 0..8 {
            let in_p = img.pixel_at(x, 0);
            let out_p = out.pixel_at(x, 0);
            let single = lut_r.convert(&in_p);
            if out_p.g.0 != single.g.0 || out_p.b.0 != single.b.0 {
                any_disagree = true;
                break;
            }
        }
        assert!(any_disagree, "expected per-channel-distinct equalization");
    }

    #[test]
    fn equalize_image_into_writes_into_provided_output() {
        let img: Image<Mono8> = Image::generate(4, 4, |x, y| Mono8::new(((x + y) * 16) as u8));
        let expected = equalize_image(&img).unwrap();
        let mut out: Image<Mono8> = Image::zero(4, 4);
        equalize_image_into(&img, &mut out).unwrap();
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(out.pixel_at(x, y), expected.pixel_at(x, y));
            }
        }
    }

    #[test]
    #[should_panic(expected = "does not match")]
    fn equalize_image_into_size_mismatch_panics() {
        let img: Image<Mono8> = Image::fill(4, 4, Mono8::new(0));
        let mut out: Image<Mono8> = Image::zero(3, 4);
        let _ = equalize_image_into(&img, &mut out);
    }
}

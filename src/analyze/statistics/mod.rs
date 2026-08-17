//! Whole-image statistics: per-channel summaries and intensity moments.
//!
//! Two questions, two entry points:
//!
//! | Question | Use | Output |
//! |---|---|---|
//! | "How bright is this image, and how uniform?" | [`image_statistics`] | [`ChannelStatistics`] per channel |
//! | "Where is the brightness, and what shape is it?" | [`image_moments`] | [`ImageMoments`] and the invariants derived from it |
//!
//! # Relationship to the histogram
//!
//! [`histogram`](crate::analyze::histogram) answers the same first question at
//! higher resolution: it reports the whole distribution, and a mean or a
//! variance can be recovered from it. The difference is cost and exactness. A
//! histogram needs a binning strategy, allocates one counter array per
//! channel, and reports a mean only as accurately as its bins; these functions
//! take no configuration, allocate nothing, and are exact for the values they
//! see. Reach for the histogram when the *shape* of the distribution matters
//! (a threshold, an equalisation, a mode) and for these when a few summary
//! numbers do.
//!
//! # Relationship to the integral image
//!
//! [`integral`](crate::analyze::integral) gives local mean and variance over
//! an arbitrary rectangle in `O(1)` per window, which is what adaptive
//! thresholding rides on. It is the right tool for "the mean *around here*";
//! this module is the right tool for "the mean of the frame". Neither
//! subsumes the other, and the integral form pays a full-image accumulator
//! allocation up front.
//!
//! # Example
//!
//! ```
//! use fovea::analyze::statistics::{ChannelStatistics, image_statistics};
//! use fovea::image::Image;
//! use fovea::pixel::{Mono8, Rgb8};
//!
//! // One channel in, one record out.
//! let grey = Image::generate(64, 48, |x, _| Mono8::new((x * 4) as u8));
//! let stats: ChannelStatistics<_> = image_statistics(&grey);
//! assert_eq!(stats.count, 64 * 48);
//! assert_eq!(stats.min().map(|c| c.0), Some(0));
//!
//! // Three channels in, three records out. The output shape is the caller's
//! // choice, exactly as it is for a histogram.
//! let colour = Image::fill(8, 8, Rgb8::new(10, 20, 30));
//! let [r, g, b]: [ChannelStatistics<_>; 3] = image_statistics(&colour);
//! assert_eq!(r.mean(), Some(10.0));
//! assert_eq!(g.mean(), Some(20.0));
//! assert_eq!(b.mean(), Some(30.0));
//! ```

pub mod moments;
pub mod summary;

#[doc(inline)]
pub use moments::{CentralMoments, ImageMoments, NormalizedMoments, image_moments};
#[doc(inline)]
pub use summary::{ChannelStatistics, StatisticsChannel};

use crate::image::RasterImage;
use crate::pixel::HomogeneousPixel;

/// Caller-chosen output shape for [`image_statistics`].
///
/// The same mechanism the histogram uses, for the same reason: a
/// single-channel image should not force the caller to unwrap a one-element
/// `Vec`, and a fixed-channel pixel type should let the array length be
/// checked rather than assumed. Implemented for
/// [`ChannelStatistics<C>`] (single-channel input only),
/// `Vec<ChannelStatistics<C>>` (any channel count) and
/// `[ChannelStatistics<C>; N]` (exactly `N` channels).
pub trait StatisticsOutput<C>: Sized {
    /// Builds the output shape by invoking `compute(channel_index)` once per
    /// channel.
    ///
    /// `channel_count` is `P::CHANNEL_COUNT` from the input image's pixel
    /// type. Implementations decide whether they accept the supplied channel
    /// count and panic otherwise.
    fn collect(channel_count: usize, compute: impl FnMut(usize) -> ChannelStatistics<C>) -> Self;
}

impl<C> StatisticsOutput<C> for ChannelStatistics<C> {
    fn collect(
        channel_count: usize,
        mut compute: impl FnMut(usize) -> ChannelStatistics<C>,
    ) -> Self {
        assert_eq!(
            channel_count, 1,
            "image_statistics() called with output type `ChannelStatistics<C>` on a pixel with \
             {channel_count} channels; use `Vec<ChannelStatistics<C>>` or \
             `[ChannelStatistics<C>; N]` instead",
        );
        compute(0)
    }
}

impl<C> StatisticsOutput<C> for Vec<ChannelStatistics<C>> {
    fn collect(
        channel_count: usize,
        mut compute: impl FnMut(usize) -> ChannelStatistics<C>,
    ) -> Self {
        (0..channel_count).map(&mut compute).collect()
    }
}

impl<C, const N: usize> StatisticsOutput<C> for [ChannelStatistics<C>; N] {
    fn collect(
        channel_count: usize,
        compute: impl FnMut(usize) -> ChannelStatistics<C>,
    ) -> Self {
        assert_eq!(
            channel_count, N,
            "image_statistics() called with output type `[ChannelStatistics<C>; {N}]` on a pixel \
             with {channel_count} channels",
        );
        core::array::from_fn(compute)
    }
}

/// Minimum, maximum, mean, variance and standard deviation of every channel.
///
/// The output shape is the caller's, via [`StatisticsOutput`]: annotate the
/// binding as [`ChannelStatistics`] for a single-channel image, as
/// `[ChannelStatistics<_>; N]` for a pixel type with `N` channels, or as a
/// `Vec` when the channel count is only known generically.
///
/// Nothing here can fail on well-formed input, so there is no `Result`. An
/// image with no pixels, or a float channel that is entirely `NaN`, produces a
/// record whose accessors are [`None`]; see [`ChannelStatistics`].
///
/// # Cost
///
/// One pass per channel, `O(channels · width · height)`, no allocation. The
/// per-channel pass mirrors the histogram engine: for a three-channel image
/// this reads the pixel data three times rather than keeping three
/// accumulators live, which keeps the inner loop a single running summary and
/// costs nothing on the single-channel images the statistics are usually
/// wanted for.
///
/// # Panics
///
/// Panics if the requested output shape does not match the pixel's channel
/// count: a [`ChannelStatistics`] binding on a multi-channel image, or an
/// `[ChannelStatistics; N]` whose `N` is wrong. This is a programmer error
/// visible in the calling line, not a data-dependent failure.
///
/// # Example
///
/// ```
/// use fovea::analyze::statistics::{ChannelStatistics, image_statistics};
/// use fovea::image::Image;
/// use fovea::pixel::MonoF32;
///
/// // A left-to-right ramp over 0.0..1.0 in eight steps.
/// let image = Image::generate(8, 4, |x, _| MonoF32::new(x as f32 / 7.0));
///
/// let stats: ChannelStatistics<_> = image_statistics(&image);
/// assert_eq!(stats.min(), Some(0.0));
/// assert_eq!(stats.max(), Some(1.0));
/// assert!((stats.mean().unwrap() - 0.5).abs() < 1e-6);
/// // A uniform ramp is far from flat, so the deviation is substantial.
/// assert!(stats.std_dev().unwrap() > 0.3);
/// ```
#[must_use]
pub fn image_statistics<I, P, O>(image: &I) -> O
where
    I: RasterImage<Pixel = P>,
    P: HomogeneousPixel,
    P::Channel: StatisticsChannel,
    O: StatisticsOutput<P::Channel>,
{
    O::collect(P::CHANNEL_COUNT, |channel| {
        let mut stats = ChannelStatistics::empty();
        for y in 0..image.height() {
            for pixel in image.row(y) {
                stats.push(pixel.channel(channel));
            }
        }
        stats
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Image;
    use crate::pixel::{Mono8, Mono16, MonoF32, Rgb8, RgbF32};

    #[test]
    fn a_uniform_image_has_zero_spread() {
        let image = Image::fill(10, 7, Mono8::new(42));
        let stats: ChannelStatistics<_> = image_statistics(&image);
        assert_eq!(stats.count, 70);
        assert_eq!(stats.min().map(|c| c.0), Some(42));
        assert_eq!(stats.max().map(|c| c.0), Some(42));
        assert_eq!(stats.mean(), Some(42.0));
        assert_eq!(stats.variance(), Some(0.0));
        assert_eq!(stats.std_dev(), Some(0.0));
    }

    #[test]
    fn an_empty_image_reports_absence() {
        let image: Image<Mono8> = Image::generate(0, 0, |_, _| Mono8::new(0));
        let stats: ChannelStatistics<_> = image_statistics(&image);
        assert_eq!(stats.count, 0);
        assert_eq!(stats.mean(), None);
        assert_eq!(stats.min(), None);
    }

    #[test]
    fn a_zero_height_image_reports_absence_too() {
        let image: Image<Mono8> = Image::generate(16, 0, |_, _| Mono8::new(0));
        let stats: ChannelStatistics<_> = image_statistics(&image);
        assert_eq!(stats.count, 0);
        assert_eq!(stats.mean(), None);
    }

    #[test]
    fn each_channel_is_summarised_independently() {
        // Channel values chosen so a channel mix-up cannot pass.
        let image = Image::generate(4, 4, |x, y| {
            Rgb8::new((x * 10) as u8, (y * 20) as u8, 200)
        });
        let [r, g, b]: [ChannelStatistics<_>; 3] = image_statistics(&image);

        assert_eq!(r.min().map(|c| c.0), Some(0));
        assert_eq!(r.max().map(|c| c.0), Some(30));
        assert_eq!(r.mean(), Some(15.0));

        assert_eq!(g.min().map(|c| c.0), Some(0));
        assert_eq!(g.max().map(|c| c.0), Some(60));
        assert_eq!(g.mean(), Some(30.0));

        assert_eq!(b.min().map(|c| c.0), Some(200));
        assert_eq!(b.variance(), Some(0.0));
    }

    #[test]
    fn the_vec_shape_matches_the_array_shape() {
        let image = Image::generate(3, 3, |x, _| RgbF32::new(x as f32, 1.0, -1.0));
        let listed: Vec<ChannelStatistics<_>> = image_statistics(&image);
        let fixed: [ChannelStatistics<_>; 3] = image_statistics(&image);
        assert_eq!(listed.len(), 3);
        for (a, b) in listed.iter().zip(fixed.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    #[should_panic(expected = "3 channels")]
    fn a_single_record_shape_on_a_colour_image_panics() {
        let image = Image::fill(2, 2, Rgb8::new(1, 2, 3));
        let _stats: ChannelStatistics<_> = image_statistics(&image);
    }

    #[test]
    #[should_panic(expected = "with 3 channels")]
    fn a_wrong_array_length_panics() {
        let image = Image::fill(2, 2, Rgb8::new(1, 2, 3));
        let _stats: [ChannelStatistics<_>; 4] = image_statistics(&image);
    }

    #[test]
    fn nan_pixels_are_counted_not_averaged_in() {
        let image = Image::generate(4, 1, |x, _| {
            MonoF32::new(if x == 2 { f32::NAN } else { 2.0 })
        });
        let stats: ChannelStatistics<_> = image_statistics(&image);
        assert_eq!(stats.count, 3);
        assert_eq!(stats.nan_count, 1);
        assert_eq!(stats.mean(), Some(2.0));
    }

    #[test]
    fn sixteen_bit_input_keeps_its_precision_in_the_mean() {
        // The case an f32 accumulator would lose: 65 535 over a 1-megapixel
        // frame sums to 6.5e10, past f32's exact-integer range.
        let image = Image::generate(1024, 1024, |x, _| {
            Mono16::new(if x % 2 == 0 { 65_535 } else { 65_533 })
        });
        let stats: ChannelStatistics<_> = image_statistics(&image);
        assert_eq!(stats.mean(), Some(65_534.0));
        assert_eq!(stats.min().map(|c| c.0), Some(65_533));
        assert_eq!(stats.max().map(|c| c.0), Some(65_535));
        assert_eq!(stats.variance(), Some(1.0));
    }

    #[test]
    fn the_generic_mono_family_is_accepted() {
        use crate::pixel::Mono;

        let image = Image::generate(8, 8, |x, _| Mono::<12>::new((x * 500) as u16));
        let stats: ChannelStatistics<_> = image_statistics(&image);
        assert_eq!(stats.count, 64);
        assert_eq!(stats.min().map(|c| c.0), Some(0));
        assert_eq!(stats.max().map(|c| c.0), Some(3500));
    }
}

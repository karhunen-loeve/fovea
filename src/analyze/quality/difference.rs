//! Pixel-value difference metrics: sum and mean squared error, RMSE, PSNR,
//! and the largest single-pixel deviation.
//!
//! One pass over both images fills a [`ChannelSquaredError`] per channel, and
//! every metric in this family is an accessor on that record. See the
//! [module docs](super) for why the two images must share a pixel type and for
//! where [`PeakValue`] comes from.

use crate::Error;
use crate::analyze::statistics::StatisticsChannel;
use crate::image::RasterImage;
use crate::pixel::HomogeneousPixel;

use super::PeakValue;

/// Accumulated squared difference between one channel of two images.
///
/// The record every value-difference metric is read off: [`mean_squared_error`]
/// is the MSE, [`root_mean_squared_error`] the RMSE,
/// [`peak_signal_to_noise_ratio`] the PSNR, and
/// [`max_absolute_error`] the worst single pixel. Produced by
/// [`squared_error`], either per channel or [pooled](SquaredError::pooled).
///
/// # Why the accessors return `Option`
///
/// An image pair with no comparable samples — two empty images, or two float
/// images whose every difference is `NaN` — has no mean error, which is
/// absence rather than failure. The counts are always available:
/// [`count`](Self::count) is how many pixel pairs the metrics rest on and
/// [`nan_count`](Self::nan_count) how many were excluded, so
/// `count + nan_count` is the pixel count. This is the same contract as
/// [`ChannelStatistics`](crate::analyze::statistics::ChannelStatistics), for
/// the same reason.
///
/// [`mean_squared_error`]: Self::mean_squared_error
/// [`root_mean_squared_error`]: Self::root_mean_squared_error
/// [`peak_signal_to_noise_ratio`]: Self::peak_signal_to_noise_ratio
/// [`max_absolute_error`]: Self::max_absolute_error
///
/// # Example
///
/// ```
/// use fovea::analyze::quality::{PeakValue, squared_error};
/// use fovea::image::Image;
/// use fovea::pixel::Mono8;
///
/// // Four pixels, each off by 2, so every squared difference is 4.
/// let a = Image::fill(2, 2, Mono8::new(10));
/// let b = Image::fill(2, 2, Mono8::new(12));
///
/// let error = squared_error(&a, &b)?.pooled();
/// assert_eq!(error.count, 4);
/// assert_eq!(error.sum_squared_error(), 16.0);
/// assert_eq!(error.mean_squared_error(), Some(4.0));
/// assert_eq!(error.root_mean_squared_error(), Some(2.0));
/// assert_eq!(error.max_absolute_error(), Some(2.0));
///
/// // 10·log10(255² / 4) ≈ 42.11 dB.
/// let psnr = error.peak_signal_to_noise_ratio(PeakValue::of_pixel::<Mono8>()).unwrap();
/// assert!((psnr - 42.11).abs() < 0.01, "{psnr}");
/// # Ok::<(), fovea::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChannelSquaredError {
    /// Pixel pairs included in every metric below: the pixel count minus
    /// [`nan_count`](Self::nan_count).
    pub count: u64,

    /// Pixel pairs excluded because their difference is not a number. Always
    /// 0 for integer channels.
    ///
    /// The test is on the *difference*, not on the inputs, so this also counts
    /// the pair `(inf, inf)`, whose difference does not exist even though
    /// neither value is `NaN`. Counted rather than dropped, so a partly
    /// invalid comparison is distinguishable from a clean one.
    pub nan_count: u64,

    /// Running `Σ (a − b)²` over the included pairs.
    sum_squared_error: f64,

    /// Running `max |a − b|` over the included pairs. Meaningless when
    /// `count == 0`.
    max_absolute_error: f64,
}

impl ChannelSquaredError {
    /// An empty record, before any pair is folded in.
    #[inline]
    pub(crate) fn empty() -> Self {
        Self {
            count: 0,
            nan_count: 0,
            sum_squared_error: 0.0,
            max_absolute_error: 0.0,
        }
    }

    /// Folds one pair of channel values in.
    ///
    /// Accumulated in `f64` whatever the channel width, for the reason
    /// [`StatisticsChannel`] gives: an `f32` accumulator carries 24 significand
    /// bits, which a megapixel frame of 16-bit squared differences overruns
    /// long before the sum is complete.
    ///
    /// Unlike a variance, a sum of squares about a *known* zero forms no
    /// difference of large nearly-equal numbers, so there is nothing here for
    /// Welford's recurrence to protect and the direct sum is exact to `f64`.
    #[inline]
    pub(crate) fn push<C: StatisticsChannel>(&mut self, a: C, b: C) {
        let difference = a.to_f64() - b.to_f64();
        if difference.is_nan() {
            self.nan_count += 1;
            return;
        }

        self.count += 1;
        self.sum_squared_error += difference * difference;
        let magnitude = difference.abs();
        if magnitude > self.max_absolute_error {
            self.max_absolute_error = magnitude;
        }
    }

    /// Adds another record's samples into this one, as
    /// [`SquaredError::pooled`] does across channels.
    #[inline]
    fn merge(&mut self, other: &Self) {
        self.count += other.count;
        self.nan_count += other.nan_count;
        self.sum_squared_error += other.sum_squared_error;
        if other.max_absolute_error > self.max_absolute_error {
            self.max_absolute_error = other.max_absolute_error;
        }
    }

    /// `Σ (a − b)²` over the included pairs, in squared channel units.
    ///
    /// Zero both for an identical pair and for a pair with no comparable
    /// samples at all; read [`count`](Self::count) to tell them apart. The
    /// derived metrics below return [`None`] in the second case instead.
    #[doc(alias = "SSE")]
    #[must_use]
    pub fn sum_squared_error(&self) -> f64 {
        self.sum_squared_error
    }

    /// Mean squared error, or [`None`] if no pair was included.
    ///
    /// `Σ (a − b)² / n`, in squared channel units, so it is comparable only
    /// between pairs of the same pixel type. Zero exactly when every included
    /// pair is equal.
    #[doc(alias = "mse")]
    #[doc(alias = "MSE")]
    #[must_use]
    pub fn mean_squared_error(&self) -> Option<f64> {
        (self.count > 0).then(|| self.sum_squared_error / self.count as f64)
    }

    /// Root mean squared error: the square root of
    /// [`mean_squared_error`](Self::mean_squared_error).
    ///
    /// Back in channel units, so "the average pixel is off by this much" is a
    /// sentence you can say about it.
    #[doc(alias = "rmse")]
    #[doc(alias = "RMSE")]
    #[must_use]
    pub fn root_mean_squared_error(&self) -> Option<f64> {
        self.mean_squared_error().map(f64::sqrt)
    }

    /// Largest `|a − b|` over the included pairs, or [`None`] if there were
    /// none.
    ///
    /// The metric a regression test usually wants: a mean error hides a single
    /// catastrophic pixel, and "no pixel moved by more than one level" is a
    /// stronger and more legible claim than a small MSE.
    #[must_use]
    pub fn max_absolute_error(&self) -> Option<f64> {
        (self.count > 0).then_some(self.max_absolute_error)
    }

    /// Peak signal-to-noise ratio in decibels, or [`None`] if no pair was
    /// included.
    ///
    /// `10 · log10(peak² / MSE)`, with `peak` the dynamic range of the
    /// representation rather than of the data — see [`PeakValue`].
    ///
    /// # Identical images are infinite, not an error
    ///
    /// An MSE of zero gives [`f64::INFINITY`], which is what the definition
    /// says and what every other library reports. It is a real value, not a
    /// sentinel: `psnr > 40.0` behaves correctly on it, and `is_finite` is the
    /// test for "the two images actually differ". `None` is reserved for
    /// having measured nothing at all.
    #[doc(alias = "psnr")]
    #[doc(alias = "PSNR")]
    #[must_use]
    pub fn peak_signal_to_noise_ratio(&self, peak: PeakValue) -> Option<f64> {
        let mse = self.mean_squared_error()?;
        if mse == 0.0 {
            return Some(f64::INFINITY);
        }
        let peak = peak.get();
        Some(10.0 * (peak * peak / mse).log10())
    }
}

/// Squared difference between two images, one record per channel.
///
/// Produced by [`squared_error`]. Read a single channel with
/// [`channel`](Self::channel), all of them with
/// [`channels`](Self::channels), or the whole image at once with
/// [`pooled`](Self::pooled).
///
/// # Why per channel, and why pooling is explicit
///
/// "The MSE of an RGB image" is ambiguous in the literature: per channel,
/// pooled across channels, or computed on luma. Reporting per channel keeps
/// all of it — pooling is the sum of the per-channel accumulators, so nothing
/// is lost — and makes the choice visible at the call site rather than baked
/// into a function name. It also matters in practice: a demosaic regression
/// suite cares that the *chroma* channels drifted, which a pooled number and a
/// luma number both hide.
///
/// Note that pooling is a sum of accumulators, not an average of results.
/// `pooled().peak_signal_to_noise_ratio(peak)` is the PSNR of the pooled MSE,
/// which is the figure other libraries report for a colour image, and is *not*
/// the mean of the three per-channel PSNRs.
///
/// # Example
///
/// ```
/// use fovea::analyze::quality::squared_error;
/// use fovea::image::Image;
/// use fovea::pixel::Rgb8;
///
/// // Red matches, green is off by 3, blue by 4.
/// let a = Image::fill(4, 4, Rgb8::new(100, 100, 100));
/// let b = Image::fill(4, 4, Rgb8::new(100, 103, 104));
///
/// let error = squared_error(&a, &b)?;
/// assert_eq!(error.channel_count(), 3);
/// assert_eq!(error.channel(0).unwrap().mean_squared_error(), Some(0.0));
/// assert_eq!(error.channel(1).unwrap().mean_squared_error(), Some(9.0));
/// assert_eq!(error.channel(2).unwrap().mean_squared_error(), Some(16.0));
/// assert_eq!(error.channel(3), None);
///
/// // Pooled: 48 pairs, of which 16 contribute 9 and 16 contribute 16.
/// let pooled = error.pooled();
/// assert_eq!(pooled.count, 48);
/// assert_eq!(pooled.mean_squared_error(), Some((16.0 * 9.0 + 16.0 * 16.0) / 48.0));
/// assert_eq!(pooled.max_absolute_error(), Some(4.0));
/// # Ok::<(), fovea::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct SquaredError {
    /// One record per channel of the compared pixel type, in channel order.
    channels: Vec<ChannelSquaredError>,
}

impl SquaredError {
    /// Number of channels compared: `P::CHANNEL_COUNT` of the input pixel
    /// type.
    #[must_use]
    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }

    /// The record for one channel, or [`None`] if `index` names no channel.
    #[must_use]
    pub fn channel(&self, index: usize) -> Option<ChannelSquaredError> {
        self.channels.get(index).copied()
    }

    /// Every channel's record, in channel order.
    #[must_use]
    pub fn channels(&self) -> &[ChannelSquaredError] {
        &self.channels
    }

    /// One record covering every channel at once.
    ///
    /// The per-channel accumulators summed, not the per-channel *results*
    /// averaged: the counts add, the squared-error sums add, and the maximum
    /// is the maximum over all channels. For a single-channel image this is
    /// [`channel(0)`](Self::channel).
    ///
    /// Because every channel of a [`HomogeneousPixel`] has the same sample
    /// count, the pooled MSE does coincide with the mean of the per-channel
    /// MSEs. The pooled PSNR does **not** coincide with the mean of the
    /// per-channel PSNRs, and the pooled one is the conventional figure.
    #[must_use]
    pub fn pooled(&self) -> ChannelSquaredError {
        let mut total = ChannelSquaredError::empty();
        for channel in &self.channels {
            total.merge(channel);
        }
        total
    }
}

/// Squared difference between two images of the same pixel type, per channel.
///
/// The entry point for MSE, RMSE, PSNR and maximum absolute error, all of
/// which are accessors on the returned record — see
/// [`ChannelSquaredError`] and [`SquaredError::pooled`].
///
/// Both images must have the same pixel type, so no range convention is
/// implied between differently-scaled representations; see the
/// [module docs](super). The *image* types are independent, so either side can
/// be an owned image, a borrowed view, or a region of one.
///
/// # `NaN` differences are counted and excluded
///
/// A pair whose difference is not a number is recorded in
/// [`nan_count`](ChannelSquaredError::nan_count) and left out of every metric,
/// which is the policy
/// [`image_statistics`](crate::analyze::statistics::image_statistics) uses,
/// and for the same reason: a summary that has somewhere to record an
/// exclusion should exclude rather than poison. [`ssim`](super::ssim) has nowhere to record one and therefore
/// propagates instead.
///
/// # Cost
///
/// One pass over both images, `O(width · height · channels)`, allocating only
/// the per-channel record vector. Unlike
/// [`image_statistics`](crate::analyze::statistics::image_statistics), which
/// reads a colour image once per channel, this keeps every channel's
/// accumulator live and reads each image exactly once: two images already cost
/// twice the traffic, and re-walking both of them `N` times to keep the inner
/// loop scalar is the worse trade.
///
/// # Errors — Tier 2
///
/// Returns [`Error::SizeMismatch`] if the two images have different
/// dimensions, the same relation
/// [`combine_images`](crate::transform::combine_images) reports.
///
/// # Example
///
/// ```
/// use fovea::analyze::quality::{PeakValue, squared_error};
/// use fovea::image::Image;
/// use fovea::pixel::Mono8;
///
/// let reference = Image::generate(8, 8, |x, _| Mono8::new((x * 30) as u8));
/// let noisy = Image::generate(8, 8, |x, y| {
///     Mono8::new((x * 30) as u8 + if (x + y) % 2 == 0 { 1 } else { 0 })
/// });
///
/// let error = squared_error(&reference, &noisy)?.pooled();
/// // Half the pixels are off by exactly one level.
/// assert_eq!(error.mean_squared_error(), Some(0.5));
/// assert_eq!(error.max_absolute_error(), Some(1.0));
/// assert!(error.peak_signal_to_noise_ratio(PeakValue::of_pixel::<Mono8>()).unwrap() > 50.0);
/// # Ok::<(), fovea::Error>(())
/// ```
pub fn squared_error<A, B, P>(a: &A, b: &B) -> Result<SquaredError, Error>
where
    A: RasterImage<Pixel = P>,
    B: RasterImage<Pixel = P>,
    P: HomogeneousPixel,
    P::Channel: StatisticsChannel,
{
    if a.size() != b.size() {
        return Err(Error::SizeMismatch {
            expected: a.size(),
            actual: b.size(),
        });
    }

    let mut channels = vec![ChannelSquaredError::empty(); P::CHANNEL_COUNT];
    for y in 0..a.height() {
        for (pa, pb) in a.row(y).iter().zip(b.row(y).iter()) {
            for (index, accumulator) in channels.iter_mut().enumerate() {
                accumulator.push(pa.channel(index), pb.channel(index));
            }
        }
    }

    Ok(SquaredError { channels })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::peak;
    use crate::image::{Image, SubView};
    use crate::pixel::{Mono8, Mono16, MonoF32, MonoF64, Rgb8};
    use crate::{Coordinate, Rectangle, Size};

    #[test]
    fn an_identical_pair_has_no_error_and_infinite_psnr() {
        let image = Image::generate(16, 16, |x, y| Mono8::new((x * y) as u8));
        let error = squared_error(&image, &image).unwrap().pooled();

        assert_eq!(error.count, 256);
        assert_eq!(error.nan_count, 0);
        assert_eq!(error.sum_squared_error(), 0.0);
        assert_eq!(error.mean_squared_error(), Some(0.0));
        assert_eq!(error.root_mean_squared_error(), Some(0.0));
        assert_eq!(error.max_absolute_error(), Some(0.0));

        let psnr = error
            .peak_signal_to_noise_ratio(PeakValue::of_pixel::<Mono8>())
            .unwrap();
        assert!(psnr.is_infinite() && psnr > 0.0, "{psnr}");
        // The documented discriminator: infinite is a value, not an absence.
        assert!(!psnr.is_finite());
        assert!(psnr > 40.0);
    }

    #[test]
    fn a_known_offset_gives_the_textbook_mse_and_psnr() {
        // Every pixel off by 10 → MSE 100, PSNR = 10·log10(255²/100).
        let a = Image::fill(10, 10, Mono8::new(100));
        let b = Image::fill(10, 10, Mono8::new(110));
        let error = squared_error(&a, &b).unwrap().pooled();

        assert_eq!(error.mean_squared_error(), Some(100.0));
        assert_eq!(error.root_mean_squared_error(), Some(10.0));
        assert_eq!(error.max_absolute_error(), Some(10.0));

        let expected = 10.0 * (255.0f64 * 255.0 / 100.0).log10();
        let psnr = error
            .peak_signal_to_noise_ratio(PeakValue::of_pixel::<Mono8>())
            .unwrap();
        assert!((psnr - expected).abs() < 1e-12, "{psnr} vs {expected}");
        assert!((psnr - 28.13).abs() < 0.01, "{psnr}");
    }

    #[test]
    fn the_error_is_symmetric_in_its_arguments() {
        let a = Image::generate(9, 7, |x, _| Mono8::new((x * 20) as u8));
        let b = Image::generate(9, 7, |_, y| Mono8::new((y * 30) as u8));
        let forward = squared_error(&a, &b).unwrap().pooled();
        let backward = squared_error(&b, &a).unwrap().pooled();
        assert_eq!(forward, backward);
    }

    #[test]
    fn each_channel_is_measured_independently() {
        // Channel offsets chosen so a mix-up cannot pass.
        let a = Image::fill(4, 4, Rgb8::new(50, 50, 50));
        let b = Image::fill(4, 4, Rgb8::new(51, 53, 57));
        let error = squared_error(&a, &b).unwrap();

        assert_eq!(error.channel_count(), 3);
        assert_eq!(error.channel(0).unwrap().mean_squared_error(), Some(1.0));
        assert_eq!(error.channel(1).unwrap().mean_squared_error(), Some(9.0));
        assert_eq!(error.channel(2).unwrap().mean_squared_error(), Some(49.0));
        assert_eq!(error.channels().len(), 3);
        assert_eq!(error.channel(3), None);
    }

    #[test]
    fn pooling_sums_accumulators_rather_than_averaging_results() {
        let a = Image::fill(4, 4, Rgb8::new(50, 50, 50));
        let b = Image::fill(4, 4, Rgb8::new(51, 53, 57));
        let error = squared_error(&a, &b).unwrap();
        let pooled = error.pooled();

        assert_eq!(pooled.count, 48);
        assert_eq!(pooled.sum_squared_error(), 16.0 * (1.0 + 9.0 + 49.0));
        assert_eq!(pooled.mean_squared_error(), Some((1.0 + 9.0 + 49.0) / 3.0));
        // The maximum is over all channels, not per channel.
        assert_eq!(pooled.max_absolute_error(), Some(7.0));

        // The documented non-identity: the pooled PSNR is not the mean of the
        // per-channel PSNRs.
        let peak = PeakValue::of_pixel::<Rgb8>();
        let pooled_psnr = pooled.peak_signal_to_noise_ratio(peak).unwrap();
        let mean_of_psnrs = error
            .channels()
            .iter()
            .map(|c| c.peak_signal_to_noise_ratio(peak).unwrap())
            .sum::<f64>()
            / 3.0;
        assert!(
            (pooled_psnr - mean_of_psnrs).abs() > 1.0,
            "{pooled_psnr} vs {mean_of_psnrs}"
        );
    }

    #[test]
    fn pooling_a_single_channel_image_is_that_channel() {
        let a = Image::fill(3, 3, Mono8::new(7));
        let b = Image::fill(3, 3, Mono8::new(9));
        let error = squared_error(&a, &b).unwrap();
        assert_eq!(error.pooled(), error.channel(0).unwrap());
    }

    #[test]
    fn a_size_mismatch_is_a_tier_two_error() {
        let a: Image<Mono8> = Image::fill(4, 4, Mono8::new(0));
        let b: Image<Mono8> = Image::fill(4, 5, Mono8::new(0));
        assert_eq!(
            squared_error(&a, &b),
            Err(Error::SizeMismatch {
                expected: Size::new(4, 4),
                actual: Size::new(4, 5),
            })
        );
    }

    #[test]
    fn an_empty_pair_reports_absence_not_zero() {
        let a: Image<Mono8> = Image::generate(0, 0, |_, _| Mono8::new(0));
        let b: Image<Mono8> = Image::generate(0, 0, |_, _| Mono8::new(0));
        let error = squared_error(&a, &b).unwrap().pooled();

        assert_eq!(error.count, 0);
        assert_eq!(error.sum_squared_error(), 0.0);
        assert_eq!(error.mean_squared_error(), None);
        assert_eq!(error.root_mean_squared_error(), None);
        assert_eq!(error.max_absolute_error(), None);
        assert_eq!(
            error.peak_signal_to_noise_ratio(PeakValue::of_pixel::<Mono8>()),
            None
        );
    }

    #[test]
    fn a_nan_difference_is_counted_and_excluded() {
        let a = Image::generate(4, 1, |x, _| {
            MonoF32::new(if x == 2 { f32::NAN } else { 1.0 })
        });
        let b = Image::fill(4, 1, MonoF32::new(3.0));
        let error = squared_error(&a, &b).unwrap().pooled();

        assert_eq!(error.count, 3);
        assert_eq!(error.nan_count, 1);
        assert_eq!(error.mean_squared_error(), Some(4.0));
        assert_eq!(error.max_absolute_error(), Some(2.0));
    }

    #[test]
    fn an_all_nan_pair_reports_absence() {
        let a = Image::fill(2, 2, MonoF64::new(f64::NAN));
        let b = Image::fill(2, 2, MonoF64::new(0.0));
        let error = squared_error(&a, &b).unwrap().pooled();

        assert_eq!(error.count, 0);
        assert_eq!(error.nan_count, 4);
        assert_eq!(error.mean_squared_error(), None);
    }

    #[test]
    fn two_equal_infinities_have_no_difference_and_are_excluded() {
        // The case testing the *inputs* for NaN would miss: neither value is
        // NaN, but `inf − inf` is, so there is no difference to record.
        let a = Image::fill(2, 1, MonoF64::new(f64::INFINITY));
        let b = Image::fill(2, 1, MonoF64::new(f64::INFINITY));
        let error = squared_error(&a, &b).unwrap().pooled();

        assert_eq!(error.count, 0);
        assert_eq!(error.nan_count, 2);
        assert_eq!(error.mean_squared_error(), None);
    }

    #[test]
    fn a_lone_infinite_difference_is_kept_and_reported_as_infinite() {
        // `inf − 0` is a real, if unbounded, difference. Excluding it would
        // report a finite MSE for an image containing an infinity.
        let a = Image::generate(2, 1, |x, _| {
            MonoF64::new(if x == 0 { f64::INFINITY } else { 1.0 })
        });
        let b = Image::fill(2, 1, MonoF64::new(0.0));
        let error = squared_error(&a, &b).unwrap().pooled();

        assert_eq!(error.count, 2);
        assert_eq!(error.nan_count, 0);
        assert_eq!(error.mean_squared_error(), Some(f64::INFINITY));
        assert_eq!(error.max_absolute_error(), Some(f64::INFINITY));
        let psnr = error
            .peak_signal_to_noise_ratio(peak!(1.0))
            .unwrap();
        assert_eq!(psnr, f64::NEG_INFINITY);
    }

    #[test]
    fn sixteen_bit_differences_keep_their_precision_in_the_sum() {
        // The case an f32 accumulator loses: a megapixel of squared 16-bit
        // differences sums past f32's exact-integer range (2^24).
        let a = Image::fill(1024, 1024, Mono16::new(65_535));
        let b = Image::fill(1024, 1024, Mono16::new(0));
        let error = squared_error(&a, &b).unwrap().pooled();

        let expected_sum = 65_535.0f64 * 65_535.0 * 1024.0 * 1024.0;
        assert_eq!(error.sum_squared_error(), expected_sum);
        assert_eq!(error.mean_squared_error(), Some(65_535.0 * 65_535.0));
        // Worst case for the representation: PSNR is exactly 0 dB.
        let psnr = error
            .peak_signal_to_noise_ratio(PeakValue::of_pixel::<Mono16>())
            .unwrap();
        assert!(psnr.abs() < 1e-12, "{psnr}");
    }

    #[test]
    fn the_max_absolute_error_survives_a_small_mean() {
        // One bad pixel in 4096 barely moves the MSE and fully shows up here,
        // which is the reason this accessor exists.
        let a: Image<Mono8> = Image::fill(64, 64, Mono8::new(0));
        let mut b = a.clone();
        {
            use crate::image::ImageViewMut;
            *b.pixel_at_mut(31, 31) = Mono8::new(255);
        }
        let error = squared_error(&a, &b).unwrap().pooled();

        assert!(error.mean_squared_error().unwrap() < 16.0);
        assert_eq!(error.max_absolute_error(), Some(255.0));
    }

    #[test]
    fn a_region_of_view_compares_against_an_owned_image() {
        // The two image types are independent, which is what lets a crop be
        // compared against a reference tile.
        let big = Image::generate(16, 16, |x, y| Mono8::new((x + y) as u8));
        let tile = Image::generate(4, 4, |x, y| Mono8::new((x + 2 + y + 3) as u8));
        let view = big
            .roi(Rectangle::new(Coordinate::new(2, 3), Size::new(4, 4)))
            .expect("in bounds");

        let error = squared_error(&view, &tile).unwrap().pooled();
        assert_eq!(error.count, 16);
        assert_eq!(error.mean_squared_error(), Some(0.0));
    }

    #[test]
    fn signed_channels_are_compared_too() {
        let a: Image<i16> = Image::fill(2, 2, -100);
        let b: Image<i16> = Image::fill(2, 2, 100);
        let error = squared_error(&a, &b).unwrap().pooled();
        assert_eq!(error.mean_squared_error(), Some(40_000.0));
        assert_eq!(error.max_absolute_error(), Some(200.0));
    }
}

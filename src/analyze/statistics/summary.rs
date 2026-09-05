//! Per-channel value summaries: extent, centre and spread.
//!
//! [`ChannelStatistics`] is the record and [`StatisticsChannel`] is the set of
//! channel types it can be computed over. The entry point that produces one
//! record per channel is
//! [`image_statistics`](crate::analyze::statistics::image_statistics).

use core::num::Saturating;

/// Sealing module for [`StatisticsChannel`].
mod statistics_channel_sealed {
    pub trait Sealed: Copy {}
}

/// Channel types that can be summarised: ordered, and widenable to `f64`.
///
/// Every channel type the crate's pixels use is a member: the eight
/// `Saturating<…>` integer widths, the two floats, and the bare integer
/// scalars that are their own channel. This trait is **sealed**, so it cannot
/// be implemented outside this crate, and membership is the whole
/// specification.
///
/// # Why `f64` and not the channel's own accumulator
///
/// A mean is not a pixel value. `Mono8`'s linear accumulator is `f32`, which
/// carries 24 significand bits: enough for one pixel, not for the running sum
/// of a 12-megapixel frame. Every statistic is therefore accumulated in `f64`
/// whatever the input width, and only [`min`] and [`max`], which are
/// selections rather than sums, come back in the channel's own type.
///
/// `f64` has 53 significand bits, so a `u64` channel value above `2^53`
/// widens with loss. That is inherent to reporting a mean as a float, and it
/// is the only place this trait loses data.
///
/// [`min`]: ChannelStatistics::min
/// [`max`]: ChannelStatistics::max
pub trait StatisticsChannel: statistics_channel_sealed::Sealed + PartialOrd + Copy {
    /// This value widened to `f64` for accumulation.
    fn to_f64(self) -> f64;

    /// Whether this value is a floating-point `NaN`.
    ///
    /// Always `false` for integer channels, where the concept does not exist.
    /// `NaN` samples are counted separately and excluded from every statistic;
    /// see [`ChannelStatistics::nan_count`].
    fn is_nan(self) -> bool;
}

/// Implements [`StatisticsChannel`] for integer channel types, which have no
/// `NaN` and widen with a plain cast.
macro_rules! impl_integer_statistics_channel {
    ($($t:ty => |$value:ident| $widen:expr),+ $(,)?) => {
        $(
            impl statistics_channel_sealed::Sealed for $t {}
            impl StatisticsChannel for $t {
                #[inline(always)]
                fn to_f64(self) -> f64 {
                    let $value = self;
                    $widen as f64
                }

                #[inline(always)]
                fn is_nan(self) -> bool {
                    false
                }
            }
        )+
    };
}

impl_integer_statistics_channel! {
    u8 => |v| v,
    u16 => |v| v,
    u32 => |v| v,
    u64 => |v| v,
    i8 => |v| v,
    i16 => |v| v,
    i32 => |v| v,
    i64 => |v| v,
    Saturating<u8> => |v| v.0,
    Saturating<u16> => |v| v.0,
    Saturating<u32> => |v| v.0,
    Saturating<u64> => |v| v.0,
    Saturating<i8> => |v| v.0,
    Saturating<i16> => |v| v.0,
    Saturating<i32> => |v| v.0,
    Saturating<i64> => |v| v.0,
}

impl statistics_channel_sealed::Sealed for f32 {}
impl StatisticsChannel for f32 {
    #[inline(always)]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }

    #[inline(always)]
    fn is_nan(self) -> bool {
        f32::is_nan(self)
    }
}

impl statistics_channel_sealed::Sealed for f64 {}
impl StatisticsChannel for f64 {
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self
    }

    #[inline(always)]
    fn is_nan(self) -> bool {
        f64::is_nan(self)
    }
}

/// Extent, centre and spread of one channel of one image.
///
/// The five numbers image analysis asks for first (minimum, maximum, mean,
/// variance and standard deviation) plus the counts that say how many samples
/// each of them rests on.
///
/// # Why the accessors return `Option`
///
/// An image with no pixels has no minimum and no mean, and neither does a
/// float image whose every sample is `NaN`. That is absence rather than
/// failure, so it is [`Option`] rather than an error or a sentinel value. The
/// counts are always available: [`count`](Self::count) is how many samples the
/// statistics rest on and [`nan_count`](Self::nan_count) is how many were
/// excluded, so `count + nan_count` is the pixel count.
///
/// # Population, not sample
///
/// [`variance`](Self::variance) and [`std_dev`](Self::std_dev) divide by `n`,
/// because an image is the whole population rather than a draw from a larger
/// one, and that is what image-processing libraries report. The unbiased
/// `n − 1` forms are [`sample_variance`](Self::sample_variance) and
/// [`sample_std_dev`](Self::sample_std_dev), for when the pixels genuinely are
/// a sample.
///
/// # Example
///
/// ```
/// use fovea::analyze::statistics::{ChannelStatistics, image_statistics};
/// use fovea::image::Image;
/// use fovea::pixel::Mono8;
///
/// let image = Image::from_vec(2, 2, vec![
///     Mono8::new(0),
///     Mono8::new(2),
///     Mono8::new(4),
///     Mono8::new(6),
/// ])?;
///
/// let stats: ChannelStatistics<_> = image_statistics(&image);
/// assert_eq!(stats.count, 4);
/// assert_eq!(stats.min().map(|c| c.0), Some(0));
/// assert_eq!(stats.max().map(|c| c.0), Some(6));
/// assert_eq!(stats.mean(), Some(3.0));
/// assert_eq!(stats.variance(), Some(5.0)); // mean of (9, 1, 1, 9)
/// # Ok::<(), fovea::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChannelStatistics<C> {
    /// Samples included in every statistic below: the pixel count minus
    /// [`nan_count`](Self::nan_count).
    pub count: u64,

    /// Samples excluded because they were `NaN`. Always 0 for integer
    /// channels.
    ///
    /// Counted rather than dropped, so that a partly-invalid image is
    /// distinguishable from a clean one. A mean over 900 of 1000 pixels is a
    /// different claim from a mean over all of them.
    pub nan_count: u64,

    /// Smallest sample seen, in the channel's own type.
    min: Option<C>,

    /// Largest sample seen, in the channel's own type.
    max: Option<C>,

    /// Running mean (Welford). Meaningless when `count == 0`.
    mean: f64,

    /// Running `Σ (xᵢ − x̄ᵢ)(xᵢ − x̄ᵢ₋₁)` (Welford's `M₂`), the numerator of
    /// both variance forms.
    sum_squared_deviations: f64,
}

impl<C: StatisticsChannel> ChannelStatistics<C> {
    /// An empty summary, before any sample is folded in.
    #[inline]
    pub(crate) fn empty() -> Self {
        Self {
            count: 0,
            nan_count: 0,
            min: None,
            max: None,
            mean: 0.0,
            sum_squared_deviations: 0.0,
        }
    }

    /// Folds one sample in.
    ///
    /// Mean and variance use **Welford's** recurrence rather than
    /// `Σx² / n − (Σx / n)²`. The textbook form subtracts two large,
    /// nearly-equal numbers and loses catastrophically on the case this crate
    /// actually meets: a 16-bit image with a small spread about a large
    /// offset, where `Σx²` is around `10¹⁴` and the variance is single digits.
    /// Welford accumulates the deviations directly and never forms that
    /// difference.
    #[inline]
    pub(crate) fn push(&mut self, value: C) {
        if value.is_nan() {
            self.nan_count += 1;
            return;
        }

        // `value` is not NaN, so `PartialOrd` is a total order over the
        // samples that reach here, and the `_ =>` arm is the "first sample"
        // case rather than an incomparable one.
        self.min = Some(match self.min {
            Some(current) if current <= value => current,
            _ => value,
        });
        self.max = Some(match self.max {
            Some(current) if current >= value => current,
            _ => value,
        });

        self.count += 1;
        let sample = value.to_f64();
        let delta = sample - self.mean;
        self.mean += delta / self.count as f64;
        self.sum_squared_deviations += delta * (sample - self.mean);
    }

    /// Smallest sample, or [`None`] if no sample was included.
    ///
    /// In the channel's own type rather than widened: this is a selection from
    /// the data, so a `u64` extreme comes back exact where
    /// [`mean`](Self::mean) could not.
    #[must_use]
    pub fn min(&self) -> Option<C> {
        self.min
    }

    /// Largest sample, or [`None`] if no sample was included. See
    /// [`min`](Self::min) on the return type.
    #[must_use]
    pub fn max(&self) -> Option<C> {
        self.max
    }

    /// Arithmetic mean, or [`None`] if no sample was included.
    #[must_use]
    pub fn mean(&self) -> Option<f64> {
        (self.count > 0).then_some(self.mean)
    }

    /// Population variance (`÷ n`), or [`None`] if no sample was included.
    ///
    /// A single-sample image has variance `0` rather than [`None`], because
    /// one pixel genuinely has no spread. It is
    /// [`sample_variance`](Self::sample_variance) that is undefined there.
    #[must_use]
    pub fn variance(&self) -> Option<f64> {
        (self.count > 0).then(|| self.sum_squared_deviations / self.count as f64)
    }

    /// Population standard deviation, the square root of
    /// [`variance`](Self::variance).
    #[must_use]
    pub fn std_dev(&self) -> Option<f64> {
        self.variance().map(f64::sqrt)
    }

    /// Unbiased sample variance (`÷ (n − 1)`), or [`None`] with fewer than two
    /// samples.
    #[must_use]
    pub fn sample_variance(&self) -> Option<f64> {
        (self.count > 1).then(|| self.sum_squared_deviations / (self.count - 1) as f64)
    }

    /// Unbiased sample standard deviation, the square root of
    /// [`sample_variance`](Self::sample_variance).
    #[must_use]
    pub fn sample_std_dev(&self) -> Option<f64> {
        self.sample_variance().map(f64::sqrt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fold<C: StatisticsChannel>(values: &[C]) -> ChannelStatistics<C> {
        let mut stats = ChannelStatistics::empty();
        for &value in values {
            stats.push(value);
        }
        stats
    }

    #[test]
    fn an_empty_summary_reports_absence_not_zero() {
        let stats = fold::<f32>(&[]);
        assert_eq!(stats.count, 0);
        assert_eq!(stats.nan_count, 0);
        assert_eq!(stats.min(), None);
        assert_eq!(stats.max(), None);
        assert_eq!(stats.mean(), None);
        assert_eq!(stats.variance(), None);
        assert_eq!(stats.std_dev(), None);
        assert_eq!(stats.sample_variance(), None);
    }

    #[test]
    fn a_single_sample_has_a_mean_and_zero_variance() {
        let stats = fold(&[7.0f64]);
        assert_eq!(stats.mean(), Some(7.0));
        assert_eq!(stats.min(), Some(7.0));
        assert_eq!(stats.max(), Some(7.0));
        assert_eq!(stats.variance(), Some(0.0));
        // The unbiased form divides by zero and is therefore absent.
        assert_eq!(stats.sample_variance(), None);
    }

    #[test]
    fn the_two_variance_forms_differ_by_the_bessel_factor() {
        let stats = fold(&[2.0f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        assert_eq!(stats.mean(), Some(5.0));
        assert_eq!(stats.variance(), Some(4.0));
        assert_eq!(stats.std_dev(), Some(2.0));
        // n / (n − 1) · 4 = 8/7 · 4
        let sample = stats.sample_variance().unwrap();
        assert!((sample - 32.0 / 7.0).abs() < 1e-12, "{sample}");
    }

    #[test]
    fn nan_samples_are_counted_and_excluded() {
        let stats = fold(&[1.0f32, f32::NAN, 3.0, f32::NAN]);
        assert_eq!(stats.count, 2);
        assert_eq!(stats.nan_count, 2);
        assert_eq!(stats.min(), Some(1.0));
        assert_eq!(stats.max(), Some(3.0));
        assert_eq!(stats.mean(), Some(2.0));
        assert_eq!(stats.variance(), Some(1.0));
    }

    #[test]
    fn an_all_nan_channel_reports_absence() {
        let stats = fold(&[f64::NAN, f64::NAN]);
        assert_eq!(stats.count, 0);
        assert_eq!(stats.nan_count, 2);
        assert_eq!(stats.mean(), None);
        assert_eq!(stats.min(), None);
    }

    #[test]
    fn welford_survives_a_large_offset_that_defeats_the_textbook_form() {
        // The regression this algorithm choice exists for: values around
        // 65 000 with a spread of 1. `Σx² / n − mean²` computes
        // 4.2e9 − 4.2e9 in f64 and returns a visibly wrong variance (often 0
        // or negative); Welford is exact here.
        let values: Vec<f64> = (0..1000).map(|i| 65_000.0 + (i % 3) as f64).collect();
        let stats = fold(&values);
        // 334 zeros, 333 ones and 333 twos, so the mean sits ≈0.999 above
        // the offset.
        let expected_mean = 65_000.0 + (333 + 666) as f64 / 1000.0;
        assert!((stats.mean().unwrap() - expected_mean).abs() < 1e-9);

        let reference = values
            .iter()
            .map(|v| (v - expected_mean) * (v - expected_mean))
            .sum::<f64>()
            / 1000.0;
        let variance = stats.variance().unwrap();
        assert!(
            (variance - reference).abs() < 1e-9,
            "{variance} vs {reference}"
        );
        assert!(
            variance > 0.6,
            "a real spread must not collapse: {variance}"
        );
    }

    #[test]
    fn integer_channels_have_no_nan_and_report_exact_extremes() {
        let stats = fold(&[
            Saturating(10u8),
            Saturating(200u8),
            Saturating(0u8),
            Saturating(50u8),
        ]);
        assert_eq!(stats.nan_count, 0);
        assert_eq!(stats.min(), Some(Saturating(0u8)));
        assert_eq!(stats.max(), Some(Saturating(200u8)));
        assert_eq!(stats.mean(), Some(65.0));
    }

    #[test]
    fn a_u64_extreme_survives_in_the_channel_type_where_the_mean_cannot() {
        // 2^53 + 1 is not representable in f64, so a widened extreme would
        // come back wrong. `min` and `max` are selections and stay exact.
        let big = (1u64 << 53) + 1;
        let stats = fold(&[big, big - 2]);
        assert_eq!(stats.max(), Some(big));
        assert_eq!(stats.min(), Some(big - 2));
    }

    #[test]
    fn signed_channels_are_summarised_too() {
        let stats = fold(&[Saturating(-5i16), Saturating(15i16)]);
        assert_eq!(stats.min(), Some(Saturating(-5i16)));
        assert_eq!(stats.max(), Some(Saturating(15i16)));
        assert_eq!(stats.mean(), Some(5.0));
        assert_eq!(stats.variance(), Some(100.0));
    }
}

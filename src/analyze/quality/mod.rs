//! Image quality metrics: how far one image is from another.
//!
//! Every function here takes **two images of the same pixel type** and
//! reports how much they differ. That is the one thing the rest of
//! [`analyze`](crate::analyze) cannot do: a histogram, a statistic and a
//! moment each describe a single image, so none of them can answer "did this
//! pipeline change the picture, and by how much".
//!
//! | Question | Use | Output |
//! |---|---|---|
//! | "How far apart are these pixel values?" | [`squared_error`] | [`SquaredError`], per channel and pooled |
//! | "What is the MSE / RMSE / PSNR?" | accessors on [`ChannelSquaredError`] | [`f64`], or [`None`] with no samples |
//! | "Do they *look* alike, structurally?" | [`ssim`] / [`ssim_map`] | One score, or a per-position map |
//!
//! # Why both images must be the same pixel type
//!
//! An error between a `Mono8` and a `MonoF32` needs a range convention — is
//! `255` the same brightness as `1.0`? — and this crate deliberately does not
//! have one: float pixels carry no intrinsic full scale. So the pixel type is
//! unified in the signature and the *conversion* is the caller's, made by name
//! through [`convert_image`](crate::transform::convert_image). The two *image*
//! types stay free, so comparing an [`Image`](crate::image::Image) against a
//! region of view of another image works.
//!
//! # The full-scale value is a parameter, sometimes a type-level one
//!
//! PSNR and SSIM both need to know what "full scale" means: PSNR divides by
//! it, SSIM's stabilizing constants are fractions of it. For integer pixels
//! the pixel type knows the answer, so [`PeakValue::of_pixel`] reads it off
//! the type, and reads `1023`, not `65535`, for `Mono<10>`. Float pixels have
//! no intrinsic full scale in this crate, so they do not implement
//! [`WhiteChannel`] and cannot use that
//! constructor; a float caller names the peak with [`peak!`](crate::peak) and owns
//! the assumption.
//!
//! # Example
//!
//! ```
//! use fovea::analyze::quality::{PeakValue, SsimParams, squared_error, ssim};
//! use fovea::image::{Image, ImageViewMut};
//! use fovea::pixel::Mono8;
//!
//! // A reference frame and the same frame with one pixel knocked out.
//! let reference = Image::generate(32, 32, |x, y| Mono8::new(((x * 8) ^ (y * 4)) as u8));
//! let mut degraded = reference.clone();
//! *degraded.pixel_at_mut(16, 16) = Mono8::new(0);
//!
//! let peak = PeakValue::of_pixel::<Mono8>();
//! assert_eq!(peak.get(), 255.0);
//!
//! // One pixel of 1024 dropped by 192 levels: MSE = 192² / 1024 = 36, so
//! // PSNR = 10·log10(255² / 36) ≈ 32.6 dB.
//! let error = squared_error(&reference, &degraded)?.pooled();
//! assert_eq!(error.count, 32 * 32);
//! assert_eq!(error.mean_squared_error(), Some(36.0));
//! assert_eq!(error.max_absolute_error(), Some(192.0));
//! assert!(error.peak_signal_to_noise_ratio(peak).unwrap() > 32.0);
//!
//! // Structurally the two are still nearly the same image.
//! let score = ssim(&reference, &degraded, SsimParams::reference(peak))?;
//! assert!(score > 0.9, "{score}");
//!
//! // An image compared against itself is exactly 1.0, and has no error.
//! assert_eq!(ssim(&reference, &reference, SsimParams::reference(peak))?, 1.0);
//! assert_eq!(
//!     squared_error(&reference, &reference)?.pooled().mean_squared_error(),
//!     Some(0.0),
//! );
//! # Ok::<(), fovea::Error>(())
//! ```

pub mod difference;
pub mod structural;

#[doc(inline)]
pub use difference::{ChannelSquaredError, SquaredError, squared_error};
#[doc(inline)]
pub use structural::{SsimParams, ssim, ssim_map};

use crate::Error;
use crate::analyze::statistics::StatisticsChannel;
use crate::pixel::WhiteChannel;

/// The value a quality metric treats as full scale: finite and strictly
/// positive.
///
/// PSNR's numerator and SSIM's `C1` / `C2` are both expressed in units of the
/// signal's dynamic range, conventionally written `L`. `PeakValue` is that
/// number as an invariant-carrying parameter type, the same discipline as
/// [`Sigma`](crate::Sigma) and [`Tolerance`](crate::Tolerance): validation
/// happens once, where the value is born, and every metric taking a
/// `PeakValue` is total in it.
///
/// Three ways in, in decreasing order of how much the type system helps:
///
/// - [`of_pixel`](Self::of_pixel) reads the pixel type's own saturated value.
///   Correct by construction, and correct for reduced-range pixels: `Mono<10>`
///   reports `1023`.
/// - [`peak!`](crate::peak) takes a literal and checks it at compile time.
/// - [`try_new`](Self::try_new) takes a computed value and reports
///   [`Error::InvalidParameter`].
///
/// [`new`](Self::new) is the checked `const fn` under the macro; it returns
/// [`Option`], so reaching for it by name cannot abort.
///
/// # This is a range, not a maximum sample
///
/// `L` is the dynamic range of the *representation*, not the largest value
/// either image happens to contain. An 8-bit frame whose brightest pixel is
/// 100 still has `L = 255`, and passing `100` would report a PSNR about 8 dB
/// better than every other library would. If you genuinely want the
/// data-derived range, that is
/// [`ChannelStatistics::max`](crate::analyze::statistics::ChannelStatistics::max)
/// minus `min`, and it is a different, non-comparable figure.
///
/// # Example
///
/// ```
/// use fovea::analyze::quality::PeakValue;
/// use fovea::pixel::{Mono, Mono8, Mono16};
///
/// // Read off the pixel type, including the reduced-range families.
/// assert_eq!(PeakValue::of_pixel::<Mono8>().get(), 255.0);
/// assert_eq!(PeakValue::of_pixel::<Mono16>().get(), 65_535.0);
/// assert_eq!(PeakValue::of_pixel::<Mono<10>>().get(), 1023.0);
///
/// // Float pixels have no intrinsic full scale, so the caller names it.
/// use fovea::peak;
/// const UNIT: PeakValue = peak!(1.0);
/// assert_eq!(UNIT.get(), 1.0);
/// ```
///
/// A float pixel type is rejected at compile time rather than guessed:
///
/// ```compile_fail
/// use fovea::analyze::quality::PeakValue;
/// use fovea::pixel::MonoF32;
///
/// // ERROR: `MonoF32: WhiteChannel` is not satisfied.
/// let _peak = PeakValue::of_pixel::<MonoF32>();
/// ```
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct PeakValue(f64);

impl PeakValue {
    /// Creates a `PeakValue`, returning `None` if the value is not finite
    /// and strictly positive.
    ///
    /// This is the `const fn` the [`peak!`](crate::peak) macro wraps. Prefer
    /// the macro for literals and [`Self::try_new`] for values computed from
    /// data.
    #[must_use]
    pub const fn new(value: f64) -> Option<Self> {
        if value.is_finite() && value > 0.0 {
            Some(Self(value))
        } else {
            None
        }
    }

    /// Creates a `PeakValue` from a computed value, validating it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidParameter`] if `value` is zero, negative, NaN
    /// or infinite.
    pub fn try_new(value: f64) -> Result<Self, Error> {
        if value.is_finite() && value > 0.0 {
            Ok(Self(value))
        } else {
            Err(Error::InvalidParameter(format!(
                "peak value must be finite and positive, got {value}"
            )))
        }
    }

    /// The full scale `P` itself declares, via
    /// [`WhiteChannel`].
    ///
    /// This is the pixel-level saturated value rather than the channel type's
    /// storage maximum, which is the distinction that makes `Mono<10>` report
    /// `1023` instead of `65535`. Float-channel pixels do not implement
    /// `WhiteChannel` and therefore cannot reach this constructor at all: the
    /// `[0.0, 1.0]` convention is a call-site assumption, so it is named at
    /// the call site with [`peak!`](crate::peak).
    ///
    /// # Panics
    ///
    /// Panics if the pixel type declares a non-positive or non-finite white
    /// value. No shipped pixel type does; a hand-written `WhiteChannel` impl
    /// that returns zero is a programmer bug in that impl, so it is Tier 3.
    #[must_use]
    pub fn of_pixel<P>() -> Self
    where
        P: WhiteChannel,
        P::Channel: StatisticsChannel,
    {
        Self::try_new(P::white_channel().to_f64()).unwrap_or_else(|_| {
            panic!(
                "PeakValue::of_pixel: {} declares a white value that is not finite and positive",
                core::any::type_name::<P>(),
            )
        })
    }

    /// Returns the raw value.
    #[must_use]
    pub const fn get(self) -> f64 {
        self.0
    }
}

/// A [`PeakValue`](crate::analyze::quality::PeakValue) literal, checked at
/// compile time.
///
/// For integer pixel types prefer
/// [`PeakValue::of_pixel`](crate::analyze::quality::PeakValue::of_pixel),
/// which reads the full scale off the type and cannot disagree with it. This
/// macro is for the float case, where the range is a call-site convention and
/// somebody has to name it.
///
/// The [`sigma!`](crate::sigma) macro's counterpart for full-scale values; see
/// it for why this is a macro and not a `const fn`. A value that is not a
/// constant expression does not compile (`error[E0435]`); use
/// [`PeakValue::try_new`](crate::analyze::quality::PeakValue::try_new) there.
///
/// # Example
///
/// ```
/// use fovea::analyze::quality::PeakValue;
/// use fovea::peak;
///
/// const UNIT: PeakValue = peak!(1.0); // the usual float convention
/// assert_eq!(UNIT.get(), 1.0);
/// ```
///
/// A full scale of zero makes every metric degenerate, so it does not build:
///
/// ```compile_fail
/// use fovea::peak;
/// // ERROR: evaluation panicked: peak value must be finite and strictly
/// // positive
/// let _ = peak!(0.0);
/// ```
#[macro_export]
macro_rules! peak {
    ($value:expr) => {
        const {
            $crate::analyze::quality::PeakValue::new($value)
                .expect("peak value must be finite and strictly positive")
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pixel::{Mono, Mono8, Mono16, Rgb8};

    #[test]
    fn the_pixel_type_supplies_its_own_full_scale() {
        assert_eq!(PeakValue::of_pixel::<Mono8>().get(), 255.0);
        assert_eq!(PeakValue::of_pixel::<Mono16>().get(), 65_535.0);
        assert_eq!(PeakValue::of_pixel::<Rgb8>().get(), 255.0);
    }

    #[test]
    fn a_reduced_range_pixel_reports_its_range_not_its_storage_maximum() {
        // The whole reason `WhiteChannel` exists rather than
        // `BoundedChannel::MAX`: `Mono<10>` stores in `Saturating<u16>`,
        // whose MAX is 65 535, but saturates at 1023.
        assert_eq!(PeakValue::of_pixel::<Mono<10>>().get(), 1023.0);
        assert_eq!(PeakValue::of_pixel::<Mono<12>>().get(), 4095.0);
        assert_eq!(PeakValue::of_pixel::<Mono<14>>().get(), 16_383.0);
    }

    #[test]
    fn a_literal_peak_is_checked_at_compile_time() {
        const UNIT: PeakValue = peak!(1.0);
        assert_eq!(UNIT.get(), 1.0);
    }

    #[test]
    fn a_computed_peak_reports_the_bad_value() {
        assert!(PeakValue::try_new(0.0).is_err());
        assert!(PeakValue::try_new(-1.0).is_err());
        assert!(PeakValue::try_new(f64::NAN).is_err());
        assert!(PeakValue::try_new(f64::INFINITY).is_err());
        assert_eq!(PeakValue::try_new(255.0).map(PeakValue::get), Ok(255.0));
    }

    #[test]
    fn a_non_positive_value_is_rejected() {
        assert!(PeakValue::new(0.0).is_none());
        assert!(PeakValue::new(-1.0).is_none());
        assert!(PeakValue::new(f64::NAN).is_none());
    }
}

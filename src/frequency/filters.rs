//! Filters in the frequency domain: the shapes, the profiles, the transfer
//! functions, and the methods that apply them to a spectrum.

use super::engine::Cplx;
use super::spectrum::{Spectrum, SpectrumSource};
use super::units::{Bins, CyclesPerPixel, Frequency, FrequencyUnit};
use crate::error::{ParameterError, Requirement, Value};
use crate::{AxialOrientation, Error};
use core::marker::PhantomData;

// ── Shapes ──────────────────────────────────────────────────────────────────

/// A radius around a frequency, in the unit `U`: finite and positive.
///
/// The cutoff of a low-pass or high-pass, and the size of a notch. Build
/// one with [`Radius::cycles_per_pixel`] or [`Radius::bins`], with
/// [`try_new`](Self::try_new) where the unit follows from the context, or
/// with [`radius!`](crate::radius) for a literal checked at compile time.
///
/// # Example
///
/// ```
/// use fovea::frequency::{Bins, Radius};
///
/// let cutoff = Radius::cycles_per_pixel(0.125)?;
/// assert_eq!(cutoff.get(), 0.125);
/// assert!(Radius::<Bins>::try_new(0.0).is_err());
/// # Ok::<(), fovea::Error>(())
/// ```
pub struct Radius<U> {
    value: f64,
    unit: PhantomData<U>,
}

impl<U: FrequencyUnit> Radius<U> {
    /// A radius of `value`, or `None` unless it is finite and positive.
    #[must_use]
    pub const fn new(value: f64) -> Option<Self> {
        if value.is_finite() && value > 0.0 {
            Some(Self {
                value,
                unit: PhantomData,
            })
        } else {
            None
        }
    }

    /// A radius of `value`, validated.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidParameter`] if `value` is zero, negative, NaN or
    /// infinite.
    pub fn try_new(value: f64) -> Result<Self, Error> {
        Self::new(value).ok_or_else(|| {
            ParameterError::new("radius", Requirement::FinitePositive, Value::F64(value)).into()
        })
    }

    /// The radius, in the unit `U`.
    #[must_use]
    pub const fn get(&self) -> f64 {
        self.value
    }
}

impl Radius<CyclesPerPixel> {
    /// A radius of `value` cycles per pixel, validated.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidParameter`] if `value` is zero, negative, NaN or
    /// infinite.
    pub fn cycles_per_pixel(value: f64) -> Result<Self, Error> {
        Self::try_new(value)
    }
}

impl Radius<Bins> {
    /// A radius of `value` bins, validated.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidParameter`] if `value` is zero, negative, NaN or
    /// infinite.
    pub fn bins(value: f64) -> Result<Self, Error> {
        Self::try_new(value)
    }
}

/// A [`Radius`] literal, checked at compile time; the unit follows from the
/// context.
///
/// A value that is not a constant expression does not compile
/// (`error[E0435]`); use [`Radius::try_new`](crate::frequency::Radius::try_new)
/// there.
///
/// # Example
///
/// ```
/// use fovea::frequency::{CyclesPerPixel, Radius};
///
/// let cutoff: Radius<CyclesPerPixel> = fovea::radius!(0.25);
/// assert_eq!(cutoff.get(), 0.25);
/// ```
///
/// ```compile_fail
/// // ERROR: evaluation panicked: must be finite and strictly positive
/// let _: fovea::frequency::Radius<fovea::frequency::Bins> = fovea::radius!(-1.0);
/// ```
#[macro_export]
macro_rules! radius {
    ($value:expr) => {
        const {
            $crate::frequency::Radius::new($value)
                .expect($crate::error::Requirement::FinitePositive.text())
        }
    };
}

/// A band of radii, `low` to `high`, in the unit `U`: finite, with
/// `0 ≤ low < high`.
///
/// The band of a band-pass or band-stop. A `low` of zero makes the band
/// a disc. Build one with [`Band::cycles_per_pixel`] or [`Band::bins`],
/// with [`try_new`](Self::try_new), or with [`band!`](crate::band) for a
/// literal checked at compile time.
///
/// # Example
///
/// ```
/// use fovea::frequency::Band;
///
/// // Structures between 16 and 4 pixels in size.
/// let band = Band::cycles_per_pixel(1.0 / 16.0, 1.0 / 4.0)?;
/// assert_eq!((band.low(), band.high()), (0.0625, 0.25));
/// assert!(Band::cycles_per_pixel(0.3, 0.2).is_err());
/// # Ok::<(), fovea::Error>(())
/// ```
pub struct Band<U> {
    low: f64,
    high: f64,
    unit: PhantomData<U>,
}

impl<U: FrequencyUnit> Band<U> {
    /// The band `low` to `high`, or `None` unless both are finite and
    /// `0 ≤ low < high`.
    #[must_use]
    pub const fn new(low: f64, high: f64) -> Option<Self> {
        if low.is_finite() && high.is_finite() && low >= 0.0 && low < high {
            Some(Self {
                low,
                high,
                unit: PhantomData,
            })
        } else {
            None
        }
    }

    /// The band `low` to `high`, validated.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidParameter`] naming `low` if it is negative or not
    /// finite, `high` if it is not finite, and the band if `low < high`
    /// does not hold.
    pub fn try_new(low: f64, high: f64) -> Result<Self, Error> {
        if !(low.is_finite() && low >= 0.0) {
            return Err(ParameterError::new(
                "low",
                Requirement::FiniteNonNegative,
                Value::F64(low),
            )
            .into());
        }
        if !high.is_finite() {
            return Err(ParameterError::new("high", Requirement::Finite, Value::F64(high)).into());
        }
        Self::new(low, high).ok_or_else(|| {
            ParameterError::new(
                "band",
                Requirement::StrictlyOrdered,
                Value::F64Pair(low, high),
            )
            .into()
        })
    }

    /// The lower edge.
    #[must_use]
    pub const fn low(&self) -> f64 {
        self.low
    }

    /// The upper edge.
    #[must_use]
    pub const fn high(&self) -> f64 {
        self.high
    }
}

impl Band<CyclesPerPixel> {
    /// The band `low` to `high` in cycles per pixel, validated.
    ///
    /// # Errors
    ///
    /// As [`try_new`](Self::try_new).
    pub fn cycles_per_pixel(low: f64, high: f64) -> Result<Self, Error> {
        Self::try_new(low, high)
    }
}

impl Band<Bins> {
    /// The band `low` to `high` in bins, validated.
    ///
    /// # Errors
    ///
    /// As [`try_new`](Self::try_new).
    pub fn bins(low: f64, high: f64) -> Result<Self, Error> {
        Self::try_new(low, high)
    }
}

/// A [`Band`] literal, checked at compile time; the unit follows from the
/// context.
///
/// # Example
///
/// ```
/// use fovea::frequency::{Band, CyclesPerPixel};
///
/// let band: Band<CyclesPerPixel> = fovea::band!(0.05, 0.2);
/// assert_eq!(band.high(), 0.2);
/// ```
///
/// ```compile_fail
/// // ERROR: evaluation panicked: must satisfy low < high
/// let _: fovea::frequency::Band<fovea::frequency::Bins> = fovea::band!(3.0, 2.0);
/// ```
#[macro_export]
macro_rules! band {
    ($low:expr, $high:expr) => {
        const {
            $crate::frequency::Band::new($low, $high)
                .expect($crate::error::Requirement::StrictlyOrdered.text())
        }
    };
}

macro_rules! impl_plain_traits {
    ($($ty:ident),*) => {$(
        impl<U> Clone for $ty<U> {
            fn clone(&self) -> Self {
                *self
            }
        }
        impl<U> Copy for $ty<U> {}
    )*};
}

impl_plain_traits!(Radius, Band);

impl<U> PartialEq for Radius<U> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<U> PartialEq for Band<U> {
    fn eq(&self, other: &Self) -> bool {
        (self.low, self.high) == (other.low, other.high)
    }
}

impl<U> core::fmt::Debug for Radius<U> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("Radius").field(&self.value).finish()
    }
}

impl<U> core::fmt::Debug for Band<U> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Band")
            .field("low", &self.low)
            .field("high", &self.high)
            .finish()
    }
}

// ── Profiles ────────────────────────────────────────────────────────────────

mod sealed {
    /// The low-pass gain of a profile at `ratio` = distance / cutoff.
    pub trait Shape: Copy {
        fn low_pass(&self, ratio: f64) -> f64;
    }
}

/// How a filter passes from one to zero: [`Ideal`], [`Butterworth`] or
/// [`Gaussian`]. Sealed.
///
/// Every smooth profile halves the amplitude at the cutoff, so exchanging
/// one for another changes the slope of the edge and not its place. As a
/// low-pass, at a distance D from the centre and a cutoff D₀:
///
/// | Profile | Gain | At D₀ |
/// |---|---|---|
/// | [`Ideal`] | 1 up to D₀, 0 beyond | 1 |
/// | [`Butterworth<N>`](Butterworth) | 1 / (1 + (D/D₀)²ᴺ) | 1/2 |
/// | [`Gaussian`] | 2^−(D/D₀)² | 1/2 |
///
/// A high-pass is one minus the low-pass, a band-pass the difference of
/// two low-passes, and a band-stop one minus the band-pass.
pub trait Profile: sealed::Shape {}

/// The ideal profile: the gain jumps from one to zero at the cutoff.
///
/// The sharpest edge, and the one that rings: its response to a step in
/// the image overshoots next to the step and oscillates away from it, the
/// Gibbs phenomenon. Measured for a low-pass at 0.1 cycles per pixel on a
/// step across a 256-pixel row: an overshoot of 8.8 %, against 3.4 % for
/// `Butterworth::<2>`, 0.0007 % for `Butterworth::<1>` and none beyond
/// rounding for the [`Gaussian`].
///
/// # Example
///
/// ```
/// use fovea::frequency::{Auto, Ideal, Radius, dft};
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::MonoF32;
///
/// // A step from 0 to 1, low-passed by the ideal profile.
/// let step = Image::generate(256, 4, |x, _| MonoF32::new(if (64..192).contains(&x) { 1.0 } else { 0.0 }));
/// let mut spectrum = dft(&step, Auto);
/// spectrum.low_pass(Radius::cycles_per_pixel(0.1)?, Ideal);
/// let smooth = spectrum.inverse(Auto);
/// let peak = (0..256).map(|x| smooth.pixel_at(x, 0).0).fold(f32::MIN, f32::max);
/// assert!(peak > 1.08, "the ideal filter rings: {peak}");
/// # Ok::<(), fovea::Error>(())
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Ideal;

/// The Butterworth profile of order `N`: 1 / (1 + (D/D₀)²ᴺ), half the
/// amplitude at the cutoff.
///
/// The order sets the slope: `Butterworth::<1>` is gentle, a high order
/// approaches the [`Ideal`] edge and its ringing. The form is the squared
/// one scikit-image uses. An order of zero does not compile.
///
/// # Example
///
/// ```
/// use fovea::frequency::{Auto, Butterworth, Radius, dft};
/// use fovea::image::Image;
/// use fovea::pixel::MonoF32;
///
/// let mut spectrum = dft(&Image::fill(64, 64, MonoF32::new(1.0)), Auto);
/// spectrum.low_pass(Radius::cycles_per_pixel(0.1)?, Butterworth::<2>);
/// # Ok::<(), fovea::Error>(())
/// ```
///
/// ```compile_fail
/// use fovea::frequency::{Auto, Butterworth, Radius, dft};
/// use fovea::image::Image;
/// use fovea::pixel::MonoF32;
///
/// let mut spectrum = dft(&Image::fill(8, 8, MonoF32::new(1.0)), Auto);
/// // ERROR: a Butterworth filter has an order of at least 1
/// spectrum.low_pass(Radius::cycles_per_pixel(0.1).unwrap(), Butterworth::<0>);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Butterworth<const N: u32>;

/// The Gaussian profile, 2^−(D/D₀)², half the amplitude at the cutoff.
///
/// It does not ring: a Gaussian is a Gaussian in the image too, positive
/// everywhere, so the step response rises monotonically. It is the
/// textbook form exp(−D²/2D₀′²) with D₀′ = D₀ / √(2 ln 2) ≈ 0.849·D₀, and
/// the same filter as a spatial Gaussian blur of σ = 1 / (2π·D₀′) pixels,
/// for D₀ in cycles per pixel.
///
/// # Example
///
/// ```
/// use fovea::frequency::{Auto, Gaussian, Radius, dft};
/// use fovea::image::Image;
/// use fovea::pixel::MonoF32;
///
/// let mut spectrum = dft(&Image::fill(64, 64, MonoF32::new(1.0)), Auto);
/// spectrum.low_pass(Radius::cycles_per_pixel(0.1)?, Gaussian);
/// # Ok::<(), fovea::Error>(())
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Gaussian;

impl sealed::Shape for Ideal {
    fn low_pass(&self, ratio: f64) -> f64 {
        if ratio <= 1.0 { 1.0 } else { 0.0 }
    }
}
impl Profile for Ideal {}

impl<const N: u32> sealed::Shape for Butterworth<N> {
    fn low_pass(&self, ratio: f64) -> f64 {
        const { assert!(N >= 1, "a Butterworth filter has an order of at least 1") };
        1.0 / (1.0 + ratio.powi(2 * N as i32))
    }
}
impl<const N: u32> Profile for Butterworth<N> {}

impl sealed::Shape for Gaussian {
    fn low_pass(&self, ratio: f64) -> f64 {
        (-(ratio * ratio)).exp2()
    }
}
impl Profile for Gaussian {}

/// The low-pass gain of `profile` at `distance` from the centre, for a
/// cutoff `cutoff`; a cutoff of zero passes nothing.
fn low<S: Profile>(profile: S, distance: f64, cutoff: f64) -> f64 {
    if cutoff == 0.0 {
        0.0
    } else {
        profile.low_pass(distance / cutoff)
    }
}

// ── Transfer functions ──────────────────────────────────────────────────────

/// A filter as a value: the real gain it applies at each frequency, the
/// parameter of [`Spectrum::filter`].
///
/// Implemented by [`LowPass`], [`HighPass`], [`BandPass`], [`BandStop`]
/// and [`Notch`], and open: a filter of one's own implements it and is
/// applied the same way. A transfer function for a real image is even,
/// `gain(f) = gain(−f)`; [`Spectrum::filter`] evaluates it on the half of
/// the plane [`Spectrum::apply`] visits, and the other half follows.
///
/// # Example
///
/// ```
/// use fovea::frequency::{Auto, CyclesPerPixel, Frequency, TransferFunction, dft};
/// use fovea::image::Image;
/// use fovea::pixel::MonoF32;
///
/// /// Halves every frequency above a quarter cycle per pixel.
/// struct Damp;
/// impl TransferFunction for Damp {
///     type Unit = CyclesPerPixel;
///     fn gain(&self, f: Frequency<CyclesPerPixel>) -> f64 {
///         if f.radius() > 0.25 { 0.5 } else { 1.0 }
///     }
/// }
///
/// let mut spectrum = dft(&Image::fill(32, 32, MonoF32::new(1.0)), Auto);
/// spectrum.filter(&Damp);
/// ```
pub trait TransferFunction {
    /// The unit the gain is a function of.
    type Unit: FrequencyUnit;

    /// The real gain at the frequency `f`.
    fn gain(&self, f: Frequency<Self::Unit>) -> f64;
}

/// A low-pass: frequencies within the cutoff pass, the profile decides
/// how the rest falls off.
///
/// # Example
///
/// ```
/// use fovea::frequency::{Gaussian, LowPass, Radius, TransferFunction, Frequency};
///
/// let lp = LowPass::new(Radius::cycles_per_pixel(0.1)?, Gaussian);
/// assert_eq!(lp.gain(Frequency::cycles_per_pixel(0.0, 0.0)), 1.0);
/// assert_eq!(lp.gain(Frequency::cycles_per_pixel(0.1, 0.0)), 0.5);
/// # Ok::<(), fovea::Error>(())
/// ```
#[derive(Debug)]
pub struct LowPass<U, S> {
    cutoff: Radius<U>,
    profile: S,
}

/// A high-pass: one minus the [`LowPass`] of the same cutoff and profile.
///
/// # Example
///
/// ```
/// use fovea::frequency::{Butterworth, Frequency, HighPass, Radius, TransferFunction};
///
/// let hp = HighPass::new(Radius::cycles_per_pixel(0.1)?, Butterworth::<2>);
/// assert_eq!(hp.gain(Frequency::cycles_per_pixel(0.0, 0.0)), 0.0);
/// assert_eq!(hp.gain(Frequency::cycles_per_pixel(0.0, 0.1)), 0.5);
/// # Ok::<(), fovea::Error>(())
/// ```
#[derive(Debug)]
pub struct HighPass<U, S> {
    cutoff: Radius<U>,
    profile: S,
}

/// A band-pass: the [`LowPass`] at the band's upper edge minus the one at
/// its lower edge.
///
/// As differences of low-passes, bands partition the spectrum exactly: a
/// low-pass at `a`, the band-passes from `a` to `b` and from `b` to `c`
/// and a high-pass at `c` add up to one at every frequency, so the bands
/// of an image add back up to the image. For the [`Gaussian`] this is the
/// difference of Gaussians. A narrow band passes little under a smooth
/// profile; the [`Ideal`] profile or a high [`Butterworth`] order keeps
/// it.
///
/// # Example
///
/// ```
/// use fovea::frequency::{Band, BandPass, Frequency, Ideal, TransferFunction};
///
/// let bp = BandPass::new(Band::cycles_per_pixel(0.1, 0.2)?, Ideal);
/// assert_eq!(bp.gain(Frequency::cycles_per_pixel(0.15, 0.0)), 1.0);
/// assert_eq!(bp.gain(Frequency::cycles_per_pixel(0.05, 0.0)), 0.0);
/// # Ok::<(), fovea::Error>(())
/// ```
#[derive(Debug)]
pub struct BandPass<U, S> {
    band: Band<U>,
    profile: S,
}

/// A band-stop: one minus the [`BandPass`] of the same band and profile.
///
/// # Example
///
/// ```
/// use fovea::frequency::{Band, BandStop, Frequency, Ideal, TransferFunction};
///
/// let bs = BandStop::new(Band::cycles_per_pixel(0.1, 0.2)?, Ideal);
/// assert_eq!(bs.gain(Frequency::cycles_per_pixel(0.15, 0.0)), 0.0);
/// # Ok::<(), fovea::Error>(())
/// ```
#[derive(Debug)]
pub struct BandStop<U, S> {
    band: Band<U>,
    profile: S,
}

/// A notch: removes the frequencies within a radius of a centre and of
/// its mirror, as a periodic disturbance shows up in the spectrum of a
/// real image.
///
/// The gain is one minus the low-pass around the centre, times the same
/// around the mirror. A radius in [`Bins`] suits a notch, since the width
/// of a peak is set by leakage, which is the same number of bins on every
/// axis. Distances are taken in the canonical range, so a notch within its
/// radius of the Nyquist frequency does not reach across it.
///
/// # Example
///
/// ```
/// use fovea::frequency::{Frequency, Ideal, Notch, Radius, TransferFunction};
///
/// let notch = Notch::new(Frequency::bins(30.0, 12.0), Radius::bins(2.0)?, Ideal);
/// assert_eq!(notch.gain(Frequency::bins(31.0, 12.0)), 0.0);
/// assert_eq!(notch.gain(Frequency::bins(-30.0, -12.0)), 0.0);
/// assert_eq!(notch.gain(Frequency::bins(0.0, 0.0)), 1.0);
/// # Ok::<(), fovea::Error>(())
/// ```
#[derive(Debug)]
pub struct Notch<U, S> {
    center: Frequency<U>,
    radius: Radius<U>,
    profile: S,
}

impl<U: FrequencyUnit, S: Profile> LowPass<U, S> {
    /// The low-pass with `cutoff` and `profile`.
    #[must_use]
    pub const fn new(cutoff: Radius<U>, profile: S) -> Self {
        Self { cutoff, profile }
    }
}

impl<U: FrequencyUnit, S: Profile> HighPass<U, S> {
    /// The high-pass with `cutoff` and `profile`.
    #[must_use]
    pub const fn new(cutoff: Radius<U>, profile: S) -> Self {
        Self { cutoff, profile }
    }
}

impl<U: FrequencyUnit, S: Profile> BandPass<U, S> {
    /// The band-pass over `band` with `profile`.
    #[must_use]
    pub const fn new(band: Band<U>, profile: S) -> Self {
        Self { band, profile }
    }
}

impl<U: FrequencyUnit, S: Profile> BandStop<U, S> {
    /// The band-stop over `band` with `profile`.
    #[must_use]
    pub const fn new(band: Band<U>, profile: S) -> Self {
        Self { band, profile }
    }
}

impl<U: FrequencyUnit, S: Profile> Notch<U, S> {
    /// The notch at `center` and its mirror, with `radius` and `profile`.
    #[must_use]
    pub const fn new(center: Frequency<U>, radius: Radius<U>, profile: S) -> Self {
        Self {
            center,
            radius,
            profile,
        }
    }
}

impl<U: FrequencyUnit, S: Profile> TransferFunction for LowPass<U, S> {
    type Unit = U;
    fn gain(&self, f: Frequency<U>) -> f64 {
        low(self.profile, f.radius(), self.cutoff.get())
    }
}

impl<U: FrequencyUnit, S: Profile> TransferFunction for HighPass<U, S> {
    type Unit = U;
    fn gain(&self, f: Frequency<U>) -> f64 {
        1.0 - low(self.profile, f.radius(), self.cutoff.get())
    }
}

impl<U: FrequencyUnit, S: Profile> TransferFunction for BandPass<U, S> {
    type Unit = U;
    fn gain(&self, f: Frequency<U>) -> f64 {
        let d = f.radius();
        low(self.profile, d, self.band.high()) - low(self.profile, d, self.band.low())
    }
}

impl<U: FrequencyUnit, S: Profile> TransferFunction for BandStop<U, S> {
    type Unit = U;
    fn gain(&self, f: Frequency<U>) -> f64 {
        let d = f.radius();
        1.0 - (low(self.profile, d, self.band.high()) - low(self.profile, d, self.band.low()))
    }
}

impl<U: FrequencyUnit, S: Profile> TransferFunction for Notch<U, S> {
    type Unit = U;
    fn gain(&self, f: Frequency<U>) -> f64 {
        let (cx, cy) = (self.center.fx(), self.center.fy());
        let r = self.radius.get();
        let near = (f.fx() - cx).hypot(f.fy() - cy);
        let mirror = (f.fx() + cx).hypot(f.fy() + cy);
        (1.0 - self.profile.low_pass(near / r)) * (1.0 - self.profile.low_pass(mirror / r))
    }
}

macro_rules! impl_filter_traits {
    ($($ty:ident { $($field:ident),* }),*) => {$(
        impl<U, S: Copy> Clone for $ty<U, S> {
            fn clone(&self) -> Self {
                *self
            }
        }
        impl<U, S: Copy> Copy for $ty<U, S> {}
        impl<U, S: PartialEq> PartialEq for $ty<U, S> {
            fn eq(&self, other: &Self) -> bool {
                true $(&& self.$field == other.$field)*
            }
        }
    )*};
}

impl_filter_traits!(
    LowPass { cutoff, profile },
    HighPass { cutoff, profile },
    BandPass { band, profile },
    BandStop { band, profile },
    Notch {
        center,
        radius,
        profile
    }
);

// ── Directions ──────────────────────────────────────────────────────────────

impl Frequency<CyclesPerPixel> {
    /// The axis this frequency points along, or `None` for the zero
    /// frequency, which has none.
    ///
    /// An axis, not a direction: in the spectrum of a real image a
    /// frequency and its mirror are one component, so only the line they
    /// lie on is defined, modulo 180°. That is also what a filter can rely
    /// on, since [`Spectrum::apply`] and [`Spectrum::filter`] visit one of
    /// the two, the one with a non-negative `fx`. Angles are measured as
    /// [`AxialOrientation`] measures them, from +x towards +y in the image.
    ///
    /// The structures that make a frequency lie at right angles to it:
    /// vertical stripes have their frequencies on the horizontal axis.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::frequency::Frequency;
    ///
    /// // 120° and −60° are one axis.
    /// let a = Frequency::cycles_per_pixel(-0.05, 0.0866).axis().unwrap();
    /// let b = Frequency::cycles_per_pixel(0.05, -0.0866).axis().unwrap();
    /// assert!(a.signed_difference(b).abs() < 1e-12);
    /// assert!(Frequency::cycles_per_pixel(0.0, 0.0).axis().is_none());
    /// ```
    #[must_use]
    pub fn axis(self) -> Option<AxialOrientation> {
        if self.fx() == 0.0 && self.fy() == 0.0 {
            return None;
        }
        AxialOrientation::from_radians(self.fy().atan2(self.fx())).ok()
    }
}

// ── On the spectrum ─────────────────────────────────────────────────────────

impl<P: SpectrumSource> Spectrum<P> {
    /// Multiplies every bin by the gain of `transfer` at its frequency.
    ///
    /// The transfer function is a value: one of the filters of this
    /// module, or one of the caller's own. The named methods
    /// [`low_pass`](Self::low_pass), [`high_pass`](Self::high_pass),
    /// [`band_pass`](Self::band_pass), [`band_stop`](Self::band_stop) and
    /// [`notch`](Self::notch) build one and call this. To write any other
    /// value into a bin, a complex one included, use
    /// [`apply`](Self::apply).
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::frequency::{Auto, Gaussian, LowPass, Radius, dft};
    /// use fovea::image::Image;
    /// use fovea::pixel::MonoF32;
    ///
    /// // Built once, applied to every frame.
    /// let smooth = LowPass::new(Radius::cycles_per_pixel(0.1)?, Gaussian);
    /// for level in 0..3 {
    ///     let frame = Image::fill(32, 32, MonoF32::new(level as f32));
    ///     let mut spectrum = dft(&frame, Auto);
    ///     spectrum.filter(&smooth);
    /// }
    /// # Ok::<(), fovea::Error>(())
    /// ```
    pub fn filter<T: TransferFunction>(&mut self, transfer: &T) {
        self.apply(|f: Frequency<T::Unit>, bin: &mut P::Bin| {
            *bin = bin.scale(transfer.gain(f));
        });
    }

    /// Keeps the frequencies within `cutoff`, as [`LowPass`] defines it.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::frequency::{Auto, Butterworth, Radius, dft};
    /// use fovea::image::Image;
    /// use fovea::pixel::MonoF32;
    ///
    /// let mut spectrum = dft(&Image::fill(64, 64, MonoF32::new(1.0)), Auto);
    /// spectrum.low_pass(Radius::cycles_per_pixel(0.1)?, Butterworth::<2>);
    /// # Ok::<(), fovea::Error>(())
    /// ```
    pub fn low_pass<U: FrequencyUnit, S: Profile>(&mut self, cutoff: Radius<U>, profile: S) {
        self.filter(&LowPass::new(cutoff, profile));
    }

    /// Removes the frequencies within `cutoff`, as [`HighPass`] defines it.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::frequency::{Auto, Gaussian, Radius, dft};
    /// use fovea::image::Image;
    /// use fovea::pixel::MonoF32;
    ///
    /// let mut spectrum = dft(&Image::fill(64, 64, MonoF32::new(1.0)), Auto);
    /// spectrum.high_pass(Radius::cycles_per_pixel(0.02)?, Gaussian);
    /// # Ok::<(), fovea::Error>(())
    /// ```
    pub fn high_pass<U: FrequencyUnit, S: Profile>(&mut self, cutoff: Radius<U>, profile: S) {
        self.filter(&HighPass::new(cutoff, profile));
    }

    /// Keeps the frequencies within `band`, as [`BandPass`] defines it.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::frequency::{Auto, Band, Gaussian, dft};
    /// use fovea::image::Image;
    /// use fovea::pixel::MonoF32;
    ///
    /// let mut spectrum = dft(&Image::fill(64, 64, MonoF32::new(1.0)), Auto);
    /// spectrum.band_pass(Band::cycles_per_pixel(0.05, 0.2)?, Gaussian);
    /// # Ok::<(), fovea::Error>(())
    /// ```
    pub fn band_pass<U: FrequencyUnit, S: Profile>(&mut self, band: Band<U>, profile: S) {
        self.filter(&BandPass::new(band, profile));
    }

    /// Removes the frequencies within `band`, as [`BandStop`] defines it.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::frequency::{Auto, Band, Ideal, dft};
    /// use fovea::image::Image;
    /// use fovea::pixel::MonoF32;
    ///
    /// let mut spectrum = dft(&Image::fill(64, 64, MonoF32::new(1.0)), Auto);
    /// spectrum.band_stop(Band::cycles_per_pixel(0.2, 0.25)?, Ideal);
    /// # Ok::<(), fovea::Error>(())
    /// ```
    pub fn band_stop<U: FrequencyUnit, S: Profile>(&mut self, band: Band<U>, profile: S) {
        self.filter(&BandStop::new(band, profile));
    }

    /// Removes the frequencies within `radius` of `center` and of its
    /// mirror, as [`Notch`] defines it.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::frequency::{Auto, Frequency, Gaussian, Radius, dft};
    /// use fovea::image::Image;
    /// use fovea::pixel::MonoF32;
    ///
    /// let mut spectrum = dft(&Image::fill(64, 64, MonoF32::new(1.0)), Auto);
    /// spectrum.notch(Frequency::bins(8.0, 3.0), Radius::bins(1.5)?, Gaussian);
    /// # Ok::<(), fovea::Error>(())
    /// ```
    pub fn notch<U: FrequencyUnit, S: Profile>(
        &mut self,
        center: Frequency<U>,
        radius: Radius<U>,
        profile: S,
    ) {
        self.filter(&Notch::new(center, radius, profile));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frequency::testing::Rng;
    use crate::frequency::{Auto, FrequencyIndex, Radix2, dft};
    use crate::image::{Image, ImageView};
    use crate::pixel::{ComplexF64, MonoF32, MonoF64};
    use crate::{Size, error::Requirement as R};

    fn cpp(fx: f64, fy: f64) -> Frequency<CyclesPerPixel> {
        Frequency::cycles_per_pixel(fx, fy)
    }

    fn requirement(e: Error) -> (R, Value) {
        match e {
            Error::InvalidParameter(p) => (p.requirement(), p.value()),
            other => panic!("not a parameter error: {other:?}"),
        }
    }

    #[test]
    fn shapes_and_filters_copy_compare_and_print() {
        let r = Radius::<Bins>::try_new(2.0).unwrap();
        let b = Band::<Bins>::try_new(1.0, 3.0).unwrap();
        assert_eq!(format!("{r:?}"), "Radius(2.0)");
        assert_eq!(format!("{b:?}"), "Band { low: 1.0, high: 3.0 }");
        assert_eq!(r.clone(), r);
        assert_eq!(b.clone(), b);
        let lp = LowPass::new(r, Gaussian);
        assert_eq!(lp.clone(), lp);
        assert_ne!(lp, LowPass::new(Radius::try_new(3.0).unwrap(), Gaussian));
        let hp = HighPass::new(r, Ideal);
        assert_eq!(hp.clone(), hp);
        let bp = BandPass::new(b, Butterworth::<1>);
        assert_eq!(bp.clone(), bp);
        let bs = BandStop::new(b, Gaussian);
        assert_eq!(bs.clone(), bs);
        let n = Notch::new(Frequency::bins(4.0, 1.0), r, Ideal);
        assert_eq!(n.clone(), n);
        assert_ne!(n, Notch::new(Frequency::bins(4.0, 2.0), r, Ideal));
        assert!(format!("{n:?}").starts_with("Notch"));
    }

    #[test]
    fn a_radius_is_finite_and_positive() {
        assert_eq!(Radius::<Bins>::try_new(2.0).unwrap().get(), 2.0);
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let e = Radius::<Bins>::try_new(bad).unwrap_err();
            assert_eq!(requirement(e).0, R::FinitePositive, "{bad}");
            assert!(Radius::<CyclesPerPixel>::new(bad).is_none());
        }
        assert_eq!(
            Radius::bins(1.5).unwrap(),
            Radius::<Bins>::new(1.5).unwrap()
        );
        assert_eq!(Radius::cycles_per_pixel(0.1).unwrap().get(), 0.1);
    }

    #[test]
    fn a_band_has_ordered_finite_edges_from_zero() {
        let band = Band::<CyclesPerPixel>::try_new(0.0, 0.2).unwrap();
        assert_eq!((band.low(), band.high()), (0.0, 0.2));
        assert_eq!(
            requirement(Band::<Bins>::try_new(-1.0, 2.0).unwrap_err()),
            (R::FiniteNonNegative, Value::F64(-1.0))
        );
        assert_eq!(
            requirement(Band::<Bins>::try_new(1.0, f64::INFINITY).unwrap_err()).0,
            R::Finite
        );
        assert_eq!(
            requirement(Band::<Bins>::try_new(3.0, 3.0).unwrap_err()),
            (R::StrictlyOrdered, Value::F64Pair(3.0, 3.0))
        );
        assert!(Band::<Bins>::new(f64::NAN, 1.0).is_none());
        assert_eq!(
            Band::bins(1.0, 2.0).unwrap(),
            Band::<Bins>::new(1.0, 2.0).unwrap()
        );
    }

    #[test]
    fn every_smooth_profile_halves_the_amplitude_at_the_cutoff() {
        use super::sealed::Shape;
        assert_eq!(Butterworth::<1>.low_pass(1.0), 0.5);
        assert_eq!(Butterworth::<4>.low_pass(1.0), 0.5);
        assert_eq!(Gaussian.low_pass(1.0), 0.5);
        for ratio in [0.0, 1.0] {
            assert_eq!(Ideal.low_pass(ratio), 1.0);
        }
        assert_eq!(Ideal.low_pass(1.0 + 1e-12), 0.0);
        for profile_gain in [
            Butterworth::<2>.low_pass(0.0),
            Gaussian.low_pass(0.0),
            Ideal.low_pass(0.0),
        ] {
            assert_eq!(profile_gain, 1.0);
        }
        for profile_gain in [
            Butterworth::<2>.low_pass(f64::INFINITY),
            Gaussian.low_pass(f64::INFINITY),
            Ideal.low_pass(f64::INFINITY),
        ] {
            assert_eq!(profile_gain, 0.0);
        }
    }

    #[test]
    fn the_gaussian_is_the_textbook_one_at_a_scaled_cutoff() {
        use super::sealed::Shape;
        let d0 = 0.1_f64;
        let textbook = d0 / (2.0 * 2f64.ln()).sqrt();
        for i in 0..50 {
            let d = i as f64 * 0.01;
            let ours = Gaussian.low_pass(d / d0);
            let theirs = (-(d * d) / (2.0 * textbook * textbook)).exp();
            assert!((ours - theirs).abs() < 1e-15, "{d}: {ours} vs {theirs}");
        }
    }

    fn even<T: TransferFunction<Unit = CyclesPerPixel>>(t: &T) {
        let mut rng = Rng::new(170);
        for _ in 0..200 {
            let (fx, fy) = (rng.value() * 0.5, rng.value() * 0.5);
            assert_eq!(t.gain(cpp(fx, fy)), t.gain(cpp(-fx, -fy)), "({fx}, {fy})");
        }
    }

    #[test]
    fn every_filter_is_even() {
        let r = Radius::cycles_per_pixel(0.1).unwrap();
        let b = Band::cycles_per_pixel(0.05, 0.2).unwrap();
        even(&LowPass::new(r, Gaussian));
        even(&HighPass::new(r, Butterworth::<3>));
        even(&BandPass::new(b, Ideal));
        even(&BandStop::new(b, Gaussian));
        even(&Notch::new(
            cpp(0.2, -0.1),
            Radius::cycles_per_pixel(0.03).unwrap(),
            Gaussian,
        ));
    }

    fn partition<S: Profile>(profile: S) {
        let (a, b, c) = (0.05, 0.12, 0.3);
        let r = |v| Radius::<CyclesPerPixel>::try_new(v).unwrap();
        let band = |lo, hi| Band::<CyclesPerPixel>::try_new(lo, hi).unwrap();
        for i in 0..=70 {
            let f = cpp(i as f64 * 0.01, 0.0);
            let total = LowPass::new(r(a), profile).gain(f)
                + BandPass::new(band(a, b), profile).gain(f)
                + BandPass::new(band(b, c), profile).gain(f)
                + HighPass::new(r(c), profile).gain(f);
            assert!((total - 1.0).abs() < 1e-15, "{}: {total}", f.fx());
            let bp = BandPass::new(band(a, b), profile).gain(f);
            let bs = BandStop::new(band(a, b), profile).gain(f);
            assert!((bp + bs - 1.0).abs() < 1e-15);
        }
    }

    #[test]
    fn bands_partition_the_spectrum_exactly() {
        partition(Ideal);
        partition(Butterworth::<2>);
        partition(Gaussian);
    }

    #[test]
    fn a_band_from_zero_is_a_low_pass() {
        let lp = LowPass::new(Radius::cycles_per_pixel(0.2).unwrap(), Gaussian);
        let bp = BandPass::new(Band::cycles_per_pixel(0.0, 0.2).unwrap(), Gaussian);
        for i in 0..=50 {
            let f = cpp(i as f64 * 0.01, 0.0);
            assert_eq!(bp.gain(f), lp.gain(f));
        }
    }

    #[test]
    fn a_notch_removes_its_centre_and_the_mirror() {
        let notch = Notch::new(
            Frequency::bins(30.0, 12.0),
            Radius::bins(2.0).unwrap(),
            Ideal,
        );
        for (fx, fy) in [(30.0, 12.0), (31.0, 13.0), (-30.0, -12.0), (-28.5, -12.0)] {
            assert_eq!(notch.gain(Frequency::bins(fx, fy)), 0.0, "({fx}, {fy})");
        }
        for (fx, fy) in [(0.0, 0.0), (30.0, 15.0), (-30.0, 12.0)] {
            assert_eq!(notch.gain(Frequency::bins(fx, fy)), 1.0, "({fx}, {fy})");
        }
    }

    #[test]
    fn an_axis_is_defined_modulo_half_a_turn() {
        let deg = |d: f64| d.to_radians();
        let a = cpp(deg(120.0).cos(), deg(120.0).sin()).axis().unwrap();
        let b = cpp(deg(-60.0).cos(), deg(-60.0).sin()).axis().unwrap();
        assert!(a.signed_difference(b).abs() < 1e-12);
        assert!(cpp(0.0, 0.0).axis().is_none());
        // Vertical stripes: their frequencies lie on the horizontal axis.
        let img = Image::generate(32, 32, |x, _| MonoF64::new((x % 4) as f64));
        let spectrum = dft(&img, Auto);
        let f: Frequency<CyclesPerPixel> =
            spectrum.frequency_of(FrequencyIndex::new(8, 0)).unwrap();
        assert!(spectrum.at(FrequencyIndex::new(8, 0)).unwrap().magnitude() > 100.0);
        assert!(
            f.axis()
                .unwrap()
                .signed_difference(AxialOrientation::from_radians(0.0).unwrap())
                .abs()
                < 1e-15
        );
    }

    #[test]
    fn the_named_filters_are_their_values() {
        let mut rng = Rng::new(180);
        let img = Image::from_vec(
            24,
            20,
            (0..480).map(|_| MonoF64::new(rng.value())).collect(),
        )
        .unwrap();
        let r = Radius::cycles_per_pixel(0.15).unwrap();
        let b = Band::cycles_per_pixel(0.05, 0.25).unwrap();
        let c = Frequency::cycles_per_pixel(0.25, 0.1);
        let base = dft(&img, Auto);
        let check = |named: &dyn Fn(&mut Spectrum<MonoF64>),
                     value: &dyn Fn(&mut Spectrum<MonoF64>)| {
            let (mut a, mut b) = (base.clone(), base.clone());
            named(&mut a);
            value(&mut b);
            assert_eq!(a, b);
        };
        check(&|s| s.low_pass(r, Gaussian), &|s| {
            s.filter(&LowPass::new(r, Gaussian))
        });
        check(&|s| s.high_pass(r, Ideal), &|s| {
            s.filter(&HighPass::new(r, Ideal))
        });
        check(&|s| s.band_pass(b, Butterworth::<2>), &|s| {
            s.filter(&BandPass::new(b, Butterworth::<2>))
        });
        check(&|s| s.band_stop(b, Gaussian), &|s| {
            s.filter(&BandStop::new(b, Gaussian))
        });
        check(
            &|s| s.notch(c, Radius::cycles_per_pixel(0.05).unwrap(), Ideal),
            &|s| {
                s.filter(&Notch::new(
                    c,
                    Radius::cycles_per_pixel(0.05).unwrap(),
                    Ideal,
                ))
            },
        );
    }

    #[test]
    fn filter_scales_every_bin_by_the_gain() {
        let mut rng = Rng::new(190);
        let img = Image::from_vec(
            16,
            12,
            (0..192).map(|_| MonoF64::new(rng.value())).collect(),
        )
        .unwrap();
        let lp = LowPass::new(Radius::cycles_per_pixel(0.2).unwrap(), Butterworth::<2>);
        let mut filtered = dft(&img, Auto);
        filtered.filter(&lp);
        let original = dft(&img, Auto);
        for ky in -6..6 {
            for kx in -8..8 {
                let i = FrequencyIndex::new(kx, ky);
                let g = lp.gain(original.frequency_of(i).unwrap());
                let want = original.at(i).unwrap();
                let got = filtered.at(i).unwrap();
                assert!(
                    (got.re - want.re * g).abs() < 1e-15 && (got.im - want.im * g).abs() < 1e-15
                );
            }
        }
    }

    /// The largest value of a step from 0 to 1, low-passed at 0.1 cycles per pixel.
    fn step_peak<S: Profile>(profile: S) -> (f32, f32) {
        let step = Image::generate(256, 4, |x, _| {
            MonoF32::new(if (64..192).contains(&x) { 1.0 } else { 0.0 })
        });
        let mut spectrum = dft(&step, Radix2).unwrap();
        spectrum.low_pass(Radius::cycles_per_pixel(0.1).unwrap(), profile);
        let smooth = spectrum.inverse(Radix2).unwrap();
        let row: Vec<f32> = (0..256).map(|x| smooth.pixel_at(x, 0).0).collect();
        (
            row.iter().copied().fold(f32::MIN, f32::max),
            row.iter().copied().fold(f32::MAX, f32::min),
        )
    }

    #[test]
    fn the_ideal_profile_rings_and_the_gaussian_does_not() {
        let (ideal_max, ideal_min) = step_peak(Ideal);
        let (gauss_max, gauss_min) = step_peak(Gaussian);
        let (bw_max, _) = step_peak(Butterworth::<2>);
        assert!(
            (1.085..=1.095).contains(&ideal_max) && ideal_min < -0.08,
            "ideal: {ideal_max} {ideal_min}"
        );
        assert!(
            gauss_max <= 1.0 + 1e-6 && gauss_min >= -1e-6,
            "gaussian: {gauss_max} {gauss_min}"
        );
        assert!(bw_max > 1.0 && bw_max < ideal_max, "butterworth: {bw_max}");
    }

    #[test]
    fn products_need_spectra_of_one_size() {
        let a = dft(&Image::fill(8, 4, MonoF32::new(1.0)), Auto);
        let mut b = dft(&Image::fill(4, 8, MonoF32::new(1.0)), Auto);
        let mismatch = Err(Error::SizeMismatch {
            expected: Size::new(4, 8),
            actual: Size::new(8, 4),
        });
        assert_eq!(b.multiply(&a), mismatch.clone());
        assert_eq!(b.multiply_conjugate(&a), mismatch);
    }

    #[test]
    fn the_product_of_spectra_is_the_cyclic_convolution() {
        let mut rng = Rng::new(200);
        let (w, h) = (6usize, 5usize);
        let x: Vec<f64> = (0..w * h).map(|_| rng.value()).collect();
        let k: Vec<f64> = (0..w * h).map(|_| rng.value()).collect();
        let img = |v: &[f64]| {
            Image::from_vec(w, h, v.iter().map(|&a| MonoF64::new(a)).collect()).unwrap()
        };
        let mut conv = dft(&img(&x), Auto);
        conv.multiply(&dft(&img(&k), Auto)).unwrap();
        let mut corr = dft(&img(&x), Auto);
        corr.multiply_conjugate(&dft(&img(&k), Auto)).unwrap();
        let (conv, corr) = (conv.inverse(Auto), corr.inverse(Auto));
        for y in 0..h {
            for xx in 0..w {
                let (mut c1, mut c2) = (0.0, 0.0);
                for j in 0..h {
                    for i in 0..w {
                        c1 += k[j * w + i] * x[(y + h - j) % h * w + (xx + w - i) % w];
                        c2 += k[j * w + i] * x[(y + j) % h * w + (xx + i) % w];
                    }
                }
                assert!((conv.pixel_at(xx, y).0 - c1).abs() < 1e-13);
                assert!((corr.pixel_at(xx, y).0 - c2).abs() < 1e-13);
            }
        }
        let _ = ComplexF64::new(0.0, 0.0);
    }
}

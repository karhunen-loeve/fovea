//! Unit markers for lengths in the plane.

use core::marker::PhantomData;

/// A unit of length, carried by [`Point`](super::Point),
/// [`Vector`](super::Vector) and [`Length`](super::Length) as a type
/// parameter.
///
/// Implemented by [`Pixels`] and by every [`Meter<P>`](Meter). The trait is
/// open: a unit of your own works with the geometry types and the mappings,
/// and converts to nothing, because fovea knows no conversion for it. A unit
/// is a marker and is never instantiated.
///
/// # Example
///
/// ```
/// use fovea::geometry::{LengthUnit, Point};
///
/// /// Thousandths of an inch.
/// struct Mil;
/// impl LengthUnit for Mil {}
///
/// let p: Point<Mil> = Point::new(3.0, 4.0);
/// assert_eq!(p.x, 3.0);
/// ```
pub trait LengthUnit {}

/// The pixels of the image a position was measured in.
///
/// Positions follow the pixel-centre convention: `Point::<Pixels>::new(0.0,
/// 0.0)` is the centre of pixel `(0, 0)`, and a step between columns 5 and
/// 6 lies at `x = 5.5`. A position found on a pyramid level is in that
/// level's pixels; map it to the base image through the level's grid
/// spacing and origin offset before combining it with base-image positions.
///
/// # Example
///
/// ```
/// use fovea::{CoordinateF64, Pixels, Point};
///
/// let p: Point<Pixels> = CoordinateF64::new(12.5, 3.0).into();
/// assert_eq!((p.x, p.y), (12.5, 3.0));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Pixels;

impl LengthUnit for Pixels {}

mod sealed {
    pub trait Sealed {}
}

/// A metric prefix: the power of ten a [`Meter<P>`](Meter) is scaled by.
///
/// Sealed, because converting between prefixes relies on the exponents
/// being right. Implemented by [`Unit`] (10⁰), [`Milli`] (10⁻³) and
/// [`Micro`] (10⁻⁶).
///
/// # Example
///
/// ```
/// use fovea::geometry::{Micro, Milli, Prefix, Unit};
///
/// assert_eq!(Unit::EXPONENT, 0);
/// assert_eq!(Milli::EXPONENT, -3);
/// assert_eq!(Micro::EXPONENT, -6);
/// ```
pub trait Prefix: sealed::Sealed {
    /// The power of ten this prefix stands for.
    const EXPONENT: i32;
}

/// No prefix: [`Meter<Unit>`](Meter), which a bare `Meter` also names.
///
/// # Example
///
/// ```
/// use fovea::geometry::{Meter, Unit};
///
/// let _: fn(Meter<Unit>) -> Meter = |m| m;
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Unit;

/// The prefix milli, 10⁻³: [`Meter<Milli>`](Meter) is the millimetre.
///
/// # Example
///
/// ```
/// use fovea::geometry::{Meter, Milli};
/// use fovea::Millimeter;
///
/// let _: fn(Meter<Milli>) -> Millimeter = |m| m;
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Milli;

/// The prefix micro, 10⁻⁶: [`Meter<Micro>`](Meter) is the micrometre.
///
/// # Example
///
/// ```
/// use fovea::geometry::{Meter, Micro};
/// use fovea::Micrometer;
///
/// let _: fn(Meter<Micro>) -> Micrometer = |m| m;
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Micro;

impl sealed::Sealed for Unit {}
impl sealed::Sealed for Milli {}
impl sealed::Sealed for Micro {}

impl Prefix for Unit {
    const EXPONENT: i32 = 0;
}
impl Prefix for Milli {
    const EXPONENT: i32 = -3;
}
impl Prefix for Micro {
    const EXPONENT: i32 = -6;
}

/// The metre with a metric prefix: `Meter<Milli>` is the millimetre, and a
/// bare `Meter` is the metre.
///
/// A value keeps the unit it was calibrated in. Changing the prefix is an
/// explicit `convert`, which is exact wherever the value times the power of
/// ten is representable: 12.5 mm is exactly 12 500 µm and comes back as
/// exactly 12.5 mm.
///
/// # Example
///
/// ```
/// use fovea::{Length, Micrometer, Millimeter};
///
/// let width: Length<Millimeter> = Length::new(12.5);
/// let fine: Length<Micrometer> = width.convert();
/// assert_eq!(fine.get(), 12_500.0);
/// assert_eq!(fine.convert::<fovea::geometry::Milli>().get(), 12.5);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Meter<P: Prefix = Unit>(PhantomData<P>);

impl<P: Prefix> LengthUnit for Meter<P> {}

/// The millimetre, `Meter<Milli>`.
pub type Millimeter = Meter<Milli>;

/// The micrometre, `Meter<Micro>`.
pub type Micrometer = Meter<Micro>;

/// The powers of ten that are exact in `f64`, 10⁰ to 10²².
const EXACT_POWERS_OF_TEN: [f64; 23] = [
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16,
    1e17, 1e18, 1e19, 1e20, 1e21, 1e22,
];

/// Rescales a value from prefix `P` to prefix `Q`.
///
/// Multiplies by an exact power of ten when `Q` is the smaller prefix and
/// divides by one when it is the larger, because 10³ is exact in `f64` and
/// 10⁻³ is not. Division by an exact power of ten is correctly rounded, so
/// a value that is representable in the target prefix comes out exactly.
pub(crate) fn rescale<P: Prefix, Q: Prefix>(value: f64) -> f64 {
    let steps = P::EXPONENT - Q::EXPONENT;
    let power = EXACT_POWERS_OF_TEN[steps.unsigned_abs() as usize];
    if steps >= 0 {
        value * power
    } else {
        value / power
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefixes_carry_their_exponents() {
        assert_eq!(Unit::EXPONENT, 0);
        assert_eq!(Milli::EXPONENT, -3);
        assert_eq!(Micro::EXPONENT, -6);
    }

    #[test]
    fn rescaling_between_prefixes_is_exact_for_representable_values() {
        assert_eq!(rescale::<Milli, Micro>(12.5), 12_500.0);
        assert_eq!(rescale::<Micro, Milli>(12_500.0), 12.5);
        assert_eq!(rescale::<Unit, Milli>(0.25), 250.0);
        assert_eq!(rescale::<Milli, Unit>(250.0), 0.25);
        assert_eq!(rescale::<Micro, Unit>(1.0), 1e-6);
        assert_eq!(rescale::<Milli, Milli>(0.1), 0.1);
    }

    #[test]
    fn a_decimal_value_lands_on_the_representable_target() {
        // 0.1 mm is not exact in f64, but 100 µm is, and a correctly
        // rounded product finds it. Going back, division by the exact 1000
        // is correctly rounded as well, and returns the f64 nearest 0.1.
        assert_eq!(rescale::<Milli, Micro>(0.1), 100.0);
        assert_eq!(rescale::<Micro, Milli>(100.0), 0.1);
        // Multiplying by 0.001 instead would miss it for 9 µm, the smallest
        // whole number of micrometres where the two differ.
        assert_eq!(rescale::<Micro, Milli>(9.0), 0.009);
        assert_ne!(9.0 * 0.001, 0.009);
    }

    #[test]
    fn exact_powers_of_ten_are_exact() {
        let mut power = 1.0f64;
        for (i, &p) in EXACT_POWERS_OF_TEN.iter().enumerate() {
            assert_eq!(p, power, "10^{i}");
            power *= 10.0;
        }
    }
}

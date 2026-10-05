//! The frequency vocabulary: the integer address of a bin, and a frequency
//! with its unit.

use crate::Size;
use core::marker::PhantomData;

mod sealed {
    use crate::Size;

    /// The conversion between a unit and cycles per pixel, which needs the
    /// size of the image the spectrum belongs to.
    pub trait Unit {
        /// The bin `(kx, ky)` in this unit, rounded once at most.
        fn from_index(kx: isize, ky: isize, size: Size) -> (f64, f64);
        /// `(fx, fy)` in this unit, in cycles per pixel.
        fn to_cycles(fx: f64, fy: f64, size: Size) -> (f64, f64);
        /// `(cx, cy)` in cycles per pixel, in this unit.
        fn from_cycles(cx: f64, cy: f64, size: Size) -> (f64, f64);
    }
}

/// A unit of frequency, carried by [`Frequency`] as a type parameter.
/// Sealed: a spectrum converts between the units it knows.
///
/// Implemented by [`Bins`], which depends on the size of the image, and
/// [`CyclesPerPixel`], which does not. A unit is a marker and is never
/// instantiated.
pub trait FrequencyUnit: sealed::Unit {}

/// The bin spacing of one spectrum: 1/W cycles per pixel along x and 1/H
/// along y, for a W × H image.
///
/// Bin 3 along x is three periods across the width of the image. The unit
/// suits what is measured in bins on every axis alike, such as the width
/// of a peak, which leakage sets: a narrow peak spreads over a bin or two
/// whatever the image size.
///
/// # Example
///
/// ```
/// use fovea::frequency::{Bins, Frequency, FrequencyIndex};
///
/// let f: Frequency<Bins> = FrequencyIndex::new(3, -2).into();
/// assert_eq!((f.fx(), f.fy()), (3.0, -2.0));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Bins;

/// Cycles per pixel: a physical frequency, the same for every image size.
///
/// 0.125 is one period every 8 pixels. The largest frequency an image
/// holds along an axis is 0.5, the Nyquist frequency. The unit suits
/// what describes structures in the image, such as the band of a filter:
/// on a 512 × 256 image a circle in cycles per pixel is a circle in the
/// image's frequencies, where a circle in bins would be an ellipse.
///
/// # Example
///
/// ```
/// use fovea::frequency::Frequency;
///
/// let f = Frequency::cycles_per_pixel(0.125, 0.0);
/// assert_eq!(f.radius(), 0.125);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CyclesPerPixel;

impl sealed::Unit for Bins {
    fn from_index(kx: isize, ky: isize, _: Size) -> (f64, f64) {
        (kx as f64, ky as f64)
    }
    fn to_cycles(fx: f64, fy: f64, size: Size) -> (f64, f64) {
        (fx / size.width as f64, fy / size.height as f64)
    }
    fn from_cycles(cx: f64, cy: f64, size: Size) -> (f64, f64) {
        (cx * size.width as f64, cy * size.height as f64)
    }
}
impl FrequencyUnit for Bins {}

impl sealed::Unit for CyclesPerPixel {
    fn from_index(kx: isize, ky: isize, size: Size) -> (f64, f64) {
        (
            kx as f64 / size.width as f64,
            ky as f64 / size.height as f64,
        )
    }
    fn to_cycles(fx: f64, fy: f64, _: Size) -> (f64, f64) {
        (fx, fy)
    }
    fn from_cycles(cx: f64, cy: f64, _: Size) -> (f64, f64) {
        (cx, cy)
    }
}
impl FrequencyUnit for CyclesPerPixel {}

/// The address of one bin: its frequency along x and along y, in bins.
///
/// A plain signed pair, exact, with no size inside. Where the spectrum
/// hands an index out, it lies in the canonical range
/// `−⌊N/2⌋ ..= ⌈N/2⌉ − 1` along a side of N, so the Nyquist bin of an even
/// side is negative, as numpy's `fftfreq` has it. An index passed in is
/// taken modulo the size, `X[k + N] = X[k]`, which is the periodicity the
/// transform is defined with.
///
/// # Example
///
/// ```
/// use fovea::frequency::{Auto, FrequencyIndex, dft};
/// use fovea::image::Image;
/// use fovea::pixel::MonoF32;
///
/// let spectrum = dft(&Image::fill(8, 8, MonoF32::new(1.0)), Auto);
/// // Bin (9, 0) is bin (1, 0) of an 8-wide spectrum.
/// assert_eq!(spectrum.at(FrequencyIndex::new(9, 0)), spectrum.at(FrequencyIndex::new(1, 0)));
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct FrequencyIndex {
    /// The frequency along x, in bins.
    pub kx: isize,
    /// The frequency along y, in bins.
    pub ky: isize,
}

impl FrequencyIndex {
    /// The bin `(kx, ky)`.
    #[must_use]
    pub const fn new(kx: isize, ky: isize) -> Self {
        Self { kx, ky }
    }
}

/// A frequency in the plane, in the unit `U`: [`Bins`] or
/// [`CyclesPerPixel`].
///
/// Stored Cartesian, as the bins lie; [`radius`](Self::radius) gives its
/// distance from the zero frequency. A frequency is not a position, and
/// there is no conversion between the two. Units convert only through a
/// [`Spectrum`](super::Spectrum), because only a spectrum knows the size
/// a bin is a fraction of.
///
/// # Example
///
/// ```
/// use fovea::frequency::{Bins, CyclesPerPixel, Frequency};
///
/// let peak: Frequency<Bins> = Frequency::bins(3.0, 4.0);
/// assert_eq!(peak.radius(), 5.0);
///
/// let band_edge: Frequency<CyclesPerPixel> = Frequency::cycles_per_pixel(0.25, 0.0);
/// assert_eq!(band_edge.fx(), 0.25);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frequency<U> {
    fx: f64,
    fy: f64,
    unit: PhantomData<U>,
}

impl Frequency<Bins> {
    /// The frequency `(fx, fy)` in bins.
    #[must_use]
    pub const fn bins(fx: f64, fy: f64) -> Self {
        Self::of(fx, fy)
    }
}

impl Frequency<CyclesPerPixel> {
    /// The frequency `(fx, fy)` in cycles per pixel.
    #[must_use]
    pub const fn cycles_per_pixel(fx: f64, fy: f64) -> Self {
        Self::of(fx, fy)
    }
}

impl<U> Frequency<U> {
    pub(crate) const fn of(fx: f64, fy: f64) -> Self {
        Self {
            fx,
            fy,
            unit: PhantomData,
        }
    }

    /// The frequency along x.
    #[must_use]
    pub const fn fx(self) -> f64 {
        self.fx
    }

    /// The frequency along y.
    #[must_use]
    pub const fn fy(self) -> f64 {
        self.fy
    }

    /// The distance from the zero frequency, `√(fx² + fy²)`, without
    /// overflow in the squares.
    #[must_use]
    pub fn radius(self) -> f64 {
        self.fx.hypot(self.fy)
    }
}

impl From<FrequencyIndex> for Frequency<Bins> {
    /// The bin's frequency in bins, exactly.
    fn from(index: FrequencyIndex) -> Self {
        Frequency::bins(index.kx as f64, index.ky as f64)
    }
}

/// `k` in the canonical range `−⌊n/2⌋ ..= ⌈n/2⌉ − 1` of a side of `n > 0`.
pub(crate) fn canonical(k: isize, n: usize) -> isize {
    let n = n as isize;
    let m = k.rem_euclid(n);
    if m <= (n - 1) / 2 { m } else { m - n }
}

/// The frequency of the canonical bin `(kx, ky)` of a spectrum of `size`,
/// in the unit `U`.
pub(crate) fn frequency_in<U: FrequencyUnit>(kx: isize, ky: isize, size: Size) -> Frequency<U> {
    let (fx, fy) = U::from_index(kx, ky, size);
    Frequency::of(fx, fy)
}

/// `f`, a frequency in `V`, in the unit `U`, for a spectrum of `size`.
pub(crate) fn convert<U: FrequencyUnit, V: FrequencyUnit>(
    f: Frequency<V>,
    size: Size,
) -> Frequency<U> {
    let (cx, cy) = V::to_cycles(f.fx, f.fy, size);
    let (fx, fy) = U::from_cycles(cx, cy, size);
    Frequency::of(fx, fy)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_canonical_range_follows_numpy() {
        // fftfreq(8) = [0, 1, 2, 3, -4, -3, -2, -1]; fftfreq(5) = [0, 1, 2, -2, -1]
        let eight: Vec<isize> = (0..8).map(|k| canonical(k, 8)).collect();
        assert_eq!(eight, [0, 1, 2, 3, -4, -3, -2, -1]);
        let five: Vec<isize> = (0..5).map(|k| canonical(k, 5)).collect();
        assert_eq!(five, [0, 1, 2, -2, -1]);
        assert_eq!(canonical(-9, 8), -1);
        assert_eq!(canonical(13, 5), -2);
        assert_eq!(canonical(0, 1), 0);
    }

    #[test]
    fn an_index_becomes_a_frequency_in_bins_exactly() {
        let f: Frequency<Bins> = FrequencyIndex::new(-7, 12).into();
        assert_eq!((f.fx(), f.fy()), (-7.0, 12.0));
        assert_eq!(Frequency::bins(3.0, -4.0).radius(), 5.0);
    }

    #[test]
    fn a_bin_in_cycles_per_pixel_is_its_fraction_of_the_size() {
        let size = Size::new(640, 480);
        let f: Frequency<CyclesPerPixel> = frequency_in(-320, 60, size);
        assert_eq!((f.fx(), f.fy()), (-0.5, 0.125));
        let b: Frequency<Bins> = convert(f, size);
        assert_eq!((b.fx(), b.fy()), (-320.0, 60.0));
    }
}

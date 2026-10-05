//! The DFT algorithms as values, and the transform's entry point.

use super::engine::{Axis, Bluestein as BluesteinTables, Radix2 as Radix2Tables};
use super::spectrum::{Spectrum, SpectrumSource};
use crate::image::{Image, RasterImage};
use crate::{Error, Size};

/// The tables along x and along y.
type Axes<C> = (Axis<C>, Axis<C>);

mod sealed {
    use super::{DftMethod, Spectrum, SpectrumSource};
    use crate::image::RasterImage;

    pub trait Method<P: SpectrumSource> {
        fn transform<I: RasterImage<Pixel = P>>(&self, src: &I) -> <Self as DftMethod<P>>::Output
        where
            Self: DftMethod<P>;

        fn invert(&self, spectrum: &Spectrum<P>) -> <Self as DftMethod<P>>::InverseOutput
        where
            Self: DftMethod<P>;
    }
}

/// An algorithm that computes the discrete Fourier transform: the second
/// parameter of [`dft`] and the parameter of [`Spectrum::inverse`].
/// Sealed.
///
/// Every method computes the same transform. They differ in the sizes they
/// accept, in speed and in rounding, and the associated types say whether
/// a method can fail: [`Radix2`] returns a `Result`, [`Bluestein`] and
/// [`Auto`] accept every size and return the value itself.
pub trait DftMethod<P: SpectrumSource>: sealed::Method<P> {
    /// What [`dft`] returns: a [`Spectrum<P>`], or a `Result` of one for a
    /// method that does not accept every size.
    type Output;
    /// What [`Spectrum::inverse`] returns: an [`Image<P>`], or a `Result`
    /// of one.
    type InverseOutput;
}

/// The discrete Fourier transform of `src`, computed by `method`.
///
/// The method is always written out; [`Auto`] is the one to write when
/// any will do. The image is not padded: it is transformed at its own
/// size, as one period of a periodic signal. To transform at another size,
/// pad first with [`pad`](crate::transform::pad), which names what fills
/// the new pixels.
///
/// # Example
///
/// ```
/// use fovea::Size;
/// use fovea::border::Constant;
/// use fovea::frequency::{Auto, Radix2, dft};
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::MonoF32;
/// use fovea::transform::pad;
///
/// let img = Image::fill(640, 480, MonoF32::new(0.5));
///
/// // Any size, by the fastest method available for it.
/// let spectrum = dft(&img, Auto);
///
/// // Radix2 wants powers of two: padding is a visible step.
/// assert!(dft(&img, Radix2).is_err());
/// let padded = pad(&img, Radix2.next_size(img.size()), &Constant(MonoF32::new(0.0)))?;
/// let spectrum = dft(&padded, Radix2)?;
/// assert_eq!(spectrum.source_size(), Size::new(1024, 512));
/// # Ok::<(), fovea::Error>(())
/// ```
#[must_use]
pub fn dft<P, I, M>(src: &I, method: M) -> M::Output
where
    P: SpectrumSource,
    I: RasterImage<Pixel = P>,
    M: DftMethod<P>,
{
    method.transform(src)
}

/// The radix-2 Cooley-Tukey algorithm: for images whose sides are powers
/// of two.
///
/// The fastest method, for the sizes it accepts: every side a power of
/// two, or an image with no pixels. Any other size is an
/// [`Error::UnsupportedDftSize`], so a call never slows down or pads
/// behind the caller's back; [`next_size`](Self::next_size) gives the size
/// to pad to.
///
/// # Accuracy
///
/// Every twiddle factor is computed from its exact integer index and
/// rounded once; none comes from a recurrence, which is where an FFT most
/// often loses accuracy. The error relative to the whole result then grows
/// with the logarithm of the size: for 2ᵗ points, Higham's worst-case
/// bound is about 6.4·t·u, where u is 2⁻²⁴ for `f32` and 2⁻⁵³ for `f64`,
/// and random data comes out well below it. A single bin can be much less
/// accurate than the whole: a component that is zero in exact arithmetic
/// comes out as rounding noise, and so does its phase.
///
/// # Example
///
/// ```
/// use fovea::Size;
/// use fovea::frequency::Radix2;
///
/// assert!(Radix2.accepts(Size::new(1024, 512)));
/// assert!(!Radix2.accepts(Size::new(1920, 1080)));
/// assert_eq!(Radix2.next_size(Size::new(1920, 1080)), Size::new(2048, 2048));
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Radix2;

impl Radix2 {
    /// Whether `Radix2` transforms an image of `size`: every side a power
    /// of two, or no pixels at all.
    #[must_use]
    pub fn accepts(self, size: Size) -> bool {
        size.area() == 0 || (size.width.is_power_of_two() && size.height.is_power_of_two())
    }

    /// The smallest size `Radix2` accepts that is at least `size`: every
    /// side rounded up to a power of two. A size without pixels is
    /// accepted as it is.
    ///
    /// # Panics
    ///
    /// Panics if a side exceeds the largest power of two a `usize` holds,
    /// a size no image can have.
    #[must_use]
    pub fn next_size(self, size: Size) -> Size {
        if size.area() == 0 {
            return size;
        }
        let up = |n: usize| {
            n.checked_next_power_of_two().unwrap_or_else(|| {
                panic!("Radix2::next_size: no power of two in usize is at least {n}")
            })
        };
        Size::new(up(size.width), up(size.height))
    }

    fn axes<P: SpectrumSource>(self, size: Size) -> Result<Axes<P::Bin>, Error> {
        if !self.accepts(size) {
            return Err(Error::UnsupportedDftSize {
                method: "Radix2",
                size,
            });
        }
        if size.area() == 0 {
            // No row and no column is transformed.
            return Ok((
                Axis::Radix2(Radix2Tables::new(0)),
                Axis::Radix2(Radix2Tables::new(0)),
            ));
        }
        Ok((
            Axis::Radix2(Radix2Tables::new(size.width)),
            Axis::Radix2(Radix2Tables::new(size.height)),
        ))
    }
}

impl<P: SpectrumSource> DftMethod<P> for Radix2 {
    type Output = Result<Spectrum<P>, Error>;
    type InverseOutput = Result<Image<P>, Error>;
}

impl<P: SpectrumSource> sealed::Method<P> for Radix2 {
    fn transform<I: RasterImage<Pixel = P>>(&self, src: &I) -> <Self as DftMethod<P>>::Output {
        let (along_x, along_y) = self.axes::<P>(src.size())?;
        Ok(Spectrum::forward(src, &along_x, &along_y))
    }

    fn invert(&self, spectrum: &Spectrum<P>) -> <Self as DftMethod<P>>::InverseOutput {
        let (along_x, along_y) = self.axes::<P>(spectrum.source_size())?;
        Ok(spectrum.invert(&along_x, &along_y))
    }
}

/// Bluestein's algorithm: for images of any size.
///
/// A transform of length n becomes a circular convolution with the chirp
/// `exp(−iπ·j²/n)`, which radix-2 transforms of the power of two at least
/// 2n − 1 evaluate. It accepts every size, and its cost does not jump
/// between neighbouring sizes; on a size [`Radix2`] accepts as well it
/// takes three to four times as long (measured from 512 × 512 to
/// 2048 × 2048 in `f32`).
///
/// # Accuracy
///
/// The chirp's index j² mod 2n is formed in integers, so it stays exact
/// for every length; in floating point it would lose its last digits once
/// j² outgrows the mantissa. The method is less accurate than `Radix2`,
/// since it runs three transforms where `Radix2` runs one: on random data
/// of up to 1920 points per side the error came out at about 4·u relative
/// to the whole result, against about 2·u for `Radix2` (u is 2⁻²⁴ for
/// `f32` and 2⁻⁵³ for `f64`).
///
/// # Example
///
/// ```
/// use fovea::frequency::{Bluestein, dft};
/// use fovea::image::Image;
/// use fovea::pixel::MonoF32;
///
/// let img = Image::fill(1920, 1080, MonoF32::new(1.0));
/// let spectrum = dft(&img, Bluestein);
/// assert_eq!(spectrum.source_size(), fovea::Size::new(1920, 1080));
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Bluestein;

impl Bluestein {
    /// Whether `Bluestein` transforms an image of `size`: always.
    #[must_use]
    pub fn accepts(self, size: Size) -> bool {
        let _ = size;
        true
    }

    /// The smallest size `Bluestein` accepts that is at least `size`:
    /// `size` itself.
    #[must_use]
    pub fn next_size(self, size: Size) -> Size {
        size
    }
}

impl<P: SpectrumSource> DftMethod<P> for Bluestein {
    type Output = Spectrum<P>;
    type InverseOutput = Image<P>;
}

impl<P: SpectrumSource> sealed::Method<P> for Bluestein {
    fn transform<I: RasterImage<Pixel = P>>(&self, src: &I) -> <Self as DftMethod<P>>::Output {
        let Size { width, height } = src.size();
        Spectrum::forward(
            src,
            &Axis::Bluestein(BluesteinTables::new(width)),
            &Axis::Bluestein(BluesteinTables::new(height)),
        )
    }

    fn invert(&self, spectrum: &Spectrum<P>) -> <Self as DftMethod<P>>::InverseOutput {
        let Size { width, height } = spectrum.source_size();
        spectrum.invert(
            &Axis::Bluestein(BluesteinTables::new(width)),
            &Axis::Bluestein(BluesteinTables::new(height)),
        )
    }
}

/// The fastest method this release has for each side: [`Radix2`] along a
/// side that is a power of two, [`Bluestein`] along any other.
///
/// It accepts every size. Which algorithm it picks may change in a later
/// release, when a faster one for some sizes arrives; the transform stays
/// the same up to rounding, and the change is recorded in the changelog.
/// To keep the algorithm fixed, write [`Radix2`] or [`Bluestein`].
///
/// # Example
///
/// ```
/// use fovea::frequency::{Auto, dft};
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::MonoF64;
///
/// // 1024 along x by radix-2, 1000 along y by Bluestein.
/// let img = Image::generate(1024, 1000, |x, y| MonoF64::new(((x ^ y) & 7) as f64));
/// let back = dft(&img, Auto).inverse(Auto);
/// assert!((back.pixel_at(9, 3).0 - 2.0).abs() < 1e-12);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Auto;

impl Auto {
    /// Whether `Auto` transforms an image of `size`: always.
    #[must_use]
    pub fn accepts(self, size: Size) -> bool {
        let _ = size;
        true
    }

    /// The smallest size `Auto` accepts that is at least `size`: `size`
    /// itself.
    #[must_use]
    pub fn next_size(self, size: Size) -> Size {
        size
    }

    fn axis<P: SpectrumSource>(len: usize) -> Axis<P::Bin> {
        if len == 0 || len.is_power_of_two() {
            Axis::Radix2(Radix2Tables::new(len))
        } else {
            Axis::Bluestein(BluesteinTables::new(len))
        }
    }
}

impl<P: SpectrumSource> DftMethod<P> for Auto {
    type Output = Spectrum<P>;
    type InverseOutput = Image<P>;
}

impl<P: SpectrumSource> sealed::Method<P> for Auto {
    fn transform<I: RasterImage<Pixel = P>>(&self, src: &I) -> <Self as DftMethod<P>>::Output {
        let Size { width, height } = src.size();
        Spectrum::forward(src, &Self::axis::<P>(width), &Self::axis::<P>(height))
    }

    fn invert(&self, spectrum: &Spectrum<P>) -> <Self as DftMethod<P>>::InverseOutput {
        let Size { width, height } = spectrum.source_size();
        spectrum.invert(&Self::axis::<P>(width), &Self::axis::<P>(height))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Rectangle;
    use crate::frequency::engine::Cplx;
    use crate::frequency::testing::{
        BLUESTEIN_F32, BLUESTEIN_F64, MU32, MU64, Rng, U32, U64, compose, higham, naive_2d,
        oracle_error, reference_root, rel_error,
    };
    use crate::image::{ImageView, SubView};
    use crate::pixel::{ComplexF32, MonoF32, MonoF64};
    use std::f64::consts::SQRT_2;

    /// The rounding of one precision, for the tolerances.
    trait Precision: SpectrumSource {
        const U: f64;
        const MU: f64;
        const BLUESTEIN: f64;
        fn of(v: f64) -> Self;
        fn get(self) -> f64;
    }

    impl Precision for MonoF32 {
        const U: f64 = U32;
        const MU: f64 = MU32;
        const BLUESTEIN: f64 = BLUESTEIN_F32;
        fn of(v: f64) -> Self {
            MonoF32::new(v as f32)
        }
        fn get(self) -> f64 {
            f64::from(self.0)
        }
    }

    impl Precision for MonoF64 {
        const U: f64 = U64;
        const MU: f64 = MU64;
        const BLUESTEIN: f64 = BLUESTEIN_F64;
        fn of(v: f64) -> Self {
            MonoF64::new(v)
        }
        fn get(self) -> f64 {
            self.0
        }
    }

    #[derive(Clone, Copy, Debug)]
    enum Kind {
        Radix2,
        Bluestein,
        Auto,
    }

    fn forward<P: Precision>(kind: Kind, img: &Image<P>) -> Spectrum<P> {
        match kind {
            Kind::Radix2 => dft(img, Radix2).unwrap(),
            Kind::Bluestein => dft(img, Bluestein),
            Kind::Auto => dft(img, Auto),
        }
    }

    fn backward<P: Precision>(kind: Kind, spectrum: &Spectrum<P>) -> Image<P> {
        match kind {
            Kind::Radix2 => spectrum.inverse(Radix2).unwrap(),
            Kind::Bluestein => spectrum.inverse(Bluestein),
            Kind::Auto => spectrum.inverse(Auto),
        }
    }

    /// The tolerance of one axis of `len` points, relative to the whole
    /// result: Higham's bound for radix-2, the measured one for Bluestein.
    fn axis<P: Precision>(kind: Kind, len: usize) -> f64 {
        let power_of_two = len == 0 || len.is_power_of_two();
        match kind {
            Kind::Radix2 => higham(len, P::U, P::MU),
            Kind::Auto if power_of_two => higham(len, P::U, P::MU),
            _ if len <= 1 => 0.0,
            _ => P::BLUESTEIN,
        }
    }

    /// The tolerance of the two-dimensional transform, relative to the
    /// whole spectrum.
    fn plane<P: Precision>(kind: Kind, w: usize, h: usize) -> f64 {
        compose(axis::<P>(kind, w), axis::<P>(kind, h))
    }

    fn image<P: Precision>(values: &[f64], w: usize, h: usize) -> Image<P> {
        Image::from_vec(w, h, values.iter().map(|&v| P::of(v)).collect()).unwrap()
    }

    fn stored<P: Precision>(spectrum: &Spectrum<P>) -> (Vec<(f64, f64)>, usize) {
        let (bins, half) = spectrum.stored();
        (bins.iter().map(|b| b.to_f64()).collect(), half)
    }

    /// The bins of a full spectrum, `h` rows of `w`, that the half keeps.
    fn half_of(full: &[(f64, f64)], w: usize, h: usize, half: usize) -> Vec<(f64, f64)> {
        (0..h)
            .flat_map(|ky| (0..half).map(move |kx| full[ky * w + kx]))
            .collect()
    }

    const RADIX2_SIZES: [(usize, usize); 8] = [
        (1, 1),
        (2, 1),
        (1, 16),
        (16, 1),
        (8, 4),
        (4, 8),
        (16, 16),
        (64, 32),
    ];
    const BLUESTEIN_SIZES: [(usize, usize); 9] = [
        (1, 1),
        (3, 1),
        (1, 6),
        (5, 3),
        (7, 9),
        (12, 10),
        (15, 17),
        (16, 16),
        (30, 7),
    ];
    const AUTO_SIZES: [(usize, usize); 6] = [(12, 8), (8, 12), (16, 16), (20, 20), (33, 4), (1, 7)];

    fn cases() -> impl Iterator<Item = (Kind, usize, usize)> {
        let radix2 = RADIX2_SIZES.into_iter().map(|(w, h)| (Kind::Radix2, w, h));
        let bluestein = BLUESTEIN_SIZES
            .into_iter()
            .map(|(w, h)| (Kind::Bluestein, w, h));
        let auto = AUTO_SIZES.into_iter().map(|(w, h)| (Kind::Auto, w, h));
        radix2.chain(bluestein).chain(auto)
    }

    /// The stored half is part of the full spectrum, so its error is at
    /// most the whole one; and the full spectrum of a real image holds at
    /// most twice the energy of the stored half, hence the √2.
    fn agrees_with_the_definition<P: Precision>() {
        let mut rng = Rng::new(50);
        for (kind, w, h) in cases() {
            let values: Vec<f64> = (0..w * h).map(|_| rng.value()).collect();
            let spectrum = forward(kind, &image::<P>(&values, w, h));
            let (got, half) = stored(&spectrum);
            let want = half_of(&naive_2d(&values, w, h), w, h, half);
            let err = rel_error(&got, &want);
            let tol =
                SQRT_2 * plane::<P>(kind, w, h) + 2.0 * compose(oracle_error(w), oracle_error(h));
            assert!(err <= tol, "{kind:?} {w}x{h}: {err:e} > {tol:e}");
        }
    }

    #[test]
    fn single_precision_agrees_with_the_definition() {
        agrees_with_the_definition::<MonoF32>();
    }

    #[test]
    fn double_precision_agrees_with_the_definition() {
        agrees_with_the_definition::<MonoF64>();
    }

    fn round_trips<P: Precision>() {
        let mut rng = Rng::new(60);
        for (kind, w, h) in cases() {
            let values: Vec<f64> = (0..w * h).map(|_| rng.value()).collect();
            let img = image::<P>(&values, w, h);
            let back = backward(kind, &forward(kind, &img));
            assert_eq!(back.size(), img.size());
            let got: Vec<(f64, f64)> = (0..h)
                .flat_map(|y| (0..w).map(move |x| (x, y)))
                .map(|(x, y)| (back.pixel_at(x, y).get(), 0.0))
                .collect();
            let want: Vec<(f64, f64)> = values.iter().map(|&v| (v, 0.0)).collect();
            let t = plane::<P>(kind, w, h);
            // Forward, inverse, and one rounding for each of the two
            // divisions by a side that is not a power of two.
            let tol = compose(SQRT_2 * t, t) + 2.0 * P::U;
            let err = rel_error(&got, &want);
            assert!(err <= tol, "{kind:?} {w}x{h}: {err:e} > {tol:e}");
        }
    }

    #[test]
    fn single_precision_round_trips() {
        round_trips::<MonoF32>();
    }

    #[test]
    fn double_precision_round_trips() {
        round_trips::<MonoF64>();
    }

    fn keeps_the_energy<P: Precision>() {
        let mut rng = Rng::new(70);
        for (kind, w, h) in cases() {
            let values: Vec<f64> = (0..w * h).map(|_| rng.value()).collect();
            let (bins, half) = stored(&forward(kind, &image::<P>(&values, w, h)));
            let energy: f64 = values.iter().map(|v| v * v).sum();
            // Every stored column but the zero frequency, and the Nyquist
            // frequency of an even width, stands for itself and its mirror.
            let spectral: f64 = bins
                .iter()
                .enumerate()
                .map(|(i, &(re, im))| {
                    let kx = i % half;
                    let once = kx == 0 || (w % 2 == 0 && kx == w / 2);
                    (if once { 1.0 } else { 2.0 }) * (re * re + im * im)
                })
                .sum::<f64>()
                / (w * h) as f64;
            let t = SQRT_2 * plane::<P>(kind, w, h);
            let tol = 2.0 * t + t * t + 1e-14;
            let err = (spectral - energy).abs() / energy;
            assert!(err <= tol, "{kind:?} {w}x{h}: {err:e} > {tol:e}");
        }
    }

    #[test]
    fn single_precision_keeps_the_energy() {
        keeps_the_energy::<MonoF32>();
    }

    #[test]
    fn double_precision_keeps_the_energy() {
        keeps_the_energy::<MonoF64>();
    }

    fn is_linear<P: Precision>() {
        let mut rng = Rng::new(80);
        let (a, b) = (0.5, -2.0);
        let norm = |v: &[(f64, f64)]| v.iter().map(|&(r, i)| r * r + i * i).sum::<f64>().sqrt();
        for (kind, w, h) in cases() {
            // Twelve significant bits, so a·x + b·y is exact in `f32`.
            let x: Vec<f64> = (0..w * h).map(|_| rng.coarse()).collect();
            let y: Vec<f64> = (0..w * h).map(|_| rng.coarse()).collect();
            let z: Vec<f64> = x.iter().zip(&y).map(|(&p, &q)| a * p + b * q).collect();
            let (fx, _) = stored(&forward(kind, &image::<P>(&x, w, h)));
            let (fy, _) = stored(&forward(kind, &image::<P>(&y, w, h)));
            let (fz, _) = stored(&forward(kind, &image::<P>(&z, w, h)));
            let combined: Vec<(f64, f64)> = fx
                .iter()
                .zip(&fy)
                .map(|(&(xr, xi), &(yr, yi))| (a * xr + b * yr, a * xi + b * yi))
                .collect();
            let (nx, ny, nz) = (norm(&fx), norm(&fy), norm(&fz));
            if nz == 0.0 {
                continue;
            }
            let t = SQRT_2 * plane::<P>(kind, w, h);
            let tol = 1.01 * t * (nz + a.abs() * nx + b.abs() * ny) / nz;
            let err = rel_error(&combined, &fz);
            assert!(err <= tol, "{kind:?} {w}x{h}: {err:e} > {tol:e}");
        }
    }

    #[test]
    fn single_precision_is_linear() {
        is_linear::<MonoF32>();
    }

    #[test]
    fn double_precision_is_linear() {
        is_linear::<MonoF64>();
    }

    /// The spectrum of a unit impulse at (x0, y0) is
    /// `exp(−2πi·(kx·x0/W + ky·y0/H))`, of modulus 1 everywhere.
    fn transforms_an_impulse<P: Precision>() {
        for (kind, w, h) in cases() {
            let (x0, y0) = (w / 3, h / 2);
            let mut values = vec![0.0; w * h];
            values[y0 * w + x0] = 1.0;
            let (got, half) = stored(&forward(kind, &image::<P>(&values, w, h)));
            let want: Vec<(f64, f64)> = (0..h)
                .flat_map(|ky| (0..half).map(move |kx| (kx, ky)))
                .map(|(kx, ky)| {
                    let (c1, s1) = reference_root(kx * x0 % w, w);
                    let (c2, s2) = reference_root(ky * y0 % h, h);
                    // (c1 − i·s1)(c2 − i·s2)
                    (c1 * c2 - s1 * s2, -(s1 * c2 + c1 * s2))
                })
                .collect();
            let tol = SQRT_2 * plane::<P>(kind, w, h) + 4.0 * U64;
            let err = rel_error(&got, &want);
            assert!(err <= tol, "{kind:?} {w}x{h}: {err:e} > {tol:e}");
        }
    }

    #[test]
    fn single_precision_transforms_an_impulse() {
        transforms_an_impulse::<MonoF32>();
    }

    #[test]
    fn double_precision_transforms_an_impulse() {
        transforms_an_impulse::<MonoF64>();
    }

    #[test]
    fn radix2_transforms_a_constant_exactly() {
        let img = Image::fill(16, 8, MonoF32::new(1.5));
        let spectrum = dft(&img, Radix2).unwrap();
        let (bins, half) = spectrum.stored();
        assert_eq!(bins.len(), 8 * half);
        assert_eq!(bins[0], ComplexF32::new(192.0, 0.0));
        assert!(bins[1..].iter().all(|b| b.re == 0.0 && b.im == 0.0));
        assert_eq!(spectrum.inverse(Radix2).unwrap(), img);
    }

    #[test]
    fn radix2_rejects_a_side_that_is_not_a_power_of_two() {
        let img = Image::fill(12, 8, MonoF32::new(1.0));
        let unsupported = Error::UnsupportedDftSize {
            method: "Radix2",
            size: Size::new(12, 8),
        };
        assert_eq!(dft(&img, Radix2), Err(unsupported.clone()));
        // Nor does it invert a spectrum of that size, whoever computed it.
        assert_eq!(dft(&img, Auto).inverse(Radix2), Err(unsupported));
    }

    #[test]
    fn an_empty_image_has_an_empty_spectrum_under_every_method() {
        for size in [Size::new(0, 0), Size::new(0, 5), Size::new(7, 0)] {
            let img = Image::<MonoF32>::zero(size.width, size.height);
            for kind in [Kind::Radix2, Kind::Bluestein, Kind::Auto] {
                let spectrum = forward(kind, &img);
                assert_eq!(spectrum.source_size(), size);
                assert!(spectrum.stored().0.is_empty());
                assert_eq!(backward(kind, &spectrum).size(), size, "{kind:?}");
            }
        }
    }

    #[test]
    fn auto_runs_radix2_on_powers_of_two_and_bluestein_otherwise() {
        let mut rng = Rng::new(90);
        let pow2: Vec<f64> = (0..16 * 8).map(|_| rng.value()).collect();
        let img = image::<MonoF64>(&pow2, 16, 8);
        assert_eq!(dft(&img, Auto), dft(&img, Radix2).unwrap());
        let other: Vec<f64> = (0..12 * 10).map(|_| rng.value()).collect();
        let img = image::<MonoF64>(&other, 12, 10);
        assert_eq!(dft(&img, Auto), dft(&img, Bluestein));
    }

    #[test]
    fn a_view_transforms_as_its_pixels_do() {
        let big = Image::generate(20, 12, |x, y| MonoF32::new(((x * 7 + y * 3) % 5) as f32));
        let view = big.roi(Rectangle::new((3, 2), (8, 4))).unwrap();
        let copy = Image::generate(8, 4, |x, y| view.pixel_at(x, y));
        assert_eq!(dft(&view, Radix2), dft(&copy, Radix2));
    }

    #[test]
    fn the_methods_answer_which_sizes_they_accept() {
        assert!(Radix2.accepts(Size::new(1024, 1)));
        assert!(Radix2.accepts(Size::new(0, 0)));
        assert!(!Radix2.accepts(Size::new(1024, 1000)));
        assert_eq!(
            Radix2.next_size(Size::new(1920, 1080)),
            Size::new(2048, 2048)
        );
        assert!(Radix2.accepts(Size::new(0, 5)));
        assert_eq!(Radix2.next_size(Size::new(0, 5)), Size::new(0, 5));
        assert_eq!(Radix2.next_size(Size::new(512, 3)), Size::new(512, 4));
        for size in [Size::new(1920, 1080), Size::new(0, 7)] {
            assert!(Bluestein.accepts(size) && Auto.accepts(size));
            assert_eq!(Bluestein.next_size(size), size);
            assert_eq!(Auto.next_size(size), size);
        }
    }

    #[test]
    #[should_panic(expected = "no power of two")]
    fn radix2_has_no_size_beyond_the_largest_power_of_two() {
        let _ = Radix2.next_size(Size::new(usize::MAX, 1));
    }
}

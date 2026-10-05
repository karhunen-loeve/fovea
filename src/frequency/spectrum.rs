//! The spectrum of an image, and the two passes that compute and invert it.

use super::engine::{Axis, Cplx, transpose};
use super::method::DftMethod;
use crate::Size;
use crate::image::{Image, RasterImage};
use crate::pixel::{ComplexF32, ComplexF64, MonoF32, MonoF64};

mod sealed {
    /// The conversion between a source pixel and the real part of a bin.
    pub trait Source: Copy {
        /// The value, exactly.
        fn to_f64(self) -> f64;
        /// The pixel holding `v`, rounded once.
        fn from_f64(v: f64) -> Self;
    }
}

/// A pixel type whose images have a spectrum: [`MonoF32`] and [`MonoF64`].
/// Sealed.
///
/// It fixes the type of a frequency bin, [`ComplexF32`] for `MonoF32` and
/// [`ComplexF64`] for `MonoF64`, and the transform computes in the
/// precision of the bin.
pub trait SpectrumSource: sealed::Source {
    /// The type of one frequency bin.
    type Bin: Cplx;
}

impl sealed::Source for MonoF32 {
    #[inline]
    fn to_f64(self) -> f64 {
        f64::from(self.0)
    }
    #[inline]
    fn from_f64(v: f64) -> Self {
        MonoF32::new(v as f32)
    }
}

impl SpectrumSource for MonoF32 {
    type Bin = ComplexF32;
}

impl sealed::Source for MonoF64 {
    #[inline]
    fn to_f64(self) -> f64 {
        self.0
    }
    #[inline]
    fn from_f64(v: f64) -> Self {
        MonoF64::new(v)
    }
}

impl SpectrumSource for MonoF64 {
    type Bin = ComplexF64;
}

/// The discrete Fourier transform of an image of pixel type `P`.
///
/// Its values are the transform as the textbook defines it, unscaled and
/// with the negative exponent,
/// `X[kx, ky] = Σ x[x, y] · exp(−2πi · (kx·x/W + ky·y/H))`, which are the
/// numbers numpy, OpenCV and FFTW produce by default in the forward
/// direction. The image is transformed as one period of a periodic
/// signal: its right edge meets its left and its bottom its top, and a
/// jump between them shows in the spectrum. Pad the image first, with
/// [`pad`](crate::transform::pad), to choose what lies beyond the edges.
///
/// A spectrum is not an image. Blurring, cropping or resizing its bins
/// would change what each of them means, so the image operations do not
/// accept it. It is made by [`dft`](super::dft) and returns to an image
/// of the source's size with [`inverse`](Self::inverse), which is the
/// exact inverse: the round trip gives back the image, up to rounding,
/// with no factor to apply.
///
/// The spectrum of a real image is Hermitian, `X[−k] = conj(X[k])`, so
/// only half of it is stored. Every change made through the spectrum
/// keeps the symmetry, and the inverse of the spectrum of a real image is
/// real.
///
/// # Example
///
/// ```
/// use fovea::frequency::{Radix2, dft};
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::MonoF32;
///
/// let img = Image::generate(8, 4, |x, y| MonoF32::new((x * y) as f32));
/// let spectrum = dft(&img, Radix2)?;
/// assert_eq!(spectrum.source_size(), img.size());
///
/// let back = spectrum.inverse(Radix2)?;
/// assert!((back.pixel_at(5, 3).0 - 15.0).abs() < 1e-5);
/// # Ok::<(), fovea::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Spectrum<P: SpectrumSource> {
    source_size: Size,
    /// `height` rows of `half_width(width)` bins, the frequencies
    /// `0..=width/2` along x for every frequency along y.
    bins: Vec<P::Bin>,
}

/// How many columns of bins the half spectrum of an image `width` wide
/// keeps: the frequencies 0 to ⌊width/2⌋.
fn half_width(width: usize) -> usize {
    if width == 0 { 0 } else { width / 2 + 1 }
}

impl<P: SpectrumSource> Spectrum<P> {
    /// The size of the image this is the spectrum of, and of the image the
    /// inverse returns.
    #[must_use]
    pub fn source_size(&self) -> Size {
        self.source_size
    }

    /// The image this is the spectrum of, by `method`.
    ///
    /// The method need not be the one that computed the spectrum: every
    /// method computes the same transform, up to rounding. Like
    /// [`dft`](super::dft), the call returns a `Result` exactly when the
    /// method does not accept every size.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::frequency::{Auto, Bluestein, dft};
    /// use fovea::image::{Image, ImageView};
    /// use fovea::pixel::MonoF64;
    ///
    /// let img = Image::generate(6, 5, |x, y| MonoF64::new((x + 10 * y) as f64));
    /// let back = dft(&img, Auto).inverse(Bluestein);
    /// assert!((back.pixel_at(4, 3).0 - 34.0).abs() < 1e-12);
    /// ```
    #[must_use]
    pub fn inverse<M: DftMethod<P>>(&self, method: M) -> M::InverseOutput {
        method.invert(self)
    }

    /// The spectrum of `src`, along x by `along_x` and along y by `along_y`.
    pub(crate) fn forward<I>(src: &I, along_x: &Axis<P::Bin>, along_y: &Axis<P::Bin>) -> Self
    where
        I: RasterImage<Pixel = P>,
    {
        let Size { width, height } = src.size();
        let half = half_width(width);
        let mut bins = Vec::with_capacity(half * height);
        let mut work = Vec::new();
        let mut row = vec![P::Bin::ZERO; width];
        for y in 0..height {
            for (b, &p) in row.iter_mut().zip(src.row(y)) {
                *b = P::Bin::from_f64(p.to_f64(), 0.0);
            }
            along_x.forward(&mut row, &mut work);
            bins.extend_from_slice(&row[..half]);
        }
        if half > 0 && height > 0 {
            let mut columns = vec![P::Bin::ZERO; half * height];
            transpose(&bins, half, height, &mut columns);
            for column in columns.chunks_exact_mut(height) {
                along_y.forward(column, &mut work);
            }
            transpose(&columns, height, half, &mut bins);
        }
        Self {
            source_size: src.size(),
            bins,
        }
    }

    /// The image this is the spectrum of, along x by `along_x` and along y
    /// by `along_y`.
    pub(crate) fn invert(&self, along_x: &Axis<P::Bin>, along_y: &Axis<P::Bin>) -> Image<P> {
        let Size { width, height } = self.source_size;
        let half = half_width(width);
        let mut pixels = Vec::with_capacity(width * height);
        if half > 0 && height > 0 {
            let mut work = Vec::new();
            let mut columns = vec![P::Bin::ZERO; half * height];
            transpose(&self.bins, half, height, &mut columns);
            for column in columns.chunks_exact_mut(height) {
                along_y.inverse_unscaled(column, &mut work);
                for z in column.iter_mut() {
                    *z = z.div_count(height);
                }
            }
            let mut rows = vec![P::Bin::ZERO; half * height];
            transpose(&columns, height, half, &mut rows);
            // Each row now holds the transform of one real row of the
            // image, so its other half follows by symmetry.
            let mut full = vec![P::Bin::ZERO; width];
            for stored in rows.chunks_exact(half) {
                full[..half].copy_from_slice(stored);
                for k in half..width {
                    full[k] = stored[width - k].conj();
                }
                along_x.inverse_unscaled(&mut full, &mut work);
                pixels.extend(
                    full.iter()
                        .map(|z| P::from_f64(z.div_count(width).to_f64().0)),
                );
            }
        }
        Image::from_vec(width, height, pixels).expect("rows fill width * height exactly")
    }

    /// The stored bins, `height` rows of `half_width(width)`.
    #[cfg(test)]
    pub(crate) fn stored(&self) -> (&[P::Bin], usize) {
        (&self.bins, half_width(self.source_size.width))
    }
}

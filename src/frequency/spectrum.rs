//! The spectrum of an image, and the two passes that compute and invert it.

use super::engine::{Axis, Cplx, transpose};
use super::method::DftMethod;
use super::units::{Frequency, FrequencyIndex, FrequencyUnit, canonical, convert, frequency_in};
use crate::image::{Image, RasterImage};
use crate::pixel::{ComplexF32, ComplexF64, MonoF32, MonoF64};
use crate::{Error, Size};

pub(super) mod sealed {
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
/// accept it. It is made by [`dft`](super::dft); it is read and changed by
/// frequency, through [`at`](Self::at) and [`apply`](Self::apply); it is
/// shown through [`centered`](Self::centered),
/// [`magnitude`](Self::magnitude), [`power`](Self::power) and
/// [`phase`](Self::phase), ordinary images with the zero frequency in the
/// middle; and it returns to an image of the source's size with
/// [`inverse`](Self::inverse), which is the exact inverse: the round trip
/// gives back the image, up to rounding, with no factor to apply.
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

    /// Multiplies every bin by the bin of `other` at the same frequency.
    ///
    /// The product of two spectra is the spectrum of the cyclic convolution
    /// of their images: the second image wraps around the edges of the
    /// first. Where that wrap is right, for an image that is periodic by
    /// nature, this is the cheapest convolution there is, since nothing is
    /// padded; for the result of a convolution with a border policy, use
    /// [`frequency::convolve`](super::convolve). The product of the spectra
    /// of two real images is the spectrum of a real image, and its symmetry
    /// is kept exactly.
    ///
    /// # Errors
    ///
    /// [`Error::SizeMismatch`] if the two images were not of one size.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::frequency::{Auto, dft};
    /// use fovea::image::{Image, ImageView};
    /// use fovea::pixel::MonoF64;
    ///
    /// // A shift by one pixel to the right, cyclic: the kernel is an impulse at x = 1.
    /// let img = Image::generate(8, 1, |x, _| MonoF64::new(x as f64));
    /// let shift = Image::generate(8, 1, |x, _| MonoF64::new(if x == 1 { 1.0 } else { 0.0 }));
    /// let mut spectrum = dft(&img, Auto);
    /// spectrum.multiply(&dft(&shift, Auto))?;
    /// let moved = spectrum.inverse(Auto);
    /// assert!((moved.pixel_at(0, 0).0 - 7.0).abs() < 1e-12);
    /// assert!((moved.pixel_at(3, 0).0 - 2.0).abs() < 1e-12);
    /// # Ok::<(), fovea::Error>(())
    /// ```
    pub fn multiply(&mut self, other: &Spectrum<P>) -> Result<(), Error> {
        self.combine(other, |a, b| a * b)
    }

    /// Multiplies every bin by the conjugate of the bin of `other` at the
    /// same frequency.
    ///
    /// The product with the conjugate is the spectrum of the cyclic
    /// cross-correlation of the two images, the basis of phase correlation.
    /// Like [`multiply`](Self::multiply) it keeps the symmetry of a real
    /// image.
    ///
    /// # Errors
    ///
    /// [`Error::SizeMismatch`] if the two images were not of one size.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::frequency::{Auto, FrequencyIndex, dft};
    /// use fovea::image::Image;
    /// use fovea::pixel::MonoF64;
    ///
    /// let img = Image::generate(6, 4, |x, y| MonoF64::new((x + 3 * y) as f64));
    /// let mut power = dft(&img, Auto);
    /// power.multiply_conjugate(&dft(&img, Auto))?;
    /// // With itself, every bin becomes its squared magnitude, which is real.
    /// assert_eq!(power.at(FrequencyIndex::new(1, 1)).im, 0.0);
    /// # Ok::<(), fovea::Error>(())
    /// ```
    pub fn multiply_conjugate(&mut self, other: &Spectrum<P>) -> Result<(), Error> {
        self.combine(other, |a, b| a * b.conj())
    }

    fn combine(
        &mut self,
        other: &Spectrum<P>,
        op: impl Fn(P::Bin, P::Bin) -> P::Bin,
    ) -> Result<(), Error> {
        if self.source_size != other.source_size {
            return Err(Error::SizeMismatch {
                expected: self.source_size,
                actual: other.source_size,
            });
        }
        for (a, &b) in self.bins.iter_mut().zip(&other.bins) {
            *a = op(*a, b);
        }
        // The products of conjugate pairs are conjugate, and those of real
        // bins real, as computed; symmetrize keeps that true whatever the
        // rounding.
        self.symmetrize();
        Ok(())
    }

    /// The bin at `index`, a frequency in bins taken modulo the size.
    ///
    /// A bin that is not stored is the conjugate of its mirror, which is.
    /// The value is the unscaled transform: a cosine of amplitude A at a
    /// bin's frequency, on a W × H image, gives A·W·H/2 at that bin and at
    /// its mirror.
    ///
    /// # Panics
    ///
    /// Panics if the image had no pixels, so the spectrum has no bin.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::frequency::{Auto, FrequencyIndex, dft};
    /// use fovea::image::Image;
    /// use fovea::pixel::{ComplexF32, MonoF32};
    ///
    /// let spectrum = dft(&Image::fill(6, 4, MonoF32::new(0.5)), Auto);
    /// // The zero frequency holds the sum of all pixels.
    /// assert_eq!(spectrum.at(FrequencyIndex::new(0, 0)), ComplexF32::new(12.0, 0.0));
    /// ```
    #[must_use]
    pub fn at(&self, index: FrequencyIndex) -> P::Bin {
        let (i, conjugate) = self.locate(index);
        if conjugate {
            self.bins[i].conj()
        } else {
            self.bins[i]
        }
    }

    /// The frequency of the bin at `index`, in the unit `U`, with the index
    /// taken into the canonical range first.
    ///
    /// # Panics
    ///
    /// Panics if the image had no pixels, so the spectrum has no bin.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::frequency::{Auto, Bins, CyclesPerPixel, Frequency, FrequencyIndex, dft};
    /// use fovea::image::Image;
    /// use fovea::pixel::MonoF32;
    ///
    /// let spectrum = dft(&Image::fill(512, 256, MonoF32::new(0.0)), Auto);
    /// let f: Frequency<CyclesPerPixel> = spectrum.frequency_of(FrequencyIndex::new(1, 1));
    /// assert_eq!((f.fx(), f.fy()), (1.0 / 512.0, 1.0 / 256.0));
    ///
    /// // Bin 256 of an even side is its Nyquist bin, which counts as negative.
    /// let n: Frequency<Bins> = spectrum.frequency_of(FrequencyIndex::new(256, 0));
    /// assert_eq!(n.fx(), -256.0);
    /// ```
    #[must_use]
    pub fn frequency_of<U: FrequencyUnit>(&self, index: FrequencyIndex) -> Frequency<U> {
        let size = self.frequencies();
        frequency_in(
            canonical(index.kx, size.width),
            canonical(index.ky, size.height),
            size,
        )
    }

    /// The frequency `f` in the unit `U`, through the size of this
    /// spectrum's image.
    ///
    /// # Panics
    ///
    /// Panics if the image had no pixels, so a bin has no size.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::frequency::{Auto, Bins, Frequency, dft};
    /// use fovea::image::Image;
    /// use fovea::pixel::MonoF32;
    ///
    /// let spectrum = dft(&Image::fill(512, 256, MonoF32::new(0.0)), Auto);
    /// let quarter = Frequency::cycles_per_pixel(0.25, 0.25);
    /// let in_bins: Frequency<Bins> = spectrum.convert(quarter);
    /// assert_eq!((in_bins.fx(), in_bins.fy()), (128.0, 64.0));
    /// ```
    #[must_use]
    pub fn convert<U: FrequencyUnit, V: FrequencyUnit>(&self, f: Frequency<V>) -> Frequency<U> {
        convert(f, self.frequencies())
    }

    /// Changes every bin by `filter`, which receives the bin's frequency in
    /// the unit it names, [`Bins`](super::Bins) or
    /// [`CyclesPerPixel`](super::CyclesPerPixel).
    ///
    /// The filter visits each independent frequency once, on half of the
    /// plane, and the other half follows by the symmetry of a real image:
    /// what it writes at `f` holds at `−f` as the conjugate. That is the
    /// form every filter for a real image has; a filter that is symmetric,
    /// as a radial band or a phase ramp is, sees no difference. Two
    /// consequences of the symmetry are kept by the spectrum, not by the
    /// filter:
    ///
    /// - In the column of the zero frequency, and in the Nyquist column of
    ///   an even width, both members of a pair `(kx, ky)` and `(kx, −ky)`
    ///   are stored. The filter visits the non-negative `ky` and the
    ///   Nyquist row, and the partner receives the conjugate.
    /// - A bin that is its own mirror (the zero frequency, and a Nyquist
    ///   bin of an even side) is real in the spectrum of any real image, so
    ///   only the real part of what the filter writes there is kept. A
    ///   shift by a fraction of a pixel, for instance, cannot be told
    ///   apart from its mirror at the Nyquist frequency.
    ///
    /// The frequencies are those of the canonical range.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::frequency::{Auto, CyclesPerPixel, Frequency, dft};
    /// use fovea::image::{Image, ImageView};
    /// use fovea::pixel::{ComplexF32, MonoF32};
    /// use std::f32::consts::PI;
    ///
    /// // Fine stripes, period 4 px, on a slow wave, period 64 px.
    /// let wave = |x: usize, period: f32| (2.0 * PI * x as f32 / period).cos();
    /// let img = Image::generate(64, 64, |x, _| MonoF32::new(wave(x, 64.0) + wave(x, 4.0)));
    ///
    /// // Keep the structures coarser than 8 px.
    /// let mut spectrum = dft(&img, Auto);
    /// spectrum.apply(|f: Frequency<CyclesPerPixel>, bin: &mut ComplexF32| {
    ///     if f.radius() > 1.0 / 8.0 {
    ///         *bin = ComplexF32::new(0.0, 0.0);
    ///     }
    /// });
    /// let slow = spectrum.inverse(Auto);
    /// assert!((slow.pixel_at(10, 5).0 - wave(10, 64.0)).abs() < 1e-5);
    /// ```
    pub fn apply<U, F>(&mut self, mut filter: F)
    where
        U: FrequencyUnit,
        F: FnMut(Frequency<U>, &mut P::Bin),
    {
        let Size { width, height } = self.source_size;
        let half = half_width(width);
        for row in 0..height {
            let ky = canonical(row as isize, height);
            let nyquist_row = height % 2 == 0 && row == height / 2;
            for col in 0..half {
                let doubled = col == 0 || (width % 2 == 0 && col == width / 2);
                if doubled && ky < 0 && !nyquist_row {
                    // The partner of a bin visited in an earlier row.
                    continue;
                }
                let kx = canonical(col as isize, width);
                let i = row * half + col;
                filter(frequency_in(kx, ky, self.source_size), &mut self.bins[i]);
                if doubled {
                    if row == 0 || nyquist_row {
                        self.bins[i] = self.bins[i].real_part();
                    } else {
                        self.bins[(height - row) * half + col] = self.bins[i].conj();
                    }
                }
            }
        }
    }

    /// The full spectrum as an image, with the zero frequency at
    /// `(⌊W/2⌋, ⌊H/2⌋)`.
    ///
    /// Pixel `(x, y)` holds the bin `(x − ⌊W/2⌋, y − ⌊H/2⌋)`, so the
    /// columns run through the canonical range from `−⌊W/2⌋`. The result
    /// is an ordinary complex image and leaves the spectrum: it is the way
    /// to look at a spectrum, and there is no way back from it.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::frequency::{Auto, dft};
    /// use fovea::image::{Image, ImageView};
    /// use fovea::pixel::{ComplexF32, MonoF32};
    ///
    /// let spectrum = dft(&Image::fill(5, 4, MonoF32::new(1.0)), Auto);
    /// let shown = spectrum.centered();
    /// assert_eq!(shown.size(), fovea::Size::new(5, 4));
    /// assert_eq!(shown.pixel_at(2, 2), ComplexF32::new(20.0, 0.0));
    /// ```
    #[must_use]
    pub fn centered(&self) -> Image<P::Bin> {
        let Size { width, height } = self.source_size;
        let (cx, cy) = ((width / 2) as isize, (height / 2) as isize);
        Image::generate(width, height, |x, y| {
            self.at(FrequencyIndex::new(x as isize - cx, y as isize - cy))
        })
    }

    /// The magnitude `|X|` of every bin, in the layout of
    /// [`centered`](Self::centered).
    ///
    /// Computed in `f64` and rounded once to the source's precision. A
    /// component the image does not contain comes out as rounding noise,
    /// not as zero.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::frequency::{Auto, dft};
    /// use fovea::image::{Image, ImageView};
    /// use fovea::pixel::MonoF32;
    ///
    /// let img = Image::generate(8, 8, |x, _| MonoF32::new(if x % 2 == 0 { 1.0 } else { -1.0 }));
    /// let magnitude = dft(&img, Auto).magnitude();
    /// // All the energy at the Nyquist frequency along x, column 0 of the centred layout.
    /// assert_eq!(magnitude.pixel_at(0, 4), MonoF32::new(64.0));
    /// ```
    #[must_use]
    pub fn magnitude(&self) -> Image<P> {
        self.centered_map(f64::hypot)
    }

    /// The power `|X|²` of every bin, in the layout of
    /// [`centered`](Self::centered).
    ///
    /// The power is the squared magnitude, as in signal processing; some
    /// libraries call the magnitude itself the power spectrum. Computed in
    /// `f64` and rounded once to the source's precision, so in `f32` a
    /// magnitude above about 1.8·10¹⁹ has an infinite power.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::frequency::{Auto, dft};
    /// use fovea::image::{Image, ImageView};
    /// use fovea::pixel::MonoF64;
    ///
    /// let power = dft(&Image::fill(4, 4, MonoF64::new(0.5)), Auto).power();
    /// assert_eq!(power.pixel_at(2, 2), MonoF64::new(64.0));
    /// ```
    #[must_use]
    pub fn power(&self) -> Image<P> {
        self.centered_map(|re, im| re * re + im * im)
    }

    /// The phase of every bin in radians, in `(−π, π]`, in the layout of
    /// [`centered`](Self::centered).
    ///
    /// The phase of a weak component is mostly rounding noise, and that of
    /// a component the image does not contain is meaningless; read it
    /// together with the magnitude.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::frequency::{Auto, dft};
    /// use fovea::image::{Image, ImageView};
    /// use fovea::pixel::MonoF64;
    ///
    /// // An impulse at x = 1: the phase falls by 2π/8 per bin along x.
    /// let img = Image::generate(8, 1, |x, _| MonoF64::new(if x == 1 { 1.0 } else { 0.0 }));
    /// let phase = dft(&img, Auto).phase();
    /// let step = -2.0 * std::f64::consts::PI / 8.0;
    /// assert!((phase.pixel_at(5, 0).0 - step).abs() < 1e-15);
    /// ```
    #[must_use]
    pub fn phase(&self) -> Image<P> {
        self.centered_map(|re, im| im.atan2(re))
    }

    /// `value(re, im)` of every bin, in the layout of `centered`.
    fn centered_map(&self, value: impl Fn(f64, f64) -> f64) -> Image<P> {
        let Size { width, height } = self.source_size;
        let (cx, cy) = ((width / 2) as isize, (height / 2) as isize);
        Image::generate(width, height, |x, y| {
            let (re, im) = self
                .at(FrequencyIndex::new(x as isize - cx, y as isize - cy))
                .to_f64();
            P::from_f64(value(re, im))
        })
    }

    /// The size, which has pixels.
    fn frequencies(&self) -> Size {
        let size = self.source_size;
        assert!(
            size.area() > 0,
            "the spectrum of an image without pixels has no frequencies"
        );
        size
    }

    /// Where the bin `index` is stored, and whether the stored value is its
    /// conjugate.
    fn locate(&self, index: FrequencyIndex) -> (usize, bool) {
        let Size { width, height } = self.frequencies();
        let half = half_width(width);
        let mx = index.kx.rem_euclid(width as isize) as usize;
        let my = index.ky.rem_euclid(height as isize) as usize;
        if mx < half {
            (my * half + mx, false)
        } else {
            ((height - my) % height * half + (width - mx), true)
        }
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
        let mut spectrum = Self {
            source_size: src.size(),
            bins,
        };
        spectrum.symmetrize();
        spectrum
    }

    /// Makes the bins stored twice exact conjugates of each other, and the
    /// bins that are their own mirror real.
    ///
    /// The transform of a real image has these properties exactly; the
    /// computed one has them up to rounding. A pair becomes the mean of one
    /// member and the conjugate of the other, which is the nearest pair
    /// that has the property, and a self-conjugate bin loses the rounding
    /// noise in its imaginary part.
    fn symmetrize(&mut self) {
        let Size { width, height } = self.source_size;
        let half = half_width(width);
        let columns = if width % 2 == 0 && width > 0 {
            vec![0, width / 2]
        } else {
            vec![0]
        };
        for col in columns.into_iter().filter(|&c| c < half) {
            for row in 0..height {
                let partner = (height - row) % height;
                let i = row * half + col;
                if partner == row {
                    self.bins[i] = self.bins[i].real_part();
                } else if row < partner {
                    let j = partner * half + col;
                    let (a, b) = (self.bins[i].to_f64(), self.bins[j].to_f64());
                    let mean = P::Bin::from_f64(0.5 * (a.0 + b.0), 0.5 * (a.1 - b.1));
                    self.bins[i] = mean;
                    self.bins[j] = mean.conj();
                }
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frequency::testing::{Rng, naive_2d, rel_error};
    use crate::frequency::{Auto, Bins, CyclesPerPixel, Radix2, dft};
    use crate::image::ImageView;
    use std::collections::HashSet;

    fn random(rng: &mut Rng, w: usize, h: usize) -> (Vec<f64>, Image<MonoF64>) {
        let values: Vec<f64> = (0..w * h).map(|_| rng.value()).collect();
        let img = Image::from_vec(w, h, values.iter().map(|&v| MonoF64::new(v)).collect()).unwrap();
        (values, img)
    }

    const SIZES: [(usize, usize); 9] = [
        (8, 4),
        (5, 3),
        (6, 5),
        (1, 4),
        (7, 1),
        (1, 1),
        (2, 2),
        (6, 6),
        (4, 7),
    ];

    /// The number of bins that are their own mirror: the zero frequency, and
    /// a Nyquist bin per even side.
    fn self_conjugate(w: usize, h: usize) -> usize {
        (1 + usize::from(w % 2 == 0)) * (1 + usize::from(h % 2 == 0))
    }

    fn mirror(i: FrequencyIndex, w: usize, h: usize) -> (isize, isize) {
        (canonical(-i.kx, w), canonical(-i.ky, h))
    }

    #[test]
    fn every_bin_agrees_with_the_full_spectrum() {
        let mut rng = Rng::new(110);
        for (w, h) in SIZES {
            let (values, img) = random(&mut rng, w, h);
            let spectrum = dft(&img, Auto);
            let full = naive_2d(&values, w, h);
            let got: Vec<(f64, f64)> = (0..h)
                .flat_map(|ky| (0..w).map(move |kx| (kx, ky)))
                .map(|(kx, ky)| {
                    spectrum
                        .at(FrequencyIndex::new(kx as isize, ky as isize))
                        .to_f64()
                })
                .collect();
            let err = rel_error(&got, &full);
            assert!(err <= 1e-14, "{w}x{h}: {err:e}");
        }
    }

    #[test]
    fn an_index_is_taken_modulo_the_size() {
        let mut rng = Rng::new(120);
        let (_, img) = random(&mut rng, 6, 5);
        let spectrum = dft(&img, Auto);
        for (kx, ky) in [(0isize, 0isize), (2, 1), (-1, 3), (3, -2), (5, 4)] {
            let base = spectrum.at(FrequencyIndex::new(kx, ky));
            for (dx, dy) in [(6, 0), (-6, 5), (12, -10)] {
                assert_eq!(spectrum.at(FrequencyIndex::new(kx + dx, ky + dy)), base);
            }
        }
    }

    #[test]
    fn the_spectrum_hands_out_the_canonical_range() {
        let spectrum = dft(&Image::fill(8, 5, MonoF64::new(1.0)), Auto);
        let bins = |kx, ky| {
            let f: Frequency<Bins> = spectrum.frequency_of(FrequencyIndex::new(kx, ky));
            (f.fx(), f.fy())
        };
        assert_eq!(bins(3, 0), (3.0, 0.0));
        assert_eq!(bins(4, 0), (-4.0, 0.0));
        assert_eq!(bins(-5, 0), (3.0, 0.0));
        assert_eq!(bins(0, 2), (0.0, 2.0));
        assert_eq!(bins(0, 3), (0.0, -2.0));
        assert_eq!(bins(0, 7), (0.0, 2.0));
        let cycles: Frequency<CyclesPerPixel> = spectrum.frequency_of(FrequencyIndex::new(1, 1));
        assert_eq!((cycles.fx(), cycles.fy()), (1.0 / 8.0, 1.0 / 5.0));
    }

    #[test]
    fn units_convert_through_the_size() {
        let spectrum = dft(&Image::fill(512, 256, MonoF32::new(1.0)), Auto);
        let f = Frequency::bins(37.0, -5.0);
        let c: Frequency<CyclesPerPixel> = spectrum.convert(f);
        assert_eq!((c.fx(), c.fy()), (37.0 / 512.0, -5.0 / 256.0));
        let back: Frequency<Bins> = spectrum.convert(c);
        assert_eq!(back, f);
        let same: Frequency<Bins> = spectrum.convert(f);
        assert_eq!(same, f);
    }

    #[test]
    fn apply_visits_each_independent_frequency_once() {
        for (w, h) in SIZES {
            let mut spectrum = dft(&Image::fill(w, h, MonoF64::new(1.0)), Auto);
            let mut seen = Vec::new();
            spectrum.apply(|f: Frequency<Bins>, _: &mut ComplexF64| {
                seen.push((f.fx() as isize, f.fy() as isize));
            });
            assert_eq!(seen.len(), (w * h + self_conjugate(w, h)) / 2, "{w}x{h}");
            let distinct: HashSet<(isize, isize)> = seen.iter().copied().collect();
            assert_eq!(distinct.len(), seen.len(), "{w}x{h}: a frequency twice");
            let mut covered = HashSet::new();
            for &(kx, ky) in &seen {
                assert_eq!(
                    (canonical(kx, w), canonical(ky, h)),
                    (kx, ky),
                    "not canonical"
                );
                covered.insert((kx, ky));
                covered.insert(mirror(FrequencyIndex::new(kx, ky), w, h));
            }
            assert_eq!(covered.len(), w * h, "{w}x{h}: not every bin is reached");
        }
    }

    #[test]
    fn apply_keeps_the_symmetry_of_a_real_image() {
        for (w, h) in SIZES {
            let mut spectrum = dft(&Image::fill(w, h, MonoF64::new(1.0)), Auto);
            // Neither symmetric nor real: i·(1 + kx + 10·ky).
            spectrum.apply(|f: Frequency<Bins>, bin: &mut ComplexF64| {
                *bin = ComplexF64::new(0.0, 1.0 + f.fx() + 10.0 * f.fy());
            });
            for ky in 0..h as isize {
                for kx in 0..w as isize {
                    let i = FrequencyIndex::new(kx, ky);
                    let (mx, my) = mirror(i, w, h);
                    let here = spectrum.at(i);
                    assert_eq!(spectrum.at(FrequencyIndex::new(mx, my)), here.conjugate());
                    if (mx, my) == (canonical(kx, w), canonical(ky, h)) {
                        assert_eq!(here.im, 0.0, "{w}x{h}: ({kx}, {ky}) is its own mirror");
                    }
                }
            }
        }
    }

    #[test]
    fn apply_that_changes_nothing_changes_no_bit() {
        let mut rng = Rng::new(130);
        for (w, h) in SIZES {
            let (_, img) = random(&mut rng, w, h);
            let before = dft(&img, Auto);
            let mut after = before.clone();
            after.apply(|_: Frequency<Bins>, _: &mut ComplexF64| {});
            assert_eq!(after, before, "{w}x{h}");
        }
    }

    #[test]
    fn the_transform_is_exactly_symmetric() {
        let mut rng = Rng::new(135);
        for (w, h) in SIZES {
            let (_, img) = random(&mut rng, w, h);
            let spectrum = dft(&img, Auto);
            for ky in 0..h as isize {
                for kx in 0..w as isize {
                    let i = FrequencyIndex::new(kx, ky);
                    let (mx, my) = mirror(i, w, h);
                    let here = spectrum.at(i);
                    assert_eq!(spectrum.at(FrequencyIndex::new(mx, my)), here.conjugate());
                    if (mx, my) == (canonical(kx, w), canonical(ky, h)) {
                        assert_eq!(here.im, 0.0, "{w}x{h}: ({kx}, {ky})");
                    }
                }
            }
        }
    }

    #[test]
    fn a_phase_ramp_shifts_the_image() {
        let mut rng = Rng::new(140);
        let (w, h) = (12, 10);
        let (_, img) = random(&mut rng, w, h);
        let (dx, dy) = (3.0, -2.0);
        let mut spectrum = dft(&img, Auto);
        spectrum.apply(|f: Frequency<CyclesPerPixel>, bin: &mut ComplexF64| {
            let angle = -2.0 * std::f64::consts::PI * (f.fx() * dx + f.fy() * dy);
            *bin = *bin * ComplexF64::new(angle.cos(), angle.sin());
        });
        let shifted = spectrum.inverse(Auto);
        for y in 0..h {
            for x in 0..w {
                let sx = (x + w - 3) % w;
                let sy = (y + 2) % h;
                let d = shifted.pixel_at(x, y).0 - img.pixel_at(sx, sy).0;
                assert!(d.abs() < 1e-13, "({x}, {y}): {d:e}");
            }
        }
    }

    #[test]
    fn a_fractional_shift_keeps_the_image_real() {
        let img = Image::generate(8, 8, |x, y| MonoF32::new(((x * 3 + y) % 5) as f32));
        let mut spectrum = dft(&img, Radix2).unwrap();
        spectrum.apply(|f: Frequency<CyclesPerPixel>, bin: &mut ComplexF32| {
            let angle = (-2.0 * std::f64::consts::PI * f.fx() * 0.3) as f32;
            *bin = *bin * ComplexF32::new(angle.cos(), angle.sin());
        });
        // The Nyquist column could not hold the ramp's imaginary part.
        let nyquist = spectrum.at(FrequencyIndex::new(-4, 0));
        assert_eq!(nyquist.im, 0.0);
        let shifted = spectrum.inverse(Radix2).unwrap();
        assert!((0..8).all(|y| (0..8).all(|x| shifted.pixel_at(x, y).0.is_finite())));
    }

    #[test]
    fn centered_puts_the_zero_frequency_in_the_middle() {
        let mut rng = Rng::new(150);
        for (w, h) in SIZES {
            let (values, img) = random(&mut rng, w, h);
            let spectrum = dft(&img, Auto);
            let full = naive_2d(&values, w, h);
            let shown = spectrum.centered();
            assert_eq!(shown.size(), Size::new(w, h));
            let (cx, cy) = (w / 2, h / 2);
            let got: Vec<(f64, f64)> = (0..h)
                .flat_map(|y| (0..w).map(move |x| (x, y)))
                .map(|(x, y)| shown.pixel_at(x, y).to_f64())
                .collect();
            // numpy's fftshift: the bin at column x is (x − ⌊W/2⌋) mod W.
            let want: Vec<(f64, f64)> = (0..h)
                .flat_map(|y| (0..w).map(move |x| (x, y)))
                .map(|(x, y)| full[(y + h - cy) % h * w + (x + w - cx) % w])
                .collect();
            assert!(rel_error(&got, &want) <= 1e-14, "{w}x{h}");
        }
    }

    #[test]
    fn magnitude_power_and_phase_follow_the_centred_bins() {
        let mut rng = Rng::new(160);
        let (_, img) = random(&mut rng, 7, 6);
        let spectrum = dft(&img, Auto);
        let shown = spectrum.centered();
        let (magnitude, power, phase) = (spectrum.magnitude(), spectrum.power(), spectrum.phase());
        for y in 0..6 {
            for x in 0..7 {
                let (re, im) = shown.pixel_at(x, y).to_f64();
                assert_eq!(magnitude.pixel_at(x, y).0, re.hypot(im));
                assert_eq!(power.pixel_at(x, y).0, re * re + im * im);
                assert_eq!(phase.pixel_at(x, y).0, im.atan2(re));
            }
        }
    }

    #[test]
    fn an_empty_spectrum_shows_as_empty_images() {
        for size in [Size::new(0, 0), Size::new(0, 3), Size::new(4, 0)] {
            let mut spectrum = dft(&Image::<MonoF32>::zero(size.width, size.height), Auto);
            spectrum.apply(|_: Frequency<Bins>, _: &mut ComplexF32| unreachable!());
            assert_eq!(spectrum.centered().size(), size);
            assert_eq!(spectrum.magnitude().size(), size);
            assert_eq!(spectrum.power().size(), size);
            assert_eq!(spectrum.phase().size(), size);
        }
    }

    #[test]
    #[should_panic(expected = "without pixels has no frequencies")]
    fn an_empty_spectrum_has_no_bin() {
        let spectrum = dft(&Image::<MonoF32>::zero(0, 3), Auto);
        let _ = spectrum.at(FrequencyIndex::new(0, 0));
    }

    #[test]
    #[should_panic(expected = "without pixels has no frequencies")]
    fn an_empty_spectrum_has_no_unit_to_convert_by() {
        let spectrum = dft(&Image::<MonoF64>::zero(4, 0), Auto);
        let _: Frequency<Bins> = spectrum.convert(Frequency::cycles_per_pixel(0.1, 0.1));
    }
}

//! Phase correlation: the translation between two images, from the phase of
//! their cross-power spectrum.

use super::engine::{AxisTables, Cplx};
use super::method::DftMethod;
use super::spectrum::{Spectrum, SpectrumSource};
use super::units::{CyclesPerPixel, Frequency, canonical};
use crate::analyze::peak::parabola_vertex;
use crate::common::Extremum;
use crate::error::{ParameterError, Requirement, Value};
use crate::geometry::{Pixels, Vector};
use crate::image::{Image, RasterImage};
use crate::{Error, Size};
use std::f64::consts::PI;

/// How many frequency bins of the image's shorter side the radius must
/// exceed. Below about three, the peak is so wide against the image that
/// the window and the wrap at the edges distort it.
const MIN_BINS: f64 = 3.0;

/// The rule a [`CorrelationRadius`] keeps, as its errors report it.
const RADIUS_RANGE: Requirement = Requirement::OpenInterval {
    low: 0.0,
    high: 0.5,
};

/// The radius of the weight phase correlation puts on the frequencies, in
/// cycles per pixel: strictly between 0 and 0.5.
///
/// The weight halves the amplitude at the radius, as the smooth profiles
/// of the frequency filters do, and it decides how accurate
/// [`phase_correlate`] is on a given kind of image. How to choose it is
/// described there. 0.5 cycles per pixel is the highest frequency an image
/// holds, so a value of 0.5 or more, such as a size in pixels written by
/// mistake, is rejected here; a radius too small for a particular image is
/// rejected by the call, which knows the image's size.
///
/// Build one with [`try_new`](Self::try_new), or with
/// [`correlation_radius!`](crate::correlation_radius) for a literal checked
/// at compile time.
///
/// # Example
///
/// ```
/// use fovea::frequency::CorrelationRadius;
///
/// let radius = CorrelationRadius::try_new(0.06)?;
/// assert_eq!(radius.get(), 0.06);
///
/// // A size in pixels is not a radius in cycles per pixel.
/// assert!(CorrelationRadius::try_new(8.0).is_err());
/// # Ok::<(), fovea::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CorrelationRadius {
    value: f64,
}

impl CorrelationRadius {
    /// A radius of `value` cycles per pixel, or `None` unless it lies
    /// strictly between 0 and 0.5.
    #[must_use]
    pub const fn new(value: f64) -> Option<Self> {
        // Written in the positive, so NaN is refused.
        if value > 0.0 && value < 0.5 {
            Some(Self { value })
        } else {
            None
        }
    }

    /// A radius of `value` cycles per pixel, validated.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidParameter`] unless `value` lies strictly between 0
    /// and 0.5.
    pub fn try_new(value: f64) -> Result<Self, Error> {
        Self::new(value)
            .ok_or_else(|| ParameterError::new("radius", RADIUS_RANGE, Value::F64(value)).into())
    }

    /// The radius, in cycles per pixel.
    #[must_use]
    pub const fn get(&self) -> f64 {
        self.value
    }
}

/// A [`CorrelationRadius`] literal, checked at compile time.
///
/// A value that is not a constant expression does not compile
/// (`error[E0435]`); use
/// [`CorrelationRadius::try_new`](crate::frequency::CorrelationRadius::try_new)
/// there.
///
/// # Example
///
/// ```
/// let radius = fovea::correlation_radius!(0.06);
/// assert_eq!(radius.get(), 0.06);
/// ```
///
/// ```compile_fail
/// // ERROR: evaluation panicked: must lie strictly between
/// let _ = fovea::correlation_radius!(8.0);
/// ```
#[macro_export]
macro_rules! correlation_radius {
    ($value:expr) => {
        const {
            $crate::frequency::CorrelationRadius::new($value).expect(
                $crate::error::Requirement::OpenInterval {
                    low: 0.0,
                    high: 0.5,
                }
                .text(),
            )
        }
    };
}

/// What [`phase_correlate`] found: the shift between the two images, and
/// the height of the correlation peak.
///
/// # Example
///
/// ```
/// use fovea::frequency::{Auto, phase_correlate};
/// use fovea::image::Image;
/// use fovea::pixel::MonoF64;
///
/// let img = Image::generate(64, 64, |x, y| MonoF64::new(((x * 7 + y * 13) % 17) as f64));
/// let found = phase_correlate(&img, &img, fovea::correlation_radius!(0.06), Auto)?
///     .expect("the image has structure");
/// assert_eq!((found.shift().x, found.shift().y), (0.0, 0.0));
/// assert!((found.peak() - 1.0).abs() < 1e-12);
/// # Ok::<(), fovea::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhaseCorrelation {
    shift: Vector<Pixels>,
    peak: f64,
}

impl PhaseCorrelation {
    /// Where the content of the moving image lies relative to the
    /// reference: a point `p` of the reference is found at `p + shift` in
    /// the moving image.
    ///
    /// Its whole part lies between −⌊W/2⌋ and ⌈W/2⌉ − 1 along x, likewise
    /// along y, for the size W × H the transform runs at: a shift by more
    /// than half the image cannot be told from its wrap, so +70 px on an
    /// image 128 px wide reads as −58 px.
    #[must_use]
    pub fn shift(&self) -> Vector<Pixels> {
        self.shift
    }

    /// The height of the fitted correlation peak, divided by its height for
    /// two identical images: 1 for a perfect match, lower with noise, with
    /// content that changed, and with a motion other than a translation.
    ///
    /// It does not warn reliably of a radius too large for the images; see
    /// [`phase_correlate`] for how to check one.
    #[must_use]
    pub fn peak(&self) -> f64 {
        self.peak
    }
}

/// A reference image prepared for phase correlation against many images of
/// its size: what [`phase_correlate`] computes from the reference, kept.
///
/// [`locate`](Self::locate) returns what [`phase_correlate`] returns for
/// the same reference, radius and method, and saves the transform of the
/// reference in the first of the two passes on every call. It keeps a copy
/// of the reference, which the second pass windows anew.
///
/// # Example
///
/// ```
/// use fovea::frequency::{Auto, PhaseReference, phase_correlate};
/// use fovea::image::Image;
/// use fovea::pixel::MonoF32;
///
/// let spot = |x: usize, y: usize, cx: f64| {
///     let (dx, dy) = (x as f64 - cx, y as f64 - 30.0);
///     MonoF32::new((-(dx * dx + dy * dy) / 20.0).exp() as f32)
/// };
/// let reference = Image::generate(96, 64, |x, y| spot(x, y, 40.0));
/// let radius = fovea::correlation_radius!(0.06);
/// let prepared = PhaseReference::new(&reference, radius, Auto)?;
///
/// for cx in [41.5, 43.25] {
///     let frame = Image::generate(96, 64, |x, y| spot(x, y, cx));
///     assert_eq!(prepared.locate(&frame)?, phase_correlate(&reference, &frame, radius, Auto)?);
/// }
///
/// // 3 frequency bins of 64 px are 0.047 cycles per pixel, and 0.04 is less.
/// assert!(PhaseReference::new(&reference, fovea::correlation_radius!(0.04), Auto).is_err());
/// # Ok::<(), fovea::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct PhaseReference<P: SpectrumSource> {
    /// The size of the reference, and of every image it is compared with.
    size: Size,
    /// The size the transforms run at: the method's next size.
    transform: Size,
    radius: CorrelationRadius,
    tables: AxisTables<P::Bin>,
    /// The reference, which the second pass windows over the part both
    /// images share.
    reference: Image<P>,
    /// The spectrum of the reference under the whole window, for the first
    /// pass.
    spectrum: Spectrum<P>,
    /// The peak's height for two identical images.
    perfect: f64,
    /// Whether every pixel of the reference is the same.
    flat: bool,
}

impl<P: SpectrumSource> PhaseReference<P> {
    /// The reference `reference` prepared for phase correlation at
    /// `radius`, with transforms by `method`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidParameter`] if `radius` does not exceed 3 frequency
    /// bins of the image's shorter side, 3 / min(W, H) cycles per pixel;
    /// the error's interval names the smallest radius that does. No radius
    /// is large enough for an image with a side under 7 pixels, or none.
    pub fn new<I, M>(reference: &I, radius: CorrelationRadius, method: M) -> Result<Self, Error>
    where
        I: RasterImage<Pixel = P>,
        M: DftMethod<P>,
    {
        let size = reference.size();
        let shorter = size.width.min(size.height);
        let smallest = if shorter == 0 {
            0.5
        } else {
            (MIN_BINS / shorter as f64).min(0.5)
        };
        // The radius is never NaN, so the comparison is total.
        if radius.get() <= smallest {
            return Err(ParameterError::new(
                "radius",
                Requirement::OpenInterval {
                    low: smallest,
                    high: 0.5,
                },
                Value::F64(radius.get()),
            )
            .into());
        }
        let transform = method.fit(size);
        let tables = method
            .tables(transform)
            .expect("a method accepts the size it fits to");
        let reference = Image::from_vec(
            size.width,
            size.height,
            (0..size.height)
                .flat_map(|y| reference.row(y).iter().copied())
                .collect(),
        )
        .expect("the rows hold width * height pixels");
        let whole = (Span::whole(size.width), Span::whole(size.height));
        let spectrum = Spectrum::forward(
            &windowed(&reference, transform, whole),
            &tables.x,
            &tables.y,
        );
        Ok(Self {
            size,
            transform,
            radius,
            tables,
            flat: is_flat(&reference),
            reference,
            spectrum,
            perfect: perfect_height(transform, radius.get()),
        })
    }

    /// The size of the reference, and of every image it is compared with.
    #[must_use]
    pub fn size(&self) -> Size {
        self.size
    }

    /// The radius the reference was prepared at.
    #[must_use]
    pub fn radius(&self) -> CorrelationRadius {
        self.radius
    }

    /// Where the content of `moving` lies relative to the reference, as
    /// [`phase_correlate`] finds it.
    ///
    /// # Errors
    ///
    /// [`Error::SizeMismatch`] if `moving` is not of the reference's size.
    pub fn locate<J>(&self, moving: &J) -> Result<Option<PhaseCorrelation>, Error>
    where
        J: RasterImage<Pixel = P>,
    {
        if moving.size() != self.size {
            return Err(Error::SizeMismatch {
                expected: self.size,
                actual: moving.size(),
            });
        }
        if self.flat || is_flat(moving) {
            return Ok(None);
        }
        // The first pass under the whole window; the second windows both
        // images over the part they share by the first estimate, so the two
        // windowed images show the same content and none is cut off.
        let whole = (Span::whole(self.size.width), Span::whole(self.size.height));
        let Some(((dx, dy), _)) = self.pass(&self.spectrum, moving, whole) else {
            return Ok(None);
        };
        let (rx, mx) = Span::shared(self.size.width, dx);
        let (ry, my) = Span::shared(self.size.height, dy);
        let reference = Spectrum::forward(
            &windowed(&self.reference, self.transform, (rx, ry)),
            &self.tables.x,
            &self.tables.y,
        );
        Ok(self
            .pass(&reference, moving, (mx, my))
            .map(|((x, y), peak)| PhaseCorrelation {
                shift: Vector::new(x, y),
                peak,
            }))
    }

    /// One pass against the windowed reference's spectrum `reference`,
    /// with `moving` under the window `window`: the shift and the peak's
    /// relative height, or `None` if no peak is located.
    fn pass<J>(
        &self,
        reference: &Spectrum<P>,
        moving: &J,
        window: (Span, Span),
    ) -> Option<((f64, f64), f64)>
    where
        J: RasterImage<Pixel = P>,
    {
        let mut cross = Spectrum::forward(
            &windowed(moving, self.transform, window),
            &self.tables.x,
            &self.tables.y,
        );
        cross
            .multiply_conjugate(reference)
            .expect("both spectra are of the transform's size");
        let r = self.radius.get();
        cross.apply(|f: Frequency<CyclesPerPixel>, bin: &mut P::Bin| {
            let (re, im) = bin.to_f64();
            let m = re.hypot(im);
            // A bin without a magnitude has no phase; a non-finite one
            // comes from a non-finite pixel. Neither carries a shift.
            *bin = if m.is_finite() && m > 0.0 {
                let g = gain(f.fx(), r) * gain(f.fy(), r);
                P::Bin::from_f64(re / m * g, im / m * g)
            } else {
                P::Bin::ZERO
            };
        });
        let surface = cross.invert(&self.tables.x, &self.tables.y);
        let (shift, height) = locate_peak(&surface)?;
        Some((shift, height / self.perfect))
    }
}

/// The translation between `reference` and `moving`, two images of one
/// size, by phase correlation: where the content of `moving` lies relative
/// to `reference`.
///
/// A point `p` of the reference is found at `p + shift` in the moving
/// image; see [`PhaseCorrelation::shift`]. The phase of the cross-power
/// spectrum carries the shift and does not depend on the images' contrast,
/// so a change of brightness, an illumination ramp, vignetting or a
/// non-linear response moves the result by less than a hundredth of a
/// pixel in the measurements below.
///
/// # Choosing the radius
///
/// Each frequency's phase is weighted by a Gaussian that halves at
/// `radius` cycles per pixel. The weight has to stay inside the frequencies
/// the images hold: a soft image, defocused or of large smooth shapes, has
/// nothing left at high frequencies but noise and the traces of the frame's
/// edges, and a radius that reaches there fails by a tenth of a pixel and
/// more, with no sign of it in the result. A sharp image holds more, and a
/// larger radius is then more accurate.
///
/// Measured during development on synthetic 128 × 128 images in `f64`: the
/// worst error over sub-pixel shifts without noise, and the root mean
/// square error with noise of 2 % of the pattern's amplitude.
///
/// | Images | radius 0.03 | 0.06 | 0.12 |
/// |---|---|---|---|
/// | sharp | 0.013 / 0.013 px | 0.002 / 0.010 px | 0.001 / 0.015 px |
/// | blurred by a Gaussian of σ = 2 px | 0.013 / 0.016 px | 0.003 / 0.027 px | 0.003 / 0.068 px |
/// | blurred by σ = 4 px | 0.014 / 0.027 px | 0.007 / 0.15 px | 0.26 / 0.47 px |
/// | soft shapes only | 0.008 / 0.015 px | 0.008 / 0.10 px | 0.18 / 0.35 px |
///
/// - **For an unknown image, start at 0.03**, which held on every image
///   measured.
/// - **A rule of thumb:** for edges blurred over σ pixels, a radius of about
///   0.12 / σ, at most 0.12.
/// - **The radius must exceed 3 bins** of the image's shorter side,
///   3 / min(W, H) cycles per pixel, or the call returns an error that names
///   the smallest radius allowed. Small images are hard whatever the radius.
///
/// **To check a radius on your images**, run a typical pair once at the
/// radius chosen and once at 0.03, or at the smallest radius the image
/// allows if that is larger, and compare the two shifts. Where the larger
/// radius reaches past what the images hold, the two differ by about as
/// much as the larger radius errs: in the measurements above, the
/// difference followed the error within 0.013 px where the error was below
/// 0.05 px, and within 20 % above.
///
/// ```
/// # use fovea::frequency::{Auto, phase_correlate};
/// # use fovea::image::Image;
/// # use fovea::pixel::MonoF32;
/// # let spots = |x: f64, y: f64| (-((x - 60.0).powi(2) + (y - 50.0).powi(2)) / 18.0).exp() as f32;
/// # let reference = Image::generate(128, 128, |x, y| MonoF32::new(spots(x as f64, y as f64)));
/// # let sample = Image::generate(128, 128, |x, y| MonoF32::new(spots(x as f64 - 1.5, y as f64)));
/// let chosen = phase_correlate(&reference, &sample, fovea::correlation_radius!(0.12), Auto)?;
/// let safe = phase_correlate(&reference, &sample, fovea::correlation_radius!(0.03), Auto)?;
/// if let (Some(chosen), Some(safe)) = (chosen, safe) {
///     let apart = (chosen.shift() - safe.shift()).length().get();
///     // An `apart` larger than the accuracy needed says that 0.12 reaches
///     // past what these images hold.
/// #   let _ = apart;
/// }
/// # Ok::<(), fovea::Error>(())
/// ```
///
/// # What happens
///
/// Each image less its mean is multiplied by a Hann window, which takes it
/// to zero at its edges; without a window, the frame's edges are a
/// structure that does not move, and on an image the content fills, their
/// peak at zero shift wins. The cross-power spectrum, the moving image's
/// spectrum times the conjugate of the reference's, is reduced to its phase
/// and weighted; its inverse is a peak at the shift, shaped as a Gaussian,
/// whose vertex a three-point fit per axis locates between the pixels. A
/// window that stays in place pulls the result towards zero shift, by more
/// the larger the shift, so a second pass windows both images over the part
/// they share by the first estimate, the moving image's window moved by it,
/// and repeats. The transforms run at `method`'s next size, padded with
/// zeros beyond the windowed images, which are zero at their edges already.
/// Measured on a development machine in a release build, one call takes
/// about 150 ms on 1024 × 1024 `MonoF32` images by
/// [`Radix2`](super::Radix2).
///
/// To compare many images with one reference, prepare it once with
/// [`PhaseReference`].
///
/// # Accuracy
///
/// No bound is promised. Besides the radius, the accuracy depends on:
///
/// - **Noise:** phase correlation weights a frequency by its phase alone,
///   not by its strength, and has two to three times the noise of a plain
///   cross-correlation.
/// - **Defocus between the two images** costs about 0.01 px.
/// - **A motion other than a translation** has no single shift. Under a
///   rotation or a change of scale each pixel moves differently, and the
///   result is an average of the motion that depends on the content: in
///   the measurements, 0.01 to 0.06 px from the motion at the image's centre
///   at a rotation of 0.25° to 1°, and 0.02 to 0.08 px at a scale of 1.0025
///   to 1.01, at a radius of 0.06. The peak's height falls a little with it.
/// - **A large shift** leaves less of the images to compare: at radius 0.06
///   on 128 × 128 pixels, 0.009 px at a quarter of the size, 0.02 px at
///   three eighths, and 0.05 to 0.08 px close to half.
///
/// # Returns
///
/// `Some` with the shift and the peak's height, or `None` where no peak is
/// located: an image whose pixels are all the same, a NaN or an infinity
/// in either image, or a peak whose three-point fit is refused.
///
/// # Errors
///
/// - [`Error::SizeMismatch`] if the two images are not of one size.
/// - [`Error::InvalidParameter`] if `radius` does not exceed 3 frequency
///   bins of the shorter side, as [`PhaseReference::new`] reports it.
///
/// # Example
///
/// ```
/// use fovea::frequency::{Auto, phase_correlate};
/// use fovea::image::Image;
/// use fovea::pixel::MonoF64;
///
/// // Three soft spots, sampled once in place and once moved by (2.3, −1.6).
/// let pattern = |x: f64, y: f64| {
///     let spot = |cx: f64, cy: f64, s: f64| (-((x - cx).powi(2) + (y - cy).powi(2)) / (2.0 * s * s)).exp();
///     spot(50.0, 60.0, 4.0) - 0.7 * spot(75.0, 52.0, 6.0) + 0.5 * spot(60.0, 80.0, 3.0)
/// };
/// let reference = Image::generate(128, 128, |x, y| MonoF64::new(pattern(x as f64, y as f64)));
/// let moving = Image::generate(128, 128, |x, y| MonoF64::new(pattern(x as f64 - 2.3, y as f64 + 1.6)));
///
/// let found = phase_correlate(&reference, &moving, fovea::correlation_radius!(0.06), Auto)?
///     .expect("both images have structure");
/// assert!((found.shift().x - 2.3).abs() < 0.01);
/// assert!((found.shift().y + 1.6).abs() < 0.01);
/// # Ok::<(), fovea::Error>(())
/// ```
pub fn phase_correlate<I, J, P, M>(
    reference: &I,
    moving: &J,
    radius: CorrelationRadius,
    method: M,
) -> Result<Option<PhaseCorrelation>, Error>
where
    I: RasterImage<Pixel = P>,
    J: RasterImage<Pixel = P>,
    P: SpectrumSource,
    M: DftMethod<P>,
{
    if moving.size() != reference.size() {
        return Err(Error::SizeMismatch {
            expected: reference.size(),
            actual: moving.size(),
        });
    }
    PhaseReference::new(reference, radius, method)?.locate(moving)
}

/// The Gaussian that halves at `r`, at the frequency `f` along one axis.
fn gain(f: f64, r: f64) -> f64 {
    (-(f / r) * (f / r)).exp2()
}

/// The value of the inverse transform at zero for two identical images:
/// the mean of the weight over every bin of the transform.
fn perfect_height(transform: Size, r: f64) -> f64 {
    let axis = |n: usize| -> f64 {
        (0..n)
            .map(|k| gain(canonical(k as isize, n) as f64 / n as f64, r))
            .sum::<f64>()
            / n as f64
    };
    axis(transform.width) * axis(transform.height)
}

/// Where a Hann window lies along one axis: from `start` over `len`
/// samples' distance, both in pixels and not necessarily whole.
#[derive(Clone, Copy, Debug)]
struct Span {
    start: f64,
    len: f64,
}

impl Span {
    /// The window over a whole side of `n` pixels.
    fn whole(n: usize) -> Self {
        Self {
            start: 0.0,
            len: (n - 1) as f64,
        }
    }

    /// The windows of the reference and of the moving image over the part
    /// of a side of `n` pixels both show, when the content has moved by
    /// `d`: a reference pixel at `t` is the moving image's at `t + d`.
    fn shared(n: usize, d: f64) -> (Self, Self) {
        let len = (n - 1) as f64 - d.abs();
        let start = (-d).max(0.0);
        (
            Self { start, len },
            Self {
                start: start + d,
                len,
            },
        )
    }

    /// The window's value at pixel `t`: zero outside it.
    fn at(self, t: f64) -> f64 {
        let u = t - self.start;
        if (0.0..=self.len).contains(&u) {
            0.5 - 0.5 * (2.0 * PI * u / self.len).cos()
        } else {
            0.0
        }
    }
}

/// `img` less its mean under the Hann window over `window`, times that
/// window, padded with zeros to `transform`.
///
/// Without the mean, a constant brightness would become a copy of the
/// window, a structure that does not move with the content.
fn windowed<I, P>(img: &I, transform: Size, window: (Span, Span)) -> Image<P>
where
    I: RasterImage<Pixel = P>,
    P: SpectrumSource,
{
    let Size { width, height } = img.size();
    let wx: Vec<f64> = (0..width).map(|x| window.0.at(x as f64)).collect();
    let wy: Vec<f64> = (0..height).map(|y| window.1.at(y as f64)).collect();
    let (mut sum, mut weight) = (0.0, 0.0);
    for (y, &fy) in wy.iter().enumerate() {
        for (&p, &fx) in img.row(y).iter().zip(&wx) {
            sum += p.to_f64() * fx * fy;
            weight += fx * fy;
        }
    }
    let mean = if weight > 0.0 { sum / weight } else { 0.0 };
    let zero = P::from_f64(0.0);
    let mut pixels = vec![zero; transform.area()];
    for (y, &fy) in wy.iter().enumerate() {
        let row = img.row(y);
        let out = &mut pixels[y * transform.width..y * transform.width + width];
        for ((o, &p), &fx) in out.iter_mut().zip(row).zip(&wx) {
            *o = P::from_f64((p.to_f64() - mean) * fx * fy);
        }
    }
    Image::from_vec(transform.width, transform.height, pixels)
        .expect("the buffer has the transform's size")
}

/// Whether every pixel of `img` is the same. A NaN differs from itself, so
/// an image with one is not flat, and its NaN surfaces as `None` later.
fn is_flat<I, P>(img: &I) -> bool
where
    I: RasterImage<Pixel = P>,
    P: SpectrumSource,
{
    let Size { width, height } = img.size();
    if width == 0 || height == 0 {
        return true;
    }
    let first = img.row(0)[0].to_f64();
    (0..height).all(|y| img.row(y).iter().all(|p| p.to_f64() == first))
}

/// The shift at the vertex of the surface's peak, in the canonical range,
/// and the vertex's height; `None` if the peak is not positive or its fit
/// is refused.
///
/// The largest sample is the earliest of several equal ones; a NaN never
/// is. Along each axis, the vertex of the parabola through the logarithms
/// of the largest sample and its two neighbours, which the cyclic surface
/// always has, is the vertex of the Gaussian through the three samples.
fn locate_peak<P: SpectrumSource>(surface: &Image<P>) -> Option<((f64, f64), f64)> {
    let Size { width, height } = crate::image::ImageView::size(surface);
    let value = |x: usize, y: usize| surface.row(y)[x].to_f64();
    let mut best: Option<(usize, usize, f64)> = None;
    for y in 0..height {
        for x in 0..width {
            let v = value(x, y);
            if best.is_none_or(|(_, _, b)| v > b) && !v.is_nan() {
                best = Some((x, y, v));
            }
        }
    }
    let (px, py, top) = best?;
    // The largest sample is never NaN, so the comparison is total.
    if top <= 0.0 {
        return None;
    }
    let along = |before: f64, after: f64| -> Option<(f64, f64)> {
        if !(before > 0.0 && after > 0.0) {
            return None;
        }
        let (a, b, c) = (before.ln(), top.ln(), after.ln());
        let offset = parabola_vertex(a, b, c, Extremum::Maximum)?;
        // The parabola's rise from the middle sample to its vertex.
        let rise = (c - a) * (c - a) / (-8.0 * (a + c - 2.0 * b));
        Some((offset, rise))
    };
    let (left, right) = ((px + width - 1) % width, (px + 1) % width);
    let (up, down) = ((py + height - 1) % height, (py + 1) % height);
    let (ox, rx) = along(value(left, py), value(right, py))?;
    let (oy, ry) = along(value(px, up), value(px, down))?;
    let x = canonical(px as isize, width) as f64 + ox;
    let y = canonical(py as isize, height) as f64 + oy;
    Some(((x, y), (top.ln() + rx + ry).exp()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Point;
    use crate::frequency::testing::Rng;
    use crate::frequency::{Auto, Bluestein, Radix2};
    use crate::image::ImageView;
    use crate::pixel::{MonoF32, MonoF64};

    /// A pattern of soft spots and soft-edged bars, integrated over each
    /// pixel's area, with its content moved by `(dx, dy)`: the pair a test
    /// builds itself, with a shift known exactly.
    #[derive(Clone)]
    struct Pattern {
        spots: Vec<(f64, f64, f64, f64)>,
        bars: Vec<(f64, f64, f64, f64, f64)>,
    }

    /// Abramowitz and Stegun 7.1.26, absolute error below 1.5·10⁻⁷, which
    /// both images of a pair share.
    fn erf(x: f64) -> f64 {
        let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
        let poly = t
            * (0.254_829_592
                + t * (-0.284_496_736
                    + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
        x.signum() * (1.0 - poly * (-x * x).exp())
    }

    fn cdf(u: f64) -> f64 {
        0.5 * (1.0 + erf(u / std::f64::consts::SQRT_2))
    }

    /// The antiderivative of the normal cdf.
    fn cdf_integral(u: f64) -> f64 {
        u * cdf(u) + (-0.5 * u * u).exp() / (2.0 * PI).sqrt()
    }

    impl Pattern {
        fn new(rng: &mut Rng, size: f64, spots: usize, bars: usize, sigma: (f64, f64)) -> Self {
            let mut at = |lo: f64, hi: f64| lo + (hi - lo) * 0.5 * (rng.value() + 1.0);
            let spots = (0..spots)
                .map(|_| {
                    let (x, y) = (at(-16.0, size + 16.0), at(-16.0, size + 16.0));
                    let s = at(sigma.0, sigma.1);
                    let a = at(-1.0, 1.0);
                    (x, y, s, a)
                })
                .collect();
            let bars = (0..bars)
                .map(|_| {
                    let (x, y) = (at(-16.0, size), at(-16.0, size));
                    let (w, h) = (at(3.0, 25.0), at(3.0, 25.0));
                    (x, y, x + w, y + h, at(-1.0, 1.0))
                })
                .collect();
            Self { spots, bars }
        }

        /// The mean over the pixel `[i − ½, i + ½]` of exp(−(t − c)²/2s²).
        fn spot_axis(i: f64, c: f64, s: f64) -> f64 {
            let k = s * std::f64::consts::SQRT_2;
            s * (PI / 2.0).sqrt() * (erf((i + 0.5 - c) / k) - erf((i - 0.5 - c) / k))
        }

        /// The mean over the pixel of a box `[lo, hi]` blurred by 0.5 px.
        fn bar_axis(i: f64, lo: f64, hi: f64) -> f64 {
            let s = 0.5;
            let edge =
                |e: f64| s * (cdf_integral((i + 0.5 - e) / s) - cdf_integral((i - 0.5 - e) / s));
            edge(lo) - edge(hi)
        }

        fn render<P: SpectrumSource>(&self, w: usize, h: usize, dx: f64, dy: f64) -> Image<P> {
            let mut v = vec![0.0; w * h];
            let mut add = |xs: Vec<f64>, ys: Vec<f64>, a: f64| {
                for (y, fy) in ys.iter().enumerate() {
                    for (x, fx) in xs.iter().enumerate() {
                        v[y * w + x] += a * fx * fy;
                    }
                }
            };
            for &(cx, cy, s, a) in &self.spots {
                let xs = (0..w)
                    .map(|i| Self::spot_axis(i as f64, cx + dx, s))
                    .collect();
                let ys = (0..h)
                    .map(|j| Self::spot_axis(j as f64, cy + dy, s))
                    .collect();
                add(xs, ys, a);
            }
            for &(x0, y0, x1, y1, a) in &self.bars {
                let xs = (0..w)
                    .map(|i| Self::bar_axis(i as f64, x0 + dx, x1 + dx))
                    .collect();
                let ys = (0..h)
                    .map(|j| Self::bar_axis(j as f64, y0 + dy, y1 + dy))
                    .collect();
                add(xs, ys, a);
            }
            Image::from_vec(w, h, v.into_iter().map(P::from_f64).collect()).unwrap()
        }
    }

    fn sharp(rng: &mut Rng, n: usize) -> Pattern {
        Pattern::new(rng, n as f64, n * n / 200, n * n / 1000, (0.7, 6.0))
    }

    fn found<P: SpectrumSource>(a: &Image<P>, b: &Image<P>, r: f64) -> PhaseCorrelation {
        phase_correlate(a, b, CorrelationRadius::try_new(r).unwrap(), Auto)
            .unwrap()
            .expect("the pattern has a peak")
    }

    #[test]
    fn it_recovers_a_known_sub_pixel_shift() {
        let mut rng = Rng::new(1310);
        let pattern = sharp(&mut rng, 128);
        let reference: Image<MonoF64> = pattern.render(128, 128, 0.0, 0.0);
        let mut worst: f64 = 0.0;
        for i in 0..10 {
            let (dx, dy) = (3.0 + 0.1 * i as f64, -1.63);
            let moving = pattern.render(128, 128, dx, dy);
            let shift = found(&reference, &moving, 0.06).shift();
            worst = worst.max((shift.x - dx).abs()).max((shift.y - dy).abs());
        }
        // Measured 0.0013 px when this test was written.
        assert!(worst < 0.003, "{worst}");
    }

    #[test]
    fn single_precision_recovers_it_too() {
        let mut rng = Rng::new(1320);
        let pattern = sharp(&mut rng, 128);
        let reference: Image<MonoF32> = pattern.render(128, 128, 0.0, 0.0);
        let moving = pattern.render(128, 128, -4.35, 2.8);
        let shift = found(&reference, &moving, 0.06).shift();
        assert!((shift.x + 4.35).abs() < 0.005, "{shift:?}");
        assert!((shift.y - 2.8).abs() < 0.005, "{shift:?}");
    }

    #[test]
    fn a_point_of_the_reference_lies_at_p_plus_the_shift() {
        let mut rng = Rng::new(1330);
        let pattern = sharp(&mut rng, 96);
        let reference: Image<MonoF64> = pattern.render(96, 96, 0.0, 0.0);
        // The content moves right by 5 and up by 2.
        let moving = pattern.render(96, 96, 5.0, -2.0);
        let shift = found(&reference, &moving, 0.06).shift();
        let p: Point<Pixels> = Point::new(30.0, 40.0);
        let q = p + shift;
        assert!(
            (q.x - 35.0).abs() < 0.01 && (q.y - 38.0).abs() < 0.01,
            "{q:?}"
        );
    }

    #[test]
    fn every_method_finds_the_same_shift() {
        let mut rng = Rng::new(1340);
        let pattern = sharp(&mut rng, 100);
        let reference: Image<MonoF64> = pattern.render(100, 100, 0.0, 0.0);
        let moving = pattern.render(100, 100, 1.4, 2.25);
        let radius = correlation_radius!(0.06);
        let auto = phase_correlate(&reference, &moving, radius, Auto)
            .unwrap()
            .unwrap();
        let bluestein = phase_correlate(&reference, &moving, radius, Bluestein)
            .unwrap()
            .unwrap();
        // Radix2 pads 100 × 100 to 128 × 128 with zeros.
        let radix2 = phase_correlate(&reference, &moving, radius, Radix2)
            .unwrap()
            .unwrap();
        for other in [bluestein, radix2] {
            let d = (other.shift() - auto.shift()).length().get();
            assert!(d < 0.005, "{d}");
        }
        assert!((auto.shift().x - 1.4).abs() < 0.005 && (auto.shift().y - 2.25).abs() < 0.005);
    }

    #[test]
    fn brightness_and_contrast_barely_move_the_result() {
        let mut rng = Rng::new(1350);
        let pattern = sharp(&mut rng, 128);
        let reference: Image<MonoF64> = pattern.render(128, 128, 0.0, 0.0);
        let moving: Image<MonoF64> = pattern.render(128, 128, 2.6, 0.4);
        let changed = Image::generate(128, 128, |x, y| {
            // Gain, offset, and a ramp across the image.
            MonoF64::new(1.5 * moving.pixel_at(x, y).0 + 0.5 + x as f64 / 128.0)
        });
        let plain = found(&reference, &moving, 0.06).shift();
        let lit = found(&reference, &changed, 0.06).shift();
        assert!((lit - plain).length().get() < 0.005, "{plain:?} {lit:?}");
    }

    #[test]
    fn identical_images_have_no_shift_and_a_peak_of_one() {
        let mut rng = Rng::new(1360);
        let img: Image<MonoF64> = sharp(&mut rng, 64).render(64, 64, 0.0, 0.0);
        let same = found(&img, &img, 0.12);
        assert_eq!((same.shift().x, same.shift().y), (0.0, 0.0));
        assert!((same.peak() - 1.0).abs() < 1e-12, "{}", same.peak());
    }

    #[test]
    fn a_shift_by_more_than_half_the_image_reads_as_its_wrap() {
        // Spots repeated with the image's period, moved by 70.4 px along x.
        let n = 128;
        let mut rng = Rng::new(1390);
        let mut spots = sharp(&mut rng, n);
        spots.bars.clear();
        let period = n as f64;
        spots.spots = spots
            .spots
            .iter()
            .flat_map(|&(x, y, s, a)| {
                (-1..=1).flat_map(move |i| {
                    (-1..=1).map(move |j| (x + i as f64 * period, y + j as f64 * period, s, a))
                })
            })
            .collect();
        let reference: Image<MonoF64> = spots.render(n, n, 0.0, 0.0);
        let moving = spots.render(n, n, 70.4, 0.0);
        let shift = found(&reference, &moving, 0.06).shift();
        assert!((shift.x - (70.4 - period)).abs() < 0.05, "{shift:?}");
        assert!(shift.y.abs() < 0.05, "{shift:?}");
    }

    #[test]
    fn a_prepared_reference_gives_what_the_function_gives() {
        let mut rng = Rng::new(1370);
        let pattern = sharp(&mut rng, 80);
        let reference: Image<MonoF32> = pattern.render(80, 80, 0.0, 0.0);
        let radius = correlation_radius!(0.05);
        let prepared = PhaseReference::new(&reference, radius, Radix2).unwrap();
        assert_eq!(prepared.size(), Size::new(80, 80));
        assert_eq!(prepared.radius(), radius);
        for (dx, dy) in [(0.3, 0.1), (-2.7, 4.2), (6.5, -6.5)] {
            let moving = pattern.render(80, 80, dx, dy);
            assert_eq!(
                prepared.locate(&moving).unwrap(),
                phase_correlate(&reference, &moving, radius, Radix2).unwrap()
            );
        }
    }

    #[test]
    fn images_of_different_sizes_are_an_error() {
        let a = Image::fill(64, 64, MonoF32::new(1.0));
        let b = Image::fill(64, 63, MonoF32::new(1.0));
        let mismatch = Error::SizeMismatch {
            expected: Size::new(64, 64),
            actual: Size::new(64, 63),
        };
        let radius = correlation_radius!(0.1);
        assert_eq!(phase_correlate(&a, &b, radius, Auto), Err(mismatch.clone()));
        let prepared = PhaseReference::new(&a, radius, Auto).unwrap();
        assert_eq!(prepared.locate(&b), Err(mismatch));
    }

    #[test]
    fn a_radius_too_small_for_the_image_names_the_smallest() {
        let img = Image::fill(128, 200, MonoF64::new(1.0));
        let err = phase_correlate(&img, &img, correlation_radius!(0.001), Auto).unwrap_err();
        let want: Error = ParameterError::new(
            "radius",
            Requirement::OpenInterval {
                low: 3.0 / 128.0,
                high: 0.5,
            },
            Value::F64(0.001),
        )
        .into();
        assert_eq!(err, want);
        // 0.03 exceeds 3 bins of 128 px.
        assert!(phase_correlate(&img, &img, correlation_radius!(0.03), Auto).is_ok());
    }

    #[test]
    fn no_radius_fits_an_image_under_seven_pixels() {
        for size in [Size::new(6, 40), Size::new(40, 0), Size::new(0, 0)] {
            let img = Image::fill(size.width, size.height, MonoF32::new(1.0));
            let err = PhaseReference::new(&img, correlation_radius!(0.49), Auto).unwrap_err();
            let Error::InvalidParameter(e) = err else {
                panic!("{err:?}")
            };
            assert_eq!(
                e.requirement(),
                Requirement::OpenInterval {
                    low: 0.5,
                    high: 0.5
                }
            );
        }
        let seven = Image::fill(7, 7, MonoF32::new(1.0));
        assert!(PhaseReference::new(&seven, correlation_radius!(0.49), Auto).is_ok());
    }

    #[test]
    fn a_flat_image_or_a_nan_gives_no_peak() {
        let mut rng = Rng::new(1380);
        let img: Image<MonoF64> = sharp(&mut rng, 64).render(64, 64, 0.0, 0.0);
        let flat = Image::fill(64, 64, MonoF64::new(0.1));
        let radius = correlation_radius!(0.1);
        assert_eq!(phase_correlate(&img, &flat, radius, Auto), Ok(None));
        assert_eq!(phase_correlate(&flat, &img, radius, Auto), Ok(None));
        for bad in [f64::NAN, f64::INFINITY] {
            let mut broken = img.clone();
            *crate::image::ImageViewMut::pixel_at_mut(&mut broken, 20, 30) = MonoF64::new(bad);
            assert_eq!(phase_correlate(&img, &broken, radius, Auto), Ok(None));
            assert_eq!(phase_correlate(&broken, &img, radius, Auto), Ok(None));
        }
    }

    #[test]
    fn the_radius_keeps_to_its_range() {
        for bad in [0.0, -0.1, 0.5, 8.0, f64::NAN, f64::INFINITY] {
            assert_eq!(CorrelationRadius::new(bad), None, "{bad}");
            assert_eq!(
                CorrelationRadius::try_new(bad).unwrap_err(),
                ParameterError::new("radius", RADIUS_RANGE, Value::F64(bad)).into(),
            );
        }
        for good in [1e-9, 0.03, 0.12, 0.499_999] {
            assert_eq!(CorrelationRadius::new(good).map(|r| r.get()), Some(good));
        }
        const R: CorrelationRadius = correlation_radius!(0.25);
        assert_eq!(R.get(), 0.25);
    }

    #[test]
    fn the_peak_fit_refuses_what_is_not_a_peak() {
        // A surface whose largest sample has a non-positive neighbour.
        let mut surface = Image::fill(8, 8, MonoF64::new(0.1));
        *crate::image::ImageViewMut::pixel_at_mut(&mut surface, 3, 3) = MonoF64::new(1.0);
        *crate::image::ImageViewMut::pixel_at_mut(&mut surface, 4, 3) = MonoF64::new(-0.2);
        assert_eq!(locate_peak(&surface), None);
        // Nor a surface that is nowhere positive.
        assert_eq!(locate_peak(&Image::fill(8, 8, MonoF64::new(-1.0))), None);
        // A sampled Gaussian is located exactly, its height too.
        let gaussian = Image::generate(16, 16, |x, y| {
            let (dx, dy) = (x as f64 - 5.3, y as f64 - 9.8);
            MonoF64::new(2.0 * (-(dx * dx + dy * dy) / 4.5).exp())
        });
        let ((x, y), height) = locate_peak(&gaussian).unwrap();
        assert!(
            (x - 5.3).abs() < 1e-12 && (y - (9.8 - 16.0)).abs() < 1e-12,
            "{x} {y}"
        );
        assert!((height - 2.0).abs() < 1e-12, "{height}");
    }
}

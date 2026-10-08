//! Interpolation kernels: the weights that resize, point sampling and the
//! remap share.
//!
//! A kernel is a one-dimensional weight function with a radius known at
//! compile time. Every kernel here is separable, so the weight of a sample at
//! offset `(dx, dy)` from the position being evaluated is
//! `weight(dx) · weight(dy)`.
//!
//! | Kernel | Taps per axis | Character |
//! |---|---|---|
//! | [`Bilinear`] | 2 | tent; no overshoot, soft |
//! | [`CatmullRom`] | 4 | Keys cubic with `a = −0.5`; sharp, reproduces linear ramps exactly |
//! | [`KeysBicubic`] | 4 | Keys cubic with a chosen `a`; `−0.75` matches OpenCV |
//! | [`Lanczos2`](type@Lanczos2), [`Lanczos3`](type@Lanczos3) | 4, 6 | windowed sinc; sharpest, rings near steps |
//!
//! There is no bare `Bicubic`: libraries disagree on its coefficient
//! (OpenCV uses `a = −0.75`, Pillow `a = −0.5`), so the same name would give
//! different pixels. Name the coefficient instead.
//!
//! A kernel on its own **interpolates**: its width is fixed, in a resize as in
//! [`sample`](crate::analyze::sampling::sample). When a resize shrinks an
//! image, a fixed kernel reads only the few source pixels nearest to each
//! target pixel, and detail finer than the target grid comes back as a false
//! coarser pattern (aliasing). [`Antialiased`] is the resize method that
//! widens the kernel by the shrink factor instead.

use crate::error::{Error, ParameterError, Requirement, Value};

/// A separable interpolation kernel: a weight as a function of the signed
/// distance from the position being evaluated.
///
/// A position `t` on one axis reads the `2 · RADIUS` samples from
/// `floor(t) − RADIUS + 1` to `floor(t) + RADIUS`, with weights
/// `weight(k − t)`. The engines normalise the weights of each position to sum
/// to one, so a flat image stays flat even for a kernel whose weights do not
/// sum to one exactly, such as Lanczos.
///
/// `RADIUS` is a **contract**: the weight must be zero for `|d| >= RADIUS`,
/// and no engine reads further. A tap whose weight is exactly zero is not
/// read at all, so a kernel that interpolates (weight 1 at 0 and 0 at every
/// other integer) returns a pixel unchanged when evaluated at its centre,
/// even at the last pixel of a row.
///
/// Not sealed. A kernel of your own implements this trait and then works with
/// [`resize`](crate::transform::resize), [`Antialiased`] and
/// [`sample`](crate::analyze::sampling::sample):
///
/// ```
/// use fovea::transform::InterpolationKernel;
///
/// /// Mitchell-Netravali with B = C = 1/3.
/// #[derive(Clone, Copy)]
/// struct Mitchell;
///
/// impl InterpolationKernel for Mitchell {
///     const RADIUS: usize = 2;
///     fn weight(&self, d: f64) -> f64 {
///         let (b, c) = (1.0 / 3.0, 1.0 / 3.0);
///         let x = d.abs();
///         let w = if x < 1.0 {
///             (12.0 - 9.0 * b - 6.0 * c) * x.powi(3) + (-18.0 + 12.0 * b + 6.0 * c) * x * x
///                 + (6.0 - 2.0 * b)
///         } else if x < 2.0 {
///             (-b - 6.0 * c) * x.powi(3) + (6.0 * b + 30.0 * c) * x * x
///                 + (-12.0 * b - 48.0 * c) * x + (8.0 * b + 24.0 * c)
///         } else {
///             0.0
///         };
///         w / 6.0
///     }
/// }
///
/// assert_eq!(Mitchell.weight(2.0), 0.0);
/// ```
pub trait InterpolationKernel: Copy {
    /// Taps on each side of the position: the weight is zero for
    /// `|d| >= RADIUS`.
    const RADIUS: usize;

    /// The weight of a sample at signed distance `d`, in pixels, from the
    /// position being evaluated.
    fn weight(&self, d: f64) -> f64;
}

/// The largest [`InterpolationKernel::RADIUS`] the point sampler accepts:
/// 8 taps on each side.
///
/// The point sampler keeps its weights on the stack, and stable Rust cannot
/// size an array by a kernel's associated constant, so the buffer has this
/// fixed size. A kernel with a larger radius fails to build with
/// [`sample`](crate::analyze::sampling::sample). Resize precomputes its
/// weights on the heap and has no such limit.
pub const MAX_SAMPLE_RADIUS: usize = 8;

// ─── Bilinear ────────────────────────────────────────────────────────────────

/// Bilinear interpolation: the tent `1 − |d|`, two taps per axis.
///
/// The softest of the kernels here and the only one that never overshoots:
/// every result lies between the samples it was computed from.
///
/// As a resize method it interpolates at a fixed width; when shrinking by more
/// than about two, use [`Antialiased(Bilinear)`](Antialiased) instead.
///
/// Types that represent gamma-encoded data (e.g. [`Srgb8`](crate::pixel::Srgb8),
/// [`Srgba8`](crate::pixel::Srgba8)) intentionally do *not* implement
/// [`LinearSpace`](crate::pixel::LinearSpace) and will be rejected at compile
/// time. Convert to linear light first (e.g. via
/// [`SrgbGamma`](crate::transform::SrgbGamma)) before resizing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Bilinear;

impl InterpolationKernel for Bilinear {
    const RADIUS: usize = 1;

    #[inline]
    fn weight(&self, d: f64) -> f64 {
        let x = d.abs();
        if x < 1.0 { 1.0 - x } else { 0.0 }
    }
}

// ─── Keys cubic ──────────────────────────────────────────────────────────────

/// Keys' cubic convolution kernel with coefficient `a`, for `|x| < 2`.
#[inline]
fn keys(a: f64, d: f64) -> f64 {
    let x = d.abs();
    if x < 1.0 {
        ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a
    } else {
        0.0
    }
}

/// The Catmull-Rom spline: Keys' cubic with `a = −0.5`, four taps per axis.
///
/// The one Keys coefficient of third-order accuracy, and the bicubic Pillow
/// uses. It reproduces linear ramps exactly and is sharper than [`Bilinear`],
/// at the cost of a slight overshoot next to a step.
///
/// # Example
///
/// ```
/// use fovea::transform::{CatmullRom, InterpolationKernel};
///
/// assert_eq!(CatmullRom.weight(0.0), 1.0);
/// assert_eq!(CatmullRom.weight(1.0), 0.0);
/// assert_eq!(CatmullRom.weight(0.5), 0.5625);
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CatmullRom;

impl InterpolationKernel for CatmullRom {
    const RADIUS: usize = 2;

    #[inline]
    fn weight(&self, d: f64) -> f64 {
        keys(-0.5, d)
    }
}

/// Keys' cubic convolution kernel with a chosen coefficient `a`, four taps
/// per axis.
///
/// Every finite `a` gives an interpolating kernel whose weights sum to one;
/// `a` only trades sharpness against overshoot. `KeysBicubic::new(-0.75)` is
/// the bicubic of OpenCV; `a = −0.5` is [`CatmullRom`], which has its own
/// name because it is the standard choice.
///
/// - Literals use [`keys_bicubic!`](crate::keys_bicubic), checked at compile
///   time.
/// - Coefficients computed at run time use [`KeysBicubic::try_new`].
///
/// # Example
///
/// ```
/// use fovea::keys_bicubic;
/// use fovea::transform::{InterpolationKernel, KeysBicubic};
///
/// const OPENCV: KeysBicubic = keys_bicubic!(-0.75);
/// assert_eq!(OPENCV.a(), -0.75);
/// assert_eq!(OPENCV.weight(1.0), 0.0);
///
/// assert!(KeysBicubic::try_new(f64::NAN).is_err());
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KeysBicubic {
    a: f64,
}

impl KeysBicubic {
    /// Creates the kernel, returning `None` if `a` is NaN or infinite.
    ///
    /// This is the `const fn` the [`keys_bicubic!`](crate::keys_bicubic)
    /// macro wraps. Prefer the macro for literals and [`Self::try_new`] for
    /// computed values.
    #[must_use]
    pub const fn new(a: f64) -> Option<Self> {
        if a.is_finite() {
            Some(Self { a })
        } else {
            None
        }
    }

    /// Creates the kernel from a computed coefficient, validating it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidParameter`] with [`Requirement::Finite`] if `a`
    /// is NaN or infinite.
    pub fn try_new(a: f64) -> Result<Self, Error> {
        Self::new(a)
            .ok_or_else(|| ParameterError::new("Keys a", Requirement::Finite, Value::F64(a)).into())
    }

    /// The coefficient `a`.
    #[must_use]
    pub const fn a(self) -> f64 {
        self.a
    }
}

impl InterpolationKernel for KeysBicubic {
    const RADIUS: usize = 2;

    #[inline]
    fn weight(&self, d: f64) -> f64 {
        keys(self.a, d)
    }
}

/// A [`KeysBicubic`] literal, checked at compile time.
///
/// Expands to an inline `const` block, so a non-finite literal is a compile
/// error and a runtime value does not type-check. For those, use
/// [`KeysBicubic::try_new`].
///
/// ```
/// use fovea::keys_bicubic;
///
/// let opencv = keys_bicubic!(-0.75);
/// assert_eq!(opencv.a(), -0.75);
/// ```
///
/// ```compile_fail
/// use fovea::keys_bicubic;
/// // ERROR: evaluation panicked: must be finite
/// let _ = keys_bicubic!(f64::NAN);
/// ```
#[macro_export]
macro_rules! keys_bicubic {
    ($a:expr) => {
        const {
            $crate::transform::KeysBicubic::new($a)
                .expect($crate::error::Requirement::Finite.text())
        }
    };
}

// ─── Lanczos ─────────────────────────────────────────────────────────────────

/// The Lanczos kernel with `A` lobes: `sinc(d) · sinc(d / A)` for `|d| < A`,
/// `2 · A` taps per axis.
///
/// The sharpest kernel here, and the one closest to ideal band-limited
/// interpolation; it rings next to a step. `A` is a const parameter, so the
/// width is known at compile time; `A = 0` fails to build. Use the aliases
/// [`Lanczos2`](type@Lanczos2) and [`Lanczos3`](type@Lanczos3), which are both a type and a value.
///
/// # Example
///
/// ```
/// use fovea::transform::{InterpolationKernel, Lanczos3};
///
/// assert_eq!(Lanczos3.weight(0.0), 1.0);
/// assert_eq!(Lanczos3.weight(2.0), 0.0);
/// assert_eq!(Lanczos3.weight(3.0), 0.0);
/// ```
///
/// ```compile_fail
/// use fovea::transform::{InterpolationKernel, Lanczos};
/// // ERROR: a Lanczos kernel needs at least one lobe
/// let _ = <Lanczos<0> as InterpolationKernel>::RADIUS;
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Lanczos<const A: usize>;

impl<const A: usize> InterpolationKernel for Lanczos<A> {
    const RADIUS: usize = {
        assert!(A >= 1, "a Lanczos kernel needs at least one lobe");
        A
    };

    #[inline]
    fn weight(&self, d: f64) -> f64 {
        let x = d.abs();
        let a = A as f64;
        if x == 0.0 {
            1.0
        } else if x >= a || x.fract() == 0.0 {
            // Zero at every other integer, exactly, so the kernel interpolates:
            // `sin(k·π)` in floating point is not quite zero.
            0.0
        } else {
            let px = core::f64::consts::PI * x;
            a * px.sin() * (px / a).sin() / (px * px)
        }
    }
}

/// Lanczos with two lobes: four taps per axis.
pub type Lanczos2 = Lanczos<2>;
/// Lanczos with three lobes: six taps per axis, the common choice.
pub type Lanczos3 = Lanczos<3>;

/// The [`Lanczos2`](type@Lanczos2) kernel as a value, so `resize(&img, size, Lanczos2)?` reads
/// like the other kernels.
#[allow(non_upper_case_globals)]
pub const Lanczos2: Lanczos2 = Lanczos;
/// The [`Lanczos3`](type@Lanczos3) kernel as a value, so `resize(&img, size, Lanczos3)?` reads
/// like the other kernels.
#[allow(non_upper_case_globals)]
pub const Lanczos3: Lanczos3 = Lanczos;

// ─── Antialiased ─────────────────────────────────────────────────────────────

/// A resize method that widens its kernel by the shrink factor.
///
/// When a resize shrinks by a factor `s > 1`, `Antialiased(kernel)` stretches
/// the kernel to `s` times its width, so each target pixel averages all the
/// source pixels it stands for and detail too fine for the target grid turns
/// into its mean instead of a false pattern. When a resize enlarges, it is
/// the bare kernel. The cost grows with the factor: shrinking by 3 reads about
/// three times as many source pixels per axis.
///
/// A resize method only; [`sample`](crate::analyze::sampling::sample)
/// interpolates at a point and has no factor to widen by.
///
/// # Example
///
/// One-pixel stripes cannot survive a shrink by 3. The bare kernel turns them
/// into a false coarse pattern; `Antialiased` into grey:
///
/// ```
/// use fovea::Size;
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::MonoF32;
/// use fovea::transform::{Antialiased, CatmullRom, resize};
///
/// let stripes = Image::generate(60, 1, |x, _| MonoF32::new((x % 2) as f32));
/// let target = Size::new(20, 1);
///
/// let aliased: Image<MonoF32> = resize(&stripes, target, CatmullRom)?;
/// let filtered: Image<MonoF32> = resize(&stripes, target, Antialiased(CatmullRom))?;
///
/// let contrast = |img: &Image<MonoF32>| {
///     (img.pixel_at(4, 0).value() - img.pixel_at(5, 0).value()).abs()
/// };
/// assert!(contrast(&aliased) > 0.9, "stripes at full contrast: a false pattern");
/// assert!(contrast(&filtered) < 0.05, "grey: the stripes' mean");
/// # Ok::<(), fovea::Error>(())
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Antialiased<K>(pub K);

// ─── Weight tables ───────────────────────────────────────────────────────────

/// Per-position tap lists for one axis: the source indices and their
/// normalised weights, computed once and reused for every row or column.
pub(crate) struct AxisWeights {
    /// `offsets[i]..offsets[i + 1]` indexes the taps of output position `i`.
    offsets: Vec<usize>,
    index: Vec<usize>,
    weight: Vec<f32>,
}

impl AxisWeights {
    /// The taps of an axis resized from `in_len` to `out_len` samples, with
    /// half-pixel centres: output sample `i` sits at source position
    /// `(i + 0.5) · in_len / out_len − 0.5`. `widen` stretches the kernel by
    /// the shrink factor. Taps outside the source repeat its edge sample.
    ///
    /// `in_len` must be non-zero.
    pub(crate) fn resize<K: InterpolationKernel>(
        kernel: &K,
        in_len: usize,
        out_len: usize,
        widen: bool,
    ) -> Self {
        debug_assert!(in_len > 0, "an empty axis has no samples to weight");
        let scale = in_len as f64 / out_len.max(1) as f64;
        let stretch = if widen { scale.max(1.0) } else { 1.0 };
        let reach = K::RADIUS as f64 * stretch;
        let last = in_len as isize - 1;

        let mut offsets = Vec::with_capacity(out_len + 1);
        let mut index = Vec::new();
        let mut weight: Vec<f32> = Vec::new();
        // The unnormalised weights, kept in f64 until each position is summed.
        let mut raw: Vec<f64> = Vec::new();
        offsets.push(0);
        for i in 0..out_len {
            let centre = (i as f64 + 0.5) * scale - 0.5;
            let first = (centre - reach).floor() as isize + 1;
            let end = (centre + reach).floor() as isize;
            let start = raw.len();
            let mut sum = 0.0;
            for k in first..=end {
                let w = kernel.weight((k as f64 - centre) / stretch);
                if w != 0.0 {
                    let k = k.clamp(0, last) as usize;
                    // Taps clamped onto the same edge sample are adjacent;
                    // merge them so the edge is read once, with one weight.
                    if raw.len() > start && index.last() == Some(&k) {
                        *raw.last_mut().expect("a tap was pushed") += w;
                    } else {
                        index.push(k);
                        raw.push(w);
                    }
                    sum += w;
                }
            }
            if raw.len() == start {
                // A kernel that is zero across the whole footprint: keep one
                // tap of weight zero so every position has a defined value.
                index.push(centre.round().clamp(0.0, last as f64) as usize);
                raw.push(0.0);
            }
            let norm = if sum != 0.0 { 1.0 / sum } else { 1.0 };
            weight.extend(raw[start..].iter().map(|w| (w * norm) as f32));
            offsets.push(weight.len());
        }
        Self {
            offsets,
            index,
            weight,
        }
    }

    /// The source indices and weights of output position `i`.
    #[inline]
    pub(crate) fn taps(&self, i: usize) -> (&[usize], &[f32]) {
        let range = self.offsets[i]..self.offsets[i + 1];
        (&self.index[range.clone()], &self.weight[range])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kernels_interpolate<K: InterpolationKernel>(k: K) {
        assert_eq!(k.weight(0.0), 1.0);
        for d in 1..=K::RADIUS as i32 {
            assert_eq!(k.weight(d as f64), 0.0, "weight at {d}");
            assert_eq!(k.weight(-d as f64), 0.0, "weight at -{d}");
        }
        assert_eq!(k.weight(K::RADIUS as f64 + 0.25), 0.0);
    }

    #[test]
    fn every_kernel_interpolates_and_respects_its_radius() {
        kernels_interpolate(Bilinear);
        kernels_interpolate(CatmullRom);
        kernels_interpolate(KeysBicubic::new(-0.75).unwrap());
        kernels_interpolate(KeysBicubic::new(2.0).unwrap());
        kernels_interpolate(Lanczos2);
        kernels_interpolate(Lanczos3);
        kernels_interpolate(Lanczos::<5>);
    }

    #[test]
    fn keys_weights_sum_to_one_for_any_finite_a() {
        for a in [-3.0, -1.0, -0.75, -0.5, 0.0, 0.5, 2.0] {
            let k = KeysBicubic::new(a).unwrap();
            for step in 0..=20 {
                let t = step as f64 / 20.0;
                let sum: f64 = (-1..=2).map(|i| k.weight(i as f64 - t)).sum();
                assert!((sum - 1.0).abs() < 1e-12, "a = {a}, t = {t}: {sum}");
            }
        }
    }

    #[test]
    fn catmull_rom_is_keys_at_minus_one_half() {
        let k = KeysBicubic::new(-0.5).unwrap();
        for step in -40..=40 {
            let d = step as f64 / 10.0;
            assert_eq!(CatmullRom.weight(d), k.weight(d));
        }
        // OpenCV's `interpolateCubic` coefficients at x = 0.5 for A = -0.75.
        let cv = KeysBicubic::new(-0.75).unwrap();
        assert_eq!(cv.weight(1.5), -0.09375);
        assert_eq!(cv.weight(0.5), 0.59375);
    }

    #[test]
    fn keys_bicubic_rejects_non_finite_a() {
        assert!(KeysBicubic::new(f64::INFINITY).is_none());
        let Err(Error::InvalidParameter(e)) = KeysBicubic::try_new(f64::NAN) else {
            panic!("a NaN coefficient must be rejected as a parameter");
        };
        assert_eq!(e.requirement(), Requirement::Finite);
        assert_eq!(e.value(), Value::F64(f64::NAN));
        assert_eq!(KeysBicubic::try_new(-0.75).unwrap().a(), -0.75);
        assert_eq!(keys_bicubic!(-0.5).a(), -0.5);
    }

    #[test]
    fn lanczos_matches_the_sinc_product() {
        let d: f64 = 0.4;
        let sinc = |x: f64| (core::f64::consts::PI * x).sin() / (core::f64::consts::PI * x);
        let expected = sinc(d) * sinc(d / 3.0);
        assert!((Lanczos3.weight(d) - expected).abs() < 1e-15);
        assert_eq!(Lanczos3.weight(-d), Lanczos3.weight(d));
        assert_eq!(<Lanczos3 as InterpolationKernel>::RADIUS, 3);
    }

    #[test]
    fn axis_weights_use_half_pixel_centres_and_normalise() {
        // Halving with the tent: output i sits at 2i + 0.5, halfway between
        // two source samples, so it averages them.
        let w = AxisWeights::resize(&Bilinear, 8, 4, false);
        for i in 0..4 {
            let (index, weight) = w.taps(i);
            assert_eq!(index, &[2 * i, 2 * i + 1]);
            assert_eq!(weight, &[0.5, 0.5]);
        }
        // Lanczos weights are normalised to sum to one.
        let w = AxisWeights::resize(&Lanczos3, 10, 7, false);
        for i in 0..7 {
            let sum: f32 = w.taps(i).1.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "{sum}");
        }
    }

    #[test]
    fn widening_stretches_only_when_shrinking() {
        // Shrinking by 3 puts output 1 exactly on source sample 4: the fixed
        // tent reads that one sample, the widened tent five, 2 to 6.
        let fixed = AxisWeights::resize(&Bilinear, 12, 4, false);
        let wide = AxisWeights::resize(&Bilinear, 12, 4, true);
        assert_eq!(fixed.taps(1).0, &[4]);
        assert_eq!(wide.taps(1).0, &[2, 3, 4, 5, 6]);
        let up_fixed = AxisWeights::resize(&CatmullRom, 4, 12, false);
        let up_wide = AxisWeights::resize(&CatmullRom, 4, 12, true);
        for i in 0..12 {
            assert_eq!(up_fixed.taps(i), up_wide.taps(i));
        }
    }

    #[test]
    fn edge_taps_repeat_the_edge_sample_once() {
        // Output 0 of 3 → 6 sits at −0.25: its taps at −2 and −1 clamp onto
        // sample 0 and merge with the tap at 0.
        let w = AxisWeights::resize(&CatmullRom, 3, 6, false);
        let (index, weight) = w.taps(0);
        assert_eq!(index, &[0, 1]);
        let sum: f32 = weight.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}

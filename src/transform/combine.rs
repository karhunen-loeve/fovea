//! Pixel-wise binary image combination.
//!
//! This module provides:
//!
//! - [`CombinePixels`] — the trait for parameterising binary pixel operations.
//! - [`ClosureCombine`] — an adapter for closures as one-off combiners.
//! - [`combine_images`] / [`combine_images_into`] — image-level driver functions.
//! - Named combiners: [`PixelAdd`], [`PixelSubtract`], [`PixelMultiply`],
//!   [`AbsDiff`], [`Max`], [`Min`], [`LinearCombine`], [`Blend`], [`Magnitude`].
//!
//! Everything is re-exported from the parent [`transform`](super) module:
//!
//! ```
//! use fovea::transform::{CombinePixels, PixelAdd, AbsDiff, Blend, Magnitude};
//! ```

use std::ops::Add as StdAdd;
use std::ops::Mul as StdMul;
use std::ops::Sub as StdSub;

use crate::error::Error;
use crate::image::{Image, RasterImage, RasterImageMut};
use crate::pixel::{HomogeneousPixel, LinearPixel, LinearSpace, ZeroablePixel};

// ─── CombinePixels trait ─────────────────────────────────────────────────────

/// Strategy for combining two pixels into one output pixel.
///
/// This is the binary analogue of [`ConvertPixel`](super::ConvertPixel) — the
/// strategy determines the operation *and* the output type.  Different
/// strategies express different semantics (saturating add, absolute
/// difference, widening multiply, etc.) and the caller picks the one that
/// matches their domain.
///
/// # Type Parameters
///
/// - `A` — pixel type of the first (left) image.
/// - `B` — pixel type of the second (right) image.
///
/// # Example: custom strategy
///
/// ```
/// use fovea::transform::CombinePixels;
/// use fovea::pixel::Mono8;
///
/// struct Average;
///
/// impl CombinePixels<Mono8, Mono8> for Average {
///     type Output = Mono8;
///     fn combine(&self, a: &Mono8, b: &Mono8) -> Mono8 {
///         Mono8::new(((a.value() as u16 + b.value() as u16) / 2) as u8)
///     }
/// }
/// ```
pub trait CombinePixels<A, B> {
    /// The output pixel type produced by this combination.
    type Output;

    /// Combine two pixels into one output pixel.
    fn combine(&self, a: &A, b: &B) -> Self::Output;
}

// ─── ClosureCombine ──────────────────────────────────────────────────────────

/// Wrapper that lets a closure be used as a [`CombinePixels`] strategy.
///
/// This is the binary analogue of [`PixelMap`](super::PixelMap).
///
/// # Example
///
/// ```
/// use fovea::transform::{CombinePixels, ClosureCombine};
/// use fovea::pixel::Mono8;
///
/// let strategy = ClosureCombine(|a: &Mono8, b: &Mono8| {
///     Mono8::new(a.value().max(b.value()))
/// });
/// let result = strategy.combine(&Mono8::new(10), &Mono8::new(20));
/// assert_eq!(result, Mono8::new(20));
/// ```
pub struct ClosureCombine<F>(pub F);

impl<A, B, Out, F> CombinePixels<A, B> for ClosureCombine<F>
where
    F: Fn(&A, &B) -> Out,
{
    type Output = Out;

    #[inline(always)]
    fn combine(&self, a: &A, b: &B) -> Out {
        (self.0)(a, b)
    }
}

// ─── Image-level functions ───────────────────────────────────────────────────

/// Combine two images pixel-wise, writing results into a pre-allocated
/// output image.
///
/// Returns `Err(Error::SizeMismatch)` if the two input images have different
/// sizes (consistent with [`zip_pixels`](crate::image::zip_pixels)).
///
/// # Panics
///
/// Panics if the output image size does not match the input size (consistent
/// with [`convert_image_into`](super::convert_image_into)).
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::Mono8;
/// use fovea::transform::{ClosureCombine, combine_images_into};
///
/// let a = Image::fill(4, 4, Mono8::new(100));
/// let b = Image::fill(4, 4, Mono8::new(50));
/// let mut out = Image::zero(4, 4);
///
/// let result = combine_images_into(&a, &b, &mut out,
///     ClosureCombine(|a: &Mono8, b: &Mono8| {
///         Mono8::new(a.value().saturating_add(b.value()))
///     }),
/// );
/// assert!(result.is_ok());
/// assert_eq!(out.pixel_at(0, 0), Mono8::new(150));
/// ```
pub fn combine_images_into<IA, IB, O, M>(
    a: &IA,
    b: &IB,
    out: &mut O,
    method: M,
) -> Result<(), Error>
where
    IA: RasterImage,
    IB: RasterImage,
    O: RasterImageMut<Pixel = M::Output>,
    M: CombinePixels<IA::Pixel, IB::Pixel>,
{
    if a.size() != b.size() {
        return Err(Error::SizeMismatch {
            expected: a.size(),
            actual: b.size(),
        });
    }

    assert_eq!(
        a.size(),
        out.size(),
        "combine_images_into: input size {:?} does not match output size {:?}",
        a.size(),
        out.size()
    );

    for y in 0..a.height() {
        let row_a = a.row(y);
        let row_b = b.row(y);
        let row_out = out.row_mut(y);
        for ((pa, pb), dst) in row_a.iter().zip(row_b.iter()).zip(row_out.iter_mut()) {
            *dst = method.combine(pa, pb);
        }
    }

    Ok(())
}

/// Combine two images pixel-wise, returning a new [`Image`] with the
/// results.
///
/// Returns `Err(Error::SizeMismatch)` if the two input images have different
/// sizes (consistent with [`zip_pixels`](crate::image::zip_pixels)).
///
/// This is a convenience wrapper around [`combine_images_into`].
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::Mono8;
/// use fovea::transform::{ClosureCombine, combine_images};
///
/// let a = Image::fill(3, 3, Mono8::new(200));
/// let b = Image::fill(3, 3, Mono8::new(100));
///
/// let result = combine_images(&a, &b,
///     ClosureCombine(|a: &Mono8, b: &Mono8| {
///         Mono8::new(a.value().saturating_add(b.value()))
///     }),
/// );
///
/// let out = result.unwrap();
/// assert_eq!(out.pixel_at(0, 0), Mono8::new(255)); // saturated
/// ```
pub fn combine_images<IA, IB, M>(a: &IA, b: &IB, method: M) -> Result<Image<M::Output>, Error>
where
    IA: RasterImage,
    IB: RasterImage,
    M: CombinePixels<IA::Pixel, IB::Pixel>,
    M::Output: ZeroablePixel,
{
    if a.size() != b.size() {
        return Err(Error::SizeMismatch {
            expected: a.size(),
            actual: b.size(),
        });
    }

    let mut out = Image::<M::Output>::zero(a.width(), a.height());
    // unwrap is safe: we just checked a.size() == b.size() above
    combine_images_into(a, b, &mut out, method).unwrap();
    Ok(out)
}

// ─── Closure convenience wrappers ────────────────────────────────────────────

/// Convenience wrapper: [`combine_images_into`] accepting a closure.
///
/// Wraps the closure in [`ClosureCombine`].  This mirrors the pattern
/// of [`fold_neighborhood_fn_into`](super::fold_neighborhood_fn_into).
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::{Mono8, Mono32};
/// use fovea::transform::combine_images_fn_into;
///
/// let a = Image::fill(2, 2, Mono8::new(200));
/// let b = Image::fill(2, 2, Mono8::new(100));
/// let mut out: Image<Mono32> = Image::zero(2, 2);
///
/// let result = combine_images_fn_into(&a, &b, &mut out, |a: &Mono8, b: &Mono8| {
///     Mono32::new(a.value() as u32 * b.value() as u32)
/// });
///
/// assert!(result.is_ok());
/// assert_eq!(out.pixel_at(0, 0), Mono32::new(20000));
/// ```
pub fn combine_images_fn_into<IA, IB, O, Out, F>(
    a: &IA,
    b: &IB,
    out: &mut O,
    f: F,
) -> Result<(), Error>
where
    IA: RasterImage,
    IB: RasterImage,
    O: RasterImageMut<Pixel = Out>,
    F: Fn(&IA::Pixel, &IB::Pixel) -> Out,
{
    combine_images_into(a, b, out, ClosureCombine(f))
}

/// Convenience wrapper: [`combine_images`] accepting a closure.
///
/// Wraps the closure in [`ClosureCombine`].  This mirrors the pattern
/// of [`fold_neighborhood_fn`](super::fold_neighborhood_fn).
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::{Mono8, Mono32};
/// use fovea::transform::combine_images_fn;
///
/// let a = Image::fill(3, 3, Mono8::new(10));
/// let b = Image::fill(3, 3, Mono8::new(20));
///
/// // Widening multiply: Mono8 × Mono8 → Mono32
/// let result = combine_images_fn(&a, &b, |a: &Mono8, b: &Mono8| {
///     Mono32::new(a.value() as u32 * b.value() as u32)
/// });
///
/// let out = result.unwrap();
/// assert_eq!(out.pixel_at(0, 0), Mono32::new(200));
/// ```
pub fn combine_images_fn<IA, IB, Out, F>(a: &IA, b: &IB, f: F) -> Result<Image<Out>, Error>
where
    IA: RasterImage,
    IB: RasterImage,
    Out: ZeroablePixel,
    F: Fn(&IA::Pixel, &IB::Pixel) -> Out,
{
    combine_images(a, b, ClosureCombine(f))
}

// ─── std::ops delegation ─────────────────────────────────────────────────────

/// Combines two pixels by adding them (delegates to [`std::ops::Add`]).
///
/// The arithmetic semantics are **entirely determined by the pixel type**:
/// saturating for integer pixels derived with `#[derive(LinearPixel)]`,
/// IEEE-754 for float pixels (`MonoF32`, `RgbF32`, …).
///
/// # Example
///
/// ```
/// use fovea::transform::{CombinePixels, PixelAdd};
/// use fovea::pixel::Mono8;
///
/// // 200 + 100 saturates to 255 for Mono8
/// assert_eq!(PixelAdd.combine(&Mono8::new(200), &Mono8::new(100)), Mono8::new(255));
/// // Normal addition without overflow
/// assert_eq!(PixelAdd.combine(&Mono8::new(100), &Mono8::new(50)), Mono8::new(150));
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PixelAdd;

impl<P: Copy + StdAdd<Output = P>> CombinePixels<P, P> for PixelAdd {
    type Output = P;

    #[inline(always)]
    fn combine(&self, a: &P, b: &P) -> P {
        *a + *b
    }
}

/// Combines two pixels by subtracting them (delegates to [`std::ops::Sub`]).
///
/// As with [`PixelAdd`], the overflow semantics come from the pixel type.
/// For unsigned integer pixels this saturates at zero; for float pixels it
/// is IEEE-754 and can produce negative results.
///
/// # Example
///
/// ```
/// use fovea::transform::{CombinePixels, PixelSubtract};
/// use fovea::pixel::Mono8;
///
/// // 50 - 100 saturates to 0 for Mono8
/// assert_eq!(PixelSubtract.combine(&Mono8::new(50), &Mono8::new(100)), Mono8::new(0));
/// assert_eq!(PixelSubtract.combine(&Mono8::new(200), &Mono8::new(50)), Mono8::new(150));
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PixelSubtract;

impl<P: Copy + StdSub<Output = P>> CombinePixels<P, P> for PixelSubtract {
    type Output = P;

    #[inline(always)]
    fn combine(&self, a: &P, b: &P) -> P {
        *a - *b
    }
}

/// Combines two pixels by multiplying them (delegates to [`std::ops::Mul`]).
///
/// For integer pixels derived with `#[derive(LinearPixel)]` this is
/// channel-wise saturating multiplication.  For float pixels it is IEEE-754.
///
/// # Example
///
/// ```
/// use fovea::transform::{CombinePixels, PixelMultiply};
/// use fovea::pixel::MonoF32;
///
/// assert_eq!(
///     PixelMultiply.combine(&MonoF32::new(0.5), &MonoF32::new(0.5)),
///     MonoF32::new(0.25)
/// );
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PixelMultiply;

impl<P: Copy + StdMul<Output = P>> CombinePixels<P, P> for PixelMultiply {
    type Output = P;

    #[inline(always)]
    fn combine(&self, a: &P, b: &P) -> P {
        *a * *b
    }
}

// ─── AbsDiff ─────────────────────────────────────────────────────────────────

/// Channel-wise absolute difference `|a − b|`.
///
/// The operation is commutative.  For standard unsigned integer pixels the
/// result always fits in the channel type without overflow.
///
/// Works for all standard integer pixel types (`Mono8`, `Mono16`, `Rgb8`, …)
/// and float pixel types (`MonoF32`, `MonoF64`, `RgbF32`, …).
///
/// **Bounds:** `P::Channel` must implement [`PartialOrd`] and
/// [`Sub<Output = P::Channel>`](std::ops::Sub).  These are satisfied by every
/// standard channel type (`Saturating<u8/u16/…>`, `f32`, `f64`).
///
/// # Example
///
/// ```
/// use fovea::transform::{CombinePixels, AbsDiff};
/// use fovea::pixel::Mono8;
///
/// assert_eq!(AbsDiff.combine(&Mono8::new(100), &Mono8::new(150)), Mono8::new(50));
/// // Result is the same regardless of operand order
/// assert_eq!(AbsDiff.combine(&Mono8::new(150), &Mono8::new(100)), Mono8::new(50));
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AbsDiff;

impl<P> CombinePixels<P, P> for AbsDiff
where
    P: HomogeneousPixel,
    P::Channel: PartialOrd + StdSub<Output = P::Channel>,
{
    type Output = P;

    fn combine(&self, a: &P, b: &P) -> P {
        // P: Copy is guaranteed by P: HomogeneousPixel → PlainPixel: Sized + Copy.
        // The initial value is irrelevant; every channel is overwritten below.
        let mut result = *a;
        for i in 0..P::CHANNEL_COUNT {
            let ac = a.channel(i);
            let bc = b.channel(i);
            // For unsigned channels: exactly one branch subtracts a smaller
            // value from a larger one, so no overflow.
            // For float channels: NaN comparisons return false, so NaN
            // propagates via the else branch, which is correct IEEE behaviour.
            result.set_channel(i, if ac >= bc { ac - bc } else { bc - ac });
        }
        result
    }
}

// ─── Min / Max ───────────────────────────────────────────────────────────────

/// Channel-wise maximum `max(a, b)`.
///
/// Requires ordered channels (`Channel: Ord`).  Works for all standard integer
/// pixel types.  Float pixel types (`MonoF32`, `RgbF32`, …) are excluded because
/// `f32`/`f64` do not implement `Ord` (NaN-safety).
///
/// Typical use: morphological dilation logic, HDR exposure merge (take brightest).
///
/// # Example
///
/// ```
/// use fovea::transform::{CombinePixels, Max};
/// use fovea::pixel::Mono8;
///
/// assert_eq!(Max.combine(&Mono8::new(100), &Mono8::new(200)), Mono8::new(200));
/// assert_eq!(Max.combine(&Mono8::new(200), &Mono8::new(100)), Mono8::new(200));
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Max;

impl<P> CombinePixels<P, P> for Max
where
    P: HomogeneousPixel,
    P::Channel: Ord,
{
    type Output = P;

    fn combine(&self, a: &P, b: &P) -> P {
        let mut result = *a;
        for i in 0..P::CHANNEL_COUNT {
            result.set_channel(i, a.channel(i).max(b.channel(i)));
        }
        result
    }
}

/// Channel-wise minimum `min(a, b)`.
///
/// Requires ordered channels (`Channel: Ord`).  Works for all standard integer
/// pixel types.  Float pixel types are excluded (see [`Max`]).
///
/// Typical use: morphological erosion logic, shadow/dark-region extraction.
///
/// # Example
///
/// ```
/// use fovea::transform::{CombinePixels, Min};
/// use fovea::pixel::Mono8;
///
/// assert_eq!(Min.combine(&Mono8::new(100), &Mono8::new(200)), Mono8::new(100));
/// assert_eq!(Min.combine(&Mono8::new(200), &Mono8::new(100)), Mono8::new(100));
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Min;

impl<P> CombinePixels<P, P> for Min
where
    P: HomogeneousPixel,
    P::Channel: Ord,
{
    type Output = P;

    fn combine(&self, a: &P, b: &P) -> P {
        let mut result = *a;
        for i in 0..P::CHANNEL_COUNT {
            result.set_channel(i, a.channel(i).min(b.channel(i)));
        }
        result
    }
}

// ─── Linear combination ──────────────────────────────────────────────────────

/// General linear combination `wa·a + wb·b`.
///
/// Requires pixels to live in a linear space ([`LinearPixel`] + [`LinearSpace`]).
/// The **output type is `P::Accumulator`** (e.g. `MonoF32` for `Mono8`, or
/// `RgbF32` for `Rgb8`), preserving full precision without rounding back to
/// the storage type.
///
/// Use [`Blend`] as a convenience when you want interpolation semantics
/// (`wa = 1 − α`, `wb = α`).
///
/// # Example
///
/// ```
/// use fovea::transform::{CombinePixels, LinearCombine};
/// use fovea::pixel::{Mono8, MonoF32};
///
/// // Equal-weight average: 0.5·100 + 0.5·200 = 150
/// let mid = LinearCombine { wa: 0.5, wb: 0.5 }.combine(&Mono8::new(100), &Mono8::new(200));
/// // `Mono8::Accumulator = MonoF32`, so `mid: MonoF32`.
/// assert!((mid - MonoF32(150.0)).abs().0 < 1e-5);
/// ```
pub struct LinearCombine {
    pub wa: f32,
    pub wb: f32,
}

impl<P: LinearPixel + LinearSpace> CombinePixels<P, P> for LinearCombine {
    type Output = P::Accumulator;

    #[inline(always)]
    fn combine(&self, a: &P, b: &P) -> P::Accumulator {
        b.scale_add(self.wb, a.scale(self.wa))
    }
}

/// Linear interpolation: `(1 − α)·a + α·b`.
///
/// Requires pixels to live in a linear space ([`LinearPixel`] + [`LinearSpace`]).
/// The **output type is `P::Accumulator`**.
///
/// - `alpha = 0.0` → output equals `a` (in accumulator space)
/// - `alpha = 1.0` → output equals `b` (in accumulator space)
/// - `alpha = 0.5` → midpoint
///
/// Equivalent to `LinearCombine { wa: 1.0 - alpha, wb: alpha }`.
///
/// # Example
///
/// ```
/// use fovea::transform::{CombinePixels, Blend};
/// use fovea::pixel::{Mono8, MonoF32};
///
/// let mid = Blend { alpha: 0.5 }.combine(&Mono8::new(0), &Mono8::new(200));
/// // `Mono8::Accumulator = MonoF32`, so `mid: MonoF32`.
/// assert!((mid - MonoF32(100.0)).abs().0 < 1e-5);
/// ```
pub struct Blend {
    pub alpha: f32,
}

impl<P: LinearPixel + LinearSpace> CombinePixels<P, P> for Blend {
    type Output = P::Accumulator;

    #[inline(always)]
    fn combine(&self, a: &P, b: &P) -> P::Accumulator {
        b.scale_add(self.alpha, a.scale(1.0 - self.alpha))
    }
}

// ─── Magnitude ───────────────────────────────────────────────────────────────

/// Sealing module for [`MagnitudeChannel`].
mod magnitude_sealed {
    pub trait Sealed: Copy {}
}

/// Channel types that support `sqrt(a² + b²)`.
///
/// Implemented for `f32` and `f64`.  This trait is **sealed**: it cannot be
/// implemented outside this crate.
pub trait MagnitudeChannel: magnitude_sealed::Sealed + Copy {
    /// Compute `sqrt(a² + b²)`, the Euclidean length of the vector `(a, b)`.
    fn magnitude(a: Self, b: Self) -> Self;
}

impl magnitude_sealed::Sealed for f32 {}
impl MagnitudeChannel for f32 {
    #[inline(always)]
    fn magnitude(a: f32, b: f32) -> f32 {
        f32::hypot(a, b)
    }
}

impl magnitude_sealed::Sealed for f64 {}
impl MagnitudeChannel for f64 {
    #[inline(always)]
    fn magnitude(a: f64, b: f64) -> f64 {
        f64::hypot(a, b)
    }
}

/// Channel-wise Euclidean magnitude `sqrt(a² + b²)`.
///
/// Primary use-case: combine X- and Y-gradient images (e.g. from [`sobel_x`] /
/// [`sobel_y`]) into a gradient-magnitude image.
///
/// Only defined for float-channel pixel types (`MonoF32`, `MonoF64`, `RgbF32`, …).
/// The underlying computation uses [`f32::hypot`] / [`f64::hypot`], which avoids
/// intermediate overflow for large values.
///
/// [`sobel_x`]: crate::transform::sobel_x
/// [`sobel_y`]: crate::transform::sobel_y
///
/// # Example
///
/// ```
/// use fovea::transform::{CombinePixels, Magnitude};
/// use fovea::pixel::MonoF32;
///
/// // 3-4-5 Pythagorean triple
/// let mag = Magnitude.combine(&MonoF32::new(3.0), &MonoF32::new(4.0));
/// assert!((mag.value() - 5.0).abs() < 1e-5);
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Magnitude;

impl<P> CombinePixels<P, P> for Magnitude
where
    P: HomogeneousPixel,
    P::Channel: MagnitudeChannel,
{
    type Output = P;

    fn combine(&self, a: &P, b: &P) -> P {
        let mut result = *a;
        for i in 0..P::CHANNEL_COUNT {
            result.set_channel(i, MagnitudeChannel::magnitude(a.channel(i), b.channel(i)));
        }
        result
    }
}

// ─── Convenience free functions ──────────────────────────────────────────────

/// Add two images pixel-wise and return the result, or `Err(Error::SizeMismatch)` if sizes differ.
///
/// This is a thin wrapper over [`combine_images`] with [`PixelAdd`].
/// The arithmetic semantics are determined by the pixel type: saturating for
/// integer pixels, IEEE-754 for float pixels.
///
/// # Example
///
/// ```
/// use fovea::transform::add;
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::Mono8;
///
/// let a = Image::fill(3, 3, Mono8::new(100));
/// let b = Image::fill(3, 3, Mono8::new(50));
/// let result = add(&a, &b).unwrap();
/// assert_eq!(result.pixel_at(0, 0), Mono8::new(150));
///
/// // Size mismatch → Err
/// let c = Image::fill(4, 4, Mono8::new(0));
/// assert!(add(&a, &c).is_err());
/// ```
pub fn add<IA, IB>(a: &IA, b: &IB) -> Result<Image<IA::Pixel>, Error>
where
    IA: RasterImage,
    IB: RasterImage<Pixel = IA::Pixel>,
    IA::Pixel: Copy + StdAdd<Output = IA::Pixel> + ZeroablePixel,
{
    combine_images(a, b, PixelAdd)
}

/// Subtract two images pixel-wise and return the result, or `Err(Error::SizeMismatch)` if sizes differ.
///
/// This is a thin wrapper over [`combine_images`] with [`PixelSubtract`].
/// The arithmetic semantics are determined by the pixel type: saturating for
/// integer pixels, IEEE-754 for float pixels.
///
/// # Example
///
/// ```
/// use fovea::transform::subtract;
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::Mono8;
///
/// let a = Image::fill(3, 3, Mono8::new(200));
/// let b = Image::fill(3, 3, Mono8::new(50));
/// let result = subtract(&a, &b).unwrap();
/// assert_eq!(result.pixel_at(0, 0), Mono8::new(150));
///
/// // Saturates to 0 for unsigned pixels
/// let c = Image::fill(3, 3, Mono8::new(255));
/// let saturated = subtract(&b, &c).unwrap();
/// assert_eq!(saturated.pixel_at(0, 0), Mono8::new(0));
/// ```
pub fn subtract<IA, IB>(a: &IA, b: &IB) -> Result<Image<IA::Pixel>, Error>
where
    IA: RasterImage,
    IB: RasterImage<Pixel = IA::Pixel>,
    IA::Pixel: Copy + StdSub<Output = IA::Pixel> + ZeroablePixel,
{
    combine_images(a, b, PixelSubtract)
}

/// Compute the absolute difference of two images pixel-wise, or `Err(Error::SizeMismatch)` if sizes differ.
///
/// This is a thin wrapper over [`combine_images`] with [`AbsDiff`].
/// For each channel: `|a - b|`. Always non-negative; the result type is the
/// same as the input type.
///
/// # Example
///
/// ```
/// use fovea::transform::abs_diff;
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::Mono8;
///
/// let a = Image::fill(3, 3, Mono8::new(200));
/// let b = Image::fill(3, 3, Mono8::new(50));
///
/// let result = abs_diff(&a, &b).unwrap();
/// assert_eq!(result.pixel_at(0, 0), Mono8::new(150));
///
/// // Symmetric: abs_diff(a, b) == abs_diff(b, a)
/// let swapped = abs_diff(&b, &a).unwrap();
/// assert_eq!(result.pixel_at(0, 0), swapped.pixel_at(0, 0));
/// ```
pub fn abs_diff<IA, IB>(a: &IA, b: &IB) -> Result<Image<IA::Pixel>, Error>
where
    IA: RasterImage,
    IB: RasterImage<Pixel = IA::Pixel>,
    IA::Pixel: HomogeneousPixel + ZeroablePixel,
    <IA::Pixel as HomogeneousPixel>::Channel:
        PartialOrd + StdSub<Output = <IA::Pixel as HomogeneousPixel>::Channel>,
{
    combine_images(a, b, AbsDiff)
}

/// Compute the channel-wise minimum of two images, or `Err(Error::SizeMismatch)` if sizes differ.
///
/// This is a thin wrapper over [`combine_images`] with [`Min`].
/// Only defined for pixel types with ordered integer channels.
/// For float pixels use [`combine_images`] with a custom strategy.
///
/// # Example
///
/// ```
/// use fovea::transform::image_min;
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::Mono8;
///
/// let a = Image::fill(3, 3, Mono8::new(100));
/// let b = Image::fill(3, 3, Mono8::new(200));
/// let result = image_min(&a, &b).unwrap();
/// assert_eq!(result.pixel_at(0, 0), Mono8::new(100));
/// ```
pub fn image_min<IA, IB>(a: &IA, b: &IB) -> Result<Image<IA::Pixel>, Error>
where
    IA: RasterImage,
    IB: RasterImage<Pixel = IA::Pixel>,
    IA::Pixel: HomogeneousPixel + ZeroablePixel,
    <IA::Pixel as HomogeneousPixel>::Channel: Ord,
{
    combine_images(a, b, Min)
}

/// Compute the channel-wise maximum of two images, or `Err(Error::SizeMismatch)` if sizes differ.
///
/// This is a thin wrapper over [`combine_images`] with [`Max`].
/// Only defined for pixel types with ordered integer channels.
/// For float pixels use [`combine_images`] with a custom strategy.
///
/// # Example
///
/// ```
/// use fovea::transform::image_max;
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::Mono8;
///
/// let a = Image::fill(3, 3, Mono8::new(100));
/// let b = Image::fill(3, 3, Mono8::new(200));
/// let result = image_max(&a, &b).unwrap();
/// assert_eq!(result.pixel_at(0, 0), Mono8::new(200));
/// ```
pub fn image_max<IA, IB>(a: &IA, b: &IB) -> Result<Image<IA::Pixel>, Error>
where
    IA: RasterImage,
    IB: RasterImage<Pixel = IA::Pixel>,
    IA::Pixel: HomogeneousPixel + ZeroablePixel,
    <IA::Pixel as HomogeneousPixel>::Channel: Ord,
{
    combine_images(a, b, Max)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Rectangle;
    use crate::image::{Image, ImageArray, ImageView, SubView};
    use crate::pixel::*;

    // ── CombinePixels trait + ClosureCombine ─────────────────────────────────

    #[test]
    fn combine_pixels_closure_mono8() {
        let strategy =
            ClosureCombine(|a: &Mono8, b: &Mono8| Mono8::new(a.value().wrapping_add(b.value())));
        let result = strategy.combine(&Mono8::new(100), &Mono8::new(50));
        assert_eq!(result, Mono8::new(150));
    }

    #[test]
    fn combine_pixels_closure_cross_type() {
        // Mono8 × Mono8 → Mono32 (widening)
        let strategy =
            ClosureCombine(|a: &Mono8, b: &Mono8| Mono32::new(a.value() as u32 * b.value() as u32));
        let result = strategy.combine(&Mono8::new(200), &Mono8::new(200));
        assert_eq!(result, Mono32::new(40000));
    }

    #[test]
    fn combine_pixels_closure_rgb8() {
        let strategy = ClosureCombine(|a: &Rgb8, b: &Rgb8| {
            Rgb8::new(a.r.0.max(b.r.0), a.g.0.max(b.g.0), a.b.0.max(b.b.0))
        });
        let result = strategy.combine(&Rgb8::new(100, 200, 50), &Rgb8::new(150, 100, 250));
        assert_eq!(result, Rgb8::new(150, 200, 250));
    }

    #[test]
    fn combine_pixels_custom_struct_strategy() {
        struct SatAdd;
        impl CombinePixels<Mono8, Mono8> for SatAdd {
            type Output = Mono8;
            fn combine(&self, a: &Mono8, b: &Mono8) -> Mono8 {
                Mono8::new(a.value().saturating_add(b.value()))
            }
        }

        let result = SatAdd.combine(&Mono8::new(200), &Mono8::new(100));
        assert_eq!(result, Mono8::new(255));
    }

    #[test]
    fn combine_pixels_asymmetric_types() {
        // Combine Mono8 and MonoF32 into MonoF32
        let strategy = ClosureCombine(|a: &Mono8, b: &MonoF32| {
            MonoF32::new(a.value() as f32 / 255.0 + b.value())
        });
        let result = strategy.combine(&Mono8::new(255), &MonoF32::new(0.5));
        assert!((result.value() - 1.5).abs() < 1e-6);
    }

    // ── combine_images / combine_images_into ─────────────────────────────────

    #[test]
    fn combine_images_same_size_mono8() {
        let a = Image::fill(4, 4, Mono8::new(100));
        let b = Image::fill(4, 4, Mono8::new(50));
        let result = combine_images(
            &a,
            &b,
            ClosureCombine(|a: &Mono8, b: &Mono8| Mono8::new(a.value().saturating_add(b.value()))),
        );
        let out = result.unwrap();
        assert_eq!(out.width(), 4);
        assert_eq!(out.height(), 4);
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(out.pixel_at(x, y), Mono8::new(150));
            }
        }
    }

    #[test]
    fn combine_images_size_mismatch_returns_none() {
        let a = Image::fill(3, 3, Mono8::new(100));
        let b = Image::fill(4, 3, Mono8::new(50));
        let result = combine_images(
            &a,
            &b,
            ClosureCombine(|a: &Mono8, b: &Mono8| Mono8::new(a.value().saturating_add(b.value()))),
        );
        assert!(result.is_err());
    }

    #[test]
    fn combine_images_size_mismatch_height_returns_none() {
        let a = Image::fill(3, 3, Mono8::new(100));
        let b = Image::fill(3, 4, Mono8::new(50));
        let result = combine_images(&a, &b, ClosureCombine(|_: &Mono8, _: &Mono8| Mono8::new(0)));
        assert!(result.is_err());
    }

    #[test]
    fn combine_images_into_same_size() {
        let a = Image::fill(3, 3, Mono8::new(200));
        let b = Image::fill(3, 3, Mono8::new(100));
        let mut out = Image::<Mono8>::zero(3, 3);
        let result = combine_images_into(
            &a,
            &b,
            &mut out,
            ClosureCombine(|a: &Mono8, b: &Mono8| *a - *b),
        );
        assert!(result.is_ok());
        for y in 0..3 {
            for x in 0..3 {
                assert_eq!(out.pixel_at(x, y), Mono8::new(100));
            }
        }
    }

    #[test]
    fn combine_images_into_returns_none_on_input_mismatch() {
        let a = Image::fill(3, 3, Mono8::new(100));
        let b = Image::fill(4, 3, Mono8::new(50));
        let mut out = Image::<Mono8>::zero(3, 3);
        let result = combine_images_into(
            &a,
            &b,
            &mut out,
            ClosureCombine(|_: &Mono8, _: &Mono8| Mono8::new(0)),
        );
        assert!(result.is_err());
    }

    #[test]
    #[should_panic(expected = "combine_images_into")]
    fn combine_images_into_panics_on_output_mismatch() {
        let a = Image::fill(3, 3, Mono8::new(100));
        let b = Image::fill(3, 3, Mono8::new(50));
        let mut out = Image::<Mono8>::zero(2, 2); // wrong size
        let _ = combine_images_into(
            &a,
            &b,
            &mut out,
            ClosureCombine(|_: &Mono8, _: &Mono8| Mono8::new(0)),
        );
    }

    #[test]
    fn combine_images_into_matches_allocating_variant() {
        let a = Image::generate(5, 5, |x, y| Mono8::new((x + y * 5) as u8));
        let b = Image::generate(5, 5, |x, y| Mono8::new((x * y) as u8));
        let strategy =
            ClosureCombine(|a: &Mono8, b: &Mono8| Mono8::new(a.value().saturating_add(b.value())));

        let allocating = combine_images(
            &a,
            &b,
            ClosureCombine(|a: &Mono8, b: &Mono8| Mono8::new(a.value().saturating_add(b.value()))),
        )
        .unwrap();

        let mut into = Image::<Mono8>::zero(5, 5);
        combine_images_into(&a, &b, &mut into, strategy).unwrap();

        for y in 0..5 {
            for x in 0..5 {
                assert_eq!(allocating.pixel_at(x, y), into.pixel_at(x, y));
            }
        }
    }

    #[test]
    fn combine_images_zero_size() {
        let a = Image::fill(0, 0, Mono8::new(0));
        let b = Image::fill(0, 0, Mono8::new(0));
        let result = combine_images(&a, &b, ClosureCombine(|_: &Mono8, _: &Mono8| Mono8::new(0)));
        let out = result.unwrap();
        assert_eq!(out.width(), 0);
        assert_eq!(out.height(), 0);
    }

    #[test]
    fn combine_images_1x1() {
        let a = Image::fill(1, 1, Mono8::new(42));
        let b = Image::fill(1, 1, Mono8::new(10));
        let result = combine_images(
            &a,
            &b,
            ClosureCombine(|a: &Mono8, b: &Mono8| Mono8::new(a.value() + b.value())),
        );
        let out = result.unwrap();
        assert_eq!(out.pixel_at(0, 0), Mono8::new(52));
    }

    #[test]
    fn combine_images_with_image_array() {
        let a: ImageArray<Mono8, 3, 3> = ImageArray::generate(|x, y| Mono8::new((x + y) as u8));
        let b: ImageArray<Mono8, 3, 3> = ImageArray::generate(|x, y| Mono8::new((x * y) as u8));
        let result = combine_images(
            &a,
            &b,
            ClosureCombine(|a: &Mono8, b: &Mono8| Mono8::new(a.value().saturating_add(b.value()))),
        );
        let out = result.unwrap();
        assert_eq!(out.width(), 3);
        assert_eq!(out.height(), 3);
        // Spot check: (1,1) -> a=(1+1)=2, b=(1*1)=1, result=3
        assert_eq!(out.pixel_at(1, 1), Mono8::new(3));
    }

    #[test]
    fn combine_images_mixed_image_types() {
        // Image + ImageArray
        let a = Image::fill(3, 3, Mono8::new(10));
        let b: ImageArray<Mono8, 3, 3> = ImageArray::generate(|_, _| Mono8::new(20));
        let result = combine_images(
            &a,
            &b,
            ClosureCombine(|a: &Mono8, b: &Mono8| Mono8::new(a.value() + b.value())),
        );
        let out = result.unwrap();
        for y in 0..3 {
            for x in 0..3 {
                assert_eq!(out.pixel_at(x, y), Mono8::new(30));
            }
        }
    }

    #[test]
    fn combine_images_with_roi_view() {
        let a = Image::generate(6, 6, |x, y| Mono8::new((x + y) as u8));
        let b = Image::generate(6, 6, |_x, _y| Mono8::new(1));
        // Take a 3x3 ROI from each
        let roi_a = a.roi(Rectangle::new((1, 1), (3, 3))).unwrap();
        let roi_b = b.roi(Rectangle::new((2, 2), (3, 3))).unwrap();
        let result = combine_images(
            &roi_a,
            &roi_b,
            ClosureCombine(|a: &Mono8, b: &Mono8| Mono8::new(a.value().saturating_add(b.value()))),
        );
        let out = result.unwrap();
        assert_eq!(out.width(), 3);
        assert_eq!(out.height(), 3);
        // ROI of a at (1,1): pixel (0,0) of roi_a is a(1,1) = (1+1)=2, plus 1 = 3
        assert_eq!(out.pixel_at(0, 0), Mono8::new(3));
    }

    #[test]
    fn combine_images_rgb8_channel_wise() {
        let a = Image::fill(2, 2, Rgb8::new(200, 100, 50));
        let b = Image::fill(2, 2, Rgb8::new(100, 200, 250));
        let result = combine_images(&a, &b, ClosureCombine(|a: &Rgb8, b: &Rgb8| *a + *b));
        let out = result.unwrap();
        // 200+100=255 (sat), 100+200=255 (sat), 50+250=255 (sat)
        assert_eq!(out.pixel_at(0, 0), Rgb8::new(255, 255, 255));
    }

    #[test]
    fn combine_images_varying_pixels() {
        let a = Image::generate(4, 4, |x, y| Mono8::new((x * 10 + y * 40) as u8));
        let b = Image::generate(4, 4, |x, y| Mono8::new((x * 5 + y * 20) as u8));
        let result = combine_images(
            &a,
            &b,
            ClosureCombine(|a: &Mono8, b: &Mono8| Mono8::new(a.value().saturating_add(b.value()))),
        )
        .unwrap();

        for y in 0..4 {
            for x in 0..4 {
                let expected = ((x * 10 + y * 40) + (x * 5 + y * 20)).min(255) as u8;
                assert_eq!(
                    result.pixel_at(x, y).value(),
                    expected,
                    "mismatch at ({}, {})",
                    x,
                    y
                );
            }
        }
    }

    // ── combine_images_fn / combine_images_fn_into ───────────────────────────

    #[test]
    fn combine_images_fn_widening_multiply() {
        let a = Image::fill(3, 3, Mono8::new(200));
        let b = Image::fill(3, 3, Mono8::new(200));
        let result = combine_images_fn(&a, &b, |a: &Mono8, b: &Mono8| {
            Mono32::new(a.value() as u32 * b.value() as u32)
        });
        let out = result.unwrap();
        assert_eq!(out.pixel_at(0, 0), Mono32::new(40000));
    }

    #[test]
    fn combine_images_fn_cross_type() {
        // Mono8 + MonoF32 → MonoF32
        let a = Image::fill(2, 2, Mono8::new(128));
        let b = Image::fill(2, 2, MonoF32::new(0.5));
        let result = combine_images_fn(&a, &b, |a: &Mono8, b: &MonoF32| {
            MonoF32::new(a.value() as f32 / 255.0 + b.value())
        });
        let out = result.unwrap();
        let expected = 128.0 / 255.0 + 0.5;
        assert!((out.pixel_at(0, 0).value() - expected).abs() < 1e-5);
    }

    #[test]
    fn combine_images_fn_same_type_custom_logic() {
        let a = Image::fill(2, 2, Mono8::new(100));
        let b = Image::fill(2, 2, Mono8::new(60));
        let result = combine_images_fn(
            &a,
            &b,
            |a: &Mono8, b: &Mono8| {
                if a.value() > b.value() { *a } else { *b }
            },
        );
        let out = result.unwrap();
        assert_eq!(out.pixel_at(0, 0), Mono8::new(100));
    }

    #[test]
    fn combine_images_fn_size_mismatch_returns_none() {
        let a = Image::fill(3, 3, Mono8::new(0));
        let b = Image::fill(4, 3, Mono8::new(0));
        let result = combine_images_fn(&a, &b, |_: &Mono8, _: &Mono8| Mono8::new(0));
        assert!(result.is_err());
    }

    #[test]
    fn combine_images_fn_into_basic() {
        let a = Image::fill(2, 2, Mono8::new(10));
        let b = Image::fill(2, 2, Mono8::new(20));
        let mut out: Image<Mono32> = Image::zero(2, 2);
        let result = combine_images_fn_into(&a, &b, &mut out, |a: &Mono8, b: &Mono8| {
            Mono32::new(a.value() as u32 + b.value() as u32)
        });
        assert!(result.is_ok());
        assert_eq!(out.pixel_at(0, 0), Mono32::new(30));
    }

    #[test]
    fn combine_images_fn_into_size_mismatch_returns_none() {
        let a = Image::fill(3, 3, Mono8::new(0));
        let b = Image::fill(4, 3, Mono8::new(0));
        let mut out: Image<Mono8> = Image::zero(3, 3);
        let result = combine_images_fn_into(&a, &b, &mut out, |_: &Mono8, _: &Mono8| Mono8::new(0));
        assert!(result.is_err());
    }

    #[test]
    fn combine_images_fn_matches_strategy_variant() {
        struct DoubleAdd;
        impl CombinePixels<Mono8, Mono8> for DoubleAdd {
            type Output = Mono16;
            fn combine(&self, a: &Mono8, b: &Mono8) -> Mono16 {
                Mono16::new(a.value() as u16 + b.value() as u16)
            }
        }

        let a = Image::generate(3, 3, |x, y| Mono8::new((x + y * 3) as u8));
        let b = Image::generate(3, 3, |x, y| Mono8::new((x * y) as u8));

        let strategy_result = combine_images(&a, &b, DoubleAdd).unwrap();
        let closure_result = combine_images_fn(&a, &b, |a: &Mono8, b: &Mono8| {
            Mono16::new(a.value() as u16 + b.value() as u16)
        })
        .unwrap();

        for y in 0..3 {
            for x in 0..3 {
                assert_eq!(
                    strategy_result.pixel_at(x, y),
                    closure_result.pixel_at(x, y),
                    "mismatch at ({}, {})",
                    x,
                    y
                );
            }
        }
    }

    #[test]
    fn combine_images_fn_into_matches_allocating() {
        let a = Image::generate(4, 4, |x, y| Mono8::new((x + y) as u8));
        let b = Image::generate(4, 4, |x, y| Mono8::new((x * y) as u8));

        let allocating = combine_images_fn(&a, &b, |a: &Mono8, b: &Mono8| {
            Mono32::new(a.value() as u32 + b.value() as u32)
        })
        .unwrap();

        let mut into = Image::<Mono32>::zero(4, 4);
        combine_images_fn_into(&a, &b, &mut into, |a: &Mono8, b: &Mono8| {
            Mono32::new(a.value() as u32 + b.value() as u32)
        })
        .unwrap();

        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(allocating.pixel_at(x, y), into.pixel_at(x, y));
            }
        }
    }

    #[test]
    fn combine_images_subtraction_saturating() {
        // Use derived Sub (saturating for Mono8 since it wraps Saturating<u8>)
        let a = Image::fill(2, 2, Mono8::new(100));
        let b = Image::fill(2, 2, Mono8::new(150));
        let result = combine_images_fn(&a, &b, |a: &Mono8, b: &Mono8| *a - *b).unwrap();
        // Saturating: 100 - 150 = 0
        assert_eq!(result.pixel_at(0, 0), Mono8::new(0));
    }

    #[test]
    fn combine_images_rgb8_subtraction() {
        let a = Image::fill(2, 2, Rgb8::new(200, 50, 100));
        let b = Image::fill(2, 2, Rgb8::new(100, 100, 50));
        let result = combine_images_fn(&a, &b, |a: &Rgb8, b: &Rgb8| *a - *b).unwrap();
        // Saturating: (200-100, 50-100, 100-50) = (100, 0, 50)
        assert_eq!(result.pixel_at(0, 0), Rgb8::new(100, 0, 50));
    }

    #[test]
    fn combine_images_monof32_subtraction() {
        let a = Image::fill(2, 2, MonoF32::new(1.0));
        let b = Image::fill(2, 2, MonoF32::new(0.3));
        let result = combine_images_fn(&a, &b, |a: &MonoF32, b: &MonoF32| *a - *b).unwrap();
        assert!((result.pixel_at(0, 0).value() - 0.7).abs() < 1e-6);
    }

    #[test]
    fn combine_images_monof64_subtraction() {
        let a = Image::fill(2, 2, MonoF64::new(1.0));
        let b = Image::fill(2, 2, MonoF64::new(0.3));
        let result = combine_images_fn(&a, &b, |a: &MonoF64, b: &MonoF64| *a - *b).unwrap();
        assert!((result.pixel_at(0, 0).value() - 0.7).abs() < 1e-12);
    }

    #[test]
    fn combine_images_with_roi() {
        let a = Image::generate(6, 6, |x, y| Mono8::new((x + y) as u8));
        let a_roi = a.roi(Rectangle::new((1, 1), (3, 3))).unwrap();
        let b: ImageArray<Mono8, 3, 3> = ImageArray::generate(|_, _| Mono8::new(1));

        let result = combine_images_fn(&a_roi, &b, |a: &Mono8, b: &Mono8| {
            Mono8::new(a.value() + b.value())
        })
        .unwrap();

        assert_eq!(result.width(), 3);
        assert_eq!(result.height(), 3);
        // a_roi(0,0) is a(1,1) = 2, + 1 = 3
        assert_eq!(result.pixel_at(0, 0), Mono8::new(3));
    }

    // ══════════════════════════════════════════════════════════════════════
    // PixelAdd
    // ══════════════════════════════════════════════════════════════════════

    #[test]
    fn add_mono8_saturates() {
        assert_eq!(
            PixelAdd.combine(&Mono8::new(200), &Mono8::new(100)),
            Mono8::new(255)
        );
    }

    #[test]
    fn add_mono8_no_overflow() {
        assert_eq!(
            PixelAdd.combine(&Mono8::new(100), &Mono8::new(50)),
            Mono8::new(150)
        );
    }

    #[test]
    fn add_mono8_zero_identity() {
        assert_eq!(
            PixelAdd.combine(&Mono8::new(42), &Mono8::new(0)),
            Mono8::new(42)
        );
        assert_eq!(
            PixelAdd.combine(&Mono8::new(0), &Mono8::new(42)),
            Mono8::new(42)
        );
    }

    #[test]
    fn add_mono8_max_plus_max() {
        assert_eq!(
            PixelAdd.combine(&Mono8::new(255), &Mono8::new(255)),
            Mono8::new(255)
        );
    }

    #[test]
    fn add_monof32_ieee() {
        let result = PixelAdd.combine(&MonoF32::new(0.5), &MonoF32::new(0.3));
        assert!((result.value() - 0.8).abs() < 1e-6);
    }

    #[test]
    fn add_monof64_ieee() {
        let result = PixelAdd.combine(&MonoF64::new(0.5), &MonoF64::new(0.3));
        assert!((result.value() - 0.8).abs() < 1e-12);
    }

    #[test]
    fn add_rgb8_channel_wise_saturating() {
        // 200+100=255(sat), 100+200=255(sat), 50+250=255(sat)
        let result = PixelAdd.combine(&Rgb8::new(200, 100, 50), &Rgb8::new(100, 200, 250));
        assert_eq!(result, Rgb8::new(255, 255, 255));
    }

    #[test]
    fn add_rgb8_no_overflow() {
        let result = PixelAdd.combine(&Rgb8::new(10, 20, 30), &Rgb8::new(5, 10, 15));
        assert_eq!(result, Rgb8::new(15, 30, 45));
    }

    #[test]
    fn add_image_mono8() {
        let a = Image::fill(3, 3, Mono8::new(100));
        let b = Image::fill(3, 3, Mono8::new(50));
        let out = combine_images(&a, &b, PixelAdd).unwrap();
        assert_eq!(out.width(), 3);
        assert_eq!(out.height(), 3);
        assert_eq!(out.pixel_at(0, 0), Mono8::new(150));
        assert_eq!(out.pixel_at(2, 2), Mono8::new(150));
    }

    #[test]
    fn add_image_size_mismatch_returns_none() {
        let a = Image::fill(4, 4, Mono8::new(10));
        let b = Image::fill(3, 3, Mono8::new(10));
        assert!(combine_images(&a, &b, PixelAdd).is_err());
    }

    // ══════════════════════════════════════════════════════════════════════
    // PixelSubtract
    // ══════════════════════════════════════════════════════════════════════

    #[test]
    fn subtract_mono8_saturates_to_zero() {
        assert_eq!(
            PixelSubtract.combine(&Mono8::new(50), &Mono8::new(100)),
            Mono8::new(0)
        );
    }

    #[test]
    fn subtract_mono8_no_underflow() {
        assert_eq!(
            PixelSubtract.combine(&Mono8::new(200), &Mono8::new(50)),
            Mono8::new(150)
        );
    }

    #[test]
    fn subtract_mono8_both_zero() {
        assert_eq!(
            PixelSubtract.combine(&Mono8::new(0), &Mono8::new(0)),
            Mono8::new(0)
        );
    }

    #[test]
    fn subtract_mono8_same_value() {
        assert_eq!(
            PixelSubtract.combine(&Mono8::new(128), &Mono8::new(128)),
            Mono8::new(0)
        );
    }

    #[test]
    fn subtract_monof32_ieee() {
        let result = PixelSubtract.combine(&MonoF32::new(1.0), &MonoF32::new(0.3));
        assert!((result.value() - 0.7).abs() < 1e-6);
    }

    #[test]
    fn subtract_monof32_goes_negative() {
        // IEEE float: can go below zero
        let result = PixelSubtract.combine(&MonoF32::new(0.3), &MonoF32::new(1.0));
        assert!((result.value() - (-0.7)).abs() < 1e-6);
    }

    #[test]
    fn subtract_rgb8_channel_wise_saturating() {
        // 100-150=0(sat), 200-100=100, 50-100=0(sat)
        let result = PixelSubtract.combine(&Rgb8::new(100, 200, 50), &Rgb8::new(150, 100, 100));
        assert_eq!(result, Rgb8::new(0, 100, 0));
    }

    #[test]
    fn subtract_image_mono8() {
        let a = Image::fill(3, 3, Mono8::new(200));
        let b = Image::fill(3, 3, Mono8::new(50));
        let out = combine_images(&a, &b, PixelSubtract).unwrap();
        assert_eq!(out.pixel_at(0, 0), Mono8::new(150));
    }

    #[test]
    fn subtract_image_size_mismatch_returns_none() {
        let a = Image::fill(4, 4, Mono8::new(100));
        let b = Image::fill(3, 3, Mono8::new(50));
        assert!(combine_images(&a, &b, PixelSubtract).is_err());
    }

    // ══════════════════════════════════════════════════════════════════════
    // PixelMultiply
    // ══════════════════════════════════════════════════════════════════════

    #[test]
    fn multiply_mono8_basic() {
        assert_eq!(
            PixelMultiply.combine(&Mono8::new(10), &Mono8::new(3)),
            Mono8::new(30)
        );
    }

    #[test]
    fn multiply_mono8_saturates() {
        // 100 * 4 = 400, saturates to 255
        assert_eq!(
            PixelMultiply.combine(&Mono8::new(100), &Mono8::new(4)),
            Mono8::new(255)
        );
    }

    #[test]
    fn multiply_mono8_by_zero() {
        assert_eq!(
            PixelMultiply.combine(&Mono8::new(255), &Mono8::new(0)),
            Mono8::new(0)
        );
        assert_eq!(
            PixelMultiply.combine(&Mono8::new(0), &Mono8::new(255)),
            Mono8::new(0)
        );
    }

    #[test]
    fn multiply_mono8_by_one() {
        assert_eq!(
            PixelMultiply.combine(&Mono8::new(123), &Mono8::new(1)),
            Mono8::new(123)
        );
    }

    #[test]
    fn multiply_monof32_ieee() {
        let result = PixelMultiply.combine(&MonoF32::new(0.5), &MonoF32::new(0.5));
        assert!((result.value() - 0.25).abs() < 1e-6);
    }

    #[test]
    fn multiply_monof64_ieee() {
        let result = PixelMultiply.combine(&MonoF64::new(0.5), &MonoF64::new(0.5));
        assert!((result.value() - 0.25).abs() < 1e-12);
    }

    #[test]
    fn multiply_monof32_by_zero() {
        let result = PixelMultiply.combine(&MonoF32::new(0.75), &MonoF32::new(0.0));
        assert_eq!(result.value(), 0.0);
    }

    #[test]
    fn multiply_monof32_by_one() {
        let result = PixelMultiply.combine(&MonoF32::new(0.75), &MonoF32::new(1.0));
        assert!((result.value() - 0.75).abs() < 1e-6);
    }

    #[test]
    fn multiply_image_mono8() {
        let a = Image::fill(3, 3, Mono8::new(5));
        let b = Image::fill(3, 3, Mono8::new(3));
        let out = combine_images(&a, &b, PixelMultiply).unwrap();
        assert_eq!(out.pixel_at(0, 0), Mono8::new(15));
    }

    #[test]
    fn multiply_image_size_mismatch_returns_none() {
        let a = Image::fill(4, 4, Mono8::new(5));
        let b = Image::fill(3, 3, Mono8::new(3));
        assert!(combine_images(&a, &b, PixelMultiply).is_err());
    }

    // ══════════════════════════════════════════════════════════════════════
    // AbsDiff
    // ══════════════════════════════════════════════════════════════════════

    #[test]
    fn abs_diff_mono8_a_greater() {
        assert_eq!(
            AbsDiff.combine(&Mono8::new(150), &Mono8::new(100)),
            Mono8::new(50)
        );
    }

    #[test]
    fn abs_diff_mono8_b_greater() {
        assert_eq!(
            AbsDiff.combine(&Mono8::new(100), &Mono8::new(150)),
            Mono8::new(50)
        );
    }

    #[test]
    fn abs_diff_mono8_max_range() {
        assert_eq!(
            AbsDiff.combine(&Mono8::new(0), &Mono8::new(255)),
            Mono8::new(255)
        );
        assert_eq!(
            AbsDiff.combine(&Mono8::new(255), &Mono8::new(0)),
            Mono8::new(255)
        );
    }

    #[test]
    fn abs_diff_mono8_same_value() {
        assert_eq!(
            AbsDiff.combine(&Mono8::new(128), &Mono8::new(128)),
            Mono8::new(0)
        );
        assert_eq!(
            AbsDiff.combine(&Mono8::new(0), &Mono8::new(0)),
            Mono8::new(0)
        );
    }

    #[test]
    fn abs_diff_mono8_commutative() {
        for v in [0u8, 1, 100, 127, 128, 200, 254, 255] {
            for u in [0u8, 1, 100, 127, 128, 200, 254, 255] {
                let a = Mono8::new(v);
                let b = Mono8::new(u);
                assert_eq!(
                    AbsDiff.combine(&a, &b),
                    AbsDiff.combine(&b, &a),
                    "commutativity failed for v={v}, u={u}"
                );
            }
        }
    }

    #[test]
    fn abs_diff_monof32_a_greater() {
        let result = AbsDiff.combine(&MonoF32::new(1.0), &MonoF32::new(0.3));
        assert!((result.value() - 0.7).abs() < 1e-6);
    }

    #[test]
    fn abs_diff_monof32_b_greater() {
        let result = AbsDiff.combine(&MonoF32::new(0.3), &MonoF32::new(1.0));
        assert!((result.value() - 0.7).abs() < 1e-6);
    }

    #[test]
    fn abs_diff_monof32_negative_inputs() {
        // |(-0.5) - 0.5| = 1.0
        let result = AbsDiff.combine(&MonoF32::new(-0.5), &MonoF32::new(0.5));
        assert!((result.value() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn abs_diff_monof64() {
        let result = AbsDiff.combine(&MonoF64::new(2.0), &MonoF64::new(3.5));
        assert!((result.value() - 1.5).abs() < 1e-12);
    }

    #[test]
    fn abs_diff_mono16() {
        assert_eq!(
            AbsDiff.combine(&Mono16::new(1000), &Mono16::new(300)),
            Mono16::new(700)
        );
        assert_eq!(
            AbsDiff.combine(&Mono16::new(300), &Mono16::new(1000)),
            Mono16::new(700)
        );
    }

    #[test]
    fn abs_diff_rgb8_channel_wise() {
        // |200-100|=100, |50-150|=100, |100-100|=0
        let result = AbsDiff.combine(&Rgb8::new(200, 50, 100), &Rgb8::new(100, 150, 100));
        assert_eq!(result, Rgb8::new(100, 100, 0));
    }

    #[test]
    fn abs_diff_image_mono8() {
        let a = Image::fill(3, 3, Mono8::new(200));
        let b = Image::fill(3, 3, Mono8::new(50));
        let out = combine_images(&a, &b, AbsDiff).unwrap();
        assert_eq!(out.pixel_at(0, 0), Mono8::new(150));
    }

    #[test]
    fn abs_diff_image_reversed_same_result() {
        // |a - b| == |b - a|
        let a = Image::fill(3, 3, Mono8::new(50));
        let b = Image::fill(3, 3, Mono8::new(200));
        let out = combine_images(&a, &b, AbsDiff).unwrap();
        assert_eq!(out.pixel_at(0, 0), Mono8::new(150));
    }

    #[test]
    fn abs_diff_image_size_mismatch_returns_none() {
        let a = Image::fill(4, 4, Mono8::new(100));
        let b = Image::fill(3, 3, Mono8::new(50));
        assert!(combine_images(&a, &b, AbsDiff).is_err());
    }

    // ══════════════════════════════════════════════════════════════════════
    // Max
    // ══════════════════════════════════════════════════════════════════════

    #[test]
    fn max_mono8_picks_larger() {
        assert_eq!(
            Max.combine(&Mono8::new(100), &Mono8::new(200)),
            Mono8::new(200)
        );
        assert_eq!(
            Max.combine(&Mono8::new(200), &Mono8::new(100)),
            Mono8::new(200)
        );
    }

    #[test]
    fn max_mono8_equal_values() {
        assert_eq!(
            Max.combine(&Mono8::new(128), &Mono8::new(128)),
            Mono8::new(128)
        );
    }

    #[test]
    fn max_mono8_extremes() {
        assert_eq!(
            Max.combine(&Mono8::new(0), &Mono8::new(255)),
            Mono8::new(255)
        );
        assert_eq!(
            Max.combine(&Mono8::new(255), &Mono8::new(0)),
            Mono8::new(255)
        );
    }

    #[test]
    fn max_mono16() {
        assert_eq!(
            Max.combine(&Mono16::new(500), &Mono16::new(1000)),
            Mono16::new(1000)
        );
        assert_eq!(
            Max.combine(&Mono16::new(1000), &Mono16::new(500)),
            Mono16::new(1000)
        );
    }

    #[test]
    fn max_rgb8_channel_wise() {
        // max(200,100)=200, max(50,150)=150, max(100,100)=100
        let result = Max.combine(&Rgb8::new(200, 50, 100), &Rgb8::new(100, 150, 100));
        assert_eq!(result, Rgb8::new(200, 150, 100));
    }

    #[test]
    fn max_image_mono8() {
        let a = Image::fill(3, 3, Mono8::new(100));
        let b = Image::fill(3, 3, Mono8::new(200));
        let out = combine_images(&a, &b, Max).unwrap();
        assert_eq!(out.pixel_at(0, 0), Mono8::new(200));
    }

    #[test]
    fn max_image_size_mismatch_returns_none() {
        let a = Image::fill(4, 4, Mono8::new(100));
        let b = Image::fill(3, 3, Mono8::new(200));
        assert!(combine_images(&a, &b, Max).is_err());
    }

    // ══════════════════════════════════════════════════════════════════════
    // Min
    // ══════════════════════════════════════════════════════════════════════

    #[test]
    fn min_mono8_picks_smaller() {
        assert_eq!(
            Min.combine(&Mono8::new(100), &Mono8::new(200)),
            Mono8::new(100)
        );
        assert_eq!(
            Min.combine(&Mono8::new(200), &Mono8::new(100)),
            Mono8::new(100)
        );
    }

    #[test]
    fn min_mono8_equal_values() {
        assert_eq!(
            Min.combine(&Mono8::new(128), &Mono8::new(128)),
            Mono8::new(128)
        );
    }

    #[test]
    fn min_mono8_extremes() {
        assert_eq!(Min.combine(&Mono8::new(0), &Mono8::new(255)), Mono8::new(0));
        assert_eq!(Min.combine(&Mono8::new(255), &Mono8::new(0)), Mono8::new(0));
    }

    #[test]
    fn min_mono16() {
        assert_eq!(
            Min.combine(&Mono16::new(500), &Mono16::new(1000)),
            Mono16::new(500)
        );
        assert_eq!(
            Min.combine(&Mono16::new(1000), &Mono16::new(500)),
            Mono16::new(500)
        );
    }

    #[test]
    fn min_rgb8_channel_wise() {
        // min(200,100)=100, min(50,150)=50, min(100,100)=100
        let result = Min.combine(&Rgb8::new(200, 50, 100), &Rgb8::new(100, 150, 100));
        assert_eq!(result, Rgb8::new(100, 50, 100));
    }

    #[test]
    fn min_image_mono8() {
        let a = Image::fill(3, 3, Mono8::new(100));
        let b = Image::fill(3, 3, Mono8::new(200));
        let out = combine_images(&a, &b, Min).unwrap();
        assert_eq!(out.pixel_at(0, 0), Mono8::new(100));
    }

    #[test]
    fn min_image_size_mismatch_returns_none() {
        let a = Image::fill(4, 4, Mono8::new(100));
        let b = Image::fill(3, 3, Mono8::new(200));
        assert!(combine_images(&a, &b, Min).is_err());
    }

    #[test]
    fn max_min_ordering_invariant() {
        // max(a, b) >= min(a, b) for all a, b
        for v in [0u8, 50, 100, 200, 255] {
            for u in [0u8, 50, 100, 200, 255] {
                let a = Mono8::new(v);
                let b = Mono8::new(u);
                let mx = Max.combine(&a, &b);
                let mn = Min.combine(&a, &b);
                assert!(
                    mx.value() >= mn.value(),
                    "max={} < min={} for v={v}, u={u}",
                    mx.value(),
                    mn.value()
                );
            }
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // LinearCombine
    // ══════════════════════════════════════════════════════════════════════

    #[test]
    fn linear_combine_half_half_mono8() {
        // 0.5·100 + 0.5·200 = 150
        let result = LinearCombine { wa: 0.5, wb: 0.5 }.combine(&Mono8::new(100), &Mono8::new(200));
        assert!((result - MonoF32(150.0)).abs().0 < 1e-5);
    }

    #[test]
    fn linear_combine_passthrough_a() {
        // wa=1, wb=0 → result equals a in accumulator space
        let result = LinearCombine { wa: 1.0, wb: 0.0 }.combine(&Mono8::new(80), &Mono8::new(200));
        assert!((result - MonoF32(80.0)).abs().0 < 1e-5);
    }

    #[test]
    fn linear_combine_passthrough_b() {
        // wa=0, wb=1 → result equals b in accumulator space
        let result = LinearCombine { wa: 0.0, wb: 1.0 }.combine(&Mono8::new(80), &Mono8::new(200));
        assert!((result - MonoF32(200.0)).abs().0 < 1e-5);
    }

    #[test]
    fn linear_combine_double_a() {
        // wa=2.0, wb=0.0 → 2·a
        let result = LinearCombine { wa: 2.0, wb: 0.0 }.combine(&Mono8::new(50), &Mono8::new(0));
        assert!((result - MonoF32(100.0)).abs().0 < 1e-5);
    }

    #[test]
    fn linear_combine_equals_blend() {
        // LinearCombine { wa: 1-alpha, wb: alpha } == Blend { alpha }
        let a = Mono8::new(40);
        let b = Mono8::new(160);
        let alpha = 0.75_f32;
        let lc = LinearCombine {
            wa: 1.0 - alpha,
            wb: alpha,
        }
        .combine(&a, &b);
        let bl = Blend { alpha }.combine(&a, &b);
        assert!((lc - bl).abs().0 < 1e-5);
    }

    #[test]
    fn linear_combine_rgbf32() {
        let a = RgbF32::new(0.0, 0.5, 1.0);
        let b = RgbF32::new(1.0, 0.5, 0.0);
        // 0.5·a + 0.5·b = midpoint per channel
        let result = LinearCombine { wa: 0.5, wb: 0.5 }.combine(&a, &b);
        assert!((result.r - 0.5).abs() < 1e-6);
        assert!((result.g - 0.5).abs() < 1e-6);
        assert!((result.b - 0.5).abs() < 1e-6);
    }

    #[test]
    fn linear_combine_image_mono8() {
        let a = Image::fill(3, 3, Mono8::new(100));
        let b = Image::fill(3, 3, Mono8::new(200));
        // `Mono8::Accumulator = MonoF32`, so the
        // combined image is `Image<MonoF32>` rather than `Image<f32>`.
        let out: Image<MonoF32> =
            combine_images(&a, &b, LinearCombine { wa: 0.5, wb: 0.5 }).unwrap();
        assert!((out.pixel_at(0, 0) - MonoF32(150.0)).abs().0 < 1e-5);
    }

    #[test]
    fn linear_combine_image_size_mismatch_returns_none() {
        let a = Image::fill(4, 4, Mono8::new(100));
        let b = Image::fill(3, 3, Mono8::new(100));
        assert!(combine_images(&a, &b, LinearCombine { wa: 0.5, wb: 0.5 }).is_err());
    }

    // ══════════════════════════════════════════════════════════════════════
    // Blend
    // ══════════════════════════════════════════════════════════════════════

    #[test]
    fn blend_alpha_zero_returns_a() {
        let result = Blend { alpha: 0.0 }.combine(&Mono8::new(100), &Mono8::new(200));
        assert!((result - MonoF32(100.0)).abs().0 < 1e-5);
    }

    #[test]
    fn blend_alpha_one_returns_b() {
        let result = Blend { alpha: 1.0 }.combine(&Mono8::new(100), &Mono8::new(200));
        assert!((result - MonoF32(200.0)).abs().0 < 1e-5);
    }

    #[test]
    fn blend_midpoint_mono8() {
        // (1-0.5)·0 + 0.5·200 = 100
        let result = Blend { alpha: 0.5 }.combine(&Mono8::new(0), &Mono8::new(200));
        assert!((result - MonoF32(100.0)).abs().0 < 1e-5);
    }

    #[test]
    fn blend_monof32_alpha_half() {
        // MonoF32 is a self-accumulator, so result is MonoF32
        let result = Blend { alpha: 0.5 }.combine(&MonoF32::new(0.0), &MonoF32::new(1.0));
        assert!((result.value() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn blend_rgbf32() {
        let black = RgbF32::new(0.0, 0.0, 0.0);
        let white = RgbF32::new(1.0, 1.0, 1.0);
        let result = Blend { alpha: 0.25 }.combine(&black, &white);
        assert!((result.r - 0.25).abs() < 1e-6);
        assert!((result.g - 0.25).abs() < 1e-6);
        assert!((result.b - 0.25).abs() < 1e-6);
    }

    #[test]
    fn blend_rgb8_accumulator_is_rgbf32() {
        // Rgb8 accumulator = RgbF32; verify channel values
        let a = Rgb8::new(0, 100, 200);
        let b = Rgb8::new(100, 0, 0);
        let result = Blend { alpha: 0.5 }.combine(&a, &b);
        // 0.5·0+0.5·100=50, 0.5·100+0.5·0=50, 0.5·200+0.5·0=100
        assert!((result.r - 50.0).abs() < 1.0);
        assert!((result.g - 50.0).abs() < 1.0);
        assert!((result.b - 100.0).abs() < 1.0);
    }

    #[test]
    fn blend_image_mono8() {
        let a = Image::fill(3, 3, Mono8::new(0));
        let b = Image::fill(3, 3, Mono8::new(200));
        // accumulator is `MonoF32`, not raw `f32`.
        let out: Image<MonoF32> = combine_images(&a, &b, Blend { alpha: 0.5 }).unwrap();
        assert!((out.pixel_at(1, 1) - MonoF32(100.0)).abs().0 < 1e-5);
    }

    #[test]
    fn blend_image_size_mismatch_returns_none() {
        let a = Image::fill(4, 4, Mono8::new(0));
        let b = Image::fill(3, 3, Mono8::new(200));
        assert!(combine_images(&a, &b, Blend { alpha: 0.5 }).is_err());
    }

    // ══════════════════════════════════════════════════════════════════════
    // Magnitude
    // ══════════════════════════════════════════════════════════════════════

    #[test]
    fn magnitude_monof32_pythagorean_3_4_5() {
        let result = Magnitude.combine(&MonoF32::new(3.0), &MonoF32::new(4.0));
        assert!((result.value() - 5.0).abs() < 1e-5);
    }

    #[test]
    fn magnitude_monof32_unit_axes() {
        // hypot(1, 0) = 1, hypot(0, 1) = 1
        let r1 = Magnitude.combine(&MonoF32::new(1.0), &MonoF32::new(0.0));
        let r2 = Magnitude.combine(&MonoF32::new(0.0), &MonoF32::new(1.0));
        assert!((r1.value() - 1.0).abs() < 1e-6);
        assert!((r2.value() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn magnitude_monof32_zeros() {
        let result = Magnitude.combine(&MonoF32::new(0.0), &MonoF32::new(0.0));
        assert_eq!(result.value(), 0.0);
    }

    #[test]
    fn magnitude_monof32_commutative() {
        let r1 = Magnitude.combine(&MonoF32::new(3.0), &MonoF32::new(4.0));
        let r2 = Magnitude.combine(&MonoF32::new(4.0), &MonoF32::new(3.0));
        assert!((r1.value() - r2.value()).abs() < 1e-6);
    }

    #[test]
    fn magnitude_monof64_pythagorean_5_12_13() {
        let result = Magnitude.combine(&MonoF64::new(5.0), &MonoF64::new(12.0));
        assert!((result.value() - 13.0).abs() < 1e-12);
    }

    #[test]
    fn magnitude_rgbf32_channel_wise() {
        // hypot(3,4)=5, hypot(0,1)=1, hypot(1,0)=1
        let a = RgbF32::new(3.0, 0.0, 1.0);
        let b = RgbF32::new(4.0, 1.0, 0.0);
        let result = Magnitude.combine(&a, &b);
        assert!((result.r - 5.0).abs() < 1e-5);
        assert!((result.g - 1.0).abs() < 1e-5);
        assert!((result.b - 1.0).abs() < 1e-5);
    }

    #[test]
    fn magnitude_image_monof32_sobel_gradient() {
        // Simulate combining Gx and Gy into magnitude
        let gx = Image::fill(3, 3, MonoF32::new(3.0));
        let gy = Image::fill(3, 3, MonoF32::new(4.0));
        let mag = combine_images(&gx, &gy, Magnitude).unwrap();
        assert!((mag.pixel_at(1, 1).value() - 5.0).abs() < 1e-5);
        assert!((mag.pixel_at(0, 0).value() - 5.0).abs() < 1e-5);
    }

    #[test]
    fn magnitude_image_size_mismatch_returns_none() {
        let gx = Image::fill(4, 4, MonoF32::new(1.0));
        let gy = Image::fill(3, 3, MonoF32::new(1.0));
        assert!(combine_images(&gx, &gy, Magnitude).is_err());
    }

    // ══════════════════════════════════════════════════════════════════════
    // MagnitudeChannel helper — direct unit tests
    // ══════════════════════════════════════════════════════════════════════

    #[test]
    fn magnitude_channel_f32_3_4_5() {
        assert!((MagnitudeChannel::magnitude(3.0_f32, 4.0_f32) - 5.0).abs() < 1e-5);
    }

    #[test]
    fn magnitude_channel_f32_commutative() {
        let r1 = MagnitudeChannel::magnitude(3.0_f32, 4.0_f32);
        let r2 = MagnitudeChannel::magnitude(4.0_f32, 3.0_f32);
        assert!((r1 - r2).abs() < 1e-6);
    }

    #[test]
    fn magnitude_channel_f32_zero_inputs() {
        assert_eq!(MagnitudeChannel::magnitude(0.0_f32, 0.0_f32), 0.0_f32);
        assert!((MagnitudeChannel::magnitude(0.0_f32, 1.0_f32) - 1.0).abs() < 1e-6);
        assert!((MagnitudeChannel::magnitude(1.0_f32, 0.0_f32) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn magnitude_channel_f64_5_12_13() {
        assert!((MagnitudeChannel::magnitude(5.0_f64, 12.0_f64) - 13.0).abs() < 1e-12);
    }

    // ══════════════════════════════════════════════════════════════════════
    // Cross-strategy consistency checks
    // ══════════════════════════════════════════════════════════════════════

    #[test]
    fn blend_is_special_case_of_linear_combine() {
        for alpha in [0.0_f32, 0.25, 0.5, 0.75, 1.0] {
            let a = Mono8::new(40);
            let b = Mono8::new(200);
            let blend_result = Blend { alpha }.combine(&a, &b);
            let lc_result = LinearCombine {
                wa: 1.0 - alpha,
                wb: alpha,
            }
            .combine(&a, &b);
            assert!(
                (blend_result - lc_result).abs().0 < 1e-5,
                "Blend and LinearCombine disagree at alpha={alpha}"
            );
        }
    }

    #[test]
    fn abs_diff_always_nonnegative_mono8() {
        for v in [0u8, 1, 50, 127, 200, 255] {
            for u in [0u8, 1, 50, 127, 200, 255] {
                let result = AbsDiff.combine(&Mono8::new(v), &Mono8::new(u));
                assert_eq!(result.value(), v.abs_diff(u), "v={v}, u={u}");
            }
        }
    }

    #[test]
    fn max_ge_min_for_all_mono8_pairs() {
        for v in [0u8, 50, 128, 255] {
            for u in [0u8, 50, 128, 255] {
                let a = Mono8::new(v);
                let b = Mono8::new(u);
                assert!(Max.combine(&a, &b).value() >= Min.combine(&a, &b).value());
            }
        }
    }

    #[test]
    fn subtract_then_abs_diff_consistent() {
        // For a >= b: PixelSubtract(a, b) == AbsDiff(a, b)
        let a = Mono8::new(200);
        let b = Mono8::new(50);
        assert_eq!(PixelSubtract.combine(&a, &b), AbsDiff.combine(&a, &b));
    }
}

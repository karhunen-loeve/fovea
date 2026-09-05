//! Convenience filter functions built on [`convolve`] and
//! [`convolve_separable`].
//!
//! Each function selects the appropriate kernel (and separable
//! decomposition where available), applies sensible defaults, and returns
//! a newly allocated output image.
//!
//! All functions that produce edge / gradient output return
//! `Image<P::Accumulator>` (the input pixel's linear accumulator type,
//! e.g. `MonoF32` for `Mono8`) to avoid truncation — the caller can
//! convert to the desired output type afterwards. Gradient outputs
//! from edge detectors are conventionally treated as pixel-role
//! images (spatial grids carrying signed intensity).
//!
//! Functions that produce blurred / sharpened output preserve the input
//! pixel type by default (using [`FromLinear`] for the final conversion).

use crate::border::BorderPolicy;
use crate::error::Error;
use crate::image::{
    Image, ImageRef, Neighborhood, RasterImage, RasterImageMut, SeparableKernel, gaussian_kernel_1d,
};
use crate::pixel::{FromLinear, HomogeneousPixel, LinearPixel, ZeroablePixel};
use crate::transform::combine::{
    Direction, DirectionChannel, Magnitude, MagnitudeChannel, combine_images,
};
use crate::transform::convolve::convolve;
use crate::transform::convolve_separable::{
    SeparableScratch, convolve_separable, convolve_separable_into,
};
use crate::{Offset, Sigma};

// ─── Box blur ────────────────────────────────────────────────────────────────

/// 3×3 box blur using a separable two-pass implementation.
///
/// Each weight is `1/3`, applied horizontally then vertically, giving an
/// effective `1/9` per pixel — identical to [`Neighborhood::box_blur_3x3`].
///
/// Sugar for `convolve_separable(image, &SeparableKernel::box_blur_3(), border)` —
/// the kernel is the variant, and this spelling is the common case.
/// Equivalence pinned by `fixed_size_blurs_equal_their_kernel_form`.
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView};
/// use fovea::border::Clamp;
/// use fovea::pixel::Mono8;
/// use fovea::transform::box_blur_3x3;
///
/// let src = Image::fill(8, 8, Mono8::new(100));
/// let result: Image<Mono8> = box_blur_3x3(&src, &Clamp);
///
/// for y in 0..result.height() {
///     for x in 0..result.width() {
///         assert_eq!(result.pixel_at(x, y), Mono8::new(100));
///     }
/// }
/// ```
#[must_use]
pub fn box_blur_3x3<I, B, P, Acc, Out>(image: &I, border: &B) -> Image<Out>
where
    I: RasterImage<Pixel = P>,
    P: Copy + LinearPixel<f32, Accumulator = Acc>,
    Acc: Copy
        + Default
        + ZeroablePixel
        + LinearPixel<f32, Accumulator = Acc>
        + std::ops::Add<Output = Acc>,
    B: BorderPolicy<I> + BorderPolicy<Image<Acc>>,
    Out: ZeroablePixel + FromLinear<Acc>,
{
    convolve_separable(image, &SeparableKernel::box_blur_3(), border)
}

/// 5×5 box blur using a separable two-pass implementation.
///
/// Each weight is `1/5`, applied horizontally then vertically, giving an
/// effective `1/25` per pixel — identical to [`Neighborhood::box_blur_5x5`].
///
/// Sugar for `convolve_separable(image, &SeparableKernel::box_blur_5(), border)` —
/// the kernel is the variant, and this spelling is the common case.
/// Equivalence pinned by `fixed_size_blurs_equal_their_kernel_form`.
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView};
/// use fovea::border::Clamp;
/// use fovea::pixel::Mono8;
/// use fovea::transform::box_blur_5x5;
///
/// let src = Image::fill(10, 10, Mono8::new(50));
/// let result: Image<Mono8> = box_blur_5x5(&src, &Clamp);
///
/// for y in 0..result.height() {
///     for x in 0..result.width() {
///         assert_eq!(result.pixel_at(x, y), Mono8::new(50));
///     }
/// }
/// ```
#[must_use]
pub fn box_blur_5x5<I, B, P, Acc, Out>(image: &I, border: &B) -> Image<Out>
where
    I: RasterImage<Pixel = P>,
    P: Copy + LinearPixel<f32, Accumulator = Acc>,
    Acc: Copy
        + Default
        + ZeroablePixel
        + LinearPixel<f32, Accumulator = Acc>
        + std::ops::Add<Output = Acc>,
    B: BorderPolicy<I> + BorderPolicy<Image<Acc>>,
    Out: ZeroablePixel + FromLinear<Acc>,
{
    convolve_separable(image, &SeparableKernel::box_blur_5(), border)
}

// ─── Gaussian blur ───────────────────────────────────────────────────────────

/// 3×3 Gaussian blur using a separable two-pass implementation.
///
/// Uses the **normalized** `[0.25, 0.5, 0.25]` kernel (`[1, 2, 1] / 4`) in
/// each direction. The combined 2D kernel sums to 1, so the blur
/// **preserves brightness** (a flat input is returned unchanged), matching
/// every mainstream library and the sibling [`box_blur_3x3`].
///
/// If you instead need the raw integer `[1, 2, 1]` kernel (sum 16) — for
/// example to reproduce a legacy ×16-scaled result — convolve directly
/// with [`Neighborhood::gaussian_3x3`], where the caller owns the scale.
///
/// Sugar for `convolve_separable(image, &SeparableKernel::gaussian_3(), border)` —
/// the kernel is the variant, and this spelling is the common case.
/// Equivalence pinned by `fixed_size_blurs_equal_their_kernel_form`.
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView};
/// use fovea::border::Clamp;
/// use fovea::pixel::MonoF32;
/// use fovea::transform::gaussian_blur_3x3;
///
/// // the pixel role for floats is `MonoF32`,
/// // not raw `f32`. `MonoF32` is `#[repr(transparent)]` over `f32`.
/// let src = Image::fill(8, 8, MonoF32::new(1.0));
/// let result: Image<MonoF32> = gaussian_blur_3x3(&src, &Clamp);
///
/// // Normalized: brightness preserved, 1.0 → 1.0.
/// for y in 0..result.height() {
///     for x in 0..result.width() {
///         assert!((result.pixel_at(x, y).0 - 1.0).abs() < 1e-4);
///     }
/// }
/// ```
#[must_use]
pub fn gaussian_blur_3x3<I, B, P, Acc, Out>(image: &I, border: &B) -> Image<Out>
where
    I: RasterImage<Pixel = P>,
    P: Copy + LinearPixel<f32, Accumulator = Acc>,
    Acc: Copy
        + Default
        + ZeroablePixel
        + LinearPixel<f32, Accumulator = Acc>
        + std::ops::Add<Output = Acc>,
    B: BorderPolicy<I> + BorderPolicy<Image<Acc>>,
    Out: ZeroablePixel + FromLinear<Acc>,
{
    convolve_separable(image, &SeparableKernel::gaussian_3(), border)
}

/// 5×5 Gaussian blur using a separable two-pass implementation.
///
/// Uses the **normalized** `[0.0625, 0.25, 0.375, 0.25, 0.0625]` kernel
/// (`[1, 4, 6, 4, 1] / 16`) in each direction. The combined 2D kernel sums
/// to 1, so the blur **preserves brightness** (a flat input is returned
/// unchanged), matching every mainstream library and the sibling
/// [`box_blur_5x5`].
///
/// If you instead need the raw integer `[1, 4, 6, 4, 1]` kernel (sum 256),
/// convolve directly with [`Neighborhood::gaussian_5x5`], where the caller
/// owns the scale.
///
/// Sugar for `convolve_separable(image, &SeparableKernel::gaussian_5(), border)` —
/// the kernel is the variant, and this spelling is the common case.
/// Equivalence pinned by `fixed_size_blurs_equal_their_kernel_form`.
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView};
/// use fovea::border::Clamp;
/// use fovea::pixel::MonoF32;
/// use fovea::transform::gaussian_blur_5x5;
///
/// // the pixel role for floats is `MonoF32`.
/// let src = Image::fill(10, 10, MonoF32::new(1.0));
/// let result: Image<MonoF32> = gaussian_blur_5x5(&src, &Clamp);
///
/// // Normalized: brightness preserved, 1.0 → 1.0.
/// for y in 0..result.height() {
///     for x in 0..result.width() {
///         assert!((result.pixel_at(x, y).0 - 1.0).abs() < 1e-4);
///     }
/// }
/// ```
#[must_use]
pub fn gaussian_blur_5x5<I, B, P, Acc, Out>(image: &I, border: &B) -> Image<Out>
where
    I: RasterImage<Pixel = P>,
    P: Copy + LinearPixel<f32, Accumulator = Acc>,
    Acc: Copy
        + Default
        + ZeroablePixel
        + LinearPixel<f32, Accumulator = Acc>
        + std::ops::Add<Output = Acc>,
    B: BorderPolicy<I> + BorderPolicy<Image<Acc>>,
    Out: ZeroablePixel + FromLinear<Acc>,
{
    convolve_separable(image, &SeparableKernel::gaussian_5(), border)
}

// ─── Parameterized Gaussian blur ───────────────────────────────────────────────

/// Default `truncate` for [`gaussian_blur`] / [`gaussian_blur_into`].
///
/// The kernel radius is `round(truncate * sigma)`. The value `4.0` matches
/// SciPy (`scipy.ndimage.gaussian_filter`) and scikit-image
/// (`filters.gaussian`), capturing the Gaussian out to 4σ. To pick a smaller
/// `truncate` (e.g. `3.0`) for a faster, slightly tighter kernel, build the
/// kernel with
/// [`gaussian_kernel_1d`](crate::image::gaussian_kernel_1d) and pass it to
/// [`convolve_separable`].
pub const DEFAULT_TRUNCATE: f32 = 4.0;

/// Gaussian blur with a separable kernel derived from `sigma`.
///
/// The kernel radius is `round(`[`DEFAULT_TRUNCATE`]` * sigma)` and the
/// weights are a normalized 1-D Gaussian (sum 1), applied separably, so
/// the blur **preserves brightness**. Unlike the fixed [`gaussian_blur_3x3`]
/// / [`gaussian_blur_5x5`] paths, this expresses an arbitrary blur amount;
/// the derived odd kernel size is reported by
/// [`gaussian_kernel_size`](crate::image::gaussian_kernel_size).
///
/// To choose `truncate` yourself, build the kernel and convolve with it —
/// `convolve_separable(image, &gaussian_kernel_1d(sigma, 3.0), border)`. The
/// kernel is the variant, so there is no second blur function for it.
///
/// σ is the invariant-carrying [`Sigma`] type: literals use
/// [`sigma!`](crate::sigma), which checks them at compile time, computed
/// values use [`Sigma::try_new`] and handle the error where the value was
/// produced. This function itself cannot see an invalid σ.
///
/// # Panics
///
/// Panics if the derived radius exceeds
/// [`MAX_RADIUS`](crate::image::MAX_RADIUS) (i.e. `sigma` is larger than
/// `MAX_RADIUS / truncate`) — a capacity bound of the stack-allocated
/// kernel, testable up front with
/// [`gaussian_kernel_size`](crate::image::gaussian_kernel_size).
///
/// # Example
///
/// ```
/// use fovea::border::Clamp;
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::MonoF32;
/// use fovea::sigma;
/// use fovea::transform::gaussian_blur;
///
/// // A flat image is returned unchanged (brightness preserved).
/// let src = Image::fill(16, 16, MonoF32::new(0.7));
/// let result: Image<MonoF32> = gaussian_blur(&src, sigma!(2.0), &Clamp);
/// for y in 0..result.height() {
///     for x in 0..result.width() {
///         assert!((result.pixel_at(x, y).0 - 0.7).abs() < 1e-4);
///     }
/// }
/// ```
#[must_use]
pub fn gaussian_blur<I, B, P, Acc, Out>(image: &I, sigma: Sigma, border: &B) -> Image<Out>
where
    I: RasterImage<Pixel = P>,
    P: Copy + LinearPixel<f32, Accumulator = Acc>,
    Acc: Copy
        + Default
        + ZeroablePixel
        + LinearPixel<f32, Accumulator = Acc>
        + std::ops::Add<Output = Acc>,
    B: BorderPolicy<I> + BorderPolicy<Image<Acc>>,
    Out: ZeroablePixel + FromLinear<Acc>,
{
    convolve_separable(image, &gaussian_kernel_1d(sigma, DEFAULT_TRUNCATE), border)
}

/// Gaussian blur derived from `sigma`, writing into a caller-owned output.
///
/// As [`gaussian_blur`], but writes into `output` instead of allocating.
///
/// # Panics
///
/// Panics if the derived radius exceeds
/// [`MAX_RADIUS`](crate::image::MAX_RADIUS), or if `output` is too small for
/// the region produced by the border policy.
pub fn gaussian_blur_into<I, B, O, P, Acc, Out>(image: &I, sigma: Sigma, border: &B, output: &mut O)
where
    I: RasterImage<Pixel = P>,
    P: Copy + LinearPixel<f32, Accumulator = Acc>,
    Acc: Copy
        + Default
        + ZeroablePixel
        + LinearPixel<f32, Accumulator = Acc>
        + std::ops::Add<Output = Acc>,
    B: BorderPolicy<I> + BorderPolicy<Image<Acc>>,
    O: RasterImageMut<Pixel = Out>,
    Out: FromLinear<Acc>,
{
    convolve_separable_into(
        image,
        &gaussian_kernel_1d(sigma, DEFAULT_TRUNCATE),
        border,
        output,
    );
}

impl<Acc> SeparableScratch<Acc>
where
    Acc: Copy
        + Default
        + ZeroablePixel
        + LinearPixel<f32, Accumulator = Acc>
        + std::ops::Add<Output = Acc>,
{
    /// Gaussian blur derived from `sigma`, writing into a caller-owned
    /// output and reusing this scratch.
    ///
    /// As the free [`gaussian_blur_into`], but the inter-pass intermediate
    /// and the engine's working buffers come from `self`. With a
    /// caller-owned `output` as well, a blur in a hot loop performs **no
    /// heap allocation after the first call** — the case pyramid and
    /// scale-space construction hit `log₂(N)` and `octaves × (S + 3)` times
    /// per image.
    ///
    /// To choose `truncate` yourself, build the kernel and use
    /// [`convolve_separable_into`](Self::convolve_separable_into) —
    /// `scratch.convolve_separable_into(&src, &gaussian_kernel_1d(sigma, 3.0),
    /// &border, &mut out)`.
    ///
    /// # Panics
    ///
    /// Panics if the derived radius exceeds
    /// [`MAX_RADIUS`](crate::image::MAX_RADIUS), or if `output` is too small
    /// for the region produced by the border policy.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::border::Clamp;
    /// use fovea::image::{Image, ImageView};
    /// use fovea::pixel::MonoF32;
    /// use fovea::sigma;
    /// use fovea::transform::SeparableScratch;
    ///
    /// let mut scratch = SeparableScratch::new();
    /// let mut out = Image::<MonoF32>::zero(32, 32);
    ///
    /// for frame in 0..3 {
    ///     let src = Image::fill(32, 32, MonoF32::new(0.25 * frame as f32));
    ///     scratch.gaussian_blur_into(&src, sigma!(1.5), &Clamp, &mut out);
    ///     assert!((out.pixel_at(16, 16).0 - 0.25 * frame as f32).abs() < 1e-4);
    /// }
    /// ```
    pub fn gaussian_blur_into<I, B, O, P, Out>(
        &mut self,
        image: &I,
        sigma: Sigma,
        border: &B,
        output: &mut O,
    ) where
        I: RasterImage<Pixel = P>,
        P: Copy + LinearPixel<f32, Accumulator = Acc>,
        B: BorderPolicy<I> + for<'r> BorderPolicy<ImageRef<'r, Acc>>,
        O: RasterImageMut<Pixel = Out>,
        Out: FromLinear<Acc>,
    {
        // A σ blur is a separable convolution with the σ-derived kernel —
        // the same path an explicit `truncate` takes, and the same path the
        // free `gaussian_blur_into` takes without a scratch.
        self.convolve_separable_into(
            image,
            &gaussian_kernel_1d(sigma, DEFAULT_TRUNCATE),
            border,
            output,
        );
    }
}

// ─── Sobel ───────────────────────────────────────────────────────────────────

/// Sobel edge detector — horizontal gradient (dI/dx).
///
/// Uses [`Neighborhood::sobel_y`] (the `[-1 0 1; -2 0 2; -1 0 1]`
/// kernel). The output is `P::Accumulator` (e.g. `MonoF32` for
/// `Mono8`) to preserve negative gradients.
///
/// The "x" in the function name refers to the **gradient direction**
/// (horizontal change), not the kernel orientation.
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView};
/// use fovea::border::Clamp;
/// use fovea::pixel::{Mono8, MonoF32};
/// use fovea::transform::sobel_x;
///
/// let src = Image::fill(6, 6, Mono8::new(50));
/// let result: Image<MonoF32> = sobel_x(&src, &Clamp);
///
/// // Uniform image: gradient is zero
/// for y in 0..result.height() {
///     for x in 0..result.width() {
///         assert!(result.pixel_at(x, y).abs().0 < 1e-4);
///     }
/// }
/// ```
#[must_use]
pub fn sobel_x<I, B, P>(image: &I, border: &B) -> Image<<P as LinearPixel<f32>>::Accumulator>
where
    I: RasterImage<Pixel = P>,
    P: Copy + LinearPixel<f32>,
    <P as LinearPixel<f32>>::Accumulator: Default + ZeroablePixel,
    B: BorderPolicy<I>,
{
    convolve(image, &Neighborhood::<f32, 3, 3>::sobel_y(), border)
}

/// Sobel edge detector — vertical gradient (dI/dy).
///
/// Uses [`Neighborhood::sobel_x`] (the `[-1 -2 -1; 0 0 0; 1 2 1]`
/// kernel). The output is `P::Accumulator` (e.g. `MonoF32` for
/// `Mono8`) to preserve negative gradients.
///
/// The "y" in the function name refers to the **gradient direction**
/// (vertical change), not the kernel orientation.
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView};
/// use fovea::border::Clamp;
/// use fovea::pixel::{Mono8, MonoF32};
/// use fovea::transform::sobel_y;
///
/// let src = Image::fill(6, 6, Mono8::new(50));
/// let result: Image<MonoF32> = sobel_y(&src, &Clamp);
///
/// // Uniform image: gradient is zero
/// for y in 0..result.height() {
///     for x in 0..result.width() {
///         assert!(result.pixel_at(x, y).abs().0 < 1e-4);
///     }
/// }
/// ```
#[must_use]
pub fn sobel_y<I, B, P>(image: &I, border: &B) -> Image<<P as LinearPixel<f32>>::Accumulator>
where
    I: RasterImage<Pixel = P>,
    P: Copy + LinearPixel<f32>,
    <P as LinearPixel<f32>>::Accumulator: Default + ZeroablePixel,
    B: BorderPolicy<I>,
{
    convolve(image, &Neighborhood::<f32, 3, 3>::sobel_x(), border)
}

// ─── Scharr ──────────────────────────────────────────────────────────────────

/// Scharr edge detector — horizontal gradient (dI/dx).
///
/// Uses [`Neighborhood::scharr_y`]. More rotation-invariant than Sobel.
/// Output is `P::Accumulator` (e.g. `MonoF32` for `Mono8`).
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView};
/// use fovea::border::Clamp;
/// use fovea::pixel::{Mono8, MonoF32};
/// use fovea::transform::scharr_x;
///
/// let src = Image::fill(6, 6, Mono8::new(50));
/// let result: Image<MonoF32> = scharr_x(&src, &Clamp);
///
/// for y in 0..result.height() {
///     for x in 0..result.width() {
///         assert!(result.pixel_at(x, y).abs().0 < 1e-4);
///     }
/// }
/// ```
#[must_use]
pub fn scharr_x<I, B, P>(image: &I, border: &B) -> Image<<P as LinearPixel<f32>>::Accumulator>
where
    I: RasterImage<Pixel = P>,
    P: Copy + LinearPixel<f32>,
    <P as LinearPixel<f32>>::Accumulator: Default + ZeroablePixel,
    B: BorderPolicy<I>,
{
    convolve(image, &Neighborhood::<f32, 3, 3>::scharr_y(), border)
}

/// Scharr edge detector — vertical gradient (dI/dy).
///
/// Uses [`Neighborhood::scharr_x`]. Output is `P::Accumulator` (e.g.
/// `MonoF32` for `Mono8`).
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView};
/// use fovea::border::Clamp;
/// use fovea::pixel::{Mono8, MonoF32};
/// use fovea::transform::scharr_y;
///
/// let src = Image::fill(6, 6, Mono8::new(50));
/// let result: Image<MonoF32> = scharr_y(&src, &Clamp);
///
/// for y in 0..result.height() {
///     for x in 0..result.width() {
///         assert!(result.pixel_at(x, y).abs().0 < 1e-4);
///     }
/// }
/// ```
#[must_use]
pub fn scharr_y<I, B, P>(image: &I, border: &B) -> Image<<P as LinearPixel<f32>>::Accumulator>
where
    I: RasterImage<Pixel = P>,
    P: Copy + LinearPixel<f32>,
    <P as LinearPixel<f32>>::Accumulator: Default + ZeroablePixel,
    B: BorderPolicy<I>,
{
    convolve(image, &Neighborhood::<f32, 3, 3>::scharr_x(), border)
}

// ─── Prewitt ─────────────────────────────────────────────────────────────────

/// Prewitt edge detector — horizontal gradient (dI/dx).
///
/// Uses [`Neighborhood::prewitt_y`]. Output is `P::Accumulator` (e.g.
/// `MonoF32` for `Mono8`).
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView};
/// use fovea::border::Clamp;
/// use fovea::pixel::{Mono8, MonoF32};
/// use fovea::transform::prewitt_x;
///
/// let src = Image::fill(6, 6, Mono8::new(50));
/// let result: Image<MonoF32> = prewitt_x(&src, &Clamp);
///
/// for y in 0..result.height() {
///     for x in 0..result.width() {
///         assert!(result.pixel_at(x, y).abs().0 < 1e-4);
///     }
/// }
/// ```
#[must_use]
pub fn prewitt_x<I, B, P>(image: &I, border: &B) -> Image<<P as LinearPixel<f32>>::Accumulator>
where
    I: RasterImage<Pixel = P>,
    P: Copy + LinearPixel<f32>,
    <P as LinearPixel<f32>>::Accumulator: Default + ZeroablePixel,
    B: BorderPolicy<I>,
{
    convolve(image, &Neighborhood::<f32, 3, 3>::prewitt_y(), border)
}

/// Prewitt edge detector — vertical gradient (dI/dy).
///
/// Uses [`Neighborhood::prewitt_x`]. Output is `P::Accumulator` (e.g.
/// `MonoF32` for `Mono8`).
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView};
/// use fovea::border::Clamp;
/// use fovea::pixel::{Mono8, MonoF32};
/// use fovea::transform::prewitt_y;
///
/// let src = Image::fill(6, 6, Mono8::new(50));
/// let result: Image<MonoF32> = prewitt_y(&src, &Clamp);
///
/// for y in 0..result.height() {
///     for x in 0..result.width() {
///         assert!(result.pixel_at(x, y).abs().0 < 1e-4);
///     }
/// }
/// ```
#[must_use]
pub fn prewitt_y<I, B, P>(image: &I, border: &B) -> Image<<P as LinearPixel<f32>>::Accumulator>
where
    I: RasterImage<Pixel = P>,
    P: Copy + LinearPixel<f32>,
    <P as LinearPixel<f32>>::Accumulator: Default + ZeroablePixel,
    B: BorderPolicy<I>,
{
    convolve(image, &Neighborhood::<f32, 3, 3>::prewitt_x(), border)
}

// ─── Laplacian ───────────────────────────────────────────────────────────────

/// 3×3 Laplacian (4-connected).
///
/// Uses [`Neighborhood::laplacian`]:
///
/// ```text
///  0 -1  0
/// -1  4 -1
///  0 -1  0
/// ```
///
/// Output is `P::Accumulator` (e.g. `MonoF32` for `Mono8`).
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView};
/// use fovea::border::Clamp;
/// use fovea::pixel::{Mono8, MonoF32};
/// use fovea::transform::laplacian;
///
/// let src = Image::fill(6, 6, Mono8::new(10));
/// let result: Image<MonoF32> = laplacian(&src, &Clamp);
///
/// // Uniform image: Laplacian is zero
/// for y in 0..result.height() {
///     for x in 0..result.width() {
///         assert!(result.pixel_at(x, y).abs().0 < 1e-4);
///     }
/// }
/// ```
#[must_use]
pub fn laplacian<I, B, P>(image: &I, border: &B) -> Image<<P as LinearPixel<f32>>::Accumulator>
where
    I: RasterImage<Pixel = P>,
    P: Copy + LinearPixel<f32>,
    <P as LinearPixel<f32>>::Accumulator: Default + ZeroablePixel,
    B: BorderPolicy<I>,
{
    convolve(image, &Neighborhood::<f32, 3, 3>::laplacian(), border)
}

/// 3×3 Laplacian (8-connected / diagonal-inclusive).
///
/// Uses [`Neighborhood::laplacian_8`]:
///
/// ```text
/// -1 -1 -1
/// -1  8 -1
/// -1 -1 -1
/// ```
///
/// Output is `P::Accumulator` (e.g. `MonoF32` for `Mono8`).
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView};
/// use fovea::border::Clamp;
/// use fovea::pixel::{Mono8, MonoF32};
/// use fovea::transform::laplacian_8;
///
/// let src = Image::fill(6, 6, Mono8::new(10));
/// let result: Image<MonoF32> = laplacian_8(&src, &Clamp);
///
/// for y in 0..result.height() {
///     for x in 0..result.width() {
///         assert!(result.pixel_at(x, y).abs().0 < 1e-4);
///     }
/// }
/// ```
#[must_use]
pub fn laplacian_8<I, B, P>(image: &I, border: &B) -> Image<<P as LinearPixel<f32>>::Accumulator>
where
    I: RasterImage<Pixel = P>,
    P: Copy + LinearPixel<f32>,
    <P as LinearPixel<f32>>::Accumulator: Default + ZeroablePixel,
    B: BorderPolicy<I>,
{
    convolve(image, &Neighborhood::<f32, 3, 3>::laplacian_8(), border)
}

// ─── Sharpen ─────────────────────────────────────────────────────────────────

/// 3×3 sharpening filter.
///
/// Uses [`Neighborhood::sharpen`] (identity + scaled Laplacian):
///
/// ```text
///  0 -1  0
/// -1  5 -1
///  0 -1  0
/// ```
///
/// The output pixel type matches the input via [`FromLinear`].
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView};
/// use fovea::border::Clamp;
/// use fovea::pixel::Mono8;
/// use fovea::transform::sharpen;
///
/// let src = Image::fill(6, 6, Mono8::new(100));
/// let result: Image<Mono8> = sharpen(&src, &Clamp);
///
/// // Uniform image: sharpening has no effect (Laplacian component = 0)
/// for y in 0..result.height() {
///     for x in 0..result.width() {
///         assert_eq!(result.pixel_at(x, y), Mono8::new(100));
///     }
/// }
/// ```
#[must_use]
pub fn sharpen<I, B, P, Out>(image: &I, border: &B) -> Image<Out>
where
    I: RasterImage<Pixel = P>,
    P: Copy + LinearPixel<f32>,
    <P as LinearPixel<f32>>::Accumulator: Default,
    B: BorderPolicy<I>,
    Out: ZeroablePixel + FromLinear<<P as LinearPixel<f32>>::Accumulator>,
{
    convolve(image, &Neighborhood::<f32, 3, 3>::sharpen(), border)
}

// ─── Emboss ──────────────────────────────────────────────────────────────────

/// 3×3 emboss filter.
///
/// Uses [`Neighborhood::emboss`]:
///
/// ```text
/// -2 -1  0
/// -1  1  1
///  0  1  2
/// ```
///
/// Output is `P::Accumulator` (e.g. `MonoF32` for `Mono8`) — emboss
/// can produce negative values.
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView};
/// use fovea::border::Clamp;
/// use fovea::pixel::{Mono8, MonoF32};
/// use fovea::transform::emboss;
///
/// let src = Image::fill(6, 6, Mono8::new(50));
/// let result: Image<MonoF32> = emboss(&src, &Clamp);
///
/// // Uniform image: emboss returns the original intensity
/// // (kernel sums to 1, so uniform × 1 = uniform)
/// for y in 0..result.height() {
///     for x in 0..result.width() {
///         assert!((result.pixel_at(x, y).0 - 50.0).abs() < 1e-4);
///     }
/// }
/// ```
#[must_use]
pub fn emboss<I, B, P>(image: &I, border: &B) -> Image<<P as LinearPixel<f32>>::Accumulator>
where
    I: RasterImage<Pixel = P>,
    P: Copy + LinearPixel<f32>,
    <P as LinearPixel<f32>>::Accumulator: Default + ZeroablePixel,
    B: BorderPolicy<I>,
{
    convolve(image, &Neighborhood::<f32, 3, 3>::emboss(), border)
}

// ─── Gradient magnitude / direction ───────────────────────────────────────────

/// Pixel-wise L2 gradient magnitude `sqrt(gx² + gy²)`.
///
/// Fuses a pair of gradient images — typically [`scharr_x`] / [`scharr_y`]
/// (or [`sobel_x`] / [`sobel_y`]) — into a single unsigned edge-strength map.
/// This is the magnitude pre-stage of a Canny pipeline and a thin, named
/// wrapper over [`combine_images`] with the [`Magnitude`] strategy.
///
/// Generic over the same float-channel pixel types as [`Magnitude`] (e.g.
/// `MonoF32`, `MonoF64`, `RgbF32`), so it follows whichever accumulator the
/// upstream gradient operator produced. The output is a raw float container:
/// values are `>= 0` but otherwise unbounded — thresholding is the caller's
/// responsibility.
///
/// [`Magnitude`] computes `sqrt(gx² + gy²)` directly, which is exact for
/// gradients of ordinary image data and vectorizes. For inputs large enough
/// that `gx²` leaves the float range, combine with
/// [`MagnitudeHypot`](crate::transform::MagnitudeHypot) instead.
///
/// # Errors
///
/// Returns [`Error::SizeMismatch`] if `gx` and `gy` differ in dimensions.
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::MonoF32;
/// use fovea::transform::gradient_magnitude;
///
/// // Classic 3-4-5 triple, pixel-wise.
/// let gx = Image::fill(2, 2, MonoF32::new(3.0));
/// let gy = Image::fill(2, 2, MonoF32::new(4.0));
/// let mag = gradient_magnitude(&gx, &gy).unwrap();
/// assert!((mag.pixel_at(0, 0).value() - 5.0).abs() < 1e-5);
/// ```
pub fn gradient_magnitude<IA, IB, P>(gx: &IA, gy: &IB) -> Result<Image<P>, Error>
where
    IA: RasterImage<Pixel = P>,
    IB: RasterImage<Pixel = P>,
    P: HomogeneousPixel + ZeroablePixel,
    P::Channel: MagnitudeChannel,
{
    combine_images(gx, gy, Magnitude)
}

/// Pixel-wise gradient direction `atan2(gy, gx)`, in radians on `(-π, π]`.
///
/// The companion to [`gradient_magnitude`]: the angle of the gradient vector
/// `(gx, gy)` at each pixel, as produced by [`scharr_x`] / [`scharr_y`]. Feed
/// both into [`non_maximum_suppression`] to thin the edge ridge. A thin,
/// named wrapper over [`combine_images`] with the [`Direction`] strategy.
///
/// The angle follows image coordinates (`y` increases downward), so a pure
/// `+x` gradient is `0`, a pure `+y` gradient is `π/2`. Generic over the same
/// float-channel pixel types as [`Direction`] (`MonoF32`, `MonoF64`, …).
///
/// # Errors
///
/// Returns [`Error::SizeMismatch`] if `gx` and `gy` differ in dimensions.
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::MonoF32;
/// use fovea::transform::gradient_direction;
/// use std::f32::consts::FRAC_PI_2;
///
/// // Pure +y gradient → π/2.
/// let gx = Image::fill(1, 1, MonoF32::new(0.0));
/// let gy = Image::fill(1, 1, MonoF32::new(1.0));
/// let dir = gradient_direction(&gx, &gy).unwrap();
/// assert!((dir.pixel_at(0, 0).value() - FRAC_PI_2).abs() < 1e-6);
/// ```
pub fn gradient_direction<IA, IB, P>(gx: &IA, gy: &IB) -> Result<Image<P>, Error>
where
    IA: RasterImage<Pixel = P>,
    IB: RasterImage<Pixel = P>,
    P: HomogeneousPixel + ZeroablePixel,
    P::Channel: DirectionChannel,
{
    combine_images(gx, gy, Direction)
}

// ─── Non-maximum suppression ──────────────────────────────────────────────────

/// Quantise a gradient direction (radians) to one of four edge sectors and
/// return the `(dx, dy)` step toward the neighbour along the gradient.
///
/// The gradient and its opposite describe the same edge, so the angle is
/// folded into `[0, π)` before quantising. The four returned steps are:
/// `(1, 0)` horizontal, `(1, 1)` main diagonal, `(0, 1)` vertical, and
/// `(-1, 1)` anti-diagonal. Every step has `dy >= 0`, which
/// [`nms_survives`] relies on when picking the two neighbour rows.
#[inline]
pub(crate) fn nms_sector(theta: f64) -> Offset {
    use core::f64::consts::PI;
    // Fold (-π, π] onto [0, π): opposite gradients share an edge orientation.
    let mut a = theta;
    if a < 0.0 {
        a += PI;
    }
    const SEG: f64 = PI / 8.0; // 22.5°
    if a < SEG {
        Offset::new(1, 0) // gradient ≈ horizontal → compare left / right
    } else if a < 3.0 * SEG {
        Offset::new(1, 1) // gradient ≈ +45° → compare the main diagonal
    } else if a < 5.0 * SEG {
        Offset::new(0, 1) // gradient ≈ vertical → compare up / down
    } else if a < 7.0 * SEG {
        Offset::new(-1, 1) // gradient ≈ −45° → compare the anti-diagonal
    } else {
        Offset::new(1, 0) // [157.5°, 180°] wraps back to horizontal
    }
}

/// The same quantisation as [`nms_sector`], computed straight from the
/// gradient components instead of from `atan2(gy, gx)`.
///
/// Only the sector survives the quantisation, so the angle itself is
/// wasted work: this compares `|gy|` against `|gx|·tan(22.5°)` and
/// `|gx|·tan(67.5°)`, with the sign of `gx·gy` choosing between the two
/// diagonals. That reproduces `nms_sector(atan2(gy, gx))` exactly (the
/// half-plane fold becomes "take absolute values"), minus one
/// transcendental call per pixel.
///
/// The one divergence is the zero gradient `(0, 0)`, reported here as
/// vertical and by the angle path as horizontal. Its magnitude is `0`, so
/// the suppressed output is `0` under either sector.
#[inline]
pub(crate) fn nms_sector_from_gradient(gx: f64, gy: f64) -> Offset {
    /// `tan(22.5°)`
    const T22: f64 = 0.414_213_562_373_095_05;
    /// `tan(67.5°)`
    const T67: f64 = 2.414_213_562_373_095;

    let ax = gx.abs();
    let ay = gy.abs();
    if gx * gy >= 0.0 {
        // Folded angle in [0, π/2]: rising diagonal.
        if ay < ax * T22 {
            Offset::new(1, 0)
        } else if ay < ax * T67 {
            Offset::new(1, 1)
        } else {
            Offset::new(0, 1)
        }
    } else {
        // Folded angle in (π/2, π): falling diagonal. The comparisons
        // mirror the branch above, hence the flipped order and strictness.
        if ay > ax * T67 {
            Offset::new(0, 1)
        } else if ay > ax * T22 {
            Offset::new(-1, 1)
        } else {
            Offset::new(1, 0)
        }
    }
}

/// Channel of `row[x + dx]`, or `None` if the row is absent (off the top or
/// bottom of the image) or the column falls outside `0..w`.
#[inline]
fn nms_at<P>(row: Option<&[P]>, x: usize, dx: isize, w: usize) -> Option<P::Channel>
where
    P: HomogeneousPixel,
{
    let row = row?;
    let nx = x.checked_add_signed(dx)?;
    if nx >= w {
        return None;
    }
    Some(row[nx].channel(0))
}

/// Whether `cur[x]` is a local maximum along `step`.
///
/// `prev` / `next` are the rows above and below `cur`, or `None` at the
/// image border. Because every [`nms_sector`] step has `dy >= 0`, the two
/// neighbours are always `next`/`prev` (for `dy == 1`) or `cur` twice (for
/// `dy == 0`) — no row indexing arithmetic is needed. A neighbour outside
/// the image suppresses the pixel: local maximality cannot be established.
#[inline]
fn nms_survives<P>(
    cur: &[P],
    prev: Option<&[P]>,
    next: Option<&[P]>,
    x: usize,
    w: usize,
    step: Offset,
) -> bool
where
    P: HomogeneousPixel,
    P::Channel: PartialOrd,
{
    let dx = step.dx as isize;
    let (forward, backward) = if step.dy == 0 {
        (Some(cur), Some(cur))
    } else {
        (next, prev)
    };
    match (nms_at(forward, x, dx, w), nms_at(backward, x, -dx, w)) {
        (Some(a), Some(b)) => {
            let m = cur[x].channel(0);
            m >= a && m >= b
        }
        _ => false,
    }
}

/// Non-maximum suppression: thin a gradient-magnitude ridge to single-pixel
/// width along the (quantised) gradient direction.
///
/// For each pixel, the gradient direction is quantised to one of four sectors
/// (0°, 45°, 90°, 135°); the pixel is kept only when its magnitude is `>=`
/// **both** neighbours along that direction, otherwise it is set to zero. The
/// inclusive `>=` (ties kept) matches the common OpenCV convention and avoids
/// erasing genuine flat-topped ridges.
///
/// Border pixels whose along-gradient neighbour lies outside the image are
/// suppressed to zero (their local maximality cannot be established).
///
/// `magnitude` and `direction` are the outputs of [`gradient_magnitude`] and
/// [`gradient_direction`] over the same gradient pair. Operates on the first
/// channel (intended for single-channel gradient images such as `MonoF32` /
/// `MonoF64`); the result is a raw float container suitable as input to a
/// hysteresis threshold.
///
/// # Errors
///
/// Returns [`Error::SizeMismatch`] if `magnitude` and `direction` differ
/// in dimensions — two separately produced input images, the same
/// input-vs-input relation as [`combine_images`](crate::transform::combine_images).
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::MonoF32;
/// use fovea::transform::non_maximum_suppression;
///
/// // Horizontal gradient (θ = 0): a [1, 2, 1] ridge thins to [0, 2, 0].
/// let mag = Image::from_vec(
///     3,
///     1,
///     vec![MonoF32::new(1.0), MonoF32::new(2.0), MonoF32::new(1.0)],
/// )
/// .unwrap();
/// let dir = Image::fill(3, 1, MonoF32::new(0.0));
/// let thin = non_maximum_suppression(&mag, &dir)?;
/// assert_eq!(thin.pixel_at(0, 0).value(), 0.0); // left border → suppressed
/// assert_eq!(thin.pixel_at(1, 0).value(), 2.0); // local maximum kept
/// assert_eq!(thin.pixel_at(2, 0).value(), 0.0); // right border → suppressed
/// # Ok::<(), fovea::Error>(())
/// ```
pub fn non_maximum_suppression<IM, IA, P>(magnitude: &IM, direction: &IA) -> Result<Image<P>, Error>
where
    IM: RasterImage<Pixel = P>,
    IA: RasterImage<Pixel = P>,
    P: HomogeneousPixel + ZeroablePixel,
    P::Channel: PartialOrd,
    f64: From<P::Channel>,
{
    if magnitude.size() != direction.size() {
        return Err(Error::SizeMismatch {
            expected: magnitude.size(),
            actual: direction.size(),
        });
    }

    let (w, h) = (magnitude.width(), magnitude.height());
    let mut out = Image::fill(w, h, P::zero());
    for y in 0..h {
        let cur = magnitude.row(y);
        let prev = (y > 0).then(|| magnitude.row(y - 1));
        let next = (y + 1 < h).then(|| magnitude.row(y + 1));
        let dir = direction.row(y);
        let dst = out.row_mut(y);
        for x in 0..w {
            let step = nms_sector(f64::from(dir[x].channel(0)));
            if nms_survives(cur, prev, next, x, w, step) {
                dst[x] = cur[x];
            }
        }
    }
    Ok(out)
}

/// [`non_maximum_suppression`] driven by the raw gradient pair instead of a
/// pre-computed direction map.
///
/// Behaviourally equivalent to
/// `non_maximum_suppression(magnitude, &gradient_direction(gx, gy)?)`, but
/// it skips building the direction image and the `atan2` per pixel that
/// fills it — see [`nms_sector_from_gradient`]. This is the path
/// [`canny`](crate::analyze::edge::canny) takes; the staged form stays
/// public so a hand-composed pipeline can still inspect the angle map.
///
/// # Panics
///
/// Panics if `magnitude`, `gx`, and `gy` do not all share a size.
#[must_use]
pub(crate) fn non_maximum_suppression_from_gradients<IM, IX, IY, P>(
    magnitude: &IM,
    gx: &IX,
    gy: &IY,
) -> Image<P>
where
    IM: RasterImage<Pixel = P>,
    IX: RasterImage<Pixel = P>,
    IY: RasterImage<Pixel = P>,
    P: HomogeneousPixel + ZeroablePixel,
    P::Channel: PartialOrd,
    f64: From<P::Channel>,
{
    assert!(
        magnitude.size() == gx.size() && gx.size() == gy.size(),
        "non_maximum_suppression: magnitude, gx and gy must have the same size",
    );

    let (w, h) = (magnitude.width(), magnitude.height());
    let mut out = Image::fill(w, h, P::zero());
    for y in 0..h {
        let cur = magnitude.row(y);
        let prev = (y > 0).then(|| magnitude.row(y - 1));
        let next = (y + 1 < h).then(|| magnitude.row(y + 1));
        let gx_row = gx.row(y);
        let gy_row = gy.row(y);
        let dst = out.row_mut(y);
        for x in 0..w {
            let step = nms_sector_from_gradient(
                f64::from(gx_row[x].channel(0)),
                f64::from(gy_row[x].channel(0)),
            );
            if nms_survives(cur, prev, next, x, w, step) {
                dst[x] = cur[x];
            }
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Size;
    use crate::border::{Clamp, Constant, Skip};
    use crate::image::{ImageView, ImageViewMut, gaussian_kernel_1d};
    use crate::pixel::{Mono8, MonoF32};
    use crate::sigma;
    use crate::transform::convolve;

    // ── helpers ──────────────────────────────────────────────────────────

    fn make_gradient_8x8() -> Image<MonoF32> {
        Image::generate(8, 8, |x, y| MonoF32::new((x + y * 8) as f32))
    }

    // ── box blur ────────────────────────────────────────────────────────

    #[test]
    fn box_blur_3x3_uniform_f32() {
        let src = Image::fill(8, 8, MonoF32::new(7.0));
        let result: Image<MonoF32> = box_blur_3x3(&src, &Clamp);

        assert_eq!(result.width(), 8);
        assert_eq!(result.height(), 8);
        for y in 0..result.height() {
            for x in 0..result.width() {
                assert!(
                    (result.pixel_at(x, y).0 - 7.0).abs() < 1e-4,
                    "at ({x}, {y}): {}",
                    result.pixel_at(x, y).0,
                );
            }
        }
    }

    #[test]
    fn box_blur_3x3_uniform_u8() {
        let src = Image::fill(8, 8, Mono8::new(100));
        let result: Image<Mono8> = box_blur_3x3(&src, &Clamp);

        for y in 0..result.height() {
            for x in 0..result.width() {
                assert_eq!(result.pixel_at(x, y), Mono8::new(100));
            }
        }
    }

    #[test]
    fn box_blur_5x5_uniform_f32() {
        let src = Image::fill(10, 10, MonoF32::new(3.0));
        let result: Image<MonoF32> = box_blur_5x5(&src, &Clamp);

        for y in 0..result.height() {
            for x in 0..result.width() {
                assert!((result.pixel_at(x, y).0 - 3.0).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn box_blur_5x5_uniform_u8() {
        let src = Image::fill(10, 10, Mono8::new(200));
        let result: Image<Mono8> = box_blur_5x5(&src, &Clamp);

        for y in 0..result.height() {
            for x in 0..result.width() {
                assert_eq!(result.pixel_at(x, y), Mono8::new(200));
            }
        }
    }

    // ── box blur matches full convolution ───────────────────────────────

    #[test]
    fn box_blur_3x3_matches_full_convolution() {
        let src = make_gradient_8x8();
        let full_kernel = Neighborhood::<f32, 3, 3>::box_blur_3x3();
        let full: Image<MonoF32> = convolve(&src, &full_kernel, &Clamp);
        let sep: Image<MonoF32> = box_blur_3x3(&src, &Clamp);

        assert_eq!(full.width(), sep.width());
        assert_eq!(full.height(), sep.height());
        for y in 0..full.height() {
            for x in 0..full.width() {
                assert!(
                    (full.pixel_at(x, y).0 - sep.pixel_at(x, y).0).abs() < 1e-3,
                    "mismatch at ({x}, {y}): full={}, sep={}",
                    full.pixel_at(x, y).0,
                    sep.pixel_at(x, y).0,
                );
            }
        }
    }

    // ── gaussian blur ───────────────────────────────────────────────────

    #[test]
    fn gaussian_blur_3x3_uniform_f32() {
        // gaussian_blur_3x3 is now normalized (sum=1), so a flat input is
        // returned unchanged (brightness preserved).
        let src = Image::fill(8, 8, MonoF32::new(1.0));
        let result: Image<MonoF32> = gaussian_blur_3x3(&src, &Clamp);

        for y in 0..result.height() {
            for x in 0..result.width() {
                assert!(
                    (result.pixel_at(x, y).0 - 1.0).abs() < 1e-4,
                    "at ({x}, {y}): {}",
                    result.pixel_at(x, y).0,
                );
            }
        }
    }

    #[test]
    fn gaussian_blur_3x3_matches_full_convolution() {
        // The normalized blur equals the raw 2D `Neighborhood` convolution
        // (sum 16) divided by 16.
        let src = make_gradient_8x8();
        let full_kernel = Neighborhood::<f32, 3, 3>::gaussian_3x3();
        let full: Image<MonoF32> = convolve(&src, &full_kernel, &Clamp);
        let sep: Image<MonoF32> = gaussian_blur_3x3(&src, &Clamp);

        for y in 0..full.height() {
            for x in 0..full.width() {
                assert!(
                    (full.pixel_at(x, y).0 / 16.0 - sep.pixel_at(x, y).0).abs() < 1e-2,
                    "mismatch at ({x}, {y})",
                );
            }
        }
    }

    #[test]
    fn gaussian_blur_5x5_uniform_f32() {
        // Normalized (sum=1): a flat input is returned unchanged.
        let src = Image::fill(10, 10, MonoF32::new(1.0));
        let result: Image<MonoF32> = gaussian_blur_5x5(&src, &Clamp);

        for y in 0..result.height() {
            for x in 0..result.width() {
                assert!((result.pixel_at(x, y).0 - 1.0).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn gaussian_blur_5x5_matches_full_convolution() {
        // The normalized blur equals the raw 2D `Neighborhood` convolution
        // (sum 256) divided by 256.
        let src = make_gradient_8x8();
        let full_kernel = Neighborhood::<f32, 5, 5>::gaussian_5x5();
        let full: Image<MonoF32> = convolve(&src, &full_kernel, &Clamp);
        let sep: Image<MonoF32> = gaussian_blur_5x5(&src, &Clamp);

        for y in 0..full.height() {
            for x in 0..full.width() {
                assert!(
                    (full.pixel_at(x, y).0 / 256.0 - sep.pixel_at(x, y).0).abs() < 1e-3,
                    "mismatch at ({x}, {y}): full/256={}, sep={}",
                    full.pixel_at(x, y).0 / 256.0,
                    sep.pixel_at(x, y).0,
                );
            }
        }
    }

    // ── parameterized gaussian blur ─────────────────────────────────────

    #[test]
    fn gaussian_blur_uniform_image_preserved_f32() {
        let src = Image::fill(16, 16, MonoF32::new(0.7));
        let result: Image<MonoF32> = gaussian_blur(&src, sigma!(2.0), &Clamp);
        assert_eq!(result.size(), src.size());
        for y in 0..result.height() {
            for x in 0..result.width() {
                assert!(
                    (result.pixel_at(x, y).0 - 0.7).abs() < 1e-4,
                    "at ({x}, {y}): {}",
                    result.pixel_at(x, y).0,
                );
            }
        }
    }

    #[test]
    fn gaussian_blur_uniform_image_preserved_u8() {
        let src = Image::fill(16, 16, Mono8::new(120));
        let result: Image<Mono8> = gaussian_blur(&src, sigma!(1.5), &Clamp);
        for y in 0..result.height() {
            for x in 0..result.width() {
                assert_eq!(result.pixel_at(x, y), Mono8::new(120));
            }
        }
    }

    #[test]
    fn gaussian_blur_impulse_response_is_the_kernel() {
        // A single bright pixel on a black field, blurred, should reproduce
        // the 2-D Gaussian (outer product of the 1-D kernel) — interior
        // only, with a zero border so nothing bleeds in.
        let sigma = sigma!(1.0);
        let truncate = 2.0; // radius 2, 5 taps
        let kernel = gaussian_kernel_1d(sigma, truncate);
        let w = kernel.weights();
        let r = kernel.radius();

        let c = 5usize;
        let mut src = Image::fill(11, 11, MonoF32::new(0.0));
        *src.pixel_at_mut(c, c) = MonoF32::new(1.0);

        let out: Image<MonoF32> = convolve_separable(&src, &kernel, &Constant(MonoF32(0.0)));

        for dy in -(r as isize)..=(r as isize) {
            for dx in -(r as isize)..=(r as isize) {
                let expected = w[(r as isize + dx) as usize] * w[(r as isize + dy) as usize];
                let got = out
                    .pixel_at((c as isize + dx) as usize, (c as isize + dy) as usize)
                    .0;
                assert!(
                    (got - expected).abs() < 1e-6,
                    "impulse response at ({dx}, {dy}): got {got}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn gaussian_blur_matches_full_2d_convolution() {
        // Separability: the two-pass blur must equal a non-separable 2-D
        // convolution with the outer-product kernel. Checked on interior
        // pixels (where the border policy has no effect).
        let sigma = sigma!(1.0);
        let truncate = 2.0; // radius 2
        let kernel = gaussian_kernel_1d(sigma, truncate);
        let w = kernel.weights();
        let r = kernel.radius();

        let src = Image::generate(9, 9, |x, y| MonoF32::new((x * 3 + y * 5) as f32));
        let out: Image<MonoF32> = convolve_separable(&src, &kernel, &Clamp);

        for y in r..(9 - r) {
            for x in r..(9 - r) {
                let mut reference = 0.0f32;
                for (j, &wj) in w.iter().enumerate() {
                    for (i, &wi) in w.iter().enumerate() {
                        let sx = x + i - r;
                        let sy = y + j - r;
                        reference += src.pixel_at(sx, sy).0 * wi * wj;
                    }
                }
                let got = out.pixel_at(x, y).0;
                assert!(
                    (got - reference).abs() < 1e-3,
                    "at ({x}, {y}): separable={got}, full2d={reference}"
                );
            }
        }
    }

    #[test]
    fn gaussian_blur_larger_sigma_smooths_more() {
        // A step edge blurred more (larger sigma) has a gentler maximum
        // slope. Measure the steepest adjacent horizontal difference in the
        // interior; it must shrink as sigma grows.
        let src = Image::generate(41, 5, |x, _y| {
            MonoF32::new(if x < 20 { 0.0 } else { 100.0 })
        });

        let max_slope = |sigma: f32| -> f32 {
            let blurred: Image<MonoF32> = gaussian_blur(&src, Sigma::new(sigma).unwrap(), &Clamp);
            let mut m = 0.0f32;
            for y in 0..blurred.height() {
                for x in 1..blurred.width() {
                    let d = (blurred.pixel_at(x, y).0 - blurred.pixel_at(x - 1, y).0).abs();
                    if d > m {
                        m = d;
                    }
                }
            }
            m
        };

        let slope_small = max_slope(1.0);
        let slope_large = max_slope(3.0);
        assert!(
            slope_large < slope_small,
            "larger sigma should reduce the max slope: sigma=1 → {slope_small}, sigma=3 → {slope_large}"
        );
    }

    #[test]
    fn gaussian_blur_into_matches_owned() {
        let src = Image::generate(12, 12, |x, y| MonoF32::new((x + y) as f32));

        let owned: Image<MonoF32> = gaussian_blur(&src, sigma!(1.5), &Clamp);

        let mut into = Image::<MonoF32>::zero(owned.width(), owned.height());
        gaussian_blur_into(&src, sigma!(1.5), &Clamp, &mut into);

        for y in 0..owned.height() {
            for x in 0..owned.width() {
                assert!(
                    (owned.pixel_at(x, y).0 - into.pixel_at(x, y).0).abs() < 1e-6,
                    "mismatch at ({x}, {y})",
                );
            }
        }
    }

    #[test]
    fn scratch_gaussian_blur_into_matches_owned_across_frames() {
        // The scratch form must equal the allocating form on every call —
        // the first (which sizes the buffers) and every later one (which
        // reuses them).
        let sigma = sigma!(1.5);
        let mut scratch = SeparableScratch::new();
        let mut reused = Image::<MonoF32>::zero(12, 12);

        for frame in 0..3 {
            let src = Image::generate(12, 12, |x, y| MonoF32::new((x + y + frame) as f32));
            let owned: Image<MonoF32> = gaussian_blur(&src, sigma, &Clamp);

            scratch.gaussian_blur_into(&src, sigma, &Clamp, &mut reused);

            for y in 0..owned.height() {
                for x in 0..owned.width() {
                    assert!(
                        (owned.pixel_at(x, y).0 - reused.pixel_at(x, y).0).abs() < 1e-6,
                        "frame {frame}: mismatch at ({x}, {y}): owned={}, scratch={}",
                        owned.pixel_at(x, y).0,
                        reused.pixel_at(x, y).0,
                    );
                }
            }
        }
    }

    #[test]
    fn gaussian_blur_equals_convolve_with_its_own_kernel() {
        // The claim that retired `gaussian_blur_with`: a σ blur *is* a
        // separable convolution with the σ-derived kernel. If this ever
        // drifts, the migration documented in the changelog is wrong.
        let src = Image::generate(17, 13, |x, y| MonoF32::new((x * 5 + y * 3) as f32));

        for sigma in [sigma!(0.05), sigma!(0.8), sigma!(1.5), sigma!(3.0)] {
            let via_blur: Image<MonoF32> = gaussian_blur(&src, sigma, &Clamp);
            let via_kernel: Image<MonoF32> =
                convolve_separable(&src, &gaussian_kernel_1d(sigma, DEFAULT_TRUNCATE), &Clamp);

            assert_eq!(via_blur.size(), via_kernel.size());
            for y in 0..via_blur.height() {
                for x in 0..via_blur.width() {
                    assert_eq!(
                        via_blur.pixel_at(x, y).0,
                        via_kernel.pixel_at(x, y).0,
                        "σ={}: mismatch at ({x}, {y})",
                        sigma.get(),
                    );
                }
            }
        }

        // Same for the `_into` pair, and for a border policy that shrinks.
        let sigma = sigma!(1.2);
        let kernel = gaussian_kernel_1d(sigma, DEFAULT_TRUNCATE);
        let expected: Image<MonoF32> = convolve_separable(&src, &kernel, &Skip);
        let mut actual = Image::<MonoF32>::zero(expected.width(), expected.height());
        gaussian_blur_into(&src, sigma, &Skip, &mut actual);
        for y in 0..expected.height() {
            for x in 0..expected.width() {
                assert_eq!(expected.pixel_at(x, y).0, actual.pixel_at(x, y).0);
            }
        }
    }

    #[test]
    fn fixed_size_blurs_equal_their_kernel_form() {
        // The four name-encoded sugar functions are documented as equivalent
        // to naming the kernel; that equivalence is pinned here rather than
        // asserted in prose.
        let src = Image::generate(11, 9, |x, y| MonoF32::new((x * 7 + y) as f32));

        let cases: [(Image<MonoF32>, Image<MonoF32>, &str); 4] = [
            (
                gaussian_blur_3x3(&src, &Clamp),
                convolve_separable(&src, &SeparableKernel::gaussian_3(), &Clamp),
                "gaussian_blur_3x3",
            ),
            (
                gaussian_blur_5x5(&src, &Clamp),
                convolve_separable(&src, &SeparableKernel::gaussian_5(), &Clamp),
                "gaussian_blur_5x5",
            ),
            (
                box_blur_3x3(&src, &Clamp),
                convolve_separable(&src, &SeparableKernel::box_blur_3(), &Clamp),
                "box_blur_3x3",
            ),
            (
                box_blur_5x5(&src, &Clamp),
                convolve_separable(&src, &SeparableKernel::box_blur_5(), &Clamp),
                "box_blur_5x5",
            ),
        ];

        for (sugar, kernel_form, name) in cases {
            assert_eq!(sugar.size(), kernel_form.size(), "{name}: size");
            for y in 0..sugar.height() {
                for x in 0..sugar.width() {
                    assert_eq!(
                        sugar.pixel_at(x, y).0,
                        kernel_form.pixel_at(x, y).0,
                        "{name}: mismatch at ({x}, {y})",
                    );
                }
            }
        }
    }

    #[test]
    fn scratch_convolve_separable_into_matches_owned_for_sigma_kernels() {
        // Explicit truncate, a shrinking border policy, and a σ change on
        // the same scratch: the kernel-position buffer is refilled per call,
        // so a different tap count must not carry over.
        let src = Image::generate(16, 11, |x, y| MonoF32::new((x * 2 + y) as f32));
        let mut scratch = SeparableScratch::new();

        for sigma in [sigma!(2.0), sigma!(0.8), sigma!(2.0)] {
            let kernel = gaussian_kernel_1d(sigma, 3.0);
            let owned: Image<MonoF32> = convolve_separable(&src, &kernel, &Skip);

            let mut actual = Image::<MonoF32>::zero(owned.width(), owned.height());
            scratch.convolve_separable_into(&src, &kernel, &Skip, &mut actual);

            for y in 0..owned.height() {
                for x in 0..owned.width() {
                    assert!(
                        (owned.pixel_at(x, y).0 - actual.pixel_at(x, y).0).abs() < 1e-4,
                        "sigma {}: mismatch at ({x}, {y}): owned={}, scratch={}",
                        sigma.get(),
                        owned.pixel_at(x, y).0,
                        actual.pixel_at(x, y).0,
                    );
                }
            }
        }
    }

    #[test]
    fn scratch_gaussian_blur_into_u8_round_trip() {
        // Input and output pixel types differ from the accumulator type the
        // scratch is parameterized on.
        let src = Image::fill(10, 10, Mono8::new(200));
        let mut scratch = SeparableScratch::<MonoF32>::new();
        let mut out = Image::<Mono8>::zero(10, 10);

        scratch.gaussian_blur_into(&src, sigma!(1.2), &Clamp, &mut out);

        for y in 0..out.height() {
            for x in 0..out.width() {
                assert_eq!(out.pixel_at(x, y), Mono8::new(200), "at ({x}, {y})");
            }
        }
    }

    // An invalid sigma is unrepresentable in the `Sigma` parameter type;
    // its rejection is tested at the type's constructors in `common.rs`.

    #[test]
    #[should_panic(expected = "exceeds MAX_RADIUS")]
    fn gaussian_blur_over_radius_sigma_panics() {
        let src = Image::fill(8, 8, MonoF32::new(1.0));
        // radius = round(4.0 * 20.0) = 80 > MAX_RADIUS (64).
        let _: Image<MonoF32> = gaussian_blur(&src, sigma!(20.0), &Clamp);
    }

    #[test]
    fn gaussian_blur_tiny_sigma_is_near_identity() {
        // round(4 * 0.05) = 0 ⇒ 1-tap identity kernel ⇒ input unchanged.
        let src = Image::generate(8, 8, |x, y| MonoF32::new((x * 2 + y) as f32));
        let result: Image<MonoF32> = gaussian_blur(&src, sigma!(0.05), &Clamp);
        for y in 0..result.height() {
            for x in 0..result.width() {
                assert!((result.pixel_at(x, y).0 - src.pixel_at(x, y).0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn smaller_truncate_is_a_smaller_kernel() {
        // Qualitative: a flat image is preserved regardless of truncate, and
        // both truncate values run without panicking on the same input.
        let src = Image::fill(20, 20, MonoF32::new(0.5));
        let r4: Image<MonoF32> =
            convolve_separable(&src, &gaussian_kernel_1d(sigma!(2.0), 4.0), &Clamp);
        let r3: Image<MonoF32> =
            convolve_separable(&src, &gaussian_kernel_1d(sigma!(2.0), 3.0), &Clamp);
        for y in 0..src.height() {
            for x in 0..src.width() {
                assert!((r4.pixel_at(x, y).0 - 0.5).abs() < 1e-4);
                assert!((r3.pixel_at(x, y).0 - 0.5).abs() < 1e-4);
            }
        }
    }

    // ── sobel ───────────────────────────────────────────────────────────

    #[test]
    fn sobel_x_uniform_is_zero() {
        let src = Image::fill(8, 8, Mono8::new(50));
        // `Mono8::Accumulator = MonoF32`.
        let result: Image<crate::pixel::MonoF32> = sobel_x(&src, &Clamp);

        for y in 0..result.height() {
            for x in 0..result.width() {
                assert!(result.pixel_at(x, y).abs().0 < 1e-4);
            }
        }
    }

    #[test]
    fn sobel_y_uniform_is_zero() {
        let src = Image::fill(8, 8, Mono8::new(50));
        let result: Image<crate::pixel::MonoF32> = sobel_y(&src, &Clamp);

        for y in 0..result.height() {
            for x in 0..result.width() {
                assert!(result.pixel_at(x, y).abs().0 < 1e-4);
            }
        }
    }

    #[test]
    fn sobel_x_on_horizontal_gradient() {
        // Horizontal gradient: each column has constant intensity = x
        let src = Image::generate(8, 8, |x, _y| MonoF32::new(x as f32));
        let result: Image<MonoF32> = sobel_x(&src, &Skip);

        // Interior pixels should have non-zero, constant response
        let first = result.pixel_at(0, 0);
        assert!(first.0.abs() > 0.1, "expected non-zero response");
        for y in 0..result.height() {
            for x in 0..result.width() {
                assert!(
                    (result.pixel_at(x, y).0 - first.0).abs() < 1e-4,
                    "at ({x}, {y}): got {}, expected {}",
                    result.pixel_at(x, y).0,
                    first.0,
                );
            }
        }
    }

    #[test]
    fn sobel_y_on_vertical_gradient() {
        // Vertical gradient: each row has constant intensity = y
        let src = Image::generate(8, 8, |_x, y| MonoF32::new(y as f32));
        let result: Image<MonoF32> = sobel_y(&src, &Skip);

        let first = result.pixel_at(0, 0);
        assert!(first.0.abs() > 0.1, "expected non-zero response");
        for y in 0..result.height() {
            for x in 0..result.width() {
                assert!(
                    (result.pixel_at(x, y).0 - first.0).abs() < 1e-4,
                    "at ({x}, {y}): got {}, expected {}",
                    result.pixel_at(x, y).0,
                    first.0,
                );
            }
        }
    }

    // ── scharr ──────────────────────────────────────────────────────────

    #[test]
    fn scharr_x_uniform_is_zero() {
        let src = Image::fill(8, 8, Mono8::new(50));
        let result: Image<crate::pixel::MonoF32> = scharr_x(&src, &Clamp);

        for y in 0..result.height() {
            for x in 0..result.width() {
                assert!(result.pixel_at(x, y).abs().0 < 1e-4);
            }
        }
    }

    #[test]
    fn scharr_y_uniform_is_zero() {
        let src = Image::fill(8, 8, Mono8::new(50));
        let result: Image<crate::pixel::MonoF32> = scharr_y(&src, &Clamp);

        for y in 0..result.height() {
            for x in 0..result.width() {
                assert!(result.pixel_at(x, y).abs().0 < 1e-4);
            }
        }
    }

    // ── prewitt ─────────────────────────────────────────────────────────

    #[test]
    fn prewitt_x_uniform_is_zero() {
        let src = Image::fill(8, 8, Mono8::new(50));
        let result: Image<crate::pixel::MonoF32> = prewitt_x(&src, &Clamp);

        for y in 0..result.height() {
            for x in 0..result.width() {
                assert!(result.pixel_at(x, y).abs().0 < 1e-4);
            }
        }
    }

    #[test]
    fn prewitt_y_uniform_is_zero() {
        let src = Image::fill(8, 8, Mono8::new(50));
        let result: Image<crate::pixel::MonoF32> = prewitt_y(&src, &Clamp);

        for y in 0..result.height() {
            for x in 0..result.width() {
                assert!(result.pixel_at(x, y).abs().0 < 1e-4);
            }
        }
    }

    // ── laplacian ───────────────────────────────────────────────────────

    #[test]
    fn laplacian_uniform_is_zero() {
        let src = Image::fill(8, 8, Mono8::new(10));
        let result: Image<crate::pixel::MonoF32> = laplacian(&src, &Clamp);

        for y in 0..result.height() {
            for x in 0..result.width() {
                assert!(result.pixel_at(x, y).abs().0 < 1e-4);
            }
        }
    }

    #[test]
    fn laplacian_8_uniform_is_zero() {
        let src = Image::fill(8, 8, Mono8::new(10));
        let result: Image<crate::pixel::MonoF32> = laplacian_8(&src, &Clamp);

        for y in 0..result.height() {
            for x in 0..result.width() {
                assert!(result.pixel_at(x, y).abs().0 < 1e-4);
            }
        }
    }

    #[test]
    fn laplacian_matches_full_convolution() {
        let src = make_gradient_8x8();
        let full_kernel = Neighborhood::<f32, 3, 3>::laplacian();
        let full: Image<MonoF32> = convolve(&src, &full_kernel, &Clamp);
        let convenience: Image<MonoF32> = laplacian(&src, &Clamp);

        for y in 0..full.height() {
            for x in 0..full.width() {
                assert!((full.pixel_at(x, y).0 - convenience.pixel_at(x, y).0).abs() < 1e-4);
            }
        }
    }

    // ── sharpen ─────────────────────────────────────────────────────────

    #[test]
    fn sharpen_uniform_is_identity() {
        let src = Image::fill(8, 8, Mono8::new(100));
        let result: Image<Mono8> = sharpen(&src, &Clamp);

        for y in 0..result.height() {
            for x in 0..result.width() {
                assert_eq!(result.pixel_at(x, y), Mono8::new(100));
            }
        }
    }

    #[test]
    fn sharpen_f32_uniform_is_identity() {
        let src = Image::fill(8, 8, MonoF32::new(3.5));
        let result: Image<MonoF32> = sharpen(&src, &Clamp);

        for y in 0..result.height() {
            for x in 0..result.width() {
                assert!(
                    (result.pixel_at(x, y).0 - 3.5).abs() < 1e-4,
                    "at ({x}, {y}): {}",
                    result.pixel_at(x, y).0,
                );
            }
        }
    }

    #[test]
    fn sharpen_matches_full_convolution() {
        let src = make_gradient_8x8();
        let full_kernel = Neighborhood::<f32, 3, 3>::sharpen();
        let full: Image<MonoF32> = convolve(&src, &full_kernel, &Clamp);
        let convenience: Image<MonoF32> = sharpen(&src, &Clamp);

        for y in 0..full.height() {
            for x in 0..full.width() {
                assert!((full.pixel_at(x, y).0 - convenience.pixel_at(x, y).0).abs() < 1e-4);
            }
        }
    }

    // ── emboss ──────────────────────────────────────────────────────────

    #[test]
    fn emboss_uniform_is_original() {
        // Emboss kernel sums to 1, so uniform image × 1 = original
        let src = Image::fill(8, 8, MonoF32::new(25.0));
        let result: Image<MonoF32> = emboss(&src, &Clamp);

        for y in 0..result.height() {
            for x in 0..result.width() {
                assert!(
                    (result.pixel_at(x, y).0 - 25.0).abs() < 1e-4,
                    "at ({x}, {y}): {}",
                    result.pixel_at(x, y).0,
                );
            }
        }
    }

    #[test]
    fn emboss_matches_full_convolution() {
        let src = make_gradient_8x8();
        let full_kernel = Neighborhood::<f32, 3, 3>::emboss();
        let full: Image<MonoF32> = convolve(&src, &full_kernel, &Clamp);
        let convenience: Image<MonoF32> = emboss(&src, &Clamp);

        for y in 0..full.height() {
            for x in 0..full.width() {
                assert!((full.pixel_at(x, y).0 - convenience.pixel_at(x, y).0).abs() < 1e-4);
            }
        }
    }

    // ── edge detectors detect edges ─────────────────────────────────────

    #[test]
    fn sobel_detects_step_edge() {
        // Left half = 0, right half = 100
        let src = Image::generate(10, 10, |x, _y| {
            MonoF32::new(if x < 5 { 0.0 } else { 100.0 })
        });
        let result: Image<MonoF32> = sobel_x(&src, &Clamp);

        // At the edge (x=4,5 boundary), the gradient should be large
        let edge_val = result.pixel_at(4, 5).0.abs();
        let flat_val = result.pixel_at(1, 5).0.abs();
        assert!(
            edge_val > flat_val * 5.0,
            "edge response ({edge_val}) should be much larger than flat ({flat_val})",
        );
    }

    #[test]
    fn laplacian_detects_blob() {
        // A single bright pixel surrounded by zeros
        let mut src = Image::fill(7, 7, MonoF32::new(0.0));
        *src.pixel_at_mut(3, 3) = MonoF32::new(100.0);

        let result: Image<MonoF32> = laplacian(&src, &Clamp);

        // The center pixel should have a strong positive response
        assert!(
            result.pixel_at(3, 3).0 > 200.0,
            "center Laplacian response should be large, got {}",
            result.pixel_at(3, 3).0,
        );
    }

    // ── single-pixel images ─────────────────────────────────────────────

    #[test]
    fn all_filters_handle_single_pixel() {
        let src_f32 = Image::fill(1, 1, MonoF32::new(42.0));
        let src_u8 = Image::fill(1, 1, Mono8::new(42));

        // These should all complete without panicking
        let _: Image<MonoF32> = box_blur_3x3(&src_f32, &Clamp);
        let _: Image<MonoF32> = box_blur_5x5(&src_f32, &Clamp);
        let _: Image<MonoF32> = gaussian_blur_3x3(&src_f32, &Clamp);
        let _: Image<MonoF32> = gaussian_blur_5x5(&src_f32, &Clamp);
        let _: Image<MonoF32> = sobel_x(&src_f32, &Clamp);
        let _: Image<MonoF32> = sobel_y(&src_f32, &Clamp);
        let _: Image<MonoF32> = scharr_x(&src_f32, &Clamp);
        let _: Image<MonoF32> = scharr_y(&src_f32, &Clamp);
        let _: Image<MonoF32> = prewitt_x(&src_f32, &Clamp);
        let _: Image<MonoF32> = prewitt_y(&src_f32, &Clamp);
        let _: Image<MonoF32> = laplacian(&src_f32, &Clamp);
        let _: Image<MonoF32> = laplacian_8(&src_f32, &Clamp);
        let _: Image<MonoF32> = sharpen(&src_f32, &Clamp);
        let _: Image<MonoF32> = emboss(&src_f32, &Clamp);
        // `Mono8::Accumulator = MonoF32`, so the
        // `Mono8` input path produces an `Image<MonoF32>` output.
        let _: Image<crate::pixel::MonoF32> = sobel_x(&src_u8, &Clamp);
        let _: Image<crate::pixel::MonoF32> = sobel_y(&src_u8, &Clamp);
    }

    // ── gradient magnitude / direction ──────────────────────────────────

    #[test]
    fn magnitude_of_axis_gradients() {
        // gx = 3, gy = 4 ⇒ hypot = 5 everywhere.
        let gx = Image::fill(4, 3, MonoF32::new(3.0));
        let gy = Image::fill(4, 3, MonoF32::new(4.0));
        let mag = gradient_magnitude(&gx, &gy).unwrap();
        for y in 0..mag.height() {
            for x in 0..mag.width() {
                assert!((mag.pixel_at(x, y).0 - 5.0).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn magnitude_size_mismatch_err() {
        let gx = Image::fill(2, 2, MonoF32::new(1.0));
        let gy = Image::fill(3, 3, MonoF32::new(1.0));
        assert!(gradient_magnitude(&gx, &gy).is_err());
        assert!(gradient_direction(&gx, &gy).is_err());
    }

    #[test]
    fn direction_cardinal_angles() {
        use std::f32::consts::{FRAC_PI_2, PI};
        let cases = [
            // (gx, gy, expected angle)
            (1.0, 0.0, 0.0),         // pure +x
            (0.0, 1.0, FRAC_PI_2),   // pure +y
            (-1.0, 0.0, PI),         // pure -x
            (0.0, -1.0, -FRAC_PI_2), // pure -y
        ];
        for (gx_v, gy_v, expected) in cases {
            let gx = Image::fill(1, 1, MonoF32::new(gx_v));
            let gy = Image::fill(1, 1, MonoF32::new(gy_v));
            let dir = gradient_direction(&gx, &gy).unwrap();
            assert!(
                (dir.pixel_at(0, 0).0 - expected).abs() < 1e-6,
                "gx={gx_v}, gy={gy_v}: got {}, expected {expected}",
                dir.pixel_at(0, 0).0,
            );
        }
    }

    #[test]
    fn magnitude_direction_generic_over_mono_f64() {
        use crate::pixel::MonoF64;

        // The same wrappers operate on `MonoF64` (the accumulator for
        // `Mono16`/`Mono32`/`Mono64`/`MonoF64` inputs), not just `MonoF32`.
        let gx = Image::fill(2, 2, MonoF64::new(3.0));
        let gy = Image::fill(2, 2, MonoF64::new(4.0));
        let mag = gradient_magnitude(&gx, &gy).unwrap();
        let dir = gradient_direction(&gx, &gy).unwrap();
        assert!((mag.pixel_at(0, 0).0 - 5.0).abs() < 1e-12);
        // f64 atan2 is exercised: atan2(4, 3) ≈ 0.9272952180016122.
        assert!((dir.pixel_at(0, 0).0 - 4.0_f64.atan2(3.0)).abs() < 1e-12);
    }

    // ── non-maximum suppression ─────────────────────────────────────────

    /// Build a `MonoF32` magnitude image from a row-major `f32` grid.
    fn mag_grid(width: usize, height: usize, vals: &[f32]) -> Image<MonoF32> {
        Image::from_vec(
            width,
            height,
            vals.iter().map(|&v| MonoF32::new(v)).collect(),
        )
        .unwrap()
    }

    #[test]
    fn ridge_thins_to_one_pixel() {
        // Horizontal gradient (θ = 0): a [1,2,3,2,1] ridge keeps only the peak.
        let mag = mag_grid(5, 1, &[1.0, 2.0, 3.0, 2.0, 1.0]);
        let dir = Image::fill(5, 1, MonoF32::new(0.0));
        let thin = non_maximum_suppression(&mag, &dir).unwrap();
        let row: Vec<f32> = (0..5).map(|x| thin.pixel_at(x, 0).0).collect();
        assert_eq!(row, vec![0.0, 0.0, 3.0, 0.0, 0.0]);
    }

    #[test]
    fn uniform_magnitude_plateau_kept() {
        // Equal along-gradient neighbours: inclusive `>=` keeps the centre
        // (a strict `>` would erase this flat ridge).
        let mag = mag_grid(3, 1, &[2.0, 2.0, 2.0]);
        let dir = Image::fill(3, 1, MonoF32::new(0.0));
        let thin = non_maximum_suppression(&mag, &dir).unwrap();
        assert_eq!(thin.pixel_at(1, 0).0, 2.0);
    }

    #[test]
    fn border_pixels_suppressed() {
        // θ = 0 compares left/right; the left and right columns have an
        // out-of-bounds neighbour and are suppressed; the centre survives.
        let mag = Image::fill(3, 3, MonoF32::new(5.0));
        let dir = Image::fill(3, 3, MonoF32::new(0.0));
        let thin = non_maximum_suppression(&mag, &dir).unwrap();
        for y in 0..3 {
            assert_eq!(thin.pixel_at(0, y).0, 0.0, "left border at y={y}");
            assert_eq!(thin.pixel_at(2, y).0, 0.0, "right border at y={y}");
            assert_eq!(thin.pixel_at(1, y).0, 5.0, "interior at y={y}");
        }
    }

    #[test]
    fn each_sector_picks_correct_neighbours() {
        use std::f32::consts::PI;
        // For each quantised sector: the two cells *along* the gradient and
        // the two *off* it. A higher along-gradient neighbour must suppress
        // the centre; a higher off-gradient neighbour must not.
        struct Case {
            theta: f32,
            along: [(usize, usize); 2],
            off: [(usize, usize); 2],
        }
        let cases = [
            // 0° → left/right; off = main diagonal.
            Case {
                theta: 0.0,
                along: [(0, 1), (2, 1)],
                off: [(0, 0), (2, 2)],
            },
            // 45° → main diagonal; off = left/right.
            Case {
                theta: PI / 4.0,
                along: [(0, 0), (2, 2)],
                off: [(0, 1), (2, 1)],
            },
            // 90° → up/down; off = main diagonal.
            Case {
                theta: PI / 2.0,
                along: [(1, 0), (1, 2)],
                off: [(0, 0), (2, 2)],
            },
            // 135° → anti-diagonal; off = up/down.
            Case {
                theta: 3.0 * PI / 4.0,
                along: [(2, 0), (0, 2)],
                off: [(1, 0), (1, 2)],
            },
        ];

        for (i, case) in cases.iter().enumerate() {
            let dir = Image::fill(3, 3, MonoF32::new(case.theta));

            // Higher neighbour ALONG the gradient ⇒ centre suppressed.
            let mut mag = Image::fill(3, 3, MonoF32::new(0.0));
            *mag.pixel_at_mut(1, 1) = MonoF32::new(5.0);
            *mag.pixel_at_mut(case.along[0].0, case.along[0].1) = MonoF32::new(9.0);
            let thin = non_maximum_suppression(&mag, &dir).unwrap();
            assert_eq!(thin.pixel_at(1, 1).0, 0.0, "case {i}: should suppress");

            // Higher neighbours only OFF the gradient ⇒ centre kept.
            let mut mag = Image::fill(3, 3, MonoF32::new(0.0));
            *mag.pixel_at_mut(1, 1) = MonoF32::new(5.0);
            for &(x, y) in &case.off {
                *mag.pixel_at_mut(x, y) = MonoF32::new(9.0);
            }
            let thin = non_maximum_suppression(&mag, &dir).unwrap();
            assert_eq!(thin.pixel_at(1, 1).0, 5.0, "case {i}: should keep");
        }
    }

    #[test]
    fn nms_generic_over_mono_f64() {
        use crate::pixel::MonoF64;

        // Suppression on the `MonoF64` accumulator directly, not merely
        // transitively through `canny`. Horizontal gradient, [1,2,3,2,1].
        let mag = Image::from_vec(
            5,
            1,
            [1.0, 2.0, 3.0, 2.0, 1.0]
                .iter()
                .map(|&v| MonoF64::new(v))
                .collect(),
        )
        .unwrap();
        let dir = Image::fill(5, 1, MonoF64::new(0.0));
        let thin = non_maximum_suppression(&mag, &dir).unwrap();
        let row: Vec<f64> = (0..5).map(|x| thin.pixel_at(x, 0).0).collect();
        assert_eq!(row, vec![0.0, 0.0, 3.0, 0.0, 0.0]);

        // A vertical gradient on the same accumulator: θ = π/2 compares
        // up/down, so a single-row image suppresses everything.
        let dir = Image::fill(5, 1, MonoF64::new(std::f64::consts::FRAC_PI_2));
        let thin = non_maximum_suppression(&mag, &dir).unwrap();
        assert!((0..5).all(|x| thin.pixel_at(x, 0).0 == 0.0));
    }

    #[test]
    fn nms_size_mismatch_is_error() {
        let mag = Image::fill(4, 4, MonoF32::new(1.0));
        let dir = Image::fill(3, 4, MonoF32::new(0.0));
        let result: Result<Image<MonoF32>, Error> = non_maximum_suppression(&mag, &dir);
        assert_eq!(
            result.unwrap_err(),
            Error::SizeMismatch {
                expected: Size::new(4, 4),
                actual: Size::new(3, 4),
            }
        );
    }

    // ── fused (gradient-driven) suppression ─────────────────────────────

    #[test]
    fn gradient_sector_matches_angle_sector() {
        use std::f64::consts::{PI, TAU};

        // The fused Canny path buckets sectors straight from (gx, gy); the
        // staged path routes through `atan2`. Sweep a full turn — offset off
        // the 22.5° boundaries, where the two disagree only by float
        // rounding — and require identical sectors.
        for i in 0..3600 {
            let theta = -PI + (i as f64 + 0.37) * TAU / 3600.0;
            let (gx, gy) = (theta.cos(), theta.sin());
            assert_eq!(
                nms_sector_from_gradient(gx, gy),
                nms_sector(gy.atan2(gx)),
                "theta = {theta}",
            );
        }

        // The axis- and diagonal-exact vectors, which the sweep skips.
        for &(gx, gy) in &[
            (1.0, 0.0),
            (0.0, 1.0),
            (-1.0, 0.0),
            (0.0, -1.0),
            (1.0, 1.0),
            (-1.0, 1.0),
            (1.0, -1.0),
            (-1.0, -1.0),
            (3.0, 0.5),
            (-0.25, 7.0),
        ] {
            assert_eq!(
                nms_sector_from_gradient(gx, gy),
                nms_sector(gy.atan2(gx)),
                "(gx, gy) = ({gx}, {gy})",
            );
        }
    }

    #[test]
    fn fused_nms_matches_staged_nms() {
        // Same inputs, both routes: the fused suppression must reproduce
        // `gradient_direction` + `non_maximum_suppression` exactly.
        let src = Image::generate(11, 9, |x, y| {
            MonoF32::new(((x * 13 + y * 7) % 5) as f32 * 0.25 + (x as f32 * 0.1).sin())
        });
        let gx = scharr_x(&src, &Clamp);
        let gy = scharr_y(&src, &Clamp);
        let mag = gradient_magnitude(&gx, &gy).unwrap();
        let dir = gradient_direction(&gx, &gy).unwrap();

        let staged = non_maximum_suppression(&mag, &dir).unwrap();
        let fused = non_maximum_suppression_from_gradients(&mag, &gx, &gy);
        for y in 0..9 {
            for x in 0..11 {
                assert_eq!(staged.pixel_at(x, y).0, fused.pixel_at(x, y).0, "({x},{y})");
            }
        }
    }

    #[test]
    #[should_panic(expected = "must have the same size")]
    fn fused_nms_size_mismatch_panics() {
        let mag = Image::fill(4, 4, MonoF32::new(1.0));
        let gx = Image::fill(4, 4, MonoF32::new(1.0));
        let gy = Image::fill(3, 3, MonoF32::new(1.0));
        let _ = non_maximum_suppression_from_gradients(&mag, &gx, &gy);
    }
}

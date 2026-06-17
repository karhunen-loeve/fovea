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
use crate::image::{
    Image, Neighborhood, RasterImage, RasterImageMut, SeparableKernel, gaussian_kernel_1d,
};
use crate::pixel::{FromLinear, LinearPixel, ZeroablePixel};
use crate::transform::convolve::convolve;
use crate::transform::convolve_separable::{
    convolve_separable, convolve_separable_raw, convolve_separable_raw_into,
};

// ─── Box blur ────────────────────────────────────────────────────────────────

/// 3×3 box blur using a separable two-pass implementation.
///
/// Each weight is `1/3`, applied horizontally then vertically, giving an
/// effective `1/9` per pixel — identical to [`Neighborhood::box_blur_3x3`].
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
/// (`filters.gaussian`), capturing the Gaussian out to 4σ. Use
/// [`gaussian_blur_with`] to pick a smaller `truncate` (e.g. `3.0`) for
/// faster, slightly tighter kernels.
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
/// Use [`gaussian_blur_with`] to override `truncate`.
///
/// # Panics
///
/// Panics (Tier 3 precondition) if `sigma <= 0.0`, or if the derived radius
/// exceeds [`MAX_RADIUS`](crate::image::MAX_RADIUS) (i.e. `sigma` is larger
/// than `MAX_RADIUS / truncate`).
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView};
/// use fovea::border::Clamp;
/// use fovea::pixel::MonoF32;
/// use fovea::transform::gaussian_blur;
///
/// // A flat image is returned unchanged (brightness preserved).
/// let src = Image::fill(16, 16, MonoF32::new(0.7));
/// let result: Image<MonoF32> = gaussian_blur(&src, 2.0, &Clamp);
/// for y in 0..result.height() {
///     for x in 0..result.width() {
///         assert!((result.pixel_at(x, y).0 - 0.7).abs() < 1e-4);
///     }
/// }
/// ```
#[must_use]
pub fn gaussian_blur<I, B, P, Acc, Out>(image: &I, sigma: f32, border: &B) -> Image<Out>
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
    gaussian_blur_with(image, sigma, DEFAULT_TRUNCATE, border)
}

/// Gaussian blur derived from `sigma` with an explicit `truncate`.
///
/// As [`gaussian_blur`], but the kernel radius is `round(truncate * sigma)`.
/// A smaller `truncate` (e.g. `3.0`) yields a smaller, faster kernel
/// capturing slightly less of the Gaussian tail; the default of `4.0`
/// matches SciPy / scikit-image.
///
/// # Panics
///
/// Panics if `sigma <= 0.0` or `truncate <= 0.0`, or if the derived radius
/// exceeds [`MAX_RADIUS`](crate::image::MAX_RADIUS).
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView};
/// use fovea::border::Clamp;
/// use fovea::pixel::MonoF32;
/// use fovea::transform::gaussian_blur_with;
///
/// let src = Image::fill(16, 16, MonoF32::new(0.5));
/// let result: Image<MonoF32> = gaussian_blur_with(&src, 1.5, 3.0, &Clamp);
/// assert_eq!(result.size(), src.size());
/// ```
#[must_use]
pub fn gaussian_blur_with<I, B, P, Acc, Out>(
    image: &I,
    sigma: f32,
    truncate: f32,
    border: &B,
) -> Image<Out>
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
    let kernel = gaussian_kernel_1d(sigma, truncate);
    let weights = kernel.weights();
    let n = kernel.len();
    let anchor = kernel.anchor();

    // The kernel is symmetric ⇒ identical horizontal and vertical weights.
    let h_img = Image::generate(n, 1, |x, _| weights[x]);
    let v_img = Image::generate(1, n, |_, y| weights[y]);

    convolve_separable_raw(image, &h_img, anchor, &v_img, anchor, border)
}

/// Gaussian blur derived from `sigma`, writing into a caller-owned output.
///
/// As [`gaussian_blur`], but writes into `output` instead of allocating.
///
/// # Panics
///
/// Panics if `sigma <= 0.0`, if the derived radius exceeds
/// [`MAX_RADIUS`](crate::image::MAX_RADIUS), or if `output` is too small for
/// the region produced by the border policy.
pub fn gaussian_blur_into<I, B, O, P, Acc, Out>(image: &I, sigma: f32, border: &B, output: &mut O)
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
    gaussian_blur_with_into(image, sigma, DEFAULT_TRUNCATE, border, output);
}

/// Gaussian blur derived from `sigma` and an explicit `truncate`, writing
/// into a caller-owned output.
///
/// As [`gaussian_blur_with`], but writes into `output` instead of
/// allocating.
///
/// # Panics
///
/// Panics if `sigma <= 0.0` or `truncate <= 0.0`, if the derived radius
/// exceeds [`MAX_RADIUS`](crate::image::MAX_RADIUS), or if `output` is too
/// small for the region produced by the border policy.
pub fn gaussian_blur_with_into<I, B, O, P, Acc, Out>(
    image: &I,
    sigma: f32,
    truncate: f32,
    border: &B,
    output: &mut O,
) where
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
    let kernel = gaussian_kernel_1d(sigma, truncate);
    let weights = kernel.weights();
    let n = kernel.len();
    let anchor = kernel.anchor();

    let h_img = Image::generate(n, 1, |x, _| weights[x]);
    let v_img = Image::generate(1, n, |_, y| weights[y]);

    convolve_separable_raw_into(image, &h_img, anchor, &v_img, anchor, border, output);
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

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::border::{Clamp, Constant, Skip};
    use crate::image::{ImageView, ImageViewMut, gaussian_kernel_1d};
    use crate::pixel::{Mono8, MonoF32};
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
        let result: Image<MonoF32> = gaussian_blur(&src, 2.0, &Clamp);
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
        let result: Image<Mono8> = gaussian_blur(&src, 1.5, &Clamp);
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
        let sigma = 1.0;
        let truncate = 2.0; // radius 2, 5 taps
        let kernel = gaussian_kernel_1d(sigma, truncate);
        let w = kernel.weights();
        let r = kernel.radius();

        let c = 5usize;
        let mut src = Image::fill(11, 11, MonoF32::new(0.0));
        *src.pixel_at_mut(c, c) = MonoF32::new(1.0);

        let out: Image<MonoF32> =
            gaussian_blur_with(&src, sigma, truncate, &Constant(MonoF32(0.0)));

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
        let sigma = 1.0;
        let truncate = 2.0; // radius 2
        let kernel = gaussian_kernel_1d(sigma, truncate);
        let w = kernel.weights();
        let r = kernel.radius();

        let src = Image::generate(9, 9, |x, y| MonoF32::new((x * 3 + y * 5) as f32));
        let out: Image<MonoF32> = gaussian_blur_with(&src, sigma, truncate, &Clamp);

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
            let blurred: Image<MonoF32> = gaussian_blur(&src, sigma, &Clamp);
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

        let owned: Image<MonoF32> = gaussian_blur(&src, 1.5, &Clamp);

        let mut into = Image::<MonoF32>::zero(owned.width(), owned.height());
        gaussian_blur_into(&src, 1.5, &Clamp, &mut into);

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
    #[should_panic(expected = "sigma must be > 0.0")]
    fn gaussian_blur_zero_sigma_panics() {
        let src = Image::fill(8, 8, MonoF32::new(1.0));
        let _: Image<MonoF32> = gaussian_blur(&src, 0.0, &Clamp);
    }

    #[test]
    #[should_panic(expected = "exceeds MAX_RADIUS")]
    fn gaussian_blur_over_radius_sigma_panics() {
        let src = Image::fill(8, 8, MonoF32::new(1.0));
        // radius = round(4.0 * 20.0) = 80 > MAX_RADIUS (64).
        let _: Image<MonoF32> = gaussian_blur(&src, 20.0, &Clamp);
    }

    #[test]
    fn gaussian_blur_tiny_sigma_is_near_identity() {
        // round(4 * 0.05) = 0 ⇒ 1-tap identity kernel ⇒ input unchanged.
        let src = Image::generate(8, 8, |x, y| MonoF32::new((x * 2 + y) as f32));
        let result: Image<MonoF32> = gaussian_blur(&src, 0.05, &Clamp);
        for y in 0..result.height() {
            for x in 0..result.width() {
                assert!((result.pixel_at(x, y).0 - src.pixel_at(x, y).0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn gaussian_blur_with_smaller_truncate_is_smaller_kernel() {
        // Qualitative: a flat image is preserved regardless of truncate, and
        // both truncate values run without panicking on the same input.
        let src = Image::fill(20, 20, MonoF32::new(0.5));
        let r4: Image<MonoF32> = gaussian_blur_with(&src, 2.0, 4.0, &Clamp);
        let r3: Image<MonoF32> = gaussian_blur_with(&src, 2.0, 3.0, &Clamp);
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
}

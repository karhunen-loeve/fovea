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
use crate::image::{Image, Neighborhood, RasterImage, SeparableKernel};
use crate::pixel::{FromLinear, LinearPixel, ZeroablePixel};
use crate::transform::convolve::convolve;
use crate::transform::convolve_separable::convolve_separable;

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
/// Uses the `[1, 2, 1]` kernel in each direction, giving the standard
/// discrete 3×3 Gaussian approximation. The combined kernel sums to 16,
/// so this is the **un-normalised** Gaussian — identical to convolving
/// with [`Neighborhood::gaussian_3x3`].
///
/// If you need a normalised blur (output ≈ input magnitude), divide the
/// result by 16 or supply pre-normalised 1D kernels.
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
/// // Un-normalised: 1.0 × 16 = 16.0
/// for y in 0..result.height() {
///     for x in 0..result.width() {
///         assert!((result.pixel_at(x, y).0 - 16.0).abs() < 1e-4);
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
/// Uses the `[1, 4, 6, 4, 1]` kernel in each direction, giving the
/// standard discrete 5×5 Gaussian approximation. The combined kernel
/// sums to 256, so this is the **un-normalised** Gaussian — identical
/// to convolving with [`Neighborhood::gaussian_5x5`].
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
/// // Un-normalised: 1.0 × 256 = 256.0
/// for y in 0..result.height() {
///     for x in 0..result.width() {
///         assert!((result.pixel_at(x, y).0 - 256.0).abs() < 1e-2);
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
    use crate::border::{Clamp, Skip};
    use crate::image::{ImageView, ImageViewMut};
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
        // gaussian_3x3 is un-normalised (sum=16), so uniform×16 is expected
        let src = Image::fill(8, 8, MonoF32::new(1.0));
        let result: Image<MonoF32> = gaussian_blur_3x3(&src, &Clamp);

        for y in 0..result.height() {
            for x in 0..result.width() {
                assert!(
                    (result.pixel_at(x, y).0 - 16.0).abs() < 1e-3,
                    "at ({x}, {y}): {}",
                    result.pixel_at(x, y).0,
                );
            }
        }
    }

    #[test]
    fn gaussian_blur_3x3_matches_full_convolution() {
        let src = make_gradient_8x8();
        let full_kernel = Neighborhood::<f32, 3, 3>::gaussian_3x3();
        let full: Image<MonoF32> = convolve(&src, &full_kernel, &Clamp);
        let sep: Image<MonoF32> = gaussian_blur_3x3(&src, &Clamp);

        for y in 0..full.height() {
            for x in 0..full.width() {
                assert!(
                    (full.pixel_at(x, y).0 - sep.pixel_at(x, y).0).abs() < 1e-2,
                    "mismatch at ({x}, {y})",
                );
            }
        }
    }

    #[test]
    fn gaussian_blur_5x5_uniform_f32() {
        let src = Image::fill(10, 10, MonoF32::new(1.0));
        let result: Image<MonoF32> = gaussian_blur_5x5(&src, &Clamp);

        for y in 0..result.height() {
            for x in 0..result.width() {
                assert!((result.pixel_at(x, y).0 - 256.0).abs() < 1e-1);
            }
        }
    }

    #[test]
    fn gaussian_blur_5x5_matches_full_convolution() {
        let src = make_gradient_8x8();
        let full_kernel = Neighborhood::<f32, 5, 5>::gaussian_5x5();
        let full: Image<MonoF32> = convolve(&src, &full_kernel, &Clamp);
        let sep: Image<MonoF32> = gaussian_blur_5x5(&src, &Clamp);

        for y in 0..full.height() {
            for x in 0..full.width() {
                assert!(
                    (full.pixel_at(x, y).0 - sep.pixel_at(x, y).0).abs() < 1e-1,
                    "mismatch at ({x}, {y}): full={}, sep={}",
                    full.pixel_at(x, y).0,
                    sep.pixel_at(x, y).0,
                );
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

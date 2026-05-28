//! 2D convolution and correlation wrappers built on [`fold_neighborhood`].
//!
//! Convolution is closely related to correlation — the only difference is
//! that convolution flips (rotates 180°) the kernel before sliding it
//! across the image. For symmetric kernels the two operations are
//! identical.
//!
//! This module provides:
//!
//! - [`convolve_into`] / [`convolve`] — true convolution (kernel is flipped)
//! - [`correlate_into`] / [`correlate`] — cross-correlation (kernel as-is)
//!
//! All functions require:
//! - `P: LinearPixel<f32>` — source pixels support `scale(f32)`
//! - `P::Accumulator: Default` — zero-initialisation of the running sum
//! - `Out: FromLinear<P::Accumulator>` — convert accumulated value to output pixel

use crate::border::BorderPolicy;
use crate::image::Kernel;
use crate::image::{Image, RasterImage, RasterImageMut};
use crate::pixel::{FromLinear, LinearPixel, ZeroablePixel};
use crate::transform::fold::{FoldItem, FoldOp, fold_neighborhood, fold_neighborhood_into};

// ─── ConvolveFold ────────────────────────────────────────────────────────────

/// A [`FoldOp`] that computes the weighted sum used by convolution and
/// correlation.
///
/// For each neighbor, it scales the source pixel by the kernel weight and
/// accumulates: `Out::from_linear( Σ pixel_i.scale(weight_i) )`.
///
/// This struct replaces the old `convolve_fold` closure. Because it
/// implements `FoldOp` with a generic `fold` method, both the interior
/// (hot) and boundary (cold) paths are fully monomorphized — no `dyn
/// Iterator` vtable dispatch.
pub(crate) struct ConvolveFold<P, Out> {
    _marker: core::marker::PhantomData<(P, Out)>,
}

impl<P, Out> ConvolveFold<P, Out> {
    #[inline(always)]
    pub(crate) fn new() -> Self {
        Self {
            _marker: core::marker::PhantomData,
        }
    }
}

impl<P, Out> FoldOp<P, f32> for ConvolveFold<P, Out>
where
    P: Copy + LinearPixel<f32>,
    <P as LinearPixel<f32>>::Accumulator: Default,
    Out: FromLinear<<P as LinearPixel<f32>>::Accumulator>,
{
    type Accumulator = <P as LinearPixel<f32>>::Accumulator;
    type Output = Out;

    #[inline(always)]
    fn init(&self) -> Self::Accumulator {
        <P as LinearPixel<f32>>::Accumulator::default()
    }

    #[inline(always)]
    fn accumulate(&self, acc: &mut Self::Accumulator, item: FoldItem<P, f32>) {
        *acc = item.pixel.scale_add(item.weight, *acc);
    }

    #[inline(always)]
    fn finalize(&mut self, acc: Self::Accumulator) -> Out {
        Out::from_linear(acc)
    }
}

/// Write the result of convolving `image` with the given kernel into `output`.
///
/// This is the **base method** — [`convolve`] is a convenience wrapper
/// that allocates the output for you.
///
/// # Convolution vs. correlation
///
/// Convolution rotates the kernel 180° before sliding. This function
/// uses [`Kernel::flipped`] to obtain the rotated kernel, then delegates
/// to [`fold_neighborhood_into`].
///
/// For symmetric kernels (box blur, Gaussian, Laplacian, etc.) the flip
/// has no effect. It only matters for asymmetric kernels like Sobel or
/// Prewitt.
///
/// # Panics
///
/// Panics if `output` is smaller than the region returned by
/// `border.output_region()`.
///
/// # Example
///
/// ```
/// use irys_cv::image::{Image, ImageView, ImageViewMut, Neighborhood};
/// use irys_cv::Size;
/// use irys_cv::border::Clamp;
/// use irys_cv::pixel::MonoF32;
/// use irys_cv::transform::convolve_into;
///
/// let src = Image::fill(5, 5, MonoF32(1.0));
/// let kernel = Neighborhood::<f32, 3, 3>::identity_3x3();
///
/// let border = Clamp;
/// let out_region = irys_cv::border::BorderPolicy::<Image<MonoF32>>::output_region(
///     &border, src.size(), kernel.weights().size(), kernel.anchor(),
/// );
/// let mut out = Image::<MonoF32>::zero(out_region.size.width, out_region.size.height);
///
/// convolve_into(&src, &kernel, &border, &mut out);
///
/// for y in 0..out.height() {
///     for x in 0..out.width() {
///         assert!((out.pixel_at(x, y).0 - 1.0).abs() < 1e-6);
///     }
/// }
/// ```
///
/// **Note:** Convolution flips the kernel before applying it. For the
/// unflipped variant (cross-correlation), see [`correlate_into`]. The
/// [module docs](crate::transform) explain the mathematical distinction.
pub fn convolve_into<I, K, B, O, P, Out>(image: &I, kernel: &K, border: &B, output: &mut O)
where
    I: RasterImage<Pixel = P>,
    P: Copy + LinearPixel<f32>,
    <P as LinearPixel<f32>>::Accumulator: Default,
    K: Kernel<Weight = f32>,
    B: BorderPolicy<I>,
    O: RasterImageMut<Pixel = Out>,
    Out: FromLinear<<P as LinearPixel<f32>>::Accumulator>,
{
    // True convolution = correlation with a 180°-rotated kernel.
    // Kernel::flipped() preserves the concrete type, so stack-backed
    // kernels stay on the stack (zero heap allocation).
    let flipped = kernel.flipped();

    fold_neighborhood_into(
        image,
        flipped.weights(),
        flipped.anchor(),
        border,
        output,
        ConvolveFold::<P, Out>::new(),
    );
}

/// Convolve `image` with the given kernel and return a newly allocated
/// output [`Image`].
///
/// This is a convenience wrapper around [`convolve_into`]. The output
/// size is determined by `border.output_region(…)`.
///
/// # Example
///
/// ```
/// use irys_cv::image::{Image, ImageView, Neighborhood};
/// use irys_cv::border::Clamp;
/// use irys_cv::pixel::Mono8;
/// use irys_cv::transform::convolve;
///
/// let src = Image::fill(5, 5, Mono8::new(10));
/// let kernel = Neighborhood::<f32, 3, 3>::box_blur_3x3();
///
/// let result: Image<Mono8> = convolve(&src, &kernel, &Clamp);
///
/// // Uniform image convolved with box blur stays the same
/// for y in 0..result.height() {
///     for x in 0..result.width() {
///         assert_eq!(result.pixel_at(x, y), Mono8::new(10));
///     }
/// }
/// ```
///
/// **Note:** Convolution flips the kernel before applying it. For the
/// unflipped variant (cross-correlation), see [`correlate`]. The
/// [module docs](crate::transform) explain the mathematical distinction.
#[must_use]
pub fn convolve<I, K, B, P, Out>(image: &I, kernel: &K, border: &B) -> Image<Out>
where
    I: RasterImage<Pixel = P>,
    P: Copy + LinearPixel<f32>,
    <P as LinearPixel<f32>>::Accumulator: Default,
    K: Kernel<Weight = f32>,
    B: BorderPolicy<I>,
    Out: ZeroablePixel + FromLinear<<P as LinearPixel<f32>>::Accumulator>,
{
    let flipped = kernel.flipped();

    fold_neighborhood(
        image,
        flipped.weights(),
        flipped.anchor(),
        border,
        ConvolveFold::<P, Out>::new(),
    )
}

/// Perform correlation (cross-correlation) — identical to convolution but
/// **without** flipping the kernel.
///
/// This is useful when you already have the kernel in the orientation you
/// want (e.g., for template matching or when the kernel is symmetric).
///
/// # Panics
///
/// Panics if `output` is smaller than the region returned by
/// `border.output_region()`.
///
/// # Example
///
/// ```
/// use irys_cv::image::{Image, ImageView, ImageViewMut, Neighborhood};
/// use irys_cv::Size;
/// use irys_cv::border::Clamp;
/// use irys_cv::pixel::MonoF32;
/// use irys_cv::transform::correlate_into;
///
/// let src = Image::fill(5, 5, MonoF32(2.0));
/// let kernel = Neighborhood::<f32, 3, 3>::identity_3x3();
///
/// let border = Clamp;
/// let out_region = irys_cv::border::BorderPolicy::<Image<MonoF32>>::output_region(
///     &border, src.size(), kernel.weights().size(), kernel.anchor(),
/// );
/// let mut out = Image::<MonoF32>::zero(out_region.size.width, out_region.size.height);
///
/// correlate_into(&src, &kernel, &border, &mut out);
///
/// for y in 0..out.height() {
///     for x in 0..out.width() {
///         assert!((out.pixel_at(x, y).0 - 2.0).abs() < 1e-6);
///     }
/// }
/// ```
///
/// **Note:** Correlation applies the kernel without flipping. For the
/// flipped variant (true convolution), see [`convolve_into`]. The
/// [module docs](crate::transform) explain the mathematical distinction.
pub fn correlate_into<I, K, B, O, P, Out>(image: &I, kernel: &K, border: &B, output: &mut O)
where
    I: RasterImage<Pixel = P>,
    P: Copy + LinearPixel<f32>,
    <P as LinearPixel<f32>>::Accumulator: Default,
    K: Kernel<Weight = f32>,
    B: BorderPolicy<I>,
    O: RasterImageMut<Pixel = Out>,
    Out: FromLinear<<P as LinearPixel<f32>>::Accumulator>,
{
    fold_neighborhood_into(
        image,
        kernel.weights(),
        kernel.anchor(),
        border,
        output,
        ConvolveFold::<P, Out>::new(),
    );
}

/// Perform correlation and return a newly allocated output [`Image`].
///
/// Correlation is identical to convolution but without flipping the kernel.
/// See [`correlate_into`] for the base method.
///
/// # Example
///
/// ```
/// use irys_cv::image::{Image, ImageView, Neighborhood};
/// use irys_cv::border::Clamp;
/// use irys_cv::pixel::MonoF32;
/// use irys_cv::transform::correlate;
///
/// let src = Image::fill(5, 5, MonoF32(2.0));
/// let kernel = Neighborhood::<f32, 3, 3>::identity_3x3();
///
/// let result: Image<MonoF32> = correlate(&src, &kernel, &Clamp);
///
/// for y in 0..result.height() {
///     for x in 0..result.width() {
///         assert!((result.pixel_at(x, y).0 - 2.0).abs() < 1e-6);
///     }
/// }
/// ```
///
/// **Note:** Correlation applies the kernel without flipping. For the
/// flipped variant (true convolution), see [`convolve`]. The
/// [module docs](crate::transform) explain the mathematical distinction.
#[must_use]
pub fn correlate<I, K, B, P, Out>(image: &I, kernel: &K, border: &B) -> Image<Out>
where
    I: RasterImage<Pixel = P>,
    P: Copy + LinearPixel<f32>,
    <P as LinearPixel<f32>>::Accumulator: Default,
    K: Kernel<Weight = f32>,
    B: BorderPolicy<I>,
    Out: ZeroablePixel + FromLinear<<P as LinearPixel<f32>>::Accumulator>,
{
    fold_neighborhood(
        image,
        kernel.weights(),
        kernel.anchor(),
        border,
        ConvolveFold::<P, Out>::new(),
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::border::{Clamp, Constant, Skip};
    use crate::image::{ImageView, Neighborhood};
    use crate::pixel::{Mono8, MonoF32};

    // ── helpers ──────────────────────────────────────────────────────────

    fn make_4x4_monof32() -> Image<MonoF32> {
        Image::generate(4, 4, |x, y| MonoF32((x + y * 4) as f32))
    }

    // ── identity kernel ─────────────────────────────────────────────────

    #[test]
    fn convolve_identity_preserves_image() {
        let src = make_4x4_monof32();
        let kernel = Neighborhood::<f32, 3, 3>::identity_3x3();
        let result: Image<MonoF32> = convolve(&src, &kernel, &Clamp);

        assert_eq!(result.width(), 4);
        assert_eq!(result.height(), 4);
        for y in 0..4 {
            for x in 0..4 {
                assert!(
                    (result.pixel_at(x, y).0 - src.pixel_at(x, y).0).abs() < 1e-6,
                    "mismatch at ({x}, {y}): got {}, expected {}",
                    result.pixel_at(x, y).0,
                    src.pixel_at(x, y).0,
                );
            }
        }
    }

    #[test]
    fn correlate_identity_preserves_image() {
        let src = make_4x4_monof32();
        let kernel = Neighborhood::<f32, 3, 3>::identity_3x3();
        let result: Image<MonoF32> = correlate(&src, &kernel, &Clamp);

        for y in 0..4 {
            for x in 0..4 {
                assert!((result.pixel_at(x, y).0 - src.pixel_at(x, y).0).abs() < 1e-6,);
            }
        }
    }

    // ── symmetric kernel: convolve == correlate ─────────────────────────

    #[test]
    fn symmetric_kernel_convolve_equals_correlate() {
        let src = make_4x4_monof32();
        let kernel = Neighborhood::<f32, 3, 3>::box_blur_3x3();

        let conv: Image<MonoF32> = convolve(&src, &kernel, &Clamp);
        let corr: Image<MonoF32> = correlate(&src, &kernel, &Clamp);

        assert_eq!(conv.width(), corr.width());
        assert_eq!(conv.height(), corr.height());
        for y in 0..conv.height() {
            for x in 0..conv.width() {
                assert!(
                    (conv.pixel_at(x, y).0 - corr.pixel_at(x, y).0).abs() < 1e-6,
                    "mismatch at ({x}, {y})",
                );
            }
        }
    }

    // ── box blur on uniform image ───────────────────────────────────────

    #[test]
    fn box_blur_uniform_image() {
        let src = Image::fill(6, 6, MonoF32(5.0));
        let kernel = Neighborhood::<f32, 3, 3>::box_blur_3x3();

        let result: Image<MonoF32> = convolve(&src, &kernel, &Clamp);

        for y in 0..result.height() {
            for x in 0..result.width() {
                assert!(
                    (result.pixel_at(x, y).0 - 5.0).abs() < 1e-5,
                    "at ({x}, {y}): {}",
                    result.pixel_at(x, y).0,
                );
            }
        }
    }

    // ── asymmetric kernel: convolve ≠ correlate ─────────────────────────

    #[test]
    fn asymmetric_kernel_convolve_differs_from_correlate() {
        // Sobel-Y kernel: [-1 0 1; -2 0 2; -1 0 1]
        // Flipped (180°):  [1 0 -1;  2 0 -2;  1 0 -1] = -sobel_y
        // So convolve(sobel_y) = correlate(-sobel_y) = -correlate(sobel_y).
        //
        // Use an image with a clear horizontal gradient so the response
        // is non-zero.
        let src = Image::generate(6, 6, |x, _y| MonoF32(x as f32));
        let kernel = Neighborhood::<f32, 3, 3>::sobel_y();

        let conv: Image<MonoF32> = convolve(&src, &kernel, &Clamp);
        let corr: Image<MonoF32> = correlate(&src, &kernel, &Clamp);

        // For this anti-symmetric kernel, convolve = -correlate
        let mut found_nonzero = false;
        for y in 0..conv.height() {
            for x in 0..conv.width() {
                let sum = conv.pixel_at(x, y).0 + corr.pixel_at(x, y).0;
                assert!(
                    sum.abs() < 1e-4,
                    "conv + corr should be ~0 at ({x}, {y}): conv={}, corr={}, sum={}",
                    conv.pixel_at(x, y).0,
                    corr.pixel_at(x, y).0,
                    sum,
                );
                if conv.pixel_at(x, y).0.abs() > 0.1 {
                    found_nonzero = true;
                }
            }
        }
        assert!(
            found_nonzero,
            "expected non-zero response on gradient image"
        );
    }

    // ── convolve_into writes correct output ──────────────────────────────

    #[test]
    fn convolve_into_writes_correct_output() {
        let src = Image::fill(4, 4, MonoF32(3.0));
        let kernel = Neighborhood::<f32, 3, 3>::identity_3x3();
        let border = Clamp;
        let out_region = BorderPolicy::<Image<MonoF32>>::output_region(
            &border,
            src.size(),
            kernel.weights().size(),
            kernel.anchor(),
        );
        let mut out = Image::<MonoF32>::zero(out_region.size.width, out_region.size.height);

        convolve_into(&src, &kernel, &border, &mut out);

        for y in 0..out.height() {
            for x in 0..out.width() {
                assert!((out.pixel_at(x, y).0 - 3.0).abs() < 1e-6);
            }
        }
    }

    // ── Skip border policy ──────────────────────────────────────────────

    #[test]
    fn convolve_with_skip_shrinks_output() {
        let src = Image::generate(6, 6, |x, y| MonoF32((x + y) as f32));
        let kernel = Neighborhood::<f32, 3, 3>::identity_3x3();

        let result: Image<MonoF32> = convolve(&src, &kernel, &Skip);

        // Skip with 3×3 kernel on 6×6 image → 4×4 output
        assert_eq!(result.width(), 4);
        assert_eq!(result.height(), 4);

        // Identity kernel should reproduce interior pixels
        for y in 0..4 {
            for x in 0..4 {
                assert!((result.pixel_at(x, y).0 - src.pixel_at(x + 1, y + 1).0).abs() < 1e-6,);
            }
        }
    }

    // ── Constant border policy ──────────────────────────────────────────

    #[test]
    fn convolve_constant_border_zero_padding() {
        let src = Image::fill(3, 3, MonoF32(1.0));
        let kernel = Neighborhood::<f32, 3, 3>::box_blur_3x3();
        let border = Constant(MonoF32(0.0));

        let result: Image<MonoF32> = convolve(&src, &kernel, &border);

        assert_eq!(result.width(), 3);
        assert_eq!(result.height(), 3);

        // Center pixel: all 9 neighbors are 1.0 → average = 1.0
        assert!((result.pixel_at(1, 1).0 - 1.0).abs() < 1e-5);

        // Corner pixel (0,0): 4 neighbors are 1.0, 5 are 0.0 → average = 4/9
        assert!((result.pixel_at(0, 0).0 - 4.0 / 9.0).abs() < 1e-5);

        // Edge pixel (1,0): 6 neighbors are 1.0, 3 are 0.0 → average = 6/9
        assert!((result.pixel_at(1, 0).0 - 6.0 / 9.0).abs() < 1e-5);
    }

    // ── u8 pixel type ───────────────────────────────────────────────────

    #[test]
    fn convolve_u8_identity() {
        let src = Image::generate(5, 5, |x, y| Mono8::new(((x + y * 5) % 256) as u8));
        let kernel = Neighborhood::<f32, 3, 3>::identity_3x3();

        let result: Image<Mono8> = convolve(&src, &kernel, &Clamp);

        for y in 0..5 {
            for x in 0..5 {
                assert_eq!(result.pixel_at(x, y), src.pixel_at(x, y));
            }
        }
    }

    #[test]
    fn convolve_u8_box_blur_uniform() {
        let src = Image::fill(4, 4, Mono8::new(100));
        let kernel = Neighborhood::<f32, 3, 3>::box_blur_3x3();

        let result: Image<Mono8> = convolve(&src, &kernel, &Clamp);

        for y in 0..result.height() {
            for x in 0..result.width() {
                assert_eq!(result.pixel_at(x, y), Mono8::new(100));
            }
        }
    }

    // ── 5×5 kernel ──────────────────────────────────────────────────────

    #[test]
    fn convolve_5x5_box_blur_uniform() {
        let src = Image::fill(8, 8, MonoF32(7.0));
        let kernel = Neighborhood::<f32, 5, 5>::box_blur_5x5();

        let result: Image<MonoF32> = convolve(&src, &kernel, &Clamp);

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

    // ── convolve and convolve_into produce same result ──────────────────

    #[test]
    fn convolve_and_convolve_into_match() {
        let src = make_4x4_monof32();
        let kernel = Neighborhood::<f32, 3, 3>::gaussian_3x3();

        let result_alloc: Image<MonoF32> = convolve(&src, &kernel, &Clamp);

        let border = Clamp;
        let out_region = BorderPolicy::<Image<MonoF32>>::output_region(
            &border,
            src.size(),
            kernel.weights().size(),
            kernel.anchor(),
        );
        let mut result_into = Image::<MonoF32>::zero(out_region.size.width, out_region.size.height);
        convolve_into(&src, &kernel, &border, &mut result_into);

        for y in 0..result_alloc.height() {
            for x in 0..result_alloc.width() {
                assert!(
                    (result_alloc.pixel_at(x, y).0 - result_into.pixel_at(x, y).0).abs() < 1e-6,
                );
            }
        }
    }

    // ── correlate and correlate_into produce same result ─────────────────

    #[test]
    fn correlate_and_correlate_into_match() {
        let src = make_4x4_monof32();
        let kernel = Neighborhood::<f32, 3, 3>::sobel_x();

        let result_alloc: Image<MonoF32> = correlate(&src, &kernel, &Clamp);

        let border = Clamp;
        let out_region = BorderPolicy::<Image<MonoF32>>::output_region(
            &border,
            src.size(),
            kernel.weights().size(),
            kernel.anchor(),
        );
        let mut result_into = Image::<MonoF32>::zero(out_region.size.width, out_region.size.height);
        correlate_into(&src, &kernel, &border, &mut result_into);

        for y in 0..result_alloc.height() {
            for x in 0..result_alloc.width() {
                assert!(
                    (result_alloc.pixel_at(x, y).0 - result_into.pixel_at(x, y).0).abs() < 1e-6,
                );
            }
        }
    }

    // ── Sobel on known gradient ─────────────────────────────────────────

    #[test]
    fn sobel_y_on_horizontal_gradient() {
        // Image where each column has constant value = x
        let src = Image::generate(5, 5, |x, _y| MonoF32(x as f32));
        let kernel = Neighborhood::<f32, 3, 3>::sobel_y();

        let result: Image<MonoF32> = convolve(&src, &kernel, &Skip);

        // Skip → 3×3 output (interior only)
        assert_eq!(result.width(), 3);
        assert_eq!(result.height(), 3);

        // All interior pixels should have the same magnitude
        let expected = result.pixel_at(0, 0).0;
        for y in 0..3 {
            for x in 0..3 {
                assert!(
                    (result.pixel_at(x, y).0 - expected).abs() < 1e-4,
                    "at ({x}, {y}): got {}, expected {expected}",
                    result.pixel_at(x, y).0,
                );
            }
        }
    }

    // ── Gaussian 3×3 un-normalised ──────────────────────────────────────

    #[test]
    fn gaussian_3x3_unnormalised_on_uniform() {
        // gaussian_3x3 weights sum to 16, so convolving a uniform image
        // of value v produces v * 16.
        let src = Image::fill(5, 5, MonoF32(1.0));
        let kernel = Neighborhood::<f32, 3, 3>::gaussian_3x3();

        let result: Image<MonoF32> = convolve(&src, &kernel, &Clamp);

        for y in 0..result.height() {
            for x in 0..result.width() {
                assert!(
                    (result.pixel_at(x, y).0 - 16.0).abs() < 1e-4,
                    "at ({x}, {y}): {}",
                    result.pixel_at(x, y).0,
                );
            }
        }
    }

    // ── Single-pixel image ──────────────────────────────────────────────

    #[test]
    fn convolve_single_pixel_clamp() {
        let src = Image::fill(1, 1, MonoF32(42.0));
        let kernel = Neighborhood::<f32, 3, 3>::identity_3x3();

        let result: Image<MonoF32> = convolve(&src, &kernel, &Clamp);

        assert_eq!(result.width(), 1);
        assert_eq!(result.height(), 1);
        assert!((result.pixel_at(0, 0).0 - 42.0).abs() < 1e-6);
    }

    // ── Laplacian on uniform → zero ─────────────────────────────────────

    #[test]
    fn laplacian_on_uniform_is_zero() {
        let src = Image::fill(6, 6, MonoF32(10.0));
        let kernel = Neighborhood::<f32, 3, 3>::laplacian();

        let result: Image<MonoF32> = convolve(&src, &kernel, &Clamp);

        for y in 0..result.height() {
            for x in 0..result.width() {
                assert!(
                    result.pixel_at(x, y).0.abs() < 1e-4,
                    "at ({x}, {y}): {}",
                    result.pixel_at(x, y).0,
                );
            }
        }
    }

    #[test]
    fn laplacian_8_on_uniform_is_zero() {
        let src = Image::fill(6, 6, MonoF32(7.0));
        let kernel = Neighborhood::<f32, 3, 3>::laplacian_8();

        let result: Image<MonoF32> = convolve(&src, &kernel, &Clamp);

        for y in 0..result.height() {
            for x in 0..result.width() {
                assert!(
                    result.pixel_at(x, y).0.abs() < 1e-4,
                    "at ({x}, {y}): {}",
                    result.pixel_at(x, y).0,
                );
            }
        }
    }

    // ── Prewitt on uniform → zero ───────────────────────────────────────

    #[test]
    fn prewitt_on_uniform_is_zero() {
        let src = Image::fill(5, 5, MonoF32(3.0));
        let kx = Neighborhood::<f32, 3, 3>::prewitt_x();
        let ky = Neighborhood::<f32, 3, 3>::prewitt_y();

        let rx: Image<MonoF32> = convolve(&src, &kx, &Clamp);
        let ry: Image<MonoF32> = convolve(&src, &ky, &Clamp);

        for y in 0..rx.height() {
            for x in 0..rx.width() {
                assert!(rx.pixel_at(x, y).0.abs() < 1e-4);
                assert!(ry.pixel_at(x, y).0.abs() < 1e-4);
            }
        }
    }

    // ── Scharr on uniform → zero ────────────────────────────────────────

    #[test]
    fn scharr_on_uniform_is_zero() {
        let src = Image::fill(5, 5, MonoF32(3.0));
        let kx = Neighborhood::<f32, 3, 3>::scharr_x();
        let ky = Neighborhood::<f32, 3, 3>::scharr_y();

        let rx: Image<MonoF32> = convolve(&src, &kx, &Clamp);
        let ry: Image<MonoF32> = convolve(&src, &ky, &Clamp);

        for y in 0..rx.height() {
            for x in 0..rx.width() {
                assert!(rx.pixel_at(x, y).0.abs() < 1e-4);
                assert!(ry.pixel_at(x, y).0.abs() < 1e-4);
            }
        }
    }
}

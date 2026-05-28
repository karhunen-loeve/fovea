//! Optimized two-pass separable convolution.
//!
//! A 2D kernel is **separable** when it can be expressed as the outer
//! product of two 1D kernels: `K = col_vector × row_vector`. For example,
//! a 5×5 Gaussian can be decomposed into a horizontal 1×5 pass followed
//! by a vertical 5×1 pass, reducing the work from O(K²) to O(2K) per
//! pixel.
//!
//! This module provides two API levels:
//!
//! ## Ergonomic API (recommended)
//!
//! Accepts a [`SeparableKernel`] that bundles both 1D weight arrays and
//! their anchors into a single value:
//!
//! - [`convolve_separable`] — allocates the output
//! - [`convolve_separable_into`] — writes into an existing output
//!
//! ## Low-level API
//!
//! Accepts raw `ImageView` weights and scalar anchors separately:
//!
//! - [`convolve_separable_raw`] — allocates the output
//! - [`convolve_separable_raw_into`] — writes into an existing output
//!
//! Both perform true **convolution** (kernel is flipped). For symmetric
//! 1D kernels the flip is a no-op.
//!
//! The intermediate image between the two passes uses the pixel's
//! [`LinearPixel::Accumulator`] type, avoiding premature quantisation.

use crate::border::BorderPolicy;
use crate::image::{Image, ImageView, RasterImage, RasterImageMut, SeparableKernel};
use crate::pixel::{FromLinear, LinearPixel, ZeroablePixel};
use crate::transform::fold::{FoldItem, FoldOp, fold_neighborhood, fold_neighborhood_into};

// ─────────────────────────────────────────────────────────────────────────────
// Internal: flip a 1D kernel stored as a raw ImageView
// ─────────────────────────────────────────────────────────────────────────────

/// Create a 180°-rotated copy of a 1D kernel stored as an `ImageView<Pixel = f32>`.
///
/// Used by the raw (low-level) separable API which accepts `ImageView`
/// weights rather than a `SeparableKernel`.
fn flip_1d(
    weights: &impl ImageView<Pixel = f32>,
    anchor: (usize, usize),
) -> (Image<f32>, (usize, usize)) {
    let w = weights.width();
    let h = weights.height();
    let flipped = Image::generate(w, h, |x, y| weights.pixel_at(w - 1 - x, h - 1 - y));
    let flipped_anchor = (w - 1 - anchor.0, h - 1 - anchor.1);
    (flipped, flipped_anchor)
}

// ─────────────────────────────────────────────────────────────────────────────
// FoldOp implementations for separable passes
// ─────────────────────────────────────────────────────────────────────────────

/// Horizontal-pass [`FoldOp`]: accumulate weighted pixels into `Acc` precision.
///
/// Fully monomorphized — no `dyn Iterator` dispatch.
pub(crate) struct HFold<P, Acc> {
    _marker: core::marker::PhantomData<(P, Acc)>,
}

impl<P, Acc> HFold<P, Acc> {
    #[inline(always)]
    pub(crate) fn new() -> Self {
        Self {
            _marker: core::marker::PhantomData,
        }
    }
}

impl<P, Acc> FoldOp<P, f32> for HFold<P, Acc>
where
    P: Copy + LinearPixel<f32, Accumulator = Acc>,
    Acc: Copy + Default + std::ops::Add<Output = Acc>,
{
    type Accumulator = Acc;
    type Output = Acc;

    #[inline(always)]
    fn init(&self) -> Acc {
        Acc::default()
    }

    #[inline(always)]
    fn accumulate(&self, acc: &mut Acc, item: FoldItem<P, f32>) {
        *acc = item.pixel.scale_add(item.weight, *acc);
    }

    #[inline(always)]
    fn finalize(&mut self, acc: Acc) -> Acc {
        acc
    }
}

/// Vertical-pass [`FoldOp`]: accumulate weighted accumulators and convert to `Out`.
///
/// Fully monomorphized — no `dyn Iterator` dispatch.
pub(crate) struct VFold<Acc, Out> {
    _marker: core::marker::PhantomData<(Acc, Out)>,
}

impl<Acc, Out> VFold<Acc, Out> {
    #[inline(always)]
    pub(crate) fn new() -> Self {
        Self {
            _marker: core::marker::PhantomData,
        }
    }
}

impl<Acc, Out> FoldOp<Acc, f32> for VFold<Acc, Out>
where
    Acc: Copy + Default + LinearPixel<f32, Accumulator = Acc> + std::ops::Add<Output = Acc>,
    Out: FromLinear<Acc>,
{
    type Accumulator = Acc;
    type Output = Out;

    #[inline(always)]
    fn init(&self) -> Acc {
        Acc::default()
    }

    #[inline(always)]
    fn accumulate(&self, acc: &mut Acc, item: FoldItem<Acc, f32>) {
        *acc = item.pixel.scale_add(item.weight, *acc);
    }

    #[inline(always)]
    fn finalize(&mut self, acc: Acc) -> Out {
        Out::from_linear(acc)
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Ergonomic API: SeparableKernel
// ═════════════════════════════════════════════════════════════════════════════

/// Write the result of a separable convolution into `output`.
///
/// The convolution is performed in two passes:
///
/// 1. **Horizontal pass** — convolve every row of `image` with the
///    kernel's horizontal weights, producing an intermediate image in
///    accumulator precision.
/// 2. **Vertical pass** — convolve every column of the intermediate
///    image with the kernel's vertical weights, converting back to the
///    output pixel type via [`FromLinear`].
///
/// Both passes flip the kernel (true convolution). For symmetric kernels
/// the flip is a no-op. Flipping uses [`SeparableKernel::flipped`],
/// which is entirely stack-based — zero heap allocation.
///
/// # Panics
///
/// Panics if `output` is too small for the region produced by the border
/// policy after both passes.
///
/// # Example
///
/// ```
/// use irys_cv::image::{Image, ImageView, ImageViewMut, SeparableKernel};
/// use irys_cv::border::Clamp;
/// use irys_cv::transform::convolve_separable_into;
///
/// use irys_cv::pixel::MonoF32;
///
/// let src = Image::fill(6, 6, MonoF32(1.0));
/// let kernel = SeparableKernel::box_blur_3();
/// let mut out = Image::<MonoF32>::zero(6, 6);
///
/// convolve_separable_into(&src, &kernel, &Clamp, &mut out);
///
/// for y in 0..out.height() {
///     for x in 0..out.width() {
///         assert!((out.pixel_at(x, y).0 - 1.0).abs() < 1e-5);
///     }
/// }
/// ```
pub fn convolve_separable_into<I, B, O, P, Acc, Out, const HK: usize, const VK: usize>(
    image: &I,
    kernel: &SeparableKernel<HK, VK>,
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
    // Delegate to the raw API using heap-allocated 1D Image views.
    // The allocation is tiny (HK or VK floats) and avoids exposing
    // private _Array2D trait bounds in the public signature.
    let h_img = kernel.to_h_image();
    let v_img = kernel.to_v_image();
    convolve_separable_raw_into(
        image,
        &h_img,
        kernel.h_anchor(),
        &v_img,
        kernel.v_anchor(),
        border,
        output,
    );
}

/// Perform a separable convolution and return a newly allocated output
/// [`Image`].
///
/// This is a convenience wrapper around [`convolve_separable_into`].
///
/// # Example
///
/// ```
/// use irys_cv::image::{Image, ImageView, SeparableKernel};
/// use irys_cv::border::Clamp;
/// use irys_cv::pixel::Mono8;
/// use irys_cv::transform::convolve_separable;
///
/// let src = Image::fill(8, 8, Mono8::new(5));
/// let kernel = SeparableKernel::box_blur_3();
///
/// let result: Image<Mono8> = convolve_separable(&src, &kernel, &Clamp);
///
/// assert_eq!(result.width(), 8);
/// assert_eq!(result.height(), 8);
/// for y in 0..result.height() {
///     for x in 0..result.width() {
///         assert_eq!(result.pixel_at(x, y), Mono8::new(5));
///     }
/// }
/// ```
#[must_use]
pub fn convolve_separable<I, B, P, Acc, Out, const HK: usize, const VK: usize>(
    image: &I,
    kernel: &SeparableKernel<HK, VK>,
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
    // Delegate to the raw API using heap-allocated 1D Image views.
    let h_img = kernel.to_h_image();
    let v_img = kernel.to_v_image();
    convolve_separable_raw(
        image,
        &h_img,
        kernel.h_anchor(),
        &v_img,
        kernel.v_anchor(),
        border,
    )
}

// ═════════════════════════════════════════════════════════════════════════════
// Low-level API: raw weights + anchor
// ═════════════════════════════════════════════════════════════════════════════

/// Write the result of a separable convolution into `output` (low-level
/// API accepting raw weights and anchors).
///
/// Prefer the [`convolve_separable_into`] overload that takes a
/// [`SeparableKernel`] for ergonomic use. This function is available for
/// dynamic or runtime-sized kernels.
///
/// # Arguments
///
/// - `image` — source image
/// - `h_weights` — horizontal 1D kernel, shape `(width, 1)`.
///   The anchor is at `(h_anchor, 0)`.
/// - `h_anchor` — x-position of the anchor within `h_weights`
/// - `v_weights` — vertical 1D kernel, shape `(1, height)`.
///   The anchor is at `(0, v_anchor)`.
/// - `v_anchor` — y-position of the anchor within `v_weights`
/// - `border` — border policy applied in **both** passes
/// - `output` — destination; must be large enough for the output region
///
/// # Panics
///
/// Panics if `output` is too small for the region produced by the border
/// policy after both passes.
///
pub(crate) fn convolve_separable_raw_into<I, HW, VW, B, O, P, Acc, Out>(
    image: &I,
    h_weights: &HW,
    h_anchor: usize,
    v_weights: &VW,
    v_anchor: usize,
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
    HW: ImageView<Pixel = f32>,
    VW: ImageView<Pixel = f32>,
    B: BorderPolicy<I> + BorderPolicy<Image<Acc>>,
    O: RasterImageMut<Pixel = Out>,
    Out: FromLinear<Acc>,
{
    // ── Pass 1: horizontal ───────────────────────────────────────────
    let (h_flipped, h_flipped_anchor) = flip_1d(h_weights, (h_anchor, 0));

    let intermediate: Image<Acc> = fold_neighborhood(
        image,
        &h_flipped,
        h_flipped_anchor,
        border,
        HFold::<P, Acc>::new(),
    );

    // ── Pass 2: vertical ─────────────────────────────────────────────
    let (v_flipped, v_flipped_anchor) = flip_1d(v_weights, (0, v_anchor));

    fold_neighborhood_into(
        &intermediate,
        &v_flipped,
        v_flipped_anchor,
        border,
        output,
        VFold::<Acc, Out>::new(),
    );
}

/// Perform a separable convolution and return a newly allocated output
/// [`Image`] (low-level API accepting raw weights and anchors).
///
/// Prefer the [`convolve_separable`] overload that takes a
/// [`SeparableKernel`] for ergonomic use.
///
pub(crate) fn convolve_separable_raw<I, HW, VW, B, P, Acc, Out>(
    image: &I,
    h_weights: &HW,
    h_anchor: usize,
    v_weights: &VW,
    v_anchor: usize,
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
    HW: ImageView<Pixel = f32>,
    VW: ImageView<Pixel = f32>,
    B: BorderPolicy<I> + BorderPolicy<Image<Acc>>,
    Out: ZeroablePixel + FromLinear<Acc>,
{
    let (_, h_flipped_anchor) = flip_1d(h_weights, (h_anchor, 0));
    let intermediate_region = <B as BorderPolicy<I>>::output_region(
        border,
        image.size(),
        h_weights.size(),
        h_flipped_anchor,
    );

    let (_, v_flipped_anchor) = flip_1d(v_weights, (0, v_anchor));
    let output_region = <B as BorderPolicy<Image<Acc>>>::output_region(
        border,
        intermediate_region.size,
        v_weights.size(),
        v_flipped_anchor,
    );

    let mut out = Image::<Out>::zero(output_region.size.width, output_region.size.height);

    convolve_separable_raw_into(
        image, h_weights, h_anchor, v_weights, v_anchor, border, &mut out,
    );

    out
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
    use crate::transform::convolve;

    // ── helpers ──────────────────────────────────────────────────────────

    fn make_4x4_monof32() -> Image<MonoF32> {
        Image::generate(4, 4, |x, y| MonoF32((x + y * 4) as f32))
    }

    fn make_6x6_monof32() -> Image<MonoF32> {
        Image::generate(6, 6, |x, y| MonoF32((x + y * 6) as f32))
    }

    // ═════════════════════════════════════════════════════════════════════
    // Tests for SeparableKernel-based API
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn sep_kernel_identity_preserves_image() {
        let src = make_4x4_monof32();
        let kernel = SeparableKernel::new([1.0], [1.0]);

        let result: Image<MonoF32> = convolve_separable(&src, &kernel, &Clamp);

        assert_eq!(result.width(), 4);
        assert_eq!(result.height(), 4);
        for y in 0..4 {
            for x in 0..4 {
                assert!(
                    (result.pixel_at(x, y).0 - src.pixel_at(x, y).0).abs() < 1e-6,
                    "mismatch at ({x}, {y})",
                );
            }
        }
    }

    #[test]
    fn sep_kernel_box_blur_3_matches_full() {
        let src = make_6x6_monof32();
        let full_kernel = Neighborhood::<f32, 3, 3>::box_blur_3x3();
        let full_result: Image<MonoF32> = convolve(&src, &full_kernel, &Clamp);

        let sep = SeparableKernel::box_blur_3();
        let sep_result: Image<MonoF32> = convolve_separable(&src, &sep, &Clamp);

        assert_eq!(full_result.width(), sep_result.width());
        assert_eq!(full_result.height(), sep_result.height());
        for y in 0..full_result.height() {
            for x in 0..full_result.width() {
                assert!(
                    (full_result.pixel_at(x, y).0 - sep_result.pixel_at(x, y).0).abs() < 1e-4,
                    "mismatch at ({x}, {y}): full={}, sep={}",
                    full_result.pixel_at(x, y).0,
                    sep_result.pixel_at(x, y).0,
                );
            }
        }
    }

    #[test]
    fn sep_kernel_box_blur_5_matches_full() {
        let src = Image::generate(8, 8, |x, y| MonoF32((x * 3 + y * 7) as f32));
        let full_kernel = Neighborhood::<f32, 5, 5>::box_blur_5x5();
        let full_result: Image<MonoF32> = convolve(&src, &full_kernel, &Clamp);

        let sep = SeparableKernel::box_blur_5();
        let sep_result: Image<MonoF32> = convolve_separable(&src, &sep, &Clamp);

        assert_eq!(full_result.width(), sep_result.width());
        assert_eq!(full_result.height(), sep_result.height());
        for y in 0..full_result.height() {
            for x in 0..full_result.width() {
                assert!(
                    (full_result.pixel_at(x, y).0 - sep_result.pixel_at(x, y).0).abs() < 1e-3,
                    "mismatch at ({x}, {y})",
                );
            }
        }
    }

    #[test]
    fn sep_kernel_gaussian_3_matches_full() {
        let src = make_6x6_monof32();
        let full_kernel = Neighborhood::<f32, 3, 3>::gaussian_3x3();
        let full_result: Image<MonoF32> = convolve(&src, &full_kernel, &Clamp);

        let sep = SeparableKernel::gaussian_3();
        let sep_result: Image<MonoF32> = convolve_separable(&src, &sep, &Clamp);

        assert_eq!(full_result.width(), sep_result.width());
        assert_eq!(full_result.height(), sep_result.height());
        for y in 0..full_result.height() {
            for x in 0..full_result.width() {
                assert!(
                    (full_result.pixel_at(x, y).0 - sep_result.pixel_at(x, y).0).abs() < 1e-3,
                    "mismatch at ({x}, {y})",
                );
            }
        }
    }

    #[test]
    fn sep_kernel_gaussian_5_matches_full() {
        let src = Image::generate(10, 10, |x, y| MonoF32((x + y) as f32));
        let full_kernel = Neighborhood::<f32, 5, 5>::gaussian_5x5();
        let full_result: Image<MonoF32> = convolve(&src, &full_kernel, &Clamp);

        let sep = SeparableKernel::gaussian_5();
        let sep_result: Image<MonoF32> = convolve_separable(&src, &sep, &Clamp);

        assert_eq!(full_result.width(), sep_result.width());
        assert_eq!(full_result.height(), sep_result.height());
        for y in 0..full_result.height() {
            for x in 0..full_result.width() {
                assert!(
                    (full_result.pixel_at(x, y).0 - sep_result.pixel_at(x, y).0).abs() < 1e-2,
                    "mismatch at ({x}, {y})",
                );
            }
        }
    }

    #[test]
    fn sep_kernel_uniform_stays_uniform() {
        let src = Image::fill(8, 8, MonoF32(42.0));
        let sep = SeparableKernel::box_blur_3();

        let result: Image<MonoF32> = convolve_separable(&src, &sep, &Clamp);

        for y in 0..result.height() {
            for x in 0..result.width() {
                assert!(
                    (result.pixel_at(x, y).0 - 42.0).abs() < 1e-4,
                    "at ({x}, {y}): {}",
                    result.pixel_at(x, y).0,
                );
            }
        }
    }

    #[test]
    fn sep_kernel_u8_round_trip() {
        let src = Image::fill(6, 6, Mono8::new(100));
        let sep = SeparableKernel::box_blur_3();

        let result: Image<Mono8> = convolve_separable(&src, &sep, &Clamp);

        for y in 0..result.height() {
            for x in 0..result.width() {
                assert_eq!(result.pixel_at(x, y), Mono8::new(100));
            }
        }
    }

    #[test]
    fn sep_kernel_into_matches_allocating() {
        let src = make_6x6_monof32();
        let sep = SeparableKernel::gaussian_3();

        let alloc_result: Image<MonoF32> = convolve_separable(&src, &sep, &Clamp);

        let mut into_result = Image::<MonoF32>::zero(alloc_result.width(), alloc_result.height());
        convolve_separable_into(&src, &sep, &Clamp, &mut into_result);

        for y in 0..alloc_result.height() {
            for x in 0..alloc_result.width() {
                assert!(
                    (alloc_result.pixel_at(x, y).0 - into_result.pixel_at(x, y).0).abs() < 1e-6,
                    "mismatch at ({x}, {y})",
                );
            }
        }
    }

    #[test]
    fn sep_kernel_skip_shrinks_output() {
        let src = Image::generate(8, 8, |x, y| MonoF32((x + y) as f32));
        let sep = SeparableKernel::box_blur_3();

        let result: Image<MonoF32> = convolve_separable(&src, &sep, &Skip);

        assert!(result.width() <= 8);
        assert!(result.height() <= 8);
    }

    #[test]
    fn sep_kernel_constant_border_single_pixel() {
        let src = Image::fill(1, 1, MonoF32(9.0));
        let border = Constant(MonoF32(0.0));
        let sep = SeparableKernel::box_blur_3();

        let result: Image<MonoF32> = convolve_separable(&src, &sep, &border);

        assert_eq!(result.width(), 1);
        assert_eq!(result.height(), 1);
        // Horizontal pass: [0, 9, 0] with [1/3, 1/3, 1/3] → 3.0
        // Vertical pass on 1×1 (value 3.0) with constant(0): [0, 3, 0] → 1.0
        assert!(
            (result.pixel_at(0, 0).0 - 1.0).abs() < 1e-4,
            "got {}",
            result.pixel_at(0, 0).0,
        );
    }

    #[test]
    fn sep_kernel_clamp_single_pixel() {
        let src = Image::fill(1, 1, MonoF32(7.0));
        let sep = SeparableKernel::box_blur_3();

        let result: Image<MonoF32> = convolve_separable(&src, &sep, &Clamp);

        assert!((result.pixel_at(0, 0).0 - 7.0).abs() < 1e-4);
    }

    #[test]
    fn sep_kernel_large_image_no_panic() {
        let src = Image::fill(100, 100, MonoF32(1.0));
        let sep = SeparableKernel::gaussian_5();

        let result: Image<MonoF32> = convolve_separable(&src, &sep, &Clamp);

        assert_eq!(result.width(), 100);
        assert_eq!(result.height(), 100);
    }

    #[test]
    fn sep_kernel_matches_raw_api() {
        let src = make_6x6_monof32();

        // SeparableKernel API
        let sep = SeparableKernel::gaussian_3();
        let sep_result: Image<MonoF32> = convolve_separable(&src, &sep, &Clamp);

        // Raw API (using the same weights)
        let h = Neighborhood::<f32, 3, 1>::gaussian_1d_3_h();
        let v = Neighborhood::<f32, 1, 3>::gaussian_1d_3_v();
        let raw_result: Image<MonoF32> = convolve_separable_raw(
            &src,
            h.weights(),
            h.anchor().0,
            v.weights(),
            v.anchor().1,
            &Clamp,
        );

        assert_eq!(sep_result.width(), raw_result.width());
        assert_eq!(sep_result.height(), raw_result.height());
        for y in 0..sep_result.height() {
            for x in 0..sep_result.width() {
                assert!(
                    (sep_result.pixel_at(x, y).0 - raw_result.pixel_at(x, y).0).abs() < 1e-4,
                    "mismatch at ({x}, {y}): sep={}, raw={}",
                    sep_result.pixel_at(x, y).0,
                    raw_result.pixel_at(x, y).0,
                );
            }
        }
    }

    #[test]
    fn sep_kernel_asymmetric_weights() {
        let src = Image::generate(5, 5, |x, y| MonoF32((x * 10 + y) as f32));

        // h = [1, 0, 0], anchor 1 ; v = [0, 0, 1], anchor 1
        let sep = SeparableKernel::with_anchors([1.0, 0.0, 0.0], 1, [0.0, 0.0, 1.0], 1);
        let result: Image<MonoF32> = convolve_separable(&src, &sep, &Clamp);

        // Same with raw API
        let h = Neighborhood::<f32, 3, 1>::with_anchor([1.0, 0.0, 0.0], (1, 0));
        let v = Neighborhood::<f32, 1, 3>::with_anchor([0.0, 0.0, 1.0], (0, 1));
        let raw: Image<MonoF32> = convolve_separable_raw(
            &src,
            h.weights(),
            h.anchor().0,
            v.weights(),
            v.anchor().1,
            &Clamp,
        );

        for y in 0..result.height() {
            for x in 0..result.width() {
                assert!(
                    (result.pixel_at(x, y).0 - raw.pixel_at(x, y).0).abs() < 1e-4,
                    "mismatch at ({x}, {y})",
                );
            }
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // Tests for raw (low-level) API — preserved from original
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn separable_identity_preserves_image() {
        let src = make_4x4_monof32();

        let h = Neighborhood::<f32, 1, 1>::new([1.0]);
        let v = Neighborhood::<f32, 1, 1>::new([1.0]);

        let result: Image<MonoF32> = convolve_separable_raw(
            &src,
            h.weights(),
            h.anchor().0,
            v.weights(),
            v.anchor().1,
            &Clamp,
        );

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
    fn separable_box_blur_3x3_matches_full() {
        let src = make_6x6_monof32();

        let full_kernel = Neighborhood::<f32, 3, 3>::box_blur_3x3();
        let full_result: Image<MonoF32> = convolve(&src, &full_kernel, &Clamp);

        let h = Neighborhood::<f32, 3, 1>::box_1d_3_h();
        let v = Neighborhood::<f32, 1, 3>::box_1d_3_v();
        let sep_result: Image<MonoF32> = convolve_separable_raw(
            &src,
            h.weights(),
            h.anchor().0,
            v.weights(),
            v.anchor().1,
            &Clamp,
        );

        assert_eq!(full_result.width(), sep_result.width());
        assert_eq!(full_result.height(), sep_result.height());

        for y in 0..full_result.height() {
            for x in 0..full_result.width() {
                assert!(
                    (full_result.pixel_at(x, y).0 - sep_result.pixel_at(x, y).0).abs() < 1e-4,
                    "mismatch at ({x}, {y}): full={}, sep={}",
                    full_result.pixel_at(x, y).0,
                    sep_result.pixel_at(x, y).0,
                );
            }
        }
    }

    #[test]
    fn separable_box_blur_5x5_matches_full() {
        let src = Image::generate(8, 8, |x, y| MonoF32((x * 3 + y * 7) as f32));

        let full_kernel = Neighborhood::<f32, 5, 5>::box_blur_5x5();
        let full_result: Image<MonoF32> = convolve(&src, &full_kernel, &Clamp);

        let h = Neighborhood::<f32, 5, 1>::box_1d_5_h();
        let v = Neighborhood::<f32, 1, 5>::box_1d_5_v();
        let sep_result: Image<MonoF32> = convolve_separable_raw(
            &src,
            h.weights(),
            h.anchor().0,
            v.weights(),
            v.anchor().1,
            &Clamp,
        );

        assert_eq!(full_result.width(), sep_result.width());
        assert_eq!(full_result.height(), sep_result.height());

        for y in 0..full_result.height() {
            for x in 0..full_result.width() {
                assert!(
                    (full_result.pixel_at(x, y).0 - sep_result.pixel_at(x, y).0).abs() < 1e-3,
                    "mismatch at ({x}, {y}): full={}, sep={}",
                    full_result.pixel_at(x, y).0,
                    sep_result.pixel_at(x, y).0,
                );
            }
        }
    }

    #[test]
    fn separable_gaussian_3x3_matches_full() {
        let src = make_6x6_monof32();

        let full_kernel = Neighborhood::<f32, 3, 3>::gaussian_3x3();
        let full_result: Image<MonoF32> = convolve(&src, &full_kernel, &Clamp);

        let h = Neighborhood::<f32, 3, 1>::gaussian_1d_3_h();
        let v = Neighborhood::<f32, 1, 3>::gaussian_1d_3_v();
        let sep_result: Image<MonoF32> = convolve_separable_raw(
            &src,
            h.weights(),
            h.anchor().0,
            v.weights(),
            v.anchor().1,
            &Clamp,
        );

        assert_eq!(full_result.width(), sep_result.width());
        assert_eq!(full_result.height(), sep_result.height());

        for y in 0..full_result.height() {
            for x in 0..full_result.width() {
                assert!(
                    (full_result.pixel_at(x, y).0 - sep_result.pixel_at(x, y).0).abs() < 1e-3,
                    "mismatch at ({x}, {y}): full={}, sep={}",
                    full_result.pixel_at(x, y).0,
                    sep_result.pixel_at(x, y).0,
                );
            }
        }
    }

    #[test]
    fn separable_gaussian_5x5_matches_full() {
        let src = Image::generate(10, 10, |x, y| MonoF32((x + y) as f32));

        let full_kernel = Neighborhood::<f32, 5, 5>::gaussian_5x5();
        let full_result: Image<MonoF32> = convolve(&src, &full_kernel, &Clamp);

        let h = Neighborhood::<f32, 5, 1>::gaussian_1d_5_h();
        let v = Neighborhood::<f32, 1, 5>::gaussian_1d_5_v();
        let sep_result: Image<MonoF32> = convolve_separable_raw(
            &src,
            h.weights(),
            h.anchor().0,
            v.weights(),
            v.anchor().1,
            &Clamp,
        );

        assert_eq!(full_result.width(), sep_result.width());
        assert_eq!(full_result.height(), sep_result.height());

        for y in 0..full_result.height() {
            for x in 0..full_result.width() {
                assert!(
                    (full_result.pixel_at(x, y).0 - sep_result.pixel_at(x, y).0).abs() < 1e-2,
                    "mismatch at ({x}, {y}): full={}, sep={}",
                    full_result.pixel_at(x, y).0,
                    sep_result.pixel_at(x, y).0,
                );
            }
        }
    }

    #[test]
    fn separable_box_blur_uniform_image() {
        let src = Image::fill(8, 8, MonoF32(42.0));

        let h = Neighborhood::<f32, 3, 1>::box_1d_3_h();
        let v = Neighborhood::<f32, 1, 3>::box_1d_3_v();

        let result: Image<MonoF32> = convolve_separable_raw(
            &src,
            h.weights(),
            h.anchor().0,
            v.weights(),
            v.anchor().1,
            &Clamp,
        );

        for y in 0..result.height() {
            for x in 0..result.width() {
                assert!(
                    (result.pixel_at(x, y).0 - 42.0).abs() < 1e-4,
                    "at ({x}, {y}): {}",
                    result.pixel_at(x, y).0,
                );
            }
        }
    }

    #[test]
    fn separable_box_blur_u8_uniform() {
        let src = Image::fill(6, 6, Mono8::new(100));

        let h = Neighborhood::<f32, 3, 1>::box_1d_3_h();
        let v = Neighborhood::<f32, 1, 3>::box_1d_3_v();

        let result: Image<Mono8> = convolve_separable_raw(
            &src,
            h.weights(),
            h.anchor().0,
            v.weights(),
            v.anchor().1,
            &Clamp,
        );

        for y in 0..result.height() {
            for x in 0..result.width() {
                assert_eq!(result.pixel_at(x, y), Mono8::new(100));
            }
        }
    }

    #[test]
    fn separable_into_matches_allocating() {
        let src = make_6x6_monof32();

        let h = Neighborhood::<f32, 3, 1>::gaussian_1d_3_h();
        let v = Neighborhood::<f32, 1, 3>::gaussian_1d_3_v();

        let alloc_result: Image<MonoF32> = convolve_separable_raw(
            &src,
            h.weights(),
            h.anchor().0,
            v.weights(),
            v.anchor().1,
            &Clamp,
        );

        let mut into_result = Image::<MonoF32>::zero(alloc_result.width(), alloc_result.height());
        convolve_separable_raw_into(
            &src,
            h.weights(),
            h.anchor().0,
            v.weights(),
            v.anchor().1,
            &Clamp,
            &mut into_result,
        );

        for y in 0..alloc_result.height() {
            for x in 0..alloc_result.width() {
                assert!(
                    (alloc_result.pixel_at(x, y).0 - into_result.pixel_at(x, y).0).abs() < 1e-6,
                    "mismatch at ({x}, {y})",
                );
            }
        }
    }

    #[test]
    fn separable_skip_shrinks_output() {
        let src = Image::generate(8, 8, |x, y| MonoF32((x + y) as f32));

        let h = Neighborhood::<f32, 3, 1>::box_1d_3_h();
        let v = Neighborhood::<f32, 1, 3>::box_1d_3_v();

        let result: Image<MonoF32> = convolve_separable_raw(
            &src,
            h.weights(),
            h.anchor().0,
            v.weights(),
            v.anchor().1,
            &Skip,
        );

        assert!(result.width() <= 8);
        assert!(result.height() <= 8);
    }

    #[test]
    fn separable_constant_border_single_pixel() {
        let src = Image::fill(1, 1, MonoF32(9.0));
        let border = Constant(MonoF32(0.0));

        let h = Neighborhood::<f32, 3, 1>::box_1d_3_h();
        let v = Neighborhood::<f32, 1, 3>::box_1d_3_v();

        let result: Image<MonoF32> = convolve_separable_raw(
            &src,
            h.weights(),
            h.anchor().0,
            v.weights(),
            v.anchor().1,
            &border,
        );

        assert_eq!(result.width(), 1);
        assert_eq!(result.height(), 1);

        assert!(
            (result.pixel_at(0, 0).0 - 1.0).abs() < 1e-4,
            "got {}",
            result.pixel_at(0, 0).0,
        );
    }

    #[test]
    fn separable_clamp_single_pixel() {
        let src = Image::fill(1, 1, MonoF32(7.0));

        let h = Neighborhood::<f32, 3, 1>::box_1d_3_h();
        let v = Neighborhood::<f32, 1, 3>::box_1d_3_v();

        let result: Image<MonoF32> = convolve_separable_raw(
            &src,
            h.weights(),
            h.anchor().0,
            v.weights(),
            v.anchor().1,
            &Clamp,
        );

        assert!((result.pixel_at(0, 0).0 - 7.0).abs() < 1e-4);
    }

    #[test]
    fn separable_large_image_no_panic() {
        let src = Image::fill(100, 100, MonoF32(1.0));

        let h = Neighborhood::<f32, 5, 1>::gaussian_1d_5_h();
        let v = Neighborhood::<f32, 1, 5>::gaussian_1d_5_v();

        let result: Image<MonoF32> = convolve_separable_raw(
            &src,
            h.weights(),
            h.anchor().0,
            v.weights(),
            v.anchor().1,
            &Clamp,
        );

        assert_eq!(result.width(), 100);
        assert_eq!(result.height(), 100);
    }

    #[test]
    fn separable_order_h_then_v() {
        let src = Image::generate(5, 5, |x, y| MonoF32((x * 10 + y) as f32));

        let h = Neighborhood::<f32, 3, 1>::with_anchor([1.0, 0.0, 0.0], (1, 0));
        let v = Neighborhood::<f32, 1, 3>::with_anchor([0.0, 0.0, 1.0], (0, 1));

        let result_hv: Image<MonoF32> = convolve_separable_raw(
            &src,
            h.weights(),
            h.anchor().0,
            v.weights(),
            v.anchor().1,
            &Clamp,
        );

        let h2 = Neighborhood::<f32, 3, 1>::with_anchor([0.0, 0.0, 1.0], (1, 0));
        let v2 = Neighborhood::<f32, 1, 3>::with_anchor([1.0, 0.0, 0.0], (0, 1));

        let result_vh: Image<MonoF32> = convolve_separable_raw(
            &src,
            h2.weights(),
            h2.anchor().0,
            v2.weights(),
            v2.anchor().1,
            &Clamp,
        );

        // Different asymmetric kernels should produce different results
        let mut differ = false;
        for y in 0..result_hv.height() {
            for x in 0..result_hv.width() {
                if (result_hv.pixel_at(x, y).0 - result_vh.pixel_at(x, y).0).abs() > 1e-4 {
                    differ = true;
                }
            }
        }
        assert!(
            differ,
            "swapping asymmetric kernels should produce different results"
        );
    }
}

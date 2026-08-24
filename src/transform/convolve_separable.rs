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
//! - [`SeparableScratch::convolve_separable_into`] — writes into an existing
//!   output *and* reuses a caller-owned working set, so a convolution in a
//!   hot loop allocates nothing after warm-up
//!
//! These perform true **convolution**: the kernel is flipped via
//! [`SeparableKernel::flipped`], which is entirely stack-based, and the
//! flipped weights are handed to the correlation core through borrowed
//! [`ImageRef`] views — so the kernel never touches the heap.
//! For symmetric 1D kernels the flip is a no-op.
//!
//! ## Correlation core (internal)
//!
//! [`correlate_separable_raw`] / [`correlate_separable_raw_into`] apply raw
//! `ImageView<Pixel = f32>` weights directly at their kernel offsets (no
//! flip). They are the allocation-free engine the ergonomic API and the
//! blur filters delegate to; callers that need true convolution flip first.
//!
//! The intermediate image between the two passes uses the pixel's
//! [`LinearPixel::Accumulator`] type, avoiding premature quantisation. It is
//! the largest per-call allocation on this path, which is what
//! [`SeparableScratch`] exists to reuse.

use crate::border::BorderPolicy;
use crate::image::{
    Image, ImageRef, ImageRefMut, ImageView, RasterImage, RasterImageMut, SeparableWeights,
};
use crate::pixel::{FromLinear, LinearPixel, ZeroablePixel};
use crate::transform::fold::{
    FoldItem, FoldOp, FoldScratch, fold_neighborhood, fold_neighborhood_into,
    fold_neighborhood_into_with_scratch,
};

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
// Ergonomic API: any SeparableWeights value
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
/// `kernel` is any [`SeparableWeights`] value — a
/// [`SeparableKernel`](crate::image::SeparableKernel) with compile-time tap
/// counts, or a σ-derived
/// [`GaussianKernel1D`](crate::image::GaussianKernel1D) from
/// [`gaussian_kernel_1d`](crate::image::gaussian_kernel_1d). The kernel *is*
/// the variant: there is no differently-named function per kernel flavour.
///
/// Both passes flip the kernel (true convolution) via
/// [`SeparableWeights::flipped`], which is entirely stack-based — zero heap
/// allocation. For symmetric kernels the flip is a no-op.
///
/// # Panics
///
/// Panics if `output` is too small for the region produced by the border
/// policy after both passes.
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView, ImageViewMut, SeparableKernel};
/// use fovea::border::Clamp;
/// use fovea::transform::convolve_separable_into;
///
/// use fovea::pixel::MonoF32;
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
///
/// A σ-derived kernel goes through the same call — this is what the removed
/// `gaussian_blur_with_into` used to spell:
///
/// ```
/// use fovea::border::Clamp;
/// use fovea::image::{Image, ImageView, gaussian_kernel_1d};
/// use fovea::pixel::MonoF32;
/// use fovea::sigma;
/// use fovea::transform::convolve_separable_into;
///
/// let src = Image::fill(16, 16, MonoF32(0.5));
/// let kernel = gaussian_kernel_1d(sigma!(1.5), 3.0); // explicit truncate
/// let mut out = Image::<MonoF32>::zero(16, 16);
///
/// convolve_separable_into(&src, &kernel, &Clamp, &mut out);
///
/// assert!((out.pixel_at(8, 8).0 - 0.5).abs() < 1e-4);
/// ```
pub fn convolve_separable_into<I, B, K, O, P, Acc, Out>(
    image: &I,
    kernel: &K,
    border: &B,
    output: &mut O,
) where
    I: RasterImage<Pixel = P>,
    K: SeparableWeights,
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
    // True convolution = correlation with the 180°-flipped kernel.
    // `SeparableWeights::flipped()` is allocation-free (stack values), and the
    // flipped weights are fed to the correlation core through borrowed
    // `ImageRef` views — so the kernel never touches the heap.
    let flipped = kernel.flipped();
    let (h, v) = weight_views(&flipped);
    correlate_separable_raw_into(
        image,
        &h,
        flipped.h_anchor(),
        &v,
        flipped.v_anchor(),
        border,
        output,
    );
}

/// Borrow a [`SeparableWeights`] value's two axes as `ImageRef` views — the
/// shape the correlation core consumes. Zero-copy: the views point into the
/// kernel's own storage.
fn weight_views<K: SeparableWeights>(kernel: &K) -> (ImageRef<'_, f32>, ImageRef<'_, f32>) {
    let h_weights = kernel.h_weights();
    let v_weights = kernel.v_weights();
    let h = ImageRef::new(h_weights.len(), 1, h_weights).expect("h kernel view: 1 row");
    let v = ImageRef::new(1, v_weights.len(), v_weights).expect("v kernel view: 1 column");
    (h, v)
}

/// Perform a separable convolution and return a newly allocated output
/// [`Image`].
///
/// This is a convenience wrapper around [`convolve_separable_into`].
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView, SeparableKernel};
/// use fovea::border::Clamp;
/// use fovea::pixel::Mono8;
/// use fovea::transform::convolve_separable;
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
pub fn convolve_separable<I, B, K, P, Acc, Out>(image: &I, kernel: &K, border: &B) -> Image<Out>
where
    I: RasterImage<Pixel = P>,
    K: SeparableWeights,
    P: Copy + LinearPixel<f32, Accumulator = Acc>,
    Acc: Copy
        + Default
        + ZeroablePixel
        + LinearPixel<f32, Accumulator = Acc>
        + std::ops::Add<Output = Acc>,
    B: BorderPolicy<I> + BorderPolicy<Image<Acc>>,
    Out: ZeroablePixel + FromLinear<Acc>,
{
    // Flip on the stack, borrow the weights as `ImageRef` views, correlate.
    // No heap allocation for the kernel.
    let flipped = kernel.flipped();
    let (h, v) = weight_views(&flipped);
    correlate_separable_raw(
        image,
        &h,
        flipped.h_anchor(),
        &v,
        flipped.v_anchor(),
        border,
    )
}

// ═════════════════════════════════════════════════════════════════════════════
// Reusable working set
// ═════════════════════════════════════════════════════════════════════════════

/// Reusable working memory for separable convolution.
///
/// A two-pass separable convolution needs three image- or kernel-shaped
/// working buffers: the inter-pass intermediate (in accumulator
/// precision), a per-row accumulator, and the kernel-position list. The
/// one-shot entry points ([`convolve_separable_into`],
/// [`gaussian_blur_into`](crate::transform::gaussian_blur_into)) allocate
/// them per call. A `SeparableScratch` owns them instead, so a blur in a
/// hot loop — video frames, pyramid levels, scale-space octaves —
/// allocates **nothing after warm-up**.
///
/// Reuse is explicit: the scratch is a value you construct and pass in.
/// There is no hidden pool.
///
/// # Growth
///
/// Buffers grow to fit and are **never shrunk** (high-water mark). A
/// larger frame grows them; a smaller frame afterwards reuses the larger
/// buffers as-is. Only their capacity survives between calls — contents
/// are always overwritten, so the same scratch can be shared across
/// different images, kernels, sigmas and border policies without
/// affecting results.
///
/// `Acc` is the accumulator pixel type of the convolution — the input
/// pixel's [`LinearPixel::Accumulator`] (`MonoF32` for `Mono8`,
/// `RgbF32` for `Rgb8`). One scratch serves one accumulator type.
///
/// # Example
///
/// ```
/// use fovea::border::Clamp;
/// use fovea::image::{Image, ImageView, SeparableKernel};
/// use fovea::pixel::{Mono8, MonoF32};
/// use fovea::transform::SeparableScratch;
///
/// let kernel = SeparableKernel::gaussian_5();
/// let mut scratch = SeparableScratch::<MonoF32>::new();
/// let mut out = Image::<Mono8>::zero(64, 64);
///
/// // Steady state: the second and later frames allocate nothing.
/// for level in 0..4 {
///     let frame = Image::fill(64, 64, Mono8::new(10 * level + 5));
///     scratch.convolve_separable_into(&frame, &kernel, &Clamp, &mut out);
///     assert_eq!(out.pixel_at(32, 32), Mono8::new(10 * level + 5));
/// }
/// ```
#[derive(Debug, Clone)]
pub struct SeparableScratch<Acc> {
    /// Inter-pass intermediate storage; `len` is the high-water pixel
    /// count, and each call views the leading `region.area()` elements as
    /// a contiguous image.
    intermediate: Vec<Acc>,
    /// The fold engine's accumulator row and kernel-position list, borrowed
    /// by both passes. Separable weights are always `f32`, so one buffer
    /// serves every kernel.
    fold: FoldScratch<Acc, f32>,
}

impl<Acc> SeparableScratch<Acc> {
    /// An empty scratch. The buffers are allocated on first use and sized
    /// to whatever the first call needs.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            intermediate: Vec::new(),
            fold: FoldScratch::new(),
        }
    }
}

impl<Acc> Default for SeparableScratch<Acc> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Acc> SeparableScratch<Acc>
where
    Acc: Copy
        + Default
        + ZeroablePixel
        + LinearPixel<f32, Accumulator = Acc>
        + std::ops::Add<Output = Acc>,
{
    /// Separable convolution into a caller-owned output, reusing this
    /// scratch.
    ///
    /// Identical in result to the free
    /// [`convolve_separable_into`] — same two passes, same stack-based
    /// kernel flip, same border handling — but the inter-pass intermediate
    /// and the engine's working buffers come from `self` instead of the
    /// heap. The first call sizes them; every later call that needs no more
    /// room than the largest so far allocates nothing.
    ///
    /// The border policy must also apply to the borrowed intermediate,
    /// which is why its bound is stated over [`ImageRef`]; every built-in
    /// policy ([`Clamp`](crate::border::Clamp),
    /// [`Mirror`](crate::border::Mirror), [`Wrap`](crate::border::Wrap),
    /// [`Skip`](crate::border::Skip),
    /// [`Constant`](crate::border::Constant)) is implemented for every
    /// image view and satisfies it.
    ///
    /// # Panics
    ///
    /// Panics if `output` is too small for the region produced by the
    /// border policy after both passes.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::border::Clamp;
    /// use fovea::image::{Image, ImageView, SeparableKernel};
    /// use fovea::pixel::MonoF32;
    /// use fovea::transform::SeparableScratch;
    ///
    /// let src = Image::fill(8, 8, MonoF32(1.0));
    /// let kernel = SeparableKernel::box_blur_3();
    /// let mut scratch = SeparableScratch::new();
    /// let mut out = Image::<MonoF32>::zero(8, 8);
    ///
    /// scratch.convolve_separable_into(&src, &kernel, &Clamp, &mut out);
    ///
    /// assert!((out.pixel_at(4, 4).0 - 1.0).abs() < 1e-5);
    /// ```
    pub fn convolve_separable_into<I, B, K, O, P, Out>(
        &mut self,
        image: &I,
        kernel: &K,
        border: &B,
        output: &mut O,
    ) where
        I: RasterImage<Pixel = P>,
        K: SeparableWeights,
        P: Copy + LinearPixel<f32, Accumulator = Acc>,
        B: BorderPolicy<I> + for<'r> BorderPolicy<ImageRef<'r, Acc>>,
        O: RasterImageMut<Pixel = Out>,
        Out: FromLinear<Acc>,
    {
        // True convolution = correlation with the 180°-flipped kernel; the
        // flip is stack-based, exactly as in `convolve_separable_into`.
        let flipped = kernel.flipped();
        let (h, v) = weight_views(&flipped);
        self.correlate_separable_raw_into(
            image,
            (&h, flipped.h_anchor()),
            (&v, flipped.v_anchor()),
            border,
            output,
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Correlation core (no flip) — the allocation-free engine
// ═════════════════════════════════════════════════════════════════════════════

/// Separable **correlation** into a caller-owned output — the no-flip core.
///
/// The weights are applied directly at their kernel offsets (no 180° flip),
/// so this is correlation, not convolution. It performs **no kernel-shaped
/// heap allocation**: both passes consume the `ImageView` weights as-is. The
/// flipping `convolve_separable_*` functions are thin wrappers that arrange
/// the flip — on the stack for [`SeparableKernel`], via [`flip_1d`] for raw
/// borrowed weights — and delegate here.
///
/// (The inter-pass intermediate image and the per-row accumulator inside
/// [`fold_neighborhood`] are still allocated; eliminating *those* is a
/// separate, deferred concern — see the allocation-free separable blur plan.)
pub(crate) fn correlate_separable_raw_into<I, HW, VW, B, O, P, Acc, Out>(
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
    // ── Pass 1: horizontal correlation (weights used directly) ───────────
    let intermediate: Image<Acc> = fold_neighborhood(
        image,
        h_weights,
        (h_anchor, 0),
        border,
        HFold::<P, Acc>::new(),
    );

    // ── Pass 2: vertical correlation ─────────────────────────────────────
    fold_neighborhood_into(
        &intermediate,
        v_weights,
        (0, v_anchor),
        border,
        output,
        VFold::<Acc, Out>::new(),
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
    /// Separable **correlation** into a caller-owned output, reusing this
    /// scratch — the no-flip core behind
    /// [`convolve_separable_into`](Self::convolve_separable_into) and the
    /// scratch-aware blurs.
    ///
    /// Same result as the free [`correlate_separable_raw_into`], but the
    /// inter-pass intermediate is a view over `self.intermediate` (grown to
    /// fit, never shrunk) rather than a fresh `Image<Acc>`, and both passes
    /// borrow the engine's accumulator row and kernel-position list from
    /// `self.fold`. After warm-up on a given shape, this path performs **no
    /// heap allocation at all**.
    ///
    /// Each pass is given as a `(weights, anchor)` pair: `h` is the row of
    /// weights with its x-anchor, `v` the column with its y-anchor.
    pub(crate) fn correlate_separable_raw_into<I, HW, VW, B, O, P, Out>(
        &mut self,
        image: &I,
        h: (&HW, usize),
        v: (&VW, usize),
        border: &B,
        output: &mut O,
    ) where
        I: RasterImage<Pixel = P>,
        P: Copy + LinearPixel<f32, Accumulator = Acc>,
        HW: ImageView<Pixel = f32>,
        VW: ImageView<Pixel = f32>,
        B: BorderPolicy<I> + for<'r> BorderPolicy<ImageRef<'r, Acc>>,
        O: RasterImageMut<Pixel = Out>,
        Out: FromLinear<Acc>,
    {
        let (h_weights, h_anchor) = h;
        let (v_weights, v_anchor) = v;

        // The intermediate is exactly the pass-1 output region: pass 1
        // writes every pixel of it, so no stale content from an earlier call
        // survives.
        let mid = <B as BorderPolicy<I>>::output_region(
            border,
            image.size(),
            h_weights.size(),
            (h_anchor, 0),
        )
        .size;
        let area = mid
            .checked_area()
            .expect("intermediate area overflows usize");

        // Destructured so the intermediate and the engine's buffers can be
        // borrowed independently.
        let Self { intermediate, fold } = self;

        // Grow to the high-water mark; `Vec::resize` upwards keeps the
        // existing elements and never shrinks capacity.
        if intermediate.len() < area {
            intermediate.resize(area, Acc::default());
        }

        // ── Pass 1: horizontal correlation into the borrowed intermediate ─
        {
            let mut mid_view = ImageRefMut::new(mid.width, mid.height, &mut intermediate[..area])
                .expect("intermediate view: len == area");
            fold_neighborhood_into_with_scratch(
                image,
                h_weights,
                (h_anchor, 0),
                border,
                &mut mid_view,
                HFold::<P, Acc>::new(),
                fold,
            );
        }

        // ── Pass 2: vertical correlation ──────────────────────────────────
        let mid_view = ImageRef::new(mid.width, mid.height, &intermediate[..area])
            .expect("intermediate view: len == area");
        fold_neighborhood_into_with_scratch(
            &mid_view,
            v_weights,
            (0, v_anchor),
            border,
            output,
            VFold::<Acc, Out>::new(),
            fold,
        );
    }
}

/// Separable correlation returning a newly allocated output (no-flip core).
///
/// See [`correlate_separable_raw_into`]. The region sizing uses the anchors
/// directly (no flip), with no kernel-shaped allocation.
pub(crate) fn correlate_separable_raw<I, HW, VW, B, P, Acc, Out>(
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
    let intermediate_region = <B as BorderPolicy<I>>::output_region(
        border,
        image.size(),
        h_weights.size(),
        (h_anchor, 0),
    );
    let output_region = <B as BorderPolicy<Image<Acc>>>::output_region(
        border,
        intermediate_region.size,
        v_weights.size(),
        (0, v_anchor),
    );

    let mut out = Image::<Out>::zero(output_region.size.width, output_region.size.height);
    correlate_separable_raw_into(
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
    use crate::image::{ImageView, Neighborhood, SeparableKernel};
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

        // `SeparableKernel::gaussian_3` is normalized (sum 1) while the raw
        // `Neighborhood` kernel sums to 16, so the separable result equals
        // the full result divided by 16.
        let sep = SeparableKernel::gaussian_3();
        let sep_result: Image<MonoF32> = convolve_separable(&src, &sep, &Clamp);

        assert_eq!(full_result.width(), sep_result.width());
        assert_eq!(full_result.height(), sep_result.height());
        for y in 0..full_result.height() {
            for x in 0..full_result.width() {
                assert!(
                    (full_result.pixel_at(x, y).0 / 16.0 - sep_result.pixel_at(x, y).0).abs()
                        < 1e-3,
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

        // `SeparableKernel::gaussian_5` is normalized (sum 1) while the raw
        // `Neighborhood` kernel sums to 256, so the separable result equals
        // the full result divided by 256.
        let sep = SeparableKernel::gaussian_5();
        let sep_result: Image<MonoF32> = convolve_separable(&src, &sep, &Clamp);

        assert_eq!(full_result.width(), sep_result.width());
        assert_eq!(full_result.height(), sep_result.height());
        for y in 0..full_result.height() {
            for x in 0..full_result.width() {
                assert!(
                    (full_result.pixel_at(x, y).0 / 256.0 - sep_result.pixel_at(x, y).0).abs()
                        < 1e-3,
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

        // Raw API using the raw integer 1D weights (`gaussian_1d_3_*` is
        // `[1, 2, 1]`, sum 4 per pass → 16 over both). The normalized
        // `SeparableKernel::gaussian_3` therefore equals the raw result / 16.
        let h = Neighborhood::<f32, 3, 1>::gaussian_1d_3_h();
        let v = Neighborhood::<f32, 1, 3>::gaussian_1d_3_v();
        let raw_result: Image<MonoF32> = correlate_separable_raw(
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
                    (sep_result.pixel_at(x, y).0 - raw_result.pixel_at(x, y).0 / 16.0).abs() < 1e-4,
                    "mismatch at ({x}, {y}): sep={}, raw/16={}",
                    sep_result.pixel_at(x, y).0,
                    raw_result.pixel_at(x, y).0 / 16.0,
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

        // The correlation core does not flip, so to match the (flipping)
        // `convolve_separable` for asymmetric weights we pre-flip the kernel
        // by hand: convolution ≡ correlation with the 180°-rotated kernel.
        // h = [1, 0, 0]/anchor 1 → flipped [0, 0, 1]/anchor 1;
        // v = [0, 0, 1]/anchor 1 → flipped [1, 0, 0]/anchor 1.
        let h = Neighborhood::<f32, 3, 1>::with_anchor([0.0, 0.0, 1.0], (1, 0));
        let v = Neighborhood::<f32, 1, 3>::with_anchor([1.0, 0.0, 0.0], (0, 1));
        let raw: Image<MonoF32> = correlate_separable_raw(
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
    // Tests for the reusable working set
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn scratch_reuse_matches_owned() {
        let src = make_6x6_monof32();
        let sep = SeparableKernel::gaussian_5();

        let expected: Image<MonoF32> = convolve_separable(&src, &sep, &Clamp);

        // Two successive calls on one scratch: the first sizes the buffers,
        // the second reuses them. Both must equal the allocating path.
        let mut scratch = SeparableScratch::new();
        for round in 0..2 {
            let mut actual = Image::<MonoF32>::zero(expected.width(), expected.height());
            scratch.convolve_separable_into(&src, &sep, &Clamp, &mut actual);

            for y in 0..expected.height() {
                for x in 0..expected.width() {
                    assert!(
                        (expected.pixel_at(x, y).0 - actual.pixel_at(x, y).0).abs() < 1e-6,
                        "round {round}: mismatch at ({x}, {y}): owned={}, scratch={}",
                        expected.pixel_at(x, y).0,
                        actual.pixel_at(x, y).0,
                    );
                }
            }
        }
    }

    #[test]
    fn scratch_handles_size_change() {
        // Grow, then shrink: the buffers are never shrunk, so the third
        // call runs on over-sized storage and must still be correct — the
        // intermediate view is sized to the pass-1 region, not the buffer.
        let sizes = [(6usize, 6usize), (13, 11), (4, 5)];
        let sep = SeparableKernel::gaussian_3();
        let mut scratch = SeparableScratch::new();

        for (w, h) in sizes {
            let src = Image::generate(w, h, |x, y| MonoF32((x * 3 + y * 7) as f32));
            let expected: Image<MonoF32> = convolve_separable(&src, &sep, &Clamp);

            let mut actual = Image::<MonoF32>::zero(expected.width(), expected.height());
            scratch.convolve_separable_into(&src, &sep, &Clamp, &mut actual);

            for y in 0..expected.height() {
                for x in 0..expected.width() {
                    assert!(
                        (expected.pixel_at(x, y).0 - actual.pixel_at(x, y).0).abs() < 1e-4,
                        "{w}×{h}: mismatch at ({x}, {y}): owned={}, scratch={}",
                        expected.pixel_at(x, y).0,
                        actual.pixel_at(x, y).0,
                    );
                }
            }
        }
    }

    #[test]
    fn scratch_matches_owned_for_shrinking_border() {
        // `Skip` gives the intermediate an offset origin and a smaller
        // region than the image — the case where a mis-sized intermediate
        // view would silently read the wrong rows.
        let src = Image::generate(9, 7, |x, y| MonoF32((x + y * 9) as f32));
        let sep = SeparableKernel::box_blur_5();

        let expected: Image<MonoF32> = convolve_separable(&src, &sep, &Skip);

        let mut scratch = SeparableScratch::new();
        let mut actual = Image::<MonoF32>::zero(expected.width(), expected.height());
        scratch.convolve_separable_into(&src, &sep, &Skip, &mut actual);

        assert!(expected.width() < src.width() && expected.height() < src.height());
        for y in 0..expected.height() {
            for x in 0..expected.width() {
                assert!(
                    (expected.pixel_at(x, y).0 - actual.pixel_at(x, y).0).abs() < 1e-4,
                    "mismatch at ({x}, {y})",
                );
            }
        }
    }

    #[test]
    fn scratch_shared_across_kernels_and_pixel_types() {
        // One scratch, one accumulator type, different kernels and
        // different input/output pixel types.
        let mut scratch = SeparableScratch::<MonoF32>::new();

        let f32_src = make_6x6_monof32();
        let sep3 = SeparableKernel::gaussian_3();
        let f32_expected: Image<MonoF32> = convolve_separable(&f32_src, &sep3, &Clamp);
        let mut f32_out = Image::<MonoF32>::zero(6, 6);
        scratch.convolve_separable_into(&f32_src, &sep3, &Clamp, &mut f32_out);

        let u8_src = Image::fill(8, 8, Mono8::new(100));
        let sep5 = SeparableKernel::box_blur_5();
        let mut u8_out = Image::<Mono8>::zero(8, 8);
        scratch.convolve_separable_into(&u8_src, &sep5, &Clamp, &mut u8_out);

        for y in 0..6 {
            for x in 0..6 {
                assert!((f32_expected.pixel_at(x, y).0 - f32_out.pixel_at(x, y).0).abs() < 1e-6);
            }
        }
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(u8_out.pixel_at(x, y), Mono8::new(100));
            }
        }
    }

    #[test]
    fn scratch_handles_empty_region() {
        // `Skip` with a kernel wider than the image leaves no valid
        // position at all: the intermediate is zero-area, so the views over
        // the scratch must be constructible from an empty slice.
        let src = Image::generate(3, 3, |x, y| MonoF32((x + y) as f32));
        let sep = SeparableKernel::box_blur_5();

        let expected: Image<MonoF32> = convolve_separable(&src, &sep, &Skip);
        assert_eq!(expected.width(), 0);

        let mut scratch = SeparableScratch::new();
        let mut actual = Image::<MonoF32>::zero(expected.width(), expected.height());
        scratch.convolve_separable_into(&src, &sep, &Skip, &mut actual);

        assert_eq!(actual.width(), expected.width());
        assert_eq!(actual.height(), expected.height());
    }

    #[test]
    fn scratch_default_matches_new() {
        let src = make_4x4_monof32();
        let sep = SeparableKernel::box_blur_3();

        let mut from_new = SeparableScratch::new();
        let mut out_new = Image::<MonoF32>::zero(4, 4);
        from_new.convolve_separable_into(&src, &sep, &Clamp, &mut out_new);

        let mut from_default = SeparableScratch::default();
        let mut out_default = Image::<MonoF32>::zero(4, 4);
        from_default.convolve_separable_into(&src, &sep, &Clamp, &mut out_default);

        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(out_new.pixel_at(x, y).0, out_default.pixel_at(x, y).0);
            }
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // Tests for the correlation core (no-flip engine) with raw weights
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn correlate_separable_matches_convolve_for_symmetric() {
        // For a symmetric kernel the 180° flip is a no-op, so the flipping
        // `convolve_separable` (SeparableKernel) and the no-flip correlation
        // core must agree exactly.
        let src = Image::generate(7, 7, |x, y| MonoF32((x * 5 + y * 2) as f32));

        let sep = SeparableKernel::gaussian_5();
        let convolved: Image<MonoF32> = convolve_separable(&src, &sep, &Clamp);

        let h = Neighborhood::<f32, 5, 1>::gaussian_1d_5_h();
        let v = Neighborhood::<f32, 1, 5>::gaussian_1d_5_v();
        // Divide the raw integer kernel (sum 256 over both passes) to match
        // the normalized SeparableKernel.
        let correlated: Image<MonoF32> = correlate_separable_raw(
            &src,
            h.weights(),
            h.anchor().0,
            v.weights(),
            v.anchor().1,
            &Clamp,
        );

        assert_eq!(convolved.width(), correlated.width());
        assert_eq!(convolved.height(), correlated.height());
        for y in 0..convolved.height() {
            for x in 0..convolved.width() {
                assert!(
                    (convolved.pixel_at(x, y).0 - correlated.pixel_at(x, y).0 / 256.0).abs() < 1e-3,
                    "mismatch at ({x}, {y})",
                );
            }
        }
    }

    #[test]
    fn separable_identity_preserves_image() {
        let src = make_4x4_monof32();

        let h = Neighborhood::<f32, 1, 1>::new([1.0]);
        let v = Neighborhood::<f32, 1, 1>::new([1.0]);

        let result: Image<MonoF32> = correlate_separable_raw(
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
        let sep_result: Image<MonoF32> = correlate_separable_raw(
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
        let sep_result: Image<MonoF32> = correlate_separable_raw(
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
        let sep_result: Image<MonoF32> = correlate_separable_raw(
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
        let sep_result: Image<MonoF32> = correlate_separable_raw(
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

        let result: Image<MonoF32> = correlate_separable_raw(
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

        let result: Image<Mono8> = correlate_separable_raw(
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

        let alloc_result: Image<MonoF32> = correlate_separable_raw(
            &src,
            h.weights(),
            h.anchor().0,
            v.weights(),
            v.anchor().1,
            &Clamp,
        );

        let mut into_result = Image::<MonoF32>::zero(alloc_result.width(), alloc_result.height());
        correlate_separable_raw_into(
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

        let result: Image<MonoF32> = correlate_separable_raw(
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

        let result: Image<MonoF32> = correlate_separable_raw(
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

        let result: Image<MonoF32> = correlate_separable_raw(
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

        let result: Image<MonoF32> = correlate_separable_raw(
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

        let result_hv: Image<MonoF32> = correlate_separable_raw(
            &src,
            h.weights(),
            h.anchor().0,
            v.weights(),
            v.anchor().1,
            &Clamp,
        );

        let h2 = Neighborhood::<f32, 3, 1>::with_anchor([0.0, 0.0, 1.0], (1, 0));
        let v2 = Neighborhood::<f32, 1, 3>::with_anchor([1.0, 0.0, 0.0], (0, 1));

        let result_vh: Image<MonoF32> = correlate_separable_raw(
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

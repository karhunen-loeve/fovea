use crate::border::{BorderPolicy, compute_interior_region};
use crate::image::{Image, ImageView, RasterImage, RasterImageMut};
use crate::pixel::ZeroablePixel;
// ADR-0044 Phase E: `MonoF32` appears in doctests and the `#[cfg(test)]`
// module as the pixel-role float output type. Scoping the import to
// `cfg(test)` keeps the non-test build free of the unused-import
// warning while leaving the doctests' `# use fovea::pixel::MonoF32;`
// preambles authoritative for rendered documentation examples.
#[cfg(test)]
use crate::pixel::MonoF32;

// ─── FoldOp trait ────────────────────────────────────────────────────────────

/// A neighborhood fold operation whose `fold` method is generic over the
/// iterator type.
///
/// Because `fold` is generic over `I`, Rust monomorphizes it separately for
/// the hot path (interior — direct `pixel_at`) and the cold path (boundary —
/// `border.pixel_at`). This eliminates the `dyn Iterator` vtable dispatch
/// that a plain closure would require, enabling full inlining and
/// auto-vectorization.
///
/// # Implementing `FoldOp`
///
/// ```
/// use fovea::transform::{FoldOp, FoldItem};
///
/// struct SumFold;
///
/// impl FoldOp<u8, f32> for SumFold {
///     type Accumulator = f32;
///     type Output = f32;
///
///     fn init(&self) -> f32 { 0.0 }
///
///     #[inline(always)]
///     fn accumulate(&self, acc: &mut f32, item: FoldItem<u8, f32>) {
///         *acc += item.pixel as f32 * item.weight;
///     }
///
///     fn finalize(&mut self, acc: f32) -> f32 { acc }
/// }
/// ```
///
/// For convenience, closures can be wrapped in [`ClosureFold`] which
/// falls back to `dyn Iterator` dispatch internally.
pub trait FoldOp<P, W> {
    /// The running accumulator type (often the same as `Output`).
    type Accumulator;

    /// The final output pixel type.
    type Output;

    /// Whether the engine should use the loop-inverted interior path
    /// (`init` / `accumulate` / `finalize` per row) for SIMD.
    ///
    /// Default: `true`.  Set to `false` for operations that override
    /// [`fold`](Self::fold) directly and whose accumulator is expensive
    /// to create in bulk (e.g. [`ClosureFold`]).
    const INVERTIBLE: bool = true;

    /// Starting accumulator value, called once per output pixel.
    fn init(&self) -> Self::Accumulator;

    /// Absorb one neighbour into the accumulator.
    ///
    /// Called once per kernel position.  The result must be independent
    /// of call order — the engine may traverse kernel positions in any
    /// order.
    fn accumulate(&self, acc: &mut Self::Accumulator, item: FoldItem<P, W>);

    /// Convert the final accumulator into the output pixel.
    fn finalize(&mut self, acc: Self::Accumulator) -> Self::Output;

    /// Process one pixel's full neighbourhood.
    ///
    /// The default implementation calls [`init`](Self::init), then
    /// [`accumulate`](Self::accumulate) for every item, then
    /// [`finalize`](Self::finalize).
    ///
    /// Override this only for operations that need all neighbours
    /// simultaneously and set [`INVERTIBLE`](Self::INVERTIBLE) to
    /// `false`.
    fn fold<I>(&mut self, neighbors: I) -> Self::Output
    where
        I: Iterator<Item = FoldItem<P, W>>,
    {
        let mut acc = self.init();
        for item in neighbors {
            self.accumulate(&mut acc, item);
        }
        self.finalize(acc)
    }
}

/// Wrapper that lets a closure be used as a [`FoldOp`].
///
/// The closure receives `&mut dyn Iterator` — the same interface as the
/// old `fold_neighborhood` API — so there is **no performance improvement**
/// over the pre-`FoldOp` code. Use this for quick prototyping and custom
/// one-off folds where the `dyn` dispatch overhead is acceptable.
///
/// Internal operations (convolution, morphology) implement `FoldOp`
/// directly on dedicated structs to get full monomorphization.
///
/// # Example
///
/// ```
/// use fovea::image::{Image, Neighborhood};
/// use fovea::pixel::MonoF32;
/// use fovea::transform::{ClosureFold, FoldItem, fold_neighborhood};
/// use fovea::border::Clamp;
///
/// let src = Image::fill(4, 4, 1u8);
/// let kernel = Neighborhood::<f32, 3, 3>::box_blur_3x3();
///
/// // ADR-0044 Phase E: the closure's output is a pixel, so it must
/// // be a pixel type — use `MonoF32` instead of raw `f32`. The
/// // `f32` weight/accumulator inside the closure stays scalar.
/// let result: Image<MonoF32> = fold_neighborhood(
///     &src,
///     kernel.weights(),
///     kernel.anchor(),
///     &Clamp,
///     ClosureFold(|neighbors: &mut dyn Iterator<Item = FoldItem<u8, f32>>| {
///         let mut sum = 0.0f32;
///         for item in neighbors {
///             sum += item.pixel as f32 * item.weight;
///         }
///         MonoF32::new(sum)
///     }),
/// );
/// ```
pub struct ClosureFold<F>(pub F);

impl<P, W, Out, F> FoldOp<P, W> for ClosureFold<F>
where
    P: Copy,
    W: Copy,
    F: FnMut(&mut dyn Iterator<Item = FoldItem<P, W>>) -> Out,
{
    type Accumulator = Vec<FoldItem<P, W>>;
    type Output = Out;

    const INVERTIBLE: bool = false;

    #[inline(always)]
    fn init(&self) -> Vec<FoldItem<P, W>> {
        Vec::new()
    }

    #[inline(always)]
    fn accumulate(&self, acc: &mut Vec<FoldItem<P, W>>, item: FoldItem<P, W>) {
        acc.push(item);
    }

    #[inline(always)]
    fn finalize(&mut self, acc: Vec<FoldItem<P, W>>) -> Out {
        let mut iter = acc.into_iter();
        (self.0)(&mut iter)
    }

    /// Direct pass-through — avoids collecting into a `Vec`.
    #[inline(always)]
    fn fold<I>(&mut self, mut neighbors: I) -> Out
    where
        I: Iterator<Item = FoldItem<P, W>>,
    {
        (self.0)(&mut neighbors)
    }
}

// ─── FoldItem ─────────────────────────────────────────────────────────────────

/// A single neighbour entry presented to [`FoldOp::fold`].
///
/// Carries the source pixel (border-resolved) and the kernel weight at that
/// position. Position information is deliberately absent: the weight grid
/// already encodes each neighbour's spatial offset, so pure weighted-
/// aggregation operations (convolution, morphology) never need to ask
/// "where is this item?".
///
/// For operations where the spatial offset matters (bilateral filter,
/// Perona-Malik, etc.) use [`MapItem`](crate::transform::MapItem) together with
/// `map_neighborhood` instead.
///
/// # Type parameters
///
/// - `P` — pixel type of the source image
/// - `W` — weight type of the kernel (`f32`, `i32`, `bool`, …)
///
/// # Example
///
/// ```
/// use fovea::transform::FoldItem;
///
/// let item = FoldItem { pixel: 42u8, weight: 1.0f32 };
/// assert_eq!(item.pixel, 42);
/// assert_eq!(item.weight, 1.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FoldItem<P, W> {
    /// Source pixel value (already border-resolved for boundary positions).
    pub pixel: P,
    /// Kernel / structuring-element weight at this position.
    pub weight: W,
}

/// Write the result of folding a neighborhood around every output pixel
/// into an existing output image.
///
/// This is the **base method** — [`fold_neighborhood`] is a convenience
/// wrapper that allocates the output for you.
///
/// # Algorithm
///
/// For every pixel in `border.output_region(…)`:
///
/// 1. Iterate the kernel positions `(dx, dy, weight)` relative to the anchor.
/// 2. Fetch the corresponding source pixel — directly for interior
///    positions (hot path), via `border.pixel_at()` for boundary positions
///    (cold path).
/// 3. Present an iterator of [`FoldItem`]s to the fold closure `f`.
/// 4. Write the closure's return value into `output` at the corresponding
///    position.
///
/// # Interior / boundary split
///
/// The engine computes the **interior rectangle** where the full kernel
/// fits inside the image. For interior positions the hot path uses
/// [`ImageView::pixel_at`] directly (no policy calls, no bounds checks).
/// Only the thin boundary strip invokes `border.pixel_at()`.
///
/// For a 1000×1000 image with a 5×5 kernel, ~99.2 % of positions are
/// interior — so the hot path dominates.
///
/// # Panics
///
/// Panics if the output image is smaller than the region returned by
/// `border.output_region()`.
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView, ImageViewMut, Neighborhood};
/// use fovea::Size;
/// use fovea::transform::FoldItem;
/// use fovea::border::{BorderPolicy, Clamp};
/// use fovea::transform::fold_neighborhood_into;
///
/// // 4×4 source filled with 1s
/// let src = Image::fill(4, 4, 1u8);
///
/// // 3×3 box kernel (normalized — each weight is 1/9)
/// let kernel = Neighborhood::<f32, 3, 3>::box_blur_3x3();
/// let anchor = kernel.anchor();
/// let weights = kernel.weights();
///
/// // Output: same size as input (Clamp extends borders)
/// let border = Clamp;
/// let out_region = BorderPolicy::<Image<u8>>::output_region(
///     &border, src.size(), weights.size(), anchor,
/// );
/// # use fovea::pixel::MonoF32;
/// let mut out = Image::<MonoF32>::zero(out_region.size.width, out_region.size.height);
///
/// use fovea::transform::ClosureFold;
///
/// fold_neighborhood_into(
///     &src, weights, anchor, &border, &mut out,
///     ClosureFold(|neighbors: &mut dyn Iterator<Item = FoldItem<u8, f32>>| {
///         let mut sum = 0.0f32;
///         for item in neighbors {
///             sum += item.pixel as f32 * item.weight;
///         }
///         MonoF32(sum)
///     }),
/// );
///
/// // Every output pixel should be 1.0 (nine 1s × weight 1/9)
/// for y in 0..out.height() {
///     for x in 0..out.width() {
///         assert!((out.pixel_at(x, y).0 - 1.0).abs() < 1e-6);
///     }
/// }
/// ```
pub fn fold_neighborhood_into<I, WI, B, O, F, P, W, Out>(
    image: &I,
    weights: &WI,
    anchor: (usize, usize),
    border: &B,
    output: &mut O,
    mut f: F,
) where
    I: RasterImage<Pixel = P>,
    P: Copy,
    WI: ImageView<Pixel = W>,
    W: Copy,
    B: BorderPolicy<I>,
    O: RasterImageMut<Pixel = Out>,
    F: FoldOp<P, W, Output = Out>,
{
    let kernel_size = weights.size();
    let output_region = border.output_region(image.size(), kernel_size, anchor);
    let interior = compute_interior_region(image.size(), kernel_size, anchor);

    assert!(
        output.width() >= output_region.size.width && output.height() >= output_region.size.height,
        "output image {}×{} is too small for the output region {}×{}",
        output.width(),
        output.height(),
        output_region.size.width,
        output_region.size.height,
    );

    // Pre-collect kernel positions (dx, dy, weight) so we don't recompute
    // them for every pixel.  For typical kernel sizes (3×3 .. 7×7) this is
    // a tiny allocation that stays in L1.
    let kernel_positions: Vec<(isize, isize, W)> = {
        let mut positions = Vec::with_capacity(kernel_size.width * kernel_size.height);
        for ky in 0..kernel_size.height {
            for kx in 0..kernel_size.width {
                let dx = kx as isize - anchor.0 as isize;
                let dy = ky as isize - anchor.1 as isize;
                let w = weights.pixel_at(kx, ky);
                positions.push((dx, dy, w));
            }
        }
        positions
    };

    // Offset from the output region origin to the image coordinate system.
    let ox = output_region.left();
    let oy = output_region.top();

    // ── HOT PATH — interior positions ────────────────────────────────
    if let Some(interior) = interior {
        let int_left = interior.left().max(ox);
        let int_top = interior.top().max(oy);
        let int_right = interior.right().min(output_region.right());
        let int_bottom = interior.bottom().min(output_region.bottom());

        let int_width = int_right.saturating_sub(int_left);

        if int_width > 0 && int_top < int_bottom {
            if F::INVERTIBLE {
                // ── Loop-inverted path (SIMD-friendly) ───────────────
                //
                // Kernel positions are outer, pixel scan is inner.
                // The inner loop is a contiguous elementwise operation
                // that LLVM auto-vectorizes (vpminub, vfmadd231ps, …).
                let mut acc_row: Vec<F::Accumulator> = (0..int_width).map(|_| f.init()).collect();

                for cy in int_top..int_bottom {
                    // Kernel-outer sweep
                    for &(dx, dy, w) in &kernel_positions {
                        let src_row = image.row((cy as isize + dy) as usize);
                        let start = (int_left as isize + dx) as usize;
                        let src_slice = &src_row[start..start + int_width];

                        for i in 0..int_width {
                            f.accumulate(
                                &mut acc_row[i],
                                FoldItem {
                                    pixel: src_slice[i],
                                    weight: w,
                                },
                            );
                        }
                    }

                    // Finalize & write, re-init for next row
                    let out_row = &mut output.row_mut(cy - oy)[int_left - ox..int_right - ox];
                    for i in 0..int_width {
                        let new_init = f.init();
                        let acc = std::mem::replace(&mut acc_row[i], new_init);
                        out_row[i] = f.finalize(acc);
                    }
                }
            } else {
                // ── Per-pixel fallback (non-invertible ops) ──────────
                for cy in int_top..int_bottom {
                    for cx in int_left..int_right {
                        let iter = kernel_positions.iter().map(|&(dx, dy, w)| {
                            let sx = (cx as isize + dx) as usize;
                            let sy = (cy as isize + dy) as usize;
                            FoldItem {
                                pixel: image.pixel_at(sx, sy),
                                weight: w,
                            }
                        });
                        let result = f.fold(iter);
                        *output.pixel_at_mut(cx - ox, cy - oy) = result;
                    }
                }
            }
        }
    }

    // ── COLD PATH — boundary positions ───────────────────────────────
    // Iterate all output positions that are NOT in the interior.
    for cy in output_region.top()..output_region.bottom() {
        for cx in output_region.left()..output_region.right() {
            // Skip if this position is inside the interior (already handled).
            if let Some(ref interior) = interior {
                if cx >= interior.left()
                    && cx < interior.right()
                    && cy >= interior.top()
                    && cy < interior.bottom()
                {
                    continue;
                }
            }

            let iter = kernel_positions.iter().map(|&(dx, dy, w)| {
                let pixel = border.pixel_at(image, cx as isize + dx, cy as isize + dy);
                FoldItem { pixel, weight: w }
            });

            let result = f.fold(iter);
            *output.pixel_at_mut(cx - ox, cy - oy) = result;
        }
    }
}

/// Fold a neighborhood around every output pixel and return a newly
/// allocated [`Image`] with the results.
///
/// This is a convenience wrapper around [`fold_neighborhood_into`].
/// It allocates an output `Image<Out>` of the correct size (determined
/// by `border.output_region(…)`) and fills it via the base method.
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView, Neighborhood};
/// use fovea::transform::FoldItem;
/// use fovea::border::Skip;
/// use fovea::transform::fold_neighborhood;
///
/// let src = Image::generate(5, 5, |x, y| (x + y * 5) as f32);
/// let kernel = Neighborhood::<f32, 3, 3>::new([
///     0.0, 0.0, 0.0,
///     0.0, 1.0, 0.0,
///     0.0, 0.0, 0.0,
/// ]);
///
/// // Identity kernel with Skip: output is 3×3 (interior only)
/// use fovea::transform::ClosureFold;
///
/// let result = fold_neighborhood(
///     &src,
///     kernel.weights(),
///     kernel.anchor(),
///     &Skip,
///     ClosureFold(|neighbors: &mut dyn Iterator<Item = FoldItem<f32, f32>>| {
///         let mut sum = 0.0f32;
///         for item in neighbors {
///             sum += item.pixel * item.weight;
///         }
///         fovea::pixel::MonoF32(sum)
///     }),
/// );
///
/// assert_eq!(result.width(), 3);
/// assert_eq!(result.height(), 3);
///
/// // The identity kernel should reproduce the interior pixels
/// for y in 0..3 {
///     for x in 0..3 {
///         assert_eq!(
///             result.pixel_at(x, y).0,
///             src.pixel_at(x + 1, y + 1),
///         );
///     }
/// }
/// ```
#[must_use]
pub fn fold_neighborhood<I, WI, B, F, P, W, Out>(
    image: &I,
    weights: &WI,
    anchor: (usize, usize),
    border: &B,
    f: F,
) -> Image<Out>
where
    I: RasterImage<Pixel = P>,
    P: Copy,
    WI: ImageView<Pixel = W>,
    W: Copy,
    B: BorderPolicy<I>,
    Out: ZeroablePixel,
    F: FoldOp<P, W, Output = Out>,
{
    let output_region = border.output_region(image.size(), weights.size(), anchor);
    let mut out = Image::<Out>::zero(output_region.size.width, output_region.size.height);
    fold_neighborhood_into(image, weights, anchor, border, &mut out, f);
    out
}

// ─── Closure-based convenience wrappers ──────────────────────────────────────

/// Convenience wrapper: [`fold_neighborhood_into`] accepting a closure.
///
/// Wraps the closure in [`ClosureFold`], which uses `dyn Iterator`
/// dispatch internally. For maximum performance, implement [`FoldOp`]
/// directly instead.
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView, ImageViewMut, Neighborhood};
/// use fovea::Size;
/// use fovea::border::{BorderPolicy, Clamp};
/// use fovea::transform::fold_neighborhood_fn_into;
///
/// let src = Image::fill(4, 4, 1u8);
/// let kernel = Neighborhood::<f32, 3, 3>::box_blur_3x3();
/// let anchor = kernel.anchor();
/// let weights = kernel.weights();
///
/// let border = Clamp;
/// let out_region = BorderPolicy::<Image<u8>>::output_region(
///     &border, src.size(), weights.size(), anchor,
/// );
/// # use fovea::pixel::MonoF32;
/// let mut out = Image::<MonoF32>::zero(out_region.size.width, out_region.size.height);
///
/// fold_neighborhood_fn_into(
///     &src, weights, anchor, &border, &mut out,
///     |neighbors| {
///         let mut sum = 0.0f32;
///         for item in neighbors {
///             sum += item.pixel as f32 * item.weight;
///         }
///         MonoF32(sum)
///     },
/// );
/// ```
pub fn fold_neighborhood_fn_into<I, WI, B, O, F, P, W, Out>(
    image: &I,
    weights: &WI,
    anchor: (usize, usize),
    border: &B,
    output: &mut O,
    f: F,
) where
    I: RasterImage<Pixel = P>,
    P: Copy,
    WI: ImageView<Pixel = W>,
    W: Copy,
    B: BorderPolicy<I>,
    O: RasterImageMut<Pixel = Out>,
    F: FnMut(&mut dyn Iterator<Item = FoldItem<P, W>>) -> Out,
{
    fold_neighborhood_into(image, weights, anchor, border, output, ClosureFold(f));
}

/// Convenience wrapper: [`fold_neighborhood`] accepting a closure.
///
/// Wraps the closure in [`ClosureFold`], which uses `dyn Iterator`
/// dispatch internally. For maximum performance, implement [`FoldOp`]
/// directly instead.
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView, Neighborhood};
/// use fovea::border::Skip;
/// use fovea::transform::fold_neighborhood_fn;
///
/// # use fovea::pixel::MonoF32;
/// let src = Image::generate(5, 5, |x, y| (x + y * 5) as f32);
/// let kernel = Neighborhood::<f32, 3, 3>::new([
///     0.0, 0.0, 0.0,
///     0.0, 1.0, 0.0,
///     0.0, 0.0, 0.0,
/// ]);
///
/// let result = fold_neighborhood_fn(
///     &src,
///     kernel.weights(),
///     kernel.anchor(),
///     &Skip,
///     |neighbors| {
///         let mut sum = 0.0f32;
///         for item in neighbors {
///             sum += item.pixel * item.weight;
///         }
///         MonoF32(sum)
///     },
/// );
///
/// assert_eq!(result.width(), 3);
/// assert_eq!(result.height(), 3);
/// ```
#[must_use]
pub fn fold_neighborhood_fn<I, WI, B, F, P, W, Out>(
    image: &I,
    weights: &WI,
    anchor: (usize, usize),
    border: &B,
    f: F,
) -> Image<Out>
where
    I: RasterImage<Pixel = P>,
    P: Copy,
    WI: ImageView<Pixel = W>,
    W: Copy,
    B: BorderPolicy<I>,
    Out: ZeroablePixel,
    F: FnMut(&mut dyn Iterator<Item = FoldItem<P, W>>) -> Out,
{
    fold_neighborhood(image, weights, anchor, border, ClosureFold(f))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::border::{Clamp, Constant, Mirror, Skip, Wrap};
    use crate::image::{ImageArray, Neighborhood};

    // ── helpers ──────────────────────────────────────────────────────

    /// Create a 4×4 image with pixel values 0..15.
    fn make_4x4() -> Image<u8> {
        Image::generate(4, 4, |x, y| (x + y * 4) as u8)
    }

    /// Create a 5×5 image with pixel values 0..24.
    fn make_5x5() -> Image<u8> {
        Image::generate(5, 5, |x, y| (x + y * 5) as u8)
    }

    /// Identity / sum fold: sums pixel × weight.
    ///
    /// Implements `FoldOp` directly — fully monomorphized, no `dyn` dispatch.
    /// Generic over pixel type `P` so it works for both `u8` and `f32` images.
    struct SumFold;

    impl FoldOp<u8, f32> for SumFold {
        type Accumulator = f32;
        type Output = MonoF32;

        #[inline(always)]
        fn init(&self) -> f32 {
            0.0
        }

        #[inline(always)]
        fn accumulate(&self, acc: &mut f32, item: FoldItem<u8, f32>) {
            *acc += item.pixel as f32 * item.weight;
        }

        #[inline(always)]
        fn finalize(&mut self, acc: f32) -> MonoF32 {
            MonoF32(acc)
        }
    }

    impl FoldOp<f32, f32> for SumFold {
        type Accumulator = f32;
        type Output = MonoF32;

        #[inline(always)]
        fn init(&self) -> f32 {
            0.0
        }

        #[inline(always)]
        fn accumulate(&self, acc: &mut f32, item: FoldItem<f32, f32>) {
            *acc += item.pixel * item.weight;
        }

        #[inline(always)]
        fn finalize(&mut self, acc: f32) -> MonoF32 {
            MonoF32(acc)
        }
    }

    /// Alias: identity_fold and sum_fold are the same operation
    /// (weighted sum with identity kernel = center pixel value).
    fn identity_fold() -> SumFold {
        SumFold
    }

    fn sum_fold() -> SumFold {
        SumFold
    }

    // ── FoldItem basic tests ──────────────────────────────────────

    #[test]
    fn fold_item_fields() {
        let item = FoldItem {
            pixel: 100u8,
            weight: 0.5f32,
        };
        assert_eq!(item.pixel, 100);
        assert_eq!(item.weight, 0.5);
    }

    #[test]
    fn fold_item_is_copy() {
        let item = FoldItem {
            pixel: 1u8,
            weight: 1.0f32,
        };
        let item2 = item; // Copy
        assert_eq!(item, item2);
    }

    #[test]
    fn fold_item_debug() {
        let item = FoldItem {
            pixel: 0u8,
            weight: 0.0f32,
        };
        let dbg = format!("{:?}", item);
        assert!(dbg.contains("FoldItem"));
    }

    // ── Identity kernel tests ───────────────────────────────────────

    #[test]
    fn identity_3x3_skip_preserves_interior() {
        let src = make_5x5();
        let kernel = Neighborhood::<f32, 3, 3>::new([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);

        let result = fold_neighborhood(
            &src,
            kernel.weights(),
            kernel.anchor(),
            &Skip,
            identity_fold(),
        );

        // Skip with 3×3 on 5×5 → 3×3 output
        assert_eq!(result.width(), 3);
        assert_eq!(result.height(), 3);

        for y in 0..3 {
            for x in 0..3 {
                assert_eq!(
                    result.pixel_at(x, y),
                    MonoF32(src.pixel_at(x + 1, y + 1) as f32),
                    "mismatch at ({}, {})",
                    x,
                    y,
                );
            }
        }
    }

    #[test]
    fn identity_3x3_clamp_preserves_all() {
        let src = make_5x5();
        let kernel = Neighborhood::<f32, 3, 3>::new([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);

        let result = fold_neighborhood(
            &src,
            kernel.weights(),
            kernel.anchor(),
            &Clamp,
            identity_fold(),
        );

        assert_eq!(result.width(), 5);
        assert_eq!(result.height(), 5);

        for y in 0..5 {
            for x in 0..5 {
                assert_eq!(
                    result.pixel_at(x, y),
                    MonoF32(src.pixel_at(x, y) as f32),
                    "mismatch at ({}, {})",
                    x,
                    y,
                );
            }
        }
    }

    #[test]
    fn identity_1x1_all_policies() {
        let src = make_4x4();
        let kernel = Neighborhood::<f32, 1, 1>::new([1.0]);

        for policy_name in &["skip", "clamp", "mirror", "wrap", "constant"] {
            let result: Image<MonoF32> = match *policy_name {
                "skip" => fold_neighborhood(
                    &src,
                    kernel.weights(),
                    kernel.anchor(),
                    &Skip,
                    identity_fold(),
                ),
                "clamp" => fold_neighborhood(
                    &src,
                    kernel.weights(),
                    kernel.anchor(),
                    &Clamp,
                    identity_fold(),
                ),
                "mirror" => fold_neighborhood(
                    &src,
                    kernel.weights(),
                    kernel.anchor(),
                    &Mirror,
                    identity_fold(),
                ),
                "wrap" => fold_neighborhood(
                    &src,
                    kernel.weights(),
                    kernel.anchor(),
                    &Wrap,
                    identity_fold(),
                ),
                "constant" => fold_neighborhood(
                    &src,
                    kernel.weights(),
                    kernel.anchor(),
                    &Constant(0u8),
                    identity_fold(),
                ),
                _ => unreachable!(),
            };

            assert_eq!(result.width(), 4, "policy={}", policy_name);
            assert_eq!(result.height(), 4, "policy={}", policy_name);
            for y in 0..4 {
                for x in 0..4 {
                    assert_eq!(
                        result.pixel_at(x, y),
                        MonoF32(src.pixel_at(x, y) as f32),
                        "policy={} at ({}, {})",
                        policy_name,
                        x,
                        y,
                    );
                }
            }
        }
    }

    // ── Box blur tests ──────────────────────────────────────────────

    #[test]
    fn box_blur_3x3_uniform_image() {
        // All pixels = 10, normalized box blur (weights = 1/9 each)
        // → every output pixel = 10 * (1/9) * 9 = 10
        let src = Image::fill(6, 6, 10u8);
        let kernel = Neighborhood::<f32, 3, 3>::box_blur_3x3();

        let result = fold_neighborhood(&src, kernel.weights(), kernel.anchor(), &Clamp, sum_fold());

        assert_eq!(result.width(), 6);
        assert_eq!(result.height(), 6);
        for y in 0..6 {
            for x in 0..6 {
                let v = result.pixel_at(x, y);
                assert!(
                    (v.0 - 10.0).abs() < 1e-4,
                    "expected ~10.0, got {} at ({}, {})",
                    v.0,
                    x,
                    y,
                );
            }
        }
    }

    #[test]
    fn box_blur_3x3_skip_interior_sum() {
        // 3×3 image, all 1s → normalized 3×3 box blur with Skip → 1×1 output
        let src = Image::fill(3, 3, 1u8);
        let kernel = Neighborhood::<f32, 3, 3>::box_blur_3x3();

        let result = fold_neighborhood(&src, kernel.weights(), kernel.anchor(), &Skip, sum_fold());

        assert_eq!(result.width(), 1);
        assert_eq!(result.height(), 1);
        // sum of 9 pixels each 1, weight = 1/9 each → sum = 1.0
        let v = result.pixel_at(0, 0);
        assert!((v.0 - 1.0).abs() < 1e-4, "expected ~1.0, got {}", v.0,);
    }

    // ── Known-value convolution tests ───────────────────────────────

    #[test]
    fn known_3x3_sum_center_pixel() {
        // 3×3 kernel with weight only at center → result = center pixel
        let src = make_4x4();
        let kernel = Neighborhood::<f32, 3, 3>::new([0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0]);

        let result = fold_neighborhood(&src, kernel.weights(), kernel.anchor(), &Clamp, sum_fold());

        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(
                    result.pixel_at(x, y),
                    MonoF32(src.pixel_at(x, y) as f32 * 2.0),
                    "at ({}, {})",
                    x,
                    y,
                );
            }
        }
    }

    #[test]
    fn horizontal_gradient_kernel() {
        // Kernel [-1, 0, 1] horizontally — should detect horizontal edges
        let src = Image::generate(5, 1, |x, _| x as u8); // [0, 1, 2, 3, 4]
        let kernel = Neighborhood::<f32, 3, 1>::new([-1.0, 0.0, 1.0]);

        let result = fold_neighborhood(&src, kernel.weights(), kernel.anchor(), &Skip, sum_fold());

        // Skip: output width = 5 - 2 = 3 (positions 1, 2, 3)
        assert_eq!(result.width(), 3);
        assert_eq!(result.height(), 1);

        // At each interior position: right - left = (x+1) - (x-1) = 2
        for x in 0..3 {
            assert_eq!(result.pixel_at(x, 0), MonoF32(2.0), "at x={}", x,);
        }
    }

    // ── fold_neighborhood_into tests ────────────────────────────────

    #[test]
    fn fold_into_writes_correct_output() {
        let src = make_4x4();
        let kernel = Neighborhood::<f32, 3, 3>::new([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        let border = Clamp;
        let out_region = BorderPolicy::<Image<u8>>::output_region(
            &border,
            src.size(),
            kernel.weights().size(),
            kernel.anchor(),
        );
        let mut out = Image::<MonoF32>::zero(out_region.size.width, out_region.size.height);

        fold_neighborhood_into(
            &src,
            kernel.weights(),
            kernel.anchor(),
            &border,
            &mut out,
            identity_fold(),
        );

        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(
                    out.pixel_at(x, y),
                    MonoF32(src.pixel_at(x, y) as f32),
                    "at ({}, {})",
                    x,
                    y,
                );
            }
        }
    }

    #[test]
    fn fold_into_skip_smaller_output() {
        let src = make_5x5();
        let kernel = Neighborhood::<f32, 3, 3>::new([0.0; 9]);
        let border = Skip;
        let out_region = BorderPolicy::<Image<u8>>::output_region(
            &border,
            src.size(),
            kernel.weights().size(),
            kernel.anchor(),
        );
        let mut out = Image::<MonoF32>::zero(out_region.size.width, out_region.size.height);

        assert_eq!(out.width(), 3);
        assert_eq!(out.height(), 3);

        fold_neighborhood_into(
            &src,
            kernel.weights(),
            kernel.anchor(),
            &border,
            &mut out,
            ClosureFold(|_: &mut dyn Iterator<Item = FoldItem<u8, f32>>| MonoF32(42.0)),
        );

        for y in 0..3 {
            for x in 0..3 {
                assert_eq!(out.pixel_at(x, y), MonoF32(42.0));
            }
        }
    }

    // ── Border policy behavior ──────────────────────────────────────

    #[test]
    fn constant_border_zero_padding() {
        // 3×3 image with all 1s, constant(0) border, normalized 3×3 box kernel
        // (weights = 1/9 each). Pixel × weight sums:
        //
        //   corner (0,0) sees: [0,0,0, 0,1,1, 0,1,1] → 4 × (1/9) = 4/9
        //   edge   (1,0) sees: [0,0,0, 1,1,1, 1,1,1] → 6 × (1/9) = 6/9
        //   center (1,1) sees: [1,1,1, 1,1,1, 1,1,1] → 9 × (1/9) = 1.0
        let src = Image::fill(3, 3, 1u8);
        let kernel = Neighborhood::<f32, 3, 3>::box_blur_3x3();

        let result = fold_neighborhood(
            &src,
            kernel.weights(),
            kernel.anchor(),
            &Constant(0u8),
            sum_fold(),
        );

        assert_eq!(result.width(), 3);
        assert_eq!(result.height(), 3);

        // Center pixel (1,1): full neighborhood in-bounds, all 1s → 1.0
        let center = result.pixel_at(1, 1);
        assert!(
            (center.0 - 1.0).abs() < 1e-4,
            "center: expected 1.0, got {}",
            center.0,
        );

        // Corner pixel (0,0): 4 in-bounds pixels → 4/9
        let corner = result.pixel_at(0, 0);
        assert!(
            (corner.0 - 4.0 / 9.0).abs() < 1e-4,
            "corner: expected {}, got {}",
            4.0 / 9.0,
            corner.0,
        );

        // Edge pixel (1,0): 6 in-bounds pixels → 6/9
        let edge = result.pixel_at(1, 0);
        assert!(
            (edge.0 - 6.0 / 9.0).abs() < 1e-4,
            "edge: expected {}, got {}",
            6.0 / 9.0,
            edge.0,
        );
    }

    #[test]
    fn clamp_border_uniform_image() {
        // Uniform image + clamp + normalized box blur (weights = 1/9)
        // → every pixel sees 9 copies of 7 → 7 * (1/9) * 9 = 7.0
        let src = Image::fill(4, 4, 7u8);
        let kernel = Neighborhood::<f32, 3, 3>::box_blur_3x3();

        let result = fold_neighborhood(&src, kernel.weights(), kernel.anchor(), &Clamp, sum_fold());

        for y in 0..4 {
            for x in 0..4 {
                let v = result.pixel_at(x, y);
                assert!(
                    (v.0 - 7.0).abs() < 1e-4,
                    "at ({}, {}): expected 7.0, got {}",
                    x,
                    y,
                    v.0,
                );
            }
        }
    }

    #[test]
    fn mirror_border_uniform_image() {
        // Uniform image + mirror + normalized → 3 * (1/9) * 9 = 3.0
        let src = Image::fill(4, 4, 3u8);
        let kernel = Neighborhood::<f32, 3, 3>::box_blur_3x3();

        let result =
            fold_neighborhood(&src, kernel.weights(), kernel.anchor(), &Mirror, sum_fold());

        for y in 0..4 {
            for x in 0..4 {
                let v = result.pixel_at(x, y);
                assert!(
                    (v.0 - 3.0).abs() < 1e-4,
                    "at ({}, {}): expected 3.0, got {}",
                    x,
                    y,
                    v.0,
                );
            }
        }
    }

    #[test]
    fn wrap_border_uniform_image() {
        // Uniform image + wrap + normalized → 5 * (1/9) * 9 = 5.0
        let src = Image::fill(4, 4, 5u8);
        let kernel = Neighborhood::<f32, 3, 3>::box_blur_3x3();

        let result = fold_neighborhood(&src, kernel.weights(), kernel.anchor(), &Wrap, sum_fold());

        for y in 0..4 {
            for x in 0..4 {
                let v = result.pixel_at(x, y);
                assert!(
                    (v.0 - 5.0).abs() < 1e-4,
                    "at ({}, {}): expected 5.0, got {}",
                    x,
                    y,
                    v.0,
                );
            }
        }
    }

    // ── Non-square kernel ───────────────────────────────────────────

    #[test]
    fn horizontal_1d_kernel() {
        // 1×3 horizontal kernel: [1, 2, 1] — weighted sum
        let src = Image::generate(5, 1, |x, _| x as u8); // [0, 1, 2, 3, 4]
        let kernel = Neighborhood::<f32, 3, 1>::new([1.0, 2.0, 1.0]);

        let result = fold_neighborhood(&src, kernel.weights(), kernel.anchor(), &Skip, sum_fold());

        assert_eq!(result.width(), 3);
        assert_eq!(result.height(), 1);

        // pos 1: 0*1 + 1*2 + 2*1 = 4
        assert_eq!(result.pixel_at(0, 0), MonoF32(4.0));
        // pos 2: 1*1 + 2*2 + 3*1 = 8
        assert_eq!(result.pixel_at(1, 0), MonoF32(8.0));
        // pos 3: 2*1 + 3*2 + 4*1 = 12
        assert_eq!(result.pixel_at(2, 0), MonoF32(12.0));
    }

    #[test]
    fn vertical_1d_kernel() {
        let src = Image::generate(1, 5, |_, y| y as u8); // [0, 1, 2, 3, 4] vertical
        let kernel = Neighborhood::<f32, 1, 3>::new([1.0, 2.0, 1.0]);

        let result = fold_neighborhood(&src, kernel.weights(), kernel.anchor(), &Skip, sum_fold());

        assert_eq!(result.width(), 1);
        assert_eq!(result.height(), 3);

        assert_eq!(result.pixel_at(0, 0), MonoF32(4.0)); // 0*1 + 1*2 + 2*1
        assert_eq!(result.pixel_at(0, 1), MonoF32(8.0)); // 1*1 + 2*2 + 3*1
        assert_eq!(result.pixel_at(0, 2), MonoF32(12.0)); // 2*1 + 3*2 + 4*1
    }

    // ── Custom anchor ───────────────────────────────────────────────

    #[test]
    fn custom_anchor_top_left() {
        // 3×3 identity kernel with anchor at (0,0) instead of (1,1)
        let src = make_5x5();
        // Weight 1.0 at array position [0][0] = anchor (0,0), all others 0.0.
        // sum_fold will return pixel_at(anchor) * 1.0 = the source pixel.
        let kernel = Neighborhood::<f32, 3, 3>::with_anchor(
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            (0, 0),
        );

        // With anchor at (0,0): left margin = 0, top margin = 0,
        // right margin = 2, bottom margin = 2.
        // Skip output: 3×3 starting at (0,0).
        // With anchor at (0,0), weight 1.0 is at kernel position (0,0) = the anchor.
        // SumFold returns pixel * weight = source pixel * 1.0 for that position.
        let result = fold_neighborhood(&src, kernel.weights(), kernel.anchor(), &Skip, sum_fold());

        assert_eq!(result.width(), 3);
        assert_eq!(result.height(), 3);

        // The center pixel at output (0,0) corresponds to src (0,0)
        for y in 0..3 {
            for x in 0..3 {
                assert_eq!(
                    result.pixel_at(x, y),
                    MonoF32(src.pixel_at(x, y) as f32),
                    "at ({}, {})",
                    x,
                    y,
                );
            }
        }
    }

    // ── 5×5 kernel ──────────────────────────────────────────────────

    #[test]
    fn box_blur_5x5_uniform() {
        // Uniform image + normalized 5×5 box → 4 * (1/25) * 25 = 4.0
        let src = Image::fill(8, 8, 4u8);
        let kernel = Neighborhood::<f32, 5, 5>::box_blur_5x5();

        let result = fold_neighborhood(&src, kernel.weights(), kernel.anchor(), &Clamp, sum_fold());

        assert_eq!(result.width(), 8);
        assert_eq!(result.height(), 8);

        for y in 0..8 {
            for x in 0..8 {
                let v = result.pixel_at(x, y);
                assert!(
                    (v.0 - 4.0).abs() < 1e-4,
                    "at ({}, {}): expected 4.0, got {}",
                    x,
                    y,
                    v.0,
                );
            }
        }
    }

    // ── ImageArray as source ────────────────────────────────────────

    #[test]
    fn works_with_image_array_source() {
        let src: ImageArray<u8, 4, 4> = ImageArray::generate(|x, y| (x + y * 4) as u8);
        let kernel = Neighborhood::<f32, 3, 3>::new([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);

        let result = fold_neighborhood(
            &src,
            kernel.weights(),
            kernel.anchor(),
            &Clamp,
            identity_fold(),
        );

        assert_eq!(result.width(), 4);
        assert_eq!(result.height(), 4);

        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(result.pixel_at(x, y), MonoF32(src.pixel_at(x, y) as f32),);
            }
        }
    }

    // ── 1×1 image edge case ─────────────────────────────────────────

    #[test]
    fn single_pixel_image_clamp() {
        // 1×1 image, clamp border, normalized box blur → 42 * (1/9) * 9 = 42.0
        let src = Image::fill(1, 1, 42u8);
        let kernel = Neighborhood::<f32, 3, 3>::box_blur_3x3();

        let result = fold_neighborhood(&src, kernel.weights(), kernel.anchor(), &Clamp, sum_fold());

        assert_eq!(result.width(), 1);
        assert_eq!(result.height(), 1);
        let v = result.pixel_at(0, 0);
        assert!((v.0 - 42.0).abs() < 1e-4, "expected 42.0, got {}", v.0,);
    }

    #[test]
    fn single_pixel_image_constant() {
        let src = Image::fill(1, 1, 9u8);
        let kernel = Neighborhood::<f32, 3, 3>::new([1.0; 9]);

        let result = fold_neighborhood(
            &src,
            kernel.weights(),
            kernel.anchor(),
            &Constant(0u8),
            sum_fold(),
        );

        assert_eq!(result.width(), 1);
        assert_eq!(result.height(), 1);
        // Only center pixel is 9, rest are 0
        let v = result.pixel_at(0, 0);
        assert_eq!(v, MonoF32(9.0));
    }

    // ── Kernel larger than image (Skip returns empty) ───────────────

    #[test]
    fn kernel_larger_than_image_skip() {
        let src = Image::fill(2, 2, 1u8);
        let kernel = Neighborhood::<f32, 5, 5>::box_blur_5x5();

        let border = Skip;
        let out_region = BorderPolicy::<Image<u8>>::output_region(
            &border,
            src.size(),
            kernel.weights().size(),
            kernel.anchor(),
        );

        // No interior → output region should be empty (width or height = 0)
        assert_eq!(out_region.area(), 0);
    }

    // ── fold_neighborhood_into with oversized output ────────────────

    #[test]
    fn fold_into_oversized_output() {
        // Output is larger than needed — only the output region should be written
        let src = make_4x4();
        let kernel = Neighborhood::<f32, 3, 3>::new([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);

        let mut out = Image::<MonoF32>::fill(6, 6, MonoF32(-1.0));

        fold_neighborhood_into(
            &src,
            kernel.weights(),
            kernel.anchor(),
            &Clamp,
            &mut out,
            identity_fold(),
        );

        // Pixels within the 4×4 output region should have been written
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(
                    out.pixel_at(x, y),
                    MonoF32(src.pixel_at(x, y) as f32),
                    "at ({}, {})",
                    x,
                    y,
                );
            }
        }

        // Pixels outside the 4×4 region should still be -1.0
        for x in 4..6 {
            assert_eq!(out.pixel_at(x, 0), MonoF32(-1.0));
        }
        for y in 4..6 {
            assert_eq!(out.pixel_at(0, y), MonoF32(-1.0));
        }
    }

    // ── Sobel-like kernel correctness ───────────────────────────────

    #[test]
    fn sobel_y_on_horizontal_gradient() {
        // Image: each row = [0, 1, 2, 3, 4] → constant horizontal gradient
        // sobel_y detects vertical edges (dI/dx):
        //   [-1, 0, 1]
        //   [-2, 0, 2]
        //   [-1, 0, 1]
        let src = Image::generate(5, 5, |x, _| x as u8);
        let kernel = Neighborhood::<f32, 3, 3>::sobel_y();

        let result = fold_neighborhood(&src, kernel.weights(), kernel.anchor(), &Skip, sum_fold());

        assert_eq!(result.width(), 3);
        assert_eq!(result.height(), 3);

        // At any interior pixel with horizontal gradient = 1:
        //   -1*(x-1) + 0 + 1*(x+1)  = 2
        //   -2*(x-1) + 0 + 2*(x+1)  = 4
        //   -1*(x-1) + 0 + 1*(x+1)  = 2
        //   Total = 8
        for y in 0..3 {
            for x in 0..3 {
                let v = result.pixel_at(x, y);
                assert!(
                    (v.0 - 8.0).abs() < 1e-4,
                    "at ({}, {}): expected 8.0, got {}",
                    x,
                    y,
                    v.0,
                );
            }
        }
    }

    #[test]
    fn sobel_x_on_vertical_gradient() {
        // Image: each column has values 0..4 vertically
        // sobel_x detects horizontal edges (dI/dy):
        //   [-1, -2, -1]
        //   [ 0,  0,  0]
        //   [ 1,  2,  1]
        let src = Image::generate(5, 5, |_, y| y as u8);
        let kernel = Neighborhood::<f32, 3, 3>::sobel_x();

        let result = fold_neighborhood(&src, kernel.weights(), kernel.anchor(), &Skip, sum_fold());

        assert_eq!(result.width(), 3);
        assert_eq!(result.height(), 3);

        // At any interior pixel with vertical gradient = 1:
        //   -1*(y-1) + -2*(y-1) + -1*(y-1) = -4*(y-1)
        //    0       +  0       +  0        = 0
        //    1*(y+1) +  2*(y+1) +  1*(y+1)  = 4*(y+1)
        //   Total = 4*(y+1) - 4*(y-1) = 8
        for y in 0..3 {
            for x in 0..3 {
                let v = result.pixel_at(x, y);
                assert!(
                    (v.0 - 8.0).abs() < 1e-4,
                    "at ({}, {}): expected 8.0, got {}",
                    x,
                    y,
                    v.0,
                );
            }
        }
    }

    #[test]
    fn sobel_x_on_uniform_is_zero() {
        let src = Image::fill(5, 5, 100u8);
        let kernel = Neighborhood::<f32, 3, 3>::sobel_x();

        let result = fold_neighborhood(&src, kernel.weights(), kernel.anchor(), &Clamp, sum_fold());

        for y in 0..5 {
            for x in 0..5 {
                let v = result.pixel_at(x, y);
                assert!(
                    v.0.abs() < 1e-4,
                    "at ({}, {}): expected ~0, got {}",
                    x,
                    y,
                    v.0,
                );
            }
        }
    }

    // ── Morphology-style: max filter ────────────────────────────────

    #[test]
    fn max_filter_3x3() {
        let src = Image::generate(5, 5, |x, y| (x + y * 5) as u8);
        let kernel = Neighborhood::<f32, 3, 3>::new([1.0; 9]);

        let result = fold_neighborhood(
            &src,
            kernel.weights(),
            kernel.anchor(),
            &Skip,
            ClosureFold(|neighbors: &mut dyn Iterator<Item = FoldItem<u8, f32>>| {
                neighbors.map(|item| item.pixel).max().unwrap_or(0)
            }),
        );

        // At interior (1,1): neighborhood is rows 0..3 × cols 0..3
        // max = pixel_at(2,2) = 2 + 2*5 = 12
        assert_eq!(result.pixel_at(0, 0), 12);

        // At interior (2,2): rows 1..4, cols 1..4 → max = pixel_at(3,3) = 18
        assert_eq!(result.pixel_at(1, 1), 18);
    }

    // ── Morphology-style: min filter ────────────────────────────────

    #[test]
    fn min_filter_3x3() {
        let src = Image::generate(5, 5, |x, y| (x + y * 5) as u8);
        let kernel = Neighborhood::<f32, 3, 3>::new([1.0; 9]);

        let result = fold_neighborhood(
            &src,
            kernel.weights(),
            kernel.anchor(),
            &Skip,
            ClosureFold(|neighbors: &mut dyn Iterator<Item = FoldItem<u8, f32>>| {
                neighbors.map(|item| item.pixel).min().unwrap_or(255)
            }),
        );

        // At interior (1,1): neighborhood rows 0..3, cols 0..3 → min = pixel_at(0,0) = 0
        assert_eq!(result.pixel_at(0, 0), 0);

        // At interior (3,3): rows 2..5, cols 2..5 → min = pixel_at(2,2) = 12
        assert_eq!(result.pixel_at(2, 2), 12);
    }

    // ── Empty interior (all boundary) ───────────────────────────────

    #[test]
    fn all_boundary_small_image_large_kernel() {
        // 3×3 image with 3×3 kernel and Clamp → interior is 1×1, boundary is 8 pixels
        let src = Image::generate(3, 3, |x, y| (x + y * 3) as u8);
        let kernel = Neighborhood::<f32, 3, 3>::new([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);

        let result = fold_neighborhood(
            &src,
            kernel.weights(),
            kernel.anchor(),
            &Clamp,
            identity_fold(),
        );

        // Identity kernel should reproduce the source
        for y in 0..3 {
            for x in 0..3 {
                assert_eq!(
                    result.pixel_at(x, y),
                    MonoF32(src.pixel_at(x, y) as f32),
                    "at ({}, {})",
                    x,
                    y,
                );
            }
        }
    }

    // ── Consistency: fold vs fold_into ───────────────────────────────

    #[test]
    fn fold_and_fold_into_produce_same_result() {
        let src = make_5x5();
        let kernel = Neighborhood::<f32, 3, 3>::box_blur_3x3();
        let border = Clamp;

        let result1 =
            fold_neighborhood(&src, kernel.weights(), kernel.anchor(), &border, sum_fold());

        let out_region = BorderPolicy::<Image<u8>>::output_region(
            &border,
            src.size(),
            kernel.weights().size(),
            kernel.anchor(),
        );
        let mut result2 = Image::<MonoF32>::zero(out_region.size.width, out_region.size.height);
        fold_neighborhood_into(
            &src,
            kernel.weights(),
            kernel.anchor(),
            &border,
            &mut result2,
            sum_fold(),
        );

        for y in 0..result1.height() {
            for x in 0..result1.width() {
                assert_eq!(
                    result1.pixel_at(x, y),
                    result2.pixel_at(x, y),
                    "at ({}, {})",
                    x,
                    y,
                );
            }
        }
    }

    // ── Large-ish image: verify interior dominates ───────────────────

    #[test]
    fn large_image_runs_without_error() {
        // Uniform image + normalized 5×5 box + mirror → 1 * (1/25) * 25 = 1.0
        let src = Image::fill(100, 100, 1u8);
        let kernel = Neighborhood::<f32, 5, 5>::box_blur_5x5();

        let result =
            fold_neighborhood(&src, kernel.weights(), kernel.anchor(), &Mirror, sum_fold());

        assert_eq!(result.width(), 100);
        assert_eq!(result.height(), 100);

        for y in 0..100 {
            for x in 0..100 {
                let v = result.pixel_at(x, y);
                assert!(
                    (v.0 - 1.0).abs() < 1e-4,
                    "at ({}, {}): expected 1.0, got {}",
                    x,
                    y,
                    v.0,
                );
            }
        }
    }

    // ── f32 pixel type ──────────────────────────────────────────────

    #[test]
    fn f32_image_identity() {
        let src = Image::generate(4, 4, |x, y| (x as f32 + y as f32 * 0.1));
        let kernel = Neighborhood::<f32, 3, 3>::new([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);

        let result = fold_neighborhood(&src, kernel.weights(), kernel.anchor(), &Clamp, sum_fold());

        for y in 0..4 {
            for x in 0..4 {
                assert!(
                    (result.pixel_at(x, y).0 - src.pixel_at(x, y)).abs() < 1e-6,
                    "at ({}, {})",
                    x,
                    y,
                );
            }
        }
    }

    // ── fold_neighborhood_fn / fold_neighborhood_fn_into ────────────

    #[test]
    fn fold_neighborhood_fn_identity_skip() {
        // fold_neighborhood_fn with an identity closure (weight=1 at center
        // only) should preserve interior pixels; Skip policy shrinks the output.
        let src = make_5x5();
        let kernel = Neighborhood::<f32, 3, 3>::new([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);

        let result = fold_neighborhood_fn(
            &src,
            kernel.weights(),
            kernel.anchor(),
            &Skip,
            |neighbors| {
                let mut sum = 0.0f32;
                for item in neighbors {
                    sum += item.pixel as f32 * item.weight;
                }
                MonoF32(sum)
            },
        );

        // Skip with 3×3 on 5×5 → 3×3 interior output
        assert_eq!(result.width(), 3);
        assert_eq!(result.height(), 3);
        for y in 0..3 {
            for x in 0..3 {
                assert_eq!(
                    result.pixel_at(x, y),
                    MonoF32(src.pixel_at(x + 1, y + 1) as f32),
                    "mismatch at ({}, {})",
                    x,
                    y,
                );
            }
        }
    }

    #[test]
    fn fold_neighborhood_fn_into_identity_clamp() {
        // fold_neighborhood_fn_into with an identity closure should write
        // pixel values unchanged into the output buffer.
        let src = make_4x4();
        let kernel = Neighborhood::<f32, 3, 3>::new([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        let border = Clamp;
        let out_region = BorderPolicy::<Image<u8>>::output_region(
            &border,
            src.size(),
            kernel.weights().size(),
            kernel.anchor(),
        );
        let mut out = Image::<MonoF32>::zero(out_region.size.width, out_region.size.height);

        fold_neighborhood_fn_into(
            &src,
            kernel.weights(),
            kernel.anchor(),
            &border,
            &mut out,
            |neighbors| {
                let mut sum = 0.0f32;
                for item in neighbors {
                    sum += item.pixel as f32 * item.weight;
                }
                MonoF32(sum)
            },
        );

        // Clamp: output same size as input; identity kernel → pixel preserved
        assert_eq!(out.width(), 4);
        assert_eq!(out.height(), 4);
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(
                    out.pixel_at(x, y),
                    MonoF32(src.pixel_at(x, y) as f32),
                    "at ({}, {})",
                    x,
                    y,
                );
            }
        }
    }

    #[test]
    fn fold_fn_matches_fold_with_closure_fold() {
        // fold_neighborhood_fn must produce identical results to calling
        // fold_neighborhood with ClosureFold wrapping the same closure body.
        let src = make_5x5();
        let kernel = Neighborhood::<f32, 3, 3>::box_blur_3x3();

        let via_fn = fold_neighborhood_fn(
            &src,
            kernel.weights(),
            kernel.anchor(),
            &Clamp,
            |neighbors| {
                let mut sum = 0.0f32;
                for item in neighbors {
                    sum += item.pixel as f32 * item.weight;
                }
                MonoF32(sum)
            },
        );

        let via_closure_fold = fold_neighborhood(
            &src,
            kernel.weights(),
            kernel.anchor(),
            &Clamp,
            ClosureFold(|neighbors: &mut dyn Iterator<Item = FoldItem<u8, f32>>| {
                let mut sum = 0.0f32;
                for item in neighbors {
                    sum += item.pixel as f32 * item.weight;
                }
                MonoF32(sum)
            }),
        );

        assert_eq!(via_fn.width(), via_closure_fold.width());
        assert_eq!(via_fn.height(), via_closure_fold.height());
        for y in 0..via_fn.height() {
            for x in 0..via_fn.width() {
                assert!(
                    (via_fn.pixel_at(x, y).0 - via_closure_fold.pixel_at(x, y).0).abs() < 1e-6,
                    "mismatch at ({}, {}): fn={}, closure_fold={}",
                    x,
                    y,
                    via_fn.pixel_at(x, y).0,
                    via_closure_fold.pixel_at(x, y).0,
                );
            }
        }
    }

    #[test]
    fn fold_fn_into_matches_fold_into_with_closure_fold() {
        // fold_neighborhood_fn_into must produce identical results to calling
        // fold_neighborhood_into with ClosureFold wrapping the same closure body.
        let src = make_4x4();
        let kernel = Neighborhood::<f32, 3, 3>::box_blur_3x3();
        let border = Clamp;
        let out_region = BorderPolicy::<Image<u8>>::output_region(
            &border,
            src.size(),
            kernel.weights().size(),
            kernel.anchor(),
        );

        let mut out_fn = Image::<MonoF32>::zero(out_region.size.width, out_region.size.height);
        fold_neighborhood_fn_into(
            &src,
            kernel.weights(),
            kernel.anchor(),
            &border,
            &mut out_fn,
            |neighbors| {
                let mut sum = 0.0f32;
                for item in neighbors {
                    sum += item.pixel as f32 * item.weight;
                }
                MonoF32(sum)
            },
        );

        let mut out_cf = Image::<MonoF32>::zero(out_region.size.width, out_region.size.height);
        fold_neighborhood_into(
            &src,
            kernel.weights(),
            kernel.anchor(),
            &border,
            &mut out_cf,
            ClosureFold(|neighbors: &mut dyn Iterator<Item = FoldItem<u8, f32>>| {
                let mut sum = 0.0f32;
                for item in neighbors {
                    sum += item.pixel as f32 * item.weight;
                }
                MonoF32(sum)
            }),
        );

        assert_eq!(out_fn.width(), out_cf.width());
        assert_eq!(out_fn.height(), out_cf.height());
        for y in 0..out_fn.height() {
            for x in 0..out_fn.width() {
                assert!(
                    (out_fn.pixel_at(x, y).0 - out_cf.pixel_at(x, y).0).abs() < 1e-6,
                    "mismatch at ({}, {})",
                    x,
                    y,
                );
            }
        }
    }

    // ── Panics ──────────────────────────────────────────────────────

    #[test]
    #[should_panic(expected = "too small")]
    fn fold_into_panics_on_undersized_output() {
        let src = make_4x4();
        let kernel = Neighborhood::<f32, 3, 3>::new([1.0; 9]);
        let mut out = Image::<MonoF32>::zero(2, 2); // Too small for 4×4 output

        fold_neighborhood_into(
            &src,
            kernel.weights(),
            kernel.anchor(),
            &Clamp,
            &mut out,
            ClosureFold(|_: &mut dyn Iterator<Item = FoldItem<u8, f32>>| MonoF32(0.0)),
        );
    }
}

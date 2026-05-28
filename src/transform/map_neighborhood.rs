//! Neighborhood map operations — data-dependent non-linear transforms.
//!
//! This module provides the counterpart to [`fold_neighborhood`]: where
//! `fold_neighborhood` works with fixed numeric weights (convolution, linear
//! filters), `map_neighborhood` works with **boolean topology** only, and passes
//! the **center pixel** as a first-class argument.
//!
//! # Choosing between `fold_neighborhood` and `map_neighborhood`
//!
//! | Property       | `fold_neighborhood`                  | `map_neighborhood`                       |
//! |----------------|--------------------------------------|------------------------------------------|
//! | Mask / kernel  | `Neighborhood<W, KW, KH>` + weights  | Any `ImageView<Pixel = bool>` (topology) |
//! | Weights        | Fixed, pre-set                       | Data-dependent, computed at call time    |
//! | Center pixel   | Just another `FoldItem` at the anchor| First-class `center` argument            |
//! | Neighbour item | `FoldItem { pixel, weight }`         | `MapItem { pixel, dx, dy }`              |
//! | Canonical uses | Convolution, blur, gradient filters  | Median, bilateral, Perona-Malik, NLM     |
//!
//! [`fold_neighborhood`]: crate::transform::fold::fold_neighborhood

use crate::border::{BorderPolicy, compute_interior_region};
use crate::image::{Image, ImageView, RasterImage, RasterImageMut};
use crate::pixel::ZeroablePixel;

// ─── MapItem ──────────────────────────────────────────────────────────────────

/// A single neighbour entry presented to [`MapOp::map`].
///
/// Carries the source pixel (border-resolved) and the spatial offset from
/// the anchor. There is no weight field: mask membership is implicit (only
/// positions where the mask is `true` are yielded), and any effective
/// weighting is computed from pixel data at runtime.
///
/// # Type parameters
///
/// - `P` — pixel type of the source image
///
/// # Example
///
/// ```
/// use fovea::transform::MapItem;
///
/// let item = MapItem { pixel: 42u8, dx: -1, dy: 0 };
/// assert_eq!(item.pixel, 42);
/// assert_eq!(item.dx, -1);
/// assert_eq!(item.dy, 0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MapItem<P> {
    /// Source pixel value (already border-resolved for boundary positions).
    pub pixel: P,
    /// Horizontal offset from the anchor (`< 0` = left, `> 0` = right).
    pub dx: isize,
    /// Vertical offset from the anchor (`< 0` = up, `> 0` = down).
    pub dy: isize,
}

// ─── MapOp trait ─────────────────────────────────────────────────────────────

/// A neighborhood map operation whose `map` method is generic over the
/// iterator type.
///
/// Because `map` is generic over `I`, Rust monomorphizes it separately for
/// the hot path (interior — direct `pixel_at`) and the cold path (boundary —
/// `border.pixel_at`). This eliminates the `dyn Iterator` vtable dispatch
/// that a plain closure would require, enabling full inlining and
/// auto-vectorization.
///
/// Unlike [`FoldOp`](crate::transform::FoldOp), `map` receives the **center pixel**
/// as a first-class argument. This is essential for data-dependent operations
/// where the effective weight for each neighbour is computed from both the
/// neighbour value and the center value (e.g. Perona-Malik, bilateral filter,
/// median).
///
/// # Semantics of `center` and `neighbors`
///
/// For every output position:
///
/// - `center` is the source pixel at the anchor coordinate.
/// - `neighbors` yields a [`MapItem`] for every `true` position in the mask,
///   **including the anchor** when `mask[anchor] == true` — it appears as
///   `MapItem { dx: 0, dy: 0, .. }`.
///
/// This means a full-rectangle mask with the anchor included will cause the
/// anchor pixel to appear in both `center` and the iterator. Operations that
/// use `center` as an initial accumulator (erosion, dilation) are unaffected
/// by the duplication because `min(x, x) == x` and `max(x, x) == x`.
/// Operations that collect all neighbour pixels (median) rely on the iterator
/// alone and should ignore `_center`.
///
/// # Implementing `MapOp`
///
/// ```
/// use fovea::transform::{MapOp, MapItem};
///
/// struct MinOp;
///
/// impl MapOp<u8> for MinOp {
///     type Accumulator = u8;
///     type Output = u8;
///
///     fn init(&self, center: u8) -> u8 { center }
///
///     #[inline(always)]
///     fn accumulate(&self, acc: &mut u8, item: MapItem<u8>) {
///         *acc = (*acc).min(item.pixel);
///     }
///
///     fn finalize(&mut self, acc: u8) -> u8 { acc }
/// }
/// ```
///
/// For convenience, closures can be wrapped in [`ClosureMap`] which falls
/// back to `dyn Iterator` dispatch internally.
pub trait MapOp<P> {
    /// The running accumulator type (often the same as `Output`).
    type Accumulator;

    /// The output pixel type produced by this map.
    type Output;

    /// Whether the engine should use the loop-inverted interior path.
    ///
    /// Default: `true`.  Set to `false` for operations that override
    /// [`map`](Self::map) directly (e.g. [`ClosureMap`], `MedianOp`).
    const INVERTIBLE: bool = true;

    /// Starting accumulator value, seeded with the center pixel.
    fn init(&self, center: P) -> Self::Accumulator;

    /// Absorb one neighbour into the accumulator.
    fn accumulate(&self, acc: &mut Self::Accumulator, item: MapItem<P>);

    /// Convert the final accumulator into the output pixel.
    fn finalize(&mut self, acc: Self::Accumulator) -> Self::Output;

    /// Process one pixel's full neighbourhood.
    ///
    /// The default implementation calls [`init`](Self::init) with the
    /// center pixel, then [`accumulate`](Self::accumulate) for every
    /// neighbour, then [`finalize`](Self::finalize).
    ///
    /// Override this only for operations that need all neighbours
    /// simultaneously (e.g. median) and set
    /// [`INVERTIBLE`](Self::INVERTIBLE) to `false`.
    fn map<I>(&mut self, center: P, neighbors: I) -> Self::Output
    where
        I: Iterator<Item = MapItem<P>>,
    {
        let mut acc = self.init(center);
        for item in neighbors {
            self.accumulate(&mut acc, item);
        }
        self.finalize(acc)
    }
}

// ─── ClosureMap ──────────────────────────────────────────────────────────────

/// Wrapper that lets a closure be used as a [`MapOp`].
///
/// The closure receives `(center: P, neighbors: &mut dyn Iterator<Item = MapItem<P>>)`.
/// Because the iterator is erased to `dyn`, there is **no performance improvement**
/// over a direct closure approach. Use this for quick prototyping and one-off
/// operations where the dispatch overhead is acceptable.
///
/// Internal operations (morphology, median) implement [`MapOp`] directly on
/// dedicated structs to get full monomorphization.
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView, Neighborhood};
/// use fovea::transform::{ClosureMap, MapItem, map_neighborhood};
/// use fovea::border::Clamp;
///
/// let src = Image::fill(5, 5, 128u8);
/// let mask = Neighborhood::<bool, 3, 3>::cross_3x3();
///
/// // Compute the minimum over the cross neighborhood. On a uniform image
/// // the minimum equals the center value everywhere.
/// let result = map_neighborhood(
///     &src,
///     mask.weights(),
///     mask.anchor(),
///     &Clamp,
///     ClosureMap(|center: u8, neighbors: &mut dyn Iterator<Item = MapItem<u8>>| {
///         neighbors.map(|n| n.pixel).fold(center, u8::min)
///     }),
/// );
///
/// assert_eq!(result.width(), 5);
/// assert_eq!(result.height(), 5);
/// for y in 0..result.height() {
///     for x in 0..result.width() {
///         assert_eq!(result.pixel_at(x, y), 128);
///     }
/// }
/// ```
pub struct ClosureMap<F>(pub F);

impl<P, Out, F> MapOp<P> for ClosureMap<F>
where
    F: FnMut(P, &mut dyn Iterator<Item = MapItem<P>>) -> Out,
{
    type Accumulator = ();
    type Output = Out;

    const INVERTIBLE: bool = false;

    #[inline(always)]
    fn init(&self, _center: P) -> () {}

    #[inline(always)]
    fn accumulate(&self, _acc: &mut (), _item: MapItem<P>) {}

    #[inline(always)]
    fn finalize(&mut self, _acc: ()) -> Out {
        unreachable!("ClosureMap uses map() override, not init/accumulate/finalize")
    }

    /// Direct pass-through to the wrapped closure.
    #[inline(always)]
    fn map<I>(&mut self, center: P, mut neighbors: I) -> Out
    where
        I: Iterator<Item = MapItem<P>>,
    {
        (self.0)(center, &mut neighbors)
    }
}

// ─── map_neighborhood_into ───────────────────────────────────────────────────

/// Write the result of mapping a neighborhood around every output pixel
/// into an existing output image.
///
/// This is the **base method** — [`map_neighborhood`] is the convenience
/// wrapper that allocates the output for you.
///
/// # Parameters
///
/// - `src` — source image.
/// - `mask_weights` — boolean mask encoding neighborhood topology.
///   Only `true` positions are visited; `false` positions are silently skipped.
/// - `anchor` — position within the mask that corresponds to the center pixel,
///   given as `(kx, ky)`.
/// - `border` — policy for handling out-of-bounds neighbour coordinates.
/// - `output` — destination image, written in-place.
/// - `op` — the [`MapOp`] to invoke for each output pixel.
///
/// # Algorithm
///
/// For every pixel in `border.output_region(…)`:
///
/// 1. Pre-collect all mask-`true` positions `(dx, dy)` relative to the anchor.
/// 2. For each output position `(cx, cy)`: fetch `center = src[cx, cy]`, build
///    an iterator of [`MapItem`]s from the mask-`true` offsets, call
///    `op.map(center, iter)`, and write the result.
///
/// # Interior / boundary split
///
/// Same strategy as [`fold_neighborhood_into`](crate::transform::fold_neighborhood_into):
/// interior positions use direct `pixel_at` access (hot path, no border calls);
/// only the thin boundary strip calls `border.pixel_at()` (cold path).
///
/// # Panics
///
/// Panics if `output` is smaller than the region returned by
/// `border.output_region()`.
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView, ImageViewMut, Neighborhood};
/// use fovea::transform::{MapOp, MapItem, map_neighborhood_into};
/// use fovea::border::{BorderPolicy, Clamp};
///
/// struct MinOp;
/// impl MapOp<u8> for MinOp {
///     type Accumulator = u8;
///     type Output = u8;
///     fn init(&self, center: u8) -> u8 { center }
///     fn accumulate(&self, acc: &mut u8, item: MapItem<u8>) {
///         *acc = (*acc).min(item.pixel);
///     }
///     fn finalize(&mut self, acc: u8) -> u8 { acc }
/// }
///
/// let src = Image::fill(5, 5, 10u8);
/// let mask = Neighborhood::<bool, 3, 3>::full_rect_3x3();
/// let anchor = mask.anchor();
/// let weights = mask.weights();
///
/// let border = Clamp;
/// let out_region = BorderPolicy::<Image<u8>>::output_region(
///     &border, src.size(), weights.size(), anchor,
/// );
/// let mut out = Image::<u8>::zero(out_region.size.width, out_region.size.height);
///
/// map_neighborhood_into(&src, weights, anchor, &border, &mut out, MinOp);
///
/// // Uniform image: min equals center value everywhere.
/// for y in 0..out.height() {
///     for x in 0..out.width() {
///         assert_eq!(out.pixel_at(x, y), 10);
///     }
/// }
/// ```
pub fn map_neighborhood_into<I, MI, B, O, M, P>(
    src: &I,
    mask_weights: &MI,
    anchor: (usize, usize),
    border: &B,
    output: &mut O,
    mut op: M,
) where
    I: RasterImage<Pixel = P>,
    P: Copy,
    MI: ImageView<Pixel = bool>,
    B: BorderPolicy<I>,
    O: RasterImageMut<Pixel = M::Output>,
    M: MapOp<P>,
    M::Output: Copy,
{
    let mask_size = mask_weights.size();
    let output_region = border.output_region(src.size(), mask_size, anchor);
    let interior = compute_interior_region(src.size(), mask_size, anchor);

    assert!(
        output.width() >= output_region.size.width && output.height() >= output_region.size.height,
        "output image {}×{} is too small for the output region {}×{}",
        output.width(),
        output.height(),
        output_region.size.width,
        output_region.size.height,
    );

    // Pre-collect mask-true positions as (dx, dy) offsets relative to the
    // anchor.  Only these positions are yielded to `op.map`; false entries
    // are silently skipped.  For typical mask sizes (3×3 .. 7×7) this Vec
    // stays in L1.
    let mask_positions: Vec<(isize, isize)> = {
        let mut positions = Vec::with_capacity(mask_size.width * mask_size.height);
        for ky in 0..mask_size.height {
            for kx in 0..mask_size.width {
                if mask_weights.pixel_at(kx, ky) {
                    let dx = kx as isize - anchor.0 as isize;
                    let dy = ky as isize - anchor.1 as isize;
                    positions.push((dx, dy));
                }
            }
        }
        positions
    };

    let ox = output_region.left();
    let oy = output_region.top();

    // ── HOT PATH — interior positions ────────────────────────────────────────
    //
    // For every interior position every (cx + dx, cy + dy) offset is
    // guaranteed to be within the image bounds, so we access pixels directly
    // without calling the border policy.
    if let Some(interior) = interior {
        let int_left = interior.left().max(ox);
        let int_top = interior.top().max(oy);
        let int_right = interior.right().min(output_region.right());
        let int_bottom = interior.bottom().min(output_region.bottom());

        let int_width = int_right.saturating_sub(int_left);

        if int_width > 0 && int_top < int_bottom {
            if M::INVERTIBLE {
                // ── Loop-inverted path (SIMD-friendly) ───────────────
                //
                // Kernel positions are outer, pixel scan is inner.
                // The inner loop is a contiguous elementwise operation
                // that LLVM auto-vectorizes (vpminub, vpmaxub, …).

                // Allocate accumulator row once, reused across all cy.
                let first_center_row = src.row(int_top);
                let mut acc_row: Vec<M::Accumulator> = (int_left..int_right)
                    .map(|cx| op.init(first_center_row[cx]))
                    .collect();

                for cy in int_top..int_bottom {
                    // Re-init accumulators from this row's center pixels
                    let center_row = src.row(cy);
                    if cy > int_top {
                        let center_slice = &center_row[int_left..int_right];
                        for i in 0..int_width {
                            acc_row[i] = op.init(center_slice[i]);
                        }
                    }

                    // Kernel-outer sweep
                    for &(dx, dy) in &mask_positions {
                        let src_row = src.row((cy as isize + dy) as usize);
                        let start = (int_left as isize + dx) as usize;
                        let src_slice = &src_row[start..start + int_width];

                        for i in 0..int_width {
                            op.accumulate(
                                &mut acc_row[i],
                                MapItem {
                                    pixel: src_slice[i],
                                    dx,
                                    dy,
                                },
                            );
                        }
                    }

                    // Finalize & write
                    let out_row = &mut output.row_mut(cy - oy)[int_left - ox..int_right - ox];
                    for i in 0..int_width {
                        let new_init = op.init(center_row[int_left + i]);
                        let acc = std::mem::replace(&mut acc_row[i], new_init);
                        out_row[i] = op.finalize(acc);
                    }
                }
            } else {
                // ── Per-pixel fallback (non-invertible ops) ──────────
                for cy in int_top..int_bottom {
                    for cx in int_left..int_right {
                        let center = src.pixel_at(cx, cy);
                        let iter = mask_positions.iter().map(|&(dx, dy)| {
                            let sx = (cx as isize + dx) as usize;
                            let sy = (cy as isize + dy) as usize;
                            MapItem {
                                pixel: src.pixel_at(sx, sy),
                                dx,
                                dy,
                            }
                        });
                        let result = op.map(center, iter);
                        *output.pixel_at_mut(cx - ox, cy - oy) = result;
                    }
                }
            }
        }
    }

    // ── COLD PATH — boundary positions ───────────────────────────────────────
    //
    // Iterate all output positions not covered by the hot path.  For `Skip`
    // the output region equals the interior, so this loop has zero actual
    // iterations.  For all other policies (cx, cy) are always valid image
    // coordinates (output region == full image size, ox == oy == 0).
    for cy in output_region.top()..output_region.bottom() {
        for cx in output_region.left()..output_region.right() {
            // Skip positions already handled by the hot path.
            if let Some(ref interior) = interior {
                if cx >= interior.left()
                    && cx < interior.right()
                    && cy >= interior.top()
                    && cy < interior.bottom()
                {
                    continue;
                }
            }

            // The center is the source pixel at the output coordinate.
            // For non-Skip policies the output region covers the full image,
            // so (cx, cy) is always a valid image coordinate here.
            let center = src.pixel_at(cx, cy);
            let iter = mask_positions.iter().map(|&(dx, dy)| {
                let pixel = border.pixel_at(src, cx as isize + dx, cy as isize + dy);
                MapItem { pixel, dx, dy }
            });
            let result = op.map(center, iter);
            *output.pixel_at_mut(cx - ox, cy - oy) = result;
        }
    }
}

// ─── map_neighborhood ────────────────────────────────────────────────────────

/// Map a neighborhood operation over every output pixel and return a newly
/// allocated [`Image`] with the results.
///
/// This is a convenience wrapper around [`map_neighborhood_into`].
/// It allocates an output `Image<M::Output>` of the correct size (determined
/// by `border.output_region(…)`) and fills it via the base method.
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView, Neighborhood};
/// use fovea::transform::{MapOp, MapItem, map_neighborhood};
/// use fovea::border::Skip;
///
/// struct Identity;
/// impl MapOp<u8> for Identity {
///     type Accumulator = u8;
///     type Output = u8;
///     fn init(&self, center: u8) -> u8 { center }
///     fn accumulate(&self, _acc: &mut u8, _item: MapItem<u8>) {}
///     fn finalize(&mut self, acc: u8) -> u8 { acc }
/// }
///
/// let src = Image::generate(5, 5, |x, y| (x + y * 5) as u8);
/// let mask = Neighborhood::<bool, 3, 3>::new([
///     false, false, false,
///     false, true,  false,
///     false, false, false,
/// ]);
///
/// // Skip: output is the 3×3 interior.
/// let result = map_neighborhood(&src, mask.weights(), mask.anchor(), &Skip, Identity);
///
/// assert_eq!(result.width(), 3);
/// assert_eq!(result.height(), 3);
/// for y in 0..3 {
///     for x in 0..3 {
///         assert_eq!(result.pixel_at(x, y), src.pixel_at(x + 1, y + 1));
///     }
/// }
/// ```
#[must_use]
pub fn map_neighborhood<I, MI, B, M, P>(
    src: &I,
    mask_weights: &MI,
    anchor: (usize, usize),
    border: &B,
    op: M,
) -> Image<M::Output>
where
    I: RasterImage<Pixel = P>,
    P: Copy,
    MI: ImageView<Pixel = bool>,
    B: BorderPolicy<I>,
    M: MapOp<P>,
    M::Output: Copy + ZeroablePixel,
{
    let output_region = border.output_region(src.size(), mask_weights.size(), anchor);
    let mut out = Image::<M::Output>::zero(output_region.size.width, output_region.size.height);
    map_neighborhood_into(src, mask_weights, anchor, border, &mut out, op);
    out
}

// ─── Closure-based convenience wrappers ──────────────────────────────────────

/// Convenience wrapper: [`map_neighborhood_into`] accepting a closure.
///
/// Wraps the closure in [`ClosureMap`], which uses `dyn Iterator` dispatch
/// internally. For maximum performance, implement [`MapOp`] directly instead.
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView, ImageViewMut, Neighborhood};
/// use fovea::transform::MapItem;
/// use fovea::border::{BorderPolicy, Clamp};
/// use fovea::transform::map_neighborhood_fn_into;
///
/// let src = Image::fill(5, 5, 7u8);
/// let mask = Neighborhood::<bool, 3, 3>::full_rect_3x3();
/// let anchor = mask.anchor();
/// let weights = mask.weights();
///
/// let border = Clamp;
/// let out_region = BorderPolicy::<Image<u8>>::output_region(
///     &border, src.size(), weights.size(), anchor,
/// );
/// let mut out = Image::<u8>::zero(out_region.size.width, out_region.size.height);
///
/// map_neighborhood_fn_into(
///     &src, weights, anchor, &border, &mut out,
///     |center: u8, _neighbors: &mut dyn Iterator<Item = MapItem<u8>>| center,
/// );
///
/// for y in 0..out.height() {
///     for x in 0..out.width() {
///         assert_eq!(out.pixel_at(x, y), 7);
///     }
/// }
/// ```
pub fn map_neighborhood_fn_into<I, MI, B, O, F, P, Out>(
    src: &I,
    mask_weights: &MI,
    anchor: (usize, usize),
    border: &B,
    output: &mut O,
    f: F,
) where
    I: RasterImage<Pixel = P>,
    P: Copy,
    MI: ImageView<Pixel = bool>,
    B: BorderPolicy<I>,
    O: RasterImageMut<Pixel = Out>,
    F: FnMut(P, &mut dyn Iterator<Item = MapItem<P>>) -> Out,
    Out: Copy,
{
    map_neighborhood_into(src, mask_weights, anchor, border, output, ClosureMap(f));
}

/// Convenience wrapper: [`map_neighborhood`] accepting a closure.
///
/// Wraps the closure in [`ClosureMap`], which uses `dyn Iterator` dispatch
/// internally. For maximum performance, implement [`MapOp`] directly instead.
///
/// # Contrast with `fold_neighborhood_fn`
///
/// Use [`fold_neighborhood_fn`](crate::transform::fold_neighborhood_fn) when you have
/// **fixed numeric weights** (convolution, box blur, edge detection). Use
/// `map_neighborhood_fn` when the operation is **data-dependent**: the
/// contribution of each neighbour is a function of pixel *values*, not
/// pre-set coefficients.
///
/// # Example — Perona-Malik anisotropic diffusion (one step)
///
/// ```
/// use fovea::image::{Image, ImageView, Neighborhood};
/// use fovea::transform::{map_neighborhood_fn, MapItem};
/// use fovea::border::Clamp;
///
/// let src = Image::fill(5, 5, 64u8);
/// // Cross-shaped neighborhood: N, W, center, E, S.
/// let mask = Neighborhood::<bool, 3, 3>::cross_3x3();
///
/// // On a uniform image every gradient is 0, so the update is 0 and the
/// // image is unchanged after one diffusion step.
/// let lambda = 0.125f32;
/// let k = 10.0f32;
/// let result = map_neighborhood_fn(
///     &src,
///     mask.weights(),
///     mask.anchor(),
///     &Clamp,
///     |center: u8, neighbors: &mut dyn Iterator<Item = MapItem<u8>>| {
///         let update: f32 = neighbors
///             .map(|n| {
///                 let diff = n.pixel as f32 - center as f32;
///                 f32::exp(-(diff * diff) / (k * k)) * diff
///             })
///             .sum();
///         (center as f32 + lambda * update).clamp(0.0, 255.0) as u8
///     },
/// );
///
/// assert_eq!(result.width(), 5);
/// assert_eq!(result.height(), 5);
/// for y in 0..result.height() {
///     for x in 0..result.width() {
///         assert_eq!(result.pixel_at(x, y), 64);
///     }
/// }
/// ```
#[must_use]
pub fn map_neighborhood_fn<I, MI, B, F, P, Out>(
    src: &I,
    mask_weights: &MI,
    anchor: (usize, usize),
    border: &B,
    f: F,
) -> Image<Out>
where
    I: RasterImage<Pixel = P>,
    P: Copy,
    MI: ImageView<Pixel = bool>,
    B: BorderPolicy<I>,
    Out: Copy + ZeroablePixel,
    F: FnMut(P, &mut dyn Iterator<Item = MapItem<P>>) -> Out,
{
    map_neighborhood(src, mask_weights, anchor, border, ClosureMap(f))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::border::{Clamp, Constant, Mirror, Skip, Wrap};
    use crate::image::{Image, ImageView, Neighborhood};

    // ── MapItem basic tests ──────────────────────────────────────────────

    #[test]
    fn map_item_fields() {
        let item = MapItem {
            pixel: 42u8,
            dx: -1,
            dy: 2,
        };
        assert_eq!(item.pixel, 42);
        assert_eq!(item.dx, -1);
        assert_eq!(item.dy, 2);
    }

    #[test]
    fn map_item_is_copy() {
        let item = MapItem {
            pixel: 7u8,
            dx: 0,
            dy: 0,
        };
        let item2 = item; // Copy
        assert_eq!(item, item2);
    }

    #[test]
    fn map_item_debug() {
        let item = MapItem {
            pixel: 0u8,
            dx: 0,
            dy: 0,
        };
        let dbg = format!("{:?}", item);
        assert!(dbg.contains("MapItem"));
    }

    // ── Helper MapOp implementations ─────────────────────────────────────────

    /// Returns the center pixel unchanged; ignores all neighbours.
    struct IdentityMap;
    impl MapOp<u8> for IdentityMap {
        type Accumulator = u8;
        type Output = u8;
        #[inline(always)]
        fn init(&self, center: u8) -> u8 {
            center
        }
        #[inline(always)]
        fn accumulate(&self, _acc: &mut u8, _item: MapItem<u8>) {}
        #[inline(always)]
        fn finalize(&mut self, acc: u8) -> u8 {
            acc
        }
    }

    /// Returns the number of neighbours yielded by the iterator
    /// (including the anchor when mask[anchor] == true).
    struct CountNeighbors;
    impl MapOp<u8> for CountNeighbors {
        type Accumulator = u8;
        type Output = u8;
        fn init(&self, _center: u8) -> u8 {
            0
        }
        fn accumulate(&self, acc: &mut u8, _item: MapItem<u8>) {
            *acc += 1;
        }
        fn finalize(&mut self, acc: u8) -> u8 {
            acc
        }
    }

    /// Returns the minimum over center + all neighbours.
    struct MinMap;
    impl MapOp<u8> for MinMap {
        type Accumulator = u8;
        type Output = u8;
        #[inline(always)]
        fn init(&self, center: u8) -> u8 {
            center
        }
        #[inline(always)]
        fn accumulate(&self, acc: &mut u8, item: MapItem<u8>) {
            *acc = (*acc).min(item.pixel);
        }
        #[inline(always)]
        fn finalize(&mut self, acc: u8) -> u8 {
            acc
        }
    }

    // ── Helper images ─────────────────────────────────────────────────────────

    fn make_5x5_gradient() -> Image<u8> {
        // pixel(x, y) = x + y * 5, giving values 0..24.
        Image::generate(5, 5, |x, y| (x + y * 5) as u8)
    }

    fn make_4x4() -> Image<u8> {
        Image::generate(4, 4, |x, y| (x + y * 4) as u8)
    }

    // ── Identity: Skip policy ─────────────────────────────────────────────────

    #[test]
    fn identity_3x3_skip_preserves_interior() {
        let src = make_5x5_gradient();
        let mask = Neighborhood::<bool, 3, 3>::full_rect_3x3();

        let result: Image<u8> =
            map_neighborhood(&src, mask.weights(), mask.anchor(), &Skip, IdentityMap);

        // Skip: interior of a 5×5 image with a 3×3 kernel is 3×3.
        assert_eq!(result.width(), 3);
        assert_eq!(result.height(), 3);
        for y in 0..3 {
            for x in 0..3 {
                assert_eq!(
                    result.pixel_at(x, y),
                    src.pixel_at(x + 1, y + 1),
                    "mismatch at output ({x},{y})"
                );
            }
        }
    }

    // ── Identity: Clamp policy ────────────────────────────────────────────────

    #[test]
    fn identity_3x3_clamp_preserves_all() {
        let src = make_5x5_gradient();
        let mask = Neighborhood::<bool, 3, 3>::full_rect_3x3();

        let result: Image<u8> =
            map_neighborhood(&src, mask.weights(), mask.anchor(), &Clamp, IdentityMap);

        assert_eq!(result.width(), 5);
        assert_eq!(result.height(), 5);
        for y in 0..5 {
            for x in 0..5 {
                assert_eq!(result.pixel_at(x, y), src.pixel_at(x, y));
            }
        }
    }

    // ── Identity: all five policies agree on interior pixels ─────────────────

    #[test]
    fn all_border_policies_identity_agree_on_interior() {
        let src = make_5x5_gradient();
        let mask = Neighborhood::<bool, 3, 3>::full_rect_3x3();

        let r_clamp = map_neighborhood(&src, mask.weights(), mask.anchor(), &Clamp, IdentityMap);
        let r_mirror = map_neighborhood(&src, mask.weights(), mask.anchor(), &Mirror, IdentityMap);
        let r_wrap = map_neighborhood(&src, mask.weights(), mask.anchor(), &Wrap, IdentityMap);
        let r_const = map_neighborhood(
            &src,
            mask.weights(),
            mask.anchor(),
            &Constant(0u8),
            IdentityMap,
        );

        // Non-Skip policies produce output the same size as the source.
        assert_eq!(r_clamp.width(), 5);
        assert_eq!(r_mirror.width(), 5);
        assert_eq!(r_wrap.width(), 5);
        assert_eq!(r_const.width(), 5);

        // All agree on interior pixels (identity ignores neighbours).
        for y in 1..4 {
            for x in 1..4 {
                let expected = src.pixel_at(x, y);
                assert_eq!(r_clamp.pixel_at(x, y), expected);
                assert_eq!(r_mirror.pixel_at(x, y), expected);
                assert_eq!(r_wrap.pixel_at(x, y), expected);
                assert_eq!(r_const.pixel_at(x, y), expected);
            }
        }
    }

    // ── Neighbour count: full-rect 3×3 mask yields 9 per pixel ───────────────

    #[test]
    fn full_rect_mask_yields_9_neighbors_per_pixel() {
        let src = Image::fill(5, 5, 1u8);
        let mask = Neighborhood::<bool, 3, 3>::full_rect_3x3();

        let result = map_neighborhood(&src, mask.weights(), mask.anchor(), &Clamp, CountNeighbors);

        for y in 0..5 {
            for x in 0..5 {
                assert_eq!(result.pixel_at(x, y), 9, "expected 9 at ({x},{y})");
            }
        }
    }

    // ── Neighbour count: cross mask (center + 4 cardinal) = 5 ────────────────

    #[test]
    fn cross_mask_yields_5_neighbors_for_interior_pixel() {
        let src = Image::fill(5, 5, 1u8);
        let mask = Neighborhood::<bool, 3, 3>::cross_3x3();

        let result = map_neighborhood(&src, mask.weights(), mask.anchor(), &Clamp, CountNeighbors);

        // Interior pixel: all 5 mask positions are in-bounds.
        assert_eq!(result.pixel_at(2, 2), 5);
    }

    // ── Neighbour count: anchor-only mask = 1 per pixel ──────────────────────

    #[test]
    fn anchor_only_mask_yields_1_neighbor_per_pixel() {
        #[rustfmt::skip]
        let mask = Neighborhood::<bool, 3, 3>::new([
            false, false, false,
            false, true,  false,
            false, false, false,
        ]);
        let src = Image::fill(5, 5, 1u8);

        let result = map_neighborhood(&src, mask.weights(), mask.anchor(), &Clamp, CountNeighbors);

        for y in 0..5 {
            for x in 0..5 {
                assert_eq!(result.pixel_at(x, y), 1);
            }
        }
    }

    // ── Anchor excluded from iterator when mask[anchor] = false ──────────────

    #[test]
    fn anchor_excluded_from_iterator_when_mask_anchor_false() {
        // Cross without center: 4 cardinal directions, no anchor.
        #[rustfmt::skip]
        let mask = Neighborhood::<bool, 3, 3>::new([
            false, true,  false,
            true,  false, true,
            false, true,  false,
        ]);
        let src = Image::fill(5, 5, 0u8);

        let result = map_neighborhood(&src, mask.weights(), mask.anchor(), &Clamp, CountNeighbors);

        // Interior pixel: exactly 4 neighbours (no anchor).
        assert_eq!(result.pixel_at(2, 2), 4);
    }

    // ── Pixel access: single off-center neighbour ─────────────────────────────

    #[test]
    fn right_neighbor_only_mask_reads_correct_source_pixel() {
        let src = make_5x5_gradient();
        #[rustfmt::skip]
        let mask = Neighborhood::<bool, 3, 3>::new([
            false, false, false,
            false, false, true,   // dx=1, dy=0
            false, false, false,
        ]);

        // Skip: output is 3×3, interior x in [1..4], y in [1..4].
        let result = map_neighborhood_fn(
            &src,
            mask.weights(),
            mask.anchor(),
            &Skip,
            |_center: u8, neighbors: &mut dyn Iterator<Item = MapItem<u8>>| {
                neighbors.next().map(|n| n.pixel).unwrap_or(0)
            },
        );

        assert_eq!(result.width(), 3);
        assert_eq!(result.height(), 3);
        for oy in 0..3 {
            for ox in 0..3 {
                // Interior pixel at image coords (ox+1, oy+1).
                // Right neighbour is at image coords (ox+2, oy+1).
                assert_eq!(
                    result.pixel_at(ox, oy),
                    src.pixel_at(ox + 2, oy + 1),
                    "mismatch at output ({ox},{oy})"
                );
            }
        }
    }

    // ── dx / dy field values ──────────────────────────────────────────────────

    #[test]
    fn bottom_right_neighbor_has_correct_dx_dy() {
        #[rustfmt::skip]
        let mask = Neighborhood::<bool, 3, 3>::new([
            false, false, false,
            false, false, false,
            false, false, true,   // dx=1, dy=1
        ]);
        let src = Image::fill(5, 5, 0u8);

        let mut last_dx = 99isize;
        let mut last_dy = 99isize;

        let _ = map_neighborhood_fn(
            &src,
            mask.weights(),
            mask.anchor(),
            &Skip,
            |_center: u8, neighbors: &mut dyn Iterator<Item = MapItem<u8>>| {
                for n in neighbors {
                    last_dx = n.dx;
                    last_dy = n.dy;
                }
                0u8
            },
        );

        assert_eq!(last_dx, 1, "expected dx = 1");
        assert_eq!(last_dy, 1, "expected dy = 1");
    }

    #[test]
    fn top_left_neighbor_has_correct_dx_dy() {
        #[rustfmt::skip]
        let mask = Neighborhood::<bool, 3, 3>::new([
            true,  false, false,  // dx=-1, dy=-1
            false, false, false,
            false, false, false,
        ]);
        let src = Image::fill(5, 5, 0u8);

        let mut last_dx = 99isize;
        let mut last_dy = 99isize;

        let _ = map_neighborhood_fn(
            &src,
            mask.weights(),
            mask.anchor(),
            &Skip,
            |_center: u8, neighbors: &mut dyn Iterator<Item = MapItem<u8>>| {
                for n in neighbors {
                    last_dx = n.dx;
                    last_dy = n.dy;
                }
                0u8
            },
        );

        assert_eq!(last_dx, -1, "expected dx = -1");
        assert_eq!(last_dy, -1, "expected dy = -1");
    }

    // ── map_into matches map ──────────────────────────────────────────────────

    #[test]
    fn map_into_produces_same_result_as_map() {
        let src = make_4x4();
        let mask = Neighborhood::<bool, 3, 3>::full_rect_3x3();

        let alloc = map_neighborhood(&src, mask.weights(), mask.anchor(), &Clamp, IdentityMap);

        let mut into_result = Image::zero(4, 4);
        map_neighborhood_into(
            &src,
            mask.weights(),
            mask.anchor(),
            &Clamp,
            &mut into_result,
            IdentityMap,
        );

        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(
                    alloc.pixel_at(x, y),
                    into_result.pixel_at(x, y),
                    "mismatch at ({x},{y})"
                );
            }
        }
    }

    // ── map_neighborhood_fn_into matches map_neighborhood_fn ─────────────────

    #[test]
    fn map_fn_into_produces_same_result_as_map_fn() {
        let src = make_4x4();
        let mask = Neighborhood::<bool, 3, 3>::full_rect_3x3();

        let alloc = map_neighborhood_fn(
            &src,
            mask.weights(),
            mask.anchor(),
            &Clamp,
            |center: u8, _: &mut dyn Iterator<Item = MapItem<u8>>| center,
        );

        let mut into_result = Image::zero(4, 4);
        map_neighborhood_fn_into(
            &src,
            mask.weights(),
            mask.anchor(),
            &Clamp,
            &mut into_result,
            |center: u8, _: &mut dyn Iterator<Item = MapItem<u8>>| center,
        );

        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(alloc.pixel_at(x, y), into_result.pixel_at(x, y));
            }
        }
    }

    // ── ClosureMap produces same result as dedicated struct ───────────────────

    #[test]
    fn closure_map_produces_same_result_as_struct() {
        let src = make_5x5_gradient();
        let mask = Neighborhood::<bool, 3, 3>::full_rect_3x3();

        let struct_result = map_neighborhood(&src, mask.weights(), mask.anchor(), &Clamp, MinMap);

        let closure_result = map_neighborhood_fn(
            &src,
            mask.weights(),
            mask.anchor(),
            &Clamp,
            |center: u8, neighbors: &mut dyn Iterator<Item = MapItem<u8>>| {
                neighbors.map(|n| n.pixel).fold(center, u8::min)
            },
        );

        for y in 0..5 {
            for x in 0..5 {
                assert_eq!(
                    struct_result.pixel_at(x, y),
                    closure_result.pixel_at(x, y),
                    "mismatch at ({x},{y})"
                );
            }
        }
    }

    // ── Min filter: known values ──────────────────────────────────────────────

    #[test]
    fn min_filter_3x3_known_values() {
        // make_5x5_gradient: pixel(x, y) = x + y*5
        // 0  1  2  3  4
        // 5  6  7  8  9
        // 10 11 12 13 14
        // 15 16 17 18 19
        // 20 21 22 23 24
        let src = make_5x5_gradient();
        let mask = Neighborhood::<bool, 3, 3>::full_rect_3x3();

        let result = map_neighborhood(&src, mask.weights(), mask.anchor(), &Skip, MinMap);

        // Skip output is 3×3; output(ox, oy) = min over 3×3 window centered
        // at source(ox+1, oy+1).
        //
        // output(0,0) = min of window at src(1,1) = min{0,1,2,5,6,7,10,11,12} = 0
        assert_eq!(result.pixel_at(0, 0), 0);
        // output(1,1) = min of window at src(2,2) = min{6,7,8,11,12,13,16,17,18} = 6
        assert_eq!(result.pixel_at(1, 1), 6);
        // output(2,2) = min of window at src(3,3) = min{12,13,14,17,18,19,22,23,24} = 12
        assert_eq!(result.pixel_at(2, 2), 12);
    }

    // ── 1×1 image edge cases ──────────────────────────────────────────────────

    #[test]
    fn single_pixel_image_clamp() {
        let src = Image::fill(1, 1, 42u8);
        let mask = Neighborhood::<bool, 3, 3>::full_rect_3x3();

        let result = map_neighborhood(&src, mask.weights(), mask.anchor(), &Clamp, IdentityMap);

        assert_eq!(result.width(), 1);
        assert_eq!(result.height(), 1);
        assert_eq!(result.pixel_at(0, 0), 42);
    }

    #[test]
    fn single_pixel_image_constant_border_reduces_to_min() {
        // 1×1 image with value 10; Constant(0) pads with 0.
        // Full-rect 3×3: anchor (value 10) + 8 border pixels (value 0).
        // MinMap: fold(10, [0,0,0,0,10,0,0,0,0]) = 0.
        let src = Image::fill(1, 1, 10u8);
        let mask = Neighborhood::<bool, 3, 3>::full_rect_3x3();

        let result = map_neighborhood(&src, mask.weights(), mask.anchor(), &Constant(0u8), MinMap);

        assert_eq!(result.width(), 1);
        assert_eq!(result.height(), 1);
        assert_eq!(result.pixel_at(0, 0), 0);
    }

    // ── Kernel larger than image with Skip ────────────────────────────────────

    #[test]
    fn kernel_larger_than_image_skip_produces_empty_output() {
        let src = Image::fill(2, 2, 0u8);
        let mask = Neighborhood::<bool, 5, 5>::full_rect_5x5();

        let result: Image<u8> =
            map_neighborhood(&src, mask.weights(), mask.anchor(), &Skip, IdentityMap);

        assert_eq!(result.width(), 0);
        assert_eq!(result.height(), 0);
    }

    // ── Undersized output panics ──────────────────────────────────────────────

    #[test]
    #[should_panic(expected = "too small")]
    fn map_into_panics_on_undersized_output() {
        let src = Image::fill(5, 5, 0u8);
        let mask = Neighborhood::<bool, 3, 3>::full_rect_3x3();
        // Clamp requires 5×5 output; we provide 3×3.
        let mut out = Image::<u8>::zero(3, 3);
        map_neighborhood_into(
            &src,
            mask.weights(),
            mask.anchor(),
            &Clamp,
            &mut out,
            IdentityMap,
        );
    }

    // ── f32 pixel type ────────────────────────────────────────────────────────

    #[test]
    fn works_with_f32_pixels() {
        use crate::pixel::MonoF32;
        let src = Image::fill(5, 5, MonoF32::new(1.0));
        #[rustfmt::skip]
        let mask = Neighborhood::<bool, 3, 3>::new([
            false, false, false,
            false, true,  false,
            false, false, false,
        ]);

        struct IdentityF32;
        impl MapOp<MonoF32> for IdentityF32 {
            type Accumulator = MonoF32;
            type Output = MonoF32;
            fn init(&self, center: MonoF32) -> MonoF32 {
                center
            }
            fn accumulate(&self, _acc: &mut MonoF32, _item: MapItem<MonoF32>) {}
            fn finalize(&mut self, acc: MonoF32) -> MonoF32 {
                acc
            }
        }

        let result = map_neighborhood(&src, mask.weights(), mask.anchor(), &Clamp, IdentityF32);

        for y in 0..5 {
            for x in 0..5 {
                assert!((result.pixel_at(x, y).0 - 1.0f32).abs() < 1e-6);
            }
        }
    }

    // ── Large image stress test ───────────────────────────────────────────────

    #[test]
    fn large_image_runs_without_error() {
        let src = Image::generate(100, 100, |x, y| ((x + y * 3) % 256) as u8);
        let mask = Neighborhood::<bool, 3, 3>::full_rect_3x3();

        let result: Image<u8> =
            map_neighborhood(&src, mask.weights(), mask.anchor(), &Clamp, IdentityMap);

        assert_eq!(result.width(), 100);
        assert_eq!(result.height(), 100);
    }
}

//! Morphological operations built on [`map_neighborhood`].
//!
//! Morphological operations use a binary structuring element
//! ([`Neighborhood<bool, …>`]) and compute local min/max (or custom
//! aggregations) over the "true" positions of that element.
//!
//! This module provides:
//!
//! - [`erode_into`] / [`erode`] — local minimum (shrinks bright regions)
//! - [`dilate_into`] / [`dilate`] — local maximum (grows bright regions)
//! - [`opening`] — erosion followed by dilation (removes small bright spots)
//! - [`closing`] — dilation followed by erosion (fills small dark holes)
//! - [`morphological_gradient`] — dilation − erosion (edge detector)
//! - [`top_hat`] — original − opening (isolates bright peaks)
//! - [`black_hat`] — closing − original (isolates dark valleys)
//! - [`median_filter`] — local median (non-linear noise removal)
//!
//! All operations require `P: Copy + Ord` so that `min` / `max` / sorting are
//! well-defined. For floating-point images, convert to an orderable
//! representation first (or implement [`MapOp`] directly with a custom
//! comparator).

use crate::border::{BorderPolicy, FullFrameBorder};
use crate::image::Kernel;
use crate::image::{Image, ImageView, RasterImage, RasterImageMut};
use crate::pixel::ZeroablePixel;
use crate::transform::combine::{PixelSubtract, combine_images};
use crate::transform::map_neighborhood::MapItem;
use crate::transform::map_neighborhood::{MapOp, map_neighborhood, map_neighborhood_into};

// ─── MapOp implementations ───────────────────────────────────────────────────

/// A [`MapOp`] that computes the local minimum over the neighborhood
/// (erosion).
///
/// `center` is used as the initial accumulator so that the result is always
/// ≤ every neighbour, including the anchor pixel itself when
/// `mask[anchor] == true`. For standard structuring elements (where the
/// anchor is always included) this is equivalent to `min` over all active
/// positions.
///
/// Fully monomorphized — no `dyn Iterator` vtable dispatch.
/// Shared between [`erode`] and [`erode_into`].
pub(crate) struct ErodeOp;

impl<P: Copy + Ord> MapOp<P> for ErodeOp {
    type Accumulator = P;
    type Output = P;

    #[inline(always)]
    fn init(&self, center: P) -> P {
        center
    }

    #[inline(always)]
    fn accumulate(&self, acc: &mut P, item: MapItem<P>) {
        *acc = (*acc).min(item.pixel);
    }

    #[inline(always)]
    fn finalize(&mut self, acc: P) -> P {
        acc
    }
}

/// A [`MapOp`] that computes the local maximum over the neighborhood
/// (dilation).
///
/// `center` is used as the initial accumulator so that the result is always
/// ≥ every neighbour, including the anchor pixel itself when
/// `mask[anchor] == true`. For standard structuring elements (where the
/// anchor is always included) this is equivalent to `max` over all active
/// positions.
///
/// Fully monomorphized — no `dyn Iterator` vtable dispatch.
/// Shared between [`dilate`] and [`dilate_into`].
pub(crate) struct DilateOp;

impl<P: Copy + Ord> MapOp<P> for DilateOp {
    type Accumulator = P;
    type Output = P;

    #[inline(always)]
    fn init(&self, center: P) -> P {
        center
    }

    #[inline(always)]
    fn accumulate(&self, acc: &mut P, item: MapItem<P>) {
        *acc = (*acc).max(item.pixel);
    }

    #[inline(always)]
    fn finalize(&mut self, acc: P) -> P {
        acc
    }
}

/// A [`MapOp`] that collects all mask-true neighbour pixels into a buffer,
/// sorts them, and returns the middle element (median).
///
/// The `center` argument is **not** used directly — the center pixel reaches
/// the sort buffer via the mask's anchor position when `mask[anchor] == true`,
/// which is the standard convention for median filters. If the mask excludes
/// the anchor the median is computed over non-center neighbours only.
///
/// The internal `Vec<P>` is pre-allocated once per [`median_filter`] call and
/// reused across all pixels, avoiding per-pixel heap allocation.
struct MedianOp<P> {
    buf: Vec<P>,
}

impl<P: Copy + Ord> MapOp<P> for MedianOp<P> {
    type Accumulator = Vec<P>;
    type Output = P;

    const INVERTIBLE: bool = false;

    fn init(&self, _center: P) -> Vec<P> {
        Vec::new()
    }

    fn accumulate(&self, acc: &mut Vec<P>, item: MapItem<P>) {
        acc.push(item.pixel);
    }

    fn finalize(&mut self, mut acc: Vec<P>) -> P {
        assert!(
            !acc.is_empty(),
            "median_filter: mask must have at least one active position"
        );
        acc.sort_unstable();
        acc[acc.len() / 2]
    }

    /// Direct override — reuses the internal buffer to avoid per-pixel allocation.
    fn map<I>(&mut self, _center: P, neighbors: I) -> P
    where
        I: Iterator<Item = MapItem<P>>,
    {
        self.buf.clear();
        self.buf.extend(neighbors.map(|n| n.pixel));
        assert!(
            !self.buf.is_empty(),
            "median_filter: mask must have at least one active position"
        );
        self.buf.sort_unstable();
        self.buf[self.buf.len() / 2]
    }
}

// ─── Erode ───────────────────────────────────────────────────────────────────

/// Write the result of an erosion (local minimum) into `output`.
///
/// For every output pixel the minimum source pixel under the "true"
/// positions of the structuring element is selected.
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
/// use fovea::Size;
/// use fovea::border::Clamp;
/// use fovea::transform::erode_into;
///
/// let src = Image::fill(5, 5, 10u8);
/// let se = Neighborhood::<bool, 3, 3>::full_rect_3x3();
///
/// let border = Clamp;
/// let out_region = fovea::border::BorderPolicy::<Image<u8>>::output_region(
///     &border, src.size(), se.weights().size(), se.anchor(),
/// );
/// let mut out = Image::<u8>::zero(out_region.size.width, out_region.size.height);
///
/// erode_into(&src, &se, &border, &mut out);
///
/// for y in 0..out.height() {
///     for x in 0..out.width() {
///         assert_eq!(out.pixel_at(x, y), 10);
///     }
/// }
/// ```
pub fn erode_into<I, K, B, O, P>(image: &I, kernel: &K, border: &B, output: &mut O)
where
    I: RasterImage<Pixel = P>,
    P: Copy + Ord,
    K: Kernel<Weight = bool>,
    B: BorderPolicy<I>,
    O: RasterImageMut<Pixel = P>,
{
    map_neighborhood_into(
        image,
        kernel.weights(),
        kernel.anchor(),
        border,
        output,
        ErodeOp,
    );
}

/// Erode (local minimum) the image and return a newly allocated result.
///
/// This is a convenience wrapper around [`erode_into`].
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView, Neighborhood};
/// use fovea::border::Clamp;
/// use fovea::transform::erode;
///
/// let src = Image::generate(5, 5, |x, y| (x + y * 5) as u8);
/// let se = Neighborhood::<bool, 3, 3>::full_rect_3x3();
///
/// let result: Image<u8> = erode(&src, &se, &Clamp);
///
/// assert_eq!(result.width(), 5);
/// assert_eq!(result.height(), 5);
/// // The minimum at the center (2,2) over a 3×3 window of
/// // [6,7,8,11,12,13,16,17,18] is 6.
/// assert_eq!(result.pixel_at(2, 2), 6);
/// ```
#[must_use]
pub fn erode<I, K, B, P>(image: &I, kernel: &K, border: &B) -> Image<P>
where
    I: RasterImage<Pixel = P>,
    P: Copy + Ord + ZeroablePixel,
    K: Kernel<Weight = bool>,
    B: BorderPolicy<I>,
{
    map_neighborhood(image, kernel.weights(), kernel.anchor(), border, ErodeOp)
}

// ─── Dilate ──────────────────────────────────────────────────────────────────

/// Write the result of a dilation (local maximum) into `output`.
///
/// For every output pixel the maximum source pixel under the "true"
/// positions of the structuring element is selected.
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
/// use fovea::Size;
/// use fovea::border::Clamp;
/// use fovea::transform::dilate_into;
///
/// let src = Image::fill(5, 5, 10u8);
/// let se = Neighborhood::<bool, 3, 3>::full_rect_3x3();
///
/// let border = Clamp;
/// let out_region = fovea::border::BorderPolicy::<Image<u8>>::output_region(
///     &border, src.size(), se.weights().size(), se.anchor(),
/// );
/// let mut out = Image::<u8>::zero(out_region.size.width, out_region.size.height);
///
/// dilate_into(&src, &se, &border, &mut out);
///
/// for y in 0..out.height() {
///     for x in 0..out.width() {
///         assert_eq!(out.pixel_at(x, y), 10);
///     }
/// }
/// ```
pub fn dilate_into<I, K, B, O, P>(image: &I, kernel: &K, border: &B, output: &mut O)
where
    I: RasterImage<Pixel = P>,
    P: Copy + Ord,
    K: Kernel<Weight = bool>,
    B: BorderPolicy<I>,
    O: RasterImageMut<Pixel = P>,
{
    map_neighborhood_into(
        image,
        kernel.weights(),
        kernel.anchor(),
        border,
        output,
        DilateOp,
    );
}

/// Dilate (local maximum) the image and return a newly allocated result.
///
/// This is a convenience wrapper around [`dilate_into`].
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView, Neighborhood};
/// use fovea::border::Clamp;
/// use fovea::transform::dilate;
///
/// let src = Image::generate(5, 5, |x, y| (x + y * 5) as u8);
/// let se = Neighborhood::<bool, 3, 3>::full_rect_3x3();
///
/// let result: Image<u8> = dilate(&src, &se, &Clamp);
///
/// assert_eq!(result.width(), 5);
/// assert_eq!(result.height(), 5);
/// // The maximum at the center (2,2) over a 3×3 window of
/// // [6,7,8,11,12,13,16,17,18] is 18.
/// assert_eq!(result.pixel_at(2, 2), 18);
/// ```
#[must_use]
pub fn dilate<I, K, B, P>(image: &I, kernel: &K, border: &B) -> Image<P>
where
    I: RasterImage<Pixel = P>,
    P: Copy + Ord + ZeroablePixel,
    K: Kernel<Weight = bool>,
    B: BorderPolicy<I>,
{
    map_neighborhood(image, kernel.weights(), kernel.anchor(), border, DilateOp)
}

// ─── Composite operations ────────────────────────────────────────────────────

/// Morphological opening with caller-provided scratch buffer.
///
/// Opening (erosion followed by dilation) removes small bright spots while
/// preserving the overall shape of larger bright regions.
///
/// `scratch` is used for the intermediate erosion result. Its contents
/// after the call are unspecified.
///
/// # Panics
///
/// Panics if `output` or `scratch` dimensions do not match the output
/// region determined by the border policy.
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView, ImageViewMut, Neighborhood};
/// use fovea::border::Clamp;
/// use fovea::transform::opening_into;
///
/// let src = Image::fill(6, 6, 50u8);
/// let se = Neighborhood::<bool, 3, 3>::full_rect_3x3();
///
/// let mut output = Image::<u8>::zero(6, 6);
/// let mut scratch = Image::<u8>::zero(6, 6);
/// opening_into(&src, &se, &Clamp, &mut output, &mut scratch);
///
/// for y in 0..output.height() {
///     for x in 0..output.width() {
///         assert_eq!(output.pixel_at(x, y), 50);
///     }
/// }
/// ```
pub fn opening_into<I, K, B, O, P>(
    image: &I,
    kernel: &K,
    border: &B,
    output: &mut O,
    scratch: &mut Image<P>,
) where
    I: RasterImage<Pixel = P>,
    P: Copy + Ord + ZeroablePixel,
    K: Kernel<Weight = bool>,
    B: BorderPolicy<I> + BorderPolicy<Image<P>>,
    O: RasterImageMut<Pixel = P>,
{
    erode_into(image, kernel, border, scratch);
    dilate_into(scratch, kernel, border, output);
}

/// Morphological closing with caller-provided scratch buffer.
///
/// Closing (dilation followed by erosion) fills small dark holes while
/// preserving the overall shape of larger dark regions.
///
/// `scratch` is used for the intermediate dilation result. Its contents
/// after the call are unspecified.
///
/// # Panics
///
/// Panics if `output` or `scratch` dimensions do not match the output
/// region determined by the border policy.
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView, ImageViewMut, Neighborhood};
/// use fovea::border::Clamp;
/// use fovea::transform::closing_into;
///
/// let src = Image::fill(6, 6, 50u8);
/// let se = Neighborhood::<bool, 3, 3>::full_rect_3x3();
///
/// let mut output = Image::<u8>::zero(6, 6);
/// let mut scratch = Image::<u8>::zero(6, 6);
/// closing_into(&src, &se, &Clamp, &mut output, &mut scratch);
///
/// for y in 0..output.height() {
///     for x in 0..output.width() {
///         assert_eq!(output.pixel_at(x, y), 50);
///     }
/// }
/// ```
pub fn closing_into<I, K, B, O, P>(
    image: &I,
    kernel: &K,
    border: &B,
    output: &mut O,
    scratch: &mut Image<P>,
) where
    I: RasterImage<Pixel = P>,
    P: Copy + Ord + ZeroablePixel,
    K: Kernel<Weight = bool>,
    B: BorderPolicy<I> + BorderPolicy<Image<P>>,
    O: RasterImageMut<Pixel = P>,
{
    dilate_into(image, kernel, border, scratch);
    erode_into(scratch, kernel, border, output);
}

/// Morphological opening: erosion followed by dilation.
///
/// Opening removes small bright spots (noise) while preserving the overall
/// shape and size of larger bright regions.
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView, Neighborhood};
/// use fovea::border::Clamp;
/// use fovea::transform::opening;
///
/// let src = Image::fill(6, 6, 50u8);
/// let se = Neighborhood::<bool, 3, 3>::full_rect_3x3();
///
/// let result: Image<u8> = opening(&src, &se, &Clamp);
///
/// // Uniform image: opening has no effect
/// for y in 0..result.height() {
///     for x in 0..result.width() {
///         assert_eq!(result.pixel_at(x, y), 50);
///     }
/// }
/// ```
#[must_use]
pub fn opening<I, K, B, P>(image: &I, kernel: &K, border: &B) -> Image<P>
where
    I: RasterImage<Pixel = P>,
    P: Copy + Ord + ZeroablePixel,
    K: Kernel<Weight = bool>,
    B: BorderPolicy<I> + BorderPolicy<Image<P>>,
{
    // #todo: can we avoid the intermediate image allocation?
    let eroded = erode(image, kernel, border);
    dilate(&eroded, kernel, border)
}

/// Morphological closing: dilation followed by erosion.
///
/// Closing fills small dark holes while preserving the overall shape and
/// size of larger dark regions.
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView, Neighborhood};
/// use fovea::border::Clamp;
/// use fovea::transform::closing;
///
/// let src = Image::fill(6, 6, 50u8);
/// let se = Neighborhood::<bool, 3, 3>::full_rect_3x3();
///
/// let result: Image<u8> = closing(&src, &se, &Clamp);
///
/// // Uniform image: closing has no effect
/// for y in 0..result.height() {
///     for x in 0..result.width() {
///         assert_eq!(result.pixel_at(x, y), 50);
///     }
/// }
/// ```
#[must_use]
pub fn closing<I, K, B, P>(image: &I, kernel: &K, border: &B) -> Image<P>
where
    I: RasterImage<Pixel = P>,
    P: Copy + Ord + ZeroablePixel,
    K: Kernel<Weight = bool>,
    B: BorderPolicy<I> + BorderPolicy<Image<P>>,
{
    // #todo: can we avoid the intermediate image allocation?
    let dilated = dilate(image, kernel, border);
    erode(&dilated, kernel, border)
}

/// Morphological gradient: dilation − erosion.
///
/// Highlights edges by computing the difference between the local maximum
/// and local minimum for each pixel. The result is always ≥ 0 for unsigned
/// types (assuming dilation ≥ erosion, which holds by definition).
///
/// The output pixel type is the same as the input. For unsigned integer
/// pixels this works naturally (dilation ≥ erosion). For signed types the
/// subtraction is standard wrapping/saturating depending on the type.
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView, Neighborhood};
/// use fovea::border::Clamp;
/// use fovea::transform::morphological_gradient;
///
/// let src = Image::fill(6, 6, 50u8);
/// let se = Neighborhood::<bool, 3, 3>::full_rect_3x3();
///
/// let result: Image<u8> = morphological_gradient(&src, &se, &Clamp);
///
/// // Uniform image: gradient is zero everywhere
/// for y in 0..result.height() {
///     for x in 0..result.width() {
///         assert_eq!(result.pixel_at(x, y), 0);
///     }
/// }
/// ```
#[must_use]
pub fn morphological_gradient<I, K, B, P>(image: &I, kernel: &K, border: &B) -> Image<P>
where
    I: RasterImage<Pixel = P>,
    P: Copy + Ord + ZeroablePixel + core::ops::Sub<Output = P>,
    K: Kernel<Weight = bool>,
    B: BorderPolicy<I> + BorderPolicy<Image<P>>,
{
    let dilated = dilate(image, kernel, border);
    let eroded = erode(image, kernel, border);

    combine_images(&dilated, &eroded, PixelSubtract)
        .expect("internal: dilated and eroded are always produced from the same source image and have matching sizes")
}

/// Top-hat transform: original − opening.
///
/// Isolates bright peaks / features that are smaller than the structuring
/// element.
///
/// # Border policy bound
///
/// `B: FullFrameBorder<I>` — the border policy must preserve the input
/// image's dimensions, because top-hat performs a pixel-wise subtraction
/// between `image` and `opening(image)`. Calling `top_hat(.., &Skip)` is
/// therefore a **compile-time error** rather than a runtime panic.
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView, Neighborhood};
/// use fovea::border::Clamp;
/// use fovea::transform::top_hat;
///
/// let src = Image::fill(6, 6, 50u8);
/// let se = Neighborhood::<bool, 3, 3>::full_rect_3x3();
///
/// let result: Image<u8> = top_hat(&src, &se, &Clamp);
///
/// // Uniform image: top-hat is zero
/// for y in 0..result.height() {
///     for x in 0..result.width() {
///         assert_eq!(result.pixel_at(x, y), 0);
///     }
/// }
/// ```
///
/// # Compile-fail
///
/// Using `Skip` would shrink the output and break the same-size invariant.
/// The trait bound rejects it at compile time:
///
/// ```compile_fail
/// use fovea::image::{Image, Neighborhood};
/// use fovea::border::Skip;
/// use fovea::transform::top_hat;
///
/// let src = Image::fill(6, 6, 50u8);
/// let se  = Neighborhood::<bool, 3, 3>::full_rect_3x3();
/// let _: Image<u8> = top_hat(&src, &se, &Skip);
/// ```
#[must_use]
pub fn top_hat<I, K, B, P>(image: &I, kernel: &K, border: &B) -> Image<P>
where
    I: RasterImage<Pixel = P>,
    P: Copy + Ord + ZeroablePixel + core::ops::Sub<Output = P>,
    K: Kernel<Weight = bool>,
    // Composite morphology subtracts `opening(image)` from `image` pixel-wise,
    // so both must have the same dimensions. `FullFrameBorder` excludes
    // `Skip` (which shrinks the output) at compile time — see ADR/P1-5.
    B: FullFrameBorder<I> + BorderPolicy<Image<P>>,
{
    let opened = opening(image, kernel, border);

    combine_images(image, &opened, PixelSubtract)
        .expect("internal: opened is produced from image and always has the same size")
}

/// Black-hat transform: closing − original.
///
/// Isolates dark valleys / features that are smaller than the structuring
/// element.
///
/// # Border policy bound
///
/// `B: FullFrameBorder<I>` — same constraint as [`top_hat`]: the closing
/// must preserve dimensions for the pixel-wise subtraction to be valid.
/// Passing `Skip` is a compile-time error.
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView, Neighborhood};
/// use fovea::border::Clamp;
/// use fovea::transform::black_hat;
///
/// let src = Image::fill(6, 6, 50u8);
/// let se = Neighborhood::<bool, 3, 3>::full_rect_3x3();
///
/// let result: Image<u8> = black_hat(&src, &se, &Clamp);
///
/// // Uniform image: black-hat is zero
/// for y in 0..result.height() {
///     for x in 0..result.width() {
///         assert_eq!(result.pixel_at(x, y), 0);
///     }
/// }
/// ```
#[must_use]
pub fn black_hat<I, K, B, P>(image: &I, kernel: &K, border: &B) -> Image<P>
where
    I: RasterImage<Pixel = P>,
    P: Copy + Ord + ZeroablePixel + core::ops::Sub<Output = P>,
    K: Kernel<Weight = bool>,
    // See `top_hat` — same frame-size constraint applies here.
    B: FullFrameBorder<I> + BorderPolicy<Image<P>>,
{
    let closed = closing(image, kernel, border);

    combine_images(&closed, image, PixelSubtract)
        .expect("internal: closed is produced from image and always has the same size")
}

// ─── Median filter ───────────────────────────────────────────────────────────

/// Apply a median filter using a boolean mask and return a newly allocated
/// result image.
///
/// For every output pixel the neighborhood pixels at `true` mask positions
/// are collected, sorted, and the middle element is returned. The mask should
/// normally include the anchor position (`mask[anchor] == true`) so that the
/// center pixel participates in the sort; this is the standard convention for
/// median filters.
///
/// The internal sort buffer is allocated **once** per call and reused across
/// all pixels, so the only heap allocation is proportional to the mask size,
/// not to the image area.
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView, ImageViewMut, Neighborhood};
/// use fovea::border::Clamp;
/// use fovea::transform::median_filter;
///
/// // 5×5 image filled with 50, with one bright spike at the center.
/// let mut src = Image::fill(5, 5, 50u8);
/// *src.pixel_at_mut(2, 2) = 200;
///
/// let se = Neighborhood::<bool, 3, 3>::full_rect_3x3();
/// let result: Image<u8> = median_filter(&src, &se, &Clamp);
///
/// // The isolated spike should be replaced by the background median (50).
/// assert_eq!(result.pixel_at(2, 2), 50);
/// ```
#[must_use]
pub fn median_filter<I, K, B, P>(image: &I, kernel: &K, border: &B) -> Image<P>
where
    I: RasterImage<Pixel = P>,
    P: Copy + Ord + ZeroablePixel,
    K: Kernel<Weight = bool>,
    B: BorderPolicy<I>,
{
    let mask_size = kernel.weights().size();
    let capacity = mask_size.width * mask_size.height;
    map_neighborhood(
        image,
        kernel.weights(),
        kernel.anchor(),
        border,
        MedianOp {
            buf: Vec::with_capacity(capacity),
        },
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::border::Clamp;
    use crate::image::{ImageViewMut, Neighborhood};

    // ── helpers ──────────────────────────────────────────────────────────

    fn make_5x5_gradient() -> Image<u8> {
        Image::generate(5, 5, |x, y| (x + y * 5) as u8)
    }

    // ── erode: uniform image stays the same ─────────────────────────────

    #[test]
    fn erode_uniform_is_identity() {
        let src = Image::fill(6, 6, 42u8);
        let se = Neighborhood::<bool, 3, 3>::full_rect_3x3();

        let result: Image<u8> = erode(&src, &se, &Clamp);

        for y in 0..result.height() {
            for x in 0..result.width() {
                assert_eq!(result.pixel_at(x, y), 42);
            }
        }
    }

    // ── dilate: uniform image stays the same ────────────────────────────

    #[test]
    fn dilate_uniform_is_identity() {
        let src = Image::fill(6, 6, 42u8);
        let se = Neighborhood::<bool, 3, 3>::full_rect_3x3();

        let result: Image<u8> = dilate(&src, &se, &Clamp);

        for y in 0..result.height() {
            for x in 0..result.width() {
                assert_eq!(result.pixel_at(x, y), 42);
            }
        }
    }

    // ── erode known gradient ────────────────────────────────────────────

    #[test]
    fn erode_gradient_center_pixel() {
        let src = make_5x5_gradient();
        let se = Neighborhood::<bool, 3, 3>::full_rect_3x3();

        let result: Image<u8> = erode(&src, &se, &Clamp);

        // Center pixel (2,2): 3×3 window covers (1,1)..(3,3) inclusive.
        // Values: (1,1)=6, (2,1)=7, (3,1)=8,
        //         (1,2)=11,(2,2)=12,(3,2)=13,
        //         (1,3)=16,(2,3)=17,(3,3)=18
        // min = 6
        assert_eq!(result.pixel_at(2, 2), 6);
    }

    // ── dilate known gradient ───────────────────────────────────────────

    #[test]
    fn dilate_gradient_center_pixel() {
        let src = make_5x5_gradient();
        let se = Neighborhood::<bool, 3, 3>::full_rect_3x3();

        let result: Image<u8> = dilate(&src, &se, &Clamp);

        // Center pixel (2,2): 3×3 window covers (1,1)..(3,3) inclusive.
        // Values: 6,7,8,11,12,13,16,17,18 → max = 18
        assert_eq!(result.pixel_at(2, 2), 18);
    }

    // ── erode ≤ original ≤ dilate ───────────────────────────────────────

    #[test]
    fn erode_le_original_le_dilate() {
        let src = make_5x5_gradient();
        let se = Neighborhood::<bool, 3, 3>::full_rect_3x3();

        let eroded: Image<u8> = erode(&src, &se, &Clamp);
        let dilated: Image<u8> = dilate(&src, &se, &Clamp);

        for y in 0..src.height() {
            for x in 0..src.width() {
                assert!(
                    eroded.pixel_at(x, y) <= src.pixel_at(x, y),
                    "erode > original at ({x}, {y})",
                );
                assert!(
                    src.pixel_at(x, y) <= dilated.pixel_at(x, y),
                    "original > dilate at ({x}, {y})",
                );
            }
        }
    }

    // ── opening ≤ original ──────────────────────────────────────────────

    #[test]
    fn opening_le_original() {
        let src = make_5x5_gradient();
        let se = Neighborhood::<bool, 3, 3>::full_rect_3x3();

        let opened: Image<u8> = opening(&src, &se, &Clamp);

        for y in 0..src.height() {
            for x in 0..src.width() {
                assert!(
                    opened.pixel_at(x, y) <= src.pixel_at(x, y),
                    "opening > original at ({x}, {y}): {} > {}",
                    opened.pixel_at(x, y),
                    src.pixel_at(x, y),
                );
            }
        }
    }

    // ── closing ≥ original ──────────────────────────────────────────────

    #[test]
    fn closing_ge_original() {
        let src = make_5x5_gradient();
        let se = Neighborhood::<bool, 3, 3>::full_rect_3x3();

        let closed: Image<u8> = closing(&src, &se, &Clamp);

        for y in 0..src.height() {
            for x in 0..src.width() {
                assert!(
                    closed.pixel_at(x, y) >= src.pixel_at(x, y),
                    "closing < original at ({x}, {y}): {} < {}",
                    closed.pixel_at(x, y),
                    src.pixel_at(x, y),
                );
            }
        }
    }

    // ── opening on uniform = identity ───────────────────────────────────

    #[test]
    fn opening_uniform_is_identity() {
        let src = Image::fill(6, 6, 50u8);
        let se = Neighborhood::<bool, 3, 3>::full_rect_3x3();

        let result: Image<u8> = opening(&src, &se, &Clamp);

        for y in 0..result.height() {
            for x in 0..result.width() {
                assert_eq!(result.pixel_at(x, y), 50);
            }
        }
    }

    // ── closing on uniform = identity ───────────────────────────────────

    #[test]
    fn closing_uniform_is_identity() {
        let src = Image::fill(6, 6, 50u8);
        let se = Neighborhood::<bool, 3, 3>::full_rect_3x3();

        let result: Image<u8> = closing(&src, &se, &Clamp);

        for y in 0..result.height() {
            for x in 0..result.width() {
                assert_eq!(result.pixel_at(x, y), 50);
            }
        }
    }

    // ── morphological gradient on uniform = zero ────────────────────────

    #[test]
    fn morphological_gradient_uniform_is_zero() {
        let src = Image::fill(6, 6, 50u8);
        let se = Neighborhood::<bool, 3, 3>::full_rect_3x3();

        let result: Image<u8> = morphological_gradient(&src, &se, &Clamp);

        for y in 0..result.height() {
            for x in 0..result.width() {
                assert_eq!(result.pixel_at(x, y), 0);
            }
        }
    }

    // ── morphological gradient = dilate - erode ─────────────────────────

    #[test]
    fn morphological_gradient_equals_dilate_minus_erode() {
        let src = make_5x5_gradient();
        let se = Neighborhood::<bool, 3, 3>::full_rect_3x3();

        let gradient: Image<u8> = morphological_gradient(&src, &se, &Clamp);
        let dilated: Image<u8> = dilate(&src, &se, &Clamp);
        let eroded: Image<u8> = erode(&src, &se, &Clamp);

        for y in 0..gradient.height() {
            for x in 0..gradient.width() {
                let expected = dilated.pixel_at(x, y) - eroded.pixel_at(x, y);
                assert_eq!(
                    gradient.pixel_at(x, y),
                    expected,
                    "gradient mismatch at ({x}, {y})",
                );
            }
        }
    }

    // ── top-hat on uniform = zero ───────────────────────────────────────

    #[test]
    fn top_hat_uniform_is_zero() {
        let src = Image::fill(6, 6, 50u8);
        let se = Neighborhood::<bool, 3, 3>::full_rect_3x3();

        let result: Image<u8> = top_hat(&src, &se, &Clamp);

        for y in 0..result.height() {
            for x in 0..result.width() {
                assert_eq!(result.pixel_at(x, y), 0);
            }
        }
    }

    // ── black-hat on uniform = zero ─────────────────────────────────────

    #[test]
    fn black_hat_uniform_is_zero() {
        let src = Image::fill(6, 6, 50u8);
        let se = Neighborhood::<bool, 3, 3>::full_rect_3x3();

        let result: Image<u8> = black_hat(&src, &se, &Clamp);

        for y in 0..result.height() {
            for x in 0..result.width() {
                assert_eq!(result.pixel_at(x, y), 0);
            }
        }
    }

    // ── top_hat = original - opening ────────────────────────────────────

    #[test]
    fn top_hat_equals_original_minus_opening() {
        let src = make_5x5_gradient();
        let se = Neighborhood::<bool, 3, 3>::full_rect_3x3();

        let th: Image<u8> = top_hat(&src, &se, &Clamp);
        let opened: Image<u8> = opening(&src, &se, &Clamp);

        for y in 0..th.height() {
            for x in 0..th.width() {
                let expected = src.pixel_at(x, y) - opened.pixel_at(x, y);
                assert_eq!(
                    th.pixel_at(x, y),
                    expected,
                    "top-hat mismatch at ({x}, {y})",
                );
            }
        }
    }

    // ── black_hat = closing - original ──────────────────────────────────

    #[test]
    fn black_hat_equals_closing_minus_original() {
        let src = make_5x5_gradient();
        let se = Neighborhood::<bool, 3, 3>::full_rect_3x3();

        let bh: Image<u8> = black_hat(&src, &se, &Clamp);
        let closed: Image<u8> = closing(&src, &se, &Clamp);

        for y in 0..bh.height() {
            for x in 0..bh.width() {
                let expected = closed.pixel_at(x, y) - src.pixel_at(x, y);
                assert_eq!(
                    bh.pixel_at(x, y),
                    expected,
                    "black-hat mismatch at ({x}, {y})",
                );
            }
        }
    }

    // ── cross structuring element ───────────────────────────────────────

    #[test]
    fn erode_with_cross_se() {
        let src = make_5x5_gradient();
        let se = Neighborhood::<bool, 3, 3>::cross_3x3();

        let result: Image<u8> = erode(&src, &se, &Clamp);

        // cross_3x3 has true at: (1,0),(0,1),(1,1),(2,1),(1,2)
        // relative to anchor (1,1), offsets: (0,-1),(-1,0),(0,0),(1,0),(0,1)
        // For center pixel (2,2): src positions (2,1),(1,2),(2,2),(3,2),(2,3)
        // Values: 7, 11, 12, 13, 17 → min = 7
        assert_eq!(result.pixel_at(2, 2), 7);
    }

    #[test]
    fn dilate_with_cross_se() {
        let src = make_5x5_gradient();
        let se = Neighborhood::<bool, 3, 3>::cross_3x3();

        let result: Image<u8> = dilate(&src, &se, &Clamp);

        // cross_3x3 offsets: (0,-1),(-1,0),(0,0),(1,0),(0,1)
        // For center pixel (2,2): src positions (2,1),(1,2),(2,2),(3,2),(2,3)
        // Values: 7, 11, 12, 13, 17 → max = 17
        assert_eq!(result.pixel_at(2, 2), 17);
    }

    // ── 5×5 structuring element ─────────────────────────────────────────

    #[test]
    fn erode_5x5_full_rect() {
        let src = Image::generate(7, 7, |x, y| (x + y * 7) as u8);
        let se = Neighborhood::<bool, 5, 5>::full_rect_5x5();

        let result: Image<u8> = erode(&src, &se, &Clamp);

        // Center pixel (3,3): 5×5 window centered at (3,3) covers
        // rows 1..=5, cols 1..=5. Minimum is at (1,1) = 1*7 + 1 = 8.
        assert_eq!(result.pixel_at(3, 3), 8);
    }

    #[test]
    fn dilate_5x5_full_rect() {
        let src = Image::generate(7, 7, |x, y| (x + y * 7) as u8);
        let se = Neighborhood::<bool, 5, 5>::full_rect_5x5();

        let result: Image<u8> = dilate(&src, &se, &Clamp);

        // Center pixel (3,3): 5×5 window centered at (3,3) covers
        // rows 1..=5, cols 1..=5. Maximum is at (5,5) = 5*7 + 5 = 40.
        assert_eq!(result.pixel_at(3, 3), 40);
    }

    // ── erode_into and erode produce same result ────────────────────────

    #[test]
    fn erode_into_matches_erode() {
        let src = make_5x5_gradient();
        let se = Neighborhood::<bool, 3, 3>::full_rect_3x3();

        let expected: Image<u8> = erode(&src, &se, &Clamp);

        let border = Clamp;
        let out_region = <Clamp as BorderPolicy<Image<u8>>>::output_region(
            &border,
            src.size(),
            se.weights().size(),
            se.anchor(),
        );
        let mut into_result = Image::<u8>::zero(out_region.size.width, out_region.size.height);
        erode_into(&src, &se, &border, &mut into_result);

        for y in 0..expected.height() {
            for x in 0..expected.width() {
                assert_eq!(
                    expected.pixel_at(x, y),
                    into_result.pixel_at(x, y),
                    "mismatch at ({x}, {y})",
                );
            }
        }
    }

    // ── dilate_into and dilate produce same result ──────────────────────

    #[test]
    fn dilate_into_matches_dilate() {
        let src = make_5x5_gradient();
        let se = Neighborhood::<bool, 3, 3>::full_rect_3x3();

        let expected: Image<u8> = dilate(&src, &se, &Clamp);

        let border = Clamp;
        let out_region = <Clamp as BorderPolicy<Image<u8>>>::output_region(
            &border,
            src.size(),
            se.weights().size(),
            se.anchor(),
        );
        let mut into_result = Image::<u8>::zero(out_region.size.width, out_region.size.height);
        dilate_into(&src, &se, &border, &mut into_result);

        for y in 0..expected.height() {
            for x in 0..expected.width() {
                assert_eq!(
                    expected.pixel_at(x, y),
                    into_result.pixel_at(x, y),
                    "mismatch at ({x}, {y})",
                );
            }
        }
    }

    // ── single-pixel image ──────────────────────────────────────────────

    #[test]
    fn erode_single_pixel() {
        let src = Image::fill(1, 1, 99u8);
        let se = Neighborhood::<bool, 3, 3>::full_rect_3x3();

        let result: Image<u8> = erode(&src, &se, &Clamp);

        assert_eq!(result.width(), 1);
        assert_eq!(result.height(), 1);
        assert_eq!(result.pixel_at(0, 0), 99);
    }

    #[test]
    fn dilate_single_pixel() {
        let src = Image::fill(1, 1, 99u8);
        let se = Neighborhood::<bool, 3, 3>::full_rect_3x3();

        let result: Image<u8> = dilate(&src, &se, &Clamp);

        assert_eq!(result.width(), 1);
        assert_eq!(result.height(), 1);
        assert_eq!(result.pixel_at(0, 0), 99);
    }

    // ── idempotence: opening(opening(x)) == opening(x) ─────────────────

    #[test]
    fn opening_is_idempotent() {
        let src = make_5x5_gradient();
        let se = Neighborhood::<bool, 3, 3>::full_rect_3x3();

        let once: Image<u8> = opening(&src, &se, &Clamp);
        let twice: Image<u8> = opening(&once, &se, &Clamp);

        for y in 0..once.height() {
            for x in 0..once.width() {
                assert_eq!(
                    once.pixel_at(x, y),
                    twice.pixel_at(x, y),
                    "opening not idempotent at ({x}, {y})",
                );
            }
        }
    }

    // ── idempotence: closing(closing(x)) == closing(x) ─────────────────

    #[test]
    fn closing_is_idempotent() {
        let src = make_5x5_gradient();
        let se = Neighborhood::<bool, 3, 3>::full_rect_3x3();

        let once: Image<u8> = closing(&src, &se, &Clamp);
        let twice: Image<u8> = closing(&once, &se, &Clamp);

        for y in 0..once.height() {
            for x in 0..once.width() {
                assert_eq!(
                    once.pixel_at(x, y),
                    twice.pixel_at(x, y),
                    "closing not idempotent at ({x}, {y})",
                );
            }
        }
    }

    // ── morphological gradient ≥ 0 ──────────────────────────────────────

    #[test]
    fn morphological_gradient_nonnegative() {
        let src = make_5x5_gradient();
        let se = Neighborhood::<bool, 3, 3>::full_rect_3x3();

        let result: Image<u8> = morphological_gradient(&src, &se, &Clamp);

        // For u8, subtraction would underflow if dilate < erode, but by
        // definition dilate ≥ erode, so all values should be valid.
        // We just verify they're computed without panic.
        for y in 0..result.height() {
            for x in 0..result.width() {
                let _ = result.pixel_at(x, y);
            }
        }
    }

    // ── bright spot removed by opening ──────────────────────────────────

    #[test]
    fn opening_removes_single_bright_pixel() {
        let mut src = Image::fill(5, 5, 10u8);
        // Place a single bright pixel at the center
        *src.pixel_at_mut(2, 2) = 200;

        let se = Neighborhood::<bool, 3, 3>::full_rect_3x3();
        let result: Image<u8> = opening(&src, &se, &Clamp);

        // The bright single pixel should be eroded away, then dilation
        // can't bring it back. The center pixel should be ≤ 10.
        assert!(
            result.pixel_at(2, 2) <= 10,
            "opening should remove single bright pixel, got {}",
            result.pixel_at(2, 2),
        );
    }

    // ── dark spot filled by closing ─────────────────────────────────────

    #[test]
    fn closing_fills_single_dark_pixel() {
        let mut src = Image::fill(5, 5, 200u8);
        // Place a single dark pixel at the center
        *src.pixel_at_mut(2, 2) = 10;

        let se = Neighborhood::<bool, 3, 3>::full_rect_3x3();
        let result: Image<u8> = closing(&src, &se, &Clamp);

        // The dark single pixel should be dilated away, then erosion
        // can't bring it back. The center pixel should be ≥ 200.
        assert!(
            result.pixel_at(2, 2) >= 200,
            "closing should fill single dark pixel, got {}",
            result.pixel_at(2, 2),
        );
    }

    // ── median_filter ────────────────────────────────────────────────────────

    #[test]
    fn median_filter_uniform_is_identity() {
        let src = Image::fill(6, 6, 42u8);
        let se = Neighborhood::<bool, 3, 3>::full_rect_3x3();

        let result: Image<u8> = median_filter(&src, &se, &Clamp);

        assert_eq!(result.width(), 6);
        assert_eq!(result.height(), 6);
        for y in 0..result.height() {
            for x in 0..result.width() {
                assert_eq!(result.pixel_at(x, y), 42);
            }
        }
    }

    #[test]
    fn median_filter_removes_isolated_bright_spike() {
        // 5×5 image filled with 50, single spike at center = 200.
        let mut src = Image::fill(5, 5, 50u8);
        *src.pixel_at_mut(2, 2) = 200;

        let se = Neighborhood::<bool, 3, 3>::full_rect_3x3();
        let result: Image<u8> = median_filter(&src, &se, &Clamp);

        // 3×3 window at (2,2): eight 50s and one 200.
        // Sorted: [50,50,50,50,50,50,50,50,200] → median (index 4) = 50.
        assert_eq!(result.pixel_at(2, 2), 50);
    }

    #[test]
    fn median_filter_known_3x3_value() {
        // 3×3 source with values 1..=9; Skip policy → 1×1 output.
        // pixel(x, y) = x + 1 + y*3 → row-major: 1,2,3,4,5,6,7,8,9.
        let src = Image::generate(3, 3, |x, y| (x + 1 + y * 3) as u8);
        let se = Neighborhood::<bool, 3, 3>::full_rect_3x3();

        use crate::border::Skip;
        let result: Image<u8> = median_filter(&src, &se, &Skip);

        assert_eq!(result.width(), 1);
        assert_eq!(result.height(), 1);
        // All 9 values [1,2,3,4,5,6,7,8,9]; median (index 4) = 5.
        assert_eq!(result.pixel_at(0, 0), 5);
    }

    #[test]
    fn median_filter_skip_output_size() {
        let src = Image::fill(7, 7, 0u8);
        let se = Neighborhood::<bool, 3, 3>::full_rect_3x3();

        use crate::border::Skip;
        let result: Image<u8> = median_filter(&src, &se, &Skip);

        // Skip with 3×3 kernel removes 1-pixel border on each side.
        assert_eq!(result.width(), 5);
        assert_eq!(result.height(), 5);
    }

    #[test]
    fn median_filter_matches_sort_by_hand() {
        // 5×5 gradient; check center pixel (2,2) with Clamp.
        // Window at (2,2) covers source rows y=1..=3, cols x=1..=3.
        // Values: 6,7,8,11,12,13,16,17,18 → sorted → median (index 4) = 12.
        let src = make_5x5_gradient();
        let se = Neighborhood::<bool, 3, 3>::full_rect_3x3();

        let result: Image<u8> = median_filter(&src, &se, &Clamp);

        assert_eq!(result.pixel_at(2, 2), 12);
    }

    // ── Panic paths ──────────────────────────────────────────────────────

    #[test]
    #[should_panic(expected = "mask must have at least one active position")]
    fn median_filter_empty_mask_panics() {
        // An all-false structuring element means the neighbours iterator is
        // always empty.  MedianOp::map asserts the buffer is non-empty, so
        // this should panic on the very first output pixel.
        let src = Image::fill(5, 5, 50u8);
        let empty_mask = Neighborhood::<bool, 3, 3>::new([false; 9]);
        let _: Image<u8> = median_filter(&src, &empty_mask, &Clamp);
    }

    // ── opening_into and opening produce same result ────────────────────

    #[test]
    fn opening_into_matches_opening() {
        let src = make_5x5_gradient();
        let se = Neighborhood::<bool, 3, 3>::full_rect_3x3();

        let expected: Image<u8> = opening(&src, &se, &Clamp);

        let border = Clamp;
        let out_region = <Clamp as BorderPolicy<Image<u8>>>::output_region(
            &border,
            src.size(),
            se.weights().size(),
            se.anchor(),
        );
        let mut into_result = Image::<u8>::zero(out_region.size.width, out_region.size.height);
        let mut scratch = Image::<u8>::zero(out_region.size.width, out_region.size.height);
        opening_into(&src, &se, &border, &mut into_result, &mut scratch);

        for y in 0..expected.height() {
            for x in 0..expected.width() {
                assert_eq!(
                    expected.pixel_at(x, y),
                    into_result.pixel_at(x, y),
                    "mismatch at ({x}, {y})",
                );
            }
        }
    }

    // ── closing_into and closing produce same result ────────────────────

    #[test]
    fn closing_into_matches_closing() {
        let src = make_5x5_gradient();
        let se = Neighborhood::<bool, 3, 3>::full_rect_3x3();

        let expected: Image<u8> = closing(&src, &se, &Clamp);

        let border = Clamp;
        let out_region = <Clamp as BorderPolicy<Image<u8>>>::output_region(
            &border,
            src.size(),
            se.weights().size(),
            se.anchor(),
        );
        let mut into_result = Image::<u8>::zero(out_region.size.width, out_region.size.height);
        let mut scratch = Image::<u8>::zero(out_region.size.width, out_region.size.height);
        closing_into(&src, &se, &border, &mut into_result, &mut scratch);

        for y in 0..expected.height() {
            for x in 0..expected.width() {
                assert_eq!(
                    expected.pixel_at(x, y),
                    into_result.pixel_at(x, y),
                    "mismatch at ({x}, {y})",
                );
            }
        }
    }

    // ── opening_into on uniform = identity ──────────────────────────────

    #[test]
    fn opening_into_uniform_is_identity() {
        let src = Image::fill(6, 6, 50u8);
        let se = Neighborhood::<bool, 3, 3>::full_rect_3x3();

        let mut output = Image::<u8>::zero(6, 6);
        let mut scratch = Image::<u8>::zero(6, 6);
        opening_into(&src, &se, &Clamp, &mut output, &mut scratch);

        for y in 0..output.height() {
            for x in 0..output.width() {
                assert_eq!(output.pixel_at(x, y), 50);
            }
        }
    }

    // ── closing_into on uniform = identity ──────────────────────────────

    #[test]
    fn closing_into_uniform_is_identity() {
        let src = Image::fill(6, 6, 50u8);
        let se = Neighborhood::<bool, 3, 3>::full_rect_3x3();

        let mut output = Image::<u8>::zero(6, 6);
        let mut scratch = Image::<u8>::zero(6, 6);
        closing_into(&src, &se, &Clamp, &mut output, &mut scratch);

        for y in 0..output.height() {
            for x in 0..output.width() {
                assert_eq!(output.pixel_at(x, y), 50);
            }
        }
    }
}

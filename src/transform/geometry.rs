//! Axis-aligned geometric transforms — physical (allocating) flips and
//! rotations.
//!
//! These are physical operations rather than view wrappers: each function
//! returns a normal image or writes into a caller-supplied output image.
//!
//! ## Functions
//!
//! - [`flip_h`] / [`flip_v`] — horizontal / vertical flip
//! - [`rotate_90`] / [`rotate_180`] / [`rotate_270`] — quarter / half rotations
//! - [`transpose`] — matrix transpose (swap x and y)
//!
//! Each function has a `_into` companion that writes into a pre-allocated
//! [`RasterImageMut`].
//!
//! ## Bounds
//!
//! These transforms only **copy** pixels — they do not read individual
//! channels or perform arithmetic. The bound on the pixel type is therefore
//! exactly [`Copy`] (plus [`ZeroablePixel`] on the allocating variants, to
//! initialise the destination buffer). No `LinearPixel`, `LinearSpace`, or
//! `PlainPixel` constraint is required, so e.g. [`crate::pixel::Indexed8`]
//! and [`crate::pixel::Srgb8`] work just like
//! [`crate::pixel::Mono8`] or [`crate::pixel::Rgb8`].
//!
//! ## Cache strategy
//!
//! [`flip_h`], [`flip_v`], and [`rotate_180`] all access input and output
//! row-major and are cache-friendly by construction.
//!
//! [`rotate_90`], [`rotate_270`], and [`transpose`] cannot be sequential
//! on both sides. The implementation iterates the input row-major (sequential
//! reads) and writes to scattered output positions. For images up to roughly
//! 512 × 512 the working set fits in L2 and the naïve loop is fast enough;
//! a blocked variant is deferred per ADR-0006 ("YAGNI for performance").
//!
//! ## Measured throughput
//!
//! `cargo bench -p fovea --bench geometry`, `Mono8`, single thread,
//! Windows / x86–64 (numbers will vary, but the relative shape is
//! what matters):
//!
//! | Function       |   256² (L2)  | 1024² (L2→L3) |  4096² (DRAM) |
//! |----------------|---------------|----------------|----------------|
//! | `flip_v`       |  ~23 GiB/s    |  ~23 GiB/s     |  ~5.7 GiB/s    |
//! | `flip_h`       |  ~6.3 GiB/s   |   ~5.8 GiB/s   |  ~3.8 GiB/s    |
//! | `rotate_180`   |  ~3.1 GiB/s   |   ~3.0 GiB/s   |  ~2.5 GiB/s    |
//! | `rotate_90`    |  660 MiB/s    |   220 MiB/s    |  130 MiB/s     |
//! | `rotate_270`   |  655 MiB/s    |   265 MiB/s    |  129 MiB/s     |
//! | `transpose`    |  640 MiB/s    |   280 MiB/s    |  125 MiB/s     |
//!
//! `flip_v` is `memcpy`-bound. `flip_h` / `rotate_180` are partially
//! auto-vectorised reverse-copies. `rotate_90` / `rotate_270` /
//! `transpose` collapse once the output working set exceeds L2 because
//! every store hits a different cache line. The textbook fix is the
//! blocked variant is deferred until there is a clear API need. The benchmark
//! suite contains a per-implementation breakdown including an
//! `transpose_row_cached` alternative that buys roughly 2× below DRAM size
//! but does not solve the cliff at ≥16 MiB.

use crate::Size;
use crate::image::{Image, RasterImage, RasterImageMut};
use crate::pixel::ZeroablePixel;

// ─── Horizontal flip ────────────────────────────────────────────────────────

/// Writes a horizontal flip of `img` into `out`.
///
/// The output pixel at `(x, y)` equals the input pixel at `(W - 1 - x, y)`,
/// where `W = img.width()`.
///
/// # Panics
///
/// Panics if `out.size() != img.size()` (Tier 3 — programmer bug).
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::Mono8;
/// use fovea::transform::flip_h_into;
///
/// let src = Image::from_vec(2, 1, vec![Mono8::new(1), Mono8::new(2)]).unwrap();
/// let mut dst = Image::<Mono8>::zero(2, 1);
/// flip_h_into(&src, &mut dst);
/// assert_eq!(dst.pixel_at(0, 0), Mono8::new(2));
/// assert_eq!(dst.pixel_at(1, 0), Mono8::new(1));
/// ```
pub fn flip_h_into<I, O>(img: &I, out: &mut O)
where
    I: RasterImage,
    O: RasterImageMut<Pixel = I::Pixel>,
{
    assert_eq!(
        img.size(),
        out.size(),
        "flip_h_into: input size {:?} does not match output size {:?}",
        img.size(),
        out.size()
    );
    let w = img.width();
    for y in 0..img.height() {
        let src = img.row(y);
        let dst = out.row_mut(y);
        for x in 0..w {
            dst[w - 1 - x] = src[x];
        }
    }
}

/// Returns a horizontally flipped copy of `img`.
///
/// See [`flip_h_into`] for the coordinate mapping.
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::Mono8;
/// use fovea::transform::flip_h;
///
/// let src = Image::from_vec(2, 1, vec![Mono8::new(1), Mono8::new(2)]).unwrap();
/// let dst = flip_h(&src);
/// assert_eq!(dst.pixel_at(0, 0), Mono8::new(2));
/// ```
#[must_use]
pub fn flip_h<I>(img: &I) -> Image<I::Pixel>
where
    I: RasterImage,
    I::Pixel: ZeroablePixel,
{
    let mut out = Image::<I::Pixel>::zero(img.width(), img.height());
    flip_h_into(img, &mut out);
    out
}

// ─── Vertical flip ──────────────────────────────────────────────────────────

/// Writes a vertical flip of `img` into `out`.
///
/// The output pixel at `(x, y)` equals the input pixel at `(x, H - 1 - y)`,
/// where `H = img.height()`. The implementation is a row-by-row copy, so
/// both input and output are accessed sequentially.
///
/// # Panics
///
/// Panics if `out.size() != img.size()`.
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::Mono8;
/// use fovea::transform::flip_v_into;
///
/// let src = Image::from_vec(1, 2, vec![Mono8::new(1), Mono8::new(2)]).unwrap();
/// let mut dst = Image::<Mono8>::zero(1, 2);
/// flip_v_into(&src, &mut dst);
/// assert_eq!(dst.pixel_at(0, 0), Mono8::new(2));
/// ```
pub fn flip_v_into<I, O>(img: &I, out: &mut O)
where
    I: RasterImage,
    O: RasterImageMut<Pixel = I::Pixel>,
{
    assert_eq!(
        img.size(),
        out.size(),
        "flip_v_into: input size {:?} does not match output size {:?}",
        img.size(),
        out.size()
    );
    let h = img.height();
    for y in 0..h {
        let src = img.row(y);
        let dst = out.row_mut(h - 1 - y);
        dst.copy_from_slice(src);
    }
}

/// Returns a vertically flipped copy of `img`.
///
/// See [`flip_v_into`] for the coordinate mapping.
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::Mono8;
/// use fovea::transform::flip_v;
///
/// let src = Image::from_vec(1, 2, vec![Mono8::new(1), Mono8::new(2)]).unwrap();
/// let dst = flip_v(&src);
/// assert_eq!(dst.pixel_at(0, 0), Mono8::new(2));
/// ```
#[must_use]
pub fn flip_v<I>(img: &I) -> Image<I::Pixel>
where
    I: RasterImage,
    I::Pixel: ZeroablePixel,
{
    let mut out = Image::<I::Pixel>::zero(img.width(), img.height());
    flip_v_into(img, &mut out);
    out
}

// ─── Rotate 90° (counter-clockwise) ─────────────────────────────────────────

/// Writes a 90° counter-clockwise rotation of `img` into `out`.
///
/// Output dimensions are `(img.height(), img.width())`. The output pixel
/// at `(x', y') = (y, W - 1 - x)` equals the input pixel at `(x, y)`.
/// This matches OpenCV's `ROTATE_90_COUNTERCLOCKWISE`.
///
/// # Panics
///
/// Panics if `out.size() != (img.height(), img.width())`.
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::Mono8;
/// use fovea::transform::rotate_90_into;
///
/// // 2×1 image: [1, 2]
/// let src = Image::from_vec(2, 1, vec![Mono8::new(1), Mono8::new(2)]).unwrap();
/// // CCW → 1×2 column [2, 1] (top-to-bottom)
/// let mut dst = Image::<Mono8>::zero(1, 2);
/// rotate_90_into(&src, &mut dst);
/// assert_eq!(dst.pixel_at(0, 0), Mono8::new(2));
/// assert_eq!(dst.pixel_at(0, 1), Mono8::new(1));
/// ```
pub fn rotate_90_into<I, O>(img: &I, out: &mut O)
where
    I: RasterImage,
    O: RasterImageMut<Pixel = I::Pixel>,
{
    let expected = Size::new(img.height(), img.width());
    assert_eq!(
        out.size(),
        expected,
        "rotate_90_into: expected output size {:?}, got {:?}",
        expected,
        out.size()
    );
    let w = img.width();
    let h = img.height();
    // (x, y) → (y, w - 1 - x). Iterate input row-major for sequential reads.
    for y in 0..h {
        let src = img.row(y);
        for (x, &p) in src.iter().enumerate() {
            *out.pixel_at_mut(y, w - 1 - x) = p;
        }
    }
    // PERF: blocked variant deferred per ADR-0006 (YAGNI).
}

/// Returns a 90° counter-clockwise rotated copy of `img`.
///
/// Output dimensions are `(img.height(), img.width())`.
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::Mono8;
/// use fovea::transform::rotate_90;
///
/// let src = Image::from_vec(2, 1, vec![Mono8::new(1), Mono8::new(2)]).unwrap();
/// let dst = rotate_90(&src);
/// assert_eq!(dst.width(), 1);
/// assert_eq!(dst.height(), 2);
/// ```
#[must_use]
pub fn rotate_90<I>(img: &I) -> Image<I::Pixel>
where
    I: RasterImage,
    I::Pixel: ZeroablePixel,
{
    let mut out = Image::<I::Pixel>::zero(img.height(), img.width());
    rotate_90_into(img, &mut out);
    out
}

// ─── Rotate 180° ────────────────────────────────────────────────────────────

/// Writes a 180° rotation of `img` into `out`.
///
/// The output pixel at `(W - 1 - x, H - 1 - y)` equals the input pixel
/// at `(x, y)`. Equivalent to `flip_h(flip_v(img))`.
///
/// # Panics
///
/// Panics if `out.size() != img.size()`.
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::Mono8;
/// use fovea::transform::rotate_180_into;
///
/// let src = Image::from_vec(2, 1, vec![Mono8::new(1), Mono8::new(2)]).unwrap();
/// let mut dst = Image::<Mono8>::zero(2, 1);
/// rotate_180_into(&src, &mut dst);
/// assert_eq!(dst.pixel_at(0, 0), Mono8::new(2));
/// ```
pub fn rotate_180_into<I, O>(img: &I, out: &mut O)
where
    I: RasterImage,
    O: RasterImageMut<Pixel = I::Pixel>,
{
    assert_eq!(
        img.size(),
        out.size(),
        "rotate_180_into: input size {:?} does not match output size {:?}",
        img.size(),
        out.size()
    );
    let w = img.width();
    let h = img.height();
    for y in 0..h {
        let src = img.row(y);
        let dst = out.row_mut(h - 1 - y);
        for x in 0..w {
            dst[w - 1 - x] = src[x];
        }
    }
}

/// Returns a 180° rotated copy of `img`.
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::Mono8;
/// use fovea::transform::rotate_180;
///
/// let src = Image::from_vec(2, 1, vec![Mono8::new(1), Mono8::new(2)]).unwrap();
/// let dst = rotate_180(&src);
/// assert_eq!(dst.pixel_at(0, 0), Mono8::new(2));
/// ```
#[must_use]
pub fn rotate_180<I>(img: &I) -> Image<I::Pixel>
where
    I: RasterImage,
    I::Pixel: ZeroablePixel,
{
    let mut out = Image::<I::Pixel>::zero(img.width(), img.height());
    rotate_180_into(img, &mut out);
    out
}

// ─── Rotate 270° (== 90° clockwise) ─────────────────────────────────────────

/// Writes a 270° counter-clockwise rotation of `img` into `out`.
///
/// Equivalent to a 90° **clockwise** rotation. Output dimensions are
/// `(img.height(), img.width())`. The output pixel at
/// `(x', y') = (H - 1 - y, x)` equals the input pixel at `(x, y)`.
///
/// # Panics
///
/// Panics if `out.size() != (img.height(), img.width())`.
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::Mono8;
/// use fovea::transform::rotate_270_into;
///
/// // 2×1 image: [1, 2]
/// let src = Image::from_vec(2, 1, vec![Mono8::new(1), Mono8::new(2)]).unwrap();
/// // CW → 1×2 column [1, 2]
/// let mut dst = Image::<Mono8>::zero(1, 2);
/// rotate_270_into(&src, &mut dst);
/// assert_eq!(dst.pixel_at(0, 0), Mono8::new(1));
/// assert_eq!(dst.pixel_at(0, 1), Mono8::new(2));
/// ```
pub fn rotate_270_into<I, O>(img: &I, out: &mut O)
where
    I: RasterImage,
    O: RasterImageMut<Pixel = I::Pixel>,
{
    let expected = Size::new(img.height(), img.width());
    assert_eq!(
        out.size(),
        expected,
        "rotate_270_into: expected output size {:?}, got {:?}",
        expected,
        out.size()
    );
    let h = img.height();
    // (x, y) → (h - 1 - y, x). Iterate input row-major for sequential reads.
    for y in 0..h {
        let src = img.row(y);
        let dst_y = h - 1 - y;
        for (x, &p) in src.iter().enumerate() {
            *out.pixel_at_mut(dst_y, x) = p;
        }
    }
    // PERF: blocked variant deferred per ADR-0006 (YAGNI).
}

/// Returns a 270° counter-clockwise (== 90° clockwise) rotated copy of `img`.
///
/// Output dimensions are `(img.height(), img.width())`.
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::Mono8;
/// use fovea::transform::rotate_270;
///
/// let src = Image::from_vec(2, 1, vec![Mono8::new(1), Mono8::new(2)]).unwrap();
/// let dst = rotate_270(&src);
/// assert_eq!(dst.width(), 1);
/// assert_eq!(dst.height(), 2);
/// ```
#[must_use]
pub fn rotate_270<I>(img: &I) -> Image<I::Pixel>
where
    I: RasterImage,
    I::Pixel: ZeroablePixel,
{
    let mut out = Image::<I::Pixel>::zero(img.height(), img.width());
    rotate_270_into(img, &mut out);
    out
}

// ─── Transpose ──────────────────────────────────────────────────────────────

/// Writes the matrix transpose of `img` into `out`.
///
/// The output pixel at `(y, x)` equals the input pixel at `(x, y)`. This is
/// equivalent to `flip_v(rotate_90(img))` (or `rotate_90(flip_h(img))`).
///
/// Output dimensions are `(img.height(), img.width())`.
///
/// # Panics
///
/// Panics if `out.size() != (img.height(), img.width())`.
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::Mono8;
/// use fovea::transform::transpose_into;
///
/// let src = Image::from_vec(2, 1, vec![Mono8::new(1), Mono8::new(2)]).unwrap();
/// let mut dst = Image::<Mono8>::zero(1, 2);
/// transpose_into(&src, &mut dst);
/// assert_eq!(dst.pixel_at(0, 0), Mono8::new(1));
/// assert_eq!(dst.pixel_at(0, 1), Mono8::new(2));
/// ```
pub fn transpose_into<I, O>(img: &I, out: &mut O)
where
    I: RasterImage,
    O: RasterImageMut<Pixel = I::Pixel>,
{
    let expected = Size::new(img.height(), img.width());
    assert_eq!(
        out.size(),
        expected,
        "transpose_into: expected output size {:?}, got {:?}",
        expected,
        out.size()
    );
    let h = img.height();
    // (x, y) → (y, x). Iterate input row-major for sequential reads.
    for y in 0..h {
        let src = img.row(y);
        for (x, &p) in src.iter().enumerate() {
            *out.pixel_at_mut(y, x) = p;
        }
    }
    // PERF: blocked variant deferred per ADR-0006 (YAGNI).
}

/// Returns the matrix transpose of `img`.
///
/// Output dimensions are `(img.height(), img.width())`.
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::Mono8;
/// use fovea::transform::transpose;
///
/// let src = Image::from_vec(2, 1, vec![Mono8::new(1), Mono8::new(2)]).unwrap();
/// let dst = transpose(&src);
/// assert_eq!(dst.width(), 1);
/// assert_eq!(dst.height(), 2);
/// ```
#[must_use]
pub fn transpose<I>(img: &I) -> Image<I::Pixel>
where
    I: RasterImage,
    I::Pixel: ZeroablePixel,
{
    let mut out = Image::<I::Pixel>::zero(img.height(), img.width());
    transpose_into(img, &mut out);
    out
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Rectangle;
    use crate::image::{Image, ImageArray, ImageView, SubView};
    use crate::pixel::{Indexed8, Mono8, Rgb8, Srgb8};

    fn mono(v: u8) -> Mono8 {
        Mono8::new(v)
    }

    fn collect<I: ImageView<Pixel = Mono8>>(img: &I) -> Vec<u8> {
        let mut out = Vec::with_capacity(img.width() * img.height());
        for y in 0..img.height() {
            for x in 0..img.width() {
                out.push(img.pixel_at(x, y).value());
            }
        }
        out
    }

    fn from_u8(width: usize, height: usize, data: &[u8]) -> Image<Mono8> {
        Image::from_vec(width, height, data.iter().copied().map(mono).collect()).unwrap()
    }

    // ─── flip_h ─────────────────────────────────────────────────────────────

    #[test]
    fn flip_h_2x2() {
        let src = from_u8(2, 2, &[1, 2, 3, 4]);
        let dst = flip_h(&src);
        assert_eq!(collect(&dst), vec![2, 1, 4, 3]);
    }

    #[test]
    fn flip_h_3x2_rectangular() {
        let src = from_u8(3, 2, &[1, 2, 3, 4, 5, 6]);
        let dst = flip_h(&src);
        assert_eq!(collect(&dst), vec![3, 2, 1, 6, 5, 4]);
    }

    #[test]
    fn flip_h_idempotent() {
        let src = from_u8(3, 2, &[1, 2, 3, 4, 5, 6]);
        let twice = flip_h(&flip_h(&src));
        assert_eq!(collect(&twice), collect(&src));
    }

    // ─── flip_v ─────────────────────────────────────────────────────────────

    #[test]
    fn flip_v_2x2() {
        let src = from_u8(2, 2, &[1, 2, 3, 4]);
        let dst = flip_v(&src);
        assert_eq!(collect(&dst), vec![3, 4, 1, 2]);
    }

    #[test]
    fn flip_v_3x2_rectangular() {
        let src = from_u8(3, 2, &[1, 2, 3, 4, 5, 6]);
        let dst = flip_v(&src);
        assert_eq!(collect(&dst), vec![4, 5, 6, 1, 2, 3]);
    }

    #[test]
    fn flip_v_idempotent() {
        let src = from_u8(3, 2, &[1, 2, 3, 4, 5, 6]);
        let twice = flip_v(&flip_v(&src));
        assert_eq!(collect(&twice), collect(&src));
    }

    // ─── rotate_90 (CCW) ────────────────────────────────────────────────────

    #[test]
    fn rotate_90_2x3() {
        // 2×3:           CCW 90° → 3×2:
        //   1 2            2 4 6
        //   3 4            1 3 5
        //   5 6
        let src = from_u8(2, 3, &[1, 2, 3, 4, 5, 6]);
        let dst = rotate_90(&src);
        assert_eq!(dst.width(), 3);
        assert_eq!(dst.height(), 2);
        assert_eq!(collect(&dst), vec![2, 4, 6, 1, 3, 5]);
    }

    #[test]
    fn rotate_90_then_270_is_identity() {
        let src = from_u8(3, 2, &[1, 2, 3, 4, 5, 6]);
        let round = rotate_270(&rotate_90(&src));
        assert_eq!(round.size(), src.size());
        assert_eq!(collect(&round), collect(&src));
    }

    #[test]
    fn rotate_90_three_times_equals_rotate_270() {
        let src = from_u8(3, 2, &[1, 2, 3, 4, 5, 6]);
        let triple = rotate_90(&rotate_90(&rotate_90(&src)));
        let direct = rotate_270(&src);
        assert_eq!(triple.size(), direct.size());
        assert_eq!(collect(&triple), collect(&direct));
    }

    // ─── rotate_180 ─────────────────────────────────────────────────────────

    #[test]
    fn rotate_180_3x2_equals_flip_h_compose_flip_v() {
        let src = from_u8(3, 2, &[1, 2, 3, 4, 5, 6]);
        let by_rotate = rotate_180(&src);
        let by_compose = flip_h(&flip_v(&src));
        assert_eq!(collect(&by_rotate), collect(&by_compose));
        assert_eq!(collect(&by_rotate), vec![6, 5, 4, 3, 2, 1]);
    }

    #[test]
    fn rotate_180_idempotent() {
        let src = from_u8(3, 2, &[1, 2, 3, 4, 5, 6]);
        let twice = rotate_180(&rotate_180(&src));
        assert_eq!(collect(&twice), collect(&src));
    }

    // ─── rotate_270 (CW) ────────────────────────────────────────────────────

    #[test]
    fn rotate_270_2x3() {
        // 2×3:           CW 90° (== CCW 270°) → 3×2:
        //   1 2            5 3 1
        //   3 4            6 4 2
        //   5 6
        let src = from_u8(2, 3, &[1, 2, 3, 4, 5, 6]);
        let dst = rotate_270(&src);
        assert_eq!(dst.width(), 3);
        assert_eq!(dst.height(), 2);
        assert_eq!(collect(&dst), vec![5, 3, 1, 6, 4, 2]);
    }

    // ─── transpose ──────────────────────────────────────────────────────────

    #[test]
    fn transpose_2x3_matches_flip_v_compose_rotate_90() {
        // Algebraic identity: transpose == flip_v ∘ rotate_90
        let src = from_u8(2, 3, &[1, 2, 3, 4, 5, 6]);
        let by_transpose = transpose(&src);
        let by_compose = flip_v(&rotate_90(&src));
        assert_eq!(by_transpose.size(), by_compose.size());
        assert_eq!(collect(&by_transpose), collect(&by_compose));
        // Hand-computed: 2×3 input rows [1,2],[3,4],[5,6] → 3×2 output rows
        // [1,3,5],[2,4,6].
        assert_eq!(collect(&by_transpose), vec![1, 3, 5, 2, 4, 6]);
    }

    #[test]
    fn transpose_square_is_self_inverse() {
        let src = from_u8(3, 3, &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let twice = transpose(&transpose(&src));
        assert_eq!(collect(&twice), collect(&src));
    }

    // ─── Output-size mismatch panics (Tier 3, ADR-0025) ─────────────────────

    #[test]
    #[should_panic(expected = "flip_h_into")]
    fn flip_h_into_size_mismatch_panics() {
        let src = from_u8(2, 2, &[1, 2, 3, 4]);
        let mut dst = Image::<Mono8>::zero(3, 2);
        flip_h_into(&src, &mut dst);
    }

    #[test]
    #[should_panic(expected = "flip_v_into")]
    fn flip_v_into_size_mismatch_panics() {
        let src = from_u8(2, 2, &[1, 2, 3, 4]);
        let mut dst = Image::<Mono8>::zero(2, 3);
        flip_v_into(&src, &mut dst);
    }

    #[test]
    #[should_panic(expected = "rotate_90_into")]
    fn rotate_90_into_size_mismatch_panics() {
        let src = from_u8(2, 3, &[1, 2, 3, 4, 5, 6]);
        // Should be 3×2, give 2×3 (same as input) to trigger panic.
        let mut dst = Image::<Mono8>::zero(2, 3);
        rotate_90_into(&src, &mut dst);
    }

    #[test]
    #[should_panic(expected = "rotate_180_into")]
    fn rotate_180_into_size_mismatch_panics() {
        let src = from_u8(2, 3, &[1, 2, 3, 4, 5, 6]);
        let mut dst = Image::<Mono8>::zero(3, 2);
        rotate_180_into(&src, &mut dst);
    }

    #[test]
    #[should_panic(expected = "rotate_270_into")]
    fn rotate_270_into_size_mismatch_panics() {
        let src = from_u8(2, 3, &[1, 2, 3, 4, 5, 6]);
        let mut dst = Image::<Mono8>::zero(2, 3);
        rotate_270_into(&src, &mut dst);
    }

    #[test]
    #[should_panic(expected = "transpose_into")]
    fn transpose_into_size_mismatch_panics() {
        let src = from_u8(2, 3, &[1, 2, 3, 4, 5, 6]);
        let mut dst = Image::<Mono8>::zero(2, 3);
        transpose_into(&src, &mut dst);
    }

    // ─── Strided SubView input ──────────────────────────────────────────────

    #[test]
    fn flip_v_subview_input() {
        // 4×4 outer image; take a 2×3 interior ROI starting at (1, 1):
        //   row 1: . a b .       a=2 b=3
        //   row 2: . c d .       c=6 d=7
        //   row 3: . e f .       e=10 f=11
        let outer = from_u8(
            4,
            4,
            &[
                0, 0, 0, 0, // row 0
                0, 2, 3, 0, // row 1
                0, 6, 7, 0, // row 2
                0, 10, 11, 0, // row 3
            ],
        );
        let roi = outer.roi(Rectangle::new((1, 1), (2, 3))).unwrap();
        let flipped = flip_v(&roi);
        // ROI rows reversed: [10,11], [6,7], [2,3]
        assert_eq!(collect(&flipped), vec![10, 11, 6, 7, 2, 3]);
    }

    #[test]
    fn rotate_90_subview_input() {
        // Same outer image as above.
        let outer = from_u8(
            4,
            4,
            &[
                0, 0, 0, 0, //
                0, 2, 3, 0, //
                0, 6, 7, 0, //
                0, 10, 11, 0, //
            ],
        );
        let roi = outer.roi(Rectangle::new((1, 1), (2, 3))).unwrap();
        // ROI is 2×3:        rotate_90 (CCW) → 3×2:
        //   2 3                 3 7 11
        //   6 7                 2 6 10
        //  10 11
        let rotated = rotate_90(&roi);
        assert_eq!(rotated.width(), 3);
        assert_eq!(rotated.height(), 2);
        assert_eq!(collect(&rotated), vec![3, 7, 11, 2, 6, 10]);
    }

    // ─── ImageArray (RasterImageMut) output ─────────────────────────────────

    #[test]
    fn rotate_90_imagearray_output() {
        // 2×3 input → 3×2 output stored in a stack-allocated ImageArray.
        let src = from_u8(2, 3, &[1, 2, 3, 4, 5, 6]);
        let mut out: ImageArray<Mono8, 3, 2> = ImageArray::new([mono(0); 6]);
        rotate_90_into(&src, &mut out);
        assert_eq!(out.pixel_at(0, 0), mono(2));
        assert_eq!(out.pixel_at(1, 0), mono(4));
        assert_eq!(out.pixel_at(2, 0), mono(6));
        assert_eq!(out.pixel_at(0, 1), mono(1));
        assert_eq!(out.pixel_at(1, 1), mono(3));
        assert_eq!(out.pixel_at(2, 1), mono(5));
    }

    // ─── Pixel-type-blind bound (Copy is sufficient on _into) ───────────────

    #[test]
    fn flip_h_rgb8() {
        let a = Rgb8::new(1, 2, 3);
        let b = Rgb8::new(4, 5, 6);
        let src = Image::from_vec(2, 1, vec![a, b]).unwrap();
        let dst = flip_h(&src);
        assert_eq!(dst.pixel_at(0, 0), b);
        assert_eq!(dst.pixel_at(1, 0), a);
    }

    #[test]
    fn rotate_90_indexed8() {
        // `Indexed8` deliberately does **not** implement `LinearPixel`.
        // This test asserts that the rotation works regardless — the bound
        // is `Copy` (+ `ZeroablePixel` for the allocating variant), nothing
        // more.
        let src = Image::from_vec(
            2,
            3,
            vec![
                Indexed8(1),
                Indexed8(2),
                Indexed8(3),
                Indexed8(4),
                Indexed8(5),
                Indexed8(6),
            ],
        )
        .unwrap();
        let dst = rotate_90(&src);
        assert_eq!(dst.width(), 3);
        assert_eq!(dst.height(), 2);
        // Same coordinate mapping as the Mono8 case.
        assert_eq!(dst.pixel_at(0, 0), Indexed8(2));
        assert_eq!(dst.pixel_at(1, 0), Indexed8(4));
        assert_eq!(dst.pixel_at(2, 0), Indexed8(6));
        assert_eq!(dst.pixel_at(0, 1), Indexed8(1));
    }

    #[test]
    fn flip_v_srgb8_gamma_encoded() {
        // `Srgb8` is gamma-encoded — also not `LinearPixel`. Pure pixel
        // copying is still legal.
        let a = Srgb8::new(10, 20, 30);
        let b = Srgb8::new(40, 50, 60);
        let src = Image::from_vec(1, 2, vec![a, b]).unwrap();
        let dst = flip_v(&src);
        assert_eq!(dst.pixel_at(0, 0), b);
        assert_eq!(dst.pixel_at(0, 1), a);
    }
}

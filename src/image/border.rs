use crate::image::ImageView;
use crate::{Rectangle, Size};

/// Determines how to handle out-of-bounds pixel access at image edges.
///
/// `BorderPolicy` is a **trait** (not an enum) because:
///
/// - **Monomorphisation** gives specialised codegen per policy — important for
///   inner-loop performance (the compiler inlines `pixel_at` per policy type).
/// - **Extensible** — users can implement custom border policies.
/// - **Simple two-method interface** — no GATs, no associated types, no buffers.
///
/// The `fold_neighborhood` engine splits iteration into an **interior hot path**
/// (where the full kernel fits inside the image — no policy calls needed) and a
/// **boundary cold path** (where `pixel_at` is called for out-of-bounds
/// coordinates). This means `pixel_at` is only ever invoked for the thin border
/// strip, making per-call overhead irrelevant.
///
/// # Built-in policies
///
/// | Policy | Behaviour | `output_region` | `pixel_at` |
/// |--------|-----------|-----------------|------------|
/// | [`Skip`] | Only interior pixels; output is smaller | Shrunken rect | Never called |
/// | [`Clamp`] | Replicate nearest edge pixel | Full image | Clamp coords to `[0, dim-1]` |
/// | [`Mirror`] | Reflect at edges | Full image | Reflect coords at boundaries |
/// | [`Wrap`] | Periodic / tiling | Full image | Modulo coords |
/// | [`Constant`] | Fixed value for out-of-bounds | Full image | Return constant for OOB |
///
/// # Example
///
/// ```
/// use irys_cv::{Size, Rectangle, Coordinate};
/// use irys_cv::image::{Image, ImageView};
/// use irys_cv::border::{BorderPolicy, Clamp};
///
/// let img = Image::generate(4, 4, |x, y| (x + y * 4) as u8);
/// let policy = Clamp;
///
/// // In-bounds access works normally
/// assert_eq!(policy.pixel_at(&img, 0, 0), 0);
///
/// // Out-of-bounds access clamps to the nearest edge
/// assert_eq!(policy.pixel_at(&img, -1, 0), img.pixel_at(0, 0));
/// assert_eq!(policy.pixel_at(&img, 4, 0), img.pixel_at(3, 0));
/// ```
pub trait BorderPolicy<I: ImageView>
where
    I::Pixel: Copy,
{
    /// Fetch a pixel, handling out-of-bounds coordinates via the policy.
    ///
    /// The `fold_neighborhood` engine guarantees this is called **only** for
    /// boundary positions — interior positions use direct slice access.
    ///
    /// Coordinates are `isize` to naturally represent negative (left/top
    /// overshoot) and beyond-dimension (right/bottom overshoot) positions.
    fn pixel_at(&self, image: &I, x: isize, y: isize) -> I::Pixel;

    /// Compute the output region for this policy given image size, kernel size,
    /// and anchor position.
    ///
    /// - For policies that extend the image (`Clamp`, `Mirror`, `Wrap`,
    ///   `Constant`), this returns a rectangle covering the full image.
    /// - For `Skip`, this returns a shrunken rectangle representing only the
    ///   interior positions where the full kernel fits.
    fn output_region(
        &self,
        image_size: Size,
        kernel_size: Size,
        anchor: (usize, usize),
    ) -> Rectangle;
}

// ─── Helper: compute the interior region ────────────────────────────────────

/// Compute the interior rectangle where the full kernel fits entirely inside
/// the image, given the image size, kernel size, and anchor position.
///
/// This is the region where **no** border policy calls are needed — every
/// kernel position maps to a valid in-bounds pixel.
///
/// Returns `None` if the kernel is larger than the image in either dimension
/// (i.e. there is no interior region at all).
///
/// # Example
///
/// ```
/// use irys_cv::Size;
/// use irys_cv::border::compute_interior_region;
///
/// // 10×10 image, 3×3 kernel, centered anchor (1,1)
/// let interior = compute_interior_region(
///     Size::new(10, 10),
///     Size::new(3, 3),
///     (1, 1),
/// );
/// let r = interior.unwrap();
/// assert_eq!(r.left(), 1);
/// assert_eq!(r.top(), 1);
/// assert_eq!(r.right(), 9);
/// assert_eq!(r.bottom(), 9);
/// ```
pub fn compute_interior_region(
    image_size: Size,
    kernel_size: Size,
    anchor: (usize, usize),
) -> Option<Rectangle> {
    let (ax, ay) = anchor;

    // Left/top margin: the anchor position itself.
    let left = ax;
    let top = ay;

    // Right/bottom margin: kernel extent past the anchor.
    let right_margin = kernel_size.width.saturating_sub(ax + 1);
    let bottom_margin = kernel_size.height.saturating_sub(ay + 1);

    // Interior width/height — how many positions fit.
    let interior_width = image_size.width.checked_sub(left + right_margin)?;
    let interior_height = image_size.height.checked_sub(top + bottom_margin)?;

    if interior_width == 0 || interior_height == 0 {
        return None;
    }

    Some(Rectangle::new(
        (left, top),
        (interior_width, interior_height),
    ))
}

// ─── Helper: full-image output region ───────────────────────────────────────

/// Returns a rectangle covering the full image — used by policies that
/// extend the image at the borders.
#[inline]
fn full_image_region(image_size: Size) -> Rectangle {
    Rectangle::new((0, 0), image_size)
}

// ─── Skip ───────────────────────────────────────────────────────────────────

/// Border policy that **skips** boundary pixels entirely.
///
/// The output region is shrunken to the interior — only positions where the
/// full kernel fits inside the image are visited. `pixel_at` is never called.
///
/// This is equivalent to MATLAB's `'valid'` convolution mode.
///
/// # Example
///
/// ```
/// use irys_cv::{Size, Rectangle};
/// use irys_cv::image::{Image, ImageView};
/// use irys_cv::border::{BorderPolicy, Skip};
///
/// let img = Image::<u8>::zero(10, 10);
/// let region = BorderPolicy::<Image<u8>>::output_region(
///     &Skip, img.size(), Size::new(3, 3), (1, 1),
/// );
/// // 10×10 image, 3×3 kernel, center anchor → 8×8 output
/// assert_eq!(region.size.width, 8);
/// assert_eq!(region.size.height, 8);
/// assert_eq!(region.left(), 1);
/// assert_eq!(region.top(), 1);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Skip;

impl<I: ImageView> BorderPolicy<I> for Skip
where
    I::Pixel: Copy,
{
    /// # Panics
    ///
    /// Always panics — `Skip` should never have `pixel_at` called because
    /// `output_region` restricts iteration to the interior.
    #[inline]
    fn pixel_at(&self, _image: &I, _x: isize, _y: isize) -> I::Pixel {
        panic!(
            "`Skip` border policy should never call `pixel_at` — the output region excludes all boundary positions"
        );
    }

    #[inline]
    fn output_region(
        &self,
        image_size: Size,
        kernel_size: Size,
        anchor: (usize, usize),
    ) -> Rectangle {
        compute_interior_region(image_size, kernel_size, anchor).unwrap_or_else(|| {
            // Kernel is larger than image — no valid output positions.
            Rectangle::new((0, 0), (0, 0))
        })
    }
}

// ─── Clamp ──────────────────────────────────────────────────────────────────

/// Border policy that **clamps** out-of-bounds coordinates to the nearest edge
/// pixel (also known as "replicate" or "extend").
///
/// For a coordinate `(x, y)`:
/// - If `x < 0`, use `x = 0`
/// - If `x >= width`, use `x = width - 1`
/// - Same for `y`
///
/// # Example
///
/// ```
/// use irys_cv::Size;
/// use irys_cv::image::{Image, ImageView};
/// use irys_cv::border::{BorderPolicy, Clamp};
///
/// let img = Image::generate(4, 4, |x, y| (x + y * 4) as u8);
/// let policy = Clamp;
///
/// // Negative x clamps to left edge
/// assert_eq!(policy.pixel_at(&img, -1, 0), img.pixel_at(0, 0));
/// // Beyond right edge clamps to rightmost column
/// assert_eq!(policy.pixel_at(&img, 4, 1), img.pixel_at(3, 1));
/// // Both out of bounds
/// assert_eq!(policy.pixel_at(&img, -1, -1), img.pixel_at(0, 0));
/// assert_eq!(policy.pixel_at(&img, 10, 10), img.pixel_at(3, 3));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Clamp;

impl<I: ImageView> BorderPolicy<I> for Clamp
where
    I::Pixel: Copy,
{
    #[inline]
    fn pixel_at(&self, image: &I, x: isize, y: isize) -> I::Pixel {
        let w = image.width() as isize;
        let h = image.height() as isize;
        let cx = x.clamp(0, w - 1) as usize;
        let cy = y.clamp(0, h - 1) as usize;
        image.pixel_at(cx, cy)
    }

    #[inline]
    fn output_region(
        &self,
        image_size: Size,
        _kernel_size: Size,
        _anchor: (usize, usize),
    ) -> Rectangle {
        full_image_region(image_size)
    }
}

// ─── Mirror ─────────────────────────────────────────────────────────────────

/// Border policy that **reflects** out-of-bounds coordinates at the image edge
/// (also known as "reflect" or "mirror").
///
/// Uses reflection that does **not** duplicate the edge pixel. For example,
/// for a 1D image of width 4 (`[A, B, C, D]`):
///
/// ```text
/// ... D C B | A B C D | C B A ...
///          -1  0 1 2 3  4 5 6
/// ```
///
/// This is "reflect101" / `cv::BORDER_REFLECT_101` in OpenCV terms, and is
/// the most common mirror mode for convolution.
///
/// # Example
///
/// ```
/// use irys_cv::image::{Image, ImageView};
/// use irys_cv::border::{BorderPolicy, Mirror};
///
/// let img = Image::generate(4, 1, |x, _| x as u8);
/// // pixels: [0, 1, 2, 3]
/// let policy = Mirror;
///
/// // x = -1 reflects to x = 1
/// assert_eq!(policy.pixel_at(&img, -1, 0), 1);
/// // x = -2 reflects to x = 2
/// assert_eq!(policy.pixel_at(&img, -2, 0), 2);
/// // x = 4 reflects to x = 2
/// assert_eq!(policy.pixel_at(&img, 4, 0), 2);
/// // x = 5 reflects to x = 1
/// assert_eq!(policy.pixel_at(&img, 5, 0), 1);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mirror;

/// Reflect a coordinate into `[0, len-1]` using reflect-101 mode.
///
/// This is "reflect without edge duplication": the edge pixel is not
/// repeated in the reflected pattern.
///
/// For `len = 4` (indices 0..3):
/// ```text
/// period = 2 * (len - 1) = 6
/// x:    -3 -2 -1  0  1  2  3  4  5  6  7  8  9
/// maps:  3  2  1  0  1  2  3  2  1  0  1  2  3
/// ```
#[inline]
fn reflect101(coord: isize, len: usize) -> usize {
    debug_assert!(len >= 1, "reflect101 requires len >= 1");
    if len == 1 {
        return 0;
    }
    let period = 2 * (len as isize - 1); // e.g. len=4 → period=6
    // Bring coord into [0, period)
    let mut c = coord % period;
    if c < 0 {
        c += period;
    }
    // Now c is in [0, period). Fold the second half back.
    if c >= len as isize {
        c = period - c;
    }
    c as usize
}

impl<I: ImageView> BorderPolicy<I> for Mirror
where
    I::Pixel: Copy,
{
    #[inline]
    fn pixel_at(&self, image: &I, x: isize, y: isize) -> I::Pixel {
        let mx = reflect101(x, image.width());
        let my = reflect101(y, image.height());
        image.pixel_at(mx, my)
    }

    #[inline]
    fn output_region(
        &self,
        image_size: Size,
        _kernel_size: Size,
        _anchor: (usize, usize),
    ) -> Rectangle {
        full_image_region(image_size)
    }
}

// ─── Wrap ───────────────────────────────────────────────────────────────────

/// Border policy that **wraps** coordinates periodically (modulo), treating the
/// image as a torus / tileable texture.
///
/// For a coordinate `(x, y)`:
/// - `x` is taken modulo `width`
/// - `y` is taken modulo `height`
///
/// Negative coordinates wrap correctly (e.g. `x = -1` maps to `width - 1`).
///
/// # Example
///
/// ```
/// use irys_cv::image::{Image, ImageView};
/// use irys_cv::border::{BorderPolicy, Wrap};
///
/// let img = Image::generate(4, 1, |x, _| x as u8);
/// // pixels: [0, 1, 2, 3]
/// let policy = Wrap;
///
/// // x = -1 wraps to x = 3
/// assert_eq!(policy.pixel_at(&img, -1, 0), 3);
/// // x = 4 wraps to x = 0
/// assert_eq!(policy.pixel_at(&img, 4, 0), 0);
/// // x = 5 wraps to x = 1
/// assert_eq!(policy.pixel_at(&img, 5, 0), 1);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Wrap;

/// Euclidean remainder — always non-negative.
#[inline]
fn wrap_coord(coord: isize, len: usize) -> usize {
    let len_i = len as isize;
    let r = coord % len_i;
    if r < 0 {
        (r + len_i) as usize
    } else {
        r as usize
    }
}

impl<I: ImageView> BorderPolicy<I> for Wrap
where
    I::Pixel: Copy,
{
    #[inline]
    fn pixel_at(&self, image: &I, x: isize, y: isize) -> I::Pixel {
        let wx = wrap_coord(x, image.width());
        let wy = wrap_coord(y, image.height());
        image.pixel_at(wx, wy)
    }

    #[inline]
    fn output_region(
        &self,
        image_size: Size,
        _kernel_size: Size,
        _anchor: (usize, usize),
    ) -> Rectangle {
        full_image_region(image_size)
    }
}

// ─── Constant ───────────────────────────────────────────────────────────────

/// Border policy that returns a **constant pixel value** for any out-of-bounds
/// access.
///
/// The constant is specified at construction time and returned verbatim for
/// coordinates outside the image. In-bounds coordinates are fetched from the
/// image normally.
///
/// This is useful for "zero-padded" convolutions (use `Constant(0u8)`) or any
/// case where the border should be a fixed colour.
///
/// # Example
///
/// ```
/// use irys_cv::image::{Image, ImageView};
/// use irys_cv::border::{BorderPolicy, Constant};
///
/// let img = Image::generate(3, 3, |x, y| (x + y * 3) as u8);
/// let policy = Constant(0u8);
///
/// // In-bounds: normal pixel
/// assert_eq!(policy.pixel_at(&img, 1, 1), img.pixel_at(1, 1));
/// // Out-of-bounds: constant value
/// assert_eq!(policy.pixel_at(&img, -1, 0), 0);
/// assert_eq!(policy.pixel_at(&img, 3, 0), 0);
/// assert_eq!(policy.pixel_at(&img, 0, -1), 0);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Constant<P>(pub P);

impl<P, I> BorderPolicy<I> for Constant<P>
where
    P: Copy,
    I: ImageView<Pixel = P>,
{
    #[inline]
    fn pixel_at(&self, image: &I, x: isize, y: isize) -> P {
        let w = image.width() as isize;
        let h = image.height() as isize;
        if x >= 0 && x < w && y >= 0 && y < h {
            image.pixel_at(x as usize, y as usize)
        } else {
            self.0
        }
    }

    #[inline]
    fn output_region(
        &self,
        image_size: Size,
        _kernel_size: Size,
        _anchor: (usize, usize),
    ) -> Rectangle {
        full_image_region(image_size)
    }
}

// ─── FullFrameBorder marker (P1-5) ─────────────────────────────────────────────────────────────

mod sealed {
    pub trait FullFrameSealed {}
}

/// Marker trait: this border policy produces an output of the **same size**
/// as the input image.
///
/// All built-in policies satisfy this except [`Skip`], which deliberately
/// shrinks the output to the interior region. Composite morphology
/// operations such as
/// [`top_hat`](crate::transform::top_hat) and
/// [`black_hat`](crate::transform::black_hat) perform a pixel-wise
/// subtraction between the original image and a filtered version, which
/// requires both operands to have identical dimensions. Those
/// composites therefore bound on `FullFrameBorder` rather than on the
/// more permissive [`BorderPolicy`].
///
/// This is a **sealed** trait — third-party code cannot implement it.
/// External custom border policies that preserve frame size cannot opt
/// in today; if that becomes a real need we can convert the seal into a
/// safety contract (or unseal it) without breaking changes.
///
/// # Coverage
///
/// | Policy | `FullFrameBorder` |
/// |--------|-------------------|
/// | [`Clamp`] | ✅ |
/// | [`Mirror`] | ✅ |
/// | [`Wrap`] | ✅ |
/// | [`Constant`] | ✅ |
/// | [`Skip`] | ❌ (output is smaller) |
pub trait FullFrameBorder<I: ImageView>: BorderPolicy<I> + sealed::FullFrameSealed
where
    I::Pixel: Copy,
{
}

impl sealed::FullFrameSealed for Clamp {}
impl<I: ImageView> FullFrameBorder<I> for Clamp where I::Pixel: Copy {}

impl sealed::FullFrameSealed for Mirror {}
impl<I: ImageView> FullFrameBorder<I> for Mirror where I::Pixel: Copy {}

impl sealed::FullFrameSealed for Wrap {}
impl<I: ImageView> FullFrameBorder<I> for Wrap where I::Pixel: Copy {}

impl<P> sealed::FullFrameSealed for Constant<P> {}
impl<P, I> FullFrameBorder<I> for Constant<P>
where
    I: ImageView<Pixel = P>,
    P: Copy,
{
}

// Note the deliberate absence of `impl FullFrameSealed for Skip`:
// adding it would silently re-enable the old runtime-panic path in
// composites like `top_hat` / `black_hat`.

// ─── Tests ───────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Image;

    // ── compute_interior_region ─────────────────────────────────────────

    #[test]
    fn interior_3x3_centered_on_10x10() {
        let r = compute_interior_region(Size::new(10, 10), Size::new(3, 3), (1, 1)).unwrap();
        assert_eq!(r.left(), 1);
        assert_eq!(r.top(), 1);
        assert_eq!(r.right(), 9);
        assert_eq!(r.bottom(), 9);
        assert_eq!(r.size, Size::new(8, 8));
    }

    #[test]
    fn interior_5x5_centered_on_10x10() {
        let r = compute_interior_region(Size::new(10, 10), Size::new(5, 5), (2, 2)).unwrap();
        assert_eq!(r.left(), 2);
        assert_eq!(r.top(), 2);
        assert_eq!(r.right(), 8);
        assert_eq!(r.bottom(), 8);
        assert_eq!(r.size, Size::new(6, 6));
    }

    #[test]
    fn interior_3x3_top_left_anchor() {
        // Anchor at (0, 0) means no left/top margin, but 2 pixels of right/bottom margin.
        let r = compute_interior_region(Size::new(10, 10), Size::new(3, 3), (0, 0)).unwrap();
        assert_eq!(r.left(), 0);
        assert_eq!(r.top(), 0);
        assert_eq!(r.right(), 8);
        assert_eq!(r.bottom(), 8);
        assert_eq!(r.size, Size::new(8, 8));
    }

    #[test]
    fn interior_3x3_bottom_right_anchor() {
        // Anchor at (2, 2) means 2 pixels of left/top margin, no right/bottom margin.
        let r = compute_interior_region(Size::new(10, 10), Size::new(3, 3), (2, 2)).unwrap();
        assert_eq!(r.left(), 2);
        assert_eq!(r.top(), 2);
        assert_eq!(r.right(), 10);
        assert_eq!(r.bottom(), 10);
        assert_eq!(r.size, Size::new(8, 8));
    }

    #[test]
    fn interior_1x1_kernel() {
        // 1×1 kernel, anchor at (0,0): the entire image is interior.
        let r = compute_interior_region(Size::new(5, 5), Size::new(1, 1), (0, 0)).unwrap();
        assert_eq!(r.left(), 0);
        assert_eq!(r.top(), 0);
        assert_eq!(r.size, Size::new(5, 5));
    }

    #[test]
    fn interior_kernel_equals_image() {
        // 5×5 kernel on 5×5 image, centered → only 1 interior position.
        let r = compute_interior_region(Size::new(5, 5), Size::new(5, 5), (2, 2)).unwrap();
        assert_eq!(r.left(), 2);
        assert_eq!(r.top(), 2);
        assert_eq!(r.size, Size::new(1, 1));
    }

    #[test]
    fn interior_kernel_larger_than_image() {
        assert!(compute_interior_region(Size::new(3, 3), Size::new(5, 5), (2, 2)).is_none());
    }

    #[test]
    fn interior_non_square_kernel() {
        // 5×3 kernel (w=5, h=3), anchor (2, 1) on 10×10 image.
        let r = compute_interior_region(Size::new(10, 10), Size::new(5, 3), (2, 1)).unwrap();
        assert_eq!(r.left(), 2);
        assert_eq!(r.top(), 1);
        assert_eq!(r.size, Size::new(6, 8));
    }

    #[test]
    fn interior_non_square_image() {
        let r = compute_interior_region(Size::new(20, 5), Size::new(3, 3), (1, 1)).unwrap();
        assert_eq!(r.left(), 1);
        assert_eq!(r.top(), 1);
        assert_eq!(r.size, Size::new(18, 3));
    }

    #[test]
    fn interior_1x1_image_1x1_kernel() {
        let r = compute_interior_region(Size::new(1, 1), Size::new(1, 1), (0, 0)).unwrap();
        assert_eq!(r.size, Size::new(1, 1));
    }

    #[test]
    fn interior_1x1_image_3x3_kernel() {
        assert!(compute_interior_region(Size::new(1, 1), Size::new(3, 3), (1, 1)).is_none());
    }

    #[test]
    fn interior_asymmetric_anchor() {
        // 5×5 kernel, anchor at (1, 3) on 10×10 image.
        // Left margin = 1, right margin = 3, top margin = 3, bottom margin = 1.
        let r = compute_interior_region(Size::new(10, 10), Size::new(5, 5), (1, 3)).unwrap();
        assert_eq!(r.left(), 1);
        assert_eq!(r.top(), 3);
        assert_eq!(r.right(), 7); // 10 - 3 = 7
        assert_eq!(r.bottom(), 9); // 10 - 1 = 9
        assert_eq!(r.size, Size::new(6, 6));
    }

    // ── Helper function tests ───────────────────────────────────────────

    #[test]
    fn reflect101_in_bounds() {
        for i in 0..5isize {
            assert_eq!(reflect101(i, 5), i as usize);
        }
    }

    #[test]
    fn reflect101_negative() {
        // len=4: [A B C D], period=6
        // x=-1 → 1, x=-2 → 2, x=-3 → 3
        assert_eq!(reflect101(-1, 4), 1);
        assert_eq!(reflect101(-2, 4), 2);
        assert_eq!(reflect101(-3, 4), 3);
    }

    #[test]
    fn reflect101_positive_overshoot() {
        // len=4: period=6
        // x=4 → 2, x=5 → 1, x=6 → 0, x=7 → 1
        assert_eq!(reflect101(4, 4), 2);
        assert_eq!(reflect101(5, 4), 1);
        assert_eq!(reflect101(6, 4), 0);
        assert_eq!(reflect101(7, 4), 1);
    }

    #[test]
    fn reflect101_len_1() {
        assert_eq!(reflect101(0, 1), 0);
        assert_eq!(reflect101(-1, 1), 0);
        assert_eq!(reflect101(1, 1), 0);
        assert_eq!(reflect101(-100, 1), 0);
        assert_eq!(reflect101(100, 1), 0);
    }

    #[test]
    fn reflect101_len_2() {
        // len=2: period=2, [A B]
        // x=-1 → 1, x=2 → 0, x=3 → 1
        assert_eq!(reflect101(-1, 2), 1);
        assert_eq!(reflect101(0, 2), 0);
        assert_eq!(reflect101(1, 2), 1);
        assert_eq!(reflect101(2, 2), 0);
        assert_eq!(reflect101(3, 2), 1);
    }

    #[test]
    fn reflect101_large_negative() {
        // len=4, period=6. x=-7 → -7 mod 6 = -1 → +6 = 5 → fold: 6-5=1
        assert_eq!(reflect101(-7, 4), 1);
    }

    #[test]
    fn wrap_coord_in_bounds() {
        for i in 0..5isize {
            assert_eq!(wrap_coord(i, 5), i as usize);
        }
    }

    #[test]
    fn wrap_coord_negative() {
        assert_eq!(wrap_coord(-1, 4), 3);
        assert_eq!(wrap_coord(-2, 4), 2);
        assert_eq!(wrap_coord(-4, 4), 0);
        assert_eq!(wrap_coord(-5, 4), 3);
    }

    #[test]
    fn wrap_coord_positive_overshoot() {
        assert_eq!(wrap_coord(4, 4), 0);
        assert_eq!(wrap_coord(5, 4), 1);
        assert_eq!(wrap_coord(7, 4), 3);
        assert_eq!(wrap_coord(8, 4), 0);
    }

    // ── Skip ────────────────────────────────────────────────────────────

    #[test]
    fn skip_output_region_3x3_on_10x10() {
        let img = Image::<u8>::zero(10, 10);
        let region =
            BorderPolicy::<Image<u8>>::output_region(&Skip, img.size(), Size::new(3, 3), (1, 1));
        assert_eq!(region.left(), 1);
        assert_eq!(region.top(), 1);
        assert_eq!(region.size, Size::new(8, 8));
    }

    #[test]
    fn skip_output_region_5x5_on_10x10() {
        let img = Image::<u8>::zero(10, 10);
        let region =
            BorderPolicy::<Image<u8>>::output_region(&Skip, img.size(), Size::new(5, 5), (2, 2));
        assert_eq!(region.left(), 2);
        assert_eq!(region.top(), 2);
        assert_eq!(region.size, Size::new(6, 6));
    }

    #[test]
    fn skip_output_region_kernel_larger_than_image() {
        let region = BorderPolicy::<Image<u8>>::output_region(
            &Skip,
            Size::new(3, 3),
            Size::new(5, 5),
            (2, 2),
        );
        assert_eq!(region.size, Size::new(0, 0));
    }

    #[test]
    fn skip_output_region_1x1_kernel() {
        let region = BorderPolicy::<Image<u8>>::output_region(
            &Skip,
            Size::new(5, 5),
            Size::new(1, 1),
            (0, 0),
        );
        assert_eq!(region.left(), 0);
        assert_eq!(region.top(), 0);
        assert_eq!(region.size, Size::new(5, 5));
    }

    #[test]
    #[should_panic(expected = "Skip")]
    fn skip_pixel_at_panics() {
        let img = Image::<u8>::zero(5, 5);
        Skip.pixel_at(&img, 0, 0);
    }

    // ── Clamp ───────────────────────────────────────────────────────────

    fn make_4x4_image() -> Image<u8> {
        Image::generate(4, 4, |x, y| (x + y * 4) as u8)
    }

    #[test]
    fn clamp_in_bounds() {
        let img = make_4x4_image();
        for y in 0..4isize {
            for x in 0..4isize {
                assert_eq!(
                    Clamp.pixel_at(&img, x, y),
                    img.pixel_at(x as usize, y as usize)
                );
            }
        }
    }

    #[test]
    fn clamp_negative_x() {
        let img = make_4x4_image();
        assert_eq!(Clamp.pixel_at(&img, -1, 0), img.pixel_at(0, 0));
        assert_eq!(Clamp.pixel_at(&img, -1, 2), img.pixel_at(0, 2));
        assert_eq!(Clamp.pixel_at(&img, -100, 1), img.pixel_at(0, 1));
    }

    #[test]
    fn clamp_negative_y() {
        let img = make_4x4_image();
        assert_eq!(Clamp.pixel_at(&img, 0, -1), img.pixel_at(0, 0));
        assert_eq!(Clamp.pixel_at(&img, 2, -1), img.pixel_at(2, 0));
        assert_eq!(Clamp.pixel_at(&img, 1, -50), img.pixel_at(1, 0));
    }

    #[test]
    fn clamp_positive_overshoot_x() {
        let img = make_4x4_image();
        assert_eq!(Clamp.pixel_at(&img, 4, 0), img.pixel_at(3, 0));
        assert_eq!(Clamp.pixel_at(&img, 4, 2), img.pixel_at(3, 2));
        assert_eq!(Clamp.pixel_at(&img, 100, 1), img.pixel_at(3, 1));
    }

    #[test]
    fn clamp_positive_overshoot_y() {
        let img = make_4x4_image();
        assert_eq!(Clamp.pixel_at(&img, 0, 4), img.pixel_at(0, 3));
        assert_eq!(Clamp.pixel_at(&img, 2, 100), img.pixel_at(2, 3));
    }

    #[test]
    fn clamp_corners() {
        let img = make_4x4_image();
        // top-left corner
        assert_eq!(Clamp.pixel_at(&img, -1, -1), img.pixel_at(0, 0));
        // top-right corner
        assert_eq!(Clamp.pixel_at(&img, 4, -1), img.pixel_at(3, 0));
        // bottom-left corner
        assert_eq!(Clamp.pixel_at(&img, -1, 4), img.pixel_at(0, 3));
        // bottom-right corner
        assert_eq!(Clamp.pixel_at(&img, 4, 4), img.pixel_at(3, 3));
    }

    #[test]
    fn clamp_output_region_full_image() {
        let region = BorderPolicy::<Image<u8>>::output_region(
            &Clamp,
            Size::new(10, 10),
            Size::new(3, 3),
            (1, 1),
        );
        assert_eq!(region.left(), 0);
        assert_eq!(region.top(), 0);
        assert_eq!(region.size, Size::new(10, 10));
    }

    #[test]
    fn clamp_1x1_image() {
        let img = Image::generate(1, 1, |_, _| 42u8);
        assert_eq!(Clamp.pixel_at(&img, 0, 0), 42);
        assert_eq!(Clamp.pixel_at(&img, -1, 0), 42);
        assert_eq!(Clamp.pixel_at(&img, 0, -1), 42);
        assert_eq!(Clamp.pixel_at(&img, 1, 0), 42);
        assert_eq!(Clamp.pixel_at(&img, 0, 1), 42);
        assert_eq!(Clamp.pixel_at(&img, -10, -10), 42);
        assert_eq!(Clamp.pixel_at(&img, 10, 10), 42);
    }

    // ── Mirror ──────────────────────────────────────────────────────────

    #[test]
    fn mirror_in_bounds() {
        let img = make_4x4_image();
        for y in 0..4isize {
            for x in 0..4isize {
                assert_eq!(
                    Mirror.pixel_at(&img, x, y),
                    img.pixel_at(x as usize, y as usize)
                );
            }
        }
    }

    #[test]
    fn mirror_negative_x() {
        let img = Image::generate(4, 1, |x, _| x as u8);
        // pixels: [0, 1, 2, 3]
        // x=-1 → 1, x=-2 → 2, x=-3 → 3
        assert_eq!(Mirror.pixel_at(&img, -1, 0), 1);
        assert_eq!(Mirror.pixel_at(&img, -2, 0), 2);
        assert_eq!(Mirror.pixel_at(&img, -3, 0), 3);
    }

    #[test]
    fn mirror_positive_overshoot_x() {
        let img = Image::generate(4, 1, |x, _| x as u8);
        // x=4 → 2, x=5 → 1, x=6 → 0
        assert_eq!(Mirror.pixel_at(&img, 4, 0), 2);
        assert_eq!(Mirror.pixel_at(&img, 5, 0), 1);
        assert_eq!(Mirror.pixel_at(&img, 6, 0), 0);
    }

    #[test]
    fn mirror_negative_y() {
        let img = Image::generate(1, 4, |_, y| y as u8);
        // pixels: [0, 1, 2, 3] (column)
        assert_eq!(Mirror.pixel_at(&img, 0, -1), 1);
        assert_eq!(Mirror.pixel_at(&img, 0, -2), 2);
        assert_eq!(Mirror.pixel_at(&img, 0, -3), 3);
    }

    #[test]
    fn mirror_positive_overshoot_y() {
        let img = Image::generate(1, 4, |_, y| y as u8);
        assert_eq!(Mirror.pixel_at(&img, 0, 4), 2);
        assert_eq!(Mirror.pixel_at(&img, 0, 5), 1);
        assert_eq!(Mirror.pixel_at(&img, 0, 6), 0);
    }

    #[test]
    fn mirror_corners() {
        let img = make_4x4_image();
        // (-1, -1): x→1, y→1
        assert_eq!(Mirror.pixel_at(&img, -1, -1), img.pixel_at(1, 1));
        // (4, -1): x→2, y→1
        assert_eq!(Mirror.pixel_at(&img, 4, -1), img.pixel_at(2, 1));
        // (-1, 4): x→1, y→2
        assert_eq!(Mirror.pixel_at(&img, -1, 4), img.pixel_at(1, 2));
        // (4, 4): x→2, y→2
        assert_eq!(Mirror.pixel_at(&img, 4, 4), img.pixel_at(2, 2));
    }

    #[test]
    fn mirror_1x1_image() {
        let img = Image::generate(1, 1, |_, _| 99u8);
        assert_eq!(Mirror.pixel_at(&img, -1, 0), 99);
        assert_eq!(Mirror.pixel_at(&img, 0, -1), 99);
        assert_eq!(Mirror.pixel_at(&img, 1, 0), 99);
        assert_eq!(Mirror.pixel_at(&img, 0, 1), 99);
        assert_eq!(Mirror.pixel_at(&img, -10, -10), 99);
        assert_eq!(Mirror.pixel_at(&img, 10, 10), 99);
    }

    #[test]
    fn mirror_2x1_image() {
        let img = Image::generate(2, 1, |x, _| x as u8);
        // pixels: [0, 1], period = 2
        assert_eq!(Mirror.pixel_at(&img, -1, 0), 1);
        assert_eq!(Mirror.pixel_at(&img, 2, 0), 0);
        assert_eq!(Mirror.pixel_at(&img, 3, 0), 1);
        assert_eq!(Mirror.pixel_at(&img, -2, 0), 0);
    }

    #[test]
    fn mirror_output_region_full_image() {
        let region = BorderPolicy::<Image<u8>>::output_region(
            &Mirror,
            Size::new(10, 10),
            Size::new(3, 3),
            (1, 1),
        );
        assert_eq!(region.left(), 0);
        assert_eq!(region.top(), 0);
        assert_eq!(region.size, Size::new(10, 10));
    }

    #[test]
    fn mirror_periodicity() {
        // Verify the reflection pattern is periodic.
        let img = Image::generate(5, 1, |x, _| x as u8);
        // period = 2*(5-1) = 8
        // The pattern should repeat every 8 positions.
        for offset in [-16isize, -8, 0, 8, 16] {
            for x in 0..5isize {
                assert_eq!(
                    Mirror.pixel_at(&img, x + offset, 0),
                    Mirror.pixel_at(&img, x, 0),
                    "Failed for x={}, offset={}",
                    x,
                    offset
                );
            }
        }
    }

    // ── Wrap ────────────────────────────────────────────────────────────

    #[test]
    fn wrap_in_bounds() {
        let img = make_4x4_image();
        for y in 0..4isize {
            for x in 0..4isize {
                assert_eq!(
                    Wrap.pixel_at(&img, x, y),
                    img.pixel_at(x as usize, y as usize)
                );
            }
        }
    }

    #[test]
    fn wrap_negative_x() {
        let img = Image::generate(4, 1, |x, _| x as u8);
        // pixels: [0, 1, 2, 3]
        assert_eq!(Wrap.pixel_at(&img, -1, 0), 3);
        assert_eq!(Wrap.pixel_at(&img, -2, 0), 2);
        assert_eq!(Wrap.pixel_at(&img, -3, 0), 1);
        assert_eq!(Wrap.pixel_at(&img, -4, 0), 0);
    }

    #[test]
    fn wrap_positive_overshoot_x() {
        let img = Image::generate(4, 1, |x, _| x as u8);
        assert_eq!(Wrap.pixel_at(&img, 4, 0), 0);
        assert_eq!(Wrap.pixel_at(&img, 5, 0), 1);
        assert_eq!(Wrap.pixel_at(&img, 6, 0), 2);
        assert_eq!(Wrap.pixel_at(&img, 7, 0), 3);
        assert_eq!(Wrap.pixel_at(&img, 8, 0), 0);
    }

    #[test]
    fn wrap_negative_y() {
        let img = Image::generate(1, 4, |_, y| y as u8);
        assert_eq!(Wrap.pixel_at(&img, 0, -1), 3);
        assert_eq!(Wrap.pixel_at(&img, 0, -4), 0);
    }

    #[test]
    fn wrap_corners() {
        let img = make_4x4_image();
        // (-1, -1) → (3, 3)
        assert_eq!(Wrap.pixel_at(&img, -1, -1), img.pixel_at(3, 3));
        // (4, -1) → (0, 3)
        assert_eq!(Wrap.pixel_at(&img, 4, -1), img.pixel_at(0, 3));
        // (-1, 4) → (3, 0)
        assert_eq!(Wrap.pixel_at(&img, -1, 4), img.pixel_at(3, 0));
        // (4, 4) → (0, 0)
        assert_eq!(Wrap.pixel_at(&img, 4, 4), img.pixel_at(0, 0));
    }

    #[test]
    fn wrap_1x1_image() {
        let img = Image::generate(1, 1, |_, _| 77u8);
        assert_eq!(Wrap.pixel_at(&img, -1, 0), 77);
        assert_eq!(Wrap.pixel_at(&img, 0, -1), 77);
        assert_eq!(Wrap.pixel_at(&img, 1, 0), 77);
        assert_eq!(Wrap.pixel_at(&img, 0, 1), 77);
        assert_eq!(Wrap.pixel_at(&img, -10, -10), 77);
    }

    #[test]
    fn wrap_output_region_full_image() {
        let region = BorderPolicy::<Image<u8>>::output_region(
            &Wrap,
            Size::new(10, 10),
            Size::new(3, 3),
            (1, 1),
        );
        assert_eq!(region.left(), 0);
        assert_eq!(region.top(), 0);
        assert_eq!(region.size, Size::new(10, 10));
    }

    #[test]
    fn wrap_periodicity() {
        let img = Image::generate(4, 1, |x, _| x as u8);
        // Verify true periodicity
        for offset in [-8isize, -4, 0, 4, 8] {
            for x in 0..4isize {
                assert_eq!(
                    Wrap.pixel_at(&img, x + offset, 0),
                    Wrap.pixel_at(&img, x, 0),
                    "Failed for x={}, offset={}",
                    x,
                    offset
                );
            }
        }
    }

    // ── Constant ────────────────────────────────────────────────────────

    #[test]
    fn constant_in_bounds() {
        let img = make_4x4_image();
        let policy = Constant(255u8);
        for y in 0..4isize {
            for x in 0..4isize {
                assert_eq!(
                    policy.pixel_at(&img, x, y),
                    img.pixel_at(x as usize, y as usize)
                );
            }
        }
    }

    #[test]
    fn constant_out_of_bounds_returns_constant() {
        let img = make_4x4_image();
        let policy = Constant(42u8);
        assert_eq!(policy.pixel_at(&img, -1, 0), 42);
        assert_eq!(policy.pixel_at(&img, 0, -1), 42);
        assert_eq!(policy.pixel_at(&img, 4, 0), 42);
        assert_eq!(policy.pixel_at(&img, 0, 4), 42);
        assert_eq!(policy.pixel_at(&img, -1, -1), 42);
        assert_eq!(policy.pixel_at(&img, 100, 100), 42);
    }

    #[test]
    fn constant_zero_padding() {
        let img = Image::generate(3, 3, |_, _| 100u8);
        let policy = Constant(0u8);
        assert_eq!(policy.pixel_at(&img, 1, 1), 100);
        assert_eq!(policy.pixel_at(&img, -1, 1), 0);
    }

    #[test]
    fn constant_1x1_image() {
        let img = Image::generate(1, 1, |_, _| 5u8);
        let policy = Constant(0u8);
        assert_eq!(policy.pixel_at(&img, 0, 0), 5);
        assert_eq!(policy.pixel_at(&img, -1, 0), 0);
        assert_eq!(policy.pixel_at(&img, 1, 0), 0);
        assert_eq!(policy.pixel_at(&img, 0, -1), 0);
        assert_eq!(policy.pixel_at(&img, 0, 1), 0);
    }

    #[test]
    fn constant_output_region_full_image() {
        let policy = Constant(0u8);
        let region = BorderPolicy::<Image<u8>>::output_region(
            &policy,
            Size::new(10, 10),
            Size::new(3, 3),
            (1, 1),
        );
        assert_eq!(region.left(), 0);
        assert_eq!(region.top(), 0);
        assert_eq!(region.size, Size::new(10, 10));
    }

    // ── Multi-channel pixel tests ───────────────────────────────────────

    #[test]
    fn clamp_with_rgb_pixel() {
        let img = Image::generate(3, 3, |x, y| [x as u8, y as u8, (x + y) as u8]);
        assert_eq!(Clamp.pixel_at(&img, -1, -1), [0, 0, 0]);
        assert_eq!(Clamp.pixel_at(&img, 1, 1), [1, 1, 2]);
        assert_eq!(Clamp.pixel_at(&img, 3, 3), [2, 2, 4]);
    }

    #[test]
    fn mirror_with_rgb_pixel() {
        let img = Image::generate(3, 3, |x, y| [x as u8, y as u8, (x + y) as u8]);
        assert_eq!(Mirror.pixel_at(&img, -1, 0), img.pixel_at(1, 0));
        assert_eq!(Mirror.pixel_at(&img, 3, 0), img.pixel_at(1, 0));
    }

    #[test]
    fn wrap_with_rgb_pixel() {
        let img = Image::generate(3, 3, |x, y| [x as u8, y as u8, (x + y) as u8]);
        assert_eq!(Wrap.pixel_at(&img, -1, 0), img.pixel_at(2, 0));
        assert_eq!(Wrap.pixel_at(&img, 3, 0), img.pixel_at(0, 0));
    }

    #[test]
    fn constant_with_rgb_pixel() {
        let img = Image::generate(3, 3, |x, y| [x as u8, y as u8, (x + y) as u8]);
        let policy = Constant([0u8, 0, 0]);
        assert_eq!(policy.pixel_at(&img, -1, 0), [0, 0, 0]);
        assert_eq!(policy.pixel_at(&img, 1, 1), [1, 1, 2]);
    }

    // ── f32 pixel tests ─────────────────────────────────────────────────

    #[test]
    fn clamp_with_f32_pixel() {
        let img = Image::generate(3, 3, |x, y| (x as f32 + y as f32 * 3.0));
        assert_eq!(Clamp.pixel_at(&img, -1, 0), img.pixel_at(0, 0));
        assert_eq!(Clamp.pixel_at(&img, 3, 2), img.pixel_at(2, 2));
    }

    #[test]
    fn constant_with_f32_pixel() {
        let img = Image::generate(3, 3, |x, y| (x as f32 + y as f32 * 3.0));
        let policy = Constant(0.0f32);
        assert_eq!(policy.pixel_at(&img, -1, 0), 0.0);
        assert_eq!(policy.pixel_at(&img, 1, 1), 4.0);
    }

    // ── Edge case: all policies agree on in-bounds access ───────────────

    #[test]
    fn all_policies_agree_in_bounds() {
        let img = make_4x4_image();
        let skip_region =
            BorderPolicy::<Image<u8>>::output_region(&Skip, img.size(), Size::new(3, 3), (1, 1));

        // For interior positions (where Skip says output exists), all policies
        // should return the same value as direct pixel access.
        for y in skip_region.top()..skip_region.bottom() {
            for x in skip_region.left()..skip_region.right() {
                let expected = img.pixel_at(x, y);
                let xi = x as isize;
                let yi = y as isize;
                assert_eq!(Clamp.pixel_at(&img, xi, yi), expected);
                assert_eq!(Mirror.pixel_at(&img, xi, yi), expected);
                assert_eq!(Wrap.pixel_at(&img, xi, yi), expected);
                assert_eq!(Constant(0u8).pixel_at(&img, xi, yi), expected);
            }
        }
    }

    // ── Consistency: extending policies return full-image output region ──

    #[test]
    fn extending_policies_full_output_region() {
        let size = Size::new(15, 20);
        let kernel = Size::new(5, 5);
        let anchor = (2, 2);

        let clamp_r = BorderPolicy::<Image<u8>>::output_region(&Clamp, size, kernel, anchor);
        let mirror_r = BorderPolicy::<Image<u8>>::output_region(&Mirror, size, kernel, anchor);
        let wrap_r = BorderPolicy::<Image<u8>>::output_region(&Wrap, size, kernel, anchor);
        let const_r =
            BorderPolicy::<Image<u8>>::output_region(&Constant(0u8), size, kernel, anchor);

        for r in [clamp_r, mirror_r, wrap_r, const_r] {
            assert_eq!(r.left(), 0);
            assert_eq!(r.top(), 0);
            assert_eq!(r.size, size);
        }
    }

    // ── Skip output region vs compute_interior_region consistency ────────

    #[test]
    fn skip_matches_compute_interior_region() {
        let cases = [
            (Size::new(10, 10), Size::new(3, 3), (1, 1)),
            (Size::new(10, 10), Size::new(5, 5), (2, 2)),
            (Size::new(10, 10), Size::new(1, 1), (0, 0)),
            (Size::new(5, 5), Size::new(5, 5), (2, 2)),
            (Size::new(20, 15), Size::new(7, 3), (3, 1)),
        ];

        for (img_size, kernel_size, anchor) in cases {
            let skip_region =
                BorderPolicy::<Image<u8>>::output_region(&Skip, img_size, kernel_size, anchor);
            match compute_interior_region(img_size, kernel_size, anchor) {
                Some(interior) => assert_eq!(skip_region, interior),
                None => assert_eq!(skip_region.size, Size::new(0, 0)),
            }
        }
    }

    // ── Debug/Clone/Copy trait tests ────────────────────────────────────

    #[test]
    fn policies_are_copy_clone_debug() {
        let s = Skip;
        let s2 = s; // Copy
        let s3 = s.clone();
        assert_eq!(s, s2);
        assert_eq!(s, s3);
        let _ = format!("{:?}", s);

        let c = Clamp;
        let c2 = c;
        let c3 = c.clone();
        assert_eq!(c, c2);
        assert_eq!(c, c3);
        let _ = format!("{:?}", c);

        let m = Mirror;
        let m2 = m;
        let m3 = m.clone();
        assert_eq!(m, m2);
        assert_eq!(m, m3);
        let _ = format!("{:?}", m);

        let w = Wrap;
        let w2 = w;
        let w3 = w.clone();
        assert_eq!(w, w2);
        assert_eq!(w, w3);
        let _ = format!("{:?}", w);

        let k = Constant(42u8);
        let k2 = k;
        let k3 = k.clone();
        assert_eq!(k, k2);
        assert_eq!(k, k3);
        let _ = format!("{:?}", k);
    }

    // ── Stress: large overshoot ─────────────────────────────────────────

    #[test]
    fn clamp_large_overshoot() {
        let img = make_4x4_image();
        assert_eq!(Clamp.pixel_at(&img, -1000, -1000), img.pixel_at(0, 0));
        assert_eq!(Clamp.pixel_at(&img, 1000, 1000), img.pixel_at(3, 3));
    }

    #[test]
    fn mirror_large_overshoot() {
        let img = Image::generate(4, 1, |x, _| x as u8);
        // Should not panic, and should return a valid pixel in [0, 3].
        let val = Mirror.pixel_at(&img, -1000, 0);
        assert!(val <= 3);
        let val = Mirror.pixel_at(&img, 1000, 0);
        assert!(val <= 3);
    }

    #[test]
    fn wrap_large_overshoot() {
        let img = Image::generate(4, 1, |x, _| x as u8);
        let val = Wrap.pixel_at(&img, -1000, 0);
        assert!(val <= 3);
        let val = Wrap.pixel_at(&img, 1000, 0);
        assert!(val <= 3);
    }

    #[test]
    fn constant_large_overshoot() {
        let img = make_4x4_image();
        let policy = Constant(0u8);
        assert_eq!(policy.pixel_at(&img, -1000, -1000), 0);
        assert_eq!(policy.pixel_at(&img, 1000, 1000), 0);
    }

    // ── Uniform API test: all policies through the trait ────────────────

    #[test]
    fn all_policies_through_trait_object() {
        // Verify that all policies can be used through a trait object (dyn).
        // This exercises the vtable path — not the primary usage (we expect
        // monomorphisation), but good to verify it compiles and works.
        let img = Image::generate(4, 4, |x, y| (x + y * 4) as u8);

        let policies: Vec<Box<dyn BorderPolicy<Image<u8>>>> = vec![
            Box::new(Clamp),
            Box::new(Mirror),
            Box::new(Wrap),
            Box::new(Constant(0u8)),
        ];

        for policy in &policies {
            // In-bounds should always match the image.
            assert_eq!(policy.pixel_at(&img, 1, 1), img.pixel_at(1, 1));
            // output_region should be the full image.
            let region = policy.output_region(img.size(), Size::new(3, 3), (1, 1));
            assert_eq!(region.size, img.size());
        }
    }

    // ── Interior region: percentage sanity check ────────────────────────

    #[test]
    fn interior_dominates_for_large_images() {
        // For a 1000×1000 image with a 5×5 kernel, ~99.2% should be interior.
        let img_size = Size::new(1000, 1000);
        let kernel_size = Size::new(5, 5);
        let anchor = (2, 2);

        let interior = compute_interior_region(img_size, kernel_size, anchor).unwrap();
        let interior_area = interior.area();
        let total_area = img_size.area();

        let ratio = interior_area as f64 / total_area as f64;
        assert!(
            ratio > 0.99,
            "Interior ratio should be >99%, got {:.2}%",
            ratio * 100.0
        );
    }
}

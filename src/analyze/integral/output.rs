//! The [`IntegralImage<A>`] output type — see ADR-0032 §4.

use std::ops::Sub;

use crate::Rectangle;
use crate::Size;
use crate::image::{Image, ImageRef, ImageView};
use crate::pixel::ZeroablePixel;

/// A summed-area table computed from a source image.
///
/// `IntegralImage<A>` is a newtype wrapper around an `Image<A>` of
/// dimensions `(W+1) × (H+1)` where `(W, H)` is the size of the
/// **source** image. The extra top row and left column are filled with
/// zero accumulators; together with the four-corner formula this lets
/// `region_sum` cover any sub-rectangle (including those touching the
/// image edges) with a single integer expression and no boundary
/// special-case.
///
/// `A` is the **accumulator pixel** — the user names it explicitly when
/// constructing the integral image (ADR-0032 §1). The valid
/// combinations of source and accumulator are gated at compile time by
/// the [`IntegralPixel`](crate::pixel::IntegralPixel) and
/// [`IntegralSquaredPixel`](crate::pixel::IntegralSquaredPixel) traits.
///
/// # What this type is *not*
///
/// `IntegralImage<A>` deliberately does **not** implement
/// [`ImageView`](crate::image::ImageView) (nor any of its descendants).
/// Passing an integral image where a regular image is expected is a
/// known bug class in other libraries (ADR-0032 §4); the distinct type
/// makes that mistake a compile error. Callers who genuinely need raw
/// access to the underlying `(W+1) × (H+1)` table go through
/// [`IntegralImage::as_table_view`] and acknowledge the offset
/// semantics by name.
///
/// # Examples
///
/// ```
/// use irys_cv::analyze::integral::{integral_image, IntegralImage};
/// use irys_cv::image::Image;
/// use irys_cv::pixel::{Mono8, Mono32};
/// use irys_cv::{Coordinate, Rectangle, Size};
///
/// let img: Image<Mono8> = Image::fill(4, 4, Mono8::new(10));
/// let sat: IntegralImage<Mono32> = integral_image::<_, Mono32>(&img)?;
/// assert_eq!(sat.source_size(), Size::new(4, 4));
///
/// // Full-image sum: 4 × 4 × 10.
/// let total = sat.region_sum(Rectangle::new(Coordinate::new(0, 0), Size::new(4, 4)));
/// assert_eq!(total, Mono32::new(160));
/// # Ok::<(), irys_cv::Error>(())
/// ```
#[derive(Clone)]
pub struct IntegralImage<A: Copy> {
    /// The `(W+1) × (H+1)` accumulator table. Row 0 and column 0 are
    /// all zero; the engine writes every other cell.
    data: Image<A>,
    /// The dimensions of the *source* image, cached so `region_sum`
    /// does not have to subtract from `data.size()` on every call.
    source: Size,
}

// Hand-rolled `Debug` / `PartialEq` because `Image<A>` itself doesn't
// implement them — deriving would emit unsatisfied trait bounds.
// `Clone` is straightforward (above) because `Image<A>` *does* clone
// when `A: Clone` (and `A: Copy` already implies `A: Clone`).
impl<A: Copy + std::fmt::Debug> std::fmt::Debug for IntegralImage<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use crate::image::ContiguousImage;
        f.debug_struct("IntegralImage")
            .field("source", &self.source)
            .field("table_size", &self.table_size())
            .field("data", &self.data.as_slice())
            .finish()
    }
}

impl<A: Copy + PartialEq> PartialEq for IntegralImage<A> {
    fn eq(&self, other: &Self) -> bool {
        use crate::image::ContiguousImage;
        self.source == other.source && self.data.as_slice() == other.data.as_slice()
    }
}

impl<A: Copy> IntegralImage<A> {
    /// Allocate a fresh integral image whose source dimensions are
    /// `source`. The underlying `(W+1) × (H+1)` table is filled with
    /// `A::zero()`.
    ///
    /// Crate-private — external code constructs `IntegralImage<A>`
    /// only via [`integral_image`](super::integral_image) and friends.
    ///
    /// # Panics — Tier 3 (`AGENTS.md`)
    ///
    /// Panics if `source.width + 1` or `source.height + 1` overflows
    /// `usize`. Such dimensions cannot correspond to an in-memory
    /// image and indicate a programmer bug upstream.
    #[inline]
    pub(crate) fn new_zero(source: Size) -> Self
    where
        A: ZeroablePixel,
    {
        let table_w = source.width.checked_add(1).unwrap_or_else(|| {
            panic!(
                "IntegralImage::new_zero: source.width + 1 overflows usize \
                 (source.width = {})",
                source.width
            )
        });
        let table_h = source.height.checked_add(1).unwrap_or_else(|| {
            panic!(
                "IntegralImage::new_zero: source.height + 1 overflows usize \
                 (source.height = {})",
                source.height
            )
        });
        let data = Image::<A>::zero(table_w, table_h);
        IntegralImage { data, source }
    }

    /// The dimensions of the *source* image (`W × H`).
    #[inline]
    pub fn source_size(&self) -> Size {
        self.source
    }

    /// The dimensions of the underlying summed-area table
    /// (`(W+1) × (H+1)`).
    ///
    /// Most callers want [`source_size`](Self::source_size) instead.
    /// This accessor exists for callers that go through
    /// [`as_table_view`](Self::as_table_view).
    #[inline]
    pub fn table_size(&self) -> Size {
        // `new_zero` already validated that both `+ 1`s fit in `usize`,
        // and the table is never resized, so a plain `+ 1` here is sound.
        Size::new(self.source.width + 1, self.source.height + 1)
    }

    /// Sum of all source pixels inside the half-open rectangle
    /// `[left, right) × [top, bottom)`.
    ///
    /// [`Rectangle`] is half-open by construction
    /// (`right = offset.x + size.width`, exclusive); this is also the
    /// exact shape the four-corner formula expects, so the lookups go
    /// directly into the `(W+1) × (H+1)` table with no `+1` adjustment.
    ///
    /// # Panics — Tier 3 (ADR-0025)
    ///
    /// Panics if `rect` is not fully contained in
    /// [`source_size`](Self::source_size). For the Tier 1 form that
    /// returns `None`, see [`get_region_sum`](Self::get_region_sum).
    #[inline]
    pub fn region_sum(&self, rect: Rectangle) -> A
    where
        A: Sub<Output = A>,
    {
        assert!(
            rect.right() <= self.source.width && rect.bottom() <= self.source.height,
            "IntegralImage::region_sum: rectangle {:?} out of bounds for source {:?}",
            rect,
            self.source,
        );
        self.region_sum_unchecked(rect)
    }

    /// Tier 1 form of [`region_sum`](Self::region_sum). Returns `None`
    /// if `rect` is not fully contained in
    /// [`source_size`](Self::source_size), **including** the case where
    /// `rect.right()` / `rect.bottom()` overflow `usize`.
    #[inline]
    #[must_use]
    pub fn get_region_sum(&self, rect: Rectangle) -> Option<A>
    where
        A: Sub<Output = A>,
    {
        let right = rect.checked_right()?;
        let bottom = rect.checked_bottom()?;
        if right <= self.source.width && bottom <= self.source.height {
            Some(self.region_sum_unchecked(rect))
        } else {
            None
        }
    }

    /// Borrow the underlying `(W+1) × (H+1)` table as a regular
    /// [`ImageRef<A>`](crate::image::ImageRef).
    ///
    /// Escape hatch for callers implementing custom queries on top of
    /// the raw accumulator table. Using it correctly requires
    /// understanding the `(W+1) × (H+1)` layout — the top row and left
    /// column are all zero and exist precisely to avoid a boundary
    /// special-case in the four-corner formula.
    #[inline]
    pub fn as_table_view(&self) -> ImageRef<'_, A> {
        // Image<A> always exposes a full-frame ImageRef of width × height.
        let w = self.data.width();
        let h = self.data.height();
        // SAFETY-FREE: data is W+1 × H+1 by construction (new_zero), and
        // the engine never resizes it.
        ImageRef::new(w, h, image_slice(&self.data)).expect("table data length matches dimensions")
    }

    // ── Crate-private hooks for the engine ────────────────────────────

    /// Mutable access to the underlying `(W+1) × (H+1)` table — used
    /// by the engine to fill in the cumulative sums. The top row and
    /// left column are pre-zeroed by [`new_zero`](Self::new_zero) and
    /// the engine deliberately does not touch them.
    #[inline]
    pub(crate) fn data_image_mut(&mut self) -> &mut Image<A> {
        &mut self.data
    }

    // ── Internals ─────────────────────────────────────────────────────

    /// Shared body of [`region_sum`] and [`get_region_sum`], after the
    /// caller has confirmed the rectangle is in bounds.
    ///
    /// # Evaluation order
    ///
    /// `(a − c) − (d − b)` per ADR-0032 §5. Important because integer
    /// accumulator pixels (`Mono32`, `Mono64`, `Rgb32`, `Rgb64`) use
    /// `Saturating<T>` arithmetic (ADR-0013). Both inner subtractions
    /// are partial-strip sums and therefore non-negative; the outer
    /// subtraction is the region sum and is also non-negative; so no
    /// intermediate underflows. Any other order is *correct* for
    /// wrapping `u32` but would saturate-to-zero spuriously on the
    /// `Saturating<T>` types this crate actually uses.
    #[inline]
    fn region_sum_unchecked(&self, rect: Rectangle) -> A
    where
        A: Sub<Output = A>,
    {
        let left = rect.left();
        let top = rect.top();
        let right = rect.right();
        let bottom = rect.bottom();

        // The (W+1, H+1) table means (x, y) in source coordinates maps
        // to (x, y) in table coordinates with no offset — the half-open
        // rectangle's `right` / `bottom` already index the "one past"
        // cell the formula wants.
        let a = self.data.pixel_at(right, bottom);
        let b = self.data.pixel_at(left, top);
        let c = self.data.pixel_at(right, top);
        let d = self.data.pixel_at(left, bottom);

        (a - c) - (d - b)
    }
}

/// Internal helper — borrow `Image<A>` as a flat slice for
/// `as_table_view`. Kept private to this module so the rest of the
/// crate does not grow accidental dependencies on `Image`'s internal
/// layout.
#[inline]
fn image_slice<A: Copy>(img: &Image<A>) -> &[A] {
    use crate::image::ContiguousImage;
    img.as_slice()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Coordinate;
    use crate::image::{ImageView, ImageViewMut};
    use crate::pixel::Mono32;

    /// Build a 4×4 `Mono8` integral image with every source pixel = 10.
    ///
    /// This is a constructor helper for the output-level tests; the
    /// real engine has its own coverage. The body manually fills the
    /// `(W+1, H+1)` accumulator table to keep these tests independent
    /// of the engine.
    fn fixture_const_10() -> IntegralImage<Mono32> {
        // Allocate via the crate-private constructor.
        let mut sat = IntegralImage::<Mono32>::new_zero(Size::new(4, 4));

        // Source is 4×4 of value 10; cumulative sum I[x, y] = 10 * x * y
        // for x, y in 0..=4 (with the row/column 0 all zero).
        let table = sat.data_image_mut();
        for y in 0..=4usize {
            for x in 0..=4usize {
                *table.pixel_at_mut(x, y) = Mono32::new((10 * x * y) as u32);
            }
        }
        sat
    }

    #[test]
    fn source_and_table_sizes() {
        let sat = fixture_const_10();
        assert_eq!(sat.source_size(), Size::new(4, 4));
        assert_eq!(sat.table_size(), Size::new(5, 5));
    }

    #[test]
    fn region_sum_corners() {
        // For a 4×4 image of constant value 10, region_sum over any
        // w×h sub-rectangle must equal w * h * 10.
        let sat = fixture_const_10();
        for y in 0..4 {
            for x in 0..4 {
                for h in 1..=(4 - y) {
                    for w in 1..=(4 - x) {
                        let rect = Rectangle::new(Coordinate::new(x, y), Size::new(w, h));
                        assert_eq!(
                            sat.region_sum(rect),
                            Mono32::new((w * h * 10) as u32),
                            "rect = {:?}",
                            rect,
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn region_sum_single_pixel() {
        let sat = fixture_const_10();
        let rect = Rectangle::new(Coordinate::new(2, 3), Size::new(1, 1));
        assert_eq!(sat.region_sum(rect), Mono32::new(10));
    }

    #[test]
    fn region_sum_full_image_no_plus_one_adjustment() {
        // Sanity check the half-open convention: a rectangle of size
        // equal to the source covers the whole image.
        let sat = fixture_const_10();
        let rect = Rectangle::new(Coordinate::new(0, 0), Size::new(4, 4));
        assert_eq!(sat.region_sum(rect), Mono32::new(4 * 4 * 10));
    }

    #[test]
    fn get_region_sum_returns_some_in_bounds() {
        let sat = fixture_const_10();
        let rect = Rectangle::new(Coordinate::new(1, 1), Size::new(2, 2));
        assert_eq!(sat.get_region_sum(rect), Some(Mono32::new(40)));
    }

    #[test]
    fn get_region_sum_returns_none_out_of_bounds() {
        let sat = fixture_const_10();
        // right = 5 > source.width = 4
        let rect = Rectangle::new(Coordinate::new(0, 0), Size::new(5, 4));
        assert_eq!(sat.get_region_sum(rect), None);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn region_sum_panics_out_of_bounds() {
        let sat = fixture_const_10();
        // bottom = 5 > source.height = 4
        let rect = Rectangle::new(Coordinate::new(0, 0), Size::new(4, 5));
        let _ = sat.region_sum(rect);
    }

    #[test]
    fn region_sum_evaluation_order_under_saturating() {
        // ADR-0032 §5: `(a - c) - (d - b)` keeps every intermediate
        // non-negative under Saturating arithmetic. To exercise the
        // path with values close to `u32::MAX / area`, build a small
        // Mono32 table where each source pixel is large.
        //
        // Source: 2×2 of value v = u32::MAX / 4; sum = v * 4 fits.
        let v = u32::MAX / 4;
        let mut sat = IntegralImage::<Mono32>::new_zero(Size::new(2, 2));
        let table = sat.data_image_mut();
        // Build cumulative sums manually.
        *table.pixel_at_mut(1, 1) = Mono32::new(v);
        *table.pixel_at_mut(2, 1) = Mono32::new(2 * v);
        *table.pixel_at_mut(1, 2) = Mono32::new(2 * v);
        *table.pixel_at_mut(2, 2) = Mono32::new(4 * v);

        // Whole-image sum.
        let rect = Rectangle::new(Coordinate::new(0, 0), Size::new(2, 2));
        assert_eq!(sat.region_sum(rect), Mono32::new(4 * v));

        // Partial: right column, both rows. Sum = 2v.
        let rect = Rectangle::new(Coordinate::new(1, 0), Size::new(1, 2));
        assert_eq!(sat.region_sum(rect), Mono32::new(2 * v));
    }

    #[test]
    fn as_table_view_reports_w_plus_one_h_plus_one() {
        let sat = fixture_const_10();
        let view = sat.as_table_view();
        assert_eq!(view.size(), Size::new(5, 5));
    }

    #[test]
    fn as_table_view_top_row_and_left_column_are_zero() {
        let sat = fixture_const_10();
        let view = sat.as_table_view();
        for x in 0..5 {
            assert_eq!(view.pixel_at(x, 0), Mono32::new(0), "top row at x={}", x);
        }
        for y in 0..5 {
            assert_eq!(
                view.pixel_at(0, y),
                Mono32::new(0),
                "left column at y={}",
                y
            );
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // P0-4 red-phase tests for integral-image overflow handling.
    //
    // 1) Table sizing must not silently wrap when the source dimensions
    //    are `usize::MAX` — the only correct behaviour is Tier-3 panic
    //    (programmer bug: such an image cannot exist in memory).
    //
    // 2) `get_region_sum` is the Tier-1 `Option` form. A rectangle whose
    //    `right()` / `bottom()` overflow must yield `None`, not panic.
    // ──────────────────────────────────────────────────────────────────────

    #[test]
    #[should_panic]
    fn integral_table_size_overflow_panics_tier3() {
        // source.width = usize::MAX => `width + 1` wraps to 0.
        let _ = IntegralImage::<Mono32>::new_zero(Size::new(usize::MAX, 1));
    }

    #[test]
    fn get_region_sum_overflowing_rect_returns_none() {
        let sat = fixture_const_10();
        // offset.x + size.width overflows usize.
        let rect = Rectangle::new(Coordinate::new(usize::MAX - 1, 0), Size::new(10, 1));
        assert_eq!(sat.get_region_sum(rect), None);

        // Same on the vertical axis.
        let rect = Rectangle::new(Coordinate::new(0, usize::MAX - 1), Size::new(1, 10));
        assert_eq!(sat.get_region_sum(rect), None);
    }
}

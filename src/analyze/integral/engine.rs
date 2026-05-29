//! Summed-area-table engines — `integral_image` and friends.
//!
//! Four public functions in two pairs: a `_into` form that writes into a
//! caller-supplied [`IntegralImage<A>`] and an allocating form that
//! returns a fresh one.
//!
//! # Why `RasterImage`, not `ContiguousImage`
//!
//! `RasterImage` is the narrowest trait that exposes a row slice. The
//! hot loop here reads every source pixel exactly once via
//! `image.row(y)`, which `RasterImage` provides for owned images,
//! `ImageRef` / `ImageRefMut`, `ImageArray`, *and* strided ROIs from
//! `SubView::roi`. Tightening the bound to `ContiguousImage` would lock
//! out strided ROIs, which must work; relaxing it to `ImageView` would
//! force a per-pixel `pixel_at` and lose the row-slice fast path.
//!
//! # Why the inner loop has no overflow check
//!
//! The pre-flight check in [`preflight`](super::preflight) is the
//! correctness gate: it guarantees that `W × H × max_integral_value()`
//! fits in the accumulator. Any intermediate sum the recurrence
//! produces is `≤` that bound, so no `Add` reaches saturation. The
//! evaluation order is chosen so no `Sub` underflows either.
//!
//! # The recurrence
//!
//! Standard summed-area-table recurrence:
//!
//! ```text
//! I[x, y] = src(x-1, y-1) + I[x-1, y] + I[x, y-1] − I[x-1, y-1]
//! ```
//!
//! Implemented as `s + ((left + up) − ul)`. The two intermediate
//! values are non-negative under `Saturating<T>`:
//!
//! - `left + up` is a sum of cumulative sums — by construction `≥ ul`,
//!   so the subtraction does not underflow.
//! - The final `s + (..)` cannot saturate because the pre-flight check
//!   bounded the *total* sum the recurrence ever produces.
//!
//! # `#[allow(private_bounds)]` on the public entry points
//!
//! The four public functions in this module have `IntegralCapacity` /
//! `IntegralSquaredCapacity` bounds. Those traits are deliberately
//! `pub(super)` (private to the module so they do not pollute the
//! public trait hierarchy; Philosophy §3 — bind on the tightest trait
//! that admits the operation, and nothing more). Promoting them to
//! `pub` would invite user impls that bypass the pre-flight overflow
//! gate — defeating its whole point. The `private_bounds` lint flags
//! this exposure; the allow is by design and applies to all four entry
//! points.

use std::ops::{Add, Sub};

use crate::Error;
use crate::image::{ImageView, ImageViewMut, RasterImage};
use crate::pixel::{IntegralPixel, IntegralSquaredPixel, ZeroablePixel};

use super::output::IntegralImage;
use super::preflight::{self, IntegralCapacity, IntegralSquaredCapacity};

/// Compute the summed-area table of `image` with accumulator pixel
/// `A`, allocating a fresh [`IntegralImage<A>`].
///
/// `A` is named explicitly by the caller (turbofish) — there is no
/// default.
///
/// # Errors — Tier 2
///
/// Returns [`Error::AccumulatorOverflow`] if the pre-flight check
/// determines that `A` is too narrow for an image of these dimensions.
/// The diagnostic data tells the caller exactly how much capacity is
/// needed and how much is available.
///
/// # Examples
///
/// ```
/// use fovea::analyze::integral::{integral_image, IntegralImage};
/// use fovea::image::Image;
/// use fovea::pixel::{Mono8, Mono32};
/// use fovea::{Coordinate, Rectangle, Size};
///
/// let img: Image<Mono8> = Image::fill(4, 4, Mono8::new(7));
/// let sat: IntegralImage<Mono32> = integral_image::<_, Mono32>(&img)?;
/// let rect = Rectangle::new(Coordinate::new(0, 0), Size::new(4, 4));
/// assert_eq!(sat.region_sum(rect), Mono32::new(4 * 4 * 7));
/// # Ok::<(), fovea::Error>(())
/// ```
#[allow(private_bounds)] // see module docs
#[must_use]
pub fn integral_image<I, A>(image: &I) -> Result<IntegralImage<A>, Error>
where
    I: RasterImage,
    I::Pixel: IntegralPixel<A>,
    A: Copy + ZeroablePixel + Add<Output = A> + Sub<Output = A> + IntegralCapacity,
{
    let mut out = IntegralImage::<A>::new_zero(image.size());
    integral_image_into(image, &mut out)?;
    Ok(out)
}

/// Compute the summed-area table of `image` into a caller-supplied
/// `out`. `out` must have been constructed with the same source size as
/// `image`.
///
/// # Errors — Tier 2
///
/// Returns [`Error::AccumulatorOverflow`] if the pre-flight check
/// fails.
///
/// # Panics — Tier 3
///
/// Panics if `out.source_size() != image.size()`. The output buffer's
/// dimensions are a caller-controlled precondition; a mismatch is a
/// programmer bug, not data-dependent failure.
#[allow(private_bounds)] // see module docs
pub fn integral_image_into<I, A>(image: &I, out: &mut IntegralImage<A>) -> Result<(), Error>
where
    I: RasterImage,
    I::Pixel: IntegralPixel<A>,
    A: Copy + ZeroablePixel + Add<Output = A> + Sub<Output = A> + IntegralCapacity,
{
    let w = image.width();
    let h = image.height();

    preflight::check::<I::Pixel, A>(w, h)?;

    assert_eq!(
        out.source_size(),
        image.size(),
        "integral_image_into: output source size does not match input",
    );

    fill_engine(image, out, |p| p.to_integral());
    Ok(())
}

/// Compute the **squared** summed-area table of `image` — the sum of
/// squared source values inside any rectangle.
///
/// Used for variance computation and normalised cross-correlation. The
/// per-pixel range is the *square* of the source range (Mono8 → 65 025
/// per pixel rather than 255), so the valid accumulators differ from
/// the non-squared variant.
///
/// # Errors — Tier 2
///
/// Returns [`Error::AccumulatorOverflow`] on pre-flight failure.
#[allow(private_bounds)] // see module docs
#[must_use]
pub fn integral_squared_image<I, A>(image: &I) -> Result<IntegralImage<A>, Error>
where
    I: RasterImage,
    I::Pixel: IntegralSquaredPixel<A>,
    A: Copy + ZeroablePixel + Add<Output = A> + Sub<Output = A> + IntegralSquaredCapacity,
{
    let mut out = IntegralImage::<A>::new_zero(image.size());
    integral_squared_image_into(image, &mut out)?;
    Ok(out)
}

/// Compute the squared summed-area table of `image` into a
/// caller-supplied `out`.
///
/// # Errors — Tier 2
///
/// Returns [`Error::AccumulatorOverflow`] on pre-flight failure.
///
/// # Panics — Tier 3
///
/// Panics if `out.source_size() != image.size()`.
#[allow(private_bounds)] // see module docs
pub fn integral_squared_image_into<I, A>(image: &I, out: &mut IntegralImage<A>) -> Result<(), Error>
where
    I: RasterImage,
    I::Pixel: IntegralSquaredPixel<A>,
    A: Copy + ZeroablePixel + Add<Output = A> + Sub<Output = A> + IntegralSquaredCapacity,
{
    let w = image.width();
    let h = image.height();

    preflight::check_squared::<I::Pixel, A>(w, h)?;

    assert_eq!(
        out.source_size(),
        image.size(),
        "integral_squared_image_into: output source size does not match input",
    );

    fill_engine(image, out, |p| p.to_integral_squared());
    Ok(())
}

// ── Shared engine body ────────────────────────────────────────────────

/// Fill the `(W+1, H+1)` accumulator table from `image`, mapping each
/// source pixel through `project` (either `to_integral` or
/// `to_integral_squared`).
///
/// Generic over the projection rather than over a trait choice so the
/// two engines compile to identical code modulo the per-pixel mapping.
/// Both callers have already verified the pre-flight check and the
/// output size match, so this function is infallible.
#[inline]
fn fill_engine<I, A, F>(image: &I, out: &mut IntegralImage<A>, project: F)
where
    I: RasterImage,
    A: Copy + ZeroablePixel + Add<Output = A> + Sub<Output = A>,
    F: Fn(I::Pixel) -> A,
{
    let w = image.width();
    let h = image.height();
    let table = out.data_image_mut();

    // The (W+1, H+1) layout means row 0 and column 0 of the table are
    // already zero (by IntegralImage::new_zero / the caller's
    // pre-zeroed buffer); we never touch them.
    //
    // Inner loop: I[x, y] = src(x-1, y-1) + I[x-1, y] + I[x, y-1] - I[x-1, y-1].
    // Evaluation order `s + ((left + up) - ul)` keeps every
    // intermediate non-negative under Saturating arithmetic.
    for y in 1..=h {
        let src_row = image.row(y - 1);
        for x in 1..=w {
            let s = project(src_row[x - 1]);
            let left = table.pixel_at(x - 1, y);
            let up = table.pixel_at(x, y - 1);
            let ul = table.pixel_at(x - 1, y - 1);
            let lu = left + up;
            let lu_minus_ul = lu - ul; // ≥ 0 because lu ⊇ ul
            *table.pixel_at_mut(x, y) = s + lu_minus_ul;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{Image, SubView};
    use crate::pixel::{Mono8, Mono16, Mono32, Mono64, MonoF32, MonoF64, Rgb8, Rgb32};
    use crate::{Coordinate, Rectangle, Size};

    #[test]
    fn mono8_to_mono32_4x4_constant_10() {
        let img: Image<Mono8> = Image::fill(4, 4, Mono8::new(10));
        let sat = integral_image::<_, Mono32>(&img).unwrap();

        // Every w×h sub-rectangle has sum w * h * 10.
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

    /// Naïve O(W²H²) reference: directly sum every pixel inside `rect`.
    fn naive_sum_mono(img: &Image<Mono8>, rect: Rectangle) -> u64 {
        let mut s = 0u64;
        for y in rect.top()..rect.bottom() {
            for x in rect.left()..rect.right() {
                s += img.pixel_at(x, y).value() as u64;
            }
        }
        s
    }

    #[test]
    fn mono8_to_mono64_matches_naive_for_deterministic_pattern() {
        // Deterministic pattern (no RNG — AGENTS.md "no external runtime deps").
        let img = Image::<Mono8>::generate(8, 6, |x, y| {
            Mono8::new(((x.wrapping_mul(37).wrapping_add(y.wrapping_mul(91))) & 0xFF) as u8)
        });
        let sat = integral_image::<_, Mono64>(&img).unwrap();

        // Compare every rectangle.
        for y in 0..6 {
            for x in 0..8 {
                for h in 1..=(6 - y) {
                    for w in 1..=(8 - x) {
                        let rect = Rectangle::new(Coordinate::new(x, y), Size::new(w, h));
                        let expected = naive_sum_mono(&img, rect);
                        assert_eq!(
                            sat.region_sum(rect),
                            Mono64::new(expected),
                            "rect = {:?}",
                            rect,
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn mono8_to_mono32_preflight_overflow_returns_err() {
        // 5000 × 5000 forces 255 × W × H = 6_375_000_000 > u32::MAX
        // (see preflight tests for the threshold derivation).
        let img = Image::<Mono8>::zero(5000, 5000);
        let err = integral_image::<_, Mono32>(&img).unwrap_err();
        let Error::AccumulatorOverflow {
            required_capacity,
            accumulator_capacity,
        } = err
        else {
            panic!("expected AccumulatorOverflow, got {:?}", err);
        };
        assert_eq!(required_capacity, 255u128 * 5000 * 5000);
        assert_eq!(accumulator_capacity, u32::MAX as u128);
    }

    #[test]
    fn monof32_to_monof64_matches_naive() {
        // Float path: deterministic linear pattern in [0, 1).
        let img =
            Image::<MonoF32>::generate(5, 4, |x, y| MonoF32::new(((x * 4 + y) as f32) / 32.0));
        let sat = integral_image::<_, MonoF64>(&img).unwrap();

        // Spot-check a few rectangles against direct summation.
        let rects = [
            Rectangle::new(Coordinate::new(0, 0), Size::new(5, 4)),
            Rectangle::new(Coordinate::new(1, 1), Size::new(3, 2)),
            Rectangle::new(Coordinate::new(2, 0), Size::new(2, 4)),
        ];
        for rect in rects {
            let mut expected = 0.0f64;
            for y in rect.top()..rect.bottom() {
                for x in rect.left()..rect.right() {
                    expected += img.pixel_at(x, y).value() as f64;
                }
            }
            let got = sat.region_sum(rect).value();
            assert!(
                (got - expected).abs() < 1e-9,
                "rect = {:?}, expected = {}, got = {}",
                rect,
                expected,
                got,
            );
        }
    }

    #[test]
    fn rgb8_to_rgb32_per_channel() {
        // Channels filled with different constants → per-channel sums
        // are independent.
        let img: Image<Rgb8> = Image::fill(3, 3, Rgb8::new(2, 5, 11));
        let sat = integral_image::<_, Rgb32>(&img).unwrap();
        let rect = Rectangle::new(Coordinate::new(0, 0), Size::new(3, 3));
        let s = sat.region_sum(rect);
        assert_eq!(s.r.0, 2 * 9);
        assert_eq!(s.g.0, 5 * 9);
        assert_eq!(s.b.0, 11 * 9);
    }

    #[test]
    #[should_panic(expected = "source size does not match")]
    fn integral_image_into_size_mismatch_panics() {
        let img: Image<Mono8> = Image::fill(4, 4, Mono8::new(1));
        let mut out = IntegralImage::<Mono32>::new_zero(Size::new(5, 5));
        let _ = integral_image_into(&img, &mut out);
    }

    #[test]
    fn integral_squared_mono8_to_mono64_3x3_constant() {
        let img: Image<Mono8> = Image::fill(3, 3, Mono8::new(4));
        let sat = integral_squared_image::<_, Mono64>(&img).unwrap();
        // Sum of squares = 9 × 16 = 144.
        let rect = Rectangle::new(Coordinate::new(0, 0), Size::new(3, 3));
        assert_eq!(sat.region_sum(rect), Mono64::new(144));
    }

    #[test]
    fn integral_squared_mono8_partial_rectangles() {
        // Distinct values so per-rectangle sums differ.
        let img = Image::<Mono8>::generate(4, 3, |x, y| Mono8::new((x + y + 1) as u8));
        let sat = integral_squared_image::<_, Mono64>(&img).unwrap();
        // Spot-check: top-left 2×2 contains {1, 2, 2, 3}, squared sum = 1+4+4+9 = 18.
        let rect = Rectangle::new(Coordinate::new(0, 0), Size::new(2, 2));
        assert_eq!(sat.region_sum(rect), Mono64::new(18));
    }

    #[test]
    fn subview_input_carries_through() {
        // Step 8.9: confirm RasterImage bound admits strided ROIs.
        let outer = Image::<Mono8>::generate(6, 6, |x, y| Mono8::new((x + y * 6 + 1) as u8));
        let rect = Rectangle::new(Coordinate::new(1, 1), Size::new(4, 4));
        let view = outer.roi(rect).expect("roi in bounds");

        let sat = integral_image::<_, Mono64>(&view).unwrap();
        assert_eq!(sat.source_size(), Size::new(4, 4));

        // Top-left 1×1 of the SubView is outer.pixel_at(1, 1) = 1+6+1 = 8.
        let single = Rectangle::new(Coordinate::new(0, 0), Size::new(1, 1));
        assert_eq!(sat.region_sum(single), Mono64::new(8));
    }

    #[test]
    fn integral_image_into_reuses_buffer() {
        // Calling `_into` twice with different inputs of the same size
        // must produce the second input's table — the engine overwrites
        // every non-zero cell.
        let mut buf = IntegralImage::<Mono32>::new_zero(Size::new(3, 3));

        let a: Image<Mono8> = Image::fill(3, 3, Mono8::new(1));
        integral_image_into(&a, &mut buf).unwrap();
        let r = Rectangle::new(Coordinate::new(0, 0), Size::new(3, 3));
        assert_eq!(buf.region_sum(r), Mono32::new(9));

        let b: Image<Mono8> = Image::fill(3, 3, Mono8::new(5));
        integral_image_into(&b, &mut buf).unwrap();
        assert_eq!(buf.region_sum(r), Mono32::new(45));
    }

    /// Compile-fail style check expressed as a positive type-level
    /// fixture: `Mono8` does **not** implement `IntegralPixel<Mono8>`,
    /// so the body of this function is only valid for accumulators
    /// where an impl exists. The mere presence of a successful build
    /// confirms the gate.
    ///
    /// (A full compile-fail UI test belongs in `tests/ui/`; this
    /// hand-rolled fixture documents the property in the same file as
    /// the engine.)
    #[allow(dead_code)]
    fn type_gate_compile_time_fixture(img: &Image<Mono8>) {
        let _: IntegralImage<Mono32> = integral_image::<_, Mono32>(img).unwrap();
        let _: IntegralImage<Mono64> = integral_image::<_, Mono64>(img).unwrap();
        let _: IntegralImage<MonoF64> = integral_image::<_, MonoF64>(img).unwrap();
        // Uncommenting the next line is intended to fail to compile
        // because `Mono8: IntegralPixel<Mono8>` is not implemented:
        //
        //     let _: IntegralImage<Mono8> = integral_image::<_, Mono8>(img).unwrap();
    }

    // Silence unused-import warnings on Mono16 — kept for symmetry with
    // the impl-coverage matrix; future tests will exercise it.
    #[allow(dead_code)]
    fn _touch_mono16(_: Mono16) {}
}

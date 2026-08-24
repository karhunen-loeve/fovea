//! Local-mean (adaptive) threshold.
//!
//! See the [module docs](super) for where thresholding lives in the
//! crate. This file holds the [`adaptive_threshold`] /
//! [`adaptive_threshold_into`] pair, the [`Bias`] offset newtype, the
//! sealed [`AdaptiveAccumulator`] trait, and their tests.
//!
//! The design — `Bias<A>` keyed on the integral *accumulator* (not its
//! unsigned channel), integer-exact `i128` comparison, clipped-window
//! edges, strict `>` boundary — is explained inline with each item below.

use std::ops::Sub;

use crate::Error;
use crate::analyze::integral::{IntegralImage, integral_image};
use crate::image::{BinaryImage, ImageView, RasterImage, RasterImageMut};
use crate::pixel::{IntegralPixel, Mono32, Mono64, MonoF64};
use crate::{Coordinate, OddWindowSide, Rectangle, Size};

mod sealed {
    /// Seals [`AdaptiveAccumulator`](super::AdaptiveAccumulator): the set
    /// of accumulators adaptive threshold accepts is closed, mirroring the
    /// sealed capacity traits of the integral engine.
    pub trait Sealed {}
    impl Sealed for crate::pixel::Mono32 {}
    impl Sealed for crate::pixel::Mono64 {}
    impl Sealed for crate::pixel::MonoF64 {}
}

/// The integral-image accumulators that [`adaptive_threshold`] accepts,
/// each paired with the signed offset domain its [`Bias`] uses.
///
/// This trait is **sealed** — it is implemented exactly for the
/// single-channel integral accumulators `Mono32`, `Mono64`, and `MonoF64`
/// and cannot be implemented downstream. Because those are the *only*
/// accumulators it admits, and `IntegralPixel<Mono32/64/F64>` is
/// implemented only for monochrome sources, a multi-channel (RGB) input
/// simply fails to compile — single-channel-ness is enforced by the type
/// system rather than a runtime assert (contrast
/// [`hysteresis_threshold`](super::hysteresis_threshold)).
///
/// The associated [`Offset`](Self::Offset) is the signed bias domain:
/// `i64` for the integer accumulators (`Mono32`, `Mono64`) and `f64` for
/// `MonoF64`. The offset is keyed on the accumulator rather than on its
/// (unsigned) channel type, which could not represent a negative bias.
pub trait AdaptiveAccumulator: sealed::Sealed + Copy + Sub<Output = Self> {
    /// Signed offset domain for [`Bias<Self>`](Bias): `i64` for integer
    /// accumulators, `f64` for `MonoF64`.
    type Offset: Copy + core::fmt::Debug + PartialEq;

    /// Build the summed-area table of `image` with `Self` as the
    /// accumulator. Delegates to
    /// [`integral_image`](crate::analyze::integral::integral_image), so it
    /// inherits the `O(1)` pre-flight overflow check
    /// ([`Error::AccumulatorOverflow`]). Keeping the call inside each
    /// concrete impl is what lets the public `adaptive_threshold` discharge
    /// the integral engine's crate-private capacity bound without naming it.
    #[doc(hidden)]
    fn integral_of<I>(image: &I) -> Result<IntegralImage<Self>, Error>
    where
        I: RasterImage,
        I::Pixel: IntegralPixel<Self>;

    /// The hot-loop decision for one pixel: `(pixel + offset) * area > sum`,
    /// evaluated in a domain wide enough to be exact (`i128` for the
    /// integer accumulators — `pixel * area` can reach `u64::MAX`, beyond
    /// `i64`; `f64` for `MonoF64`). `pixel` is the source value already
    /// projected into the accumulator via
    /// [`IntegralPixel::to_integral`]; `sum` is the clipped window's
    /// `region_sum`; `area` is the clipped pixel count. Equality is
    /// **background** (strict `>`).
    #[doc(hidden)]
    fn exceeds_local_mean(pixel: Self, sum: Self, area: u64, offset: Self::Offset) -> bool;
}

impl AdaptiveAccumulator for Mono32 {
    type Offset = i64;

    #[inline]
    fn integral_of<I>(image: &I) -> Result<IntegralImage<Self>, Error>
    where
        I: RasterImage,
        I::Pixel: IntegralPixel<Self>,
    {
        integral_image::<I, Self>(image)
    }

    #[inline]
    fn exceeds_local_mean(pixel: Self, sum: Self, area: u64, offset: i64) -> bool {
        // Exact integer comparison in i128: pixel.value() ≤ u32::MAX and
        // area ≤ W·H, so the product stays far inside i128.
        let p = pixel.value() as i128;
        let s = sum.value() as i128;
        (p + offset as i128) * (area as i128) > s
    }
}

impl AdaptiveAccumulator for Mono64 {
    type Offset = i64;

    #[inline]
    fn integral_of<I>(image: &I) -> Result<IntegralImage<Self>, Error>
    where
        I: RasterImage,
        I::Pixel: IntegralPixel<Self>,
    {
        integral_image::<I, Self>(image)
    }

    #[inline]
    fn exceeds_local_mean(pixel: Self, sum: Self, area: u64, offset: i64) -> bool {
        // i128 is mandatory here: pixel.value() can reach u64::MAX, whose
        // product with `area` overflows i64.
        let p = pixel.value() as i128;
        let s = sum.value() as i128;
        (p + offset as i128) * (area as i128) > s
    }
}

impl AdaptiveAccumulator for MonoF64 {
    type Offset = f64;

    #[inline]
    fn integral_of<I>(image: &I) -> Result<IntegralImage<Self>, Error>
    where
        I: RasterImage,
        I::Pixel: IntegralPixel<Self>,
    {
        integral_image::<I, Self>(image)
    }

    #[inline]
    fn exceeds_local_mean(pixel: Self, sum: Self, area: u64, offset: f64) -> bool {
        let p = pixel.value();
        let s = sum.value();
        (p + offset) * (area as f64) > s
    }
}

/// A signed bias on the local mean, in the domain of the integral
/// accumulator `A`.
///
/// `Bias<A>` shifts the per-pixel threshold: a pixel is foreground iff
/// `pixel > local_mean − bias`. **Positive** bias lowers the threshold and
/// thus biases toward **foreground**; **negative** bias biases toward
/// **background**. A `Bias::new(0)` makes the decision a strict
/// `pixel > local_mean`.
///
/// The wrapped value is `A::Offset` — `i64` for the integer accumulators
/// (`Mono32`, `Mono64`) and `f64` for `MonoF64`. The newtype exists to
/// keep the sign explicit and to stop the offset being accidentally
/// transposed with `adaptive_threshold`'s window argument, which carries
/// its own type ([`OddWindowSide`]) for the same reason.
///
/// # Examples
///
/// ```
/// use fovea::analyze::threshold::{adaptive_threshold, Bias};
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::{Mono8, Mono32};
/// use fovea::window;
///
/// let img = Image::fill(5, 5, Mono8::new(100));
/// let window = window!(3);
/// // Zero bias: a pixel equal to its local mean is background (strict `>`).
/// let mask = adaptive_threshold::<_, Mono32>(&img, window, Bias::new(0)).unwrap();
/// assert!(!mask.pixel_at(2, 2));
/// // Positive bias lowers the threshold → foreground.
/// let mask = adaptive_threshold::<_, Mono32>(&img, window, Bias::new(1)).unwrap();
/// assert!(mask.pixel_at(2, 2));
/// ```
pub struct Bias<A: AdaptiveAccumulator>(A::Offset);

impl<A: AdaptiveAccumulator> Bias<A> {
    /// Wrap a signed offset value in the accumulator's bias domain.
    #[inline]
    pub fn new(offset: A::Offset) -> Self {
        Bias(offset)
    }

    /// The wrapped offset value.
    #[inline]
    pub fn get(self) -> A::Offset {
        self.0
    }
}

// Hand-written (not derived) so the bounds land on `A::Offset` — which is
// always `Copy + Debug + PartialEq` per the trait — rather than on `A`.
impl<A: AdaptiveAccumulator> Clone for Bias<A> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<A: AdaptiveAccumulator> Copy for Bias<A> {}
impl<A: AdaptiveAccumulator> core::fmt::Debug for Bias<A> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("Bias").field(&self.0).finish()
    }
}
impl<A: AdaptiveAccumulator> PartialEq for Bias<A> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

/// Local-mean adaptive threshold over a `window × window` neighbourhood.
///
/// A pixel is foreground iff it exceeds the mean of its local window,
/// shifted by `offset`:
///
/// > foreground iff `pixel > local_mean(window) − bias`
///
/// This is the standard remedy for uneven illumination, vignetting, and
/// reflections that defeat a single global threshold (e.g. Otsu): the cut
/// adapts per pixel to local brightness. The output is a [`BinaryImage`].
///
/// The accumulator `A` is named explicitly with a turbofish, exactly as
/// for [`integral_image`](crate::analyze::integral::integral_image) — pick
/// `Mono32` for small 8-bit images, `Mono64` for large or wide-bit-depth
/// images, `MonoF64` for float input. `A` also fixes the
/// [`Bias`] domain (`i64` for integer accumulators, `f64` for `MonoF64`).
///
/// `window` is the side of the square neighbourhood, carried by
/// [`OddWindowSide`] so that the "odd and non-zero" requirement (an even
/// window has no centre pixel to threshold) is settled where the value is
/// written rather than on entry here.
///
/// # Algorithm
///
/// Builds one summed-area table ([`integral_image`](crate::analyze::integral::integral_image)),
/// then for each pixel takes the `O(1)` window sum and tests
/// `(pixel + offset) * area > sum` in an exact integer (`i128`) / `f64`
/// domain — no per-pixel division and no rounding. Total cost is `O(n)`,
/// independent of `window`.
///
/// # Edges — clipped window
///
/// Near the border the window is **clipped** to the image and the mean is
/// taken over the true (smaller) clipped pixel count. This is exact and
/// allocation-free, and matches scikit-image's `threshold_local`. It
/// **differs** from OpenCV's `adaptiveThreshold`, which replicates the
/// border; callers needing OpenCV parity should pre-pad.
///
/// # Boundary is strict `>`
///
/// Equality (`(pixel + offset) * area == sum`) is **background**,
/// consistent with the `>`-for-foreground convention used by Otsu in this
/// crate.
///
/// # Single channel only — enforced at compile time
///
/// `A` is one of `Mono32` / `Mono64` / `MonoF64`, and `IntegralPixel` into
/// those is implemented only for monochrome sources, so a multi-channel
/// input does not compile. Reduce RGB to one channel first.
///
/// # Errors — Tier 2
///
/// Returns [`Error::AccumulatorOverflow`] if `A` is too narrow for an
/// image of these dimensions — the same data-dependent failure surfaced by
/// the integral pre-flight. Choose a wider accumulator.
///
/// # Panics
///
/// Never, on the parameters: the window's invariant lives in [`OddWindowSide`]
/// and the bias domain in [`Bias`], so this function has no parameter
/// precondition left to violate.
///
/// # Examples
///
/// ```
/// use fovea::analyze::threshold::{adaptive_threshold, Bias};
/// use fovea::image::{Image, ImageView, ImageViewMut};
/// use fovea::pixel::{Mono8, Mono32};
/// use fovea::window;
///
/// // A flat field (value 50) with one locally bright spot (90).
/// let mut img = Image::fill(7, 3, Mono8::new(50));
/// *img.pixel_at_mut(3, 1) = Mono8::new(90);
///
/// let mask = adaptive_threshold::<_, Mono32>(&img, window!(3), Bias::new(0)).unwrap();
/// assert!(mask.pixel_at(3, 1));   // brighter than its local mean
/// assert!(!mask.pixel_at(0, 0));  // flat field → equals local mean
/// ```
pub fn adaptive_threshold<I, A>(
    image: &I,
    window: OddWindowSide,
    offset: Bias<A>,
) -> Result<BinaryImage, Error>
where
    I: RasterImage,
    I::Pixel: IntegralPixel<A>,
    A: AdaptiveAccumulator,
{
    // Owned variant allocates the output and delegates, matching the
    // `integral_image` / `hysteresis_threshold` convention.
    let mut out = BinaryImage::fill(image.width(), image.height(), false);
    adaptive_threshold_into(image, window, offset, &mut out)?;
    Ok(out)
}

/// As [`adaptive_threshold`], writing into a caller-owned mask.
///
/// Reusing `out` across frames avoids reallocating the output per call
/// (e.g. a per-frame threshold over video). The summed-area table is
/// unavoidable internal scratch and is still allocated per call.
///
/// # Errors — Tier 2
///
/// Returns [`Error::AccumulatorOverflow`] on pre-flight failure (see
/// [`adaptive_threshold`]).
///
/// # Panics — Tier 3
///
/// Panics if `out.size() != image.size()` (the caller allocated the mask
/// from sizes in hand).
pub fn adaptive_threshold_into<I, A>(
    image: &I,
    window: OddWindowSide,
    offset: Bias<A>,
    out: &mut BinaryImage,
) -> Result<(), Error>
where
    I: RasterImage,
    I::Pixel: IntegralPixel<A>,
    A: AdaptiveAccumulator,
{
    assert_eq!(
        out.size(),
        image.size(),
        "adaptive_threshold_into: output size {:?} does not match input {:?}",
        out.size(),
        image.size()
    );

    let w = image.width();
    let h = image.height();

    // One summed-area table; this is where the Tier 2 pre-flight runs.
    let sat = A::integral_of(image)?;
    // Exact for an odd side, which is the invariant `OddWindowSide` carries.
    let half = window.radius();
    let off = offset.0;

    for y in 0..h {
        let src_row = image.row(y);
        let out_row = out.row_mut(y);
        // Window rows, clipped to the image (Decision: clipped, not replicate).
        let top = y.saturating_sub(half);
        let bottom = (y + half + 1).min(h);
        for x in 0..w {
            let left = x.saturating_sub(half);
            let right = (x + half + 1).min(w);
            let rect = Rectangle::new(
                Coordinate::new(left, top),
                Size::new(right - left, bottom - top),
            );
            let sum = sat.region_sum(rect);
            let area = ((right - left) * (bottom - top)) as u64;
            let pixel = src_row[x].to_integral();
            out_row[x] = A::exceeds_local_mean(pixel, sum, area, off);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{Image, ImageView, ImageViewMut};
    use crate::pixel::{Mono8, MonoF32};

    /// Shorthand for the window literal each behaviour test pins.
    fn win(side: usize) -> OddWindowSide {
        OddWindowSide::new(side).unwrap()
    }

    /// Collect the `true` pixel coordinates of a mask into a sorted set.
    fn set_true(m: &BinaryImage) -> std::collections::BTreeSet<(usize, usize)> {
        let mut s = std::collections::BTreeSet::new();
        for y in 0..m.height() {
            for x in 0..m.width() {
                if m.pixel_at(x, y) {
                    s.insert((x, y));
                }
            }
        }
        s
    }

    /// Brute-force per-pixel reference: clipped window, exact integer mean.
    /// Mirrors the production decision `(pixel + offset) * area > sum`.
    fn naive(img: &Image<Mono8>, window: usize, offset: i64) -> BinaryImage {
        let w = img.width();
        let h = img.height();
        let half = window / 2;
        Image::generate(w, h, |x, y| {
            let left = x.saturating_sub(half);
            let top = y.saturating_sub(half);
            let right = (x + half + 1).min(w);
            let bottom = (y + half + 1).min(h);
            let mut sum = 0i128;
            let mut area = 0i128;
            for yy in top..bottom {
                for xx in left..right {
                    sum += img.pixel_at(xx, yy).value() as i128;
                    area += 1;
                }
            }
            let p = img.pixel_at(x, y).value() as i128;
            (p + offset as i128) * area > sum
        })
    }

    // ── Behaviours (one test per TDD-list item) ──────────────────────────────

    #[test]
    fn uniform_image_with_zero_offset_all_background() {
        // Flat image: every pixel equals its local mean, so the strict `>`
        // is false everywhere — pins the no-rounding property too.
        let img = Image::fill(5, 5, Mono8::new(100));
        let out = adaptive_threshold::<_, Mono32>(&img, win(3), Bias::new(0)).unwrap();
        assert!(set_true(&out).is_empty());
    }

    #[test]
    fn uniform_image_positive_offset_all_foreground() {
        // Positive bias lowers the threshold below the (equal) local mean,
        // so every pixel clears it. (The plan's TDD list mislabels this as
        // "negative_offset"; that contradicts the plan's own formula
        // `(pixel + offset) * area > sum` and prose "positive offset biases
        // toward foreground". This implementation follows the formula.)
        let img = Image::fill(5, 5, Mono8::new(100));
        let out = adaptive_threshold::<_, Mono32>(&img, win(3), Bias::new(1)).unwrap();
        for y in 0..out.height() {
            for x in 0..out.width() {
                assert!(out.pixel_at(x, y), "({x},{y}) should be foreground");
            }
        }
    }

    #[test]
    fn step_illumination_gradient() {
        // The motivating case. A dark half (50) and a bright half (150)
        // defeat any single global threshold; a small spot (90) sits in the
        // dark half — brighter than its dark surroundings but darker than
        // the bright half, so a global cut either misses it or floods the
        // bright half. Adaptive isolates it on local contrast.
        let mut img = Image::generate(8, 3, |x, _| Mono8::new(if x < 4 { 50 } else { 150 }));
        *img.pixel_at_mut(1, 1) = Mono8::new(90);

        let out = adaptive_threshold::<_, Mono32>(&img, win(3), Bias::new(0)).unwrap();
        assert!(out.pixel_at(1, 1), "local spot must be foreground");
        // Bright flat interior: pixel equals its local mean → background,
        // even though it is the brightest region in the image.
        assert!(
            !out.pixel_at(6, 1),
            "flat bright interior must be background"
        );
    }

    #[test]
    fn single_pixel_window_equals_self_threshold() {
        // window == 1 ⇒ the window is the pixel itself, mean == pixel.
        // Degenerate but well-defined: zero bias is background everywhere,
        // a positive bias is foreground everywhere, regardless of content.
        let img = Image::generate(4, 4, |x, y| Mono8::new((x * 16 + y * 4) as u8));
        let bg = adaptive_threshold::<_, Mono32>(&img, win(1), Bias::new(0)).unwrap();
        assert!(set_true(&bg).is_empty());
        let fg = adaptive_threshold::<_, Mono32>(&img, win(1), Bias::new(1)).unwrap();
        assert_eq!(set_true(&fg).len(), 16);
    }

    #[test]
    fn window_larger_than_image_clamps() {
        // window (9) larger than the 3×3 image ⇒ every window clamps to the
        // full image, so the threshold is the global mean. Values 1..=9 have
        // global mean 5 (sum 45 / area 9); foreground iff pixel > 5.
        let img = Image::generate(3, 3, |x, y| Mono8::new((y * 3 + x + 1) as u8));
        let out = adaptive_threshold::<_, Mono32>(&img, win(9), Bias::new(0)).unwrap();
        let expected: std::collections::BTreeSet<_> = (0..3)
            .flat_map(|y| (0..3).map(move |x| (x, y)))
            .filter(|&(x, y)| (y * 3 + x + 1) > 5)
            .collect();
        assert_eq!(set_true(&out), expected);
    }

    #[test]
    fn border_pixels_use_clipped_window() {
        // The corner (0,0) sees only its 2×2 window (area 4), not 3×3.
        // Cells: P=10 at (0,0); the other three sum to S=50 (20,20,10).
        // Correct (area 4): mean = 60/4 = 15, so 10 < 15 → background.
        // A bug dividing by 9 would give mean 6.7 → 10 > 6.7 → foreground.
        // Asserting background pins the clipped area divisor at 4.
        let mut img = Image::fill(4, 4, Mono8::new(0));
        *img.pixel_at_mut(0, 0) = Mono8::new(10);
        *img.pixel_at_mut(1, 0) = Mono8::new(20);
        *img.pixel_at_mut(0, 1) = Mono8::new(20);
        *img.pixel_at_mut(1, 1) = Mono8::new(10);
        let out = adaptive_threshold::<_, Mono32>(&img, win(3), Bias::new(0)).unwrap();
        assert!(
            !out.pixel_at(0, 0),
            "corner must divide by the clipped area (4), not the full 9"
        );
    }

    // The even / zero window cases are no longer reachable through
    // `adaptive_threshold`: the parity invariant moved into `OddWindowSide`, so
    // this function has no window precondition left to violate and the
    // rejection is tested at the constructor instead (see the `odd_window_side_*`
    // tests in `common`). `window!(2)` does not compile at all, wherever it is
    // written, which is the point of the move.

    #[test]
    fn accumulator_overflow_is_err() {
        // 255 × 5000 × 5000 > u32::MAX, so the Mono32 accumulator fails the
        // integral pre-flight regardless of pixel data (worst-case bound).
        let img = Image::<Mono8>::zero(5000, 5000);
        let err = adaptive_threshold::<_, Mono32>(&img, win(3), Bias::new(0)).unwrap_err();
        assert!(
            matches!(err, Error::AccumulatorOverflow { .. }),
            "expected AccumulatorOverflow, got {err:?}"
        );
    }

    #[test]
    fn into_matches_owned() {
        let img = Image::generate(6, 5, |x, y| {
            Mono8::new(((x.wrapping_mul(53).wrapping_add(y.wrapping_mul(97))) & 0xFF) as u8)
        });
        let owned = adaptive_threshold::<_, Mono32>(&img, win(3), Bias::new(-3)).unwrap();

        // Pre-fill with the opposite pattern to prove every pixel is written.
        let mut into = BinaryImage::fill(img.width(), img.height(), true);
        adaptive_threshold_into::<_, Mono32>(&img, win(3), Bias::new(-3), &mut into).unwrap();

        assert_eq!(set_true(&owned), set_true(&into));
    }

    #[test]
    #[should_panic(expected = "does not match input")]
    fn into_wrong_size_panics() {
        let img = Image::fill(4, 4, Mono8::new(1));
        let mut out = BinaryImage::fill(5, 5, false);
        let _ = adaptive_threshold_into::<_, Mono32>(&img, win(3), Bias::new(0), &mut out);
    }

    #[test]
    fn matches_naive_local_mean() {
        // Deterministic pattern (no RNG — AGENTS.md "no external runtime deps").
        let img = Image::generate(7, 6, |x, y| {
            Mono8::new(((x.wrapping_mul(37).wrapping_add(y.wrapping_mul(91))) & 0xFF) as u8)
        });
        for &window in &[1usize, 3, 5] {
            for &offset in &[0i64, 5, -7] {
                let got =
                    adaptive_threshold::<_, Mono32>(&img, win(window), Bias::new(offset)).unwrap();
                let want = naive(&img, window, offset);
                assert_eq!(
                    set_true(&got),
                    set_true(&want),
                    "mismatch at window={window}, offset={offset}"
                );
            }
        }
    }

    #[test]
    fn float_input_monof64_accumulator() {
        // Float path: MonoF32 source in [0, 1] → MonoF64 accumulator,
        // exercising the f64 branch of `exceeds_local_mean`.
        let img = Image::generate(5, 4, |x, y| MonoF32::new(((x * 4 + y) as f32) / 32.0));

        // Brute-force f64 reference.
        let w = img.width();
        let h = img.height();
        let half = 1usize; // window 3
        let want = Image::generate(w, h, |x, y| {
            let left = x.saturating_sub(half);
            let top = y.saturating_sub(half);
            let right = (x + half + 1).min(w);
            let bottom = (y + half + 1).min(h);
            let mut sum = 0.0f64;
            let mut area = 0.0f64;
            for yy in top..bottom {
                for xx in left..right {
                    sum += img.pixel_at(xx, yy).value() as f64;
                    area += 1.0;
                }
            }
            let p = img.pixel_at(x, y).value() as f64;
            (p + 0.0) * area > sum
        });

        let got = adaptive_threshold::<_, MonoF64>(&img, win(3), Bias::new(0.0)).unwrap();
        assert_eq!(set_true(&got), set_true(&want));
    }

    #[test]
    fn bias_is_debug_and_eq() {
        // The hand-written impls compile and behave.
        let a = Bias::<Mono32>::new(-5);
        let b = Bias::<Mono32>::new(-5);
        assert_eq!(a, b);
        assert_eq!(a.get(), -5);
        assert_eq!(format!("{a:?}"), "Bias(-5)");
    }
}

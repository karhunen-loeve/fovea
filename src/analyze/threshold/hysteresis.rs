//! Double-threshold (hysteresis) segmentation.
//!
//! See the [module docs](super) for where thresholding lives in the
//! crate. This file holds the [`hysteresis_threshold`] /
//! [`hysteresis_threshold_into`] pair, the [`HysteresisThresholds`]
//! parameter type, and their tests.

use crate::Error;
use crate::analyze::components::{Connectivity8, connected_components};
use crate::image::{BinaryImage, ImageView, RasterImage, RasterImageMut};
use crate::pixel::{Label32, LabelPixel, SingleChannel};

/// A validated pair of hysteresis thresholds: `low <= high`.
///
/// The double-threshold parameter of [`hysteresis_threshold`] and
/// [`canny`](crate::analyze::edge::canny), carried as one value because
/// the invariant is a *relation* between the two: neither number is wrong
/// on its own. Validation happens once, where the pair is born, and both
/// consumers are total in it.
///
/// `C` is the channel type the comparison happens in: `Saturating<u8>`
/// for a [`Mono8`](crate::pixel::Mono8) mask, `f32` for the gradient
/// magnitude a Canny pipeline produces. Only [`PartialOrd`] is required,
/// which is what makes the float case work. Note also that `!(low <=
/// high)` is exactly the test that rejects a **NaN** threshold, since
/// every comparison against NaN is false. A NaN low threshold would
/// otherwise pass every pixel silently.
///
/// # One constructor, and why
///
/// [`try_new`](Self::try_new) is the only way in. The other parameter
/// types in the crate ([`Sigma`](crate::Sigma),
/// [`OddWindowSide`](crate::OddWindowSide)) also offer a `const fn new`
/// returning [`Option`], which is what lets a literal be checked at
/// compile time behind a macro such as [`sigma!`](crate::sigma). That is
/// impossible here: the check is `PartialOrd` on a generic channel, and
/// trait methods cannot be called in a `const fn` on stable Rust. With no
/// compile-time tier to preserve, an `Option`-returning `new` would differ
/// from `try_new` only by discarding the reason, so it does not exist.
/// Literals and computed values take the same road.
///
/// `low == high` is deliberately valid. It degenerates to a single
/// global threshold (every kept pixel is strong, so nothing propagates),
/// which is the useful baseline for showing what hysteresis adds.
///
/// # Example
///
/// ```
/// use fovea::analyze::threshold::HysteresisThresholds;
/// use std::num::Saturating;
///
/// let t = HysteresisThresholds::try_new(Saturating(100u8), Saturating(200u8))?;
/// assert_eq!(t.low(), Saturating(100u8));
/// assert_eq!(t.high(), Saturating(200u8));
///
/// // Misordered and NaN pairs are rejected where they are born.
/// assert!(HysteresisThresholds::try_new(0.3_f32, 0.1).is_err());
/// assert!(HysteresisThresholds::try_new(f32::NAN, 0.1).is_err());
/// # Ok::<(), fovea::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HysteresisThresholds<C> {
    low: C,
    high: C,
}

impl<C> HysteresisThresholds<C>
where
    C: PartialOrd + Copy + core::fmt::Debug,
{
    /// Creates a threshold pair, validating the `low <= high` relation.
    ///
    /// The only constructor; see the type documentation for why there is no
    /// `const` sibling.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidParameter`] if `!(low <= high)`, which
    /// includes either value being NaN.
    pub fn try_new(low: C, high: C) -> Result<Self, Error> {
        if low <= high {
            Ok(Self { low, high })
        } else {
            Err(Error::InvalidParameter(format!(
                "hysteresis thresholds must satisfy low <= high, got low {low:?} and high {high:?}"
            )))
        }
    }
}

impl<C: Copy> HysteresisThresholds<C> {
    /// The lower (weak) threshold.
    #[must_use]
    pub const fn low(self) -> C {
        self.low
    }

    /// The upper (strong) threshold.
    #[must_use]
    pub const fn high(self) -> C {
        self.high
    }

    /// Re-expresses the pair in another channel type through a
    /// **monotone** conversion, without re-validating.
    ///
    /// Internal, and internal precisely because the invariant survives
    /// only if `convert` is order-preserving. The one caller is
    /// [`canny`](crate::analyze::edge::canny), which widens `f32`
    /// thresholds into its accumulator's channel; that channel is `f32` or
    /// `f64` and nothing else, because
    /// [`MagnitudeChannel`](crate::transform::MagnitudeChannel) is sealed
    /// over exactly those two, and `From<f32>` for both is the identity or
    /// an exact widening.
    pub(crate) fn map_monotone<D: Copy>(self, convert: impl Fn(C) -> D) -> HysteresisThresholds<D> {
        HysteresisThresholds {
            low: convert(self.low),
            high: convert(self.high),
        }
    }
}

/// Double-threshold segmentation with 8-connected weak-edge propagation.
///
/// A pixel is kept in the output mask iff it is **weak**
/// (`value >= low`) **and** its 8-connected weak component contains at
/// least one **strong** pixel (`value >= high`). Concretely:
///
/// - strong:   `value >= high`
/// - weak:     `low <= value < high`
/// - non-edge: `value < low`
///
/// Strong pixels are always kept (a strong pixel is itself the strong
/// member of its own component); weak pixels survive only by 8-connected
/// propagation from a strong pixel; non-edge pixels are always dropped.
/// This is the final segmentation stage of a Canny edge detector, but it
/// is a self-contained primitive usable on any single-channel image.
///
/// # Thresholds are inclusive
///
/// Both comparisons are inclusive (`>=`), matching the Canny literature
/// and OpenCV. Note this **differs** from
/// [`otsu_binary_mask`](crate::analyze::histogram::otsu_binary_mask),
/// whose split is exclusive (`value > t`); the two are intentionally not
/// identical.
///
/// # Single channel only
///
/// Comparison uses channel 0, and the [`SingleChannel`] bound makes that
/// a compile-time requirement: passing a multi-channel pixel does not
/// compile. The bound is the marker rather than a monochrome-specific
/// type so the function still accepts both integer masks
/// ([`Mono8`](crate::pixel::Mono8)) and the float gradient-magnitude image
/// ([`MonoF32`](crate::pixel::MonoF32)) the Canny pipeline produces —
/// hence [`PartialOrd`] (not [`Ord`]) on the channel.
///
/// # Total in its thresholds
///
/// The `low <= high` relation lives in [`HysteresisThresholds`], so this
/// function has no threshold precondition to violate and does not panic.
/// Build the pair with [`HysteresisThresholds::try_new`], whether the two
/// numbers are literals or derived from the data.
///
/// # Examples
///
/// ```
/// use fovea::analyze::threshold::{HysteresisThresholds, hysteresis_threshold};
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::Mono8;
/// use std::num::Saturating;
///
/// // A strong pixel (255) with a weak neighbour (128) and an isolated
/// // weak pixel (128) two columns away.
/// let img = Image::from_vec(
///     4, 1,
///     vec![Mono8::new(255), Mono8::new(128), Mono8::new(0), Mono8::new(128)],
/// ).unwrap();
/// let thresholds = HysteresisThresholds::try_new(Saturating(100u8), Saturating(200u8)).unwrap();
/// let mask = hysteresis_threshold(&img, thresholds);
/// assert!(mask.pixel_at(0, 0));  // strong
/// assert!(mask.pixel_at(1, 0));  // weak, bridged to the strong pixel
/// assert!(!mask.pixel_at(2, 0)); // non-edge
/// assert!(!mask.pixel_at(3, 0)); // weak but isolated → dropped
/// ```
pub fn hysteresis_threshold<I, P>(
    image: &I,
    thresholds: HysteresisThresholds<P::Channel>,
) -> BinaryImage
where
    I: RasterImage<Pixel = P>,
    P: SingleChannel,
    P::Channel: PartialOrd + Copy + core::fmt::Debug,
{
    // Owned variant allocates the output and delegates, matching the
    // `connected_components` / `connected_components_into` convention.
    let mut out = BinaryImage::fill(image.width(), image.height(), false);
    hysteresis_threshold_into(image, thresholds, &mut out);
    out
}

/// As [`hysteresis_threshold`], writing into a caller-owned mask.
///
/// Reusing `out` across frames avoids reallocating the output for each
/// call (e.g. a Canny pass over video). The weak mask and label image
/// are unavoidable internal scratch and are still allocated per call.
///
/// # Panics
///
/// Panics if `out.size() != image.size()` (Tier 3: the caller allocated
/// the mask from sizes in hand).
pub fn hysteresis_threshold_into<I, P>(
    image: &I,
    thresholds: HysteresisThresholds<P::Channel>,
    out: &mut BinaryImage,
) where
    I: RasterImage<Pixel = P>,
    P: SingleChannel,
    P::Channel: PartialOrd + Copy + core::fmt::Debug,
{
    assert_eq!(
        out.size(),
        image.size(),
        "hysteresis_threshold_into: output size {:?} does not match input {:?}",
        out.size(),
        image.size()
    );
    // No threshold check here: `low <= high`, NaN included, was
    // established once by `HysteresisThresholds`, whose fields are
    // private, so there is nothing left to assert.
    let (low, high) = (thresholds.low(), thresholds.high());

    let w = image.width();
    let h = image.height();

    // 1. Weak mask: every pixel that clears the low threshold. Built row
    //    by row so the source is read through a contiguous slice rather
    //    than a per-pixel `pixel_at` index computation.
    let mut weak = BinaryImage::fill(w, h, false);
    for y in 0..h {
        let src = image.row(y);
        let dst = weak.row_mut(y);
        for (out_px, src_px) in dst.iter_mut().zip(src) {
            *out_px = src_px.channel(0) >= low;
        }
    }

    // 2. Label the weak mask with 8-connectivity. The strong mask is
    //    never materialised — `>= high` is tested inline in step 3.
    let labeling = connected_components::<Label32, Connectivity8>(&weak).expect(
        "hysteresis_threshold: weak-component count exceeds Label32 capacity (u32::MAX); \
         an image with that many components is not representable",
    );

    // 3. keep[label] = true iff that weak component holds a strong pixel.
    //    Index 0 is the background label and stays false. Widen before
    //    the `+ 1`: `label_count` can be `u32::MAX` itself.
    let mut keep = vec![false; labeling.label_count as usize + 1];
    for y in 0..h {
        let weak_row = weak.row(y);
        let img_row = image.row(y);
        let label_row = labeling.labels.row(y);
        for x in 0..w {
            if weak_row[x] && img_row[x].channel(0) >= high {
                keep[label_row[x].to_label_index() as usize] = true;
            }
        }
    }

    // 4. out[p] = weak[p] && keep[label[p]]. Background pixels carry
    //    label 0 (keep[0] == false), so the `weak_row[x] &&` guard is a
    //    short-circuit, not a correctness requirement.
    for y in 0..h {
        let weak_row = weak.row(y);
        let label_row = labeling.labels.row(y);
        let out_row = out.row_mut(y);
        for x in 0..w {
            out_row[x] = weak_row[x] && keep[label_row[x].to_label_index() as usize];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{Image, ImageView};
    use crate::pixel::{Mono8, MonoF32};
    use std::num::Saturating;

    // ── Fixtures ────────────────────────────────────────────────────────────

    /// Build a `Mono8` image from an ASCII grid: `#` = 255 (strong),
    /// `+` = 128 (weak), `.` = 0 (non-edge). All rows must share a width.
    /// Pairs with thresholds `low = 100`, `high = 200`.
    fn mask(rows: &[&str]) -> Image<Mono8> {
        let h = rows.len();
        let w = rows[0].chars().count();
        for r in rows {
            assert_eq!(r.chars().count(), w, "ragged fixture row: {r:?}");
        }
        Image::generate(w, h, |x, y| {
            let c = rows[y].chars().nth(x).unwrap();
            let v = match c {
                '#' => 255u8,
                '+' => 128,
                '.' => 0,
                other => panic!("unexpected fixture char {other:?}"),
            };
            Mono8::new(v)
        })
    }

    const LOW: Saturating<u8> = Saturating(100);
    const HIGH: Saturating<u8> = Saturating(200);

    /// The `(100, 200)` pair every ASCII fixture is written against.
    ///
    /// A `fn` and not a `const`: the constructor compares through
    /// `PartialOrd`, which a `const fn` cannot call.
    fn pair() -> HysteresisThresholds<Saturating<u8>> {
        HysteresisThresholds::try_new(LOW, HIGH).unwrap()
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

    // ── Behaviours (one test per TDD-list item) ──────────────────────────────

    #[test]
    fn all_below_low_is_empty() {
        let img = mask(&["...", "..."]);
        let out = hysteresis_threshold(&img, pair());
        assert!(set_true(&out).is_empty());
    }

    #[test]
    fn all_above_high_is_full() {
        let img = mask(&["###", "###"]);
        let out = hysteresis_threshold(&img, pair());
        for y in 0..out.height() {
            for x in 0..out.width() {
                assert!(out.pixel_at(x, y), "({x},{y}) should be kept");
            }
        }
    }

    #[test]
    fn isolated_weak_is_dropped() {
        // A 2×2 weak blob with no strong pixel anywhere.
        let img = mask(&["....", ".++.", ".++.", "...."]);
        let out = hysteresis_threshold(&img, pair());
        assert!(set_true(&out).is_empty());
    }

    #[test]
    fn weak_bridges_to_strong_kept() {
        // Strong at (1,1); weak chain (2,1),(3,1) is 4-connected to it.
        let img = mask(&["....", ".#++", "...."]);
        let out = hysteresis_threshold(&img, pair());
        let expected: std::collections::BTreeSet<_> =
            [(1, 1), (2, 1), (3, 1)].into_iter().collect();
        assert_eq!(set_true(&out), expected);
    }

    #[test]
    fn diagonal_propagation_uses_8_connectivity() {
        // Strong at (0,0); weak pixels reachable from it only diagonally.
        // Under Connectivity4 these would be three separate components and
        // the weak ones dropped; Connectivity8 keeps the whole chain.
        let img = mask(&["#..", ".+.", "..+"]);
        let out = hysteresis_threshold(&img, pair());
        let expected: std::collections::BTreeSet<_> =
            [(0, 0), (1, 1), (2, 2)].into_iter().collect();
        assert_eq!(set_true(&out), expected);
    }

    #[test]
    fn two_separate_components_independent() {
        // Component A (strong, top-left) kept; component B (weak only,
        // bottom-right) dropped. Row/column 2 of background keeps the two
        // 2×2 blocks from touching even under 8-connectivity.
        let img = mask(&["##...", "##...", ".....", "...++", "...++"]);
        let out = hysteresis_threshold(&img, pair());
        let expected: std::collections::BTreeSet<_> =
            [(0, 0), (1, 0), (0, 1), (1, 1)].into_iter().collect();
        assert_eq!(set_true(&out), expected);
    }

    #[test]
    fn boundary_inclusive_at_low_and_high() {
        // low = 128, high = 200. Pixel == low must be weak (kept here via
        // its strong neighbour), pixel == high must be strong, pixel just
        // below low must be non-edge. This single fixture distinguishes
        // inclusive from exclusive at *both* thresholds: with exclusive
        // `>` the 128 would be non-edge and the 200 would be a lone weak
        // pixel, so the whole row would drop to false.
        let img = Image::from_vec(
            3,
            1,
            vec![Mono8::new(127), Mono8::new(128), Mono8::new(200)],
        )
        .unwrap();
        let t = HysteresisThresholds::try_new(Saturating(128), Saturating(200)).unwrap();
        let out = hysteresis_threshold(&img, t);
        assert!(!out.pixel_at(0, 0), "127 < low → non-edge");
        assert!(out.pixel_at(1, 0), "128 == low → weak, bridged to strong");
        assert!(out.pixel_at(2, 0), "200 == high → strong");
    }

    #[test]
    fn float_magnitude_input() {
        // The real Canny path: an `Image<MonoF32>` gradient-magnitude map.
        let img = Image::from_vec(
            3,
            1,
            vec![MonoF32::new(0.1), MonoF32::new(0.3), MonoF32::new(0.6)],
        )
        .unwrap();
        let t = HysteresisThresholds::try_new(0.2f32, 0.5f32).unwrap();
        let out = hysteresis_threshold(&img, t);
        assert!(!out.pixel_at(0, 0), "0.1 < low → non-edge");
        assert!(out.pixel_at(1, 0), "0.3 weak, bridged to the strong 0.6");
        assert!(out.pixel_at(2, 0), "0.6 >= high → strong");
    }

    #[test]
    fn low_equals_high_keeps_only_strong() {
        // `low == high` is allowed (`low <= high`) and collapses the weak
        // band to nothing: the output is exactly `value >= high`, with no
        // propagation. The isolated 200 survives on its own, and the 199
        // next to it does not get bridged.
        let img = Image::from_vec(
            4,
            1,
            vec![
                Mono8::new(0),
                Mono8::new(199),
                Mono8::new(200),
                Mono8::new(255),
            ],
        )
        .unwrap();
        let t = HysteresisThresholds::try_new(Saturating(200), Saturating(200)).unwrap();
        let out = hysteresis_threshold(&img, t);
        assert!(!out.pixel_at(0, 0), "0 < high");
        assert!(!out.pixel_at(1, 0), "199 < high, and there is no weak band");
        assert!(out.pixel_at(2, 0), "200 == high → strong");
        assert!(out.pixel_at(3, 0), "255 >= high → strong");
    }

    #[test]
    fn into_variant_matches_owned() {
        let img = mask(&["#++.", ".+..", "..+#", "++.."]);
        let owned = hysteresis_threshold(&img, pair());

        // Pre-fill the destination with the opposite pattern to prove
        // every pixel is written, not merely OR-ed in.
        let mut into = BinaryImage::fill(img.width(), img.height(), true);
        hysteresis_threshold_into(&img, pair(), &mut into);

        assert_eq!(set_true(&owned), set_true(&into));
    }

    #[test]
    #[should_panic(expected = "does not match input")]
    fn into_wrong_size_panics() {
        let img = mask(&["##", "##"]);
        let mut out = BinaryImage::fill(3, 3, false);
        hysteresis_threshold_into(&img, pair(), &mut out);
    }

    // ── The threshold relation, now carried by the parameter type ────────────
    //
    // `hysteresis_threshold` has no misordered-threshold case left to test:
    // it cannot be handed one. The rejection is tested where it now happens.

    #[test]
    fn misordered_pair_is_rejected_at_construction() {
        // Literals go through the same road as computed values, so the
        // rejection is an error rather than an abort.
        assert!(HysteresisThresholds::try_new(HIGH, LOW).is_err());
    }

    #[test]
    fn computed_pair_reports_misordering_as_a_value() {
        // The `try_new` half: thresholds picked from a magnitude histogram
        // can come out inverted, which is a value the caller handles.
        let err = HysteresisThresholds::try_new(0.5_f32, 0.2).unwrap_err();
        assert!(
            matches!(err, Error::InvalidParameter(ref m) if m.contains("low <= high")),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn nan_threshold_is_rejected() {
        // The load-bearing float case: `!(low <= high)` is false for NaN in
        // either position, so a NaN never reaches the comparison loop, where
        // `value >= NaN` would be false for every pixel and the mask would
        // come back silently empty.
        assert!(HysteresisThresholds::try_new(f32::NAN, 0.5).is_err());
        assert!(HysteresisThresholds::try_new(0.2_f32, f32::NAN).is_err());
        assert!(HysteresisThresholds::try_new(f32::NAN, f32::NAN).is_err());
    }

    #[test]
    fn equal_thresholds_are_valid() {
        // `low == high` degenerates to a single global cut, which is the
        // baseline the examples use to show what hysteresis adds. It is not
        // an error.
        let t = HysteresisThresholds::try_new(LOW, LOW).unwrap();
        assert_eq!(t.low(), t.high());
    }

    #[test]
    fn accessors_return_what_was_given() {
        let t = pair();
        assert_eq!(t.low(), LOW);
        assert_eq!(t.high(), HIGH);
    }

    #[test]
    fn map_monotone_re_types_without_re_validating() {
        // The internal hook `canny` uses to widen `f32` thresholds into its
        // accumulator channel. Order-preserving conversion in, valid pair out.
        let t = HysteresisThresholds::try_new(0.1_f32, 0.3).unwrap();
        let widened: HysteresisThresholds<f64> = t.map_monotone(f64::from);
        assert_eq!(widened.low(), 0.1_f32 as f64);
        assert_eq!(widened.high(), 0.3_f32 as f64);
    }
}

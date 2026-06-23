//! Double-threshold (hysteresis) segmentation.
//!
//! See the [module docs](super) for where thresholding lives in the
//! crate. This file holds the [`hysteresis_threshold`] /
//! [`hysteresis_threshold_into`] pair and their tests.

use crate::analyze::components::{Connectivity8, connected_components};
use crate::image::{BinaryImage, Image, ImageView, RasterImage, RasterImageMut};
use crate::pixel::{HomogeneousPixel, Label32, LabelPixel};

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
/// Comparison uses channel 0. The bound is
/// [`HomogeneousPixel`] rather than a monochrome-specific type so the
/// function accepts both integer masks ([`Mono8`](crate::pixel::Mono8))
/// and the float gradient-magnitude image
/// ([`MonoF32`](crate::pixel::MonoF32)) the Canny pipeline produces —
/// hence [`PartialOrd`] (not [`Ord`]) on the channel.
///
/// # Panics
///
/// - Panics if `P::CHANNEL_COUNT != 1` (Tier 3 — programmer bug; convert
///   multi-channel input to single channel first).
/// - Panics if `!(low <= high)` (Tier 3 — misordered or NaN thresholds
///   are a precondition violation). The message names both values.
///
/// # Examples
///
/// ```
/// use fovea::analyze::threshold::hysteresis_threshold;
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
/// let mask = hysteresis_threshold(&img, Saturating(100u8), Saturating(200u8));
/// assert!(mask.pixel_at(0, 0));  // strong
/// assert!(mask.pixel_at(1, 0));  // weak, bridged to the strong pixel
/// assert!(!mask.pixel_at(2, 0)); // non-edge
/// assert!(!mask.pixel_at(3, 0)); // weak but isolated → dropped
/// ```
pub fn hysteresis_threshold<I, P>(image: &I, low: P::Channel, high: P::Channel) -> BinaryImage
where
    I: RasterImage<Pixel = P>,
    P: HomogeneousPixel,
    P::Channel: PartialOrd + Copy + core::fmt::Debug,
{
    // Owned variant allocates the output and delegates, matching the
    // `connected_components` / `connected_components_into` convention.
    let mut out = BinaryImage::fill(image.width(), image.height(), false);
    hysteresis_threshold_into(image, low, high, &mut out);
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
/// In addition to the panics documented on [`hysteresis_threshold`]:
/// panics if `out.size() != image.size()` (Tier 3).
pub fn hysteresis_threshold_into<I, P>(
    image: &I,
    low: P::Channel,
    high: P::Channel,
    out: &mut BinaryImage,
) where
    I: RasterImage<Pixel = P>,
    P: HomogeneousPixel,
    P::Channel: PartialOrd + Copy + core::fmt::Debug,
{
    assert_eq!(
        P::CHANNEL_COUNT,
        1,
        "hysteresis_threshold: requires a single-channel pixel; got CHANNEL_COUNT = {}",
        P::CHANNEL_COUNT
    );
    assert_eq!(
        out.size(),
        image.size(),
        "hysteresis_threshold_into: output size {:?} does not match input {:?}",
        out.size(),
        image.size()
    );
    // Decision 2 (Tier 3): misordered thresholds are a caller
    // bug, not a data failure. Panic with both values named. `!(low <=
    // high)` also rejects a NaN threshold on float inputs.
    assert!(
        low <= high,
        "hysteresis_threshold: low ({low:?}) must be <= high ({high:?})"
    );

    let w = image.width();
    let h = image.height();

    // 1. Weak mask: every pixel that clears the low threshold.
    let weak: BinaryImage = Image::generate(w, h, |x, y| image.pixel_at(x, y).channel(0) >= low);

    // 2. Label the weak mask with 8-connectivity. The strong mask is
    //    never materialised — `>= high` is tested inline in step 3.
    let labeling = connected_components::<Label32, Connectivity8>(&weak).expect(
        "hysteresis_threshold: weak-component count exceeds Label32 capacity (u32::MAX); \
         an image with that many components is not representable",
    );

    // 3. keep[label] = true iff that weak component holds a strong pixel.
    //    Index 0 is the background label and stays false.
    let mut keep = vec![false; (labeling.label_count + 1) as usize];
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
        let out = hysteresis_threshold(&img, LOW, HIGH);
        assert!(set_true(&out).is_empty());
    }

    #[test]
    fn all_above_high_is_full() {
        let img = mask(&["###", "###"]);
        let out = hysteresis_threshold(&img, LOW, HIGH);
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
        let out = hysteresis_threshold(&img, LOW, HIGH);
        assert!(set_true(&out).is_empty());
    }

    #[test]
    fn weak_bridges_to_strong_kept() {
        // Strong at (1,1); weak chain (2,1),(3,1) is 4-connected to it.
        let img = mask(&["....", ".#++", "...."]);
        let out = hysteresis_threshold(&img, LOW, HIGH);
        let expected: std::collections::BTreeSet<_> = [(1, 1), (2, 1), (3, 1)].into_iter().collect();
        assert_eq!(set_true(&out), expected);
    }

    #[test]
    fn diagonal_propagation_uses_8_connectivity() {
        // Strong at (0,0); weak pixels reachable from it only diagonally.
        // Under Connectivity4 these would be three separate components and
        // the weak ones dropped; Connectivity8 keeps the whole chain.
        let img = mask(&["#..", ".+.", "..+"]);
        let out = hysteresis_threshold(&img, LOW, HIGH);
        let expected: std::collections::BTreeSet<_> = [(0, 0), (1, 1), (2, 2)].into_iter().collect();
        assert_eq!(set_true(&out), expected);
    }

    #[test]
    fn two_separate_components_independent() {
        // Component A (strong, top-left) kept; component B (weak only,
        // bottom-right) dropped. Row/column 2 of background keeps the two
        // 2×2 blocks from touching even under 8-connectivity.
        let img = mask(&["##...", "##...", ".....", "...++", "...++"]);
        let out = hysteresis_threshold(&img, LOW, HIGH);
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
        let img =
            Image::from_vec(3, 1, vec![Mono8::new(127), Mono8::new(128), Mono8::new(200)]).unwrap();
        let out = hysteresis_threshold(&img, Saturating(128), Saturating(200));
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
        let out = hysteresis_threshold(&img, 0.2f32, 0.5f32);
        assert!(!out.pixel_at(0, 0), "0.1 < low → non-edge");
        assert!(out.pixel_at(1, 0), "0.3 weak, bridged to the strong 0.6");
        assert!(out.pixel_at(2, 0), "0.6 >= high → strong");
    }

    #[test]
    fn into_variant_matches_owned() {
        let img = mask(&["#++.", ".+..", "..+#", "++.."]);
        let owned = hysteresis_threshold(&img, LOW, HIGH);

        // Pre-fill the destination with the opposite pattern to prove
        // every pixel is written, not merely OR-ed in.
        let mut into = BinaryImage::fill(img.width(), img.height(), true);
        hysteresis_threshold_into(&img, LOW, HIGH, &mut into);

        assert_eq!(set_true(&owned), set_true(&into));
    }

    #[test]
    #[should_panic(expected = "does not match input")]
    fn into_wrong_size_panics() {
        let img = mask(&["##", "##"]);
        let mut out = BinaryImage::fill(3, 3, false);
        hysteresis_threshold_into(&img, LOW, HIGH, &mut out);
    }

    #[test]
    #[should_panic(expected = "must be <= high")]
    fn low_greater_than_high_panics() {
        let img = mask(&["#+.", "+.#"]);
        let _ = hysteresis_threshold(&img, HIGH, LOW); // low > high
    }
}

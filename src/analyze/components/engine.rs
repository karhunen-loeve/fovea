//! Two-pass union-find connected-components engine.
//!
//! Implements [ADR-0047] \u00a72\u20133. See the module-level docs for the
//! public surface and the worked 4\u00d74 example.
//!
//! [ADR-0047]: https://github.com/karhunen-loeve/fovea/blob/main/docs/adr/0047-connected-components-design.md

use crate::Error;
use crate::image::{Image, ImageView, ImageViewMut, RasterImage};
use crate::pixel::LabelPixel;

use super::Labeling;
use super::connectivity::Connectivity;
use super::stats::ComponentStats;
use super::stats::sink::{NoStats, StatsSink, WithStats};
use super::union_find::UnionFind;

/// Compute the connected-component labeling of `image`, allocating a
/// fresh [`Labeling<L>`] (ADR-0047 \u00a72).
///
/// The label pixel type `L` and connectivity strategy `C` are named
/// explicitly by the caller (turbofish), e.g.
/// `connected_components::<Label32, Connectivity8>(&binary)`.
///
/// # Errors \u2014 Tier 2 ([ADR-0025])
///
/// Returns [`Error::LabelOverflow`] if the input contains more
/// connected components than `L::MAX_LABEL` can encode.
///
/// [ADR-0025]: https://github.com/karhunen-loeve/fovea/blob/main/docs/adr/0025-error-handling-conventions.md
///
/// # Examples
///
/// ```
/// use fovea::analyze::components::{connected_components, Connectivity4};
/// use fovea::image::BinaryImage;
/// use fovea::pixel::Label32;
///
/// let img = BinaryImage::fill(8, 8, true);
/// let r = connected_components::<Label32, Connectivity4>(&img).unwrap();
/// assert_eq!(r.label_count, 1);
/// ```
pub fn connected_components<L, C>(
    image: &impl RasterImage<Pixel = bool>,
) -> Result<Labeling<L>, Error>
where
    L: LabelPixel,
    C: Connectivity,
{
    let mut labels = Image::<L>::zero(image.width(), image.height());
    let label_count = connected_components_into::<L, C>(image, &mut labels)?;
    Ok(Labeling {
        labels,
        label_count,
    })
}

/// Compute the connected-component labeling of `image`, writing the
/// label image into `out` and returning `label_count` (ADR-0047 \u00a72).
///
/// # Panics
///
/// Panics if `out.size() != image.size()` (Tier 3 \u2014 programmer bug
/// per [ADR-0025]).
///
/// # Errors \u2014 Tier 2
///
/// Returns [`Error::LabelOverflow`] if the input contains more
/// connected components than `L::MAX_LABEL` can encode.
///
/// [ADR-0025]: https://github.com/karhunen-loeve/fovea/blob/main/docs/adr/0025-error-handling-conventions.md
pub fn connected_components_into<L, C>(
    image: &impl RasterImage<Pixel = bool>,
    out: &mut Image<L>,
) -> Result<u64, Error>
where
    L: LabelPixel,
    C: Connectivity,
{
    assert_eq!(
        out.size(),
        image.size(),
        "connected_components_into: output size {:?} does not match input {:?}",
        out.size(),
        image.size()
    );

    run::<L, C, _, NoStats>(image, out, &mut NoStats)
}

/// Compute the connected-component labeling of `image`, plus one
/// [`ComponentStats`] per foreground component (area, bounding box,
/// centroid sums).
///
/// Stats are accumulated inline during pass 2; see ADR-0047 \u00a78 for the
/// rationale and scope decision (option 3B).
///
/// # Errors \u2014 Tier 2
///
/// Returns [`Error::LabelOverflow`] if the input contains more
/// connected components than `L::MAX_LABEL` can encode.
///
/// # Examples
///
/// ```
/// use fovea::analyze::components::{
///     connected_components_with_stats, Connectivity4,
/// };
/// use fovea::image::BinaryImage;
/// use fovea::pixel::Label32;
///
/// // A 2x2 square.
/// let img = BinaryImage::fill(2, 2, true);
/// let (lab, stats) =
///     connected_components_with_stats::<Label32, Connectivity4>(&img).unwrap();
/// assert_eq!(lab.label_count, 1);
/// assert_eq!(stats[0].area, 4);
/// assert_eq!(stats[0].centroid(), (0.5, 0.5));
/// ```
pub fn connected_components_with_stats<L, C>(
    image: &impl RasterImage<Pixel = bool>,
) -> Result<(Labeling<L>, Vec<ComponentStats>), Error>
where
    L: LabelPixel,
    C: Connectivity,
{
    let mut labels = Image::<L>::zero(image.width(), image.height());
    let mut stats: Vec<ComponentStats> = Vec::new();
    let label_count = {
        let mut sink = WithStats { out: &mut stats };
        run::<L, C, _, WithStats<'_>>(image, &mut labels, &mut sink)?
    };
    debug_assert_eq!(stats.len() as u64, label_count);
    Ok((
        Labeling {
            labels,
            label_count,
        },
        stats,
    ))
}

// ──────────────────────────────────────────────────────────────────────
// Engine \u2014 monomorphised over `S: StatsSink`.
// ──────────────────────────────────────────────────────────────────────

/// Maximum number of raster-preceding neighbours examined per pixel
/// across every shipped [`Connectivity`]. Currently 4 (for
/// [`Connectivity8`](super::Connectivity8)). The pass-1 inner buffer
/// `others: [u64; MAX_NEIGHBOURS]` hardcodes this constant. Adding a
/// connectivity with more predecessors requires lifting this and is
/// flagged in ADR-0047 \u00a77.
const MAX_NEIGHBOURS: usize = 4;

fn run<L, C, I, S>(image: &I, out: &mut Image<L>, sink: &mut S) -> Result<u64, Error>
where
    L: LabelPixel,
    C: Connectivity,
    I: RasterImage<Pixel = bool>,
    S: StatsSink,
{
    let w = image.width();
    let h = image.height();
    if w == 0 || h == 0 {
        return Ok(0);
    }

    // ── Pass 1 ───────────────────────────────────────────────────────
    // Provisional labels live in a flat W*H Vec<u64>, raster-scan
    // order. Zero is the background sentinel.
    let mut prov: Vec<u64> = vec![0; w * h];
    // Capacity hint: pathological all-stripes input produces ~W*H/4
    // labels; using that as the initial allocation keeps `make_set`
    // amortised cheap without over-allocating in the common case.
    let cap_hint = (w * h) / 4 + 1;
    let mut uf = UnionFind::with_capacity(cap_hint);

    for y in 0..h {
        let row = image.row(y);
        for x in 0..w {
            if !row[x] {
                continue;
            }

            // Collect provisional labels of the already-visited
            // foreground neighbours. The smallest is tracked
            // separately; everything else goes in `others`, which is
            // unioned with `smallest` at the end.
            let mut smallest: u64 = u64::MAX;
            let mut others: [u64; MAX_NEIGHBOURS] = [0; MAX_NEIGHBOURS];
            let mut other_count = 0usize;

            for &(dx, dy) in C::OFFSETS {
                let nx = x as i64 + dx as i64;
                let ny = y as i64 + dy as i64;
                if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                    continue;
                }
                let p = prov[ny as usize * w + nx as usize];
                if p == 0 {
                    continue;
                }
                if p < smallest {
                    if smallest != u64::MAX {
                        others[other_count] = smallest;
                        other_count += 1;
                    }
                    smallest = p;
                } else if p != smallest {
                    others[other_count] = p;
                    other_count += 1;
                }
            }

            let label = if smallest == u64::MAX {
                let new_label = uf.make_set();
                if new_label > L::MAX_LABEL {
                    return Err(Error::LabelOverflow {
                        label_capacity: L::MAX_LABEL,
                    });
                }
                new_label
            } else {
                for &o in &others[..other_count] {
                    uf.union(smallest, o);
                }
                smallest
            };

            prov[y * w + x] = label;
        }
    }

    // ── Pass 2 ───────────────────────────────────────────────────────
    // Resolve roots and compact labels to a dense `1..=label_count`,
    // writing the output pixels and forwarding `(label, first, x, y)`
    // to the stats sink.
    let mut compact: Vec<u64> = vec![0; uf.len() as usize];
    let mut compact_counter: u64 = 1;

    for y in 0..h {
        for x in 0..w {
            let p = prov[y * w + x];
            let cell = out.pixel_at_mut(x, y);
            if p == 0 {
                *cell = L::zero();
            } else {
                let root = uf.find(p);
                let existing = compact[root as usize];
                let (c, first) = if existing == 0 {
                    let assigned = compact_counter;
                    compact[root as usize] = assigned;
                    compact_counter += 1;
                    (assigned, true)
                } else {
                    (existing, false)
                };
                // Invariant: 0 < c <= compact_counter - 1 <= L::MAX_LABEL
                // (the pass-1 overflow check guarantees this).
                debug_assert!(
                    c <= L::MAX_LABEL,
                    "internal invariant violated: compact label {} > MAX_LABEL {}",
                    c,
                    L::MAX_LABEL
                );
                *cell = L::from_label_index(c).expect(
                    "internal error: compact label exceeds L::MAX_LABEL despite \
                     pass-1 overflow check (analyze::components engine)",
                );
                sink.record(c, first, x, y);
            }
        }
    }

    Ok(compact_counter - 1)
}

// ══════════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::super::{
        Connectivity4, Connectivity8, Labeling, connected_components, connected_components_into,
        connected_components_with_stats,
    };
    use crate::Error;
    use crate::image::{BinaryImage, Image, ImageView, SubView};
    use crate::pixel::{Label32, LabelPixel, ZeroablePixel};
    use crate::{Coordinate, Rectangle, Size};

    // Helpers ─────────────────────────────────────────────────────────

    /// Build a binary image from a string where `#` is foreground and
    /// any other non-whitespace char is background. Lines must all
    /// have the same width.
    fn img_from_str(text: &str) -> BinaryImage {
        let lines: Vec<&str> = text
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .collect();
        let h = lines.len();
        let w = lines[0].chars().count();
        for l in &lines {
            assert_eq!(l.chars().count(), w);
        }
        let mut data = Vec::with_capacity(w * h);
        for l in &lines {
            for c in l.chars() {
                data.push(c == '#');
            }
        }
        BinaryImage::from_vec(w, h, data).unwrap()
    }

    /// Group `(x, y)` foreground pixel positions by their compact
    /// label. Returns a vector of label-coordinate sets, indexed by
    /// `compact_label - 1`, plus the foreground-count total.
    fn partition_by_label(
        lab: &Labeling<Label32>,
    ) -> Vec<std::collections::BTreeSet<(usize, usize)>> {
        use std::collections::BTreeSet;
        let mut groups: Vec<BTreeSet<(usize, usize)>> =
            (0..lab.label_count).map(|_| BTreeSet::new()).collect();
        for y in 0..lab.labels.height() {
            for x in 0..lab.labels.width() {
                let v = lab.labels.pixel_at(x, y).value();
                if v != 0 {
                    groups[(v - 1) as usize].insert((x, y));
                }
            }
        }
        groups
    }

    // Engine tests ────────────────────────────────────────────────────

    #[test]
    fn empty_image_returns_zero_labels() {
        let img = BinaryImage::from_vec(0, 0, Vec::new()).unwrap();
        let r = connected_components::<Label32, Connectivity4>(&img).unwrap();
        assert_eq!(r.label_count, 0);
        assert_eq!(r.labels.size(), Size::new(0, 0));
    }

    #[test]
    fn all_background() {
        let img = BinaryImage::fill(8, 8, false);
        let r = connected_components::<Label32, Connectivity4>(&img).unwrap();
        assert_eq!(r.label_count, 0);
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(r.labels.pixel_at(x, y), Label32::BACKGROUND);
            }
        }
    }

    #[test]
    fn all_foreground_4connected() {
        let img = BinaryImage::fill(5, 4, true);
        let r = connected_components::<Label32, Connectivity4>(&img).unwrap();
        assert_eq!(r.label_count, 1);
        for y in 0..4 {
            for x in 0..5 {
                assert_eq!(r.labels.pixel_at(x, y), Label32::new(1));
            }
        }
    }

    #[test]
    fn all_foreground_8connected() {
        let img = BinaryImage::fill(5, 4, true);
        let r = connected_components::<Label32, Connectivity8>(&img).unwrap();
        assert_eq!(r.label_count, 1);
    }

    #[test]
    fn worked_4x4_example_conn4() {
        let img = img_from_str(
            r#"
            .##.
            ##..
            ..##
            ..#.
        "#,
        );
        let r = connected_components::<Label32, Connectivity4>(&img).unwrap();
        assert_eq!(r.label_count, 2);
        // The partition: 4-pixel top L and 3-pixel bottom L.
        let groups = partition_by_label(&r);
        let sizes: Vec<usize> = groups.iter().map(|g| g.len()).collect();
        let mut sorted = sizes.clone();
        sorted.sort();
        assert_eq!(sorted, vec![3, 4]);
        // Total foreground = 7.
        let total: usize = sizes.iter().sum();
        assert_eq!(total, 7);
    }

    #[test]
    fn worked_4x4_example_conn8_merges_diagonal() {
        let img = img_from_str(
            r#"
            .##.
            ##..
            ..##
            ..#.
        "#,
        );
        let r = connected_components::<Label32, Connectivity8>(&img).unwrap();
        // The diagonal at (2,2) touches (1,1) so all 8 foreground pixels collapse.
        assert_eq!(r.label_count, 1);
    }

    #[test]
    fn u_shape_forces_pass1_merge_conn4() {
        let img = img_from_str(
            r#"
            #.#
            #.#
            ###
        "#,
        );
        let r = connected_components::<Label32, Connectivity4>(&img).unwrap();
        assert_eq!(r.label_count, 1);
    }

    #[test]
    fn single_pixel_blob_at_corner() {
        let mut data = vec![false; 4 * 4];
        data[0] = true;
        let img = BinaryImage::from_vec(4, 4, data).unwrap();
        let r = connected_components::<Label32, Connectivity4>(&img).unwrap();
        assert_eq!(r.label_count, 1);
        assert_eq!(r.labels.pixel_at(0, 0), Label32::new(1));
    }

    #[test]
    fn single_row() {
        let img = BinaryImage::from_vec(5, 1, vec![true, false, true, true, false]).unwrap();
        let r = connected_components::<Label32, Connectivity4>(&img).unwrap();
        assert_eq!(r.label_count, 2);
    }

    #[test]
    fn single_column() {
        let img = BinaryImage::from_vec(1, 5, vec![true, false, true, true, false]).unwrap();
        let r = connected_components::<Label32, Connectivity4>(&img).unwrap();
        assert_eq!(r.label_count, 2);
    }

    #[test]
    fn checkerboard_conn4_yields_one_per_pixel() {
        // 8x8 checkerboard, true at (x+y)%2==0
        let mut data = Vec::with_capacity(64);
        for y in 0..8 {
            for x in 0..8 {
                data.push((x + y) % 2 == 0);
            }
        }
        let img = BinaryImage::from_vec(8, 8, data).unwrap();
        let r = connected_components::<Label32, Connectivity4>(&img).unwrap();
        assert_eq!(r.label_count, 32);
    }

    #[test]
    fn checkerboard_conn8_yields_one_component() {
        let mut data = Vec::with_capacity(64);
        for y in 0..8 {
            for x in 0..8 {
                data.push((x + y) % 2 == 0);
            }
        }
        let img = BinaryImage::from_vec(8, 8, data).unwrap();
        let r = connected_components::<Label32, Connectivity8>(&img).unwrap();
        assert_eq!(r.label_count, 1);
    }

    #[test]
    fn subview_input_round_trips() {
        // Outer 6x6 image; ROI is the inner 4x4.
        let img = img_from_str(
            r#"
            ......
            .####.
            .#..#.
            .#..#.
            .####.
            ......
        "#,
        );
        let roi = img
            .roi(Rectangle::new(Coordinate::new(1, 1), Size::new(4, 4)))
            .unwrap();
        let r = connected_components::<Label32, Connectivity4>(&roi).unwrap();
        // The ring around the 4x4 ROI is a single component (12 pixels).
        assert_eq!(r.label_count, 1);
        let groups = partition_by_label(&r);
        assert_eq!(groups[0].len(), 12);
    }

    #[test]
    #[should_panic(expected = "does not match input")]
    fn into_size_mismatch_panics() {
        let img = BinaryImage::fill(4, 4, false);
        let mut out: Image<Label32> = Image::zero(8, 8);
        let _ = connected_components_into::<Label32, Connectivity4>(&img, &mut out);
    }

    #[test]
    fn label_count_matches_between_entry_points() {
        let img = img_from_str(
            r#"
            #.#.#
            .....
            #.#.#
            .....
            #.#.#
        "#,
        );
        let r = connected_components::<Label32, Connectivity4>(&img).unwrap();
        let mut out: Image<Label32> = Image::zero(5, 5);
        let count = connected_components_into::<Label32, Connectivity4>(&img, &mut out).unwrap();
        assert_eq!(count, r.label_count);
        // And the label images agree pixel-for-pixel.
        for y in 0..5 {
            for x in 0..5 {
                assert_eq!(out.pixel_at(x, y), r.labels.pixel_at(x, y));
            }
        }
    }

    #[test]
    fn total_foreground_area_matches_input() {
        let img = img_from_str(
            r#"
            ##..##
            .#..#.
            ..##..
            ##..##
            ##..##
        "#,
        );
        let r = connected_components::<Label32, Connectivity4>(&img).unwrap();
        // Count input foreground pixels.
        let mut fg = 0usize;
        for y in 0..img.height() {
            for x in 0..img.width() {
                if img.pixel_at(x, y) {
                    fg += 1;
                }
            }
        }
        // Sum component areas from stats.
        let (_, stats) = connected_components_with_stats::<Label32, Connectivity4>(&img).unwrap();
        let total_area: u64 = stats.iter().map(|s| s.area).sum();
        assert_eq!(total_area as usize, fg);
        assert_eq!(stats.len() as u64, r.label_count);
    }

    // Stats tests ─────────────────────────────────────────────────────

    #[test]
    fn stats_bbox_is_tight() {
        // Single 3x2 rectangle at (2..=4, 1..=2).
        let img = img_from_str(
            r#"
            .......
            ..###..
            ..###..
            .......
        "#,
        );
        let (_, stats) = connected_components_with_stats::<Label32, Connectivity4>(&img).unwrap();
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].area, 6);
        assert_eq!(stats[0].bbox_min, Coordinate::new(2, 1));
        assert_eq!(stats[0].bbox_max_inclusive, Coordinate::new(4, 2));
        let bb = stats[0].bbox();
        assert_eq!(bb, Rectangle::new(Coordinate::new(2, 1), Size::new(3, 2)));
    }

    #[test]
    fn stats_centroid_of_centred_square() {
        // 3x3 square at (1..=3, 1..=3) in a 5x5 image
        let img = img_from_str(
            r#"
            .....
            .###.
            .###.
            .###.
            .....
        "#,
        );
        let (_, stats) = connected_components_with_stats::<Label32, Connectivity4>(&img).unwrap();
        assert_eq!(stats.len(), 1);
        let (cx, cy) = stats[0].centroid();
        assert!((cx - 2.0).abs() < 1e-9);
        assert!((cy - 2.0).abs() < 1e-9);
    }

    #[test]
    fn stats_multi_component() {
        // Three components: 1x1 dot at (0,0); 2x2 square at (3..=4,
        // 0..=1); 1x3 vertical bar at (0..=0, 3..=5).
        let img = img_from_str(
            r#"
            #..##.
            ...##.
            ......
            #.....
            #.....
            #.....
        "#,
        );
        let (lab, stats) = connected_components_with_stats::<Label32, Connectivity4>(&img).unwrap();
        assert_eq!(lab.label_count, 3);
        assert_eq!(stats.len(), 3);
        // Find each component by area.
        let mut sorted = stats.clone();
        sorted.sort_by_key(|s| s.area);
        // 1x1 dot
        assert_eq!(sorted[0].area, 1);
        // 1x3 bar
        assert_eq!(sorted[1].area, 3);
        assert_eq!(
            sorted[1].bbox(),
            Rectangle::new(Coordinate::new(0, 3), Size::new(1, 3))
        );
        // 2x2 square
        assert_eq!(sorted[2].area, 4);
        assert_eq!(
            sorted[2].bbox(),
            Rectangle::new(Coordinate::new(3, 0), Size::new(2, 2))
        );
    }

    // Overflow test using a test-only narrow `LabelPixel`. ────────────

    /// Test-only label type with `MAX_LABEL = 3`. Wraps a `u8`.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
    struct TinyLabel(u8);

    impl ZeroablePixel for TinyLabel {
        fn zero() -> Self {
            TinyLabel(0)
        }
    }

    impl LabelPixel for TinyLabel {
        const MAX_LABEL: u64 = 3;
        fn from_label_index(i: u64) -> Option<Self> {
            if i == 0 || i > 3 {
                None
            } else {
                Some(TinyLabel(i as u8))
            }
        }
        fn to_label_index(self) -> u64 {
            self.0 as u64
        }
    }

    #[test]
    fn label_overflow_when_components_exceed_capacity() {
        // Five horizontally-isolated dots in a single row \u2192 5
        // components, but TinyLabel::MAX_LABEL == 3.
        let img = BinaryImage::from_vec(
            9,
            1,
            vec![true, false, true, false, true, false, true, false, true],
        )
        .unwrap();
        // First sanity-check with Label32 that there really are 5 components.
        let r = connected_components::<Label32, Connectivity4>(&img).unwrap();
        assert_eq!(r.label_count, 5);

        let err = connected_components::<TinyLabel, Connectivity4>(&img).unwrap_err();
        match err {
            Error::LabelOverflow { label_capacity } => assert_eq!(label_capacity, 3),
            other => panic!("expected LabelOverflow, got {:?}", other),
        }
    }

    #[test]
    fn label_overflow_succeeds_when_at_capacity() {
        // Exactly 3 components \u2014 should fit TinyLabel.
        let img = BinaryImage::from_vec(5, 1, vec![true, false, true, false, true]).unwrap();
        let mut out: Image<TinyLabel> = Image::zero(5, 1);
        let n = connected_components_into::<TinyLabel, Connectivity4>(&img, &mut out).unwrap();
        assert_eq!(n, 3);
        assert_eq!(out.pixel_at(0, 0), TinyLabel(1));
        assert_eq!(out.pixel_at(2, 0), TinyLabel(2));
        assert_eq!(out.pixel_at(4, 0), TinyLabel(3));
    }

    // Step 9 \u2014 trait audit on Labeling ────────────────────────────────

    #[test]
    fn labeling_is_clone_and_debug() {
        let img = BinaryImage::fill(2, 2, true);
        let r = connected_components::<Label32, Connectivity4>(&img).unwrap();
        let cloned: Labeling<Label32> = r.clone();
        assert_eq!(cloned.label_count, 1);
        let s = format!("{:?}", cloned);
        assert!(s.contains("Labeling"));
    }
}

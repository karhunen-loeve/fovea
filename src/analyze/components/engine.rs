//! Two-pass union-find connected-components engine.
//!
//! Implements the two-pass union-find engine. See the module-level docs
//! for the public surface and the worked 4×4 example.

use crate::image::{Image, ImageView, ImageViewMut, RasterImage};
use crate::{Coordinate, Error};
use crate::pixel::LabelPixel;

use super::Labeling;
use super::connectivity::Connectivity;
use super::measurements::BlobMeasurements;
use super::stats::ComponentStats;
use super::stats::sink::{NoStats, StatsSink, WithMeasurements, WithStats};
use super::union_find::UnionFind;

/// Compute the connected-component labeling of `image`, allocating a
/// fresh [`Labeling<L>`].
///
/// The label pixel type `L` and connectivity strategy `C` are named
/// explicitly by the caller (turbofish), e.g.
/// `connected_components::<Label32, Connectivity8>(&binary)`.
///
/// # Errors — Tier 2
///
/// Returns [`Error::LabelOverflow`] if the input contains more
/// connected components than `L::MAX_LABEL` can encode.
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
/// label image into `out` and returning `label_count`.
///
/// # Panics
///
/// Panics if `out.size() != image.size()` (Tier 3 — programmer bug).
///
/// # Errors \u2014 Tier 2
///
/// Returns [`Error::LabelOverflow`] if the input contains more
/// connected components than `L::MAX_LABEL` can encode.
pub fn connected_components_into<L, C>(
    image: &impl RasterImage<Pixel = bool>,
    out: &mut Image<L>,
) -> Result<u32, Error>
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
/// Stats are accumulated inline during pass 2 so the image is not
/// rescanned after labeling.
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
/// assert_eq!(stats[0].centroid(), fovea::CoordinateF64::new(0.5, 0.5));
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
    debug_assert_eq!(stats.len(), label_count as usize);
    Ok((
        Labeling {
            labels,
            label_count,
        },
        stats,
    ))
}

/// Compute the connected-component labeling of `image`, plus one
/// [`BlobMeasurements`] per foreground component: area, bounding box,
/// centroid sums, the raw second-order moment sums, and a 4-connected
/// boundary-pixel count. From these the shape descriptors
/// (equivalent diameter, orientation, eccentricity, circularity) are
/// derived on demand — see [`BlobMeasurements`].
///
/// Everything is accumulated in the *same* single pass 2 as the labeling;
/// there is no separate contour extraction. This is the heavier sibling
/// of [`connected_components_with_stats`]: it additionally runs a
/// per-foreground-pixel 4-neighbour boundary check to count the
/// boundary pixels. Reach for [`connected_components_with_stats`] when
/// you only need area / bounding box / centroid.
///
/// # Two non-obvious contracts
///
/// - **The boundary test is 4-connected regardless of the
///   labeling [`Connectivity`] `C`.** `C` decides which pixels form a
///   blob; the boundary test decides how its outline is counted. A
///   `Connectivity8` caller still gets a 4-connected boundary count —
///   correct, because the boundary is a property of the blob's pixel set,
///   not of the grouping rule.
/// - **Measurements are view-relative.** A blob clipped by the view edge
///   is measured as clipped: its cut edge counts toward the boundary and
///   `area` / `bbox` / centroid cover only the in-view part. When tiling,
///   use an overlapping margin and keep only blobs whose full extent lies
///   in the non-overlapped core.
///
/// # Errors — Tier 2
///
/// Returns [`Error::LabelOverflow`] if the input contains more connected
/// components than `L::MAX_LABEL` can encode.
///
/// # Examples
///
/// ```
/// use fovea::analyze::components::{
///     connected_components_with_measurements, Connectivity4,
/// };
/// use fovea::image::BinaryImage;
/// use fovea::pixel::Label32;
///
/// // A solid 3x3 square: area 9, boundary pixels 8 (4·3 − 4).
/// let img = BinaryImage::fill(3, 3, true);
/// let (lab, m) =
///     connected_components_with_measurements::<Label32, Connectivity4>(&img).unwrap();
/// assert_eq!(lab.label_count, 1);
/// assert_eq!(m[0].area, 9);
/// assert_eq!(m[0].boundary_pixels, 8);
/// ```
pub fn connected_components_with_measurements<L, C>(
    image: &impl RasterImage<Pixel = bool>,
) -> Result<(Labeling<L>, Vec<BlobMeasurements>), Error>
where
    L: LabelPixel,
    C: Connectivity,
{
    let mut labels = Image::<L>::zero(image.width(), image.height());
    let mut measurements: Vec<BlobMeasurements> = Vec::new();
    let label_count = {
        let mut sink = WithMeasurements {
            out: &mut measurements,
        };
        run::<L, C, _, WithMeasurements<'_>>(image, &mut labels, &mut sink)?
    };
    debug_assert_eq!(measurements.len(), label_count as usize);
    Ok((
        Labeling {
            labels,
            label_count,
        },
        measurements,
    ))
}

// ──────────────────────────────────────────────────────────────────────
// Engine \u2014 monomorphised over `S: StatsSink`.
// ──────────────────────────────────────────────────────────────────────

/// Maximum number of raster-preceding neighbours examined per pixel
/// across every shipped [`Connectivity`]. Currently 4 (for
/// [`Connectivity8`](super::Connectivity8)). The pass-1 inner buffer
/// `others: [u32; MAX_NEIGHBOURS]` hardcodes this constant. Adding a
/// connectivity with more predecessors requires lifting this and is
/// flagged for follow-up design.
const MAX_NEIGHBOURS: usize = 4;

fn run<L, C, I, S>(image: &I, out: &mut Image<L>, sink: &mut S) -> Result<u32, Error>
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
    // Provisional labels live in a flat W*H Vec<u32>, raster-scan
    // order. Zero is the background sentinel. `u32` is the width of
    // `LabelPixel::MAX_LABEL`, and this buffer is read and written by
    // both full-image passes, so its width is the engine's memory
    // traffic.
    let mut prov: Vec<u32> = vec![0; w * h];
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
            // unioned with `smallest` at the end. `0` doubles as the
            // "none seen yet" sentinel: provisional labels start at 1,
            // so no foreground neighbour can carry it (`u32::MAX` could
            // not serve — it is itself a valid label).
            let mut smallest: u32 = 0;
            let mut others: [u32; MAX_NEIGHBOURS] = [0; MAX_NEIGHBOURS];
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
                if smallest == 0 {
                    smallest = p;
                } else if p < smallest {
                    others[other_count] = smallest;
                    other_count += 1;
                    smallest = p;
                } else if p != smallest {
                    others[other_count] = p;
                    other_count += 1;
                }
            }

            let label = if smallest == 0 {
                // `make_set` returns `None` only when the u32 label
                // space itself is spent; a narrower `L` trips the
                // `MAX_LABEL` comparison long before.
                match uf.make_set() {
                    Some(new_label) if new_label <= L::MAX_LABEL => new_label,
                    _ => {
                        return Err(Error::LabelOverflow {
                            label_capacity: L::MAX_LABEL,
                        });
                    }
                }
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
    let mut compact: Vec<u32> = vec![0; uf.len()];
    // Compact labels assigned so far. `label_count + 1` cannot wrap:
    // a new root only appears while `label_count` is strictly below
    // the provisional-label total, which pass 1 capped at `u32::MAX`.
    let mut label_count: u32 = 0;

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
                    let assigned = label_count + 1;
                    compact[root as usize] = assigned;
                    label_count = assigned;
                    (assigned, true)
                } else {
                    (existing, false)
                };
                // Invariant: 0 < c <= label_count <= L::MAX_LABEL
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
                // The boundary neighbour-check is gated behind the sink's
                // `NEEDS_BOUNDARY` const so it is const-folded away (and
                // `is_boundary` reduces to a literal `false`) for the
                // `NoStats` / `WithStats` paths.
                let is_boundary = if S::NEEDS_BOUNDARY {
                    is_4_boundary(image, x, y)
                } else {
                    false
                };
                sink.record(c, first, Coordinate::new(x, y), is_boundary);
            }
        }
    }

    Ok(label_count)
}

/// Returns `true` if the foreground pixel at `(x, y)` is a 4-connected
/// boundary pixel: at least one of its four orthogonal neighbours is
/// background (`false`) or lies off the analyzed view.
///
/// The caller guarantees `(x, y)` is itself foreground. Off-view
/// neighbours (`image.get` → `None`) count as boundary — this is what
/// makes measurements *view-relative*: a blob clipped by the view edge
/// has its cut edge counted as boundary.
///
/// This test is fixed at 4-connectivity regardless of the labeling
/// [`Connectivity`], because a blob's boundary is a property of its
/// pixel *set*, not of the rule that grouped the pixels.
#[inline]
fn is_4_boundary<I>(image: &I, x: usize, y: usize) -> bool
where
    I: RasterImage<Pixel = bool>,
{
    // Left, right, up, down. `checked_sub` handles the x==0 / y==0 edges
    // (underflow → off-view → boundary); `get` handles the far edges.
    let left = x.checked_sub(1).and_then(|nx| image.get(nx, y));
    let right = image.get(x + 1, y);
    let up = y.checked_sub(1).and_then(|ny| image.get(x, ny));
    let down = image.get(x, y + 1);
    // A neighbour that is off-view (`None`) or background (`Some(false)`)
    // makes this a boundary pixel.
    [left, right, up, down]
        .iter()
        .any(|n| !matches!(n, Some(true)))
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
    use super::connected_components_with_measurements;
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
        assert_eq!(stats.len(), r.label_count as usize);
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
        let c = stats[0].centroid();
        assert!((c.x - 2.0).abs() < 1e-9);
        assert!((c.y - 2.0).abs() < 1e-9);
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
        const MAX_LABEL: u32 = 3;
        fn from_label_index(i: u32) -> Option<Self> {
            if i == 0 || i > 3 {
                None
            } else {
                Some(TinyLabel(i as u8))
            }
        }
        fn to_label_index(self) -> u32 {
            self.0 as u32
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

    // Measurements entry-point tests ──────────────────────────────────

    #[test]
    fn measurements_boundary_pixels_of_square_is_exact() {
        // Solid 5x5 square → boundary-pixel count = 4·5 − 4 = 16.
        let img = BinaryImage::fill(5, 5, true);
        let (lab, m) =
            connected_components_with_measurements::<Label32, Connectivity4>(&img).unwrap();
        assert_eq!(lab.label_count, 1);
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].area, 25);
        assert_eq!(m[0].boundary_pixels, 16);
        // Moments match the cheap path for the shared fields.
        assert_eq!(m[0].centroid(), crate::CoordinateF64::new(2.0, 2.0));
    }

    #[test]
    fn measurements_single_pixel_blob_is_finite() {
        let mut data = vec![false; 3 * 3];
        data[4] = true; // centre pixel
        let img = BinaryImage::from_vec(3, 3, data).unwrap();
        let (_, m) =
            connected_components_with_measurements::<Label32, Connectivity4>(&img).unwrap();
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].area, 1);
        assert_eq!(m[0].boundary_pixels, 1);
        assert_eq!(m[0].eccentricity(), 0.0);
        assert!(m[0].orientation().radians().is_finite());
        assert!(m[0].circularity().is_finite());
    }

    #[test]
    fn measurements_moments_match_cheap_path() {
        // The shared fields (area, bbox, sum_x/y) must agree with
        // connected_components_with_stats — regression guard for the sink
        // signature change.
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
        let (_, stats) = connected_components_with_stats::<Label32, Connectivity4>(&img).unwrap();
        let (_, meas) =
            connected_components_with_measurements::<Label32, Connectivity4>(&img).unwrap();
        assert_eq!(stats.len(), meas.len());
        for (s, m) in stats.iter().zip(meas.iter()) {
            assert_eq!(s.area, m.area);
            assert_eq!(s.bbox_min, m.bbox_min);
            assert_eq!(s.bbox_max_inclusive, m.bbox_max_inclusive);
            assert_eq!(s.sum_x, m.sum_x);
            assert_eq!(s.sum_y, m.sum_y);
        }
    }

    #[test]
    fn measurements_connectivity4_vs_8_boundary_pixels() {
        // Two pixels touching only diagonally. Under Connectivity4 they
        // are two separate 1-pixel blobs (boundary count 1 each); under
        // Connectivity8 they form one blob whose boundary count is the
        // 4-connected boundary count of the two-pixel set = 2 (each pixel
        // has a background 4-neighbour, so both are boundary pixels). The
        // boundary test stays 4-connected regardless of the labeling C.
        let img = img_from_str(
            r#"
            #..
            .#.
            ...
        "#,
        );
        let (lab4, m4) =
            connected_components_with_measurements::<Label32, Connectivity4>(&img).unwrap();
        assert_eq!(lab4.label_count, 2);
        assert_eq!(m4.len(), 2);
        for m in &m4 {
            assert_eq!(m.area, 1);
            assert_eq!(m.boundary_pixels, 1);
        }

        let (lab8, m8) =
            connected_components_with_measurements::<Label32, Connectivity8>(&img).unwrap();
        assert_eq!(lab8.label_count, 1);
        assert_eq!(m8.len(), 1);
        assert_eq!(m8[0].area, 2);
        assert_eq!(m8[0].boundary_pixels, 2);
    }

    #[test]
    fn measurements_roi_clips_boundary_pixels() {
        // A solid 3-wide, full-height bar in a 5x5 image; the ROI is the
        // left 3x3 corner. Inside the ROI the visible blob is a 3x3 solid
        // square, but its right and bottom edges are cut by the view, so
        // those pixels still count as boundary (off-view neighbour). The
        // clipped square measures area 9, boundary count 8 — as if it were a
        // standalone 3x3 square — confirming the view-relative contract.
        let img = img_from_str(
            r#"
            #####
            #####
            #####
            #####
            #####
        "#,
        );
        let roi = img
            .roi(Rectangle::new(Coordinate::new(0, 0), Size::new(3, 3)))
            .unwrap();
        let (lab, m) =
            connected_components_with_measurements::<Label32, Connectivity4>(&roi).unwrap();
        assert_eq!(lab.label_count, 1);
        assert_eq!(m[0].area, 9);
        // 3x3 block with all four view edges cutting it → every pixel is a
        // boundary pixel except the centre → boundary count 8.
        assert_eq!(m[0].boundary_pixels, 8);
        assert_eq!(m[0].bbox(), Rectangle::new(Coordinate::new(0, 0), Size::new(3, 3)));
    }
}

//! Border tracing and hierarchy derivation over the component labeling.

use crate::analyze::components::{Connectivity, Labeling, connected_components};
use crate::image::{Image, ImageView, RasterImage};
use crate::pixel::LabelPixel;
use crate::{Coordinate, Error};

use super::hierarchy::{ComponentContour, Contour, ContourHierarchy, ContourKind};

/// Clockwise Moore ring in image coordinates (y grows downward),
/// starting west: W, NW, N, NE, E, SE, S, SW.
const MOORE_RING: [(i64, i64); 8] = [
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
];

/// Trace every component's borders and derive the outer/hole hierarchy.
///
/// Runs the connected-components labeling twice — the foreground with the
/// caller's connectivity `C`, the background with its
/// [dual](Connectivity::Dual) (8-connected foreground pairs with
/// 4-connected background and vice versa; anything else breaks the
/// discrete Jordan property that makes "hole" well defined). Background
/// regions that touch the view edge are outside; the rest are holes, each
/// owned by the foreground component enclosing it. Every component's
/// outer border, and one inner border per hole, is then traced with
/// Moore-neighbor following.
///
/// Returns the foreground [`Labeling`] alongside the hierarchy: component
/// `i` in the hierarchy is label `i + 1` in the labeling, the same
/// convention as the stats and measurements functions, so all three join
/// on the label. The background labeling is internal and dropped.
///
/// # Contract: view-relative, like the measurements
///
/// Off-view is background to the tracer, so a blob clipped by the view
/// edge traces **along the clip edge** — the contour describes the
/// clipped shape, exactly as the component measurements measure it.
///
/// # Errors — Tier 2
///
/// Returns [`Error::LabelOverflow`] if either the foreground components
/// or the background regions outnumber what `L` can encode. (Every hole
/// and every outside region needs a background label, so `L` must cover
/// both counts.)
///
/// # Examples
///
/// ```
/// use fovea::analyze::contours::{Connectivity8, ContourKind, extract_contours};
/// use fovea::image::{BinaryImage, Image};
/// use fovea::pixel::Label32;
///
/// // A 5×5 ring: one component, one hole.
/// let img: BinaryImage = Image::generate(7, 7, |x, y| {
///     (1..6).contains(&x) && (1..6).contains(&y) && !(x == 3 && y == 3)
/// });
/// let (labeling, hierarchy) = extract_contours::<Label32, Connectivity8>(&img).unwrap();
/// assert_eq!(labeling.label_count, 1);
///
/// let ring = &hierarchy.components()[0];
/// assert_eq!(ring.holes().len(), 1);
/// assert_eq!(ring.euler_number(), 0);
/// assert_eq!(ring.outer().kind(), ContourKind::Outer);
/// assert_eq!(ring.holes()[0].kind(), ContourKind::Hole);
/// ```
pub fn extract_contours<L, C>(
    image: &impl RasterImage<Pixel = bool>,
) -> Result<(Labeling<L>, ContourHierarchy), Error>
where
    L: LabelPixel,
    C: Connectivity,
{
    let foreground = connected_components::<L, C>(image)?;

    let inverted: Image<bool> =
        Image::generate(image.width(), image.height(), |x, y| !image.pixel_at(x, y));
    let background = connected_components::<L, C::Dual>(&inverted)?;

    let fg_first = first_pixels(&foreground);
    let bg_first = first_pixels(&background);

    // Outside ⇔ the background region has a pixel on the view edge.
    let mut is_outside = vec![false; background.label_count as usize];
    mark_frame_labels(&background.labels, &mut is_outside);

    // A hole's owner: the pixel directly above its first raster pixel.
    // That pixel is foreground — a background pixel there would be
    // 4-adjacent (or 8-adjacent) to the hole and therefore part of it —
    // and it cannot be off-view, because a topmost row of 0 would make
    // the region touch the frame and be outside.
    let component_count = foreground.label_count as usize;
    let mut holes_of: Vec<Vec<Coordinate>> = vec![Vec::new(); component_count];
    let mut hole_owner: Vec<Option<usize>> = vec![None; background.label_count as usize];
    for (bg_index, &first) in bg_first.iter().enumerate() {
        if is_outside[bg_index] {
            continue;
        }
        let owner_label = label_at(&foreground.labels, first.x as i64, first.y as i64 - 1);
        debug_assert!(
            owner_label > 0,
            "the pixel above a hole's first pixel must be foreground"
        );
        hole_owner[bg_index] = Some(owner_label as usize - 1);
        holes_of[owner_label as usize - 1].push(first);
    }

    // A component's parent: the pixel above its first raster pixel is
    // background (same adjacency argument, or off-view for y = 0); if
    // that region is a hole, the hole's owner encloses this component.
    let components = fg_first
        .iter()
        .enumerate()
        .map(|(fg_index, &first)| {
            let enclosing = match label_at(&background.labels, first.x as i64, first.y as i64 - 1)
            {
                0 => None, // off-view: a topmost component is top-level
                bg_label => hole_owner[bg_label as usize - 1],
            };
            let label = fg_index as u32 + 1;
            let outer = Contour::new(
                trace_border(
                    &foreground.labels,
                    label,
                    first,
                    (first.x as i64 - 1, first.y as i64),
                ),
                ContourKind::Outer,
            );
            let holes = holes_of[fg_index]
                .iter()
                .map(|&hole_first| {
                    // Seed on the owning foreground pixel above the hole,
                    // backtracking into the hole.
                    let seed = Coordinate::new(hole_first.x, hole_first.y - 1);
                    Contour::new(
                        trace_border(
                            &foreground.labels,
                            label,
                            seed,
                            (hole_first.x as i64, hole_first.y as i64),
                        ),
                        ContourKind::Hole,
                    )
                })
                .collect();
            ComponentContour {
                outer,
                holes,
                enclosing,
            }
        })
        .collect();

    Ok((foreground, ContourHierarchy { components }))
}

/// Label index at `(x, y)`, `0` for background **and** off-view.
fn label_at<L: LabelPixel>(labels: &Image<L>, x: i64, y: i64) -> u32 {
    if x < 0 || y < 0 {
        return 0;
    }
    labels
        .get(x as usize, y as usize)
        .map_or(0, LabelPixel::to_label_index)
}

/// First raster-order pixel of every label, indexed by `label - 1`.
fn first_pixels<L: LabelPixel>(labeling: &Labeling<L>) -> Vec<Coordinate> {
    let mut first: Vec<Option<Coordinate>> = vec![None; labeling.label_count as usize];
    let mut seen = 0usize;
    'scan: for y in 0..labeling.labels.height() {
        let row = labeling.labels.row(y);
        for (x, pixel) in row.iter().enumerate() {
            let label = pixel.to_label_index();
            if label == 0 {
                continue;
            }
            let slot = &mut first[label as usize - 1];
            if slot.is_none() {
                *slot = Some(Coordinate::new(x, y));
                seen += 1;
                if seen == first.len() {
                    break 'scan;
                }
            }
        }
    }
    first
        .into_iter()
        .map(|c| c.expect("labels are dense 1..=label_count"))
        .collect()
}

/// Set `out[label - 1]` for every label appearing on the view edge.
fn mark_frame_labels<L: LabelPixel>(labels: &Image<L>, out: &mut [bool]) {
    let (w, h) = (labels.width(), labels.height());
    if w == 0 || h == 0 {
        return;
    }
    let mark = |x: usize, y: usize, out: &mut [bool]| {
        let label = labels.pixel_at(x, y).to_label_index();
        if label > 0 {
            out[label as usize - 1] = true;
        }
    };
    for x in 0..w {
        mark(x, 0, out);
        mark(x, h - 1, out);
    }
    for y in 0..h {
        mark(0, y, out);
        mark(w - 1, y, out);
    }
}

/// Moore-neighbor border following over the pixels of one label.
///
/// Starts at `start` (a pixel of `label`) with `backtrack` (a
/// non-`label` position 8-adjacent to `start`) and walks the border
/// clockwise. Terminates with Jacob's criterion phrased as "stop when
/// the first move is about to repeat" — the naive "stop on re-entering
/// the start with the original backtrack" never fires on 1-px-thin
/// shapes, where the trace re-enters the start from the far side with a
/// different backtrack.
fn trace_border<L: LabelPixel>(
    labels: &Image<L>,
    label: u32,
    start: Coordinate,
    backtrack: (i64, i64),
) -> Vec<Coordinate> {
    let s = (start.x as i64, start.y as i64);
    let matches = |p: (i64, i64)| label_at(labels, p.0, p.1) == label;
    debug_assert!(matches(s) && !matches(backtrack));

    let mut contour = vec![start];
    let mut current = s;
    let mut back = backtrack;
    let mut first_move: Option<(i64, i64)> = None;
    // Every border pixel is visited at most 4 times (once per approach
    // side), so this bound is unreachable in correct code.
    let budget = 4 * labels.width() * labels.height() + 8;

    for _ in 0..budget {
        let back_dir = MOORE_RING
            .iter()
            .position(|&(dx, dy)| (current.0 + dx, current.1 + dy) == back)
            .expect("backtrack is 8-adjacent to the current pixel");
        let mut next = None;
        for k in 1..=8 {
            let dir = (back_dir + k) % 8;
            let p = (current.0 + MOORE_RING[dir].0, current.1 + MOORE_RING[dir].1);
            if matches(p) {
                next = Some(p);
                break;
            }
            back = p; // last rejected position becomes the next backtrack
        }
        let Some(next) = next else {
            return contour; // isolated pixel: nothing 8-adjacent matches
        };
        if current == s {
            match first_move {
                None => first_move = Some(next),
                Some(first) if first == next => {
                    // About to repeat the first move: the loop is closed.
                    // The previous step pushed `start` again; drop it.
                    contour.pop();
                    return contour;
                }
                Some(_) => {}
            }
        }
        current = next;
        contour.push(Coordinate::new(current.0 as usize, current.1 as usize));
    }
    unreachable!("border trace exceeded the visit budget");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyze::components::{Connectivity4, Connectivity8};
    use crate::image::BinaryImage;
    use crate::pixel::Label32;

    /// Build a binary image from ASCII art: `#` is foreground.
    fn img_from_str(s: &str) -> BinaryImage {
        let rows: Vec<&str> = s.trim().lines().map(str::trim).collect();
        let height = rows.len();
        let width = rows[0].len();
        assert!(rows.iter().all(|r| r.len() == width), "ragged fixture");
        Image::generate(width, height, |x, y| rows[y].as_bytes()[x] == b'#')
    }

    fn extract(img: &BinaryImage) -> (Labeling<Label32>, ContourHierarchy) {
        extract_contours::<Label32, Connectivity8>(img).unwrap()
    }

    #[test]
    fn solid_square_traces_its_border() {
        let img = img_from_str(
            "......
             .####.
             .####.
             .####.
             .####.
             ......",
        );
        let (labeling, hierarchy) = extract(&img);
        assert_eq!(labeling.label_count, 1);
        let component = &hierarchy.components()[0];
        assert!(component.holes().is_empty());
        assert_eq!(component.enclosing(), None);
        assert_eq!(component.euler_number(), 1);

        let outer = component.outer();
        // 4×4 square: 12 border pixels, polygon 3×3.
        assert_eq!(outer.points().len(), 12);
        assert_eq!(outer.area(), 9.0);
        assert_eq!(outer.perimeter(), 12.0);
        // Every traced point is a foreground pixel.
        for p in outer.points() {
            assert!(img.pixel_at(p.x, p.y), "{p:?} is not foreground");
        }
    }

    #[test]
    fn ring_yields_hole_with_inner_border_on_foreground() {
        let img = img_from_str(
            ".........
             .#######.
             .#######.
             .##...##.
             .##...##.
             .##...##.
             .#######.
             .#######.
             .........",
        );
        let (labeling, hierarchy) = extract(&img);
        assert_eq!(labeling.label_count, 1);
        let ring = &hierarchy.components()[0];
        assert_eq!(ring.holes().len(), 1);
        assert_eq!(ring.euler_number(), 0);

        let hole = &ring.holes()[0];
        assert_eq!(hole.kind(), ContourKind::Hole);
        // Inner border runs on foreground pixels around the 3×3 hole and
        // must enclose at least the hole's pixel area.
        for p in hole.points() {
            assert!(img.pixel_at(p.x, p.y), "{p:?} is not foreground");
        }
        assert!(hole.area() >= 9.0, "hole border area {}", hole.area());

        // Winding: outer clockwise on screen (positive y-down shoelace),
        // hole counterclockwise (negative) — the documented property.
        assert!(signed_doubled_area(ring.outer().points()) > 0);
        assert!(signed_doubled_area(hole.points()) < 0);
    }

    fn signed_doubled_area(points: &[Coordinate]) -> i64 {
        let mut acc = 0i64;
        for i in 0..points.len() {
            let a = points[i];
            let b = points[(i + 1) % points.len()];
            acc += (a.x as i64) * (b.y as i64) - (b.x as i64) * (a.y as i64);
        }
        acc
    }

    #[test]
    fn four_level_nesting_resolves_the_parent_chain() {
        let img = img_from_str(
            "...............
             .#############.
             .#...........#.
             .#.#########.#.
             .#.#.......#.#.
             .#.#.#####.#.#.
             .#.#.#...#.#.#.
             .#.#.#.#.#.#.#.
             .#.#.#...#.#.#.
             .#.#.#####.#.#.
             .#.#.......#.#.
             .#.#########.#.
             .#...........#.
             .#############.
             ...............",
        );
        let (labeling, hierarchy) = extract(&img);
        // Three nested rings and the dot: a four-deep chain.
        assert_eq!(labeling.label_count, 4);
        let parents: Vec<Option<usize>> = hierarchy
            .components()
            .iter()
            .map(ComponentContour::enclosing)
            .collect();
        assert_eq!(parents, vec![None, Some(0), Some(1), Some(2)]);
        let holes: Vec<usize> = hierarchy
            .components()
            .iter()
            .map(|component| component.holes().len())
            .collect();
        assert_eq!(holes, vec![1, 1, 1, 0]);
    }

    #[test]
    fn two_holes_give_euler_minus_one() {
        let img = img_from_str(
            "...........
             .#########.
             .#.##...##.
             .#.##...##.
             .#########.
             ...........",
        );
        let (_, hierarchy) = extract(&img);
        assert_eq!(hierarchy.components().len(), 1);
        assert_eq!(hierarchy.components()[0].euler_number(), -1);
        assert_eq!(hierarchy.components()[0].holes().len(), 2);
    }

    #[test]
    fn thin_and_single_pixel_shapes_trace_and_terminate() {
        let img = img_from_str(
            "#....
             .....
             .###.
             .....
             ..#..",
        );
        let (labeling, hierarchy) = extract(&img);
        assert_eq!(labeling.label_count, 3);
        // Corner pixel: single-point contour, top-level despite touching
        // the frame at (0, 0).
        let corner = &hierarchy.components()[0];
        assert_eq!(corner.outer().points(), &[Coordinate::new(0, 0)]);
        assert_eq!(corner.enclosing(), None);
        // 3-px line traces out and back: 4 points, zero enclosed area.
        let line = &hierarchy.components()[1];
        assert_eq!(line.outer().points().len(), 4);
        assert_eq!(line.outer().area(), 0.0);
        // All three are top-level and hole-free.
        for component in hierarchy.components() {
            assert_eq!(component.enclosing(), None);
            assert!(component.holes().is_empty());
            assert_eq!(component.euler_number(), 1);
        }
    }

    #[test]
    fn clipped_blob_traces_along_the_view_edge() {
        // Foreground bleeding off every edge: the trace follows the frame.
        let img = img_from_str(
            "####
             ####
             ####
             ####",
        );
        let (labeling, hierarchy) = extract(&img);
        assert_eq!(labeling.label_count, 1);
        let outer = hierarchy.components()[0].outer();
        assert_eq!(outer.points().len(), 12);
        assert_eq!(outer.area(), 9.0);
    }

    #[test]
    fn connectivity4_diagonal_pixels_are_separate_components() {
        // Two pixels touching diagonally: one component under 8, two
        // under 4 (whose dual labels the background 8-connected).
        let img = img_from_str(
            "#.
             .#",
        );
        let (labeling8, _) = extract_contours::<Label32, Connectivity8>(&img).unwrap();
        assert_eq!(labeling8.label_count, 1);
        let (labeling4, hierarchy4) = extract_contours::<Label32, Connectivity4>(&img).unwrap();
        assert_eq!(labeling4.label_count, 2);
        for component in hierarchy4.components() {
            assert_eq!(component.enclosing(), None);
            assert!(component.holes().is_empty());
        }
    }

    #[test]
    fn hierarchy_joins_the_labeling_on_the_label() {
        let img = img_from_str(
            "##...
             ##...
             ...##
             ...##",
        );
        let (labeling, hierarchy) = extract(&img);
        assert_eq!(labeling.label_count, 2);
        for (index, component) in hierarchy.components().iter().enumerate() {
            let label = index as u32 + 1;
            let first = component.outer().points()[0];
            assert_eq!(
                labeling.labels.pixel_at(first.x, first.y).to_label_index(),
                label
            );
            assert!(std::ptr::eq(
                hierarchy.component_for_label(label).unwrap(),
                component
            ));
        }
    }

    #[test]
    fn empty_and_blank_images_yield_no_components() {
        let blank: BinaryImage = Image::fill(5, 4, false);
        let (labeling, hierarchy) = extract(&blank);
        assert_eq!(labeling.label_count, 0);
        assert!(hierarchy.components().is_empty());
    }

    #[test]
    fn chain_code_round_trips_every_traced_contour() {
        let img = img_from_str(
            ".........
             .#######.
             .##...##.
             .##.#.##.
             .##...##.
             .#######.
             .........",
        );
        let (_, hierarchy) = extract(&img);
        for component in hierarchy.components() {
            let all = std::iter::once(component.outer()).chain(component.holes());
            for contour in all {
                let chain = contour.chain_code();
                assert_eq!(chain.to_points(), contour.points());
                assert_eq!(chain.start(), contour.points()[0]);
            }
        }
    }
}

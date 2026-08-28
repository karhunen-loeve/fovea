//! Free polygon-geometry functions over vertex lists.
//!
//! These operate on any `&[Coordinate]` interpreted as a **closed**
//! polygon (an implicit edge connects the last vertex back to the first),
//! so they serve both traced [`Contour`](super::Contour) point chains and
//! the simplified vertex lists produced by [`approximate_polygon`].

use crate::{Coordinate, CoordinateF64, Tolerance};

/// Twice the signed shoelace area, exact in integers.
///
/// Positive for clockwise vertex order in image coordinates (y grows
/// downward). `i128` keeps every product exact for any in-memory image.
fn shoelace_doubled(points: &[Coordinate]) -> i128 {
    let mut acc: i128 = 0;
    for i in 0..points.len() {
        let a = points[i];
        let b = points[(i + 1) % points.len()];
        acc += (a.x as i128) * (b.y as i128) - (b.x as i128) * (a.y as i128);
    }
    acc
}

/// Area enclosed by a closed polygon (shoelace formula), in square pixels.
///
/// The result is the area enclosed by the **vertex path itself**. For a
/// traced border, vertices are pixel *centers*, so the value is smaller
/// than the pixel-count area: a solid 6×6-pixel square traces a 5×5
/// polygon and measures `25.0`, not `36`. The two answer different
/// questions; pixel-count area remains available from the component
/// measurements.
///
/// Fewer than 3 vertices enclose nothing and return `0.0`. A
/// self-overlapping path (a traced 1-px line, which runs out and back)
/// also returns `0.0` — its enclosed area genuinely is zero.
///
/// # Examples
///
/// ```
/// use fovea::Coordinate;
/// use fovea::analyze::contours::polygon_area;
///
/// let square = [
///     Coordinate::new(0, 0),
///     Coordinate::new(4, 0),
///     Coordinate::new(4, 4),
///     Coordinate::new(0, 4),
/// ];
/// assert_eq!(polygon_area(&square), 16.0);
/// ```
#[must_use]
pub fn polygon_area(points: &[Coordinate]) -> f64 {
    if points.len() < 3 {
        return 0.0;
    }
    (shoelace_doubled(points).unsigned_abs() as f64) / 2.0
}

/// Length of a closed polygon's outline, in pixels.
///
/// Sums the Euclidean edge lengths **including the implicit closing edge**
/// from the last vertex back to the first. Diagonal steps of a traced
/// border count √2. Fewer than 2 vertices return `0.0`; exactly 2 count
/// the edge twice (out and back — the closed reading of a degenerate
/// polygon).
///
/// This is the exact length of the polygon. As an estimate of a *smooth*
/// outline that was rasterised, the traced chain overestimates by roughly
/// 5–8% (the staircase effect); simplifying with [`approximate_polygon`]
/// first removes most of that. See
/// [`Contour::circularity`](super::Contour::circularity) for measured
/// numbers.
///
/// # Examples
///
/// ```
/// use fovea::Coordinate;
/// use fovea::analyze::contours::polygon_perimeter;
///
/// let square = [
///     Coordinate::new(0, 0),
///     Coordinate::new(4, 0),
///     Coordinate::new(4, 4),
///     Coordinate::new(0, 4),
/// ];
/// assert_eq!(polygon_perimeter(&square), 16.0);
/// ```
#[must_use]
pub fn polygon_perimeter(points: &[Coordinate]) -> f64 {
    if points.len() < 2 {
        return 0.0;
    }
    (0..points.len())
        .map(|i| {
            let a = points[i];
            let b = points[(i + 1) % points.len()];
            (a.x as f64 - b.x as f64).hypot(a.y as f64 - b.y as f64)
        })
        .sum()
}

/// Centroid of the region enclosed by a closed polygon.
///
/// Computed from the shoelace decomposition (the area-weighted mean of
/// the polygon's interior, not the mean of its vertices). Returns `None`
/// when the polygon encloses no area (fewer than 3 vertices, collinear
/// vertices, or an out-and-back degenerate path) — there is no region to
/// have a centre.
///
/// # Examples
///
/// ```
/// use fovea::Coordinate;
/// use fovea::analyze::contours::polygon_centroid;
///
/// let square = [
///     Coordinate::new(0, 0),
///     Coordinate::new(4, 0),
///     Coordinate::new(4, 4),
///     Coordinate::new(0, 4),
/// ];
/// let c = polygon_centroid(&square).unwrap();
/// assert_eq!((c.x, c.y), (2.0, 2.0));
/// ```
#[must_use]
pub fn polygon_centroid(points: &[Coordinate]) -> Option<CoordinateF64> {
    if points.len() < 3 {
        return None;
    }
    let doubled = shoelace_doubled(points);
    if doubled == 0 {
        return None;
    }
    let (mut cx, mut cy): (i128, i128) = (0, 0);
    for i in 0..points.len() {
        let a = points[i];
        let b = points[(i + 1) % points.len()];
        let cross = (a.x as i128) * (b.y as i128) - (b.x as i128) * (a.y as i128);
        cx += (a.x as i128 + b.x as i128) * cross;
        cy += (a.y as i128 + b.y as i128) * cross;
    }
    // centroid = Σ(a+b)·cross / (6·A_signed), with 2·A_signed = doubled.
    let denom = 3.0 * doubled as f64;
    Some(CoordinateF64::new(cx as f64 / denom, cy as f64 / denom))
}

/// Cross product `(a − o) × (b − o)`, exact in integers.
fn cross(o: Coordinate, a: Coordinate, b: Coordinate) -> i128 {
    let (ox, oy) = (o.x as i128, o.y as i128);
    (a.x as i128 - ox) * (b.y as i128 - oy) - (a.y as i128 - oy) * (b.x as i128 - ox)
}

/// Convex hull of a point set (Andrew's monotone chain), in integer
/// arithmetic — no epsilon, no float comparison.
///
/// Returns the hull vertices in **clockwise order on screen** (image
/// coordinates, y grows downward; equivalently counterclockwise in
/// mathematical orientation), starting from the lexicographically
/// smallest point. Collinear points on a hull edge are **not** included —
/// every returned vertex is a strict corner. Degenerate inputs degrade
/// honestly: an empty slice returns empty, a single point returns that
/// point, and a fully collinear set returns its two extreme points.
///
/// The input need not be sorted, closed, or free of duplicates.
///
/// # Examples
///
/// ```
/// use fovea::Coordinate;
/// use fovea::analyze::contours::convex_hull;
///
/// // A square plus an interior point: the hull is the square.
/// let points = [
///     Coordinate::new(0, 0),
///     Coordinate::new(4, 0),
///     Coordinate::new(2, 2), // interior — not a hull vertex
///     Coordinate::new(4, 4),
///     Coordinate::new(0, 4),
/// ];
/// let hull = convex_hull(&points);
/// assert_eq!(hull.len(), 4);
/// assert!(!hull.contains(&Coordinate::new(2, 2)));
/// ```
#[must_use]
pub fn convex_hull(points: &[Coordinate]) -> Vec<Coordinate> {
    let mut sorted: Vec<Coordinate> = points.to_vec();
    sorted.sort_by_key(|p| (p.x, p.y));
    sorted.dedup();
    if sorted.len() < 3 {
        return sorted;
    }

    let mut hull: Vec<Coordinate> = Vec::with_capacity(sorted.len() + 1);
    // Lower hull, then upper hull over the reversed order; `<= 0` pops
    // collinear points so only strict corners remain.
    for pass in 0..2 {
        let base = hull.len();
        let iter: Box<dyn Iterator<Item = &Coordinate>> = if pass == 0 {
            Box::new(sorted.iter())
        } else {
            Box::new(sorted.iter().rev())
        };
        for &p in iter {
            while hull.len() >= base + 2
                && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0
            {
                hull.pop();
            }
            hull.push(p);
        }
        hull.pop(); // each pass's last point is the other pass's first
    }
    hull
}

/// Simplify a closed polygon with the Douglas–Peucker algorithm.
///
/// Returns the subset of `points` (in their original order) whose removal
/// would move the outline by more than `tolerance` pixels: recursively,
/// the vertex farthest from the current anchor chord is kept iff its
/// perpendicular distance exceeds the tolerance. The input is treated as
/// **closed**; the two initial anchors are the first vertex and the vertex
/// farthest from it.
///
/// A tolerance of `0.0` removes exactly the collinear vertices. The
/// simplification is **never applied implicitly** by any other operation
/// in this crate — a traced [`Contour`](super::Contour) keeps every border
/// pixel until the caller names this step, because the right ε is a claim
/// about *your* images (how smooth the true outline is), not a library
/// default.
///
/// Inputs with fewer than 3 vertices are returned unchanged.
///
/// # Examples
///
/// ```
/// use fovea::Coordinate;
/// use fovea::analyze::contours::approximate_polygon;
/// use fovea::tolerance;
///
/// // A traced 4×4 square border has 12 points; only its 4 corners
/// // survive collinear removal.
/// let border: Vec<Coordinate> = [
///     (0, 0), (1, 0), (2, 0), (3, 0),
///     (3, 1), (3, 2), (3, 3),
///     (2, 3), (1, 3), (0, 3),
///     (0, 2), (0, 1),
/// ]
/// .map(Coordinate::from)
/// .to_vec();
/// let corners = approximate_polygon(&border, tolerance!(0.0));
/// assert_eq!(corners.len(), 4);
/// ```
#[must_use]
pub fn approximate_polygon(points: &[Coordinate], tolerance: Tolerance) -> Vec<Coordinate> {
    if points.len() < 3 {
        return points.to_vec();
    }
    let first = points[0];
    let dist2 = |p: Coordinate| {
        (p.x as f64 - first.x as f64).powi(2) + (p.y as f64 - first.y as f64).powi(2)
    };
    let farthest = points
        .iter()
        .enumerate()
        .max_by(|(_, p), (_, q)| dist2(**p).total_cmp(&dist2(**q)))
        .map(|(i, _)| i)
        .expect("non-empty by the length check above");

    let mut out = Vec::new();
    // Simplify the two halves; each call emits its final vertex's
    // predecessor chain but not the final vertex itself, and the halves
    // share endpoints, so the concatenation is the closed vertex list.
    simplify_open(&points[..=farthest], tolerance.get(), &mut out);
    let mut second: Vec<Coordinate> = points[farthest..].to_vec();
    second.push(points[0]);
    simplify_open(&second, tolerance.get(), &mut out);
    out
}

/// Douglas–Peucker over an open polyline. Emits every kept vertex except
/// the last one (the caller chains segments that share endpoints).
fn simplify_open(points: &[Coordinate], epsilon: f64, out: &mut Vec<Coordinate>) {
    if points.len() < 3 {
        if let Some((_, rest)) = points.split_last() {
            out.extend_from_slice(rest);
        }
        return;
    }
    let (a, b) = (points[0], points[points.len() - 1]);
    let (ax, ay) = (a.x as f64, a.y as f64);
    let (bx, by) = (b.x as f64, b.y as f64);
    let chord = (bx - ax).hypot(by - ay);
    let mut far = (0usize, -1.0f64);
    for (i, p) in points.iter().enumerate().take(points.len() - 1).skip(1) {
        let (px, py) = (p.x as f64, p.y as f64);
        // Perpendicular distance to the chord; distance to `a` when the
        // chord is degenerate (closed-polygon halves can share endpoints).
        let d = if chord == 0.0 {
            (px - ax).hypot(py - ay)
        } else {
            ((bx - ax) * (ay - py) - (ax - px) * (by - ay)).abs() / chord
        };
        if d > far.1 {
            far = (i, d);
        }
    }
    if far.1 > epsilon {
        simplify_open(&points[..=far.0], epsilon, out);
        simplify_open(&points[far.0..], epsilon, out);
    } else {
        out.push(a);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tolerance;

    fn c(x: usize, y: usize) -> Coordinate {
        Coordinate::new(x, y)
    }

    #[test]
    fn area_of_triangle() {
        let tri = [c(0, 0), c(4, 0), c(0, 4)];
        assert_eq!(polygon_area(&tri), 8.0);
    }

    #[test]
    fn area_degenerate_is_zero() {
        assert_eq!(polygon_area(&[]), 0.0);
        assert_eq!(polygon_area(&[c(1, 1)]), 0.0);
        assert_eq!(polygon_area(&[c(1, 1), c(5, 5)]), 0.0);
        // Collinear.
        assert_eq!(polygon_area(&[c(0, 0), c(2, 0), c(4, 0)]), 0.0);
    }

    #[test]
    fn perimeter_counts_the_closing_edge() {
        let tri = [c(0, 0), c(3, 0), c(3, 4)];
        // 3 + 4 + 5.
        assert_eq!(polygon_perimeter(&tri), 12.0);
        // Two points: out and back.
        assert_eq!(polygon_perimeter(&[c(0, 0), c(3, 4)]), 10.0);
        assert_eq!(polygon_perimeter(&[c(7, 7)]), 0.0);
    }

    #[test]
    fn centroid_of_square_and_of_degenerate() {
        let sq = [c(1, 1), c(5, 1), c(5, 5), c(1, 5)];
        let ctr = polygon_centroid(&sq).unwrap();
        assert_eq!((ctr.x, ctr.y), (3.0, 3.0));
        assert!(polygon_centroid(&[c(0, 0), c(2, 0), c(4, 0)]).is_none());
        assert!(polygon_centroid(&[c(0, 0)]).is_none());
    }

    #[test]
    fn centroid_is_area_weighted_not_vertex_mean() {
        // An L-shape whose vertex mean differs from its region centroid.
        let l = [c(0, 0), c(4, 0), c(4, 2), c(2, 2), c(2, 6), c(0, 6)];
        let ctr = polygon_centroid(&l).unwrap();
        // Region = 4×2 rectangle (centroid (2,1), area 8) + 2×4 rectangle
        // (centroid (1,4), area 8) → combined (1.5, 2.5).
        assert!((ctr.x - 1.5).abs() < 1e-12, "cx = {}", ctr.x);
        assert!((ctr.y - 2.5).abs() < 1e-12, "cy = {}", ctr.y);
    }

    #[test]
    fn hull_orientation_and_strict_corners() {
        let pts = [c(0, 0), c(2, 0), c(4, 0), c(4, 4), c(0, 4), c(2, 2)];
        let hull = convex_hull(&pts);
        // Collinear (2,0) and interior (2,2) removed.
        assert_eq!(hull, vec![c(0, 0), c(4, 0), c(4, 4), c(0, 4)]);
        // Clockwise on screen: positive shoelace in y-down coordinates.
        assert!(super::shoelace_doubled(&hull) > 0);
    }

    #[test]
    fn hull_degenerate_inputs() {
        assert!(convex_hull(&[]).is_empty());
        assert_eq!(convex_hull(&[c(3, 3)]), vec![c(3, 3)]);
        assert_eq!(convex_hull(&[c(3, 3), c(3, 3)]), vec![c(3, 3)]);
        // Fully collinear: the two extremes.
        assert_eq!(
            convex_hull(&[c(0, 0), c(1, 1), c(2, 2), c(3, 3)]),
            vec![c(0, 0), c(3, 3)]
        );
    }

    #[test]
    fn hull_is_invariant_to_input_order() {
        let mut pts = vec![c(0, 0), c(4, 0), c(4, 4), c(0, 4), c(2, 2), c(3, 1)];
        let expected = convex_hull(&pts);
        pts.reverse();
        assert_eq!(convex_hull(&pts), expected);
        pts.swap(0, 3);
        assert_eq!(convex_hull(&pts), expected);
    }

    #[test]
    fn approximate_keeps_corners_drops_collinear() {
        let border: Vec<Coordinate> = [
            (0, 0),
            (1, 0),
            (2, 0),
            (3, 0),
            (3, 1),
            (3, 2),
            (3, 3),
            (2, 3),
            (1, 3),
            (0, 3),
            (0, 2),
            (0, 1),
        ]
        .map(Coordinate::from)
        .to_vec();
        let simplified = approximate_polygon(&border, tolerance!(0.0));
        assert_eq!(simplified.len(), 4);
        for corner in [c(0, 0), c(3, 0), c(3, 3), c(0, 3)] {
            assert!(simplified.contains(&corner), "missing {corner:?}");
        }
        // Area and perimeter are unchanged by collinear removal.
        assert_eq!(polygon_area(&simplified), polygon_area(&border));
        assert_eq!(polygon_perimeter(&simplified), polygon_perimeter(&border));
    }

    #[test]
    fn approximate_with_tolerance_removes_small_bumps() {
        // A long edge with a 1-px bump: ε = 1.5 flattens it, ε = 0.5 keeps it.
        let outline: Vec<Coordinate> = [
            (0, 0),
            (3, 0),
            (4, 1), // the bump, 1 px off the (0,0)–(9,0) chord
            (5, 0),
            (9, 0),
            (9, 5),
            (0, 5),
        ]
        .map(Coordinate::from)
        .to_vec();
        let coarse = approximate_polygon(&outline, tolerance!(1.5));
        assert!(
            !coarse.contains(&c(4, 1)),
            "bump survived ε=1.5: {coarse:?}"
        );
        let fine = approximate_polygon(&outline, tolerance!(0.5));
        assert!(fine.contains(&c(4, 1)), "bump lost at ε=0.5: {fine:?}");
    }

    #[test]
    fn approximate_degenerate_inputs_pass_through() {
        assert_eq!(approximate_polygon(&[], tolerance!(1.0)), vec![]);
        assert_eq!(
            approximate_polygon(&[c(1, 2)], tolerance!(1.0)),
            vec![c(1, 2)]
        );
        assert_eq!(
            approximate_polygon(&[c(1, 2), c(3, 4)], tolerance!(1.0)),
            vec![c(1, 2), c(3, 4)]
        );
    }
}

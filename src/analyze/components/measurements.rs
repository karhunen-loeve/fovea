//! Shape descriptors for connected components.
//!
//! See [`BlobMeasurements`]. Accumulated inline during pass 2 of the
//! engine when callers invoke
//! [`connected_components_with_measurements`](super::connected_components_with_measurements).
//!
//! Where [`ComponentStats`](super::ComponentStats) stops at `area`,
//! bounding box, and centroid sums, `BlobMeasurements` adds the raw
//! second-order moment sums and a boundary-pixel count, from which the
//! standard machine-vision shape descriptors (equivalent diameter,
//! orientation, eccentricity, circularity) are derived on demand in
//! `f64`. All of it comes from the *same* single accumulation pass — no
//! separate contour extraction.

use crate::analyze::statistics::moments::{axis_eccentricity, axis_orientation};
use crate::{AxialOrientation, Coordinate, CoordinateF64, Rectangle, Size};

/// Shape descriptors for one connected component.
///
/// The struct stores only *raw* per-pixel sums (all integer); the shape
/// quantities are derived methods computed in `f64` on demand, following
/// the crate's "store raw, derive on demand" convention (see
/// [`ComponentStats`](super::ComponentStats)).
///
/// # Field semantics
///
/// - `area` — pixel count of the component.
/// - `bbox_min` / `bbox_max_inclusive` — inclusive corners of the
///   tightest axis-aligned bounding box (see
///   [`ComponentStats`](super::ComponentStats)).
/// - `sum_x` / `sum_y` — first raw moments (`Σx`, `Σy`), for the centroid.
/// - `sum_x2` / `sum_y2` / `sum_xy` — second raw moments (`Σx²`, `Σy²`,
///   `Σxy`), for the central second moments.
/// - `boundary_pixels` — count of 4-connected boundary pixels (see below).
///
/// # Two non-obvious contracts
///
/// 1. **The boundary test is 4-connected, independent of the
///    labeling [`Connectivity`](super::Connectivity).** The connectivity
///    parameter decides *which pixels form a blob*; the boundary test
///    decides *how a blob's outline is counted* once the blob exists.
///    They are orthogonal — a `Connectivity8` caller still gets a
///    4-connected boundary count, which is correct because the boundary is
///    a property of the blob's pixel set, not of the rule that grouped it.
///    A pixel is a boundary pixel iff at least one of its four orthogonal
///    neighbours is background or off the analyzed view.
///
/// 2. **All measurements are view-relative.** A blob clipped by the view
///    edge is measured *as clipped*: its cut edges count toward the
///    boundary, and `area` / `bbox` / centroid describe only the in-view
///    part. This matches [`ComponentStats`](super::ComponentStats) on an
///    ROI; the boundary count merely makes it more visible. When tiling a
///    large image, give each tile an overlapping margin and keep only
///    blobs whose full extent lies inside the non-overlapped core.
///
/// # Overflow bound
///
/// `sum_x2` / `sum_y2` can reach `area · (dim − 1)²`, which stays within
/// `u64` for images up to ~46 000 px on a side. A fully-foreground image
/// larger than that can overflow the squared sums; this is guarded by a
/// `debug_assert` during accumulation. Widening the fields would be a
/// semver-major change, so `u64` is the deliberate choice for the
/// industrial frame sizes this crate targets.
///
/// The 4-connected boundary-*pixel* count has a known discretisation
/// bias: it undercounts diagonal outline (a √2 Euclidean rim step counts
/// as one pixel), so [`circularity`](BlobMeasurements::circularity) of a
/// rasterised disc lands *above* 1 (converging to ≈1.25), while an
/// axis-aligned square lands at ≈π/4. Compare against tolerance bands,
/// not exact 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlobMeasurements {
    /// Number of pixels in the component.
    pub area: u64,
    /// Inclusive top-left corner of the bounding box.
    pub bbox_min: Coordinate,
    /// Inclusive bottom-right corner of the bounding box.
    pub bbox_max_inclusive: Coordinate,
    /// First raw moment `Σx`.
    pub sum_x: u64,
    /// First raw moment `Σy`.
    pub sum_y: u64,
    /// Second raw moment `Σx²`.
    pub sum_x2: u64,
    /// Second raw moment `Σy²`.
    pub sum_y2: u64,
    /// Mixed second raw moment `Σ(x·y)`.
    pub sum_xy: u64,
    /// Count of 4-connected boundary pixels (see type docs).
    ///
    /// Deliberately **not** named "perimeter": a pixel *count* is not a
    /// geometric length (it undercounts diagonal outline). For a true
    /// polygon perimeter, trace the blob's border with
    /// [`extract_contours`](crate::analyze::contours::extract_contours)
    /// and use [`Contour::perimeter`](crate::analyze::contours::Contour::perimeter).
    pub boundary_pixels: u64,
}

impl BlobMeasurements {
    /// Seed a new record from the first pixel of a component.
    ///
    /// `is_boundary` is the pixel's 4-connected boundary flag as computed
    /// by the engine.
    #[inline]
    pub(super) fn from_seed(at: Coordinate, is_boundary: bool) -> Self {
        let (xu, yu) = (at.x as u64, at.y as u64);
        Self {
            area: 1,
            bbox_min: at,
            bbox_max_inclusive: at,
            sum_x: xu,
            sum_y: yu,
            sum_x2: xu * xu,
            sum_y2: yu * yu,
            sum_xy: xu * yu,
            boundary_pixels: is_boundary as u64,
        }
    }

    /// Extend an existing record with another pixel of the same component.
    #[inline]
    pub(super) fn extend(&mut self, at: Coordinate, is_boundary: bool) {
        self.area += 1;
        if at.x < self.bbox_min.x {
            self.bbox_min.x = at.x;
        }
        if at.y < self.bbox_min.y {
            self.bbox_min.y = at.y;
        }
        if at.x > self.bbox_max_inclusive.x {
            self.bbox_max_inclusive.x = at.x;
        }
        if at.y > self.bbox_max_inclusive.y {
            self.bbox_max_inclusive.y = at.y;
        }
        let (xu, yu) = (at.x as u64, at.y as u64);
        let (x2, y2, xy) = (xu * xu, yu * yu, xu * yu);
        // Guard the documented `u64` overflow regime in debug builds; in
        // release the sums simply follow the documented bound.
        debug_assert!(
            self.sum_x2.checked_add(x2).is_some()
                && self.sum_y2.checked_add(y2).is_some()
                && self.sum_xy.checked_add(xy).is_some(),
            "BlobMeasurements second-moment sum overflowed u64 \
             (image dimension beyond the documented ~46 000 px bound)"
        );
        self.sum_x += xu;
        self.sum_y += yu;
        self.sum_x2 += x2;
        self.sum_y2 += y2;
        self.sum_xy += xy;
        self.boundary_pixels += is_boundary as u64;
    }

    /// Centroid (centre of mass) as a sub-pixel [`CoordinateF64`].
    ///
    /// Returns `(sum_x / area, sum_y / area)`. `area` is always `>= 1`
    /// for records produced by the engine, so the division is safe.
    pub fn centroid(&self) -> CoordinateF64 {
        let inv = 1.0 / self.area as f64;
        CoordinateF64::new(self.sum_x as f64 * inv, self.sum_y as f64 * inv)
    }

    /// Axis-aligned bounding box as a half-open [`Rectangle`].
    ///
    /// Width and height are `bbox_max_inclusive - bbox_min + 1` on each
    /// axis.
    pub fn bbox(&self) -> Rectangle {
        let w = self.bbox_max_inclusive.x - self.bbox_min.x + 1;
        let h = self.bbox_max_inclusive.y - self.bbox_min.y + 1;
        Rectangle::new(self.bbox_min, Size::new(w, h))
    }

    /// Central second moments `(μ20, μ02, μ11)`, normalised by area
    /// (i.e. the variance/covariance of the pixel coordinates about the
    /// centroid), computed in `f64`.
    ///
    /// ```text
    /// μ20 = sum_x2/area − x̄²
    /// μ02 = sum_y2/area − ȳ²
    /// μ11 = sum_xy/area − x̄·ȳ
    /// ```
    ///
    /// The raw sums are promoted to `f64` *before* the subtraction; the
    /// squared-sum terms far exceed integer range, so central moments must
    /// never be formed in integer space.
    pub fn central_moments(&self) -> (f64, f64, f64) {
        let inv = 1.0 / self.area as f64;
        let xbar = self.sum_x as f64 * inv;
        let ybar = self.sum_y as f64 * inv;
        let mu20 = self.sum_x2 as f64 * inv - xbar * xbar;
        let mu02 = self.sum_y2 as f64 * inv - ybar * ybar;
        let mu11 = self.sum_xy as f64 * inv - xbar * ybar;
        (mu20, mu02, mu11)
    }

    /// Diameter of the circle with the same area: `2·√(area/π)`.
    pub fn equivalent_diameter(&self) -> f64 {
        2.0 * (self.area as f64 / std::f64::consts::PI).sqrt()
    }

    /// Orientation of the major axis, as an [`AxialOrientation`].
    ///
    /// `orientation = ½·atan2(2·μ11, μ20 − μ02)`, giving a value in
    /// `(−π/2, π/2]` measured from the +x axis **in image (y-down)
    /// coordinates** — positive angles rotate toward +y (downward on
    /// screen). Note the y-down flip versus math-convention plots, or the
    /// sign reads backwards. A rotationally-symmetric or single-pixel blob
    /// returns `0`.
    ///
    /// The return type is *axial*, not directed: an ellipse's major axis has
    /// no head and no tail, so `+80°` and `−100°` are the same axis. That is
    /// why comparing two blobs' orientations goes through
    /// [`AxialOrientation::signed_difference`], which wraps at π — a raw
    /// subtraction would read those two as `160°` apart instead of `20°`.
    /// Call [`AxialOrientation::radians`] for the bare angle.
    pub fn orientation(&self) -> AxialOrientation {
        let (mu20, mu02, mu11) = self.central_moments();
        axis_orientation(mu20, mu02, mu11)
    }

    /// Eccentricity of the equivalent ellipse, in `[0, 1]`.
    ///
    /// `0` is a perfect circle and the value rises toward `1` as the blob
    /// becomes more line-like. `1` is attained **exactly**, not merely
    /// approached: a straight axis-aligned run of pixels has `λ₂ = 0` and
    /// therefore `ecc = 1.0`. Derived from the eigenvalues `λ₁ ≥ λ₂ ≥ 0` of
    /// the central second-moment matrix as `√(1 − λ₂/λ₁)`. A degenerate blob
    /// (single pixel, `λ₁ = 0`) returns `0` rather than `NaN`.
    pub fn eccentricity(&self) -> f64 {
        let (mu20, mu02, mu11) = self.central_moments();
        axis_eccentricity(mu20, mu02, mu11)
    }

    /// Circularity (roundness): `4π·area / boundary_pixels²` — the cheap,
    /// single-pass relative score.
    ///
    /// `1` is the continuous-geometry ideal for a disc. Because
    /// `boundary_pixels` is a pixel *count*, which undercounts diagonal
    /// outline (a √2 Euclidean rim step counts as one pixel), a rasterised
    /// disc actually reads slightly *above* 1 (≈1.25), while an
    /// axis-aligned square reads ≈π/4. Treat this circularity as a relative
    /// shape score compared within a tolerance band, not an absolute; for
    /// the geometric score computed from a traced polygon, use
    /// [`Contour::circularity`](crate::analyze::contours::Contour::circularity).
    /// Returns `0` for a zero boundary count (which the engine never
    /// produces, since every foreground pixel of an in-view blob has at
    /// least one boundary pixel).
    pub fn circularity(&self) -> f64 {
        if self.boundary_pixels == 0 {
            return 0.0;
        }
        let p = self.boundary_pixels as f64;
        4.0 * std::f64::consts::PI * self.area as f64 / (p * p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Build a `BlobMeasurements` from an explicit pixel set, marking a
    /// pixel as boundary iff any 4-neighbour is absent from the set.
    /// Mirrors the engine's boundary rule for unit-testing the math in
    /// isolation.
    fn from_pixels(pixels: &[(usize, usize)]) -> BlobMeasurements {
        use std::collections::HashSet;
        let set: HashSet<(usize, usize)> = pixels.iter().copied().collect();
        let is_boundary = |x: usize, y: usize| {
            let n4 = [
                x.checked_sub(1).map(|nx| (nx, y)),
                Some((x + 1, y)),
                y.checked_sub(1).map(|ny| (x, ny)),
                Some((x, y + 1)),
            ];
            n4.iter().any(|n| match n {
                Some(p) => !set.contains(p),
                None => true,
            })
        };
        let mut it = pixels.iter();
        let &(x0, y0) = it.next().expect("at least one pixel");
        let mut m = BlobMeasurements::from_seed(Coordinate::new(x0, y0), is_boundary(x0, y0));
        for &(x, y) in it {
            m.extend(Coordinate::new(x, y), is_boundary(x, y));
        }
        m
    }

    /// Solid N×N square anchored at the origin.
    fn square(n: usize) -> Vec<(usize, usize)> {
        let mut v = Vec::with_capacity(n * n);
        for y in 0..n {
            for x in 0..n {
                v.push((x, y));
            }
        }
        v
    }

    #[test]
    fn from_seed_single_pixel_is_finite() {
        let m = BlobMeasurements::from_seed(Coordinate::new(3, 5), true);
        assert_eq!(m.area, 1);
        assert_eq!(m.boundary_pixels, 1);
        assert_eq!(m.sum_x2, 9);
        assert_eq!(m.sum_y2, 25);
        assert_eq!(m.sum_xy, 15);
        assert_eq!(m.centroid(), CoordinateF64::new(3.0, 5.0));
        // No NaN / div-by-zero for a degenerate blob.
        assert_eq!(m.eccentricity(), 0.0);
        assert_eq!(m.orientation().radians(), 0.0);
        assert!(m.circularity().is_finite());
        let (mu20, mu02, mu11) = m.central_moments();
        assert_eq!((mu20, mu02, mu11), (0.0, 0.0, 0.0));
    }

    #[test]
    fn boundary_pixels_of_square_is_4n_minus_4() {
        for n in 1..=10usize {
            let m = from_pixels(&square(n));
            let expected = if n == 1 { 1 } else { (4 * n - 4) as u64 };
            assert_eq!(m.boundary_pixels, expected, "N={n}");
        }
    }

    #[test]
    fn equivalent_diameter_of_known_area() {
        let m = from_pixels(&square(10)); // area 100
        assert_eq!(m.area, 100);
        let expected = 2.0 * (100.0 / PI).sqrt();
        assert!((m.equivalent_diameter() - expected).abs() < 1e-12);
    }

    #[test]
    fn square_circularity_below_one() {
        // 4-connected boundary count biases circularity below 1; for a large
        // square it approaches 4π·N²/(4N)² = π/4 ≈ 0.785.
        let m = from_pixels(&square(40));
        let c = m.circularity();
        assert!(c < 1.0, "circularity {c} should be < 1");
        assert!((c - PI / 4.0).abs() < 0.05, "circularity {c} ≈ π/4");
    }

    #[test]
    fn horizontal_bar_orientation_zero() {
        // 11-wide, 1-tall bar → major axis along x → orientation ≈ 0.
        let pixels: Vec<(usize, usize)> = (0..11).map(|x| (x, 0)).collect();
        let m = from_pixels(&pixels);
        assert!(
            m.orientation().radians().abs() < 1e-9,
            "got {:?}",
            m.orientation()
        );
    }

    #[test]
    fn vertical_bar_orientation_half_pi() {
        // 1-wide, 11-tall bar → major axis along y → orientation ≈ π/2.
        let pixels: Vec<(usize, usize)> = (0..11).map(|y| (0, y)).collect();
        let m = from_pixels(&pixels);
        assert!(
            (m.orientation().radians().abs() - PI / 2.0).abs() < 1e-9,
            "got {:?}",
            m.orientation()
        );
    }

    #[test]
    fn diagonal_bar_orientation_sign() {
        // Bar along x==y (top-left → bottom-right, i.e. downward-right in
        // y-down image coords) → orientation +π/4. Pins the sign convention.
        let pixels: Vec<(usize, usize)> = (0..11).map(|i| (i, i)).collect();
        let m = from_pixels(&pixels);
        let expected = AxialOrientation::from_radians(PI / 4.0).unwrap();
        assert!(
            m.orientation().signed_difference(expected).abs() < 1e-9,
            "got {:?}",
            m.orientation()
        );

        // Anti-diagonal along x==-y (top-right → bottom-left) → −π/4.
        let pixels: Vec<(usize, usize)> = (0..11).map(|i| (10 - i, i)).collect();
        let m = from_pixels(&pixels);
        let expected = AxialOrientation::from_radians(-PI / 4.0).unwrap();
        assert!(
            m.orientation().signed_difference(expected).abs() < 1e-9,
            "got {:?}",
            m.orientation()
        );
    }

    #[test]
    fn line_eccentricity_is_exactly_one() {
        // A straight axis-aligned run has μ02 = μ11 = 0, so λ₂ = 0 and the
        // eccentricity is exactly 1 — the documented upper bound is closed,
        // not open.
        let pixels: Vec<(usize, usize)> = (0..50).map(|x| (x, 0)).collect();
        let m = from_pixels(&pixels);
        assert_eq!(m.eccentricity(), 1.0);

        // Vertical too, so the result does not depend on which moment vanishes.
        let pixels: Vec<(usize, usize)> = (0..50).map(|y| (0, y)).collect();
        let m = from_pixels(&pixels);
        assert_eq!(m.eccentricity(), 1.0);
    }

    #[test]
    fn disc_eccentricity_near_zero_and_circularity_near_one() {
        // Rasterised disc of radius 20 centred at (25, 25).
        let (cx, cy, r) = (25i64, 25i64, 20i64);
        let mut pixels = Vec::new();
        for y in 0..=50i64 {
            for x in 0..=50i64 {
                let (dx, dy) = (x - cx, y - cy);
                if dx * dx + dy * dy <= r * r {
                    pixels.push((x as usize, y as usize));
                }
            }
        }
        let m = from_pixels(&pixels);
        assert!(m.eccentricity() < 0.05, "ecc {}", m.eccentricity());
        // The 4-connected boundary-*pixel* count undercounts a disc's
        // circumference (a diagonal rim step counts one pixel where the
        // Euclidean edge is √2), so circularity of a rasterised disc lands
        // *above* 1, converging to ≈1.25 as radius grows — the opposite of
        // the crack/edge-count bias. Pin it to that band.
        let c = m.circularity();
        assert!(c > 1.1 && c < 1.4, "circularity {c}");
    }

    #[test]
    fn moments_match_brute_force() {
        // Deterministic pseudo-random mask; compare against a naïve sum.
        let mut pixels = Vec::new();
        let mut state: u64 = 0x1234_5678;
        for y in 0..30usize {
            for x in 0..30usize {
                // xorshift-ish deterministic bit
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                if state & 1 == 1 {
                    pixels.push((x, y));
                }
            }
        }
        let m = from_pixels(&pixels);
        let (mut sx, mut sy, mut sx2, mut sy2, mut sxy) = (0u64, 0u64, 0u64, 0u64, 0u64);
        for &(x, y) in &pixels {
            let (x, y) = (x as u64, y as u64);
            sx += x;
            sy += y;
            sx2 += x * x;
            sy2 += y * y;
            sxy += x * y;
        }
        assert_eq!(m.area, pixels.len() as u64);
        assert_eq!((m.sum_x, m.sum_y), (sx, sy));
        assert_eq!((m.sum_x2, m.sum_y2, m.sum_xy), (sx2, sy2, sxy));
    }
}

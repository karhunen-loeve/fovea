//! The contour data model: [`Contour`], [`ComponentContour`],
//! [`ContourHierarchy`].

use crate::{Coordinate, CoordinateF64};

use super::chain::ChainCode;
use super::polygon::{
    convex_hull, polygon_area, polygon_centroid, polygon_perimeter,
};

/// Whether a contour is a component's outer border or the border of one
/// of its holes.
///
/// Carried explicitly rather than encoded in the winding direction —
/// outer contours do wind clockwise on screen and hole contours
/// counterclockwise, but that is a documented property to *check*, not a
/// convention to *decode*.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ContourKind {
    /// The border around a component's outside.
    Outer,
    /// The border around one of a component's holes. The points are
    /// foreground pixels of the component, adjacent to the hole.
    Hole,
}

/// A closed 8-connected pixel chain produced by border tracing.
///
/// The points are the border **pixels the tracer visited, in order** —
/// integer coordinates, because that is what tracing honestly knows.
/// The chain is implicitly closed (the last point connects back to the
/// first) and consecutive points are 8-adjacent, which is what makes
/// [`Contour::chain_code`] total. There is no public constructor: a
/// `Contour` certifies these invariants, and they come from the tracer.
/// For arbitrary vertex lists (including simplified polygons from
/// [`approximate_polygon`](super::approximate_polygon)), use the free
/// polygon functions instead.
///
/// Shape descriptors are derived on demand in `f64`, nothing is cached.
/// Descriptors that would divide by zero on degenerate chains (fewer
/// than 3 points, or an out-and-back trace of a 1-px-thin shape) return
/// `Option` — absence, not `NaN`.
///
/// Thin structures trace **out and back**: a 3-px horizontal line yields
/// 4 points (3 out, 1 back), encloses no area, and has a perimeter — the
/// length of the out-and-back path — of 4.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Contour {
    points: Vec<Coordinate>,
    kind: ContourKind,
}

impl Contour {
    /// Certify a traced point chain. Tracer-internal.
    pub(super) fn new(points: Vec<Coordinate>, kind: ContourKind) -> Self {
        debug_assert!(!points.is_empty(), "a traced contour has at least one point");
        Self { points, kind }
    }

    /// The border pixels in trace order. Never empty.
    #[must_use]
    pub fn points(&self) -> &[Coordinate] {
        &self.points
    }

    /// Outer border or hole border.
    #[must_use]
    pub const fn kind(&self) -> ContourKind {
        self.kind
    }

    /// Area enclosed by the traced polygon, in square pixels.
    ///
    /// This is [`polygon_area`](super::polygon_area) of the points: the
    /// area enclosed by the path through the border pixel *centers*, so
    /// a solid 6×6-pixel square measures `25.0` where its pixel-count
    /// `area` is `36`. Degenerate chains return `0.0`.
    #[must_use]
    pub fn area(&self) -> f64 {
        polygon_area(&self.points)
    }

    /// Length of the traced polygon including the closing edge, in
    /// pixels — diagonal steps count √2.
    ///
    /// Exact for the polygon; as an estimate of a smooth outline it
    /// overestimates by roughly 5–8% (staircase effect). See
    /// [`Contour::circularity`].
    #[must_use]
    pub fn perimeter(&self) -> f64 {
        polygon_perimeter(&self.points)
    }

    /// Centroid of the enclosed region, or `None` when the chain
    /// encloses no area.
    #[must_use]
    pub fn centroid(&self) -> Option<CoordinateF64> {
        polygon_centroid(&self.points)
    }

    /// Circularity `4π·area / perimeter²` of the traced polygon, or
    /// `None` for a single-point contour (zero perimeter).
    ///
    /// `1` is the ideal disc; an axis-aligned square reads `π/4 ≈ 0.785`.
    /// Unlike the boundary-pixel-count circularity of the component
    /// measurements (which reads ≈ 1.25 for a rasterised disc), this one
    /// is bounded by ~1 — but the staircase in the traced chain still
    /// depresses it: a rasterised disc of radius 20 measures ≈ 0.87 raw,
    /// and ≈ 0.94 after
    /// [`approximate_polygon`](super::approximate_polygon) with ε = 0.8
    /// (computing `4π·A/P²` from the simplified vertices). If your
    /// underlying outlines are smooth, simplify before scoring.
    #[must_use]
    pub fn circularity(&self) -> Option<f64> {
        let p = self.perimeter();
        if p == 0.0 {
            return None;
        }
        Some(4.0 * std::f64::consts::PI * self.area() / (p * p))
    }

    /// Convex hull of the border pixels — see
    /// [`convex_hull`](super::convex_hull).
    #[must_use]
    pub fn convex_hull(&self) -> Vec<Coordinate> {
        convex_hull(&self.points)
    }

    /// Solidity: enclosed area / convex-hull area, in `(0, 1]` for
    /// non-degenerate contours — `1` means the shape *is* its hull
    /// (convex), lower values mean concavities.
    ///
    /// `None` when the hull encloses no area (degenerate chain).
    #[must_use]
    pub fn solidity(&self) -> Option<f64> {
        let hull_area = polygon_area(&self.convex_hull());
        if hull_area == 0.0 {
            return None;
        }
        Some(self.area() / hull_area)
    }

    /// Encode as a Freeman chain code — see [`ChainCode`].
    #[must_use]
    pub fn chain_code(&self) -> ChainCode {
        ChainCode::from_contour(self)
    }
}

/// All contours of one connected component: its outer border, the
/// borders of its holes, and its place in the nesting hierarchy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComponentContour {
    pub(super) outer: Contour,
    pub(super) holes: Vec<Contour>,
    pub(super) enclosing: Option<usize>,
}

impl ComponentContour {
    /// The component's outer border.
    #[must_use]
    pub fn outer(&self) -> &Contour {
        &self.outer
    }

    /// One border per hole, in raster order of the holes' first pixels.
    /// Empty for a component without holes.
    #[must_use]
    pub fn holes(&self) -> &[Contour] {
        &self.holes
    }

    /// Index (into [`ContourHierarchy::components`]) of the component
    /// inside whose hole this component sits, or `None` for a top-level
    /// component.
    ///
    /// Chains arbitrarily deep: a blob inside a hole inside a blob
    /// reports the inner blob's `enclosing` as the outer blob's index.
    #[must_use]
    pub const fn enclosing(&self) -> Option<usize> {
        self.enclosing
    }

    /// Euler number of the component: `1 − number of holes`.
    ///
    /// `1` for a solid blob, `0` for a ring, `−1` for a blob with two
    /// holes.
    #[must_use]
    pub fn euler_number(&self) -> i64 {
        1 - self.holes.len() as i64
    }
}

/// Every component's contours, indexed like the labeling.
///
/// Component `i` corresponds to label `i + 1` in the
/// [`Labeling`](crate::analyze::components::Labeling) returned alongside —
/// the same convention as the stats and measurements vectors, so contours
/// join those tables with no translation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContourHierarchy {
    pub(super) components: Vec<ComponentContour>,
}

impl ContourHierarchy {
    /// All components, in label order (component `i` ↔ label `i + 1`).
    #[must_use]
    pub fn components(&self) -> &[ComponentContour] {
        &self.components
    }

    /// The component with the given **label** (1-based, as stored in the
    /// label image), or `None` if the label is `0` (background) or out of
    /// range.
    #[must_use]
    pub fn component_for_label(&self, label: u32) -> Option<&ComponentContour> {
        let index = usize::try_from(label.checked_sub(1)?).ok()?;
        self.components.get(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(x: usize, y: usize) -> Coordinate {
        Coordinate::new(x, y)
    }

    fn square_contour() -> Contour {
        // Traced border of a 4×4 solid square anchored at (0, 0).
        let points = [
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
        Contour::new(points, ContourKind::Outer)
    }

    #[test]
    fn square_descriptors() {
        let sq = square_contour();
        assert_eq!(sq.area(), 9.0);
        assert_eq!(sq.perimeter(), 12.0);
        let ctr = sq.centroid().unwrap();
        assert_eq!((ctr.x, ctr.y), (1.5, 1.5));
        // Square: 4πA/P² = 4π·9/144 = π/4.
        let circ = sq.circularity().unwrap();
        assert!((circ - std::f64::consts::FRAC_PI_4).abs() < 1e-12);
        // Convex already: solidity exactly 1.
        assert_eq!(sq.solidity(), Some(1.0));
        assert_eq!(sq.convex_hull().len(), 4);
    }

    #[test]
    fn degenerate_contour_descriptors_are_absent_not_nan() {
        let dot = Contour::new(vec![c(5, 5)], ContourKind::Outer);
        assert_eq!(dot.area(), 0.0);
        assert_eq!(dot.perimeter(), 0.0);
        assert_eq!(dot.centroid(), None);
        assert_eq!(dot.circularity(), None);
        assert_eq!(dot.solidity(), None);

        // Out-and-back line: perimeter exists, area does not.
        let line = Contour::new(vec![c(1, 2), c(2, 2), c(3, 2), c(2, 2)], ContourKind::Outer);
        assert_eq!(line.area(), 0.0);
        assert_eq!(line.perimeter(), 4.0);
        assert_eq!(line.centroid(), None);
        // Perimeter is nonzero, so circularity is defined — and 0.
        assert_eq!(line.circularity(), Some(0.0));
        assert_eq!(line.solidity(), None);
    }

    #[test]
    fn concave_shape_solidity_below_one() {
        // An L: the traced border of an L-shaped pixel set.
        let points = [
            (0, 0),
            (1, 0),
            (2, 0),
            (2, 1),
            (1, 1),
            (1, 2),
            (2, 2),
            (2, 3),
            (1, 3),
            (0, 3),
            (0, 2),
            (0, 1),
        ]
        .map(Coordinate::from)
        .to_vec();
        let l = Contour::new(points, ContourKind::Outer);
        let s = l.solidity().unwrap();
        assert!(s < 1.0, "solidity {s} of a concave shape must be < 1");
        assert!(s > 0.0);
    }

    #[test]
    fn euler_numbers() {
        let outer = square_contour();
        let solid = ComponentContour {
            outer: outer.clone(),
            holes: vec![],
            enclosing: None,
        };
        assert_eq!(solid.euler_number(), 1);
        let two_holes = ComponentContour {
            outer: outer.clone(),
            holes: vec![
                Contour::new(vec![c(1, 1)], ContourKind::Hole),
                Contour::new(vec![c(2, 2)], ContourKind::Hole),
            ],
            enclosing: None,
        };
        assert_eq!(two_holes.euler_number(), -1);
    }

    #[test]
    fn component_for_label_is_one_based() {
        let hierarchy = ContourHierarchy {
            components: vec![ComponentContour {
                outer: square_contour(),
                holes: vec![],
                enclosing: None,
            }],
        };
        assert!(hierarchy.component_for_label(0).is_none());
        assert!(hierarchy.component_for_label(1).is_some());
        assert!(hierarchy.component_for_label(2).is_none());
    }
}

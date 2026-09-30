//! The mapping traits: what a mapping of the plane can convert.

use super::point::{Length, Point, Vector};
use super::units::LengthUnit;

/// A mapping of the plane from one unit of length to another.
///
/// Every mapping converts points. Which other objects it converts depends on
/// its class, and the class is a type: the affine mappings, whose metric is
/// the same everywhere, implement [`AffineMap`], and the similarities among
/// them also implement [`ConformalMap`].
///
/// `try_map_point` returns `None` where the mapping has no image, as a
/// projective mapping has none on its vanishing line. The affine mappings
/// are total and also offer [`AffineMap::map_point`].
///
/// # Example
///
/// ```
/// use fovea::{Millimeter, Pixels, Point};
/// use fovea::geometry::{PlaneMap, UniformScale};
///
/// let scale: UniformScale<Pixels, Millimeter> = fovea::uniform_scale!(0.0125);
/// let p = scale.try_map_point(Point::new(800.0, 400.0));
/// assert_eq!(p, Some(Point::new(10.0, 5.0)));
/// ```
///
/// The units must match:
///
/// ```compile_fail
/// use fovea::{Micrometer, Millimeter, Pixels, Point};
/// use fovea::geometry::{PlaneMap, UniformScale};
///
/// let scale: UniformScale<Pixels, Millimeter> = fovea::uniform_scale!(0.0125);
/// let p: Point<Micrometer> = Point::new(1.0, 1.0);
/// let _ = scale.try_map_point(p);
/// ```
pub trait PlaneMap {
    /// The unit of the points the mapping takes.
    type Domain: LengthUnit;
    /// The unit of the points it returns.
    type Codomain: LengthUnit;

    /// The image of `p`, or `None` where the mapping has none.
    fn try_map_point(&self, p: Point<Self::Domain>) -> Option<Point<Self::Codomain>>;
}

/// A mapping whose metric is the same everywhere: `x ↦ A·x + t`.
///
/// Because `A` is constant, displacements convert on their own, without the
/// position they start from, and the mapping pulls back one inner product,
/// `⟨u, v⟩ = (A·u) · (A·v)`, to the whole domain. Lines stay lines, parallel
/// lines stay parallel, midpoints stay midpoints, and a circle becomes an
/// ellipse. Lengths without a direction convert only under a
/// [`ConformalMap`].
///
/// The mapping is total, and construction rejects a singular or overflowing
/// `A`, so the inverse always exists.
///
/// # Example
///
/// ```
/// use fovea::{Millimeter, Pixels, Point};
/// use fovea::geometry::{AffineMap, AxisScale, Vector};
///
/// // A line-scan camera: 20 µm across the line, 50 µm along the feed.
/// let scale: AxisScale<Pixels, Millimeter> = fovea::axis_scale!(0.02, 0.05);
/// assert_eq!(scale.map_point(Point::new(100.0, 10.0)), Point::new(2.0, 0.5));
///
/// // A diagonal step of one pixel is 0.0539 mm long, which no single
/// // factor applied to its pixel length would give.
/// let step = scale.length_of(Vector::new(1.0, 1.0));
/// assert!((step.get() - 0.02f64.hypot(0.05)).abs() < 1e-15);
///
/// let back = scale.inverse().map_point(Point::new(2.0, 0.5));
/// assert!((back.x - 100.0).abs() < 1e-12 && (back.y - 10.0).abs() < 1e-12);
/// ```
pub trait AffineMap: PlaneMap {
    /// The inverse mapping's type, with domain and codomain swapped.
    type Inverse: AffineMap<Domain = Self::Codomain, Codomain = Self::Domain>;

    /// The image of `p`.
    fn map_point(&self, p: Point<Self::Domain>) -> Point<Self::Codomain>;

    /// The image of the displacement `v`, which is `A·v`: the translation
    /// does not move a displacement.
    fn map_vector(&self, v: Vector<Self::Domain>) -> Vector<Self::Codomain>;

    /// The length in the codomain of the displacement `v` in the domain.
    ///
    /// This is the norm of the pulled-back inner product, `|A·v|`, and it
    /// depends on the direction of `v` unless the mapping is conformal.
    fn length_of(&self, v: Vector<Self::Domain>) -> Length<Self::Codomain> {
        self.map_vector(v).length()
    }

    /// The inverse mapping, computed in closed form.
    fn inverse(&self) -> Self::Inverse;

    /// Whether the mapping is a reflection composed with a rotation and a
    /// stretch (`det A < 0`).
    ///
    /// Signed angles, signed areas and the winding direction of a contour
    /// change sign under such a mapping. Converting an image frame with `y`
    /// pointing down into a machine frame with `y` pointing up is one.
    fn reverses_orientation(&self) -> bool;
}

/// An affine mapping that scales every direction by the same factor:
/// `A = a·R` with `R` a rotation or a reflection.
///
/// Such a mapping preserves angles, so lengths convert without a direction
/// (multiplied by [`factor`](ConformalMap::factor)) and a circle stays a
/// circle. A line or circle fitted to points in the domain, together with
/// its residuals, therefore converts after the fit. Under any other mapping
/// the points convert first and the fit runs in the codomain, because
/// distances measured in the domain are not proportional to distances in
/// the codomain.
///
/// # Example
///
/// ```
/// use fovea::{Length, Micrometer, Pixels};
/// use fovea::geometry::{ConformalMap, UniformScale};
///
/// let scale: UniformScale<Pixels, Micrometer> = fovea::uniform_scale!(3.45);
/// let radius: Length<Pixels> = Length::new(200.0);
/// assert_eq!(scale.map_length(radius).get(), 690.0);
/// ```
///
/// An axis scale is not conformal, so it converts no length:
///
/// ```compile_fail
/// use fovea::{Length, Millimeter, Pixels};
/// use fovea::geometry::{AxisScale, ConformalMap};
///
/// let scale: AxisScale<Pixels, Millimeter> = fovea::axis_scale!(0.02, 0.05);
/// let _ = scale.map_length(Length::new(1.0));
/// ```
pub trait ConformalMap: AffineMap {
    /// The factor `a` by which every length is multiplied.
    fn factor(&self) -> f64;

    /// The length in the codomain of a length in the domain.
    fn map_length(&self, l: Length<Self::Domain>) -> Length<Self::Codomain> {
        Length::new(l.get() * self.factor())
    }
}

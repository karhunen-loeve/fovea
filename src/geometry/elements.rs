//! Lines, segments, circles and ellipses in a unit of length.

use core::fmt;

use super::point::{Length, Point, Vector};
use super::units::LengthUnit;
use crate::AxialOrientation;
use crate::Error;
use crate::error::{ParameterError, Requirement, Value};

mod sealed {
    pub trait Sealed {}
}

/// A geometric element in a unit of length: [`Line`], [`Segment`],
/// [`Circle`] or [`Ellipse`].
///
/// The trait names the unit, so code that is generic over the element, such
/// as [`Fit`](crate::measure::Fit), can report lengths in it. Sealed: the
/// mappings of the plane and the fits know how to handle exactly these four.
pub trait Element: sealed::Sealed + Copy {
    /// The unit of length the element is measured in.
    type Unit: LengthUnit;
}

/// An infinite straight line in the plane, in the unit `U`: a point on it
/// and a direction of unit length.
///
/// The direction gives the line a sense, so "the side a point lies on" and
/// "the order of two points along the line" mean something. A fitted line
/// points from its first input point towards its last.
///
/// Not to be confused with [`draw::Line`](crate::draw::Line), which is an
/// instruction to paint pixels between two integer positions.
///
/// # Example
///
/// ```
/// use fovea::{Line, Millimeter, Point};
/// use fovea::geometry::Vector;
///
/// let line: Line<Millimeter> = Line::try_through(Point::new(0.0, 1.0), Point::new(4.0, 4.0))?;
/// assert_eq!(line.point(), Point::new(0.0, 1.0));
/// assert_eq!(line.direction(), Vector::new(0.8, 0.6));
///
/// // Two equal points give no direction.
/// assert!(Line::<Millimeter>::try_through(Point::new(2.0, 2.0), Point::new(2.0, 2.0)).is_err());
/// # Ok::<(), fovea::Error>(())
/// ```
pub struct Line<U> {
    point: Point<U>,
    direction: Vector<U>,
}

/// A straight segment from `start` to `end`, in the unit `U`.
///
/// Any two points form a segment, two equal ones included. The extent of a
/// fitted line is one: the part of the line its points cover.
///
/// # Example
///
/// ```
/// use fovea::{Pixels, Point, Segment};
///
/// let s: Segment<Pixels> = Segment::new(Point::new(1.0, 1.0), Point::new(4.0, 5.0));
/// assert_eq!(s.length().get(), 5.0);
/// ```
pub struct Segment<U> {
    /// Where the segment starts.
    pub start: Point<U>,
    /// Where it ends.
    pub end: Point<U>,
}

/// A circle in the plane, in the unit `U`: a centre and a radius that is
/// finite and positive.
///
/// Not to be confused with [`draw::Circle`](crate::draw::Circle), which is an
/// instruction to paint the pixels of a circle with an integer radius.
///
/// # Example
///
/// ```
/// use fovea::{Circle, Length, Pixels, Point};
///
/// let c: Circle<Pixels> = Circle::try_new(Point::new(10.0, 20.0), Length::new(3.5))?;
/// assert_eq!(c.radius().get(), 3.5);
/// assert!(Circle::<Pixels>::try_new(Point::new(0.0, 0.0), Length::new(0.0)).is_err());
/// # Ok::<(), fovea::Error>(())
/// ```
pub struct Circle<U> {
    center: Point<U>,
    radius: Length<U>,
}

/// An ellipse in the plane, in the unit `U`: a centre, a semi-major and a
/// semi-minor axis, and the orientation of the major axis.
///
/// The semi-axes are finite with `semi_major >= semi_minor > 0`; equal
/// semi-axes make a circle, whose orientation is arbitrary. The orientation
/// is an [`AxialOrientation`], because an axis has no head and no tail. It
/// is measured from `+x` towards `+y` of the frame the unit names, which
/// for [`Pixels`](crate::Pixels) is the image's frame with `y` pointing
/// down.
///
/// # Example
///
/// ```
/// use fovea::{AxialOrientation, Ellipse, Length, Pixels, Point};
///
/// let e: Ellipse<Pixels> = Ellipse::try_new(
///     Point::new(50.0, 30.0),
///     Length::new(20.0),
///     Length::new(8.0),
///     AxialOrientation::from_radians(0.6)?,
/// )?;
/// assert_eq!(e.semi_minor().get(), 8.0);
///
/// // The semi-axes in the wrong order are an error, not a swap, because a
/// // swap would turn the ellipse by a quarter turn.
/// let swapped = Ellipse::<Pixels>::try_new(
///     Point::new(50.0, 30.0),
///     Length::new(8.0),
///     Length::new(20.0),
///     AxialOrientation::from_radians(0.6)?,
/// );
/// assert!(swapped.is_err());
/// # Ok::<(), fovea::Error>(())
/// ```
pub struct Ellipse<U> {
    center: Point<U>,
    semi_major: Length<U>,
    semi_minor: Length<U>,
    orientation: AxialOrientation,
}

impl<U: LengthUnit> Line<U> {
    /// The line through `point` along `direction`, which is scaled to unit
    /// length.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidParameter`] if `direction` has a component that is
    /// not finite, or a length of zero or one too small to scale.
    pub fn try_new(point: Point<U>, direction: Vector<U>) -> Result<Self, Error> {
        match unit_direction(direction) {
            Some(direction) => Ok(Self { point, direction }),
            None => Err(ParameterError::new(
                "line direction",
                Requirement::FiniteNonZero,
                Value::F64Pair(direction.x, direction.y),
            )
            .into()),
        }
    }

    /// The line through `from` and `to`, directed from `from` towards `to`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidParameter`] if the two points are equal or their
    /// difference is not finite, as in [`Self::try_new`].
    pub fn try_through(from: Point<U>, to: Point<U>) -> Result<Self, Error> {
        Self::try_new(from, to - from)
    }

    /// A point on the line. For a fitted line, the centroid of the points
    /// that shaped it.
    #[must_use]
    pub fn point(&self) -> Point<U> {
        self.point
    }

    /// The direction of the line, of unit length.
    #[must_use]
    pub fn direction(&self) -> Vector<U> {
        self.direction
    }

    /// Builds a line from parts the caller has already checked: `direction`
    /// has unit length.
    pub(crate) fn from_parts(point: Point<U>, direction: Vector<U>) -> Self {
        debug_assert!((direction.length().get() - 1.0).abs() < 1e-12);
        Self { point, direction }
    }

    /// The position of the foot of `p` along the line, measured from
    /// [`point`](Self::point) in the line's direction.
    pub(crate) fn along(&self, p: Point<U>) -> f64 {
        let d = p - self.point;
        d.x * self.direction.x + d.y * self.direction.y
    }

    /// The distance of `p` from the line, positive on the side the
    /// direction turned by `+90°` points to.
    pub(crate) fn signed_distance(&self, p: Point<U>) -> f64 {
        let d = p - self.point;
        d.y * self.direction.x - d.x * self.direction.y
    }
}

impl<U: LengthUnit> Segment<U> {
    /// The segment from `start` to `end`.
    #[must_use]
    pub const fn new(start: Point<U>, end: Point<U>) -> Self {
        Self { start, end }
    }

    /// The length of the segment.
    #[must_use]
    pub fn length(&self) -> Length<U> {
        self.start.distance(self.end)
    }
}

impl<U: LengthUnit> Circle<U> {
    /// The circle about `center` with `radius`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidParameter`] if `radius` is not finite and positive.
    pub fn try_new(center: Point<U>, radius: Length<U>) -> Result<Self, Error> {
        let r = radius.get();
        if r.is_finite() && r > 0.0 {
            Ok(Self { center, radius })
        } else {
            Err(ParameterError::new("radius", Requirement::FinitePositive, Value::F64(r)).into())
        }
    }

    /// The centre.
    #[must_use]
    pub fn center(&self) -> Point<U> {
        self.center
    }

    /// The radius.
    #[must_use]
    pub fn radius(&self) -> Length<U> {
        self.radius
    }

    /// Builds a circle from parts the caller has already checked.
    pub(crate) fn from_parts(center: Point<U>, radius: Length<U>) -> Self {
        debug_assert!(radius.get() > 0.0);
        Self { center, radius }
    }

    /// The distance of `p` from the circle, positive outside.
    pub(crate) fn signed_distance(&self, p: Point<U>) -> f64 {
        self.center.distance(p).get() - self.radius.get()
    }
}

impl<U: LengthUnit> Ellipse<U> {
    /// The ellipse about `center` with the given semi-axes, its major axis
    /// along `orientation`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidParameter`] if `semi_minor` is not finite and
    /// positive, if `semi_major` is not finite, or if `semi_major` is
    /// smaller than `semi_minor`.
    pub fn try_new(
        center: Point<U>,
        semi_major: Length<U>,
        semi_minor: Length<U>,
        orientation: AxialOrientation,
    ) -> Result<Self, Error> {
        let (a, b) = (semi_major.get(), semi_minor.get());
        if !(b.is_finite() && b > 0.0) {
            return Err(ParameterError::new(
                "semi-minor axis",
                Requirement::FinitePositive,
                Value::F64(b),
            )
            .into());
        }
        if !a.is_finite() {
            return Err(ParameterError::new(
                "semi-major axis",
                Requirement::FinitePositive,
                Value::F64(a),
            )
            .into());
        }
        if a < b {
            return Err(ParameterError::new(
                "semi-axes (minor, major)",
                Requirement::Ordered,
                Value::F64Pair(b, a),
            )
            .into());
        }
        Ok(Self {
            center,
            semi_major,
            semi_minor,
            orientation,
        })
    }

    /// The centre.
    #[must_use]
    pub fn center(&self) -> Point<U> {
        self.center
    }

    /// Half the length of the major axis.
    #[must_use]
    pub fn semi_major(&self) -> Length<U> {
        self.semi_major
    }

    /// Half the length of the minor axis.
    #[must_use]
    pub fn semi_minor(&self) -> Length<U> {
        self.semi_minor
    }

    /// The orientation of the major axis.
    #[must_use]
    pub fn orientation(&self) -> AxialOrientation {
        self.orientation
    }

    /// Builds an ellipse from parts the caller has already checked.
    pub(crate) fn from_parts(
        center: Point<U>,
        semi_major: f64,
        semi_minor: f64,
        orientation: AxialOrientation,
    ) -> Self {
        debug_assert!(semi_major >= semi_minor && semi_minor > 0.0);
        Self {
            center,
            semi_major: Length::new(semi_major),
            semi_minor: Length::new(semi_minor),
            orientation,
        }
    }

    /// The exact distance of `p` from the ellipse, positive outside.
    ///
    /// The foot point is found by Eberly's bisection: in the ellipse's own
    /// frame, folded into the first quadrant, the foot point's normal passes
    /// through `p`, and the parameter of that normal is the unique root of a
    /// strictly decreasing function on a known interval. Bisection to the
    /// last representable step finds it without the failure modes of
    /// Newton's method or of solving the quartic.
    pub(crate) fn signed_distance(&self, p: Point<U>) -> f64 {
        self.foot_and_signed_distance(p).1
    }

    /// The nearest point to `p` on the ellipse, and the signed distance of
    /// `p` from it.
    ///
    /// Where two points are equally near, the one on the side the minor
    /// axis's positive direction points to: a coordinate of exactly zero in
    /// the ellipse's frame counts as positive. At the exact centre that is
    /// the end of the minor axis, `(0, semi_minor)` in the ellipse's frame,
    /// also when the semi-axes are equal.
    pub(crate) fn foot_and_signed_distance(&self, p: Point<U>) -> (Point<U>, f64) {
        let (sin, cos) = self.orientation.radians().sin_cos();
        let d = p - self.center;
        let q0 = cos * d.x + sin * d.y;
        let q1 = cos * d.y - sin * d.x;
        let (e0, e1) = (self.semi_major.get(), self.semi_minor.get());
        let (y0, y1) = (q0.abs(), q1.abs());
        let (x0, x1) = if y0 == 0.0 && y1 == 0.0 {
            (0.0, e1)
        } else {
            foot_on_standard_ellipse(e0, e1, y0, y1)
        };
        let distance = (x0 - y0).hypot(x1 - y1);
        let x0 = if q0 < 0.0 { -x0 } else { x0 };
        let x1 = if q1 < 0.0 { -x1 } else { x1 };
        let foot = self.center + Vector::new(cos * x0 - sin * x1, sin * x0 + cos * x1);
        let (u, v) = (q0 / e0, q1 / e1);
        if u * u + v * v > 1.0 {
            (foot, distance)
        } else {
            (foot, -distance)
        }
    }
}

/// `v` scaled to unit length, or `None` if that is not possible.
fn unit_direction<U: LengthUnit>(v: Vector<U>) -> Option<Vector<U>> {
    let n = v.length().get();
    if !(n.is_finite() && n.is_normal()) {
        return None;
    }
    let u = Vector::new(v.x / n, v.y / n);
    (u.x.is_finite() && u.y.is_finite()).then_some(u)
}

/// The number of bisection steps that exhausts every `f64` interval: the
/// mantissa's digits plus the exponent range down to the smallest normal.
const BISECTION_STEPS: usize = (f64::MANTISSA_DIGITS as i32 - f64::MIN_EXP) as usize;

/// The point of the ellipse `(x0/e0)² + (x1/e1)² = 1` nearest to
/// `(y0, y1)`, for `e0 >= e1 > 0`, `y0 >= 0` and `y1 >= 0`; it lies in the
/// same quadrant.
///
/// Eberly, "Distance from a Point to an Ellipse, an Ellipsoid, or a
/// Hyperellipsoid", Geometric Tools, 2013 (revised 2020), Listing 2.
fn foot_on_standard_ellipse(e0: f64, e1: f64, y0: f64, y1: f64) -> (f64, f64) {
    if y1 > 0.0 {
        if y0 > 0.0 {
            let z0 = y0 / e0;
            let z1 = y1 / e1;
            let g = z0 * z0 + z1 * z1 - 1.0;
            if g == 0.0 {
                return (y0, y1);
            }
            let r0 = (e0 / e1) * (e0 / e1);
            let s = bisect_root(r0, z0, z1, g);
            (r0 * y0 / (s + r0), y1 / (s + 1.0))
        } else {
            (0.0, e1)
        }
    } else {
        let numer0 = e0 * y0;
        let denom0 = e0 * e0 - e1 * e1;
        if numer0 < denom0 {
            let xde0 = numer0 / denom0;
            (e0 * xde0, e1 * (1.0 - xde0 * xde0).sqrt())
        } else {
            (e0, 0.0)
        }
    }
}

/// The root of `(r0·z0 / (s + r0))² + (z1 / (s + 1))² − 1` on
/// `[z1 − 1, |(r0·z0, z1)| − 1]`, where `g` is its value at `s = 0`.
fn bisect_root(r0: f64, z0: f64, z1: f64, g: f64) -> f64 {
    let n0 = r0 * z0;
    let mut s0 = z1 - 1.0;
    let mut s1 = if g < 0.0 { 0.0 } else { n0.hypot(z1) - 1.0 };
    let mut s = 0.0;
    for _ in 0..BISECTION_STEPS {
        s = 0.5 * (s0 + s1);
        if s == s0 || s == s1 {
            break;
        }
        let ratio0 = n0 / (s + r0);
        let ratio1 = z1 / (s + 1.0);
        let g = ratio0 * ratio0 + ratio1 * ratio1 - 1.0;
        if g > 0.0 {
            s0 = s;
        } else if g < 0.0 {
            s1 = s;
        } else {
            break;
        }
    }
    s
}

// ── Element and the common traits, without bounds on the unit ───────────────

macro_rules! element {
    ($($ty:ident),*) => {$(
        impl<U> sealed::Sealed for $ty<U> {}
        impl<U: LengthUnit> Element for $ty<U> {
            type Unit = U;
        }
        impl<U> Clone for $ty<U> {
            #[inline]
            fn clone(&self) -> Self {
                *self
            }
        }
        impl<U> Copy for $ty<U> {}
    )*};
}

element!(Line, Segment, Circle, Ellipse);

impl<U> PartialEq for Line<U> {
    fn eq(&self, other: &Self) -> bool {
        self.point == other.point && self.direction == other.direction
    }
}
impl<U> PartialEq for Segment<U> {
    fn eq(&self, other: &Self) -> bool {
        self.start == other.start && self.end == other.end
    }
}
impl<U> PartialEq for Circle<U> {
    fn eq(&self, other: &Self) -> bool {
        self.center == other.center && self.radius == other.radius
    }
}
impl<U> PartialEq for Ellipse<U> {
    fn eq(&self, other: &Self) -> bool {
        self.center == other.center
            && self.semi_major == other.semi_major
            && self.semi_minor == other.semi_minor
            && self.orientation == other.orientation
    }
}

impl<U> fmt::Debug for Line<U> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Line")
            .field("point", &self.point)
            .field("direction", &self.direction)
            .finish()
    }
}
impl<U> fmt::Debug for Segment<U> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Segment")
            .field("start", &self.start)
            .field("end", &self.end)
            .finish()
    }
}
impl<U> fmt::Debug for Circle<U> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Circle")
            .field("center", &self.center)
            .field("radius", &self.radius)
            .finish()
    }
}
impl<U> fmt::Debug for Ellipse<U> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Ellipse")
            .field("center", &self.center)
            .field("semi_major", &self.semi_major)
            .field("semi_minor", &self.semi_minor)
            .field("orientation", &self.orientation)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{Millimeter, Pixels};
    use core::f64::consts::{FRAC_PI_2, TAU};

    fn px(x: f64, y: f64) -> Point<Pixels> {
        Point::new(x, y)
    }

    fn axis(radians: f64) -> AxialOrientation {
        AxialOrientation::from_radians(radians).unwrap()
    }

    fn ellipse(cx: f64, cy: f64, a: f64, b: f64, phi: f64) -> Ellipse<Pixels> {
        Ellipse::try_new(px(cx, cy), Length::new(a), Length::new(b), axis(phi)).unwrap()
    }

    fn on_ellipse(e: &Ellipse<Pixels>, t: f64) -> Point<Pixels> {
        let (sin, cos) = e.orientation().radians().sin_cos();
        let (u, v) = (
            e.semi_major().get() * t.cos(),
            e.semi_minor().get() * t.sin(),
        );
        e.center() + Vector::new(u * cos - v * sin, u * sin + v * cos)
    }

    /// The distance to the ellipse by dense sampling, refined by golden
    /// section search around the best sample.
    fn brute_distance(e: &Ellipse<Pixels>, p: Point<Pixels>) -> f64 {
        let f = |t: f64| on_ellipse(e, t).distance(p).get();
        let n = 4096;
        let best = (0..n)
            .map(|k| TAU * k as f64 / n as f64)
            .min_by(|a, b| f(*a).total_cmp(&f(*b)))
            .unwrap();
        let (mut lo, mut hi) = (best - TAU / n as f64, best + TAU / n as f64);
        let g = (5f64.sqrt() - 1.0) / 2.0;
        for _ in 0..200 {
            let m1 = hi - g * (hi - lo);
            let m2 = lo + g * (hi - lo);
            if f(m1) < f(m2) {
                hi = m2;
            } else {
                lo = m1;
            }
        }
        f(0.5 * (lo + hi))
    }

    #[test]
    fn a_line_has_a_unit_direction_and_rejects_none() {
        let l: Line<Pixels> = Line::try_new(px(1.0, 2.0), Vector::new(0.0, -5.0)).unwrap();
        assert_eq!(l.direction(), Vector::new(0.0, -1.0));
        for bad in [
            Vector::new(0.0, 0.0),
            Vector::new(f64::NAN, 1.0),
            Vector::new(f64::INFINITY, 0.0),
            Vector::new(1e-320, 0.0),
        ] {
            let Err(Error::InvalidParameter(e)) = Line::try_new(px(0.0, 0.0), bad) else {
                panic!("{bad:?} accepted");
            };
            assert_eq!(e.requirement(), Requirement::FiniteNonZero);
            assert_eq!(e.value(), Value::F64Pair(bad.x, bad.y));
        }
        // A huge direction scales without overflowing.
        let l: Line<Pixels> = Line::try_new(px(0.0, 0.0), Vector::new(1e300, 1e300)).unwrap();
        assert!((l.direction().x - core::f64::consts::FRAC_1_SQRT_2).abs() < 1e-15);
    }

    #[test]
    fn line_distances_and_positions_along() {
        let l: Line<Pixels> = Line::try_through(px(0.0, 0.0), px(10.0, 0.0)).unwrap();
        assert_eq!(l.signed_distance(px(3.0, 2.0)), 2.0);
        assert_eq!(l.signed_distance(px(3.0, -2.0)), -2.0);
        assert_eq!(l.along(px(3.0, 2.0)), 3.0);
        assert_eq!(l.along(px(-4.0, 7.0)), -4.0);
    }

    #[test]
    fn segments_measure_their_length() {
        let s: Segment<Millimeter> = Segment::new(Point::new(0.0, 0.0), Point::new(0.0, 0.0));
        assert_eq!(s.length().get(), 0.0);
        let s: Segment<Millimeter> = Segment::new(Point::new(-1.0, 2.0), Point::new(5.0, 10.0));
        assert_eq!(s.length().get(), 10.0);
    }

    #[test]
    fn circles_validate_the_radius() {
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let Err(Error::InvalidParameter(e)) =
                Circle::<Pixels>::try_new(px(0.0, 0.0), Length::new(bad))
            else {
                panic!("radius {bad} accepted");
            };
            assert_eq!(e.requirement(), Requirement::FinitePositive);
        }
        let c: Circle<Pixels> = Circle::try_new(px(1.0, 1.0), Length::new(5.0)).unwrap();
        assert_eq!(c.signed_distance(px(4.0, 5.0)), 0.0);
        assert_eq!(c.signed_distance(px(1.0, 9.0)), 3.0);
        assert_eq!(c.signed_distance(px(1.0, 1.0)), -5.0);
    }

    #[test]
    fn ellipses_validate_the_semi_axes() {
        let at = |a: f64, b: f64| {
            Ellipse::<Pixels>::try_new(px(0.0, 0.0), Length::new(a), Length::new(b), axis(0.0))
        };
        assert!(at(3.0, 3.0).is_ok());
        let Err(Error::InvalidParameter(e)) = at(2.0, 3.0) else {
            panic!("wrong order accepted");
        };
        assert_eq!(e.requirement(), Requirement::Ordered);
        assert_eq!(e.value(), Value::F64Pair(3.0, 2.0));
        for (a, b) in [
            (3.0, 0.0),
            (3.0, -1.0),
            (3.0, f64::NAN),
            (f64::INFINITY, 1.0),
        ] {
            let Err(Error::InvalidParameter(e)) = at(a, b) else {
                panic!("({a}, {b}) accepted");
            };
            assert_eq!(e.requirement(), Requirement::FinitePositive);
        }
        let Err(Error::InvalidParameter(e)) = at(f64::NAN, 1.0) else {
            panic!("NaN semi-major accepted");
        };
        assert_eq!(e.parameter(), "semi-major axis");
    }

    #[test]
    fn ellipse_distances_on_the_axes_and_at_the_centre() {
        let e = ellipse(0.0, 0.0, 20.0, 8.0, 0.0);
        assert_eq!(e.signed_distance(px(0.0, 0.0)), -8.0);
        assert_eq!(e.signed_distance(px(25.0, 0.0)), 5.0);
        assert_eq!(e.signed_distance(px(0.0, -10.0)), 2.0);
        assert_eq!(e.signed_distance(px(0.0, 3.0)), -5.0);
        // On the major axis inside, beyond the centre of curvature of the
        // vertex (at 20 − 8²/20 = 16.8): the nearest points leave the axis.
        let d = e.signed_distance(px(10.0, 0.0));
        assert!((d + brute_distance(&e, px(10.0, 0.0))).abs() < 1e-9, "{d}");
        assert!(-d < 10.0);
        // Between the centre of curvature and the vertex the vertex is
        // nearest.
        assert!((e.signed_distance(px(18.0, 0.0)) + 2.0).abs() < 1e-12);
    }

    #[test]
    fn ellipse_distances_agree_with_a_brute_force_search() {
        let e = ellipse(50.0, 30.0, 20.0, 8.0, 0.6);
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        for _ in 0..200 {
            let p = px(20.0 + 60.0 * next(), 5.0 + 50.0 * next());
            let exact = e.signed_distance(p);
            let brute = brute_distance(&e, p);
            assert!(
                (exact.abs() - brute).abs() < 1e-9,
                "{p:?}: {exact} vs {brute}"
            );
        }
    }

    #[test]
    fn points_on_an_ellipse_are_at_distance_zero_and_the_sign_tells_the_side() {
        let e = ellipse(-3.0, 7.0, 12.0, 4.0, -1.1);
        for k in 0..64 {
            let t = TAU * k as f64 / 64.0;
            let on = on_ellipse(&e, t);
            assert!(e.signed_distance(on).abs() < 1e-12);
            let out = e.center() + (on - e.center()) * 1.1;
            assert!(e.signed_distance(out) > 0.0);
            let inside = e.center() + (on - e.center()) * 0.9;
            assert!(e.signed_distance(inside) < 0.0);
        }
    }

    #[test]
    fn a_circular_ellipse_measures_like_a_circle() {
        let e = ellipse(1.0, 2.0, 5.0, 5.0, FRAC_PI_2);
        let c: Circle<Pixels> = Circle::try_new(px(1.0, 2.0), Length::new(5.0)).unwrap();
        for p in [px(9.0, 2.0), px(1.0, 4.0), px(-3.0, -5.0), px(1.0, 2.0)] {
            assert!((e.signed_distance(p) - c.signed_distance(p)).abs() < 1e-12);
        }
    }

    #[test]
    fn mapped_elements_hold_the_mapped_points() {
        use crate::geometry::{Affine, AffineMap, ConformalMap, Similarity};
        let shear: Affine<Pixels, Millimeter> =
            Affine::try_new([[0.02, 0.007], [-0.003, 0.05]], Vector::new(1.0, -2.0)).unwrap();
        let e = ellipse(30.0, -10.0, 15.0, 6.0, 0.9);
        let mapped = shear.map_ellipse(e);
        assert!(mapped.semi_major() >= mapped.semi_minor());
        let l: Line<Pixels> = Line::try_through(px(1.0, 2.0), px(7.0, -3.0)).unwrap();
        let mapped_line = shear.map_line(l);
        assert!((mapped_line.direction().length().get() - 1.0).abs() < 1e-15);
        for k in 0..32 {
            let t = TAU * k as f64 / 32.0;
            let on = shear.map_point(on_ellipse(&e, t));
            assert!(mapped.signed_distance(on).abs() < 1e-12, "{t}");
            let along = shear.map_point(l.point() + l.direction() * (t - 3.0));
            assert!(mapped_line.signed_distance(along).abs() < 1e-12);
        }
        let s: Segment<Pixels> = Segment::new(px(0.0, 0.0), px(100.0, 0.0));
        assert_eq!(shear.map_segment(s).end, shear.map_point(px(100.0, 0.0)));

        let turn: Similarity<Pixels, Millimeter> = Similarity::try_linear(0.5, 1.0).unwrap();
        let c: Circle<Pixels> = Circle::try_new(px(4.0, 0.0), Length::new(10.0)).unwrap();
        let mapped = turn.map_circle(c);
        assert!((mapped.radius().get() - 5.0).abs() < 1e-15);
        assert_eq!(mapped.center(), turn.map_point(px(4.0, 0.0)));
    }

    #[test]
    fn elements_compare_and_print_their_parts() {
        let l: Line<Pixels> = Line::try_through(px(0.0, 0.0), px(0.0, 2.0)).unwrap();
        assert_eq!(l, l.clone());
        assert_ne!(l, Line::try_through(px(0.0, 0.0), px(2.0, 0.0)).unwrap());
        assert_eq!(
            format!("{l:?}"),
            "Line { point: Point { x: 0.0, y: 0.0 }, direction: Vector { x: 0.0, y: 1.0 } }"
        );
        let s: Segment<Pixels> = Segment::new(px(1.0, 2.0), px(3.0, 4.0));
        assert_eq!(s, s.clone());
        assert_ne!(s, Segment::new(px(1.0, 2.0), px(3.0, 5.0)));
        assert_eq!(
            format!("{s:?}"),
            "Segment { start: Point { x: 1.0, y: 2.0 }, end: Point { x: 3.0, y: 4.0 } }"
        );
        let e = ellipse(0.0, 0.0, 2.0, 1.0, 0.0);
        assert_eq!(e, e.clone());
        assert_ne!(e, ellipse(0.0, 0.0, 2.0, 1.5, 0.0));
        assert!(format!("{e:?}").starts_with("Ellipse { center: Point { x: 0.0, y: 0.0 }"));
    }

    #[test]
    fn bisection_runs_to_the_last_step() {
        assert_eq!(BISECTION_STEPS, 1074);
    }

    #[test]
    fn elements_name_their_unit_and_need_no_derives_on_it() {
        struct Mil;
        impl LengthUnit for Mil {}
        fn unit_of<E: Element>(_: E) -> &'static str {
            core::any::type_name::<E::Unit>()
        }
        let c: Circle<Mil> = Circle::try_new(Point::new(0.0, 0.0), Length::new(1.0)).unwrap();
        let d = c;
        assert_eq!(c, d);
        assert!(unit_of(c).ends_with("Mil"));
        assert_eq!(
            format!("{c:?}"),
            "Circle { center: Point { x: 0.0, y: 0.0 }, radius: Length(1.0) }"
        );
    }
}

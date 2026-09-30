//! Positions, displacements and lengths in a unit of length.

use core::fmt;
use core::marker::PhantomData;
use core::ops::{Add, Mul, Neg, Sub};

use super::units::{LengthUnit, Meter, Pixels, Prefix, rescale};
use crate::CoordinateF64;

/// A position in the plane, in the unit `U`.
///
/// The unit is part of the type, so a position in pixels and one in
/// millimetres cannot be mixed: subtracting them, or passing one where the
/// other is expected, does not compile. The fields are public for reading
/// and writing; construction goes through [`Point::new`].
///
/// Points are not validated, as [`CoordinateF64`] is not: a point is any
/// pair of numbers, NaN included, and a length computed from a NaN point is
/// NaN.
///
/// The arithmetic is that of positions. The difference of two points is a
/// [`Vector`], a point plus a vector is a point, and the sum of two points
/// does not compile, because it has no meaning. The midpoint is
/// `a + (b - a) * 0.5`.
///
/// # Example
///
/// ```
/// use fovea::{Millimeter, Point};
/// use fovea::geometry::Vector;
///
/// let a: Point<Millimeter> = Point::new(1.0, 2.0);
/// let b: Point<Millimeter> = Point::new(4.0, 6.0);
/// assert_eq!(a.distance(b).get(), 5.0);
///
/// let midpoint = a + (b - a) * 0.5;
/// assert_eq!((midpoint.x, midpoint.y), (2.5, 4.0));
/// assert_eq!(b - a, Vector::new(3.0, 4.0));
/// ```
///
/// Two positions do not add up to a third:
///
/// ```compile_fail
/// use fovea::{Millimeter, Point};
///
/// let a: Point<Millimeter> = Point::new(1.0, 2.0);
/// let _ = a + a;
/// ```
///
/// and a position in pixels is not one in millimetres:
///
/// ```compile_fail
/// use fovea::{Millimeter, Pixels, Point};
///
/// let a: Point<Pixels> = Point::new(1.0, 2.0);
/// let b: Point<Millimeter> = Point::new(1.0, 2.0);
/// let _ = a - b;
/// ```
pub struct Point<U> {
    /// Horizontal position.
    pub x: f64,
    /// Vertical position, growing downwards in an image.
    pub y: f64,
    unit: PhantomData<fn() -> U>,
}

/// A displacement in the plane, in the unit `U`: the difference of two
/// [`Point`]s.
///
/// Vectors add, subtract, negate and scale by a number, and a point moves by
/// one. [`length`](Vector::length) gives the Euclidean length.
///
/// # Example
///
/// ```
/// use fovea::Pixels;
/// use fovea::geometry::Vector;
///
/// let v: Vector<Pixels> = Vector::new(3.0, 4.0);
/// assert_eq!(v.length().get(), 5.0);
/// assert_eq!(v * 2.0 - v, v);
/// assert_eq!(-v, Vector::new(-3.0, -4.0));
/// ```
pub struct Vector<U> {
    /// Horizontal component.
    pub x: f64,
    /// Vertical component.
    pub y: f64,
    unit: PhantomData<fn() -> U>,
}

/// A length in the unit `U`: a distance, a radius, a residual.
///
/// A unit carrier around one `f64`, not validated, for the same reason as
/// [`Point`]: a distance computed from NaN coordinates is NaN. A parameter
/// that must be positive, such as a tolerance, is its own validated type.
///
/// With non-square pixels a length in pixels has no single length in the
/// world, so only a conformal mapping converts one; see
/// [`ConformalMap`](super::ConformalMap).
///
/// # Example
///
/// ```
/// use fovea::{Length, Millimeter};
///
/// let nominal: Length<Millimeter> = Length::new(10.0);
/// let measured: Length<Millimeter> = Length::new(10.02);
/// assert!(measured > nominal);
/// assert!(((measured - nominal).get() - 0.02).abs() < 1e-12);
/// ```
pub struct Length<U> {
    value: f64,
    unit: PhantomData<fn() -> U>,
}

impl<U: LengthUnit> Point<U> {
    /// Creates a point at `(x, y)`.
    #[must_use]
    #[inline]
    pub const fn new(x: f64, y: f64) -> Self {
        Self {
            x,
            y,
            unit: PhantomData,
        }
    }

    /// The Euclidean distance from this point to `other`.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::{Pixels, Point};
    ///
    /// let a: Point<Pixels> = Point::new(0.0, 0.0);
    /// assert_eq!(a.distance(Point::new(6.0, 8.0)).get(), 10.0);
    /// ```
    #[must_use]
    #[inline]
    pub fn distance(self, other: Self) -> Length<U> {
        (other - self).length()
    }
}

impl<U: LengthUnit> Vector<U> {
    /// Creates a vector with components `(x, y)`.
    #[must_use]
    #[inline]
    pub const fn new(x: f64, y: f64) -> Self {
        Self {
            x,
            y,
            unit: PhantomData,
        }
    }

    /// The Euclidean length of the vector.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::Millimeter;
    /// use fovea::geometry::Vector;
    ///
    /// let v: Vector<Millimeter> = Vector::new(-5.0, 12.0);
    /// assert_eq!(v.length().get(), 13.0);
    /// ```
    #[must_use]
    #[inline]
    pub fn length(self) -> Length<U> {
        Length::new(self.x.hypot(self.y))
    }
}

impl<U: LengthUnit> Length<U> {
    /// Creates a length of `value` in the unit `U`.
    #[must_use]
    #[inline]
    pub const fn new(value: f64) -> Self {
        Self {
            value,
            unit: PhantomData,
        }
    }

    /// Returns the value in the unit `U`.
    #[must_use]
    #[inline]
    pub const fn get(self) -> f64 {
        self.value
    }
}

// ── Prefix conversion ────────────────────────────────────────────────────────

impl<P: Prefix> Point<Meter<P>> {
    /// The same point with the prefix `Q`, rescaled by an exact power of
    /// ten.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::{Millimeter, Point};
    /// use fovea::geometry::Micro;
    ///
    /// let p: Point<Millimeter> = Point::new(1.5, 0.25);
    /// let q = p.convert::<Micro>();
    /// assert_eq!((q.x, q.y), (1500.0, 250.0));
    /// ```
    #[must_use]
    pub fn convert<Q: Prefix>(self) -> Point<Meter<Q>> {
        Point::new(rescale::<P, Q>(self.x), rescale::<P, Q>(self.y))
    }
}

impl<P: Prefix> Vector<Meter<P>> {
    /// The same vector with the prefix `Q`, rescaled by an exact power of
    /// ten.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::Micrometer;
    /// use fovea::geometry::{Milli, Vector};
    ///
    /// let v: Vector<Micrometer> = Vector::new(250.0, -9.0);
    /// let w = v.convert::<Milli>();
    /// assert_eq!((w.x, w.y), (0.25, -0.009));
    /// ```
    #[must_use]
    pub fn convert<Q: Prefix>(self) -> Vector<Meter<Q>> {
        Vector::new(rescale::<P, Q>(self.x), rescale::<P, Q>(self.y))
    }
}

impl<P: Prefix> Length<Meter<P>> {
    /// The same length with the prefix `Q`, rescaled by an exact power of
    /// ten.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::{Length, Meter, Millimeter};
    /// use fovea::geometry::Unit;
    ///
    /// let l: Length<Millimeter> = Length::new(2500.0);
    /// let m: Length<Meter> = l.convert::<Unit>();
    /// assert_eq!(m.get(), 2.5);
    /// ```
    #[must_use]
    pub fn convert<Q: Prefix>(self) -> Length<Meter<Q>> {
        Length::new(rescale::<P, Q>(self.value))
    }
}

// ── Bridges to the pixel coordinate family ──────────────────────────────────

impl From<CoordinateF64> for Point<Pixels> {
    #[inline]
    fn from(c: CoordinateF64) -> Self {
        Point::new(c.x, c.y)
    }
}

impl From<Point<Pixels>> for CoordinateF64 {
    #[inline]
    fn from(p: Point<Pixels>) -> Self {
        CoordinateF64::new(p.x, p.y)
    }
}

// ── Arithmetic ───────────────────────────────────────────────────────────────

impl<U: LengthUnit> Sub for Point<U> {
    type Output = Vector<U>;
    #[inline]
    fn sub(self, rhs: Self) -> Vector<U> {
        Vector::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl<U: LengthUnit> Add<Vector<U>> for Point<U> {
    type Output = Point<U>;
    #[inline]
    fn add(self, rhs: Vector<U>) -> Point<U> {
        Point::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl<U: LengthUnit> Sub<Vector<U>> for Point<U> {
    type Output = Point<U>;
    #[inline]
    fn sub(self, rhs: Vector<U>) -> Point<U> {
        Point::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl<U: LengthUnit> Add for Vector<U> {
    type Output = Vector<U>;
    #[inline]
    fn add(self, rhs: Self) -> Vector<U> {
        Vector::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl<U: LengthUnit> Sub for Vector<U> {
    type Output = Vector<U>;
    #[inline]
    fn sub(self, rhs: Self) -> Vector<U> {
        Vector::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl<U: LengthUnit> Neg for Vector<U> {
    type Output = Vector<U>;
    #[inline]
    fn neg(self) -> Vector<U> {
        Vector::new(-self.x, -self.y)
    }
}

impl<U: LengthUnit> Mul<f64> for Vector<U> {
    type Output = Vector<U>;
    #[inline]
    fn mul(self, k: f64) -> Vector<U> {
        Vector::new(self.x * k, self.y * k)
    }
}

impl<U: LengthUnit> Mul<Vector<U>> for f64 {
    type Output = Vector<U>;
    #[inline]
    fn mul(self, v: Vector<U>) -> Vector<U> {
        v * self
    }
}

impl<U: LengthUnit> Add for Length<U> {
    type Output = Length<U>;
    #[inline]
    fn add(self, rhs: Self) -> Length<U> {
        Length::new(self.value + rhs.value)
    }
}

impl<U: LengthUnit> Sub for Length<U> {
    type Output = Length<U>;
    #[inline]
    fn sub(self, rhs: Self) -> Length<U> {
        Length::new(self.value - rhs.value)
    }
}

// ── Common traits, without bounds on the unit ───────────────────────────────
//
// Derives would require `U: Clone`, `U: Debug` and so on, which a marker of
// the caller's own need not implement; the unit is never stored.

impl<U> Clone for Point<U> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}
impl<U> Copy for Point<U> {}
impl<U> PartialEq for Point<U> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y
    }
}
impl<U> fmt::Debug for Point<U> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Point")
            .field("x", &self.x)
            .field("y", &self.y)
            .finish()
    }
}

impl<U> Clone for Vector<U> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}
impl<U> Copy for Vector<U> {}
impl<U> PartialEq for Vector<U> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y
    }
}
impl<U> fmt::Debug for Vector<U> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Vector")
            .field("x", &self.x)
            .field("y", &self.y)
            .finish()
    }
}

impl<U> Clone for Length<U> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}
impl<U> Copy for Length<U> {}
impl<U> PartialEq for Length<U> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}
impl<U> PartialOrd for Length<U> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.value.partial_cmp(&other.value)
    }
}
impl<U> fmt::Debug for Length<U> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Length").field(&self.value).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{Micro, Micrometer, Milli, Millimeter, Unit};

    fn mm(x: f64, y: f64) -> Point<Millimeter> {
        Point::new(x, y)
    }

    #[test]
    fn differences_of_points_are_vectors_and_move_points() {
        let a = mm(1.0, 2.0);
        let b = mm(4.0, -2.0);
        let v = b - a;
        assert_eq!(v, Vector::new(3.0, -4.0));
        assert_eq!(a + v, b);
        assert_eq!(b - v, a);
        assert_eq!(a.distance(b), Length::new(5.0));
        assert_eq!(b.distance(a), Length::new(5.0));
    }

    #[test]
    fn vector_arithmetic() {
        let v: Vector<Pixels> = Vector::new(1.0, -2.0);
        let w: Vector<Pixels> = Vector::new(0.5, 4.0);
        assert_eq!(v + w, Vector::new(1.5, 2.0));
        assert_eq!(v - w, Vector::new(0.5, -6.0));
        assert_eq!(-v, Vector::new(-1.0, 2.0));
        assert_eq!(v * 3.0, Vector::new(3.0, -6.0));
        assert_eq!(3.0 * v, v * 3.0);
        assert_eq!(Vector::<Pixels>::new(8.0, 15.0).length(), Length::new(17.0));
    }

    #[test]
    fn lengths_add_subtract_and_order() {
        let a: Length<Micrometer> = Length::new(3.0);
        let b: Length<Micrometer> = Length::new(4.5);
        assert_eq!((a + b).get(), 7.5);
        assert_eq!((b - a).get(), 1.5);
        assert!(a < b);
        assert!(
            Length::<Micrometer>::new(f64::NAN)
                .partial_cmp(&a)
                .is_none()
        );
    }

    #[test]
    fn a_nan_coordinate_gives_a_nan_distance() {
        let d = mm(f64::NAN, 0.0).distance(mm(1.0, 1.0));
        assert!(d.get().is_nan());
    }

    #[test]
    fn prefix_conversion_is_exact_for_representable_values() {
        let p = mm(12.5, -0.25).convert::<Micro>();
        assert_eq!((p.x, p.y), (12_500.0, -250.0));
        let back = p.convert::<Milli>();
        assert_eq!(back, mm(12.5, -0.25));

        let v: Vector<Micrometer> = Vector::new(9.0, 1.0);
        assert_eq!(v.convert::<Milli>(), Vector::new(0.009, 0.001));

        let l: Length<Millimeter> = Length::new(1500.0);
        assert_eq!(l.convert::<Unit>().get(), 1.5);
    }

    #[test]
    fn coordinates_bridge_both_ways() {
        let c = CoordinateF64::new(3.25, 7.5);
        let p: Point<Pixels> = c.into();
        assert_eq!((p.x, p.y), (3.25, 7.5));
        let back: CoordinateF64 = p.into();
        assert_eq!(back, c);
    }

    #[test]
    fn a_unit_of_ones_own_needs_no_derives() {
        struct Mil;
        impl LengthUnit for Mil {}
        let a: Point<Mil> = Point::new(1.0, 1.0);
        let b = a; // Copy without `Mil: Copy`
        assert_eq!(a, b);
        assert_eq!(format!("{a:?}"), "Point { x: 1.0, y: 1.0 }");
        assert_eq!(format!("{:?}", Length::<Mil>::new(2.0)), "Length(2.0)");
    }
}

//! Relations between points and elements: distances, nearest points, the
//! angle between two lines and where they meet.
//!
//! Every relation takes its operands in one unit, which the types enforce,
//! and computes in that unit's Euclidean distances. Under a calibration that
//! is not conformal, convert the points before fitting and relating.

use core::f64::consts::PI;

use super::elements::{Circle, Ellipse, Line, Segment};
use super::point::{Length, Point, Vector};
use super::units::LengthUnit;

fn dot<U>(a: Vector<U>, b: Vector<U>) -> f64 {
    a.x * b.x + a.y * b.y
}

fn cross<U>(a: Vector<U>, b: Vector<U>) -> f64 {
    a.x * b.y - a.y * b.x
}

impl<U: LengthUnit> Line<U> {
    /// The same line, pointing the other way.
    ///
    /// For when the points arrived in the opposite order to the one a
    /// directed relation such as [`angle_to`](Self::angle_to) needs.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::{Line, Pixels, Point};
    /// use fovea::geometry::Vector;
    ///
    /// let l: Line<Pixels> = Line::try_through(Point::new(0.0, 0.0), Point::new(3.0, 0.0))?;
    /// assert_eq!(l.reversed().direction(), Vector::new(-1.0, 0.0));
    /// assert_eq!(l.reversed().point(), l.point());
    /// # Ok::<(), fovea::Error>(())
    /// ```
    #[must_use]
    pub fn reversed(&self) -> Self {
        Self::from_parts(self.point(), -self.direction())
    }

    /// The distance of `p` from the line.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::{Line, Pixels, Point};
    ///
    /// // A hole centre 12.5 px from a horizontal edge.
    /// let edge: Line<Pixels> = Line::try_through(Point::new(0.0, 40.0), Point::new(100.0, 40.0))?;
    /// assert_eq!(edge.distance(Point::new(30.0, 52.5)).get(), 12.5);
    /// # Ok::<(), fovea::Error>(())
    /// ```
    #[must_use]
    pub fn distance(&self, p: Point<U>) -> Length<U> {
        Length::new(self.signed_distance(p).abs())
    }

    /// The point of the line nearest to `p`: the foot of the perpendicular
    /// from `p`.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::{Line, Pixels, Point};
    ///
    /// let edge: Line<Pixels> = Line::try_through(Point::new(0.0, 40.0), Point::new(100.0, 40.0))?;
    /// assert_eq!(edge.closest_point(Point::new(30.0, 52.5)), Point::new(30.0, 40.0));
    /// # Ok::<(), fovea::Error>(())
    /// ```
    #[must_use]
    pub fn closest_point(&self, p: Point<U>) -> Point<U> {
        self.point() + self.direction() * self.along(p)
    }

    /// The signed angle, in radians in `(−π, π]`, that turns this line's
    /// direction onto `other`'s.
    ///
    /// Positive turns `+x` towards `+y`, as every angle in the crate, which in
    /// an image with `y` pointing down is clockwise on the screen. The angle
    /// depends on both directions: a fitted line points from its first point
    /// towards its last, and [`reversed`](Self::reversed) turns one round.
    /// A calibration that reflects, such as one into a frame with `y`
    /// pointing up, changes the sign.
    ///
    /// The angle between the two lines' axes, regardless of direction, is
    /// `θ.abs().min(π − θ.abs())`, in `[0, π/2]`.
    ///
    /// # Example
    ///
    /// ```
    /// use core::f64::consts::PI;
    /// use fovea::{Line, Pixels, Point};
    ///
    /// // Two sides of a hexagon, both measured away from their corner.
    /// let corner: Point<Pixels> = Point::new(50.0, 50.0);
    /// let (s, c) = (120f64.to_radians()).sin_cos();
    /// let a: Line<Pixels> = Line::try_through(corner, Point::new(60.0, 50.0))?;
    /// let b: Line<Pixels> = Line::try_through(corner, Point::new(50.0 + 10.0 * c, 50.0 + 10.0 * s))?;
    ///
    /// let theta = a.angle_to(&b);
    /// assert!((theta.to_degrees() - 120.0).abs() < 1e-12);
    /// assert!((b.angle_to(&a).to_degrees() + 120.0).abs() < 1e-12);
    ///
    /// // Without direction: the acute angle between the axes.
    /// let axes = theta.abs().min(PI - theta.abs());
    /// assert!((axes.to_degrees() - 60.0).abs() < 1e-12);
    /// # Ok::<(), fovea::Error>(())
    /// ```
    #[must_use]
    pub fn angle_to(&self, other: &Self) -> f64 {
        let (d, e) = (self.direction(), other.direction());
        let theta = cross(d, e).atan2(dot(d, e));
        // `atan2` returns −π for a negative zero beside a negative cosine;
        // the canonical range keeps +π.
        if theta == -PI { PI } else { theta }
    }

    /// Where the two lines meet, or `None` if they are parallel.
    ///
    /// The lines are infinite, so two fitted sides of a part meet at the
    /// corner a drawing dimensions even where a chamfer or a radius stops
    /// the measured edges short of it. Lines that are nearly parallel meet
    /// far away; `None` comes only for lines that are exactly parallel, or
    /// that meet beyond the range of `f64`.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::{Line, Pixels, Point};
    ///
    /// // The top side ends at x = 8 and the right side starts at y = 3,
    /// // short of the corner, which lies at (11, 0).
    /// let top: Line<Pixels> = Line::try_through(Point::new(0.0, 0.0), Point::new(8.0, 0.0))?;
    /// let right: Line<Pixels> = Line::try_through(Point::new(11.0, 3.0), Point::new(11.0, 10.0))?;
    /// assert_eq!(top.intersection(&right), Some(Point::new(11.0, 0.0)));
    ///
    /// let below: Line<Pixels> = Line::try_through(Point::new(0.0, 5.0), Point::new(8.0, 5.0))?;
    /// assert_eq!(top.intersection(&below), None);
    /// # Ok::<(), fovea::Error>(())
    /// ```
    #[must_use]
    pub fn intersection(&self, other: &Self) -> Option<Point<U>> {
        let (d, e) = (self.direction(), other.direction());
        let denominator = cross(d, e);
        if denominator == 0.0 {
            return None;
        }
        let s = cross(other.point() - self.point(), e) / denominator;
        let p = self.point() + d * s;
        (p.x.is_finite() && p.y.is_finite()).then_some(p)
    }
}

impl<U: LengthUnit> Segment<U> {
    /// The distance of `p` from the segment: from the nearest point between
    /// its ends, both included.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::{Pixels, Point, Segment};
    ///
    /// let s: Segment<Pixels> = Segment::new(Point::new(0.0, 0.0), Point::new(10.0, 0.0));
    /// assert_eq!(s.distance(Point::new(4.0, 3.0)).get(), 3.0);
    /// // Beyond an end, the distance is to the end.
    /// assert_eq!(s.distance(Point::new(13.0, 4.0)).get(), 5.0);
    /// ```
    #[must_use]
    pub fn distance(&self, p: Point<U>) -> Length<U> {
        p.distance(self.closest_point(p))
    }

    /// The point of the segment nearest to `p`, its ends included.
    ///
    /// A segment of length zero answers with its start.
    #[must_use]
    pub fn closest_point(&self, p: Point<U>) -> Point<U> {
        let v = self.end - self.start;
        let length_sq = dot(v, v);
        if length_sq == 0.0 {
            return self.start;
        }
        let t = (dot(p - self.start, v) / length_sq).clamp(0.0, 1.0);
        self.start + v * t
    }

    /// The smallest and the largest distance of the segment's points from
    /// `line`, in that order.
    ///
    /// The distance from a line changes linearly along a segment, so both
    /// come from the ends, and the smallest is zero where the segment
    /// crosses the line. For the width between two nearly parallel fitted
    /// edges, take the `extent()` of one [`Fit`](crate::measure::Fit) and
    /// the line of the other: the two numbers are the width at the
    /// ends of the measured part and say how much it tapers, over the part
    /// of the edge that was measured.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::{Line, Pixels, Point, Segment};
    ///
    /// // A slot that widens from 20.0 to 20.2 px over the 100 px measured.
    /// let edge_a: Segment<Pixels> = Segment::new(Point::new(0.0, 0.0), Point::new(100.0, 0.0));
    /// let edge_b: Line<Pixels> = Line::try_through(Point::new(0.0, 20.0), Point::new(100.0, 20.2))?;
    /// let (min, max) = edge_a.distances_to(&edge_b);
    /// // Perpendicular to the slightly tilted edge b: 20 and 20.2 times the
    /// // cosine of its tilt, 19.99996 and 20.19996.
    /// assert!((min.get() - 20.0).abs() < 1e-4 && (max.get() - 20.2).abs() < 1e-4);
    /// # Ok::<(), fovea::Error>(())
    /// ```
    #[must_use]
    pub fn distances_to(&self, line: &Line<U>) -> (Length<U>, Length<U>) {
        let (s0, s1) = (
            line.signed_distance(self.start),
            line.signed_distance(self.end),
        );
        let (a0, a1) = (s0.abs(), s1.abs());
        let crosses = (s0 <= 0.0 && s1 >= 0.0) || (s0 >= 0.0 && s1 <= 0.0);
        let min = if crosses { 0.0 } else { a0.min(a1) };
        (Length::new(min), Length::new(a0.max(a1)))
    }
}

impl<U: LengthUnit> Circle<U> {
    /// The distance of `p` from the circle: from its outline, inside or
    /// outside.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::{Circle, Length, Pixels, Point};
    ///
    /// let hole: Circle<Pixels> = Circle::try_new(Point::new(0.0, 0.0), Length::new(5.0))?;
    /// assert_eq!(hole.distance(Point::new(0.0, 8.0)).get(), 3.0);
    /// assert_eq!(hole.distance(Point::new(3.0, 0.0)).get(), 2.0);
    /// # Ok::<(), fovea::Error>(())
    /// ```
    #[must_use]
    pub fn distance(&self, p: Point<U>) -> Length<U> {
        Length::new(self.signed_distance(p).abs())
    }

    /// The point of the circle nearest to `p`.
    ///
    /// Every point of the circle is nearest to its centre; there the answer
    /// is `center + (0, radius)`, the point in the `+y` direction, as for an
    /// ellipse with equal semi-axes and orientation zero.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::{Circle, Length, Pixels, Point};
    ///
    /// let hole: Circle<Pixels> = Circle::try_new(Point::new(10.0, 10.0), Length::new(5.0))?;
    /// assert_eq!(hole.closest_point(Point::new(10.0, 30.0)), Point::new(10.0, 15.0));
    /// assert_eq!(hole.closest_point(hole.center()), Point::new(10.0, 15.0));
    /// # Ok::<(), fovea::Error>(())
    /// ```
    #[must_use]
    pub fn closest_point(&self, p: Point<U>) -> Point<U> {
        let r = self.radius().get();
        let d = p - self.center();
        let n = d.length().get();
        if n == 0.0 {
            return self.center() + Vector::new(0.0, r);
        }
        self.center() + Vector::new(d.x / n * r, d.y / n * r)
    }
}

impl<U: LengthUnit> Ellipse<U> {
    /// The exact distance of `p` from the ellipse, inside or outside: the
    /// length of the perpendicular from `p` to its nearest point.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::{AxialOrientation, Ellipse, Length, Pixels, Point};
    ///
    /// let e: Ellipse<Pixels> = Ellipse::try_new(
    ///     Point::new(0.0, 0.0), Length::new(20.0), Length::new(8.0),
    ///     AxialOrientation::from_radians(0.0)?,
    /// )?;
    /// assert_eq!(e.distance(Point::new(25.0, 0.0)).get(), 5.0);
    /// assert_eq!(e.distance(Point::new(0.0, 3.0)).get(), 5.0);
    /// # Ok::<(), fovea::Error>(())
    /// ```
    #[must_use]
    pub fn distance(&self, p: Point<U>) -> Length<U> {
        Length::new(self.foot_and_signed_distance(p).1.abs())
    }

    /// The point of the ellipse nearest to `p`.
    ///
    /// Two points are equally near to a point on the major axis between the
    /// centres of curvature of the two vertices, the centre included. There
    /// the answer is the one on the side the minor axis's positive direction
    /// points to (the orientation of the major axis turned by `+π/2`), which
    /// at the centre is the end of the minor axis.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::{AxialOrientation, Ellipse, Length, Pixels, Point};
    ///
    /// let e: Ellipse<Pixels> = Ellipse::try_new(
    ///     Point::new(0.0, 0.0), Length::new(20.0), Length::new(8.0),
    ///     AxialOrientation::from_radians(0.0)?,
    /// )?;
    /// assert_eq!(e.closest_point(Point::new(25.0, 0.0)), Point::new(20.0, 0.0));
    /// assert_eq!(e.closest_point(e.center()), Point::new(0.0, 8.0));
    /// # Ok::<(), fovea::Error>(())
    /// ```
    #[must_use]
    pub fn closest_point(&self, p: Point<U>) -> Point<U> {
        self.foot_and_signed_distance(p).0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AxialOrientation;
    use crate::geometry::Pixels;
    use core::f64::consts::{FRAC_PI_2, TAU};

    fn px(x: f64, y: f64) -> Point<Pixels> {
        Point::new(x, y)
    }

    fn line(from: (f64, f64), to: (f64, f64)) -> Line<Pixels> {
        Line::try_through(px(from.0, from.1), px(to.0, to.1)).unwrap()
    }

    fn ellipse(a: f64, b: f64, phi: f64) -> Ellipse<Pixels> {
        Ellipse::try_new(
            px(3.0, -2.0),
            Length::new(a),
            Length::new(b),
            AxialOrientation::from_radians(phi).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn angles_between_directed_lines() {
        let east = line((0.0, 0.0), (1.0, 0.0));
        let cases = [
            ((1.0, 0.0), 0.0),
            ((0.0, 1.0), FRAC_PI_2),
            ((0.0, -1.0), -FRAC_PI_2),
            ((-1.0, 0.0), PI),
            ((1.0, 1.0), FRAC_PI_2 / 2.0),
        ];
        for (to, expected) in cases {
            let other = line((0.0, 0.0), to);
            assert!((east.angle_to(&other) - expected).abs() < 1e-15, "{to:?}");
        }
        // Opposite directions are +π, never −π, whichever way round.
        let west = east.reversed();
        assert_eq!(east.angle_to(&west), PI);
        assert_eq!(west.angle_to(&east), PI);
        // Reversing one line turns the angle by π.
        let north_east = line((0.0, 0.0), (1.0, 1.0));
        let turned = east.angle_to(&north_east.reversed());
        assert!((turned - (FRAC_PI_2 / 2.0 - PI)).abs() < 1e-15);
    }

    #[test]
    fn the_angle_does_not_depend_on_where_the_lines_lie() {
        let a = line((0.0, 0.0), (10.0, 0.0));
        let b = line((500.0, -300.0), (500.0, -290.0));
        assert_eq!(a.angle_to(&b), FRAC_PI_2);
    }

    #[test]
    fn lines_meet_at_their_intersection_or_not_at_all() {
        let a = line((0.0, 0.0), (4.0, 2.0));
        let b = line((0.0, 6.0), (6.0, 0.0));
        let p = a.intersection(&b).unwrap();
        assert!(p.distance(px(4.0, 2.0)).get() < 1e-12);
        assert_eq!(
            b.intersection(&a).map(|q| q.distance(p).get() < 1e-12),
            Some(true)
        );
        // Parallel and identical lines have no single intersection.
        let parallel = line((0.0, 1.0), (4.0, 3.0));
        assert_eq!(a.intersection(&parallel), None);
        assert_eq!(a.intersection(&a), None);
        // Nearly parallel lines meet far away, and the point is on both.
        let nearly = line((0.0, 1.0), (1e6, 0.5e6 + 1.5));
        let far = a.intersection(&nearly).unwrap();
        let reach = far.x.abs();
        assert!(reach > 1e5);
        assert!(a.distance(far).get() < 1e-6 * reach && nearly.distance(far).get() < 1e-6 * reach);
    }

    #[test]
    fn a_line_measures_points_and_their_feet() {
        let l = line((1.0, 1.0), (4.0, 5.0));
        let p = px(5.0, -2.0);
        // (5, −2) − (1, 1) = (4, −3), perpendicular to (3, 4) and 5 long.
        assert!((l.distance(p).get() - 5.0).abs() < 1e-12);
        assert!(l.closest_point(p).distance(px(1.0, 1.0)).get() < 1e-12);
        let on = px(7.0, 9.0);
        assert!(l.distance(on).get() < 1e-12);
        assert!(l.closest_point(on).distance(on).get() < 1e-12);
    }

    #[test]
    fn a_segment_clamps_to_its_ends() {
        let s: Segment<Pixels> = Segment::new(px(0.0, 0.0), px(10.0, 0.0));
        assert_eq!(s.closest_point(px(-3.0, 4.0)), px(0.0, 0.0));
        assert_eq!(s.distance(px(-3.0, 4.0)).get(), 5.0);
        assert_eq!(s.closest_point(px(6.0, -2.0)), px(6.0, 0.0));
        let point: Segment<Pixels> = Segment::new(px(2.0, 2.0), px(2.0, 2.0));
        assert_eq!(point.closest_point(px(5.0, 6.0)), px(2.0, 2.0));
        assert_eq!(point.distance(px(5.0, 6.0)).get(), 5.0);
    }

    #[test]
    fn a_segment_reports_its_nearest_and_furthest_distance_from_a_line() {
        let l = line((0.0, 0.0), (1.0, 0.0));
        let above: Segment<Pixels> = Segment::new(px(0.0, 2.0), px(10.0, 3.0));
        assert_eq!(above.distances_to(&l), (Length::new(2.0), Length::new(3.0)));
        let below: Segment<Pixels> = Segment::new(px(0.0, -3.0), px(10.0, -2.0));
        assert_eq!(below.distances_to(&l), (Length::new(2.0), Length::new(3.0)));
        let crossing: Segment<Pixels> = Segment::new(px(0.0, -1.0), px(10.0, 4.0));
        assert_eq!(
            crossing.distances_to(&l),
            (Length::new(0.0), Length::new(4.0))
        );
        let touching: Segment<Pixels> = Segment::new(px(0.0, 0.0), px(10.0, 4.0));
        assert_eq!(
            touching.distances_to(&l),
            (Length::new(0.0), Length::new(4.0))
        );
    }

    #[test]
    fn a_circle_measures_points_inside_and_outside() {
        let c: Circle<Pixels> = Circle::try_new(px(1.0, 2.0), Length::new(5.0)).unwrap();
        assert_eq!(c.distance(px(1.0, 9.0)).get(), 2.0);
        assert_eq!(c.distance(px(4.0, 6.0)).get(), 0.0);
        assert_eq!(c.distance(px(1.0, 2.0)).get(), 5.0);
        assert_eq!(c.closest_point(px(13.0, 2.0)), px(6.0, 2.0));
        assert_eq!(c.closest_point(px(1.0, 2.0)), px(1.0, 7.0));
        let on = px(-2.0, -2.0);
        assert!(c.closest_point(on).distance(on).get() < 1e-12);
    }

    #[test]
    fn ellipse_feet_lie_on_the_ellipse_along_the_normal() {
        let e = ellipse(20.0, 8.0, 0.6);
        let (sin, cos) = 0.6f64.sin_cos();
        for k in 0..48 {
            let t = TAU * k as f64 / 48.0;
            let (u, v) = (20.0 * t.cos(), 8.0 * t.sin());
            let on = e.center() + Vector::new(u * cos - v * sin, u * sin + v * cos);
            // The outward normal at `on`, in the ellipse's frame (u/a², v/b²).
            let (nu, nv) = (u / 400.0, v / 64.0);
            let n = Vector::<Pixels>::new(nu * cos - nv * sin, nu * sin + nv * cos);
            let n = n * (1.0 / n.length().get());
            let out = on + n * 3.0;
            assert!(e.closest_point(out).distance(on).get() < 1e-9, "{t}");
            assert!((e.distance(out).get() - 3.0).abs() < 1e-9, "{t}");
            assert!(e.closest_point(on).distance(on).get() < 1e-9);
        }
    }

    #[test]
    fn ties_on_an_ellipse_go_to_the_positive_minor_side() {
        let e = ellipse(20.0, 8.0, 0.0);
        let c = e.center();
        // The centre: the end of the minor axis in its positive direction.
        assert_eq!(e.closest_point(c), c + Vector::new(0.0, 8.0));
        // On the major axis inside the centres of curvature (16.8 from the
        // centre): the upper of two equally near points.
        let p = c + Vector::new(10.0, 0.0);
        let q = e.closest_point(p);
        assert!(q.y > c.y);
        let mirrored = px(q.x, 2.0 * c.y - q.y);
        assert!((p.distance(q).get() - p.distance(mirrored).get()).abs() < 1e-12);
        // Equal semi-axes keep the same rule at the centre.
        let round = ellipse(5.0, 5.0, 0.0);
        assert_eq!(
            round.closest_point(round.center()),
            round.center() + Vector::new(0.0, 5.0)
        );
        let turned = ellipse(5.0, 5.0, FRAC_PI_2);
        let q = turned.closest_point(turned.center());
        assert!(q.distance(turned.center() + Vector::new(-5.0, 0.0)).get() < 1e-12);
    }
}

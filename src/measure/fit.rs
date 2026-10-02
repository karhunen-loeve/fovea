//! Fitting lines, circles and ellipses to points, with their residuals.

use super::estimators::{fit_circle, fit_ellipse, fit_line};
use crate::Error;
use crate::error::{ParameterError, Requirement, Value};
use crate::geometry::{
    Circle, ConformalMap, Element, Ellipse, Length, LengthUnit, Line, Point, Segment,
};

mod sealed {
    use crate::Error;
    use crate::geometry::{LengthUnit, Point};

    pub trait Fitter {
        fn fit_weighted<U: LengthUnit>(
            &self,
            points: &[Point<U>],
            weights: &[f64],
        ) -> Result<<Self as super::Estimator>::Element<U>, Error>
        where
            Self: super::Estimator;

        fn signed_residual<U: LengthUnit>(
            element: &<Self as super::Estimator>::Element<U>,
            p: Point<U>,
        ) -> f64
        where
            Self: super::Estimator;

        /// The element in its final form, and the extent of the points used
        /// along it as two positions; the default keeps the element and has
        /// no extent.
        fn finish<U: LengthUnit>(
            element: <Self as super::Estimator>::Element<U>,
            _points: &[Point<U>],
            _used: &[bool],
        ) -> (<Self as super::Estimator>::Element<U>, [f64; 2])
        where
            Self: super::Estimator,
        {
            (element, [0.0, 0.0])
        }
    }

    #[derive(Clone, Copy, Debug)]
    pub enum Weighting {
        AllPoints,
        Huber(f64),
        Tukey(f64),
    }

    pub trait Rule {
        fn weighting(&self) -> Weighting;
    }
}

use sealed::Weighting;

/// A method that fits one kind of element to points: the second parameter
/// of [`try_fit`].
///
/// The estimator is a value, so the method is written at the call site and
/// a different one for the same element can join later without changing
/// `try_fit`. Sealed.
pub trait Estimator: sealed::Fitter + Copy {
    /// The element it fits to points in the unit `U`.
    type Element<U: LengthUnit>: Element<Unit = U>;

    /// The fewest points that determine one element. Fewer is an
    /// [`Error::TooFewPoints`].
    const MIN_POINTS: usize;
}

/// The line through points by total least squares: the line that minimises
/// the sum of the squared orthogonal distances.
///
/// Every direction is represented the same way, a vertical line included,
/// which a fit of `y = m·x + b` cannot do; and the distances it minimises
/// are the ones its residuals report. It needs two distinct points.
///
/// # Example
///
/// ```
/// use fovea::{Pixels, Point};
/// use fovea::measure::{AllPoints, TotalLeastSquares, try_fit};
///
/// // A vertical edge.
/// let points: Vec<Point<Pixels>> = (0..5).map(|k| Point::new(12.5, k as f64)).collect();
/// let fit = try_fit(&points, TotalLeastSquares, AllPoints)?;
/// assert_eq!(fit.element().point().x, 12.5);
/// assert_eq!(fit.max_residual().get(), 0.0);
/// # Ok::<(), fovea::Error>(())
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TotalLeastSquares;

/// The circle through points by Taubin's method.
///
/// An algebraic fit in closed form, with no iteration that could diverge,
/// whose constraint removes most of the plain algebraic fit's bias towards
/// small circles; on a short arc it stays close to the geometric fit. It
/// needs three points that do not lie on one line.
///
/// # Example
///
/// ```
/// use fovea::{Pixels, Point};
/// use fovea::measure::{AllPoints, Taubin, try_fit};
///
/// let points: [Point<Pixels>; 4] =
///     [Point::new(15.0, 10.0), Point::new(10.0, 15.0), Point::new(5.0, 10.0), Point::new(10.0, 5.0)];
/// let fit = try_fit(&points, Taubin, AllPoints)?;
/// assert!((fit.element().radius().get() - 5.0).abs() < 1e-12);
/// # Ok::<(), fovea::Error>(())
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Taubin;

/// The ellipse through points by the direct least squares method of
/// Fitzgibbon, Pilu and Fisher, in the numerically stable form of Halíř and
/// Flusser.
///
/// An algebraic fit in closed form whose constraint admits only ellipses,
/// so it returns one even for points from a short arc. It needs five
/// points that do not lie on one line. The residuals are exact geometric
/// distances to the ellipse, not the algebraic error the method minimises.
///
/// # Example
///
/// ```
/// use fovea::{Pixels, Point};
/// use fovea::measure::{AllPoints, Fitzgibbon, try_fit};
///
/// // Points of the ellipse (x / 20)² + (y / 8)² = 1 about (50, 30).
/// let points: Vec<Point<Pixels>> = (0..12)
///     .map(|k| {
///         let t = k as f64 * core::f64::consts::TAU / 12.0;
///         Point::new(50.0 + 20.0 * t.cos(), 30.0 + 8.0 * t.sin())
///     })
///     .collect();
/// let fit = try_fit(&points, Fitzgibbon, AllPoints)?;
/// assert!((fit.element().semi_major().get() - 20.0).abs() < 1e-9);
/// assert!((fit.element().semi_minor().get() - 8.0).abs() < 1e-9);
/// # Ok::<(), fovea::Error>(())
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Fitzgibbon;

impl sealed::Fitter for TotalLeastSquares {
    fn fit_weighted<U: LengthUnit>(
        &self,
        points: &[Point<U>],
        weights: &[f64],
    ) -> Result<Line<U>, Error> {
        fit_line(points, weights)
    }

    fn signed_residual<U: LengthUnit>(element: &Line<U>, p: Point<U>) -> f64 {
        element.signed_distance(p)
    }

    /// Points the line from the first point used towards the last, and
    /// measures the extent of the points used along it.
    fn finish<U: LengthUnit>(
        element: Line<U>,
        points: &[Point<U>],
        used: &[bool],
    ) -> (Line<U>, [f64; 2]) {
        let mut kept = points
            .iter()
            .zip(used)
            .filter(|(_, u)| **u)
            .map(|(p, _)| *p);
        let first = kept.next();
        let last = kept.next_back().or(first);
        let mut line = element;
        if let (Some(first), Some(last)) = (first, last) {
            let d = line.direction();
            let span = last - first;
            if span.x * d.x + span.y * d.y < 0.0 {
                line = Line::from_parts(line.point(), -d);
            }
        }
        let mut extent = [f64::INFINITY, f64::NEG_INFINITY];
        for (p, _) in points.iter().zip(used).filter(|(_, u)| **u) {
            let t = line.along(*p);
            extent = [extent[0].min(t), extent[1].max(t)];
        }
        (line, extent)
    }
}

impl Estimator for TotalLeastSquares {
    type Element<U: LengthUnit> = Line<U>;
    const MIN_POINTS: usize = 2;
}

impl sealed::Fitter for Taubin {
    fn fit_weighted<U: LengthUnit>(
        &self,
        points: &[Point<U>],
        weights: &[f64],
    ) -> Result<Circle<U>, Error> {
        fit_circle(points, weights)
    }

    fn signed_residual<U: LengthUnit>(element: &Circle<U>, p: Point<U>) -> f64 {
        element.signed_distance(p)
    }
}

impl Estimator for Taubin {
    type Element<U: LengthUnit> = Circle<U>;
    const MIN_POINTS: usize = 3;
}

impl sealed::Fitter for Fitzgibbon {
    fn fit_weighted<U: LengthUnit>(
        &self,
        points: &[Point<U>],
        weights: &[f64],
    ) -> Result<Ellipse<U>, Error> {
        fit_ellipse(points, weights)
    }

    fn signed_residual<U: LengthUnit>(element: &Ellipse<U>, p: Point<U>) -> f64 {
        element.signed_distance(p)
    }
}

impl Estimator for Fitzgibbon {
    type Element<U: LengthUnit> = Ellipse<U>;
    const MIN_POINTS: usize = 5;
}

/// How a fit treats points far from the element: the third parameter of
/// [`try_fit`].
///
/// There is no default: [`AllPoints`] is written out. [`Huber`] and
/// [`Tukey`] take a threshold in the unit of the points, so a threshold in
/// pixels does not compile against points in millimetres. Sealed.
pub trait OutlierRule<U: LengthUnit>: sealed::Rule + Copy {}

/// Every point counts with the same weight; nothing is an outlier.
///
/// # Example
///
/// ```
/// use fovea::{Pixels, Point};
/// use fovea::measure::{AllPoints, TotalLeastSquares, try_fit};
///
/// let points: [Point<Pixels>; 3] = [Point::new(0.0, 0.0), Point::new(1.0, 0.0), Point::new(2.0, 0.0)];
/// let fit = try_fit(&points, TotalLeastSquares, AllPoints)?;
/// assert_eq!((fit.used_count(), fit.outlier_count()), (3, 0));
/// # Ok::<(), fovea::Error>(())
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AllPoints;

/// Huber's weights: a point within the threshold of the element counts
/// fully, one further out with a weight that falls as `threshold / distance`.
///
/// No point is dropped, so every point is used, and a point beyond the
/// threshold counts as an outlier while still pulling the element a
/// little. The fit is reweighted until the weights settle. Gentler than
/// [`Tukey`], and the start [`Tukey`] begins from.
///
/// Same construction discipline as the crate's other parameter types:
/// [`huber!`](crate::huber) for literals, [`Huber::try_new`] for computed
/// values, and [`Huber::new`] as the checked `const fn` under the macro.
///
/// # Example
///
/// ```
/// use fovea::{Length, Pixels};
/// use fovea::measure::Huber;
///
/// let h: Huber<Pixels> = fovea::huber!(0.5);
/// assert_eq!(h.threshold().get(), 0.5);
/// assert!(Huber::<Pixels>::try_new(Length::new(0.0)).is_err());
/// ```
pub struct Huber<U> {
    threshold: Length<U>,
}

/// Tukey's biweight: a point within the threshold of the element counts
/// with a weight that falls smoothly from one to zero at the threshold, and
/// a point at or beyond it does not shape the element at all.
///
/// Such a point is an outlier and is not used, so a chip or a burr on an
/// edge does not pull the element. Because the biweight can settle on a
/// poor element when the start is far off, the fit first runs [`Huber`]
/// with the same threshold and reweights from there until the weights
/// settle. The figures that judge the part, the maximum residual and the
/// peak to valley, still include the outliers; see [`Fit`].
///
/// # Example
///
/// ```
/// use fovea::{Length, Pixels};
/// use fovea::measure::Tukey;
///
/// let t: Tukey<Pixels> = fovea::tukey!(0.5);
/// assert_eq!(t.threshold().get(), 0.5);
/// assert!(Tukey::<Pixels>::try_new(Length::new(f64::NAN)).is_err());
/// ```
///
/// A threshold in millimetres does not apply to points in pixels:
///
/// ```compile_fail
/// use fovea::{Millimeter, Pixels, Point};
/// use fovea::measure::{Taubin, Tukey, try_fit};
///
/// let points: Vec<Point<Pixels>> = Vec::new();
/// let rule: Tukey<Millimeter> = fovea::tukey!(0.01);
/// let _ = try_fit(&points, Taubin, rule);
/// ```
pub struct Tukey<U> {
    threshold: Length<U>,
}

macro_rules! threshold_rule {
    ($($ty:ident, $name:literal, $variant:ident;)*) => {$(
        impl<U: LengthUnit> $ty<U> {
            #[doc = concat!("Creates a `", stringify!($ty), "` rule, returning `None` unless")]
            /// `threshold` is finite and positive.
            #[must_use]
            pub const fn new(threshold: Length<U>) -> Option<Self> {
                let t = threshold.get();
                if t.is_finite() && t > 0.0 {
                    Some(Self { threshold })
                } else {
                    None
                }
            }

            #[doc = concat!("Creates a `", stringify!($ty), "` rule from a computed threshold,")]
            /// validating it.
            ///
            /// # Errors
            ///
            /// [`Error::InvalidParameter`] if `threshold` is zero, negative,
            /// NaN or infinite.
            pub fn try_new(threshold: Length<U>) -> Result<Self, Error> {
                Self::new(threshold).ok_or_else(|| {
                    ParameterError::new(
                        $name,
                        Requirement::FinitePositive,
                        Value::F64(threshold.get()),
                    )
                    .into()
                })
            }

            /// The threshold, in the unit of the points.
            #[must_use]
            pub const fn threshold(&self) -> Length<U> {
                self.threshold
            }
        }

        impl<U> Clone for $ty<U> {
            #[inline]
            fn clone(&self) -> Self {
                *self
            }
        }
        impl<U> Copy for $ty<U> {}
        impl<U> PartialEq for $ty<U> {
            fn eq(&self, other: &Self) -> bool {
                self.threshold == other.threshold
            }
        }
        impl<U> core::fmt::Debug for $ty<U> {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.debug_tuple(stringify!($ty)).field(&self.threshold).finish()
            }
        }

        impl<U: LengthUnit> sealed::Rule for $ty<U> {
            fn weighting(&self) -> Weighting {
                Weighting::$variant(self.threshold.get())
            }
        }
        impl<U: LengthUnit> OutlierRule<U> for $ty<U> {}
    )*};
}

threshold_rule! {
    Huber, "Huber threshold", Huber;
    Tukey, "Tukey threshold", Tukey;
}

impl sealed::Rule for AllPoints {
    fn weighting(&self) -> Weighting {
        Weighting::AllPoints
    }
}
impl<U: LengthUnit> OutlierRule<U> for AllPoints {}

/// A [`Huber`] literal, checked at compile time; the unit follows from the
/// context.
///
/// A value that is not a constant expression does not compile
/// (`error[E0435]`); use [`Huber::try_new`](crate::measure::Huber::try_new)
/// there.
///
/// # Example
///
/// ```
/// use fovea::Millimeter;
/// use fovea::measure::Huber;
///
/// let rule: Huber<Millimeter> = fovea::huber!(0.02);
/// assert_eq!(rule.threshold().get(), 0.02);
/// ```
///
/// ```compile_fail
/// // ERROR: evaluation panicked: must be finite and strictly positive
/// let _: fovea::measure::Huber<fovea::Pixels> = fovea::huber!(-1.0);
/// ```
#[macro_export]
macro_rules! huber {
    ($value:expr) => {
        const {
            $crate::measure::Huber::new($crate::Length::new($value))
                .expect($crate::error::Requirement::FinitePositive.text())
        }
    };
}

/// A [`Tukey`] literal, checked at compile time; the unit follows from the
/// context.
///
/// A value that is not a constant expression does not compile
/// (`error[E0435]`); use [`Tukey::try_new`](crate::measure::Tukey::try_new)
/// there.
///
/// # Example
///
/// ```
/// use fovea::Pixels;
/// use fovea::measure::Tukey;
///
/// let rule: Tukey<Pixels> = fovea::tukey!(0.5);
/// assert_eq!(rule.threshold().get(), 0.5);
/// ```
///
/// ```compile_fail
/// // ERROR: evaluation panicked: must be finite and strictly positive
/// let _: fovea::measure::Tukey<fovea::Pixels> = fovea::tukey!(0.0);
/// ```
#[macro_export]
macro_rules! tukey {
    ($value:expr) => {
        const {
            $crate::measure::Tukey::new($crate::Length::new($value))
                .expect($crate::error::Requirement::FinitePositive.text())
        }
    };
}

/// A fitted element and how well it fits: the result of [`try_fit`].
///
/// Residuals are geometric distances from the points to the element, exact
/// for every element, the ellipse included, in the unit of the points. The
/// figures answer two questions, and each runs over the points that answer
/// it:
///
/// - **Does the element fit the edge?** [`rms_residual`](Self::rms_residual),
///   over the points used, the ones that shaped the element.
/// - **Is the part good?** [`max_residual`](Self::max_residual) and
///   [`peak_to_valley`](Self::peak_to_valley), over all points, outliers
///   included, so the outlier handling chooses the element but never hides
///   a defect from the figures that judge the part.
///
/// Under [`AllPoints`] both sets are all points.
///
/// A fit in pixels converts to world units with `map` through a
/// [`ConformalMap`] only, because
/// under any other calibration the distances in the image are not
/// proportional to those on the part. There, convert the points first and
/// fit in world units.
///
/// # Example
///
/// ```
/// use fovea::{Pixels, Point};
/// use fovea::measure::{Taubin, try_fit};
///
/// // A hole of radius 10 px with a chip: three points 2 px inside.
/// let points: Vec<Point<Pixels>> = (0..360)
///     .map(|k| {
///         let t = (k as f64).to_radians();
///         let r = if (100..103).contains(&k) { 8.0 } else { 10.0 };
///         Point::new(50.0 + r * t.cos(), 40.0 + r * t.sin())
///     })
///     .collect();
///
/// let fit = try_fit(&points, Taubin, fovea::tukey!(0.5))?;
/// assert_eq!(fit.outlier_count(), 3);
/// // The circle fits the good edge exactly...
/// assert!(fit.rms_residual().get() < 1e-9);
/// assert!((fit.element().radius().get() - 10.0).abs() < 1e-9);
/// // ...and the chip still shows where the part is judged.
/// assert!((fit.max_residual().get() - 2.0).abs() < 1e-9);
/// # Ok::<(), fovea::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fit<T> {
    element: T,
    rms_residual: f64,
    max_residual: f64,
    peak_to_valley: f64,
    used_count: usize,
    outlier_count: usize,
    extent: [f64; 2],
}

impl<T: Element> Fit<T> {
    /// The fitted element.
    #[must_use]
    pub fn element(&self) -> &T {
        &self.element
    }

    /// The root mean square of the distances of the points used from the
    /// element: how well the element describes the points that shaped it.
    #[doc(alias = "rms")]
    #[doc(alias = "rmse")]
    #[must_use]
    pub fn rms_residual(&self) -> Length<T::Unit> {
        Length::new(self.rms_residual)
    }

    /// The largest distance of any point from the element, outliers
    /// included.
    ///
    /// This is **not** the form deviation: one point 0.3 outside and one 0.2
    /// inside give a maximum residual of 0.3 and a form deviation of 0.5.
    /// See [`peak_to_valley`](Self::peak_to_valley).
    #[must_use]
    pub fn max_residual(&self) -> Length<T::Unit> {
        Length::new(self.max_residual)
    }

    /// The form deviation relative to the fitted element: the distance of
    /// the furthest point on one side plus that of the furthest point on the
    /// other, over all points, outliers included.
    ///
    /// For a circle this is the peak-to-valley roundness deviation as the
    /// roundness standards define it, measured from the fitted circle; for
    /// a line, the straightness measured from the fitted line. The
    /// standards' minimum zone reference is chosen to make exactly this
    /// figure as small as possible, so a tolerance evaluated against it can
    /// give a smaller value, never a larger one.
    #[must_use]
    pub fn peak_to_valley(&self) -> Length<T::Unit> {
        Length::new(self.peak_to_valley)
    }

    /// The number of points that shaped the element: every point, except
    /// under [`Tukey`], where a point at or beyond the threshold is not
    /// used.
    #[must_use]
    pub fn used_count(&self) -> usize {
        self.used_count
    }

    /// The number of points further from the element than the threshold of
    /// [`Huber`] or [`Tukey`]; zero under [`AllPoints`].
    ///
    /// Under [`Huber`] an outlier is still used, with a reduced weight, so
    /// it counts in both figures and their sum can exceed the number of
    /// points.
    #[must_use]
    pub fn outlier_count(&self) -> usize {
        self.outlier_count
    }

    fn mapped<S: Element>(&self, element: S, factor: f64) -> Fit<S> {
        Fit {
            element,
            rms_residual: self.rms_residual * factor,
            max_residual: self.max_residual * factor,
            peak_to_valley: self.peak_to_valley * factor,
            used_count: self.used_count,
            outlier_count: self.outlier_count,
            extent: [self.extent[0] * factor, self.extent[1] * factor],
        }
    }
}

impl<U: LengthUnit> Fit<Line<U>> {
    /// The part of the line the points used cover: from the foot of the
    /// first along the line's direction to the foot of the last.
    ///
    /// The line itself is infinite, so two fitted sides meet at the corner
    /// a drawing dimensions even where a chamfer stops the edge points
    /// short of it; the extent is for drawing the line and for the length
    /// of the measured edge.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::{Pixels, Point};
    /// use fovea::measure::{AllPoints, TotalLeastSquares, try_fit};
    ///
    /// let points: Vec<Point<Pixels>> = (2..=8).map(|k| Point::new(k as f64, 5.0)).collect();
    /// let fit = try_fit(&points, TotalLeastSquares, AllPoints)?;
    /// let extent = fit.extent();
    /// assert_eq!((extent.start, extent.end), (Point::new(2.0, 5.0), Point::new(8.0, 5.0)));
    /// # Ok::<(), fovea::Error>(())
    /// ```
    #[must_use]
    pub fn extent(&self) -> Segment<U> {
        let line = &self.element;
        let at = |t: f64| line.point() + line.direction() * t;
        Segment::new(at(self.extent[0]), at(self.extent[1]))
    }

    /// The fit in the codomain of a conformal mapping: the line, its extent
    /// and every residual figure convert, since every distance scales by the
    /// mapping's factor.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::{Millimeter, Pixels, Point};
    /// use fovea::geometry::UniformScale;
    /// use fovea::measure::{AllPoints, TotalLeastSquares, try_fit};
    ///
    /// let points: [Point<Pixels>; 3] = [Point::new(0.0, 0.0), Point::new(40.0, 1.0), Point::new(80.0, 0.0)];
    /// let fit = try_fit(&points, TotalLeastSquares, AllPoints)?;
    /// let scale: UniformScale<Pixels, Millimeter> = fovea::uniform_scale!(0.025);
    /// let in_mm = fit.map(&scale);
    /// assert!((in_mm.peak_to_valley().get() - fit.peak_to_valley().get() * 0.025).abs() < 1e-15);
    /// # Ok::<(), fovea::Error>(())
    /// ```
    #[must_use]
    pub fn map<M: ConformalMap<Domain = U>>(&self, mapping: &M) -> Fit<Line<M::Codomain>> {
        self.mapped(mapping.map_line(self.element), mapping.factor())
    }
}

impl<U: LengthUnit> Fit<Circle<U>> {
    /// The fit in the codomain of a conformal mapping: the circle and every
    /// residual figure convert, since every distance scales by the mapping's
    /// factor.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::{Millimeter, Pixels, Point};
    /// use fovea::geometry::UniformScale;
    /// use fovea::measure::{AllPoints, Taubin, try_fit};
    ///
    /// let points: [Point<Pixels>; 3] = [Point::new(160.0, 0.0), Point::new(0.0, 160.0), Point::new(-160.0, 0.0)];
    /// let fit = try_fit(&points, Taubin, AllPoints)?;
    /// let scale: UniformScale<Pixels, Millimeter> = fovea::uniform_scale!(0.0125);
    /// assert!((fit.map(&scale).element().radius().get() - 2.0).abs() < 1e-12);
    /// # Ok::<(), fovea::Error>(())
    /// ```
    ///
    /// An axis scale is not conformal, so a fitted circle does not convert
    /// through it; convert the points and fit in the world:
    ///
    /// ```compile_fail
    /// use fovea::{Millimeter, Pixels, Point};
    /// use fovea::geometry::AxisScale;
    /// use fovea::measure::{AllPoints, Taubin, try_fit};
    ///
    /// let points: [Point<Pixels>; 3] = [Point::new(1.0, 0.0), Point::new(0.0, 1.0), Point::new(-1.0, 0.0)];
    /// let fit = try_fit(&points, Taubin, AllPoints).unwrap();
    /// let scale: AxisScale<Pixels, Millimeter> = fovea::axis_scale!(0.02, 0.05);
    /// let _ = fit.map(&scale);
    /// ```
    #[must_use]
    pub fn map<M: ConformalMap<Domain = U>>(&self, mapping: &M) -> Fit<Circle<M::Codomain>> {
        self.mapped(mapping.map_circle(self.element), mapping.factor())
    }
}

impl<U: LengthUnit> Fit<Ellipse<U>> {
    /// The fit in the codomain of a conformal mapping: the ellipse and every
    /// residual figure convert, since every distance scales by the mapping's
    /// factor.
    #[must_use]
    pub fn map<M: ConformalMap<Domain = U>>(&self, mapping: &M) -> Fit<Ellipse<M::Codomain>> {
        self.mapped(mapping.map_ellipse(self.element), mapping.factor())
    }
}

/// Fits an element to `points` with `estimator`, treating points far from
/// it by `outliers`.
///
/// The element is in the unit of the points, and so are its residuals: fit
/// in pixels when the calibration is conformal and convert the result with
/// `Fit::map`, and convert the points first under any other calibration.
///
/// The fit is deterministic. Under [`Huber`] and [`Tukey`] it reweights
/// until no weight changes by more than 10⁻¹² between two rounds, or for at
/// most 100 rounds, and the figures are those of the element it returns.
///
/// # Errors
///
/// - [`Error::InvalidParameter`] if a point has a coordinate that is not
///   finite; its index is recorded.
/// - [`Error::TooFewPoints`] if there are fewer points than
///   [`Estimator::MIN_POINTS`], or if [`Tukey`] leaves fewer used.
/// - [`Error::DegeneratePoints`] if the points determine no unique element.
///
/// # Example
///
/// ```
/// use fovea::{Pixels, Point};
/// use fovea::measure::{AllPoints, TotalLeastSquares, try_fit};
///
/// // An edge that wanders a quarter pixel either side of y = 0.
/// let points: [Point<Pixels>; 4] = [
///     Point::new(0.0, 0.25), Point::new(1.0, -0.25), Point::new(2.0, -0.25), Point::new(3.0, 0.25),
/// ];
/// let fit = try_fit(&points, TotalLeastSquares, AllPoints)?;
/// assert_eq!(fit.max_residual().get(), 0.25);
/// assert_eq!(fit.peak_to_valley().get(), 0.5);
///
/// // A single point determines no line.
/// assert!(try_fit(&points[..1], TotalLeastSquares, AllPoints).is_err());
/// # Ok::<(), fovea::Error>(())
/// ```
pub fn try_fit<U, E, O>(
    points: &[Point<U>],
    estimator: E,
    outliers: O,
) -> Result<Fit<E::Element<U>>, Error>
where
    U: LengthUnit,
    E: Estimator,
    O: OutlierRule<U>,
{
    if let Some(i) = points
        .iter()
        .position(|p| !(p.x.is_finite() && p.y.is_finite()))
    {
        let p = points[i];
        return Err(
            ParameterError::new("point", Requirement::Finite, Value::F64Pair(p.x, p.y))
                .at(i)
                .into(),
        );
    }
    if points.len() < E::MIN_POINTS {
        return Err(Error::TooFewPoints {
            required: E::MIN_POINTS,
            actual: points.len(),
        });
    }

    let mut weights = vec![1.0; points.len()];
    let mut element = estimator.fit_weighted(points, &weights)?;
    let weighting = outliers.weighting();
    match weighting {
        Weighting::AllPoints => {}
        Weighting::Huber(c) => {
            element = reweight(estimator, points, element, &mut weights, |r| huber(r, c))?;
        }
        Weighting::Tukey(c) => {
            element = reweight(estimator, points, element, &mut weights, |r| huber(r, c))?;
            element = reweight(estimator, points, element, &mut weights, |r| tukey(r, c))?;
        }
    }

    let residuals: Vec<f64> = points
        .iter()
        .map(|p| E::signed_residual(&element, *p))
        .collect();
    let used: Vec<bool> = residuals
        .iter()
        .map(|r| match weighting {
            Weighting::Tukey(c) => r.abs() < c,
            Weighting::AllPoints | Weighting::Huber(_) => true,
        })
        .collect();
    let used_count = used.iter().filter(|u| **u).count();
    if used_count < E::MIN_POINTS {
        return Err(Error::TooFewPoints {
            required: E::MIN_POINTS,
            actual: used_count,
        });
    }
    let outlier_count = residuals
        .iter()
        .filter(|r| match weighting {
            Weighting::AllPoints => false,
            Weighting::Huber(c) => r.abs() > c,
            Weighting::Tukey(c) => r.abs() >= c,
        })
        .count();

    let sum_sq: f64 = residuals
        .iter()
        .zip(&used)
        .filter(|(_, u)| **u)
        .map(|(r, _)| r * r)
        .sum();
    let rms_residual = (sum_sq / used_count as f64).sqrt();
    let (mut max_residual, mut above, mut below) = (0.0f64, 0.0f64, 0.0f64);
    for &r in &residuals {
        max_residual = max_residual.max(r.abs());
        above = above.max(r);
        below = below.max(-r);
    }

    let (element, extent) = E::finish(element, points, &used);
    Ok(Fit {
        element,
        rms_residual,
        max_residual,
        peak_to_valley: above + below,
        used_count,
        outlier_count,
        extent,
    })
}

/// Rounds of reweighting before the fit returns what it has.
const MAX_ROUNDS: usize = 100;

/// The largest change of any weight between two rounds that counts as
/// settled.
const WEIGHT_TOLERANCE: f64 = 1e-12;

fn huber(r: f64, c: f64) -> f64 {
    if r <= c { 1.0 } else { c / r }
}

fn tukey(r: f64, c: f64) -> f64 {
    if r >= c {
        0.0
    } else {
        let u = r / c;
        (1.0 - u * u) * (1.0 - u * u)
    }
}

/// Iteratively reweighted least squares: weights from the distances to the
/// current element, then a weighted fit, until the weights settle.
fn reweight<U, E>(
    estimator: E,
    points: &[Point<U>],
    mut element: E::Element<U>,
    weights: &mut [f64],
    weight: impl Fn(f64) -> f64,
) -> Result<E::Element<U>, Error>
where
    U: LengthUnit,
    E: Estimator,
{
    for _ in 0..MAX_ROUNDS {
        let mut change = 0.0f64;
        let mut used = 0;
        for (p, w) in points.iter().zip(weights.iter_mut()) {
            let next = weight(E::signed_residual(&element, *p).abs());
            change = change.max((next - *w).abs());
            used += usize::from(next > 0.0);
            *w = next;
        }
        if used < E::MIN_POINTS {
            return Err(Error::TooFewPoints {
                required: E::MIN_POINTS,
                actual: used,
            });
        }
        element = estimator.fit_weighted(points, weights)?;
        if change <= WEIGHT_TOLERANCE {
            break;
        }
    }
    Ok(element)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AxialOrientation;
    use crate::geometry::{Millimeter, Pixels, Similarity, UniformScale, Vector};
    use core::f64::consts::{FRAC_PI_2, PI, TAU};

    fn px(x: f64, y: f64) -> Point<Pixels> {
        Point::new(x, y)
    }

    /// Deterministic Gaussian noise: xorshift and Box-Muller.
    struct Noise(u64);

    impl Noise {
        fn uniform(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }

        fn gauss(&mut self, sd: f64) -> f64 {
            let (u, v) = (self.uniform(), self.uniform());
            sd * (-2.0 * u.ln()).sqrt() * (TAU * v).cos()
        }
    }

    fn circle_points(cx: f64, cy: f64, r: f64, n: usize, from: f64, to: f64) -> Vec<Point<Pixels>> {
        (0..n)
            .map(|k| {
                let t = from + (to - from) * k as f64 / n as f64;
                px(cx + r * t.cos(), cy + r * t.sin())
            })
            .collect()
    }

    fn ellipse_point(cx: f64, cy: f64, a: f64, b: f64, phi: f64, t: f64) -> Point<Pixels> {
        let (sin, cos) = phi.sin_cos();
        let (u, v) = (a * t.cos(), b * t.sin());
        px(cx + u * cos - v * sin, cy + u * sin + v * cos)
    }

    // ── Line ────────────────────────────────────────────────────────────

    #[test]
    fn a_line_fits_exactly_whatever_its_direction() {
        for angle in [0.0, 0.3, FRAC_PI_2, 2.0, -1.2] {
            let (sin, cos) = f64::sin_cos(angle);
            let points: Vec<_> = (0..10)
                .map(|k| px(3.0 + k as f64 * cos, -7.0 + k as f64 * sin))
                .collect();
            let fit = try_fit(&points, TotalLeastSquares, AllPoints).unwrap();
            assert!(fit.max_residual().get() < 1e-12, "angle {angle}");
            // The direction runs from the first point towards the last.
            let d = fit.element().direction();
            assert!(
                (d.x - cos).abs() < 1e-12 && (d.y - sin).abs() < 1e-12,
                "angle {angle}"
            );
        }
    }

    #[test]
    fn a_line_points_from_its_first_input_to_its_last() {
        let forward: Vec<_> = (0..5).map(|k| px(k as f64, 2.0)).collect();
        let backward: Vec<_> = forward.iter().rev().copied().collect();
        let f = try_fit(&forward, TotalLeastSquares, AllPoints).unwrap();
        let b = try_fit(&backward, TotalLeastSquares, AllPoints).unwrap();
        assert_eq!(f.element().direction(), Vector::new(1.0, 0.0));
        assert_eq!(b.element().direction(), Vector::new(-1.0, 0.0));
        assert_eq!(b.extent().start, px(4.0, 2.0));
        assert_eq!(b.extent().end, px(0.0, 2.0));
    }

    #[test]
    fn the_extent_covers_the_points_used_only() {
        // A stray point far along the line, and off it: an outlier under
        // Tukey, so it neither shapes the line nor lengthens the extent.
        let mut points: Vec<_> = (0..20).map(|k| px(k as f64, 10.0)).collect();
        points.push(px(60.0, 13.0));
        let fit = try_fit(&points, TotalLeastSquares, crate::tukey!(0.5)).unwrap();
        assert_eq!((fit.used_count(), fit.outlier_count()), (20, 1));
        let extent = fit.extent();
        assert!(extent.start.distance(px(0.0, 10.0)).get() < 1e-9);
        assert!(extent.end.distance(px(19.0, 10.0)).get() < 1e-9);
        assert!((extent.length().get() - 19.0).abs() < 1e-9);
        assert!((fit.max_residual().get() - 3.0).abs() < 1e-9);
    }

    #[test]
    fn two_sides_of_a_chamfered_corner_meet_beyond_their_points() {
        // The top side ends at x = 8, the right side starts at y = 3; the
        // corner the drawing dimensions is (11, 0), outside both extents.
        let top: Vec<_> = (0..=8).map(|k| px(k as f64, 0.0)).collect();
        let right: Vec<_> = (3..=10).map(|k| px(11.0, k as f64)).collect();
        let top = try_fit(&top, TotalLeastSquares, AllPoints).unwrap();
        let right = try_fit(&right, TotalLeastSquares, AllPoints).unwrap();
        // Where the infinite lines meet: solve p + s·d = q + t·e.
        let (p, d) = (top.element().point(), top.element().direction());
        let (q, e) = (right.element().point(), right.element().direction());
        let w = q - p;
        let s = (w.x * e.y - w.y * e.x) / (d.x * e.y - d.y * e.x);
        let corner = p + d * s;
        assert!(corner.distance(px(11.0, 0.0)).get() < 1e-12);
        assert!(top.extent().end.x < corner.x && right.extent().start.y > corner.y);
    }

    // ── Errors ──────────────────────────────────────────────────────────

    #[test]
    fn too_few_points_are_reported_with_the_minimum() {
        let points = circle_points(0.0, 0.0, 1.0, 4, 0.0, TAU);
        assert_eq!(
            try_fit(&points[..1], TotalLeastSquares, AllPoints).unwrap_err(),
            Error::TooFewPoints {
                required: 2,
                actual: 1
            }
        );
        assert_eq!(
            try_fit(&points[..2], Taubin, AllPoints).unwrap_err(),
            Error::TooFewPoints {
                required: 3,
                actual: 2
            }
        );
        assert_eq!(
            try_fit(&points, Fitzgibbon, AllPoints).unwrap_err(),
            Error::TooFewPoints {
                required: 5,
                actual: 4
            }
        );
        assert_eq!(TotalLeastSquares::MIN_POINTS, 2);
        assert_eq!(Taubin::MIN_POINTS, 3);
        assert_eq!(Fitzgibbon::MIN_POINTS, 5);
    }

    #[test]
    fn a_point_that_is_not_finite_is_named_by_its_index() {
        let points = [px(0.0, 0.0), px(1.0, 1.0), px(f64::NAN, 2.0), px(3.0, 3.0)];
        let Err(Error::InvalidParameter(e)) = try_fit(&points, TotalLeastSquares, AllPoints) else {
            panic!("a NaN point was accepted");
        };
        assert_eq!(e.requirement(), Requirement::Finite);
        assert_eq!(e.index(), Some(2));
        assert_eq!(e.value(), Value::F64Pair(f64::NAN, 2.0));
    }

    #[test]
    fn degenerate_points_are_an_error() {
        let same = [px(2.0, 3.0); 6];
        assert_eq!(
            try_fit(&same, TotalLeastSquares, AllPoints).unwrap_err(),
            Error::DegeneratePoints
        );
        // Spread equally in every direction: no line is better than another.
        let square = [px(1.0, 1.0), px(-1.0, 1.0), px(-1.0, -1.0), px(1.0, -1.0)];
        assert_eq!(
            try_fit(&square, TotalLeastSquares, AllPoints).unwrap_err(),
            Error::DegeneratePoints
        );
        let collinear: Vec<_> = (0..8).map(|k| px(k as f64, 2.0 * k as f64 + 1.0)).collect();
        assert_eq!(
            try_fit(&collinear, Taubin, AllPoints).unwrap_err(),
            Error::DegeneratePoints
        );
        assert_eq!(
            try_fit(&collinear, Fitzgibbon, AllPoints).unwrap_err(),
            Error::DegeneratePoints
        );
        assert_eq!(
            try_fit(&same, Fitzgibbon, AllPoints).unwrap_err(),
            Error::DegeneratePoints
        );
    }

    // ── Circle ──────────────────────────────────────────────────────────

    #[test]
    fn a_circle_fits_exactly_through_three_points_and_far_from_the_origin() {
        let three = [px(1.0, 0.0), px(0.0, 1.0), px(-1.0, 0.0)];
        let fit = try_fit(&three, Taubin, AllPoints).unwrap();
        assert!(fit.element().center().distance(px(0.0, 0.0)).get() < 1e-14);
        assert!((fit.element().radius().get() - 1.0).abs() < 1e-14);

        for (cx, cy, r) in [(1.0e4, -2.0e4, 3.0), (0.5, 0.25, 1.0e3), (-7.0, 4.0, 0.01)] {
            let points = circle_points(cx, cy, r, 9, 0.0, TAU);
            let fit = try_fit(&points, Taubin, AllPoints).unwrap();
            let c = fit.element();
            assert!(
                c.center().distance(px(cx, cy)).get() < 1e-9 * r.max(1.0),
                "{c:?}"
            );
            assert!((c.radius().get() - r).abs() < 1e-9 * r.max(1.0), "{c:?}");
        }
    }

    #[test]
    fn a_circle_from_a_noisy_short_arc_stays_close() {
        let mut noise = Noise(0x9e37_79b9_7f4a_7c15);
        let points: Vec<_> = circle_points(10.0, -4.0, 70.0, 200, 0.0, 1.5)
            .into_iter()
            .map(|p| px(p.x + noise.gauss(0.2), p.y + noise.gauss(0.2)))
            .collect();
        let fit = try_fit(&points, Taubin, AllPoints).unwrap();
        // A test criterion, not a promise: a fifth of a pixel of noise on a
        // quarter turn of a 70 px circle.
        assert!(fit.element().center().distance(px(10.0, -4.0)).get() < 0.5);
        assert!((fit.element().radius().get() - 70.0).abs() < 0.5);
        assert!((fit.rms_residual().get() - 0.2).abs() < 0.05);
    }

    #[test]
    fn peak_to_valley_is_the_furthest_outside_plus_the_furthest_inside() {
        // Alternating radii 10.3 and 9.8 about the origin, symmetric, so
        // the fitted centre is the origin.
        let points: Vec<_> = (0..16)
            .map(|k| {
                let t = TAU * k as f64 / 16.0;
                let r = if k % 2 == 0 { 10.3 } else { 9.8 };
                px(r * t.cos(), r * t.sin())
            })
            .collect();
        let fit = try_fit(&points, Taubin, AllPoints).unwrap();
        assert!(fit.element().center().distance(px(0.0, 0.0)).get() < 1e-12);
        let r = fit.element().radius().get();
        assert!((fit.peak_to_valley().get() - 0.5).abs() < 1e-12);
        let expected_max = (10.3 - r).max(r - 9.8);
        assert!((fit.max_residual().get() - expected_max).abs() < 1e-12);
        assert!(fit.max_residual() < fit.peak_to_valley());
    }

    #[test]
    fn a_chip_is_rejected_by_tukey_and_still_judges_the_part() {
        let mut noise = Noise(0x0123_4567_89ab_cdef);
        let points: Vec<_> = (0..1000)
            .map(|k| {
                let t = TAU * k as f64 / 1000.0;
                let r = if (100..103).contains(&k) { 98.0 } else { 100.0 } + noise.gauss(0.05);
                px(200.0 + r * t.cos(), 150.0 + r * t.sin())
            })
            .collect();
        let fit = try_fit(&points, Taubin, crate::tukey!(0.5)).unwrap();
        assert_eq!((fit.used_count(), fit.outlier_count()), (997, 3));
        assert!(fit.element().center().distance(px(200.0, 150.0)).get() < 0.01);
        assert!((fit.rms_residual().get() - 0.05).abs() < 0.01);
        assert!((fit.max_residual().get() - 2.0).abs() < 0.2);
        assert!(fit.peak_to_valley().get() > 2.0 && fit.peak_to_valley().get() < 2.5);

        // Huber keeps every point and counts the chip as outliers.
        let fit = try_fit(&points, Taubin, crate::huber!(0.5)).unwrap();
        assert_eq!((fit.used_count(), fit.outlier_count()), (1000, 3));

        // All points: the chip pulls the circle, and nothing is an outlier.
        let fit = try_fit(&points, Taubin, AllPoints).unwrap();
        assert_eq!((fit.used_count(), fit.outlier_count()), (1000, 0));
    }

    #[test]
    fn reweighting_that_drops_every_point_reports_too_few_points() {
        // Huber's start passes close to some points even far below the
        // noise, as a least absolute deviation fit does, so Tukey rarely
        // drops them all; the path is tested on the reweighting itself.
        let points: Vec<_> = (0..20)
            .map(|k| px(k as f64, if k % 2 == 0 { 1.0 } else { -1.0 }))
            .collect();
        let mut weights = vec![1.0; points.len()];
        let start = fit_line(&points, &weights).unwrap();
        let result = reweight(TotalLeastSquares, &points, start, &mut weights, |_| 0.0);
        assert_eq!(
            result.unwrap_err(),
            Error::TooFewPoints {
                required: 2,
                actual: 0
            }
        );

        // On data with a clear majority a tiny threshold keeps exactly the
        // points the start passes through, and says so.
        let mut noise = Noise(42);
        let noisy: Vec<_> = circle_points(0.0, 0.0, 50.0, 40, 0.0, TAU)
            .into_iter()
            .map(|p| px(p.x + noise.gauss(1.0), p.y + noise.gauss(1.0)))
            .collect();
        let rule = Tukey::<Pixels>::try_new(Length::new(1e-6)).unwrap();
        let fit = try_fit(&noisy, Taubin, rule).unwrap();
        assert_eq!(fit.used_count() + fit.outlier_count(), noisy.len());
        assert!(fit.used_count() >= Taubin::MIN_POINTS);
    }

    // ── Ellipse ─────────────────────────────────────────────────────────

    #[test]
    fn an_ellipse_fits_exactly_in_every_orientation() {
        for phi in [0.0, 0.6, FRAC_PI_2, -1.3, PI - 0.01] {
            let points: Vec<_> = (0..12)
                .map(|k| ellipse_point(50.0, 30.0, 20.0, 8.0, phi, TAU * k as f64 / 12.0))
                .collect();
            let fit = try_fit(&points, Fitzgibbon, AllPoints).unwrap();
            let e = fit.element();
            assert!(
                e.center().distance(px(50.0, 30.0)).get() < 1e-9,
                "{phi}: {e:?}"
            );
            assert!((e.semi_major().get() - 20.0).abs() < 1e-9, "{phi}: {e:?}");
            assert!((e.semi_minor().get() - 8.0).abs() < 1e-9, "{phi}: {e:?}");
            let expected = AxialOrientation::from_radians(phi).unwrap();
            assert!(
                e.orientation().signed_difference(expected).abs() < 1e-9,
                "{phi}: {e:?}"
            );
            assert!(fit.max_residual().get() < 1e-9);
        }
    }

    #[test]
    fn an_ellipse_fits_five_points_and_a_circle() {
        let five: Vec<_> = (0..5)
            .map(|k| ellipse_point(-3.0, 7.0, 12.0, 4.0, 0.4, 1.1 * k as f64))
            .collect();
        let fit = try_fit(&five, Fitzgibbon, AllPoints).unwrap();
        assert!((fit.element().semi_major().get() - 12.0).abs() < 1e-8);
        assert!((fit.element().semi_minor().get() - 4.0).abs() < 1e-8);

        let round = circle_points(5.0, 5.0, 9.0, 24, 0.0, TAU);
        let fit = try_fit(&round, Fitzgibbon, AllPoints).unwrap();
        assert!((fit.element().semi_major().get() - 9.0).abs() < 1e-9);
        assert!((fit.element().semi_minor().get() - 9.0).abs() < 1e-9);
    }

    #[test]
    fn the_ellipse_residuals_are_exact_distances() {
        // An elongated ellipse and one point moved 5 px along the outward
        // normal where a first-order approximation is furthest off. Tukey
        // keeps the exact points, so the element is exact, and the
        // maximum residual is the displacement itself.
        let (a, b, phi) = (100.0, 20.0, 0.0);
        let mut points: Vec<_> = (0..60)
            .map(|k| ellipse_point(0.0, 0.0, a, b, phi, TAU * k as f64 / 60.0))
            .collect();
        let t = 12f64.to_radians();
        let on = ellipse_point(0.0, 0.0, a, b, phi, t);
        let normal = Vector::<Pixels>::new(b * t.cos(), a * t.sin());
        let normal = normal * (1.0 / normal.length().get());
        points.push(on + normal * 5.0);
        let fit = try_fit(&points, Fitzgibbon, crate::tukey!(0.5)).unwrap();
        assert_eq!(fit.outlier_count(), 1);
        assert!((fit.max_residual().get() - 5.0).abs() < 1e-6);
    }

    // ── Parameters and mappings ─────────────────────────────────────────

    #[test]
    fn thresholds_are_finite_and_positive() {
        assert_eq!(
            Tukey::<Pixels>::new(Length::new(0.5)).map(|t| t.threshold().get()),
            Some(0.5)
        );
        for bad in [0.0, -0.5, f64::NAN, f64::INFINITY] {
            assert!(Tukey::<Pixels>::new(Length::new(bad)).is_none());
            assert!(Huber::<Pixels>::new(Length::new(bad)).is_none());
            let Err(Error::InvalidParameter(e)) = Tukey::<Pixels>::try_new(Length::new(bad)) else {
                panic!("{bad} accepted");
            };
            assert_eq!(e.parameter(), "Tukey threshold");
            assert_eq!(e.requirement(), Requirement::FinitePositive);
            let Err(Error::InvalidParameter(e)) = Huber::<Pixels>::try_new(Length::new(bad)) else {
                panic!("{bad} accepted");
            };
            assert_eq!(e.parameter(), "Huber threshold");
        }
        let h: Huber<Millimeter> = crate::huber!(0.02);
        assert_eq!(format!("{h:?}"), "Huber(Length(0.02))");
        assert_eq!(h, h.clone());
    }

    #[test]
    fn a_fit_converts_through_a_conformal_mapping() {
        let points: Vec<_> = (0..16)
            .map(|k| {
                let t = TAU * k as f64 / 16.0;
                let r = if k % 2 == 0 { 10.3 } else { 9.8 };
                px(40.0 + r * t.cos(), 20.0 + r * t.sin())
            })
            .collect();
        let fit = try_fit(&points, Taubin, AllPoints).unwrap();
        let scale: UniformScale<Pixels, Millimeter> = crate::uniform_scale!(0.0125);
        let mm = fit.map(&scale);
        assert!((mm.element().center().x - 0.5).abs() < 1e-12);
        assert!((mm.element().center().y - 0.25).abs() < 1e-12);
        assert!((mm.peak_to_valley().get() - 0.5 * 0.0125).abs() < 1e-14);
        assert_eq!(mm.used_count(), fit.used_count());

        // A line under a rotation: the extent maps with it.
        let line_points: Vec<_> = (0..=10).map(|k| px(k as f64, 0.0)).collect();
        let line = try_fit(&line_points, TotalLeastSquares, AllPoints).unwrap();
        let turn: Similarity<Pixels, Millimeter> = Similarity::try_linear(2.0, FRAC_PI_2).unwrap();
        let mapped = line.map(&turn);
        let extent = mapped.extent();
        assert!(extent.start.distance(Point::new(0.0, 0.0)).get() < 1e-12);
        assert!(extent.end.distance(Point::new(0.0, 20.0)).get() < 1e-12);

        let ellipse_points: Vec<_> = (0..12)
            .map(|k| ellipse_point(0.0, 0.0, 20.0, 8.0, 0.0, TAU * k as f64 / 12.0))
            .collect();
        let ellipse = try_fit(&ellipse_points, Fitzgibbon, AllPoints).unwrap();
        let mapped = ellipse.map(&turn);
        assert!((mapped.element().semi_major().get() - 40.0).abs() < 1e-8);
        let vertical = AxialOrientation::from_radians(FRAC_PI_2).unwrap();
        assert!(
            mapped
                .element()
                .orientation()
                .signed_difference(vertical)
                .abs()
                < 1e-9
        );
    }
}

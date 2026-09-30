//! The affine family: uniform scale, axis scale, similarity and affine.
//!
//! Every mapping here stores its inverse, computed once at construction and
//! checked there, so [`AffineMap::inverse`] is a swap: it cannot fail, and
//! `m.inverse().inverse()` returns `m` bit for bit.

use core::fmt;
use core::marker::PhantomData;

use super::map::{AffineMap, ConformalMap, PlaneMap};
use super::point::{Point, Vector};
use super::units::LengthUnit;
use crate::Error;
use crate::error::{ParameterError, Requirement, Value};

type Units<D, C> = PhantomData<fn() -> (D, C)>;

/// Whether a factor and its reciprocal are both normal, so that the mapping
/// and its inverse have finite, non-zero coefficients.
const fn invertible(factor: f64) -> bool {
    factor.is_normal() && (1.0 / factor).is_normal()
}

fn finite_pair(v: [f64; 2]) -> bool {
    v[0].is_finite() && v[1].is_finite()
}

fn translation_error(t: [f64; 2]) -> Error {
    ParameterError::new(
        "translation",
        Requirement::Finite,
        Value::F64Pair(t[0], t[1]),
    )
    .into()
}

/// `x ↦ a·x + t`: the same factor on both axes, square pixels.
///
/// The calibration of a camera with square pixels, aligned with the axes of
/// the world, measured with a gauge block. Conformal: lengths and circles
/// convert. The factor is positive; a negative one would be a half-turn,
/// which [`Similarity`] expresses.
///
/// # Example
///
/// ```
/// use fovea::{Length, Millimeter, Pixels, Point};
/// use fovea::geometry::{AffineMap, ConformalMap, UniformScale, Vector};
///
/// // One pixel is 12.5 µm on the part, checked at compile time.
/// let scale: UniformScale<Pixels, Millimeter> = fovea::uniform_scale!(0.0125);
/// assert_eq!(scale.map_point(Point::new(800.0, 400.0)), Point::new(10.0, 5.0));
/// assert_eq!(scale.map_length(Length::new(200.0)).get(), 2.5);
///
/// // With the world origin at pixel (40, 20):
/// let placed = UniformScale::<Pixels, Millimeter>::try_new(0.0125, Vector::new(-0.5, -0.25))?;
/// assert_eq!(placed.map_point(Point::new(40.0, 20.0)), Point::new(0.0, 0.0));
/// # Ok::<(), fovea::Error>(())
/// ```
///
/// A factor that is not positive does not build:
///
/// ```compile_fail
/// use fovea::{Millimeter, Pixels};
/// use fovea::geometry::UniformScale;
///
/// // ERROR: evaluation panicked: must be finite and strictly positive
/// let _: UniformScale<Pixels, Millimeter> = fovea::uniform_scale!(0.0);
/// ```
pub struct UniformScale<D, C> {
    factor: f64,
    inverse_factor: f64,
    translation: [f64; 2],
    inverse_translation: [f64; 2],
    units: Units<D, C>,
}

impl<D: LengthUnit, C: LengthUnit> UniformScale<D, C> {
    /// The scale by `factor` about the origin, or `None` unless `factor` is
    /// finite and positive and its reciprocal is finite.
    ///
    /// The `const fn` under [`uniform_scale!`](crate::uniform_scale). Prefer
    /// the macro for literals and [`Self::try_linear`] for computed values.
    #[must_use]
    pub const fn linear(factor: f64) -> Option<Self> {
        if factor > 0.0 && invertible(factor) {
            Some(Self {
                factor,
                inverse_factor: 1.0 / factor,
                translation: [0.0, 0.0],
                inverse_translation: [0.0, 0.0],
                units: PhantomData,
            })
        } else {
            None
        }
    }

    /// The scale by `factor` about the origin.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidParameter`] if `factor` is not finite and positive,
    /// or so small that its reciprocal overflows.
    pub fn try_linear(factor: f64) -> Result<Self, Error> {
        Self::linear(factor).ok_or_else(|| {
            ParameterError::new("factor", Requirement::FinitePositive, Value::F64(factor)).into()
        })
    }

    /// The scale by `factor` followed by a shift by `translation`, which is
    /// in the codomain's unit.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidParameter`] if `factor` is rejected as in
    /// [`Self::try_linear`], or if the translation, or the inverse's
    /// translation, is not finite.
    pub fn try_new(factor: f64, translation: Vector<C>) -> Result<Self, Error> {
        let linear = Self::try_linear(factor)?;
        let t = [translation.x, translation.y];
        let inverse_t = [-t[0] / factor, -t[1] / factor];
        if !(finite_pair(t) && finite_pair(inverse_t)) {
            return Err(translation_error(t));
        }
        Ok(Self {
            translation: t,
            inverse_translation: inverse_t,
            ..linear
        })
    }

    /// The translation, in the codomain's unit.
    #[must_use]
    pub fn translation(&self) -> Vector<C> {
        Vector::new(self.translation[0], self.translation[1])
    }
}

/// `x ↦ diag(a, b)·x + t`: a factor per axis, non-square pixels.
///
/// The calibration of a line-scan camera, whose pixel pitch across the line
/// and whose feed along it differ, or of a binned sensor. A negative factor
/// reflects its axis: `axis_scale!(0.02, -0.02)` maps an image frame with
/// `y` pointing down to a machine frame with `y` pointing up. Not
/// conformal: lengths convert only with their direction, through
/// [`AffineMap::length_of`], and a circle becomes an ellipse.
///
/// # Example
///
/// ```
/// use fovea::{Millimeter, Pixels, Point};
/// use fovea::geometry::{AffineMap, AxisScale};
///
/// let scale: AxisScale<Pixels, Millimeter> = fovea::axis_scale!(0.02, -0.02);
/// assert!(scale.reverses_orientation());
/// assert_eq!(scale.map_point(Point::new(50.0, 25.0)), Point::new(1.0, -0.5));
/// ```
///
/// A zero factor does not build:
///
/// ```compile_fail
/// use fovea::{Millimeter, Pixels};
/// use fovea::geometry::AxisScale;
///
/// // ERROR: evaluation panicked: must be finite and non-zero
/// let _: AxisScale<Pixels, Millimeter> = fovea::axis_scale!(0.02, 0.0);
/// ```
pub struct AxisScale<D, C> {
    factors: [f64; 2],
    inverse_factors: [f64; 2],
    translation: [f64; 2],
    inverse_translation: [f64; 2],
    units: Units<D, C>,
}

impl<D: LengthUnit, C: LengthUnit> AxisScale<D, C> {
    /// The scale by `x_factor` horizontally and `y_factor` vertically about
    /// the origin, or `None` unless both are finite and non-zero with finite
    /// reciprocals.
    ///
    /// The `const fn` under [`axis_scale!`](crate::axis_scale). Prefer the
    /// macro for literals and [`Self::try_linear`] for computed values.
    #[must_use]
    pub const fn linear(x_factor: f64, y_factor: f64) -> Option<Self> {
        if invertible(x_factor) && invertible(y_factor) {
            Some(Self {
                factors: [x_factor, y_factor],
                inverse_factors: [1.0 / x_factor, 1.0 / y_factor],
                translation: [0.0, 0.0],
                inverse_translation: [0.0, 0.0],
                units: PhantomData,
            })
        } else {
            None
        }
    }

    /// The scale by `x_factor` and `y_factor` about the origin.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidParameter`] naming the factor that is zero, not
    /// finite, or so small that its reciprocal overflows.
    pub fn try_linear(x_factor: f64, y_factor: f64) -> Result<Self, Error> {
        for (name, value) in [("x factor", x_factor), ("y factor", y_factor)] {
            if !invertible(value) {
                return Err(ParameterError::new(
                    name,
                    Requirement::FiniteNonZero,
                    Value::F64(value),
                )
                .into());
            }
        }
        Ok(Self::linear(x_factor, y_factor).expect("both factors were just checked"))
    }

    /// The scale by `x_factor` and `y_factor` followed by a shift by
    /// `translation`, which is in the codomain's unit.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidParameter`] if a factor is rejected as in
    /// [`Self::try_linear`], or if the translation, or the inverse's
    /// translation, is not finite.
    pub fn try_new(x_factor: f64, y_factor: f64, translation: Vector<C>) -> Result<Self, Error> {
        let linear = Self::try_linear(x_factor, y_factor)?;
        let t = [translation.x, translation.y];
        let inverse_t = [-t[0] / x_factor, -t[1] / y_factor];
        if !(finite_pair(t) && finite_pair(inverse_t)) {
            return Err(translation_error(t));
        }
        Ok(Self {
            translation: t,
            inverse_translation: inverse_t,
            ..linear
        })
    }

    /// The horizontal factor.
    #[must_use]
    pub fn x_factor(&self) -> f64 {
        self.factors[0]
    }

    /// The vertical factor.
    #[must_use]
    pub fn y_factor(&self) -> f64 {
        self.factors[1]
    }

    /// The translation, in the codomain's unit.
    #[must_use]
    pub fn translation(&self) -> Vector<C> {
        Vector::new(self.translation[0], self.translation[1])
    }
}

/// `x ↦ a·R·x + t`: a rotation by an angle, a positive factor, and a
/// translation, optionally followed by a reflection.
///
/// The calibration of a camera with square pixels that sits at an angle to
/// the machine axes. Conformal: lengths and circles convert, and angles are
/// kept (their sign flips under a reflection).
///
/// The angle is in radians and turns the `x` axis towards the `y` axis;
/// with `y` pointing down, as in an image, that is clockwise on screen.
/// [`mirrored_y`](Self::mirrored_y) follows the mapping with `y ↦ −y`,
/// which together with the rotation gives every reflection.
///
/// # Example
///
/// ```
/// use std::f64::consts::FRAC_PI_2;
/// use fovea::{Millimeter, Pixels, Point};
/// use fovea::geometry::{AffineMap, ConformalMap, Similarity};
///
/// let s = Similarity::<Pixels, Millimeter>::try_linear(0.01, FRAC_PI_2)?;
/// let p = s.map_point(Point::new(100.0, 0.0));
/// assert!(p.x.abs() < 1e-15 && (p.y - 1.0).abs() < 1e-15);
/// assert_eq!(s.factor(), 0.01);
///
/// // The same camera, reported in a machine frame with y pointing up:
/// let up = s.mirrored_y();
/// assert!(up.reverses_orientation());
/// let q = up.map_point(Point::new(100.0, 0.0));
/// assert!(q.x.abs() < 1e-15 && (q.y + 1.0).abs() < 1e-15);
/// # Ok::<(), fovea::Error>(())
/// ```
pub struct Similarity<D, C> {
    scale: f64,
    inverse_scale: f64,
    rotation: [f64; 2],
    inverse_rotation: [f64; 2],
    mirrored: bool,
    translation: [f64; 2],
    inverse_translation: [f64; 2],
    units: Units<D, C>,
}

/// The complex product `u·p` of two vectors read as complex numbers.
#[inline]
fn complex_mul(u: [f64; 2], p: [f64; 2]) -> [f64; 2] {
    [u[0] * p[0] - u[1] * p[1], u[1] * p[0] + u[0] * p[1]]
}

#[inline]
fn conj(v: [f64; 2]) -> [f64; 2] {
    [v[0], -v[1]]
}

/// `scale·u·p` or, mirrored, `scale·u·conj(p)`: the linear part of a
/// similarity applied to `p`.
#[inline]
fn similarity_linear(scale: f64, u: [f64; 2], mirrored: bool, p: [f64; 2]) -> [f64; 2] {
    let p = if mirrored { conj(p) } else { p };
    let r = complex_mul(u, p);
    [scale * r[0], scale * r[1]]
}

impl<D: LengthUnit, C: LengthUnit> Similarity<D, C> {
    /// The rotation by `angle` radians and scale by `scale` about the
    /// origin.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidParameter`] if `scale` is not finite and positive,
    /// or so small or large that its reciprocal is not normal, or if
    /// `angle` is not finite.
    pub fn try_linear(scale: f64, angle: f64) -> Result<Self, Error> {
        Self::try_new(scale, angle, Vector::new(0.0, 0.0))
    }

    /// The rotation by `angle` radians and scale by `scale`, followed by a
    /// shift by `translation`, which is in the codomain's unit.
    ///
    /// # Errors
    ///
    /// As [`Self::try_linear`], and if the translation, or the inverse's
    /// translation, is not finite.
    pub fn try_new(scale: f64, angle: f64, translation: Vector<C>) -> Result<Self, Error> {
        if !(scale > 0.0 && invertible(scale)) {
            return Err(ParameterError::new(
                "scale",
                Requirement::FinitePositive,
                Value::F64(scale),
            )
            .into());
        }
        if !angle.is_finite() {
            return Err(
                ParameterError::new("angle", Requirement::Finite, Value::F64(angle)).into(),
            );
        }
        let (sin, cos) = angle.sin_cos();
        let rotation = [cos, sin];
        let t = [translation.x, translation.y];
        // Inverse of p ↦ s·u·p + t: q ↦ (1/s)·conj(u)·(q − t).
        let inverse_rotation = conj(rotation);
        let rotated = complex_mul(inverse_rotation, t);
        let inverse_t = [-rotated[0] / scale, -rotated[1] / scale];
        if !(finite_pair(t) && finite_pair(inverse_t)) {
            return Err(translation_error(t));
        }
        Ok(Self {
            scale,
            inverse_scale: 1.0 / scale,
            rotation,
            inverse_rotation,
            mirrored: false,
            translation: t,
            inverse_translation: inverse_t,
            units: PhantomData,
        })
    }

    /// This mapping followed by the reflection `y ↦ −y` in the codomain.
    ///
    /// The way from an image frame, `y` down, to a machine frame, `y` up.
    /// Applying it twice returns the original mapping exactly.
    #[must_use]
    pub fn mirrored_y(self) -> Self {
        // conj(s·u·p + t) = s·conj(u)·conj(p) + conj(t), and the same with
        // the roles of p and conj(p) exchanged; the inverse is the old one
        // applied after y ↦ −y, which only toggles its reflection.
        Self {
            rotation: conj(self.rotation),
            translation: conj(self.translation),
            mirrored: !self.mirrored,
            ..self
        }
    }

    /// The translation, in the codomain's unit.
    #[must_use]
    pub fn translation(&self) -> Vector<C> {
        Vector::new(self.translation[0], self.translation[1])
    }
}

/// `x ↦ A·x + t` with any invertible 2×2 matrix `A`.
///
/// The general member of the family: a camera that views the part
/// obliquely, in the affine approximation, or any composition of the other
/// classes. Lines, parallels, midpoints and ellipses convert; lengths only
/// with their direction.
///
/// The matrix is given by rows, `[[a11, a12], [a21, a22]]`, so that
/// `x' = a11·x + a12·y + tx` and `y' = a21·x + a22·y + ty`, the layout of
/// OpenCV's 2×3 affine matrices without the last column.
///
/// # Example
///
/// ```
/// use fovea::{Pixels, Point};
/// use fovea::geometry::{Affine, AffineMap, Vector};
///
/// // A shear of one pixel per row.
/// let shear = Affine::<Pixels, Pixels>::try_linear([[1.0, 1.0], [0.0, 1.0]])?;
/// assert_eq!(shear.map_point(Point::new(2.0, 3.0)), Point::new(5.0, 3.0));
/// assert_eq!(shear.inverse().map_point(Point::new(5.0, 3.0)), Point::new(2.0, 3.0));
///
/// // A singular matrix is rejected.
/// assert!(Affine::<Pixels, Pixels>::try_linear([[1.0, 2.0], [2.0, 4.0]]).is_err());
/// # Ok::<(), fovea::Error>(())
/// ```
pub struct Affine<D, C> {
    matrix: [[f64; 2]; 2],
    inverse_matrix: [[f64; 2]; 2],
    translation: [f64; 2],
    inverse_translation: [f64; 2],
    units: Units<D, C>,
}

#[inline]
fn matrix_mul(m: [[f64; 2]; 2], p: [f64; 2]) -> [f64; 2] {
    [
        m[0][0] * p[0] + m[0][1] * p[1],
        m[1][0] * p[0] + m[1][1] * p[1],
    ]
}

impl<D: LengthUnit, C: LengthUnit> Affine<D, C> {
    /// The linear mapping with matrix `matrix`, given by rows.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidParameter`] if an entry is not finite (its index,
    /// row-major, is recorded), or if the determinant is zero, not finite,
    /// or so small that the inverse overflows.
    pub fn try_linear(matrix: [[f64; 2]; 2]) -> Result<Self, Error> {
        Self::try_new(matrix, Vector::new(0.0, 0.0))
    }

    /// The mapping `x ↦ matrix·x + translation`, with the translation in the
    /// codomain's unit.
    ///
    /// # Errors
    ///
    /// As [`Self::try_linear`], and if the translation, or the inverse's
    /// translation, is not finite.
    pub fn try_new(matrix: [[f64; 2]; 2], translation: Vector<C>) -> Result<Self, Error> {
        let entries = [matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]];
        if let Some(i) = entries.iter().position(|v| !v.is_finite()) {
            return Err(
                ParameterError::new("matrix", Requirement::Finite, Value::F64(entries[i]))
                    .at(i)
                    .into(),
            );
        }
        let [[a, b], [c, d]] = matrix;
        let det = a * d - b * c;
        let inverse_matrix = [[d / det, -b / det], [-c / det, a / det]];
        if !invertible(det) || !inverse_matrix.iter().flatten().all(|v| v.is_finite()) {
            return Err(ParameterError::new(
                "determinant",
                Requirement::FiniteNonZero,
                Value::F64(det),
            )
            .into());
        }
        let t = [translation.x, translation.y];
        let moved = matrix_mul(inverse_matrix, t);
        let inverse_t = [-moved[0], -moved[1]];
        if !(finite_pair(t) && finite_pair(inverse_t)) {
            return Err(translation_error(t));
        }
        Ok(Self {
            matrix,
            inverse_matrix,
            translation: t,
            inverse_translation: inverse_t,
            units: PhantomData,
        })
    }

    /// The matrix `A`, by rows.
    #[must_use]
    pub fn linear_part(&self) -> [[f64; 2]; 2] {
        self.matrix
    }

    /// The translation, in the codomain's unit.
    #[must_use]
    pub fn translation(&self) -> Vector<C> {
        Vector::new(self.translation[0], self.translation[1])
    }
}

// ── The mapping traits ───────────────────────────────────────────────────────

impl<D: LengthUnit, C: LengthUnit> PlaneMap for UniformScale<D, C> {
    type Domain = D;
    type Codomain = C;
    #[inline]
    fn try_map_point(&self, p: Point<D>) -> Option<Point<C>> {
        Some(self.map_point(p))
    }
}

impl<D: LengthUnit, C: LengthUnit> AffineMap for UniformScale<D, C> {
    type Inverse = UniformScale<C, D>;
    #[inline]
    fn map_point(&self, p: Point<D>) -> Point<C> {
        Point::new(
            self.factor * p.x + self.translation[0],
            self.factor * p.y + self.translation[1],
        )
    }
    #[inline]
    fn map_vector(&self, v: Vector<D>) -> Vector<C> {
        Vector::new(self.factor * v.x, self.factor * v.y)
    }
    fn inverse(&self) -> UniformScale<C, D> {
        UniformScale {
            factor: self.inverse_factor,
            inverse_factor: self.factor,
            translation: self.inverse_translation,
            inverse_translation: self.translation,
            units: PhantomData,
        }
    }
    fn reverses_orientation(&self) -> bool {
        false
    }
}

impl<D: LengthUnit, C: LengthUnit> ConformalMap for UniformScale<D, C> {
    fn factor(&self) -> f64 {
        self.factor
    }
}

impl<D: LengthUnit, C: LengthUnit> PlaneMap for AxisScale<D, C> {
    type Domain = D;
    type Codomain = C;
    #[inline]
    fn try_map_point(&self, p: Point<D>) -> Option<Point<C>> {
        Some(self.map_point(p))
    }
}

impl<D: LengthUnit, C: LengthUnit> AffineMap for AxisScale<D, C> {
    type Inverse = AxisScale<C, D>;
    #[inline]
    fn map_point(&self, p: Point<D>) -> Point<C> {
        Point::new(
            self.factors[0] * p.x + self.translation[0],
            self.factors[1] * p.y + self.translation[1],
        )
    }
    #[inline]
    fn map_vector(&self, v: Vector<D>) -> Vector<C> {
        Vector::new(self.factors[0] * v.x, self.factors[1] * v.y)
    }
    fn inverse(&self) -> AxisScale<C, D> {
        AxisScale {
            factors: self.inverse_factors,
            inverse_factors: self.factors,
            translation: self.inverse_translation,
            inverse_translation: self.translation,
            units: PhantomData,
        }
    }
    fn reverses_orientation(&self) -> bool {
        (self.factors[0] < 0.0) != (self.factors[1] < 0.0)
    }
}

impl<D: LengthUnit, C: LengthUnit> PlaneMap for Similarity<D, C> {
    type Domain = D;
    type Codomain = C;
    #[inline]
    fn try_map_point(&self, p: Point<D>) -> Option<Point<C>> {
        Some(self.map_point(p))
    }
}

impl<D: LengthUnit, C: LengthUnit> AffineMap for Similarity<D, C> {
    type Inverse = Similarity<C, D>;
    #[inline]
    fn map_point(&self, p: Point<D>) -> Point<C> {
        let q = similarity_linear(self.scale, self.rotation, self.mirrored, [p.x, p.y]);
        Point::new(q[0] + self.translation[0], q[1] + self.translation[1])
    }
    #[inline]
    fn map_vector(&self, v: Vector<D>) -> Vector<C> {
        let q = similarity_linear(self.scale, self.rotation, self.mirrored, [v.x, v.y]);
        Vector::new(q[0], q[1])
    }
    fn inverse(&self) -> Similarity<C, D> {
        Similarity {
            scale: self.inverse_scale,
            inverse_scale: self.scale,
            rotation: self.inverse_rotation,
            inverse_rotation: self.rotation,
            mirrored: self.mirrored,
            translation: self.inverse_translation,
            inverse_translation: self.translation,
            units: PhantomData,
        }
    }
    fn reverses_orientation(&self) -> bool {
        self.mirrored
    }
}

impl<D: LengthUnit, C: LengthUnit> ConformalMap for Similarity<D, C> {
    fn factor(&self) -> f64 {
        self.scale
    }
}

impl<D: LengthUnit, C: LengthUnit> PlaneMap for Affine<D, C> {
    type Domain = D;
    type Codomain = C;
    #[inline]
    fn try_map_point(&self, p: Point<D>) -> Option<Point<C>> {
        Some(self.map_point(p))
    }
}

impl<D: LengthUnit, C: LengthUnit> AffineMap for Affine<D, C> {
    type Inverse = Affine<C, D>;
    #[inline]
    fn map_point(&self, p: Point<D>) -> Point<C> {
        let q = matrix_mul(self.matrix, [p.x, p.y]);
        Point::new(q[0] + self.translation[0], q[1] + self.translation[1])
    }
    #[inline]
    fn map_vector(&self, v: Vector<D>) -> Vector<C> {
        let q = matrix_mul(self.matrix, [v.x, v.y]);
        Vector::new(q[0], q[1])
    }
    fn inverse(&self) -> Affine<C, D> {
        Affine {
            matrix: self.inverse_matrix,
            inverse_matrix: self.matrix,
            translation: self.inverse_translation,
            inverse_translation: self.translation,
            units: PhantomData,
        }
    }
    fn reverses_orientation(&self) -> bool {
        let [[a, b], [c, d]] = self.matrix;
        a * d - b * c < 0.0
    }
}

// ── Every class is an affine mapping ────────────────────────────────────────

impl<D: LengthUnit, C: LengthUnit> From<UniformScale<D, C>> for Affine<D, C> {
    fn from(s: UniformScale<D, C>) -> Self {
        Affine {
            matrix: [[s.factor, 0.0], [0.0, s.factor]],
            inverse_matrix: [[s.inverse_factor, 0.0], [0.0, s.inverse_factor]],
            translation: s.translation,
            inverse_translation: s.inverse_translation,
            units: PhantomData,
        }
    }
}

impl<D: LengthUnit, C: LengthUnit> From<AxisScale<D, C>> for Affine<D, C> {
    fn from(s: AxisScale<D, C>) -> Self {
        let [a, b] = s.factors;
        let [ia, ib] = s.inverse_factors;
        Affine {
            matrix: [[a, 0.0], [0.0, b]],
            inverse_matrix: [[ia, 0.0], [0.0, ib]],
            translation: s.translation,
            inverse_translation: s.inverse_translation,
            units: PhantomData,
        }
    }
}

/// The matrix of `p ↦ scale·u·p` or, mirrored, `p ↦ scale·u·conj(p)`.
fn similarity_matrix(scale: f64, u: [f64; 2], mirrored: bool) -> [[f64; 2]; 2] {
    let [c, s] = [scale * u[0], scale * u[1]];
    if mirrored {
        [[c, s], [s, -c]]
    } else {
        [[c, -s], [s, c]]
    }
}

impl<D: LengthUnit, C: LengthUnit> From<Similarity<D, C>> for Affine<D, C> {
    fn from(s: Similarity<D, C>) -> Self {
        Affine {
            matrix: similarity_matrix(s.scale, s.rotation, s.mirrored),
            inverse_matrix: similarity_matrix(s.inverse_scale, s.inverse_rotation, s.mirrored),
            translation: s.translation,
            inverse_translation: s.inverse_translation,
            units: PhantomData,
        }
    }
}

// ── Common traits, without bounds on the units ──────────────────────────────

macro_rules! copy_without_bounds {
    ($($ty:ident),*) => {$(
        impl<D, C> Clone for $ty<D, C> {
            #[inline]
            fn clone(&self) -> Self {
                *self
            }
        }
        impl<D, C> Copy for $ty<D, C> {}
    )*};
}
copy_without_bounds!(UniformScale, AxisScale, Similarity, Affine);

impl<D, C> PartialEq for UniformScale<D, C> {
    fn eq(&self, other: &Self) -> bool {
        self.factor == other.factor && self.translation == other.translation
    }
}
impl<D, C> PartialEq for AxisScale<D, C> {
    fn eq(&self, other: &Self) -> bool {
        self.factors == other.factors && self.translation == other.translation
    }
}
impl<D, C> PartialEq for Similarity<D, C> {
    fn eq(&self, other: &Self) -> bool {
        self.scale == other.scale
            && self.rotation == other.rotation
            && self.mirrored == other.mirrored
            && self.translation == other.translation
    }
}
impl<D, C> PartialEq for Affine<D, C> {
    fn eq(&self, other: &Self) -> bool {
        self.matrix == other.matrix && self.translation == other.translation
    }
}

impl<D, C> fmt::Debug for UniformScale<D, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UniformScale")
            .field("factor", &self.factor)
            .field("translation", &self.translation)
            .finish()
    }
}
impl<D, C> fmt::Debug for AxisScale<D, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AxisScale")
            .field("factors", &self.factors)
            .field("translation", &self.translation)
            .finish()
    }
}
impl<D, C> fmt::Debug for Similarity<D, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Similarity")
            .field("scale", &self.scale)
            .field("rotation", &self.rotation)
            .field("mirrored", &self.mirrored)
            .field("translation", &self.translation)
            .finish()
    }
}
impl<D, C> fmt::Debug for Affine<D, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Affine")
            .field("matrix", &self.matrix)
            .field("translation", &self.translation)
            .finish()
    }
}

// ── Literal macros ───────────────────────────────────────────────────────────

/// A [`UniformScale`](crate::geometry::UniformScale) literal about the
/// origin, checked at compile time.
///
/// The units come from the annotation. A factor that is not a constant
/// expression does not compile (`error[E0435]`); use
/// [`UniformScale::try_linear`](crate::geometry::UniformScale::try_linear)
/// there.
///
/// # Example
///
/// ```
/// use fovea::{Micrometer, Pixels};
/// use fovea::geometry::{ConformalMap, UniformScale};
///
/// const CAMERA: UniformScale<Pixels, Micrometer> = fovea::uniform_scale!(3.45);
/// assert_eq!(CAMERA.factor(), 3.45);
/// ```
///
/// ```compile_fail
/// use fovea::{Micrometer, Pixels};
/// use fovea::geometry::UniformScale;
///
/// // ERROR: evaluation panicked: must be finite and strictly positive
/// let _: UniformScale<Pixels, Micrometer> = fovea::uniform_scale!(-3.45);
/// ```
#[macro_export]
macro_rules! uniform_scale {
    ($factor:expr) => {
        const {
            $crate::geometry::UniformScale::linear($factor)
                .expect($crate::error::Requirement::FinitePositive.text())
        }
    };
}

/// An [`AxisScale`](crate::geometry::AxisScale) literal about the origin,
/// checked at compile time.
///
/// The units come from the annotation. Factors that are not constant
/// expressions do not compile (`error[E0435]`); use
/// [`AxisScale::try_linear`](crate::geometry::AxisScale::try_linear) there.
///
/// # Example
///
/// ```
/// use fovea::{Millimeter, Pixels};
/// use fovea::geometry::AxisScale;
///
/// // 20 µm across the line, 50 µm of feed per line, y pointing up.
/// const LINE_SCAN: AxisScale<Pixels, Millimeter> = fovea::axis_scale!(0.02, -0.05);
/// assert_eq!((LINE_SCAN.x_factor(), LINE_SCAN.y_factor()), (0.02, -0.05));
/// ```
///
/// ```compile_fail
/// use fovea::{Millimeter, Pixels};
/// use fovea::geometry::AxisScale;
///
/// // ERROR: evaluation panicked: must be finite and non-zero
/// let _: AxisScale<Pixels, Millimeter> = fovea::axis_scale!(0.0, 0.05);
/// ```
#[macro_export]
macro_rules! axis_scale {
    ($x_factor:expr, $y_factor:expr) => {
        const {
            $crate::geometry::AxisScale::linear($x_factor, $y_factor)
                .expect($crate::error::Requirement::FiniteNonZero.text())
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{Length, Micrometer, Millimeter, Pixels};
    use core::f64::consts::{FRAC_PI_2, FRAC_PI_6};

    fn px(x: f64, y: f64) -> Point<Pixels> {
        Point::new(x, y)
    }

    fn close<U: LengthUnit>(a: Point<U>, b: Point<U>, tol: f64) -> bool {
        (a.x - b.x).abs() <= tol && (a.y - b.y).abs() <= tol
    }

    fn requirement_of(e: Error) -> (&'static str, Requirement, Value, Option<usize>) {
        match e {
            Error::InvalidParameter(p) => (p.parameter(), p.requirement(), p.value(), p.index()),
            other => panic!("expected InvalidParameter, got {other:?}"),
        }
    }

    const SAMPLES: [(f64, f64); 5] = [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (-37.5, 12.25),
        (4096.0, 3000.0),
    ];

    /// Checks the round trip, the inverse's inverse, vector consistency and
    /// that `try_map_point` agrees with `map_point`.
    fn check_affine<M>(m: &M, tol: f64)
    where
        M: AffineMap<Domain = Pixels> + PartialEq + fmt::Debug,
        M::Inverse: AffineMap<Inverse = M>,
    {
        let inv = m.inverse();
        assert_eq!(&inv.inverse(), m, "the inverse's inverse is the mapping");
        for (x, y) in SAMPLES {
            let p = px(x, y);
            let q = m.map_point(p);
            assert_eq!(m.try_map_point(p), Some(q));
            assert!(close(inv.map_point(q), p, tol), "{m:?} at {p:?}");
            let o = px(0.0, 0.0);
            let v = p - o;
            let mv = m.map_vector(v);
            let diff = m.map_point(p) - m.map_point(o);
            assert!((mv.x - diff.x).abs() <= tol && (mv.y - diff.y).abs() <= tol);
            assert_eq!(m.length_of(v), mv.length());
        }
    }

    #[test]
    fn uniform_scale_maps_and_inverts() {
        let s: UniformScale<Pixels, Millimeter> =
            UniformScale::try_new(0.0125, Vector::new(2.0, -1.0)).unwrap();
        assert_eq!(s.map_point(px(800.0, 400.0)), Point::new(12.0, 4.0));
        assert_eq!(s.map_vector(Vector::new(8.0, 0.0)), Vector::new(0.1, 0.0));
        assert_eq!(s.factor(), 0.0125);
        assert_eq!(s.map_length(Length::new(80.0)), Length::new(1.0));
        assert!(!s.reverses_orientation());
        assert_eq!(s.translation(), Vector::new(2.0, -1.0));
        check_affine(&s, 1e-9);
    }

    #[test]
    fn uniform_scale_rejects_bad_factors() {
        // 1e-310 is subnormal, and 1 / f64::MAX is: neither has an
        // inverse with normal coefficients.
        for bad in [0.0, -0.5, f64::NAN, f64::INFINITY, 1e-310, f64::MAX] {
            let e = UniformScale::<Pixels, Millimeter>::try_linear(bad).unwrap_err();
            let (name, req, value, index) = requirement_of(e);
            assert_eq!(
                (name, req, index),
                ("factor", Requirement::FinitePositive, None)
            );
            assert_eq!(value, Value::F64(bad));
            assert!(UniformScale::<Pixels, Millimeter>::linear(bad).is_none());
        }
        let e = UniformScale::<Pixels, Millimeter>::try_new(1.0, Vector::new(f64::NAN, 0.0))
            .unwrap_err();
        assert_eq!(requirement_of(e).0, "translation");
    }

    #[test]
    fn axis_scale_maps_inverts_and_reflects() {
        let s: AxisScale<Pixels, Millimeter> =
            AxisScale::try_new(0.02, -0.05, Vector::new(1.0, 30.0)).unwrap();
        assert_eq!(s.map_point(px(100.0, 200.0)), Point::new(3.0, 20.0));
        assert!(s.reverses_orientation());
        assert_eq!((s.x_factor(), s.y_factor()), (0.02, -0.05));
        check_affine(&s, 1e-9);

        let both: AxisScale<Pixels, Millimeter> = AxisScale::try_linear(-1.0, -1.0).unwrap();
        assert!(
            !both.reverses_orientation(),
            "two reflections are a half-turn"
        );
    }

    #[test]
    fn axis_scale_lengths_depend_on_direction() {
        let s: AxisScale<Pixels, Micrometer> = crate::axis_scale!(20.0, 50.0);
        assert_eq!(s.length_of(Vector::new(1.0, 0.0)).get(), 20.0);
        assert_eq!(s.length_of(Vector::new(0.0, 1.0)).get(), 50.0);
    }

    #[test]
    fn axis_scale_names_the_rejected_factor() {
        let e = AxisScale::<Pixels, Millimeter>::try_linear(0.02, 0.0).unwrap_err();
        assert_eq!(
            requirement_of(e),
            (
                "y factor",
                Requirement::FiniteNonZero,
                Value::F64(0.0),
                None
            )
        );
        let e = AxisScale::<Pixels, Millimeter>::try_linear(f64::NAN, 1.0).unwrap_err();
        assert_eq!(requirement_of(e).0, "x factor");
    }

    #[test]
    fn similarity_rotates_scales_and_inverts() {
        let s: Similarity<Pixels, Millimeter> =
            Similarity::try_new(0.01, FRAC_PI_2, Vector::new(5.0, 0.0)).unwrap();
        assert!(close(
            s.map_point(px(100.0, 0.0)),
            Point::new(5.0, 1.0),
            1e-14
        ));
        assert!(close(
            s.map_point(px(0.0, 100.0)),
            Point::new(4.0, 0.0),
            1e-14
        ));
        assert_eq!(s.factor(), 0.01);
        assert!(!s.reverses_orientation());
        check_affine(&s, 1e-9);

        let tilted: Similarity<Pixels, Millimeter> =
            Similarity::try_new(0.02, FRAC_PI_6, Vector::new(-3.0, 7.0)).unwrap();
        check_affine(&tilted, 1e-9);
        let v = Vector::new(3.0, 4.0);
        assert!((tilted.length_of(v).get() - 0.1).abs() < 1e-15, "conformal");
    }

    #[test]
    fn mirrored_similarity_reflects_and_inverts() {
        let s: Similarity<Pixels, Millimeter> =
            Similarity::try_new(0.02, FRAC_PI_6, Vector::new(-3.0, 7.0)).unwrap();
        let m = s.mirrored_y();
        assert!(m.reverses_orientation());
        for (x, y) in SAMPLES {
            let a = s.map_point(px(x, y));
            let b = m.map_point(px(x, y));
            assert!(close(b, Point::new(a.x, -a.y), 1e-12));
        }
        check_affine(&m, 1e-9);
        assert_eq!(m.mirrored_y(), s, "mirroring twice is exact");
        assert_eq!(m.inverse().inverse(), m);
    }

    #[test]
    fn similarity_rejects_bad_parameters() {
        let e = Similarity::<Pixels, Millimeter>::try_linear(0.0, 0.0).unwrap_err();
        assert_eq!(requirement_of(e).1, Requirement::FinitePositive);
        let e = Similarity::<Pixels, Millimeter>::try_linear(1.0, f64::INFINITY).unwrap_err();
        assert_eq!(
            requirement_of(e),
            (
                "angle",
                Requirement::Finite,
                Value::F64(f64::INFINITY),
                None
            )
        );
    }

    #[test]
    fn affine_maps_inverts_and_reports_orientation() {
        let a: Affine<Pixels, Millimeter> =
            Affine::try_new([[0.02, 0.001], [-0.002, 0.03]], Vector::new(10.0, -4.0)).unwrap();
        let p = a.map_point(px(100.0, 50.0));
        assert!(close(p, Point::new(12.05, -2.7), 1e-12));
        assert!(!a.reverses_orientation());
        assert_eq!(a.linear_part(), [[0.02, 0.001], [-0.002, 0.03]]);
        check_affine(&a, 1e-9);

        let flipped: Affine<Pixels, Pixels> = Affine::try_linear([[0.0, 1.0], [1.0, 0.0]]).unwrap();
        assert!(
            flipped.reverses_orientation(),
            "swapping axes is a reflection"
        );
    }

    #[test]
    fn affine_rejects_singular_and_non_finite_matrices() {
        let e = Affine::<Pixels, Pixels>::try_linear([[1.0, 2.0], [2.0, 4.0]]).unwrap_err();
        assert_eq!(
            requirement_of(e),
            (
                "determinant",
                Requirement::FiniteNonZero,
                Value::F64(0.0),
                None
            )
        );
        let e = Affine::<Pixels, Pixels>::try_linear([[1.0, 0.0], [f64::NAN, 1.0]]).unwrap_err();
        let (name, req, _, index) = requirement_of(e);
        assert_eq!((name, req, index), ("matrix", Requirement::Finite, Some(2)));
        let e = Affine::<Pixels, Pixels>::try_linear([[1e-200, 0.0], [0.0, 1e-200]]).unwrap_err();
        assert_eq!(requirement_of(e).0, "determinant", "det underflows to zero");
    }

    #[test]
    fn every_class_converts_to_the_same_affine_mapping() {
        let u: UniformScale<Pixels, Millimeter> =
            UniformScale::try_new(0.5, Vector::new(1.0, 2.0)).unwrap();
        let x: AxisScale<Pixels, Millimeter> =
            AxisScale::try_new(0.5, -2.0, Vector::new(1.0, 2.0)).unwrap();
        let s: Similarity<Pixels, Millimeter> =
            Similarity::try_new(1.5, 0.7, Vector::new(-1.0, 3.0)).unwrap();
        let m = s.mirrored_y();
        let (au, ax, as_, am) = (
            Affine::from(u),
            Affine::from(x),
            Affine::from(s),
            Affine::from(m),
        );
        for (px_, py) in SAMPLES {
            let p = px(px_, py);
            assert!(close(au.map_point(p), u.map_point(p), 1e-9));
            assert!(close(ax.map_point(p), x.map_point(p), 1e-9));
            assert!(close(as_.map_point(p), s.map_point(p), 1e-9));
            assert!(close(am.map_point(p), m.map_point(p), 1e-9));
            let q = s.map_point(p);
            assert!(close(as_.inverse().map_point(q), p, 1e-9));
        }
        assert_eq!(am.reverses_orientation(), m.reverses_orientation());
        assert_eq!(ax.reverses_orientation(), x.reverses_orientation());
        check_affine(&am, 1e-9);
    }

    #[test]
    fn literal_macros_build_in_const_context() {
        const U: UniformScale<Pixels, Micrometer> = crate::uniform_scale!(3.45);
        const A: AxisScale<Pixels, Millimeter> = crate::axis_scale!(0.02, -0.02);
        assert_eq!(U.factor(), 3.45);
        assert!(A.reverses_orientation());
    }
}

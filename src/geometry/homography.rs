//! The projective mapping of the plane: a homography.

use core::fmt;
use core::marker::PhantomData;

use super::map::PlaneMap;
use super::point::Point;
use super::units::LengthUnit;
use crate::Error;
use crate::error::{ParameterError, Requirement, Value};

type M3 = [[f64; 3]; 3];

/// `x ↦ (H·[x, 1]ᵀ)` divided by its third coordinate: a camera looking at
/// a plane from an angle, and the mapping that rectifies its view.
///
/// Lines stay lines, but parallels meet, lengths and angles change across
/// the image, and a circle becomes a conic whose centre is not the image of
/// the circle's centre. So a homography converts points only: it is a
/// [`PlaneMap`] and not an [`AffineMap`](super::AffineMap). A point on its
/// vanishing line, where the third coordinate is zero, has no image, and
/// [`try_map_point`](PlaneMap::try_map_point) answers `None` there.
///
/// The matrix is given by rows, the layout of a 3×3 matrix in OpenCV, and
/// only up to scale: multiplying it by a non-zero number gives the same
/// mapping. Construction rejects a matrix that is not finite or not
/// invertible, and stores the inverse, so [`inverse`](Self::inverse)
/// cannot fail.
///
/// # Example
///
/// ```
/// use fovea::{Pixels, Point};
/// use fovea::geometry::{Homography, PlaneMap};
///
/// // A perspective that stretches x twice and foreshortens towards y = 1.
/// let h: Homography<Pixels, Pixels> = Homography::try_new([
///     [2.0, 0.0, 0.0],
///     [0.0, 1.0, 0.0],
///     [0.0, -1.0, 1.0],
/// ])?;
/// assert_eq!(h.try_map_point(Point::new(1.0, 0.0)), Some(Point::new(2.0, 0.0)));
/// // The line y = 1 is the vanishing line: it has no image.
/// assert_eq!(h.try_map_point(Point::new(0.5, 1.0)), None);
///
/// let back = h.inverse().try_map_point(Point::new(2.0, 0.0)).unwrap();
/// assert!((back.x - 1.0).abs() < 1e-12 && back.y.abs() < 1e-12);
/// # Ok::<(), fovea::Error>(())
/// ```
pub struct Homography<D, C> {
    matrix: M3,
    inverse: M3,
    units: PhantomData<fn() -> (D, C)>,
}

fn determinant(m: &M3) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// The adjugate divided by the determinant.
fn invert(m: &M3, det: f64) -> M3 {
    let [[a, b, c], [d, e, f], [g, h, i]] = *m;
    let adjugate = [
        [e * i - f * h, c * h - b * i, b * f - c * e],
        [f * g - d * i, a * i - c * g, c * d - a * f],
        [d * h - e * g, b * g - a * h, a * e - b * d],
    ];
    adjugate.map(|row| row.map(|v| v / det))
}

fn apply(m: &M3, x: f64, y: f64) -> Option<(f64, f64)> {
    let w = m[2][0] * x + m[2][1] * y + m[2][2];
    if w == 0.0 {
        return None;
    }
    let u = (m[0][0] * x + m[0][1] * y + m[0][2]) / w;
    let v = (m[1][0] * x + m[1][1] * y + m[1][2]) / w;
    (u.is_finite() && v.is_finite()).then_some((u, v))
}

impl<D: LengthUnit, C: LengthUnit> Homography<D, C> {
    /// The homography of the matrix given by `rows`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidParameter`] if an entry is not finite, the index
    /// counting row by row, or if the matrix is singular or so close to it
    /// that its inverse is not finite.
    pub fn try_new(rows: [[f64; 3]; 3]) -> Result<Self, Error> {
        let entries = rows.as_flattened();
        if let Some(i) = entries.iter().position(|v| !v.is_finite()) {
            return Err(ParameterError::new(
                "homography entry",
                Requirement::Finite,
                Value::F64(entries[i]),
            )
            .at(i)
            .into());
        }
        let det = determinant(&rows);
        let inverse = invert(&rows, det);
        let invertible = det.is_normal() && inverse.as_flattened().iter().all(|v| v.is_finite());
        if !invertible {
            return Err(ParameterError::new(
                "homography determinant",
                Requirement::FiniteNonZero,
                Value::F64(det),
            )
            .into());
        }
        Ok(Self {
            matrix: rows,
            inverse,
            units: PhantomData,
        })
    }

    /// The matrix, by rows, as given.
    #[must_use]
    pub fn rows(&self) -> [[f64; 3]; 3] {
        self.matrix
    }

    /// The inverse homography, from the codomain back to the domain.
    #[must_use]
    pub fn inverse(&self) -> Homography<C, D> {
        Homography {
            matrix: self.inverse,
            inverse: self.matrix,
            units: PhantomData,
        }
    }
}

impl<D: LengthUnit, C: LengthUnit> PlaneMap for Homography<D, C> {
    type Domain = D;
    type Codomain = C;

    fn try_map_point(&self, p: Point<D>) -> Option<Point<C>> {
        apply(&self.matrix, p.x, p.y).map(|(x, y)| Point::new(x, y))
    }
}

impl<D, C> Clone for Homography<D, C> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<D, C> Copy for Homography<D, C> {}
impl<D, C> PartialEq for Homography<D, C> {
    fn eq(&self, other: &Self) -> bool {
        self.matrix == other.matrix
    }
}
impl<D, C> fmt::Debug for Homography<D, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Homography")
            .field("rows", &self.matrix)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{Millimeter, Pixels};

    fn px(x: f64, y: f64) -> Point<Pixels> {
        Point::new(x, y)
    }

    #[test]
    fn a_homography_rejects_entries_that_are_not_finite_and_singular_matrices() {
        let Err(Error::InvalidParameter(e)) = Homography::<Pixels, Pixels>::try_new([
            [1.0, 0.0, 0.0],
            [0.0, f64::NAN, 0.0],
            [0.0, 0.0, 1.0],
        ]) else {
            panic!("NaN accepted");
        };
        assert_eq!((e.requirement(), e.index()), (Requirement::Finite, Some(4)));
        let singular = [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.0, 0.0, 1.0]];
        let Err(Error::InvalidParameter(e)) = Homography::<Pixels, Pixels>::try_new(singular)
        else {
            panic!("a singular matrix was accepted");
        };
        assert_eq!(e.requirement(), Requirement::FiniteNonZero);
        let tiny = [[1e-200, 0.0, 0.0], [0.0, 1e-200, 0.0], [0.0, 0.0, 1e-200]];
        assert!(Homography::<Pixels, Pixels>::try_new(tiny).is_err());
    }

    #[test]
    fn the_inverse_undoes_the_mapping_and_scale_does_not_matter() {
        let rows = [[1.2, 0.1, 30.0], [-0.05, 0.9, 12.0], [1e-4, 2e-4, 1.0]];
        let h: Homography<Pixels, Millimeter> = Homography::try_new(rows).unwrap();
        let scaled: Homography<Pixels, Millimeter> =
            Homography::try_new(rows.map(|r| r.map(|v| v * -3.0))).unwrap();
        for p in [px(0.0, 0.0), px(640.0, 480.0), px(-100.0, 900.0)] {
            let q = h.try_map_point(p).unwrap();
            let r = scaled.try_map_point(p).unwrap();
            assert!((q.x - r.x).abs() < 1e-9 && (q.y - r.y).abs() < 1e-9);
            let back = h.inverse().try_map_point(q).unwrap();
            assert!(back.distance(p).get() < 1e-9, "{p:?}");
        }
        assert_eq!(h.inverse().inverse(), h);
        assert_eq!(h.rows(), rows);
    }

    #[test]
    fn an_affine_matrix_maps_like_the_affine_family() {
        use crate::geometry::{Affine, AffineMap, Vector};
        let a: Affine<Pixels, Pixels> =
            Affine::try_new([[2.0, 0.5], [-0.25, 1.5]], Vector::new(3.0, -4.0)).unwrap();
        let h: Homography<Pixels, Pixels> =
            Homography::try_new([[2.0, 0.5, 3.0], [-0.25, 1.5, -4.0], [0.0, 0.0, 1.0]]).unwrap();
        for p in [px(1.0, 2.0), px(-7.5, 0.25)] {
            let q = h.try_map_point(p).unwrap();
            assert!(q.distance(a.map_point(p)).get() < 1e-12);
        }
    }

    #[test]
    fn the_vanishing_line_has_no_image() {
        let h: Homography<Pixels, Pixels> =
            Homography::try_new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.5, 1.0]]).unwrap();
        assert_eq!(h.try_map_point(px(5.0, -2.0)), None);
        assert!(h.try_map_point(px(5.0, -1.5)).is_some());
        assert!(format!("{h:?}").starts_with("Homography { rows: "));
    }
}

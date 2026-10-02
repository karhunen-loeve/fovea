//! The camera matrix and the Brown-Conrady lens distortion model.

use super::map::PlaneMap;
use super::point::Point;
use super::units::Pixels;
use crate::Error;
use crate::error::{ParameterError, Requirement, Value};

/// The focal length of a camera in pixels, along `x` and `y`: the focal
/// length divided by the pixel pitch on each axis.
///
/// The two differ when the pixels are not square.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FocalLength {
    /// The focal length in pixels along `x` (`fx`).
    pub x: f64,
    /// The focal length in pixels along `y` (`fy`).
    pub y: f64,
}

/// The intrinsic matrix of a pinhole camera without skew: the focal length
/// in pixels and the principal point.
///
/// It relates pixels to normalised camera coordinates, in which a lens
/// model is written: `x = (u − cx) / fx`, `y = (v − cy) / fy`. Estimating
/// it is calibration, which fovea does not do; a camera matrix from a
/// calibration tool is an input.
///
/// # Example
///
/// ```
/// use fovea::Point;
/// use fovea::geometry::{CameraMatrix, FocalLength};
///
/// let cam = CameraMatrix::try_new(FocalLength { x: 1000.0, y: 1000.0 }, Point::new(640.0, 480.0))?;
/// assert_eq!(cam.principal_point(), Point::new(640.0, 480.0));
/// assert!(CameraMatrix::try_new(FocalLength { x: 0.0, y: 1000.0 }, Point::new(0.0, 0.0)).is_err());
/// # Ok::<(), fovea::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraMatrix {
    focal_length: FocalLength,
    principal_point: Point<Pixels>,
}

impl CameraMatrix {
    /// The camera matrix of `focal_length` and `principal_point`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidParameter`] if a focal length is not finite and
    /// positive, or the principal point is not finite.
    pub fn try_new(
        focal_length: FocalLength,
        principal_point: Point<Pixels>,
    ) -> Result<Self, Error> {
        for (name, f) in [
            ("focal length x", focal_length.x),
            ("focal length y", focal_length.y),
        ] {
            if !(f.is_finite() && f > 0.0) {
                return Err(
                    ParameterError::new(name, Requirement::FinitePositive, Value::F64(f)).into(),
                );
            }
        }
        let (cx, cy) = (principal_point.x, principal_point.y);
        if !(cx.is_finite() && cy.is_finite()) {
            return Err(ParameterError::new(
                "principal point",
                Requirement::Finite,
                Value::F64Pair(cx, cy),
            )
            .into());
        }
        Ok(Self {
            focal_length,
            principal_point,
        })
    }

    /// The focal length in pixels.
    #[must_use]
    pub fn focal_length(&self) -> FocalLength {
        self.focal_length
    }

    /// The principal point, where the optical axis meets the image.
    #[must_use]
    pub fn principal_point(&self) -> Point<Pixels> {
        self.principal_point
    }

    fn to_normalized(self, p: Point<Pixels>) -> (f64, f64) {
        (
            (p.x - self.principal_point.x) / self.focal_length.x,
            (p.y - self.principal_point.y) / self.focal_length.y,
        )
    }

    fn to_pixels(self, x: f64, y: f64) -> Point<Pixels> {
        Point::new(
            self.focal_length.x * x + self.principal_point.x,
            self.focal_length.y * y + self.principal_point.y,
        )
    }
}

/// The radial coefficients of [`BrownConrady`]: `k1`, `k2` and `k3`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Radial {
    /// The coefficient of `r²`.
    pub k1: f64,
    /// The coefficient of `r⁴`.
    pub k2: f64,
    /// The coefficient of `r⁶`.
    pub k3: f64,
}

/// The tangential (decentring) coefficients of [`BrownConrady`]: `p1` and
/// `p2`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tangential {
    /// The first tangential coefficient.
    pub p1: f64,
    /// The second tangential coefficient.
    pub p2: f64,
}

/// The five coefficients of [`BrownConrady`], each by its name.
///
/// Every field must be written, so a coefficient cannot be forgotten and
/// silently stay zero, and their order does not matter. OpenCV's array
/// order is the job of [`BrownConrady::from_opencv`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BrownConradyCoefficients {
    /// `k1`, `k2`, `k3`.
    pub radial: Radial,
    /// `p1`, `p2`.
    pub tangential: Tangential,
}

/// The Brown-Conrady lens distortion model: radial terms in `r²`, `r⁴` and
/// `r⁶` and two tangential terms, in normalised camera coordinates.
///
/// In normalised coordinates `(x, y)`, with `r² = x² + y²`, a point is
/// distorted to
///
/// ```text
/// x_d = x·(1 + k1·r² + k2·r⁴ + k3·r⁶) + 2·p1·x·y + p2·(r² + 2·x²)
/// y_d = y·(1 + k1·r² + k2·r⁴ + k3·r⁶) + p1·(r² + 2·y²) + 2·p2·x·y
/// ```
///
/// the model OpenCV's calibration estimates, with its five most used
/// coefficients. As a [`PlaneMap`] it maps a pixel of the ideal,
/// undistorted image to the pixel of the camera image where the lens put
/// it, both through the same [`CameraMatrix`]. That is the direction a
/// remap reads, so an undistorted image is
/// `DestToSourceTable::new(&DestToSource(lens), size)`.
///
/// **For measurement, correct points rather than images.** Measure edges,
/// corners and fits on the camera image as it is, and pass the measured
/// points through [`undistort_point`](Self::undistort_point): resampling the
/// image first adds an interpolation error to every point, and correcting
/// the points adds none.
///
/// # Example
///
/// ```
/// use fovea::Point;
/// use fovea::geometry::{BrownConrady, CameraMatrix, FocalLength, PlaneMap};
///
/// let cam = CameraMatrix::try_new(FocalLength { x: 1000.0, y: 1000.0 }, Point::new(640.0, 480.0))?;
/// // OpenCV's order: k1, k2, p1, p2, k3.
/// let lens = BrownConrady::from_opencv(cam, &[-0.3, 0.1, 0.001, -0.0005, 0.0])?;
///
/// // A corner of the image is pulled towards the centre by the barrel.
/// let ideal = Point::new(0.0, 0.0);
/// let seen = lens.try_map_point(ideal).unwrap();
/// assert!(seen.x > 0.0 && seen.y > 0.0);
///
/// // Correcting the measured point gives back the ideal one.
/// let corrected = lens.undistort_point(seen)?;
/// assert!(corrected.distance(ideal).get() < 1e-6);
/// # Ok::<(), fovea::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BrownConrady {
    camera: CameraMatrix,
    coefficients: BrownConradyCoefficients,
}

/// The distance in pixels within which a corrected point, distorted again,
/// must land on the measured one. Newton's method reaches it in three or
/// four steps on ordinary lenses, with a step to spare.
const UNDISTORT_TOLERANCE_PX: f64 = 1e-9;

/// The steps after which a correction that has not reached the tolerance
/// gives up.
const UNDISTORT_MAX_STEPS: usize = 20;

/// The array lengths OpenCV uses for its distortion coefficients.
const OPENCV_LENGTHS: &[usize] = &[4, 5, 8, 12, 14];

impl BrownConrady {
    /// The model of `camera` with `coefficients`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidParameter`] if a coefficient is not finite.
    pub fn try_new(
        camera: CameraMatrix,
        coefficients: BrownConradyCoefficients,
    ) -> Result<Self, Error> {
        let BrownConradyCoefficients {
            radial: Radial { k1, k2, k3 },
            tangential: Tangential { p1, p2 },
        } = coefficients;
        for (name, v) in [("k1", k1), ("k2", k2), ("k3", k3), ("p1", p1), ("p2", p2)] {
            if !v.is_finite() {
                return Err(ParameterError::new(name, Requirement::Finite, Value::F64(v)).into());
            }
        }
        Ok(Self {
            camera,
            coefficients,
        })
    }

    /// The model of `camera` with the coefficients in OpenCV's order,
    /// `(k1, k2, p1, p2[, k3[, k4, k5, k6[, s1, s2, s3, s4[, τx, τy]]]])`.
    ///
    /// Four or five coefficients are the model itself; with four, `k3` is
    /// zero, as in OpenCV. A longer array (8, 12 or 14, from OpenCV's
    /// rational, thin-prism or tilted models) is accepted only when every
    /// coefficient after the fifth is exactly zero, so a model this one does
    /// not describe is an error and is never truncated.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidParameter`] if the length is not one of 4, 5, 8, 12
    /// or 14, if a coefficient is not finite, or if a coefficient after the
    /// fifth is not zero; the index is recorded.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::Point;
    /// use fovea::geometry::{BrownConrady, CameraMatrix, FocalLength};
    ///
    /// let cam = CameraMatrix::try_new(FocalLength { x: 900.0, y: 900.0 }, Point::new(320.0, 240.0))?;
    /// let lens = BrownConrady::from_opencv(cam, &[-0.2, 0.05, 0.0, 0.0])?;
    /// assert_eq!(lens.coefficients().radial.k3, 0.0);
    ///
    /// // The rational model's k4 is not part of this one.
    /// let rational = [-0.2, 0.05, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0];
    /// assert!(BrownConrady::from_opencv(cam, &rational).is_err());
    /// # Ok::<(), fovea::Error>(())
    /// ```
    pub fn from_opencv(camera: CameraMatrix, coefficients: &[f64]) -> Result<Self, Error> {
        if !OPENCV_LENGTHS.contains(&coefficients.len()) {
            return Err(ParameterError::new(
                "distortion coefficient count",
                Requirement::OneOf(OPENCV_LENGTHS),
                Value::Usize(coefficients.len()),
            )
            .into());
        }
        for (i, &v) in coefficients.iter().enumerate() {
            if !v.is_finite() {
                return Err(ParameterError::new(
                    "distortion coefficient",
                    Requirement::Finite,
                    Value::F64(v),
                )
                .at(i)
                .into());
            }
            if i >= 5 && v != 0.0 {
                return Err(ParameterError::new(
                    "distortion coefficient",
                    Requirement::Zero,
                    Value::F64(v),
                )
                .at(i)
                .into());
            }
        }
        let c = coefficients;
        Self::try_new(
            camera,
            BrownConradyCoefficients {
                radial: Radial {
                    k1: c[0],
                    k2: c[1],
                    k3: c.get(4).copied().unwrap_or(0.0),
                },
                tangential: Tangential { p1: c[2], p2: c[3] },
            },
        )
    }

    /// The camera matrix.
    #[must_use]
    pub fn camera(&self) -> CameraMatrix {
        self.camera
    }

    /// The coefficients.
    #[must_use]
    pub fn coefficients(&self) -> BrownConradyCoefficients {
        self.coefficients
    }

    /// The point of the ideal, undistorted image whose distorted image is
    /// `measured`: a point measured on the camera image, corrected for the
    /// lens.
    ///
    /// The model has no closed-form inverse. Newton's method, with the
    /// model's exact derivative and starting from the measured point, runs
    /// until the corrected point, distorted again, lands within 10⁻⁹ px of
    /// `measured`, which takes three or four steps on ordinary lenses. That
    /// tolerance is a stopping rule checked on the result; how far the
    /// corrected point is from the true one also depends on how well the
    /// model describes the lens.
    ///
    /// Far enough from the centre a polynomial model folds back on itself,
    /// and other ideal points, some on the far side of the centre, distort
    /// onto the same measured one. A point is accepted only where the model
    /// is one-to-one: its radial factor positive and its derivative
    /// orientation-preserving.
    ///
    /// # Errors
    ///
    /// - [`Error::DidNotConverge`] if no such point is found within 20
    ///   steps, which in practice means `measured` lies outside the field
    ///   the lens was calibrated over.
    /// - [`Error::InvalidParameter`] if `measured` is not finite.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::{Error, Point};
    /// use fovea::geometry::{BrownConrady, CameraMatrix, FocalLength, PlaneMap};
    ///
    /// let cam = CameraMatrix::try_new(FocalLength { x: 1000.0, y: 1000.0 }, Point::new(640.0, 480.0))?;
    /// let lens = BrownConrady::from_opencv(cam, &[-0.4, 0.0, 0.0, 0.0])?;
    ///
    /// let edge = Point::new(1100.0, 300.0);
    /// let ideal = lens.undistort_point(edge)?;
    /// let again = lens.try_map_point(ideal).unwrap();
    /// assert!(again.distance(edge).get() < 1e-8);
    ///
    /// // A barrel this strong maps no ideal point further than about 609 px
    /// // from the centre, so a point 800 px out has no correction.
    /// let beyond = Point::new(640.0 + 800.0, 480.0);
    /// assert!(matches!(lens.undistort_point(beyond), Err(Error::DidNotConverge { .. })));
    /// # Ok::<(), fovea::Error>(())
    /// ```
    pub fn undistort_point(&self, measured: Point<Pixels>) -> Result<Point<Pixels>, Error> {
        if !(measured.x.is_finite() && measured.y.is_finite()) {
            return Err(ParameterError::new(
                "point",
                Requirement::Finite,
                Value::F64Pair(measured.x, measured.y),
            )
            .into());
        }
        let FocalLength { x: fx, y: fy } = self.camera.focal_length;
        let (tx, ty) = self.camera.to_normalized(measured);
        let (mut x, mut y) = (tx, ty);
        for step in 0..UNDISTORT_MAX_STEPS {
            let (dx, dy) = self.distort(x, y);
            let (ex, ey) = (dx - tx, dy - ty);
            let [[a, b], [c, d]] = self.jacobian(x, y);
            let det = a * d - b * c;
            if (ex * fx).hypot(ey * fy) <= UNDISTORT_TOLERANCE_PX {
                // Beyond the fold the model maps further ideal points onto
                // the same measured one, some through the centre; only a
                // point where the model is one-to-one is the correction.
                return if self.radial_factor(x, y) > 0.0 && det > 0.0 {
                    Ok(self.camera.to_pixels(x, y))
                } else {
                    Err(Error::DidNotConverge { steps: step })
                };
            }
            let (sx, sy) = ((d * ex - b * ey) / det, (a * ey - c * ex) / det);
            if !(sx.is_finite() && sy.is_finite()) {
                return Err(Error::DidNotConverge { steps: step });
            }
            (x, y) = (x - sx, y - sy);
        }
        Err(Error::DidNotConverge {
            steps: UNDISTORT_MAX_STEPS,
        })
    }

    /// `1 + k1·r² + k2·r⁴ + k3·r⁶`.
    fn radial_factor(&self, x: f64, y: f64) -> f64 {
        let Radial { k1, k2, k3 } = self.coefficients.radial;
        let r2 = x * x + y * y;
        1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
    }

    /// The model in normalised coordinates.
    fn distort(&self, x: f64, y: f64) -> (f64, f64) {
        let BrownConradyCoefficients {
            radial: Radial { k1, k2, k3 },
            tangential: Tangential { p1, p2 },
        } = self.coefficients;
        let r2 = x * x + y * y;
        let radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3));
        (
            x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x),
            y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y,
        )
    }

    /// The derivative of [`distort`](Self::distort), by rows:
    /// `[[∂x_d/∂x, ∂x_d/∂y], [∂y_d/∂x, ∂y_d/∂y]]`.
    fn jacobian(&self, x: f64, y: f64) -> [[f64; 2]; 2] {
        let BrownConradyCoefficients {
            radial: Radial { k1, k2, k3 },
            tangential: Tangential { p1, p2 },
        } = self.coefficients;
        let r2 = x * x + y * y;
        let radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3));
        // The derivative of the radial factor with respect to r².
        let slope = k1 + r2 * (2.0 * k2 + 3.0 * k3 * r2);
        [
            [
                radial + 2.0 * x * x * slope + 2.0 * p1 * y + 6.0 * p2 * x,
                2.0 * x * y * slope + 2.0 * p1 * x + 2.0 * p2 * y,
            ],
            [
                2.0 * x * y * slope + 2.0 * p1 * x + 2.0 * p2 * y,
                radial + 2.0 * y * y * slope + 6.0 * p1 * y + 2.0 * p2 * x,
            ],
        ]
    }
}

impl PlaneMap for BrownConrady {
    type Domain = Pixels;
    type Codomain = Pixels;

    /// The pixel of the camera image where the lens puts the ideal pixel
    /// `p`; `None` only where that position is not finite.
    fn try_map_point(&self, p: Point<Pixels>) -> Option<Point<Pixels>> {
        let (x, y) = self.camera.to_normalized(p);
        let (dx, dy) = self.distort(x, y);
        let q = self.camera.to_pixels(dx, dy);
        (q.x.is_finite() && q.y.is_finite()).then_some(q)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn camera() -> CameraMatrix {
        CameraMatrix::try_new(
            FocalLength {
                x: 1000.0,
                y: 1000.0,
            },
            Point::new(640.0, 480.0),
        )
        .unwrap()
    }

    fn lens(k1: f64, k2: f64, k3: f64, p1: f64, p2: f64) -> BrownConrady {
        BrownConrady::try_new(
            camera(),
            BrownConradyCoefficients {
                radial: Radial { k1, k2, k3 },
                tangential: Tangential { p1, p2 },
            },
        )
        .unwrap()
    }

    #[test]
    fn the_camera_matrix_validates_its_parts() {
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let e = CameraMatrix::try_new(FocalLength { x: 1.0, y: bad }, Point::new(0.0, 0.0));
            let Err(Error::InvalidParameter(e)) = e else {
                panic!("{bad} accepted");
            };
            assert_eq!(e.parameter(), "focal length y");
        }
        let e = CameraMatrix::try_new(FocalLength { x: 1.0, y: 1.0 }, Point::new(f64::NAN, 0.0));
        let Err(Error::InvalidParameter(e)) = e else {
            panic!("a NaN principal point was accepted");
        };
        assert_eq!(e.value(), Value::F64Pair(f64::NAN, 0.0));
    }

    #[test]
    fn the_model_matches_the_formula_and_opencvs_order() {
        let l = lens(-0.3, 0.1, 0.02, 0.001, -0.0005);
        // The pixel (840, 330) is (0.2, −0.15) in normalised coordinates.
        let (x, y) = (0.2f64, -0.15f64);
        let r2 = x * x + y * y;
        let radial = 1.0 + -0.3 * r2 + 0.1 * r2 * r2 + 0.02 * r2 * r2 * r2;
        let xd = x * radial + 2.0 * 0.001 * x * y + -0.0005 * (r2 + 2.0 * x * x);
        let yd = y * radial + 0.001 * (r2 + 2.0 * y * y) + 2.0 * -0.0005 * x * y;
        let q = l.try_map_point(Point::new(840.0, 330.0)).unwrap();
        assert!((q.x - (1000.0 * xd + 640.0)).abs() < 1e-9);
        assert!((q.y - (1000.0 * yd + 480.0)).abs() < 1e-9);

        let from_cv =
            BrownConrady::from_opencv(camera(), &[-0.3, 0.1, 0.001, -0.0005, 0.02]).unwrap();
        assert_eq!(from_cv, l);
    }

    #[test]
    fn from_opencv_checks_length_finiteness_and_unused_coefficients() {
        let Err(Error::InvalidParameter(e)) = BrownConrady::from_opencv(camera(), &[0.0; 6]) else {
            panic!("six coefficients accepted");
        };
        assert_eq!(e.requirement(), Requirement::OneOf(OPENCV_LENGTHS));
        let mut fourteen = [0.0; 14];
        fourteen[0] = -0.1;
        assert!(BrownConrady::from_opencv(camera(), &fourteen).is_ok());
        fourteen[12] = 1e-3;
        let Err(Error::InvalidParameter(e)) = BrownConrady::from_opencv(camera(), &fourteen) else {
            panic!("a tilt coefficient was dropped");
        };
        assert_eq!((e.requirement(), e.index()), (Requirement::Zero, Some(12)));
        let Err(Error::InvalidParameter(e)) =
            BrownConrady::from_opencv(camera(), &[0.0, f64::NAN, 0.0, 0.0])
        else {
            panic!("NaN accepted");
        };
        assert_eq!((e.requirement(), e.index()), (Requirement::Finite, Some(1)));
        let bad = BrownConradyCoefficients {
            radial: Radial {
                k1: 0.0,
                k2: 0.0,
                k3: f64::INFINITY,
            },
            tangential: Tangential { p1: 0.0, p2: 0.0 },
        };
        let Err(Error::InvalidParameter(e)) = BrownConrady::try_new(camera(), bad) else {
            panic!("an infinite k3 was accepted");
        };
        assert_eq!(e.parameter(), "k3");
    }

    #[test]
    fn the_jacobian_matches_finite_differences() {
        let l = lens(-0.3, 0.1, -0.05, 0.002, -0.001);
        let h = 1e-6;
        for (x, y) in [(0.3, -0.2), (-0.5, 0.4), (0.05, 0.6)] {
            let j = l.jacobian(x, y);
            let (xp, yp) = l.distort(x + h, y);
            let (xm, ym) = l.distort(x - h, y);
            assert!((j[0][0] - (xp - xm) / (2.0 * h)).abs() < 1e-8);
            assert!((j[1][0] - (yp - ym) / (2.0 * h)).abs() < 1e-8);
            let (xp, yp) = l.distort(x, y + h);
            let (xm, ym) = l.distort(x, y - h);
            assert!((j[0][1] - (xp - xm) / (2.0 * h)).abs() < 1e-8);
            assert!((j[1][1] - (yp - ym) / (2.0 * h)).abs() < 1e-8);
        }
    }

    #[test]
    fn undistorting_inverts_the_model_over_the_whole_image() {
        for l in [
            lens(-0.12, 0.05, 0.0, 0.0005, -0.0003),
            lens(-0.3, 0.1, 0.0, 0.001, -0.0005),
            lens(-0.4, 0.2, -0.05, 0.001, -0.0005),
            lens(0.15, 0.02, 0.0, 0.0, 0.0),
        ] {
            for i in (0..=1280).step_by(64) {
                for j in (0..=960).step_by(64) {
                    let ideal = Point::new(i as f64, j as f64);
                    let seen = l.try_map_point(ideal).unwrap();
                    let back = l.undistort_point(seen).unwrap();
                    assert!(back.distance(ideal).get() < 1e-8, "{ideal:?} {l:?}");
                }
            }
        }
    }

    #[test]
    fn without_distortion_the_correction_is_the_identity() {
        let l = lens(0.0, 0.0, 0.0, 0.0, 0.0);
        let p = Point::new(123.25, 456.5);
        assert_eq!(l.undistort_point(p).unwrap(), p);
        assert_eq!(l.try_map_point(p), Some(p));
    }

    #[test]
    fn a_root_beyond_the_fold_is_not_a_correction() {
        // With k1 = −0.4 alone, the ideal point 1210 px right of the centre
        // distorts onto about the same place as the one 570 px right: the
        // second lies beyond the fold, where the model runs backwards.
        let l = lens(-0.4, 0.0, 0.0, 0.0, 0.0);
        let far_ideal = Point::new(640.0 + 1210.0, 480.0);
        let seen = l.try_map_point(far_ideal).unwrap();
        let corrected = l.undistort_point(seen);
        match corrected {
            Ok(p) => assert!(p.x - 640.0 < 913.0, "a root beyond the fold: {p:?}"),
            Err(e) => assert!(matches!(e, Error::DidNotConverge { .. })),
        }
        // Over a wide field, every accepted correction lies where the model
        // is one-to-one.
        for i in (-1500..=1500).step_by(50) {
            for j in (-1500..=1500).step_by(50) {
                let p = Point::new(640.0 + i as f64, 480.0 + j as f64);
                if let Ok(q) = l.undistort_point(p) {
                    let (x, y) = l.camera.to_normalized(q);
                    let [[a, b], [c, d]] = l.jacobian(x, y);
                    assert!(l.radial_factor(x, y) > 0.0 && a * d - b * c > 0.0, "{p:?}");
                }
            }
        }
    }

    #[test]
    fn a_point_beyond_the_fold_does_not_converge() {
        let l = lens(-0.4, 0.0, 0.0, 0.0, 0.0);
        let r = l.undistort_point(Point::new(640.0 + 800.0, 480.0));
        assert!(matches!(r, Err(Error::DidNotConverge { .. })), "{r:?}");
        let r = l.undistort_point(Point::new(f64::NAN, 1.0));
        assert!(matches!(r, Err(Error::InvalidParameter(_))));
    }
}

//! The numerics of the three estimators: total least squares for the line,
//! Taubin's method for the circle, and the direct least squares ellipse.
//!
//! Each takes the points with a weight per point, all weights finite and
//! not negative, at least the estimator's minimum of them positive, and
//! every point finite; `try_fit` checks all of that before calling.

use crate::AxialOrientation;
use crate::Error;
use crate::geometry::{Circle, Ellipse, Length, LengthUnit, Line, Point, Vector};

/// The weighted centroid of `points`.
fn centroid<U: LengthUnit>(points: &[Point<U>], weights: &[f64]) -> (f64, f64, f64) {
    let (mut sw, mut sx, mut sy) = (0.0, 0.0, 0.0);
    for (p, &w) in points.iter().zip(weights) {
        sw += w;
        sx += w * p.x;
        sy += w * p.y;
    }
    (sw, sx / sw, sy / sw)
}

/// The line that minimises the weighted sum of squared orthogonal
/// distances.
///
/// It passes through the weighted centroid along the eigenvector of the
/// larger eigenvalue of the weighted scatter matrix, taken from whichever
/// row of the eigen equation is better conditioned, so a line along an axis
/// comes out exactly along it. The direction is the one in
/// `(−π/2, π/2]`; the caller orients it. A scatter with no principal axis
/// (all points at one place, or spread equally in every direction)
/// determines no line.
pub(super) fn fit_line<U: LengthUnit>(
    points: &[Point<U>],
    weights: &[f64],
) -> Result<Line<U>, Error> {
    let (_, cx, cy) = centroid(points, weights);
    let (mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0);
    for (p, &w) in points.iter().zip(weights) {
        let (dx, dy) = (p.x - cx, p.y - cy);
        sxx += w * dx * dx;
        syy += w * dy * dy;
        sxy += w * dx * dy;
    }
    let half_gap = 0.5 * (sxx - syy).hypot(2.0 * sxy);
    if half_gap == 0.0 {
        return Err(Error::DegeneratePoints);
    }
    let lambda = 0.5 * (sxx + syy) + half_gap;
    let (u, v) = ((sxy, lambda - sxx), (lambda - syy, sxy));
    let (x, y) = if u.0.hypot(u.1) > v.0.hypot(v.1) {
        u
    } else {
        v
    };
    let n = x.hypot(y);
    let (x, y) = if x < 0.0 || (x == 0.0 && y < 0.0) {
        (-x / n, -y / n)
    } else {
        (x / n, y / n)
    };
    Ok(Line::from_parts(Point::new(cx, cy), Vector::new(x, y)))
}

/// The circle by Taubin's method.
///
/// The algebraic fit `A·(x² + y²) + B·x + C·y + D = 0` minimises
/// `Σ w·(A·zᵢ + B·xᵢ + C·yᵢ + D)²` under the constraint that the mean squared
/// gradient of the algebraic function is one, which removes most of the
/// bias of the plain algebraic fit towards small circles. With the data
/// centred on the weighted centroid the solution is the smallest root of a
/// cubic, found by Newton's method from zero, as in Chernov's
/// implementation; the iteration rises monotonically to the root and stops
/// when it no longer gets closer. Points on one line give no finite circle.
pub(super) fn fit_circle<U: LengthUnit>(
    points: &[Point<U>],
    weights: &[f64],
) -> Result<Circle<U>, Error> {
    let (sw, cx, cy) = centroid(points, weights);
    let (mut mxx, mut myy, mut mxy, mut mxz, mut myz, mut mzz) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for (p, &w) in points.iter().zip(weights) {
        let (x, y) = (p.x - cx, p.y - cy);
        let z = x * x + y * y;
        mxx += w * x * x;
        myy += w * y * y;
        mxy += w * x * y;
        mxz += w * x * z;
        myz += w * y * z;
        mzz += w * z * z;
    }
    let (mxx, myy, mxy, mxz, myz, mzz) =
        (mxx / sw, myy / sw, mxy / sw, mxz / sw, myz / sw, mzz / sw);
    let mz = mxx + myy;
    let cov_xy = mxx * myy - mxy * mxy;
    let var_z = mzz - mz * mz;
    let a3 = 4.0 * mz;
    let a2 = -3.0 * mz * mz - mzz;
    let a1 = var_z * mz + 4.0 * cov_xy * mz - mxz * mxz - myz * myz;
    let a0 = mxz * (mxz * myy - myz * mxy) + myz * (myz * mxx - mxz * mxy) - var_z * cov_xy;
    let poly = |x: f64| a0 + x * (a1 + x * (a2 + x * a3));
    let slope = |x: f64| a1 + x * (2.0 * a2 + x * 3.0 * a3);
    let (mut x, mut y) = (0.0, a0);
    for _ in 0..NEWTON_STEPS {
        if y == 0.0 {
            break;
        }
        let next = x - y / slope(x);
        if !next.is_finite() {
            break;
        }
        let y_next = poly(next);
        if y_next.abs() >= y.abs() {
            break;
        }
        (x, y) = (next, y_next);
    }
    // Rounding can leave the root a hair below zero on exact data, where
    // it is zero.
    let x = x.max(0.0);
    let det = x * x - x * mz + cov_xy;
    let ux = (mxz * (myy - x) - myz * mxy) / det / 2.0;
    let uy = (myz * (mxx - x) - mxz * mxy) / det / 2.0;
    let r = (ux * ux + uy * uy + mz).sqrt();
    if !(ux.is_finite() && uy.is_finite() && r.is_finite() && r > 0.0) {
        return Err(Error::DegeneratePoints);
    }
    Ok(Circle::from_parts(
        Point::new(ux + cx, uy + cy),
        Length::new(r),
    ))
}

/// Newton's method on the cubic converges in a handful of steps; this bound
/// only stops a pathological input from looping.
const NEWTON_STEPS: usize = 100;

type M3 = [[f64; 3]; 3];

fn inverse3(m: &M3) -> Option<M3> {
    let [[a, b, c], [d, e, f], [g, h, i]] = *m;
    let co = [
        [e * i - f * h, c * h - b * i, b * f - c * e],
        [f * g - d * i, a * i - c * g, c * d - a * f],
        [d * h - e * g, b * g - a * h, a * e - b * d],
    ];
    let det = a * co[0][0] + b * co[1][0] + c * co[2][0];
    if det == 0.0 || !det.is_finite() {
        return None;
    }
    let mut out = [[0.0; 3]; 3];
    for (r, row) in out.iter_mut().enumerate() {
        for (s, v) in row.iter_mut().enumerate() {
            *v = co[r][s] / det;
        }
    }
    Some(out)
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// The real roots of `t³ + c2·t² + c1·t + c0`.
fn cubic_roots(c2: f64, c1: f64, c0: f64) -> ([f64; 3], usize) {
    let p = c1 - c2 * c2 / 3.0;
    let q = 2.0 * c2 * c2 * c2 / 27.0 - c2 * c1 / 3.0 + c0;
    let shift = -c2 / 3.0;
    let disc = (q / 2.0) * (q / 2.0) + (p / 3.0) * (p / 3.0) * (p / 3.0);
    if p < 0.0 && disc <= 0.0 {
        let m = 2.0 * (-p / 3.0).sqrt();
        let theta = (3.0 * q / (p * m)).clamp(-1.0, 1.0).acos() / 3.0;
        let third = core::f64::consts::TAU / 3.0;
        (
            [
                m * theta.cos() + shift,
                m * (theta - third).cos() + shift,
                m * (theta - 2.0 * third).cos() + shift,
            ],
            3,
        )
    } else {
        let sq = disc.max(0.0).sqrt();
        let root = (-q / 2.0 + sq).cbrt() + (-q / 2.0 - sq).cbrt() + shift;
        ([root, 0.0, 0.0], 1)
    }
}

/// A unit vector spanning the null space of `m − mu·I`, or `None` when the
/// null space is not one line.
fn null_vector(m: &M3, mu: f64) -> Option<[f64; 3]> {
    let mut r = *m;
    for (i, row) in r.iter_mut().enumerate() {
        row[i] -= mu;
    }
    let candidates = [cross3(r[0], r[1]), cross3(r[0], r[2]), cross3(r[1], r[2])];
    let norm2 = |v: &[f64; 3]| v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    let best = candidates
        .iter()
        .copied()
        .max_by(|a, b| norm2(a).total_cmp(&norm2(b)))?;
    let n = norm2(&best).sqrt();
    (n > 0.0 && n.is_finite()).then(|| [best[0] / n, best[1] / n, best[2] / n])
}

/// The ellipse by the direct least squares method of Fitzgibbon, Pilu and
/// Fisher, in the numerically stable form of Halíř and Flusser.
///
/// The conic `A·x² + B·xy + C·y² + D·x + E·y + F = 0` minimises the
/// weighted algebraic error under the constraint `4AC − B² = 1`, which
/// admits only ellipses. The linear part is eliminated in closed form, the
/// quadratic part is an eigenvector of a 3×3 matrix, and the one eigenvector
/// that satisfies the constraint is the ellipse. The points are centred and
/// scaled to unit spread first, which keeps the scatter matrices well
/// conditioned for points at any position and size.
pub(super) fn fit_ellipse<U: LengthUnit>(
    points: &[Point<U>],
    weights: &[f64],
) -> Result<Ellipse<U>, Error> {
    let (sw, cx, cy) = centroid(points, weights);
    let spread = points
        .iter()
        .zip(weights)
        .map(|(p, &w)| w * ((p.x - cx) * (p.x - cx) + (p.y - cy) * (p.y - cy)))
        .sum::<f64>()
        / sw;
    if spread.is_nan() || spread <= 0.0 {
        return Err(Error::DegeneratePoints);
    }
    let scale = spread.sqrt();

    let (mut s1, mut s2, mut s3) = ([[0.0; 3]; 3], [[0.0; 3]; 3], [[0.0; 3]; 3]);
    for (p, &w) in points.iter().zip(weights) {
        let (x, y) = ((p.x - cx) / scale, (p.y - cy) / scale);
        let d1 = [x * x, x * y, y * y];
        let d2 = [x, y, 1.0];
        for i in 0..3 {
            for j in 0..3 {
                s1[i][j] += w * d1[i] * d1[j];
                s2[i][j] += w * d1[i] * d2[j];
                s3[i][j] += w * d2[i] * d2[j];
            }
        }
    }
    let s3_inv = inverse3(&s3).ok_or(Error::DegeneratePoints)?;
    // T = −S3⁻¹·S2ᵀ expresses the linear part (D, E, F) through the
    // quadratic part (A, B, C).
    let mut t = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            t[i][j] = -(0..3).map(|k| s3_inv[i][k] * s2[j][k]).sum::<f64>();
        }
    }
    let mut m = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = s1[i][j] + (0..3).map(|k| s2[i][k] * t[k][j]).sum::<f64>();
        }
    }
    // Premultiplied by the inverse of the constraint matrix.
    let m = [
        [m[2][0] / 2.0, m[2][1] / 2.0, m[2][2] / 2.0],
        [-m[1][0], -m[1][1], -m[1][2]],
        [m[0][0] / 2.0, m[0][1] / 2.0, m[0][2] / 2.0],
    ];
    let trace = m[0][0] + m[1][1] + m[2][2];
    let minors = m[0][0] * m[1][1] - m[0][1] * m[1][0] + m[0][0] * m[2][2] - m[0][2] * m[2][0]
        + m[1][1] * m[2][2]
        - m[1][2] * m[2][1];
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let (roots, count) = cubic_roots(-trace, minors, -det);

    // Exactly one eigenvector satisfies the constraint in exact arithmetic;
    // should rounding admit two, the one of the smaller eigenvalue fits
    // better.
    let mut best: Option<(f64, [f64; 3])> = None;
    for &mu in &roots[..count] {
        let Some(v) = null_vector(&m, mu) else {
            continue;
        };
        if 4.0 * v[0] * v[2] - v[1] * v[1] > 0.0 && best.is_none_or(|(b, _)| mu.abs() < b.abs()) {
            best = Some((mu, v));
        }
    }
    let (_, [mut a, mut b, mut c]) = best.ok_or(Error::DegeneratePoints)?;
    let lin: [f64; 3] = core::array::from_fn(|i| t[i][0] * a + t[i][1] * b + t[i][2] * c);
    let (d, e, f) = (lin[0], lin[1], lin[2]);

    let k = 4.0 * a * c - b * b;
    let x0 = (b * e - 2.0 * c * d) / k;
    let y0 = (b * d - 2.0 * a * e) / k;
    let mut f0 = f + (d * x0 + e * y0) / 2.0;
    if a + c < 0.0 {
        (a, b, c, f0) = (-a, -b, -c, -f0);
    }
    let gap = (a - c).hypot(b);
    let (small, large) = ((a + c - gap) / 2.0, (a + c + gap) / 2.0);
    let is_real_ellipse = f0 < 0.0 && small > 0.0;
    if !is_real_ellipse {
        return Err(Error::DegeneratePoints);
    }
    let semi_major = (-f0 / small).sqrt() * scale;
    let semi_minor = (-f0 / large).sqrt() * scale;
    let center = Point::new(x0 * scale + cx, y0 * scale + cy);
    let finite = center.x.is_finite() && center.y.is_finite() && semi_major.is_finite();
    if !finite || semi_minor.is_nan() || semi_minor <= 0.0 {
        return Err(Error::DegeneratePoints);
    }
    // The eigenvector of the larger eigenvalue of [[A, B/2], [B/2, C]] lies
    // at ½·atan2(B, A − C) and is the minor axis; the major axis is a
    // quarter turn from it, at ½·atan2(−B, C − A).
    Ok(Ellipse::from_parts(
        center,
        semi_major,
        semi_minor.min(semi_major),
        AxialOrientation::from_half_atan2(-b, c - a),
    ))
}

//! The caliper's description and the intensity profile along it.

use core::f64::consts::TAU;

use crate::CoordinateF64;
use crate::Error;
use crate::analyze::sampling::sample;
use crate::border::BorderPolicy;
use crate::error::{ParameterError, Requirement, Value};
use crate::geometry::{Length, Pixels, Point};
use crate::image::ImageView;
use crate::pixel::{HomogeneousPixel, LinearPixel, LinearSpace, SingleChannel};
use crate::transform::InterpolationKernel;

/// The most samples a caliper takes along its path, and across it.
///
/// A bound on the buffer a profile allocates. A path of 16 000 pixels
/// sampled every thousandth of a pixel stays below it; a step that is
/// accidentally tiny is reported as an error instead of exhausting memory.
pub const MAX_CALIPER_SAMPLES: usize = 1 << 24;

/// The path of a caliper.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Path {
    Segment {
        start: [f64; 2],
        /// Unit vector from start to end.
        direction: [f64; 2],
        length: f64,
    },
    Arc {
        centre: [f64; 2],
        radius: f64,
        start_angle: f64,
        sweep: f64,
    },
}

/// Where to measure: a path across the expected edges, the width to average
/// over, and the sampling step.
///
/// A caliper is a description and holds no image, so one caliper serves
/// every frame. [`profile`] reads the intensities along it.
///
/// The path is a segment or an arc. Profile positions lie at `0, step,
/// 2·step, …` from the start of the path, up to its end. At each position
/// the caliper averages the lines at offsets `k·step` across the path,
/// `|k·step| ≤ width / 2`, perpendicular to a segment and radial for an
/// arc; a width of zero reads a single line. Polarity is read in the path's
/// direction.
///
/// # Example
///
/// ```
/// use fovea::{Length, Pixels, Point};
/// use fovea::measure::Caliper;
///
/// // 40 px long, averaging 5 px across, a sample every quarter pixel.
/// let cal = Caliper::try_segment(
///     Point::new(10.0, 16.0),
///     Point::new(50.0, 16.0),
///     Length::new(5.0),
///     Length::new(0.25),
/// )?;
/// assert_eq!(cal.samples(), 161);
/// assert_eq!(cal.lines(), 21);
/// assert_eq!(cal.point_at(Length::new(8.0)), Point::new(18.0, 16.0));
/// # Ok::<(), fovea::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Caliper {
    path: Path,
    width: f64,
    step: f64,
    samples: usize,
    lines: usize,
}

fn finite_point(name: &'static str, p: Point<Pixels>) -> Result<[f64; 2], Error> {
    if p.x.is_finite() && p.y.is_finite() {
        Ok([p.x, p.y])
    } else {
        Err(ParameterError::new(name, Requirement::Finite, Value::F64Pair(p.x, p.y)).into())
    }
}

fn check(
    ok: bool,
    name: &'static str,
    requirement: Requirement,
    value: Value,
) -> Result<(), Error> {
    if ok {
        Ok(())
    } else {
        Err(ParameterError::new(name, requirement, value).into())
    }
}

/// The number of samples in `0, step, …` up to `extent`, checked against
/// [`MAX_CALIPER_SAMPLES`]. A quotient within rounding of a whole number
/// counts as that number, so a step that divides the extent reaches its end.
fn sample_count(name: &'static str, extent: f64, step: f64) -> Result<usize, Error> {
    let steps = (extent / step * (1.0 + 1e-12)).floor();
    let count = steps + 1.0;
    check(
        count <= MAX_CALIPER_SAMPLES as f64,
        name,
        Requirement::AtMost(MAX_CALIPER_SAMPLES),
        Value::F64(count),
    )?;
    Ok(count as usize)
}

fn width_and_step(width: Length<Pixels>, step: Length<Pixels>) -> Result<(f64, f64), Error> {
    let (w, s) = (width.get(), step.get());
    check(
        w.is_finite() && w >= 0.0,
        "width",
        Requirement::FiniteNonNegative,
        Value::F64(w),
    )?;
    check(
        s.is_finite() && s > 0.0,
        "step",
        Requirement::FinitePositive,
        Value::F64(s),
    )?;
    Ok((w, s))
}

impl Caliper {
    /// A caliper along the segment from `start` to `end`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidParameter`] if a point is not finite, the segment
    /// has no length, `width` is negative or not finite, `step` is not
    /// finite and positive, or more than [`MAX_CALIPER_SAMPLES`] samples
    /// would be taken along or across the path.
    pub fn try_segment(
        start: Point<Pixels>,
        end: Point<Pixels>,
        width: Length<Pixels>,
        step: Length<Pixels>,
    ) -> Result<Self, Error> {
        let a = finite_point("start", start)?;
        let b = finite_point("end", end)?;
        let (width, step) = width_and_step(width, step)?;
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let length = dx.hypot(dy);
        check(
            length.is_normal(),
            "path length",
            Requirement::FinitePositive,
            Value::F64(length),
        )?;
        let path = Path::Segment {
            start: a,
            direction: [dx / length, dy / length],
            length,
        };
        Self::assemble(path, length, width, step)
    }

    /// A caliper along the circular arc about `centre` of radius `radius`,
    /// from `start_angle` over `sweep` radians.
    ///
    /// Angles are in radians and turn the `x` axis towards the `y` axis,
    /// which with `y` pointing down is clockwise on screen. A negative
    /// `sweep` runs the other way. The width is measured radially.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidParameter`] if the centre or `start_angle` is not
    /// finite, `radius` is not finite and positive, `sweep` is zero, not
    /// finite or a full turn or more, half the width is not below the
    /// radius, or a parameter is rejected as in [`Self::try_segment`].
    ///
    /// # Example
    ///
    /// ```
    /// use std::f64::consts::PI;
    /// use fovea::{Length, Point};
    /// use fovea::measure::Caliper;
    ///
    /// // A quarter circle of radius 20 px about (32, 32), starting at +x.
    /// let cal = Caliper::try_arc(
    ///     Point::new(32.0, 32.0),
    ///     Length::new(20.0),
    ///     0.0,
    ///     PI / 2.0,
    ///     Length::new(2.0),
    ///     Length::new(0.5),
    /// )?;
    /// assert!((cal.length().get() - 10.0 * PI).abs() < 1e-12);
    /// let end = cal.point_at(cal.length());
    /// assert!((end.x - 32.0).abs() < 1e-9 && (end.y - 52.0).abs() < 1e-9);
    /// # Ok::<(), fovea::Error>(())
    /// ```
    pub fn try_arc(
        centre: Point<Pixels>,
        radius: Length<Pixels>,
        start_angle: f64,
        sweep: f64,
        width: Length<Pixels>,
        step: Length<Pixels>,
    ) -> Result<Self, Error> {
        let c = finite_point("centre", centre)?;
        let (width, step) = width_and_step(width, step)?;
        let r = radius.get();
        check(
            r.is_finite() && r > 0.0,
            "radius",
            Requirement::FinitePositive,
            Value::F64(r),
        )?;
        check(
            start_angle.is_finite(),
            "start angle",
            Requirement::Finite,
            Value::F64(start_angle),
        )?;
        check(
            sweep.is_normal(),
            "sweep",
            Requirement::FiniteNonZero,
            Value::F64(sweep),
        )?;
        check(
            sweep.abs() < TAU,
            "sweep",
            Requirement::OpenInterval {
                low: -TAU,
                high: TAU,
            },
            Value::F64(sweep),
        )?;
        check(
            width / 2.0 < r,
            "half width and radius",
            Requirement::StrictlyOrdered,
            Value::F64Pair(width / 2.0, r),
        )?;
        let length = r * sweep.abs();
        let path = Path::Arc {
            centre: c,
            radius: r,
            start_angle,
            sweep,
        };
        Self::assemble(path, length, width, step)
    }

    fn assemble(path: Path, length: f64, width: f64, step: f64) -> Result<Self, Error> {
        let samples = sample_count("samples along the path", length, step)?;
        let half_lines = sample_count("samples across the path", width / 2.0, step)? - 1;
        Ok(Self {
            path,
            width,
            step,
            samples,
            lines: 2 * half_lines + 1,
        })
    }

    /// The length of the path: the segment's length, or the arc's.
    #[must_use]
    pub fn length(&self) -> Length<Pixels> {
        Length::new(match self.path {
            Path::Segment { length, .. } => length,
            Path::Arc { radius, sweep, .. } => radius * sweep.abs(),
        })
    }

    /// The width averaged across the path.
    #[must_use]
    pub fn width(&self) -> Length<Pixels> {
        Length::new(self.width)
    }

    /// The sampling step, along the path and across it.
    #[must_use]
    pub fn step(&self) -> Length<Pixels> {
        Length::new(self.step)
    }

    /// The number of profile positions along the path.
    #[must_use]
    pub fn samples(&self) -> usize {
        self.samples
    }

    /// The number of lines averaged across the path at each position.
    #[must_use]
    pub fn lines(&self) -> usize {
        self.lines
    }

    /// The point on the path at distance `along` from its start.
    #[must_use]
    pub fn point_at(&self, along: Length<Pixels>) -> Point<Pixels> {
        let [x, y] = self.position(along.get(), 0.0);
        Point::new(x, y)
    }

    /// The position at distance `s` along the path and `offset` across it.
    fn position(&self, s: f64, offset: f64) -> [f64; 2] {
        match self.path {
            Path::Segment {
                start, direction, ..
            } => {
                let normal = [-direction[1], direction[0]];
                [
                    start[0] + s * direction[0] + offset * normal[0],
                    start[1] + s * direction[1] + offset * normal[1],
                ]
            }
            Path::Arc {
                centre,
                radius,
                start_angle,
                sweep,
            } => {
                let angle = start_angle + sweep.signum() * s / radius;
                let (sin, cos) = angle.sin_cos();
                let r = radius + offset;
                [centre[0] + r * cos, centre[1] + r * sin]
            }
        }
    }
}

/// The averaged intensities along a [`Caliper`], one value per position.
///
/// Returned by [`profile`]. The values are kept, so a caller can inspect,
/// plot or reuse them; [`edges`](Profile::edges) finds the edges on them.
///
/// # Example
///
/// ```
/// use fovea::{Length, Point};
/// use fovea::border::Clamp;
/// use fovea::image::Image;
/// use fovea::measure::{Caliper, profile};
/// use fovea::pixel::MonoF32;
/// use fovea::transform::Bilinear;
///
/// // A ramp: pixel x holds 2·x.
/// let img = Image::generate(32, 8, |x, _| MonoF32::new(2.0 * x as f32));
/// let cal = Caliper::try_segment(
///     Point::new(4.0, 4.0),
///     Point::new(6.0, 4.0),
///     Length::new(0.0),
///     Length::new(0.5),
/// )?;
/// let prof = profile(&img, &cal, Bilinear, &Clamp)?;
/// assert_eq!(prof.values(), &[8.0, 9.0, 10.0, 11.0, 12.0]);
/// # Ok::<(), fovea::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Profile {
    caliper: Caliper,
    values: Vec<f64>,
}

impl Profile {
    /// The averaged value at each position along the path.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// The caliper the profile was read along.
    #[must_use]
    pub fn caliper(&self) -> &Caliper {
        &self.caliper
    }
}

/// Reads the intensity profile of `image` along `caliper`.
///
/// Every position along the path averages the caliper's lines across it,
/// each sampled with `kernel` at its sub-pixel position under `border`. The
/// image's pixels are single-channel values in a linear space; a colour
/// image is converted to grey first, by a named conversion.
///
/// `Constant(0)` invents an edge where the caliper leaves the image, and
/// `Clamp` a plateau that hides one, so a measurement should use `Skip`,
/// which reports the footprint leaving the image as an error instead.
///
/// # Errors
///
/// [`Error::CaliperOutsideImage`] if a tap of the kernel falls outside the
/// image under a policy that does not extend it, such as `Skip`, naming the
/// first position at which it happens.
///
/// # Example
///
/// ```
/// use fovea::{Error, Length, Point};
/// use fovea::border::Skip;
/// use fovea::image::Image;
/// use fovea::measure::{Caliper, profile};
/// use fovea::pixel::Mono8;
/// use fovea::transform::CatmullRom;
///
/// let img = Image::fill(32, 32, Mono8::new(100));
/// let inside = Caliper::try_segment(
///     Point::new(4.0, 16.0), Point::new(28.0, 16.0), Length::new(4.0), Length::new(0.5),
/// )?;
/// let prof = profile(&img, &inside, CatmullRom, &Skip)?;
/// assert!(prof.values().iter().all(|&v| (v - 100.0).abs() < 1e-4));
///
/// // Catmull-Rom reads two pixels each side, so a caliper starting at
/// // x = 0.5 has no value there under `Skip`.
/// let edge = Caliper::try_segment(
///     Point::new(0.5, 16.0), Point::new(28.0, 16.0), Length::new(4.0), Length::new(0.5),
/// )?;
/// assert_eq!(
///     profile(&img, &edge, CatmullRom, &Skip),
///     Err(Error::CaliperOutsideImage { sample: 0 })
/// );
/// # Ok::<(), fovea::Error>(())
/// ```
pub fn profile<I, K, B, Q>(
    image: &I,
    caliper: &Caliper,
    kernel: K,
    border: &B,
) -> Result<Profile, Error>
where
    I: ImageView,
    I::Pixel: SingleChannel + LinearPixel<Accumulator = Q> + LinearSpace,
    Q: SingleChannel,
    <Q as HomogeneousPixel>::Channel: Into<f64>,
    K: InterpolationKernel,
    B: BorderPolicy<I>,
{
    let half = (caliper.lines / 2) as f64;
    let mut values = Vec::with_capacity(caliper.samples);
    for i in 0..caliper.samples {
        let s = i as f64 * caliper.step;
        let mut sum = 0.0;
        for k in 0..caliper.lines {
            let offset = (k as f64 - half) * caliper.step;
            let [x, y] = caliper.position(s, offset);
            match sample(image, CoordinateF64::new(x, y), kernel, border) {
                Some(q) => sum += q.channel(0).into(),
                None => return Err(Error::CaliperOutsideImage { sample: i }),
            }
        }
        values.push(sum / caliper.lines as f64);
    }
    Ok(Profile {
        caliper: *caliper,
        values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::border::{Clamp, Skip};
    use crate::image::Image;
    use crate::pixel::{Mono8, MonoF32};
    use crate::transform::{Bilinear, CatmullRom};
    use core::f64::consts::PI;

    fn p(x: f64, y: f64) -> Point<Pixels> {
        Point::new(x, y)
    }

    fn l(v: f64) -> Length<Pixels> {
        Length::new(v)
    }

    fn rejected(r: Result<Caliper, Error>) -> (&'static str, Requirement) {
        match r {
            Err(Error::InvalidParameter(e)) => (e.parameter(), e.requirement()),
            other => panic!("expected a parameter error, got {other:?}"),
        }
    }

    #[test]
    fn a_segment_samples_its_whole_length() {
        let cal = Caliper::try_segment(p(0.0, 0.0), p(3.0, 4.0), l(1.0), l(0.5)).unwrap();
        assert_eq!(cal.length(), l(5.0));
        assert_eq!(cal.samples(), 11);
        assert_eq!(cal.lines(), 3);
        assert_eq!(cal.point_at(l(5.0)), p(3.0, 4.0));
        // 0.3 / 0.1 is 2.9999999999999996 in f64, and the end still counts.
        let fine = Caliper::try_segment(p(0.0, 0.0), p(0.3, 0.0), l(0.0), l(0.1)).unwrap();
        assert_eq!(fine.samples(), 4);
        assert_eq!(fine.lines(), 1);
    }

    #[test]
    fn lines_across_lie_on_the_normal() {
        let cal = Caliper::try_segment(p(10.0, 10.0), p(10.0, 20.0), l(2.0), l(1.0)).unwrap();
        // The path points down (+y); its normal points to -x.
        let a = cal.position(0.0, -1.0);
        let b = cal.position(0.0, 1.0);
        assert_eq!(a, [11.0, 10.0]);
        assert_eq!(b, [9.0, 10.0]);
    }

    #[test]
    fn an_arc_runs_either_way() {
        let cw = Caliper::try_arc(p(0.0, 0.0), l(10.0), 0.0, PI, l(0.0), l(1.0)).unwrap();
        let ccw = Caliper::try_arc(p(0.0, 0.0), l(10.0), 0.0, -PI, l(0.0), l(1.0)).unwrap();
        let q = cw.point_at(l(5.0 * PI));
        assert!(q.x.abs() < 1e-12 && (q.y - 10.0).abs() < 1e-12);
        let q = ccw.point_at(l(5.0 * PI));
        assert!(q.x.abs() < 1e-12 && (q.y + 10.0).abs() < 1e-12);
        assert_eq!(cw.samples(), 32); // floor(10π) + 1
        // Offsets across an arc are radial.
        let out = cw.position(0.0, 2.0);
        assert!((out[0] - 12.0).abs() < 1e-12 && out[1].abs() < 1e-12);
    }

    #[test]
    fn invalid_descriptions_are_rejected_with_the_parameter_named() {
        let seg = |a, b, w, s| Caliper::try_segment(a, b, l(w), l(s));
        assert_eq!(
            rejected(seg(p(f64::NAN, 0.0), p(1.0, 0.0), 1.0, 1.0)),
            ("start", Requirement::Finite)
        );
        assert_eq!(
            rejected(seg(p(1.0, 1.0), p(1.0, 1.0), 1.0, 1.0)),
            ("path length", Requirement::FinitePositive)
        );
        assert_eq!(
            rejected(seg(p(0.0, 0.0), p(1.0, 0.0), -1.0, 1.0)),
            ("width", Requirement::FiniteNonNegative)
        );
        assert_eq!(
            rejected(seg(p(0.0, 0.0), p(1.0, 0.0), 1.0, 0.0)),
            ("step", Requirement::FinitePositive)
        );
        assert_eq!(
            rejected(seg(p(0.0, 0.0), p(1e9, 0.0), 0.0, 1e-3)),
            (
                "samples along the path",
                Requirement::AtMost(MAX_CALIPER_SAMPLES)
            )
        );
        let arc = |r, sweep, w| Caliper::try_arc(p(0.0, 0.0), l(r), 0.0, sweep, l(w), l(1.0));
        assert_eq!(
            rejected(arc(0.0, 1.0, 0.0)),
            ("radius", Requirement::FinitePositive)
        );
        assert_eq!(
            rejected(arc(5.0, 0.0, 0.0)),
            ("sweep", Requirement::FiniteNonZero)
        );
        assert_eq!(
            rejected(arc(5.0, 7.0, 0.0)).0,
            "sweep",
            "a full turn or more"
        );
        assert_eq!(
            rejected(arc(5.0, 1.0, 10.0)),
            ("half width and radius", Requirement::StrictlyOrdered)
        );
    }

    #[test]
    fn the_profile_follows_a_ramp_and_averages_across() {
        // Pixel (x, y) holds 3·x + y: along x the ramp, across y it cancels
        // out in a symmetric average.
        let img = Image::generate(40, 40, |x, y| MonoF32::new(3.0 * x as f32 + y as f32));
        let cal = Caliper::try_segment(p(10.0, 20.0), p(20.0, 20.0), l(6.0), l(1.0)).unwrap();
        let prof = profile(&img, &cal, Bilinear, &Skip).unwrap();
        assert_eq!(prof.values().len(), 11);
        for (i, &v) in prof.values().iter().enumerate() {
            let expected = 3.0 * (10.0 + i as f64) + 20.0;
            assert!((v - expected).abs() < 1e-4, "{i}: {v} vs {expected}");
        }
        assert_eq!(prof.caliper(), &cal);
    }

    #[test]
    fn skip_reports_where_the_footprint_leaves_and_clamp_does_not() {
        let img = Image::fill(16, 16, Mono8::new(50));
        let cal = Caliper::try_segment(p(4.0, 8.0), p(15.5, 8.0), l(0.0), l(0.5)).unwrap();
        assert_eq!(
            profile(&img, &cal, CatmullRom, &Skip),
            Err(Error::CaliperOutsideImage { sample: 21 })
        );
        let clamped = profile(&img, &cal, CatmullRom, &Clamp).unwrap();
        assert!(clamped.values().iter().all(|&v| (v - 50.0).abs() < 1e-4));
    }
}

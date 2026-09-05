//! Edge detection: the composed Canny pipeline.
//!
//! [`canny`] is an orchestrator, not a primitive. Every stage it runs is a
//! public function in its own right, so a caller who needs a different
//! operator, border policy, or intermediate inspection can rebuild the
//! pipeline by hand instead of reaching for a configuration knob.
//!
//! The hand-built form below produces the same mask as [`canny`]. It is not
//! the same code path: `canny` fuses the direction stage into suppression
//! (the sector each gradient falls into is read from `gx`/`gy` directly,
//! skipping a full-image `atan2`), whereas composing by hand materialises
//! the angle map so you can look at it.
//!
//! ```
//! use fovea::analyze::threshold::{HysteresisThresholds, hysteresis_threshold};
//! use fovea::border::Clamp;
//! use fovea::image::{BinaryImage, Image};
//! use fovea::pixel::MonoF32;
//! use fovea::sigma;
//! use fovea::transform::{
//!     gaussian_blur, gradient_direction, gradient_magnitude, non_maximum_suppression,
//!     scharr_x, scharr_y,
//! };
//!
//! // A hand-built Canny, equivalent to `canny(&image, thresholds, sigma)`.
//! let image = Image::fill(16, 16, MonoF32::new(0.5));
//! let thresholds = HysteresisThresholds::try_new(0.05_f32, 0.15).unwrap();
//! let sigma = sigma!(1.4);
//!
//! let blurred: Image<MonoF32> = gaussian_blur(&image, sigma, &Clamp);
//! let gx = scharr_x(&blurred, &Clamp);
//! let gy = scharr_y(&blurred, &Clamp);
//! let mag = gradient_magnitude(&gx, &gy).unwrap();
//! let dir = gradient_direction(&gx, &gy).unwrap();
//! let thin = non_maximum_suppression(&mag, &dir).unwrap();
//! let edges: BinaryImage = hysteresis_threshold(&thin, thresholds);
//! ```
//!
//! ## From a mask to positions
//!
//! [`canny`] answers "which pixels are edge pixels". A measurement usually
//! needs "where is the edge", which is a position between pixels:
//! [`interpolate_edge_points`] turns the mask into a point list by fitting
//! the gradient magnitude across each kept pixel. It needs the `magnitude`
//! and gradient stages of the pipeline above, so it composes with the
//! hand-built form rather than with `canny` alone.

use core::ops::Add;

use crate::border::Clamp;
use crate::image::{BinaryImage, Image, RasterImage};
use crate::pixel::{FromLinear, LinearPixel, SingleChannel, ZeroablePixel};
use crate::transform::{
    MagnitudeChannel, gaussian_blur, gradient_magnitude, non_maximum_suppression_from_gradients,
    scharr_x, scharr_y,
};
use crate::{Coordinate, CoordinateF64, Error, Sigma};

use crate::analyze::peak::interpolate_ridge_points;
use crate::analyze::threshold::{HysteresisThresholds, hysteresis_threshold};

/// Single-scale Canny edge detector.
///
/// Runs the canonical Canny pipeline and returns a thinned, hysteresis-linked
/// edge mask:
///
/// ```text
/// gaussian_blur(σ) → Scharr Gx, Gy → magnitude + direction
///    → non-maximum suppression → hysteresis threshold → BinaryImage
/// ```
///
/// `sigma` is a true Gaussian standard deviation (it parameterises
/// [`gaussian_blur`]); larger values smooth away more detail before
/// differentiation. `thresholds` carries the absolute gradient-magnitude
/// pair applied after suppression: a pixel survives iff its magnitude is
/// `>= low` **and** its 8-connected ridge component reaches a `>= high`
/// pixel. Because the blur preserves brightness, these thresholds keep a
/// stable, kernel-independent meaning across `sigma`.
///
/// Works for any single-channel input whose linear accumulator is a float
/// pixel — `Mono8`, `Mono16`, `Mono<BITS>` and `MonoF32` accumulate in
/// `MonoF32`; `Mono32`, `Mono64` and `MonoF64` accumulate in `MonoF64`. The
/// thresholds are taken in `f32` for ergonomics and widened to the
/// accumulator's channel as needed; that widening is order-preserving, so
/// the pair stays valid without a second check.
///
/// Scharr is used for the gradient (better rotational symmetry than Sobel)
/// and [`Clamp`] for every border (so the output keeps the input size). To
/// vary any stage — a Sobel gradient, a different border, an L1 magnitude,
/// or to inspect an intermediate — compose the public stage functions
/// directly; see the [module documentation](self).
///
/// Both parameters are invariant-carrying types: a σ literal uses
/// [`sigma!`](crate::sigma), which checks it at compile time, and every
/// other value uses the matching `try_new`, so an invalid σ or a misordered
/// threshold pair is caught where it is produced, not here.
///
/// # Panics
///
/// Panics if σ's derived kernel radius exceeds
/// [`MAX_RADIUS`](crate::image::MAX_RADIUS) (via [`gaussian_blur`]). The
/// threshold relation cannot fail here, because it is carried by
/// [`HysteresisThresholds`].
///
/// # Example
///
/// ```
/// use fovea::analyze::edge::canny;
/// use fovea::analyze::threshold::HysteresisThresholds;
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::MonoF32;
/// use fovea::sigma;
///
/// // A vertical black/white step edge in a 6×4 image.
/// let image = Image::generate(6, 4, |x, _| {
///     MonoF32::new(if x < 3 { 0.0 } else { 1.0 })
/// });
///
/// let thresholds = HysteresisThresholds::try_new(0.10_f32, 0.30).unwrap();
/// let edges = canny(&image, thresholds, sigma!(1.0));
///
/// // The response is a thin edge at the boundary (the step sits between
/// // columns 2 and 3, so the kept ridge is one or two columns wide there).
/// let columns: Vec<usize> = (0..edges.width())
///     .filter(|&x| (0..edges.height()).any(|y| edges.pixel_at(x, y)))
///     .collect();
/// assert!(!columns.is_empty());
/// assert!(columns.iter().all(|&x| x == 2 || x == 3), "{columns:?}");
/// ```
#[must_use]
pub fn canny<I, P, Acc>(
    image: &I,
    thresholds: HysteresisThresholds<f32>,
    sigma: Sigma,
) -> BinaryImage
where
    I: RasterImage<Pixel = P>,
    P: Copy + LinearPixel<f32, Accumulator = Acc>,
    Acc: Copy
        + Default
        + ZeroablePixel
        + SingleChannel
        + FromLinear<Acc>
        + LinearPixel<f32, Accumulator = Acc>
        + Add<Output = Acc>,
    Acc::Channel: PartialOrd + Copy + core::fmt::Debug + From<f32> + MagnitudeChannel,
    f64: From<Acc::Channel>,
{
    let blurred: Image<Acc> = gaussian_blur(image, sigma, &Clamp);
    let gx = scharr_x(&blurred, &Clamp);
    let gy = scharr_y(&blurred, &Clamp);

    // gx / gy are produced from the same blurred image, so their sizes match
    // by construction — the `Result` cannot be `Err` here.
    let magnitude = gradient_magnitude(&gx, &gy).expect("gx and gy share a size");

    // Suppression only needs the *sector* each gradient falls into, so it
    // reads `gx`/`gy` directly rather than materialising an `atan2`
    // direction map whose precision is then discarded. The staged
    // `gradient_direction` → `non_maximum_suppression` form is public and
    // yields the same mask; see the module docs.
    let thinned = non_maximum_suppression_from_gradients(&magnitude, &gx, &gy);
    // `Acc::Channel` is `f32` or `f64`, because `MagnitudeChannel` is
    // sealed over exactly those two, so `From<f32>` is the identity or an
    // exact widening and the `low <= high` relation survives it. That is
    // what lets the pair be re-typed rather than re-validated.
    hysteresis_threshold(
        &thinned,
        thresholds.map_monotone(<Acc::Channel as From<f32>>::from),
    )
}

/// Interpolated edge positions for the `true` pixels of `mask`.
///
/// The measurement form of an edge result. [`canny`] reports a set of
/// pixels, which quantizes every edge position to the pixel grid; this fits
/// the gradient magnitude across each kept pixel and reports where the
/// crest actually falls, removing up to half a pixel of that quantization.
/// Fitting is [`interpolate_ridge_points`], one 1-D fit per site along that
/// site's own gradient, and everything it documents about the fit applies
/// here.
///
/// Points come out in raster order, one per kept pixel whose fit succeeded.
/// A pixel is dropped rather than reported at its integer position when the
/// fit is refused, which happens on the image border and where the
/// magnitude is flat across the site. Use
/// [`interpolate_ridge_points`] directly if the correspondence with the
/// input pixels has to be preserved.
///
/// # `magnitude` is the unthinned magnitude
///
/// Pass the [`gradient_magnitude`] output, **not** the
/// [`non_maximum_suppression`](crate::transform::non_maximum_suppression)
/// output. Suppression zeroes exactly the two neighbours each fit reads, so
/// a thinned map leaves every site looking like an isolated spike and every
/// point unmoved from its pixel centre: the wrong answer, quietly. The
/// thinned map's job here is choosing the sites, which is what `mask`
/// already carries.
///
/// The gradient pair must be the same `gx` / `gy` the magnitude was built
/// from. Nothing can check that, and a mismatched pair fits along the wrong
/// axis.
///
/// # Errors
///
/// Returns [`Error::SizeMismatch`] if `mask`, `magnitude`, `gx` and `gy` do
/// not all share a size. Separately produced images, so this is a
/// data-dependent relation rather than a caller precondition.
///
/// # Example
///
/// ```
/// use fovea::analyze::edge::{canny, interpolate_edge_points};
/// use fovea::analyze::threshold::HysteresisThresholds;
/// use fovea::border::Clamp;
/// use fovea::image::Image;
/// use fovea::pixel::MonoF32;
/// use fovea::sigma;
/// use fovea::transform::{gaussian_blur, gradient_magnitude, scharr_x, scharr_y};
///
/// // A step from black to white between columns 5 and 6, so the edge is
/// // at x = 5.5 and no pixel centre is on it.
/// let image: Image<MonoF32> =
///     Image::generate(12, 6, |x, _| MonoF32::new(if x < 6 { 0.0 } else { 1.0 }));
///
/// let sigma = sigma!(1.0);
/// let thresholds = HysteresisThresholds::try_new(0.10_f32, 0.30).unwrap();
/// let mask = canny(&image, thresholds, sigma);
///
/// // The same gradient stages `canny` runs internally.
/// let blurred: Image<MonoF32> = gaussian_blur(&image, sigma, &Clamp);
/// let gx = scharr_x(&blurred, &Clamp);
/// let gy = scharr_y(&blurred, &Clamp);
/// let magnitude = gradient_magnitude(&gx, &gy)?;
///
/// let points = interpolate_edge_points(&mask, &magnitude, &gx, &gy)?;
/// assert!(!points.is_empty());
/// for p in &points {
///     assert!((p.x - 5.5).abs() < 1e-4, "{p:?}");
/// }
/// # Ok::<(), fovea::Error>(())
/// ```
pub fn interpolate_edge_points<IB, IM, IX, IY, P>(
    mask: &IB,
    magnitude: &IM,
    gx: &IX,
    gy: &IY,
) -> Result<Vec<CoordinateF64>, Error>
where
    IB: RasterImage<Pixel = bool>,
    IM: RasterImage<Pixel = P>,
    IX: RasterImage<Pixel = P>,
    IY: RasterImage<Pixel = P>,
    P: SingleChannel,
    f64: From<P::Channel>,
{
    if mask.size() != magnitude.size() {
        return Err(Error::SizeMismatch {
            expected: magnitude.size(),
            actual: mask.size(),
        });
    }

    let sites = (0..mask.height()).flat_map(|y| {
        mask.row(y)
            .iter()
            .enumerate()
            .filter(|&(_, &on)| on)
            .map(move |(x, _)| Coordinate::new(x, y))
    });

    Ok(interpolate_ridge_points(sites, magnitude, gx, gy)?
        .into_iter()
        .flatten()
        .collect())
}

#[cfg(test)]
mod tests {
    use super::{HysteresisThresholds, canny, interpolate_edge_points};
    use crate::image::{Image, ImageView, RasterImage};
    use crate::pixel::{Mono8, MonoF32, MonoF64};
    use crate::sigma;

    /// The `(low, high)` pair as one argument, so the call sites stay short.
    fn t(low: f32, high: f32) -> HysteresisThresholds<f32> {
        HysteresisThresholds::try_new(low, high).unwrap()
    }

    /// Number of `true` pixels in a binary mask.
    fn count_true(mask: &crate::image::BinaryImage) -> usize {
        (0..mask.height())
            .map(|y| mask.row(y).iter().filter(|&&b| b).count())
            .sum()
    }

    /// Columns that contain at least one edge pixel.
    fn edge_columns(mask: &crate::image::BinaryImage) -> Vec<usize> {
        (0..mask.width())
            .filter(|&x| (0..mask.height()).any(|y| mask.pixel_at(x, y)))
            .collect()
    }

    /// Assert the edge response is non-empty, confined to `allowed` columns,
    /// and thin (at most two columns wide). A step sitting *between* two pixel
    /// columns yields equal gradient on both, so the inclusive-`>=` NMS keeps
    /// one or two columns there depending on float rounding — both are correct.
    fn assert_thin_edge(mask: &crate::image::BinaryImage, allowed: &[usize]) {
        let cols = edge_columns(mask);
        assert!(!cols.is_empty(), "expected an edge, got none");
        assert!(cols.len() <= 2, "expected a thin edge, got {cols:?}");
        assert!(
            cols.iter().all(|x| allowed.contains(x)),
            "edge columns {cols:?} not within {allowed:?}",
        );
    }

    #[test]
    fn step_edge_single_response() {
        // A vertical step between x = 3 and x = 4 in an 8×6 image ⇒ a thin
        // edge confined to that boundary.
        let image = Image::generate(8, 6, |x, _| MonoF32::new(if x < 4 { 0.0 } else { 1.0 }));
        let edges = canny(&image, t(0.10, 0.30), sigma!(1.0));
        assert_thin_edge(&edges, &[3, 4]);
    }

    #[test]
    fn uniform_image_no_edges() {
        let image = Image::fill(12, 12, MonoF32::new(0.5));
        let edges = canny(&image, t(0.05, 0.15), sigma!(1.2));
        assert_eq!(count_true(&edges), 0);
    }

    #[test]
    fn noise_below_low_suppressed() {
        // A faint checkerboard whose gradients never reach `low` ⇒ no edges.
        let image = Image::generate(16, 16, |x, y| {
            MonoF32::new(if (x + y) % 2 == 0 { 0.50 } else { 0.502 })
        });
        let edges = canny(&image, t(0.10, 0.30), sigma!(1.0));
        assert_eq!(count_true(&edges), 0);
    }

    #[test]
    fn weak_edge_linked_to_strong_kept() {
        // Left half of the boundary is a strong step (0 → 1); the right half is
        // a weak step (0 → 0.1) that is collinear with it. Hysteresis should
        // bridge the weak segment to the strong one and keep the whole column,
        // while a weak edge in isolation is dropped.
        let h = 8;
        let image = Image::generate(10, h, |x, y| {
            let high_side = if y < h / 2 { 1.0 } else { 0.10 };
            MonoF32::new(if x < 5 { 0.0 } else { high_side })
        });
        // Thresholds chosen so the weak step alone is below `high` but above
        // `low`, and the strong step is above `high`.
        let edges = canny(&image, t(0.02, 0.20), sigma!(1.0));
        let cols = edge_columns(&edges);
        assert!(cols.contains(&5), "edge column present: {cols:?}");
        // The weak rows (lower half) are linked through the boundary column.
        let weak_rows_present = (h / 2..h).any(|y| edges.pixel_at(5, y));
        assert!(weak_rows_present, "weak segment linked to strong and kept");
    }

    #[test]
    fn accepts_integer_input() {
        // `Mono8` accumulates in `MonoF32`; canny accepts it directly.
        let image = Image::generate(8, 6, |x, _| Mono8::new(if x < 4 { 0 } else { 255 }));
        let edges = canny(&image, t(8.0, 30.0), sigma!(1.0));
        assert_thin_edge(&edges, &[3, 4]);
    }

    #[test]
    fn generic_over_mono_f64() {
        // The pipeline runs end to end on a 64-bit float accumulator.
        let image = Image::generate(8, 6, |x, _| MonoF64::new(if x < 4 { 0.0 } else { 1.0 }));
        let edges = canny(&image, t(0.10, 0.30), sigma!(1.0));
        assert_thin_edge(&edges, &[3, 4]);
    }

    #[test]
    fn fused_pipeline_matches_staged_composition() {
        // `canny` reads NMS sectors straight from gx/gy; the documented
        // hand-composed form goes through `gradient_direction`'s atan2 map.
        // On a pattern carrying all four sector orientations the two must
        // agree pixel for pixel.
        use crate::analyze::threshold::hysteresis_threshold;
        use crate::border::Clamp;
        use crate::transform::{
            gaussian_blur, gradient_direction, gradient_magnitude, non_maximum_suppression,
            scharr_x, scharr_y,
        };

        const N: usize = 24;
        let image = Image::generate(N, N, |x, y| {
            let (dx, dy) = (x as i32 - 12, y as i32 - 12);
            let v = if dx * dx + dy * dy < 36 {
                1.0 // a disc: every gradient orientation on its rim
            } else if x % 7 == 0 || y % 5 == 0 {
                0.6 // axis-aligned rules
            } else if x == y || x + y == N - 1 {
                0.35 // both diagonals
            } else {
                0.1
            };
            MonoF32::new(v)
        });
        let thresholds = t(0.02, 0.08);
        let sigma = sigma!(1.2);

        let fused = canny(&image, thresholds, sigma);

        let blurred: Image<MonoF32> = gaussian_blur(&image, sigma, &Clamp);
        let gx = scharr_x(&blurred, &Clamp);
        let gy = scharr_y(&blurred, &Clamp);
        let mag = gradient_magnitude(&gx, &gy).unwrap();
        let dir = gradient_direction(&gx, &gy).unwrap();
        let thin = non_maximum_suppression(&mag, &dir).unwrap();
        let staged = hysteresis_threshold(&thin, thresholds);

        assert!(count_true(&fused) > 0, "the fixture should produce edges");
        for y in 0..N {
            for x in 0..N {
                assert_eq!(
                    fused.pixel_at(x, y),
                    staged.pixel_at(x, y),
                    "fused and staged disagree at ({x},{y})"
                );
            }
        }
    }

    // An invalid sigma is unrepresentable in the `Sigma` parameter type;
    // its rejection is tested at the type's constructors in `common.rs`.

    // ── interpolate_edge_points ──────────────────────────────────────────

    use crate::CoordinateF64;
    use crate::border::Clamp;
    use crate::error::Error;
    use crate::transform::{gaussian_blur, gradient_magnitude, scharr_x, scharr_y};

    /// The mask, magnitude and gradient pair `interpolate_edge_points`
    /// takes, for a vertical step between columns `edge_x - 1` and
    /// `edge_x`, so the true edge is at `edge_x - 0.5`.
    fn step_edge_stages(
        w: usize,
        h: usize,
        edge_x: usize,
    ) -> (
        crate::image::BinaryImage,
        Image<MonoF32>,
        Image<MonoF32>,
        Image<MonoF32>,
    ) {
        let sigma = sigma!(1.0);
        let image: Image<MonoF32> = Image::generate(w, h, |x, _| {
            MonoF32::new(if x < edge_x { 0.0 } else { 1.0 })
        });
        let mask = canny(&image, t(0.10, 0.30), sigma);
        let blurred: Image<MonoF32> = gaussian_blur(&image, sigma, &Clamp);
        let gx = scharr_x(&blurred, &Clamp);
        let gy = scharr_y(&blurred, &Clamp);
        let magnitude = gradient_magnitude(&gx, &gy).unwrap();
        (mask, magnitude, gx, gy)
    }

    #[test]
    fn edge_points_land_between_the_pixel_columns() {
        // The step sits between columns 5 and 6, so no pixel centre is on
        // the edge and every interpolated point must be at x = 5.5.
        let (mask, magnitude, gx, gy) = step_edge_stages(12, 6, 6);
        let points = interpolate_edge_points(&mask, &magnitude, &gx, &gy).unwrap();
        assert!(!points.is_empty(), "the fixture should produce edges");
        for p in &points {
            assert!((p.x - 5.5).abs() < 1e-4, "{p:?}");
        }
    }

    #[test]
    fn an_edge_on_a_pixel_centre_is_not_moved() {
        // A step whose transition is centred *on* column 6 (one half-value
        // sample there) rather than between two columns. The magnitude
        // ridge is then symmetric about column 6, so the fit reports the
        // pixel centre it started from: interpolation is not a
        // perturbation, and an edge that really is on a pixel stays there.
        let sigma = sigma!(1.0);
        let image: Image<MonoF32> = Image::generate(13, 5, |x, _| {
            MonoF32::new(match x.cmp(&6) {
                core::cmp::Ordering::Less => 0.0,
                core::cmp::Ordering::Equal => 0.5,
                core::cmp::Ordering::Greater => 1.0,
            })
        });
        let mask = canny(&image, t(0.05, 0.15), sigma);
        let blurred: Image<MonoF32> = gaussian_blur(&image, sigma, &Clamp);
        let gx = scharr_x(&blurred, &Clamp);
        let gy = scharr_y(&blurred, &Clamp);
        let magnitude = gradient_magnitude(&gx, &gy).unwrap();

        let points = interpolate_edge_points(&mask, &magnitude, &gx, &gy).unwrap();
        assert!(!points.is_empty(), "the fixture should produce edges");
        for p in &points {
            assert!((p.x - 6.0).abs() < 1e-6, "{p:?}");
        }
    }

    #[test]
    fn edge_points_come_out_in_raster_order() {
        let (mask, magnitude, gx, gy) = step_edge_stages(12, 6, 6);
        let points = interpolate_edge_points(&mask, &magnitude, &gx, &gy).unwrap();
        let ys: Vec<f64> = points.iter().map(|p| p.y).collect();
        assert!(ys.windows(2).all(|w| w[0] <= w[1]), "{ys:?}");
    }

    #[test]
    fn an_empty_mask_yields_no_points() {
        let (_, magnitude, gx, gy) = step_edge_stages(12, 6, 6);
        let empty = Image::fill(12, 6, false);
        let points = interpolate_edge_points(&empty, &magnitude, &gx, &gy).unwrap();
        assert_eq!(points, Vec::<CoordinateF64>::new());
    }

    #[test]
    fn a_thinned_magnitude_leaves_every_point_on_its_pixel() {
        // The documented misuse: passing the suppressed map instead of the
        // magnitude. Suppression zeroed the two neighbours each fit reads,
        // so every site looks like an isolated spike and no point moves.
        // The failure is silent, which is why the docs pin it and this test
        // records it.
        use crate::transform::non_maximum_suppression;

        let sigma = sigma!(1.0);
        let image: Image<MonoF32> =
            Image::generate(12, 6, |x, _| MonoF32::new(if x < 6 { 0.0 } else { 1.0 }));
        let mask = canny(&image, t(0.10, 0.30), sigma);
        let blurred: Image<MonoF32> = gaussian_blur(&image, sigma, &Clamp);
        let gx = scharr_x(&blurred, &Clamp);
        let gy = scharr_y(&blurred, &Clamp);
        let magnitude = gradient_magnitude(&gx, &gy).unwrap();
        let direction = crate::transform::gradient_direction(&gx, &gy).unwrap();
        let thinned = non_maximum_suppression(&magnitude, &direction).unwrap();

        let wrong = interpolate_edge_points(&mask, &thinned, &gx, &gy).unwrap();
        assert!(!wrong.is_empty());
        for p in &wrong {
            assert_eq!(p.x, p.x.round(), "{p:?}");
        }

        // The same sites on the magnitude do move off the grid.
        let right = interpolate_edge_points(&mask, &magnitude, &gx, &gy).unwrap();
        assert!(right.iter().all(|p| (p.x - p.x.round()).abs() > 1e-6));
    }

    #[test]
    fn mismatched_stages_are_rejected() {
        let (mask, magnitude, gx, gy) = step_edge_stages(12, 6, 6);
        let small: crate::image::BinaryImage = Image::fill(11, 6, true);
        assert!(matches!(
            interpolate_edge_points(&small, &magnitude, &gx, &gy),
            Err(Error::SizeMismatch { .. })
        ));

        let wrong_gradient: Image<MonoF32> = Image::fill(11, 6, MonoF32::new(1.0));
        assert!(matches!(
            interpolate_edge_points(&mask, &magnitude, &wrong_gradient, &gy),
            Err(Error::SizeMismatch { .. })
        ));
    }
}

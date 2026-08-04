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
//! use fovea::Sigma;
//! use fovea::analyze::threshold::hysteresis_threshold;
//! use fovea::border::Clamp;
//! use fovea::image::{BinaryImage, Image};
//! use fovea::pixel::MonoF32;
//! use fovea::transform::{
//!     gaussian_blur, gradient_direction, gradient_magnitude, non_maximum_suppression,
//!     scharr_x, scharr_y,
//! };
//!
//! // A hand-built Canny, equivalent to `canny(&image, low, high, sigma)`.
//! let image = Image::fill(16, 16, MonoF32::new(0.5));
//! let (low, high, sigma) = (0.05, 0.15, Sigma::new(1.4));
//!
//! let blurred: Image<MonoF32> = gaussian_blur(&image, sigma, &Clamp);
//! let gx = scharr_x(&blurred, &Clamp);
//! let gy = scharr_y(&blurred, &Clamp);
//! let mag = gradient_magnitude(&gx, &gy).unwrap();
//! let dir = gradient_direction(&gx, &gy).unwrap();
//! let thin = non_maximum_suppression(&mag, &dir).unwrap();
//! let edges: BinaryImage = hysteresis_threshold(&thin, low, high);
//! ```

use core::ops::Add;

use crate::Sigma;
use crate::border::Clamp;
use crate::image::{BinaryImage, Image, RasterImage};
use crate::pixel::{FromLinear, LinearPixel, SingleChannel, ZeroablePixel};
use crate::transform::{
    MagnitudeChannel, gaussian_blur, gradient_magnitude, non_maximum_suppression_from_gradients,
    scharr_x, scharr_y,
};

use crate::analyze::threshold::hysteresis_threshold;

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
/// differentiation. `low` and `high` are absolute gradient-magnitude
/// thresholds applied after suppression: a pixel survives iff its magnitude
/// is `>= low` **and** its 8-connected ridge component reaches a `>= high`
/// pixel. Because the blur preserves brightness, these thresholds keep a
/// stable, kernel-independent meaning across `sigma`.
///
/// Works for any single-channel input whose linear accumulator is a float
/// pixel — `Mono8` and `MonoF32` accumulate in `MonoF32`; `Mono16` / `Mono32`
/// / `Mono64` / `MonoF64` accumulate in `MonoF64`. The thresholds are taken as
/// `f32` for ergonomics and widened to the accumulator's channel as needed.
///
/// Scharr is used for the gradient (better rotational symmetry than Sobel)
/// and [`Clamp`] for every border (so the output keeps the input size). To
/// vary any stage — a Sobel gradient, a different border, an L1 magnitude,
/// or to inspect an intermediate — compose the public stage functions
/// directly; see the [module documentation](self).
///
/// σ is the invariant-carrying [`Sigma`](crate::Sigma) type: literals use
/// `Sigma::new`, computed values `Sigma::try_new` — an invalid σ is caught
/// where it is produced, not here.
///
/// # Panics
///
/// Panics if `!(low <= high)` (via [`hysteresis_threshold`]), or if σ's
/// derived kernel radius exceeds
/// [`MAX_RADIUS`](crate::image::MAX_RADIUS) (via [`gaussian_blur`]).
///
/// # Example
///
/// ```
/// use fovea::Sigma;
/// use fovea::analyze::edge::canny;
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::MonoF32;
///
/// // A vertical black/white step edge in a 6×4 image.
/// let image = Image::generate(6, 4, |x, _| {
///     MonoF32::new(if x < 3 { 0.0 } else { 1.0 })
/// });
///
/// let edges = canny(&image, 0.10, 0.30, Sigma::new(1.0));
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
pub fn canny<I, P, Acc>(image: &I, low: f32, high: f32, sigma: Sigma) -> BinaryImage
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
    hysteresis_threshold(
        &thinned,
        <Acc::Channel as From<f32>>::from(low),
        <Acc::Channel as From<f32>>::from(high),
    )
}

#[cfg(test)]
mod tests {
    use super::canny;
    use crate::Sigma;
    use crate::image::{Image, ImageView, RasterImage};
    use crate::pixel::{Mono8, MonoF32, MonoF64};

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
        let edges = canny(&image, 0.10, 0.30, Sigma::new(1.0));
        assert_thin_edge(&edges, &[3, 4]);
    }

    #[test]
    fn uniform_image_no_edges() {
        let image = Image::fill(12, 12, MonoF32::new(0.5));
        let edges = canny(&image, 0.05, 0.15, Sigma::new(1.2));
        assert_eq!(count_true(&edges), 0);
    }

    #[test]
    fn noise_below_low_suppressed() {
        // A faint checkerboard whose gradients never reach `low` ⇒ no edges.
        let image = Image::generate(16, 16, |x, y| {
            MonoF32::new(if (x + y) % 2 == 0 { 0.50 } else { 0.502 })
        });
        let edges = canny(&image, 0.10, 0.30, Sigma::new(1.0));
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
        let edges = canny(&image, 0.02, 0.20, Sigma::new(1.0));
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
        let edges = canny(&image, 8.0, 30.0, Sigma::new(1.0));
        assert_thin_edge(&edges, &[3, 4]);
    }

    #[test]
    fn generic_over_mono_f64() {
        // The pipeline runs end to end on a 64-bit float accumulator.
        let image = Image::generate(8, 6, |x, _| MonoF64::new(if x < 4 { 0.0 } else { 1.0 }));
        let edges = canny(&image, 0.10, 0.30, Sigma::new(1.0));
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
        let (low, high, sigma) = (0.02f32, 0.08f32, Sigma::new(1.2));

        let fused = canny(&image, low, high, sigma);

        let blurred: Image<MonoF32> = gaussian_blur(&image, sigma, &Clamp);
        let gx = scharr_x(&blurred, &Clamp);
        let gy = scharr_y(&blurred, &Clamp);
        let mag = gradient_magnitude(&gx, &gy).unwrap();
        let dir = gradient_direction(&gx, &gy).unwrap();
        let thin = non_maximum_suppression(&mag, &dir).unwrap();
        let staged = hysteresis_threshold(&thin, low, high);

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
}

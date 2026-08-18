//! Peak selection: turning a scalar corner map into keypoints.
//!
//! Shared by every detector in this module — the structure-tensor family's
//! response map and the segment test's score map differ in what they measure
//! and not at all in how their maxima are picked.

use crate::analyze::peak::{Extremum, interpolate_peak};
use crate::features::Corner;
use crate::image::RasterImage;
use crate::pixel::SingleChannel;
use crate::{Coordinate, CoordinateF64};

/// Selects the local maxima of a corner map as keypoints.
///
/// A pixel is reported when it is at least `threshold` **and** is the
/// maximum of the square window of half-side `radius` around it — so
/// `radius` is the minimum separation between two reported corners. The
/// window is clipped at the image border rather than skipped, so a corner
/// against the edge of the frame can still be reported; whether the value
/// *there* is trustworthy is a property of how the map was built (see
/// [`corner_response_map`](super::corner_response_map) and
/// [`fast_score_map`](super::fast_score_map)).
///
/// Returned corners are in **raster order** (top to bottom, left to right),
/// not response order. Ranking is a separate, named step:
/// [`retain_top_n`](crate::features::retain_top_n) or
/// [`sort_by_response`](crate::features::sort_by_response).
///
/// # Plateaus and ties
///
/// Exact ties are the rule on synthetic images, and a naïve `>=` comparison
/// reports every pixel of a flat plateau while a naïve `>` reports none of
/// them. The comparison here is asymmetric instead: strictly greater than
/// neighbours *earlier* in raster order, greater or equal to *later* ones.
/// Exactly one corner then survives per group of tied pixels — the group's
/// raster-first — as long as each member of the group lies within `radius`
/// of an earlier one; two tied blobs further apart than `radius` are two
/// groups and yield two corners, which is the same rule the radius states
/// everywhere else. Nothing here depends on the order the map was scanned
/// in. A `NaN` neighbour makes every comparison false, so it suppresses
/// rather than wins.
///
/// # Example
///
/// ```
/// use fovea::features::detect::corner_peaks;
/// use fovea::features::HasPosition;
/// use fovea::image::Image;
/// use fovea::pixel::MonoF32;
///
/// // Two isolated peaks of different strength, plus sub-threshold noise.
/// let response: Image<MonoF32> = Image::generate(9, 9, |x, y| {
///     MonoF32::new(match (x, y) {
///         (2, 2) => 1.0,
///         (6, 6) => 0.5,
///         _ => 0.01,
///     })
/// });
///
/// let corners = corner_peaks(&response, 0.1, 2);
/// let positions: Vec<(f64, f64)> = corners
///     .iter()
///     .map(|c| (c.position().x, c.position().y))
///     .collect();
/// assert_eq!(positions, [(2.0, 2.0), (6.0, 6.0)]);
/// ```
#[must_use]
pub fn corner_peaks<I, P>(response: &I, threshold: f32, radius: usize) -> Vec<Corner>
where
    I: RasterImage<Pixel = P>,
    P: SingleChannel,
    P::Channel: PartialOrd + From<f32>,
    f64: From<P::Channel>,
{
    scan_peaks(response, threshold, radius)
        .into_iter()
        .map(|(at, response)| Corner::new(at, response))
        .collect()
}

/// Moves each corner to the interpolated peak of `response`, in place.
///
/// [`corner_peaks`] reports the pixel that won its neighbourhood, so every
/// position it returns is an integer and carries up to half a pixel of
/// quantization error along each axis. This fits a quadratic to the 3x3
/// neighbourhood of the response map around each corner and moves the
/// corner to the fitted vertex, which removes that half pixel. Responses
/// are left alone: the value at the vertex is not one the detector
/// measured.
///
/// Composes after the fact, like [`retain_top_n`](crate::features::retain_top_n)
/// and [`sort_by_response`](crate::features::sort_by_response), so the
/// detectors keep one output shape and interpolation stays a named,
/// skippable step.
///
/// Returns how many corners were interpolated. That counts fits that
/// succeeded, which includes a perfectly symmetric peak whose vertex is its
/// own pixel centre, so it is a report about the fit rather than about
/// distance travelled. A corner is left exactly where it was, rather than
/// dropped, when its fit is refused, which happens when
///
/// * it sits on the border of the map, where the 3x3 window is incomplete
///   (and [`corner_peaks`] does report border corners), or
/// * the response around it is flat, a saddle, or curves the wrong way, on
///   the conditions [`interpolate_peak`] lists.
///
/// # The corners and the map must be in the same frame
///
/// `response` is the map the corners were found on, and the positions are
/// read as coordinates *in that map*. This is therefore a step for
/// [`corner_peaks`] output, before any lift into another frame: passing
/// corners from [`detect_corners_in_level`](super::detect_corners_in_level)
/// or [`fast_in_level`](super::fast_in_level) together with the level's own
/// response map interpolates them at the wrong sites, because those
/// functions already report in the base-image frame. A [`Corner`] does not
/// carry its frame, so nothing here can detect that.
///
/// # A segment-test map is graded only at the edges of its plateaus
///
/// [`fast_score_map`](super::fast_score_map)'s score saturates: several
/// pixels around one corner reach the identical full contrast, and
/// [`corner_peaks`] reports the raster-first of that tied cluster. Its 3x3
/// window then straddles the *edge* of the plateau rather than sitting
/// inside it, so the fit is not refused. It moves the corner toward the
/// plateau's interior, which undoes part of the raster-first bias and is
/// worth having.
///
/// It is not a substitute for measuring the corner properly, because the
/// move is bounded by the one-pixel window: a tied cluster wider than three
/// pixels keeps a residual this fit cannot see, and the segment test's
/// documented "up to two pixels from the geometric corner" is exactly that
/// case. The structure tensor's
/// [`corner_response_map`](super::corner_response_map) is the graded surface
/// this is really for.
///
/// # Example
///
/// ```
/// use fovea::features::HasPosition;
/// use fovea::features::detect::{corner_peaks, interpolate_corners};
/// use fovea::image::Image;
/// use fovea::pixel::MonoF32;
///
/// // A response peak whose crest is at (3.25, 4.0), not on a pixel.
/// let response: Image<MonoF32> = Image::generate(9, 9, |x, y| {
///     let (dx, dy) = (x as f64 - 3.25, y as f64 - 4.0);
///     MonoF32::new((1.0 - 0.1 * (dx * dx + dy * dy)) as f32)
/// });
///
/// let mut corners = corner_peaks(&response, 0.5, 2);
/// assert_eq!(corners[0].position().x, 3.0);
///
/// assert_eq!(interpolate_corners(&mut corners, &response), 1);
/// assert!((corners[0].position().x - 3.25).abs() < 1e-3, "{corners:?}");
/// assert!((corners[0].position().y - 4.00).abs() < 1e-3, "{corners:?}");
/// ```
pub fn interpolate_corners<I, P>(corners: &mut [Corner], response: &I) -> usize
where
    I: RasterImage<Pixel = P>,
    P: SingleChannel,
    f64: From<P::Channel>,
{
    let mut fitted = 0;
    for corner in corners {
        let Some(at) = pixel_site(corner.at) else {
            continue;
        };
        if let Some(vertex) = interpolate_peak(response, at, Extremum::Maximum) {
            corner.at = vertex;
            fitted += 1;
        }
    }
    fitted
}

/// The pixel a corner position names, or `None` if it names none.
///
/// Positions from [`corner_peaks`] are whole numbers, so this is a cast in
/// the ordinary case; rounding covers a caller who interpolated once
/// already, and the sign and finiteness tests are what keep the cast from
/// wrapping a negative or saturating a NaN into a valid-looking index.
#[inline]
fn pixel_site(at: CoordinateF64) -> Option<Coordinate> {
    let (x, y) = (at.x.round(), at.y.round());
    if x >= 0.0 && y >= 0.0 && x.is_finite() && y.is_finite() {
        Some(Coordinate::new(x as usize, y as usize))
    } else {
        None
    }
}

/// Collects the local maxima of `response` as `(position, response)` pairs in
/// the map's own coordinate frame.
///
/// The shared engine behind [`corner_peaks`] (which reports positions as
/// found) and the `_in_level` detectors (which lift them into the base-image
/// frame). Keeping it separate is what stops the level variants from having
/// to re-interpret an already-built [`Corner`]'s position as local
/// coordinates.
pub(super) fn scan_peaks<I, P>(
    response: &I,
    threshold: f32,
    radius: usize,
) -> Vec<(CoordinateF64, f32)>
where
    I: RasterImage<Pixel = P>,
    P: SingleChannel,
    P::Channel: PartialOrd + From<f32>,
    f64: From<P::Channel>,
{
    let threshold = <P::Channel as From<f32>>::from(threshold);
    let (w, h) = (response.width(), response.height());
    let mut peaks = Vec::new();

    for y in 0..h {
        for x in 0..w {
            let value = response.row(y)[x].channel(0);
            // Both tests are written in the positive, so a NaN response —
            // which compares false against everything — is excluded rather
            // than reported as an infinitely strong corner.
            if value >= threshold && is_local_max(response, x, y, radius, value) {
                let at = CoordinateF64::new(x as f64, y as f64);
                peaks.push((at, f64::from(value) as f32));
            }
        }
    }
    peaks
}

/// Whether `value` at `(x, y)` is the maximum of its clipped
/// `(2·radius + 1)²` window, with ties resolved in favour of the pixel
/// earlier in raster order.
#[inline]
fn is_local_max<I, P>(response: &I, x: usize, y: usize, radius: usize, value: P::Channel) -> bool
where
    I: RasterImage<Pixel = P>,
    P: SingleChannel,
    P::Channel: PartialOrd,
{
    let (w, h) = (response.width(), response.height());
    let y_lo = y.saturating_sub(radius);
    let y_hi = (y + radius + 1).min(h);
    let x_lo = x.saturating_sub(radius);
    let x_hi = (x + radius + 1).min(w);

    for ny in y_lo..y_hi {
        let row = response.row(ny);
        for (offset, pixel) in row[x_lo..x_hi].iter().enumerate() {
            let nx = x_lo + offset;
            if nx == x && ny == y {
                continue;
            }
            let neighbour = pixel.channel(0);
            // Earlier in raster order ⇒ strict, later ⇒ inclusive. A flat
            // plateau then keeps exactly its raster-first pixel.
            let earlier = ny < y || (ny == y && nx < x);
            let survives = if earlier {
                value > neighbour
            } else {
                value >= neighbour
            };
            if !survives {
                return false;
            }
        }
    }
    true
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::{HasPosition, HasResponse};
    use crate::image::Image;
    use crate::pixel::MonoF32;

    /// The `(x, y)` positions of `corners`, for set comparisons.
    fn positions(corners: &[Corner]) -> Vec<(f64, f64)> {
        corners
            .iter()
            .map(|c| (c.position().x, c.position().y))
            .collect()
    }

    #[test]
    fn peaks_are_returned_in_raster_order() {
        let response: Image<MonoF32> = Image::generate(9, 9, |x, y| {
            MonoF32::new(match (x, y) {
                (6, 2) => 0.4, // weaker, but earlier in raster order
                (2, 6) => 0.9,
                _ => 0.0,
            })
        });
        let corners = corner_peaks(&response, 0.1, 2);
        assert_eq!(positions(&corners), [(6.0, 2.0), (2.0, 6.0)]);
        assert_eq!(corners[0].response(), 0.4);
    }

    #[test]
    fn peaks_below_the_threshold_are_dropped() {
        let response: Image<MonoF32> = Image::generate(9, 9, |x, y| {
            MonoF32::new(if (x, y) == (4, 4) { 0.05 } else { 0.0 })
        });
        assert!(corner_peaks(&response, 0.1, 2).is_empty());
        // The threshold is inclusive.
        assert_eq!(corner_peaks(&response, 0.05, 2).len(), 1);
    }

    #[test]
    fn a_plateau_yields_exactly_one_peak() {
        // Four exactly tied pixels: `>=` everywhere would report all four,
        // `>` everywhere none. The raster-first one wins.
        let response: Image<MonoF32> = Image::generate(9, 9, |x, y| {
            MonoF32::new(if (3..5).contains(&x) && (3..5).contains(&y) {
                1.0
            } else {
                0.0
            })
        });
        let corners = corner_peaks(&response, 0.5, 2);
        assert_eq!(corners.len(), 1);
        assert_eq!(corners[0].position(), CoordinateF64::new(3.0, 3.0));
    }

    #[test]
    fn a_uniform_response_map_yields_a_single_peak() {
        // Every pixel ties and every pixel has an earlier tied neighbour
        // within the radius, except the very first — so the whole image is
        // one group and reports one corner, not 64 and not none.
        let response: Image<MonoF32> = Image::fill(8, 8, MonoF32::new(1.0));
        let corners = corner_peaks(&response, 0.5, 2);
        assert_eq!(corners.len(), 1);
        assert_eq!(corners[0].position(), CoordinateF64::new(0.0, 0.0));
    }

    #[test]
    fn tied_groups_further_apart_than_the_radius_are_separate_peaks() {
        // The other half of the plateau rule: ties are grouped by the same
        // radius that separates ordinary peaks, so two equal-valued blobs
        // five pixels apart report one corner each.
        let response: Image<MonoF32> = Image::generate(12, 3, |x, y| {
            let tied = (y == 1) && ((2..4).contains(&x) || (8..10).contains(&x));
            MonoF32::new(if tied { 1.0 } else { 0.0 })
        });
        let corners = corner_peaks(&response, 0.5, 2);
        assert_eq!(positions(&corners), [(2.0, 1.0), (8.0, 1.0)]);
    }

    #[test]
    fn the_suppression_radius_sets_the_minimum_separation() {
        // Two peaks four pixels apart: kept at radius 3, merged at radius 4.
        let response: Image<MonoF32> = Image::generate(12, 3, |x, y| {
            MonoF32::new(match (x, y) {
                (3, 1) => 1.0,
                (7, 1) => 0.8,
                _ => 0.0,
            })
        });
        assert_eq!(corner_peaks(&response, 0.1, 3).len(), 2);
        let merged = corner_peaks(&response, 0.1, 4);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].position(), CoordinateF64::new(3.0, 1.0));
    }

    #[test]
    fn a_peak_against_the_border_is_reported() {
        // The window is clipped, not skipped: a corner at (0, 0) survives.
        let response: Image<MonoF32> = Image::generate(6, 6, |x, y| {
            MonoF32::new(if (x, y) == (0, 0) { 1.0 } else { 0.0 })
        });
        let corners = corner_peaks(&response, 0.5, 2);
        assert_eq!(corners.len(), 1);
        assert_eq!(corners[0].position(), CoordinateF64::new(0.0, 0.0));
    }

    #[test]
    fn a_nan_response_neither_wins_nor_survives() {
        let response: Image<MonoF32> = Image::generate(7, 7, |x, y| {
            MonoF32::new(match (x, y) {
                (3, 3) => f32::NAN,
                (5, 5) => 1.0,
                _ => 0.0,
            })
        });
        // The NaN pixel fails its own threshold test, and the genuine peak
        // is far enough away to be unaffected.
        assert_eq!(positions(&corner_peaks(&response, 0.5, 1)), [(5.0, 5.0)]);
    }

    #[test]
    fn a_nan_neighbour_suppresses_a_peak() {
        let response: Image<MonoF32> = Image::generate(7, 7, |x, y| {
            MonoF32::new(match (x, y) {
                (3, 3) => 1.0,
                (4, 3) => f32::NAN,
                _ => 0.0,
            })
        });
        assert!(corner_peaks(&response, 0.5, 1).is_empty());
    }

    #[test]
    fn an_empty_response_map_yields_no_peaks() {
        let response: Image<MonoF32> = Image::fill(5, 5, MonoF32::new(0.0));
        assert!(corner_peaks(&response, 0.5, 2).is_empty());
    }

    // ── interpolate_corners ──────────────────────────────────────────────

    /// A response paraboloid with its crest at `(cx, cy)`.
    fn crest_at(w: usize, h: usize, cx: f64, cy: f64) -> Image<MonoF32> {
        Image::generate(w, h, |x, y| {
            let (dx, dy) = (x as f64 - cx, y as f64 - cy);
            MonoF32::new((1.0 - 0.05 * (dx * dx + dy * dy)) as f32)
        })
    }

    #[test]
    fn interpolation_moves_a_corner_onto_the_response_crest() {
        let response = crest_at(11, 11, 4.25, 5.6);
        let mut corners = corner_peaks(&response, 0.5, 2);
        assert_eq!(positions(&corners), [(4.0, 6.0)], "the discrete winner");

        assert_eq!(interpolate_corners(&mut corners, &response), 1);
        let at = corners[0].position();
        assert!((at.x - 4.25).abs() < 1e-3, "{at:?}");
        assert!((at.y - 5.60).abs() < 1e-3, "{at:?}");
    }

    #[test]
    fn interpolation_leaves_the_response_alone() {
        let response = crest_at(11, 11, 4.25, 5.6);
        let mut corners = corner_peaks(&response, 0.5, 2);
        let before = corners[0].response();
        interpolate_corners(&mut corners, &response);
        assert_eq!(
            corners[0].response(),
            before,
            "the value at the vertex is not one the detector measured",
        );
    }

    #[test]
    fn a_corner_against_the_border_stays_put() {
        // `corner_peaks` reports border corners by clipping its window;
        // the 3x3 fit cannot, so the corner is left where it was rather
        // than dropped.
        let response: Image<MonoF32> = Image::generate(6, 6, |x, y| {
            MonoF32::new(if (x, y) == (0, 0) { 1.0 } else { 0.0 })
        });
        let mut corners = corner_peaks(&response, 0.5, 2);
        assert_eq!(positions(&corners), [(0.0, 0.0)]);

        assert_eq!(interpolate_corners(&mut corners, &response), 0);
        assert_eq!(positions(&corners), [(0.0, 0.0)]);
    }

    #[test]
    fn a_saturated_plateau_is_pulled_toward_its_interior() {
        // The segment-test case. `corner_peaks` reports the raster-first
        // pixel of the tied block, whose 3x3 window straddles the plateau's
        // edge rather than sitting inside it, so there *is* a vertex to fit
        // and it lies toward the block's centre at (4, 4).
        let response: Image<MonoF32> = Image::generate(9, 9, |x, y| {
            MonoF32::new(if (3..6).contains(&x) && (3..6).contains(&y) {
                1.0
            } else {
                0.0
            })
        });
        let mut corners = corner_peaks(&response, 0.5, 3);
        assert_eq!(positions(&corners), [(3.0, 3.0)]);

        assert_eq!(interpolate_corners(&mut corners, &response), 1);
        let at = corners[0].position();
        assert!(at.x > 3.0 && at.y > 3.0, "moved inward: {at:?}");
        // And bounded by the sampled window, so it does not reach the
        // centre: the move is a partial correction, not a measurement.
        assert!(at.x < 4.0 && at.y < 4.0, "bounded by the window: {at:?}");
    }

    #[test]
    fn a_corner_inside_a_flat_region_is_refused() {
        // The genuinely flat window, which `corner_peaks` never produces
        // (its plateau rule always lands on a tied group's boundary) but a
        // caller supplying its own positions can.
        let response: Image<MonoF32> = Image::fill(7, 7, MonoF32::new(1.0));
        let mut corners = vec![Corner::new(CoordinateF64::new(3.0, 3.0), 1.0)];
        assert_eq!(interpolate_corners(&mut corners, &response), 0);
        assert_eq!(corners[0].position(), CoordinateF64::new(3.0, 3.0));
    }

    #[test]
    fn the_count_reports_only_the_fits_that_succeeded() {
        // One interpolable crest and one border corner in the same call.
        let response: Image<MonoF32> = Image::generate(13, 7, |x, y| {
            let (dx, dy) = (x as f64 - 8.4, y as f64 - 3.0);
            let crest = 1.0 - 0.08 * (dx * dx + dy * dy);
            MonoF32::new(if (x, y) == (0, 0) {
                0.9
            } else {
                crest.max(0.0) as f32
            })
        });
        let mut corners = corner_peaks(&response, 0.5, 2);
        assert_eq!(corners.len(), 2, "{corners:?}");

        assert_eq!(interpolate_corners(&mut corners, &response), 1);
        // Raster order: the border corner first, unmoved; the crest second.
        assert_eq!(corners[0].position(), CoordinateF64::new(0.0, 0.0));
        assert!((corners[1].position().x - 8.4).abs() < 1e-3, "{corners:?}");
    }

    #[test]
    fn interpolating_no_corners_is_no_work() {
        let response = crest_at(9, 9, 4.0, 4.0);
        assert_eq!(interpolate_corners(&mut [], &response), 0);
    }

    #[test]
    fn a_position_off_the_map_is_left_alone() {
        // The frame hazard the docs describe, in its detectable form: a
        // position that names no pixel of this map cannot be interpolated
        // against it. (A *wrong* position that happens to name a pixel is
        // not detectable, which is why the requirement is documented.)
        let response = crest_at(9, 9, 4.25, 4.0);
        let mut corners = vec![
            Corner::new(CoordinateF64::new(40.0, 40.0), 0.9),
            Corner::new(CoordinateF64::new(-3.0, 2.0), 0.9),
        ];
        assert_eq!(interpolate_corners(&mut corners, &response), 0);
        assert_eq!(corners[0].position(), CoordinateF64::new(40.0, 40.0));
        assert_eq!(corners[1].position(), CoordinateF64::new(-3.0, 2.0));
    }
}

//! Corner refinement: moving a detected corner to the point the gradient
//! field says the corner is.
//!
//! The user-facing overview of the two-step localization story (interpolate
//! for precision, refine for accuracy) lives on the parent module.

use crate::error::Error;
use crate::features::Corner;
use crate::image::RasterImage;
use crate::pixel::SingleChannel;
use crate::{Coordinate, CoordinateF64};

use super::peaks::{NmsRadius, pixel_site};

/// How often the window may be re-centred before the fit is refused.
///
/// One re-centring is the ordinary case (the detected pixel and the corner
/// straddle a rounding boundary), two happen when the detection was a pixel
/// or more off. An estimate still landing in a new window after three moves
/// has wandered at least three pixels from where it started, which is not
/// convergence, and the honest answer is that these gradients do not locate
/// the corner.
const MAX_RECENTRES: usize = 3;

/// The relative rank test on the normal matrix: refuse unless
/// `det > RANK_EPSILON * trace^2`.
///
/// `det / trace^2` is dimensionless (both scale with the fourth power of the
/// gradients) and at most 1/4. A window that sees a single straight edge has
/// every gradient parallel, so its determinant is zero in exact arithmetic
/// and rounding noise in floats; a window that sees two edges of a genuine
/// corner sits many orders of magnitude above this bound.
const RANK_EPSILON: f64 = 1e-12;

/// Moves each corner to the least-squares intersection of the edge lines in
/// its gradient neighbourhood, in place.
///
/// This is the **accuracy** step, and it is deliberately not
/// [`interpolate_corners`](super::interpolate_corners). Interpolation locates
/// the extremum of the response map, and both detector families put that
/// extremum in the wrong place: the structure tensor's window drags its peak
/// inward as σ grows (a full pixel along both axes by σ = 1.6 on a clean
/// synthetic corner), and the segment test's saturated score reports the
/// raster-first pixel of a tied plateau, up to two pixels out. Interpolating
/// either map yields a precise position on the *displaced* surface. This
/// function never reads a response map. It solves, per corner,
///
/// ```text
/// ĉ = argmin_c Σ ( ∇f(x, y)ᵀ · ((x, y) − c) )²
/// ```
///
/// over the window of half-side `radius` around the corner: near a corner,
/// every pixel's gradient is perpendicular to the edge line through that
/// pixel, so each gradient sample constrains the corner to one line and the
/// corner is the least-squares intersection of all of them (Förstner's
/// observation, the same one behind OpenCV's `cornerSubPix`). Each pixel's
/// influence scales with its squared gradient magnitude, so flat pixels
/// contribute nothing and no explicit mask is needed.
///
/// `gx` and `gy` are the horizontal and vertical **gradients of the image**
/// (for example [`sobel_x`](crate::transform::sobel_x) /
/// [`sobel_y`](crate::transform::sobel_y) with a
/// [`Clamp`](crate::border::Clamp) border, the pair the tensor family's
/// detectors compute internally), not the windowed tensor products and not a
/// gradient magnitude. For a segment-test corner, which never computed a
/// gradient, take the same pair from the image.
///
/// Composes after the fact like [`interpolate_corners`](super::interpolate_corners),
/// [`retain_top_n`](crate::features::retain_top_n) and
/// [`sort_by_response`](crate::features::sort_by_response): in place, over
/// whatever corners the caller kept. Refinement and interpolation answer
/// different questions and need not be combined; for corner *positions*,
/// refinement alone supersedes interpolation, while interpolation remains
/// the right tool for surfaces that are not corners of an intensity image
/// (a template-match score, a histogram bin).
///
/// Returns how many corners were refined. Responses are left alone: the
/// refined position is not a place the detector scored.
///
/// # Choosing the radius
///
/// The window must be large enough to see both edges *and* the true corner,
/// which sits inward-bias-plus-operator-support away from the detected
/// pixel. In practice: about `2σ` (rounded up) for a structure-tensor
/// detection with window `σ`, and 3 for a segment-test detection, whose
/// plateau bias is up to two pixels. Below 2 the window rarely clears the
/// gradient operator's own support; much beyond the feature's isolation
/// distance it starts to see neighbouring structure, whose edge lines then
/// vote too.
///
/// # The window re-centres, then stops
///
/// The first window is centred on the corner's nearest pixel. When the
/// solved position rounds to a different pixel, the window is re-centred
/// there and solved again, so a detection a pixel or two off ends with a
/// window that actually surrounds the corner; the window is integer-aligned,
/// so a solve from an unchanged centre is a fixed point and iterating it
/// further would change nothing. At most three re-centrings are attempted,
/// after which the fit is refused as divergent.
///
/// # A corner is left in place when its fit is refused
///
/// As with [`interpolate_corners`](super::interpolate_corners), a refused corner keeps its position and
/// is not counted. A fit is refused when
///
/// * the corner's position does not name a pixel of the gradient images
///   (negative, non-finite, or off the frame),
/// * the window of half-side `radius` around the (possibly re-centred)
///   pixel is not completely inside the frame, since gradients outside it
///   do not exist and a border gradient under [`Clamp`](crate::border::Clamp)
///   is extrapolated rather than measured,
/// * the window's gradients do not describe two edge directions: a flat
///   window, a single straight edge (whose gradients are all parallel, so
///   the normal matrix is rank one and the intersection is undefined), or
///   any `NaN` sample. The test is written in the positive, so `NaN`
///   refuses rather than propagates,
/// * the solved position falls outside the window that produced it, which
///   means these samples do not locate the corner, or
/// * the solve keeps re-centring past the cap above.
///
/// # The corners and the gradients must be in the same frame
///
/// The same hazard [`interpolate_corners`](super::interpolate_corners)
/// documents: positions are read as
/// coordinates in `gx` / `gy`'s frame, so this is a step for
/// [`corner_peaks`](super::corner_peaks) / [`fast`](super::fast) output
/// before any lift, never for `_in_level` output, which already reports in
/// the base-image frame. A [`Corner`] does not carry its frame, so nothing
/// here can detect the mismatch.
///
/// # What remains after refinement
///
/// The gradient operator has support of its own, and pixels within that
/// support of the corner see both edges at once; their gradients are
/// perpendicular to neither edge, and their constraint lines miss the
/// corner slightly. On a clean unit-contrast synthetic corner with Sobel
/// gradients the resulting residual is at most a few hundredths of a pixel
/// toward the corner's interior (regression-tested at 0.05 px across window
/// σ from 0.8 to 2.0, measured at most 0.04 px per axis). That is the noise
/// floor of this method, two orders of magnitude below the biases it
/// removes, not a bound the caller needs to design around.
///
/// # Errors (Tier 2)
///
/// Returns [`Error::SizeMismatch`] if `gx` and `gy` differ in dimensions
/// (two separately obtained runtime images, the same relation
/// [`StructureTensor::from_gradients`](super::StructureTensor::from_gradients)
/// reports). The fitting-window radius carries its own at-least-one-pixel
/// invariant as an [`NmsRadius`](super::NmsRadius) — a single-pixel window
/// has a rank-one normal matrix and would silently refuse every corner —
/// so it cannot fail here.
///
/// # Example
///
/// ```
/// use fovea::border::Clamp;
/// use fovea::features::HasPosition;
/// use fovea::features::detect::{detect_corners, refine_corners, CornerParams, NmsRadius, ShiTomasi};
/// use fovea::image::Image;
/// use fovea::pixel::MonoF32;
/// use fovea::sigma;
/// use fovea::transform::{sobel_x, sobel_y};
///
/// // A white square whose top-left geometric corner is at (7.5, 7.5).
/// let image: Image<MonoF32> = Image::generate(24, 24, |x, y| {
///     MonoF32::new(if (8..16).contains(&x) && (8..16).contains(&y) { 1.0 } else { 0.0 })
/// });
///
/// // At this window σ the detected peak has drifted inward, to (9, 9), and
/// // no amount of response-map interpolation can bring it back.
/// let sigma = sigma!(1.6);
/// let radius = NmsRadius::new(3).unwrap();
/// let mut corners =
///     detect_corners(&image, ShiTomasi, CornerParams::new(sigma, 0.05, radius).unwrap());
/// assert_eq!(corners[0].position().x, 9.0);
///
/// // Refinement reads the gradient field instead and recovers the corner.
/// let gx = sobel_x(&image, &Clamp);
/// let gy = sobel_y(&image, &Clamp);
/// let refined = refine_corners(&mut corners, &gx, &gy, NmsRadius::new(4).unwrap())?;
/// assert_eq!(refined, 4);
/// assert!((corners[0].position().x - 7.5).abs() <= 0.05, "{corners:?}");
/// assert!((corners[0].position().y - 7.5).abs() <= 0.05, "{corners:?}");
/// # Ok::<(), fovea::Error>(())
/// ```
pub fn refine_corners<IX, IY, P>(
    corners: &mut [Corner],
    gx: &IX,
    gy: &IY,
    radius: NmsRadius,
) -> Result<usize, Error>
where
    IX: RasterImage<Pixel = P>,
    IY: RasterImage<Pixel = P>,
    P: SingleChannel,
    f64: From<P::Channel>,
{
    if gx.size() != gy.size() {
        return Err(Error::SizeMismatch {
            expected: gx.size(),
            actual: gy.size(),
        });
    }

    let mut refined = 0;
    for corner in corners {
        if let Some(at) = refine_one(gx, gy, corner.at, radius.get()) {
            corner.at = at;
            refined += 1;
        }
    }
    Ok(refined)
}

/// One corner: solve, re-centre while the solution names a new pixel, and
/// report the fixed point.
fn refine_one<IX, IY, P>(
    gx: &IX,
    gy: &IY,
    at: CoordinateF64,
    radius: usize,
) -> Option<CoordinateF64>
where
    IX: RasterImage<Pixel = P>,
    IY: RasterImage<Pixel = P>,
    P: SingleChannel,
    f64: From<P::Channel>,
{
    let mut centre = pixel_site(at)?;
    for _ in 0..=MAX_RECENTRES {
        let solved = solve_window(gx, gy, centre, radius)?;
        let nearest = pixel_site(solved)?;
        if nearest == centre {
            return Some(solved);
        }
        centre = nearest;
    }
    None
}

/// The normal-equation solve over one window: `(Σ ggᵀ) d = Σ ggᵀ p`, with
/// `p` relative to the window centre so the arithmetic stays small.
fn solve_window<IX, IY, P>(
    gx: &IX,
    gy: &IY,
    centre: Coordinate,
    radius: usize,
) -> Option<CoordinateF64>
where
    IX: RasterImage<Pixel = P>,
    IY: RasterImage<Pixel = P>,
    P: SingleChannel,
    f64: From<P::Channel>,
{
    // The window must be complete. `checked_add` keeps a huge rounded
    // position from wrapping into a plausible bound.
    if centre.x < radius || centre.y < radius {
        return None;
    }
    let hi_x = centre.x.checked_add(radius)?;
    let hi_y = centre.y.checked_add(radius)?;
    if hi_x >= gx.width() || hi_y >= gx.height() {
        return None;
    }

    let (mut sxx, mut sxy, mut syy) = (0.0f64, 0.0f64, 0.0f64);
    let (mut bx, mut by) = (0.0f64, 0.0f64);
    for y in centre.y - radius..=hi_y {
        let row_x = gx.row(y);
        let row_y = gy.row(y);
        let dy = y as f64 - centre.y as f64;
        for x in centre.x - radius..=hi_x {
            let gxv = f64::from(row_x[x].channel(0));
            let gyv = f64::from(row_y[x].channel(0));
            let dx = x as f64 - centre.x as f64;
            let (xx, xy, yy) = (gxv * gxv, gxv * gyv, gyv * gyv);
            sxx += xx;
            sxy += xy;
            syy += yy;
            bx += xx * dx + xy * dy;
            by += xy * dx + yy * dy;
        }
    }

    // Two edge directions or nothing. Written in the positive, so a NaN
    // gradient makes the comparison false and is refused rather than solved.
    let determinant = sxx * syy - sxy * sxy;
    let trace = sxx + syy;
    let two_edge_directions = determinant > RANK_EPSILON * trace * trace;
    if !two_edge_directions {
        return None;
    }

    // Cramer's rule on the 2x2 system.
    let dx = (syy * bx - sxy * by) / determinant;
    let dy = (sxx * by - sxy * bx) / determinant;

    // A solution outside the window was voted for by samples that do not
    // surround it; the honest answer is that this window does not locate it.
    // Also written in the positive, for the same NaN reason.
    let reach = radius as f64;
    let inside_the_window = dx.abs() <= reach && dy.abs() <= reach;
    if !inside_the_window {
        return None;
    }
    Some(CoordinateF64::new(
        centre.x as f64 + dx,
        centre.y as f64 + dy,
    ))
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Sigma;
    use crate::border::{Clamp, Skip};
    use crate::features::detect::{
        CornerParams, FastParams, SegmentTest, ShiTomasi, corner_response_map, detect_corners,
        fast, interpolate_corners,
    };
    use crate::features::{HasPosition, HasResponse};
    use crate::image::{Image, ImageView, ImageViewMut};
    use crate::pixel::{MonoF32, MonoF64};
    use crate::sigma;
    use crate::transform::{sobel_x, sobel_y};

    // ── Fixtures ────────────────────────────────────────────────────────

    /// A white square on a black field, the fixture whose corners the
    /// detectors provably misplace.
    fn square(n: usize, lo: usize, hi: usize) -> Image<MonoF32> {
        Image::generate(n, n, |x, y| {
            let inside = (lo..hi).contains(&x) && (lo..hi).contains(&y);
            MonoF32::new(if inside { 1.0 } else { 0.0 })
        })
    }

    /// The true corner positions of `square(_, lo, hi)`: the step sits
    /// between pixels, half a pixel outside the first inside pixel.
    fn square_corners(lo: usize, hi: usize) -> [(f64, f64); 4] {
        let (a, b) = ((lo as f64) - 0.5, (hi as f64) - 0.5);
        [(a, a), (b, a), (a, b), (b, b)]
    }

    fn nearest_truth(truth: &[(f64, f64); 4], p: CoordinateF64) -> (f64, f64) {
        *truth
            .iter()
            .min_by(|a, b| {
                let da = (p.x - a.0).powi(2) + (p.y - a.1).powi(2);
                let db = (p.x - b.0).powi(2) + (p.y - b.1).powi(2);
                da.total_cmp(&db)
            })
            .unwrap()
    }

    fn max_response(map: &Image<MonoF32>) -> f32 {
        (0..map.height())
            .flat_map(|y| (0..map.width()).map(move |x| (x, y)))
            .map(|(x, y)| map.pixel_at(x, y).value())
            .fold(f32::NEG_INFINITY, f32::max)
    }

    /// A synthetic gradient pair whose edge lines cross at exactly
    /// (3.5, 5.2): a vertical edge shared equally by columns 3 and 4 (rows
    /// 3 and 4 only, so no pixel carries both gradients), and a horizontal
    /// edge whose rows 5 and 6 are weighted 4 : 1 by squared magnitude,
    /// putting the crossing at 5 + 4/(4+1) fifths of the way to 6.
    fn crossing_gradients() -> (Image<MonoF32>, Image<MonoF32>) {
        let gx = Image::generate(9, 9, |x, y| {
            let on = (x == 3 || x == 4) && (y == 3 || y == 4);
            MonoF32::new(if on { 1.0 } else { 0.0 })
        });
        let gy = Image::generate(9, 9, |_, y| {
            MonoF32::new(match y {
                5 => 2.0,
                6 => 1.0,
                _ => 0.0,
            })
        });
        (gx, gy)
    }

    // ── The reason this function exists ─────────────────────────────────

    #[test]
    fn the_windows_inward_drift_is_removed_for_every_sigma() {
        // The item's validation fixture: the same square whose detected
        // peak `a_larger_window_drags_the_peak_inward` pins at (8, 8) for
        // small windows and (9, 9) for large ones. Refinement must land on
        // the geometric corner within 0.05 px per axis for every window σ,
        // which interpolation provably cannot do at any precision.
        let image = square(24, 8, 16);
        let gx = sobel_x(&image, &Clamp);
        let gy = sobel_y(&image, &Clamp);
        let truth = square_corners(8, 16);

        for sigma in [0.8f32, 1.0, 1.2, 1.6, 2.0] {
            let window = Sigma::new(sigma).unwrap();
            let map: Image<MonoF32> = corner_response_map(&image, ShiTomasi, window);
            let params =
                CornerParams::try_new(window, 0.3 * max_response(&map), NmsRadius::new(3).unwrap())
                    .unwrap();
            let mut corners = detect_corners(&image, ShiTomasi, params);
            assert_eq!(corners.len(), 4, "sigma {sigma}: {corners:?}");

            let radius = (2.0 * f64::from(sigma)).ceil() as usize;
            assert_eq!(
                refine_corners(&mut corners, &gx, &gy, NmsRadius::new(radius).unwrap()).unwrap(),
                4,
                "sigma {sigma}"
            );

            for corner in &corners {
                let p = corner.position();
                let (tx, ty) = nearest_truth(&truth, p);
                assert!(
                    (p.x - tx).abs() <= 0.05 && (p.y - ty).abs() <= 0.05,
                    "sigma {sigma}, radius {radius}: refined to {p:?}, corner at ({tx}, {ty})",
                );
            }
        }
    }

    #[test]
    fn interpolation_locates_the_drifted_peak_and_refinement_the_corner() {
        // The measured fact that makes 12a and 12b two different items: at
        // σ = 1.6 the response peak sits a full pixel inside the corner
        // along both axes. Interpolating that map recovers the peak of that
        // map, still more than a pixel from the corner; refinement reads
        // the gradients instead and lands on it.
        let image = square(24, 8, 16);
        let window = sigma!(1.6);
        let map: Image<MonoF32> = corner_response_map(&image, ShiTomasi, window);
        let params =
            CornerParams::try_new(window, 0.3 * max_response(&map), NmsRadius::new(3).unwrap())
                .unwrap();
        let corners = detect_corners(&image, ShiTomasi, params);
        assert_eq!(corners[0].position(), CoordinateF64::new(9.0, 9.0));

        let distance = |p: CoordinateF64| ((p.x - 7.5).powi(2) + (p.y - 7.5).powi(2)).sqrt();

        let mut interpolated = corners.clone();
        assert!(interpolate_corners(&mut interpolated, &map) >= 1);
        assert!(
            distance(interpolated[0].position()) > 1.0,
            "the drift survives interpolation: {:?}",
            interpolated[0]
        );

        let mut refined = corners;
        let gx = sobel_x(&image, &Clamp);
        let gy = sobel_y(&image, &Clamp);
        assert_eq!(
            refine_corners(&mut refined, &gx, &gy, NmsRadius::new(4).unwrap()).unwrap(),
            4
        );
        assert!(
            distance(refined[0].position()) <= 0.06,
            "refinement removes it: {:?}",
            refined[0]
        );
    }

    #[test]
    fn a_segment_test_corner_refines_onto_the_geometric_corner() {
        // The other family's bias: FAST-9's saturated score ties along a
        // plateau around each corner, and the raster-first rule reports a
        // pixel up to two pixels from the corner; no segment-test parameter
        // moves it. On this square the top-right cluster's raster-first
        // pixel is (13, 8), a full two pixels from the corner at
        // (15.5, 7.5). The segment test never computes a gradient, so the
        // caller takes the same Sobel pair from the image.
        let image = square(24, 8, 16);
        let params = FastParams::new(
            SegmentTest::new(0.5, 9).unwrap(),
            NmsRadius::new(2).unwrap(),
        );
        let mut corners = fast(&image, params, &Skip);
        assert_eq!(corners.len(), 4, "{corners:?}");
        let truth = square_corners(8, 16);
        let bias = corners
            .iter()
            .map(|c| {
                let p = c.position();
                let (tx, ty) = nearest_truth(&truth, p);
                ((p.x - tx).powi(2) + (p.y - ty).powi(2)).sqrt()
            })
            .fold(0.0f64, f64::max);
        assert!(
            bias > 2.0,
            "the detection bias being corrected: {corners:?}"
        );

        let gx = sobel_x(&image, &Clamp);
        let gy = sobel_y(&image, &Clamp);
        assert_eq!(
            refine_corners(&mut corners, &gx, &gy, NmsRadius::new(3).unwrap()).unwrap(),
            4
        );

        for corner in &corners {
            let p = corner.position();
            let (tx, ty) = nearest_truth(&truth, p);
            assert!(
                (p.x - tx).abs() <= 0.05 && (p.y - ty).abs() <= 0.05,
                "refined to {p:?}, corner at ({tx}, {ty})",
            );
        }
    }

    // ── The solve, on exactly constructed gradients ─────────────────────

    #[test]
    fn the_solve_is_exact_when_no_pixel_sees_both_edges() {
        // Two clean edge lines crossing at (3.5, 5.2), no mixed pixels: the
        // least squares has a consistent solution and must hit it exactly.
        let (gx, gy) = crossing_gradients();
        let mut corners = vec![Corner::new(CoordinateF64::new(4.0, 5.0), 1.0)];
        assert_eq!(
            refine_corners(&mut corners, &gx, &gy, NmsRadius::new(2).unwrap()).unwrap(),
            1
        );
        let p = corners[0].position();
        assert!((p.x - 3.5).abs() < 1e-9, "{p:?}");
        assert!((p.y - 5.2).abs() < 1e-9, "{p:?}");
    }

    #[test]
    fn the_window_recentres_onto_the_corner() {
        // A start two pixels off: the first window sees only a sliver of
        // the vertical edge and solves toward the corner, the re-centred
        // window sees all of it, and the result is the same exact point a
        // well-placed start finds.
        let (gx, gy) = crossing_gradients();
        let mut corners = vec![Corner::new(CoordinateF64::new(6.0, 6.0), 1.0)];
        assert_eq!(
            refine_corners(&mut corners, &gx, &gy, NmsRadius::new(2).unwrap()).unwrap(),
            1
        );
        let p = corners[0].position();
        assert!((p.x - 3.5).abs() < 1e-9, "{p:?}");
        assert!((p.y - 5.2).abs() < 1e-9, "{p:?}");
    }

    // ── Refusals ────────────────────────────────────────────────────────

    #[test]
    fn a_window_leaving_the_frame_is_refused() {
        let (gx, gy) = crossing_gradients();
        // Radius 5 around (4, 5) runs off every side of the 9x9 frame.
        let mut corners = vec![Corner::new(CoordinateF64::new(4.0, 5.0), 1.0)];
        assert_eq!(
            refine_corners(&mut corners, &gx, &gy, NmsRadius::new(5).unwrap()).unwrap(),
            0
        );
        assert_eq!(corners[0].position(), CoordinateF64::new(4.0, 5.0));
    }

    #[test]
    fn a_window_seeing_one_straight_edge_is_refused() {
        // Mid-edge of the square: every gradient in the window is parallel,
        // the normal matrix is rank one, and "the intersection of one line"
        // is not a corner. The rank test refuses rather than divides.
        let image = square(24, 8, 16);
        let gx = sobel_x(&image, &Clamp);
        let gy = sobel_y(&image, &Clamp);
        let mut corners = vec![Corner::new(CoordinateF64::new(8.0, 12.0), 1.0)];
        assert_eq!(
            refine_corners(&mut corners, &gx, &gy, NmsRadius::new(2).unwrap()).unwrap(),
            0
        );
        assert_eq!(corners[0].position(), CoordinateF64::new(8.0, 12.0));
    }

    #[test]
    fn a_flat_window_is_refused() {
        let gx: Image<MonoF32> = Image::zero(9, 9);
        let gy: Image<MonoF32> = Image::zero(9, 9);
        let mut corners = vec![Corner::new(CoordinateF64::new(4.0, 4.0), 1.0)];
        assert_eq!(
            refine_corners(&mut corners, &gx, &gy, NmsRadius::new(2).unwrap()).unwrap(),
            0
        );
        assert_eq!(corners[0].position(), CoordinateF64::new(4.0, 4.0));
    }

    #[test]
    fn a_nan_gradient_refuses_the_fit() {
        let (gx, mut gy) = crossing_gradients();
        *gy.pixel_at_mut(4, 5) = MonoF32::new(f32::NAN);
        let mut corners = vec![Corner::new(CoordinateF64::new(4.0, 5.0), 1.0)];
        assert_eq!(
            refine_corners(&mut corners, &gx, &gy, NmsRadius::new(2).unwrap()).unwrap(),
            0
        );
        assert_eq!(corners[0].position(), CoordinateF64::new(4.0, 5.0));
    }

    #[test]
    fn a_position_off_the_map_is_left_alone() {
        // The frame hazard in its detectable form, matching
        // `interpolate_corners`: a position naming no pixel of these
        // gradients cannot be refined against them.
        let (gx, gy) = crossing_gradients();
        let mut corners = vec![
            Corner::new(CoordinateF64::new(40.0, 40.0), 0.9),
            Corner::new(CoordinateF64::new(-3.0, 2.0), 0.9),
            Corner::new(CoordinateF64::new(f64::NAN, 2.0), 0.9),
        ];
        assert_eq!(
            refine_corners(&mut corners, &gx, &gy, NmsRadius::new(2).unwrap()).unwrap(),
            0
        );
        assert_eq!(corners[0].position(), CoordinateF64::new(40.0, 40.0));
        assert_eq!(corners[1].position(), CoordinateF64::new(-3.0, 2.0));
    }

    // ── Contract ────────────────────────────────────────────────────────

    #[test]
    fn mismatched_gradient_sizes_are_reported() {
        let gx: Image<MonoF32> = Image::zero(9, 9);
        let gy: Image<MonoF32> = Image::zero(9, 8);
        let err = refine_corners(&mut [], &gx, &gy, NmsRadius::new(2).unwrap()).unwrap_err();
        assert!(matches!(err, Error::SizeMismatch { .. }), "{err:?}");
    }

    // A zero radius is no longer representable: the parameter is an
    // `NmsRadius`, whose constructor carries the at-least-one-pixel
    // invariant (a single-pixel window has a rank-one normal matrix, so
    // every fit would be refused). The rejection is pinned where the type
    // lives, in `nms_radius_carries_the_at_least_one_invariant`.

    #[test]
    fn the_count_reports_only_the_fits_that_succeeded() {
        let (gx, gy) = crossing_gradients();
        let mut corners = vec![
            Corner::new(CoordinateF64::new(0.0, 0.0), 0.5), // window off the frame
            Corner::new(CoordinateF64::new(4.0, 5.0), 0.5), // refines
        ];
        assert_eq!(
            refine_corners(&mut corners, &gx, &gy, NmsRadius::new(2).unwrap()).unwrap(),
            1
        );
        assert_eq!(corners[0].position(), CoordinateF64::new(0.0, 0.0));
        assert!((corners[1].position().x - 3.5).abs() < 1e-9);
    }

    #[test]
    fn the_response_is_left_alone() {
        let (gx, gy) = crossing_gradients();
        let mut corners = vec![Corner::new(CoordinateF64::new(4.0, 5.0), 0.75)];
        assert_eq!(
            refine_corners(&mut corners, &gx, &gy, NmsRadius::new(2).unwrap()).unwrap(),
            1
        );
        assert_eq!(
            corners[0].response(),
            0.75,
            "the refined position is not a place the detector scored",
        );
    }

    #[test]
    fn refining_no_corners_is_no_work() {
        let (gx, gy) = crossing_gradients();
        assert_eq!(
            refine_corners(&mut [], &gx, &gy, NmsRadius::new(2).unwrap()).unwrap(),
            0
        );
    }

    #[test]
    fn f64_gradients_are_accepted() {
        // The other float accumulator width, which Mono32 and up produce.
        let gx: Image<MonoF64> = Image::generate(9, 9, |x, y| {
            let on = (x == 3 || x == 4) && (y == 3 || y == 4);
            MonoF64::new(if on { 1.0 } else { 0.0 })
        });
        let gy: Image<MonoF64> = Image::generate(9, 9, |_, y| {
            MonoF64::new(match y {
                5 => 2.0,
                6 => 1.0,
                _ => 0.0,
            })
        });
        let mut corners = vec![Corner::new(CoordinateF64::new(4.0, 5.0), 1.0)];
        assert_eq!(
            refine_corners(&mut corners, &gx, &gy, NmsRadius::new(2).unwrap()).unwrap(),
            1
        );
        assert!((corners[0].position().y - 5.2).abs() < 1e-9);
    }
}

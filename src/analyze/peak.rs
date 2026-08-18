//! Peak interpolation: where a sampled surface's extremum falls between
//! pixels.
//!
//! Every operation that reports "the strongest position" reports a *pixel*:
//! a corner peak, a template match, an edge on a thinned gradient ridge.
//! The surface those operations searched is continuous, so the pixel it was
//! sampled onto is at best the nearest sample to the real extremum, and the
//! report carries a quantization error of up to half a pixel along each
//! axis. Fitting a quadratic to the samples around the winner and reporting
//! the fitted vertex removes that half pixel.
//!
//! One fit serves every caller. Only the surface and the dimensionality
//! differ:
//!
//! | Surface | Function | Fit |
//! |---|---|---|
//! | Corner response map | [`interpolate_peak`] | 2-D over the 3x3 neighbourhood |
//! | Template match score map | [`interpolate_peak`] | 2-D over the 3x3 neighbourhood |
//! | Gradient magnitude at one edge pixel | [`interpolate_peak_along`] | 1-D along the gradient |
//! | Gradient magnitude at many edge pixels or contour vertices | [`interpolate_ridge_points`] | 1-D along each site's own gradient |
//!
//! [`parabola_vertex`] is the 1-D arithmetic on its own, for three samples
//! that did not come from an image.
//!
//! ## What this does not do
//!
//! Interpolation locates the extremum of the surface it is given. It does
//! not ask whether that surface's extremum is where the feature actually
//! is. Those are different errors with different sizes:
//!
//! * **Quantization**, up to 0.5 px, from reporting a pixel index for a
//!   continuous position. That is what this module removes.
//! * **Localization bias**, from the surface itself being a displaced or
//!   smoothed version of the structure it was built from. A structure-tensor
//!   response map, for instance, moves its peak inward from a corner as the
//!   window grows, and interpolating a displaced peak yields a precise
//!   displaced peak. Removing that needs a different computation over a
//!   different input (the gradient field, not the response map), and it is
//!   not what any function here does.
//!
//! So nothing in this module is called "refinement", and none of it is
//! called "sub-pixel". Both words are used in the literature for both
//! concepts, which is how a position that is still a pixel off comes to be
//! described as sub-pixel refined.
//!
//! ## Absence is normal
//!
//! Every entry point returns an [`Option`]. A fit is refused, rather than
//! reported approximately, when the samples do not describe the requested
//! kind of extremum at all: a flat plateau has no unique vertex, a saddle
//! is not a maximum, and a site on the image border has no neighbour on one
//! side. Each function documents its own conditions.
//!
//! # Example
//!
//! ```
//! use fovea::Coordinate;
//! use fovea::analyze::peak::{interpolate_peak, Extremum};
//! use fovea::image::Image;
//! use fovea::pixel::MonoF32;
//!
//! // A paraboloid whose crest sits at (3.25, 2.4), sampled on the grid.
//! let surface: Image<MonoF32> = Image::generate(7, 7, |x, y| {
//!     let (dx, dy) = (x as f64 - 3.25, y as f64 - 2.4);
//!     MonoF32::new((1.0 - dx * dx - dy * dy) as f32)
//! });
//!
//! // The largest sample is (3, 2); the fit recovers where the crest is.
//! let at = interpolate_peak(&surface, Coordinate::new(3, 2), Extremum::Maximum)
//!     .expect("a paraboloid has a vertex");
//! assert!((at.x - 3.25).abs() < 1e-3, "{at:?}");
//! assert!((at.y - 2.40).abs() < 1e-3, "{at:?}");
//! ```

use crate::error::Error;
use crate::image::RasterImage;
use crate::pixel::SingleChannel;
use crate::transform::{nms_sector, nms_sector_from_gradient};
use crate::{Coordinate, CoordinateF64, Orientation};

/// Which kind of stationary point the samples are expected to describe.
///
/// The fit itself is polarity-free arithmetic; this is what turns it into a
/// checked operation. A score map that is *minimized* at the best position
/// ([`SSD`](crate::transform::SSD), [`SAD`](crate::transform::SAD)) and one
/// that is maximized ([`NCC`](crate::transform::NCC), any corner response)
/// go through the same code, and naming which is expected is what lets the
/// fit refuse a surface that curves the other way instead of returning the
/// wrong stationary point with no indication.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Extremum {
    /// The centre sample is expected to be the largest, and the fitted
    /// surface to curve downward away from its vertex.
    Maximum,
    /// The centre sample is expected to be the smallest, and the fitted
    /// surface to curve upward away from its vertex.
    Minimum,
}

/// Vertex of the parabola through `(-1, before)`, `(0, at)` and
/// `(1, after)`, as an offset from the centre sample.
///
/// The 1-D core of this module, exposed for three samples that did not come
/// from an image: a 1-D signal, a histogram bin and its neighbours, a
/// correlation curve. The unit of the result is one *step*, whatever the
/// caller spaced the samples by, so a caller sampling every second element
/// scales the result itself.
///
/// # Returns
///
/// `Some(offset)` with `offset` in `-1.0..=1.0`, or `None` when
///
/// * the three samples do not curve the way `kind` asks (which covers a
///   flat or linear run, whose second difference is zero and which has no
///   vertex, and any sample being `NaN`), or
/// * the vertex falls outside the interval the three samples span, which
///   means the parabola through them is extrapolating.
///
/// **The tighter bound is the usual one.** When the centre sample is the
/// extreme of the three, the vertex provably lies within *half* a step of
/// it, and the two conditions above are the only ways to fail. The wider
/// bound exists for sites that were not chosen by a local-extremum test at
/// all: a contour vertex is a border pixel of a mask, which can sit one
/// pixel to either side of the gradient crest, and a boundary falling
/// exactly half way between two pixels leaves the crest tied between them
/// with rounding to break the tie. Refusing those would refuse precisely
/// the positions the fit is most needed for, and their vertex is a little
/// past the half step, not somewhere unrelated.
///
/// # Example
///
/// ```
/// use fovea::analyze::peak::{parabola_vertex, Extremum};
///
/// // Symmetric samples put the vertex on the centre.
/// assert_eq!(parabola_vertex(1.0, 2.0, 1.0, Extremum::Maximum), Some(0.0));
///
/// // A heavier right neighbour pulls it right.
/// let offset = parabola_vertex(1.0, 2.0, 1.5, Extremum::Maximum).unwrap();
/// assert!((offset - 1.0 / 6.0).abs() < 1e-12, "{offset}");
///
/// // A minimum-seeking fit refuses a maximum, and vice versa.
/// assert_eq!(parabola_vertex(1.0, 2.0, 1.5, Extremum::Minimum), None);
///
/// // A flat run has no vertex to report.
/// assert_eq!(parabola_vertex(2.0, 2.0, 2.0, Extremum::Maximum), None);
///
/// // Nor has a straight one: no curvature, so no fit.
/// assert_eq!(parabola_vertex(0.0, 1.0, 2.0, Extremum::Maximum), None);
///
/// // A vertex the samples do not bracket is extrapolation, and refused.
/// assert_eq!(parabola_vertex(0.0, 1.0, 1.5, Extremum::Maximum), None);
/// ```
#[must_use]
pub fn parabola_vertex(before: f64, at: f64, after: f64, kind: Extremum) -> Option<f64> {
    // The second difference. Written in the positive, so a NaN sample makes
    // every comparison false and is refused rather than divided by.
    let curvature = before + after - 2.0 * at;
    let curves_toward_kind = match kind {
        Extremum::Maximum => curvature < 0.0,
        Extremum::Minimum => curvature > 0.0,
    };
    if !curves_toward_kind {
        return None;
    }
    let offset = 0.5 * (before - after) / curvature;
    // The three samples span one step either side; a vertex beyond that was
    // extrapolated from an interval that does not contain it.
    if offset.abs() > 1.0 {
        return None;
    }
    Some(offset)
}

/// Vertex of the quadratic fitted to the 3x3 neighbourhood of `at`, in the
/// surface's own coordinate frame.
///
/// The 2-D fit: six coefficients from the nine samples by central
/// differences, which reproduces a true quadratic surface exactly. The
/// cross term is part of the fit, so the vertex of a ridge lying at an
/// angle to the pixel axes comes out right. Two independent 1-D fits along
/// x and y would ignore that term and report a biased offset for exactly
/// the elongated, rotated response peaks that a corner or a match surface
/// produces.
///
/// The result is an absolute position in the same frame as `at`, not an
/// offset. For a map that is itself a pyramid level, that is the *level's*
/// frame; lift it with
/// [`Decimated::to_base`](crate::image::Decimated::to_base) as usual.
///
/// # Returns
///
/// `Some(position)` within one pixel of `at` along each axis, or `None`
/// when
///
/// * `at` is on the image border, so the 3x3 window is incomplete (which
///   is why a corner reported against the edge of the frame stays where it
///   was),
/// * the fitted surface has no isolated stationary point of the kind
///   `kind` asks for: its Hessian is singular (a plateau, or a perfect
///   ridge with no curvature along its length) or indefinite (a saddle),
///   or definite the other way round, or any sample is `NaN`, or
/// * the vertex falls outside the 3x3 window the samples came from. A fit
///   that puts the extremum among samples it never saw is extrapolating,
///   and the honest answer is that these nine samples do not locate it.
///
/// The window bound is the 2-D counterpart of [`parabola_vertex`]'s
/// interval bound, and it is the only bound available here: in 1-D a centre
/// sample that is the largest of three puts the vertex within half a step,
/// while in 2-D a centre sample that is the largest of nine implies no such
/// thing, because an elongated peak lying at an angle to the pixel axes can
/// put its vertex further out along its own ridge.
///
/// # Example
///
/// ```
/// use fovea::Coordinate;
/// use fovea::analyze::peak::{interpolate_peak, Extremum};
/// use fovea::image::Image;
/// use fovea::pixel::MonoF32;
///
/// // A score map minimized between two columns.
/// let scores: Image<MonoF32> = Image::generate(5, 5, |x, y| {
///     let (dx, dy) = (x as f64 - 2.5, y as f64 - 2.0);
///     MonoF32::new((dx * dx + dy * dy) as f32)
/// });
///
/// // Columns 2 and 3 tie for smallest; the fit lands between them.
/// let at = interpolate_peak(&scores, Coordinate::new(2, 2), Extremum::Minimum)
///     .expect("a paraboloid has a vertex");
/// assert!((at.x - 2.5).abs() < 1e-3, "{at:?}");
/// assert!((at.y - 2.0).abs() < 1e-3, "{at:?}");
///
/// // The same samples are not a maximum, and are refused as one.
/// assert!(interpolate_peak(&scores, Coordinate::new(2, 2), Extremum::Maximum).is_none());
/// ```
#[must_use]
pub fn interpolate_peak<I, P>(surface: &I, at: Coordinate, kind: Extremum) -> Option<CoordinateF64>
where
    I: RasterImage<Pixel = P>,
    P: SingleChannel,
    f64: From<P::Channel>,
{
    // The window needs one pixel of margin on all four sides. Written with
    // `saturating_sub` rather than `at.x + 1 >= w` so that a site past the
    // end of the address space is refused rather than wrapping into a
    // plausible index, and so that a zero-sized image needs no separate arm.
    let (w, h) = (surface.width(), surface.height());
    if at.x == 0 || at.y == 0 || at.x >= w.saturating_sub(1) || at.y >= h.saturating_sub(1) {
        return None;
    }

    // Rows above, at, and below the site. `s[dy][dx]` with both indices
    // shifted by one, so the centre is `s[1][1]`.
    let mut s = [[0.0f64; 3]; 3];
    for (row, y) in s.iter_mut().zip(at.y - 1..=at.y + 1) {
        let source = surface.row(y);
        for (sample, x) in row.iter_mut().zip(at.x - 1..=at.x + 1) {
            *sample = f64::from(source[x].channel(0));
        }
    }

    let (dx, dy) = quadratic_vertex(&s, kind)?;
    Some(CoordinateF64::new(at.x as f64 + dx, at.y as f64 + dy))
}

/// Vertex of the quadratic through a 3x3 sample block, as an offset from
/// the centre sample.
///
/// `s[dy][dx]`, both indices shifted by one. Separate from
/// [`interpolate_peak`] only so the arithmetic is readable on its own; it
/// stays private because its input shape is always an image window.
fn quadratic_vertex(s: &[[f64; 3]; 3], kind: Extremum) -> Option<(f64, f64)> {
    // Gradient and Hessian by central differences. Exact for a quadratic,
    // which is the surface being assumed.
    let gx = (s[1][2] - s[1][0]) / 2.0;
    let gy = (s[2][1] - s[0][1]) / 2.0;
    let hxx = s[1][2] - 2.0 * s[1][1] + s[1][0];
    let hyy = s[2][1] - 2.0 * s[1][1] + s[0][1];
    let hxy = (s[2][2] - s[2][0] - s[0][2] + s[0][0]) / 4.0;

    // An isolated extremum needs a definite Hessian, and its kind is the
    // sign of `hxx` once the determinant is positive. Both tests are
    // written in the positive, so a NaN sample yields `None`.
    let determinant = hxx * hyy - hxy * hxy;
    let definite = determinant > 0.0
        && match kind {
            Extremum::Maximum => hxx < 0.0,
            Extremum::Minimum => hxx > 0.0,
        };
    if !definite {
        return None;
    }

    // Solve H · offset = -gradient by Cramer's rule.
    let dx = -(hyy * gx - hxy * gy) / determinant;
    let dy = -(hxx * gy - hxy * gx) / determinant;

    // The samples span one pixel either side; a vertex beyond that was
    // extrapolated from a window that does not contain it.
    if dx.abs() > 1.0 || dy.abs() > 1.0 {
        return None;
    }
    Some((dx, dy))
}

/// Vertex of the parabola fitted to `magnitude` across `at` along
/// `gradient`, in the surface's own coordinate frame.
///
/// The edge fit. A gradient magnitude ridge is a crest that is only sharp
/// *across* itself, so the useful fit is one-dimensional and runs along the
/// gradient. The three samples are `at` and its two neighbours one step
/// along the axis nearest `gradient`, quantized to the same four axes
/// [`non_maximum_suppression`](crate::transform::non_maximum_suppression)
/// compares against. Those are the neighbours that decided `at` was a
/// ridge pixel in the first place, so the fit interpolates the surface as
/// suppression sampled it, and no value has to be resampled between pixels.
///
/// ## The quantized axis costs less than it looks
///
/// The reported point lands on the true edge line even though the axis it
/// was fitted along is up to 22.5 degrees off the gradient. Along any
/// straight line crossing a straight ridge, the ridge profile is the same
/// profile stretched by one over the cosine of the angle between them, so
/// the vertex found along that line is exactly where the line crosses the
/// crest. Quantizing the axis therefore displaces the point *along* the
/// edge, not across it, and the across-edge component is what edge geometry
/// (a fitted line, a fitted circle, a measured width) reads.
///
/// The sector choice is what keeps that stretch bounded: never more than
/// 1.09, because the axis is never more than 22.5 degrees off. Fitting
/// along an axis nearly parallel to the edge would instead divide by a
/// cosine near zero.
///
/// ## Pass the magnitude, not the thinned magnitude
///
/// `magnitude` must be the gradient magnitude **before** non-maximum
/// suppression. Suppression zeroes exactly the two neighbours this fit
/// reads, so a thinned map makes both of them `0`, which is a valid-looking
/// parabola with a meaningless vertex. Keep the
/// [`gradient_magnitude`](crate::transform::gradient_magnitude) output and
/// pass it here; use the thinned map or the
/// [`canny`](crate::analyze::edge::canny) mask to choose *which* sites to
/// interpolate.
///
/// # Returns
///
/// `Some(position)` within one step of `at`, and within half a step of it
/// whenever `at` is the crest of the three (which
/// [`non_maximum_suppression`](crate::transform::non_maximum_suppression)
/// guarantees for the sites it keeps). `None` when `at` or a neighbour along
/// the axis lies outside the image, or on the conditions [`parabola_vertex`]
/// lists. The step is one pixel on the two cardinal axes and the square
/// root of two on the two diagonals, so a diagonal fit scales those
/// distances by 1.41.
///
/// A ridge crest is a maximum by construction, so this takes no
/// [`Extremum`].
///
/// # Example
///
/// ```
/// use fovea::{Coordinate, Orientation};
/// use fovea::analyze::peak::interpolate_peak_along;
/// use fovea::image::Image;
/// use fovea::pixel::MonoF32;
///
/// // A magnitude ridge running down the image, crest near column 2.
/// let magnitude: Image<MonoF32> = Image::generate(5, 3, |x, _| {
///     MonoF32::new(match x { 1 => 1.0, 2 => 2.0, 3 => 1.5, _ => 0.0 })
/// });
///
/// // The gradient points along +x, so the fit runs left to right.
/// let across = Orientation::from_atan2(0.0, 1.0);
/// let at = interpolate_peak_along(&magnitude, Coordinate::new(2, 1), across)
///     .expect("the crest is in the middle of the three");
/// assert!((at.x - 2.0 - 1.0 / 6.0).abs() < 1e-12, "{at:?}");
/// assert_eq!(at.y, 1.0);
/// ```
#[must_use]
pub fn interpolate_peak_along<I, P>(
    magnitude: &I,
    at: Coordinate,
    gradient: Orientation,
) -> Option<CoordinateF64>
where
    I: RasterImage<Pixel = P>,
    P: SingleChannel,
    f64: From<P::Channel>,
{
    // Annotated rather than `f64::from`, which is ambiguous here: the
    // `f64: From<P::Channel>` bound is a second candidate impl.
    let theta: f64 = gradient.radians().into();
    fit_along_step(magnitude, at, nms_sector(theta))
}

/// Interpolated positions for many ridge sites, each along its own
/// gradient.
///
/// [`interpolate_peak_along`] over a list of sites, taking each site's
/// direction from the gradient pair instead of from a caller-supplied
/// angle. This is the batch form both edge and contour work wants, and the
/// reason the two are one function: an edge point cloud is the `true`
/// pixels of a thinned mask, a contour is a traced chain of border pixels,
/// and past the choice of sites they are the same operation on the same
/// surfaces.
///
/// Sites come from a thinned edge mask
/// ([`interpolate_edge_points`](crate::analyze::edge::interpolate_edge_points)
/// does that walk), from
/// [`Contour::points`](crate::analyze::contours::Contour::points), or from
/// anywhere else. `magnitude` is the **unthinned** gradient magnitude, for
/// the reason [`interpolate_peak_along`] gives.
///
/// # The result is aligned with the sites
///
/// One entry per site, in site order, `None` where that site's fit was
/// refused. Failures are not dropped, because a contour's vertices are a
/// sequence and silently removing one splices two unrelated parts of the
/// outline together. An edge point cloud has no such ordering and wants the
/// flattened form, which is `.into_iter().flatten()`.
///
/// # Errors
///
/// Returns [`Error::SizeMismatch`] if `magnitude`, `gx` and `gy` do not all
/// share a size. Three separately produced images, so this is the same
/// input-versus-input relation
/// [`combine_images`](crate::transform::combine_images) reports.
///
/// A site outside the image is not an error: it is a `None` entry, the same
/// as a site whose fit was refused.
///
/// # Example
///
/// ```
/// use fovea::Coordinate;
/// use fovea::analyze::peak::interpolate_ridge_points;
/// use fovea::image::Image;
/// use fovea::pixel::MonoF32;
///
/// // A vertical magnitude ridge, and a gradient field pointing along +x.
/// let magnitude: Image<MonoF32> = Image::generate(5, 2, |x, _| {
///     MonoF32::new(match x { 1 => 1.0, 2 => 2.0, 3 => 1.5, _ => 0.0 })
/// });
/// let gx: Image<MonoF32> = Image::fill(5, 2, MonoF32::new(1.0));
/// let gy: Image<MonoF32> = Image::fill(5, 2, MonoF32::new(0.0));
///
/// let sites = [Coordinate::new(2, 0), Coordinate::new(2, 1), Coordinate::new(0, 0)];
/// let points = interpolate_ridge_points(sites, &magnitude, &gx, &gy)?;
///
/// assert_eq!(points.len(), 3);
/// assert!((points[0].unwrap().x - 2.0 - 1.0 / 6.0).abs() < 1e-12);
/// assert!((points[1].unwrap().x - 2.0 - 1.0 / 6.0).abs() < 1e-12);
/// // The third site is on the border: no neighbour to its left.
/// assert_eq!(points[2], None);
/// # Ok::<(), fovea::Error>(())
/// ```
pub fn interpolate_ridge_points<IM, IX, IY, P>(
    sites: impl IntoIterator<Item = Coordinate>,
    magnitude: &IM,
    gx: &IX,
    gy: &IY,
) -> Result<Vec<Option<CoordinateF64>>, Error>
where
    IM: RasterImage<Pixel = P>,
    IX: RasterImage<Pixel = P>,
    IY: RasterImage<Pixel = P>,
    P: SingleChannel,
    f64: From<P::Channel>,
{
    if magnitude.size() != gx.size() {
        return Err(Error::SizeMismatch {
            expected: magnitude.size(),
            actual: gx.size(),
        });
    }
    if magnitude.size() != gy.size() {
        return Err(Error::SizeMismatch {
            expected: magnitude.size(),
            actual: gy.size(),
        });
    }

    let (w, h) = (magnitude.width(), magnitude.height());
    Ok(sites
        .into_iter()
        .map(|at| {
            if at.x >= w || at.y >= h {
                return None;
            }
            // The same quantization as `interpolate_peak_along`, straight
            // from the components: the sector is all that survives it, so
            // the angle would be a discarded `atan2` per site.
            let step = nms_sector_from_gradient(
                f64::from(gx.row(at.y)[at.x].channel(0)),
                f64::from(gy.row(at.y)[at.x].channel(0)),
            );
            fit_along_step(magnitude, at, step)
        })
        .collect())
}

/// [`parabola_vertex`] over `at` and its two neighbours one `step` away,
/// reported as a position.
///
/// The shared tail of [`interpolate_peak_along`] and
/// [`interpolate_ridge_points`], which differ only in where the step comes
/// from. `step` is a signed pixel displacement; the offset the fit returns
/// is in units of it, so a diagonal step scales the reported displacement
/// by the square root of two without any special case here.
fn fit_along_step<I, P>(
    magnitude: &I,
    at: Coordinate,
    step: (isize, isize),
) -> Option<CoordinateF64>
where
    I: RasterImage<Pixel = P>,
    P: SingleChannel,
    f64: From<P::Channel>,
{
    // Checked up front rather than left to fall out of the two neighbour
    // tests: both of those happen to imply it for every one of the four
    // sectors, but "the site is on the image" should not be a case analysis
    // over the sector table.
    if at.x >= magnitude.width() || at.y >= magnitude.height() {
        return None;
    }

    let (dx, dy) = step;
    let sample = |sign: isize| -> Option<f64> {
        let x = at.x.checked_add_signed(sign * dx)?;
        let y = at.y.checked_add_signed(sign * dy)?;
        if x >= magnitude.width() || y >= magnitude.height() {
            return None;
        }
        Some(f64::from(magnitude.row(y)[x].channel(0)))
    };

    let before = sample(-1)?;
    let after = sample(1)?;
    let centre = f64::from(magnitude.row(at.y)[at.x].channel(0));
    let offset = parabola_vertex(before, centre, after, Extremum::Maximum)?;
    Some(CoordinateF64::new(
        at.x as f64 + offset * dx as f64,
        at.y as f64 + offset * dy as f64,
    ))
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Image;
    use crate::pixel::{MonoF32, MonoF64};

    /// A paraboloid with its crest at `(cx, cy)` and no cross term.
    fn paraboloid(w: usize, h: usize, cx: f64, cy: f64) -> Image<MonoF64> {
        Image::generate(w, h, |x, y| {
            let (dx, dy) = (x as f64 - cx, y as f64 - cy);
            MonoF64::new(1.0 - dx * dx - dy * dy)
        })
    }

    // ── parabola_vertex ──────────────────────────────────────────────────

    #[test]
    fn a_symmetric_triple_puts_the_vertex_on_the_centre() {
        assert_eq!(parabola_vertex(0.0, 1.0, 0.0, Extremum::Maximum), Some(0.0));
        assert_eq!(parabola_vertex(1.0, 0.0, 1.0, Extremum::Minimum), Some(0.0));
    }

    #[test]
    fn the_vertex_leans_toward_the_larger_neighbour() {
        let right = parabola_vertex(1.0, 2.0, 1.5, Extremum::Maximum).unwrap();
        let left = parabola_vertex(1.5, 2.0, 1.0, Extremum::Maximum).unwrap();
        assert!(right > 0.0 && left < 0.0);
        assert!(
            (right + left).abs() < 1e-12,
            "mirror images: {right} {left}"
        );
    }

    #[test]
    fn a_centred_crest_keeps_the_vertex_within_half_a_step() {
        // The extreme case is an exact tie with one neighbour, which puts
        // the vertex exactly on the half step rather than past it.
        assert_eq!(parabola_vertex(0.0, 1.0, 1.0, Extremum::Maximum), Some(0.5));
        assert_eq!(
            parabola_vertex(1.0, 1.0, 0.0, Extremum::Maximum),
            Some(-0.5)
        );

        // Anything strictly inside that stays strictly inside.
        for after in [0.0, 0.25, 0.5, 0.75, 0.9] {
            let offset = parabola_vertex(0.0, 1.0, after, Extremum::Maximum).unwrap();
            assert!(offset.abs() < 0.5, "after = {after} gave {offset}");
        }
    }

    #[test]
    fn a_crest_one_pixel_over_is_still_fitted() {
        // The contour case: the site is next to the crest rather than on
        // it, which happens when a traced border pixel sits on the far side
        // of the boundary. The vertex is a little past the half step and is
        // reported, because refusing it would refuse exactly the half-pixel
        // boundaries this fit exists for.
        let offset = parabola_vertex(2.0, 1.9, 0.0, Extremum::Maximum).unwrap();
        assert!((-1.0..-0.5).contains(&offset), "{offset}");

        // Mirrored.
        let mirrored = parabola_vertex(0.0, 1.9, 2.0, Extremum::Maximum).unwrap();
        assert!((0.5..1.0).contains(&mirrored), "{mirrored}");
    }

    #[test]
    fn a_vertex_the_samples_do_not_bracket_is_refused() {
        // Past one whole step the parabola is extrapolating from an
        // interval that does not contain its own vertex.
        assert_eq!(parabola_vertex(0.0, 1.0, 1.5, Extremum::Maximum), None);
        assert_eq!(parabola_vertex(1.5, 1.0, 0.0, Extremum::Maximum), None);
        assert_eq!(parabola_vertex(0.0, 1.0, 1.5, Extremum::Minimum), None);
    }

    #[test]
    fn a_straight_run_has_no_curvature_to_fit() {
        assert_eq!(parabola_vertex(0.0, 1.0, 2.0, Extremum::Maximum), None);
        assert_eq!(parabola_vertex(2.0, 1.0, 0.0, Extremum::Minimum), None);
    }

    #[test]
    fn the_requested_kind_is_enforced() {
        assert_eq!(parabola_vertex(0.0, 1.0, 0.5, Extremum::Minimum), None);
        assert_eq!(parabola_vertex(1.0, 0.0, 0.5, Extremum::Maximum), None);
    }

    #[test]
    fn a_flat_triple_has_no_vertex() {
        assert_eq!(parabola_vertex(2.0, 2.0, 2.0, Extremum::Maximum), None);
        assert_eq!(parabola_vertex(2.0, 2.0, 2.0, Extremum::Minimum), None);
    }

    #[test]
    fn a_nan_sample_is_refused_rather_than_propagated() {
        for kind in [Extremum::Maximum, Extremum::Minimum] {
            assert_eq!(parabola_vertex(f64::NAN, 1.0, 0.0, kind), None);
            assert_eq!(parabola_vertex(0.0, f64::NAN, 0.0, kind), None);
            assert_eq!(parabola_vertex(0.0, 1.0, f64::NAN, kind), None);
        }
    }

    // ── interpolate_peak ─────────────────────────────────────────────────

    #[test]
    fn the_fit_recovers_a_quadratic_surfaces_vertex_exactly() {
        // Central differences are exact for a quadratic, so this is an
        // equality test up to float arithmetic, not an accuracy test.
        for (cx, cy) in [(3.0, 3.0), (3.25, 2.4), (2.6, 3.9), (3.5, 3.5)] {
            let surface = paraboloid(7, 7, cx, cy);
            let at = Coordinate::new(cx.round() as usize, cy.round() as usize);
            let found = interpolate_peak(&surface, at, Extremum::Maximum).unwrap();
            assert!(
                (found.x - cx).abs() < 1e-9 && (found.y - cy).abs() < 1e-9,
                "crest ({cx}, {cy}) recovered as {found:?}",
            );
        }
    }

    #[test]
    fn the_cross_term_is_part_of_the_fit() {
        // A paraboloid rotated 45 degrees and elongated 4:1. Two separate
        // 1-D fits would ignore the xy term and miss the vertex; the 2-D
        // fit recovers it.
        let (cx, cy) = (4.3, 3.6);
        let surface: Image<MonoF64> = Image::generate(9, 9, |x, y| {
            let (dx, dy) = (x as f64 - cx, y as f64 - cy);
            let (u, v) = ((dx + dy) / 2.0f64.sqrt(), (dx - dy) / 2.0f64.sqrt());
            MonoF64::new(1.0 - u * u - 16.0 * v * v)
        });
        let found = interpolate_peak(&surface, Coordinate::new(4, 4), Extremum::Maximum).unwrap();
        assert!(
            (found.x - cx).abs() < 1e-9 && (found.y - cy).abs() < 1e-9,
            "{found:?}",
        );
    }

    #[test]
    fn a_site_on_the_border_has_no_complete_window() {
        let surface = paraboloid(5, 5, 0.0, 0.0);
        for at in [
            Coordinate::new(0, 2),
            Coordinate::new(2, 0),
            Coordinate::new(4, 2),
            Coordinate::new(2, 4),
        ] {
            assert_eq!(interpolate_peak(&surface, at, Extremum::Maximum), None);
        }
    }

    #[test]
    fn a_plateau_has_no_isolated_vertex() {
        let surface: Image<MonoF32> = Image::fill(5, 5, MonoF32::new(1.0));
        assert_eq!(
            interpolate_peak(&surface, Coordinate::new(2, 2), Extremum::Maximum),
            None
        );
    }

    #[test]
    fn a_saddle_is_neither_maximum_nor_minimum() {
        // z = x² − y²: a stationary point at the centre that is neither.
        let surface: Image<MonoF64> = Image::generate(5, 5, |x, y| {
            let (dx, dy) = (x as f64 - 2.0, y as f64 - 2.0);
            MonoF64::new(dx * dx - dy * dy)
        });
        for kind in [Extremum::Maximum, Extremum::Minimum] {
            assert_eq!(
                interpolate_peak(&surface, Coordinate::new(2, 2), kind),
                None
            );
        }
    }

    #[test]
    fn a_ridge_with_no_curvature_along_it_is_refused() {
        // Curved across, flat along: the Hessian is singular, so there is
        // no isolated vertex to report and the fit says so rather than
        // dividing by zero.
        let surface: Image<MonoF64> = Image::generate(5, 5, |x, _| {
            let dx = x as f64 - 2.0;
            MonoF64::new(1.0 - dx * dx)
        });
        assert_eq!(
            interpolate_peak(&surface, Coordinate::new(2, 2), Extremum::Maximum),
            None
        );
    }

    #[test]
    fn a_vertex_outside_the_sampled_window_is_refused() {
        // A quadratic whose crest is three pixels away from the site: the
        // fit is exact but it is extrapolating, and the site was never a
        // discrete maximum to begin with.
        let surface = paraboloid(9, 9, 7.0, 4.0);
        assert_eq!(
            interpolate_peak(&surface, Coordinate::new(4, 4), Extremum::Maximum),
            None
        );
    }

    #[test]
    fn the_requested_kind_is_enforced_in_two_dimensions() {
        let bowl: Image<MonoF64> = Image::generate(7, 7, |x, y| {
            let (dx, dy) = (x as f64 - 3.25, y as f64 - 3.0);
            MonoF64::new(dx * dx + dy * dy)
        });
        let at = Coordinate::new(3, 3);
        assert!(interpolate_peak(&bowl, at, Extremum::Minimum).is_some());
        assert_eq!(interpolate_peak(&bowl, at, Extremum::Maximum), None);
    }

    #[test]
    fn a_nan_neighbour_refuses_the_fit() {
        let surface: Image<MonoF32> = Image::generate(5, 5, |x, y| {
            MonoF32::new(if (x, y) == (3, 2) {
                f32::NAN
            } else if (x, y) == (2, 2) {
                1.0
            } else {
                0.0
            })
        });
        assert_eq!(
            interpolate_peak(&surface, Coordinate::new(2, 2), Extremum::Maximum),
            None
        );
    }

    #[test]
    fn both_float_accumulator_widths_are_accepted() {
        // The bound is `f64: From<Channel>`, the same one `corner_peaks`
        // carries, so the two float mono types are the surfaces in
        // practice. The fit itself is always in `f64`.
        let wide = paraboloid(7, 7, 3.25, 3.0);
        let narrow: Image<MonoF32> = Image::generate(7, 7, |x, y| {
            let (dx, dy) = (x as f64 - 3.25, y as f64 - 3.0);
            MonoF32::new((1.0 - dx * dx - dy * dy) as f32)
        });
        let at = Coordinate::new(3, 3);
        let a = interpolate_peak(&wide, at, Extremum::Maximum).unwrap();
        let b = interpolate_peak(&narrow, at, Extremum::Maximum).unwrap();
        assert!((a.x - 3.25).abs() < 1e-9, "{a:?}");
        assert!((b.x - 3.25).abs() < 1e-3, "{b:?}");
    }

    // ── interpolate_peak_along ───────────────────────────────────────────

    /// A magnitude image with an explicit column profile, constant in y.
    fn column_ridge(profile: [f32; 5]) -> Image<MonoF32> {
        Image::generate(5, 3, |x, _| MonoF32::new(profile[x]))
    }

    #[test]
    fn the_ridge_fit_runs_along_the_gradient() {
        let magnitude = column_ridge([0.0, 1.0, 2.0, 1.5, 0.0]);
        let across = Orientation::from_atan2(0.0, 1.0);
        let at = interpolate_peak_along(&magnitude, Coordinate::new(2, 1), across).unwrap();
        assert!((at.x - (2.0 + 1.0 / 6.0)).abs() < 1e-12, "{at:?}");
        assert_eq!(at.y, 1.0, "the fit does not move across its own axis");
    }

    #[test]
    fn opposite_gradients_give_the_same_edge_point() {
        // The sector folds direction onto axis, so a gradient pointing the
        // other way across the same ridge locates the same crest.
        let magnitude = column_ridge([0.0, 1.0, 2.0, 1.5, 0.0]);
        let east = interpolate_peak_along(
            &magnitude,
            Coordinate::new(2, 1),
            Orientation::from_atan2(0.0, 1.0),
        );
        let west = interpolate_peak_along(
            &magnitude,
            Coordinate::new(2, 1),
            Orientation::from_atan2(0.0, -1.0),
        );
        assert_eq!(east, west);
    }

    #[test]
    fn a_diagonal_gradient_moves_the_point_on_both_axes() {
        // A ridge running along the anti-diagonal, crest offset toward
        // (+1, +1). The step is diagonal, so the reported displacement is
        // the offset on both axes at once. One diagonal step changes
        // `x + y` by two, which is why the shoulders sit at ±2.
        let magnitude: Image<MonoF32> = Image::generate(5, 5, |x, y| {
            let d = x as isize + y as isize - 4;
            MonoF32::new(match d {
                0 => 2.0,
                2 => 1.5,
                -2 => 1.0,
                _ => 0.0,
            })
        });
        let along = Orientation::from_atan2(1.0, 1.0);
        let at = interpolate_peak_along(&magnitude, Coordinate::new(2, 2), along).unwrap();
        let expected = 2.0 + 1.0 / 6.0;
        assert!((at.x - expected).abs() < 1e-12, "{at:?}");
        assert!((at.y - expected).abs() < 1e-12, "{at:?}");
    }

    #[test]
    fn a_ridge_site_against_the_border_is_refused() {
        let magnitude = column_ridge([2.0, 1.5, 1.0, 0.5, 0.0]);
        let across = Orientation::from_atan2(0.0, 1.0);
        assert_eq!(
            interpolate_peak_along(&magnitude, Coordinate::new(0, 1), across),
            None
        );
    }

    #[test]
    fn a_thinned_magnitude_is_not_silently_accepted() {
        // Suppression zeroes the two neighbours the fit reads. With both
        // at zero the centre is still the largest of three, so the
        // parabola is well formed and its vertex is the centre: the fit
        // reports no movement rather than a wrong position, which is the
        // failure mode the docs warn about.
        let thinned = column_ridge([0.0, 0.0, 2.0, 0.0, 0.0]);
        let across = Orientation::from_atan2(0.0, 1.0);
        let at = interpolate_peak_along(&thinned, Coordinate::new(2, 1), across).unwrap();
        assert_eq!(at, CoordinateF64::new(2.0, 1.0));

        // The same site on the unthinned magnitude does move.
        let magnitude = column_ridge([0.0, 1.0, 2.0, 1.5, 0.0]);
        let moved = interpolate_peak_along(&magnitude, Coordinate::new(2, 1), across).unwrap();
        assert!(moved.x > 2.0, "{moved:?}");
    }

    // ── interpolate_ridge_points ─────────────────────────────────────────

    #[test]
    fn ridge_points_take_their_direction_from_the_gradient_pair() {
        let magnitude = column_ridge([0.0, 1.0, 2.0, 1.5, 0.0]);
        let gx: Image<MonoF32> = Image::fill(5, 3, MonoF32::new(1.0));
        let gy: Image<MonoF32> = Image::fill(5, 3, MonoF32::new(0.0));

        let sites = [Coordinate::new(2, 0), Coordinate::new(2, 2)];
        let points = interpolate_ridge_points(sites, &magnitude, &gx, &gy).unwrap();
        let expected = 2.0 + 1.0 / 6.0;
        assert_eq!(points.len(), 2);
        assert!((points[0].unwrap().x - expected).abs() < 1e-12);
        assert!((points[1].unwrap().x - expected).abs() < 1e-12);
    }

    #[test]
    fn ridge_points_stay_aligned_with_their_sites() {
        let magnitude = column_ridge([0.0, 1.0, 2.0, 1.5, 0.0]);
        let gx: Image<MonoF32> = Image::fill(5, 3, MonoF32::new(1.0));
        let gy: Image<MonoF32> = Image::fill(5, 3, MonoF32::new(0.0));

        // A good site, a border site, a site off the image, and a site on
        // the straight part of the profile.
        let sites = [
            Coordinate::new(2, 1),
            Coordinate::new(0, 1),
            Coordinate::new(9, 9),
            Coordinate::new(1, 1),
        ];
        let points = interpolate_ridge_points(sites, &magnitude, &gx, &gy).unwrap();
        assert_eq!(points.len(), 4);
        assert!(points[0].is_some());
        assert_eq!(points[1], None, "no left neighbour");
        assert_eq!(points[2], None, "off the image");
        assert_eq!(points[3], None, "0, 1, 2 is straight: no curvature");
    }

    #[test]
    fn ridge_points_reject_mismatched_inputs() {
        let magnitude = column_ridge([0.0, 1.0, 2.0, 1.5, 0.0]);
        let wrong: Image<MonoF32> = Image::fill(4, 3, MonoF32::new(1.0));
        let right: Image<MonoF32> = Image::fill(5, 3, MonoF32::new(0.0));

        let sites = [Coordinate::new(2, 1)];
        assert!(matches!(
            interpolate_ridge_points(sites, &magnitude, &wrong, &right),
            Err(Error::SizeMismatch { .. })
        ));
        assert!(matches!(
            interpolate_ridge_points(sites, &magnitude, &right, &wrong),
            Err(Error::SizeMismatch { .. })
        ));
    }

    #[test]
    fn a_traced_contour_interpolates_onto_the_grey_boundary() {
        // The fourth call site, end to end: trace a mask, then interpolate
        // the traced vertices against the greyscale the mask came from.
        // The square's true boundaries are at 9.5 and 25.5, which no pixel
        // centre is on, so a correct fit moves every mid-side vertex by
        // exactly half a pixel and the direction it moves in comes from
        // that vertex's own gradient.
        use crate::analyze::contours::{Connectivity8, extract_contours};
        use crate::border::Clamp;
        use crate::pixel::Label32;
        use crate::transform::{gaussian_blur, gradient_magnitude, scharr_x, scharr_y};
        use crate::{Sigma, image::BinaryImage};

        const LO: usize = 10;
        const HI: usize = 25;
        let inside = |v: usize| (LO..=HI).contains(&v);

        let image: Image<MonoF32> = Image::generate(36, 36, |x, y| {
            MonoF32::new(if inside(x) && inside(y) { 1.0 } else { 0.0 })
        });
        let mask: BinaryImage = Image::generate(36, 36, |x, y| inside(x) && inside(y));

        let blurred: Image<MonoF32> = gaussian_blur(&image, Sigma::new(1.0), &Clamp);
        let gx = scharr_x(&blurred, &Clamp);
        let gy = scharr_y(&blurred, &Clamp);
        let magnitude = gradient_magnitude(&gx, &gy).unwrap();

        let (_, hierarchy) = extract_contours::<Label32, Connectivity8>(&mask).unwrap();
        let contour = hierarchy.components()[0].outer();
        let vertices = contour.points();
        let points =
            interpolate_ridge_points(vertices.iter().copied(), &magnitude, &gx, &gy).unwrap();

        assert_eq!(
            points.len(),
            vertices.len(),
            "one entry per vertex, in vertex order",
        );

        // Away from the corners a side's gradient is axis-aligned, so the
        // fit runs across that side and the other coordinate is untouched.
        // Five pixels of clearance, because within two or three of a corner
        // the rounding of the *other* side is still measurable and the
        // magnitude ridge is no longer symmetric about the true boundary.
        let mid = |v: usize| (LO + 5..=HI - 5).contains(&v);
        let mut checked = 0;
        for (vertex, point) in vertices.iter().zip(&points) {
            let (vx, vy) = (vertex.x as f64, vertex.y as f64);
            let expected = match (*vertex, mid(vertex.x), mid(vertex.y)) {
                (v, _, true) if v.x == LO => Some((LO as f64 - 0.5, vy)),
                (v, _, true) if v.x == HI => Some((HI as f64 + 0.5, vy)),
                (v, true, _) if v.y == LO => Some((vx, LO as f64 - 0.5)),
                (v, true, _) if v.y == HI => Some((vx, HI as f64 + 0.5)),
                _ => None, // a corner, where the sector is diagonal
            };
            if let Some((ex, ey)) = expected {
                let point =
                    point.unwrap_or_else(|| panic!("mid-side vertex {vertex:?} was refused a fit"));
                assert!(
                    (point.x - ex).abs() < 1e-3 && (point.y - ey).abs() < 1e-3,
                    "vertex {vertex:?} interpolated to {point:?}, expected ({ex}, {ey})",
                );
                checked += 1;
            }
        }
        assert_eq!(checked, 4 * (HI - LO - 9), "every mid-side vertex checked");
    }

    #[test]
    fn no_sites_is_no_points() {
        let magnitude = column_ridge([0.0, 1.0, 2.0, 1.5, 0.0]);
        let flat: Image<MonoF32> = Image::fill(5, 3, MonoF32::new(1.0));
        let points = interpolate_ridge_points([], &magnitude, &flat, &flat).unwrap();
        assert!(points.is_empty());
    }
}

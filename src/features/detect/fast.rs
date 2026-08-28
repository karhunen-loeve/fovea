//! The segment-test corner detector, known as FAST.
//!
//! Reads 16 raw intensities on a radius-3 ring and asks whether a contiguous
//! arc of them is entirely brighter, or entirely darker, than the centre.

use crate::border::BorderPolicy;
use crate::error::Error;
use crate::features::Corner;
use crate::image::{Decimated, Image, ImageView, RasterImage, RasterImageMut};
use crate::pixel::{LinearPixel, MonoF32, SingleChannel};
use crate::{CoordinateF64, Offset, Rectangle, Size};

use super::peaks::{NmsRadius, corner_peaks, scan_peaks};

// ─── The ring ────────────────────────────────────────────────────────────────

/// The radius of the segment-test ring, in pixels.
///
/// Fixed at 3 — the value the detector was defined with and the one every
/// other implementation uses, so `FAST-9` means the same thing here as
/// elsewhere. It is also what makes [`FAST_RING`] exactly 16 pixels: a
/// Bresenham circle of radius 3 is the smallest one whose circumference is
/// large enough for the arc lengths that discriminate corners from edges
/// (see [`SegmentTest`]).
pub const FAST_RING_RADIUS: usize = 3;

/// The 16 ring offsets, in clockwise order starting from directly above the
/// centre.
///
/// The Bresenham circle of radius [`FAST_RING_RADIUS`]. Public because the
/// ring is the detector's whole geometry: drawing it over an image is how
/// you see *why* a pixel scored what it did, and the order is what
/// "contiguous" in [`SegmentTest::arc_length`] means — index 15 is adjacent
/// to index 0, so arcs wrap.
///
/// # Example
///
/// ```
/// use fovea::Offset;
/// use fovea::features::detect::{FAST_RING, FAST_RING_RADIUS};
///
/// let radius = FAST_RING_RADIUS as i32;
/// assert_eq!(FAST_RING.len(), 16);
/// assert_eq!(FAST_RING[0], Offset::new(0, -radius)); // straight up
/// assert_eq!(FAST_RING[4], Offset::new(radius, 0)); // a quarter turn on
///
/// // Every offset lies on the circle, and consecutive offsets are adjacent.
/// for (i, &offset) in FAST_RING.iter().enumerate() {
///     assert!(offset.dx.abs() <= 3 && offset.dy.abs() <= 3);
///     let next = FAST_RING[(i + 1) % 16];
///     assert!((next.dx - offset.dx).abs() <= 1 && (next.dy - offset.dy).abs() <= 1);
/// }
/// ```
pub const FAST_RING: [Offset; 16] = [
    Offset::new(0, -3),
    Offset::new(1, -3),
    Offset::new(2, -2),
    Offset::new(3, -1),
    Offset::new(3, 0),
    Offset::new(3, 1),
    Offset::new(2, 2),
    Offset::new(1, 3),
    Offset::new(0, 3),
    Offset::new(-1, 3),
    Offset::new(-2, 2),
    Offset::new(-3, 1),
    Offset::new(-3, 0),
    Offset::new(-3, -1),
    Offset::new(-2, -2),
    Offset::new(-1, -3),
];

/// The footprint the ring occupies: `(2·radius + 1)²`, the "kernel size" the
/// border policy is asked about.
const FOOTPRINT: usize = 2 * FAST_RING_RADIUS + 1;

/// [`FAST_RING`] rewritten as `(row, dx)`, where `row` indexes the
/// [`FOOTPRINT`] scan lines `y − 3 ..= y + 3` rather than naming a `dy`.
///
/// The interior scan hoists those seven rows once per scan line, so a ring
/// sample is `rows[row][x + dx]`: one slice index instead of a `row()` call
/// and a pair of range tests per sample.
const FAST_RING_ROWS: [(usize, isize); 16] = {
    let mut table = [(0usize, 0isize); 16];
    let mut index = 0;
    while index < 16 {
        let offset = FAST_RING[index];
        table[index] = (
            (offset.dy + FAST_RING_RADIUS as i32) as usize,
            offset.dx as isize,
        );
        index += 1;
    }
    table
};

// ─── SegmentTest ─────────────────────────────────────────────────────────────

/// The segment test itself — an intensity threshold and an arc length,
/// validated once.
///
/// The pixel `p` is a corner at threshold `t` when some run of `arc_length`
/// *contiguous* ring pixels is entirely at least `I_p + t`, or entirely at
/// most `I_p − t`. Both numbers are the test, which is why they are one
/// type: an arc length without a threshold decides nothing, and a threshold
/// without an arc length is not a corner criterion.
///
/// This is the analogue of [`Harris`](super::Harris) in the structure-tensor
/// family — the validated object that names *which* test to run — and it
/// carries its own invariants the same way, so no separate newtype has to be
/// threaded through the API: `const fn` [`new`](Self::new) for literals,
/// [`try_new`](Self::try_new) for values computed from data.
///
/// # The threshold is in intensity units
///
/// Unlike a structure-tensor response, which is a squared or fourth power of
/// contrast in the gradient operator's own gain, `t` is a plain intensity
/// difference: `20` on a `Mono8` image means "20 grey levels", and `0.08` on
/// a `MonoF32` image in `0.0..=1.0` means the same thing. That is the
/// practical reason to reach for this detector — the threshold can be
/// reasoned about rather than calibrated.
///
/// It must be **strictly positive**. At `t = 0` the test is "at least as
/// bright as the centre", which every pixel of a flat field satisfies on all
/// 16 ring positions: the detector would report the entire image. That is
/// the same class of silent failure as
/// [`Harris`](super::Harris)'s `k ≥ 0.25`, and is rejected in the same place.
///
/// # Why `9 ≤ arc_length ≤ 16`
///
/// The ring has 16 pixels, so an arc of 17 can never be found and the
/// detector would be dead — the upper bound is arithmetic.
///
/// The lower bound is the corner/edge distinction. A straight step edge
/// through the pixel cuts the ring into two arcs, and the brighter side can
/// be as long as 8 pixels; at `arc_length ≤ 8` every pixel along every edge
/// in the image therefore passes, and the detector has stopped discriminating.
/// `9` is the smallest length that cannot be satisfied by a straight edge —
/// and, independently, the one Rosten and Drummond measured as the most
/// repeatable, which is why `FAST-9` is the variant ORB uses. `12` is the
/// original formulation, more selective and the one the classical four-point
/// early-rejection test is derived for.
///
/// # Example
///
/// ```
/// use fovea::features::detect::SegmentTest;
///
/// // FAST-9 on a MonoF32 image in 0.0..=1.0: 8 % contrast.
/// const FAST9: SegmentTest = SegmentTest::new(0.08, 9).unwrap();
/// assert_eq!(FAST9.arc_length(), 9);
///
/// // A threshold derived from a noise estimate is checked where it is computed.
/// let noise_sigma = 0.011_f32;
/// let tuned = SegmentTest::try_new(5.0 * noise_sigma, 9)?;
/// assert!((tuned.threshold() - 0.055).abs() < 1e-6);
///
/// // Both silent failure modes are errors, not empty results.
/// assert!(SegmentTest::try_new(0.0, 9).is_err()); // every pixel is a corner
/// assert!(SegmentTest::try_new(0.08, 8).is_err()); // every edge is a corner
/// assert!(SegmentTest::try_new(0.08, 17).is_err()); // nothing can ever fire
/// # Ok::<(), fovea::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SegmentTest {
    threshold: f32,
    arc_length: usize,
}

impl SegmentTest {
    /// Creates a segment test, returning `None` unless `threshold` is
    /// finite and strictly positive and `9 <= arc_length <= 16` (see the
    /// type documentation for why those are the bounds).
    ///
    /// `const`, so binding the result to a `const` item checks a pair of
    /// literals at compile time. There is deliberately no literal macro:
    /// the two arguments are not interchangeable and their names are the
    /// information, so `SegmentTest::new(0.08, 9).unwrap()` is the more
    /// readable form. For values computed from data use
    /// [`try_new`](Self::try_new), which reports which bound failed.
    #[must_use]
    pub const fn new(threshold: f32, arc_length: usize) -> Option<Self> {
        if !(threshold.is_finite() && threshold > 0.0) {
            return None;
        }
        if arc_length < 9 || arc_length > 16 {
            return None;
        }
        Some(Self {
            threshold,
            arc_length,
        })
    }

    /// Creates a segment test from computed values, validating them.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidParameter`] if `threshold` is not finite and
    /// strictly positive, or if `arc_length` is outside `9..=16`.
    pub fn try_new(threshold: f32, arc_length: usize) -> Result<Self, Error> {
        if !(threshold.is_finite() && threshold > 0.0) {
            return Err(Error::InvalidParameter(format!(
                "FAST threshold must be finite and strictly positive, got {threshold}"
            )));
        }
        if !(9..=16).contains(&arc_length) {
            return Err(Error::InvalidParameter(format!(
                "FAST arc_length must satisfy 9 <= n <= 16, got {arc_length}"
            )));
        }
        Ok(Self {
            threshold,
            arc_length,
        })
    }

    /// Returns the intensity threshold `t`.
    #[must_use]
    pub const fn threshold(self) -> f32 {
        self.threshold
    }

    /// Returns the required arc length `n` — the `n` in "FAST-n".
    #[must_use]
    pub const fn arc_length(self) -> usize {
        self.arc_length
    }
}

// ─── FastParams ──────────────────────────────────────────────────────────────

/// A [`SegmentTest`] plus the suppression radius the detection stage needs.
///
/// The counterpart of [`CornerParams`](super::CornerParams), and shorter by
/// two fields: there is no window σ because the segment test does not
/// integrate anything, and no separate response threshold because the test's
/// own `t` is already the right one — [`fast_score_map`] is built so that a
/// pixel is a corner at `t` exactly when its score is at least `t`.
///
/// `nms_radius` is the suppression window's radius; see
/// [`NmsRadius`](super::NmsRadius) for what it means and why it is at
/// least 1.
///
/// There is deliberately no `Default` — a default threshold would be a claim
/// about *your* images that this crate is not in a position to make.
///
/// # Example
///
/// ```
/// use fovea::features::detect::{FastParams, NmsRadius, SegmentTest};
///
/// const PARAMS: FastParams =
///     FastParams::new(SegmentTest::new(0.08, 9).unwrap(), NmsRadius::new(3).unwrap());
/// assert_eq!(PARAMS.nms_radius().get(), 3);
/// assert_eq!(PARAMS.test().arc_length(), 9);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FastParams {
    test: SegmentTest,
    nms_radius: NmsRadius,
}

impl FastParams {
    /// Creates detection parameters. **Total**: both fields carry their own
    /// invariants, so there is nothing left to validate and no `try_new`.
    #[must_use]
    pub const fn new(test: SegmentTest, nms_radius: NmsRadius) -> Self {
        Self { test, nms_radius }
    }

    /// Returns the segment test.
    #[must_use]
    pub const fn test(self) -> SegmentTest {
        self.test
    }

    /// Returns the non-maximum-suppression radius.
    #[must_use]
    pub const fn nms_radius(self) -> NmsRadius {
        self.nms_radius
    }
}

// ─── The score ───────────────────────────────────────────────────────────────

/// Scores one pixel against the segment test, or reports that the border
/// policy declines to score it.
///
/// The **specification** of the detector, and the unit
/// [`fast_score_map`] applies at every position: everything else in this
/// module is scanning and selection.
///
/// The score is the **largest threshold at which this pixel is still a
/// corner** — the smallest margin along its best arc — or `0` if it is not a
/// corner at `test`'s own threshold. So for every `t >= test.threshold()`
/// the segment test passes at `t` **exactly when** `score >= t`, which is
/// what lets [`fast`] use one number for both the test and the peak stage,
/// and what makes raising the threshold on an already-computed map exact.
///
/// `0` means "no corner here": a flat field, a straight edge, a corner too
/// faint for this test, or a position the border policy skipped. Margins
/// *below* the threshold are deliberately not reported — the four-cardinal
/// early rejection discards those pixels before computing one, which is where
/// the detector's speed comes from. To see a fainter corner, lower the
/// threshold and build the map again.
///
/// Returns [`None`] when the position is not scored: outside the image, or —
/// with the [`Skip`](crate::border::Skip) policy — inside the
/// [`FAST_RING_RADIUS`]-pixel border where the ring does not fit. The
/// distinction is in the signature rather than in a doc note because
/// "declined" and "scored zero" are different answers, even though
/// [`fast_score_map`] writes both as `0.0`.
///
/// # Example
///
/// ```
/// use fovea::border::{Clamp, Skip};
/// use fovea::features::detect::{fast_score_at, SegmentTest};
/// use fovea::image::Image;
/// use fovea::pixel::MonoF32;
///
/// // A bright quadrant: (8, 8) is its top-left corner pixel.
/// let image: Image<MonoF32> = Image::generate(24, 24, |x, y| {
///     MonoF32::new(if x >= 8 && y >= 8 { 1.0 } else { 0.0 })
/// });
/// let test = SegmentTest::new(0.1, 9).unwrap();
///
/// // The corner scores the full contrast: it is a corner up to t = 1.0.
/// let corner = fast_score_at(&image, 8, 8, test, &Skip).expect("well inside");
/// assert!((corner - 1.0).abs() < 1e-6);
///
/// // A pixel on the straight edge below it is not a corner at any threshold.
/// assert_eq!(fast_score_at(&image, 8, 16, test, &Skip), Some(0.0));
///
/// // `Skip` declines the 3-pixel border; `Clamp` extrapolates into it.
/// assert_eq!(fast_score_at(&image, 1, 1, test, &Skip), None);
/// assert_eq!(fast_score_at(&image, 1, 1, test, &Clamp), Some(0.0));
/// // Outside the image, every policy declines.
/// assert_eq!(fast_score_at(&image, 24, 0, test, &Clamp), None);
/// ```
#[must_use]
pub fn fast_score_at<I, P, Acc, B>(
    image: &I,
    x: usize,
    y: usize,
    test: SegmentTest,
    border: &B,
) -> Option<f32>
where
    I: RasterImage<Pixel = P>,
    P: Copy + LinearPixel<f32, Accumulator = Acc>,
    Acc: SingleChannel,
    f64: From<Acc::Channel>,
    B: BorderPolicy<I>,
{
    let region = scored_region(image, border);
    if x < region.left() || x >= region.right() || y < region.top() || y >= region.bottom() {
        return None;
    }
    Some(score_in_region(image, x, y, test, border))
}

/// Scores every pixel the border policy admits, as an image.
///
/// The public intermediate of [`fast`], the counterpart of
/// [`corner_response_map`](super::corner_response_map): the score map is
/// worth looking at, and thresholding it differently, or feeding it to
/// [`corner_peaks`], needs no new API.
///
/// The map is **zero wherever there is no corner** and the detection's own
/// strength wherever there is one, so it is already thresholded by
/// `test.threshold()` — see [`fast_score_at`] for exactly what the number
/// means and why sub-threshold margins are not in it.
///
/// It is always the **same size as the input**, so a position in it is a
/// position in the image and no offset has to be threaded through the peak
/// stage. Positions the policy declines — the
/// [`FAST_RING_RADIUS`]-pixel border under [`Skip`](crate::border::Skip) —
/// are written as `0.0` too, which means the same thing here. Use
/// [`fast_score_at`], whose [`None`] is distinct from its `Some(0.0)`, where
/// the two must be told apart.
///
/// The output pixel type is [`MonoF32`] whatever the input is, because
/// [`Corner::response`](crate::features::HasResponse::response) is `f32`:
/// precision the keypoint cannot carry would be precision this map only
/// pretends to have. The scores themselves are computed in `f64` and are
/// exact for every 8- and 16-bit input.
///
/// # Example
///
/// ```
/// use fovea::border::Skip;
/// use fovea::features::detect::{fast_score_map, SegmentTest};
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::{Mono8, MonoF32};
///
/// // A white square on black.
/// let image: Image<Mono8> = Image::generate(32, 32, |x, y| {
///     Mono8::new(if (10..22).contains(&x) && (10..22).contains(&y) { 255 } else { 0 })
/// });
///
/// let scores: Image<MonoF32> = fast_score_map(&image, SegmentTest::new(20.0, 9).unwrap(), &Skip);
/// assert_eq!(scores.size(), image.size());
///
/// // A corner of the square scores the full 255 levels of contrast; the
/// // middle of an edge and the flat interior score nothing at all.
/// assert_eq!(scores.pixel_at(10, 10).value(), 255.0);
/// assert_eq!(scores.pixel_at(16, 10).value(), 0.0);
/// assert_eq!(scores.pixel_at(16, 16).value(), 0.0);
/// ```
#[must_use]
pub fn fast_score_map<I, P, Acc, B>(image: &I, test: SegmentTest, border: &B) -> Image<MonoF32>
where
    I: RasterImage<Pixel = P>,
    P: Copy + LinearPixel<f32, Accumulator = Acc>,
    Acc: SingleChannel,
    f64: From<Acc::Channel>,
    B: BorderPolicy<I>,
{
    let region = scored_region(image, border);
    let (width, height) = (image.width(), image.height());
    let mut out = Image::fill(width, height, MonoF32::new(0.0));

    // The columns whose ring fits horizontally, intersected with the region.
    // `Skip` already excludes the border, so under it this *is* the region
    // and the two cold strips below are empty, which is the point.
    // Both bounds are clamped into the region, in both directions: on an
    // image narrower than the ring, `Clamp` still scores every column, so
    // `region.right()` can be below `FAST_RING_RADIUS` and an unclamped
    // `hot_left` would index past the row.
    let hot_left = region.left().max(FAST_RING_RADIUS).min(region.right());
    let hot_right = region
        .right()
        .min(width.saturating_sub(FAST_RING_RADIUS))
        .max(hot_left);

    for y in region.top()..region.bottom() {
        let rows_fit = y >= FAST_RING_RADIUS && y + FAST_RING_RADIUS < height;
        let (hot_left, hot_right) = if rows_fit {
            (hot_left, hot_right)
        } else {
            (region.left(), region.left())
        };

        // The seven scan lines the ring spans, fetched once for the whole
        // hot span instead of once per sample.
        let rows: Option<[&[P]; FOOTPRINT]> = rows_fit
            .then(|| core::array::from_fn(|offset| image.row(y - FAST_RING_RADIUS + offset)));

        let row = out.row_mut(y);
        let cold = |slots: &mut [MonoF32], from: usize| {
            for (offset, slot) in slots.iter_mut().enumerate() {
                *slot = MonoF32::new(score_in_region(image, from + offset, y, test, border));
            }
        };
        cold(&mut row[region.left()..hot_left], region.left());
        cold(&mut row[hot_right..region.right()], hot_right);

        if let Some(rows) = rows {
            for (offset, slot) in row[hot_left..hot_right].iter_mut().enumerate() {
                *slot = MonoF32::new(score_interior(&rows, hot_left + offset, test));
            }
        }
    }
    out
}

// ─── Detectors ───────────────────────────────────────────────────────────────

/// Detects corners with the segment test: score map, then peak selection.
///
/// The composed pipeline, and exactly that — [`fast_score_map`] followed by
/// [`corner_peaks`] at the test's own threshold. Corners are returned in
/// raster order, positioned at pixel centres; ranking is the separate,
/// already-existing step
/// [`retain_top_n`](crate::features::retain_top_n).
///
/// # Choosing the border policy
///
/// [`Skip`](crate::border::Skip) is the natural default: the ring does not
/// fit within [`FAST_RING_RADIUS`] pixels of the frame edge, and a detection
/// there would be made from samples that were invented rather than measured.
/// The alternative is any full-frame policy —
/// [`Clamp`](crate::border::Clamp), [`Mirror`](crate::border::Mirror) — which
/// extends the image and lets corners be reported against the edge. Nothing
/// FAST-specific is introduced for this; it is the same vocabulary
/// [`sobel_x`](crate::transform::sobel_x) and every other neighbourhood
/// operation in the crate takes.
///
/// # Example
///
/// ```
/// use fovea::border::Skip;
/// use fovea::features::detect::{fast, FastParams, NmsRadius, SegmentTest};
/// use fovea::features::{retain_top_n, HasPosition, HasResponse};
/// use fovea::image::Image;
/// use fovea::pixel::MonoF32;
///
/// // Two squares, the left one at full contrast, the right one faint.
/// let image: Image<MonoF32> = Image::generate(48, 24, |x, y| {
///     let inside = |x0: usize| (x0..x0 + 10).contains(&x) && (7..17).contains(&y);
///     MonoF32::new(if inside(6) { 1.0 } else if inside(30) { 0.3 } else { 0.0 })
/// });
///
/// // 8 % contrast, FAST-9, corners at least 3 px apart.
/// let params = FastParams::new(SegmentTest::new(0.08, 9).unwrap(), NmsRadius::new(3).unwrap());
/// let mut corners = fast(&image, params, &Skip);
/// assert_eq!(corners.len(), 8); // four per square
///
/// // The score is in intensity units, so the faint square's corners score
/// // 0.3 — not 0.3⁴ as a Harris response would.
/// retain_top_n(&mut corners, 4);
/// assert!(corners.iter().all(|c| c.position().x < 24.0), "{corners:?}");
/// assert!(corners.iter().all(|c| (c.response() - 1.0).abs() < 1e-6));
/// ```
#[must_use]
pub fn fast<I, P, Acc, B>(image: &I, params: FastParams, border: &B) -> Vec<Corner>
where
    I: RasterImage<Pixel = P>,
    P: Copy + LinearPixel<f32, Accumulator = Acc>,
    Acc: SingleChannel,
    f64: From<Acc::Channel>,
    B: BorderPolicy<I>,
{
    let scores = fast_score_map(image, params.test(), border);
    score_peaks(&scores, params.test().threshold(), params.nms_radius().get())
}

/// Detects corners on a pyramid level, reporting them in the **base-image**
/// frame.
///
/// The multi-resolution variant of [`fast`], and the same detector — the
/// exact counterpart of
/// [`detect_corners_in_level`](super::detect_corners_in_level), including its
/// deliberate refusal to upgrade the result type. Detection runs in the
/// level's own coordinates and every position is lifted through
/// [`Corner::from_level`](crate::features::Corner::from_level), so results
/// from different levels are comparable and concatenate.
///
/// The output stays [`Corner`] rather than
/// [`ScaleKeypoint`](crate::features::ScaleKeypoint): the segment test has no
/// scale parameter at all — not even a window σ — so a level's σ is even more
/// plainly something the detector did not choose.
///
/// # Example
///
/// ```
/// use fovea::CoordinateF64;
/// use fovea::border::Skip;
/// use fovea::features::HasPosition;
/// use fovea::features::detect::{fast_in_level, FastParams, NmsRadius, SegmentTest};
/// use fovea::image::{Image, ScaledImage};
/// use fovea::pixel::MonoF32;
/// use fovea::{pixel_distance, sigma};
/// use fovea::transform::pyr_down;
///
/// let base: Image<MonoF32> = Image::generate(64, 64, |x, y| {
///     MonoF32::new(if (20..44).contains(&x) && (20..44).contains(&y) { 1.0 } else { 0.0 })
/// });
///
/// // Octave 1: pyr_down keeps even samples — distance 2, origin unshifted.
/// let level = ScaledImage::new(
///     pyr_down(&base),
///     pixel_distance!(2.0),
///     CoordinateF64::new(0.0, 0.0),
///     sigma!(1.0),
/// );
///
/// let params = FastParams::new(SegmentTest::new(0.15, 9).unwrap(), NmsRadius::new(2).unwrap());
/// let corners = fast_in_level(&level, params, &Skip);
///
/// // Found on a 32×32 level, reported in the 64×64 base frame.
/// assert_eq!(corners.len(), 4, "{corners:?}");
/// assert!(corners.iter().any(|c| c.position().x > 32.0), "{corners:?}");
/// ```
#[must_use]
pub fn fast_in_level<L, P, Acc, B>(level: &L, params: FastParams, border: &B) -> Vec<Corner>
where
    L: Decimated<Pixel = P>,
    P: Copy + LinearPixel<f32, Accumulator = Acc>,
    Acc: SingleChannel,
    f64: From<Acc::Channel>,
    B: BorderPolicy<Image<P>>,
{
    let scores = fast_score_map(level.as_image(), params.test(), border);
    scan_score_peaks(&scores, params.test().threshold(), params.nms_radius().get())
        .into_iter()
        .map(|(local, response)| Corner::from_level(level, local, response))
        .collect()
}

// ─── Internals ───────────────────────────────────────────────────────────────

/// [`corner_peaks`] at the concrete score-map type.
///
/// The map is always `Image<MonoF32>`, and pinning that here is what keeps
/// the generic peak stage from being resolved against *this* module's
/// `f64: From<P::Channel>` bound: with two type parameters to infer and an
/// environment bound of exactly that shape, the compiler unifies the map's
/// pixel type with the input image's.
fn score_peaks(scores: &Image<MonoF32>, threshold: f32, radius: usize) -> Vec<Corner> {
    corner_peaks(scores, threshold, radius)
}

/// [`scan_peaks`] at the concrete score-map type; see [`score_peaks`].
fn scan_score_peaks(
    scores: &Image<MonoF32>,
    threshold: f32,
    radius: usize,
) -> Vec<(CoordinateF64, f32)> {
    scan_peaks(scores, threshold, radius)
}

/// The set of positions the border policy admits for a ring-sized footprint,
/// clipped to the image so a custom policy cannot widen it.
fn scored_region<I, B>(image: &I, border: &B) -> Rectangle
where
    I: ImageView,
    I::Pixel: Copy,
    B: BorderPolicy<I>,
{
    let size = image.size();
    let region = border.output_region(
        size,
        Size::new(FOOTPRINT, FOOTPRINT),
        (FAST_RING_RADIUS, FAST_RING_RADIUS),
    );
    let left = region.left().min(size.width);
    let top = region.top().min(size.height);
    Rectangle::new(
        (left, top),
        (
            region.right().min(size.width) - left,
            region.bottom().min(size.height) - top,
        ),
    )
}

/// Scores `(x, y)`, which the caller has established is inside
/// [`scored_region`] — so the centre read is in bounds and the policy is
/// willing to supply whatever ring samples fall outside the frame.
///
/// The four cardinal ring samples are read and tested first; the other twelve
/// and the arc scan are reached only by the pixels [`cardinals_admit`] cannot
/// rule out. On the `benches/features.rs` texture (512×512 `Mono8`,
/// 2026-08-07) that is worth ≈1.6× at `arc_length = 9`, ≈5.8× at 12 and ≈14×
/// at 16 — the filter needs `arc_length / 4` cardinals, so it grows teeth as
/// the arc gets longer, and is weakest exactly where FAST-9 needs it most.
fn score_in_region<I, P, Acc, B>(
    image: &I,
    x: usize,
    y: usize,
    test: SegmentTest,
    border: &B,
) -> f32
where
    I: RasterImage<Pixel = P>,
    P: Copy + LinearPixel<f32, Accumulator = Acc>,
    Acc: SingleChannel,
    f64: From<Acc::Channel>,
    B: BorderPolicy<I>,
{
    let (w, h) = (image.width() as isize, image.height() as isize);
    let sample = |index: usize| {
        let offset = FAST_RING[index];
        let (nx, ny) = (
            x as isize + offset.dx as isize,
            y as isize + offset.dy as isize,
        );
        // In bounds is a direct read; only genuinely outside positions reach
        // the policy — which matters because `Skip` *panics* rather than
        // inventing a sample, and must never be asked inside its own region.
        let pixel = if (0..w).contains(&nx) && (0..h).contains(&ny) {
            image.row(ny as usize)[nx as usize]
        } else {
            border.pixel_at(image, nx, ny)
        };
        intensity::<P, Acc>(pixel)
    };

    score_ring(intensity::<P, Acc>(image.row(y)[x]), test, sample)
}

/// [`score_in_region`] for a position whose whole ring lies inside the frame,
/// reading the seven scan lines the caller has already hoisted.
///
/// `rows[k]` is image row `y − FAST_RING_RADIUS + k`, and the caller
/// guarantees `FAST_RING_RADIUS <= x < width − FAST_RING_RADIUS`. Both
/// promises together are what remove the per-sample bounds test and the
/// per-sample `row()` call. The border policy is unreachable from here by
/// construction, so it is not a parameter.
fn score_interior<P, Acc>(rows: &[&[P]; FOOTPRINT], x: usize, test: SegmentTest) -> f32
where
    P: Copy + LinearPixel<f32, Accumulator = Acc>,
    Acc: SingleChannel,
    f64: From<Acc::Channel>,
{
    let sample = |index: usize| {
        let (row, dx) = FAST_RING_ROWS[index];
        intensity::<P, Acc>(rows[row][x.wrapping_add_signed(dx)])
    };

    score_ring(intensity::<P, Acc>(rows[FAST_RING_RADIUS][x]), test, sample)
}

/// The centre-relative intensity of one pixel, in the accumulator's units.
///
/// `to_accumulator` is the crate's named widening and does **not** rescale,
/// so a `Mono8` 255 arrives as 255.0 and the threshold stays in grey levels.
#[inline(always)]
fn intensity<P, Acc>(pixel: P) -> f64
where
    P: Copy + LinearPixel<f32, Accumulator = Acc>,
    Acc: SingleChannel,
    f64: From<Acc::Channel>,
{
    f64::from(pixel.to_accumulator().channel(0))
}

/// The segment test itself, over a `sample(index) -> intensity` callback.
///
/// The two scans differ only in how a ring position is fetched (one consults
/// the border policy, the other indexes hoisted rows), so the ordering that
/// makes the detector fast (cardinals first, the other twelve only for the
/// pixels [`cardinals_admit`] cannot rule out) lives here, once.
#[inline(always)]
fn score_ring(centre: f64, test: SegmentTest, sample: impl Fn(usize) -> f64) -> f32 {
    // Fully qualified: an unadorned `f64::from` here resolves against a
    // caller's `f64: From<Acc::Channel>` bound instead of `f32`.
    let threshold = <f64 as From<f32>>::from(test.threshold());

    let mut ring = [0.0f64; 16];
    for index in CARDINALS {
        ring[index] = sample(index);
    }
    if !cardinals_admit(centre, &ring, test.arc_length(), threshold) {
        return 0.0;
    }
    for index in NON_CARDINALS {
        ring[index] = sample(index);
    }

    let score = segment_score(centre, &ring, test.arc_length());
    if score >= threshold {
        score as f32
    } else {
        0.0
    }
}

/// The four ring positions at the compass points: indices 0, 4, 8 and 12 of
/// [`FAST_RING`], i.e. straight up, right, down and left of the centre.
const CARDINALS: [usize; 4] = [0, 4, 8, 12];

/// The other twelve ring positions, in ring order: the complement of
/// [`CARDINALS`], spelled out rather than filtered at run time.
const NON_CARDINALS: [usize; 12] = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15];

/// The classical high-speed rejection test, generalised to any arc length.
///
/// Returns whether an arc could *still* pass — never whether one does. It is
/// therefore an optimization and not part of the specification: removing it
/// cannot change an answer, only the time taken to reach it.
///
/// The [`CARDINALS`] are four apart on a ring of sixteen, so a window of
/// `arc_length` consecutive positions contains at least `arc_length / 4` of
/// them however it is placed. A passing arc's pixels are all bright by
/// `threshold` (or all dark by it), so at least that many cardinals must be
/// too — and a pixel where neither count reaches it cannot pass, whatever the
/// other twelve samples say. For the classical FAST-12 this is the familiar
/// "three of the four", and for FAST-9 it is two.
///
/// A `NaN` cardinal counts as neither bright nor dark, which is the safe
/// direction: an arc containing it does not pass anyway
/// (see [`segment_score`]), so declining to count it cannot reject an arc
/// that would have survived.
fn cardinals_admit(centre: f64, ring: &[f64; 16], arc_length: usize, threshold: f64) -> bool {
    let needed = arc_length / 4;
    let (mut bright, mut dark) = (0usize, 0usize);
    for index in CARDINALS {
        let difference = ring[index] - centre;
        if difference >= threshold {
            bright += 1;
        } else if -difference >= threshold {
            dark += 1;
        }
    }
    bright >= needed || dark >= needed
}

/// The largest threshold at which `centre` still passes the segment test
/// against `ring`, floored at zero.
///
/// Every one of the 16 possible arc positions is tried, with no early exit:
/// this is what the detector *means*, and [`cardinals_admit`] is the separate
/// optimization that keeps most pixels from reaching it.
///
/// An arc passes at threshold `t` when every one of its `arc_length` pixels
/// differs from the centre by at least `t` **in the same direction**, so the
/// largest `t` that arc admits is the smallest of its signed differences; the
/// score is the best such value over all arcs and both directions.
///
/// A `NaN` sample discards the arcs that contain it rather than being
/// silently skipped: `f64::min` would drop it and let the remaining samples
/// decide, which would report a corner from data that is not there. Arcs
/// clear of the `NaN` still count, so one bad sample costs the detection only
/// if every arc needs it; a `NaN` centre costs all of them. This is
/// [`corner_peaks`]'s "neither wins nor survives" rule at the sample level.
///
/// # Why this is not a sliding-window minimum
///
/// The double loop is `O(16 · arc_length)`, and what it computes is a
/// sliding-window minimum of the signed differences over a circular buffer of
/// 16 (plus a maximum, for the dark direction), which a doubled array and a
/// prefix/suffix block decomposition would give in `O(16 + arc_length)`. That
/// was built and measured, and it lost: on a 512x512 texture the block
/// decomposition is ≈6 % *slower* at `arc_length = 9`, the case that matters
/// most, because four 31-element scratch arrays cost more than the 144 cheap
/// comparisons they replace. It won ≈6 % at 12 and was a wash at 16, which
/// does not pay for a second scoring path. Sixteen elements is simply too few
/// for the asymptotics to matter.
fn segment_score(centre: f64, ring: &[f64; 16], arc_length: usize) -> f64 {
    let mut best = f64::NEG_INFINITY;

    for start in 0..16 {
        let (mut bright, mut dark) = (f64::INFINITY, f64::INFINITY);
        let mut usable = true;
        for k in 0..arc_length {
            let difference = ring[(start + k) % 16] - centre;
            if difference.is_nan() {
                usable = false;
                break;
            }
            if difference < bright {
                bright = difference;
            }
            if -difference < dark {
                dark = -difference;
            }
        }
        if !usable {
            continue;
        }
        if bright > best {
            best = bright;
        }
        if dark > best {
            best = dark;
        }
    }

    // Below zero the value would say "fails even at t = 0", which no caller
    // can act on: `SegmentTest` rejects t <= 0. Flooring keeps one meaning
    // for one number — 0 is "no corner here" — and keeps the map displayable.
    if best > 0.0 { best } else { 0.0 }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Coordinate;
    use crate::border::{Clamp, Constant, Mirror, Skip};
    use crate::features::{HasPosition, HasResponse, retain_top_n};
    use crate::image::{Pyramid, ScaledImage};
    use crate::pixel::{Mono8, Mono16, MonoF64};
    use crate::transform::{pyr_down, rotate_90};
    use crate::{pixel_distance, sigma};

    // ── Fixtures ────────────────────────────────────────────────────────

    /// A white square on a black field: four corners, four straight edges.
    fn square(n: usize, lo: usize, hi: usize) -> Image<MonoF32> {
        Image::generate(n, n, |x, y| {
            let inside = (lo..hi).contains(&x) && (lo..hi).contains(&y);
            MonoF32::new(if inside { 1.0 } else { 0.0 })
        })
    }

    /// A single bright pixel on black: the one fixture the segment test
    /// answers *exactly*, because its whole ring is on the far side of the
    /// step and no neighbouring pixel comes close to tying with it.
    fn dot(n: usize, at: (usize, usize)) -> Image<MonoF32> {
        Image::generate(n, n, |x, y| {
            MonoF32::new(if (x, y) == at { 1.0 } else { 0.0 })
        })
    }

    /// The four **pixels** of `square(_, lo, hi)` that carry the corners.
    fn square_corner_pixels(lo: usize, hi: usize) -> [(f64, f64); 4] {
        let (a, b) = (lo as f64, (hi - 1) as f64);
        [(a, a), (b, a), (a, b), (b, b)]
    }

    /// Asserts that `corners` is one detection per true corner, each within
    /// `tolerance` pixels of it.
    ///
    /// Exact positions are *not* asserted for a step-edge square: several
    /// pixels around each corner tie at the full contrast (see
    /// `a_right_angle_saturates_into_a_tied_cluster`), and the plateau rule
    /// then reports each cluster's raster-first member, which is the true
    /// corner only for the top-left one.
    fn assert_one_per_corner(corners: &[Corner], truth: [(f64, f64); 4], tolerance: f64) {
        assert_eq!(corners.len(), 4, "{corners:?}");
        for &(tx, ty) in &truth {
            let nearest = corners
                .iter()
                .map(|c| {
                    let p = c.position();
                    ((p.x - tx).powi(2) + (p.y - ty).powi(2)).sqrt()
                })
                .fold(f64::INFINITY, f64::min);
            assert!(
                nearest <= tolerance,
                "({tx}, {ty}) unmatched in {corners:?}"
            );
        }
    }

    /// The `(x, y)` positions of `corners`, for set comparisons.
    fn positions(corners: &[Corner]) -> Vec<(f64, f64)> {
        corners
            .iter()
            .map(|c| (c.position().x, c.position().y))
            .collect()
    }

    /// A ring whose first `bright` entries are `+contrast` above the centre
    /// and the rest exactly at it — the minimal arc fixture.
    fn arc_ring(bright: usize, contrast: f64) -> [f64; 16] {
        let mut ring = [0.0; 16];
        for slot in ring.iter_mut().take(bright) {
            *slot = contrast;
        }
        ring
    }

    // ── The ring geometry ───────────────────────────────────────────────

    #[test]
    fn the_ring_is_a_closed_bresenham_circle_of_radius_three() {
        assert_eq!(FAST_RING.len(), 16);
        assert_eq!(FAST_RING_RADIUS, 3);

        for (i, &offset) in FAST_RING.iter().enumerate() {
            // On the circle: the Bresenham rasterisation of radius 3 puts
            // every offset at squared distance 8, 9 or 10 — the diagonals
            // (±2, ±2) sit at 2.83, not 3.
            let squared = offset.dx * offset.dx + offset.dy * offset.dy;
            assert!((8..=10).contains(&squared), "offset {i} is off the ring");
            // Consecutive offsets are 8-neighbours, so "contiguous" means
            // what the arc test assumes — including 15 → 0.
            let next = FAST_RING[(i + 1) % 16];
            assert!(
                (next.dx - offset.dx).abs() <= 1 && (next.dy - offset.dy).abs() <= 1,
                "offsets {i} and {} are not adjacent",
                (i + 1) % 16
            );
        }
    }

    #[test]
    fn the_ring_is_symmetric_under_a_quarter_turn() {
        // Index i + 4 is index i rotated 90° clockwise: (x, y) ↦ (−y, x).
        for (i, &offset) in FAST_RING.iter().enumerate() {
            assert_eq!(
                FAST_RING[(i + 4) % 16],
                Offset::new(-offset.dy, offset.dx),
                "at index {i}"
            );
        }
    }

    // ── SegmentTest invariants ──────────────────────────────────────────

    #[test]
    fn segment_test_accepts_the_documented_range() {
        for n in 9..=16 {
            assert_eq!(SegmentTest::try_new(0.1, n).unwrap().arc_length(), n);
        }
        const FAST9: SegmentTest = SegmentTest::new(0.08, 9).unwrap();
        assert_eq!(FAST9.threshold(), 0.08);
        assert_eq!(FAST9.arc_length(), 9);
    }

    #[test]
    fn segment_test_rejects_a_threshold_that_admits_everything() {
        for threshold in [0.0, -0.1, f32::NAN, f32::INFINITY] {
            match SegmentTest::try_new(threshold, 9).unwrap_err() {
                Error::InvalidParameter(reason) => assert!(
                    reason.contains("threshold"),
                    "reason {reason:?} does not mention the threshold"
                ),
                other => panic!("expected InvalidParameter, got {other:?}"),
            }
        }
    }

    #[test]
    fn segment_test_rejects_an_arc_that_admits_edges_or_nothing() {
        for n in [0, 1, 8, 17, 100] {
            match SegmentTest::try_new(0.1, n).unwrap_err() {
                Error::InvalidParameter(reason) => assert!(
                    reason.contains("arc_length"),
                    "reason {reason:?} does not mention arc_length"
                ),
                other => panic!("expected InvalidParameter, got {other:?}"),
            }
        }
    }

    #[test]
    fn segment_test_new_rejects_a_zero_threshold() {
        assert!(SegmentTest::new(0.0, 9).is_none());
    }

    #[test]
    fn segment_test_new_rejects_an_edge_admitting_arc() {
        assert!(SegmentTest::new(0.1, 8).is_none());
        assert!(SegmentTest::new(0.1, 17).is_none());
    }

    #[test]
    fn fast_params_round_trip() {
        const PARAMS: FastParams =
            FastParams::new(SegmentTest::new(0.08, 12).unwrap(), NmsRadius::new(4).unwrap());
        assert_eq!(PARAMS.nms_radius().get(), 4);
        assert_eq!(PARAMS.test(), SegmentTest::new(0.08, 12).unwrap());
    }

    #[test]
    fn nms_radius_carries_the_at_least_one_invariant() {
        // The validation FastParams and CornerParams used to hand-write.
        assert!(NmsRadius::new(0).is_none());
        match NmsRadius::try_new(0).unwrap_err() {
            Error::InvalidParameter(reason) => assert!(reason.contains("radius")),
            other => panic!("expected InvalidParameter, got {other:?}"),
        }
        assert_eq!(NmsRadius::try_new(2).unwrap().get(), 2);
    }

    // ── segment_score: the specification ────────────────────────────────

    #[test]
    fn a_flat_ring_scores_zero() {
        assert_eq!(segment_score(0.5, &[0.5; 16], 9), 0.0);
        assert_eq!(segment_score(0.0, &[0.0; 16], 16), 0.0);
    }

    #[test]
    fn an_arc_one_short_of_the_requirement_scores_zero() {
        // Eight contiguous bright pixels is exactly what a straight edge
        // produces, and is exactly what FAST-9 must refuse.
        assert_eq!(segment_score(0.0, &arc_ring(8, 1.0), 9), 0.0);
        assert!(segment_score(0.0, &arc_ring(9, 1.0), 9) > 0.0);
    }

    #[test]
    fn the_score_is_the_arc_weakest_link() {
        // Nine bright pixels, one of them only half as far from the centre:
        // the arc survives only up to that pixel's own margin.
        let mut ring = arc_ring(9, 1.0);
        ring[4] = 0.4;
        assert!((segment_score(0.0, &ring, 9) - 0.4).abs() < 1e-12);
    }

    #[test]
    fn an_arc_may_wrap_around_index_zero() {
        // Indices 12..16 and 0..5 — contiguous on the ring, not in the array.
        let mut ring = [0.0; 16];
        for i in (12..16).chain(0..5) {
            ring[i] = 1.0;
        }
        assert!((segment_score(0.0, &ring, 9) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn the_test_is_symmetric_in_brightness() {
        // A dark arc scores exactly what the mirrored bright arc does — the
        // detector finds dark corners on light backgrounds too.
        let bright = arc_ring(9, 1.0);
        let dark: [f64; 16] = core::array::from_fn(|i| -bright[i]);
        assert_eq!(segment_score(0.0, &bright, 9), segment_score(0.0, &dark, 9));
    }

    #[test]
    fn the_score_is_exactly_the_largest_passing_threshold() {
        // The invariant `fast` relies on to use one number in both stages.
        // A hand-rolled reference test over a spread of thresholds.
        let mut ring = arc_ring(11, 0.9);
        ring[2] = 0.62;
        ring[9] = 0.4; // outside the bright run: must not matter
        let score = segment_score(0.0, &ring, 9);

        let passes = |t: f64| {
            (0..16).any(|start| {
                (0..9).all(|k| ring[(start + k) % 16] >= t)
                    || (0..9).all(|k| ring[(start + k) % 16] <= -t)
            })
        };
        for step in 1..200 {
            let t = f64::from(step) * 0.005;
            assert_eq!(passes(t), score >= t, "at t = {t}, score = {score}");
        }
    }

    #[test]
    fn a_longer_arc_requirement_never_scores_higher() {
        // Monotone in n: FAST-12 is strictly more selective than FAST-9.
        let mut ring = arc_ring(12, 1.0);
        ring[10] = 0.3;
        let (nine, twelve) = (segment_score(0.0, &ring, 9), segment_score(0.0, &ring, 12));
        assert!(nine >= twelve, "{nine} < {twelve}");
        assert!((nine - 1.0).abs() < 1e-12);
        assert!((twelve - 0.3).abs() < 1e-12);
    }

    #[test]
    fn a_full_ring_requirement_needs_every_pixel() {
        let mut ring = [1.0; 16];
        assert!((segment_score(0.0, &ring, 16) - 1.0).abs() < 1e-12);
        ring[7] = 0.0;
        assert_eq!(segment_score(0.0, &ring, 16), 0.0);
    }

    #[test]
    fn a_nan_sample_poisons_only_the_arcs_through_it() {
        let mut ring = [1.0; 16];
        ring[0] = f64::NAN;
        // Fifteen clean bright pixels remain, so FAST-9 still finds an arc …
        assert!((segment_score(0.0, &ring, 9) - 1.0).abs() < 1e-12);
        // … but nothing can span the whole ring any more.
        assert_eq!(segment_score(0.0, &ring, 16), 0.0);
        // A NaN centre poisons every arc.
        assert_eq!(segment_score(f64::NAN, &[1.0; 16], 9), 0.0);
    }

    // ── The early rejection is an optimization, not a rule ──────────────

    #[test]
    fn the_cardinal_bound_holds_for_every_arc_placement() {
        // What makes `cardinals_admit` sound: however an arc of `n` is placed
        // on the ring, it covers at least `n / 4` of the four cardinals.
        for arc_length in 9..=16usize {
            for start in 0..16 {
                let covered = (0..arc_length)
                    .filter(|k| CARDINALS.contains(&((start + k) % 16)))
                    .count();
                assert!(
                    covered >= arc_length / 4,
                    "n = {arc_length}, start = {start}: only {covered} cardinals"
                );
            }
        }
    }

    #[test]
    fn the_early_rejection_never_changes_an_answer() {
        // The optimization must be invisible: on every pixel of a busy
        // texture, at every arc length, the shipped path agrees with the
        // plain scan thresholded by hand.
        let image: Image<MonoF32> = Image::generate(64, 64, |x, y| {
            let checker = if (x / 7 + y / 5) % 2 == 0 { 0.2 } else { 0.8 };
            let ripple = ((x * 5 + y * 3) % 11) as f32 * 0.01;
            MonoF32::new(checker + ripple)
        });

        for arc_length in [9usize, 12, 16] {
            for &threshold in &[0.05f32, 0.2, 0.5] {
                let test = SegmentTest::new(threshold, arc_length).unwrap();
                let scores = fast_score_map(&image, test, &Skip);

                for y in FAST_RING_RADIUS..64 - FAST_RING_RADIUS {
                    for x in FAST_RING_RADIUS..64 - FAST_RING_RADIUS {
                        let ring: [f64; 16] = core::array::from_fn(|i| {
                            let at = Coordinate::new(x, y)
                                .checked_add(FAST_RING[i])
                                .expect("the ring fits inside the scored margin");
                            f64::from(image.pixel_at(at.x, at.y).value())
                        });
                        let plain = segment_score(
                            f64::from(image.pixel_at(x, y).value()),
                            &ring,
                            arc_length,
                        );
                        let expected = if plain >= f64::from(threshold) {
                            plain as f32
                        } else {
                            0.0
                        };
                        assert_eq!(
                            scores.pixel_at(x, y).value(),
                            expected,
                            "n = {arc_length}, t = {threshold}, at ({x}, {y})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_interior_and_boundary_paths_agree() {
        // `fast_score_map` splits each scan line into a hot span that reads
        // hoisted rows and cold strips that go through the border policy;
        // `fast_score_at` always takes the second path. Under `Clamp` the
        // whole image is scored, so every pixel is a comparison, and the
        // 3-pixel frame is where the split lands.
        let image: Image<MonoF32> = Image::generate(24, 20, |x, y| {
            let checker = if (x / 5 + y / 3) % 2 == 0 { 0.15 } else { 0.85 };
            MonoF32::new(checker + ((x * 3 + y * 7) % 9) as f32 * 0.01)
        });

        for arc_length in [9usize, 12, 16] {
            let test = SegmentTest::new(0.05, arc_length).unwrap();
            let clamped = fast_score_map(&image, test, &Clamp);
            let skipped = fast_score_map(&image, test, &Skip);
            for y in 0..image.height() {
                for x in 0..image.width() {
                    assert_eq!(
                        clamped.pixel_at(x, y).value(),
                        fast_score_at(&image, x, y, test, &Clamp).unwrap(),
                        "clamp, n = {arc_length}, at ({x}, {y})"
                    );
                    let expected = fast_score_at(&image, x, y, test, &Skip).unwrap_or(0.0);
                    assert_eq!(
                        skipped.pixel_at(x, y).value(),
                        expected,
                        "skip, n = {arc_length}, at ({x}, {y})"
                    );
                }
            }
        }
    }

    #[test]
    fn an_image_narrower_or_shorter_than_the_ring_has_no_hot_span() {
        // The interior/boundary split has to survive an image with no
        // interior at all. Under `Clamp` every pixel is still scored, so the
        // hot span's column bounds can land outside the row: a 2-wide image
        // has `region.right() == 2` while the ring needs column 3. Both
        // orientations, and both policies.
        for (w, h) in [(2usize, 12usize), (12, 2), (1, 1), (7, 7), (6, 40)] {
            let image: Image<MonoF32> =
                Image::generate(w, h, |x, y| MonoF32::new(((x * 3 + y) % 5) as f32 * 0.2));
            for arc_length in [9usize, 16] {
                let test = SegmentTest::new(0.05, arc_length).unwrap();
                for &clamped in &[true, false] {
                    let scores = if clamped {
                        fast_score_map(&image, test, &Clamp)
                    } else {
                        fast_score_map(&image, test, &Skip)
                    };
                    assert_eq!(scores.size(), image.size(), "{w}x{h}");
                    for y in 0..h {
                        for x in 0..w {
                            let expected = if clamped {
                                fast_score_at(&image, x, y, test, &Clamp).unwrap_or(0.0)
                            } else {
                                fast_score_at(&image, x, y, test, &Skip).unwrap_or(0.0)
                            };
                            assert_eq!(
                                scores.pixel_at(x, y).value(),
                                expected,
                                "{w}x{h}, clamp = {clamped}, n = {arc_length}, at ({x}, {y})"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn a_sub_threshold_corner_is_reported_as_zero_not_as_its_margin() {
        // The documented consequence of the early rejection: the map is
        // already thresholded, so a corner too faint for the test reads 0
        // rather than its true margin. Lowering the threshold recovers it.
        let faint: Image<MonoF32> = Image::generate(32, 32, |x, y| {
            MonoF32::new(if x >= 16 && y >= 16 { 0.1 } else { 0.0 })
        });
        let coarse = fast_score_map(&faint, SegmentTest::new(0.3, 9).unwrap(), &Skip);
        assert_eq!(coarse.pixel_at(16, 16).value(), 0.0);

        let fine = fast_score_map(&faint, SegmentTest::new(0.05, 9).unwrap(), &Skip);
        assert!((fine.pixel_at(16, 16).value() - 0.1).abs() < 1e-6);
    }

    #[test]
    fn raising_the_threshold_on_an_existing_map_is_exact() {
        // The half of the contract that survives the flooring: above the
        // test's own threshold, the map still answers every question exactly.
        let image = square(32, 10, 22);
        let scores = fast_score_map(&image, SegmentTest::new(0.05, 9).unwrap(), &Skip);
        for step in 1..20 {
            let t = 0.05 * step as f32;
            let from_map = corner_peaks(&scores, t, 3);
            let params = FastParams::new(SegmentTest::new(t, 9).unwrap(), NmsRadius::new(3).unwrap());
            let rebuilt = fast(&image, params, &Skip);
            assert_eq!(from_map, rebuilt, "at t = {t}");
        }
    }

    // ── fast_score_at / fast_score_map ──────────────────────────────────

    #[test]
    fn a_quadrant_corner_scores_its_full_contrast() {
        let image: Image<MonoF32> = Image::generate(24, 24, |x, y| {
            MonoF32::new(if x >= 8 && y >= 8 { 1.0 } else { 0.0 })
        });
        let test = SegmentTest::new(0.1, 9).unwrap();
        let score = fast_score_at(&image, 8, 8, test, &Skip).unwrap();
        assert!((score - 1.0).abs() < 1e-6, "{score}");
    }

    #[test]
    fn a_straight_edge_scores_nothing_anywhere() {
        // The deterministic negative case, and the reason arc_length >= 9.
        let image: Image<MonoF32> =
            Image::generate(24, 24, |x, _| MonoF32::new(if x < 12 { 0.0 } else { 1.0 }));
        let scores = fast_score_map(&image, SegmentTest::new(0.05, 9).unwrap(), &Skip);
        for y in 0..24 {
            for x in 0..24 {
                assert_eq!(scores.pixel_at(x, y).value(), 0.0, "at ({x}, {y})");
            }
        }
    }

    #[test]
    fn a_diagonal_edge_scores_nothing_either() {
        let image: Image<MonoF32> =
            Image::generate(32, 32, |x, y| MonoF32::new(if x < y { 0.0 } else { 1.0 }));
        let scores = fast_score_map(&image, SegmentTest::new(0.05, 9).unwrap(), &Skip);
        let peak = (3..29)
            .flat_map(|y| (3..29).map(move |x| (x, y)))
            .map(|(x, y)| scores.pixel_at(x, y).value())
            .fold(0.0f32, f32::max);
        assert_eq!(peak, 0.0);
    }

    #[test]
    fn a_flat_field_scores_nothing() {
        let image = Image::fill(16, 16, MonoF32::new(0.5));
        let scores = fast_score_map(&image, SegmentTest::new(0.01, 9).unwrap(), &Skip);
        assert_eq!(scores.pixel_at(8, 8).value(), 0.0);
    }

    #[test]
    fn the_score_map_keeps_the_input_size_under_every_policy() {
        let image = square(24, 8, 16);
        let test = SegmentTest::new(0.1, 9).unwrap();
        for size in [
            fast_score_map(&image, test, &Skip).size(),
            fast_score_map(&image, test, &Clamp).size(),
            fast_score_map(&image, test, &Mirror).size(),
            fast_score_map(&image, test, &Constant(MonoF32::new(0.0))).size(),
        ] {
            assert_eq!(size, image.size());
        }
    }

    #[test]
    fn skip_declines_the_ring_radius_border_and_clamp_does_not() {
        // A corner deliberately placed inside the 3-pixel margin.
        let image: Image<MonoF32> = Image::generate(24, 24, |x, y| {
            MonoF32::new(if x >= 2 && y >= 2 { 1.0 } else { 0.0 })
        });
        let test = SegmentTest::new(0.1, 9).unwrap();

        for d in 0..FAST_RING_RADIUS {
            assert_eq!(fast_score_at(&image, d, d, test, &Skip), None, "at {d}");
            assert!(fast_score_at(&image, d, d, test, &Clamp).is_some());
        }
        assert!(fast_score_at(&image, 3, 3, test, &Skip).is_some());

        // Under Skip the margin is written as zero; under Clamp the corner at
        // (2, 2) is found from replicated samples.
        assert_eq!(
            fast_score_map(&image, test, &Skip).pixel_at(2, 2).value(),
            0.0
        );
        assert!(fast_score_map(&image, test, &Clamp).pixel_at(2, 2).value() > 0.5);
    }

    #[test]
    fn fast_score_at_declines_positions_outside_the_image() {
        let image = square(24, 8, 16);
        let test = SegmentTest::new(0.1, 9).unwrap();
        assert_eq!(fast_score_at(&image, 24, 0, test, &Clamp), None);
        assert_eq!(fast_score_at(&image, 0, 24, test, &Clamp), None);
        assert_eq!(fast_score_at(&image, 999, 999, test, &Skip), None);
    }

    #[test]
    fn an_image_smaller_than_the_ring_yields_an_empty_skip_region() {
        // The footprint does not fit at all: Skip scores nothing, and the map
        // is still the input's size rather than a zero-sized surprise.
        let image = Image::fill(5, 5, MonoF32::new(0.5));
        let test = SegmentTest::new(0.1, 9).unwrap();
        let scores = fast_score_map(&image, test, &Skip);
        assert_eq!(scores.size(), image.size());
        assert_eq!(fast_score_at(&image, 2, 2, test, &Skip), None);
        // Clamp still works: every ring sample is invented, but consistently.
        assert_eq!(fast_score_at(&image, 2, 2, test, &Clamp), Some(0.0));
    }

    #[test]
    fn the_score_map_agrees_with_the_single_pixel_score() {
        // The map must be exactly `fast_score_at` applied everywhere, with a
        // declined position written as zero.
        let image = square(24, 8, 16);
        let test = SegmentTest::new(0.1, 9).unwrap();
        let scores = fast_score_map(&image, test, &Skip);
        for y in 0..24 {
            for x in 0..24 {
                let expected = fast_score_at(&image, x, y, test, &Skip).unwrap_or(0.0);
                assert_eq!(scores.pixel_at(x, y).value(), expected, "at ({x}, {y})");
            }
        }
    }

    // ── fast on synthetic geometry ──────────────────────────────────────

    #[test]
    fn a_square_has_four_corners() {
        let image = square(32, 10, 22);
        let params = FastParams::new(SegmentTest::new(0.1, 9).unwrap(), NmsRadius::new(3).unwrap());
        let corners = fast(&image, params, &Skip);

        assert_one_per_corner(&corners, square_corner_pixels(10, 22), 2.0);
        // Nothing is averaged, so the score is the full contrast exactly —
        // not a filtered approximation of it.
        assert!(corners.iter().all(|c| (c.response() - 1.0).abs() < 1e-6));
    }

    #[test]
    fn a_right_angle_saturates_into_a_tied_cluster() {
        // The honest limit of this detector's localization, and the reason
        // the test above does not assert exact pixels. Six pixels around each
        // corner of a step-edge square score *identically* — the full
        // contrast — because the score saturates once the arc clears the
        // threshold everywhere. The plateau rule then reports the cluster's
        // raster-first member, which is the geometric corner only where the
        // cluster opens down and to the right.
        let image = square(32, 10, 22);
        let scores = fast_score_map(&image, SegmentTest::new(0.1, 9).unwrap(), &Skip);

        let cluster: Vec<(usize, usize)> = (9..14)
            .flat_map(|y| (9..14).map(move |x| (x, y)))
            .filter(|&(x, y)| scores.pixel_at(x, y).value() > 0.0)
            .collect();
        assert_eq!(
            cluster,
            [(10, 10), (11, 10), (12, 10), (10, 11), (11, 11), (10, 12)]
        );
        assert!(
            cluster
                .iter()
                .all(|&(x, y)| scores.pixel_at(x, y).value() == 1.0)
        );

        // Top-left: the raster-first member *is* the corner. Top-right: it is
        // two pixels along the edge, and no threshold or radius moves it.
        let params = FastParams::new(SegmentTest::new(0.1, 9).unwrap(), NmsRadius::new(3).unwrap());
        let corners = fast(&image, params, &Skip);
        assert_eq!(
            positions(&corners),
            [(10.0, 10.0), (19.0, 10.0), (10.0, 19.0), (21.0, 19.0)]
        );
    }

    #[test]
    fn the_reported_pixels_do_not_move_with_the_threshold() {
        // The counterpart of the structure tensor's inward drift as σ grows:
        // there is no window here, so the threshold cannot move a detection.
        // Any threshold below the contrast gives the identical answer. (The
        // arc length carries no such guarantee — it changes the score map
        // itself; see `a_longer_arc_rejects_a_right_angle_entirely`.)
        let image = square(32, 10, 22);
        let reference = positions(&fast(
            &image,
            FastParams::new(SegmentTest::new(0.05, 9).unwrap(), NmsRadius::new(3).unwrap()),
            &Skip,
        ));
        for threshold in [0.2f32, 0.5, 0.9, 1.0] {
            let params = FastParams::new(SegmentTest::new(threshold, 9).unwrap(), NmsRadius::new(3).unwrap());
            assert_eq!(
                positions(&fast(&image, params, &Skip)),
                reference,
                "t = {threshold}"
            );
        }
    }

    #[test]
    fn raising_the_threshold_filters_detections_without_moving_the_survivors() {
        // The module doc's claim in its filtering half: two squares of
        // different contrast, so a raised threshold genuinely removes the
        // faint one's corners, and the bright one's corners must survive at
        // their exact pixels.
        let image: Image<MonoF32> = Image::generate(44, 24, |x, y| {
            let bright = (4..16).contains(&x) && (4..16).contains(&y);
            let faint = (26..38).contains(&x) && (4..16).contains(&y);
            MonoF32::new(if bright {
                1.0
            } else if faint {
                0.3
            } else {
                0.0
            })
        });
        let both = positions(&fast(
            &image,
            FastParams::new(SegmentTest::new(0.1, 9).unwrap(), NmsRadius::new(3).unwrap()),
            &Skip,
        ));
        assert_eq!(both.len(), 8, "{both:?}");

        let bright_only = positions(&fast(
            &image,
            FastParams::new(SegmentTest::new(0.5, 9).unwrap(), NmsRadius::new(3).unwrap()),
            &Skip,
        ));
        assert_eq!(bright_only.len(), 4, "{bright_only:?}");
        for p in &bright_only {
            assert!(both.contains(p), "{p:?} moved when the threshold rose");
        }
    }

    #[test]
    fn a_dot_is_a_corner_at_every_arc_length() {
        // The one fixture with no ambiguity at all: an isolated bright pixel
        // puts the entire ring on the far side of the step, so every arc
        // passes and exactly one pixel is reported — the dot itself.
        let image = dot(16, (8, 8));
        for n in 9..=16 {
            let params = FastParams::new(SegmentTest::new(0.1, n).unwrap(), NmsRadius::new(3).unwrap());
            let corners = fast(&image, params, &Skip);
            assert_eq!(positions(&corners), [(8.0, 8.0)], "n = {n}");
            assert_eq!(corners[0].response(), 1.0);
        }
    }

    #[test]
    fn a_longer_arc_rejects_a_right_angle_entirely() {
        // Where the arc length bites, on real geometry. A 90° corner leaves
        // 11 contiguous ring pixels on the outside — enough for FAST-9 and
        // FAST-11, one short for FAST-12. This is what "more selective"
        // means, and why 12 is not simply a better 9.
        let image = square(32, 10, 22);
        for n in [9, 10, 11] {
            let params = FastParams::new(SegmentTest::new(0.1, n).unwrap(), NmsRadius::new(3).unwrap());
            assert_eq!(fast(&image, params, &Skip).len(), 4, "n = {n}");
        }
        for n in [12, 16] {
            let params = FastParams::new(SegmentTest::new(0.1, n).unwrap(), NmsRadius::new(3).unwrap());
            assert!(fast(&image, params, &Skip).is_empty(), "n = {n}");
        }
    }

    #[test]
    fn an_l_junction_has_one_corner() {
        let image: Image<MonoF32> = Image::generate(24, 24, |x, y| {
            MonoF32::new(if x >= 12 && y >= 12 { 1.0 } else { 0.0 })
        });
        let params = FastParams::new(SegmentTest::new(0.1, 9).unwrap(), NmsRadius::new(4).unwrap());
        assert_eq!(positions(&fast(&image, params, &Skip)), [(12.0, 12.0)]);
    }

    #[test]
    fn a_dark_square_on_a_light_field_has_four_corners_too() {
        // The dark half of the test is not a second code path, and the
        // inverted image must give the identical answer.
        let bright = square(32, 10, 22);
        let dark: Image<MonoF32> = Image::generate(32, 32, |x, y| {
            MonoF32::new(1.0 - bright.pixel_at(x, y).value())
        });
        let params = FastParams::new(SegmentTest::new(0.1, 9).unwrap(), NmsRadius::new(3).unwrap());
        assert_eq!(
            positions(&fast(&dark, params, &Skip)),
            positions(&fast(&bright, params, &Skip))
        );
    }

    #[test]
    fn a_flat_field_and_a_straight_edge_have_no_corners() {
        let params = FastParams::new(SegmentTest::new(0.01, 9).unwrap(), NmsRadius::new(2).unwrap());
        assert!(fast(&Image::fill(16, 16, MonoF32::new(0.5)), params, &Skip).is_empty());

        let edge: Image<MonoF32> =
            Image::generate(24, 24, |x, _| MonoF32::new(if x < 12 { 0.0 } else { 1.0 }));
        assert!(fast(&edge, params, &Skip).is_empty());
    }

    #[test]
    fn a_higher_threshold_only_ever_removes_corners() {
        // Three squares at three contrasts: raising t peels them off in order.
        let image: Image<MonoF32> = Image::generate(60, 24, |x, y| {
            let inside = |x0: usize| (x0..x0 + 8).contains(&x) && (8..16).contains(&y);
            MonoF32::new(if inside(4) {
                1.0
            } else if inside(24) {
                0.5
            } else if inside(44) {
                0.2
            } else {
                0.0
            })
        });
        // Each square is 8 px wide, so at radius 3 its four tied corner
        // clusters chain into one detection — one corner per square, which is
        // all this test needs.
        let count = |t: f32| {
            let params = FastParams::new(SegmentTest::new(t, 9).unwrap(), NmsRadius::new(3).unwrap());
            fast(&image, params, &Skip).len()
        };
        assert_eq!(count(0.1), 3);
        assert_eq!(count(0.3), 2);
        assert_eq!(count(0.6), 1);
        assert_eq!(count(1.1), 0);
    }

    #[test]
    fn the_score_is_invariant_under_a_quarter_turn() {
        // The ring is symmetric under 90° rotation, so the score map must be
        // too — exactly, not approximately: no filtering is involved.
        let image = square(24, 7, 17);
        let rotated: Image<MonoF32> = rotate_90(&image);
        let test = SegmentTest::new(0.1, 9).unwrap();

        let scores = fast_score_map(&image, test, &Skip);
        let rotated_scores = fast_score_map(&rotated, test, &Skip);
        for y in 0..24 {
            for x in 0..24 {
                // rotate_90 is clockwise: (x, y) ↦ (h − 1 − y, x).
                assert_eq!(
                    scores.pixel_at(x, y).value(),
                    rotated_scores.pixel_at(23 - y, x).value(),
                    "at ({x}, {y})"
                );
            }
        }
    }

    #[test]
    fn the_suppression_radius_merges_clusters_it_can_reach() {
        // On an 8-px square the tied clusters of adjacent corners are three
        // pixels apart, so radius 3 chains all four into a single group while
        // radius 2 keeps them separate. The radius is a minimum separation,
        // and here it is doing exactly that — visibly.
        let image = square(24, 8, 16);
        let at = |r: usize| {
            let params =
                FastParams::new(SegmentTest::new(0.1, 9).unwrap(), NmsRadius::new(r).unwrap());
            fast(&image, params, &Skip)
        };
        assert_eq!(at(2).len(), 4);
        assert_eq!(positions(&at(3)), [(8.0, 8.0)]);

        // Widening the gap by widening the square separates them again.
        let wider = square(32, 10, 22);
        let params = FastParams::new(SegmentTest::new(0.1, 9).unwrap(), NmsRadius::new(3).unwrap());
        assert_eq!(fast(&wider, params, &Skip).len(), 4);
    }

    #[test]
    fn fast_is_the_documented_composition() {
        // The orchestrator must be exactly score map + peaks at the test's own
        // threshold, so rebuilding it by hand is not a different detector.
        let image = square(24, 8, 16);
        let params = FastParams::new(SegmentTest::new(0.1, 9).unwrap(), NmsRadius::new(3).unwrap());

        let staged = {
            let scores = fast_score_map(&image, params.test(), &Skip);
            corner_peaks(&scores, params.test().threshold(), params.nms_radius().get())
        };
        assert_eq!(fast(&image, params, &Skip), staged);
        assert!(!staged.is_empty());
    }

    #[test]
    fn detection_and_the_score_agree_on_the_threshold() {
        // The invariant that makes one number serve both stages, checked
        // through the public API on a real image.
        let image = square(24, 8, 16);
        let scores = fast_score_map(&image, SegmentTest::new(0.05, 9).unwrap(), &Skip);
        for step in 1..20 {
            let t = 0.05 * step as f32;
            let params = FastParams::new(SegmentTest::new(t, 9).unwrap(), NmsRadius::new(3).unwrap());
            let detected = fast(&image, params, &Skip);
            let above = (0..24)
                .flat_map(|y| (0..24).map(move |x| (x, y)))
                .filter(|&(x, y)| scores.pixel_at(x, y).value() >= t)
                .count();
            // Every detection is above the threshold in the *fixed* map, so
            // the map's threshold semantics and the test's coincide.
            assert_eq!(detected.is_empty(), above == 0, "at t = {t}");
            assert!(detected.len() <= above, "at t = {t}");
        }
    }

    // ── Pixel-type genericity ───────────────────────────────────────────

    #[test]
    fn accepts_integer_input_with_a_threshold_in_grey_levels() {
        // The practical payoff of an intensity-unit threshold: 20 levels out
        // of 255 needs no calibration against a response map.
        let image: Image<Mono8> = Image::generate(32, 32, |x, y| {
            let inside = (10..22).contains(&x) && (10..22).contains(&y);
            Mono8::new(if inside { 255 } else { 0 })
        });
        let params = FastParams::new(SegmentTest::new(20.0, 9).unwrap(), NmsRadius::new(3).unwrap());
        let corners = fast(&image, params, &Skip);
        assert_one_per_corner(&corners, square_corner_pixels(10, 22), 2.0);
        // `to_accumulator` widens without rescaling, so the score is in grey
        // levels too — 255, not 1.0.
        assert!(corners.iter().all(|c| c.response() == 255.0));
    }

    #[test]
    fn accepts_sixteen_bit_input() {
        let image: Image<Mono16> = Image::generate(32, 32, |x, y| {
            let inside = (10..22).contains(&x) && (10..22).contains(&y);
            Mono16::new(if inside { 65535 } else { 0 })
        });
        let corners = fast(
            &image,
            FastParams::new(SegmentTest::new(5000.0, 9).unwrap(), NmsRadius::new(3).unwrap()),
            &Skip,
        );
        assert_one_per_corner(&corners, square_corner_pixels(10, 22), 2.0);
        // 65535 is exactly representable in f32, so the score is exact.
        assert!(corners.iter().all(|c| c.response() == 65535.0));
    }

    #[test]
    fn accepts_f64_float_input() {
        let image: Image<MonoF64> = Image::generate(32, 32, |x, y| {
            let inside = (10..22).contains(&x) && (10..22).contains(&y);
            MonoF64::new(if inside { 1.0 } else { 0.0 })
        });
        let params = FastParams::new(SegmentTest::new(0.1, 9).unwrap(), NmsRadius::new(3).unwrap());
        assert_one_per_corner(
            &fast(&image, params, &Skip),
            square_corner_pixels(10, 22),
            2.0,
        );
    }

    #[test]
    fn the_detector_is_invariant_to_a_constant_brightness_shift() {
        // The test is purely relative to the centre pixel, so adding a
        // constant cannot change a single score.
        let base = square(32, 10, 22);
        let lifted: Image<MonoF32> = Image::generate(32, 32, |x, y| {
            MonoF32::new(base.pixel_at(x, y).value() * 0.5 + 0.25)
        });
        let test = SegmentTest::new(0.1, 9).unwrap();
        let params = FastParams::new(test, NmsRadius::new(3).unwrap());
        assert_eq!(
            positions(&fast(&base, params, &Skip)),
            positions(&fast(&lifted, params, &Skip))
        );
        // Contrast halved ⇒ every score halved, exactly.
        for corner in fast(&lifted, params, &Skip) {
            assert!((corner.response() - 0.5).abs() < 1e-6);
        }
    }

    // ── Pyramid levels ──────────────────────────────────────────────────

    #[test]
    fn detection_on_the_base_level_is_the_identity_lift() {
        let image = square(24, 8, 16);
        let level = ScaledImage::new(
            image.clone(),
            pixel_distance!(1.0),
            CoordinateF64::new(0.0, 0.0),
            sigma!(0.5),
        );
        let params = FastParams::new(SegmentTest::new(0.1, 9).unwrap(), NmsRadius::new(3).unwrap());
        assert_eq!(
            fast_in_level(&level, params, &Skip),
            fast(&image, params, &Skip)
        );
    }

    #[test]
    fn detection_on_a_coarse_level_reports_base_coordinates() {
        let base = square(48, 16, 32);
        let level = ScaledImage::new(
            pyr_down(&base),
            pixel_distance!(2.0),
            CoordinateF64::new(0.0, 0.0),
            sigma!(1.0),
        );
        let params = FastParams::new(SegmentTest::new(0.1, 9).unwrap(), NmsRadius::new(2).unwrap());
        let corners = fast_in_level(&level, params, &Skip);
        assert_eq!(corners.len(), 4, "{corners:?}");

        for corner in &corners {
            let p = corner.position();
            // Every coordinate is even: proof the lift multiplied by the
            // level's pixel distance rather than reporting local coordinates.
            assert!(p.x % 2.0 == 0.0 && p.y % 2.0 == 0.0, "{corner:?}");
            assert!((14.0..34.0).contains(&p.x), "{corner:?}");
        }
    }

    #[test]
    fn a_level_with_an_origin_offset_lifts_through_it() {
        let base = square(48, 16, 32);
        let coarse = pyr_down(&base);
        let params = FastParams::new(SegmentTest::new(0.1, 9).unwrap(), NmsRadius::new(2).unwrap());

        let unshifted = ScaledImage::new(
            coarse.clone(),
            pixel_distance!(2.0),
            CoordinateF64::new(0.0, 0.0),
            sigma!(1.0),
        );
        let shifted = ScaledImage::new(
            coarse,
            pixel_distance!(2.0),
            CoordinateF64::new(0.5, 0.5),
            sigma!(1.0),
        );

        let a = fast_in_level(&unshifted, params, &Skip);
        let b = fast_in_level(&shifted, params, &Skip);
        assert!(!a.is_empty());
        assert_eq!(a.len(), b.len());
        for (unshifted, shifted) in a.iter().zip(&b) {
            assert_eq!(shifted.position().x - unshifted.position().x, 0.5);
            assert_eq!(shifted.position().y - unshifted.position().y, 0.5);
        }
    }

    #[test]
    fn a_pyramid_can_be_swept_level_by_level() {
        let base = square(48, 12, 36);
        let levels = vec![
            ScaledImage::new(
                base.clone(),
                pixel_distance!(1.0),
                CoordinateF64::new(0.0, 0.0),
                sigma!(0.5),
            ),
            ScaledImage::new(
                pyr_down(&base),
                pixel_distance!(2.0),
                CoordinateF64::new(0.0, 0.0),
                sigma!(1.0),
            ),
        ];
        let pyramid = Pyramid::try_from_levels(levels).unwrap();
        let params = FastParams::new(SegmentTest::new(0.1, 9).unwrap(), NmsRadius::new(2).unwrap());

        let mut corners: Vec<Corner> = pyramid
            .iter()
            .flat_map(|level| fast_in_level(level, params, &Skip))
            .collect();
        // Both levels see the same square, so each physical corner is found
        // twice — deduplication across levels is the caller's policy.
        assert_eq!(corners.len(), 8, "{corners:?}");
        retain_top_n(&mut corners, 4);
        assert!(corners.iter().all(|c| c.response() > 0.5));
    }

    // ── The two families agree on where the corners are ─────────────────

    #[test]
    fn fast_and_the_structure_tensor_find_the_same_square_corners() {
        // The point of shipping a second detector: the keypoint model and the
        // peak stage are shared, and only the measure differs.
        use super::super::{CornerParams, ShiTomasi, detect_corners};

        let image = square(32, 10, 22);
        let truth = square_corner_pixels(10, 22);
        let tensor = detect_corners(
            &image,
            ShiTomasi,
            CornerParams::new(sigma!(1.0), 0.5, NmsRadius::new(3).unwrap()).unwrap(),
        );
        let params = FastParams::new(SegmentTest::new(0.1, 9).unwrap(), NmsRadius::new(3).unwrap());
        let segment = fast(&image, params, &Skip);

        assert_one_per_corner(&tensor, truth, 2.0);
        assert_one_per_corner(&segment, truth, 2.0);
        // Both agree exactly on the top-left corner, where neither measure's
        // bias has anywhere to push: the tensor peak has not yet drifted
        // inward at σ = 1.0, and the tied cluster opens away from it.
        assert_eq!(tensor[0].position(), segment[0].position());
        // The responses are not comparable, and nothing pretends they are:
        // λ_min is a squared gradient, the FAST score is a raw contrast.
        assert!(tensor[0].response() != segment[0].response());
    }
}

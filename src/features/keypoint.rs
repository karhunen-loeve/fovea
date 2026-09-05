//! Keypoint capability traits, the concrete keypoint types, and the
//! deterministic response ordering.

use core::cmp::Ordering;

use crate::image::{Decimated, ScaleLevel};
use crate::{CoordinateF64, Orientation, Sigma};

// ─── Capability traits ──────────────────────────────────────────────────────

/// A keypoint with a sub-pixel location.
///
/// The position is **always** in the base-image frame — the frame of the
/// image the pyramid or scale space was built from, never the frame of the
/// level the keypoint was detected on. Detectors working on a coarse level
/// establish that by construction, through
/// [`Corner::from_level`] / [`ScaleKeypoint::from_level`], which lift the
/// local position with [`Decimated::to_base`](crate::image::Decimated::to_base).
/// One frame for every reported position is what makes keypoints from
/// different levels comparable.
///
/// # Example
///
/// ```
/// use fovea::CoordinateF64;
/// use fovea::features::{Corner, HasPosition};
///
/// let corner = Corner::new(CoordinateF64::new(12.25, 4.5), 0.8);
/// assert_eq!(corner.position(), CoordinateF64::new(12.25, 4.5));
/// ```
pub trait HasPosition {
    /// Returns the sub-pixel location, in base-image coordinates.
    fn position(&self) -> CoordinateF64;
}

/// A keypoint carrying a detector strength.
///
/// The value defines **ranking only**. Its sign and magnitude are
/// detector-specific — Harris cornerness and a difference-of-Gaussians
/// contrast are both "response" and are not comparable with each other — so
/// the only meaning guaranteed here is that, within one detector's output,
/// a larger value is a stronger detection. That is exactly what
/// [`retain_top_n`] needs and no more than any detector can honestly
/// promise.
///
/// # Example
///
/// ```
/// use fovea::CoordinateF64;
/// use fovea::features::{Corner, HasResponse};
///
/// let corner = Corner::new(CoordinateF64::new(3.0, 7.0), 1.5);
/// assert_eq!(corner.response(), 1.5);
/// ```
pub trait HasResponse {
    /// Returns the detector response. Ordering only — the sign and
    /// magnitude are detector-defined.
    fn response(&self) -> f32;
}

/// A keypoint localized in scale as well as position.
///
/// The scale is a characteristic Gaussian σ expressed in **base-image
/// pixels**, the same convention and the same type as
/// [`ScaleLevel::sigma`](crate::image::ScaleLevel::sigma). Carrying it as
/// [`Sigma`] rather than a bare `f32` means the value arrives already
/// validated (finite, strictly positive) and can flow onward — into a
/// [`gaussian_blur`](crate::transform::gaussian_blur), into a descriptor's
/// patch size — without being re-checked at every boundary.
///
/// A detector that does not select scale must not implement this trait: a
/// corner found at one fixed resolution has no characteristic σ, and
/// inventing one (`1.0`, say) is the "field that might be garbage" this
/// trait split exists to prevent.
///
/// # Example
///
/// ```
/// use fovea::CoordinateF64;
/// use fovea::features::{HasScale, ScaleKeypoint};
/// use fovea::sigma;
///
/// let kp = ScaleKeypoint::new(CoordinateF64::new(8.0, 2.5), 0.4, sigma!(1.6));
/// assert_eq!(kp.scale().get(), 1.6);
/// ```
pub trait HasScale {
    /// Returns the characteristic scale as a Gaussian σ in base-image
    /// pixels.
    fn scale(&self) -> Sigma;
}

/// A keypoint carrying a dominant gradient orientation.
///
/// The angle is an [`Orientation`] — a **directed** angle modulo 2π, so a
/// feature pointing one way is distinct from one pointing the opposite way.
/// The type, rather than a bare `f32`, is what states the modulus: it makes
/// `==` mean "the same direction" and puts the seam-crossing subtraction in
/// [`Orientation::signed_difference`] instead of at every call site that
/// compares two features' orientations. The undirected sibling —
/// [`AxialOrientation`](crate::AxialOrientation), for quantities like a
/// region's major axis, where θ and θ + π are the same thing — is
/// deliberately a different type.
///
/// **No type in this crate implements this trait yet.** It is defined now
/// because it is the capability descriptors bind against, and because
/// adding a capability later must not mean editing the existing keypoint
/// types. The oriented keypoint type that implements it arrives with the
/// orientation-assignment step that can compute an orientation; until then
/// there is nothing to implement it *honestly*, and a placeholder
/// implementor would be exactly the meaningless field this module avoids.
///
/// # Example
///
/// Implementing it is a two-line affair, which is the point — a downstream
/// detector with its own keypoint type joins the same vocabulary:
///
/// ```
/// use fovea::{CoordinateF64, Orientation};
/// use fovea::features::{HasOrientation, HasPosition};
///
/// struct MyOrientedCorner {
///     at: CoordinateF64,
///     angle: Orientation,
/// }
///
/// impl HasPosition for MyOrientedCorner {
///     fn position(&self) -> CoordinateF64 {
///         self.at
///     }
/// }
///
/// impl HasOrientation for MyOrientedCorner {
///     fn orientation(&self) -> Orientation {
///         self.angle
///     }
/// }
///
/// // Built straight from a gradient vector — no wrapping, nothing to unwrap.
/// let kp = MyOrientedCorner {
///     at: CoordinateF64::new(1.0, 2.0),
///     angle: Orientation::from_atan2(1.0, 1.0),
/// };
/// assert_eq!(kp.orientation(), Orientation::from_atan2(1.0, 1.0));
/// ```
pub trait HasOrientation {
    /// Returns the dominant gradient orientation, a directed angle.
    fn orientation(&self) -> Orientation;
}

// ─── Corner ─────────────────────────────────────────────────────────────────

/// A keypoint with a sub-pixel position and a response, and nothing else.
///
/// The output type of single-resolution corner detectors (Harris,
/// Shi-Tomasi, FAST): they localize a point and score it, but select no
/// scale and compute no orientation. Implements
/// [`HasPosition`] and [`HasResponse`] — and, deliberately, neither
/// [`HasScale`] nor [`HasOrientation`], so handing a `Corner` to an
/// operation that needs a scale is a compile error rather than a silently
/// wrong result.
///
/// # Example
///
/// ```
/// use fovea::CoordinateF64;
/// use fovea::features::{Corner, HasPosition, HasResponse};
///
/// let corner = Corner::new(CoordinateF64::new(10.5, 20.25), 0.93);
/// assert_eq!(corner.position(), CoordinateF64::new(10.5, 20.25));
/// assert_eq!(corner.response(), 0.93);
/// ```
///
/// A consumer that needs a scale rejects a `Corner` at compile time, not at
/// runtime — the reason the capabilities are separate traits:
///
/// ```compile_fail
/// use fovea::CoordinateF64;
/// use fovea::features::{Corner, HasPosition, HasScale};
///
/// fn patch_radius(kp: &(impl HasPosition + HasScale)) -> f32 {
///     3.0 * kp.scale().get()
/// }
///
/// let corner = Corner::new(CoordinateF64::new(10.5, 20.25), 0.93);
/// // ERROR: `Corner: HasScale` is not satisfied.
/// let _r = patch_radius(&corner);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Corner {
    /// Sub-pixel position in the base-image frame.
    pub at: CoordinateF64,
    /// Detector response; ordering only.
    pub response: f32,
}

impl Corner {
    /// Creates a corner at a position already expressed in the base-image
    /// frame.
    ///
    /// For a detection made on a pyramid level, prefer
    /// [`from_level`](Self::from_level), which performs the lift.
    pub fn new(at: CoordinateF64, response: f32) -> Self {
        Self { at, response }
    }

    /// Creates a corner from a detection made in a level's **local**
    /// coordinates, lifting the position into the base-image frame.
    ///
    /// This is the named level→base lift for corners: it delegates to
    /// [`Decimated::to_base`](crate::image::Decimated::to_base), so the
    /// affine map (sample distance *and* grid-origin offset) is applied in
    /// one place instead of being re-derived as `x · 2^level` per detector
    /// — which drops the origin term and drifts by half a pixel per
    /// octave.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::CoordinateF64;
    /// use fovea::features::{Corner, HasPosition};
    /// use fovea::image::{Image, OriginOffset, ScaledImage};
    /// use fovea::pixel::MonoF32;
    /// use fovea::{pixel_distance, sigma};
    ///
    /// // Level 1 of a 2× pyramid built by even-sample decimation.
    /// let level = ScaledImage::new(
    ///     Image::<MonoF32>::zero(8, 8),
    ///     pixel_distance!(2.0),
    ///     OriginOffset::ZERO,
    ///     sigma!(1.0),
    /// );
    ///
    /// // Detected at (3.5, 2.0) on the level → (7.0, 4.0) in the base image.
    /// let corner = Corner::from_level(&level, CoordinateF64::new(3.5, 2.0), 0.7);
    /// assert_eq!(corner.position(), CoordinateF64::new(7.0, 4.0));
    /// ```
    pub fn from_level(level: &impl Decimated, local: CoordinateF64, response: f32) -> Self {
        Self {
            at: level.to_base(local),
            response,
        }
    }
}

impl HasPosition for Corner {
    #[inline]
    fn position(&self) -> CoordinateF64 {
        self.at
    }
}

impl HasResponse for Corner {
    #[inline]
    fn response(&self) -> f32 {
        self.response
    }
}

// ─── ScaleKeypoint ──────────────────────────────────────────────────────────

/// A keypoint localized in position, response, **and** scale.
///
/// The output type of scale-selecting detectors: a corner or blob found by
/// searching a pyramid or scale space, whose characteristic σ is a
/// detection result rather than a fixed parameter. Implements
/// [`HasPosition`], [`HasResponse`] and [`HasScale`], but not
/// [`HasOrientation`].
///
/// # Example
///
/// ```
/// use fovea::CoordinateF64;
/// use fovea::features::{HasScale, ScaleKeypoint};
/// use fovea::sigma;
///
/// let kp = ScaleKeypoint::new(CoordinateF64::new(4.0, 9.5), 0.2, sigma!(2.4));
/// assert_eq!(kp.scale(), sigma!(2.4));
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScaleKeypoint {
    /// Sub-pixel position in the base-image frame.
    pub at: CoordinateF64,
    /// Detector response; ordering only.
    pub response: f32,
    /// Characteristic Gaussian σ, in base-image pixels.
    pub scale: Sigma,
}

impl ScaleKeypoint {
    /// Creates a scale keypoint from values already expressed in the
    /// base-image frame.
    ///
    /// For a detection made on a pyramid level, prefer
    /// [`from_level`](Self::from_level).
    pub fn new(at: CoordinateF64, response: f32, scale: Sigma) -> Self {
        Self {
            at,
            response,
            scale,
        }
    }

    /// Creates a scale keypoint from a detection made in a level's
    /// **local** coordinates, taking its scale from the level.
    ///
    /// Both base-frame quantities come from the level's own metadata: the
    /// position through
    /// [`Decimated::to_base`](crate::image::Decimated::to_base), the scale
    /// through [`ScaleLevel::sigma`](crate::image::ScaleLevel::sigma) —
    /// which is already an absolute σ in base-image pixels, so no rescaling
    /// happens here either. The function is total: both invariants were
    /// established when the level was built.
    ///
    /// This assigns the level's σ, which is the right answer for a
    /// detection at a level's own scale. A detector that interpolates a
    /// peak *across* scales computes its own σ and uses
    /// [`new`](Self::new) with a lifted position instead.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::CoordinateF64;
    /// use fovea::features::{HasPosition, HasScale, ScaleKeypoint};
    /// use fovea::image::{Image, OriginOffset, ScaledImage};
    /// use fovea::pixel::MonoF32;
    /// use fovea::{pixel_distance, sigma};
    ///
    /// let level = ScaledImage::new(
    ///     Image::<MonoF32>::zero(8, 8),
    ///     pixel_distance!(2.0),
    ///     OriginOffset::ZERO,
    ///     sigma!(1.6),
    /// );
    ///
    /// let kp = ScaleKeypoint::from_level(&level, CoordinateF64::new(1.0, 2.5), 0.5);
    /// assert_eq!(kp.position(), CoordinateF64::new(2.0, 5.0));
    /// assert_eq!(kp.scale(), sigma!(1.6));
    /// ```
    pub fn from_level<L>(level: &L, local: CoordinateF64, response: f32) -> Self
    where
        L: Decimated + ScaleLevel,
    {
        Self {
            at: level.to_base(local),
            response,
            scale: level.sigma(),
        }
    }
}

impl HasPosition for ScaleKeypoint {
    #[inline]
    fn position(&self) -> CoordinateF64 {
        self.at
    }
}

impl HasResponse for ScaleKeypoint {
    #[inline]
    fn response(&self) -> f32 {
        self.response
    }
}

impl HasScale for ScaleKeypoint {
    #[inline]
    fn scale(&self) -> Sigma {
        self.scale
    }
}

// ─── Deterministic response ordering ────────────────────────────────────────

/// The canonical keypoint ordering: **strongest response first**, ties
/// broken by position `(y, x)` ascending.
///
/// Response alone is not a total order on keypoint sets — a detector run on
/// a synthetic image produces exact ties by the dozen (a checkerboard's
/// corners all score identically), and which of them survives a top-N cut
/// would otherwise depend on the order the detector happened to visit
/// pixels in. The `(y, x)` tie-break makes "the top 50 corners" a
/// reproducible set, which is what makes it testable.
///
/// Comparisons use [`f32::total_cmp`] / [`f64::total_cmp`], so a NaN
/// response or position from a misbehaving detector still yields a
/// consistent total order rather than a sort that silently loses elements.
///
/// # Example
///
/// ```
/// use core::cmp::Ordering;
/// use fovea::CoordinateF64;
/// use fovea::features::{by_response_then_position, Corner};
///
/// let strong = Corner::new(CoordinateF64::new(0.0, 9.0), 0.9);
/// let weak = Corner::new(CoordinateF64::new(0.0, 0.0), 0.1);
/// assert_eq!(by_response_then_position(&strong, &weak), Ordering::Less);
///
/// // Equal response: the smaller y comes first.
/// let upper = Corner::new(CoordinateF64::new(5.0, 1.0), 0.5);
/// let lower = Corner::new(CoordinateF64::new(5.0, 2.0), 0.5);
/// assert_eq!(by_response_then_position(&upper, &lower), Ordering::Less);
/// ```
pub fn by_response_then_position<K>(a: &K, b: &K) -> Ordering
where
    K: HasPosition + HasResponse,
{
    // Descending in response: `b` first, so the strongest sorts to index 0.
    b.response().total_cmp(&a.response()).then_with(|| {
        let (pa, pb) = (a.position(), b.position());
        pa.y.total_cmp(&pb.y).then_with(|| pa.x.total_cmp(&pb.x))
    })
}

/// Sorts keypoints into the canonical order of
/// [`by_response_then_position`]: strongest first, ties by `(y, x)`.
///
/// The sort is **stable**, so keypoints identical in response *and*
/// position — two scales of the same corner, for instance — keep the order
/// the detector produced them in.
///
/// # Example
///
/// ```
/// use fovea::CoordinateF64;
/// use fovea::features::{sort_by_response, Corner, HasResponse};
///
/// let mut corners = vec![
///     Corner::new(CoordinateF64::new(0.0, 0.0), 0.2),
///     Corner::new(CoordinateF64::new(1.0, 1.0), 0.9),
///     Corner::new(CoordinateF64::new(2.0, 2.0), 0.5),
/// ];
/// sort_by_response(&mut corners);
///
/// let responses: Vec<f32> = corners.iter().map(HasResponse::response).collect();
/// assert_eq!(responses, [0.9, 0.5, 0.2]);
/// ```
pub fn sort_by_response<K>(keypoints: &mut [K])
where
    K: HasPosition + HasResponse,
{
    keypoints.sort_by(by_response_then_position);
}

/// Keeps the `n` strongest keypoints and discards the rest.
///
/// Sorts with [`sort_by_response`] and truncates, so the retained
/// keypoints are left in canonical order and the selection is reproducible
/// across runs even when responses tie. `n` larger than the input length
/// keeps everything; `n == 0` empties the vector.
///
/// This is the "keep the top-N" step every detector needs and none should
/// re-implement — the interesting part is not the truncation but the
/// determinism of what survives it.
///
/// # Example
///
/// ```
/// use fovea::CoordinateF64;
/// use fovea::features::{retain_top_n, Corner, HasPosition};
///
/// let mut corners = vec![
///     Corner::new(CoordinateF64::new(9.0, 9.0), 0.1),
///     // Two identical responses: the smaller y wins the cut.
///     Corner::new(CoordinateF64::new(0.0, 5.0), 0.6),
///     Corner::new(CoordinateF64::new(0.0, 3.0), 0.6),
/// ];
/// retain_top_n(&mut corners, 2);
///
/// let ys: Vec<f64> = corners.iter().map(|c| c.position().y).collect();
/// assert_eq!(ys, [3.0, 5.0]);
/// ```
pub fn retain_top_n<K>(keypoints: &mut Vec<K>, n: usize)
where
    K: HasPosition + HasResponse,
{
    sort_by_response(keypoints);
    keypoints.truncate(n);
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PixelDistance;
    use crate::image::{Image, OriginOffset, ScaledImage};
    use crate::pixel::MonoF32;
    use crate::sigma;

    fn level(distance: f64, offset: (f64, f64), sigma: f32) -> ScaledImage<MonoF32> {
        ScaledImage::new(
            Image::<MonoF32>::zero(4, 4),
            PixelDistance::new(distance).unwrap(),
            OriginOffset::try_new(offset.0, offset.1).unwrap(),
            Sigma::new(sigma).unwrap(),
        )
    }

    // ── Corner ──────────────────────────────────────────────────────────

    #[test]
    fn corner_new_reports_its_fields() {
        let corner = Corner::new(CoordinateF64::new(1.5, 2.5), 0.75);
        assert_eq!(corner.position(), CoordinateF64::new(1.5, 2.5));
        assert_eq!(corner.response(), 0.75);
        assert_eq!(corner.at, CoordinateF64::new(1.5, 2.5));
        assert_eq!(corner.response, 0.75);
    }

    #[test]
    fn corner_from_level_lifts_position() {
        let corner = Corner::from_level(
            &level(2.0, (0.0, 0.0), 1.0),
            CoordinateF64::new(3.5, 2.0),
            0.7,
        );
        assert_eq!(corner.position(), CoordinateF64::new(7.0, 4.0));
        assert_eq!(corner.response(), 0.7);
    }

    #[test]
    fn corner_from_level_applies_the_origin_offset() {
        // The half-pixel term is the whole reason the lift is named: a
        // ratio-only lift would report (4.0, 2.0) here.
        let corner = Corner::from_level(
            &level(2.0, (0.5, 0.5), 1.0),
            CoordinateF64::new(2.0, 1.0),
            0.1,
        );
        assert_eq!(corner.position(), CoordinateF64::new(4.5, 2.5));
    }

    #[test]
    fn corner_from_level_on_the_base_level_is_the_identity() {
        let corner = Corner::from_level(
            &level(1.0, (0.0, 0.0), 0.5),
            CoordinateF64::new(6.25, 7.75),
            0.3,
        );
        assert_eq!(corner.position(), CoordinateF64::new(6.25, 7.75));
    }

    #[test]
    fn corner_from_level_handles_an_upsampled_level() {
        // Octave −1: samples half a base pixel apart, so local coordinates
        // shrink rather than grow.
        let corner = Corner::from_level(
            &level(0.5, (0.0, 0.0), 0.8),
            CoordinateF64::new(6.0, 10.0),
            0.2,
        );
        assert_eq!(corner.position(), CoordinateF64::new(3.0, 5.0));
    }

    #[test]
    fn corner_is_copy_and_comparable() {
        let a = Corner::new(CoordinateF64::new(1.0, 1.0), 0.5);
        let b = a; // Copy, not move
        assert_eq!(a, b);
        assert_ne!(a, Corner::new(CoordinateF64::new(1.0, 1.0), 0.6));
        assert!(format!("{a:?}").contains("Corner"));
    }

    // ── ScaleKeypoint ───────────────────────────────────────────────────

    #[test]
    fn scale_keypoint_new_reports_its_fields() {
        let kp = ScaleKeypoint::new(CoordinateF64::new(2.0, 3.0), 0.4, sigma!(1.6));
        assert_eq!(kp.position(), CoordinateF64::new(2.0, 3.0));
        assert_eq!(kp.response(), 0.4);
        assert_eq!(kp.scale(), sigma!(1.6));
    }

    #[test]
    fn scale_keypoint_from_level_takes_position_and_sigma_from_the_level() {
        let kp = ScaleKeypoint::from_level(
            &level(2.0, (0.0, 0.0), 1.6),
            CoordinateF64::new(1.0, 2.5),
            0.5,
        );
        assert_eq!(kp.position(), CoordinateF64::new(2.0, 5.0));
        assert_eq!(kp.scale(), sigma!(1.6));
        assert_eq!(kp.response(), 0.5);
    }

    #[test]
    fn scale_keypoint_from_level_sigma_is_absolute_not_rescaled() {
        // ScaleLevel::sigma is already in base-image pixels, so the level's
        // pixel distance must not multiply into it.
        let coarse = ScaleKeypoint::from_level(
            &level(4.0, (0.0, 0.0), 3.2),
            CoordinateF64::new(0.0, 0.0),
            1.0,
        );
        assert_eq!(coarse.scale(), sigma!(3.2));
    }

    #[test]
    fn scale_keypoint_is_copy_and_comparable() {
        let a = ScaleKeypoint::new(CoordinateF64::new(1.0, 1.0), 0.5, sigma!(1.0));
        let b = a;
        assert_eq!(a, b);
        assert_ne!(
            a,
            ScaleKeypoint::new(CoordinateF64::new(1.0, 1.0), 0.5, sigma!(2.0))
        );
    }

    // ── Ordering ────────────────────────────────────────────────────────

    #[test]
    fn ordering_puts_the_strongest_response_first() {
        let strong = Corner::new(CoordinateF64::new(0.0, 9.0), 0.9);
        let weak = Corner::new(CoordinateF64::new(0.0, 0.0), 0.1);
        assert_eq!(by_response_then_position(&strong, &weak), Ordering::Less);
        assert_eq!(by_response_then_position(&weak, &strong), Ordering::Greater);
    }

    #[test]
    fn ordering_breaks_response_ties_on_y_then_x() {
        let r = 0.5;
        let upper = Corner::new(CoordinateF64::new(9.0, 1.0), r);
        let lower = Corner::new(CoordinateF64::new(0.0, 2.0), r);
        // y dominates x.
        assert_eq!(by_response_then_position(&upper, &lower), Ordering::Less);

        let left = Corner::new(CoordinateF64::new(1.0, 5.0), r);
        let right = Corner::new(CoordinateF64::new(2.0, 5.0), r);
        assert_eq!(by_response_then_position(&left, &right), Ordering::Less);
    }

    #[test]
    fn ordering_is_equal_for_identical_keypoints() {
        let a = Corner::new(CoordinateF64::new(3.0, 4.0), 0.5);
        assert_eq!(by_response_then_position(&a, &a), Ordering::Equal);
    }

    #[test]
    fn ordering_of_negative_responses_is_still_descending() {
        // Harris cornerness is signed; a less-negative value is stronger.
        let better = Corner::new(CoordinateF64::new(0.0, 0.0), -0.1);
        let worse = Corner::new(CoordinateF64::new(0.0, 0.0), -0.9);
        assert_eq!(by_response_then_position(&better, &worse), Ordering::Less);
    }

    #[test]
    fn sort_by_response_orders_descending() {
        let mut corners = vec![
            Corner::new(CoordinateF64::new(0.0, 0.0), 0.2),
            Corner::new(CoordinateF64::new(1.0, 1.0), 0.9),
            Corner::new(CoordinateF64::new(2.0, 2.0), 0.5),
        ];
        sort_by_response(&mut corners);
        let responses: Vec<f32> = corners.iter().map(HasResponse::response).collect();
        assert_eq!(responses, [0.9, 0.5, 0.2]);
    }

    #[test]
    fn sort_is_independent_of_input_order() {
        // The same set fed in two orders must come out identically — this is
        // the property that makes detector output testable.
        let a = Corner::new(CoordinateF64::new(4.0, 1.0), 0.5);
        let b = Corner::new(CoordinateF64::new(2.0, 1.0), 0.5);
        let c = Corner::new(CoordinateF64::new(0.0, 7.0), 0.5);

        let mut forward = vec![a, b, c];
        let mut reverse = vec![c, b, a];
        sort_by_response(&mut forward);
        sort_by_response(&mut reverse);
        assert_eq!(forward, reverse);
        assert_eq!(forward, vec![b, a, c]);
    }

    #[test]
    fn sort_handles_a_nan_response_without_losing_keypoints() {
        let mut corners = vec![
            Corner::new(CoordinateF64::new(0.0, 0.0), 0.5),
            Corner::new(CoordinateF64::new(1.0, 1.0), f32::NAN),
            Corner::new(CoordinateF64::new(2.0, 2.0), 0.1),
        ];
        sort_by_response(&mut corners);
        assert_eq!(corners.len(), 3);
        // total_cmp gives a consistent total order: a positive NaN ranks
        // above every finite response rather than corrupting the sort.
        assert!(corners[0].response().is_nan());
        assert_eq!(corners[1].response(), 0.5);
        assert_eq!(corners[2].response(), 0.1);
    }

    #[test]
    fn sort_of_empty_and_single_slices() {
        let mut empty: Vec<Corner> = vec![];
        sort_by_response(&mut empty);
        assert!(empty.is_empty());

        let mut one = vec![Corner::new(CoordinateF64::new(1.0, 2.0), 0.3)];
        sort_by_response(&mut one);
        assert_eq!(one.len(), 1);
    }

    #[test]
    fn sort_works_for_scale_keypoints_too() {
        // The ordering binds on the capabilities, not on a concrete type.
        let mut kps = vec![
            ScaleKeypoint::new(CoordinateF64::new(0.0, 0.0), 0.1, sigma!(1.0)),
            ScaleKeypoint::new(CoordinateF64::new(0.0, 0.0), 0.8, sigma!(2.0)),
        ];
        sort_by_response(&mut kps);
        assert_eq!(kps[0].scale(), sigma!(2.0));
    }

    #[test]
    fn sort_is_stable_for_fully_tied_keypoints() {
        // Same response and same position, different scale: the detector's
        // order is preserved rather than being arbitrary.
        let coarse = ScaleKeypoint::new(CoordinateF64::new(1.0, 1.0), 0.5, sigma!(3.2));
        let fine = ScaleKeypoint::new(CoordinateF64::new(1.0, 1.0), 0.5, sigma!(1.6));
        let mut kps = vec![coarse, fine];
        sort_by_response(&mut kps);
        assert_eq!(kps, vec![coarse, fine]);
    }

    // ── retain_top_n ────────────────────────────────────────────────────

    #[test]
    fn retain_top_n_keeps_the_strongest() {
        let mut corners = vec![
            Corner::new(CoordinateF64::new(0.0, 0.0), 0.2),
            Corner::new(CoordinateF64::new(1.0, 1.0), 0.9),
            Corner::new(CoordinateF64::new(2.0, 2.0), 0.5),
        ];
        retain_top_n(&mut corners, 2);
        let responses: Vec<f32> = corners.iter().map(HasResponse::response).collect();
        assert_eq!(responses, [0.9, 0.5]);
    }

    #[test]
    fn retain_top_n_resolves_a_tie_at_the_cut_deterministically() {
        let mut corners = vec![
            Corner::new(CoordinateF64::new(9.0, 9.0), 0.1),
            Corner::new(CoordinateF64::new(0.0, 5.0), 0.6),
            Corner::new(CoordinateF64::new(0.0, 3.0), 0.6),
        ];
        retain_top_n(&mut corners, 2);
        let ys: Vec<f64> = corners.iter().map(|c| c.position().y).collect();
        assert_eq!(ys, [3.0, 5.0]);
    }

    #[test]
    fn retain_top_n_with_n_over_length_keeps_everything_sorted() {
        let mut corners = vec![
            Corner::new(CoordinateF64::new(0.0, 0.0), 0.2),
            Corner::new(CoordinateF64::new(1.0, 1.0), 0.9),
        ];
        retain_top_n(&mut corners, 10);
        assert_eq!(corners.len(), 2);
        assert_eq!(corners[0].response(), 0.9);
    }

    #[test]
    fn retain_top_n_with_zero_empties() {
        let mut corners = vec![Corner::new(CoordinateF64::new(0.0, 0.0), 0.2)];
        retain_top_n(&mut corners, 0);
        assert!(corners.is_empty());
    }

    #[test]
    fn retain_top_n_on_empty_input() {
        let mut corners: Vec<Corner> = vec![];
        retain_top_n(&mut corners, 5);
        assert!(corners.is_empty());
    }

    // ── Capability bounds ───────────────────────────────────────────────

    #[test]
    fn a_function_can_bind_the_minimum_capability_it_needs() {
        // What the trait split buys: this compiles for both keypoint types,
        // while a `HasScale` bound would reject `Corner` at compile time.
        fn patch_center(kp: &impl HasPosition) -> (f64, f64) {
            let p = kp.position();
            (p.x, p.y)
        }
        fn patch_radius(kp: &(impl HasPosition + HasScale)) -> f64 {
            3.0 * f64::from(kp.scale().get())
        }

        let corner = Corner::new(CoordinateF64::new(2.0, 4.0), 0.5);
        let kp = ScaleKeypoint::new(CoordinateF64::new(2.0, 4.0), 0.5, sigma!(2.0));
        assert_eq!(patch_center(&corner), (2.0, 4.0));
        assert_eq!(patch_center(&kp), (2.0, 4.0));
        assert_eq!(patch_radius(&kp), 6.0);
    }
}

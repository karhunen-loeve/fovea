//! Corner detectors, and the peak selection they share.
//!
//! Two families live here. They answer the same question by opposite means,
//! and the difference is visible in what they read:
//!
//! | Family | Reads | Says "corner" when | Named by |
//! |---|---|---|---|
//! | **Structure tensor** | the gradient, averaged over a Gaussian window | both eigenvalues of `M` are large | [`Harris`], [`ShiTomasi`] |
//! | **Segment test** | 16 raw intensities on a radius-3 ring | some contiguous arc is entirely brighter, or entirely darker, than the centre | [`SegmentTest`] — the detector known as FAST |
//!
//! ## Which one?
//!
//! - **[`Harris`] / [`ShiTomasi`]**, through [`detect_corners`], when the
//!   result must line up with another library's `cornerHarris`, when the
//!   image is noisy (the window integrates noise away before the decision),
//!   or when you want a graded measure with an algebraic meaning rather than
//!   a pass/fail with a margin.
//! - **[`fast`]** when the threshold has to be chosen without calibration
//!   (it is in intensity units — see below), when localization must not move
//!   with a parameter, when nothing may be allocated per frame beyond the one
//!   score map, or when you are building towards ORB, whose corner stage this
//!   is.
//!
//! Note what is *not* on that list. FAST is famous for being cheap, and in
//! this crate it is not yet reliably cheaper — it depends on the arc length.
//! Map against map on the `benches/features.rs` 512×512 `Mono8` texture
//! (2026-08-07):
//!
//! | Stage | Median |
//! |---|---|
//! | [`fast_score_map`], `arc_length = 9` | 87 ms |
//! | [`fast_score_map`], `arc_length = 12` | 25 ms |
//! | [`fast_score_map`], `arc_length = 16` | 13 ms |
//! | [`corner_response_map`], [`Harris`] | 53 ms |
//! | [`corner_response_map`], [`ShiTomasi`] | 50 ms |
//!
//! So the segment test is 1.7× *slower* at the arc length most people want
//! and 4× faster at the one most people do not. The tensor family spends its
//! time in separable blurs that vectorize; the segment test is a scalar
//! per-pixel scan whose four-cardinal early rejection only bites in
//! proportion to the arc length. Closing that gap is a job for the
//! performance pass, not a reason to prefer one detector's *answers* over the
//! other's — and peak selection is not where the time goes either, at under
//! 3 ms for both families.
//!
//! Both produce [`Corner`](crate::features::Corner), both go through
//! [`corner_peaks`], and both have a pyramid variant that reports in the
//! base-image frame. Swapping one for the other changes two lines.
//!
//! ## The structure tensor
//!
//! A corner is a point where the image gradient points in two different
//! directions at once. The structure tensor
//!
//! ```text
//! M = [ Sxx  Sxy ]     Sxx = w * (Gx·Gx)
//!     [ Sxy  Syy ]     Syy = w * (Gy·Gy)      w = the window, `*` = convolution
//!                      Sxy = w * (Gx·Gy)
//! ```
//!
//! is the local average of the gradient's outer product, and the whole family
//! differs only in how its two eigenvalues are collapsed into one number:
//!
//! | Response | Formula | Reach for it when |
//! |---|---|---|
//! | [`Harris`] | `det(M) − k·tr(M)²` | You want the classical detector, comparable with other libraries' `cornerHarris`. |
//! | [`ShiTomasi`] | `λ_min(M)` | You want a response with an interpretable scale (it is a squared gradient, not a fourth power) and no `k` to tune. |
//!
//! Both are strategies over one engine, so a third measure is an
//! implementation of [`CornerResponse`] rather than a copied pipeline.
//!
//! ## The segment test
//!
//! The pixel `p` is a corner at threshold `t` when some run of
//! [`arc_length`](SegmentTest::arc_length) contiguous pixels on the
//! [`FAST_RING`] is entirely at least `I_p + t`, or entirely at most
//! `I_p − t`. There is no window, no gradient and no filtering: the answer
//! depends on 17 raw samples, and nothing between them and the decision. One
//! consequence is worth relying on — **no parameter moves a detection**. The
//! reported pixels are the same at every threshold and every arc length,
//! where the tensor family's peak drifts inward as its window grows.
//!
//! The score is the largest `t` at which the pixel still passes, so one
//! number serves as both the test's threshold and the peak stage's.
//!
//! What it costs in exchange is grading: the score **saturates**. Once an arc
//! clears the threshold everywhere, several pixels around a corner reach the
//! identical full contrast, and [`corner_peaks`]'s plateau rule then reports
//! the raster-first of each tied cluster rather than its centre. On a clean
//! synthetic step edge that is up to two pixels from the geometric corner —
//! so "does not move with a parameter" is not the same as "is exact". It is
//! the segment test's analogue of the tensor family's inward drift, and the
//! same answer applies: a refinement step, not a different threshold.
//!
//! ## The pipeline, and how to take it apart
//!
//! [`detect_corners`] and [`fast`] are orchestrators, not primitives — the
//! same relationship [`canny`](crate::analyze::edge::canny) has to its
//! stages. Both funnel into one shared tail:
//!
//! ```text
//! Sobel Gx, Gy → products Gx², Gy², Gx·Gy → Gaussian window (σ) ─┐
//!                                                                ├→ scalar map
//! 16 ring samples → segment test → largest passing threshold ────┘      │
//!                                                                       ▼
//!                          corner_peaks: threshold + local maximum → Vec<Corner>
//! ```
//!
//! Every stage is public. [`corner_response_map`] and [`fast_score_map`]
//! return the map itself — to visualize, or to threshold differently, though
//! only *upward* for the segment test, whose map is already floored at its
//! own threshold —
//! [`StructureTensor`] builds the three sums from *your* gradients — so
//! Scharr instead of Sobel, or a box window instead of a Gaussian, needs no
//! new API — [`fast_score_at`] is the segment test on a single pixel, and
//! [`corner_peaks`] turns any map into keypoints.
//!
//! ```
//! use fovea::Sigma;
//! use fovea::border::Clamp;
//! use fovea::features::detect::{corner_peaks, Harris, StructureTensor};
//! use fovea::image::Image;
//! use fovea::pixel::MonoF32;
//! use fovea::transform::{scharr_x, scharr_y};
//!
//! // A hand-built Harris with a Scharr gradient instead of the pinned Sobel.
//! let image: Image<MonoF32> = Image::generate(24, 24, |x, y| {
//!     MonoF32::new(if (8..16).contains(&x) && (8..16).contains(&y) { 1.0 } else { 0.0 })
//! });
//!
//! let gx = scharr_x(&image, &Clamp);
//! let gy = scharr_y(&image, &Clamp);
//! let tensor = StructureTensor::from_gradients(&gx, &gy, Sigma::new(1.2))?;
//! let response = tensor.response(&Harris::new(0.04));
//!
//! // The square's four corners, and nothing else.
//! let peak = corner_peaks(&response, 0.0, 3)
//!     .into_iter()
//!     .map(|c| c.response)
//!     .fold(0.0f32, f32::max);
//! let corners = corner_peaks(&response, 0.2 * peak, 3);
//! assert_eq!(corners.len(), 4);
//! # Ok::<(), fovea::Error>(())
//! ```
//!
//! ## Choosing a threshold
//!
//! The two families differ here more than anywhere else, and it is the
//! practical reason to prefer one over the other.
//!
//! A **structure-tensor threshold is absolute, in the response map's own
//! units**, and those units are not intuitive: the gradient operator's gain
//! enters the response at its own power (Sobel's `[-1 0 1; -2 0 2; -1 0 1]`
//! has a positive-lobe gain of 4, so a unit-contrast step yields `|G| = 4`),
//! and so does image contrast — squared for [`ShiTomasi`], to the fourth
//! power for [`Harris`]. A `Mono8` image therefore produces responses larger
//! than the same picture as `MonoF32` in `0.0..=1.0` by a factor of
//! `255⁴ ≈ 4·10⁹`.
//!
//! Do not guess. Calibrate against [`corner_response_map`] on a
//! representative image and take a fraction of its maximum — that is what
//! the examples here do, and it is the only recipe that survives a change of
//! pixel type, gradient operator, or window σ.
//!
//! A **segment-test threshold is a plain intensity difference**: `20` on a
//! `Mono8` image means twenty grey levels, `0.08` on a `MonoF32` image in
//! `0.0..=1.0` means eight per cent contrast, and the two say the same thing.
//! Nothing is squared and no operator gain enters, so the number can be
//! reasoned about — from a noise estimate, say — instead of calibrated.
//!
//! What both refuse is a "quality level" knob relative to the strongest
//! corner in *this* frame: it makes a detection depend on the rest of the
//! frame, which is a decision for the caller (PHILOSOPHY §8), not for the
//! detector. It is also three lines over a public map.
//!
//! ## Scale
//!
//! No detector here selects scale — they find corners at the resolution (and,
//! for the tensor family, the window σ) they are given, which is why they all
//! return [`Corner`](crate::features::Corner) (position + response) rather
//! than [`ScaleKeypoint`](crate::features::ScaleKeypoint). Running a detector
//! over a [`Pyramid`](crate::image::Pyramid) is multi-resolution, not scale
//! selection: it finds more corners, but none of them has a *characteristic*
//! σ that the detector chose. [`detect_corners_in_level`] and
//! [`fast_in_level`] are those variants, reporting every level's detections
//! in the base-image frame so they are directly comparable:
//!
//! ```
//! use fovea::{CoordinateF64, PixelDistance, Sigma};
//! use fovea::features::HasPosition;
//! use fovea::features::detect::{detect_corners_in_level, CornerParams, ShiTomasi};
//! use fovea::image::{Image, ScaledImage};
//! use fovea::pixel::MonoF32;
//! use fovea::transform::pyr_down;
//!
//! let base: Image<MonoF32> = Image::generate(48, 48, |x, y| {
//!     MonoF32::new(if (16..32).contains(&x) && (16..32).contains(&y) { 1.0 } else { 0.0 })
//! });
//!
//! // Octave 1: pyr_down keeps even samples — distance 2, origin unshifted.
//! let level = ScaledImage::new(
//!     pyr_down(&base),
//!     PixelDistance::new(2.0),
//!     CoordinateF64::new(0.0, 0.0),
//!     Sigma::new(1.0),
//! );
//!
//! let params = CornerParams::try_new(Sigma::new(1.0), 0.05, 2)?;
//! let corners = detect_corners_in_level(&level, &ShiTomasi, params);
//!
//! // Found on a 24×24 level, reported in the 48×48 base frame: an x of 30
//! // is not a coordinate the level could have produced.
//! assert_eq!(corners.len(), 4);
//! assert!(corners.iter().any(|c| c.position().x > 24.0), "{corners:?}");
//! # Ok::<(), fovea::Error>(())
//! ```

mod fast;
mod peaks;
mod structure_tensor;

pub use fast::{
    FAST_RING, FAST_RING_RADIUS, FastParams, SegmentTest, fast, fast_in_level, fast_score_at,
    fast_score_map,
};
pub use peaks::corner_peaks;
pub use structure_tensor::{
    CornerParams, CornerResponse, CornerResponseChannel, Harris, ShiTomasi, StructureTensor,
    corner_response_map, detect_corners, detect_corners_in_level,
};

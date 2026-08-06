//! Feature detection: keypoints and the capabilities that describe them.
//!
//! A keypoint is the hinge of the feature pipeline — detectors produce them,
//! descriptors and matchers consume them. This module defines that data
//! model. The naïve version is one struct with every field
//! (`x, y, scale, orientation, response, octave, …`), which leaves half the
//! fields meaningless for half the detectors: a FAST corner has no scale and
//! no orientation, a Harris corner has a response but no intrinsic scale.
//! Here, each property a keypoint may carry is **one trait adding one
//! guarantee**, the same progressive layering as the pixel and image traits.
//!
//! ## Which capability?
//!
//! | Trait | Guarantee | Bind it when |
//! |---|---|---|
//! | [`HasPosition`](crate::features::HasPosition) | sub-pixel location, always base-image frame | You sample or draw at the keypoint. |
//! | [`HasResponse`](crate::features::HasResponse) | detector strength, ranking only | You rank, threshold, or keep the top-N. |
//! | [`HasScale`](crate::features::HasScale) | characteristic σ in base-image pixels | You size a patch or window from the detection. |
//! | [`HasOrientation`](crate::features::HasOrientation) | dominant gradient direction, radians in (−π, π] | You need rotation invariance. |
//!
//! ## Which keypoint type?
//!
//! | Type | Carries | Produced by |
//! |---|---|---|
//! | [`Corner`](crate::features::Corner) | position + response | Single-resolution corner detectors (Harris, Shi-Tomasi, FAST). |
//! | [`ScaleKeypoint`](crate::features::ScaleKeypoint) | position + response + scale | Scale-selecting detectors searching a pyramid or scale space. |
//!
//! Detectors return the *precise* type they can justify; consumers state the
//! *minimum* they require in their bounds. Handing a `Corner` to an
//! operation that needs a scale is then a compile error rather than a
//! silently wrong patch size — and a new capability (an affine shape, say)
//! arrives as a new trait without editing any existing type.
//!
//! ## Where the keypoints come from
//!
//! [`detect`](crate::features::detect) holds the detectors themselves.
//! Today that is the
//! structure-tensor family: [`Harris`](crate::features::detect::Harris) and
//! [`ShiTomasi`](crate::features::detect::ShiTomasi) are two responses over
//! one shared pipeline, reached through
//! [`detect_corners`](crate::features::detect::detect_corners) for a single
//! image and
//! [`detect_corners_in_level`](crate::features::detect::detect_corners_in_level)
//! for a pyramid level. Both produce [`Corner`](crate::features::Corner):
//! they localize and score, and select no scale.
//!
//! ## Positions live in the base-image frame
//!
//! A keypoint detected on a coarse pyramid level is *reported* in the frame
//! of the base image, so keypoints from different levels are comparable.
//! The lift happens in exactly one place —
//! [`Decimated::to_base`](crate::image::Decimated::to_base), reached through
//! [`Corner::from_level`](crate::features::Corner::from_level) and
//! [`ScaleKeypoint::from_level`](crate::features::ScaleKeypoint::from_level)
//! — because the
//! hand-rolled `x · 2^level` alternative drops the grid-origin term and
//! drifts by half a pixel per octave, which would defeat the sub-pixel
//! localization the `f64` positions exist to preserve. Sampling *back* into
//! a level applies the inverse,
//! [`Decimated::to_local`](crate::image::Decimated::to_local).
//!
//! ## Selection is deterministic
//!
//! [`retain_top_n`](crate::features::retain_top_n) and
//! [`sort_by_response`](crate::features::sort_by_response) order by response
//! with a tie-break on `(y, x)`
//! ([`by_response_then_position`](crate::features::by_response_then_position)).
//! Exact response
//! ties are the rule rather than the exception on synthetic images, so
//! without the tie-break "the strongest 50 corners" would depend on the
//! order the detector visited pixels in — and could not be asserted in a
//! test.
//!
//! # Example
//!
//! Detect on a coarse level, report in base-image coordinates, keep the
//! strongest:
//!
//! ```
//! use fovea::{CoordinateF64, PixelDistance, Sigma};
//! use fovea::features::{retain_top_n, Corner, HasPosition};
//! use fovea::image::{Image, ImageView, ScaledImage};
//! use fovea::pixel::MonoF32;
//! use fovea::transform::pyr_down;
//!
//! let base: Image<MonoF32> = Image::fill(16, 16, MonoF32::new(0.5));
//! let coarse = pyr_down(&base);
//!
//! // pyr_down keeps even samples: distance 2, origin unshifted, σ = 1.
//! let level = ScaledImage::new(
//!     coarse,
//!     PixelDistance::new(2.0),
//!     CoordinateF64::new(0.0, 0.0),
//!     Sigma::new(1.0),
//! );
//! assert_eq!(level.size().width, 8);
//!
//! // Two detections in the level's own coordinates.
//! let mut corners = vec![
//!     Corner::from_level(&level, CoordinateF64::new(1.0, 1.0), 0.3),
//!     Corner::from_level(&level, CoordinateF64::new(3.5, 2.0), 0.9),
//! ];
//!
//! retain_top_n(&mut corners, 1);
//! assert_eq!(corners[0].position(), CoordinateF64::new(7.0, 4.0));
//! ```

pub mod detect;

mod keypoint;

pub use keypoint::{
    Corner, HasOrientation, HasPosition, HasResponse, HasScale, ScaleKeypoint,
    by_response_then_position, retain_top_n, sort_by_response,
};

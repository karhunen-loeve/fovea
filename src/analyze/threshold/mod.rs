//! Threshold-based segmentation.
//!
//! Operations that turn a single-channel image into a
//! [`BinaryImage`](crate::image::BinaryImage) by comparing pixel values
//! against one or more explicit thresholds.
//!
//! Each threshold is a named caller argument — this module never *infers*
//! a cut point from the data. The data-driven counterpart
//! is [`otsu_threshold`](crate::analyze::histogram::otsu_threshold), which
//! lives in [`histogram`](crate::analyze::histogram) because it consumes a
//! histogram to *choose* its threshold; the functions here take the
//! threshold as given.
//!
//! ## Which threshold?
//!
//! | Question | Reach for |
//! |---|---|
//! | "Keep weak edges only if they connect to a strong one." | [`hysteresis_threshold`] |
//! | "Threshold each pixel against its own local neighbourhood (uneven lighting)." | [`adaptive_threshold`] |
//! | "Pick the threshold for me from the histogram." | [`otsu_binary_mask`](crate::analyze::histogram::otsu_binary_mask) |

mod adaptive;
mod hysteresis;

pub use adaptive::{AdaptiveAccumulator, Bias, adaptive_threshold, adaptive_threshold_into};
pub use hysteresis::{HysteresisThresholds, hysteresis_threshold, hysteresis_threshold_into};

//! Image analysis operations.
//!
//! Analysis operations consume an image and produce data *about* it —
//! histograms, statistics, descriptors — rather than producing a new image.
//! That distinguishes this module from [`crate::transform`], whose
//! operations produce images.
//!
//! Currently exposed:
//!
//! - [`histogram`](crate::analyze::histogram) — per-channel value histograms with explicit binning
//!   strategies.
//! - [`integral`](crate::analyze::integral) — summed-area tables for `O(1)` rectangular region
//!   sums (and sums of squares) with an explicit, type-checked
//!   accumulator pixel.
//! - [`components`] — connected-component labeling on binary images,
//!   with optional per-component stats (area, bounding box, centroid).

pub mod components;
pub mod histogram;
pub mod integral;

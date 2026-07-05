//! Image analysis operations.
//!
//! Analysis operations consume an image and produce data *about* it —
//! histograms, statistics, descriptors — rather than producing a new image.
//! That distinguishes this module from [`crate::transform`], whose
//! operations produce images.
//!
//! ## Which analysis?
//!
//! | Question | Start with | Output |
//! |---|---|---|
//! | "How are channel values distributed?" | [`histogram`](crate::analyze::histogram) | Counts per bin, per channel. |
//! | "Where are the edges in this image?" | [`edge::canny`](crate::analyze::edge::canny) | Binary edge mask. |
//! | "What threshold separates foreground?" | [`histogram::otsu_threshold`](crate::analyze::histogram::otsu_threshold) / [`histogram::otsu_binary_mask`](crate::analyze::histogram::otsu_binary_mask) | Threshold value or binary mask. |
//! | "Keep weak edges only if connected to a strong one?" | [`threshold::hysteresis_threshold`](crate::analyze::threshold::hysteresis_threshold) | Binary mask. |
//! | "Threshold each pixel against its local neighbourhood (uneven lighting)?" | [`threshold::adaptive_threshold`](crate::analyze::threshold::adaptive_threshold) | Binary mask. |
//! | "What is the sum of this rectangle?" | [`integral`](crate::analyze::integral) | Summed-area table with explicit accumulator pixels. |
//! | "How many foreground blobs are in this mask?" | [`components`](crate::analyze::components) | Label image and optional component stats. |
//! | "What shape are the blobs (perimeter, orientation, roundness)?" | [`components::connected_components_with_measurements`](crate::analyze::components::connected_components_with_measurements) | Per-blob moments + perimeter, with derived shape descriptors. |
//!
//! Do not use this module for operations that produce another image of the
//! same conceptual kind. Those belong in [`crate::transform`].

pub mod components;
pub mod edge;
pub mod histogram;
pub mod integral;
pub mod threshold;

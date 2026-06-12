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
//! | "What threshold separates foreground?" | [`histogram::otsu_threshold`](crate::analyze::histogram::otsu_threshold) / [`histogram::otsu_binary_mask`](crate::analyze::histogram::otsu_binary_mask) | Threshold value or binary mask. |
//! | "What is the sum of this rectangle?" | [`integral`](crate::analyze::integral) | Summed-area table with explicit accumulator pixels. |
//! | "How many foreground blobs are in this mask?" | [`components`](crate::analyze::components) | Label image and optional component stats. |
//!
//! Do not use this module for operations that produce another image of the
//! same conceptual kind. Those belong in [`crate::transform`].

pub mod components;
pub mod histogram;
pub mod integral;

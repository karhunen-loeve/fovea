#![doc = include_str!("../README.md")]

//! ## Modules
//!
//! - [`image`] — Image types, views, kernels, and containers
//! - [`pixel`] — Pixel type definitions and traits
//! - [`transform`] — Image transformations (convolution, morphology, combine, etc.)
//! - [`analyze`] — Image analysis (histograms, integral images / summed-area tables)
//! - [`border`] — Border policies for neighborhood operations
//!
//! ## Transform overview
//!
//! The [`transform`] module organises operations into three levels:
//!
//! 1. **Unary pixel transforms** — apply a function to every pixel independently
//!    ([`transform::convert_image`], strategies like [`transform::Luminance`],
//!    [`transform::SrgbGamma`], [`transform::Narrow`]).
//! 2. **Binary pixel transforms** — combine two same-sized images pixel-wise
//!    ([`transform::combine_images`], strategies like [`transform::PixelAdd`],
//!    [`transform::AbsDiff`], [`transform::Blend`]).
//! 3. **Neighbourhood transforms** — compute each output pixel from a window of
//!    input pixels: [`transform::fold_neighborhood`] for weighted aggregation
//!    (convolution, separable filters) and [`transform::map_neighborhood`] for
//!    non-linear operations (erosion, dilation, median).

extern crate self as fovea;

mod common;
mod error;
mod internal;

/// Image types, views, and containers.
///
/// This module contains all image-related types:
/// - [`Image`](image::Image), [`ImageRef`](image::ImageRef), [`ImageRefMut`](image::ImageRefMut) — owned and borrowed images
/// - [`ImageArray`](image::ImageArray) — compile-time sized images
/// - [`ImageView`](image::ImageView), [`ImageViewMut`](image::ImageViewMut) — trait-based access
/// - [`ContiguousImage`](image::ContiguousImage), [`PlainImage`](image::PlainImage), [`PlainImageMut`](image::PlainImageMut) — progressive trait layers
/// - [`SubView`](image::SubView), [`SubViewMut`](image::SubViewMut) — region-of-interest access
/// - [`Neighborhood`](image::Neighborhood), [`Kernel`](image::Kernel) — kernel/mask types
/// - [`ImagePlanes`](image::ImagePlanes) — planar image representation
/// - [`zip_pixels`](image::zip_pixels) — pixel-pair iteration
pub mod image;

/// Border policies for out-of-bounds pixel access in neighborhood operations.
pub mod border {
    pub use crate::image::border::*;
}

pub use fovea_derive::HomogeneousPixel;
pub use fovea_derive::LinearPixel;
pub use fovea_derive::PlainPixel;
pub use fovea_derive::ZeroablePixel;

/// The pixel module contains definitions and implementations related to pixel types and operations.
pub mod pixel;

/// The `transform` module contains definitions and implementations related to image transformations.
pub mod transform;

/// Image analysis operations (histograms, statistics, descriptors).
///
/// Distinguished from [`transform`] by output: analysis operations produce
/// data *about* an image (counts, scalars, descriptors), not new images.
pub mod analyze;

// ── Core vocabulary types (module-agnostic, kept at root) ────────────────────
pub use common::{Coordinate, Rectangle, Size, Stride};
pub use error::Error;

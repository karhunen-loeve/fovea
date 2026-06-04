#![doc = include_str!("../README.md")]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
#![warn(unreachable_pub)]
#![deny(rustdoc::broken_intra_doc_links)]

//! ## docs.rs guide pages
//!
//! The crate root above is the quick evaluation path. For task-oriented docs,
//! use the documentation-only [`guide`] module on docs.rs:
//!
//! - [`guide`] — first working examples and the core pipeline shape
//! - [`guide::faq`] — common questions about images, pixels, conversions, ROIs, and large images
//! - [`guide::pixel_types`] — how to choose the right pixel type
//! - [`guide::camera_buffers`] — raw bytes, camera SDK buffers, and byte layout
//! - [`guide::large_images`] — slices, rows, tiles, sliding windows, and parallel runtimes

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
///
/// Start with [`Clamp`](border::Clamp) for ordinary image filtering,
/// [`Skip`](border::Skip) for valid-only convolution, and
/// [`Constant`](border::Constant) when the outside of the image should have a
/// known value. These policies are used by neighborhood transforms such as
/// convolution, filters, and morphology.
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

#[cfg(doc)]
pub mod guide;

// ── Core vocabulary types (module-agnostic, kept at root) ────────────────────
pub use common::{Coordinate, Rectangle, Size, Stride};
pub use error::Error;

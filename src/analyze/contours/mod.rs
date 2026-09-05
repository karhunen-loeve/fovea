//! Contour extraction: border tracing, outer/hole hierarchy, chain
//! codes, polygon simplification, and contour-derived shape descriptors.
//!
//! Where [`components`](crate::analyze::components) tells you *which*
//! pixels form each blob and measures them in aggregate, this module
//! recovers each blob's **geometry**: the closed chain of border pixels
//! around its outside ([`ContourKind::Outer`]) and around each of its
//! holes ([`ContourKind::Hole`]), plus how components nest inside each
//! other's holes. Everything starts from [`extract_contours`].
//!
//! ## Which representation?
//!
//! | Question | Use | Output |
//! |---|---|---|
//! | "What is the outline of each blob, with holes and nesting?" | [`extract_contours`] | [`ContourHierarchy`] alongside the [`Labeling`](crate::analyze::components::Labeling). |
//! | "How round / convex is this blob, geometrically?" | [`Contour::circularity`] / [`Contour::solidity`] | On-demand `f64` descriptors from the traced polygon. |
//! | "How many holes does this blob have?" | [`ComponentContour::euler_number`] | `1 − holes`. |
//! | "A compact encoding to store or compare?" | [`Contour::chain_code`] | [`ChainCode`], one byte per border step. |
//! | "Fewer vertices / smoother outline?" | [`approximate_polygon`] | Simplified vertex list, explicit ε. |
//! | "Geometry of an arbitrary vertex list?" | [`polygon_area`] / [`polygon_perimeter`] / [`polygon_centroid`] / [`convex_hull`] | Free functions, no `Contour` needed. |
//!
//! Contour points are **integer pixel coordinates** — the pixels the
//! tracer visited. Tracing knows which pixels form the border, not where
//! the underlying edge crosses them, and it works from a binary mask, which
//! no longer holds the greyscale evidence of where the boundary really is.
//! So interpolating the border between pixels is a separate step over
//! different inputs and is deliberately not folded in here:
//! [`analyze::peak::interpolate_ridge_points`](crate::analyze::peak::interpolate_ridge_points)
//! takes the contour's points as its sites (`points().iter().copied()`,
//! since the parameter is `impl IntoIterator<Item = Coordinate>`), plus the
//! gradient magnitude and gradient pair of the image the mask came from,
//! and returns one interpolated position per vertex in vertex order.
//!
//! Cheap aggregate measurements (pixel-count area, boundary-pixel count,
//! moments) remain single-pass in
//! [`connected_components_with_measurements`](crate::analyze::components::connected_components_with_measurements);
//! extraction here costs two labeling passes plus the traces, so reach
//! for it when you need the geometry, not for area alone.

mod chain;
mod extract;
mod hierarchy;
mod polygon;

pub use chain::{ChainCode, ChainDirection};
pub use extract::extract_contours;
pub use hierarchy::{ComponentContour, Contour, ContourHierarchy, ContourKind};
pub use polygon::{
    approximate_polygon, convex_hull, polygon_area, polygon_centroid, polygon_perimeter,
};

// The connectivity vocabulary is shared with `components`; re-exported
// here because every `extract_contours` call names one.
pub use crate::analyze::components::{Connectivity, Connectivity4, Connectivity8};

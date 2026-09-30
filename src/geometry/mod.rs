//! Units, positions and mappings of the plane.
//!
//! fovea reports positions in pixels. This module gives a position its unit
//! and converts it into the units of the world through a mapping whose
//! class is part of its type, so that the class decides at compile time
//! which geometric objects convert.
//!
//! ## Units
//!
//! A position is a [`Point<U>`](crate::geometry::Point), a displacement a
//! [`Vector<U>`](crate::geometry::Vector), a length a [`Length<U>`](crate::geometry::Length). The unit `U` is
//! [`Pixels`] or a metre with a prefix, [`Meter<P>`](crate::geometry::Meter): [`Millimeter`],
//! [`Micrometer`], or a bare [`Meter`]. Mixing units does not compile, and
//! a prefix changes only through an explicit, exact `convert`.
//!
//! ## Mappings
//!
//! A calibration is a mapping from [`Pixels`] to a unit of the world. The
//! classes differ in what they preserve:
//!
//! | Mapping | Form | Typical case | Converts |
//! |---|---|---|---|
//! | [`UniformScale`] | `a·x + t` | square pixels, camera aligned with the axes | points, vectors, lengths, circles |
//! | [`AxisScale`] | `diag(a, b)·x + t` | line-scan camera, binned sensor | points, vectors |
//! | [`Similarity`] | `a·R·x + t` | square pixels, camera at an angle | points, vectors, lengths, circles |
//! | [`Affine`] | `A·x + t` | oblique view, in the affine approximation | points, vectors |
//!
//! Three traits carry the capabilities. Every mapping is a [`PlaneMap`] and
//! converts points. The four classes above are [`AffineMap`]s: their metric
//! is the same everywhere, so displacements convert on their own, each has a
//! closed-form inverse, and a circle becomes an ellipse. The uniform scale
//! and the similarity are also [`ConformalMap`]s, which multiply every
//! length by one factor, keep angles and keep circles round.
//!
//! Reflections are allowed: `axis_scale!(0.02, -0.02)` maps an image frame
//! with `y` pointing down to a machine frame with `y` pointing up, and
//! [`AffineMap::reverses_orientation`] says so.
//!
//! ## Measuring in the right space
//!
//! A fit or a distance is computed in a space whose Euclidean distances are
//! the ones the measurement means. Under a conformal mapping the image
//! qualifies up to the factor, so a line or circle fitted in pixels, and its
//! residuals, convert afterwards. Under any other mapping the points convert
//! first: with non-square pixels a circle in the image is an ellipse on the
//! part, and a fit in pixels minimises the wrong distances. The types
//! enforce the first half: only a [`ConformalMap`] converts a
//! [`Length`].
//!
//! [`Pixels`]: crate::geometry::Pixels
//! [`Meter`]: crate::geometry::Meter
//! [`Millimeter`]: crate::geometry::Millimeter
//! [`Micrometer`]: crate::geometry::Micrometer
//! [`Length`]: crate::geometry::Length
//! [`UniformScale`]: crate::geometry::UniformScale
//! [`AxisScale`]: crate::geometry::AxisScale
//! [`Similarity`]: crate::geometry::Similarity
//! [`Affine`]: crate::geometry::Affine
//! [`PlaneMap`]: crate::geometry::PlaneMap
//! [`AffineMap`]: crate::geometry::AffineMap
//! [`ConformalMap`]: crate::geometry::ConformalMap
//! [`AffineMap::reverses_orientation`]: crate::geometry::AffineMap::reverses_orientation
//!
//! # Example
//!
//! ```
//! use fovea::{Millimeter, Pixels, Point};
//! use fovea::geometry::{AffineMap, AxisScale, Vector};
//!
//! // A line-scan camera: 20 µm per pixel across the line, 50 µm of feed
//! // per line.
//! let scale: AxisScale<Pixels, Millimeter> = fovea::axis_scale!(0.02, 0.05);
//!
//! // Two edges found in the image, converted before measuring.
//! let left = scale.map_point(Point::new(120.25, 40.0));
//! let right = scale.map_point(Point::new(870.75, 40.0));
//! assert!((left.distance(right).get() - 15.01).abs() < 1e-12);
//!
//! // A displacement converts on its own, and its length depends on its
//! // direction: one pixel down is 2.5 times one pixel across.
//! let across = scale.length_of(Vector::new(1.0, 0.0));
//! let down = scale.length_of(Vector::new(0.0, 1.0));
//! assert!((down.get() / across.get() - 2.5).abs() < 1e-12);
//! ```

mod affine;
mod map;
mod point;
mod units;

pub use affine::{Affine, AxisScale, Similarity, UniformScale};
pub use map::{AffineMap, ConformalMap, PlaneMap};
pub use point::{Length, Point, Vector};
pub use units::{LengthUnit, Meter, Micro, Micrometer, Milli, Millimeter, Pixels, Prefix, Unit};

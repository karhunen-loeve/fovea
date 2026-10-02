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
//! | [`UniformScale`] | `a·x + t` | square pixels, camera aligned with the axes | points, vectors, lines, ellipses, lengths, circles |
//! | [`AxisScale`] | `diag(a, b)·x + t` | line-scan camera, binned sensor | points, vectors, lines, ellipses |
//! | [`Similarity`] | `a·R·x + t` | square pixels, camera at an angle | points, vectors, lines, ellipses, lengths, circles |
//! | [`Affine`] | `A·x + t` | oblique view, in the affine approximation | points, vectors, lines, ellipses |
//! | [`Homography`] | `H·x`, projective | camera at an angle to a plane | points |
//! | [`BrownConrady`] | lens distortion | a real lens, from its calibration | points |
//!
//! Three traits carry the capabilities. Every mapping is a [`PlaneMap`] and
//! converts points. The first four classes are [`AffineMap`]s: their metric
//! is the same everywhere, so displacements convert on their own, each has a
//! closed-form inverse, and a circle becomes an ellipse. The uniform scale
//! and the similarity are also [`ConformalMap`]s, which multiply every
//! length by one factor, keep angles and keep circles round. Under a
//! homography or a lens, lengths change across the image, so only points
//! convert, and a measurement converts its points before it fits.
//!
//! ## Remapping
//!
//! A mapping from pixels to pixels rectifies or undistorts an image through
//! [`remap`](crate::transform::remap). The direction is in the type:
//! [`DestToSource`] reads a mapping from each destination pixel to its
//! source, which is how a lens model is used; [`SourceToDest`] takes the
//! mapping the way it is usually thought of and inverts it once.
//! [`PlaneMap::then`] chains a mapping with a lens model, so undistorting
//! and rectifying is one remap and one interpolation. For measurement,
//! correct the measured points with [`BrownConrady::undistort_point`]
//! instead of resampling the image.
//!
//! Reflections are allowed: `axis_scale!(0.02, -0.02)` maps an image frame
//! with `y` pointing down to a machine frame with `y` pointing up, and
//! [`AffineMap::reverses_orientation`] says so.
//!
//! ## Elements
//!
//! A [`Line`] (infinite, with a direction), a [`Segment`], a [`Circle`] and
//! an [`Ellipse`] carry the unit of their points. They are what the fits of
//! [`measure`](crate::measure) return, and they convert by the same rule:
//! lines, segments and ellipses under every [`AffineMap`], circles only
//! under a [`ConformalMap`], because under any other mapping a circle
//! becomes an ellipse. All four are also at the crate root.
//!
//! ## Relations
//!
//! Each element answers the distance of a point from it and the point of
//! it nearest to a point (`distance`, `closest_point`), exactly for the
//! ellipse too. Two lines give the directed angle from one to the other
//! ([`Line::angle_to`]) and where they meet ([`Line::intersection`]); a
//! segment gives its nearest and furthest distance from a line
//! ([`Segment::distances_to`]), which is the width between two fitted
//! edges. Relations compute in the unit of their operands, so fit and
//! relate in pixels under a conformal calibration and convert the result,
//! and convert the points first under any other.
//!
//! ```
//! use fovea::{Millimeter, Pixels, Point};
//! use fovea::geometry::{ConformalMap, UniformScale};
//! use fovea::measure::{AllPoints, Taubin, try_fit};
//!
//! // Two holes of radius 40 px, their edges found in the image.
//! let edge = |cx: f64, cy: f64| -> Vec<Point<Pixels>> {
//!     (0..36)
//!         .map(|k| {
//!             let t = (10.0 * k as f64).to_radians();
//!             Point::new(cx + 40.0 * t.cos(), cy + 40.0 * t.sin())
//!         })
//!         .collect()
//! };
//! let left = try_fit(&edge(200.0, 300.0), Taubin, AllPoints)?;
//! let right = try_fit(&edge(680.0, 300.0), Taubin, AllPoints)?;
//!
//! // The pitch of the holes, in pixels and on the part.
//! let pitch = left.element().center().distance(right.element().center());
//! let scale: UniformScale<Pixels, Millimeter> = fovea::uniform_scale!(0.0125);
//! assert!((scale.map_length(pitch).get() - 6.0).abs() < 1e-9);
//! # Ok::<(), fovea::Error>(())
//! ```
//!
//! [`Line::angle_to`]: crate::geometry::Line::angle_to
//! [`Line::intersection`]: crate::geometry::Line::intersection
//! [`Segment::distances_to`]: crate::geometry::Segment::distances_to
//!
//! [`Line`]: crate::geometry::Line
//! [`Segment`]: crate::geometry::Segment
//! [`Circle`]: crate::geometry::Circle
//! [`Ellipse`]: crate::geometry::Ellipse
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
//! [`PlaneMap::then`]: crate::geometry::PlaneMap::then
//! [`Homography`]: crate::geometry::Homography
//! [`BrownConrady`]: crate::geometry::BrownConrady
//! [`BrownConrady::undistort_point`]: crate::geometry::BrownConrady::undistort_point
//! [`DestToSource`]: crate::geometry::DestToSource
//! [`SourceToDest`]: crate::geometry::SourceToDest
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
mod compose;
mod elements;
mod homography;
mod lens;
mod map;
mod point;
mod relations;
mod units;

pub use affine::{Affine, AxisScale, Similarity, UniformScale};
pub use compose::{Chain, Compose, DestToSource, Invertible, SourceLookup, SourceToDest};
pub use elements::{Circle, Element, Ellipse, Line, Segment};
pub use homography::Homography;
pub use lens::{
    BrownConrady, BrownConradyCoefficients, CameraMatrix, FocalLength, Radial, Tangential,
};
pub use map::{AffineMap, ConformalMap, PlaneMap};
pub use point::{Length, Point, Vector};
pub use units::{LengthUnit, Meter, Micro, Micrometer, Milli, Millimeter, Pixels, Prefix, Unit};

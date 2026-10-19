//! Measurement tools: the caliper and the geometric fits.
//!
//! A caliper is a path placed across the edges to be measured, a width to
//! average over, and a sampling step. [`profile`](crate::measure::profile)
//! reads the intensities along it, [`Profile::edges`](crate::measure::Profile::edges)
//! finds every edge of a polarity above a contrast, and
//! [`Edges::pairs`](crate::measure::Edges::pairs) forms edge pairs by a rule
//! the caller names. Picking one edge or one pair is always a named step;
//! nothing is chosen silently. An edge with no other edge within about
//! `7σ` and no more blur than σ is placed to about 0.001 px; closer edges
//! and blurrier ones are biased, by amounts the accuracy section of
//! [`Profile::edges`](crate::measure::Profile::edges) gives, so a pair of
//! width `W` wants σ at most `W / 7`.
//!
//! [`fit`](crate::measure::fit) fits a line, circle or ellipse to
//! points, from caliper edges, from
//! [`interpolate_edge_points`](crate::analyze::edge::interpolate_edge_points)
//! or from a contour, with an estimator and an outlier rule the caller
//! writes out: `fit(&points, Taubin, tukey!(0.5))`. The
//! [`Fit`](crate::measure::Fit) carries the element and how well it fits,
//! as exact geometric distances: the RMS residual over the points that
//! shaped it, and the maximum residual and the form deviation over all
//! points, so a defect the outlier rule rejected still shows.
//!
//! Through a lens that distorts there are two paths, and the types keep
//! them apart:
//!
//! - **Measure on the camera image, then correct the points.** The edges
//!   are [`Point<Pixels>`](crate::Point), and
//!   [`BrownConrady::undistort_point`](crate::geometry::BrownConrady::undistort_point)
//!   turns each into a point of the ideal image,
//!   [`IdealPixels`](crate::IdealPixels), before fitting. This is the
//!   recommended path.
//! - **Remap the image through the lens model, then measure.** The
//!   [`remap`](crate::transform::remap) returns the image in `IdealPixels`,
//!   and a caliper on it reports its edges there. The remap interpolates
//!   the image once before the caliper interpolates it again, moves each
//!   edge by a fraction of a pixel that depends on where it lies, and
//!   rounds and clips into the output pixel type unless that is a float
//!   type.
//!
//! Either way the points end in the ideal image, where a calibration made
//! on the ideal image takes them. A point corrected twice, or a camera
//! position put through such a calibration, does not compile.
//!
//! A caliper lies in the space of the image it reads: its constructors take
//! the space of their points, and [`profile`](crate::measure::profile)
//! requires the image's. A caliper planned in world units is converted into
//! the space the calibration starts from: its points through the inverse
//! mapping, its width and step through a
//! [`ConformalMap`](crate::geometry::ConformalMap), the only class that
//! converts lengths. Widths along the path are [`Length`](crate::Length)s in
//! the caliper's space. A width converts to world units through a conformal
//! map; under any other calibration, convert the two edge positions and
//! take their distance.
//!
//! # Example
//!
//! ```
//! use fovea::{Length, Millimeter, Pixels, Point};
//! use fovea::border::Skip;
//! use fovea::geometry::{ConformalMap, UniformScale};
//! use fovea::image::Image;
//! use fovea::measure::{Caliper, Neighbors, Polarity, profile};
//! use fovea::pixel::MonoF32;
//! use fovea::transform::CatmullRom;
//!
//! // A light bar on a dark background, from x = 20.5 to x = 35.5.
//! let img = Image::generate(64, 32, |x, _| {
//!     MonoF32::new(if (21..=35).contains(&x) { 220.0 } else { 20.0 })
//! });
//!
//! // Across the bar, averaging 5 px, a sample every quarter pixel.
//! let cal = Caliper::try_segment(
//!     Point::new(5.0, 16.0), Point::new(55.0, 16.0), Length::new(5.0), Length::new(0.25),
//! )?;
//! let edges = profile(&img, &cal, CatmullRom, &Skip)?
//!     .edges(Polarity::Either, fovea::sigma!(1.0), fovea::min_contrast!(50.0));
//! let bar = edges.pairs(Polarity::DarkToLight, Neighbors)[0];
//! assert!((bar.width().get() - 15.0).abs() < 1e-6);
//!
//! // 12.5 µm per pixel, square: the width converts directly.
//! let scale: UniformScale<Pixels, Millimeter> = fovea::uniform_scale!(0.0125);
//! assert!((scale.map_length(bar.width()).get() - 0.1875).abs() < 1e-9);
//! # Ok::<(), fovea::Error>(())
//! ```

mod caliper;
mod edges;
mod estimators;
mod fit;

pub use caliper::{Caliper, MAX_CALIPER_SAMPLES, Profile, profile};
pub use edges::{
    Edge, EdgePair, Edges, MinContrast, Neighbors, PairRule, Polarity, StrongestOfRun,
};
pub use fit::{
    AllPoints, Estimator, Fit, Fitzgibbon, Huber, OutlierRule, Taubin, TotalLeastSquares, Tukey,
    fit,
};

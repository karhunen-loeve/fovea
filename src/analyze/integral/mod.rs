//! Summed-area tables (integral images).
//!
//! An *integral image*, also known as a *summed-area table*, is a
//! pre-computed `(W+1) × (H+1)` table that lets the sum of any
//! axis-aligned rectangle in the source image be computed with **four
//! memory lookups and three arithmetic operations**, independent of the
//! rectangle's size. It is the underlying primitive for fast box
//! filters, adaptive thresholding, Haar-feature evaluation, and the
//! `sum / sum-of-squares` half of normalised cross-correlation.
//!
//! This module is the implementation of
//! [ADR-0032](https://github.com/karhunen-loeve/irys-cv/blob/main/docs/adr/0032-integral-image-design.md);
//! the file/PR breakdown lives in the project's `INTEGRAL_IMAGE_PLAN.md`.
//!
//! # Surface
//!
//! - [`IntegralImage<A>`] — the output type. `A` is the **accumulator
//!   pixel** (named explicitly by the caller, ADR-0032 §1). It is *not*
//!   an [`ImageView`](crate::image::ImageView) — see ADR-0032 §4 — so
//!   regular image consumers cannot accidentally accept it.
//! - [`integral_image`] / [`integral_image_into`] — compute the
//!   summed-area table of a source image, with an `O(1)` pre-flight
//!   overflow check that guarantees the inner loop is overflow-free
//!   when it succeeds.
//! - [`integral_squared_image`] / [`integral_squared_image_into`] —
//!   the sum-of-squares variant, needed for variance / normalised
//!   cross-correlation.
//!
//! Valid `(source pixel, accumulator pixel)` pairs are gated at compile
//! time by the [`IntegralPixel`](crate::pixel::IntegralPixel) and
//! [`IntegralSquaredPixel`](crate::pixel::IntegralSquaredPixel) traits
//! (declared in [`crate::pixel`]). Adding two `Mono8` values into a
//! `Mono8` accumulator does not compile, because that impl deliberately
//! does not exist.
//!
//! # Closed accumulator set (0.1)
//!
//! The `(source, accumulator)` pairings shipped in this release are a
//! **closed built-in set**: the [`IntegralPixel`] /
//! [`IntegralSquaredPixel`] traits are sealed against external impls,
//! and the pre-flight capacity calculation is hard-wired to the shipped
//! accumulator types (`Mono32`, `Mono64`, `MonoF64`, and their RGB
//! analogues). User-defined accumulator pixels are **not** a supported
//! extension point in 0.1, and the `IntegralAccumulator` / capacity
//! machinery is intentionally not part of the public API surface.
//!
//! If you need an accumulator type that isn't on the shipped list,
//! please open an issue describing the camera bit depth / image size /
//! pipeline so we can add it as a first-class built-in pair. Opening
//! this up to user-defined accumulators would require a stable trait
//! contract for the worst-case capacity calculation (ADR-0032 §3), and
//! we'd rather settle that design once we have several real use cases
//! than commit to it speculatively.
//!
//! # Example
//!
//! ```
//! use irys_cv::analyze::integral::{integral_image, IntegralImage};
//! use irys_cv::image::Image;
//! use irys_cv::pixel::{Mono8, Mono32};
//! use irys_cv::{Coordinate, Rectangle, Size};
//!
//! // 4×4 image filled with the value 10.
//! let img: Image<Mono8> = Image::fill(4, 4, Mono8::new(10));
//!
//! // Accumulator pixel chosen explicitly via turbofish (ADR-0032 §1).
//! let sat: IntegralImage<Mono32> = integral_image::<_, Mono32>(&img)?;
//!
//! // Sum over a 2×3 rectangle: 2 × 3 × 10 = 60.
//! let sum = sat.region_sum(Rectangle::new(Coordinate::new(1, 0), Size::new(2, 3)));
//! assert_eq!(sum, Mono32::new(60));
//! # Ok::<(), irys_cv::Error>(())
//! ```

mod engine;
mod output;
mod preflight;

pub use engine::{
    integral_image, integral_image_into, integral_squared_image, integral_squared_image_into,
};
pub use output::IntegralImage;

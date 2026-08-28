//! Drawing primitives that burn annotations into image pixels.
//!
//! Everything in this module **mutates the image it draws on**. That is the
//! point: an inspection record with the defect location, ROI, and markers in
//! the pixels themselves is self-contained — it opens in any image viewer,
//! survives being copied around, and needs no sidecar file. The intended
//! pattern is to run algorithms on the original and draw on a clone:
//!
//! ```
//! use fovea::Size;
//! use fovea::draw::{draw_crosshair, draw_rect};
//! use fovea::image::Image;
//! use fovea::pixel::Mono8;
//!
//! let source: Image<Mono8> = Image::zero(64, 64);
//! // ... run detection on `source` ...
//!
//! let mut annotated = source.clone();
//! draw_rect(&mut annotated, (8, 8), Size::new(24, 16), Mono8::new(255), false);
//! draw_crosshair(&mut annotated, (20, 16), 5, Mono8::new(255));
//! ```
//!
//! ## Shapes and free functions
//!
//! Each shape exists twice: as a struct implementing [`Drawable`], for when
//! the shape needs to be stored, cloned, or built up before drawing, and as
//! a one-shot free function wrapping it. Neither form duplicates logic.
//!
//! | Shape | Free function | Rasterisation |
//! |---|---|---|
//! | [`Line`] | [`draw_line`] | Bresenham's line algorithm |
//! | [`Rect`] | [`draw_rect`] | Row/column spans (outline or filled) |
//! | [`Circle`] | [`draw_circle`] | Midpoint circle (outline or filled) |
//! | [`Polyline`] | [`draw_polyline`] | Sequential Bresenham segments |
//! | [`Crosshair`] | [`draw_crosshair`] | Two perpendicular axis-aligned lines |
//!
//! ## Signed coordinates, silent clipping
//!
//! Drawing positions are `(i32, i32)` — signed, unlike the unsigned
//! [`Coordinate`](crate::Coordinate) used for ROI offsets — because a shape
//! is routinely centred on a feature near the image edge and legitimately
//! extends past it. Every primitive clips: the in-bounds portion is drawn,
//! out-of-bounds pixels are skipped, no error is returned and no panic
//! occurs. This is the universal convention for 2D rasterisation.
//!
//! Clipping bounds the cost as well as the writes: the walks skip the
//! invisible portion of a shape, so a segment or circle whose ideal extent
//! is billions of pixels costs what its visible portion costs, not what
//! the ideal shape would.
//!
//! ## Crisp, single-pixel rendering
//!
//! All primitives write hard single-pixel strokes: each touched pixel is set
//! to exactly `color`, untouched pixels keep their value. There is no
//! anti-aliasing and no alpha blending — deliberate for annotation burn-in,
//! where hard edges survive JPEG compression without smearing and the only
//! requirement on the pixel type is `Copy`. Thick strokes, anti-aliased
//! variants, and text rendering are out of scope for now.
//!
//! ## Extension by addition
//!
//! Custom markers — arrows, calipers, target diamonds — are implemented by
//! writing a type with [`Drawable`], not by modifying this module. See the
//! trait documentation for an example.
//!
//! [`Drawable`]: crate::draw::Drawable
//! [`Line`]: crate::draw::Line
//! [`Rect`]: crate::draw::Rect
//! [`Circle`]: crate::draw::Circle
//! [`Polyline`]: crate::draw::Polyline
//! [`Crosshair`]: crate::draw::Crosshair
//! [`draw_line`]: crate::draw::draw_line
//! [`draw_rect`]: crate::draw::draw_rect
//! [`draw_circle`]: crate::draw::draw_circle
//! [`draw_polyline`]: crate::draw::draw_polyline
//! [`draw_crosshair`]: crate::draw::draw_crosshair

mod circle;
mod line;
mod marker;
mod polyline;
mod rect;

pub use circle::{Circle, draw_circle};
pub use line::{Line, draw_line};
pub use marker::{Crosshair, draw_crosshair};
pub use polyline::{Polyline, draw_polyline};
pub use rect::{Rect, draw_rect};

use crate::image::ImageViewMut;

/// A shape that renders itself into any mutable image of pixel type `P`.
///
/// This is the extension point of the [`draw`](self) module: the built-in
/// shapes implement it, and user-defined shapes implement it to become
/// drawable everywhere the built-ins are — no library modification needed.
///
/// # Clipping
///
/// Implementations must silently clip pixels that fall outside the image
/// bounds. Drawing a shape that is fully or partially outside the image is
/// defined behaviour — the in-bounds portion is drawn, out-of-bounds pixels
/// are skipped. No error is returned; no panic occurs.
///
/// # Minimal bounds
///
/// `P: Copy` is the tightest bound that permits writing a pixel value into
/// an image location, so any pixel type — including user-defined ones — can
/// be drawn onto.
///
/// # Object safety
///
/// This trait is intentionally not object-safe: `draw_into` takes
/// `impl ImageViewMut`, making the method generic, so a `Line<P>` drawn into
/// an owned image, a borrowed buffer, or an ROI view monomorphises to direct
/// pixel writes with no dynamic dispatch.
///
/// # Examples
///
/// A custom shape — an X marker — built from two [`Line`]s:
///
/// ```
/// use fovea::draw::{Drawable, Line};
/// use fovea::image::{Image, ImageView, ImageViewMut};
/// use fovea::pixel::Mono8;
///
/// struct XMarker {
///     center: (i32, i32),
///     arm: i32,
///     color: Mono8,
/// }
///
/// impl Drawable<Mono8> for XMarker {
///     fn draw_into(&self, image: &mut impl ImageViewMut<Pixel = Mono8>) {
///         let (cx, cy) = self.center;
///         let a = self.arm;
///         Line { from: (cx - a, cy - a), to: (cx + a, cy + a), color: self.color }
///             .draw_into(image);
///         Line { from: (cx - a, cy + a), to: (cx + a, cy - a), color: self.color }
///             .draw_into(image);
///     }
/// }
///
/// let mut image: Image<Mono8> = Image::zero(9, 9);
/// XMarker { center: (4, 4), arm: 3, color: Mono8::new(255) }.draw_into(&mut image);
/// assert_eq!(image.pixel_at(4, 4), Mono8::new(255));
/// assert_eq!(image.pixel_at(1, 1), Mono8::new(255));
/// assert_eq!(image.pixel_at(1, 7), Mono8::new(255));
/// ```
pub trait Drawable<P: Copy> {
    /// Renders this shape into `image`, clipping to the image bounds.
    fn draw_into(&self, image: &mut impl ImageViewMut<Pixel = P>);
}

/// Writes `color` at signed `(x, y)`, skipping out-of-bounds positions.
#[inline]
fn put<P: Copy>(image: &mut impl ImageViewMut<Pixel = P>, x: i64, y: i64, color: P) {
    if x >= 0 && y >= 0 {
        if let Some(pixel) = image.get_mut(x as usize, y as usize) {
            *pixel = color;
        }
    }
}

/// Writes the horizontal run from `x0` to `x1` (either order, inclusive) on
/// row `y`, clipped to the image bounds.
fn hspan<P: Copy>(image: &mut impl ImageViewMut<Pixel = P>, x0: i64, x1: i64, y: i64, color: P) {
    let size = image.size();
    if y < 0 || y >= size.height as i64 {
        return;
    }
    let lo = x0.min(x1).max(0);
    let hi = x0.max(x1).min(size.width as i64 - 1);
    for x in lo..=hi {
        *image.pixel_at_mut(x as usize, y as usize) = color;
    }
}

/// Writes the vertical run from `y0` to `y1` (either order, inclusive) in
/// column `x`, clipped to the image bounds.
fn vspan<P: Copy>(image: &mut impl ImageViewMut<Pixel = P>, x: i64, y0: i64, y1: i64, color: P) {
    let size = image.size();
    if x < 0 || x >= size.width as i64 {
        return;
    }
    let lo = y0.min(y1).max(0);
    let hi = y0.max(y1).min(size.height as i64 - 1);
    for y in lo..=hi {
        *image.pixel_at_mut(x as usize, y as usize) = color;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{Image, ImageView};
    use crate::pixel::Mono8;

    fn ink() -> Mono8 {
        Mono8::new(255)
    }

    /// Coordinates of every pixel that differs from zero, in row-major order.
    pub(super) fn inked(image: &Image<Mono8>) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        for y in 0..image.height() {
            for x in 0..image.width() {
                if image.pixel_at(x, y) != Mono8::new(0) {
                    out.push((x, y));
                }
            }
        }
        out
    }

    #[test]
    fn put_clips_all_four_sides() {
        let mut image: Image<Mono8> = Image::zero(4, 4);
        put(&mut image, -1, 2, ink());
        put(&mut image, 2, -1, ink());
        put(&mut image, 4, 2, ink());
        put(&mut image, 2, 4, ink());
        assert!(inked(&image).is_empty());
        put(&mut image, 3, 0, ink());
        assert_eq!(inked(&image), vec![(3, 0)]);
    }

    #[test]
    fn spans_accept_either_order_and_clip() {
        let mut image: Image<Mono8> = Image::zero(5, 5);
        hspan(&mut image, 3, 1, 2, ink());
        assert_eq!(inked(&image), vec![(1, 2), (2, 2), (3, 2)]);

        let mut image: Image<Mono8> = Image::zero(5, 5);
        hspan(&mut image, -10, 10, 0, ink());
        assert_eq!(inked(&image), (0..5).map(|x| (x, 0)).collect::<Vec<_>>());
        hspan(&mut image, 0, 4, -1, ink());
        hspan(&mut image, 0, 4, 5, ink());
        assert_eq!(inked(&image).len(), 5);

        let mut image: Image<Mono8> = Image::zero(5, 5);
        vspan(&mut image, 2, 10, -10, ink());
        assert_eq!(inked(&image), (0..5).map(|y| (2, y)).collect::<Vec<_>>());
        vspan(&mut image, -1, 0, 4, ink());
        vspan(&mut image, 5, 0, 4, ink());
        assert_eq!(inked(&image).len(), 5);
    }
}

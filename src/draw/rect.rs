//! Axis-aligned rectangles — outline and filled.

use super::{hspan, vspan, Drawable};
use crate::image::ImageViewMut;
use crate::Size;

/// An axis-aligned rectangle, outlined or filled.
///
/// The rectangle covers the pixels from `top_left` (inclusive) extending
/// `size.width × size.height` pixels right and down, so a `Rect` at
/// `(2, 3)` with size `4 × 2` touches columns 2–5 of rows 3–4. With
/// `fill = false` only the one-pixel border of that region is drawn; with
/// `fill = true` the whole region is. An empty `size` (either dimension
/// zero) draws nothing.
///
/// `top_left` is signed and may lie outside the image; the visible portion
/// is drawn and the rest is clipped (see [`Drawable`]).
///
/// # Examples
///
/// ```
/// use fovea::Size;
/// use fovea::draw::{Drawable, Rect};
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::Mono8;
///
/// let mut image: Image<Mono8> = Image::zero(8, 8);
/// let rect = Rect {
///     top_left: (1, 1),
///     size: Size::new(5, 4),
///     color: Mono8::new(255),
///     fill: false,
/// };
/// rect.draw_into(&mut image);
/// assert_eq!(image.pixel_at(1, 1), Mono8::new(255)); // corner
/// assert_eq!(image.pixel_at(5, 4), Mono8::new(255)); // opposite corner
/// assert_eq!(image.pixel_at(3, 2), Mono8::new(0)); // interior stays clear
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect<P> {
    /// Top-left corner of the covered region.
    pub top_left: (i32, i32),
    /// Width and height of the covered region, in pixels.
    pub size: Size,
    /// Pixel value written along the border, or over the whole region
    /// when `fill` is set.
    pub color: P,
    /// `false` draws the one-pixel border; `true` fills the region.
    pub fill: bool,
}

impl<P: Copy> Drawable<P> for Rect<P> {
    fn draw_into(&self, image: &mut impl ImageViewMut<Pixel = P>) {
        if self.size.width == 0 || self.size.height == 0 {
            return;
        }
        let (x0, y0) = (i64::from(self.top_left.0), i64::from(self.top_left.1));
        let x1 = x0 + self.size.width as i64 - 1;
        let y1 = y0 + self.size.height as i64 - 1;
        if self.fill {
            // Clip the row range up front so a mostly-off-image rectangle
            // does not iterate its invisible rows.
            let lo = y0.max(0);
            let hi = y1.min(image.size().height as i64 - 1);
            for y in lo..=hi {
                hspan(image, x0, x1, y, self.color);
            }
        } else {
            hspan(image, x0, x1, y0, self.color);
            if y1 > y0 {
                hspan(image, x0, x1, y1, self.color);
            }
            // The corner pixels already belong to the top and bottom rows;
            // with fewer than three rows there is nothing left in between.
            if y1 > y0 + 1 {
                vspan(image, x0, y0 + 1, y1 - 1, self.color);
                if x1 > x0 {
                    vspan(image, x1, y0 + 1, y1 - 1, self.color);
                }
            }
        }
    }
}

/// Draws an axis-aligned rectangle with its top-left corner at `top_left`.
///
/// One-shot wrapper over [`Rect`]; see there for the covered region and
/// clipping contract.
///
/// # Examples
///
/// ```
/// use fovea::Size;
/// use fovea::draw::draw_rect;
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::Mono8;
///
/// let mut image: Image<Mono8> = Image::zero(8, 8);
/// draw_rect(&mut image, (2, 2), Size::new(3, 3), Mono8::new(255), true);
/// assert_eq!(image.pixel_at(3, 3), Mono8::new(255)); // filled interior
/// ```
pub fn draw_rect<P: Copy>(
    image: &mut impl ImageViewMut<Pixel = P>,
    top_left: (i32, i32),
    size: Size,
    color: P,
    fill: bool,
) {
    Rect {
        top_left,
        size,
        color,
        fill,
    }
    .draw_into(image);
}

#[cfg(test)]
mod tests {
    use super::super::tests::inked;
    use super::*;
    use crate::image::Image;
    use crate::pixel::Mono8;

    fn ink() -> Mono8 {
        Mono8::new(255)
    }

    #[test]
    fn outline_touches_exactly_the_border() {
        let mut image: Image<Mono8> = Image::zero(8, 8);
        draw_rect(&mut image, (1, 2), Size::new(5, 4), ink(), false);
        let drawn = inked(&image);
        // Perimeter of a 5×4 region: 2·5 + 2·4 − 4 corners counted once.
        assert_eq!(drawn.len(), 14);
        for (x, y) in drawn {
            let inside = (1..=5).contains(&x) && (2..=5).contains(&y);
            let on_border = x == 1 || x == 5 || y == 2 || y == 5;
            assert!(inside && on_border, "({x}, {y}) not on the border");
        }
    }

    #[test]
    fn fill_covers_the_whole_region() {
        let mut image: Image<Mono8> = Image::zero(8, 8);
        draw_rect(&mut image, (1, 2), Size::new(5, 4), ink(), true);
        let drawn = inked(&image);
        assert_eq!(drawn.len(), 20);
        for x in 1..=5 {
            for y in 2..=5 {
                assert!(drawn.contains(&(x, y)), "({x}, {y}) not filled");
            }
        }
    }

    #[test]
    fn degenerate_sizes() {
        // Empty: nothing drawn.
        let mut image: Image<Mono8> = Image::zero(6, 6);
        draw_rect(&mut image, (2, 2), Size::new(0, 3), ink(), false);
        draw_rect(&mut image, (2, 2), Size::new(3, 0), ink(), true);
        assert!(inked(&image).is_empty());

        // 1×1: a single pixel, outlined or filled.
        draw_rect(&mut image, (4, 4), Size::new(1, 1), ink(), false);
        assert_eq!(inked(&image), vec![(4, 4)]);

        // 1×n: a vertical bar without double-drawn pixels.
        let mut image: Image<Mono8> = Image::zero(6, 6);
        draw_rect(&mut image, (3, 1), Size::new(1, 4), ink(), false);
        assert_eq!(inked(&image), vec![(3, 1), (3, 2), (3, 3), (3, 4)]);
    }

    #[test]
    fn clips_across_every_edge() {
        // Rectangle larger than the image: outline invisible, fill total.
        let mut image: Image<Mono8> = Image::zero(4, 4);
        draw_rect(&mut image, (-2, -2), Size::new(8, 8), ink(), false);
        assert!(inked(&image).is_empty());
        draw_rect(&mut image, (-2, -2), Size::new(8, 8), ink(), true);
        assert_eq!(inked(&image).len(), 16);

        // Corner overlap.
        let mut image: Image<Mono8> = Image::zero(4, 4);
        draw_rect(&mut image, (2, 2), Size::new(5, 5), ink(), false);
        assert_eq!(inked(&image), vec![(2, 2), (3, 2), (2, 3)]);
    }
}

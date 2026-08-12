//! Circles — midpoint algorithm, outline and filled.

use super::{hspan, put, Drawable};
use crate::image::ImageViewMut;

/// A circle around a center point, outlined or filled.
///
/// Rasterised with the midpoint circle algorithm: the outline is the closed
/// one-pixel ring closest to the ideal circle of the given radius, and the
/// filled variant covers that ring plus everything inside it — the outline
/// pixels are always a subset of the fill. A radius of `0` draws the center
/// pixel.
///
/// `center` is signed and may lie outside the image; the visible portion is
/// drawn and the rest is clipped (see [`Drawable`]).
///
/// # Examples
///
/// ```
/// use fovea::draw::{Circle, Drawable};
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::Mono8;
///
/// let mut image: Image<Mono8> = Image::zero(9, 9);
/// let circle = Circle { center: (4, 4), radius: 3, color: Mono8::new(255), fill: false };
/// circle.draw_into(&mut image);
/// assert_eq!(image.pixel_at(7, 4), Mono8::new(255)); // on the ring
/// assert_eq!(image.pixel_at(4, 4), Mono8::new(0)); // center stays clear
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Circle<P> {
    /// Center of the circle.
    pub center: (i32, i32),
    /// Radius in pixels, measured from `center` to the ring.
    pub radius: u32,
    /// Pixel value written on the ring, or over the whole disk when
    /// `fill` is set.
    pub color: P,
    /// `false` draws the one-pixel ring; `true` fills the disk.
    pub fill: bool,
}

impl<P: Copy> Drawable<P> for Circle<P> {
    fn draw_into(&self, image: &mut impl ImageViewMut<Pixel = P>) {
        let size = image.size();
        let (cx, cy) = (i64::from(self.center.0), i64::from(self.center.1));
        let r = i64::from(self.radius);
        // A circle whose bounding box misses the image has no visible pixels.
        if cx + r < 0 || cx - r >= size.width as i64 || cy + r < 0 || cy - r >= size.height as i64 {
            return;
        }
        // Midpoint walk over one octant; the seven mirrors follow by
        // symmetry. `d` tracks the sign of the implicit circle function at
        // the midpoint between the two candidate pixels.
        let mut x = r;
        let mut y = 0;
        let mut d = 1 - r;
        while y <= x {
            if self.fill {
                hspan(image, cx - x, cx + x, cy + y, self.color);
                hspan(image, cx - x, cx + x, cy - y, self.color);
                hspan(image, cx - y, cx + y, cy + x, self.color);
                hspan(image, cx - y, cx + y, cy - x, self.color);
            } else {
                for (px, py) in [
                    (cx + x, cy + y),
                    (cx + x, cy - y),
                    (cx - x, cy + y),
                    (cx - x, cy - y),
                    (cx + y, cy + x),
                    (cx + y, cy - x),
                    (cx - y, cy + x),
                    (cx - y, cy - x),
                ] {
                    put(image, px, py, self.color);
                }
            }
            y += 1;
            if d < 0 {
                d += 2 * y + 1;
            } else {
                x -= 1;
                d += 2 * (y - x) + 1;
            }
        }
    }
}

/// Draws a circle of the given `radius` around `center`.
///
/// One-shot wrapper over [`Circle`]; see there for the rasterisation and
/// clipping contract.
///
/// # Examples
///
/// ```
/// use fovea::draw::draw_circle;
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::Mono8;
///
/// let mut image: Image<Mono8> = Image::zero(9, 9);
/// draw_circle(&mut image, (4, 4), 2, Mono8::new(255), true);
/// assert_eq!(image.pixel_at(4, 4), Mono8::new(255)); // filled disk
/// ```
pub fn draw_circle<P: Copy>(
    image: &mut impl ImageViewMut<Pixel = P>,
    center: (i32, i32),
    radius: u32,
    color: P,
    fill: bool,
) {
    Circle {
        center,
        radius,
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
    fn radius_zero_is_the_center_pixel() {
        let mut image: Image<Mono8> = Image::zero(5, 5);
        draw_circle(&mut image, (2, 2), 0, ink(), false);
        assert_eq!(inked(&image), vec![(2, 2)]);
    }

    #[test]
    fn outline_is_eightfold_symmetric_and_hits_the_axes() {
        let mut image: Image<Mono8> = Image::zero(15, 15);
        draw_circle(&mut image, (7, 7), 5, ink(), false);
        let drawn = inked(&image);
        // The four axis extremes are exact.
        for extreme in [(12, 7), (2, 7), (7, 12), (7, 2)] {
            assert!(drawn.contains(&extreme), "missing {extreme:?}");
        }
        // Every drawn pixel has its eight symmetric mirrors drawn too.
        for &(x, y) in &drawn {
            let (dx, dy) = (x as i32 - 7, y as i32 - 7);
            for (mx, my) in [
                (dx, dy),
                (dx, -dy),
                (-dx, dy),
                (-dx, -dy),
                (dy, dx),
                (dy, -dx),
                (-dy, dx),
                (-dy, -dx),
            ] {
                let mirror = ((7 + mx) as usize, (7 + my) as usize);
                assert!(drawn.contains(&mirror), "missing mirror {mirror:?}");
            }
        }
        // Ring distance: every pixel is within half a pixel of the radius.
        for &(x, y) in &drawn {
            let dist = ((x as f64 - 7.0).powi(2) + (y as f64 - 7.0).powi(2)).sqrt();
            assert!((dist - 5.0).abs() <= 0.5, "({x}, {y}) at distance {dist}");
        }
    }

    #[test]
    fn fill_contains_the_outline_and_the_interior() {
        let mut outlined: Image<Mono8> = Image::zero(15, 15);
        draw_circle(&mut outlined, (7, 7), 5, ink(), false);
        let mut filled: Image<Mono8> = Image::zero(15, 15);
        draw_circle(&mut filled, (7, 7), 5, ink(), true);

        let ring = inked(&outlined);
        let disk = inked(&filled);
        for pixel in &ring {
            assert!(disk.contains(pixel), "outline pixel {pixel:?} not filled");
        }
        assert!(disk.contains(&(7, 7)));
        assert!(disk.len() > ring.len());
    }

    #[test]
    fn clips_when_the_center_is_off_image() {
        let mut image: Image<Mono8> = Image::zero(6, 6);
        draw_circle(&mut image, (-2, 3), 4, ink(), false);
        let drawn = inked(&image);
        assert!(!drawn.is_empty());
        for (x, y) in drawn {
            let dist = ((x as f64 + 2.0).powi(2) + (y as f64 - 3.0).powi(2)).sqrt();
            assert!((dist - 4.0).abs() <= 0.5, "({x}, {y}) at distance {dist}");
        }
    }

    #[test]
    fn fully_outside_draws_nothing() {
        let mut image: Image<Mono8> = Image::zero(6, 6);
        draw_circle(&mut image, (-10, 3), 4, ink(), true);
        draw_circle(&mut image, (3, 20), 4, ink(), false);
        assert!(inked(&image).is_empty());
    }
}

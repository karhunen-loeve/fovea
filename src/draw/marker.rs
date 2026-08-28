//! Point markers — the crosshair.

use super::{Drawable, hspan, vspan};
use crate::image::ImageViewMut;

/// A `+`-shaped marker centred on a point.
///
/// Two perpendicular axis-aligned lines through `center`, each extending
/// `arm_length` pixels in both directions — so each stroke is
/// `2 · arm_length + 1` pixels long and the marker touches
/// `4 · arm_length + 1` pixels. An `arm_length` of `0` draws the center
/// pixel. This is the conventional marker for keypoints and measurement
/// positions: unlike a filled dot it stays visually locatable to the exact
/// pixel.
///
/// `center` is signed and may lie outside the image; the visible portion is
/// drawn and the rest is clipped (see [`Drawable`]).
///
/// # Examples
///
/// ```
/// use fovea::draw::{Crosshair, Drawable};
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::Mono8;
///
/// let mut image: Image<Mono8> = Image::zero(9, 9);
/// let marker = Crosshair { center: (4, 4), arm_length: 3, color: Mono8::new(255) };
/// marker.draw_into(&mut image);
/// assert_eq!(image.pixel_at(4, 4), Mono8::new(255)); // center
/// assert_eq!(image.pixel_at(1, 4), Mono8::new(255)); // arm tip
/// assert_eq!(image.pixel_at(3, 3), Mono8::new(0)); // off both arms
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Crosshair<P> {
    /// Center of the marker, drawn.
    pub center: (i32, i32),
    /// Pixels each arm extends from the center, in all four directions.
    pub arm_length: u32,
    /// Pixel value written along both strokes.
    pub color: P,
}

impl<P: Copy> Drawable<P> for Crosshair<P> {
    fn draw_into(&self, image: &mut impl ImageViewMut<Pixel = P>) {
        let (cx, cy) = (i64::from(self.center.0), i64::from(self.center.1));
        let arm = i64::from(self.arm_length);
        hspan(image, cx - arm, cx + arm, cy, self.color);
        vspan(image, cx, cy - arm, cy + arm, self.color);
    }
}

/// Draws a `+`-shaped crosshair marker centred on `center`.
///
/// One-shot wrapper over [`Crosshair`]; see there for the geometry and
/// clipping contract.
///
/// # Examples
///
/// ```
/// use fovea::draw::draw_crosshair;
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::Mono8;
///
/// let mut image: Image<Mono8> = Image::zero(9, 9);
/// draw_crosshair(&mut image, (4, 4), 2, Mono8::new(255));
/// assert_eq!(image.pixel_at(4, 2), Mono8::new(255));
/// ```
pub fn draw_crosshair<P: Copy>(
    image: &mut impl ImageViewMut<Pixel = P>,
    center: (i32, i32),
    arm_length: u32,
    color: P,
) {
    Crosshair {
        center,
        arm_length,
        color,
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
    fn touches_exactly_the_two_arms() {
        let mut image: Image<Mono8> = Image::zero(9, 9);
        draw_crosshair(&mut image, (4, 4), 3, ink());
        let drawn = inked(&image);
        assert_eq!(drawn.len(), 13); // 4 · 3 + 1
        for (x, y) in drawn {
            assert!(
                (y == 4 && (1..=7).contains(&x)) || (x == 4 && (1..=7).contains(&y)),
                "({x}, {y}) off both arms"
            );
        }
    }

    #[test]
    fn arm_length_zero_is_the_center_pixel() {
        let mut image: Image<Mono8> = Image::zero(5, 5);
        draw_crosshair(&mut image, (2, 2), 0, ink());
        assert_eq!(inked(&image), vec![(2, 2)]);
    }

    #[test]
    fn clips_at_the_image_corner() {
        let mut image: Image<Mono8> = Image::zero(5, 5);
        draw_crosshair(&mut image, (0, 0), 2, ink());
        assert_eq!(inked(&image), vec![(0, 0), (1, 0), (2, 0), (0, 1), (0, 2)]);
    }

    #[test]
    fn off_image_center_shows_one_arm() {
        // Center above the image: only the vertical arm's lower part shows.
        let mut image: Image<Mono8> = Image::zero(5, 5);
        draw_crosshair(&mut image, (2, -2), 3, ink());
        assert_eq!(inked(&image), vec![(2, 0), (2, 1)]);
    }
}

//! Straight line segments — Bresenham's algorithm.

use super::{put, Drawable};
use crate::image::ImageViewMut;

/// A straight line segment between two points, drawn one pixel wide.
///
/// Rasterised with Bresenham's algorithm: both endpoints are drawn, every
/// step moves to the pixel closest to the ideal line, and the stroke is
/// exactly one pixel per row (or per column, whichever axis is major). A
/// zero-length line (`from == to`) draws that single pixel.
///
/// Both endpoints are signed and may lie outside the image; the visible
/// portion is drawn and the rest is clipped (see [`Drawable`]).
///
/// # Examples
///
/// ```
/// use fovea::draw::{Drawable, Line};
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::Mono8;
///
/// let mut image: Image<Mono8> = Image::zero(8, 8);
/// Line { from: (0, 0), to: (7, 7), color: Mono8::new(255) }.draw_into(&mut image);
/// assert_eq!(image.pixel_at(3, 3), Mono8::new(255));
/// assert_eq!(image.pixel_at(3, 4), Mono8::new(0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Line<P> {
    /// First endpoint, drawn.
    pub from: (i32, i32),
    /// Second endpoint, drawn.
    pub to: (i32, i32),
    /// Pixel value written along the stroke.
    pub color: P,
}

impl<P: Copy> Drawable<P> for Line<P> {
    fn draw_into(&self, image: &mut impl ImageViewMut<Pixel = P>) {
        segment(image, self.from, self.to, self.color);
    }
}

/// Draws a one-pixel-wide line segment from `from` to `to`.
///
/// One-shot wrapper over [`Line`]; see there for the rasterisation and
/// clipping contract.
///
/// # Examples
///
/// ```
/// use fovea::draw::draw_line;
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::Mono8;
///
/// let mut image: Image<Mono8> = Image::zero(8, 8);
/// draw_line(&mut image, (1, 6), (6, 1), Mono8::new(255));
/// assert_eq!(image.pixel_at(1, 6), Mono8::new(255));
/// assert_eq!(image.pixel_at(6, 1), Mono8::new(255));
/// ```
pub fn draw_line<P: Copy>(
    image: &mut impl ImageViewMut<Pixel = P>,
    from: (i32, i32),
    to: (i32, i32),
    color: P,
) {
    Line { from, to, color }.draw_into(image);
}

/// Bresenham walk shared by [`Line`], [`Polyline`](super::Polyline), and the
/// free functions. Widens to `i64` so the error terms cannot overflow for any
/// `i32` endpoints.
pub(super) fn segment<P: Copy>(
    image: &mut impl ImageViewMut<Pixel = P>,
    from: (i32, i32),
    to: (i32, i32),
    color: P,
) {
    let size = image.size();
    let (mut x, mut y) = (i64::from(from.0), i64::from(from.1));
    let (x1, y1) = (i64::from(to.0), i64::from(to.1));
    // A segment whose bounding box misses the image has no visible pixels;
    // skip the walk entirely instead of clipping it pixel by pixel.
    if x.max(x1) < 0
        || x.min(x1) >= size.width as i64
        || y.max(y1) < 0
        || y.min(y1) >= size.height as i64
    {
        return;
    }
    let dx = (x1 - x).abs();
    let dy = -(y1 - y).abs();
    let sx = if x < x1 { 1 } else { -1 };
    let sy = if y < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    loop {
        put(image, x, y, color);
        if x == x1 && y == y1 {
            return;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
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
    fn horizontal_vertical_and_point() {
        let mut image: Image<Mono8> = Image::zero(6, 6);
        draw_line(&mut image, (1, 2), (4, 2), ink());
        assert_eq!(inked(&image), vec![(1, 2), (2, 2), (3, 2), (4, 2)]);

        let mut image: Image<Mono8> = Image::zero(6, 6);
        draw_line(&mut image, (3, 4), (3, 1), ink());
        assert_eq!(inked(&image), vec![(3, 1), (3, 2), (3, 3), (3, 4)]);

        let mut image: Image<Mono8> = Image::zero(6, 6);
        draw_line(&mut image, (2, 5), (2, 5), ink());
        assert_eq!(inked(&image), vec![(2, 5)]);
    }

    #[test]
    fn perfect_diagonals_in_all_four_directions() {
        for (from, to) in [
            ((0, 0), (5, 5)),
            ((5, 5), (0, 0)),
            ((0, 5), (5, 0)),
            ((5, 0), (0, 5)),
        ] {
            let mut image: Image<Mono8> = Image::zero(6, 6);
            draw_line(&mut image, from, to, ink());
            let drawn = inked(&image);
            assert_eq!(drawn.len(), 6, "{from:?} -> {to:?}: {drawn:?}");
            for (x, y) in drawn {
                let on_main = x == y;
                let on_anti = x + y == 5;
                assert!(on_main || on_anti, "({x}, {y}) off both diagonals");
            }
        }
    }

    #[test]
    fn shallow_slope_is_one_pixel_per_column() {
        let mut image: Image<Mono8> = Image::zero(10, 4);
        draw_line(&mut image, (0, 0), (9, 3), ink());
        let drawn = inked(&image);
        assert_eq!(drawn.len(), 10);
        let mut columns: Vec<usize> = drawn.iter().map(|&(x, _)| x).collect();
        columns.sort_unstable();
        assert_eq!(columns, (0..10).collect::<Vec<_>>());
        // Monotone: y never decreases as x grows.
        let mut ys: Vec<usize> = (0..10)
            .map(|x| drawn.iter().find(|&&(px, _)| px == x).unwrap().1)
            .collect();
        let sorted = ys.clone();
        ys.sort_unstable();
        assert_eq!(ys, sorted);
    }

    #[test]
    fn steep_slope_is_one_pixel_per_row() {
        let mut image: Image<Mono8> = Image::zero(4, 10);
        draw_line(&mut image, (0, 0), (3, 9), ink());
        let drawn = inked(&image);
        assert_eq!(drawn.len(), 10);
        let rows: Vec<usize> = drawn.iter().map(|&(_, y)| y).collect();
        assert_eq!(rows, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn clips_a_partially_visible_line() {
        let mut image: Image<Mono8> = Image::zero(4, 4);
        draw_line(&mut image, (-3, 1), (7, 1), ink());
        assert_eq!(inked(&image), vec![(0, 1), (1, 1), (2, 1), (3, 1)]);
    }

    #[test]
    fn fully_outside_draws_nothing() {
        let mut image: Image<Mono8> = Image::zero(4, 4);
        draw_line(&mut image, (-5, -5), (-1, -2), ink());
        draw_line(&mut image, (4, 0), (9, 3), ink());
        draw_line(&mut image, (0, 4), (3, 9), ink());
        draw_line(&mut image, (i32::MIN, i32::MIN), (-1, i32::MAX), ink());
        assert!(inked(&image).is_empty());
    }

    #[test]
    fn extreme_endpoints_do_not_overflow() {
        // Bounding box straddles the image, so the early-out does not fire;
        // the arithmetic must still be sound. Keep the walk short by making
        // the segment cross near the origin.
        let mut image: Image<Mono8> = Image::zero(4, 4);
        draw_line(&mut image, (-2, -2), (5, 5), ink());
        assert_eq!(inked(&image), vec![(0, 0), (1, 1), (2, 2), (3, 3)]);
    }
}

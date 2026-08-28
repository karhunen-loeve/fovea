//! Straight line segments — Bresenham's algorithm.

use super::{Drawable, put};
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
///
/// The walk is clipped to the iterations that can touch the frame, so the
/// cost is proportional to the visible portion, not to the ideal segment:
/// endpoints a million pixels off-image cost the same as endpoints one pixel
/// off. The clip fast-forwards the exact walk state (closed forms for the
/// minor-axis step count and the error term), so the painted pixels are
/// identical to those of the unclipped walk; `clipping_matches_the_unclipped_
/// walk_exactly` pins that equivalence against a reference walk.
pub(super) fn segment<P: Copy>(
    image: &mut impl ImageViewMut<Pixel = P>,
    from: (i32, i32),
    to: (i32, i32),
    color: P,
) {
    let size = image.size();
    let (w, h) = (size.width as i64, size.height as i64);
    let (x0, y0) = (i64::from(from.0), i64::from(from.1));
    let (x1, y1) = (i64::from(to.0), i64::from(to.1));
    // A segment whose bounding box misses the image has no visible pixels;
    // skip the walk entirely instead of clipping it pixel by pixel.
    if x0.max(x1) < 0 || x0.min(x1) >= w || y0.max(y1) < 0 || y0.min(y1) >= h {
        return;
    }
    let a = (x1 - x0).abs();
    let b = (y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };

    // Iteration `t` of the walk paints the pixel whose major coordinate is
    // `t` steps from `from`; the minor coordinate has advanced `minor(t)`
    // steps, where (derived from the error recurrence, and pinned by test)
    //
    //     minor(t) = max(0, floor((2*n*t - m) / (2*m)) + 1)
    //
    // with `m` the major and `n` the minor extent. Restrict the walk to the
    // iterations whose pixel lies inside the frame on both axes; each axis
    // contributes one contiguous `t` interval because the walk is monotone.
    let steps = a.max(b);
    let (m, n) = (a.max(b), a.min(b));
    let x_is_major = a >= b;

    // Major axis: the position is `major0 + s_major * t`.
    let (major0, s_major, major_len) = if x_is_major { (x0, sx, w) } else { (y0, sy, h) };
    let (t_major_lo, t_major_hi) = if s_major == 1 {
        (-major0, major_len - 1 - major0)
    } else {
        (major0 - (major_len - 1), major0)
    };

    // Minor axis: the position is `minor0 + s_minor * minor(t)` with
    // `minor(t)` in `0..=n`; invert the closed form to a `t` interval.
    let (minor0, s_minor, minor_len) = if x_is_major { (y0, sy, h) } else { (x0, sx, w) };
    let (t_minor_lo, t_minor_hi) = if n == 0 {
        // The minor coordinate never moves, and the bounding-box test above
        // already guarantees it is inside the frame.
        (0, steps)
    } else {
        let (k_lo, k_hi) = if s_minor == 1 {
            (-minor0, minor_len - 1 - minor0)
        } else {
            (minor0 - (minor_len - 1), minor0)
        };
        let (k_lo, k_hi) = (k_lo.max(0), k_hi.min(n));
        if k_lo > k_hi {
            return;
        }
        // Smallest t with minor(t) >= k is ceil(m*(2k - 1) / (2n)); largest
        // t with minor(t) <= k is ceil(m*(2k + 1) / (2n)) - 1. The products
        // reach 2^65 for i32 endpoints, hence the i128 arithmetic.
        let ceil_div = |p: i128, q: i128| ((p + q - 1) / q) as i64;
        let lo = if k_lo <= 0 {
            0
        } else {
            ceil_div((m as i128) * (2 * k_lo as i128 - 1), 2 * n as i128)
        };
        let hi = if k_hi >= n {
            steps
        } else {
            ceil_div((m as i128) * (2 * k_hi as i128 + 1), 2 * n as i128) - 1
        };
        (lo, hi)
    };

    let t_lo = t_major_lo.max(t_minor_lo).max(0);
    let t_hi = t_major_hi.min(t_minor_hi).min(steps);
    if t_lo > t_hi {
        return;
    }

    // Fast-forward the walk state to iteration `t_lo` in closed form. The
    // error term there is `(a - b) - t*n + m*minor(t)` up to the axis swap.
    let minor_at = |t: i64| -> i64 {
        if n == 0 || t == 0 {
            0
        } else {
            (((2 * n as i128 * t as i128 - m as i128).div_euclid(2 * m as i128)) as i64 + 1).max(0)
        }
    };
    let k = minor_at(t_lo);
    let mut err = ((a - b) as i128
        + if x_is_major {
            m as i128 * k as i128 - n as i128 * t_lo as i128
        } else {
            n as i128 * t_lo as i128 - m as i128 * k as i128
        }) as i64;
    let (mut x, mut y) = if x_is_major {
        (x0 + sx * t_lo, y0 + sy * k)
    } else {
        (x0 + sx * k, y0 + sy * t_lo)
    };

    let dx = a;
    let dy = -b;
    for _ in t_lo..=t_hi {
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

    /// The unclipped walk, kept as the behavioural reference for the clip.
    fn reference_segment(image: &mut Image<Mono8>, from: (i32, i32), to: (i32, i32), color: Mono8) {
        use crate::image::ImageView;
        let (mut x, mut y) = (i64::from(from.0), i64::from(from.1));
        let (x1, y1) = (i64::from(to.0), i64::from(to.1));
        let dx = (x1 - x).abs();
        let dy = -(y1 - y).abs();
        let sx = if x < x1 { 1 } else { -1 };
        let sy = if y < y1 { 1 } else { -1 };
        let mut err = dx + dy;
        loop {
            if x >= 0 && y >= 0 && x < image.width() as i64 && y < image.height() as i64 {
                *image.pixel_at_mut(x as usize, y as usize) = color;
            }
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

    #[test]
    fn clipping_matches_the_unclipped_walk_exactly() {
        // The clip fast-forwards the exact walk state, so it must not move
        // a single pixel relative to the unclipped walk: every endpoint
        // pair around and across a 5x4 frame, in both directions.
        let coords: Vec<(i32, i32)> = (-6..=9)
            .flat_map(|x| (-6..=9).map(move |y| (x, y)))
            .collect();
        for &from in &coords {
            for &to in &coords {
                let mut clipped: Image<Mono8> = Image::zero(5, 4);
                draw_line(&mut clipped, from, to, ink());
                let mut reference: Image<Mono8> = Image::zero(5, 4);
                reference_segment(&mut reference, from, to, ink());
                assert_eq!(
                    inked(&clipped),
                    inked(&reference),
                    "{from:?} -> {to:?} diverged from the unclipped walk"
                );
            }
        }
    }

    #[test]
    fn far_off_image_endpoints_cost_only_the_visible_span() {
        // The walk is clipped to the frame, so a segment whose ideal length
        // is the whole i32 range completes immediately instead of stepping
        // pixel by pixel through billions of invisible positions. A hang
        // here is the regression this test pins.
        let mut image: Image<Mono8> = Image::zero(4, 4);
        draw_line(&mut image, (i32::MIN, 0), (i32::MAX, 0), ink());
        assert_eq!(inked(&image), vec![(0, 0), (1, 0), (2, 0), (3, 0)]);

        let mut image: Image<Mono8> = Image::zero(4, 4);
        draw_line(
            &mut image,
            (-10_000_000, -10_000_000),
            (10_000_000, 10_000_000),
            ink(),
        );
        assert_eq!(inked(&image), vec![(0, 0), (1, 1), (2, 2), (3, 3)]);

        // Steep counterpart, and a shallow segment whose bounding box
        // straddles the frame although the segment itself crosses the
        // frame's rows a billion pixels to the left of it.
        let mut image: Image<Mono8> = Image::zero(4, 4);
        draw_line(&mut image, (1, i32::MIN), (2, i32::MAX), ink());
        assert_eq!(inked(&image).len(), 4);

        let mut image: Image<Mono8> = Image::zero(4, 4);
        draw_line(&mut image, (i32::MIN, -20), (0, 20), ink());
        assert!(inked(&image).is_empty());
    }
}

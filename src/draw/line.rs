//! Straight line segments — Bresenham's algorithm.

use super::{Drawable, put};
use crate::SignedCoordinate;
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
/// Line { from: (0, 0).into(), to: (7, 7).into(), color: Mono8::new(255) }.draw_into(&mut image);
/// assert_eq!(image.pixel_at(3, 3), Mono8::new(255));
/// assert_eq!(image.pixel_at(3, 4), Mono8::new(0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Line<P> {
    /// First endpoint, drawn.
    pub from: SignedCoordinate,
    /// Second endpoint, drawn.
    pub to: SignedCoordinate,
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
    from: impl Into<SignedCoordinate>,
    to: impl Into<SignedCoordinate>,
    color: P,
) {
    Line {
        from: from.into(),
        to: to.into(),
        color,
    }
    .draw_into(image);
}

/// Bresenham walk shared by [`Line`], [`Polyline`](super::Polyline), and the
/// free functions. Widens to `i128`, so the extents of any two `isize`
/// endpoints (up to 2^64) and every walk quantity fit.
///
/// The walk is clipped to the iterations that can touch the frame, so the
/// cost is proportional to the visible portion, not to the ideal segment:
/// endpoints a million pixels off-image cost the same as endpoints one pixel
/// off. The clip fast-forwards the exact walk state (closed forms for the
/// minor-axis step count and the error term), so the painted pixels are
/// identical to those of the unclipped walk; `clipping_matches_the_unclipped_
/// walk_exactly` pins that equivalence against a reference walk, and
/// `extreme_endpoints_paint_what_moderate_ones_do` carries it to the ends of
/// the `isize` range.
pub(super) fn segment<P: Copy>(
    image: &mut impl ImageViewMut<Pixel = P>,
    from: SignedCoordinate,
    to: SignedCoordinate,
    color: P,
) {
    let size = image.size();
    let (w, h) = (size.width as i128, size.height as i128);
    let (x0, y0) = (from.x as i128, from.y as i128);
    let (x1, y1) = (to.x as i128, to.y as i128);
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
        // t with minor(t) <= k is ceil(m*(2k + 1) / (2n)) - 1. With m and k
        // up to 2^64 the products reach 2^129, past i128, so they go
        // through the exact 256-bit `mul_div_ceil`. All operands are
        // non-negative here, and each quotient is at most m.
        let lo = if k_lo <= 0 {
            0
        } else {
            mul_div_ceil(m as u128, (2 * k_lo - 1) as u128, (2 * n) as u128) as i128
        };
        let hi = if k_hi >= n {
            steps
        } else {
            mul_div_ceil(m as u128, (2 * k_hi + 1) as u128, (2 * n) as u128) as i128 - 1
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
    let minor_at = |t: i128| -> i128 {
        if n == 0 || t == 0 {
            return 0;
        }
        // max(0, floor((2*n*t - m) / (2*m)) + 1). The product 2*n*t reaches
        // 2^129, so it is formed in 256 bits. If it is below m the floor is
        // -1 and the result 0; otherwise the quotient is at most n.
        let (hi, lo) = mul_wide((2 * n) as u128, t as u128);
        let (lo, borrow) = lo.overflowing_sub(m as u128);
        if hi < u128::from(borrow) {
            return 0;
        }
        let hi = hi - u128::from(borrow);
        div_wide(hi, lo, (2 * m) as u128).0 as i128 + 1
    };
    let k = minor_at(t_lo);
    // The two products reach 2^128, but the error term itself is bounded by
    // the extents (|err| <= 2^64), so arithmetic modulo 2^128 is exact: the
    // wrapping operations give the true value because it fits an i128.
    let mut err = (a - b).wrapping_add(if x_is_major {
        m.wrapping_mul(k).wrapping_sub(n.wrapping_mul(t_lo))
    } else {
        n.wrapping_mul(t_lo).wrapping_sub(m.wrapping_mul(k))
    });
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

/// The full 256-bit product of two `u128`, as `(high, low)` halves.
fn mul_wide(a: u128, b: u128) -> (u128, u128) {
    const LOW: u128 = u64::MAX as u128;
    let (a_hi, a_lo) = (a >> 64, a & LOW);
    let (b_hi, b_lo) = (b >> 64, b & LOW);
    let ll = a_lo * b_lo;
    let lh = a_lo * b_hi;
    let hl = a_hi * b_lo;
    let hh = a_hi * b_hi;
    // At most three 64-bit terms, so the middle column cannot overflow.
    let mid = (ll >> 64) + (lh & LOW) + (hl & LOW);
    let low = (ll & LOW) | ((mid & LOW) << 64);
    let high = hh + (lh >> 64) + (hl >> 64) + (mid >> 64);
    (high, low)
}

/// `(high, low) / d` as `(quotient, remainder)`, for a quotient that fits a
/// `u128`, which is exactly the condition `high < d`.
fn div_wide(high: u128, low: u128, d: u128) -> (u128, u128) {
    debug_assert!(d != 0 && high < d, "the quotient must fit a u128");
    // Restoring long division, one bit of `low` per step. The remainder
    // stays below `d`, so after the shift it is below 2*d; a carry out of
    // bit 127 means it is at least 2^128 > d, and the wrapping subtraction
    // then yields the true, smaller difference.
    let mut rem = high;
    let mut quotient = 0u128;
    for bit in (0..128).rev() {
        let carry = rem >> 127;
        rem = (rem << 1) | ((low >> bit) & 1);
        quotient <<= 1;
        if carry == 1 || rem >= d {
            rem = rem.wrapping_sub(d);
            quotient |= 1;
        }
    }
    (quotient, rem)
}

/// `ceil(a * b / d)` without overflow in the product, for a result that
/// fits a `u128`.
fn mul_div_ceil(a: u128, b: u128, d: u128) -> u128 {
    let (high, low) = mul_wide(a, b);
    let (quotient, rem) = div_wide(high, low, d);
    quotient + u128::from(rem != 0)
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
        draw_line(
            &mut image,
            (isize::MIN, isize::MIN),
            (-1, isize::MAX),
            ink(),
        );
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
    fn reference_segment(
        image: &mut Image<Mono8>,
        from: (isize, isize),
        to: (isize, isize),
        color: Mono8,
    ) {
        use crate::image::ImageView;
        let (mut x, mut y) = (from.0 as i128, from.1 as i128);
        let (x1, y1) = (to.0 as i128, to.1 as i128);
        let dx = (x1 - x).abs();
        let dy = -(y1 - y).abs();
        let sx = if x < x1 { 1 } else { -1 };
        let sy = if y < y1 { 1 } else { -1 };
        let mut err = dx + dy;
        loop {
            if x >= 0 && y >= 0 && x < image.width() as i128 && y < image.height() as i128 {
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
        let coords: Vec<(isize, isize)> = (-6..=9)
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
        // is the whole isize range completes immediately instead of stepping
        // pixel by pixel through quintillions of invisible positions. A hang
        // here is the regression this test pins.
        let mut image: Image<Mono8> = Image::zero(4, 4);
        draw_line(&mut image, (isize::MIN, 0), (isize::MAX, 0), ink());
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
        // frame's rows far to the left of it.
        let mut image: Image<Mono8> = Image::zero(4, 4);
        draw_line(&mut image, (1, isize::MIN), (2, isize::MAX), ink());
        assert_eq!(inked(&image).len(), 4);

        let mut image: Image<Mono8> = Image::zero(4, 4);
        draw_line(&mut image, (isize::MIN, -20), (0, 20), ink());
        assert!(inked(&image).is_empty());

        // The full diagonal of the isize plane.
        let mut image: Image<Mono8> = Image::zero(4, 4);
        draw_line(
            &mut image,
            (isize::MIN, isize::MIN),
            (isize::MAX, isize::MAX),
            ink(),
        );
        assert_eq!(inked(&image), vec![(0, 0), (1, 1), (2, 2), (3, 3)]);
    }

    #[test]
    fn extreme_endpoints_paint_what_moderate_ones_do() {
        // A segment from `c - K*(p, q)` to `c + K*(p, q)` crosses the frame
        // at `c`, and there its minor-axis step count is
        // `q*K + floor((2*q*j - p) / (2*p)) + 1` at major offset `j`: the
        // `K` terms cancel, so every K large enough to span the frame
        // paints the same pixels. K = 1000 is checked against the unclipped
        // reference walk; K near isize::MAX / 5 drives the extents to 2^64
        // and the clip's products to 2^129, the range `mul_div_ceil`,
        // `mul_wide` and the modular error term exist for.
        // Headroom for the centre offset: c + K*p must not overflow.
        let huge = (isize::MAX - 8) / 5;
        for (cx, cy) in [(2isize, 1isize), (0, 3), (4, 0)] {
            for p in -5isize..=5 {
                for q in -5isize..=5 {
                    if p == 0 && q == 0 {
                        continue;
                    }
                    let seg = |k: isize| ((cx - k * p, cy - k * q), (cx + k * p, cy + k * q));
                    let (from, to) = seg(1000);
                    let mut moderate: Image<Mono8> = Image::zero(5, 4);
                    draw_line(&mut moderate, from, to, ink());
                    let mut reference: Image<Mono8> = Image::zero(5, 4);
                    reference_segment(&mut reference, from, to, ink());
                    assert_eq!(inked(&moderate), inked(&reference), "p {p}, q {q}");

                    let (from, to) = seg(huge);
                    let mut extreme: Image<Mono8> = Image::zero(5, 4);
                    draw_line(&mut extreme, from, to, ink());
                    assert_eq!(
                        inked(&extreme),
                        inked(&moderate),
                        "p {p}, q {q}, centre ({cx}, {cy}): extreme endpoints diverged"
                    );
                }
            }
        }
    }

    #[test]
    fn the_wide_arithmetic_is_exact() {
        // Against native u128 where the product fits.
        for &(a, b, d) in &[
            (0u128, 5, 3),
            (7, 9, 4),
            (1 << 60, 1 << 60, 3),
            (12345, 67890, 7),
        ] {
            let (hi, lo) = mul_wide(a, b);
            assert_eq!((hi, lo), (0, a * b));
            assert_eq!(div_wide(hi, lo, d), (a * b / d, a * b % d));
            assert_eq!(mul_div_ceil(a, b, d), (a * b).div_ceil(d));
        }
        // Past it: (2^128 - 1)^2 = 2^256 - 2^129 + 1.
        assert_eq!(mul_wide(u128::MAX, u128::MAX), (u128::MAX - 1, 1));
        // 2^64 * 2^65 / 2^66 = 2^63, exactly, with no remainder.
        let (hi, lo) = mul_wide(1 << 64, 1 << 65);
        assert_eq!((hi, lo), (2, 0));
        assert_eq!(div_wide(hi, lo, 1 << 66), (1 << 63, 0));
        // (2^127 + 1) * 3 / 2 = 3 * 2^126 + 1.5, which rounds up.
        assert_eq!(mul_div_ceil((1 << 127) + 1, 3, 2), 3 * (1 << 126) + 2);
        // Division by a divisor above 2^127 exercises the carry branch.
        let d = u128::MAX - 6;
        let (hi, lo) = mul_wide(d - 1, 5);
        let (q, r) = div_wide(hi, lo, d);
        assert_eq!((q, r), (4, d - 5));
    }
}

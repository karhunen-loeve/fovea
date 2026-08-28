//! Circles — midpoint algorithm, outline and filled.

use super::{Drawable, hspan, put};
use crate::CoordinateI32;
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
/// let circle = Circle { center: (4, 4).into(), radius: 3, color: Mono8::new(255), fill: false };
/// circle.draw_into(&mut image);
/// assert_eq!(image.pixel_at(7, 4), Mono8::new(255)); // on the ring
/// assert_eq!(image.pixel_at(4, 4), Mono8::new(0)); // center stays clear
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Circle<P> {
    /// Center of the circle.
    pub center: CoordinateI32,
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
        let (w, h) = (size.width as i64, size.height as i64);
        let (cx, cy) = (i64::from(self.center.x), i64::from(self.center.y));
        let r = i64::from(self.radius);
        // A circle whose bounding box misses the image has no visible pixels.
        if cx + r < 0 || cx - r >= w || cy + r < 0 || cy - r >= h {
            return;
        }

        // The midpoint walk runs the octant parameter `y` from 0 towards
        // r/sqrt(2), which for a huge radius is billions of iterations even
        // when only a handful of pixels are visible. Restrict the walk to
        // the `y` intervals whose mirrors can touch the frame: a mirror of
        // the form (cx ± x, cy ± y) needs `y` to be a visible row offset
        // and `x(y)` a visible column offset, the (cx ± y, cy ± x) form
        // needs the transpose, and the fill spans need their row visible
        // and their half-width to reach the columns. Each constraint on
        // `x(y)` converts to a `y` interval because `x(y)` is
        // nonincreasing. The intervals carry one step of slack; `put` and
        // `hspan` still clip per pixel, so the slack costs a few
        // iterations, never correctness.
        let row_windows = offset_windows(cy, h);
        let col_windows = offset_windows(cx, w);
        // Smallest half-width at which a span `cx - o ..= cx + o` reaches
        // the frame's columns.
        let span_reach = (-cx).max(cx - (w - 1)).max(0);

        let mut intervals = [(0i64, -1i64); 8];
        let mut count = 0;
        let mut push = |lo: i64, hi: i64| {
            let (lo, hi) = (lo.max(0), hi.min(r));
            if lo <= hi {
                intervals[count] = (lo, hi);
                count += 1;
            }
        };
        if self.fill {
            for &rows in &row_windows {
                // Spans on rows cy ± y: the half-width x(y) must reach the
                // columns, which bounds y from above.
                if let Some((_, y_hi)) = walk_interval_for_x(r, span_reach, r) {
                    push(rows.0, rows.1.min(y_hi));
                }
                // Spans on rows cy ± x: the half-width y must reach the
                // columns, which bounds y from below.
                if let Some((y_lo, y_hi)) = walk_interval_for_x(r, rows.0, rows.1) {
                    push(y_lo.max(span_reach), y_hi);
                }
            }
        } else {
            for &rows in &row_windows {
                for &cols in &col_windows {
                    // Mirrors (cx ± x, cy ± y).
                    if let Some((y_lo, y_hi)) = walk_interval_for_x(r, cols.0, cols.1) {
                        push(rows.0.max(y_lo), rows.1.min(y_hi));
                    }
                    // Mirrors (cx ± y, cy ± x).
                    if let Some((y_lo, y_hi)) = walk_interval_for_x(r, rows.0, rows.1) {
                        push(cols.0.max(y_lo), cols.1.min(y_hi));
                    }
                }
            }
        }
        let intervals = &mut intervals[..count];
        intervals.sort_unstable();

        // Walk each merged interval. `d` tracks the sign of the implicit
        // circle function at the midpoint between the two candidate pixels;
        // at the top of every iteration `d = (y+1)^2 + x^2 - x - r^2`, and
        // `x(y)` is the smallest x with `x*(x+1) >= r^2 - y^2`, so the walk
        // state at any `y` is available in closed form and skipping the
        // invisible iterations paints exactly the pixels the full walk
        // would (pinned by `clipping_matches_the_unclipped_walk_exactly`).
        let mut resume = 0;
        for &(lo, hi) in intervals.iter() {
            let lo = lo.max(resume);
            if lo > hi {
                continue;
            }
            let r2 = r as i128 * r as i128;
            let target = r2 - lo as i128 * lo as i128;
            let guess = isqrt(target);
            let mut x = if guess as i128 * (guess as i128 + 1) >= target {
                guess
            } else {
                guess + 1
            };
            let mut y = lo;
            let mut d =
                ((y as i128 + 1) * (y as i128 + 1) + x as i128 * (x as i128 - 1) - r2) as i64;
            while y <= x && y <= hi {
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
            resume = y.max(lo);
        }
    }
}

/// The at most two intervals of non-negative offsets `o` for which
/// `c + o` or `c - o` lands inside `0..len`. Either may be empty
/// (`lo > hi`); callers clamp `lo` to zero.
fn offset_windows(c: i64, len: i64) -> [(i64, i64); 2] {
    [(-c, len - 1 - c), (c - (len - 1), c)]
}

/// The walk interval over which `x(y)` can lie in `x_lo..=x_hi`, widened
/// by one step of slack, or `None` when that x range misses `0..=r`.
///
/// The walk holds `x(y) = min { x : x*(x+1) >= r^2 - y^2 }`, which sits up
/// to half a pixel above the ideal `sqrt(r^2 - y^2)`, so the inversion must
/// use the same discrete rule: `x(y) <= x_hi` from
/// `y^2 >= r^2 - x_hi*(x_hi + 1)` and `x(y) >= x_lo` while
/// `y^2 <= r^2 - x_lo*(x_lo - 1) - 1`.
fn walk_interval_for_x(r: i64, x_lo: i64, x_hi: i64) -> Option<(i64, i64)> {
    let x_lo = x_lo.max(0);
    let x_hi = x_hi.min(r);
    if x_lo > x_hi {
        return None;
    }
    let r2 = r as i128 * r as i128;
    let y_lo = isqrt(r2 - x_hi as i128 * (x_hi as i128 + 1) - 1);
    let y_hi = isqrt(r2 - x_lo as i128 * (x_lo as i128 - 1) - 1) + 1;
    Some((y_lo, y_hi))
}

/// Integer square root of a non-negative `i128`; `r` for `i32` centers and
/// a `u32` radius stays far inside the exactly-representable range.
fn isqrt(v: i128) -> i64 {
    (v.max(0) as u128).isqrt() as i64
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
    center: impl Into<CoordinateI32>,
    radius: u32,
    color: P,
    fill: bool,
) {
    Circle {
        center: center.into(),
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

    /// The unclipped octant walk, kept as the behavioural reference for
    /// the interval clip.
    fn reference_circle(
        image: &mut Image<Mono8>,
        center: (i32, i32),
        radius: u32,
        color: Mono8,
        fill: bool,
    ) {
        let (cx, cy) = (i64::from(center.0), i64::from(center.1));
        let r = i64::from(radius);
        let mut x = r;
        let mut y = 0;
        let mut d = 1 - r;
        while y <= x {
            if fill {
                hspan(image, cx - x, cx + x, cy + y, color);
                hspan(image, cx - x, cx + x, cy - y, color);
                hspan(image, cx - y, cx + y, cy + x, color);
                hspan(image, cx - y, cx + y, cy - x, color);
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
                    put(image, px, py, color);
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

    #[test]
    fn clipping_matches_the_unclipped_walk_exactly() {
        // The interval clip fast-forwards the exact walk state, so it must
        // not move a single pixel relative to the unclipped walk: radii
        // through several octant-step patterns, centers inside, straddling
        // and outside a 9x7 frame, outlined and filled.
        let centers = [-40, -9, -3, 0, 4, 8, 15, 40];
        for radius in 0..=32u32 {
            for &cx in &centers {
                for &cy in &centers {
                    for fill in [false, true] {
                        let mut clipped: Image<Mono8> = Image::zero(9, 7);
                        draw_circle(&mut clipped, (cx, cy), radius, ink(), fill);
                        let mut reference: Image<Mono8> = Image::zero(9, 7);
                        reference_circle(&mut reference, (cx, cy), radius, ink(), fill);
                        assert_eq!(
                            inked(&clipped),
                            inked(&reference),
                            "center ({cx}, {cy}), radius {radius}, fill {fill} \
                             diverged from the unclipped walk"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn a_huge_radius_costs_only_the_visible_arc() {
        // The walk is clipped to the frame, so a radius in the billions
        // completes immediately instead of stepping through the whole
        // octant. A hang here is the regression this test pins.
        let mut image: Image<Mono8> = Image::zero(4, 4);
        draw_circle(&mut image, (2, 2), u32::MAX, ink(), false);
        assert!(inked(&image).is_empty());
        // The filled disk of that radius contains the whole frame.
        draw_circle(&mut image, (2, 2), u32::MAX, ink(), true);
        assert_eq!(inked(&image).len(), 16);

        // A distant center whose arc does cross the frame: the visible
        // pixels sit on the ideal ring.
        let mut image: Image<Mono8> = Image::zero(6, 6);
        draw_circle(&mut image, (-1_000_000, 2), 1_000_002, ink(), false);
        let drawn = inked(&image);
        assert!(drawn.contains(&(2, 2)), "{drawn:?}");
        for (x, y) in drawn {
            let dist = ((x as f64 + 1_000_000.0).powi(2) + (y as f64 - 2.0).powi(2)).sqrt();
            assert!((dist - 1_000_002.0).abs() <= 0.5, "({x}, {y}) at {dist}");
        }

        // The filled counterpart from below: the disk boundary crosses the
        // frame between rows 2 and 3, and every pixel is on the right side
        // of it.
        let mut image: Image<Mono8> = Image::zero(6, 6);
        draw_circle(&mut image, (3, -1_000_000), 1_000_002, ink(), true);
        let drawn = inked(&image);
        assert!(
            drawn.contains(&(3, 0)) && drawn.contains(&(0, 2)),
            "{drawn:?}"
        );
        assert!(!drawn.contains(&(3, 3)), "{drawn:?}");
        for &(x, y) in &drawn {
            let dist = ((x as f64 - 3.0).powi(2) + (y as f64 + 1_000_000.0).powi(2)).sqrt();
            assert!(dist <= 1_000_002.5, "({x}, {y}) at {dist}");
        }
    }

    #[test]
    fn fill_pins_the_disc_shape_exactly() {
        // Coverage that pins the shape rather than a subset relation: for
        // each radius the fill must have no holes inside the ideal disc and
        // no reach beyond the outline's own half-pixel overshoot at octant
        // transitions (r + 0.75).
        use crate::image::ImageView;

        for r in 0..=32u32 {
            let n = (2 * r + 3) as usize;
            let c = (r + 1) as f64;
            let mut image: Image<Mono8> = Image::zero(n, n);
            draw_circle(&mut image, ((r + 1) as i32, (r + 1) as i32), r, ink(), true);
            for y in 0..n {
                for x in 0..n {
                    let dist = ((x as f64 - c).powi(2) + (y as f64 - c).powi(2)).sqrt();
                    let painted = image.pixel_at(x, y) != Mono8::new(0);
                    if dist <= r as f64 {
                        assert!(painted, "hole at ({x}, {y}), r = {r}, dist {dist}");
                    } else if dist > r as f64 + 0.75 {
                        assert!(!painted, "over-reach at ({x}, {y}), r = {r}, dist {dist}");
                    }
                }
            }
        }
    }
}

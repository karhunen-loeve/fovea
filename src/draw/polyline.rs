//! Connected line chains — open polylines and closed polygons.

use super::Drawable;
use super::line::segment;
use crate::CoordinateI32;
use crate::image::ImageViewMut;

/// A chain of line segments through a list of points, open or closed.
///
/// Consecutive points are connected with one-pixel Bresenham segments; with
/// `closed = true` an additional segment connects the last point back to the
/// first, turning the chain into a polygon outline. This is the natural way
/// to render a traced contour or a simplified polygon: pass its vertices.
///
/// Degenerate inputs degrade to no-ops rather than panics, so a polyline can
/// be pushed around while still under construction: fewer than two points
/// draw nothing, and `closed` adds its extra segment only from three points
/// up (closing a two-point chain would retrace the same segment).
///
/// Points are signed and may lie outside the image; the visible portions are
/// drawn and the rest is clipped (see [`Drawable`]).
///
/// # Examples
///
/// ```
/// use fovea::draw::{Drawable, Polyline};
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::Mono8;
///
/// let triangle = Polyline {
///     points: [(1, 1), (6, 1), (1, 6)].map(Into::into).to_vec(),
///     color: Mono8::new(255),
///     closed: true,
/// };
/// let mut image: Image<Mono8> = Image::zero(8, 8);
/// triangle.draw_into(&mut image);
/// assert_eq!(image.pixel_at(3, 1), Mono8::new(255)); // top edge
/// assert_eq!(image.pixel_at(1, 3), Mono8::new(255)); // left edge
/// assert_eq!(image.pixel_at(3, 3), Mono8::new(0)); // interior stays clear
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Polyline<P> {
    /// Vertices of the chain, connected in order.
    pub points: Vec<CoordinateI32>,
    /// Pixel value written along every segment.
    pub color: P,
    /// `true` connects the last point back to the first.
    pub closed: bool,
}

impl<P: Copy> Drawable<P> for Polyline<P> {
    fn draw_into(&self, image: &mut impl ImageViewMut<Pixel = P>) {
        draw_path(image, &self.points, self.color, self.closed);
    }
}

/// Draws a chain of line segments through `points`.
///
/// One-shot wrapper over [`Polyline`] that borrows its points instead of
/// owning them; see there for the segment and clipping contract.
///
/// # Examples
///
/// ```
/// use fovea::draw::draw_polyline;
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::Mono8;
///
/// let mut image: Image<Mono8> = Image::zero(8, 8);
/// draw_polyline(&mut image, &[(0, 4), (3, 1), (7, 5)], Mono8::new(255), false);
/// assert_eq!(image.pixel_at(3, 1), Mono8::new(255));
/// ```
pub fn draw_polyline<P: Copy, C: Into<CoordinateI32> + Copy>(
    image: &mut impl ImageViewMut<Pixel = P>,
    points: &[C],
    color: P,
    closed: bool,
) {
    draw_path(image, points, color, closed);
}

/// Segment chain shared by [`Polyline`] and [`draw_polyline`], so the free
/// function does not have to clone borrowed points into a `Vec`.
fn draw_path<P: Copy, C: Into<CoordinateI32> + Copy>(
    image: &mut impl ImageViewMut<Pixel = P>,
    points: &[C],
    color: P,
    closed: bool,
) {
    for pair in points.windows(2) {
        segment(image, pair[0].into(), pair[1].into(), color);
    }
    if closed && points.len() >= 3 {
        segment(
            image,
            points[points.len() - 1].into(),
            points[0].into(),
            color,
        );
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
    fn open_chain_connects_consecutive_points() {
        let mut image: Image<Mono8> = Image::zero(7, 7);
        draw_polyline(&mut image, &[(0, 0), (4, 0), (4, 4)], ink(), false);
        let drawn = inked(&image);
        assert_eq!(drawn.len(), 9); // 5 + 5 − shared corner
        for x in 0..=4 {
            assert!(drawn.contains(&(x, 0)));
        }
        for y in 0..=4 {
            assert!(drawn.contains(&(4, y)));
        }
        // The closing edge is absent.
        assert!(!drawn.contains(&(2, 2)));
    }

    #[test]
    fn closed_chain_adds_the_return_edge() {
        let mut image: Image<Mono8> = Image::zero(7, 7);
        draw_polyline(&mut image, &[(0, 0), (4, 0), (4, 4)], ink(), true);
        let drawn = inked(&image);
        assert!(drawn.contains(&(2, 2)), "closing diagonal missing");
    }

    #[test]
    fn degenerate_inputs_are_no_ops() {
        let mut image: Image<Mono8> = Image::zero(5, 5);
        let empty: &[(i32, i32)] = &[];
        draw_polyline(&mut image, empty, ink(), false);
        draw_polyline(&mut image, empty, ink(), true);
        draw_polyline(&mut image, &[(2, 2)], ink(), true);
        assert!(inked(&image).is_empty());
    }

    #[test]
    fn two_points_closed_draws_the_segment_once_not_thrice() {
        let mut image: Image<Mono8> = Image::zero(6, 6);
        draw_polyline(&mut image, &[(1, 1), (4, 1)], ink(), true);
        assert_eq!(inked(&image), vec![(1, 1), (2, 1), (3, 1), (4, 1)]);
    }

    #[test]
    fn struct_and_free_function_agree() {
        let points: Vec<CoordinateI32> = [(0, 5), (3, 0), (6, 5), (0, 5)].map(Into::into).to_vec();
        let mut via_struct: Image<Mono8> = Image::zero(7, 7);
        Polyline {
            points: points.clone(),
            color: ink(),
            closed: false,
        }
        .draw_into(&mut via_struct);
        let mut via_fn: Image<Mono8> = Image::zero(7, 7);
        draw_polyline(&mut via_fn, &points, ink(), false);
        assert_eq!(inked(&via_struct), inked(&via_fn));
    }
}

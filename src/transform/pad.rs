//! Enlarging an image, with a border policy that names what fills the new
//! pixels.

use crate::border::FullFrameBorder;
use crate::image::{Image, ImageView};
use crate::{Error, SignedCoordinate, Size};

mod sealed {
    use crate::Size;

    pub trait Placement {
        /// The size of the result for an image of `source`, and where the
        /// image's origin lies in it.
        fn place(self, source: Size) -> Result<(Size, usize, usize), crate::Error>;
    }
}

/// Where [`pad`] puts the image and how large the result is: the second
/// parameter of `pad`. Sealed.
///
/// A [`Size`] is a target, with the image at the origin and the new pixels
/// to the right and below; a target smaller than the image is an error.
/// [`Margins`] add pixels on each side.
pub trait PadGeometry: sealed::Placement {}

/// The pixels [`pad`] adds on each side of an image.
///
/// # Example
///
/// ```
/// use fovea::Size;
/// use fovea::border::Wrap;
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::Mono8;
/// use fovea::transform::{Margins, pad};
///
/// let img = Image::fill(4, 3, Mono8::new(9));
/// let framed = pad(&img, Margins { left: 2, right: 2, top: 1, bottom: 1 }, &Wrap)?;
/// assert_eq!(framed.size(), Size::new(8, 5));
/// # Ok::<(), fovea::Error>(())
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Margins {
    /// Columns added on the left.
    pub left: usize,
    /// Columns added on the right.
    pub right: usize,
    /// Rows added above.
    pub top: usize,
    /// Rows added below.
    pub bottom: usize,
}

/// `img` enlarged by `geometry`, the new pixels filled by `border`.
///
/// The geometry is a target [`Size`], with the image at the origin and the
/// new pixels to the right and below, or [`Margins`] on each side.
///
/// The border policy is the caller's choice of what lies beyond the image:
/// [`Constant`](crate::border::Constant) fills with a value (zeros for a
/// linear convolution through the DFT), [`Mirror`](crate::border::Mirror)
/// and [`Clamp`](crate::border::Clamp) continue the image without a jump,
/// [`Wrap`](crate::border::Wrap) repeats it. [`Skip`](crate::border::Skip)
/// has nothing to fill with, so it does not compile here.
///
/// # Errors
///
/// - [`Error::PadTargetTooSmall`] if a target is smaller than the image
///   along a side: padding never crops.
/// - [`Error::EmptySource`] if the image has no pixels, the result has
///   some, and the border policy copies from the image. `Constant` needs
///   no pixel and fills such a result with its value.
///
/// # Panics
///
/// Panics if margins make a side overflow `usize`, a size no image can
/// have: it is the limit at which allocating any image fails.
///
/// # Example
///
/// ```
/// use fovea::Size;
/// use fovea::border::{Constant, Mirror};
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::Mono8;
/// use fovea::transform::{Margins, pad};
///
/// let img = Image::generate(3, 2, |x, y| Mono8::new((10 * y + x) as u8));
///
/// // To a size, the image at the origin.
/// let big = pad(&img, Size::new(4, 3), &Constant(Mono8::new(0)))?;
/// assert_eq!(big.pixel_at(2, 1), Mono8::new(12));
/// assert_eq!(big.pixel_at(3, 2), Mono8::new(0));
/// assert!(pad(&img, Size::new(2, 8), &Mirror).is_err());
///
/// // A margin on each side, mirrored about the edge pixel.
/// let framed = pad(&img, Margins { left: 1, right: 1, top: 0, bottom: 0 }, &Mirror)?;
/// assert_eq!(framed.size(), Size::new(5, 2));
/// assert_eq!(framed.pixel_at(0, 0), Mono8::new(1));
///
/// // An empty image: a constant fills the result, a mirror has nothing to copy.
/// let empty = Image::<Mono8>::zero(0, 2);
/// assert_eq!(pad(&empty, Size::new(2, 2), &Constant(Mono8::new(7)))?.pixel_at(1, 1), Mono8::new(7));
/// assert!(pad(&empty, Size::new(2, 2), &Mirror).is_err());
/// # Ok::<(), fovea::Error>(())
/// ```
///
/// `Skip` gives no value outside the image:
///
/// ```compile_fail
/// use fovea::Size;
/// use fovea::border::Skip;
/// use fovea::image::Image;
/// use fovea::pixel::Mono8;
/// use fovea::transform::pad;
///
/// let img = Image::fill(3, 2, Mono8::new(1));
/// // ERROR: `Skip: FullFrameBorder<_>` is not satisfied.
/// let _ = pad(&img, Size::new(4, 4), &Skip)?;
/// ```
pub fn pad<I, G, B>(img: &I, geometry: G, border: &B) -> Result<Image<I::Pixel>, Error>
where
    I: ImageView,
    I::Pixel: Copy,
    G: PadGeometry,
    B: FullFrameBorder<I>,
{
    let (size, left, top) = geometry.place(img.size())?;
    if img.size().area() == 0 && size.area() > 0 {
        let value = border
            .value_without_image()
            .ok_or(Error::EmptySource { target: size })?;
        return Ok(Image::generate(size.width, size.height, |_, _| value));
    }
    Ok(Image::generate(size.width, size.height, |x, y| {
        border.pixel_at(
            img,
            SignedCoordinate::new(x as isize - left as isize, y as isize - top as isize),
        )
    }))
}

impl PadGeometry for Size {}

impl sealed::Placement for Size {
    fn place(self, source: Size) -> Result<(Size, usize, usize), Error> {
        if self.width < source.width || self.height < source.height {
            return Err(Error::PadTargetTooSmall {
                source,
                target: self,
            });
        }
        Ok((self, 0, 0))
    }
}

impl PadGeometry for Margins {}

impl sealed::Placement for Margins {
    fn place(self, source: Size) -> Result<(Size, usize, usize), Error> {
        let grow = |side: usize, a: usize, b: usize| {
            side.checked_add(a)
                .and_then(|s| s.checked_add(b))
                .expect("pad: a padded side overflows usize")
        };
        let size = Size::new(
            grow(source.width, self.left, self.right),
            grow(source.height, self.top, self.bottom),
        );
        Ok((size, self.left, self.top))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Rectangle;
    use crate::border::{Clamp, Constant, Mirror, Wrap};
    use crate::image::SubView;
    use crate::pixel::Mono8;

    fn img() -> Image<Mono8> {
        // 0 1 2
        // 10 11 12
        Image::generate(3, 2, |x, y| Mono8::new((10 * y + x) as u8))
    }

    fn values(i: &Image<Mono8>) -> Vec<Vec<u8>> {
        (0..i.height())
            .map(|y| (0..i.width()).map(|x| i.pixel_at(x, y).value()).collect())
            .collect()
    }

    #[test]
    fn a_target_keeps_the_image_at_the_origin() {
        let out = pad(&img(), Size::new(5, 3), &Constant(Mono8::new(99))).unwrap();
        assert_eq!(
            values(&out),
            [
                [0, 1, 2, 99, 99],
                [10, 11, 12, 99, 99],
                [99, 99, 99, 99, 99]
            ]
        );
    }

    #[test]
    fn each_policy_fills_the_new_pixels_its_own_way() {
        let target = Size::new(5, 2);
        let row = |b: &Image<Mono8>| values(b)[0].clone();
        assert_eq!(row(&pad(&img(), target, &Clamp).unwrap()), [0, 1, 2, 2, 2]);
        assert_eq!(row(&pad(&img(), target, &Mirror).unwrap()), [0, 1, 2, 1, 0]);
        assert_eq!(row(&pad(&img(), target, &Wrap).unwrap()), [0, 1, 2, 0, 1]);
    }

    #[test]
    fn a_target_smaller_along_a_side_is_an_error() {
        for target in [Size::new(2, 2), Size::new(3, 1), Size::new(100, 1)] {
            assert_eq!(
                pad(&img(), target, &Clamp),
                Err(Error::PadTargetTooSmall {
                    source: Size::new(3, 2),
                    target
                })
            );
        }
    }

    #[test]
    fn the_own_size_and_zero_margins_copy_the_image() {
        assert_eq!(pad(&img(), Size::new(3, 2), &Clamp).unwrap(), img());
        assert_eq!(pad(&img(), Margins::default(), &Clamp).unwrap(), img());
    }

    #[test]
    fn margins_put_the_image_inside() {
        let m = Margins {
            left: 2,
            right: 1,
            top: 1,
            bottom: 2,
        };
        let out = pad(&img(), m, &Wrap).unwrap();
        assert_eq!(out.size(), Size::new(6, 5));
        assert_eq!(
            values(&out),
            [
                [11, 12, 10, 11, 12, 10],
                [1, 2, 0, 1, 2, 0],
                [11, 12, 10, 11, 12, 10],
                [1, 2, 0, 1, 2, 0],
                [11, 12, 10, 11, 12, 10],
            ]
        );
    }

    #[test]
    fn a_view_pads_as_its_pixels_do() {
        let big = Image::generate(8, 8, |x, y| Mono8::new((10 * y + x) as u8));
        let view = big.roi(Rectangle::new((2, 3), (3, 2))).unwrap();
        let out = pad(&view, Size::new(4, 2), &Constant(Mono8::new(0))).unwrap();
        assert_eq!(values(&out), [[32, 33, 34, 0], [42, 43, 44, 0]]);
    }

    #[test]
    fn an_empty_image_pads_to_an_empty_result() {
        let empty = Image::<Mono8>::zero(0, 4);
        assert_eq!(
            pad(&empty, Size::new(0, 6), &Clamp).unwrap().size(),
            Size::new(0, 6)
        );
        assert_eq!(
            pad(&empty, Margins::default(), &Clamp).unwrap().size(),
            Size::new(0, 4)
        );
    }

    #[test]
    fn an_empty_image_fills_pixels_only_with_a_constant() {
        let empty = Image::<Mono8>::zero(0, 4);
        let filled = pad(&empty, Size::new(2, 4), &Constant(Mono8::new(5))).unwrap();
        assert_eq!(values(&filled), [[5, 5], [5, 5], [5, 5], [5, 5]]);
        let m = Margins {
            left: 1,
            right: 0,
            top: 0,
            bottom: 0,
        };
        assert_eq!(
            pad(&empty, m, &Constant(Mono8::new(5))).unwrap().size(),
            Size::new(1, 4)
        );
        for result in [
            pad(&empty, Size::new(2, 4), &Clamp),
            pad(&empty, Size::new(2, 4), &Mirror),
            pad(&empty, m, &Wrap),
        ] {
            assert!(
                matches!(result, Err(Error::EmptySource { .. })),
                "{result:?}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "overflows usize")]
    fn margins_that_overflow_a_side_panic() {
        let m = Margins {
            left: usize::MAX,
            right: 1,
            top: 0,
            bottom: 0,
        };
        let _ = pad(&img(), m, &Clamp).unwrap();
    }
}

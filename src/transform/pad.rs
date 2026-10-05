//! Enlarging an image, with a border policy that names what fills the new
//! pixels.

use crate::border::FullFrameBorder;
use crate::image::{Image, ImageView};
use crate::{Error, SignedCoordinate, Size};

mod sealed {
    use crate::border::FullFrameBorder;
    use crate::image::{Image, ImageView};

    pub trait Placement {
        fn pad_image<I, B>(
            self,
            img: &I,
            border: &B,
        ) -> <Self as super::PadGeometry>::Output<Image<I::Pixel>>
        where
            Self: super::PadGeometry,
            I: ImageView,
            I::Pixel: Copy,
            B: FullFrameBorder<I>;
    }
}

/// Where [`pad`] puts the image and how large the result is: the second
/// parameter of `pad`. Sealed.
///
/// A [`Size`] is a target, with the image at the origin and the new pixels
/// to the right and below. [`Margins`] add pixels on each side. The
/// associated type says whether a geometry can fail: a target can be
/// smaller than the image, margins cannot.
pub trait PadGeometry: sealed::Placement {
    /// What [`pad`] returns: `Result<T, Error>` for a geometry that can
    /// fail, `T` for one that cannot.
    type Output<T>;
}

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
/// let framed = pad(&img, Margins { left: 2, right: 2, top: 1, bottom: 1 }, &Wrap);
/// assert_eq!(framed.size(), Size::new(8, 5));
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
/// new pixels to the right and below, or [`Margins`] on each side. A
/// target smaller than the image along a side is an
/// [`Error::PadTargetTooSmall`]: padding never crops. Margins cannot fail,
/// so with them `pad` returns the image itself.
///
/// The border policy is the caller's choice of what lies beyond the image:
/// [`Constant`](crate::border::Constant) fills with a value (zeros for a
/// linear convolution through the DFT), [`Mirror`](crate::border::Mirror)
/// and [`Clamp`](crate::border::Clamp) continue the image without a jump,
/// [`Wrap`](crate::border::Wrap) repeats it. [`Skip`](crate::border::Skip)
/// has nothing to fill with, so it does not compile here.
///
/// # Panics
///
/// Panics if the image has no pixels and the result has some, since no
/// pixel is there to fill from, and if margins make a side overflow
/// `usize`.
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
/// let framed = pad(&img, Margins { left: 1, right: 1, top: 0, bottom: 0 }, &Mirror);
/// assert_eq!(framed.size(), Size::new(5, 2));
/// assert_eq!(framed.pixel_at(0, 0), Mono8::new(1));
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
/// let _ = pad(&img, Size::new(4, 4), &Skip);
/// ```
#[must_use]
pub fn pad<I, G, B>(img: &I, geometry: G, border: &B) -> G::Output<Image<I::Pixel>>
where
    I: ImageView,
    I::Pixel: Copy,
    G: PadGeometry,
    B: FullFrameBorder<I>,
{
    geometry.pad_image(img, border)
}

/// The image of `size` holding `img` with its origin at `(left, top)`, the
/// rest from `border`.
fn place<I, B>(img: &I, size: Size, left: usize, top: usize, border: &B) -> Image<I::Pixel>
where
    I: ImageView,
    I::Pixel: Copy,
    B: FullFrameBorder<I>,
{
    let source = img.size();
    assert!(
        source.area() > 0 || size.area() == 0,
        "pad: the image is {}x{}, so there is no pixel to fill the {}x{} result from",
        source.width,
        source.height,
        size.width,
        size.height
    );
    Image::generate(size.width, size.height, |x, y| {
        border.pixel_at(
            img,
            SignedCoordinate::new(x as isize - left as isize, y as isize - top as isize),
        )
    })
}

impl PadGeometry for Size {
    type Output<T> = Result<T, Error>;
}

impl sealed::Placement for Size {
    fn pad_image<I, B>(self, img: &I, border: &B) -> Result<Image<I::Pixel>, Error>
    where
        I: ImageView,
        I::Pixel: Copy,
        B: FullFrameBorder<I>,
    {
        let source = img.size();
        if self.width < source.width || self.height < source.height {
            return Err(Error::PadTargetTooSmall {
                source,
                target: self,
            });
        }
        Ok(place(img, self, 0, 0, border))
    }
}

impl PadGeometry for Margins {
    type Output<T> = T;
}

impl sealed::Placement for Margins {
    fn pad_image<I, B>(self, img: &I, border: &B) -> Image<I::Pixel>
    where
        I: ImageView,
        I::Pixel: Copy,
        B: FullFrameBorder<I>,
    {
        let grow = |side: usize, a: usize, b: usize| {
            side.checked_add(a)
                .and_then(|s| s.checked_add(b))
                .expect("pad: a padded side overflows usize")
        };
        let size = Size::new(
            grow(img.width(), self.left, self.right),
            grow(img.height(), self.top, self.bottom),
        );
        place(img, size, self.left, self.top, border)
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
        assert_eq!(pad(&img(), Margins::default(), &Clamp), img());
    }

    #[test]
    fn margins_put_the_image_inside() {
        let m = Margins {
            left: 2,
            right: 1,
            top: 1,
            bottom: 2,
        };
        let out = pad(&img(), m, &Wrap);
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
            pad(&empty, Margins::default(), &Clamp).size(),
            Size::new(0, 4)
        );
    }

    #[test]
    #[should_panic(expected = "no pixel to fill")]
    fn an_empty_image_cannot_fill_pixels() {
        let empty = Image::<Mono8>::zero(0, 4);
        let _ = pad(&empty, Size::new(2, 4), &Constant(Mono8::new(0)));
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
        let _ = pad(&img(), m, &Clamp);
    }
}

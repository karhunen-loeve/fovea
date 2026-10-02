//! Remapping an image through a mapping of the plane: rectification and
//! lens correction.

use crate::analyze::sampling::sample;
use crate::border::FullFrameBorder;
use crate::geometry::{Pixels, Point, SourceLookup};
use crate::image::{Image, ImageView};
use crate::pixel::{FromLinear, LinearPixel, LinearSpace};
use crate::transform::InterpolationKernel;
use crate::{SignedCoordinate, Size};

/// A position far outside every image, where a border policy gives the value
/// of "no source".
const FAR_OUTSIDE: SignedCoordinate = SignedCoordinate::new(isize::MIN / 2, isize::MIN / 2);

/// The value of `image` at `source`, interpolated with `kernel`, or the
/// border's value far outside the image where there is no usable source.
fn value_at<I, K, B, Q>(image: &I, source: Option<Point<Pixels>>, kernel: K, border: &B) -> I::Pixel
where
    I: ImageView,
    I::Pixel: LinearPixel<Accumulator = Q> + LinearSpace + FromLinear<Q>,
    K: InterpolationKernel,
    B: FullFrameBorder<I>,
{
    source
        .and_then(|p| sample(image, p.into(), kernel, border))
        .map(I::Pixel::from_linear)
        .unwrap_or_else(|| border.pixel_at(image, FAR_OUTSIDE))
}

fn assert_source_not_empty(source: Size, target: Size) {
    assert!(
        source.width > 0 && source.height > 0 || target.width == 0 || target.height == 0,
        "remap: the source is {}x{}, so there is no sample to fill the {}x{} target from",
        source.width,
        source.height,
        target.width,
        target.height
    );
}

/// Remaps `image` into a new image of `size`: each destination pixel takes
/// the value of the source where `lookup` sends it, interpolated with
/// `kernel`.
///
/// The direction is in the type of `lookup`. A
/// [`DestToSource`](crate::geometry::DestToSource) mapping is read as it is,
/// which is how a remap works and how a lens model is used; a
/// [`SourceToDest`](crate::geometry::SourceToDest) mapping is the one a
/// caller thinks of, where each source pixel ends up, and was inverted once
/// when it was built. A bare matrix is not accepted.
///
/// Positions follow the crate's pixel-centre convention: `(0.0, 0.0)` is the
/// centre of pixel `(0, 0)`, so a mapping that is the identity copies the
/// image exactly.
///
/// The border policy decides what a destination pixel gets where the
/// kernel's taps leave the source, and must give every position a value:
/// [`Constant`](crate::border::Constant), [`Clamp`](crate::border::Clamp),
/// [`Mirror`](crate::border::Mirror) or [`Wrap`](crate::border::Wrap).
/// `Constant(black)` leaves the area without a source black, and blends the
/// image's edge into it. A destination pixel whose source does not exist at
/// all, on a homography's vanishing line, is treated as far outside the
/// image.
///
/// To apply the same mapping to many frames, build a
/// [`DestToSourceTable`] once.
///
/// # Panics
///
/// Panics if `image` is empty and `size` is not: there is no sample to fill
/// the target from. This mirrors [`resize`](crate::transform::resize).
///
/// # Example
///
/// ```
/// use fovea::{Pixels, Point, Size};
/// use fovea::border::Constant;
/// use fovea::geometry::{DestToSource, Similarity};
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::Mono8;
/// use fovea::transform::{Bilinear, remap};
///
/// let img = Image::generate(8, 8, |x, y| Mono8::new((10 * x + y) as u8));
///
/// // A quarter turn about the origin, read from the destination: each
/// // destination pixel (x, y) reads the source at (y, −x)…
/// let turn: Similarity<Pixels, Pixels> =
///     Similarity::try_linear(1.0, -core::f64::consts::FRAC_PI_2)?;
/// let out = remap(&img, &DestToSource(turn), Bilinear, &Constant(Mono8::new(0)), Size::new(8, 8));
/// // …which lies inside the source only for x = 0.
/// assert_eq!(out.pixel_at(0, 3), img.pixel_at(3, 0));
/// assert_eq!(out.pixel_at(4, 3), Mono8::new(0));
/// # Ok::<(), fovea::Error>(())
/// ```
///
/// A border policy that leaves pixels without a value does not compile:
///
/// ```compile_fail
/// use fovea::{Pixels, Size};
/// use fovea::border::Skip;
/// use fovea::geometry::{DestToSource, UniformScale};
/// use fovea::image::Image;
/// use fovea::pixel::Mono8;
/// use fovea::transform::{Bilinear, remap};
///
/// let img = Image::fill(8, 8, Mono8::new(1));
/// let s: UniformScale<Pixels, Pixels> = fovea::uniform_scale!(2.0);
/// let _ = remap(&img, &DestToSource(s), Bilinear, &Skip, Size::new(4, 4));
/// ```
#[must_use]
pub fn remap<I, L, K, B, Q>(
    image: &I,
    lookup: &L,
    kernel: K,
    border: &B,
    size: Size,
) -> Image<I::Pixel>
where
    I: ImageView,
    I::Pixel: LinearPixel<Accumulator = Q> + LinearSpace + FromLinear<Q>,
    L: SourceLookup,
    K: InterpolationKernel,
    B: FullFrameBorder<I>,
{
    assert_source_not_empty(image.size(), size);
    Image::generate(size.width, size.height, |x, y| {
        let source = lookup.source_of(Point::new(x as f64, y as f64));
        value_at(image, source, kernel, border)
    })
}

/// Where each destination pixel's source lies, computed once, for remapping
/// many frames through the same mapping.
///
/// A lens model costs a polynomial per pixel, a homography a division; the
/// table turns either into a lookup. It stores, per destination pixel, the
/// source minus the destination as two `f32`. A displacement is a small
/// number where a coordinate is a large one, so it rounds far finer than a
/// stored coordinate would: about 2·10⁻⁶ px for a shift of 50 px, where an
/// `f32` coordinate at x = 4000 rounds by 1.2·10⁻⁴ px. The source is formed
/// in `f64` when the table is applied. A destination pixel without a source
/// is stored as NaN and treated as far outside the image.
///
/// # Example
///
/// ```
/// use fovea::{Point, Size};
/// use fovea::border::Constant;
/// use fovea::geometry::{BrownConrady, CameraMatrix, DestToSource, FocalLength};
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::Mono8;
/// use fovea::transform::{CatmullRom, DestToSourceTable};
///
/// let cam = CameraMatrix::try_new(FocalLength { x: 600.0, y: 600.0 }, Point::new(320.0, 240.0))?;
/// let lens = BrownConrady::from_opencv(cam, &[-0.25, 0.08, 0.0, 0.0, 0.0])?;
///
/// // Built once…
/// let undistort = DestToSourceTable::new(&DestToSource(lens), Size::new(640, 480));
///
/// // …applied to every frame.
/// let frame = Image::fill(640, 480, Mono8::new(90));
/// let ideal = undistort.remap(&frame, CatmullRom, &Constant(Mono8::new(0)));
/// assert_eq!(ideal.size(), Size::new(640, 480));
/// # Ok::<(), fovea::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct DestToSourceTable {
    size: Size,
    shift: Vec<[f32; 2]>,
}

impl DestToSourceTable {
    /// The table of `lookup` for a destination of `size`.
    #[must_use]
    pub fn new<L: SourceLookup>(lookup: &L, size: Size) -> Self {
        let mut shift = Vec::with_capacity(size.area());
        for y in 0..size.height {
            for x in 0..size.width {
                let dest = Point::new(x as f64, y as f64);
                shift.push(match lookup.source_of(dest) {
                    Some(s) => [(s.x - dest.x) as f32, (s.y - dest.y) as f32],
                    None => [f32::NAN, f32::NAN],
                });
            }
        }
        Self { size, shift }
    }

    /// The size of the destination the table was built for.
    #[must_use]
    pub fn size(&self) -> Size {
        self.size
    }

    /// The source of the destination pixel `(x, y)` as the table stores it,
    /// or `None` outside the table or where the mapping has no source.
    #[must_use]
    pub fn source_of(&self, x: usize, y: usize) -> Option<Point<Pixels>> {
        if x >= self.size.width || y >= self.size.height {
            return None;
        }
        let [dx, dy] = self.shift[y * self.size.width + x];
        let s = Point::new(x as f64 + f64::from(dx), y as f64 + f64::from(dy));
        (s.x.is_finite() && s.y.is_finite()).then_some(s)
    }

    /// Remaps `image` through the table into an image of the table's size,
    /// as [`remap`] does with the mapping the table was built from.
    ///
    /// # Panics
    ///
    /// Panics if `image` is empty and the table is not, as [`remap`] does.
    #[must_use]
    pub fn remap<I, K, B, Q>(&self, image: &I, kernel: K, border: &B) -> Image<I::Pixel>
    where
        I: ImageView,
        I::Pixel: LinearPixel<Accumulator = Q> + LinearSpace + FromLinear<Q>,
        K: InterpolationKernel,
        B: FullFrameBorder<I>,
    {
        assert_source_not_empty(image.size(), self.size);
        Image::generate(self.size.width, self.size.height, |x, y| {
            value_at(image, self.source_of(x, y), kernel, border)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::border::{Clamp, Constant};
    use crate::geometry::{
        Affine, BrownConrady, CameraMatrix, DestToSource, FocalLength, Homography, PlaneMap,
        SourceToDest, UniformScale, Vector,
    };
    use crate::pixel::{Mono8, MonoF32};
    use crate::transform::{Bilinear, CatmullRom};

    fn field() -> Image<MonoF32> {
        Image::generate(16, 12, |x, y| {
            MonoF32::new(((x * 37 + y * 11) % 17) as f32 - 3.5)
        })
    }

    fn lens() -> BrownConrady {
        let cam = CameraMatrix::try_new(FocalLength { x: 400.0, y: 400.0 }, Point::new(8.0, 6.0))
            .unwrap();
        BrownConrady::from_opencv(cam, &[-0.3, 0.1, 0.001, -0.0005, 0.0]).unwrap()
    }

    #[test]
    fn the_identity_copies_the_image_exactly() {
        let img = field();
        let id: UniformScale<Pixels, Pixels> = crate::uniform_scale!(1.0);
        let out = remap(&img, &DestToSource(id), CatmullRom, &Clamp, img.size());
        assert_eq!(out, img);
    }

    #[test]
    fn an_integer_shift_moves_pixels_and_fills_from_the_border() {
        let img = field();
        let shift: Affine<Pixels, Pixels> =
            Affine::try_new([[1.0, 0.0], [0.0, 1.0]], Vector::new(3.0, -2.0)).unwrap();
        let fill = MonoF32::new(99.0);
        let out = remap(
            &img,
            &DestToSource(shift),
            Bilinear,
            &Constant(fill),
            img.size(),
        );
        for y in 0..12 {
            for x in 0..16 {
                let (sx, sy) = (x as isize + 3, y as isize - 2);
                let expected = if (0..16).contains(&sx) && (0..12).contains(&sy) {
                    img.pixel_at(sx as usize, sy as usize)
                } else {
                    fill
                };
                assert_eq!(out.pixel_at(x, y), expected, "({x}, {y})");
            }
        }
        // The same move given from source to destination is its inverse.
        let back: Affine<Pixels, Pixels> =
            Affine::try_new([[1.0, 0.0], [0.0, 1.0]], Vector::new(-3.0, 2.0)).unwrap();
        let again = remap(
            &img,
            &SourceToDest::new(back),
            Bilinear,
            &Constant(fill),
            img.size(),
        );
        assert_eq!(again, out);
    }

    #[test]
    fn a_pixel_without_a_source_takes_the_border_value() {
        let img = field();
        // A homography whose vanishing line is y = 4, inside the destination.
        let h: Homography<Pixels, Pixels> =
            Homography::try_new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -0.25, 1.0]]).unwrap();
        assert_eq!(h.try_map_point(Point::new(5.0, 4.0)), None);
        let fill = MonoF32::new(-7.0);
        let out = remap(
            &img,
            &DestToSource(h),
            Bilinear,
            &Constant(fill),
            img.size(),
        );
        assert_eq!(out.pixel_at(5, 4), fill);
        let table = DestToSourceTable::new(&DestToSource(h), img.size());
        assert_eq!(table.source_of(5, 4), None);
        assert_eq!(
            table.remap(&img, Bilinear, &Constant(fill)).pixel_at(5, 4),
            fill
        );
    }

    #[test]
    fn the_table_stores_displacements_finely() {
        let l = lens();
        let size = Size::new(16, 12);
        let table = DestToSourceTable::new(&DestToSource(l), size);
        assert_eq!(table.size(), size);
        assert_eq!(table.source_of(16, 0), None);
        for y in 0..12 {
            for x in 0..16 {
                let exact = l.try_map_point(Point::new(x as f64, y as f64)).unwrap();
                let stored = table.source_of(x, y).unwrap();
                assert!(stored.distance(exact).get() < 1e-5, "({x}, {y})");
            }
        }
    }

    #[test]
    fn the_table_and_the_mapping_remap_alike() {
        let img = field();
        let table = DestToSourceTable::new(&DestToSource(lens()), img.size());
        let direct = remap(&img, &DestToSource(lens()), CatmullRom, &Clamp, img.size());
        let tabled = table.remap(&img, CatmullRom, &Clamp);
        for y in 0..12 {
            for x in 0..16 {
                let (a, b) = (direct.pixel_at(x, y).value(), tabled.pixel_at(x, y).value());
                assert!((a - b).abs() < 1e-3, "({x}, {y}): {a} vs {b}");
            }
        }
    }

    #[test]
    fn undistorting_an_image_distorted_by_the_model_restores_it() {
        // A smooth pattern in the ideal image, the camera image the lens
        // makes of it, and the camera image undistorted again.
        let pattern = |x: f64, y: f64| (x * 0.21).sin() * 40.0 + (y * 0.17).cos() * 30.0 + 100.0;
        // A short focal length, so the 16×12 image spans a strong barrel:
        // its corners move by more than half a pixel.
        let cam =
            CameraMatrix::try_new(FocalLength { x: 20.0, y: 20.0 }, Point::new(8.0, 6.0)).unwrap();
        let l = BrownConrady::from_opencv(cam, &[-0.3, 0.1, 0.001, -0.0005, 0.0]).unwrap();
        assert!(l.try_map_point(Point::new(0.0, 0.0)).unwrap().x > 0.5);
        let ideal = Image::generate(16, 12, |x, y| {
            MonoF32::new(pattern(x as f64, y as f64) as f32)
        });
        // The camera image: each camera pixel shows the ideal point that the
        // lens sends there.
        let camera = Image::generate(16, 12, |x, y| {
            let p = l.undistort_point(Point::new(x as f64, y as f64)).unwrap();
            MonoF32::new(pattern(p.x, p.y) as f32)
        });
        let restored = remap(&camera, &DestToSource(l), CatmullRom, &Clamp, ideal.size());
        for y in 2..10 {
            for x in 2..14 {
                let (a, b) = (
                    restored.pixel_at(x, y).value(),
                    ideal.pixel_at(x, y).value(),
                );
                // The interpolation of a smooth pattern, a test criterion.
                assert!((a - b).abs() < 0.5, "({x}, {y}): {a} vs {b}");
            }
        }
    }

    #[test]
    fn an_integer_image_rounds_back_to_its_type() {
        let img = Image::generate(4, 4, |x, _| Mono8::new(10 * x as u8 + 5));
        let half: UniformScale<Pixels, Pixels> = crate::uniform_scale!(0.5);
        let out = remap(&img, &DestToSource(half), Bilinear, &Clamp, Size::new(4, 4));
        // Destination x = 1 reads the source at x = 0.5: between 5 and 15.
        assert_eq!(out.pixel_at(1, 0), Mono8::new(10));
        assert_eq!(out.pixel_at(2, 0), Mono8::new(15));
    }

    #[test]
    #[should_panic(expected = "no sample to fill")]
    fn an_empty_source_cannot_fill_a_target() {
        let empty: Image<MonoF32> = Image::zero(0, 0);
        let id: UniformScale<Pixels, Pixels> = crate::uniform_scale!(1.0);
        let _ = remap(&empty, &DestToSource(id), Bilinear, &Clamp, Size::new(2, 2));
    }

    #[test]
    fn an_empty_target_needs_no_source() {
        let empty: Image<MonoF32> = Image::zero(0, 0);
        let id: UniformScale<Pixels, Pixels> = crate::uniform_scale!(1.0);
        let out = remap(&empty, &DestToSource(id), Bilinear, &Clamp, Size::new(0, 3));
        assert_eq!(out.size(), Size::new(0, 3));
    }
}

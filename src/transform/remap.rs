//! Remapping an image through a mapping of the plane: rectification and
//! lens correction.

use core::marker::PhantomData;

use crate::analyze::sampling::{MAX_POSITION, sample};
use crate::border::FullFrameBorder;
use crate::geometry::{PixelSpace, Point, SourceLookup};
use crate::image::{Image, ImageView};
use crate::pixel::{FromLinear, LinearPixel, LinearSpace};
use crate::transform::{InterpolationKernel, NearestNeighbor};
use crate::{Error, SignedCoordinate, Size};

/// A position far outside every image, where a border policy gives the value
/// of "no source".
const FAR_OUTSIDE: SignedCoordinate = SignedCoordinate::new(isize::MIN / 2, isize::MIN / 2);

mod sealed {
    use crate::border::FullFrameBorder;
    use crate::geometry::Point;
    use crate::image::ImageView;

    /// The engine of a [`RemapMethod`](super::RemapMethod).
    pub trait Method<I: ImageView, P> {
        /// The value of `image` at `at`, or `None` where the method has no
        /// usable value there.
        fn value_at<B: FullFrameBorder<I>>(
            &self,
            image: &I,
            at: Point<I::Space>,
            border: &B,
        ) -> Option<P>;

        /// A pixel of the source as an output pixel: a border policy's value.
        fn convert(&self, pixel: I::Pixel) -> P;
    }
}

/// How a remap reads the source between pixel centres, and what it may
/// write.
///
/// The shape of [`ResizeMethod`](crate::transform::ResizeMethod): each
/// method brings its own requirements on the pixel types, so the remap
/// itself requires none.
///
/// - Every [`InterpolationKernel`] (`Bilinear`, `CatmullRom`, the Lanczos
///   family, …) weighs the source pixels around a position, so it needs
///   pixels that can be summed: `I::Pixel` in a linear space, and an output
///   pixel `P` that a weighted sum converts into. The output type is the
///   caller's to name, so an 8-bit image can be remapped into `MonoF32`
///   without rounding.
/// - [`NearestNeighbor`] copies the pixel whose extent contains the
///   position, so it works for every pixel type: masks, label images,
///   sRGB. The output pixel is the source pixel, converted by `Into`.
///
/// Sealed: the engine runs inside the remap's own checks. A kernel of your
/// own joins through [`InterpolationKernel`], which is open.
///
/// # Example
///
/// ```
/// use fovea::{Pixels, Size};
/// use fovea::border::Constant;
/// use fovea::geometry::{DestToSource, UniformScale};
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::Label32;
/// use fovea::transform::{NearestNeighbor, remap};
///
/// // A label image cannot be interpolated, but it can be remapped.
/// let labels = Image::generate(4, 4, |x, _| Label32::new(x as u32));
/// let half: UniformScale<Pixels, Pixels> = fovea::uniform_scale!(2.0);
/// let out: Image<Label32> =
///     remap(&labels, &DestToSource(half), NearestNeighbor, &Constant(Label32::new(0)), Size::new(2, 2))?;
/// assert_eq!(out.pixel_at(1, 0), Label32::new(2));
/// # Ok::<(), fovea::Error>(())
/// ```
pub trait RemapMethod<I: ImageView, P>: sealed::Method<I, P> {}

impl<K, I, P, Q> sealed::Method<I, P> for K
where
    K: InterpolationKernel,
    I: ImageView,
    I::Pixel: LinearPixel<Accumulator = Q> + LinearSpace,
    P: FromLinear<Q>,
{
    fn value_at<B: FullFrameBorder<I>>(
        &self,
        image: &I,
        at: Point<I::Space>,
        border: &B,
    ) -> Option<P> {
        sample(image, at, *self, border).map(P::from_linear)
    }

    fn convert(&self, pixel: I::Pixel) -> P {
        P::from_linear(pixel.to_accumulator())
    }
}

impl<K, I, P, Q> RemapMethod<I, P> for K
where
    K: InterpolationKernel,
    I: ImageView,
    I::Pixel: LinearPixel<Accumulator = Q> + LinearSpace,
    P: FromLinear<Q>,
{
}

impl<I, P> sealed::Method<I, P> for NearestNeighbor
where
    I: ImageView,
    I::Pixel: Into<P>,
{
    /// The pixel whose extent `[k − 0.5, k + 0.5)` contains the position,
    /// the half-open rule [`resize`](crate::transform::resize) uses, read
    /// through the border policy where it lies outside the image.
    fn value_at<B: FullFrameBorder<I>>(
        &self,
        image: &I,
        at: Point<I::Space>,
        border: &B,
    ) -> Option<P> {
        let usable = |t: f64| t.is_finite() && t.abs() < MAX_POSITION;
        if !(usable(at.x) && usable(at.y)) {
            return None;
        }
        // Exact: below 2^52 every half-integer is representable, and the
        // floor of the shifted position fits `isize`.
        let (x, y) = ((at.x + 0.5).floor() as isize, (at.y + 0.5).floor() as isize);
        let size = image.size();
        let inside = x >= 0 && y >= 0 && (x as usize) < size.width && (y as usize) < size.height;
        let pixel = if inside {
            image.pixel_at(x as usize, y as usize)
        } else {
            border.pixel_at(image, SignedCoordinate::new(x, y))
        };
        Some(pixel.into())
    }

    fn convert(&self, pixel: I::Pixel) -> P {
        pixel.into()
    }
}

impl<I, P> RemapMethod<I, P> for NearestNeighbor
where
    I: ImageView,
    I::Pixel: Into<P>,
{
}

/// The value of `image` at `source` by `method`, or the border's value far
/// outside the image where there is no usable source.
fn value_at<I, M, B, P>(image: &I, source: Option<Point<I::Space>>, method: &M, border: &B) -> P
where
    I: ImageView,
    M: RemapMethod<I, P>,
    B: FullFrameBorder<I>,
{
    source
        .and_then(|p| method.value_at(image, p, border))
        .unwrap_or_else(|| method.convert(border.pixel_at(image, FAR_OUTSIDE)))
}

/// The target of `size` filled with the border policy's value when `image`
/// has no pixels and the target has some: `Some(Ok)` under `Constant`,
/// `Some(Err)` under a policy that copies from the image, and `None` when
/// there is a source to sample.
fn from_empty_source<I, M, B, P, D>(
    image: &I,
    method: &M,
    border: &B,
    size: Size,
) -> Option<Result<Image<P, D>, Error>>
where
    I: ImageView,
    M: RemapMethod<I, P>,
    B: FullFrameBorder<I>,
    P: Clone,
    D: PixelSpace,
{
    if image.size().area() > 0 || size.area() == 0 {
        return None;
    }
    Some(match border.value_without_image() {
        Some(value) => Ok(Image::fill_in(
            size.width,
            size.height,
            method.convert(value),
        )),
        None => Err(Error::EmptySource { target: size }),
    })
}

/// Remaps `image` into a new image of `size`: each destination pixel takes
/// the value of the source where `lookup` sends it, read by `method`.
///
/// The direction is in the type of `lookup`. A
/// [`DestToSource`](crate::geometry::DestToSource) mapping is read as it is,
/// which is how a remap works and how a lens model is used; a
/// [`SourceToDest`](crate::geometry::SourceToDest) mapping is the one a
/// caller thinks of, where each source pixel ends up, and was inverted once
/// when it was built. A bare matrix is not accepted.
///
/// The spaces are in the type as well: `image` must lie in the lookup's
/// source space, and the result lies in its destination space. A remap
/// through a lens model reads the camera image, in
/// [`Pixels`](crate::Pixels), and returns the ideal image, in
/// [`IdealPixels`](crate::IdealPixels).
///
/// `method` is an [`InterpolationKernel`] or [`NearestNeighbor`]; see
/// [`RemapMethod`]. The output pixel type `P` is the caller's to name, as
/// for [`resize`](crate::transform::resize).
///
/// Positions follow the crate's pixel-centre convention: `(0.0, 0.0)` is the
/// centre of pixel `(0, 0)`, so a mapping that is the identity copies the
/// image exactly.
///
/// The border policy decides what a destination pixel gets where the
/// method's taps leave the source, and must give every position a value:
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
/// # Errors
///
/// [`Error::EmptySource`] if `image` has no pixels, `size` has some, and
/// the border policy copies from the image; under
/// [`Constant`](crate::border::Constant), which needs no pixel, the target
/// is filled with its value instead.
///
/// # Example
///
/// ```
/// use fovea::{Angle, Pixels, Point, Size};
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
///     Similarity::try_linear(1.0, Angle::from_degrees(-90.0)?)?;
/// let out: Image<Mono8> =
///     remap(&img, &DestToSource(turn), Bilinear, &Constant(Mono8::new(0)), Size::new(8, 8))?;
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
/// let _: Result<Image<Mono8>, _> = remap(&img, &DestToSource(s), Bilinear, &Skip, Size::new(4, 4));
/// ```
///
/// Nor does a remap through a calibration, which maps into a unit of the
/// world rather than into an image:
///
/// ```compile_fail
/// use fovea::{Millimeter, Pixels, Size};
/// use fovea::border::Constant;
/// use fovea::geometry::{DestToSource, UniformScale};
/// use fovea::image::Image;
/// use fovea::pixel::Mono8;
/// use fovea::transform::{Bilinear, remap};
///
/// let img = Image::fill(8, 8, Mono8::new(1));
/// let cal: UniformScale<Pixels, Millimeter> = fovea::uniform_scale!(0.05);
/// let _: Result<Image<Mono8>, _> =
///     remap(&img, &DestToSource(cal), Bilinear, &Constant(Mono8::new(0)), Size::new(4, 4));
/// ```
pub fn remap<I, L, M, B, P>(
    image: &I,
    lookup: &L,
    method: M,
    border: &B,
    size: Size,
) -> Result<Image<P, L::Dest>, Error>
where
    I: ImageView,
    L: SourceLookup<Source = I::Space>,
    M: RemapMethod<I, P>,
    B: FullFrameBorder<I>,
    P: Clone,
{
    if let Some(filled) = from_empty_source(image, &method, border, size) {
        return filled;
    }
    Ok(Image::generate_in(size.width, size.height, |x, y| {
        let source = lookup.source_of(Point::new(x as f64, y as f64));
        value_at(image, source, &method, border)
    }))
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
/// `D` and `S` are the destination and source spaces of the lookup the
/// table was built from: a table built from a lens model reads camera
/// images and returns ideal ones.
///
/// # Example
///
/// ```
/// use fovea::{IdealPixels, Point, Size};
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
/// let ideal: Image<Mono8, IdealPixels> =
///     undistort.remap(&frame, CatmullRom, &Constant(Mono8::new(0)))?;
/// assert_eq!(ideal.size(), Size::new(640, 480));
/// # Ok::<(), fovea::Error>(())
/// ```
pub struct DestToSourceTable<D, S> {
    size: Size,
    shift: Vec<[f32; 2]>,
    spaces: PhantomData<fn() -> (D, S)>,
}

// Written by hand rather than derived: a derive would require `D` and `S`
// to be `Clone` and `Debug`, markers that are never instantiated. There is
// no `PartialEq`: an unmapped pixel is stored as NaN, so a table whose
// mapping leaves pixels unmapped would be unequal to its own clone.
impl<D, S> Clone for DestToSourceTable<D, S> {
    fn clone(&self) -> Self {
        Self {
            size: self.size,
            shift: self.shift.clone(),
            spaces: PhantomData,
        }
    }
}

impl<D, S> core::fmt::Debug for DestToSourceTable<D, S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DestToSourceTable")
            .field("size", &self.size)
            .field("shift", &self.shift)
            .finish()
    }
}

impl<D: PixelSpace, S: PixelSpace> DestToSourceTable<D, S> {
    /// The table of `lookup` for a destination of `size`.
    #[must_use]
    pub fn new<L: SourceLookup<Dest = D, Source = S>>(lookup: &L, size: Size) -> Self {
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
        Self {
            size,
            shift,
            spaces: PhantomData,
        }
    }

    /// The size of the destination the table was built for.
    #[must_use]
    pub fn size(&self) -> Size {
        self.size
    }

    /// The source of the destination pixel `(x, y)` as the table stores it,
    /// or `None` outside the table or where the mapping has no source.
    #[must_use]
    pub fn source_of(&self, x: usize, y: usize) -> Option<Point<S>> {
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
    /// # Errors
    ///
    /// [`Error::EmptySource`] as [`remap`] reports it.
    pub fn remap<I, M, B, P>(&self, image: &I, method: M, border: &B) -> Result<Image<P, D>, Error>
    where
        I: ImageView<Space = S>,
        M: RemapMethod<I, P>,
        B: FullFrameBorder<I>,
        P: Clone,
    {
        if let Some(filled) = from_empty_source(image, &method, border, self.size) {
            return filled;
        }
        Ok(Image::generate_in(
            self.size.width,
            self.size.height,
            |x, y| value_at(image, self.source_of(x, y), &method, border),
        ))
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
    use crate::{IdealPixels, Pixels};

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
        let out = remap(&img, &DestToSource(id), CatmullRom, &Clamp, img.size()).unwrap();
        assert_eq!(out, img);
    }

    #[test]
    fn an_integer_shift_moves_pixels_and_fills_from_the_border() {
        let img = field();
        let shift: Affine<Pixels, Pixels> =
            Affine::try_new([[1.0, 0.0], [0.0, 1.0]], Vector::new(3.0, -2.0)).unwrap();
        let fill = MonoF32::new(99.0);
        let out: Image<MonoF32> = remap(
            &img,
            &DestToSource(shift),
            Bilinear,
            &Constant(fill),
            img.size(),
        )
        .unwrap();
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
        )
        .unwrap();
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
        let out: Image<MonoF32> = remap(
            &img,
            &DestToSource(h),
            Bilinear,
            &Constant(fill),
            img.size(),
        )
        .unwrap();
        assert_eq!(out.pixel_at(5, 4), fill);
        let table = DestToSourceTable::new(&DestToSource(h), img.size());
        assert_eq!(table.source_of(5, 4), None);
        let tabled: Image<MonoF32> = table.remap(&img, Bilinear, &Constant(fill)).unwrap();
        assert_eq!(tabled.pixel_at(5, 4), fill);
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
        let direct: Image<MonoF32, IdealPixels> =
            remap(&img, &DestToSource(lens()), CatmullRom, &Clamp, img.size()).unwrap();
        let tabled: Image<MonoF32, IdealPixels> = table.remap(&img, CatmullRom, &Clamp).unwrap();
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
        let restored: Image<MonoF32, IdealPixels> =
            remap(&camera, &DestToSource(l), CatmullRom, &Clamp, ideal.size()).unwrap();
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
        let out: Image<Mono8> =
            remap(&img, &DestToSource(half), Bilinear, &Clamp, Size::new(4, 4)).unwrap();
        // Destination x = 1 reads the source at x = 0.5: between 5 and 15.
        assert_eq!(out.pixel_at(1, 0), Mono8::new(10));
        assert_eq!(out.pixel_at(2, 0), Mono8::new(15));
    }

    #[test]
    fn an_empty_source_fills_a_target_only_with_a_constant() {
        let empty: Image<MonoF32> = Image::zero(0, 0);
        let id: UniformScale<Pixels, Pixels> = crate::uniform_scale!(1.0);
        let target = Size::new(2, 2);
        let failed: Result<Image<MonoF32>, Error> =
            remap(&empty, &DestToSource(id), Bilinear, &Clamp, target);
        assert_eq!(failed, Err(Error::EmptySource { target }));
        let filled: Image<MonoF32> = remap(
            &empty,
            &DestToSource(id),
            Bilinear,
            &Constant(MonoF32::new(3.0)),
            target,
        )
        .unwrap();
        assert_eq!(filled, Image::fill(2, 2, MonoF32::new(3.0)));
        let table = DestToSourceTable::new(&DestToSource(id), target);
        let failed: Result<Image<MonoF32>, Error> = table.remap(&empty, Bilinear, &Clamp);
        assert_eq!(failed, Err(Error::EmptySource { target }));
        assert_eq!(
            table
                .remap(&empty, Bilinear, &Constant(MonoF32::new(3.0)))
                .unwrap(),
            filled
        );
    }

    #[test]
    fn an_empty_target_needs_no_source() {
        let empty: Image<MonoF32> = Image::zero(0, 0);
        let id: UniformScale<Pixels, Pixels> = crate::uniform_scale!(1.0);
        let out: Image<MonoF32> =
            remap(&empty, &DestToSource(id), Bilinear, &Clamp, Size::new(0, 3)).unwrap();
        assert_eq!(out.size(), Size::new(0, 3));
    }

    // ── Nearest neighbour and the output type ──────────────────────────

    #[test]
    fn nearest_neighbour_remaps_labels_masks_and_srgb() {
        use crate::pixel::{Label32, Srgb8};
        // Each destination pixel reads the source at twice its position.
        let half: UniformScale<Pixels, Pixels> = crate::uniform_scale!(2.0);

        let labels = Image::generate(4, 4, |x, y| Label32::new((4 * y + x) as u32));
        let out: Image<Label32> = remap(
            &labels,
            &DestToSource(half),
            NearestNeighbor,
            &Constant(Label32::new(0)),
            Size::new(2, 2),
        )
        .unwrap();
        assert_eq!(out.pixel_at(1, 1), Label32::new(10));

        let mask = Image::generate(4, 4, |x, _| x >= 2);
        let out: Image<bool> = remap(
            &mask,
            &DestToSource(half),
            NearestNeighbor,
            &Constant(false),
            Size::new(2, 2),
        )
        .unwrap();
        assert_eq!((out.pixel_at(0, 0), out.pixel_at(1, 0)), (false, true));

        // Gamma-encoded values are copied, never blended.
        let srgb = Image::generate(4, 1, |x, _| Srgb8::new(60 * x as u8, 0, 0));
        let out: Image<Srgb8> = remap(
            &srgb,
            &DestToSource(half),
            NearestNeighbor,
            &Constant(Srgb8::new(0, 0, 0)),
            Size::new(2, 1),
        )
        .unwrap();
        assert_eq!(out.pixel_at(1, 0), Srgb8::new(120, 0, 0));
    }

    #[test]
    fn nearest_neighbour_takes_the_pixel_whose_extent_holds_the_position() {
        let img = Image::generate(4, 1, |x, _| Mono8::new(10 * x as u8));
        // The extent of pixel k is [k − 0.5, k + 0.5): a source at x + 0.5
        // lies in pixel x + 1, one at x + 0.49 in pixel x.
        for (dx, expected) in [(0.5, [10, 20, 30, 99]), (0.49, [0, 10, 20, 30])] {
            let shift: Affine<Pixels, Pixels> =
                Affine::try_new([[1.0, 0.0], [0.0, 1.0]], Vector::new(dx, 0.0)).unwrap();
            let out: Image<Mono8> = remap(
                &img,
                &DestToSource(shift),
                NearestNeighbor,
                &Constant(Mono8::new(99)),
                Size::new(4, 1),
            )
            .unwrap();
            let got: Vec<u8> = (0..4).map(|x| out.pixel_at(x, 0).value()).collect();
            assert_eq!(got, expected, "shift {dx}");
        }
        // Outside the image the border policy answers, as for a kernel.
        let shift: Affine<Pixels, Pixels> =
            Affine::try_new([[1.0, 0.0], [0.0, 1.0]], Vector::new(2.0, 0.0)).unwrap();
        let out: Image<Mono8> = remap(
            &img,
            &DestToSource(shift),
            NearestNeighbor,
            &Clamp,
            Size::new(4, 1),
        )
        .unwrap();
        assert_eq!(out.pixel_at(3, 0), Mono8::new(30));
    }

    #[test]
    fn nearest_neighbour_without_a_source_reads_far_outside() {
        // A homography whose vanishing line is y = 4, inside the destination.
        let h: Homography<Pixels, Pixels> =
            Homography::try_new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -0.25, 1.0]]).unwrap();
        let img = Image::fill(16, 12, Mono8::new(5));
        let out: Image<Mono8> = remap(
            &img,
            &DestToSource(h),
            NearestNeighbor,
            &Constant(Mono8::new(77)),
            img.size(),
        )
        .unwrap();
        assert_eq!(out.pixel_at(5, 4), Mono8::new(77));
    }

    #[test]
    fn an_interpolating_remap_may_write_a_float_image() {
        let img = Image::generate(4, 1, |x, _| Mono8::new(10 * x as u8 + 5));
        let quarter: Affine<Pixels, Pixels> =
            Affine::try_new([[1.0, 0.0], [0.0, 1.0]], Vector::new(0.25, 0.0)).unwrap();
        let exact: Image<MonoF32> = remap(
            &img,
            &DestToSource(quarter),
            Bilinear,
            &Clamp,
            Size::new(4, 1),
        )
        .unwrap();
        // 5 · 0.75 + 15 · 0.25, kept rather than rounded into a byte.
        assert_eq!(exact.pixel_at(0, 0), MonoF32::new(7.5));
        let rounded: Image<Mono8> = remap(
            &img,
            &DestToSource(quarter),
            Bilinear,
            &Clamp,
            Size::new(4, 1),
        )
        .unwrap();
        assert_eq!(rounded.pixel_at(0, 0), Mono8::new(8));
    }

    #[test]
    fn a_remap_through_the_lens_returns_the_ideal_image() {
        let img = field();
        let ideal: Image<MonoF32, IdealPixels> = remap(
            &img,
            &DestToSource(lens()),
            NearestNeighbor,
            &Clamp,
            img.size(),
        )
        .unwrap();
        // The ideal image's space is the one the lens maps from, so the
        // lens reads it again as the domain of its own mapping.
        let p: Point<IdealPixels> = Point::new(3.0, 4.0);
        assert!(lens().try_map_point(p).is_some());
        assert_eq!(ideal.size(), img.size());
    }

    #[test]
    fn the_table_traits_ask_nothing_of_the_space() {
        let table = DestToSourceTable::new(&DestToSource(lens()), Size::new(4, 3));
        assert_eq!(format!("{:?}", table.clone()), format!("{table:?}"));
        assert!(format!("{table:?}").starts_with("DestToSourceTable { size:"));
    }
}

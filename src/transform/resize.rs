//! Resize images using pluggable [`ResizeMethod`] strategies.
//!
//! ## Which method?
//!
//! | Method | Pixel constraint | Use when |
//! |---|---|---|
//! | [`NearestNeighbor`] | `Copy` + `Into` target pixel | Speed matters or the source is gamma-encoded |
//! | an [`InterpolationKernel`]: [`Bilinear`], [`CatmullRom`], [`KeysBicubic`], [`Lanczos2`](type@Lanczos2), [`Lanczos3`](type@Lanczos3) | [`LinearSpace`](crate::pixel::LinearSpace) | Enlarging, or shrinking by less than about two |
//! | [`Antialiased`] around a kernel | [`LinearSpace`](crate::pixel::LinearSpace) | Shrinking by more than about two |
//!
//! A kernel on its own interpolates at a fixed width. Shrinking with it reads
//! only the few source pixels nearest to each target pixel, and detail finer
//! than the target grid comes back as a false coarser pattern (aliasing).
//! `Antialiased(kernel)` widens the kernel by the shrink factor, so each
//! target pixel averages everything it stands for; when enlarging it is the
//! bare kernel.
//!
//! **Geometry.** Pixel centres map onto pixel centres through the extent of
//! the image: target pixel `x` samples the source at
//! `(x + 0.5) · in / out − 0.5`, as OpenCV and Pillow do. Source pixels
//! beyond the edge repeat the edge pixel.
//!
//! **Important:** the kernels will not compile for `Srgb8` or any other gamma-encoded pixel type.
//! Interpolation blends neighboring samples; doing that in a non-linear encoding
//! produces subtly wrong results. Linearize first with
//! [`convert_image`](crate::transform::convert_image) + [`SrgbGamma`](crate::transform::SrgbGamma),
//! resize, then re-encode if needed.
//!
//! ```rust
//! use fovea::Size;
//! use fovea::image::{Image, ImageView};
//! use fovea::pixel::{RgbF32, Srgb8};
//! use fovea::transform::{SrgbGamma, Bilinear, NearestNeighbor, convert_image, resize};
//!
//! let srgb = Image::generate(4, 3, |x, y| Srgb8::new((x * 40) as u8, (y * 60) as u8, 128));
//!
//! // NearestNeighbor copies samples — works directly on gamma-encoded pixels.
//! let preview: Image<Srgb8> = resize(&srgb, Size::new(8, 6), NearestNeighbor)?;
//! assert_eq!(preview.size(), Size::new(8, 6));
//!
//! // Bilinear requires LinearSpace — linearize first.
//! let linear: Image<RgbF32> = convert_image(&srgb, SrgbGamma);
//! let resized: Image<RgbF32> = resize(&linear, Size::new(8, 6), Bilinear)?;
//! assert_eq!(resized.size(), Size::new(8, 6));
//! # Ok::<(), fovea::Error>(())
//! ```
//!
//! ## Implementing a custom resize strategy
//!
//! Implement [`ResizeMethod`] to plug in your own algorithm. The only requirement is to
//! fill `out` (pre-sized to the target dimensions) from `img`. The pixel-level constraints
//! live in each `impl` block, not in the trait itself, so you can express exactly the bounds
//! your algorithm needs.

use crate::image::{Image, ImageView, ImageViewMut};
use crate::pixel::{FromLinear, LinearPixel, LinearSpace, ZeroablePixel};
use crate::{Error, Size};

use super::interpolate::{Antialiased, AxisWeights, InterpolationKernel};
#[cfg(doc)]
use super::interpolate::{Bilinear, CatmullRom, KeysBicubic, Lanczos2, Lanczos3};

/// Trait for different resizing methods.
///
/// The `ResizeMethod` trait decouples the resizing algorithm from the resize function,
/// allowing for easy extension and customization of resizing strategies.
/// It also allows different restrictions for different methods.
///
/// Pixel-level constraints (e.g. `I::Pixel: Into<O::Pixel>` for nearest-neighbour,
/// or `I::Pixel: LinearPixel + LinearSpace` for bilinear) belong in the `impl`
/// blocks, not in the trait definition itself.
pub trait ResizeMethod<I: ImageView, O: ImageViewMut> {
    /// Resizes `img` into `out`, which must already have the desired target dimensions.
    fn resize_into(&self, img: &I, out: &mut O);
}

/// Nearest Neighbor resizing method
///
/// The NearestNeighbor struct implements the ResizeMethod trait using the nearest neighbor algorithm.
/// This method is fast and simple, but may produce blocky artifacts when enlarging images.
/// It is the resizing method with the least restrictions on pixel types.
///
/// Target pixel `x` copies the source pixel whose extent contains
/// `(x + 0.5) · in / out`, the same half-pixel geometry as the kernels.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NearestNeighbor;
impl<I, O> ResizeMethod<I, O> for NearestNeighbor
where
    I: ImageView,
    O: ImageViewMut,
    I::Pixel: Into<O::Pixel> + Copy,
{
    fn resize_into(&self, img: &I, out: &mut O) {
        resize_nearest_neighbor_into(img, out);
    }
}

/// Every interpolation kernel is a resize method: it interpolates at a fixed
/// width, which is right for enlarging and for shrinking by less than about
/// two. [`Antialiased`] is the method for stronger shrinking.
impl<K, I, O, Q> ResizeMethod<I, O> for K
where
    K: InterpolationKernel,
    I: ImageView,
    O: ImageViewMut,
    I::Pixel: LinearPixel<Accumulator = Q> + LinearSpace,
    Q: LinearPixel<Accumulator = Q>,
    O::Pixel: FromLinear<Q>,
{
    fn resize_into(&self, img: &I, out: &mut O) {
        resize_separable_into(self, img, out, false);
    }
}

/// Shrinking widens the kernel by the factor; enlarging uses it as it is.
impl<K, I, O, Q> ResizeMethod<I, O> for Antialiased<K>
where
    K: InterpolationKernel,
    I: ImageView,
    O: ImageViewMut,
    I::Pixel: LinearPixel<Accumulator = Q> + LinearSpace,
    Q: LinearPixel<Accumulator = Q>,
    O::Pixel: FromLinear<Q>,
{
    fn resize_into(&self, img: &I, out: &mut O) {
        resize_separable_into(&self.0, img, out, true);
    }
}

/// Resize an image into a pre-allocated output image using the specified method.
/// The output image must have the desired size.
///
/// # Type Parameters
/// - `I`: Input image type implementing [`ImageView`].
/// - `O`: Output image type implementing [`ImageViewMut`] (e.g. `Image`, `ImageArray`, or a mutable ROI).
/// - `M`: Resizing method implementing the [`ResizeMethod`] trait.
///
/// # Parameters
/// - `img`: Reference to the input image.
/// - `out`: Mutable reference to the output image or view.
/// - `method`: Resizing method to use (e.g., `NearestNeighbor`, `Bilinear`).
///
/// # Constraints
/// Specific pixel-level constraints depend on the chosen resize method:
/// - [`NearestNeighbor`] requires `I::Pixel: Into<O::Pixel> + Copy`.
/// - [`Bilinear`] requires `I::Pixel: LinearPixel + LinearSpace` and `O::Pixel: FromLinear`.
///
/// # Example
/// ```
/// # use fovea::image::Image;
/// # use fovea::pixel::MonoF32;
/// # use fovea::transform::{resize_into, NearestNeighbor, Bilinear};
/// // the pixel role for floats is `MonoF32`,
/// // not raw `f32`. `MonoF32` is `#[repr(transparent)]` over `f32`.
/// let img: Image<MonoF32> = Image::fill(300, 400, MonoF32::new(3.0)); // Input image
/// let mut out: Image<MonoF32> = Image::zero(100, 100); // Pre-allocated output image
/// resize_into(&img, &mut out, NearestNeighbor)?;
///
/// // or using bilinear interpolation
/// resize_into(&img, &mut out, Bilinear)?;
/// # Ok::<(), fovea::Error>(())
/// ```
///
/// # Example with more complex pixel types
/// ```
/// # use fovea::image::Image;
/// # use fovea::pixel::{Rgb8, RgbF32};
/// # use fovea::transform::{resize_into, NearestNeighbor, Bilinear};
/// let img: Image<Rgb8> = Image::fill(300, 400, Rgb8::new(255, 0, 0)); // Input image
/// let mut out: Image<Rgb8> = Image::zero(100, 100); // Pre-allocated output image
/// resize_into(&img, &mut out, NearestNeighbor)?;
///
/// // or using bilinear interpolation
/// resize_into(&img, &mut out, Bilinear)?;
/// # Ok::<(), fovea::Error>(())
/// ```
///
/// # Example with fix array images
/// ```
/// # use fovea::image::ImageArray;
/// # use fovea::pixel::Rgba16;
/// # use fovea::transform::{resize_into, NearestNeighbor, Bilinear};
/// let img: ImageArray<Rgba16, 3, 3> = ImageArray::generate(|x,y| Rgba16::new((y*10 + x) as u16, (y*10 + x) as u16, (y*10 + x) as u16, 65535));
/// let mut out: ImageArray<Rgba16, 2, 2> = ImageArray::generate(|_,_| Rgba16::new(0,0,0,0));
///
/// resize_into(&img, &mut out, NearestNeighbor)?;
///
/// // or using bilinear interpolation
/// resize_into(&img, &mut out, Bilinear)?;
/// # Ok::<(), fovea::Error>(())
/// ```
///
pub fn resize_into<I, O, M>(img: &I, out: &mut O, method: M) -> Result<(), Error>
where
    I: ImageView,
    O: ImageViewMut,
    M: ResizeMethod<I, O>,
{
    if img.size().area() == 0 && out.size().area() > 0 {
        return Err(Error::EmptySource { target: out.size() });
    }
    method.resize_into(img, out);
    Ok(())
}

/// Resizes `img` to `new_size`, allocating and returning a new output image.
///
/// Use [`NearestNeighbor`] for any pixel type when speed matters; use an
/// interpolation kernel such as [`Bilinear`] or [`Lanczos3`](type@Lanczos3) for photos and
/// camera frames, and wrap it in [`Antialiased`] when shrinking by more than
/// about two. The kernels require `I::Pixel: LinearSpace` to prevent subtly
/// incorrect results from gamma-encoded data.
///
/// To resize into an existing buffer, use [`resize_into`] instead.
///
/// # Errors
///
/// [`Error::EmptySource`] if `img` has no pixels and `new_size` has some:
/// there is no sample to resize from.
///
/// # Example
/// ```
/// # use fovea::image::{Image, ImageView};
/// # use fovea::pixel::Rgb8;
/// # use fovea::Size;
/// # use fovea::transform::{resize, NearestNeighbor};
/// let src: Image<Rgb8> = Image::fill(640, 480, Rgb8::new(128, 64, 32));
/// let dst: Image<Rgb8> = resize(&src, Size::new(320, 240), NearestNeighbor)?;
/// assert_eq!(dst.width(), 320);
/// assert_eq!(dst.height(), 240);
/// # Ok::<(), fovea::Error>(())
/// ```
pub fn resize<I, P, M>(img: &I, new_size: Size, method: M) -> Result<Image<P>, Error>
where
    I: ImageView,
    P: ZeroablePixel,
    M: ResizeMethod<I, Image<P>>,
{
    if img.size().area() == 0 && new_size.area() > 0 {
        return Err(Error::EmptySource { target: new_size });
    }
    let mut out = Image::<P>::zero(new_size.width, new_size.height);
    resize_into(img, &mut out, method)?;
    Ok(out)
}

fn resize_nearest_neighbor_into<I, O>(img: &I, out: &mut O)
where
    I: ImageView,
    O: ImageViewMut,
    I::Pixel: Into<O::Pixel> + Copy,
{
    let (src, dst) = (img.size(), out.size());
    if dst.width == 0 || dst.height == 0 {
        return;
    }
    assert_source_not_empty(src, dst);
    // The source pixel whose extent [k, k + 1) contains (x + 0.5) · in / out.
    let pick = |x: usize, in_len: usize, out_len: usize| {
        (((x as f64 + 0.5) * in_len as f64 / out_len as f64) as usize).min(in_len - 1)
    };
    for y in 0..dst.height {
        let src_y = pick(y, src.height, dst.height);
        for x in 0..dst.width {
            let src_x = pick(x, src.width, dst.width);
            *out.pixel_at_mut(x, y) = img.pixel_at(src_x, src_y).into();
        }
    }
}

fn assert_source_not_empty(src: Size, dst: Size) {
    assert!(
        src.width > 0 && src.height > 0,
        "resize: the source is {}x{}, so there is no sample to fill the {}x{} target from",
        src.width,
        src.height,
        dst.width,
        dst.height
    );
}

/// The separable kernel engine: a horizontal pass into accumulator rows at
/// the target width, then a vertical pass into the target.
fn resize_separable_into<K, I, O, Q>(kernel: &K, img: &I, out: &mut O, widen: bool)
where
    K: InterpolationKernel,
    I: ImageView,
    O: ImageViewMut,
    I::Pixel: LinearPixel<Accumulator = Q>,
    Q: LinearPixel<Accumulator = Q>,
    O::Pixel: FromLinear<Q>,
{
    let (src, dst) = (img.size(), out.size());
    if dst.width == 0 || dst.height == 0 {
        return;
    }
    assert_source_not_empty(src, dst);
    let columns = AxisWeights::resize(kernel, src.width, dst.width, widen);
    let rows = AxisWeights::resize(kernel, src.height, dst.height, widen);

    let mut wide: Vec<Q> = Vec::with_capacity(dst.width * src.height);
    for y in 0..src.height {
        for x in 0..dst.width {
            let (index, weight) = columns.taps(x);
            let mut acc = img.pixel_at(index[0], y).scale(weight[0]);
            for (&k, &w) in index[1..].iter().zip(&weight[1..]) {
                acc = img.pixel_at(k, y).scale_add(w, acc);
            }
            wide.push(acc);
        }
    }

    for y in 0..dst.height {
        let (index, weight) = rows.taps(y);
        for x in 0..dst.width {
            let mut acc = wide[index[0] * dst.width + x].scale(weight[0]);
            for (&k, &w) in index[1..].iter().zip(&weight[1..]) {
                acc = wide[k * dst.width + x].scale_add(w, acc);
            }
            *out.pixel_at_mut(x, y) = O::Pixel::from_linear(acc);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::Rectangle;
    use crate::image::{Image, ImageView, SubView, SubViewMut};
    use crate::pixel::{
        Bgr8, Bgr10, Bgr12, Bgr14, Bgr16, Bgr32, Bgr64, BgrF32, BgrF64, Bgra8, Bgra10, Bgra12,
        Bgra14, Bgra16, Bgra32, Bgra64, BgraF32, BgraF64, Mono8, Mono10, Mono12, Mono14, Mono16,
        Mono32, Mono64, MonoF32, MonoF64, Rgb8, Rgb10, Rgb12, Rgb14, Rgb16, Rgb32, Rgb64, RgbF32,
        RgbF64, Rgba8, Rgba10, Rgba12, Rgba14, Rgba16, Rgba32, Rgba64, RgbaF32, RgbaF64,
    };
    use crate::transform::{Bilinear, NearestNeighbor, resize, resize_into};

    #[test]
    fn an_empty_source_fills_no_target() {
        let empty = Image::<MonoF32>::zero(0, 3);
        let target = crate::Size::new(4, 2);
        assert_eq!(
            resize::<_, MonoF32, _>(&empty, target, Bilinear),
            Err(crate::Error::EmptySource { target })
        );
        let mut out = Image::<MonoF32>::zero(4, 2);
        assert_eq!(
            resize_into(&empty, &mut out, NearestNeighbor),
            Err(crate::Error::EmptySource { target })
        );
        // An empty target needs no source.
        let none: Image<MonoF32> = resize(&empty, crate::Size::new(0, 5), Bilinear).unwrap();
        assert_eq!(none.size(), crate::Size::new(0, 5));
    }

    macro_rules! resize_test {
        ($name:ident, $tp:ty,  $method:ident) => {
            #[test]
            fn $name() {
                let img: Image<$tp> = Image::zero(3, 3);
                let resized: Image<$tp> = resize(
                    &img,
                    crate::Size {
                        width: 2,
                        height: 2,
                    },
                    $method,
                )
                .unwrap();

                assert_eq!(resized.size().width, 2);
                assert_eq!(resized.size().height, 2);
            }
        };
    }

    resize_test!(test_resize_f32, MonoF32, NearestNeighbor);
    resize_test!(test_resize_bilinear_f32, MonoF32, Bilinear);
    resize_test!(test_resize_f64, MonoF64, NearestNeighbor);
    resize_test!(test_resize_bilinear_f64, MonoF64, Bilinear);
    resize_test!(test_resize_mono8, Mono8, NearestNeighbor);
    resize_test!(test_resize_bilinear_mono8, Mono8, Bilinear);
    resize_test!(test_resize_mono10, Mono10, NearestNeighbor);
    resize_test!(test_resize_bilinear_mono10, Mono10, Bilinear);
    resize_test!(test_resize_mono12, Mono12, NearestNeighbor);
    resize_test!(test_resize_bilinear_mono12, Mono12, Bilinear);
    resize_test!(test_resize_mono14, Mono14, NearestNeighbor);
    resize_test!(test_resize_bilinear_mono14, Mono14, Bilinear);
    resize_test!(test_resize_mono16, Mono16, NearestNeighbor);
    resize_test!(test_resize_bilinear_mono16, Mono16, Bilinear);
    resize_test!(test_resize_mono32, Mono32, NearestNeighbor);
    resize_test!(test_resize_bilinear_mono32, Mono32, Bilinear);
    resize_test!(test_resize_mono64, Mono64, NearestNeighbor);
    resize_test!(test_resize_bilinear_mono64, Mono64, Bilinear);
    resize_test!(test_resize_rgb8, Rgb8, NearestNeighbor);
    resize_test!(test_resize_bilinear_rgb8, Rgb8, Bilinear);
    resize_test!(test_resize_rgb10, Rgb10, NearestNeighbor);
    resize_test!(test_resize_bilinear_rgb10, Rgb10, Bilinear);
    resize_test!(test_resize_rgb12, Rgb12, NearestNeighbor);
    resize_test!(test_resize_bilinear_rgb12, Rgb12, Bilinear);
    resize_test!(test_resize_rgb14, Rgb14, NearestNeighbor);
    resize_test!(test_resize_bilinear_rgb14, Rgb14, Bilinear);
    resize_test!(test_resize_rgb16, Rgb16, NearestNeighbor);
    resize_test!(test_resize_bilinear_rgb16, Rgb16, Bilinear);
    resize_test!(test_resize_rgb32, Rgb32, NearestNeighbor);
    resize_test!(test_resize_bilinear_rgb32, Rgb32, Bilinear);
    resize_test!(test_resize_rgb64, Rgb64, NearestNeighbor);
    resize_test!(test_resize_bilinear_rgb64, Rgb64, Bilinear);
    resize_test!(test_resize_rgbf32, RgbF32, NearestNeighbor);
    resize_test!(test_resize_bilinear_rgbf32, RgbF32, Bilinear);
    resize_test!(test_resize_rgbf64, RgbF64, NearestNeighbor);
    resize_test!(test_resize_bilinear_rgbf64, RgbF64, Bilinear);
    resize_test!(test_resize_rgba8, Rgba8, NearestNeighbor);
    resize_test!(test_resize_bilinear_rgba8, Rgba8, Bilinear);
    resize_test!(test_resize_rgba10, Rgba10, NearestNeighbor);
    resize_test!(test_resize_bilinear_rgba10, Rgba10, Bilinear);
    resize_test!(test_resize_rgba12, Rgba12, NearestNeighbor);
    resize_test!(test_resize_bilinear_rgba12, Rgba12, Bilinear);
    resize_test!(test_resize_rgba14, Rgba14, NearestNeighbor);
    resize_test!(test_resize_bilinear_rgba14, Rgba14, Bilinear);
    resize_test!(test_resize_rgba16, Rgba16, NearestNeighbor);
    resize_test!(test_resize_bilinear_rgba16, Rgba16, Bilinear);
    resize_test!(test_resize_rgba32, Rgba32, NearestNeighbor);
    resize_test!(test_resize_bilinear_rgba32, Rgba32, Bilinear);
    resize_test!(test_resize_rgba64, Rgba64, NearestNeighbor);
    resize_test!(test_resize_bilinear_rgba64, Rgba64, Bilinear);
    resize_test!(test_resize_rgbaf32, RgbaF32, NearestNeighbor);
    resize_test!(test_resize_bilinear_rgbaf32, RgbaF32, Bilinear);
    resize_test!(test_resize_rgbaf64, RgbaF64, NearestNeighbor);
    resize_test!(test_resize_bilinear_rgbaf64, RgbaF64, Bilinear);
    resize_test!(test_resize_bgr8, Bgr8, NearestNeighbor);
    resize_test!(test_resize_bilinear_bgr8, Bgr8, Bilinear);
    resize_test!(test_resize_bgr10, Bgr10, NearestNeighbor);
    resize_test!(test_resize_bilinear_bgr10, Bgr10, Bilinear);
    resize_test!(test_resize_bgr12, Bgr12, NearestNeighbor);
    resize_test!(test_resize_bilinear_bgr12, Bgr12, Bilinear);
    resize_test!(test_resize_bgr14, Bgr14, NearestNeighbor);
    resize_test!(test_resize_bilinear_bgr14, Bgr14, Bilinear);
    resize_test!(test_resize_bgr16, Bgr16, NearestNeighbor);
    resize_test!(test_resize_bilinear_bgr16, Bgr16, Bilinear);
    resize_test!(test_resize_bgr32, Bgr32, NearestNeighbor);
    resize_test!(test_resize_bilinear_bgr32, Bgr32, Bilinear);
    resize_test!(test_resize_bgr64, Bgr64, NearestNeighbor);
    resize_test!(test_resize_bilinear_bgr64, Bgr64, Bilinear);
    resize_test!(test_resize_bgrf32, BgrF32, NearestNeighbor);
    resize_test!(test_resize_bilinear_bgrf32, BgrF32, Bilinear);
    resize_test!(test_resize_bgrf64, BgrF64, NearestNeighbor);
    resize_test!(test_resize_bilinear_bgrf64, BgrF64, Bilinear);
    resize_test!(test_resize_bgra8, Bgra8, NearestNeighbor);
    resize_test!(test_resize_bilinear_bgra8, Bgra8, Bilinear);
    resize_test!(test_resize_bgra10, Bgra10, NearestNeighbor);
    resize_test!(test_resize_bilinear_bgra10, Bgra10, Bilinear);
    resize_test!(test_resize_bgra12, Bgra12, NearestNeighbor);
    resize_test!(test_resize_bilinear_bgra12, Bgra12, Bilinear);
    resize_test!(test_resize_bgra14, Bgra14, NearestNeighbor);
    resize_test!(test_resize_bilinear_bgra14, Bgra14, Bilinear);
    resize_test!(test_resize_bgra16, Bgra16, NearestNeighbor);
    resize_test!(test_resize_bilinear_bgra16, Bgra16, Bilinear);
    resize_test!(test_resize_bgra32, Bgra32, NearestNeighbor);
    resize_test!(test_resize_bilinear_bgra32, Bgra32, Bilinear);
    resize_test!(test_resize_bgra64, Bgra64, NearestNeighbor);
    resize_test!(test_resize_bilinear_bgra64, Bgra64, Bilinear);
    resize_test!(test_resize_bgraf32, BgraF32, NearestNeighbor);
    resize_test!(test_resize_bilinear_bgraf32, BgraF32, Bilinear);
    resize_test!(test_resize_bgraf64, BgraF64, NearestNeighbor);
    resize_test!(test_resize_bilinear_bgraf64, BgraF64, Bilinear);

    #[test]
    fn test_downsize_nearest_neighbor() {
        let img_u8: Image<Mono8> = Image::generate(3, 3, |x, y| Mono8::new((y * 10 + x) as u8));
        let resized: Image<Mono8> = resize(
            &img_u8,
            crate::Size {
                width: 2,
                height: 2,
            },
            NearestNeighbor,
        )
        .unwrap();

        // Input image:
        // 0 1 2
        // 10 11 12
        // 20 21 22
        // Resized image (2x2) using nearest neighbor: target x copies the
        // source pixel containing (x + 0.5) * 1.5, which is 0 and 2:
        // 0 2
        // 20 22
        assert_eq!(resized.get(0, 0).unwrap(), Mono8::new(0));
        assert_eq!(resized.get(1, 0).unwrap(), Mono8::new(2));
        assert_eq!(resized.get(0, 1).unwrap(), Mono8::new(20));
        assert_eq!(resized.get(1, 1).unwrap(), Mono8::new(22));

        let img: Image<MonoF32> = Image::generate(3, 3, |x, y| MonoF32::new((y * 10 + x) as f32));

        let resized: Image<MonoF32> = resize(
            &img,
            crate::Size {
                width: 2,
                height: 2,
            },
            NearestNeighbor,
        )
        .unwrap();

        assert_eq!(resized.get(0, 0).unwrap(), MonoF32::new(0.0));
        assert_eq!(resized.get(1, 0).unwrap(), MonoF32::new(2.0));
        assert_eq!(resized.get(0, 1).unwrap(), MonoF32::new(20.0));
        assert_eq!(resized.get(1, 1).unwrap(), MonoF32::new(22.0));

        let img: Image<Rgb8> = Image::generate(3, 3, |x, y| {
            Rgb8::new((y * 10 + x) as u8, (y * 10 + x) as u8, (y * 10 + x) as u8)
        });
        let resized: Image<Rgb8> = resize(
            &img,
            crate::Size {
                width: 2,
                height: 2,
            },
            NearestNeighbor,
        )
        .unwrap();

        // Input image:
        assert_eq!(resized.get(0, 0).unwrap(), Rgb8::new(0, 0, 0));
        assert_eq!(resized.get(1, 0).unwrap(), Rgb8::new(2, 2, 2));
        assert_eq!(resized.get(0, 1).unwrap(), Rgb8::new(20, 20, 20));
        assert_eq!(resized.get(1, 1).unwrap(), Rgb8::new(22, 22, 22));

        let img: Image<RgbF32> = Image::generate(3, 3, |x, y| {
            RgbF32::new(
                (y * 10 + x) as f32,
                (y * 10 + x) as f32,
                (y * 10 + x) as f32,
            )
        });
        let resized: Image<RgbF32> = resize(
            &img,
            crate::Size {
                width: 2,
                height: 2,
            },
            NearestNeighbor,
        )
        .unwrap();

        assert_eq!(resized.get(0, 0).unwrap(), RgbF32::new(0.0, 0.0, 0.0));
        assert_eq!(resized.get(1, 0).unwrap(), RgbF32::new(2.0, 2.0, 2.0));
        assert_eq!(resized.get(0, 1).unwrap(), RgbF32::new(20.0, 20.0, 20.0));
        assert_eq!(resized.get(1, 1).unwrap(), RgbF32::new(22.0, 22.0, 22.0));
    }

    #[test]
    fn test_upsize_nearest_neighbor() {
        let img: Image<MonoF32> = Image::generate(2, 2, |x, y| MonoF32::new((y * 10 + x) as f32));

        let resized: Image<MonoF32> = resize(
            &img,
            crate::Size {
                width: 3,
                height: 3,
            },
            NearestNeighbor,
        )
        .unwrap();

        // Input image:
        // 0 1
        // 10 11

        // Resized image (3x3) using nearest neighbor: target x copies the
        // source pixel containing (x + 0.5) * 2 / 3, which is 0, 1, 1:
        // 0 1 1
        // 10 11 11
        // 10 11 11

        assert_eq!(resized.get(0, 0).unwrap(), MonoF32::new(0.0));
        assert_eq!(resized.get(1, 0).unwrap(), MonoF32::new(1.0));
        assert_eq!(resized.get(2, 0).unwrap(), MonoF32::new(1.0));
        assert_eq!(resized.get(0, 1).unwrap(), MonoF32::new(10.0));
        assert_eq!(resized.get(1, 1).unwrap(), MonoF32::new(11.0));
        assert_eq!(resized.get(2, 1).unwrap(), MonoF32::new(11.0));
        assert_eq!(resized.get(0, 2).unwrap(), MonoF32::new(10.0));
        assert_eq!(resized.get(1, 2).unwrap(), MonoF32::new(11.0));
        assert_eq!(resized.get(2, 2).unwrap(), MonoF32::new(11.0));

        let img_u8: Image<u8> = Image::generate(2, 2, |x, y| (y * 10 + x) as u8);

        let resized_u8: Image<u8> = resize(
            &img_u8,
            crate::Size {
                width: 3,
                height: 3,
            },
            NearestNeighbor,
        )
        .unwrap();

        assert_eq!(resized_u8.get(0, 0).unwrap(), 0);
        assert_eq!(resized_u8.get(1, 0).unwrap(), 1);
        assert_eq!(resized_u8.get(2, 0).unwrap(), 1);
        assert_eq!(resized_u8.get(0, 1).unwrap(), 10);
        assert_eq!(resized_u8.get(1, 1).unwrap(), 11);
        assert_eq!(resized_u8.get(2, 1).unwrap(), 11);
        assert_eq!(resized_u8.get(0, 2).unwrap(), 10);
        assert_eq!(resized_u8.get(1, 2).unwrap(), 11);
        assert_eq!(resized_u8.get(2, 2).unwrap(), 11);

        let img: Image<Rgb8> = Image::generate(2, 2, |x, y| {
            Rgb8::new((y * 10 + x) as u8, (y * 10 + x) as u8, (y * 10 + x) as u8)
        });

        let resized: Image<Rgb8> = resize(
            &img,
            crate::Size {
                width: 3,
                height: 3,
            },
            NearestNeighbor,
        )
        .unwrap();

        assert_eq!(resized.get(0, 0).unwrap(), Rgb8::new(0, 0, 0));
        assert_eq!(resized.get(1, 0).unwrap(), Rgb8::new(1, 1, 1));
        assert_eq!(resized.get(2, 0).unwrap(), Rgb8::new(1, 1, 1));
        assert_eq!(resized.get(0, 1).unwrap(), Rgb8::new(10, 10, 10));
        assert_eq!(resized.get(1, 1).unwrap(), Rgb8::new(11, 11, 11));
        assert_eq!(resized.get(2, 1).unwrap(), Rgb8::new(11, 11, 11));
        assert_eq!(resized.get(0, 2).unwrap(), Rgb8::new(10, 10, 10));
        assert_eq!(resized.get(1, 2).unwrap(), Rgb8::new(11, 11, 11));
        assert_eq!(resized.get(2, 2).unwrap(), Rgb8::new(11, 11, 11));

        let img: Image<RgbF32> = Image::generate(2, 2, |x, y| {
            RgbF32::new(
                (y * 10 + x) as f32,
                (y * 10 + x) as f32,
                (y * 10 + x) as f32,
            )
        });
        let resized: Image<RgbF32> = resize(
            &img,
            crate::Size {
                width: 3,
                height: 3,
            },
            NearestNeighbor,
        )
        .unwrap();

        assert_eq!(resized.get(0, 0).unwrap(), RgbF32::new(0.0, 0.0, 0.0));
        assert_eq!(resized.get(1, 0).unwrap(), RgbF32::new(1.0, 1.0, 1.0));
        assert_eq!(resized.get(2, 0).unwrap(), RgbF32::new(1.0, 1.0, 1.0));
        assert_eq!(resized.get(0, 1).unwrap(), RgbF32::new(10.0, 10.0, 10.0));
        assert_eq!(resized.get(1, 1).unwrap(), RgbF32::new(11.0, 11.0, 11.0));
        assert_eq!(resized.get(2, 1).unwrap(), RgbF32::new(11.0, 11.0, 11.0));
        assert_eq!(resized.get(0, 2).unwrap(), RgbF32::new(10.0, 10.0, 10.0));
        assert_eq!(resized.get(1, 2).unwrap(), RgbF32::new(11.0, 11.0, 11.0));
        assert_eq!(resized.get(2, 2).unwrap(), RgbF32::new(11.0, 11.0, 11.0));
    }

    #[test]
    fn test_downsize_bilinear() {
        // Input image:
        // 0 1 2
        // 10 11 12
        // 20 21 22

        // Resized image (2x2) using bilinear interpolation: the target
        // centres sit at (x + 0.5) * 1.5 - 0.5 = 0.25 and 1.75 on each axis,
        // so the values are 10 * y + x at those positions:
        // 2.75 4.25
        // 17.75 19.25

        let img: Image<Mono8> = Image::generate(3, 3, |x, y| Mono8::new((y * 10 + x) as u8));

        let mut resized = Image::<Mono8>::zero(2, 2);

        resize_into(&img, &mut resized, Bilinear).unwrap();

        assert_eq!(resized.get(0, 0).unwrap(), Mono8::new(3));
        assert_eq!(resized.get(1, 0).unwrap(), Mono8::new(4));
        assert_eq!(resized.get(0, 1).unwrap(), Mono8::new(18));
        assert_eq!(resized.get(1, 1).unwrap(), Mono8::new(19));

        let img: Image<MonoF32> = Image::generate(3, 3, |x, y| MonoF32::new((y * 10 + x) as f32));

        let resized: Image<MonoF32> = resize(
            &img,
            crate::Size {
                width: 2,
                height: 2,
            },
            Bilinear,
        )
        .unwrap();

        assert_eq!(resized.get(0, 0).unwrap(), MonoF32::new(2.75));
        assert_eq!(resized.get(1, 0).unwrap(), MonoF32::new(4.25));
        assert_eq!(resized.get(0, 1).unwrap(), MonoF32::new(17.75));
        assert_eq!(resized.get(1, 1).unwrap(), MonoF32::new(19.25));
    }

    #[test]
    fn test_upsize_bilinear() {
        let img: Image<MonoF32> = Image::generate(2, 2, |x, y| MonoF32::new((y * 10 + x) as f32));

        let resized: Image<MonoF32> = resize(
            &img,
            crate::Size {
                width: 3,
                height: 3,
            },
            Bilinear,
        )
        .unwrap();

        // Input image:
        // 0 1
        // 10 11

        // Resized image (3x3) using bilinear interpolation:
        // 0 0.5 1
        // 5 5.5 6
        // 10 10.5 11

        assert_eq!(resized.get(0, 0).unwrap(), MonoF32::new(0.0));
        assert_eq!(resized.get(1, 0).unwrap(), MonoF32::new(0.5)); // Approximation of 0.5
        assert_eq!(resized.get(2, 0).unwrap(), MonoF32::new(1.0));
        assert_eq!(resized.get(0, 1).unwrap(), MonoF32::new(5.0)); // Approximation of 5
        assert_eq!(resized.get(1, 1).unwrap(), MonoF32::new(5.5)); // Approximation of 5.5
        assert_eq!(resized.get(2, 1).unwrap(), MonoF32::new(6.0)); // Approximation of 6
        assert_eq!(resized.get(0, 2).unwrap(), MonoF32::new(10.0));
        assert_eq!(resized.get(1, 2).unwrap(), MonoF32::new(10.5)); // Approximation of 10.5
        assert_eq!(resized.get(2, 2).unwrap(), MonoF32::new(11.0));
    }

    #[test]
    fn test_resize_complex_pixel_types() {
        let img: Image<Rgb8> = Image::generate(3, 3, |x, y| {
            Rgb8::new((y * 10 + x) as u8, (y * 10 + x) as u8, (y * 10 + x) as u8)
        });
        let resized: Image<Rgb8> = resize(
            &img,
            crate::Size {
                width: 2,
                height: 2,
            },
            NearestNeighbor,
        )
        .unwrap();

        // Input image:
        assert_eq!(resized.get(0, 0).unwrap(), Rgb8::new(0, 0, 0));
        assert_eq!(resized.get(1, 0).unwrap(), Rgb8::new(2, 2, 2));
        assert_eq!(resized.get(0, 1).unwrap(), Rgb8::new(20, 20, 20));
        assert_eq!(resized.get(1, 1).unwrap(), Rgb8::new(22, 22, 22));

        let resized: Image<Rgb8> = resize(
            &img,
            crate::Size {
                width: 2,
                height: 2,
            },
            Bilinear,
        )
        .unwrap();

        // 2.75, 4.25, 17.75 and 19.25, rounded (see test_downsize_bilinear).
        assert_eq!(resized.get(0, 0).unwrap(), Rgb8::new(3, 3, 3));
        assert_eq!(resized.get(1, 0).unwrap(), Rgb8::new(4, 4, 4));
        assert_eq!(resized.get(0, 1).unwrap(), Rgb8::new(18, 18, 18));
        assert_eq!(resized.get(1, 1).unwrap(), Rgb8::new(19, 19, 19));
    }

    #[test]
    fn test_resize_roi_with_complex_pixel_types() {
        let img: Image<Rgb8> = Image::generate(4, 4, |x, y| {
            Rgb8::new((y * 10 + x) as u8, (y * 10 + x) as u8, (y * 10 + x) as u8)
        });
        let roi = img.roi(Rectangle::new((1, 1), (2, 2))).unwrap();
        let resized: Image<Rgb8> = resize(
            &roi,
            crate::Size {
                width: 2,
                height: 2,
            },
            NearestNeighbor,
        )
        .unwrap();

        // Input image:
        // 0 1 2 3
        // 10 11 12 13
        // 20 21 22 23
        // 30 31 32 33

        // ROI (1,1) to (3,3):
        // 11 12
        // 21 22

        assert_eq!(resized.get(0, 0).unwrap(), Rgb8::new(11, 11, 11));
        assert_eq!(resized.get(1, 0).unwrap(), Rgb8::new(12, 12, 12));
        assert_eq!(resized.get(0, 1).unwrap(), Rgb8::new(21, 21, 21));
        assert_eq!(resized.get(1, 1).unwrap(), Rgb8::new(22, 22, 22));
    }

    // -----------------------------------------------------------------------
    // ROI-as-output tests — writing resize results into a mutable ROI
    // -----------------------------------------------------------------------

    #[test]
    fn test_resize_nearest_neighbor_into_roi_output() {
        // Resize a 4x4 source into a 2x2 ROI within a 4x4 target
        let src: Image<Mono8> = Image::generate(4, 4, |x, y| Mono8::new((y * 10 + x) as u8));
        let mut target: Image<Mono8> = Image::fill(4, 4, Mono8::new(255));

        {
            let mut roi_out = target.roi_mut(Rectangle::new((1, 1), (2, 2))).unwrap();
            resize_into(&src, &mut roi_out, NearestNeighbor).unwrap();
        }

        // The ROI region should contain the resized result
        // Target x copies the source pixel containing (x + 0.5) * 2: 1 and 3.
        assert_eq!(target.get(1, 1).unwrap(), Mono8::new(11)); // src (1,1)
        assert_eq!(target.get(2, 1).unwrap(), Mono8::new(13)); // src (3,1)
        assert_eq!(target.get(1, 2).unwrap(), Mono8::new(31)); // src (1,3)
        assert_eq!(target.get(2, 2).unwrap(), Mono8::new(33)); // src (3,3)

        // Outside the ROI should be untouched
        assert_eq!(target.get(0, 0).unwrap(), Mono8::new(255));
        assert_eq!(target.get(3, 3).unwrap(), Mono8::new(255));
    }

    #[test]
    fn test_resize_bilinear_into_roi_output() {
        // Resize a 2x2 f32 source into a 2x2 ROI within a 4x4 target
        let src: Image<MonoF32> = Image::generate(2, 2, |x, y| MonoF32::new((y * 10 + x) as f32));
        let mut target: Image<MonoF32> = Image::fill(4, 4, MonoF32::new(-1.0));

        {
            let mut roi_out = target.roi_mut(Rectangle::new((0, 0), (2, 2))).unwrap();
            resize_into(&src, &mut roi_out, Bilinear).unwrap();
        }

        // Same size, so values should match the source exactly
        assert_eq!(target.get(0, 0).unwrap(), MonoF32::new(0.0));
        assert_eq!(target.get(1, 0).unwrap(), MonoF32::new(1.0));
        assert_eq!(target.get(0, 1).unwrap(), MonoF32::new(10.0));
        assert_eq!(target.get(1, 1).unwrap(), MonoF32::new(11.0));

        // Outside the ROI should be untouched
        assert_eq!(target.get(2, 0).unwrap(), MonoF32::new(-1.0));
        assert_eq!(target.get(0, 2).unwrap(), MonoF32::new(-1.0));
    }

    #[test]
    fn test_resize_roi_input_to_roi_output() {
        // Read from an ROI, resize into a different ROI
        let src: Image<Rgb8> = Image::generate(6, 6, |x, y| {
            Rgb8::new((y * 10 + x) as u8, (y * 10 + x) as u8, (y * 10 + x) as u8)
        });
        let roi_in = src.roi(Rectangle::new((2, 2), (4, 4))).unwrap();

        // roi_in is a 4x4 region starting at (2,2):
        // 22 23 24 25
        // 32 33 34 35
        // 42 43 44 45
        // 52 53 54 55

        let mut target: Image<Rgb8> = Image::zero(4, 4);
        {
            let mut roi_out = target.roi_mut(Rectangle::new((0, 0), (2, 2))).unwrap();
            resize_into(&roi_in, &mut roi_out, NearestNeighbor).unwrap();
        }

        // Resized from 4x4 to 2x2 nearest neighbor picks ROI pixels 1 and 3
        assert_eq!(target.get(0, 0).unwrap(), Rgb8::new(33, 33, 33));
        assert_eq!(target.get(1, 0).unwrap(), Rgb8::new(35, 35, 35));
        assert_eq!(target.get(0, 1).unwrap(), Rgb8::new(53, 53, 53));
        assert_eq!(target.get(1, 1).unwrap(), Rgb8::new(55, 55, 55));

        // Rest should be zero
        assert_eq!(target.get(2, 0).unwrap(), Rgb8::new(0, 0, 0));
    }

    // -----------------------------------------------------------------------
    // 1x1 resize tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_resize_1x1_to_1x1_nearest_neighbor() {
        let img: Image<u8> = Image::generate(1, 1, |_, _| 42);
        let resized: Image<u8> = resize(
            &img,
            crate::Size {
                width: 1,
                height: 1,
            },
            NearestNeighbor,
        )
        .unwrap();
        assert_eq!(resized.get(0, 0).unwrap(), 42);
    }

    #[test]
    fn test_resize_1x1_to_1x1_bilinear() {
        let img: Image<MonoF32> = Image::generate(1, 1, |_, _| MonoF32::new(42.0));
        let resized: Image<MonoF32> = resize(
            &img,
            crate::Size {
                width: 1,
                height: 1,
            },
            Bilinear,
        )
        .unwrap();
        assert_eq!(resized.get(0, 0).unwrap(), MonoF32::new(42.0));
    }

    #[test]
    fn test_resize_3x3_to_1x1_nearest_neighbor() {
        let img: Image<u8> = Image::generate(3, 3, |x, y| (x + y * 3) as u8);
        let resized: Image<u8> = resize(
            &img,
            crate::Size {
                width: 1,
                height: 1,
            },
            NearestNeighbor,
        )
        .unwrap();
        // The single target pixel's centre is the source's centre, pixel (1, 1).
        assert_eq!(resized.get(0, 0).unwrap(), 4);
    }

    #[test]
    fn test_resize_3x3_to_1x1_bilinear() {
        let img: Image<MonoF32> = Image::generate(3, 3, |x, y| MonoF32::new((x + y * 3) as f32));
        let resized: Image<MonoF32> = resize(
            &img,
            crate::Size {
                width: 1,
                height: 1,
            },
            Bilinear,
        )
        .unwrap();
        // Centre onto centre: the source's middle pixel, exactly.
        assert_eq!(resized.get(0, 0).unwrap(), MonoF32::new(4.0));
    }

    #[test]
    fn test_resize_1x1_to_3x3_nearest_neighbor() {
        let img: Image<u8> = Image::generate(1, 1, |_, _| 77);
        let resized: Image<u8> = resize(
            &img,
            crate::Size {
                width: 3,
                height: 3,
            },
            NearestNeighbor,
        )
        .unwrap();
        // All pixels should be 77 since the source is a single pixel
        for y in 0..3 {
            for x in 0..3 {
                assert_eq!(resized.get(x, y).unwrap(), 77);
            }
        }
    }

    #[test]
    fn test_resize_1x1_to_3x3_bilinear() {
        let img: Image<MonoF32> = Image::generate(1, 1, |_, _| MonoF32::new(77.0));
        let resized: Image<MonoF32> = resize(
            &img,
            crate::Size {
                width: 3,
                height: 3,
            },
            Bilinear,
        )
        .unwrap();
        for y in 0..3 {
            for x in 0..3 {
                assert_eq!(resized.get(x, y).unwrap(), MonoF32::new(77.0));
            }
        }
    }
}

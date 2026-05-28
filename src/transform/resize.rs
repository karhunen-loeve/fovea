use crate::Size;
use crate::image::{Image, ImageView, ImageViewMut};
use crate::pixel::{FromLinear, LinearPixel, LinearSpace, ZeroablePixel, blend};

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
    fn resize_into(&self, img: &I, out: &mut O);
}

/// Nearest Neighbor resizing method
///
/// The NearestNeighbor struct implements the ResizeMethod trait using the nearest neighbor algorithm.
/// This method is fast and simple, but may produce blocky artifacts when enlarging images.
/// It is the resizing method with the least restrictions on pixel types.
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

/// Bilinear resizing method
///
/// The Bilinear struct implements the ResizeMethod trait using the bilinear interpolation algorithm.
/// This method provides smoother results than nearest neighbor, especially when enlarging images.
/// However, it requires that the pixel types support addition and multiplication operations
/// **and** that the pixel values live in a linear space where interpolation is meaningful.
///
/// Types that represent gamma-encoded data (e.g. [`Srgb8`](crate::pixel::Srgb8),
/// [`Srgba8`](crate::pixel::Srgba8)) intentionally do *not* implement
/// [`LinearSpace`](crate::pixel::LinearSpace) and will be rejected at compile time.
/// Convert to linear light first (e.g. via
/// [`SrgbGamma`](crate::transform::SrgbGamma)) before resizing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Bilinear;
impl<I, O, Q> ResizeMethod<I, O> for Bilinear
where
    I: ImageView,
    O: ImageViewMut,
    I::Pixel: LinearPixel<Accumulator = Q> + LinearSpace,
    Q: LinearPixel<Accumulator = Q> + LinearSpace,
    O::Pixel: FromLinear<Q>,
{
    fn resize_into(&self, img: &I, out: &mut O) {
        resize_bilinear_into(img, out);
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
/// # use irys_cv::image::Image;
/// # use irys_cv::pixel::MonoF32;
/// # use irys_cv::transform::{resize_into, NearestNeighbor, Bilinear};
/// // ADR-0044 Phase E: the pixel role for floats is `MonoF32`,
/// // not raw `f32`. `MonoF32` is `#[repr(transparent)]` over `f32`.
/// let img: Image<MonoF32> = Image::fill(300, 400, MonoF32::new(3.0)); // Input image
/// let mut out: Image<MonoF32> = Image::zero(100, 100); // Pre-allocated output image
/// resize_into(&img, &mut out, NearestNeighbor);
///
/// // or using bilinear interpolation
/// resize_into(&img, &mut out, Bilinear);
/// ```
///
/// # Example with more complex pixel types
/// ```
/// # use irys_cv::image::Image;
/// # use irys_cv::pixel::{Rgb8, RgbF32};
/// # use irys_cv::transform::{resize_into, NearestNeighbor, Bilinear};
/// let img: Image<Rgb8> = Image::fill(300, 400, Rgb8::new(255, 0, 0)); // Input image
/// let mut out: Image<Rgb8> = Image::zero(100, 100); // Pre-allocated output image
/// resize_into(&img, &mut out, NearestNeighbor);
///
/// // or using bilinear interpolation
/// resize_into(&img, &mut out, Bilinear);
/// ```
///
/// # Example with fix array images
/// ```
/// # use irys_cv::image::ImageArray;
/// # use irys_cv::pixel::Rgba16;
/// # use irys_cv::transform::{resize_into, NearestNeighbor, Bilinear};
/// let img: ImageArray<Rgba16, 3, 3> = ImageArray::generate(|x,y| Rgba16::new((y*10 + x) as u16, (y*10 + x) as u16, (y*10 + x) as u16, 65535));
/// let mut out: ImageArray<Rgba16, 2, 2> = ImageArray::generate(|_,_| Rgba16::new(0,0,0,0));
///
/// resize_into(&img, &mut out, NearestNeighbor);
///
/// // or using bilinear interpolation
/// resize_into(&img, &mut out, Bilinear);
/// ```
///
pub fn resize_into<I, O, M>(img: &I, out: &mut O, method: M)
where
    I: ImageView,
    O: ImageViewMut,
    M: ResizeMethod<I, O>,
{
    method.resize_into(img, out)
}

#[must_use]
pub fn resize<I, P, M>(img: &I, new_size: Size, method: M) -> Image<P>
where
    I: ImageView,
    P: ZeroablePixel,
    M: ResizeMethod<I, Image<P>>,
{
    let mut out = Image::<P>::zero(new_size.width, new_size.height);
    resize_into(img, &mut out, method);
    out
}

fn resize_nearest_neighbor_into<I, O>(img: &I, out: &mut O)
where
    I: ImageView,
    O: ImageViewMut,
    I::Pixel: Into<O::Pixel> + Copy,
{
    let in_size = img.size();
    let out_size = out.size();
    let scale_x = if out_size.width <= 1 || in_size.width <= 1 {
        0.0
    } else {
        (in_size.width - 1) as f32 / (out_size.width - 1) as f32
    };
    let scale_y = if out_size.height <= 1 || in_size.height <= 1 {
        0.0
    } else {
        (in_size.height - 1) as f32 / (out_size.height - 1) as f32
    };

    for y in 0..out_size.height {
        for x in 0..out_size.width {
            let src_x = (x as f32 * scale_x) as usize;
            let src_y = (y as f32 * scale_y) as usize;
            let pixel = img.pixel_at(src_x, src_y);
            *out.pixel_at_mut(x, y) = pixel.into();
        }
    }
}

fn resize_bilinear_into<I, O, Q>(img: &I, out: &mut O)
where
    I: ImageView,
    O: ImageViewMut,
    I::Pixel: LinearPixel<Accumulator = Q> + LinearSpace,
    Q: LinearPixel<Accumulator = Q> + LinearSpace,
    O::Pixel: FromLinear<Q>,
{
    // Implement bilinear resizing logic here
    let in_size = img.size();
    let out_size = out.size();
    let scale_x = if out_size.width <= 1 || in_size.width <= 1 {
        0.0
    } else {
        (in_size.width - 1) as f32 / (out_size.width - 1) as f32
    };
    let scale_y = if out_size.height <= 1 || in_size.height <= 1 {
        0.0
    } else {
        (in_size.height - 1) as f32 / (out_size.height - 1) as f32
    };

    for y in 0..out_size.height {
        for x in 0..out_size.width {
            let src_x = x as f32 * scale_x;
            let src_y = y as f32 * scale_y;

            let x0 = src_x.floor() as usize;
            let x1 = (x0 + 1).min(in_size.width - 1);
            let y0 = src_y.floor() as usize;
            let y1 = (y0 + 1).min(in_size.height - 1);

            let dx = src_x - x0 as f32;
            let dy = src_y - y0 as f32;

            let p00 = img.pixel_at(x0, y0);
            let p10 = img.pixel_at(x1, y0);
            let p01 = img.pixel_at(x0, y1);
            let p11 = img.pixel_at(x1, y1);

            let p0 = blend(&p00, &p10, dx);
            let p1 = blend(&p01, &p11, dx);
            let p = blend(&p0, &p1, dy);

            let r = out.pixel_at_mut(x, y);
            *r = O::Pixel::from_linear(p);
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
                );

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
        );

        // Input image:
        // 0 1 2
        // 10 11 12
        // 20 21 22
        // Resized image (2x2) using nearest neighbor:
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
        );

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
        );

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
        );

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
        );

        // Input image:
        // 0 1
        // 10 11

        // Resized image (3x3) using nearest neighbor:
        // 0 0 1
        // 0 0 1
        // 10 10 11

        assert_eq!(resized.get(0, 0).unwrap(), MonoF32::new(0.0));
        assert_eq!(resized.get(1, 0).unwrap(), MonoF32::new(0.0));
        assert_eq!(resized.get(2, 0).unwrap(), MonoF32::new(1.0));
        assert_eq!(resized.get(0, 1).unwrap(), MonoF32::new(0.0));
        assert_eq!(resized.get(1, 1).unwrap(), MonoF32::new(0.0));
        assert_eq!(resized.get(2, 1).unwrap(), MonoF32::new(1.0));
        assert_eq!(resized.get(0, 2).unwrap(), MonoF32::new(10.0));
        assert_eq!(resized.get(1, 2).unwrap(), MonoF32::new(10.0));
        assert_eq!(resized.get(2, 2).unwrap(), MonoF32::new(11.0));

        let img_u8: Image<u8> = Image::generate(2, 2, |x, y| (y * 10 + x) as u8);

        let resized_u8: Image<u8> = resize(
            &img_u8,
            crate::Size {
                width: 3,
                height: 3,
            },
            NearestNeighbor,
        );

        assert_eq!(resized_u8.get(0, 0).unwrap(), 0);
        assert_eq!(resized_u8.get(1, 0).unwrap(), 0);
        assert_eq!(resized_u8.get(2, 0).unwrap(), 1);
        assert_eq!(resized_u8.get(0, 1).unwrap(), 0);
        assert_eq!(resized_u8.get(1, 1).unwrap(), 0);
        assert_eq!(resized_u8.get(2, 1).unwrap(), 1);
        assert_eq!(resized_u8.get(0, 2).unwrap(), 10);
        assert_eq!(resized_u8.get(1, 2).unwrap(), 10);
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
        );

        assert_eq!(resized.get(0, 0).unwrap(), Rgb8::new(0, 0, 0));
        assert_eq!(resized.get(1, 0).unwrap(), Rgb8::new(0, 0, 0));
        assert_eq!(resized.get(2, 0).unwrap(), Rgb8::new(1, 1, 1));
        assert_eq!(resized.get(0, 1).unwrap(), Rgb8::new(0, 0, 0));
        assert_eq!(resized.get(1, 1).unwrap(), Rgb8::new(0, 0, 0));
        assert_eq!(resized.get(2, 1).unwrap(), Rgb8::new(1, 1, 1));
        assert_eq!(resized.get(0, 2).unwrap(), Rgb8::new(10, 10, 10));
        assert_eq!(resized.get(1, 2).unwrap(), Rgb8::new(10, 10, 10));
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
        );

        assert_eq!(resized.get(0, 0).unwrap(), RgbF32::new(0.0, 0.0, 0.0));
        assert_eq!(resized.get(1, 0).unwrap(), RgbF32::new(0.0, 0.0, 0.0));
        assert_eq!(resized.get(2, 0).unwrap(), RgbF32::new(1.0, 1.0, 1.0));
        assert_eq!(resized.get(0, 1).unwrap(), RgbF32::new(0.0, 0.0, 0.0));
        assert_eq!(resized.get(1, 1).unwrap(), RgbF32::new(0.0, 0.0, 0.0));
        assert_eq!(resized.get(2, 1).unwrap(), RgbF32::new(1.0, 1.0, 1.0));
        assert_eq!(resized.get(0, 2).unwrap(), RgbF32::new(10.0, 10.0, 10.0));
        assert_eq!(resized.get(1, 2).unwrap(), RgbF32::new(10.0, 10.0, 10.0));
        assert_eq!(resized.get(2, 2).unwrap(), RgbF32::new(11.0, 11.0, 11.0));
    }

    #[test]
    fn test_downsize_bilinear() {
        // Input image:
        // 0 1 2
        // 10 11 120
        // 20 21 22

        // Resized image (2x2) using bilinear interpolation:
        // 0 2
        // 20 22

        let img: Image<Mono8> = Image::generate(3, 3, |x, y| Mono8::new((y * 10 + x) as u8));

        let mut resized = Image::<Mono8>::zero(2, 2);

        resize_into(&img, &mut resized, Bilinear);

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
            Bilinear,
        );

        assert_eq!(resized.get(0, 0).unwrap(), MonoF32::new(0.0));
        assert_eq!(resized.get(1, 0).unwrap(), MonoF32::new(2.0));
        assert_eq!(resized.get(0, 1).unwrap(), MonoF32::new(20.0));
        assert_eq!(resized.get(1, 1).unwrap(), MonoF32::new(22.0));
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
        );

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
        );

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
        );

        assert_eq!(resized.get(0, 0).unwrap(), Rgb8::new(0, 0, 0));
        assert_eq!(resized.get(1, 0).unwrap(), Rgb8::new(2, 2, 2));
        assert_eq!(resized.get(0, 1).unwrap(), Rgb8::new(20, 20, 20));
        assert_eq!(resized.get(1, 1).unwrap(), Rgb8::new(22, 22, 22));
        assert_eq!(resized.get(0, 0).unwrap(), Rgb8::new(0, 0, 0));
        assert_eq!(resized.get(1, 0).unwrap(), Rgb8::new(2, 2, 2));
        assert_eq!(resized.get(0, 1).unwrap(), Rgb8::new(20, 20, 20));
        assert_eq!(resized.get(1, 1).unwrap(), Rgb8::new(22, 22, 22));
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
        );

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
        let src: Image<u8> = Image::generate(4, 4, |x, y| (y * 10 + x) as u8);
        let mut target: Image<u8> = Image::fill(4, 4, 255);

        {
            let mut roi_out = target.roi_mut(Rectangle::new((1, 1), (2, 2))).unwrap();
            resize_into(&src, &mut roi_out, NearestNeighbor);
        }

        // The ROI region should contain the resized result
        assert_eq!(target.get(1, 1).unwrap(), 0); // src (0,0)
        assert_eq!(target.get(2, 1).unwrap(), 3); // src (3,0)
        assert_eq!(target.get(1, 2).unwrap(), 30); // src (0,3)
        assert_eq!(target.get(2, 2).unwrap(), 33); // src (3,3)

        // Outside the ROI should be untouched
        assert_eq!(target.get(0, 0).unwrap(), 255);
        assert_eq!(target.get(3, 3).unwrap(), 255);
    }

    #[test]
    fn test_resize_bilinear_into_roi_output() {
        // Resize a 2x2 f32 source into a 2x2 ROI within a 4x4 target
        let src: Image<MonoF32> = Image::generate(2, 2, |x, y| MonoF32::new((y * 10 + x) as f32));
        let mut target: Image<MonoF32> = Image::fill(4, 4, MonoF32::new(-1.0));

        {
            let mut roi_out = target.roi_mut(Rectangle::new((0, 0), (2, 2))).unwrap();
            resize_into(&src, &mut roi_out, Bilinear);
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
            resize_into(&roi_in, &mut roi_out, NearestNeighbor);
        }

        // Resized from 4x4 to 2x2 nearest neighbor picks corners
        assert_eq!(target.get(0, 0).unwrap(), Rgb8::new(22, 22, 22));
        assert_eq!(target.get(1, 0).unwrap(), Rgb8::new(25, 25, 25));
        assert_eq!(target.get(0, 1).unwrap(), Rgb8::new(52, 52, 52));
        assert_eq!(target.get(1, 1).unwrap(), Rgb8::new(55, 55, 55));

        // Rest should be zero
        assert_eq!(target.get(2, 0).unwrap(), Rgb8::new(0, 0, 0));
    }

    // -----------------------------------------------------------------------
    // 1x1 resize tests — covers the `width <= 1 || height <= 1` branches
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
        );
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
        );
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
        );
        // scale_x and scale_y are 0.0, so src_x=0, src_y=0
        assert_eq!(resized.get(0, 0).unwrap(), 0);
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
        );
        assert_eq!(resized.get(0, 0).unwrap(), MonoF32::new(0.0));
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
        );
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
        );
        for y in 0..3 {
            for x in 0..3 {
                assert_eq!(resized.get(x, y).unwrap(), MonoF32::new(77.0));
            }
        }
    }
}

//! Convolution through the DFT: the result of `transform::convolve`, by
//! another path.

use super::method::DftMethod;
use super::spectrum::{Spectrum, SpectrumSource};
use crate::border::BorderPolicy;
use crate::image::{Image, ImageView, Kernel, RasterImage};
use crate::{SignedCoordinate, Size};

/// The convolution of `image` with `kernel`, computed through the DFT by
/// `method`: what [`transform::convolve`](crate::transform::convolve)
/// computes, with the same border policy and the same output region, up to
/// rounding.
///
/// # What happens
///
/// A product of spectra is a *cyclic* convolution, in which the kernel
/// wraps around the edges of the image. To give the result of a convolution
/// with a border policy instead, one call
///
/// 1. pads the image by the kernel's extent, `KW − 1` columns and `KH − 1`
///    rows in all, split by the anchor, with values from `border` (what
///    the output region needs and no more: under
///    [`Skip`](crate::border::Skip) nothing is padded);
/// 2. pads further with zeros to the size `method` transforms fast, which
///    leaves the kept part unchanged: a power of two per side for
///    [`Radix2`](super::Radix2), nothing more for
///    [`Bluestein`](super::Bluestein) and [`Auto`](super::Auto);
/// 3. places the kernel, zero elsewhere, into an image of that size and
///    transforms it, and transforms the padded image;
/// 4. multiplies the spectra, inverts, and crops to the output region.
///
/// What is padded is the image, by the kernel's spatial extent; how broad
/// the kernel's spectrum is plays no part. The kernel is transformed on
/// every call.
///
/// # What it costs
///
/// The padding can cost more than the convolution saves. Measured on a
/// development machine in a release build, `f32`: one transform of a
/// 1024 × 1024 image takes 16.5 ms by `Radix2`. A 31 × 31 kernel pads it to
/// 1054 × 1054, which `Auto` transforms by `Bluestein` in 124 ms and
/// `Radix2`, padded on with zeros to 2048 × 2048, in 73 ms; a call runs
/// three transforms. Where the image is periodic by nature, or the wrap at
/// the edges does no harm, the cyclic convolution needs no padding at all:
/// multiply the spectra with [`Spectrum::multiply`].
///
/// Against the direct convolution, measured on the same machine on a
/// 512 × 512 `MonoF32` image under `Clamp`, two runs agreeing: the direct
/// path is faster up to about 15 × 15 (three times as fast at 9 × 9),
/// this one by `Radix2` from about 21 × 21 (four and a half times as fast
/// at 31 × 31, twenty-four times at 63 × 63); by `Auto`, which transforms
/// the exact padded size, the two meet at about 21 × 21. Where they meet
/// depends on the machine and the image size. It is information for
/// choosing, and no function switches on it.
///
/// # What differs from the direct convolution
///
/// - **Pixel types:** `MonoF32` and `MonoF64`, computed in their precision.
/// - **A NaN or an infinity** anywhere in the image or the kernel spreads to
///   every output pixel, where the direct convolution confines it to the
///   kernel's neighbourhood.
/// - **Accuracy** is relative to the whole image: the rounding error is a
///   small fraction of the image's total energy, spread over all pixels, so
///   a dark region beside a bright one keeps less of its relative accuracy
///   than under the direct convolution.
///
/// # Example
///
/// ```
/// use fovea::border::Mirror;
/// use fovea::frequency::{self, Radix2};
/// use fovea::image::{Image, ImageView, Neighborhood};
/// use fovea::pixel::MonoF32;
/// use fovea::transform;
///
/// let img = Image::generate(40, 30, |x, y| MonoF32::new(((x * 7 + y * 3) % 11) as f32));
/// let kernel = Neighborhood::<f32, 3, 3>::box_blur_3x3();
///
/// let direct: Image<MonoF32> = transform::convolve(&img, &kernel, &Mirror);
/// let through_dft = frequency::convolve(&img, &kernel, &Mirror, Radix2);
/// assert_eq!(through_dft.size(), direct.size());
/// assert!((through_dft.pixel_at(20, 15).0 - direct.pixel_at(20, 15).0).abs() < 1e-5);
/// ```
#[must_use]
pub fn convolve<I, K, B, P, M>(image: &I, kernel: &K, border: &B, method: M) -> Image<P>
where
    I: RasterImage<Pixel = P>,
    P: SpectrumSource,
    K: Kernel<Weight = f32>,
    B: BorderPolicy<I>,
    M: DftMethod<P>,
{
    // As `transform::convolve` does: correlate with the kernel turned by 180°.
    let flipped = kernel.flipped();
    let weights = flipped.weights();
    let anchor = flipped.anchor();
    let (kw, kh) = (weights.width(), weights.height());
    let region = border.output_region(image.size(), Size::new(kw, kh), anchor);
    let Size {
        width: rw,
        height: rh,
    } = region.size;
    if rw == 0 || rh == 0 {
        return Image::from_vec(rw, rh, Vec::new()).expect("an empty image holds no pixel");
    }

    // Output pixel x of the region reads the input at region.x + x + i − anchor.x
    // for the kernel's taps i; the padded image starts at the first of them.
    let padded = Size::new(rw + kw - 1, rh + kh - 1);
    let size = method.fit(padded);
    let tables = method
        .tables(size)
        .expect("a method accepts the size it fits to");
    let (along_x, along_y) = (&tables.x, &tables.y);
    let x0 = region.offset.x as isize - anchor.x as isize;
    let y0 = region.offset.y as isize - anchor.y as isize;
    let zero = P::from_f64(0.0);
    let input = Image::generate(size.width, size.height, |x, y| {
        if x >= padded.width || y >= padded.height {
            return zero;
        }
        let at = SignedCoordinate::new(x0 + x as isize, y0 + y as isize);
        match at.within(image.size()) {
            Some(c) => image.pixel_at(c.x, c.y),
            None => border.pixel_at(image, at),
        }
    });

    // With tap i of the turned kernel at (anchor − i) mod size, the cyclic
    // convolution at x + anchor is the sum over the taps of w(i)·padded(x + i),
    // which never wraps for the kept x.
    let mut placed = vec![zero; size.area()];
    for j in 0..kh {
        for i in 0..kw {
            let px = (anchor.x as isize - i as isize).rem_euclid(size.width as isize) as usize;
            let py = (anchor.y as isize - j as isize).rem_euclid(size.height as isize) as usize;
            placed[py * size.width + px] = P::from_f64(f64::from(weights.pixel_at(i, j)));
        }
    }
    let placed = Image::from_vec(size.width, size.height, placed)
        .expect("the kernel image has width * height pixels");

    let mut spectrum = Spectrum::forward(&input, along_x, along_y);
    spectrum
        .multiply(&Spectrum::forward(&placed, along_x, along_y))
        .expect("both images have the same size");
    let full = spectrum.invert(along_x, along_y);
    Image::generate(rw, rh, |x, y| full.pixel_at(x + anchor.x, y + anchor.y))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::border::{Clamp, Constant, Mirror, Skip, Wrap};
    use crate::frequency::testing::{Rng, U32, U64};
    use crate::frequency::{Auto, Bluestein, Radix2};
    use crate::image::Neighborhood;
    use crate::pixel::{MonoF32, MonoF64};
    use crate::transform;

    fn random<P: SpectrumSource>(
        rng: &mut Rng,
        w: usize,
        h: usize,
        of: impl Fn(f64) -> P,
    ) -> Image<P> {
        let values: Vec<P> = (0..w * h).map(|_| of(rng.value())).collect();
        Image::from_vec(w, h, values).unwrap()
    }

    /// An asymmetric 3 × 2 kernel with its anchor off centre, so that a
    /// mistake in the turn or the anchor shows.
    fn asymmetric() -> Neighborhood<f32, 3, 2> {
        Neighborhood::with_anchor::<2, 0>([1.0, -2.0, 0.5, 0.25, 3.0, -1.0])
    }

    fn largest_difference<P: SpectrumSource + Copy>(
        a: &Image<P>,
        b: &Image<P>,
        get: impl Fn(P) -> f64,
    ) -> f64 {
        assert_eq!(a.size(), b.size());
        let mut worst = 0.0_f64;
        for y in 0..a.height() {
            for x in 0..a.width() {
                worst = worst.max((get(a.pixel_at(x, y)) - get(b.pixel_at(x, y))).abs());
            }
        }
        worst
    }

    /// The tolerance of a convolution through the DFT against the direct
    /// one: 50 · u · Σ|w| · max|x|, four times the largest error measured
    /// over the cases of these tests on 2026-10-08 (10 u in `f64`, 12.4 u
    /// in `f32`, in the same units). The error is relative to the whole
    /// image, so it scales with the largest value and the kernel's weight.
    fn tolerance<K: Kernel<Weight = f32>, P: Copy>(
        img: &Image<P>,
        kernel: &K,
        u: f64,
        get: impl Fn(P) -> f64,
    ) -> f64 {
        let w = kernel.weights();
        let weight: f64 = (0..w.height())
            .flat_map(|j| (0..w.width()).map(move |i| (i, j)))
            .map(|(i, j)| f64::from(w.pixel_at(i, j)).abs())
            .sum();
        let peak = (0..img.height())
            .flat_map(|y| (0..img.width()).map(move |x| (x, y)))
            .map(|(x, y)| get(img.pixel_at(x, y)).abs())
            .fold(0.0, f64::max);
        50.0 * u * weight * peak
    }

    macro_rules! agree_under {
        ($img:expr, $kernel:expr, $border:expr, $u:expr, $get:expr) => {{
            let direct = transform::convolve($img, $kernel, $border);
            let tol = tolerance($img, $kernel, $u, $get);
            for (name, through) in [
                ("Radix2", convolve($img, $kernel, $border, Radix2)),
                ("Bluestein", convolve($img, $kernel, $border, Bluestein)),
                ("Auto", convolve($img, $kernel, $border, Auto)),
            ] {
                let d = largest_difference(&direct, &through, $get);
                assert!(d <= tol, "{name}: {d:e} > {tol:e}");
            }
        }};
    }

    #[test]
    fn it_agrees_with_the_direct_convolution_under_every_border() {
        let mut rng = Rng::new(210);
        for (w, h) in [(16usize, 8usize), (13, 9), (1, 5), (7, 1), (32, 31)] {
            let img = random(&mut rng, w, h, MonoF64::new);
            let get = |p: MonoF64| p.0;
            for kernel in [
                Neighborhood::<f32, 3, 3>::box_blur_3x3(),
                Neighborhood::sobel_x(),
            ] {
                agree_under!(&img, &kernel, &Clamp, U64, get);
                agree_under!(&img, &kernel, &Mirror, U64, get);
                agree_under!(&img, &kernel, &Wrap, U64, get);
                agree_under!(&img, &kernel, &Constant(MonoF64::new(0.75)), U64, get);
                agree_under!(&img, &kernel, &Skip, U64, get);
            }
            let kernel = asymmetric();
            agree_under!(&img, &kernel, &Mirror, U64, get);
            agree_under!(&img, &kernel, &Skip, U64, get);
            agree_under!(&img, &kernel, &Constant(MonoF64::new(-1.0)), U64, get);
            let wide = Neighborhood::<f32, 5, 5>::gaussian_5x5();
            agree_under!(&img, &wide, &Clamp, U64, get);
        }
    }

    #[test]
    fn single_precision_agrees_too() {
        let mut rng = Rng::new(220);
        let img = random(&mut rng, 40, 30, |v| MonoF32::new(v as f32));
        let get = |p: MonoF32| f64::from(p.0);
        agree_under!(&img, &asymmetric(), &Mirror, U32, get);
        agree_under!(
            &img,
            &Neighborhood::<f32, 5, 5>::gaussian_5x5(),
            &Wrap,
            U32,
            get
        );
    }

    #[test]
    fn a_kernel_larger_than_the_image_under_skip_gives_an_empty_image() {
        let img = Image::fill(2, 2, MonoF32::new(1.0));
        let out = convolve(
            &img,
            &Neighborhood::<f32, 3, 3>::box_blur_3x3(),
            &Skip,
            Auto,
        );
        assert_eq!(
            out.size(),
            transform::convolve::<_, _, _, _, MonoF32>(
                &img,
                &Neighborhood::<f32, 3, 3>::box_blur_3x3(),
                &Skip
            )
            .size()
        );
        assert_eq!(out.size().area(), 0);
    }

    #[test]
    fn one_nan_spreads_to_every_pixel() {
        let mut img = Image::fill(8, 8, MonoF32::new(1.0));
        *crate::image::ImageViewMut::pixel_at_mut(&mut img, 3, 3) = MonoF32::new(f32::NAN);
        let out = convolve(
            &img,
            &Neighborhood::<f32, 3, 3>::box_blur_3x3(),
            &Clamp,
            Radix2,
        );
        assert!((0..8).all(|y| (0..8).all(|x| out.pixel_at(x, y).0.is_nan())));
    }
}

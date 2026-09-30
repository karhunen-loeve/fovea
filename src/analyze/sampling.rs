//! Point sampling: the value of an image at a sub-pixel position.
//!
//! [`sample`] evaluates an image between its pixels with an
//! [`InterpolationKernel`], the same kernels
//! [`resize`](crate::transform::resize) uses. It is the building block of a
//! caliper, a profile along a line, or any measurement that reads the image
//! where no pixel centre is.
//!
//! Positions follow the crate's pixel-centre convention: `(0.0, 0.0)` is the
//! centre of pixel `(0, 0)`, so `(2.5, 0.0)` lies halfway between pixels
//! `(2, 0)` and `(3, 0)`.

use crate::border::BorderPolicy;
use crate::image::ImageView;
use crate::pixel::{LinearPixel, LinearSpace};
use crate::transform::{InterpolationKernel, MAX_SAMPLE_RADIUS};
use crate::{Coordinate, CoordinateF64, Rectangle, SignedCoordinate, Size};

/// The largest `|position|` sampled. Beyond it, `f64` no longer separates
/// neighbouring pixels, and tap indices could overflow `isize`.
const MAX_POSITION: f64 = (1u64 << 52) as f64;

/// The taps of one axis: the first tap's index and up to
/// `2 · MAX_SAMPLE_RADIUS` normalised weights.
struct AxisTaps {
    first: isize,
    count: usize,
    weight: [f64; 2 * MAX_SAMPLE_RADIUS],
}

impl AxisTaps {
    /// The taps for position `t`, with zero-weight taps trimmed from both
    /// ends: they are never read, so a pixel centre does not need the
    /// neighbour a zero weight would multiply.
    fn new<K: InterpolationKernel>(kernel: &K, t: f64) -> Self {
        let radius = K::RADIUS as isize;
        let base = t.floor() as isize;
        let mut weight = [0.0; 2 * MAX_SAMPLE_RADIUS];
        let mut first = None;
        let mut last = 0;
        let mut sum = 0.0;
        for (i, k) in (base - radius + 1..=base + radius).enumerate() {
            let w = kernel.weight(k as f64 - t);
            weight[i] = w;
            sum += w;
            if w != 0.0 {
                first.get_or_insert(i);
                last = i;
            }
        }
        let Some(start) = first else {
            // A kernel that is zero across its footprint contributes nothing;
            // keep the tap under `t` so the sum stays defined.
            return Self {
                first: base,
                count: 1,
                weight: [0.0; 2 * MAX_SAMPLE_RADIUS],
            };
        };
        let norm = if sum != 0.0 { 1.0 / sum } else { 1.0 };
        let mut trimmed = [0.0; 2 * MAX_SAMPLE_RADIUS];
        for (dst, &w) in trimmed.iter_mut().zip(&weight[start..=last]) {
            *dst = w * norm;
        }
        Self {
            first: base - radius + 1 + start as isize,
            count: last - start + 1,
            weight: trimmed,
        }
    }

    /// Whether every tap lies in `0..len`.
    fn inside(&self, len: usize) -> bool {
        self.first >= 0 && self.first + self.count as isize <= len as isize
    }
}

/// The value of `image` at the sub-pixel position `at`, interpolated with
/// `kernel`.
///
/// Returns the pixel type's accumulator (`MonoF32` for a `Mono8` image), so
/// the fraction survives; convert it with
/// [`FromLinear`](crate::pixel::FromLinear) if a pixel is wanted. At a pixel
/// centre an interpolating kernel returns that pixel exactly.
///
/// The border policy has no default, because near the edge it decides the
/// answer:
///
/// - [`Skip`](crate::border::Skip) returns `None` when the kernel's taps leave
///   the image. **This is the policy for measurement**: it never reports a
///   value that is partly made up.
/// - [`Clamp`](crate::border::Clamp), [`Mirror`](crate::border::Mirror) and
///   [`Wrap`](crate::border::Wrap) extend the image and always return a
///   value, also for a position outside it.
/// - [`Constant`](crate::border::Constant) extends it with a fixed pixel.
///   `Constant(0)` invents an edge at the image border, which a caliper or an
///   edge fit will report as real.
///
/// Also `None` for a non-finite position and for one further than `2^52`
/// pixels from the origin, where `f64` no longer separates neighbouring
/// pixels. The weights are normalised to sum to one over all the kernel's
/// taps, so a flat image samples flat; the normalisation never runs over the
/// in-image taps alone, which would shift the kernel's centre of mass and with
/// it a measured position.
///
/// # Example
///
/// ```
/// use fovea::CoordinateF64;
/// use fovea::analyze::sampling::sample;
/// use fovea::border::Skip;
/// use fovea::image::Image;
/// use fovea::pixel::{Mono8, MonoF32};
/// use fovea::transform::{Bilinear, CatmullRom};
///
/// // A ramp: pixel x holds 10 · x.
/// let ramp = Image::generate(8, 8, |x, _| Mono8::new(10 * x as u8));
///
/// let v = sample(&ramp, CoordinateF64::new(2.5, 3.0), Bilinear, &Skip);
/// assert_eq!(v, Some(MonoF32::new(25.0)));
///
/// // Catmull-Rom reproduces a ramp exactly, too, and reads two taps each side.
/// let v = sample(&ramp, CoordinateF64::new(4.25, 3.0), CatmullRom, &Skip).unwrap();
/// assert!((v.value() - 42.5).abs() < 1e-4);
///
/// // Too close to the edge for Catmull-Rom's taps: no value rather than a guess.
/// assert_eq!(sample(&ramp, CoordinateF64::new(0.5, 3.0), CatmullRom, &Skip), None);
/// ```
///
/// A Bayer mosaic is not in a linear space, so interpolating it does not
/// compile:
///
/// ```compile_fail
/// use fovea::CoordinateF64;
/// use fovea::analyze::sampling::sample;
/// use fovea::border::Skip;
/// use fovea::image::Image;
/// use fovea::pixel::bayer::BayerRggb8;
/// use fovea::transform::Bilinear;
///
/// let raw = Image::fill(8, 8, BayerRggb8::new(100));
/// // ERROR: `BayerRggb8: LinearSpace` is not satisfied.
/// let _ = sample(&raw, CoordinateF64::new(2.5, 2.5), Bilinear, &Skip);
/// ```
///
/// A kernel wider than [`MAX_SAMPLE_RADIUS`] taps a side fails to build:
///
/// ```compile_fail
/// use fovea::CoordinateF64;
/// use fovea::analyze::sampling::sample;
/// use fovea::border::Clamp;
/// use fovea::image::Image;
/// use fovea::pixel::MonoF32;
/// use fovea::transform::Lanczos;
///
/// let img = Image::fill(32, 32, MonoF32::new(1.0));
/// // ERROR: the kernel is wider than the point sampler's buffer.
/// let _ = sample(&img, CoordinateF64::new(8.5, 8.5), Lanczos::<9>, &Clamp);
/// ```
pub fn sample<I, K, B, Q>(image: &I, at: CoordinateF64, kernel: K, border: &B) -> Option<Q>
where
    I: ImageView,
    I::Pixel: LinearPixel<Accumulator = Q> + LinearSpace,
    K: InterpolationKernel,
    B: BorderPolicy<I>,
{
    const {
        assert!(
            K::RADIUS <= MAX_SAMPLE_RADIUS,
            "the kernel is wider than the point sampler's buffer of MAX_SAMPLE_RADIUS taps a side"
        )
    };
    let usable = |t: f64| t.is_finite() && t.abs() < MAX_POSITION;
    if !(usable(at.x) && usable(at.y)) {
        return None;
    }
    let size = image.size();
    let xs = AxisTaps::new(&kernel, at.x);
    let ys = AxisTaps::new(&kernel, at.y);

    let direct = xs.inside(size.width) && ys.inside(size.height);
    if !direct {
        // A policy that extends the image reports the whole image as its
        // output region; `Skip` reports only where the full kernel fits, and
        // its taps must not be read outside the image.
        let taps = 2 * K::RADIUS;
        let region = border.output_region(
            size,
            Size::new(taps, taps),
            Coordinate::new(K::RADIUS - 1, K::RADIUS - 1),
        );
        if region != Rectangle::new((0, 0), size) || size.width == 0 || size.height == 0 {
            return None;
        }
    }

    let mut acc: Option<Q> = None;
    for j in 0..ys.count {
        let y = ys.first + j as isize;
        for i in 0..xs.count {
            let x = xs.first + i as isize;
            let pixel = if direct {
                image.pixel_at(x as usize, y as usize)
            } else {
                border.pixel_at(image, SignedCoordinate::new(x, y))
            };
            let w = (xs.weight[i] * ys.weight[j]) as f32;
            acc = Some(match acc {
                None => pixel.scale(w),
                Some(sum) => pixel.scale_add(w, sum),
            });
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::border::{Clamp, Constant, Mirror, Skip};
    use crate::image::{Image, SubView};
    use crate::pixel::{Mono8, Mono16, MonoF32, MonoF64, RgbF32};
    use crate::transform::{Bilinear, CatmullRom, KeysBicubic, Lanczos, Lanczos2, Lanczos3};

    fn at(x: f64, y: f64) -> CoordinateF64 {
        CoordinateF64::new(x, y)
    }

    fn field() -> Image<MonoF32> {
        Image::generate(9, 7, |x, y| {
            MonoF32::new(((x * 37 + y * 11) % 17) as f32 - 3.5)
        })
    }

    fn centres_are_exact<K: InterpolationKernel>(kernel: K) {
        let img = field();
        for y in 0..7 {
            for x in 0..9 {
                let expected = img.pixel_at(x, y);
                let got = sample(&img, at(x as f64, y as f64), kernel, &Clamp).unwrap();
                assert_eq!(got, expected, "({x}, {y})");
            }
        }
    }

    #[test]
    fn a_pixel_centre_returns_the_pixel_for_every_kernel() {
        centres_are_exact(Bilinear);
        centres_are_exact(CatmullRom);
        centres_are_exact(KeysBicubic::new(-0.75).unwrap());
        centres_are_exact(Lanczos2);
        centres_are_exact(Lanczos3);
    }

    #[test]
    fn skip_accepts_the_last_pixel_centre() {
        // The tap past the last pixel has weight zero there and is not read.
        let img = field();
        assert_eq!(
            sample(&img, at(8.0, 6.0), Bilinear, &Skip),
            Some(img.pixel_at(8, 6))
        );
        assert_eq!(
            sample(&img, at(7.0, 5.0), CatmullRom, &Skip),
            Some(img.pixel_at(7, 5))
        );
    }

    #[test]
    fn bilinear_and_catmull_rom_reproduce_a_plane() {
        let plane = Image::generate(12, 12, |x, y| MonoF64::new(2.0 * x as f64 - 3.0 * y as f64));
        for (x, y) in [(3.25, 4.5), (5.9, 2.1), (7.5, 7.5)] {
            let expected = 2.0 * x - 3.0 * y;
            let b = sample(&plane, at(x, y), Bilinear, &Skip).unwrap();
            let c = sample(&plane, at(x, y), CatmullRom, &Skip).unwrap();
            assert!(
                (b.value() - expected).abs() < 1e-5,
                "bilinear {x},{y}: {b:?}"
            );
            assert!(
                (c.value() - expected).abs() < 1e-5,
                "catmull-rom {x},{y}: {c:?}"
            );
        }
    }

    #[test]
    fn a_flat_image_samples_flat_with_lanczos() {
        let flat = Image::fill(16, 16, MonoF32::new(0.7));
        for (x, y) in [(5.3, 6.9), (8.5, 8.5), (7.01, 9.99)] {
            let v = sample(&flat, at(x, y), Lanczos3, &Skip).unwrap();
            assert!((v.value() - 0.7).abs() < 1e-6, "{v:?}");
        }
    }

    #[test]
    fn skip_refuses_a_footprint_that_leaves_the_image() {
        let img = field();
        assert_eq!(sample(&img, at(0.5, 3.0), CatmullRom, &Skip), None);
        assert_eq!(sample(&img, at(3.0, 5.5), CatmullRom, &Skip), None);
        assert_eq!(sample(&img, at(-0.25, 3.0), Bilinear, &Skip), None);
        assert!(sample(&img, at(1.5, 3.0), CatmullRom, &Skip).is_some());
        assert!(sample(&img, at(7.5, 5.0), Bilinear, &Skip).is_some());
    }

    #[test]
    fn extending_policies_answer_outside_the_image() {
        let img = field();
        // Clamp: far to the left is the left column.
        let v = sample(&img, at(-4.0, 2.0), CatmullRom, &Clamp).unwrap();
        assert_eq!(v, img.pixel_at(0, 2));
        // Constant: far outside is the constant.
        let v = sample(&img, at(40.0, 40.0), Bilinear, &Constant(MonoF32::new(9.0))).unwrap();
        assert_eq!(v, MonoF32::new(9.0));
        // Mirror near the edge is defined and finite.
        let v = sample(&img, at(0.5, 0.5), Lanczos3, &Mirror).unwrap();
        assert!(v.value().is_finite());
    }

    #[test]
    fn an_integer_image_keeps_the_fraction() {
        let img = Image::generate(4, 4, |x, _| Mono8::new(10 + x as u8));
        let v = sample(&img, at(1.5, 1.0), Bilinear, &Skip).unwrap();
        assert_eq!(v, MonoF32::new(11.5));
        let wide = Image::generate(4, 4, |x, _| Mono16::new(1000 * x as u16));
        let v = sample(&wide, at(2.25, 2.0), Bilinear, &Skip).unwrap();
        assert_eq!(v, MonoF32::new(2250.0));
    }

    #[test]
    fn colour_pixels_interpolate_per_channel() {
        let img = Image::generate(4, 4, |x, y| RgbF32::new(x as f32, y as f32, 1.0));
        let v = sample(&img, at(1.5, 2.25), Bilinear, &Skip).unwrap();
        assert_eq!(v, RgbF32::new(1.5, 2.25, 1.0));
    }

    #[test]
    fn a_view_samples_in_its_own_coordinates() {
        let img = field();
        let roi = img.roi(Rectangle::new((2, 1), Size::new(5, 5))).unwrap();
        assert_eq!(
            sample(&roi, at(1.0, 1.0), Bilinear, &Skip),
            Some(img.pixel_at(3, 2))
        );
    }

    #[test]
    fn unusable_positions_give_none() {
        let img = field();
        for p in [
            at(f64::NAN, 1.0),
            at(1.0, f64::INFINITY),
            at(1e300, 1.0),
            at(-(MAX_POSITION), 0.0),
        ] {
            assert_eq!(sample(&img, p, Bilinear, &Clamp), None, "{p:?}");
        }
    }

    #[test]
    fn an_empty_image_has_no_samples() {
        let empty: Image<MonoF32> = Image::zero(0, 0);
        assert_eq!(sample(&empty, at(0.0, 0.0), Bilinear, &Clamp), None);
        assert_eq!(sample(&empty, at(0.0, 0.0), Bilinear, &Skip), None);
    }

    #[test]
    fn the_widest_kernel_the_buffer_holds_builds_and_samples() {
        let flat = Image::fill(40, 40, MonoF32::new(2.0));
        let v = sample(&flat, at(20.5, 20.5), Lanczos::<8>, &Skip).unwrap();
        assert!((v.value() - 2.0).abs() < 1e-5);
    }
}

//! Pyramid construction primitives and strategies.
//!
//! [`pyr_down`] and [`pyr_up`] are the two fundamental resolution-halving /
//! -doubling operations, useful on their own and composed by the
//! [`PyramidMethod`] strategies ([`Gaussian`]) that build a
//! [`Pyramid`](crate::image::Pyramid).
//!
//! Both operations are **named standard operations with a pinned contract**:
//! the smoothing filter is the binomial 5-tap `[1, 4, 6, 4, 1] / 16` per
//! axis (effective σ exactly 1.0) and out-of-bounds access reflects at the
//! edge without duplicating the edge pixel — matching OpenCV's `pyrDown` /
//! `pyrUp` defaults for cross-library comparability. Callers who need a
//! different anti-aliasing filter or border treatment build their own
//! reducer from the parameterized
//! [`gaussian_blur`](crate::transform::gaussian_blur) /
//! [`convolve_separable`](crate::transform::convolve_separable); that path
//! stays fully available.

use crate::Size;
use crate::border::Mirror;
use crate::error::Error;
use crate::image::{Image, ImageView, ImageViewMut, Pyramid, RasterImage, SeparableKernel};
use crate::pixel::{FromLinear, LinearPixel, LinearSpace, ZeroablePixel};
use crate::transform::convolve_separable::convolve_separable;

/// The `pyr_up` interpolation kernel: the binomial `[1, 4, 6, 4, 1] / 8`
/// per axis — the `pyr_down` kernel with weights ×2 per axis (×4 combined),
/// compensating for the zero-inserted samples so brightness is preserved.
const PYR_UP_WEIGHTS: [f32; 5] = [0.125, 0.5, 0.75, 0.5, 0.125];

// ─── pyr_down / pyr_up ──────────────────────────────────────────────────────

/// Blurs and decimates the image by a factor of 2.
///
/// Applies the binomial 5×5 Gaussian (`[1, 4, 6, 4, 1] / 16` per axis — the
/// [`gaussian_blur_5x5`](crate::transform::gaussian_blur_5x5) kernel,
/// effective σ exactly 1.0) followed by 2× downsampling that keeps the
/// even-indexed samples (pixels 0, 2, 4, …).
///
/// The output dimensions are `((width + 1) / 2, (height + 1) / 2)`
/// (ceiling division): an `n`-wide row has `ceil(n / 2)` even samples, so
/// the last column/row of an odd-sized image stays represented, and the
/// sizes match OpenCV's `pyrDown`.
///
/// The kernel and the border treatment (reflection without edge
/// duplication, OpenCV's `BORDER_REFLECT_101`) are part of this function's
/// contract — a named standard operation, not a moving definition. For a
/// different anti-aliasing filter, compose your own reducer from
/// [`gaussian_blur`](crate::transform::gaussian_blur).
///
/// Because the smoothing blends neighboring samples, the pixel type must
/// live in a linear space ([`LinearSpace`]) — linearize sRGB first.
///
/// # Example
///
/// ```
/// use fovea::Size;
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::MonoF32;
/// use fovea::transform::pyr_down;
///
/// let src = Image::fill(9, 6, MonoF32::new(0.5));
/// let half: Image<MonoF32> = pyr_down(&src);
///
/// // Ceiling division: 9 → 5, 6 → 3.
/// assert_eq!(half.size(), Size::new(5, 3));
/// // A flat image stays flat — the kernel preserves brightness.
/// assert!((half.pixel_at(2, 1).0 - 0.5).abs() < 1e-6);
/// ```
#[must_use]
pub fn pyr_down<I, P, Acc>(image: &I) -> Image<P>
where
    I: RasterImage<Pixel = P>,
    P: LinearPixel<f32, Accumulator = Acc> + LinearSpace + ZeroablePixel + FromLinear<Acc>,
    Acc: Copy
        + Default
        + ZeroablePixel
        + LinearPixel<f32, Accumulator = Acc>
        + std::ops::Add<Output = Acc>,
{
    let blurred: Image<P> = convolve_separable(image, &SeparableKernel::gaussian_5(), &Mirror);
    let out_width = image.width().div_ceil(2);
    let out_height = image.height().div_ceil(2);
    Image::generate(out_width, out_height, |x, y| blurred.pixel_at(2 * x, 2 * y))
}

/// Upsamples the image by a factor of 2 to an explicit target size.
///
/// Inserts zero rows/columns (input pixel `(x, y)` lands at output
/// `(2x, 2y)`) and interpolates the missing values with the same binomial
/// kernel as [`pyr_down`], weights ×4 so brightness is preserved after
/// zero-insertion. The border treatment matches `pyr_down` (reflection
/// without edge duplication).
///
/// The explicit `target` is deliberate: [`pyr_down`] maps both an odd and
/// an even dimension onto the same output size, so an upsampler that always
/// doubles would reconstruct the wrong size for odd parents. Naming the
/// parent size removes the ambiguity — `target` must be a size whose
/// `pyr_down` result is this image's size.
///
/// Because the interpolation blends neighboring samples, the pixel type
/// must live in a linear space ([`LinearSpace`]) — linearize sRGB first.
///
/// # Errors
///
/// Returns [`Error::InvalidPyrUpTarget`] if `target.width ∉ {2·w − 1, 2·w}`
/// or `target.height ∉ {2·h − 1, 2·h}` — the caller named a size this
/// image cannot be the `pyr_down` of. Because the valid target is a
/// relation between two runtime sizes (often originating from camera or
/// file dimensions), this is a recoverable error, not a panic.
///
/// # Example
///
/// ```
/// use fovea::Size;
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::MonoF32;
/// use fovea::transform::{pyr_down, pyr_up};
///
/// let src = Image::fill(9, 7, MonoF32::new(0.25));
/// let half: Image<MonoF32> = pyr_down(&src);
/// assert_eq!(half.size(), Size::new(5, 4));
///
/// // The explicit target restores the odd parent size exactly.
/// let restored: Image<MonoF32> = pyr_up(&half, src.size())?;
/// assert_eq!(restored.size(), Size::new(9, 7));
/// assert!((restored.pixel_at(4, 3).0 - 0.25).abs() < 1e-6);
/// # Ok::<(), fovea::Error>(())
/// ```
pub fn pyr_up<I, P, Acc>(image: &I, target: Size) -> Result<Image<P>, Error>
where
    I: RasterImage<Pixel = P>,
    P: LinearPixel<f32, Accumulator = Acc> + LinearSpace + ZeroablePixel + FromLinear<Acc>,
    Acc: Copy
        + Default
        + ZeroablePixel
        + LinearPixel<f32, Accumulator = Acc>
        + std::ops::Add<Output = Acc>,
{
    let (w, h) = (image.width(), image.height());
    let width_ok = target.width == 2 * w || target.width + 1 == 2 * w;
    let height_ok = target.height == 2 * h || target.height + 1 == 2 * h;
    if !width_ok || !height_ok {
        return Err(Error::InvalidPyrUpTarget {
            source: image.size(),
            target,
        });
    }

    // Zero-insertion: every input sample keeps its even-even position; the
    // in-between positions start at zero and are filled by the smoothing.
    let mut upsampled = Image::<P>::zero(target.width, target.height);
    for y in 0..h {
        let row = image.row(y);
        for (x, &pixel) in row.iter().enumerate() {
            *upsampled.pixel_at_mut(2 * x, 2 * y) = pixel;
        }
    }

    let kernel = SeparableKernel::symmetric(PYR_UP_WEIGHTS);
    Ok(convolve_separable(&upsampled, &kernel, &Mirror))
}

// ─── PyramidMethod strategy ─────────────────────────────────────────────────

/// Strategy trait for pyramid construction.
///
/// A `PyramidMethod` produces a [`Pyramid`] and is then discarded — the
/// result does not remember how it was built, following the same pattern as
/// [`ResizeMethod`](crate::transform::ResizeMethod) and
/// [`ConvertPixel`](crate::transform::ConvertPixel). Implement this trait
/// for custom decomposition schemes; assemble the result with
/// [`Pyramid::try_from_levels`](crate::image::Pyramid::try_from_levels).
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::MonoF32;
/// use fovea::transform::{Gaussian, PyramidMethod};
///
/// let img = Image::fill(32, 32, MonoF32::new(1.0));
/// let pyramid = Gaussian.build(&img, 4);
///
/// assert_eq!(pyramid.depth(), 4);
/// assert_eq!(pyramid.coarsest().size().width, 4);
/// ```
pub trait PyramidMethod<P: Copy> {
    /// The level type of the pyramid this method produces.
    type Level: crate::image::PyramidLevel;

    /// Builds a pyramid from the given image.
    ///
    /// `max_depth` is an **upper bound, not a promise**. If the image is
    /// too small to support the requested depth, `build` clamps at the
    /// method's minimum usable level size — it never panics, never errors,
    /// and never mutates the caller's parameters. The resolved depth is
    /// whatever [`Pyramid::depth`] reports afterwards. The result always
    /// contains at least one level.
    fn build(&self, image: &Image<P>, max_depth: usize) -> Pyramid<Self::Level>;
}

/// Gaussian pyramid construction: repeated [`pyr_down`].
///
/// Level 0 is a copy of the input image; each further level is the
/// [`pyr_down`] of the previous one, halving the resolution (ceiling
/// division) with the pinned binomial smoothing. The levels are plain
/// [`Image<P>`] values — no wrapper, no stored strategy.
///
/// The build stops early once a level cannot shrink further (1×1), so the
/// resolved depth may be smaller than requested; a `max_depth` of 0 is
/// treated as 1, because a pyramid always contains at least its base level.
///
/// # Example
///
/// ```
/// use fovea::Size;
/// use fovea::image::{GaussianPyramid, Image, ImageView};
/// use fovea::pixel::MonoF32;
/// use fovea::transform::{Gaussian, PyramidMethod};
///
/// let img = Image::fill(20, 12, MonoF32::new(0.5));
/// let pyramid: GaussianPyramid<MonoF32> = Gaussian.build(&img, 3);
///
/// let sizes: Vec<Size> = pyramid.iter().map(|l| l.size()).collect();
/// assert_eq!(sizes, [Size::new(20, 12), Size::new(10, 6), Size::new(5, 3)]);
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Gaussian;

impl<P, Acc> PyramidMethod<P> for Gaussian
where
    P: LinearPixel<f32, Accumulator = Acc> + LinearSpace + ZeroablePixel + FromLinear<Acc>,
    Acc: Copy
        + Default
        + ZeroablePixel
        + LinearPixel<f32, Accumulator = Acc>
        + std::ops::Add<Output = Acc>,
{
    type Level = Image<P>;

    fn build(&self, image: &Image<P>, max_depth: usize) -> Pyramid<Image<P>> {
        let resolved = max_depth.max(1);
        let mut levels = vec![image.clone()];
        while levels.len() < resolved {
            let prev = levels.last().expect("levels start non-empty");
            let Size { width, height } = prev.size();
            // Minimum usable level size: a level that cannot shrink
            // further (or has no pixels at all) ends the chain.
            if width <= 1 && height <= 1 || width == 0 || height == 0 {
                break;
            }
            let next = pyr_down(prev);
            levels.push(next);
        }
        Pyramid::try_from_levels(levels)
            .expect("Gaussian::build produces non-empty, strictly shrinking levels")
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{pixel_distance, sigma};
    use crate::image::{Decimated, PyramidLevel, ScaledImage};
    use crate::pixel::{Mono8, MonoF32};
    use crate::CoordinateF64;

    // ── pyr_down: size contract ─────────────────────────────────────────

    #[test]
    fn pyr_down_even_dimensions_halve() {
        let src = Image::fill(8, 6, MonoF32::new(0.0));
        let out: Image<MonoF32> = pyr_down(&src);
        assert_eq!(out.size(), Size::new(4, 3));
    }

    #[test]
    fn pyr_down_odd_dimensions_use_ceiling_division() {
        let src = Image::fill(7, 5, MonoF32::new(0.0));
        let out: Image<MonoF32> = pyr_down(&src);
        assert_eq!(out.size(), Size::new(4, 3));
    }

    #[test]
    fn pyr_down_one_pixel_image_stays_one_pixel() {
        let src = Image::fill(1, 1, MonoF32::new(0.3));
        let out: Image<MonoF32> = pyr_down(&src);
        assert_eq!(out.size(), Size::new(1, 1));
        assert!((out.pixel_at(0, 0).0 - 0.3).abs() < 1e-6);
    }

    #[test]
    fn pyr_down_single_row_and_column() {
        let row = Image::fill(9, 1, MonoF32::new(0.5));
        let out: Image<MonoF32> = pyr_down(&row);
        assert_eq!(out.size(), Size::new(5, 1));

        let col = Image::fill(1, 8, MonoF32::new(0.5));
        let out: Image<MonoF32> = pyr_down(&col);
        assert_eq!(out.size(), Size::new(1, 4));
    }

    // ── pyr_down: value contract ────────────────────────────────────────

    #[test]
    fn pyr_down_flat_image_preserves_brightness() {
        let src = Image::fill(10, 10, MonoF32::new(0.7));
        let out: Image<MonoF32> = pyr_down(&src);
        for y in 0..out.height() {
            for x in 0..out.width() {
                assert!(
                    (out.pixel_at(x, y).0 - 0.7).abs() < 1e-6,
                    "flat value drifted at ({x}, {y}): {}",
                    out.pixel_at(x, y).0
                );
            }
        }
    }

    #[test]
    fn pyr_down_flat_mono8_preserves_brightness() {
        let src = Image::fill(12, 8, Mono8::new(100));
        let out: Image<Mono8> = pyr_down(&src);
        for y in 0..out.height() {
            for x in 0..out.width() {
                assert_eq!(out.pixel_at(x, y), Mono8::new(100));
            }
        }
    }

    #[test]
    fn pyr_down_keeps_even_samples() {
        // The symmetric binomial kernel preserves a linear ramp in the
        // interior, so out(x) must equal ramp(2x) — the even-sample
        // convention (origin offset (0, 0)), not 2x + 0.5 (area average).
        let src = Image::generate(16, 16, |x, _| MonoF32::new(x as f32));
        let out: Image<MonoF32> = pyr_down(&src);
        for y in 2..out.height() - 2 {
            for x in 2..out.width() - 2 {
                assert!(
                    (out.pixel_at(x, y).0 - 2.0 * x as f32).abs() < 1e-4,
                    "expected even sample 2·{x} at ({x}, {y}), got {}",
                    out.pixel_at(x, y).0
                );
            }
        }
    }

    #[test]
    fn pyr_down_impulse_center_weight() {
        // A unit impulse picks out the kernel's center weight: the 2D
        // binomial center is (6/16)² = 0.140625.
        let src = Image::generate(9, 9, |x, y| {
            if x == 4 && y == 4 {
                MonoF32::new(1.0)
            } else {
                MonoF32::new(0.0)
            }
        });
        let out: Image<MonoF32> = pyr_down(&src);
        assert!((out.pixel_at(2, 2).0 - 0.140625).abs() < 1e-6);
    }

    // ── pyr_up: size contract ───────────────────────────────────────────

    #[test]
    fn pyr_up_accepts_both_valid_widths() {
        let src = Image::fill(4, 4, MonoF32::new(0.5));
        let a: Image<MonoF32> = pyr_up(&src, Size::new(8, 8)).unwrap();
        assert_eq!(a.size(), Size::new(8, 8));
        let b: Image<MonoF32> = pyr_up(&src, Size::new(7, 7)).unwrap();
        assert_eq!(b.size(), Size::new(7, 7));
    }

    #[test]
    fn pyr_up_rejects_invalid_targets() {
        // Too-large width, too-small width, invalid height: each must
        // report the rejected target and the source size.
        let src = Image::fill(4, 4, MonoF32::new(0.5));
        for target in [Size::new(9, 8), Size::new(6, 8), Size::new(8, 10)] {
            let result: Result<Image<MonoF32>, Error> = pyr_up(&src, target);
            assert_eq!(
                result.unwrap_err(),
                Error::InvalidPyrUpTarget {
                    source: Size::new(4, 4),
                    target,
                },
                "target {target:?} must be rejected"
            );
        }
    }

    #[test]
    fn pyr_up_round_trips_odd_sizes() {
        // The reason target is explicit: odd parents reconstruct exactly.
        let src = Image::fill(9, 7, MonoF32::new(0.25));
        let half: Image<MonoF32> = pyr_down(&src);
        assert_eq!(half.size(), Size::new(5, 4));
        let restored: Image<MonoF32> = pyr_up(&half, src.size()).unwrap();
        assert_eq!(restored.size(), src.size());
    }

    // ── pyr_up: value contract ──────────────────────────────────────────

    #[test]
    fn pyr_up_flat_image_preserves_brightness() {
        // The ×4 weight compensation must hold at every position parity
        // (even/odd × even/odd) and at the borders, for both target
        // parities.
        let src = Image::fill(5, 4, MonoF32::new(0.6));
        for target in [Size::new(10, 8), Size::new(9, 7)] {
            let out: Image<MonoF32> = pyr_up(&src, target).unwrap();
            for y in 0..out.height() {
                for x in 0..out.width() {
                    assert!(
                        (out.pixel_at(x, y).0 - 0.6).abs() < 1e-6,
                        "flat value drifted at ({x}, {y}) for target {target:?}: {}",
                        out.pixel_at(x, y).0
                    );
                }
            }
        }
    }

    #[test]
    fn pyr_up_impulse_spreads_interpolation_weights() {
        // Input sample (1, 1) lands at output (2, 2); the separable
        // interpolation weights around it are the per-axis
        // [1, 4, 6, 4, 1] / 8 taps that hit non-zero samples.
        let src = Image::generate(3, 3, |x, y| {
            if x == 1 && y == 1 {
                MonoF32::new(1.0)
            } else {
                MonoF32::new(0.0)
            }
        });
        let out: Image<MonoF32> = pyr_up(&src, Size::new(6, 6)).unwrap();
        // Even-even: center weight 0.75².
        assert!((out.pixel_at(2, 2).0 - 0.5625).abs() < 1e-6);
        // Odd-even: 0.5 · 0.75.
        assert!((out.pixel_at(3, 2).0 - 0.375).abs() < 1e-6);
        // Odd-odd: 0.5 · 0.5.
        assert!((out.pixel_at(3, 3).0 - 0.25).abs() < 1e-6);
    }

    #[test]
    fn pyr_up_mono8_flat() {
        let src = Image::fill(6, 6, Mono8::new(80));
        let out: Image<Mono8> = pyr_up(&src, Size::new(12, 12)).unwrap();
        for y in 0..out.height() {
            for x in 0..out.width() {
                assert_eq!(out.pixel_at(x, y), Mono8::new(80));
            }
        }
    }

    // ── Gaussian PyramidMethod ──────────────────────────────────────────

    #[test]
    fn gaussian_build_level_zero_is_the_input() {
        let src = Image::generate(8, 8, |x, y| MonoF32::new((x + y) as f32));
        let pyramid = Gaussian.build(&src, 3);
        let level0 = pyramid.finest();
        for y in 0..src.height() {
            for x in 0..src.width() {
                assert_eq!(level0.pixel_at(x, y), src.pixel_at(x, y));
            }
        }
    }

    #[test]
    fn gaussian_build_halves_each_level() {
        let src = Image::fill(20, 12, MonoF32::new(0.5));
        let pyramid = Gaussian.build(&src, 3);
        let sizes: Vec<Size> = pyramid.iter().map(|l| l.size()).collect();
        assert_eq!(
            sizes,
            [Size::new(20, 12), Size::new(10, 6), Size::new(5, 3)]
        );
    }

    #[test]
    fn gaussian_build_clamps_depth_on_small_images() {
        // Resolved ≠ requested: 4×4 supports 4, 2, 1 — then 1×1 stops.
        let src = Image::fill(4, 4, MonoF32::new(0.5));
        let pyramid = Gaussian.build(&src, 100);
        assert_eq!(pyramid.depth(), 3);
        assert_eq!(pyramid.coarsest().size(), Size::new(1, 1));
    }

    #[test]
    fn gaussian_build_max_depth_zero_yields_base_level() {
        let src = Image::fill(8, 8, MonoF32::new(0.5));
        let pyramid = Gaussian.build(&src, 0);
        assert_eq!(pyramid.depth(), 1);
        assert_eq!(pyramid.finest().size(), Size::new(8, 8));
    }

    #[test]
    fn gaussian_build_respects_requested_depth() {
        let src = Image::fill(64, 64, MonoF32::new(0.5));
        let pyramid = Gaussian.build(&src, 3);
        assert_eq!(pyramid.depth(), 3);
    }

    #[test]
    fn gaussian_build_flat_stays_flat_at_every_level() {
        let src = Image::fill(16, 16, MonoF32::new(0.4));
        let pyramid = Gaussian.build(&src, 5);
        for (i, level) in pyramid.iter().enumerate() {
            for y in 0..level.height() {
                for x in 0..level.width() {
                    assert!(
                        (level.pixel_at(x, y).0 - 0.4).abs() < 1e-5,
                        "level {i} drifted at ({x}, {y})"
                    );
                }
            }
        }
    }

    #[test]
    fn gaussian_build_mono8() {
        let src = Image::fill(16, 12, Mono8::new(200));
        let pyramid = Gaussian.build(&src, 3);
        assert_eq!(pyramid.depth(), 3);
        assert_eq!(pyramid.coarsest().size(), Size::new(4, 3));
        assert_eq!(pyramid.coarsest().pixel_at(0, 0), Mono8::new(200));
    }

    // ── Level→base coordinate lift property ─────────────────────────────

    #[test]
    fn decimated_lift_recovers_base_position() {
        // A bright Gaussian-ish blob at base position (12, 8): after two
        // pyr_down steps its maximum sits at level coordinates that must
        // lift back to (12, 8) via the even-sample convention.
        let (cx, cy) = (12.0f32, 8.0f32);
        let src = Image::generate(33, 25, |x, y| {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            MonoF32::new((-(dx * dx + dy * dy) / 18.0).exp())
        });

        let mut level: Image<MonoF32> = pyr_down(&src);
        level = pyr_down(&level);
        let scaled = ScaledImage::new(
            level,
            pixel_distance!(4.0),
            CoordinateF64::new(0.0, 0.0),
            sigma!(1.0),
        );

        // Find the argmax on the coarse level.
        let img = scaled.as_image();
        let mut best = (0usize, 0usize, f32::MIN);
        for y in 0..img.height() {
            for x in 0..img.width() {
                let v = img.pixel_at(x, y).0;
                if v > best.2 {
                    best = (x, y, v);
                }
            }
        }

        let lifted = scaled.to_base(CoordinateF64::new(best.0 as f64, best.1 as f64));
        assert_eq!(lifted, CoordinateF64::new(f64::from(cx), f64::from(cy)));
    }
}

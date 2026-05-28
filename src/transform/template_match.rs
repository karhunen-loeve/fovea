//! Template matching via sliding-window comparison.
//!
//! Slide a template image over a source image and compute a per-position
//! similarity score.  Three strategies are provided out of the box:
//!
//! * [`SAD`] — Sum of Absolute Differences
//! * [`SSD`] — Sum of Squared Differences
//! * [`NCC`] — Normalized Cross-Correlation
//!
//! All three produce an `f32` score map whose dimensions are
//! `(image_w − template_w + 1, image_h − template_h + 1)`.
//!
//! # Examples
//!
//! ```
//! use fovea::image::{Image, ImageView};
//! use fovea::transform::{match_template, SAD};
//!
//! let image = Image::fill(10, 10, 0u8);
//! let template = Image::fill(3, 3, 0u8);
//! let result = match_template(&image, &template, SAD).unwrap();
//! assert_eq!(result.width(), 8);
//! assert_eq!(result.height(), 8);
//! ```

use core::marker::PhantomData;
use std::ops::Sub as StdSub;

use super::fold::{FoldItem, FoldOp, fold_neighborhood_into};
use crate::border::Skip;
use crate::error::Error;
use crate::image::sequential::Image;
use crate::image::{ImageView, ImageViewMut, RasterImage, RasterImageMut};
use crate::pixel::{HomogeneousPixel, LinearChannel, MonoF32, ZeroablePixel};

// ─── MatchMethod trait ───────────────────────────────────────────────────────

/// Strategy trait for template matching algorithms.
///
/// Each strategy defines how to score the similarity between an image
/// patch and a template. The trait is parameterized over input, template,
/// and output image types so that each strategy can express its own
/// pixel-level constraints in its `impl` block — following the same
/// pattern as [`ResizeMethod`](crate::transform::ResizeMethod).
pub trait MatchMethod<I: ImageView, T: ImageView, O: ImageViewMut> {
    /// Compute the per-position similarity score map.
    ///
    /// # Errors
    ///
    /// Implementations **must** preflight via
    /// [`match_template_preflight`] (or perform the equivalent checks)
    /// before doing any subtraction on dimensions. Returning
    /// [`Error::TemplateTooLarge`] for oversized templates is
    /// mandatory so that direct callers of the trait method cannot
    /// reintroduce the wrap/panic behaviour that the
    /// [`match_template_into`] wrapper guards against.
    ///
    /// # Panics
    ///
    /// Panics if `output` dimensions do not match the expected score
    /// map size (programmer precondition — Tier 3).
    fn match_into(&self, image: &I, template: &T, output: &mut O) -> Result<(), Error>;
}

/// Shared preflight used by [`match_template_into`] **and** by every
/// built-in [`MatchMethod`] implementation. Verifies that the template
/// has non-zero dimensions and fits inside `image`.
///
/// External strategies should call this at the top of their
/// [`MatchMethod::match_into`] so that direct invocations of the trait
/// method cannot bypass the safety net the wrapper provides.
///
/// # Errors
///
/// Returns [`Error::TemplateTooLarge`] if the template does not fit in
/// `image` along either axis. Tier-2 (data-dependent) per
/// `AGENTS.md`.
///
/// # Panics
///
/// Panics if the template has zero width or height (Tier-3 programmer
/// bug).
#[inline]
pub fn match_template_preflight<I, T>(image: &I, template: &T) -> Result<(), Error>
where
    I: ImageView,
    T: ImageView,
{
    assert!(
        template.width() > 0 && template.height() > 0,
        "template must have non-zero dimensions, got {}x{}",
        template.width(),
        template.height()
    );
    if template.width() > image.width() || template.height() > image.height() {
        return Err(Error::TemplateTooLarge {
            image_size: image.size(),
            template_size: template.size(),
        });
    }
    Ok(())
}

// ─── Convenience functions ───────────────────────────────────────────────────

/// Slide `template` over `image`, writing per-position scores to `output`.
///
/// Output dimensions must be `(image_w - template_w + 1, image_h - template_h + 1)`.
///
/// # Errors
///
/// Returns [`Error::TemplateTooLarge`] if the template does not fit inside
/// `image` (data-dependent failure — Tier 2).
///
/// # Panics
///
/// Panics if `output` dimensions do not match the expected score map size
/// (programmer precondition — Tier 3).
///
/// Panics if the template has zero width or height (Tier 3).
pub fn match_template_into<I, T, O, M>(
    image: &I,
    template: &T,
    output: &mut O,
    method: M,
) -> Result<(), Error>
where
    I: ImageView,
    T: ImageView,
    O: ImageViewMut,
    M: MatchMethod<I, T, O>,
{
    assert!(
        template.width() > 0 && template.height() > 0,
        "template must have non-zero dimensions, got {}x{}",
        template.width(),
        template.height()
    );
    if template.width() > image.width() || template.height() > image.height() {
        return Err(Error::TemplateTooLarge {
            image_size: image.size(),
            template_size: template.size(),
        });
    }
    method.match_into(image, template, output)
}

/// Slide `template` over `image` and return the score map.
///
/// The returned image has dimensions
/// `(image_w - template_w + 1, image_h - template_h + 1)`.
///
/// Returns `Err` if the template is larger than the image in either
/// dimension (Tier 2 — data-dependent failure).
///
/// # Panics
///
/// Panics if the template has zero width or height (Tier 3 — programmer bug).
///
/// # Examples
///
/// ```
/// use fovea::image::{Image, ImageView};
/// use fovea::transform::{match_template, SAD};
///
/// let image = Image::fill(10, 10, 0u8);
/// let template = Image::fill(3, 3, 0u8);
/// let result = match_template(&image, &template, SAD).unwrap();
/// assert_eq!(result.width(), 8);
/// assert_eq!(result.height(), 8);
/// ```
#[must_use]
pub fn match_template<I, T, S, M>(image: &I, template: &T, method: M) -> Result<Image<S>, Error>
where
    I: ImageView,
    T: ImageView,
    S: ZeroablePixel,
    M: MatchMethod<I, T, Image<S>>,
{
    assert!(
        template.width() > 0 && template.height() > 0,
        "template must have non-zero dimensions, got {}x{}",
        template.width(),
        template.height()
    );

    if template.width() > image.width() || template.height() > image.height() {
        return Err(Error::TemplateTooLarge {
            image_size: image.size(),
            template_size: template.size(),
        });
    }

    let out_w = image.width() - template.width() + 1;
    let out_h = image.height() - template.height() + 1;
    let mut output = Image::<S>::zero(out_w, out_h);
    // SAFETY (Tier 3): out_w/out_h are computed from the same image and
    // template sizes, so the size check inside match_template_into cannot
    // fail. We unwrap to surface any internal regression as a panic.
    match_template_into(image, template, &mut output, method)
        .expect("match_template: preflight succeeded but match_template_into returned an error");
    Ok(output)
}

// ─── SadFold ─────────────────────────────────────────────────────────────────

/// `FoldOp<P, P>` for Sum of Absolute Differences.
///
/// Template pixel IS the weight — `FoldItem { pixel, weight }` where
/// `weight` is the template pixel at the current kernel position.
pub(crate) struct SadFold<P> {
    _marker: PhantomData<P>,
}

impl<P> SadFold<P> {
    pub(crate) fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<P> FoldOp<P, P> for SadFold<P>
where
    P: HomogeneousPixel,
    P::Channel: PartialOrd + StdSub<Output = P::Channel> + LinearChannel<f32, Accumulator = f32>,
{
    type Accumulator = f32;
    type Output = MonoF32;

    #[inline(always)]
    fn init(&self) -> f32 {
        0.0
    }

    #[inline(always)]
    fn accumulate(&self, acc: &mut f32, item: FoldItem<P, P>) {
        for c in 0..P::CHANNEL_COUNT {
            let a = item.pixel.channel(c);
            let b = item.weight.channel(c);
            // Unsigned abs-diff: max(a,b) - min(a,b). No signed overflow
            // because we check ordering first.
            let diff = if a >= b { a - b } else { b - a };
            *acc += diff.to_accumulator();
        }
    }

    #[inline(always)]
    fn finalize(&mut self, acc: f32) -> MonoF32 {
        MonoF32(acc)
    }
}

// ─── SsdFold ─────────────────────────────────────────────────────────────────

/// `FoldOp<P, P>` for Sum of Squared Differences.
///
/// Uses `f64` accumulator to avoid precision loss with large templates,
/// then converts to `f32` in `finalize`.
pub(crate) struct SsdFold<P> {
    _marker: PhantomData<P>,
}

impl<P> SsdFold<P> {
    pub(crate) fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<P> FoldOp<P, P> for SsdFold<P>
where
    P: HomogeneousPixel,
    P::Channel: PartialOrd + StdSub<Output = P::Channel> + LinearChannel<f32>,
    <P::Channel as LinearChannel<f32>>::Accumulator: Into<f64>,
{
    type Accumulator = f64;
    type Output = MonoF32;

    #[inline(always)]
    fn init(&self) -> f64 {
        0.0
    }

    #[inline(always)]
    fn accumulate(&self, acc: &mut f64, item: FoldItem<P, P>) {
        for c in 0..P::CHANNEL_COUNT {
            let a = item.pixel.channel(c);
            let b = item.weight.channel(c);
            let diff = if a >= b { a - b } else { b - a };
            let d: f64 = diff.to_accumulator().into();
            *acc = d.mul_add(d, *acc);
        }
    }

    #[inline(always)]
    fn finalize(&mut self, acc: f64) -> MonoF32 {
        MonoF32(acc as f32)
    }
}

// ─── SAD strategy ────────────────────────────────────────────────────────────

/// Sum of Absolute Differences (SAD) template matching.
///
/// For each position `(x, y)` where the template fits inside the image,
/// computes:
///
/// `SAD(x, y) = Σ_c Σ_(dx, dy) |I(x+dx, y+dy, c) − T(dx, dy, c)|`
///
/// where `c` iterates over channels, and `(dx, dy)` iterates over
/// template positions.
///
/// Lower scores indicate better matches. A score of `0.0` means a
/// perfect match.
///
/// # Pixel requirements
///
/// Works on any pixel implementing [`HomogeneousPixel`] whose channel
/// supports ordering, subtraction, and [`LinearPixel<f32>`](crate::pixel::LinearPixel) — including
/// `Mono8`, `Rgb8`, raw sensor types, and primitive pixel types.
/// The [`LinearPixel`](crate::pixel::LinearPixel) bound on the channel ensures meaningful
/// arithmetic; it is already implemented for all built-in channel types.
///
/// # Examples
///
/// ```
/// use fovea::image::{Image, ImageView};
/// use fovea::transform::{match_template, SAD};
///
/// let image = Image::fill(5, 5, 100u8);
/// let template = Image::fill(3, 3, 100u8);
/// let result = match_template(&image, &template, SAD).unwrap();
///
/// // Perfect match everywhere — all scores are 0
/// assert_eq!(result.width(), 3);
/// assert_eq!(result.height(), 3);
/// # use fovea::pixel::MonoF32;
/// for y in 0..result.height() {
///     for x in 0..result.width() {
///         assert_eq!(result.pixel_at(x, y), MonoF32(0.0));
///     }
/// }
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SAD;

impl<I, T, O> MatchMethod<I, T, O> for SAD
where
    I: RasterImage,
    T: ImageView<Pixel = I::Pixel>,
    O: RasterImageMut<Pixel = MonoF32>,
    I::Pixel: HomogeneousPixel,
    <I::Pixel as HomogeneousPixel>::Channel: PartialOrd
        + StdSub<Output = <I::Pixel as HomogeneousPixel>::Channel>
        + LinearChannel<f32, Accumulator = f32>,
{
    fn match_into(&self, image: &I, template: &T, output: &mut O) -> Result<(), Error> {
        match_template_preflight(image, template)?;
        let expected_w = image.width() - template.width() + 1;
        let expected_h = image.height() - template.height() + 1;
        assert_eq!(
            (output.width(), output.height()),
            (expected_w, expected_h),
            "match_template_into(SAD): output size {}x{} does not match expected {}x{}",
            output.width(),
            output.height(),
            expected_w,
            expected_h
        );
        fold_neighborhood_into(image, template, (0, 0), &Skip, output, SadFold::new());
        Ok(())
    }
}

// ─── SSD strategy ────────────────────────────────────────────────────────────

/// Sum of Squared Differences (SSD) template matching.
///
/// For each position `(x, y)` where the template fits inside the image,
/// computes:
///
/// `SSD(x, y) = Σ_c Σ_(dx, dy) (I(x+dx, y+dy, c) − T(dx, dy, c))²`
///
/// Lower scores indicate better matches. A score of `0.0` means a
/// perfect match.
///
/// Uses `f64` internally for the squared accumulator (avoiding precision
/// loss for large templates), then converts to `f32` output.
///
/// # Pixel requirements
///
/// Same as [`SAD`] — [`HomogeneousPixel`] with ordered, subtractable
/// channels and [`LinearPixel<f32>`](crate::pixel::LinearPixel).
///
/// # Examples
///
/// ```
/// use fovea::image::{Image, ImageView};
/// use fovea::transform::{match_template, SSD};
///
/// let image = Image::fill(5, 5, 100u8);
/// let template = Image::fill(3, 3, 100u8);
/// let result = match_template(&image, &template, SSD).unwrap();
///
/// // Perfect match everywhere — all scores are 0
/// # use fovea::pixel::MonoF32;
/// for y in 0..result.height() {
///     for x in 0..result.width() {
///         assert_eq!(result.pixel_at(x, y), MonoF32(0.0));
///     }
/// }
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SSD;

impl<I, T, O> MatchMethod<I, T, O> for SSD
where
    I: RasterImage,
    T: ImageView<Pixel = I::Pixel>,
    O: RasterImageMut<Pixel = MonoF32>,
    I::Pixel: HomogeneousPixel,
    <I::Pixel as HomogeneousPixel>::Channel:
        PartialOrd + StdSub<Output = <I::Pixel as HomogeneousPixel>::Channel> + LinearChannel<f32>,
    <<I::Pixel as HomogeneousPixel>::Channel as LinearChannel<f32>>::Accumulator: Into<f64>,
{
    fn match_into(&self, image: &I, template: &T, output: &mut O) -> Result<(), Error> {
        match_template_preflight(image, template)?;
        let expected_w = image.width() - template.width() + 1;
        let expected_h = image.height() - template.height() + 1;
        assert_eq!(
            (output.width(), output.height()),
            (expected_w, expected_h),
            "match_template_into(SSD): output size {}x{} does not match expected {}x{}",
            output.width(),
            output.height(),
            expected_w,
            expected_h
        );
        fold_neighborhood_into(image, template, (0, 0), &Skip, output, SsdFold::new());
        Ok(())
    }
}

// ─── NccAccum ────────────────────────────────────────────────────────────────

/// Running accumulator for Normalized Cross-Correlation.
///
/// Tracks three sums across all channels per output position:
/// - `sum_i`: Σ I(c) over all template positions and channels
/// - `sum_i_sq`: Σ I(c)²
/// - `cross`: Σ I(c)·T(c)
#[derive(Clone, Copy, Debug)]
pub(crate) struct NccAccum {
    sum_i: f64,
    sum_i_sq: f64,
    cross: f64,
}

// ─── NccFold ─────────────────────────────────────────────────────────────────

/// `FoldOp<P, P>` for Normalized Cross-Correlation.
///
/// Precomputed template statistics are stored in the struct:
/// - `template_sum`: Σ T(c)
/// - `template_sum_sq`: Σ T(c)²
/// - `n`: total element count (pixels × channels)
pub(crate) struct NccFold<P> {
    template_sum: f64,
    template_sum_sq: f64,
    n: f64,
    _marker: PhantomData<P>,
}

impl<P> FoldOp<P, P> for NccFold<P>
where
    P: HomogeneousPixel,
    P::Channel: LinearChannel<f32>,
    <P::Channel as LinearChannel<f32>>::Accumulator: Into<f64>,
{
    type Accumulator = NccAccum;
    type Output = MonoF32;

    #[inline(always)]
    fn init(&self) -> NccAccum {
        NccAccum {
            sum_i: 0.0,
            sum_i_sq: 0.0,
            cross: 0.0,
        }
    }

    #[inline(always)]
    fn accumulate(&self, acc: &mut NccAccum, item: FoldItem<P, P>) {
        for c in 0..P::CHANNEL_COUNT {
            let i_val: f64 = item.pixel.channel(c).to_accumulator().into();
            let t_val: f64 = item.weight.channel(c).to_accumulator().into();
            acc.sum_i += i_val;
            acc.sum_i_sq = i_val.mul_add(i_val, acc.sum_i_sq);
            acc.cross = i_val.mul_add(t_val, acc.cross);
        }
    }

    #[inline(always)]
    fn finalize(&mut self, acc: NccAccum) -> MonoF32 {
        let n = self.n;
        let mean_i = acc.sum_i / n;
        let mean_t = self.template_sum / n;
        let var_i = acc.sum_i_sq / n - mean_i * mean_i;
        let var_t = self.template_sum_sq / n - mean_t * mean_t;
        let cov = acc.cross / n - mean_i * mean_t;
        let denom = (var_i * var_t).sqrt();
        if denom < 1e-12 {
            MonoF32(0.0)
        } else {
            MonoF32((cov / denom) as f32)
        }
    }
}

// ─── NCC strategy ────────────────────────────────────────────────────────────

/// Normalized Cross-Correlation (NCC) template matching.
///
/// For each position `(x, y)` where the template fits inside the image,
/// computes the Pearson correlation coefficient between the image patch
/// and the template:
///
/// `NCC(x, y) = cov(I, T) / (σ_I · σ_T)`
///
/// where the statistics are computed over all channels of all pixels in
/// the patch.
///
/// - A score of `1.0` means perfect positive correlation (identical up to
///   affine scaling).
/// - A score of `-1.0` means perfect negative correlation.
/// - A score of `0.0` means no linear correlation, or that one of the
///   patches has zero variance (constant region).
///
/// # Pixel requirements
///
/// Works on any pixel implementing [`HomogeneousPixel`] whose channel
/// implements [`LinearPixel<f32>`](crate::pixel::LinearPixel). Does **not** require ordering or
/// subtraction on the channel type (unlike [`SAD`]/[`SSD`]).
///
/// # Examples
///
/// ```
/// use fovea::image::{Image, ImageView};
/// use fovea::transform::{match_template, NCC};
///
/// // 3×1 image with a gradient
/// let image = Image::from_vec(3, 1, vec![10u8, 20, 30]).unwrap();
/// // Template matches the left side
/// let template = Image::from_vec(2, 1, vec![10u8, 20]).unwrap();
/// let result = match_template(&image, &template, NCC).unwrap();
/// assert_eq!(result.width(), 2);
/// assert_eq!(result.height(), 1);
/// // Perfect positive correlation at position (0,0)
/// assert!((result.pixel_at(0, 0).0 - 1.0).abs() < 1e-6);
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NCC;

impl<I, T, O> MatchMethod<I, T, O> for NCC
where
    I: RasterImage,
    T: ImageView<Pixel = I::Pixel>,
    O: RasterImageMut<Pixel = MonoF32>,
    I::Pixel: HomogeneousPixel,
    <I::Pixel as HomogeneousPixel>::Channel: LinearChannel<f32>,
    <<I::Pixel as HomogeneousPixel>::Channel as LinearChannel<f32>>::Accumulator: Into<f64>,
{
    fn match_into(&self, image: &I, template: &T, output: &mut O) -> Result<(), Error> {
        match_template_preflight(image, template)?;
        let tw = template.width();
        let th = template.height();
        let expected_w = image.width() - tw + 1;
        let expected_h = image.height() - th + 1;
        assert_eq!(
            (output.width(), output.height()),
            (expected_w, expected_h),
            "match_template_into(NCC): output size {}x{} does not match expected {}x{}",
            output.width(),
            output.height(),
            expected_w,
            expected_h
        );

        // Precompute template statistics.
        let channels = <I::Pixel as HomogeneousPixel>::CHANNEL_COUNT;
        let mut template_sum = 0.0f64;
        let mut template_sum_sq = 0.0f64;
        for y in 0..th {
            for x in 0..tw {
                let p = template.pixel_at(x, y);
                for c in 0..channels {
                    let v: f64 = p.channel(c).to_accumulator().into();
                    template_sum += v;
                    template_sum_sq = v.mul_add(v, template_sum_sq);
                }
            }
        }
        let n = (tw * th * channels) as f64;

        fold_neighborhood_into(
            image,
            template,
            (0, 0),
            &Skip,
            output,
            NccFold {
                template_sum,
                template_sum_sq,
                n,
                _marker: PhantomData,
            },
        );
        Ok(())
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{Image, ImageView};
    use crate::pixel::{Mono8, Rgb8};
    use std::num::Saturating;

    // ── helpers ──────────────────────────────────────────────────────

    fn make_5x5_u8() -> Image<Mono8> {
        Image::generate(5, 5, |x, y| Mono8::new((x + y * 5) as u8))
    }

    // ── SAD tests ───────────────────────────────────────────────────

    #[test]
    fn sad_uniform_image_uniform_template_zero() {
        let image = Image::fill(5, 5, Mono8::new(42));
        let template = Image::fill(3, 3, Mono8::new(42));
        let result = match_template(&image, &template, SAD).unwrap();
        assert_eq!(result.width(), 3);
        assert_eq!(result.height(), 3);
        for y in 0..result.height() {
            for x in 0..result.width() {
                assert_eq!(result.pixel_at(x, y), MonoF32(0.0));
            }
        }
    }

    #[test]
    fn ssd_uniform_image_uniform_template_zero() {
        let image = Image::fill(5, 5, Mono8::new(42));
        let template = Image::fill(3, 3, Mono8::new(42));
        let result = match_template(&image, &template, SSD).unwrap();
        assert_eq!(result.width(), 3);
        assert_eq!(result.height(), 3);
        for y in 0..result.height() {
            for x in 0..result.width() {
                assert_eq!(result.pixel_at(x, y), MonoF32(0.0));
            }
        }
    }

    #[test]
    fn sad_known_3x3_on_5x5() {
        // 5×5 image with values 0..24 (row-major)
        let image = make_5x5_u8();
        // Template: 3×3 patch from top-left corner of image
        // Image patch at (0,0): 0,1,2,5,6,7,10,11,12
        let template = Image::from_vec(
            3,
            3,
            vec![0, 1, 2, 5, 6, 7, 10, 11, 12]
                .into_iter()
                .map(Mono8::new)
                .collect(),
        )
        .unwrap();
        let result = match_template(&image, &template, SAD).unwrap();
        assert_eq!(result.width(), 3);
        assert_eq!(result.height(), 3);
        // At position (0,0) the patch matches exactly → score = 0
        assert_eq!(result.pixel_at(0, 0), MonoF32(0.0));
        // At position (1,0) the image patch is [1,2,3, 6,7,8, 11,12,13]
        // Diffs vs template [0,1,2, 5,6,7, 10,11,12]: each diff is 1
        // SAD = 9 * 1 = 9
        assert_eq!(result.pixel_at(1, 0), MonoF32(9.0));
    }

    #[test]
    fn ssd_known_3x3_on_5x5() {
        let image = make_5x5_u8();
        let template = Image::from_vec(
            3,
            3,
            vec![0, 1, 2, 5, 6, 7, 10, 11, 12]
                .into_iter()
                .map(Mono8::new)
                .collect(),
        )
        .unwrap();
        let result = match_template(&image, &template, SSD).unwrap();
        // At (0,0): exact match → 0
        assert_eq!(result.pixel_at(0, 0), MonoF32(0.0));
        // At (1,0): each diff is 1, SSD = 9 * 1² = 9
        assert_eq!(result.pixel_at(1, 0), MonoF32(9.0));
    }

    #[test]
    fn sad_exact_match_at_source() {
        let image = make_5x5_u8();
        // Extract a 2×2 patch from position (2, 1)
        // Image values: row 1 = [5,6,7,8,9], row 2 = [10,11,12,13,14]
        // Patch at (2,1): [7,8, 12,13]
        let template = Image::from_vec(
            2,
            2,
            vec![7, 8, 12, 13].into_iter().map(Mono8::new).collect(),
        )
        .unwrap();
        let result = match_template(&image, &template, SAD).unwrap();
        assert_eq!(result.width(), 4);
        assert_eq!(result.height(), 4);
        // Score at (2,1) should be 0 (exact match)
        assert_eq!(result.pixel_at(2, 1), MonoF32(0.0));
        // Score elsewhere should be non-zero
        assert!(result.pixel_at(0, 0).0 > 0.0);
    }

    #[test]
    fn output_size_correct() {
        let image = Image::fill(10, 8, Mono8::new(0));
        let template = Image::fill(3, 2, Mono8::new(0));
        let result = match_template(&image, &template, SAD).unwrap();
        assert_eq!(result.width(), 10 - 3 + 1);
        assert_eq!(result.height(), 8 - 2 + 1);
    }

    #[test]
    fn template_larger_than_image_returns_err() {
        let image = Image::fill(3, 3, Mono8::new(0));
        let template = Image::fill(5, 5, Mono8::new(0));
        let result = match_template(&image, &template, SAD);
        assert!(result.is_err());
    }

    // P1-1: calling the trait method directly (bypassing
    // `match_template_into`) must also return `Err(TemplateTooLarge)`
    // and **must not** wrap/panic on dimension subtraction.
    #[test]
    fn direct_match_into_rejects_oversized_template_sad() {
        let image = Image::fill(3, 3, Mono8::new(0));
        let template = Image::fill(5, 5, Mono8::new(0));
        let mut output: Image<MonoF32> = Image::fill(1, 1, MonoF32(0.0));
        let res = MatchMethod::match_into(&SAD, &image, &template, &mut output);
        assert!(matches!(res, Err(Error::TemplateTooLarge { .. })));
    }

    #[test]
    fn direct_match_into_rejects_oversized_template_ssd() {
        let image = Image::fill(3, 3, Mono8::new(0));
        let template = Image::fill(5, 5, Mono8::new(0));
        let mut output: Image<MonoF32> = Image::fill(1, 1, MonoF32(0.0));
        let res = MatchMethod::match_into(&SSD, &image, &template, &mut output);
        assert!(matches!(res, Err(Error::TemplateTooLarge { .. })));
    }

    #[test]
    fn direct_match_into_rejects_oversized_template_ncc() {
        let image = Image::fill(3, 3, Mono8::new(0));
        let template = Image::fill(5, 5, Mono8::new(0));
        let mut output: Image<MonoF32> = Image::fill(1, 1, MonoF32(0.0));
        let res = MatchMethod::match_into(&NCC, &image, &template, &mut output);
        assert!(matches!(res, Err(Error::TemplateTooLarge { .. })));
    }

    #[test]
    fn template_wider_than_image_returns_err() {
        let image = Image::fill(3, 10, Mono8::new(0));
        let template = Image::fill(5, 2, Mono8::new(0));
        let result = match_template(&image, &template, SAD);
        assert!(result.is_err());
    }

    #[test]
    fn template_taller_than_image_returns_err() {
        let image = Image::fill(10, 3, Mono8::new(0));
        let template = Image::fill(2, 5, Mono8::new(0));
        let result = match_template(&image, &template, SAD);
        assert!(result.is_err());
    }

    #[test]
    #[should_panic]
    fn into_wrong_output_size_panics() {
        let image = Image::fill(5, 5, Mono8::new(0));
        let template = Image::fill(3, 3, Mono8::new(0));
        let mut output = Image::<MonoF32>::zero(1, 1); // wrong size
        match_template_into(&image, &template, &mut output, SAD).unwrap();
    }

    #[test]
    fn match_template_into_rejects_oversized_template() {
        // H2: match_template_into used to delegate directly to the strategy,
        // which then subtracted dimensions and either panicked in debug or
        // wrapped in release. Now it returns Err(TemplateTooLarge) just
        // like the allocating match_template variant.
        let image = Image::fill(3, 3, Mono8::new(0));
        let template = Image::fill(5, 5, Mono8::new(0));
        let mut output = Image::<MonoF32>::zero(1, 1);
        let err = match_template_into(&image, &template, &mut output, SAD);
        assert!(matches!(err, Err(Error::TemplateTooLarge { .. })));
    }

    #[test]
    fn match_template_into_matches_match_template() {
        let image = make_5x5_u8();
        let template =
            Image::from_vec(2, 2, vec![0, 1, 5, 6].into_iter().map(Mono8::new).collect()).unwrap();

        let result = match_template(&image, &template, SAD).unwrap();

        let mut result_into = Image::<MonoF32>::zero(result.width(), result.height());
        match_template_into(&image, &template, &mut result_into, SAD).unwrap();

        for y in 0..result.height() {
            for x in 0..result.width() {
                assert_eq!(result.pixel_at(x, y), result_into.pixel_at(x, y));
            }
        }
    }

    // ── Multi-channel tests ─────────────────────────────────────────

    #[test]
    fn sad_mono8_uniform() {
        let image = Image::fill(5, 5, Mono8::new(100));
        let template = Image::fill(3, 3, Mono8::new(100));
        let result = match_template(&image, &template, SAD).unwrap();
        for y in 0..result.height() {
            for x in 0..result.width() {
                assert_eq!(result.pixel_at(x, y), MonoF32(0.0));
            }
        }
    }

    #[test]
    fn sad_rgb8_uniform() {
        let pixel = Rgb8 {
            r: Saturating(50),
            g: Saturating(100),
            b: Saturating(200),
        };
        let image = Image::fill(5, 5, pixel);
        let template = Image::fill(3, 3, pixel);
        let result = match_template(&image, &template, SAD).unwrap();
        for y in 0..result.height() {
            for x in 0..result.width() {
                assert_eq!(result.pixel_at(x, y), MonoF32(0.0));
            }
        }
    }

    #[test]
    fn sad_rgb8_known_values() {
        // 2x1 image with two different pixels
        let p1 = Rgb8 {
            r: Saturating(10),
            g: Saturating(20),
            b: Saturating(30),
        };
        let p2 = Rgb8 {
            r: Saturating(20),
            g: Saturating(30),
            b: Saturating(40),
        };
        let image = Image::from_vec(2, 1, vec![p1, p2]).unwrap();
        // 1x1 template
        let template = Image::from_vec(1, 1, vec![p1]).unwrap();
        let result = match_template(&image, &template, SAD).unwrap();
        assert_eq!(result.width(), 2);
        assert_eq!(result.height(), 1);
        // At (0,0): exact match
        assert_eq!(result.pixel_at(0, 0), MonoF32(0.0));
        // At (1,0): SAD = |20-10| + |30-20| + |40-30| = 30
        assert_eq!(result.pixel_at(1, 0), MonoF32(30.0));
    }

    #[test]
    fn ssd_rgb8_known_values() {
        let p1 = Rgb8 {
            r: Saturating(10),
            g: Saturating(20),
            b: Saturating(30),
        };
        let p2 = Rgb8 {
            r: Saturating(20),
            g: Saturating(30),
            b: Saturating(40),
        };
        let image = Image::from_vec(2, 1, vec![p1, p2]).unwrap();
        let template = Image::from_vec(1, 1, vec![p1]).unwrap();
        let result = match_template(&image, &template, SSD).unwrap();
        // At (0,0): exact match
        assert_eq!(result.pixel_at(0, 0), MonoF32(0.0));
        // At (1,0): SSD = 10² + 10² + 10² = 300
        assert_eq!(result.pixel_at(1, 0), MonoF32(300.0));
    }

    // ── f32 pixel tests ─────────────────────────────────────────────

    #[test]
    fn sad_f32_pixels() {
        let image = Image::from_vec(3, 1, vec![MonoF32(1.0), MonoF32(2.0), MonoF32(3.0)]).unwrap();
        let template = Image::from_vec(2, 1, vec![MonoF32(1.0), MonoF32(2.0)]).unwrap();
        let result = match_template(&image, &template, SAD).unwrap();
        assert_eq!(result.width(), 2);
        assert_eq!(result.height(), 1);
        // At (0,0): |1-1| + |2-2| = 0
        assert_eq!(result.pixel_at(0, 0), MonoF32(0.0));
        // At (1,0): |2-1| + |3-2| = 2
        assert_eq!(result.pixel_at(1, 0), MonoF32(2.0));
    }

    #[test]
    fn ssd_f32_pixels() {
        let image = Image::from_vec(3, 1, vec![MonoF32(1.0), MonoF32(2.0), MonoF32(3.0)]).unwrap();
        let template = Image::from_vec(2, 1, vec![MonoF32(1.0), MonoF32(2.0)]).unwrap();
        let result = match_template(&image, &template, SSD).unwrap();
        // At (0,0): 0² + 0² = 0
        assert_eq!(result.pixel_at(0, 0), MonoF32(0.0));
        // At (1,0): 1² + 1² = 2
        assert_eq!(result.pixel_at(1, 0), MonoF32(2.0));
    }

    // ── Edge cases ──────────────────────────────────────────────────

    #[test]
    fn template_same_size_as_image() {
        let image = Image::fill(3, 3, Mono8::new(0));
        let template = Image::fill(3, 3, Mono8::new(0));
        let result = match_template(&image, &template, SAD).unwrap();
        assert_eq!(result.width(), 1);
        assert_eq!(result.height(), 1);
        assert_eq!(result.pixel_at(0, 0), MonoF32(0.0));
    }

    #[test]
    fn template_1x1() {
        let image =
            Image::from_vec(3, 1, vec![10, 20, 30].into_iter().map(Mono8::new).collect()).unwrap();
        let template = Image::from_vec(1, 1, vec![Mono8::new(15)]).unwrap();
        let result = match_template(&image, &template, SAD).unwrap();
        assert_eq!(result.width(), 3);
        assert_eq!(result.height(), 1);
        assert_eq!(result.pixel_at(0, 0), MonoF32(5.0)); // |10-15|
        assert_eq!(result.pixel_at(1, 0), MonoF32(5.0)); // |20-15|
        assert_eq!(result.pixel_at(2, 0), MonoF32(15.0)); // |30-15|
    }

    #[test]
    #[should_panic]
    fn zero_size_template_panics() {
        let image = Image::fill(5, 5, Mono8::new(0));
        let template = Image::<Mono8>::zero(0, 3);
        let _ = match_template(&image, &template, SAD);
    }

    // ── NCC tests ───────────────────────────────────────────────────

    #[test]
    fn ncc_exact_copy_is_one() {
        // Image patch [0, 10] with template [0, 10] → NCC = 1.0
        let image = Image::from_vec(2, 1, vec![Mono8::new(0), Mono8::new(10)]).unwrap();
        let template = Image::from_vec(2, 1, vec![Mono8::new(0), Mono8::new(10)]).unwrap();
        let result = match_template(&image, &template, NCC).unwrap();
        assert_eq!(result.width(), 1);
        assert_eq!(result.height(), 1);
        assert!((result.pixel_at(0, 0).0 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn ncc_inverted_copy_is_minus_one() {
        // Image patch [0, 10] with template [10, 0] → NCC = -1.0
        let image = Image::from_vec(2, 1, vec![Mono8::new(0), Mono8::new(10)]).unwrap();
        let template = Image::from_vec(2, 1, vec![Mono8::new(10), Mono8::new(0)]).unwrap();
        let result = match_template(&image, &template, NCC).unwrap();
        assert!((result.pixel_at(0, 0).0 - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn ncc_uniform_template_returns_zero() {
        // Uniform template on non-uniform image → zero variance in template → 0.0
        let image =
            Image::from_vec(3, 1, vec![10, 20, 30].into_iter().map(Mono8::new).collect()).unwrap();
        let template = Image::from_vec(2, 1, vec![Mono8::new(5), Mono8::new(5)]).unwrap();
        let result = match_template(&image, &template, NCC).unwrap();
        for y in 0..result.height() {
            for x in 0..result.width() {
                assert_eq!(result.pixel_at(x, y), MonoF32(0.0));
            }
        }
    }

    #[test]
    fn ncc_uniform_image_returns_zero() {
        // Uniform image region → zero variance in image patch → 0.0
        let image = Image::fill(5, 5, Mono8::new(42));
        let template = Image::from_vec(
            2,
            2,
            vec![10, 20, 30, 40].into_iter().map(Mono8::new).collect(),
        )
        .unwrap();
        let result = match_template(&image, &template, NCC).unwrap();
        for y in 0..result.height() {
            for x in 0..result.width() {
                assert_eq!(result.pixel_at(x, y), MonoF32(0.0));
            }
        }
    }

    #[test]
    fn ncc_known_values() {
        // Hand-computed NCC for a small case.
        // Image: 3×1 = [2, 4, 6], Template: 2×1 = [2, 4]
        //
        // At (0,0): patch = [2, 4]
        //   n=2, sum_i=6, sum_i_sq=20, cross=4+16=20
        //   t_sum=6, t_sum_sq=20
        //   mean_i=3, mean_t=3, var_i=1, var_t=1, cov=1
        //   ncc = 1.0
        //
        // At (1,0): patch = [4, 6]
        //   n=2, sum_i=10, sum_i_sq=52, cross=8+24=32
        //   mean_i=5, mean_t=3, var_i=1, var_t=1, cov=1
        //   ncc = 1.0
        let image =
            Image::from_vec(3, 1, vec![2, 4, 6].into_iter().map(Mono8::new).collect()).unwrap();
        let template = Image::from_vec(2, 1, vec![Mono8::new(2), Mono8::new(4)]).unwrap();
        let result = match_template(&image, &template, NCC).unwrap();
        assert_eq!(result.width(), 2);
        assert!((result.pixel_at(0, 0).0 - 1.0).abs() < 1e-6);
        assert!((result.pixel_at(1, 0).0 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn ncc_output_in_range() {
        // Generate a non-trivial image and template, verify all NCC values in [-1, 1]
        let image = Image::generate(10, 10, |x, y| Mono8::new((x * 7 + y * 13) as u8));
        let template = Image::generate(3, 3, |x, y| Mono8::new((x * 11 + y * 3 + 50) as u8));
        let result = match_template(&image, &template, NCC).unwrap();
        for y in 0..result.height() {
            for x in 0..result.width() {
                let score = result.pixel_at(x, y).0;
                assert!(
                    score >= -1.0 - 1e-6 && score <= 1.0 + 1e-6,
                    "NCC score {} at ({}, {}) is out of [-1, 1] range",
                    score,
                    x,
                    y
                );
            }
        }
    }

    #[test]
    fn ncc_multi_channel_rgb8() {
        let p1 = Rgb8 {
            r: Saturating(10),
            g: Saturating(20),
            b: Saturating(30),
        };
        let p2 = Rgb8 {
            r: Saturating(40),
            g: Saturating(50),
            b: Saturating(60),
        };
        let image = Image::from_vec(2, 1, vec![p1, p2]).unwrap();
        let template = Image::from_vec(2, 1, vec![p1, p2]).unwrap();
        let result = match_template(&image, &template, NCC).unwrap();
        assert!((result.pixel_at(0, 0).0 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn ncc_f32_pixels() {
        let image = Image::from_vec(3, 1, vec![MonoF32(1.0), MonoF32(2.0), MonoF32(3.0)]).unwrap();
        let template = Image::from_vec(2, 1, vec![MonoF32(1.0), MonoF32(2.0)]).unwrap();
        let result = match_template(&image, &template, NCC).unwrap();
        assert!((result.pixel_at(0, 0).0 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn ncc_match_template_into_matches_allocating() {
        let image = Image::generate(6, 6, |x, y| Mono8::new((x + y * 3) as u8));
        let template = Image::generate(3, 3, |x, y| Mono8::new((x + y * 2 + 1) as u8));

        let result = match_template(&image, &template, NCC).unwrap();
        let mut result_into = Image::<MonoF32>::zero(result.width(), result.height());
        match_template_into(&image, &template, &mut result_into, NCC).unwrap();

        for y in 0..result.height() {
            for x in 0..result.width() {
                assert!(
                    (result.pixel_at(x, y).0 - result_into.pixel_at(x, y).0).abs() < 1e-6,
                    "Mismatch at ({}, {}): {} vs {}",
                    x,
                    y,
                    result.pixel_at(x, y).0,
                    result_into.pixel_at(x, y).0
                );
            }
        }
    }

    #[test]
    fn ncc_template_larger_than_image_returns_err() {
        let image = Image::fill(3, 3, Mono8::new(0));
        let template = Image::fill(5, 5, Mono8::new(0));
        let result = match_template(&image, &template, NCC);
        assert!(result.is_err());
    }
}

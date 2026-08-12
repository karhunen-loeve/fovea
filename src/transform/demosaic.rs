//! Demosaicing a raw CFA mosaic into RGB, and the white balance that
//! precedes it.
//!
//! A [Bayer sensor](crate::pixel::bayer) measures one colour per photosite.
//! Demosaicing is the interpolation that gives every site the two colours it
//! did not measure — the step that turns an `Image<BayerRggb12>` into a
//! viewable `Image<Rgb12>`.
//!
//! | Task | Reach for |
//! |---|---|
//! | A viewable image, as simply as possible | [`demosaic`] + [`BayerBilinear`] |
//! | A viewable image, as well as a linear filter can | [`demosaic`] + [`MalvarHeCutler`] |
//! | Per-channel sensor gains, before interpolation | [`white_balance`] + [`BayerGains`] |
//!
//! This module is private; everything below is re-exported at
//! [`crate::transform`], and the user-facing contract — depth preservation,
//! the pinned border reflection and the type-level guards — lives on
//! [`demosaic`], which is where a caller looks for it.
//!
//! # Example
//!
//! ```
//! use fovea::image::{Image, ImageView};
//! use fovea::pixel::{Rgb8, bayer::BayerRggb8};
//! use fovea::transform::{MalvarHeCutler, demosaic};
//!
//! // A flat magenta frame, mosaicked: R sites hold 200, G sites 40, B 150.
//! let raw = Image::generate(8, 8, |x, y| {
//!     BayerRggb8::new(match (x % 2, y % 2) {
//!         (0, 0) => 200, // R site
//!         (1, 1) => 150, // B site
//!         _ => 40,       // the two G sites
//!     })
//! });
//!
//! let rgb: Image<Rgb8> = demosaic(&raw, MalvarHeCutler);
//!
//! // Every site recovers the colour the mosaic encoded — including the
//! // border sites, because reflection preserves the CFA phase.
//! assert_eq!(rgb.pixel_at(4, 4), Rgb8::new(200, 40, 150));
//! assert_eq!(rgb.pixel_at(0, 0), Rgb8::new(200, 40, 150));
//! ```

use crate::Size;
use crate::border::{BorderPolicy, Mirror, compute_interior_region};
use crate::error::Error;
use crate::image::{Image, ImageView, ImageViewMut};
use crate::pixel::{
    FromLinear, LinearPixel, MonoF32, RgbF32, ZeroablePixel,
    bayer::{BayerPixel, CfaColor},
};

// ═══════════════════════════════════════════════════════════════════════════
// DemosaicMethod — the strategy trait
// ═══════════════════════════════════════════════════════════════════════════

/// Strategy trait for CFA interpolation: how one site's missing two colours
/// are estimated.
///
/// Implementors are handed a site coordinate and an accessor for the raw
/// samples around it, and return the full RGB triple for that site. The
/// engine ([`demosaic`]) owns the traversal, the interior/border split and
/// the pinned reflection, so a new algorithm is a new implementation of this
/// one method — no central edit, and no way to accidentally change the
/// border contract.
///
/// # Why a site coordinate rather than a fixed kernel
///
/// The crate's [`FoldOp`](crate::transform::FoldOp) engine applies *one*
/// weight grid to the whole image and deliberately tells the operation
/// nothing about position. A demosaic kernel is chosen by the site's colour,
/// which is a function of `(x % 2, y % 2)`, so it cannot be that one grid.
/// Passing the coordinate is the smallest thing that makes the choice
/// possible; `B::PATTERN` is a compile-time constant, so
/// [`color_at`](crate::pixel::bayer::BayerPattern::color_at) folds to a pair
/// of parity tests and the kernel selection folds to a branch on them.
///
/// # Implementing
///
/// ```
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::{Rgb8, RgbF32, bayer::{BayerPixel, BayerRggb8, CfaColor}};
/// use fovea::transform::{DemosaicMethod, demosaic};
///
/// /// The cheapest possible demosaic: every site keeps its own sample and
/// /// borrows its right-hand and lower neighbours for the other two
/// /// colours. Fast, and visibly wrong — but a complete strategy.
/// struct NearestSite;
///
/// impl<B: BayerPixel> DemosaicMethod<B> for NearestSite {
///     const RADIUS: usize = 1;
///
///     fn interpolate<S>(&self, x: usize, y: usize, sample: S) -> RgbF32
///     where
///         S: Fn(isize, isize) -> f32,
///     {
///         let (mut r, mut g, mut b) = (0.0, 0.0, 0.0);
///         for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
///             let value = sample(dx, dy);
///             match B::PATTERN.color_at(x + dx as usize, y + dy as usize) {
///                 CfaColor::Red => r = value,
///                 CfaColor::Green => g = value,
///                 CfaColor::Blue => b = value,
///             }
///         }
///         RgbF32::new(r, g, b)
///     }
/// }
///
/// let raw = Image::fill(4, 4, BayerRggb8::new(60));
/// let rgb: Image<Rgb8> = demosaic(&raw, NearestSite);
/// assert_eq!(rgb.pixel_at(1, 1), Rgb8::new(60, 60, 60));
/// ```
pub trait DemosaicMethod<B: BayerPixel> {
    /// How far beyond the site this method reads, in samples: `1` for a 3×3
    /// window, `2` for 5×5.
    ///
    /// The engine uses it to size the fast interior path, where taps are
    /// read without a bounds check against the border policy. It is a
    /// **contract**: reading further than `RADIUS` panics on an interior
    /// site near the frame edge rather than silently reading the wrong
    /// sample.
    const RADIUS: usize;

    /// Estimates the full RGB triple at the CFA site `(x, y)`.
    ///
    /// `sample(dx, dy)` returns the raw sample at `(x + dx, y + dy)` as an
    /// `f32`, already resolved against the frame border. The site's own
    /// sample is `sample(0, 0)`, and its colour is
    /// `B::PATTERN.color_at(x, y)`.
    ///
    /// Coordinates are **image** coordinates, so parity is meaningful. Two
    /// consequences implementors may rely on:
    ///
    /// * Because the engine reflects without duplicating the edge sample,
    ///   the tap at `(dx, dy)` has the colour
    ///   `B::PATTERN.color_at(x + dx, y + dy)` even where that position lies
    ///   outside the frame.
    /// * `x + 1` and `y + 1` never overflow a `usize` for a real image, so
    ///   the colour of a neighbouring site can be queried directly — which
    ///   is how a green site tells a red row from a blue one.
    fn interpolate<S>(&self, x: usize, y: usize, sample: S) -> RgbF32
    where
        S: Fn(isize, isize) -> f32;
}

// ═══════════════════════════════════════════════════════════════════════════
// The engine
// ═══════════════════════════════════════════════════════════════════════════

/// Interpolates a raw [Bayer mosaic](crate::pixel::bayer) into RGB,
/// allocating the output.
///
/// A single-sensor colour camera measures one colour per photosite; this is
/// the step that gives every site the two colours it did not measure.
/// [`BayerBilinear`] is the reference algorithm, [`MalvarHeCutler`] the
/// quality path, and [`DemosaicMethod`] the trait for supplying your own.
///
/// # Depth is preserved
///
/// The output pixel type is [`BayerPixel::RgbOutput`] — `BayerRggb8`
/// demosaics to [`Rgb8`](crate::pixel::Rgb8), `BayerRggb12` to
/// [`Rgb12`](crate::pixel::Rgb12) — and the output has the same dimensions
/// as the input. The depth is not a type parameter you choose at the call
/// site, so a twelve-bit frame cannot quietly become eight-bit here;
/// changing depth afterwards is a second, named
/// [conversion](crate::transform::convert_image).
///
/// # The border treatment is pinned, and that is load-bearing
///
/// Every strategy reads a neighbourhood, so every strategy needs samples
/// beyond the frame at the outermost one or two rows and columns. This
/// function always reflects **without duplicating the edge sample** — the
/// [`Mirror`](crate::border::Mirror) policy — and takes no border parameter.
///
/// That is not a default; it is the only correct choice. A CFA sample's
/// colour is a function of its coordinate parity, and reflection about the
/// edge sample maps a coordinate to another of the *same* parity, so a
/// reflected tap carries the colour the interpolation expects. Every other
/// border treatment in the crate breaks that:
///
/// | Policy | Effect on a CFA frame |
/// |---|---|
/// | [`Mirror`](crate::border::Mirror) | parity preserved — the tap has the colour the kernel assumes |
/// | [`Clamp`](crate::border::Clamp) | duplicates the edge sample, flipping parity: a red sample is read where green is expected |
/// | [`Wrap`](crate::border::Wrap) | parity preserved only when both dimensions are even — a runtime property |
/// | [`Constant`](crate::border::Constant) | injects a value with no CFA colour at all |
///
/// The failure mode of the wrong choice is a wrong *colour* in the outermost
/// rows and columns, not a blur — visible, but easy to mistake for lens
/// shading. Exposing the choice would have meant exposing three ways to get
/// it wrong, so the contract is pinned instead, the same way
/// [`pyr_down`](crate::transform::pyr_down) pins its kernel and border.
///
/// A consequence worth knowing: those outermost rows and columns are
/// interpolated from reflected data, so they carry more interpolation error
/// than the interior. Callers who want no fabricated support at all should
/// crop first with
/// [`aligned_bayer_roi`](crate::image::BayerSubView::aligned_bayer_roi) —
/// whose even-origin guarantee is exactly what keeps a sub-view's CFA phase
/// identical to its parent's, so demosaicing a sub-view is well-defined.
///
/// # Example
///
/// ```
/// use fovea::Size;
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::{Rgb12, bayer::BayerRggb12};
/// use fovea::transform::{BayerBilinear, demosaic};
///
/// let raw = Image::fill(6, 4, BayerRggb12::new(2048));
/// let rgb: Image<Rgb12> = demosaic(&raw, BayerBilinear);
///
/// assert_eq!(rgb.size(), Size::new(6, 4));
/// // A mosaic with one value everywhere is a grey frame.
/// assert_eq!(rgb.pixel_at(3, 2), Rgb12::new(2048, 2048, 2048));
/// ```
///
/// # What will not compile
///
/// Raw CFA data that has not been *typed* as CFA data cannot be demosaiced.
/// The pattern is not a parameter here, and there is no runtime argument to
/// get wrong, because the only thing that knows the mosaic layout is the
/// pixel type:
///
/// ```compile_fail
/// use fovea::image::Image;
/// use fovea::pixel::{Mono8, Rgb8};
/// use fovea::transform::{BayerBilinear, demosaic};
///
/// // Sensor data that was decoded as monochrome.
/// let raw = Image::fill(8, 8, Mono8::new(100));
/// // ERROR: the trait bound `Mono8: BayerPixel` is not satisfied.
/// let _: Image<Rgb8> = demosaic(&raw, BayerBilinear);
/// ```
///
/// Nor is the output depth the caller's choice — it follows from the sample
/// type, so a twelve-bit frame cannot be silently narrowed by annotating the
/// result:
///
/// ```compile_fail
/// use fovea::image::Image;
/// use fovea::pixel::{Rgb8, bayer::BayerRggb12};
/// use fovea::transform::{BayerBilinear, demosaic};
///
/// let raw = Image::fill(8, 8, BayerRggb12::new(2048));
/// // ERROR: expected `Image<Rgb12>`, found `Image<Rgb8>`.
/// let _: Image<Rgb8> = demosaic(&raw, BayerBilinear);
/// ```
#[must_use]
pub fn demosaic<I, B, M>(image: &I, method: M) -> Image<B::RgbOutput>
where
    I: ImageView<Pixel = B>,
    B: BayerPixel + LinearPixel<f32, Accumulator = MonoF32>,
    B::RgbOutput: ZeroablePixel + FromLinear<RgbF32>,
    M: DemosaicMethod<B>,
{
    let mut output = Image::<B::RgbOutput>::zero(image.width(), image.height());
    demosaic_into(image, &mut output, method);
    output
}

/// Writes the demosaiced RGB image into a caller-supplied output.
///
/// This is the **base method** — [`demosaic`] allocates the output for you,
/// and documents the border contract, the depth rule, and the type-level
/// guards that both share.
///
/// # Panics
///
/// Panics if `output.size() != image.size()`. Unlike a resize, demosaicing
/// has exactly one correct output size, so a mismatch is a programmer error
/// rather than data that did not fit.
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::{Rgb8, bayer::BayerGbrg8};
/// use fovea::transform::{MalvarHeCutler, demosaic_into};
///
/// let raw = Image::fill(8, 8, BayerGbrg8::new(90));
/// let mut rgb = Image::<Rgb8>::zero(8, 8);
///
/// demosaic_into(&raw, &mut rgb, MalvarHeCutler);
/// assert_eq!(rgb.pixel_at(4, 4), Rgb8::new(90, 90, 90));
/// ```
///
/// A mismatched output *size* is the panic above, but a mismatched output
/// *pixel type* never gets that far:
///
/// ```compile_fail
/// use fovea::image::Image;
/// use fovea::pixel::{Rgb8, bayer::BayerRggb12};
/// use fovea::transform::{BayerBilinear, demosaic_into};
///
/// let raw = Image::fill(8, 8, BayerRggb12::new(2048));
/// let mut out = Image::<Rgb8>::zero(8, 8);
/// // ERROR: `Rgb8` is not `<BayerRggb12 as BayerPixel>::RgbOutput`.
/// demosaic_into(&raw, &mut out, BayerBilinear);
/// ```
pub fn demosaic_into<I, B, O, M>(image: &I, output: &mut O, method: M)
where
    I: ImageView<Pixel = B>,
    B: BayerPixel + LinearPixel<f32, Accumulator = MonoF32>,
    O: ImageViewMut<Pixel = B::RgbOutput>,
    B::RgbOutput: FromLinear<RgbF32>,
    M: DemosaicMethod<B>,
{
    assert_eq!(
        image.size(),
        output.size(),
        "demosaic_into: input size {:?} does not match output size {:?}",
        image.size(),
        output.size(),
    );

    let radius = M::RADIUS;
    let window = 2 * radius + 1;
    let interior =
        compute_interior_region(image.size(), Size::new(window, window), (radius, radius));

    // ── HOT PATH — every tap is inside the frame ──────────────────────────
    //
    // The accessor is a distinct closure type from the boundary one below,
    // so `interpolate` monomorphises twice: this copy indexes the image
    // directly, the other goes through the reflection.
    if let Some(interior) = interior {
        for y in interior.top()..interior.bottom() {
            for x in interior.left()..interior.right() {
                let rgb = method.interpolate(x, y, |dx, dy| {
                    let sx = (x as isize + dx) as usize;
                    let sy = (y as isize + dy) as usize;
                    image.pixel_at(sx, sy).to_accumulator().0
                });
                *output.pixel_at_mut(x, y) = FromLinear::from_linear(rgb);
            }
        }
    }

    // ── COLD PATH — the strip where taps leave the frame ──────────────────
    //
    // Reflection without edge duplication, which is the whole reason this
    // module takes no border parameter: it is the one policy that maps a
    // coordinate to another of the same parity, so a reflected tap keeps
    // the CFA colour the strategy assumes it has.
    for y in 0..image.height() {
        for x in 0..image.width() {
            if let Some(ref interior) = interior {
                if x >= interior.left()
                    && x < interior.right()
                    && y >= interior.top()
                    && y < interior.bottom()
                {
                    continue;
                }
            }

            let rgb = method.interpolate(x, y, |dx, dy| {
                Mirror
                    .pixel_at(image, x as isize + dx, y as isize + dy)
                    .to_accumulator()
                    .0
            });
            *output.pixel_at_mut(x, y) = FromLinear::from_linear(rgb);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// BayerBilinear
// ═══════════════════════════════════════════════════════════════════════════

/// Bilinear CFA interpolation: the average of the nearest sites of each
/// missing colour.
///
/// The reference algorithm. A site keeps its own sample untouched and fills
/// the other two colours from a 3×3 window: the four edge-adjacent sites
/// (for the colour sampled there) and the four diagonal ones (for the
/// remaining colour). It is **exact at the sampled sites** — every value the
/// sensor actually measured survives into the output — and exact for any
/// image whose channels vary linearly across the window.
///
/// Reach for it when the demosaic is a means to an end (a preview, a
/// visualization, an input to something that will blur anyway), or as the
/// baseline to measure a better algorithm against. Its weakness is the
/// classic one: across a sharp edge the interpolation averages samples from
/// both sides, so the three channels disagree about where the edge is and
/// the result shows coloured fringes and "zipper" teeth. That is what
/// [`MalvarHeCutler`] exists to reduce, at roughly a 5×5 window's extra
/// cost.
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::{Rgb8, bayer::BayerRggb8};
/// use fovea::transform::{BayerBilinear, demosaic};
///
/// // Red sites hold 100, everything else 0.
/// let raw = Image::generate(6, 6, |x, y| {
///     BayerRggb8::new(if x % 2 == 0 && y % 2 == 0 { 100 } else { 0 })
/// });
/// let rgb: Image<Rgb8> = demosaic(&raw, BayerBilinear);
///
/// // A red site keeps its own sample …
/// assert_eq!(rgb.pixel_at(2, 2).r.0, 100);
/// // … a green site averages the two red sites in its row …
/// assert_eq!(rgb.pixel_at(3, 2).r.0, 100);
/// // … and the green and blue planes are empty, as sampled.
/// assert_eq!(rgb.pixel_at(2, 2).g.0, 0);
/// assert_eq!(rgb.pixel_at(2, 2).b.0, 0);
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BayerBilinear;

impl<B: BayerPixel> DemosaicMethod<B> for BayerBilinear {
    const RADIUS: usize = 1;

    #[inline]
    fn interpolate<S>(&self, x: usize, y: usize, sample: S) -> RgbF32
    where
        S: Fn(isize, isize) -> f32,
    {
        let center = sample(0, 0);

        match B::PATTERN.color_at(x, y) {
            // A red or blue site: the four edge-adjacent sites are green,
            // the four diagonal ones carry the third colour.
            color @ (CfaColor::Red | CfaColor::Blue) => {
                let green = (sample(-1, 0) + sample(1, 0) + sample(0, -1) + sample(0, 1)) * 0.25;
                let other = (sample(-1, -1) + sample(1, -1) + sample(-1, 1) + sample(1, 1)) * 0.25;
                if color == CfaColor::Red {
                    RgbF32::new(center, green, other)
                } else {
                    RgbF32::new(other, green, center)
                }
            }
            // A green site: one missing colour lies along the row, the
            // other along the column. Which is which follows from the
            // colour of the site to the right — only its parity matters,
            // so `x + 1` is safe at the last column.
            CfaColor::Green => {
                let along_row = (sample(-1, 0) + sample(1, 0)) * 0.5;
                let along_column = (sample(0, -1) + sample(0, 1)) * 0.5;
                if B::PATTERN.color_at(x + 1, y) == CfaColor::Red {
                    RgbF32::new(along_row, center, along_column)
                } else {
                    RgbF32::new(along_column, center, along_row)
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MalvarHeCutler
// ═══════════════════════════════════════════════════════════════════════════

/// Malvar–He–Cutler gradient-corrected interpolation: bilinear plus a
/// second-difference correction read from the *other* colours.
///
/// The quality path, and still a single linear filter — four fixed 5×5
/// kernels selected by the site's CFA colour, each with weights summing to
/// `8/8`. The idea is that the channels of a natural image are correlated:
/// where the red samples around a site show a strong second difference, the
/// green channel almost certainly does the same, so the bilinear estimate
/// can be corrected by a term computed from a colour that *was* sampled
/// there. That correction is what damps the coloured fringes and zipper
/// teeth [`BayerBilinear`] leaves on edges.
///
/// Properties worth knowing:
///
/// * **Exact at the sampled sites**, like bilinear — the site's own measured
///   sample is passed through, never mixed.
/// * **Exact for linearly-varying channels**, also like bilinear: every
///   kernel is symmetric under `(dx, dy) → (−dx, −dy)` and sums to one.
/// * **A fainter fringe over a wider band.** On a measured grey step it cuts
///   the worst channel disagreement of any one pixel by 38% (95 → 59 levels
///   of 255), but the fringed band is *twice as wide* — two columns either
///   side of the edge instead of one, because the window is 5×5 rather than
///   3×3. Total colour error falls, so this is the better default; a caller
///   who needs the transition to be narrow rather than faint should measure
///   both.
/// * **Overshoots.** The correction terms carry negative weights, so an
///   estimate can fall outside the sampled range near a hard edge. The
///   output conversion clamps to the pixel type's depth, which is a real
///   clip rather than a wrap.
/// * It reads a 5×5 window, so the two outermost rows and columns rest on
///   reflected samples rather than one.
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::{Rgb8, bayer::BayerRggb8};
/// use fovea::transform::{MalvarHeCutler, demosaic};
///
/// // A grey step edge: dark left of x = 4, bright from there on. Every
/// // channel steps together, so a perfect demosaic returns grey pixels.
/// let raw = Image::generate(12, 12, |x, _| {
///     BayerRggb8::new(if x < 4 { 40 } else { 200 })
/// });
/// let rgb: Image<Rgb8> = demosaic(&raw, MalvarHeCutler);
///
/// // Two sites away from the step the grey is recovered exactly.
/// let far = rgb.pixel_at(8, 6);
/// assert_eq!((far.r.0, far.g.0, far.b.0), (200, 200, 200));
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MalvarHeCutler;

impl MalvarHeCutler {
    /// Green at a red or blue site: bilinear green, corrected by the
    /// second difference of the site's own colour.
    ///
    /// ```text
    ///  0  0 -1  0  0
    ///  0  0  2  0  0
    /// -1  2  4  2 -1   ÷ 8
    ///  0  0  2  0  0
    ///  0  0 -1  0  0
    /// ```
    #[inline(always)]
    fn green_at_red_or_blue<S>(sample: &S) -> f32
    where
        S: Fn(isize, isize) -> f32,
    {
        let adjacent = sample(-1, 0) + sample(1, 0) + sample(0, -1) + sample(0, 1);
        let outer = sample(-2, 0) + sample(2, 0) + sample(0, -2) + sample(0, 2);
        (4.0 * sample(0, 0) + 2.0 * adjacent - outer) * 0.125
    }

    /// The colour whose sites lie along the row of a green site.
    ///
    /// ```text
    ///  0    0  1/2  0    0
    ///  0   -1   0  -1    0
    /// -1    4   5   4   -1   ÷ 8
    ///  0   -1   0  -1    0
    ///  0    0  1/2  0    0
    /// ```
    #[inline(always)]
    fn along_row<S>(sample: &S) -> f32
    where
        S: Fn(isize, isize) -> f32,
    {
        let row_adjacent = sample(-1, 0) + sample(1, 0);
        let row_outer = sample(-2, 0) + sample(2, 0);
        let diagonal = sample(-1, -1) + sample(1, -1) + sample(-1, 1) + sample(1, 1);
        let column_outer = sample(0, -2) + sample(0, 2);
        (5.0 * sample(0, 0) + 4.0 * row_adjacent - row_outer - diagonal + 0.5 * column_outer)
            * 0.125
    }

    /// The colour whose sites lie along the column of a green site — the
    /// transpose of [`along_row`](Self::along_row).
    #[inline(always)]
    fn along_column<S>(sample: &S) -> f32
    where
        S: Fn(isize, isize) -> f32,
    {
        let column_adjacent = sample(0, -1) + sample(0, 1);
        let column_outer = sample(0, -2) + sample(0, 2);
        let diagonal = sample(-1, -1) + sample(1, -1) + sample(-1, 1) + sample(1, 1);
        let row_outer = sample(-2, 0) + sample(2, 0);
        (5.0 * sample(0, 0) + 4.0 * column_adjacent - column_outer - diagonal + 0.5 * row_outer)
            * 0.125
    }

    /// Red at a blue site, or blue at a red site: the diagonal neighbours,
    /// corrected by the second difference of the site's own colour.
    ///
    /// ```text
    ///  0    0  -3/2  0    0
    ///  0    2   0    2    0
    /// -3/2  0   6    0  -3/2  ÷ 8
    ///  0    2   0    2    0
    ///  0    0  -3/2  0    0
    /// ```
    #[inline(always)]
    fn diagonal<S>(sample: &S) -> f32
    where
        S: Fn(isize, isize) -> f32,
    {
        let diagonal = sample(-1, -1) + sample(1, -1) + sample(-1, 1) + sample(1, 1);
        let outer = sample(-2, 0) + sample(2, 0) + sample(0, -2) + sample(0, 2);
        (6.0 * sample(0, 0) + 2.0 * diagonal - 1.5 * outer) * 0.125
    }
}

impl<B: BayerPixel> DemosaicMethod<B> for MalvarHeCutler {
    const RADIUS: usize = 2;

    #[inline]
    fn interpolate<S>(&self, x: usize, y: usize, sample: S) -> RgbF32
    where
        S: Fn(isize, isize) -> f32,
    {
        let center = sample(0, 0);

        match B::PATTERN.color_at(x, y) {
            CfaColor::Red => RgbF32::new(
                center,
                Self::green_at_red_or_blue(&sample),
                Self::diagonal(&sample),
            ),
            CfaColor::Blue => RgbF32::new(
                Self::diagonal(&sample),
                Self::green_at_red_or_blue(&sample),
                center,
            ),
            // A green site: red lies along one axis and blue along the
            // other. The site to the right names which.
            CfaColor::Green => {
                if B::PATTERN.color_at(x + 1, y) == CfaColor::Red {
                    RgbF32::new(
                        Self::along_row(&sample),
                        center,
                        Self::along_column(&sample),
                    )
                } else {
                    RgbF32::new(
                        Self::along_column(&sample),
                        center,
                        Self::along_row(&sample),
                    )
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// White balance
// ═══════════════════════════════════════════════════════════════════════════

/// Per-colour gains for [`white_balance`], one factor per CFA colour.
///
/// A sensor's three colour filters have different sensitivities, and the
/// illuminant is rarely neutral, so a grey card does not produce equal raw
/// samples. The correction is one multiplier per colour — the quantity a
/// GenICam camera exposes as `BalanceRatio`.
///
/// The type carries its own invariant: a gain must be finite and
/// non-negative. Literals go through the `const fn` [`new`](Self::new),
/// which fails at compile time in a `const` context; values estimated from
/// image data go through [`try_new`](Self::try_new).
///
/// Both green sites of the tile share one gain. Sensors whose two green
/// filters differ measurably need a fourth ratio, which this type does not
/// model.
///
/// # Example
///
/// ```
/// use fovea::pixel::bayer::CfaColor;
/// use fovea::transform::BayerGains;
///
/// // Green is the reference; red and blue are lifted to match it.
/// const DAYLIGHT: BayerGains = BayerGains::new(1.9, 1.0, 1.6);
/// assert_eq!(DAYLIGHT.gain(CfaColor::Blue), 1.6);
///
/// // A ratio computed from a grey patch is checked where it is computed.
/// let measured = BayerGains::try_new(2.0, 1.0, f32::NAN);
/// assert!(measured.is_err());
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BayerGains {
    red: f32,
    green: f32,
    blue: f32,
}

impl BayerGains {
    /// Creates gains from literal or otherwise proven-valid ratios.
    ///
    /// # Panics
    ///
    /// Panics unless every gain is finite and non-negative. As a `const fn`
    /// this is a **compile error** when evaluated in a `const` context; for
    /// ratios computed from data use [`try_new`](Self::try_new).
    #[must_use]
    pub const fn new(red: f32, green: f32, blue: f32) -> Self {
        assert!(
            red.is_finite() && green.is_finite() && blue.is_finite(),
            "BayerGains::new: gains must be finite"
        );
        assert!(
            red >= 0.0 && green >= 0.0 && blue >= 0.0,
            "BayerGains::new: gains must be non-negative"
        );
        Self { red, green, blue }
    }

    /// Creates gains from computed ratios, validating them.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidParameter`] unless every gain is finite and
    /// non-negative. A negative gain would invert the channel and a NaN
    /// would erase it — both silent failures at the pixel level, so they
    /// are rejected where the number enters the API.
    pub fn try_new(red: f32, green: f32, blue: f32) -> Result<Self, Error> {
        let all = [red, green, blue];
        if all.iter().all(|g| g.is_finite() && *g >= 0.0) {
            Ok(Self { red, green, blue })
        } else {
            Err(Error::InvalidParameter(format!(
                "BayerGains must be finite and non-negative, got \
                 red = {red}, green = {green}, blue = {blue}"
            )))
        }
    }

    /// The identity: every gain `1.0`.
    #[must_use]
    pub const fn unity() -> Self {
        Self {
            red: 1.0,
            green: 1.0,
            blue: 1.0,
        }
    }

    /// The gain for one CFA colour.
    #[must_use]
    #[inline(always)]
    pub const fn gain(self, color: CfaColor) -> f32 {
        match color {
            CfaColor::Red => self.red,
            CfaColor::Green => self.green,
            CfaColor::Blue => self.blue,
        }
    }

    /// The red gain.
    #[must_use]
    pub const fn red(self) -> f32 {
        self.red
    }

    /// The green gain, shared by both green sites of the tile.
    #[must_use]
    pub const fn green(self) -> f32 {
        self.green
    }

    /// The blue gain.
    #[must_use]
    pub const fn blue(self) -> f32 {
        self.blue
    }
}

/// Scales every raw sample by the gain of the colour its site sampled,
/// returning a new CFA image of the same type.
///
/// This is white balance in its industrial position: **on the mosaic, before
/// demosaicing**. Doing it here rather than on the RGB result matters for a
/// gradient-corrected algorithm like [`MalvarHeCutler`], whose estimates mix
/// channels — correcting the channels first means the mixing sees balanced
/// data. For a strictly per-channel interpolation such as
/// [`BayerBilinear`] the two orders agree up to rounding.
///
/// Because the result is the same Bayer type as the input, it keeps the CFA
/// phase and can be handed straight to [`demosaic`]. That also means gains
/// above `1.0` **clip at the sample depth**: `BayerRggb12` saturates at
/// 4095, and a highlight already near the maximum cannot be lifted. Where
/// headroom matters, balance after demosaicing into a wider type instead.
///
/// This is not a [`ConvertPixel`](crate::transform::ConvertPixel) strategy,
/// and cannot be: a CFA sample's colour depends on its coordinate, and a
/// per-pixel conversion is deliberately blind to position.
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::bayer::BayerRggb12;
/// use fovea::transform::{BayerGains, white_balance};
///
/// let raw = Image::fill(4, 4, BayerRggb12::new(1000));
/// let balanced = white_balance(&raw, BayerGains::new(1.5, 1.0, 1.25));
///
/// assert_eq!(balanced.pixel_at(0, 0).value(), 1500); // R site
/// assert_eq!(balanced.pixel_at(1, 0).value(), 1000); // G site
/// assert_eq!(balanced.pixel_at(1, 1).value(), 1250); // B site
/// ```
#[must_use]
pub fn white_balance<I, B>(image: &I, gains: BayerGains) -> Image<B>
where
    I: ImageView<Pixel = B>,
    B: BayerPixel + LinearPixel<f32, Accumulator = MonoF32> + FromLinear<MonoF32> + ZeroablePixel,
{
    let mut output = Image::<B>::zero(image.width(), image.height());
    white_balance_into(image, &mut output, gains);
    output
}

/// Writes the white-balanced mosaic into a caller-supplied output.
///
/// This is the **base method** — [`white_balance`] allocates for you. The
/// output may be the same size as the input only; see [`white_balance`] for
/// what the operation does and where it belongs in a pipeline.
///
/// # Panics
///
/// Panics if `output.size() != image.size()`.
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::bayer::BayerBggr8;
/// use fovea::transform::{BayerGains, white_balance_into};
///
/// let raw = Image::fill(4, 4, BayerBggr8::new(100));
/// let mut balanced = Image::<BayerBggr8>::zero(4, 4);
///
/// white_balance_into(&raw, &mut balanced, BayerGains::new(2.0, 1.0, 1.0));
/// // A BGGR tile starts on blue, so (0, 0) keeps the unity blue gain …
/// assert_eq!(balanced.pixel_at(0, 0).value(), 100);
/// // … and the red site is the one that doubles.
/// assert_eq!(balanced.pixel_at(1, 1).value(), 200);
/// ```
pub fn white_balance_into<I, B, O>(image: &I, output: &mut O, gains: BayerGains)
where
    I: ImageView<Pixel = B>,
    B: BayerPixel + LinearPixel<f32, Accumulator = MonoF32> + FromLinear<MonoF32>,
    O: ImageViewMut<Pixel = B>,
{
    assert_eq!(
        image.size(),
        output.size(),
        "white_balance_into: input size {:?} does not match output size {:?}",
        image.size(),
        output.size(),
    );

    for y in 0..image.height() {
        for x in 0..image.width() {
            // `B::PATTERN` is a constant, so the colour lookup folds to a
            // pair of parity tests and the gain to a three-way select.
            let gain = gains.gain(B::PATTERN.color_at(x, y));
            let scaled = image.pixel_at(x, y).scale(gain);
            *output.pixel_at_mut(x, y) = FromLinear::from_linear(scaled);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pixel::bayer::{
        BayerBggr8, BayerGbrg8, BayerGrbg8, BayerRggb8, BayerRggb12, BayerRggb16,
    };
    use crate::pixel::{Rgb8, Rgb12, Rgb16};

    // ── Helpers ─────────────────────────────────────────────────────────

    /// Mosaics an RGB image: every site keeps only the channel its CFA
    /// colour sampled. This is the sensor model the round-trip tests invert.
    fn mosaic<B>(rgb: &Image<Rgb8>) -> Image<B>
    where
        B: BayerPixel + From<u8>,
    {
        Image::generate(rgb.width(), rgb.height(), |x, y| {
            let p = rgb.pixel_at(x, y);
            B::from(match B::PATTERN.color_at(x, y) {
                CfaColor::Red => p.r.0,
                CfaColor::Green => p.g.0,
                CfaColor::Blue => p.b.0,
            })
        })
    }

    /// The channel of `rgb` that a site of `color` sampled.
    fn channel(rgb: Rgb8, color: CfaColor) -> u8 {
        match color {
            CfaColor::Red => rgb.r.0,
            CfaColor::Green => rgb.g.0,
            CfaColor::Blue => rgb.b.0,
        }
    }

    /// Mean absolute error per channel between a reference and a result.
    fn mean_abs_error(reference: &Image<Rgb8>, result: &Image<Rgb8>) -> f64 {
        let mut sum = 0.0;
        for y in 0..reference.height() {
            for x in 0..reference.width() {
                let a = reference.pixel_at(x, y);
                let b = result.pixel_at(x, y);
                sum += (a.r.0 as f64 - b.r.0 as f64).abs()
                    + (a.g.0 as f64 - b.g.0 as f64).abs()
                    + (a.b.0 as f64 - b.b.0 as f64).abs();
            }
        }
        sum / (3 * reference.width() * reference.height()) as f64
    }

    /// A synthetic scene with structure in all three channels: a bright
    /// disc on a graded background, plus a hard vertical step.
    fn scene(width: usize, height: usize) -> Image<Rgb8> {
        Image::generate(width, height, |x, y| {
            let (cx, cy) = (width as f64 / 2.0, height as f64 / 2.0);
            let radius = ((x as f64 - cx).powi(2) + (y as f64 - cy).powi(2)).sqrt();
            let disc = if radius < width as f64 / 5.0 {
                90.0
            } else {
                0.0
            };
            let step = if x > 2 * width / 3 { 60.0 } else { 0.0 };
            let ramp = 40.0 + 100.0 * x as f64 / width as f64;
            let l = ramp + disc + step;
            Rgb8::new(
                (l * 1.0).min(255.0) as u8,
                (l * 0.8).min(255.0) as u8,
                (l * 0.6).min(255.0) as u8,
            )
        })
    }

    // ── Exactness at the sampled sites ──────────────────────────────────

    #[test]
    fn bilinear_reproduces_every_sampled_site_exactly() {
        let reference = scene(24, 20);
        let raw: Image<BayerRggb8> = mosaic(&reference);
        let out: Image<Rgb8> = demosaic(&raw, BayerBilinear);

        for y in 0..reference.height() {
            for x in 0..reference.width() {
                let color = BayerRggb8::PATTERN.color_at(x, y);
                assert_eq!(
                    channel(out.pixel_at(x, y), color),
                    channel(reference.pixel_at(x, y), color),
                    "sampled {color:?} channel changed at ({x}, {y})",
                );
            }
        }
    }

    #[test]
    fn malvar_reproduces_every_sampled_site_exactly() {
        let reference = scene(24, 20);
        let raw: Image<BayerRggb8> = mosaic(&reference);
        let out: Image<Rgb8> = demosaic(&raw, MalvarHeCutler);

        for y in 0..reference.height() {
            for x in 0..reference.width() {
                let color = BayerRggb8::PATTERN.color_at(x, y);
                assert_eq!(
                    channel(out.pixel_at(x, y), color),
                    channel(reference.pixel_at(x, y), color),
                    "sampled {color:?} channel changed at ({x}, {y})",
                );
            }
        }
    }

    // ── Flat colour: the border test ────────────────────────────────────
    //
    // A single colour everywhere must survive a mosaic → demosaic round
    // trip *at every pixel, including the frame edge*. This is the
    // executable form of the module's border argument: with an
    // edge-duplicating policy the outermost sites read a neighbour of the
    // wrong CFA colour and the recovered colour is visibly wrong.

    fn flat_round_trip<B>(method: impl DemosaicMethod<B> + Copy, label: &str)
    where
        B: BayerPixel<RgbOutput = Rgb8> + From<u8> + LinearPixel<f32, Accumulator = MonoF32>,
    {
        let color = Rgb8::new(200, 120, 60);
        let reference = Image::fill(9, 7, color);
        let raw: Image<B> = mosaic(&reference);
        let out: Image<Rgb8> = demosaic(&raw, method);

        for y in 0..out.height() {
            for x in 0..out.width() {
                assert_eq!(
                    out.pixel_at(x, y),
                    color,
                    "{label} {:?}: flat colour not recovered at ({x}, {y})",
                    B::PATTERN,
                );
            }
        }
    }

    #[test]
    fn bilinear_recovers_a_flat_colour_in_all_four_patterns() {
        flat_round_trip::<BayerRggb8>(BayerBilinear, "bilinear");
        flat_round_trip::<BayerBggr8>(BayerBilinear, "bilinear");
        flat_round_trip::<BayerGrbg8>(BayerBilinear, "bilinear");
        flat_round_trip::<BayerGbrg8>(BayerBilinear, "bilinear");
    }

    #[test]
    fn malvar_recovers_a_flat_colour_in_all_four_patterns() {
        flat_round_trip::<BayerRggb8>(MalvarHeCutler, "malvar");
        flat_round_trip::<BayerBggr8>(MalvarHeCutler, "malvar");
        flat_round_trip::<BayerGrbg8>(MalvarHeCutler, "malvar");
        flat_round_trip::<BayerGbrg8>(MalvarHeCutler, "malvar");
    }

    // ── Linear ramps ────────────────────────────────────────────────────

    #[test]
    fn both_methods_are_exact_on_a_linear_ramp_in_the_interior() {
        // Every kernel is symmetric about the site and sums to one, so a
        // channel that varies linearly is reproduced exactly wherever no
        // tap has to be reflected.
        let reference = Image::generate(16, 12, |x, _| {
            let v = (4 * x + 20) as u8;
            Rgb8::new(v, v, v)
        });
        let raw: Image<BayerRggb8> = mosaic(&reference);

        let bilinear: Image<Rgb8> = demosaic(&raw, BayerBilinear);
        let malvar: Image<Rgb8> = demosaic(&raw, MalvarHeCutler);

        for y in 2..10 {
            for x in 2..14 {
                assert_eq!(
                    bilinear.pixel_at(x, y),
                    reference.pixel_at(x, y),
                    "bilinear ({x}, {y})"
                );
                assert_eq!(
                    malvar.pixel_at(x, y),
                    reference.pixel_at(x, y),
                    "malvar ({x}, {y})"
                );
            }
        }
    }

    // ── Quality ─────────────────────────────────────────────────────────

    #[test]
    fn malvar_beats_bilinear_on_an_edge_scene() {
        // The claim that justifies shipping the second algorithm. The
        // scene has hard luminance edges with smooth chroma, which is
        // where bilinear's cross-channel disagreement shows.
        let reference = scene(48, 40);
        let raw: Image<BayerRggb8> = mosaic(&reference);

        // Measured 2026-08-11 on this fixture: bilinear 1.4304, Malvar
        // 1.2936 mean absolute error per channel in 8-bit levels. The
        // margin is modest because most of the scene is a smooth ramp,
        // where both methods are near-exact; the artifact test below is
        // where the difference is stark.
        let bilinear_error = mean_abs_error(&reference, &demosaic(&raw, BayerBilinear));
        let malvar_error = mean_abs_error(&reference, &demosaic(&raw, MalvarHeCutler));

        assert!(
            malvar_error < bilinear_error,
            "expected Malvar-He-Cutler to beat bilinear: \
             malvar = {malvar_error:.4}, bilinear = {bilinear_error:.4}",
        );
    }

    #[test]
    fn malvar_lowers_the_fringe_peak_and_widens_the_band() {
        // Coloured fringing is the artifact that justifies the second
        // algorithm, and a grey scene isolates it: wherever the three
        // channels of the input agree, they must agree in the output too.
        let reference = Image::generate(16, 12, |x, _| {
            let v = if x < 8 { 30u8 } else { 220 };
            Rgb8::new(v, v, v)
        });
        let raw: Image<BayerRggb8> = mosaic(&reference);

        // The widest disagreement between the three channels of one pixel,
        // and how many pixels disagree at all.
        let fringe = |out: &Image<Rgb8>| -> (i32, usize) {
            let mut worst = 0;
            let mut count = 0;
            for y in 0..out.height() {
                for x in 0..out.width() {
                    let p = out.pixel_at(x, y);
                    let channels = [p.r.0 as i32, p.g.0 as i32, p.b.0 as i32];
                    let spread = channels.iter().max().unwrap() - channels.iter().min().unwrap();
                    worst = worst.max(spread);
                    count += usize::from(spread > 0);
                }
            }
            (worst, count)
        };

        let bilinear: Image<Rgb8> = demosaic(&raw, BayerBilinear);
        let malvar: Image<Rgb8> = demosaic(&raw, MalvarHeCutler);

        // Measured 2026-08-11 on this fixture (a 190-level grey step at
        // x = 8, 16×12 px). The trade is not one-directional, and the
        // honest numbers are worth pinning:
        //
        //            worst spread   fringed pixels
        //   bilinear      95            24  (2 columns)
        //   Malvar        59            48  (4 columns)
        //
        // Malvar cuts the *peak* fringe by 38%, but its 5×5 window
        // straddles the step for two columns either side instead of one,
        // so the band is twice as wide. Overall colour error still falls
        // (see `malvar_beats_bilinear_on_an_edge_scene`) — a wider, much
        // fainter fringe is the better artifact — but "Malvar removes the
        // fringe" would be false.
        let (bilinear_worst, bilinear_count) = fringe(&bilinear);
        let (malvar_worst, malvar_count) = fringe(&malvar);
        assert!(
            malvar_worst * 3 < bilinear_worst * 2,
            "expected Malvar-He-Cutler to cut the fringe peak by a third: \
             malvar = {malvar_worst}, bilinear = {bilinear_worst}",
        );
        assert_eq!(
            (bilinear_count, malvar_count),
            (2 * 12, 4 * 12),
            "the fringed band should be exactly as wide as each method's window",
        );

        // Both are exactly grey outside their own window's reach.
        for y in 0..12 {
            for x in (0..16).filter(|x| !(6..=9).contains(x)) {
                let p = malvar.pixel_at(x, y);
                assert_eq!(
                    (p.r.0, p.g.0, p.b.0),
                    (p.r.0, p.r.0, p.r.0),
                    "colour fringe away from the step at ({x}, {y}): {p:?}",
                );
            }
        }
    }

    // ── Shape, depth, and containers ────────────────────────────────────

    #[test]
    fn output_size_matches_the_input_including_odd_dimensions() {
        let raw = Image::fill(7, 5, BayerRggb8::new(80));
        let out: Image<Rgb8> = demosaic(&raw, MalvarHeCutler);
        assert_eq!(out.size(), Size::new(7, 5));
    }

    #[test]
    fn images_smaller_than_the_window_are_all_border() {
        // Nothing is interior, so every site goes through the reflection.
        // The degenerate 1×1 case reflects onto itself.
        for (w, h) in [(1, 1), (2, 2), (3, 3), (4, 1), (1, 4)] {
            let raw = Image::fill(w, h, BayerRggb8::new(120));
            let out: Image<Rgb8> = demosaic(&raw, MalvarHeCutler);
            assert_eq!(out.size(), Size::new(w, h));
            assert_eq!(out.pixel_at(0, 0), Rgb8::new(120, 120, 120));
        }
    }

    #[test]
    fn depth_is_preserved_across_the_family() {
        let raw12 = Image::fill(8, 8, BayerRggb12::new(3000));
        let out12: Image<Rgb12> = demosaic(&raw12, MalvarHeCutler);
        assert_eq!(out12.pixel_at(4, 4), Rgb12::new(3000, 3000, 3000));

        let raw16 = Image::fill(8, 8, BayerRggb16::new(60000));
        let out16: Image<Rgb16> = demosaic(&raw16, BayerBilinear);
        assert_eq!(out16.pixel_at(4, 4), Rgb16::new(60000, 60000, 60000));
    }

    #[test]
    fn malvar_overshoot_clamps_at_the_sample_depth() {
        // A hard step from 0 to the depth maximum makes the correction
        // terms overshoot past 4095; the conversion clips rather than
        // wrapping to a dark pixel.
        let raw = Image::generate(16, 12, |x, _| {
            BayerRggb12::new(if x < 8 { 0 } else { 4095 })
        });
        let out: Image<Rgb12> = demosaic(&raw, MalvarHeCutler);

        for y in 0..12 {
            for x in 0..16 {
                let p = out.pixel_at(x, y);
                for value in [p.r.value(), p.g.value(), p.b.value()] {
                    assert!(value <= 4095, "value {value} exceeds 12 bits at ({x}, {y})");
                }
            }
        }
    }

    #[test]
    fn demosaic_accepts_a_phase_aligned_sub_view() {
        use crate::Rectangle;
        use crate::image::BayerSubView;

        let reference = scene(16, 16);
        let raw: Image<BayerRggb8> = mosaic(&reference);
        let whole: Image<Rgb8> = demosaic(&raw, BayerBilinear);

        // An even origin preserves the CFA phase, so the sub-view's
        // interior must demosaic to the same values as the parent's.
        let roi = raw
            .aligned_bayer_roi(Rectangle::new((4, 4), (8, 8)))
            .expect("even origin inside bounds");
        let cropped: Image<Rgb8> = demosaic(&roi, BayerBilinear);

        for y in 1..7 {
            for x in 1..7 {
                assert_eq!(
                    cropped.pixel_at(x, y),
                    whole.pixel_at(x + 4, y + 4),
                    "sub-view disagrees with the parent at ({x}, {y})",
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "does not match output size")]
    fn demosaic_into_rejects_a_mismatched_output() {
        let raw = Image::fill(8, 8, BayerRggb8::new(10));
        let mut out = Image::<Rgb8>::zero(4, 4);
        demosaic_into(&raw, &mut out, BayerBilinear);
    }

    #[test]
    fn demosaic_and_demosaic_into_agree() {
        let raw: Image<BayerRggb8> = mosaic(&scene(12, 10));
        let allocated: Image<Rgb8> = demosaic(&raw, MalvarHeCutler);

        let mut written = Image::<Rgb8>::zero(12, 10);
        demosaic_into(&raw, &mut written, MalvarHeCutler);

        for y in 0..10 {
            for x in 0..12 {
                assert_eq!(allocated.pixel_at(x, y), written.pixel_at(x, y));
            }
        }
    }

    // ── White balance ───────────────────────────────────────────────────

    #[test]
    fn gains_apply_per_cfa_colour() {
        let raw = Image::fill(6, 6, BayerRggb12::new(1000));
        let out = white_balance(&raw, BayerGains::new(1.5, 1.0, 0.5));

        for y in 0..6 {
            for x in 0..6 {
                let expected = match BayerRggb12::PATTERN.color_at(x, y) {
                    CfaColor::Red => 1500,
                    CfaColor::Green => 1000,
                    CfaColor::Blue => 500,
                };
                assert_eq!(out.pixel_at(x, y).value(), expected, "at ({x}, {y})");
            }
        }
    }

    #[test]
    fn every_pattern_balances_its_own_tile() {
        // The gain follows the CFA colour, not the coordinate, so the same
        // gains produce different images for different patterns.
        let rggb = white_balance(
            &Image::fill(4, 4, BayerRggb8::new(100)),
            BayerGains::new(2.0, 1.0, 1.0),
        );
        let bggr = white_balance(
            &Image::fill(4, 4, BayerBggr8::new(100)),
            BayerGains::new(2.0, 1.0, 1.0),
        );

        assert_eq!(rggb.pixel_at(0, 0).value(), 200); // R sits at (0, 0)
        assert_eq!(bggr.pixel_at(0, 0).value(), 100); // B sits at (0, 0)
        assert_eq!(bggr.pixel_at(1, 1).value(), 200); // R sits at (1, 1)
    }

    #[test]
    fn unity_gains_are_the_identity() {
        let raw: Image<BayerRggb12> = Image::generate(8, 8, |x, y| {
            BayerRggb12::new(((x * 37 + y * 11) % 4096) as u16)
        });
        let out = white_balance(&raw, BayerGains::unity());

        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(out.pixel_at(x, y), raw.pixel_at(x, y));
            }
        }
    }

    #[test]
    fn gains_clip_at_the_sample_depth() {
        let raw = Image::fill(4, 4, BayerRggb12::new(3000));
        let out = white_balance(&raw, BayerGains::new(4.0, 1.0, 1.0));
        // 3000 × 4 = 12000, well past the 12-bit maximum.
        assert_eq!(out.pixel_at(0, 0).value(), 4095);
    }

    #[test]
    fn a_zero_gain_empties_its_colour() {
        let raw = Image::fill(4, 4, BayerRggb8::new(200));
        let out = white_balance(&raw, BayerGains::new(0.0, 1.0, 1.0));
        assert_eq!(out.pixel_at(0, 0).value(), 0);
        assert_eq!(out.pixel_at(1, 0).value(), 200);
    }

    #[test]
    fn try_new_rejects_unusable_gains() {
        assert!(BayerGains::try_new(1.0, 1.0, 1.0).is_ok());
        assert!(BayerGains::try_new(0.0, 0.0, 0.0).is_ok());
        assert!(BayerGains::try_new(-1.0, 1.0, 1.0).is_err());
        assert!(BayerGains::try_new(1.0, f32::NAN, 1.0).is_err());
        assert!(BayerGains::try_new(1.0, 1.0, f32::INFINITY).is_err());
    }

    #[test]
    #[should_panic(expected = "must be non-negative")]
    fn new_panics_on_a_negative_gain() {
        let _ = BayerGains::new(1.0, -0.5, 1.0);
    }

    #[test]
    fn accessors_report_what_was_given() {
        let gains = BayerGains::new(1.9, 1.0, 1.6);
        assert_eq!(gains.red(), 1.9);
        assert_eq!(gains.green(), 1.0);
        assert_eq!(gains.blue(), 1.6);
        assert_eq!(gains.gain(CfaColor::Red), 1.9);
        assert_eq!(gains.gain(CfaColor::Green), 1.0);
        assert_eq!(gains.gain(CfaColor::Blue), 1.6);
    }

    #[test]
    fn white_balance_into_agrees_with_the_allocating_form() {
        let raw = Image::fill(6, 4, BayerGrbg8::new(90));
        let gains = BayerGains::new(1.4, 1.0, 1.2);

        let allocated = white_balance(&raw, gains);
        let mut written = Image::<BayerGrbg8>::zero(6, 4);
        white_balance_into(&raw, &mut written, gains);

        for y in 0..4 {
            for x in 0..6 {
                assert_eq!(allocated.pixel_at(x, y), written.pixel_at(x, y));
            }
        }
    }

    #[test]
    #[should_panic(expected = "does not match output size")]
    fn white_balance_into_rejects_a_mismatched_output() {
        let raw = Image::fill(8, 8, BayerRggb8::new(10));
        let mut out = Image::<BayerRggb8>::zero(8, 4);
        white_balance_into(&raw, &mut out, BayerGains::unity());
    }

    #[test]
    fn balancing_before_or_after_a_bilinear_demosaic_agrees() {
        // Bilinear interpolates each channel independently, so scaling a
        // channel commutes with interpolating it — up to the rounding of
        // two integer round trips. Malvar mixes channels and does *not*
        // commute, which is why white balance belongs on the mosaic.
        let reference = scene(20, 16);
        let raw: Image<BayerRggb8> = mosaic(&reference);
        let gains = BayerGains::new(1.25, 1.0, 0.75);

        let before: Image<Rgb8> = demosaic(&white_balance(&raw, gains), BayerBilinear);
        let after: Image<Rgb8> = demosaic(&raw, BayerBilinear);

        for y in 0..16 {
            for x in 0..20 {
                let b = before.pixel_at(x, y);
                let a = after.pixel_at(x, y);
                let scaled = [
                    (a.r.0 as f32 * gains.red()).round().min(255.0) as i32,
                    (a.g.0 as f32 * gains.green()).round().min(255.0) as i32,
                    (a.b.0 as f32 * gains.blue()).round().min(255.0) as i32,
                ];
                let got = [b.r.0 as i32, b.g.0 as i32, b.b.0 as i32];
                for (channel, (g, s)) in got.iter().zip(scaled.iter()).enumerate() {
                    assert!(
                        (g - s).abs() <= 1,
                        "channel {channel} at ({x}, {y}): {g} vs {s}",
                    );
                }
            }
        }
    }

    // ── Reflection preserves CFA parity ─────────────────────────────────

    #[test]
    fn the_pinned_reflection_preserves_coordinate_parity() {
        // The property the whole module rests on, checked directly against
        // the border policy rather than inferred from a demosaic result.
        use crate::pixel::Mono8;

        for len in 2..12usize {
            let row: Image<Mono8> = Image::generate(len, 1, |x, _| Mono8::new(x as u8));
            for coord in -6isize..(len as isize + 6) {
                let reflected = Mirror.pixel_at(&row, coord, 0).value() as isize;
                assert_eq!(
                    reflected.rem_euclid(2),
                    coord.rem_euclid(2),
                    "len {len}: reflecting {coord} to {reflected} flipped parity",
                );
            }
        }
    }
}

//! Bayer colour-filter-array (CFA) raw sensor pixel types.
//!
//! A single-sensor colour camera puts a mosaic of colour filters over the
//! photosites, so each sample carries *one* colour channel and which channel
//! that is depends on the sample's `(x, y)` position inside a repeating 2×2
//! tile. The four standard arrangements are:
//!
//! ```text
//! RGGB:  R G    BGGR:  B G    GRBG:  G R    GBRG:  G B
//!        G B           G R           B G           R G
//! ```
//!
//! Typed as `Image<Mono12>`, that data is indistinguishable from a monochrome
//! frame, and the compiler will happily Gaussian-blur it (smearing red into
//! green), bilinear-resize it (destroying the mosaic), crop it at an odd
//! origin (silently shifting the CFA phase so "red" samples become "green"),
//! or mix RGGB and BGGR frames from two cameras. This module gives that data
//! a type so those become compile errors — or, where the check is a runtime
//! property of the crop rectangle, a `None`.
//!
//! # The type family
//!
//! Four patterns × five bit depths = 20 `#[repr(transparent)]` newtypes:
//!
//! | Pattern | 8-bit | 10/12/14-bit | 16-bit |
//! |---|---|---|---|
//! | RGGB | [`BayerRggb8`] | [`BayerRggb10`] / [`BayerRggb12`] / [`BayerRggb14`] | [`BayerRggb16`] |
//! | BGGR | [`BayerBggr8`] | [`BayerBggr10`] / [`BayerBggr12`] / [`BayerBggr14`] | [`BayerBggr16`] |
//! | GRBG | [`BayerGrbg8`] | [`BayerGrbg10`] / [`BayerGrbg12`] / [`BayerGrbg14`] | [`BayerGrbg16`] |
//! | GBRG | [`BayerGbrg8`] | [`BayerGbrg10`] / [`BayerGbrg12`] / [`BayerGbrg14`] | [`BayerGbrg16`] |
//!
//! The sub-word depths are aliases of a const-generic
//! `BayerRggb<BITS>` (etc.) over [`Mono<BITS>`](crate::pixel::Mono), so they
//! inherit its clamping invariant: [`BayerRggb12::new(9000)`](BayerRggb12::new)
//! stores `4095`, never a value that does not fit twelve bits.
//!
//! Because every type is `#[repr(transparent)]` over its sample, a camera
//! buffer can be reinterpreted in place:
//!
//! ```
//! # use fovea::pixel::PlainPixel;
//! # use fovea::pixel::bayer::BayerRggb8;
//! let camera_bytes = [10u8, 20, 30, 40];
//! let samples: &[BayerRggb8] = BayerRggb8::cast_slice(&camera_bytes).unwrap();
//! assert_eq!(samples.len(), 4);
//! ```
//!
//! # What these types can and cannot do
//!
//! | Capability | Bayer? | Consequence |
//! |---|---|---|
//! | [`PlainPixel`] | yes | zero-copy from camera / file buffers |
//! | [`HomogeneousPixel`], [`SingleChannel`] | yes | one raw sample per pixel |
//! | [`Ord`] | yes | thresholding, min/max, saturation detection, histograms |
//! | [`LinearPixel`] | yes | convolution, defect-pixel interpolation, noise estimation |
//! | [`LinearSpace`](crate::pixel::LinearSpace) | **no** | [`blend`](crate::pixel::blend) and `Bilinear` resize are compile errors |
//! | [`OriginInvariantPixel`](crate::pixel::OriginInvariantPixel) | **no** | ordinary [`SubView::roi`](crate::image::SubView::roi), tiling, and sliding windows are unavailable |
//!
//! The two "no" rows are the point of the module. Interpolating between
//! neighbouring CFA samples mixes different colour channels, and translating
//! the origin by an odd number of pixels changes what every sample means —
//! so neither is expressible with these types.
//!
//! `LinearPixel` **without** `LinearSpace` is deliberate rather than
//! accidental: a weighted sum of raw samples is a perfectly good operation
//! (that is what defect-pixel correction and local-variance noise estimates
//! are), and its accumulator is [`MonoF32`] — a raw
//! intensity with no CFA phase, which is the honest type for a quantity
//! averaged across a neighbourhood.
//!
//! Cropping is available through the phase-preserving
//! [`BayerSubView::aligned_bayer_roi`](crate::image::BayerSubView::aligned_bayer_roi),
//! which returns `None` for an odd origin instead of lying about the pattern.
//!
//! Both refusals are the compiler's, not a runtime check that can be skipped
//! in release. Bilinear resize:
//!
//! ```compile_fail
//! use fovea::image::Image;
//! use fovea::pixel::bayer::BayerRggb12;
//! use fovea::transform::{Bilinear, resize};
//! use fovea::Size;
//!
//! let raw = Image::fill(8, 8, BayerRggb12::new(2048));
//! // ERROR: `BayerRggb12: LinearSpace` is not satisfied.
//! let _ = resize(&raw, Size::new(4, 4), Bilinear);
//! ```
//!
//! Alpha-style blending of two samples:
//!
//! ```compile_fail
//! use fovea::pixel::{blend, bayer::BayerRggb12};
//!
//! // ERROR: `BayerRggb12: LinearSpace` is not satisfied.
//! let _ = blend(&BayerRggb12::new(1000), &BayerRggb12::new(3000), 0.5);
//! ```
//!
//! Mixing two cameras' patterns at the same depth:
//!
//! ```compile_fail
//! use fovea::image::Image;
//! use fovea::pixel::bayer::{BayerBggr12, BayerRggb12};
//!
//! fn takes_rggb(_: &Image<BayerRggb12>) {}
//!
//! let bggr = Image::fill(4, 4, BayerBggr12::new(1000));
//! // ERROR: expected `&Image<BayerRggb12>`, found `&Image<BayerBggr12>`.
//! takes_rggb(&bggr);
//! ```
//!
//! What *does* compile is the arithmetic these types exist to allow — here a
//! 3×3 box blur, which is a weighted sum and therefore legitimate on raw
//! samples (it is how a defect-pixel estimate or a local noise figure is
//! built):
//!
//! ```
//! use fovea::border::Clamp;
//! use fovea::image::{Image, ImageView};
//! use fovea::pixel::bayer::BayerRggb12;
//! use fovea::transform::box_blur_3x3;
//!
//! let raw = Image::fill(8, 8, BayerRggb12::new(2048));
//! let smoothed: Image<BayerRggb12> = box_blur_3x3(&raw, &Clamp);
//! assert_eq!(smoothed.pixel_at(4, 4), BayerRggb12::new(2048));
//! ```
//!
//! # Escape hatch
//!
//! When you genuinely want coordinate-blind raw samples — writing a raw file,
//! feeding a generic monochrome statistic — name the semantic drop with the
//! [`BayerToMono`](crate::transform::BayerToMono) conversion strategy.
//!
//! # Generic code
//!
//! Algorithms bind on [`BayerPixel`], which carries the tile arrangement as
//! the compile-time constant [`BayerPixel::PATTERN`] and names the pixel type
//! demosaicing produces as [`BayerPixel::RgbOutput`]:
//!
//! ```
//! use fovea::pixel::bayer::{BayerBggr12, BayerPixel, BayerPattern, CfaColor};
//!
//! fn top_left_colour<B: BayerPixel>() -> CfaColor {
//!     B::PATTERN.color_at(0, 0)
//! }
//!
//! assert_eq!(BayerBggr12::PATTERN, BayerPattern::Bggr);
//! assert_eq!(top_left_colour::<BayerBggr12>(), CfaColor::Blue);
//! ```

use fovea_derive::{HomogeneousPixel, LinearPixel, PlainPixel, WhiteChannel, ZeroablePixel};

use std::num::Saturating;

use crate::pixel::{
    FromLinear, HomogeneousPixel, LinearPixel, Mono, MonoF32, PlainChannel, PlainPixel, Rgb, Rgb8,
    Rgb16, SingleChannel, WhiteChannel, ZeroablePixel, impl_single_channel, single_channel_sealed,
};

// ═══════════════════════════════════════════════════════════════════════════
// Pattern vocabulary
// ═══════════════════════════════════════════════════════════════════════════

/// The colour a single CFA photosite samples.
///
/// This is what a Bayer pattern resolves a coordinate to. It is *not* a pixel
/// type — a `CfaColor` names which of the three primaries a raw sample
/// measured, and carries no value.
///
/// The enum is exhaustive: a Bayer CFA has exactly these three colours, so
/// adding a variant would be a breaking change rather than a silent default
/// case.
///
/// # Examples
///
/// ```
/// # use fovea::pixel::bayer::{BayerPattern, CfaColor};
/// // In an RGGB tile the two green sites are the off-diagonal ones.
/// assert_eq!(BayerPattern::Rggb.color_at(1, 0), CfaColor::Green);
/// assert_eq!(BayerPattern::Rggb.color_at(0, 1), CfaColor::Green);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CfaColor {
    /// The site sees red.
    Red,
    /// The site sees green. Half the sites in every Bayer pattern are green.
    Green,
    /// The site sees blue.
    Blue,
}

/// The 2×2 colour-filter-array tile arrangement of a Bayer sensor.
///
/// The four variants are the four standard arrangements, named after the
/// tile read left-to-right then top-to-bottom. They correspond to the
/// GenICam SFNC pixel-format names as follows — note that SFNC abbreviates
/// the tile to its **first row**, which is why `BayerRG12` means RGGB and
/// not "red-green":
///
/// | This crate | SFNC | Tile |
/// |---|---|---|
/// | [`Rggb`](BayerPattern::Rggb) | `BayerRG…` | `R G` / `G B` |
/// | [`Bggr`](BayerPattern::Bggr) | `BayerBG…` | `B G` / `G R` |
/// | [`Grbg`](BayerPattern::Grbg) | `BayerGR…` | `G R` / `B G` |
/// | [`Gbrg`](BayerPattern::Gbrg) | `BayerGB…` | `G B` / `R G` |
///
/// The enum is exhaustive: these four patterns are the specification, so a
/// new variant would be semver-major rather than a silent default case.
///
/// A `BayerPattern` value is normally reached through
/// [`BayerPixel::PATTERN`], which makes it a compile-time constant that
/// [`color_at`](BayerPattern::color_at) folds away.
///
/// # Examples
///
/// ```
/// # use fovea::pixel::bayer::{BayerPattern, CfaColor};
/// let tile = BayerPattern::Grbg.tile();
/// assert_eq!(tile, [[CfaColor::Green, CfaColor::Red],
///                   [CfaColor::Blue,  CfaColor::Green]]);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BayerPattern {
    /// `R G` over `G B`. SFNC `BayerRG8` / `BayerRG12` / …
    Rggb,
    /// `B G` over `G R`. SFNC `BayerBG8` / `BayerBG12` / …
    Bggr,
    /// `G R` over `B G`. SFNC `BayerGR8` / `BayerGR12` / …
    Grbg,
    /// `G B` over `R G`. SFNC `BayerGB8` / `BayerGB12` / …
    Gbrg,
}

impl BayerPattern {
    /// The 2×2 tile as `[row0, row1]`, each row `[x = 0, x = 1]`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fovea::pixel::bayer::{BayerPattern, CfaColor};
    /// assert_eq!(
    ///     BayerPattern::Rggb.tile(),
    ///     [[CfaColor::Red, CfaColor::Green], [CfaColor::Green, CfaColor::Blue]]
    /// );
    /// ```
    #[inline(always)]
    pub const fn tile(self) -> [[CfaColor; 2]; 2] {
        use CfaColor::{Blue as B, Green as G, Red as R};
        match self {
            BayerPattern::Rggb => [[R, G], [G, B]],
            BayerPattern::Bggr => [[B, G], [G, R]],
            BayerPattern::Grbg => [[G, R], [B, G]],
            BayerPattern::Gbrg => [[G, B], [R, G]],
        }
    }

    /// The colour sampled at image coordinate `(x, y)`.
    ///
    /// Only the parity of `x` and `y` matters. When the pattern comes from
    /// [`BayerPixel::PATTERN`] the whole lookup is a compile-time constant
    /// and folds to a pair of parity tests.
    ///
    /// The coordinates are **image** coordinates: this is exactly why Bayer
    /// pixels are not
    /// [`OriginInvariantPixel`](crate::pixel::OriginInvariantPixel) — an
    /// odd-origin crop would change the answer for every sample it contains.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fovea::pixel::bayer::{BayerPattern, CfaColor};
    /// let p = BayerPattern::Rggb;
    /// assert_eq!(p.color_at(0, 0), CfaColor::Red);
    /// assert_eq!(p.color_at(1, 1), CfaColor::Blue);
    /// // Parity is all that matters.
    /// assert_eq!(p.color_at(640, 480), CfaColor::Red);
    /// ```
    #[inline(always)]
    pub const fn color_at(self, x: usize, y: usize) -> CfaColor {
        self.tile()[y % 2][x % 2]
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// BayerPixel — the generic-dispatch trait
// ═══════════════════════════════════════════════════════════════════════════

/// A pixel type that holds one raw sample of a Bayer colour-filter array.
///
/// Implementors associate the CFA tile arrangement — as a **compile-time
/// constant**, so pattern-dependent offsets in a demosaic kernel fold away —
/// with the RGB pixel type that demosaicing them produces.
///
/// # Supertraits
///
/// Only [`PlainPixel`], because that is the one guarantee every Bayer type
/// must make for a camera buffer to be reinterpreted in place. Algorithms
/// add what they actually use: a demosaic kernel that computes weighted sums
/// also asks for [`LinearPixel`], a caller that allocates an output image
/// also asks for `Self::RgbOutput: ZeroablePixel`. The trait is deliberately
/// **not sealed** — a downstream float CFA type or a non-Bayer mosaic with
/// the same 2×2 structure is a legitimate implementor.
///
/// # Examples
///
/// ```
/// use fovea::pixel::bayer::{BayerGrbg8, BayerPattern, BayerPixel, CfaColor};
/// use fovea::pixel::Rgb8;
///
/// fn describe<B: BayerPixel>() -> (BayerPattern, CfaColor) {
///     (B::PATTERN, B::PATTERN.color_at(0, 0))
/// }
///
/// assert_eq!(describe::<BayerGrbg8>(), (BayerPattern::Grbg, CfaColor::Green));
/// // The demosaic output type is part of the pixel type, not a parameter.
/// let _: <BayerGrbg8 as BayerPixel>::RgbOutput = Rgb8::new(0, 0, 0);
/// ```
pub trait BayerPixel: PlainPixel {
    /// The 2×2 CFA tile arrangement of the sensor this sample came from.
    const PATTERN: BayerPattern;

    /// The RGB pixel type demosaicing this sample type produces.
    ///
    /// Depth is preserved: `BayerRggb12` demosaics to
    /// [`Rgb12`](crate::pixel::Rgb12), not to `Rgb8`. Changing depth is a
    /// separate, named conversion.
    type RgbOutput: PlainPixel;
}

// ═══════════════════════════════════════════════════════════════════════════
// The 20 newtypes
// ═══════════════════════════════════════════════════════════════════════════

/// Generates one pattern's five pixel types and all their trait impls.
///
/// `macro_rules!` cannot build identifiers by concatenation and the core
/// crate has no dependencies (so no `paste!`), which is why every name is
/// spelled out at the call site. That is also the readable form: the
/// invocation below is the whole catalogue of Bayer types in the crate.
macro_rules! define_bayer_pattern {
    (
        pattern: $pattern:expr,
        tile_doc: $tile_doc:expr,
        sfnc: $sfnc:expr,
        generic: $Generic:ident,
        p8: $P8:ident, p16: $P16:ident,
        p10: $P10:ident, p12: $P12:ident, p14: $P14:ident,
    ) => {
        // ── 8-bit ───────────────────────────────────────────────────────
        #[doc = concat!("An 8-bit raw Bayer sample from a ", $tile_doc, " sensor.")]
        ///
        #[doc = concat!("GenICam SFNC name: `", $sfnc, "8`.")]
        ///
        /// See the [module documentation](self) for what this type can and
        /// cannot do; the short version is that weighted sums compile and
        /// interpolation does not.
        ///
        /// # Examples
        ///
        /// ```
        #[doc = concat!("# use fovea::pixel::bayer::", stringify!($P8), ";")]
        #[doc = concat!("let s = ", stringify!($P8), "::new(42);")]
        /// assert_eq!(s.value(), 42);
        /// ```
        #[repr(transparent)]
        #[derive(
            Clone,
            Copy,
            Debug,
            PartialEq,
            Eq,
            Hash,
            Ord,
            PartialOrd,
            PlainPixel,
            HomogeneousPixel,
            ZeroablePixel,
            LinearPixel,
            WhiteChannel,
        )]
        // A weighted sum of raw samples is meaningful (defect-pixel
        // interpolation, noise estimation); interpolating *between* them is
        // not, because neighbouring samples are different colours. Hence
        // `no_space` — see the module docs.
        #[linear(accumulator = MonoF32, no_space)]
        pub struct $P8(Saturating<u8>);

        impl $P8 {
            #[doc = concat!("Creates a `", stringify!($P8), "` from a raw 8-bit sample.")]
            #[inline]
            pub const fn new(value: u8) -> Self {
                $P8(Saturating(value))
            }

            /// Returns the raw sample value.
            ///
            /// This is the *channel* scalar, not a monochrome pixel: it
            /// carries no claim that the number is an intensity you may
            /// treat as grey. Converting the whole image to a monochrome
            /// pixel type is a named operation —
            /// [`BayerToMono`](crate::transform::BayerToMono).
            #[inline]
            pub const fn value(self) -> u8 {
                self.0.0
            }
        }

        impl From<$P8> for u8 {
            #[inline]
            fn from(p: $P8) -> u8 {
                p.0.0
            }
        }

        impl From<u8> for $P8 {
            #[inline]
            fn from(v: u8) -> Self {
                $P8::new(v)
            }
        }

        impl BayerPixel for $P8 {
            const PATTERN: BayerPattern = $pattern;
            type RgbOutput = Rgb8;
        }

        // ── 16-bit ──────────────────────────────────────────────────────
        #[doc = concat!("A 16-bit raw Bayer sample from a ", $tile_doc, " sensor.")]
        ///
        #[doc = concat!("GenICam SFNC name: `", $sfnc, "16`.")]
        ///
        /// See the [module documentation](self) for the capability table.
        ///
        /// # Examples
        ///
        /// ```
        #[doc = concat!("# use fovea::pixel::bayer::", stringify!($P16), ";")]
        #[doc = concat!("let s = ", stringify!($P16), "::new(4242);")]
        /// assert_eq!(s.value(), 4242);
        /// ```
        #[repr(transparent)]
        #[derive(
            Clone,
            Copy,
            Debug,
            PartialEq,
            Eq,
            Hash,
            Ord,
            PartialOrd,
            PlainPixel,
            HomogeneousPixel,
            ZeroablePixel,
            LinearPixel,
            WhiteChannel,
        )]
        #[linear(accumulator = MonoF32, no_space)]
        pub struct $P16(Saturating<u16>);

        impl $P16 {
            #[doc = concat!("Creates a `", stringify!($P16), "` from a raw 16-bit sample.")]
            #[inline]
            pub const fn new(value: u16) -> Self {
                $P16(Saturating(value))
            }

            /// Returns the raw sample value.
            ///
            /// See [`BayerToMono`](crate::transform::BayerToMono) for the
            /// named whole-image conversion.
            #[inline]
            pub const fn value(self) -> u16 {
                self.0.0
            }
        }

        impl From<$P16> for u16 {
            #[inline]
            fn from(p: $P16) -> u16 {
                p.0.0
            }
        }

        impl From<u16> for $P16 {
            #[inline]
            fn from(v: u16) -> Self {
                $P16::new(v)
            }
        }

        impl BayerPixel for $P16 {
            const PATTERN: BayerPattern = $pattern;
            type RgbOutput = Rgb16;
        }

        // ── 10 / 12 / 14-bit ────────────────────────────────────────────
        #[doc = concat!(
                    "A sub-word (10/12/14-bit) raw Bayer sample from a ", $tile_doc, " sensor."
                )]
        ///
        /// `BITS` must be 10, 12, or 14 — the same compile-time constraint
        /// [`Mono<BITS>`](crate::pixel::Mono) enforces, since that is the
        /// storage this wraps. Use the
        #[doc = concat!(
                    "[`", stringify!($P10), "`] / [`", stringify!($P12),
                    "`] / [`", stringify!($P14), "`] aliases."
                )]
        ///
        /// Values above the depth maximum are **clamped**, not wrapped or
        /// rejected, exactly as `Mono<BITS>` clamps them.
        ///
        /// # Examples
        ///
        /// ```
        #[doc = concat!("# use fovea::pixel::bayer::", stringify!($P12), ";")]
        #[doc = concat!("let s = ", stringify!($P12), "::new(9000);")]
        /// assert_eq!(s.value(), 4095); // clamped to the 12-bit maximum
        /// ```
        #[repr(transparent)]
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
        pub struct $Generic<const BITS: usize>(Mono<BITS>);

        #[doc = concat!("A 10-bit raw Bayer sample from a ", $tile_doc, " sensor.")]
        ///
        #[doc = concat!("GenICam SFNC name: `", $sfnc, "10`.")]
        pub type $P10 = $Generic<10>;
        #[doc = concat!("A 12-bit raw Bayer sample from a ", $tile_doc, " sensor.")]
        ///
        #[doc = concat!("GenICam SFNC name: `", $sfnc, "12`.")]
        pub type $P12 = $Generic<12>;
        #[doc = concat!("A 14-bit raw Bayer sample from a ", $tile_doc, " sensor.")]
        ///
        #[doc = concat!("GenICam SFNC name: `", $sfnc, "14`.")]
        pub type $P14 = $Generic<14>;

        impl<const BITS: usize> $Generic<BITS> {
            #[doc = concat!(
                        "Creates a `", stringify!($Generic),
                        "<BITS>`, clamping `value` to the bit-depth maximum."
                    )]
            #[inline]
            pub fn new(value: u16) -> Self {
                $Generic(Mono::new(value))
            }

            /// Returns the raw sample value.
            ///
            /// See [`BayerToMono`](crate::transform::BayerToMono) for the
            /// named whole-image conversion.
            #[inline]
            pub fn value(self) -> u16 {
                self.0.value()
            }
        }

        impl<const BITS: usize> From<$Generic<BITS>> for u16 {
            #[inline]
            fn from(p: $Generic<BITS>) -> u16 {
                p.0.value()
            }
        }

        impl<const BITS: usize> From<u16> for $Generic<BITS> {
            #[inline]
            fn from(v: u16) -> Self {
                $Generic::new(v)
            }
        }

        impl<const BITS: usize> BayerPixel for $Generic<BITS> {
            const PATTERN: BayerPattern = $pattern;
            type RgbOutput = Rgb<BITS>;
        }

        // Hand-written trait impls for the const-generic variant. The
        // derive macros take concrete types only, so this mirrors what
        // `Mono<BITS>` does for the same reason.
        //
        // SAFETY: `#[repr(transparent)]` over `Mono<BITS>`, which is itself
        // `#[repr(transparent)]` over `Saturating<u16>` — layout, size, and
        // bit-pattern validity are inherited from `u16`.
        unsafe impl<const BITS: usize> PlainChannel for $Generic<BITS> {}
        unsafe impl<const BITS: usize> PlainPixel for $Generic<BITS> {
            const CHANNELS: &'static [usize] = &[2];
        }

        // The channel is the raw `u16` sample, not the wrapped `Mono<BITS>`
        // pixel: a Bayer sample has exactly one *channel*, and a pixel is
        // not a channel of an outer pixel (design principle §9). The layout
        // claim holds transitively through the two transparent wrappers.
        unsafe impl<const BITS: usize> HomogeneousPixel for $Generic<BITS> {
            type Channel = Saturating<u16>;
            type Channels = [Saturating<u16>; 1];
        }

        impl<const BITS: usize> ZeroablePixel for $Generic<BITS> {
            #[inline]
            fn zero() -> Self {
                $Generic(<Mono<BITS> as ZeroablePixel>::zero())
            }
        }

        // Reduced-range pixels must not use the `WhiteChannel` derive: the
        // channel type's `BoundedChannel::MAX` is 65535, which would break
        // the `BITS`-bit invariant when written back through
        // `from_channels`. Delegating to `Mono<BITS>` returns
        // `(1 << BITS) - 1` instead.
        impl<const BITS: usize> WhiteChannel for $Generic<BITS> {
            #[inline(always)]
            fn white_channel() -> Saturating<u16> {
                <Mono<BITS> as WhiteChannel>::white_channel()
            }
        }

        // `LinearPixel` **without** `LinearSpace` — the const-generic
        // counterpart of `#[linear(accumulator = MonoF32, no_space)]` on the
        // fixed-width variants above. The missing `impl LinearSpace` is the
        // load-bearing part of this block; do not add one.
        impl<const BITS: usize> LinearPixel for $Generic<BITS> {
            type Accumulator = MonoF32;
            #[inline(always)]
            fn to_accumulator(&self) -> MonoF32 {
                <Mono<BITS> as LinearPixel>::to_accumulator(&self.0)
            }
            #[inline(always)]
            fn scale(&self, scalar: f32) -> MonoF32 {
                <Mono<BITS> as LinearPixel>::scale(&self.0, scalar)
            }
            #[inline(always)]
            fn scale_add(&self, scalar: f32, addend: MonoF32) -> MonoF32 {
                <Mono<BITS> as LinearPixel>::scale_add(&self.0, scalar, addend)
            }
            #[inline(always)]
            fn uniform(scalar: f32) -> MonoF32 {
                MonoF32(scalar)
            }
        }

        impl<const BITS: usize> FromLinear<MonoF32> for $Generic<BITS> {
            #[inline(always)]
            fn from_linear(acc: MonoF32) -> Self {
                $Generic(<Mono<BITS> as FromLinear<MonoF32>>::from_linear(acc))
            }
        }

        // Channel-wise arithmetic, matching what `#[derive(LinearPixel)]`
        // emits for the fixed-width variants so the family has one surface
        // at all five depths. The inner `Mono<BITS>` re-clamps.
        impl<const BITS: usize> std::ops::Add for $Generic<BITS> {
            type Output = Self;
            #[inline]
            fn add(self, other: Self) -> Self {
                $Generic(self.0 + other.0)
            }
        }

        impl<const BITS: usize> std::ops::Sub for $Generic<BITS> {
            type Output = Self;
            #[inline]
            fn sub(self, other: Self) -> Self {
                $Generic(self.0 - other.0)
            }
        }

        impl<const BITS: usize> std::ops::Mul for $Generic<BITS> {
            type Output = Self;
            #[inline]
            fn mul(self, other: Self) -> Self {
                $Generic(self.0 * other.0)
            }
        }

        // One raw sample per pixel, at every depth.
        impl_single_channel!($P8, $P16);
        impl<const BITS: usize> single_channel_sealed::Sealed for $Generic<BITS> {}
        impl<const BITS: usize> SingleChannel for $Generic<BITS> {}
    };
}

define_bayer_pattern! {
    pattern: BayerPattern::Rggb,
    tile_doc: "`R G` / `G B` (RGGB)",
    sfnc: "BayerRG",
    generic: BayerRggb,
    p8: BayerRggb8, p16: BayerRggb16,
    p10: BayerRggb10, p12: BayerRggb12, p14: BayerRggb14,
}

define_bayer_pattern! {
    pattern: BayerPattern::Bggr,
    tile_doc: "`B G` / `G R` (BGGR)",
    sfnc: "BayerBG",
    generic: BayerBggr,
    p8: BayerBggr8, p16: BayerBggr16,
    p10: BayerBggr10, p12: BayerBggr12, p14: BayerBggr14,
}

define_bayer_pattern! {
    pattern: BayerPattern::Grbg,
    tile_doc: "`G R` / `B G` (GRBG)",
    sfnc: "BayerGR",
    generic: BayerGrbg,
    p8: BayerGrbg8, p16: BayerGrbg16,
    p10: BayerGrbg10, p12: BayerGrbg12, p14: BayerGrbg14,
}

define_bayer_pattern! {
    pattern: BayerPattern::Gbrg,
    tile_doc: "`G B` / `R G` (GBRG)",
    sfnc: "BayerGB",
    generic: BayerGbrg,
    p8: BayerGbrg8, p16: BayerGbrg16,
    p10: BayerGbrg10, p12: BayerGbrg12, p14: BayerGbrg14,
}

// ---------------------------------------------------------------------------
// Deliberate non-impls
// ---------------------------------------------------------------------------
//
// There is intentionally no `impl_origin_invariant_pixel!` and no
// `impl LinearSpace` anywhere in this file. Both omissions are load-bearing:
//
//   * no `OriginInvariantPixel` → `SubView::roi`, `tiles`, and
//     `sliding_windows` do not exist for `Image<BayerRggb12>`. The
//     replacement is `BayerSubView::aligned_bayer_roi`, which checks the
//     origin parity at runtime and returns `None` rather than handing back a
//     view whose pattern silently changed.
//   * no `LinearSpace` → `blend()` and `Bilinear` resize are compile errors.
//     Both mix neighbouring samples, which in a mosaic means mixing colours.
//
// There is also no `IntegralPixel` / `IntegralSquaredPixel`: a summed-area
// table over a mosaic adds red to green to blue, which is not a quantity.
//
// Adding any of these impls would silently remove a guarantee this module
// exists to provide. If a future caller needs one of the operations, it needs
// a *named* API (`aligned_bayer_roi`, `BayerToMono`, a demosaic strategy),
// not a widened trait bound.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pixel::{LinearSpace, Mono12, PlainChannel};

    // ── Pattern vocabulary ──────────────────────────────────────────────

    #[test]
    fn every_tile_has_two_greens_one_red_one_blue() {
        for p in [
            BayerPattern::Rggb,
            BayerPattern::Bggr,
            BayerPattern::Grbg,
            BayerPattern::Gbrg,
        ] {
            let flat: Vec<CfaColor> = p.tile().into_iter().flatten().collect();
            assert_eq!(
                flat.iter().filter(|c| **c == CfaColor::Green).count(),
                2,
                "{p:?} must have two green sites"
            );
            assert_eq!(flat.iter().filter(|c| **c == CfaColor::Red).count(), 1);
            assert_eq!(flat.iter().filter(|c| **c == CfaColor::Blue).count(), 1);
        }
    }

    #[test]
    fn color_at_matches_the_tile_for_every_parity() {
        for p in [
            BayerPattern::Rggb,
            BayerPattern::Bggr,
            BayerPattern::Grbg,
            BayerPattern::Gbrg,
        ] {
            let tile = p.tile();
            for y in 0..7usize {
                for x in 0..7usize {
                    assert_eq!(p.color_at(x, y), tile[y % 2][x % 2], "{p:?} at ({x}, {y})");
                }
            }
        }
    }

    #[test]
    fn the_four_patterns_are_the_four_odd_origin_shifts_of_each_other() {
        // Shifting an RGGB frame by one column yields GRBG, by one row GBRG,
        // by one of each BGGR. This is the table that makes an odd-origin
        // crop a lie, so it is worth pinning even though phase-adjusting ROI
        // is deferred.
        let r = BayerPattern::Rggb;
        for (dx, dy, expected) in [
            (1usize, 0usize, BayerPattern::Grbg),
            (0, 1, BayerPattern::Gbrg),
            (1, 1, BayerPattern::Bggr),
        ] {
            for y in 0..4usize {
                for x in 0..4usize {
                    assert_eq!(
                        r.color_at(x + dx, y + dy),
                        expected.color_at(x, y),
                        "shift ({dx}, {dy}) at ({x}, {y})"
                    );
                }
            }
        }
    }

    #[test]
    fn pattern_const_is_distinct_per_family() {
        assert_eq!(BayerRggb8::PATTERN, BayerPattern::Rggb);
        assert_eq!(BayerBggr16::PATTERN, BayerPattern::Bggr);
        assert_eq!(BayerGrbg12::PATTERN, BayerPattern::Grbg);
        assert_eq!(BayerGbrg10::PATTERN, BayerPattern::Gbrg);
    }

    // ── Storage ─────────────────────────────────────────────────────────

    #[test]
    fn transparent_layout_matches_the_underlying_sample() {
        assert_eq!(size_of::<BayerRggb8>(), size_of::<u8>());
        assert_eq!(size_of::<BayerRggb16>(), size_of::<u16>());
        assert_eq!(size_of::<BayerRggb12>(), size_of::<u16>());
        assert_eq!(<BayerRggb8 as PlainChannel>::SIZE, 1);
        assert_eq!(<BayerRggb12 as PlainChannel>::SIZE, 2);
    }

    #[test]
    fn cast_slice_reinterprets_a_camera_buffer_without_copying() {
        let raw = [1u8, 2, 3, 4];
        let samples = BayerRggb8::cast_slice(&raw).unwrap();
        assert_eq!(samples.len(), 4);
        assert_eq!(samples[2].value(), 3);
        assert_eq!(samples.as_ptr() as usize, raw.as_ptr() as usize);
    }

    #[test]
    fn sub_word_depths_clamp_like_mono() {
        assert_eq!(BayerRggb10::new(2000).value(), 1023);
        assert_eq!(BayerRggb12::new(9000).value(), 4095);
        assert_eq!(BayerRggb14::new(65535).value(), 16383);
        assert_eq!(BayerRggb12::new(1234).value(), 1234);
    }

    #[test]
    fn white_channel_is_the_depth_maximum_not_the_storage_maximum() {
        assert_eq!(
            <BayerRggb12 as WhiteChannel>::white_channel(),
            <Mono12 as WhiteChannel>::white_channel()
        );
        assert_eq!(
            <BayerRggb12 as WhiteChannel>::white_channel(),
            Saturating(4095)
        );
        assert_eq!(
            <BayerRggb16 as WhiteChannel>::white_channel(),
            Saturating(u16::MAX)
        );
    }

    #[test]
    fn zero_is_the_zero_sample() {
        assert_eq!(<BayerGbrg14 as ZeroablePixel>::zero().value(), 0);
        assert_eq!(<BayerGbrg8 as ZeroablePixel>::zero().value(), 0);
    }

    #[test]
    fn round_trips_through_the_raw_scalar() {
        for v in [0u8, 1, 42, 255] {
            let p: BayerBggr8 = v.into();
            let back: u8 = p.into();
            assert_eq!(back, v);
        }
        for v in [0u16, 1, 4095] {
            let p: BayerGrbg12 = v.into();
            let back: u16 = p.into();
            assert_eq!(back, v);
        }
    }

    // ── Arithmetic ──────────────────────────────────────────────────────

    #[test]
    fn weighted_sums_are_available_at_every_depth() {
        // The operation `LinearPixel` exists for: a weighted sum of raw
        // samples (defect-pixel interpolation, noise estimation).
        let a = BayerRggb8::new(100);
        let b = BayerRggb8::new(200);
        assert_eq!(a.scale_add(0.5, b.scale(0.5)), MonoF32(150.0));

        let c = BayerRggb12::new(1000);
        let d = BayerRggb12::new(3000);
        assert_eq!(c.scale_add(0.5, d.scale(0.5)), MonoF32(2000.0));
    }

    #[test]
    fn from_linear_returns_to_the_bayer_type_and_re_clamps() {
        let back: BayerRggb12 = FromLinear::from_linear(MonoF32(9000.0));
        assert_eq!(back.value(), 4095);
        let back: BayerRggb8 = FromLinear::from_linear(MonoF32(42.4));
        assert_eq!(back.value(), 42);
    }

    #[test]
    fn channel_wise_arithmetic_is_uniform_across_depths() {
        assert_eq!(
            BayerRggb8::new(100) + BayerRggb8::new(50),
            BayerRggb8::new(150)
        );
        assert_eq!(
            BayerRggb12::new(1000) + BayerRggb12::new(500),
            BayerRggb12::new(1500)
        );
        assert_eq!(
            BayerRggb12::new(1000) - BayerRggb12::new(500),
            BayerRggb12::new(500)
        );
    }

    // ── Negative space: the impls that must not exist ───────────────────

    #[test]
    fn bayer_types_are_not_in_a_linear_space() {
        // A compile-time property, asserted at compile time: the function is
        // only callable for types that carry the marker, and no Bayer type
        // is passed to it. The `compile_fail` doctests in `crate::pixel` are
        // the executable half of this claim.
        fn assert_linear_space<P: LinearSpace>() {}
        assert_linear_space::<Mono12>();
        // assert_linear_space::<BayerRggb12>();  // ← would not compile
    }

    #[test]
    fn every_shipped_bayer_type_implements_bayer_pixel() {
        fn assert_bayer<B: BayerPixel>() {}
        macro_rules! all {
            ($($t:ty),+ $(,)?) => {{ $( assert_bayer::<$t>(); )+ }};
        }
        all!(
            BayerRggb8,
            BayerRggb10,
            BayerRggb12,
            BayerRggb14,
            BayerRggb16,
            BayerBggr8,
            BayerBggr10,
            BayerBggr12,
            BayerBggr14,
            BayerBggr16,
            BayerGrbg8,
            BayerGrbg10,
            BayerGrbg12,
            BayerGrbg14,
            BayerGrbg16,
            BayerGbrg8,
            BayerGbrg10,
            BayerGbrg12,
            BayerGbrg14,
            BayerGbrg16,
        );
    }

    #[test]
    fn distinct_patterns_are_distinct_types_at_the_same_depth() {
        // Not a runtime assertion so much as a compile-time one: if
        // `BayerRggb12` and `BayerBggr12` were the same type this would not
        // build, and `Image<BayerRggb12>` could be handed to code expecting
        // BGGR data.
        fn output_of<B: BayerPixel>() -> BayerPattern {
            B::PATTERN
        }
        assert_ne!(output_of::<BayerRggb12>(), output_of::<BayerBggr12>());
    }
}

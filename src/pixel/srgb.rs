//! sRGB gamma-encoded pixel types.
//!
//! Same memory layouts as their linear counterparts (`Rgb8`, `Rgba8`, etc.)
//! but representing sRGB-encoded values. These types intentionally do NOT
//! implement `LinearPixel` or `LinearSpace`, so the compiler rejects
//! attempts to use them with interpolation algorithms. Convert to linear
//! light first with the `SrgbGamma` strategy.

use fovea_derive::{HomogeneousPixel, PlainPixel, WhiteChannel, ZeroablePixel};

use std::num::Saturating;

use crate::pixel::impl_origin_invariant_pixel;

// ═══════════════════════════════════════════════════════════════════════════════
// sRGB gamma-encoded pixel types
//
// Same memory layout as Rgb8 / Rgba8 but representing sRGB-encoded values.
// These types intentionally do NOT implement `LinearPixel` or `LinearSpace`,
// so they cannot be passed to algorithms that assume linear light (e.g.
// bilinear resize).  Convert to linear with the `SrgbGamma` strategy first.
// ═══════════════════════════════════════════════════════════════════════════════

/// sRGB-encoded RGB pixel with 8-bit depth per channel.
///
/// This type has the same memory layout as [`Rgb8`](crate::pixel::Rgb8) (three `Saturating<u8>`
/// in R, G, B order) but represents **gamma-encoded** sRGB values.
///
/// It does **not** implement [`LinearPixel`](crate::pixel::LinearPixel) or [`LinearSpace`](crate::pixel::LinearSpace), so the
/// compiler will reject attempts to use it with interpolation algorithms
/// like bilinear resize.  Convert to linear light first using the
/// `SrgbGamma` conversion strategy:
///
/// ```
/// # use fovea::pixel::{Srgb8, RgbF32};
/// # use fovea::transform::{ConvertPixel, SrgbGamma};
/// let srgb = Srgb8::new(128, 64, 200);
/// let linear: RgbF32 = SrgbGamma.convert(&srgb);
/// ```
#[repr(C)]
#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    Hash,
    PlainPixel,
    HomogeneousPixel,
    ZeroablePixel,
    WhiteChannel,
)]
pub struct Srgb8 {
    pub r: Saturating<u8>,
    pub g: Saturating<u8>,
    pub b: Saturating<u8>,
}

impl Srgb8 {
    pub fn new(r: u8, g: u8, b: u8) -> Self {
        Srgb8 {
            r: Saturating(r),
            g: Saturating(g),
            b: Saturating(b),
        }
    }
}

/// sRGB-encoded RGBA pixel with 8-bit depth per channel.
///
/// This type has the same memory layout as [`Rgba8`](crate::pixel::Rgba8) (four `Saturating<u8>`
/// in R, G, B, A order) but the R, G, B channels represent **gamma-encoded**
/// sRGB values.  The alpha channel is always linear, as required by the sRGB
/// specification.
///
/// It does **not** implement [`LinearPixel`](crate::pixel::LinearPixel) or [`LinearSpace`](crate::pixel::LinearSpace).  Convert to
/// linear light first using the `SrgbGamma` conversion strategy:
///
/// ```
/// # use fovea::pixel::{Srgba8, RgbaF32};
/// # use fovea::transform::{ConvertPixel, SrgbGamma};
/// let srgb = Srgba8::new(128, 64, 200, 255);
/// let linear: RgbaF32 = SrgbGamma.convert(&srgb);
/// ```
#[repr(C)]
#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    Hash,
    PlainPixel,
    HomogeneousPixel,
    ZeroablePixel,
    WhiteChannel,
)]
pub struct Srgba8 {
    pub r: Saturating<u8>,
    pub g: Saturating<u8>,
    pub b: Saturating<u8>,
    pub a: Saturating<u8>,
}

impl Srgba8 {
    pub fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Srgba8 {
            r: Saturating(r),
            g: Saturating(g),
            b: Saturating(b),
            a: Saturating(a),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// sRGB gamma-encoded grayscale pixel types
//
// Same memory layout as Mono8 / MonoA8 but representing sRGB-encoded values.
// These types intentionally do NOT implement `LinearPixel` or `LinearSpace`,
// so they cannot be passed to algorithms that assume linear light (e.g.
// bilinear resize).  Convert to linear with the `SrgbGamma` strategy first.
//
// Why "sRGB" for a grayscale pixel?
// The sRGB specification (IEC 61966-2-1) defines a single transfer function
// (gamma curve) that applies identically to each channel.  For a grayscale
// image there is only one channel, but the *same* transfer function applies.
// PNG 8-bit grayscale is sRGB-encoded by convention, so calling the type
// `SrgbMono8` is technically correct — the best kind of correct. 😉
// ═══════════════════════════════════════════════════════════════════════════════

/// sRGB-encoded grayscale pixel with 8-bit depth.
///
/// This type has the same memory layout as [`Mono8`](crate::pixel::Mono8) (a single
/// `Saturating<u8>`) but represents a **gamma-encoded** sRGB value.
///
/// It does **not** implement [`LinearPixel`](crate::pixel::LinearPixel) or [`LinearSpace`](crate::pixel::LinearSpace), so the
/// compiler will reject attempts to use it with interpolation algorithms
/// like bilinear resize.  Convert to linear light first using the
/// `SrgbGamma` conversion strategy:
///
/// ```
/// # use fovea::pixel::{SrgbMono8, MonoF32};
/// # use fovea::transform::{ConvertPixel, SrgbGamma};
/// let srgb = SrgbMono8::new(128);
/// let linear: MonoF32 = SrgbGamma.convert(&srgb);
/// // Mid-gray sRGB ≈ 0.216 linear, not 0.502
/// assert!((linear.0 - 0.216).abs() < 0.001);
/// ```
///
/// # Why "sRGB" for grayscale?
///
/// The sRGB specification (IEC 61966-2-1) defines a single transfer
/// function that applies per-channel.  A grayscale image has one channel
/// but uses the exact same curve.  PNG 8-bit grayscale is sRGB-encoded
/// by convention.
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
    WhiteChannel,
)]
pub struct SrgbMono8(pub Saturating<u8>);

impl SrgbMono8 {
    pub fn new(value: u8) -> Self {
        SrgbMono8(Saturating(value))
    }
}

/// sRGB-encoded grayscale-with-alpha pixel, 8-bit depth per channel.
///
/// This type has the same memory layout as [`MonoA8`](crate::pixel::MonoA8) (two `Saturating<u8>`
/// in V, A order) but the value channel represents a **gamma-encoded** sRGB
/// intensity.  The alpha channel is always linear, as required by the sRGB
/// specification.
///
/// It does **not** implement [`LinearPixel`](crate::pixel::LinearPixel) or [`LinearSpace`](crate::pixel::LinearSpace), so the
/// compiler will reject attempts to use it with interpolation algorithms
/// like bilinear resize.  Convert to linear light first using the
/// `SrgbGamma` conversion strategy.
///
/// # Why "sRGB" for grayscale?
///
/// See [`SrgbMono8`] — the sRGB transfer function is defined per-channel
/// and applies identically to a single-channel image.
#[repr(C)]
#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    Hash,
    PlainPixel,
    HomogeneousPixel,
    ZeroablePixel,
    WhiteChannel,
)]
pub struct SrgbMonoA8 {
    pub v: Saturating<u8>,
    pub a: Saturating<u8>,
}

impl SrgbMonoA8 {
    pub fn new(v: u8, a: u8) -> Self {
        SrgbMonoA8 {
            v: Saturating(v),
            a: Saturating(a),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// sRGB gamma-encoded 16-bit pixel types
//
// Same memory layout as Rgb16 / Rgba16 / Mono16 / MonoA16 but representing
// sRGB-encoded values.  These types intentionally do NOT implement
// `LinearPixel` or `LinearSpace`, so they cannot be passed to algorithms
// that assume linear light (e.g. bilinear resize).  Convert to linear with
// the `SrgbGamma` strategy first.
//
// 16-bit sRGB PNGs are not rare: any 16-bit PNG exported from Photoshop,
// GIMP, Lightroom, Krita, or darktable in an sRGB working space carries
// sRGB/iCCP metadata.  The type encodes the transfer function so the
// compiler can enforce correct usage.
// ═══════════════════════════════════════════════════════════════════════════════

/// sRGB-encoded RGB pixel with 16-bit depth per channel.
///
/// This type has the same memory layout as [`Rgb16`](crate::pixel::Rgb16) (three `Saturating<u16>`
/// in R, G, B order) but represents **gamma-encoded** sRGB values.
///
/// It does **not** implement [`LinearPixel`](crate::pixel::LinearPixel) or [`LinearSpace`](crate::pixel::LinearSpace), so the
/// compiler will reject attempts to use it with interpolation algorithms
/// like bilinear resize.  Convert to linear light first using the
/// `SrgbGamma` conversion strategy:
///
/// ```
/// # use fovea::pixel::{Srgb16, RgbF32};
/// # use fovea::transform::{ConvertPixel, SrgbGamma};
/// let srgb = Srgb16::new(32768, 16384, 65535);
/// let linear: RgbF32 = SrgbGamma.convert(&srgb);
/// ```
#[repr(C)]
#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    Hash,
    PlainPixel,
    HomogeneousPixel,
    ZeroablePixel,
    WhiteChannel,
)]
pub struct Srgb16 {
    pub r: Saturating<u16>,
    pub g: Saturating<u16>,
    pub b: Saturating<u16>,
}

impl Srgb16 {
    pub fn new(r: u16, g: u16, b: u16) -> Self {
        Srgb16 {
            r: Saturating(r),
            g: Saturating(g),
            b: Saturating(b),
        }
    }
}

/// sRGB-encoded RGBA pixel with 16-bit depth per channel.
///
/// This type has the same memory layout as [`Rgba16`](crate::pixel::Rgba16) (four `Saturating<u16>`
/// in R, G, B, A order) but the R, G, B channels represent **gamma-encoded**
/// sRGB values.  The alpha channel is always linear, as required by the sRGB
/// specification.
///
/// It does **not** implement [`LinearPixel`](crate::pixel::LinearPixel) or [`LinearSpace`](crate::pixel::LinearSpace).  Convert to
/// linear light first using the `SrgbGamma` conversion strategy:
///
/// ```
/// # use fovea::pixel::{Srgba16, RgbaF32};
/// # use fovea::transform::{ConvertPixel, SrgbGamma};
/// let srgb = Srgba16::new(32768, 16384, 65535, 65535);
/// let linear: RgbaF32 = SrgbGamma.convert(&srgb);
/// ```
#[repr(C)]
#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    Hash,
    PlainPixel,
    HomogeneousPixel,
    ZeroablePixel,
    WhiteChannel,
)]
pub struct Srgba16 {
    pub r: Saturating<u16>,
    pub g: Saturating<u16>,
    pub b: Saturating<u16>,
    pub a: Saturating<u16>,
}

impl Srgba16 {
    pub fn new(r: u16, g: u16, b: u16, a: u16) -> Self {
        Srgba16 {
            r: Saturating(r),
            g: Saturating(g),
            b: Saturating(b),
            a: Saturating(a),
        }
    }
}

/// sRGB-encoded grayscale pixel with 16-bit depth.
///
/// This type has the same memory layout as [`Mono16`](crate::pixel::Mono16) (a single
/// `Saturating<u16>`) but represents a **gamma-encoded** sRGB value.
///
/// It does **not** implement [`LinearPixel`](crate::pixel::LinearPixel) or [`LinearSpace`](crate::pixel::LinearSpace), so the
/// compiler will reject attempts to use it with interpolation algorithms
/// like bilinear resize.  Convert to linear light first using the
/// `SrgbGamma` conversion strategy:
///
/// ```
/// # use fovea::pixel::{SrgbMono16, MonoF32};
/// # use fovea::transform::{ConvertPixel, SrgbGamma};
/// let srgb = SrgbMono16::new(32768);
/// let linear: MonoF32 = SrgbGamma.convert(&srgb);
/// ```
///
/// # Why "sRGB" for grayscale?
///
/// See [`SrgbMono8`] — the sRGB transfer function is defined per-channel
/// and applies identically to a single-channel image.
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
    WhiteChannel,
)]
pub struct SrgbMono16(pub Saturating<u16>);

impl SrgbMono16 {
    pub fn new(value: u16) -> Self {
        SrgbMono16(Saturating(value))
    }
}

/// sRGB-encoded grayscale-with-alpha pixel, 16-bit depth per channel.
///
/// This type has the same memory layout as [`MonoA16`](crate::pixel::MonoA16) (two `Saturating<u16>`
/// in V, A order) but the value channel represents a **gamma-encoded** sRGB
/// intensity.  The alpha channel is always linear, as required by the sRGB
/// specification.
///
/// It does **not** implement [`LinearPixel`](crate::pixel::LinearPixel) or [`LinearSpace`](crate::pixel::LinearSpace), so the
/// compiler will reject attempts to use it with interpolation algorithms
/// like bilinear resize.  Convert to linear light first using the
/// `SrgbGamma` conversion strategy.
///
/// # Why "sRGB" for grayscale?
///
/// See [`SrgbMono8`] — the sRGB transfer function is defined per-channel
/// and applies identically to a single-channel image.
#[repr(C)]
#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    Hash,
    PlainPixel,
    HomogeneousPixel,
    ZeroablePixel,
    WhiteChannel,
)]
pub struct SrgbMonoA16 {
    pub v: Saturating<u16>,
    pub a: Saturating<u16>,
}

impl SrgbMonoA16 {
    pub fn new(v: u16, a: u16) -> Self {
        SrgbMonoA16 {
            v: Saturating(v),
            a: Saturating(a),
        }
    }
}

// ---------------------------------------------------------------------------
// OriginInvariantPixel impls
// ---------------------------------------------------------------------------
//
// The sRGB gamma encoding is a per-pixel value transform, not a
// coordinate-dependent one: a gamma-encoded sample means the same thing
// wherever it sits, so cropping preserves its meaning. (These types still
// reject *interpolation* by withholding `LinearSpace`; origin-invariance and
// linear-space membership are independent axes — Philosophy §2.)
impl_origin_invariant_pixel!(
    Srgb8, Srgba8, SrgbMono8, SrgbMonoA8, Srgb16, Srgba16, SrgbMono16, SrgbMonoA16,
);

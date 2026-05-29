use core::marker::PhantomData;

#[cfg(test)]
use crate::image::ImageView;
use crate::image::{Image, RasterImage, RasterImageMut};
use crate::pixel::{
    Array, Bgr8, Bgr16, Bgr32, Bgr64, BgrF32, BgrF64, Bgra8, Bgra16, Bgra32, Bgra64, BgraF32,
    BgraF64, HomogeneousPixel, Indexed8, Mono, Mono8, Mono16, Mono32, Mono64, MonoA8, MonoA16,
    MonoA32, MonoA64, MonoAF32, MonoAF64, MonoF32, MonoF64, PlainChannel, Rgb8, Rgb16, Rgb32,
    Rgb64, RgbF32, RgbF64, Rgba8, Rgba16, Rgba32, Rgba64, RgbaF32, RgbaF64, Srgb8, Srgb16,
    SrgbMono8, SrgbMono16, SrgbMonoA8, SrgbMonoA16, Srgba8, Srgba16, WhiteChannel, ZeroablePixel,
};

// ───────────────────────────────────────────────────────────────────────────────
// Core trait
// ───────────────────────────────────────────────────────────────────────────────

/// Pixel conversion parameterized by strategy.
///
/// The `ConvertPixel` trait decouples the conversion algorithm from the conversion
/// function, following the same pattern as [`super::ResizeMethod`].  Different
/// strategies express different semantics for how pixel values are mapped between
/// types.
///
/// # Type Parameters
/// - `Src`: Source pixel type.
/// - `Dst`: Destination pixel type.
///
/// # Implementing a custom strategy
/// ```
/// # use fovea::pixel::{Rgb8, Mono8};
/// # use fovea::transform::ConvertPixel;
/// struct MaxChannel;
///
/// impl ConvertPixel<Rgb8, Mono8> for MaxChannel {
///     fn convert(&self, src: &Rgb8) -> Mono8 {
///         let mx = src.r.0.max(src.g.0).max(src.b.0);
///         Mono8::new(mx)
///     }
/// }
///
/// let red = Rgb8::new(200, 50, 100);
/// let mono = MaxChannel.convert(&red);
/// assert_eq!(mono, Mono8::new(200));
/// ```
pub trait ConvertPixel<Src, Dst> {
    /// Convert a single pixel from the source to the destination type.
    fn convert(&self, src: &Src) -> Dst;
}

// ───────────────────────────────────────────────────────────────────────────────
// Strategy types
// ───────────────────────────────────────────────────────────────────────────────

/// Full-range conversion strategy.
///
/// Maps the entire dynamic range of the source type onto the entire dynamic
/// range of the destination type.  This preserves *intensity*: maximum
/// brightness in the source stays maximum brightness in the destination.
///
/// For floating-point ↔ integer conversions the floating-point range is
/// assumed to be `[0.0, 1.0]`.
///
/// # Examples
/// - `Mono8(255)` → `Mono16(65535)` — white stays white
/// - `Mono16(32768)` → `Mono8(128)` — mid-gray stays mid-gray
/// - `Mono8(0)` → `Mono16(0)` — black stays black
/// - `Mono8(255)` → `f32(1.0)` — max maps to 1.0
///
/// ```
/// # use fovea::pixel::{Mono8, Mono16};
/// # use fovea::transform::{ConvertPixel, FullRange};
/// let lo = Mono8::new(0);
/// let hi = Mono8::new(255);
/// let lo16: Mono16 = FullRange.convert(&lo);
/// let hi16: Mono16 = FullRange.convert(&hi);
/// assert_eq!(lo16, Mono16::new(0));
/// assert_eq!(hi16, Mono16::new(65535));
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FullRange;

/// Narrowing conversion strategy.
///
/// Preserves the numeric value of each channel.  On widening conversions the
/// value is simply zero-extended.  On narrowing conversions values that exceed
/// the destination range are clamped to the maximum.
///
/// # Examples
/// - `Mono8(42)` → `Mono16(42)` — value preserved
/// - `Mono16(300)` → `Mono8(255)` — clamped to `u8::MAX`
/// - `Mono16(42)` → `Mono8(42)` — fits, so preserved
///
/// ```
/// # use fovea::pixel::{Mono8, Mono16};
/// # use fovea::transform::{ConvertPixel, Narrow};
/// let a: Mono8 = Narrow.convert(&Mono16::new(42));
/// assert_eq!(a, Mono8::new(42));
/// let b: Mono8 = Narrow.convert(&Mono16::new(300));
/// assert_eq!(b, Mono8::new(255));
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Narrow;

/// Luminance conversion strategy (BT.601).
///
/// Converts color pixels to grayscale using the ITU-R BT.601 luminance
/// formula:
///
/// ```text
/// Y = 0.299 · R + 0.587 · G + 0.114 · B
/// ```
///
/// For integer types an exact integer approximation is used:
///
/// ```text
/// Y = (77·R + 150·G + 29·B + 128) >> 8
/// ```
///
/// where `77 + 150 + 29 = 256`.
///
/// For RGBA / BGRA sources the alpha channel is ignored.
///
/// # Examples
/// ```
/// # use fovea::pixel::{Rgb8, Mono8};
/// # use fovea::transform::{ConvertPixel, Luminance};
/// let white = Rgb8::new(255, 255, 255);
/// assert_eq!(Luminance.convert(&white), Mono8::new(255));
///
/// let red = Rgb8::new(255, 0, 0);
/// assert_eq!(Luminance.convert(&red), Mono8::new(77));
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Luminance;

/// Broadcast conversion strategy.
///
/// Converts single-channel (mono) pixels to multi-channel (RGB / BGR / RGBA /
/// BGRA) pixels by copying the value to every color channel.  For RGBA / BGRA
/// targets the alpha channel is set to the maximum value (fully opaque).
///
/// # Examples
/// ```
/// # use fovea::pixel::{Mono8, Rgb8};
/// # use fovea::transform::{ConvertPixel, Broadcast};
/// let gray = Mono8::new(128);
/// let rgb: Rgb8 = Broadcast.convert(&gray);
/// assert_eq!(rgb, Rgb8::new(128, 128, 128));
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Broadcast;

/// Color-swap conversion strategy.
///
/// Converts between RGB-order and BGR-order pixel types by swapping the
/// red and blue channels while leaving green (and alpha, if present)
/// unchanged.
///
/// # Examples
/// ```
/// # use fovea::pixel::{Rgb8, Bgr8};
/// # use fovea::transform::{ConvertPixel, ColorSwap};
/// let rgb = Rgb8::new(200, 100, 50);
/// let bgr: Bgr8 = ColorSwap.convert(&rgb);
/// assert_eq!(bgr, Bgr8::new(50, 100, 200));
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ColorSwap;

/// Add-alpha conversion strategy.
///
/// Converts a 3-channel pixel (RGB / BGR) to the corresponding 4-channel
/// pixel (RGBA / BGRA) by copying every colour channel and setting the
/// alpha channel to the type's maximum value (fully opaque): `255` for u8,
/// `65535` for u16, `1.0` for f32.
///
/// # Examples
/// ```
/// # use fovea::pixel::{Rgb8, Rgba8};
/// # use fovea::transform::{ConvertPixel, AddAlpha};
/// let rgb = Rgb8::new(10, 20, 30);
/// let rgba: Rgba8 = AddAlpha.convert(&rgb);
/// assert_eq!(rgba, Rgba8::new(10, 20, 30, 255));
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AddAlpha;

/// sRGB gamma conversion strategy.
///
/// Converts between sRGB gamma-encoded pixels ([`Srgb8`], [`Srgba8`]) and
/// linear-light floating-point pixels ([`RgbF32`], [`RgbaF32`]) by applying
/// or removing the sRGB transfer function.
///
/// The sRGB transfer function is a piecewise curve defined by IEC 61966-2-1:
/// - **Decode** (sRGB → linear): `if v ≤ 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055)^2.4 }`
/// - **Encode** (linear → sRGB): `if v ≤ 0.0031308 { v × 12.92 } else { 1.055 × v^(1/2.4) − 0.055 }`
///
/// For [`Srgba8`] ↔ [`RgbaF32`], only the R, G, B channels are
/// gamma-converted; the alpha channel is transferred linearly (scaled
/// to/from `[0, 1]`), as required by the sRGB specification.
///
/// # Examples
///
/// Decode sRGB to linear:
/// ```
/// # use fovea::pixel::{Srgb8, RgbF32};
/// # use fovea::transform::{ConvertPixel, SrgbGamma};
/// let srgb = Srgb8::new(128, 64, 255);
/// let linear: RgbF32 = SrgbGamma.convert(&srgb);
/// // Mid-gray sRGB ≈ 0.216 linear, not 0.502
/// assert!((linear.r - 0.216).abs() < 0.001);
/// ```
///
/// Encode linear to sRGB:
/// ```
/// # use fovea::pixel::{Srgb8, RgbF32};
/// # use fovea::transform::{ConvertPixel, SrgbGamma};
/// let linear = RgbF32::new(0.5, 0.0, 1.0);
/// let srgb: Srgb8 = SrgbGamma.convert(&linear);
/// // 0.5 linear ≈ 188 sRGB, not 128
/// assert_eq!(srgb.r.0, 188);
/// ```
///
/// Compose with other strategies via [`.then()`](ConvertPixelExt::then):
/// ```
/// # use fovea::pixel::{Srgb8, Rgb16, RgbF32};
/// # use fovea::transform::{ConvertPixel, ConvertPixelExt, SrgbGamma, FullRange};
/// // sRGB 8-bit → linear f32 → linear 16-bit
/// let method = SrgbGamma.then::<RgbF32, _>(FullRange);
/// let result: Rgb16 = method.convert(&Srgb8::new(128, 128, 128));
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SrgbGamma;

// ───────────────────────────────────────────────────────────────────────────────
// Internal helpers – full-range arithmetic
// ───────────────────────────────────────────────────────────────────────────────

/// Widen u8 → u16 preserving full range.  255 × 257 = 65 535.
#[inline(always)]
fn fr_u8_to_u16(v: u8) -> u16 {
    (v as u16) * 257
}

/// Narrow u16 → u8 preserving full range with rounding.
/// `(v + 128) / 257` maps 0→0, 32 768→128, 65 535→255.
#[inline(always)]
fn fr_u16_to_u8(v: u16) -> u8 {
    ((v as u32 + 128) / 257) as u8
}

/// Widen u8 → u32 preserving full range.  255 × 0x01010101 = 0xFFFFFFFF.
#[inline(always)]
fn fr_u8_to_u32(v: u8) -> u32 {
    (v as u32) * 0x0101_0101
}

/// Narrow u32 → u8 preserving full range with rounding.
#[inline(always)]
fn fr_u32_to_u8(v: u32) -> u8 {
    ((v as u64 + 0x0080_8080) / 0x0101_0101) as u8
}

/// Widen u16 → u32 preserving full range.  65 535 × 65 537 = 0xFFFFFFFF.
#[inline(always)]
fn fr_u16_to_u32(v: u16) -> u32 {
    (v as u32) * 0x0001_0001
}

/// Narrow u32 → u16 preserving full range with rounding.
#[inline(always)]
fn fr_u32_to_u16(v: u32) -> u16 {
    ((v as u64 + 0x0000_8000) / 0x0001_0001) as u16
}

/// Widen u8 → u64 preserving full range.  255 × 0x0101010101010101 = u64::MAX.
#[inline(always)]
fn fr_u8_to_u64(v: u8) -> u64 {
    (v as u64) * 0x0101_0101_0101_0101
}

/// Narrow u64 → u8 preserving full range with rounding.
#[inline(always)]
fn fr_u64_to_u8(v: u64) -> u8 {
    ((v as u128 + 0x0080_8080_8080_8080) / 0x0101_0101_0101_0101) as u8
}

/// Widen u16 → u64 preserving full range.
/// 65 535 × 0x0001000100010001 = u64::MAX.
#[inline(always)]
fn fr_u16_to_u64(v: u16) -> u64 {
    (v as u64) * 0x0001_0001_0001_0001
}

/// Narrow u64 → u16 preserving full range with rounding.
#[inline(always)]
fn fr_u64_to_u16(v: u64) -> u16 {
    ((v as u128 + 0x0000_8000_0000_8000) / 0x0001_0001_0001_0001) as u16
}

/// Widen u32 → u64 preserving full range.
/// 0xFFFFFFFF × 0x0000000100000001 = u64::MAX.
#[inline(always)]
fn fr_u32_to_u64(v: u32) -> u64 {
    (v as u64) * 0x0000_0001_0000_0001
}

/// Narrow u64 → u32 preserving full range with rounding.
#[inline(always)]
fn fr_u64_to_u32(v: u64) -> u32 {
    ((v as u128 + 0x0000_0000_8000_0000) / 0x0000_0001_0000_0001) as u32
}

/// Convert u8 → f32 [0, 1].
#[inline(always)]
fn fr_u8_to_f32(v: u8) -> f32 {
    v as f32 / 255.0
}

/// Convert f32 [0, 1] → u8 with clamping and rounding.
#[inline(always)]
fn fr_f32_to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// Convert u16 → f32 [0, 1].
#[inline(always)]
fn fr_u16_to_f32(v: u16) -> f32 {
    v as f32 / 65535.0
}

/// Convert f32 [0, 1] → u16 with clamping and rounding.
#[inline(always)]
fn fr_f32_to_u16(v: f32) -> u16 {
    (v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16
}

/// Convert u32 → f32 [0, 1].
///
/// Note: `u32::MAX as f32` rounds to 2³², so values near `u32::MAX` may
/// produce `1.0` rather than a value just below it.  This is inherent to
/// f32's 24-bit mantissa and matches the same trade-off in `u16→f32`.
#[inline(always)]
fn fr_u32_to_f32(v: u32) -> f32 {
    v as f32 / u32::MAX as f32
}

/// Convert f32 [0, 1] → u32 with clamping and rounding.
/// Uses f64 intermediate to avoid overflow when scaling to `u32::MAX`.
#[inline(always)]
fn fr_f32_to_u32(v: f32) -> u32 {
    (v.clamp(0.0, 1.0) as f64 * u32::MAX as f64 + 0.5) as u32
}

/// Convert u64 → f32 [0, 1].
#[inline(always)]
fn fr_u64_to_f32(v: u64) -> f32 {
    (v as f64 / u64::MAX as f64) as f32
}

/// Convert f32 [0, 1] → u64 with clamping and rounding.
#[inline(always)]
fn fr_f32_to_u64(v: f32) -> u64 {
    (v.clamp(0.0, 1.0) as f64 * u64::MAX as f64 + 0.5) as u64
}

/// Convert u8 → f64 [0, 1].
#[inline(always)]
fn fr_u8_to_f64(v: u8) -> f64 {
    v as f64 / 255.0
}

/// Convert f64 [0, 1] → u8 with clamping and rounding.
#[inline(always)]
fn fr_f64_to_u8(v: f64) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// Convert u16 → f64 [0, 1].
#[inline(always)]
fn fr_u16_to_f64(v: u16) -> f64 {
    v as f64 / 65535.0
}

/// Convert f64 [0, 1] → u16 with clamping and rounding.
#[inline(always)]
fn fr_f64_to_u16(v: f64) -> u16 {
    (v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16
}

/// Convert u32 → f64 [0, 1].  f64 has 53-bit mantissa, so this is exact.
#[inline(always)]
fn fr_u32_to_f64(v: u32) -> f64 {
    v as f64 / u32::MAX as f64
}

/// Convert f64 [0, 1] → u32 with clamping and rounding.
#[inline(always)]
fn fr_f64_to_u32(v: f64) -> u32 {
    (v.clamp(0.0, 1.0) * u32::MAX as f64 + 0.5) as u32
}

/// Convert u64 → f64 [0, 1].
///
/// Note: `u64::MAX as f64` rounds up to 2⁶⁴, so values near `u64::MAX` may
/// produce exactly `1.0`.  Same trade-off as `u32→f32`.
#[inline(always)]
fn fr_u64_to_f64(v: u64) -> f64 {
    v as f64 / u64::MAX as f64
}

/// Convert f64 [0, 1] → u64 with clamping and rounding.
#[inline(always)]
fn fr_f64_to_u64(v: f64) -> u64 {
    // u64::MAX as f64 rounds up to 2^64; Rust's saturating float→int cast
    // clamps the result to u64::MAX, which is the correct behaviour here.
    (v.clamp(0.0, 1.0) * u64::MAX as f64 + 0.5) as u64
}

/// Convert f32 → f64 (widening, lossless).
#[inline(always)]
fn fr_f32_to_f64(v: f32) -> f64 {
    v as f64
}

/// Convert f64 → f32 (narrowing, may lose precision).
#[inline(always)]
fn fr_f64_to_f32(v: f64) -> f32 {
    v as f32
}

// ───────────────────────────────────────────────────────────────────────────────
// Internal helpers – sRGB transfer function (IEC 61966-2-1)
// ───────────────────────────────────────────────────────────────────────────────

/// Decode a single sRGB u8 channel to linear f32 in [0, 1].
#[inline]
fn srgb_decode(v: u8) -> f32 {
    let s = v as f32 / 255.0;
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

/// Encode a single linear f32 channel (in [0, 1]) to sRGB u8.
#[inline]
fn srgb_encode(v: f32) -> u8 {
    let c = v.clamp(0.0, 1.0);
    let s = if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0 + 0.5) as u8
}

/// Decode a single sRGB u16 channel to linear f32 in [0, 1].
///
/// Same transfer function as [`srgb_decode`] but the input is normalised
/// from `[0, 65535]` instead of `[0, 255]`.
#[inline]
fn srgb_decode_16(v: u16) -> f32 {
    let s = v as f32 / 65535.0;
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

/// Encode a single linear f32 channel (in [0, 1]) to sRGB u16.
///
/// Same transfer function as [`srgb_encode`] but the output is scaled
/// to `[0, 65535]` instead of `[0, 255]`.
#[inline]
fn srgb_encode_16(v: f32) -> u16 {
    let c = v.clamp(0.0, 1.0);
    let s = if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (s * 65535.0 + 0.5) as u16
}

// ───────────────────────────────────────────────────────────────────────────────
// Internal helpers – reading Mono8 / Mono16 values through the public API
//
// Mono8  is #[repr(transparent)] over Saturating<u8>.
// Mono16 is #[repr(transparent)] over Saturating<u16>.
// Their inner fields are private, so we go through PlainPixel::as_bytes().
// ───────────────────────────────────────────────────────────────────────────────

#[inline(always)]
fn mono8_val(src: &Mono8) -> u8 {
    src.as_bytes()[0]
}

#[inline(always)]
fn mono16_val(src: &Mono16) -> u16 {
    let b = src.as_bytes();
    u16::from_ne_bytes([b[0], b[1]])
}

#[inline(always)]
fn mono32_val(src: &Mono32) -> u32 {
    let b = src.as_bytes();
    u32::from_ne_bytes([b[0], b[1], b[2], b[3]])
}

#[inline(always)]
fn mono64_val(src: &Mono64) -> u64 {
    let b = src.as_bytes();
    u64::from_ne_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
}

// ───────────────────────────────────────────────────────────────────────────────
// Internal helpers – luminance
// ───────────────────────────────────────────────────────────────────────────────

/// BT.601 integer luminance for u8 channels.  77 + 150 + 29 = 256.
#[inline(always)]
fn lum_u8(r: u8, g: u8, b: u8) -> u8 {
    ((77u16 * r as u16 + 150u16 * g as u16 + 29u16 * b as u16 + 128) >> 8) as u8
}

/// BT.601 integer luminance for u16 channels.
#[inline(always)]
fn lum_u16(r: u16, g: u16, b: u16) -> u16 {
    ((77u32 * r as u32 + 150u32 * g as u32 + 29u32 * b as u32 + 128) >> 8) as u16
}

/// BT.601 integer luminance for u32 channels.
#[inline(always)]
fn lum_u32(r: u32, g: u32, b: u32) -> u32 {
    ((77u64 * r as u64 + 150u64 * g as u64 + 29u64 * b as u64 + 128) >> 8) as u32
}

/// BT.601 integer luminance for u64 channels.
#[inline(always)]
fn lum_u64(r: u64, g: u64, b: u64) -> u64 {
    ((77u128 * r as u128 + 150u128 * g as u128 + 29u128 * b as u128 + 128) >> 8) as u64
}

/// BT.601 floating-point luminance (f32).
#[inline(always)]
fn lum_f32(r: f32, g: f32, b: f32) -> f32 {
    0.299 * r + 0.587 * g + 0.114 * b
}

/// BT.601 floating-point luminance (f64).
#[inline(always)]
fn lum_f64(r: f64, g: f64, b: f64) -> f64 {
    0.299 * r + 0.587 * g + 0.114 * b
}

// ───────────────────────────────────────────────────────────────────────────────
// Internal helpers – clamping
// ───────────────────────────────────────────────────────────────────────────────

/// Identity widen u8 → u16 (value-preserving).
#[inline(always)]
fn clamp_u8_to_u16(v: u8) -> u16 {
    v as u16
}

/// Narrow u16 → u8 clamping to `u8::MAX`.
#[inline(always)]
fn clamp_u16_to_u8(v: u16) -> u8 {
    v.min(u8::MAX as u16) as u8
}

/// Identity widen u8 → u32 (value-preserving).
#[inline(always)]
fn clamp_u8_to_u32(v: u8) -> u32 {
    v as u32
}

/// Narrow u32 → u8 clamping to `u8::MAX`.
#[inline(always)]
fn clamp_u32_to_u8(v: u32) -> u8 {
    v.min(u8::MAX as u32) as u8
}

/// Identity widen u16 → u32 (value-preserving).
#[inline(always)]
fn clamp_u16_to_u32(v: u16) -> u32 {
    v as u32
}

/// Narrow u32 → u16 clamping to `u16::MAX`.
#[inline(always)]
fn clamp_u32_to_u16(v: u32) -> u16 {
    v.min(u16::MAX as u32) as u16
}

/// Identity widen u8 → u64 (value-preserving).
#[inline(always)]
fn clamp_u8_to_u64(v: u8) -> u64 {
    v as u64
}

/// Narrow u64 → u8 clamping to `u8::MAX`.
#[inline(always)]
fn clamp_u64_to_u8(v: u64) -> u8 {
    v.min(u8::MAX as u64) as u8
}

/// Identity widen u16 → u64 (value-preserving).
#[inline(always)]
fn clamp_u16_to_u64(v: u16) -> u64 {
    v as u64
}

/// Narrow u64 → u16 clamping to `u16::MAX`.
#[inline(always)]
fn clamp_u64_to_u16(v: u64) -> u16 {
    v.min(u16::MAX as u64) as u16
}

/// Identity widen u32 → u64 (value-preserving).
#[inline(always)]
fn clamp_u32_to_u64(v: u32) -> u64 {
    v as u64
}

/// Narrow u64 → u32 clamping to `u32::MAX`.
#[inline(always)]
fn clamp_u64_to_u32(v: u64) -> u32 {
    v.min(u32::MAX as u64) as u32
}

// ═══════════════════════════════════════════════════════════════════════════════
// Macros — eliminate the cross-product boilerplate
//
// Each pixel-family (Rgb, Rgba, Bgr, Bgra) needs the same set of depth
// conversions (u8↔u16, u8↔f32, u16↔f32).  The logic is identical across
// families — only the field names differ.  These macros generate all
// required `ConvertPixel` impls from a single declaration per family.
// ═══════════════════════════════════════════════════════════════════════════════

/// Generate a single `ConvertPixel` impl that applies `$conv` to each channel.
///
/// The `|$s|` binding names the source reference, and `[$($ch),+]` lists
/// the expressions that extract each channel's scalar value from it.
macro_rules! impl_convert {
    ($Strategy:ty: $Src:ty => $Dst:ty, |$s:ident| [$($ch:expr),+], $conv:ident) => {
        impl ConvertPixel<$Src, $Dst> for $Strategy {
            #[inline]
            fn convert(&self, $s: &$Src) -> $Dst {
                <$Dst>::new($($conv($ch)),+)
            }
        }
    };
}

/// Generate all depth-conversion impls for a pixel family whose integer
/// variants use `Saturating<T>` fields and whose float variant uses plain
/// `f32` fields.
///
/// One invocation produces **8 impls**: 6 for `FullRange` and 2 for `Narrow`.
///
/// # Arguments
/// - `$U8`  — the 8-bit pixel type  (e.g. `Rgb8`)
/// - `$U16` — the 16-bit pixel type (e.g. `Rgb16`)
/// - `$F32` — the f32 pixel type    (e.g. `RgbF32`)
/// - `[$($f),+]` — field names in constructor order (e.g. `[r, g, b]`)
macro_rules! impl_family_conversions {
    ($U8:ty, $U16:ty, $F32:ty, [$($f:ident),+]) => {
        // FullRange: u8 ↔ u16
        impl_convert!(FullRange: $U8  => $U16, |s| [$(s.$f.0),+], fr_u8_to_u16);
        impl_convert!(FullRange: $U16 => $U8,  |s| [$(s.$f.0),+], fr_u16_to_u8);

        // FullRange: u8 ↔ f32
        impl_convert!(FullRange: $U8  => $F32, |s| [$(s.$f.0),+], fr_u8_to_f32);
        impl_convert!(FullRange: $F32 => $U8,  |s| [$(s.$f),+],   fr_f32_to_u8);

        // FullRange: u16 ↔ f32
        impl_convert!(FullRange: $U16 => $F32, |s| [$(s.$f.0),+], fr_u16_to_f32);
        impl_convert!(FullRange: $F32 => $U16, |s| [$(s.$f),+],   fr_f32_to_u16);

        // Narrow: u8 ↔ u16
        impl_convert!(Narrow: $U8  => $U16, |s| [$(s.$f.0),+], clamp_u8_to_u16);
        impl_convert!(Narrow: $U16 => $U8,  |s| [$(s.$f.0),+], clamp_u16_to_u8);
    };
}

/// Generate depth-conversion impls involving 32-bit and 64-bit integer
/// variants plus the f64 float variant of a pixel family.
///
/// One invocation produces **38 impls** covering all remaining pairs among
/// { u8, u16, u32, u64, f32, f64 } that are not already covered by
/// [`impl_family_conversions!`].
///
/// # Arguments
/// - `$U8`  — the 8-bit pixel type   (e.g. `Rgb8`)
/// - `$U16` — the 16-bit pixel type  (e.g. `Rgb16`)
/// - `$U32` — the 32-bit pixel type  (e.g. `Rgb32`)
/// - `$U64` — the 64-bit pixel type  (e.g. `Rgb64`)
/// - `$F32` — the f32 pixel type     (e.g. `RgbF32`)
/// - `$F64` — the f64 pixel type     (e.g. `RgbF64`)
/// - `[$($f),+]` — field names in constructor order (e.g. `[r, g, b]`)
macro_rules! impl_extended_family_conversions {
    ($U8:ty, $U16:ty, $U32:ty, $U64:ty, $F32:ty, $F64:ty, [$($f:ident),+]) => {
        // ── FullRange: integer pairs involving u32 ──────────────────────────
        impl_convert!(FullRange: $U8  => $U32, |s| [$(s.$f.0),+], fr_u8_to_u32);
        impl_convert!(FullRange: $U32 => $U8,  |s| [$(s.$f.0),+], fr_u32_to_u8);
        impl_convert!(FullRange: $U16 => $U32, |s| [$(s.$f.0),+], fr_u16_to_u32);
        impl_convert!(FullRange: $U32 => $U16, |s| [$(s.$f.0),+], fr_u32_to_u16);

        // ── FullRange: integer pairs involving u64 ──────────────────────────
        impl_convert!(FullRange: $U8  => $U64, |s| [$(s.$f.0),+], fr_u8_to_u64);
        impl_convert!(FullRange: $U64 => $U8,  |s| [$(s.$f.0),+], fr_u64_to_u8);
        impl_convert!(FullRange: $U16 => $U64, |s| [$(s.$f.0),+], fr_u16_to_u64);
        impl_convert!(FullRange: $U64 => $U16, |s| [$(s.$f.0),+], fr_u64_to_u16);
        impl_convert!(FullRange: $U32 => $U64, |s| [$(s.$f.0),+], fr_u32_to_u64);
        impl_convert!(FullRange: $U64 => $U32, |s| [$(s.$f.0),+], fr_u64_to_u32);

        // ── FullRange: u32/u64 ↔ f32 ───────────────────────────────────────
        impl_convert!(FullRange: $U32 => $F32, |s| [$(s.$f.0),+], fr_u32_to_f32);
        impl_convert!(FullRange: $F32 => $U32, |s| [$(s.$f),+],   fr_f32_to_u32);
        impl_convert!(FullRange: $U64 => $F32, |s| [$(s.$f.0),+], fr_u64_to_f32);
        impl_convert!(FullRange: $F32 => $U64, |s| [$(s.$f),+],   fr_f32_to_u64);

        // ── FullRange: all integer ↔ f64 ───────────────────────────────────
        impl_convert!(FullRange: $U8  => $F64, |s| [$(s.$f.0),+], fr_u8_to_f64);
        impl_convert!(FullRange: $F64 => $U8,  |s| [$(s.$f),+],   fr_f64_to_u8);
        impl_convert!(FullRange: $U16 => $F64, |s| [$(s.$f.0),+], fr_u16_to_f64);
        impl_convert!(FullRange: $F64 => $U16, |s| [$(s.$f),+],   fr_f64_to_u16);
        impl_convert!(FullRange: $U32 => $F64, |s| [$(s.$f.0),+], fr_u32_to_f64);
        impl_convert!(FullRange: $F64 => $U32, |s| [$(s.$f),+],   fr_f64_to_u32);
        impl_convert!(FullRange: $U64 => $F64, |s| [$(s.$f.0),+], fr_u64_to_f64);
        impl_convert!(FullRange: $F64 => $U64, |s| [$(s.$f),+],   fr_f64_to_u64);

        // ── FullRange: f32 ↔ f64 ───────────────────────────────────────────
        impl_convert!(FullRange: $F32 => $F64, |s| [$(s.$f),+], fr_f32_to_f64);
        impl_convert!(FullRange: $F64 => $F32, |s| [$(s.$f),+], fr_f64_to_f32);

        // ── Narrow: integer pairs involving u32 ─────────────────────────────
        impl_convert!(Narrow: $U8  => $U32, |s| [$(s.$f.0),+], clamp_u8_to_u32);
        impl_convert!(Narrow: $U32 => $U8,  |s| [$(s.$f.0),+], clamp_u32_to_u8);
        impl_convert!(Narrow: $U16 => $U32, |s| [$(s.$f.0),+], clamp_u16_to_u32);
        impl_convert!(Narrow: $U32 => $U16, |s| [$(s.$f.0),+], clamp_u32_to_u16);

        // ── Narrow: integer pairs involving u64 ─────────────────────────────
        impl_convert!(Narrow: $U8  => $U64, |s| [$(s.$f.0),+], clamp_u8_to_u64);
        impl_convert!(Narrow: $U64 => $U8,  |s| [$(s.$f.0),+], clamp_u64_to_u8);
        impl_convert!(Narrow: $U16 => $U64, |s| [$(s.$f.0),+], clamp_u16_to_u64);
        impl_convert!(Narrow: $U64 => $U16, |s| [$(s.$f.0),+], clamp_u64_to_u16);
        impl_convert!(Narrow: $U32 => $U64, |s| [$(s.$f.0),+], clamp_u32_to_u64);
        impl_convert!(Narrow: $U64 => $U32, |s| [$(s.$f.0),+], clamp_u64_to_u32);
    };
}

/// Generate a `Luminance` impl for a source with `Saturating<T>` fields.
macro_rules! impl_luminance_sat {
    ($Src:ty => $Dst:ty, $lum:ident, $r:ident, $g:ident, $b:ident) => {
        impl ConvertPixel<$Src, $Dst> for Luminance {
            #[inline]
            fn convert(&self, src: &$Src) -> $Dst {
                <$Dst>::new($lum(src.$r.0, src.$g.0, src.$b.0))
            }
        }
    };
}

// The `impl_luminance_f32!` / `impl_luminance_f64!` macros — which
// emitted `ConvertPixel<_, f32>` / `ConvertPixel<_, f64>` impls for
// `Luminance` — are intentionally not provided. Float pixel roles are
// served exclusively by the named `MonoF32` / `MonoF64` accumulator
// types; the corresponding `Luminance → MonoF32/MonoF64` impls are
// defined further below via `impl_convert_expr!`.

/// Generate a `Broadcast` impl from a mono type to a 3-channel color type.
///
/// `$val` is an expression (function or closure) that extracts the scalar
/// value from the source pixel.
macro_rules! impl_broadcast {
    ($Mono:ty => $Color:ty, $val:expr) => {
        impl ConvertPixel<$Mono, $Color> for Broadcast {
            #[inline]
            fn convert(&self, src: &$Mono) -> $Color {
                let v = ($val)(src);
                <$Color>::new(v, v, v)
            }
        }
    };
    ($Mono:ty => $Color:ty, $val:expr, alpha = $alpha:expr) => {
        impl ConvertPixel<$Mono, $Color> for Broadcast {
            #[inline]
            fn convert(&self, src: &$Mono) -> $Color {
                let v = ($val)(src);
                <$Color>::new(v, v, v, $alpha)
            }
        }
    };
}

/// Generate a bidirectional `ColorSwap` impl between an RGB-order and a
/// BGR-order pixel type.
///
/// `[$($af),+]` lists the field names in A's constructor order.
/// `[$($bf),+]` lists the field names in B's constructor order.
/// When creating B from A we read A's fields in B's constructor order (and
/// vice versa), which swaps R and B while keeping G (and A) in place.
macro_rules! impl_color_swap {
    (sat: $A:ty [$($af:ident),+] <=> $B:ty [$($bf:ident),+]) => {
        impl ConvertPixel<$A, $B> for ColorSwap {
            #[inline]
            fn convert(&self, src: &$A) -> $B { <$B>::new($(src.$bf.0),+) }
        }
        impl ConvertPixel<$B, $A> for ColorSwap {
            #[inline]
            fn convert(&self, src: &$B) -> $A { <$A>::new($(src.$af.0),+) }
        }
    };
    (f32: $A:ty [$($af:ident),+] <=> $B:ty [$($bf:ident),+]) => {
        impl ConvertPixel<$A, $B> for ColorSwap {
            #[inline]
            fn convert(&self, src: &$A) -> $B { <$B>::new($(src.$bf),+) }
        }
        impl ConvertPixel<$B, $A> for ColorSwap {
            #[inline]
            fn convert(&self, src: &$B) -> $A { <$A>::new($(src.$af),+) }
        }
    };
}

/// Generate an `AddAlpha` impl from a 3-channel type to its 4-channel
/// counterpart, setting alpha to `$alpha`.
///
/// `[$($f),+]` lists the field names in the destination's constructor order
/// (minus alpha, which is appended automatically).
macro_rules! impl_add_alpha {
    (sat: $Src:ty => $Dst:ty, [$($f:ident),+], $alpha:expr) => {
        impl ConvertPixel<$Src, $Dst> for AddAlpha {
            #[inline]
            fn convert(&self, src: &$Src) -> $Dst {
                <$Dst>::new($(src.$f.0),+, $alpha)
            }
        }
    };
    (f32: $Src:ty => $Dst:ty, [$($f:ident),+], $alpha:expr) => {
        impl ConvertPixel<$Src, $Dst> for AddAlpha {
            #[inline]
            fn convert(&self, src: &$Src) -> $Dst {
                <$Dst>::new($(src.$f),+, $alpha)
            }
        }
    };
}

/// Single-direction conversion where the body is an arbitrary expression.
///
/// Used for mono/single-channel types that don't fit the multi-channel
/// [`impl_convert!`] pattern (which assumes named fields and a uniform
/// per-channel conversion function).
macro_rules! impl_convert_expr {
    ($Strategy:ty: $Src:ty => $Dst:ty, |$s:ident| $body:expr) => {
        impl ConvertPixel<$Src, $Dst> for $Strategy {
            #[inline]
            fn convert(&self, $s: &$Src) -> $Dst {
                $body
            }
        }
    };
}

/// Broadcast MonoA → 4-channel type, preserving alpha.
///
/// `sat:` variant for types with `Saturating<T>` fields (access via `.0`).
/// `float:` variant for types with plain float fields.
macro_rules! impl_broadcast_monoa {
    (sat: $Src:ty => $Dst:ty) => {
        impl ConvertPixel<$Src, $Dst> for Broadcast {
            #[inline]
            fn convert(&self, src: &$Src) -> $Dst {
                <$Dst>::new(src.v.0, src.v.0, src.v.0, src.a.0)
            }
        }
    };
    (float: $Src:ty => $Dst:ty) => {
        impl ConvertPixel<$Src, $Dst> for Broadcast {
            #[inline]
            fn convert(&self, src: &$Src) -> $Dst {
                <$Dst>::new(src.v, src.v, src.v, src.a)
            }
        }
    };
}

/// Luminance from 4-channel type to MonoA, preserving alpha.
///
/// `sat:` variant for saturating integer fields.
/// `f32:` / `f64:` variants for float fields.
macro_rules! impl_luminance_monoa {
    (sat: $Src:ty => $Dst:ty, $lum:ident, $r:ident, $g:ident, $b:ident, $a:ident) => {
        impl ConvertPixel<$Src, $Dst> for Luminance {
            #[inline]
            fn convert(&self, src: &$Src) -> $Dst {
                <$Dst>::new($lum(src.$r.0, src.$g.0, src.$b.0), src.$a.0)
            }
        }
    };
    (f32: $Src:ty => $Dst:ty, $r:ident, $g:ident, $b:ident, $a:ident) => {
        impl ConvertPixel<$Src, $Dst> for Luminance {
            #[inline]
            fn convert(&self, src: &$Src) -> $Dst {
                <$Dst>::new(lum_f32(src.$r, src.$g, src.$b), src.$a)
            }
        }
    };
    (f64: $Src:ty => $Dst:ty, $r:ident, $g:ident, $b:ident, $a:ident) => {
        impl ConvertPixel<$Src, $Dst> for Luminance {
            #[inline]
            fn convert(&self, src: &$Src) -> $Dst {
                <$Dst>::new(lum_f64(src.$r, src.$g, src.$b), src.$a)
            }
        }
    };
}

// ═══════════════════════════════════════════════════════════════════════════════
// FullRange + Narrow — Mono family (macro-generated, tuple-struct access)
// ═══════════════════════════════════════════════════════════════════════════════

// ── Mono depth conversions ──────────────────────────────────────────────────
impl_convert_expr!(FullRange: Mono8  => Mono16, |src| Mono16::new(fr_u8_to_u16(mono8_val(src))));
impl_convert_expr!(FullRange: Mono16 => Mono8,  |src| Mono8::new(fr_u16_to_u8(mono16_val(src))));

// ── Mono ↔ f32: intentionally omitted. Use the `MonoF32` siblings
//    defined below.

// ── Narrow: Mono ─────────────────────────────────────────────────────────────
impl_convert_expr!(Narrow: Mono8  => Mono16, |src| Mono16::new(clamp_u8_to_u16(mono8_val(src))));
impl_convert_expr!(Narrow: Mono16 => Mono8,  |src| Mono8::new(clamp_u16_to_u8(mono16_val(src))));

// ═══════════════════════════════════════════════════════════════════════════════
// FullRange + Narrow — Mono<BITS> (const-generic, special scaling)
// ═══════════════════════════════════════════════════════════════════════════════

impl<const BITS: usize> ConvertPixel<Mono<BITS>, Mono8> for FullRange {
    #[inline]
    fn convert(&self, src: &Mono<BITS>) -> Mono8 {
        let max_src = (1u32 << BITS) - 1;
        let v = src.value() as u32;
        Mono8::new(((v * 255 + max_src / 2) / max_src) as u8)
    }
}

impl<const BITS: usize> ConvertPixel<Mono<BITS>, Mono16> for FullRange {
    #[inline]
    fn convert(&self, src: &Mono<BITS>) -> Mono16 {
        let max_src = (1u32 << BITS) - 1;
        let v = src.value() as u32;
        Mono16::new(((v * 65535 + max_src / 2) / max_src) as u16)
    }
}

impl<const BITS: usize> ConvertPixel<Mono<BITS>, Mono32> for FullRange {
    #[inline]
    fn convert(&self, src: &Mono<BITS>) -> Mono32 {
        let max_src = (1u64 << BITS) - 1;
        let v = src.value() as u64;
        Mono32::new(((v * u32::MAX as u64 + max_src / 2) / max_src) as u32)
    }
}

// `ConvertPixel<Mono<BITS>, f32>` is intentionally not provided. Use
// `ConvertPixel<Mono<BITS>, MonoF32>` defined below.

impl<const BITS: usize> ConvertPixel<Mono<BITS>, Mono64> for FullRange {
    #[inline]
    fn convert(&self, src: &Mono<BITS>) -> Mono64 {
        let max_src = (1u128 << BITS) - 1;
        let v = src.value() as u128;
        Mono64::new(((v * u64::MAX as u128 + max_src / 2) / max_src) as u64)
    }
}

// `ConvertPixel<Mono<BITS>, f64>` is intentionally not provided. Use
// `ConvertPixel<Mono<BITS>, MonoF64>` defined below.

impl<const BITS: usize> ConvertPixel<Mono<BITS>, Mono8> for Narrow {
    #[inline]
    fn convert(&self, src: &Mono<BITS>) -> Mono8 {
        Mono8::new(clamp_u16_to_u8(src.value()))
    }
}

impl<const BITS: usize> ConvertPixel<Mono<BITS>, Mono16> for Narrow {
    #[inline]
    fn convert(&self, src: &Mono<BITS>) -> Mono16 {
        Mono16::new(src.value())
    }
}

impl<const BITS: usize> ConvertPixel<Mono<BITS>, Mono32> for Narrow {
    #[inline]
    fn convert(&self, src: &Mono<BITS>) -> Mono32 {
        Mono32::new(src.value() as u32)
    }
}

impl<const BITS: usize> ConvertPixel<Mono<BITS>, Mono64> for Narrow {
    #[inline]
    fn convert(&self, src: &Mono<BITS>) -> Mono64 {
        Mono64::new(src.value() as u64)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// FullRange + Narrow — Mono32 / Mono64 / f64 depth conversions (macro-generated)
// ═══════════════════════════════════════════════════════════════════════════════

// ── Mono8/16 ↔ Mono32 ──────────────────────────────────────────────────────
impl_convert_expr!(FullRange: Mono8  => Mono32, |src| Mono32::new(fr_u8_to_u32(mono8_val(src))));
impl_convert_expr!(FullRange: Mono32 => Mono8,  |src| Mono8::new(fr_u32_to_u8(mono32_val(src))));
impl_convert_expr!(FullRange: Mono16 => Mono32, |src| Mono32::new(fr_u16_to_u32(mono16_val(src))));
impl_convert_expr!(FullRange: Mono32 => Mono16, |src| Mono16::new(fr_u32_to_u16(mono32_val(src))));

// ── Mono32 ↔ f32/f64: intentionally omitted. Use `MonoF32` /
//    `MonoF64` siblings below.

// ── Mono8/16/32 ↔ Mono64 ───────────────────────────────────────────────────
impl_convert_expr!(FullRange: Mono8  => Mono64, |src| Mono64::new(fr_u8_to_u64(mono8_val(src))));
impl_convert_expr!(FullRange: Mono64 => Mono8,  |src| Mono8::new(fr_u64_to_u8(mono64_val(src))));
impl_convert_expr!(FullRange: Mono16 => Mono64, |src| Mono64::new(fr_u16_to_u64(mono16_val(src))));
impl_convert_expr!(FullRange: Mono64 => Mono16, |src| Mono16::new(fr_u64_to_u16(mono64_val(src))));
impl_convert_expr!(FullRange: Mono32 => Mono64, |src| Mono64::new(fr_u32_to_u64(mono32_val(src))));
impl_convert_expr!(FullRange: Mono64 => Mono32, |src| Mono32::new(fr_u64_to_u32(mono64_val(src))));

// ── Mono64 ↔ f32/f64: intentionally omitted. Use `MonoF32` /
//    `MonoF64` siblings below.

// ── Mono8/16 ↔ f64: intentionally omitted. Use `MonoF64`
//    siblings below.

// ── f32 ↔ f64: intentionally omitted. Use the `MonoF32`
//    ↔ `MonoF64` impls below.

// ── MonoF32 ↔ integer Mono types ────────────────────────────────────────────
impl_convert_expr!(FullRange: Mono8   => MonoF32, |src| MonoF32::new(fr_u8_to_f32(mono8_val(src))));
impl_convert_expr!(FullRange: MonoF32 => Mono8,   |src| Mono8::new(fr_f32_to_u8(src.0)));
impl_convert_expr!(FullRange: Mono16  => MonoF32, |src| MonoF32::new(fr_u16_to_f32(mono16_val(src))));
impl_convert_expr!(FullRange: MonoF32 => Mono16,  |src| Mono16::new(fr_f32_to_u16(src.0)));
impl_convert_expr!(FullRange: Mono32  => MonoF32, |src| MonoF32::new(fr_u32_to_f32(mono32_val(src))));
impl_convert_expr!(FullRange: MonoF32 => Mono32,  |src| Mono32::new(fr_f32_to_u32(src.0)));
impl_convert_expr!(FullRange: Mono64  => MonoF32, |src| MonoF32::new(fr_u64_to_f32(mono64_val(src))));
impl_convert_expr!(FullRange: MonoF32 => Mono64,  |src| Mono64::new(fr_f32_to_u64(src.0)));

impl<const BITS: usize> ConvertPixel<Mono<BITS>, MonoF32> for FullRange {
    #[inline]
    fn convert(&self, src: &Mono<BITS>) -> MonoF32 {
        let max_src = ((1u32 << BITS) - 1) as f32;
        MonoF32::new(src.value() as f32 / max_src)
    }
}

// ── MonoF64 ↔ integer Mono types ────────────────────────────────────────────
impl_convert_expr!(FullRange: Mono8   => MonoF64, |src| MonoF64::new(fr_u8_to_f64(mono8_val(src))));
impl_convert_expr!(FullRange: MonoF64 => Mono8,   |src| Mono8::new(fr_f64_to_u8(src.0)));
impl_convert_expr!(FullRange: Mono16  => MonoF64, |src| MonoF64::new(fr_u16_to_f64(mono16_val(src))));
impl_convert_expr!(FullRange: MonoF64 => Mono16,  |src| Mono16::new(fr_f64_to_u16(src.0)));
impl_convert_expr!(FullRange: Mono32  => MonoF64, |src| MonoF64::new(fr_u32_to_f64(mono32_val(src))));
impl_convert_expr!(FullRange: MonoF64 => Mono32,  |src| Mono32::new(fr_f64_to_u32(src.0)));
impl_convert_expr!(FullRange: Mono64  => MonoF64, |src| MonoF64::new(fr_u64_to_f64(mono64_val(src))));
impl_convert_expr!(FullRange: MonoF64 => Mono64,  |src| Mono64::new(fr_f64_to_u64(src.0)));

impl<const BITS: usize> ConvertPixel<Mono<BITS>, MonoF64> for FullRange {
    #[inline]
    fn convert(&self, src: &Mono<BITS>) -> MonoF64 {
        let max_src = ((1u32 << BITS) - 1) as f64;
        MonoF64::new(src.value() as f64 / max_src)
    }
}

// ── MonoF32 ↔ MonoF64 ──────────────────────────────────────────────────────
impl_convert_expr!(FullRange: MonoF32 => MonoF64, |src| MonoF64::new(src.0 as f64));
impl_convert_expr!(FullRange: MonoF64 => MonoF32, |src| MonoF32::new(src.0 as f32));

// ── MonoF32/MonoF64 ↔ bare f32/f64 identity conversions: intentionally
//    omitted. Bare floats are not pixel types; callers that hold a raw
//    `f32`/`f64` scalar can wrap via `MonoF32::new` / `MonoF64::new`
//    directly, or use `From`/`Into`.

// ── Narrow: Mono32 / Mono64 ─────────────────────────────────────────────────
impl_convert_expr!(Narrow: Mono8  => Mono32, |src| Mono32::new(clamp_u8_to_u32(mono8_val(src))));
impl_convert_expr!(Narrow: Mono32 => Mono8,  |src| Mono8::new(clamp_u32_to_u8(mono32_val(src))));
impl_convert_expr!(Narrow: Mono16 => Mono32, |src| Mono32::new(clamp_u16_to_u32(mono16_val(src))));
impl_convert_expr!(Narrow: Mono32 => Mono16, |src| Mono16::new(clamp_u32_to_u16(mono32_val(src))));
impl_convert_expr!(Narrow: Mono8  => Mono64, |src| Mono64::new(clamp_u8_to_u64(mono8_val(src))));
impl_convert_expr!(Narrow: Mono64 => Mono8,  |src| Mono8::new(clamp_u64_to_u8(mono64_val(src))));
impl_convert_expr!(Narrow: Mono16 => Mono64, |src| Mono64::new(clamp_u16_to_u64(mono16_val(src))));
impl_convert_expr!(Narrow: Mono64 => Mono16, |src| Mono16::new(clamp_u64_to_u16(mono64_val(src))));
impl_convert_expr!(Narrow: Mono32 => Mono64, |src| Mono64::new(clamp_u32_to_u64(mono32_val(src))));
impl_convert_expr!(Narrow: Mono64 => Mono32, |src| Mono32::new(clamp_u64_to_u32(mono64_val(src))));

// ═══════════════════════════════════════════════════════════════════════════════
// FullRange + Narrow — Multi-channel families (macro-generated)
//
// Each invocation of `impl_family_conversions!` produces 8 impls:
//   FullRange:  u8↔u16, u8↔f32, u16↔f32   (6 impls)
//   Narrow:      u8↔u16                      (2 impls)
//
// Each invocation of `impl_extended_family_conversions!` produces 38 impls
// covering all remaining pairs among { u8, u16, u32, u64, f32, f64 }.
// ═══════════════════════════════════════════════════════════════════════════════

impl_family_conversions!(Rgb8, Rgb16, RgbF32, [r, g, b]);
impl_family_conversions!(Rgba8, Rgba16, RgbaF32, [r, g, b, a]);
impl_family_conversions!(Bgr8, Bgr16, BgrF32, [b, g, r]);
impl_family_conversions!(Bgra8, Bgra16, BgraF32, [b, g, r, a]);

impl_extended_family_conversions!(Rgb8, Rgb16, Rgb32, Rgb64, RgbF32, RgbF64, [r, g, b]);
impl_extended_family_conversions!(
    Rgba8,
    Rgba16,
    Rgba32,
    Rgba64,
    RgbaF32,
    RgbaF64,
    [r, g, b, a]
);
impl_extended_family_conversions!(Bgr8, Bgr16, Bgr32, Bgr64, BgrF32, BgrF64, [b, g, r]);
impl_extended_family_conversions!(
    Bgra8,
    Bgra16,
    Bgra32,
    Bgra64,
    BgraF32,
    BgraF64,
    [b, g, r, a]
);

// ═══════════════════════════════════════════════════════════════════════════════
// Luminance implementations (BT.601)
//
// For RGBA / BGRA sources the alpha channel is simply ignored.
// ═══════════════════════════════════════════════════════════════════════════════

// ── RGB → Mono ──────────────────────────────────────────────────────────────
impl_luminance_sat!(Rgb8  => Mono8,  lum_u8,  r, g, b);
impl_luminance_sat!(Rgb16 => Mono16, lum_u16, r, g, b);
impl_luminance_sat!(Rgb32 => Mono32, lum_u32, r, g, b);
impl_luminance_sat!(Rgb64 => Mono64, lum_u64, r, g, b);

// ── RGBA → Mono (alpha ignored) ────────────────────────────────────────────
impl_luminance_sat!(Rgba8  => Mono8,  lum_u8,  r, g, b);
impl_luminance_sat!(Rgba16 => Mono16, lum_u16, r, g, b);
impl_luminance_sat!(Rgba32 => Mono32, lum_u32, r, g, b);
impl_luminance_sat!(Rgba64 => Mono64, lum_u64, r, g, b);

// ── BGR → Mono ──────────────────────────────────────────────────────────────
impl_luminance_sat!(Bgr8  => Mono8,  lum_u8,  r, g, b);
impl_luminance_sat!(Bgr16 => Mono16, lum_u16, r, g, b);
impl_luminance_sat!(Bgr32 => Mono32, lum_u32, r, g, b);
impl_luminance_sat!(Bgr64 => Mono64, lum_u64, r, g, b);

// ── BGRA → Mono (alpha ignored) ────────────────────────────────────────────
impl_luminance_sat!(Bgra8  => Mono8,  lum_u8,  r, g, b);
impl_luminance_sat!(Bgra16 => Mono16, lum_u16, r, g, b);
impl_luminance_sat!(Bgra32 => Mono32, lum_u32, r, g, b);
impl_luminance_sat!(Bgra64 => Mono64, lum_u64, r, g, b);

// ── RGB/RGBA/BGR/BGRA → MonoF32 (BT.601 luminance) ─────────────────────────
impl_convert_expr!(Luminance: RgbF32  => MonoF32, |src| MonoF32::new(lum_f32(src.r, src.g, src.b)));
impl_convert_expr!(Luminance: RgbaF32 => MonoF32, |src| MonoF32::new(lum_f32(src.r, src.g, src.b)));
impl_convert_expr!(Luminance: BgrF32  => MonoF32, |src| MonoF32::new(lum_f32(src.r, src.g, src.b)));
impl_convert_expr!(Luminance: BgraF32 => MonoF32, |src| MonoF32::new(lum_f32(src.r, src.g, src.b)));

// ── RGB/RGBA/BGR/BGRA → MonoF64 (BT.601 luminance) ─────────────────────────
impl_convert_expr!(Luminance: RgbF64  => MonoF64, |src| MonoF64::new(lum_f64(src.r, src.g, src.b)));
impl_convert_expr!(Luminance: RgbaF64 => MonoF64, |src| MonoF64::new(lum_f64(src.r, src.g, src.b)));
impl_convert_expr!(Luminance: BgrF64  => MonoF64, |src| MonoF64::new(lum_f64(src.r, src.g, src.b)));
impl_convert_expr!(Luminance: BgraF64 => MonoF64, |src| MonoF64::new(lum_f64(src.r, src.g, src.b)));

// ═══════════════════════════════════════════════════════════════════════════════
// Broadcast implementations
//
// For RGBA / BGRA targets the alpha channel is set to the type's maximum
// value (fully opaque): 255 for u8, 65 535 for u16, 1.0 for f32.
// ═══════════════════════════════════════════════════════════════════════════════

// ── Mono → 3-channel ────────────────────────────────────────────────────────
impl_broadcast!(Mono8  => Rgb8,   mono8_val);
impl_broadcast!(Mono8  => Bgr8,   mono8_val);
impl_broadcast!(Mono16 => Rgb16,  mono16_val);
impl_broadcast!(Mono16 => Bgr16,  mono16_val);
impl_broadcast!(Mono32 => Rgb32,  mono32_val);
impl_broadcast!(Mono32 => Bgr32,  mono32_val);
impl_broadcast!(Mono64 => Rgb64,  mono64_val);
impl_broadcast!(Mono64 => Bgr64,  mono64_val);
// `Broadcast` from bare f32/f64 is intentionally not provided. Use
// `MonoF32` / `MonoF64` sources below.
impl_broadcast!(MonoF32 => RgbF32, |s: &MonoF32| s.0);
impl_broadcast!(MonoF32 => BgrF32, |s: &MonoF32| s.0);
impl_broadcast!(MonoF64 => RgbF64, |s: &MonoF64| s.0);
impl_broadcast!(MonoF64 => BgrF64, |s: &MonoF64| s.0);

// ── Mono → 4-channel (alpha = max) ─────────────────────────────────────────
impl_broadcast!(Mono8  => Rgba8,   mono8_val,  alpha = u8::MAX);
impl_broadcast!(Mono8  => Bgra8,   mono8_val,  alpha = u8::MAX);
impl_broadcast!(Mono16 => Rgba16,  mono16_val, alpha = u16::MAX);
impl_broadcast!(Mono16 => Bgra16,  mono16_val, alpha = u16::MAX);
impl_broadcast!(Mono32 => Rgba32,  mono32_val, alpha = u32::MAX);
impl_broadcast!(Mono32 => Bgra32,  mono32_val, alpha = u32::MAX);
impl_broadcast!(Mono64 => Rgba64,  mono64_val, alpha = u64::MAX);
impl_broadcast!(Mono64 => Bgra64,  mono64_val, alpha = u64::MAX);
// `Broadcast` from bare f32/f64 to *A is intentionally not provided.
// Use `MonoF32` / `MonoF64` sources below.
impl_broadcast!(MonoF32 => RgbaF32, |s: &MonoF32| s.0, alpha = 1.0);
impl_broadcast!(MonoF32 => BgraF32, |s: &MonoF32| s.0, alpha = 1.0);
impl_broadcast!(MonoF64 => RgbaF64, |s: &MonoF64| s.0, alpha = 1.0);
impl_broadcast!(MonoF64 => BgraF64, |s: &MonoF64| s.0, alpha = 1.0);

// ═══════════════════════════════════════════════════════════════════════════════
// ColorSwap implementations (RGB ↔ BGR channel reorder)
// ═══════════════════════════════════════════════════════════════════════════════

impl_color_swap!(sat: Rgb8  [r, g, b]    <=> Bgr8  [b, g, r]);
impl_color_swap!(sat: Rgb16 [r, g, b]    <=> Bgr16 [b, g, r]);
impl_color_swap!(sat: Rgb32 [r, g, b]    <=> Bgr32 [b, g, r]);
impl_color_swap!(sat: Rgb64 [r, g, b]    <=> Bgr64 [b, g, r]);
impl_color_swap!(f32: RgbF32 [r, g, b]   <=> BgrF32 [b, g, r]);
impl_color_swap!(f32: RgbF64 [r, g, b]   <=> BgrF64 [b, g, r]);
impl_color_swap!(sat: Rgba8  [r, g, b, a] <=> Bgra8  [b, g, r, a]);
impl_color_swap!(sat: Rgba16 [r, g, b, a] <=> Bgra16 [b, g, r, a]);
impl_color_swap!(sat: Rgba32 [r, g, b, a] <=> Bgra32 [b, g, r, a]);
impl_color_swap!(sat: Rgba64 [r, g, b, a] <=> Bgra64 [b, g, r, a]);
impl_color_swap!(f32: RgbaF32 [r, g, b, a] <=> BgraF32 [b, g, r, a]);
impl_color_swap!(f32: RgbaF64 [r, g, b, a] <=> BgraF64 [b, g, r, a]);

// ═══════════════════════════════════════════════════════════════════════════════
// AddAlpha implementations (3-channel → 4-channel, alpha = max)
// ═══════════════════════════════════════════════════════════════════════════════

impl_add_alpha!(sat: Rgb8  => Rgba8,  [r, g, b], u8::MAX);
impl_add_alpha!(sat: Rgb16 => Rgba16, [r, g, b], u16::MAX);
impl_add_alpha!(sat: Rgb32 => Rgba32, [r, g, b], u32::MAX);
impl_add_alpha!(sat: Rgb64 => Rgba64, [r, g, b], u64::MAX);
impl_add_alpha!(f32: RgbF32 => RgbaF32, [r, g, b], 1.0);
impl_add_alpha!(f32: RgbF64 => RgbaF64, [r, g, b], 1.0);
impl_add_alpha!(sat: Bgr8  => Bgra8,  [b, g, r], u8::MAX);
impl_add_alpha!(sat: Bgr16 => Bgra16, [b, g, r], u16::MAX);
impl_add_alpha!(sat: Bgr32 => Bgra32, [b, g, r], u32::MAX);
impl_add_alpha!(sat: Bgr64 => Bgra64, [b, g, r], u64::MAX);
impl_add_alpha!(f32: BgrF32 => BgraF32, [b, g, r], 1.0);
impl_add_alpha!(f32: BgrF64 => BgraF64, [b, g, r], 1.0);

// ── sRGB AddAlpha ───────────────────────────────────────────────────────────
impl_add_alpha!(sat: Srgb8 => Srgba8, [r, g, b], u8::MAX);
impl ConvertPixel<SrgbMono8, SrgbMonoA8> for AddAlpha {
    #[inline]
    fn convert(&self, src: &SrgbMono8) -> SrgbMonoA8 {
        SrgbMonoA8::new(src.0.0, u8::MAX)
    }
}

// ── sRGB 16-bit AddAlpha ────────────────────────────────────────────────────
impl_add_alpha!(sat: Srgb16 => Srgba16, [r, g, b], u16::MAX);
impl ConvertPixel<SrgbMono16, SrgbMonoA16> for AddAlpha {
    #[inline]
    fn convert(&self, src: &SrgbMono16) -> SrgbMonoA16 {
        SrgbMonoA16::new(src.0.0, u16::MAX)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MonoA family — FullRange + Narrow depth conversions
//
// Same pattern as the multi-channel families but the MonoA types use [v, a]
// fields.  We reuse the existing impl_family_conversions! and
// impl_extended_family_conversions! macros.
// ═══════════════════════════════════════════════════════════════════════════════

impl_family_conversions!(MonoA8, MonoA16, MonoAF32, [v, a]);
impl_extended_family_conversions!(
    MonoA8,
    MonoA16,
    MonoA32,
    MonoA64,
    MonoAF32,
    MonoAF64,
    [v, a]
);

// ═══════════════════════════════════════════════════════════════════════════════
// AddAlpha — Mono → MonoA  (value copied, alpha = max)
//
// Mono types are tuple structs so we write these by hand instead of using
// the impl_add_alpha! macro which expects named fields on the source.
// ═══════════════════════════════════════════════════════════════════════════════

impl ConvertPixel<Mono8, MonoA8> for AddAlpha {
    #[inline]
    fn convert(&self, src: &Mono8) -> MonoA8 {
        MonoA8::new(src.as_bytes()[0], u8::MAX)
    }
}

impl ConvertPixel<Mono16, MonoA16> for AddAlpha {
    #[inline]
    fn convert(&self, src: &Mono16) -> MonoA16 {
        let bytes = src.as_bytes();
        let v = u16::from_ne_bytes([bytes[0], bytes[1]]);
        MonoA16::new(v, u16::MAX)
    }
}

impl ConvertPixel<Mono32, MonoA32> for AddAlpha {
    #[inline]
    fn convert(&self, src: &Mono32) -> MonoA32 {
        let bytes = src.as_bytes();
        let v = u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        MonoA32::new(v, u32::MAX)
    }
}

impl ConvertPixel<Mono64, MonoA64> for AddAlpha {
    #[inline]
    fn convert(&self, src: &Mono64) -> MonoA64 {
        let bytes = src.as_bytes();
        let v = u64::from_ne_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]);
        MonoA64::new(v, u64::MAX)
    }
}

// `AddAlpha` from bare `f32`/`f64` is intentionally not provided. Use
// `ConvertPixel<MonoF32, MonoAF32>` / `ConvertPixel<MonoF64, MonoAF64>`
// defined immediately below.

impl ConvertPixel<MonoF32, MonoAF32> for AddAlpha {
    #[inline]
    fn convert(&self, src: &MonoF32) -> MonoAF32 {
        MonoAF32::new(src.0, 1.0)
    }
}

impl ConvertPixel<MonoF64, MonoAF64> for AddAlpha {
    #[inline]
    fn convert(&self, src: &MonoF64) -> MonoAF64 {
        MonoAF64::new(src.0, 1.0)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Broadcast — MonoA → RGBA / BGRA  (v → R,G,B; alpha preserved)
// ═══════════════════════════════════════════════════════════════════════════════

// ── MonoA → RGBA ────────────────────────────────────────────────────────────
impl_broadcast_monoa!(sat:   MonoA8    => Rgba8);
impl_broadcast_monoa!(sat:   MonoA16   => Rgba16);
impl_broadcast_monoa!(sat:   MonoA32   => Rgba32);
impl_broadcast_monoa!(sat:   MonoA64   => Rgba64);
impl_broadcast_monoa!(float: MonoAF32  => RgbaF32);
impl_broadcast_monoa!(float: MonoAF64  => RgbaF64);

// ── MonoA → BGRA ────────────────────────────────────────────────────────────
impl_broadcast_monoa!(sat:   MonoA8    => Bgra8);
impl_broadcast_monoa!(sat:   MonoA16   => Bgra16);
impl_broadcast_monoa!(sat:   MonoA32   => Bgra32);
impl_broadcast_monoa!(sat:   MonoA64   => Bgra64);
impl_broadcast_monoa!(float: MonoAF32  => BgraF32);
impl_broadcast_monoa!(float: MonoAF64  => BgraF64);

// ═══════════════════════════════════════════════════════════════════════════════
// Luminance — RGBA / BGRA → MonoA  (BT.601 on R,G,B; alpha preserved)
// ═══════════════════════════════════════════════════════════════════════════════

// ── RGBA → MonoA ────────────────────────────────────────────────────────────
impl_luminance_monoa!(sat: Rgba8   => MonoA8,   lum_u8,  r, g, b, a);
impl_luminance_monoa!(sat: Rgba16  => MonoA16,  lum_u16, r, g, b, a);
impl_luminance_monoa!(sat: Rgba32  => MonoA32,  lum_u32, r, g, b, a);
impl_luminance_monoa!(sat: Rgba64  => MonoA64,  lum_u64, r, g, b, a);
impl_luminance_monoa!(f32: RgbaF32 => MonoAF32, r, g, b, a);
impl_luminance_monoa!(f64: RgbaF64 => MonoAF64, r, g, b, a);

// ── BGRA → MonoA ────────────────────────────────────────────────────────────
impl_luminance_monoa!(sat: Bgra8   => MonoA8,   lum_u8,  r, g, b, a);
impl_luminance_monoa!(sat: Bgra16  => MonoA16,  lum_u16, r, g, b, a);
impl_luminance_monoa!(sat: Bgra32  => MonoA32,  lum_u32, r, g, b, a);
impl_luminance_monoa!(sat: Bgra64  => MonoA64,  lum_u64, r, g, b, a);
impl_luminance_monoa!(f32: BgraF32 => MonoAF32, r, g, b, a);
impl_luminance_monoa!(f64: BgraF64 => MonoAF64, r, g, b, a);

// ═══════════════════════════════════════════════════════════════════════════════
// SrgbGamma implementations (sRGB ↔ linear)
//
// Converts between gamma-encoded Srgb8/Srgba8 and linear-light RgbF32/RgbaF32.
// For RGBA the alpha channel is always linear (scaled to [0,1], no gamma).
// ═══════════════════════════════════════════════════════════════════════════════

// ── Srgb8 ↔ RgbF32 ─────────────────────────────────────────────────────────

impl ConvertPixel<Srgb8, RgbF32> for SrgbGamma {
    #[inline]
    fn convert(&self, src: &Srgb8) -> RgbF32 {
        RgbF32::new(
            srgb_decode(src.r.0),
            srgb_decode(src.g.0),
            srgb_decode(src.b.0),
        )
    }
}

impl ConvertPixel<RgbF32, Srgb8> for SrgbGamma {
    #[inline]
    fn convert(&self, src: &RgbF32) -> Srgb8 {
        Srgb8::new(srgb_encode(src.r), srgb_encode(src.g), srgb_encode(src.b))
    }
}

// ── Srgba8 ↔ RgbaF32 ───────────────────────────────────────────────────────

impl ConvertPixel<Srgba8, RgbaF32> for SrgbGamma {
    #[inline]
    fn convert(&self, src: &Srgba8) -> RgbaF32 {
        RgbaF32::new(
            srgb_decode(src.r.0),
            srgb_decode(src.g.0),
            srgb_decode(src.b.0),
            src.a.0 as f32 / 255.0, // alpha is always linear
        )
    }
}

impl ConvertPixel<RgbaF32, Srgba8> for SrgbGamma {
    #[inline]
    fn convert(&self, src: &RgbaF32) -> Srgba8 {
        Srgba8::new(
            srgb_encode(src.r),
            srgb_encode(src.g),
            srgb_encode(src.b),
            (src.a.clamp(0.0, 1.0) * 255.0 + 0.5) as u8, // alpha is always linear
        )
    }
}

// ── SrgbMono8 ↔ f32 ────────────────────────────────────────────────────────

// `SrgbGamma` between `SrgbMono8` and bare `f32` is intentionally not
// provided. Use `ConvertPixel<SrgbMono8, MonoF32>` /
// `ConvertPixel<MonoF32, SrgbMono8>` defined immediately below.

// ── SrgbMono8 ↔ MonoF32 ────────────────────────────────────────────────────

impl ConvertPixel<SrgbMono8, MonoF32> for SrgbGamma {
    #[inline]
    fn convert(&self, src: &SrgbMono8) -> MonoF32 {
        MonoF32::new(srgb_decode(src.0.0))
    }
}

impl ConvertPixel<MonoF32, SrgbMono8> for SrgbGamma {
    #[inline]
    fn convert(&self, src: &MonoF32) -> SrgbMono8 {
        SrgbMono8::new(srgb_encode(src.0))
    }
}

// ── SrgbMonoA8 ↔ MonoAF32 ──────────────────────────────────────────────────

impl ConvertPixel<SrgbMonoA8, MonoAF32> for SrgbGamma {
    #[inline]
    fn convert(&self, src: &SrgbMonoA8) -> MonoAF32 {
        MonoAF32::new(
            srgb_decode(src.v.0),
            src.a.0 as f32 / 255.0, // alpha is always linear
        )
    }
}

impl ConvertPixel<MonoAF32, SrgbMonoA8> for SrgbGamma {
    #[inline]
    fn convert(&self, src: &MonoAF32) -> SrgbMonoA8 {
        SrgbMonoA8::new(
            srgb_encode(src.v),
            (src.a.clamp(0.0, 1.0) * 255.0 + 0.5) as u8, // alpha is always linear
        )
    }
}

// ── Srgb16 ↔ RgbF32 ────────────────────────────────────────────────────────

impl ConvertPixel<Srgb16, RgbF32> for SrgbGamma {
    #[inline]
    fn convert(&self, src: &Srgb16) -> RgbF32 {
        RgbF32::new(
            srgb_decode_16(src.r.0),
            srgb_decode_16(src.g.0),
            srgb_decode_16(src.b.0),
        )
    }
}

impl ConvertPixel<RgbF32, Srgb16> for SrgbGamma {
    #[inline]
    fn convert(&self, src: &RgbF32) -> Srgb16 {
        Srgb16::new(
            srgb_encode_16(src.r),
            srgb_encode_16(src.g),
            srgb_encode_16(src.b),
        )
    }
}

// ── Srgba16 ↔ RgbaF32 ──────────────────────────────────────────────────────

impl ConvertPixel<Srgba16, RgbaF32> for SrgbGamma {
    #[inline]
    fn convert(&self, src: &Srgba16) -> RgbaF32 {
        RgbaF32::new(
            srgb_decode_16(src.r.0),
            srgb_decode_16(src.g.0),
            srgb_decode_16(src.b.0),
            src.a.0 as f32 / 65535.0, // alpha is always linear
        )
    }
}

impl ConvertPixel<RgbaF32, Srgba16> for SrgbGamma {
    #[inline]
    fn convert(&self, src: &RgbaF32) -> Srgba16 {
        Srgba16::new(
            srgb_encode_16(src.r),
            srgb_encode_16(src.g),
            srgb_encode_16(src.b),
            (src.a.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16, // alpha is always linear
        )
    }
}

// ── SrgbMono16 ↔ f32 ───────────────────────────────────────────────────────

// `SrgbGamma` between `SrgbMono16` and bare `f32` is intentionally not
// provided. Use `ConvertPixel<SrgbMono16, MonoF32>` /
// `ConvertPixel<MonoF32, SrgbMono16>` defined immediately below.

// ── SrgbMono16 ↔ MonoF32 ───────────────────────────────────────────────────

impl ConvertPixel<SrgbMono16, MonoF32> for SrgbGamma {
    #[inline]
    fn convert(&self, src: &SrgbMono16) -> MonoF32 {
        MonoF32::new(srgb_decode_16(src.0.0))
    }
}

impl ConvertPixel<MonoF32, SrgbMono16> for SrgbGamma {
    #[inline]
    fn convert(&self, src: &MonoF32) -> SrgbMono16 {
        SrgbMono16::new(srgb_encode_16(src.0))
    }
}

// ── SrgbMonoA16 ↔ MonoAF32 ─────────────────────────────────────────────────

impl ConvertPixel<SrgbMonoA16, MonoAF32> for SrgbGamma {
    #[inline]
    fn convert(&self, src: &SrgbMonoA16) -> MonoAF32 {
        MonoAF32::new(
            srgb_decode_16(src.v.0),
            src.a.0 as f32 / 65535.0, // alpha is always linear
        )
    }
}

impl ConvertPixel<MonoAF32, SrgbMonoA16> for SrgbGamma {
    #[inline]
    fn convert(&self, src: &MonoAF32) -> SrgbMonoA16 {
        SrgbMonoA16::new(
            srgb_encode_16(src.v),
            (src.a.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16, // alpha is always linear
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Unary pixel strategies — threshold family + Invert
// ═══════════════════════════════════════════════════════════════════════════════
//
// These strategies implement `ConvertPixel<P, P>` (or `ConvertPixel<P, bool>`
// for `BinaryMask`) over any `HomogeneousPixel` whose channel type carries
// the operations the strategy performs. They bind to the minimum trait layer
// that admits the operation:
//
//   - `TruncateThreshold<P>`    : channel `Ord`              (min of values)
//   - `ToZeroThreshold<P>`      : channel `Ord + Zeroable`
//   - `ToZeroThresholdInv<P>`   : channel `Ord + Zeroable`
//   - `BinaryMask<P>`           : channel `Ord`              (output `bool`)
//   - `BinaryThreshold<P>`      : channel `Ord + Zeroable + BoundedChannel`
//   - `BinaryThresholdInv<P>`   : channel `Ord + Zeroable + BoundedChannel`
//   - `Invert`                  : channel `BoundedChannel + Sub<Output=Self>`
//
// `BoundedChannel` is what grants access to the channel's intrinsic
// maximum; its absence on `f32` / `f64` is load-bearing and is what
// makes `Invert` / `BinaryThreshold[ Inv]` refuse to compile for
// float-channel pixels (Philosophy §1, §8).

/// Binary threshold: output channel is `Channel::MAX` if `value > thresh`,
/// else `Channel::zero()`.
///
/// Comparison is per-channel against the corresponding channel of `thresh`.
/// The strategy preserves the input pixel type (`P → P`); to produce a
/// true `BinaryImage` (`P → bool`) instead, use [`BinaryMask`].
///
/// # Why pixel-typed `thresh: P` (not channel-typed)
///
/// Storing the threshold as a full pixel keeps the user in the same
/// vocabulary they already use, naturally admits per-channel thresholds
/// on multi-channel pixels, and avoids leaking the `Saturating<_>`
/// channel wrapper into call sites.
///
/// # Example
/// ```
/// # use fovea::image::{Image, ImageView};
/// # use fovea::pixel::Mono8;
/// # use fovea::transform::{BinaryThreshold, convert_image};
/// let img = Image::fill(4, 4, Mono8::new(200));
/// let out: Image<Mono8> = convert_image(&img, BinaryThreshold { thresh: Mono8::new(128) });
/// assert_eq!(out.pixel_at(0, 0), Mono8::new(255));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BinaryThreshold<P> {
    /// Threshold value, compared channel-wise against the input.
    pub thresh: P,
}

impl<P> ConvertPixel<P, P> for BinaryThreshold<P>
where
    P: WhiteChannel,
    P::Channel: Ord + ZeroablePixel,
{
    #[inline]
    fn convert(&self, src: &P) -> P {
        let white = P::white_channel();
        let channels = <P::Channels as Array<P::Channel>>::from_fn(|i| {
            if src.channel(i) > self.thresh.channel(i) {
                white
            } else {
                P::Channel::zero()
            }
        });
        P::from_channels(channels.as_ref())
    }
}

/// Inverted binary threshold: output channel is `Channel::zero()` if
/// `value > thresh`, else `Channel::MAX`.
///
/// The complement of [`BinaryThreshold`]; same bounds, same composition
/// rules.
///
/// # Example
/// ```
/// # use fovea::image::{Image, ImageView};
/// # use fovea::pixel::Mono8;
/// # use fovea::transform::{BinaryThresholdInv, convert_image};
/// let img = Image::fill(4, 4, Mono8::new(200));
/// let out: Image<Mono8> = convert_image(&img, BinaryThresholdInv { thresh: Mono8::new(128) });
/// assert_eq!(out.pixel_at(0, 0), Mono8::new(0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BinaryThresholdInv<P> {
    /// Threshold value, compared channel-wise against the input.
    pub thresh: P,
}

impl<P> ConvertPixel<P, P> for BinaryThresholdInv<P>
where
    P: WhiteChannel,
    P::Channel: Ord + ZeroablePixel,
{
    #[inline]
    fn convert(&self, src: &P) -> P {
        let white = P::white_channel();
        let channels = <P::Channels as Array<P::Channel>>::from_fn(|i| {
            if src.channel(i) > self.thresh.channel(i) {
                P::Channel::zero()
            } else {
                white
            }
        });
        P::from_channels(channels.as_ref())
    }
}

/// Truncate threshold: output channel is `min(value, thresh)` per channel.
///
/// Also known as “trunc” thresholding in the OpenCV vocabulary. Does not
/// need a channel maximum, does not need a channel zero — bound is just
/// `Ord`.
///
/// # Example
/// ```
/// # use fovea::image::{Image, ImageView};
/// # use fovea::pixel::Mono8;
/// # use fovea::transform::{TruncateThreshold, convert_image};
/// let img = Image::fill(4, 4, Mono8::new(200));
/// let out: Image<Mono8> = convert_image(&img, TruncateThreshold { thresh: Mono8::new(128) });
/// assert_eq!(out.pixel_at(0, 0), Mono8::new(128));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TruncateThreshold<P> {
    /// Upper bound for each channel.
    pub thresh: P,
}

impl<P> ConvertPixel<P, P> for TruncateThreshold<P>
where
    P: HomogeneousPixel,
    P::Channel: Ord,
{
    #[inline]
    fn convert(&self, src: &P) -> P {
        let channels = <P::Channels as Array<P::Channel>>::from_fn(|i| {
            let v = src.channel(i);
            let t = self.thresh.channel(i);
            if v > t { t } else { v }
        });
        P::from_channels(channels.as_ref())
    }
}

/// To-zero threshold: output channel is `value` if `value > thresh`, else
/// `zero` — per channel.
///
/// Preserves values above the threshold; zeroes everything at or below.
///
/// # Example
/// ```
/// # use fovea::image::{Image, ImageView};
/// # use fovea::pixel::Mono8;
/// # use fovea::transform::{ToZeroThreshold, convert_image};
/// let img = Image::fill(4, 4, Mono8::new(200));
/// let out: Image<Mono8> = convert_image(&img, ToZeroThreshold { thresh: Mono8::new(128) });
/// assert_eq!(out.pixel_at(0, 0), Mono8::new(200));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ToZeroThreshold<P> {
    /// Threshold value, compared channel-wise against the input.
    pub thresh: P,
}

impl<P> ConvertPixel<P, P> for ToZeroThreshold<P>
where
    P: HomogeneousPixel,
    P::Channel: Ord + ZeroablePixel,
{
    #[inline]
    fn convert(&self, src: &P) -> P {
        let channels = <P::Channels as Array<P::Channel>>::from_fn(|i| {
            let v = src.channel(i);
            if v > self.thresh.channel(i) {
                v
            } else {
                P::Channel::zero()
            }
        });
        P::from_channels(channels.as_ref())
    }
}

/// To-zero inverted threshold: output channel is `zero` if `value > thresh`,
/// else `value` — per channel.
///
/// The complement of [`ToZeroThreshold`]: keeps values at or below the
/// threshold, zeroes everything above.
///
/// # Example
/// ```
/// # use fovea::image::{Image, ImageView};
/// # use fovea::pixel::Mono8;
/// # use fovea::transform::{ToZeroThresholdInv, convert_image};
/// let img = Image::fill(4, 4, Mono8::new(200));
/// let out: Image<Mono8> = convert_image(&img, ToZeroThresholdInv { thresh: Mono8::new(128) });
/// assert_eq!(out.pixel_at(0, 0), Mono8::new(0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ToZeroThresholdInv<P> {
    /// Threshold value, compared channel-wise against the input.
    pub thresh: P,
}

impl<P> ConvertPixel<P, P> for ToZeroThresholdInv<P>
where
    P: HomogeneousPixel,
    P::Channel: Ord + ZeroablePixel,
{
    #[inline]
    fn convert(&self, src: &P) -> P {
        let channels = <P::Channels as Array<P::Channel>>::from_fn(|i| {
            let v = src.channel(i);
            if v > self.thresh.channel(i) {
                P::Channel::zero()
            } else {
                v
            }
        });
        P::from_channels(channels.as_ref())
    }
}

/// Binary-valued threshold producing a `bool` output (i.e. a
/// [`BinaryImage`](crate::image::BinaryImage)).
///
/// Collapses a potentially multi-channel pixel into a single boolean via
/// **all-channels-above** reduction: the output is `true` iff *every*
/// channel exceeds the corresponding channel of `thresh`.
///
/// Use this (rather than [`BinaryThreshold`]) when the downstream
/// consumer expects `ImageView<Pixel = bool>` — for example a morphology
/// operator (`erode` / `dilate`), a connected-components routine, or a
/// blob-analysis pass. No byte is wasted encoding `0` or `MAX`, and no
/// `!= 0` bridge conversion is needed at the boundary.
///
/// For per-channel binary output on multi-channel pixels, use
/// [`BinaryThreshold`] instead (which preserves per-channel structure).
/// For reductions other than all-channels-above (any-channel,
/// luminance-based), compose explicitly — e.g. convert to grayscale
/// first, then apply `BinaryMask`.
///
/// # Example
/// ```
/// # use fovea::image::{BinaryImage, Image, ImageView};
/// # use fovea::pixel::Mono8;
/// # use fovea::transform::{BinaryMask, convert_image};
/// let img = Image::fill(4, 4, Mono8::new(200));
/// let mask: BinaryImage = convert_image(&img, BinaryMask { thresh: Mono8::new(128) });
/// assert!(mask.pixel_at(0, 0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BinaryMask<P> {
    /// Threshold value, compared channel-wise against the input.
    pub thresh: P,
}

impl<P> ConvertPixel<P, bool> for BinaryMask<P>
where
    P: HomogeneousPixel,
    P::Channel: Ord,
{
    #[inline]
    fn convert(&self, src: &P) -> bool {
        for i in 0..P::CHANNEL_COUNT {
            if src.channel(i) <= self.thresh.channel(i) {
                return false;
            }
        }
        true
    }
}

/// Per-channel inversion: `white - value`, where `white` is the
/// pixel-level saturated channel value ([`WhiteChannel::white_channel`]).
///
/// Bound via [`WhiteChannel`](crate::pixel::WhiteChannel), so
/// floating-point pixel families (`MonoF32`, `RgbF32`, …) are
/// **deliberately excluded**: there is no intrinsic maximum for `f32` /
/// `f64`, and the library refuses to bake in a `[0, 1]` assumption
/// (Philosophy §1 "Types are the spec", §8 "Surface information, don't
/// decide"). Users who want float inversion name the range assumption
/// explicitly — for example with `PixelMap(|p: &MonoF32| MonoF32(1.0 - p.0))`.
///
/// # Reduced-range pixels
///
/// For reduced-range pixels like [`Mono<BITS>`](crate::pixel::Mono),
/// `Invert` uses the pixel-level saturated value (e.g. `1023` for
/// `Mono<10>`), not the channel type's storage maximum
/// (`Saturating<u16>::MAX == 65535`). This preserves the pixel's
/// invariant (without it, inverting a `Mono<10>` pixel would corrupt
/// the high-bit zeroing the type guarantees).
///
/// # Example
/// ```
/// # use fovea::image::{Image, ImageView};
/// # use fovea::pixel::Mono8;
/// # use fovea::transform::{Invert, convert_image};
/// let img = Image::fill(4, 4, Mono8::new(10));
/// let out: Image<Mono8> = convert_image(&img, Invert);
/// assert_eq!(out.pixel_at(0, 0), Mono8::new(245));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Invert;

impl<P> ConvertPixel<P, P> for Invert
where
    P: WhiteChannel,
    P::Channel: core::ops::Sub<Output = P::Channel>,
{
    #[inline]
    fn convert(&self, src: &P) -> P {
        let white = P::white_channel();
        let channels = <P::Channels as Array<P::Channel>>::from_fn(|i| white - src.channel(i));
        P::from_channels(channels.as_ref())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Clamp<P> — per-pixel value-range restriction
// ═══════════════════════════════════════════════════════════════════════════════
//
// Note: This is NOT the same as the cross-type narrowing strategy previously
// named `Clamp`, which has been renamed to `Narrow` (Phase 0). `Clamp` here
// is the classical CV operation: restrict each channel value to a closed
// interval `[lo, hi]` *within the same pixel type*.
//
// The bounds are stored as full pixels — same vocabulary the user already
// works in — so uniform-range clamping (`lo: Mono8::new(20)`) and
// per-channel ranges (`lo: Rgb8::new(16, 16, 16)`) are both natural —
// the same parallel design argument that motivates `BinaryThreshold`'s
// pixel-typed `thresh`.

/// Per-pixel value-range restriction: `clamp(channel, lo, hi)` per channel.
///
/// The result has every channel clamped to `[lo.channel, hi.channel]`;
/// the pixel type is preserved.
///
/// # Construction
///
/// Use [`Clamp::new`] — the only public constructor. It validates
/// `lo <= hi` channel-wise so that inverted ranges (which would collapse
/// every input to `hi`) are rejected at construction time. Fields are
/// **private** to keep this invariant load-bearing; read them back with
/// [`Clamp::lo`] / [`Clamp::hi`] if you need them.
///
/// # Not to be confused with [`Narrow`]
///
/// `Narrow` is the cross-type narrowing conversion (e.g. `u16 → u8`,
/// clamping values that don't fit). `Clamp` is the same-type value
/// restriction — the classical computer-vision operation.
///
/// # Example
/// ```
/// # use fovea::image::{Image, ImageView};
/// # use fovea::pixel::Mono8;
/// # use fovea::transform::{Clamp, convert_image};
/// let img = Image::fill(4, 4, Mono8::new(10));
/// let out: Image<Mono8> = convert_image(
///     &img,
///     Clamp::new(Mono8::new(20), Mono8::new(235)),
/// );
/// assert_eq!(out.pixel_at(0, 0), Mono8::new(20)); // clamped up to lo
/// ```
///
/// Per-channel ranges on multi-channel pixels are naturally expressible:
/// ```
/// # use fovea::pixel::Rgb8;
/// # use fovea::transform::{Clamp, ConvertPixel};
/// let strat = Clamp::new(
///     Rgb8::new(16, 16, 16),
///     Rgb8::new(235, 240, 235),
/// );
/// assert_eq!(strat.convert(&Rgb8::new(5, 250, 100)), Rgb8::new(16, 240, 100));
/// ```
///
/// Direct struct-literal construction is rejected so the `lo <= hi`
/// invariant cannot be bypassed:
///
/// ```compile_fail
/// # use fovea::pixel::Mono8;
/// # use fovea::transform::Clamp;
/// // ERROR: field `lo` of struct `Clamp` is private.
/// let _ = Clamp { lo: Mono8::new(200), hi: Mono8::new(50) };
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Clamp<P> {
    // Private: the `lo <= hi` invariant established by `Clamp::new`
    // must not be bypassable via struct literals. See P1-6 / the
    // `convert` impl, which assumes well-ordered bounds.
    lo: P,
    hi: P,
}

impl<P> Clamp<P>
where
    P: HomogeneousPixel,
    P::Channel: Ord,
{
    /// Construct a [`Clamp`] strategy after validating that `lo <= hi`
    /// channel-wise.
    ///
    /// # Panics (Tier 3 — programmer bug)
    ///
    /// Panics if any channel of `lo` is greater than the corresponding
    /// channel of `hi`. An inverted range collapses every input to `hi`
    /// (`min(max(v, lo), hi) == hi`), which is almost certainly not what
    /// the caller intended.
    ///
    /// # Example
    ///
    /// ```
    /// # use fovea::pixel::Mono8;
    /// # use fovea::transform::Clamp;
    /// let strat = Clamp::new(Mono8::new(20), Mono8::new(235));
    /// assert_eq!(strat.lo(), Mono8::new(20));
    /// assert_eq!(strat.hi(), Mono8::new(235));
    /// ```
    ///
    /// Inverted ranges are rejected:
    ///
    /// ```should_panic
    /// # use fovea::pixel::Mono8;
    /// # use fovea::transform::Clamp;
    /// let _ = Clamp::new(Mono8::new(200), Mono8::new(50));
    /// ```
    #[inline]
    pub fn new(lo: P, hi: P) -> Self {
        // Per-channel validation: matches the channel-wise semantics of
        // `convert`. Done once at construction so the hot loop pays
        // nothing for it (PHILOSOPHY § "checks belong where the data
        // becomes a contract").
        let n = <<P as HomogeneousPixel>::Channels as Array<P::Channel>>::LEN;
        for i in 0..n {
            if lo.channel(i) > hi.channel(i) {
                panic!(
                    "Clamp::new: lo > hi on channel {i} — every input would \
                     collapse to `hi`. Did you swap the arguments?"
                );
            }
        }
        Self { lo, hi }
    }
}

impl<P: Copy> Clamp<P> {
    /// The lower bound supplied at construction.
    #[inline]
    pub fn lo(&self) -> P {
        self.lo
    }

    /// The upper bound supplied at construction.
    #[inline]
    pub fn hi(&self) -> P {
        self.hi
    }
}

impl<P> ConvertPixel<P, P> for Clamp<P>
where
    P: HomogeneousPixel,
    P::Channel: Ord,
{
    #[inline]
    fn convert(&self, src: &P) -> P {
        let channels = <P::Channels as Array<P::Channel>>::from_fn(|i| {
            let v = src.channel(i);
            let lo = self.lo.channel(i);
            let hi = self.hi.channel(i);
            // Explicit two-step: clamp up to `lo`, then down to `hi`.
            // No `lo <= hi` precondition check here: that invariant is
            // established once by `Clamp::new` (P1-6) and the private
            // fields prevent it from being violated. Re-checking per
            // pixel would burn N*M cycles for a constant property.
            let v = if v < lo { lo } else { v };
            if v > hi { hi } else { v }
        });
        P::from_channels(channels.as_ref())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// BrightnessContrast<S> — affine per-pixel transform
// ═══════════════════════════════════════════════════════════════════════════════
//
// Implements the classical affine intensity transform
//
//     output_channel = contrast * input_channel + brightness
//
// via the `LinearPixel<S>` pathway: `scale_add` (FMA-optimized per
// OPT-002) + `uniform` (broadcast brightness across channels per PLAN
// §3.4) + `FromLinear` (round + clamp).
//
// The scalar parameter `S` defaults to `f32`; users of f64-accumulator
// pipelines (`MonoF64`, `Mono64`, …) write `BrightnessContrast::<f64>`
// to pick up the native-f64 `LinearPixel<f64>` path, avoiding an
// f32→f64 widening in the hot loop.

/// Linear point transform: `output = contrast * input + brightness` per
/// channel, with proper rounding and clamping back to the storage type.
///
/// # Scalar type parameter `S`
///
/// `S` defaults to `f32`. Users of f64-accumulator pixels (`MonoF64`,
/// `Mono64`, `Mono32`, …) who need full-precision scalars write
/// `BrightnessContrast::<f64> { brightness, contrast }`. The
/// `LinearPixel<S>` bound then resolves to the matching scalar path.
///
/// # Trait bound
///
/// The strategy requires `LinearPixel<S> + FromLinear<P::Accumulator>` —
/// it does **not** require `LinearSpace`. This is a point transform,
/// not an interpolation (Philosophy §3 — bind to the minimum layer that
/// admits the operation).
///
/// # Example
/// ```
/// # use fovea::image::{Image, ImageView};
/// # use fovea::pixel::Mono8;
/// # use fovea::transform::{BrightnessContrast, convert_image};
/// let img = Image::fill(4, 4, Mono8::new(100));
/// // Increase contrast 1.5x and brightness +10.
/// let out: Image<Mono8> = convert_image(
///     &img,
///     BrightnessContrast { brightness: 10.0, contrast: 1.5 },
/// );
/// // 100 * 1.5 + 10 = 160
/// assert_eq!(out.pixel_at(0, 0), Mono8::new(160));
/// ```
///
/// Identity (contrast = 1.0, brightness = 0.0) preserves the image:
/// ```
/// # use fovea::image::{Image, ImageView};
/// # use fovea::pixel::Mono8;
/// # use fovea::transform::{BrightnessContrast, convert_image};
/// let img = Image::fill(2, 2, Mono8::new(123));
/// let out: Image<Mono8> = convert_image(
///     &img,
///     BrightnessContrast { brightness: 0.0, contrast: 1.0 },
/// );
/// assert_eq!(out.pixel_at(0, 0), Mono8::new(123));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BrightnessContrast<S = f32> {
    /// Additive term applied to every channel.
    pub brightness: S,
    /// Multiplicative gain applied to every channel before the additive term.
    pub contrast: S,
}

impl<P, S> ConvertPixel<P, P> for BrightnessContrast<S>
where
    P: crate::pixel::LinearPixel<S>,
    P: crate::pixel::FromLinear<<P as crate::pixel::LinearPixel<S>>::Accumulator>,
    S: Copy,
{
    #[inline]
    fn convert(&self, src: &P) -> P {
        // scale_add (FMA where available) + uniform (channel broadcast).
        // Three lines. Reuses the derive-macro-generated per-field code;
        // the compiler sees named field operations, not byte reinterpretations.
        let addend = <P as crate::pixel::LinearPixel<S>>::uniform(self.brightness);
        let acc = <P as crate::pixel::LinearPixel<S>>::scale_add(src, self.contrast, addend);
        <P as crate::pixel::FromLinear<_>>::from_linear(acc)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Lut<V> / ChannelLut — lookup-table conversion strategies
// ═══════════════════════════════════════════════════════════════════════════════
//
// Two strategies, two operations:
//
//   * `Lut<V>`      — `Mono8 -> V` cross-type table (256 entries). Classic
//                     pseudocolor / calibration-curve use case: map every
//                     8-bit grayscale value to an output pixel of any type.
//
//   * `ChannelLut`  — `P -> P` per-channel `u8 -> u8` table. For Rgb8 /
//                     Rgba8 / MonoA8 / Mono8 etc. — apply the same 256-entry
//                     u8→u8 mapping independently to every channel.
//
// Separate types (rather than one generic) because the two operations have
// genuinely different signatures: `Lut<V>` is parameterized by the output
// pixel type and produces cross-type conversions; `ChannelLut` is tied to
// u8-channel pixels and is same-type. A single generic design would either
// conflict on the blanket `ConvertPixel<Mono8, Mono8>` impl or force users
// to turbofish the output type everywhere. (The parallel argument
// applies to `BinaryThreshold` vs `BinaryMask`.)

/// Lookup-table conversion from `Mono8` to any pixel type `V`.
///
/// The table holds 256 entries, one per possible `u8` input value. Every
/// index is valid (the input channel is `u8`, max 255), so no bounds
/// check is needed at runtime.
///
/// # Cross-type by design
///
/// `V` need not equal `Mono8`. Pseudocolor mapping (`Mono8 → Rgb8`),
/// calibration-curve linearization (`Mono8 → MonoF32`), and intensity
/// remapping (`Mono8 → Mono8`) all use the same strategy — the only
/// thing that changes is the output type carried through the type
/// system.
///
/// # Example
///
/// Pseudocolor mapping:
///
/// ```
/// # use fovea::image::{Image, ImageView};
/// # use fovea::pixel::{Mono8, Rgb8};
/// # use fovea::transform::{Lut, convert_image};
/// // Heat map: black → red → yellow → white.
/// let lut: Lut<Rgb8> = Lut::from_fn(|v| {
///     if v < 128 {
///         Rgb8::new(v.saturating_mul(2), 0, 0)
///     } else {
///         Rgb8::new(255, (v - 128).saturating_mul(2), 0)
///     }
/// });
/// let gray: Image<Mono8> = Image::fill(2, 2, Mono8::new(200));
/// let color: Image<Rgb8> = convert_image(&gray, lut);
/// assert_eq!(color.pixel_at(0, 0), Rgb8::new(255, 144, 0));
/// ```
///
/// Identity LUT is a no-op on `Mono8`:
///
/// ```
/// # use fovea::pixel::Mono8;
/// # use fovea::transform::{ConvertPixel, Lut};
/// let lut: Lut<Mono8> = Lut::from_fn(Mono8::new);
/// assert_eq!(lut.convert(&Mono8::new(42)), Mono8::new(42));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lut<V> {
    table: [V; 256],
}

impl<V> Lut<V> {
    /// Construct a LUT from an explicit 256-entry table.
    ///
    /// `const`-callable and does not require `V: Copy` — this enables
    /// compile-time-constructed pseudocolor palettes and other tables
    /// whose entries are non-`Copy` pixel types (hypothetical future
    /// cases; all current pixel types are `Copy`).
    #[inline]
    pub const fn new(table: [V; 256]) -> Self {
        Self { table }
    }
}

impl<V: Copy> Lut<V> {
    /// Construct a LUT by evaluating `f(index)` for every `u8` index.
    ///
    /// Ergonomic wrapper around [`Lut::new`] for the common case where
    /// the table is computed from a closed-form function (gamma curve,
    /// pseudocolor palette, contrast curve, …).
    #[inline]
    pub fn from_fn<F: FnMut(u8) -> V>(mut f: F) -> Self {
        let table = <[V; 256] as Array<V>>::from_fn(|i| f(i as u8));
        Self { table }
    }
}

impl<V: Copy> ConvertPixel<Mono8, V> for Lut<V> {
    #[inline]
    fn convert(&self, src: &Mono8) -> V {
        // `Mono8::value()` returns `u8`; indices 0..=255 are always valid.
        self.table[src.value() as usize]
    }
}

/// Per-channel `u8 → u8` lookup table for pixels whose channel type is
/// `Saturating<u8>`.
///
/// Applies the same 256-entry table independently to every channel of a
/// [`HomogeneousPixel`] whose channel is `Saturating<u8>` — so `Rgb8`,
/// `Rgba8`, `Bgr8`, `Bgra8`, `MonoA8`, and `Mono8` all accept the same
/// strategy value with no user-side boilerplate.
///
/// # Not generic over the channel type
///
/// A `u16 → u16` variant would need a 65 536-entry table — a different
/// operation that deserves a different name. If that becomes a real
/// need, it can be added as `ChannelLut16` later without breaking
/// changes (Philosophy §9 — "Extension by addition").
///
/// # Example
///
/// Per-channel gamma-like curve applied to all three channels of an Rgb8:
///
/// ```
/// # use fovea::pixel::Rgb8;
/// # use fovea::transform::{ChannelLut, ConvertPixel};
/// // Simple contrast-boost curve.
/// let curve = ChannelLut::from_fn(|v| ((v as u16) * 3 / 2).min(255) as u8);
/// assert_eq!(curve.convert(&Rgb8::new(100, 150, 200)), Rgb8::new(150, 225, 255));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChannelLut {
    table: [u8; 256],
}

impl ChannelLut {
    /// Construct a per-channel LUT from an explicit 256-entry table.
    #[inline]
    pub const fn new(table: [u8; 256]) -> Self {
        Self { table }
    }

    /// Construct a per-channel LUT by evaluating `f(index)` for every
    /// `u8` index.
    #[inline]
    pub fn from_fn<F: FnMut(u8) -> u8>(mut f: F) -> Self {
        let table = <[u8; 256] as Array<u8>>::from_fn(|i| f(i as u8));
        Self { table }
    }

    /// Look up a single `u8` value in the table.
    ///
    /// Equivalent to `self.table()[idx as usize]`. Useful when applying
    /// a `ChannelLut` to a single channel by hand — e.g. inside a
    /// per-channel-distinct LUT loop that cannot be expressed as one
    /// `ConvertPixel` call.
    #[inline]
    pub const fn lookup(&self, idx: u8) -> u8 {
        self.table[idx as usize]
    }
}

impl<P> ConvertPixel<P, P> for ChannelLut
where
    P: HomogeneousPixel<Channel = std::num::Saturating<u8>>,
{
    #[inline]
    fn convert(&self, src: &P) -> P {
        let channels = <P::Channels as Array<P::Channel>>::from_fn(|i| {
            let ch: std::num::Saturating<u8> = src.channel(i);
            std::num::Saturating(self.table[ch.0 as usize])
        });
        P::from_channels(channels.as_ref())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Depalettize — palette-lookup conversion strategy
// ═══════════════════════════════════════════════════════════════════════════════

/// Palette-lookup conversion strategy.
///
/// Converts an [`Indexed8`] pixel to a color pixel of type `P` by
/// indexing into the stored palette.  The type parameter `P` encodes
/// the palette's color space, so the type system carries that
/// information through the entire conversion pipeline.
///
/// Because `Indexed8` wraps a `u8` (max value 255) and the palette is
/// always 256 entries, every possible index is valid and no bounds
/// check is needed at runtime.
///
/// # Examples
///
/// ```
/// # use fovea::pixel::{Indexed8, Rgb8};
/// # use fovea::transform::{ConvertPixel, Depalettize};
/// let palette = [Rgb8::new(0, 0, 0); 256]; // all black
/// let strategy = Depalettize::new(palette);
/// assert_eq!(strategy.convert(&Indexed8(0)), Rgb8::new(0, 0, 0));
/// ```
pub struct Depalettize<P> {
    palette: [P; 256],
}

impl<P: Copy> Depalettize<P> {
    /// Create a new `Depalettize` strategy from a full 256-entry palette.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fovea::pixel::{Indexed8, Rgb8};
    /// # use fovea::transform::{ConvertPixel, Depalettize};
    /// let palette = [Rgb8::new(0, 0, 0); 256];
    /// let strategy = Depalettize::new(palette);
    /// assert_eq!(strategy.convert(&Indexed8(0)), Rgb8::new(0, 0, 0));
    /// ```
    pub fn new(palette: [P; 256]) -> Self {
        Self { palette }
    }
}

impl<P: Copy + ZeroablePixel> Depalettize<P> {
    /// Build from a slice shorter than 256 entries.
    /// Remaining entries are zero-filled.
    ///
    /// # Panics
    ///
    /// Panics if `entries.len() > 256`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fovea::pixel::{Indexed8, Rgb8};
    /// # use fovea::transform::{ConvertPixel, Depalettize};
    /// let entries = [Rgb8::new(255, 0, 0), Rgb8::new(0, 255, 0)];
    /// let strategy = Depalettize::from_slice(&entries);
    /// assert_eq!(strategy.convert(&Indexed8(0)), Rgb8::new(255, 0, 0));
    /// assert_eq!(strategy.convert(&Indexed8(1)), Rgb8::new(0, 255, 0));
    /// assert_eq!(strategy.convert(&Indexed8(2)), Rgb8::new(0, 0, 0)); // zero-filled
    /// ```
    pub fn from_slice(entries: &[P]) -> Self {
        assert!(entries.len() <= 256);
        let mut palette = [P::zero(); 256];
        palette[..entries.len()].copy_from_slice(entries);
        Self { palette }
    }
}

impl<P: Copy> ConvertPixel<Indexed8, P> for Depalettize<P> {
    #[inline]
    fn convert(&self, src: &Indexed8) -> P {
        self.palette[src.0 as usize]
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PixelMap — closure-based custom conversion
// ═══════════════════════════════════════════════════════════════════════════════

/// Closure-based pixel conversion strategy.
///
/// Wraps any `Fn(&Src) -> Dst` closure and uses it as a [`ConvertPixel`]
/// implementation.  This lets you pass arbitrary per-pixel logic to
/// [`convert_image`] and [`convert_image_into`] without defining a new
/// strategy type.
///
/// # Examples
/// ```
/// # use fovea::image::{Image, ImageView};
/// # use fovea::pixel::{Mono8, PlainChannel};
/// # use fovea::transform::{ConvertPixel, PixelMap, convert_image};
/// let img = Image::fill(2, 2, Mono8::new(200));
/// // Byte-layout items (`as_bytes`, `SIZE`, …) live on `PlainChannel`;
/// // `PlainPixel` extends it but the call below only needs the
/// // channel-role trait in scope.
/// let inverted: Image<Mono8> = convert_image(&img, PixelMap(|src: &Mono8| {
///     Mono8::new(255 - src.as_bytes()[0])
/// }));
/// assert_eq!(inverted.pixel_at(0, 0), Mono8::new(55));
/// ```
///
/// Custom pixel types work just as well:
/// ```
/// # use fovea::transform::{ConvertPixel, PixelMap};
/// // Hypothetical: convert a complex-valued pixel to its magnitude.
/// // let res: Image<f64> = convert_image(&img, PixelMap(|c: &Complex| c.magnitude()));
/// ```
pub struct PixelMap<F>(pub F);

impl<Src, Dst, F> ConvertPixel<Src, Dst> for PixelMap<F>
where
    F: Fn(&Src) -> Dst,
{
    #[inline]
    fn convert(&self, src: &Src) -> Dst {
        (self.0)(src)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Then — strategy combinator (pipe two conversions)
// ═══════════════════════════════════════════════════════════════════════════════

/// Combinator that chains two conversion strategies into one.
///
/// `Then<A, B, Mid>` first applies `A` to convert `Src → Mid`, then applies
/// `B` to convert `Mid → Dst`.  The intermediate pixel is never materialized
/// as an image — the conversion happens per-pixel in a single pass, avoiding
/// any intermediate allocation.
///
/// When both `A` and `B` are zero-sized strategy types (like all built-in
/// strategies), `Then<A, B, Mid>` is also zero-sized and the compiler inlines
/// the entire chain into a single loop body.
///
/// You rarely need to construct `Then` directly — use the
/// [`ConvertPixelExt::then`] method instead:
///
/// ```
/// # use fovea::pixel::{Rgb8, Bgr8, Bgr16};
/// # use fovea::transform::{ConvertPixel, ConvertPixelExt, ColorSwap, FullRange};
/// // Rgb8 → Bgr8 → Bgr16  (cross-depth colour swap, single pass)
/// let method = ColorSwap.then::<Bgr8, _>(FullRange);
/// let result: Bgr16 = method.convert(&Rgb8::new(200, 100, 50));
/// ```
///
/// Chains of any length are supported via repeated `.then()` calls:
///
/// ```
/// # use fovea::pixel::{Rgb8, Bgr8, Bgra8, Bgra16};
/// # use fovea::transform::{ConvertPixel, ConvertPixelExt, ColorSwap, AddAlpha, FullRange};
/// // Rgb8 → Bgr8 → Bgra8 → Bgra16  (three steps, zero intermediate images)
/// let method = ColorSwap.then::<Bgr8, _>(AddAlpha).then::<Bgra8, _>(FullRange);
/// let result: Bgra16 = method.convert(&Rgb8::new(255, 0, 0));
/// // pure red: R channel survives swap+alpha+widen
/// assert_eq!(result, Bgra16::new(0, 0, 65535, 65535));
/// ```
pub struct Then<A, B, Mid> {
    first: A,
    second: B,
    _mid: PhantomData<fn(Mid) -> Mid>,
}

impl<Src, Mid, Dst, A, B> ConvertPixel<Src, Dst> for Then<A, B, Mid>
where
    A: ConvertPixel<Src, Mid>,
    B: ConvertPixel<Mid, Dst>,
{
    #[inline]
    fn convert(&self, src: &Src) -> Dst {
        let mid = self.first.convert(src);
        self.second.convert(&mid)
    }
}

/// Extension trait that adds the [`.then()`](ConvertPixelExt::then) combinator
/// to any conversion strategy.
///
/// This trait is automatically implemented for every [`Sized`] type, so you
/// never need to implement it yourself — just import it and call `.then()`.
///
/// The first type parameter `Mid` names the intermediate pixel type that the
/// first strategy produces and the second strategy consumes.  Because built-in
/// strategies like [`FullRange`] implement [`ConvertPixel`] for many type
/// pairs, Rust's trait solver usually cannot infer `Mid` on its own — you
/// supply it via turbofish: `.then::<Mid, _>(next)`.
///
/// This is intentional: naming the intermediate type makes the pipeline
/// explicit and self-documenting.
///
/// # Examples
///
/// Cross-depth colour swap (`Rgb8 → Bgr8 → Bgr16`):
///
/// ```
/// # use fovea::image::Image;
/// # use fovea::pixel::{Rgb8, Bgr8, Bgr16};
/// # use fovea::transform::{convert_image, ConvertPixelExt, ColorSwap, FullRange};
/// let img = Image::fill(4, 4, Rgb8::new(200, 100, 50));
/// let out: Image<Bgr16> = convert_image(&img, ColorSwap.then::<Bgr8, _>(FullRange));
/// ```
///
/// Cross-depth luminance (`Rgb16 → Mono16 → Mono8`):
///
/// ```
/// # use fovea::image::Image;
/// # use fovea::pixel::{Rgb16, Mono8, Mono16};
/// # use fovea::transform::{convert_image, ConvertPixelExt, Luminance, FullRange};
/// let img = Image::fill(4, 4, Rgb16::new(65535, 65535, 65535));
/// let out: Image<Mono8> = convert_image(&img, Luminance.then::<Mono16, _>(FullRange));
/// ```
///
/// Triple chain (`Rgb8 → Bgr8 → Bgra8 → Bgra16`):
///
/// ```
/// # use fovea::pixel::{Rgb8, Bgr8, Bgra8, Bgra16};
/// # use fovea::transform::{ConvertPixel, ConvertPixelExt, ColorSwap, AddAlpha, FullRange};
/// let px = Rgb8::new(255, 0, 0);
/// let result: Bgra16 = ColorSwap
///     .then::<Bgr8, _>(AddAlpha)
///     .then::<Bgra8, _>(FullRange)
///     .convert(&px);
/// // pure red → Bgra16 with R=65535, others=0, A=65535
/// assert_eq!(result, Bgra16::new(0, 0, 65535, 65535));
/// ```
pub trait ConvertPixelExt: Sized {
    /// Chain this strategy with a second strategy, producing a [`Then`]
    /// combinator that converts `Src → Mid → Dst` in a single step.
    ///
    /// # Type Parameters
    ///
    /// - `Mid` — the intermediate pixel type (first parameter so you can
    ///   write `.then::<Mid, _>(next)` and let the compiler infer `B`).
    /// - `B` — the second strategy (inferred from the argument).
    ///
    /// # Example
    ///
    /// ```
    /// # use fovea::pixel::{Rgb8, Bgr8, Bgr16};
    /// # use fovea::transform::{ConvertPixel, ConvertPixelExt, ColorSwap, FullRange};
    /// let bgr16: Bgr16 = ColorSwap.then::<Bgr8, _>(FullRange).convert(&Rgb8::new(1, 2, 3));
    /// ```
    fn then<Mid, B>(self, next: B) -> Then<Self, B, Mid> {
        Then {
            first: self,
            second: next,
            _mid: PhantomData,
        }
    }
}

impl<T: Sized> ConvertPixelExt for T {}

// ═══════════════════════════════════════════════════════════════════════════════
// Image-level conversion functions
// ═══════════════════════════════════════════════════════════════════════════════

/// Convert an image into a pre-allocated output image using the specified method.
///
/// The output image **must** have the same dimensions as the input image.
///
/// # Panics
/// Panics if `img.size() != out.size()`.
///
/// # Example
/// ```
/// # use fovea::image::Image;
/// # use fovea::pixel::{Mono8, Mono16};
/// # use fovea::transform::{convert_image_into, FullRange};
/// let img: Image<Mono8> = Image::fill(4, 4, Mono8::new(128));
/// let mut out: Image<Mono16> = Image::zero(4, 4);
/// convert_image_into(&img, &mut out, FullRange);
/// ```
pub fn convert_image_into<I, O, T, P, M>(img: &I, out: &mut O, method: M)
where
    I: RasterImage<Pixel = T>,
    O: RasterImageMut<Pixel = P>,
    M: ConvertPixel<T, P>,
{
    assert_eq!(
        img.size(),
        out.size(),
        "convert_image_into: input size {:?} does not match output size {:?}",
        img.size(),
        out.size()
    );

    for y in 0..img.height() {
        let src_row = img.row(y);
        let dst_row = out.row_mut(y);
        for (src, dst) in src_row.iter().zip(dst_row.iter_mut()) {
            *dst = method.convert(src);
        }
    }
}

/// Convert an image, returning a new [`Image`] with the converted pixels.
///
/// # Example
/// ```
/// # use fovea::image::Image;
/// # use fovea::pixel::{Mono8, Mono16};
/// # use fovea::transform::{convert_image, FullRange};
/// let img: Image<Mono8> = Image::fill(4, 4, Mono8::new(128));
/// let out: Image<Mono16> = convert_image(&img, FullRange);
/// ```
#[must_use]
pub fn convert_image<I, T, P, M>(img: &I, method: M) -> Image<P>
where
    I: RasterImage<Pixel = T>,
    P: ZeroablePixel,
    M: ConvertPixel<T, P>,
{
    let mut out = Image::<P>::zero(img.width(), img.height());
    convert_image_into(img, &mut out, method);
    out
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Rectangle;
    use crate::image::{Image, SubView, SubViewMut};
    use crate::pixel::*;
    use std::num::Saturating;

    // ─── Helper: approximate float comparison ───────────────────────────────

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    fn approx_f64(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // FullRange — Mono depth
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn fullrange_mono8_to_mono16_extremes() {
        let a: Mono16 = FullRange.convert(&Mono8::new(0));
        let b: Mono16 = FullRange.convert(&Mono8::new(255));
        assert_eq!(a, Mono16::new(0));
        assert_eq!(b, Mono16::new(65535));
    }

    #[test]
    fn fullrange_mono8_to_mono16_midpoint() {
        let m: Mono16 = FullRange.convert(&Mono8::new(128));
        assert_eq!(m, Mono16::new(32896));
        // 128 * 257 = 32896
    }

    #[test]
    fn fullrange_mono16_to_mono8_extremes() {
        let a: Mono8 = FullRange.convert(&Mono16::new(0));
        let b: Mono8 = FullRange.convert(&Mono16::new(65535));
        assert_eq!(a, Mono8::new(0));
        assert_eq!(b, Mono8::new(255));
    }

    #[test]
    fn fullrange_mono16_to_mono8_midpoint() {
        // 32768 → (32768 + 128) / 257 = 32896 / 257 = 128
        let m: Mono8 = FullRange.convert(&Mono16::new(32768));
        assert_eq!(m, Mono8::new(128));
    }

    #[test]
    fn fullrange_mono8_mono16_roundtrip() {
        for v in 0..=255u8 {
            let m8 = Mono8::new(v);
            let m16: Mono16 = FullRange.convert(&m8);
            let back: Mono8 = FullRange.convert(&m16);
            assert_eq!(back, m8, "roundtrip failed for {v}");
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // FullRange — Mono ↔ f32
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn fullrange_mono8_to_f32() {
        let zero: MonoF32 = FullRange.convert(&Mono8::new(0));
        let max: MonoF32 = FullRange.convert(&Mono8::new(255));
        let mid: MonoF32 = FullRange.convert(&Mono8::new(128));
        assert!(approx(zero.0, 0.0, 1e-6));
        assert!(approx(max.0, 1.0, 1e-6));
        assert!(approx(mid.0, 128.0 / 255.0, 1e-4));
    }

    #[test]
    fn fullrange_f32_to_mono8() {
        let a: Mono8 = FullRange.convert(&MonoF32::new(0.0));
        let b: Mono8 = FullRange.convert(&MonoF32::new(1.0));
        let c: Mono8 = FullRange.convert(&MonoF32::new(0.5));
        assert_eq!(a, Mono8::new(0));
        assert_eq!(b, Mono8::new(255));
        assert_eq!(c, Mono8::new(128));
    }

    #[test]
    fn fullrange_f32_to_mono8_clamps() {
        let a: Mono8 = FullRange.convert(&MonoF32::new(-0.5));
        let b: Mono8 = FullRange.convert(&MonoF32::new(1.5));
        assert_eq!(a, Mono8::new(0));
        assert_eq!(b, Mono8::new(255));
    }

    #[test]
    fn fullrange_mono16_to_f32() {
        let zero: MonoF32 = FullRange.convert(&Mono16::new(0));
        let max: MonoF32 = FullRange.convert(&Mono16::new(65535));
        assert!(approx(zero.0, 0.0, 1e-6));
        assert!(approx(max.0, 1.0, 1e-6));
    }

    #[test]
    fn fullrange_f32_to_mono16() {
        let a: Mono16 = FullRange.convert(&MonoF32::new(0.0));
        let b: Mono16 = FullRange.convert(&MonoF32::new(1.0));
        assert_eq!(a, Mono16::new(0));
        assert_eq!(b, Mono16::new(65535));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // FullRange — Mono<BITS> → Mono8 / Mono16
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn fullrange_mono10_to_mono8() {
        let a: Mono8 = FullRange.convert(&Mono10::new(0));
        let b: Mono8 = FullRange.convert(&Mono10::new(1023));
        assert_eq!(a, Mono8::new(0));
        assert_eq!(b, Mono8::new(255));
        // Midpoint: 512 * 255 / 1023 ≈ 127.6 → 128 with rounding
        let mid: Mono8 = FullRange.convert(&Mono10::new(512));
        assert_eq!(mid, Mono8::new(128));
    }

    #[test]
    fn fullrange_mono12_to_mono8() {
        let a: Mono8 = FullRange.convert(&Mono12::new(0));
        let b: Mono8 = FullRange.convert(&Mono12::new(4095));
        assert_eq!(a, Mono8::new(0));
        assert_eq!(b, Mono8::new(255));
    }

    #[test]
    fn fullrange_mono14_to_mono8() {
        let a: Mono8 = FullRange.convert(&Mono14::new(0));
        let b: Mono8 = FullRange.convert(&Mono14::new(16383));
        assert_eq!(a, Mono8::new(0));
        assert_eq!(b, Mono8::new(255));
    }

    #[test]
    fn fullrange_mono10_to_mono16() {
        let a: Mono16 = FullRange.convert(&Mono10::new(0));
        let b: Mono16 = FullRange.convert(&Mono10::new(1023));
        assert_eq!(a, Mono16::new(0));
        assert_eq!(b, Mono16::new(65535));
    }

    #[test]
    fn fullrange_mono12_to_mono16() {
        let a: Mono16 = FullRange.convert(&Mono12::new(0));
        let b: Mono16 = FullRange.convert(&Mono12::new(4095));
        assert_eq!(a, Mono16::new(0));
        assert_eq!(b, Mono16::new(65535));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // FullRange — Rgb depth
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn fullrange_rgb8_to_rgb16() {
        let src = Rgb8::new(0, 128, 255);
        let dst: Rgb16 = FullRange.convert(&src);
        assert_eq!(dst, Rgb16::new(0, 32896, 65535));
    }

    #[test]
    fn fullrange_rgb16_to_rgb8() {
        let src = Rgb16::new(0, 32896, 65535);
        let dst: Rgb8 = FullRange.convert(&src);
        assert_eq!(dst, Rgb8::new(0, 128, 255));
    }

    #[test]
    fn fullrange_rgb8_rgb16_roundtrip() {
        let src = Rgb8::new(10, 100, 200);
        let wide: Rgb16 = FullRange.convert(&src);
        let back: Rgb8 = FullRange.convert(&wide);
        assert_eq!(back, src);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // FullRange — Rgba depth
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn fullrange_rgba8_to_rgba16() {
        let src = Rgba8::new(0, 128, 255, 64);
        let dst: Rgba16 = FullRange.convert(&src);
        assert_eq!(dst, Rgba16::new(0, 32896, 65535, 16448));
    }

    #[test]
    fn fullrange_rgba16_to_rgba8() {
        let src = Rgba16::new(0, 32896, 65535, 16448);
        let dst: Rgba8 = FullRange.convert(&src);
        assert_eq!(dst, Rgba8::new(0, 128, 255, 64));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // FullRange — Bgr / Bgra depth
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn fullrange_bgr8_to_bgr16() {
        let src = Bgr8::new(10, 20, 30);
        let dst: Bgr16 = FullRange.convert(&src);
        assert_eq!(dst, Bgr16::new(2570, 5140, 7710));
    }

    #[test]
    fn fullrange_bgr16_to_bgr8() {
        let src = Bgr16::new(2570, 5140, 7710);
        let dst: Bgr8 = FullRange.convert(&src);
        assert_eq!(dst, Bgr8::new(10, 20, 30));
    }

    #[test]
    fn fullrange_bgra8_to_bgra16() {
        let src = Bgra8::new(10, 20, 30, 40);
        let dst: Bgra16 = FullRange.convert(&src);
        assert_eq!(dst, Bgra16::new(2570, 5140, 7710, 10280));
    }

    #[test]
    fn fullrange_bgra16_to_bgra8() {
        let src = Bgra16::new(2570, 5140, 7710, 10280);
        let dst: Bgra8 = FullRange.convert(&src);
        assert_eq!(dst, Bgra8::new(10, 20, 30, 40));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // FullRange — Rgb ↔ RgbF32
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn fullrange_rgb8_to_rgbf32() {
        let src = Rgb8::new(0, 128, 255);
        let dst: RgbF32 = FullRange.convert(&src);
        assert!(approx(dst.r, 0.0, 1e-6));
        assert!(approx(dst.g, 128.0 / 255.0, 1e-4));
        assert!(approx(dst.b, 1.0, 1e-6));
    }

    #[test]
    fn fullrange_rgbf32_to_rgb8() {
        let src = RgbF32::new(0.0, 0.5, 1.0);
        let dst: Rgb8 = FullRange.convert(&src);
        assert_eq!(dst, Rgb8::new(0, 128, 255));
    }

    #[test]
    fn fullrange_rgb16_to_rgbf32() {
        let src = Rgb16::new(0, 32768, 65535);
        let dst: RgbF32 = FullRange.convert(&src);
        assert!(approx(dst.r, 0.0, 1e-6));
        assert!(approx(dst.b, 1.0, 1e-6));
    }

    #[test]
    fn fullrange_rgbf32_to_rgb16() {
        let src = RgbF32::new(0.0, 0.5, 1.0);
        let dst: Rgb16 = FullRange.convert(&src);
        assert_eq!(dst.r.0, 0);
        assert_eq!(dst.b.0, 65535);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // FullRange — Bgr ↔ BgrF32
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn fullrange_bgr8_to_bgrf32() {
        let src = Bgr8::new(0, 128, 255);
        let dst: BgrF32 = FullRange.convert(&src);
        assert!(approx(dst.b, 0.0, 1e-6));
        assert!(approx(dst.r, 1.0, 1e-6));
    }

    #[test]
    fn fullrange_bgrf32_to_bgr8() {
        let src = BgrF32::new(0.0, 0.5, 1.0);
        let dst: Bgr8 = FullRange.convert(&src);
        assert_eq!(dst.b.0, 0);
        assert_eq!(dst.r.0, 255);
    }

    #[test]
    fn fullrange_bgr16_to_bgrf32() {
        let src = Bgr16::new(0, 32768, 65535);
        let dst: BgrF32 = FullRange.convert(&src);
        assert!(approx(dst.b, 0.0, 1e-6));
        assert!(approx(dst.r, 1.0, 1e-6));
    }

    #[test]
    fn fullrange_bgrf32_to_bgr16() {
        let src = BgrF32::new(0.0, 0.5, 1.0);
        let dst: Bgr16 = FullRange.convert(&src);
        assert_eq!(dst.b.0, 0);
        assert_eq!(dst.r.0, 65535);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // FullRange — Rgba ↔ RgbaF32 (NEW)
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn fullrange_rgba8_to_rgbaf32() {
        let src = Rgba8::new(0, 128, 255, 64);
        let dst: RgbaF32 = FullRange.convert(&src);
        assert!(approx(dst.r, 0.0, 1e-6));
        assert!(approx(dst.g, 128.0 / 255.0, 1e-4));
        assert!(approx(dst.b, 1.0, 1e-6));
        assert!(approx(dst.a, 64.0 / 255.0, 1e-4));
    }

    #[test]
    fn fullrange_rgbaf32_to_rgba8() {
        let src = RgbaF32::new(0.0, 0.5, 1.0, 0.25);
        let dst: Rgba8 = FullRange.convert(&src);
        assert_eq!(dst, Rgba8::new(0, 128, 255, 64));
    }

    #[test]
    fn fullrange_rgba16_to_rgbaf32() {
        let src = Rgba16::new(0, 32768, 65535, 16384);
        let dst: RgbaF32 = FullRange.convert(&src);
        assert!(approx(dst.r, 0.0, 1e-6));
        assert!(approx(dst.b, 1.0, 1e-6));
    }

    #[test]
    fn fullrange_rgbaf32_to_rgba16() {
        let src = RgbaF32::new(0.0, 0.5, 1.0, 0.0);
        let dst: Rgba16 = FullRange.convert(&src);
        assert_eq!(dst.r.0, 0);
        assert_eq!(dst.b.0, 65535);
        assert_eq!(dst.a.0, 0);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // FullRange — Bgra ↔ BgraF32 (NEW)
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn fullrange_bgra8_to_bgraf32() {
        let src = Bgra8::new(0, 128, 255, 64);
        let dst: BgraF32 = FullRange.convert(&src);
        assert!(approx(dst.b, 0.0, 1e-6));
        assert!(approx(dst.r, 1.0, 1e-6));
        assert!(approx(dst.a, 64.0 / 255.0, 1e-4));
    }

    #[test]
    fn fullrange_bgraf32_to_bgra8() {
        let src = BgraF32::new(0.0, 0.5, 1.0, 0.25);
        let dst: Bgra8 = FullRange.convert(&src);
        assert_eq!(dst.b.0, 0);
        assert_eq!(dst.r.0, 255);
        assert_eq!(dst.a.0, 64);
    }

    #[test]
    fn fullrange_bgra16_to_bgraf32() {
        let src = Bgra16::new(0, 32768, 65535, 16384);
        let dst: BgraF32 = FullRange.convert(&src);
        assert!(approx(dst.b, 0.0, 1e-6));
        assert!(approx(dst.r, 1.0, 1e-6));
    }

    #[test]
    fn fullrange_bgraf32_to_bgra16() {
        let src = BgraF32::new(0.0, 0.5, 1.0, 0.0);
        let dst: Bgra16 = FullRange.convert(&src);
        assert_eq!(dst.b.0, 0);
        assert_eq!(dst.r.0, 65535);
        assert_eq!(dst.a.0, 0);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Narrow — Mono
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn clamp_mono8_to_mono16() {
        let a: Mono16 = Narrow.convert(&Mono8::new(42));
        let b: Mono16 = Narrow.convert(&Mono8::new(0));
        let c: Mono16 = Narrow.convert(&Mono8::new(255));
        assert_eq!(a, Mono16::new(42));
        assert_eq!(b, Mono16::new(0));
        assert_eq!(c, Mono16::new(255));
    }

    #[test]
    fn clamp_mono16_to_mono8_fits() {
        let a: Mono8 = Narrow.convert(&Mono16::new(42));
        let b: Mono8 = Narrow.convert(&Mono16::new(0));
        let c: Mono8 = Narrow.convert(&Mono16::new(255));
        assert_eq!(a, Mono8::new(42));
        assert_eq!(b, Mono8::new(0));
        assert_eq!(c, Mono8::new(255));
    }

    #[test]
    fn clamp_mono16_to_mono8_clamps() {
        let a: Mono8 = Narrow.convert(&Mono16::new(256));
        let b: Mono8 = Narrow.convert(&Mono16::new(1000));
        let c: Mono8 = Narrow.convert(&Mono16::new(65535));
        assert_eq!(a, Mono8::new(255));
        assert_eq!(b, Mono8::new(255));
        assert_eq!(c, Mono8::new(255));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Narrow — Rgb
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn clamp_rgb8_to_rgb16() {
        let src = Rgb8::new(10, 20, 30);
        let dst: Rgb16 = Narrow.convert(&src);
        assert_eq!(dst, Rgb16::new(10, 20, 30));
    }

    #[test]
    fn clamp_rgb16_to_rgb8() {
        let src = Rgb16::new(10, 300, 65535);
        let dst: Rgb8 = Narrow.convert(&src);
        assert_eq!(dst, Rgb8::new(10, 255, 255));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Narrow — Rgba
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn clamp_rgba8_to_rgba16() {
        let src = Rgba8::new(10, 20, 30, 40);
        let dst: Rgba16 = Narrow.convert(&src);
        assert_eq!(dst, Rgba16::new(10, 20, 30, 40));
    }

    #[test]
    fn clamp_rgba16_to_rgba8_clamps() {
        let src = Rgba16::new(0, 256, 65535, 100);
        let dst: Rgba8 = Narrow.convert(&src);
        assert_eq!(dst, Rgba8::new(0, 255, 255, 100));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Narrow — Bgr / Bgra
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn clamp_bgr8_to_bgr16() {
        let src = Bgr8::new(10, 20, 30);
        let dst: Bgr16 = Narrow.convert(&src);
        assert_eq!(dst, Bgr16::new(10, 20, 30));
    }

    #[test]
    fn clamp_bgr16_to_bgr8() {
        let src = Bgr16::new(300, 20, 65535);
        let dst: Bgr8 = Narrow.convert(&src);
        assert_eq!(dst, Bgr8::new(255, 20, 255));
    }

    #[test]
    fn clamp_bgra8_to_bgra16() {
        let src = Bgra8::new(10, 20, 30, 40);
        let dst: Bgra16 = Narrow.convert(&src);
        assert_eq!(dst, Bgra16::new(10, 20, 30, 40));
    }

    #[test]
    fn clamp_bgra16_to_bgra8() {
        let src = Bgra16::new(300, 20, 65535, 0);
        let dst: Bgra8 = Narrow.convert(&src);
        assert_eq!(dst, Bgra8::new(255, 20, 255, 0));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Narrow — Mono<BITS>
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn clamp_mono10_to_mono8() {
        let a: Mono8 = Narrow.convert(&Mono10::new(42));
        let b: Mono8 = Narrow.convert(&Mono10::new(1023));
        assert_eq!(a, Mono8::new(42));
        assert_eq!(b, Mono8::new(255));
    }

    #[test]
    fn clamp_mono12_to_mono8() {
        let a: Mono8 = Narrow.convert(&Mono12::new(200));
        let b: Mono8 = Narrow.convert(&Mono12::new(4095));
        assert_eq!(a, Mono8::new(200));
        assert_eq!(b, Mono8::new(255));
    }

    #[test]
    fn clamp_mono10_to_mono16() {
        let a: Mono16 = Narrow.convert(&Mono10::new(1023));
        assert_eq!(a, Mono16::new(1023));
    }

    #[test]
    fn clamp_mono12_to_mono16() {
        let a: Mono16 = Narrow.convert(&Mono12::new(4095));
        assert_eq!(a, Mono16::new(4095));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Luminance
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn luminance_rgb8_white() {
        let white = Rgb8::new(255, 255, 255);
        // (77*255 + 150*255 + 29*255 + 128) >> 8 = (65280 + 128) >> 8 = 255
        assert_eq!(Luminance.convert(&white), Mono8::new(255));
    }

    #[test]
    fn luminance_rgb8_black() {
        assert_eq!(Luminance.convert(&Rgb8::new(0, 0, 0)), Mono8::new(0));
    }

    #[test]
    fn luminance_rgb8_pure_red() {
        // (77*255 + 150*0 + 29*0 + 128) >> 8 = (19635 + 128) >> 8 = 19763 >> 8 = 77
        let red = Rgb8::new(255, 0, 0);
        assert_eq!(Luminance.convert(&red), Mono8::new(77));
    }

    #[test]
    fn luminance_rgb8_pure_green() {
        // (150*255 + 128) >> 8 = (38250 + 128) >> 8 = 38378 >> 8 = 149
        let green = Rgb8::new(0, 255, 0);
        assert_eq!(Luminance.convert(&green), Mono8::new(149));
    }

    #[test]
    fn luminance_rgb8_pure_blue() {
        // (29*255 + 128) >> 8 = (7395 + 128) >> 8 = 7523 >> 8 = 29
        let blue = Rgb8::new(0, 0, 255);
        assert_eq!(Luminance.convert(&blue), Mono8::new(29));
    }

    #[test]
    fn luminance_bgr8() {
        // BGR stores b,g,r but the luminance formula uses r,g,b semantics
        let bgr = Bgr8::new(0, 0, 255); // b=0, g=0, r=255
        assert_eq!(Luminance.convert(&bgr), Mono8::new(77));
    }

    #[test]
    fn luminance_rgb16_white() {
        let white = Rgb16::new(65535, 65535, 65535);
        assert_eq!(Luminance.convert(&white), Mono16::new(65535));
    }

    #[test]
    fn luminance_bgr16() {
        let bgr = Bgr16::new(0, 0, 65535); // b=0, g=0, r=65535
        let mono: Mono16 = Luminance.convert(&bgr);
        // (77*65535 + 128) >> 8 = (5046195 + 128) >> 8 = 5046323 >> 8 = 19712
        assert_eq!(mono, Mono16::new(19712));
    }

    #[test]
    fn luminance_rgbf32() {
        let white = RgbF32::new(1.0, 1.0, 1.0);
        let y: MonoF32 = Luminance.convert(&white);
        assert!(approx(y.0, 1.0, 1e-4));

        let red = RgbF32::new(1.0, 0.0, 0.0);
        let y: MonoF32 = Luminance.convert(&red);
        assert!(approx(y.0, 0.299, 1e-4));
    }

    #[test]
    fn luminance_bgrf32() {
        let bgr = BgrF32::new(0.0, 0.0, 1.0); // b=0, g=0, r=1
        let y: MonoF32 = Luminance.convert(&bgr);
        assert!(approx(y.0, 0.299, 1e-4));
    }

    // ── Luminance — RGBA / BGRA (NEW) ──────────────────────────────────────

    #[test]
    fn luminance_rgba8_white() {
        let src = Rgba8::new(255, 255, 255, 128);
        let mono: Mono8 = Luminance.convert(&src);
        assert_eq!(mono, Mono8::new(255));
    }

    #[test]
    fn luminance_rgba8_pure_red() {
        let src = Rgba8::new(255, 0, 0, 0);
        let mono: Mono8 = Luminance.convert(&src);
        assert_eq!(mono, Mono8::new(77));
    }

    #[test]
    fn luminance_rgba16() {
        let src = Rgba16::new(65535, 65535, 65535, 0);
        let mono: Mono16 = Luminance.convert(&src);
        assert_eq!(mono, Mono16::new(65535));
    }

    #[test]
    fn luminance_rgbaf32() {
        let src = RgbaF32::new(1.0, 0.0, 0.0, 0.5);
        let y: MonoF32 = Luminance.convert(&src);
        assert!(approx(y.0, 0.299, 1e-4));
    }

    #[test]
    fn luminance_bgra8() {
        // b=0, g=0, r=255, a=42 → luminance based on r,g,b only
        let src = Bgra8::new(0, 0, 255, 42);
        let mono: Mono8 = Luminance.convert(&src);
        assert_eq!(mono, Mono8::new(77));
    }

    #[test]
    fn luminance_bgra16() {
        let src = Bgra16::new(0, 0, 65535, 1000);
        let mono: Mono16 = Luminance.convert(&src);
        assert_eq!(mono, Mono16::new(19712));
    }

    #[test]
    fn luminance_bgraf32() {
        let src = BgraF32::new(0.0, 0.0, 1.0, 0.5);
        let y: MonoF32 = Luminance.convert(&src);
        assert!(approx(y.0, 0.299, 1e-4));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Broadcast
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn broadcast_mono8_to_rgb8() {
        let dst: Rgb8 = Broadcast.convert(&Mono8::new(128));
        assert_eq!(dst, Rgb8::new(128, 128, 128));
    }

    #[test]
    fn broadcast_mono8_to_bgr8() {
        let dst: Bgr8 = Broadcast.convert(&Mono8::new(42));
        assert_eq!(dst, Bgr8::new(42, 42, 42));
    }

    #[test]
    fn broadcast_mono16_to_rgb16() {
        let dst: Rgb16 = Broadcast.convert(&Mono16::new(1000));
        assert_eq!(dst, Rgb16::new(1000, 1000, 1000));
    }

    #[test]
    fn broadcast_mono16_to_bgr16() {
        let dst: Bgr16 = Broadcast.convert(&Mono16::new(500));
        assert_eq!(dst, Bgr16::new(500, 500, 500));
    }

    #[test]
    fn broadcast_f32_to_rgbf32() {
        let dst: RgbF32 = Broadcast.convert(&MonoF32::new(0.5));
        assert_eq!(dst, RgbF32::new(0.5, 0.5, 0.5));
    }

    #[test]
    fn broadcast_f32_to_bgrf32() {
        let dst: BgrF32 = Broadcast.convert(&MonoF32::new(0.25));
        assert_eq!(dst, BgrF32::new(0.25, 0.25, 0.25));
    }

    #[test]
    fn broadcast_mono8_extremes() {
        let a: Rgb8 = Broadcast.convert(&Mono8::new(0));
        let b: Rgb8 = Broadcast.convert(&Mono8::new(255));
        assert_eq!(a, Rgb8::new(0, 0, 0));
        assert_eq!(b, Rgb8::new(255, 255, 255));
    }

    // ── Broadcast — RGBA / BGRA (NEW) ──────────────────────────────────────

    #[test]
    fn broadcast_mono8_to_rgba8() {
        let dst: Rgba8 = Broadcast.convert(&Mono8::new(128));
        assert_eq!(dst, Rgba8::new(128, 128, 128, 255));
    }

    #[test]
    fn broadcast_mono8_to_bgra8() {
        let dst: Bgra8 = Broadcast.convert(&Mono8::new(42));
        assert_eq!(dst, Bgra8::new(42, 42, 42, 255));
    }

    #[test]
    fn broadcast_mono16_to_rgba16() {
        let dst: Rgba16 = Broadcast.convert(&Mono16::new(1000));
        assert_eq!(dst, Rgba16::new(1000, 1000, 1000, 65535));
    }

    #[test]
    fn broadcast_mono16_to_bgra16() {
        let dst: Bgra16 = Broadcast.convert(&Mono16::new(500));
        assert_eq!(dst, Bgra16::new(500, 500, 500, 65535));
    }

    #[test]
    fn broadcast_f32_to_rgbaf32() {
        let dst: RgbaF32 = Broadcast.convert(&MonoF32::new(0.5));
        assert_eq!(dst, RgbaF32::new(0.5, 0.5, 0.5, 1.0));
    }

    #[test]
    fn broadcast_f32_to_bgraf32() {
        let dst: BgraF32 = Broadcast.convert(&MonoF32::new(0.25));
        assert_eq!(dst, BgraF32::new(0.25, 0.25, 0.25, 1.0));
    }

    #[test]
    fn broadcast_mono8_to_rgba8_extremes() {
        let a: Rgba8 = Broadcast.convert(&Mono8::new(0));
        let b: Rgba8 = Broadcast.convert(&Mono8::new(255));
        assert_eq!(a, Rgba8::new(0, 0, 0, 255));
        assert_eq!(b, Rgba8::new(255, 255, 255, 255));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Image-level conversions
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn convert_image_mono8_to_mono16_fullrange() {
        let img = Image::fill(3, 2, Mono8::new(128));
        let out: Image<Mono16> = convert_image(&img, FullRange);
        assert_eq!(out.width(), 3);
        assert_eq!(out.height(), 2);
        for y in 0..2 {
            for x in 0..3 {
                assert_eq!(out.pixel_at(x, y), Mono16::new(32896));
            }
        }
    }

    #[test]
    fn convert_image_into_rgb8_to_mono8_luminance() {
        let img = Image::fill(2, 2, Rgb8::new(255, 255, 255));
        let mut out = Image::<Mono8>::zero(2, 2);
        convert_image_into(&img, &mut out, Luminance);
        for y in 0..2 {
            for x in 0..2 {
                assert_eq!(out.pixel_at(x, y), Mono8::new(255));
            }
        }
    }

    #[test]
    fn convert_image_mono8_to_rgb8_broadcast() {
        let img = Image::fill(2, 3, Mono8::new(100));
        let out: Image<Rgb8> = convert_image(&img, Broadcast);
        assert_eq!(out.width(), 2);
        assert_eq!(out.height(), 3);
        for y in 0..3 {
            for x in 0..2 {
                assert_eq!(out.pixel_at(x, y), Rgb8::new(100, 100, 100));
            }
        }
    }

    #[test]
    fn convert_image_preserves_varying_pixels() {
        let img = Image::generate(4, 4, |x, y| Mono8::new((x * 64 + y * 16) as u8));
        let out: Image<Mono16> = convert_image(&img, FullRange);
        for y in 0..4 {
            for x in 0..4 {
                let expected_u8 = (x * 64 + y * 16) as u8;
                let expected_u16 = fr_u8_to_u16(expected_u8);
                assert_eq!(out.pixel_at(x, y), Mono16::new(expected_u16));
            }
        }
    }

    #[test]
    #[should_panic(expected = "does not match")]
    fn convert_image_into_panics_on_size_mismatch() {
        let img = Image::fill(3, 3, Mono8::new(0));
        let mut out = Image::<Mono16>::zero(2, 2);
        convert_image_into(&img, &mut out, FullRange);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // ROI-as-output tests — writing converted pixels into a mutable ROI
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn convert_image_into_roi_output_mono8_to_mono16() {
        let src: Image<Mono8> = Image::fill(2, 2, Mono8::new(128));
        let mut target: Image<Mono16> = Image::zero(4, 4);

        {
            let mut roi_out = target.roi_mut(Rectangle::new((1, 1), (2, 2))).unwrap();
            convert_image_into(&src, &mut roi_out, FullRange);
        }

        // Inside the ROI: converted values
        let expected = fr_u8_to_u16(128);
        assert_eq!(target.get(1, 1).unwrap(), Mono16::new(expected));
        assert_eq!(target.get(2, 1).unwrap(), Mono16::new(expected));
        assert_eq!(target.get(1, 2).unwrap(), Mono16::new(expected));
        assert_eq!(target.get(2, 2).unwrap(), Mono16::new(expected));

        // Outside the ROI: untouched zeros
        assert_eq!(target.get(0, 0).unwrap(), Mono16::new(0));
        assert_eq!(target.get(3, 3).unwrap(), Mono16::new(0));
        assert_eq!(target.get(0, 1).unwrap(), Mono16::new(0));
        assert_eq!(target.get(3, 0).unwrap(), Mono16::new(0));
    }

    #[test]
    fn convert_image_into_roi_input_to_roi_output() {
        // Read from an ROI of the source, write into an ROI of the target
        let src: Image<Mono8> = Image::generate(4, 4, |x, y| Mono8::new((y * 10 + x) as u8));
        let roi_in = src.roi(Rectangle::new((1, 1), (2, 2))).unwrap();
        // roi_in contains: 11, 12, 21, 22

        let mut target: Image<Mono16> = Image::zero(4, 4);
        {
            let mut roi_out = target.roi_mut(Rectangle::new((2, 2), (2, 2))).unwrap();
            convert_image_into(&roi_in, &mut roi_out, FullRange);
        }

        assert_eq!(target.get(2, 2).unwrap(), Mono16::new(fr_u8_to_u16(11)));
        assert_eq!(target.get(3, 2).unwrap(), Mono16::new(fr_u8_to_u16(12)));
        assert_eq!(target.get(2, 3).unwrap(), Mono16::new(fr_u8_to_u16(21)));
        assert_eq!(target.get(3, 3).unwrap(), Mono16::new(fr_u8_to_u16(22)));

        // Outside the ROI: untouched
        assert_eq!(target.get(0, 0).unwrap(), Mono16::new(0));
        assert_eq!(target.get(1, 1).unwrap(), Mono16::new(0));
    }

    #[test]
    fn convert_image_into_roi_output_rgb8_to_bgr8_colorswap() {
        let src: Image<Rgb8> = Image::fill(2, 2, Rgb8::new(10, 20, 30));
        let mut target: Image<Bgr8> = Image::zero(4, 4);

        {
            let mut roi_out = target.roi_mut(Rectangle::new((0, 0), (2, 2))).unwrap();
            convert_image_into(&src, &mut roi_out, ColorSwap);
        }

        assert_eq!(target.get(0, 0).unwrap(), Bgr8::new(30, 20, 10));
        assert_eq!(target.get(1, 1).unwrap(), Bgr8::new(30, 20, 10));
        // Outside
        assert_eq!(target.get(2, 0).unwrap(), Bgr8::new(0, 0, 0));
    }

    #[test]
    fn convert_image_into_roi_output_luminance() {
        let white = Rgb8::new(255, 255, 255);
        let src: Image<Rgb8> = Image::fill(2, 2, white);
        let mut target: Image<Mono8> = Image::zero(4, 4);

        {
            let mut roi_out = target.roi_mut(Rectangle::new((1, 0), (2, 2))).unwrap();
            convert_image_into(&src, &mut roi_out, Luminance);
        }

        // White → lum = 255
        assert_eq!(target.get(1, 0).unwrap(), Mono8::new(255));
        assert_eq!(target.get(2, 1).unwrap(), Mono8::new(255));
        // Outside
        assert_eq!(target.get(0, 0).unwrap(), Mono8::new(0));
        assert_eq!(target.get(3, 0).unwrap(), Mono8::new(0));
    }

    #[test]
    #[should_panic(expected = "does not match")]
    fn convert_image_into_roi_output_panics_on_size_mismatch() {
        let src: Image<Mono8> = Image::fill(3, 3, Mono8::new(0));
        let mut target: Image<Mono16> = Image::zero(4, 4);
        let mut roi_out = target.roi_mut(Rectangle::new((0, 0), (2, 2))).unwrap();
        // src is 3x3 but roi_out is 2x2 — should panic
        convert_image_into(&src, &mut roi_out, FullRange);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Composition: verify that strategies compose naturally
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn compose_luminance_then_fullrange() {
        // Rgb8 → Mono8 via Luminance, then Mono8 → Mono16 via FullRange
        let src = Rgb8::new(200, 100, 50);
        let gray: Mono8 = Luminance.convert(&src);
        let wide: Mono16 = FullRange.convert(&gray);
        // Verify it's the same as doing the math manually
        let expected_y = lum_u8(200, 100, 50);
        let expected_16 = fr_u8_to_u16(expected_y);
        assert_eq!(wide, Mono16::new(expected_16));
    }

    #[test]
    fn compose_broadcast_then_fullrange() {
        // Mono8 → Rgb8 via Broadcast, then Rgb8 → Rgb16 via FullRange
        let src = Mono8::new(100);
        let color: Rgb8 = Broadcast.convert(&src);
        let wide: Rgb16 = FullRange.convert(&color);
        assert_eq!(wide, Rgb16::new(25700, 25700, 25700));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Internal helper tests
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn helper_fr_u8_to_u16_properties() {
        assert_eq!(fr_u8_to_u16(0), 0);
        assert_eq!(fr_u8_to_u16(1), 257);
        assert_eq!(fr_u8_to_u16(127), 32639);
        assert_eq!(fr_u8_to_u16(128), 32896);
        assert_eq!(fr_u8_to_u16(255), 65535);
    }

    #[test]
    fn helper_fr_u16_to_u8_properties() {
        assert_eq!(fr_u16_to_u8(0), 0);
        assert_eq!(fr_u16_to_u8(257), 1);
        assert_eq!(fr_u16_to_u8(32896), 128);
        assert_eq!(fr_u16_to_u8(65535), 255);
    }

    #[test]
    fn helper_fr_roundtrip_exhaustive_u8() {
        for v in 0..=255u8 {
            let wide = fr_u8_to_u16(v);
            let back = fr_u16_to_u8(wide);
            assert_eq!(back, v, "roundtrip failed for u8 value {v}");
        }
    }

    #[test]
    fn helper_lum_coefficients_sum_to_256() {
        // Ensure the integer coefficients sum correctly
        assert_eq!(77 + 150 + 29, 256);
    }

    #[test]
    fn helper_clamp_u16_to_u8_values() {
        assert_eq!(clamp_u16_to_u8(0), 0);
        assert_eq!(clamp_u16_to_u8(100), 100);
        assert_eq!(clamp_u16_to_u8(255), 255);
        assert_eq!(clamp_u16_to_u8(256), 255);
        assert_eq!(clamp_u16_to_u8(65535), 255);
    }

    #[test]
    fn helper_fr_f32_u8_roundtrip() {
        for v in 0..=255u8 {
            let f = fr_u8_to_f32(v);
            let back = fr_f32_to_u8(f);
            assert_eq!(back, v, "f32 roundtrip failed for u8 value {v}");
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // ColorSwap tests
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn colorswap_rgb8_to_bgr8() {
        let src = Rgb8::new(10, 20, 30);
        let dst: Bgr8 = ColorSwap.convert(&src);
        assert_eq!(dst, Bgr8::new(30, 20, 10));
    }

    #[test]
    fn colorswap_bgr8_to_rgb8() {
        let src = Bgr8::new(30, 20, 10);
        let dst: Rgb8 = ColorSwap.convert(&src);
        assert_eq!(dst, Rgb8::new(10, 20, 30));
    }

    #[test]
    fn colorswap_rgb16_to_bgr16() {
        let src = Rgb16::new(1000, 2000, 3000);
        let dst: Bgr16 = ColorSwap.convert(&src);
        assert_eq!(dst, Bgr16::new(3000, 2000, 1000));
    }

    #[test]
    fn colorswap_bgr16_to_rgb16() {
        let src = Bgr16::new(3000, 2000, 1000);
        let dst: Rgb16 = ColorSwap.convert(&src);
        assert_eq!(dst, Rgb16::new(1000, 2000, 3000));
    }

    #[test]
    fn colorswap_rgbf32_to_bgrf32() {
        let src = RgbF32::new(0.1, 0.2, 0.3);
        let dst: BgrF32 = ColorSwap.convert(&src);
        assert_eq!(dst, BgrF32::new(0.3, 0.2, 0.1));
    }

    #[test]
    fn colorswap_bgrf32_to_rgbf32() {
        let src = BgrF32::new(0.3, 0.2, 0.1);
        let dst: RgbF32 = ColorSwap.convert(&src);
        assert_eq!(dst, RgbF32::new(0.1, 0.2, 0.3));
    }

    #[test]
    fn colorswap_rgba8_to_bgra8() {
        let src = Rgba8::new(10, 20, 30, 40);
        let dst: Bgra8 = ColorSwap.convert(&src);
        assert_eq!(dst, Bgra8::new(30, 20, 10, 40));
    }

    #[test]
    fn colorswap_bgra8_to_rgba8() {
        let src = Bgra8::new(30, 20, 10, 40);
        let dst: Rgba8 = ColorSwap.convert(&src);
        assert_eq!(dst, Rgba8::new(10, 20, 30, 40));
    }

    #[test]
    fn colorswap_rgba16_to_bgra16() {
        let src = Rgba16::new(1000, 2000, 3000, 4000);
        let dst: Bgra16 = ColorSwap.convert(&src);
        assert_eq!(dst, Bgra16::new(3000, 2000, 1000, 4000));
    }

    #[test]
    fn colorswap_bgra16_to_rgba16() {
        let src = Bgra16::new(3000, 2000, 1000, 4000);
        let dst: Rgba16 = ColorSwap.convert(&src);
        assert_eq!(dst, Rgba16::new(1000, 2000, 3000, 4000));
    }

    #[test]
    fn colorswap_rgbaf32_to_bgraf32() {
        let src = RgbaF32::new(0.1, 0.2, 0.3, 0.4);
        let dst: BgraF32 = ColorSwap.convert(&src);
        assert_eq!(dst, BgraF32::new(0.3, 0.2, 0.1, 0.4));
    }

    #[test]
    fn colorswap_bgraf32_to_rgbaf32() {
        let src = BgraF32::new(0.3, 0.2, 0.1, 0.4);
        let dst: RgbaF32 = ColorSwap.convert(&src);
        assert_eq!(dst, RgbaF32::new(0.1, 0.2, 0.3, 0.4));
    }

    #[test]
    fn colorswap_roundtrip_rgb8() {
        let src = Rgb8::new(42, 137, 200);
        let swapped: Bgr8 = ColorSwap.convert(&src);
        let back: Rgb8 = ColorSwap.convert(&swapped);
        assert_eq!(back, src);
    }

    #[test]
    fn colorswap_roundtrip_rgba16() {
        let src = Rgba16::new(100, 200, 300, 400);
        let swapped: Bgra16 = ColorSwap.convert(&src);
        let back: Rgba16 = ColorSwap.convert(&swapped);
        assert_eq!(back, src);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // AddAlpha tests
    // ═══════════════════════════════════════════════════════════════════════════
    #[test]
    fn addalpha_rgb8_to_rgba8() {
        let src = Rgb8::new(10, 20, 30);
        let dst: Rgba8 = AddAlpha.convert(&src);
        assert_eq!(dst, Rgba8::new(10, 20, 30, 255));
    }

    #[test]
    fn addalpha_rgb16_to_rgba16() {
        let src = Rgb16::new(1000, 2000, 3000);
        let dst: Rgba16 = AddAlpha.convert(&src);
        assert_eq!(dst, Rgba16::new(1000, 2000, 3000, 65535));
    }

    #[test]
    fn addalpha_rgbf32_to_rgbaf32() {
        let src = RgbF32::new(0.1, 0.2, 0.3);
        let dst: RgbaF32 = AddAlpha.convert(&src);
        assert_eq!(dst, RgbaF32::new(0.1, 0.2, 0.3, 1.0));
    }

    #[test]
    fn addalpha_bgr8_to_bgra8() {
        let src = Bgr8::new(30, 20, 10);
        let dst: Bgra8 = AddAlpha.convert(&src);
        assert_eq!(dst, Bgra8::new(30, 20, 10, 255));
    }

    #[test]
    fn addalpha_bgr16_to_bgra16() {
        let src = Bgr16::new(3000, 2000, 1000);
        let dst: Bgra16 = AddAlpha.convert(&src);
        assert_eq!(dst, Bgra16::new(3000, 2000, 1000, 65535));
    }

    #[test]
    fn addalpha_bgrf32_to_bgraf32() {
        let src = BgrF32::new(0.3, 0.2, 0.1);
        let dst: BgraF32 = AddAlpha.convert(&src);
        assert_eq!(dst, BgraF32::new(0.3, 0.2, 0.1, 1.0));
    }

    #[test]
    fn addalpha_rgb8_black() {
        let src = Rgb8::new(0, 0, 0);
        let dst: Rgba8 = AddAlpha.convert(&src);
        assert_eq!(dst, Rgba8::new(0, 0, 0, 255));
    }

    #[test]
    fn addalpha_rgb8_white() {
        let src = Rgb8::new(255, 255, 255);
        let dst: Rgba8 = AddAlpha.convert(&src);
        assert_eq!(dst, Rgba8::new(255, 255, 255, 255));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Image-level ColorSwap tests
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn convert_image_rgb8_to_bgr8_colorswap() {
        let img = Image::fill(3, 2, Rgb8::new(10, 20, 30));
        let out: Image<Bgr8> = convert_image(&img, ColorSwap);
        assert_eq!(out.width(), 3);
        assert_eq!(out.height(), 2);
        for y in 0..2 {
            for x in 0..3 {
                assert_eq!(out.pixel_at(x, y), Bgr8::new(30, 20, 10));
            }
        }
    }

    #[test]
    fn convert_image_into_bgr8_to_rgb8_colorswap() {
        let img = Image::fill(2, 2, Bgr8::new(50, 100, 200));
        let mut out = Image::<Rgb8>::zero(2, 2);
        convert_image_into(&img, &mut out, ColorSwap);
        for y in 0..2 {
            for x in 0..2 {
                assert_eq!(out.pixel_at(x, y), Rgb8::new(200, 100, 50));
            }
        }
    }

    #[test]
    fn convert_image_rgba8_to_bgra8_colorswap() {
        let img = Image::fill(4, 3, Rgba8::new(10, 20, 30, 40));
        let out: Image<Bgra8> = convert_image(&img, ColorSwap);
        assert_eq!(out.width(), 4);
        assert_eq!(out.height(), 3);
        for y in 0..3 {
            for x in 0..4 {
                assert_eq!(out.pixel_at(x, y), Bgra8::new(30, 20, 10, 40));
            }
        }
    }

    #[test]
    fn convert_image_colorswap_preserves_varying_pixels() {
        let img = Image::generate(4, 4, |x, y| {
            Rgb8::new((x * 60) as u8, (y * 40) as u8, (x * 20 + y * 10) as u8)
        });
        let out: Image<Bgr8> = convert_image(&img, ColorSwap);
        for y in 0..4 {
            for x in 0..4 {
                let src = img.pixel_at(x, y);
                let dst = out.pixel_at(x, y);
                assert_eq!(dst.r.0, src.r.0);
                assert_eq!(dst.g.0, src.g.0);
                assert_eq!(dst.b.0, src.b.0);
            }
        }
    }

    #[test]
    fn convert_image_colorswap_roundtrip() {
        let img = Image::generate(3, 3, |x, y| {
            Rgb8::new((x * 80) as u8, (y * 50) as u8, ((x + y) * 30) as u8)
        });
        let swapped: Image<Bgr8> = convert_image(&img, ColorSwap);
        let back: Image<Rgb8> = convert_image(&swapped, ColorSwap);
        for y in 0..3 {
            for x in 0..3 {
                assert_eq!(back.pixel_at(x, y), img.pixel_at(x, y));
            }
        }
    }

    #[test]
    fn convert_image_colorswap_f32() {
        let img = Image::fill(2, 2, RgbF32::new(0.1, 0.2, 0.3));
        let out: Image<BgrF32> = convert_image(&img, ColorSwap);
        for y in 0..2 {
            for x in 0..2 {
                assert_eq!(out.pixel_at(x, y), BgrF32::new(0.3, 0.2, 0.1));
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Image-level AddAlpha tests
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn convert_image_rgb8_to_rgba8_addalpha() {
        let img = Image::fill(3, 2, Rgb8::new(10, 20, 30));
        let out: Image<Rgba8> = convert_image(&img, AddAlpha);
        assert_eq!(out.width(), 3);
        assert_eq!(out.height(), 2);
        for y in 0..2 {
            for x in 0..3 {
                assert_eq!(out.pixel_at(x, y), Rgba8::new(10, 20, 30, 255));
            }
        }
    }

    #[test]
    fn convert_image_into_bgr8_to_bgra8_addalpha() {
        let img = Image::fill(2, 2, Bgr8::new(50, 100, 200));
        let mut out = Image::<Bgra8>::zero(2, 2);
        convert_image_into(&img, &mut out, AddAlpha);
        for y in 0..2 {
            for x in 0..2 {
                assert_eq!(out.pixel_at(x, y), Bgra8::new(50, 100, 200, 255));
            }
        }
    }

    #[test]
    fn convert_image_rgb16_to_rgba16_addalpha() {
        let img = Image::fill(2, 3, Rgb16::new(1000, 2000, 3000));
        let out: Image<Rgba16> = convert_image(&img, AddAlpha);
        assert_eq!(out.width(), 2);
        assert_eq!(out.height(), 3);
        for y in 0..3 {
            for x in 0..2 {
                assert_eq!(out.pixel_at(x, y), Rgba16::new(1000, 2000, 3000, 65535));
            }
        }
    }

    #[test]
    fn convert_image_addalpha_preserves_varying_pixels() {
        let img = Image::generate(4, 4, |x, y| {
            Rgb8::new((x * 60) as u8, (y * 40) as u8, (x * 20 + y * 10) as u8)
        });
        let out: Image<Rgba8> = convert_image(&img, AddAlpha);
        for y in 0..4 {
            for x in 0..4 {
                let src = img.pixel_at(x, y);
                let dst = out.pixel_at(x, y);
                assert_eq!(dst.r.0, src.r.0);
                assert_eq!(dst.g.0, src.g.0);
                assert_eq!(dst.b.0, src.b.0);
                assert_eq!(dst.a.0, 255);
            }
        }
    }

    #[test]
    fn convert_image_addalpha_f32() {
        let img = Image::fill(2, 2, RgbF32::new(0.1, 0.2, 0.3));
        let out: Image<RgbaF32> = convert_image(&img, AddAlpha);
        for y in 0..2 {
            for x in 0..2 {
                assert_eq!(out.pixel_at(x, y), RgbaF32::new(0.1, 0.2, 0.3, 1.0));
            }
        }
    }

    #[test]
    fn compose_colorswap_then_addalpha_image() {
        // Rgb8 → Bgr8 via ColorSwap, then Bgr8 → Bgra8 via AddAlpha
        let img = Image::fill(2, 2, Rgb8::new(10, 20, 30));
        let swapped: Image<Bgr8> = convert_image(&img, ColorSwap);
        let out: Image<Bgra8> = convert_image(&swapped, AddAlpha);
        for y in 0..2 {
            for x in 0..2 {
                assert_eq!(out.pixel_at(x, y), Bgra8::new(30, 20, 10, 255));
            }
        }
    }

    #[test]
    fn compose_addalpha_then_colorswap_image() {
        // Rgb8 → Rgba8 via AddAlpha, then Rgba8 → Bgra8 via ColorSwap
        let img = Image::fill(2, 2, Rgb8::new(10, 20, 30));
        let with_alpha: Image<Rgba8> = convert_image(&img, AddAlpha);
        let out: Image<Bgra8> = convert_image(&with_alpha, ColorSwap);
        for y in 0..2 {
            for x in 0..2 {
                assert_eq!(out.pixel_at(x, y), Bgra8::new(30, 20, 10, 255));
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // PixelMap tests
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn pixelmap_mono8_invert() {
        let src = Mono8::new(200);
        let dst: Mono8 = PixelMap(|s: &Mono8| Mono8::new(255 - s.as_bytes()[0])).convert(&src);
        assert_eq!(dst, Mono8::new(55));
    }

    #[test]
    fn pixelmap_rgb8_to_mono8_max_channel() {
        let src = Rgb8::new(10, 200, 50);
        let dst: Mono8 = PixelMap(|s: &Rgb8| Mono8::new(s.r.0.max(s.g.0).max(s.b.0))).convert(&src);
        assert_eq!(dst, Mono8::new(200));
    }

    #[test]
    fn pixelmap_f32_to_mono8_threshold() {
        let threshold = PixelMap(|s: &f32| {
            if *s > 0.5 {
                Mono8::new(255)
            } else {
                Mono8::new(0)
            }
        });
        let above: Mono8 = threshold.convert(&0.8f32);
        let below: Mono8 = threshold.convert(&0.2f32);
        assert_eq!(above, Mono8::new(255));
        assert_eq!(below, Mono8::new(0));
    }

    #[test]
    fn pixelmap_convert_image() {
        let img = Image::fill(3, 2, Mono8::new(200));
        let out: Image<Mono8> = convert_image(
            &img,
            PixelMap(|s: &Mono8| Mono8::new(255 - s.as_bytes()[0])),
        );
        assert_eq!(out.width(), 3);
        assert_eq!(out.height(), 2);
        for y in 0..2 {
            for x in 0..3 {
                assert_eq!(out.pixel_at(x, y), Mono8::new(55));
            }
        }
    }

    #[test]
    fn pixelmap_convert_image_into() {
        let img = Image::fill(2, 2, Rgb8::new(100, 150, 200));
        let mut out = Image::<Mono8>::zero(2, 2);
        convert_image_into(
            &img,
            &mut out,
            PixelMap(|s: &Rgb8| Mono8::new(s.r.0.max(s.g.0).max(s.b.0))),
        );
        for y in 0..2 {
            for x in 0..2 {
                assert_eq!(out.pixel_at(x, y), Mono8::new(200));
            }
        }
    }

    #[test]
    fn pixelmap_with_captured_state() {
        let offset = 42u8;
        let img = Image::fill(2, 2, Mono8::new(100));
        let out: Image<Mono8> = convert_image(
            &img,
            PixelMap(|s: &Mono8| Mono8::new(s.as_bytes()[0].wrapping_add(offset))),
        );
        for y in 0..2 {
            for x in 0..2 {
                assert_eq!(out.pixel_at(x, y), Mono8::new(142));
            }
        }
    }

    #[test]
    fn pixelmap_cross_type_conversion() {
        // Simulate the user's example: custom type → MonoF32 via closure.
        // We use Mono16 as a stand-in for "ComplexPixel".
        //
        // The output type is `MonoF32` (the named pixel wrapper over
        // `f32`), not raw `f32` — `f32` flowing through `convert_image`
        // in a pixel-semantic context is a pixel role and must be named.
        let img = Image::fill(2, 2, Mono16::new(1000));
        let out: Image<MonoF32> = convert_image(
            &img,
            PixelMap(|s: &Mono16| {
                let b = s.as_bytes();
                let val = u16::from_ne_bytes([b[0], b[1]]);
                MonoF32(val as f32 / 65535.0)
            }),
        );
        for y in 0..2 {
            for x in 0..2 {
                assert!(approx(out.pixel_at(x, y).0, 1000.0 / 65535.0, 1e-6));
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Then combinator — pixel-level tests
    // ═══════════════════════════════════════════════════════════════════════════

    // ── Cross-depth colour swap ─────────────────────────────────────────────

    #[test]
    fn then_colorswap_fullrange_rgb8_to_bgr16() {
        // Rgb8 → Bgr8 → Bgr16
        let src = Rgb8::new(200, 100, 50);
        let result: Bgr16 = ColorSwap.then::<Bgr8, _>(FullRange).convert(&src);
        // ColorSwap: Rgb8(200,100,50) → Bgr8(50,100,200)
        // FullRange: Bgr8(50,100,200) → Bgr16(12850,25700,51400)
        assert_eq!(
            result,
            Bgr16::new(fr_u8_to_u16(50), fr_u8_to_u16(100), fr_u8_to_u16(200))
        );
    }

    #[test]
    fn then_fullrange_colorswap_rgb8_to_bgr16() {
        // Rgb8 → Rgb16 → Bgr16  (depth first, then swap)
        let src = Rgb8::new(200, 100, 50);
        let result: Bgr16 = FullRange.then::<Rgb16, _>(ColorSwap).convert(&src);
        assert_eq!(
            result,
            Bgr16::new(fr_u8_to_u16(50), fr_u8_to_u16(100), fr_u8_to_u16(200))
        );
    }

    #[test]
    fn then_colorswap_fullrange_bgr16_to_rgb8() {
        // Bgr16 → Rgb16 → Rgb8
        let src = Bgr16::new(12850, 25700, 51400);
        let result: Rgb8 = ColorSwap.then::<Rgb16, _>(FullRange).convert(&src);
        assert_eq!(
            result,
            Rgb8::new(
                fr_u16_to_u8(51400),
                fr_u16_to_u8(25700),
                fr_u16_to_u8(12850)
            )
        );
    }

    #[test]
    fn then_colorswap_fullrange_rgba8_to_bgra16() {
        // Rgba8 → Bgra8 → Bgra16
        let src = Rgba8::new(10, 20, 30, 255);
        let result: Bgra16 = ColorSwap.then::<Bgra8, _>(FullRange).convert(&src);
        assert_eq!(
            result,
            Bgra16::new(
                fr_u8_to_u16(30),
                fr_u8_to_u16(20),
                fr_u8_to_u16(10),
                fr_u8_to_u16(255)
            )
        );
    }

    #[test]
    fn then_colorswap_fullrange_rgb8_to_bgrf32() {
        // Rgb8 → Bgr8 → BgrF32
        let src = Rgb8::new(255, 128, 0);
        let result: BgrF32 = ColorSwap.then::<Bgr8, _>(FullRange).convert(&src);
        assert!(approx(result.b, 0.0, 1e-3));
        assert!(approx(result.g, 128.0 / 255.0, 1e-3));
        assert!(approx(result.r, 1.0, 1e-3));
    }

    // ── Cross-depth add alpha ───────────────────────────────────────────────

    #[test]
    fn then_addalpha_fullrange_rgb8_to_rgba16() {
        // Rgb8 → Rgba8 → Rgba16
        let src = Rgb8::new(10, 20, 30);
        let result: Rgba16 = AddAlpha.then::<Rgba8, _>(FullRange).convert(&src);
        assert_eq!(
            result,
            Rgba16::new(fr_u8_to_u16(10), fr_u8_to_u16(20), fr_u8_to_u16(30), 65535)
        );
    }

    #[test]
    fn then_fullrange_addalpha_rgb8_to_rgba16() {
        // Rgb8 → Rgb16 → Rgba16  (depth first, then add alpha)
        let src = Rgb8::new(10, 20, 30);
        let result: Rgba16 = FullRange.then::<Rgb16, _>(AddAlpha).convert(&src);
        assert_eq!(
            result,
            Rgba16::new(fr_u8_to_u16(10), fr_u8_to_u16(20), fr_u8_to_u16(30), 65535)
        );
    }

    #[test]
    fn then_addalpha_fullrange_bgr8_to_bgra16() {
        // Bgr8 → Bgra8 → Bgra16
        let src = Bgr8::new(50, 100, 200);
        let result: Bgra16 = AddAlpha.then::<Bgra8, _>(FullRange).convert(&src);
        assert_eq!(
            result,
            Bgra16::new(
                fr_u8_to_u16(50),
                fr_u8_to_u16(100),
                fr_u8_to_u16(200),
                65535
            )
        );
    }

    #[test]
    fn then_addalpha_fullrange_rgb8_to_rgbaf32() {
        // Rgb8 → Rgba8 → RgbaF32
        let src = Rgb8::new(255, 128, 0);
        let result: RgbaF32 = AddAlpha.then::<Rgba8, _>(FullRange).convert(&src);
        assert!(approx(result.r, 1.0, 1e-3));
        assert!(approx(result.g, 128.0 / 255.0, 1e-3));
        assert!(approx(result.b, 0.0, 1e-3));
        assert!(approx(result.a, 1.0, 1e-3));
    }

    // ── Cross-depth luminance ───────────────────────────────────────────────

    #[test]
    fn then_luminance_fullrange_rgb16_to_mono8() {
        // Rgb16 → Mono16 → Mono8
        let src = Rgb16::new(65535, 65535, 65535);
        let result: Mono8 = Luminance.then::<Mono16, _>(FullRange).convert(&src);
        assert_eq!(result, Mono8::new(255));
    }

    #[test]
    fn then_luminance_fullrange_rgb8_to_mono16() {
        // Rgb8 → Mono8 → Mono16
        let src = Rgb8::new(200, 100, 50);
        let result: Mono16 = Luminance.then::<Mono8, _>(FullRange).convert(&src);
        let expected_y = lum_u8(200, 100, 50);
        assert_eq!(result, Mono16::new(fr_u8_to_u16(expected_y)));
    }

    #[test]
    fn then_luminance_fullrange_rgb8_to_f32() {
        // Rgb8 → Mono8 → MonoF32
        let src = Rgb8::new(255, 255, 255);
        let result: MonoF32 = Luminance.then::<Mono8, _>(FullRange).convert(&src);
        assert!(approx(result.0, 1.0, 1e-3));
    }

    #[test]
    fn then_luminance_fullrange_bgr16_to_mono8() {
        // Bgr16 → Mono16 → Mono8
        let src = Bgr16::new(0, 0, 0);
        let result: Mono8 = Luminance.then::<Mono16, _>(FullRange).convert(&src);
        assert_eq!(result, Mono8::new(0));
    }

    #[test]
    fn then_luminance_fullrange_rgba8_to_mono16() {
        // Rgba8 → Mono8 → Mono16  (alpha ignored by Luminance)
        let src = Rgba8::new(200, 100, 50, 255);
        let result: Mono16 = Luminance.then::<Mono8, _>(FullRange).convert(&src);
        let expected_y = lum_u8(200, 100, 50);
        assert_eq!(result, Mono16::new(fr_u8_to_u16(expected_y)));
    }

    // ── Cross-depth broadcast ───────────────────────────────────────────────

    #[test]
    fn then_fullrange_broadcast_mono8_to_rgb16() {
        // Mono8 → Mono16 → Rgb16  (widen first, then broadcast)
        let src = Mono8::new(100);
        let result: Rgb16 = FullRange.then::<Mono16, _>(Broadcast).convert(&src);
        let wide = fr_u8_to_u16(100);
        assert_eq!(result, Rgb16::new(wide, wide, wide));
    }

    #[test]
    fn then_broadcast_fullrange_mono8_to_rgb16() {
        // Mono8 → Rgb8 → Rgb16  (broadcast first, then widen)
        let src = Mono8::new(100);
        let result: Rgb16 = Broadcast.then::<Rgb8, _>(FullRange).convert(&src);
        let wide = fr_u8_to_u16(100);
        assert_eq!(result, Rgb16::new(wide, wide, wide));
    }

    #[test]
    fn then_fullrange_broadcast_mono8_to_bgra16() {
        // Mono8 → Mono16 → Bgra16  (widen first, then broadcast with alpha)
        let src = Mono8::new(50);
        let result: Bgra16 = FullRange.then::<Mono16, _>(Broadcast).convert(&src);
        let wide = fr_u8_to_u16(50);
        assert_eq!(result, Bgra16::new(wide, wide, wide, 65535));
    }

    #[test]
    fn then_broadcast_fullrange_mono8_to_rgbf32() {
        // Mono8 → Rgb8 → RgbF32
        let src = Mono8::new(255);
        let result: RgbF32 = Broadcast.then::<Rgb8, _>(FullRange).convert(&src);
        assert!(approx(result.r, 1.0, 1e-3));
        assert!(approx(result.g, 1.0, 1e-3));
        assert!(approx(result.b, 1.0, 1e-3));
    }

    // ── Cross-order add alpha ───────────────────────────────────────────────

    #[test]
    fn then_colorswap_addalpha_rgb8_to_bgra8() {
        // Rgb8 → Bgr8 → Bgra8
        let src = Rgb8::new(10, 20, 30);
        let result: Bgra8 = ColorSwap.then::<Bgr8, _>(AddAlpha).convert(&src);
        assert_eq!(result, Bgra8::new(30, 20, 10, 255));
    }

    #[test]
    fn then_addalpha_colorswap_rgb8_to_bgra8() {
        // Rgb8 → Rgba8 → Bgra8
        let src = Rgb8::new(10, 20, 30);
        let result: Bgra8 = AddAlpha.then::<Rgba8, _>(ColorSwap).convert(&src);
        assert_eq!(result, Bgra8::new(30, 20, 10, 255));
    }

    #[test]
    fn then_colorswap_addalpha_bgr8_to_rgba8() {
        // Bgr8 → Rgb8 → Rgba8
        let src = Bgr8::new(30, 20, 10);
        let result: Rgba8 = ColorSwap.then::<Rgb8, _>(AddAlpha).convert(&src);
        assert_eq!(result, Rgba8::new(10, 20, 30, 255));
    }

    // ── Triple chains ───────────────────────────────────────────────────────

    #[test]
    fn then_triple_rgb8_to_bgra16() {
        // Rgb8 → Bgr8 → Bgra8 → Bgra16
        let src = Rgb8::new(10, 20, 30);
        let result: Bgra16 = ColorSwap
            .then::<Bgr8, _>(AddAlpha)
            .then::<Bgra8, _>(FullRange)
            .convert(&src);
        assert_eq!(
            result,
            Bgra16::new(fr_u8_to_u16(30), fr_u8_to_u16(20), fr_u8_to_u16(10), 65535)
        );
    }

    #[test]
    fn then_triple_mono8_to_bgr16() {
        // Mono8 → Rgb8 → Bgr8 → Bgr16
        let src = Mono8::new(128);
        let result: Bgr16 = Broadcast
            .then::<Rgb8, _>(ColorSwap)
            .then::<Bgr8, _>(FullRange)
            .convert(&src);
        let wide = fr_u8_to_u16(128);
        assert_eq!(result, Bgr16::new(wide, wide, wide));
    }

    #[test]
    fn then_triple_mono8_to_bgra16() {
        // Mono8 → Bgr8 → Bgra8 → Bgra16
        let src = Mono8::new(100);
        let result: Bgra16 = Broadcast
            .then::<Bgr8, _>(AddAlpha)
            .then::<Bgra8, _>(FullRange)
            .convert(&src);
        let wide = fr_u8_to_u16(100);
        assert_eq!(result, Bgra16::new(wide, wide, wide, 65535));
    }

    #[test]
    fn then_triple_rgb16_to_bgra8() {
        // Rgb16 → Rgb8 → Bgr8 → Bgra8
        let src = Rgb16::new(65535, 32768, 0);
        let result: Bgra8 = FullRange
            .then::<Rgb8, _>(ColorSwap)
            .then::<Bgr8, _>(AddAlpha)
            .convert(&src);
        let r8 = fr_u16_to_u8(65535);
        let g8 = fr_u16_to_u8(32768);
        let b8 = fr_u16_to_u8(0);
        assert_eq!(result, Bgra8::new(b8, g8, r8, 255));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Then combinator — image-level tests
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn then_convert_image_cross_depth_colorswap() {
        // Rgb8 → Bgr16 in one pass via convert_image
        let img = Image::fill(3, 3, Rgb8::new(200, 100, 50));
        let out: Image<Bgr16> = convert_image(&img, ColorSwap.then::<Bgr8, _>(FullRange));
        let expected = Bgr16::new(fr_u8_to_u16(50), fr_u8_to_u16(100), fr_u8_to_u16(200));
        for y in 0..3 {
            for x in 0..3 {
                assert_eq!(out.pixel_at(x, y), expected);
            }
        }
    }

    #[test]
    fn then_convert_image_cross_depth_addalpha() {
        // Rgb8 → Rgba16 in one pass
        let img = Image::fill(2, 2, Rgb8::new(10, 20, 30));
        let out: Image<Rgba16> = convert_image(&img, AddAlpha.then::<Rgba8, _>(FullRange));
        let expected = Rgba16::new(fr_u8_to_u16(10), fr_u8_to_u16(20), fr_u8_to_u16(30), 65535);
        for y in 0..2 {
            for x in 0..2 {
                assert_eq!(out.pixel_at(x, y), expected);
            }
        }
    }

    #[test]
    fn then_convert_image_cross_depth_luminance() {
        // Rgb16 → Mono8 in one pass
        let img = Image::fill(2, 2, Rgb16::new(65535, 65535, 65535));
        let out: Image<Mono8> = convert_image(&img, Luminance.then::<Mono16, _>(FullRange));
        for y in 0..2 {
            for x in 0..2 {
                assert_eq!(out.pixel_at(x, y), Mono8::new(255));
            }
        }
    }

    #[test]
    fn then_convert_image_cross_depth_broadcast() {
        // Mono8 → Rgb16 in one pass
        let img = Image::fill(2, 2, Mono8::new(100));
        let out: Image<Rgb16> = convert_image(&img, Broadcast.then::<Rgb8, _>(FullRange));
        let wide = fr_u8_to_u16(100);
        let expected = Rgb16::new(wide, wide, wide);
        for y in 0..2 {
            for x in 0..2 {
                assert_eq!(out.pixel_at(x, y), expected);
            }
        }
    }

    #[test]
    fn then_convert_image_triple_chain() {
        // Rgb8 → Bgra16 in one pass (3 strategies chained)
        let img = Image::fill(2, 2, Rgb8::new(10, 20, 30));
        let out: Image<Bgra16> = convert_image(
            &img,
            ColorSwap
                .then::<Bgr8, _>(AddAlpha)
                .then::<Bgra8, _>(FullRange),
        );
        let expected = Bgra16::new(fr_u8_to_u16(30), fr_u8_to_u16(20), fr_u8_to_u16(10), 65535);
        for y in 0..2 {
            for x in 0..2 {
                assert_eq!(out.pixel_at(x, y), expected);
            }
        }
    }

    #[test]
    fn then_convert_image_into_cross_depth_colorswap() {
        // Rgb8 → Bgr16 via convert_image_into
        let img = Image::fill(2, 2, Rgb8::new(200, 100, 50));
        let mut out = Image::<Bgr16>::zero(2, 2);
        convert_image_into(&img, &mut out, ColorSwap.then::<Bgr8, _>(FullRange));
        let expected = Bgr16::new(fr_u8_to_u16(50), fr_u8_to_u16(100), fr_u8_to_u16(200));
        for y in 0..2 {
            for x in 0..2 {
                assert_eq!(out.pixel_at(x, y), expected);
            }
        }
    }

    // ── Then + PixelMap ─────────────────────────────────────────────────────

    #[test]
    fn then_with_pixelmap_luminance_then_custom() {
        // Rgb8 → Mono8 via Luminance, then Mono8 → f32 via a PixelMap closure
        let src = Rgb8::new(255, 255, 255);
        let result: f32 = Luminance
            .then::<Mono8, _>(PixelMap(|m: &Mono8| m.as_bytes()[0] as f32 / 255.0))
            .convert(&src);
        assert!(approx(result, 1.0, 1e-3));
    }

    #[test]
    fn then_with_pixelmap_custom_then_broadcast() {
        // f32 → Mono8 via PixelMap, then Mono8 → Rgb8 via Broadcast
        let src: f32 = 0.5;
        let result: Rgb8 = PixelMap(|v: &f32| Mono8::new((*v * 255.0) as u8))
            .then::<Mono8, _>(Broadcast)
            .convert(&src);
        assert_eq!(result, Rgb8::new(127, 127, 127));
    }

    // ── Commutativity / equivalence checks ──────────────────────────────────

    #[test]
    fn then_order_equivalence_colorswap_fullrange() {
        // For cross-depth colour swap, both orderings should produce the same
        // result (swap-then-widen vs widen-then-swap).
        let src = Rgb8::new(200, 100, 50);
        let a: Bgr16 = ColorSwap.then::<Bgr8, _>(FullRange).convert(&src);
        let b: Bgr16 = FullRange.then::<Rgb16, _>(ColorSwap).convert(&src);
        assert_eq!(a, b);
    }

    #[test]
    fn then_order_equivalence_addalpha_fullrange() {
        // AddAlpha then FullRange vs FullRange then AddAlpha
        let src = Rgb8::new(10, 20, 30);
        let a: Rgba16 = AddAlpha.then::<Rgba8, _>(FullRange).convert(&src);
        let b: Rgba16 = FullRange.then::<Rgb16, _>(AddAlpha).convert(&src);
        assert_eq!(a, b);
    }

    // ─── sRGB pixel types ───────────────────────────────────────────────────

    #[test]
    fn srgb8_new_and_fields() {
        let p = Srgb8::new(10, 20, 30);
        assert_eq!(p.r.0, 10);
        assert_eq!(p.g.0, 20);
        assert_eq!(p.b.0, 30);
    }

    #[test]
    fn srgba8_new_and_fields() {
        let p = Srgba8::new(10, 20, 30, 255);
        assert_eq!(p.r.0, 10);
        assert_eq!(p.g.0, 20);
        assert_eq!(p.b.0, 30);
        assert_eq!(p.a.0, 255);
    }

    #[test]
    fn srgb8_zero() {
        let p = Srgb8::zero();
        assert_eq!(p, Srgb8::new(0, 0, 0));
    }

    #[test]
    fn srgba8_zero() {
        let p = Srgba8::zero();
        assert_eq!(p, Srgba8::new(0, 0, 0, 0));
    }

    #[test]
    fn srgb8_plain_pixel_size() {
        assert_eq!(Srgb8::SIZE, 3);
        assert_eq!(Srgb8::DIM, 3);
    }

    #[test]
    fn srgba8_plain_pixel_size() {
        assert_eq!(Srgba8::SIZE, 4);
        assert_eq!(Srgba8::DIM, 4);
    }

    #[test]
    fn srgb8_as_bytes_roundtrip() {
        let p = Srgb8::new(100, 150, 200);
        let bytes = p.as_bytes();
        assert_eq!(bytes, &[100, 150, 200]);
        let q = Srgb8::from_bytes(bytes).unwrap();
        assert_eq!(p, q);
    }

    #[test]
    fn srgba8_as_bytes_roundtrip() {
        let p = Srgba8::new(100, 150, 200, 255);
        let bytes = p.as_bytes();
        assert_eq!(bytes, &[100, 150, 200, 255]);
        let q = Srgba8::from_bytes(bytes).unwrap();
        assert_eq!(p, q);
    }

    #[test]
    fn srgb8_uniform_channels() {
        let p = Srgb8::new(10, 20, 30);
        assert_eq!(Srgb8::CHANNEL_COUNT, 3);
        assert_eq!(p.channel(0), Saturating(10u8));
        assert_eq!(p.channel(1), Saturating(20u8));
        assert_eq!(p.channel(2), Saturating(30u8));
    }

    #[test]
    fn srgba8_uniform_channels() {
        let p = Srgba8::new(10, 20, 30, 40);
        assert_eq!(Srgba8::CHANNEL_COUNT, 4);
        assert_eq!(p.channel(0), Saturating(10u8));
        assert_eq!(p.channel(1), Saturating(20u8));
        assert_eq!(p.channel(2), Saturating(30u8));
        assert_eq!(p.channel(3), Saturating(40u8));
    }

    // ─── SrgbGamma: decode (sRGB → linear) ─────────────────────────────────

    #[test]
    fn srgb_gamma_decode_black() {
        let linear: RgbF32 = SrgbGamma.convert(&Srgb8::new(0, 0, 0));
        assert_eq!(linear.r, 0.0);
        assert_eq!(linear.g, 0.0);
        assert_eq!(linear.b, 0.0);
    }

    #[test]
    fn srgb_gamma_decode_white() {
        let linear: RgbF32 = SrgbGamma.convert(&Srgb8::new(255, 255, 255));
        assert!(approx(linear.r, 1.0, 0.001));
        assert!(approx(linear.g, 1.0, 0.001));
        assert!(approx(linear.b, 1.0, 0.001));
    }

    #[test]
    fn srgb_gamma_decode_mid_gray() {
        // sRGB 128 ≈ 0.2158 linear (not 0.502)
        let linear: RgbF32 = SrgbGamma.convert(&Srgb8::new(128, 128, 128));
        assert!((linear.r - 0.2158).abs() < 0.002);
        assert!((linear.g - 0.2158).abs() < 0.002);
        assert!((linear.b - 0.2158).abs() < 0.002);
    }

    #[test]
    fn srgb_gamma_decode_low_value_linear_region() {
        // Values ≤ ~10 are in the linear segment of the sRGB curve
        // sRGB 10 / 255 ≈ 0.03922, which is < 0.04045 → linear region
        let linear: RgbF32 = SrgbGamma.convert(&Srgb8::new(10, 0, 0));
        let expected = (10.0 / 255.0) / 12.92;
        assert!((linear.r - expected).abs() < 0.0001);
    }

    #[test]
    fn srgb_gamma_decode_per_channel() {
        let linear: RgbF32 = SrgbGamma.convert(&Srgb8::new(255, 0, 128));
        assert!(approx(linear.r, 1.0, 0.001));
        assert_eq!(linear.g, 0.0);
        assert!((linear.b - 0.2158).abs() < 0.002);
    }

    // ─── SrgbGamma: encode (linear → sRGB) ─────────────────────────────────

    #[test]
    fn srgb_gamma_encode_black() {
        let srgb: Srgb8 = SrgbGamma.convert(&RgbF32::new(0.0, 0.0, 0.0));
        assert_eq!(srgb, Srgb8::new(0, 0, 0));
    }

    #[test]
    fn srgb_gamma_encode_white() {
        let srgb: Srgb8 = SrgbGamma.convert(&RgbF32::new(1.0, 1.0, 1.0));
        assert_eq!(srgb, Srgb8::new(255, 255, 255));
    }

    #[test]
    fn srgb_gamma_encode_half_linear() {
        // 0.5 linear ≈ 188 sRGB (not 128)
        let srgb: Srgb8 = SrgbGamma.convert(&RgbF32::new(0.5, 0.5, 0.5));
        assert_eq!(srgb.r.0, 188);
        assert_eq!(srgb.g.0, 188);
        assert_eq!(srgb.b.0, 188);
    }

    #[test]
    fn srgb_gamma_encode_clamps_negative() {
        let srgb: Srgb8 = SrgbGamma.convert(&RgbF32::new(-0.5, 0.0, 0.0));
        assert_eq!(srgb.r.0, 0);
    }

    #[test]
    fn srgb_gamma_encode_clamps_above_one() {
        let srgb: Srgb8 = SrgbGamma.convert(&RgbF32::new(1.5, 0.0, 0.0));
        assert_eq!(srgb.r.0, 255);
    }

    #[test]
    fn srgb_gamma_encode_low_value_linear_region() {
        // Values ≤ 0.0031308 use the linear segment
        let v = 0.002;
        let srgb: Srgb8 = SrgbGamma.convert(&RgbF32::new(v, 0.0, 0.0));
        let expected = (v * 12.92 * 255.0 + 0.5) as u8;
        assert_eq!(srgb.r.0, expected);
    }

    // ─── SrgbGamma: roundtrip ───────────────────────────────────────────────

    #[test]
    fn srgb_gamma_roundtrip_black() {
        let orig = Srgb8::new(0, 0, 0);
        let linear: RgbF32 = SrgbGamma.convert(&orig);
        let back: Srgb8 = SrgbGamma.convert(&linear);
        assert_eq!(orig, back);
    }

    #[test]
    fn srgb_gamma_roundtrip_white() {
        let orig = Srgb8::new(255, 255, 255);
        let linear: RgbF32 = SrgbGamma.convert(&orig);
        let back: Srgb8 = SrgbGamma.convert(&linear);
        assert_eq!(orig, back);
    }

    #[test]
    fn srgb_gamma_roundtrip_all_values() {
        // Every u8 value should survive a decode→encode roundtrip
        for v in 0..=255u8 {
            let orig = Srgb8::new(v, v, v);
            let linear: RgbF32 = SrgbGamma.convert(&orig);
            let back: Srgb8 = SrgbGamma.convert(&linear);
            assert_eq!(orig, back, "roundtrip failed for sRGB value {v}");
        }
    }

    // ─── SrgbGamma: Srgba8 ↔ RgbaF32 ───────────────────────────────────────

    #[test]
    fn srgba_gamma_decode_alpha_is_linear() {
        // Alpha must be linear (scaled to [0,1]), NOT gamma-converted
        let src = Srgba8::new(0, 0, 0, 128);
        let linear: RgbaF32 = SrgbGamma.convert(&src);
        // 128/255 ≈ 0.502, not 0.2158 (which gamma decode would give)
        assert!((linear.a - 128.0 / 255.0).abs() < 0.001);
    }

    #[test]
    fn srgba_gamma_decode_full_alpha() {
        let src = Srgba8::new(128, 64, 200, 255);
        let linear: RgbaF32 = SrgbGamma.convert(&src);
        assert!(approx(linear.a, 1.0, 0.001));
    }

    #[test]
    fn srgba_gamma_decode_zero_alpha() {
        let src = Srgba8::new(128, 64, 200, 0);
        let linear: RgbaF32 = SrgbGamma.convert(&src);
        assert_eq!(linear.a, 0.0);
    }

    #[test]
    fn srgba_gamma_decode_rgb_matches_srgb8() {
        // The RGB channels of Srgba8 should decode identically to Srgb8
        let srgb = Srgb8::new(100, 150, 200);
        let srgba = Srgba8::new(100, 150, 200, 255);
        let rgb: RgbF32 = SrgbGamma.convert(&srgb);
        let rgba: RgbaF32 = SrgbGamma.convert(&srgba);
        assert_eq!(rgb.r, rgba.r);
        assert_eq!(rgb.g, rgba.g);
        assert_eq!(rgb.b, rgba.b);
    }

    #[test]
    fn srgba_gamma_encode_alpha_is_linear() {
        let src = RgbaF32::new(0.0, 0.0, 0.0, 0.5);
        let srgba: Srgba8 = SrgbGamma.convert(&src);
        // 0.5 * 255 + 0.5 = 128  (linear, not gamma)
        assert_eq!(srgba.a.0, 128);
    }

    #[test]
    fn srgba_gamma_encode_clamps_alpha() {
        let src = RgbaF32::new(0.0, 0.0, 0.0, 1.5);
        let srgba: Srgba8 = SrgbGamma.convert(&src);
        assert_eq!(srgba.a.0, 255);
    }

    #[test]
    fn srgba_gamma_roundtrip_all_values() {
        for v in 0..=255u8 {
            let orig = Srgba8::new(v, v, v, v);
            let linear: RgbaF32 = SrgbGamma.convert(&orig);
            let back: Srgba8 = SrgbGamma.convert(&linear);
            assert_eq!(orig, back, "roundtrip failed for sRGBA value {v}");
        }
    }

    // ─── SrgbGamma: composability with .then() ─────────────────────────────

    #[test]
    fn srgb_gamma_then_fullrange_srgb8_to_rgb16() {
        // Srgb8 → RgbF32 → Rgb16
        let method = SrgbGamma.then::<RgbF32, _>(FullRange);
        let result: Rgb16 = method.convert(&Srgb8::new(255, 0, 128));
        assert_eq!(result.r.0, 65535); // 1.0 → 65535
        assert_eq!(result.g.0, 0); // 0.0 → 0
        // sRGB 128 ≈ 0.2158 linear → ~14145 in u16
        assert!((result.b.0 as i32 - 14145).unsigned_abs() < 100);
    }

    #[test]
    fn fullrange_then_srgb_gamma_rgb16_to_srgb8() {
        // Rgb16 → RgbF32 → Srgb8
        let method = FullRange.then::<RgbF32, _>(SrgbGamma);
        let result: Srgb8 = method.convert(&Rgb16::new(65535, 0, 32768));
        assert_eq!(result.r.0, 255); // 1.0 linear → 255 sRGB
        assert_eq!(result.g.0, 0); // 0.0 → 0
        // 32768/65535 ≈ 0.5 linear → 188 sRGB
        assert_eq!(result.b.0, 188);
    }

    #[test]
    fn srgb_addalpha() {
        let src = Srgb8::new(10, 20, 30);
        let dst: Srgba8 = AddAlpha.convert(&src);
        assert_eq!(dst, Srgba8::new(10, 20, 30, 255));
    }

    // ─── SrgbGamma: convert_image integration ───────────────────────────────

    #[test]
    fn convert_image_srgb8_to_rgbf32() {
        let img: Image<Srgb8> = Image::fill(3, 3, Srgb8::new(128, 128, 128));
        let linear: Image<RgbF32> = convert_image(&img, SrgbGamma);
        let p = linear.pixel_at(1, 1);
        assert!((p.r - 0.2158).abs() < 0.002);
        assert!((p.g - 0.2158).abs() < 0.002);
        assert!((p.b - 0.2158).abs() < 0.002);
    }

    #[test]
    fn convert_image_rgbf32_to_srgb8() {
        let img: Image<RgbF32> = Image::fill(3, 3, RgbF32::new(0.5, 0.5, 0.5));
        let srgb: Image<Srgb8> = convert_image(&img, SrgbGamma);
        let p = srgb.pixel_at(1, 1);
        assert_eq!(p.r.0, 188);
        assert_eq!(p.g.0, 188);
        assert_eq!(p.b.0, 188);
    }

    #[test]
    fn convert_image_srgba8_roundtrip() {
        let orig: Image<Srgba8> = Image::generate(4, 4, |x, y| {
            Srgba8::new((x * 60) as u8, (y * 60) as u8, 128, 200)
        });
        let linear: Image<RgbaF32> = convert_image(&orig, SrgbGamma);
        let back: Image<Srgba8> = convert_image(&linear, SrgbGamma);
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(
                    orig.pixel_at(x, y),
                    back.pixel_at(x, y),
                    "roundtrip failed at ({x}, {y})"
                );
            }
        }
    }

    // ─── SrgbGamma: nearest-neighbor resize works on sRGB ───────────────────

    #[test]
    fn nearest_neighbor_resize_srgb8() {
        use crate::transform::{NearestNeighbor, resize};
        let img: Image<Srgb8> = Image::fill(4, 4, Srgb8::new(128, 64, 200));
        let resized: Image<Srgb8> = resize(&img, crate::Size::new(2, 2), NearestNeighbor);
        assert_eq!(resized.pixel_at(0, 0), Srgb8::new(128, 64, 200));
    }

    #[test]
    fn nearest_neighbor_resize_srgba8() {
        use crate::transform::{NearestNeighbor, resize};
        let img: Image<Srgba8> = Image::fill(4, 4, Srgba8::new(128, 64, 200, 255));
        let resized: Image<Srgba8> = resize(&img, crate::Size::new(2, 2), NearestNeighbor);
        assert_eq!(resized.pixel_at(0, 0), Srgba8::new(128, 64, 200, 255));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // FullRange — Mono32 / Mono64 depth conversions
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn fullrange_mono8_to_mono32_extremes() {
        let a: Mono32 = FullRange.convert(&Mono8::new(0));
        let b: Mono32 = FullRange.convert(&Mono8::new(255));
        assert_eq!(a, Mono32::new(0));
        assert_eq!(b, Mono32::new(u32::MAX));
    }

    #[test]
    fn fullrange_mono32_to_mono8_extremes() {
        let a: Mono8 = FullRange.convert(&Mono32::new(0));
        let b: Mono8 = FullRange.convert(&Mono32::new(u32::MAX));
        assert_eq!(a, Mono8::new(0));
        assert_eq!(b, Mono8::new(255));
    }

    #[test]
    fn fullrange_mono16_to_mono32_extremes() {
        let a: Mono32 = FullRange.convert(&Mono16::new(0));
        let b: Mono32 = FullRange.convert(&Mono16::new(65535));
        assert_eq!(a, Mono32::new(0));
        assert_eq!(b, Mono32::new(u32::MAX));
    }

    #[test]
    fn fullrange_mono32_to_mono16_extremes() {
        let a: Mono16 = FullRange.convert(&Mono32::new(0));
        let b: Mono16 = FullRange.convert(&Mono32::new(u32::MAX));
        assert_eq!(a, Mono16::new(0));
        assert_eq!(b, Mono16::new(65535));
    }

    #[test]
    fn fullrange_mono8_mono32_roundtrip() {
        for v in 0..=255u8 {
            let m32: Mono32 = FullRange.convert(&Mono8::new(v));
            let back: Mono8 = FullRange.convert(&m32);
            assert_eq!(back, Mono8::new(v), "roundtrip failed for {v}");
        }
    }

    #[test]
    fn fullrange_mono32_to_f32() {
        let a: MonoF32 = FullRange.convert(&Mono32::new(0));
        let b: MonoF32 = FullRange.convert(&Mono32::new(u32::MAX));
        assert!(approx(a.0, 0.0, 1e-6));
        assert!(approx(b.0, 1.0, 1e-6));
    }

    #[test]
    fn fullrange_f32_to_mono32() {
        let a: Mono32 = FullRange.convert(&MonoF32::new(0.0));
        let b: Mono32 = FullRange.convert(&MonoF32::new(1.0));
        assert_eq!(a, Mono32::new(0));
        assert_eq!(b, Mono32::new(u32::MAX));
    }

    #[test]
    fn fullrange_mono32_to_f64() {
        let a: MonoF64 = FullRange.convert(&Mono32::new(0));
        let b: MonoF64 = FullRange.convert(&Mono32::new(u32::MAX));
        assert!((a.0 - 0.0).abs() < 1e-12);
        assert!((b.0 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn fullrange_f64_to_mono32() {
        let a: Mono32 = FullRange.convert(&MonoF64::new(0.0));
        let b: Mono32 = FullRange.convert(&MonoF64::new(1.0));
        assert_eq!(a, Mono32::new(0));
        assert_eq!(b, Mono32::new(u32::MAX));
    }

    // ── Mono64 ──────────────────────────────────────────────────────────────

    #[test]
    fn fullrange_mono8_to_mono64_extremes() {
        let a: Mono64 = FullRange.convert(&Mono8::new(0));
        let b: Mono64 = FullRange.convert(&Mono8::new(255));
        assert_eq!(a, Mono64::new(0));
        assert_eq!(b, Mono64::new(u64::MAX));
    }

    #[test]
    fn fullrange_mono64_to_mono8_extremes() {
        let a: Mono8 = FullRange.convert(&Mono64::new(0));
        let b: Mono8 = FullRange.convert(&Mono64::new(u64::MAX));
        assert_eq!(a, Mono8::new(0));
        assert_eq!(b, Mono8::new(255));
    }

    #[test]
    fn fullrange_mono16_to_mono64_extremes() {
        let a: Mono64 = FullRange.convert(&Mono16::new(0));
        let b: Mono64 = FullRange.convert(&Mono16::new(65535));
        assert_eq!(a, Mono64::new(0));
        assert_eq!(b, Mono64::new(u64::MAX));
    }

    #[test]
    fn fullrange_mono64_to_mono16_extremes() {
        let a: Mono16 = FullRange.convert(&Mono64::new(0));
        let b: Mono16 = FullRange.convert(&Mono64::new(u64::MAX));
        assert_eq!(a, Mono16::new(0));
        assert_eq!(b, Mono16::new(65535));
    }

    #[test]
    fn fullrange_mono32_to_mono64_extremes() {
        let a: Mono64 = FullRange.convert(&Mono32::new(0));
        let b: Mono64 = FullRange.convert(&Mono32::new(u32::MAX));
        assert_eq!(a, Mono64::new(0));
        assert_eq!(b, Mono64::new(u64::MAX));
    }

    #[test]
    fn fullrange_mono64_to_mono32_extremes() {
        let a: Mono32 = FullRange.convert(&Mono64::new(0));
        let b: Mono32 = FullRange.convert(&Mono64::new(u64::MAX));
        assert_eq!(a, Mono32::new(0));
        assert_eq!(b, Mono32::new(u32::MAX));
    }

    #[test]
    fn fullrange_mono8_mono64_roundtrip() {
        for v in 0..=255u8 {
            let m64: Mono64 = FullRange.convert(&Mono8::new(v));
            let back: Mono8 = FullRange.convert(&m64);
            assert_eq!(back, Mono8::new(v), "roundtrip failed for {v}");
        }
    }

    #[test]
    fn fullrange_mono64_to_f64() {
        let a: MonoF64 = FullRange.convert(&Mono64::new(0));
        let b: MonoF64 = FullRange.convert(&Mono64::new(u64::MAX));
        assert!((a.0 - 0.0).abs() < 1e-12);
        assert!((b.0 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn fullrange_f64_to_mono64() {
        let a: Mono64 = FullRange.convert(&MonoF64::new(0.0));
        let b: Mono64 = FullRange.convert(&MonoF64::new(1.0));
        assert_eq!(a, Mono64::new(0));
        assert_eq!(b, Mono64::new(u64::MAX));
    }

    #[test]
    fn fullrange_mono64_to_f32() {
        let a: MonoF32 = FullRange.convert(&Mono64::new(0));
        let b: MonoF32 = FullRange.convert(&Mono64::new(u64::MAX));
        assert!(approx(a.0, 0.0, 1e-6));
        assert!(approx(b.0, 1.0, 1e-6));
    }

    #[test]
    fn fullrange_f32_to_mono64() {
        let a: Mono64 = FullRange.convert(&MonoF32::new(0.0));
        let b: Mono64 = FullRange.convert(&MonoF32::new(1.0));
        assert_eq!(a, Mono64::new(0));
        assert_eq!(b, Mono64::new(u64::MAX));
    }

    // ── MonoF32 ↔ MonoF64 (mono) ───────────────────────────────────────────

    #[test]
    fn fullrange_f32_to_f64_mono() {
        let a: MonoF64 = FullRange.convert(&MonoF32::new(0.0));
        let b: MonoF64 = FullRange.convert(&MonoF32::new(1.0));
        let c: MonoF64 = FullRange.convert(&MonoF32::new(0.5));
        assert!((a.0 - 0.0).abs() < 1e-12);
        assert!((b.0 - 1.0).abs() < 1e-12);
        assert!((c.0 - 0.5).abs() < 1e-6);
    }

    #[test]
    fn fullrange_f64_to_f32_mono() {
        let a: MonoF32 = FullRange.convert(&MonoF64::new(0.0));
        let b: MonoF32 = FullRange.convert(&MonoF64::new(1.0));
        let c: MonoF32 = FullRange.convert(&MonoF64::new(0.5));
        assert!(approx(a.0, 0.0, 1e-6));
        assert!(approx(b.0, 1.0, 1e-6));
        assert!(approx(c.0, 0.5, 1e-6));
    }

    // ── Mono8/Mono16 ↔ MonoF64 ─────────────────────────────────────────────

    #[test]
    fn fullrange_mono8_to_f64() {
        let a: MonoF64 = FullRange.convert(&Mono8::new(0));
        let b: MonoF64 = FullRange.convert(&Mono8::new(255));
        assert!((a.0 - 0.0).abs() < 1e-12);
        assert!((b.0 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn fullrange_f64_to_mono8() {
        let a: Mono8 = FullRange.convert(&MonoF64::new(0.0));
        let b: Mono8 = FullRange.convert(&MonoF64::new(1.0));
        assert_eq!(a, Mono8::new(0));
        assert_eq!(b, Mono8::new(255));
    }

    #[test]
    fn fullrange_mono16_to_f64() {
        let a: MonoF64 = FullRange.convert(&Mono16::new(0));
        let b: MonoF64 = FullRange.convert(&Mono16::new(65535));
        assert!((a.0 - 0.0).abs() < 1e-12);
        assert!((b.0 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn fullrange_f64_to_mono16() {
        let a: Mono16 = FullRange.convert(&MonoF64::new(0.0));
        let b: Mono16 = FullRange.convert(&MonoF64::new(1.0));
        assert_eq!(a, Mono16::new(0));
        assert_eq!(b, Mono16::new(65535));
    }

    // ── Mono<BITS> → Mono32, f32 ───────────────────────────────────────────

    #[test]
    fn fullrange_mono10_to_mono32() {
        let a: Mono32 = FullRange.convert(&Mono::<10>::new(0));
        let b: Mono32 = FullRange.convert(&Mono::<10>::new(1023));
        assert_eq!(a, Mono32::new(0));
        assert_eq!(b, Mono32::new(u32::MAX));
    }

    #[test]
    fn fullrange_mono12_to_mono32() {
        let a: Mono32 = FullRange.convert(&Mono::<12>::new(0));
        let b: Mono32 = FullRange.convert(&Mono::<12>::new(4095));
        assert_eq!(a, Mono32::new(0));
        assert_eq!(b, Mono32::new(u32::MAX));
    }

    #[test]
    fn fullrange_mono10_to_f32() {
        let a: MonoF32 = FullRange.convert(&Mono::<10>::new(0));
        let b: MonoF32 = FullRange.convert(&Mono::<10>::new(1023));
        assert!(approx(a.0, 0.0, 1e-6));
        assert!(approx(b.0, 1.0, 1e-6));
    }

    // ── Mono<BITS> → Mono64, f64 ───────────────────────────────────────────

    #[test]
    fn fullrange_mono10_to_mono64() {
        let a: Mono64 = FullRange.convert(&Mono::<10>::new(0));
        let b: Mono64 = FullRange.convert(&Mono::<10>::new(1023));
        assert_eq!(a, Mono64::new(0));
        assert_eq!(b, Mono64::new(u64::MAX));
    }

    #[test]
    fn fullrange_mono12_to_mono64() {
        let a: Mono64 = FullRange.convert(&Mono::<12>::new(0));
        let b: Mono64 = FullRange.convert(&Mono::<12>::new(4095));
        assert_eq!(a, Mono64::new(0));
        assert_eq!(b, Mono64::new(u64::MAX));
    }

    #[test]
    fn fullrange_mono14_to_mono64() {
        let a: Mono64 = FullRange.convert(&Mono::<14>::new(0));
        let b: Mono64 = FullRange.convert(&Mono::<14>::new(16383));
        assert_eq!(a, Mono64::new(0));
        assert_eq!(b, Mono64::new(u64::MAX));
    }

    #[test]
    fn fullrange_mono10_to_f64() {
        let a: MonoF64 = FullRange.convert(&Mono::<10>::new(0));
        let b: MonoF64 = FullRange.convert(&Mono::<10>::new(1023));
        assert!((a.0 - 0.0).abs() < 1e-12);
        assert!((b.0 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn fullrange_mono12_to_f64() {
        let a: MonoF64 = FullRange.convert(&Mono::<12>::new(0));
        let b: MonoF64 = FullRange.convert(&Mono::<12>::new(4095));
        assert!((a.0 - 0.0).abs() < 1e-12);
        assert!((b.0 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn fullrange_mono10_to_mono64_midpoint() {
        let mid: Mono64 = FullRange.convert(&Mono::<10>::new(512));
        // 512/1023 ≈ 0.5005 → scaled to u64 range
        let expected = (512u128 * u64::MAX as u128 + 1023 / 2) / 1023;
        assert_eq!(mid, Mono64::new(expected as u64));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Narrow — Mono32 / Mono64
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn clamp_mono8_to_mono32() {
        let a: Mono32 = Narrow.convert(&Mono8::new(42));
        assert_eq!(a, Mono32::new(42));
    }

    #[test]
    fn clamp_mono32_to_mono8_fits() {
        let a: Mono8 = Narrow.convert(&Mono32::new(42));
        assert_eq!(a, Mono8::new(42));
    }

    #[test]
    fn clamp_mono32_to_mono8_clamps() {
        let a: Mono8 = Narrow.convert(&Mono32::new(1000));
        assert_eq!(a, Mono8::new(255));
    }

    #[test]
    fn clamp_mono16_to_mono32() {
        let a: Mono32 = Narrow.convert(&Mono16::new(60000));
        assert_eq!(a, Mono32::new(60000));
    }

    #[test]
    fn clamp_mono32_to_mono16_fits() {
        let a: Mono16 = Narrow.convert(&Mono32::new(60000));
        assert_eq!(a, Mono16::new(60000));
    }

    #[test]
    fn clamp_mono32_to_mono16_clamps() {
        let a: Mono16 = Narrow.convert(&Mono32::new(100_000));
        assert_eq!(a, Mono16::new(65535));
    }

    #[test]
    fn clamp_mono8_to_mono64() {
        let a: Mono64 = Narrow.convert(&Mono8::new(42));
        assert_eq!(a, Mono64::new(42));
    }

    #[test]
    fn clamp_mono64_to_mono8_clamps() {
        let a: Mono8 = Narrow.convert(&Mono64::new(1_000_000));
        assert_eq!(a, Mono8::new(255));
    }

    #[test]
    fn clamp_mono32_to_mono64() {
        let a: Mono64 = Narrow.convert(&Mono32::new(u32::MAX));
        assert_eq!(a, Mono64::new(u32::MAX as u64));
    }

    #[test]
    fn clamp_mono64_to_mono32_clamps() {
        let a: Mono32 = Narrow.convert(&Mono64::new(u64::MAX));
        assert_eq!(a, Mono32::new(u32::MAX));
    }

    #[test]
    fn clamp_mono64_to_mono32_fits() {
        let a: Mono32 = Narrow.convert(&Mono64::new(42));
        assert_eq!(a, Mono32::new(42));
    }

    #[test]
    fn clamp_mono16_to_mono64() {
        let a: Mono64 = Narrow.convert(&Mono16::new(0));
        let b: Mono64 = Narrow.convert(&Mono16::new(65535));
        assert_eq!(a, Mono64::new(0));
        assert_eq!(b, Mono64::new(65535));
    }

    #[test]
    fn clamp_mono64_to_mono16_clamps() {
        let a: Mono16 = Narrow.convert(&Mono64::new(1_000_000));
        assert_eq!(a, Mono16::new(65535));
    }

    #[test]
    fn clamp_mono64_to_mono16_fits() {
        let a: Mono16 = Narrow.convert(&Mono64::new(300));
        assert_eq!(a, Mono16::new(300));
    }

    #[test]
    fn clamp_mono10_to_mono32() {
        let a: Mono32 = Narrow.convert(&Mono::<10>::new(1023));
        assert_eq!(a, Mono32::new(1023));
    }

    #[test]
    fn clamp_mono10_to_mono64() {
        let a: Mono64 = Narrow.convert(&Mono::<10>::new(1023));
        assert_eq!(a, Mono64::new(1023));
    }

    #[test]
    fn clamp_mono12_to_mono64() {
        let a: Mono64 = Narrow.convert(&Mono::<12>::new(4095));
        assert_eq!(a, Mono64::new(4095));
    }

    #[test]
    fn clamp_mono14_to_mono64() {
        let a: Mono64 = Narrow.convert(&Mono::<14>::new(16383));
        assert_eq!(a, Mono64::new(16383));
    }

    #[test]
    fn clamp_mono10_to_mono64_zero() {
        let a: Mono64 = Narrow.convert(&Mono::<10>::new(0));
        assert_eq!(a, Mono64::new(0));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // FullRange — Multi-channel 32/64-bit depth conversions
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn fullrange_rgb8_to_rgb32() {
        let p: Rgb32 = FullRange.convert(&Rgb8::new(0, 128, 255));
        assert_eq!(p.r.0, 0);
        assert_eq!(p.b.0, u32::MAX);
    }

    #[test]
    fn fullrange_rgb32_to_rgb8() {
        let p: Rgb8 = FullRange.convert(&Rgb32::new(0, u32::MAX / 2, u32::MAX));
        assert_eq!(p.r.0, 0);
        assert_eq!(p.b.0, 255);
    }

    #[test]
    fn fullrange_rgb8_to_rgb64() {
        let p: Rgb64 = FullRange.convert(&Rgb8::new(0, 128, 255));
        assert_eq!(p.r.0, 0);
        assert_eq!(p.b.0, u64::MAX);
    }

    #[test]
    fn fullrange_rgb64_to_rgb8() {
        let p: Rgb8 = FullRange.convert(&Rgb64::new(0, u64::MAX / 2, u64::MAX));
        assert_eq!(p.r.0, 0);
        assert_eq!(p.b.0, 255);
    }

    #[test]
    fn fullrange_rgb16_to_rgb64() {
        let p: Rgb64 = FullRange.convert(&Rgb16::new(0, 32768, 65535));
        assert_eq!(p.r.0, 0);
        assert_eq!(p.b.0, u64::MAX);
    }

    #[test]
    fn fullrange_rgb64_to_rgb16() {
        let p: Rgb16 = FullRange.convert(&Rgb64::new(0, u64::MAX / 2, u64::MAX));
        assert_eq!(p.r.0, 0);
        assert_eq!(p.b.0, 65535);
    }

    #[test]
    fn fullrange_rgb32_to_rgb64() {
        let p: Rgb64 = FullRange.convert(&Rgb32::new(0, u32::MAX / 2, u32::MAX));
        assert_eq!(p.r.0, 0);
        assert_eq!(p.b.0, u64::MAX);
    }

    #[test]
    fn fullrange_rgb64_to_rgb32() {
        let p: Rgb32 = FullRange.convert(&Rgb64::new(0, u64::MAX / 2, u64::MAX));
        assert_eq!(p.r.0, 0);
        assert_eq!(p.b.0, u32::MAX);
    }

    #[test]
    fn fullrange_rgb8_rgb64_roundtrip() {
        let orig = Rgb8::new(100, 200, 50);
        let wide: Rgb64 = FullRange.convert(&orig);
        let back: Rgb8 = FullRange.convert(&wide);
        assert_eq!(back, orig);
    }

    // ── f64 color conversions ───────────────────────────────────────────────

    #[test]
    fn fullrange_rgb8_to_rgbf64() {
        let p: RgbF64 = FullRange.convert(&Rgb8::new(0, 128, 255));
        assert!((p.r - 0.0).abs() < 1e-12);
        assert!((p.b - 1.0).abs() < 1e-12);
    }

    #[test]
    fn fullrange_rgbf64_to_rgb8() {
        let p: Rgb8 = FullRange.convert(&RgbF64::new(0.0, 0.5, 1.0));
        assert_eq!(p.r.0, 0);
        assert_eq!(p.b.0, 255);
    }

    #[test]
    fn fullrange_rgb64_to_rgbf64() {
        let p: RgbF64 = FullRange.convert(&Rgb64::new(0, u64::MAX / 2, u64::MAX));
        assert!((p.r - 0.0).abs() < 1e-6);
        assert!((p.b - 1.0).abs() < 1e-6);
    }

    #[test]
    fn fullrange_rgbf64_to_rgb64() {
        let p: Rgb64 = FullRange.convert(&RgbF64::new(0.0, 0.5, 1.0));
        assert_eq!(p.r.0, 0);
        assert_eq!(p.b.0, u64::MAX);
    }

    #[test]
    fn fullrange_rgbf32_to_rgbf64() {
        let p: RgbF64 = FullRange.convert(&RgbF32::new(0.0, 0.5, 1.0));
        assert!((p.r - 0.0).abs() < 1e-6);
        assert!((p.g - 0.5).abs() < 1e-6);
        assert!((p.b - 1.0).abs() < 1e-6);
    }

    #[test]
    fn fullrange_rgbf64_to_rgbf32() {
        let p: RgbF32 = FullRange.convert(&RgbF64::new(0.0, 0.5, 1.0));
        assert!(approx(p.r, 0.0, 1e-6));
        assert!(approx(p.g, 0.5, 1e-6));
        assert!(approx(p.b, 1.0, 1e-6));
    }

    // ── RGBA 64-bit ─────────────────────────────────────────────────────────

    #[test]
    fn fullrange_rgba8_to_rgba64() {
        let p: Rgba64 = FullRange.convert(&Rgba8::new(0, 128, 255, 255));
        assert_eq!(p.r.0, 0);
        assert_eq!(p.b.0, u64::MAX);
        assert_eq!(p.a.0, u64::MAX);
    }

    #[test]
    fn fullrange_rgba64_to_rgba8() {
        let p: Rgba8 = FullRange.convert(&Rgba64::new(0, u64::MAX / 2, u64::MAX, u64::MAX));
        assert_eq!(p.r.0, 0);
        assert_eq!(p.b.0, 255);
        assert_eq!(p.a.0, 255);
    }

    #[test]
    fn fullrange_rgba8_to_rgbaf64() {
        let p: RgbaF64 = FullRange.convert(&Rgba8::new(0, 128, 255, 255));
        assert!((p.r - 0.0).abs() < 1e-12);
        assert!((p.b - 1.0).abs() < 1e-12);
        assert!((p.a - 1.0).abs() < 1e-12);
    }

    #[test]
    fn fullrange_rgbaf64_to_rgba8() {
        let p: Rgba8 = FullRange.convert(&RgbaF64::new(0.0, 0.5, 1.0, 1.0));
        assert_eq!(p.r.0, 0);
        assert_eq!(p.b.0, 255);
        assert_eq!(p.a.0, 255);
    }

    // ── BGR 64-bit ──────────────────────────────────────────────────────────

    #[test]
    fn fullrange_bgr8_to_bgr64() {
        let p: Bgr64 = FullRange.convert(&Bgr8::new(255, 128, 0));
        assert_eq!(p.b.0, u64::MAX);
        assert_eq!(p.r.0, 0);
    }

    #[test]
    fn fullrange_bgr64_to_bgr8() {
        let p: Bgr8 = FullRange.convert(&Bgr64::new(u64::MAX, u64::MAX / 2, 0));
        assert_eq!(p.b.0, 255);
        assert_eq!(p.r.0, 0);
    }

    #[test]
    fn fullrange_bgr8_to_bgrf64() {
        let p: BgrF64 = FullRange.convert(&Bgr8::new(255, 128, 0));
        assert!((p.b - 1.0).abs() < 1e-12);
        assert!((p.r - 0.0).abs() < 1e-12);
    }

    // ── BGRA 64-bit ─────────────────────────────────────────────────────────

    #[test]
    fn fullrange_bgra8_to_bgra64() {
        let p: Bgra64 = FullRange.convert(&Bgra8::new(255, 128, 0, 255));
        assert_eq!(p.b.0, u64::MAX);
        assert_eq!(p.r.0, 0);
        assert_eq!(p.a.0, u64::MAX);
    }

    #[test]
    fn fullrange_bgra64_to_bgra8() {
        let p: Bgra8 = FullRange.convert(&Bgra64::new(u64::MAX, u64::MAX / 2, 0, u64::MAX));
        assert_eq!(p.b.0, 255);
        assert_eq!(p.r.0, 0);
        assert_eq!(p.a.0, 255);
    }

    #[test]
    fn fullrange_bgra8_to_bgraf64() {
        let p: BgraF64 = FullRange.convert(&Bgra8::new(255, 128, 0, 255));
        assert!((p.b - 1.0).abs() < 1e-12);
        assert!((p.r - 0.0).abs() < 1e-12);
        assert!((p.a - 1.0).abs() < 1e-12);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Narrow — Multi-channel 32/64-bit
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn clamp_rgb8_to_rgb32() {
        let p: Rgb32 = Narrow.convert(&Rgb8::new(10, 20, 30));
        assert_eq!(p, Rgb32::new(10, 20, 30));
    }

    #[test]
    fn clamp_rgb32_to_rgb8_clamps() {
        let p: Rgb8 = Narrow.convert(&Rgb32::new(1000, 20, 30));
        assert_eq!(p, Rgb8::new(255, 20, 30));
    }

    #[test]
    fn clamp_rgb8_to_rgb64() {
        let p: Rgb64 = Narrow.convert(&Rgb8::new(10, 20, 30));
        assert_eq!(p, Rgb64::new(10, 20, 30));
    }

    #[test]
    fn clamp_rgb64_to_rgb8_clamps() {
        let p: Rgb8 = Narrow.convert(&Rgb64::new(1_000_000, 20, 30));
        assert_eq!(p, Rgb8::new(255, 20, 30));
    }

    #[test]
    fn clamp_rgb32_to_rgb64() {
        let p: Rgb64 = Narrow.convert(&Rgb32::new(u32::MAX, 42, 0));
        assert_eq!(p, Rgb64::new(u32::MAX as u64, 42, 0));
    }

    #[test]
    fn clamp_rgb64_to_rgb32_clamps() {
        let p: Rgb32 = Narrow.convert(&Rgb64::new(u64::MAX, 42, 0));
        assert_eq!(p, Rgb32::new(u32::MAX, 42, 0));
    }

    #[test]
    fn clamp_bgra8_to_bgra64() {
        let p: Bgra64 = Narrow.convert(&Bgra8::new(10, 20, 30, 40));
        assert_eq!(p, Bgra64::new(10, 20, 30, 40));
    }

    #[test]
    fn clamp_bgra64_to_bgra8_clamps() {
        let p: Bgra8 = Narrow.convert(&Bgra64::new(1_000_000, 20, 30, 40));
        assert_eq!(p, Bgra8::new(255, 20, 30, 40));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Luminance — 32/64-bit and f64
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn luminance_rgb32_white() {
        let p: Mono32 = Luminance.convert(&Rgb32::new(u32::MAX, u32::MAX, u32::MAX));
        // Should be close to u32::MAX (slight rounding from BT.601 coefficients)
        assert!(p == Mono32::new(u32::MAX) || (u32::MAX - mono32_val(&p)) < 256);
    }

    #[test]
    fn luminance_rgb64_white() {
        let p: Mono64 = Luminance.convert(&Rgb64::new(u64::MAX, u64::MAX, u64::MAX));
        assert!(p == Mono64::new(u64::MAX) || (u64::MAX - mono64_val(&p)) < 256);
    }

    #[test]
    fn luminance_rgbf64() {
        let p: MonoF64 = Luminance.convert(&RgbF64::new(1.0, 1.0, 1.0));
        assert!((p.0 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn luminance_bgrf64() {
        let p: MonoF64 = Luminance.convert(&BgrF64::new(0.0, 0.0, 1.0));
        // Luminance of pure red (r field) ≈ 0.299
        assert!((p.0 - 0.299).abs() < 1e-10);
    }

    #[test]
    fn luminance_rgba64_ignores_alpha() {
        let a: Mono64 = Luminance.convert(&Rgba64::new(u64::MAX, u64::MAX, u64::MAX, 0));
        let b: Mono64 = Luminance.convert(&Rgba64::new(u64::MAX, u64::MAX, u64::MAX, u64::MAX));
        assert_eq!(a, b);
    }

    #[test]
    fn luminance_bgra64() {
        let p: Mono64 = Luminance.convert(&Bgra64::new(0, 0, u64::MAX, 0));
        // Pure red ≈ 0.299 * u64::MAX
        assert!(mono64_val(&p) > 0);
    }

    #[test]
    fn luminance_rgbaf64() {
        let p: MonoF64 = Luminance.convert(&RgbaF64::new(1.0, 0.0, 0.0, 0.5));
        assert!((p.0 - 0.299).abs() < 1e-10);
    }

    #[test]
    fn luminance_bgraf64() {
        let p: MonoF64 = Luminance.convert(&BgraF64::new(0.0, 0.0, 1.0, 1.0));
        assert!((p.0 - 0.299).abs() < 1e-10);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Broadcast — 32/64-bit and f64
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn broadcast_mono32_to_rgb32() {
        let p: Rgb32 = Broadcast.convert(&Mono32::new(42));
        assert_eq!(p, Rgb32::new(42, 42, 42));
    }

    #[test]
    fn broadcast_mono64_to_rgb64() {
        let p: Rgb64 = Broadcast.convert(&Mono64::new(42));
        assert_eq!(p, Rgb64::new(42, 42, 42));
    }

    #[test]
    fn broadcast_mono32_to_bgr32() {
        let p: Bgr32 = Broadcast.convert(&Mono32::new(42));
        assert_eq!(p, Bgr32::new(42, 42, 42));
    }

    #[test]
    fn broadcast_mono64_to_bgr64() {
        let p: Bgr64 = Broadcast.convert(&Mono64::new(42));
        assert_eq!(p, Bgr64::new(42, 42, 42));
    }

    #[test]
    fn broadcast_f64_to_rgbf64() {
        let p: RgbF64 = Broadcast.convert(&MonoF64::new(0.5));
        assert!((p.r - 0.5).abs() < 1e-12);
        assert!((p.g - 0.5).abs() < 1e-12);
        assert!((p.b - 0.5).abs() < 1e-12);
    }

    #[test]
    fn broadcast_f64_to_bgrf64() {
        let p: BgrF64 = Broadcast.convert(&MonoF64::new(0.5));
        assert!((p.b - 0.5).abs() < 1e-12);
        assert!((p.g - 0.5).abs() < 1e-12);
        assert!((p.r - 0.5).abs() < 1e-12);
    }

    #[test]
    fn broadcast_mono32_to_rgba32() {
        let p: Rgba32 = Broadcast.convert(&Mono32::new(42));
        assert_eq!(p, Rgba32::new(42, 42, 42, u32::MAX));
    }

    #[test]
    fn broadcast_mono64_to_rgba64() {
        let p: Rgba64 = Broadcast.convert(&Mono64::new(42));
        assert_eq!(p, Rgba64::new(42, 42, 42, u64::MAX));
    }

    #[test]
    fn broadcast_mono64_to_bgra64() {
        let p: Bgra64 = Broadcast.convert(&Mono64::new(42));
        assert_eq!(p, Bgra64::new(42, 42, 42, u64::MAX));
    }

    #[test]
    fn broadcast_f64_to_rgbaf64() {
        let p: RgbaF64 = Broadcast.convert(&MonoF64::new(0.5));
        assert!((p.r - 0.5).abs() < 1e-12);
        assert!((p.a - 1.0).abs() < 1e-12);
    }

    #[test]
    fn broadcast_f64_to_bgraf64() {
        let p: BgraF64 = Broadcast.convert(&MonoF64::new(0.5));
        assert!((p.b - 0.5).abs() < 1e-12);
        assert!((p.a - 1.0).abs() < 1e-12);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // ColorSwap — 32/64-bit and f64
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn colorswap_rgb32_to_bgr32() {
        let p: Bgr32 = ColorSwap.convert(&Rgb32::new(1, 2, 3));
        assert_eq!(p, Bgr32::new(3, 2, 1));
    }

    #[test]
    fn colorswap_bgr32_to_rgb32() {
        let p: Rgb32 = ColorSwap.convert(&Bgr32::new(3, 2, 1));
        assert_eq!(p, Rgb32::new(1, 2, 3));
    }

    #[test]
    fn colorswap_rgb64_to_bgr64() {
        let p: Bgr64 = ColorSwap.convert(&Rgb64::new(1, 2, 3));
        assert_eq!(p, Bgr64::new(3, 2, 1));
    }

    #[test]
    fn colorswap_bgr64_to_rgb64() {
        let p: Rgb64 = ColorSwap.convert(&Bgr64::new(3, 2, 1));
        assert_eq!(p, Rgb64::new(1, 2, 3));
    }

    #[test]
    fn colorswap_rgbf64_to_bgrf64() {
        let p: BgrF64 = ColorSwap.convert(&RgbF64::new(0.1, 0.2, 0.3));
        assert!((p.b - 0.3).abs() < 1e-12);
        assert!((p.r - 0.1).abs() < 1e-12);
    }

    #[test]
    fn colorswap_bgrf64_to_rgbf64() {
        let p: RgbF64 = ColorSwap.convert(&BgrF64::new(0.3, 0.2, 0.1));
        assert!((p.r - 0.1).abs() < 1e-12);
        assert!((p.b - 0.3).abs() < 1e-12);
    }

    #[test]
    fn colorswap_rgba64_to_bgra64() {
        let p: Bgra64 = ColorSwap.convert(&Rgba64::new(1, 2, 3, 4));
        assert_eq!(p, Bgra64::new(3, 2, 1, 4));
    }

    #[test]
    fn colorswap_bgra64_to_rgba64() {
        let p: Rgba64 = ColorSwap.convert(&Bgra64::new(3, 2, 1, 4));
        assert_eq!(p, Rgba64::new(1, 2, 3, 4));
    }

    #[test]
    fn colorswap_rgbaf64_to_bgraf64() {
        let p: BgraF64 = ColorSwap.convert(&RgbaF64::new(0.1, 0.2, 0.3, 0.4));
        assert!((p.b - 0.3).abs() < 1e-12);
        assert!((p.r - 0.1).abs() < 1e-12);
        assert!((p.a - 0.4).abs() < 1e-12);
    }

    #[test]
    fn colorswap_bgraf64_to_rgbaf64() {
        let p: RgbaF64 = ColorSwap.convert(&BgraF64::new(0.3, 0.2, 0.1, 0.4));
        assert!((p.r - 0.1).abs() < 1e-12);
        assert!((p.b - 0.3).abs() < 1e-12);
        assert!((p.a - 0.4).abs() < 1e-12);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // AddAlpha — 32/64-bit and f64
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn addalpha_rgb32_to_rgba32() {
        let p: Rgba32 = AddAlpha.convert(&Rgb32::new(1, 2, 3));
        assert_eq!(p, Rgba32::new(1, 2, 3, u32::MAX));
    }

    #[test]
    fn addalpha_rgb64_to_rgba64() {
        let p: Rgba64 = AddAlpha.convert(&Rgb64::new(1, 2, 3));
        assert_eq!(p, Rgba64::new(1, 2, 3, u64::MAX));
    }

    #[test]
    fn addalpha_rgbf64_to_rgbaf64() {
        let p: RgbaF64 = AddAlpha.convert(&RgbF64::new(0.1, 0.2, 0.3));
        assert!((p.r - 0.1).abs() < 1e-12);
        assert!((p.a - 1.0).abs() < 1e-12);
    }

    #[test]
    fn addalpha_bgr32_to_bgra32() {
        let p: Bgra32 = AddAlpha.convert(&Bgr32::new(1, 2, 3));
        assert_eq!(p, Bgra32::new(1, 2, 3, u32::MAX));
    }

    #[test]
    fn addalpha_bgr64_to_bgra64() {
        let p: Bgra64 = AddAlpha.convert(&Bgr64::new(1, 2, 3));
        assert_eq!(p, Bgra64::new(1, 2, 3, u64::MAX));
    }

    #[test]
    fn addalpha_bgrf64_to_bgraf64() {
        let p: BgraF64 = AddAlpha.convert(&BgrF64::new(0.1, 0.2, 0.3));
        assert!((p.b - 0.1).abs() < 1e-12);
        assert!((p.a - 1.0).abs() < 1e-12);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Helper verification — new full-range functions
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn helper_fr_u8_to_u32_properties() {
        assert_eq!(fr_u8_to_u32(0), 0);
        assert_eq!(fr_u8_to_u32(255), u32::MAX);
        assert_eq!(fr_u8_to_u32(1), 0x01010101);
        assert_eq!(fr_u8_to_u32(128), 128 * 0x01010101);
    }

    #[test]
    fn helper_fr_u32_to_u8_properties() {
        assert_eq!(fr_u32_to_u8(0), 0);
        assert_eq!(fr_u32_to_u8(u32::MAX), 255);
    }

    #[test]
    fn helper_fr_u8_u32_roundtrip_exhaustive() {
        for v in 0..=255u8 {
            assert_eq!(fr_u32_to_u8(fr_u8_to_u32(v)), v, "failed at {v}");
        }
    }

    #[test]
    fn helper_fr_u8_to_u64_properties() {
        assert_eq!(fr_u8_to_u64(0), 0);
        assert_eq!(fr_u8_to_u64(255), u64::MAX);
        assert_eq!(fr_u8_to_u64(1), 0x0101010101010101);
    }

    #[test]
    fn helper_fr_u64_to_u8_properties() {
        assert_eq!(fr_u64_to_u8(0), 0);
        assert_eq!(fr_u64_to_u8(u64::MAX), 255);
    }

    #[test]
    fn helper_fr_u8_u64_roundtrip_exhaustive() {
        for v in 0..=255u8 {
            assert_eq!(fr_u64_to_u8(fr_u8_to_u64(v)), v, "failed at {v}");
        }
    }

    #[test]
    fn helper_fr_u16_u32_roundtrip() {
        for &v in &[0u16, 1, 128, 255, 32768, 65535] {
            assert_eq!(fr_u32_to_u16(fr_u16_to_u32(v)), v, "failed at {v}");
        }
    }

    #[test]
    fn helper_fr_u16_u64_roundtrip() {
        for &v in &[0u16, 1, 128, 255, 32768, 65535] {
            assert_eq!(fr_u64_to_u16(fr_u16_to_u64(v)), v, "failed at {v}");
        }
    }

    #[test]
    fn helper_fr_u32_u64_roundtrip() {
        for &v in &[0u32, 1, u32::MAX / 2, u32::MAX] {
            assert_eq!(fr_u64_to_u32(fr_u32_to_u64(v)), v, "failed at {v}");
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Then — chains involving 64-bit types
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn then_colorswap_fullrange_rgb8_to_bgr64() {
        let method = ColorSwap.then::<Bgr8, _>(FullRange);
        let p: Bgr64 = method.convert(&Rgb8::new(255, 0, 0));
        assert_eq!(p.r.0, u64::MAX);
        assert_eq!(p.b.0, 0);
    }

    #[test]
    fn then_luminance_fullrange_rgb64_to_mono8() {
        let method = Luminance.then::<Mono64, _>(FullRange);
        let p: Mono8 = method.convert(&Rgb64::new(u64::MAX, u64::MAX, u64::MAX));
        assert_eq!(p, Mono8::new(255));
    }

    #[test]
    fn then_broadcast_fullrange_mono8_to_rgb64() {
        let method = Broadcast.then::<Rgb8, _>(FullRange);
        let p: Rgb64 = method.convert(&Mono8::new(255));
        assert_eq!(p, Rgb64::new(u64::MAX, u64::MAX, u64::MAX));
    }

    #[test]
    fn then_addalpha_fullrange_rgb64_to_rgba64() {
        let p: Rgba64 = AddAlpha.convert(&Rgb64::new(100, 200, 300));
        assert_eq!(p, Rgba64::new(100, 200, 300, u64::MAX));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // MonoA family — FullRange depth conversion tests
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn fullrange_monoa8_to_monoa16_extremes() {
        let black: MonoA16 = FullRange.convert(&MonoA8::new(0, 0));
        assert_eq!(black, MonoA16::new(0, 0));
        let white: MonoA16 = FullRange.convert(&MonoA8::new(255, 255));
        assert_eq!(white, MonoA16::new(65535, 65535));
    }

    #[test]
    fn fullrange_monoa16_to_monoa8_extremes() {
        let black: MonoA8 = FullRange.convert(&MonoA16::new(0, 0));
        assert_eq!(black, MonoA8::new(0, 0));
        let white: MonoA8 = FullRange.convert(&MonoA16::new(65535, 65535));
        assert_eq!(white, MonoA8::new(255, 255));
    }

    #[test]
    fn fullrange_monoa8_monoa16_roundtrip() {
        let orig = MonoA8::new(128, 64);
        let wide: MonoA16 = FullRange.convert(&orig);
        let back: MonoA8 = FullRange.convert(&wide);
        assert_eq!(back, orig);
    }

    #[test]
    fn fullrange_monoa8_to_monoaf32() {
        let p: MonoAF32 = FullRange.convert(&MonoA8::new(0, 255));
        assert!(p.v.abs() < 1e-6);
        assert!((p.a - 1.0).abs() < 1e-6);
    }

    #[test]
    fn fullrange_monoaf32_to_monoa8() {
        let p: MonoA8 = FullRange.convert(&MonoAF32::new(0.0, 1.0));
        assert_eq!(p, MonoA8::new(0, 255));
    }

    #[test]
    fn fullrange_monoa16_to_monoaf32() {
        let p: MonoAF32 = FullRange.convert(&MonoA16::new(0, 65535));
        assert!(p.v.abs() < 1e-4);
        assert!((p.a - 1.0).abs() < 1e-4);
    }

    #[test]
    fn fullrange_monoaf32_to_monoa16() {
        let p: MonoA16 = FullRange.convert(&MonoAF32::new(1.0, 0.0));
        assert_eq!(p, MonoA16::new(65535, 0));
    }

    #[test]
    fn fullrange_monoa8_to_monoa32_extremes() {
        let white: MonoA32 = FullRange.convert(&MonoA8::new(255, 255));
        assert_eq!(white, MonoA32::new(u32::MAX, u32::MAX));
    }

    #[test]
    fn fullrange_monoa32_to_monoa8_extremes() {
        let white: MonoA8 = FullRange.convert(&MonoA32::new(u32::MAX, u32::MAX));
        assert_eq!(white, MonoA8::new(255, 255));
    }

    #[test]
    fn fullrange_monoa8_to_monoa64_extremes() {
        let white: MonoA64 = FullRange.convert(&MonoA8::new(255, 255));
        assert_eq!(white, MonoA64::new(u64::MAX, u64::MAX));
    }

    #[test]
    fn fullrange_monoa64_to_monoa8_extremes() {
        let white: MonoA8 = FullRange.convert(&MonoA64::new(u64::MAX, u64::MAX));
        assert_eq!(white, MonoA8::new(255, 255));
    }

    #[test]
    fn fullrange_monoa16_to_monoa32_extremes() {
        let white: MonoA32 = FullRange.convert(&MonoA16::new(65535, 65535));
        assert_eq!(white, MonoA32::new(u32::MAX, u32::MAX));
    }

    #[test]
    fn fullrange_monoa32_to_monoa16_extremes() {
        let white: MonoA16 = FullRange.convert(&MonoA32::new(u32::MAX, u32::MAX));
        assert_eq!(white, MonoA16::new(65535, 65535));
    }

    #[test]
    fn fullrange_monoa16_to_monoa64_extremes() {
        let white: MonoA64 = FullRange.convert(&MonoA16::new(65535, 65535));
        assert_eq!(white, MonoA64::new(u64::MAX, u64::MAX));
    }

    #[test]
    fn fullrange_monoa64_to_monoa16_extremes() {
        let white: MonoA16 = FullRange.convert(&MonoA64::new(u64::MAX, u64::MAX));
        assert_eq!(white, MonoA16::new(65535, 65535));
    }

    #[test]
    fn fullrange_monoa32_to_monoa64_extremes() {
        let white: MonoA64 = FullRange.convert(&MonoA32::new(u32::MAX, u32::MAX));
        assert_eq!(white, MonoA64::new(u64::MAX, u64::MAX));
    }

    #[test]
    fn fullrange_monoa64_to_monoa32_extremes() {
        let white: MonoA32 = FullRange.convert(&MonoA64::new(u64::MAX, u64::MAX));
        assert_eq!(white, MonoA32::new(u32::MAX, u32::MAX));
    }

    #[test]
    fn fullrange_monoa32_to_monoaf32() {
        let p: MonoAF32 = FullRange.convert(&MonoA32::new(u32::MAX, 0));
        assert!((p.v - 1.0).abs() < 1e-6);
        assert!(p.a.abs() < 1e-6);
    }

    #[test]
    fn fullrange_monoaf32_to_monoa32() {
        let p: MonoA32 = FullRange.convert(&MonoAF32::new(1.0, 0.0));
        assert_eq!(p, MonoA32::new(u32::MAX, 0));
    }

    #[test]
    fn fullrange_monoa64_to_monoaf32() {
        let p: MonoAF32 = FullRange.convert(&MonoA64::new(u64::MAX, 0));
        assert!((p.v - 1.0).abs() < 1e-6);
        assert!(p.a.abs() < 1e-6);
    }

    #[test]
    fn fullrange_monoaf32_to_monoa64() {
        let p: MonoA64 = FullRange.convert(&MonoAF32::new(1.0, 0.0));
        assert_eq!(p, MonoA64::new(u64::MAX, 0));
    }

    #[test]
    fn fullrange_monoa8_to_monoaf64() {
        let p: MonoAF64 = FullRange.convert(&MonoA8::new(255, 0));
        assert!((p.v - 1.0).abs() < 1e-6);
        assert!(p.a.abs() < 1e-6);
    }

    #[test]
    fn fullrange_monoaf64_to_monoa8() {
        let p: MonoA8 = FullRange.convert(&MonoAF64::new(1.0, 0.0));
        assert_eq!(p, MonoA8::new(255, 0));
    }

    #[test]
    fn fullrange_monoa16_to_monoaf64() {
        let p: MonoAF64 = FullRange.convert(&MonoA16::new(65535, 0));
        assert!((p.v - 1.0).abs() < 1e-6);
    }

    #[test]
    fn fullrange_monoaf64_to_monoa16() {
        let p: MonoA16 = FullRange.convert(&MonoAF64::new(1.0, 0.5));
        assert_eq!(p.v, Saturating(65535u16));
    }

    #[test]
    fn fullrange_monoa32_to_monoaf64() {
        let p: MonoAF64 = FullRange.convert(&MonoA32::new(u32::MAX, 0));
        assert!((p.v - 1.0).abs() < 1e-9);
    }

    #[test]
    fn fullrange_monoaf64_to_monoa32() {
        let p: MonoA32 = FullRange.convert(&MonoAF64::new(1.0, 0.0));
        assert_eq!(p, MonoA32::new(u32::MAX, 0));
    }

    #[test]
    fn fullrange_monoa64_to_monoaf64() {
        let p: MonoAF64 = FullRange.convert(&MonoA64::new(u64::MAX, 0));
        assert!((p.v - 1.0).abs() < 1e-9);
    }

    #[test]
    fn fullrange_monoaf64_to_monoa64() {
        let p: MonoA64 = FullRange.convert(&MonoAF64::new(1.0, 0.0));
        assert_eq!(p, MonoA64::new(u64::MAX, 0));
    }

    #[test]
    fn fullrange_monoaf32_to_monoaf64() {
        let p: MonoAF64 = FullRange.convert(&MonoAF32::new(0.5, 0.25));
        assert!((p.v - 0.5).abs() < 1e-6);
        assert!((p.a - 0.25).abs() < 1e-6);
    }

    #[test]
    fn fullrange_monoaf64_to_monoaf32() {
        let p: MonoAF32 = FullRange.convert(&MonoAF64::new(0.5, 0.25));
        assert!((p.v - 0.5).abs() < 1e-6);
        assert!((p.a - 0.25).abs() < 1e-6);
    }

    #[test]
    fn fullrange_monoa8_monoa32_roundtrip() {
        let orig = MonoA8::new(128, 64);
        let wide: MonoA32 = FullRange.convert(&orig);
        let back: MonoA8 = FullRange.convert(&wide);
        assert_eq!(back, orig);
    }

    #[test]
    fn fullrange_monoa8_monoa64_roundtrip() {
        let orig = MonoA8::new(128, 64);
        let wide: MonoA64 = FullRange.convert(&orig);
        let back: MonoA8 = FullRange.convert(&wide);
        assert_eq!(back, orig);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // MonoA family — Narrow depth conversion tests
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn clamp_monoa8_to_monoa16() {
        let p: MonoA16 = Narrow.convert(&MonoA8::new(200, 100));
        assert_eq!(p, MonoA16::new(200, 100));
    }

    #[test]
    fn clamp_monoa16_to_monoa8_fits() {
        let p: MonoA8 = Narrow.convert(&MonoA16::new(200, 100));
        assert_eq!(p, MonoA8::new(200, 100));
    }

    #[test]
    fn clamp_monoa16_to_monoa8_clamps() {
        let p: MonoA8 = Narrow.convert(&MonoA16::new(1000, 500));
        assert_eq!(p, MonoA8::new(255, 255));
    }

    #[test]
    fn clamp_monoa8_to_monoa32() {
        let p: MonoA32 = Narrow.convert(&MonoA8::new(200, 100));
        assert_eq!(p, MonoA32::new(200, 100));
    }

    #[test]
    fn clamp_monoa32_to_monoa8_clamps() {
        let p: MonoA8 = Narrow.convert(&MonoA32::new(1000, 500));
        assert_eq!(p, MonoA8::new(255, 255));
    }

    #[test]
    fn clamp_monoa16_to_monoa32() {
        let p: MonoA32 = Narrow.convert(&MonoA16::new(60000, 30000));
        assert_eq!(p, MonoA32::new(60000, 30000));
    }

    #[test]
    fn clamp_monoa32_to_monoa16_fits() {
        let p: MonoA16 = Narrow.convert(&MonoA32::new(60000, 30000));
        assert_eq!(p, MonoA16::new(60000, 30000));
    }

    #[test]
    fn clamp_monoa32_to_monoa16_clamps() {
        let p: MonoA16 = Narrow.convert(&MonoA32::new(100_000, 200_000));
        assert_eq!(p, MonoA16::new(65535, 65535));
    }

    #[test]
    fn clamp_monoa8_to_monoa64() {
        let p: MonoA64 = Narrow.convert(&MonoA8::new(200, 100));
        assert_eq!(p, MonoA64::new(200, 100));
    }

    #[test]
    fn clamp_monoa64_to_monoa8_clamps() {
        let p: MonoA8 = Narrow.convert(&MonoA64::new(1000, 500));
        assert_eq!(p, MonoA8::new(255, 255));
    }

    #[test]
    fn clamp_monoa16_to_monoa64() {
        let p: MonoA64 = Narrow.convert(&MonoA16::new(60000, 30000));
        assert_eq!(p, MonoA64::new(60000, 30000));
    }

    #[test]
    fn clamp_monoa64_to_monoa16_clamps() {
        let p: MonoA16 = Narrow.convert(&MonoA64::new(100_000, 200_000));
        assert_eq!(p, MonoA16::new(65535, 65535));
    }

    #[test]
    fn clamp_monoa32_to_monoa64() {
        let p: MonoA64 = Narrow.convert(&MonoA32::new(u32::MAX, 100));
        assert_eq!(p, MonoA64::new(u32::MAX as u64, 100));
    }

    #[test]
    fn clamp_monoa64_to_monoa32_fits() {
        let p: MonoA32 = Narrow.convert(&MonoA64::new(100, 200));
        assert_eq!(p, MonoA32::new(100, 200));
    }

    #[test]
    fn clamp_monoa64_to_monoa32_clamps() {
        let p: MonoA32 = Narrow.convert(&MonoA64::new(u64::MAX, u64::MAX));
        assert_eq!(p, MonoA32::new(u32::MAX, u32::MAX));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // AddAlpha — Mono → MonoA tests
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn addalpha_mono8_to_monoa8() {
        let p: MonoA8 = AddAlpha.convert(&Mono8::new(128));
        assert_eq!(p, MonoA8::new(128, 255));
    }

    #[test]
    fn addalpha_mono8_to_monoa8_black() {
        let p: MonoA8 = AddAlpha.convert(&Mono8::new(0));
        assert_eq!(p, MonoA8::new(0, 255));
    }

    #[test]
    fn addalpha_mono8_to_monoa8_white() {
        let p: MonoA8 = AddAlpha.convert(&Mono8::new(255));
        assert_eq!(p, MonoA8::new(255, 255));
    }

    #[test]
    fn addalpha_mono16_to_monoa16() {
        let p: MonoA16 = AddAlpha.convert(&Mono16::new(1000));
        assert_eq!(p, MonoA16::new(1000, u16::MAX));
    }

    #[test]
    fn addalpha_mono32_to_monoa32() {
        let p: MonoA32 = AddAlpha.convert(&Mono32::new(100_000));
        assert_eq!(p, MonoA32::new(100_000, u32::MAX));
    }

    #[test]
    fn addalpha_mono64_to_monoa64() {
        let p: MonoA64 = AddAlpha.convert(&Mono64::new(1_000_000));
        assert_eq!(p, MonoA64::new(1_000_000, u64::MAX));
    }

    #[test]
    fn addalpha_f32_to_monoaf32() {
        let p: MonoAF32 = AddAlpha.convert(&MonoF32::new(0.5));
        assert_eq!(p, MonoAF32::new(0.5, 1.0));
    }

    #[test]
    fn addalpha_f64_to_monoaf64() {
        let p: MonoAF64 = AddAlpha.convert(&MonoF64::new(0.25));
        assert_eq!(p, MonoAF64::new(0.25, 1.0));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Broadcast — MonoA → RGBA tests
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn broadcast_monoa8_to_rgba8() {
        let p: Rgba8 = Broadcast.convert(&MonoA8::new(128, 200));
        assert_eq!(p, Rgba8::new(128, 128, 128, 200));
    }

    #[test]
    fn broadcast_monoa8_to_rgba8_opaque() {
        let p: Rgba8 = Broadcast.convert(&MonoA8::new(255, 255));
        assert_eq!(p, Rgba8::new(255, 255, 255, 255));
    }

    #[test]
    fn broadcast_monoa8_to_rgba8_transparent() {
        let p: Rgba8 = Broadcast.convert(&MonoA8::new(128, 0));
        assert_eq!(p, Rgba8::new(128, 128, 128, 0));
    }

    #[test]
    fn broadcast_monoa16_to_rgba16() {
        let p: Rgba16 = Broadcast.convert(&MonoA16::new(1000, 2000));
        assert_eq!(p, Rgba16::new(1000, 1000, 1000, 2000));
    }

    #[test]
    fn broadcast_monoa32_to_rgba32() {
        let p: Rgba32 = Broadcast.convert(&MonoA32::new(100_000, 200_000));
        assert_eq!(p, Rgba32::new(100_000, 100_000, 100_000, 200_000));
    }

    #[test]
    fn broadcast_monoa64_to_rgba64() {
        let p: Rgba64 = Broadcast.convert(&MonoA64::new(42, 99));
        assert_eq!(p, Rgba64::new(42, 42, 42, 99));
    }

    #[test]
    fn broadcast_monoaf32_to_rgbaf32() {
        let p: RgbaF32 = Broadcast.convert(&MonoAF32::new(0.5, 0.8));
        assert_eq!(p, RgbaF32::new(0.5, 0.5, 0.5, 0.8));
    }

    #[test]
    fn broadcast_monoaf64_to_rgbaf64() {
        let p: RgbaF64 = Broadcast.convert(&MonoAF64::new(0.25, 0.75));
        assert_eq!(p, RgbaF64::new(0.25, 0.25, 0.25, 0.75));
    }

    // ── Broadcast — MonoA → BGRA tests ──────────────────────────────────

    #[test]
    fn broadcast_monoa8_to_bgra8() {
        let p: Bgra8 = Broadcast.convert(&MonoA8::new(128, 200));
        assert_eq!(p, Bgra8::new(128, 128, 128, 200));
    }

    #[test]
    fn broadcast_monoa16_to_bgra16() {
        let p: Bgra16 = Broadcast.convert(&MonoA16::new(1000, 2000));
        assert_eq!(p, Bgra16::new(1000, 1000, 1000, 2000));
    }

    #[test]
    fn broadcast_monoa32_to_bgra32() {
        let p: Bgra32 = Broadcast.convert(&MonoA32::new(100_000, 200_000));
        assert_eq!(p, Bgra32::new(100_000, 100_000, 100_000, 200_000));
    }

    #[test]
    fn broadcast_monoa64_to_bgra64() {
        let p: Bgra64 = Broadcast.convert(&MonoA64::new(42, 99));
        assert_eq!(p, Bgra64::new(42, 42, 42, 99));
    }

    #[test]
    fn broadcast_monoaf32_to_bgraf32() {
        let p: BgraF32 = Broadcast.convert(&MonoAF32::new(0.5, 0.8));
        assert_eq!(p, BgraF32::new(0.5, 0.5, 0.5, 0.8));
    }

    #[test]
    fn broadcast_monoaf64_to_bgraf64() {
        let p: BgraF64 = Broadcast.convert(&MonoAF64::new(0.25, 0.75));
        assert_eq!(p, BgraF64::new(0.25, 0.25, 0.25, 0.75));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Luminance — RGBA → MonoA tests (alpha preserved)
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn luminance_rgba8_to_monoa8_white() {
        let p: MonoA8 = Luminance.convert(&Rgba8::new(255, 255, 255, 128));
        assert_eq!(p.v, Saturating(255u8));
        assert_eq!(p.a, Saturating(128u8));
    }

    #[test]
    fn luminance_rgba8_to_monoa8_black() {
        let p: MonoA8 = Luminance.convert(&Rgba8::new(0, 0, 0, 200));
        assert_eq!(p.v, Saturating(0u8));
        assert_eq!(p.a, Saturating(200u8));
    }

    #[test]
    fn luminance_rgba8_to_monoa8_pure_red() {
        let p: MonoA8 = Luminance.convert(&Rgba8::new(255, 0, 0, 255));
        // BT.601: (77*255 + 128) >> 8 = 77
        assert_eq!(p.v.0, 77);
        assert_eq!(p.a.0, 255);
    }

    #[test]
    fn luminance_rgba8_to_monoa8_preserves_alpha() {
        let p: MonoA8 = Luminance.convert(&Rgba8::new(100, 100, 100, 42));
        assert_eq!(p.a.0, 42);
    }

    #[test]
    fn luminance_rgba16_to_monoa16() {
        let p: MonoA16 = Luminance.convert(&Rgba16::new(65535, 65535, 65535, 1000));
        assert_eq!(p.v.0, 65535);
        assert_eq!(p.a.0, 1000);
    }

    #[test]
    fn luminance_rgba32_to_monoa32() {
        let p: MonoA32 = Luminance.convert(&Rgba32::new(u32::MAX, u32::MAX, u32::MAX, 42));
        assert_eq!(p.v.0, u32::MAX);
        assert_eq!(p.a.0, 42);
    }

    #[test]
    fn luminance_rgba64_to_monoa64() {
        let p: MonoA64 = Luminance.convert(&Rgba64::new(u64::MAX, u64::MAX, u64::MAX, 99));
        assert_eq!(p.v.0, u64::MAX);
        assert_eq!(p.a.0, 99);
    }

    #[test]
    fn luminance_rgbaf32_to_monoaf32() {
        let p: MonoAF32 = Luminance.convert(&RgbaF32::new(1.0, 1.0, 1.0, 0.5));
        assert!((p.v - 1.0).abs() < 1e-4);
        assert!((p.a - 0.5).abs() < 1e-6);
    }

    #[test]
    fn luminance_rgbaf64_to_monoaf64() {
        let p: MonoAF64 = Luminance.convert(&RgbaF64::new(1.0, 1.0, 1.0, 0.25));
        assert!((p.v - 1.0).abs() < 1e-9);
        assert!((p.a - 0.25).abs() < 1e-9);
    }

    // ── Luminance — BGRA → MonoA tests (alpha preserved) ────────────────

    #[test]
    fn luminance_bgra8_to_monoa8() {
        let p: MonoA8 = Luminance.convert(&Bgra8::new(0, 0, 255, 128));
        // BT.601: pure red (r=255) → 77
        assert_eq!(p.v.0, 77);
        assert_eq!(p.a.0, 128);
    }

    #[test]
    fn luminance_bgra8_to_monoa8_white() {
        let p: MonoA8 = Luminance.convert(&Bgra8::new(255, 255, 255, 42));
        assert_eq!(p.v.0, 255);
        assert_eq!(p.a.0, 42);
    }

    #[test]
    fn luminance_bgra16_to_monoa16() {
        let p: MonoA16 = Luminance.convert(&Bgra16::new(65535, 65535, 65535, 1000));
        assert_eq!(p.v.0, 65535);
        assert_eq!(p.a.0, 1000);
    }

    #[test]
    fn luminance_bgra32_to_monoa32() {
        let p: MonoA32 = Luminance.convert(&Bgra32::new(u32::MAX, u32::MAX, u32::MAX, 42));
        assert_eq!(p.v.0, u32::MAX);
        assert_eq!(p.a.0, 42);
    }

    #[test]
    fn luminance_bgra64_to_monoa64() {
        let p: MonoA64 = Luminance.convert(&Bgra64::new(u64::MAX, u64::MAX, u64::MAX, 99));
        assert_eq!(p.v.0, u64::MAX);
        assert_eq!(p.a.0, 99);
    }

    #[test]
    fn luminance_bgraf32_to_monoaf32() {
        let p: MonoAF32 = Luminance.convert(&BgraF32::new(1.0, 1.0, 1.0, 0.5));
        assert!((p.v - 1.0).abs() < 1e-4);
        assert!((p.a - 0.5).abs() < 1e-6);
    }

    #[test]
    fn luminance_bgraf64_to_monoaf64() {
        let p: MonoAF64 = Luminance.convert(&BgraF64::new(1.0, 1.0, 1.0, 0.25));
        assert!((p.v - 1.0).abs() < 1e-9);
        assert!((p.a - 0.25).abs() < 1e-9);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Luminance roundtrip: RGBA → MonoA → RGBA (via Broadcast)
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn luminance_broadcast_roundtrip_gray_preserves_alpha() {
        // A gray RGBA pixel should roundtrip through MonoA losslessly.
        let orig = Rgba8::new(100, 100, 100, 42);
        let mono: MonoA8 = Luminance.convert(&orig);
        let back: Rgba8 = Broadcast.convert(&mono);
        // Gray input => v == 100, alpha preserved
        assert_eq!(mono.v.0, 100);
        assert_eq!(back, Rgba8::new(100, 100, 100, 42));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Then combinator — MonoA cross-depth and cross-strategy
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn then_addalpha_fullrange_mono8_to_monoa16() {
        let method = AddAlpha.then::<MonoA8, _>(FullRange);
        let p: MonoA16 = method.convert(&Mono8::new(255));
        assert_eq!(p, MonoA16::new(65535, 65535));
    }

    #[test]
    fn then_fullrange_addalpha_mono8_to_monoa16() {
        let method = FullRange.then::<Mono16, _>(AddAlpha);
        let p: MonoA16 = method.convert(&Mono8::new(255));
        assert_eq!(p, MonoA16::new(65535, 65535));
    }

    #[test]
    fn then_luminance_fullrange_rgba8_to_monoa16() {
        let method = Luminance.then::<MonoA8, _>(FullRange);
        let p: MonoA16 = method.convert(&Rgba8::new(255, 255, 255, 128));
        assert_eq!(p.v.0, 65535);
        // alpha: 128 FullRange→ 128 * 65535 / 255 = ~32896
        let alpha_16: MonoA16 = FullRange.convert(&MonoA8::new(255, 128));
        assert_eq!(p.a, alpha_16.a);
    }

    #[test]
    fn then_broadcast_fullrange_monoa8_to_rgba16() {
        let method = Broadcast.then::<Rgba8, _>(FullRange);
        let p: Rgba16 = method.convert(&MonoA8::new(255, 128));
        assert_eq!(p.r.0, 65535);
        assert_eq!(p.g.0, 65535);
        assert_eq!(p.b.0, 65535);
    }

    #[test]
    fn then_monoa_fullrange_broadcast_to_bgra() {
        let method = FullRange.then::<MonoA16, _>(Broadcast);
        let p: Bgra16 = method.convert(&MonoA8::new(128, 255));
        let expected: MonoA16 = FullRange.convert(&MonoA8::new(128, 255));
        assert_eq!(p.b.0, expected.v.0);
        assert_eq!(p.g.0, expected.v.0);
        assert_eq!(p.r.0, expected.v.0);
        assert_eq!(p.a.0, expected.a.0);
    }

    #[test]
    fn then_addalpha_broadcast_mono8_to_rgba8() {
        // Mono8 → MonoA8 (AddAlpha) → Rgba8 (Broadcast)
        let method = AddAlpha.then::<MonoA8, _>(Broadcast);
        let p: Rgba8 = method.convert(&Mono8::new(200));
        assert_eq!(p, Rgba8::new(200, 200, 200, 255));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // convert_image / convert_image_into — MonoA
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn convert_image_monoa8_to_monoa16_fullrange() {
        let img = Image::fill(3, 2, MonoA8::new(200, 100));
        let out: Image<MonoA16> = convert_image(&img, FullRange);
        let expected: MonoA16 = FullRange.convert(&MonoA8::new(200, 100));
        assert_eq!(out.pixel_at(0, 0), expected);
        assert_eq!(out.pixel_at(2, 1), expected);
    }

    #[test]
    fn convert_image_rgba8_to_monoa8_luminance() {
        let img = Image::fill(2, 2, Rgba8::new(255, 255, 255, 128));
        let out: Image<MonoA8> = convert_image(&img, Luminance);
        assert_eq!(out.pixel_at(0, 0), MonoA8::new(255, 128));
    }

    #[test]
    fn convert_image_monoa8_to_rgba8_broadcast() {
        let img = Image::fill(2, 2, MonoA8::new(100, 200));
        let out: Image<Rgba8> = convert_image(&img, Broadcast);
        assert_eq!(out.pixel_at(0, 0), Rgba8::new(100, 100, 100, 200));
    }

    #[test]
    fn convert_image_mono8_to_monoa8_addalpha() {
        let img = Image::fill(2, 2, Mono8::new(128));
        let out: Image<MonoA8> = convert_image(&img, AddAlpha);
        assert_eq!(out.pixel_at(0, 0), MonoA8::new(128, 255));
    }

    #[test]
    fn convert_image_into_monoa8_to_monoa16() {
        let src = Image::fill(3, 2, MonoA8::new(42, 128));
        let mut dst = Image::fill(3, 2, MonoA16::new(0, 0));
        convert_image_into(&src, &mut dst, FullRange);
        let expected: MonoA16 = FullRange.convert(&MonoA8::new(42, 128));
        assert_eq!(dst.pixel_at(0, 0), expected);
        assert_eq!(dst.pixel_at(2, 1), expected);
    }

    #[test]
    fn convert_image_monoa_clamp() {
        let img = Image::fill(2, 2, MonoA16::new(1000, 500));
        let out: Image<MonoA8> = convert_image(&img, Narrow);
        assert_eq!(out.pixel_at(0, 0), MonoA8::new(255, 255));
    }

    #[test]
    fn convert_image_monoa_varying_pixels() {
        use crate::image::{Image, ImageViewMut};
        let mut img = Image::fill(2, 2, MonoA8::new(0, 0));
        *img.pixel_at_mut(0, 0) = MonoA8::new(10, 20);
        *img.pixel_at_mut(1, 0) = MonoA8::new(30, 40);
        *img.pixel_at_mut(0, 1) = MonoA8::new(50, 60);
        *img.pixel_at_mut(1, 1) = MonoA8::new(70, 80);
        let out: Image<MonoA16> = convert_image(&img, FullRange);
        assert_eq!(out.pixel_at(0, 0), FullRange.convert(&MonoA8::new(10, 20)));
        assert_eq!(out.pixel_at(1, 1), FullRange.convert(&MonoA8::new(70, 80)));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Depalettize — palette-lookup conversion
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn depalettize_new_rgb8_basic() {
        let mut palette = [Rgb8::new(0, 0, 0); 256];
        palette[0] = Rgb8::new(255, 0, 0);
        palette[1] = Rgb8::new(0, 255, 0);
        palette[255] = Rgb8::new(0, 0, 255);
        let strategy = Depalettize::new(palette);
        assert_eq!(strategy.convert(&Indexed8(0)), Rgb8::new(255, 0, 0));
        assert_eq!(strategy.convert(&Indexed8(1)), Rgb8::new(0, 255, 0));
        assert_eq!(strategy.convert(&Indexed8(255)), Rgb8::new(0, 0, 255));
    }

    #[test]
    fn depalettize_new_all_black() {
        let palette = [Rgb8::new(0, 0, 0); 256];
        let strategy = Depalettize::new(palette);
        assert_eq!(strategy.convert(&Indexed8(0)), Rgb8::new(0, 0, 0));
        assert_eq!(strategy.convert(&Indexed8(128)), Rgb8::new(0, 0, 0));
        assert_eq!(strategy.convert(&Indexed8(255)), Rgb8::new(0, 0, 0));
    }

    #[test]
    fn depalettize_from_slice_zero_fills() {
        let entries = [Rgb8::new(255, 0, 0), Rgb8::new(0, 255, 0)];
        let strategy = Depalettize::from_slice(&entries);
        assert_eq!(strategy.convert(&Indexed8(0)), Rgb8::new(255, 0, 0));
        assert_eq!(strategy.convert(&Indexed8(1)), Rgb8::new(0, 255, 0));
        // Remaining entries are zero-filled
        assert_eq!(strategy.convert(&Indexed8(2)), Rgb8::new(0, 0, 0));
        assert_eq!(strategy.convert(&Indexed8(255)), Rgb8::new(0, 0, 0));
    }

    #[test]
    fn depalettize_from_slice_empty() {
        let strategy = Depalettize::<Rgb8>::from_slice(&[]);
        assert_eq!(strategy.convert(&Indexed8(0)), Rgb8::new(0, 0, 0));
        assert_eq!(strategy.convert(&Indexed8(255)), Rgb8::new(0, 0, 0));
    }

    #[test]
    fn depalettize_from_slice_full_256() {
        let mut entries = [Rgb8::new(0, 0, 0); 256];
        for (i, entry) in entries.iter_mut().enumerate() {
            *entry = Rgb8::new(i as u8, 0, 0);
        }
        let strategy = Depalettize::from_slice(&entries);
        for i in 0..256u16 {
            assert_eq!(
                strategy.convert(&Indexed8(i as u8)),
                Rgb8::new(i as u8, 0, 0)
            );
        }
    }

    #[test]
    fn depalettize_from_slice_single_entry() {
        let entries = [Rgb8::new(42, 43, 44)];
        let strategy = Depalettize::from_slice(&entries);
        assert_eq!(strategy.convert(&Indexed8(0)), Rgb8::new(42, 43, 44));
        assert_eq!(strategy.convert(&Indexed8(1)), Rgb8::new(0, 0, 0));
    }

    #[test]
    #[should_panic]
    fn depalettize_from_slice_panics_over_256() {
        let entries = vec![Rgb8::new(0, 0, 0); 257];
        let _ = Depalettize::from_slice(&entries);
    }

    #[test]
    fn depalettize_all_indices_valid() {
        // Every u8 index maps to a valid entry — no bounds check needed
        let mut palette = [Mono8::new(0); 256];
        for (i, entry) in palette.iter_mut().enumerate() {
            *entry = Mono8::new(i as u8);
        }
        let strategy = Depalettize::new(palette);
        for i in 0..=255u8 {
            assert_eq!(strategy.convert(&Indexed8(i)), Mono8::new(i));
        }
    }

    #[test]
    fn depalettize_rgba8_with_alpha() {
        // Simulates PNG PLTE + tRNS: palette entries carry alpha
        let entries = vec![
            Rgba8::new(255, 0, 0, 255), // opaque red
            Rgba8::new(0, 255, 0, 128), // semi-transparent green
            Rgba8::new(0, 0, 255, 0),   // fully transparent blue
        ];
        let strategy = Depalettize::<Rgba8>::from_slice(&entries);
        assert_eq!(strategy.convert(&Indexed8(0)), Rgba8::new(255, 0, 0, 255));
        assert_eq!(strategy.convert(&Indexed8(1)), Rgba8::new(0, 255, 0, 128));
        assert_eq!(strategy.convert(&Indexed8(2)), Rgba8::new(0, 0, 255, 0));
        // Zero-filled beyond palette
        assert_eq!(strategy.convert(&Indexed8(3)), Rgba8::new(0, 0, 0, 0));
    }

    #[test]
    fn depalettize_srgb8() {
        // Type parameter carries color-space semantics
        let mut palette = [Srgb8::new(0, 0, 0); 256];
        palette[0] = Srgb8::new(128, 64, 200);
        let strategy = Depalettize::new(palette);
        let result: Srgb8 = strategy.convert(&Indexed8(0));
        assert_eq!(result, Srgb8::new(128, 64, 200));
    }

    #[test]
    fn depalettize_srgba8() {
        let mut palette = [Srgba8::new(0, 0, 0, 0); 256];
        palette[5] = Srgba8::new(100, 150, 200, 255);
        let strategy = Depalettize::new(palette);
        let result: Srgba8 = strategy.convert(&Indexed8(5));
        assert_eq!(result, Srgba8::new(100, 150, 200, 255));
    }

    #[test]
    fn depalettize_mono8() {
        let mut palette = [Mono8::new(0); 256];
        palette[10] = Mono8::new(42);
        let strategy = Depalettize::new(palette);
        assert_eq!(strategy.convert(&Indexed8(10)), Mono8::new(42));
    }

    #[test]
    fn depalettize_rgbf32() {
        // High-precision palette
        let mut palette = [RgbF32::new(0.0, 0.0, 0.0); 256];
        palette[0] = RgbF32::new(1.0, 0.5, 0.25);
        let strategy = Depalettize::new(palette);
        let result: RgbF32 = strategy.convert(&Indexed8(0));
        assert_eq!(result.r, 1.0);
        assert_eq!(result.g, 0.5);
        assert_eq!(result.b, 0.25);
    }

    #[test]
    fn depalettize_bgr8() {
        let mut palette = [Bgr8::new(0, 0, 0); 256];
        palette[7] = Bgr8::new(10, 20, 30);
        let strategy = Depalettize::new(palette);
        assert_eq!(strategy.convert(&Indexed8(7)), Bgr8::new(10, 20, 30));
    }

    #[test]
    fn depalettize_convert_image() {
        let img = Image::fill(3, 2, Indexed8(1));
        let mut palette = [Rgb8::new(0, 0, 0); 256];
        palette[1] = Rgb8::new(100, 200, 50);
        let strategy = Depalettize::new(palette);
        let out: Image<Rgb8> = convert_image(&img, strategy);
        assert_eq!(out.pixel_at(0, 0), Rgb8::new(100, 200, 50));
        assert_eq!(out.pixel_at(2, 1), Rgb8::new(100, 200, 50));
    }

    #[test]
    fn depalettize_convert_image_into() {
        let img = Image::fill(2, 2, Indexed8(0));
        let mut palette = [Rgba8::new(0, 0, 0, 0); 256];
        palette[0] = Rgba8::new(255, 128, 64, 255);
        let strategy = Depalettize::new(palette);
        let mut out = Image::<Rgba8>::zero(2, 2);
        convert_image_into(&img, &mut out, strategy);
        assert_eq!(out.pixel_at(0, 0), Rgba8::new(255, 128, 64, 255));
        assert_eq!(out.pixel_at(1, 1), Rgba8::new(255, 128, 64, 255));
    }

    #[test]
    fn depalettize_convert_image_varying_indices() {
        use crate::image::ImageViewMut;
        let mut img = Image::fill(2, 2, Indexed8(0));
        *img.pixel_at_mut(0, 0) = Indexed8(0);
        *img.pixel_at_mut(1, 0) = Indexed8(1);
        *img.pixel_at_mut(0, 1) = Indexed8(2);
        *img.pixel_at_mut(1, 1) = Indexed8(3);
        let mut palette = [Rgb8::new(0, 0, 0); 256];
        palette[0] = Rgb8::new(10, 20, 30);
        palette[1] = Rgb8::new(40, 50, 60);
        palette[2] = Rgb8::new(70, 80, 90);
        palette[3] = Rgb8::new(100, 110, 120);
        let strategy = Depalettize::new(palette);
        let out: Image<Rgb8> = convert_image(&img, strategy);
        assert_eq!(out.pixel_at(0, 0), Rgb8::new(10, 20, 30));
        assert_eq!(out.pixel_at(1, 0), Rgb8::new(40, 50, 60));
        assert_eq!(out.pixel_at(0, 1), Rgb8::new(70, 80, 90));
        assert_eq!(out.pixel_at(1, 1), Rgb8::new(100, 110, 120));
    }

    // ─── Depalettize composed via .then() ───────────────────────────────

    #[test]
    fn depalettize_then_fullrange_rgb8_to_rgb16() {
        // Indexed8 → Rgb8 (Depalettize) → Rgb16 (FullRange)
        let mut palette = [Rgb8::new(0, 0, 0); 256];
        palette[0] = Rgb8::new(255, 128, 0);
        let method = Depalettize::new(palette).then::<Rgb8, _>(FullRange);
        let result: Rgb16 = method.convert(&Indexed8(0));
        assert_eq!(result, FullRange.convert(&Rgb8::new(255, 128, 0)));
    }

    #[test]
    fn depalettize_then_srgb_gamma() {
        // Indexed8 → Srgb8 (Depalettize) → RgbF32 (SrgbGamma)
        let mut palette = [Srgb8::new(0, 0, 0); 256];
        palette[0] = Srgb8::new(128, 128, 128);
        let method = Depalettize::new(palette).then::<Srgb8, _>(SrgbGamma);
        let result: RgbF32 = method.convert(&Indexed8(0));
        let expected: RgbF32 = SrgbGamma.convert(&Srgb8::new(128, 128, 128));
        assert_eq!(result.r, expected.r);
        assert_eq!(result.g, expected.g);
        assert_eq!(result.b, expected.b);
    }

    #[test]
    fn depalettize_then_colorswap() {
        // Indexed8 → Rgb8 (Depalettize) → Bgr8 (ColorSwap)
        // ColorSwap preserves color semantics: r stays r, b stays b,
        // but Bgr8::new takes (b, g, r) order.
        let mut palette = [Rgb8::new(0, 0, 0); 256];
        palette[0] = Rgb8::new(10, 20, 30);
        let method = Depalettize::new(palette).then::<Rgb8, _>(ColorSwap);
        let result: Bgr8 = method.convert(&Indexed8(0));
        assert_eq!(result, Bgr8::new(30, 20, 10));
    }

    #[test]
    fn depalettize_then_addalpha() {
        // Indexed8 → Rgb8 (Depalettize) → Rgba8 (AddAlpha)
        let mut palette = [Rgb8::new(0, 0, 0); 256];
        palette[5] = Rgb8::new(50, 100, 150);
        let method = Depalettize::new(palette).then::<Rgb8, _>(AddAlpha);
        let result: Rgba8 = method.convert(&Indexed8(5));
        assert_eq!(result, Rgba8::new(50, 100, 150, 255));
    }

    #[test]
    fn depalettize_then_luminance() {
        // Indexed8 → Rgb8 (Depalettize) → Mono8 (Luminance)
        let mut palette = [Rgb8::new(0, 0, 0); 256];
        palette[0] = Rgb8::new(255, 255, 255);
        let method = Depalettize::new(palette).then::<Rgb8, _>(Luminance);
        let result: Mono8 = method.convert(&Indexed8(0));
        assert_eq!(result, Mono8::new(255));
    }

    #[test]
    fn depalettize_triple_chain() {
        // Indexed8 → Rgb8 → Bgr8 → Bgr16
        // Rgb8(r=100, g=150, b=200) → ColorSwap → Bgr8(b=200, g=150, r=100)
        let mut palette = [Rgb8::new(0, 0, 0); 256];
        palette[0] = Rgb8::new(100, 150, 200);
        let method = Depalettize::new(palette)
            .then::<Rgb8, _>(ColorSwap)
            .then::<Bgr8, _>(FullRange);
        let result: Bgr16 = method.convert(&Indexed8(0));
        let expected: Bgr16 = FullRange.convert(&Bgr8::new(200, 150, 100));
        assert_eq!(result, expected);
    }

    #[test]
    fn depalettize_then_convert_image() {
        // Image-level: Indexed8 → Rgb8 → Rgb16
        let img = Image::fill(2, 2, Indexed8(1));
        let mut palette = [Rgb8::new(0, 0, 0); 256];
        palette[1] = Rgb8::new(200, 100, 50);
        let method = Depalettize::new(palette).then::<Rgb8, _>(FullRange);
        let out: Image<Rgb16> = convert_image(&img, method);
        let expected: Rgb16 = FullRange.convert(&Rgb8::new(200, 100, 50));
        assert_eq!(out.pixel_at(0, 0), expected);
    }

    #[test]
    fn depalettize_with_pixelmap() {
        // Depalettize can chain with PixelMap for custom post-processing
        let mut palette = [Mono8::new(0); 256];
        palette[0] = Mono8::new(100);
        let method = Depalettize::new(palette)
            .then::<Mono8, _>(PixelMap(|s: &Mono8| Mono8::new(255 - s.as_bytes()[0])));
        let result: Mono8 = method.convert(&Indexed8(0));
        assert_eq!(result, Mono8::new(155));
    }

    // ─── Indexed8 with NearestNeighbor resize ───────────────────────────

    #[test]
    fn indexed8_nearest_neighbor_resize() {
        use crate::transform::{NearestNeighbor, resize};
        let img = Image::fill(4, 4, Indexed8(42));
        let out: Image<Indexed8> = resize(&img, crate::Size::new(2, 2), NearestNeighbor);
        assert_eq!(out.pixel_at(0, 0), Indexed8(42));
        assert_eq!(out.pixel_at(1, 1), Indexed8(42));
    }

    #[test]
    fn indexed8_nearest_neighbor_resize_enlarge() {
        use crate::image::ImageViewMut;
        use crate::transform::{NearestNeighbor, resize};
        let mut img = Image::fill(2, 2, Indexed8(0));
        *img.pixel_at_mut(0, 0) = Indexed8(1);
        *img.pixel_at_mut(1, 0) = Indexed8(2);
        *img.pixel_at_mut(0, 1) = Indexed8(3);
        *img.pixel_at_mut(1, 1) = Indexed8(4);
        let out: Image<Indexed8> = resize(&img, crate::Size::new(4, 4), NearestNeighbor);
        // scale = (2-1)/(4-1) = 1/3; src_x = floor(dst_x * 1/3)
        // dst_x 0,1,2 → src_x 0; dst_x 3 → src_x 1
        assert_eq!(out.pixel_at(0, 0), Indexed8(1)); // src(0,0)
        assert_eq!(out.pixel_at(1, 0), Indexed8(1)); // src(0,0)
        assert_eq!(out.pixel_at(3, 0), Indexed8(2)); // src(1,0)
        // Bottom row: dst_y 3 → src_y 1
        assert_eq!(out.pixel_at(0, 3), Indexed8(3)); // src(0,1)
        assert_eq!(out.pixel_at(3, 3), Indexed8(4)); // src(1,1)
    }

    // ─── SrgbGamma: SrgbMono8 ↔ f32 ────────────────────────────────────────

    #[test]
    fn srgb_gamma_mono_decode_black() {
        let linear: MonoF32 = SrgbGamma.convert(&SrgbMono8::new(0));
        assert_eq!(linear.0, 0.0);
    }

    #[test]
    fn srgb_gamma_mono_decode_white() {
        let linear: MonoF32 = SrgbGamma.convert(&SrgbMono8::new(255));
        assert!(approx(linear.0, 1.0, 0.001));
    }

    #[test]
    fn srgb_gamma_mono_decode_mid_gray() {
        // sRGB 128 ≈ 0.2158 linear (not 0.502)
        let linear: MonoF32 = SrgbGamma.convert(&SrgbMono8::new(128));
        assert!(approx(linear.0, 0.216, 0.001));
    }

    #[test]
    fn srgb_gamma_mono_decode_low_value_linear_region() {
        // Values ≤ ~10 are in the linear segment of the sRGB curve
        let linear: MonoF32 = SrgbGamma.convert(&SrgbMono8::new(10));
        assert!(linear.0 < 0.04);
    }

    #[test]
    fn srgb_gamma_mono_encode_black() {
        let srgb: SrgbMono8 = SrgbGamma.convert(&MonoF32::new(0.0));
        assert_eq!(srgb, SrgbMono8::new(0));
    }

    #[test]
    fn srgb_gamma_mono_encode_white() {
        let srgb: SrgbMono8 = SrgbGamma.convert(&MonoF32::new(1.0));
        assert_eq!(srgb, SrgbMono8::new(255));
    }

    #[test]
    fn srgb_gamma_mono_encode_half_linear() {
        // 0.5 linear ≈ 188 sRGB (not 128)
        let srgb: SrgbMono8 = SrgbGamma.convert(&MonoF32::new(0.5));
        assert_eq!(srgb.0.0, 188);
    }

    #[test]
    fn srgb_gamma_mono_encode_clamps_negative() {
        let srgb: SrgbMono8 = SrgbGamma.convert(&MonoF32::new(-0.5));
        assert_eq!(srgb.0.0, 0);
    }

    #[test]
    fn srgb_gamma_mono_encode_clamps_above_one() {
        let srgb: SrgbMono8 = SrgbGamma.convert(&MonoF32::new(1.5));
        assert_eq!(srgb.0.0, 255);
    }

    #[test]
    fn srgb_gamma_mono_roundtrip_black() {
        let orig = SrgbMono8::new(0);
        let linear: MonoF32 = SrgbGamma.convert(&orig);
        let back: SrgbMono8 = SrgbGamma.convert(&linear);
        assert_eq!(orig, back);
    }

    #[test]
    fn srgb_gamma_mono_roundtrip_white() {
        let orig = SrgbMono8::new(255);
        let linear: MonoF32 = SrgbGamma.convert(&orig);
        let back: SrgbMono8 = SrgbGamma.convert(&linear);
        assert_eq!(orig, back);
    }

    #[test]
    fn srgb_gamma_mono_roundtrip_all_values() {
        // Every u8 value should survive a decode→encode roundtrip
        for v in 0..=255u8 {
            let orig = SrgbMono8::new(v);
            let linear: MonoF32 = SrgbGamma.convert(&orig);
            let back: SrgbMono8 = SrgbGamma.convert(&linear);
            assert_eq!(orig, back, "roundtrip failed for v={v}");
        }
    }

    #[test]
    fn srgb_gamma_mono_matches_rgb_channel() {
        // SrgbMono8 decode should produce the same value as the
        // corresponding Srgb8 channel
        for v in 0..=255u8 {
            let mono_linear: MonoF32 = SrgbGamma.convert(&SrgbMono8::new(v));
            let rgb_linear: RgbF32 = SrgbGamma.convert(&Srgb8::new(v, v, v));
            assert_eq!(mono_linear.0, rgb_linear.r, "mismatch at v={v}");
        }
    }

    #[test]
    fn srgb_gamma_mono_then_fullrange_to_mono16() {
        // SrgbMono8 → MonoF32 → Mono16
        let method = SrgbGamma.then::<MonoF32, _>(FullRange);
        let result: Mono16 = method.convert(&SrgbMono8::new(255));
        assert_eq!(result, Mono16::new(65535));
    }

    #[test]
    fn convert_image_srgb_mono8_to_f32() {
        let img = Image::from_vec(2, 1, vec![SrgbMono8::new(0), SrgbMono8::new(255)]).unwrap();
        let out: Image<MonoF32> = convert_image(&img, SrgbGamma);
        assert_eq!(out.pixel_at(0, 0).0, 0.0);
        assert!(approx(out.pixel_at(1, 0).0, 1.0, 0.001));
    }

    #[test]
    fn convert_image_f32_to_srgb_mono8() {
        let img = Image::from_vec(2, 1, vec![MonoF32::new(0.0), MonoF32::new(1.0)]).unwrap();
        let out: Image<SrgbMono8> = convert_image(&img, SrgbGamma);
        assert_eq!(out.pixel_at(0, 0), SrgbMono8::new(0));
        assert_eq!(out.pixel_at(1, 0), SrgbMono8::new(255));
    }

    #[test]
    fn nearest_neighbor_resize_srgb_mono8() {
        use crate::transform::{NearestNeighbor, resize};
        let img = Image::from_vec(1, 1, vec![SrgbMono8::new(128)]).unwrap();
        let out: Image<SrgbMono8> = resize(&img, crate::Size::new(2, 2), NearestNeighbor);
        assert_eq!(out.pixel_at(0, 0), SrgbMono8::new(128));
        assert_eq!(out.pixel_at(1, 1), SrgbMono8::new(128));
    }

    // ─── SrgbMono8 → SrgbMonoA8 (AddAlpha) ─────────────────────────────────

    #[test]
    fn srgb_mono8_addalpha() {
        let src = SrgbMono8::new(128);
        let dst: SrgbMonoA8 = AddAlpha.convert(&src);
        assert_eq!(dst, SrgbMonoA8::new(128, 255));
    }

    #[test]
    fn srgb_mono8_addalpha_extremes() {
        let black: SrgbMonoA8 = AddAlpha.convert(&SrgbMono8::new(0));
        assert_eq!(black, SrgbMonoA8::new(0, 255));
        let white: SrgbMonoA8 = AddAlpha.convert(&SrgbMono8::new(255));
        assert_eq!(white, SrgbMonoA8::new(255, 255));
    }

    // ─── SrgbMonoA8 ↔ MonoAF32 (SrgbGamma) ─────────────────────────────────

    #[test]
    fn srgb_gamma_srgb_mono_a8_to_mono_af32() {
        let src = SrgbMonoA8::new(128, 200);
        let dst: MonoAF32 = SrgbGamma.convert(&src);
        // 128 sRGB ≈ 0.2158 linear
        assert!(approx(dst.v, 0.2158, 0.002));
        // alpha 200/255 ≈ 0.7843 (linear, no gamma)
        assert!(approx(dst.a, 200.0 / 255.0, 0.001));
    }

    #[test]
    fn srgb_gamma_mono_af32_to_srgb_mono_a8() {
        let src = MonoAF32::new(1.0, 1.0);
        let dst: SrgbMonoA8 = SrgbGamma.convert(&src);
        assert_eq!(dst, SrgbMonoA8::new(255, 255));

        let src = MonoAF32::new(0.0, 0.0);
        let dst: SrgbMonoA8 = SrgbGamma.convert(&src);
        assert_eq!(dst, SrgbMonoA8::new(0, 0));
    }

    #[test]
    fn srgb_gamma_srgb_mono_a8_roundtrip() {
        for v in [0u8, 1, 64, 128, 200, 254, 255] {
            for a in [0u8, 128, 255] {
                let orig = SrgbMonoA8::new(v, a);
                let linear: MonoAF32 = SrgbGamma.convert(&orig);
                let back: SrgbMonoA8 = SrgbGamma.convert(&linear);
                assert_eq!(orig, back, "roundtrip failed for v={v}, a={a}");
            }
        }
    }

    #[test]
    fn srgb_gamma_srgb_mono_a8_matches_srgba8() {
        // The value channel should decode identically to Srgba8's channels
        for v in 0..=255u8 {
            let mono_linear: MonoAF32 = SrgbGamma.convert(&SrgbMonoA8::new(v, 255));
            let rgba_linear: RgbaF32 = SrgbGamma.convert(&Srgba8::new(v, v, v, 255));
            assert_eq!(mono_linear.v, rgba_linear.r, "value mismatch at v={v}");
            assert_eq!(mono_linear.a, rgba_linear.a, "alpha mismatch at v={v}");
        }
    }

    #[test]
    fn srgb_gamma_srgb_mono_a8_alpha_clamped() {
        // Out-of-range alpha in MonoAF32 should clamp to [0, 255]
        let src = MonoAF32::new(0.5, 2.0);
        let dst: SrgbMonoA8 = SrgbGamma.convert(&src);
        assert_eq!(dst.a.0, 255);

        let src = MonoAF32::new(0.5, -1.0);
        let dst: SrgbMonoA8 = SrgbGamma.convert(&src);
        assert_eq!(dst.a.0, 0);
    }

    #[test]
    fn convert_image_srgb_mono_a8_to_mono_af32() {
        let img = Image::from_vec(
            2,
            1,
            vec![SrgbMonoA8::new(0, 255), SrgbMonoA8::new(255, 128)],
        )
        .unwrap();
        let out: Image<MonoAF32> = convert_image(&img, SrgbGamma);
        assert_eq!(out.pixel_at(0, 0).v, 0.0);
        assert!(approx(out.pixel_at(0, 0).a, 1.0, 0.001));
        assert!(approx(out.pixel_at(1, 0).v, 1.0, 0.001));
        assert!(approx(out.pixel_at(1, 0).a, 128.0 / 255.0, 0.001));
    }

    #[test]
    fn convert_image_mono_af32_to_srgb_mono_a8() {
        let img =
            Image::from_vec(2, 1, vec![MonoAF32::new(0.0, 0.0), MonoAF32::new(1.0, 1.0)]).unwrap();
        let out: Image<SrgbMonoA8> = convert_image(&img, SrgbGamma);
        assert_eq!(out.pixel_at(0, 0), SrgbMonoA8::new(0, 0));
        assert_eq!(out.pixel_at(1, 0), SrgbMonoA8::new(255, 255));
    }

    #[test]
    fn srgb_gamma_mono_a8_then_fullrange_to_monoa16() {
        // SrgbMonoA8 → MonoAF32 → MonoA16
        let method = SrgbGamma.then::<MonoAF32, _>(FullRange);
        let result: MonoA16 = method.convert(&SrgbMonoA8::new(255, 255));
        assert_eq!(result.v.0, 65535);
        assert_eq!(result.a.0, 65535);
    }

    // ─── Srgb16 ↔ RgbF32 via SrgbGamma ─────────────────────────────────

    #[test]
    fn srgb16_gamma_decode_black() {
        let linear: RgbF32 = SrgbGamma.convert(&Srgb16::new(0, 0, 0));
        assert_eq!(linear.r, 0.0);
        assert_eq!(linear.g, 0.0);
        assert_eq!(linear.b, 0.0);
    }

    #[test]
    fn srgb16_gamma_decode_white() {
        let linear: RgbF32 = SrgbGamma.convert(&Srgb16::new(65535, 65535, 65535));
        assert!(approx(linear.r, 1.0, 0.001));
        assert!(approx(linear.g, 1.0, 0.001));
        assert!(approx(linear.b, 1.0, 0.001));
    }

    #[test]
    fn srgb16_gamma_decode_mid_gray() {
        // sRGB mid-gray (≈ 0.5 normalized) decodes to ≈ 0.214 linear
        let srgb = Srgb16::new(32768, 32768, 32768);
        let linear: RgbF32 = SrgbGamma.convert(&srgb);
        assert!(approx(linear.r, 0.214, 0.002));
        assert!(approx(linear.g, 0.214, 0.002));
    }

    #[test]
    fn srgb16_gamma_decode_per_channel() {
        let srgb = Srgb16::new(0, 32768, 65535);
        let linear: RgbF32 = SrgbGamma.convert(&srgb);
        assert_eq!(linear.r, 0.0);
        assert!(linear.g > 0.0 && linear.g < 1.0);
        assert!(approx(linear.b, 1.0, 0.001));
    }

    #[test]
    fn srgb16_gamma_encode_black() {
        let srgb: Srgb16 = SrgbGamma.convert(&RgbF32::new(0.0, 0.0, 0.0));
        assert_eq!(srgb.r.0, 0);
        assert_eq!(srgb.g.0, 0);
        assert_eq!(srgb.b.0, 0);
    }

    #[test]
    fn srgb16_gamma_encode_white() {
        let srgb: Srgb16 = SrgbGamma.convert(&RgbF32::new(1.0, 1.0, 1.0));
        assert_eq!(srgb.r.0, 65535);
        assert_eq!(srgb.g.0, 65535);
        assert_eq!(srgb.b.0, 65535);
    }

    #[test]
    fn srgb16_gamma_encode_half_linear() {
        // 0.5 linear → sRGB encode → ~0.735 normalised → ~48190 in u16
        let srgb: Srgb16 = SrgbGamma.convert(&RgbF32::new(0.5, 0.5, 0.5));
        // Compute expected via the same encode path used for SrgbMono16
        let expected: SrgbMono16 = SrgbGamma.convert(&MonoF32::new(0.5));
        assert!(
            (srgb.r.0 as i32 - expected.0.0 as i32).unsigned_abs() <= 1,
            "encode mismatch: r={} expected={}",
            srgb.r.0,
            expected.0.0
        );
    }

    #[test]
    fn srgb16_gamma_encode_clamps_negative() {
        let srgb: Srgb16 = SrgbGamma.convert(&RgbF32::new(-1.0, -0.5, -0.001));
        assert_eq!(srgb.r.0, 0);
        assert_eq!(srgb.g.0, 0);
        assert_eq!(srgb.b.0, 0);
    }

    #[test]
    fn srgb16_gamma_encode_clamps_above_one() {
        let srgb: Srgb16 = SrgbGamma.convert(&RgbF32::new(1.5, 2.0, 100.0));
        assert_eq!(srgb.r.0, 65535);
        assert_eq!(srgb.g.0, 65535);
        assert_eq!(srgb.b.0, 65535);
    }

    #[test]
    fn srgb16_gamma_roundtrip_black() {
        let orig = Srgb16::new(0, 0, 0);
        let linear: RgbF32 = SrgbGamma.convert(&orig);
        let back: Srgb16 = SrgbGamma.convert(&linear);
        assert_eq!(orig, back);
    }

    #[test]
    fn srgb16_gamma_roundtrip_white() {
        let orig = Srgb16::new(65535, 65535, 65535);
        let linear: RgbF32 = SrgbGamma.convert(&orig);
        let back: Srgb16 = SrgbGamma.convert(&linear);
        assert_eq!(orig, back);
    }

    #[test]
    fn srgb16_gamma_roundtrip_sample_values() {
        // Roundtrip several representative values; allow ±1 due to float precision
        for v in [0u16, 1, 100, 1000, 10000, 32768, 50000, 65534, 65535] {
            let orig = Srgb16::new(v, v, v);
            let linear: RgbF32 = SrgbGamma.convert(&orig);
            let back: Srgb16 = SrgbGamma.convert(&linear);
            assert!(
                (back.r.0 as i32 - orig.r.0 as i32).unsigned_abs() <= 1,
                "roundtrip failed for {v}: got {}, expected {v}",
                back.r.0
            );
        }
    }

    // ─── Srgba16 ↔ RgbaF32 via SrgbGamma ───────────────────────────────

    #[test]
    fn srgba16_gamma_decode_alpha_is_linear() {
        let src = Srgba16::new(32768, 32768, 32768, 32768);
        let linear: RgbaF32 = SrgbGamma.convert(&src);
        // Alpha should be linearly scaled, not gamma-decoded
        let expected_alpha = 32768.0 / 65535.0;
        assert!(approx(linear.a, expected_alpha, 0.001));
        // R, G, B should be gamma-decoded (much less than 0.5)
        assert!(linear.r < 0.25);
    }

    #[test]
    fn srgba16_gamma_decode_full_alpha() {
        let src = Srgba16::new(0, 0, 0, 65535);
        let linear: RgbaF32 = SrgbGamma.convert(&src);
        assert!(approx(linear.a, 1.0, 0.001));
    }

    #[test]
    fn srgba16_gamma_decode_zero_alpha() {
        let src = Srgba16::new(65535, 65535, 65535, 0);
        let linear: RgbaF32 = SrgbGamma.convert(&src);
        assert_eq!(linear.a, 0.0);
    }

    #[test]
    fn srgba16_gamma_decode_rgb_matches_srgb16() {
        let srgb = Srgb16::new(10000, 30000, 60000);
        let srgba = Srgba16::new(10000, 30000, 60000, 65535);
        let lin_rgb: RgbF32 = SrgbGamma.convert(&srgb);
        let lin_rgba: RgbaF32 = SrgbGamma.convert(&srgba);
        assert!(approx(lin_rgb.r, lin_rgba.r, 1e-6));
        assert!(approx(lin_rgb.g, lin_rgba.g, 1e-6));
        assert!(approx(lin_rgb.b, lin_rgba.b, 1e-6));
    }

    #[test]
    fn srgba16_gamma_encode_alpha_is_linear() {
        let lin = RgbaF32::new(0.0, 0.0, 0.0, 0.5);
        let srgba: Srgba16 = SrgbGamma.convert(&lin);
        let expected = (0.5 * 65535.0 + 0.5) as u16;
        assert_eq!(srgba.a.0, expected);
    }

    #[test]
    fn srgba16_gamma_encode_clamps_alpha() {
        let lin = RgbaF32::new(0.0, 0.0, 0.0, 1.5);
        let srgba: Srgba16 = SrgbGamma.convert(&lin);
        assert_eq!(srgba.a.0, 65535);
    }

    #[test]
    fn srgba16_gamma_roundtrip_sample_values() {
        for v in [0u16, 1, 1000, 32768, 65535] {
            let orig = Srgba16::new(v, v, v, v);
            let linear: RgbaF32 = SrgbGamma.convert(&orig);
            let back: Srgba16 = SrgbGamma.convert(&linear);
            assert!(
                (back.r.0 as i32 - orig.r.0 as i32).unsigned_abs() <= 1,
                "roundtrip r failed for {v}: got {}, expected {v}",
                back.r.0
            );
            assert!(
                (back.a.0 as i32 - orig.a.0 as i32).unsigned_abs() <= 1,
                "roundtrip a failed for {v}: got {}, expected {v}",
                back.a.0
            );
        }
    }

    // ─── SrgbMono16 ↔ f32 via SrgbGamma ────────────────────────────────

    #[test]
    fn srgb_gamma_mono16_decode_black() {
        let linear: MonoF32 = SrgbGamma.convert(&SrgbMono16::new(0));
        assert_eq!(linear.0, 0.0);
    }

    #[test]
    fn srgb_gamma_mono16_decode_white() {
        let linear: MonoF32 = SrgbGamma.convert(&SrgbMono16::new(65535));
        assert!(approx(linear.0, 1.0, 0.001));
    }

    #[test]
    fn srgb_gamma_mono16_decode_mid_gray() {
        let linear: MonoF32 = SrgbGamma.convert(&SrgbMono16::new(32768));
        // sRGB mid-gray ≈ 0.214 linear
        assert!(approx(linear.0, 0.214, 0.002));
    }

    #[test]
    fn srgb_gamma_mono16_decode_low_value_linear_region() {
        // Very small values fall in the linear segment of sRGB
        let linear: MonoF32 = SrgbGamma.convert(&SrgbMono16::new(671)); // ≈ 0.01024 normalized
        assert!(linear.0 > 0.0 && linear.0 < 0.001);
    }

    #[test]
    fn srgb_gamma_mono16_encode_black() {
        let srgb: SrgbMono16 = SrgbGamma.convert(&MonoF32::new(0.0));
        assert_eq!(srgb.0.0, 0);
    }

    #[test]
    fn srgb_gamma_mono16_encode_white() {
        let srgb: SrgbMono16 = SrgbGamma.convert(&MonoF32::new(1.0));
        assert_eq!(srgb.0.0, 65535);
    }

    #[test]
    fn srgb_gamma_mono16_encode_half_linear() {
        let srgb: SrgbMono16 = SrgbGamma.convert(&MonoF32::new(0.5));
        // 0.5 linear → sRGB ≈ 0.735 normalised → ~48190 in u16
        // Verify it's in a reasonable range and matches 8-bit behaviour
        let srgb8: SrgbMono8 = SrgbGamma.convert(&MonoF32::new(0.5));
        assert_eq!(srgb8.0.0, 188); // sanity: 8-bit gives 188
        // 16-bit should be close to the full-precision value (not 188/255*65535)
        assert!(
            srgb.0.0 > 48000 && srgb.0.0 < 48400,
            "unexpected encode: {}",
            srgb.0.0
        );
    }

    #[test]
    fn srgb_gamma_mono16_encode_clamps_negative() {
        let srgb: SrgbMono16 = SrgbGamma.convert(&MonoF32::new(-1.0));
        assert_eq!(srgb.0.0, 0);
    }

    #[test]
    fn srgb_gamma_mono16_encode_clamps_above_one() {
        let srgb: SrgbMono16 = SrgbGamma.convert(&MonoF32::new(2.0));
        assert_eq!(srgb.0.0, 65535);
    }

    #[test]
    fn srgb_gamma_mono16_roundtrip_black() {
        let orig = SrgbMono16::new(0);
        let linear: MonoF32 = SrgbGamma.convert(&orig);
        let back: SrgbMono16 = SrgbGamma.convert(&linear);
        assert_eq!(orig, back);
    }

    #[test]
    fn srgb_gamma_mono16_roundtrip_white() {
        let orig = SrgbMono16::new(65535);
        let linear: MonoF32 = SrgbGamma.convert(&orig);
        let back: SrgbMono16 = SrgbGamma.convert(&linear);
        assert_eq!(orig, back);
    }

    #[test]
    fn srgb_gamma_mono16_roundtrip_sample_values() {
        for v in [0u16, 1, 100, 1000, 10000, 32768, 50000, 65534, 65535] {
            let orig = SrgbMono16::new(v);
            let linear: MonoF32 = SrgbGamma.convert(&orig);
            let back: SrgbMono16 = SrgbGamma.convert(&linear);
            assert!(
                (back.0.0 as i32 - orig.0.0 as i32).unsigned_abs() <= 1,
                "roundtrip failed for {v}: got {}, expected {v}",
                back.0.0
            );
        }
    }

    #[test]
    fn srgb_gamma_mono16_matches_rgb_channel() {
        // SrgbMono16 decode should match single-channel Srgb16 decode
        for v in [0u16, 1000, 32768, 65535] {
            let mono_lin: MonoF32 = SrgbGamma.convert(&SrgbMono16::new(v));
            let rgb_lin: RgbF32 = SrgbGamma.convert(&Srgb16::new(v, v, v));
            assert!(
                (mono_lin.0 - rgb_lin.r).abs() < 1e-6,
                "mismatch at {v}: mono={} rgb.r={}",
                mono_lin.0,
                rgb_lin.r
            );
        }
    }

    // ─── SrgbMonoA16 ↔ MonoAF32 via SrgbGamma ──────────────────────────

    #[test]
    fn srgb_gamma_srgb_mono_a16_to_mono_af32() {
        let src = SrgbMonoA16::new(32768, 65535);
        let linear: MonoAF32 = SrgbGamma.convert(&src);
        // Value channel is gamma-decoded
        assert!(linear.v < 0.25); // 0.5 sRGB ≈ 0.214 linear
        // Alpha is linearly scaled
        assert!(approx(linear.a, 1.0, 0.001));
    }

    #[test]
    fn srgb_gamma_mono_af32_to_srgb_mono_a16() {
        let src = MonoAF32::new(0.5, 0.75);
        let srgb: SrgbMonoA16 = SrgbGamma.convert(&src);
        // Value channel is gamma-encoded — should match standalone SrgbMono16 encode
        let expected_v: SrgbMono16 = SrgbGamma.convert(&MonoF32::new(0.5));
        assert!(
            (srgb.v.0 as i32 - expected_v.0.0 as i32).unsigned_abs() <= 1,
            "v mismatch: got {} expected {}",
            srgb.v.0,
            expected_v.0.0
        );
        // Alpha is linearly scaled (0.75 → ≈ 49151)
        let expected_alpha = (0.75 * 65535.0 + 0.5) as u16;
        assert_eq!(srgb.a.0, expected_alpha);
    }

    #[test]
    fn srgb_gamma_srgb_mono_a16_roundtrip() {
        for v in [0u16, 100, 32768, 65535] {
            for a in [0u16, 32768, 65535] {
                let orig = SrgbMonoA16::new(v, a);
                let linear: MonoAF32 = SrgbGamma.convert(&orig);
                let back: SrgbMonoA16 = SrgbGamma.convert(&linear);
                assert!(
                    (back.v.0 as i32 - orig.v.0 as i32).unsigned_abs() <= 1,
                    "v roundtrip failed for ({v},{a}): got {}, expected {v}",
                    back.v.0
                );
                assert!(
                    (back.a.0 as i32 - orig.a.0 as i32).unsigned_abs() <= 1,
                    "a roundtrip failed for ({v},{a}): got {}, expected {a}",
                    back.a.0
                );
            }
        }
    }

    #[test]
    fn srgb_gamma_srgb_mono_a16_alpha_clamped() {
        let src = MonoAF32::new(0.0, 1.5);
        let srgb: SrgbMonoA16 = SrgbGamma.convert(&src);
        assert_eq!(srgb.a.0, 65535); // clamped to max
    }

    #[test]
    fn srgb_gamma_srgb_mono_a16_matches_srgba16() {
        // Verify mono-with-alpha and rgb-with-alpha agree on the value decode
        for v in [0u16, 1000, 32768, 65535] {
            let mono_lin: MonoAF32 = SrgbGamma.convert(&SrgbMonoA16::new(v, 65535));
            let rgba_lin: RgbaF32 = SrgbGamma.convert(&Srgba16::new(v, v, v, 65535));
            assert!(
                (mono_lin.v - rgba_lin.r).abs() < 1e-6,
                "mismatch at {v}: mono.v={} rgba.r={}",
                mono_lin.v,
                rgba_lin.r
            );
        }
    }

    // ─── AddAlpha for 16-bit sRGB types ─────────────────────────────────

    #[test]
    fn srgb16_addalpha() {
        let src = Srgb16::new(100, 200, 300);
        let dst: Srgba16 = AddAlpha.convert(&src);
        assert_eq!(dst.r.0, 100);
        assert_eq!(dst.g.0, 200);
        assert_eq!(dst.b.0, 300);
        assert_eq!(dst.a.0, u16::MAX);
    }

    #[test]
    fn srgb16_addalpha_extremes() {
        let black: Srgba16 = AddAlpha.convert(&Srgb16::new(0, 0, 0));
        assert_eq!(black.a.0, 65535);
        let white: Srgba16 = AddAlpha.convert(&Srgb16::new(65535, 65535, 65535));
        assert_eq!(white.r.0, 65535);
        assert_eq!(white.a.0, 65535);
    }

    #[test]
    fn srgb_mono16_addalpha() {
        let src = SrgbMono16::new(12345);
        let dst: SrgbMonoA16 = AddAlpha.convert(&src);
        assert_eq!(dst.v.0, 12345);
        assert_eq!(dst.a.0, u16::MAX);
    }

    #[test]
    fn srgb_mono16_addalpha_extremes() {
        let z: SrgbMonoA16 = AddAlpha.convert(&SrgbMono16::new(0));
        assert_eq!(z.v.0, 0);
        assert_eq!(z.a.0, 65535);
        let m: SrgbMonoA16 = AddAlpha.convert(&SrgbMono16::new(65535));
        assert_eq!(m.v.0, 65535);
        assert_eq!(m.a.0, 65535);
    }

    // ─── .then() composition for 16-bit sRGB types ──────────────────────

    #[test]
    fn srgb16_gamma_then_fullrange_to_rgb16() {
        // Srgb16 → RgbF32 → Rgb16 (sRGB decode then quantise to linear 16-bit)
        let method = SrgbGamma.then::<RgbF32, _>(FullRange);
        let result: Rgb16 = method.convert(&Srgb16::new(65535, 65535, 65535));
        assert_eq!(result.r.0, 65535);
        assert_eq!(result.g.0, 65535);
        assert_eq!(result.b.0, 65535);
    }

    #[test]
    fn fullrange_then_srgb16_gamma_rgb16_to_srgb16() {
        // Rgb16 → RgbF32 → Srgb16 (linear 16-bit → sRGB encode)
        let method = FullRange.then::<RgbF32, _>(SrgbGamma);
        let result: Srgb16 = method.convert(&Rgb16::new(65535, 0, 32768));
        assert_eq!(result.r.0, 65535);
        assert_eq!(result.g.0, 0);
        // 32768/65535 ≈ 0.5 linear → sRGB encode
        assert!(result.b.0 > 40000);
    }

    #[test]
    fn srgb_gamma_mono16_then_fullrange_to_mono16() {
        // SrgbMono16 → MonoF32 → Mono16
        let method = SrgbGamma.then::<MonoF32, _>(FullRange);
        let result: Mono16 = method.convert(&SrgbMono16::new(65535));
        assert_eq!(result, Mono16::new(65535));
    }

    #[test]
    fn srgb_gamma_mono_a16_then_fullrange_to_monoa16() {
        // SrgbMonoA16 → MonoAF32 → MonoA16
        let method = SrgbGamma.then::<MonoAF32, _>(FullRange);
        let result: MonoA16 = method.convert(&SrgbMonoA16::new(65535, 65535));
        assert_eq!(result.v.0, 65535);
        assert_eq!(result.a.0, 65535);
    }

    // ─── convert_image for 16-bit sRGB types ────────────────────────────

    #[test]
    fn convert_image_srgb16_to_rgbf32() {
        let img: Image<Srgb16> = Image::fill(2, 2, Srgb16::new(0, 32768, 65535));
        let out: Image<RgbF32> = convert_image(&img, SrgbGamma);
        let p = out[(0, 0)];
        assert_eq!(p.r, 0.0);
        assert!(approx(p.b, 1.0, 0.001));
    }

    #[test]
    fn convert_image_rgbf32_to_srgb16() {
        let img: Image<RgbF32> = Image::fill(2, 2, RgbF32::new(0.0, 0.5, 1.0));
        let out: Image<Srgb16> = convert_image(&img, SrgbGamma);
        let p = out[(0, 0)];
        assert_eq!(p.r.0, 0);
        assert_eq!(p.b.0, 65535);
    }

    #[test]
    fn convert_image_srgb_mono16_to_f32() {
        let img: Image<SrgbMono16> = Image::fill(2, 2, SrgbMono16::new(65535));
        let out: Image<MonoF32> = convert_image(&img, SrgbGamma);
        assert!(approx(out[(0, 0)].0, 1.0, 0.001));
    }

    #[test]
    fn convert_image_srgb_mono_a16_roundtrip() {
        let img: Image<SrgbMonoA16> = Image::fill(3, 2, SrgbMonoA16::new(32768, 65535));
        let lin: Image<MonoAF32> = convert_image(&img, SrgbGamma);
        let back: Image<SrgbMonoA16> = convert_image(&lin, SrgbGamma);
        let p = back[(0, 0)];
        assert!((p.v.0 as i32 - 32768).unsigned_abs() <= 1);
        assert!((p.a.0 as i32 - 65535).unsigned_abs() <= 1);
    }

    // ─── Nearest-neighbor resize for 16-bit sRGB (no LinearSpace) ───────

    #[test]
    fn nearest_neighbor_resize_srgb16() {
        use crate::transform::{NearestNeighbor, resize};
        let img: Image<Srgb16> = Image::fill(4, 4, Srgb16::new(100, 200, 300));
        let out: Image<Srgb16> = resize(&img, crate::Size::new(2, 2), NearestNeighbor);
        assert_eq!(out.pixel_at(0, 0), Srgb16::new(100, 200, 300));
    }

    #[test]
    fn nearest_neighbor_resize_srgba16() {
        use crate::transform::{NearestNeighbor, resize};
        let img: Image<Srgba16> = Image::fill(4, 4, Srgba16::new(100, 200, 300, 400));
        let out: Image<Srgba16> = resize(&img, crate::Size::new(2, 2), NearestNeighbor);
        assert_eq!(out.pixel_at(0, 0), Srgba16::new(100, 200, 300, 400));
    }

    #[test]
    fn nearest_neighbor_resize_srgb_mono16() {
        use crate::transform::{NearestNeighbor, resize};
        let img: Image<SrgbMono16> = Image::fill(4, 4, SrgbMono16::new(42000));
        let out: Image<SrgbMono16> = resize(&img, crate::Size::new(2, 2), NearestNeighbor);
        assert_eq!(out.pixel_at(0, 0), SrgbMono16::new(42000));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // MonoF32 / MonoF64 conversion tests
    // ═══════════════════════════════════════════════════════════════════════

    // ── FullRange: MonoF32 ↔ integer Mono types ─────────────────────────

    #[test]
    fn fullrange_mono8_to_monof32() {
        let src = Mono8::new(255);
        let dst: MonoF32 = FullRange.convert(&src);
        assert!(approx(dst.0, 1.0, 1e-3));

        let src = Mono8::new(0);
        let dst: MonoF32 = FullRange.convert(&src);
        assert_eq!(dst.0, 0.0);
    }

    #[test]
    fn fullrange_monof32_to_mono8() {
        let src = MonoF32::new(1.0);
        let dst: Mono8 = FullRange.convert(&src);
        assert_eq!(dst.value(), 255);

        let src = MonoF32::new(0.0);
        let dst: Mono8 = FullRange.convert(&src);
        assert_eq!(dst.value(), 0);
    }

    #[test]
    fn fullrange_mono16_to_monof32() {
        let src = Mono16::new(65535);
        let dst: MonoF32 = FullRange.convert(&src);
        assert!(approx(dst.0, 1.0, 1e-3));
    }

    #[test]
    fn fullrange_monof32_to_mono16() {
        let src = MonoF32::new(1.0);
        let dst: Mono16 = FullRange.convert(&src);
        assert_eq!(dst.value(), 65535);
    }

    #[test]
    fn fullrange_mono32_to_monof32() {
        let src = Mono32::new(u32::MAX);
        let dst: MonoF32 = FullRange.convert(&src);
        assert!(approx(dst.0, 1.0, 1e-3));
    }

    #[test]
    fn fullrange_monof32_to_mono32() {
        let src = MonoF32::new(0.0);
        let dst: Mono32 = FullRange.convert(&src);
        assert_eq!(dst.value(), 0);
    }

    #[test]
    fn fullrange_mono64_to_monof32() {
        let src = Mono64::new(u64::MAX);
        let dst: MonoF32 = FullRange.convert(&src);
        assert!(approx(dst.0, 1.0, 1e-3));
    }

    #[test]
    fn fullrange_monof32_to_mono64() {
        let src = MonoF32::new(0.0);
        let dst: Mono64 = FullRange.convert(&src);
        assert_eq!(dst.value(), 0);
    }

    #[test]
    fn fullrange_mono10_to_monof32() {
        let src = Mono::<10>::new(1023);
        let dst: MonoF32 = FullRange.convert(&src);
        assert!(approx(dst.0, 1.0, 1e-3));

        let src = Mono::<10>::new(0);
        let dst: MonoF32 = FullRange.convert(&src);
        assert_eq!(dst.0, 0.0);
    }

    // ── FullRange: MonoF64 ↔ integer Mono types ─────────────────────────

    #[test]
    fn fullrange_mono8_to_monof64() {
        let src = Mono8::new(255);
        let dst: MonoF64 = FullRange.convert(&src);
        assert!((dst.0 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn fullrange_monof64_to_mono8() {
        let src = MonoF64::new(1.0);
        let dst: Mono8 = FullRange.convert(&src);
        assert_eq!(dst.value(), 255);
    }

    #[test]
    fn fullrange_mono16_to_monof64() {
        let src = Mono16::new(65535);
        let dst: MonoF64 = FullRange.convert(&src);
        assert!((dst.0 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn fullrange_monof64_to_mono16() {
        let src = MonoF64::new(1.0);
        let dst: Mono16 = FullRange.convert(&src);
        assert_eq!(dst.value(), 65535);
    }

    #[test]
    fn fullrange_mono10_to_monof64() {
        let src = Mono::<10>::new(1023);
        let dst: MonoF64 = FullRange.convert(&src);
        assert!((dst.0 - 1.0).abs() < 1e-6);
    }

    // ── FullRange: MonoF32 ↔ MonoF64 ────────────────────────────────────

    #[test]
    fn fullrange_monof32_to_monof64() {
        let src = MonoF32::new(0.5);
        let dst: MonoF64 = FullRange.convert(&src);
        assert!((dst.0 - 0.5).abs() < 1e-6);
    }

    #[test]
    fn fullrange_monof64_to_monof32() {
        let src = MonoF64::new(0.5);
        let dst: MonoF32 = FullRange.convert(&src);
        assert!(approx(dst.0, 0.5, 1e-6));
    }

    // ── FullRange: MonoF32 ↔ bare f32, MonoF64 ↔ bare f64 ──────────────
    //
    // These identity conversions are intentionally not provided.
    // Tests omitted accordingly.

    // ── Luminance: RGB/BGR → MonoF32 / MonoF64 ─────────────────────────

    #[test]
    fn luminance_rgbf32_to_monof32() {
        let white = RgbF32::new(1.0, 1.0, 1.0);
        let y: MonoF32 = Luminance.convert(&white);
        assert!(approx(y.0, 1.0, 1e-4));

        let red = RgbF32::new(1.0, 0.0, 0.0);
        let y: MonoF32 = Luminance.convert(&red);
        assert!(approx(y.0, 0.299, 1e-4));
    }

    #[test]
    fn luminance_rgbaf32_to_monof32() {
        let src = RgbaF32::new(1.0, 0.0, 0.0, 0.5);
        let y: MonoF32 = Luminance.convert(&src);
        assert!(approx(y.0, 0.299, 1e-4));
    }

    #[test]
    fn luminance_bgrf32_to_monof32() {
        let bgr = BgrF32::new(0.0, 0.0, 1.0); // b=0, g=0, r=1
        let y: MonoF32 = Luminance.convert(&bgr);
        assert!(approx(y.0, 0.299, 1e-4));
    }

    #[test]
    fn luminance_bgraf32_to_monof32() {
        let src = BgraF32::new(0.0, 0.0, 1.0, 0.5);
        let y: MonoF32 = Luminance.convert(&src);
        assert!(approx(y.0, 0.299, 1e-4));
    }

    #[test]
    fn luminance_rgbf64_to_monof64() {
        let white = RgbF64::new(1.0, 1.0, 1.0);
        let y: MonoF64 = Luminance.convert(&white);
        assert!((y.0 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn luminance_bgraf64_to_monof64() {
        let src = BgraF64::new(1.0, 1.0, 1.0, 0.5);
        let y: MonoF64 = Luminance.convert(&src);
        assert!((y.0 - 1.0).abs() < 1e-6);
    }

    // ── Broadcast: MonoF32 / MonoF64 → RGB/BGR ─────────────────────────

    #[test]
    fn broadcast_monof32_to_rgbf32() {
        let dst: RgbF32 = Broadcast.convert(&MonoF32::new(0.5));
        assert_eq!(dst, RgbF32::new(0.5, 0.5, 0.5));
    }

    #[test]
    fn broadcast_monof32_to_bgrf32() {
        let dst: BgrF32 = Broadcast.convert(&MonoF32::new(0.25));
        assert_eq!(dst, BgrF32::new(0.25, 0.25, 0.25));
    }

    #[test]
    fn broadcast_monof32_to_rgbaf32() {
        let dst: RgbaF32 = Broadcast.convert(&MonoF32::new(0.5));
        assert_eq!(dst, RgbaF32::new(0.5, 0.5, 0.5, 1.0));
    }

    #[test]
    fn broadcast_monof32_to_bgraf32() {
        let dst: BgraF32 = Broadcast.convert(&MonoF32::new(0.25));
        assert_eq!(dst, BgraF32::new(0.25, 0.25, 0.25, 1.0));
    }

    #[test]
    fn broadcast_monof64_to_rgbf64() {
        let dst: RgbF64 = Broadcast.convert(&MonoF64::new(0.5));
        assert_eq!(dst, RgbF64::new(0.5, 0.5, 0.5));
    }

    #[test]
    fn broadcast_monof64_to_rgbaf64() {
        let dst: RgbaF64 = Broadcast.convert(&MonoF64::new(0.5));
        assert_eq!(dst, RgbaF64::new(0.5, 0.5, 0.5, 1.0));
    }

    // ── AddAlpha: MonoF32 / MonoF64 → MonoAF32 / MonoAF64 ──────────────

    #[test]
    fn addalpha_monof32_to_monoaf32() {
        let p: MonoAF32 = AddAlpha.convert(&MonoF32::new(0.5));
        assert_eq!(p, MonoAF32::new(0.5, 1.0));
    }

    #[test]
    fn addalpha_monof64_to_monoaf64() {
        let p: MonoAF64 = AddAlpha.convert(&MonoF64::new(0.5));
        assert_eq!(p, MonoAF64::new(0.5, 1.0));
    }

    // ── SrgbGamma: SrgbMono8 / SrgbMono16 ↔ MonoF32 ────────────────────

    #[test]
    fn srgb_gamma_srgb_mono8_to_monof32() {
        let linear: MonoF32 = SrgbGamma.convert(&SrgbMono8::new(0));
        assert_eq!(linear.0, 0.0);

        let linear: MonoF32 = SrgbGamma.convert(&SrgbMono8::new(255));
        assert!(approx(linear.0, 1.0, 0.001));
    }

    #[test]
    fn srgb_gamma_monof32_to_srgb_mono8() {
        let srgb: SrgbMono8 = SrgbGamma.convert(&MonoF32::new(0.0));
        assert_eq!(srgb.0.0, 0);

        let srgb: SrgbMono8 = SrgbGamma.convert(&MonoF32::new(1.0));
        assert_eq!(srgb.0.0, 255);
    }

    #[test]
    fn srgb_gamma_monof32_roundtrip() {
        let orig = SrgbMono8::new(128);
        let linear: MonoF32 = SrgbGamma.convert(&orig);
        let back: SrgbMono8 = SrgbGamma.convert(&linear);
        assert_eq!(orig, back);
    }

    #[test]
    fn srgb_gamma_srgb_mono16_to_monof32() {
        let linear: MonoF32 = SrgbGamma.convert(&SrgbMono16::new(0));
        assert_eq!(linear.0, 0.0);

        let linear: MonoF32 = SrgbGamma.convert(&SrgbMono16::new(65535));
        assert!(approx(linear.0, 1.0, 0.001));
    }

    #[test]
    fn srgb_gamma_monof32_to_srgb_mono16() {
        let srgb: SrgbMono16 = SrgbGamma.convert(&MonoF32::new(0.0));
        assert_eq!(srgb.0.0, 0);

        let srgb: SrgbMono16 = SrgbGamma.convert(&MonoF32::new(1.0));
        assert_eq!(srgb.0.0, 65535);
    }

    // ── convert_image with MonoF32 ──────────────────────────────────────

    #[test]
    fn convert_image_mono8_to_monof32_fullrange() {
        let img = Image::fill(2, 2, Mono8::new(255));
        let out: Image<MonoF32> = convert_image(&img, FullRange);
        assert!(approx(out.pixel_at(0, 0).0, 1.0, 1e-3));
    }

    #[test]
    fn convert_image_monof32_to_mono8_fullrange() {
        let img = Image::fill(2, 2, MonoF32::new(0.5));
        let out: Image<Mono8> = convert_image(&img, FullRange);
        assert_eq!(out.pixel_at(0, 0).value(), 128);
    }

    #[test]
    fn convert_image_srgb_mono8_to_monof32() {
        let img = Image::fill(2, 1, SrgbMono8::new(0));
        let out: Image<MonoF32> = convert_image(&img, SrgbGamma);
        assert_eq!(out.pixel_at(0, 0).0, 0.0);
    }

    #[test]
    fn convert_image_monof32_to_srgb_mono8() {
        let img = Image::fill(2, 1, MonoF32::new(1.0));
        let out: Image<SrgbMono8> = convert_image(&img, SrgbGamma);
        assert_eq!(out.pixel_at(0, 0).0.0, 255);
    }

    // ── Resize with MonoF32 ─────────────────────────────────────────────

    #[test]
    fn nearest_neighbor_resize_monof32() {
        use crate::transform::{NearestNeighbor, resize};
        let img: Image<MonoF32> = Image::fill(4, 4, MonoF32::new(0.5));
        let out: Image<MonoF32> = resize(&img, crate::Size::new(2, 2), NearestNeighbor);
        assert_eq!(out.pixel_at(0, 0), MonoF32::new(0.5));
    }

    #[test]
    fn bilinear_resize_monof32() {
        use crate::transform::{Bilinear, resize};
        let img: Image<MonoF32> = Image::fill(4, 4, MonoF32::new(0.5));
        let out: Image<MonoF32> = resize(&img, crate::Size::new(2, 2), Bilinear);
        assert!(approx(out.pixel_at(0, 0).0, 0.5, 1e-3));
    }

    // ───────────────────────────────────────────────────────────────────
    // Coverage: FullRange Mono32 ↔ MonoF64, Mono64 ↔ MonoF64
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn fullrange_mono32_to_monof64() {
        let src = Mono32::new(0);
        let dst: MonoF64 = FullRange.convert(&src);
        assert!(approx_f64(dst.0, 0.0, 1e-12));

        let src = Mono32::new(u32::MAX);
        let dst: MonoF64 = FullRange.convert(&src);
        assert!(approx_f64(dst.0, 1.0, 1e-6));

        let src = Mono32::new(u32::MAX / 2);
        let dst: MonoF64 = FullRange.convert(&src);
        assert!(
            dst.0 > 0.4 && dst.0 < 0.6,
            "midpoint should be ~0.5, got {}",
            dst.0
        );
    }

    #[test]
    fn fullrange_monof64_to_mono32() {
        let src = MonoF64::new(0.0);
        let dst: Mono32 = FullRange.convert(&src);
        assert_eq!(dst.value(), 0);

        let src = MonoF64::new(1.0);
        let dst: Mono32 = FullRange.convert(&src);
        assert_eq!(dst.value(), u32::MAX);

        let src = MonoF64::new(0.5);
        let dst: Mono32 = FullRange.convert(&src);
        let half = u32::MAX / 2;
        assert!(
            dst.value() >= half - 1 && dst.value() <= half + 1,
            "midpoint should be ~{half}, got {}",
            dst.value()
        );
    }

    #[test]
    fn fullrange_mono32_monof64_roundtrip() {
        for v in [0u32, 1, 1000, u32::MAX / 2, u32::MAX] {
            let src = Mono32::new(v);
            let mid: MonoF64 = FullRange.convert(&src);
            let dst: Mono32 = FullRange.convert(&mid);
            assert!(
                (dst.value() as i64 - v as i64).unsigned_abs() <= 1,
                "roundtrip failed for {v}: got {}",
                dst.value()
            );
        }
    }

    #[test]
    fn fullrange_mono64_to_monof64() {
        let src = Mono64::new(0);
        let dst: MonoF64 = FullRange.convert(&src);
        assert!(approx_f64(dst.0, 0.0, 1e-12));

        let src = Mono64::new(u64::MAX);
        let dst: MonoF64 = FullRange.convert(&src);
        assert!(approx_f64(dst.0, 1.0, 1e-6));

        let src = Mono64::new(u64::MAX / 2);
        let dst: MonoF64 = FullRange.convert(&src);
        assert!(
            dst.0 > 0.4 && dst.0 < 0.6,
            "midpoint should be ~0.5, got {}",
            dst.0
        );
    }

    #[test]
    fn fullrange_monof64_to_mono64() {
        let src = MonoF64::new(0.0);
        let dst: Mono64 = FullRange.convert(&src);
        assert_eq!(dst.value(), 0);

        let src = MonoF64::new(1.0);
        let dst: Mono64 = FullRange.convert(&src);
        assert_eq!(dst.value(), u64::MAX);

        let src = MonoF64::new(0.5);
        let dst: Mono64 = FullRange.convert(&src);
        let half = u64::MAX / 2;
        assert!(
            dst.value() >= half - 1 && dst.value() <= half + 1,
            "midpoint should be ~{half}, got {}",
            dst.value()
        );
    }

    #[test]
    fn fullrange_mono64_monof64_roundtrip() {
        for v in [0u64, 1, 1_000_000, u64::MAX / 2, u64::MAX] {
            let src = Mono64::new(v);
            let mid: MonoF64 = FullRange.convert(&src);
            let dst: Mono64 = FullRange.convert(&mid);
            // f64 has 53 bits of mantissa, u64 has 64 bits, so large
            // values lose precision. Allow wider tolerance for large values.
            let tol = if v > (1u64 << 53) {
                v / (1u64 << 50)
            } else {
                1
            };
            assert!(
                (dst.value() as i128 - v as i128).unsigned_abs() <= tol as u128,
                "roundtrip failed for {v}: got {}",
                dst.value()
            );
        }
    }

    // ───────────────────────────────────────────────────────────────────
    // Coverage: Luminance RgbaF64 → MonoF64, BgrF64 → MonoF64
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn luminance_rgbaf64_to_monof64() {
        // Pure white → 1.0
        let src = RgbaF64::new(1.0, 1.0, 1.0, 1.0);
        let dst: MonoF64 = Luminance.convert(&src);
        assert!(approx_f64(dst.0, 1.0, 1e-10));

        // Pure black → 0.0
        let src = RgbaF64::new(0.0, 0.0, 0.0, 1.0);
        let dst: MonoF64 = Luminance.convert(&src);
        assert!(approx_f64(dst.0, 0.0, 1e-10));

        // Pure red → 0.299 (NTSC coefficient)
        let src = RgbaF64::new(1.0, 0.0, 0.0, 1.0);
        let dst: MonoF64 = Luminance.convert(&src);
        assert!(approx_f64(dst.0, 0.299, 1e-4));

        // Pure green → 0.587
        let src = RgbaF64::new(0.0, 1.0, 0.0, 1.0);
        let dst: MonoF64 = Luminance.convert(&src);
        assert!(approx_f64(dst.0, 0.587, 1e-4));

        // Alpha is ignored by luminance
        let src = RgbaF64::new(1.0, 1.0, 1.0, 0.0);
        let dst: MonoF64 = Luminance.convert(&src);
        assert!(approx_f64(dst.0, 1.0, 1e-10));
    }

    #[test]
    fn luminance_bgrf64_to_monof64() {
        // Pure white → 1.0
        let src = BgrF64 {
            r: 1.0,
            g: 1.0,
            b: 1.0,
        };
        let dst: MonoF64 = Luminance.convert(&src);
        assert!(approx_f64(dst.0, 1.0, 1e-10));

        // Pure black → 0.0
        let src = BgrF64 {
            r: 0.0,
            g: 0.0,
            b: 0.0,
        };
        let dst: MonoF64 = Luminance.convert(&src);
        assert!(approx_f64(dst.0, 0.0, 1e-10));

        // Pure red → 0.299
        let src = BgrF64 {
            r: 1.0,
            g: 0.0,
            b: 0.0,
        };
        let dst: MonoF64 = Luminance.convert(&src);
        assert!(approx_f64(dst.0, 0.299, 1e-4));

        // Pure blue → 0.114
        let src = BgrF64 {
            r: 0.0,
            g: 0.0,
            b: 1.0,
        };
        let dst: MonoF64 = Luminance.convert(&src);
        assert!(approx_f64(dst.0, 0.114, 1e-4));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Phase 1 — Threshold family + Invert + BinaryMask
    // ═══════════════════════════════════════════════════════════════════════════

    use crate::image::BinaryImage;

    // ─── BinaryThreshold ──────────────────────────────────────────────────

    #[test]
    fn binary_threshold_mono8_above_and_below() {
        let strat = BinaryThreshold {
            thresh: Mono8::new(128),
        };
        assert_eq!(strat.convert(&Mono8::new(200)), Mono8::new(255));
        assert_eq!(strat.convert(&Mono8::new(50)), Mono8::new(0));
        // Exact threshold: "> thresh" is false, so lower branch.
        assert_eq!(strat.convert(&Mono8::new(128)), Mono8::new(0));
    }

    #[test]
    fn binary_threshold_mono8_extremes() {
        let strat = BinaryThreshold {
            thresh: Mono8::new(128),
        };
        assert_eq!(strat.convert(&Mono8::new(0)), Mono8::new(0));
        assert_eq!(strat.convert(&Mono8::new(255)), Mono8::new(255));
    }

    #[test]
    fn binary_threshold_mono16() {
        let strat = BinaryThreshold {
            thresh: Mono16::new(32768),
        };
        assert_eq!(strat.convert(&Mono16::new(40000)), Mono16::new(65535));
        assert_eq!(strat.convert(&Mono16::new(1000)), Mono16::new(0));
    }

    #[test]
    fn binary_threshold_rgb8_uniform() {
        let strat = BinaryThreshold {
            thresh: Rgb8::new(128, 128, 128),
        };
        assert_eq!(
            strat.convert(&Rgb8::new(200, 50, 200)),
            Rgb8::new(255, 0, 255)
        );
        assert_eq!(strat.convert(&Rgb8::new(10, 10, 10)), Rgb8::new(0, 0, 0));
    }

    #[test]
    fn binary_threshold_rgb8_per_channel() {
        // Per-channel thresholds — crude color keying.
        let strat = BinaryThreshold {
            thresh: Rgb8::new(200, 50, 50),
        };
        assert_eq!(
            strat.convert(&Rgb8::new(210, 100, 100)),
            Rgb8::new(255, 255, 255)
        );
        assert_eq!(
            strat.convert(&Rgb8::new(180, 40, 100)),
            Rgb8::new(0, 0, 255)
        );
    }

    // ─── BinaryThresholdInv ───────────────────────────────────────────────

    #[test]
    fn binary_threshold_inv_mono8() {
        let strat = BinaryThresholdInv {
            thresh: Mono8::new(128),
        };
        assert_eq!(strat.convert(&Mono8::new(200)), Mono8::new(0));
        assert_eq!(strat.convert(&Mono8::new(50)), Mono8::new(255));
        assert_eq!(strat.convert(&Mono8::new(128)), Mono8::new(255));
    }

    #[test]
    fn binary_threshold_inv_rgb8() {
        let strat = BinaryThresholdInv {
            thresh: Rgb8::new(128, 128, 128),
        };
        assert_eq!(
            strat.convert(&Rgb8::new(200, 50, 200)),
            Rgb8::new(0, 255, 0)
        );
    }

    // ─── TruncateThreshold ────────────────────────────────────────────────

    #[test]
    fn truncate_threshold_mono8() {
        let strat = TruncateThreshold {
            thresh: Mono8::new(128),
        };
        assert_eq!(strat.convert(&Mono8::new(200)), Mono8::new(128));
        assert_eq!(strat.convert(&Mono8::new(50)), Mono8::new(50));
        assert_eq!(strat.convert(&Mono8::new(128)), Mono8::new(128));
    }

    #[test]
    fn truncate_threshold_rgb8_per_channel() {
        let strat = TruncateThreshold {
            thresh: Rgb8::new(100, 150, 200),
        };
        assert_eq!(
            strat.convert(&Rgb8::new(50, 200, 250)),
            Rgb8::new(50, 150, 200)
        );
    }

    #[test]
    fn truncate_threshold_mono16() {
        let strat = TruncateThreshold {
            thresh: Mono16::new(1000),
        };
        assert_eq!(strat.convert(&Mono16::new(5000)), Mono16::new(1000));
        assert_eq!(strat.convert(&Mono16::new(500)), Mono16::new(500));
    }

    // ─── ToZeroThreshold / ToZeroThresholdInv ─────────────────────────────

    #[test]
    fn to_zero_threshold_mono8() {
        let strat = ToZeroThreshold {
            thresh: Mono8::new(128),
        };
        assert_eq!(strat.convert(&Mono8::new(200)), Mono8::new(200));
        assert_eq!(strat.convert(&Mono8::new(50)), Mono8::new(0));
        assert_eq!(strat.convert(&Mono8::new(128)), Mono8::new(0));
    }

    #[test]
    fn to_zero_threshold_inv_mono8() {
        let strat = ToZeroThresholdInv {
            thresh: Mono8::new(128),
        };
        assert_eq!(strat.convert(&Mono8::new(200)), Mono8::new(0));
        assert_eq!(strat.convert(&Mono8::new(50)), Mono8::new(50));
        assert_eq!(strat.convert(&Mono8::new(128)), Mono8::new(128));
    }

    #[test]
    fn to_zero_threshold_rgb8_per_channel() {
        let strat = ToZeroThreshold {
            thresh: Rgb8::new(100, 100, 100),
        };
        assert_eq!(
            strat.convert(&Rgb8::new(150, 50, 200)),
            Rgb8::new(150, 0, 200)
        );
    }

    // ─── Invert ───────────────────────────────────────────────────────────

    #[test]
    fn invert_mono8_roundtrip() {
        for v in 0u8..=255 {
            let inv = Invert.convert(&Mono8::new(v));
            assert_eq!(inv, Mono8::new(255 - v));
            // Involutive: invert(invert(x)) == x
            let back = Invert.convert(&inv);
            assert_eq!(back, Mono8::new(v));
        }
    }

    #[test]
    fn invert_mono16_extremes() {
        assert_eq!(Invert.convert(&Mono16::new(0)), Mono16::new(65535));
        assert_eq!(Invert.convert(&Mono16::new(65535)), Mono16::new(0));
        assert_eq!(Invert.convert(&Mono16::new(1000)), Mono16::new(64535));
    }

    #[test]
    fn invert_mono32_extremes() {
        assert_eq!(Invert.convert(&Mono32::new(0)), Mono32::new(u32::MAX));
        assert_eq!(Invert.convert(&Mono32::new(u32::MAX)), Mono32::new(0));
    }

    #[test]
    fn invert_rgb8_per_channel() {
        assert_eq!(
            Invert.convert(&Rgb8::new(10, 100, 250)),
            Rgb8::new(245, 155, 5)
        );
    }

    #[test]
    fn invert_rgba8_inverts_alpha_too() {
        // Alpha is a channel; Invert operates on every channel. Users who
        // want to preserve alpha should invert the RGB portion explicitly
        // (e.g. via PixelMap) — the strategy is deliberately uniform.
        assert_eq!(
            Invert.convert(&Rgba8::new(10, 100, 250, 200)),
            Rgba8::new(245, 155, 5, 55)
        );
    }

    // Negative test: Invert must NOT compile for float-channel pixel types.
    // `f32` and `f64` do not implement `BoundedChannel`, so the
    // channel-bound on `Invert`'s impl rejects them at compile time.
    //
    // We cannot express "does not compile" inside a `#[cfg(test)]` block
    // without a dedicated compile-fail harness, so we enumerate the
    // positive inventory in the helper below. If somebody mistakenly adds
    // `impl BoundedChannel for f32`, the plan's float-exclusion guarantee
    // regresses silently — but the inventory in `pixel::primitives::tests`
    // (`bounded_channel_inventory`) is the canonical guard.
    #[test]
    fn invert_compiles_for_documented_integer_types() {
        // Spot-check: each statement is rejected at compile time if the
        // corresponding impl regresses.
        let _: Mono8 = Invert.convert(&Mono8::new(1));
        let _: Mono16 = Invert.convert(&Mono16::new(1));
        let _: Mono32 = Invert.convert(&Mono32::new(1));
        let _: Mono64 = Invert.convert(&Mono64::new(1));
        let _: Rgb8 = Invert.convert(&Rgb8::new(1, 2, 3));
        let _: Rgba8 = Invert.convert(&Rgba8::new(1, 2, 3, 4));
        let _: Bgr8 = Invert.convert(&Bgr8::new(1, 2, 3));
        let _: Bgra8 = Invert.convert(&Bgra8::new(1, 2, 3, 4));
        let _: MonoA8 = Invert.convert(&MonoA8::new(1, 2));
    }

    // ─── BinaryMask (P → bool) ────────────────────────────────────────────

    #[test]
    fn binary_mask_mono8() {
        let strat = BinaryMask {
            thresh: Mono8::new(128),
        };
        assert!(strat.convert(&Mono8::new(200)));
        assert!(!strat.convert(&Mono8::new(50)));
        assert!(!strat.convert(&Mono8::new(128)));
    }

    #[test]
    fn binary_mask_mono16() {
        let strat = BinaryMask {
            thresh: Mono16::new(32768),
        };
        assert!(strat.convert(&Mono16::new(40000)));
        assert!(!strat.convert(&Mono16::new(1000)));
    }

    #[test]
    fn binary_mask_rgb8_all_channels_above() {
        // All-channels-above reduction: true only if every channel exceeds
        // the corresponding threshold channel.
        let strat = BinaryMask {
            thresh: Rgb8::new(100, 100, 100),
        };
        assert!(strat.convert(&Rgb8::new(200, 200, 200)));
        // One channel fails → false.
        assert!(!strat.convert(&Rgb8::new(200, 50, 200)));
        assert!(!strat.convert(&Rgb8::new(200, 200, 100)));
    }

    #[test]
    fn binary_mask_convert_image_produces_binary_image() {
        use crate::image::{Image, ImageView};
        let img: Image<Mono8> = Image::fill(4, 4, Mono8::new(200));
        let mask: BinaryImage = convert_image(
            &img,
            BinaryMask {
                thresh: Mono8::new(128),
            },
        );
        assert_eq!(mask.size(), img.size());
        for y in 0..mask.height() {
            for x in 0..mask.width() {
                assert!(mask.pixel_at(x, y));
            }
        }
    }

    // ─── convert_image / convert_image_into integration ───────────────────

    #[test]
    fn convert_image_binary_threshold_rgb8() {
        use crate::image::{Image, ImageView};
        let img: Image<Rgb8> = Image::generate(4, 4, |x, y| {
            // Vary per pixel: diagonal gradient.
            let v = (x * 40 + y * 20) as u8;
            Rgb8::new(v, v, v)
        });
        let out: Image<Rgb8> = convert_image(
            &img,
            BinaryThreshold {
                thresh: Rgb8::new(100, 100, 100),
            },
        );
        for y in 0..4 {
            for x in 0..4 {
                let v = (x * 40 + y * 20) as u8;
                let expected = if v > 100 { 255 } else { 0 };
                assert_eq!(out.pixel_at(x, y), Rgb8::new(expected, expected, expected));
            }
        }
    }

    #[test]
    fn convert_image_invert_mono8_varying() {
        use crate::image::{Image, ImageView};
        let img: Image<Mono8> = Image::generate(3, 3, |x, y| Mono8::new((x * 30 + y * 10) as u8));
        let out: Image<Mono8> = convert_image(&img, Invert);
        for y in 0..3 {
            for x in 0..3 {
                let v = (x * 30 + y * 10) as u8;
                assert_eq!(out.pixel_at(x, y), Mono8::new(255 - v));
            }
        }
    }

    // ─── Composability via .then() ────────────────────────────────────────

    #[test]
    fn binary_threshold_then_broadcast() {
        // BinaryThreshold on Mono8 → Broadcast to Rgb8.
        let method = BinaryThreshold {
            thresh: Mono8::new(128),
        }
        .then::<Mono8, _>(Broadcast);
        let rgb: Rgb8 = method.convert(&Mono8::new(200));
        assert_eq!(rgb, Rgb8::new(255, 255, 255));
        let rgb: Rgb8 = method.convert(&Mono8::new(50));
        assert_eq!(rgb, Rgb8::new(0, 0, 0));
    }

    #[test]
    fn invert_then_full_range_mono8_to_mono16() {
        // Invert (Mono8→Mono8) then FullRange (Mono8→Mono16).
        let method = Invert.then::<Mono8, _>(FullRange);
        let out: Mono16 = method.convert(&Mono8::new(0));
        assert_eq!(out, Mono16::new(65535));
        let out: Mono16 = method.convert(&Mono8::new(255));
        assert_eq!(out, Mono16::new(0));
    }

    // ─── ROI (SubView) input ──────────────────────────────────────────────

    #[test]
    fn convert_image_binary_threshold_on_subview_roi() {
        use crate::image::{Image, ImageView};
        let img: Image<Mono8> = Image::generate(8, 8, |x, y| Mono8::new((x * 16 + y * 8) as u8));
        // ROI: centre 4x4 block at (2, 2).
        let roi = img.roi(Rectangle::new((2, 2), (4, 4))).unwrap();
        let roi_size = roi.size();
        let out: Image<Mono8> = convert_image(
            &roi,
            BinaryThreshold {
                thresh: Mono8::new(64),
            },
        );
        assert_eq!(out.size(), roi_size);
        for y in 0..4 {
            for x in 0..4 {
                let sv = ((x + 2) * 16 + (y + 2) * 8) as u8;
                let expected = if sv > 64 { 255 } else { 0 };
                assert_eq!(out.pixel_at(x, y), Mono8::new(expected));
            }
        }
    }

    // ─── BinaryMask → morphology integration ──────────────
    //
    // Demonstrates the core value proposition of the `bool`-valued threshold
    // path: `BinaryMask` produces a `BinaryImage`, which feeds `erode` /
    // `dilate` directly — no bridging conversion, no `!= 0` test. The
    // downstream morphology call typechecks because `bool` is `Copy + Ord +
    // ZeroablePixel` and `BinaryImage = Image<bool>`.

    #[test]
    fn binary_mask_output_feeds_erode_directly() {
        use crate::border::Clamp as BorderClamp;
        use crate::image::{Image, ImageView, Neighborhood};
        use crate::transform::erode;

        // Gradient image: only the right half exceeds the threshold.
        let img: Image<Mono8> = Image::generate(8, 8, |x, _| Mono8::new((x * 32) as u8));
        let mask: BinaryImage = convert_image(
            &img,
            BinaryMask {
                thresh: Mono8::new(100),
            },
        );

        // No bridging conversion: mask (BinaryImage) goes straight into erode.
        let se = Neighborhood::<bool, 3, 3>::full_rect_3x3();
        let eroded: BinaryImage = erode(&mask, &se, &BorderClamp);

        assert_eq!(eroded.size(), mask.size());
        // The interior of the "above threshold" region stays `true` after
        // erosion; the boundary shrinks by one pixel.
        // Threshold is 100, so pixels with x*32 > 100 are true: x >= 4.
        // After 3x3 erosion (AND of 3x3 neighborhood), a pixel stays true
        // only if all 8 neighbors plus itself are true — so the leftmost
        // column of the "true" region (x == 4) gets eroded away.
        for y in 1..7 {
            assert!(!eroded.pixel_at(4, y), "x=4 y={y} should erode to false");
            for x in 5..7 {
                assert!(eroded.pixel_at(x, y), "x={x} y={y} should stay true");
            }
        }
    }

    // ─── Phase 1 composability with `.then()` ───────────────────
    //
    // PLAN Phase 1 checklist explicitly requires:
    //   "Tests: composability with `.then()` (e.g.
    //    BinaryThreshold.then::<Mono8, _>(Broadcast))"
    //
    // These tests exercise the named-strategy composition pathway for the
    // Phase 1 strategies that preserve the pixel type (`BinaryThreshold`,
    // `Invert`) and one that does not (`BinaryMask` → bool). The goal is
    // not to re-test each strategy's behaviour but to verify that the
    // combined `ConvertPixel<Src, Dst>` bound resolves cleanly.

    #[test]
    fn then_binary_threshold_broadcast_mono8_to_rgb8() {
        // BinaryThreshold<Mono8>: Mono8 → Mono8 ({0, 255})
        // Broadcast:              Mono8 → Rgb8
        let method = BinaryThreshold {
            thresh: Mono8::new(128),
        }
        .then::<Mono8, _>(Broadcast);
        let hi: Rgb8 = method.convert(&Mono8::new(200));
        let lo: Rgb8 = method.convert(&Mono8::new(50));
        assert_eq!(hi, Rgb8::new(255, 255, 255));
        assert_eq!(lo, Rgb8::new(0, 0, 0));
    }

    #[test]
    fn then_invert_broadcast_mono8_to_rgb8() {
        // Invert on Mono8, then broadcast to Rgb8.
        let method = Invert.then::<Mono8, _>(Broadcast);
        let out: Rgb8 = method.convert(&Mono8::new(10));
        assert_eq!(out, Rgb8::new(245, 245, 245));
    }

    #[test]
    fn then_invert_invert_is_identity_on_mono8() {
        // Invert ∘ Invert = identity on integer-channel pixels.
        let method = Invert.then::<Mono8, _>(Invert);
        for v in [0u8, 1, 50, 128, 200, 254, 255] {
            assert_eq!(method.convert(&Mono8::new(v)), Mono8::new(v));
        }
    }

    #[test]
    fn then_binary_threshold_invert_inverts_mask() {
        // BinaryThreshold followed by Invert is equivalent to
        // BinaryThresholdInv.
        let lhs = BinaryThreshold {
            thresh: Mono8::new(128),
        }
        .then::<Mono8, _>(Invert);
        let rhs = BinaryThresholdInv {
            thresh: Mono8::new(128),
        };
        for v in [0u8, 50, 128, 129, 200, 255] {
            assert_eq!(lhs.convert(&Mono8::new(v)), rhs.convert(&Mono8::new(v)));
        }
    }

    #[test]
    fn then_luminance_binary_threshold_rgb8_to_mono8() {
        // Compose a grayscale conversion with a binary threshold — the
        // classic "threshold a color image on its luminance" pipeline.
        let method = Luminance.then::<Mono8, _>(BinaryThreshold {
            thresh: Mono8::new(100),
        });
        // White → luminance 255 → above threshold → 255.
        assert_eq!(method.convert(&Rgb8::new(255, 255, 255)), Mono8::new(255));
        // Black → luminance 0 → below threshold → 0.
        assert_eq!(method.convert(&Rgb8::new(0, 0, 0)), Mono8::new(0));
    }

    #[test]
    fn then_luminance_binary_mask_rgb8_to_bool() {
        // Rgb8 → Mono8 (luminance) → bool (BinaryMask). This is the
        // canonical way to drop a color image into morphology / blob
        // analysis without committing to a per-channel rule.
        let method = Luminance.then::<Mono8, _>(BinaryMask {
            thresh: Mono8::new(100),
        });
        assert!(method.convert(&Rgb8::new(255, 255, 255)));
        assert!(!method.convert(&Rgb8::new(10, 20, 30)));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Phase 2 — Clamp<P> and BrightnessContrast<S>
    // ═══════════════════════════════════════════════════════════════════════════

    // ─── Clamp<P> ─────────────────────────────────────────────────────────

    // ── P1-6: Clamp::new constructor and inverted-range rejection ──────────────────────────────────────────────

    #[test]
    fn clamp_new_valid_range_constructs() {
        let strat = Clamp::new(Mono8::new(20), Mono8::new(235));
        assert_eq!(strat.lo(), Mono8::new(20));
        assert_eq!(strat.hi(), Mono8::new(235));
        // Equivalent to the struct-literal form.
        assert_eq!(strat, Clamp::new(Mono8::new(20), Mono8::new(235)));
    }

    #[test]
    fn clamp_new_equal_bounds_is_valid() {
        // lo == hi is degenerate but technically valid: every input
        // collapses to that exact value. Treated as a deliberate choice,
        // not a bug.
        let strat = Clamp::new(Mono8::new(128), Mono8::new(128));
        assert_eq!(strat.convert(&Mono8::new(0)), Mono8::new(128));
        assert_eq!(strat.convert(&Mono8::new(255)), Mono8::new(128));
    }

    #[test]
    #[should_panic(expected = "lo > hi on channel 0")]
    fn clamp_new_inverted_mono_panics() {
        let _ = Clamp::new(Mono8::new(200), Mono8::new(50));
    }

    #[test]
    #[should_panic(expected = "lo > hi on channel 1")]
    fn clamp_new_inverted_single_channel_panics_with_index() {
        // Channels 0 and 2 are fine; channel 1 (green) is inverted.
        let _ = Clamp::new(Rgb8::new(10, 200, 10), Rgb8::new(200, 50, 200));
    }

    #[test]
    fn clamp_new_rgb_all_equal_lo_hi() {
        let strat = Clamp::new(Rgb8::new(0, 0, 0), Rgb8::new(255, 255, 255));
        assert_eq!(
            strat.convert(&Rgb8::new(128, 64, 32)),
            Rgb8::new(128, 64, 32)
        );
    }

    #[test]
    fn clamp_mono8_basic() {
        let strat = Clamp::new(Mono8::new(20), Mono8::new(235));
        // Inside range — unchanged.
        assert_eq!(strat.convert(&Mono8::new(100)), Mono8::new(100));
        // Below lo — clamped up.
        assert_eq!(strat.convert(&Mono8::new(5)), Mono8::new(20));
        // Above hi — clamped down.
        assert_eq!(strat.convert(&Mono8::new(250)), Mono8::new(235));
        // Exact bounds preserved.
        assert_eq!(strat.convert(&Mono8::new(20)), Mono8::new(20));
        assert_eq!(strat.convert(&Mono8::new(235)), Mono8::new(235));
    }

    #[test]
    fn clamp_mono8_lo_equals_hi_collapses_to_constant() {
        let strat = Clamp::new(Mono8::new(128), Mono8::new(128));
        for v in [0u8, 50, 128, 200, 255] {
            assert_eq!(strat.convert(&Mono8::new(v)), Mono8::new(128));
        }
    }

    #[test]
    fn clamp_mono8_full_range_is_identity() {
        // lo = 0, hi = MAX — no channel can fall outside, so the strategy
        // is the identity.
        let strat = Clamp::new(Mono8::new(0), Mono8::new(255));
        for v in 0u8..=255 {
            assert_eq!(strat.convert(&Mono8::new(v)), Mono8::new(v));
        }
    }

    #[test]
    fn clamp_rgb8_per_channel_ranges() {
        let strat = Clamp::new(Rgb8::new(16, 16, 16), Rgb8::new(235, 240, 235));
        assert_eq!(
            strat.convert(&Rgb8::new(5, 250, 100)),
            Rgb8::new(16, 240, 100)
        );
        assert_eq!(
            strat.convert(&Rgb8::new(100, 100, 100)),
            Rgb8::new(100, 100, 100)
        );
    }

    #[test]
    fn clamp_mono16() {
        let strat = Clamp::new(Mono16::new(1000), Mono16::new(50000));
        assert_eq!(strat.convert(&Mono16::new(500)), Mono16::new(1000));
        assert_eq!(strat.convert(&Mono16::new(60000)), Mono16::new(50000));
        assert_eq!(strat.convert(&Mono16::new(25000)), Mono16::new(25000));
    }

    #[test]
    fn clamp_rgba8_includes_alpha_channel() {
        // Alpha is a channel; Clamp restricts it along with the rest.
        let strat = Clamp::new(Rgba8::new(10, 10, 10, 10), Rgba8::new(200, 200, 200, 200));
        assert_eq!(
            strat.convert(&Rgba8::new(5, 150, 220, 255)),
            Rgba8::new(10, 150, 200, 200)
        );
    }

    #[test]
    fn convert_image_clamp_mono8() {
        use crate::image::{Image, ImageView};
        let img: Image<Mono8> = Image::generate(4, 4, |x, y| Mono8::new((x * 30 + y * 20) as u8));
        let out: Image<Mono8> = convert_image(&img, Clamp::new(Mono8::new(30), Mono8::new(70)));
        for y in 0..4 {
            for x in 0..4 {
                let v = (x * 30 + y * 20) as u8;
                let expected = v.clamp(30, 70);
                assert_eq!(out.pixel_at(x, y), Mono8::new(expected));
            }
        }
    }

    // ─── BrightnessContrast<S> ────────────────────────────────────────────

    #[test]
    fn brightness_contrast_identity_mono8() {
        // contrast = 1.0, brightness = 0.0 → identity.
        let strat = BrightnessContrast {
            brightness: 0.0f32,
            contrast: 1.0f32,
        };
        for v in 0u8..=255 {
            assert_eq!(strat.convert(&Mono8::new(v)), Mono8::new(v));
        }
    }

    #[test]
    fn brightness_contrast_mono8_linear_transform() {
        let strat = BrightnessContrast {
            brightness: 10.0f32,
            contrast: 1.5f32,
        };
        // 100 * 1.5 + 10 = 160
        assert_eq!(strat.convert(&Mono8::new(100)), Mono8::new(160));
        // 0 * 1.5 + 10 = 10
        assert_eq!(strat.convert(&Mono8::new(0)), Mono8::new(10));
        // 200 * 1.5 + 10 = 310 → clamped to 255
        assert_eq!(strat.convert(&Mono8::new(200)), Mono8::new(255));
    }

    #[test]
    fn brightness_contrast_mono8_saturates_negative() {
        // Contrast > 1 + negative brightness can push results below 0.
        let strat = BrightnessContrast {
            brightness: -50.0f32,
            contrast: 1.0f32,
        };
        assert_eq!(strat.convert(&Mono8::new(100)), Mono8::new(50));
        // 30 - 50 = -20 → clamped to 0.
        assert_eq!(strat.convert(&Mono8::new(30)), Mono8::new(0));
    }

    #[test]
    fn brightness_contrast_rgb8_per_channel() {
        let strat = BrightnessContrast {
            brightness: 20.0f32,
            contrast: 2.0f32,
        };
        // Each channel: v * 2 + 20, clamped to [0, 255].
        assert_eq!(
            strat.convert(&Rgb8::new(50, 100, 10)),
            Rgb8::new(120, 220, 40)
        );
        // Saturation:
        assert_eq!(
            strat.convert(&Rgb8::new(200, 200, 200)),
            Rgb8::new(255, 255, 255)
        );
    }

    #[test]
    fn brightness_contrast_rgb16_per_channel() {
        // Multi-channel, 16-bit saturating path. Exercises the
        // derive-generated `scale_add` / `uniform` through `Saturating<u16>`
        // channels (accumulator = RgbF32), with one channel saturating high
        // and one low.
        let strat = BrightnessContrast {
            brightness: 1000.0f32,
            contrast: 2.0f32,
        };
        // (10000*2 + 1000, 50000*2 + 1000, 0*2 + 1000)
        //   = (21000, 101000→65535, 1000)
        assert_eq!(
            strat.convert(&Rgb16::new(10000, 50000, 0)),
            Rgb16::new(21000, 65535, 1000)
        );

        // Negative brightness clamps low channels to zero.
        let strat_neg = BrightnessContrast {
            brightness: -2000.0f32,
            contrast: 1.0f32,
        };
        assert_eq!(
            strat_neg.convert(&Rgb16::new(100, 5000, 65535)),
            Rgb16::new(0, 3000, 63535)
        );
    }

    #[test]
    fn brightness_contrast_mono16() {
        let strat = BrightnessContrast {
            brightness: 1000.0f32,
            contrast: 0.5f32,
        };
        // 20000 * 0.5 + 1000 = 11000
        assert_eq!(strat.convert(&Mono16::new(20000)), Mono16::new(11000));
    }

    #[test]
    fn brightness_contrast_monof32_no_clamping() {
        // MonoF32's FromLinear is the identity (Accumulator = Self), so no
        // clamping is applied. This is correct — floats have no intrinsic
        // range, and the library refuses to invent one (Philosophy §8).
        let strat = BrightnessContrast {
            brightness: 0.1f32,
            contrast: 2.0f32,
        };
        let out = strat.convert(&MonoF32(0.25));
        assert!((out.0 - (0.25 * 2.0 + 0.1)).abs() < 1e-6);
        // Values outside [0, 1] are simply preserved as the caller produced.
        let out = strat.convert(&MonoF32(1.5));
        assert!((out.0 - (1.5 * 2.0 + 0.1)).abs() < 1e-6);
    }

    #[test]
    fn brightness_contrast_f64_scalar_on_monof64() {
        // f64-precision scalar path: picks up the
        // LinearPixel<f64> impl on MonoF64 — no f32 → f64 widening.
        let strat = BrightnessContrast::<f64> {
            brightness: 0.01,
            contrast: 1.2,
        };
        let out = strat.convert(&MonoF64(0.5));
        assert!((out.0 - (0.5 * 1.2 + 0.01)).abs() < 1e-12);
    }

    #[test]
    fn brightness_contrast_f64_scalar_on_mono64() {
        // Mono64 has Accumulator = f64 and a LinearPixel<f64> impl.
        let strat = BrightnessContrast::<f64> {
            brightness: 1000.0,
            contrast: 0.5,
        };
        // 4_000_000 * 0.5 + 1000 = 2_001_000
        assert_eq!(
            strat.convert(&Mono64::new(4_000_000)),
            Mono64::new(2_001_000)
        );
    }

    #[test]
    fn convert_image_brightness_contrast() {
        use crate::image::{Image, ImageView};
        let img: Image<Mono8> = Image::generate(3, 3, |x, y| Mono8::new((x * 20 + y * 10) as u8));
        let out: Image<Mono8> = convert_image(
            &img,
            BrightnessContrast {
                brightness: 5.0f32,
                contrast: 1.5f32,
            },
        );
        for y in 0..3 {
            for x in 0..3 {
                let v = (x * 20 + y * 10) as u8;
                let expected = ((v as f32) * 1.5 + 5.0).round().clamp(0.0, 255.0) as u8;
                assert_eq!(out.pixel_at(x, y), Mono8::new(expected));
            }
        }
    }

    // ─── Composability: BrightnessContrast → Clamp ────────────────────────

    #[test]
    fn brightness_contrast_then_clamp_pipeline() {
        // Apply brightness+contrast, then clamp the result into a narrower
        // range — everything stays in Mono8, single pass, no intermediate
        // allocation (Then combinator).
        let method = BrightnessContrast {
            brightness: 10.0f32,
            contrast: 2.0f32,
        }
        .then::<Mono8, _>(Clamp::new(Mono8::new(50), Mono8::new(200)));
        // 100 * 2 + 10 = 210 → clamped to 200
        assert_eq!(method.convert(&Mono8::new(100)), Mono8::new(200));
        // 10 * 2 + 10 = 30 → clamped up to 50
        assert_eq!(method.convert(&Mono8::new(10)), Mono8::new(50));
        // 50 * 2 + 10 = 110 — already inside [50, 200]
        assert_eq!(method.convert(&Mono8::new(50)), Mono8::new(110));
    }

    #[test]
    fn clamp_then_binary_threshold_pipeline() {
        // Clamp to a lower band, then threshold — demonstrates that Phase 1
        // and Phase 2 strategies compose naturally through `.then()`.
        let method = Clamp::new(Mono8::new(0), Mono8::new(100)).then::<Mono8, _>(BinaryThreshold {
            thresh: Mono8::new(50),
        });
        // 200 → clamped to 100 → above 50 → 255
        assert_eq!(method.convert(&Mono8::new(200)), Mono8::new(255));
        // 40 → unchanged → below 50 → 0
        assert_eq!(method.convert(&Mono8::new(40)), Mono8::new(0));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Phase 3 — Lut<V> and ChannelLut
    // ═══════════════════════════════════════════════════════════════════════════

    // ─── Lut<V> — cross-type Mono8 → V ────────────────────────────────────

    #[test]
    fn lut_mono8_identity_is_no_op() {
        let lut: Lut<Mono8> = Lut::from_fn(Mono8::new);
        for v in 0u8..=255 {
            assert_eq!(lut.convert(&Mono8::new(v)), Mono8::new(v));
        }
    }

    #[test]
    fn lut_mono8_to_mono8_invert() {
        // Use a LUT to implement the same operation as `Invert`; results
        // must match byte-for-byte.
        let lut: Lut<Mono8> = Lut::from_fn(|v| Mono8::new(255 - v));
        for v in 0u8..=255 {
            assert_eq!(lut.convert(&Mono8::new(v)), Invert.convert(&Mono8::new(v)));
        }
    }

    #[test]
    fn lut_mono8_to_rgb8_pseudocolor() {
        // Classic heat-map pseudocolor.
        let lut: Lut<Rgb8> = Lut::from_fn(|v| {
            if v < 128 {
                Rgb8::new(v.saturating_mul(2), 0, 0)
            } else {
                Rgb8::new(255, (v - 128).saturating_mul(2), 0)
            }
        });
        assert_eq!(lut.convert(&Mono8::new(0)), Rgb8::new(0, 0, 0));
        assert_eq!(lut.convert(&Mono8::new(64)), Rgb8::new(128, 0, 0));
        assert_eq!(lut.convert(&Mono8::new(128)), Rgb8::new(255, 0, 0));
        assert_eq!(lut.convert(&Mono8::new(192)), Rgb8::new(255, 128, 0));
    }

    #[test]
    fn lut_mono8_to_mono16_widen() {
        // Cross-depth LUT: Mono8 index → Mono16 output.
        let lut: Lut<Mono16> = Lut::from_fn(|v| Mono16::new((v as u16) * 257));
        assert_eq!(lut.convert(&Mono8::new(0)), Mono16::new(0));
        assert_eq!(lut.convert(&Mono8::new(1)), Mono16::new(257));
        assert_eq!(lut.convert(&Mono8::new(255)), Mono16::new(65535));
    }

    #[test]
    fn lut_new_explicit_table() {
        // `Lut::new` accepts an explicit 256-entry array.
        let mut table = [Mono8::new(0); 256];
        table[100] = Mono8::new(200);
        table[200] = Mono8::new(100);
        let lut = Lut::new(table);
        assert_eq!(lut.convert(&Mono8::new(100)), Mono8::new(200));
        assert_eq!(lut.convert(&Mono8::new(200)), Mono8::new(100));
        assert_eq!(lut.convert(&Mono8::new(50)), Mono8::new(0));
    }

    #[test]
    fn lut_convert_image_mono8_to_rgb8() {
        use crate::image::{Image, ImageView};
        let lut: Lut<Rgb8> = Lut::from_fn(|v| Rgb8::new(v, 255 - v, 128));
        let gray: Image<Mono8> = Image::generate(3, 3, |x, y| Mono8::new((x * 30 + y * 10) as u8));
        let color: Image<Rgb8> = convert_image(&gray, lut);
        for y in 0..3 {
            for x in 0..3 {
                let v = (x * 30 + y * 10) as u8;
                assert_eq!(color.pixel_at(x, y), Rgb8::new(v, 255 - v, 128));
            }
        }
    }

    #[test]
    fn lut_covers_every_u8_index() {
        // Every possible Mono8 value (0..=255) must resolve to the
        // corresponding table entry — this is the "every index is valid"
        // guarantee that justifies skipping a bounds check.
        let lut: Lut<Mono8> = Lut::from_fn(|v| Mono8::new(v.wrapping_add(1)));
        for v in 0u8..=255 {
            assert_eq!(
                lut.convert(&Mono8::new(v)),
                Mono8::new(v.wrapping_add(1)),
                "wrong output for input {v}"
            );
        }
    }

    // ─── ChannelLut — per-channel u8 → u8 ─────────────────────────────────

    #[test]
    fn channel_lut_identity_mono8() {
        let lut = ChannelLut::from_fn(|v| v);
        for v in 0u8..=255 {
            assert_eq!(lut.convert(&Mono8::new(v)), Mono8::new(v));
        }
    }

    #[test]
    fn channel_lut_per_channel_on_rgb8() {
        // Simple "double with saturation" curve applied independently to
        // every channel.
        let lut = ChannelLut::from_fn(|v| ((v as u16) * 2).min(255) as u8);
        assert_eq!(
            lut.convert(&Rgb8::new(50, 100, 200)),
            Rgb8::new(100, 200, 255)
        );
    }

    #[test]
    fn channel_lut_on_rgba8_includes_alpha() {
        // Alpha is a channel; the same u8→u8 mapping applies to it too.
        let lut = ChannelLut::from_fn(|v| 255 - v);
        assert_eq!(
            lut.convert(&Rgba8::new(10, 100, 250, 200)),
            Rgba8::new(245, 155, 5, 55)
        );
    }

    #[test]
    fn channel_lut_on_bgr8() {
        // Channel ordering is irrelevant — the per-channel table applies
        // independently regardless of semantic position.
        let lut = ChannelLut::from_fn(|v| v / 2);
        assert_eq!(
            lut.convert(&Bgr8::new(100, 200, 50)),
            Bgr8::new(50, 100, 25)
        );
    }

    #[test]
    fn channel_lut_on_monoa8() {
        // Grayscale-with-alpha: both channels receive the same mapping.
        let lut = ChannelLut::from_fn(|v| v.saturating_sub(10));
        assert_eq!(lut.convert(&MonoA8::new(100, 50)), MonoA8::new(90, 40));
        assert_eq!(lut.convert(&MonoA8::new(5, 3)), MonoA8::new(0, 0));
    }

    #[test]
    fn channel_lut_lookup_returns_table_entry() {
        let lut = ChannelLut::from_fn(|i| i.wrapping_add(10));
        assert_eq!(lut.lookup(0), 10);
        assert_eq!(lut.lookup(100), 110);
        assert_eq!(lut.lookup(245), 255);
        // wrap-around past 255 (wrapping_add): 250 + 10 = 4
        assert_eq!(lut.lookup(250), 4);
    }

    #[test]
    fn channel_lut_lookup_is_const() {
        // Compile-time evaluation check: `lookup` is `const fn`.
        const TABLE: ChannelLut = ChannelLut::new([7u8; 256]);
        const SEVEN: u8 = TABLE.lookup(123);
        assert_eq!(SEVEN, 7);
    }

    #[test]
    fn channel_lut_lookup_matches_convert_per_channel() {
        // The hand-rolled per-channel application via `lookup` must
        // match what `ConvertPixel::convert` produces.
        let lut = ChannelLut::from_fn(|v| ((v as u16) * 3 / 2).min(255) as u8);
        let src = Rgb8::new(40, 130, 200);
        let via_convert = lut.convert(&src);
        let via_lookup = Rgb8::new(
            lut.lookup(src.r.0),
            lut.lookup(src.g.0),
            lut.lookup(src.b.0),
        );
        assert_eq!(via_convert, via_lookup);
    }

    #[test]
    fn channel_lut_new_const_table() {
        // `ChannelLut::new` is `const` — can be called in a const context.
        const IDENTITY: ChannelLut = {
            let mut t = [0u8; 256];
            let mut i = 0;
            while i < 256 {
                t[i] = i as u8;
                i += 1;
            }
            ChannelLut::new(t)
        };
        assert_eq!(
            IDENTITY.convert(&Rgb8::new(10, 20, 30)),
            Rgb8::new(10, 20, 30)
        );
    }

    #[test]
    fn channel_lut_convert_image_rgb8() {
        use crate::image::{Image, ImageView};
        let lut = ChannelLut::from_fn(|v| v.saturating_add(50));
        let img: Image<Rgb8> =
            Image::generate(2, 2, |x, y| Rgb8::new((x * 50) as u8, (y * 50) as u8, 100));
        let out: Image<Rgb8> = convert_image(&img, lut);
        for y in 0..2 {
            for x in 0..2 {
                let r = ((x * 50) as u8).saturating_add(50);
                let g = ((y * 50) as u8).saturating_add(50);
                let b = 100u8.saturating_add(50);
                assert_eq!(out.pixel_at(x, y), Rgb8::new(r, g, b));
            }
        }
    }

    // ─── Composability — Lut / ChannelLut in pipelines ────────────────────

    #[test]
    fn lut_then_channel_lut_pipeline() {
        // Mono8 → Rgb8 (pseudocolor via Lut) → Rgb8 (per-channel curve).
        let pseudo: Lut<Rgb8> = Lut::from_fn(|v| Rgb8::new(v, v / 2, v / 4));
        let curve = ChannelLut::from_fn(|v| v.saturating_add(10));
        let method = pseudo.then::<Rgb8, _>(curve);
        // 200 → Rgb8(200, 100, 50) → Rgb8(210, 110, 60)
        assert_eq!(method.convert(&Mono8::new(200)), Rgb8::new(210, 110, 60));
    }

    #[test]
    fn channel_lut_then_binary_threshold() {
        // ChannelLut produces a contrast-boosted Mono8, then we threshold.
        let boost = ChannelLut::from_fn(|v| ((v as u16) * 2).min(255) as u8);
        let method = boost.then::<Mono8, _>(BinaryThreshold {
            thresh: Mono8::new(128),
        });
        // 100 → 200 → above 128 → 255
        assert_eq!(method.convert(&Mono8::new(100)), Mono8::new(255));
        // 50 → 100 → below 128 → 0
        assert_eq!(method.convert(&Mono8::new(50)), Mono8::new(0));
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Mono<BITS> invariant-preservation regression tests
    // ══════════════════════════════════════════════════════════════════════════
    //
    // The bug these tests guard against:
    //
    //   `BoundedChannel::MAX` on `Saturating<u16>` is `65535`. `Mono<BITS>`
    //   uses `Saturating<u16>` as its channel type but maintains the tighter
    //   invariant that the raw channel value must fit in `BITS` bits
    //   (`value <= (1 << BITS) - 1`). If `Invert` / `BinaryThreshold` /
    //   `BinaryThresholdInv` write `P::Channel::MAX = 65535` back through
    //   `from_channels` (a layout-only primitive that does NOT validate
    //   pixel invariants), the resulting `Mono<10>` claims to be a valid
    //   `Mono<10>` while actually holding `65535` — silently violating the
    //   invariant that every constructor on `Mono<BITS>` enforces.
    //
    // The fix: `Invert` / `BinaryThreshold` / `BinaryThresholdInv`
    // bind on `WhiteChannel` (pixel-level), not `BoundedChannel` (channel-
    // type-level). `Mono<BITS>` overrides `WhiteChannel::white_channel()` to
    // return `Saturating(Self::MAX) == Saturating((1 << BITS) - 1)`, which
    // IS a valid `Mono<BITS>` value.
    //
    // These tests assert the pixel-level invariant is preserved, using the
    // public `.value()` accessor. A regression would manifest as a returned
    // `Mono<10>` with `.value() > 1023`.

    #[test]
    fn invert_preserves_mono10_invariant() {
        // Pre-fix bug: Saturating(65535) - Saturating(100) == Saturating(65435)
        // Post-fix:    Saturating(1023)  - Saturating(100) == Saturating(923)
        let inv: Mono<10> = Invert.convert(&Mono::<10>::new(100));
        assert_eq!(inv.value(), 923);
        assert!(
            inv.value() <= 1023,
            "Mono<10> invariant violated: {}",
            inv.value()
        );
    }

    #[test]
    fn invert_preserves_mono12_invariant() {
        // Mono<12> range: 0..=4095
        let inv: Mono<12> = Invert.convert(&Mono::<12>::new(100));
        assert_eq!(inv.value(), 4095 - 100);
        assert!(inv.value() <= 4095);
    }

    #[test]
    fn invert_preserves_mono14_invariant() {
        // Mono<14> range: 0..=16383
        let inv: Mono<14> = Invert.convert(&Mono::<14>::new(1234));
        assert_eq!(inv.value(), 16383 - 1234);
        assert!(inv.value() <= 16383);
    }

    #[test]
    fn invert_mono10_extremes() {
        // Zero → white (1023); white (1023) → zero.
        assert_eq!(Invert.convert(&Mono::<10>::new(0)).value(), 1023);
        assert_eq!(Invert.convert(&Mono::<10>::new(1023)).value(), 0);
    }

    #[test]
    fn invert_mono12_extremes() {
        assert_eq!(Invert.convert(&Mono::<12>::new(0)).value(), 4095);
        assert_eq!(Invert.convert(&Mono::<12>::new(4095)).value(), 0);
    }

    #[test]
    fn invert_mono14_extremes() {
        assert_eq!(Invert.convert(&Mono::<14>::new(0)).value(), 16383);
        assert_eq!(Invert.convert(&Mono::<14>::new(16383)).value(), 0);
    }

    #[test]
    fn invert_mono10_all_values_stay_in_range() {
        // Exhaustive sweep across the entire Mono<10> input domain.
        // Any value above 1023 in the output is a regression.
        for v in 0u16..=1023 {
            let inv: Mono<10> = Invert.convert(&Mono::<10>::new(v));
            assert!(
                inv.value() <= 1023,
                "Mono<10> invariant violated at input {}: output = {}",
                v,
                inv.value()
            );
            // And it is the correct negation.
            assert_eq!(inv.value(), 1023 - v, "wrong inversion at input {}", v);
        }
    }

    #[test]
    fn invert_mono12_all_values_stay_in_range() {
        // Exhaustive sweep across the entire Mono<12> input domain (0..=4095).
        for v in 0u16..=4095 {
            let inv: Mono<12> = Invert.convert(&Mono::<12>::new(v));
            assert!(
                inv.value() <= 4095,
                "out-of-range at {}: {}",
                v,
                inv.value()
            );
            assert_eq!(inv.value(), 4095 - v);
        }
    }

    #[test]
    fn invert_mono_bits_is_involutive() {
        // invert(invert(x)) == x, across all three BITS values.
        for v in [0u16, 1, 42, 500, 1023] {
            let p: Mono<10> = Mono::<10>::new(v);
            assert_eq!(Invert.convert(&Invert.convert(&p)), p);
        }
        for v in [0u16, 1, 42, 2000, 4095] {
            let p: Mono<12> = Mono::<12>::new(v);
            assert_eq!(Invert.convert(&Invert.convert(&p)), p);
        }
        for v in [0u16, 1, 42, 10_000, 16383] {
            let p: Mono<14> = Mono::<14>::new(v);
            assert_eq!(Invert.convert(&Invert.convert(&p)), p);
        }
    }

    #[test]
    fn binary_threshold_respects_mono10_max() {
        // Pre-fix bug: "above threshold" writes Saturating(65535), which
        // is outside Mono<10>'s valid range.
        // Post-fix:     "above threshold" writes Saturating(1023).
        let strat = BinaryThreshold {
            thresh: Mono::<10>::new(100),
        };
        let hi: Mono<10> = strat.convert(&Mono::<10>::new(500));
        assert_eq!(hi.value(), 1023);
        assert!(hi.value() <= 1023);

        let lo: Mono<10> = strat.convert(&Mono::<10>::new(50));
        assert_eq!(lo.value(), 0);
    }

    #[test]
    fn binary_threshold_respects_mono12_max() {
        let strat = BinaryThreshold {
            thresh: Mono::<12>::new(1000),
        };
        let hi: Mono<12> = strat.convert(&Mono::<12>::new(2000));
        assert_eq!(hi.value(), 4095);
        assert!(hi.value() <= 4095);
    }

    #[test]
    fn binary_threshold_respects_mono14_max() {
        let strat = BinaryThreshold {
            thresh: Mono::<14>::new(5000),
        };
        let hi: Mono<14> = strat.convert(&Mono::<14>::new(10_000));
        assert_eq!(hi.value(), 16383);
        assert!(hi.value() <= 16383);
    }

    #[test]
    fn binary_threshold_inv_respects_mono10_max() {
        // Inverted variant: "at or below threshold" writes white.
        let strat = BinaryThresholdInv {
            thresh: Mono::<10>::new(100),
        };
        let lo: Mono<10> = strat.convert(&Mono::<10>::new(50));
        assert_eq!(lo.value(), 1023);
        assert!(lo.value() <= 1023);

        let hi: Mono<10> = strat.convert(&Mono::<10>::new(500));
        assert_eq!(hi.value(), 0);
    }

    #[test]
    fn binary_threshold_inv_respects_mono12_max() {
        let strat = BinaryThresholdInv {
            thresh: Mono::<12>::new(1000),
        };
        let lo: Mono<12> = strat.convert(&Mono::<12>::new(500));
        assert_eq!(lo.value(), 4095);
        assert!(lo.value() <= 4095);
    }

    #[test]
    fn binary_threshold_mono10_exhaustive_invariant() {
        // Sweep: every input produces an output whose .value() is either
        // 0 or exactly (1 << 10) - 1 = 1023. Never 65535 (which is what
        // the pre-fix impl would have written).
        let strat = BinaryThreshold {
            thresh: Mono::<10>::new(512),
        };
        for v in 0u16..=1023 {
            let out: Mono<10> = strat.convert(&Mono::<10>::new(v));
            let ov = out.value();
            assert!(
                ov == 0 || ov == 1023,
                "Mono<10> binary threshold produced invalid value {} at input {}",
                ov,
                v
            );
        }
    }

    // ─── Composition: Invert / BinaryThreshold on Mono<BITS> through convert_image
    // These exercise the full driver path (Image<Mono<BITS>> → Image<Mono<BITS>>),
    // ensuring the invariant is preserved end-to-end, not just on a single pixel.

    #[test]
    fn convert_image_invert_mono10_preserves_invariant() {
        use crate::image::{Image, ImageView};
        let img: Image<Mono<10>> = Image::fill(4, 4, Mono::<10>::new(100));
        let out: Image<Mono<10>> = convert_image(&img, Invert);
        for y in 0..out.height() {
            for x in 0..out.width() {
                let v = out.pixel_at(x, y).value();
                assert_eq!(v, 923);
                assert!(v <= 1023);
            }
        }
    }

    #[test]
    fn convert_image_binary_threshold_mono12_preserves_invariant() {
        use crate::image::{Image, ImageView};
        let img: Image<Mono<12>> = Image::fill(4, 4, Mono::<12>::new(2000));
        let out: Image<Mono<12>> = convert_image(
            &img,
            BinaryThreshold {
                thresh: Mono::<12>::new(1000),
            },
        );
        for y in 0..out.height() {
            for x in 0..out.width() {
                let v = out.pixel_at(x, y).value();
                assert_eq!(v, 4095);
                assert!(v <= 4095);
            }
        }
    }

    // ─── WhiteChannel inventory ────────────────────────────────────────────
    //
    // Positive: every integer-channel homogeneous pixel the library ships
    // must implement `WhiteChannel` (via derive for standard-range pixels,
    // via manual override for `Mono<BITS>`). If a new pixel type is added
    // without the `WhiteChannel` derive, the test below stops compiling.
    //
    // Negative: float-channel pixels (MonoF32/MonoF64/RgbF32/…) must NOT
    // implement WhiteChannel. That absence is load-bearing — it is what
    // makes `Invert`, `BinaryThreshold`, and `BinaryThresholdInv` refuse
    // to compile for float pixels. We cannot express "does not
    // compile" without a compile-fail harness, so the inventory below
    // serves as the living positive-list.
    #[test]
    fn white_channel_inventory_integer_pixels() {
        fn assert_white_channel<P: WhiteChannel>() {}

        // Monochrome (fixed-depth)
        assert_white_channel::<Mono8>();
        assert_white_channel::<Mono16>();
        assert_white_channel::<Mono32>();
        assert_white_channel::<Mono64>();

        // Mono<BITS> — manual override (preserves the reduced-range invariant)
        assert_white_channel::<Mono<10>>();
        assert_white_channel::<Mono<12>>();
        assert_white_channel::<Mono<14>>();

        // Monochrome + alpha
        assert_white_channel::<MonoA8>();
        assert_white_channel::<MonoA16>();
        assert_white_channel::<MonoA32>();
        assert_white_channel::<MonoA64>();

        // RGB / RGBA
        assert_white_channel::<Rgb8>();
        assert_white_channel::<Rgb16>();
        assert_white_channel::<Rgb32>();
        assert_white_channel::<Rgb64>();
        assert_white_channel::<Rgba8>();
        assert_white_channel::<Rgba16>();
        assert_white_channel::<Rgba32>();
        assert_white_channel::<Rgba64>();

        // BGR / BGRA
        assert_white_channel::<Bgr8>();
        assert_white_channel::<Bgr16>();
        assert_white_channel::<Bgr32>();
        assert_white_channel::<Bgr64>();
        assert_white_channel::<Bgra8>();
        assert_white_channel::<Bgra16>();
        assert_white_channel::<Bgra32>();
        assert_white_channel::<Bgra64>();

        // sRGB
        assert_white_channel::<Srgb8>();
        assert_white_channel::<Srgb16>();
        assert_white_channel::<Srgba8>();
        assert_white_channel::<Srgba16>();
        assert_white_channel::<SrgbMono8>();
        assert_white_channel::<SrgbMono16>();
        assert_white_channel::<SrgbMonoA8>();
        assert_white_channel::<SrgbMonoA16>();

        // Indexed
        assert_white_channel::<Indexed8>();

        // Float pixels (MonoF32/MonoF64, RgbF32/.../BgraF64, MonoAF32/MonoAF64)
        // are intentionally omitted — they do not implement `WhiteChannel`,
        // and that absence is what rejects `Invert` / `BinaryThreshold` /
        // `BinaryThresholdInv` on float pixels at compile time.
    }

    #[test]
    fn white_channel_values_mono_bits() {
        // Spot-check: `Mono<BITS>::white_channel()` must return the
        // pixel-level max, not the channel type's storage max.
        assert_eq!(
            <Mono<10> as WhiteChannel>::white_channel(),
            std::num::Saturating(1023u16)
        );
        assert_eq!(
            <Mono<12> as WhiteChannel>::white_channel(),
            std::num::Saturating(4095u16)
        );
        assert_eq!(
            <Mono<14> as WhiteChannel>::white_channel(),
            std::num::Saturating(16383u16)
        );
    }

    #[test]
    fn white_channel_values_integer_pixels_match_channel_max() {
        // For every non-reduced-range integer pixel, `white_channel()`
        // must equal the channel type's `BoundedChannel::MAX`. This is
        // the derive contract — verified here so it cannot drift.
        use crate::pixel::BoundedChannel;
        assert_eq!(
            <Mono8 as WhiteChannel>::white_channel(),
            <<Mono8 as HomogeneousPixel>::Channel as BoundedChannel>::MAX
        );
        assert_eq!(
            <Mono16 as WhiteChannel>::white_channel(),
            <<Mono16 as HomogeneousPixel>::Channel as BoundedChannel>::MAX
        );
        assert_eq!(
            <Rgb8 as WhiteChannel>::white_channel(),
            <<Rgb8 as HomogeneousPixel>::Channel as BoundedChannel>::MAX
        );
        assert_eq!(
            <Rgba16 as WhiteChannel>::white_channel(),
            <<Rgba16 as HomogeneousPixel>::Channel as BoundedChannel>::MAX
        );
        assert_eq!(
            <Bgr8 as WhiteChannel>::white_channel(),
            <<Bgr8 as HomogeneousPixel>::Channel as BoundedChannel>::MAX
        );
        assert_eq!(
            <MonoA32 as WhiteChannel>::white_channel(),
            <<MonoA32 as HomogeneousPixel>::Channel as BoundedChannel>::MAX
        );
    }
}

//! Pixel types and the traits that make image operations type-safe.
//!
//! Start with concrete pixels such as [`Srgb8`](crate::pixel::Srgb8), [`RgbF32`](crate::pixel::RgbF32), [`Mono16`](crate::pixel::Mono16), or
//! `Mono<12>`. Reach for traits when you are writing generic algorithms:
//!
//! | Trait | What it promises | Typical use |
//! |---|---|---|
//! | [`PlainPixel`](crate::pixel::PlainPixel) | stable byte layout, no padding, no invalid bit patterns | raw camera bytes, FFI, GPU upload |
//! | [`HomogeneousPixel`](crate::pixel::HomogeneousPixel) | every channel has the same channel type | histograms and per-channel transforms |
//! | [`LinearPixel`](crate::pixel::LinearPixel) | arithmetic support for interpolation/blending | filters, resize, combine operations |
//! | [`LinearSpace`](crate::pixel::LinearSpace) | the values are in a linear space | prevents gamma-incorrect blending |
//! | [`ZeroablePixel`](crate::pixel::ZeroablePixel) | an all-zero pixel exists | image allocation and output buffers |
//!
//! The most important distinction is semantic, not structural: [`Srgb8`](crate::pixel::Srgb8) is
//! gamma-encoded display/file data, while [`Rgb8`](crate::pixel::Rgb8) is linear-light RGB. They may
//! occupy similar bytes, but they are not the same pixel type.
//!
//! Do not use this module to choose storage access. For ownership, borrowing,
//! rows, slices, ROIs, and tiles, use [`crate::image`].
//!
//! For conversion strategies and common paths between pixel types,
//! see [`crate::guide::pixel_conversions`].

mod indexed;
mod label;
mod mono;
mod mono_alpha;
mod pixel_traits;
mod primitives;
mod rgb;
mod srgb;

#[cfg(test)]
mod family_tests;
#[cfg(test)]
mod tests;

// ── Float hash canonicalization helpers ──────────────────────────────────────
// Used by manual Hash impls in mono.rs, rgb.rs, and mono_alpha.rs.

/// Canonicalize an `f32` for hashing so that `a == b` implies `hash(a) == hash(b)`.
///
/// - Both `+0.0` and `-0.0` map to `0u32` (since `0.0 == -0.0`).
/// - All NaN bit patterns map to a single canonical quiet NaN.
/// - All other values use `to_bits()` directly.
#[inline]
pub(crate) fn canonicalize_f32(v: f32) -> u32 {
    if v == 0.0 {
        0u32
    } else if v.is_nan() {
        0x7FC0_0000
    } else {
        v.to_bits()
    }
}

/// Canonicalize an `f64` for hashing so that `a == b` implies `hash(a) == hash(b)`.
#[inline]
pub(crate) fn canonicalize_f64(v: f64) -> u64 {
    if v == 0.0 {
        0u64
    } else if v.is_nan() {
        0x7FF8_0000_0000_0000
    } else {
        v.to_bits()
    }
}

// ── Re-exports ──────────────────────────────────────────────────────────────

pub use mono::{Mono, Mono8, Mono10, Mono12, Mono14, Mono16, Mono32, Mono64, MonoF32, MonoF64};

pub use mono_alpha::{MonoA8, MonoA16, MonoA32, MonoA64, MonoAF32, MonoAF64};

pub use rgb::{
    Bgr, Bgr8, Bgr10, Bgr12, Bgr14, Bgr16, Bgr32, Bgr64, BgrF32, BgrF64, Bgra, Bgra8, Bgra10,
    Bgra12, Bgra14, Bgra16, Bgra32, Bgra64, BgraF32, BgraF64, Rgb, Rgb8, Rgb10, Rgb12, Rgb14,
    Rgb16, Rgb32, Rgb64, RgbF32, RgbF64, Rgba, Rgba8, Rgba10, Rgba12, Rgba14, Rgba16, Rgba32,
    Rgba64, RgbaF32, RgbaF64,
};

pub use srgb::{
    Srgb8, Srgb16, SrgbBgr8, SrgbBgr16, SrgbBgra8, SrgbBgra16, SrgbMono8, SrgbMono16, SrgbMonoA8,
    SrgbMonoA16, Srgba8, Srgba16,
};

pub use indexed::Indexed8;

pub use label::Label32;

pub use pixel_traits::{
    Array, BoundedChannel, FromLinear, HomogeneousPixel, IntegralPixel, IntegralSquaredPixel,
    LinearChannel, LinearPixel, LinearSpace, OriginInvariantPixel, PlainChannel, PlainPixel,
    WhiteChannel, ZeroablePixel, blend,
};

pub use pixel_traits::LabelPixel;

pub(crate) use pixel_traits::MAX_PIXEL_SIZE;
pub(crate) use pixel_traits::impl_origin_invariant_pixel;

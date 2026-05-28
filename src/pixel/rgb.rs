//! RGB, RGBA, BGR, and BGRA pixel types.
//!
//! Fixed-depth variants (8, 16, 32, 64-bit per channel), const-generic
//! variants (`Rgb<BITS>`, etc. for 10/12/14-bit), and floating-point
//! variants (F32, F64).

use fovea_derive::{HomogeneousPixel, LinearPixel, PlainPixel, WhiteChannel, ZeroablePixel};

use crate::pixel::{
    HomogeneousPixel, IntegralPixel, IntegralSquaredPixel, PlainChannel, PlainPixel, ZeroablePixel,
};
use std::{
    hash::{Hash, Hasher},
    num::Saturating,
};

use super::mono::Mono;
use super::{canonicalize_f32, canonicalize_f64};

// ═══════════════════════════════════════════════════════════════════════════════
// RGB pixel types
// ═══════════════════════════════════════════════════════════════════════════════

/// The `Rgb8` struct represents an RGB pixel with 8-bit depth per channel.
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
    LinearPixel,
    WhiteChannel,
)]
#[linear(accumulator = RgbF32)]
pub struct Rgb8 {
    pub r: Saturating<u8>,
    pub g: Saturating<u8>,
    pub b: Saturating<u8>,
}
impl Rgb8 {
    pub fn new(r: u8, g: u8, b: u8) -> Self {
        Rgb8 {
            r: Saturating(r),
            g: Saturating(g),
            b: Saturating(b),
        }
    }
}

/// The `Rgba8` struct represents an RGBA pixel with 8-bit depth per channel.
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
    LinearPixel,
    WhiteChannel,
)]
#[linear(accumulator = RgbaF32)]
pub struct Rgba8 {
    pub r: Saturating<u8>,
    pub g: Saturating<u8>,
    pub b: Saturating<u8>,
    pub a: Saturating<u8>,
}
impl Rgba8 {
    pub fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Rgba8 {
            r: Saturating(r),
            g: Saturating(g),
            b: Saturating(b),
            a: Saturating(a),
        }
    }
}

/// The `Rgb16` struct represents an RGB pixel with 16-bit depth per channel.
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
    LinearPixel,
    WhiteChannel,
)]
#[linear(accumulator = RgbF32)]
pub struct Rgb16 {
    pub r: Saturating<u16>,
    pub g: Saturating<u16>,
    pub b: Saturating<u16>,
}
impl Rgb16 {
    pub fn new(r: u16, g: u16, b: u16) -> Self {
        Rgb16 {
            r: Saturating(r),
            g: Saturating(g),
            b: Saturating(b),
        }
    }
}

/// The `Rgba16` struct represents an RGBA pixel with 16-bit depth per channel.
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
    LinearPixel,
    WhiteChannel,
)]
#[linear(accumulator = RgbaF32)]
pub struct Rgba16 {
    pub r: Saturating<u16>,
    pub g: Saturating<u16>,
    pub b: Saturating<u16>,
    pub a: Saturating<u16>,
}
impl Rgba16 {
    pub fn new(r: u16, g: u16, b: u16, a: u16) -> Self {
        Rgba16 {
            r: Saturating(r),
            g: Saturating(g),
            b: Saturating(b),
            a: Saturating(a),
        }
    }
}

/// The `Rgb32` struct represents an RGB pixel with 32-bit depth per channel.
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
    LinearPixel,
    WhiteChannel,
)]
#[linear(accumulator = RgbF64)]
pub struct Rgb32 {
    pub r: Saturating<u32>,
    pub g: Saturating<u32>,
    pub b: Saturating<u32>,
}
impl Rgb32 {
    pub fn new(r: u32, g: u32, b: u32) -> Self {
        Rgb32 {
            r: Saturating(r),
            g: Saturating(g),
            b: Saturating(b),
        }
    }
}

/// The `Rgba32` struct represents an RGBA pixel with 32-bit depth per channel.
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
    LinearPixel,
    WhiteChannel,
)]
#[linear(accumulator = RgbaF64)]
pub struct Rgba32 {
    pub r: Saturating<u32>,
    pub g: Saturating<u32>,
    pub b: Saturating<u32>,
    pub a: Saturating<u32>,
}
impl Rgba32 {
    pub fn new(r: u32, g: u32, b: u32, a: u32) -> Self {
        Rgba32 {
            r: Saturating(r),
            g: Saturating(g),
            b: Saturating(b),
            a: Saturating(a),
        }
    }
}

/// The `Rgb64` struct represents an RGB pixel with 64-bit depth per channel.
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
    LinearPixel,
    WhiteChannel,
)]
#[linear(accumulator = RgbF64)]
pub struct Rgb64 {
    pub r: Saturating<u64>,
    pub g: Saturating<u64>,
    pub b: Saturating<u64>,
}
impl Rgb64 {
    pub fn new(r: u64, g: u64, b: u64) -> Self {
        Rgb64 {
            r: Saturating(r),
            g: Saturating(g),
            b: Saturating(b),
        }
    }
}

/// The `Rgba64` struct represents an RGBA pixel with 64-bit depth per channel.
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
    LinearPixel,
    WhiteChannel,
)]
#[linear(accumulator = RgbaF64)]
pub struct Rgba64 {
    pub r: Saturating<u64>,
    pub g: Saturating<u64>,
    pub b: Saturating<u64>,
    pub a: Saturating<u64>,
}
impl Rgba64 {
    pub fn new(r: u64, g: u64, b: u64, a: u64) -> Self {
        Rgba64 {
            r: Saturating(r),
            g: Saturating(g),
            b: Saturating(b),
            a: Saturating(a),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Const-generic RGB / RGBA (10-bit, 12-bit, 14-bit)
// ═══════════════════════════════════════════════════════════════════════════════

/// The `Rgb` struct represents an RGB pixel with 10-bit, 12-bit, or 14-bit depth per channel.
/// This format is often used for industrial cameras.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, LinearPixel)]
#[linear(accumulator = RgbF32)]
pub struct Rgb<const BITS: usize> {
    #[linear(nested)]
    pub r: Mono<BITS>,
    #[linear(nested)]
    pub g: Mono<BITS>,
    #[linear(nested)]
    pub b: Mono<BITS>,
}
impl<const BITS: usize> Rgb<BITS> {
    pub fn new(r: u16, g: u16, b: u16) -> Self {
        Rgb {
            r: Mono::new(r),
            g: Mono::new(g),
            b: Mono::new(b),
        }
    }
}

pub type Rgb10 = Rgb<10>;
pub type Rgb12 = Rgb<12>;
pub type Rgb14 = Rgb<14>;

/// The `Rgba` struct represents an RGBA pixel with 10-bit, 12-bit, or 14-bit depth per channel.
/// This format is quite unusual, but is included for completeness.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, LinearPixel)]
#[linear(accumulator = RgbaF32)]
pub struct Rgba<const BITS: usize> {
    #[linear(nested)]
    pub r: Mono<BITS>,
    #[linear(nested)]
    pub g: Mono<BITS>,
    #[linear(nested)]
    pub b: Mono<BITS>,
    #[linear(nested)]
    pub a: Mono<BITS>,
}
impl<const BITS: usize> Rgba<BITS> {
    pub fn new(r: u16, g: u16, b: u16, a: u16) -> Self {
        Rgba {
            r: Mono::new(r),
            g: Mono::new(g),
            b: Mono::new(b),
            a: Mono::new(a),
        }
    }
}

pub type Rgba10 = Rgba<10>;
pub type Rgba12 = Rgba<12>;
pub type Rgba14 = Rgba<14>;

// ═══════════════════════════════════════════════════════════════════════════════
// Floating-point RGB / RGBA
// ═══════════════════════════════════════════════════════════════════════════════

/// The `RgbF32` struct represents an RGB pixel with 32-bit floating point depth per channel.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, PlainPixel, HomogeneousPixel, ZeroablePixel, LinearPixel,
)]
#[linear(accumulator = Self)]
pub struct RgbF32 {
    // ADR-0044 + ADR-0046: inner `f32` is a channel, not a pixel.
    #[zero(default)]
    pub r: f32,
    #[zero(default)]
    pub g: f32,
    #[zero(default)]
    pub b: f32,
}
impl RgbF32 {
    pub fn new(r: f32, g: f32, b: f32) -> Self {
        RgbF32 { r, g, b }
    }
}

impl Hash for RgbF32 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        canonicalize_f32(self.r).hash(state);
        canonicalize_f32(self.g).hash(state);
        canonicalize_f32(self.b).hash(state);
    }
}

/// The `RgbaF32` struct represents an RGBA pixel with 32-bit floating point depth per channel.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, PlainPixel, HomogeneousPixel, ZeroablePixel, LinearPixel,
)]
#[linear(accumulator = Self)]
pub struct RgbaF32 {
    // ADR-0044 + ADR-0046: inner `f32` is a channel, not a pixel.
    #[zero(default)]
    pub r: f32,
    #[zero(default)]
    pub g: f32,
    #[zero(default)]
    pub b: f32,
    #[zero(default)]
    pub a: f32,
}
impl RgbaF32 {
    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        RgbaF32 { r, g, b, a }
    }
}

impl Hash for RgbaF32 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        canonicalize_f32(self.r).hash(state);
        canonicalize_f32(self.g).hash(state);
        canonicalize_f32(self.b).hash(state);
        canonicalize_f32(self.a).hash(state);
    }
}

/// The `RgbF64` struct represents an RGB pixel with 64-bit floating point depth per channel.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, PlainPixel, HomogeneousPixel, ZeroablePixel, LinearPixel,
)]
#[linear(accumulator = Self)]
pub struct RgbF64 {
    // ADR-0044 + ADR-0046: inner `f64` is a channel, not a pixel.
    #[zero(default)]
    pub r: f64,
    #[zero(default)]
    pub g: f64,
    #[zero(default)]
    pub b: f64,
}
impl RgbF64 {
    pub fn new(r: f64, g: f64, b: f64) -> Self {
        RgbF64 { r, g, b }
    }
}

impl Hash for RgbF64 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        canonicalize_f64(self.r).hash(state);
        canonicalize_f64(self.g).hash(state);
        canonicalize_f64(self.b).hash(state);
    }
}

/// The `RgbaF64` struct represents an RGBA pixel with 64-bit floating point depth per channel.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, PlainPixel, HomogeneousPixel, ZeroablePixel, LinearPixel,
)]
#[linear(accumulator = Self)]
pub struct RgbaF64 {
    // ADR-0044 + ADR-0046: inner `f64` is a channel, not a pixel.
    #[zero(default)]
    pub r: f64,
    #[zero(default)]
    pub g: f64,
    #[zero(default)]
    pub b: f64,
    #[zero(default)]
    pub a: f64,
}
impl RgbaF64 {
    pub fn new(r: f64, g: f64, b: f64, a: f64) -> Self {
        RgbaF64 { r, g, b, a }
    }
}

impl Hash for RgbaF64 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        canonicalize_f64(self.r).hash(state);
        canonicalize_f64(self.g).hash(state);
        canonicalize_f64(self.b).hash(state);
        canonicalize_f64(self.a).hash(state);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// BGR pixel types
// ═══════════════════════════════════════════════════════════════════════════════

/// The `Bgr8` struct represents a BGR pixel with 8-bit depth per channel.
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
    LinearPixel,
    WhiteChannel,
)]
#[linear(accumulator = BgrF32)]
pub struct Bgr8 {
    pub b: Saturating<u8>,
    pub g: Saturating<u8>,
    pub r: Saturating<u8>,
}
impl Bgr8 {
    pub fn new(b: u8, g: u8, r: u8) -> Self {
        Bgr8 {
            b: Saturating(b),
            g: Saturating(g),
            r: Saturating(r),
        }
    }
}

/// The `Bgr16` struct represents a BGR pixel with 16-bit depth per channel.
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
    LinearPixel,
    WhiteChannel,
)]
#[linear(accumulator = BgrF32)]
pub struct Bgr16 {
    pub b: Saturating<u16>,
    pub g: Saturating<u16>,
    pub r: Saturating<u16>,
}
impl Bgr16 {
    pub fn new(b: u16, g: u16, r: u16) -> Self {
        Bgr16 {
            b: Saturating(b),
            g: Saturating(g),
            r: Saturating(r),
        }
    }
}

/// The `Bgr32` struct represents a BGR pixel with 32-bit depth per channel.
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
    LinearPixel,
    WhiteChannel,
)]
#[linear(accumulator = BgrF64)]
pub struct Bgr32 {
    pub b: Saturating<u32>,
    pub g: Saturating<u32>,
    pub r: Saturating<u32>,
}
impl Bgr32 {
    pub fn new(b: u32, g: u32, r: u32) -> Self {
        Bgr32 {
            b: Saturating(b),
            g: Saturating(g),
            r: Saturating(r),
        }
    }
}

/// The `Bgr64` struct represents a BGR pixel with 64-bit depth per channel.
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
    LinearPixel,
    WhiteChannel,
)]
#[linear(accumulator = BgrF64)]
pub struct Bgr64 {
    pub b: Saturating<u64>,
    pub g: Saturating<u64>,
    pub r: Saturating<u64>,
}
impl Bgr64 {
    pub fn new(b: u64, g: u64, r: u64) -> Self {
        Bgr64 {
            b: Saturating(b),
            g: Saturating(g),
            r: Saturating(r),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// BGRA pixel types
// ═══════════════════════════════════════════════════════════════════════════════

/// The `Bgra8` struct represents a BGRA pixel with 8-bit depth per channel.
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
    LinearPixel,
    WhiteChannel,
)]
#[linear(accumulator = BgraF32)]
pub struct Bgra8 {
    pub b: Saturating<u8>,
    pub g: Saturating<u8>,
    pub r: Saturating<u8>,
    pub a: Saturating<u8>,
}
impl Bgra8 {
    pub fn new(b: u8, g: u8, r: u8, a: u8) -> Self {
        Bgra8 {
            b: Saturating(b),
            g: Saturating(g),
            r: Saturating(r),
            a: Saturating(a),
        }
    }
}

/// The `Bgra16` struct represents a BGRA pixel with 16-bit depth per channel.
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
    LinearPixel,
    WhiteChannel,
)]
#[linear(accumulator = BgraF32)]
pub struct Bgra16 {
    pub b: Saturating<u16>,
    pub g: Saturating<u16>,
    pub r: Saturating<u16>,
    pub a: Saturating<u16>,
}
impl Bgra16 {
    pub fn new(b: u16, g: u16, r: u16, a: u16) -> Self {
        Bgra16 {
            b: Saturating(b),
            g: Saturating(g),
            r: Saturating(r),
            a: Saturating(a),
        }
    }
}

/// The `Bgra32` struct represents a BGRA pixel with 32-bit depth per channel.
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
    LinearPixel,
    WhiteChannel,
)]
#[linear(accumulator = BgraF64)]
pub struct Bgra32 {
    pub b: Saturating<u32>,
    pub g: Saturating<u32>,
    pub r: Saturating<u32>,
    pub a: Saturating<u32>,
}
impl Bgra32 {
    pub fn new(b: u32, g: u32, r: u32, a: u32) -> Self {
        Bgra32 {
            b: Saturating(b),
            g: Saturating(g),
            r: Saturating(r),
            a: Saturating(a),
        }
    }
}

/// The `Bgra64` struct represents a BGRA pixel with 64-bit depth per channel.
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
    LinearPixel,
    WhiteChannel,
)]
#[linear(accumulator = BgraF64)]
pub struct Bgra64 {
    pub b: Saturating<u64>,
    pub g: Saturating<u64>,
    pub r: Saturating<u64>,
    pub a: Saturating<u64>,
}
impl Bgra64 {
    pub fn new(b: u64, g: u64, r: u64, a: u64) -> Self {
        Bgra64 {
            b: Saturating(b),
            g: Saturating(g),
            r: Saturating(r),
            a: Saturating(a),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Const-generic BGR / BGRA (10-bit, 12-bit, 14-bit)
// ═══════════════════════════════════════════════════════════════════════════════

/// The `Bgr` struct represents a BGR pixel with 10-bit, 12-bit, or 14-bit depth per channel.
/// This format is often used for industrial cameras.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, LinearPixel)]
#[linear(accumulator = BgrF32)]
pub struct Bgr<const BITS: usize> {
    #[linear(nested)]
    pub b: Mono<BITS>,
    #[linear(nested)]
    pub g: Mono<BITS>,
    #[linear(nested)]
    pub r: Mono<BITS>,
}
impl<const BITS: usize> Bgr<BITS> {
    pub fn new(b: u16, g: u16, r: u16) -> Self {
        Bgr {
            b: Mono::new(b),
            g: Mono::new(g),
            r: Mono::new(r),
        }
    }
}
pub type Bgr10 = Bgr<10>;
pub type Bgr12 = Bgr<12>;
pub type Bgr14 = Bgr<14>;

/// The `Bgra` struct represents a BGRA pixel with 10-bit, 12-bit, or 14-bit depth per channel.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, LinearPixel)]
#[linear(accumulator = BgraF32)]
pub struct Bgra<const BITS: usize> {
    #[linear(nested)]
    pub b: Mono<BITS>,
    #[linear(nested)]
    pub g: Mono<BITS>,
    #[linear(nested)]
    pub r: Mono<BITS>,
    #[linear(nested)]
    pub a: Mono<BITS>,
}
impl<const BITS: usize> Bgra<BITS> {
    pub fn new(b: u16, g: u16, r: u16, a: u16) -> Self {
        Bgra {
            b: Mono::new(b),
            g: Mono::new(g),
            r: Mono::new(r),
            a: Mono::new(a),
        }
    }
}

pub type Bgra10 = Bgra<10>;
pub type Bgra12 = Bgra<12>;
pub type Bgra14 = Bgra<14>;

// ═══════════════════════════════════════════════════════════════════════════════
// Floating-point BGR / BGRA
// ═══════════════════════════════════════════════════════════════════════════════

/// The `BgrF32` struct represents a BGR pixel with 32-bit floating point depth per channel.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, PlainPixel, HomogeneousPixel, ZeroablePixel, LinearPixel,
)]
#[linear(accumulator = Self)]
pub struct BgrF32 {
    // ADR-0044 + ADR-0046: inner `f32` is a channel, not a pixel.
    #[zero(default)]
    pub b: f32,
    #[zero(default)]
    pub g: f32,
    #[zero(default)]
    pub r: f32,
}
impl BgrF32 {
    pub fn new(b: f32, g: f32, r: f32) -> Self {
        BgrF32 { b, g, r }
    }
}

impl Hash for BgrF32 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        canonicalize_f32(self.b).hash(state);
        canonicalize_f32(self.g).hash(state);
        canonicalize_f32(self.r).hash(state);
    }
}

/// The `BgraF32` struct represents a BGRA pixel with 32-bit floating point depth per channel.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, PlainPixel, HomogeneousPixel, ZeroablePixel, LinearPixel,
)]
#[linear(accumulator = Self)]
pub struct BgraF32 {
    // ADR-0044 + ADR-0046: inner `f32` is a channel, not a pixel.
    #[zero(default)]
    pub b: f32,
    #[zero(default)]
    pub g: f32,
    #[zero(default)]
    pub r: f32,
    #[zero(default)]
    pub a: f32,
}
impl BgraF32 {
    pub fn new(b: f32, g: f32, r: f32, a: f32) -> Self {
        BgraF32 { b, g, r, a }
    }
}

impl Hash for BgraF32 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        canonicalize_f32(self.b).hash(state);
        canonicalize_f32(self.g).hash(state);
        canonicalize_f32(self.r).hash(state);
        canonicalize_f32(self.a).hash(state);
    }
}

/// The `BgrF64` struct represents a BGR pixel with 64-bit floating point depth per channel.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, PlainPixel, HomogeneousPixel, ZeroablePixel, LinearPixel,
)]
#[linear(accumulator = Self)]
pub struct BgrF64 {
    // ADR-0044 + ADR-0046: inner `f64` is a channel, not a pixel.
    #[zero(default)]
    pub b: f64,
    #[zero(default)]
    pub g: f64,
    #[zero(default)]
    pub r: f64,
}
impl BgrF64 {
    pub fn new(b: f64, g: f64, r: f64) -> Self {
        BgrF64 { b, g, r }
    }
}

impl Hash for BgrF64 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        canonicalize_f64(self.b).hash(state);
        canonicalize_f64(self.g).hash(state);
        canonicalize_f64(self.r).hash(state);
    }
}

/// The `BgraF64` struct represents a BGRA pixel with 64-bit floating point depth per channel.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, PlainPixel, HomogeneousPixel, ZeroablePixel, LinearPixel,
)]
#[linear(accumulator = Self)]
pub struct BgraF64 {
    // ADR-0044 + ADR-0046: inner `f64` is a channel, not a pixel.
    #[zero(default)]
    pub b: f64,
    #[zero(default)]
    pub g: f64,
    #[zero(default)]
    pub r: f64,
    #[zero(default)]
    pub a: f64,
}
impl BgraF64 {
    pub fn new(b: f64, g: f64, r: f64, a: f64) -> Self {
        BgraF64 { b, g, r, a }
    }
}

impl Hash for BgraF64 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        canonicalize_f64(self.b).hash(state);
        canonicalize_f64(self.g).hash(state);
        canonicalize_f64(self.r).hash(state);
        canonicalize_f64(self.a).hash(state);
    }
}

// ---------------------------------------------------------------------------
// PlainPixel / ZeroablePixel / HomogeneousPixel for const-generic composite types
// (Add, LinearPixel, FromLinear, LinearSpace are generated by #[derive(LinearPixel)])
// ---------------------------------------------------------------------------

// ADR-0046: `PlainPixel` extends `PlainChannel`. Each hand-written
// `PlainPixel` impl below is paired with a `PlainChannel` witness.
//
// SAFETY: `Rgb<BITS>` / `Rgba<BITS>` / `Bgr<BITS>` / `Bgra<BITS>`
// are `#[repr(C)]` over `Mono<BITS>` fields; layout, round-trip,
// and bit-pattern validity are inherited from `Mono<BITS>`.
unsafe impl<const BITS: usize> PlainChannel for Rgb<BITS> {}
unsafe impl<const BITS: usize> PlainPixel for Rgb<BITS> {
    const CHANNELS: &'static [usize] = &[2, 2, 2];
}
impl<const BITS: usize> ZeroablePixel for Rgb<BITS> {
    fn zero() -> Self {
        Rgb {
            r: Mono::new(0),
            g: Mono::new(0),
            b: Mono::new(0),
        }
    }
}
unsafe impl<const BITS: usize> HomogeneousPixel for Rgb<BITS> {
    type Channel = Mono<BITS>;
    type Channels = [Mono<BITS>; 3];
}

unsafe impl<const BITS: usize> PlainChannel for Rgba<BITS> {}
unsafe impl<const BITS: usize> PlainPixel for Rgba<BITS> {
    const CHANNELS: &'static [usize] = &[2, 2, 2, 2];
}
impl<const BITS: usize> ZeroablePixel for Rgba<BITS> {
    fn zero() -> Self {
        Rgba {
            r: Mono::new(0),
            g: Mono::new(0),
            b: Mono::new(0),
            a: Mono::new(0),
        }
    }
}
unsafe impl<const BITS: usize> HomogeneousPixel for Rgba<BITS> {
    type Channel = Mono<BITS>;
    type Channels = [Mono<BITS>; 4];
}

unsafe impl<const BITS: usize> PlainChannel for Bgr<BITS> {}
unsafe impl<const BITS: usize> PlainPixel for Bgr<BITS> {
    const CHANNELS: &'static [usize] = &[2, 2, 2];
}
impl<const BITS: usize> ZeroablePixel for Bgr<BITS> {
    fn zero() -> Self {
        Bgr {
            b: Mono::new(0),
            g: Mono::new(0),
            r: Mono::new(0),
        }
    }
}
unsafe impl<const BITS: usize> HomogeneousPixel for Bgr<BITS> {
    type Channel = Mono<BITS>;
    type Channels = [Mono<BITS>; 3];
}

unsafe impl<const BITS: usize> PlainChannel for Bgra<BITS> {}
unsafe impl<const BITS: usize> PlainPixel for Bgra<BITS> {
    const CHANNELS: &'static [usize] = &[2, 2, 2, 2];
}
impl<const BITS: usize> ZeroablePixel for Bgra<BITS> {
    fn zero() -> Self {
        Bgra {
            b: Mono::new(0),
            g: Mono::new(0),
            r: Mono::new(0),
            a: Mono::new(0),
        }
    }
}
unsafe impl<const BITS: usize> HomogeneousPixel for Bgra<BITS> {
    type Channel = Mono<BITS>;
    type Channels = [Mono<BITS>; 4];
}

// ---------------------------------------------------------------------------
// IntegralPixel / IntegralSquaredPixel impls (ADR-0032)
// ---------------------------------------------------------------------------
//
// Per-channel accumulation — the worst case is bounded per channel, so the
// pre-flight check (preflight::IntegralCapacity) is per-channel.
// `max_integral_value()` returns the accumulator pixel with every channel
// set to the source channel's maximum.
//
// Source coverage: `Rgb8`, `Rgb16`, `RgbF32`. `Rgb32`/`Rgb64` are
// accumulators only — summing arbitrary 32/64-bit RGB inputs is rarely
// useful and the impls can be added later when a use case demands it.
// Alpha (`Rgba*`) and BGR variants are deliberately omitted (ADR-0032 §2)
// — summing alpha has no well-defined semantic.

// ── Rgb8 ───────────────────────────────────────────────────────────────────
impl IntegralPixel<Rgb32> for Rgb8 {
    #[inline]
    fn to_integral(self) -> Rgb32 {
        Rgb32::new(self.r.0 as u32, self.g.0 as u32, self.b.0 as u32)
    }
    #[inline]
    fn max_integral_value() -> Rgb32 {
        Rgb32::new(u8::MAX as u32, u8::MAX as u32, u8::MAX as u32)
    }
}
impl IntegralPixel<Rgb64> for Rgb8 {
    #[inline]
    fn to_integral(self) -> Rgb64 {
        Rgb64::new(self.r.0 as u64, self.g.0 as u64, self.b.0 as u64)
    }
    #[inline]
    fn max_integral_value() -> Rgb64 {
        Rgb64::new(u8::MAX as u64, u8::MAX as u64, u8::MAX as u64)
    }
}
impl IntegralPixel<RgbF64> for Rgb8 {
    #[inline]
    fn to_integral(self) -> RgbF64 {
        RgbF64::new(self.r.0 as f64, self.g.0 as f64, self.b.0 as f64)
    }
    #[inline]
    fn max_integral_value() -> RgbF64 {
        RgbF64::new(u8::MAX as f64, u8::MAX as f64, u8::MAX as f64)
    }
}

impl IntegralSquaredPixel<Rgb64> for Rgb8 {
    #[inline]
    fn to_integral_squared(self) -> Rgb64 {
        let r = self.r.0 as u64;
        let g = self.g.0 as u64;
        let b = self.b.0 as u64;
        Rgb64::new(r * r, g * g, b * b)
    }
    #[inline]
    fn max_integral_squared_value() -> Rgb64 {
        let m = u8::MAX as u64;
        Rgb64::new(m * m, m * m, m * m)
    }
}
impl IntegralSquaredPixel<RgbF64> for Rgb8 {
    #[inline]
    fn to_integral_squared(self) -> RgbF64 {
        let r = self.r.0 as f64;
        let g = self.g.0 as f64;
        let b = self.b.0 as f64;
        RgbF64::new(r * r, g * g, b * b)
    }
    #[inline]
    fn max_integral_squared_value() -> RgbF64 {
        let m = u8::MAX as f64;
        RgbF64::new(m * m, m * m, m * m)
    }
}

// ── Rgb16 ──────────────────────────────────────────────────────────────────
impl IntegralPixel<Rgb64> for Rgb16 {
    #[inline]
    fn to_integral(self) -> Rgb64 {
        Rgb64::new(self.r.0 as u64, self.g.0 as u64, self.b.0 as u64)
    }
    #[inline]
    fn max_integral_value() -> Rgb64 {
        Rgb64::new(u16::MAX as u64, u16::MAX as u64, u16::MAX as u64)
    }
}
impl IntegralPixel<RgbF64> for Rgb16 {
    #[inline]
    fn to_integral(self) -> RgbF64 {
        RgbF64::new(self.r.0 as f64, self.g.0 as f64, self.b.0 as f64)
    }
    #[inline]
    fn max_integral_value() -> RgbF64 {
        RgbF64::new(u16::MAX as f64, u16::MAX as f64, u16::MAX as f64)
    }
}

impl IntegralSquaredPixel<RgbF64> for Rgb16 {
    #[inline]
    fn to_integral_squared(self) -> RgbF64 {
        let r = self.r.0 as f64;
        let g = self.g.0 as f64;
        let b = self.b.0 as f64;
        RgbF64::new(r * r, g * g, b * b)
    }
    #[inline]
    fn max_integral_squared_value() -> RgbF64 {
        let m = u16::MAX as f64;
        RgbF64::new(m * m, m * m, m * m)
    }
}

// ── RgbF32 (float, conventional [0, 1] range per channel) ──────────────────
impl IntegralPixel<RgbF64> for RgbF32 {
    #[inline]
    fn to_integral(self) -> RgbF64 {
        RgbF64::new(self.r as f64, self.g as f64, self.b as f64)
    }
    #[inline]
    fn max_integral_value() -> RgbF64 {
        RgbF64::new(1.0, 1.0, 1.0)
    }
}

impl IntegralSquaredPixel<RgbF64> for RgbF32 {
    #[inline]
    fn to_integral_squared(self) -> RgbF64 {
        let r = self.r as f64;
        let g = self.g as f64;
        let b = self.b as f64;
        RgbF64::new(r * r, g * g, b * b)
    }
    #[inline]
    fn max_integral_squared_value() -> RgbF64 {
        RgbF64::new(1.0, 1.0, 1.0)
    }
}

#[cfg(test)]
mod integral_rgb_tests {
    use super::*;

    #[test]
    fn rgb8_to_rgb32_per_channel() {
        let p = Rgb8::new(10, 20, 30);
        let acc = <Rgb8 as IntegralPixel<Rgb32>>::to_integral(p);
        assert_eq!(acc, Rgb32::new(10, 20, 30));
    }

    #[test]
    fn rgb8_max_integral_value() {
        assert_eq!(
            <Rgb8 as IntegralPixel<Rgb32>>::max_integral_value(),
            Rgb32::new(255, 255, 255)
        );
        assert_eq!(
            <Rgb8 as IntegralPixel<Rgb64>>::max_integral_value(),
            Rgb64::new(255, 255, 255)
        );
    }

    #[test]
    fn rgb8_squared_rgb64() {
        let p = Rgb8::new(2, 3, 4);
        assert_eq!(
            <Rgb8 as IntegralSquaredPixel<Rgb64>>::to_integral_squared(p),
            Rgb64::new(4, 9, 16)
        );
    }

    #[test]
    fn rgb16_to_rgb64_max() {
        assert_eq!(
            <Rgb16 as IntegralPixel<Rgb64>>::max_integral_value(),
            Rgb64::new(u16::MAX as u64, u16::MAX as u64, u16::MAX as u64)
        );
    }

    #[test]
    fn rgbf32_to_rgbf64() {
        let p = RgbF32::new(0.1, 0.5, 0.9);
        let acc = <RgbF32 as IntegralPixel<RgbF64>>::to_integral(p);
        assert!((acc.r - 0.1).abs() < 1e-6);
        assert!((acc.g - 0.5).abs() < 1e-6);
        assert!((acc.b - 0.9).abs() < 1e-6);
        assert_eq!(
            <RgbF32 as IntegralPixel<RgbF64>>::max_integral_value(),
            RgbF64::new(1.0, 1.0, 1.0)
        );
    }
}

// ── Accumulator Add / Sub regression tests ──────────────────────────
//
// `Add` / `Sub` on these pixel types are emitted by the `LinearPixel`
// derive macro (`fovea-derive/src/linear_pixel.rs`); these tests
// pin the per-channel arithmetic the summed-area-table engine in
// `crate::analyze::integral` relies on (ADR-0032 §§3, 5).
#[cfg(test)]
mod accumulator_arith_tests {
    use super::*;

    #[test]
    fn rgb32_add_per_channel() {
        let a = Rgb32::new(100, 200, 300);
        let b = Rgb32::new(7, 8, 9);
        let s = a + b;
        assert_eq!(s, Rgb32::new(107, 208, 309));
    }

    #[test]
    fn rgb32_sub_per_channel() {
        let a = Rgb32::new(1_000, 2_000, 3_000);
        let b = Rgb32::new(100, 200, 300);
        let d = a - b;
        assert_eq!(d, Rgb32::new(900, 1_800, 2_700));
    }

    #[test]
    fn rgb32_add_saturates_per_channel() {
        // Saturation is per-channel: only the channels that overflow clamp.
        let a = Rgb32::new(u32::MAX - 1, 100, 50);
        let b = Rgb32::new(10, 200, 25);
        let s = a + b;
        assert_eq!(s, Rgb32::new(u32::MAX, 300, 75));
    }

    #[test]
    fn rgb32_sub_saturates_per_channel() {
        let a = Rgb32::new(5, 100, 50);
        let b = Rgb32::new(100, 30, 25);
        let d = a - b;
        assert_eq!(d, Rgb32::new(0, 70, 25));
    }

    #[test]
    fn rgb32_add_sub_roundtrip() {
        let a = Rgb32::new(12_345, 67_890, 11_111);
        let b = Rgb32::new(100, 200, 300);
        assert_eq!((a + b) - b, a);
    }

    #[test]
    fn rgb64_add_per_channel() {
        let a = Rgb64::new(10_000_000_000, 20_000_000_000, 30_000_000_000);
        let b = Rgb64::new(1, 2, 3);
        let s = a + b;
        assert_eq!(
            s,
            Rgb64::new(10_000_000_001, 20_000_000_002, 30_000_000_003)
        );
    }

    #[test]
    fn rgb64_sub_per_channel() {
        let a = Rgb64::new(100, 200, 300);
        let b = Rgb64::new(10, 20, 30);
        let d = a - b;
        assert_eq!(d, Rgb64::new(90, 180, 270));
    }

    #[test]
    fn rgb64_add_saturates_per_channel() {
        let a = Rgb64::new(u64::MAX - 1, 0, 5);
        let b = Rgb64::new(10, 1, 1);
        let s = a + b;
        assert_eq!(s, Rgb64::new(u64::MAX, 1, 6));
    }

    #[test]
    fn rgb64_sub_saturates_per_channel() {
        let a = Rgb64::new(5, 100, 50);
        let b = Rgb64::new(100, 30, 25);
        let d = a - b;
        assert_eq!(d, Rgb64::new(0, 70, 25));
    }

    #[test]
    fn rgbf64_add_per_channel() {
        let a = RgbF64::new(1.0, 2.0, 3.0);
        let b = RgbF64::new(0.25, 0.5, 0.75);
        let s = a + b;
        assert!((s.r - 1.25).abs() < 1e-12);
        assert!((s.g - 2.5).abs() < 1e-12);
        assert!((s.b - 3.75).abs() < 1e-12);
    }

    #[test]
    fn rgbf64_sub_per_channel() {
        let a = RgbF64::new(1.25, 2.5, 3.75);
        let b = RgbF64::new(0.25, 0.5, 0.75);
        let d = a - b;
        assert!((d.r - 1.0).abs() < 1e-12);
        assert!((d.g - 2.0).abs() < 1e-12);
        assert!((d.b - 3.0).abs() < 1e-12);
    }

    #[test]
    fn rgbf64_add_sub_roundtrip() {
        let a = RgbF64::new(123.456, 789.012, 345.678);
        let b = RgbF64::new(0.5, 0.25, 0.125);
        let r = (a + b) - b;
        assert!((r.r - a.r).abs() < 1e-9);
        assert!((r.g - a.g).abs() < 1e-9);
        assert!((r.b - a.b).abs() < 1e-9);
    }
}

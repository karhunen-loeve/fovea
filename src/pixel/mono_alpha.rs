//! Grayscale-with-alpha (MonoA) pixel types.
//!
//! Two-channel pixels: value (v) + alpha (a). See ADR-0009 for rationale.

use irys_cv_derive::{HomogeneousPixel, LinearPixel, PlainPixel, WhiteChannel, ZeroablePixel};

use std::{
    hash::{Hash, Hasher},
    num::Saturating,
};

use super::{canonicalize_f32, canonicalize_f64};

// ═══════════════════════════════════════════════════════════════════════════════
// MonoA (Grayscale-with-Alpha) pixel types
//
// Two-channel pixels: value (v) + alpha (a).  Same derive pattern as Rgba.
// See ADR-0009 for rationale.
// ═══════════════════════════════════════════════════════════════════════════════

/// Grayscale-with-alpha pixel, 8-bit depth per channel.
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
#[linear(accumulator = MonoAF32)]
pub struct MonoA8 {
    pub v: Saturating<u8>,
    pub a: Saturating<u8>,
}
impl MonoA8 {
    pub fn new(v: u8, a: u8) -> Self {
        MonoA8 {
            v: Saturating(v),
            a: Saturating(a),
        }
    }
}

/// Grayscale-with-alpha pixel, 16-bit depth per channel.
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
#[linear(accumulator = MonoAF32)]
pub struct MonoA16 {
    pub v: Saturating<u16>,
    pub a: Saturating<u16>,
}
impl MonoA16 {
    pub fn new(v: u16, a: u16) -> Self {
        MonoA16 {
            v: Saturating(v),
            a: Saturating(a),
        }
    }
}

/// Grayscale-with-alpha pixel, 32-bit depth per channel.
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
#[linear(accumulator = MonoAF64)]
pub struct MonoA32 {
    pub v: Saturating<u32>,
    pub a: Saturating<u32>,
}
impl MonoA32 {
    pub fn new(v: u32, a: u32) -> Self {
        MonoA32 {
            v: Saturating(v),
            a: Saturating(a),
        }
    }
}

/// Grayscale-with-alpha pixel, 64-bit depth per channel.
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
#[linear(accumulator = MonoAF64)]
pub struct MonoA64 {
    pub v: Saturating<u64>,
    pub a: Saturating<u64>,
}
impl MonoA64 {
    pub fn new(v: u64, a: u64) -> Self {
        MonoA64 {
            v: Saturating(v),
            a: Saturating(a),
        }
    }
}

/// Grayscale-with-alpha pixel, 32-bit floating point depth per channel.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, PlainPixel, HomogeneousPixel, ZeroablePixel, LinearPixel,
)]
#[linear(accumulator = Self)]
pub struct MonoAF32 {
    // ADR-0044 + ADR-0046: inner `f32` is a channel, not a pixel.
    #[zero(default)]
    pub v: f32,
    #[zero(default)]
    pub a: f32,
}
impl MonoAF32 {
    pub fn new(v: f32, a: f32) -> Self {
        MonoAF32 { v, a }
    }
}

impl Hash for MonoAF32 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        canonicalize_f32(self.v).hash(state);
        canonicalize_f32(self.a).hash(state);
    }
}

/// Grayscale-with-alpha pixel, 64-bit floating point depth per channel.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, PartialEq, PlainPixel, HomogeneousPixel, ZeroablePixel, LinearPixel,
)]
#[linear(accumulator = Self)]
pub struct MonoAF64 {
    // ADR-0044 + ADR-0046: inner `f64` is a channel, not a pixel.
    #[zero(default)]
    pub v: f64,
    #[zero(default)]
    pub a: f64,
}
impl MonoAF64 {
    pub fn new(v: f64, a: f64) -> Self {
        MonoAF64 { v, a }
    }
}

impl Hash for MonoAF64 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        canonicalize_f64(self.v).hash(state);
        canonicalize_f64(self.a).hash(state);
    }
}

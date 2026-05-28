//! Pixel trait implementations for Rust primitive types.
//!
//! After ADR-0045, channel primitives (`u8`–`u64`, `i8`–`i64`,
//! `Saturating<_>`) implement [`LinearChannel`] (the arithmetic
//! substrate the derive macro walks when composing a pixel's
//! accumulator) and no longer implement [`LinearPixel`].
//!
//! After ADR-0044 Phase E, `f32` and `f64` are *not* pixels. They
//! remain `LinearChannel` implementors (their legitimate arithmetic
//! role — see ADR-0045) but no longer implement `PlainPixel`,
//! `HomogeneousPixel`, `ZeroablePixel`, `LinearPixel`, or
//! `LinearSpace`. Spatial-sample sites that previously used
//! `Image<f32>` / `Image<f64>` now use `Image<MonoF32>` /
//! `Image<MonoF64>`. Kernel-weight sites (`Image<T: Copy>`) are
//! unaffected and continue to use bare `f32` / `f64` coefficients.

use crate::pixel::{
    BoundedChannel, FromLinear, HomogeneousPixel, LinearChannel, MonoF32, MonoF64, PlainChannel,
    PlainPixel, ZeroablePixel,
};
use std::num::Saturating;

// ADR-0046: `PlainChannel` impls for every byte-layout primitive.
// These are the channel-role byte-layout witnesses. Every primitive
// that currently implements `PlainPixel` (u/i{8,16,32,64}) gets a
// matching `PlainChannel` impl — `PlainPixel` now extends
// `PlainChannel`, so both roles must be supplied. `f32` and `f64` get
// `PlainChannel` impls but NOT `PlainPixel` (ADR-0044): they are
// first-class channels inside `MonoF32` / `RgbF32` / etc., never
// first-class pixels.
//
// Method bodies are the `PlainChannel` defaults in all cases; the
// `unsafe impl` blocks exist only to assert the layout invariants
// and to wire up the type-level membership.

// SAFETY: each primitive has `size_of::<Self>() == SIZE`, no padding,
// and any bit pattern is a valid value (including `f32` / `f64` NaN
// bit patterns — valid to construct, never UB). Round-trip through
// `&[u8]` is preserved because `as_bytes` / `from_bytes` reinterpret
// the native byte representation unchanged.
unsafe impl PlainChannel for u8 {}
unsafe impl PlainChannel for u16 {}
unsafe impl PlainChannel for u32 {}
unsafe impl PlainChannel for u64 {}
unsafe impl PlainChannel for i8 {}
unsafe impl PlainChannel for i16 {}
unsafe impl PlainChannel for i32 {}
unsafe impl PlainChannel for i64 {}
// ADR-0046 raison d'être: raw floats are channels but not pixels.
unsafe impl PlainChannel for f32 {}
unsafe impl PlainChannel for f64 {}
// SAFETY: `#[repr(transparent)]` over `T` preserves layout; the
// blanket inherits the invariants from `T: PlainChannel`.
unsafe impl<T: PlainChannel> PlainChannel for Saturating<T> {}

unsafe impl PlainPixel for u8 {
    const CHANNELS: &'static [usize] = &[1];
}
impl ZeroablePixel for u8 {
    fn zero() -> Self {
        0
    }
}
// `impl LinearPixel for u8` removed (ADR-0045 Phase S4). The
// arithmetic now lives on `LinearChannel<f32> for u8` below.
impl FromLinear<f32> for u8 {
    #[inline(always)]
    fn from_linear(acc: f32) -> Self {
        acc.round().clamp(0.0, u8::MAX as f32) as u8
    }
}
// ADR-0045 Phase A: named-float-accumulator sibling. Delegates to
// the bare-float body; `MonoF32` is `#[repr(transparent)] over f32`
// so the `.0` extraction is zero-cost.
impl FromLinear<MonoF32> for u8 {
    #[inline(always)]
    fn from_linear(acc: MonoF32) -> Self {
        <Self as FromLinear<f32>>::from_linear(acc.0)
    }
}
unsafe impl HomogeneousPixel for u8 {
    type Channel = u8;
    type Channels = [u8; 1];
}

unsafe impl PlainPixel for u16 {
    const CHANNELS: &'static [usize] = &[2];
}
impl ZeroablePixel for u16 {
    fn zero() -> Self {
        0
    }
}
// `impl LinearPixel for u16` removed (ADR-0045 Phase S4).
impl FromLinear<f32> for u16 {
    #[inline(always)]
    fn from_linear(acc: f32) -> Self {
        acc.round().clamp(0.0, u16::MAX as f32) as u16
    }
}
// ADR-0045 Phase A: named-float-accumulator sibling.
impl FromLinear<MonoF32> for u16 {
    #[inline(always)]
    fn from_linear(acc: MonoF32) -> Self {
        <Self as FromLinear<f32>>::from_linear(acc.0)
    }
}
unsafe impl HomogeneousPixel for u16 {
    type Channel = u16;
    type Channels = [u16; 1];
}
unsafe impl PlainPixel for u32 {
    const CHANNELS: &'static [usize] = &[4];
}
impl ZeroablePixel for u32 {
    fn zero() -> Self {
        0
    }
}
// `impl LinearPixel for u32` and `impl LinearPixel<f64> for u32`
// removed (ADR-0045 Phase S4). Both scalar roles live on
// `LinearChannel<f32|f64> for u32` below.
impl FromLinear<f64> for u32 {
    #[inline(always)]
    fn from_linear(acc: f64) -> Self {
        acc.round().clamp(0.0, u32::MAX as f64) as u32
    }
}
// ADR-0045 Phase A: named-float-accumulator sibling.
impl FromLinear<MonoF64> for u32 {
    #[inline(always)]
    fn from_linear(acc: MonoF64) -> Self {
        <Self as FromLinear<f64>>::from_linear(acc.0)
    }
}
unsafe impl HomogeneousPixel for u32 {
    type Channel = u32;
    type Channels = [u32; 1];
}
unsafe impl PlainPixel for u64 {
    const CHANNELS: &'static [usize] = &[8];
}
impl ZeroablePixel for u64 {
    fn zero() -> Self {
        0
    }
}
// `impl LinearPixel for u64` and `impl LinearPixel<f64> for u64`
// removed (ADR-0045 Phase S4).
impl FromLinear<f64> for u64 {
    #[inline(always)]
    fn from_linear(acc: f64) -> Self {
        acc.round().clamp(0.0, u64::MAX as f64) as u64
    }
}
// ADR-0045 Phase A: named-float-accumulator sibling.
impl FromLinear<MonoF64> for u64 {
    #[inline(always)]
    fn from_linear(acc: MonoF64) -> Self {
        <Self as FromLinear<f64>>::from_linear(acc.0)
    }
}
unsafe impl HomogeneousPixel for u64 {
    type Channel = u64;
    type Channels = [u64; 1];
}
unsafe impl PlainPixel for i8 {
    const CHANNELS: &'static [usize] = &[1];
}
impl ZeroablePixel for i8 {
    fn zero() -> Self {
        0
    }
}
// `impl LinearPixel for i8` removed (ADR-0045 Phase S4).
// `impl FromLinear<f32> for i8` removed (ADR-0045 Phase S4.2 —
// signed-integer `FromLinear` impls were speculative; no
// library-shipping pixel used them, and they were parallel to
// the ADR-0043 signed-`BoundedChannel` removal).
unsafe impl HomogeneousPixel for i8 {
    type Channel = i8;
    type Channels = [i8; 1];
}
unsafe impl PlainPixel for i16 {
    const CHANNELS: &'static [usize] = &[2];
}
impl ZeroablePixel for i16 {
    fn zero() -> Self {
        0
    }
}
// `impl LinearPixel for i16` and `impl FromLinear<f32> for i16`
// removed (ADR-0045 Phase S4 / S4.2).
unsafe impl HomogeneousPixel for i16 {
    type Channel = i16;
    type Channels = [i16; 1];
}
unsafe impl PlainPixel for i32 {
    const CHANNELS: &'static [usize] = &[4];
}
impl ZeroablePixel for i32 {
    fn zero() -> Self {
        0
    }
}
// `impl LinearPixel for i32`, `impl LinearPixel<f64> for i32` and
// `impl FromLinear<f64> for i32` removed (ADR-0045 Phase S4 / S4.2).
unsafe impl HomogeneousPixel for i32 {
    type Channel = i32;
    type Channels = [i32; 1];
}
unsafe impl PlainPixel for i64 {
    const CHANNELS: &'static [usize] = &[8];
}
impl ZeroablePixel for i64 {
    fn zero() -> Self {
        0
    }
}
// `impl LinearPixel for i64`, `impl LinearPixel<f64> for i64` and
// `impl FromLinear<f64> for i64` removed (ADR-0045 Phase S4 / S4.2).
unsafe impl HomogeneousPixel for i64 {
    type Channel = i64;
    type Channels = [i64; 1];
}

// ADR-0044 Phase E: `f32` / `f64` are no longer pixels. They remain
// `LinearChannel` implementors below (the arithmetic role) but the
// `PlainPixel` / `ZeroablePixel` / `HomogeneousPixel` / `LinearPixel`
// impls that used to live here have been removed. Spatial-sample
// code should use `MonoF32` / `MonoF64` instead.

unsafe impl<T> PlainPixel for Saturating<T>
where
    T: PlainPixel,
{
    const CHANNELS: &'static [usize] = T::CHANNELS;
}
impl<T: ZeroablePixel> ZeroablePixel for Saturating<T> {
    fn zero() -> Self {
        Saturating(T::zero())
    }
}
// `impl LinearPixel for Saturating<u8>` removed (ADR-0045 Phase S4).
impl FromLinear<f32> for Saturating<u8> {
    #[inline(always)]
    fn from_linear(acc: f32) -> Self {
        Saturating(acc.round().clamp(0.0, u8::MAX as f32) as u8)
    }
}
// ADR-0045 Phase A: named-float-accumulator sibling.
impl FromLinear<MonoF32> for Saturating<u8> {
    #[inline(always)]
    fn from_linear(acc: MonoF32) -> Self {
        <Self as FromLinear<f32>>::from_linear(acc.0)
    }
}
// `impl LinearPixel for Saturating<u16>` removed (ADR-0045 Phase S4).
impl FromLinear<f32> for Saturating<u16> {
    #[inline(always)]
    fn from_linear(acc: f32) -> Self {
        Saturating(acc.round().clamp(0.0, u16::MAX as f32) as u16)
    }
}
// ADR-0045 Phase A: named-float-accumulator sibling.
impl FromLinear<MonoF32> for Saturating<u16> {
    #[inline(always)]
    fn from_linear(acc: MonoF32) -> Self {
        <Self as FromLinear<f32>>::from_linear(acc.0)
    }
}
// `impl LinearPixel for Saturating<u32>` and
// `impl LinearPixel<f64> for Saturating<u32>` removed
// (ADR-0045 Phase S4).
impl FromLinear<f64> for Saturating<u32> {
    #[inline(always)]
    fn from_linear(acc: f64) -> Self {
        Saturating(acc.round().clamp(0.0, u32::MAX as f64) as u32)
    }
}
// ADR-0045 Phase A: named-float-accumulator sibling.
impl FromLinear<MonoF64> for Saturating<u32> {
    #[inline(always)]
    fn from_linear(acc: MonoF64) -> Self {
        <Self as FromLinear<f64>>::from_linear(acc.0)
    }
}
// `impl LinearPixel for Saturating<u64>` and
// `impl LinearPixel<f64> for Saturating<u64>` removed
// (ADR-0045 Phase S4).
impl FromLinear<f64> for Saturating<u64> {
    #[inline(always)]
    fn from_linear(acc: f64) -> Self {
        Saturating(acc.round().clamp(0.0, u64::MAX as f64) as u64)
    }
}
// ADR-0045 Phase A: named-float-accumulator sibling.
impl FromLinear<MonoF64> for Saturating<u64> {
    #[inline(always)]
    fn from_linear(acc: MonoF64) -> Self {
        <Self as FromLinear<f64>>::from_linear(acc.0)
    }
}

// `LinearSpace` is a pixel-role marker (`LinearSpace: LinearPixel`).
// After ADR-0045 Phase S4, channel primitives no longer implement
// `LinearPixel`, so the `LinearSpace` impls on them are both
// unreachable (the super-trait bound fails) and semantically wrong
// (a channel is not a pixel). After ADR-0044 Phase E, the `f32` /
// `f64` impls have been removed as well — `LinearSpace` now lives
// exclusively on actual pixel types (e.g. `MonoF32`, `RgbF32`).

// ─── LinearChannel (ADR-0045) ────────────────────────────────────────────
//
// Mirror of the LinearPixel impls above. After ADR-0045 every channel
// primitive implements `LinearChannel`; the derive macro probes this
// trait first when composing a pixel's accumulator from its channel
// fields. The bodies are byte-identical copies of the LinearPixel
// versions; the distinction is purely taxonomic (see PHILOSOPHY §2 and
// ADR-0045 §2).
//
// Phase S2 keeps the parallel `LinearPixel` impls in place so the
// derive macro continues to resolve. Phase S4 deletes them once the
// derive macro has switched to the `LinearChannel` path.

impl LinearChannel<f32> for u8 {
    type Accumulator = f32;
    #[inline(always)]
    fn to_accumulator(&self) -> f32 {
        *self as f32
    }
    #[inline(always)]
    fn scale(&self, scalar: f32) -> f32 {
        *self as f32 * scalar
    }
    #[inline(always)]
    fn scale_add(&self, scalar: f32, addend: f32) -> f32 {
        #[cfg(target_feature = "fma")]
        {
            (*self as f32).mul_add(scalar, addend)
        }
        #[cfg(not(target_feature = "fma"))]
        {
            *self as f32 * scalar + addend
        }
    }
    #[inline(always)]
    fn uniform(scalar: f32) -> f32 {
        scalar
    }
}

impl LinearChannel<f32> for u16 {
    type Accumulator = f32;
    #[inline(always)]
    fn to_accumulator(&self) -> f32 {
        *self as f32
    }
    #[inline(always)]
    fn scale(&self, scalar: f32) -> f32 {
        *self as f32 * scalar
    }
    #[inline(always)]
    fn scale_add(&self, scalar: f32, addend: f32) -> f32 {
        #[cfg(target_feature = "fma")]
        {
            (*self as f32).mul_add(scalar, addend)
        }
        #[cfg(not(target_feature = "fma"))]
        {
            *self as f32 * scalar + addend
        }
    }
    #[inline(always)]
    fn uniform(scalar: f32) -> f32 {
        scalar
    }
}

impl LinearChannel<f32> for u32 {
    type Accumulator = f64;
    #[inline(always)]
    fn to_accumulator(&self) -> f64 {
        *self as f64
    }
    #[inline(always)]
    fn scale(&self, scalar: f32) -> f64 {
        *self as f64 * scalar as f64
    }
    #[inline(always)]
    fn scale_add(&self, scalar: f32, addend: f64) -> f64 {
        #[cfg(target_feature = "fma")]
        {
            (*self as f64).mul_add(scalar as f64, addend)
        }
        #[cfg(not(target_feature = "fma"))]
        {
            *self as f64 * scalar as f64 + addend
        }
    }
    #[inline(always)]
    fn uniform(scalar: f32) -> f64 {
        scalar as f64
    }
}

impl LinearChannel<f64> for u32 {
    type Accumulator = f64;
    #[inline(always)]
    fn to_accumulator(&self) -> f64 {
        *self as f64
    }
    #[inline(always)]
    fn scale(&self, scalar: f64) -> f64 {
        *self as f64 * scalar
    }
    #[inline(always)]
    fn scale_add(&self, scalar: f64, addend: f64) -> f64 {
        #[cfg(target_feature = "fma")]
        {
            (*self as f64).mul_add(scalar, addend)
        }
        #[cfg(not(target_feature = "fma"))]
        {
            *self as f64 * scalar + addend
        }
    }
    #[inline(always)]
    fn uniform(scalar: f64) -> f64 {
        scalar
    }
}

impl LinearChannel<f32> for u64 {
    type Accumulator = f64;
    #[inline(always)]
    fn to_accumulator(&self) -> f64 {
        *self as f64
    }
    #[inline(always)]
    fn scale(&self, scalar: f32) -> f64 {
        *self as f64 * scalar as f64
    }
    #[inline(always)]
    fn scale_add(&self, scalar: f32, addend: f64) -> f64 {
        #[cfg(target_feature = "fma")]
        {
            (*self as f64).mul_add(scalar as f64, addend)
        }
        #[cfg(not(target_feature = "fma"))]
        {
            *self as f64 * scalar as f64 + addend
        }
    }
    #[inline(always)]
    fn uniform(scalar: f32) -> f64 {
        scalar as f64
    }
}

impl LinearChannel<f64> for u64 {
    type Accumulator = f64;
    #[inline(always)]
    fn to_accumulator(&self) -> f64 {
        *self as f64
    }
    #[inline(always)]
    fn scale(&self, scalar: f64) -> f64 {
        *self as f64 * scalar
    }
    #[inline(always)]
    fn scale_add(&self, scalar: f64, addend: f64) -> f64 {
        #[cfg(target_feature = "fma")]
        {
            (*self as f64).mul_add(scalar, addend)
        }
        #[cfg(not(target_feature = "fma"))]
        {
            *self as f64 * scalar + addend
        }
    }
    #[inline(always)]
    fn uniform(scalar: f64) -> f64 {
        scalar
    }
}

impl LinearChannel<f32> for i8 {
    type Accumulator = f32;
    #[inline(always)]
    fn to_accumulator(&self) -> f32 {
        *self as f32
    }
    #[inline(always)]
    fn scale(&self, scalar: f32) -> f32 {
        *self as f32 * scalar
    }
    #[inline(always)]
    fn scale_add(&self, scalar: f32, addend: f32) -> f32 {
        #[cfg(target_feature = "fma")]
        {
            (*self as f32).mul_add(scalar, addend)
        }
        #[cfg(not(target_feature = "fma"))]
        {
            *self as f32 * scalar + addend
        }
    }
    #[inline(always)]
    fn uniform(scalar: f32) -> f32 {
        scalar
    }
}

impl LinearChannel<f32> for i16 {
    type Accumulator = f32;
    #[inline(always)]
    fn to_accumulator(&self) -> f32 {
        *self as f32
    }
    #[inline(always)]
    fn scale(&self, scalar: f32) -> f32 {
        *self as f32 * scalar
    }
    #[inline(always)]
    fn scale_add(&self, scalar: f32, addend: f32) -> f32 {
        #[cfg(target_feature = "fma")]
        {
            (*self as f32).mul_add(scalar, addend)
        }
        #[cfg(not(target_feature = "fma"))]
        {
            *self as f32 * scalar + addend
        }
    }
    #[inline(always)]
    fn uniform(scalar: f32) -> f32 {
        scalar
    }
}

impl LinearChannel<f32> for i32 {
    type Accumulator = f64;
    #[inline(always)]
    fn to_accumulator(&self) -> f64 {
        *self as f64
    }
    #[inline(always)]
    fn scale(&self, scalar: f32) -> f64 {
        *self as f64 * scalar as f64
    }
    #[inline(always)]
    fn scale_add(&self, scalar: f32, addend: f64) -> f64 {
        #[cfg(target_feature = "fma")]
        {
            (*self as f64).mul_add(scalar as f64, addend)
        }
        #[cfg(not(target_feature = "fma"))]
        {
            *self as f64 * scalar as f64 + addend
        }
    }
    #[inline(always)]
    fn uniform(scalar: f32) -> f64 {
        scalar as f64
    }
}

impl LinearChannel<f64> for i32 {
    type Accumulator = f64;
    #[inline(always)]
    fn to_accumulator(&self) -> f64 {
        *self as f64
    }
    #[inline(always)]
    fn scale(&self, scalar: f64) -> f64 {
        *self as f64 * scalar
    }
    #[inline(always)]
    fn scale_add(&self, scalar: f64, addend: f64) -> f64 {
        #[cfg(target_feature = "fma")]
        {
            (*self as f64).mul_add(scalar, addend)
        }
        #[cfg(not(target_feature = "fma"))]
        {
            *self as f64 * scalar + addend
        }
    }
    #[inline(always)]
    fn uniform(scalar: f64) -> f64 {
        scalar
    }
}

impl LinearChannel<f32> for i64 {
    type Accumulator = f64;
    #[inline(always)]
    fn to_accumulator(&self) -> f64 {
        *self as f64
    }
    #[inline(always)]
    fn scale(&self, scalar: f32) -> f64 {
        *self as f64 * scalar as f64
    }
    #[inline(always)]
    fn scale_add(&self, scalar: f32, addend: f64) -> f64 {
        #[cfg(target_feature = "fma")]
        {
            (*self as f64).mul_add(scalar as f64, addend)
        }
        #[cfg(not(target_feature = "fma"))]
        {
            *self as f64 * scalar as f64 + addend
        }
    }
    #[inline(always)]
    fn uniform(scalar: f32) -> f64 {
        scalar as f64
    }
}

impl LinearChannel<f64> for i64 {
    type Accumulator = f64;
    #[inline(always)]
    fn to_accumulator(&self) -> f64 {
        *self as f64
    }
    #[inline(always)]
    fn scale(&self, scalar: f64) -> f64 {
        *self as f64 * scalar
    }
    #[inline(always)]
    fn scale_add(&self, scalar: f64, addend: f64) -> f64 {
        #[cfg(target_feature = "fma")]
        {
            (*self as f64).mul_add(scalar, addend)
        }
        #[cfg(not(target_feature = "fma"))]
        {
            *self as f64 * scalar + addend
        }
    }
    #[inline(always)]
    fn uniform(scalar: f64) -> f64 {
        scalar
    }
}

impl LinearChannel<f32> for f32 {
    type Accumulator = f32;
    #[inline(always)]
    fn to_accumulator(&self) -> f32 {
        *self
    }
    #[inline(always)]
    fn scale(&self, scalar: f32) -> f32 {
        *self * scalar
    }
    #[inline(always)]
    fn scale_add(&self, scalar: f32, addend: f32) -> f32 {
        #[cfg(target_feature = "fma")]
        {
            self.mul_add(scalar, addend)
        }
        #[cfg(not(target_feature = "fma"))]
        {
            *self * scalar + addend
        }
    }
    #[inline(always)]
    fn uniform(scalar: f32) -> f32 {
        scalar
    }
}

impl LinearChannel<f32> for f64 {
    type Accumulator = f64;
    #[inline(always)]
    fn to_accumulator(&self) -> f64 {
        *self
    }
    #[inline(always)]
    fn scale(&self, scalar: f32) -> f64 {
        *self * scalar as f64
    }
    #[inline(always)]
    fn scale_add(&self, scalar: f32, addend: f64) -> f64 {
        #[cfg(target_feature = "fma")]
        {
            self.mul_add(scalar as f64, addend)
        }
        #[cfg(not(target_feature = "fma"))]
        {
            *self * scalar as f64 + addend
        }
    }
    #[inline(always)]
    fn uniform(scalar: f32) -> f64 {
        scalar as f64
    }
}

impl LinearChannel<f64> for f64 {
    type Accumulator = f64;
    #[inline(always)]
    fn to_accumulator(&self) -> f64 {
        *self
    }
    #[inline(always)]
    fn scale(&self, scalar: f64) -> f64 {
        *self * scalar
    }
    #[inline(always)]
    fn scale_add(&self, scalar: f64, addend: f64) -> f64 {
        #[cfg(target_feature = "fma")]
        {
            self.mul_add(scalar, addend)
        }
        #[cfg(not(target_feature = "fma"))]
        {
            *self * scalar + addend
        }
    }
    #[inline(always)]
    fn uniform(scalar: f64) -> f64 {
        scalar
    }
}

impl LinearChannel<f32> for Saturating<u8> {
    type Accumulator = f32;
    #[inline(always)]
    fn to_accumulator(&self) -> f32 {
        self.0 as f32
    }
    #[inline(always)]
    fn scale(&self, scalar: f32) -> f32 {
        self.0 as f32 * scalar
    }
    #[inline(always)]
    fn scale_add(&self, scalar: f32, addend: f32) -> f32 {
        #[cfg(target_feature = "fma")]
        {
            (self.0 as f32).mul_add(scalar, addend)
        }
        #[cfg(not(target_feature = "fma"))]
        {
            self.0 as f32 * scalar + addend
        }
    }
    #[inline(always)]
    fn uniform(scalar: f32) -> f32 {
        scalar
    }
}

impl LinearChannel<f32> for Saturating<u16> {
    type Accumulator = f32;
    #[inline(always)]
    fn to_accumulator(&self) -> f32 {
        self.0 as f32
    }
    #[inline(always)]
    fn scale(&self, scalar: f32) -> f32 {
        self.0 as f32 * scalar
    }
    #[inline(always)]
    fn scale_add(&self, scalar: f32, addend: f32) -> f32 {
        #[cfg(target_feature = "fma")]
        {
            (self.0 as f32).mul_add(scalar, addend)
        }
        #[cfg(not(target_feature = "fma"))]
        {
            self.0 as f32 * scalar + addend
        }
    }
    #[inline(always)]
    fn uniform(scalar: f32) -> f32 {
        scalar
    }
}

impl LinearChannel<f32> for Saturating<u32> {
    type Accumulator = f64;
    #[inline(always)]
    fn to_accumulator(&self) -> f64 {
        self.0 as f64
    }
    #[inline(always)]
    fn scale(&self, scalar: f32) -> f64 {
        self.0 as f64 * scalar as f64
    }
    #[inline(always)]
    fn scale_add(&self, scalar: f32, addend: f64) -> f64 {
        #[cfg(target_feature = "fma")]
        {
            (self.0 as f64).mul_add(scalar as f64, addend)
        }
        #[cfg(not(target_feature = "fma"))]
        {
            self.0 as f64 * scalar as f64 + addend
        }
    }
    #[inline(always)]
    fn uniform(scalar: f32) -> f64 {
        scalar as f64
    }
}

impl LinearChannel<f64> for Saturating<u32> {
    type Accumulator = f64;
    #[inline(always)]
    fn to_accumulator(&self) -> f64 {
        self.0 as f64
    }
    #[inline(always)]
    fn scale(&self, scalar: f64) -> f64 {
        self.0 as f64 * scalar
    }
    #[inline(always)]
    fn scale_add(&self, scalar: f64, addend: f64) -> f64 {
        #[cfg(target_feature = "fma")]
        {
            (self.0 as f64).mul_add(scalar, addend)
        }
        #[cfg(not(target_feature = "fma"))]
        {
            self.0 as f64 * scalar + addend
        }
    }
    #[inline(always)]
    fn uniform(scalar: f64) -> f64 {
        scalar
    }
}

impl LinearChannel<f32> for Saturating<u64> {
    type Accumulator = f64;
    #[inline(always)]
    fn to_accumulator(&self) -> f64 {
        self.0 as f64
    }
    #[inline(always)]
    fn scale(&self, scalar: f32) -> f64 {
        self.0 as f64 * scalar as f64
    }
    #[inline(always)]
    fn scale_add(&self, scalar: f32, addend: f64) -> f64 {
        #[cfg(target_feature = "fma")]
        {
            (self.0 as f64).mul_add(scalar as f64, addend)
        }
        #[cfg(not(target_feature = "fma"))]
        {
            self.0 as f64 * scalar as f64 + addend
        }
    }
    #[inline(always)]
    fn uniform(scalar: f32) -> f64 {
        scalar as f64
    }
}

impl LinearChannel<f64> for Saturating<u64> {
    type Accumulator = f64;
    #[inline(always)]
    fn to_accumulator(&self) -> f64 {
        self.0 as f64
    }
    #[inline(always)]
    fn scale(&self, scalar: f64) -> f64 {
        self.0 as f64 * scalar
    }
    #[inline(always)]
    fn scale_add(&self, scalar: f64, addend: f64) -> f64 {
        #[cfg(target_feature = "fma")]
        {
            (self.0 as f64).mul_add(scalar, addend)
        }
        #[cfg(not(target_feature = "fma"))]
        {
            self.0 as f64 * scalar + addend
        }
    }
    #[inline(always)]
    fn uniform(scalar: f64) -> f64 {
        scalar
    }
}

// ─── BoundedChannel (ADR-0042) ────────────────────────────────────────────
//
// Intrinsic maximum value for every integer channel type the library ships.
//
// Deliberately NOT implemented for `f32` / `f64` — floating-point pixels do
// not have an intrinsic maximum in this library (Philosophy §8, ADR-0042).
// The absence is load-bearing: it is what makes `Invert` refuse to compile
// for float-channel pixels.

impl BoundedChannel for u8 {
    const MAX: Self = u8::MAX;
}
impl BoundedChannel for u16 {
    const MAX: Self = u16::MAX;
}
impl BoundedChannel for u32 {
    const MAX: Self = u32::MAX;
}
impl BoundedChannel for u64 {
    const MAX: Self = u64::MAX;
}
// Signed-integer `BoundedChannel` impls were removed by ADR-0043:
//
// 1. "Brightest" is not a signed-integer concept; there is no
//    well-defined pixel-world meaning for "`i8::MAX = 127` is white" on
//    a signed-channel pixel.
// 2. `Invert` on signed ints is a footgun: `i8::MAX - (-100i8)` overflows
//    (debug panic, release wrap), and the `Sub<Output = Self>` bound
//    cannot reject this.
// 3. No currently-shipping pixel has a signed channel; the impls were
//    speculative and untested.
//
// If signed channels ever become needed they get their own trait
// (`SignedChannelRange { MIN; MAX; }`) with strategies that respect the
// signed semantics. That is future work, out of scope here.

/// `Saturating<T>` inherits the intrinsic maximum of its inner type.
impl<T: BoundedChannel> BoundedChannel for Saturating<T> {
    const MAX: Self = Saturating(T::MAX);
}

// ─── bool as a pixel (PLAN §1.1) ──────────────────────────────────────────
//
// `bool` already rides the `T: Copy` pathway through `Image<T>`,
// `ImageView`, `SubView`, tiles, sliding windows, zip, and the parallel
// iteration machinery. It is also the pixel type that `map_neighborhood*`
// accepts as its topology mask (`MI: ImageView<Pixel = bool>`), so
// morphology and neighborhood operations natively consume binary images.
//
// To make `bool` flow through the `convert_image` driver (which allocates
// its output via `Image::zero` and therefore requires `ZeroablePixel`),
// we implement `ZeroablePixel` here. The canonical zero for a binary
// pixel is `false` ("off" / "background"), matching the convention used
// everywhere else in this codebase for bitmask / mask imagery.
//
// We deliberately do NOT implement `PlainPixel`, `HomogeneousPixel`,
// `LinearPixel`, or `BoundedChannel` for `bool`:
//
// - `PlainPixel` would require a stable, well-defined byte layout. Rust
//   does not guarantee the representation of `bool` beyond "1 byte with
//   valid bit patterns `0x00` and `0x01`", which is not the same
//   invariant as `PlainPixel` carries (no invalid bit patterns, arbitrary
//   reinterpretation of bytes). See PLAN §10 — a nominal `Binary` pixel
//   type and `PlainPixel for bool` are explicit non-goals pending
//   shape-analysis feature pressure.
// - `LinearPixel` / `BoundedChannel` have no meaningful semantics on a
//   two-valued type.
// - `HomogeneousPixel` is withheld for the same layout reasons as
//   `PlainPixel` (it is a sub-trait).

impl ZeroablePixel for bool {
    fn zero() -> Self {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── BoundedChannel (ADR-0042) ────────────────────────────────────────
    //
    // Verify that every integer channel type the library ships exposes its
    // intrinsic maximum via `BoundedChannel::MAX`, and that `Saturating<T>`
    // correctly lifts that maximum through the blanket impl.

    #[test]
    fn bounded_channel_unsigned_primitives() {
        assert_eq!(<u8 as BoundedChannel>::MAX, u8::MAX);
        assert_eq!(<u16 as BoundedChannel>::MAX, u16::MAX);
        assert_eq!(<u32 as BoundedChannel>::MAX, u32::MAX);
        assert_eq!(<u64 as BoundedChannel>::MAX, u64::MAX);
    }

    // Signed-integer `BoundedChannel` impls were removed by ADR-0043
    // (see comment above the `Saturating` impl). No signed-primitives
    // test remains; the `bounded_channel_inventory` test below asserts
    // the positive inventory without the signed types.

    #[test]
    fn bounded_channel_saturating_wraps() {
        assert_eq!(<Saturating<u8> as BoundedChannel>::MAX, Saturating(u8::MAX));
        assert_eq!(
            <Saturating<u16> as BoundedChannel>::MAX,
            Saturating(u16::MAX)
        );
        assert_eq!(
            <Saturating<u32> as BoundedChannel>::MAX,
            Saturating(u32::MAX)
        );
        assert_eq!(
            <Saturating<u64> as BoundedChannel>::MAX,
            Saturating(u64::MAX)
        );
    }

    #[test]
    fn bounded_channel_u8_is_255() {
        // Spot-check the canonical pixel-world max.
        assert_eq!(<u8 as BoundedChannel>::MAX, 255);
        assert_eq!(<Saturating<u8> as BoundedChannel>::MAX, Saturating(255u8));
    }

    // Compile-time assertion: `BoundedChannel` must NOT be implemented for
    // `f32` / `f64`. This absence is load-bearing per ADR-0042 — it is what
    // makes `Invert` and `BinaryThreshold` refuse to compile for float
    // channels (Philosophy §1 "Types are the spec", §8 "Surface information,
    // don't decide").
    //
    // We cannot test "does not implement" directly inside a `#[cfg(test)]`
    // block without a dedicated compile-fail harness (which the core crate
    // does not ship). Instead, we assert the contrapositive at compile time
    // using a helper that is generic over `BoundedChannel` — any code path
    // below would stop compiling if somebody accidentally added
    // `impl BoundedChannel for f32 { ... }`.
    //
    // Enumerating the implementing types here also serves as a living
    // inventory: every type that ought to carry the trait appears in this
    // list, and every type that ought NOT to carry it is conspicuously
    // absent.
    #[test]
    fn bounded_channel_inventory() {
        fn assert_bounded<T: BoundedChannel>() {}
        assert_bounded::<u8>();
        assert_bounded::<u16>();
        assert_bounded::<u32>();
        assert_bounded::<u64>();
        assert_bounded::<Saturating<u8>>();
        assert_bounded::<Saturating<u16>>();
        assert_bounded::<Saturating<u32>>();
        assert_bounded::<Saturating<u64>>();
        // f32 / f64: intentionally omitted (see ADR-0042).
        // i8 / i16 / i32 / i64: intentionally omitted (see ADR-0043).
    }

    // ─── LinearPixel::uniform (PLAN §3.4) ─────────────────────────────────
    //
    // For primitive `LinearPixel<f32>` impls `uniform(scalar)` is the
    // identity on the scalar (for f32-accumulator types) or a widening
    // cast to f64 (for f64-accumulator types). Verify both shapes.

    #[test]
    fn linear_channel_uniform_f32_accumulator_primitives() {
        // Post-ADR-0045: channel primitives bind on `LinearChannel`.
        assert_eq!(<u8 as LinearChannel>::uniform(0.5), 0.5f32);
        assert_eq!(<u16 as LinearChannel>::uniform(0.5), 0.5f32);
        assert_eq!(<i8 as LinearChannel>::uniform(0.25), 0.25f32);
        assert_eq!(<i16 as LinearChannel>::uniform(0.25), 0.25f32);
        assert_eq!(<f32 as LinearChannel>::uniform(42.0), 42.0f32);
        assert_eq!(<Saturating<u8> as LinearChannel>::uniform(0.5), 0.5f32);
        assert_eq!(<Saturating<u16> as LinearChannel>::uniform(0.5), 0.5f32);
    }

    #[test]
    fn linear_channel_uniform_f64_accumulator_primitives() {
        assert_eq!(<u32 as LinearChannel>::uniform(0.5), 0.5f64);
        assert_eq!(<u64 as LinearChannel>::uniform(0.5), 0.5f64);
        assert_eq!(<i32 as LinearChannel>::uniform(0.25), 0.25f64);
        assert_eq!(<i64 as LinearChannel>::uniform(0.25), 0.25f64);
        assert_eq!(<f64 as LinearChannel>::uniform(42.0), 42.0f64);
        assert_eq!(<Saturating<u32> as LinearChannel>::uniform(0.5), 0.5f64);
        assert_eq!(<Saturating<u64> as LinearChannel>::uniform(0.5), 0.5f64);
    }

    #[test]
    fn linear_channel_uniform_zero_and_negative() {
        // Zero and negative scalars pass through unchanged.
        assert_eq!(<f32 as LinearChannel>::uniform(0.0), 0.0f32);
        assert_eq!(<f32 as LinearChannel>::uniform(-1.5), -1.5f32);
        assert_eq!(<f64 as LinearChannel>::uniform(-1.5), -1.5f64);
        assert_eq!(<u8 as LinearChannel>::uniform(0.0), 0.0f32);
    }

    // ─── LinearChannel<f64> — f64-scalar paths (PLAN §3.4) ────────────────
    //
    // Every channel primitive whose `Accumulator = f64` ships a second
    // `LinearChannel<f64>` impl (with the same `Accumulator = f64`) so
    // that strategies parameterized by scalar type — e.g.
    // `BrightnessContrast<f64>` — can pick up a native-f64 path via trait
    // resolution, without an `f32 -> f64` widening in the hot loop. These
    // tests exercise `to_accumulator`, `scale`, `scale_add`, and `uniform`
    // on each such type via the f64 scalar parameter.

    #[test]
    fn linear_channel_f64_scalar_u32_u64() {
        let a: u32 = 1_000_000;
        assert_eq!(
            <u32 as LinearChannel<f64>>::to_accumulator(&a),
            1_000_000.0f64
        );
        let s = <u32 as LinearChannel<f64>>::scale(&a, 0.5);
        assert_eq!(s, 500_000.0f64);
        let sa = <u32 as LinearChannel<f64>>::scale_add(&a, 0.5, 10.0);
        assert_eq!(sa, 500_010.0f64);
        assert_eq!(<u32 as LinearChannel<f64>>::uniform(42.0), 42.0f64);

        let b: u64 = 2_000_000;
        assert_eq!(
            <u64 as LinearChannel<f64>>::to_accumulator(&b),
            2_000_000.0f64
        );
        assert_eq!(<u64 as LinearChannel<f64>>::scale(&b, 0.25), 500_000.0f64);
        assert_eq!(
            <u64 as LinearChannel<f64>>::scale_add(&b, 0.25, -5.0),
            499_995.0f64
        );
        assert_eq!(<u64 as LinearChannel<f64>>::uniform(-7.5), -7.5f64);
    }

    #[test]
    fn linear_channel_f64_scalar_i32_i64() {
        let a: i32 = -1000;
        assert_eq!(<i32 as LinearChannel<f64>>::to_accumulator(&a), -1000.0f64);
        assert_eq!(<i32 as LinearChannel<f64>>::scale(&a, 2.0), -2000.0f64);
        assert_eq!(
            <i32 as LinearChannel<f64>>::scale_add(&a, 2.0, 100.0),
            -1900.0f64
        );
        assert_eq!(<i32 as LinearChannel<f64>>::uniform(0.125), 0.125f64);

        let b: i64 = -2000;
        assert_eq!(<i64 as LinearChannel<f64>>::to_accumulator(&b), -2000.0f64);
        assert_eq!(<i64 as LinearChannel<f64>>::scale(&b, 0.5), -1000.0f64);
        assert_eq!(
            <i64 as LinearChannel<f64>>::scale_add(&b, 0.5, 1.0),
            -999.0f64
        );
        assert_eq!(<i64 as LinearChannel<f64>>::uniform(9.0), 9.0f64);
    }

    #[test]
    fn linear_channel_f64_scalar_f64() {
        let a: f64 = 3.5;
        assert_eq!(<f64 as LinearChannel<f64>>::to_accumulator(&a), 3.5);
        assert_eq!(<f64 as LinearChannel<f64>>::scale(&a, 2.0), 7.0);
        assert_eq!(<f64 as LinearChannel<f64>>::scale_add(&a, 2.0, 1.0), 8.0);
        assert_eq!(<f64 as LinearChannel<f64>>::uniform(0.5), 0.5f64);
    }

    #[test]
    fn linear_channel_f64_scalar_saturating_u32_u64() {
        let a = Saturating(50_000u32);
        assert_eq!(
            <Saturating<u32> as LinearChannel<f64>>::to_accumulator(&a),
            50_000.0f64
        );
        assert_eq!(
            <Saturating<u32> as LinearChannel<f64>>::scale(&a, 0.5),
            25_000.0f64
        );
        assert_eq!(
            <Saturating<u32> as LinearChannel<f64>>::scale_add(&a, 0.5, 7.0),
            25_007.0f64
        );
        assert_eq!(
            <Saturating<u32> as LinearChannel<f64>>::uniform(1.25),
            1.25f64
        );

        let b = Saturating(2_000_000u64);
        assert_eq!(
            <Saturating<u64> as LinearChannel<f64>>::to_accumulator(&b),
            2_000_000.0f64
        );
        assert_eq!(
            <Saturating<u64> as LinearChannel<f64>>::scale(&b, 0.1),
            200_000.0f64
        );
        assert_eq!(
            <Saturating<u64> as LinearChannel<f64>>::uniform(-2.5),
            -2.5f64
        );
    }

    #[test]
    fn linear_channel_f64_scalar_matches_f32_scalar_on_same_input() {
        // For values representable exactly in both f32 and f64, the two
        // scalar paths must agree up to the widening cast.
        let a: u32 = 1000;
        let via_f32 = <u32 as LinearChannel<f32>>::scale(&a, 0.5);
        let via_f64 = <u32 as LinearChannel<f64>>::scale(&a, 0.5);
        assert_eq!(via_f32, 500.0f64);
        assert_eq!(via_f64, 500.0f64);
        assert_eq!(via_f32, via_f64);
    }

    // ─── LinearChannel inventory (ADR-0045 guardrail) ─────────────────────
    //
    // Positive list of every channel primitive that implements
    // `LinearChannel`. Serves as a living inventory (every type that ought
    // to carry the trait appears here) and catches regressions at compile
    // time — if somebody accidentally drops a `LinearChannel` impl this
    // test stops compiling on that exact line.
    #[test]
    fn linear_channel_inventory() {
        fn assert_channel_f32<T: LinearChannel<f32>>() {}
        fn assert_channel_f64<T: LinearChannel<f64>>() {}

        // f32-scalar impls (always present).
        assert_channel_f32::<u8>();
        assert_channel_f32::<u16>();
        assert_channel_f32::<u32>();
        assert_channel_f32::<u64>();
        assert_channel_f32::<i8>();
        assert_channel_f32::<i16>();
        assert_channel_f32::<i32>();
        assert_channel_f32::<i64>();
        assert_channel_f32::<f32>();
        assert_channel_f32::<f64>();
        assert_channel_f32::<Saturating<u8>>();
        assert_channel_f32::<Saturating<u16>>();
        assert_channel_f32::<Saturating<u32>>();
        assert_channel_f32::<Saturating<u64>>();

        // f64-scalar impls (only for f64-accumulator primitives).
        assert_channel_f64::<u32>();
        assert_channel_f64::<u64>();
        assert_channel_f64::<i32>();
        assert_channel_f64::<i64>();
        assert_channel_f64::<f64>();
        assert_channel_f64::<Saturating<u32>>();
        assert_channel_f64::<Saturating<u64>>();
    }

    // ─── ADR-0044 Phase 5 guardrails ──────────────────────────────────────
    //
    // Raw `f32` / `f64` are channels, not pixels. The
    // `linear_channel_inventory` test above is the positive half of this
    // invariant (`f32: LinearChannel`, `f64: LinearChannel`); the tests
    // below are the negative half — they assert the *pixel* inventory
    // excludes raw floats. The absences are encoded as commented-out
    // lines; uncommenting either is sufficient to catch a regression
    // that re-introduces `impl LinearPixel for f32` (or any of the
    // sibling pixel traits).
    #[test]
    fn linear_pixel_inventory_excludes_raw_floats() {
        fn assert_linear_pixel<T: crate::pixel::LinearPixel>() {}
        // The commented lines below must NOT compile. If a regression
        // re-introduces `impl LinearPixel for f32`, uncommenting either
        // line is enough to catch it.
        // assert_linear_pixel::<f32>();
        // assert_linear_pixel::<f64>();

        // Sanity: a real pixel type still satisfies the bound.
        assert_linear_pixel::<crate::pixel::MonoF32>();
        assert_linear_pixel::<crate::pixel::MonoF64>();
    }

    #[test]
    fn plain_pixel_inventory_excludes_raw_floats() {
        fn assert_plain_pixel<T: PlainPixel>() {}
        // assert_plain_pixel::<f32>(); // must not compile
        // assert_plain_pixel::<f64>(); // must not compile
        assert_plain_pixel::<u8>();
        assert_plain_pixel::<u16>();
        assert_plain_pixel::<crate::pixel::MonoF32>();
        assert_plain_pixel::<crate::pixel::MonoF64>();
    }

    // ─── ADR-0044 / ADR-0045 / ADR-0046 Phase F.1 positive-list guardrails ──
    //
    // The four inventory tests below are the *positive* half of the
    // post-Phase-E invariant. Each test enumerates every type the library
    // ships that is supposed to satisfy the corresponding trait. If a new
    // pixel type is added to the public API without the corresponding
    // trait derivation/impl, the test stops compiling on the line
    // listing it. If a regression silently drops one of these impls
    // (e.g. a PR removes `LinearPixel` from `Mono16`), this test fails
    // to compile too.
    //
    // The omissions are load-bearing:
    //
    //   * `plain_channel_inventory` includes `f32` / `f64` (they ARE
    //     channels — ADR-0046).
    //   * `plain_pixel_inventory`, `linear_pixel_inventory`, and
    //     `linear_space_inventory` all OMIT raw `f32` / `f64`. That
    //     absence is the post-ADR-0044 invariant: floats are channels,
    //     not pixels.
    //   * `linear_space_inventory` additionally omits Indexed* and
    //     sRGB* pixels (they are pixels but intentionally not in linear
    //     space — see ADR-0033 / `pixel/srgb.rs`).
    //
    // Negative coverage (commented `assert_*::<f32>();` lines) lives in
    // the `_excludes_raw_floats` tests above. A `trybuild`-based hard
    // negative compile harness is deferred (ADR-0044 Phase F.2).

    #[test]
    fn plain_channel_inventory() {
        fn assert_plain_channel<T: PlainChannel>() {}

        // Unsigned integer primitives.
        assert_plain_channel::<u8>();
        assert_plain_channel::<u16>();
        assert_plain_channel::<u32>();
        assert_plain_channel::<u64>();

        // Signed integer primitives.
        assert_plain_channel::<i8>();
        assert_plain_channel::<i16>();
        assert_plain_channel::<i32>();
        assert_plain_channel::<i64>();

        // Floating-point primitives — ADR-0046 raison d'être: raw
        // floats are channels (this list) but NOT pixels (omitted from
        // `plain_pixel_inventory` below).
        assert_plain_channel::<f32>();
        assert_plain_channel::<f64>();

        // `Saturating<T>` blanket — every wrapped channel primitive.
        assert_plain_channel::<Saturating<u8>>();
        assert_plain_channel::<Saturating<u16>>();
        assert_plain_channel::<Saturating<u32>>();
        assert_plain_channel::<Saturating<u64>>();
        assert_plain_channel::<Saturating<i8>>();
        assert_plain_channel::<Saturating<i16>>();
        assert_plain_channel::<Saturating<i32>>();
        assert_plain_channel::<Saturating<i64>>();

        // `PlainPixel: PlainChannel` (ADR-0046 §Decision). Spot-check
        // that the supertrait relationship is wired correctly: every
        // pixel is also a channel.
        assert_plain_channel::<crate::pixel::Mono8>();
        assert_plain_channel::<crate::pixel::MonoF32>();
        assert_plain_channel::<crate::pixel::Rgb8>();
        assert_plain_channel::<crate::pixel::RgbF32>();
    }

    #[test]
    fn plain_pixel_inventory() {
        fn assert_plain_pixel<T: PlainPixel>() {}

        // Channel primitives that double as single-channel pixels.
        // (These predate ADR-0044 and remain pixels — only `f32` /
        // `f64` were revoked.)
        assert_plain_pixel::<u8>();
        assert_plain_pixel::<u16>();
        assert_plain_pixel::<u32>();
        assert_plain_pixel::<u64>();
        assert_plain_pixel::<Saturating<u8>>();
        assert_plain_pixel::<Saturating<u16>>();
        assert_plain_pixel::<Saturating<u32>>();
        assert_plain_pixel::<Saturating<u64>>();

        // Mono / Mono<BITS> / float-mono.
        assert_plain_pixel::<crate::pixel::Mono8>();
        assert_plain_pixel::<crate::pixel::Mono16>();
        assert_plain_pixel::<crate::pixel::Mono32>();
        assert_plain_pixel::<crate::pixel::Mono64>();
        assert_plain_pixel::<crate::pixel::Mono<10>>();
        assert_plain_pixel::<crate::pixel::Mono<12>>();
        assert_plain_pixel::<crate::pixel::Mono<14>>();
        assert_plain_pixel::<crate::pixel::MonoF32>();
        assert_plain_pixel::<crate::pixel::MonoF64>();

        // Mono + alpha.
        assert_plain_pixel::<crate::pixel::MonoA8>();
        assert_plain_pixel::<crate::pixel::MonoA16>();
        assert_plain_pixel::<crate::pixel::MonoA32>();
        assert_plain_pixel::<crate::pixel::MonoA64>();
        assert_plain_pixel::<crate::pixel::MonoAF32>();
        assert_plain_pixel::<crate::pixel::MonoAF64>();

        // RGB / RGBA (fixed-depth + const-generic + float).
        assert_plain_pixel::<crate::pixel::Rgb8>();
        assert_plain_pixel::<crate::pixel::Rgb16>();
        assert_plain_pixel::<crate::pixel::Rgb32>();
        assert_plain_pixel::<crate::pixel::Rgb64>();
        assert_plain_pixel::<crate::pixel::Rgb<10>>();
        assert_plain_pixel::<crate::pixel::Rgb<12>>();
        assert_plain_pixel::<crate::pixel::Rgb<14>>();
        assert_plain_pixel::<crate::pixel::RgbF32>();
        assert_plain_pixel::<crate::pixel::RgbF64>();
        assert_plain_pixel::<crate::pixel::Rgba8>();
        assert_plain_pixel::<crate::pixel::Rgba16>();
        assert_plain_pixel::<crate::pixel::Rgba32>();
        assert_plain_pixel::<crate::pixel::Rgba64>();
        assert_plain_pixel::<crate::pixel::Rgba<10>>();
        assert_plain_pixel::<crate::pixel::Rgba<12>>();
        assert_plain_pixel::<crate::pixel::Rgba<14>>();
        assert_plain_pixel::<crate::pixel::RgbaF32>();
        assert_plain_pixel::<crate::pixel::RgbaF64>();

        // BGR / BGRA (fixed-depth + const-generic + float).
        assert_plain_pixel::<crate::pixel::Bgr8>();
        assert_plain_pixel::<crate::pixel::Bgr16>();
        assert_plain_pixel::<crate::pixel::Bgr32>();
        assert_plain_pixel::<crate::pixel::Bgr64>();
        assert_plain_pixel::<crate::pixel::Bgr<10>>();
        assert_plain_pixel::<crate::pixel::Bgr<12>>();
        assert_plain_pixel::<crate::pixel::Bgr<14>>();
        assert_plain_pixel::<crate::pixel::BgrF32>();
        assert_plain_pixel::<crate::pixel::BgrF64>();
        assert_plain_pixel::<crate::pixel::Bgra8>();
        assert_plain_pixel::<crate::pixel::Bgra16>();
        assert_plain_pixel::<crate::pixel::Bgra32>();
        assert_plain_pixel::<crate::pixel::Bgra64>();
        assert_plain_pixel::<crate::pixel::Bgra<10>>();
        assert_plain_pixel::<crate::pixel::Bgra<12>>();
        assert_plain_pixel::<crate::pixel::Bgra<14>>();
        assert_plain_pixel::<crate::pixel::BgraF32>();
        assert_plain_pixel::<crate::pixel::BgraF64>();

        // sRGB pixels (PlainPixel only — they live in a non-linear
        // space, so `LinearPixel` / `LinearSpace` are intentionally
        // not implemented; see `linear_pixel_inventory`).
        assert_plain_pixel::<crate::pixel::Srgb8>();
        assert_plain_pixel::<crate::pixel::Srgb16>();
        assert_plain_pixel::<crate::pixel::Srgba8>();
        assert_plain_pixel::<crate::pixel::Srgba16>();
        assert_plain_pixel::<crate::pixel::SrgbMono8>();
        assert_plain_pixel::<crate::pixel::SrgbMono16>();
        assert_plain_pixel::<crate::pixel::SrgbMonoA8>();
        assert_plain_pixel::<crate::pixel::SrgbMonoA16>();

        // Indexed (PlainPixel only — interpolating palette indices is
        // mathematically meaningless, so `LinearPixel` /
        // `LinearSpace` are intentionally not implemented).
        assert_plain_pixel::<crate::pixel::Indexed8>();

        // Intentionally omitted (ADR-0044 Phase E):
        //   assert_plain_pixel::<f32>();
        //   assert_plain_pixel::<f64>();
        // Raw floats are channels (`plain_channel_inventory`), not
        // pixels. Use `MonoF32` / `MonoF64` for the pixel role.
    }

    #[test]
    fn linear_pixel_inventory() {
        fn assert_linear_pixel<T: crate::pixel::LinearPixel>() {}

        // Channel-primitive single-channel pixels: NOT in this list.
        // After ADR-0045 Phase S4, channel primitives are channels,
        // not pixels — `u8`, `Saturating<u8>`, etc. no longer
        // implement `LinearPixel`. Use `Mono8` / `Mono16` / etc.

        // Mono / Mono<BITS> / float-mono.
        assert_linear_pixel::<crate::pixel::Mono8>();
        assert_linear_pixel::<crate::pixel::Mono16>();
        assert_linear_pixel::<crate::pixel::Mono32>();
        assert_linear_pixel::<crate::pixel::Mono64>();
        assert_linear_pixel::<crate::pixel::Mono<10>>();
        assert_linear_pixel::<crate::pixel::Mono<12>>();
        assert_linear_pixel::<crate::pixel::Mono<14>>();
        assert_linear_pixel::<crate::pixel::MonoF32>();
        assert_linear_pixel::<crate::pixel::MonoF64>();

        // Mono + alpha.
        assert_linear_pixel::<crate::pixel::MonoA8>();
        assert_linear_pixel::<crate::pixel::MonoA16>();
        assert_linear_pixel::<crate::pixel::MonoA32>();
        assert_linear_pixel::<crate::pixel::MonoA64>();
        assert_linear_pixel::<crate::pixel::MonoAF32>();
        assert_linear_pixel::<crate::pixel::MonoAF64>();

        // RGB / RGBA.
        assert_linear_pixel::<crate::pixel::Rgb8>();
        assert_linear_pixel::<crate::pixel::Rgb16>();
        assert_linear_pixel::<crate::pixel::Rgb32>();
        assert_linear_pixel::<crate::pixel::Rgb64>();
        assert_linear_pixel::<crate::pixel::Rgb<10>>();
        assert_linear_pixel::<crate::pixel::Rgb<12>>();
        assert_linear_pixel::<crate::pixel::Rgb<14>>();
        assert_linear_pixel::<crate::pixel::RgbF32>();
        assert_linear_pixel::<crate::pixel::RgbF64>();
        assert_linear_pixel::<crate::pixel::Rgba8>();
        assert_linear_pixel::<crate::pixel::Rgba16>();
        assert_linear_pixel::<crate::pixel::Rgba32>();
        assert_linear_pixel::<crate::pixel::Rgba64>();
        assert_linear_pixel::<crate::pixel::Rgba<10>>();
        assert_linear_pixel::<crate::pixel::Rgba<12>>();
        assert_linear_pixel::<crate::pixel::Rgba<14>>();
        assert_linear_pixel::<crate::pixel::RgbaF32>();
        assert_linear_pixel::<crate::pixel::RgbaF64>();

        // BGR / BGRA.
        assert_linear_pixel::<crate::pixel::Bgr8>();
        assert_linear_pixel::<crate::pixel::Bgr16>();
        assert_linear_pixel::<crate::pixel::Bgr32>();
        assert_linear_pixel::<crate::pixel::Bgr64>();
        assert_linear_pixel::<crate::pixel::Bgr<10>>();
        assert_linear_pixel::<crate::pixel::Bgr<12>>();
        assert_linear_pixel::<crate::pixel::Bgr<14>>();
        assert_linear_pixel::<crate::pixel::BgrF32>();
        assert_linear_pixel::<crate::pixel::BgrF64>();
        assert_linear_pixel::<crate::pixel::Bgra8>();
        assert_linear_pixel::<crate::pixel::Bgra16>();
        assert_linear_pixel::<crate::pixel::Bgra32>();
        assert_linear_pixel::<crate::pixel::Bgra64>();
        assert_linear_pixel::<crate::pixel::Bgra<10>>();
        assert_linear_pixel::<crate::pixel::Bgra<12>>();
        assert_linear_pixel::<crate::pixel::Bgra<14>>();
        assert_linear_pixel::<crate::pixel::BgraF32>();
        assert_linear_pixel::<crate::pixel::BgraF64>();

        // Intentionally omitted:
        //   * `f32` / `f64` (ADR-0044 Phase E — channels, not pixels).
        //   * `u8..u64`, `i8..i64`, `Saturating<_>` (ADR-0045 Phase S4
        //     — channels, not pixels).
        //   * `Srgb*` (non-linear space — convert with `SrgbGamma`).
        //   * `Indexed8` (palette index, not a color value).
    }

    #[test]
    fn linear_space_inventory() {
        fn assert_linear_space<T: crate::pixel::LinearSpace>() {}

        // `LinearSpace: LinearPixel` is a marker trait that asserts
        // the pixel's values can be linearly interpolated. Every
        // `LinearPixel` shipped by the library that lives in a linear
        // space is enumerated below.

        // Mono / Mono<BITS> / float-mono.
        assert_linear_space::<crate::pixel::Mono8>();
        assert_linear_space::<crate::pixel::Mono16>();
        assert_linear_space::<crate::pixel::Mono32>();
        assert_linear_space::<crate::pixel::Mono64>();
        assert_linear_space::<crate::pixel::Mono<10>>();
        assert_linear_space::<crate::pixel::Mono<12>>();
        assert_linear_space::<crate::pixel::Mono<14>>();
        assert_linear_space::<crate::pixel::MonoF32>();
        assert_linear_space::<crate::pixel::MonoF64>();

        // Mono + alpha.
        assert_linear_space::<crate::pixel::MonoA8>();
        assert_linear_space::<crate::pixel::MonoA16>();
        assert_linear_space::<crate::pixel::MonoA32>();
        assert_linear_space::<crate::pixel::MonoA64>();
        assert_linear_space::<crate::pixel::MonoAF32>();
        assert_linear_space::<crate::pixel::MonoAF64>();

        // RGB / RGBA.
        assert_linear_space::<crate::pixel::Rgb8>();
        assert_linear_space::<crate::pixel::Rgb16>();
        assert_linear_space::<crate::pixel::Rgb32>();
        assert_linear_space::<crate::pixel::Rgb64>();
        assert_linear_space::<crate::pixel::Rgb<10>>();
        assert_linear_space::<crate::pixel::Rgb<12>>();
        assert_linear_space::<crate::pixel::Rgb<14>>();
        assert_linear_space::<crate::pixel::RgbF32>();
        assert_linear_space::<crate::pixel::RgbF64>();
        assert_linear_space::<crate::pixel::Rgba8>();
        assert_linear_space::<crate::pixel::Rgba16>();
        assert_linear_space::<crate::pixel::Rgba32>();
        assert_linear_space::<crate::pixel::Rgba64>();
        assert_linear_space::<crate::pixel::Rgba<10>>();
        assert_linear_space::<crate::pixel::Rgba<12>>();
        assert_linear_space::<crate::pixel::Rgba<14>>();
        assert_linear_space::<crate::pixel::RgbaF32>();
        assert_linear_space::<crate::pixel::RgbaF64>();

        // BGR / BGRA.
        assert_linear_space::<crate::pixel::Bgr8>();
        assert_linear_space::<crate::pixel::Bgr16>();
        assert_linear_space::<crate::pixel::Bgr32>();
        assert_linear_space::<crate::pixel::Bgr64>();
        assert_linear_space::<crate::pixel::Bgr<10>>();
        assert_linear_space::<crate::pixel::Bgr<12>>();
        assert_linear_space::<crate::pixel::Bgr<14>>();
        assert_linear_space::<crate::pixel::BgrF32>();
        assert_linear_space::<crate::pixel::BgrF64>();
        assert_linear_space::<crate::pixel::Bgra8>();
        assert_linear_space::<crate::pixel::Bgra16>();
        assert_linear_space::<crate::pixel::Bgra32>();
        assert_linear_space::<crate::pixel::Bgra64>();
        assert_linear_space::<crate::pixel::Bgra<10>>();
        assert_linear_space::<crate::pixel::Bgra<12>>();
        assert_linear_space::<crate::pixel::Bgra<14>>();
        assert_linear_space::<crate::pixel::BgraF32>();
        assert_linear_space::<crate::pixel::BgraF64>();

        // Intentionally omitted:
        //   * `f32` / `f64` (ADR-0044 Phase E — not pixels).
        //   * Channel primitives (ADR-0045 Phase S4 — not pixels).
        //   * `Srgb*` (gamma-encoded — convert to linear first).
        //   * `Indexed8` (palette index — interpolation meaningless).
    }
}

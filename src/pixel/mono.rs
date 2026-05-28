//! Monochrome (single-channel) pixel types.
//!
//! - [`Mono<BITS>`] — const-generic for 10/12/14-bit depths
//! - [`Mono8`], [`Mono16`], [`Mono32`], [`Mono64`] — fixed-depth wrappers
//! - [`MonoF32`], [`MonoF64`] — floating-point monochrome

use fovea_derive::{HomogeneousPixel, LinearPixel, PlainPixel, WhiteChannel, ZeroablePixel};

use crate::pixel::{
    FromLinear, HomogeneousPixel, IntegralPixel, IntegralSquaredPixel, LinearChannel, LinearPixel,
    LinearSpace, PlainChannel, PlainPixel, WhiteChannel, ZeroablePixel,
};
use std::{
    hash::{Hash, Hasher},
    num::Saturating,
    ops::Mul,
};

// Re-use the canonicalize helpers from the parent module
use super::{canonicalize_f32, canonicalize_f64};

/// The `Mono` struct represents a grayscale pixel with 10-bit depth,
/// 12-bit depth, or 14-bit depth.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct Mono<const BITS: usize> {
    d: Saturating<u16>,
}
pub type Mono10 = Mono<10>;
pub type Mono12 = Mono<12>;
pub type Mono14 = Mono<14>;

impl<const BITS: usize> Mono<BITS> {
    /// Compile-time assertion: BITS must be 10, 12, or 14.
    /// Referenced by `MAX` to ensure evaluation whenever the impl is instantiated.
    const _ASSERT_BITS: () = assert!(BITS == 10 || BITS == 12 || BITS == 14);

    // By destructuring `_ASSERT_BITS` in the initializer we force the compiler
    // to evaluate the assertion for every concrete `BITS` that reaches `MAX`.
    // `MAX` is used by `new()`, `From`, and arithmetic ops, so any real usage
    // of `Mono<BITS>` will trigger the check.
    pub(crate) const MAX: u16 = {
        // Force evaluation of the assertion
        let () = Self::_ASSERT_BITS;
        (1 << BITS) - 1
    };

    pub fn new(value: u16) -> Self {
        Mono {
            d: Saturating(value.min(Self::MAX)),
        }
    }

    pub fn value(&self) -> u16 {
        self.d.0
    }
}
impl<const BITS: usize> From<Mono<BITS>> for u16 {
    fn from(value: Mono<BITS>) -> Self {
        value.d.0
    }
}
impl<const BITS: usize> From<u16> for Mono<BITS> {
    fn from(value: u16) -> Self {
        Mono::new(value)
    }
}
impl<const BITS: usize> From<Mono<BITS>> for Saturating<u16> {
    fn from(value: Mono<BITS>) -> Self {
        value.d
    }
}
impl<const BITS: usize> From<Saturating<u16>> for Mono<BITS> {
    fn from(value: Saturating<u16>) -> Self {
        Mono::new(value.0)
    }
}

impl<const BITS: usize> std::ops::Add for Mono<BITS> {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        (self.d + other.d).into()
    }
}
impl<const BITS: usize> std::ops::Add for &Mono<BITS> {
    type Output = Mono<BITS>;

    fn add(self, other: Self) -> Self::Output {
        (self.d + other.d).into()
    }
}
impl<const BITS: usize> std::ops::AddAssign for Mono<BITS> {
    fn add_assign(&mut self, other: Self) {
        self.d += other.d;
        self.d.0 = self.d.0.min(Self::MAX);
    }
}
impl<const BITS: usize> std::ops::AddAssign<&Mono<BITS>> for Mono<BITS> {
    fn add_assign(&mut self, other: &Mono<BITS>) {
        self.d += other.d;
        self.d.0 = self.d.0.min(Self::MAX);
    }
}
impl<const BITS: usize> std::ops::AddAssign<&u16> for Mono<BITS> {
    fn add_assign(&mut self, other: &u16) {
        self.d += *other;
        self.d.0 = self.d.0.min(Self::MAX);
    }
}

impl<const BITS: usize> std::ops::Sub for Mono<BITS> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        (self.d - other.d).into()
    }
}
impl<const BITS: usize> std::ops::Sub for &Mono<BITS> {
    type Output = Mono<BITS>;

    fn sub(self, other: Self) -> Mono<BITS> {
        (self.d - other.d).into()
    }
}
impl<const BITS: usize> std::ops::SubAssign for Mono<BITS> {
    fn sub_assign(&mut self, other: Self) {
        self.d -= other.d;
    }
}
impl<const BITS: usize> std::ops::SubAssign<&Mono<BITS>> for Mono<BITS> {
    fn sub_assign(&mut self, other: &Mono<BITS>) {
        self.d -= other.d;
    }
}
impl<const BITS: usize> std::ops::SubAssign<&u16> for Mono<BITS> {
    fn sub_assign(&mut self, other: &u16) {
        self.d -= *other;
    }
}

impl<const BITS: usize> std::ops::Mul for Mono<BITS> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        // Widen to u32 to avoid u16 overflow, then clamp to MAX.
        let result = (self.d.0 as u32) * (other.d.0 as u32);
        Mono::new((result.min(Self::MAX as u32)) as u16)
    }
}
impl<const BITS: usize> std::ops::Mul for &Mono<BITS> {
    type Output = Mono<BITS>;

    fn mul(self, other: Self) -> Mono<BITS> {
        // Widen to u32 to avoid u16 overflow, then clamp to MAX.
        let result = (self.d.0 as u32) * (other.d.0 as u32);
        Mono::new((result.min(Mono::<BITS>::MAX as u32)) as u16)
    }
}
impl<const BITS: usize> std::ops::MulAssign for Mono<BITS> {
    fn mul_assign(&mut self, other: Self) {
        // Widen to u32 to avoid u16 overflow, then clamp to MAX.
        let result = (self.d.0 as u32) * (other.d.0 as u32);
        self.d.0 = (result.min(Self::MAX as u32)) as u16;
    }
}
impl<const BITS: usize> std::ops::MulAssign<&Mono<BITS>> for Mono<BITS> {
    fn mul_assign(&mut self, other: &Mono<BITS>) {
        // Widen to u32 to avoid u16 overflow, then clamp to MAX.
        let result = (self.d.0 as u32) * (other.d.0 as u32);
        self.d.0 = (result.min(Self::MAX as u32)) as u16;
    }
}
impl<const BITS: usize> std::ops::MulAssign<&u16> for Mono<BITS> {
    fn mul_assign(&mut self, other: &u16) {
        // Widen to u32 to avoid u16 overflow, then clamp to MAX.
        let result = (self.d.0 as u32) * (*other as u32);
        self.d.0 = (result.min(Self::MAX as u32)) as u16;
    }
}

impl<const BITS: usize> std::ops::Div for Mono<BITS> {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        (self.d / other.d).into()
    }
}
impl<const BITS: usize> std::ops::Div for &Mono<BITS> {
    type Output = Mono<BITS>;

    fn div(self, other: Self) -> Mono<BITS> {
        (self.d / other.d).into()
    }
}
impl<const BITS: usize> std::ops::DivAssign for Mono<BITS> {
    fn div_assign(&mut self, other: Self) {
        self.d /= other.d;
        // No clamping needed: integer division can only decrease a value.
    }
}
impl<const BITS: usize> std::ops::DivAssign<&Mono<BITS>> for Mono<BITS> {
    fn div_assign(&mut self, other: &Mono<BITS>) {
        self.d /= other.d;
        // No clamping needed: integer division can only decrease a value.
    }
}
impl<const BITS: usize> std::ops::DivAssign<&u16> for Mono<BITS> {
    fn div_assign(&mut self, other: &u16) {
        self.d.0 /= *other;
        // No clamping needed: integer division can only decrease a value.
    }
}

/// The `Mono8Pixel` struct represents a grayscale pixel with 8-bit depth.
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
#[repr(transparent)]
// ADR-0045 Phase B: accumulator promoted from raw `f32` to the named
// pixel type `MonoF32`. The derive macro wraps via `Into::into` at
// the single-field boundary; `MonoF32` is `#[repr(transparent)]` over
// `f32`, so the numeric body and bit pattern are unchanged.
#[linear(accumulator = MonoF32)]
pub struct Mono8(Saturating<u8>);
impl Mono8 {
    pub fn new(value: u8) -> Self {
        Mono8(Saturating(value))
    }

    /// Returns the raw `u8` intensity value.
    #[inline]
    pub fn value(self) -> u8 {
        self.0.0
    }
}
impl From<Mono8> for u8 {
    #[inline]
    fn from(p: Mono8) -> Self {
        p.0.0
    }
}
impl From<u8> for Mono8 {
    #[inline]
    fn from(v: u8) -> Self {
        Mono8::new(v)
    }
}
impl Mul<f32> for Mono8 {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self::Output {
        let r = (self.0.0 as f32 * rhs).round();
        Mono8(Saturating(r.min(255.0) as u8))
    }
}
impl Mul<f32> for &Mono8 {
    type Output = Mono8;
    fn mul(self, rhs: f32) -> Self::Output {
        let r = (self.0.0 as f32 * rhs).round();
        Mono8(Saturating(r.min(255.0) as u8))
    }
}

/// The `Mono16Pixel` struct represents a grayscale pixel with 16-bit depth.
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
// ADR-0045 Phase B: accumulator → `MonoF32` (see `Mono8` above).
#[linear(accumulator = MonoF32)]
pub struct Mono16(Saturating<u16>);
impl Mono16 {
    pub fn new(value: u16) -> Self {
        Mono16(Saturating(value))
    }

    /// Returns the raw `u16` intensity value.
    #[inline]
    pub fn value(self) -> u16 {
        self.0.0
    }
}
impl From<Mono16> for u16 {
    #[inline]
    fn from(p: Mono16) -> Self {
        p.0.0
    }
}
impl From<u16> for Mono16 {
    #[inline]
    fn from(v: u16) -> Self {
        Mono16::new(v)
    }
}

/// The `Mono32Pixel` struct represents a grayscale pixel with 32-bit depth.
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
// ADR-0045 Phase B: accumulator → `MonoF64` (see `Mono8` above).
#[linear(accumulator = MonoF64)]
pub struct Mono32(Saturating<u32>);
impl Mono32 {
    pub fn new(value: u32) -> Self {
        Mono32(Saturating(value))
    }

    /// Returns the raw `u32` intensity value.
    #[inline]
    pub fn value(self) -> u32 {
        self.0.0
    }
}
impl From<Mono32> for u32 {
    #[inline]
    fn from(p: Mono32) -> Self {
        p.0.0
    }
}
impl From<u32> for Mono32 {
    #[inline]
    fn from(v: u32) -> Self {
        Mono32::new(v)
    }
}

// f64-scalar LinearPixel impl for Mono32 — preserves precision for f64
// pipelines (PLAN §3.4 scalar-precision note). The derive above already
// emits `impl LinearPixel<f32> for Mono32` with `Accumulator = MonoF64`;
// this adds a parallel `LinearPixel<f64>` impl (same accumulator) so
// strategies parameterized by scalar type (e.g. `BrightnessContrast<f64>`)
// can pick up the matching impl via trait resolution without an
// `f32 → f64` widening in the hot loop.
//
// ADR-0045 Phase S4: the underlying `Saturating<u32>` is a channel
// (implements `LinearChannel<f64>`, not `LinearPixel<f64>`). Delegate
// through the channel trait.
//
// ADR-0045 Phase B: accumulator promoted from raw `f64` to `MonoF64`.
// The `Saturating<u32>` channel's accumulator is still `f64`; the
// pixel-level accumulator wraps it via `MonoF64(..)`.
impl LinearPixel<f64> for Mono32 {
    type Accumulator = MonoF64;
    #[inline(always)]
    fn to_accumulator(&self) -> MonoF64 {
        MonoF64(<Saturating<u32> as LinearChannel<f64>>::to_accumulator(
            &self.0,
        ))
    }
    #[inline(always)]
    fn scale(&self, scalar: f64) -> MonoF64 {
        MonoF64(<Saturating<u32> as LinearChannel<f64>>::scale(
            &self.0, scalar,
        ))
    }
    #[inline(always)]
    fn uniform(scalar: f64) -> MonoF64 {
        MonoF64(<Saturating<u32> as LinearChannel<f64>>::uniform(scalar))
    }
    #[inline(always)]
    fn scale_add(&self, scalar: f64, addend: MonoF64) -> MonoF64 {
        MonoF64(<Saturating<u32> as LinearChannel<f64>>::scale_add(
            &self.0, scalar, addend.0,
        ))
    }
}

/// The `Mono64Pixel` struct represents a grayscale pixel with 64-bit depth.
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
// ADR-0045 Phase B: accumulator → `MonoF64` (see `Mono32` above).
#[linear(accumulator = MonoF64)]
pub struct Mono64(Saturating<u64>);
impl Mono64 {
    pub fn new(value: u64) -> Self {
        Mono64(Saturating(value))
    }

    /// Returns the raw `u64` intensity value.
    #[inline]
    pub fn value(self) -> u64 {
        self.0.0
    }
}
impl From<Mono64> for u64 {
    #[inline]
    fn from(p: Mono64) -> Self {
        p.0.0
    }
}
impl From<u64> for Mono64 {
    #[inline]
    fn from(v: u64) -> Self {
        Mono64::new(v)
    }
}

// f64-scalar LinearPixel impl for Mono64 — preserves precision for f64
// pipelines (PLAN §3.4). See the Mono32 counterpart above for rationale.
// ADR-0045 Phase S4: delegate through `LinearChannel<f64>`.
// ADR-0045 Phase B: accumulator → `MonoF64`.
impl LinearPixel<f64> for Mono64 {
    type Accumulator = MonoF64;
    #[inline(always)]
    fn to_accumulator(&self) -> MonoF64 {
        MonoF64(<Saturating<u64> as LinearChannel<f64>>::to_accumulator(
            &self.0,
        ))
    }
    #[inline(always)]
    fn scale(&self, scalar: f64) -> MonoF64 {
        MonoF64(<Saturating<u64> as LinearChannel<f64>>::scale(
            &self.0, scalar,
        ))
    }
    #[inline(always)]
    fn uniform(scalar: f64) -> MonoF64 {
        MonoF64(<Saturating<u64> as LinearChannel<f64>>::uniform(scalar))
    }
    #[inline(always)]
    fn scale_add(&self, scalar: f64, addend: MonoF64) -> MonoF64 {
        MonoF64(<Saturating<u64> as LinearChannel<f64>>::scale_add(
            &self.0, scalar, addend.0,
        ))
    }
}

/// Grayscale pixel with 32-bit floating point depth.
///
/// Bare `f32` is a [`PlainChannel`](crate::pixel::PlainChannel) but **not** a
/// [`PlainPixel`](crate::pixel::PlainPixel) (ADR-0046 — channels are not
/// pixels). `MonoF32` is the actual pixel type: it carries the semantic
/// meaning "this is a pixel intensity value" per Philosophy §1 and
/// participates in the pixel-typed APIs.
///
/// # Examples
///
/// ```
/// # use fovea::pixel::MonoF32;
/// let p = MonoF32::new(0.5);
/// assert_eq!(p.value(), 0.5);
/// assert_eq!(f32::from(p), 0.5);
/// ```
#[repr(transparent)]
#[derive(
    Clone, Copy, Debug, PartialEq, PartialOrd, PlainPixel, HomogeneousPixel, ZeroablePixel,
)]
// ADR-0044 + ADR-0046: the inner `f32` is a `PlainChannel` /
// `LinearChannel` but not a `ZeroablePixel` (that role is
// pixel-only). Use `Default` for field zero initialisation.
pub struct MonoF32(#[zero(default)] pub f32);
impl MonoF32 {
    #[inline]
    pub fn new(value: f32) -> Self {
        MonoF32(value)
    }

    /// Returns the raw `f32` intensity value.
    #[inline]
    pub fn value(self) -> f32 {
        self.0
    }

    /// Returns a new `MonoF32` with the absolute value of the underlying
    /// `f32` intensity.
    ///
    /// Ergonomic helper for test asserts and per-pixel arithmetic under
    /// ADR-0045 Phase B / ADR-0044 Phase C, where pixel-role `f32` values
    /// have been promoted to `MonoF32`. Mirrors `f32::abs` at the pixel
    /// layer and avoids repeated `.0` extraction at comparison boundaries.
    #[inline]
    pub fn abs(self) -> MonoF32 {
        MonoF32(self.0.abs())
    }
}
impl From<MonoF32> for f32 {
    #[inline]
    fn from(p: MonoF32) -> Self {
        p.0
    }
}
impl From<f32> for MonoF32 {
    #[inline]
    fn from(v: f32) -> Self {
        MonoF32::new(v)
    }
}
// Widening `MonoF32 -> f64` mirrors the primitive widening `f32 -> f64`.
// Used by scan-based strategies (`AutoContrast::scan`) whose pixel bound
// is `V::Pixel: Into<f64>`. The inner `f32 -> f64` cast is lossless and
// standard; providing it here keeps `MonoF32` a drop-in replacement for
// raw-`f32` pixel callsites after ADR-0044 Phase E.
impl From<MonoF32> for f64 {
    #[inline]
    fn from(p: MonoF32) -> Self {
        p.0 as f64
    }
}

impl Hash for MonoF32 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        canonicalize_f32(self.0).hash(state);
    }
}

/// Grayscale pixel with 64-bit floating point depth.
///
/// Bare `f64` is a [`PlainChannel`](crate::pixel::PlainChannel) but **not** a
/// [`PlainPixel`](crate::pixel::PlainPixel) (ADR-0046 — channels are not
/// pixels). `MonoF64` is the actual pixel type: it carries the semantic
/// meaning "this is a pixel intensity value" per Philosophy §1 and
/// participates in the pixel-typed APIs.
///
/// # Examples
///
/// ```
/// # use fovea::pixel::MonoF64;
/// let p = MonoF64::new(0.5);
/// assert_eq!(p.value(), 0.5);
/// assert_eq!(f64::from(p), 0.5);
/// ```
#[repr(transparent)]
#[derive(
    Clone, Copy, Debug, PartialEq, PartialOrd, PlainPixel, HomogeneousPixel, ZeroablePixel,
)]
// ADR-0044 + ADR-0046: inner `f64` is a channel, not a pixel.
pub struct MonoF64(#[zero(default)] pub f64);
impl MonoF64 {
    #[inline]
    pub fn new(value: f64) -> Self {
        MonoF64(value)
    }

    /// Returns the raw `f64` intensity value.
    #[inline]
    pub fn value(self) -> f64 {
        self.0
    }

    /// Returns a new `MonoF64` with the absolute value of the underlying
    /// `f64` intensity. See [`MonoF32::abs`] for rationale.
    #[inline]
    pub fn abs(self) -> MonoF64 {
        MonoF64(self.0.abs())
    }
}
impl From<MonoF64> for f64 {
    #[inline]
    fn from(p: MonoF64) -> Self {
        p.0
    }
}
impl From<f64> for MonoF64 {
    #[inline]
    fn from(v: f64) -> Self {
        MonoF64::new(v)
    }
}

impl Hash for MonoF64 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        canonicalize_f64(self.0).hash(state);
    }
}

// ---------------------------------------------------------------------------
// PlainPixel / ZeroablePixel / LinearPixel / HomogeneousPixel for Mono<BITS>
// ---------------------------------------------------------------------------

// ADR-0046: `PlainPixel` extends `PlainChannel`, so the generic
// hand-written impl for `Mono<BITS>` needs a matching
// `PlainChannel` impl. Body is empty — defaults suffice.
//
// SAFETY: `Mono<BITS>` is `#[repr(transparent)]` over
// `Saturating<u16>`; layout, round-trip, and bit-pattern
// validity are inherited from `u16`.
unsafe impl<const BITS: usize> PlainChannel for Mono<BITS> {}
unsafe impl<const BITS: usize> PlainPixel for Mono<BITS> {
    const CHANNELS: &'static [usize] = &[2];
}
impl<const BITS: usize> ZeroablePixel for Mono<BITS> {
    fn zero() -> Self {
        Saturating(0).into()
    }
}
// ADR-0045 Phase B: accumulator promoted from raw `f32` to `MonoF32`.
// The underlying `Saturating<u16>` channel keeps `LinearChannel::
// Accumulator = f32`; the pixel-level accumulator wraps via
// `MonoF32(..)`. Numerical identity preserved because `MonoF32` is
// `#[repr(transparent)]` over `f32`.
impl<const BITS: usize> LinearPixel for Mono<BITS> {
    type Accumulator = MonoF32;
    #[inline(always)]
    fn to_accumulator(&self) -> MonoF32 {
        MonoF32(<Saturating<u16> as LinearChannel<f32>>::to_accumulator(
            &self.d,
        ))
    }
    #[inline(always)]
    fn scale(&self, scalar: f32) -> Self::Accumulator {
        MonoF32(<Saturating<u16> as LinearChannel<f32>>::scale(
            &self.d, scalar,
        ))
    }
    #[inline(always)]
    fn scale_add(&self, scalar: f32, addend: MonoF32) -> MonoF32 {
        MonoF32(<Saturating<u16> as LinearChannel<f32>>::scale_add(
            &self.d, scalar, addend.0,
        ))
    }
    #[inline(always)]
    fn uniform(scalar: f32) -> MonoF32 {
        MonoF32(scalar)
    }
}
impl<const BITS: usize> FromLinear<f32> for Mono<BITS> {
    #[inline(always)]
    fn from_linear(acc: f32) -> Self {
        Mono::new(acc.round().clamp(0.0, Mono::<BITS>::MAX as f32) as u16)
    }
}
// ADR-0045 Phase A: named-float-accumulator sibling. Delegates to
// the bare-float body; `MonoF32` is `#[repr(transparent)] over f32`,
// so the `.0` extraction is zero-cost.
impl<const BITS: usize> FromLinear<MonoF32> for Mono<BITS> {
    #[inline(always)]
    fn from_linear(acc: MonoF32) -> Self {
        <Self as FromLinear<f32>>::from_linear(acc.0)
    }
}
unsafe impl<const BITS: usize> HomogeneousPixel for Mono<BITS> {
    type Channel = Saturating<u16>;
    type Channels = [Saturating<u16>; 1];
}

// Manual `WhiteChannel` impl for `Mono<BITS>` (ADR-0043).
//
// `Mono<BITS>` uses `Saturating<u16>` as its channel type but carries the
// stronger invariant that the raw value fits in `BITS` bits. The
// `BoundedChannel::MAX` of `Saturating<u16>` is `65535`, which would
// violate `Mono<10>`'s `value <= 1023` invariant when written back via
// `HomogeneousPixel::from_channels` (a layout-only primitive — see
// ADR-0015). Returning `Saturating(Self::MAX)` here gives strategies
// like `Invert` and `BinaryThreshold` the pixel-level "white" value
// (`(1 << BITS) - 1`), preserving the invariant that every
// constructor on `Mono<BITS>` enforces.
impl<const BITS: usize> WhiteChannel for Mono<BITS> {
    #[inline(always)]
    fn white_channel() -> Saturating<u16> {
        // Forces evaluation of `_ASSERT_BITS` for every instantiated BITS.
        Saturating(<Mono<BITS>>::MAX)
    }
}

// ---------------------------------------------------------------------------
// LinearSpace impls for MonoF32, MonoF64, and Mono<BITS>
// ---------------------------------------------------------------------------

impl LinearSpace for MonoF32 {}
impl LinearSpace for MonoF64 {}

impl std::ops::Add for MonoF32 {
    type Output = Self;
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        MonoF32(self.0 + other.0)
    }
}

impl std::ops::Sub for MonoF32 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        MonoF32(self.0 - other.0)
    }
}

impl std::ops::Mul for MonoF32 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        MonoF32(self.0 * other.0)
    }
}

impl LinearPixel for MonoF32 {
    type Accumulator = Self;
    #[inline(always)]
    fn to_accumulator(&self) -> Self {
        *self
    }
    #[inline(always)]
    fn scale(&self, scalar: f32) -> Self {
        MonoF32(self.0 * scalar)
    }
    #[inline(always)]
    fn scale_add(&self, scalar: f32, addend: Self) -> Self {
        #[cfg(target_feature = "fma")]
        {
            MonoF32(self.0.mul_add(scalar, addend.0))
        }
        #[cfg(not(target_feature = "fma"))]
        {
            MonoF32(self.0 * scalar + addend.0)
        }
    }
    #[inline(always)]
    fn uniform(scalar: f32) -> Self {
        MonoF32(scalar)
    }
}

impl std::ops::Add for MonoF64 {
    type Output = Self;
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        MonoF64(self.0 + other.0)
    }
}

impl std::ops::Sub for MonoF64 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        MonoF64(self.0 - other.0)
    }
}

impl std::ops::Mul for MonoF64 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        MonoF64(self.0 * other.0)
    }
}

impl LinearPixel for MonoF64 {
    type Accumulator = Self;
    #[inline(always)]
    fn to_accumulator(&self) -> Self {
        *self
    }
    #[inline(always)]
    fn scale(&self, scalar: f32) -> Self {
        MonoF64(self.0 * scalar as f64)
    }
    #[inline(always)]
    fn uniform(scalar: f32) -> Self {
        MonoF64(scalar as f64)
    }
    #[inline(always)]
    fn scale_add(&self, scalar: f32, addend: Self) -> Self {
        #[cfg(target_feature = "fma")]
        {
            MonoF64(self.0.mul_add(scalar as f64, addend.0))
        }
        #[cfg(not(target_feature = "fma"))]
        {
            MonoF64(self.0 * scalar as f64 + addend.0)
        }
    }
}

// f64-scalar LinearPixel impl for MonoF64 — preserves precision for f64
// pipelines (PLAN §3.4 scalar-precision note). Both the f32-scalar impl
// above and this impl share `Accumulator = Self`; the trait's scalar
// parameter is what distinguishes them.
impl LinearPixel<f64> for MonoF64 {
    type Accumulator = Self;
    #[inline(always)]
    fn to_accumulator(&self) -> Self {
        *self
    }
    #[inline(always)]
    fn scale(&self, scalar: f64) -> Self {
        MonoF64(self.0 * scalar)
    }
    #[inline(always)]
    fn uniform(scalar: f64) -> Self {
        MonoF64(scalar)
    }
    #[inline(always)]
    fn scale_add(&self, scalar: f64, addend: Self) -> Self {
        #[cfg(target_feature = "fma")]
        {
            MonoF64(self.0.mul_add(scalar, addend.0))
        }
        #[cfg(not(target_feature = "fma"))]
        {
            MonoF64(self.0 * scalar + addend.0)
        }
    }
}

impl<const BITS: usize> LinearSpace for Mono<BITS> {}

// ---------------------------------------------------------------------------
// IntegralPixel / IntegralSquaredPixel impls (ADR-0032)
// ---------------------------------------------------------------------------
//
// Each `impl IntegralPixel<A> for Src` connects a *source* pixel `Src` to a
// permitted accumulator pixel `A` for the summed-area-table engine in
// `crate::analyze::integral`. Combinations are deliberately closed and
// small (no macros) per the plan; the trait's correctness contract is
// the tightness of `max_integral_value()` (ADR-0032 §2 / Philosophy §11).
//
// Source coverage (this file): `Mono8`, `Mono16`, `Mono32`, `MonoF32`,
// `MonoF64`. RGB sources live in `rgb.rs`. `Mono<BITS>`,
// `MonoA*`, `Srgb*`, and `Indexed8` are intentionally omitted — see
// ADR-0032 §2 / `INTEGRAL_IMAGE_PLAN.md` for the rationale (reduced-range
// summing is not a well-defined operation without an explicit strategy,
// alpha + sRGB-gamma summing is meaningless, and palette indices do not
// add).
//
// Float impls assume the conventional `[0.0, 1.0]` range and document
// the assumption at the trait level (Philosophy §8 — surface, don't decide).

// ── Mono8 (u8, range 0..=255) ──────────────────────────────────────────────
impl IntegralPixel<Mono32> for Mono8 {
    #[inline]
    fn to_integral(self) -> Mono32 {
        Mono32::new(self.value() as u32)
    }
    #[inline]
    fn max_integral_value() -> Mono32 {
        Mono32::new(u8::MAX as u32)
    }
}
impl IntegralPixel<Mono64> for Mono8 {
    #[inline]
    fn to_integral(self) -> Mono64 {
        Mono64::new(self.value() as u64)
    }
    #[inline]
    fn max_integral_value() -> Mono64 {
        Mono64::new(u8::MAX as u64)
    }
}
impl IntegralPixel<MonoF64> for Mono8 {
    #[inline]
    fn to_integral(self) -> MonoF64 {
        MonoF64::new(self.value() as f64)
    }
    #[inline]
    fn max_integral_value() -> MonoF64 {
        MonoF64::new(u8::MAX as f64)
    }
}

// Squared: 255² = 65_025 — too wide for u32 over realistic images, so the
// only impls are Mono64 / MonoF64 (ADR-0032 §8 table).
impl IntegralSquaredPixel<Mono64> for Mono8 {
    #[inline]
    fn to_integral_squared(self) -> Mono64 {
        let v = self.value() as u64;
        Mono64::new(v * v)
    }
    #[inline]
    fn max_integral_squared_value() -> Mono64 {
        let m = u8::MAX as u64;
        Mono64::new(m * m)
    }
}
impl IntegralSquaredPixel<MonoF64> for Mono8 {
    #[inline]
    fn to_integral_squared(self) -> MonoF64 {
        let v = self.value() as f64;
        MonoF64::new(v * v)
    }
    #[inline]
    fn max_integral_squared_value() -> MonoF64 {
        let m = u8::MAX as f64;
        MonoF64::new(m * m)
    }
}

// ── Mono16 (u16, range 0..=65535) ──────────────────────────────────────────
// Mono32 is never safe for 16-bit sources (65535 × 65537 > u32::MAX).
impl IntegralPixel<Mono64> for Mono16 {
    #[inline]
    fn to_integral(self) -> Mono64 {
        Mono64::new(self.value() as u64)
    }
    #[inline]
    fn max_integral_value() -> Mono64 {
        Mono64::new(u16::MAX as u64)
    }
}
impl IntegralPixel<MonoF64> for Mono16 {
    #[inline]
    fn to_integral(self) -> MonoF64 {
        MonoF64::new(self.value() as f64)
    }
    #[inline]
    fn max_integral_value() -> MonoF64 {
        MonoF64::new(u16::MAX as f64)
    }
}

// Squared: 65535² ≈ 4.3 × 10⁹ already saturates u32 at one pixel.
// Mono64 fits any reasonable image. Float-accumulator impl is `MonoF64`
// only — ADR-0032 §8 explicitly rules out non-f64 squared accumulators.
impl IntegralSquaredPixel<MonoF64> for Mono16 {
    #[inline]
    fn to_integral_squared(self) -> MonoF64 {
        let v = self.value() as f64;
        MonoF64::new(v * v)
    }
    #[inline]
    fn max_integral_squared_value() -> MonoF64 {
        let m = u16::MAX as f64;
        MonoF64::new(m * m)
    }
}

// ── Mono32 (u32) ───────────────────────────────────────────────────────────
impl IntegralPixel<Mono64> for Mono32 {
    #[inline]
    fn to_integral(self) -> Mono64 {
        Mono64::new(self.value() as u64)
    }
    #[inline]
    fn max_integral_value() -> Mono64 {
        Mono64::new(u32::MAX as u64)
    }
}
impl IntegralPixel<MonoF64> for Mono32 {
    #[inline]
    fn to_integral(self) -> MonoF64 {
        MonoF64::new(self.value() as f64)
    }
    #[inline]
    fn max_integral_value() -> MonoF64 {
        MonoF64::new(u32::MAX as f64)
    }
}

// Squared: only MonoF64 is offered — a u32 source squared overflows u64
// for very large images, but the more important point is that the
// downstream consumers (variance / NCC) always need float anyway.
impl IntegralSquaredPixel<MonoF64> for Mono32 {
    #[inline]
    fn to_integral_squared(self) -> MonoF64 {
        let v = self.value() as f64;
        MonoF64::new(v * v)
    }
    #[inline]
    fn max_integral_squared_value() -> MonoF64 {
        let m = u32::MAX as f64;
        MonoF64::new(m * m)
    }
}

// ── MonoF32 / MonoF64 (float, conventional [0, 1] range) ──────────────────
// Per ADR-0032 §7: f32 accumulators are never offered — only `MonoF64`.
// The `[0, 1]` convention is documented on the trait; this library does
// not silently rescale data outside that range (Philosophy §8).
impl IntegralPixel<MonoF64> for MonoF32 {
    #[inline]
    fn to_integral(self) -> MonoF64 {
        MonoF64::new(self.value() as f64)
    }
    #[inline]
    fn max_integral_value() -> MonoF64 {
        MonoF64::new(1.0)
    }
}
impl IntegralPixel<MonoF64> for MonoF64 {
    #[inline]
    fn to_integral(self) -> MonoF64 {
        self
    }
    #[inline]
    fn max_integral_value() -> MonoF64 {
        MonoF64::new(1.0)
    }
}

impl IntegralSquaredPixel<MonoF64> for MonoF32 {
    #[inline]
    fn to_integral_squared(self) -> MonoF64 {
        let v = self.value() as f64;
        MonoF64::new(v * v)
    }
    #[inline]
    fn max_integral_squared_value() -> MonoF64 {
        MonoF64::new(1.0)
    }
}
impl IntegralSquaredPixel<MonoF64> for MonoF64 {
    #[inline]
    fn to_integral_squared(self) -> MonoF64 {
        MonoF64::new(self.value() * self.value())
    }
    #[inline]
    fn max_integral_squared_value() -> MonoF64 {
        MonoF64::new(1.0)
    }
}

#[cfg(test)]
mod integral_tests {
    use super::*;

    #[test]
    fn mono8_to_mono32_to_integral_and_max() {
        assert_eq!(
            <Mono8 as IntegralPixel<Mono32>>::to_integral(Mono8::new(200)),
            Mono32::new(200)
        );
        assert_eq!(
            <Mono8 as IntegralPixel<Mono32>>::max_integral_value(),
            Mono32::new(255)
        );
    }

    #[test]
    fn mono8_to_mono64_max() {
        assert_eq!(
            <Mono8 as IntegralPixel<Mono64>>::max_integral_value(),
            Mono64::new(255)
        );
    }

    #[test]
    fn mono8_squared_mono64() {
        assert_eq!(
            <Mono8 as IntegralSquaredPixel<Mono64>>::to_integral_squared(Mono8::new(16)),
            Mono64::new(256)
        );
        assert_eq!(
            <Mono8 as IntegralSquaredPixel<Mono64>>::max_integral_squared_value(),
            Mono64::new(255 * 255)
        );
    }

    #[test]
    fn mono16_to_mono64_max() {
        assert_eq!(
            <Mono16 as IntegralPixel<Mono64>>::max_integral_value(),
            Mono64::new(u16::MAX as u64)
        );
    }

    #[test]
    fn mono16_squared_monof64_max() {
        let expected = (u16::MAX as f64) * (u16::MAX as f64);
        assert_eq!(
            <Mono16 as IntegralSquaredPixel<MonoF64>>::max_integral_squared_value(),
            MonoF64::new(expected)
        );
    }

    #[test]
    fn mono32_to_monof64_roundtrip() {
        let v = Mono32::new(123_456_789);
        let acc = <Mono32 as IntegralPixel<MonoF64>>::to_integral(v);
        assert_eq!(acc, MonoF64::new(123_456_789.0));
    }

    #[test]
    fn monof32_to_monof64_max_is_one() {
        assert_eq!(
            <MonoF32 as IntegralPixel<MonoF64>>::max_integral_value(),
            MonoF64::new(1.0)
        );
        assert_eq!(
            <MonoF32 as IntegralSquaredPixel<MonoF64>>::max_integral_squared_value(),
            MonoF64::new(1.0)
        );
    }

    #[test]
    fn monof64_self_identity_to_integral() {
        let v = MonoF64::new(0.42);
        assert_eq!(
            <MonoF64 as IntegralPixel<MonoF64>>::to_integral(v),
            MonoF64::new(0.42)
        );
    }
}

// ── Accumulator Add / Sub regression tests ──────────────────────────
//
// `Add` / `Sub` on these pixel types are emitted by the `LinearPixel`
// derive macro (`fovea-derive/src/linear_pixel.rs`). These tests
// pin the per-channel saturating behaviour the summed-area-table
// engine in `crate::analyze::integral` relies on — specifically the
// `(a - c) - (d - b)` evaluation order argument in ADR-0032 §5 and
// the no-overflow guarantee the pre-flight check (ADR-0032 §3) gives
// the inner-loop recurrence.
#[cfg(test)]
mod accumulator_arith_tests {
    use super::*;

    #[test]
    fn mono32_add_basic() {
        let a = Mono32::new(1_000_000);
        let b = Mono32::new(2_500_000);
        assert_eq!(a + b, Mono32::new(3_500_000));
    }

    #[test]
    fn mono32_sub_basic() {
        let a = Mono32::new(5_000_000);
        let b = Mono32::new(1_500_000);
        assert_eq!(a - b, Mono32::new(3_500_000));
    }

    #[test]
    fn mono32_add_saturates_at_u32_max() {
        // Property: Saturating<u32>::Add saturates at u32::MAX.
        let a = Mono32::new(u32::MAX - 1);
        let b = Mono32::new(10);
        assert_eq!(a + b, Mono32::new(u32::MAX));
    }

    #[test]
    fn mono32_sub_saturates_at_zero() {
        // Property: Saturating<u32>::Sub clamps to 0 on underflow.
        let a = Mono32::new(5);
        let b = Mono32::new(100);
        assert_eq!(a - b, Mono32::new(0));
    }

    #[test]
    fn mono32_add_sub_roundtrip() {
        let a = Mono32::new(123_456);
        let b = Mono32::new(7_890);
        assert_eq!((a + b) - b, a);
    }

    #[test]
    fn mono64_add_basic() {
        let a = Mono64::new(10_000_000_000);
        let b = Mono64::new(25_000_000_000);
        assert_eq!(a + b, Mono64::new(35_000_000_000));
    }

    #[test]
    fn mono64_sub_basic() {
        let a = Mono64::new(50_000_000_000);
        let b = Mono64::new(15_000_000_000);
        assert_eq!(a - b, Mono64::new(35_000_000_000));
    }

    #[test]
    fn mono64_add_saturates_at_u64_max() {
        let a = Mono64::new(u64::MAX - 1);
        let b = Mono64::new(10);
        assert_eq!(a + b, Mono64::new(u64::MAX));
    }

    #[test]
    fn mono64_sub_saturates_at_zero() {
        let a = Mono64::new(5);
        let b = Mono64::new(100);
        assert_eq!(a - b, Mono64::new(0));
    }

    #[test]
    fn mono64_add_sub_roundtrip() {
        let a = Mono64::new(999_999_999_999);
        let b = Mono64::new(123_456_789);
        assert_eq!((a + b) - b, a);
    }
}

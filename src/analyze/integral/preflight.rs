//! Pre-flight overflow check for the integral-image engine.
//!
//! This module implements the **O(1), zero-per-pixel-cost** correctness
//! gate from ADR-0032 §3. The idea: before the engine touches any pixel
//! data, verify that the worst-case sum
//!
//! ```text
//! W × H × max_integral_value()
//! ```
//!
//! fits in the accumulator type. If it does, the inner loop is
//! **guaranteed** overflow-free with no per-pixel branch; if it does
//! not, the engine returns [`Error::AccumulatorOverflow`] up front and
//! the caller picks a wider accumulator.
//!
//! The capacity arithmetic is done in `u128` to sidestep overflow in
//! the check itself. For float accumulators the "capacity" is the
//! exact-integer range of `f64` (`2^53`) — beyond that, integer sums
//! lose precision, which is what the check is trying to prevent.
//!
//! See ADR-0032 §3 for the design rationale and ADR-0025 for why this
//! is a Tier-2 (`Result`) failure rather than a Tier-3 panic — the
//! caller's choice of accumulator may be data-dependent.

use crate::Error;
use crate::pixel::{
    IntegralPixel, IntegralSquaredPixel, Mono32, Mono64, MonoF64, Rgb32, Rgb64, RgbF64,
};

// ── Internal capacity traits ──────────────────────────────────────────
//
// These traits stay crate-private: they are an implementation detail of
// the pre-flight check, not part of the public surface. Promoting them
// to `pub` would invite users to add custom accumulators that bypass
// the check, undermining ADR-0032 §3's guarantee. Philosophy §3 — bind
// on the tightest trait that admits the operation, and nothing more.

/// Per-channel capacity of an accumulator pixel, in `u128` units, for
/// the *non-squared* integral check.
///
/// The pre-flight verdict is "per-channel max × W × H ≤ per-channel
/// capacity". Because accumulation is per-channel (ADR-0032 §2 / RGB
/// row of the impl table), the worst-case bound is independent across
/// channels and a single per-channel comparison suffices.
pub(super) trait IntegralCapacity: Copy {
    /// The capacity of one channel of this accumulator, expressed as
    /// `u128`. For integer pixels this is the underlying scalar's
    /// `MAX`; for `f64`-backed pixels it is `2^53` (the largest
    /// integer that round-trips through `f64` losslessly).
    fn channel_capacity_u128() -> u128;

    /// The per-pixel max produced by `IntegralPixel::to_integral`,
    /// reduced to a single `u128` representing the **largest**
    /// per-channel value. The pre-flight multiplication uses this.
    fn pixel_to_per_channel_u128(value: Self) -> u128;
}

/// Per-channel capacity of an accumulator pixel for the *squared*
/// integral check. Same shape as [`IntegralCapacity`], applied to
/// `IntegralSquaredPixel::max_integral_squared_value()`.
pub(super) trait IntegralSquaredCapacity: Copy {
    fn channel_capacity_u128() -> u128;
    fn pixel_to_per_channel_u128(value: Self) -> u128;
}

// ── Capacity constants ────────────────────────────────────────────────

/// `2^53` — the largest integer exactly representable in `f64`.
const F64_INTEGER_CAPACITY: u128 = 1u128 << 53;

// ── Mono accumulators ─────────────────────────────────────────────────

impl IntegralCapacity for Mono32 {
    #[inline]
    fn channel_capacity_u128() -> u128 {
        u32::MAX as u128
    }
    #[inline]
    fn pixel_to_per_channel_u128(value: Self) -> u128 {
        value.value() as u128
    }
}

impl IntegralCapacity for Mono64 {
    #[inline]
    fn channel_capacity_u128() -> u128 {
        u64::MAX as u128
    }
    #[inline]
    fn pixel_to_per_channel_u128(value: Self) -> u128 {
        value.value() as u128
    }
}

impl IntegralCapacity for MonoF64 {
    #[inline]
    fn channel_capacity_u128() -> u128 {
        F64_INTEGER_CAPACITY
    }
    #[inline]
    fn pixel_to_per_channel_u128(value: Self) -> u128 {
        // Float per-pixel max is always non-negative in our use
        // (max_integral_value is a bound, not a signed sample).
        // Floor to u128; values larger than the f64 integer-exact
        // range would already fail the comparison below.
        value.value().max(0.0) as u128
    }
}

impl IntegralSquaredCapacity for Mono64 {
    #[inline]
    fn channel_capacity_u128() -> u128 {
        u64::MAX as u128
    }
    #[inline]
    fn pixel_to_per_channel_u128(value: Self) -> u128 {
        value.value() as u128
    }
}

impl IntegralSquaredCapacity for MonoF64 {
    #[inline]
    fn channel_capacity_u128() -> u128 {
        F64_INTEGER_CAPACITY
    }
    #[inline]
    fn pixel_to_per_channel_u128(value: Self) -> u128 {
        value.value().max(0.0) as u128
    }
}

// ── RGB accumulators ──────────────────────────────────────────────────
//
// Per-channel maximum across `r`, `g`, `b` — the comparison is "the
// worst channel's worst sum fits". Because every RGB IntegralPixel impl
// reports the same value on every channel, the max-channel reduction is
// trivial, but spelling it out keeps the check robust if a future
// hetero-channel accumulator appears.

impl IntegralCapacity for Rgb32 {
    #[inline]
    fn channel_capacity_u128() -> u128 {
        u32::MAX as u128
    }
    #[inline]
    fn pixel_to_per_channel_u128(value: Self) -> u128 {
        let r = value.r.0 as u128;
        let g = value.g.0 as u128;
        let b = value.b.0 as u128;
        r.max(g).max(b)
    }
}

impl IntegralCapacity for Rgb64 {
    #[inline]
    fn channel_capacity_u128() -> u128 {
        u64::MAX as u128
    }
    #[inline]
    fn pixel_to_per_channel_u128(value: Self) -> u128 {
        let r = value.r.0 as u128;
        let g = value.g.0 as u128;
        let b = value.b.0 as u128;
        r.max(g).max(b)
    }
}

impl IntegralCapacity for RgbF64 {
    #[inline]
    fn channel_capacity_u128() -> u128 {
        F64_INTEGER_CAPACITY
    }
    #[inline]
    fn pixel_to_per_channel_u128(value: Self) -> u128 {
        let r = value.r.max(0.0) as u128;
        let g = value.g.max(0.0) as u128;
        let b = value.b.max(0.0) as u128;
        r.max(g).max(b)
    }
}

impl IntegralSquaredCapacity for Rgb64 {
    #[inline]
    fn channel_capacity_u128() -> u128 {
        u64::MAX as u128
    }
    #[inline]
    fn pixel_to_per_channel_u128(value: Self) -> u128 {
        let r = value.r.0 as u128;
        let g = value.g.0 as u128;
        let b = value.b.0 as u128;
        r.max(g).max(b)
    }
}

impl IntegralSquaredCapacity for RgbF64 {
    #[inline]
    fn channel_capacity_u128() -> u128 {
        F64_INTEGER_CAPACITY
    }
    #[inline]
    fn pixel_to_per_channel_u128(value: Self) -> u128 {
        let r = value.r.max(0.0) as u128;
        let g = value.g.max(0.0) as u128;
        let b = value.b.max(0.0) as u128;
        r.max(g).max(b)
    }
}

// ── Pre-flight functions ──────────────────────────────────────────────

/// Verify that an integral image of a `width × height` source whose
/// pixel type implements `IntegralPixel<A>` cannot overflow the
/// accumulator `A`.
///
/// Returns `Ok(())` if the worst-case per-channel sum fits, or
/// `Err(Error::AccumulatorOverflow { .. })` with the required and
/// available capacities (both in `u128`).
#[inline]
pub(super) fn check<P, A>(width: usize, height: usize) -> Result<(), Error>
where
    P: IntegralPixel<A>,
    A: Copy + IntegralCapacity,
{
    let per_channel = A::pixel_to_per_channel_u128(P::max_integral_value());
    let cap = A::channel_capacity_u128();
    verify(per_channel, width, height, cap)
}

/// Squared-variant pre-flight check. Mirrors [`check`] but uses
/// `IntegralSquaredPixel::max_integral_squared_value` and the
/// [`IntegralSquaredCapacity`] trait.
#[inline]
pub(super) fn check_squared<P, A>(width: usize, height: usize) -> Result<(), Error>
where
    P: IntegralSquaredPixel<A>,
    A: Copy + IntegralSquaredCapacity,
{
    let per_channel = A::pixel_to_per_channel_u128(P::max_integral_squared_value());
    let cap = A::channel_capacity_u128();
    verify(per_channel, width, height, cap)
}

/// Shared verification body — the only place the `u128` arithmetic
/// happens. Extracting it keeps the squared / non-squared paths
/// byte-identical.
#[inline]
fn verify(per_channel: u128, width: usize, height: usize, cap: u128) -> Result<(), Error> {
    let required = per_channel
        .checked_mul(width as u128)
        .and_then(|v| v.checked_mul(height as u128))
        .ok_or(Error::AccumulatorOverflow {
            // The multiplication itself overflowed u128 — flag the
            // maximum representable capacity as the "required" lower
            // bound so the caller still gets actionable information.
            required_capacity: u128::MAX,
            accumulator_capacity: cap,
        })?;
    if required <= cap {
        Ok(())
    } else {
        Err(Error::AccumulatorOverflow {
            required_capacity: required,
            accumulator_capacity: cap,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pixel::{Mono8, Mono16, MonoF32, Rgb8};

    #[test]
    fn mono8_to_mono32_passes_for_small_image() {
        // 255 × 1024 × 1024 = 267_386_880 ≤ u32::MAX
        assert!(check::<Mono8, Mono32>(1024, 1024).is_ok());
    }

    #[test]
    fn mono8_to_mono32_fails_at_4k_squared() {
        // 255 × W × H must exceed u32::MAX = 4_294_967_295.
        // 255 × 5000 × 5000 = 6_375_000_000 — comfortably above.
        let w = 5000;
        let h = 5000;
        let err = check::<Mono8, Mono32>(w, h).unwrap_err();
        let Error::AccumulatorOverflow {
            required_capacity,
            accumulator_capacity,
        } = err
        else {
            panic!("expected AccumulatorOverflow, got {:?}", err);
        };
        assert_eq!(
            required_capacity,
            255u128 * (w as u128) * (h as u128),
            "required_capacity should be 255 × W × H",
        );
        assert_eq!(accumulator_capacity, u32::MAX as u128);
    }

    #[test]
    fn mono8_to_mono64_passes_at_huge_size() {
        // u8::MAX × usize::MAX² will certainly fit in u64 for any
        // realistic image size.
        assert!(check::<Mono8, Mono64>(1_000_000, 1_000_000).is_ok());
    }

    #[test]
    fn mono16_to_mono64_passes() {
        assert!(check::<Mono16, Mono64>(65_536, 65_536).is_ok());
    }

    #[test]
    fn monof32_to_monof64_passes_for_megapixel_image() {
        // max_integral_value = 1.0; 1.0 × 1024 × 1024 = ~1e6, well under 2^53.
        assert!(check::<MonoF32, MonoF64>(1024, 1024).is_ok());
    }

    #[test]
    fn rgb8_to_rgb32_passes_for_small_image() {
        // Per-channel check: 255 × 1024 × 1024 ≤ u32::MAX.
        assert!(check::<Rgb8, Rgb32>(1024, 1024).is_ok());
    }

    #[test]
    fn rgb8_to_rgb32_fails_for_huge_image() {
        let w = 5000;
        let h = 5000;
        let err = check::<Rgb8, Rgb32>(w, h).unwrap_err();
        let Error::AccumulatorOverflow {
            required_capacity,
            accumulator_capacity,
        } = err
        else {
            panic!("expected AccumulatorOverflow, got {:?}", err);
        };
        assert_eq!(required_capacity, 255u128 * (w as u128) * (h as u128));
        assert_eq!(accumulator_capacity, u32::MAX as u128);
    }

    #[test]
    fn check_squared_mono8_to_mono64() {
        // 255² × 1024 × 1024 = 65_025 × 1_048_576 = 68_175_298_560 ≤ u64::MAX
        assert!(check_squared::<Mono8, Mono64>(1024, 1024).is_ok());
    }

    #[test]
    fn check_squared_mono16_to_monof64_passes_for_megapixel_image() {
        // 65535² × 1024 × 1024 ≈ 4.5e15, just under 2^53 ≈ 9e15.
        assert!(check_squared::<Mono16, MonoF64>(1024, 1024).is_ok());
    }

    #[test]
    fn check_squared_mono16_to_monof64_fails_at_large_image() {
        // 65535² × 4096 × 4096 ≈ 7.2e16, > 2^53 (~9e15).
        let err = check_squared::<Mono16, MonoF64>(4096, 4096).unwrap_err();
        let Error::AccumulatorOverflow {
            required_capacity,
            accumulator_capacity,
        } = err
        else {
            panic!("expected AccumulatorOverflow");
        };
        assert!(required_capacity > accumulator_capacity);
        assert_eq!(accumulator_capacity, F64_INTEGER_CAPACITY);
    }

    #[test]
    fn zero_sized_image_passes() {
        // Degenerate but well-formed.
        assert!(check::<Mono8, Mono32>(0, 0).is_ok());
        assert!(check::<Mono8, Mono32>(1024, 0).is_ok());
        assert!(check::<Mono8, Mono32>(0, 1024).is_ok());
    }
}

//! Palette-indexed pixel type.
//!
//! See [`Indexed8`] — a palette index, NOT a color value. The meaning depends
//! entirely on an external lookup table (see `Depalettize`).

use fovea_derive::{HomogeneousPixel, PlainPixel, WhiteChannel, ZeroablePixel};

use crate::pixel::impl_origin_invariant_pixel;

// ═══════════════════════════════════════════════════════════════════════════════
// Indexed (palette) pixel type
//
// A palette index, NOT a color value.  The meaning depends entirely on an
// external lookup table (see `Depalettize`).
// These types intentionally do NOT implement `LinearPixel` or `LinearSpace`,
// because interpolating indices is mathematically meaningless.  The compiler
// rejects attempts to use `Indexed8` with `Bilinear` resize.
// `NearestNeighbor` resize compiles and works correctly (copying indices is
// valid).
// ═══════════════════════════════════════════════════════════════════════════════

/// A palette-indexed pixel.
///
/// The value is an index into an external color palette, NOT a color
/// value.  This type implements `PlainPixel` and `ZeroablePixel` but
/// intentionally does **not** implement `LinearPixel` or `LinearSpace`,
/// so the compiler rejects attempts to use it with interpolation
/// algorithms like bilinear resize.
///
/// Convert to a color pixel via [`Depalettize`](crate::transform::Depalettize).
///
/// # Examples
///
/// ```
/// # use fovea::pixel::Indexed8;
/// let idx = Indexed8(42);
/// assert_eq!(idx.0, 42);
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
    WhiteChannel,
)]
pub struct Indexed8(pub u8);

impl From<u8> for Indexed8 {
    #[inline]
    fn from(v: u8) -> Self {
        Indexed8(v)
    }
}

impl From<Indexed8> for u8 {
    #[inline]
    fn from(p: Indexed8) -> Self {
        p.0
    }
}

// ---------------------------------------------------------------------------
// OriginInvariantPixel impl
// ---------------------------------------------------------------------------
//
// A palette index resolves through an external lookup table; that mapping is
// the same regardless of where the pixel sits, so an origin-translated crop
// preserves its meaning. (Interpolating indices remains a type error, since
// `Indexed8` withholds `LinearSpace` — a separate axis from origin-invariance.)
impl_origin_invariant_pixel!(Indexed8);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_u8_wraps_value() {
        assert_eq!(Indexed8::from(42), Indexed8(42));
    }

    #[test]
    fn into_u8_unwraps_value() {
        let v: u8 = Indexed8(199).into();
        assert_eq!(v, 199);
    }

    #[test]
    fn from_into_round_trip() {
        for v in [0u8, 1, 42, 127, 200, 255] {
            let p: Indexed8 = v.into();
            let back: u8 = p.into();
            assert_eq!(back, v);
        }
    }
}

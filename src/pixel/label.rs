//! Connected-component label pixel types.
//!
//! See [`Label32`] and the [`LabelPixel`](crate::pixel::LabelPixel)
//! trait. Label pixels name *which* connected blob a foreground pixel
//! belongs to; they are deliberately not intensities and not arithmetic.
//!
//! See [ADR-0047] and the consumer module
//! [`crate::analyze::components`].
//!
//! [ADR-0047]: https://github.com/karhunen-loeve/fovea/blob/main/docs/adr/0047-connected-components-design.md

use std::num::Saturating;

use fovea_derive::{HomogeneousPixel, PlainPixel, ZeroablePixel};

use crate::pixel::LabelPixel;

/// A 32-bit foreground component label.
///
/// `Label32` is the v1 concrete [`LabelPixel`] type produced by
/// [`connected_components`](crate::analyze::components::connected_components).
/// Background pixels carry the value [`Label32::BACKGROUND`] (== `0`);
/// foreground pixels carry a label in `1 ..= u32::MAX`.
///
/// # Trait surface
///
/// Implements: [`Copy`], [`Clone`], [`Debug`](core::fmt::Debug),
/// [`PartialEq`], [`Eq`], [`Hash`](core::hash::Hash),
/// [`Ord`], [`PartialOrd`],
/// [`PlainPixel`](crate::pixel::PlainPixel),
/// [`HomogeneousPixel`](crate::pixel::HomogeneousPixel),
/// [`ZeroablePixel`], and [`LabelPixel`].
///
/// Deliberately does **not** implement
/// [`LinearPixel`](crate::pixel::LinearPixel),
/// [`LinearChannel`](crate::pixel::LinearChannel),
/// [`LinearSpace`](crate::pixel::LinearSpace),
/// [`BoundedChannel`](crate::pixel::BoundedChannel),
/// [`WhiteChannel`](crate::pixel::WhiteChannel),
/// [`FromLinear`](crate::pixel::FromLinear), or any arithmetic
/// operator. Averaging two labels, gamma-converting them, thresholding
/// them, inverting them, or adding them is meaningless; excluding those
/// traits makes such operations *fail to compile* on label images
/// (Philosophy §1).
///
/// # Layout
///
/// `#[repr(transparent)]` over `Saturating<u32>`. Byte layout is exactly
/// four bytes, native endian, identical to a raw `u32`. The `Saturating`
/// wrapper exists for `PlainChannel` ergonomics (see the codebase
/// convention used by `Mono*` and `Rgb*` pixels); `Label32` never relies
/// on its saturating arithmetic, since the label set is not closed
/// under arithmetic in the first place.
///
/// # Examples
///
/// ```
/// use fovea::pixel::{Label32, LabelPixel, ZeroablePixel};
///
/// // Round-trip through the label-index API.
/// let l = Label32::from_label_index(7).unwrap();
/// assert_eq!(l.value(), 7);
/// assert_eq!(l.to_label_index(), 7);
///
/// // Background and capacity.
/// assert_eq!(Label32::BACKGROUND.value(), 0);
/// assert_eq!(<Label32 as ZeroablePixel>::zero(), Label32::BACKGROUND);
/// assert_eq!(<Label32 as LabelPixel>::MAX_LABEL, u32::MAX as u64);
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
)]
pub struct Label32(Saturating<u32>);

impl Label32 {
    /// The background label \u2014 the value the labeling engine writes
    /// to every non-foreground pixel. Equal to `<Label32 as
    /// ZeroablePixel>::zero()`.
    pub const BACKGROUND: Self = Label32(Saturating(0));

    /// Construct a label from a raw `u32`. `Label32::new(0)` is
    /// [`BACKGROUND`](Self::BACKGROUND).
    #[inline]
    pub const fn new(value: u32) -> Self {
        Label32(Saturating(value))
    }

    /// Read the underlying label as a `u32`.
    #[inline]
    pub const fn value(self) -> u32 {
        self.0.0
    }
}

impl LabelPixel for Label32 {
    const MAX_LABEL: u64 = u32::MAX as u64;

    #[inline]
    fn from_label_index(index: u64) -> Option<Self> {
        if index == 0 || index > Self::MAX_LABEL {
            None
        } else {
            // Cast is exact: `0 < index <= u32::MAX`.
            Some(Label32::new(index as u32))
        }
    }

    #[inline]
    fn to_label_index(self) -> u64 {
        self.value() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pixel::{HomogeneousPixel, PlainChannel, PlainPixel, ZeroablePixel};

    #[test]
    fn background_is_zero() {
        assert_eq!(Label32::BACKGROUND.value(), 0);
        assert_eq!(<Label32 as ZeroablePixel>::zero(), Label32::BACKGROUND);
    }

    #[test]
    fn new_and_value_round_trip() {
        for &v in &[0u32, 1, 2, 42, 1_000_000, u32::MAX] {
            assert_eq!(Label32::new(v).value(), v);
        }
    }

    #[test]
    fn label_pixel_round_trip() {
        for &i in &[1u64, 2, 42, 1_000_000, u32::MAX as u64] {
            let l = Label32::from_label_index(i).expect("in range");
            assert_eq!(l.to_label_index(), i);
        }
    }

    #[test]
    fn from_label_index_rejects_zero_and_overflow() {
        assert_eq!(Label32::from_label_index(0), None);
        assert_eq!(Label32::from_label_index((u32::MAX as u64) + 1), None);
        assert_eq!(Label32::from_label_index(u64::MAX), None);
    }

    #[test]
    fn from_label_index_accepts_boundary() {
        assert!(Label32::from_label_index(1).is_some());
        assert!(Label32::from_label_index(u32::MAX as u64).is_some());
    }

    #[test]
    fn to_label_index_of_zero_is_zero() {
        assert_eq!(Label32::BACKGROUND.to_label_index(), 0);
    }

    #[test]
    fn ordering_matches_inner_value() {
        let a = Label32::new(3);
        let b = Label32::new(10);
        assert!(a < b);
        assert!(b > a);
    }

    #[test]
    fn plain_pixel_layout() {
        assert_eq!(<Label32 as PlainChannel>::SIZE, 4);
        assert_eq!(<Label32 as PlainPixel>::CHANNELS, &[4]);
        assert_eq!(<Label32 as HomogeneousPixel>::CHANNEL_COUNT, 1);
    }

    #[test]
    fn plain_pixel_byte_round_trip() {
        let l = Label32::new(0xDEAD_BEEF);
        let bytes = <Label32 as PlainChannel>::as_bytes(&l);
        let back = <Label32 as PlainChannel>::from_bytes(bytes).unwrap();
        assert_eq!(l, back);
    }

    #[test]
    fn max_label_is_u32_max() {
        assert_eq!(<Label32 as LabelPixel>::MAX_LABEL, u32::MAX as u64);
    }

    #[test]
    fn hashable() {
        use std::collections::HashSet;
        let mut s = HashSet::new();
        s.insert(Label32::new(1));
        s.insert(Label32::new(2));
        s.insert(Label32::new(1));
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn debug_format_contains_value() {
        let s = format!("{:?}", Label32::new(7));
        assert!(s.contains('7'), "debug output: {}", s);
    }
}

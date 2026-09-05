//! Compile-time connectivity strategies for connected-component
//! labeling.
//!
//! See [`Connectivity4`] and [`Connectivity8`]. The trait is sealed;
//! only the two shipped marker types may implement it. Adding new
//! connectivities (e.g. 6-connected on a hex grid, knight's-move)
//! is intentionally out of scope.

use crate::Offset;

mod sealed {
    pub trait Sealed {}
}

/// A compile-time connectivity strategy for connected-component
/// labeling.
///
/// Sealed: only [`Connectivity4`] and [`Connectivity8`] implement this
/// trait. The trait is consumed as a type parameter, not a value \u2014
/// the engine never instantiates a `Connectivity`.
///
/// Implementors declare `OFFSETS`, the [`Offset`]s of the
/// already-visited neighbours that the labeling pass should examine
/// when classifying each pixel. All offsets are *raster-scan-preceding*:
/// `dy < 0`, or `dy == 0 && dx < 0`. This invariant lets pass 1 collect
/// each pixel's neighbour labels from the in-progress provisional
/// label buffer (rather than the final output) without requiring
/// random access to the future.
pub trait Connectivity: sealed::Sealed + Copy {
    /// Number of already-visited neighbours examined per pixel
    /// (2 for 4-connectivity: N, W; 4 for 8-connectivity: NW, N, NE, W).
    const NEIGHBOURS: usize;

    /// Offsets of those neighbours. All have `dy <= 0`, and when
    /// `dy == 0` then `dx < 0`: raster-scan-preceding only.
    const OFFSETS: &'static [Offset];

    /// The connectivity the *background* must be labeled with when the
    /// foreground is labeled with `Self`.
    ///
    /// 8-connected foreground pairs with 4-connected background and vice
    /// versa. Using the same connectivity for both sides breaks the
    /// discrete Jordan curve property: a diagonal foreground line would
    /// either connect across itself *and* leave the background it
    /// separates connected, or neither. Contour extraction relies on the
    /// duality to classify holes.
    type Dual: Connectivity;
}

/// 4-connectivity: a pixel is connected to its N and W already-visited
/// neighbours (and, transitively, its S and E neighbours when those
/// pixels are processed).
#[derive(Clone, Copy, Debug)]
pub struct Connectivity4;

/// 8-connectivity: a pixel is connected to its NW, N, NE, and W
/// already-visited neighbours (and, transitively, the four
/// later-visited diagonals).
#[derive(Clone, Copy, Debug)]
pub struct Connectivity8;

impl sealed::Sealed for Connectivity4 {}
impl sealed::Sealed for Connectivity8 {}

impl Connectivity for Connectivity4 {
    const NEIGHBOURS: usize = 2;
    // W, N \u2014 the two raster-preceding orthogonal neighbours.
    const OFFSETS: &'static [Offset] = &[Offset::new(-1, 0), Offset::new(0, -1)];
    type Dual = Connectivity8;
}

impl Connectivity for Connectivity8 {
    const NEIGHBOURS: usize = 4;
    // NW, N, NE, W \u2014 the four raster-preceding neighbours, top row
    // left-to-right then current-row west.
    const OFFSETS: &'static [Offset] = &[
        Offset::new(-1, -1),
        Offset::new(0, -1),
        Offset::new(1, -1),
        Offset::new(-1, 0),
    ];
    type Dual = Connectivity4;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn connectivity4_offsets_are_w_n_only() {
        assert_eq!(Connectivity4::NEIGHBOURS, 2);
        assert_eq!(
            Connectivity4::OFFSETS,
            &[Offset::new(-1, 0), Offset::new(0, -1)]
        );
    }

    #[test]
    fn connectivity8_offsets_are_nw_n_ne_w() {
        assert_eq!(Connectivity8::NEIGHBOURS, 4);
        assert_eq!(
            Connectivity8::OFFSETS,
            &[
                Offset::new(-1, -1),
                Offset::new(0, -1),
                Offset::new(1, -1),
                Offset::new(-1, 0),
            ]
        );
    }

    #[test]
    fn all_offsets_are_raster_preceding() {
        for offsets in [Connectivity4::OFFSETS, Connectivity8::OFFSETS] {
            for &off in offsets {
                assert!(
                    off.dy < 0 || (off.dy == 0 && off.dx < 0),
                    "offset {off:?} is not raster-scan-preceding"
                );
            }
        }
    }

    #[test]
    fn neighbours_matches_offsets_length() {
        assert_eq!(Connectivity4::NEIGHBOURS, Connectivity4::OFFSETS.len());
        assert_eq!(Connectivity8::NEIGHBOURS, Connectivity8::OFFSETS.len());
    }
}

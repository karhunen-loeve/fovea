//! Freeman chain-code encoding of traced contours.
//!
//! A [`ChainCode`] stores a contour as its start pixel plus one
//! [`ChainDirection`] per step — one byte instead of two `usize`s per
//! point. Conversion from a traced [`Contour`](super::Contour) is total
//! (consecutive traced pixels are always 8-adjacent), and
//! [`ChainCode::to_points`] round-trips exactly.

use crate::Coordinate;

use super::Contour;

/// One step of an 8-connected chain code, in image coordinates.
///
/// The numbering follows Freeman's convention (`East = 0`, ascending
/// counterclockwise in mathematical orientation). Because image
/// coordinates grow **downward** in y, "North" here means `y − 1` —
/// visually up on screen.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ChainDirection {
    /// `(+1, 0)`.
    East = 0,
    /// `(+1, −1)`.
    NorthEast = 1,
    /// `(0, −1)`.
    North = 2,
    /// `(−1, −1)`.
    NorthWest = 3,
    /// `(−1, 0)`.
    West = 4,
    /// `(−1, +1)`.
    SouthWest = 5,
    /// `(0, +1)`.
    South = 6,
    /// `(+1, +1)`.
    SouthEast = 7,
}

impl ChainDirection {
    /// All eight directions, in Freeman code order (`East` first).
    pub const ALL: [Self; 8] = [
        Self::East,
        Self::NorthEast,
        Self::North,
        Self::NorthWest,
        Self::West,
        Self::SouthWest,
        Self::South,
        Self::SouthEast,
    ];

    /// The `(dx, dy)` step this direction takes, y growing downward.
    #[must_use]
    pub const fn offset(self) -> (i32, i32) {
        match self {
            Self::East => (1, 0),
            Self::NorthEast => (1, -1),
            Self::North => (0, -1),
            Self::NorthWest => (-1, -1),
            Self::West => (-1, 0),
            Self::SouthWest => (-1, 1),
            Self::South => (0, 1),
            Self::SouthEast => (1, 1),
        }
    }

    /// The direction with the given step, or `None` if `(dx, dy)` is not
    /// one of the eight unit king moves.
    #[must_use]
    pub const fn from_offset(dx: i32, dy: i32) -> Option<Self> {
        match (dx, dy) {
            (1, 0) => Some(Self::East),
            (1, -1) => Some(Self::NorthEast),
            (0, -1) => Some(Self::North),
            (-1, -1) => Some(Self::NorthWest),
            (-1, 0) => Some(Self::West),
            (-1, 1) => Some(Self::SouthWest),
            (0, 1) => Some(Self::South),
            (1, 1) => Some(Self::SouthEast),
            _ => None,
        }
    }
}

/// A contour encoded as a start pixel plus per-step directions.
///
/// The compact alternative to a point list: one byte per border step.
/// The final move closes the loop back to the start pixel, so a contour
/// of *n* points encodes as *n* moves (and a single-pixel contour as
/// zero moves).
///
/// Only traced contours can be encoded — the 8-adjacency of consecutive
/// points that makes the encoding total is a property
/// [`Contour`](super::Contour) certifies by construction. There is no
/// constructor from a raw move list.
///
/// # Examples
///
/// ```
/// use fovea::analyze::contours::{Connectivity8, extract_contours};
/// use fovea::image::{BinaryImage, Image};
/// use fovea::pixel::Label32;
///
/// let img: BinaryImage = Image::generate(4, 4, |x, y| (1..3).contains(&x) && (1..3).contains(&y));
/// let (_, hierarchy) = extract_contours::<Label32, Connectivity8>(&img).unwrap();
/// let contour = hierarchy.components()[0].outer();
///
/// let chain = contour.chain_code();
/// assert_eq!(chain.moves().len(), contour.points().len());
/// assert_eq!(chain.to_points(), contour.points());
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChainCode {
    start: Coordinate,
    moves: Vec<ChainDirection>,
}

impl ChainCode {
    /// Encode a traced contour.
    ///
    /// Total: every consecutive pair of traced points — including the
    /// closing pair last→first — is 8-adjacent by construction.
    #[must_use]
    pub fn from_contour(contour: &Contour) -> Self {
        let points = contour.points();
        let start = points[0];
        let moves = if points.len() < 2 {
            Vec::new()
        } else {
            (0..points.len())
                .map(|i| {
                    let a = points[i];
                    let b = points[(i + 1) % points.len()];
                    let dx = (b.x as i64 - a.x as i64) as i32;
                    let dy = (b.y as i64 - a.y as i64) as i32;
                    ChainDirection::from_offset(dx, dy)
                        .expect("traced contour points are 8-adjacent")
                })
                .collect()
        };
        Self { start, moves }
    }

    /// The pixel the chain starts from.
    #[must_use]
    pub const fn start(&self) -> Coordinate {
        self.start
    }

    /// The per-step directions, ending with the move that closes the loop.
    #[must_use]
    pub fn moves(&self) -> &[ChainDirection] {
        &self.moves
    }

    /// Decode back into the traced point list.
    ///
    /// The final (closing) move's destination is the start pixel and is
    /// not repeated, so the result has exactly as many points as the
    /// contour that was encoded.
    #[must_use]
    pub fn to_points(&self) -> Vec<Coordinate> {
        let mut points = Vec::with_capacity(self.moves.len().max(1));
        let (mut x, mut y) = (self.start.x as i64, self.start.y as i64);
        points.push(self.start);
        for step in self.moves.iter().take(self.moves.len().saturating_sub(1)) {
            let (dx, dy) = step.offset();
            x += dx as i64;
            y += dy as i64;
            debug_assert!(x >= 0 && y >= 0, "chain code left the image quadrant");
            points.push(Coordinate::new(x as usize, y as usize));
        }
        points
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offsets_round_trip() {
        for dir in ChainDirection::ALL {
            let (dx, dy) = dir.offset();
            assert_eq!(ChainDirection::from_offset(dx, dy), Some(dir));
        }
        assert_eq!(ChainDirection::from_offset(0, 0), None);
        assert_eq!(ChainDirection::from_offset(2, 0), None);
        assert_eq!(ChainDirection::from_offset(-1, 2), None);
    }

    #[test]
    fn freeman_codes_ascend_counterclockwise_from_east() {
        // Pin the discriminants — the numbering is the exchange format.
        assert_eq!(ChainDirection::East as u8, 0);
        assert_eq!(ChainDirection::NorthEast as u8, 1);
        assert_eq!(ChainDirection::North as u8, 2);
        assert_eq!(ChainDirection::NorthWest as u8, 3);
        assert_eq!(ChainDirection::West as u8, 4);
        assert_eq!(ChainDirection::SouthWest as u8, 5);
        assert_eq!(ChainDirection::South as u8, 6);
        assert_eq!(ChainDirection::SouthEast as u8, 7);
    }
}

//! Monospaced bitmap fonts for [`Text`](super::Text).

mod misc_fixed_10x20;
mod misc_fixed_6x13;

use core::fmt;

use crate::Size;
use crate::error::{Error, ParameterError, Requirement, Value};

use misc_fixed_6x13::PIXELS_6X13;
use misc_fixed_10x20::PIXELS_10X20;

/// The characters of the built-in fonts, in ascending order: printable
/// ASCII (U+0020 to U+007E), then the printable half of ISO 8859-1
/// (U+00A0 to U+00FF), 191 in all.
static LATIN_1: [char; 191] = latin_1();

const fn latin_1() -> [char; 191] {
    let mut chars = ['\0'; 191];
    let mut i = 0;
    while i < 95 {
        chars[i] = (0x20 + i as u8) as char;
        i += 1;
    }
    while i < 191 {
        chars[i] = (0xA0 + (i - 95) as u8) as char;
        i += 1;
    }
    chars
}

/// The X11 misc-fixed font 6x13: cells of 6 × 13 pixels, for short labels
/// close to the geometry they describe.
///
/// Covers printable ASCII and the printable half of ISO 8859-1, so
/// `12.5 µm`, `45.0°`, `±0.02` and `Länge` render as written. The font is
/// in the public domain.
pub const FONT_6X13: BitmapFont<'static> = BitmapFont::from_parts(
    Size {
        width: 6,
        height: 13,
    },
    &LATIN_1,
    &PIXELS_6X13,
);

/// The X11 misc-fixed font 10x20: cells of 10 × 20 pixels, for labels that
/// must read at a glance.
///
/// Covers the same characters as [`FONT_6X13`]. The font is in the public
/// domain.
pub const FONT_10X20: BitmapFont<'static> = BitmapFont::from_parts(
    Size {
        width: 10,
        height: 20,
    },
    &LATIN_1,
    &PIXELS_10X20,
);

/// A monospaced bitmap font: every character occupies one cell of the same
/// size, drawn from a table of pixel rows.
///
/// The built-in fonts are [`FONT_6X13`] and [`FONT_10X20`]. A font of your
/// own, with other sizes or other characters, comes from
/// [`try_new`](Self::try_new) and draws everywhere the built-in ones do.
///
/// A character the font does not cover is drawn as its `?`, or as an empty
/// cell if the font has no `?` either.
///
/// # Examples
///
/// Measuring a label before placing it:
///
/// ```
/// use fovea::Size;
/// use fovea::draw::FONT_6X13;
///
/// assert_eq!(FONT_6X13.cell_size(), Size::new(6, 13));
/// assert_eq!(FONT_6X13.text_size("12.5 µm", 1), Size::new(42, 13));
/// assert_eq!(FONT_6X13.text_size("12.5 µm\n±0.02", 2), Size::new(84, 52));
/// ```
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct BitmapFont<'a> {
    cell: Size,
    chars: &'a [char],
    pixels: &'a [u8],
}

impl<'a> BitmapFont<'a> {
    /// A font with cells of `cell` pixels covering `chars`, whose glyphs
    /// are `pixels`.
    ///
    /// `chars` lists the covered characters in strictly ascending order.
    /// `pixels` holds, for each of them in the same order, `cell.height`
    /// rows of `cell.width.div_ceil(8)` bytes; the most significant bit of
    /// a row's first byte is its leftmost pixel, and a set bit is ink.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidParameter`] with [`Requirement::AtLeast`] 1 when
    ///   the cell is zero pixels wide or high.
    /// - [`Error::InvalidParameter`] with [`Requirement::StrictlyOrdered`]
    ///   when `chars` is not strictly ascending; the index is that of the
    ///   first character not above its predecessor.
    /// - [`Error::LengthMismatch`] when `pixels` does not hold exactly one
    ///   glyph per character.
    ///
    /// # Examples
    ///
    /// A two-character font with 3 × 3 cells, a plus sign and a box:
    ///
    /// ```
    /// use fovea::Size;
    /// use fovea::draw::BitmapFont;
    ///
    /// let chars = ['+', '□'];
    /// let pixels = [
    ///     0b010_00000, 0b111_00000, 0b010_00000, // '+'
    ///     0b111_00000, 0b101_00000, 0b111_00000, // '□'
    /// ];
    /// let font = BitmapFont::try_new(Size::new(3, 3), &chars, &pixels)?;
    /// assert_eq!(font.text_size("+□+", 1), Size::new(9, 3));
    ///
    /// // A glyph short of a full table is refused.
    /// assert!(BitmapFont::try_new(Size::new(3, 3), &chars, &pixels[..5]).is_err());
    /// # Ok::<(), fovea::Error>(())
    /// ```
    pub fn try_new(cell: Size, chars: &'a [char], pixels: &'a [u8]) -> Result<Self, Error> {
        for (parameter, extent) in [("cell width", cell.width), ("cell height", cell.height)] {
            if extent == 0 {
                return Err(ParameterError::new(
                    parameter,
                    Requirement::AtLeast(1),
                    Value::Usize(extent),
                )
                .into());
            }
        }
        if let Some(i) = (1..chars.len()).find(|&i| chars[i - 1] >= chars[i]) {
            return Err(ParameterError::new(
                "font characters",
                Requirement::StrictlyOrdered,
                Value::NotRecorded,
            )
            .at(i)
            .into());
        }
        let expected = glyph_len(cell)
            .and_then(|glyph| glyph.checked_mul(chars.len()))
            .unwrap_or(usize::MAX);
        if pixels.len() != expected {
            return Err(Error::LengthMismatch {
                expected,
                actual: pixels.len(),
            });
        }
        Ok(Self::from_parts(cell, chars, pixels))
    }

    /// The font from parts whose fit the caller has proven, as the tests
    /// do for the built-in fonts through [`try_new`](Self::try_new).
    const fn from_parts(cell: Size, chars: &'a [char], pixels: &'a [u8]) -> Self {
        Self {
            cell,
            chars,
            pixels,
        }
    }

    /// The size of one character cell, in pixels.
    #[must_use]
    pub const fn cell_size(&self) -> Size {
        self.cell
    }

    /// The size `text` covers when drawn at `scale`: the widest line by the
    /// number of lines, each character one cell scaled by `scale`.
    ///
    /// Lines end at `\n`. Every other character counts as one cell, whether
    /// the font covers it or not. A text without characters is one empty
    /// line: zero pixels wide and one line high. A scale of 0 covers
    /// nothing. A size beyond `usize` saturates.
    #[must_use]
    pub fn text_size(&self, text: &str, scale: u32) -> Size {
        let (columns, lines) = text
            .split('\n')
            .fold((0usize, 0usize), |(widest, lines), line| {
                (widest.max(line.chars().count()), lines + 1)
            });
        let scale = scale as usize;
        Size::new(
            columns
                .saturating_mul(self.cell.width)
                .saturating_mul(scale),
            lines.saturating_mul(self.cell.height).saturating_mul(scale),
        )
    }

    /// The glyph rows of `c`, or of `?` when the font does not cover `c`,
    /// or `None` when it covers neither.
    pub(super) fn glyph(&self, c: char) -> Option<&'a [u8]> {
        let index = match self.chars.binary_search(&c) {
            Ok(index) => index,
            Err(_) => self.chars.binary_search(&'?').ok()?,
        };
        let len = glyph_len(self.cell).expect("a constructed font's glyph length fits usize");
        Some(&self.pixels[index * len..(index + 1) * len])
    }

    /// The number of bytes in one row of a glyph.
    pub(super) const fn bytes_per_row(&self) -> usize {
        self.cell.width.div_ceil(8)
    }
}

/// The number of bytes one glyph of a `cell`-sized font takes, if it fits
/// `usize`.
fn glyph_len(cell: Size) -> Option<usize> {
    cell.width.div_ceil(8).checked_mul(cell.height)
}

impl fmt::Debug for BitmapFont<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BitmapFont")
            .field("cell", &self.cell)
            .field("chars", &self.chars.len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_built_in_fonts_pass_their_own_checks() {
        for font in [FONT_6X13, FONT_10X20] {
            let checked = BitmapFont::try_new(font.cell, font.chars, font.pixels).unwrap();
            assert_eq!(checked, font);
        }
    }

    #[test]
    fn the_built_in_fonts_cover_ascii_and_latin_1() {
        for font in [FONT_6X13, FONT_10X20] {
            for c in ['A', '~', ' ', 'µ', '°', '±', 'ä', 'ß', 'ÿ'] {
                assert!(font.chars.binary_search(&c).is_ok(), "{c:?}");
            }
            for c in ['\u{7F}', '\u{9F}', '⌀', 'Δ'] {
                assert!(font.chars.binary_search(&c).is_err(), "{c:?}");
            }
        }
    }

    #[test]
    fn an_uncovered_character_takes_the_question_mark() {
        let question = FONT_6X13.glyph('?').unwrap();
        assert_eq!(FONT_6X13.glyph('⌀'), Some(question));
        assert_ne!(FONT_6X13.glyph('A'), Some(question));
    }

    #[test]
    fn a_font_without_a_question_mark_has_no_glyph_for_the_uncovered() {
        let chars = ['a'];
        let pixels = [0xFF];
        let font = BitmapFont::try_new(Size::new(8, 1), &chars, &pixels).unwrap();
        assert_eq!(font.glyph('a'), Some(&pixels[..]));
        assert_eq!(font.glyph('b'), None);
    }

    #[test]
    fn try_new_reports_what_does_not_fit() {
        let chars = ['a', 'b'];
        let pixels = [0u8; 4];
        assert_eq!(
            BitmapFont::try_new(Size::new(0, 2), &chars, &pixels).err(),
            Some(Error::from(ParameterError::new(
                "cell width",
                Requirement::AtLeast(1),
                Value::Usize(0),
            )))
        );
        assert_eq!(
            BitmapFont::try_new(Size::new(8, 0), &chars, &pixels).err(),
            Some(Error::from(ParameterError::new(
                "cell height",
                Requirement::AtLeast(1),
                Value::Usize(0),
            )))
        );
        assert_eq!(
            BitmapFont::try_new(Size::new(8, 2), &['a', 'c', 'b'], &[0u8; 6]).err(),
            Some(Error::from(
                ParameterError::new(
                    "font characters",
                    Requirement::StrictlyOrdered,
                    Value::NotRecorded
                )
                .at(2)
            ))
        );
        assert_eq!(
            BitmapFont::try_new(Size::new(8, 2), &['a', 'a'], &pixels).err(),
            Some(Error::from(
                ParameterError::new(
                    "font characters",
                    Requirement::StrictlyOrdered,
                    Value::NotRecorded
                )
                .at(1)
            ))
        );
        assert_eq!(
            BitmapFont::try_new(Size::new(9, 2), &chars, &pixels).err(),
            Some(Error::LengthMismatch {
                expected: 8,
                actual: 4,
            })
        );
        assert_eq!(
            BitmapFont::try_new(Size::new(usize::MAX, usize::MAX), &chars, &pixels).err(),
            Some(Error::LengthMismatch {
                expected: usize::MAX,
                actual: 4,
            })
        );
        assert!(BitmapFont::try_new(Size::new(8, 2), &chars, &pixels).is_ok());
        assert!(BitmapFont::try_new(Size::new(8, 2), &[], &[]).is_ok());
    }

    #[test]
    fn text_size_counts_lines_and_the_widest_one() {
        let font = FONT_6X13;
        assert_eq!(font.text_size("", 1), Size::new(0, 13));
        assert_eq!(font.text_size("ab", 1), Size::new(12, 13));
        assert_eq!(font.text_size("ab\nxyz\n", 1), Size::new(18, 39));
        assert_eq!(font.text_size("ab\nxyz", 3), Size::new(54, 78));
        assert_eq!(font.text_size("ab", 0), Size::new(0, 0));
        // An uncovered character is still one cell.
        assert_eq!(font.text_size("⌀", 1), Size::new(6, 13));
        assert_eq!(
            font.text_size("ab", u32::MAX),
            Size::new(12 * u32::MAX as usize, 13 * u32::MAX as usize)
        );
    }

    #[test]
    fn debug_shows_the_shape_not_the_table() {
        assert_eq!(
            format!("{FONT_6X13:?}"),
            "BitmapFont { cell: Size { width: 6, height: 13 }, chars: 191, .. }"
        );
    }
}

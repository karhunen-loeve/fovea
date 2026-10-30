//! Text in a monospaced bitmap font.

use super::{BitmapFont, Drawable, hspan};
use crate::SignedCoordinate;
use crate::image::ImageViewMut;

/// A text in a [`BitmapFont`], with the top-left corner of its first
/// character at `position`.
///
/// Each character takes one cell of the font, scaled by `scale` in both
/// directions: a set pixel of the glyph becomes a `scale` × `scale` block
/// of `color`. `\n` starts a new line one cell height (times `scale`)
/// further down, at the left edge again. A character the font does not
/// cover is drawn as its `?`.
///
/// With `background` set, every pixel of every cell that is not ink is
/// written with it, so the text reads on any image; with `None` only the
/// ink is written. A `scale` of 0 draws nothing, as a [`Rect`](super::Rect)
/// of zero size does.
///
/// [`BitmapFont::text_size`] tells the size the text covers before it is
/// drawn, which is what placing a label next to the geometry it describes
/// needs.
///
/// # Examples
///
/// ```
/// use fovea::draw::{Drawable, FONT_6X13, Text};
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::Mono8;
///
/// let mut image: Image<Mono8> = Image::zero(64, 32);
/// let label = Text {
///     content: "12.5 µm\n±0.02",
///     position: (2, 2).into(),
///     color: Mono8::new(255),
///     background: Some(Mono8::new(40)),
///     scale: 1,
///     font: &FONT_6X13,
/// };
/// label.draw_into(&mut image);
///
/// // Two lines of 13 pixels below the corner, in cells painted with the
/// // background where they are not ink.
/// assert_eq!(image.pixel_at(2, 2), Mono8::new(40));
/// assert_eq!(image.pixel_at(2, 2 + 26), Mono8::new(0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Text<'a, P> {
    /// The text. `\n` breaks lines; every other character takes one cell.
    pub content: &'a str,
    /// Top-left corner of the first character's cell.
    pub position: SignedCoordinate,
    /// Pixel value written for the ink of each glyph.
    pub color: P,
    /// Pixel value written for the rest of each cell, or `None` to leave
    /// it as it is.
    pub background: Option<P>,
    /// Integer magnification: 1 draws the font at its own size, 2 at twice
    /// its size, 0 draws nothing.
    pub scale: u32,
    /// The font the text is drawn in.
    pub font: &'a BitmapFont<'a>,
}

impl<P: Copy> Drawable<P> for Text<'_, P> {
    fn draw_into(&self, image: &mut impl ImageViewMut<Pixel = P>) {
        if self.scale == 0 {
            return;
        }
        let size = image.size();
        let (width, height) = (size.width as i128, size.height as i128);
        let cell = self.font.cell_size();
        let scale = i128::from(self.scale);
        // A usize extent times a u32 scale fits i128, and every position
        // below stays within one such step of the image, so nothing
        // overflows.
        let advance = cell.width as i128 * scale;
        let line_height = cell.height as i128 * scale;

        let mut top = self.position.y as i128;
        for line in self.content.split('\n') {
            if top >= height {
                break;
            }
            if top + line_height > 0 {
                let mut left = self.position.x as i128;
                for c in line.chars() {
                    if left >= width {
                        break;
                    }
                    if left + advance > 0 {
                        self.draw_cell(image, c, left, top);
                    }
                    left += advance;
                }
            }
            top += line_height;
        }
    }
}

impl<P: Copy> Text<'_, P> {
    /// Draws the cell of `c` with its top-left corner at `(left, top)`,
    /// visiting only the glyph rows and columns whose blocks reach into the
    /// image, so the cost follows the visible part whatever the scale.
    fn draw_cell(&self, image: &mut impl ImageViewMut<Pixel = P>, c: char, left: i128, top: i128) {
        let size = image.size();
        let cell = self.font.cell_size();
        let scale = i128::from(self.scale);
        let glyph = self.font.glyph(c);
        let bytes_per_row = self.font.bytes_per_row();

        let rows = visible(top, scale, size.height as i128, cell.height);
        let columns = visible(left, scale, size.width as i128, cell.width);
        for row in rows {
            let y = top + row as i128 * scale;
            for column in columns.clone() {
                let ink = glyph.is_some_and(|glyph| {
                    let byte = glyph[row * bytes_per_row + column / 8];
                    byte & (0x80 >> (column % 8)) != 0
                });
                let value = if ink {
                    Some(self.color)
                } else {
                    self.background
                };
                if let Some(value) = value {
                    let x = left + column as i128 * scale;
                    for block_row in y.max(0)..(y + scale).min(size.height as i128) {
                        hspan(image, x, x + scale - 1, block_row, value);
                    }
                }
            }
        }
    }
}

/// The glyph rows (or columns) of a cell at `start` whose `scale`-sized
/// blocks overlap `0..extent`, as a range of indices below `count`.
fn visible(start: i128, scale: i128, extent: i128, count: usize) -> core::ops::Range<usize> {
    // Block k covers start + k·scale ..= start + k·scale + scale − 1.
    let first = (-start).div_euclid(scale).max(0);
    let last = (extent - 1 - start).div_euclid(scale);
    let end = (last + 1).clamp(0, count as i128);
    let first = first.min(end);
    first as usize..end as usize
}

/// Draws `content` in `font` with the top-left corner of its first
/// character at `position`.
///
/// One-shot wrapper over [`Text`]; see there for lines, scale, background
/// and clipping. Nothing is copied: the text is borrowed for the call.
///
/// # Examples
///
/// Labelling a measurement next to the point it was taken at, measured
/// first so the label ends just left of the point:
///
/// ```
/// use fovea::draw::{FONT_6X13, draw_crosshair, draw_text};
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::Mono8;
///
/// let mut image: Image<Mono8> = Image::zero(120, 40);
/// let at = (100_isize, 20_isize);
/// draw_crosshair(&mut image, at, 4, Mono8::new(255));
///
/// let label = "Ø 4.02 mm";
/// let size = FONT_6X13.text_size(label, 1);
/// let corner = (at.0 - 6 - size.width as isize, at.1 - size.height as isize / 2);
/// draw_text(&mut image, label, corner, Mono8::new(255), None, 1, &FONT_6X13);
///
/// // The label ends six pixels left of the point and never reaches it.
/// assert_eq!(corner.0 + size.width as isize, 94);
/// ```
pub fn draw_text<P: Copy>(
    image: &mut impl ImageViewMut<Pixel = P>,
    content: &str,
    position: impl Into<SignedCoordinate>,
    color: P,
    background: Option<P>,
    scale: u32,
    font: &BitmapFont<'_>,
) {
    Text {
        content,
        position: position.into(),
        color,
        background,
        scale,
        font,
    }
    .draw_into(image);
}

#[cfg(test)]
mod tests {
    use super::super::FONT_6X13;
    use super::super::tests::inked;
    use super::*;
    use crate::Size;
    use crate::image::{Image, ImageView};
    use crate::pixel::Mono8;

    fn ink() -> Mono8 {
        Mono8::new(255)
    }

    /// A 2 × 2 font with two characters: 'a' is the top-left pixel, 'b'
    /// the bottom-right one.
    fn tiny() -> BitmapFont<'static> {
        static CHARS: [char; 2] = ['a', 'b'];
        static PIXELS: [u8; 4] = [0b1000_0000, 0, 0, 0b0100_0000];
        BitmapFont::try_new(Size::new(2, 2), &CHARS, &PIXELS).unwrap()
    }

    fn text<'a>(content: &'a str, at: (isize, isize), font: &'a BitmapFont<'a>) -> Text<'a, Mono8> {
        Text {
            content,
            position: at.into(),
            color: ink(),
            background: None,
            scale: 1,
            font,
        }
    }

    #[test]
    fn glyphs_sit_in_cells_one_after_another() {
        let font = tiny();
        let mut image: Image<Mono8> = Image::zero(6, 2);
        text("ab", (1, 0), &font).draw_into(&mut image);
        assert_eq!(inked(&image), vec![(1, 0), (4, 1)]);
    }

    #[test]
    fn a_newline_starts_the_next_line_at_the_left_edge() {
        let font = tiny();
        let mut image: Image<Mono8> = Image::zero(4, 4);
        text("b\na", (0, 0), &font).draw_into(&mut image);
        assert_eq!(inked(&image), vec![(1, 1), (0, 2)]);
    }

    #[test]
    fn scale_turns_a_pixel_into_a_block() {
        let font = tiny();
        let mut image: Image<Mono8> = Image::zero(6, 6);
        Text {
            scale: 3,
            ..text("a", (0, 0), &font)
        }
        .draw_into(&mut image);
        let block: Vec<_> = (0..3).flat_map(|y| (0..3).map(move |x| (x, y))).collect();
        assert_eq!(inked(&image), block);
    }

    #[test]
    fn a_scale_of_zero_draws_nothing() {
        let mut image: Image<Mono8> = Image::zero(20, 20);
        Text {
            scale: 0,
            background: Some(ink()),
            ..text("AB", (0, 0), &FONT_6X13)
        }
        .draw_into(&mut image);
        assert!(inked(&image).is_empty());
    }

    #[test]
    fn the_background_fills_the_cell_and_none_leaves_it() {
        let font = tiny();
        let mut image: Image<Mono8> = Image::fill(2, 2, Mono8::new(7));
        Text {
            background: Some(Mono8::new(1)),
            ..text("a", (0, 0), &font)
        }
        .draw_into(&mut image);
        assert_eq!(image.pixel_at(0, 0), ink());
        assert_eq!(image.pixel_at(1, 1), Mono8::new(1));

        let mut image: Image<Mono8> = Image::fill(2, 2, Mono8::new(7));
        text("a", (0, 0), &font).draw_into(&mut image);
        assert_eq!(image.pixel_at(1, 1), Mono8::new(7));
    }

    #[test]
    fn an_uncovered_character_is_drawn_as_the_question_mark() {
        let mut uncovered: Image<Mono8> = Image::zero(6, 13);
        draw_text(&mut uncovered, "⌀", (0, 0), ink(), None, 1, &FONT_6X13);
        let mut question: Image<Mono8> = Image::zero(6, 13);
        draw_text(&mut question, "?", (0, 0), ink(), None, 1, &FONT_6X13);
        assert!(!inked(&question).is_empty());
        assert_eq!(inked(&uncovered), inked(&question));
    }

    #[test]
    fn a_font_without_question_mark_leaves_the_cell_to_the_background() {
        let font = tiny();
        let mut image: Image<Mono8> = Image::zero(2, 2);
        Text {
            background: Some(Mono8::new(3)),
            ..text("z", (0, 0), &font)
        }
        .draw_into(&mut image);
        assert!((0..2).all(|y| (0..2).all(|x| image.pixel_at(x, y) == Mono8::new(3))));
    }

    #[test]
    fn latin_1_renders_as_its_own_glyphs() {
        for c in ['µ', '°', '±', 'ä'] {
            let mut drawn: Image<Mono8> = Image::zero(6, 13);
            draw_text(
                &mut drawn,
                &c.to_string(),
                (0, 0),
                ink(),
                None,
                1,
                &FONT_6X13,
            );
            let mut question: Image<Mono8> = Image::zero(6, 13);
            draw_text(&mut question, "?", (0, 0), ink(), None, 1, &FONT_6X13);
            assert!(!inked(&drawn).is_empty(), "{c:?}");
            assert_ne!(inked(&drawn), inked(&question), "{c:?}");
        }
    }

    #[test]
    fn text_clips_on_every_side() {
        let font = tiny();
        let mut image: Image<Mono8> = Image::zero(3, 3);
        // 'b' inks the bottom-right of its cell: at (-1, -1) the cell's ink
        // lands on (0, 0); at (2, 2) it lands on (3, 3), outside.
        text("b", (-1, -1), &font).draw_into(&mut image);
        text("b", (2, 2), &font).draw_into(&mut image);
        assert_eq!(inked(&image), vec![(0, 0)]);
    }

    #[test]
    fn the_drawn_extent_is_the_measured_size() {
        let content = "12.5 µm\n±0.02";
        for scale in [1, 2, 3] {
            let size = FONT_6X13.text_size(content, scale);
            let mut image: Image<Mono8> = Image::zero(size.width + 4, size.height + 4);
            draw_text(
                &mut image,
                content,
                (2, 2),
                ink(),
                Some(Mono8::new(1)),
                scale,
                &FONT_6X13,
            );
            // The background paints whole cells, so the painted region is
            // exactly the cells of both lines, the second line shorter.
            let painted: Vec<_> = (0..image.height())
                .flat_map(|y| (0..image.width()).map(move |x| (x, y)))
                .filter(|&(x, y)| image.pixel_at(x, y) != Mono8::new(0))
                .collect();
            let (xs, ys): (Vec<_>, Vec<_>) = painted.iter().copied().unzip();
            assert_eq!(xs.iter().min(), Some(&2));
            assert_eq!(ys.iter().min(), Some(&2));
            assert_eq!(xs.iter().max(), Some(&(2 + size.width - 1)));
            assert_eq!(ys.iter().max(), Some(&(2 + size.height - 1)));
        }
    }

    #[test]
    fn huge_scale_and_far_positions_cost_only_the_visible_part() {
        // A scale of u32::MAX covers the image with the first ink block;
        // positions near the isize limits draw nothing and must not
        // overflow or walk the invisible extent.
        let mut image: Image<Mono8> = Image::zero(8, 8);
        draw_text(
            &mut image,
            "A",
            (0, 0),
            ink(),
            Some(Mono8::new(1)),
            u32::MAX,
            &FONT_6X13,
        );
        assert!((0..8).all(|y| (0..8).all(|x| image.pixel_at(x, y) != Mono8::new(0))));

        let mut image: Image<Mono8> = Image::zero(8, 8);
        let long = "W".repeat(10_000);
        draw_text(
            &mut image,
            &long,
            (isize::MIN, 0),
            ink(),
            None,
            u32::MAX,
            &FONT_6X13,
        );
        draw_text(
            &mut image,
            &long,
            (isize::MAX, isize::MAX),
            ink(),
            None,
            u32::MAX,
            &FONT_6X13,
        );
        draw_text(
            &mut image,
            "A\nA",
            (0, isize::MIN),
            ink(),
            None,
            u32::MAX,
            &FONT_6X13,
        );
    }

    #[test]
    fn visible_keeps_the_blocks_that_overlap_the_image() {
        // Blocks of 3 starting at -4: block 0 covers -4..=-2, block 1
        // -1..=1, block 2 2..=4, block 3 5..=7.
        assert_eq!(visible(-4, 3, 5, 10), 1..3);
        assert_eq!(visible(0, 1, 5, 3), 0..3);
        assert!(visible(10, 1, 5, 3).is_empty());
        assert!(visible(-100, 1, 5, 3).is_empty());
        assert!(visible(isize::MIN as i128, i128::from(u32::MAX), 5, 13).is_empty());
    }
}

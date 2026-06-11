use std::marker::PhantomData;

use crate::image::ImageRefMut;
use crate::image::sequential::ContiguousImageMut;
use crate::image::{ImageView, ImageViewMut};
use crate::{Coordinate, Rectangle, Size, Stride};

/// Enable tiling and sub-views
///
/// Implement to provide sub-view (region of interest)
/// and tiling iterators
pub trait SubView: ImageView {
    /// The immutable sub-view type returned by [`roi`](SubView::roi) and the tile/window iterators.
    type Sub<'a>: ImageView<Pixel = Self::Pixel>
    where
        Self: 'a;

    /// Returns a sub-view (region of interest) as ImageView
    fn roi(&self, rect: Rectangle) -> Option<Self::Sub<'_>>;

    /// Splits the image into a grid of non-overlapping tiles of the given `size`.
    ///
    /// Tiles at the right and bottom edges may be smaller than `size` when the
    /// image dimensions are not exact multiples of the tile size.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::Size;
    /// use fovea::image::{Image, ImageView, SubView};
    /// use fovea::pixel::Mono8;
    ///
    /// let img = Image::generate(6, 4, |x, y| Mono8::new((x + y * 6) as u8));
    /// let tiles: Vec<_> = img.tiles(Size::new(3, 2)).collect();
    /// assert_eq!(tiles.len(), 4);
    /// assert_eq!(tiles[0].size(), Size::new(3, 2));
    /// ```
    fn tiles(&self, size: Size) -> TileIter<'_, Self>
    where
        Self: Sized,
    {
        TileIter::new(self, size)
    }

    /// Returns a sliding-window iterator with stride 1 over the image.
    ///
    /// Slides a window of the given `size` across the image, advancing by one
    /// pixel in each direction. Only yields windows that fit **entirely**
    /// within the image — no partial windows are produced.
    ///
    /// For custom strides, use the [`SlidingWindow`] builder instead:
    ///
    /// ```
    /// use fovea::{Size, Stride};
    /// use fovea::image::{Image, ImageView, SubView, SlidingWindow};
    /// use fovea::pixel::Mono8;
    ///
    /// let img = Image::generate(8, 8, |x, y| Mono8::new((x + y) as u8));
    ///
    /// // Builder API for stride > 1:
    /// let windows: Vec<_> = SlidingWindow::new(Size::new(3, 3))
    ///     .stride(Stride::new(2, 2))
    ///     .iter(&img)
    ///     .collect();
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::Size;
    /// use fovea::image::{Image, ImageView, SubView};
    /// use fovea::pixel::Mono8;
    ///
    /// let img = Image::generate(4, 4, |x, y| Mono8::new((x + y * 4) as u8));
    /// // 3×3 window on a 4×4 image with stride 1 → 2×2 = 4 positions
    /// let windows: Vec<_> = img.sliding_windows(Size::new(3, 3)).collect();
    /// assert_eq!(windows.len(), 4);
    /// assert!(windows.iter().all(|w| w.size() == Size::new(3, 3)));
    /// ```
    fn sliding_windows(&self, size: Size) -> SlidingWindowIter<'_, Self>
    where
        Self: Sized,
    {
        SlidingWindowIter::new(self, size, Stride::one())
    }
}

/// Enable mutable tiling and sub-views
///
/// Implement to provide mutable sub-view (region of interest)
/// and tiling iterators
pub trait SubViewMut: SubView + ImageViewMut {
    /// The mutable sub-view type returned by [`roi_mut`](SubViewMut::roi_mut).
    type SubMut<'a>: ImageViewMut<Pixel = Self::Pixel>
    where
        Self: 'a;

    /// Returns a mutable sub-view for `rect`, or `None` if `rect` exceeds image bounds.
    fn roi_mut(&mut self, rect: Rectangle) -> Option<Self::SubMut<'_>>;
}

/// An iterator that yields non-overlapping sub-views (tiles) of a fixed size.
///
/// Produced by [`SubView::tiles`]. Partial tiles appear at the right and bottom
/// edges when the image dimensions are not exact multiples of the tile size.
#[derive(Clone, Debug)]
pub struct TileIter<'a, T: SubView> {
    size: Size,
    current: crate::Coordinate,
    img: &'a T,
}
impl<'a, T> TileIter<'a, T>
where
    T: SubView,
{
    pub(crate) fn new(img: &'a T, size: Size) -> TileIter<'a, T> {
        assert!(
            size.width > 0 && size.height > 0,
            "TileIter: tile size must be non-zero in both dimensions, got {size:?}"
        );
        TileIter {
            size,
            current: crate::Coordinate::new(0, 0),
            img,
        }
    }
}

impl<'a, T> Iterator for TileIter<'a, T>
where
    T: SubView,
{
    type Item = <T as SubView>::Sub<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        // if the index is above the whole image
        // the iterator has finished
        if self.current.y >= self.img.height() {
            return None;
        }

        // Clamp tile size to fit within image bounds (enables partial tiles at edges)
        let clamped_width = self.size.width.min(self.img.width() - self.current.x);
        let clamped_height = self.size.height.min(self.img.height() - self.current.y);
        let clamped_size = Size::new(clamped_width, clamped_height);

        let roi = self.img.roi(Rectangle::new(self.current, clamped_size));

        self.current = Coordinate {
            x: self.current.x + self.size.width,
            y: self.current.y,
        };

        if self.current.x >= self.img.width() {
            self.current = Coordinate {
                x: 0,
                y: self.current.y + self.size.height,
            };
        }

        roi
    }
}

// ───────────────────────────────────────────────────────────────────
// Sliding window iterator
// ───────────────────────────────────────────────────────────────────

/// A builder for constructing [`SlidingWindowIter`] with configurable
/// window size and stride.
///
/// The builder defaults to stride `(1, 1)` when [`stride`](SlidingWindow::stride)
/// is not called.
///
/// # Example
///
/// ```
/// use fovea::{Size, Stride};
/// use fovea::image::{Image, ImageView, SubView, SlidingWindow};
/// use fovea::pixel::Mono8;
///
/// let img = Image::generate(10, 10, |x, y| Mono8::new((x + y) as u8));
///
/// // Stride-1 (default) — same as img.sliding_windows(size)
/// let iter = SlidingWindow::new(Size::new(3, 3)).iter(&img);
/// assert_eq!(iter.count(), 8 * 8);
///
/// // Custom stride
/// let iter = SlidingWindow::new(Size::new(3, 3))
///     .stride(Stride::new(2, 2))
///     .iter(&img);
/// assert_eq!(iter.count(), 4 * 4);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlidingWindow {
    window_size: Size,
    stride: Stride,
}

impl SlidingWindow {
    /// Creates a new builder with the given window size and stride `(1, 1)`.
    pub fn new(window_size: Size) -> Self {
        Self {
            window_size,
            stride: Stride::one(),
        }
    }

    /// Sets the stride (step size between successive window positions).
    pub fn stride(mut self, stride: Stride) -> Self {
        self.stride = stride;
        self
    }

    /// Produces a [`SlidingWindowIter`] over the given image.
    ///
    /// Only positions where the full window fits inside the image are visited.
    pub fn iter<'a, I>(&self, image: &'a I) -> SlidingWindowIter<'a, I>
    where
        I: SubView,
    {
        SlidingWindowIter::new(image, self.window_size, self.stride)
    }
}

/// An iterator that slides a fixed-size window across an image,
/// yielding zero-copy sub-views at each position.
///
/// Only positions where the window fits **entirely** within the image
/// are visited — no partial or clamped windows are produced. This is
/// the key difference from [`TileIter`], which yields partial tiles at
/// the edges.
///
/// # Construction
///
/// - Via [`SubView::sliding_windows`] for stride-1 iteration (the common case).
/// - Via the [`SlidingWindow`] builder for custom strides.
///
/// # Example
///
/// ```
/// use fovea::{Size, Stride};
/// use fovea::image::{Image, ImageView, SubView, SlidingWindow};
/// use fovea::pixel::Mono8;
///
/// let img = Image::generate(6, 6, |x, y| Mono8::new((x + y * 6) as u8));
///
/// // Stride 1 via the SubView convenience method
/// let count = img.sliding_windows(Size::new(3, 3)).count();
/// assert_eq!(count, 4 * 4); // (6-3+1) × (6-3+1) = 16
///
/// // Stride 2 via the builder
/// let count = SlidingWindow::new(Size::new(3, 3))
///     .stride(Stride::new(2, 2))
///     .iter(&img)
///     .count();
/// assert_eq!(count, 2 * 2); // positions (0,0),(2,0),(0,2),(2,2)
/// ```
#[derive(Clone, Debug)]
pub struct SlidingWindowIter<'a, T: SubView> {
    window_size: Size,
    stride: Stride,
    current: Coordinate,
    img: &'a T,
    /// Number of valid x-positions (computed once in constructor).
    cols: usize,
    /// Number of valid y-positions (computed once in constructor).
    rows: usize,
}

impl<'a, T> SlidingWindowIter<'a, T>
where
    T: SubView,
{
    pub(crate) fn new(img: &'a T, window_size: Size, stride: Stride) -> Self {
        assert!(
            stride.horizontal() > 0 && stride.vertical() > 0,
            "SlidingWindowIter: stride must be non-zero in both dimensions, got {stride:?}"
        );
        // Compute the number of valid positions along each axis.
        // A window fits at position p if p + window_size <= image_dim,
        // i.e. p <= image_dim - window_size. With stride s, valid
        // positions are 0, s, 2s, … up to that limit.
        let (cols, rows) = if window_size.width > img.width()
            || window_size.height > img.height()
            || window_size.width == 0
            || window_size.height == 0
        {
            (0, 0)
        } else {
            let max_x = img.width() - window_size.width; // inclusive
            let max_y = img.height() - window_size.height;
            let cols = max_x / stride.horizontal() + 1;
            let rows = max_y / stride.vertical() + 1;
            (cols, rows)
        };

        Self {
            window_size,
            stride,
            current: Coordinate::new(0, 0),
            img,
            cols,
            rows,
        }
    }

    /// Wraps this iterator to also yield the `(col, row)` grid position
    /// of each window, similar to [`Iterator::enumerate`].
    ///
    /// The positions are **grid indices** (0-based column and row within
    /// the sliding window grid), not pixel coordinates. To get pixel
    /// coordinates, multiply by the stride.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::Size;
    /// use fovea::image::{Image, ImageView, SubView};
    /// use fovea::pixel::Mono8;
    ///
    /// let img = Image::generate(5, 5, |x, y| Mono8::new((x + y * 5) as u8));
    /// for ((col, row), window) in img.sliding_windows(Size::new(3, 3)).enumerate_positions() {
    ///     assert_eq!(window.size(), Size::new(3, 3));
    ///     // col in 0..3, row in 0..3
    ///     assert!(col < 3);
    ///     assert!(row < 3);
    /// }
    /// ```
    pub fn enumerate_positions(self) -> EnumeratePositions<'a, T> {
        EnumeratePositions {
            inner: self,
            col: 0,
            row: 0,
        }
    }
}

impl<'a, T> Iterator for SlidingWindowIter<'a, T>
where
    T: SubView,
{
    type Item = <T as SubView>::Sub<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current.y >= self.rows {
            return None;
        }

        let px = self.current.x * self.stride.horizontal();
        let py = self.current.y * self.stride.vertical();

        let roi = self
            .img
            .roi(Rectangle::new(Coordinate::new(px, py), self.window_size));

        // Advance to next column, wrap to next row
        self.current.x += 1;
        if self.current.x >= self.cols {
            self.current.x = 0;
            self.current.y += 1;
        }

        roi
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = if self.current.y >= self.rows {
            0
        } else {
            let remaining_in_row = self.cols - self.current.x;
            let remaining_full_rows = self.rows - self.current.y - 1;
            remaining_in_row + remaining_full_rows * self.cols
        };
        (remaining, Some(remaining))
    }
}

impl<'a, T> ExactSizeIterator for SlidingWindowIter<'a, T> where T: SubView {}

/// Iterator adapter that pairs each sliding window with its `(col, row)`
/// grid position. Produced by [`SlidingWindowIter::enumerate_positions`].
#[derive(Clone, Debug)]
pub struct EnumeratePositions<'a, T: SubView> {
    inner: SlidingWindowIter<'a, T>,
    col: usize,
    row: usize,
}

impl<'a, T> Iterator for EnumeratePositions<'a, T>
where
    T: SubView,
{
    type Item = ((usize, usize), <T as SubView>::Sub<'a>);

    fn next(&mut self) -> Option<Self::Item> {
        let window = self.inner.next()?;
        let pos = (self.col, self.row);

        self.col += 1;
        if self.col >= self.inner.cols {
            self.col = 0;
            self.row += 1;
        }

        Some((pos, window))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a, T> ExactSizeIterator for EnumeratePositions<'a, T> where T: SubView {}

/// A mutable tile iterator that yields disjoint [`ImageRefMut`] views
/// over a contiguously-stored image buffer.
///
/// `TileIterMut` splits an image into a grid of non-overlapping rectangular
/// tiles and yields each tile as an [`ImageRefMut`] — a mutable view
/// into the underlying pixel buffer. Tiles at the right and bottom edges
/// may be smaller than the requested tile size when the image dimensions
/// are not exact multiples.
///
/// # Internal representation
///
/// Uses a raw pointer (`*mut T`) internally so that multiple disjoint
/// `ImageRefMut` tiles can coexist simultaneously. This is safe
/// because the grid-based tiling with monotonic advancement guarantees
/// that no two tiles share a pixel.
///
/// # Construction
///
/// Construction is **`pub(crate)`** — instances are created by the
/// [`IntoTilesMut`] blanket impl. Users obtain a `TileIterMut` by calling
/// [`IntoTilesMut::into_tiles_mut`] on a mutable image reference.
///
/// # Example
///
/// ```
/// use fovea::Size;
/// use fovea::image::{Image, ImageView, ImageViewMut, IntoTilesMut};
///
/// let mut img = Image::fill(6, 4, 0u8);
/// let mut tile_count = 0;
/// for mut tile in (&mut img).into_tiles_mut(Size::new(3, 2)) {
///     *tile.pixel_at_mut(0, 0) = 42;
///     tile_count += 1;
/// }
/// assert_eq!(tile_count, 4);
/// assert_eq!(img.get(0, 0), Some(42));
/// assert_eq!(img.get(3, 0), Some(42));
/// ```
///
/// [`IntoTilesMut`]: crate::image::IntoTilesMut
/// [`IntoTilesMut::into_tiles_mut`]: crate::image::IntoTilesMut::into_tiles_mut
pub struct TileIterMut<'a, T> {
    data: *mut T,
    len: usize,
    image_size: Size,
    tile_size: Size,
    current: Coordinate,
    _marker: PhantomData<&'a mut T>,
}

// SAFETY: TileIterMut has exclusive access to the pixel buffer for lifetime 'a.
// Sending it to another thread is safe when T is Send (same as &'a mut [T]).
unsafe impl<T: Send> Send for TileIterMut<'_, T> {}

// SAFETY: Shared access (&TileIterMut) does not expose &T directly, but even if
// it did, Sync where T: Sync matches the guarantees of &'a mut [T].
unsafe impl<T: Sync> Sync for TileIterMut<'_, T> {}

impl<'a, T> TileIterMut<'a, T> {
    /// Creates a new mutable tile iterator.
    ///
    /// # Safety
    /// - `data` must come from a valid `&'a mut [T]` of length `len`
    /// - `image_size.width * image_size.height` must equal `len`
    /// - The caller must not retain any other reference to the data for `'a`
    pub(crate) unsafe fn new(data: *mut T, len: usize, image_size: Size, tile_size: Size) -> Self {
        assert!(
            tile_size.width > 0 && tile_size.height > 0,
            "TileIterMut: tile size must be non-zero in both dimensions, got {tile_size:?}"
        );
        // Real (non-debug) assertion: the blanket `IntoTilesMut` impl is
        // reachable through `ContiguousImageMut`, whose `as_slice()` length
        // contract is sealed today but still depends on the
        // implementor reporting a consistent `size()`. Failing this check
        // in safe code must never silently produce out-of-bounds tiles.
        let expected = image_size
            .checked_area()
            .expect("TileIterMut::new: image_size area overflows usize");
        assert_eq!(
            expected, len,
            "TileIterMut::new: ContiguousImageMut reported size {image_size:?} \
             whose area ({expected}) does not match the underlying slice length ({len})",
        );
        Self {
            data,
            len,
            image_size,
            tile_size,
            current: Coordinate::new(0, 0),
            _marker: PhantomData,
        }
    }
}

impl<'a, T> Iterator for TileIterMut<'a, T> {
    type Item = ImageRefMut<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        // If the current position is past the image, the iterator is exhausted.
        if self.current.y >= self.image_size.height {
            return None;
        }

        // Clamp tile size to fit within image bounds (partial tiles at edges).
        let clamped_width = self
            .tile_size
            .width
            .min(self.image_size.width - self.current.x);
        let clamped_height = self
            .tile_size
            .height
            .min(self.image_size.height - self.current.y);
        let clamped_size = Size::new(clamped_width, clamped_height);

        let rect = Rectangle::new(self.current, clamped_size);

        let stride = self.image_size.width;
        let offset = rect.top() * stride + rect.left();

        // SAFETY:
        // 1. Tiles are non-overlapping: grid-based tiling with monotonic
        //    advancement ensures no two tiles share a pixel.
        // 2. All indices within `rect` fall within 0..len: guaranteed by
        //    clamping to image bounds.
        // 3. Lifetime 'a ties each tile to the original &mut borrow.
        // 4. Each pixel region is yielded exactly once (forward-only iterator).
        let roi = unsafe { ImageRefMut::strided(rect.size, stride, offset, self.data, self.len) };

        // Advance to next tile position (same logic as TileIter).
        self.current = Coordinate {
            x: self.current.x + self.tile_size.width,
            y: self.current.y,
        };

        if self.current.x >= self.image_size.width {
            self.current = Coordinate {
                x: 0,
                y: self.current.y + self.tile_size.height,
            };
        }

        Some(roi)
    }
}

// ───────────────────────────────────────────────────────────────────
// Sealed IntoTilesMut trait
// ───────────────────────────────────────────────────────────────────

mod sealed {
    pub trait Sealed {}
}

/// Trait for splitting a mutable image into a grid of non-overlapping mutable tiles.
///
/// Each tile is a disjoint [`ImageViewMut`] backed by the same pixel buffer. Because
/// the tiles do not overlap, multiple tiles can safely coexist as mutable references
/// (enforced internally via raw pointers with strict safety invariants).
///
/// # Sealed
///
/// This trait is **sealed** — it cannot be implemented outside this crate. A
/// user-provided implementation that yields overlapping tiles would cause
/// undefined behaviour.
///
/// # Example
///
/// ```
/// use fovea::Size;
/// use fovea::image::{Image, ImageView, ImageViewMut, IntoTilesMut};
///
/// let mut img = Image::fill(8, 8, 0u8);
/// for mut tile in (&mut img).into_tiles_mut(Size::new(4, 4)) {
///     // each tile is a disjoint &mut view
///     *tile.pixel_at_mut(0, 0) = 255;
/// }
/// assert_eq!(img.get(0, 0), Some(255));
/// assert_eq!(img.get(4, 0), Some(255));
/// assert_eq!(img.get(0, 4), Some(255));
/// assert_eq!(img.get(4, 4), Some(255));
/// ```
pub trait IntoTilesMut<'a>: sealed::Sealed {
    /// The pixel type of the tiles.
    type Pixel;
    /// The mutable tile view type yielded by the iterator.
    type TileMut<'b>: ImageViewMut<Pixel = Self::Pixel>
    where
        Self: 'b;
    /// The iterator type over mutable tiles.
    type TilesIterMut<'b>: Iterator<Item = Self::TileMut<'b>>
    where
        Self: 'b;

    /// Splits the image into a grid of non-overlapping mutable tiles of the given `size`.
    ///
    /// Tiles at the right and bottom edges may be smaller than `size` when the
    /// image dimensions are not exact multiples of the tile size.
    fn into_tiles_mut(self, size: Size) -> Self::TilesIterMut<'a>;
}

// ── Blanket impl IntoTilesMut for &'a mut I where I: ContiguousImageMut ─

impl<I: ContiguousImageMut> sealed::Sealed for &mut I {}

impl<'a, I> IntoTilesMut<'a> for &'a mut I
where
    I: ContiguousImageMut,
{
    type Pixel = <I as ImageView>::Pixel;
    type TileMut<'b>
        = ImageRefMut<'b, Self::Pixel>
    where
        Self: 'b;
    type TilesIterMut<'b>
        = TileIterMut<'b, Self::Pixel>
    where
        Self: 'b;

    fn into_tiles_mut(self, size: Size) -> Self::TilesIterMut<'a> {
        let image_size = self.size();
        let slice = self.as_mut_slice();
        let len = slice.len();
        let ptr = slice.as_mut_ptr();
        // SAFETY: ptr comes from a valid &'a mut [T] of length len.
        // ContiguousImageMut guarantees width * height == len.
        // We hold the only &mut reference (consumed by into_tiles_mut).
        unsafe { TileIterMut::new(ptr, len, image_size, size) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::ImageView;
    use crate::image::sequential::{
        ContiguousImage, ContiguousImageMut, Image, ImageArray, ImageRefMut,
    };
    use crate::pixel::Mono8;

    // ───────────────────────────────────────────────────────────────────
    // SlidingWindowIter tests
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_sliding_window_stride1_count() {
        // (W - kw + 1) * (H - kh + 1) = (6-3+1)*(6-3+1) = 16
        let img: Image<Mono8> = Image::generate(6, 6, |x, y| Mono8::new((x + y * 6) as u8));
        assert_eq!(img.sliding_windows(Size::new(3, 3)).count(), 16);
    }

    #[test]
    fn test_sliding_window_stride1_all_same_size() {
        let img: Image<Mono8> = Image::generate(5, 5, |x, y| Mono8::new((x + y * 5) as u8));
        for w in img.sliding_windows(Size::new(3, 3)) {
            assert_eq!(w.size(), Size::new(3, 3));
        }
    }

    #[test]
    fn test_sliding_window_stride1_data() {
        // 4×4 image, 2×2 window, stride 1 → 3×3 = 9 windows
        let img: Image<Mono8> = Image::generate(4, 4, |x, y| Mono8::new((x + y * 4) as u8));
        let windows: Vec<_> = img.sliding_windows(Size::new(2, 2)).collect();
        assert_eq!(windows.len(), 9);
        // First window at (0,0)
        assert_eq!(windows[0].get(0, 0), Some(Mono8::new(0)));
        assert_eq!(windows[0].get(1, 0), Some(Mono8::new(1)));
        assert_eq!(windows[0].get(0, 1), Some(Mono8::new(4)));
        assert_eq!(windows[0].get(1, 1), Some(Mono8::new(5)));
        // Second window at (1,0)
        assert_eq!(windows[1].get(0, 0), Some(Mono8::new(1)));
        assert_eq!(windows[1].get(1, 1), Some(Mono8::new(6)));
        // Window at (0,1) — 4th window (index 3)
        assert_eq!(windows[3].get(0, 0), Some(Mono8::new(4)));
        // Last window at (2,2) — index 8
        assert_eq!(windows[8].get(0, 0), Some(Mono8::new(10)));
        assert_eq!(windows[8].get(1, 1), Some(Mono8::new(15)));
    }

    #[test]
    fn test_sliding_window_stride2() {
        // 8×8 image, 3×3 window, stride 2
        // max_x = 8-3 = 5, cols = 5/2+1 = 3
        // max_y = 8-3 = 5, rows = 5/2+1 = 3
        // → 9 windows
        let img: Image<Mono8> = Image::generate(8, 8, |x, y| Mono8::new((x + y * 8) as u8));
        let windows: Vec<_> = SlidingWindow::new(Size::new(3, 3))
            .stride(Stride::new(2, 2))
            .iter(&img)
            .collect();
        assert_eq!(windows.len(), 9);
        // First at (0,0)
        assert_eq!(windows[0].get(0, 0), Some(Mono8::new(0)));
        // Second at (2,0)
        assert_eq!(windows[1].get(0, 0), Some(Mono8::new(2)));
        // Third at (4,0)
        assert_eq!(windows[2].get(0, 0), Some(Mono8::new(4)));
        // Fourth at (0,2)
        assert_eq!(windows[3].get(0, 0), Some(Mono8::new(16)));
    }

    #[test]
    fn test_sliding_window_non_square_stride() {
        // 10×8, 3×3, stride (3, 2)
        // max_x = 10-3 = 7, cols = 7/3+1 = 3  (positions 0,3,6)
        // max_y = 8-3 = 5,  rows = 5/2+1 = 3  (positions 0,2,4)
        let img: Image<Mono8> = Image::generate(10, 8, |x, y| Mono8::new((x + y * 10) as u8));
        let windows: Vec<_> = SlidingWindow::new(Size::new(3, 3))
            .stride(Stride::new(3, 2))
            .iter(&img)
            .collect();
        assert_eq!(windows.len(), 9);
        // Check pixel origin of each window
        assert_eq!(windows[0].get(0, 0), Some(Mono8::new(0))); // (0,0)
        assert_eq!(windows[1].get(0, 0), Some(Mono8::new(3))); // (3,0)
        assert_eq!(windows[2].get(0, 0), Some(Mono8::new(6))); // (6,0)
        assert_eq!(windows[3].get(0, 0), Some(Mono8::new(20))); // (0,2)
        assert_eq!(windows[6].get(0, 0), Some(Mono8::new(40))); // (0,4)
    }

    #[test]
    fn test_sliding_window_window_equals_image() {
        let img: Image<Mono8> = Image::generate(3, 3, |x, y| Mono8::new((x + y * 3) as u8));
        let windows: Vec<_> = img.sliding_windows(Size::new(3, 3)).collect();
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].size(), Size::new(3, 3));
        assert_eq!(windows[0].get(0, 0), Some(Mono8::new(0)));
        assert_eq!(windows[0].get(2, 2), Some(Mono8::new(8)));
    }

    #[test]
    fn test_sliding_window_larger_than_image() {
        let img: Image<Mono8> = Image::generate(3, 3, |x, y| Mono8::new((x + y * 3) as u8));
        let windows: Vec<_> = img.sliding_windows(Size::new(4, 4)).collect();
        assert_eq!(windows.len(), 0);
    }

    #[test]
    fn test_sliding_window_1x1() {
        // 1×1 window → every pixel is a window
        let img: Image<Mono8> = Image::generate(3, 4, |x, y| Mono8::new((x + y * 3) as u8));
        let windows: Vec<_> = img.sliding_windows(Size::new(1, 1)).collect();
        assert_eq!(windows.len(), 12); // 3*4
        for w in &windows {
            assert_eq!(w.size(), Size::new(1, 1));
        }
        assert_eq!(windows[0].get(0, 0), Some(Mono8::new(0)));
        assert_eq!(windows[11].get(0, 0), Some(Mono8::new(11)));
    }

    #[test]
    fn test_sliding_window_non_square_window() {
        // 6×4 image, 2×3 window, stride 1
        // cols = 6-2+1 = 5, rows = 4-3+1 = 2 → 10
        let img: Image<Mono8> = Image::generate(6, 4, |x, y| Mono8::new((x + y * 6) as u8));
        let windows: Vec<_> = img.sliding_windows(Size::new(2, 3)).collect();
        assert_eq!(windows.len(), 10);
        for w in &windows {
            assert_eq!(w.size(), Size::new(2, 3));
        }
    }

    #[test]
    fn test_sliding_window_stride_larger_than_window() {
        // 10×10, 2×2 window, stride 4
        // max_x = 10-2 = 8, cols = 8/4+1 = 3  (pos 0,4,8)
        // max_y = 10-2 = 8, rows = 8/4+1 = 3
        // → 9 windows
        let img: Image<Mono8> = Image::generate(10, 10, |x, y| Mono8::new((x + y * 10) as u8));
        let windows: Vec<_> = SlidingWindow::new(Size::new(2, 2))
            .stride(Stride::new(4, 4))
            .iter(&img)
            .collect();
        assert_eq!(windows.len(), 9);
    }

    #[test]
    fn test_sliding_window_stride_skips_last_position() {
        // 7×7, 3×3 window, stride 3
        // max_x = 7-3 = 4, cols = 4/3+1 = 2  (positions 0, 3)
        // Position 6 would be 6+3=9 > 7, not visited. That's correct:
        // max_x=4, 2*3=6 > 4, so only 0 and 3.
        let img: Image<Mono8> = Image::generate(7, 7, |x, y| Mono8::new((x + y * 7) as u8));
        let windows: Vec<_> = SlidingWindow::new(Size::new(3, 3))
            .stride(Stride::new(3, 3))
            .iter(&img)
            .collect();
        assert_eq!(windows.len(), 4); // 2×2
        assert_eq!(windows[0].get(0, 0), Some(Mono8::new(0))); // (0,0)
        assert_eq!(windows[1].get(0, 0), Some(Mono8::new(3))); // (3,0)
        assert_eq!(windows[2].get(0, 0), Some(Mono8::new(21))); // (0,3)
        assert_eq!(windows[3].get(0, 0), Some(Mono8::new(24))); // (3,3)
    }

    #[test]
    fn test_sliding_window_exact_size_iterator() {
        let img: Image<Mono8> = Image::generate(6, 6, |x, y| Mono8::new((x + y * 6) as u8));
        let mut iter = img.sliding_windows(Size::new(3, 3));
        assert_eq!(iter.len(), 16);
        iter.next();
        assert_eq!(iter.len(), 15);
        // Exhaust
        for _ in &mut iter {}
        assert_eq!(iter.len(), 0);
    }

    #[test]
    fn test_sliding_window_exhaustion() {
        let img: Image<Mono8> = Image::generate(4, 4, |x, y| Mono8::new((x + y * 4) as u8));
        let mut iter = img.sliding_windows(Size::new(3, 3));
        let mut count = 0;
        while iter.next().is_some() {
            count += 1;
        }
        assert_eq!(count, 4);
        assert!(iter.next().is_none());
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_sliding_window_enumerate_positions() {
        let img: Image<Mono8> = Image::generate(5, 5, |x, y| Mono8::new((x + y * 5) as u8));
        let items: Vec<_> = img
            .sliding_windows(Size::new(3, 3))
            .enumerate_positions()
            .collect();
        // 3×3 grid of positions
        assert_eq!(items.len(), 9);
        assert_eq!(items[0].0, (0, 0));
        assert_eq!(items[1].0, (1, 0));
        assert_eq!(items[2].0, (2, 0));
        assert_eq!(items[3].0, (0, 1));
        assert_eq!(items[8].0, (2, 2));
        // Data check: position (1,2) → pixel origin (1,2)
        assert_eq!(items[7].0, (1, 2));
        assert_eq!(items[7].1.get(0, 0), Some(Mono8::new(11))); // pixel at (1,2)=1+2*5=11
    }

    #[test]
    fn test_sliding_window_enumerate_positions_with_stride() {
        let img: Image<Mono8> = Image::generate(8, 8, |x, y| Mono8::new((x + y * 8) as u8));
        let items: Vec<_> = SlidingWindow::new(Size::new(3, 3))
            .stride(Stride::new(2, 2))
            .iter(&img)
            .enumerate_positions()
            .collect();
        assert_eq!(items.len(), 9); // 3×3 grid
        // Grid position (1,1) → pixel origin (2,2)
        assert_eq!(items[4].0, (1, 1));
        assert_eq!(items[4].1.get(0, 0), Some(Mono8::new(18))); // 2+2*8=18
    }

    #[test]
    fn test_sliding_window_enumerate_positions_exact_size() {
        let img: Image<Mono8> = Image::generate(5, 5, |x, y| Mono8::new((x + y * 5) as u8));
        let mut iter = img.sliding_windows(Size::new(3, 3)).enumerate_positions();
        assert_eq!(iter.len(), 9);
        iter.next();
        assert_eq!(iter.len(), 8);
    }

    #[test]
    fn test_sliding_window_imagearray() {
        let img: ImageArray<Mono8, 6, 6> = ImageArray::generate(|x, y| Mono8::new((x + y * 6) as u8));
        let windows: Vec<_> = img.sliding_windows(Size::new(3, 3)).collect();
        assert_eq!(windows.len(), 16);
        assert_eq!(windows[0].get(0, 0), Some(Mono8::new(0)));
        // cols=4, rows=4. Last window index 15 → col=3, row=3 → px=(3,3) → 3+3*6=21
        assert_eq!(windows[15].get(0, 0), Some(Mono8::new(21)));
    }

    #[test]
    fn test_sliding_window_does_not_consume() {
        let img: Image<Mono8> = Image::generate(4, 4, |x, y| Mono8::new((x + y * 4) as u8));
        let w1: Vec<_> = img.sliding_windows(Size::new(2, 2)).collect();
        let w2: Vec<_> = img.sliding_windows(Size::new(2, 2)).collect();
        assert_eq!(w1.len(), w2.len());
        assert_eq!(img.get(0, 0), Some(Mono8::new(0)));
    }

    #[test]
    fn test_sliding_window_builder_default_stride() {
        // Builder with no .stride() call should produce stride-1 results
        let img: Image<Mono8> = Image::generate(5, 5, |x, y| Mono8::new((x + y * 5) as u8));
        let from_method: Vec<_> = img.sliding_windows(Size::new(3, 3)).collect();
        let from_builder: Vec<_> = SlidingWindow::new(Size::new(3, 3)).iter(&img).collect();
        assert_eq!(from_method.len(), from_builder.len());
        for (a, b) in from_method.iter().zip(from_builder.iter()) {
            assert_eq!(a.get(0, 0), b.get(0, 0));
        }
    }

    #[test]
    fn test_sliding_window_row_major_order() {
        // Verify windows are yielded in row-major order
        let img: Image<Mono8> = Image::generate(5, 5, |x, y| Mono8::new((x + y * 5) as u8));
        let windows: Vec<_> = img.sliding_windows(Size::new(2, 2)).collect();
        // 4×4 = 16 windows. The top-left pixel of each window should
        // follow row-major: (0,0),(1,0),(2,0),(3,0),(0,1),(1,1),...
        let origins: Vec<Mono8> = windows.iter().map(|w| w.pixel_at(0, 0)).collect();
        let expected: Vec<Mono8> = (0..4)
            .flat_map(|y| (0..4).map(move |x| Mono8::new((x + y * 5) as u8)))
            .collect();
        assert_eq!(origins, expected);
    }

    // ───────────────────────────────────────────────────────────────────
    // Immutable tile iterator tests (existing)
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_tiles_iter_basic() {
        let img: Image<Mono8> = Image::generate(4, 4, |x, y| Mono8::new((x + y * 4) as u8 + 1));
        let tiles: Vec<_> = img.tiles(Size::new(2, 2)).collect();
        // 4x4 image with 2x2 tiles = 4 tiles (2x2 grid of tiles)
        assert_eq!(tiles.len(), 4);
        // First tile should have correct data
        assert_eq!(tiles[0].get(0, 0), Some(Mono8::new(1)));
        assert_eq!(tiles[0].size(), Size::new(2, 2));
    }

    #[test]
    fn test_tiles_iter_partial_edges() {
        // 10x10 image with 3x3 tiles should produce partial tiles at edges
        let img: Image<Mono8> = Image::generate(10, 10, |x, y| Mono8::new((x + y * 10) as u8));
        let tiles: Vec<_> = img.tiles(Size::new(3, 3)).collect();

        // Should have 4x4 = 16 tiles (at x: 0,3,6,9 and y: 0,3,6,9)
        assert_eq!(tiles.len(), 16);

        // First tile at (0,0) should be full 3x3
        assert_eq!(tiles[0].size(), Size::new(3, 3));

        // Tile at (9,0) - rightmost in first row - should be 1x3 (clamped width)
        assert_eq!(tiles[3].size(), Size::new(1, 3));

        // Tile at (0,9) - bottom-left - should be 3x1 (clamped height)
        assert_eq!(tiles[12].size(), Size::new(3, 1));

        // Tile at (9,9) - bottom-right corner - should be 1x1 (clamped both)
        assert_eq!(tiles[15].size(), Size::new(1, 1));
    }

    #[test]
    fn test_tiles_iter_partial_edges_data() {
        // Verify the data in partial tiles is correct
        let img: Image<Mono8> = Image::generate(5, 5, |x, y| Mono8::new((x + y * 5) as u8));
        let tiles: Vec<_> = img.tiles(Size::new(3, 3)).collect();

        // Should have 2x2 = 4 tiles
        assert_eq!(tiles.len(), 4);

        // Top-right tile at (3,0) should be 2x3
        assert_eq!(tiles[1].size(), Size::new(2, 3));
        assert_eq!(tiles[1].get(0, 0), Some(Mono8::new(3))); // pixel at (3,0) in original
        assert_eq!(tiles[1].get(1, 0), Some(Mono8::new(4))); // pixel at (4,0) in original

        // Bottom-left tile at (0,3) should be 3x2
        assert_eq!(tiles[2].size(), Size::new(3, 2));
        assert_eq!(tiles[2].get(0, 0), Some(Mono8::new(15))); // pixel at (0,3) in original

        // Bottom-right tile at (3,3) should be 2x2
        assert_eq!(tiles[3].size(), Size::new(2, 2));
        assert_eq!(tiles[3].get(0, 0), Some(Mono8::new(18))); // pixel at (3,3) in original
        assert_eq!(tiles[3].get(1, 1), Some(Mono8::new(24))); // pixel at (4,4) in original
    }

    #[test]
    fn test_tiles_iter_larger_than_image() {
        let img: Image<Mono8> = Image::generate(2, 2, |x, y| Mono8::new((x + y * 2) as u8));
        let tiles: Vec<_> = img.tiles(Size::new(5, 5)).collect();
        // Tile is larger than image, should get 1 tile clamped to image size (2x2)
        assert_eq!(tiles.len(), 1);
        assert_eq!(tiles[0].size(), Size::new(2, 2));
    }

    #[test]
    fn test_tiles_iter_with_imagearray() {
        let img: ImageArray<Mono8, 8, 8> = ImageArray::generate(|x, y| Mono8::new((x + y * 8) as u8));
        let tiles: Vec<_> = img.tiles(Size::new(4, 4)).collect();
        // 8x8 image with 4x4 tiles = 4 tiles
        assert_eq!(tiles.len(), 4);
        assert_eq!(tiles[0].size(), Size::new(4, 4));
    }

    #[test]
    fn test_tiles_iter_imagearray_partial() {
        let img: ImageArray<Mono8, 7, 7> = ImageArray::generate(|x, y| Mono8::new((x + y * 7) as u8));
        let tiles: Vec<_> = img.tiles(Size::new(3, 3)).collect();
        // 7x7 image with 3x3 tiles: positions 0,3,6 in both dimensions = 9 tiles
        assert_eq!(tiles.len(), 9);

        // Last tile in first row at (6,0) should be 1x3
        assert_eq!(tiles[2].size(), Size::new(1, 3));

        // Bottom-right tile at (6,6) should be 1x1
        assert_eq!(tiles[8].size(), Size::new(1, 1));
    }

    #[test]
    fn test_sub_view_trait() {
        let img: Image<Mono8> = Image::generate(4, 4, |x, y| Mono8::new((x + y * 4) as u8));
        let roi = img.roi(Rectangle::new((1, 1), (2, 2)));
        assert!(roi.is_some());

        let roi = roi.unwrap();
        assert_eq!(roi.size(), Size::new(2, 2));
    }

    #[test]
    fn test_sub_view_mut_trait() {
        let mut img: Image<Mono8> = Image::generate(4, 4, |x, y| Mono8::new((x + y * 4) as u8));
        let roi = img.roi_mut(Rectangle::new((1, 1), (2, 2)));
        assert!(roi.is_some());

        let roi = roi.unwrap();
        assert_eq!(roi.size(), Size::new(2, 2));
    }

    #[test]
    fn test_tiles_iter_exhaustion() {
        let img: Image<Mono8> = Image::generate(4, 4, |x, y| Mono8::new((x + y * 4) as u8));
        let mut iter = img.tiles(Size::new(2, 2));

        // Consume all tiles (should be exactly 4)
        let mut count = 0;
        while iter.next().is_some() {
            count += 1;
        }
        assert_eq!(count, 4);

        // Iterator should remain exhausted
        assert!(iter.next().is_none());
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_tiles_single_pixel() {
        // Edge case: 1x1 tiles
        let img: Image<Mono8> = Image::generate(3, 3, |x, y| Mono8::new((x + y * 3) as u8));
        let tiles: Vec<_> = img.tiles(Size::new(1, 1)).collect();
        // Should get 9 tiles (3x3 grid of single pixels)
        assert_eq!(tiles.len(), 9);
        for tile in &tiles {
            assert_eq!(tile.size(), Size::new(1, 1));
        }
    }

    #[test]
    fn test_tiles_single_row() {
        // Edge case: tiles that span full height
        let img: Image<Mono8> = Image::generate(7, 4, |x, y| Mono8::new((x + y * 7) as u8));
        let tiles: Vec<_> = img.tiles(Size::new(3, 4)).collect();
        // 7-wide with 3-wide tiles: positions 0, 3, 6 = 3 tiles
        assert_eq!(tiles.len(), 3);
        assert_eq!(tiles[0].size(), Size::new(3, 4));
        assert_eq!(tiles[1].size(), Size::new(3, 4));
        assert_eq!(tiles[2].size(), Size::new(1, 4)); // partial width
    }

    #[test]
    fn test_tiles_single_column() {
        // Edge case: tiles that span full width
        let img: Image<Mono8> = Image::generate(4, 7, |x, y| Mono8::new((x + y * 4) as u8));
        let tiles: Vec<_> = img.tiles(Size::new(4, 3)).collect();
        // 7-tall with 3-tall tiles: positions 0, 3, 6 = 3 tiles
        assert_eq!(tiles.len(), 3);
        assert_eq!(tiles[0].size(), Size::new(4, 3));
        assert_eq!(tiles[1].size(), Size::new(4, 3));
        assert_eq!(tiles[2].size(), Size::new(4, 1)); // partial height
    }

    // ───────────────────────────────────────────────────────────────────
    // Coverage: Immutable tiles with Mono8 on ImageArray
    // (exercises a different monomorphization of TileIter + ImageRef)
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_tiles_iter_imagearray_mono8() {
        use crate::pixel::Mono8;
        let img: ImageArray<Mono8, 6, 4> =
            ImageArray::generate(|x, y| Mono8::new((x + y * 6) as u8));
        let tiles: Vec<_> = img.tiles(Size::new(3, 2)).collect();
        assert_eq!(tiles.len(), 4);
        // Top-left tile
        assert_eq!(tiles[0].size(), Size::new(3, 2));
        assert_eq!(tiles[0].get(0, 0), Some(Mono8::new(0)));
        assert_eq!(tiles[0].get(2, 1), Some(Mono8::new(8)));
        assert_eq!(tiles[0].pixel_at(1, 0), Mono8::new(1));
        assert_eq!(tiles[0].width(), 3);
        assert_eq!(tiles[0].height(), 2);
        // Top-right tile
        assert_eq!(tiles[1].get(0, 0), Some(Mono8::new(3)));
        // Bottom-left tile
        assert_eq!(tiles[2].get(0, 0), Some(Mono8::new(12)));
        // Bottom-right tile
        assert_eq!(tiles[3].get(0, 0), Some(Mono8::new(15)));
        // Out of bounds on tile
        assert_eq!(tiles[0].get(3, 0), None);
        assert_eq!(tiles[0].get(0, 2), None);
    }

    #[test]
    fn test_sub_view_imagearray_mono8() {
        use crate::pixel::Mono8;
        let img: ImageArray<Mono8, 4, 4> =
            ImageArray::generate(|x, y| Mono8::new((x + y * 4) as u8));
        let roi = img.roi(Rectangle::new((1, 1), (2, 2)));
        assert!(roi.is_some());
        let roi = roi.unwrap();
        assert_eq!(roi.size(), Size::new(2, 2));
        assert_eq!(roi.width(), 2);
        assert_eq!(roi.height(), 2);
        assert_eq!(roi.get(0, 0), Some(Mono8::new(5)));
        assert_eq!(roi.pixel_at(1, 1), Mono8::new(10));
        assert_eq!(roi.get(2, 0), None);
    }

    #[test]
    fn test_sub_view_mut_imagearray_mono8() {
        use crate::pixel::Mono8;
        let mut img: ImageArray<Mono8, 4, 4> =
            ImageArray::generate(|x, y| Mono8::new((x + y * 4) as u8));
        let mut roi = img.roi_mut(Rectangle::new((1, 1), (2, 2))).unwrap();
        assert_eq!(roi.size(), Size::new(2, 2));
        assert_eq!(roi.get(0, 0), Some(Mono8::new(5)));
        *roi.pixel_at_mut(0, 0) = Mono8::new(99);
        assert_eq!(roi.get(0, 0), Some(Mono8::new(99)));
        assert_eq!(roi.get_mut(2, 0), None);
    }

    // ───────────────────────────────────────────────────────────────────
    // Mutable tile iterator tests (TileIterMut)
    // ───────────────────────────────────────────────────────────────────

    /// Helper: create a TileIterMut from a mutable Image.
    fn tiles_mut<T: Copy>(img: &mut Image<T>, tile_size: Size) -> TileIterMut<'_, T> {
        let image_size = img.size();
        let slice = img.as_mut_slice();
        let len = slice.len();
        let ptr = slice.as_mut_ptr();
        // SAFETY: ptr from valid &mut [T]; len matches image dimensions;
        // no other references exist (we hold &mut img).
        unsafe { TileIterMut::new(ptr, len, image_size, tile_size) }
    }

    #[test]
    fn test_tiles_mut_basic() {
        let mut img: Image<u8> = Image::generate(4, 4, |x, y| (x + y * 4) as u8 + 1);
        let tiles: Vec<_> = tiles_mut(&mut img, Size::new(2, 2)).collect();

        // 4x4 image with 2x2 tiles = 4 tiles
        assert_eq!(tiles.len(), 4);

        // Verify sizes
        for tile in &tiles {
            assert_eq!(tile.size(), Size::new(2, 2));
        }

        // Verify first tile data
        assert_eq!(tiles[0].get(0, 0), Some(1));
        assert_eq!(tiles[0].get(1, 0), Some(2));
        assert_eq!(tiles[0].get(0, 1), Some(5));
        assert_eq!(tiles[0].get(1, 1), Some(6));

        // Verify second tile data (top-right)
        assert_eq!(tiles[1].get(0, 0), Some(3));
        assert_eq!(tiles[1].get(1, 0), Some(4));

        // Verify third tile data (bottom-left)
        assert_eq!(tiles[2].get(0, 0), Some(9));

        // Verify fourth tile data (bottom-right)
        assert_eq!(tiles[3].get(0, 0), Some(11));
        assert_eq!(tiles[3].get(1, 1), Some(16));
    }

    #[test]
    fn test_tiles_mut_write_and_verify() {
        let mut img: Image<u8> = Image::fill(4, 4, 0u8);
        {
            let mut tiles: Vec<_> = tiles_mut(&mut img, Size::new(2, 2)).collect();
            // Write distinct value per tile
            for (i, tile) in tiles.iter_mut().enumerate() {
                let val = (i as u8 + 1) * 10;
                for y in 0..tile.height() {
                    for x in 0..tile.width() {
                        *tile.pixel_at_mut(x, y) = val;
                    }
                }
            }
        }
        // Tile 0 = top-left 2x2 → value 10
        assert_eq!(img.get(0, 0), Some(10));
        assert_eq!(img.get(1, 1), Some(10));
        // Tile 1 = top-right 2x2 → value 20
        assert_eq!(img.get(2, 0), Some(20));
        assert_eq!(img.get(3, 1), Some(20));
        // Tile 2 = bottom-left 2x2 → value 30
        assert_eq!(img.get(0, 2), Some(30));
        assert_eq!(img.get(1, 3), Some(30));
        // Tile 3 = bottom-right 2x2 → value 40
        assert_eq!(img.get(2, 2), Some(40));
        assert_eq!(img.get(3, 3), Some(40));
    }

    #[test]
    fn test_tiles_mut_partial_edges() {
        let mut img: Image<u8> = Image::generate(10, 10, |x, y| (x + y * 10) as u8);
        let tiles: Vec<_> = tiles_mut(&mut img, Size::new(3, 3)).collect();

        // 10/3 → positions 0,3,6,9 in both dimensions = 4×4 = 16 tiles
        assert_eq!(tiles.len(), 16);

        // First tile at (0,0): full 3×3
        assert_eq!(tiles[0].size(), Size::new(3, 3));

        // Tile at (9,0): rightmost in first row → 1×3
        assert_eq!(tiles[3].size(), Size::new(1, 3));

        // Tile at (0,9): bottom-left → 3×1
        assert_eq!(tiles[12].size(), Size::new(3, 1));

        // Tile at (9,9): bottom-right corner → 1×1
        assert_eq!(tiles[15].size(), Size::new(1, 1));
    }

    #[test]
    fn test_tiles_mut_matches_immutable_sizes() {
        // Verify mutable and immutable iterators yield identical tile sizes
        let mut img: Image<Mono8> = Image::generate(7, 5, |x, y| Mono8::new((x + y * 7) as u8));
        let tile_size = Size::new(3, 2);

        let immut_sizes: Vec<Size> = img.tiles(tile_size).map(|t| t.size()).collect();
        let mut_sizes: Vec<Size> = tiles_mut(&mut img, tile_size).map(|t| t.size()).collect();

        assert_eq!(immut_sizes, mut_sizes);
    }

    #[test]
    fn test_tiles_mut_larger_than_image() {
        let mut img: Image<u8> = Image::generate(2, 2, |x, y| (x + y * 2) as u8);
        let tiles: Vec<_> = tiles_mut(&mut img, Size::new(5, 5)).collect();
        // Single tile clamped to image size
        assert_eq!(tiles.len(), 1);
        assert_eq!(tiles[0].size(), Size::new(2, 2));
    }

    #[test]
    fn test_tiles_mut_1x1_tiles() {
        let mut img: Image<u8> = Image::generate(3, 3, |x, y| (x + y * 3) as u8);
        let tiles: Vec<_> = tiles_mut(&mut img, Size::new(1, 1)).collect();
        // 9 single-pixel tiles
        assert_eq!(tiles.len(), 9);
        for tile in &tiles {
            assert_eq!(tile.size(), Size::new(1, 1));
        }
        // Verify each tile reads the correct pixel
        assert_eq!(tiles[0].get(0, 0), Some(0)); // (0,0)
        assert_eq!(tiles[1].get(0, 0), Some(1)); // (1,0)
        assert_eq!(tiles[4].get(0, 0), Some(4)); // (1,1)
        assert_eq!(tiles[8].get(0, 0), Some(8)); // (2,2)
    }

    #[test]
    fn test_tiles_mut_disjointness() {
        // Write a unique value to every pixel through tiles and verify
        // no pixel is missed or written twice.
        let mut img: Image<u8> = Image::fill(6, 4, 0u8);
        {
            let mut counter: u8 = 1;
            let mut tiles: Vec<_> = tiles_mut(&mut img, Size::new(3, 2)).collect();
            for tile in tiles.iter_mut() {
                for y in 0..tile.height() {
                    for x in 0..tile.width() {
                        *tile.pixel_at_mut(x, y) = counter;
                        counter += 1;
                    }
                }
            }
        }
        // Every pixel should have a unique non-zero value from 1..=24
        let slice = img.as_slice();
        assert_eq!(slice.len(), 24);
        // Collect into a set to verify uniqueness
        let mut seen = std::collections::HashSet::new();
        for &v in slice {
            assert_ne!(v, 0, "pixel was not written");
            assert!(seen.insert(v), "pixel value {} written twice", v);
        }
        assert_eq!(seen.len(), 24);
    }

    #[test]
    fn test_tiles_mut_mutation_roundtrip() {
        // Increment every pixel by 1 through tiles, verify entire image
        let mut img: Image<u8> = Image::generate(5, 5, |x, y| (x + y * 5) as u8);
        let expected: Vec<u8> = (0..25).map(|v: u8| v + 1).collect();
        {
            let mut tiles: Vec<_> = tiles_mut(&mut img, Size::new(2, 3)).collect();
            for tile in tiles.iter_mut() {
                for y in 0..tile.height() {
                    for x in 0..tile.width() {
                        let px = tile.pixel_at_mut(x, y);
                        *px += 1;
                    }
                }
            }
        }
        assert_eq!(img.as_slice(), &expected[..]);
    }

    #[test]
    fn test_tiles_mut_exhaustion() {
        let mut img: Image<u8> = Image::generate(4, 4, |x, y| (x + y * 4) as u8);
        let mut iter = tiles_mut(&mut img, Size::new(2, 2));

        let mut count = 0;
        while iter.next().is_some() {
            count += 1;
        }
        assert_eq!(count, 4);

        // Iterator should remain exhausted
        assert!(iter.next().is_none());
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_tiles_mut_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<TileIterMut<'_, u8>>();
        assert_sync::<TileIterMut<'_, u8>>();
        assert_send::<ImageRefMut<'_, u8>>();
        assert_sync::<ImageRefMut<'_, u8>>();
    }

    #[test]
    fn test_tiles_mut_imagearray() {
        let mut img: ImageArray<u8, 8, 8> = ImageArray::generate(|x, y| (x + y * 8) as u8);
        let slice = img.as_mut_slice();
        let len = slice.len();
        let ptr = slice.as_mut_ptr();
        let tiles: Vec<_> =
            unsafe { TileIterMut::new(ptr, len, Size::new(8, 8), Size::new(4, 4)) }.collect();
        assert_eq!(tiles.len(), 4);
        assert_eq!(tiles[0].size(), Size::new(4, 4));
        assert_eq!(tiles[0].get(0, 0), Some(0));
    }

    #[test]
    fn test_tiles_mut_partial_edges_data() {
        // Mirror the immutable test_tiles_iter_partial_edges_data
        let mut img: Image<u8> = Image::generate(5, 5, |x, y| (x + y * 5) as u8);
        let tiles: Vec<_> = tiles_mut(&mut img, Size::new(3, 3)).collect();

        assert_eq!(tiles.len(), 4);

        // Top-right tile at (3,0) should be 2×3
        assert_eq!(tiles[1].size(), Size::new(2, 3));
        assert_eq!(tiles[1].get(0, 0), Some(3));
        assert_eq!(tiles[1].get(1, 0), Some(4));

        // Bottom-left tile at (0,3) should be 3×2
        assert_eq!(tiles[2].size(), Size::new(3, 2));
        assert_eq!(tiles[2].get(0, 0), Some(15));

        // Bottom-right tile at (3,3) should be 2×2
        assert_eq!(tiles[3].size(), Size::new(2, 2));
        assert_eq!(tiles[3].get(0, 0), Some(18));
        assert_eq!(tiles[3].get(1, 1), Some(24));
    }

    #[test]
    fn test_tiles_mut_single_row_tiles() {
        let mut img: Image<u8> = Image::generate(7, 4, |x, y| (x + y * 7) as u8);
        let tiles: Vec<_> = tiles_mut(&mut img, Size::new(3, 4)).collect();
        assert_eq!(tiles.len(), 3);
        assert_eq!(tiles[0].size(), Size::new(3, 4));
        assert_eq!(tiles[1].size(), Size::new(3, 4));
        assert_eq!(tiles[2].size(), Size::new(1, 4)); // partial width
    }

    #[test]
    fn test_tiles_mut_single_column_tiles() {
        let mut img: Image<u8> = Image::generate(4, 7, |x, y| (x + y * 4) as u8);
        let tiles: Vec<_> = tiles_mut(&mut img, Size::new(4, 3)).collect();
        assert_eq!(tiles.len(), 3);
        assert_eq!(tiles[0].size(), Size::new(4, 3));
        assert_eq!(tiles[1].size(), Size::new(4, 3));
        assert_eq!(tiles[2].size(), Size::new(4, 1)); // partial height
    }

    // ───────────────────────────────────────────────────────────────────
    // IntoTilesMut trait tests (Step 3)
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_into_tiles_mut_symmetry_with_into_tiles() {
        // For same image and tile size, verify into_tiles and into_tiles_mut
        // yield tiles with identical sizes and pixel values.
        let mut img: Image<Mono8> = Image::generate(7, 5, |x, y| Mono8::new((x + y * 7) as u8));
        let tile_size = Size::new(3, 2);

        let immut_sizes: Vec<Size> = img.tiles(tile_size).map(|t| t.size()).collect();
        let immut_values: Vec<Vec<Mono8>> = img
            .tiles(tile_size)
            .map(|t| {
                let mut vals = Vec::new();
                for y in 0..t.height() {
                    for x in 0..t.width() {
                        vals.push(t.pixel_at(x, y));
                    }
                }
                vals
            })
            .collect();

        let mut_tiles: Vec<_> = (&mut img).into_tiles_mut(tile_size).collect();
        let mut_sizes: Vec<Size> = mut_tiles.iter().map(|t| t.size()).collect();
        let mut_values: Vec<Vec<Mono8>> = mut_tiles
            .iter()
            .map(|t| {
                let mut vals = Vec::new();
                for y in 0..t.height() {
                    for x in 0..t.width() {
                        vals.push(t.pixel_at(x, y));
                    }
                }
                vals
            })
            .collect();

        assert_eq!(immut_sizes, mut_sizes);
        assert_eq!(immut_values, mut_values);
    }

    #[test]
    fn test_into_tiles_mut_image_6x4_3x2() {
        // 6×4 image, 3×2 tiles → 4 tiles. Write distinct values per tile, verify.
        let mut img: Image<u8> = Image::fill(6, 4, 0u8);
        {
            let mut tiles: Vec<_> = (&mut img).into_tiles_mut(Size::new(3, 2)).collect();
            assert_eq!(tiles.len(), 4);

            // All tiles should be full 3×2
            assert_eq!(tiles[0].size(), Size::new(3, 2));
            assert_eq!(tiles[1].size(), Size::new(3, 2));
            assert_eq!(tiles[2].size(), Size::new(3, 2));
            assert_eq!(tiles[3].size(), Size::new(3, 2));

            for (i, tile) in tiles.iter_mut().enumerate() {
                let val = (i as u8 + 1) * 10;
                for y in 0..tile.height() {
                    for x in 0..tile.width() {
                        *tile.pixel_at_mut(x, y) = val;
                    }
                }
            }
        }
        // Verify tile 0 (top-left 3×2)
        assert_eq!(img.get(0, 0), Some(10));
        assert_eq!(img.get(2, 1), Some(10));
        // Verify tile 1 (top-right 3×2)
        assert_eq!(img.get(3, 0), Some(20));
        assert_eq!(img.get(5, 1), Some(20));
        // Verify tile 2 (bottom-left 3×2)
        assert_eq!(img.get(0, 2), Some(30));
        assert_eq!(img.get(2, 3), Some(30));
        // Verify tile 3 (bottom-right 3×2)
        assert_eq!(img.get(3, 2), Some(40));
        assert_eq!(img.get(5, 3), Some(40));
    }

    #[test]
    fn test_into_tiles_mut_imagearray() {
        // Same test with ImageArray<Mono8, 6, 4>
        let mut img: ImageArray<Mono8, 6, 4> = ImageArray::generate(|x, y| Mono8::new((x + y * 6) as u8));
        let tile_size = Size::new(3, 2);

        // Verify sizes match immutable
        let immut_sizes: Vec<Size> = img.tiles(tile_size).map(|t| t.size()).collect();

        let tiles: Vec<_> = (&mut img).into_tiles_mut(tile_size).collect();
        let mut_sizes: Vec<Size> = tiles.iter().map(|t| t.size()).collect();
        assert_eq!(immut_sizes, mut_sizes);
        assert_eq!(tiles.len(), 4);

        // Verify data
        assert_eq!(tiles[0].get(0, 0), Some(Mono8::new(0)));
        assert_eq!(tiles[1].get(0, 0), Some(Mono8::new(3)));
        assert_eq!(tiles[2].get(0, 0), Some(Mono8::new(12)));
        assert_eq!(tiles[3].get(0, 0), Some(Mono8::new(15)));
    }

    #[test]
    fn test_into_tiles_mut_imagearray_write_and_verify() {
        let mut img: ImageArray<u8, 6, 4> = ImageArray::generate(|_, _| 0u8);
        {
            let mut tiles: Vec<_> = (&mut img).into_tiles_mut(Size::new(3, 2)).collect();
            for (i, tile) in tiles.iter_mut().enumerate() {
                let val = (i as u8 + 1) * 10;
                for y in 0..tile.height() {
                    for x in 0..tile.width() {
                        *tile.pixel_at_mut(x, y) = val;
                    }
                }
            }
        }
        assert_eq!(img.get(0, 0), Some(10));
        assert_eq!(img.get(3, 0), Some(20));
        assert_eq!(img.get(0, 2), Some(30));
        assert_eq!(img.get(3, 2), Some(40));
    }

    #[test]
    fn test_into_tiles_mut_parallel_ready_simulation() {
        // Collect all mutable tiles into a Vec, write to each independently,
        // verify no data races (all writes visible, no corruption).
        // This simulates what rayon::par_iter would do.
        let mut img: Image<u8> = Image::fill(8, 6, 0u8);
        {
            let mut tiles: Vec<_> = (&mut img).into_tiles_mut(Size::new(4, 3)).collect();
            assert_eq!(tiles.len(), 4);

            // Simulate independent parallel writes: each tile gets a unique value
            for (i, tile) in tiles.iter_mut().enumerate() {
                let val = (i as u8) + 1;
                for y in 0..tile.height() {
                    for x in 0..tile.width() {
                        *tile.pixel_at_mut(x, y) = val;
                    }
                }
            }
        }
        // Verify: top-left quadrant = 1, top-right = 2, bottom-left = 3, bottom-right = 4
        for y in 0..6 {
            for x in 0..8 {
                let expected = match (x < 4, y < 3) {
                    (true, true) => 1,
                    (false, true) => 2,
                    (true, false) => 3,
                    (false, false) => 4,
                };
                assert_eq!(img.get(x, y), Some(expected), "mismatch at ({}, {})", x, y);
            }
        }
    }

    #[test]
    fn test_into_tiles_mut_image_partial_edges() {
        // Non-exact division: 10×10, 3×3 tiles
        let mut img: Image<u8> = Image::generate(10, 10, |x, y| (x + y * 10) as u8);
        let tiles: Vec<_> = (&mut img).into_tiles_mut(Size::new(3, 3)).collect();
        assert_eq!(tiles.len(), 16);

        assert_eq!(tiles[0].size(), Size::new(3, 3));
        assert_eq!(tiles[3].size(), Size::new(1, 3)); // rightmost col
        assert_eq!(tiles[12].size(), Size::new(3, 1)); // bottom row
        assert_eq!(tiles[15].size(), Size::new(1, 1)); // corner
    }

    #[test]
    fn test_into_tiles_mut_imagearray_partial_edges() {
        let mut img: ImageArray<u8, 7, 7> = ImageArray::generate(|x, y| (x + y * 7) as u8);
        let tiles: Vec<_> = (&mut img).into_tiles_mut(Size::new(3, 3)).collect();
        assert_eq!(tiles.len(), 9);
        assert_eq!(tiles[2].size(), Size::new(1, 3));
        assert_eq!(tiles[8].size(), Size::new(1, 1));
    }

    #[test]
    fn test_into_tiles_mut_larger_than_image() {
        let mut img: Image<u8> = Image::generate(2, 2, |x, y| (x + y * 2) as u8);
        let tiles: Vec<_> = (&mut img).into_tiles_mut(Size::new(10, 10)).collect();
        assert_eq!(tiles.len(), 1);
        assert_eq!(tiles[0].size(), Size::new(2, 2));
    }

    #[test]
    fn test_into_tiles_mut_disjointness_via_trait() {
        // Use the trait (not the helper) and verify every pixel written exactly once
        let mut img: Image<u8> = Image::fill(5, 5, 0u8);
        {
            let mut counter: u8 = 1;
            let mut tiles: Vec<_> = (&mut img).into_tiles_mut(Size::new(2, 3)).collect();
            for tile in tiles.iter_mut() {
                for y in 0..tile.height() {
                    for x in 0..tile.width() {
                        *tile.pixel_at_mut(x, y) = counter;
                        counter += 1;
                    }
                }
            }
        }
        let slice = img.as_slice();
        assert_eq!(slice.len(), 25);
        let mut seen = std::collections::HashSet::new();
        for &v in slice {
            assert_ne!(v, 0, "pixel was not written");
            assert!(seen.insert(v), "pixel value {} written twice", v);
        }
        assert_eq!(seen.len(), 25);
    }

    #[test]
    fn test_into_tiles_mut_mutation_roundtrip_via_trait() {
        // Increment every pixel by 1 through tiles via the trait, verify entire image
        let mut img: Image<u8> = Image::generate(6, 4, |x, y| (x + y * 6) as u8);
        let expected: Vec<u8> = (0..24).map(|v: u8| v + 1).collect();
        {
            let mut tiles: Vec<_> = (&mut img).into_tiles_mut(Size::new(3, 2)).collect();
            for tile in tiles.iter_mut() {
                for y in 0..tile.height() {
                    for x in 0..tile.width() {
                        let px = tile.pixel_at_mut(x, y);
                        *px += 1;
                    }
                }
            }
        }
        assert_eq!(img.as_slice(), &expected[..]);
    }

    // ───────────────────────────────────────────────────────────────────
    // SubView::into_tiles provided method tests
    // (verifies the unified API after "Operation Great SubView Unification")
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_subview_into_tiles_called_on_owned_image() {
        // tiles takes &self, so calling on an owned Image auto-borrows
        let img: Image<Mono8> = Image::generate(6, 4, |x, y| Mono8::new((x + y * 6) as u8));
        let tiles: Vec<_> = img.tiles(Size::new(3, 2)).collect();
        assert_eq!(tiles.len(), 4);
        assert_eq!(tiles[0].size(), Size::new(3, 2));
        assert_eq!(tiles[0].get(0, 0), Some(Mono8::new(0)));
        assert_eq!(tiles[1].get(0, 0), Some(Mono8::new(3)));
        assert_eq!(tiles[2].get(0, 0), Some(Mono8::new(12)));
        assert_eq!(tiles[3].get(0, 0), Some(Mono8::new(15)));
    }

    #[test]
    fn test_subview_into_tiles_called_on_ref() {
        // Explicitly calling on a reference — same result
        let img: Image<Mono8> = Image::generate(6, 4, |x, y| Mono8::new((x + y * 6) as u8));
        let img_ref = &img;
        let tiles: Vec<_> = img_ref.tiles(Size::new(3, 2)).collect();
        assert_eq!(tiles.len(), 4);
        assert_eq!(tiles[0].size(), Size::new(3, 2));
        assert_eq!(tiles[0].get(0, 0), Some(Mono8::new(0)));
    }

    #[test]
    fn test_subview_into_tiles_does_not_consume() {
        // tiles borrows, so the image is still usable afterwards
        let img: Image<Mono8> = Image::generate(4, 4, |x, y| Mono8::new((x + y * 4) as u8));
        let tiles1: Vec<_> = img.tiles(Size::new(2, 2)).collect();
        let tiles2: Vec<_> = img.tiles(Size::new(2, 2)).collect();
        // Both iterations yield identical results
        assert_eq!(tiles1.len(), tiles2.len());
        for (a, b) in tiles1.iter().zip(tiles2.iter()) {
            assert_eq!(a.size(), b.size());
            assert_eq!(a.get(0, 0), b.get(0, 0));
        }
        // Image is still accessible
        assert_eq!(img.get(0, 0), Some(Mono8::new(0)));
    }

    #[test]
    fn test_subview_into_tiles_imagearray() {
        // Works on ImageArray without any trait import beyond SubView
        let img: ImageArray<Mono8, 8, 6> = ImageArray::generate(|x, y| Mono8::new((x + y * 8) as u8));
        let tiles: Vec<_> = img.tiles(Size::new(4, 3)).collect();
        assert_eq!(tiles.len(), 4);
        assert_eq!(tiles[0].size(), Size::new(4, 3));
        assert_eq!(tiles[1].size(), Size::new(4, 3));
        assert_eq!(tiles[2].size(), Size::new(4, 3));
        assert_eq!(tiles[3].size(), Size::new(4, 3));
        // Check data in each tile
        assert_eq!(tiles[0].get(0, 0), Some(Mono8::new(0))); // (0,0)
        assert_eq!(tiles[1].get(0, 0), Some(Mono8::new(4))); // (4,0)
        assert_eq!(tiles[2].get(0, 0), Some(Mono8::new(24))); // (0,3)
        assert_eq!(tiles[3].get(0, 0), Some(Mono8::new(28))); // (4,3)
    }

    #[test]
    fn test_subview_into_tiles_partial_edges() {
        // Non-divisible dimensions produce partial edge tiles
        let img: Image<Mono8> = Image::generate(5, 5, |x, y| Mono8::new((x + y * 5) as u8));
        let tiles: Vec<_> = img.tiles(Size::new(3, 3)).collect();
        assert_eq!(tiles.len(), 4);
        assert_eq!(tiles[0].size(), Size::new(3, 3)); // (0,0)
        assert_eq!(tiles[1].size(), Size::new(2, 3)); // (3,0) partial width
        assert_eq!(tiles[2].size(), Size::new(3, 2)); // (0,3) partial height
        assert_eq!(tiles[3].size(), Size::new(2, 2)); // (3,3) partial both
    }

    #[test]
    fn test_subview_into_tiles_multiple_borrows_simultaneously() {
        // Multiple tile iterators can coexist since tiles only borrows
        let img: Image<Mono8> = Image::generate(4, 4, |x, y| Mono8::new((x + y * 4) as u8));
        let tiles_2x2: Vec<_> = img.tiles(Size::new(2, 2)).collect();
        let tiles_4x4: Vec<_> = img.tiles(Size::new(4, 4)).collect();
        assert_eq!(tiles_2x2.len(), 4);
        assert_eq!(tiles_4x4.len(), 1);
        // Both views see the same underlying data
        assert_eq!(tiles_2x2[0].get(0, 0), tiles_4x4[0].get(0, 0));
    }

    #[test]
    fn test_subview_into_tiles_symmetry_with_into_tiles_mut() {
        // Verify the unified into_tiles and separate IntoTilesMut
        // yield tiles with identical sizes and pixel values
        let mut img: Image<Mono8> = Image::generate(7, 5, |x, y| Mono8::new((x + y * 7) as u8));
        let tile_size = Size::new(3, 2);

        let immut_sizes: Vec<Size> = img.tiles(tile_size).map(|t| t.size()).collect();
        let immut_values: Vec<Vec<Mono8>> = img
            .tiles(tile_size)
            .map(|t| {
                let mut vals = Vec::new();
                for y in 0..t.height() {
                    for x in 0..t.width() {
                        vals.push(t.pixel_at(x, y));
                    }
                }
                vals
            })
            .collect();

        let mut_tiles: Vec<_> = (&mut img).into_tiles_mut(tile_size).collect();
        let mut_sizes: Vec<Size> = mut_tiles.iter().map(|t| t.size()).collect();
        let mut_values: Vec<Vec<Mono8>> = mut_tiles
            .iter()
            .map(|t| {
                let mut vals = Vec::new();
                for y in 0..t.height() {
                    for x in 0..t.width() {
                        vals.push(t.pixel_at(x, y));
                    }
                }
                vals
            })
            .collect();

        assert_eq!(immut_sizes, mut_sizes);
        assert_eq!(immut_values, mut_values);
    }

    // ── M3: zero tile sizes and zero strides are rejected at construction ──

    #[test]
    #[should_panic(expected = "tile size must be non-zero")]
    fn tile_iter_rejects_zero_width() {
        let img = Image::<Mono8>::zero(4, 4);
        let _ = img.tiles(Size::new(0, 2)).count();
    }

    #[test]
    #[should_panic(expected = "tile size must be non-zero")]
    fn tile_iter_rejects_zero_height() {
        let img = Image::<Mono8>::zero(4, 4);
        let _ = img.tiles(Size::new(2, 0)).count();
    }

    #[test]
    #[should_panic(expected = "stride must be non-zero")]
    fn sliding_window_iter_rejects_zero_horizontal_stride() {
        let img = Image::<Mono8>::zero(4, 4);
        let _ = SlidingWindow::new(Size::new(2, 2))
            .stride(Stride::new(0, 1))
            .iter(&img);
    }

    #[test]
    #[should_panic(expected = "stride must be non-zero")]
    fn sliding_window_iter_rejects_zero_vertical_stride() {
        let img = Image::<Mono8>::zero(4, 4);
        let _ = SlidingWindow::new(Size::new(2, 2))
            .stride(Stride::new(1, 0))
            .iter(&img);
    }

    #[test]
    #[should_panic(expected = "tile size must be non-zero")]
    fn tile_iter_mut_rejects_zero_tile_size() {
        let mut img = Image::<u8>::zero(4, 4);
        let _ = (&mut img).into_tiles_mut(Size::new(0, 2)).count();
    }
}

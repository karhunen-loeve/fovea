/// The `Size` struct represents the dimensions of an image.
///
/// # Example
/// ```
/// # use fovea::Size;
/// let size = Size::new(640, 480);
/// assert_eq!(size.area(), 640 * 480);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Size {
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
}
impl Size {
    /// Creates a `Size` with the given `width` and `height`.
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height }
    }
    /// Computes the area as `width * height`.
    ///
    /// # Panics
    /// Panics if the multiplication overflows `usize`. For untrusted or
    /// large dimensions prefer [`Self::checked_area`].
    pub fn area(&self) -> usize {
        self.width
            .checked_mul(self.height)
            .expect("Size::area: width * height overflows usize")
    }

    /// Computes `width * height`, returning `None` on overflow.
    ///
    /// Used by storage constructors that must validate buffer sizes
    /// without panicking on hostile input.
    #[inline]
    pub fn checked_area(&self) -> Option<usize> {
        self.width.checked_mul(self.height)
    }
}

impl From<(usize, usize)> for Size {
    fn from(value: (usize, usize)) -> Self {
        Self::new(value.0, value.1)
    }
}

/// The `Coordinate` struct represents a coordinate in 2D space with x and y coordinates.
///
/// # Example
/// ```
/// # use fovea::Coordinate;
/// let coordinate = Coordinate::new(10, 20);
/// assert_eq!(coordinate.x, 10);
/// assert_eq!(coordinate.y, 20);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Coordinate {
    /// Horizontal position.
    pub x: usize,
    /// Vertical position.
    pub y: usize,
}
impl Coordinate {
    /// Creates a `Coordinate` at the given `(x, y)` position.
    pub fn new(x: usize, y: usize) -> Self {
        Self { x, y }
    }
}

impl From<(usize, usize)> for Coordinate {
    fn from(value: (usize, usize)) -> Self {
        Self::new(value.0, value.1)
    }
}

/// The `Rectangle` struct represents a rectangle defined by an offset coordinate and size.
///
/// # Example
/// ```
/// # use fovea::{Coordinate, Size, Rectangle};
/// let rect = Rectangle::new(Coordinate::new(10, 20), Size::new(100, 50));
/// assert_eq!(rect.offset.x, 10);
/// assert_eq!(rect.offset.y, 20);
/// assert_eq!(rect.size.width, 100);
/// assert_eq!(rect.size.height, 50);
/// assert_eq!(rect.area(), 100 * 50);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rectangle {
    /// Top-left corner of the rectangle.
    pub offset: Coordinate,
    /// Width and height of the rectangle.
    pub size: Size,
}

impl Rectangle {
    /// Creates a `Rectangle` with the given top-left `offset` and `size`.
    pub fn new(offset: impl Into<Coordinate>, size: impl Into<Size>) -> Self {
        Self {
            offset: offset.into(),
            size: size.into(),
        }
    }
    /// Returns the area as `size.width * size.height`.
    pub fn area(&self) -> usize {
        self.size.area()
    }
    /// Returns the x-coordinate of the left edge (`offset.x`).
    pub fn left(&self) -> usize {
        self.offset.x
    }
    /// Returns the exclusive right edge `offset.x + size.width`.
    ///
    /// # Panics
    /// Panics on `usize` overflow. For untrusted geometry, use
    /// [`Self::checked_right`].
    pub fn right(&self) -> usize {
        self.offset
            .x
            .checked_add(self.size.width)
            .expect("Rectangle::right: offset.x + size.width overflows usize")
    }
    /// Returns the y-coordinate of the top edge (`offset.y`).
    pub fn top(&self) -> usize {
        self.offset.y
    }
    /// Returns the exclusive bottom edge `offset.y + size.height`.
    ///
    /// # Panics
    /// Panics on `usize` overflow. For untrusted geometry, use
    /// [`Self::checked_bottom`].
    pub fn bottom(&self) -> usize {
        self.offset
            .y
            .checked_add(self.size.height)
            .expect("Rectangle::bottom: offset.y + size.height overflows usize")
    }

    /// Returns `Some(offset.x + size.width)`, or `None` on overflow.
    #[inline]
    pub fn checked_right(&self) -> Option<usize> {
        self.offset.x.checked_add(self.size.width)
    }

    /// Returns `Some(offset.y + size.height)`, or `None` on overflow.
    #[inline]
    pub fn checked_bottom(&self) -> Option<usize> {
        self.offset.y.checked_add(self.size.height)
    }
}

/// A step size for sliding window iteration, wrapping a [`Size`].
///
/// `Stride` is a newtype around `Size` that represents how far a sliding
/// window advances between successive positions (horizontal and vertical
/// step). It exists to prevent accidental argument swapping between window
/// size and stride — both are `Size`-shaped, but mean different things.
///
/// # Example
///
/// ```
/// # use fovea::{Stride, Size};
/// // Explicit construction
/// let stride = Stride::new(2, 2);
/// assert_eq!(stride.horizontal(), 2);
/// assert_eq!(stride.vertical(), 2);
///
/// // From a Size
/// let stride = Stride::from(Size::new(3, 1));
/// assert_eq!(stride.horizontal(), 3);
/// assert_eq!(stride.vertical(), 1);
///
/// // From a tuple
/// let stride = Stride::from((4, 4));
/// assert_eq!(stride.horizontal(), 4);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stride(Size);

impl Stride {
    /// Creates a new `Stride` with the given horizontal and vertical step.
    pub fn new(horizontal: usize, vertical: usize) -> Self {
        Self(Size::new(horizontal, vertical))
    }

    /// Unit stride — advances by one pixel in each direction.
    pub fn one() -> Self {
        Self(Size::new(1, 1))
    }

    /// The horizontal step (number of pixels to advance in x).
    pub fn horizontal(&self) -> usize {
        self.0.width
    }

    /// The vertical step (number of pixels to advance in y).
    pub fn vertical(&self) -> usize {
        self.0.height
    }

    /// Returns the inner `Size`.
    pub fn as_size(&self) -> Size {
        self.0
    }
}

impl From<Size> for Stride {
    fn from(size: Size) -> Self {
        Self(size)
    }
}

impl From<(usize, usize)> for Stride {
    fn from(value: (usize, usize)) -> Self {
        Self(Size::new(value.0, value.1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_new() {
        let size = Size::new(640, 480);
        assert_eq!(size.width, 640);
        assert_eq!(size.height, 480);
    }

    #[test]
    fn test_size_area() {
        let size = Size::new(10, 20);
        assert_eq!(size.area(), 200);
    }

    #[test]
    fn test_size_from_tuple() {
        let size = Size::from((100, 200));
        assert_eq!(size.width, 100);
        assert_eq!(size.height, 200);
    }

    #[test]
    fn test_size_clone() {
        let size1 = Size::new(50, 60);
        let size2 = size1;
        assert_eq!(size1, size2);
    }

    #[test]
    fn test_size_copy() {
        let size1 = Size::new(50, 60);
        let size2 = size1; // Copy, not move
        assert_eq!(size1, size2); // size1 is still valid
        assert_eq!(size1.width, 50);
        assert_eq!(size1.height, 60);
    }

    #[test]
    fn test_coordinate_new() {
        let coord = Coordinate::new(10, 20);
        assert_eq!(coord.x, 10);
        assert_eq!(coord.y, 20);
    }

    #[test]
    fn test_coordinate_from_tuple() {
        let coord = Coordinate::from((15, 25));
        assert_eq!(coord.x, 15);
        assert_eq!(coord.y, 25);
    }

    #[test]
    fn test_coordinate_copy() {
        let coord1 = Coordinate::new(5, 10);
        let coord2 = coord1;
        assert_eq!(coord1, coord2);
    }

    #[test]
    fn test_rectangle_new() {
        let rect = Rectangle::new((10, 20), (100, 50));
        assert_eq!(rect.offset.x, 10);
        assert_eq!(rect.offset.y, 20);
        assert_eq!(rect.size.width, 100);
        assert_eq!(rect.size.height, 50);
    }

    #[test]
    fn test_rectangle_new_with_coordinate_and_size() {
        let rect = Rectangle::new(Coordinate::new(5, 15), Size::new(200, 100));
        assert_eq!(rect.offset.x, 5);
        assert_eq!(rect.offset.y, 15);
        assert_eq!(rect.size.width, 200);
        assert_eq!(rect.size.height, 100);
    }

    #[test]
    fn test_rectangle_area() {
        let rect = Rectangle::new((0, 0), (10, 20));
        assert_eq!(rect.area(), 200);
    }

    #[test]
    fn test_rectangle_left() {
        let rect = Rectangle::new((10, 20), (100, 50));
        assert_eq!(rect.left(), 10);
    }

    #[test]
    fn test_rectangle_right() {
        let rect = Rectangle::new((10, 20), (100, 50));
        assert_eq!(rect.right(), 110);
    }

    #[test]
    fn test_rectangle_top() {
        let rect = Rectangle::new((10, 20), (100, 50));
        assert_eq!(rect.top(), 20);
    }

    #[test]
    fn test_rectangle_bottom() {
        let rect = Rectangle::new((10, 20), (100, 50));
        assert_eq!(rect.bottom(), 70);
    }

    #[test]
    fn test_rectangle_clone() {
        let rect1 = Rectangle::new((5, 10), (50, 60));
        let rect2 = rect1;
        assert_eq!(rect1, rect2);
    }

    #[test]
    fn test_rectangle_copy() {
        let rect1 = Rectangle::new((5, 10), (50, 60));
        let rect2 = rect1; // Copy, not move
        assert_eq!(rect1, rect2); // rect1 is still valid
        assert_eq!(rect1.offset.x, 5);
        assert_eq!(rect1.size.width, 50);
    }

    #[test]
    fn test_rectangle_zero_area() {
        let rect = Rectangle::new((0, 0), (0, 0));
        assert_eq!(rect.area(), 0);
    }

    // ───────────────────────────────────────────────────────────────────
    // Stride tests
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_stride_new() {
        let s = Stride::new(3, 5);
        assert_eq!(s.horizontal(), 3);
        assert_eq!(s.vertical(), 5);
    }

    #[test]
    fn test_stride_one() {
        let s = Stride::one();
        assert_eq!(s.horizontal(), 1);
        assert_eq!(s.vertical(), 1);
    }

    #[test]
    fn test_stride_as_size() {
        let s = Stride::new(4, 7);
        let sz = s.as_size();
        assert_eq!(sz.width, 4);
        assert_eq!(sz.height, 7);
    }

    #[test]
    fn test_stride_from_size() {
        let sz = Size::new(2, 3);
        let s = Stride::from(sz);
        assert_eq!(s.horizontal(), 2);
        assert_eq!(s.vertical(), 3);
    }

    #[test]
    fn test_stride_from_tuple() {
        let s = Stride::from((10, 20));
        assert_eq!(s.horizontal(), 10);
        assert_eq!(s.vertical(), 20);
    }

    #[test]
    fn test_stride_copy() {
        let s1 = Stride::new(5, 6);
        let s2 = s1; // Copy
        assert_eq!(s1, s2);
        assert_eq!(s1.horizontal(), 5);
    }

    #[test]
    fn test_stride_clone() {
        let s1 = Stride::new(8, 9);
        let s2 = s1;
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_stride_debug() {
        let s = Stride::new(1, 2);
        let dbg = format!("{:?}", s);
        assert!(dbg.contains("Stride"));
    }

    #[test]
    fn test_stride_eq() {
        assert_eq!(Stride::new(3, 3), Stride::new(3, 3));
        assert_ne!(Stride::new(3, 3), Stride::new(3, 4));
        assert_ne!(Stride::new(3, 3), Stride::new(4, 3));
    }

    // ── B2: checked arithmetic for sizes and rectangles ──

    #[test]
    fn size_checked_area_returns_some_for_normal_values() {
        assert_eq!(Size::new(10, 20).checked_area(), Some(200));
        assert_eq!(Size::new(0, usize::MAX).checked_area(), Some(0));
    }

    #[test]
    fn size_checked_area_returns_none_on_overflow() {
        let huge = Size::new(usize::MAX, 2);
        assert!(huge.checked_area().is_none());
    }

    #[test]
    #[should_panic(expected = "overflow")]
    fn size_area_panics_on_overflow() {
        let huge = Size::new(usize::MAX, 2);
        let _ = huge.area();
    }

    #[test]
    fn rectangle_checked_right_and_bottom() {
        let r = Rectangle::new((10, 20), (100, 50));
        assert_eq!(r.checked_right(), Some(110));
        assert_eq!(r.checked_bottom(), Some(70));

        let r2 = Rectangle::new((usize::MAX - 1, 0), (10, 1));
        assert!(r2.checked_right().is_none());

        let r3 = Rectangle::new((0, usize::MAX - 1), (1, 10));
        assert!(r3.checked_bottom().is_none());
    }

    #[test]
    #[should_panic(expected = "overflow")]
    fn rectangle_right_panics_on_overflow() {
        let r = Rectangle::new((usize::MAX - 1, 0), (10, 1));
        let _ = r.right();
    }

    #[test]
    #[should_panic(expected = "overflow")]
    fn rectangle_bottom_panics_on_overflow() {
        let r = Rectangle::new((0, usize::MAX - 1), (1, 10));
        let _ = r.bottom();
    }
}

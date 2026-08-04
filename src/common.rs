use crate::error::Error;

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

/// A sub-pixel coordinate in 2D space with `f64` `x` and `y`.
///
/// The floating-point companion to [`Coordinate`], for quantities that
/// fall between pixel centres — a component centroid, a refined feature
/// location, an interpolated sample point.
///
/// # Example
/// ```
/// # use fovea::CoordinateF64;
/// let c = CoordinateF64::new(2.5, 4.0);
/// assert_eq!(c.x, 2.5);
/// assert_eq!(c.y, 4.0);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoordinateF64 {
    /// Horizontal position.
    pub x: f64,
    /// Vertical position.
    pub y: f64,
}
impl CoordinateF64 {
    /// Creates a `CoordinateF64` at the given `(x, y)` position.
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

impl From<(f64, f64)> for CoordinateF64 {
    fn from(value: (f64, f64)) -> Self {
        Self::new(value.0, value.1)
    }
}

impl From<Coordinate> for CoordinateF64 {
    fn from(value: Coordinate) -> Self {
        Self::new(value.x as f64, value.y as f64)
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

/// A validated Gaussian σ: finite and strictly positive.
///
/// `Sigma` is an invariant-carrying parameter type: the validation
/// happens once, where the value is born, and every function taking a
/// `Sigma` is total in it — the same idea as `std::num::NonZeroUsize`.
///
/// - Literals use [`Sigma::new`], a `const fn`: in a `const` context an
///   invalid literal fails to **compile**; at runtime it panics on first
///   execution (a deterministic programmer error, not a data condition).
/// - Values computed from data (an estimator, a scale-space formula, an
///   image statistic) use [`Sigma::try_new`] and handle the error where
///   the computation produced the bad value.
///
/// # Example
///
/// ```
/// use fovea::Sigma;
///
/// const BLUR: Sigma = Sigma::new(1.4); // checked at compile time
///
/// let estimated = 0.8_f32 * 2.0;
/// let sigma = Sigma::try_new(estimated)?; // checked where it is computed
/// assert_eq!(sigma.get(), 1.6);
/// # Ok::<(), fovea::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Sigma(f32);

impl Sigma {
    /// Creates a `Sigma` from a literal or otherwise proven-valid value.
    ///
    /// # Panics
    ///
    /// Panics if `value` is not finite and strictly positive. As a
    /// `const fn`, this is a **compile error** when evaluated in a
    /// `const` context. For values computed from data, use
    /// [`Self::try_new`].
    #[must_use]
    pub const fn new(value: f32) -> Self {
        assert!(
            value.is_finite() && value > 0.0,
            "Sigma::new: sigma must be finite and positive"
        );
        Self(value)
    }

    /// Creates a `Sigma` from a computed value, validating it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidParameter`] if `value` is zero, negative,
    /// NaN, or infinite.
    pub fn try_new(value: f32) -> Result<Self, Error> {
        if value.is_finite() && value > 0.0 {
            Ok(Self(value))
        } else {
            Err(Error::InvalidParameter(format!(
                "sigma must be finite and positive, got {value}"
            )))
        }
    }

    /// Returns the raw value.
    #[must_use]
    pub const fn get(self) -> f32 {
        self.0
    }
}

/// A validated sampling distance in base-image pixels: finite and
/// strictly positive.
///
/// Carries the [`Decimated`](crate::image::Decimated) grid spacing —
/// `2.0` for octave 1 of a 2× pyramid, `0.5` for an upsampled
/// octave −1. Same construction discipline as [`Sigma`]:
/// [`PixelDistance::new`] (const, panics — a compile error in `const`
/// contexts) for literals, [`PixelDistance::try_new`] for values derived
/// from a decimation chain.
///
/// # Example
///
/// ```
/// use fovea::PixelDistance;
///
/// const OCTAVE_1: PixelDistance = PixelDistance::new(2.0);
/// assert_eq!(OCTAVE_1.get(), 2.0);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct PixelDistance(f64);

impl PixelDistance {
    /// Creates a `PixelDistance` from a literal or otherwise proven-valid
    /// value.
    ///
    /// # Panics
    ///
    /// Panics if `value` is not finite and strictly positive. As a
    /// `const fn`, this is a **compile error** when evaluated in a
    /// `const` context. For values computed from data, use
    /// [`Self::try_new`].
    #[must_use]
    pub const fn new(value: f64) -> Self {
        assert!(
            value.is_finite() && value > 0.0,
            "PixelDistance::new: pixel distance must be finite and positive"
        );
        Self(value)
    }

    /// Creates a `PixelDistance` from a computed value, validating it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidParameter`] if `value` is zero, negative,
    /// NaN, or infinite.
    pub fn try_new(value: f64) -> Result<Self, Error> {
        if value.is_finite() && value > 0.0 {
            Ok(Self(value))
        } else {
            Err(Error::InvalidParameter(format!(
                "pixel distance must be finite and positive, got {value}"
            )))
        }
    }

    /// Returns the raw value.
    #[must_use]
    pub const fn get(self) -> f64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigma_valid_values_round_trip() {
        assert_eq!(Sigma::new(1.4).get(), 1.4);
        assert_eq!(Sigma::try_new(0.5).unwrap().get(), 0.5);
        // Const construction: an invalid literal here would not compile.
        const S: Sigma = Sigma::new(2.0);
        assert_eq!(S.get(), 2.0);
    }

    #[test]
    fn sigma_try_new_rejects_invalid_values() {
        // Zero, negative, NaN, infinite: each can flow out of a
        // computation over data, so each is an error value.
        for value in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let err = Sigma::try_new(value).unwrap_err();
            match err {
                Error::InvalidParameter(reason) => assert!(
                    reason.contains("sigma"),
                    "reason {reason:?} does not mention sigma"
                ),
                other => panic!("expected InvalidParameter, got {other:?}"),
            }
        }
    }

    #[test]
    #[should_panic(expected = "finite and positive")]
    fn sigma_new_panics_on_invalid_literal() {
        let _ = Sigma::new(-1.5);
    }

    #[test]
    fn pixel_distance_valid_values_round_trip() {
        assert_eq!(PixelDistance::new(2.0).get(), 2.0);
        assert_eq!(PixelDistance::try_new(0.5).unwrap().get(), 0.5);
        const D: PixelDistance = PixelDistance::new(0.5);
        assert_eq!(D.get(), 0.5);
    }

    #[test]
    fn pixel_distance_try_new_rejects_invalid_values() {
        for value in [0.0, -2.0, f64::NAN, f64::INFINITY] {
            let err = PixelDistance::try_new(value).unwrap_err();
            match err {
                Error::InvalidParameter(reason) => assert!(
                    reason.contains("pixel distance"),
                    "reason {reason:?} does not mention pixel distance"
                ),
                other => panic!("expected InvalidParameter, got {other:?}"),
            }
        }
    }

    #[test]
    #[should_panic(expected = "finite and positive")]
    fn pixel_distance_new_panics_on_invalid_literal() {
        let _ = PixelDistance::new(0.0);
    }

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

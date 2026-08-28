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
    #[inline]
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height }
    }
    /// Computes the area as `width * height`.
    ///
    /// # Panics
    /// Panics if the multiplication overflows `usize`. For untrusted or
    /// large dimensions prefer [`Self::checked_area`].
    #[inline]
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
    #[inline]
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
    #[inline]
    pub fn new(x: usize, y: usize) -> Self {
        Self { x, y }
    }

    /// Applies an [`Offset`], returning `None` if the result would leave
    /// the non-negative quadrant.
    ///
    /// This is the lower half of a neighbourhood bounds check, and it is
    /// the only one the crate open-codes. The upper half needs no new
    /// concept: [`ImageView::get`](crate::image::ImageView::get) already
    /// returns `Option`, so the whole check composes from two operations
    /// that each say what they mean.
    ///
    /// # Example
    /// ```
    /// # use fovea::{Coordinate, Offset};
    /// let c = Coordinate::new(0, 4);
    /// assert_eq!(c.checked_add(Offset::new(2, -1)), Some(Coordinate::new(2, 3)));
    /// assert_eq!(c.checked_add(Offset::new(-1, 0)), None); // off the left edge
    /// ```
    #[inline]
    #[must_use]
    pub const fn checked_add(self, offset: Offset) -> Option<Self> {
        match (
            self.x.checked_add_signed(offset.dx as isize),
            self.y.checked_add_signed(offset.dy as isize),
        ) {
            (Some(x), Some(y)) => Some(Self { x, y }),
            _ => None,
        }
    }

    /// The [`Offset`] that carries `self` to `other`.
    ///
    /// The inverse of [`checked_add`](Self::checked_add) for any pair
    /// whose separation fits an `i32`, which is every pair of positions in
    /// an image this crate can hold in memory. Components beyond that
    /// range **saturate** rather than wrapping, so a nonsensical input
    /// stays ordered instead of changing sign.
    ///
    /// # Example
    /// ```
    /// # use fovea::{Coordinate, Offset};
    /// let a = Coordinate::new(4, 4);
    /// let b = Coordinate::new(5, 3);
    /// assert_eq!(a.offset_to(b), Offset::new(1, -1));
    /// assert_eq!(a.checked_add(a.offset_to(b)), Some(b));
    /// ```
    #[inline]
    #[must_use]
    pub fn offset_to(self, other: Self) -> Offset {
        #[inline]
        fn delta(from: usize, to: usize) -> i32 {
            if to >= from {
                i32::try_from(to - from).unwrap_or(i32::MAX)
            } else {
                i32::try_from(from - to).map_or(i32::MIN, |d| -d)
            }
        }
        Offset::new(delta(self.x, other.x), delta(self.y, other.y))
    }
}

impl From<(usize, usize)> for Coordinate {
    #[inline]
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
    #[inline]
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

impl From<(f64, f64)> for CoordinateF64 {
    #[inline]
    fn from(value: (f64, f64)) -> Self {
        Self::new(value.0, value.1)
    }
}

impl From<Coordinate> for CoordinateF64 {
    #[inline]
    fn from(value: Coordinate) -> Self {
        Self::new(value.x as f64, value.y as f64)
    }
}

/// A signed step on the pixel grid: the displacement from one
/// [`Coordinate`] to another.
///
/// `Offset` is the crate's vocabulary for "a neighbour is this way" —
/// a connectivity neighbourhood, a detector's sampling ring, a chain-code
/// direction, a filter tap. All of those are the same concept, and before
/// this type existed they were spelled as bare tuples in four widths.
///
/// It is **not** an invariant-carrying parameter type like [`Sigma`]:
/// every `(dx, dy)` pair is a meaningful step, so there is nothing to
/// validate and no `try_new`. What it buys is the field names. A
/// positional pair lets `from_offset(dy, dx)` compile and quietly answer
/// the wrong question, and a transposed filter tap is a different kernel
/// that still type-checks; neither survives named `dx` and `dy`.
///
/// Deliberately not an arithmetic type. There is no `Add`, no `Neg` and no
/// `From<(i32, i32)>` — the last of those would hand the positional
/// hazard straight back. The one operation is
/// [`Coordinate::checked_add`], where the widening and the
/// non-negative check happen together.
///
/// `i32` because ring and kernel radii are single digits, and because it
/// was already the majority spelling of the four.
///
/// # Example
/// ```
/// # use fovea::{Coordinate, Offset};
/// const NORTH: Offset = Offset::new(0, -1); // y grows downward
/// assert_eq!(NORTH.dy, -1);
///
/// // Applying one is checked, so the image edge is not a special case.
/// assert_eq!(Coordinate::new(3, 3).checked_add(NORTH), Some(Coordinate::new(3, 2)));
/// assert_eq!(Coordinate::new(3, 0).checked_add(NORTH), None);
/// ```
///
/// # What will not compile
///
/// There is no conversion from a pair, and that is the point: the pair is
/// the shape a transposed step slips through.
///
/// ```compile_fail
/// use fovea::{Coordinate, Offset};
///
/// // ERROR: the trait bound `Offset: From<(i32, i32)>` is not satisfied.
/// let step: Offset = (0, -1).into();
/// let _ = Coordinate::new(3, 3).checked_add(step);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Offset {
    /// Horizontal step, positive to the right.
    pub dx: i32,
    /// Vertical step, positive **downward** — image coordinates.
    pub dy: i32,
}

impl Offset {
    /// The zero step: the site itself.
    pub const ZERO: Self = Self::new(0, 0);

    /// Creates an `Offset` of `dx` columns and `dy` rows.
    #[inline]
    #[must_use]
    pub const fn new(dx: i32, dy: i32) -> Self {
        Self { dx, dy }
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
    #[inline]
    pub fn new(offset: impl Into<Coordinate>, size: impl Into<Size>) -> Self {
        Self {
            offset: offset.into(),
            size: size.into(),
        }
    }
    /// Returns the area as `size.width * size.height`.
    #[inline]
    pub fn area(&self) -> usize {
        self.size.area()
    }
    /// Returns the x-coordinate of the left edge (`offset.x`).
    #[inline]
    pub fn left(&self) -> usize {
        self.offset.x
    }
    /// Returns the exclusive right edge `offset.x + size.width`.
    ///
    /// # Panics
    /// Panics on `usize` overflow. For untrusted geometry, use
    /// [`Self::checked_right`].
    #[inline]
    pub fn right(&self) -> usize {
        self.offset
            .x
            .checked_add(self.size.width)
            .expect("Rectangle::right: offset.x + size.width overflows usize")
    }
    /// Returns the y-coordinate of the top edge (`offset.y`).
    #[inline]
    pub fn top(&self) -> usize {
        self.offset.y
    }
    /// Returns the exclusive bottom edge `offset.y + size.height`.
    ///
    /// # Panics
    /// Panics on `usize` overflow. For untrusted geometry, use
    /// [`Self::checked_bottom`].
    #[inline]
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
    #[inline]
    pub fn new(horizontal: usize, vertical: usize) -> Self {
        Self(Size::new(horizontal, vertical))
    }

    /// Unit stride — advances by one pixel in each direction.
    #[inline]
    pub fn one() -> Self {
        Self(Size::new(1, 1))
    }

    /// The horizontal step (number of pixels to advance in x).
    #[inline]
    pub fn horizontal(&self) -> usize {
        self.0.width
    }

    /// The vertical step (number of pixels to advance in y).
    #[inline]
    pub fn vertical(&self) -> usize {
        self.0.height
    }

    /// Returns the inner `Size`.
    #[inline]
    pub fn as_size(&self) -> Size {
        self.0
    }
}

impl From<Size> for Stride {
    #[inline]
    fn from(size: Size) -> Self {
        Self(size)
    }
}

impl From<(usize, usize)> for Stride {
    #[inline]
    fn from(value: (usize, usize)) -> Self {
        Self(Size::new(value.0, value.1))
    }
}

/// A validated Gaussian σ: finite and strictly positive.
///
/// `Sigma` is an invariant-carrying parameter type: the validation
/// happens once, where the value is born, and every function taking a
/// `Sigma` is total in it, the same idea as `std::num::NonZeroUsize`.
///
/// - Literals use the [`sigma!`](crate::sigma) macro. It is an inline
///   `const` block, so an invalid literal is a **compile error** and a
///   runtime value does not type-check at all.
/// - Values computed from data (an estimator, a scale-space formula, an
///   image statistic) use [`Sigma::try_new`] and handle the error where
///   the computation produced the bad value.
/// - [`Sigma::new`] is the checked `const fn` underneath the macro. It
///   returns [`Option`], so reaching for it by name cannot abort.
///
/// # Example
///
/// ```
/// use fovea::{Sigma, sigma};
///
/// const BLUR: Sigma = sigma!(1.4); // checked at compile time
///
/// let estimated = 0.8_f32 * 2.0;
/// let sigma = Sigma::try_new(estimated)?; // checked where it is computed
/// assert_eq!(sigma.get(), 1.6);
/// # Ok::<(), fovea::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Sigma(f32);

impl Sigma {
    /// Creates a `Sigma`, returning `None` if the value is not finite and
    /// strictly positive.
    ///
    /// This is the `const fn` the [`sigma!`](crate::sigma) macro wraps.
    /// Prefer the macro for literals, since it moves the check to compile
    /// time, and [`Self::try_new`] for values computed from data, since it
    /// reports a reason.
    #[must_use]
    pub const fn new(value: f32) -> Option<Self> {
        if value.is_finite() && value > 0.0 {
            Some(Self(value))
        } else {
            None
        }
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

/// A [`Sigma`] literal, checked at compile time.
///
/// Expands to an inline `const` block, which has two consequences worth
/// knowing before reaching for it:
///
/// - An invalid literal is a **compile error**, not a runtime abort. This
///   is the reason the macro exists: a `const fn` would be checked at
///   compile time only when the compiler happens to evaluate it there.
/// - A value that is not a constant expression does not compile
///   (`error[E0435]`). For those, use [`Sigma::try_new`], which reports a
///   reason the caller can act on.
///
/// # Example
///
/// ```
/// use fovea::{Sigma, sigma};
/// use fovea::transform::gaussian_blur;
/// # use fovea::border::Clamp;
/// # use fovea::image::Image;
/// # use fovea::pixel::MonoF32;
///
/// // In argument position, where the macro is shortest.
/// # let img: Image<MonoF32> = Image::fill(8, 8, MonoF32::new(0.5));
/// let blurred: Image<MonoF32> = gaussian_blur(&img, sigma!(1.4), &Clamp);
///
/// // And in a `const` item.
/// const BLUR: Sigma = sigma!(1.4);
/// assert_eq!(BLUR.get(), 1.4);
/// ```
///
/// A σ of zero is not a blur, so it does not build:
///
/// ```compile_fail
/// use fovea::sigma;
/// // ERROR: evaluation panicked: sigma must be finite and strictly positive
/// let _ = sigma!(0.0);
/// ```
#[macro_export]
macro_rules! sigma {
    ($value:expr) => {
        const { $crate::Sigma::new($value).expect("sigma must be finite and strictly positive") }
    };
}

/// A validated sampling distance in base-image pixels: finite and
/// strictly positive.
///
/// Carries the [`Decimated`](crate::image::Decimated) grid spacing:
/// `2.0` for octave 1 of a 2× pyramid, `0.5` for an upsampled
/// octave −1. Same construction discipline as [`Sigma`]: the
/// [`pixel_distance!`](crate::pixel_distance) macro for literals,
/// [`PixelDistance::try_new`] for values derived from a decimation chain,
/// and [`PixelDistance::new`] as the checked `const fn` under the macro.
///
/// # Example
///
/// ```
/// use fovea::{PixelDistance, pixel_distance};
///
/// const OCTAVE_1: PixelDistance = pixel_distance!(2.0);
/// assert_eq!(OCTAVE_1.get(), 2.0);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct PixelDistance(f64);

impl PixelDistance {
    /// Creates a `PixelDistance`, returning `None` if the value is not
    /// finite and strictly positive.
    ///
    /// This is the `const fn` the
    /// [`pixel_distance!`](crate::pixel_distance) macro wraps. Prefer the
    /// macro for literals and [`Self::try_new`] for values computed from
    /// data.
    #[must_use]
    pub const fn new(value: f64) -> Option<Self> {
        if value.is_finite() && value > 0.0 {
            Some(Self(value))
        } else {
            None
        }
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

/// A [`PixelDistance`] literal, checked at compile time.
///
/// The [`sigma!`](crate::sigma) macro's counterpart for grid spacings; see
/// it for why this is a macro and not a `const fn`. A value that is not a
/// constant expression does not compile (`error[E0435]`); use
/// [`PixelDistance::try_new`] there.
///
/// # Example
///
/// ```
/// use fovea::{PixelDistance, pixel_distance};
///
/// const OCTAVE_1: PixelDistance = pixel_distance!(2.0);
/// assert_eq!(OCTAVE_1.get(), 2.0);
/// assert_eq!(pixel_distance!(0.5).get(), 0.5); // an upsampled octave
/// ```
///
/// A spacing of zero would collapse the grid, so it does not build:
///
/// ```compile_fail
/// use fovea::pixel_distance;
/// // ERROR: evaluation panicked: pixel distance must be finite and
/// // strictly positive
/// let _ = pixel_distance!(0.0);
/// ```
#[macro_export]
macro_rules! pixel_distance {
    ($value:expr) => {
        const {
            $crate::PixelDistance::new($value)
                .expect("pixel distance must be finite and strictly positive")
        }
    };
}

/// A validated geometric tolerance in pixels: finite and non-negative.
///
/// The maximum deviation a caller is willing to accept, e.g. the ε of
/// [`approximate_polygon`](crate::analyze::contours::approximate_polygon).
/// Unlike [`Sigma`] and [`PixelDistance`], **zero is a valid value**: a
/// tolerance of `0.0` accepts no deviation at all (for polygon
/// approximation, only exactly collinear vertices are removed). Same
/// construction discipline as the other parameter types: the
/// [`tolerance!`](crate::tolerance) macro for literals,
/// [`Tolerance::try_new`] for computed values, and [`Tolerance::new`] as
/// the checked `const fn` under the macro.
///
/// # Example
///
/// ```
/// use fovea::{Tolerance, tolerance};
///
/// const HALF_PIXEL: Tolerance = tolerance!(0.5);
/// assert_eq!(HALF_PIXEL.get(), 0.5);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Tolerance(f64);

impl Tolerance {
    /// Creates a `Tolerance`, returning `None` if the value is NaN,
    /// infinite, or negative.
    ///
    /// This is the `const fn` the [`tolerance!`](crate::tolerance) macro
    /// wraps. Prefer the macro for literals and [`Self::try_new`] for
    /// values computed from data.
    #[must_use]
    pub const fn new(value: f64) -> Option<Self> {
        if value.is_finite() && value >= 0.0 {
            Some(Self(value))
        } else {
            None
        }
    }

    /// Creates a `Tolerance` from a computed value, validating it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidParameter`] if `value` is NaN, infinite,
    /// or negative.
    pub fn try_new(value: f64) -> Result<Self, Error> {
        if value.is_finite() && value >= 0.0 {
            Ok(Self(value))
        } else {
            Err(Error::InvalidParameter(format!(
                "tolerance must be finite and non-negative, got {value}"
            )))
        }
    }

    /// Returns the raw value.
    #[must_use]
    pub const fn get(self) -> f64 {
        self.0
    }
}

/// A [`Tolerance`] literal, checked at compile time.
///
/// The [`sigma!`](crate::sigma) macro's counterpart for geometric
/// tolerances; see it for why this is a macro and not a `const fn`. A
/// value that is not a constant expression does not compile
/// (`error[E0435]`); use [`Tolerance::try_new`] there.
///
/// # Example
///
/// ```
/// use fovea::{Tolerance, tolerance};
///
/// const HALF_PIXEL: Tolerance = tolerance!(0.5);
/// assert_eq!(HALF_PIXEL.get(), 0.5);
/// assert_eq!(tolerance!(0.0).get(), 0.0); // zero accepts no deviation
/// ```
///
/// A negative tolerance accepts nothing at all, so it does not build:
///
/// ```compile_fail
/// use fovea::tolerance;
/// // ERROR: evaluation panicked: tolerance must be finite and non-negative
/// let _ = tolerance!(-0.5);
/// ```
#[macro_export]
macro_rules! tolerance {
    ($value:expr) => {
        const { $crate::Tolerance::new($value).expect("tolerance must be finite and non-negative") }
    };
}

/// A validated square-window side length: odd and non-zero.
///
/// The side of a neighbourhood centred on the pixel being processed, in
/// pixels: the `window` of
/// [`adaptive_threshold`](crate::analyze::threshold::adaptive_threshold).
/// An even side has no centre pixel and a zero side has no pixels at all,
/// so both are rejected; `1` is valid and degenerate (the window is the
/// pixel itself).
///
/// This is **one axis**, not a window. Whether a consumer applies it to
/// both axes is the consumer's choice: `adaptive_threshold` does, and
/// documents its neighbourhood as `window × window`. A window with
/// independent width and height is a [`Size`], which is what
/// [`SlidingWindow`](crate::image::SlidingWindow) takes.
///
/// Unlike [`Sigma`] and [`Tolerance`], whose invariants are inequalities
/// on a float, this one is a *parity* property, which is why the name
/// states it, the same choice `NonZeroUsize` makes. Same construction
/// discipline as the other parameter types: the
/// [`window!`](crate::window) macro for literals,
/// [`OddWindowSide::try_new`] for values computed from data, and
/// [`OddWindowSide::new`] as the checked `const fn` under the macro.
///
/// The side length rather than the radius is the wrapped quantity,
/// matching OpenCV's `blockSize` and scikit-image's `block_size`, so a
/// number read off a reference implementation transfers unchanged. Use
/// [`radius`](Self::radius) where the half-width is what the loop needs;
/// for an odd side it is exact.
///
/// # Example
///
/// ```
/// use fovea::{OddWindowSide, window};
///
/// const LOCAL: OddWindowSide = window!(15);
/// assert_eq!(LOCAL.get(), 15);
/// assert_eq!(LOCAL.radius(), 7);
///
/// // An even side is rejected where it is computed, not where it is used.
/// let from_data = 2 * 8;
/// assert!(OddWindowSide::try_new(from_data).is_err());
/// ```
///
/// An invalid literal does not compile at all, which is the point of the
/// macro:
///
/// ```compile_fail
/// use fovea::window;
/// // ERROR: evaluation panicked: window side must be odd and non-zero
/// let _ = window!(16);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OddWindowSide(usize);

impl OddWindowSide {
    /// Creates an `OddWindowSide`, returning `None` if `side` is zero or
    /// even.
    ///
    /// This is the `const fn` the [`window!`](crate::window) macro wraps.
    /// Prefer the macro for literals and [`Self::try_new`] for values
    /// computed from data.
    #[must_use]
    pub const fn new(side: usize) -> Option<Self> {
        if side != 0 && side % 2 == 1 {
            Some(Self(side))
        } else {
            None
        }
    }

    /// Creates an `OddWindowSide` from a computed value, validating it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidParameter`] if `side` is zero or even.
    pub fn try_new(side: usize) -> Result<Self, Error> {
        if side != 0 && side % 2 == 1 {
            Ok(Self(side))
        } else {
            Err(Error::InvalidParameter(format!(
                "window side must be odd and non-zero, got {side}"
            )))
        }
    }

    /// Returns the side length.
    #[must_use]
    pub const fn get(self) -> usize {
        self.0
    }

    /// Returns the half-width: `(side − 1) / 2`, exact because the side is
    /// odd.
    ///
    /// A window of side `s` centred on `(x, y)` spans
    /// `x − radius ..= x + radius` before any clipping.
    #[must_use]
    pub const fn radius(self) -> usize {
        self.0 / 2
    }
}

/// An [`OddWindowSide`] literal, checked at compile time.
///
/// Named for the call site rather than the type: what a caller writes is a
/// window size, and `window!(31)` beside a `bias` argument reads as one.
/// The type keeps the longer name because it is *one axis*, which is the
/// thing a reader of the signature has to know.
///
/// The [`sigma!`](crate::sigma) macro's counterpart for window sides; see
/// it for why this is a macro and not a `const fn`. A value that is not a
/// constant expression does not compile (`error[E0435]`); use
/// [`OddWindowSide::try_new`] there.
///
/// # Example
///
/// ```
/// use fovea::{OddWindowSide, window};
///
/// const LOCAL: OddWindowSide = window!(31);
/// assert_eq!(LOCAL.radius(), 15);
/// assert_eq!(window!(1).get(), 1); // valid and degenerate
/// ```
///
/// An even side has no centre pixel, so it does not build:
///
/// ```compile_fail
/// use fovea::window;
/// // ERROR: evaluation panicked: window side must be odd and non-zero
/// let _ = window!(16);
/// ```
#[macro_export]
macro_rules! window {
    ($side:expr) => {
        const { $crate::OddWindowSide::new($side).expect("window side must be odd and non-zero") }
    };
}

/// Canonicalizes a radian value into `(−π, π]`.
fn wrap_two_pi(radians: f32) -> f32 {
    const PI: f32 = core::f32::consts::PI;
    let wrapped = radians.rem_euclid(2.0 * PI); // [0, 2π)
    if wrapped > PI {
        wrapped - 2.0 * PI
    } else {
        wrapped
    }
}

/// Canonicalizes a radian value into `(−π/2, π/2]`.
fn wrap_pi(radians: f64) -> f64 {
    const PI: f64 = core::f64::consts::PI;
    let wrapped = radians.rem_euclid(PI); // [0, π)
    if wrapped > PI / 2.0 {
        wrapped - PI
    } else {
        wrapped
    }
}

/// A **direction** in the image plane: an angle modulo 2π, canonicalized to
/// `(−π, π]`.
///
/// Use this for quantities that distinguish a direction from its opposite —
/// a gradient direction, a dominant feature orientation. For an *undirected*
/// axis, where θ and θ + π mean the same thing, use [`AxialOrientation`].
/// The two are separate types precisely because they are not
/// interchangeable: they wrap at different moduli, so subtracting one from
/// the other is meaningless, and the type system is the only thing that can
/// say so.
///
/// Angles are measured from the +x axis in **image (y-down) coordinates**,
/// so a positive angle rotates toward +y — *downward* on screen. This is the
/// flip versus math-convention plots; read the sign accordingly.
///
/// # Why a type and not `f32`
///
/// A raw float leaves both the unit and — the load-bearing part — the
/// *modulus* unstated. Canonicalizing at construction is what makes `==`
/// mean "the same direction" instead of "the same float": without it, `0`
/// and `2π` compare unequal. The seam is also where hand-rolled arithmetic
/// goes wrong: directions at `179°` and `−179°` are `2°` apart, not `358°`,
/// which is why the subtraction lives in
/// [`signed_difference`](Self::signed_difference) rather than at call sites.
///
/// Canonicalization makes equality *meaningful*, not *exact* — wrapping
/// rounds, so two mathematically equal angles built by different routes can
/// still differ in the last bit. Compare with
/// `a.signed_difference(b).abs() < tolerance`, not `==`. There is
/// deliberately no `PartialOrd`: on a circle there is no least angle, and
/// `a < b` would invite reading "counter-clockwise of", which it is not.
///
/// # Example
///
/// ```
/// use fovea::Orientation;
///
/// // Any finite angle is valid — it wraps rather than being rejected.
/// let turned = Orientation::from_radians(3.0 * core::f32::consts::PI)?;
/// let half = Orientation::from_radians(core::f32::consts::PI)?;
/// // Compare by difference, not `==`: wrapping rounds, so these agree to
/// // within an ULP rather than bit-exactly.
/// assert!(turned.signed_difference(half).abs() < 1e-6);
///
/// // The ±π seam is 2 degrees wide, not 358.
/// let east = Orientation::from_radians(179_f32.to_radians())?;
/// let west = Orientation::from_radians((-179_f32).to_radians())?;
/// let apart = east.signed_difference(west).abs().to_degrees();
/// assert!((apart - 2.0).abs() < 1e-3, "got {apart}");
/// # Ok::<(), fovea::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Orientation(f32);

impl Orientation {
    /// Creates an orientation from radians, wrapping into `(−π, π]`.
    ///
    /// Unlike [`Sigma`] or [`PixelDistance`], whose invariants *reject*
    /// out-of-range input, this constructor **normalizes**: `7.0` radians
    /// is not an invalid angle, it is `0.717` radians. Only a value that
    /// cannot be canonicalized at all is an error.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidParameter`] if `radians` is NaN or infinite.
    pub fn from_radians(radians: f32) -> Result<Self, Error> {
        if radians.is_finite() {
            Ok(Self(wrap_two_pi(radians)))
        } else {
            Err(Error::InvalidParameter(format!(
                "orientation must be a finite angle in radians, got {radians}"
            )))
        }
    }

    /// Creates an orientation from a gradient vector, as `atan2(y, x)`.
    ///
    /// The dominant construction site, and **total**: nothing can fail.
    /// `atan2` returns `[−π, π]`, one value wider than the canonical range
    /// at the bottom — a negative-zero `y` beside a negative `x` yields
    /// exactly `−π` — so that single boundary value is folded to `+π`, the
    /// same angle spelled inside `(−π, π]`. (`atan2(0, 0)` is `0` — an
    /// arbitrary but defined direction for a zero-length vector, matching
    /// [`f32::atan2`].)
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::Orientation;
    ///
    /// // Gradient pointing along +x.
    /// assert_eq!(Orientation::from_atan2(0.0, 1.0).radians(), 0.0);
    /// ```
    #[must_use]
    pub fn from_atan2(y: f32, x: f32) -> Self {
        let angle = y.atan2(x);
        if angle == -core::f32::consts::PI {
            Self(core::f32::consts::PI)
        } else {
            Self(angle)
        }
    }

    /// Returns the angle in radians, in `(−π, π]`.
    #[must_use]
    pub const fn radians(self) -> f32 {
        self.0
    }

    /// Returns the signed angle **from `other` to `self`**, in `(−π, π]`.
    ///
    /// Wraps across the ±π seam, so the result is always the shorter of the
    /// two ways round and its magnitude never exceeds π.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::Orientation;
    ///
    /// let a = Orientation::from_radians(0.5)?;
    /// let b = Orientation::from_radians(0.2)?;
    /// assert!((a.signed_difference(b) - 0.3).abs() < 1e-6);
    /// # Ok::<(), fovea::Error>(())
    /// ```
    #[must_use]
    pub fn signed_difference(self, other: Self) -> f32 {
        wrap_two_pi(self.0 - other.0)
    }

    /// Discards the sense of direction, yielding the undirected axis this
    /// orientation lies along.
    ///
    /// Well-defined in this direction only: θ and θ + π collapse onto one
    /// axis. The reverse is not a function — an axis corresponds to *two*
    /// opposite directions — which is why [`AxialOrientation`] has no
    /// `to_directed`.
    ///
    /// Widening `f32` to `f64` moves the ±π/2 boundary by an ULP, so an
    /// input sitting exactly on it may be reported at either end of the
    /// canonical range (`+π/2` or `−π/2`). Both name the same axis, and
    /// [`AxialOrientation::signed_difference`] reads them as zero apart —
    /// but it is another reason to compare by difference rather than `==`.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::Orientation;
    ///
    /// let north = Orientation::from_radians(core::f32::consts::FRAC_PI_2)?;
    /// let south = Orientation::from_radians(-core::f32::consts::FRAC_PI_2)?;
    /// // Opposite directions, one axis.
    /// assert!(north.to_axial().signed_difference(south.to_axial()).abs() < 1e-6);
    /// # Ok::<(), fovea::Error>(())
    /// ```
    #[must_use]
    pub fn to_axial(self) -> AxialOrientation {
        AxialOrientation(wrap_pi(f64::from(self.0)))
    }
}

/// An **undirected axis** in the image plane: an angle modulo π,
/// canonicalized to `(−π/2, π/2]`.
///
/// Use this for quantities where an angle and its opposite are the same
/// thing — the major axis of a region's equivalent ellipse, an edge's
/// orientation. An ellipse's axis has no head and no tail, so `+80°` and
/// `−100°` are *the same axis*, and a type that wrapped at 2π would treat
/// them as nearly opposite.
///
/// The sibling type for directed quantities is [`Orientation`]. They are
/// deliberately not interchangeable, and the conversion runs one way only
/// ([`Orientation::to_axial`]): folding a direction onto an axis loses
/// information that cannot be recovered.
///
/// Angles are measured from the +x axis in **image (y-down) coordinates**,
/// so a positive angle rotates toward +y — *downward* on screen.
///
/// The same equality and ordering caveats as [`Orientation`] apply:
/// canonicalization makes `==` meaningful but not bit-exact, so compare via
/// [`signed_difference`](Self::signed_difference); and there is no
/// `PartialOrd`.
///
/// # Example
///
/// ```
/// use fovea::AxialOrientation;
///
/// let a = AxialOrientation::from_radians(80_f64.to_radians())?;
/// let b = AxialOrientation::from_radians((-100_f64).to_radians())?;
/// // Same axis, reached from opposite directions.
/// assert!(a.signed_difference(b).abs() < 1e-12);
/// # Ok::<(), fovea::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AxialOrientation(f64);

impl AxialOrientation {
    /// Creates an axis orientation from radians, wrapping into
    /// `(−π/2, π/2]`.
    ///
    /// **Normalizes** rather than rejects, for the reason given on
    /// [`Orientation::from_radians`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidParameter`] if `radians` is NaN or infinite.
    pub fn from_radians(radians: f64) -> Result<Self, Error> {
        if radians.is_finite() {
            Ok(Self(wrap_pi(radians)))
        } else {
            Err(Error::InvalidParameter(format!(
                "axis orientation must be a finite angle in radians, got {radians}"
            )))
        }
    }

    /// Creates an axis orientation as `½·atan2(y, x)`.
    ///
    /// The half-angle form that second-moment axis extraction produces.
    /// **Total**: nothing can fail. `atan2` returns `[−π, π]`, one value
    /// wider than twice the canonical range at the bottom — a negative-zero
    /// `y` beside a negative `x` yields exactly `−π` — so the halved
    /// boundary value `−π/2` is folded to `+π/2`, the same axis spelled
    /// inside `(−π/2, π/2]`.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::AxialOrientation;
    ///
    /// // A horizontal major axis.
    /// assert_eq!(AxialOrientation::from_half_atan2(0.0, 1.0).radians(), 0.0);
    /// ```
    #[must_use]
    pub fn from_half_atan2(y: f64, x: f64) -> Self {
        let axis = 0.5 * y.atan2(x);
        if axis == -core::f64::consts::FRAC_PI_2 {
            Self(core::f64::consts::FRAC_PI_2)
        } else {
            Self(axis)
        }
    }

    /// Returns the angle in radians, in `(−π/2, π/2]`.
    #[must_use]
    pub const fn radians(self) -> f64 {
        self.0
    }

    /// Returns the signed angle **from `other` to `self`**, in
    /// `(−π/2, π/2]`.
    ///
    /// Wraps at π, not 2π: two axes are never more than a quarter turn
    /// apart, so the magnitude never exceeds π/2. This is the operation
    /// that a raw float gets wrong — subtracting `−80°` from `80°` reads as
    /// `160°` when the axes are in fact `20°` apart.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::AxialOrientation;
    ///
    /// let a = AxialOrientation::from_radians(80_f64.to_radians())?;
    /// let b = AxialOrientation::from_radians((-80_f64).to_radians())?;
    /// // 20° apart, not 160°. (The sign says which way round: the axis at
    /// // 80° is a fifth of a quarter-turn clockwise of the one at −80°.)
    /// let apart = a.signed_difference(b).abs().to_degrees();
    /// assert!((apart - 20.0).abs() < 1e-9, "got {apart}");
    /// # Ok::<(), fovea::Error>(())
    /// ```
    #[must_use]
    pub fn signed_difference(self, other: Self) -> f64 {
        wrap_pi(self.0 - other.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigma_valid_values_round_trip() {
        assert_eq!(sigma!(1.4).get(), 1.4);
        assert_eq!(Sigma::try_new(0.5).unwrap().get(), 0.5);
        // Const construction: an invalid literal here would not compile.
        const S: Sigma = sigma!(2.0);
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
    fn sigma_new_rejects_an_invalid_value() {
        // `new` is checked and total, so no call site can abort. The
        // `sigma!` macro turns the same rejection into a compile error,
        // which a `compile_fail` doctest on the macro covers.
        assert!(Sigma::new(-1.5).is_none());
        assert!(Sigma::new(0.0).is_none());
        assert!(Sigma::new(f32::NAN).is_none());
        assert!(Sigma::new(f32::INFINITY).is_none());
    }

    #[test]
    fn pixel_distance_valid_values_round_trip() {
        assert_eq!(pixel_distance!(2.0).get(), 2.0);
        assert_eq!(PixelDistance::try_new(0.5).unwrap().get(), 0.5);
        const D: PixelDistance = pixel_distance!(0.5);
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
    fn pixel_distance_new_rejects_an_invalid_value() {
        assert!(PixelDistance::new(0.0).is_none());
        assert!(PixelDistance::new(-2.0).is_none());
        assert!(PixelDistance::new(f64::NAN).is_none());
    }

    #[test]
    fn odd_window_side_valid_values_round_trip() {
        assert_eq!(window!(31).get(), 31);
        assert_eq!(OddWindowSide::try_new(3).unwrap().get(), 3);
        // Const construction: an even literal here would not compile.
        const W: OddWindowSide = window!(15);
        assert_eq!(W.get(), 15);
        // A side of 1 is degenerate but valid: the window is the pixel.
        assert_eq!(window!(1).radius(), 0);
    }

    #[test]
    fn odd_window_side_try_new_rejects_even_and_zero() {
        // A window side can be computed (from a physical feature size and a
        // pixel pitch, say) and land on an even number or zero, so each is
        // an error value rather than a crash.
        for side in [0, 2, 4, 100] {
            let err = OddWindowSide::try_new(side).unwrap_err();
            match err {
                Error::InvalidParameter(reason) => assert!(
                    reason.contains("odd and non-zero"),
                    "reason {reason:?} does not name the invariant"
                ),
                other => panic!("expected InvalidParameter, got {other:?}"),
            }
        }
    }

    #[test]
    fn odd_window_side_new_rejects_an_invalid_side() {
        assert!(OddWindowSide::new(8).is_none());
        assert!(OddWindowSide::new(0).is_none());
    }

    #[test]
    fn odd_window_side_radius_is_the_exact_half_width() {
        // The reason the side is the wrapped quantity and the radius is
        // derived: for an odd side the two determine each other exactly, so
        // a consumer can take either without a rounding decision.
        for side in [1, 3, 5, 31, 101] {
            let w = OddWindowSide::new(side).unwrap();
            assert_eq!(w.radius(), side / 2);
            assert_eq!(2 * w.radius() + 1, side);
        }
    }

    // ───────────────────────────────────────────────────────────────────
    // Orientation (mod 2π) and AxialOrientation (mod π)
    // ───────────────────────────────────────────────────────────────────

    const PI32: f32 = core::f32::consts::PI;
    const PI64: f64 = core::f64::consts::PI;

    #[test]
    fn orientation_canonicalizes_into_half_open_range() {
        // The upper bound is inclusive, the lower exclusive: −π folds to +π.
        assert_eq!(Orientation::from_radians(PI32).unwrap().radians(), PI32);
        assert_eq!(Orientation::from_radians(-PI32).unwrap().radians(), PI32);
        assert_eq!(Orientation::from_radians(0.0).unwrap().radians(), 0.0);

        // Every canonical value lies in (−π, π].
        for turns in [-3.0, -1.5, -0.25, 0.0, 0.75, 2.0, 5.5] {
            let a = Orientation::from_radians(turns * PI32).unwrap();
            assert!(
                a.radians() > -PI32 && a.radians() <= PI32,
                "{turns} turns → {}",
                a.radians()
            );
        }
    }

    #[test]
    fn orientation_wraps_full_turns_to_the_same_direction() {
        // The point of the type: 0 and 2π are one direction, not two.
        let zero = Orientation::from_radians(0.0).unwrap();
        let full = Orientation::from_radians(2.0 * PI32).unwrap();
        assert_eq!(zero, full); // exact here: 2π wraps to a clean 0

        // 3π and π agree to within an ULP, not bit-exactly — wrapping
        // rounds, which is why the documented comparison is by difference.
        let three_halves = Orientation::from_radians(3.0 * PI32).unwrap();
        let half = Orientation::from_radians(PI32).unwrap();
        assert!(three_halves.signed_difference(half).abs() < 1e-6);
    }

    #[test]
    fn orientation_from_radians_rejects_non_finite() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let err = Orientation::from_radians(value).unwrap_err();
            match err {
                Error::InvalidParameter(reason) => assert!(
                    reason.contains("orientation"),
                    "reason {reason:?} does not mention orientation"
                ),
                other => panic!("expected InvalidParameter, got {other:?}"),
            }
        }
    }

    #[test]
    fn orientation_from_atan2_is_total_and_canonical() {
        assert_eq!(Orientation::from_atan2(0.0, 1.0).radians(), 0.0);
        assert_eq!(Orientation::from_atan2(1.0, 0.0).radians(), PI32 / 2.0);
        assert_eq!(Orientation::from_atan2(0.0, -1.0).radians(), PI32);
        // A zero-length gradient is defined, not NaN.
        assert_eq!(Orientation::from_atan2(0.0, 0.0).radians(), 0.0);
        // The atan2 boundary: a negative-zero y beside a negative x is the
        // one input that produces exactly −π, and it folds to +π so the
        // result stays inside the canonical (−π, π].
        assert_eq!(Orientation::from_atan2(-0.0, -1.0).radians(), PI32);
        assert!(Orientation::from_atan2(-0.0, 1.0).radians() == 0.0);
    }

    #[test]
    fn orientation_signed_difference_crosses_the_seam() {
        // The bug this type exists to prevent: 2° apart, not 358°.
        let east = Orientation::from_radians(179_f32.to_radians()).unwrap();
        let west = Orientation::from_radians((-179_f32).to_radians()).unwrap();
        let delta = east.signed_difference(west).to_degrees();
        assert!((delta.abs() - 2.0).abs() < 1e-3, "got {delta}");
    }

    #[test]
    fn orientation_signed_difference_is_signed_and_directed() {
        let a = Orientation::from_radians(0.5).unwrap();
        let b = Orientation::from_radians(0.2).unwrap();
        assert!((a.signed_difference(b) - 0.3).abs() < 1e-6);
        assert!((b.signed_difference(a) + 0.3).abs() < 1e-6);
        assert_eq!(a.signed_difference(a), 0.0);
    }

    #[test]
    fn orientation_signed_difference_never_exceeds_pi() {
        for degrees in [0.0_f32, 45.0, 90.0, 179.0, 181.0, 270.0, 359.0] {
            let a = Orientation::from_radians(degrees.to_radians()).unwrap();
            let b = Orientation::from_radians(0.0).unwrap();
            assert!(
                a.signed_difference(b).abs() <= PI32 + 1e-6,
                "{degrees}° → {}",
                a.signed_difference(b)
            );
        }
    }

    #[test]
    fn orientation_to_axial_collapses_opposite_directions() {
        // Opposite directions share one axis. Compared by difference: these
        // sit exactly on the ±π/2 boundary, where the f32→f64 widening can
        // report either end of the canonical range — both naming the same
        // axis, which is precisely what `signed_difference` sees.
        let north = Orientation::from_radians(PI32 / 2.0).unwrap();
        let south = Orientation::from_radians(-PI32 / 2.0).unwrap();
        assert!(north.to_axial().signed_difference(south.to_axial()).abs() < 1e-6);

        let east = Orientation::from_radians(0.0).unwrap();
        let west = Orientation::from_radians(PI32).unwrap();
        assert!(east.to_axial().signed_difference(west.to_axial()).abs() < 1e-6);

        // ...and the two axes are a quarter turn apart, not the same.
        let apart = north.to_axial().signed_difference(east.to_axial()).abs();
        assert!((apart - PI64 / 2.0).abs() < 1e-6, "got {apart}");
    }

    #[test]
    fn orientation_is_copy_and_debug() {
        let a = Orientation::from_radians(1.0).unwrap();
        let b = a; // Copy
        assert_eq!(a, b);
        assert!(format!("{a:?}").contains("Orientation"));
    }

    #[test]
    fn axial_orientation_canonicalizes_into_quarter_turn_range() {
        let half_pi = PI64 / 2.0;
        assert_eq!(
            AxialOrientation::from_radians(half_pi).unwrap().radians(),
            half_pi
        );
        // −π/2 is the same axis as +π/2 and folds onto it.
        assert_eq!(
            AxialOrientation::from_radians(-half_pi).unwrap().radians(),
            half_pi
        );

        for turns in [-2.0, -0.75, 0.0, 0.3, 1.0, 3.5] {
            let a = AxialOrientation::from_radians(turns * PI64).unwrap();
            assert!(
                a.radians() > -half_pi && a.radians() <= half_pi,
                "{turns}·π → {}",
                a.radians()
            );
        }
    }

    #[test]
    fn axial_orientation_treats_opposite_angles_as_one_axis() {
        // An axis has no head or tail: θ and θ + π are equal.
        let a = AxialOrientation::from_radians(0.4).unwrap();
        let b = AxialOrientation::from_radians(0.4 + PI64).unwrap();
        assert!(a.signed_difference(b).abs() < 1e-12);

        // 80° and −100° are the same axis, reached the other way round.
        let c = AxialOrientation::from_radians(80_f64.to_radians()).unwrap();
        let d = AxialOrientation::from_radians((-100_f64).to_radians()).unwrap();
        assert!(c.signed_difference(d).abs() < 1e-12);
    }

    #[test]
    fn axial_orientation_from_radians_rejects_non_finite() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let err = AxialOrientation::from_radians(value).unwrap_err();
            match err {
                Error::InvalidParameter(reason) => assert!(
                    reason.contains("axis orientation"),
                    "reason {reason:?} does not mention axis orientation"
                ),
                other => panic!("expected InvalidParameter, got {other:?}"),
            }
        }
    }

    #[test]
    fn axial_orientation_from_half_atan2_is_total_and_canonical() {
        assert_eq!(AxialOrientation::from_half_atan2(0.0, 1.0).radians(), 0.0);
        // atan2(0, −1) = π → halved = π/2, the vertical axis.
        assert_eq!(
            AxialOrientation::from_half_atan2(0.0, -1.0).radians(),
            PI64 / 2.0
        );
        assert_eq!(AxialOrientation::from_half_atan2(0.0, 0.0).radians(), 0.0);
        // The atan2 boundary: atan2(−0.0, −1) = −π → halved = −π/2, which
        // folds to +π/2 so the result stays inside (−π/2, π/2].
        assert_eq!(
            AxialOrientation::from_half_atan2(-0.0, -1.0).radians(),
            PI64 / 2.0
        );
    }

    #[test]
    fn axial_orientation_difference_never_exceeds_a_quarter_turn() {
        // The axial seam: this is what a raw float subtraction gets wrong,
        // reading 160° where the axes are 20° apart.
        let a = AxialOrientation::from_radians(80_f64.to_radians()).unwrap();
        let b = AxialOrientation::from_radians((-80_f64).to_radians()).unwrap();
        // 20° apart, not 160°. The sign is negative: the axis at 80° is
        // reached from the one at −80° (≡ 100°) by turning back 20°.
        let delta = a.signed_difference(b).to_degrees();
        assert!((delta.abs() - 20.0).abs() < 1e-9, "got {delta}");

        for degrees in [0.0_f64, 10.0, 89.0, 91.0, 170.0, 269.0] {
            let x = AxialOrientation::from_radians(degrees.to_radians()).unwrap();
            let y = AxialOrientation::from_radians(0.0).unwrap();
            assert!(
                x.signed_difference(y).abs() <= PI64 / 2.0 + 1e-12,
                "{degrees}° → {}",
                x.signed_difference(y)
            );
        }
    }

    #[test]
    fn axial_orientation_is_copy_and_debug() {
        let a = AxialOrientation::from_radians(1.0).unwrap();
        let b = a; // Copy
        assert_eq!(a, b);
        assert!(format!("{a:?}").contains("AxialOrientation"));
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

    // ── Offset ──────────────────────────────────────────────────────────

    #[test]
    fn checked_add_rejects_only_the_negative_half() {
        let c = Coordinate::new(0, 0);
        assert_eq!(c.checked_add(Offset::ZERO), Some(c));
        assert_eq!(c.checked_add(Offset::new(-1, 0)), None);
        assert_eq!(c.checked_add(Offset::new(0, -1)), None);
        assert_eq!(
            c.checked_add(Offset::new(3, 4)),
            Some(Coordinate::new(3, 4))
        );
        // The far edge is not this operation's business: a position past
        // the image is still a coordinate, and `ImageView::get` is what
        // rejects it.
        assert_eq!(
            Coordinate::new(usize::MAX - 1, 0).checked_add(Offset::new(1, 0)),
            Some(Coordinate::new(usize::MAX, 0))
        );
        assert_eq!(
            Coordinate::new(usize::MAX, 0).checked_add(Offset::new(1, 0)),
            None
        );
    }

    #[test]
    fn offset_to_inverts_checked_add() {
        let corners = [
            Coordinate::new(0, 0),
            Coordinate::new(7, 0),
            Coordinate::new(0, 5),
            Coordinate::new(7, 5),
        ];
        for &a in &corners {
            for &b in &corners {
                assert_eq!(a.checked_add(a.offset_to(b)), Some(b), "{a:?} -> {b:?}");
            }
        }
    }

    #[test]
    fn offset_to_saturates_rather_than_wrapping() {
        // Out of `i32` range, so the components clamp. The point is the
        // sign: a wrapping cast would turn a step right into a step left.
        let far = Coordinate::new(usize::MAX, 0).offset_to(Coordinate::new(0, usize::MAX));
        assert_eq!(far, Offset::new(i32::MIN, i32::MAX));
    }

    #[test]
    fn an_offset_is_transposition_sensitive() {
        // The whole reason the type exists: the two field orders are
        // distinguishable values, where a positional pair would have made
        // the swap invisible.
        assert_ne!(Offset::new(1, -1), Offset::new(-1, 1));
    }
}

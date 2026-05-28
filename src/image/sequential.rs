use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

use crate::image::tiles::{SubView, SubViewMut};
use crate::image::{ImageView, ImageViewMut, RasterImage, RasterImageMut};
use crate::{Coordinate, Rectangle, Size, internal};

/// Strided analogue of [`crate::internal::checked_index_or_panic`] used by
/// `ImageRef::pixel_at` and `ImageRef::row`. Same Tier-3 semantics: panic
/// on out-of-bounds or arithmetic overflow rather than wrap silently.
#[inline]
fn strided_checked_index_or_panic(
    size: &Size,
    stride: usize,
    offset: usize,
    x: usize,
    y: usize,
) -> usize {
    assert!(
        x < size.width && y < size.height,
        "pixel index out of bounds: ({x}, {y}) is not inside {}x{}",
        size.width,
        size.height
    );
    y.checked_mul(stride)
        .and_then(|row_off| offset.checked_add(row_off))
        .and_then(|row_start| row_start.checked_add(x))
        .unwrap_or_else(|| {
            panic!(
                "pixel index arithmetic overflowed usize: ({x}, {y}) on \
                 strided view (stride={stride}, offset={offset})"
            )
        })
}

#[inline]
fn strided_checked_row_start_or_panic(
    size: &Size,
    stride: usize,
    offset: usize,
    y: usize,
) -> usize {
    assert!(
        y < size.height,
        "row index out of bounds: {y} is not less than height {}",
        size.height
    );
    y.checked_mul(stride)
        .and_then(|row_off| offset.checked_add(row_off))
        .unwrap_or_else(|| {
            panic!(
                "row index arithmetic overflowed usize: y={y} on strided view \
                 (stride={stride}, offset={offset})"
            )
        })
}

use std::borrow::Cow;
use std::fmt;

use crate::error::Error;
use crate::pixel::{PlainChannel, PlainPixel, ZeroablePixel};

/// Sealed helper — compile-time gate for zero-copy byte reinterpretation.
trait AssertByteAligned: PlainPixel {
    const _ASSERT: () = assert!(
        std::mem::align_of::<Self>() == 1,
        "from_raw_bytes requires a pixel type with ALIGN == 1; \
         use from_bytes_copy for aligned pixel types"
    );
}
impl<T: PlainPixel> AssertByteAligned for T {}

pub(crate) mod private {
    use std::ops::{Index, IndexMut};
    /// A helper trait to associate a pixel type and an array type with specific dimensions.
    /// This trait is implemented for the `Dim` struct for various width and height combinations.
    pub trait _Array2D {
        type Pixel: Copy;
        type Array: Index<usize, Output = Self::Pixel>
            + IndexMut<usize, Output = Self::Pixel>
            + AsRef<[Self::Pixel]>
            + AsMut<[Self::Pixel]>
            + Sized;
    }

    /// A marker type to represent image dimensions at compile time.
    /// This is used to associate a pixel type and an array type with specific dimensions.
    pub struct Dim<T, const W: usize, const H: usize> {
        _marker: std::marker::PhantomData<T>,
    }

    macro_rules! define_array_2D {
        ($x:literal, $y:literal) => {
            impl<T: Copy> _Array2D for Dim<T, $x, $y> {
                type Pixel = T;
                type Array = [T; $x * $y];
            }
        };
    }

    /// Helper: given a single width `$w` and a list of heights, emit
    /// one `define_array_2D!` call for each height.
    macro_rules! define_row {
        ($w:literal; $($h:literal),+ $(,)?) => {
            $(
                define_array_2D!($w, $h);
            )+
        };
    }

    /// Generates `_Array2D` impls for every `(W, H)` pair in the
    /// cartesian product of the supplied dimension values.
    ///
    /// This is a workaround for the lack of `generic_const_exprs` on
    /// stable Rust — `[T; W * H]` cannot be written directly as a
    /// const-generic expression, so we enumerate the pairs we support.
    ///
    /// Uses a recursive peeling approach: peel the first width off
    /// the list, emit impls for that width paired with every height,
    /// then recurse on the remaining widths.
    macro_rules! define_all_array_2D {
        // Base case: single width remaining.
        ([$w:literal], [$($h:literal),+ $(,)?]) => {
            define_row!($w; $($h),+);
        };
        // Recursive case: peel the first width, emit its row, recurse.
        ([$w:literal, $($rest:literal),+ $(,)?], [$($h:literal),+ $(,)?]) => {
            define_row!($w; $($h),+);
            define_all_array_2D!([$($rest),+], [$($h),+]);
        };
    }

    define_all_array_2D!(
        [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
            47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64
        ],
        [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
            47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64
        ]
    );
}

/// Crate-internal seal used by [`ContiguousImage`] and
/// [`ContiguousImageMut`].
///
/// External implementations of those traits are forbidden because the
/// blanket [`IntoTilesMut`](crate::image::IntoTilesMut) implementation and
/// the various `as_bytes` helpers rely on memory-safety invariants (the
/// returned slice length equals `width * height` of an image with the
/// reported `size()`) that the compiler cannot enforce. See
/// ADR-0048 for the full rationale.
pub(crate) mod contiguous_sealed {
    pub trait Sealed {}
}

/// The `ContiguousImage` trait extends `ImageView` for images stored in a contiguous array format.
/// It provides a method to access the underlying pixel data as an immutable slice.
///
/// # Sealed
///
/// This trait is **sealed** — it cannot be implemented outside this crate.
/// The contract "`as_slice().len() == width * height`" is relied on by
/// safe code (notably [`IntoTilesMut`](crate::image::IntoTilesMut))
/// to derive raw pointers and slice ranges; an incorrect external impl
/// would cause undefined behaviour. See ADR-0048 for details on why this
/// trait is sealed today and how it may be reopened as an `unsafe trait`
/// in the future.
///
/// # Example
/// ```
/// # use fovea::image::{Image, ImageView, ContiguousImage};
/// let img = Image::generate(3, 2, |x, y| (x + y * 3) as u8);
/// let slice = img.as_slice();
/// assert_eq!(slice, &[0, 1, 2, 3, 4, 5]);
/// ```
pub trait ContiguousImage: RasterImage + contiguous_sealed::Sealed {
    fn as_slice(&self) -> &[Self::Pixel];
}

/// The `ContiguousImageMut` trait extends `ContiguousImage + ImageViewMut` for images
/// that also allow mutable access to the underlying pixel data as a slice.
///
/// # Sealed
///
/// This trait is **sealed** — it cannot be implemented outside this crate.
/// See [`ContiguousImage`] and ADR-0048 for the rationale.
///
/// # Example
/// ```
/// # use fovea::image::{Image, ImageView, ContiguousImageMut};
/// let mut img = Image::generate(2, 2, |_, _| 0u8);
/// img.as_mut_slice().fill(42);
/// assert_eq!(img.pixel_at(1, 1), 42);
/// ```
pub trait ContiguousImageMut: ContiguousImage + RasterImageMut {
    fn as_mut_slice(&mut self) -> &mut [Self::Pixel];
}

/// The `PlainImage` trait allows read-only byte serialization of an image
/// whose pixels implement [`PlainPixel`].
///
/// The endianness of [`as_bytes`](PlainImage::as_bytes) depends on the host;
/// use [`as_bytes_le`](PlainImage::as_bytes_le) or
/// [`as_bytes_be`](PlainImage::as_bytes_be) for explicit control.
///
/// For mutable byte access, see [`PlainImageMut`].
///
/// This trait only requires [`ContiguousImage`], so it is implementable for
/// read-only contiguous backends. Mutable byte access lives on the separate
/// [`PlainImageMut`] trait to avoid forcing immutable consumers to require
/// mutable storage.
pub trait PlainImage: ContiguousImage
where
    Self::Pixel: PlainPixel,
{
    fn as_bytes(&self) -> &[u8];
    fn as_bytes_le(&self) -> Cow<'_, [u8]>;
    fn as_bytes_be(&self) -> Cow<'_, [u8]>;
}

/// The `PlainImageMut` trait extends [`PlainImage`] with mutable byte access.
///
/// The endianness of [`as_mut_bytes`](PlainImageMut::as_mut_bytes) depends on
/// the host; convert to an explicit endianness via [`PlainImage::as_bytes_le`]
/// or [`PlainImage::as_bytes_be`] when needed.
pub trait PlainImageMut: PlainImage + ContiguousImageMut
where
    Self::Pixel: PlainPixel,
{
    fn as_mut_bytes(&mut self) -> &mut [u8];
}

/// An image stored in a fixed-size array.
///
/// The dimensions of the image are specified at compile time using const generics.
///
/// # Example
/// ```
/// # use fovea::image::{ImageView, ImageArray};
/// let arr = [1u8,2,3,4,5,6,7,8,9,10,12,12];
/// let mut img: ImageArray<u8, 3, 4> = ImageArray::new(arr);
/// assert_eq!(img.width(), 3);
/// assert_eq!(img.height(), 4);
/// assert_eq!(img[(0, 0)], 1);
///
/// // does not compile if the array size is incorrect
/// // let arr = [1u8,2,3,4,5,6,7,8,9,10,11];
/// // let mut img: ImageArray<u8, 3, 4> = ImageArray::new(arr);
/// ```
///
/// The `ImageArray` can be generated using a closure:
/// ```
/// # use fovea::Size;
/// # use fovea::image::{ImageView, ImageArray};
/// let img: ImageArray<u8, 3, 4> = ImageArray::generate(|x,y| (x + y * 3) as u8 + 1);
/// assert_eq!(img.size(), Size::new(3, 4));
/// assert_eq!(img.get(0, 0), Some(1));
/// assert_eq!(img.get(1, 2), Some(8));
/// assert_eq!(img.get(2, 3), Some(12));
/// assert_eq!(img[(2, 3)], 12);
/// ```
///
/// Since the image data is stored on the stack, this type is suitable for small images
/// like convolution kernels. The maximum supported dimensions are **64×64** (4 096 elements).
///
/// This limit exists because stable Rust does not yet support `generic_const_exprs`,
/// so `[T; W * H]` cannot be used directly in a const-generic context.  The
/// supported `(W, H)` pairs are enumerated via a macro.  When `generic_const_exprs`
/// stabilises, this restriction will be lifted.
pub struct ImageArray<T, const W: usize, const H: usize>
where
    private::Dim<T, W, H>: private::_Array2D,
{
    data: <private::Dim<T, W, H> as private::_Array2D>::Array,
}

impl<T: Copy, const W: usize, const H: usize> ImageArray<T, W, H>
where
    private::Dim<T, W, H>: private::_Array2D,
{
    /// Create a new ImageArray from the given data array.
    /// The length of the array must be equal to W * H.
    /// Does not compile if the length is incorrect.
    ///
    /// # Example
    /// ```
    /// use fovea::image::{ImageView, ImageViewMut, ImageArray};
    ///
    /// let mut img: ImageArray<u8, 2, 2> = ImageArray::new([0; 4]);
    /// assert_eq!(img.width(), 2);
    /// assert_eq!(img.height(), 2);
    /// assert_eq!(img.get(0, 0), Some(0));
    /// assert_eq!(img.get(1, 1), Some(0));
    /// assert_eq!(img.get(2, 2), None);
    ///
    /// if let Some(pixel) = img.get_mut(0, 0) {
    ///     *pixel = 10;
    /// }
    /// assert_eq!(img.get(0, 0), Some(10));
    /// ```
    pub fn new(data: <private::Dim<T, W, H> as private::_Array2D>::Array) -> Self {
        Self { data }
    }

    /// Generate a new ImageArray using the provided closure.
    /// The closure is called for each pixel coordinate (x, y) to produce the pixel value.
    /// The dimensions W and H are specified as const generics.
    ///
    /// # Example
    /// ```
    /// # use fovea::image::ImageArray;
    /// let img: ImageArray<u8, 3, 4> = ImageArray::generate(|x,y| (x + y * 3) as u8 + 1);
    /// ```
    pub fn generate(f: impl Fn(usize, usize) -> <ImageArray<T, W, H> as ImageView>::Pixel) -> Self {
        // Use MaybeUninit to avoid initializing array elements before setting them
        let mut uninit_data: std::mem::MaybeUninit<
            <private::Dim<T, W, H> as private::_Array2D>::Array,
        > = std::mem::MaybeUninit::uninit();
        let data_ptr = uninit_data.as_mut_ptr() as *mut <ImageArray<T, W, H> as ImageView>::Pixel;

        // Initialize all elements using the provided function
        for y in 0..H {
            for x in 0..W {
                unsafe {
                    data_ptr.add(x + y * W).write(f(x, y));
                }
            }
        }

        // # Safety: All elements have been initialized in the loops above
        let data = unsafe { uninit_data.assume_init() };
        Self { data }
    }
}

impl<T: Copy, const W: usize, const H: usize> ImageView for ImageArray<T, W, H>
where
    private::Dim<T, W, H>: private::_Array2D,
{
    type Pixel = <private::Dim<T, W, H> as private::_Array2D>::Pixel;

    fn size(&self) -> Size {
        Size::new(W, H)
    }

    fn width(&self) -> usize {
        W
    }

    fn height(&self) -> usize {
        H
    }

    #[inline(always)]
    fn get(&self, x: usize, y: usize) -> Option<Self::Pixel> {
        if x < W && y < H {
            Some(self.data[x + y * W])
        } else {
            None
        }
    }

    #[inline(always)]
    fn pixel_at(&self, x: usize, y: usize) -> Self::Pixel {
        self.data[x + y * W]
    }
}

impl<T: Copy, const W: usize, const H: usize> ImageViewMut for ImageArray<T, W, H>
where
    private::Dim<T, W, H>: private::_Array2D,
{
    #[inline(always)]
    fn get_mut(&mut self, x: usize, y: usize) -> Option<&mut Self::Pixel> {
        if x < W && y < H {
            Some(&mut self.data[x + y * W])
        } else {
            None
        }
    }

    #[inline(always)]
    fn pixel_at_mut(&mut self, x: usize, y: usize) -> &mut Self::Pixel {
        &mut self.data[x + y * W]
    }
}

impl<T: Copy, const W: usize, const H: usize> SubView for ImageArray<T, W, H>
where
    private::Dim<T, W, H>: private::_Array2D,
{
    type Sub<'a>
        = ImageRef<'a, Self::Pixel>
    where
        Self: 'a;

    fn roi(&self, rect: Rectangle) -> Option<Self::Sub<'_>> {
        let right = rect.checked_right()?;
        let bottom = rect.checked_bottom()?;
        if right <= W && bottom <= H {
            let offset = rect.top().checked_mul(W)?.checked_add(rect.left())?;
            ImageRef::strided(rect.size, W, offset, self.as_slice())
        } else {
            None
        }
    }
}

impl<T: Copy, const W: usize, const H: usize> SubViewMut for ImageArray<T, W, H>
where
    private::Dim<T, W, H>: private::_Array2D,
{
    type SubMut<'a>
        = ImageRefMut<'a, Self::Pixel>
    where
        Self: 'a;

    fn roi_mut(&mut self, rect: Rectangle) -> Option<Self::SubMut<'_>> {
        let right = rect.checked_right()?;
        let bottom = rect.checked_bottom()?;
        if right <= W && bottom <= H {
            let stride = W;
            let offset = rect.top().checked_mul(W)?.checked_add(rect.left())?;
            let slice = self.as_mut_slice();
            let len = slice.len();
            let ptr = slice.as_mut_ptr();
            // SAFETY: ptr comes from a valid &mut [T] of length len.
            // rect is within image bounds (checked above).
            // Exclusive access is guaranteed by &mut self.
            Some(unsafe { ImageRefMut::strided(rect.size, stride, offset, ptr, len) })
        } else {
            None
        }
    }
}

impl<T: Copy, const W: usize, const H: usize> RasterImage for ImageArray<T, W, H>
where
    private::Dim<T, W, H>: private::_Array2D,
{
    #[inline(always)]
    fn row(&self, y: usize) -> &[Self::Pixel] {
        &self.data.as_ref()[y * W..y * W + W]
    }
}

impl<T: Copy, const W: usize, const H: usize> RasterImageMut for ImageArray<T, W, H>
where
    private::Dim<T, W, H>: private::_Array2D,
{
    #[inline(always)]
    fn row_mut(&mut self, y: usize) -> &mut [Self::Pixel] {
        &mut self.data.as_mut()[y * W..y * W + W]
    }
}

impl<T: Copy, const W: usize, const H: usize> contiguous_sealed::Sealed for ImageArray<T, W, H> where
    private::Dim<T, W, H>: private::_Array2D
{
}

impl<T: Copy, const W: usize, const H: usize> ContiguousImage for ImageArray<T, W, H>
where
    private::Dim<T, W, H>: private::_Array2D,
{
    fn as_slice(&self) -> &[Self::Pixel] {
        self.data.as_ref()
    }
}

impl<T: Copy, const W: usize, const H: usize> ContiguousImageMut for ImageArray<T, W, H>
where
    private::Dim<T, W, H>: private::_Array2D,
{
    fn as_mut_slice(&mut self) -> &mut [Self::Pixel] {
        self.data.as_mut()
    }
}

impl<T: Copy, const W: usize, const H: usize> PlainImage for ImageArray<T, W, H>
where
    T: PlainPixel,
    private::Dim<T, W, H>: private::_Array2D<Pixel = T>,
{
    fn as_bytes(&self) -> &[u8] {
        unsafe { internal::as_bytes(self.as_slice()) }
    }

    fn as_bytes_le(&self) -> Cow<'_, [u8]> {
        unsafe { internal::as_bytes_le(self.as_slice()) }
    }

    fn as_bytes_be(&self) -> Cow<'_, [u8]> {
        unsafe { internal::as_bytes_be(self.as_slice()) }
    }
}

impl<T: Copy, const W: usize, const H: usize> PlainImageMut for ImageArray<T, W, H>
where
    T: PlainPixel,
    private::Dim<T, W, H>: private::_Array2D<Pixel = T>,
{
    fn as_mut_bytes(&mut self) -> &mut [u8] {
        unsafe { internal::as_mut_bytes(self.as_mut_slice()) }
    }
}

impl<T: Copy, C: Into<Coordinate>, const W: usize, const H: usize> Index<C> for ImageArray<T, W, H>
where
    private::Dim<T, W, H>: private::_Array2D<Pixel = T>,
{
    type Output = T;

    #[inline(always)]
    fn index(&self, index: C) -> &Self::Output {
        let Coordinate { x, y } = index.into();
        &self.data[x + y * W]
    }
}

impl<T: Copy, C: Into<Coordinate>, const W: usize, const H: usize> IndexMut<C>
    for ImageArray<T, W, H>
where
    private::Dim<T, W, H>: private::_Array2D<Pixel = T>,
{
    #[inline(always)]
    fn index_mut(&mut self, index: C) -> &mut Self::Output {
        let Coordinate { x, y } = index.into();
        self.pixel_at_mut(x, y)
    }
}

// ───────────────────────────────────────────────────────────────────
// ImageRef — borrowed, strided image view
// ───────────────────────────────────────────────────────────────────

/// A borrowed, strided image view.
///
/// Holds a reference to an existing pixel buffer without copying.
/// Two modes of use:
///
/// **Contiguous full-frame view** (`stride == size.width`, `offset == 0`):
/// constructed via [`ImageRef::new`]. This is the zero-copy camera-SDK
/// entry point.
///
/// **Strided sub-region view** (produced by [`SubView::roi`]):
/// `stride` equals the parent image width; `offset` locates the
/// top-left corner within the parent buffer.
#[derive(Copy, Clone, Debug)]
pub struct ImageRef<'a, T> {
    size: Size,
    stride: usize,
    offset: usize,
    data: &'a [T],
}

impl<'a, T> ImageRef<'a, T> {
    /// Contiguous full-frame view. Returns `Err` if `data.len() != width * height`.
    pub fn new(width: usize, height: usize, data: &'a [T]) -> Result<Self, Error> {
        let size = Size::new(width, height);
        let expected = size.checked_area().ok_or(Error::LengthMismatch {
            expected: usize::MAX,
            actual: data.len(),
        })?;
        if data.len() == expected {
            Ok(Self {
                size,
                stride: width,
                offset: 0,
                data,
            })
        } else {
            Err(Error::LengthMismatch {
                expected,
                actual: data.len(),
            })
        }
    }

    /// General strided view. Checks that all pixel indices are in bounds.
    pub(crate) fn strided(size: Size, stride: usize, offset: usize, data: &'a [T]) -> Option<Self> {
        if size.height == 0 || size.width == 0 {
            return Some(Self {
                size,
                stride,
                offset,
                data,
            });
        }
        // Check that the last pixel is in bounds using checked arithmetic
        // so that hostile or accidental size/stride combinations cannot
        // wrap around `usize` and silently pass the bounds check.
        let last_row = size.height - 1;
        let last_col = size.width - 1;
        let last_index = last_row
            .checked_mul(stride)
            .and_then(|r| r.checked_add(offset))
            .and_then(|s| s.checked_add(last_col))?;
        if last_index < data.len() {
            Some(Self {
                size,
                stride,
                offset,
                data,
            })
        } else {
            None
        }
    }

    /// Returns true when stride == size.width && offset == 0.
    #[inline]
    pub fn is_contiguous(&self) -> bool {
        self.stride == self.size.width && self.offset == 0
    }
}

impl<T: Copy> ImageView for ImageRef<'_, T> {
    type Pixel = T;

    fn size(&self) -> Size {
        self.size
    }

    fn width(&self) -> usize {
        self.size.width
    }

    fn height(&self) -> usize {
        self.size.height
    }

    #[inline(always)]
    fn get(&self, x: usize, y: usize) -> Option<Self::Pixel> {
        if x < self.size.width && y < self.size.height {
            Some(self.data[self.offset + y * self.stride + x])
        } else {
            None
        }
    }

    #[inline(always)]
    fn pixel_at(&self, x: usize, y: usize) -> Self::Pixel {
        self.data[strided_checked_index_or_panic(&self.size, self.stride, self.offset, x, y)]
    }
}

impl<T: Copy> RasterImage for ImageRef<'_, T> {
    #[inline(always)]
    fn row(&self, y: usize) -> &[Self::Pixel] {
        let start = strided_checked_row_start_or_panic(&self.size, self.stride, self.offset, y);
        &self.data[start..start + self.size.width]
    }
}

impl<T: Copy> SubView for ImageRef<'_, T> {
    type Sub<'b>
        = ImageRef<'b, T>
    where
        Self: 'b;

    fn roi(&self, rect: Rectangle) -> Option<ImageRef<'_, T>> {
        let right = rect.checked_right()?;
        let bottom = rect.checked_bottom()?;
        if right <= self.size.width && bottom <= self.size.height {
            let offset = self
                .offset
                .checked_add(rect.top().checked_mul(self.stride)?)?
                .checked_add(rect.left())?;
            ImageRef::strided(rect.size, self.stride, offset, self.data)
        } else {
            None
        }
    }
}

// ──────────────────────────────────────────────────────────────────
// ImageRefMut — mutably borrowed, strided image view
// ───────────────────────────────────────────────────────────────────

/// A mutably borrowed, strided image view.
///
/// Uses a raw pointer internally so that [`TileIterMut`](crate::image::TileIterMut) can yield
/// multiple disjoint mutable tiles from a single image simultaneously
/// without violating Rust's aliasing rules.
pub struct ImageRefMut<'a, T> {
    size: Size,
    stride: usize,
    offset: usize,
    data: *mut T,
    data_len: usize,
    _marker: PhantomData<&'a mut T>,
}

// SAFETY: ImageRefMut has exclusive access to its pixel region.
// Sending it to another thread is safe when T is Send (same as &'a mut [T]).
unsafe impl<T: Send> Send for ImageRefMut<'_, T> {}

// SAFETY: Shared access (&ImageRefMut) only provides &T references.
// This is safe when T is Sync (same as &'a mut [T]).
unsafe impl<T: Sync> Sync for ImageRefMut<'_, T> {}

impl<'a, T> ImageRefMut<'a, T> {
    /// Contiguous full-frame mutable view. Returns `Err` if `data.len() != width * height`.
    pub fn new(width: usize, height: usize, data: &'a mut [T]) -> Result<Self, Error> {
        let size = Size::new(width, height);
        let expected = size.checked_area().ok_or(Error::LengthMismatch {
            expected: usize::MAX,
            actual: data.len(),
        })?;
        if data.len() == expected {
            let len = data.len();
            let ptr = data.as_mut_ptr();
            Ok(Self {
                size,
                stride: width,
                offset: 0,
                data: ptr,
                data_len: len,
                _marker: PhantomData,
            })
        } else {
            Err(Error::LengthMismatch {
                expected,
                actual: data.len(),
            })
        }
    }

    /// General strided mutable view.
    ///
    /// # Safety
    /// - `data` must point to a valid allocation of at least `data_len` elements
    /// - The caller must guarantee exclusive access within the described rect for lifetime `'a`
    pub(crate) unsafe fn strided(
        size: Size,
        stride: usize,
        offset: usize,
        data: *mut T,
        data_len: usize,
    ) -> Self {
        Self {
            size,
            stride,
            offset,
            data,
            data_len,
            _marker: PhantomData,
        }
    }

    /// Returns true when stride == size.width && offset == 0.
    #[inline]
    pub fn is_contiguous(&self) -> bool {
        self.stride == self.size.width && self.offset == 0
    }
}

impl<T> ImageRefMut<'_, T> {
    /// Compute a checked element offset within the underlying allocation.
    ///
    /// Returns `Some(idx)` iff `offset + y*stride + x` does not overflow
    /// `usize` and `idx < data_len`. Used by every safe accessor below to
    /// guarantee that the subsequent raw-pointer dereference stays inside
    /// the original allocation, even in release builds and even if the
    /// (`pub(crate)`) `strided` constructor was called with values that
    /// would otherwise wrap on multiplication or addition.
    #[inline(always)]
    fn checked_elem_offset(&self, x: usize, y: usize) -> Option<usize> {
        let row_off = y.checked_mul(self.stride)?;
        let idx = self.offset.checked_add(row_off)?.checked_add(x)?;
        if idx < self.data_len { Some(idx) } else { None }
    }

    /// Compute a checked start-of-row offset and verify the whole row fits
    /// inside the underlying allocation. See [`Self::checked_elem_offset`].
    #[inline(always)]
    fn checked_row_range(&self, y: usize) -> Option<usize> {
        let row_off = y.checked_mul(self.stride)?;
        let start = self.offset.checked_add(row_off)?;
        let end = start.checked_add(self.size.width)?;
        if end <= self.data_len {
            Some(start)
        } else {
            None
        }
    }
}

impl<T: Copy> ImageView for ImageRefMut<'_, T> {
    type Pixel = T;

    fn size(&self) -> Size {
        self.size
    }

    fn width(&self) -> usize {
        self.size.width
    }

    fn height(&self) -> usize {
        self.size.height
    }

    #[inline(always)]
    fn get(&self, x: usize, y: usize) -> Option<Self::Pixel> {
        if x < self.size.width && y < self.size.height {
            let idx = self
                .checked_elem_offset(x, y)
                .expect("ImageRefMut::get: element offset overflowed or escaped allocation");
            // SAFETY: `checked_elem_offset` verified `idx < self.data_len`,
            // and `&self` guarantees we hold a valid borrow for `'a`.
            Some(unsafe { self.data.add(idx).read() })
        } else {
            None
        }
    }

    #[inline(always)]
    fn pixel_at(&self, x: usize, y: usize) -> Self::Pixel {
        assert!(
            x < self.size.width && y < self.size.height,
            "ImageRefMut::pixel_at: ({x}, {y}) out of bounds for size {:?}",
            self.size
        );
        let idx = self
            .checked_elem_offset(x, y)
            .expect("ImageRefMut::pixel_at: element offset overflowed or escaped allocation");
        // SAFETY: bounds checked above; `idx < self.data_len` verified by
        // `checked_elem_offset`.
        unsafe { self.data.add(idx).read() }
    }
}

impl<T: Copy> RasterImage for ImageRefMut<'_, T> {
    #[inline(always)]
    fn row(&self, y: usize) -> &[Self::Pixel] {
        assert!(
            y < self.size.height,
            "ImageRefMut::row: y={y} out of bounds for height {}",
            self.size.height
        );
        let start = self
            .checked_row_range(y)
            .expect("ImageRefMut::row: row range overflowed or escaped allocation");
        // SAFETY: `checked_row_range` verified `start + width <= data_len`;
        // `&self` borrow lifetime ensures the underlying allocation is live.
        unsafe { std::slice::from_raw_parts(self.data.add(start), self.size.width) }
    }
}

impl<T: Copy> RasterImageMut for ImageRefMut<'_, T> {
    #[inline(always)]
    fn row_mut(&mut self, y: usize) -> &mut [Self::Pixel] {
        assert!(
            y < self.size.height,
            "ImageRefMut::row_mut: y={y} out of bounds for height {}",
            self.size.height
        );
        let start = self
            .checked_row_range(y)
            .expect("ImageRefMut::row_mut: row range overflowed or escaped allocation");
        // SAFETY: `checked_row_range` verified `start + width <= data_len`;
        // `&mut self` borrow lifetime ensures exclusive access to the row.
        unsafe { std::slice::from_raw_parts_mut(self.data.add(start), self.size.width) }
    }
}

impl<T: Copy> ImageViewMut for ImageRefMut<'_, T> {
    fn get_mut(&mut self, x: usize, y: usize) -> Option<&mut Self::Pixel> {
        if x < self.size.width && y < self.size.height {
            let idx = self
                .checked_elem_offset(x, y)
                .expect("ImageRefMut::get_mut: element offset overflowed or escaped allocation");
            // SAFETY: `idx < self.data_len` verified above; `&mut self`
            // ensures exclusive access for the returned reference's lifetime.
            Some(unsafe { &mut *self.data.add(idx) })
        } else {
            None
        }
    }

    fn pixel_at_mut(&mut self, x: usize, y: usize) -> &mut Self::Pixel {
        assert!(
            x < self.size.width && y < self.size.height,
            "ImageRefMut::pixel_at_mut: ({x}, {y}) out of bounds for size {:?}",
            self.size
        );
        let idx = self
            .checked_elem_offset(x, y)
            .expect("ImageRefMut::pixel_at_mut: element offset overflowed or escaped allocation");
        // SAFETY: bounds checked above; `idx < self.data_len` verified.
        unsafe { &mut *self.data.add(idx) }
    }
}

impl<T: Copy> SubViewMut for ImageRefMut<'_, T> {
    type SubMut<'b>
        = ImageRefMut<'b, T>
    where
        Self: 'b;

    fn roi_mut(&mut self, rect: Rectangle) -> Option<ImageRefMut<'_, T>> {
        let right = rect.checked_right()?;
        let bottom = rect.checked_bottom()?;
        if right <= self.size.width && bottom <= self.size.height {
            let offset = self
                .offset
                .checked_add(rect.top().checked_mul(self.stride)?)?
                .checked_add(rect.left())?;
            // SAFETY: rect within bounds (checked above); &mut self guarantees exclusivity.
            Some(unsafe {
                ImageRefMut::strided(rect.size, self.stride, offset, self.data, self.data_len)
            })
        } else {
            None
        }
    }
}

// ImageRefMut also implements SubView (read-only roi yields an ImageRef)
impl<T: Copy> SubView for ImageRefMut<'_, T> {
    type Sub<'b>
        = ImageRef<'b, T>
    where
        Self: 'b;

    fn roi(&self, rect: Rectangle) -> Option<ImageRef<'_, T>> {
        let right = rect.checked_right()?;
        let bottom = rect.checked_bottom()?;
        if right <= self.size.width && bottom <= self.size.height {
            let offset = self
                .offset
                .checked_add(rect.top().checked_mul(self.stride)?)?
                .checked_add(rect.left())?;
            // SAFETY: we only produce a shared reference; the raw pointer is valid for reads
            let slice = unsafe { std::slice::from_raw_parts(self.data, self.data_len) };
            ImageRef::strided(rect.size, self.stride, offset, slice)
        } else {
            None
        }
    }
}

// ───────────────────────────────────────────────────────────────────
// Image<T> — heap-allocated image
// ───────────────────────────────────────────────────────────────────

/// An image stored in a heap-allocated array.
///
/// The dimensions of the image are specified at runtime.
///
/// # Example
/// ```
/// # use fovea::image::{ImageView, ImageViewMut, Image};
/// # use fovea::pixel::MonoF32;
/// // ADR-0044 / ADR-0045 Phase C: `MonoF32` is the named pixel type
/// // over `f32` — use it whenever an image holds pixel data rather
/// // than scalar kernel weights.
/// let mut img: Image<MonoF32> = Image::fill(3, 4, MonoF32(17.0));
///
/// assert_eq!(img.width(), 3);
/// assert_eq!(img.height(), 4);
/// assert_eq!(img.get(0, 0), Some(MonoF32(17.0)));
///
/// if let Some(pixel) = img.get_mut(0, 0) {
///     *pixel = MonoF32(10.0);
/// }
/// assert_eq!(img.get(0, 0), Some(MonoF32(10.0)));
/// ```
///
/// The `Image` can be generated using a closure:
/// ```
/// # use fovea::Size;
/// # use fovea::image::{ImageView, Image};
/// let img: Image<u8> = Image::generate(3, 4, |x,y| (x + y * 3) as u8 + 1);
/// assert_eq!(img.size(), Size::new(3, 4));
/// assert_eq!(img.get(0, 0), Some(1));
/// assert_eq!(img.get(1, 2), Some(8));
/// assert_eq!(img.get(2, 3), Some(12));
/// ```
#[derive(Clone)]
pub struct Image<T> {
    size: Size,
    data: Box<[T]>,
}

impl<T> Image<T> {
    /// Create an `Image` from a `Vec` of pixel data.
    ///
    /// Returns `Err` if `data.len() != width * height`.
    pub fn from_vec(width: usize, height: usize, data: Vec<T>) -> Result<Self, Error> {
        let size = Size::new(width, height);
        let expected = size.checked_area().ok_or(Error::LengthMismatch {
            expected: usize::MAX,
            actual: data.len(),
        })?;
        if data.len() == expected {
            Ok(Self {
                size,
                data: data.into_boxed_slice(),
            })
        } else {
            Err(Error::LengthMismatch {
                expected,
                actual: data.len(),
            })
        }
    }

    /// Create an `Image` by reinterpreting raw bytes as pixels, without copying.
    ///
    /// This performs a zero-copy transmute of the underlying allocation.
    /// It is only available when `T::ALIGN == 1` (true for all standard 8-bit
    /// pixel types), because the allocation layout of `Vec<u8>` and `Box<[T]>`
    /// must match for safe deallocation.
    ///
    /// Calling this with a pixel type whose alignment is greater than 1 is a
    /// **compile-time error** (enforced by `AssertByteAligned`).
    ///
    /// Returns `Err` if:
    /// - `bytes.len() != width * height * T::SIZE`
    ///
    /// # Examples
    ///
    /// ```
    /// # use fovea::image::{ImageView, Image};
    /// # use fovea::pixel::Mono8;
    /// let raw = vec![10u8, 20, 30, 40, 50, 60];
    /// let img: Image<Mono8> = Image::from_raw_bytes(3, 2, raw).unwrap();
    /// assert_eq!(img.width(), 3);
    /// assert_eq!(img.height(), 2);
    /// ```
    ///
    /// `Mono16` (and any pixel with `ALIGN > 1`) is rejected at compile
    /// time, not at runtime:
    ///
    /// ```compile_fail
    /// # use fovea::image::Image;
    /// # use fovea::pixel::Mono16;
    /// let raw = vec![0u8; 4];
    /// // ERROR: T::ALIGN > 1 — use Image::from_bytes_copy for aligned pixels.
    /// let _: Image<Mono16> = Image::from_raw_bytes(2, 1, raw).unwrap();
    /// ```
    pub fn from_raw_bytes(width: usize, height: usize, bytes: Vec<u8>) -> Result<Self, Error>
    where
        T: PlainPixel,
    {
        // Compile-time guarantees:
        // - T must have alignment 1 (otherwise zero-copy transmute UB).
        // - T::SIZE must match `size_of::<T>()` and ALIGN match
        //   `align_of::<T>()` (PHILOSOPHY.md §5: layout is a contract).
        //   This guards against a bad `unsafe impl PlainChannel` that
        //   would otherwise let us reinterpret bytes against an
        //   incorrect element size.
        let () = <T as AssertByteAligned>::_ASSERT;
        let () = <T as PlainChannel>::_ASSERT_SIZE;

        let size = Size::new(width, height);
        let expected = width
            .saturating_mul(height)
            .saturating_mul(<T as PlainChannel>::SIZE);
        if bytes.len() != expected {
            return Err(Error::LengthMismatch {
                expected,
                actual: bytes.len(),
            });
        }
        // Convert Vec<u8> → Box<[u8]>.  This shrinks capacity to match len,
        // which may reallocate if capacity > len (typically a no-op after
        // truncate when the allocator gave the exact requested size).
        let boxed_bytes: Box<[u8]> = bytes.into_boxed_slice();
        let count = size.area();
        let raw_ptr = Box::into_raw(boxed_bytes) as *mut u8;
        // SAFETY:
        // - T: PlainPixel → valid for any bit pattern, no padding
        // - T::ALIGN == 1 guaranteed at compile time by AssertByteAligned
        //   → layout matches:
        //   Box<[u8]> deallocates with Layout { size: expected, align: 1 }
        //   Box<[T]>  deallocates with Layout { size: count * T::SIZE, align: 1 }
        //   These are identical since count * T::SIZE == expected.
        // - count * T::SIZE == expected == boxed_bytes.len() (exact fit)
        let data =
            unsafe { Box::from_raw(std::slice::from_raw_parts_mut(raw_ptr as *mut T, count)) };
        Ok(Self { size, data })
    }

    /// Create an `Image` by copying and reinterpreting raw bytes as pixels.
    ///
    /// Unlike [`from_raw_bytes`](Self::from_raw_bytes), this method works for
    /// all `PlainPixel` types regardless of alignment. It copies the byte data
    /// into a properly aligned allocation.
    ///
    /// Returns `Err(Error::LengthMismatch)` if `bytes.len() != width * height * T::SIZE`.
    ///
    /// # Examples
    ///
    /// ```
    /// use fovea::image::{Image, ImageView};
    /// use fovea::pixel::Mono16;
    ///
    /// // Mono16 has ALIGN == 2, so from_raw_bytes won't compile.
    /// // from_bytes_copy works for any PlainPixel type.
    /// let bytes: Vec<u8> = vec![0x00, 0x01, 0x02, 0x03]; // two u16 values
    /// let img: Image<Mono16> = Image::from_bytes_copy(2, 1, &bytes).unwrap();
    /// assert_eq!(img.width(), 2);
    /// ```
    pub fn from_bytes_copy(width: usize, height: usize, bytes: &[u8]) -> Result<Self, Error>
    where
        T: PlainPixel,
    {
        let size = Size::new(width, height);
        let expected = width
            .saturating_mul(height)
            .saturating_mul(<T as PlainChannel>::SIZE);
        if bytes.len() != expected {
            return Err(Error::LengthMismatch {
                expected,
                actual: bytes.len(),
            });
        }
        let count = size.area();
        let mut data = Vec::with_capacity(count);
        for chunk in bytes.chunks_exact(<T as PlainChannel>::SIZE) {
            // PlainChannel::from_bytes uses ptr::read_unaligned internally,
            // so this handles arbitrary source alignment correctly.
            // chunks_exact guarantees each chunk has exactly T::SIZE bytes,
            // so from_bytes always returns Some.
            data.push(
                <T as PlainChannel>::from_bytes(chunk)
                    .expect("chunk size guaranteed by chunks_exact"),
            );
        }
        Ok(Self {
            size,
            data: data.into_boxed_slice(),
        })
    }

    pub fn generate(width: usize, height: usize, f: impl Fn(usize, usize) -> T) -> Self {
        let size = Size::new(width, height);
        let mut data = Vec::with_capacity(size.area());
        for y in 0..height {
            for x in 0..width {
                data.push(f(x, y));
            }
        }
        Self {
            size,
            data: data.into_boxed_slice(),
        }
    }
}

impl<T: ZeroablePixel> Image<T> {
    pub fn zero(width: usize, height: usize) -> Self {
        let size = Size::new(width, height);
        let data = vec![T::zero(); size.area()];
        Self {
            size,
            data: data.into_boxed_slice(),
        }
    }
}

impl<T: Clone> Image<T> {
    /// Construct an image of the given dimensions, filled with copies of `value`.
    ///
    /// Only requires `T: Clone` — the value is cloned into every pixel.
    /// This is intentionally weaker than `Image::zero`'s `ZeroablePixel`
    /// bound: filling does not need a meaningful zero, just the ability
    /// to duplicate the supplied seed.
    pub fn fill(width: usize, height: usize, value: T) -> Self {
        let size = Size::new(width, height);
        let data = vec![value; size.area()];
        Self {
            size,
            data: data.into_boxed_slice(),
        }
    }
}

// ── Debug ─────────────────────────────────────────────────────────────────────────────────────────────────
//
// The default `derive(Debug)` would print every pixel — useless for a
// typical industrial frame (e.g. 1080p × Srgb8 ≈ 6 MB of formatted text).
// We print a fixed-size summary instead: dimensions + pixel type name.
//
// `Debug` has no "depth" knob in stable Rust; `{:#?}` only toggles
// pretty-printing. So we always emit a compact, O(1)-sized record.
// Callers who need pixel data should iterate or `as_bytes()` explicitly.
//
// No `T: Debug` bound, so `Image<T>` is `Debug` for any `T`. The pixel
// type name is recovered via `std::any::type_name::<T>()` (best-effort
// diagnostic string — the stdlib does not guarantee a stable format,
// but it is stable enough for human reading).

impl<T> fmt::Debug for Image<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pixel_ty = std::any::type_name::<T>();
        // Strip the leading module path for readability:
        //   "fovea::pixel::srgb::Srgb8"  →  "Srgb8"
        let pixel_short = pixel_ty.rsplit("::").next().unwrap_or(pixel_ty);
        f.debug_struct("Image")
            .field("width", &self.size.width)
            .field("height", &self.size.height)
            .field("pixel", &pixel_short)
            .finish()
    }
}

// ── PartialEq / Eq ─────────────────────────────────────────────────────────────────────────────────────
//
// Two images are equal iff their dimensions match and every pixel
// compares equal. Dimensions are checked first so we never compare
// pixel slices of different lengths.

impl<T: PartialEq> PartialEq for Image<T> {
    fn eq(&self, other: &Self) -> bool {
        self.size == other.size && self.data[..] == other.data[..]
    }
}

impl<T: Eq> Eq for Image<T> {}

impl<T: Copy> ImageView for Image<T> {
    type Pixel = T;

    fn size(&self) -> Size {
        self.size
    }
    fn width(&self) -> usize {
        self.size.width
    }
    fn height(&self) -> usize {
        self.size.height
    }

    #[inline(always)]
    fn get(&self, x: usize, y: usize) -> Option<Self::Pixel> {
        if x < self.size.width && y < self.size.height {
            Some(self.data[internal::index(&self.size, x, y)])
        } else {
            None
        }
    }

    #[inline(always)]
    fn pixel_at(&self, x: usize, y: usize) -> Self::Pixel {
        self.data[internal::checked_index_or_panic(&self.size, x, y)]
    }
}

impl<T: Copy> ImageViewMut for Image<T> {
    #[inline(always)]
    fn pixel_at_mut(&mut self, x: usize, y: usize) -> &mut Self::Pixel {
        let idx = internal::checked_index_or_panic(&self.size, x, y);
        &mut self.data[idx]
    }
}

impl<T: Copy> SubView for Image<T> {
    type Sub<'a>
        = ImageRef<'a, T>
    where
        Self: 'a;

    fn roi(&self, rect: Rectangle) -> Option<ImageRef<'_, T>> {
        let right = rect.checked_right()?;
        let bottom = rect.checked_bottom()?;
        if right <= self.width() && bottom <= self.height() {
            let width = self.width();
            let offset = rect.top().checked_mul(width)?.checked_add(rect.left())?;
            ImageRef::strided(rect.size, width, offset, self.as_slice())
        } else {
            None
        }
    }
}

impl<T: Copy> SubViewMut for Image<T> {
    type SubMut<'a>
        = ImageRefMut<'a, T>
    where
        Self: 'a;

    fn roi_mut(&mut self, rect: Rectangle) -> Option<ImageRefMut<'_, T>> {
        let right = rect.checked_right()?;
        let bottom = rect.checked_bottom()?;
        if right <= self.width() && bottom <= self.height() {
            let stride = self.width();
            let offset = rect.top().checked_mul(stride)?.checked_add(rect.left())?;
            let slice = self.as_mut_slice();
            let len = slice.len();
            let ptr = slice.as_mut_ptr();
            Some(unsafe { ImageRefMut::strided(rect.size, stride, offset, ptr, len) })
        } else {
            None
        }
    }
}

impl<T: Copy> ImageView for &Image<T> {
    type Pixel = T;

    fn size(&self) -> Size {
        (**self).size()
    }
    fn width(&self) -> usize {
        (**self).width()
    }
    fn height(&self) -> usize {
        (**self).height()
    }

    #[inline(always)]
    fn get(&self, x: usize, y: usize) -> Option<Self::Pixel> {
        (**self).get(x, y)
    }

    #[inline(always)]
    fn pixel_at(&self, x: usize, y: usize) -> Self::Pixel {
        (**self).pixel_at(x, y)
    }
}

impl<T: Copy> RasterImage for &Image<T> {
    #[inline(always)]
    fn row(&self, y: usize) -> &[Self::Pixel] {
        (**self).row(y)
    }
}

impl<T: Copy> ImageView for &mut Image<T> {
    type Pixel = T;

    fn size(&self) -> Size {
        (**self).size()
    }
    fn width(&self) -> usize {
        (**self).width()
    }
    fn height(&self) -> usize {
        (**self).height()
    }

    #[inline(always)]
    fn get(&self, x: usize, y: usize) -> Option<Self::Pixel> {
        (**self).get(x, y)
    }

    #[inline(always)]
    fn pixel_at(&self, x: usize, y: usize) -> Self::Pixel {
        (**self).pixel_at(x, y)
    }
}

impl<T: Copy> ImageViewMut for &mut Image<T> {
    #[inline(always)]
    fn get_mut(&mut self, x: usize, y: usize) -> Option<&mut Self::Pixel> {
        (*self).get_mut(x, y)
    }

    #[inline(always)]
    fn pixel_at_mut(&mut self, x: usize, y: usize) -> &mut Self::Pixel {
        (*self).pixel_at_mut(x, y)
    }
}

impl<T: Copy> RasterImage for &mut Image<T> {
    #[inline(always)]
    fn row(&self, y: usize) -> &[Self::Pixel] {
        (**self).row(y)
    }
}

impl<T: Copy> RasterImageMut for &mut Image<T> {
    #[inline(always)]
    fn row_mut(&mut self, y: usize) -> &mut [Self::Pixel] {
        (*self).row_mut(y)
    }
}

impl<T: Copy> RasterImage for Image<T> {
    #[inline(always)]
    fn row(&self, y: usize) -> &[Self::Pixel] {
        let (start, end) = internal::checked_row_range_or_panic(&self.size, y);
        &self.data[start..end]
    }
}

impl<T: Copy> RasterImageMut for Image<T> {
    #[inline(always)]
    fn row_mut(&mut self, y: usize) -> &mut [Self::Pixel] {
        let (start, end) = internal::checked_row_range_or_panic(&self.size, y);
        &mut self.data[start..end]
    }
}

impl<T: Copy> contiguous_sealed::Sealed for Image<T> {}

impl<T: Copy> ContiguousImage for Image<T> {
    fn as_slice(&self) -> &[Self::Pixel] {
        &self.data
    }
}

impl<T: Copy> ContiguousImageMut for Image<T> {
    fn as_mut_slice(&mut self) -> &mut [Self::Pixel] {
        &mut self.data
    }
}

impl<T: PlainPixel> PlainImage for Image<T> {
    fn as_bytes(&self) -> &[u8] {
        unsafe { internal::as_bytes(self.as_slice()) }
    }

    fn as_bytes_le(&self) -> Cow<'_, [u8]> {
        unsafe { internal::as_bytes_le(self.as_slice()) }
    }

    fn as_bytes_be(&self) -> Cow<'_, [u8]> {
        unsafe { internal::as_bytes_be(self.as_slice()) }
    }
}

impl<T: PlainPixel> PlainImageMut for Image<T> {
    fn as_mut_bytes(&mut self) -> &mut [u8] {
        unsafe { internal::as_mut_bytes(self.as_mut_slice()) }
    }
}

impl<P: Copy, C: Into<Coordinate>> Index<C> for Image<P> {
    type Output = P;

    #[inline(always)]
    fn index(&self, index: C) -> &Self::Output {
        let Coordinate { x, y } = index.into();
        &self.data[internal::checked_index_or_panic(&self.size, x, y)]
    }
}

impl<P: Copy, C: Into<Coordinate>> IndexMut<C> for Image<P> {
    #[inline(always)]
    fn index_mut(&mut self, index: C) -> &mut Self::Output {
        let Coordinate { x, y } = index.into();
        self.pixel_at_mut(x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::SubView;
    use crate::pixel::{Mono8, Mono10, Mono16, MonoA8, Rgb8, Rgb10, Rgb16, RgbF32, Rgba8};

    #[test]
    fn test_image_array() {
        let img: ImageArray<u8, 2, 2> = ImageArray::new([0; 4]);
        assert_eq!(img.width(), 2);
        assert_eq!(img.height(), 2);
    }

    #[test]
    fn test_image_array_get() {
        let mut img = ImageArray::<u8, 2, 2> { data: [1, 2, 3, 4] };
        assert_eq!(img.get(0, 0), Some(1));
        assert_eq!(img.get(1, 0), Some(2));
        assert_eq!(img.get(0, 1), Some(3));
        assert_eq!(img[(1, 1)], 4);
        assert_eq!(img.get(2, 2), None);

        *img.get_mut(0, 0).unwrap() = 10;
        assert_eq!(img.get(0, 0), Some(10));
    }

    #[test]
    fn test_compile_time_size_check() {
        // This should compile
        let _img: ImageArray<u8, 3, 3> = ImageArray::new([0; 9]);

        // Uncommenting the following line should cause a compile-time error
        // because the array size does not match W * H.
        // let _img_invalid: ImageArray<u8, 3, 3> = ImageArray::new([0; 8]);
    }

    #[test]
    fn test_rgb_image_to_bytes() {
        let img: ImageArray<Rgb8, 2, 2> = ImageArray::new([
            Rgb8::new(10, 20, 30),
            Rgb8::new(40, 50, 60),
            Rgb8::new(70, 80, 90),
            Rgb8::new(100, 110, 120),
        ]);
        let bytes = img.as_bytes();
        assert_eq!(bytes.len(), 2 * 2 * 3);
        assert_eq!(bytes, &[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]);

        let bytes_be = img.as_bytes_be();
        assert_eq!(&*bytes_be, bytes);
        assert_eq!(bytes, &[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]);
    }

    #[test]
    fn test_mono10_image_to_bytes() {
        let img: ImageArray<Mono10, 2, 2> = ImageArray::new([
            Mono10::new(512),
            Mono10::new(1023),
            Mono10::new(256),
            Mono10::new(0),
        ]);
        let bytes = img.as_bytes();
        assert_eq!(bytes.len(), 2 * 2 * 2);
        assert_eq!(bytes, &[0, 2, 255, 3, 0, 1, 0, 0]);
        assert_eq!(&*img.as_bytes_le(), bytes);

        let bytes_be = img.as_bytes_be();
        assert_eq!(&*bytes_be, &[2, 0, 3, 255, 1, 0, 0, 0]);
    }

    #[test]
    fn test_image2d() {
        let mut img = Image::<u8>::zero(2, 3);
        assert_eq!(img.width(), 2);
        assert_eq!(img.height(), 3);
        assert_eq!(img.get(0, 0), Some(0));
        assert_eq!(img.get(1, 2), Some(0));
        assert_eq!(img.get(2, 2), None);

        *img.get_mut(1, 1).unwrap() = 42;
        assert_eq!(img.get(1, 1), Some(42));

        let bytes = img.as_bytes();
        assert_eq!(bytes.len(), 2 * 3);
        assert_eq!(bytes, &[0, 0, 0, 42, 0, 0]);
    }

    #[test]
    fn test_rgb10_image2d() {
        let mut img: Image<Rgb10> = Image::fill(2, 2, Rgb10::new(1, 2, 3));
        assert_eq!(img.width(), 2);
        assert_eq!(img.height(), 2);
        assert_eq!(img.get(0, 0), Some(Rgb10::new(1, 2, 3)));
        assert_eq!(img.get(1, 1), Some(Rgb10::new(1, 2, 3)));

        *img.get_mut(1, 0).unwrap() = Rgb10::new(10, 20, 30);
        assert_eq!(img.get(1, 0), Some(Rgb10::new(10, 20, 30)));

        let bytes = img.as_bytes();
        assert_eq!(bytes.len(), 2 * 2 * 3 * 2);
        assert_eq!(
            bytes,
            &[
                1, 0, 2, 0, 3, 0, 10, 0, 20, 0, 30, 0, 1, 0, 2, 0, 3, 0, 1, 0, 2, 0, 3, 0
            ]
        );

        let bytes_be = img.as_bytes_be();
        assert_eq!(
            &*bytes_be,
            &[
                0, 1, 0, 2, 0, 3, 0, 10, 0, 20, 0, 30, 0, 1, 0, 2, 0, 3, 0, 1, 0, 2, 0, 3
            ]
        );
    }

    #[test]
    fn test_generate_image_array() {
        let img: ImageArray<Rgb16, 3, 4> =
            ImageArray::generate(|x, y| Rgb16::new(x as u16, y as u16, (x + y) as u16));
        assert_eq!(img.size(), Size::new(3, 4));
        assert_eq!(img.get(0, 0), Some(Rgb16::new(0, 0, 0)));
        assert_eq!(img.get(1, 2), Some(Rgb16::new(1, 2, 3)));
        assert_eq!(img.get(2, 3), Some(Rgb16::new(2, 3, 5)));
    }

    #[test]
    fn test_generate_image2d() {
        let img: Image<Rgb16> =
            Image::generate(3, 4, |x, y| Rgb16::new(x as u16, y as u16, (x + y) as u16));
        assert_eq!(img.size(), Size::new(3, 4));
        assert_eq!(img.get(0, 0), Some(Rgb16::new(0, 0, 0)));
        assert_eq!(img.get(1, 2), Some(Rgb16::new(1, 2, 3)));
        assert_eq!(img.get(2, 3), Some(Rgb16::new(2, 3, 5)));

        let img: Image<RgbF32> =
            Image::generate(2, 2, |x, y| RgbF32::new(x as f32, y as f32, (x + y) as f32));
        assert_eq!(img.size(), Size::new(2, 2));
        assert_eq!(img.get(0, 0), Some(RgbF32::new(0.0, 0.0, 0.0)));
        assert_eq!(img.get(1, 0), Some(RgbF32::new(1.0, 0.0, 1.0)));
        assert_eq!(img.get(0, 1), Some(RgbF32::new(0.0, 1.0, 1.0)));
        assert_eq!(img.get(1, 1), Some(RgbF32::new(1.0, 1.0, 2.0)));
    }

    #[test]
    fn test_roi() {
        let img: ImageArray<u8, 4, 4> = ImageArray::generate(|x, y| (x + y * 4) as u8 + 1);
        let roi = img.roi(Rectangle::new((1, 1), (2, 2))).unwrap();
        assert_eq!(roi.size(), Size::new(2, 2));
        assert_eq!(roi.get(0, 0), Some(6));
        assert_eq!(roi.get(1, 0), Some(7));
        assert_eq!(roi.get(0, 1), Some(10));
        assert_eq!(roi.get(1, 1), Some(11));
    }

    #[test]
    fn test_roi_image2d() {
        let img: Image<u8> = Image::generate(4, 4, |x, y| (x + y * 4) as u8 + 1);
        let roi = img.roi(Rectangle::new((1, 1), (2, 2))).unwrap();
        assert_eq!(roi.size(), Size::new(2, 2));
        assert_eq!(roi.get(0, 0), Some(6));
        assert_eq!(roi.get(1, 0), Some(7));
        assert_eq!(roi.get(0, 1), Some(10));
        assert_eq!(roi.get(1, 1), Some(11));
    }

    #[test]
    fn test_tiles_iter() {
        let img: ImageArray<u8, 4, 4> = ImageArray::generate(|x, y| (x + y * 4) as u8 + 1);
        let img = &img;
        let mut iter = img.into_tiles(Size::new(2, 2));

        let roi = iter.next().unwrap();
        assert_eq!(roi.size(), Size::new(2, 2));
        assert_eq!(roi.get(0, 0), Some(1));
        assert_eq!(roi.get(1, 0), Some(2));
        assert_eq!(roi.get(0, 1), Some(5));
        assert_eq!(roi.get(1, 1), Some(6));

        //        let roi = iter.next().unwrap();
        //        assert_eq!(roi.size(), Size::new(2, 2));
        //        assert_eq!(roi.get(0, 0), Some(&3));
        //        assert_eq!(roi.get(1, 0), Some(&4));
        //        assert_eq!(roi.get(0, 1), Some(&7));
        //        assert_eq!(roi.get(1, 1), Some(&8));
        //
        //        let roi = iter.next().unwrap();
        //        assert_eq!(roi.size(), Size::new(2, 2));
        //        assert_eq!(roi.get(0, 0), Some(&9));
        //        assert_eq!(roi.get(1, 0), Some(&10));
        //        assert_eq!(roi.get(0, 1), Some(&13));
        //        assert_eq!(roi.get(1, 1), Some(&14));
        //
        //        let roi = iter.next().unwrap();
        //        assert_eq!(roi.size(), Size::new(2, 2));
        //        assert_eq!(roi.get(0, 0), Some(&11));
        //        assert_eq!(roi.get(1, 0), Some(&12));
        //        assert_eq!(roi.get(0, 1), Some(&15));
        //        assert_eq!(roi.get(1, 1), Some(&16));
        //
        //        assert_eq!(iter.next().is_none(), true);
    }

    #[test]
    fn test_roi_out_of_bounds() {
        let img: ImageArray<u8, 4, 4> = ImageArray::generate(|x, y| (x + y * 4) as u8 + 1);
        // ROI extends beyond image bounds
        let roi = img.roi(Rectangle::new((3, 3), (2, 2)));
        assert!(roi.is_none());
    }

    #[test]
    fn test_roi_mut_out_of_bounds() {
        let mut img: Image<u8> = Image::generate(4, 4, |x, y| (x + y * 4) as u8 + 1);
        // ROI extends beyond image bounds
        let roi = img.roi_mut(Rectangle::new((3, 3), (2, 2)));
        assert!(roi.is_none());
    }

    #[test]
    fn test_roi_mut() {
        let mut img: Image<u8> = Image::generate(4, 4, |x, y| (x + y * 4) as u8 + 1);
        let mut roi = img.roi_mut(Rectangle::new((1, 1), (2, 2))).unwrap();
        assert_eq!(roi.size(), Size::new(2, 2));
        assert_eq!(roi.get(0, 0), Some(6));

        // Modify through ROI
        *roi.get_mut(0, 0).unwrap() = 100;
        assert_eq!(roi.get(0, 0), Some(100));
    }

    #[test]
    fn test_roi_get_out_of_bounds() {
        let img: ImageArray<u8, 4, 4> = ImageArray::generate(|x, y| (x + y * 4) as u8 + 1);
        let roi = img.roi(Rectangle::new((1, 1), (2, 2))).unwrap();
        assert_eq!(roi.get(2, 0), None);
        assert_eq!(roi.get(0, 2), None);
        assert_eq!(roi.get(3, 3), None);
    }

    #[test]
    fn test_image_from_vec() {
        let data = vec![1u8, 2, 3, 4, 5, 6];
        let img = Image::from_vec(3, 2, data);
        assert!(img.is_ok());
        let img = img.unwrap();
        assert_eq!(img.width(), 3);
        assert_eq!(img.height(), 2);
    }

    #[test]
    fn test_image_from_vec_wrong_size() {
        let data = vec![1u8, 2, 3, 4, 5];
        let img = Image::from_vec(3, 2, data);
        assert!(img.is_err());
    }

    #[test]
    fn test_image_from_raw_bytes_mono8() {
        use crate::pixel::Mono8;
        let raw = vec![10u8, 20, 30, 40, 50, 60];
        let img: Image<Mono8> = Image::from_raw_bytes(3, 2, raw).unwrap();
        assert_eq!(img.width(), 3);
        assert_eq!(img.height(), 2);
        assert_eq!(img.get(0, 0), Some(Mono8::new(10)));
        assert_eq!(img.get(2, 1), Some(Mono8::new(60)));
    }

    #[test]
    fn test_image_from_raw_bytes_rgb8() {
        use crate::pixel::Rgb8;
        let raw = vec![10, 20, 30, 40, 50, 60];
        let img: Image<Rgb8> = Image::from_raw_bytes(2, 1, raw).unwrap();
        assert_eq!(img.get(0, 0), Some(Rgb8::new(10, 20, 30)));
        assert_eq!(img.get(1, 0), Some(Rgb8::new(40, 50, 60)));
    }

    #[test]
    fn test_image_from_raw_bytes_wrong_size() {
        use crate::pixel::Mono8;
        let raw = vec![10u8, 20, 30, 40, 50];
        assert!(Image::<Mono8>::from_raw_bytes(3, 2, raw).is_err());
    }

    #[test]
    fn test_image_fill() {
        let img: Image<u8> = Image::fill(3, 3, 42);
        for y in 0..3 {
            for x in 0..3 {
                assert_eq!(img.get(x, y), Some(42));
            }
        }
    }

    #[test]
    fn test_image_indexing() {
        let mut img: Image<u8> = Image::generate(3, 3, |x, y| (x + y * 3) as u8);
        assert_eq!(img[(0, 0)], 0);
        assert_eq!(img[(1, 0)], 1);
        assert_eq!(img[(2, 1)], 5);

        img[(1, 1)] = 99;
        assert_eq!(img[(1, 1)], 99);
    }

    #[test]
    fn test_image_array_indexing() {
        let mut img: ImageArray<u8, 3, 3> = ImageArray::generate(|x, y| (x + y * 3) as u8);
        assert_eq!(img[(0, 0)], 0);
        assert_eq!(img[(1, 0)], 1);
        assert_eq!(img[(2, 1)], 5);

        img[(1, 1)] = 99;
        assert_eq!(img[(1, 1)], 99);
    }

    #[test]
    fn test_image_array_as_slice() {
        let img: ImageArray<u8, 2, 2> = ImageArray::new([1, 2, 3, 4]);
        let slice = img.as_slice();
        assert_eq!(slice, &[1, 2, 3, 4]);
    }

    #[test]
    fn test_image_array_as_mut_slice() {
        let mut img: ImageArray<u8, 2, 2> = ImageArray::new([1, 2, 3, 4]);
        {
            let slice = img.as_mut_slice();
            slice[0] = 10;
            slice[3] = 40;
        }
        assert_eq!(img.get(0, 0), Some(10));
        assert_eq!(img.get(1, 1), Some(40));
    }

    #[test]
    fn test_image_as_slice() {
        let img: Image<u8> = Image::generate(2, 2, |x, y| (x + y * 2) as u8 + 1);
        let slice = img.as_slice();
        assert_eq!(slice, &[1, 2, 3, 4]);
    }

    #[test]
    fn test_image_as_mut_slice() {
        let mut img: Image<u8> = Image::generate(2, 2, |x, y| (x + y * 2) as u8 + 1);
        {
            let slice = img.as_mut_slice();
            slice[0] = 10;
            slice[3] = 40;
        }
        assert_eq!(img.get(0, 0), Some(10));
        assert_eq!(img.get(1, 1), Some(40));
    }

    #[test]
    fn test_image_array_as_mut_bytes() {
        let mut img: ImageArray<u8, 2, 2> = ImageArray::new([1, 2, 3, 4]);
        {
            let bytes = img.as_mut_bytes();
            bytes[0] = 10;
            bytes[3] = 40;
        }
        assert_eq!(img.get(0, 0), Some(10));
        assert_eq!(img.get(1, 1), Some(40));
    }

    #[test]
    fn test_image_as_mut_bytes() {
        let mut img: Image<u8> = Image::generate(2, 2, |x, y| (x + y * 2) as u8 + 1);
        {
            let bytes = img.as_mut_bytes();
            bytes[0] = 10;
            bytes[3] = 40;
        }
        assert_eq!(img.get(0, 0), Some(10));
        assert_eq!(img.get(1, 1), Some(40));
    }

    #[test]
    fn test_image_ref_imageview() {
        let img: Image<u8> = Image::generate(3, 4, |x, y| (x + y * 3) as u8 + 1);
        let img_ref = &img;
        assert_eq!(img_ref.width(), 3);
        assert_eq!(img_ref.height(), 4);
        assert_eq!(img_ref.size(), Size::new(3, 4));
        assert_eq!(img_ref.get(1, 2), Some(8));
        assert_eq!(img_ref.pixel_at(2, 3), 12);

        // Use a generic function to force dispatch through `impl ImageView for &Image<T>`
        fn check_view<V: ImageView<Pixel = u8>>(v: V) {
            assert_eq!(v.size(), Size::new(3, 4));
            assert_eq!(v.width(), 3);
            assert_eq!(v.height(), 4);
            assert_eq!(v.get(1, 2), Some(8));
            assert_eq!(v.get(99, 99), None);
            assert_eq!(v.pixel_at(2, 3), 12);
        }
        check_view(&img);
    }

    #[test]
    fn test_image_mut_ref_imageview() {
        let mut img: Image<u8> = Image::generate(3, 4, |x, y| (x + y * 3) as u8 + 1);
        {
            let img_ref = &mut img;
            assert_eq!(img_ref.width(), 3);
            assert_eq!(img_ref.height(), 4);
            assert_eq!(img_ref.size(), Size::new(3, 4));
            assert_eq!(img_ref.get(1, 2), Some(8));
            assert_eq!(img_ref.pixel_at(2, 3), 12);
        }

        // Use a generic function to force dispatch through `impl ImageView for &mut Image<T>`
        fn check_view<V: ImageView<Pixel = u8>>(v: V) {
            assert_eq!(v.size(), Size::new(3, 4));
            assert_eq!(v.width(), 3);
            assert_eq!(v.height(), 4);
            assert_eq!(v.get(1, 2), Some(8));
            assert_eq!(v.get(99, 99), None);
            assert_eq!(v.pixel_at(2, 3), 12);
        }
        check_view(&mut img);
    }

    #[test]
    fn test_image_mut_ref_imageviewmut() {
        let mut img: Image<u8> = Image::generate(3, 4, |x, y| (x + y * 3) as u8 + 1);
        {
            let img_ref = &mut img;
            *img_ref.get_mut(1, 2).unwrap() = 100;
            assert_eq!(img_ref.get(1, 2), Some(100));

            let pixel = img_ref.pixel_at_mut(2, 3);
            *pixel = 200;
            assert_eq!(img_ref.get(2, 3), Some(200));
        }

        // Use a generic function to force dispatch through `impl ImageViewMut for &mut Image<T>`
        fn check_view_mut<V: ImageViewMut<Pixel = u8>>(mut v: V) {
            *v.pixel_at_mut(0, 0) = 42;
            assert_eq!(v.get_mut(0, 0), Some(&mut 42));
            assert_eq!(v.get_mut(99, 99), None);
        }
        check_view_mut(&mut img);
    }

    #[test]
    fn test_roi_width_height() {
        let img: ImageArray<u8, 4, 4> = ImageArray::generate(|x, y| (x + y * 4) as u8 + 1);
        let roi = img.roi(Rectangle::new((1, 1), (2, 2))).unwrap();
        assert_eq!(roi.width(), 2);
        assert_eq!(roi.height(), 2);
    }

    #[test]
    fn test_roi_mut_width_height() {
        let mut img: Image<u8> = Image::generate(4, 4, |x, y| (x + y * 4) as u8 + 1);
        let roi = img.roi_mut(Rectangle::new((1, 1), (2, 2))).unwrap();
        assert_eq!(roi.width(), 2);
        assert_eq!(roi.height(), 2);
    }

    #[test]
    fn test_roi_pixel_at() {
        let img: ImageArray<u8, 4, 4> = ImageArray::generate(|x, y| (x + y * 4) as u8 + 1);
        let roi = img.roi(Rectangle::new((1, 1), (2, 2))).unwrap();
        assert_eq!(roi.pixel_at(0, 0), 6);
        assert_eq!(roi.pixel_at(1, 1), 11);
    }

    #[test]
    fn test_roi_mut_pixel_at() {
        let mut img: Image<u8> = Image::generate(4, 4, |x, y| (x + y * 4) as u8 + 1);
        let roi = img.roi_mut(Rectangle::new((1, 1), (2, 2))).unwrap();
        assert_eq!(roi.pixel_at(0, 0), 6);
        assert_eq!(roi.pixel_at(1, 1), 11);
    }

    #[test]
    fn test_roi_mut_pixel_at_mut() {
        let mut img: Image<u8> = Image::generate(4, 4, |x, y| (x + y * 4) as u8 + 1);
        let mut roi = img.roi_mut(Rectangle::new((1, 1), (2, 2))).unwrap();
        let pixel = roi.pixel_at_mut(0, 0);
        *pixel = 200;
        assert_eq!(roi.get(0, 0), Some(200));
    }

    #[test]
    fn test_roi_mut_get_mut_out_of_bounds() {
        let mut img: Image<u8> = Image::generate(4, 4, |x, y| (x + y * 4) as u8 + 1);
        let mut roi = img.roi_mut(Rectangle::new((1, 1), (2, 2))).unwrap();
        assert_eq!(roi.get_mut(2, 0), None);
        assert_eq!(roi.get_mut(0, 2), None);
    }

    #[test]
    fn test_image_array_with_different_sizes() {
        let img1: ImageArray<u8, 1, 1> = ImageArray::new([42]);
        assert_eq!(img1[(0, 0)], 42);

        let img2: ImageArray<u8, 5, 5> = ImageArray::generate(|x, y| (x + y) as u8);
        assert_eq!(img2[(4, 4)], 8);

        let img3: ImageArray<u8, 10, 1> = ImageArray::generate(|x, _| x as u8);
        assert_eq!(img3[(9, 0)], 9);

        let img4: ImageArray<u8, 1, 10> = ImageArray::generate(|_, y| y as u8);
        assert_eq!(img4[(0, 9)], 9);
    }

    #[test]
    fn test_image_array_64x64() {
        // Verify that the maximum supported dimension (64×64) works correctly.
        let img: ImageArray<u8, 64, 64> = ImageArray::generate(|x, y| ((x + y) % 256) as u8);
        assert_eq!(img.width(), 64);
        assert_eq!(img.height(), 64);
        assert_eq!(img.size(), Size::new(64, 64));
        assert_eq!(img[(0, 0)], 0);
        assert_eq!(img[(63, 63)], ((63 + 63) % 256) as u8);
        assert_eq!(img[(32, 16)], ((32 + 16) % 256) as u8);
        assert_eq!(img.get(64, 0), None); // out of bounds
        assert_eq!(img.as_slice().len(), 64 * 64);
    }

    #[test]
    fn test_image_array_32x64() {
        let img: ImageArray<u8, 32, 64> = ImageArray::generate(|x, y| ((x * 2 + y) % 256) as u8);
        assert_eq!(img.width(), 32);
        assert_eq!(img.height(), 64);
        assert_eq!(img[(31, 63)], ((31 * 2 + 63) % 256) as u8);
    }

    #[test]
    fn test_image_array_64x1() {
        let img: ImageArray<u8, 64, 1> = ImageArray::generate(|x, _| x as u8);
        assert_eq!(img.width(), 64);
        assert_eq!(img.height(), 1);
        assert_eq!(img[(63, 0)], 63);
    }

    #[test]
    fn test_image_array_1x64() {
        let img: ImageArray<u8, 1, 64> = ImageArray::generate(|_, y| y as u8);
        assert_eq!(img.width(), 1);
        assert_eq!(img.height(), 64);
        assert_eq!(img[(0, 63)], 63);
    }

    #[test]
    fn test_image_array_large_dimensions_beyond_13() {
        // Verify dimensions that were previously unsupported (> 13×13).
        let img14: ImageArray<u8, 14, 14> = ImageArray::generate(|x, y| (x + y) as u8);
        assert_eq!(img14[(13, 13)], 26);

        let img20: ImageArray<u8, 20, 20> = ImageArray::generate(|x, y| (x + y) as u8);
        assert_eq!(img20[(19, 19)], 38);

        let img50: ImageArray<u8, 50, 50> = ImageArray::generate(|x, y| ((x + y) % 256) as u8);
        assert_eq!(img50[(49, 49)], ((49 + 49) % 256) as u8);
    }

    #[test]
    fn test_size_is_copy() {
        let s1 = Size::new(640, 480);
        let s2 = s1; // Copy, not move
        assert_eq!(s1, s2); // s1 is still usable
        assert_eq!(s1.width, 640);
        assert_eq!(s2.height, 480);
    }

    #[test]
    fn test_rectangle_is_copy() {
        let r1 = Rectangle::new((10, 20), (100, 200));
        let r2 = r1; // Copy, not move
        assert_eq!(r1, r2); // r1 is still usable
        assert_eq!(r1.offset.x, 10);
        assert_eq!(r2.size.width, 100);
    }

    #[test]
    fn test_size_copy_in_function_call() {
        fn takes_size(s: Size) -> usize {
            s.area()
        }
        let s = Size::new(3, 4);
        // Pass by value (Copy) twice — both calls should work.
        assert_eq!(takes_size(s), 12);
        assert_eq!(takes_size(s), 12);
    }

    #[test]
    fn test_image_array_pixel_at_name() {
        // Verify the renamed method `pixel_at` works as expected.
        let img: ImageArray<u8, 3, 3> = ImageArray::generate(|x, y| (x + y * 3) as u8);
        assert_eq!(img.pixel_at(0, 0), 0);
        assert_eq!(img.pixel_at(2, 2), 8);
    }

    #[test]
    fn test_image_pixel_at_name() {
        // Verify the renamed method `pixel_at` works as expected.
        let img: Image<u8> = Image::generate(3, 3, |x, y| (x + y * 3) as u8);
        assert_eq!(img.pixel_at(0, 0), 0);
        assert_eq!(img.pixel_at(2, 2), 8);
    }

    #[test]
    fn test_image_pixel_at_mut_name() {
        // Verify the renamed method `pixel_at_mut` works as expected.
        let mut img: Image<u8> = Image::zero(2, 2);
        *img.pixel_at_mut(0, 0) = 42;
        *img.pixel_at_mut(1, 1) = 99;
        assert_eq!(img.pixel_at(0, 0), 42);
        assert_eq!(img.pixel_at(1, 1), 99);
    }

    #[test]
    fn test_coordinate_indexing() {
        let mut img: Image<u8> = Image::generate(3, 3, |x, y| (x + y * 3) as u8);
        let coord = Coordinate::new(1, 2);
        assert_eq!(img[coord], 7);

        img[coord] = 99;
        assert_eq!(img[coord], 99);
    }

    #[test]
    fn test_image_zero_with_zero_size() {
        let img: Image<u8> = Image::zero(0, 0);
        assert_eq!(img.width(), 0);
        assert_eq!(img.height(), 0);
        assert_eq!(img.size().area(), 0);
    }

    #[test]
    fn test_image_generate_1x1() {
        let img: Image<u8> = Image::generate(1, 1, |_, _| 42);
        assert_eq!(img.width(), 1);
        assert_eq!(img.height(), 1);
        assert_eq!(img.get(0, 0), Some(42));
    }

    #[test]
    fn test_image_array_roi_at_edges() {
        let img: ImageArray<u8, 4, 4> = ImageArray::generate(|x, y| (x + y * 4) as u8 + 1);

        // ROI at top-left corner
        let roi = img.roi(Rectangle::new((0, 0), (2, 2))).unwrap();
        assert_eq!(roi.get(0, 0), Some(1));

        // ROI near bottom-right corner (must not exceed bounds)
        let roi = img.roi(Rectangle::new((2, 2), (1, 1))).unwrap();
        assert_eq!(roi.get(0, 0), Some(11));
    }

    #[test]
    fn test_image_from_vec_exact_size() {
        let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9];
        let img = Image::from_vec(3, 3, data).unwrap();
        assert_eq!(img.get(0, 0), Some(1));
        assert_eq!(img.get(2, 2), Some(9));
    }

    #[test]
    fn test_roi_1x1_size() {
        let img: Image<u8> = Image::generate(4, 4, |x, y| (x + y * 4) as u8 + 1);
        let roi = img.roi(Rectangle::new((2, 2), (1, 1))).unwrap();
        assert_eq!(roi.width(), 1);
        assert_eq!(roi.height(), 1);
        assert_eq!(roi.get(0, 0), Some(11));
    }

    #[test]
    fn test_contiguous_image_trait() {
        let mut img: Image<u8> = Image::zero(2, 2);
        let slice = img.as_slice();
        assert_eq!(slice.len(), 4);

        let mut_slice = img.as_mut_slice();
        mut_slice[0] = 100;
        assert_eq!(img.get(0, 0), Some(100));
    }

    #[test]
    fn test_image_array_width_height() {
        let img: ImageArray<u8, 7, 9> = ImageArray::generate(|x, y| (x + y) as u8);
        assert_eq!(img.width(), 7);
        assert_eq!(img.height(), 9);
    }

    #[test]
    fn test_image_get_out_of_bounds() {
        let img: Image<u8> = Image::zero(3, 3);
        assert_eq!(img.get(3, 0), None);
        assert_eq!(img.get(0, 3), None);
        assert_eq!(img.get(10, 10), None);
    }

    #[test]
    fn test_image_array_get_out_of_bounds() {
        let img: ImageArray<u8, 3, 3> = ImageArray::generate(|x, y| (x + y) as u8);
        assert_eq!(img.get(3, 0), None);
        assert_eq!(img.get(0, 3), None);
        assert_eq!(img.get(10, 10), None);
    }

    #[test]
    fn test_roi_mut_modify_through_roi() {
        let mut img: Image<u8> = Image::generate(4, 4, |x, y| (x + y * 4) as u8 + 1);
        {
            let mut roi = img.roi_mut(Rectangle::new((1, 1), (2, 2))).unwrap();
            let pixel = roi.pixel_at_mut(1, 1);
            *pixel = 255;
        }
        // Verify change is reflected in original image
        assert_eq!(img.get(2, 2), Some(255));
    }

    // -----------------------------------------------------------------------
    // ImageArray get_mut out-of-bounds
    // -----------------------------------------------------------------------

    #[test]
    fn test_image_array_get_mut_out_of_bounds() {
        let mut img: ImageArray<u8, 3, 3> = ImageArray::generate(|x, y| (x + y) as u8);
        assert!(img.get_mut(3, 0).is_none());
        assert!(img.get_mut(0, 3).is_none());
        assert!(img.get_mut(10, 10).is_none());
        // Also verify valid get_mut works
        assert!(img.get_mut(0, 0).is_some());
    }

    // -----------------------------------------------------------------------
    // ImageArray roi_mut
    // -----------------------------------------------------------------------

    #[test]
    fn test_image_array_roi_mut_valid() {
        let mut img: ImageArray<u8, 4, 4> = ImageArray::generate(|x, y| (x + y * 4) as u8);
        {
            let mut roi = img.roi_mut(Rectangle::new((1, 1), (2, 2))).unwrap();
            assert_eq!(roi.width(), 2);
            assert_eq!(roi.height(), 2);
            // Read pixel through roi
            assert_eq!(roi.pixel_at(0, 0), 5); // (1 + 1*4)
            // Modify through roi
            *roi.pixel_at_mut(0, 0) = 255;
        }
        // Verify change propagated to original
        assert_eq!(img.pixel_at(1, 1), 255);
    }

    #[test]
    fn test_image_array_roi_mut_out_of_bounds() {
        let mut img: ImageArray<u8, 4, 4> = ImageArray::generate(|x, y| (x + y * 4) as u8);
        let roi = img.roi_mut(Rectangle::new((3, 3), (2, 2)));
        assert!(roi.is_none());
    }

    // -----------------------------------------------------------------------
    // ImageRefMut ImageView::get (read through mutable ROI)
    // -----------------------------------------------------------------------

    #[test]
    fn test_image_ref_mut_get() {
        let mut img: Image<u8> = Image::generate(4, 4, |x, y| (x + y * 4) as u8);
        let roi = img.roi_mut(Rectangle::new((1, 1), (2, 2))).unwrap();
        // Test the get method on ImageRefMut (ImageView impl)
        assert_eq!(roi.get(0, 0), Some(5)); // (1 + 1*4)
        assert_eq!(roi.get(1, 0), Some(6)); // (2 + 1*4)
        assert_eq!(roi.get(0, 1), Some(9)); // (1 + 2*4)
        // Out of bounds
        assert_eq!(roi.get(2, 0), None);
        assert_eq!(roi.get(0, 2), None);
    }

    // -----------------------------------------------------------------------
    // ImageRefMut — pointer-based internals validation
    // -----------------------------------------------------------------------

    #[test]
    fn test_image_ref_mut_edge_bounds() {
        // ROI at the very edge of the image
        let mut img: Image<u8> = Image::generate(4, 4, |x, y| (x + y * 4) as u8);
        let roi = img.roi_mut(Rectangle::new((2, 2), (2, 2))).unwrap();
        assert_eq!(roi.size(), Size::new(2, 2));
        assert_eq!(roi.get(0, 0), Some(10)); // (2 + 2*4)
        assert_eq!(roi.get(1, 1), Some(15)); // (3 + 3*4)
        assert_eq!(roi.get(2, 0), None);
        assert_eq!(roi.get(0, 2), None);
    }

    #[test]
    fn test_image_ref_mut_mutation_reflected_in_image() {
        let mut img: Image<u8> = Image::generate(4, 4, |x, y| (x + y * 4) as u8);
        {
            let mut roi = img.roi_mut(Rectangle::new((1, 1), (2, 2))).unwrap();
            *roi.pixel_at_mut(0, 0) = 200;
            *roi.pixel_at_mut(1, 1) = 201;
        }
        // Verify changes reflected in original image at global coords
        assert_eq!(img.get(1, 1), Some(200));
        assert_eq!(img.get(2, 2), Some(201));
    }

    #[test]
    fn test_image_ref_mut_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<ImageRefMut<'_, u8>>();
        assert_sync::<ImageRefMut<'_, u8>>();
    }

    #[test]
    fn test_image_ref_mut_full_image_roi() {
        // ROI covering the entire image
        let mut img: Image<u8> = Image::generate(3, 3, |x, y| (x + y * 3) as u8);
        let roi = img.roi_mut(Rectangle::new((0, 0), (3, 3))).unwrap();
        assert_eq!(roi.size(), Size::new(3, 3));
        assert_eq!(roi.get(0, 0), Some(0));
        assert_eq!(roi.get(2, 2), Some(8));
    }

    #[test]
    fn test_image_ref_mut_1x1_roi() {
        let mut img: Image<u8> = Image::generate(4, 4, |x, y| (x + y * 4) as u8);
        let mut roi = img.roi_mut(Rectangle::new((2, 3), (1, 1))).unwrap();
        assert_eq!(roi.size(), Size::new(1, 1));
        assert_eq!(roi.get(0, 0), Some(14)); // (2 + 3*4)
        *roi.pixel_at_mut(0, 0) = 99;
        assert_eq!(roi.get(0, 0), Some(99));
    }

    #[test]
    fn test_image_array_roi_mut_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        // Verify ImageArray's roi_mut also produces Send+Sync ROIs
        let mut img: ImageArray<u8, 4, 4> = ImageArray::generate(|x, y| (x + y * 4) as u8);
        let roi = img.roi_mut(Rectangle::new((0, 0), (2, 2))).unwrap();
        assert_eq!(roi.get(0, 0), Some(0));
        assert_send::<ImageRefMut<'_, u8>>();
        assert_sync::<ImageRefMut<'_, u8>>();
    }

    // -----------------------------------------------------------------------
    // Image roi out-of-bounds (covers the None branch in Image::roi)
    // -----------------------------------------------------------------------

    #[test]
    fn test_image_roi_out_of_bounds() {
        let img: Image<u8> = Image::generate(4, 4, |x, y| (x + y * 4) as u8);
        let roi = img.roi(Rectangle::new((3, 3), (2, 2)));
        assert!(roi.is_none());
    }

    // -----------------------------------------------------------------------
    // &mut Image as ImageView and ImageViewMut
    // -----------------------------------------------------------------------

    #[test]
    fn test_mut_ref_image2d_imageview_methods() {
        let mut img: Image<u8> = Image::generate(3, 3, |x, y| (x + y * 3) as u8);
        let img_ref: &mut Image<u8> = &mut img;

        // Test ImageView methods on &mut Image
        assert_eq!(img_ref.size(), Size::new(3, 3));
        assert_eq!(img_ref.width(), 3);
        assert_eq!(img_ref.height(), 3);
        assert_eq!(img_ref.get(0, 0), Some(0));
        assert_eq!(img_ref.get(1, 0), Some(1));
        assert_eq!(img_ref.get(3, 0), None);
        assert_eq!(img_ref.pixel_at(2, 2), 8);
    }

    #[test]
    fn test_mut_ref_image2d_imageviewmut_methods() {
        let mut img: Image<u8> = Image::generate(3, 3, |x, y| (x + y * 3) as u8);
        let img_ref: &mut Image<u8> = &mut img;

        // Test ImageViewMut methods on &mut Image
        assert_eq!(img_ref.get_mut(0, 0), Some(&mut 0));
        assert_eq!(img_ref.get_mut(3, 0), None);
        *img_ref.pixel_at_mut(1, 1) = 99;
        assert_eq!(img_ref.pixel_at(1, 1), 99);
    }

    // -----------------------------------------------------------------------
    // PlainImage as_bytes_le / as_bytes_be for Image
    // -----------------------------------------------------------------------

    #[test]
    fn test_image_as_bytes_le() {
        let img: Image<u8> = Image::generate(2, 2, |x, y| (x + y * 2) as u8);
        let bytes_le = img.as_bytes_le();
        // For u8, LE and native should be the same
        assert_eq!(&*bytes_le, &[0, 1, 2, 3]);
    }

    #[test]
    fn test_image_as_bytes_be() {
        let img: Image<u8> = Image::generate(2, 2, |x, y| (x + y * 2) as u8);
        let bytes_be = img.as_bytes_be();
        assert_eq!(&*bytes_be, &[0, 1, 2, 3]);
    }

    #[test]
    fn test_image_as_bytes_le_u16() {
        let img: Image<Rgb16> = Image::generate(1, 1, |_, _| Rgb16::new(0x0102, 0x0304, 0x0506));
        let bytes_le = img.as_bytes_le();
        // Little-endian: each u16 stored as [low, high]
        assert_eq!(&*bytes_le, &[0x02, 0x01, 0x04, 0x03, 0x06, 0x05]);
    }

    #[test]
    fn test_image_as_bytes_be_u16() {
        let img: Image<Rgb16> = Image::generate(1, 1, |_, _| Rgb16::new(0x0102, 0x0304, 0x0506));
        let bytes_be = img.as_bytes_be();
        // Big-endian: each u16 stored as [high, low]
        assert_eq!(&*bytes_be, &[0x01, 0x02, 0x03, 0x04, 0x05, 0x06]);
    }

    // ─── &mut Image<T> ImageView delegation ───────────────────────────

    #[test]
    fn test_mut_ref_image2d_imageview_via_trait_object() {
        // Exercise the `ImageView for &mut Image<T>` impl through a
        // generic function that takes `impl ImageView`, ensuring the
        // delegation layer is monomorphised and hit by llvm-cov.
        fn read_via_view<V: ImageView<Pixel = u8>>(v: V) -> (usize, usize, Option<u8>, u8) {
            let w = v.width();
            let h = v.height();
            let oob = v.get(100, 100);
            let val = v.pixel_at(0, 0);
            (w, h, oob, val)
        }

        let mut img: Image<u8> = Image::generate(3, 2, |x, y| (x + y * 3) as u8);
        let r: &mut Image<u8> = &mut img;
        let (w, h, oob, val) = read_via_view(r);
        assert_eq!(w, 3);
        assert_eq!(h, 2);
        assert_eq!(oob, None);
        assert_eq!(val, 0);
    }

    #[test]
    fn test_mut_ref_image2d_imageviewmut_via_trait_object() {
        fn write_via_view<V: ImageView<Pixel = u8> + ImageViewMut>(mut v: V) {
            // get_mut in bounds
            if let Some(px) = v.get_mut(0, 0) {
                *px = 77;
            }
            // get_mut out of bounds
            assert!(v.get_mut(100, 100).is_none());
            // pixel_at_mut
            *v.pixel_at_mut(1, 0) = 88;
        }

        let mut img: Image<u8> = Image::zero(3, 2);
        write_via_view(&mut img);
        assert_eq!(img.pixel_at(0, 0), 77);
        assert_eq!(img.pixel_at(1, 0), 88);
    }

    #[test]
    fn test_mut_ref_image2d_size() {
        let mut img: Image<u8> = Image::zero(5, 7);
        let r: &mut Image<u8> = &mut img;
        assert_eq!(r.size(), Size::new(5, 7));
    }

    // ─── Image::get returning None (direct call, not via ref) ─────────

    #[test]
    fn test_image_get_none_direct() {
        let img: Image<Mono8> = Image::fill(2, 2, Mono8::new(1));
        assert!(img.get(2, 0).is_none());
        assert!(img.get(0, 2).is_none());
        assert!(img.get(999, 999).is_none());
        assert!(img.get(0, 0).is_some());
    }

    // ─── Image::get_mut (ImageViewMut) ────────────────────────────────

    #[test]
    fn test_image_get_mut_none() {
        let mut img: Image<u8> = Image::zero(3, 3);
        assert!(img.get_mut(3, 0).is_none());
        assert!(img.get_mut(0, 3).is_none());
        assert!(img.get_mut(0, 0).is_some());
    }

    // ─── Image::roi returning None ────────────────────────────────────

    #[test]
    fn test_image_roi_none_x() {
        let img: Image<u8> = Image::zero(4, 4);
        assert!(img.roi(Rectangle::new((3, 0), (2, 1))).is_none());
    }

    #[test]
    fn test_image_roi_none_y() {
        let img: Image<u8> = Image::zero(4, 4);
        assert!(img.roi(Rectangle::new((0, 3), (1, 2))).is_none());
    }

    // ─── Image::roi_mut returning None ────────────────────────────────

    #[test]
    fn test_image_roi_mut_none_x() {
        let mut img: Image<u8> = Image::zero(4, 4);
        assert!(img.roi_mut(Rectangle::new((3, 0), (2, 1))).is_none());
    }

    #[test]
    fn test_image_roi_mut_none_y() {
        let mut img: Image<u8> = Image::zero(4, 4);
        assert!(img.roi_mut(Rectangle::new((0, 3), (1, 2))).is_none());
    }

    // ─── from_raw_bytes edge cases ──────────────────────────────────────

    #[test]
    fn test_from_raw_bytes_size_mismatch() {
        // bytes.len() doesn't match width * height * SIZE
        let bytes = vec![0u8; 7];
        let result: Result<Image<Mono8>, Error> = Image::from_raw_bytes(2, 2, bytes);
        assert!(result.is_err()); // 7 != 2*2*1
    }

    #[test]
    fn test_from_raw_bytes_srgb_mono8() {
        use crate::pixel::SrgbMono8;
        let raw = vec![10u8, 20, 30, 40, 50, 60];
        let img: Image<SrgbMono8> = Image::from_raw_bytes(3, 2, raw).unwrap();
        assert_eq!(img.width(), 3);
        assert_eq!(img.height(), 2);
        assert_eq!(img.pixel_at(0, 0), SrgbMono8::new(10));
        assert_eq!(img.pixel_at(2, 1), SrgbMono8::new(60));
    }

    #[test]
    fn test_from_raw_bytes_rgba8() {
        let raw = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let img: Image<Rgba8> = Image::from_raw_bytes(2, 1, raw).unwrap();
        assert_eq!(img.pixel_at(0, 0), Rgba8::new(1, 2, 3, 4));
        assert_eq!(img.pixel_at(1, 0), Rgba8::new(5, 6, 7, 8));
    }

    #[test]
    fn test_from_raw_bytes_monoa8() {
        let raw = vec![10, 255, 20, 128];
        let img: Image<MonoA8> = Image::from_raw_bytes(2, 1, raw).unwrap();
        assert_eq!(img.pixel_at(0, 0), MonoA8::new(10, 255));
        assert_eq!(img.pixel_at(1, 0), MonoA8::new(20, 128));
    }

    // ───────────────────────────────────────────────────────────────────
    // Coverage: ImageArray PlainImage (as_mut_bytes, as_bytes_le, as_bytes_be)
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_image_array_plain_data_mut_bytes_rgb8() {
        let mut img: ImageArray<Rgb8, 2, 1> =
            ImageArray::new([Rgb8::new(10, 20, 30), Rgb8::new(40, 50, 60)]);
        let bytes = img.as_mut_bytes();
        assert_eq!(bytes, &[10, 20, 30, 40, 50, 60]);
        // Modify through mut bytes
        bytes[0] = 99;
        assert_eq!(img.pixel_at(0, 0).r.0, 99);
    }

    #[test]
    fn test_image_array_plain_data_le_be_mono8() {
        let img: ImageArray<Mono8, 3, 1> =
            ImageArray::new([Mono8::new(1), Mono8::new(2), Mono8::new(3)]);
        let le = img.as_bytes_le();
        let be = img.as_bytes_be();
        // Mono8 is single-byte — LE and BE should be identical
        assert_eq!(&*le, &[1, 2, 3]);
        assert_eq!(&*be, &[1, 2, 3]);
    }

    // ───────────────────────────────────────────────────────────────────
    // Coverage: ImageArray ContiguousImage as_slice / as_mut_slice
    // with PlainPixel types (specific monomorphization)
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_image_array_contiguous_image_mono8() {
        let mut img: ImageArray<Mono8, 2, 2> =
            ImageArray::generate(|x, y| Mono8::new((x + y) as u8));
        let slice = img.as_slice();
        assert_eq!(slice.len(), 4);
        assert_eq!(slice[0], Mono8::new(0));
        let mslice = img.as_mut_slice();
        mslice[0] = Mono8::new(42);
        assert_eq!(img.pixel_at(0, 0), Mono8::new(42));
    }

    // ───────────────────────────────────────────────────────────────────
    // Coverage: ImageArray Index / IndexMut with Coordinate
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_image_array_coordinate_indexing() {
        let mut img: ImageArray<Mono8, 3, 3> =
            ImageArray::generate(|x, y| Mono8::new((x + y * 3) as u8));
        let coord = Coordinate::new(2, 1);
        assert_eq!(img[coord], Mono8::new(5));
        img[coord] = Mono8::new(99);
        assert_eq!(img[coord], Mono8::new(99));
    }

    #[test]
    fn test_image_array_tuple_indexing() {
        let mut img: ImageArray<u8, 2, 3> = ImageArray::generate(|x, y| (x + y * 2) as u8);
        assert_eq!(img[(0, 0)], 0);
        assert_eq!(img[(1, 2)], 5);
        img[(1, 2)] = 77;
        assert_eq!(img[(1, 2)], 77);
    }

    // ───────────────────────────────────────────────────────────────────
    // Coverage: ImageRef — width(), height(), pixel_at(), roi properties
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_image_ref_properties_via_roi() {
        // Exercises ImageRef properties produced via SubView::roi
        let img: Image<u8> = Image::generate(4, 4, |x, y| (x + y * 4) as u8);
        let rect = Rectangle::new((1, 2), (2, 1));
        let roi = img.roi(rect).unwrap();
        assert_eq!(roi.width(), 2);
        assert_eq!(roi.height(), 1);
        assert_eq!(roi.size(), Size::new(2, 1));
    }

    #[test]
    fn test_image_ref_width_height_pixel_at() {
        // Exercises ImageRef via ImageArray ROI
        let img: ImageArray<u8, 5, 5> = ImageArray::generate(|x, y| (x + y * 5) as u8);
        let roi = img.roi(Rectangle::new((1, 1), (3, 2))).unwrap();
        assert_eq!(roi.width(), 3);
        assert_eq!(roi.height(), 2);
        assert_eq!(roi.pixel_at(0, 0), 6); // (1 + 1*5)
        assert_eq!(roi.pixel_at(2, 1), 13); // (3 + 2*5)
    }

    #[test]
    fn test_image_ref_get_out_of_bounds_imagearray() {
        let img: ImageArray<u8, 4, 4> = ImageArray::generate(|x, y| (x + y * 4) as u8);
        let roi = img.roi(Rectangle::new((0, 0), (2, 2))).unwrap();
        assert_eq!(roi.get(0, 0), Some(0));
        assert_eq!(roi.get(1, 1), Some(5));
        assert_eq!(roi.get(2, 0), None);
        assert_eq!(roi.get(0, 2), None);
    }

    // ───────────────────────────────────────────────────────────────────
    // Coverage: ImageRefMut — get_mut default impl path
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_image_ref_mut_get_mut_write_read() {
        let mut img: Image<u8> = Image::fill(3, 3, 0u8);
        let mut roi = img.roi_mut(Rectangle::new((0, 0), (3, 3))).unwrap();
        // Write every pixel through get_mut
        for y in 0..3 {
            for x in 0..3 {
                *roi.get_mut(x, y).unwrap() = (x + y * 3) as u8;
            }
        }
        // Read back through get (ImageView impl)
        for y in 0..3 {
            for x in 0..3 {
                assert_eq!(roi.get(x, y), Some((x + y * 3) as u8));
            }
        }
    }

    #[test]
    fn test_image_ref_mut_width_height() {
        let mut img: Image<u8> = Image::generate(6, 4, |x, y| (x + y) as u8);
        let roi = img.roi_mut(Rectangle::new((2, 1), (3, 2))).unwrap();
        assert_eq!(roi.width(), 3);
        assert_eq!(roi.height(), 2);
        assert_eq!(roi.size(), Size::new(3, 2));
    }

    // ───────────────────────────────────────────────────────────────────
    // Coverage: Image Index / IndexMut with Coordinate
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_image_coordinate_indexing_mono8() {
        let mut img: Image<Mono8> = Image::generate(3, 3, |x, y| Mono8::new((x + y * 3) as u8));
        let coord = Coordinate::new(2, 2);
        assert_eq!(img[coord], Mono8::new(8));
        img[coord] = Mono8::new(42);
        assert_eq!(img[coord], Mono8::new(42));
    }

    // ───────────────────────────────────────────────────────────────────
    // Coverage: &Image<T> ImageView delegation
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_ref_image_imageview_delegation() {
        let img: Image<u8> = Image::generate(3, 2, |x, y| (x + y * 3) as u8);
        let r: &Image<u8> = &img;
        assert_eq!(r.size(), Size::new(3, 2));
        assert_eq!(r.width(), 3);
        assert_eq!(r.height(), 2);
        assert_eq!(r.get(0, 0), Some(0));
        assert_eq!(r.get(2, 1), Some(5));
        assert_eq!(r.get(3, 0), None);
        assert_eq!(r.pixel_at(1, 0), 1);
    }

    // ───────────────────────────────────────────────────────────────────
    // Coverage: Image PlainImage (as_mut_bytes, as_bytes_le, as_bytes_be)
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_image_plain_data_mut_bytes() {
        let mut img: Image<Mono8> = Image::fill(2, 2, Mono8::new(10));
        let bytes = img.as_mut_bytes();
        assert_eq!(bytes, &[10, 10, 10, 10]);
        bytes[0] = 99;
        assert_eq!(img.pixel_at(0, 0), Mono8::new(99));
    }

    // ───────────────────────────────────────────────────────────────────
    // Coverage: Image ContiguousImage as_slice / as_mut_slice
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_image_contiguous_image_mono8() {
        let mut img: Image<Mono8> = Image::fill(2, 2, Mono8::new(5));
        assert_eq!(img.as_slice().len(), 4);
        assert_eq!(img.as_slice()[0], Mono8::new(5));
        img.as_mut_slice()[0] = Mono8::new(42);
        assert_eq!(img.pixel_at(0, 0), Mono8::new(42));
    }

    // ───────────────────────────────────────────────────────────────────
    // Coverage: ImageRef via ImageArray tiling
    // (exercises the TileIter + ImageRef pipeline on ImageArray)
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_imagearray_tiling_exercises_image_ref() {
        let img: ImageArray<Mono8, 4, 4> =
            ImageArray::generate(|x, y| Mono8::new((x + y * 4) as u8));
        let tiles: Vec<_> = img.into_tiles(Size::new(2, 2)).collect();
        assert_eq!(tiles.len(), 4);
        // Each tile is an ImageRef — exercise all ImageView methods
        let t = &tiles[0];
        assert_eq!(t.size(), Size::new(2, 2));
        assert_eq!(t.width(), 2);
        assert_eq!(t.height(), 2);
        assert_eq!(t.get(0, 0), Some(Mono8::new(0)));
        assert_eq!(t.get(1, 1), Some(Mono8::new(5)));
        assert_eq!(t.get(2, 0), None);
        assert_eq!(t.pixel_at(0, 0), Mono8::new(0));
    }

    #[test]
    fn test_imagearray_tiling_partial_exercises_image_ref() {
        let img: ImageArray<Mono8, 5, 3> =
            ImageArray::generate(|x, y| Mono8::new((x + y * 5) as u8));
        let tiles: Vec<_> = img.into_tiles(Size::new(3, 2)).collect();
        assert_eq!(tiles.len(), 4); // 2 cols × 2 rows
        // Partial tile at (3,0) → 2×2
        assert_eq!(tiles[1].size(), Size::new(2, 2));
        assert_eq!(tiles[1].get(0, 0), Some(Mono8::new(3)));
        assert_eq!(tiles[1].pixel_at(1, 0), Mono8::new(4));
        // Partial tile at (0,2) → 3×1
        assert_eq!(tiles[2].size(), Size::new(3, 1));
        assert_eq!(tiles[2].get(0, 0), Some(Mono8::new(10)));
    }

    // ───────────────────────────────────────────────────────────────────
    // Coverage: Image::from_raw_bytes — overflow / edge cases
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_from_raw_bytes_zero_size() {
        let raw: Vec<u8> = vec![];
        let img: Result<Image<Mono8>, Error> = Image::from_raw_bytes(0, 0, raw);
        assert!(img.is_ok());
        let img = img.unwrap();
        assert_eq!(img.width(), 0);
        assert_eq!(img.height(), 0);
    }

    // ─── ImageRef tests ───────────────────────────────────────────────

    #[test]
    fn test_image_ref_new_success() {
        let data = [1u8, 2, 3, 4, 5, 6];
        let view = ImageRef::new(3, 2, &data).unwrap();
        assert_eq!(view.width(), 3);
        assert_eq!(view.height(), 2);
        assert_eq!(view.size(), Size::new(3, 2));
        assert!(view.is_contiguous());
    }

    #[test]
    fn test_image_ref_new_size_mismatch() {
        let data = [1u8, 2, 3, 4, 5];
        assert!(ImageRef::new(3, 2, &data).is_err());
    }

    #[test]
    fn test_image_ref_pixel_at() {
        let data: Vec<u8> = (0..12).collect();
        let view = ImageRef::new(4, 3, &data).unwrap();
        assert_eq!(view.pixel_at(0, 0), 0);
        assert_eq!(view.pixel_at(3, 0), 3);
        assert_eq!(view.pixel_at(0, 1), 4);
        assert_eq!(view.pixel_at(3, 2), 11);
    }

    #[test]
    fn test_image_ref_get_out_of_bounds() {
        let data = [1u8, 2, 3, 4];
        let view = ImageRef::new(2, 2, &data).unwrap();
        assert_eq!(view.get(0, 0), Some(1));
        assert_eq!(view.get(1, 1), Some(4));
        assert_eq!(view.get(2, 0), None);
        assert_eq!(view.get(0, 2), None);
    }

    #[test]
    fn test_image_ref_is_contiguous() {
        let data = [1u8, 2, 3, 4, 5, 6];
        let view = ImageRef::new(3, 2, &data).unwrap();
        assert!(view.is_contiguous());

        // Strided view is not contiguous
        let strided = ImageRef::strided(Size::new(2, 2), 3, 0, &data).unwrap();
        assert!(!strided.is_contiguous());
    }

    #[test]
    fn test_image_ref_strided_pixel_at() {
        // 4x3 buffer, viewing a 2x2 sub-region starting at (1, 1)
        let data: Vec<u8> = (0..12).collect();
        let view = ImageRef::strided(Size::new(2, 2), 4, 5, &data).unwrap();
        assert_eq!(view.pixel_at(0, 0), 5); // data[5]
        assert_eq!(view.pixel_at(1, 0), 6); // data[6]
        assert_eq!(view.pixel_at(0, 1), 9); // data[9]
        assert_eq!(view.pixel_at(1, 1), 10); // data[10]
    }

    #[test]
    fn test_image_ref_roi_contiguous() {
        let data: Vec<u8> = (0..16).collect();
        let view = ImageRef::new(4, 4, &data).unwrap();
        let sub = view.roi(Rectangle::new((1, 1), (2, 2))).unwrap();
        assert_eq!(sub.size(), Size::new(2, 2));
        assert_eq!(sub.pixel_at(0, 0), 5);
        assert_eq!(sub.pixel_at(1, 0), 6);
        assert_eq!(sub.pixel_at(0, 1), 9);
        assert_eq!(sub.pixel_at(1, 1), 10);
    }

    #[test]
    fn test_image_ref_roi_out_of_bounds() {
        let data: Vec<u8> = (0..16).collect();
        let view = ImageRef::new(4, 4, &data).unwrap();
        assert!(view.roi(Rectangle::new((3, 3), (2, 2))).is_none());
    }

    #[test]
    fn test_image_ref_nested_roi() {
        // Sub-view of a sub-view
        let data: Vec<u8> = (0..36).collect();
        let view = ImageRef::new(6, 6, &data).unwrap();
        let sub1 = view.roi(Rectangle::new((1, 1), (4, 4))).unwrap();
        let sub2 = sub1.roi(Rectangle::new((1, 1), (2, 2))).unwrap();
        // sub2 starts at (2, 2) in the original image
        assert_eq!(sub2.pixel_at(0, 0), 14); // 2 + 2*6
        assert_eq!(sub2.pixel_at(1, 0), 15); // 3 + 2*6
        assert_eq!(sub2.pixel_at(0, 1), 20); // 2 + 3*6
        assert_eq!(sub2.pixel_at(1, 1), 21); // 3 + 3*6
    }

    #[test]
    fn test_image_ref_copy() {
        let data = [1u8, 2, 3, 4];
        let view = ImageRef::new(2, 2, &data).unwrap();
        let copy = view; // ImageRef is Copy
        assert_eq!(view.pixel_at(0, 0), copy.pixel_at(0, 0));
    }

    #[test]
    fn test_image_ref_zero_size() {
        let data: [u8; 0] = [];
        let view = ImageRef::new(0, 0, &data).unwrap();
        assert_eq!(view.width(), 0);
        assert_eq!(view.height(), 0);
        assert!(view.is_contiguous());
    }

    // ─── ImageRefMut tests ────────────────────────────────────────────

    #[test]
    fn test_image_ref_mut_new_success() {
        let mut data = [1u8, 2, 3, 4];
        let view = ImageRefMut::new(2, 2, &mut data).unwrap();
        assert_eq!(view.width(), 2);
        assert_eq!(view.height(), 2);
        assert!(view.is_contiguous());
    }

    #[test]
    fn test_image_ref_mut_new_size_mismatch() {
        let mut data = [1u8, 2, 3];
        assert!(ImageRefMut::new(2, 2, &mut data).is_err());
    }

    #[test]
    fn test_image_ref_mut_pixel_at_mut() {
        let mut data = [0u8; 4];
        let mut view = ImageRefMut::new(2, 2, &mut data).unwrap();
        *view.pixel_at_mut(1, 1) = 42;
        assert_eq!(view.pixel_at(1, 1), 42);
    }

    #[test]
    fn test_image_ref_mut_mutation_visible_in_parent() {
        let mut img: Image<u8> = Image::generate(4, 4, |x, y| (x + y * 4) as u8);
        {
            let mut roi = img.roi_mut(Rectangle::new((1, 1), (2, 2))).unwrap();
            *roi.pixel_at_mut(0, 0) = 200;
        }
        assert_eq!(img.get(1, 1), Some(200));
    }

    #[test]
    fn test_image_ref_mut_send_sync_standalone() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<ImageRefMut<'_, u8>>();
        assert_sync::<ImageRefMut<'_, u8>>();
    }

    #[test]
    fn test_image_ref_mut_get_mut_out_of_bounds() {
        let mut data = [0u8; 4];
        let mut view = ImageRefMut::new(2, 2, &mut data).unwrap();
        assert!(view.get_mut(2, 0).is_none());
        assert!(view.get_mut(0, 2).is_none());
        assert!(view.get_mut(0, 0).is_some());
    }

    #[test]
    fn test_image_ref_mut_roi_mut() {
        let mut data: Vec<u8> = (0..16).collect();
        let mut view = ImageRefMut::new(4, 4, &mut data).unwrap();
        let mut sub = view.roi_mut(Rectangle::new((1, 1), (2, 2))).unwrap();
        assert_eq!(sub.pixel_at(0, 0), 5);
        *sub.pixel_at_mut(0, 0) = 99;
        assert_eq!(sub.pixel_at(0, 0), 99);
    }

    #[test]
    fn test_image_ref_mut_roi_read_only() {
        let mut data: Vec<u8> = (0..16).collect();
        let view = ImageRefMut::new(4, 4, &mut data).unwrap();
        // SubView::roi on an ImageRefMut yields an ImageRef (read-only)
        let sub = view.roi(Rectangle::new((1, 1), (2, 2))).unwrap();
        assert_eq!(sub.pixel_at(0, 0), 5);
    }

    #[test]
    fn test_from_bytes_copy_mono8() {
        let bytes = vec![10u8, 20, 30, 40, 50, 60];
        let img: Image<Mono8> = Image::from_bytes_copy(3, 2, &bytes).unwrap();
        assert_eq!(img.width(), 3);
        assert_eq!(img.height(), 2);
        assert_eq!(img.pixel_at(0, 0), Mono8::new(10));
        assert_eq!(img.pixel_at(2, 0), Mono8::new(30));
        assert_eq!(img.pixel_at(0, 1), Mono8::new(40));
        assert_eq!(img.pixel_at(2, 1), Mono8::new(60));
    }

    #[test]
    fn test_from_bytes_copy_mono16() {
        // Mono16 has ALIGN == 2. Two pixels: native-endian u16 values.
        let bytes: Vec<u8> = vec![0x00, 0x01, 0x02, 0x03];
        let img: Image<Mono16> = Image::from_bytes_copy(2, 1, &bytes).unwrap();
        assert_eq!(img.width(), 2);
        assert_eq!(img.height(), 1);
        // Verify pixel values match native-endian reinterpretation
        let expected_0 = u16::from_ne_bytes([0x00, 0x01]);
        let expected_1 = u16::from_ne_bytes([0x02, 0x03]);
        assert_eq!(img.pixel_at(0, 0).value(), expected_0);
        assert_eq!(img.pixel_at(1, 0).value(), expected_1);
    }

    #[test]
    fn test_from_bytes_copy_rgb8() {
        // Rgb8 is 3 bytes per pixel, ALIGN == 1
        let bytes = vec![10, 20, 30, 40, 50, 60];
        let img: Image<Rgb8> = Image::from_bytes_copy(2, 1, &bytes).unwrap();
        assert_eq!(img.width(), 2);
        assert_eq!(img.height(), 1);
        assert_eq!(img.pixel_at(0, 0), Rgb8::new(10, 20, 30));
        assert_eq!(img.pixel_at(1, 0), Rgb8::new(40, 50, 60));
    }

    #[test]
    fn test_from_bytes_copy_wrong_size() {
        // Mono8: 1 byte per pixel, 2x2 = 4 bytes expected
        let bytes = vec![1, 2, 3]; // only 3 bytes
        let result = Image::<Mono8>::from_bytes_copy(2, 2, &bytes);
        assert!(result.is_err());
        match result {
            Err(Error::LengthMismatch { expected, actual }) => {
                assert_eq!(expected, 4);
                assert_eq!(actual, 3);
            }
            Err(other) => panic!("expected LengthMismatch, got {:?}", other),
            Ok(_) => panic!("expected Err, got Ok"),
        }
    }

    #[test]
    fn test_from_bytes_copy_zero_size() {
        let bytes: Vec<u8> = vec![];
        let img: Image<Mono8> = Image::from_bytes_copy(0, 0, &bytes).unwrap();
        assert_eq!(img.width(), 0);
        assert_eq!(img.height(), 0);
    }

    #[test]
    fn test_from_bytes_copy_roundtrip() {
        // Create an image, get its bytes, reconstruct with from_bytes_copy
        let original = Image::from_vec(
            2,
            2,
            vec![
                Mono16::new(100),
                Mono16::new(200),
                Mono16::new(300),
                Mono16::new(400),
            ],
        )
        .unwrap();
        let bytes_data: Vec<u8> = original.as_bytes().to_vec();
        let reconstructed: Image<Mono16> = Image::from_bytes_copy(2, 2, &bytes_data).unwrap();
        assert_eq!(reconstructed.pixel_at(0, 0).value(), 100);
        assert_eq!(reconstructed.pixel_at(1, 0).value(), 200);
        assert_eq!(reconstructed.pixel_at(0, 1).value(), 300);
        assert_eq!(reconstructed.pixel_at(1, 1).value(), 400);
    }

    // ───────────────────────────────────────────────────────────────────
    // RasterImage / RasterImageMut tests
    // ───────────────────────────────────────────────────────────────────

    // ── Image<T> ────────────────────────────────────────────────────

    #[test]
    fn test_raster_image_row_basic() {
        let img = Image::generate(4, 3, |x, y| (y * 4 + x) as u8);
        assert_eq!(img.row(0), &[0, 1, 2, 3]);
        assert_eq!(img.row(1), &[4, 5, 6, 7]);
        assert_eq!(img.row(2), &[8, 9, 10, 11]);
    }

    #[test]
    fn test_raster_image_row_1x1() {
        let img = Image::fill(1, 1, 42u8);
        assert_eq!(img.row(0), &[42]);
    }

    #[test]
    fn test_raster_image_row_wide() {
        let img = Image::generate(10, 1, |x, _| x as u8);
        assert_eq!(img.row(0), &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_raster_image_row_tall() {
        let img = Image::generate(1, 5, |_, y| y as u8);
        for y in 0..5 {
            assert_eq!(img.row(y), &[y as u8]);
        }
    }

    #[test]
    #[should_panic]
    fn test_raster_image_row_out_of_bounds() {
        let img = Image::fill(3, 3, 0u8);
        let _ = img.row(3);
    }

    #[test]
    fn test_raster_image_mut_row_mut_basic() {
        let mut img = Image::generate(4, 3, |x, y| (y * 4 + x) as u8);
        img.row_mut(1).fill(99);
        assert_eq!(img.row(0), &[0, 1, 2, 3]);
        assert_eq!(img.row(1), &[99, 99, 99, 99]);
        assert_eq!(img.row(2), &[8, 9, 10, 11]);
    }

    #[test]
    fn test_raster_image_mut_row_mut_individual() {
        let mut img = Image::fill(3, 2, 0u8);
        img.row_mut(0)[1] = 42;
        assert_eq!(img.row(0), &[0, 42, 0]);
        assert_eq!(img.row(1), &[0, 0, 0]);
    }

    #[test]
    #[should_panic]
    fn test_raster_image_mut_row_mut_out_of_bounds() {
        let mut img = Image::fill(3, 3, 0u8);
        let _ = img.row_mut(3);
    }

    #[test]
    fn test_raster_image_row_len_equals_width() {
        let img = Image::fill(7, 5, 0u8);
        for y in 0..img.height() {
            assert_eq!(img.row(y).len(), 7);
        }
    }

    #[test]
    fn test_raster_image_row_consistent_with_pixel_at() {
        let img = Image::generate(5, 4, |x, y| (y * 10 + x) as u8);
        for y in 0..img.height() {
            let row = img.row(y);
            for x in 0..img.width() {
                assert_eq!(row[x], img.pixel_at(x, y));
            }
        }
    }

    #[test]
    fn test_raster_image_mut_row_mut_consistent_with_pixel_at_mut() {
        let mut img = Image::generate(4, 3, |x, y| (y * 4 + x) as u8);
        for y in 0..img.height() {
            for x in 0..img.width() {
                img.row_mut(y)[x] += 100;
            }
        }
        for y in 0..img.height() {
            for x in 0..img.width() {
                assert_eq!(img.pixel_at(x, y), (y * 4 + x) as u8 + 100);
            }
        }
    }

    // ── ImageArray<T, W, H> ────────────────────────────────────────

    #[test]
    fn test_raster_image_array_row_basic() {
        let img: ImageArray<u8, 3, 2> = ImageArray::generate(|x, y| (y * 3 + x) as u8);
        assert_eq!(img.row(0), &[0, 1, 2]);
        assert_eq!(img.row(1), &[3, 4, 5]);
    }

    #[test]
    fn test_raster_image_array_row_1x1() {
        let img: ImageArray<u8, 1, 1> = ImageArray::new([7]);
        assert_eq!(img.row(0), &[7]);
    }

    #[test]
    #[should_panic]
    fn test_raster_image_array_row_out_of_bounds() {
        let img: ImageArray<u8, 3, 2> = ImageArray::generate(|_, _| 0);
        let _ = img.row(2);
    }

    #[test]
    fn test_raster_image_array_row_mut_basic() {
        let mut img: ImageArray<u8, 3, 2> = ImageArray::generate(|x, y| (y * 3 + x) as u8);
        img.row_mut(0).fill(99);
        assert_eq!(img.row(0), &[99, 99, 99]);
        assert_eq!(img.row(1), &[3, 4, 5]);
    }

    #[test]
    fn test_raster_image_array_row_consistent_with_pixel_at() {
        let img: ImageArray<u8, 4, 4> = ImageArray::generate(|x, y| (y * 4 + x) as u8);
        for y in 0..img.height() {
            let row = img.row(y);
            for x in 0..img.width() {
                assert_eq!(row[x], img.pixel_at(x, y));
            }
        }
    }

    // ── ImageRef (contiguous) ──────────────────────────────────────

    #[test]
    fn test_raster_image_ref_row_contiguous() {
        let data: Vec<u8> = (0..12).collect();
        let img = ImageRef::new(4, 3, &data).unwrap();
        assert_eq!(img.row(0), &[0, 1, 2, 3]);
        assert_eq!(img.row(1), &[4, 5, 6, 7]);
        assert_eq!(img.row(2), &[8, 9, 10, 11]);
    }

    #[test]
    fn test_raster_image_ref_row_consistent_with_pixel_at() {
        let data: Vec<u8> = (0..20).collect();
        let img = ImageRef::new(5, 4, &data).unwrap();
        for y in 0..img.height() {
            let row = img.row(y);
            for x in 0..img.width() {
                assert_eq!(row[x], img.pixel_at(x, y));
            }
        }
    }

    #[test]
    #[should_panic]
    fn test_raster_image_ref_row_out_of_bounds() {
        let data: Vec<u8> = (0..6).collect();
        let img = ImageRef::new(3, 2, &data).unwrap();
        let _ = img.row(2);
    }

    // ── ImageRef (strided ROI) ─────────────────────────────────────

    #[test]
    fn test_raster_image_ref_strided_row() {
        // Create a 4x4 image and take a 2x2 ROI starting at (1,1)
        let img = Image::generate(4, 4, |x, y| (y * 4 + x) as u8);
        let roi = img.roi(Rectangle::new((1, 1), (2, 2))).unwrap();
        // ROI should see pixels at (1,1),(2,1) and (1,2),(2,2)
        assert_eq!(roi.row(0), &[5, 6]);
        assert_eq!(roi.row(1), &[9, 10]);
    }

    #[test]
    fn test_raster_image_ref_strided_row_consistent_with_pixel_at() {
        let img = Image::generate(6, 6, |x, y| (y * 6 + x) as u8);
        let roi = img.roi(Rectangle::new((2, 1), (3, 4))).unwrap();
        for y in 0..roi.height() {
            let row = roi.row(y);
            for x in 0..roi.width() {
                assert_eq!(row[x], roi.pixel_at(x, y));
            }
        }
    }

    #[test]
    fn test_raster_image_ref_strided_row_len_equals_roi_width() {
        let img = Image::generate(8, 8, |x, y| (y * 8 + x) as u8);
        let roi = img.roi(Rectangle::new((1, 1), (5, 3))).unwrap();
        for y in 0..roi.height() {
            assert_eq!(roi.row(y).len(), 5);
        }
    }

    #[test]
    fn test_raster_image_ref_nested_roi_row() {
        let img = Image::generate(8, 8, |x, y| (y * 8 + x) as u8);
        let roi1 = img.roi(Rectangle::new((1, 1), (6, 6))).unwrap();
        let roi2 = roi1.roi(Rectangle::new((1, 1), (3, 3))).unwrap();
        // roi2 sees absolute pixels starting at (2,2)
        assert_eq!(roi2.row(0), &[18, 19, 20]);
        assert_eq!(roi2.row(1), &[26, 27, 28]);
        assert_eq!(roi2.row(2), &[34, 35, 36]);
    }

    // ── ImageRefMut ────────────────────────────────────────────────

    #[test]
    fn test_raster_image_ref_mut_row() {
        let mut data: Vec<u8> = (0..12).collect();
        let view = ImageRefMut::new(4, 3, &mut data).unwrap();
        assert_eq!(view.row(0), &[0, 1, 2, 3]);
        assert_eq!(view.row(1), &[4, 5, 6, 7]);
        assert_eq!(view.row(2), &[8, 9, 10, 11]);
    }

    #[test]
    fn test_raster_image_ref_mut_row_mut() {
        let mut data: Vec<u8> = (0..12).collect();
        let mut view = ImageRefMut::new(4, 3, &mut data).unwrap();
        view.row_mut(1).fill(42);
        assert_eq!(view.row(0), &[0, 1, 2, 3]);
        assert_eq!(view.row(1), &[42, 42, 42, 42]);
        assert_eq!(view.row(2), &[8, 9, 10, 11]);
    }

    #[test]
    fn test_raster_image_ref_mut_row_mut_reflected_in_data() {
        let mut data: Vec<u8> = (0..6).collect();
        let mut view = ImageRefMut::new(3, 2, &mut data).unwrap();
        view.row_mut(0)[1] = 99;
        drop(view);
        assert_eq!(data, vec![0, 99, 2, 3, 4, 5]);
    }

    #[test]
    fn test_raster_image_ref_mut_strided_row() {
        let mut img = Image::generate(4, 4, |x, y| (y * 4 + x) as u8);
        let roi = img.roi_mut(Rectangle::new((1, 1), (2, 2))).unwrap();
        assert_eq!(roi.row(0), &[5, 6]);
        assert_eq!(roi.row(1), &[9, 10]);
    }

    #[test]
    fn test_raster_image_ref_mut_strided_row_mut() {
        let mut img = Image::generate(4, 4, |x, y| (y * 4 + x) as u8);
        {
            let mut roi = img.roi_mut(Rectangle::new((1, 1), (2, 2))).unwrap();
            roi.row_mut(0).fill(99);
        }
        // Row 0 of roi = row 1, cols 1..3 of original
        assert_eq!(img.pixel_at(0, 1), 4); // untouched
        assert_eq!(img.pixel_at(1, 1), 99); // modified
        assert_eq!(img.pixel_at(2, 1), 99); // modified
        assert_eq!(img.pixel_at(3, 1), 7); // untouched
    }

    #[test]
    fn test_raster_image_ref_mut_row_consistent_with_pixel_at() {
        let mut data: Vec<u8> = (0..20).collect();
        let view = ImageRefMut::new(5, 4, &mut data).unwrap();
        for y in 0..view.height() {
            let row = view.row(y);
            for x in 0..view.width() {
                assert_eq!(row[x], view.pixel_at(x, y));
            }
        }
    }

    #[test]
    #[should_panic]
    fn test_raster_image_ref_mut_row_out_of_bounds() {
        let mut data: Vec<u8> = (0..6).collect();
        let view = ImageRefMut::new(3, 2, &mut data).unwrap();
        let _ = view.row(2);
    }

    #[test]
    #[should_panic]
    fn test_raster_image_ref_mut_row_mut_out_of_bounds() {
        let mut data: Vec<u8> = (0..6).collect();
        let mut view = ImageRefMut::new(3, 2, &mut data).unwrap();
        let _ = view.row_mut(2);
    }

    // ── &Image<T> and &mut Image<T> ───────────────────────────────

    #[test]
    fn test_raster_image_ref_to_image_row() {
        let img = Image::generate(3, 2, |x, y| (y * 3 + x) as u8);
        let r: &Image<u8> = &img;
        assert_eq!(r.row(0), &[0, 1, 2]);
        assert_eq!(r.row(1), &[3, 4, 5]);
    }

    #[test]
    fn test_raster_image_mut_ref_to_image_row_mut() {
        let mut img = Image::generate(3, 2, |x, y| (y * 3 + x) as u8);
        let r: &mut Image<u8> = &mut img;
        r.row_mut(1).fill(42);
        assert_eq!(r.row(0), &[0, 1, 2]);
        assert_eq!(r.row(1), &[42, 42, 42]);
    }

    // ── RasterImage via trait object (dyn) ─────────────────────────

    #[test]
    fn test_raster_image_via_trait() {
        fn read_row(img: &dyn RasterImage<Pixel = u8>, y: usize) -> Vec<u8> {
            img.row(y).to_vec()
        }

        let img = Image::generate(3, 2, |x, y| (y * 3 + x) as u8);
        assert_eq!(read_row(&img, 0), vec![0, 1, 2]);
        assert_eq!(read_row(&img, 1), vec![3, 4, 5]);
    }

    #[test]
    fn test_raster_image_mut_via_trait() {
        fn fill_row(img: &mut dyn RasterImageMut<Pixel = u8>, y: usize, val: u8) {
            img.row_mut(y).fill(val);
        }

        let mut img = Image::generate(3, 2, |x, _| x as u8);
        fill_row(&mut img, 0, 99);
        assert_eq!(img.row(0), &[99, 99, 99]);
        assert_eq!(img.row(1), &[0, 1, 2]);
    }

    // ── RasterImage with Rgb8 pixel type ───────────────────────────

    #[test]
    fn test_raster_image_row_rgb8() {
        let img = Image::generate(2, 2, |x, y| Rgb8::new((x * 10) as u8, (y * 10) as u8, 0));
        let row0 = img.row(0);
        assert_eq!(row0.len(), 2);
        assert_eq!(row0[0], Rgb8::new(0, 0, 0));
        assert_eq!(row0[1], Rgb8::new(10, 0, 0));
        let row1 = img.row(1);
        assert_eq!(row1[0], Rgb8::new(0, 10, 0));
        assert_eq!(row1[1], Rgb8::new(10, 10, 0));
    }

    // ── ContiguousImage still works with RasterImage re-parenting ──

    #[test]
    fn test_contiguous_image_implies_raster_image() {
        fn check_contiguous_has_row<I: ContiguousImage>(img: &I) -> usize {
            // ContiguousImage: RasterImage, so row() must be available
            img.row(0).len()
        }

        let img = Image::fill(5, 3, 0u8);
        assert_eq!(check_contiguous_has_row(&img), 5);

        let arr: ImageArray<u8, 4, 2> = ImageArray::generate(|_, _| 0);
        assert_eq!(check_contiguous_has_row(&arr), 4);
    }

    #[test]
    fn test_contiguous_image_mut_implies_raster_image_mut() {
        fn fill_via_row<I: ContiguousImageMut>(img: &mut I, val: I::Pixel)
        where
            I::Pixel: Copy,
        {
            for y in 0..img.height() {
                img.row_mut(y).fill(val);
            }
        }

        let mut img = Image::fill(3, 2, 0u8);
        fill_via_row(&mut img, 42);
        assert_eq!(img.as_slice(), &[42, 42, 42, 42, 42, 42]);
    }

    // ── Zero-size edge cases ───────────────────────────────────────

    #[test]
    fn test_raster_image_zero_width() {
        let img = Image::<u8>::zero(0, 3);
        // No rows should be accessible (height is 3 but width is 0)
        // row(0) should return an empty slice
        assert_eq!(img.row(0), &[] as &[u8]);
    }

    #[test]
    fn test_raster_image_zero_height() {
        let img = Image::<u8>::zero(3, 0);
        // No rows to access — any row() call would panic
        assert_eq!(img.height(), 0);
    }

    // ── B1: ImageRefMut safe accessors must assert (not debug_assert) bounds ──
    //
    // The accessors below dereference raw pointers; if a bug ever drops
    // the assertion, these tests fail loudly even in release builds.

    #[test]
    #[should_panic(expected = "pixel_at")]
    fn imageref_mut_pixel_at_out_of_bounds_panics() {
        let mut buf = [0u8; 4];
        let v = ImageRefMut::new(2, 2, &mut buf).unwrap();
        let _ = v.pixel_at(5, 5);
    }

    #[test]
    #[should_panic(expected = "pixel_at_mut")]
    fn imageref_mut_pixel_at_mut_out_of_bounds_panics() {
        let mut buf = [0u8; 4];
        let mut v = ImageRefMut::new(2, 2, &mut buf).unwrap();
        let _ = v.pixel_at_mut(5, 5);
    }

    #[test]
    #[should_panic(expected = "row")]
    fn imageref_mut_row_out_of_bounds_panics() {
        let mut buf = [0u8; 4];
        let v = ImageRefMut::new(2, 2, &mut buf).unwrap();
        let _ = v.row(5);
    }

    #[test]
    #[should_panic(expected = "row_mut")]
    fn imageref_mut_row_mut_out_of_bounds_panics() {
        let mut buf = [0u8; 4];
        let mut v = ImageRefMut::new(2, 2, &mut buf).unwrap();
        let _ = v.row_mut(5);
    }

    #[test]
    fn imageref_mut_get_out_of_bounds_returns_none() {
        let mut buf = [1u8, 2, 3, 4];
        let v = ImageRefMut::new(2, 2, &mut buf).unwrap();
        assert!(v.get(5, 5).is_none());
        assert_eq!(v.get(1, 1), Some(4));
    }

    // ── B2: storage constructors reject overflowing dimensions ──

    #[test]
    fn image_from_vec_rejects_overflowing_dimensions() {
        let data = vec![0u8; 4];
        let res = Image::from_vec(usize::MAX, 2, data);
        assert!(matches!(res, Err(Error::LengthMismatch { .. })));
    }

    #[test]
    fn imageref_new_rejects_overflowing_dimensions() {
        let data = [0u8; 4];
        let res = ImageRef::new(usize::MAX, 2, &data);
        assert!(matches!(res, Err(Error::LengthMismatch { .. })));
    }

    #[test]
    fn imageref_mut_new_rejects_overflowing_dimensions() {
        let mut data = [0u8; 4];
        let res = ImageRefMut::new(usize::MAX, 2, &mut data);
        assert!(matches!(res, Err(Error::LengthMismatch { .. })));
    }

    #[test]
    fn image_roi_rejects_overflowing_rectangle() {
        let img = Image::<u8>::zero(4, 4);
        let rect = Rectangle::new((usize::MAX - 1, 0), (10, 1));
        assert!(img.roi(rect).is_none());
    }

    #[test]
    fn image_roi_mut_rejects_overflowing_rectangle() {
        let mut img = Image::<u8>::zero(4, 4);
        let rect = Rectangle::new((0, usize::MAX - 1), (1, 10));
        assert!(img.roi_mut(rect).is_none());
    }

    // ─────────────────────────────────────────────────────────────────────
    // P0-3 red-phase tests for strict-accessor index overflow.
    //
    // These exercise the Tier-3 strict accessors (`pixel_at`, `row`, and
    // the `Index` impl) with coordinates whose `y * width + x` wraps in
    // release builds. Before the fix, the wrapped index could land back
    // inside the backing allocation and silently return a *wrong* pixel
    // instead of panicking. After the fix, every strict accessor must
    // panic (Tier-3 programmer-bug semantics) on out-of-bounds input,
    // including the overflow case.
    //
    // We use coordinates that, for a 4×4 = 16-pixel image, wrap to an
    // in-range index: `y = 1 << (USIZE_BITS - 2)`, so `y * 4` wraps to 0
    // on 64-bit usize. Without checked arithmetic, `pixel_at(0, y)` then
    // returns `data[0]` instead of panicking.
    // ─────────────────────────────────────────────────────────────────────

    const OVERFLOW_Y: usize = 1usize << (usize::BITS as usize - 2);

    #[test]
    #[should_panic]
    fn pixel_at_overflow_panics_instead_of_returning_wrong_pixel() {
        let img: Image<u8> = Image::fill(4, 4, 7);
        // y * 4 wraps to 0 on 64-bit; index would equal 0 unchecked.
        let _ = img.pixel_at(0, OVERFLOW_Y);
    }

    #[test]
    #[should_panic]
    fn pixel_at_mut_overflow_panics_instead_of_returning_wrong_pixel() {
        let mut img: Image<u8> = Image::fill(4, 4, 7);
        let _ = img.pixel_at_mut(0, OVERFLOW_Y);
    }

    #[test]
    #[should_panic]
    fn row_overflow_panics_instead_of_returning_wrong_row() {
        let img: Image<u8> = Image::fill(4, 4, 7);
        // y * 4 wraps to 0; row(y) would alias row(0) unchecked.
        let _ = img.row(OVERFLOW_Y);
    }

    #[test]
    #[should_panic]
    fn row_mut_overflow_panics_instead_of_returning_wrong_row() {
        let mut img: Image<u8> = Image::fill(4, 4, 7);
        let _ = img.row_mut(OVERFLOW_Y);
    }

    #[test]
    #[should_panic]
    fn index_overflow_panics_instead_of_returning_wrong_pixel() {
        let img: Image<u8> = Image::fill(4, 4, 7);
        let _ = img[(0usize, OVERFLOW_Y)];
    }

    // ── P1-4: core API polish ─────────────────────────────────────────────────────────────────────────────

    #[test]
    fn debug_image_is_compact_and_contains_dimensions() {
        // 1080p × Srgb8 would be ~6 MB of formatted text with derive(Debug);
        // ours is bounded to a small constant regardless of pixel count.
        let img: Image<Mono8> = Image::fill(1920, 1080, Mono8::new(0));
        let s = format!("{:?}", img);
        assert!(s.contains("1920"), "missing width: {s}");
        assert!(s.contains("1080"), "missing height: {s}");
        assert!(s.contains("Mono8"), "missing pixel type: {s}");
        // Bound the size to confirm we did not dump pixel data.
        // A 6 MB string would obviously exceed this; the actual output
        // is tens of characters.
        assert!(
            s.len() < 256,
            "Debug output unexpectedly large ({} bytes)",
            s.len()
        );
    }

    #[test]
    fn debug_image_works_without_t_debug_bound() {
        // A type that intentionally is NOT Debug.
        #[derive(Copy, Clone)]
        struct OpaquePixel(#[allow(dead_code)] u32);
        let img: Image<OpaquePixel> = Image::fill(2, 3, OpaquePixel(0));
        // The fact that this compiles is the test.
        let s = format!("{:?}", img);
        assert!(s.contains("OpaquePixel"), "got: {s}");
    }

    #[test]
    fn image_partial_eq_reflexive_and_symmetric() {
        let a: Image<Mono8> = Image::fill(4, 3, Mono8::new(5));
        let b: Image<Mono8> = Image::fill(4, 3, Mono8::new(5));
        let c: Image<Mono8> = Image::fill(4, 3, Mono8::new(6));
        let d: Image<Mono8> = Image::fill(5, 3, Mono8::new(5));
        assert_eq!(a, a); // reflexive
        assert_eq!(a, b);
        assert_eq!(b, a); // symmetric
        assert_ne!(a, c); // pixel-value differs
        assert_ne!(a, d); // dimension differs
    }

    #[test]
    fn fill_works_with_clone_only_type() {
        // `String` is `Clone` but not `ZeroablePixel` (nor `Copy`).
        // Prior to relaxing `fill`'s bound this would not compile.
        // We verify construction succeeds and every slot got the fill
        // value cloned. `ImageView` is not available here (it needs
        // `T: Copy`); that is intentional and not in scope of this test.
        let img: Image<String> = Image::fill(2, 2, "hello".to_string());
        assert_eq!(img.data.len(), 4);
        assert_eq!(&img.data[0], "hello");
        assert_eq!(&img.data[3], "hello");
    }
}

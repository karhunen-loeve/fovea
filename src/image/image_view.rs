use crate::geometry::PixelSpace;
use crate::{Size, internal};

// ───────────────────────────────────────────────────────────────────
// ImageView / ImageViewMut — random pixel access (universal)
// ───────────────────────────────────────────────────────────────────

/// The `ImageView` trait is a non-mutable representation of a part of
/// an image. This trait should be like a 2D slice for an image.
pub trait ImageView {
    /// The pixel type stored in this image view.
    type Pixel: Copy;

    /// The space the image's grid lies in:
    /// [`Pixels`](crate::Pixels) for an image as the camera delivered it,
    /// [`IdealPixels`](crate::IdealPixels) for one remapped through a lens
    /// model, or a [`PixelSpace`] of your own.
    ///
    /// Every operation passes the space through: a filter, a resize or a
    /// region of interest returns an image in the space of its input, a
    /// position found on an image is a [`Point`](crate::Point) in that
    /// space, and an operation that combines two images requires them to
    /// share one. Images built from numbers are in `Pixels`; a named
    /// relabelling such as [`Image::into_space`](crate::image::Image::into_space)
    /// states another.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::image::{Image, ImageView};
    /// use fovea::pixel::Mono8;
    /// use fovea::{IdealPixels, Pixels};
    ///
    /// fn space_name<I: ImageView>(_: &I) -> &'static str {
    ///     core::any::type_name::<I::Space>().rsplit("::").next().unwrap()
    /// }
    ///
    /// let raw = Image::fill(4, 4, Mono8::new(0));
    /// let ideal = raw.clone().into_space::<IdealPixels>();
    /// assert_eq!(space_name(&raw), "Pixels");
    /// assert_eq!(space_name(&ideal), "IdealPixels");
    /// let _: &Image<Mono8, Pixels> = &raw;
    /// ```
    ///
    /// An image whose space is a unit of the world is no image view, so no
    /// operation accepts it:
    ///
    /// ```compile_fail
    /// use fovea::Millimeter;
    /// use fovea::image::{Image, ImageView};
    /// use fovea::pixel::Mono8;
    ///
    /// fn width(image: &impl ImageView) -> usize {
    ///     image.width()
    /// }
    ///
    /// fn in_millimetres(image: &Image<Mono8, Millimeter>) -> usize {
    ///     width(image)
    /// }
    /// ```
    type Space: PixelSpace;

    /// Returns the dimensions of the image as a [`Size`].
    fn size(&self) -> Size;
    /// Returns the image width in pixels.
    #[inline(always)]
    fn width(&self) -> usize {
        self.size().width
    }
    /// Returns the image height in pixels.
    #[inline(always)]
    fn height(&self) -> usize {
        self.size().height
    }

    /// Returns the pixel at the given `(x, y)` coordinate by value.
    ///
    /// All pixel types implement `Copy` (enforced by `PlainPixel`), so
    /// returning by value is zero-cost (one register move for types up
    /// to 16 bytes) and eliminates pointer indirection.  This also
    /// enables future function-based images that compute pixels on
    /// demand without backing memory.
    ///
    /// # Panics
    ///
    /// Panics if `(x, y)` is out of bounds (via slice indexing).
    fn pixel_at(&self, x: usize, y: usize) -> Self::Pixel;

    /// Returns the pixel at `(x, y)`, or `None` if the coordinate is out of bounds.
    #[inline(always)]
    fn get(&self, x: usize, y: usize) -> Option<Self::Pixel> {
        if internal::in_bounds(&self.size(), x, y) {
            Some(self.pixel_at(x, y))
        } else {
            None
        }
    }
}

/// The `ImageViewMut` trait is a mutable representation of a part of
/// an image. This trait should be like a 2D mut slice for an image
pub trait ImageViewMut: ImageView {
    /// Returns a mutable reference to the pixel at the given `(x, y)` coordinate.
    ///
    /// # Panics
    ///
    /// Panics if `(x, y)` is out of bounds (via slice indexing).
    fn pixel_at_mut(&mut self, x: usize, y: usize) -> &mut Self::Pixel;

    /// Returns a mutable reference to the pixel at `(x, y)`, or `None` if out of bounds.
    #[inline(always)]
    fn get_mut(&mut self, x: usize, y: usize) -> Option<&mut Self::Pixel> {
        if internal::in_bounds(&self.size(), x, y) {
            Some(self.pixel_at_mut(x, y))
        } else {
            None
        }
    }
}

// ───────────────────────────────────────────────────────────────────
// RasterImage / RasterImageMut — row-level slice access (memory-backed)
// ───────────────────────────────────────────────────────────────────

/// A memory-backed image where each row is a contiguous slice of pixels.
///
/// `RasterImage` sits between [`ImageView`] (random pixel access) and
/// [`ContiguousImage`](crate::image::ContiguousImage) (entire buffer is
/// flat). Every row is a dense `&[Self::Pixel]` slice of exactly
/// `width()` elements, but rows may be separated by a stride (e.g. for
/// strided ROIs produced by [`SubView::roi`](crate::image::SubView::roi)).
///
/// This trait is the foundation for SIMD-friendly iteration: transforms
/// can process one row slice at a time instead of calling `pixel_at` per
/// pixel, enabling both auto-vectorization and explicit SIMD kernels.
///
/// # Who implements this
///
/// | Type                     | `RasterImage` | `ContiguousImage` |
/// |--------------------------|---------------|-------------------|
/// | `Image<T>`               | ✅            | ✅                |
/// | `ImageArray<T, W, H>`    | ✅            | ✅                |
/// | `ImageRef` (any)         | ✅            | ❌                |
/// | `ImageRefMut` (any)      | ✅            | —                 |
/// | Function image (future)  | ❌            | ❌                |
///
/// # Example
/// ```
/// # use fovea::image::{Image, ImageView, RasterImage};
/// let img = Image::generate(4, 3, |x, y| (y * 4 + x) as u8);
/// let row1 = img.row(1);
/// assert_eq!(row1, &[4, 5, 6, 7]);
/// ```
pub trait RasterImage: ImageView {
    /// Returns an immutable slice of pixels for row `y`.
    ///
    /// The returned slice has exactly `self.width()` elements and
    /// represents a dense, contiguous run of pixels in memory.
    ///
    /// # Panics
    ///
    /// Panics if `y >= self.height()` (Tier 3 — programmer bug).
    fn row(&self, y: usize) -> &[Self::Pixel];
}

/// A memory-backed image with mutable row-level slice access.
///
/// Extends [`RasterImage`] with `row_mut`, enabling in-place
/// modification of an entire row at a time.
///
/// # Example
/// ```
/// # use fovea::image::{Image, RasterImage, RasterImageMut};
/// let mut img = Image::generate(3, 2, |x, y| (y * 3 + x) as u8);
/// img.row_mut(0).fill(42);
/// assert_eq!(img.row(0), &[42, 42, 42]);
/// assert_eq!(img.row(1), &[3, 4, 5]);
/// ```
pub trait RasterImageMut: RasterImage + ImageViewMut {
    /// Returns a mutable slice of pixels for row `y`.
    ///
    /// The returned slice has exactly `self.width()` elements.
    ///
    /// # Panics
    ///
    /// Panics if `y >= self.height()`.
    fn row_mut(&mut self, y: usize) -> &mut [Self::Pixel];
}

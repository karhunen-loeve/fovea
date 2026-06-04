//! Image storage, views, ROIs, tiles, and neighborhood shapes.
//!
//! Start with [`Image`] when you own pixels, [`ImageRef`] / [`ImageRefMut`]
//! when you borrow existing storage, and [`ImageView`] when you are writing an
//! algorithm that only needs random access.
//!
//! ## Which access trait?
//!
//! | Trait | Guarantee | Reach for it when |
//! |---|---|---|
//! | [`ImageView`] | `pixel_at(x, y)` by value | You only need random access. |
//! | [`ImageViewMut`] | mutable random access | You need to edit individual pixels. |
//! | [`RasterImage`] | dense row slices | You scan rows or want cache-friendly loops. |
//! | [`RasterImageMut`] | mutable row slices | You mutate rows in place. |
//! | [`ContiguousImage`] | one dense pixel slice | You need the fastest whole-buffer path. |
//! | [`PlainImage`] | byte access to `PlainPixel` storage | You write camera, file, FFI, or GPU boundaries. |
//!
//! ## Views do not allocate
//!
//! [`SubView::roi`] returns a borrowed region of interest. [`SubView::tiles`]
//! splits an image into borrowed immutable tiles. [`IntoTilesMut`] yields
//! disjoint mutable tiles for safe chunked in-place processing.
//!
//! Do not use this module for pixel semantics. If the question is "is this
//! gamma-encoded?" or "can this be interpolated?", look in [`crate::pixel`].

/// Border policies for out-of-bounds neighborhood access.
pub mod border;
mod image_view;
mod neighborhood;
mod separable;
pub(crate) mod tiles;
mod zip;

mod planar;
pub(crate) mod sequential;

pub use image_view::{ImageView, ImageViewMut, RasterImage, RasterImageMut};
pub use neighborhood::{
    Kernel, Kernel3x3, Kernel3x3i, Kernel5x5, Kernel5x5i, Kernel7x7, Mask, Mask3x3, Mask5x5,
    Neighborhood, PositionsIter,
};
pub use planar::ImagePlanes;
pub use separable::SeparableKernel;
pub use sequential::{
    ContiguousImage, ContiguousImageMut, Image, ImageArray, ImageRef, ImageRefMut, PlainImage,
    PlainImageMut,
};
pub use tiles::{
    EnumeratePositions, IntoTilesMut, SlidingWindow, SlidingWindowIter, SubView, SubViewMut,
    TileIter, TileIterMut,
};
pub use zip::{ZipPixelsIter, zip_pixels};

// ─── Binary image vocabulary ────────────────────────────────────
//
// Binary images are images whose pixel type is `bool`. The `bool` type
// already rides the `T: Copy` pathway through `Image<T>`, `ImageView`,
// `ImageRef`, `SubView`, tiles, sliding windows, zip, and the parallel
// iteration machinery — all of which work today with no changes. `bool` is
// also the pixel type that `map_neighborhood*` already consumes as its
// topology mask parameter (`MI: ImageView<Pixel = bool>`), so morphology
// and neighborhood operations natively accept binary images with no
// bridging conversion.
//
// These aliases give that concept a first-class name. They are zero-cost
// documentation: every `BinaryImage` is structurally identical to the
// `Image<bool>` it aliases.
//
// Naming: `BinaryImage`, not `Mask`. The name `Mask` is already taken in
// this codebase for compile-time-sized structuring elements
// (`Mask<KW, KH> = Neighborhood<bool, KW, KH>`). Reusing `Mask` for whole
// images would be a three-way collision.

/// An image whose pixels are binary (`bool`).
///
/// Produced by strategies such as [`BinaryMask`](crate::transform::BinaryMask);
/// consumed directly by morphology operations
/// ([`erode`](crate::transform::erode), [`dilate`](crate::transform::dilate)),
/// connected-components analysis, and anything else that accepts
/// `ImageView<Pixel = bool>`.
///
/// This is a type alias for `Image<bool>`; every method, trait impl, and
/// storage guarantee of `Image<bool>` applies transparently.
///
/// # Example
/// ```
/// # use fovea::image::{BinaryImage, Image, ImageView};
/// # use fovea::pixel::Mono8;
/// # use fovea::transform::{BinaryMask, convert_image};
/// let img: Image<Mono8> = Image::fill(4, 4, Mono8::new(200));
/// let mask: BinaryImage = convert_image(&img, BinaryMask { thresh: Mono8::new(128) });
/// assert!(mask.pixel_at(0, 0));
/// ```
pub type BinaryImage = Image<bool>;

/// An immutable reference to a binary image. Alias for [`ImageRef<bool>`].
pub type BinaryImageRef<'a> = ImageRef<'a, bool>;

/// A mutable reference to a binary image. Alias for [`ImageRefMut<bool>`].
pub type BinaryImageRefMut<'a> = ImageRefMut<'a, bool>;

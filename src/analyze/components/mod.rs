//! Connected-component labeling on binary images.
//!
//! Two-pass union-find labeling on a [`BinaryImage`](crate::image::BinaryImage)
//! (`ImageView<Pixel = bool>`), parameterised by a compile-time
//! [`Connectivity`] strategy and a [`LabelPixel`]
//! output pixel type.
//!
//! Entry points:
//!
//! - [`connected_components`] \u2014 allocating; returns a [`Labeling<L>`].
//! - [`connected_components_into`] \u2014 in-place into a caller-supplied
//!   `Image<L>`.
//! - [`connected_components_with_stats`] \u2014 allocating; additionally
//!   returns one [`ComponentStats`] per foreground component, accumulated
//!   inline during pass 2 (area, bounding box, sums of `x` / `y` for
//!   centroid; aspect ratio is a derived helper).
//!
//! The implementation uses a conventional two-pass union-find labeling
//! design with explicit connectivity and label-pixel types.
//! # Example
//!
//! ```
//! use fovea::analyze::components::{
//!     connected_components, Connectivity4, Labeling,
//! };
//! use fovea::image::{BinaryImage, ImageView};
//! use fovea::pixel::Label32;
//!
//! //  . # # .
//! //  # # . .
//! //  . . # #
//! //  . . # .
//! let pixels = vec![
//!     false, true,  true,  false,
//!     true,  true,  false, false,
//!     false, false, true,  true,
//!     false, false, true,  false,
//! ];
//! let img = BinaryImage::from_vec(4, 4, pixels).unwrap();
//! let result: Labeling<Label32> =
//!     connected_components::<Label32, Connectivity4>(&img).unwrap();
//! assert_eq!(result.label_count, 2);
//! ```

mod connectivity;
mod engine;
mod stats;
mod union_find;

pub use connectivity::{Connectivity, Connectivity4, Connectivity8};
pub use engine::{
    connected_components, connected_components_into, connected_components_with_stats,
};
pub use stats::ComponentStats;

use crate::image::{Image, ImageView};
use crate::pixel::LabelPixel;

/// The result of a connected-components pass.
///
/// Plain-fields struct: callers read `labels` and `label_count`
/// directly. `labels` is a regular [`Image<L>`] that flows through
/// every `ImageView` / `RasterImage` / `SubView` consumer unchanged;
/// `label_count` exists so callers don't have to rescan the image to
/// recover the number of distinct foreground components.
///
/// Invariants:
///
/// - `labels.pixel_at(x, y) == L::zero()` iff the input pixel was
///   background (`false`).
/// - Every foreground pixel carries a label in
///   `1 ..= label_count` (dense \u2014 no gaps).
///
/// `Debug` and `Clone` are implemented manually (since `Image<L>` does
/// not implement `Debug`); `Eq`, `Hash`, and `PartialEq` are
/// deliberately *not* derived because there is no canonical notion of
/// equality for two labelings that survives a permutation of labels
/// (relabel-equivalence is a separate, follow-up concept).
pub struct Labeling<L: LabelPixel> {
    /// The label image. Pixel `(x, y) == L::zero()` is background;
    /// any other value is a foreground label in `1..=label_count`.
    pub labels: Image<L>,

    /// Number of distinct foreground components in `labels`.
    pub label_count: u64,
}

impl<L: LabelPixel> core::fmt::Debug for Labeling<L> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Labeling")
            .field("width", &self.labels.width())
            .field("height", &self.labels.height())
            .field("label_count", &self.label_count)
            .finish()
    }
}

impl<L: LabelPixel> Clone for Labeling<L> {
    fn clone(&self) -> Self {
        // Clone the underlying buffer pixel-by-pixel via `Image::generate`.
        let w = self.labels.width();
        let h = self.labels.height();
        let labels = Image::<L>::generate(w, h, |x, y| self.labels.pixel_at(x, y));
        Self {
            labels,
            label_count: self.label_count,
        }
    }
}

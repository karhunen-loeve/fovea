use crate::image::ImageView;
use crate::image::sequential::private;
use crate::image::sequential::{ContiguousImage, ImageArray};

/// A kernel / structuring element that bundles weights and an anchor point.
///
/// This trait abstracts over different kernel representations (e.g.,
/// compile-time-sized [`Neighborhood`] or runtime-sized [`Image`](crate::image::Image)).
/// It enables ergonomic APIs where weights and anchor travel together.
///
/// The key method is [`flipped`](Kernel::flipped), which returns a 180°-rotated
/// copy of the kernel. The `Self` return type preserves the concrete type,
/// so a stack-backed `Neighborhood` flips into another stack-backed
/// `Neighborhood` — zero heap allocation.
///
/// # Example
///
/// ```
/// use fovea::image::{Neighborhood, Kernel, ImageView};
///
/// let kernel = Neighborhood::<f32, 3, 3>::sobel_x();
/// let flipped = kernel.flipped();
///
/// // Flipping mirrors the anchor
/// assert_eq!(kernel.anchor(), flipped.anchor()); // centered kernel stays centered
/// assert_eq!(flipped.weights().width(), 3);
/// assert_eq!(flipped.weights().height(), 3);
///
/// // Flipping is an involution: flipping twice gives back the original
/// let double_flipped = flipped.flipped();
/// assert_eq!(kernel.as_slice(), double_flipped.as_slice());
/// ```
pub trait Kernel {
    /// The scalar weight type (e.g. `f32`, `bool`, `i32`).
    type Weight: Copy;

    /// The backing storage for weights, implementing [`ImageView`].
    type Weights: ImageView<Pixel = Self::Weight>;

    /// Returns a reference to the weight grid.
    fn weights(&self) -> &Self::Weights;

    /// Returns the anchor position `(x, y)` within the kernel.
    fn anchor(&self) -> (usize, usize);

    /// Returns a 180°-rotated copy of this kernel.
    ///
    /// For a kernel of size (W, H), the flipped kernel satisfies:
    /// ```text
    /// flipped(x, y) = original(W - 1 - x, H - 1 - y)
    /// ```
    /// The anchor is also mirrored: `(W - 1 - ax, H - 1 - ay)`.
    ///
    /// This is used internally by convolution (which is correlation with
    /// a flipped kernel).
    fn flipped(&self) -> Self
    where
        Self: Sized;
}

/// A kernel / structuring element: a small grid of **weights** plus an
/// **anchor point**.
///
/// `Neighborhood` unifies convolution kernels and morphological structuring
/// elements — the only difference is what fold operation the consumer applies.
///
/// The weight type `W` is generic:
/// - `f32` for convolution kernels
/// - `i16` / `i32` for integer-only pipelines
/// - `bool` for binary structuring elements
/// - `u8` for greyscale structuring elements
///
/// Dimensions are **const-generic**, enabling the compiler to unroll inner
/// loops and perform compile-time size checks.
///
/// For runtime-sized kernels, bypass `Neighborhood` and call
/// `fold_neighborhood` directly with `&Image<W>` + an explicit anchor.
///
/// # Example
///
/// ```
/// use fovea::image::{Neighborhood, ImageView};
///
/// // A simple 3×3 box blur kernel (un-normalised)
/// let kernel = Neighborhood::<f32, 3, 3>::new([
///     1.0, 1.0, 1.0,
///     1.0, 1.0, 1.0,
///     1.0, 1.0, 1.0,
/// ]);
/// assert_eq!(kernel.anchor(), (1, 1));
/// assert_eq!(kernel.weights().width(), 3);
/// assert_eq!(kernel.weights().height(), 3);
/// ```
///
/// ```
/// use fovea::image::{Neighborhood, ImageView};
///
/// // Iterate over kernel positions relative to anchor
/// let kernel = Neighborhood::<i32, 3, 1>::new([1, 2, 1]);
/// let positions: Vec<_> = kernel.positions().collect();
/// // dx runs from -1 to +1, dy = 0 (1D horizontal kernel)
/// assert_eq!(positions.len(), 3);
/// assert_eq!(positions[0], (-1, 0, 1));
/// assert_eq!(positions[1], (0, 0, 2));
/// assert_eq!(positions[2], (1, 0, 1));
/// ```
pub struct Neighborhood<W: Copy, const KW: usize, const KH: usize>
where
    private::Dim<W, KW, KH>: private::_Array2D<Pixel = W>,
{
    weights: ImageArray<W, KW, KH>,
    anchor: (usize, usize),
}

impl<W: Copy, const KW: usize, const KH: usize> Neighborhood<W, KW, KH>
where
    private::Dim<W, KW, KH>: private::_Array2D<Pixel = W>,
{
    /// Creates a new `Neighborhood` with the given weight data and the
    /// anchor at the center `(KW / 2, KH / 2)`.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::image::Neighborhood;
    ///
    /// let kernel = Neighborhood::<f32, 3, 3>::new([
    ///     0.0, -1.0,  0.0,
    ///    -1.0,  4.0, -1.0,
    ///     0.0, -1.0,  0.0,
    /// ]);
    /// assert_eq!(kernel.anchor(), (1, 1));
    /// ```
    pub fn new(data: <private::Dim<W, KW, KH> as private::_Array2D>::Array) -> Self {
        Self {
            weights: ImageArray::new(data),
            anchor: (KW / 2, KH / 2),
        }
    }

    /// Creates a new `Neighborhood` with the given weight data and an
    /// explicit anchor position.
    ///
    /// # Panics
    ///
    /// Panics if `anchor` is outside the kernel bounds
    /// (`anchor.0 >= KW || anchor.1 >= KH`).
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::image::Neighborhood;
    ///
    /// // Anchor at top-left corner
    /// let kernel = Neighborhood::<f32, 3, 3>::with_anchor(
    ///     [1.0; 9],
    ///     (0, 0),
    /// );
    /// assert_eq!(kernel.anchor(), (0, 0));
    /// ```
    pub fn with_anchor(
        data: <private::Dim<W, KW, KH> as private::_Array2D>::Array,
        anchor: (usize, usize),
    ) -> Self {
        assert!(
            anchor.0 < KW && anchor.1 < KH,
            "anchor ({}, {}) is out of bounds for {}x{} kernel",
            anchor.0,
            anchor.1,
            KW,
            KH,
        );
        Self {
            weights: ImageArray::new(data),
            anchor,
        }
    }

    /// Returns a reference to the underlying weight storage.
    ///
    /// The returned `ImageArray` implements [`ImageView`], so it can be
    /// passed directly to `fold_neighborhood` or composed with
    /// [`zip_pixels`](crate::image::zip_pixels).
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::image::{Neighborhood, ImageView};
    ///
    /// let kernel = Neighborhood::<f32, 3, 3>::new([1.0; 9]);
    /// assert_eq!(kernel.weights().size(), fovea::Size::new(3, 3));
    /// assert_eq!(kernel.weights().pixel_at(0, 0), 1.0);
    /// ```
    pub fn weights(&self) -> &ImageArray<W, KW, KH> {
        &self.weights
    }

    /// Returns the anchor position within the kernel.
    ///
    /// The anchor is the point that is aligned with the current output
    /// pixel during convolution or morphological operations. Default is
    /// the center: `(KW / 2, KH / 2)`.
    pub fn anchor(&self) -> (usize, usize) {
        self.anchor
    }

    /// Returns the kernel width (compile-time constant).
    pub fn kernel_width(&self) -> usize {
        KW
    }

    /// Returns the kernel height (compile-time constant).
    pub fn kernel_height(&self) -> usize {
        KH
    }

    /// Returns the weights as a flat slice in row-major order.
    ///
    /// This is a convenience method delegating to
    /// [`ContiguousImage::as_slice`](crate::image::ContiguousImage::as_slice).
    pub fn as_slice(&self) -> &[W] {
        self.weights.as_slice()
    }

    /// Iterates over all kernel positions, yielding
    /// `(dx, dy, &weight)` tuples where `dx` and `dy` are **signed
    /// offsets relative to the anchor**.
    ///
    /// Iteration order is row-major (left-to-right, top-to-bottom).
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::image::Neighborhood;
    ///
    /// let kernel = Neighborhood::<f32, 3, 3>::new([
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 9.0,
    /// ]);
    /// let positions: Vec<_> = kernel.positions().collect();
    /// assert_eq!(positions.len(), 9);
    ///
    /// // top-left corner: dx = -1, dy = -1, weight = 1.0
    /// assert_eq!(positions[0], (-1, -1, 1.0));
    /// // center: dx = 0, dy = 0, weight = 5.0
    /// assert_eq!(positions[4], (0, 0, 5.0));
    /// // bottom-right: dx = 1, dy = 1, weight = 9.0
    /// assert_eq!(positions[8], (1, 1, 9.0));
    /// ```
    pub fn positions(&self) -> PositionsIter<'_, W, KW, KH> {
        PositionsIter {
            weights: &self.weights,
            anchor: self.anchor,
            x: 0,
            y: 0,
        }
    }
}

// ─── Kernel trait impl ──────────────────────────────────────────────────

impl<W: Copy, const KW: usize, const KH: usize> Kernel for Neighborhood<W, KW, KH>
where
    private::Dim<W, KW, KH>: private::_Array2D<Pixel = W>,
{
    type Weight = W;
    type Weights = ImageArray<W, KW, KH>;

    fn weights(&self) -> &ImageArray<W, KW, KH> {
        &self.weights
    }

    fn anchor(&self) -> (usize, usize) {
        self.anchor
    }

    fn flipped(&self) -> Self {
        // Reverse the flat pixel array to achieve 180° rotation.
        // For row-major data, reversing all elements is equivalent to
        // reflecting both rows and columns.
        let flipped_weights =
            ImageArray::generate(|x, y| self.weights.pixel_at(KW - 1 - x, KH - 1 - y));
        let flipped_anchor = (KW - 1 - self.anchor.0, KH - 1 - self.anchor.1);
        Self {
            weights: flipped_weights,
            anchor: flipped_anchor,
        }
    }
}

// ─── Clone ──────────────────────────────────────────────────────────────

impl<W: Copy, const KW: usize, const KH: usize> Clone for Neighborhood<W, KW, KH>
where
    private::Dim<W, KW, KH>: private::_Array2D<Pixel = W>,
    <private::Dim<W, KW, KH> as private::_Array2D>::Array: Clone,
{
    fn clone(&self) -> Self {
        Self {
            weights: ImageArray::new(
                // # Safety rationale: The Array type implements Clone, so we
                // can clone the underlying data via as_ref() which gives us
                // &[W], reconstruct through the Array's Clone impl.
                //
                // We access the raw array through ContiguousImage::as_slice
                // → AsRef<[W]> → Clone the Array directly.
                //
                // Actually we just need to get at the array. ImageArray
                // stores `data: Array` and we can clone via the slice +
                // ptr read approach. But simpler: since Array: Clone, and
                // ImageArray wraps it, we can use a helper.
                //
                // The simplest correct approach: read the underlying array
                // from the slice. Since the array is [W; KW*KH] and Clone,
                // we can reconstruct it.
                {
                    // We know the data is [W; KW*KH]. We read it through
                    // a pointer copy since Array: Clone.
                    let slice = self.weights.as_slice();
                    // SAFETY: `slice` points to the start of `[W; KW*KH]`
                    // which is exactly `Array`. We read it, clone-style.
                    // Since `Array: Clone`, all elements are valid to copy.
                    // However we need `W: Clone` which is implied by
                    // `Array: Clone` for `[W; N]`.
                    unsafe {
                        std::ptr::read(slice.as_ptr()
                            as *const <private::Dim<W, KW, KH> as private::_Array2D>::Array)
                    }
                },
            ),
            anchor: self.anchor,
        }
    }
}

// ─── Debug ──────────────────────────────────────────────────────────────

impl<W: Copy + std::fmt::Debug, const KW: usize, const KH: usize> std::fmt::Debug
    for Neighborhood<W, KW, KH>
where
    private::Dim<W, KW, KH>: private::_Array2D<Pixel = W>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Neighborhood")
            .field("weights", &self.as_slice())
            .field("anchor", &self.anchor)
            .field("size", &(KW, KH))
            .finish()
    }
}

// ─── PositionsIter ──────────────────────────────────────────────────────

/// Iterator over kernel positions yielding `(dx, dy, weight)` tuples
/// relative to the anchor. Created by [`Neighborhood::positions`].
pub struct PositionsIter<'a, W, const KW: usize, const KH: usize>
where
    private::Dim<W, KW, KH>: private::_Array2D<Pixel = W>,
{
    weights: &'a ImageArray<W, KW, KH>,
    anchor: (usize, usize),
    x: usize,
    y: usize,
}

impl<'a, W: Copy, const KW: usize, const KH: usize> Iterator for PositionsIter<'a, W, KW, KH>
where
    private::Dim<W, KW, KH>: private::_Array2D<Pixel = W>,
{
    type Item = (isize, isize, W);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.y >= KH {
            return None;
        }

        let dx = self.x as isize - self.anchor.0 as isize;
        let dy = self.y as isize - self.anchor.1 as isize;
        let weight = self.weights.pixel_at(self.x, self.y);

        self.x += 1;
        if self.x >= KW {
            self.x = 0;
            self.y += 1;
        }

        Some((dx, dy, weight))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = if self.y >= KH {
            0
        } else {
            (KH - self.y - 1) * KW + (KW - self.x)
        };
        (remaining, Some(remaining))
    }
}

impl<'a, W: Copy, const KW: usize, const KH: usize> ExactSizeIterator
    for PositionsIter<'a, W, KW, KH>
where
    private::Dim<W, KW, KH>: private::_Array2D<Pixel = W>,
{
}

// ─── Type aliases ───────────────────────────────────────────────────────

/// A 3×3 convolution kernel with `f32` weights.
pub type Kernel3x3 = Neighborhood<f32, 3, 3>;

/// A 5×5 convolution kernel with `f32` weights.
pub type Kernel5x5 = Neighborhood<f32, 5, 5>;

/// A 7×7 convolution kernel with `f32` weights.
pub type Kernel7x7 = Neighborhood<f32, 7, 7>;

/// A 3×3 integer convolution kernel with `i32` weights.
pub type Kernel3x3i = Neighborhood<i32, 3, 3>;

/// A 5×5 integer convolution kernel with `i32` weights.
pub type Kernel5x5i = Neighborhood<i32, 5, 5>;

/// A binary mask of arbitrary size — a `Neighborhood` whose weights are
/// `bool` values indicating which positions are active.
///
/// Used by morphological operations and (once implemented) `map_neighborhood`
/// to express topology without numeric weights.
pub type Mask<const KW: usize, const KH: usize> = Neighborhood<bool, KW, KH>;

/// A 3×3 binary mask.
pub type Mask3x3 = Mask<3, 3>;

/// A 5×5 binary mask.
pub type Mask5x5 = Mask<5, 5>;

// ─── Built-in 2D f32 kernels ────────────────────────────────────────────

impl Neighborhood<f32, 3, 3> {
    /// 3×3 box blur kernel (normalized — each weight is `1.0 / 9.0`).
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::image::Neighborhood;
    ///
    /// let k = Neighborhood::box_blur_3x3();
    /// assert_eq!(k.anchor(), (1, 1));
    /// let sum: f32 = k.positions().map(|(_, _, w)| w).sum();
    /// assert!((sum - 1.0).abs() < 1e-6);
    /// ```
    pub fn box_blur_3x3() -> Self {
        Self::new([1.0 / 9.0; 9])
    }

    /// 3×3 Gaussian blur kernel (approximate, un-normalised).
    ///
    /// Weights: `[1 2 1; 2 4 2; 1 2 1]`. Sum = 16.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::image::Neighborhood;
    ///
    /// let k = Neighborhood::gaussian_3x3();
    /// let sum: f32 = k.positions().map(|(_, _, w)| w).sum();
    /// assert_eq!(sum, 16.0);
    /// ```
    pub fn gaussian_3x3() -> Self {
        Self::new([1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0])
    }

    /// Sobel operator for horizontal edges (detects vertical gradients: dI/dy).
    ///
    /// ```text
    /// -1 -2 -1
    ///  0  0  0
    ///  1  2  1
    /// ```
    pub fn sobel_x() -> Self {
        Self::new([-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0])
    }

    /// Sobel operator for vertical edges (detects horizontal gradients: dI/dx).
    ///
    /// ```text
    /// -1  0  1
    /// -2  0  2
    /// -1  0  1
    /// ```
    pub fn sobel_y() -> Self {
        Self::new([-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0])
    }

    /// Scharr operator for horizontal edges (more rotation-invariant than Sobel).
    ///
    /// ```text
    ///  -3 -10  -3
    ///   0   0   0
    ///   3  10   3
    /// ```
    pub fn scharr_x() -> Self {
        Self::new([-3.0, -10.0, -3.0, 0.0, 0.0, 0.0, 3.0, 10.0, 3.0])
    }

    /// Scharr operator for vertical edges.
    ///
    /// ```text
    ///  -3   0   3
    /// -10   0  10
    ///  -3   0   3
    /// ```
    pub fn scharr_y() -> Self {
        Self::new([-3.0, 0.0, 3.0, -10.0, 0.0, 10.0, -3.0, 0.0, 3.0])
    }

    /// Prewitt operator for horizontal edges.
    ///
    /// ```text
    /// -1 -1 -1
    ///  0  0  0
    ///  1  1  1
    /// ```
    pub fn prewitt_x() -> Self {
        Self::new([-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    }

    /// Prewitt operator for vertical edges.
    ///
    /// ```text
    /// -1  0  1
    /// -1  0  1
    /// -1  0  1
    /// ```
    pub fn prewitt_y() -> Self {
        Self::new([-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0])
    }

    /// 3×3 Laplacian kernel (4-connected).
    ///
    /// ```text
    ///  0 -1  0
    /// -1  4 -1
    ///  0 -1  0
    /// ```
    pub fn laplacian() -> Self {
        Self::new([0.0, -1.0, 0.0, -1.0, 4.0, -1.0, 0.0, -1.0, 0.0])
    }

    /// 3×3 Laplacian kernel (8-connected / diagonal-inclusive).
    ///
    /// ```text
    /// -1 -1 -1
    /// -1  8 -1
    /// -1 -1 -1
    /// ```
    pub fn laplacian_8() -> Self {
        Self::new([-1.0, -1.0, -1.0, -1.0, 8.0, -1.0, -1.0, -1.0, -1.0])
    }

    /// 3×3 sharpening kernel (identity + scaled Laplacian).
    ///
    /// ```text
    ///  0 -1  0
    /// -1  5 -1
    ///  0 -1  0
    /// ```
    pub fn sharpen() -> Self {
        Self::new([0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0])
    }

    /// 3×3 identity kernel. Convolution with this kernel produces the
    /// original image (useful for testing).
    ///
    /// ```text
    /// 0 0 0
    /// 0 1 0
    /// 0 0 0
    /// ```
    pub fn identity_3x3() -> Self {
        Self::new([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    }

    /// 3×3 emboss kernel.
    ///
    /// ```text
    /// -2 -1  0
    /// -1  1  1
    ///  0  1  2
    /// ```
    pub fn emboss() -> Self {
        Self::new([-2.0, -1.0, 0.0, -1.0, 1.0, 1.0, 0.0, 1.0, 2.0])
    }
}

impl Neighborhood<f32, 5, 5> {
    /// 5×5 Gaussian blur kernel (approximate, un-normalised).
    ///
    /// Weights based on binomial coefficients. Sum = 256.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::image::Neighborhood;
    ///
    /// let k = Neighborhood::gaussian_5x5();
    /// let sum: f32 = k.positions().map(|(_, _, w)| w).sum();
    /// assert_eq!(sum, 256.0);
    /// ```
    pub fn gaussian_5x5() -> Self {
        #[rustfmt::skip]
        let data = [
             1.0,  4.0,  6.0,  4.0,  1.0,
             4.0, 16.0, 24.0, 16.0,  4.0,
             6.0, 24.0, 36.0, 24.0,  6.0,
             4.0, 16.0, 24.0, 16.0,  4.0,
             1.0,  4.0,  6.0,  4.0,  1.0,
        ];
        Self::new(data)
    }

    /// 5×5 box blur kernel (normalized — each weight is `1.0 / 25.0`).
    pub fn box_blur_5x5() -> Self {
        Self::new([1.0 / 25.0; 25])
    }

    /// 5×5 identity kernel.
    pub fn identity_5x5() -> Self {
        #[rustfmt::skip]
        let data = [
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        Self::new(data)
    }
}

// ─── Built-in 1D kernels (for separable convolution) ────────────────────

impl Neighborhood<f32, 3, 1> {
    /// 1D Gaussian kernel (horizontal), weights `[1, 2, 1]`.
    ///
    /// Pair with [`Neighborhood::gaussian_1d_3_v`] for a separable 3×3
    /// Gaussian blur.
    pub fn gaussian_1d_3_h() -> Self {
        Self::new([1.0, 2.0, 1.0])
    }

    /// 1D box kernel (horizontal, normalized), size 3.
    pub fn box_1d_3_h() -> Self {
        Self::new([1.0 / 3.0; 3])
    }
}

impl Neighborhood<f32, 1, 3> {
    /// 1D Gaussian kernel (vertical), weights `[1; 2; 1]`.
    ///
    /// Pair with [`Neighborhood::gaussian_1d_3_h`] for a separable 3×3
    /// Gaussian blur.
    pub fn gaussian_1d_3_v() -> Self {
        Self::new([1.0, 2.0, 1.0])
    }

    /// 1D box kernel (vertical, normalized), size 3.
    pub fn box_1d_3_v() -> Self {
        Self::new([1.0 / 3.0; 3])
    }
}

impl Neighborhood<f32, 5, 1> {
    /// 1D Gaussian kernel (horizontal), weights `[1, 4, 6, 4, 1]`.
    pub fn gaussian_1d_5_h() -> Self {
        Self::new([1.0, 4.0, 6.0, 4.0, 1.0])
    }

    /// 1D box kernel (horizontal, normalized), size 5.
    pub fn box_1d_5_h() -> Self {
        Self::new([1.0 / 5.0; 5])
    }
}

impl Neighborhood<f32, 1, 5> {
    /// 1D Gaussian kernel (vertical), weights `[1; 4; 6; 4; 1]`.
    pub fn gaussian_1d_5_v() -> Self {
        Self::new([1.0, 4.0, 6.0, 4.0, 1.0])
    }

    /// 1D box kernel (vertical, normalized), size 5.
    pub fn box_1d_5_v() -> Self {
        Self::new([1.0 / 5.0; 5])
    }
}

// ─── Built-in structuring elements ──────────────────────────────────────

impl Neighborhood<bool, 3, 3> {
    /// 3×3 full-rectangle structuring element (all `true`).
    ///
    /// Used for standard erosion/dilation.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::image::Neighborhood;
    ///
    /// let se = Neighborhood::full_rect_3x3();
    /// assert_eq!(se.positions().filter(|&(_, _, w)| w).count(), 9);
    /// ```
    pub fn full_rect_3x3() -> Self {
        Self::new([true; 9])
    }

    /// 3×3 cross (plus-shaped) structuring element.
    ///
    /// ```text
    /// .  #  .
    /// #  #  #
    /// .  #  .
    /// ```
    pub fn cross_3x3() -> Self {
        #[rustfmt::skip]
        let data = [
            false, true, false,
            true,  true, true,
            false, true, false,
        ];
        Self::new(data)
    }

    /// 3×3 diamond structuring element (same as cross for 3×3).
    pub fn diamond_3x3() -> Self {
        Self::cross_3x3()
    }
}

impl Neighborhood<bool, 5, 5> {
    /// 5×5 full-rectangle structuring element.
    pub fn full_rect_5x5() -> Self {
        Self::new([true; 25])
    }

    /// 5×5 cross structuring element.
    ///
    /// ```text
    /// .  .  #  .  .
    /// .  .  #  .  .
    /// #  #  #  #  #
    /// .  .  #  .  .
    /// .  .  #  .  .
    /// ```
    pub fn cross_5x5() -> Self {
        #[rustfmt::skip]
        let data = [
            false, false, true,  false, false,
            false, false, true,  false, false,
            true,  true,  true,  true,  true,
            false, false, true,  false, false,
            false, false, true,  false, false,
        ];
        Self::new(data)
    }

    /// 5×5 diamond structuring element.
    ///
    /// ```text
    /// .  .  #  .  .
    /// .  #  #  #  .
    /// #  #  #  #  #
    /// .  #  #  #  .
    /// .  .  #  .  .
    /// ```
    pub fn diamond_5x5() -> Self {
        #[rustfmt::skip]
        let data = [
            false, false, true,  false, false,
            false, true,  true,  true,  false,
            true,  true,  true,  true,  true,
            false, true,  true,  true,  false,
            false, false, true,  false, false,
        ];
        Self::new(data)
    }

    /// 5×5 approximate circle (disc) structuring element.
    ///
    /// ```text
    /// .  #  #  #  .
    /// #  #  #  #  #
    /// #  #  #  #  #
    /// #  #  #  #  #
    /// .  #  #  #  .
    /// ```
    pub fn circle_5x5() -> Self {
        #[rustfmt::skip]
        let data = [
            false, true,  true,  true,  false,
            true,  true,  true,  true,  true,
            true,  true,  true,  true,  true,
            true,  true,  true,  true,  true,
            false, true,  true,  true,  false,
        ];
        Self::new(data)
    }
}

// ─── Integer kernels ────────────────────────────────────────────────────

impl Neighborhood<i32, 3, 3> {
    /// 3×3 Sobel operator (horizontal edges) with integer weights.
    ///
    /// ```text
    /// -1 -2 -1
    ///  0  0  0
    ///  1  2  1
    /// ```
    pub fn sobel_x_i32() -> Self {
        Self::new([-1, -2, -1, 0, 0, 0, 1, 2, 1])
    }

    /// 3×3 Sobel operator (vertical edges) with integer weights.
    pub fn sobel_y_i32() -> Self {
        Self::new([-1, 0, 1, -2, 0, 2, -1, 0, 1])
    }

    /// 3×3 Laplacian with integer weights (4-connected).
    pub fn laplacian_i32() -> Self {
        Self::new([0, -1, 0, -1, 4, -1, 0, -1, 0])
    }

    /// 3×3 Scharr operator (horizontal edges) with integer weights.
    pub fn scharr_x_i32() -> Self {
        Self::new([-3, -10, -3, 0, 0, 0, 3, 10, 3])
    }

    /// 3×3 Scharr operator (vertical edges) with integer weights.
    pub fn scharr_y_i32() -> Self {
        Self::new([-3, 0, 3, -10, 0, 10, -3, 0, 3])
    }

    /// 3×3 Prewitt operator (horizontal edges) with integer weights.
    pub fn prewitt_x_i32() -> Self {
        Self::new([-1, -1, -1, 0, 0, 0, 1, 1, 1])
    }

    /// 3×3 Prewitt operator (vertical edges) with integer weights.
    pub fn prewitt_y_i32() -> Self {
        Self::new([-1, 0, 1, -1, 0, 1, -1, 0, 1])
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Size;
    use crate::image::ImageView;

    // ── Construction ────────────────────────────────────────────────

    #[test]
    fn test_new_default_anchor_3x3() {
        let k = Neighborhood::<f32, 3, 3>::new([0.0; 9]);
        assert_eq!(k.anchor(), (1, 1));
        assert_eq!(k.kernel_width(), 3);
        assert_eq!(k.kernel_height(), 3);
    }

    #[test]
    fn test_new_default_anchor_5x5() {
        let k = Neighborhood::<f32, 5, 5>::new([0.0; 25]);
        assert_eq!(k.anchor(), (2, 2));
    }

    #[test]
    fn test_new_default_anchor_1x1() {
        let k = Neighborhood::<f32, 1, 1>::new([1.0]);
        assert_eq!(k.anchor(), (0, 0));
    }

    #[test]
    fn test_new_default_anchor_even_size() {
        // 4×4 kernel: anchor at (2, 2) — integer division
        let k = Neighborhood::<f32, 4, 4>::new([0.0; 16]);
        assert_eq!(k.anchor(), (2, 2));
    }

    #[test]
    fn test_new_default_anchor_non_square() {
        let k = Neighborhood::<f32, 5, 3>::new([0.0; 15]);
        assert_eq!(k.anchor(), (2, 1));
    }

    #[test]
    fn test_with_anchor_custom() {
        let k = Neighborhood::<f32, 3, 3>::with_anchor([0.0; 9], (0, 0));
        assert_eq!(k.anchor(), (0, 0));
    }

    #[test]
    fn test_with_anchor_bottom_right() {
        let k = Neighborhood::<f32, 3, 3>::with_anchor([0.0; 9], (2, 2));
        assert_eq!(k.anchor(), (2, 2));
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_with_anchor_out_of_bounds_x() {
        Neighborhood::<f32, 3, 3>::with_anchor([0.0; 9], (3, 1));
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_with_anchor_out_of_bounds_y() {
        Neighborhood::<f32, 3, 3>::with_anchor([0.0; 9], (1, 3));
    }

    // ── Accessors ───────────────────────────────────────────────────

    #[test]
    fn test_weights_accessor() {
        let k = Neighborhood::<f32, 3, 3>::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let w = k.weights();
        assert_eq!(w.size(), Size::new(3, 3));
        assert_eq!(w.pixel_at(0, 0), 1.0);
        assert_eq!(w.pixel_at(1, 1), 5.0);
        assert_eq!(w.pixel_at(2, 2), 9.0);
    }

    #[test]
    fn test_as_slice() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let k = Neighborhood::<f32, 3, 3>::new(data);
        assert_eq!(k.as_slice(), &data);
    }

    #[test]
    fn test_kernel_width_height() {
        let k = Neighborhood::<f32, 7, 5>::new([0.0; 35]);
        assert_eq!(k.kernel_width(), 7);
        assert_eq!(k.kernel_height(), 5);
    }

    // ── positions() iterator ────────────────────────────────────────

    #[test]
    fn test_positions_3x3_centered() {
        let k = Neighborhood::<f32, 3, 3>::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let positions: Vec<_> = k.positions().collect();
        assert_eq!(positions.len(), 9);

        // Row 0: dy = -1
        assert_eq!(positions[0], (-1, -1, 1.0));
        assert_eq!(positions[1], (0, -1, 2.0));
        assert_eq!(positions[2], (1, -1, 3.0));

        // Row 1: dy = 0
        assert_eq!(positions[3], (-1, 0, 4.0));
        assert_eq!(positions[4], (0, 0, 5.0));
        assert_eq!(positions[5], (1, 0, 6.0));

        // Row 2: dy = 1
        assert_eq!(positions[6], (-1, 1, 7.0));
        assert_eq!(positions[7], (0, 1, 8.0));
        assert_eq!(positions[8], (1, 1, 9.0));
    }

    #[test]
    fn test_positions_1d_horizontal() {
        let k = Neighborhood::<i32, 3, 1>::new([1, 2, 1]);
        let positions: Vec<_> = k.positions().collect();
        assert_eq!(positions.len(), 3);
        assert_eq!(positions[0], (-1, 0, 1));
        assert_eq!(positions[1], (0, 0, 2));
        assert_eq!(positions[2], (1, 0, 1));
    }

    #[test]
    fn test_positions_1d_vertical() {
        let k = Neighborhood::<i32, 1, 3>::new([1, 2, 1]);
        let positions: Vec<_> = k.positions().collect();
        assert_eq!(positions.len(), 3);
        assert_eq!(positions[0], (0, -1, 1));
        assert_eq!(positions[1], (0, 0, 2));
        assert_eq!(positions[2], (0, 1, 1));
    }

    #[test]
    fn test_positions_custom_anchor() {
        let k = Neighborhood::<f32, 3, 3>::with_anchor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            (0, 0),
        );
        let positions: Vec<_> = k.positions().collect();
        // With anchor at (0,0), dx and dy are all non-negative
        assert_eq!(positions[0], (0, 0, 1.0));
        assert_eq!(positions[1], (1, 0, 2.0));
        assert_eq!(positions[2], (2, 0, 3.0));
        assert_eq!(positions[3], (0, 1, 4.0));
        assert_eq!(positions[8], (2, 2, 9.0));
    }

    #[test]
    fn test_positions_1x1() {
        let k = Neighborhood::<f32, 1, 1>::new([42.0]);
        let positions: Vec<_> = k.positions().collect();
        assert_eq!(positions.len(), 1);
        assert_eq!(positions[0], (0, 0, 42.0));
    }

    #[test]
    fn test_positions_5x5_center() {
        let k = Neighborhood::<f32, 5, 5>::new([0.0; 25]);
        let positions: Vec<_> = k.positions().collect();
        assert_eq!(positions.len(), 25);
        // First position: top-left
        assert_eq!(positions[0].0, -2); // dx
        assert_eq!(positions[0].1, -2); // dy
        // Center position (index 12 = 2*5 + 2)
        assert_eq!(positions[12].0, 0);
        assert_eq!(positions[12].1, 0);
        // Last position: bottom-right
        assert_eq!(positions[24].0, 2);
        assert_eq!(positions[24].1, 2);
    }

    // ── ExactSizeIterator ───────────────────────────────────────────

    #[test]
    fn test_positions_exact_size() {
        let k = Neighborhood::<f32, 3, 3>::new([0.0; 9]);
        let mut iter = k.positions();
        assert_eq!(iter.len(), 9);
        iter.next();
        assert_eq!(iter.len(), 8);
        for _ in &mut iter {}
        assert_eq!(iter.len(), 0);
    }

    #[test]
    fn test_positions_size_hint() {
        let k = Neighborhood::<f32, 5, 5>::new([0.0; 25]);
        let iter = k.positions();
        let (lo, hi) = iter.size_hint();
        assert_eq!(lo, 25);
        assert_eq!(hi, Some(25));
    }

    #[test]
    fn test_positions_size_hint_after_partial_consume() {
        let k = Neighborhood::<f32, 3, 3>::new([0.0; 9]);
        let mut iter = k.positions();
        iter.next(); // consume 1
        iter.next(); // consume 2
        iter.next(); // consume 3
        let (lo, hi) = iter.size_hint();
        assert_eq!(lo, 6);
        assert_eq!(hi, Some(6));
    }

    #[test]
    fn test_positions_exhausted() {
        let k = Neighborhood::<f32, 1, 1>::new([1.0]);
        let mut iter = k.positions();
        assert!(iter.next().is_some());
        assert!(iter.next().is_none());
        assert!(iter.next().is_none()); // fused-like behaviour
        assert_eq!(iter.len(), 0);
    }

    // ── Built-in f32 kernels ────────────────────────────────────────

    #[test]
    fn test_box_blur_3x3() {
        let k = Neighborhood::box_blur_3x3();
        assert_eq!(k.anchor(), (1, 1));
        let sum: f32 = k.positions().map(|(_, _, w)| w).sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_gaussian_3x3() {
        let k = Neighborhood::gaussian_3x3();
        assert_eq!(k.anchor(), (1, 1));
        let sum: f32 = k.positions().map(|(_, _, w)| w).sum();
        assert_eq!(sum, 16.0);
    }

    #[test]
    fn test_gaussian_5x5() {
        let k = Neighborhood::gaussian_5x5();
        assert_eq!(k.anchor(), (2, 2));
        let sum: f32 = k.positions().map(|(_, _, w)| w).sum();
        assert_eq!(sum, 256.0);
    }

    #[test]
    fn test_sobel_x_antisymmetric() {
        let k = Neighborhood::sobel_x();
        let sum: f32 = k.positions().map(|(_, _, w)| w).sum();
        assert_eq!(sum, 0.0);
    }

    #[test]
    fn test_sobel_y_antisymmetric() {
        let k = Neighborhood::sobel_y();
        let sum: f32 = k.positions().map(|(_, _, w)| w).sum();
        assert_eq!(sum, 0.0);
    }

    #[test]
    fn test_scharr_x_antisymmetric() {
        let k = Neighborhood::scharr_x();
        let sum: f32 = k.positions().map(|(_, _, w)| w).sum();
        assert_eq!(sum, 0.0);
    }

    #[test]
    fn test_scharr_y_antisymmetric() {
        let k = Neighborhood::scharr_y();
        let sum: f32 = k.positions().map(|(_, _, w)| w).sum();
        assert_eq!(sum, 0.0);
    }

    #[test]
    fn test_prewitt_x_antisymmetric() {
        let k = Neighborhood::prewitt_x();
        let sum: f32 = k.positions().map(|(_, _, w)| w).sum();
        assert_eq!(sum, 0.0);
    }

    #[test]
    fn test_prewitt_y_antisymmetric() {
        let k = Neighborhood::prewitt_y();
        let sum: f32 = k.positions().map(|(_, _, w)| w).sum();
        assert_eq!(sum, 0.0);
    }

    #[test]
    fn test_laplacian_sum_zero() {
        let k = Neighborhood::laplacian();
        let sum: f32 = k.positions().map(|(_, _, w)| w).sum();
        assert_eq!(sum, 0.0);
    }

    #[test]
    fn test_laplacian_8_sum_zero() {
        let k = Neighborhood::laplacian_8();
        let sum: f32 = k.positions().map(|(_, _, w)| w).sum();
        assert_eq!(sum, 0.0);
    }

    #[test]
    fn test_sharpen_center_weight() {
        let k = Neighborhood::sharpen();
        assert_eq!(k.weights().pixel_at(1, 1), 5.0);
    }

    #[test]
    fn test_identity_3x3() {
        let k = Neighborhood::identity_3x3();
        let sum: f32 = k.positions().map(|(_, _, w)| w).sum();
        assert_eq!(sum, 1.0);
        assert_eq!(k.weights().pixel_at(1, 1), 1.0);
        // All non-center weights should be 0
        for (dx, dy, w) in k.positions() {
            if dx == 0 && dy == 0 {
                assert_eq!(w, 1.0);
            } else {
                assert_eq!(w, 0.0);
            }
        }
    }

    #[test]
    fn test_identity_5x5() {
        let k = Neighborhood::identity_5x5();
        let sum: f32 = k.positions().map(|(_, _, w)| w).sum();
        assert_eq!(sum, 1.0);
        assert_eq!(k.weights().pixel_at(2, 2), 1.0);
    }

    #[test]
    fn test_box_blur_5x5() {
        let k = Neighborhood::box_blur_5x5();
        let sum: f32 = k.positions().map(|(_, _, w)| w).sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_emboss() {
        let k = Neighborhood::emboss();
        assert_eq!(k.anchor(), (1, 1));
        // Emboss kernel: -2 + -1 + 0 + -1 + 1 + 1 + 0 + 1 + 2 = 1
        let sum: f32 = k.positions().map(|(_, _, w)| w).sum();
        assert_eq!(sum, 1.0);
    }

    // ── Built-in 1D kernels ─────────────────────────────────────────

    #[test]
    fn test_gaussian_1d_3_h() {
        let k = Neighborhood::gaussian_1d_3_h();
        assert_eq!(k.anchor(), (1, 0));
        assert_eq!(k.kernel_width(), 3);
        assert_eq!(k.kernel_height(), 1);
        let sum: f32 = k.positions().map(|(_, _, w)| w).sum();
        assert_eq!(sum, 4.0);
    }

    #[test]
    fn test_gaussian_1d_3_v() {
        let k = Neighborhood::gaussian_1d_3_v();
        assert_eq!(k.anchor(), (0, 1));
        assert_eq!(k.kernel_width(), 1);
        assert_eq!(k.kernel_height(), 3);
        let sum: f32 = k.positions().map(|(_, _, w)| w).sum();
        assert_eq!(sum, 4.0);
    }

    #[test]
    fn test_gaussian_1d_5_h() {
        let k = Neighborhood::gaussian_1d_5_h();
        assert_eq!(k.anchor(), (2, 0));
        let sum: f32 = k.positions().map(|(_, _, w)| w).sum();
        assert_eq!(sum, 16.0);
    }

    #[test]
    fn test_gaussian_1d_5_v() {
        let k = Neighborhood::gaussian_1d_5_v();
        assert_eq!(k.anchor(), (0, 2));
        let sum: f32 = k.positions().map(|(_, _, w)| w).sum();
        assert_eq!(sum, 16.0);
    }

    #[test]
    fn test_box_1d_3_h() {
        let k = Neighborhood::box_1d_3_h();
        let sum: f32 = k.as_slice().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_box_1d_3_v() {
        let k = Neighborhood::box_1d_3_v();
        let sum: f32 = k.as_slice().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_box_1d_5_h() {
        let k = Neighborhood::box_1d_5_h();
        let sum: f32 = k.as_slice().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_box_1d_5_v() {
        let k = Neighborhood::box_1d_5_v();
        let sum: f32 = k.as_slice().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    // ── Structuring elements ────────────────────────────────────────

    #[test]
    fn test_full_rect_3x3() {
        let se = Neighborhood::full_rect_3x3();
        assert_eq!(se.anchor(), (1, 1));
        assert!(se.as_slice().iter().all(|&v| v));
    }

    #[test]
    fn test_cross_3x3() {
        let se = Neighborhood::cross_3x3();
        let active: Vec<_> = se.positions().filter(|&(_, _, w)| w).collect();
        assert_eq!(active.len(), 5); // center row + center column, minus overlap
        // The center pixel should be active
        assert!(active.iter().any(|&(dx, dy, _)| dx == 0 && dy == 0));
        // Corners should be inactive
        let corners: Vec<_> = se
            .positions()
            .filter(|&(dx, dy, _)| dx.abs() == 1 && dy.abs() == 1)
            .collect();
        assert!(corners.iter().all(|&(_, _, w)| !w));
    }

    #[test]
    fn test_diamond_3x3_same_as_cross() {
        let d = Neighborhood::diamond_3x3();
        let c = Neighborhood::cross_3x3();
        assert_eq!(d.as_slice(), c.as_slice());
    }

    #[test]
    fn test_full_rect_5x5() {
        let se = Neighborhood::full_rect_5x5();
        assert_eq!(se.anchor(), (2, 2));
        assert_eq!(se.as_slice().len(), 25);
        assert!(se.as_slice().iter().all(|&v| v));
    }

    #[test]
    fn test_cross_5x5() {
        let se = Neighborhood::cross_5x5();
        let active: Vec<_> = se.positions().filter(|&(_, _, w)| w).collect();
        // Center row (5) + center column (5) - overlap (1) = 9
        assert_eq!(active.len(), 9);
    }

    #[test]
    fn test_diamond_5x5() {
        let se = Neighborhood::diamond_5x5();
        let active: Vec<_> = se.positions().filter(|&(_, _, w)| w).collect();
        // Diamond: 1 + 3 + 5 + 3 + 1 = 13
        assert_eq!(active.len(), 13);
    }

    #[test]
    fn test_circle_5x5() {
        let se = Neighborhood::circle_5x5();
        let active: Vec<_> = se.positions().filter(|&(_, _, w)| w).collect();
        // Circle: 3 + 5 + 5 + 5 + 3 = 21
        assert_eq!(active.len(), 21);
    }

    // ── Integer kernels ─────────────────────────────────────────────

    #[test]
    fn test_sobel_x_i32() {
        let k = Neighborhood::sobel_x_i32();
        let sum: i32 = k.positions().map(|(_, _, w)| w).sum();
        assert_eq!(sum, 0);
    }

    #[test]
    fn test_sobel_y_i32() {
        let k = Neighborhood::sobel_y_i32();
        let sum: i32 = k.positions().map(|(_, _, w)| w).sum();
        assert_eq!(sum, 0);
    }

    #[test]
    fn test_laplacian_i32() {
        let k = Neighborhood::laplacian_i32();
        let sum: i32 = k.positions().map(|(_, _, w)| w).sum();
        assert_eq!(sum, 0);
    }

    #[test]
    fn test_scharr_x_i32() {
        let k = Neighborhood::scharr_x_i32();
        let sum: i32 = k.positions().map(|(_, _, w)| w).sum();
        assert_eq!(sum, 0);
    }

    #[test]
    fn test_scharr_y_i32() {
        let k = Neighborhood::scharr_y_i32();
        let sum: i32 = k.positions().map(|(_, _, w)| w).sum();
        assert_eq!(sum, 0);
    }

    #[test]
    fn test_prewitt_x_i32() {
        let k = Neighborhood::prewitt_x_i32();
        let sum: i32 = k.positions().map(|(_, _, w)| w).sum();
        assert_eq!(sum, 0);
    }

    #[test]
    fn test_prewitt_y_i32() {
        let k = Neighborhood::prewitt_y_i32();
        let sum: i32 = k.positions().map(|(_, _, w)| w).sum();
        assert_eq!(sum, 0);
    }

    // ── Type aliases ────────────────────────────────────────────────

    #[test]
    fn test_kernel3x3_alias() {
        let k: Kernel3x3 = Neighborhood::identity_3x3();
        assert_eq!(k.weights().size(), Size::new(3, 3));
    }

    #[test]
    fn test_kernel5x5_alias() {
        let k: Kernel5x5 = Neighborhood::identity_5x5();
        assert_eq!(k.weights().size(), Size::new(5, 5));
    }

    #[test]
    fn test_kernel3x3i_alias() {
        let k: Kernel3x3i = Neighborhood::sobel_x_i32();
        assert_eq!(k.weights().size(), Size::new(3, 3));
    }

    #[test]
    fn test_mask_3x3_alias() {
        let m: Mask3x3 = Neighborhood::full_rect_3x3();
        assert_eq!(m.weights().size(), Size::new(3, 3));
    }

    #[test]
    fn test_mask_5x5_alias() {
        let m: Mask5x5 = Neighborhood::full_rect_5x5();
        assert_eq!(m.weights().size(), Size::new(5, 5));
    }

    #[test]
    fn test_mask_general_alias() {
        let m: Mask<7, 7> = Neighborhood::new([true; 49]);
        assert_eq!(m.weights().size(), Size::new(7, 7));
    }

    // ── Clone ───────────────────────────────────────────────────────

    #[test]
    fn test_clone() {
        let k = Neighborhood::<f32, 3, 3>::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let k2 = k.clone();
        assert_eq!(k.as_slice(), k2.as_slice());
        assert_eq!(k.anchor(), k2.anchor());
    }

    #[test]
    fn test_clone_with_custom_anchor() {
        let k = Neighborhood::<f32, 3, 3>::with_anchor([1.0; 9], (0, 2));
        let k2 = k.clone();
        assert_eq!(k2.anchor(), (0, 2));
        assert_eq!(k2.as_slice(), k.as_slice());
    }

    #[test]
    fn test_clone_bool() {
        let se = Neighborhood::cross_3x3();
        let se2 = se.clone();
        assert_eq!(se.as_slice(), se2.as_slice());
    }

    // ── Debug ───────────────────────────────────────────────────────

    #[test]
    fn test_debug_output() {
        let k = Neighborhood::<f32, 3, 3>::new([0.0; 9]);
        let dbg = format!("{:?}", k);
        assert!(dbg.contains("Neighborhood"));
        assert!(dbg.contains("anchor"));
        assert!(dbg.contains("size"));
    }

    // ── Weights implement ImageView ─────────────────────────────────

    #[test]
    fn test_weights_imageview() {
        let k = Neighborhood::<f32, 3, 3>::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let w = k.weights();
        assert_eq!(w.width(), 3);
        assert_eq!(w.height(), 3);
        assert_eq!(w.pixel_at(0, 0), 1.0);
        assert_eq!(w.pixel_at(2, 2), 9.0);
        assert_eq!(w.get(3, 0), None);
    }

    // ── Composability with zip_pixels ───────────────────────────────

    #[test]
    fn test_weights_composable_with_zip_pixels() {
        use crate::image::zip_pixels;

        let k1 = Neighborhood::<f32, 3, 3>::new([1.0, 0.0, -1.0, 2.0, 0.0, -2.0, 1.0, 0.0, -1.0]);
        let k2 = Neighborhood::<f32, 3, 3>::new([1.0; 9]);

        let dot: f32 = zip_pixels(k1.weights(), k2.weights())
            .unwrap()
            .map(|(a, b)| a * b)
            .sum();
        assert_eq!(dot, 0.0);
    }

    // ── Non-square kernels ──────────────────────────────────────────

    #[test]
    fn test_non_square_5x3() {
        let k = Neighborhood::<f32, 5, 3>::new([1.0; 15]);
        assert_eq!(k.anchor(), (2, 1));
        assert_eq!(k.kernel_width(), 5);
        assert_eq!(k.kernel_height(), 3);
        assert_eq!(k.positions().len(), 15);
    }

    #[test]
    fn test_non_square_3x5() {
        let k = Neighborhood::<f32, 3, 5>::new([1.0; 15]);
        assert_eq!(k.anchor(), (1, 2));
        assert_eq!(k.kernel_width(), 3);
        assert_eq!(k.kernel_height(), 5);
        assert_eq!(k.positions().len(), 15);
    }

    // ── Edge case: u8 weights ───────────────────────────────────────

    #[test]
    fn test_u8_weight_neighborhood() {
        let k = Neighborhood::<u8, 3, 3>::new([0, 1, 0, 1, 1, 1, 0, 1, 0]);
        let active: Vec<_> = k.positions().filter(|&(_, _, w)| w > 0).collect();
        assert_eq!(active.len(), 5);
    }

    // ── Separable decomposition check ───────────────────────────────

    #[test]
    fn test_separable_gaussian_product() {
        // A separable Gaussian can be reconstructed from outer product
        // of 1D kernels: [1,2,1]^T * [1,2,1] = [1,2,1; 2,4,2; 1,2,1]
        let h = Neighborhood::gaussian_1d_3_h();
        let v = Neighborhood::gaussian_1d_3_v();
        let full = Neighborhood::gaussian_3x3();

        for ky in 0..3usize {
            for kx in 0..3usize {
                let expected = v.weights().pixel_at(0, ky) * h.weights().pixel_at(kx, 0);
                assert_eq!(
                    full.weights().pixel_at(kx, ky),
                    expected,
                    "mismatch at ({}, {})",
                    kx,
                    ky,
                );
            }
        }
    }

    // ── Sobel weight verification ───────────────────────────────────

    #[test]
    fn test_sobel_x_weights() {
        let k = Neighborhood::sobel_x();
        let expected = [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];
        assert_eq!(k.as_slice(), &expected);
    }

    #[test]
    fn test_sobel_y_weights() {
        let k = Neighborhood::sobel_y();
        let expected = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
        assert_eq!(k.as_slice(), &expected);
    }

    // ── i32 matches f32 kernels ─────────────────────────────────────

    #[test]
    fn test_sobel_i32_matches_f32() {
        let fi = Neighborhood::sobel_x();
        let ii = Neighborhood::sobel_x_i32();
        for (f, i) in fi.as_slice().iter().zip(ii.as_slice().iter()) {
            assert_eq!(*f as i32, *i);
        }
    }

    #[test]
    fn test_scharr_i32_matches_f32() {
        let fi = Neighborhood::scharr_x();
        let ii = Neighborhood::scharr_x_i32();
        for (f, i) in fi.as_slice().iter().zip(ii.as_slice().iter()) {
            assert_eq!(*f as i32, *i);
        }
    }

    // ── Kernel trait tests ──────────────────────────────────────────────

    #[test]
    fn test_flipped_identity_3x3_is_identity() {
        let kernel = Neighborhood::<f32, 3, 3>::identity_3x3();
        let flipped = kernel.flipped();
        assert_eq!(kernel.as_slice(), flipped.as_slice());
        assert_eq!(kernel.anchor(), flipped.anchor());
    }

    #[test]
    fn test_flipped_involution_f32() {
        let kernel = Neighborhood::<f32, 3, 3>::sobel_x();
        let double_flipped = kernel.flipped().flipped();
        assert_eq!(kernel.as_slice(), double_flipped.as_slice());
        assert_eq!(kernel.anchor(), double_flipped.anchor());
    }

    #[test]
    fn test_flipped_involution_bool() {
        let kernel = Neighborhood::<bool, 3, 3>::cross_3x3();
        let double_flipped = kernel.flipped().flipped();
        assert_eq!(kernel.as_slice(), double_flipped.as_slice());
        assert_eq!(kernel.anchor(), double_flipped.anchor());
    }

    #[test]
    fn test_flipped_involution_i32() {
        let kernel = Neighborhood::<i32, 3, 3>::sobel_x_i32();
        let double_flipped = kernel.flipped().flipped();
        assert_eq!(kernel.as_slice(), double_flipped.as_slice());
        assert_eq!(kernel.anchor(), double_flipped.anchor());
    }

    #[test]
    fn test_flipped_sobel_x_content() {
        // sobel_x = [-1, -2, -1, 0, 0, 0, 1, 2, 1]
        // 180°-rotated = [1, 2, 1, 0, 0, 0, -1, -2, -1]
        let kernel = Neighborhood::<f32, 3, 3>::sobel_x();
        let flipped = kernel.flipped();
        let expected = [1.0, 2.0, 1.0, 0.0, 0.0, 0.0, -1.0, -2.0, -1.0f32];
        for (a, b) in flipped.as_slice().iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6, "got {a}, expected {b}");
        }
    }

    #[test]
    fn test_flipped_non_centered_anchor() {
        let kernel = Neighborhood::<f32, 3, 3>::with_anchor([1.0; 9], (0, 0));
        let flipped = kernel.flipped();
        assert_eq!(flipped.anchor(), (2, 2));

        let kernel2 = Neighborhood::<f32, 5, 3>::with_anchor([1.0; 15], (1, 0));
        let flipped2 = kernel2.flipped();
        assert_eq!(flipped2.anchor(), (3, 2));
    }

    #[test]
    fn test_flipped_centered_anchor_stays_centered() {
        let kernel = Neighborhood::<f32, 3, 3>::box_blur_3x3();
        let flipped = kernel.flipped();
        // 3x3 with center anchor (1,1) → flipped anchor = (3-1-1, 3-1-1) = (1,1)
        assert_eq!(flipped.anchor(), (1, 1));

        let kernel5 = Neighborhood::<f32, 5, 5>::box_blur_5x5();
        let flipped5 = kernel5.flipped();
        assert_eq!(flipped5.anchor(), (2, 2));
    }

    #[test]
    fn test_flipped_symmetric_kernel_unchanged() {
        let kernel = Neighborhood::<f32, 3, 3>::box_blur_3x3();
        let flipped = kernel.flipped();
        // Box blur is symmetric, so flipping should not change it
        for (a, b) in kernel.as_slice().iter().zip(flipped.as_slice().iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_flipped_1d_horizontal() {
        let kernel = Neighborhood::<f32, 3, 1>::new([1.0, 2.0, 3.0]);
        let flipped = kernel.flipped();
        assert_eq!(flipped.as_slice(), &[3.0, 2.0, 1.0]);
        assert_eq!(flipped.anchor(), (1, 0)); // centered stays centered
    }

    #[test]
    fn test_flipped_1d_vertical() {
        let kernel = Neighborhood::<f32, 1, 3>::new([1.0, 2.0, 3.0]);
        let flipped = kernel.flipped();
        assert_eq!(flipped.as_slice(), &[3.0, 2.0, 1.0]);
        assert_eq!(flipped.anchor(), (0, 1));
    }

    #[test]
    fn test_flipped_bool_full_rect() {
        let kernel = Neighborhood::<bool, 3, 3>::full_rect_3x3();
        let flipped = kernel.flipped();
        // All true, so flipping should be the same
        assert_eq!(kernel.as_slice(), flipped.as_slice());
    }

    #[test]
    fn test_kernel_trait_weights_matches_method() {
        use crate::image::Kernel;
        let kernel = Neighborhood::<f32, 3, 3>::gaussian_3x3();
        // The Kernel::weights() and Neighborhood::weights() should return the same thing
        let trait_weights: &ImageArray<f32, 3, 3> = Kernel::weights(&kernel);
        let method_weights = Neighborhood::weights(&kernel);
        assert_eq!(trait_weights.as_slice(), method_weights.as_slice());
    }

    #[test]
    fn test_kernel_trait_anchor_matches_method() {
        use crate::image::Kernel;
        let kernel = Neighborhood::<f32, 3, 3>::with_anchor([1.0; 9], (2, 0));
        assert_eq!(Kernel::anchor(&kernel), Neighborhood::anchor(&kernel));
    }
}

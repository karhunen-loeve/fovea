//! Separable kernel: a pair of 1D weight arrays (horizontal + vertical)
//! that together define a 2D convolution kernel via their outer product.
//!
//! A [`SeparableKernel`] bundles both 1D weight vectors and their anchors
//! into a single value, eliminating the error-prone pattern of creating
//! and managing two separate [`Neighborhood`](crate::image::Neighborhood) values.
//!
//! # Why a struct, not a trait?
//!
//! There is exactly one representation: two 1D weight arrays + two
//! anchors. No meaningful alternative implementations exist. The
//! `convolve_separable` engine needs to build `ImageView`-compatible
//! shapes from the 1D data, and a concrete struct with const generics
//! makes this trivial — sizes are known at compile time.
//!
//! # Example
//!
//! ```
//! use fovea::image::SeparableKernel;
//!
//! // Symmetric 3×3 box blur: [1/3, 1/3, 1/3] in both directions
//! let kernel = SeparableKernel::<3, 3>::box_blur_3();
//! assert_eq!(kernel.h_weights(), &[1.0 / 3.0; 3]);
//! assert_eq!(kernel.v_weights(), &[1.0 / 3.0; 3]);
//! assert_eq!(kernel.h_anchor(), 1);
//! assert_eq!(kernel.v_anchor(), 1);
//! ```

use crate::image::Image;

/// A separable convolution kernel: two 1D weight arrays (horizontal and
/// vertical) plus their anchor positions.
///
/// The effective 2D kernel is the outer product of the two 1D arrays.
/// Separable convolution applies the horizontal pass first, then the
/// vertical pass, reducing per-pixel work from O(HK × VK) to O(HK + VK).
///
/// Both weight arrays and anchors are stored inline (no heap allocation).
/// `flipped()` returns a new `SeparableKernel` with reversed arrays and
/// mirrored anchors — entirely on the stack.
///
/// # Type Parameters
///
/// - `HK` — length of the horizontal 1D kernel
/// - `VK` — length of the vertical 1D kernel
///
/// # Example
///
/// ```
/// use fovea::image::SeparableKernel;
///
/// let kernel = SeparableKernel::symmetric([1.0, 2.0, 1.0]);
/// assert_eq!(kernel.h_anchor(), 1);
/// assert_eq!(kernel.v_anchor(), 1);
///
/// let flipped = kernel.flipped();
/// // [1,2,1] is symmetric, so flipping is a no-op
/// assert_eq!(flipped.h_weights(), kernel.h_weights());
/// assert_eq!(flipped.v_weights(), kernel.v_weights());
/// ```
///
/// Zero-sized kernels are rejected at compile time:
///
/// ```compile_fail
/// use fovea::image::SeparableKernel;
/// // SeparableKernel<0, _> and SeparableKernel<_, 0> would underflow
/// // `HK - 1` / `VK - 1` in `flipped()` and the convolution passes.
/// let _ = SeparableKernel::<0, 3>::new([], [1.0, 2.0, 1.0]);
/// ```
#[derive(Clone, Debug)]
pub struct SeparableKernel<const HK: usize, const VK: usize> {
    h_weights: [f32; HK],
    h_anchor: usize,
    v_weights: [f32; VK],
    v_anchor: usize,
}

impl<const HK: usize, const VK: usize> SeparableKernel<HK, VK> {
    /// Compile-time assertion: separable kernels must have non-zero
    /// horizontal and vertical dimensions.
    ///
    /// `flipped()`, `convolve_separable_into`, and the various row/column
    /// passes all index relative to `HK - 1` / `VK - 1`. A zero dimension
    /// underflows that subtraction. Forcing this assertion through every
    /// constructor (`new`, `with_anchors`, `symmetric`) means any attempt
    /// to construct `SeparableKernel<0, _>` or `SeparableKernel<_, 0>` is
    /// rejected at compile time.
    const _ASSERT_NONZERO: () = {
        assert!(
            HK > 0,
            "SeparableKernel: horizontal kernel length HK must be > 0"
        );
        assert!(
            VK > 0,
            "SeparableKernel: vertical kernel length VK must be > 0"
        );
    };

    /// Creates a separable kernel with explicit weights and centered
    /// anchors (`HK / 2` and `VK / 2`).
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::image::SeparableKernel;
    ///
    /// let k = SeparableKernel::new([1.0, 2.0, 1.0], [1.0, 4.0, 6.0, 4.0, 1.0]);
    /// assert_eq!(k.h_anchor(), 1); // 3 / 2
    /// assert_eq!(k.v_anchor(), 2); // 5 / 2
    /// ```
    pub fn new(h_weights: [f32; HK], v_weights: [f32; VK]) -> Self {
        let () = Self::_ASSERT_NONZERO;
        Self {
            h_weights,
            h_anchor: HK / 2,
            v_weights,
            v_anchor: VK / 2,
        }
    }

    /// Creates a separable kernel with explicit weights and explicit
    /// anchor positions.
    ///
    /// # Panics
    ///
    /// Panics if `h_anchor >= HK` or `v_anchor >= VK`.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::image::SeparableKernel;
    ///
    /// let k = SeparableKernel::with_anchors(
    ///     [1.0, 0.0, 0.0], 0,
    ///     [0.0, 0.0, 1.0], 2,
    /// );
    /// assert_eq!(k.h_anchor(), 0);
    /// assert_eq!(k.v_anchor(), 2);
    /// ```
    pub fn with_anchors(
        h_weights: [f32; HK],
        h_anchor: usize,
        v_weights: [f32; VK],
        v_anchor: usize,
    ) -> Self {
        let () = Self::_ASSERT_NONZERO;
        assert!(
            h_anchor < HK,
            "h_anchor ({h_anchor}) out of bounds for horizontal kernel of size {HK}"
        );
        assert!(
            v_anchor < VK,
            "v_anchor ({v_anchor}) out of bounds for vertical kernel of size {VK}"
        );
        Self {
            h_weights,
            h_anchor,
            v_weights,
            v_anchor,
        }
    }

    /// Returns the horizontal 1D weight array.
    pub fn h_weights(&self) -> &[f32; HK] {
        &self.h_weights
    }

    /// Returns the vertical 1D weight array.
    pub fn v_weights(&self) -> &[f32; VK] {
        &self.v_weights
    }

    /// Returns the horizontal anchor position.
    pub fn h_anchor(&self) -> usize {
        self.h_anchor
    }

    /// Returns the vertical anchor position.
    pub fn v_anchor(&self) -> usize {
        self.v_anchor
    }

    /// Returns a 180°-rotated copy of this separable kernel.
    ///
    /// Both 1D weight arrays are reversed and both anchors are mirrored.
    /// This is entirely stack-based — zero heap allocation.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::image::SeparableKernel;
    ///
    /// let k = SeparableKernel::with_anchors(
    ///     [1.0, 2.0, 3.0], 0,
    ///     [4.0, 5.0], 0,
    /// );
    /// let f = k.flipped();
    /// assert_eq!(f.h_weights(), &[3.0, 2.0, 1.0]);
    /// assert_eq!(f.v_weights(), &[5.0, 4.0]);
    /// assert_eq!(f.h_anchor(), 2);
    /// assert_eq!(f.v_anchor(), 1);
    /// ```
    pub fn flipped(&self) -> Self {
        let mut h = self.h_weights;
        h.reverse();
        let mut v = self.v_weights;
        v.reverse();
        Self {
            h_weights: h,
            h_anchor: HK - 1 - self.h_anchor,
            v_weights: v,
            v_anchor: VK - 1 - self.v_anchor,
        }
    }

    /// Returns the horizontal weights as a heap-allocated `Image<f32>`
    /// with shape `(HK, 1)`.
    ///
    /// This is used by the `convolve_separable` functions to pass
    /// weights to `fold_neighborhood` without requiring private trait
    /// bounds in the public API.
    pub(crate) fn to_h_image(&self) -> Image<f32> {
        Image::generate(HK, 1, |x, _y| self.h_weights[x])
    }

    /// Returns the vertical weights as a heap-allocated `Image<f32>`
    /// with shape `(1, VK)`.
    pub(crate) fn to_v_image(&self) -> Image<f32> {
        Image::generate(1, VK, |_x, y| self.v_weights[y])
    }
}

// ─── Symmetric constructors (HK == VK) ─────────────────────────────────

impl<const K: usize> SeparableKernel<K, K> {
    /// Creates a symmetric separable kernel where h and v share the same
    /// 1D weights and centered anchors.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::image::SeparableKernel;
    ///
    /// let k = SeparableKernel::symmetric([1.0, 2.0, 1.0]);
    /// assert_eq!(k.h_weights(), k.v_weights());
    /// assert_eq!(k.h_anchor(), k.v_anchor());
    /// ```
    pub fn symmetric(weights: [f32; K]) -> Self {
        let () = Self::_ASSERT_NONZERO;
        Self {
            h_weights: weights,
            h_anchor: K / 2,
            v_weights: weights,
            v_anchor: K / 2,
        }
    }
}

// ─── Factory methods: 3×3 ───────────────────────────────────────────────

impl SeparableKernel<3, 3> {
    /// 3×3 Gaussian kernel: `[1, 2, 1]` in both directions.
    ///
    /// This is the **un-normalised** separable Gaussian. The combined 2D
    /// kernel sums to 16 (matching [`Neighborhood::gaussian_3x3`](crate::image::Neighborhood::gaussian_3x3)).
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::image::SeparableKernel;
    ///
    /// let k = SeparableKernel::gaussian_3();
    /// assert_eq!(k.h_weights(), &[1.0, 2.0, 1.0]);
    /// assert_eq!(k.v_weights(), &[1.0, 2.0, 1.0]);
    /// ```
    pub fn gaussian_3() -> Self {
        Self::symmetric([1.0, 2.0, 1.0])
    }

    /// 3×3 box blur kernel: `[1/3, 1/3, 1/3]` in both directions.
    ///
    /// The combined 2D kernel averages over a 3×3 window (each weight
    /// is 1/9), matching [`Neighborhood::box_blur_3x3`](crate::image::Neighborhood::box_blur_3x3).
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::image::SeparableKernel;
    ///
    /// let k = SeparableKernel::box_blur_3();
    /// let third = 1.0f32 / 3.0;
    /// assert_eq!(k.h_weights(), &[third, third, third]);
    /// ```
    pub fn box_blur_3() -> Self {
        let w = 1.0 / 3.0;
        Self::symmetric([w, w, w])
    }
}

// ─── Factory methods: 5×5 ───────────────────────────────────────────────

impl SeparableKernel<5, 5> {
    /// 5×5 Gaussian kernel: `[1, 4, 6, 4, 1]` in both directions.
    ///
    /// This is the **un-normalised** separable Gaussian. The combined 2D
    /// kernel sums to 256 (matching [`Neighborhood::gaussian_5x5`](crate::image::Neighborhood::gaussian_5x5)).
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::image::SeparableKernel;
    ///
    /// let k = SeparableKernel::gaussian_5();
    /// assert_eq!(k.h_weights(), &[1.0, 4.0, 6.0, 4.0, 1.0]);
    /// ```
    pub fn gaussian_5() -> Self {
        Self::symmetric([1.0, 4.0, 6.0, 4.0, 1.0])
    }

    /// 5×5 box blur kernel: `[1/5, 1/5, 1/5, 1/5, 1/5]` in both
    /// directions.
    ///
    /// The combined 2D kernel averages over a 5×5 window (each weight
    /// is 1/25), matching [`Neighborhood::box_blur_5x5`](crate::image::Neighborhood::box_blur_5x5).
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::image::SeparableKernel;
    ///
    /// let k = SeparableKernel::box_blur_5();
    /// let fifth = 1.0f32 / 5.0;
    /// assert_eq!(k.h_weights(), &[fifth; 5]);
    /// ```
    pub fn box_blur_5() -> Self {
        let w = 1.0 / 5.0;
        Self::symmetric([w, w, w, w, w])
    }
}

// ─── PartialEq ──────────────────────────────────────────────────────────

impl<const HK: usize, const VK: usize> PartialEq for SeparableKernel<HK, VK> {
    fn eq(&self, other: &Self) -> bool {
        self.h_anchor == other.h_anchor
            && self.v_anchor == other.v_anchor
            && self.h_weights == other.h_weights
            && self.v_weights == other.v_weights
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::ImageView;

    // ── constructors ────────────────────────────────────────────────────

    #[test]
    fn new_centered_anchors() {
        let k = SeparableKernel::new([1.0, 2.0, 1.0], [1.0, 4.0, 6.0, 4.0, 1.0]);
        assert_eq!(k.h_anchor(), 1); // 3 / 2
        assert_eq!(k.v_anchor(), 2); // 5 / 2
        assert_eq!(k.h_weights(), &[1.0, 2.0, 1.0]);
        assert_eq!(k.v_weights(), &[1.0, 4.0, 6.0, 4.0, 1.0]);
    }

    #[test]
    fn new_even_sizes_center_left() {
        let k = SeparableKernel::new([1.0; 4], [1.0; 2]);
        assert_eq!(k.h_anchor(), 2); // 4 / 2
        assert_eq!(k.v_anchor(), 1); // 2 / 2
    }

    #[test]
    fn with_anchors_explicit() {
        let k = SeparableKernel::with_anchors([1.0, 2.0, 3.0], 0, [4.0, 5.0], 1);
        assert_eq!(k.h_anchor(), 0);
        assert_eq!(k.v_anchor(), 1);
        assert_eq!(k.h_weights(), &[1.0, 2.0, 3.0]);
        assert_eq!(k.v_weights(), &[4.0, 5.0]);
    }

    #[test]
    #[should_panic(expected = "h_anchor")]
    fn with_anchors_h_out_of_bounds() {
        SeparableKernel::with_anchors([1.0, 2.0, 3.0], 3, [1.0], 0);
    }

    #[test]
    #[should_panic(expected = "v_anchor")]
    fn with_anchors_v_out_of_bounds() {
        SeparableKernel::with_anchors([1.0], 0, [1.0, 2.0], 2);
    }

    #[test]
    fn symmetric_constructor() {
        let k = SeparableKernel::symmetric([1.0, 2.0, 1.0]);
        assert_eq!(k.h_weights(), k.v_weights());
        assert_eq!(k.h_anchor(), k.v_anchor());
        assert_eq!(k.h_anchor(), 1);
    }

    // ── flipped ─────────────────────────────────────────────────────────

    #[test]
    fn flipped_reverses_weights() {
        let k = SeparableKernel::new([1.0, 2.0, 3.0], [4.0, 5.0]);
        let f = k.flipped();
        assert_eq!(f.h_weights(), &[3.0, 2.0, 1.0]);
        assert_eq!(f.v_weights(), &[5.0, 4.0]);
    }

    #[test]
    fn flipped_mirrors_anchors() {
        let k = SeparableKernel::with_anchors([1.0, 2.0, 3.0], 0, [4.0, 5.0, 6.0], 2);
        let f = k.flipped();
        assert_eq!(f.h_anchor(), 2); // 3 - 1 - 0
        assert_eq!(f.v_anchor(), 0); // 3 - 1 - 2
    }

    #[test]
    fn flipped_centered_anchor_stays_centered() {
        let k = SeparableKernel::gaussian_3();
        let f = k.flipped();
        assert_eq!(f.h_anchor(), 1);
        assert_eq!(f.v_anchor(), 1);
    }

    #[test]
    fn flipped_involution() {
        let k = SeparableKernel::with_anchors([1.0, 2.0, 3.0], 0, [4.0, 5.0], 1);
        let ff = k.flipped().flipped();
        assert_eq!(k, ff);
    }

    #[test]
    fn flipped_symmetric_kernel_unchanged() {
        let k = SeparableKernel::box_blur_3();
        let f = k.flipped();
        assert_eq!(k, f);
    }

    #[test]
    fn flipped_gaussian_5_symmetric() {
        let k = SeparableKernel::gaussian_5();
        let f = k.flipped();
        // [1,4,6,4,1] is symmetric
        assert_eq!(k, f);
    }

    // ── factory methods ─────────────────────────────────────────────────

    #[test]
    fn gaussian_3_weights() {
        let k = SeparableKernel::gaussian_3();
        assert_eq!(k.h_weights(), &[1.0, 2.0, 1.0]);
        assert_eq!(k.v_weights(), &[1.0, 2.0, 1.0]);
        assert_eq!(k.h_anchor(), 1);
        assert_eq!(k.v_anchor(), 1);
    }

    #[test]
    fn gaussian_5_weights() {
        let k = SeparableKernel::gaussian_5();
        assert_eq!(k.h_weights(), &[1.0, 4.0, 6.0, 4.0, 1.0]);
        assert_eq!(k.v_weights(), &[1.0, 4.0, 6.0, 4.0, 1.0]);
        assert_eq!(k.h_anchor(), 2);
        assert_eq!(k.v_anchor(), 2);
    }

    #[test]
    fn box_blur_3_weights() {
        let k = SeparableKernel::box_blur_3();
        let third = 1.0f32 / 3.0;
        for &w in k.h_weights() {
            assert!((w - third).abs() < 1e-7);
        }
        for &w in k.v_weights() {
            assert!((w - third).abs() < 1e-7);
        }
    }

    #[test]
    fn box_blur_5_weights() {
        let k = SeparableKernel::box_blur_5();
        let fifth = 1.0f32 / 5.0;
        for &w in k.h_weights() {
            assert!((w - fifth).abs() < 1e-7);
        }
        for &w in k.v_weights() {
            assert!((w - fifth).abs() < 1e-7);
        }
    }

    // ── outer product matches 2D factory kernels ────────────────────────

    #[test]
    fn gaussian_3_outer_product_matches_neighborhood() {
        let sep = SeparableKernel::gaussian_3();
        let full = crate::image::Neighborhood::<f32, 3, 3>::gaussian_3x3();

        for y in 0..3 {
            for x in 0..3 {
                let outer = sep.h_weights()[x] * sep.v_weights()[y];
                let expected = full.weights().pixel_at(x, y);
                assert!(
                    (outer - expected).abs() < 1e-6,
                    "mismatch at ({x}, {y}): outer={outer}, expected={expected}"
                );
            }
        }
    }

    #[test]
    fn gaussian_5_outer_product_matches_neighborhood() {
        let sep = SeparableKernel::gaussian_5();
        let full = crate::image::Neighborhood::<f32, 5, 5>::gaussian_5x5();

        for y in 0..5 {
            for x in 0..5 {
                let outer = sep.h_weights()[x] * sep.v_weights()[y];
                let expected = full.weights().pixel_at(x, y);
                assert!(
                    (outer - expected).abs() < 1e-4,
                    "mismatch at ({x}, {y}): outer={outer}, expected={expected}"
                );
            }
        }
    }

    #[test]
    fn box_blur_3_outer_product_matches_neighborhood() {
        let sep = SeparableKernel::box_blur_3();
        let full = crate::image::Neighborhood::<f32, 3, 3>::box_blur_3x3();

        for y in 0..3 {
            for x in 0..3 {
                let outer = sep.h_weights()[x] * sep.v_weights()[y];
                let expected = full.weights().pixel_at(x, y);
                assert!(
                    (outer - expected).abs() < 1e-6,
                    "mismatch at ({x}, {y}): outer={outer}, expected={expected}"
                );
            }
        }
    }

    #[test]
    fn box_blur_5_outer_product_matches_neighborhood() {
        let sep = SeparableKernel::box_blur_5();
        let full = crate::image::Neighborhood::<f32, 5, 5>::box_blur_5x5();

        for y in 0..5 {
            for x in 0..5 {
                let outer = sep.h_weights()[x] * sep.v_weights()[y];
                let expected = full.weights().pixel_at(x, y);
                assert!(
                    (outer - expected).abs() < 1e-6,
                    "mismatch at ({x}, {y}): outer={outer}, expected={expected}"
                );
            }
        }
    }

    // ── to_h_image / to_v_image helpers ─────────────────────────────────

    #[test]
    fn to_h_image_shape_and_content() {
        let k = SeparableKernel::gaussian_3();
        let arr = k.to_h_image();
        assert_eq!(arr.width(), 3);
        assert_eq!(arr.height(), 1);
        assert_eq!(arr.pixel_at(0, 0), 1.0);
        assert_eq!(arr.pixel_at(1, 0), 2.0);
        assert_eq!(arr.pixel_at(2, 0), 1.0);
    }

    #[test]
    fn to_v_image_shape_and_content() {
        let k = SeparableKernel::gaussian_3();
        let arr = k.to_v_image();
        assert_eq!(arr.width(), 1);
        assert_eq!(arr.height(), 3);
        assert_eq!(arr.pixel_at(0, 0), 1.0);
        assert_eq!(arr.pixel_at(0, 1), 2.0);
        assert_eq!(arr.pixel_at(0, 2), 1.0);
    }

    #[test]
    fn to_h_image_5() {
        let k = SeparableKernel::gaussian_5();
        let arr = k.to_h_image();
        assert_eq!(arr.width(), 5);
        assert_eq!(arr.height(), 1);
        assert_eq!(arr.pixel_at(0, 0), 1.0);
        assert_eq!(arr.pixel_at(1, 0), 4.0);
        assert_eq!(arr.pixel_at(2, 0), 6.0);
        assert_eq!(arr.pixel_at(3, 0), 4.0);
        assert_eq!(arr.pixel_at(4, 0), 1.0);
    }

    #[test]
    fn to_v_image_5() {
        let k = SeparableKernel::gaussian_5();
        let arr = k.to_v_image();
        assert_eq!(arr.width(), 1);
        assert_eq!(arr.height(), 5);
        assert_eq!(arr.pixel_at(0, 0), 1.0);
        assert_eq!(arr.pixel_at(0, 1), 4.0);
        assert_eq!(arr.pixel_at(0, 2), 6.0);
        assert_eq!(arr.pixel_at(0, 3), 4.0);
        assert_eq!(arr.pixel_at(0, 4), 1.0);
    }

    // ── Clone / Debug / PartialEq ───────────────────────────────────────

    #[test]
    fn clone_produces_equal_kernel() {
        let k = SeparableKernel::with_anchors([1.0, 2.0, 3.0], 0, [4.0, 5.0], 1);
        let c = k.clone();
        assert_eq!(k, c);
    }

    #[test]
    fn debug_format_contains_weights() {
        let k = SeparableKernel::new([1.0, 2.0], [3.0]);
        let dbg = format!("{k:?}");
        assert!(dbg.contains("SeparableKernel"));
        assert!(dbg.contains("h_weights"));
        assert!(dbg.contains("v_weights"));
    }

    #[test]
    fn partial_eq_different_weights() {
        let a = SeparableKernel::new([1.0, 2.0, 3.0], [1.0]);
        let b = SeparableKernel::new([3.0, 2.0, 1.0], [1.0]);
        assert_ne!(a, b);
    }

    #[test]
    fn partial_eq_different_anchors() {
        let a = SeparableKernel::with_anchors([1.0, 2.0, 3.0], 0, [1.0], 0);
        let b = SeparableKernel::with_anchors([1.0, 2.0, 3.0], 2, [1.0], 0);
        assert_ne!(a, b);
    }

    // ── non-square kernels ──────────────────────────────────────────────

    #[test]
    fn asymmetric_3x5() {
        let k = SeparableKernel::new([1.0, 2.0, 1.0], [1.0, 4.0, 6.0, 4.0, 1.0]);
        assert_eq!(k.h_anchor(), 1);
        assert_eq!(k.v_anchor(), 2);

        let f = k.flipped();
        // [1,2,1] reversed = [1,2,1] (symmetric)
        assert_eq!(f.h_weights(), &[1.0, 2.0, 1.0]);
        // [1,4,6,4,1] reversed = [1,4,6,4,1] (symmetric)
        assert_eq!(f.v_weights(), &[1.0, 4.0, 6.0, 4.0, 1.0]);
    }

    #[test]
    fn asymmetric_weights_flip() {
        let k = SeparableKernel::new([1.0, 0.0, 0.0], [0.0, 1.0]);
        let f = k.flipped();
        assert_eq!(f.h_weights(), &[0.0, 0.0, 1.0]);
        assert_eq!(f.v_weights(), &[1.0, 0.0]);
    }

    // ── 1×1 degenerate case ─────────────────────────────────────────────

    #[test]
    fn identity_1x1() {
        let k = SeparableKernel::new([1.0], [1.0]);
        assert_eq!(k.h_anchor(), 0);
        assert_eq!(k.v_anchor(), 0);

        let f = k.flipped();
        assert_eq!(f, k);
    }
}

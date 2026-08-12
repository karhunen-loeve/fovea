//! Separable kernel: a pair of 1D weight arrays (horizontal + vertical)
//! that together define a 2D convolution kernel via their outer product.
//!
//! A [`SeparableKernel`] bundles both 1D weight vectors and their anchors
//! into a single value, eliminating the error-prone pattern of creating
//! and managing two separate [`Neighborhood`](crate::image::Neighborhood) values.
//!
//! # Two representations, one trait
//!
//! [`SeparableKernel`] stores its lengths as **const generics**, so a fixed
//! 3-tap or 5-tap kernel is checked and unrolled at compile time.
//! [`GaussianKernel1D`] — produced by [`gaussian_kernel_1d`] from a σ and a
//! `truncate` — has a **runtime** tap count in a fixed stack buffer, because
//! the radius follows from σ.
//!
//! [`SeparableWeights`] is what the separable engine actually consumes, and
//! both types implement it. That is why there is no separate blur function
//! per kernel flavour: the kernel *is* the variant, and
//! `convolve_separable(&src, &kernel, &border)` takes either one.
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

use crate::Sigma;

/// A pair of 1-D weight arrays plus anchors — everything the separable
/// convolution engine needs from a kernel.
///
/// Implemented by [`SeparableKernel`] (compile-time tap counts) and
/// [`GaussianKernel1D`] (σ-derived tap count, symmetric). Every separable
/// entry point — [`convolve_separable`](crate::transform::convolve_separable),
/// [`convolve_separable_into`](crate::transform::convolve_separable_into) and
/// [`SeparableScratch`](crate::transform::SeparableScratch)'s methods — is
/// generic over this trait, so a caller selects the variant by choosing a
/// **value**, not by choosing a differently-named function.
///
/// # Contract
///
/// - Weight slices are non-empty. The engine indexes relative to
///   `len() - 1`, so an empty axis would underflow.
/// - Anchors are in bounds: `h_anchor() < h_weights().len()`, likewise for
///   the vertical axis.
/// - [`flipped`](Self::flipped) reverses both axes and mirrors both anchors
///   (`len - 1 - anchor`). It must not allocate — every implementor in this
///   crate returns a stack value, and a symmetric kernel may return a copy of
///   itself unchanged.
///
/// # Implementing it
///
/// ```
/// use fovea::image::SeparableWeights;
///
/// /// A 3-tap sharpening kernel applied along both axes.
/// #[derive(Clone, Copy)]
/// struct Sharpen3;
///
/// impl SeparableWeights for Sharpen3 {
///     fn h_weights(&self) -> &[f32] { &[-1.0, 3.0, -1.0] }
///     fn h_anchor(&self) -> usize { 1 }
///     fn v_weights(&self) -> &[f32] { &[-1.0, 3.0, -1.0] }
///     fn v_anchor(&self) -> usize { 1 }
///     fn flipped(&self) -> Self { *self } // palindrome ⇒ flip is a no-op
/// }
/// ```
pub trait SeparableWeights: Sized {
    /// The horizontal 1-D weights, applied left to right. Never empty.
    fn h_weights(&self) -> &[f32];

    /// The horizontal anchor: the tap index that lands on the output pixel.
    fn h_anchor(&self) -> usize;

    /// The vertical 1-D weights, applied top to bottom. Never empty.
    fn v_weights(&self) -> &[f32];

    /// The vertical anchor: the tap index that lands on the output pixel.
    fn v_anchor(&self) -> usize;

    /// The 180°-rotated kernel: both axes reversed, both anchors mirrored to
    /// `len - 1 - anchor`.
    ///
    /// Convolution is correlation with the flipped kernel, which is the only
    /// reason this is on the trait. Implementations must stay on the stack;
    /// for a symmetric kernel the honest implementation is `*self`.
    #[must_use]
    fn flipped(&self) -> Self;
}

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
}

impl<const HK: usize, const VK: usize> SeparableWeights for SeparableKernel<HK, VK> {
    #[inline]
    fn h_weights(&self) -> &[f32] {
        &self.h_weights
    }

    #[inline]
    fn h_anchor(&self) -> usize {
        self.h_anchor
    }

    #[inline]
    fn v_weights(&self) -> &[f32] {
        &self.v_weights
    }

    #[inline]
    fn v_anchor(&self) -> usize {
        self.v_anchor
    }

    /// Delegates to the inherent [`flipped`](Self::flipped) — reversed stack
    /// arrays, mirrored anchors, no allocation.
    #[inline]
    fn flipped(&self) -> Self {
        SeparableKernel::flipped(self)
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
    /// 3×3 Gaussian kernel: the **normalized** `[1, 2, 1] / 4` weights
    /// `[0.25, 0.5, 0.25]` in both directions.
    ///
    /// Each 1D pass sums to 1, so the combined 2D kernel sums to 1 and the
    /// blur **preserves brightness** — consistent with
    /// [`SeparableKernel::box_blur_3`]. The raw integer `[1, 2, 1]` kernel
    /// (sum 16) lives at the lower layer as
    /// [`Neighborhood::gaussian_3x3`](crate::image::Neighborhood::gaussian_3x3)
    /// / [`Neighborhood::gaussian_1d_3_h`](crate::image::Neighborhood::gaussian_1d_3_h),
    /// where the caller owns the scale.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::image::SeparableKernel;
    ///
    /// let k = SeparableKernel::gaussian_3();
    /// assert_eq!(k.h_weights(), &[0.25, 0.5, 0.25]);
    /// assert_eq!(k.v_weights(), &[0.25, 0.5, 0.25]);
    /// ```
    pub fn gaussian_3() -> Self {
        Self::symmetric([0.25, 0.5, 0.25])
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
    /// 5×5 Gaussian kernel: the **normalized** `[1, 4, 6, 4, 1] / 16`
    /// weights `[0.0625, 0.25, 0.375, 0.25, 0.0625]` in both directions.
    ///
    /// Each 1D pass sums to 1, so the combined 2D kernel sums to 1 and the
    /// blur **preserves brightness** — consistent with
    /// [`SeparableKernel::box_blur_5`]. The raw integer `[1, 4, 6, 4, 1]`
    /// kernel (sum 256) lives at the lower layer as
    /// [`Neighborhood::gaussian_5x5`](crate::image::Neighborhood::gaussian_5x5)
    /// / [`Neighborhood::gaussian_1d_5_h`](crate::image::Neighborhood::gaussian_1d_5_h),
    /// where the caller owns the scale.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::image::SeparableKernel;
    ///
    /// let k = SeparableKernel::gaussian_5();
    /// assert_eq!(k.h_weights(), &[0.0625, 0.25, 0.375, 0.25, 0.0625]);
    /// ```
    pub fn gaussian_5() -> Self {
        Self::symmetric([0.0625, 0.25, 0.375, 0.25, 0.0625])
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

// ─── Parameterized Gaussian kernel ──────────────────────────────────────

/// Maximum supported radius for a [`gaussian_kernel_1d`] kernel.
///
/// The generated 1-D kernel lives in a fixed stack buffer of
/// `2 * MAX_RADIUS + 1` taps (129 weights, ~516 bytes), so generation is
/// allocation-free. A `sigma` whose derived radius exceeds this bound is a
/// caller precondition violation and panics. At the default `truncate` of
/// [`DEFAULT_TRUNCATE`](crate::transform::DEFAULT_TRUNCATE) (4.0) this
/// supports `sigma` up to `MAX_RADIUS / truncate` ≈ 16.
pub const MAX_RADIUS: usize = 64;

/// Length of the stack buffer backing [`GaussianKernel1D`].
const MAX_GAUSSIAN_TAPS: usize = 2 * MAX_RADIUS + 1;

/// A normalized 1-D Gaussian kernel held in a bounded, stack-allocated
/// buffer.
///
/// Produced by [`gaussian_kernel_1d`]. The active weights (the first
/// [`len`](Self::len) taps) sum to 1, so convolving with this kernel
/// preserves brightness. Because a 1-D Gaussian is symmetric, the same
/// kernel is used for both the horizontal and vertical passes of a
/// separable blur, and the anchor sits at the centre tap
/// ([`anchor`](Self::anchor) == [`radius`](Self::radius)).
#[derive(Clone)]
pub struct GaussianKernel1D {
    weights: [f32; MAX_GAUSSIAN_TAPS],
    len: usize,
    anchor: usize,
}

impl GaussianKernel1D {
    /// The active weights, a slice of [`len`](Self::len) taps summing to 1.
    pub fn weights(&self) -> &[f32] {
        &self.weights[..self.len]
    }

    /// The number of active taps (`2 * radius + 1`, always odd, always ≥ 1).
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the kernel has no taps. Always `false` — a Gaussian kernel
    /// has at least the single centre tap — but provided for API
    /// completeness alongside [`len`](Self::len).
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The anchor (centre) tap index, equal to the [`radius`](Self::radius).
    pub fn anchor(&self) -> usize {
        self.anchor
    }

    /// The kernel radius: `(len - 1) / 2`.
    pub fn radius(&self) -> usize {
        self.anchor
    }
}

impl SeparableWeights for GaussianKernel1D {
    #[inline]
    fn h_weights(&self) -> &[f32] {
        self.weights()
    }

    #[inline]
    fn h_anchor(&self) -> usize {
        self.anchor
    }

    /// The same taps as the horizontal axis — a Gaussian is isotropic, so one
    /// weight array serves both passes.
    #[inline]
    fn v_weights(&self) -> &[f32] {
        self.weights()
    }

    #[inline]
    fn v_anchor(&self) -> usize {
        self.anchor
    }

    /// Returns a copy of `self`, because a Gaussian kernel is its own flip:
    /// the taps are a palindrome (`w[i] == w[len - 1 - i]` by construction)
    /// and the anchor is the centre tap, so mirroring it
    /// (`len - 1 - radius == radius`) is also a no-op. Pinned by
    /// `gaussian_kernel_is_its_own_flip`. The clone copies a fixed stack
    /// buffer — no allocation, as the trait requires.
    #[inline]
    fn flipped(&self) -> Self {
        self.clone()
    }
}

impl core::fmt::Debug for GaussianKernel1D {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GaussianKernel1D")
            .field("len", &self.len)
            .field("anchor", &self.anchor)
            .field("weights", &self.weights())
            .finish()
    }
}

/// Computes the radius of a Gaussian kernel for `(sigma, truncate)`.
///
/// Uses the SciPy / scikit-image convention `radius = floor(truncate *
/// sigma + 0.5)` (round-to-nearest), which is what those libraries
/// actually compute and is the parity this kernel targets. For very small
/// `sigma` the radius rounds down to 0, yielding a 1-tap identity kernel.
///
/// The result is **not** clamped to [`MAX_RADIUS`]; enforcing that bound is
/// the caller's job, and only the kernel *builders* do it. See
/// [`gaussian_kernel_size`] for why the size query stays total.
///
/// σ needs no check here — [`Sigma`] carries the finite-and-positive
/// invariant by construction.
///
/// # Panics
///
/// Panics if `truncate <= 0.0` (a structural kernel-shape constant, not a
/// data-derived value).
fn gaussian_radius(sigma: Sigma, truncate: f32) -> usize {
    assert!(
        truncate > 0.0,
        "gaussian kernel: truncate must be > 0.0 (got {truncate})"
    );
    (truncate * sigma.get() + 0.5).floor() as usize
}

/// Returns the odd tap count of the Gaussian kernel derived from `sigma`
/// and `truncate`: `2 * floor(truncate * sigma + 0.5) + 1`.
///
/// This reports the derived size without building the kernel, so callers
/// can size buffers or reason about cost up front — the derived size is
/// surfaced rather than hidden inside the blur.
///
/// # `MAX_RADIUS` is not enforced here
///
/// This is a pure arithmetic query and deliberately does **not** panic when
/// the derived radius exceeds [`MAX_RADIUS`]; that is what makes it usable
/// as an admissibility *check*. [`gaussian_kernel_1d`] and the
/// [`gaussian_blur`](crate::transform::gaussian_blur) family do panic in
/// that case, so a size reported here above `2 * MAX_RADIUS + 1` (= 129)
/// means "this `sigma` is out of range", not "allocate a bigger buffer":
///
/// ```
/// use fovea::Sigma;
/// use fovea::image::{MAX_RADIUS, gaussian_kernel_size};
///
/// let supported = |sigma, truncate| {
///     gaussian_kernel_size(sigma, truncate) <= 2 * MAX_RADIUS + 1
/// };
/// assert!(supported(Sigma::new(16.0), 4.0));
/// assert!(!supported(Sigma::new(20.0), 4.0)); // 161 taps — `gaussian_blur` would panic
/// ```
///
/// # Panics
///
/// Panics if `truncate <= 0.0`.
///
/// # Example
///
/// ```
/// use fovea::Sigma;
/// use fovea::image::gaussian_kernel_size;
///
/// // radius = round(4.0 * 1.0) = 4 → 9 taps
/// assert_eq!(gaussian_kernel_size(Sigma::new(1.0), 4.0), 9);
/// // tiny sigma rounds down to radius 0 → 1 tap (identity)
/// assert_eq!(gaussian_kernel_size(Sigma::new(0.05), 4.0), 1);
/// ```
#[must_use]
pub fn gaussian_kernel_size(sigma: Sigma, truncate: f32) -> usize {
    2 * gaussian_radius(sigma, truncate) + 1
}

/// Builds a normalized 1-D Gaussian kernel for the given `sigma`.
///
/// The radius is `floor(truncate * sigma + 0.5)` and the tap count is
/// `2 * radius + 1`. Weights are `w_i = exp(-(i - radius)^2 / (2 sigma^2))`
/// normalized to sum 1, so convolving with the kernel preserves
/// brightness. The kernel is symmetric, so the same weights serve both the
/// horizontal and vertical passes of a separable blur.
///
/// # Panics
///
/// - `truncate <= 0.0` — a structural kernel-shape constant. (σ carries
///   its finite-and-positive invariant in the [`Sigma`] type; construct
///   computed values with [`Sigma::try_new`].)
/// - radius exceeds [`MAX_RADIUS`] — the kernel would not fit the bounded
///   stack buffer; the message names the largest supported `sigma`. Test
///   admissibility up front with [`gaussian_kernel_size`].
///
/// # Example
///
/// ```
/// use fovea::Sigma;
/// use fovea::image::gaussian_kernel_1d;
///
/// let k = gaussian_kernel_1d(Sigma::new(1.0), 4.0);
/// assert_eq!(k.len(), 9);
/// assert_eq!(k.anchor(), 4);
/// // Normalized: the weights sum to 1.
/// let sum: f32 = k.weights().iter().sum();
/// assert!((sum - 1.0).abs() < 1e-6);
/// // Symmetric about the centre.
/// assert!((k.weights()[0] - k.weights()[8]).abs() < 1e-7);
/// ```
#[must_use]
pub fn gaussian_kernel_1d(sigma: Sigma, truncate: f32) -> GaussianKernel1D {
    let radius = gaussian_radius(sigma, truncate);
    let sigma = sigma.get();
    assert!(
        radius <= MAX_RADIUS,
        "gaussian kernel: sigma {sigma} (truncate {truncate}) needs radius {radius}, \
         which exceeds MAX_RADIUS ({MAX_RADIUS}); the largest supported sigma is {}",
        MAX_RADIUS as f32 / truncate
    );

    let n = 2 * radius + 1;
    let mut weights = [0.0f32; MAX_GAUSSIAN_TAPS];
    let inv_two_sigma_sq = 1.0 / (2.0 * sigma * sigma);

    let mut sum = 0.0f32;
    for (i, w) in weights[..n].iter_mut().enumerate() {
        let d = i as f32 - radius as f32;
        let value = (-d * d * inv_two_sigma_sq).exp();
        *w = value;
        sum += value;
    }

    let inv_sum = 1.0 / sum;
    for w in weights[..n].iter_mut() {
        *w *= inv_sum;
    }

    GaussianKernel1D {
        weights,
        len: n,
        anchor: radius,
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
        // [0.0625, 0.25, 0.375, 0.25, 0.0625] is symmetric
        assert_eq!(k, f);
    }

    // ── factory methods ─────────────────────────────────────────────────

    #[test]
    fn gaussian_3_weights() {
        let k = SeparableKernel::gaussian_3();
        assert_eq!(k.h_weights(), &[0.25, 0.5, 0.25]);
        assert_eq!(k.v_weights(), &[0.25, 0.5, 0.25]);
        assert_eq!(k.h_anchor(), 1);
        assert_eq!(k.v_anchor(), 1);
        // Normalized: each 1D pass sums to 1 (brightness-preserving).
        assert!((k.h_weights().iter().sum::<f32>() - 1.0).abs() < 1e-7);
    }

    #[test]
    fn gaussian_5_weights() {
        let k = SeparableKernel::gaussian_5();
        assert_eq!(k.h_weights(), &[0.0625, 0.25, 0.375, 0.25, 0.0625]);
        assert_eq!(k.v_weights(), &[0.0625, 0.25, 0.375, 0.25, 0.0625]);
        assert_eq!(k.h_anchor(), 2);
        assert_eq!(k.v_anchor(), 2);
        // Normalized: each 1D pass sums to 1 (brightness-preserving).
        assert!((k.h_weights().iter().sum::<f32>() - 1.0).abs() < 1e-7);
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
        // The separable kernel is now normalized (sum 1), while the raw
        // `Neighborhood` kernel sums to 16, so the outer product equals the
        // raw 2D kernel divided by 16 (same shape, normalized scale).
        let sep = SeparableKernel::gaussian_3();
        let full = crate::image::Neighborhood::<f32, 3, 3>::gaussian_3x3();

        for y in 0..3 {
            for x in 0..3 {
                let outer = sep.h_weights()[x] * sep.v_weights()[y];
                let expected = full.weights().pixel_at(x, y) / 16.0;
                assert!(
                    (outer - expected).abs() < 1e-6,
                    "mismatch at ({x}, {y}): outer={outer}, expected={expected}"
                );
            }
        }
    }

    #[test]
    fn gaussian_5_outer_product_matches_neighborhood() {
        // Normalized separable kernel (sum 1) vs raw `Neighborhood` kernel
        // (sum 256): the outer product equals the raw 2D kernel / 256.
        let sep = SeparableKernel::gaussian_5();
        let full = crate::image::Neighborhood::<f32, 5, 5>::gaussian_5x5();

        for y in 0..5 {
            for x in 0..5 {
                let outer = sep.h_weights()[x] * sep.v_weights()[y];
                let expected = full.weights().pixel_at(x, y) / 256.0;
                assert!(
                    (outer - expected).abs() < 1e-6,
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

    // ── parameterized Gaussian kernel ───────────────────────────────────

    #[test]
    fn gaussian_kernel_weights_sum_to_one() {
        // The DC / normalization invariant — the single most important
        // property (brightness preservation).
        for &sigma in &[0.5f32, 0.8, 1.0, 1.7, 3.0, 8.0] {
            let k = gaussian_kernel_1d(Sigma::new(sigma), 4.0);
            let sum: f32 = k.weights().iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "sigma {sigma}: weights sum to {sum}, expected 1.0"
            );
        }
    }

    #[test]
    fn gaussian_kernel_weights_symmetric() {
        let k = gaussian_kernel_1d(Sigma::new(1.5), 4.0);
        let w = k.weights();
        let n = w.len();
        for i in 0..n {
            assert!(
                (w[i] - w[n - 1 - i]).abs() < 1e-7,
                "asymmetry at tap {i}: {} vs {}",
                w[i],
                w[n - 1 - i]
            );
        }
    }

    #[test]
    fn gaussian_kernel_matches_reference_formula() {
        // Compare to an independent brute-force normalized reference.
        let sigma = 1.3f32;
        let truncate = 4.0f32;
        let k = gaussian_kernel_1d(Sigma::new(sigma), truncate);
        let radius = k.radius();
        let n = k.len();

        let mut reference = vec![0.0f32; n];
        let mut sum = 0.0f32;
        for (i, r) in reference.iter_mut().enumerate() {
            let d = i as f32 - radius as f32;
            *r = (-(d * d) / (2.0 * sigma * sigma)).exp();
            sum += *r;
        }
        for r in &mut reference {
            *r /= sum;
        }

        for (i, (&got, &want)) in k.weights().iter().zip(&reference).enumerate() {
            assert!((got - want).abs() < 1e-6, "tap {i}: got {got}, want {want}");
        }
    }

    #[test]
    fn gaussian_kernel_size_follows_truncate() {
        // size = 2 * round(truncate * sigma) + 1
        assert_eq!(gaussian_kernel_size(Sigma::new(1.0), 4.0), 9); // radius 4
        assert_eq!(gaussian_kernel_size(Sigma::new(2.0), 3.0), 13); // radius 6
        assert_eq!(gaussian_kernel_size(Sigma::new(1.0), 3.0), 7); // radius 3
        // size() agrees with the built kernel's tap count.
        assert_eq!(
            gaussian_kernel_1d(Sigma::new(1.0), 4.0).len(),
            gaussian_kernel_size(Sigma::new(1.0), 4.0)
        );
    }

    #[test]
    fn gaussian_kernel_tiny_sigma_is_identity() {
        // sigma small enough that round(truncate * sigma) == 0 ⇒ 1 tap [1.0].
        let k = gaussian_kernel_1d(Sigma::new(0.05), 4.0);
        assert_eq!(k.len(), 1);
        assert_eq!(k.radius(), 0);
        assert_eq!(k.anchor(), 0);
        assert!((k.weights()[0] - 1.0).abs() < 1e-7);
    }

    // Invalid sigma is unrepresentable in the `Sigma` parameter type —
    // its rejection is tested at the type's constructors in `common.rs`.

    #[test]
    #[should_panic(expected = "truncate must be > 0.0")]
    fn gaussian_kernel_zero_truncate_panics() {
        let _ = gaussian_kernel_1d(Sigma::new(1.0), 0.0);
    }

    #[test]
    #[should_panic(expected = "exceeds MAX_RADIUS")]
    fn gaussian_kernel_over_radius_panics() {
        // radius = round(4.0 * 20.0) = 80 > MAX_RADIUS (64).
        let _ = gaussian_kernel_1d(Sigma::new(20.0), 4.0);
    }

    #[test]
    fn gaussian_kernel_size_reports_over_max_radius_without_panicking() {
        // The size query is total where the builder is not: it must report
        // the derived size for an out-of-range sigma so callers can test
        // admissibility instead of catching a panic.
        assert_eq!(gaussian_kernel_size(Sigma::new(20.0), 4.0), 161);
        assert!(gaussian_kernel_size(Sigma::new(20.0), 4.0) > 2 * MAX_RADIUS + 1);
        // The largest admissible sigma sits exactly on the bound.
        assert_eq!(
            gaussian_kernel_size(Sigma::new(16.0), 4.0),
            2 * MAX_RADIUS + 1
        );
    }

    #[test]
    fn gaussian_kernel_at_max_radius_is_ok() {
        // radius exactly MAX_RADIUS must succeed: round(4.0 * 16.0) = 64.
        let k = gaussian_kernel_1d(Sigma::new(16.0), 4.0);
        assert_eq!(k.radius(), MAX_RADIUS);
        assert_eq!(k.len(), 2 * MAX_RADIUS + 1);
        let sum: f32 = k.weights().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    // ── SeparableWeights ─────────────────────────────────────────────────

    #[test]
    fn gaussian_kernel_is_its_own_flip() {
        // `GaussianKernel1D::flipped` returns a copy of itself. That is only
        // honest if the taps are a palindrome and the anchor is the centre —
        // both asserted here, across the σ range and both truncate values in
        // use, so the shortcut cannot rot silently.
        for sigma in [0.05f32, 0.8, 1.5, 4.0] {
            for truncate in [3.0f32, 4.0] {
                let k = gaussian_kernel_1d(Sigma::new(sigma), truncate);
                let w = k.weights();
                let n = w.len();

                for i in 0..n {
                    assert!(
                        (w[i] - w[n - 1 - i]).abs() < f32::EPSILON,
                        "σ={sigma} t={truncate}: tap {i} != tap {}",
                        n - 1 - i,
                    );
                }
                assert_eq!(
                    k.anchor(),
                    n - 1 - k.anchor(),
                    "σ={sigma}: anchor off-centre"
                );

                let f = SeparableWeights::flipped(&k);
                assert_eq!(f.h_weights(), k.h_weights());
                assert_eq!(f.v_weights(), k.v_weights());
                assert_eq!(f.h_anchor(), k.h_anchor());
                assert_eq!(f.v_anchor(), k.v_anchor());
            }
        }
    }

    #[test]
    fn gaussian_kernel_reports_the_same_taps_on_both_axes() {
        // A Gaussian is isotropic: one weight array serves both passes.
        let k = gaussian_kernel_1d(Sigma::new(1.5), 4.0);
        assert_eq!(k.h_weights(), k.v_weights());
        assert_eq!(k.h_anchor(), k.v_anchor());
        assert_eq!(k.h_weights(), k.weights());
        assert_eq!(k.h_anchor(), k.anchor());
    }

    #[test]
    fn separable_kernel_trait_form_matches_its_inherent_methods() {
        // The trait must not paraphrase the struct: same weights, same
        // anchors, same flip — including for an asymmetric kernel, where a
        // wrong flip would be invisible on a palindrome.
        let k = SeparableKernel::with_anchors([1.0, 2.0, 3.0], 0, [4.0, 5.0], 1);

        assert_eq!(SeparableWeights::h_weights(&k), &k.h_weights()[..]);
        assert_eq!(SeparableWeights::v_weights(&k), &k.v_weights()[..]);
        assert_eq!(SeparableWeights::h_anchor(&k), k.h_anchor());
        assert_eq!(SeparableWeights::v_anchor(&k), k.v_anchor());

        let inherent = k.flipped();
        let via_trait = SeparableWeights::flipped(&k);
        assert_eq!(
            SeparableWeights::h_weights(&via_trait),
            &inherent.h_weights()[..]
        );
        assert_eq!(
            SeparableWeights::v_weights(&via_trait),
            &inherent.v_weights()[..]
        );
        assert_eq!(via_trait.h_anchor(), inherent.h_anchor());
        assert_eq!(via_trait.v_anchor(), inherent.v_anchor());
    }

    #[test]
    fn both_kernel_types_satisfy_the_trait_contract() {
        // One generic reader over both representations — the property the
        // separable entry points rely on.
        fn axes<K: SeparableWeights>(k: &K) -> (usize, usize, usize, usize) {
            assert!(!k.h_weights().is_empty(), "h axis must be non-empty");
            assert!(!k.v_weights().is_empty(), "v axis must be non-empty");
            assert!(k.h_anchor() < k.h_weights().len(), "h anchor out of bounds");
            assert!(k.v_anchor() < k.v_weights().len(), "v anchor out of bounds");
            (
                k.h_weights().len(),
                k.h_anchor(),
                k.v_weights().len(),
                k.v_anchor(),
            )
        }

        assert_eq!(axes(&SeparableKernel::gaussian_5()), (5, 2, 5, 2));
        assert_eq!(axes(&SeparableKernel::box_blur_3()), (3, 1, 3, 1));

        // radius = round(4.0 * 1.0) = 4 ⇒ 9 taps, anchor 4.
        assert_eq!(
            axes(&gaussian_kernel_1d(Sigma::new(1.0), 4.0)),
            (9, 4, 9, 4)
        );
        // Tiny σ collapses to the 1-tap identity, the trait's edge case.
        assert_eq!(
            axes(&gaussian_kernel_1d(Sigma::new(0.05), 4.0)),
            (1, 0, 1, 0)
        );
    }
}

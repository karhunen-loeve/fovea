//! Corner detectors: the structure-tensor family (Harris, Shi-Tomasi).
//!
//! A corner is a point where the image gradient points in two different
//! directions at once. The structure tensor
//!
//! ```text
//! M = [ Sxx  Sxy ]     Sxx = w * (Gx·Gx)
//!     [ Sxy  Syy ]     Syy = w * (Gy·Gy)      w = the window, `*` = convolution
//!                      Sxy = w * (Gx·Gy)
//! ```
//!
//! is the local average of the gradient's outer product, and the whole family
//! differs only in how its two eigenvalues are collapsed into one number:
//!
//! | Response | Formula | Reach for it when |
//! |---|---|---|
//! | [`Harris`] | `det(M) − k·tr(M)²` | You want the classical detector, comparable with other libraries' `cornerHarris`. |
//! | [`ShiTomasi`] | `λ_min(M)` | You want a response with an interpretable scale (it is a squared gradient, not a fourth power) and no `k` to tune. |
//!
//! Both are strategies over one engine, so a third measure is an
//! implementation of [`CornerResponse`] rather than a copied pipeline.
//!
//! ## The pipeline, and how to take it apart
//!
//! [`detect_corners`] is an orchestrator, not a primitive — the same
//! relationship [`canny`](crate::analyze::edge::canny) has to its stages:
//!
//! ```text
//! Sobel Gx, Gy → products Gx², Gy², Gx·Gy → Gaussian window (σ)
//!    → response map → threshold + local-maximum selection → Vec<Corner>
//! ```
//!
//! Every stage is public. [`corner_response_map`] returns the response image
//! itself (to visualize, or to threshold differently), [`StructureTensor`]
//! builds the three sums from *your* gradients — so Scharr instead of Sobel,
//! or a box window instead of a Gaussian, needs no new API — and
//! [`corner_peaks`] turns any response map into keypoints.
//!
//! ```
//! use fovea::Sigma;
//! use fovea::border::Clamp;
//! use fovea::features::detect::{corner_peaks, Harris, StructureTensor};
//! use fovea::image::Image;
//! use fovea::pixel::MonoF32;
//! use fovea::transform::{scharr_x, scharr_y};
//!
//! // A hand-built Harris with a Scharr gradient instead of the pinned Sobel.
//! let image: Image<MonoF32> = Image::generate(24, 24, |x, y| {
//!     MonoF32::new(if (8..16).contains(&x) && (8..16).contains(&y) { 1.0 } else { 0.0 })
//! });
//!
//! let gx = scharr_x(&image, &Clamp);
//! let gy = scharr_y(&image, &Clamp);
//! let tensor = StructureTensor::from_gradients(&gx, &gy, Sigma::new(1.2))?;
//! let response = tensor.response(&Harris::new(0.04));
//!
//! // The square's four corners, and nothing else.
//! let peak = corner_peaks(&response, 0.0, 3)
//!     .into_iter()
//!     .map(|c| c.response)
//!     .fold(0.0f32, f32::max);
//! let corners = corner_peaks(&response, 0.2 * peak, 3);
//! assert_eq!(corners.len(), 4);
//! # Ok::<(), fovea::Error>(())
//! ```
//!
//! ## Choosing a threshold
//!
//! The threshold is **absolute**, in the response map's own units, and those
//! units are not intuitive: the gradient operator's gain enters the response
//! at its own power (Sobel's `[-1 0 1; -2 0 2; -1 0 1]` has a positive-lobe
//! gain of 4, so a unit-contrast step yields `|G| = 4`), and so does image
//! contrast — squared for [`ShiTomasi`], to the fourth power for [`Harris`].
//! A `Mono8` image therefore produces responses larger than the same picture
//! as `MonoF32` in `0.0..=1.0` by a factor of `255⁴ ≈ 4·10⁹`.
//!
//! Do not guess. Calibrate against [`corner_response_map`] on a
//! representative image and take a fraction of its maximum — that is what
//! the examples here do, and it is the only recipe that survives a change of
//! pixel type, gradient operator, or window σ. The alternative, a
//! "quality level" knob relative to the strongest corner in *this* frame,
//! is deliberately absent: it makes a detection depend on the rest of the
//! frame, which is a decision for the caller (PHILOSOPHY §8), not for the
//! detector.
//!
//! ## Scale
//!
//! These detectors do **not** select scale — they find corners at the
//! resolution and window σ they are given, which is why they return
//! [`Corner`] (position + response) rather than
//! [`ScaleKeypoint`](crate::features::ScaleKeypoint). Running the same
//! detector over a [`Pyramid`](crate::image::Pyramid) is multi-resolution,
//! not scale selection: it finds more corners, but none of them has a
//! *characteristic* σ that the detector chose. [`detect_corners_in_level`]
//! is that variant, reporting every level's detections in the base-image
//! frame so they are directly comparable:
//!
//! ```
//! use fovea::{CoordinateF64, PixelDistance, Sigma};
//! use fovea::features::HasPosition;
//! use fovea::features::detect::{detect_corners_in_level, CornerParams, ShiTomasi};
//! use fovea::image::{Image, ScaledImage};
//! use fovea::pixel::MonoF32;
//! use fovea::transform::pyr_down;
//!
//! let base: Image<MonoF32> = Image::generate(48, 48, |x, y| {
//!     MonoF32::new(if (16..32).contains(&x) && (16..32).contains(&y) { 1.0 } else { 0.0 })
//! });
//!
//! // Octave 1: pyr_down keeps even samples — distance 2, origin unshifted.
//! let level = ScaledImage::new(
//!     pyr_down(&base),
//!     PixelDistance::new(2.0),
//!     CoordinateF64::new(0.0, 0.0),
//!     Sigma::new(1.0),
//! );
//!
//! let params = CornerParams::try_new(Sigma::new(1.0), 0.05, 2)?;
//! let corners = detect_corners_in_level(&level, &ShiTomasi, params);
//!
//! // Found on a 24×24 level, reported in the 48×48 base frame: an x of 30
//! // is not a coordinate the level could have produced.
//! assert_eq!(corners.len(), 4);
//! assert!(corners.iter().any(|c| c.position().x > 24.0), "{corners:?}");
//! # Ok::<(), fovea::Error>(())
//! ```

use core::ops::{Add, Mul};

use crate::border::Clamp;
use crate::error::Error;
use crate::features::Corner;
use crate::image::{Decimated, Image, ImageView, RasterImage, RasterImageMut};
use crate::pixel::{FromLinear, LinearPixel, SingleChannel, ZeroablePixel};
use crate::transform::{PixelMultiply, combine_images, gaussian_blur, sobel_x, sobel_y};
use crate::{CoordinateF64, Sigma, Size};

// ─── Response channel arithmetic ─────────────────────────────────────────────

/// Sealing module for [`CornerResponseChannel`].
mod response_sealed {
    pub trait Sealed: Copy {}
}

/// Channel types the structure-tensor response formulas are defined over.
///
/// Implemented for `f32` and `f64` — the channels of the accumulator pixels
/// the gradient stage produces. This trait is **sealed**: it cannot be
/// implemented outside this crate.
///
/// It exists for the same reason
/// [`MagnitudeChannel`](crate::transform::MagnitudeChannel) does: the
/// formulas are one line of float arithmetic each, and putting them here
/// makes them generic over precision without every caller re-deriving them.
/// Which one you get follows the input's accumulator: `Mono8`, `Mono16` and
/// `MonoF32` land on `f32`, `Mono32`, `Mono64` and `MonoF64` on `f64` — and a
/// fourth-power response is exactly the kind of quantity where that
/// difference is worth stating.
pub trait CornerResponseChannel: response_sealed::Sealed + Copy {
    /// The Harris response `det(M) − k·tr(M)²`.
    ///
    /// `k` is the sensitivity, taken as `f32` and widened: it is a
    /// dimensionless constant, so it never needs the accumulator's
    /// precision. Its invariant lives in [`Harris`], not here.
    fn harris(sxx: Self, sxy: Self, syy: Self, k: f32) -> Self;

    /// The Shi-Tomasi response `λ_min(M)`, the smaller eigenvalue.
    ///
    /// Computed as `½·tr − sqrt((½·(Sxx − Syy))² + Sxy²)`, the closed form
    /// for a symmetric 2×2 matrix. The half-difference form is used rather
    /// than `½·(tr − sqrt(tr² − 4·det))` because the latter subtracts two
    /// nearly equal fourth-power quantities, and loses most of its
    /// significant digits exactly where the response is small.
    fn shi_tomasi(sxx: Self, sxy: Self, syy: Self) -> Self;
}

impl response_sealed::Sealed for f32 {}
impl CornerResponseChannel for f32 {
    #[inline(always)]
    fn harris(sxx: f32, sxy: f32, syy: f32, k: f32) -> f32 {
        let trace = sxx + syy;
        sxx * syy - sxy * sxy - k * trace * trace
    }

    #[inline(always)]
    fn shi_tomasi(sxx: f32, sxy: f32, syy: f32) -> f32 {
        let half_sum = 0.5 * (sxx + syy);
        let half_diff = 0.5 * (sxx - syy);
        half_sum - (half_diff * half_diff + sxy * sxy).sqrt()
    }
}

impl response_sealed::Sealed for f64 {}
impl CornerResponseChannel for f64 {
    #[inline(always)]
    fn harris(sxx: f64, sxy: f64, syy: f64, k: f32) -> f64 {
        let trace = sxx + syy;
        sxx * syy - sxy * sxy - f64::from(k) * trace * trace
    }

    #[inline(always)]
    fn shi_tomasi(sxx: f64, sxy: f64, syy: f64) -> f64 {
        let half_sum = 0.5 * (sxx + syy);
        let half_diff = 0.5 * (sxx - syy);
        half_sum - (half_diff * half_diff + sxy * sxy).sqrt()
    }
}

// ─── CornerResponse strategies ───────────────────────────────────────────────

/// Strategy for collapsing a structure tensor into a single cornerness value.
///
/// The one step Harris and Shi-Tomasi disagree on. Everything around it —
/// gradients, products, window, peak selection — is shared, so a new measure
/// (Noble's `det/tr`, Förstner's, a curvature ratio) is an implementation of
/// this trait and not a second copy of the pipeline.
///
/// `C` is the response channel: `f32` or `f64`, following whichever
/// accumulator the gradient stage produced. Implementations delegate to
/// [`CornerResponseChannel`] to stay generic over both.
///
/// `Sxy` appears once because `M` is symmetric — there is no second
/// off-diagonal argument to get wrong.
///
/// # Example
///
/// A response that scores only how *isotropic* the gradient is, ignoring its
/// strength:
///
/// ```
/// use fovea::features::detect::CornerResponse;
///
/// struct Isotropy;
///
/// impl CornerResponse<f32> for Isotropy {
///     fn response(&self, sxx: f32, sxy: f32, syy: f32) -> f32 {
///         let trace = sxx + syy;
///         if trace == 0.0 {
///             return 0.0;
///         }
///         (sxx * syy - sxy * sxy) / (trace * trace)
///     }
/// }
///
/// // A perfectly isotropic tensor reaches the 1/4 maximum.
/// assert!((Isotropy.response(1.0, 0.0, 1.0) - 0.25).abs() < 1e-6);
/// // A pure edge (one zero eigenvalue) scores nothing.
/// assert_eq!(Isotropy.response(1.0, 0.0, 0.0), 0.0);
/// ```
pub trait CornerResponse<C> {
    /// Returns the cornerness of the structure tensor
    /// `[[sxx, sxy], [sxy, syy]]`.
    fn response(&self, sxx: C, sxy: C, syy: C) -> C;
}

/// The Harris response `det(M) − k·tr(M)²`, carrying its validated
/// sensitivity `k`.
///
/// The classical corner measure, and the one to reach for when results must
/// line up with another library's `cornerHarris`. `k` trades corner
/// detections against edge rejections: smaller values accept more corners
/// (and more edges), larger values fewer. `0.04` is the conventional
/// starting point, `0.04..=0.06` the usual range.
///
/// This type *is* the invariant-carrying parameter type for `k` — the same
/// discipline as [`Sigma`](crate::Sigma) and `std::num::NonZeroUsize`, with
/// the strategy that consumes the value also owning its validation, so
/// there is no separate newtype to thread through the API. Literals use the
/// `const fn` [`new`](Self::new); values computed from data use
/// [`try_new`](Self::try_new).
///
/// # Why `0 < k < 0.25`
///
/// Not a convention — a consequence. For a symmetric 2×2 matrix
/// `det = λ₁λ₂` and `tr = λ₁ + λ₂`, and the AM-GM inequality gives
/// `det ≤ tr²/4`. At `k = 0.25` the response is therefore `≤ 0` for
/// *every* tensor, and no corner can ever be reported; above it the
/// detector is dead. Below zero the `−k·tr²` term becomes a *reward* for a
/// large trace, which is exactly the edge response the term exists to
/// subtract. Both ends are silent failures — an empty result and a corner
/// detector that prefers edges — so both are rejected at construction
/// instead.
///
/// # Example
///
/// ```
/// use fovea::features::detect::Harris;
///
/// const CLASSIC: Harris = Harris::new(0.04); // checked at compile time
/// assert_eq!(CLASSIC.k(), 0.04);
///
/// // A value from a tuning sweep is checked where it is computed.
/// let swept = Harris::try_new(0.02 * 3.0)?;
/// assert!((swept.k() - 0.06).abs() < 1e-6);
///
/// // k ≥ 0.25 makes the response non-positive everywhere: rejected.
/// assert!(Harris::try_new(0.25).is_err());
/// # Ok::<(), fovea::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Harris(f32);

impl Harris {
    /// Creates a `Harris` response from a literal or otherwise proven-valid
    /// sensitivity.
    ///
    /// # Panics
    ///
    /// Panics unless `0.0 < k < 0.25` (see the type documentation for why
    /// those are the bounds). As a `const fn`, this is a **compile error**
    /// when evaluated in a `const` context. For values computed from data,
    /// use [`try_new`](Self::try_new).
    #[must_use]
    pub const fn new(k: f32) -> Self {
        assert!(
            k.is_finite() && k > 0.0 && k < 0.25,
            "Harris::new: k must satisfy 0 < k < 0.25"
        );
        Self(k)
    }

    /// Creates a `Harris` response from a computed sensitivity, validating
    /// it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidParameter`] unless `0.0 < k < 0.25`.
    pub fn try_new(k: f32) -> Result<Self, Error> {
        if k.is_finite() && k > 0.0 && k < 0.25 {
            Ok(Self(k))
        } else {
            Err(Error::InvalidParameter(format!(
                "Harris k must satisfy 0 < k < 0.25, got {k}"
            )))
        }
    }

    /// Returns the sensitivity `k`.
    #[must_use]
    pub const fn k(self) -> f32 {
        self.0
    }
}

impl<C: CornerResponseChannel> CornerResponse<C> for Harris {
    #[inline(always)]
    fn response(&self, sxx: C, sxy: C, syy: C) -> C {
        C::harris(sxx, sxy, syy, self.0)
    }
}

/// The Shi-Tomasi response `λ_min(M)` — "Good Features to Track".
///
/// The smaller eigenvalue of the structure tensor: a corner is a point where
/// *both* eigenvalues are large, so the weaker one is already the honest
/// score. Two practical consequences over [`Harris`]:
///
/// - **No `k`.** There is nothing to tune, and no way to tune the detector
///   into never firing.
/// - **Interpretable units.** The response is a squared gradient rather than
///   a fourth power, so a threshold moves with contrast squared instead of
///   contrast to the fourth — an easier number to carry between images.
///
/// The cost is a square root per pixel where Harris needs only multiplies.
///
/// # Example
///
/// ```
/// use fovea::features::detect::{CornerResponse, ShiTomasi};
///
/// // Eigenvalues 4 and 1 on the diagonal: the response is the smaller one.
/// assert!((ShiTomasi.response(4.0f32, 0.0, 1.0) - 1.0).abs() < 1e-6);
/// // A pure edge has a zero eigenvalue, so it scores nothing …
/// assert!(ShiTomasi.response(9.0f32, 0.0, 0.0).abs() < 1e-6);
/// // … however strong the edge is.
/// assert!(ShiTomasi.response(1e6f32, 0.0, 0.0).abs() < 1e-3);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShiTomasi;

impl<C: CornerResponseChannel> CornerResponse<C> for ShiTomasi {
    #[inline(always)]
    fn response(&self, sxx: C, sxy: C, syy: C) -> C {
        C::shi_tomasi(sxx, sxy, syy)
    }
}

// ─── StructureTensor ─────────────────────────────────────────────────────────

/// The three windowed gradient products of the structure tensor, as images.
///
/// `Sxx`, `Sxy` and `Syy` are the entries of the symmetric 2×2 matrix `M`
/// held per pixel — the local average of the gradient's outer product. This
/// type is the reusable half of every detector in this module: it owns the
/// gradient-product and windowing stages, and knows nothing about which
/// [`CornerResponse`] will consume it.
///
/// It is also where the pipeline is opened up. [`from_gradients`] takes
/// *your* gradient images, so the operator is your choice (Sobel, Scharr,
/// Prewitt, a derivative-of-Gaussian you convolved yourself), and
/// [`from_smoothed`] takes the three windowed products directly, so the
/// window is your choice too — a box window from an integral image needs no
/// new parameter type here.
///
/// [`from_gradients`]: Self::from_gradients
/// [`from_smoothed`]: Self::from_smoothed
///
/// # Example
///
/// ```
/// use fovea::Sigma;
/// use fovea::border::Clamp;
/// use fovea::features::detect::{Harris, StructureTensor};
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::MonoF32;
/// use fovea::transform::{sobel_x, sobel_y};
///
/// // A vertical step edge: strong Gx, no Gy.
/// let image: Image<MonoF32> = Image::generate(16, 16, |x, _| {
///     MonoF32::new(if x < 8 { 0.0 } else { 1.0 })
/// });
///
/// let gx = sobel_x(&image, &Clamp);
/// let gy = sobel_y(&image, &Clamp);
/// let tensor = StructureTensor::from_gradients(&gx, &gy, Sigma::new(1.0))?;
///
/// // On the edge, Sxx is large and Syy vanishes …
/// assert!(tensor.xx().pixel_at(8, 8).value() > 1.0);
/// assert!(tensor.yy().pixel_at(8, 8).value() < 1e-6);
/// // … so the Harris response there is negative: an edge, not a corner.
/// assert!(tensor.response(&Harris::new(0.04)).pixel_at(8, 8).value() < 0.0);
/// # Ok::<(), fovea::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct StructureTensor<P: Copy> {
    xx: Image<P>,
    xy: Image<P>,
    yy: Image<P>,
}

impl<P> StructureTensor<P>
where
    P: Copy
        + Default
        + ZeroablePixel
        + FromLinear<P>
        + LinearPixel<f32, Accumulator = P>
        + Add<Output = P>
        + Mul<Output = P>,
{
    /// Builds the structure tensor from a pair of gradient images, windowed
    /// by a Gaussian of standard deviation `window`.
    ///
    /// The two stages, in order: the pixel-wise products `Gx·Gx`, `Gx·Gy`,
    /// `Gy·Gy`, then a [`gaussian_blur`] of each with a [`Clamp`] border.
    ///
    /// The window is what makes the tensor *have* two eigenvalues: without
    /// it every `M` is the outer product of a single gradient vector, whose
    /// smaller eigenvalue is identically zero, and no measure built on it
    /// could ever report a corner. A larger σ integrates over more of the
    /// neighbourhood — more robust to noise, less able to separate corners
    /// that sit close together.
    ///
    /// A Gaussian is the named default because it is isotropic, which is
    /// what keeps the response from depending on how the corner happens to
    /// be rotated. For a box window — cheaper, computable from an integral
    /// image, and not isotropic — smooth the products yourself and use
    /// [`from_smoothed`](Self::from_smoothed).
    ///
    /// # Errors — Tier 2
    ///
    /// Returns [`Error::SizeMismatch`] if `gx` and `gy` differ in
    /// dimensions: two separately obtained runtime images, exactly the
    /// relation [`combine_images`] reports.
    ///
    /// # Panics
    ///
    /// Panics if `window`'s derived kernel radius exceeds
    /// [`MAX_RADIUS`](crate::image::MAX_RADIUS) (via [`gaussian_blur`]),
    /// testable up front with
    /// [`gaussian_kernel_size`](crate::image::gaussian_kernel_size).
    pub fn from_gradients<IX, IY>(gx: &IX, gy: &IY, window: Sigma) -> Result<Self, Error>
    where
        IX: RasterImage<Pixel = P>,
        IY: RasterImage<Pixel = P>,
    {
        // `gx·gy` is the only product that can mismatch; taking it first
        // means the error surfaces before two blurs have been paid for.
        let xy = combine_images(gx, gy, PixelMultiply)?;
        let xx = combine_images(gx, gx, PixelMultiply).expect("gx shares its own size");
        let yy = combine_images(gy, gy, PixelMultiply).expect("gy shares its own size");

        Ok(Self {
            xx: gaussian_blur(&xx, window, &Clamp),
            xy: gaussian_blur(&xy, window, &Clamp),
            yy: gaussian_blur(&yy, window, &Clamp),
        })
    }
}

impl<P: Copy> StructureTensor<P> {
    /// Assembles a structure tensor from three already-windowed product
    /// images.
    ///
    /// The escape hatch from [`from_gradients`](Self::from_gradients)'s
    /// pinned Gaussian: whatever produced `sxx`, `sxy` and `syy` — a box
    /// blur, an integral-image region sum, a bilateral window — the
    /// downstream response and peak stages are unchanged. Nothing is
    /// recomputed here; the three images are taken as given.
    ///
    /// # Errors — Tier 2
    ///
    /// Returns [`Error::SizeMismatch`] unless all three images have the same
    /// dimensions.
    pub fn from_smoothed(sxx: Image<P>, sxy: Image<P>, syy: Image<P>) -> Result<Self, Error> {
        for other in [sxy.size(), syy.size()] {
            if other != sxx.size() {
                return Err(Error::SizeMismatch {
                    expected: sxx.size(),
                    actual: other,
                });
            }
        }
        Ok(Self {
            xx: sxx,
            xy: sxy,
            yy: syy,
        })
    }

    /// Returns the windowed `Gx·Gx` image.
    #[must_use]
    pub fn xx(&self) -> &Image<P> {
        &self.xx
    }

    /// Returns the windowed `Gx·Gy` image, the off-diagonal entry.
    #[must_use]
    pub fn xy(&self) -> &Image<P> {
        &self.xy
    }

    /// Returns the windowed `Gy·Gy` image.
    #[must_use]
    pub fn yy(&self) -> &Image<P> {
        &self.yy
    }

    /// Returns the common dimensions of the three entries.
    #[must_use]
    pub fn size(&self) -> Size {
        self.xx.size()
    }

    /// Collapses the tensor into a cornerness map with the given response
    /// strategy.
    ///
    /// The output is a fresh image of the same pixel type and size: the
    /// response at each pixel is a scalar derived from that pixel's `M`, so
    /// nothing here reads a neighbourhood. Sign and magnitude are
    /// strategy-defined — see [`Harris`] and [`ShiTomasi`], and the module
    /// documentation on choosing a threshold.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::Sigma;
    /// use fovea::border::Clamp;
    /// use fovea::features::detect::{ShiTomasi, StructureTensor};
    /// use fovea::image::{Image, ImageView};
    /// use fovea::pixel::MonoF32;
    /// use fovea::transform::{sobel_x, sobel_y};
    ///
    /// // A flat field has no gradient, so every response is zero.
    /// let flat = Image::fill(12, 12, MonoF32::new(0.5));
    /// let tensor = StructureTensor::from_gradients(
    ///     &sobel_x(&flat, &Clamp),
    ///     &sobel_y(&flat, &Clamp),
    ///     Sigma::new(1.0),
    /// )?;
    /// assert!(tensor.response(&ShiTomasi).pixel_at(6, 6).value().abs() < 1e-6);
    /// # Ok::<(), fovea::Error>(())
    /// ```
    #[must_use]
    pub fn response<M>(&self, method: &M) -> Image<P>
    where
        P: SingleChannel + ZeroablePixel,
        P::Channel: CornerResponseChannel,
        M: CornerResponse<P::Channel>,
    {
        let (w, h) = (self.xx.width(), self.xx.height());
        let mut out = Image::fill(w, h, P::zero());
        for y in 0..h {
            let (xx, xy, yy) = (self.xx.row(y), self.xy.row(y), self.yy.row(y));
            let dst = out.row_mut(y);
            for x in 0..w {
                let value = method.response(xx[x].channel(0), xy[x].channel(0), yy[x].channel(0));
                dst[x] = P::from_channels(&[value]);
            }
        }
        out
    }
}

// ─── CornerParams ────────────────────────────────────────────────────────────

/// The three parameters every structure-tensor detector needs, validated
/// once.
///
/// - `window` — σ of the Gaussian that integrates the gradient products.
/// - `threshold` — absolute minimum response; see the module documentation,
///   because these units are not intuitive.
/// - `nms_radius` — half-side of the square window a detection must be the
///   maximum of, in pixels. It sets the *minimum separation* between two
///   reported corners.
///
/// The type carries the invariants so the detectors are total in it — the
/// same discipline as [`Sigma`](crate::Sigma), which is one of its fields.
/// `threshold` must be finite, because a `NaN`
/// threshold silently rejects every pixel — an empty result that looks like
/// "no corners here" rather than like the mistake it is. `nms_radius` must
/// be at least 1, since a radius of zero asks for the local maximum of a
/// one-pixel window, which every pixel trivially is; the thresholded
/// response map that request really wants is one
/// [`corner_response_map`] call away.
///
/// There is deliberately no `Default`. A default σ and a default threshold
/// would be a claim about *your* images that this crate is not in a
/// position to make (PHILOSOPHY §8).
///
/// # Example
///
/// ```
/// use fovea::Sigma;
/// use fovea::features::detect::CornerParams;
///
/// // Literals: checked at compile time in a const context.
/// const PARAMS: CornerParams = CornerParams::new(Sigma::new(1.4), 0.01, 3);
/// assert_eq!(PARAMS.nms_radius(), 3);
///
/// // Computed: checked where the computation happened.
/// let calibrated = 0.05 * 0.2;
/// let params = CornerParams::try_new(Sigma::new(1.4), calibrated, 3)?;
/// assert!((params.threshold() - 0.01).abs() < 1e-9);
///
/// assert!(CornerParams::try_new(Sigma::new(1.0), f32::NAN, 3).is_err());
/// assert!(CornerParams::try_new(Sigma::new(1.0), 0.01, 0).is_err());
/// # Ok::<(), fovea::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CornerParams {
    window: Sigma,
    threshold: f32,
    nms_radius: usize,
}

impl CornerParams {
    /// Creates parameters from literals or otherwise proven-valid values.
    ///
    /// # Panics
    ///
    /// Panics if `threshold` is not finite, or if `nms_radius == 0`. As a
    /// `const fn`, both are **compile errors** when evaluated in a `const`
    /// context. For values computed from data, use
    /// [`try_new`](Self::try_new).
    #[must_use]
    pub const fn new(window: Sigma, threshold: f32, nms_radius: usize) -> Self {
        assert!(
            threshold.is_finite(),
            "CornerParams::new: threshold must be finite"
        );
        assert!(
            nms_radius > 0,
            "CornerParams::new: nms_radius must be at least 1"
        );
        Self {
            window,
            threshold,
            nms_radius,
        }
    }

    /// Creates parameters from computed values, validating them.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidParameter`] if `threshold` is NaN or
    /// infinite, or if `nms_radius == 0`.
    pub fn try_new(window: Sigma, threshold: f32, nms_radius: usize) -> Result<Self, Error> {
        if !threshold.is_finite() {
            return Err(Error::InvalidParameter(format!(
                "corner response threshold must be finite, got {threshold}"
            )));
        }
        if nms_radius == 0 {
            return Err(Error::InvalidParameter(
                "corner nms_radius must be at least 1".to_string(),
            ));
        }
        Ok(Self {
            window,
            threshold,
            nms_radius,
        })
    }

    /// Returns the structure-tensor window σ.
    #[must_use]
    pub const fn window(self) -> Sigma {
        self.window
    }

    /// Returns the absolute response threshold.
    #[must_use]
    pub const fn threshold(self) -> f32 {
        self.threshold
    }

    /// Returns the non-maximum-suppression radius, in pixels.
    #[must_use]
    pub const fn nms_radius(self) -> usize {
        self.nms_radius
    }
}

// ─── Response map ────────────────────────────────────────────────────────────

/// Computes the cornerness map of an image: gradients, structure tensor,
/// response.
///
/// The public intermediate of [`detect_corners`], for the same reason
/// [`canny`](crate::analyze::edge::canny)'s stages are public — a response
/// map is worth looking at. Threshold it yourself, visualize it, or feed it
/// to [`corner_peaks`]; `detect_corners` is exactly this function followed
/// by that one.
///
/// The gradient is [`sobel_x`] / [`sobel_y`] with a [`Clamp`] border, so the
/// output keeps the input size and matches the operator other libraries'
/// Harris implementations use. Border pixels are computed from replicated
/// edge samples and are therefore extrapolated, not measured — crop or
/// filter by position if that matters. To vary the operator, build the
/// [`StructureTensor`] from your own gradients.
///
/// Works for any single-channel input whose linear accumulator is a float
/// pixel: `Mono8`, `Mono16`, `Mono<BITS>` and `MonoF32` accumulate in
/// `MonoF32`; `Mono32`, `Mono64` and `MonoF64` accumulate in `MonoF64`.
///
/// # Panics
///
/// Panics if `window`'s derived kernel radius exceeds
/// [`MAX_RADIUS`](crate::image::MAX_RADIUS) (via [`gaussian_blur`]).
///
/// # Example
///
/// ```
/// use fovea::Sigma;
/// use fovea::features::detect::{corner_response_map, ShiTomasi};
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::{Mono8, MonoF32};
///
/// // A white square on black: corners at its four corners.
/// let image: Image<Mono8> = Image::generate(24, 24, |x, y| {
///     Mono8::new(if (8..16).contains(&x) && (8..16).contains(&y) { 255 } else { 0 })
/// });
///
/// let response: Image<MonoF32> = corner_response_map(&image, &ShiTomasi, Sigma::new(1.2));
/// assert_eq!(response.size(), image.size());
///
/// // Stronger at a corner of the square than in the middle of its edge.
/// assert!(response.pixel_at(8, 8).value() > response.pixel_at(12, 8).value());
/// ```
#[must_use]
pub fn corner_response_map<I, M, P, Acc>(image: &I, method: &M, window: Sigma) -> Image<Acc>
where
    I: RasterImage<Pixel = P>,
    P: Copy + LinearPixel<f32, Accumulator = Acc>,
    Acc: Copy
        + Default
        + ZeroablePixel
        + SingleChannel
        + FromLinear<Acc>
        + LinearPixel<f32, Accumulator = Acc>
        + Add<Output = Acc>
        + Mul<Output = Acc>,
    Acc::Channel: CornerResponseChannel,
    M: CornerResponse<Acc::Channel>,
{
    let gx = sobel_x(image, &Clamp);
    let gy = sobel_y(image, &Clamp);
    // Both gradients come from one image, so the sizes match by
    // construction — the `Result` cannot be `Err` here.
    let tensor = StructureTensor::from_gradients(&gx, &gy, window).expect("gx and gy share a size");
    tensor.response(method)
}

// ─── Peak selection ──────────────────────────────────────────────────────────

/// Selects the local maxima of a response map as keypoints.
///
/// A pixel is reported when it is at least `threshold` **and** is the
/// maximum of the square window of half-side `radius` around it — so
/// `radius` is the minimum separation between two reported corners. The
/// window is clipped at the image border rather than skipped, so a corner
/// against the edge of the frame can still be reported; whether the response
/// *there* is trustworthy is a property of how the map was built (see
/// [`corner_response_map`]).
///
/// Returned corners are in **raster order** (top to bottom, left to right),
/// not response order. Ranking is a separate, named step:
/// [`retain_top_n`](crate::features::retain_top_n) or
/// [`sort_by_response`](crate::features::sort_by_response).
///
/// # Plateaus and ties
///
/// Exact ties are the rule on synthetic images, and a naïve `>=` comparison
/// reports every pixel of a flat plateau while a naïve `>` reports none of
/// them. The comparison here is asymmetric instead: strictly greater than
/// neighbours *earlier* in raster order, greater or equal to *later* ones.
/// Exactly one corner then survives per group of tied pixels — the group's
/// raster-first — as long as each member of the group lies within `radius`
/// of an earlier one; two tied blobs further apart than `radius` are two
/// groups and yield two corners, which is the same rule the radius states
/// everywhere else. Nothing here depends on the order the map was scanned
/// in. A `NaN` neighbour makes every comparison false, so it suppresses
/// rather than wins.
///
/// # Example
///
/// ```
/// use fovea::features::detect::corner_peaks;
/// use fovea::features::HasPosition;
/// use fovea::image::Image;
/// use fovea::pixel::MonoF32;
///
/// // Two isolated peaks of different strength, plus sub-threshold noise.
/// let response: Image<MonoF32> = Image::generate(9, 9, |x, y| {
///     MonoF32::new(match (x, y) {
///         (2, 2) => 1.0,
///         (6, 6) => 0.5,
///         _ => 0.01,
///     })
/// });
///
/// let corners = corner_peaks(&response, 0.1, 2);
/// let positions: Vec<(f64, f64)> = corners
///     .iter()
///     .map(|c| (c.position().x, c.position().y))
///     .collect();
/// assert_eq!(positions, [(2.0, 2.0), (6.0, 6.0)]);
/// ```
#[must_use]
pub fn corner_peaks<I, P>(response: &I, threshold: f32, radius: usize) -> Vec<Corner>
where
    I: RasterImage<Pixel = P>,
    P: SingleChannel,
    P::Channel: PartialOrd + From<f32>,
    f64: From<P::Channel>,
{
    scan_peaks(response, threshold, radius)
        .into_iter()
        .map(|(at, response)| Corner::new(at, response))
        .collect()
}

/// Collects the local maxima of `response` as `(position, response)` pairs in
/// the map's own coordinate frame.
///
/// The shared engine behind [`corner_peaks`] (which reports positions as
/// found) and [`detect_corners_in_level`] (which lifts them into the
/// base-image frame). Keeping it separate is what stops the level variant
/// from having to re-interpret an already-built [`Corner`]'s position as
/// local coordinates.
fn scan_peaks<I, P>(response: &I, threshold: f32, radius: usize) -> Vec<(CoordinateF64, f32)>
where
    I: RasterImage<Pixel = P>,
    P: SingleChannel,
    P::Channel: PartialOrd + From<f32>,
    f64: From<P::Channel>,
{
    let threshold = <P::Channel as From<f32>>::from(threshold);
    let (w, h) = (response.width(), response.height());
    let mut peaks = Vec::new();

    for y in 0..h {
        for x in 0..w {
            let value = response.row(y)[x].channel(0);
            // Both tests are written in the positive, so a NaN response —
            // which compares false against everything — is excluded rather
            // than reported as an infinitely strong corner.
            if value >= threshold && is_local_max(response, x, y, radius, value) {
                let at = CoordinateF64::new(x as f64, y as f64);
                peaks.push((at, f64::from(value) as f32));
            }
        }
    }
    peaks
}

/// Whether `value` at `(x, y)` is the maximum of its clipped
/// `(2·radius + 1)²` window, with ties resolved in favour of the pixel
/// earlier in raster order.
#[inline]
fn is_local_max<I, P>(response: &I, x: usize, y: usize, radius: usize, value: P::Channel) -> bool
where
    I: RasterImage<Pixel = P>,
    P: SingleChannel,
    P::Channel: PartialOrd,
{
    let (w, h) = (response.width(), response.height());
    let y_lo = y.saturating_sub(radius);
    let y_hi = (y + radius + 1).min(h);
    let x_lo = x.saturating_sub(radius);
    let x_hi = (x + radius + 1).min(w);

    for ny in y_lo..y_hi {
        let row = response.row(ny);
        for (offset, pixel) in row[x_lo..x_hi].iter().enumerate() {
            let nx = x_lo + offset;
            if nx == x && ny == y {
                continue;
            }
            let neighbour = pixel.channel(0);
            // Earlier in raster order ⇒ strict, later ⇒ inclusive. A flat
            // plateau then keeps exactly its raster-first pixel.
            let earlier = ny < y || (ny == y && nx < x);
            let survives = if earlier {
                value > neighbour
            } else {
                value >= neighbour
            };
            if !survives {
                return false;
            }
        }
    }
    true
}

// ─── Detectors ───────────────────────────────────────────────────────────────

/// Detects corners in an image with the given response strategy.
///
/// The composed pipeline: [`corner_response_map`] followed by
/// [`corner_peaks`]. Corners are returned in raster order, positioned at
/// pixel centres — sub-pixel refinement is a separate step, not something
/// this function invents.
///
/// Ranking and top-N selection are also separate and already exist:
///
/// ```
/// use fovea::Sigma;
/// use fovea::features::retain_top_n;
/// use fovea::features::detect::{corner_response_map, detect_corners, CornerParams, Harris};
/// use fovea::image::{Image, ImageView};
/// use fovea::pixel::MonoF32;
///
/// // Two squares, the left one at full contrast, the right one faint.
/// let image: Image<MonoF32> = Image::generate(40, 20, |x, y| {
///     let inside = |x0: usize| (x0..x0 + 8).contains(&x) && (6..14).contains(&y);
///     MonoF32::new(if inside(4) { 1.0 } else if inside(28) { 0.3 } else { 0.0 })
/// });
///
/// // Calibrate the threshold against the map's own maximum.
/// let map: Image<MonoF32> = corner_response_map(&image, &Harris::new(0.04), Sigma::new(1.2));
/// let peak = (0..map.height())
///     .flat_map(|y| (0..map.width()).map(move |x| (x, y)))
///     .map(|(x, y)| map.pixel_at(x, y).value())
///     .fold(0.0f32, f32::max);
///
/// // 0.1 % of the peak, not 10 %: the faint square's contrast is 0.3 of the
/// // bright one's, and a Harris response is a *fourth* power — so its
/// // corners score 0.3⁴ ≈ 1/120 as strongly.
/// let params = CornerParams::try_new(Sigma::new(1.2), 0.001 * peak, 3)?;
/// let mut corners = detect_corners(&image, &Harris::new(0.04), params);
/// assert_eq!(corners.len(), 8); // four per square
///
/// // The strongest four are the full-contrast square's.
/// retain_top_n(&mut corners, 4);
/// assert!(corners.iter().all(|c| c.at.x < 20.0), "{corners:?}");
/// # Ok::<(), fovea::Error>(())
/// ```
///
/// # Panics
///
/// Panics if the window σ's derived kernel radius exceeds
/// [`MAX_RADIUS`](crate::image::MAX_RADIUS) (via [`gaussian_blur`]).
#[must_use]
pub fn detect_corners<I, M, P, Acc>(image: &I, method: &M, params: CornerParams) -> Vec<Corner>
where
    I: RasterImage<Pixel = P>,
    P: Copy + LinearPixel<f32, Accumulator = Acc>,
    Acc: Copy
        + Default
        + ZeroablePixel
        + SingleChannel
        + FromLinear<Acc>
        + LinearPixel<f32, Accumulator = Acc>
        + Add<Output = Acc>
        + Mul<Output = Acc>,
    Acc::Channel: CornerResponseChannel + PartialOrd + From<f32>,
    f64: From<Acc::Channel>,
    M: CornerResponse<Acc::Channel>,
{
    let response = corner_response_map(image, method, params.window());
    corner_peaks(&response, params.threshold(), params.nms_radius())
}

/// Detects corners on a pyramid level, reporting them in the **base-image**
/// frame.
///
/// The multi-resolution variant of [`detect_corners`] — the same detector,
/// not a different one. Detection runs in the level's own coordinates and
/// every position is lifted through
/// [`Corner::from_level`](crate::features::Corner::from_level), so results
/// from different levels are directly comparable and can be concatenated.
///
/// The window σ in `params` is in the **level's** pixels, not the base
/// image's: it sizes the structure-tensor window relative to the data
/// actually being differentiated. The level's own σ
/// ([`ScaleLevel`](crate::image::ScaleLevel)) is not consulted, and the
/// result is a [`Corner`] rather than a
/// [`ScaleKeypoint`](crate::features::ScaleKeypoint) — searching a pyramid
/// finds corners at more resolutions, but the detector still selects no
/// scale, and a σ it did not choose is not a characteristic scale (see
/// [`HasScale`](crate::features::HasScale)).
///
/// # Example
///
/// A whole pyramid in one expression — and the duplicate suppression that
/// concatenating levels then needs, which is the caller's policy to set:
///
/// ```
/// use fovea::Sigma;
/// use fovea::features::detect::{detect_corners_in_level, CornerParams, ShiTomasi};
/// use fovea::features::sort_by_response;
/// use fovea::image::{Image, Pyramid, ScaledImage};
/// use fovea::pixel::MonoF32;
/// use fovea::{CoordinateF64, PixelDistance};
/// use fovea::transform::pyr_down;
///
/// let base: Image<MonoF32> = Image::generate(32, 32, |x, y| {
///     MonoF32::new(if (8..24).contains(&x) && (8..24).contains(&y) { 1.0 } else { 0.0 })
/// });
///
/// // Two levels, each carrying the sampling geometry `pyr_down` produced.
/// let levels = vec![
///     ScaledImage::new(
///         base.clone(),
///         PixelDistance::new(1.0),
///         CoordinateF64::new(0.0, 0.0),
///         Sigma::new(0.5),
///     ),
///     ScaledImage::new(
///         pyr_down(&base),
///         PixelDistance::new(2.0),
///         CoordinateF64::new(0.0, 0.0),
///         Sigma::new(1.0),
///     ),
/// ];
/// let pyramid = Pyramid::try_from_levels(levels)?;
///
/// let params = CornerParams::try_new(Sigma::new(1.0), 0.02, 2)?;
/// let mut corners: Vec<_> = pyramid
///     .iter()
///     .flat_map(|level| detect_corners_in_level(level, &ShiTomasi, params))
///     .collect();
/// sort_by_response(&mut corners);
///
/// // Every level reports in base-image coordinates, so the same physical
/// // corner appears at (roughly) the same position twice.
/// assert!(corners.len() >= 8, "{corners:?}");
/// # Ok::<(), fovea::Error>(())
/// ```
///
/// # Panics
///
/// Panics if the window σ's derived kernel radius exceeds
/// [`MAX_RADIUS`](crate::image::MAX_RADIUS) (via [`gaussian_blur`]).
#[must_use]
pub fn detect_corners_in_level<L, M, P, Acc>(
    level: &L,
    method: &M,
    params: CornerParams,
) -> Vec<Corner>
where
    L: Decimated<Pixel = P>,
    P: Copy + LinearPixel<f32, Accumulator = Acc>,
    Acc: Copy
        + Default
        + ZeroablePixel
        + SingleChannel
        + FromLinear<Acc>
        + LinearPixel<f32, Accumulator = Acc>
        + Add<Output = Acc>
        + Mul<Output = Acc>,
    Acc::Channel: CornerResponseChannel + PartialOrd + From<f32>,
    f64: From<Acc::Channel>,
    M: CornerResponse<Acc::Channel>,
{
    let response = corner_response_map(level.as_image(), method, params.window());
    scan_peaks(&response, params.threshold(), params.nms_radius())
        .into_iter()
        .map(|(local, response)| Corner::from_level(level, local, response))
        .collect()
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::{HasPosition, HasResponse};
    use crate::image::{ImageView, Pyramid, PyramidLevel, ScaledImage};
    use crate::pixel::{Mono8, Mono16, MonoF32, MonoF64};
    use crate::transform::{pyr_down, rotate_90};
    use crate::{CoordinateF64, PixelDistance};

    // ── Fixtures ────────────────────────────────────────────────────────

    /// A white square on a black field: four corners, four straight edges.
    fn square(n: usize, lo: usize, hi: usize) -> Image<MonoF32> {
        Image::generate(n, n, |x, y| {
            let inside = (lo..hi).contains(&x) && (lo..hi).contains(&y);
            MonoF32::new(if inside { 1.0 } else { 0.0 })
        })
    }

    /// The true corner positions of `square(_, lo, hi)`: the step sits
    /// *between* pixels, so the geometric corner is half a pixel outside the
    /// first inside pixel.
    fn square_corners(lo: usize, hi: usize) -> [(f64, f64); 4] {
        let (a, b) = ((lo as f64) - 0.5, (hi as f64) - 0.5);
        [(a, a), (b, a), (a, b), (b, b)]
    }

    /// The four **pixels** of `square(_, lo, hi)` that carry the corners, in
    /// raster order — what a detector reports when the window is small
    /// enough not to drag the peak inward.
    fn square_corner_pixels(lo: usize, hi: usize) -> Vec<(f64, f64)> {
        let (a, b) = (lo as f64, (hi - 1) as f64);
        vec![(a, a), (b, a), (a, b), (b, b)]
    }

    /// The `(x, y)` positions of `corners`, for set comparisons.
    fn positions(corners: &[Corner]) -> Vec<(f64, f64)> {
        corners
            .iter()
            .map(|c| (c.position().x, c.position().y))
            .collect()
    }

    fn max_response<P>(map: &Image<P>) -> f32
    where
        P: SingleChannel,
        f64: From<P::Channel>,
    {
        (0..map.height())
            .flat_map(|y| (0..map.width()).map(move |x| (x, y)))
            .map(|(x, y)| f64::from(map.pixel_at(x, y).channel(0)) as f32)
            .fold(f32::NEG_INFINITY, f32::max)
    }

    /// Distance from `corner` to the nearest true square corner.
    fn distance_to_nearest(corner: &Corner, truth: &[(f64, f64); 4]) -> f64 {
        truth
            .iter()
            .map(|&(tx, ty)| {
                let p = corner.position();
                ((p.x - tx).powi(2) + (p.y - ty).powi(2)).sqrt()
            })
            .fold(f64::INFINITY, f64::min)
    }

    // ── Harris: the k invariant ─────────────────────────────────────────

    #[test]
    fn harris_accepts_the_conventional_range() {
        for k in [0.01, 0.04, 0.06, 0.2, 0.249] {
            assert_eq!(Harris::try_new(k).unwrap().k(), k);
        }
        const CLASSIC: Harris = Harris::new(0.04);
        assert_eq!(CLASSIC.k(), 0.04);
    }

    #[test]
    fn harris_try_new_rejects_a_dead_or_inverted_detector() {
        // 0 and below invert the edge penalty into a reward; 0.25 and above
        // make det − k·tr² non-positive for every tensor, so nothing can
        // ever be detected. Both are silent failures, hence errors.
        for k in [0.0, -0.04, 0.25, 0.5, f32::NAN, f32::INFINITY] {
            let err = Harris::try_new(k).unwrap_err();
            match err {
                Error::InvalidParameter(reason) => {
                    assert!(reason.contains("k"), "reason {reason:?} does not mention k");
                }
                other => panic!("expected InvalidParameter, got {other:?}"),
            }
        }
    }

    #[test]
    #[should_panic(expected = "0 < k < 0.25")]
    fn harris_new_panics_on_an_invalid_literal() {
        let _ = Harris::new(0.3);
    }

    #[test]
    fn harris_k_at_the_upper_bound_would_zero_every_response() {
        // The reason 0.25 is excluded, demonstrated on the most favourable
        // tensor there is (λ₁ = λ₂, where det = tr²/4 exactly).
        let (sxx, sxy, syy) = (1.0f32, 0.0, 1.0);
        let at_bound = CornerResponseChannel::harris(sxx, sxy, syy, 0.25);
        assert!(at_bound.abs() < 1e-6, "{at_bound}");
        assert!(Harris::new(0.249).response(sxx, sxy, syy) > 0.0);
    }

    // ── Response formulas ───────────────────────────────────────────────

    #[test]
    fn harris_response_matches_the_formula() {
        let (sxx, sxy, syy) = (5.0f32, 2.0, 3.0);
        let expected = (5.0 * 3.0 - 2.0 * 2.0) - 0.04 * (5.0 + 3.0) * (5.0 + 3.0);
        assert!((Harris::new(0.04).response(sxx, sxy, syy) - expected).abs() < 1e-6);
    }

    #[test]
    fn shi_tomasi_response_is_the_smaller_eigenvalue() {
        // Diagonal tensor: the eigenvalues are the diagonal entries.
        assert!((ShiTomasi.response(4.0f32, 0.0, 1.0) - 1.0).abs() < 1e-6);
        assert!((ShiTomasi.response(1.0f32, 0.0, 4.0) - 1.0).abs() < 1e-6);
        // [[2, 1], [1, 2]] has eigenvalues 3 and 1.
        assert!((ShiTomasi.response(2.0f32, 1.0, 2.0) - 1.0).abs() < 1e-6);
        // [[3, 4], [4, 3]] has eigenvalues 7 and −1: λ_min may be negative
        // for an arbitrary symmetric matrix, though a real structure tensor
        // is positive semi-definite.
        assert!((ShiTomasi.response(3.0f32, 4.0, 3.0) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn both_responses_reject_a_pure_edge() {
        // One zero eigenvalue: λ_min is 0 and Harris is strictly negative,
        // however large the surviving eigenvalue is.
        for strength in [1.0f32, 100.0, 1e6] {
            assert!(ShiTomasi.response(strength, 0.0, 0.0).abs() <= 1e-3 * strength);
            assert!(Harris::new(0.04).response(strength, 0.0, 0.0) < 0.0);
        }
    }

    #[test]
    fn responses_are_generic_over_f64() {
        // The same formulas at f64 precision, which is what Mono16 and up
        // accumulate in.
        // `k` is an f32 constant widened into the f64 formula, so the
        // agreement is to f32 precision in `k`, not to f64 in the result.
        assert!((Harris::new(0.04).response(5.0f64, 2.0, 3.0) - 8.44).abs() < 1e-6);
        assert!((ShiTomasi.response(2.0f64, 1.0, 2.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn shi_tomasi_keeps_precision_on_a_near_degenerate_tensor() {
        // The subtraction the closed form avoids: λ₁ ≈ λ₂ ≈ 1e8 with a tiny
        // off-diagonal. Computing λ_min as ½(tr − sqrt(tr² − 4·det)) here
        // cancels ~16 digits; the half-difference form does not.
        let (sxx, sxy, syy) = (1e8f64, 1.0, 1e8);
        let lambda_min = ShiTomasi.response(sxx, sxy, syy);
        assert!((lambda_min - (1e8 - 1.0)).abs() < 1e-3, "{lambda_min}");
    }

    #[test]
    fn a_custom_response_strategy_composes() {
        // The point of the trait: a third measure is an impl, not a fork.
        struct Noble;
        impl CornerResponse<f32> for Noble {
            fn response(&self, sxx: f32, sxy: f32, syy: f32) -> f32 {
                let trace = sxx + syy;
                if trace == 0.0 {
                    0.0
                } else {
                    2.0 * (sxx * syy - sxy * sxy) / trace
                }
            }
        }

        let image = square(24, 8, 16);
        let map: Image<MonoF32> = corner_response_map(&image, &Noble, Sigma::new(1.2));
        let peak = max_response(&map);
        assert_eq!(corner_peaks(&map, 0.3 * peak, 3).len(), 4);
    }

    // ── CornerParams ────────────────────────────────────────────────────

    #[test]
    fn corner_params_round_trip() {
        const PARAMS: CornerParams = CornerParams::new(Sigma::new(1.4), 0.01, 3);
        assert_eq!(PARAMS.window(), Sigma::new(1.4));
        assert_eq!(PARAMS.threshold(), 0.01);
        assert_eq!(PARAMS.nms_radius(), 3);

        let computed = CornerParams::try_new(Sigma::new(2.0), -1.5, 1).unwrap();
        // A negative threshold is legitimate: the Harris response is signed.
        assert_eq!(computed.threshold(), -1.5);
    }

    #[test]
    fn corner_params_try_new_rejects_a_non_finite_threshold() {
        for threshold in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let err = CornerParams::try_new(Sigma::new(1.0), threshold, 2).unwrap_err();
            match err {
                Error::InvalidParameter(reason) => assert!(
                    reason.contains("threshold"),
                    "reason {reason:?} does not mention the threshold"
                ),
                other => panic!("expected InvalidParameter, got {other:?}"),
            }
        }
    }

    #[test]
    fn corner_params_try_new_rejects_a_zero_radius() {
        let err = CornerParams::try_new(Sigma::new(1.0), 0.01, 0).unwrap_err();
        match err {
            Error::InvalidParameter(reason) => assert!(
                reason.contains("nms_radius"),
                "reason {reason:?} does not mention nms_radius"
            ),
            other => panic!("expected InvalidParameter, got {other:?}"),
        }
    }

    #[test]
    #[should_panic(expected = "threshold must be finite")]
    fn corner_params_new_panics_on_a_nan_literal() {
        let _ = CornerParams::new(Sigma::new(1.0), f32::NAN, 2);
    }

    #[test]
    #[should_panic(expected = "nms_radius must be at least 1")]
    fn corner_params_new_panics_on_a_zero_radius_literal() {
        let _ = CornerParams::new(Sigma::new(1.0), 0.01, 0);
    }

    // ── StructureTensor ─────────────────────────────────────────────────

    #[test]
    fn structure_tensor_of_a_vertical_edge_is_all_xx() {
        let image: Image<MonoF32> =
            Image::generate(16, 16, |x, _| MonoF32::new(if x < 8 { 0.0 } else { 1.0 }));
        let tensor = StructureTensor::from_gradients(
            &sobel_x(&image, &Clamp),
            &sobel_y(&image, &Clamp),
            Sigma::new(1.0),
        )
        .unwrap();

        assert_eq!(tensor.size(), image.size());
        // Sobel's positive-lobe gain is 4, so a unit step gives |Gx| = 4 and
        // Gx² = 16 before the (brightness-preserving) window.
        assert!(tensor.xx().pixel_at(8, 8).value() > 1.0);
        assert!(tensor.yy().pixel_at(8, 8).value().abs() < 1e-6);
        assert!(tensor.xy().pixel_at(8, 8).value().abs() < 1e-6);
    }

    #[test]
    fn structure_tensor_of_a_diagonal_edge_has_an_off_diagonal_term() {
        let image: Image<MonoF32> =
            Image::generate(16, 16, |x, y| MonoF32::new(if x < y { 0.0 } else { 1.0 }));
        let tensor = StructureTensor::from_gradients(
            &sobel_x(&image, &Clamp),
            &sobel_y(&image, &Clamp),
            Sigma::new(1.0),
        )
        .unwrap();
        // Gx and Gy have opposite signs along this edge, so Sxy < 0 — the
        // term a "sum of squares" tensor could not represent.
        assert!(tensor.xy().pixel_at(8, 8).value() < 0.0);
    }

    #[test]
    fn structure_tensor_reports_a_gradient_size_mismatch() {
        let gx: Image<MonoF32> = Image::zero(8, 8);
        let gy: Image<MonoF32> = Image::zero(8, 4);
        let err = StructureTensor::from_gradients(&gx, &gy, Sigma::new(1.0)).unwrap_err();
        assert_eq!(
            err,
            Error::SizeMismatch {
                expected: Size::new(8, 8),
                actual: Size::new(8, 4),
            }
        );
    }

    #[test]
    fn structure_tensor_from_smoothed_takes_the_images_as_given() {
        // The escape hatch: a caller-supplied (here, constant) window.
        let tensor = StructureTensor::from_smoothed(
            Image::fill(4, 4, MonoF32::new(4.0)),
            Image::fill(4, 4, MonoF32::new(0.0)),
            Image::fill(4, 4, MonoF32::new(1.0)),
        )
        .unwrap();
        assert_eq!(tensor.size(), Size::new(4, 4));
        // Eigenvalues 4 and 1 everywhere ⇒ λ_min = 1.
        let response = tensor.response(&ShiTomasi);
        assert!((response.pixel_at(2, 2).value() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn structure_tensor_from_smoothed_reports_a_size_mismatch() {
        let err = StructureTensor::from_smoothed(
            Image::<MonoF32>::zero(4, 4),
            Image::<MonoF32>::zero(4, 4),
            Image::<MonoF32>::zero(4, 3),
        )
        .unwrap_err();
        assert_eq!(
            err,
            Error::SizeMismatch {
                expected: Size::new(4, 4),
                actual: Size::new(4, 3),
            }
        );
    }

    // ── corner_peaks ────────────────────────────────────────────────────

    #[test]
    fn peaks_are_returned_in_raster_order() {
        let response: Image<MonoF32> = Image::generate(9, 9, |x, y| {
            MonoF32::new(match (x, y) {
                (6, 2) => 0.4, // weaker, but earlier in raster order
                (2, 6) => 0.9,
                _ => 0.0,
            })
        });
        let corners = corner_peaks(&response, 0.1, 2);
        let positions: Vec<(f64, f64)> = corners
            .iter()
            .map(|c| (c.position().x, c.position().y))
            .collect();
        assert_eq!(positions, [(6.0, 2.0), (2.0, 6.0)]);
        assert_eq!(corners[0].response(), 0.4);
    }

    #[test]
    fn peaks_below_the_threshold_are_dropped() {
        let response: Image<MonoF32> = Image::generate(9, 9, |x, y| {
            MonoF32::new(if (x, y) == (4, 4) { 0.05 } else { 0.0 })
        });
        assert!(corner_peaks(&response, 0.1, 2).is_empty());
        // The threshold is inclusive.
        assert_eq!(corner_peaks(&response, 0.05, 2).len(), 1);
    }

    #[test]
    fn a_plateau_yields_exactly_one_peak() {
        // Four exactly tied pixels: `>=` everywhere would report all four,
        // `>` everywhere none. The raster-first one wins.
        let response: Image<MonoF32> = Image::generate(9, 9, |x, y| {
            MonoF32::new(if (3..5).contains(&x) && (3..5).contains(&y) {
                1.0
            } else {
                0.0
            })
        });
        let corners = corner_peaks(&response, 0.5, 2);
        assert_eq!(corners.len(), 1);
        assert_eq!(corners[0].position(), CoordinateF64::new(3.0, 3.0));
    }

    #[test]
    fn a_uniform_response_map_yields_a_single_peak() {
        // Every pixel ties and every pixel has an earlier tied neighbour
        // within the radius, except the very first — so the whole image is
        // one group and reports one corner, not 64 and not none.
        let response: Image<MonoF32> = Image::fill(8, 8, MonoF32::new(1.0));
        let corners = corner_peaks(&response, 0.5, 2);
        assert_eq!(corners.len(), 1);
        assert_eq!(corners[0].position(), CoordinateF64::new(0.0, 0.0));
    }

    #[test]
    fn tied_groups_further_apart_than_the_radius_are_separate_peaks() {
        // The other half of the plateau rule: ties are grouped by the same
        // radius that separates ordinary peaks, so two equal-valued blobs
        // five pixels apart report one corner each.
        let response: Image<MonoF32> = Image::generate(12, 3, |x, y| {
            let tied = (y == 1) && ((2..4).contains(&x) || (8..10).contains(&x));
            MonoF32::new(if tied { 1.0 } else { 0.0 })
        });
        let corners = corner_peaks(&response, 0.5, 2);
        assert_eq!(positions(&corners), [(2.0, 1.0), (8.0, 1.0)]);
    }

    #[test]
    fn the_suppression_radius_sets_the_minimum_separation() {
        // Two peaks four pixels apart: kept at radius 3, merged at radius 4.
        let response: Image<MonoF32> = Image::generate(12, 3, |x, y| {
            MonoF32::new(match (x, y) {
                (3, 1) => 1.0,
                (7, 1) => 0.8,
                _ => 0.0,
            })
        });
        assert_eq!(corner_peaks(&response, 0.1, 3).len(), 2);
        let merged = corner_peaks(&response, 0.1, 4);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].position(), CoordinateF64::new(3.0, 1.0));
    }

    #[test]
    fn a_peak_against_the_border_is_reported() {
        // The window is clipped, not skipped: a corner at (0, 0) survives.
        let response: Image<MonoF32> = Image::generate(6, 6, |x, y| {
            MonoF32::new(if (x, y) == (0, 0) { 1.0 } else { 0.0 })
        });
        let corners = corner_peaks(&response, 0.5, 2);
        assert_eq!(corners.len(), 1);
        assert_eq!(corners[0].position(), CoordinateF64::new(0.0, 0.0));
    }

    #[test]
    fn a_nan_response_neither_wins_nor_survives() {
        let response: Image<MonoF32> = Image::generate(7, 7, |x, y| {
            MonoF32::new(match (x, y) {
                (3, 3) => f32::NAN,
                (5, 5) => 1.0,
                _ => 0.0,
            })
        });
        let corners = corner_peaks(&response, 0.5, 1);
        // The NaN pixel fails its own threshold test, and the genuine peak
        // is far enough away to be unaffected.
        let positions: Vec<(f64, f64)> = corners
            .iter()
            .map(|c| (c.position().x, c.position().y))
            .collect();
        assert_eq!(positions, [(5.0, 5.0)]);
    }

    #[test]
    fn a_nan_neighbour_suppresses_a_peak() {
        let response: Image<MonoF32> = Image::generate(7, 7, |x, y| {
            MonoF32::new(match (x, y) {
                (3, 3) => 1.0,
                (4, 3) => f32::NAN,
                _ => 0.0,
            })
        });
        assert!(corner_peaks(&response, 0.5, 1).is_empty());
    }

    #[test]
    fn an_empty_response_map_yields_no_peaks() {
        let response: Image<MonoF32> = Image::fill(5, 5, MonoF32::new(0.0));
        assert!(corner_peaks(&response, 0.5, 2).is_empty());
    }

    // ── detect_corners on synthetic geometry ────────────────────────────

    #[test]
    fn a_square_has_four_corners() {
        let image = square(24, 8, 16);
        let method = Harris::new(0.04);
        let map: Image<MonoF32> = corner_response_map(&image, &method, Sigma::new(1.0));
        let params = CornerParams::try_new(Sigma::new(1.0), 0.2 * max_response(&map), 3).unwrap();

        // Exactly the four corner pixels, in raster order, and nothing along
        // the four edges between them.
        let corners = detect_corners(&image, &method, params);
        assert_eq!(positions(&corners), square_corner_pixels(8, 16));
        // Each is within half a pixel of the geometric corner, which sits
        // between the last outside and first inside sample.
        let truth = square_corners(8, 16);
        assert!(
            corners
                .iter()
                .all(|c| distance_to_nearest(c, &truth) <= 0.75),
            "{corners:?}"
        );
    }

    #[test]
    fn shi_tomasi_finds_the_same_four_corners() {
        // Same geometry, same answer: the two measures disagree on how
        // strongly to score a corner, not on where it is.
        let image = square(24, 8, 16);
        let map: Image<MonoF32> = corner_response_map(&image, &ShiTomasi, Sigma::new(1.0));
        let params = CornerParams::try_new(Sigma::new(1.0), 0.3 * max_response(&map), 3).unwrap();

        let corners = detect_corners(&image, &ShiTomasi, params);
        assert_eq!(positions(&corners), square_corner_pixels(8, 16));

        let harris_map: Image<MonoF32> =
            corner_response_map(&image, &Harris::new(0.04), Sigma::new(1.0));
        let harris_params =
            CornerParams::try_new(Sigma::new(1.0), 0.2 * max_response(&harris_map), 3).unwrap();
        assert_eq!(
            positions(&detect_corners(&image, &Harris::new(0.04), harris_params)),
            positions(&corners)
        );
    }

    #[test]
    fn a_larger_window_drags_the_peak_inward() {
        // Worth pinning down, because it is the honest limit of pixel-level
        // localization: the window averages the two edges meeting at a
        // corner, and the average is strongest slightly *inside* it. At
        // σ = 1.0 the peak is on the corner pixel; by σ = 1.6 it has moved a
        // pixel in along both axes. The answer to that is a refinement step
        // reading the gradient field, not a smaller window — and a
        // refinement step is deliberately not this function's job.
        let image = square(24, 8, 16);
        let corners_of = |sigma: Sigma| {
            let map: Image<MonoF32> = corner_response_map(&image, &ShiTomasi, sigma);
            let params = CornerParams::try_new(sigma, 0.3 * max_response(&map), 3).unwrap();
            positions(&detect_corners(&image, &ShiTomasi, params))
        };

        assert_eq!(corners_of(Sigma::new(1.0)), square_corner_pixels(8, 16));
        assert_eq!(
            corners_of(Sigma::new(1.6)),
            vec![(9.0, 9.0), (14.0, 9.0), (9.0, 14.0), (14.0, 14.0)]
        );
    }

    #[test]
    fn a_straight_edge_has_no_corners() {
        // The deterministic negative case: a step edge is where a naïve
        // gradient-magnitude "corner" detector fires hardest.
        let image: Image<MonoF32> =
            Image::generate(24, 24, |x, _| MonoF32::new(if x < 12 { 0.0 } else { 1.0 }));
        let params = CornerParams::new(Sigma::new(1.2), 1e-4, 3);
        assert!(detect_corners(&image, &Harris::new(0.04), params).is_empty());
        assert!(detect_corners(&image, &ShiTomasi, params).is_empty());
    }

    #[test]
    fn a_flat_field_has_no_corners() {
        let image = Image::fill(16, 16, MonoF32::new(0.5));
        let params = CornerParams::new(Sigma::new(1.0), 1e-6, 2);
        assert!(detect_corners(&image, &Harris::new(0.04), params).is_empty());
    }

    #[test]
    fn an_l_junction_has_one_corner() {
        // Two half-planes meeting at (12, 12): exactly one corner, unlike
        // the square's four.
        let image: Image<MonoF32> = Image::generate(24, 24, |x, y| {
            MonoF32::new(if x >= 12 && y >= 12 { 1.0 } else { 0.0 })
        });
        let map: Image<MonoF32> = corner_response_map(&image, &ShiTomasi, Sigma::new(1.0));
        let params = CornerParams::try_new(Sigma::new(1.0), 0.4 * max_response(&map), 4).unwrap();

        let corners = detect_corners(&image, &ShiTomasi, params);
        assert_eq!(positions(&corners), [(12.0, 12.0)], "{corners:?}");
    }

    #[test]
    fn the_response_is_invariant_under_a_quarter_turn() {
        // A 90° rotation maps Gx → Gy and Gy → −Gx, so Sxx and Syy swap and
        // Sxy changes sign — both det and tr are unchanged, and so is every
        // response built from them. This is the isotropy the Gaussian
        // window exists to preserve.
        let image = square(24, 7, 17);
        let rotated: Image<MonoF32> = rotate_90(&image);

        let map: Image<MonoF32> = corner_response_map(&image, &Harris::new(0.04), Sigma::new(1.2));
        let rotated_map: Image<MonoF32> =
            corner_response_map(&rotated, &Harris::new(0.04), Sigma::new(1.2));

        let scale = max_response(&map);
        assert!(scale > 0.0);
        for y in 0..24 {
            for x in 0..24 {
                // rotate_90 is clockwise: (x, y) ↦ (h − 1 − y, x).
                let here = map.pixel_at(x, y).value();
                let there = rotated_map.pixel_at(23 - y, x).value();
                assert!(
                    (here - there).abs() <= 1e-3 * scale,
                    "({x},{y}): {here} vs {there}"
                );
            }
        }
    }

    #[test]
    fn detect_corners_is_the_documented_composition() {
        // The orchestrator must be exactly response map + peaks, so that
        // rebuilding it by hand is not a different detector.
        let image = square(24, 8, 16);
        let method = Harris::new(0.05);
        let params = CornerParams::new(Sigma::new(1.1), 1.0, 3);

        let staged = {
            let map: Image<MonoF32> = corner_response_map(&image, &method, params.window());
            corner_peaks(&map, params.threshold(), params.nms_radius())
        };
        assert_eq!(detect_corners(&image, &method, params), staged);
        assert!(!staged.is_empty());
    }

    // ── Pixel-type genericity ───────────────────────────────────────────

    #[test]
    fn accepts_integer_input() {
        // Mono8 accumulates in MonoF32; the response is 255⁴ larger than the
        // same picture in 0.0..=1.0 floats, which is why thresholds are
        // calibrated and not guessed.
        let image: Image<Mono8> = Image::generate(24, 24, |x, y| {
            let inside = (8..16).contains(&x) && (8..16).contains(&y);
            Mono8::new(if inside { 255 } else { 0 })
        });
        let map: Image<MonoF32> = corner_response_map(&image, &ShiTomasi, Sigma::new(1.2));
        let params = CornerParams::try_new(Sigma::new(1.2), 0.3 * max_response(&map), 3).unwrap();
        assert_eq!(detect_corners(&image, &ShiTomasi, params).len(), 4);
    }

    #[test]
    fn accepts_sixteen_bit_input() {
        // `Mono16` accumulates in `MonoF32` (`Mono32` / `Mono64` are the
        // integer types that reach for `MonoF64`). At full 16-bit contrast a
        // Harris response is ≈ (4·65535)⁴ ≈ 5·10²¹ — well inside f32's range,
        // but a reminder that the response is a fourth power.
        let image: Image<Mono16> = Image::generate(24, 24, |x, y| {
            let inside = (8..16).contains(&x) && (8..16).contains(&y);
            Mono16::new(if inside { 65535 } else { 0 })
        });
        let map: Image<MonoF32> = corner_response_map(&image, &Harris::new(0.04), Sigma::new(1.2));
        let params = CornerParams::try_new(Sigma::new(1.2), 0.2 * max_response(&map), 3).unwrap();
        assert_eq!(detect_corners(&image, &Harris::new(0.04), params).len(), 4);
    }

    #[test]
    fn accepts_f64_float_input() {
        let image: Image<MonoF64> = Image::generate(24, 24, |x, y| {
            let inside = (8..16).contains(&x) && (8..16).contains(&y);
            MonoF64::new(if inside { 1.0 } else { 0.0 })
        });
        let map: Image<MonoF64> = corner_response_map(&image, &ShiTomasi, Sigma::new(1.2));
        let params = CornerParams::try_new(Sigma::new(1.2), 0.3 * max_response(&map), 3).unwrap();
        assert_eq!(detect_corners(&image, &ShiTomasi, params).len(), 4);
    }

    // ── Pyramid levels ──────────────────────────────────────────────────

    #[test]
    fn detection_on_the_base_level_is_the_identity_lift() {
        let image = square(24, 8, 16);
        let level = ScaledImage::new(
            image.clone(),
            PixelDistance::new(1.0),
            CoordinateF64::new(0.0, 0.0),
            Sigma::new(0.5),
        );
        let params = CornerParams::new(Sigma::new(1.2), 1.0, 3);

        assert_eq!(
            detect_corners_in_level(&level, &ShiTomasi, params),
            detect_corners(&image, &ShiTomasi, params)
        );
    }

    #[test]
    fn detection_on_a_coarse_level_reports_base_coordinates() {
        let base = square(48, 16, 32);
        let level = ScaledImage::new(
            pyr_down(&base),
            PixelDistance::new(2.0),
            CoordinateF64::new(0.0, 0.0),
            Sigma::new(1.0),
        );
        let map: Image<MonoF32> =
            corner_response_map(level.as_image(), &ShiTomasi, Sigma::new(1.0));
        let params = CornerParams::try_new(Sigma::new(1.0), 0.3 * max_response(&map), 2).unwrap();

        let corners = detect_corners_in_level(&level, &ShiTomasi, params);
        assert_eq!(corners.len(), 4, "{corners:?}");

        let truth = square_corners(16, 32);
        for corner in &corners {
            // Every coordinate is even: proof the lift multiplied by the
            // level's pixel distance rather than reporting local coordinates.
            let p = corner.position();
            assert!(p.x % 2.0 == 0.0 && p.y % 2.0 == 0.0, "{corner:?}");
            // Localization is quantized to the level's 2-pixel grid, and
            // `pyr_down`'s own blur adds the inward bias
            // `a_larger_window_drags_the_peak_inward` pins down — so two base
            // pixels of error is the best this level can do.
            assert!(distance_to_nearest(corner, &truth) <= 4.0, "{corner:?}");
        }
    }

    #[test]
    fn a_pyramid_can_be_swept_level_by_level() {
        let base = square(32, 8, 24);
        let levels = vec![
            ScaledImage::new(
                base.clone(),
                PixelDistance::new(1.0),
                CoordinateF64::new(0.0, 0.0),
                Sigma::new(0.5),
            ),
            ScaledImage::new(
                pyr_down(&base),
                PixelDistance::new(2.0),
                CoordinateF64::new(0.0, 0.0),
                Sigma::new(1.0),
            ),
        ];
        let pyramid = Pyramid::try_from_levels(levels).unwrap();
        let params = CornerParams::new(Sigma::new(1.0), 0.5, 2);

        let corners: Vec<Corner> = pyramid
            .iter()
            .flat_map(|level| detect_corners_in_level(level, &ShiTomasi, params))
            .collect();

        // Both levels see the same square, so each physical corner is found
        // twice — deduplication across levels is the caller's policy.
        assert!(corners.len() >= 8, "{corners:?}");
        let truth = square_corners(8, 24);
        assert!(
            corners
                .iter()
                .all(|c| distance_to_nearest(c, &truth) <= 4.0),
            "{corners:?}"
        );
    }

    #[test]
    fn a_level_with_an_origin_offset_lifts_through_it() {
        // An area-averaging reduction shifts the grid by half a base pixel;
        // the lift must carry that term rather than just scaling.
        let base = square(48, 16, 32);
        let coarse = pyr_down(&base);
        let params = CornerParams::new(Sigma::new(1.0), 0.5, 2);

        let unshifted = ScaledImage::new(
            coarse.clone(),
            PixelDistance::new(2.0),
            CoordinateF64::new(0.0, 0.0),
            Sigma::new(1.0),
        );
        let shifted = ScaledImage::new(
            coarse,
            PixelDistance::new(2.0),
            CoordinateF64::new(0.5, 0.5),
            Sigma::new(1.0),
        );

        let a = detect_corners_in_level(&unshifted, &ShiTomasi, params);
        let b = detect_corners_in_level(&shifted, &ShiTomasi, params);
        assert!(!a.is_empty());
        assert_eq!(a.len(), b.len());
        for (unshifted, shifted) in a.iter().zip(&b) {
            assert_eq!(shifted.position().x - unshifted.position().x, 0.5);
            assert_eq!(shifted.position().y - unshifted.position().y, 0.5);
        }
    }
}

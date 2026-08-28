//! Intensity-weighted image moments, and the invariants derived from them.
//!
//! Three named stages, each a distinct mathematical object rather than a
//! convenience wrapper on the one before it:
//!
//! | Stage | Type | Invariant under |
//! |---|---|---|
//! | Raw `m_pq` | [`ImageMoments`] | nothing |
//! | Central `μ_pq` | [`CentralMoments`] | translation |
//! | Normalized `η_pq` | [`NormalizedMoments`] | translation and scale |
//! | Hu `h₁..h₇` | [`NormalizedMoments::hu`] | translation, scale and rotation |
//!
//! Each step can fail to exist, and says so with [`Option`]: an image whose
//! total intensity is zero has no centroid to take moments about, and one
//! whose total intensity is negative has no real scale normalisation. Those
//! are absences rather than errors, so nothing here returns a `Result`.

use crate::CoordinateF64;
use crate::common::AxialOrientation;
use crate::image::RasterImage;
use crate::pixel::SingleChannel;

use super::summary::StatisticsChannel;

/// Raw intensity-weighted moments of an image, up to third order.
///
/// `m_pq = Σ_y Σ_x xᵖ yᵈ I(x, y)`, with `x` and `y` the pixel-centre
/// coordinates and `I` the channel value in its own units (a `Mono8` 255
/// contributes 255, not 1.0). The ten values below are every `m_pq` with
/// `p + q ≤ 3`, which is the set the derived quantities need: `m00` is total
/// intensity, first order gives the centroid, second order the orientation and
/// eccentricity of the equivalent ellipse, and third order the skew terms the
/// Hu invariants need.
///
/// # This is not the blob moment
///
/// [`BlobMeasurements`](crate::analyze::components::BlobMeasurements) reports
/// moments of a labelled *region*, weighting every member pixel by 1 and
/// ignoring everything outside it. These are moments of the whole image,
/// weighted by intensity. Use the blob form to describe a segmented object and
/// this form to describe a distribution of brightness. The two also differ in
/// normalisation: `BlobMeasurements::central_moments` divides by area, so its
/// values are variances, whereas the `μ_pq` here are unnormalized sums, which
/// is the standard definition and what the Hu invariants are built on.
///
/// # `NaN` propagates
///
/// A single `NaN` sample makes every moment `NaN`, and every derived quantity
/// with it. That is deliberate: a moment is a sum over all pixels, so skipping
/// one would report a figure for an image that was not measured. To find out
/// whether an image is clean first, read
/// [`ChannelStatistics::nan_count`](super::ChannelStatistics::nan_count).
///
/// # Example
///
/// ```
/// use fovea::analyze::statistics::image_moments;
/// use fovea::image::Image;
/// use fovea::pixel::Mono8;
///
/// // A 4x4 bright block in the top-left of a 16x16 frame. Its centre of
/// // brightness is the block's centre, at (1.5, 1.5).
/// let image = Image::generate(16, 16, |x, y| {
///     Mono8::new(if x < 4 && y < 4 { 100 } else { 0 })
/// });
///
/// let moments = image_moments(&image);
/// assert_eq!(moments.m00, 16.0 * 100.0);
///
/// let centroid = moments.centroid().expect("the block carries intensity");
/// assert!((centroid.x - 1.5).abs() < 1e-12);
/// assert!((centroid.y - 1.5).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ImageMoments {
    /// `m00`: total intensity, the zeroth moment.
    pub m00: f64,
    /// `m10`: `Σ x·I`.
    pub m10: f64,
    /// `m01`: `Σ y·I`.
    pub m01: f64,
    /// `m20`: `Σ x²·I`.
    pub m20: f64,
    /// `m11`: `Σ x·y·I`.
    pub m11: f64,
    /// `m02`: `Σ y²·I`.
    pub m02: f64,
    /// `m30`: `Σ x³·I`.
    pub m30: f64,
    /// `m21`: `Σ x²·y·I`.
    pub m21: f64,
    /// `m12`: `Σ x·y²·I`.
    pub m12: f64,
    /// `m03`: `Σ y³·I`.
    pub m03: f64,
}

impl ImageMoments {
    /// Centre of brightness, or [`None`] when total intensity is exactly zero.
    ///
    /// `(m10 / m00, m01 / m00)`, in pixel-centre coordinates. `m00 == 0` is
    /// reachable both from an all-black image and from a signed-channel image
    /// whose positive and negative intensity cancel, and in neither case is
    /// there a centre to report.
    #[must_use]
    pub fn centroid(&self) -> Option<CoordinateF64> {
        (self.m00 != 0.0).then(|| CoordinateF64::new(self.m10 / self.m00, self.m01 / self.m00))
    }

    /// Moments taken about the centroid, so translating the image does not
    /// change them. [`None`] exactly when [`centroid`](Self::centroid) is.
    ///
    /// The raw sums are formed first and the centroid subtracted afterwards,
    /// which is the standard identity rather than a second pass over the
    /// pixels. It does mean the third-order terms subtract large, nearly-equal
    /// numbers on an image whose brightness sits far from the origin: for a
    /// 12-megapixel frame the `m30` term reaches about `10¹⁷`, so `μ30` keeps
    /// roughly nine of `f64`'s sixteen digits. Second-order values, and
    /// therefore [`orientation`](CentralMoments::orientation) and
    /// [`eccentricity`](CentralMoments::eccentricity), are unaffected at those
    /// sizes.
    #[must_use]
    pub fn central_moments(&self) -> Option<CentralMoments> {
        let centroid = self.centroid()?;
        let (x, y) = (centroid.x, centroid.y);

        Some(CentralMoments {
            mu00: self.m00,
            mu20: self.m20 - x * self.m10,
            mu11: self.m11 - x * self.m01,
            mu02: self.m02 - y * self.m01,
            mu30: self.m30 - 3.0 * x * self.m20 + 2.0 * x * x * self.m10,
            mu21: self.m21 - 2.0 * x * self.m11 - y * self.m20 + 2.0 * x * x * self.m01,
            mu12: self.m12 - 2.0 * y * self.m11 - x * self.m02 + 2.0 * y * y * self.m10,
            mu03: self.m03 - 3.0 * y * self.m02 + 2.0 * y * y * self.m01,
        })
    }
}

/// Moments about the centroid: unchanged by translating the image.
///
/// `μ_pq = Σ (x − x̄)ᵖ (y − ȳ)ᵈ I(x, y)`. `μ10` and `μ01` are zero by
/// construction and are not stored. `μ00` is carried because the scale
/// normalisation in [`normalized`](Self::normalized) needs it.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CentralMoments {
    /// `μ00`, which equals `m00`: total intensity.
    pub mu00: f64,
    /// `μ20`: spread along x.
    pub mu20: f64,
    /// `μ11`: the xy covariance term.
    pub mu11: f64,
    /// `μ02`: spread along y.
    pub mu02: f64,
    /// `μ30`: skew along x.
    pub mu30: f64,
    /// `μ21`.
    pub mu21: f64,
    /// `μ12`.
    pub mu12: f64,
    /// `μ03`: skew along y.
    pub mu03: f64,
}

impl CentralMoments {
    /// Orientation of the major axis of the equivalent ellipse.
    ///
    /// `½·atan2(2·μ11, μ20 − μ02)`, in `(−π/2, π/2]` measured from the +x axis
    /// **in image (y-down) coordinates**, so positive angles rotate toward +y,
    /// which is downward on screen. Note that flip against math-convention
    /// plots or the sign reads backwards. A rotationally symmetric
    /// distribution returns `0`.
    ///
    /// The return type is *axial* rather than directed, because an ellipse's
    /// major axis has no head and no tail: `+80°` and `−100°` are the same
    /// axis. Compare two of them with
    /// [`AxialOrientation::signed_difference`], which wraps at π; a raw
    /// subtraction would read those two as `160°` apart instead of `20°`.
    #[must_use]
    pub fn orientation(&self) -> AxialOrientation {
        axis_orientation(self.mu20, self.mu02, self.mu11)
    }

    /// Eccentricity of the equivalent ellipse, in `[0, 1]`.
    ///
    /// `0` is a perfect circle and the value rises toward `1` as the
    /// distribution becomes more line-like. `1` is attained exactly, not
    /// merely approached: a straight axis-aligned run of brightness has
    /// `λ₂ = 0` and therefore `ecc = 1.0`. Derived from the eigenvalues
    /// `λ₁ ≥ λ₂ ≥ 0` of the second-moment matrix as `√(1 − λ₂/λ₁)`. A
    /// degenerate distribution (`λ₁ = 0`, e.g. a single bright pixel) returns
    /// `0` rather than `NaN`.
    #[must_use]
    pub fn eccentricity(&self) -> f64 {
        axis_eccentricity(self.mu20, self.mu02, self.mu11)
    }

    /// Scale-normalized central moments, or [`None`] when `μ00 <= 0`.
    ///
    /// `η_pq = μ_pq / μ00^(1 + (p + q) / 2)`. Scaling the image by `s` scales
    /// `μ_pq` by `s^(p + q + 2)` and `μ00` by `s²`, so the ratio is
    /// invariant. The exponent is fractional for odd orders, which is why a
    /// non-positive `μ00` has no normalisation rather than a negative one: an
    /// all-black image, or a signed-channel image whose intensity cancels or
    /// is net negative.
    #[must_use]
    pub fn normalized(&self) -> Option<NormalizedMoments> {
        // `NaN` is excluded explicitly: it is neither positive nor
        // non-positive, and a normalisation of it would be a plausible-looking
        // `NaN` rather than a reported absence.
        if self.mu00.is_nan() || self.mu00 <= 0.0 {
            return None;
        }
        // μ00^2 for second order, μ00^2.5 for third.
        let second = self.mu00 * self.mu00;
        let third = second * self.mu00.sqrt();

        Some(NormalizedMoments {
            eta20: self.mu20 / second,
            eta11: self.mu11 / second,
            eta02: self.mu02 / second,
            eta30: self.mu30 / third,
            eta21: self.mu21 / third,
            eta12: self.mu12 / third,
            eta03: self.mu03 / third,
        })
    }
}

/// Central moments divided out by scale: unchanged by translating *or*
/// resizing the image.
///
/// The last stage before [`hu`](Self::hu), which adds rotation invariance on
/// top. Zeroth and first order are omitted: `η00` is 1 by construction and
/// `η10` and `η01` are 0.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct NormalizedMoments {
    /// `η20`.
    pub eta20: f64,
    /// `η11`.
    pub eta11: f64,
    /// `η02`.
    pub eta02: f64,
    /// `η30`.
    pub eta30: f64,
    /// `η21`.
    pub eta21: f64,
    /// `η12`.
    pub eta12: f64,
    /// `η03`.
    pub eta03: f64,
}

impl NormalizedMoments {
    /// Hu's seven moment invariants, `[h₁, …, h₇]`.
    ///
    /// Combinations of the normalized central moments chosen so that they do
    /// not change when the image is translated, uniformly scaled or rotated,
    /// which makes them a compact shape signature: two views of the same
    /// object at different positions, sizes and angles produce nearly the same
    /// seven numbers.
    ///
    /// Two properties worth knowing before comparing them:
    ///
    /// - **`h₇` is skew-invariant, not reflection-invariant.** It changes sign
    ///   under a mirror image, while `h₁..h₆` do not. That is the standard
    ///   definition, and it is what makes `h₇` the term that distinguishes a
    ///   shape from its mirror.
    /// - **The magnitudes span many orders.** `h₁` is around `10⁻¹` for
    ///   ordinary shapes and `h₇` around `10⁻¹²`. Comparing raw values weights
    ///   `h₁` almost exclusively; the usual remedy is to compare
    ///   `sign(hᵢ)·log|hᵢ|`, which this method deliberately does not do for
    ///   you, since the choice belongs to the caller.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::analyze::statistics::image_moments;
    /// use fovea::image::Image;
    /// use fovea::pixel::Mono8;
    ///
    /// // The same L-shape, drawn twice at different positions and scales.
    /// let small = Image::generate(24, 24, |x, y| {
    ///     let inside = (2..6).contains(&x) && (2..12).contains(&y)
    ///         || (2..12).contains(&x) && (8..12).contains(&y);
    ///     Mono8::new(if inside { 255 } else { 0 })
    /// });
    /// let large = Image::generate(48, 48, |x, y| {
    ///     let inside = (20..28).contains(&x) && (20..40).contains(&y)
    ///         || (20..40).contains(&x) && (32..40).contains(&y);
    ///     Mono8::new(if inside { 255 } else { 0 })
    /// });
    ///
    /// let hu = |image: &Image<Mono8>| {
    ///     image_moments(image)
    ///         .central_moments()
    ///         .and_then(|c| c.normalized())
    ///         .map(|n| n.hu())
    ///         .unwrap()
    /// };
    ///
    /// let (a, b) = (hu(&small), hu(&large));
    /// // The first two invariants agree closely despite the 2x scale and the
    /// // move; rasterisation keeps them from matching exactly.
    /// assert!((a[0] - b[0]).abs() < 0.01, "{} vs {}", a[0], b[0]);
    /// assert!((a[1] - b[1]).abs() < 0.01, "{} vs {}", a[1], b[1]);
    /// ```
    #[must_use]
    pub fn hu(&self) -> [f64; 7] {
        let (n20, n11, n02) = (self.eta20, self.eta11, self.eta02);
        let (n30, n21, n12, n03) = (self.eta30, self.eta21, self.eta12, self.eta03);

        // The four recurring groups, named once so the seven expressions
        // below read as the textbook forms rather than as nested arithmetic.
        let sum_x = n30 + n12;
        let sum_y = n21 + n03;
        let skew_x = n30 - 3.0 * n12;
        let skew_y = 3.0 * n21 - n03;
        let (sum_x2, sum_y2) = (sum_x * sum_x, sum_y * sum_y);

        [
            n20 + n02,
            (n20 - n02) * (n20 - n02) + 4.0 * n11 * n11,
            skew_x * skew_x + skew_y * skew_y,
            sum_x2 + sum_y2,
            skew_x * sum_x * (sum_x2 - 3.0 * sum_y2) + skew_y * sum_y * (3.0 * sum_x2 - sum_y2),
            (n20 - n02) * (sum_x2 - sum_y2) + 4.0 * n11 * sum_x * sum_y,
            skew_y * sum_x * (sum_x2 - 3.0 * sum_y2) - skew_x * sum_y * (3.0 * sum_x2 - sum_y2),
        ]
    }
}

/// Orientation of the major axis of a second-moment matrix.
///
/// Shared by [`CentralMoments::orientation`] and
/// [`BlobMeasurements::orientation`](crate::analyze::components::BlobMeasurements::orientation),
/// which differ only in whether their moments were divided by area. The
/// half-angle `atan2` reads the same either way, since a uniform positive
/// scale on all three arguments cancels.
#[inline]
pub(crate) fn axis_orientation(mu20: f64, mu02: f64, mu11: f64) -> AxialOrientation {
    AxialOrientation::from_half_atan2(2.0 * mu11, mu20 - mu02)
}

/// Eccentricity of the ellipse equivalent to a second-moment matrix, in
/// `[0, 1]`.
///
/// Shared for the same reason as [`axis_orientation`], and scale-free for the
/// same reason: it is a ratio of the two eigenvalues.
#[inline]
pub(crate) fn axis_eccentricity(mu20: f64, mu02: f64, mu11: f64) -> f64 {
    let average = 0.5 * (mu20 + mu02);
    let difference = 0.5 * (mu20 - mu02);
    let discriminant = (difference * difference + mu11 * mu11).sqrt();
    let larger = average + discriminant;
    let smaller = average - discriminant;
    if larger <= 0.0 {
        return 0.0;
    }
    // The clamp guards float error at both extremes: a near-circular blob
    // can put the ratio a hair above 1, and a near-collinear one can push
    // `smaller` a hair below 0 through cancellation, which would take the
    // result above the documented 1 (measured 1.000_000_19 at x = 65 535).
    (1.0 - smaller / larger).clamp(0.0, 1.0).sqrt()
}

/// Raw intensity-weighted moments of a single-channel image, up to third
/// order.
///
/// Single-channel by a compile-time bound rather than a documented
/// precondition: "the centre of brightness of an RGB image" has no one
/// meaning, so the choice of channel belongs upstream of this function.
/// Convert with a named strategy first
/// ([`Luminance`](crate::transform::Luminance) for a perceptual grey, or
/// [`ImagePlanes::plane`](crate::image::ImagePlanes::plane) for one component
/// as it stands) and the choice is recorded in the calling code.
///
/// # Cost
///
/// One pass, `O(width · height)`, with four accumulators per row and ten adds
/// per row rather than per pixel: `Σ xᵈ yᵈ I` factors into a row sum times a
/// power of `y`, so the `y` powers are computed once per scan line.
///
/// # Example
///
/// ```
/// use fovea::analyze::statistics::image_moments;
/// use fovea::image::Image;
/// use fovea::pixel::MonoF32;
///
/// // A horizontal bar: wide in x, thin in y, so it is nearly a line and its
/// // major axis is horizontal.
/// let image = Image::generate(32, 32, |x, y| {
///     MonoF32::new(if (4..28).contains(&x) && (15..17).contains(&y) { 1.0 } else { 0.0 })
/// });
///
/// let central = image_moments(&image).central_moments().unwrap();
/// assert!(central.eccentricity() > 0.98, "{}", central.eccentricity());
/// assert!(central.orientation().radians().abs() < 1e-9);
/// ```
#[must_use]
pub fn image_moments<I, P>(image: &I) -> ImageMoments
where
    I: RasterImage<Pixel = P>,
    P: SingleChannel,
    P::Channel: StatisticsChannel,
{
    let mut moments = ImageMoments::default();

    for y in 0..image.height() {
        // Row sums of I, x·I, x²·I and x³·I. Every moment of this row is one
        // of these four times a power of y, which is what keeps the y powers
        // off the per-pixel path.
        let (mut s0, mut s1, mut s2, mut s3) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for (x, pixel) in image.row(y).iter().enumerate() {
            let value = pixel.channel(0).to_f64();
            let x = x as f64;
            let vx = value * x;
            let vx2 = vx * x;
            s0 += value;
            s1 += vx;
            s2 += vx2;
            s3 += vx2 * x;
        }

        let y = y as f64;
        let y2 = y * y;
        moments.m00 += s0;
        moments.m10 += s1;
        moments.m01 += y * s0;
        moments.m20 += s2;
        moments.m11 += y * s1;
        moments.m02 += y2 * s0;
        moments.m30 += s3;
        moments.m21 += y * s2;
        moments.m12 += y2 * s1;
        moments.m03 += y2 * y * s0;
    }

    moments
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{Image, ImageView};
    use crate::pixel::{Mono8, MonoF32, MonoF64};

    /// The moments of `image` computed straight from the definition, with no
    /// row factoring and no identities. The specification the fast path and
    /// the central-moment identities are checked against.
    fn brute_force(image: &Image<MonoF64>) -> (ImageMoments, CentralMoments) {
        let mut raw = ImageMoments::default();
        for y in 0..image.height() {
            for x in 0..image.width() {
                let i = image.row(y)[x].0;
                let (xf, yf) = (x as f64, y as f64);
                raw.m00 += i;
                raw.m10 += xf * i;
                raw.m01 += yf * i;
                raw.m20 += xf * xf * i;
                raw.m11 += xf * yf * i;
                raw.m02 += yf * yf * i;
                raw.m30 += xf * xf * xf * i;
                raw.m21 += xf * xf * yf * i;
                raw.m12 += xf * yf * yf * i;
                raw.m03 += yf * yf * yf * i;
            }
        }

        let (xb, yb) = (raw.m10 / raw.m00, raw.m01 / raw.m00);
        let mut central = CentralMoments {
            mu00: raw.m00,
            ..CentralMoments::default()
        };
        for y in 0..image.height() {
            for x in 0..image.width() {
                let i = image.row(y)[x].0;
                let (dx, dy) = (x as f64 - xb, y as f64 - yb);
                central.mu20 += dx * dx * i;
                central.mu11 += dx * dy * i;
                central.mu02 += dy * dy * i;
                central.mu30 += dx * dx * dx * i;
                central.mu21 += dx * dx * dy * i;
                central.mu12 += dx * dy * dy * i;
                central.mu03 += dy * dy * dy * i;
            }
        }
        (raw, central)
    }

    fn texture(width: usize, height: usize) -> Image<MonoF64> {
        Image::generate(width, height, |x, y| {
            MonoF64::new(((x * 7 + y * 13) % 23) as f64 * 0.5 + (x % 3) as f64)
        })
    }

    #[test]
    fn the_row_factored_pass_matches_the_definition() {
        let image = texture(19, 23);
        let (expected, _) = brute_force(&image);
        let actual = image_moments(&image);
        // Same additions in a different order, so a tolerance rather than
        // exact equality; the scale of m30 is ~10^7 here.
        for (name, a, b) in [
            ("m00", actual.m00, expected.m00),
            ("m10", actual.m10, expected.m10),
            ("m01", actual.m01, expected.m01),
            ("m20", actual.m20, expected.m20),
            ("m11", actual.m11, expected.m11),
            ("m02", actual.m02, expected.m02),
            ("m30", actual.m30, expected.m30),
            ("m21", actual.m21, expected.m21),
            ("m12", actual.m12, expected.m12),
            ("m03", actual.m03, expected.m03),
        ] {
            assert!(
                (a - b).abs() <= 1e-9 * b.abs().max(1.0),
                "{name}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn the_central_identities_match_a_second_pass_about_the_centroid() {
        let image = texture(21, 17);
        let (_, expected) = brute_force(&image);
        let actual = image_moments(&image).central_moments().unwrap();

        for (name, a, b) in [
            ("mu00", actual.mu00, expected.mu00),
            ("mu20", actual.mu20, expected.mu20),
            ("mu11", actual.mu11, expected.mu11),
            ("mu02", actual.mu02, expected.mu02),
            ("mu30", actual.mu30, expected.mu30),
            ("mu21", actual.mu21, expected.mu21),
            ("mu12", actual.mu12, expected.mu12),
            ("mu03", actual.mu03, expected.mu03),
        ] {
            assert!(
                (a - b).abs() <= 1e-6 * b.abs().max(1.0),
                "{name}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn first_order_central_moments_are_zero_by_construction() {
        // Not stored, so this asserts the property the omission relies on:
        // the identity for mu10 / mu01 evaluates to zero.
        let image = texture(13, 11);
        let raw = image_moments(&image);
        let centroid = raw.centroid().unwrap();
        let mu10 = raw.m10 - centroid.x * raw.m00;
        let mu01 = raw.m01 - centroid.y * raw.m00;
        assert!(mu10.abs() <= 1e-9 * raw.m10.abs(), "{mu10}");
        assert!(mu01.abs() <= 1e-9 * raw.m01.abs(), "{mu01}");
    }

    #[test]
    fn a_black_image_has_no_centroid_and_no_central_moments() {
        let image = Image::fill(8, 8, MonoF32::new(0.0));
        let raw = image_moments(&image);
        assert_eq!(raw.m00, 0.0);
        assert_eq!(raw.centroid(), None);
        assert_eq!(raw.central_moments(), None);
    }

    #[test]
    fn an_empty_image_has_no_centroid() {
        let image: Image<MonoF32> = Image::generate(0, 0, |_, _| MonoF32::new(0.0));
        let raw = image_moments(&image);
        assert_eq!(raw.m00, 0.0);
        assert_eq!(raw.centroid(), None);
    }

    #[test]
    fn the_centroid_of_a_symmetric_block_is_its_centre() {
        // A block spanning 4..=11 on both axes: centre at 7.5.
        let image = Image::generate(16, 16, |x, y| {
            Mono8::new(if (4..12).contains(&x) && (4..12).contains(&y) {
                200
            } else {
                0
            })
        });
        let centroid = image_moments(&image).centroid().unwrap();
        assert!((centroid.x - 7.5).abs() < 1e-12, "{}", centroid.x);
        assert!((centroid.y - 7.5).abs() < 1e-12, "{}", centroid.y);
    }

    #[test]
    fn a_nan_sample_poisons_every_moment() {
        let image = Image::generate(4, 4, |x, y| {
            MonoF32::new(if (x, y) == (2, 1) { f32::NAN } else { 1.0 })
        });
        let raw = image_moments(&image);
        assert!(raw.m00.is_nan());
        assert!(raw.m21.is_nan());
        // The centroid exists (m00 is not *equal* to zero) but is NaN, which
        // is the visible signal rather than a silently plausible number.
        let centroid = raw.centroid().unwrap();
        assert!(centroid.x.is_nan() && centroid.y.is_nan());
    }

    #[test]
    fn translating_the_image_leaves_the_central_moments_alone() {
        let shape = |ox: usize, oy: usize| {
            Image::generate(40, 40, |x, y| {
                let inside = (ox..ox + 6).contains(&x) && (oy..oy + 14).contains(&y);
                MonoF32::new(if inside { 1.0 } else { 0.0 })
            })
        };
        let a = image_moments(&shape(3, 3)).central_moments().unwrap();
        let b = image_moments(&shape(20, 17)).central_moments().unwrap();

        for (name, x, y) in [
            ("mu20", a.mu20, b.mu20),
            ("mu11", a.mu11, b.mu11),
            ("mu02", a.mu02, b.mu02),
            ("mu30", a.mu30, b.mu30),
            ("mu03", a.mu03, b.mu03),
        ] {
            assert!((x - y).abs() < 1e-6, "{name}: {x} vs {y}");
        }
    }

    #[test]
    fn normalization_needs_positive_total_intensity() {
        let black = CentralMoments::default();
        assert_eq!(black.normalized(), None);

        let negative = CentralMoments {
            mu00: -4.0,
            ..CentralMoments::default()
        };
        assert_eq!(negative.normalized(), None);

        let nan = CentralMoments {
            mu00: f64::NAN,
            ..CentralMoments::default()
        };
        assert_eq!(nan.normalized(), None, "NaN is not > 0");
    }

    #[test]
    fn normalization_matches_the_closed_form_and_is_scale_invariant_up_to_rasterisation() {
        // For a w×h block of unit intensity, η20 = (w² − 1) / (12·w·h)
        // exactly. The "− 1" is the discretisation term: pixel centres are a
        // sample of the block, not the block, so a *rasterised* shape is only
        // approximately scale-invariant. Pin the closed form at two scales
        // first, then the agreement the term allows.
        let eta = |width: usize, height: usize, side: usize| {
            let image = Image::generate(side, side, |x, y| {
                MonoF32::new(if x < width && y < height { 1.0 } else { 0.0 })
            });
            image_moments(&image)
                .central_moments()
                .and_then(|c| c.normalized())
                .unwrap()
        };
        let closed_form = |w: f64, h: f64| (w * w - 1.0) / (12.0 * w * h);

        let small = eta(20, 40, 64);
        let large = eta(60, 120, 128);
        assert!(
            (small.eta20 - closed_form(20.0, 40.0)).abs() < 1e-12,
            "{}",
            small.eta20
        );
        assert!(
            (large.eta20 - closed_form(60.0, 120.0)).abs() < 1e-12,
            "{}",
            large.eta20
        );
        // A 3x scale-up leaves the discretisation term behind, so the two
        // agree to well under a percent, and the mixed term stays zero.
        let relative = (small.eta20 - large.eta20).abs() / large.eta20;
        assert!(relative < 0.005, "{relative}");
        assert_eq!(small.eta11, 0.0);
        assert_eq!(large.eta11, 0.0);
    }

    #[test]
    fn hu_is_invariant_under_a_quarter_turn() {
        // A quarter turn is exact on a square raster, so the first six
        // invariants must survive it; h7 flips sign only under reflection,
        // not under rotation, so it survives too.
        const N: usize = 33;
        let upright = Image::generate(N, N, |x, y| {
            let inside = (6..12).contains(&x) && (6..24).contains(&y)
                || (6..20).contains(&x) && (18..24).contains(&y);
            MonoF32::new(if inside { 1.0 } else { 0.0 })
        });
        // (x, y) -> (N - 1 - y, x)
        let turned = Image::generate(N, N, |x, y| upright.row(N - 1 - x)[y]);

        let hu = |image: &Image<MonoF32>| {
            image_moments(image)
                .central_moments()
                .and_then(|c| c.normalized())
                .unwrap()
                .hu()
        };
        let (a, b) = (hu(&upright), hu(&turned));
        for i in 0..7 {
            let scale = a[i].abs().max(b[i].abs()).max(1e-12);
            assert!(
                (a[i] - b[i]).abs() <= 1e-9 * scale,
                "h{}: {} vs {}",
                i + 1,
                a[i],
                b[i]
            );
        }
    }

    #[test]
    fn hu_h7_changes_sign_under_a_mirror() {
        const N: usize = 33;
        let upright = Image::generate(N, N, |x, y| {
            let inside = (6..12).contains(&x) && (6..24).contains(&y)
                || (6..20).contains(&x) && (18..24).contains(&y);
            MonoF32::new(if inside { 1.0 } else { 0.0 })
        });
        let mirrored = Image::generate(N, N, |x, y| upright.row(y)[N - 1 - x]);

        let hu = |image: &Image<MonoF32>| {
            image_moments(image)
                .central_moments()
                .and_then(|c| c.normalized())
                .unwrap()
                .hu()
        };
        let (a, b) = (hu(&upright), hu(&mirrored));
        // h1..h6 are reflection-invariant.
        for i in 0..6 {
            let scale = a[i].abs().max(b[i].abs()).max(1e-12);
            assert!((a[i] - b[i]).abs() <= 1e-9 * scale, "h{}", i + 1);
        }
        assert!(a[6].abs() > 1e-12, "the fixture must have a nonzero h7");
        assert!(
            (a[6] + b[6]).abs() <= 1e-9 * a[6].abs(),
            "{} vs {}",
            a[6],
            b[6]
        );
    }

    #[test]
    fn a_horizontal_bar_is_eccentric_and_axis_aligned() {
        let image = Image::generate(32, 32, |x, y| {
            MonoF32::new(if (4..28).contains(&x) && (15..17).contains(&y) {
                1.0
            } else {
                0.0
            })
        });
        let central = image_moments(&image).central_moments().unwrap();
        assert!(central.eccentricity() > 0.98, "{}", central.eccentricity());
        assert!(central.orientation().radians().abs() < 1e-9);
    }

    #[test]
    fn a_symmetric_disc_is_barely_eccentric() {
        let image = Image::generate(41, 41, |x, y| {
            let (dx, dy) = (x as f64 - 20.0, y as f64 - 20.0);
            MonoF32::new(if dx * dx + dy * dy <= 15.0 * 15.0 {
                1.0
            } else {
                0.0
            })
        });
        let central = image_moments(&image).central_moments().unwrap();
        assert!(central.eccentricity() < 0.05, "{}", central.eccentricity());
    }

    #[test]
    fn a_single_bright_pixel_is_degenerate_not_nan() {
        let image = Image::generate(8, 8, |x, y| {
            MonoF32::new(if (x, y) == (3, 4) { 1.0 } else { 0.0 })
        });
        let central = image_moments(&image).central_moments().unwrap();
        assert_eq!(central.mu20, 0.0);
        assert_eq!(central.mu02, 0.0);
        assert_eq!(central.eccentricity(), 0.0);
    }

    #[test]
    fn a_collinear_run_at_a_large_offset_stays_in_the_documented_range() {
        // At x = 65 535 the second-moment cancellation can leave mu20 a
        // hair below zero, which used to push the eccentricity above its
        // documented [0, 1] (1.000_000_19 measured). The clamp is now
        // two-sided; a collinear run stays maximally eccentric and in
        // range.
        let image: Image<MonoF32> = Image::generate(65_536, 5, |x, _| {
            MonoF32::new(if x == 65_535 { 1.0 } else { 0.0 })
        });
        let central = image_moments(&image).central_moments().unwrap();
        let ecc = central.eccentricity();
        assert!(ecc <= 1.0, "eccentricity {ecc} escapes [0, 1]");
        assert!(ecc > 0.999, "a collinear run is maximally eccentric: {ecc}");
    }

    #[test]
    fn integer_and_float_inputs_agree_on_the_same_shape() {
        let eight = Image::generate(24, 24, |x, y| {
            Mono8::new(if (5..15).contains(&x) && (7..19).contains(&y) {
                255
            } else {
                0
            })
        });
        let float = Image::generate(24, 24, |x, y| {
            MonoF32::new(if (5..15).contains(&x) && (7..19).contains(&y) {
                255.0
            } else {
                0.0
            })
        });
        let a = image_moments(&eight);
        let b = image_moments(&float);
        assert_eq!(a.m00, b.m00);
        assert_eq!(a.m21, b.m21);
    }
}

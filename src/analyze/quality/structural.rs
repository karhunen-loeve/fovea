//! Structural similarity (SSIM): the metric that asks whether two images
//! *look* alike rather than how far apart their numbers are.
//!
//! [`ssim`] reports one score, [`ssim_map`] reports where the score came from,
//! and [`SsimParams`] carries the window and the stabilizing constants. See
//! the [module docs](super) for the full-scale value both need.

use crate::image::{
    Image, ImageView, ImageViewMut, MAX_RADIUS, RasterImage, RasterImageMut, gaussian_kernel_1d,
    gaussian_kernel_size,
};
use crate::pixel::{MonoF64, SingleChannel};
use crate::transform::convolve_separable;
use crate::{Error, Sigma, sigma};

use crate::analyze::statistics::StatisticsChannel;
use crate::border::Skip;

use super::PeakValue;

/// Window and stabilizing constants for [`ssim`] and [`ssim_map`].
///
/// A parameter *value*, not a suffix: varying the window is varying this
/// argument, which is the crate's one mechanism for variant selection. [`reference`](Self::reference) builds the published
/// defaults and is what nearly every caller wants.
///
/// # The reference parameters, and why they are the default
///
/// Wang, Bovik, Sheikh and Simoncelli (2004) specify an 11×11
/// circularly-symmetric Gaussian window with σ = 1.5, `K1 = 0.01` and
/// `K2 = 0.03`, and that is what MATLAB's `ssim`, OpenCV's SSIM sample and
/// scikit-image's `gaussian_weights=True` path all compute. An SSIM number is
/// almost always quoted without its parameters, so a default that did not
/// reproduce the published one would silently make every comparison against
/// another implementation wrong.
///
/// Note that scikit-image's *default* is a 7×7 **uniform** window with a
/// sample-variance correction, which is a different number. Nothing here
/// reproduces that; [`try_new`](Self::try_new) varies σ, not the window shape.
///
/// # Why the window is Gaussian and not a box
///
/// A uniform window is what a summed-area table computes in `O(1)`, and it is
/// the cheaper option. It is not the one that ships, for two reasons. The
/// covariance term `σxy` has no summed-area support in this crate at all —
/// there is no product accumulator — so the integral route would only ever
/// cover two of the three moments SSIM needs. And a box window produces the
/// blocking artifacts the 2004 paper introduced the Gaussian to remove, at a
/// score that no published figure matches. The Gaussian route costs five
/// separable convolutions, which the existing engine already performs in
/// `O(taps)` per pixel per axis.
///
/// # Example
///
/// ```
/// use fovea::analyze::quality::{PeakValue, SsimParams};
/// use fovea::pixel::Mono8;
/// use fovea::sigma;
///
/// let params = SsimParams::reference(PeakValue::of_pixel::<Mono8>());
/// assert_eq!(params.sigma(), sigma!(1.5));
/// assert_eq!(params.window_size(), 11); // the published 11×11 window
/// assert_eq!(params.k1(), 0.01);
/// assert_eq!(params.k2(), 0.03);
///
/// // A wider window, for a large frame where 11 pixels is a small detail.
/// let wide = SsimParams::try_new(params.peak(), sigma!(3.0), 0.01, 0.03)?;
/// assert_eq!(wide.window_size(), 19);
/// # Ok::<(), fovea::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SsimParams {
    peak: PeakValue,
    sigma: Sigma,
    k1: f64,
    k2: f64,
}

impl SsimParams {
    /// The window σ of the published parameters: `1.5`.
    pub const REFERENCE_SIGMA: Sigma = sigma!(1.5);

    /// The luminance stabilizing constant of the published parameters:
    /// `0.01`. `C1 = (K1 · peak)²`.
    pub const REFERENCE_K1: f64 = 0.01;

    /// The contrast stabilizing constant of the published parameters:
    /// `0.03`. `C2 = (K2 · peak)²`.
    pub const REFERENCE_K2: f64 = 0.03;

    /// Kernel extent in units of σ, fixed at `3.0`.
    ///
    /// Not a parameter, because its only job is to reproduce the published
    /// window: `radius = round(3.0 · 1.5) = 5`, so
    /// [`REFERENCE_SIGMA`](Self::REFERENCE_SIGMA) yields exactly the 11 taps
    /// the 2004 paper specifies. The crate's blur default of
    /// [`4.0`](crate::transform::DEFAULT_TRUNCATE) would yield 13 and a score
    /// nothing else reports.
    pub const TRUNCATE: f32 = 3.0;

    /// The published parameters of Wang et al. (2004): σ = 1.5 over an 11×11
    /// window, `K1 = 0.01`, `K2 = 0.03`.
    ///
    /// The scores this produces are comparable with other implementations'
    /// defaults; a hand-built [`try_new`](Self::try_new) generally is not.
    #[must_use]
    pub const fn reference(peak: PeakValue) -> Self {
        Self {
            peak,
            sigma: Self::REFERENCE_SIGMA,
            k1: Self::REFERENCE_K1,
            k2: Self::REFERENCE_K2,
        }
    }

    /// Parameters with a chosen window σ and stabilizing constants.
    ///
    /// The σ is validated against the kernel builder's capacity here rather
    /// than at the call to [`ssim`], so a `SsimParams` value carries the
    /// guarantee that its window can actually be built. That is the reason
    /// [`ssim`] has no `MAX_RADIUS` panic where
    /// [`gaussian_blur`](crate::transform::gaussian_blur) does.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidParameter`] if `k1` or `k2` is not finite and
    /// strictly positive (both are denominators of the stabilized ratio, and a
    /// zero constant is what SSIM's instability at flat regions comes from),
    /// if `sigma` needs a kernel radius above [`MAX_RADIUS`], or if `sigma` is
    /// small enough to derive a **one-tap** window.
    ///
    /// The last of those is a provable degeneracy rather than a taste call. A
    /// single-sample window has `E[x²] − E[x]²` identically zero, so every
    /// variance and covariance vanishes, the contrast-and-structure factor is
    /// `C2 / C2 = 1` everywhere, and what comes back is the luminance term
    /// alone — a number that is no longer SSIM but does not look wrong. Since
    /// this `Result` already exists for the two constants, refusing is free;
    /// the same reasoning rejected a zero radius in
    /// [`refine_corners`](crate::features::detect::refine_corners).
    pub fn try_new(peak: PeakValue, sigma: Sigma, k1: f64, k2: f64) -> Result<Self, Error> {
        for (name, value) in [("k1", k1), ("k2", k2)] {
            if !(value.is_finite() && value > 0.0) {
                return Err(Error::InvalidParameter(format!(
                    "ssim {name} must be finite and positive, got {value}"
                )));
            }
        }

        let taps = gaussian_kernel_size(sigma, Self::TRUNCATE);
        if taps < 3 {
            return Err(Error::InvalidParameter(format!(
                "ssim sigma {} derives a one-tap window, which has no variance \
                 and reduces the score to its luminance term",
                sigma.get(),
            )));
        }
        if taps > 2 * MAX_RADIUS + 1 {
            return Err(Error::InvalidParameter(format!(
                "ssim sigma {} needs a {taps}-tap window, above the {}-tap maximum",
                sigma.get(),
                2 * MAX_RADIUS + 1,
            )));
        }

        Ok(Self {
            peak,
            sigma,
            k1,
            k2,
        })
    }

    /// The full-scale value `C1` and `C2` are expressed in.
    #[must_use]
    pub const fn peak(&self) -> PeakValue {
        self.peak
    }

    /// The Gaussian window σ, in pixels.
    #[must_use]
    pub const fn sigma(&self) -> Sigma {
        self.sigma
    }

    /// The luminance stabilizing coefficient.
    #[must_use]
    pub const fn k1(&self) -> f64 {
        self.k1
    }

    /// The contrast stabilizing coefficient.
    #[must_use]
    pub const fn k2(&self) -> f64 {
        self.k2
    }

    /// Side length of the square window σ derives, in pixels: always odd.
    ///
    /// Surfaced rather than hidden inside the metric, because it is also the
    /// amount [`ssim_map`] shrinks by and the smallest image [`ssim`] accepts.
    #[must_use]
    pub fn window_size(&self) -> usize {
        gaussian_kernel_size(self.sigma, Self::TRUNCATE)
    }
}

/// Structural similarity of two images, as one score.
///
/// The mean of [`ssim_map`] over every position where the window fits: `1.0` for
/// identical images, falling toward `0.0` as structure diverges and able to go
/// negative where local contrast inverts. Unlike
/// [`peak_signal_to_noise_ratio`](super::ChannelSquaredError::peak_signal_to_noise_ratio),
/// it is bounded, so a threshold on it means something across images.
///
/// # Single channel by a compile-time bound
///
/// SSIM is defined on one intensity signal: `C1` and `C2` are fractions of one
/// dynamic range, and the paper computes on luminance. Colour SSIM has no
/// settled definition — per channel then averaged, on Y'CbCr, on luma alone —
/// so this crate declines to pick one silently, and a multi-channel pixel type
/// simply does not compile. Convert first and name the choice:
/// `convert_image(&rgb, Luminance)` for the luma reading, or compare planes for
/// the per-channel one. This is the same bound, for the same reason, as
/// [`image_moments`](crate::analyze::statistics::image_moments).
///
/// # Errors — Tier 2
///
/// - [`Error::SizeMismatch`] if the two images have different dimensions.
/// - [`Error::InvalidParameter`] if the window does not fit: an image narrower
///   or shorter than [`SsimParams::window_size`] has no position where the
///   full window lies inside the frame, and there is nothing to average.
///
/// # Example
///
/// ```
/// use fovea::analyze::quality::{PeakValue, SsimParams, ssim};
/// use fovea::image::Image;
/// use fovea::pixel::Mono8;
///
/// let params = SsimParams::reference(PeakValue::of_pixel::<Mono8>());
/// let checker = Image::generate(32, 32, |x, y| {
///     Mono8::new(if (x / 4 + y / 4) % 2 == 0 { 220 } else { 30 })
/// });
///
/// // Identical inputs score exactly 1.0.
/// assert_eq!(ssim(&checker, &checker, params)?, 1.0);
///
/// // Inverting every pixel keeps the structure and destroys the similarity.
/// let inverted = Image::generate(32, 32, |x, y| {
///     Mono8::new(255 - checker.pixel_at(x, y).value())
/// });
/// # use fovea::image::ImageView;
/// assert!(ssim(&checker, &inverted, params)? < 0.0);
/// # Ok::<(), fovea::Error>(())
/// ```
pub fn ssim<A, B, P>(a: &A, b: &B, params: SsimParams) -> Result<f64, Error>
where
    A: RasterImage<Pixel = P>,
    B: RasterImage<Pixel = P>,
    P: SingleChannel,
    P::Channel: StatisticsChannel,
{
    let map = ssim_map(a, b, params)?;
    let count = map.width() * map.height();
    // `ssim_map` refuses an image the window does not fit, so the map is never
    // empty here and the division is total.
    debug_assert!(count > 0, "ssim_map returned an empty map");

    let mut total = 0.0;
    for y in 0..map.height() {
        for pixel in map.row(y) {
            total += pixel.0;
        }
    }
    Ok(total / count as f64)
}

/// Per-position structural similarity of two images.
///
/// The map [`ssim`] averages. Reach for it when *where* the two images differ
/// matters — the map is the thing you look at to see that a demosaic lost
/// detail along edges rather than uniformly, which no single score can say.
///
/// # The map is smaller than the input, and by how much
///
/// A window that leaves the frame is not evaluated, exactly as
/// [`Skip`] means for a convolution and for the same
/// reason: a value extrapolated past the edge is not a measurement. So the map
/// is `(width − 2r) × (height − 2r)` for `r = window_size / 2`, and map
/// position `(x, y)` is the similarity around **input** position
/// `(x + r, y + r)`. This is the convention
/// [`match_template`](crate::transform::match_template) uses for its score
/// map.
///
/// # `NaN` propagates
///
/// A single `NaN` sample spreads across its whole window and, through the
/// global centring below, across the frame. That is the
/// [`image_moments`](crate::analyze::statistics::image_moments) policy rather
/// than the [`squared_error`](super::squared_error) one: a window statistic has
/// no per-position count to record an exclusion in, so silently dropping a
/// sample would report a similarity for pixels that were not compared. To find
/// out whether an image is clean first, read
/// [`ChannelStatistics::nan_count`](crate::analyze::statistics::ChannelStatistics::nan_count).
///
/// # How the moments are computed, and the one deviation that matters
///
/// The five window moments are five separable Gaussian convolutions, in `f64`,
/// of the two images and of their squares and product. Before any of that,
/// **each image has its own global mean subtracted**. That shift is exactly
/// value-preserving — `E[x²] − E[x]²` does not depend on it, and it is added
/// back before the luminance term uses the means — and it is load-bearing, not
/// tidiness.
///
/// The reason is the weights. A separable kernel in this crate carries `f32`
/// taps, so each moment inherits about `2 · μ² · 2⁻²⁴` counts of absolute
/// error. Centred, `μ` is the local deviation and that error is nothing.
/// Uncentred, `μ` is the absolute brightness, and on the case this crate
/// actually meets — a narrow signal riding a large dark offset in a wide
/// container, say a 10-bit swing on a pedestal of 40 000 counts in `Mono16` —
/// the error reaches about 190 counts against a true window variance near 14.
/// The uncentred variance then comes out **negative** (measured: −124 on the
/// regression fixture), the contrast term degenerates, and the score moves from
/// 0.967 to 0.703. With the peak named for the signal rather than the
/// container it gets worse than wrong: uncentred scores of −30 and below, well
/// outside SSIM's range. `a_pedestal_does_not_collapse_the_variance` pins it.
///
/// This is the same class of failure
/// [`ChannelStatistics`](crate::analyze::statistics::ChannelStatistics) uses
/// Welford's recurrence against. Welford is not available to a sliding window,
/// which has to form `E[x²] − E[x]²`; centring is what is available, and on
/// second-order moments it is exact.
///
/// # Cost
///
/// Five separable convolutions plus three full-frame `f64` planes:
/// `O(width · height · window_size)` time, and about six `f64` images of peak
/// live allocation. That is the same working set as the reference
/// implementations, and it is the reason this is a measurement rather than a
/// per-frame pipeline step.
///
/// # Errors — Tier 2
///
/// As [`ssim`].
///
/// # Example
///
/// ```
/// use fovea::analyze::quality::{PeakValue, SsimParams, ssim_map};
/// use fovea::image::{Image, ImageView, ImageViewMut};
/// use fovea::pixel::Mono8;
///
/// let params = SsimParams::reference(PeakValue::of_pixel::<Mono8>());
/// let radius = params.window_size() / 2;
///
/// // A smooth ramp, with one block of it blanked out.
/// let reference = Image::generate(48, 48, |x, _| Mono8::new((x * 5) as u8));
/// let mut damaged = reference.clone();
/// for y in 20..28 {
///     for x in 20..28 {
///         *damaged.pixel_at_mut(x, y) = Mono8::new(0);
///     }
/// }
///
/// let map = ssim_map(&reference, &damaged, params)?;
/// assert_eq!(map.width(), 48 - 2 * radius);
/// assert_eq!(map.height(), 48 - 2 * radius);
///
/// // The map names the damage: input (24, 24) is map (24 − r, 24 − r).
/// let damaged_score = map.pixel_at(24 - radius, 24 - radius).value();
/// let clean_score = map.pixel_at(5, 5).value();
/// assert!(damaged_score < 0.5, "{damaged_score}");
/// assert!(clean_score > 0.99, "{clean_score}");
/// # Ok::<(), fovea::Error>(())
/// ```
pub fn ssim_map<A, B, P>(a: &A, b: &B, params: SsimParams) -> Result<Image<MonoF64>, Error>
where
    A: RasterImage<Pixel = P>,
    B: RasterImage<Pixel = P>,
    P: SingleChannel,
    P::Channel: StatisticsChannel,
{
    if a.size() != b.size() {
        return Err(Error::SizeMismatch {
            expected: a.size(),
            actual: b.size(),
        });
    }

    let window = params.window_size();
    if a.width() < window || a.height() < window {
        return Err(Error::InvalidParameter(format!(
            "ssim: a {}x{} window does not fit a {}x{} image; no position has the \
             full window inside the frame",
            window,
            window,
            a.width(),
            a.height(),
        )));
    }

    // ── Centre each image on its own mean ────────────────────────────────
    //
    // Exactly value-preserving for the second-order moments and restored
    // before the luminance term uses them. See the rustdoc above for the
    // pedestal case this exists for, and why `f32` kernel taps make it
    // load-bearing rather than cosmetic. A frame with no finite mean (all
    // `NaN`, or an infinity) centres at zero instead: the constant is a
    // numerical convenience, so any finite choice is correct, and subtracting
    // an infinity would destroy samples that are fine.
    let offset_a = finite_mean(a);
    let offset_b = finite_mean(b);

    let mut plane_a = centered_plane(a, offset_a);
    let mut plane_b = centered_plane(b, offset_b);
    let mut plane_ab = Image::<MonoF64>::zero(a.width(), a.height());
    for y in 0..plane_ab.height() {
        for x in 0..plane_ab.width() {
            let product = plane_a.pixel_at(x, y).0 * plane_b.pixel_at(x, y).0;
            *plane_ab.pixel_at_mut(x, y) = MonoF64(product);
        }
    }

    // ── Five window moments, valid positions only ────────────────────────
    //
    // `Skip` shrinks the output by the kernel extent on each axis, so each of
    // these is already the `(w − 2r) × (h − 2r)` valid block with no border to
    // discard. Each plane is dropped as soon as its moment exists, which caps
    // the live set below the naive eight.
    let kernel = gaussian_kernel_1d(params.sigma, SsimParams::TRUNCATE);
    let mean_a: Image<MonoF64> = convolve_separable(&plane_a, &kernel, &Skip);
    let mean_b: Image<MonoF64> = convolve_separable(&plane_b, &kernel, &Skip);
    let moment_ab: Image<MonoF64> = convolve_separable(&plane_ab, &kernel, &Skip);
    drop(plane_ab);

    square_in_place(&mut plane_a);
    let moment_aa: Image<MonoF64> = convolve_separable(&plane_a, &kernel, &Skip);
    drop(plane_a);

    square_in_place(&mut plane_b);
    let moment_bb: Image<MonoF64> = convolve_separable(&plane_b, &kernel, &Skip);
    drop(plane_b);

    // ── Combine ──────────────────────────────────────────────────────────
    let peak = params.peak.get();
    let c1 = (params.k1 * peak) * (params.k1 * peak);
    let c2 = (params.k2 * peak) * (params.k2 * peak);

    let mut map = Image::<MonoF64>::zero(mean_a.width(), mean_a.height());
    for y in 0..map.height() {
        for x in 0..map.width() {
            // Window means of the centred planes; the true luminances add the
            // offsets back.
            let centered_mean_a = mean_a.pixel_at(x, y).0;
            let centered_mean_b = mean_b.pixel_at(x, y).0;
            let luminance_a = centered_mean_a + offset_a;
            let luminance_b = centered_mean_b + offset_b;

            // A variance cannot be negative; `E[x²] − E[x]²` in floating point
            // can land a rounding step below zero on a flat window. Written as
            // a comparison rather than `f64::max`, which would turn a `NaN`
            // into 0.0 and hide it.
            let variance_a =
                non_negative(moment_aa.pixel_at(x, y).0 - centered_mean_a * centered_mean_a);
            let variance_b =
                non_negative(moment_bb.pixel_at(x, y).0 - centered_mean_b * centered_mean_b);
            // Cauchy-Schwarz bounds the covariance by √(σ²a·σ²b) — over the
            // *clamped* variances, so that an image against itself stays
            // exactly 1.0 even where its raw variance rounded a step below
            // zero, and a residual covariance over zeroed variances cannot
            // push a score past 1. Written as comparisons so a NaN sample
            // propagates instead of being clamped or panicking.
            let raw = moment_ab.pixel_at(x, y).0 - centered_mean_a * centered_mean_b;
            let bound = (variance_a * variance_b).sqrt();
            let covariance = if raw > bound {
                bound
            } else if raw < -bound {
                -bound
            } else {
                raw
            };

            let numerator = (2.0 * luminance_a * luminance_b + c1) * (2.0 * covariance + c2);
            let denominator = (luminance_a * luminance_a + luminance_b * luminance_b + c1)
                * (variance_a + variance_b + c2);
            *map.pixel_at_mut(x, y) = MonoF64(numerator / denominator);
        }
    }

    Ok(map)
}

/// The image's mean channel value if it is usable as a centring constant, and
/// `0.0` otherwise.
fn finite_mean<I, P>(image: &I) -> f64
where
    I: RasterImage<Pixel = P>,
    P: SingleChannel,
    P::Channel: StatisticsChannel,
{
    use crate::analyze::statistics::{ChannelStatistics, image_statistics};

    let stats: ChannelStatistics<P::Channel> = image_statistics(image);
    match stats.mean() {
        Some(mean) if mean.is_finite() => mean,
        _ => 0.0,
    }
}

/// The image's single channel widened to `f64`, with `offset` subtracted.
fn centered_plane<I, P>(image: &I, offset: f64) -> Image<MonoF64>
where
    I: RasterImage<Pixel = P>,
    P: SingleChannel,
    P::Channel: StatisticsChannel,
{
    let mut plane = Image::<MonoF64>::zero(image.width(), image.height());
    for y in 0..image.height() {
        let source = image.row(y);
        let destination = plane.row_mut(y);
        for (pixel, cell) in source.iter().zip(destination.iter_mut()) {
            *cell = MonoF64(pixel.channel(0).to_f64() - offset);
        }
    }
    plane
}

/// Squares every value of `plane` in place.
fn square_in_place(plane: &mut Image<MonoF64>) {
    for y in 0..plane.height() {
        for cell in plane.row_mut(y) {
            cell.0 *= cell.0;
        }
    }
}

/// `value` clamped up to zero, leaving `NaN` alone.
#[inline]
fn non_negative(value: f64) -> f64 {
    if value < 0.0 { 0.0 } else { value }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Size;
    use crate::analyze::statistics::{ChannelStatistics, image_statistics};
    use crate::image::ContiguousImage;
    use crate::peak;
    use crate::pixel::{Mono8, Mono16, MonoF32};

    fn reference_params() -> SsimParams {
        SsimParams::reference(PeakValue::of_pixel::<Mono8>())
    }

    fn checkerboard(size: usize, block: usize) -> Image<Mono8> {
        Image::generate(size, size, |x, y| {
            Mono8::new(if (x / block + y / block) % 2 == 0 {
                220
            } else {
                30
            })
        })
    }

    // ── Parameters ───────────────────────────────────────────────────────

    #[test]
    fn the_reference_parameters_are_the_published_eleven_tap_window() {
        let params = reference_params();
        assert_eq!(params.sigma(), sigma!(1.5));
        assert_eq!(params.k1(), 0.01);
        assert_eq!(params.k2(), 0.03);
        // The whole reason `TRUNCATE` is 3.0 and not the crate's blur default
        // of 4.0: 4.0 would give 13 taps and a score no publication reports.
        assert_eq!(params.window_size(), 11);
        assert_eq!(gaussian_kernel_size(sigma!(1.5), 4.0), 13);
    }

    #[test]
    fn a_non_positive_stabilizing_constant_is_refused() {
        let peak = PeakValue::of_pixel::<Mono8>();
        assert!(SsimParams::try_new(peak, sigma!(1.5), 0.0, 0.03).is_err());
        assert!(SsimParams::try_new(peak, sigma!(1.5), 0.01, -0.03).is_err());
        assert!(SsimParams::try_new(peak, sigma!(1.5), f64::NAN, 0.03).is_err());
        assert!(SsimParams::try_new(peak, sigma!(1.5), 0.01, 0.03).is_ok());
    }

    #[test]
    fn a_sigma_that_derives_a_one_tap_window_is_refused() {
        // radius = floor(3σ + 0.5), so σ < 1/6 gives a single tap: no window
        // variance at all, and a score that is only the luminance term.
        let peak = PeakValue::of_pixel::<Mono8>();
        assert_eq!(gaussian_kernel_size(sigma!(0.1), SsimParams::TRUNCATE), 1);
        let refused = SsimParams::try_new(peak, sigma!(0.1), 0.01, 0.03);
        assert!(
            matches!(refused, Err(Error::InvalidParameter(_))),
            "{refused:?}"
        );

        // The smallest accepted window is three taps.
        let accepted = SsimParams::try_new(peak, sigma!(0.2), 0.01, 0.03).unwrap();
        assert_eq!(accepted.window_size(), 3);
    }

    #[test]
    fn a_sigma_past_the_kernel_capacity_is_refused_at_construction() {
        // The check that lets `ssim` be panic-free where `gaussian_blur` is
        // not: MAX_RADIUS is 64, so radius = round(3σ) > 64 means σ > 21.5.
        let peak = PeakValue::of_pixel::<Mono8>();
        assert!(SsimParams::try_new(peak, sigma!(21.0), 0.01, 0.03).is_ok());
        let refused = SsimParams::try_new(peak, sigma!(40.0), 0.01, 0.03);
        assert!(
            matches!(refused, Err(Error::InvalidParameter(_))),
            "{refused:?}"
        );
    }

    // ── The identities SSIM must satisfy ─────────────────────────────────

    #[test]
    fn an_image_against_itself_is_exactly_one() {
        // Not "close to one": with identical inputs every factor of the
        // numerator equals its partner in the denominator, so the ratio is
        // exactly 1.0 in floating point too.
        let image = checkerboard(32, 4);
        let map = ssim_map(&image, &image, reference_params()).unwrap();
        for y in 0..map.height() {
            for (x, pixel) in map.row(y).iter().enumerate() {
                assert_eq!(pixel.0, 1.0, "at ({x}, {y})");
            }
        }
        assert_eq!(ssim(&image, &image, reference_params()).unwrap(), 1.0);
    }

    #[test]
    fn the_score_is_symmetric_in_its_arguments() {
        let a = checkerboard(40, 5);
        let b = Image::generate(40, 40, |x, y| Mono8::new(((x * 7 + y * 3) % 256) as u8));
        let forward = ssim(&a, &b, reference_params()).unwrap();
        let backward = ssim(&b, &a, reference_params()).unwrap();
        assert!(
            (forward - backward).abs() < 1e-12,
            "{forward} vs {backward}"
        );
    }

    #[test]
    fn the_score_stays_within_minus_one_and_one() {
        let a = checkerboard(48, 6);
        let inverted = Image::generate(48, 48, |x, y| Mono8::new(255 - a.pixel_at(x, y).value()));
        let noise = Image::generate(48, 48, |x, y| Mono8::new(((x * 37 + y * 101) % 256) as u8));

        for other in [&inverted, &noise] {
            let map = ssim_map(&a, other, reference_params()).unwrap();
            for pixel in map.as_slice() {
                assert!(pixel.0 >= -1.0 && pixel.0 <= 1.0, "{}", pixel.0);
            }
        }
    }

    #[test]
    fn inverting_contrast_drives_the_score_negative() {
        // Structure preserved, correlation reversed: the covariance term goes
        // negative and takes the score with it.
        let a = checkerboard(48, 6);
        let inverted = Image::generate(48, 48, |x, y| Mono8::new(255 - a.pixel_at(x, y).value()));
        let score = ssim(&a, &inverted, reference_params()).unwrap();
        assert!(score < 0.0, "{score}");
    }

    #[test]
    fn a_flat_pair_of_equal_brightness_is_similar_and_a_different_one_is_not() {
        // The case C1 and C2 exist for: zero variance everywhere, so the
        // contrast and structure terms are 0/0 without them.
        let params = reference_params();
        let grey = Image::fill(32, 32, Mono8::new(128));
        let same = Image::fill(32, 32, Mono8::new(128));
        assert_eq!(ssim(&grey, &same, params).unwrap(), 1.0);

        let brighter = Image::fill(32, 32, Mono8::new(200));
        let score = ssim(&grey, &brighter, params).unwrap();
        assert!(score < 1.0 && score > 0.0, "{score}");
        // Only the luminance term can move: 2·128·200 + C1 over 128² + 200² + C1.
        let c1 = (0.01 * 255.0f64).powi(2);
        let expected = (2.0 * 128.0 * 200.0 + c1) / (128.0 * 128.0 + 200.0 * 200.0 + c1);
        assert!((score - expected).abs() < 1e-9, "{score} vs {expected}");
    }

    #[test]
    fn more_degradation_scores_lower() {
        // Monotonicity is the property that makes a threshold meaningful.
        let params = reference_params();
        let reference =
            Image::generate(64, 64, |x, y| Mono8::new((((x * 5) ^ (y * 3)) % 256) as u8));

        let mut previous = 1.000_001;
        for amplitude in [0u8, 4, 16, 48] {
            let degraded = Image::generate(64, 64, |x, y| {
                let base = reference.pixel_at(x, y).value();
                let wobble = if (x + y) % 2 == 0 { amplitude } else { 0 };
                Mono8::new(base.saturating_add(wobble))
            });
            let score = ssim(&reference, &degraded, params).unwrap();
            assert!(
                score < previous,
                "amplitude {amplitude}: {score} !< {previous}"
            );
            previous = score;
        }
    }

    // ── The map's geometry ───────────────────────────────────────────────

    #[test]
    fn the_map_is_the_valid_window_block_and_is_offset_by_the_radius() {
        let params = reference_params();
        let radius = params.window_size() / 2;
        let reference = Image::generate(48, 40, |x, _| Mono8::new((x * 5) as u8));
        let mut damaged = reference.clone();
        for y in 18..26 {
            for x in 18..26 {
                *damaged.pixel_at_mut(x, y) = Mono8::new(0);
            }
        }

        let map = ssim_map(&reference, &damaged, params).unwrap();
        assert_eq!(map.size(), Size::new(48 - 2 * radius, 40 - 2 * radius));

        // Input (22, 22) sits at the centre of the damage; map (22 − r, 22 − r)
        // is the position that reports it.
        assert!(map.pixel_at(22 - radius, 22 - radius).0 < 0.5);
        // Far from the damage the ramp is intact.
        assert!(map.pixel_at(2, 2).0 > 0.99);
    }

    #[test]
    fn an_image_the_window_does_not_fit_is_refused_rather_than_clipped() {
        let params = reference_params();
        let small: Image<Mono8> = Image::fill(10, 32, Mono8::new(0));
        let other: Image<Mono8> = Image::fill(10, 32, Mono8::new(0));
        let refused = ssim_map(&small, &other, params);
        assert!(
            matches!(refused, Err(Error::InvalidParameter(_))),
            "{refused:?}"
        );

        // Exactly the window size is the smallest accepted image, and it has
        // one valid position.
        let exact: Image<Mono8> = Image::fill(11, 11, Mono8::new(50));
        let map = ssim_map(&exact, &exact, params).unwrap();
        assert_eq!(map.size(), Size::new(1, 1));
        assert_eq!(ssim(&exact, &exact, params).unwrap(), 1.0);
    }

    #[test]
    fn a_size_mismatch_is_a_tier_two_error() {
        let params = reference_params();
        let a: Image<Mono8> = Image::fill(32, 32, Mono8::new(0));
        let b: Image<Mono8> = Image::fill(32, 33, Mono8::new(0));
        assert_eq!(
            ssim(&a, &b, params),
            Err(Error::SizeMismatch {
                expected: Size::new(32, 32),
                actual: Size::new(32, 33),
            })
        );
    }

    #[test]
    fn the_score_is_the_mean_of_the_map() {
        let params = reference_params();
        let a = checkerboard(40, 5);
        let b = Image::generate(40, 40, |x, y| {
            Mono8::new(a.pixel_at(x, y).value().wrapping_add(((x * y) % 20) as u8))
        });

        let map = ssim_map(&a, &b, params).unwrap();
        let expected =
            map.as_slice().iter().map(|p| p.0).sum::<f64>() / (map.width() * map.height()) as f64;
        let score = ssim(&a, &b, params).unwrap();
        assert!((score - expected).abs() < 1e-12, "{score} vs {expected}");
    }

    // ── The numerical care, pinned ───────────────────────────────────────

    #[test]
    fn a_pedestal_does_not_collapse_the_variance() {
        // The regression the global centring exists for, and the industrial
        // case that produces it: a narrow signal riding a large dark offset in
        // a wide container — here a ~10-bit swing on a pedestal of 40 000
        // counts in `Mono16`, with the peak named for the signal rather than
        // the container.
        //
        // The uncentred `E[x²] − E[x]²` forms 1.6e9 − 1.6e9. The separable
        // engine's kernel weights are `f32`, so each moment carries about
        // `2·μ²·2⁻²⁴ ≈ 190` counts of absolute error, which swamps a true
        // window variance of ~14 and comes out **negative**: measured at
        // −124 on this fixture. Centred, the same moments are formed from
        // deviations of order 10 and the variance is exact to `f64`.
        //
        // What that costs, measured on this fixture: the uncentred score is
        // 0.703 where the centred one is 0.967, and at a named peak of 100 the
        // uncentred score reaches −30, outside SSIM's range entirely.
        let peak = peak!(1023.0);
        let params = SsimParams::reference(peak);

        // Two images with identical value distributions and transposed
        // structure: every global statistic matches, the local correlation
        // does not.
        let structure = |pedestal: u16, transpose: bool| {
            Image::generate(64, 64, |x, y| {
                let (u, v) = if transpose { (y, x) } else { (x, y) };
                Mono16::new(pedestal + (((u * 3 + v) % 7) * 2) as u16)
            })
        };

        let on_pedestal =
            ssim(&structure(40_000, false), &structure(40_000, true), params).unwrap();
        let at_zero = ssim(&structure(0, false), &structure(0, true), params).unwrap();

        // The structural half of SSIM is shift-invariant by construction, and
        // the luminance term is ~1 either way because both images share their
        // mean. So the pedestal must not move the score. This is the assertion
        // the uncentred form fails, by 0.26.
        assert!(
            (on_pedestal - at_zero).abs() < 1e-4,
            "the pedestal moved the score: {on_pedestal} vs {at_zero}",
        );
        assert!(on_pedestal > 0.9 && on_pedestal < 1.0, "{on_pedestal}");
    }

    #[test]
    fn a_pedestal_cannot_push_the_score_out_of_range() {
        // The same mechanism at its most visible: with the peak named for
        // a narrow signal, an uncentred variance error of ~190 counts
        // against a `C2` of `(0.03·20)² = 0.36` produces scores in the
        // thousands (−1749.9 measured for this pedestal/peak pair). The
        // bound is the invariant that catches it.
        let params = SsimParams::reference(peak!(20.0));
        let a = Image::generate(48, 48, |x, y| {
            Mono16::new(60_000 + ((x * 3 + y) % 7) as u16)
        });
        let b = Image::generate(48, 48, |x, y| {
            Mono16::new(60_000 + ((y * 3 + x) % 7) as u16)
        });

        let map = ssim_map(&a, &b, params).unwrap();
        for pixel in map.as_slice() {
            assert!(pixel.0 >= -1.0 && pixel.0 <= 1.0, "{}", pixel.0);
        }
    }

    #[test]
    fn the_variance_never_lands_negative_in_the_map() {
        // A flat window's `E[x²] − E[x]²` is a rounding step either side of
        // zero; the clamp keeps the denominator honest. If it were skipped the
        // scores would drift slightly above 1.0 on the flat regions.
        let params = reference_params();
        let half_flat = Image::generate(48, 48, |x, _| {
            Mono8::new(if x < 24 { 200 } else { (x * 3) as u8 })
        });
        let map = ssim_map(&half_flat, &half_flat, params).unwrap();
        for pixel in map.as_slice() {
            assert!(pixel.0 <= 1.0, "{}", pixel.0);
        }
    }

    #[test]
    fn nan_propagates_rather_than_being_silently_dropped() {
        // The stated policy: a window statistic has no count to record an
        // exclusion in, so it must not pretend the sample was not there.
        let params = SsimParams::reference(peak!(1.0));
        let mut a = Image::generate(32, 32, |x, _| MonoF32::new(x as f32 / 31.0));
        let b = a.clone();
        *a.pixel_at_mut(16, 16) = MonoF32::new(f32::NAN);

        let score = ssim(&a, &b, params).unwrap();
        assert!(score.is_nan(), "{score}");

        // And the caller's documented way to find out first.
        let stats: ChannelStatistics<_> = image_statistics(&a);
        assert_eq!(stats.nan_count, 1);
    }

    #[test]
    fn a_wider_window_smooths_the_map_and_shrinks_it_further() {
        let peak = PeakValue::of_pixel::<Mono8>();
        let reference =
            Image::generate(64, 64, |x, y| Mono8::new((((x * 5) ^ (y * 3)) % 256) as u8));
        let degraded = Image::generate(64, 64, |x, y| {
            Mono8::new(
                reference
                    .pixel_at(x, y)
                    .value()
                    .saturating_add(if (x + y) % 3 == 0 { 20 } else { 0 }),
            )
        });

        let narrow = SsimParams::try_new(peak, sigma!(1.5), 0.01, 0.03).unwrap();
        let wide = SsimParams::try_new(peak, sigma!(4.0), 0.01, 0.03).unwrap();
        assert_eq!(narrow.window_size(), 11);
        assert_eq!(wide.window_size(), 25);

        let narrow_map = ssim_map(&reference, &degraded, narrow).unwrap();
        let wide_map = ssim_map(&reference, &degraded, wide).unwrap();
        assert_eq!(narrow_map.size(), Size::new(54, 54));
        assert_eq!(wide_map.size(), Size::new(40, 40));

        // A wider window averages over more of the degradation, so the map's
        // own spread falls.
        let spread = |map: &Image<MonoF64>| {
            let stats: ChannelStatistics<_> = image_statistics(map);
            stats.std_dev().unwrap()
        };
        assert!(spread(&wide_map) < spread(&narrow_map));
    }

    #[test]
    fn the_reference_window_matches_a_hand_computed_gaussian_window() {
        // An independent check of the whole moment path on one position: build
        // the 11-tap kernel, compute the five moments by hand over the window
        // centred on the map's only position, and compare.
        let params = reference_params();
        let a = Image::generate(11, 11, |x, y| Mono8::new(((x * 17 + y * 5) % 200) as u8));
        let b = Image::generate(11, 11, |x, y| Mono8::new(((y * 11 + x * 3) % 200) as u8));

        let taps: Vec<f64> = {
            let sigma = 1.5f64;
            let raw: Vec<f64> = (0..11)
                .map(|i| {
                    let d = i as f64 - 5.0;
                    (-d * d / (2.0 * sigma * sigma)).exp()
                })
                .collect();
            let sum: f64 = raw.iter().sum();
            raw.iter().map(|w| w / sum).collect()
        };

        let mut mean_a = 0.0;
        let mut mean_b = 0.0;
        let mut moment_aa = 0.0;
        let mut moment_bb = 0.0;
        let mut moment_ab = 0.0;
        for y in 0..11 {
            for x in 0..11 {
                let weight = taps[x] * taps[y];
                let va = f64::from(a.pixel_at(x, y).value());
                let vb = f64::from(b.pixel_at(x, y).value());
                mean_a += weight * va;
                mean_b += weight * vb;
                moment_aa += weight * va * va;
                moment_bb += weight * vb * vb;
                moment_ab += weight * va * vb;
            }
        }
        let variance_a = moment_aa - mean_a * mean_a;
        let variance_b = moment_bb - mean_b * mean_b;
        let covariance = moment_ab - mean_a * mean_b;
        let c1 = (0.01 * 255.0f64).powi(2);
        let c2 = (0.03 * 255.0f64).powi(2);
        let expected = ((2.0 * mean_a * mean_b + c1) * (2.0 * covariance + c2))
            / ((mean_a * mean_a + mean_b * mean_b + c1) * (variance_a + variance_b + c2));

        let map = ssim_map(&a, &b, params).unwrap();
        assert_eq!(map.size(), Size::new(1, 1));
        let actual = map.pixel_at(0, 0).0;
        // The kernel weights the engine carries are `f32`, so parity with an
        // `f64` hand computation is bounded by that, not by `f64` epsilon.
        assert!((actual - expected).abs() < 1e-6, "{actual} vs {expected}");
    }
}

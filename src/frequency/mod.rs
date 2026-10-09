//! The frequency domain: the discrete Fourier transform of an image.
//!
//! [`dft`](crate::frequency::dft) computes the
//! [`Spectrum`](crate::frequency::Spectrum) of an image by a method written
//! at the call site, and
//! [`Spectrum::inverse`](crate::frequency::Spectrum::inverse) returns to
//! the image. The sources are [`MonoF32`](crate::pixel::MonoF32) and
//! [`MonoF64`](crate::pixel::MonoF64) images, and the transform computes in
//! their precision.
//!
//! ## What the transform assumes
//!
//! The image is transformed at its own size, as one period of a periodic
//! signal: its right edge meets its left and its bottom its top. Nothing is
//! padded behind the caller's back. To transform at another size, or to
//! choose what lies beyond the edges, pad first with
//! [`pad`](crate::transform::pad), which takes a border policy.
//!
//! The spectrum holds the transform as the textbook defines it,
//! `X[k] = Σ x[n]·exp(−2πi·kn/N)` along each axis: unscaled, with the
//! negative exponent, as numpy, OpenCV and FFTW compute it by default. The
//! inverse is the exact inverse, so the round trip gives back the image
//! without a factor.
//!
//! ## Phase correlation
//!
//! [`phase_correlate`](crate::frequency::phase_correlate) measures where
//! the content of one image lies relative to another, to a fraction of a
//! pixel, from the phase of their cross-power spectrum, so that brightness
//! and contrast barely move the result. Its documentation describes how to
//! choose the one parameter it needs, the radius of the weight on the
//! frequencies.
//!
//! ## Choosing a method
//!
//! | Method | Sizes | Returns |
//! |---|---|---|
//! | [`Radix2`](crate::frequency::Radix2) | every side a power of two | `Result`, an error for any other size |
//! | [`Bluestein`](crate::frequency::Bluestein) | any | the spectrum |
//! | [`Auto`](crate::frequency::Auto) | any: `Radix2` along a side that is a power of two, `Bluestein` along any other | the spectrum |
//!
//! Every method computes the same transform; they differ in speed and in
//! rounding. Which sizes a method accepts is a question to the method:
//! `Radix2.next_size(size)` is the size to pad to for `Radix2`, and it keeps
//! that meaning when faster methods for other sizes arrive.
//!
//! ## Accuracy
//!
//! No bound on the error is promised. What it depends on: the twiddle
//! factors are computed from their exact integer indices, never by a
//! recurrence; the error relative to the whole result grows slowly with the
//! size; and [`Bluestein`](crate::frequency::Bluestein) rounds more than
//! [`Radix2`](crate::frequency::Radix2), since it runs three transforms
//! where `Radix2` runs one. A single bin is less accurate than the whole: a
//! component that is zero in exact arithmetic comes out as rounding noise,
//! and so does its phase. `MonoF64` images transform in double precision.
//!
//! # Example
//!
//! ```
//! use fovea::frequency::{Auto, dft};
//! use fovea::image::{Image, ImageView};
//! use fovea::pixel::MonoF32;
//!
//! // Vertical stripes, a period of 8 px across a 640 px wide image.
//! let img = Image::generate(640, 480, |x, _| {
//!     MonoF32::new((2.0 * std::f32::consts::PI * x as f32 / 8.0).cos())
//! });
//! let spectrum = dft(&img, Auto);
//!
//! // Back to the image, unchanged up to rounding.
//! let back = spectrum.inverse(Auto);
//! assert!((back.pixel_at(16, 7).0 - 1.0).abs() < 1e-4);
//! ```

mod convolve;
mod correlate;
mod engine;
mod filters;
mod method;
mod spectrum;
#[cfg(test)]
mod testing;
mod units;

pub use convolve::convolve;
pub use correlate::{CorrelationRadius, PhaseCorrelation, PhaseReference, phase_correlate};
pub use filters::{
    Band, BandPass, BandStop, Butterworth, Gaussian, HighPass, Ideal, LowPass, Notch, Profile,
    Radius, TransferFunction,
};
pub use method::{Auto, Bluestein, DftMethod, Radix2, dft};
pub use spectrum::{Spectrum, SpectrumSource};
pub use units::{Bins, CyclesPerPixel, Frequency, FrequencyIndex, FrequencyUnit};

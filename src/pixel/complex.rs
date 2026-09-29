//! Complex pixel types.
//!
//! - [`ComplexF32`], [`ComplexF64`]: a complex value `re + im·i` per pixel
//!
//! A complex pixel is one value with two coordinates. Its layout is two
//! floats, `re` before `im`, so it is [`HomogeneousPixel`] and an image of
//! it splits into planes. Its parts are not two quantities of their own,
//! so it is not [`ChannelwiseMath`](crate::pixel::ChannelwiseMath): which
//! direction counts as "real" is a convention, and a per-part maximum,
//! difference or threshold changes when the whole image is rotated by a
//! common phase. Linear operations do not, which is why blurring, resizing
//! and blending a complex image are correct and compile.

use fovea_derive::{HomogeneousPixel, PlainPixel, ZeroablePixel};

use crate::pixel::{LinearPixel, LinearSpace, impl_origin_invariant_pixel};
use std::hash::{Hash, Hasher};
use std::ops::{Add, Mul, Neg, Sub};

use super::{canonicalize_f32, canonicalize_f64};

/// A complex pixel `re + im·i` with `f32` parts.
///
/// The pixel type of complex images: the response of a quadrature filter,
/// an analytic signal, or a complex field sampled on the pixel grid.
///
/// `*` between two complex pixels is the **complex product**,
/// `(a + bi)(c + di) = (ac − bd) + (ad + bc)i`, so
/// [`PixelMultiply`](crate::transform::PixelMultiply) multiplies two
/// complex images as complex numbers. `+`, `-` and multiplication by a real
/// scalar work on both parts.
///
/// # Example
///
/// ```
/// use fovea::pixel::ComplexF32;
///
/// let z = ComplexF32::new(3.0, 4.0);
/// assert_eq!(z.magnitude(), 5.0);
/// assert_eq!(z * z.conjugate(), ComplexF32::new(25.0, 0.0));
/// assert_eq!(ComplexF32::new(0.0, 1.0) * ComplexF32::new(0.0, 1.0), ComplexF32::new(-1.0, 0.0));
/// ```
///
/// # What does not compile
///
/// There is no order on complex numbers, so there is no `PartialOrd`, and
/// the channel-wise combiners reject complex images. The `Magnitude`
/// combiner would compute `hypot(a.re, b.re)` and `hypot(a.im, b.im)`,
/// which is the magnitude of nothing:
///
/// ```compile_fail
/// use fovea::image::Image;
/// use fovea::pixel::ComplexF32;
/// use fovea::transform::{Magnitude, combine_images};
///
/// let z = Image::fill(4, 4, ComplexF32::new(1.0, 2.0));
/// // ERROR: `ComplexF32: ChannelwiseMath` is not satisfied.
/// let _ = combine_images(&z, &z, Magnitude);
/// ```
///
/// A per-part absolute difference depends on where the real axis lies:
///
/// ```compile_fail
/// use fovea::image::Image;
/// use fovea::pixel::ComplexF32;
/// use fovea::transform::{AbsDiff, combine_images};
///
/// let z = Image::fill(4, 4, ComplexF32::new(1.0, 2.0));
/// // ERROR: `ComplexF32: ChannelwiseMath` is not satisfied.
/// let _ = combine_images(&z, &z, AbsDiff);
/// ```
///
/// And a maximum has nothing to compare by (rejected twice over, since
/// `f32` is not `Ord` either):
///
/// ```compile_fail
/// use fovea::image::Image;
/// use fovea::pixel::ComplexF32;
/// use fovea::transform::{Max, combine_images};
///
/// let z = Image::fill(4, 4, ComplexF32::new(1.0, 2.0));
/// // ERROR: `ComplexF32: ChannelwiseMath` is not satisfied.
/// let _ = combine_images(&z, &z, Max);
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, PlainPixel, HomogeneousPixel, ZeroablePixel)]
pub struct ComplexF32 {
    /// Real part.
    #[zero(default)]
    pub re: f32,
    /// Imaginary part.
    #[zero(default)]
    pub im: f32,
}

/// A complex pixel `re + im·i` with `f64` parts.
///
/// The double-precision sibling of [`ComplexF32`], with the same traits and
/// the same operations.
///
/// # Example
///
/// ```
/// use fovea::pixel::ComplexF64;
///
/// let z = ComplexF64::new(0.0, 2.0);
/// assert_eq!(z.magnitude(), 2.0);
/// assert_eq!(z.phase(), core::f64::consts::FRAC_PI_2);
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, PlainPixel, HomogeneousPixel, ZeroablePixel)]
pub struct ComplexF64 {
    /// Real part.
    #[zero(default)]
    pub re: f64,
    /// Imaginary part.
    #[zero(default)]
    pub im: f64,
}

/// The methods, operators and `LinearPixel<f32>` impl shared by both
/// complex pixels. The `LinearPixel` derive is not used: it would generate
/// a part-wise `Mul<Self>`, and `*` on a complex pixel is the complex
/// product.
macro_rules! impl_complex {
    ($T:ident, $F:ty, $canonicalize:ident) => {
        impl $T {
            /// Creates a complex pixel from its real and imaginary parts.
            #[must_use]
            #[inline]
            pub const fn new(re: $F, im: $F) -> Self {
                Self { re, im }
            }

            /// The absolute value `|z|`, computed with `hypot` so that the
            /// intermediate square cannot overflow.
            #[must_use]
            #[inline]
            pub fn magnitude(self) -> $F {
                self.re.hypot(self.im)
            }

            /// The argument of `z` in radians, in `(−π, π]`, as `atan2(im, re)`.
            #[must_use]
            #[inline]
            pub fn phase(self) -> $F {
                self.im.atan2(self.re)
            }

            /// The complex conjugate `re − im·i`.
            #[must_use]
            #[inline]
            pub fn conjugate(self) -> Self {
                Self {
                    re: self.re,
                    im: -self.im,
                }
            }

            /// The squared magnitude `re² + im²`, without the square root.
            #[must_use]
            #[inline]
            pub fn norm_sqr(self) -> $F {
                self.re * self.re + self.im * self.im
            }
        }

        impl Add for $T {
            type Output = Self;
            #[inline(always)]
            fn add(self, other: Self) -> Self {
                Self {
                    re: self.re + other.re,
                    im: self.im + other.im,
                }
            }
        }

        impl Sub for $T {
            type Output = Self;
            #[inline(always)]
            fn sub(self, other: Self) -> Self {
                Self {
                    re: self.re - other.re,
                    im: self.im - other.im,
                }
            }
        }

        impl Neg for $T {
            type Output = Self;
            #[inline(always)]
            fn neg(self) -> Self {
                Self {
                    re: -self.re,
                    im: -self.im,
                }
            }
        }

        /// The complex product `(ac − bd) + (ad + bc)i`.
        impl Mul for $T {
            type Output = Self;
            #[inline(always)]
            fn mul(self, other: Self) -> Self {
                Self {
                    re: self.re * other.re - self.im * other.im,
                    im: self.re * other.im + self.im * other.re,
                }
            }
        }

        /// Multiplication by a real scalar, which scales both parts.
        impl Mul<$F> for $T {
            type Output = Self;
            #[inline(always)]
            fn mul(self, scalar: $F) -> Self {
                Self {
                    re: self.re * scalar,
                    im: self.im * scalar,
                }
            }
        }

        impl Hash for $T {
            fn hash<H: Hasher>(&self, state: &mut H) {
                $canonicalize(self.re).hash(state);
                $canonicalize(self.im).hash(state);
            }
        }

        impl LinearPixel for $T {
            type Accumulator = Self;
            #[inline(always)]
            fn to_accumulator(&self) -> Self {
                *self
            }
            #[inline(always)]
            fn scale(&self, scalar: f32) -> Self {
                let s = scalar as $F;
                Self {
                    re: self.re * s,
                    im: self.im * s,
                }
            }
            #[inline(always)]
            fn scale_add(&self, scalar: f32, addend: Self) -> Self {
                let s = scalar as $F;
                #[cfg(target_feature = "fma")]
                {
                    Self {
                        re: self.re.mul_add(s, addend.re),
                        im: self.im.mul_add(s, addend.im),
                    }
                }
                #[cfg(not(target_feature = "fma"))]
                {
                    Self {
                        re: self.re * s + addend.re,
                        im: self.im * s + addend.im,
                    }
                }
            }
            /// Every channel equals `scalar`, as the trait defines it: the
            /// result is `scalar + scalar·i`.
            #[inline(always)]
            fn uniform(scalar: f32) -> Self {
                let s = scalar as $F;
                Self { re: s, im: s }
            }
        }

        // ℂ is a real vector space, so interpolating or blending complex
        // pixels is correct.
        impl LinearSpace for $T {}
    };
}

impl_complex!(ComplexF32, f32, canonicalize_f32);
impl_complex!(ComplexF64, f64, canonicalize_f64);

// Double-precision scalars for double-precision pixels, as `MonoF64` has.
impl LinearPixel<f64> for ComplexF64 {
    type Accumulator = Self;
    #[inline(always)]
    fn to_accumulator(&self) -> Self {
        *self
    }
    #[inline(always)]
    fn scale(&self, scalar: f64) -> Self {
        *self * scalar
    }
    #[inline(always)]
    fn scale_add(&self, scalar: f64, addend: Self) -> Self {
        #[cfg(target_feature = "fma")]
        {
            Self {
                re: self.re.mul_add(scalar, addend.re),
                im: self.im.mul_add(scalar, addend.im),
            }
        }
        #[cfg(not(target_feature = "fma"))]
        {
            *self * scalar + addend
        }
    }
    #[inline(always)]
    fn uniform(scalar: f64) -> Self {
        Self {
            re: scalar,
            im: scalar,
        }
    }
}

// A complex value means the same wherever the crop puts the origin.
impl_origin_invariant_pixel!(ComplexF32, ComplexF64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pixel::{HomogeneousPixel, ZeroablePixel, blend};
    use std::collections::hash_map::DefaultHasher;

    fn hash_of<T: Hash>(v: &T) -> u64 {
        let mut h = DefaultHasher::new();
        v.hash(&mut h);
        h.finish()
    }

    #[test]
    fn the_product_is_complex_multiplication() {
        let a = ComplexF32::new(1.0, 2.0);
        let b = ComplexF32::new(3.0, -1.0);
        // (1 + 2i)(3 − i) = 3 − i + 6i − 2i² = 5 + 5i
        assert_eq!(a * b, ComplexF32::new(5.0, 5.0));
        assert_eq!(b * a, a * b);
        let i = ComplexF64::new(0.0, 1.0);
        assert_eq!(i * i, ComplexF64::new(-1.0, 0.0));
    }

    #[test]
    fn scalar_and_additive_operators_work_on_both_parts() {
        let a = ComplexF32::new(1.0, -2.0);
        let b = ComplexF32::new(0.5, 4.0);
        assert_eq!(a + b, ComplexF32::new(1.5, 2.0));
        assert_eq!(a - b, ComplexF32::new(0.5, -6.0));
        assert_eq!(-a, ComplexF32::new(-1.0, 2.0));
        assert_eq!(a * 3.0, ComplexF32::new(3.0, -6.0));
        assert_eq!(ComplexF64::new(1.0, 1.0) * 0.5, ComplexF64::new(0.5, 0.5));
    }

    #[test]
    fn magnitude_phase_conjugate_and_norm() {
        let z = ComplexF32::new(3.0, -4.0);
        assert_eq!(z.magnitude(), 5.0);
        assert_eq!(z.norm_sqr(), 25.0);
        assert_eq!(z.conjugate(), ComplexF32::new(3.0, 4.0));
        assert_eq!(ComplexF32::new(-1.0, 0.0).phase(), core::f32::consts::PI);
        assert_eq!(
            ComplexF64::new(0.0, -1.0).phase(),
            -core::f64::consts::FRAC_PI_2
        );
        // `hypot` does not overflow where the squares would.
        assert_eq!(ComplexF32::new(3.0e30, 4.0e30).magnitude(), 5.0e30);
        assert!(ComplexF32::new(3.0e30, 4.0e30).norm_sqr().is_infinite());
    }

    #[test]
    fn zero_is_zero_plus_zero_i() {
        assert_eq!(ComplexF32::zero(), ComplexF32::new(0.0, 0.0));
        assert_eq!(ComplexF64::zero(), ComplexF64::new(0.0, 0.0));
    }

    #[test]
    fn layout_is_re_then_im() {
        assert_eq!(<ComplexF32 as HomogeneousPixel>::CHANNEL_COUNT, 2);
        assert_eq!(core::mem::size_of::<ComplexF32>(), 8);
        assert_eq!(core::mem::size_of::<ComplexF64>(), 16);
        let z = ComplexF32::new(1.5, -2.5);
        assert_eq!(z.channel(0), 1.5);
        assert_eq!(z.channel(1), -2.5);
        assert_eq!(
            ComplexF64::from_channels(&[7.0, 8.0]),
            ComplexF64::new(7.0, 8.0)
        );
    }

    #[test]
    fn linear_pixel_scales_and_blends_in_the_plane() {
        let a = ComplexF32::new(0.0, 0.0);
        let b = ComplexF32::new(2.0, -4.0);
        assert_eq!(b.scale(0.5), ComplexF32::new(1.0, -2.0));
        assert_eq!(b.scale_add(0.5, a), ComplexF32::new(1.0, -2.0));
        assert_eq!(blend(&a, &b, 0.5), ComplexF32::new(1.0, -2.0));
        assert_eq!(
            <ComplexF32 as LinearPixel>::uniform(0.25),
            ComplexF32::new(0.25, 0.25)
        );
        let c = ComplexF64::new(1.0, 3.0);
        assert_eq!(
            <ComplexF64 as LinearPixel<f64>>::scale(&c, 2.0),
            ComplexF64::new(2.0, 6.0)
        );
        assert_eq!(
            <ComplexF64 as LinearPixel<f64>>::scale_add(&c, 2.0, c),
            ComplexF64::new(3.0, 9.0)
        );
        assert_eq!(
            <ComplexF64 as LinearPixel<f64>>::uniform(-1.0),
            ComplexF64::new(-1.0, -1.0)
        );
        assert_eq!(
            <ComplexF64 as LinearPixel>::scale(&c, 0.5),
            ComplexF64::new(0.5, 1.5)
        );
        assert_eq!(
            <ComplexF64 as LinearPixel>::scale_add(&c, 1.0, c),
            ComplexF64::new(2.0, 6.0)
        );
        assert_eq!(
            <ComplexF64 as LinearPixel>::uniform(2.0),
            ComplexF64::new(2.0, 2.0)
        );
        assert_eq!(<ComplexF64 as LinearPixel>::to_accumulator(&c), c);
        assert_eq!(<ComplexF64 as LinearPixel<f64>>::to_accumulator(&c), c);
    }

    #[test]
    fn an_image_splits_into_planes_and_back() {
        use crate::image::{Image, ImagePlanes, ImageView};

        let z = Image::generate(3, 2, |x, y| ComplexF32::new(x as f32, -(y as f32)));
        let planes = ImagePlanes::from_interleaved(&z);
        assert_eq!(planes.channel_count(), 2);
        assert_eq!(planes.plane(0).unwrap().pixel_at(2, 1), 2.0);
        assert_eq!(planes.plane(1).unwrap().pixel_at(2, 1), -1.0);
        assert_eq!(planes.to_interleaved(), z);

        let w = Image::fill(2, 2, ComplexF64::new(0.5, 1.5));
        assert_eq!(ImagePlanes::from_interleaved(&w).to_interleaved(), w);
    }

    #[test]
    fn the_named_parts_round_trip_through_from_parts() {
        use crate::image::{ContiguousImage, Image};
        use crate::pixel::{MonoF32, MonoF64};
        use crate::transform::{
            ComplexMagnitude, ComplexPhase, FromParts, ImaginaryPart, RealPart, combine_images,
            convert_image,
        };

        let z = Image::generate(4, 3, |x, y| ComplexF32::new(x as f32 - 1.5, y as f32 + 0.5));
        let re: Image<MonoF32> = convert_image(&z, RealPart);
        let im: Image<MonoF32> = convert_image(&z, ImaginaryPart);
        assert_eq!(combine_images(&re, &im, FromParts).unwrap(), z);

        let mag: Image<MonoF32> = convert_image(&z, ComplexMagnitude);
        let phase: Image<MonoF32> = convert_image(&z, ComplexPhase);
        for (i, p) in z.as_slice().iter().enumerate() {
            assert_eq!(mag.as_slice()[i].0, p.magnitude());
            assert_eq!(phase.as_slice()[i].0, p.phase());
        }

        let w = Image::fill(2, 2, ComplexF64::new(-2.0, 0.25));
        let re: Image<MonoF64> = convert_image(&w, RealPart);
        let im: Image<MonoF64> = convert_image(&w, ImaginaryPart);
        assert_eq!(combine_images(&re, &im, FromParts).unwrap(), w);
    }

    #[test]
    fn complex_multiply_is_the_pixel_product() {
        use crate::image::{ContiguousImage, Image};
        use crate::transform::{ComplexMultiply, PixelMultiply, combine_images};

        let a = Image::generate(3, 3, |x, y| ComplexF32::new(x as f32, y as f32));
        let b = Image::fill(3, 3, ComplexF32::new(0.0, 1.0));
        let named = combine_images(&a, &b, ComplexMultiply).unwrap();
        assert_eq!(named, combine_images(&a, &b, PixelMultiply).unwrap());
        // Multiplying by i turns (x, y) into (−y, x).
        assert_eq!(named.as_slice()[5], ComplexF32::new(-1.0, 2.0));

        let c = Image::fill(1, 1, ComplexF64::new(2.0, 0.0));
        let d = Image::fill(1, 1, ComplexF64::new(0.0, 3.0));
        assert_eq!(
            combine_images(&c, &d, ComplexMultiply).unwrap().as_slice()[0],
            ComplexF64::new(0.0, 6.0)
        );
    }

    #[test]
    fn equal_values_hash_equally() {
        assert_eq!(
            hash_of(&ComplexF32::new(0.0, 1.0)),
            hash_of(&ComplexF32::new(-0.0, 1.0))
        );
        assert_eq!(
            hash_of(&ComplexF64::new(f64::NAN, 0.0)),
            hash_of(&ComplexF64::new(-f64::NAN, 0.0))
        );
        assert_ne!(
            hash_of(&ComplexF32::new(1.0, 2.0)),
            hash_of(&ComplexF32::new(2.0, 1.0))
        );
    }
}

//! The transform engine: the tables of one length, and the one-dimensional
//! transforms that use them.
//!
//! Setup and execution are kept apart. The tables of a length (the twiddle
//! factors and the bit-reversal order of radix-2, the chirp and its
//! transform for Bluestein) are values the transforms take as an argument,
//! so a plan that keeps them across calls is a wrapper around them and
//! leaves this code unchanged.
//!
//! Every root of unity is computed from its exact integer index and
//! rounded once; none comes from a recurrence. The index is reduced to the
//! first octant in integer arithmetic, so the argument of `sin` and `cos` is
//! at most π/4, and multiples of a quarter turn come out exact.

use crate::pixel::{ComplexF32, ComplexF64};
use std::fmt::Debug;
use std::ops::{Add, Mul, Sub};

/// A complex number the engine computes with: [`ComplexF32`] or
/// [`ComplexF64`]. The precision of the bins is the precision of the
/// arithmetic.
///
/// Public in a private module, as a sealed trait is: it bounds the public
/// `SpectrumSource::Bin`, and no caller can name it.
#[allow(
    unreachable_pub,
    reason = "reachable as the bound of `SpectrumSource::Bin`, which rustc 1.85 does not see"
)]
pub trait Cplx:
    Copy + Debug + PartialEq + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self>
{
    /// `0 + 0i`.
    const ZERO: Self;
    /// The complex number with these parts, each rounded once.
    fn from_f64(re: f64, im: f64) -> Self;
    /// The parts, exactly.
    fn to_f64(self) -> (f64, f64);
    /// The complex conjugate.
    fn conj(self) -> Self;
    /// Both parts divided by `n`, each rounded once.
    fn div_count(self, n: usize) -> Self;

    /// The real part alone, `re + 0i`.
    fn real_part(self) -> Self {
        Self::from_f64(self.to_f64().0, 0.0)
    }
}

macro_rules! impl_cplx {
    ($T:ident, $F:ty) => {
        impl Cplx for $T {
            const ZERO: Self = $T { re: 0.0, im: 0.0 };

            #[inline(always)]
            fn from_f64(re: f64, im: f64) -> Self {
                $T {
                    re: re as $F,
                    im: im as $F,
                }
            }

            #[inline(always)]
            fn to_f64(self) -> (f64, f64) {
                (f64::from(self.re), f64::from(self.im))
            }

            #[inline(always)]
            fn conj(self) -> Self {
                self.conjugate()
            }

            #[inline(always)]
            fn div_count(self, n: usize) -> Self {
                let d = n as $F;
                $T {
                    re: self.re / d,
                    im: self.im / d,
                }
            }
        }
    };
}

impl_cplx!(ComplexF32, f32);
impl_cplx!(ComplexF64, f64);

/// `cos(2π·k/n)` and `sin(2π·k/n)` in `f64`, for `n > 0`.
///
/// `k` is reduced modulo `n`, and the angle to the first octant, in integer
/// arithmetic: 8k = q·n + r, so the angle is π/4 · (q + r/n). Within an
/// octant the angle measured from its nearer multiple of π/2 is at most
/// π/4, and the eight symmetries of the circle map its sine and cosine
/// back without rounding.
pub(crate) fn unit_root(k: usize, n: usize) -> (f64, f64) {
    debug_assert!(n > 0, "a root of unity of order zero");
    let n_wide = n as u128;
    let eighths = 8 * (k as u128 % n_wide);
    let q = eighths / n_wide;
    let r = eighths % n_wide;
    // Even octants count from their start, odd ones back from their end.
    let from_axis = if q % 2 == 0 { r } else { n_wide - r };
    let phi = std::f64::consts::FRAC_PI_4 * (from_axis as f64 / n as f64);
    let (s, c) = phi.sin_cos();
    match q {
        0 => (c, s),
        1 => (s, c),
        2 => (-s, c),
        3 => (-c, s),
        4 => (-c, -s),
        5 => (-s, -c),
        6 => (s, -c),
        _ => (c, -s),
    }
}

/// `exp(−2πi·k/n)`, the root of unity of the forward transform, rounded
/// once to `C`.
pub(crate) fn forward_root<C: Cplx>(k: usize, n: usize) -> C {
    let (c, s) = unit_root(k, n);
    C::from_f64(c, -s)
}

/// The tables of a radix-2 transform of one length: zero or a power of two.
#[derive(Clone, Debug)]
pub(crate) struct Radix2<C> {
    len: usize,
    /// `exp(−2πi·j/len)` for `j` in `0..len/2`.
    twiddles: Vec<C>,
    /// The bit-reversed position of each index.
    bitrev: Vec<usize>,
}

impl<C: Cplx> Radix2<C> {
    /// The tables for `len`, which is zero or a power of two.
    pub(crate) fn new(len: usize) -> Self {
        debug_assert!(len == 0 || len.is_power_of_two(), "radix-2 of length {len}");
        let twiddles = (0..len / 2).map(|j| forward_root(j, len)).collect();
        let bits = if len == 0 { 0 } else { len.trailing_zeros() };
        let bitrev = (0..len)
            .map(|i| {
                if bits == 0 {
                    0
                } else {
                    i.reverse_bits() >> (usize::BITS - bits)
                }
            })
            .collect();
        Self {
            len,
            twiddles,
            bitrev,
        }
    }

    /// The forward transform of `buf` in place, decimation in time.
    pub(crate) fn forward(&self, buf: &mut [C]) {
        debug_assert_eq!(buf.len(), self.len);
        for (i, &j) in self.bitrev.iter().enumerate() {
            if i < j {
                buf.swap(i, j);
            }
        }
        let mut half = 1;
        while half < self.len {
            let size = 2 * half;
            let stride = self.len / size;
            for block in buf.chunks_exact_mut(size) {
                let (lo, hi) = block.split_at_mut(half);
                for (j, (a, b)) in lo.iter_mut().zip(hi.iter_mut()).enumerate() {
                    let t = *b * self.twiddles[j * stride];
                    let u = *a;
                    *a = u + t;
                    *b = u - t;
                }
            }
            half = size;
        }
    }

    #[cfg(test)]
    pub(crate) fn twiddles(&self) -> &[C] {
        &self.twiddles
    }
}

/// The tables of Bluestein's transform of one length, which may be any.
///
/// With jk = (j² + k² − (k − j)²) / 2, a DFT of length n becomes a
/// circular convolution with the chirp `exp(−iπ·j²/n)`, evaluated by
/// radix-2 transforms of the length m, the power of two at least 2n − 1.
#[derive(Clone, Debug)]
pub(crate) struct Bluestein<C> {
    len: usize,
    /// `exp(−iπ·j²/len)` for `j` in `0..len`, from the index j² mod 2·len.
    chirp: Vec<C>,
    /// The radix-2 transform of the conjugate chirp, wrapped to length m,
    /// divided by m, which is exact for a power of two.
    kernel: Vec<C>,
    inner: Radix2<C>,
}

impl<C: Cplx> Bluestein<C> {
    /// The tables for `len`.
    pub(crate) fn new(len: usize) -> Self {
        let m = if len == 0 {
            0
        } else {
            (2 * len - 1).next_power_of_two()
        };
        let inner = Radix2::new(m);
        let chirp: Vec<C> = (0..len)
            .map(|j| forward_root(chirp_index(j, len), 2 * len))
            .collect();
        let mut kernel = vec![C::ZERO; m];
        for (j, &c) in chirp.iter().enumerate() {
            kernel[j] = c.conj();
            if j > 0 {
                kernel[m - j] = c.conj();
            }
        }
        inner.forward(&mut kernel);
        for z in &mut kernel {
            *z = z.div_count(m);
        }
        Self {
            len,
            chirp,
            kernel,
            inner,
        }
    }

    /// The forward transform of `buf` in place, with `work` as the buffer
    /// of the convolution.
    pub(crate) fn forward(&self, buf: &mut [C], work: &mut Vec<C>) {
        debug_assert_eq!(buf.len(), self.len);
        let m = self.kernel.len();
        work.clear();
        work.extend(buf.iter().zip(&self.chirp).map(|(&x, &c)| x * c));
        work.resize(m, C::ZERO);
        self.inner.forward(work);
        for (z, &k) in work.iter_mut().zip(&self.kernel) {
            // The conjugate, so that the forward transform below inverts.
            *z = (*z * k).conj();
        }
        self.inner.forward(work);
        for ((out, &z), &c) in buf.iter_mut().zip(work.iter()).zip(&self.chirp) {
            *out = c * z.conj();
        }
    }

    #[cfg(test)]
    pub(crate) fn chirp(&self) -> &[C] {
        &self.chirp
    }
}

/// j² mod 2·len, in integers: the index of `exp(−iπ·j²/len)` as a root of
/// unity of order 2·len. Formed in floating point, j² loses its last digits
/// once it exceeds the mantissa, and the angle with them.
fn chirp_index(j: usize, len: usize) -> usize {
    (j as u128 * j as u128 % (2 * len as u128)) as usize
}

/// How one axis is transformed.
#[derive(Clone, Debug)]
pub(crate) enum Axis<C> {
    Radix2(Radix2<C>),
    Bluestein(Bluestein<C>),
}

impl<C: Cplx> Axis<C> {
    /// The forward transform of `buf` in place.
    pub(crate) fn forward(&self, buf: &mut [C], work: &mut Vec<C>) {
        match self {
            Axis::Radix2(t) => t.forward(buf),
            Axis::Bluestein(t) => t.forward(buf, work),
        }
    }

    /// The inverse transform of `buf` in place, without the division by
    /// the length: the conjugate of the forward transform of the
    /// conjugate, which is exact apart from the forward transform itself.
    pub(crate) fn inverse_unscaled(&self, buf: &mut [C], work: &mut Vec<C>) {
        for z in buf.iter_mut() {
            *z = z.conj();
        }
        self.forward(buf, work);
        for z in buf.iter_mut() {
            *z = z.conj();
        }
    }
}

/// Side of the square blocks the transpose moves at a time.
const BLOCK: usize = 32;

/// Writes the transpose of `src`, `height` rows of `width` values, into
/// `dst`, `width` rows of `height` values, in square blocks so that both
/// sides stay in cache.
pub(crate) fn transpose<T: Copy>(src: &[T], width: usize, height: usize, dst: &mut [T]) {
    debug_assert_eq!(src.len(), width * height);
    debug_assert_eq!(dst.len(), width * height);
    for by in (0..height).step_by(BLOCK) {
        let y_end = (by + BLOCK).min(height);
        for bx in (0..width).step_by(BLOCK) {
            let x_end = (bx + BLOCK).min(width);
            for y in by..y_end {
                let row = &src[y * width..(y + 1) * width];
                for (x, &v) in row.iter().enumerate().take(x_end).skip(bx) {
                    dst[x * height + y] = v;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frequency::testing::{
        BLUESTEIN_F32, BLUESTEIN_F64, MU32, MU64, Rng, U32, U64, higham, naive, oracle_error,
        reference_root, rel_error,
    };

    /// cos(2πk/n) and sin(2πk/n) for (k, n), correctly rounded to `f64`,
    /// computed with 70 decimal digits on 2026-10-05.
    const REFERENCE_ROOTS: [(usize, usize, u64, u64); 75] = [
        (1, 8, 0x3fe6a09e667f3bcd, 0x3fe6a09e667f3bcd),
        (2, 8, 0x0000000000000000, 0x3ff0000000000000),
        (3, 8, 0xbfe6a09e667f3bcd, 0x3fe6a09e667f3bcd),
        (5, 8, 0xbfe6a09e667f3bcd, 0xbfe6a09e667f3bcd),
        (7, 8, 0x3fe6a09e667f3bcd, 0xbfe6a09e667f3bcd),
        (1, 12, 0x3febb67ae8584caa, 0x3fe0000000000000),
        (2, 12, 0x3fe0000000000000, 0x3febb67ae8584caa),
        (4, 12, 0xbfe0000000000000, 0x3febb67ae8584caa),
        (5, 12, 0xbfebb67ae8584caa, 0x3fe0000000000000),
        (8, 12, 0xbfe0000000000000, 0xbfebb67ae8584caa),
        (10, 12, 0x3fe0000000000000, 0xbfebb67ae8584caa),
        (11, 12, 0x3febb67ae8584caa, 0xbfe0000000000000),
        (1, 1000, 0x3fefffd69aa0b99d, 0x3f79bc5a9d91f679),
        (124, 1000, 0x3fe6c4e677f1a28e, 0x3fe67c1bca1a4947),
        (125, 1000, 0x3fe6a09e667f3bcd, 0x3fe6a09e667f3bcd),
        (126, 1000, 0x3fe67c1bca1a4947, 0x3fe6c4e677f1a28e),
        (333, 1000, 0xbfdfe243c7b26539, 0x3febbf0b0e1a47e0),
        (499, 1000, 0xbfefffd69aa0b99d, 0x3f79bc5a9d91f679),
        (714, 1000, 0xbfccb4e77fdd0216, 0xbfef2f52faf620c7),
        (751, 1000, 0x3f79bc5a9d91f679, 0xbfefffd69aa0b99d),
        (999, 1000, 0x3fefffd69aa0b99d, 0xbf79bc5a9d91f679),
        (1, 1024, 0x3fefffd8858e8a92, 0x3f7921f0fe670071),
        (127, 1024, 0x3fe6c40d73c18275, 0x3fe67cf78491af10),
        (128, 1024, 0x3fe6a09e667f3bcd, 0x3fe6a09e667f3bcd),
        (129, 1024, 0x3fe67cf78491af10, 0x3fe6c40d73c18275),
        (341, 1024, 0xbfdfe2f64be71210, 0x3febbed7c49380ea),
        (511, 1024, 0xbfefffd8858e8a92, 0x3f7921f0fe670071),
        (731, 1024, 0xbfcccf8cb312b286, 0xbfef2dc9c9089a9d),
        (769, 1024, 0x3f7921f0fe670071, 0xbfefffd8858e8a92),
        (1023, 1024, 0x3fefffd8858e8a92, 0xbf7921f0fe670071),
        (1, 1080, 0x3fefffdc827509f6, 0x3f77d4555e817be2),
        (134, 1080, 0x3fe6c2387aad6389, 0x3fe67ed2216b390f),
        (135, 1080, 0x3fe6a09e667f3bcd, 0x3fe6a09e667f3bcd),
        (136, 1080, 0x3fe67ed2216b390f, 0x3fe6c2387aad6389),
        (360, 1080, 0xbfe0000000000000, 0x3febb67ae8584caa),
        (539, 1080, 0xbfefffdc827509f6, 0x3f77d4555e817be2),
        (771, 1080, 0xbfcccb3236cdc675, 0xbfef2e0a214e870f),
        (811, 1080, 0x3f77d4555e817be2, 0xbfefffdc827509f6),
        (1079, 1080, 0x3fefffdc827509f6, 0xbf77d4555e817be2),
        (1, 4096, 0x3feffffd88586ee6, 0x3f5921faaee6472e),
        (511, 4096, 0x3fe6a97f692c82e9, 0x3fe697b9e686941c),
        (512, 4096, 0x3fe6a09e667f3bcd, 0x3fe6a09e667f3bcd),
        (513, 4096, 0x3fe697b9e686941c, 0x3fe6a97f692c82e9),
        (1365, 4096, 0xbfdff8be6537615f, 0x3febb892d5d5dad5),
        (2047, 4096, 0xbfeffffd88586ee6, 0x3f5921faaee6472e),
        (2925, 4096, 0xbfcc9e90b824a6a9, 0xbfef309b794b719f),
        (3073, 4096, 0x3f5921faaee6472e, 0xbfeffffd88586ee6),
        (4095, 4096, 0x3feffffd88586ee6, 0xbf5921faaee6472e),
        (1, 15015, 0x3fefffffd0fe95b7, 0x3f3b6c9a5df7e012),
        (1875, 15015, 0x3fe6a52974950ec8, 0x3fe69c126eb52452),
        (1876, 15015, 0x3fe6a2bd45e5d607, 0x3fe69e7f54335d0b),
        (1877, 15015, 0x3fe6a050d4b690c0, 0x3fe6a0ebf73dff93),
        (5005, 15015, 0xbfe0000000000000, 0x3febb67ae8584caa),
        (7506, 15015, 0xbfefffff963cd0fb, 0x3f449173b9e32e2a),
        (10725, 15015, 0xbfcc7b90e3024582, 0xbfef329c0558e969),
        (11262, 15015, 0x3f349173cae1dc52, 0xbfefffffe58f3434),
        (15014, 15015, 0x3fefffffd0fe95b7, 0xbf3b6c9a5df7e012),
        (1, 1048576, 0x3feffffffffd8858, 0x3ed921fb544387ba),
        (131071, 1048576, 0x3fe6a0a7493f0a95, 0x3fe6a09583bbefb8),
        (131072, 1048576, 0x3fe6a09e667f3bcd, 0x3fe6a09e667f3bcd),
        (131073, 1048576, 0x3fe6a09583bbefb8, 0x3fe6a0a7493f0a95),
        (349525, 1048576, 0xbfdffff8beab1af5, 0x3febb67d008256e8),
        (524287, 1048576, 0xbfeffffffffd8858, 0x3ed921fb544387ba),
        (748982, 1048576, 0xbfcc7bbae425fd59, 0xbfef32999fc1ae5e),
        (786433, 1048576, 0x3ed921fb544387ba, 0xbfeffffffffd8858),
        (1048575, 1048576, 0x3feffffffffd8858, 0xbed921fb544387ba),
        (1, 2160, 0x3feffff7209c0799, 0x3f67d45bf9f2cd4b),
        (269, 2160, 0x3fe6b171bb5e3434, 0x3fe68fbe85650f51),
        (270, 2160, 0x3fe6a09e667f3bcd, 0x3fe6a09e667f3bcd),
        (271, 2160, 0x3fe68fbe85650f51, 0x3fe6b171bb5e3434),
        (720, 2160, 0xbfe0000000000000, 0x3febb67ae8584caa),
        (1079, 2160, 0xbfeffff7209c0799, 0x3f67d45bf9f2cd4b),
        (1542, 2160, 0xbfcccb3236cdc675, 0xbfef2e0a214e870f),
        (1621, 2160, 0x3f67d45bf9f2cd4b, 0xbfeffff7209c0799),
        (2159, 2160, 0x3feffff7209c0799, 0xbf67d45bf9f2cd4b),
    ];

    /// The gap from |x| to the next `f64` away from zero; zero for zero, so
    /// an exact zero must be met exactly.
    fn ulp(x: f64) -> f64 {
        let a = x.abs();
        if a == 0.0 {
            0.0
        } else {
            f64::from_bits(a.to_bits() + 1) - a
        }
    }

    fn data(rng: &mut Rng, n: usize) -> Vec<(f64, f64)> {
        (0..n).map(|_| (rng.value(), rng.value())).collect()
    }

    fn as_f64<C: Cplx>(v: &[C]) -> Vec<(f64, f64)> {
        v.iter().map(|z| z.to_f64()).collect()
    }

    fn of<C: Cplx>(x: &[(f64, f64)]) -> Vec<C> {
        x.iter().map(|&(re, im)| C::from_f64(re, im)).collect()
    }

    #[test]
    fn roots_of_unity_lie_within_two_units_in_the_last_place() {
        for &(k, n, c_bits, s_bits) in &REFERENCE_ROOTS {
            let (c, s) = unit_root(k, n);
            let (rc, rs) = (f64::from_bits(c_bits), f64::from_bits(s_bits));
            assert!(
                (c - rc).abs() <= 2.0 * ulp(rc),
                "cos 2π·{k}/{n}: {c:e} vs {rc:e}"
            );
            assert!(
                (s - rs).abs() <= 2.0 * ulp(rs),
                "sin 2π·{k}/{n}: {s:e} vs {rs:e}"
            );
        }
    }

    #[test]
    fn the_oracles_roots_round_to_the_reference() {
        for &(k, n, c_bits, s_bits) in &REFERENCE_ROOTS {
            let (c, s) = reference_root(k, n);
            // `==` so that a zero matches whatever its sign.
            let (rc, rs) = (f64::from_bits(c_bits), f64::from_bits(s_bits));
            assert!(
                c == rc && s == rs,
                "2π·{k}/{n}: ({c:e}, {s:e}) vs ({rc:e}, {rs:e})"
            );
        }
    }

    #[test]
    fn quarter_turns_are_exact() {
        for n in [4usize, 8, 12, 1000, 1 << 20] {
            for (q, expected) in [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]
                .into_iter()
                .enumerate()
            {
                assert_eq!(unit_root(q * n / 4, n), expected, "{q}/4 of {n}");
            }
        }
        // The index is taken modulo the order.
        assert_eq!(unit_root(13, 12), unit_root(1, 12));
    }

    #[test]
    fn single_precision_roots_lie_within_mu_of_the_exact_root() {
        let distance = |w: ComplexF32, (c, s): (f64, f64)| {
            ((f64::from(w.re) - c).powi(2) + (f64::from(w.im) + s).powi(2)).sqrt()
        };
        for len in [2usize, 8, 64, 1024, 1 << 14] {
            let t = Radix2::<ComplexF32>::new(len);
            for (j, &w) in t.twiddles().iter().enumerate() {
                assert!(
                    distance(w, unit_root(j, len)) <= MU32,
                    "twiddle {j} of {len}"
                );
            }
        }
        for len in [3usize, 5, 1000, 1080] {
            let t = Bluestein::<ComplexF32>::new(len);
            for (j, &w) in t.chirp().iter().enumerate() {
                let index = (j * j) % (2 * len);
                assert!(
                    distance(w, unit_root(index, 2 * len)) <= MU32,
                    "chirp {j} of {len}"
                );
            }
        }
    }

    #[test]
    fn the_chirp_index_is_reduced_in_integers() {
        // j² exceeds 2⁵³ here: formed in f64, j² mod 2n comes out as
        // 100000070, one step of π/n away from the exact 100000071.
        let (j, len) = (99_999_999_usize, 100_000_007_usize);
        assert_eq!(chirp_index(j, len), 100_000_071);
        assert_eq!((j as f64 * j as f64) % (2 * len) as f64, 100_000_070.0);
    }

    #[test]
    fn radix2_agrees_with_the_definition_within_highams_bound() {
        let mut rng = Rng::new(10);
        for t in 0..=11 {
            let n = 1usize << t;
            let x = data(&mut rng, n);
            let reference = naive(&x);

            let mut single: Vec<ComplexF32> = of(&x);
            Radix2::new(n).forward(&mut single);
            let err = rel_error(&as_f64(&single), &reference);
            let tol = higham(n, U32, MU32) + 2.0 * oracle_error(n);
            assert!(err <= tol, "f32, n = {n}: {err:e} > {tol:e}");

            let mut double: Vec<ComplexF64> = of(&x);
            Radix2::new(n).forward(&mut double);
            let err = rel_error(&as_f64(&double), &reference);
            let tol = higham(n, U64, MU64) + 2.0 * oracle_error(n);
            assert!(err <= tol, "f64, n = {n}: {err:e} > {tol:e}");
        }
    }

    /// The sizes Bluestein's transform is tested at: every length up to 40,
    /// powers of two and their neighbours, primes, and common image sides.
    const BLUESTEIN_SIZES: [usize; 13] = [
        97, 127, 128, 129, 240, 255, 256, 257, 480, 640, 1000, 1080, 1920,
    ];

    #[test]
    fn bluestein_agrees_with_the_definition() {
        let mut rng = Rng::new(20);
        let (mut worst32, mut worst64) = (0.0_f64, 0.0_f64);
        for n in (0..=40).chain(BLUESTEIN_SIZES) {
            let x = data(&mut rng, n);
            let reference = naive(&x);
            let mut work = Vec::new();

            let mut single: Vec<ComplexF32> = of(&x);
            Bluestein::new(n).forward(&mut single, &mut work);
            worst32 = worst32.max(rel_error(&as_f64(&single), &reference));

            let mut work = Vec::new();
            let mut double: Vec<ComplexF64> = of(&x);
            Bluestein::new(n).forward(&mut double, &mut work);
            worst64 = worst64.max(rel_error(&as_f64(&double), &reference));
        }
        assert!(
            worst32 <= BLUESTEIN_F32 && worst64 <= BLUESTEIN_F64,
            "worst f32 {worst32:e} = {:.2} u, worst f64 {worst64:e} = {:.2} u",
            worst32 / U32,
            worst64 / U64
        );
    }

    #[test]
    fn the_inverse_undoes_the_forward_transform() {
        let mut rng = Rng::new(30);
        for n in [0usize, 1, 2, 3, 7, 16, 100, 128] {
            let x = data(&mut rng, n);
            let axis: Axis<ComplexF64> = if n.is_power_of_two() || n == 0 {
                Axis::Radix2(Radix2::new(n))
            } else {
                Axis::Bluestein(Bluestein::new(n))
            };
            let mut buf: Vec<ComplexF64> = of(&x);
            let mut work = Vec::new();
            axis.forward(&mut buf, &mut work);
            axis.inverse_unscaled(&mut buf, &mut work);
            let back: Vec<ComplexF64> = buf.iter().map(|z| z.div_count(n)).collect();
            let err = rel_error(&as_f64(&back), &x);
            assert!(err <= 1e-13, "n = {n}: {err:e}");
        }
    }

    #[test]
    fn lengths_zero_and_one_change_nothing() {
        let mut work = Vec::new();
        let mut empty: Vec<ComplexF32> = Vec::new();
        Radix2::new(0).forward(&mut empty);
        Bluestein::new(0).forward(&mut empty, &mut work);
        assert!(empty.is_empty());
        let z = ComplexF32::new(1.5, -2.25);
        let mut one = vec![z];
        Radix2::new(1).forward(&mut one);
        assert_eq!(one, [z]);
        Bluestein::new(1).forward(&mut one, &mut work);
        assert_eq!(one, [z]);
    }

    #[test]
    fn radix2_and_bluestein_compute_the_same_transform() {
        let mut rng = Rng::new(40);
        let x = data(&mut rng, 64);
        let mut a: Vec<ComplexF64> = of(&x);
        let mut b = a.clone();
        Radix2::new(64).forward(&mut a);
        Bluestein::new(64).forward(&mut b, &mut Vec::new());
        assert!(rel_error(&as_f64(&a), &as_f64(&b)) <= 1e-14);
    }

    #[test]
    fn the_transpose_moves_every_value_across_blocks() {
        for (w, h) in [
            (0usize, 0usize),
            (1, 5),
            (5, 1),
            (33, 70),
            (64, 64),
            (100, 3),
        ] {
            let src: Vec<usize> = (0..w * h).collect();
            let mut dst = vec![usize::MAX; w * h];
            transpose(&src, w, h, &mut dst);
            for y in 0..h {
                for x in 0..w {
                    assert_eq!(dst[x * h + y], src[y * w + x], "({x}, {y}) of {w}x{h}");
                }
            }
        }
    }
}

//! Test support for the frequency module: the DFT by its definition as the
//! oracle, the tolerances the tests assert, and a seeded generator.
//!
//! A tolerance here is a criterion of the tests and promises callers
//! nothing.

/// The unit roundoff of `f32`, 2⁻²⁴.
pub(crate) const U32: f64 = 1.0 / (1u64 << 24) as f64;
/// The unit roundoff of `f64`, 2⁻⁵³.
pub(crate) const U64: f64 = f64::EPSILON / 2.0;

/// How far a stored `f32` root of unity may lie from the exact one: it is
/// an `f64` root, accurate to a few units of `f64`, rounded once.
pub(crate) const MU32: f64 = 1.001 * U32;
/// How far an `f64` root of unity may lie from the exact one: each part
/// within two units in the last place of the correctly rounded value, so
/// within 2.5 units of the exact one, which is 5u of the root's modulus.
pub(crate) const MU64: f64 = 5.0 * U64;

/// The worst-case error of a radix-2 transform of `points` points, relative
/// to the whole result: Higham's bound, as restated by Brisebarre, Joldeş,
/// Muller, Naneş and Picot (ACM TOMS 46(2), 2020, Theorem 3), t·η / (1 − t·η)
/// for 2ᵗ points, with η = μ + 4u / (1 − 4u) · (√2 + μ), where μ bounds the
/// error of the roots of unity.
pub(crate) fn higham(points: usize, u: f64, mu: f64) -> f64 {
    if points <= 1 {
        return 0.0;
    }
    let t = f64::from(points.trailing_zeros());
    let eta = mu + 4.0 * u / (1.0 - 4.0 * u) * (std::f64::consts::SQRT_2 + mu);
    t * eta / (1.0 - t * eta)
}

/// How far [`naive`] may lie from the exact DFT of `points` points,
/// relative to the whole result. It computes in double-double arithmetic,
/// about 106 bits, so its error is of the order of n·u², some ten orders
/// of magnitude below any `f64` transform; the factor is generous.
pub(crate) fn oracle_error(points: usize) -> f64 {
    1000.0 * points as f64 * U64 * U64
}

/// The tolerance of a transform whose two passes are each within `a` and
/// `b` relative to the whole result: a + b + a·b.
pub(crate) fn compose(a: f64, b: f64) -> f64 {
    a + b + a * b
}

/// The largest error Bluestein's transform showed against [`naive`] over
/// the sizes of the engine's test, relative to the whole result, times a
/// margin of 4: 4.20·u, measured on 2026-10-05.
pub(crate) const BLUESTEIN_F32: f64 = 4.0 * 4.21 * U32;
/// As [`BLUESTEIN_F32`], for `f64`: 4.08·u, measured on 2026-10-05.
pub(crate) const BLUESTEIN_F64: f64 = 4.0 * 4.09 * U64;

/// A double-double number `hi + lo`, |lo| ≤ ulp(hi)/2: about 106
/// significant bits from two `f64`.
#[derive(Clone, Copy, Debug)]
struct Dd {
    hi: f64,
    lo: f64,
}

/// `a + b` exactly, as a rounded sum and its error (Knuth).
fn two_sum(a: f64, b: f64) -> Dd {
    let s = a + b;
    let v = s - a;
    Dd {
        hi: s,
        lo: (a - (s - v)) + (b - v),
    }
}

/// `a · b` exactly, as a rounded product and its error, through a fused
/// multiply-add.
fn two_prod(a: f64, b: f64) -> Dd {
    let p = a * b;
    Dd {
        hi: p,
        lo: a.mul_add(b, -p),
    }
}

impl Dd {
    const ZERO: Dd = Dd { hi: 0.0, lo: 0.0 };

    fn from(v: f64) -> Dd {
        Dd { hi: v, lo: 0.0 }
    }

    fn normalized(hi: f64, lo: f64) -> Dd {
        let s = hi + lo;
        Dd {
            hi: s,
            lo: lo - (s - hi),
        }
    }

    fn add(self, o: Dd) -> Dd {
        let s = two_sum(self.hi, o.hi);
        let t = two_sum(self.lo, o.lo);
        let u = Dd::normalized(s.hi, s.lo + t.hi);
        Dd::normalized(u.hi, u.lo + t.lo)
    }

    fn neg(self) -> Dd {
        Dd {
            hi: -self.hi,
            lo: -self.lo,
        }
    }

    fn mul(self, o: Dd) -> Dd {
        let p = two_prod(self.hi, o.hi);
        Dd::normalized(p.hi, p.lo + (self.hi * o.lo + self.lo * o.hi))
    }

    fn mul_f64(self, b: f64) -> Dd {
        let p = two_prod(self.hi, b);
        Dd::normalized(p.hi, p.lo + self.lo * b)
    }

    /// `a / b` for integers held exactly in `f64`.
    fn ratio(a: f64, b: f64) -> Dd {
        let q = a / b;
        let r = (-q).mul_add(b, a);
        Dd::normalized(q, r / b)
    }

    fn to_f64(self) -> f64 {
        self.hi + self.lo
    }
}

/// π/4 as a double-double.
const FRAC_PI_4_DD: Dd = Dd {
    hi: std::f64::consts::FRAC_PI_4,
    lo: 3.061616997868383e-17,
};

/// cos φ and sin φ by their Taylor series, for 0 ≤ φ ≤ π/4, in
/// double-double arithmetic.
fn cos_sin_dd(phi: Dd) -> (Dd, Dd) {
    let mut cos = Dd::ZERO;
    let mut sin = Dd::ZERO;
    let mut term = Dd::from(1.0); // φ⁰ / 0!
    for i in 0..40 {
        match i % 4 {
            0 => cos = cos.add(term),
            1 => sin = sin.add(term),
            2 => cos = cos.add(term.neg()),
            _ => sin = sin.add(term.neg()),
        }
        term = term.mul(phi).mul(Dd::ratio(1.0, f64::from(i + 1)));
    }
    (cos, sin)
}

/// cos(2π·k/n) and sin(2π·k/n) in double-double, reduced to the first
/// octant in integers as the engine does.
fn unit_root_dd(k: usize, n: usize) -> (Dd, Dd) {
    let n_wide = n as u128;
    let eighths = 8 * (k as u128 % n_wide);
    let q = eighths / n_wide;
    let r = eighths % n_wide;
    let from_axis = if q % 2 == 0 { r } else { n_wide - r };
    let phi = FRAC_PI_4_DD.mul(Dd::ratio(from_axis as f64, n as f64));
    let (c, s) = cos_sin_dd(phi);
    match q {
        0 => (c, s),
        1 => (s, c),
        2 => (s.neg(), c),
        3 => (c.neg(), s),
        4 => (c.neg(), s.neg()),
        5 => (s.neg(), c.neg()),
        6 => (s, c.neg()),
        _ => (c, s.neg()),
    }
}

/// cos(2π·k/n) and sin(2π·k/n) from the double-double computation,
/// rounded to `f64`.
pub(crate) fn reference_root(k: usize, n: usize) -> (f64, f64) {
    let (c, s) = unit_root_dd(k, n);
    (c.to_f64(), s.to_f64())
}

/// The DFT of `x` by its definition, `X[k] = Σ x[j]·exp(−2πi·jk/n)`, in
/// double-double arithmetic, each root of unity from the exact index
/// jk mod n; rounded to `f64` at the end.
pub(crate) fn naive(x: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let n = x.len();
    let roots: Vec<(Dd, Dd)> = (0..n).map(|k| unit_root_dd(k, n)).collect();
    (0..n)
        .map(|k| {
            let mut re = Dd::ZERO;
            let mut im = Dd::ZERO;
            for (j, &(a, b)) in x.iter().enumerate() {
                let (c, s) = roots[((j as u128 * k as u128) % n as u128) as usize];
                // (a + bi)(c − si) = (ac + bs) + (bc − as)i
                re = re.add(c.mul_f64(a)).add(s.mul_f64(b));
                im = im.add(c.mul_f64(b)).add(s.mul_f64(a).neg());
            }
            (re.to_f64(), im.to_f64())
        })
        .collect()
}

/// The full spectrum of a real image, `height` rows of `width` values,
/// by [`naive`] along the rows and then along the columns; row-major,
/// `height` rows of `width` bins.
pub(crate) fn naive_2d(x: &[f64], width: usize, height: usize) -> Vec<(f64, f64)> {
    let mut rows = Vec::with_capacity(width * height);
    for row in x.chunks_exact(width.max(1)).take(height) {
        let complex: Vec<(f64, f64)> = row.iter().map(|&v| (v, 0.0)).collect();
        rows.extend(naive(&complex));
    }
    let mut out = vec![(0.0, 0.0); width * height];
    for kx in 0..width {
        let column: Vec<(f64, f64)> = (0..height).map(|y| rows[y * width + kx]).collect();
        for (ky, v) in naive(&column).into_iter().enumerate() {
            out[ky * width + kx] = v;
        }
    }
    out
}

/// The Euclidean norm of the difference, relative to the norm of
/// `reference`; zero when both are zero.
pub(crate) fn rel_error(a: &[(f64, f64)], reference: &[(f64, f64)]) -> f64 {
    assert_eq!(a.len(), reference.len());
    let diff: f64 = a
        .iter()
        .zip(reference)
        .map(|(&(ar, ai), &(br, bi))| (ar - br).powi(2) + (ai - bi).powi(2))
        .sum();
    let norm: f64 = reference.iter().map(|&(r, i)| r * r + i * i).sum();
    if norm == 0.0 {
        diff.sqrt()
    } else {
        (diff / norm).sqrt()
    }
}

/// A seeded xorshift generator, so a test sees the same data on every run.
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// A value in [−1, 1) with 24 significant bits, so `f32` holds it
    /// exactly.
    pub(crate) fn value(&mut self) -> f64 {
        (self.next_u64() >> 40) as f64 / (1u64 << 23) as f64 - 1.0
    }

    /// A value in [−1, 1) with 12 significant bits, so that small linear
    /// combinations of such values are exact in `f32` as well.
    pub(crate) fn coarse(&mut self) -> f64 {
        (self.next_u64() >> 52) as f64 / (1u64 << 11) as f64 - 1.0
    }
}

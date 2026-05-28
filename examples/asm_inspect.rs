//! Assembly inspection harness for verifying SIMD auto-vectorization.
//!
//! Each function is isolated, `#[inline(never)]`, and `#[no_mangle]` so it
//! appears as a distinct symbol in the assembly output.  Compile with:
//!
//! ```sh
//! cargo asm -p irys-cv --example asm_inspect --target-cpu native <symbol>
//! ```
//!
//! Useful symbols to inspect:
//!
//! - `fold_convolve_u8_hot`       — u8→f32 convolution inner loop (FoldOp)
//! - `fold_convolve_f32_hot`      — f32 convolution inner loop (FoldOp)
//! - `map_erode_u8_hot`           — u8 erosion inner loop (MapOp)
//! - `map_dilate_u8_hot`          — u8 dilation inner loop (MapOp)
//! - `real_convolve_u8`           — full `convolve` pipeline on u8
//! - `real_erode_u8`              — full `erode` pipeline on u8
//! - `real_dilate_u8`             — full `dilate` pipeline on u8
//! - `real_fold_neighborhood_u8`  — full `fold_neighborhood` with DirectSumFold
//!
//! Look for:
//! - `vpminub` / `vpmaxub` — packed u8 min/max (erode/dilate)
//! - `vfmadd231ps` / `vfmadd213ps` — fused multiply-add (convolution f32)
//! - `vmulps` / `vaddps` — packed f32 mul/add (convolution without FMA)
//! - `vcvtdq2ps` / `vpmovzxbd` — u8→f32 conversion chain (u8 convolution)
//! - `vmovdqu` / `vmovups` — packed loads (any vectorised path)

use irys_cv::border::Clamp;
use irys_cv::image::{Image, Kernel3x3, Mask3x3, RasterImage};
use irys_cv::transform::{
    FoldItem, FoldOp, MapItem, MapOp, convolve, dilate, erode, fold_neighborhood,
};
use std::hint::black_box;

// ═══════════════════════════════════════════════════════════════════════════════
// Isolated inner-loop micro-kernels
// ═══════════════════════════════════════════════════════════════════════════════

/// Simulates the fold_neighborhood_into inverted inner loop for u8 convolution.
///
/// The loop body is: acc[i] += src[i] as f32 * weight
///
/// With AVX2+FMA you should see: vpmovzxbd → vcvtdq2ps → vfmadd231ps
/// (or vmulps + vaddps without FMA).
#[unsafe(no_mangle)]
#[inline(never)]
pub fn fold_convolve_u8_hot(acc: &mut [f32], src: &[u8], weight: f32) {
    let n = acc.len().min(src.len());
    for i in 0..n {
        acc[i] += src[i] as f32 * weight;
    }
}

/// Simulates the fold_neighborhood_into inverted inner loop for f32 convolution.
///
/// The loop body is: acc[i] += src[i] * weight
///
/// With FMA you should see: vfmadd231ps (or vmulps + vaddps).
#[unsafe(no_mangle)]
#[inline(never)]
pub fn fold_convolve_f32_hot(acc: &mut [f32], src: &[f32], weight: f32) {
    let n = acc.len().min(src.len());
    for i in 0..n {
        acc[i] += src[i] * weight;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// FMA variants — using f32::mul_add to explicitly request fused multiply-add
// ═══════════════════════════════════════════════════════════════════════════════

/// Same as fold_convolve_u8_hot but using `f32::mul_add` to request FMA.
///
/// Expected: vpmovzxbd → vcvtdq2ps → vfmadd231ps (single fused instruction
/// replacing separate vmulps + vaddps).
#[unsafe(no_mangle)]
#[inline(never)]
pub fn fold_convolve_u8_fma(acc: &mut [f32], src: &[u8], weight: f32) {
    let n = acc.len().min(src.len());
    for i in 0..n {
        acc[i] = (src[i] as f32).mul_add(weight, acc[i]);
    }
}

/// Same as fold_convolve_f32_hot but using `f32::mul_add` to request FMA.
///
/// Expected: vfmadd231ps (or vfmadd213ps) — single fused instruction.
#[unsafe(no_mangle)]
#[inline(never)]
pub fn fold_convolve_f32_fma(acc: &mut [f32], src: &[f32], weight: f32) {
    let n = acc.len().min(src.len());
    for i in 0..n {
        acc[i] = src[i].mul_add(weight, acc[i]);
    }
}

/// Multi-offset u8 convolution with FMA — the full kernel-outer, pixel-inner
/// pattern using mul_add.
#[unsafe(no_mangle)]
#[inline(never)]
pub fn fold_multi_offset_u8_fma(acc: &mut [f32], rows: &[(&[u8], f32)]) {
    let n = acc.len();
    for &(src, w) in rows {
        let len = n.min(src.len());
        for i in 0..len {
            acc[i] = (src[i] as f32).mul_add(w, acc[i]);
        }
    }
}

/// Multi-offset f32 convolution with FMA.
#[unsafe(no_mangle)]
#[inline(never)]
pub fn fold_multi_offset_f32_fma(acc: &mut [f32], rows: &[(&[f32], f32)]) {
    let n = acc.len();
    for &(src, w) in rows {
        let len = n.min(src.len());
        for i in 0..len {
            acc[i] = src[i].mul_add(w, acc[i]);
        }
    }
}

/// Simulates the map_neighborhood_into inverted inner loop for u8 erosion.
///
/// The loop body is: acc[i] = min(acc[i], src[i])
///
/// With AVX2 you should see: vpminub (packed unsigned byte min).
#[unsafe(no_mangle)]
#[inline(never)]
pub fn map_erode_u8_hot(acc: &mut [u8], src: &[u8]) {
    let n = acc.len().min(src.len());
    for i in 0..n {
        acc[i] = acc[i].min(src[i]);
    }
}

/// Simulates the map_neighborhood_into inverted inner loop for u8 dilation.
///
/// The loop body is: acc[i] = max(acc[i], src[i])
///
/// With AVX2 you should see: vpmaxub (packed unsigned byte max).
#[unsafe(no_mangle)]
#[inline(never)]
pub fn map_dilate_u8_hot(acc: &mut [u8], src: &[u8]) {
    let n = acc.len().min(src.len());
    for i in 0..n {
        acc[i] = acc[i].max(src[i]);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Trait-based inner loops (through FoldOp/MapOp)
// ═══════════════════════════════════════════════════════════════════════════════

/// A direct FoldOp for weighted sum (same as ConvolveFold but visible here).
struct DirectSumFold;

// ADR-0045 Phase S4: `u8` no longer implements `LinearPixel`, so the
// fold pipelines that go through pixel-role traits migrated from
// `u8` to `Mono8`. `Mono8` is `#[repr(transparent)]` over
// `Saturating<u8>`, so codegen is equivalent to the old `u8` path.
impl FoldOp<irys_cv::pixel::Mono8, f32> for DirectSumFold {
    type Accumulator = f32;
    // ADR-0044 Phase E: `f32` no longer implements `ZeroablePixel`, so the
    // `Output` of a `FoldOp` used with `fold_neighborhood` must be a real
    // pixel type. `MonoF32` is `#[repr(transparent)]` over `f32`, so this
    // is a zero-cost wrapping of the scalar weighted-sum result.
    type Output = irys_cv::pixel::MonoF32;

    #[inline(always)]
    fn init(&self) -> f32 {
        0.0
    }

    #[inline(always)]
    fn accumulate(&self, acc: &mut f32, item: FoldItem<irys_cv::pixel::Mono8, f32>) {
        *acc += item.pixel.value() as f32 * item.weight;
    }

    #[inline(always)]
    fn finalize(&mut self, acc: f32) -> irys_cv::pixel::MonoF32 {
        irys_cv::pixel::MonoF32::new(acc)
    }
}

impl FoldOp<f32, f32> for DirectSumFold {
    type Accumulator = f32;
    type Output = f32;

    #[inline(always)]
    fn init(&self) -> f32 {
        0.0
    }

    #[inline(always)]
    fn accumulate(&self, acc: &mut f32, item: FoldItem<f32, f32>) {
        *acc += item.pixel * item.weight;
    }

    #[inline(always)]
    fn finalize(&mut self, acc: f32) -> f32 {
        acc
    }
}

/// Trait-based inner loop for u8 FoldOp convolution.
///
/// Tests whether monomorphisation through FoldOp::accumulate preserves
/// vectorisation.
#[unsafe(no_mangle)]
#[inline(never)]
pub fn fold_trait_convolve_u8_hot(acc: &mut [f32], src: &[u8], weight: f32) {
    let op = DirectSumFold;
    let n = acc.len().min(src.len());
    for i in 0..n {
        // ADR-0045 Phase S4: `DirectSumFold` now implements
        // `FoldOp<Mono8, f32>` (not `FoldOp<u8, f32>`). Wrap the raw
        // `u8` byte into `Mono8` at the call site — this is a
        // zero-cost reinterpretation because `Mono8` is
        // `#[repr(transparent)]` over `Saturating<u8>`.
        op.accumulate(
            &mut acc[i],
            FoldItem {
                pixel: irys_cv::pixel::Mono8::new(src[i]),
                weight,
            },
        );
    }
}

/// Trait-based inner loop for f32 FoldOp convolution.
#[unsafe(no_mangle)]
#[inline(never)]
pub fn fold_trait_convolve_f32_hot(acc: &mut [f32], src: &[f32], weight: f32) {
    let op = DirectSumFold;
    let n = acc.len().min(src.len());
    for i in 0..n {
        op.accumulate(
            &mut acc[i],
            FoldItem {
                pixel: src[i],
                weight,
            },
        );
    }
}

/// Trait-based inner loop for u8 MapOp erosion (min).
struct ErodeInspect;

impl MapOp<u8> for ErodeInspect {
    type Accumulator = u8;
    type Output = u8;

    #[inline(always)]
    fn init(&self, center: u8) -> u8 {
        center
    }

    #[inline(always)]
    fn accumulate(&self, acc: &mut u8, item: MapItem<u8>) {
        *acc = (*acc).min(item.pixel);
    }

    #[inline(always)]
    fn finalize(&mut self, acc: u8) -> u8 {
        acc
    }
}

struct DilateInspect;

impl MapOp<u8> for DilateInspect {
    type Accumulator = u8;
    type Output = u8;

    #[inline(always)]
    fn init(&self, center: u8) -> u8 {
        center
    }

    #[inline(always)]
    fn accumulate(&self, acc: &mut u8, item: MapItem<u8>) {
        *acc = (*acc).max(item.pixel);
    }

    #[inline(always)]
    fn finalize(&mut self, acc: u8) -> u8 {
        acc
    }
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn map_trait_erode_u8_hot(acc: &mut [u8], src: &[u8]) {
    let op = ErodeInspect;
    let n = acc.len().min(src.len());
    for i in 0..n {
        op.accumulate(
            &mut acc[i],
            MapItem {
                pixel: src[i],
                dx: 0,
                dy: 0,
            },
        );
    }
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn map_trait_dilate_u8_hot(acc: &mut [u8], src: &[u8]) {
    let op = DilateInspect;
    let n = acc.len().min(src.len());
    for i in 0..n {
        op.accumulate(
            &mut acc[i],
            MapItem {
                pixel: src[i],
                dx: 0,
                dy: 0,
            },
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Multi-offset loops (simulating the kernel-outer, pixel-inner pattern)
// ═══════════════════════════════════════════════════════════════════════════════

/// Simulates the complete inverted-loop pattern for u8 convolution:
/// kernel offsets are outer, pixel scan is inner.
///
/// `rows` is a Vec of (src_slice, weight) pairs pre-sliced for each
/// kernel position — this mirrors what fold_neighborhood_into builds.
#[unsafe(no_mangle)]
#[inline(never)]
pub fn fold_multi_offset_u8(acc: &mut [f32], rows: &[(&[u8], f32)]) {
    let n = acc.len();
    for &(src, w) in rows {
        let len = n.min(src.len());
        for i in 0..len {
            acc[i] += src[i] as f32 * w;
        }
    }
}

/// Same pattern for f32 convolution.
#[unsafe(no_mangle)]
#[inline(never)]
pub fn fold_multi_offset_f32(acc: &mut [f32], rows: &[(&[f32], f32)]) {
    let n = acc.len();
    for &(src, w) in rows {
        let len = n.min(src.len());
        for i in 0..len {
            acc[i] += src[i] * w;
        }
    }
}

/// Multi-offset erode (u8 min across kernel positions).
#[unsafe(no_mangle)]
#[inline(never)]
pub fn map_multi_offset_erode_u8(acc: &mut [u8], rows: &[&[u8]]) {
    let n = acc.len();
    for &src in rows {
        let len = n.min(src.len());
        for i in 0..len {
            acc[i] = acc[i].min(src[i]);
        }
    }
}

/// Multi-offset dilate (u8 max across kernel positions).
#[unsafe(no_mangle)]
#[inline(never)]
pub fn map_multi_offset_dilate_u8(acc: &mut [u8], rows: &[&[u8]]) {
    let n = acc.len();
    for &src in rows {
        let len = n.min(src.len());
        for i in 0..len {
            acc[i] = acc[i].max(src[i]);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Full pipeline calls (the real public API)
// ═══════════════════════════════════════════════════════════════════════════════

#[unsafe(no_mangle)]
#[inline(never)]
pub fn real_convolve_u8(
    img: &Image<irys_cv::pixel::Mono8>,
    kernel: &Kernel3x3,
) -> Image<irys_cv::pixel::MonoF32> {
    // ADR-0045 Phase C: `Mono8::Accumulator = MonoF32`, so the
    // `Out` turbofish flips from raw `f32` to the named pixel type.
    convolve::<_, _, _, _, irys_cv::pixel::MonoF32>(img, kernel, &Clamp)
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn real_erode_u8(img: &Image<u8>, se: &Mask3x3) -> Image<u8> {
    erode(img, se, &Clamp)
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn real_dilate_u8(img: &Image<u8>, se: &Mask3x3) -> Image<u8> {
    dilate(img, se, &Clamp)
}

#[unsafe(no_mangle)]
#[inline(never)]
// Scalar fold accumulator — kept as `f32` per ADR-0044 §C.3 (not a pixel role).
// `DirectSumFold::Accumulator = f32` is a scalar weighted-sum accumulator,
// so the output `Image<f32>` carries scalar semantics rather than a pixel
// intensity. No migration to `MonoF32` under ADR-0045 Phase C.
pub fn real_fold_neighborhood_u8(
    img: &Image<irys_cv::pixel::Mono8>,
    kernel: &Kernel3x3,
) -> Image<irys_cv::pixel::MonoF32> {
    fold_neighborhood(
        img,
        kernel.weights(),
        kernel.anchor(),
        &Clamp,
        DirectSumFold,
    )
}

// ═══════════════════════════════════════════════════════════════════════════════
// Bounds-check experiment: assert equal lengths before the loop
// ═══════════════════════════════════════════════════════════════════════════════

/// Same as fold_convolve_u8_hot but with an explicit assert to help LLVM
/// eliminate bounds checks.
#[unsafe(no_mangle)]
#[inline(never)]
pub fn fold_convolve_u8_asserted(acc: &mut [f32], src: &[u8], weight: f32) {
    assert!(acc.len() == src.len());
    let n = acc.len();
    for i in 0..n {
        acc[i] += src[i] as f32 * weight;
    }
}

/// Same for erode.
#[unsafe(no_mangle)]
#[inline(never)]
pub fn map_erode_u8_asserted(acc: &mut [u8], src: &[u8]) {
    assert!(acc.len() == src.len());
    let n = acc.len();
    for i in 0..n {
        acc[i] = acc[i].min(src[i]);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Iterator-based zip pattern (alternative inner loop shape)
// ═══════════════════════════════════════════════════════════════════════════════

/// Uses zip() instead of manual indexing — this sometimes helps LLVM
/// prove aliasing properties and emit better code.
#[unsafe(no_mangle)]
#[inline(never)]
pub fn fold_convolve_u8_zip(acc: &mut [f32], src: &[u8], weight: f32) {
    for (a, &s) in acc.iter_mut().zip(src.iter()) {
        *a += s as f32 * weight;
    }
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn map_erode_u8_zip(acc: &mut [u8], src: &[u8]) {
    for (a, &s) in acc.iter_mut().zip(src.iter()) {
        *a = (*a).min(s);
    }
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn fold_convolve_f32_zip(acc: &mut [f32], src: &[f32], weight: f32) {
    for (a, &s) in acc.iter_mut().zip(src.iter()) {
        *a += s * weight;
    }
}

/// Zip-based u8 FMA variant.
#[unsafe(no_mangle)]
#[inline(never)]
pub fn fold_convolve_u8_zip_fma(acc: &mut [f32], src: &[u8], weight: f32) {
    for (a, &s) in acc.iter_mut().zip(src.iter()) {
        *a = (s as f32).mul_add(weight, *a);
    }
}

/// Zip-based f32 FMA variant.
#[unsafe(no_mangle)]
#[inline(never)]
pub fn fold_convolve_f32_zip_fma(acc: &mut [f32], src: &[f32], weight: f32) {
    for (a, &s) in acc.iter_mut().zip(src.iter()) {
        *a = s.mul_add(weight, *a);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main — exercises all paths so nothing is dead-code-eliminated
// ═══════════════════════════════════════════════════════════════════════════════

fn main() {
    let w = 256;
    let h = 256;
    let img_u8 = Image::generate(w, h, |x, y| ((x * 17 + y * 31) % 256) as u8);
    // ADR-0045 Phase S4: `u8` is no longer a pixel. For the pipeline
    // calls that go through `LinearPixel` (convolve, fold_neighborhood),
    // build a parallel `Image<Mono8>` with the same underlying bytes.
    let img_mono8: Image<irys_cv::pixel::Mono8> = Image::generate(w, h, |x, y| {
        irys_cv::pixel::Mono8::new(((x * 17 + y * 31) % 256) as u8)
    });
    let img_f32 = Image::generate(w, h, |x, y| (x * 17 + y * 31) as f32 / 256.0);
    let kernel = Kernel3x3::gaussian_3x3();
    let se = Mask3x3::full_rect_3x3();

    // ── Micro-kernels ────────────────────────────────────────────────────
    let row_u8 = img_u8.row(10);
    let row_f32 = img_f32.row(10);

    let mut acc_f32 = vec![0.0f32; w];
    let mut acc_u8: Vec<u8> = img_u8.row(5).to_vec();

    fold_convolve_u8_hot(black_box(&mut acc_f32), black_box(row_u8), black_box(0.5));
    fold_convolve_f32_hot(black_box(&mut acc_f32), black_box(row_f32), black_box(0.5));
    fold_convolve_u8_fma(black_box(&mut acc_f32), black_box(row_u8), black_box(0.5));
    fold_convolve_f32_fma(black_box(&mut acc_f32), black_box(row_f32), black_box(0.5));
    map_erode_u8_hot(black_box(&mut acc_u8), black_box(row_u8));
    map_dilate_u8_hot(black_box(&mut acc_u8), black_box(row_u8));

    // ── Trait-based ──────────────────────────────────────────────────────
    acc_f32.fill(0.0);
    fold_trait_convolve_u8_hot(black_box(&mut acc_f32), black_box(row_u8), black_box(0.5));
    fold_trait_convolve_f32_hot(black_box(&mut acc_f32), black_box(row_f32), black_box(0.5));
    acc_u8 = img_u8.row(5).to_vec();
    map_trait_erode_u8_hot(black_box(&mut acc_u8), black_box(row_u8));
    map_trait_dilate_u8_hot(black_box(&mut acc_u8), black_box(row_u8));

    // ── Multi-offset ─────────────────────────────────────────────────────
    let rows_u8_w: Vec<(&[u8], f32)> = (0..9).map(|i| (img_u8.row(i + 1), 1.0 / 9.0)).collect();
    let rows_f32_w: Vec<(&[f32], f32)> = (0..9).map(|i| (img_f32.row(i + 1), 1.0 / 9.0)).collect();
    let rows_u8: Vec<&[u8]> = (0..9).map(|i| img_u8.row(i + 1)).collect();

    acc_f32.fill(0.0);
    fold_multi_offset_u8(black_box(&mut acc_f32), black_box(&rows_u8_w));
    fold_multi_offset_f32(black_box(&mut acc_f32), black_box(&rows_f32_w));
    fold_multi_offset_u8_fma(black_box(&mut acc_f32), black_box(&rows_u8_w));
    fold_multi_offset_f32_fma(black_box(&mut acc_f32), black_box(&rows_f32_w));

    acc_u8 = img_u8.row(5).to_vec();
    map_multi_offset_erode_u8(black_box(&mut acc_u8), black_box(&rows_u8));
    map_multi_offset_dilate_u8(black_box(&mut acc_u8), black_box(&rows_u8));

    // ── Asserted lengths ─────────────────────────────────────────────────
    acc_f32.fill(0.0);
    fold_convolve_u8_asserted(black_box(&mut acc_f32), black_box(row_u8), black_box(0.5));
    acc_u8 = img_u8.row(5).to_vec();
    map_erode_u8_asserted(black_box(&mut acc_u8), black_box(row_u8));

    // ── Zip-based ────────────────────────────────────────────────────────
    acc_f32.fill(0.0);
    fold_convolve_u8_zip(black_box(&mut acc_f32), black_box(row_u8), black_box(0.5));
    fold_convolve_f32_zip(black_box(&mut acc_f32), black_box(row_f32), black_box(0.5));
    fold_convolve_u8_zip_fma(black_box(&mut acc_f32), black_box(row_u8), black_box(0.5));
    fold_convolve_f32_zip_fma(black_box(&mut acc_f32), black_box(row_f32), black_box(0.5));
    acc_u8 = img_u8.row(5).to_vec();
    map_erode_u8_zip(black_box(&mut acc_u8), black_box(row_u8));

    // ── Full pipeline ────────────────────────────────────────────────────
    // `real_convolve_u8` and `real_fold_neighborhood_u8` now take
    // `Image<Mono8>` (ADR-0045 Phase S4); `real_erode_u8` /
    // `real_dilate_u8` do not go through `LinearPixel` and keep their
    // `Image<u8>` input.
    let _conv = real_convolve_u8(black_box(&img_mono8), black_box(&kernel));
    let _ero = real_erode_u8(black_box(&img_u8), black_box(&se));
    let _dil = real_dilate_u8(black_box(&img_u8), black_box(&se));
    let _fold = real_fold_neighborhood_u8(black_box(&img_mono8), black_box(&kernel));

    // Prevent DCE
    black_box(&acc_f32);
    black_box(&acc_u8);
    black_box(&_conv);
    black_box(&_ero);
    black_box(&_dil);
    black_box(&_fold);

    println!("asm_inspect: all functions exercised");
}

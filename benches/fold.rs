//! Benchmarks for `fold_neighborhood`, convolution, and morphology operations.
//!
//! Includes a direct **`FoldOp` vs `ClosureFold`** comparison (B15) that
//! demonstrates the performance difference between monomorphized `FoldOp`
//! structs and `ClosureFold` wrappers (which go through `dyn Iterator`
//! dispatch on every `.next()` call).
//!
//! Run with:
//! ```sh
//! cargo bench --bench fold
//! ```

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use irys_cv::border::Clamp;
use irys_cv::image::{
    Image, Kernel3x3, Kernel5x5, Kernel7x7, Mask3x3, Mask5x5, Neighborhood, SeparableKernel,
};
use irys_cv::pixel::{Mono8, MonoF32};
use irys_cv::transform::{
    ClosureFold, FoldItem, FoldOp, convolve, convolve_separable, dilate, erode, fold_neighborhood,
    opening,
};
use std::hint::black_box;

// ─── Helpers ─────────────────────────────────────────────────────────────────

// ADR-0045 Phase S4: channel primitives no longer implement `LinearPixel`.
// The bench inputs migrated from `Image<u8>` to `Image<Mono8>`; `Mono8` is
// `#[repr(transparent)]` over `Saturating<u8>`, so the bench layout and
// numerical path are unchanged.
fn make_u8_image(width: usize, height: usize) -> Image<Mono8> {
    Image::generate(width, height, |x, y| {
        Mono8::new(((x * 17 + y * 31) % 256) as u8)
    })
}

/// Identity fold: returns the anchor pixel value as f32.
/// Uses a weight-1 identity kernel; the anchor pixel comes through as the
/// sole item with weight == 1.0.
// ADR-0044 Phase E: `f32` no longer implements `ZeroablePixel`, so
// `fold_neighborhood`'s output pixel type migrated from `f32` to
// `MonoF32`. `MonoF32` is `#[repr(transparent)]` over `f32`, so the
// numerical path is unchanged. Closure accumulator remains scalar `f32`.
fn identity_fold()
-> ClosureFold<impl FnMut(&mut dyn Iterator<Item = FoldItem<Mono8, f32>>) -> MonoF32> {
    ClosureFold(
        |neighbors: &mut dyn Iterator<Item = FoldItem<Mono8, f32>>| {
            MonoF32::new(
                neighbors
                    .map(|item| item.pixel.value() as f32 * item.weight)
                    .sum(),
            )
        },
    )
}

/// Weighted-sum fold (same logic as convolution).
fn sum_fold() -> ClosureFold<impl FnMut(&mut dyn Iterator<Item = FoldItem<Mono8, f32>>) -> MonoF32>
{
    ClosureFold(
        |neighbors: &mut dyn Iterator<Item = FoldItem<Mono8, f32>>| {
            let mut sum = 0.0f32;
            for item in neighbors {
                sum += item.pixel.value() as f32 * item.weight;
            }
            MonoF32::new(sum)
        },
    )
}

// ─── Direct FoldOp struct (monomorphized, no dyn dispatch) ───────────────────

/// A weighted-sum fold implemented as a direct `FoldOp` struct.
///
/// This is the **monomorphized** path: `fold` is generic over `I`, so the
/// compiler generates specialized code for interior and boundary iterators
/// with full inlining and no vtable dispatch.
struct DirectSumFold;

impl FoldOp<Mono8, f32> for DirectSumFold {
    type Accumulator = f32;
    // ADR-0044 Phase E: pixel-role output type must be a real pixel.
    // `MonoF32` is `#[repr(transparent)]` over `f32`; codegen unchanged.
    type Output = MonoF32;

    #[inline(always)]
    fn init(&self) -> f32 {
        0.0
    }

    #[inline(always)]
    fn accumulate(&self, acc: &mut f32, item: FoldItem<Mono8, f32>) {
        #[cfg(target_feature = "fma")]
        {
            *acc = (item.pixel.value() as f32).mul_add(item.weight, *acc);
        }
        #[cfg(not(target_feature = "fma"))]
        {
            *acc += item.pixel.value() as f32 * item.weight;
        }
    }

    #[inline(always)]
    fn finalize(&mut self, acc: f32) -> MonoF32 {
        MonoF32::new(acc)
    }
}

// ─── Image sizes used across benchmarks ──────────────────────────────────────

const SIZES: &[(usize, usize)] = &[(256, 256), (1024, 1024), (2048, 2048)];

// ─── 1. fold_neighborhood raw ────────────────────────────────────────────────

fn bench_fold_neighborhood(c: &mut Criterion) {
    let mut group = c.benchmark_group("fold_neighborhood");

    for &(w, h) in SIZES {
        let img = make_u8_image(w, h);
        let kernel = Kernel3x3::new([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);

        group.bench_with_input(
            BenchmarkId::new("identity_3x3", format!("{w}x{h}")),
            &(&img, &kernel),
            |b, &(img, kernel)| {
                b.iter(|| {
                    black_box(fold_neighborhood(
                        img,
                        kernel.weights(),
                        kernel.anchor(),
                        &Clamp,
                        identity_fold(),
                    ))
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("sum_3x3", format!("{w}x{h}")),
            &(&img, &kernel),
            |b, &(img, kernel)| {
                b.iter(|| {
                    black_box(fold_neighborhood(
                        img,
                        kernel.weights(),
                        kernel.anchor(),
                        &Clamp,
                        sum_fold(),
                    ))
                });
            },
        );
    }

    group.finish();
}

// ─── 2. Convolution ──────────────────────────────────────────────────────────

fn bench_convolve(c: &mut Criterion) {
    let mut group = c.benchmark_group("convolve");

    // ── 3×3 Gaussian ─────────────────────────────────────────────────────
    let gauss3 = Kernel3x3::gaussian_3x3();

    for &(w, h) in SIZES {
        let img = make_u8_image(w, h);

        group.bench_with_input(
            BenchmarkId::new("gaussian_3x3_u8", format!("{w}x{h}")),
            &(&img, &gauss3),
            |b, &(img, kernel)| {
                b.iter(|| black_box(convolve::<_, _, _, _, MonoF32>(img, kernel, &Clamp)));
            },
        );
    }

    // ── 5×5 Gaussian ─────────────────────────────────────────────────────
    let gauss5 = Kernel5x5::gaussian_5x5();

    for &(w, h) in SIZES {
        let img = make_u8_image(w, h);

        group.bench_with_input(
            BenchmarkId::new("gaussian_5x5_u8", format!("{w}x{h}")),
            &(&img, &gauss5),
            |b, &(img, kernel)| {
                b.iter(|| black_box(convolve::<_, _, _, _, MonoF32>(img, kernel, &Clamp)));
            },
        );
    }

    // ── 7×7 box blur ─────────────────────────────────────────────────────
    let box7 = Kernel7x7::new([1.0 / 49.0; 49]);

    for &(w, h) in SIZES {
        let img = make_u8_image(w, h);

        group.bench_with_input(
            BenchmarkId::new("box_7x7_u8", format!("{w}x{h}")),
            &(&img, &box7),
            |b, &(img, kernel)| {
                b.iter(|| black_box(convolve::<_, _, _, _, MonoF32>(img, kernel, &Clamp)));
            },
        );
    }

    group.finish();
}

// ─── 3. Separable convolution ────────────────────────────────────────────────

fn bench_convolve_separable(c: &mut Criterion) {
    let mut group = c.benchmark_group("convolve_separable");

    let sep_gauss3: SeparableKernel<3, 3> = SeparableKernel::gaussian_3();
    let sep_gauss5: SeparableKernel<5, 5> = SeparableKernel::gaussian_5();

    for &(w, h) in SIZES {
        let img = make_u8_image(w, h);

        group.bench_with_input(
            BenchmarkId::new("gaussian_3_u8", format!("{w}x{h}")),
            &(&img, &sep_gauss3),
            |b, &(img, kernel)| {
                b.iter(|| {
                    black_box(convolve_separable::<_, _, _, MonoF32, MonoF32, 3, 3>(
                        img, kernel, &Clamp,
                    ))
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("gaussian_5_u8", format!("{w}x{h}")),
            &(&img, &sep_gauss5),
            |b, &(img, kernel)| {
                b.iter(|| {
                    black_box(convolve_separable::<_, _, _, MonoF32, MonoF32, 5, 5>(
                        img, kernel, &Clamp,
                    ))
                });
            },
        );
    }

    // ── Compare: 2D vs separable for same 3×3 Gaussian ──────────────────
    let full_gauss3 = Kernel3x3::gaussian_3x3();

    for &(w, h) in SIZES {
        let img = make_u8_image(w, h);

        group.bench_with_input(
            BenchmarkId::new("vs_full_2d_gaussian_3x3", format!("{w}x{h}")),
            &(&img, &full_gauss3),
            |b, &(img, kernel)| {
                b.iter(|| black_box(convolve::<_, _, _, _, MonoF32>(img, kernel, &Clamp)));
            },
        );
    }

    group.finish();
}

// ─── 4. Morphology ───────────────────────────────────────────────────────────

fn bench_morphology(c: &mut Criterion) {
    let mut group = c.benchmark_group("morphology");

    let se3 = Mask3x3::full_rect_3x3();
    let se5 = Mask5x5::full_rect_5x5();

    for &(w, h) in SIZES {
        let img = make_u8_image(w, h);

        // ── Erode ────────────────────────────────────────────────────────
        group.bench_with_input(
            BenchmarkId::new("erode_3x3_u8", format!("{w}x{h}")),
            &(&img, &se3),
            |b, &(img, se)| {
                b.iter(|| black_box(erode(img, se, &Clamp)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("erode_5x5_u8", format!("{w}x{h}")),
            &(&img, &se5),
            |b, &(img, se)| {
                b.iter(|| black_box(erode(img, se, &Clamp)));
            },
        );

        // ── Dilate ───────────────────────────────────────────────────────
        group.bench_with_input(
            BenchmarkId::new("dilate_3x3_u8", format!("{w}x{h}")),
            &(&img, &se3),
            |b, &(img, se)| {
                b.iter(|| black_box(dilate(img, se, &Clamp)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("dilate_5x5_u8", format!("{w}x{h}")),
            &(&img, &se5),
            |b, &(img, se)| {
                b.iter(|| black_box(dilate(img, se, &Clamp)));
            },
        );

        // ── Opening (erode + dilate, includes intermediate alloc) ────────
        group.bench_with_input(
            BenchmarkId::new("opening_3x3_u8", format!("{w}x{h}")),
            &(&img, &se3),
            |b, &(img, se)| {
                b.iter(|| black_box(opening(img, se, &Clamp)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("opening_5x5_u8", format!("{w}x{h}")),
            &(&img, &se5),
            |b, &(img, se)| {
                b.iter(|| black_box(opening(img, se, &Clamp)));
            },
        );
    }

    group.finish();
}

// ─── 5. Kernel size scaling ──────────────────────────────────────────────────

/// Measures how per-pixel cost scales with kernel size, isolating the
/// `dyn Iterator` dispatch overhead (which grows linearly with K²).
fn bench_kernel_size_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("kernel_size_scaling");

    let img = make_u8_image(512, 512);

    // 3×3
    let k3 = Kernel3x3::box_blur_3x3();
    group.bench_function("convolve_3x3_512x512", |b| {
        b.iter(|| black_box(convolve::<_, _, _, _, MonoF32>(&img, &k3, &Clamp)));
    });

    // 5×5
    let k5 = Kernel5x5::box_blur_5x5();
    group.bench_function("convolve_5x5_512x512", |b| {
        b.iter(|| black_box(convolve::<_, _, _, _, MonoF32>(&img, &k5, &Clamp)));
    });

    // 7×7
    let k7 = Kernel7x7::new([1.0 / 49.0; 49]);
    group.bench_function("convolve_7x7_512x512", |b| {
        b.iter(|| black_box(convolve::<_, _, _, _, MonoF32>(&img, &k7, &Clamp)));
    });

    // Morphology scaling
    let se3 = Mask3x3::full_rect_3x3();
    group.bench_function("erode_3x3_512x512", |b| {
        b.iter(|| black_box(erode(&img, &se3, &Clamp)));
    });

    let se5 = Mask5x5::full_rect_5x5();
    group.bench_function("erode_5x5_512x512", |b| {
        b.iter(|| black_box(erode(&img, &se5, &Clamp)));
    });

    // 7×7 structuring element
    let se7 = Neighborhood::<bool, 7, 7>::new([true; 49]);
    group.bench_function("erode_7x7_512x512", |b| {
        b.iter(|| black_box(erode(&img, &se7, &Clamp)));
    });

    group.finish();
}

// ─── 6. FoldOp vs ClosureFold (B15) ─────────────────────────────────────────

/// Head-to-head comparison: same weighted-sum operation done via a direct
/// `FoldOp` struct (fully monomorphized) vs `ClosureFold` (dyn dispatch).
///
/// The `FoldOp` path eliminates `dyn Iterator` vtable calls on every
/// `.next()` invocation, enabling inlining and auto-vectorization.
fn bench_foldop_vs_closure(c: &mut Criterion) {
    let mut group = c.benchmark_group("foldop_vs_closure");

    for &(w, h) in SIZES {
        let img = make_u8_image(w, h);
        let kernel = Kernel3x3::gaussian_3x3();

        // ── ClosureFold (dyn dispatch) ───────────────────────────────────
        group.bench_with_input(
            BenchmarkId::new("closure_sum_3x3", format!("{w}x{h}")),
            &(&img, &kernel),
            |b, &(img, kernel)| {
                b.iter(|| {
                    black_box(fold_neighborhood(
                        img,
                        kernel.weights(),
                        kernel.anchor(),
                        &Clamp,
                        sum_fold(),
                    ))
                });
            },
        );

        // ── Direct FoldOp (monomorphized) ────────────────────────────────
        group.bench_with_input(
            BenchmarkId::new("foldop_sum_3x3", format!("{w}x{h}")),
            &(&img, &kernel),
            |b, &(img, kernel)| {
                b.iter(|| {
                    black_box(fold_neighborhood(
                        img,
                        kernel.weights(),
                        kernel.anchor(),
                        &Clamp,
                        DirectSumFold,
                    ))
                });
            },
        );
    }

    // ── 5×5 kernel to show scaling effect ────────────────────────────────
    let img_512 = make_u8_image(512, 512);
    let kernel5 = Kernel5x5::gaussian_5x5();

    group.bench_function("closure_sum_5x5_512x512", |b| {
        b.iter(|| {
            black_box(fold_neighborhood(
                &img_512,
                kernel5.weights(),
                kernel5.anchor(),
                &Clamp,
                sum_fold(),
            ))
        });
    });

    group.bench_function("foldop_sum_5x5_512x512", |b| {
        b.iter(|| {
            black_box(fold_neighborhood(
                &img_512,
                kernel5.weights(),
                kernel5.anchor(),
                &Clamp,
                DirectSumFold,
            ))
        });
    });

    group.finish();
}

// ─── Criterion harness ──────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_fold_neighborhood,
    bench_convolve,
    bench_convolve_separable,
    bench_morphology,
    bench_kernel_size_scaling,
    bench_foldop_vs_closure,
);
criterion_main!(benches);

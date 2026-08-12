//! Demosaicing: what a phase-aware pass costs against the engines it did
//! not reuse.
//!
//! The question this bench exists to answer is the one Roadmap item 8 left
//! open. Its engine is pixel-outer / kernel-inner, because the kernel is
//! selected by the site's parity — so it cannot use the loop-inverted,
//! auto-vectorising interior path `fold_neighborhood` has. That is a
//! plausible-sounding cost with no number attached, and the honest way to
//! attach one is to measure the same 5×5 tap count both ways:
//! `MalvarHeCutler` reads two 5×5 kernels per site, `convolve` with a 5×5
//! kernel reads one, and both run over the same mosaic.
//!
//! `white_balance` is included as the memory-bound floor: one multiply and
//! one parity branch per sample, same traversal, no neighbourhood at all.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use fovea::border::Mirror;
use fovea::image::{Image, Neighborhood};
use fovea::pixel::bayer::{BayerRggb8, BayerRggb12};
use fovea::pixel::{MonoF32, Rgb8, Rgb12};
use fovea::transform::{
    BayerBilinear, BayerGains, MalvarHeCutler, convolve, demosaic, demosaic_into, white_balance,
};

/// A deterministic pseudo-mosaic with structure in all three channels: a
/// coloured checker plus a ripple, sampled through an RGGB tile. A flat
/// frame would flatter both algorithms equally and hide nothing, but it
/// would also not exercise the clamping path.
fn mosaic8(width: usize, height: usize) -> Image<BayerRggb8> {
    Image::generate(width, height, |x, y| {
        let checker = if (x / 17 + y / 13) % 2 == 0 { 40 } else { 200 };
        let ripple = ((x * 7 + y * 11) % 37) as i32 - 18;
        let tint = match (x % 2, y % 2) {
            (0, 0) => 0,   // R site
            (1, 1) => -30, // B site
            _ => 15,       // G sites
        };
        BayerRggb8::new((checker + ripple + tint).clamp(0, 255) as u8)
    })
}

fn mosaic12(width: usize, height: usize) -> Image<BayerRggb12> {
    Image::generate(width, height, |x, y| {
        let checker = if (x / 17 + y / 13) % 2 == 0 { 640 } else { 3200 };
        let ripple = ((x * 7 + y * 11) % 37) as i32 - 18;
        BayerRggb12::new((checker + ripple).clamp(0, 4095) as u16)
    })
}

fn criterion_benchmark(c: &mut Criterion) {
    let raw8 = mosaic8(512, 512);
    let raw12 = mosaic12(512, 512);

    let mut group = c.benchmark_group("demosaic");
    group.sample_size(20);

    // The two strategies, allocating output. 9 taps against ~50.
    group.bench_function("bilinear 512x512 BayerRggb8", |b| {
        b.iter(|| -> Image<Rgb8> { demosaic(black_box(&raw8), BayerBilinear) })
    });
    group.bench_function("malvar 512x512 BayerRggb8", |b| {
        b.iter(|| -> Image<Rgb8> { demosaic(black_box(&raw8), MalvarHeCutler) })
    });

    // Sub-word depth: the same arithmetic through `Mono<BITS>`' clamping
    // `FromLinear`, which is where a 12-bit frame differs from an 8-bit one.
    group.bench_function("malvar 512x512 BayerRggb12", |b| {
        b.iter(|| -> Image<Rgb12> { demosaic(black_box(&raw12), MalvarHeCutler) })
    });

    // The pre-allocated form, to price the output allocation on its own.
    let mut out8 = Image::<Rgb8>::zero(512, 512);
    group.bench_function("malvar into 512x512 BayerRggb8", |b| {
        b.iter(|| demosaic_into(black_box(&raw8), &mut out8, MalvarHeCutler))
    });

    // The like-for-like reference: one 5×5 kernel over the same mosaic
    // through the loop-inverted `fold_neighborhood` interior path. Malvar
    // evaluates two such kernels per site, so a proportionate cost is
    // ~2× this; anything far above is the vectorisation gap.
    let kernel5 = Neighborhood::<f32, 5, 5>::box_blur_5x5();
    group.bench_function("convolve 5x5 reference 512x512 BayerRggb8", |b| {
        b.iter(|| convolve::<_, _, _, _, MonoF32>(black_box(&raw8), &kernel5, &Mirror))
    });

    // The memory-bound floor: same traversal, no neighbourhood.
    let gains = BayerGains::new(1.9, 1.0, 1.6);
    group.bench_function("white balance 512x512 BayerRggb8", |b| {
        b.iter(|| white_balance(black_box(&raw8), gains))
    });

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

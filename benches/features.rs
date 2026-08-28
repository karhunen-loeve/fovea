//! Corner detection: the two families, measured against each other.
//!
//! The question this bench exists to answer: whether the classical
//! four-point early rejection is worth building into the segment test, and
//! the wider one behind it: what a segment test costs relative to a
//! structure tensor on the same frame.
//!
//! The image is a synthetic texture rather than a clean square, because the
//! interesting cost of FAST is how quickly it *rejects* ordinary pixels, and
//! a mostly-flat image would flatter it.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use fovea::border::Skip;
use fovea::features::detect::{
    CornerParams, FastParams, NmsRadius, SegmentTest, ShiTomasi, corner_response_map,
    detect_corners, fast, fast_score_map,
};
use fovea::image::Image;
use fovea::pixel::{Mono8, MonoF32};
use fovea::{harris, sigma};

/// A deterministic pseudo-texture: enough structure that the detectors have
/// real work to do, and no dependence on a random-number generator.
fn texture(width: usize, height: usize) -> Image<Mono8> {
    Image::generate(width, height, |x, y| {
        let checker = if (x / 17 + y / 13) % 2 == 0 { 40 } else { 200 };
        let ripple = ((x * 7 + y * 11) % 37) as i32 - 18;
        Mono8::new((checker + ripple).clamp(0, 255) as u8)
    })
}

fn criterion_benchmark(c: &mut Criterion) {
    let image = texture(512, 512);

    let mut group = c.benchmark_group("corners");
    group.sample_size(20);

    // The score map alone, at each arc length that changes the early-exit
    // arithmetic: 9..=11 need two of the four cardinal samples, 12..=15 need
    // three, 16 needs all four.
    for n in [9usize, 12, 16] {
        let test = SegmentTest::new(20.0, n).unwrap();
        group.bench_function(format!("fast-{n} score map 512x512 Mono8"), |b| {
            b.iter(|| fast_score_map(black_box(&image), test, &Skip))
        });
    }

    // The tensor family's equivalent stage, so the map-to-map comparison is
    // like for like: both produce one `Image<MonoF32>` and neither has
    // selected a peak yet.
    group.bench_function("harris response map 512x512 Mono8", |b| {
        b.iter(|| {
            corner_response_map::<_, _, _, MonoF32>(black_box(&image), harris!(0.04), sigma!(1.4))
        })
    });
    group.bench_function("shi-tomasi response map 512x512 Mono8", |b| {
        b.iter(|| {
            corner_response_map::<_, _, _, MonoF32>(black_box(&image), ShiTomasi, sigma!(1.4))
        })
    });

    // The whole detector, so the peak stage is included in the comparison.
    let fast_params = FastParams::new(
        SegmentTest::new(20.0, 9).unwrap(),
        NmsRadius::new(3).unwrap(),
    );
    group.bench_function("fast-9 detect 512x512 Mono8", |b| {
        b.iter(|| fast(black_box(&image), fast_params, &Skip))
    });

    let corner_params = CornerParams::new(sigma!(1.4), 1e7, NmsRadius::new(3).unwrap()).unwrap();
    group.bench_function("harris detect 512x512 Mono8", |b| {
        b.iter(|| detect_corners(black_box(&image), harris!(0.04), corner_params))
    });
    group.bench_function("shi-tomasi detect 512x512 Mono8", |b| {
        b.iter(|| detect_corners(black_box(&image), ShiTomasi, corner_params))
    });

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

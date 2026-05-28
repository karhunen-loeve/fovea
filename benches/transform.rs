use criterion::{Criterion, criterion_group, criterion_main};
use irys_cv::{
    Size,
    image::Image,
    pixel::{FromLinear, LinearPixel, LinearSpace, Mono8, Rgb8, ZeroablePixel},
    transform::{Bilinear, NearestNeighbor, resize},
};
use std::hint::black_box;

fn resize_nearest<T: ZeroablePixel>(img: &Image<T>, size: Size) -> Image<T> {
    resize(img, size, NearestNeighbor)
}

fn resize_bilinear<T>(img: &Image<T>, size: Size) -> Image<T>
where
    T: ZeroablePixel + LinearPixel + LinearSpace,
    T::Accumulator: LinearPixel<Accumulator = T::Accumulator> + LinearSpace,
    T: FromLinear<T::Accumulator>,
{
    resize(img, size, Bilinear)
}

fn criterion_benchmark(c: &mut Criterion) {
    // ADR-0045 Phase S4: `u8` no longer implements `LinearPixel`, so the
    // bilinear resize bench inputs migrated from `Image<u8>` to
    // `Image<Mono8>`. `Mono8` is `#[repr(transparent)]` over
    // `Saturating<u8>`, so the bench layout and numerical path are
    // unchanged. The benchmark labels keep the "u8" tag for continuity
    // with historical measurements.
    let img: Image<Mono8> = Image::generate(300, 400, |x, y| Mono8::new((x + y) as u8));
    let img_big: Image<Mono8> = Image::generate(900, 1200, |x, y| Mono8::new((x + y) as u8));
    let img_rgb: Image<Rgb8> = Image::generate(300, 400, |x, y| {
        let v = (x + y) as u8;
        Rgb8::new(v, v, v)
    });

    let mut group = c.benchmark_group("resize");
    group.measurement_time(std::time::Duration::from_secs(100)); // 10 seconds per benchmark
    for size in [300, 600, 1200, 2400].iter() {
        group.bench_function(
            format!("resize u8 nearest 300x400 -> {}x{}", size, size),
            |b| b.iter(|| resize_nearest(black_box(&img), Size::new(*size, *size))),
        );
        group.bench_function(
            format!("resize u8 nearest 900x1200 -> {}x{}", size, size),
            |b| b.iter(|| resize_nearest(black_box(&img_big), Size::new(*size, *size))),
        );
        group.bench_function(
            format!("resize rgb nearest 300x400 -> {}x{}", size, size),
            |b| b.iter(|| resize_nearest(black_box(&img_rgb), Size::new(*size, *size))),
        );
    }

    for size in [300, 600, 1200, 2400].iter() {
        group.bench_function(
            format!("resize u8 bilinear 300x400 -> {}x{}", size, size),
            |b| b.iter(|| resize_bilinear(black_box(&img), Size::new(*size, *size))),
        );
        group.bench_function(
            format!("resize u8 bilinear 900x1200 -> {}x{}", size, size),
            |b| b.iter(|| resize_bilinear(black_box(&img_big), Size::new(*size, *size))),
        );
        group.bench_function(
            format!("resize rgb bilinear 300x400 -> {}x{}", size, size),
            |b| b.iter(|| resize_bilinear(black_box(&img_rgb), Size::new(*size, *size))),
        );
    }
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

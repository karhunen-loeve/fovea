//! Benchmarks for `irys_cv::transform::geometry`.
//!
//! Goal: measure each axis-aligned transform on a typical
//! megapixel-scale image and confirm the cache-locality story claimed
//! in the module docs:
//!
//! - `flip_v` (full-row `memcpy`) and `flip_h`/`rotate_180` (row-major
//!   reverse-copy) should be roughly **memcpy-bound** — i.e. limited
//!   by RAM/L3 bandwidth, not by per-pixel arithmetic.
//! - `rotate_90`, `rotate_270`, `transpose` cannot iterate sequentially
//!   on both sides; they are expected to be **noticeably slower** at
//!   sizes that exceed L2 (= a few megabytes on modern x86).
//!
//! Two **alternative** implementations are benched alongside the
//! production ones to quantify what we leave on the table by keeping
//! the naïve loops:
//!
//! - `flip_h_zip`: replaces the `dst[w-1-x] = src[x]` index pair with
//!   a `zip(src.iter(), dst.iter_mut().rev())` loop. Removes any
//!   residual bounds-check that LLVM might fail to elide.
//! - `transpose_row_cached`: iterates the **output** row-major instead
//!   of the input. The destination row pointer is fetched **once** per
//!   output row (avoids `pixel_at_mut` per pixel) while reads become
//!   strided.
//!
//! Run with:
//!
//! ```text
//! cargo bench -p irys-cv --bench geometry
//! ```

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use irys_cv::image::{Image, RasterImage, RasterImageMut};
use irys_cv::pixel::Mono8;
use irys_cv::transform::{
    flip_h_into, flip_v_into, rotate_90_into, rotate_180_into, rotate_270_into, transpose_into,
};

// ── Alternative implementations under test ─────────────────────────────────

/// `flip_h_into` rewritten using iterator zip. Same result, no per-element
/// index expression.
fn flip_h_zip<I, O>(img: &I, out: &mut O)
where
    I: RasterImage,
    O: RasterImageMut<Pixel = I::Pixel>,
{
    assert_eq!(img.size(), out.size());
    for y in 0..img.height() {
        let src = img.row(y);
        let dst = out.row_mut(y);
        for (d, s) in dst.iter_mut().rev().zip(src.iter()) {
            *d = *s;
        }
    }
}

/// Transpose driven by the **output** row pointer.
///
/// The production `transpose_into` iterates the input row-major and
/// scatters writes via `pixel_at_mut`. This variant flips the roles:
/// the destination row slice is obtained **once per output row**, then
/// filled from a strided gather across the input.
fn transpose_row_cached<I, O>(img: &I, out: &mut O)
where
    I: RasterImage,
    O: RasterImageMut<Pixel = I::Pixel>,
{
    assert_eq!(out.width(), img.height());
    assert_eq!(out.height(), img.width());
    let h_out = out.height();
    let w_out = out.width();
    for y_out in 0..h_out {
        let dst = out.row_mut(y_out);
        // out[x_out, y_out] = in[y_out, x_out]
        for x_out in 0..w_out {
            dst[x_out] = img.pixel_at(y_out, x_out);
        }
    }
}

// ── Benchmark driver ───────────────────────────────────────────────────────

fn make_image(w: usize, h: usize) -> Image<Mono8> {
    Image::generate(w, h, |x, y| Mono8::new(((x * 31) ^ (y * 17)) as u8))
}

fn bench_geometry(c: &mut Criterion) {
    // Three sizes covering: comfortably in L2 (256² = 64 KiB),
    // straddles L2/L3 (1024² = 1 MiB), and forces DRAM traffic
    // (4096² = 16 MiB).
    for &n in &[256usize, 1024, 4096] {
        let mut group = c.benchmark_group(format!("geometry/{n}x{n}/Mono8"));
        group.throughput(Throughput::Bytes((n * n) as u64));

        let img = make_image(n, n);
        let mut buf_same = Image::<Mono8>::zero(n, n);
        let mut buf_swap = Image::<Mono8>::zero(n, n);

        group.bench_function("flip_h_into", |b| {
            b.iter(|| flip_h_into(black_box(&img), black_box(&mut buf_same)))
        });
        group.bench_function("flip_h_zip (alt)", |b| {
            b.iter(|| flip_h_zip(black_box(&img), black_box(&mut buf_same)))
        });
        group.bench_function("flip_v_into", |b| {
            b.iter(|| flip_v_into(black_box(&img), black_box(&mut buf_same)))
        });
        group.bench_function("rotate_180_into", |b| {
            b.iter(|| rotate_180_into(black_box(&img), black_box(&mut buf_same)))
        });
        group.bench_function("rotate_90_into", |b| {
            b.iter(|| rotate_90_into(black_box(&img), black_box(&mut buf_swap)))
        });
        group.bench_function("rotate_270_into", |b| {
            b.iter(|| rotate_270_into(black_box(&img), black_box(&mut buf_swap)))
        });
        group.bench_function("transpose_into", |b| {
            b.iter(|| transpose_into(black_box(&img), black_box(&mut buf_swap)))
        });
        group.bench_function("transpose_row_cached (alt)", |b| {
            b.iter(|| transpose_row_cached(black_box(&img), black_box(&mut buf_swap)))
        });

        group.finish();
    }
}

criterion_group!(benches, bench_geometry);
criterion_main!(benches);

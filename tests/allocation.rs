//! Allocation invariants for the reusable separable working set.
//!
//! The claim "a blur in a hot loop allocates nothing after warm-up" is only
//! worth making if it is enforced, so this binary installs a counting
//! global allocator and asserts the count directly.
//!
//! The counter is **thread-local**: only allocations made on the thread
//! running a test are attributed to it, so the test harness running other
//! tests in parallel cannot pollute a measurement. The `Cell`s are
//! const-initialised and have no destructors, so reading them never
//! allocates and cannot recurse into the allocator.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use fovea::border::{Clamp, Skip};
use fovea::image::{Image, SeparableKernel};
use fovea::pixel::{Mono8, MonoF32};
use fovea::sigma;
use fovea::transform::{SeparableScratch, gaussian_blur_into};

thread_local! {
    /// Allocations (including reallocations) made by this thread.
    static ALLOCS: Cell<u64> = const { Cell::new(0) };
}

struct CountingAllocator;

fn record() {
    // `try_with` so an allocation during thread teardown (after the TLS
    // slot is gone) is simply not counted instead of panicking.
    let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
}

// SAFETY: every method forwards to `System` unchanged; the only addition is
// a counter increment that cannot allocate.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record();
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record();
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record();
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Run `f` and report how many allocations it made on this thread.
fn allocations_of<T>(f: impl FnOnce() -> T) -> (T, u64) {
    let before = ALLOCS.with(Cell::get);
    let value = f();
    let after = ALLOCS.with(Cell::get);
    (value, after - before)
}

#[test]
fn counter_observes_a_known_allocation() {
    // Guards the guard: if this ever reports zero, every assertion below
    // would pass vacuously.
    let (v, allocations) = allocations_of(|| vec![0u8; 4096]);
    assert_eq!(v.len(), 4096);
    assert!(
        allocations >= 1,
        "the counting allocator saw {allocations} allocations for one Vec"
    );
}

#[test]
fn one_shot_blur_allocates_its_working_set() {
    // The baseline the scratch form improves on: even writing into a
    // caller-owned output, the one-shot path allocates the inter-pass
    // intermediate plus the engine's two working buffers per pass.
    let src = Image::generate(64, 64, |x, y| Mono8::new(((x * 3 + y) % 256) as u8));
    let mut out = Image::<Mono8>::zero(64, 64);
    let sigma = sigma!(1.5);

    // Warm up first, so the count reflects steady state rather than any
    // one-time initialisation elsewhere in the call graph.
    gaussian_blur_into(&src, sigma, &Clamp, &mut out);

    let (_, allocations) = allocations_of(|| {
        gaussian_blur_into(&src, sigma, &Clamp, &mut out);
    });

    // Intermediate image, and per pass an accumulator row and a
    // kernel-position list: at least three.
    assert!(
        allocations >= 3,
        "expected the one-shot path to allocate its working set, saw {allocations}"
    );
}

#[test]
fn second_same_size_blur_allocates_nothing() {
    let src = Image::generate(64, 64, |x, y| Mono8::new(((x + y * 7) % 256) as u8));
    let mut out = Image::<Mono8>::zero(64, 64);
    let mut scratch = SeparableScratch::<MonoF32>::new();
    let sigma = sigma!(1.5);

    // First call sizes the buffers.
    scratch.gaussian_blur_into(&src, sigma, &Clamp, &mut out);

    let (_, allocations) = allocations_of(|| {
        scratch.gaussian_blur_into(&src, sigma, &Clamp, &mut out);
    });

    assert_eq!(
        allocations, 0,
        "a second same-size blur through the scratch must not allocate",
    );
}

#[test]
fn shrinking_pyramid_of_blurs_allocates_only_on_the_first_level() {
    // The motivating case: a pyramid blurs `log₂(N)` times per image with a
    // strictly shrinking size. The buffers are never shrunk, so every level
    // after the first fits in what the first one sized.
    let sizes = [64usize, 32, 16, 8];
    let sources: Vec<Image<Mono8>> = sizes
        .iter()
        .map(|&n| Image::generate(n, n, |x, y| Mono8::new(((x + y) % 256) as u8)))
        .collect();
    let mut outputs: Vec<Image<Mono8>> = sizes.iter().map(|&n| Image::zero(n, n)).collect();
    let mut scratch = SeparableScratch::<MonoF32>::new();
    let sigma = sigma!(1.0);

    // Level 0 warms the buffers up.
    scratch.gaussian_blur_into(&sources[0], sigma, &Clamp, &mut outputs[0]);

    for level in 1..sizes.len() {
        let (_, allocations) = allocations_of(|| {
            scratch.gaussian_blur_into(&sources[level], sigma, &Clamp, &mut outputs[level]);
        });
        assert_eq!(
            allocations, 0,
            "level {level} ({0}×{0}) allocated after warm-up",
            sizes[level],
        );
    }
}

#[test]
fn separable_convolution_through_scratch_allocates_nothing_after_warmup() {
    // Same invariant for the kernel-based entry point, and with a border
    // policy whose output region is smaller than the input.
    let src = Image::generate(48, 40, |x, y| MonoF32::new((x + y) as f32));
    let kernel = SeparableKernel::gaussian_5();
    let mut scratch = SeparableScratch::<MonoF32>::new();
    let mut out = Image::<MonoF32>::zero(44, 36);

    scratch.convolve_separable_into(&src, &kernel, &Skip, &mut out);

    let (_, allocations) = allocations_of(|| {
        scratch.convolve_separable_into(&src, &kernel, &Skip, &mut out);
    });

    assert_eq!(
        allocations, 0,
        "a second identical separable convolution must not allocate",
    );
}

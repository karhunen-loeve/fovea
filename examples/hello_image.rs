//! # Hello Image
//!
//! Your first irys-cv program: create an image, fill pixels, read them back.
//!
//! Run with: `cargo run --example hello_image`

fn main() {
    // Import the types we need:
    // - Image: owned, heap-allocated image (like Vec<T> for 2D pixel data)
    // - ImageView/ImageViewMut: trait-based read/write access (like &[T] / &mut [T])
    // - ContiguousImage: trait that provides as_slice() for row-major pixel access
    // - Mono8: single-channel 8-bit grayscale pixel
    use irys_cv::image::{ContiguousImage, Image, ImageView, ImageViewMut};
    use irys_cv::pixel::Mono8;

    // ── 1. Create an image filled with zeros ─────────────────────────────────

    // Image::zero requires the pixel type to be ZeroablePixel (Mono8 qualifies).
    // The image owns its pixel data on the heap, just like Vec<T>.
    let mut img = Image::<Mono8>::zero(4, 3);
    println!("Created {}×{} image", img.width(), img.height());

    // ── 2. Set individual pixels ─────────────────────────────────────────────

    // pixel_at_mut(x, y) returns &mut Mono8.
    // Coordinates are (column, row) — x is horizontal, y is vertical.
    // Panics on out-of-bounds, just like slice indexing with [].
    *img.pixel_at_mut(0, 0) = Mono8::new(100);
    *img.pixel_at_mut(1, 0) = Mono8::new(200);
    *img.pixel_at_mut(2, 0) = Mono8::new(50);
    *img.pixel_at_mut(3, 0) = Mono8::new(255);

    // ── 3. Read pixels back ──────────────────────────────────────────────────

    // pixel_at(x, y) returns &Mono8 — the immutable counterpart.
    let p = img.pixel_at(0, 0);
    // .value() extracts the raw u8 intensity from the Mono8 wrapper.
    println!("Pixel at (0,0): {:?} — raw value = {}", p, p.value());

    // get(x, y) is the checked version — returns Option<&Mono8> instead of panicking.
    let inside = img.get(3, 2);
    let outside = img.get(4, 0); // x=4 is out of bounds for width=4
    println!("get(3,2) = {:?}", inside); // Some(Mono8(0))
    println!("get(4,0) = {:?}", outside); // None — safely handled

    // ── 4. Iterate over all pixels ───────────────────────────────────────────

    // as_slice() comes from the ContiguousImage trait.
    // It gives &[Mono8] — the full pixel buffer in row-major order.
    let sum: u32 = img.as_slice().iter().map(|px| px.value() as u32).sum();
    println!("Sum of all pixel values: {sum}");

    // Count how many pixels are brighter than a threshold.
    let bright_count = img.as_slice().iter().filter(|px| px.value() > 100).count();
    println!("Pixels with value > 100: {bright_count}");

    // ── 5. Create an image from a Vec ────────────────────────────────────────

    // from_vec checks that data.len() == width * height at runtime.
    // Returns Result — Err if the length doesn't match.
    let pixels = vec![Mono8::new(10); 8]; // exactly 4×2 = 8 pixels
    let img2 = Image::from_vec(4, 2, pixels).expect("length must match width × height");
    println!(
        "img2: {}×{}, first pixel = {:?}",
        img2.width(),
        img2.height(),
        img2.pixel_at(0, 0),
    );

    // Demonstrate the error case: 7 pixels can't fill a 4×2 image.
    let bad = Image::from_vec(4, 2, vec![Mono8::new(0); 7]);
    println!("from_vec with wrong length: {:?}", bad.err());

    // ── 6. Generate an image with a closure ──────────────────────────────────

    // Image::generate calls f(x, y) for every pixel position.
    // Here we create a horizontal gradient: intensity increases with x.
    let gradient = Image::generate(8, 4, |x, _y| {
        // Scale x from [0..7] into [0..255] so the gradient spans full range.
        let intensity = (x * 255 / 7) as u8;
        Mono8::new(intensity)
    });
    // Print the first row to show the gradient.
    print!("Gradient row 0:");
    for x in 0..gradient.width() {
        print!(" {}", gradient.pixel_at(x, 0).value());
    }
    println!();

    // ── 7. Fill an image with a constant value ───────────────────────────────

    // Image::fill is like zero(), but with an arbitrary pixel value.
    let white = Image::fill(3, 3, Mono8::new(255));
    // Every pixel should be 255.
    let all_white = white.as_slice().iter().all(|px| px.value() == 255);
    println!("3×3 white image — all pixels 255? {all_white}");

    // ── 8. Pixel value manipulation ──────────────────────────────────────────

    // Mono8 wraps a Saturating<u8>, but arithmetic ops on the pixel type
    // itself are limited (only Mul<f32> is provided for scaling).
    // For general math, extract with .value(), compute, then wrap back.
    let original = Mono8::new(180);
    let brightened = Mono8::new(original.value().saturating_add(50)); // 180 + 50 = 230
    let dimmed = Mono8::new(original.value().saturating_sub(200)); // 180 - 200 = 0 (saturates)
    println!(
        "original = {}, bright = {}, dim = {}",
        original.value(),
        brightened.value(),
        dimmed.value()
    );

    // Mono8 does support Mul<f32> for scaling:
    let scaled = original * 0.5; // 180 * 0.5 = 90
    println!("180 * 0.5 = {}", scaled.value());

    println!("\nDone! You've created and manipulated images with irys-cv.");
}

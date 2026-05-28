//! # Custom Filter
//!
//! Implement a custom neighborhood filter using the `FoldOp` trait.
//!
//! The `FoldOp` trait is the extensibility point for neighborhood operations in
//! `irys-cv`.  The library provides built-in operations (convolution, erosion,
//! dilation, median), but you can define your own by implementing `FoldOp`.
//!
//! This example builds a simple **weighted-average** filter from scratch — the
//! same thing a box-blur convolution does internally — to show how the pieces
//! fit together.
//!
//! Run with: `cargo run --example custom_filter`

fn main() {
    // ── Imports ──────────────────────────────────────────────────────────────
    //
    // `Image` is the owned image container; `ImageView` / `ImageViewMut` are
    // the read / write trait interfaces that all image types implement.
    // `Neighborhood` bundles a weight grid with an anchor point.
    use irys_cv::image::{Image, ImageView, ImageViewMut, Neighborhood};
    // `Mono8` is an 8-bit grayscale pixel (single channel, u8 storage).
    use irys_cv::pixel::Mono8;
    // `Clamp` replicates the nearest edge pixel for out-of-bounds accesses —
    // the same as OpenCV's `BORDER_REPLICATE`.  (The library calls it `Clamp`
    // because it clamps coordinates to the valid range.)
    use irys_cv::border::Clamp;
    // `FoldOp` is the trait we implement; `FoldItem` is the per-neighbor
    // (pixel, weight) pair; `fold_neighborhood` drives the iteration.
    use irys_cv::transform::{FoldItem, FoldOp, fold_neighborhood};

    // =====================================================================
    // 1. Define a custom FoldOp
    // =====================================================================
    //
    // `FoldOp<P, W>` has three required methods:
    //
    //   fn init(&self) -> Self::Accumulator;
    //   fn accumulate(&self, acc: &mut Self::Accumulator, item: FoldItem<P, W>);
    //   fn finalize(&mut self, acc: Self::Accumulator) -> Self::Output;
    //
    // The engine calls `init` once per output pixel, `accumulate` for every
    // kernel position, then `finalize` to produce the result.  This split
    // lets the engine loop-invert the interior path for SIMD-friendly
    // access patterns.

    /// Computes a weighted average over the neighborhood.
    ///
    /// For each neighbor we accumulate `pixel_value * weight`, then divide
    /// by the sum of weights.  This is mathematically equivalent to
    /// normalized convolution, but we do it longhand to demonstrate `FoldOp`.
    struct WeightedAverage;

    impl FoldOp<Mono8, f32> for WeightedAverage {
        // The accumulator tracks both the weighted sum and total weight.
        type Accumulator = (f32, f32);
        // The output pixel type — we produce another `Mono8`.
        type Output = Mono8;

        fn init(&self) -> Self::Accumulator {
            (0.0_f32, 0.0_f32) // (weighted_sum, weight_sum)
        }

        #[inline(always)]
        fn accumulate(&self, acc: &mut Self::Accumulator, item: FoldItem<Mono8, f32>) {
            // `item.pixel` is the source `Mono8`; `.value()` extracts
            // the raw `u8`.  `item.weight` is the `f32` kernel weight.
            acc.0 += item.pixel.value() as f32 * item.weight;
            acc.1 += item.weight;
        }

        fn finalize(&mut self, acc: Self::Accumulator) -> Self::Output {
            let (sum, weight_sum) = acc;

            // Normalize by total weight so the filter is a true average.
            // Guard against a zero-weight kernel (degenerate but possible).
            let avg = if weight_sum > 0.0 {
                sum / weight_sum
            } else {
                0.0
            };

            // Clamp to [0, 255] and round to the nearest integer.
            Mono8::new(avg.round().clamp(0.0, 255.0) as u8)
        }
    }

    // =====================================================================
    // 2. Build a test image with a single bright spot
    // =====================================================================

    // Start with an 8×8 all-black image.
    let mut img = Image::<Mono8>::zero(8, 8);

    // Place a single white pixel at column 4, row 4.
    *img.pixel_at_mut(4, 4) = Mono8::new(255);

    println!("Input: 8×8 black image with a bright spot at (4,4)");
    println!("  pixel(4,4) = {}", img.pixel_at(4, 4).value()); // 255
    println!("  pixel(3,4) = {}", img.pixel_at(3, 4).value()); // 0
    println!("  pixel(0,0) = {}", img.pixel_at(0, 0).value()); // 0

    // =====================================================================
    // 3. Define a 3×3 uniform kernel (box filter)
    // =====================================================================
    //
    // `Neighborhood::<W, KW, KH>::new(data)` takes a flat array in
    // row-major order.  The anchor defaults to the center `(KW/2, KH/2)`.
    //
    // We use equal weights of 1.0 everywhere.  Our `WeightedAverage` op
    // will divide by the weight sum (9.0), so the effective operation is
    // a 3×3 box blur.

    let kernel = Neighborhood::<f32, 3, 3>::new([
        1.0, 1.0, 1.0, // top row
        1.0, 1.0, 1.0, // middle row
        1.0, 1.0, 1.0, // bottom row
    ]);

    // =====================================================================
    // 4. Apply the custom filter
    // =====================================================================
    //
    // `fold_neighborhood` takes:
    //   - source image (anything implementing `ImageView`)
    //   - weight grid  (the kernel's weight sub-image)
    //   - anchor point (center of the kernel window)
    //   - border policy (how to handle out-of-bounds accesses)
    //   - fold operation (our custom `WeightedAverage`)
    //
    // It returns a new `Image<Out>` whose size is determined by the border
    // policy.  `Clamp` produces an output the same size as the input.

    let result = fold_neighborhood(
        &img,             // source image
        kernel.weights(), // &ImageArray<f32, 3, 3> — the weight grid
        kernel.anchor(),  // (1, 1) — center of a 3×3 kernel
        &Clamp,           // border policy: replicate edge pixels
        WeightedAverage,  // our custom fold operation
    );

    // =====================================================================
    // 5. Inspect the results
    // =====================================================================

    let center = result.pixel_at(4, 4).value();
    let neighbor = result.pixel_at(3, 4).value();
    let far = result.pixel_at(0, 0).value();

    println!("\nOutput after 3×3 weighted-average filter:");
    println!("  pixel(4,4) = {} (was 255)", center);
    println!("  pixel(3,4) = {} (was 0)", neighbor);
    println!("  pixel(0,0) = {} (was 0)", far);

    // The bright spot (255) is surrounded by 8 black neighbors.
    // A 3×3 uniform average gives 255 / 9 ≈ 28 at the center.
    assert!(
        center > 0 && center < 255,
        "center should be dimmed by averaging with its dark neighbors"
    );

    // The pixel next to the bright spot should pick up some brightness,
    // because the 3×3 window centered on (3,4) overlaps the bright spot.
    assert!(
        neighbor > 0,
        "immediate neighbor should be non-zero after averaging"
    );

    // A pixel far from the bright spot should still be zero — the kernel
    // only reaches 1 pixel in each direction.
    assert_eq!(
        far, 0,
        "pixels far from the bright spot should remain black"
    );

    // The expected value at center is round(255 / 9) = round(28.333) = 28.
    assert_eq!(
        center, 28,
        "255 averaged over 9 equal-weight cells should round to 28"
    );

    // The expected value at the neighbor is the same: its 3×3 window
    // contains exactly one pixel at 255 and eight at 0.
    assert_eq!(
        neighbor, 28,
        "neighbor's 3×3 window also contains exactly one bright pixel"
    );

    println!("\nCustom filter verified — the bright spot was spread by the 3×3 weighted average!");
    println!("Implementing FoldOp lets you plug arbitrary per-pixel logic");
    println!("into the same optimized interior/boundary engine that the");
    println!("built-in convolution and morphology operations use.");
}

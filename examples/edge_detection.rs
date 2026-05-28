//! # Edge Detection
//!
//! Apply Sobel edge detection to a synthetic image and inspect the results.
//!
//! This example demonstrates:
//! - Building a synthetic test image with a known feature (a vertical step edge)
//! - Applying a neighbourhood transform (`sobel_x`) that produces `f32` output
//! - Using a border policy (`Clamp`) to handle pixels near the image boundary
//! - Reading back floating-point results and reasoning about their meaning
//!
//! Run with: `cargo run --example edge_detection`

fn main() {
    // ── Imports ──────────────────────────────────────────────────────────
    //
    // ImageView / ImageViewMut are *traits* — they must be in scope so we
    // can call `.width()`, `.height()`, `.pixel_at()`, and `.pixel_at_mut()`
    // on any image type.
    use irys_cv::image::{Image, ImageView, ImageViewMut};

    // Mono8 is a single-channel, 8-bit grayscale pixel backed by
    // `Saturating<u8>`, which means arithmetic saturates instead of wrapping.
    use irys_cv::pixel::{Mono8, MonoF32};

    // Clamp is a border policy: when a filter kernel overlaps the image
    // boundary, out-of-bounds reads return the nearest edge pixel.
    // This is the most common choice for gradient filters because it
    // avoids introducing artificial edges at the border.
    use irys_cv::border::Clamp;

    // sobel_x computes the horizontal image derivative (dI/dx).
    // Vertical edges — where intensity changes left-to-right — produce a
    // strong response.  The output is `Image<MonoF32>` — the input
    // pixel's linear accumulator — so that negative gradients
    // (bright→dark) are preserved. Under ADR-0045 / ADR-0044, gradient
    // images are pixel-role spatial grids of signed intensities; the
    // `MonoF32` wrapper names that semantic explicitly.
    use irys_cv::transform::sobel_x;

    // ── Build a synthetic test image ────────────────────────────────────
    //
    // 16×16 image with a vertical step edge in the middle:
    //   columns 0..7  → dark  (intensity 0)
    //   columns 8..15 → bright (intensity 200)
    //
    // This gives us a clean, predictable feature to detect.
    let width: usize = 16;
    let height: usize = 16;
    let edge_col: usize = 8; // the column where the brightness step occurs

    // Start with an all-black image.
    let mut img = Image::<Mono8>::zero(width, height);

    // Fill the right half with a bright value.
    for y in 0..height {
        for x in edge_col..width {
            // pixel_at_mut returns `&mut Mono8` — we overwrite in place.
            *img.pixel_at_mut(x, y) = Mono8::new(200);
        }
    }

    println!(
        "Input image: {}×{} Mono8 with a vertical step edge at x={}",
        img.width(),
        img.height(),
        edge_col,
    );

    // Quick sanity check: verify that the step edge is where we expect it.
    assert_eq!(img.pixel_at(edge_col - 1, 0).value(), 0);
    assert_eq!(img.pixel_at(edge_col, 0).value(), 200);

    // ── Apply the Sobel-X filter ────────────────────────────────────────
    //
    // sobel_x applies a 3×3 convolution kernel that approximates dI/dx:
    //
    //   [-1  0  +1]
    //   [-2  0  +2]
    //   [-1  0  +1]
    //
    // The output pixel type is `MonoF32` — a transparent wrapper over
    // `f32` — because the derivative can be negative (bright-to-dark
    // transition) and fractional, neither of which `u8` can represent.
    // `MonoF32` carries the pixel-role semantic (gradient magnitudes are
    // a spatial grid, not a scalar weight).
    let edges: Image<MonoF32> = sobel_x(&img, &Clamp);

    println!(
        "Output image: {}×{} MonoF32 (one value per pixel)",
        edges.width(),
        edges.height(),
    );

    // ── Inspect the results ─────────────────────────────────────────────
    //
    // We sample three representative locations along the middle row (y=8):
    //   • x=2  — well inside the dark (flat) region → near-zero response
    //   • x=8  — right at the step edge → strong positive response
    //   • x=14 — well inside the bright (flat) region → near-zero response

    let sample_y: usize = height / 2; // middle row

    // pixel_at returns `MonoF32` by value; `.0` extracts the underlying
    // `f32` exactly at the display / comparison boundary.
    let response_flat_left = edges.pixel_at(2, sample_y).0;
    let response_at_edge = edges.pixel_at(edge_col, sample_y).0;
    let response_flat_right = edges.pixel_at(14, sample_y).0;

    println!("\nSobel-X responses along row y={sample_y}:");
    println!("  x=2  (flat dark region):   {response_flat_left:>8.1}");
    println!("  x={edge_col}  (step edge):          {response_at_edge:>8.1}");
    println!("  x=14 (flat bright region): {response_flat_right:>8.1}");

    // ── Verify that the filter actually found the edge ──────────────────
    //
    // In flat regions the derivative should be (close to) zero.
    // At the step edge the derivative should be large and positive
    // (because intensity increases from left to right).
    assert!(
        response_at_edge.abs() > response_flat_left.abs(),
        "edge response ({}) should dominate flat-left response ({})",
        response_at_edge,
        response_flat_left,
    );
    assert!(
        response_at_edge.abs() > response_flat_right.abs(),
        "edge response ({}) should dominate flat-right response ({})",
        response_at_edge,
        response_flat_right,
    );

    // The sign of the response depends on the kernel convention (row vs
    // column orientation).  What matters is that the *magnitude* at the
    // edge is large — we already checked that above.
    println!(
        "Edge response sign: {}",
        if response_at_edge > 0.0 {
            "positive"
        } else {
            "negative"
        }
    );

    // ── Print a small cross-section ─────────────────────────────────────
    //
    // Show the Sobel response for every column in the middle row so the
    // user can see the spike at the edge visually.
    println!("\nFull Sobel-X profile along row y={sample_y}:");
    for x in 0..width {
        // Extract the scalar once at the display boundary.
        let val = edges.pixel_at(x, sample_y).0;
        // A simple text-mode bar: each '#' represents ~25 units of response.
        let bar_len = (val.abs() / 25.0).round() as usize;
        let bar: String = "#".repeat(bar_len);
        let sign = if val >= 0.0 { '+' } else { '-' };
        println!("  x={x:>2}: {sign}{:>7.1}  {bar}", val.abs());
    }

    println!("\nEdge detection verified — the Sobel-X filter found the vertical edge!");
}

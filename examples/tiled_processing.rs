//! # Tiled Processing
//!
//! Demonstrates region-of-interest (ROI) access and tile-based processing,
//! a pattern commonly used in industrial vision for parallel-friendly pipelines.
//!
//! Run with: `cargo run --example tiled_processing`

fn main() {
    use fovea::Size;
    use fovea::image::{Image, ImageView, SubView};
    use fovea::pixel::Mono8;

    // ── Build a test image ────────────────────────────────────────────
    //
    // Create a 12×8 image with a simple gradient so we can observe
    // how tile boundaries partition the data.  Each pixel's intensity
    // is its linear index mod 256.

    let pixels: Vec<Mono8> = (0u32..96).map(|i| Mono8::new((i % 256) as u8)).collect();

    // from_vec checks that len == width * height; unwrap is safe here.
    let img = Image::from_vec(12, 8, pixels).unwrap();
    println!("Source image: {}×{}", img.width(), img.height());

    // ── Tile iteration ────────────────────────────────────────────────
    //
    // `tiles` splits the image into a grid of non-overlapping tiles.
    // Edge tiles are automatically clamped when dimensions aren't exact
    // multiples of the tile size.
    //
    // 12×8 with 4×4 tiles → 3 columns × 2 rows = 6 tiles, all full-size.

    let tile_size = Size::new(4, 4);
    let tiles: Vec<_> = img.tiles(tile_size).collect();

    println!(
        "\nTile grid: {} tiles (target {}×{})",
        tiles.len(),
        tile_size.width,
        tile_size.height
    );

    // Verify the expected tile count for this exact-multiple case.
    assert_eq!(tiles.len(), 6, "12/4 * 8/4 = 3*2 = 6 tiles");

    // ── Compute per-tile statistics ───────────────────────────────────
    //
    // Each tile implements `ImageView`, so we access pixels through
    // `pixel_at(x, y)`.  We use nested loops rather than `as_slice()`
    // because tiles are *strided* sub-views — their rows are not
    // contiguous in memory.

    println!("\nPer-tile average intensity:");
    for (i, tile) in tiles.iter().enumerate() {
        let avg = average_intensity(*tile);
        println!(
            "  Tile {i}: {}×{} — avg intensity = {avg:.1}",
            tile.width(),
            tile.height()
        );
    }

    // ── Find the brightest tile ───────────────────────────────────────
    //
    // A common industrial-vision pattern: score each tile and select
    // regions of interest for further (expensive) processing, skipping
    // tiles that are uninteresting.

    let (brightest_idx, brightest_avg) = tiles
        .iter()
        .enumerate()
        .map(|(i, tile)| (i, average_intensity(*tile)))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();

    println!("\nBrightest tile: #{brightest_idx} (avg = {brightest_avg:.1})");

    // ── Partial-edge tiles ────────────────────────────────────────────
    //
    // When tile size doesn't evenly divide the image, edge tiles are
    // smaller.  Let's demonstrate with 5×5 tiles on our 12×8 image.
    //
    //   columns: 5, 5, 2   (12 = 5+5+2)
    //   rows:    5, 3       ( 8 = 5+3)
    //   total:   3 × 2 = 6 tiles

    let big_tile = Size::new(5, 5);
    let partial_tiles: Vec<_> = img.tiles(big_tile).collect();

    println!(
        "\nPartial-edge tiles ({}×{}):",
        big_tile.width, big_tile.height
    );
    for (i, tile) in partial_tiles.iter().enumerate() {
        // Edge tiles will be smaller than 5×5.
        println!("  Tile {i}: {}×{}", tile.width(), tile.height());
    }

    // 3 columns × 2 rows = 6 tiles total.
    assert_eq!(partial_tiles.len(), 6);

    // The bottom-right tile should be the smallest (2×3).
    let last = partial_tiles.last().unwrap();
    assert_eq!(last.width(), 2, "rightmost column: 12 - 5 - 5 = 2");
    assert_eq!(last.height(), 3, "bottom row: 8 - 5 = 3");

    // ── ROI (Region of Interest) access ───────────────────────────────
    //
    // `SubView::roi` extracts an arbitrary rectangular sub-view.
    // Returns `None` if the rectangle extends beyond image bounds.

    let roi_rect = fovea::Rectangle::new((2, 1), (4, 3));
    let roi = img.roi(roi_rect).expect("ROI is within bounds");

    println!("\nROI at (2,1) size 4×3:");
    for y in 0..roi.height() {
        for x in 0..roi.width() {
            // pixel_at returns &Mono8; value() extracts the u8.
            let v = roi.pixel_at(x, y).value();
            print!("{v:3} ");
        }
        println!();
    }

    // Verify that the ROI view reads the correct pixels from the
    // parent image — pixel (0,0) of the ROI should equal pixel (2,1)
    // of the original.
    assert_eq!(
        roi.pixel_at(0, 0).value(),
        img.pixel_at(2, 1).value(),
        "ROI origin maps to parent (2,1)"
    );

    println!("\nDone! Tile-based processing enables efficient, parallel-friendly pipelines.");
}

/// Compute the mean intensity of all pixels in an `ImageView<Pixel = Mono8>`.
///
/// Uses nested x/y loops because sub-views (tiles, ROIs) are strided —
/// their pixel rows are not contiguous in memory, so `as_slice()` is
/// not available.
fn average_intensity(view: impl fovea::image::ImageView<Pixel = fovea::pixel::Mono8>) -> f64 {
    let mut sum = 0u64;
    for y in 0..view.height() {
        for x in 0..view.width() {
            // pixel_at is the universal accessor on ImageView.
            sum += view.pixel_at(x, y).value() as u64;
        }
    }
    // Guard against zero-area views (shouldn't happen with tiles, but
    // good practice).
    let count = view.width() * view.height();
    if count == 0 {
        return 0.0;
    }
    sum as f64 / count as f64
}

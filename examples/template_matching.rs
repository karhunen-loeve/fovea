//! # Template Matching
//!
//! Demonstrate template matching with the three built-in strategies:
//! [`SAD`], [`SSD`], and [`NCC`].
//!
//! Template matching slides a small template image over a larger source
//! image and computes a similarity score at every valid position.  The
//! result is a score map whose dimensions are
//! `(image_w − template_w + 1, image_h − template_h + 1)`.
//!
//! All three strategies build on the loop-inverted `fold_neighborhood`
//! engine, which means the heavy inner loop auto-vectorizes on x86
//! (confirmed by OPT-001 benchmarks).
//!
//! Run with: `cargo run --example template_matching`

fn main() {
    // ── Imports ──────────────────────────────────────────────────────────
    use fovea::image::{Image, ImageView};
    use fovea::transform::{NCC, SAD, SSD, match_template};

    // =====================================================================
    // 1. Build a synthetic test image with a known pattern
    // =====================================================================
    //
    // A 10×10 gradient image where pixel(x, y) = x + y * 10.
    // This gives us predictable values for hand-verification.

    let image = Image::generate(10, 10, |x, y| (x + y * 10) as u8);

    println!("Source image: 10×10 gradient (pixel = x + y*10)");
    println!("  pixel(0,0) = {}", image.pixel_at(0, 0)); // 0
    println!("  pixel(5,3) = {}", image.pixel_at(5, 3)); // 35
    println!("  pixel(9,9) = {}", image.pixel_at(9, 9)); // 99

    // =====================================================================
    // 2. Extract a 3×3 template from the image
    // =====================================================================
    //
    // We copy the 3×3 patch starting at (2, 3).  This patch contains:
    //   row 3: [32, 33, 34]
    //   row 4: [42, 43, 44]
    //   row 5: [52, 53, 54]

    let template = Image::generate(3, 3, |x, y| image.pixel_at(x + 2, y + 3));

    println!("\nTemplate: 3×3 patch from position (2,3)");
    for y in 0..3 {
        print!("  ");
        for x in 0..3 {
            print!("{:3} ", template.pixel_at(x, y));
        }
        println!();
    }

    // =====================================================================
    // 3. SAD — Sum of Absolute Differences
    // =====================================================================
    //
    // SAD computes Σ|I − T| for each position.  Lower is better.
    // A score of 0 means a perfect match.

    let sad_result = match_template(&image, &template, SAD).unwrap();

    println!("\n── SAD (Sum of Absolute Differences) ──");
    println!(
        "Score map size: {}×{}",
        sad_result.width(),
        sad_result.height()
    );

    // Output size should be (10 - 3 + 1) × (10 - 3 + 1) = 8×8
    assert_eq!(sad_result.width(), 8);
    assert_eq!(sad_result.height(), 8);

    // At position (2, 3) the patch matches exactly → score = 0
    let sad_at_source = sad_result.pixel_at(2, 3);
    println!("  SAD at source (2,3) = {}", sad_at_source.0);
    assert_eq!(sad_at_source.0, 0.0, "exact match should have SAD = 0");

    // Find the position with the minimum SAD score
    let mut min_sad = f32::MAX;
    let mut min_sad_pos = (0, 0);
    for y in 0..sad_result.height() {
        for x in 0..sad_result.width() {
            let score = sad_result.pixel_at(x, y).0;
            if score < min_sad {
                min_sad = score;
                min_sad_pos = (x, y);
            }
        }
    }
    println!(
        "  Best match: ({}, {}) with SAD = {}",
        min_sad_pos.0, min_sad_pos.1, min_sad
    );
    assert_eq!(min_sad_pos, (2, 3));
    assert_eq!(min_sad, 0.0);

    // =====================================================================
    // 4. SSD — Sum of Squared Differences
    // =====================================================================
    //
    // SSD computes Σ(I − T)² for each position.  Lower is better.
    // Penalizes large differences more than SAD does.

    let ssd_result = match_template(&image, &template, SSD).unwrap();

    println!("\n── SSD (Sum of Squared Differences) ──");

    let ssd_at_source = ssd_result.pixel_at(2, 3);
    println!("  SSD at source (2,3) = {}", ssd_at_source.0);
    assert_eq!(ssd_at_source.0, 0.0, "exact match should have SSD = 0");

    // At position (3, 3), each pixel differs by 1 from the template.
    // SSD = 9 × 1² = 9
    let ssd_shifted = ssd_result.pixel_at(3, 3);
    println!("  SSD at (3,3) = {} (shifted by 1 in x)", ssd_shifted.0);
    assert_eq!(ssd_shifted.0, 9.0);

    // Find the position with the minimum SSD score
    let mut min_ssd = f32::MAX;
    let mut min_ssd_pos = (0, 0);
    for y in 0..ssd_result.height() {
        for x in 0..ssd_result.width() {
            let score = ssd_result.pixel_at(x, y).0;
            if score < min_ssd {
                min_ssd = score;
                min_ssd_pos = (x, y);
            }
        }
    }
    println!(
        "  Best match: ({}, {}) with SSD = {}",
        min_ssd_pos.0, min_ssd_pos.1, min_ssd
    );
    assert_eq!(min_ssd_pos, (2, 3));

    // =====================================================================
    // 5. NCC — Normalized Cross-Correlation
    // =====================================================================
    //
    // NCC computes the Pearson correlation coefficient between the image
    // patch and the template.  Higher is better:
    //   +1.0 = perfect positive correlation
    //   -1.0 = perfect negative correlation
    //    0.0 = no linear correlation

    let ncc_result = match_template(&image, &template, NCC).unwrap();

    println!("\n── NCC (Normalized Cross-Correlation) ──");

    let ncc_at_source = ncc_result.pixel_at(2, 3);
    println!("  NCC at source (2,3) = {}", ncc_at_source.0);
    assert!(
        (ncc_at_source.0 - 1.0).abs() < 1e-6,
        "exact match should have NCC ≈ 1.0"
    );

    // NCC is invariant to affine brightness changes, so even shifted
    // patches that maintain the same gradient should score high.
    // All patches in our gradient image have the same pattern (just
    // shifted in value), so all NCC scores should be ≈ 1.0.
    println!("  NCC scores across the map:");
    let mut all_high = true;
    for y in 0..ncc_result.height() {
        for x in 0..ncc_result.width() {
            let score = ncc_result.pixel_at(x, y).0;
            if (score - 1.0).abs() > 1e-4 {
                all_high = false;
            }
        }
    }
    println!("  All NCC scores ≈ 1.0? {}", all_high);
    assert!(
        all_high,
        "gradient image: all patches have the same structure, NCC should be ≈ 1.0 everywhere"
    );

    // =====================================================================
    // 6. Error handling — template larger than image
    // =====================================================================
    //
    // `match_template` returns `Err` when the template doesn't fit.
    // This is a Tier 2 error (data-dependent), not a panic.

    let big_template = Image::fill(20, 20, 0u8);
    let result: Result<Image<fovea::pixel::MonoF32>, _> =
        match_template(&image, &big_template, SAD);
    println!("\n── Error handling ──");
    match result {
        Err(e) => println!("  Template 20×20 on 10×10 image: {}", e),
        Ok(_) => panic!("expected Err for oversized template"),
    }

    assert!(
        match_template::<_, _, fovea::pixel::MonoF32, _>(&image, &big_template, SAD).is_err(),
        "oversized template should return Err"
    );

    // =====================================================================
    // 7. Summary
    // =====================================================================

    println!("\n── Summary ──");
    println!("SAD/SSD: distance measures (lower = better match)");
    println!("NCC:     correlation (higher = better match, invariant to brightness shifts)");
    println!("All three use the loop-inverted fold_neighborhood engine for vectorization.");
    println!("Template matching verified!");
}

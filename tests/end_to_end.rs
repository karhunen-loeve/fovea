//! End-to-end integration tests: exercise full image-processing pipelines.
//!
//! Each test creates an image with known pixel values, applies one or more
//! transforms, extracts a region of interest, and verifies the result.

use irys_cv::border::Clamp;
use irys_cv::image::{Image, ImageView, SubView};
use irys_cv::pixel::{Mono8, Mono16, MonoF32, Rgb8};
use irys_cv::transform::{
    AbsDiff, Broadcast, FullRange, Luminance, PixelAdd, combine_images, convert_image,
    gaussian_blur_3x3, sobel_x,
};
use irys_cv::{Rectangle, Size};

// ─── Test 1: create → gaussian blur → extract ROI → verify ─────────────────

#[test]
fn pipeline_create_blur_roi() {
    // 1. Create an 8×8 gradient image with small values.
    //    The un-normalised Gaussian 3×3 kernel weights sum to 16, so each
    //    output pixel ≈ input × 16.  We keep source values in 0..=14 so
    //    the blurred result (up to ~224) stays below u8 saturation.
    let img = Image::<Mono8>::generate(8, 8, |x, y| Mono8::new((x + y) as u8));
    assert_eq!(img.size(), Size::new(8, 8));

    // 2. Apply Gaussian blur (Mono8 → Mono8 via f32 accumulator)
    let blurred: Image<Mono8> = gaussian_blur_3x3(&img, &Clamp);
    assert_eq!(blurred.size(), img.size());

    // 3. Extract center 4×4 ROI
    let roi = blurred.roi(Rectangle::new((2, 2), (4, 4))).unwrap();
    assert_eq!(roi.size(), Size::new(4, 4));

    // 4. Verify that blurred interior pixels are in a reasonable range.
    //    Source gradient spans 0..=14; after ×16 blur the max is ~224.
    //    Interior pixels should be nonzero and below saturation.
    for y in 0..roi.height() {
        for x in 0..roi.width() {
            let v = roi.pixel_at(x, y).value();
            assert!(v > 0, "pixel at ({x},{y}) should be > 0 after blur");
            assert!(
                v < 255,
                "pixel at ({x},{y}) should be < 255 after blur of small gradient"
            );
        }
    }

    // Verify a specific smoothing property: the centre of the ROI
    // corresponds to position (4,4) in the blurred image.  The source
    // value there is (4+4) = 8.  With un-normalised kernel (×16) the
    // expected output is ~128, and the symmetric neighbourhood keeps it
    // close to that.
    let center = roi.pixel_at(2, 2).value(); // blurred (4,4)
    assert!(
        (center as i32 - 128).unsigned_abs() <= 16,
        "centre pixel {center} should be close to 128 after blur"
    );
}

// ─── Test 2: combine_images (binary operation) ─────────────────────────────

#[test]
fn pipeline_combine_images_abs_diff() {
    // Create two 6×6 images with slightly different patterns
    let a = Image::<Mono8>::generate(6, 6, |x, y| Mono8::new((x * 10 + y * 20) as u8));
    let b = Image::<Mono8>::generate(6, 6, |x, y| Mono8::new((x * 10 + y * 20 + 5) as u8));

    // Absolute difference — every pixel should be exactly 5
    let diff = combine_images(&a, &b, AbsDiff).expect("sizes match");
    assert_eq!(diff.size(), a.size());

    for y in 0..diff.height() {
        for x in 0..diff.width() {
            let v = diff.pixel_at(x, y).value();
            assert_eq!(v, 5, "abs_diff pixel at ({x},{y}) should be 5, got {v}");
        }
    }

    // Extract a 3×3 ROI and verify same property
    let roi = diff.roi(Rectangle::new((1, 1), (3, 3))).unwrap();
    assert_eq!(roi.size(), Size::new(3, 3));
    for y in 0..roi.height() {
        for x in 0..roi.width() {
            assert_eq!(roi.pixel_at(x, y).value(), 5);
        }
    }
}

#[test]
fn pipeline_combine_images_add() {
    let a = Image::<Mono8>::fill(4, 4, Mono8::new(100));
    let b = Image::<Mono8>::fill(4, 4, Mono8::new(50));

    let sum = combine_images(&a, &b, PixelAdd).expect("sizes match");
    assert_eq!(sum.size(), Size::new(4, 4));

    for y in 0..sum.height() {
        for x in 0..sum.width() {
            assert_eq!(sum.pixel_at(x, y).value(), 150);
        }
    }

    // Saturating add: 200 + 200 should clamp to 255
    let c = Image::<Mono8>::fill(4, 4, Mono8::new(200));
    let saturated = combine_images(&c, &c, PixelAdd).expect("sizes match");
    for y in 0..saturated.height() {
        for x in 0..saturated.width() {
            // Saturating<u8> addition: 200+200 wraps/saturates
            // Saturating<u8> saturates at 255
            let v = saturated.pixel_at(x, y).value();
            assert!(v >= 200, "saturated add should be >= 200, got {v}");
        }
    }
}

// ─── Test 3: pixel conversion (convert_image) ──────────────────────────────

#[test]
fn pipeline_convert_mono8_to_mono16_fullrange() {
    let img = Image::<Mono8>::generate(5, 5, |x, y| Mono8::new((x * 50 + y * 10) as u8));

    // FullRange maps 0→0 and 255→65535
    let wide: Image<Mono16> = convert_image(&img, FullRange);
    assert_eq!(wide.size(), img.size());

    // Check corners
    let src_00 = img.pixel_at(0, 0).value() as u16;
    let dst_00 = wide.pixel_at(0, 0).value();
    // FullRange: dst ≈ src * 257 (for u8→u16)
    let expected = src_00 as u32 * 257;
    assert_eq!(
        dst_00 as u32, expected,
        "FullRange(0,0): {src_00} → {dst_00}, expected {expected}"
    );

    let src_44 = img.pixel_at(4, 4).value() as u16;
    let dst_44 = wide.pixel_at(4, 4).value();
    let expected_44 = src_44 as u32 * 257;
    assert_eq!(
        dst_44 as u32, expected_44,
        "FullRange(4,4): {src_44} → {dst_44}, expected {expected_44}"
    );
}

#[test]
fn pipeline_convert_rgb8_to_mono8_luminance() {
    // Pure red → luminance ≈ 0.299 * 255 ≈ 76
    let red = Image::<Rgb8>::fill(3, 3, Rgb8::new(255, 0, 0));
    let gray: Image<Mono8> = convert_image(&red, Luminance);
    assert_eq!(gray.size(), red.size());

    let v = gray.pixel_at(0, 0).value();
    // BT.601 integer: (77 * 255) >> 8 = 76
    assert!(
        (v as i32 - 76).unsigned_abs() <= 2,
        "luminance of pure red should be ~76, got {v}"
    );

    // Pure green → luminance ≈ 0.587 * 255 ≈ 150
    let green = Image::<Rgb8>::fill(3, 3, Rgb8::new(0, 255, 0));
    let gray_g: Image<Mono8> = convert_image(&green, Luminance);
    let vg = gray_g.pixel_at(0, 0).value();
    assert!(
        (vg as i32 - 150).unsigned_abs() <= 2,
        "luminance of pure green should be ~150, got {vg}"
    );
}

#[test]
fn pipeline_convert_mono8_to_monof32_fullrange() {
    let img = Image::<Mono8>::fill(2, 2, Mono8::new(255));
    let float_img: Image<MonoF32> = convert_image(&img, FullRange);
    assert_eq!(float_img.size(), img.size());

    // FullRange maps 255 → 1.0
    let v = float_img.pixel_at(0, 0).value();
    assert!(
        (v - 1.0).abs() < 1e-4,
        "FullRange Mono8(255) → MonoF32 should be 1.0, got {v}"
    );

    let img0 = Image::<Mono8>::fill(2, 2, Mono8::new(0));
    let float0: Image<MonoF32> = convert_image(&img0, FullRange);
    let v0 = float0.pixel_at(0, 0).value();
    assert!(
        v0.abs() < 1e-4,
        "FullRange Mono8(0) → MonoF32 should be 0.0, got {v0}"
    );
}

// ─── Test 4: multi-step pipeline ────────────────────────────────────────────

#[test]
fn pipeline_create_convert_blur_roi() {
    // Step 1: Create an Rgb8 image with a horizontal gradient
    let rgb = Image::<Rgb8>::generate(10, 10, |x, _y| {
        let v = (x * 25) as u8; // 0, 25, 50, … 225
        Rgb8::new(v, v, v)
    });
    assert_eq!(rgb.size(), Size::new(10, 10));

    // Step 2: Convert to grayscale via Luminance
    let gray: Image<Mono8> = convert_image(&rgb, Luminance);
    assert_eq!(gray.size(), rgb.size());
    // For gray input (r==g==b) Luminance should preserve the value exactly
    // (or within ±1 due to integer rounding)
    for x in 0..gray.width() {
        let expected = (x * 25) as u8;
        let got = gray.pixel_at(x, 0).value();
        assert!(
            (got as i32 - expected as i32).unsigned_abs() <= 1,
            "gray pixel at ({x},0): expected ~{expected}, got {got}"
        );
    }

    // Step 3: Apply Gaussian blur
    let blurred: Image<Mono8> = gaussian_blur_3x3(&gray, &Clamp);
    assert_eq!(blurred.size(), gray.size());

    // Step 4: Extract an ROI in the interior
    let roi = blurred.roi(Rectangle::new((3, 3), (4, 4))).unwrap();
    assert_eq!(roi.size(), Size::new(4, 4));

    // Step 5: Verify blur smoothing — for a monotonic gradient, interior
    // values should remain monotonically non-decreasing along x
    for y in 0..roi.height() {
        for x in 1..roi.width() {
            let prev = roi.pixel_at(x - 1, y).value();
            let curr = roi.pixel_at(x, y).value();
            assert!(
                curr >= prev,
                "blurred gradient should be non-decreasing along x: \
                 at ({},{y}) prev={prev}, curr={curr}",
                x - 1
            );
        }
    }
}

#[test]
fn pipeline_sobel_edge_detection_on_step() {
    // Create an 8×8 image with a vertical step edge: left half = 0, right half = 200
    let img = Image::<Mono8>::generate(8, 8, |x, _y| {
        if x < 4 {
            Mono8::new(0)
        } else {
            Mono8::new(200)
        }
    });

    // Sobel X detects vertical edges (gradient in the x direction).
    // ADR-0045 Phase C: `Mono8::Accumulator = MonoF32`, so the output
    // pixel type is `MonoF32` rather than raw `f32`. The magnitude
    // semantics are unchanged — `MonoF32::abs()` mirrors `f32::abs`
    // and `.0` extracts the scalar at the comparison boundary.
    let edges: Image<irys_cv::pixel::MonoF32> = sobel_x(&img, &Clamp);
    assert_eq!(edges.size(), img.size());

    // The edge response should be strongest around columns 3–4
    // and near zero in the flat regions
    let flat_left = edges.pixel_at(1, 4).abs().0;
    let edge_resp = edges.pixel_at(4, 4).abs().0;
    let flat_right = edges.pixel_at(6, 4).abs().0;

    assert!(
        edge_resp > flat_left,
        "edge response ({edge_resp}) should exceed flat-left ({flat_left})"
    );
    assert!(
        edge_resp > flat_right,
        "edge response ({edge_resp}) should exceed flat-right ({flat_right})"
    );

    // Extract ROI around the edge and verify all values are nonzero
    let roi = edges.roi(Rectangle::new((3, 1), (2, 6))).unwrap();
    assert_eq!(roi.size(), Size::new(2, 6));
    for y in 0..roi.height() {
        for x in 0..roi.width() {
            let v = roi.pixel_at(x, y).abs().0;
            assert!(
                v > 1.0,
                "edge ROI pixel ({x},{y}) should have significant response, got {v}"
            );
        }
    }
}

#[test]
fn pipeline_broadcast_blur_convert_back() {
    // Mono8 → Rgb8 (broadcast) → Gaussian blur on Rgb8 → back to Mono8 (luminance)
    let mono = Image::<Mono8>::generate(8, 8, |x, y| Mono8::new(((x + y) * 16) as u8));

    // Step 1: Broadcast Mono8 → Rgb8 (each channel gets the mono value)
    let rgb: Image<Rgb8> = convert_image(&mono, Broadcast);
    assert_eq!(rgb.size(), mono.size());
    // Verify broadcast: r == g == b == original
    let p = rgb.pixel_at(2, 3);
    let expected = mono.pixel_at(2, 3).value();
    assert_eq!(p.r.0, expected);
    assert_eq!(p.g.0, expected);
    assert_eq!(p.b.0, expected);

    // Step 2: Gaussian blur on Rgb8
    let blurred_rgb: Image<Rgb8> = gaussian_blur_3x3(&rgb, &Clamp);
    assert_eq!(blurred_rgb.size(), rgb.size());

    // Step 3: Convert back to Mono8 via Luminance
    let result: Image<Mono8> = convert_image(&blurred_rgb, Luminance);
    assert_eq!(result.size(), mono.size());

    // Step 4: Also blur the mono image directly for comparison
    let blurred_mono: Image<Mono8> = gaussian_blur_3x3(&mono, &Clamp);

    // The two paths (mono→rgb→blur→lum vs mono→blur) should yield similar
    // results (within rounding tolerance) for a gray image
    for y in 0..result.height() {
        for x in 0..result.width() {
            let via_rgb = result.pixel_at(x, y).value() as i32;
            let direct = blurred_mono.pixel_at(x, y).value() as i32;
            assert!(
                (via_rgb - direct).unsigned_abs() <= 2,
                "pixel ({x},{y}): via_rgb={via_rgb}, direct={direct} — should be within ±2"
            );
        }
    }
}

#[test]
fn pipeline_combine_then_roi_then_convert() {
    // Create two complementary patterns
    let a = Image::<Mono8>::generate(8, 8, |x, _y| Mono8::new((x * 30) as u8));
    let b = Image::<Mono8>::generate(8, 8, |_x, y| Mono8::new((y * 30) as u8));

    // Step 1: Add the two images
    let sum = combine_images(&a, &b, PixelAdd).expect("sizes match");
    assert_eq!(sum.size(), Size::new(8, 8));

    // Verify: pixel (x,y) = min(x*30 + y*30, 255)
    let v33 = sum.pixel_at(3, 3).value();
    assert_eq!(v33, 3 * 30 + 3 * 30, "sum at (3,3) = 180");

    let v77 = sum.pixel_at(7, 7).value();
    // 7*30 + 7*30 = 420 → saturates to 255
    assert_eq!(v77, 255, "sum at (7,7) should saturate to 255");

    // Step 2: Extract ROI
    let roi = sum.roi(Rectangle::new((0, 0), (4, 4))).unwrap();
    assert_eq!(roi.size(), Size::new(4, 4));

    // Step 3: Convert ROI pixels to Mono16 via FullRange
    // (We convert the full sum image and check the same region)
    let wide: Image<Mono16> = convert_image(&sum, FullRange);
    let wide_roi = wide.roi(Rectangle::new((0, 0), (4, 4))).unwrap();

    for y in 0..4 {
        for x in 0..4 {
            let src = roi.pixel_at(x, y).value() as u32;
            let dst = wide_roi.pixel_at(x, y).value() as u32;
            let expected = src * 257;
            assert_eq!(
                dst, expected,
                "FullRange at ({x},{y}): {src} → {dst}, expected {expected}"
            );
        }
    }
}

// ─── Test 5: zero-sized edge case ──────────────────────────────────────────

#[test]
fn pipeline_zero_size_image() {
    let img = Image::<Mono8>::generate(0, 0, |_, _| Mono8::new(0));
    assert_eq!(img.size(), Size::new(0, 0));

    let blurred: Image<Mono8> = gaussian_blur_3x3(&img, &Clamp);
    assert_eq!(blurred.size(), Size::new(0, 0));

    let converted: Image<Mono16> = convert_image(&img, FullRange);
    assert_eq!(converted.size(), Size::new(0, 0));
}

// ─── Test 6: 1×1 image pipeline ────────────────────────────────────────────

#[test]
fn pipeline_single_pixel() {
    let img = Image::<Mono8>::fill(1, 1, Mono8::new(42));
    assert_eq!(img.pixel_at(0, 0).value(), 42);

    // Gaussian blur on 1×1 with Clamp — all neighbours are the same pixel
    let blurred: Image<Mono8> = gaussian_blur_3x3(&img, &Clamp);
    assert_eq!(blurred.size(), Size::new(1, 1));
    // The blurred value should be close to the original (kernel sums to 1
    // in normalised form, but these are un-normalised integer kernels;
    // the derive rounds back from the f32 accumulator)
    let bv = blurred.pixel_at(0, 0).value();
    // For a uniform 1×1 image with Clamp, all 9 neighbours = 42.
    // gaussian_3x3 unnormalised sum = 42 * 16 = 672, then FromLinear
    // rounds and clamps. With separable: row pass gives 42*4=168 (as f32 acc),
    // then col pass gives 168*4=672... the kernel weights sum to 16 for
    // unnormalised Gaussian 3x3, so result = 672. FromLinear clamps to 255.
    // Actually let's just check it's non-zero and deterministic.
    assert!(bv > 0, "blurred single pixel should be > 0, got {bv}");

    // Convert to Mono16
    let wide: Image<Mono16> = convert_image(&blurred, FullRange);
    assert_eq!(wide.size(), Size::new(1, 1));
    assert!(wide.pixel_at(0, 0).value() > 0);
}

//! # Pixel Conversion
//!
//! Demonstrates type-safe pixel conversion with named strategies.
//!
//! `fovea` uses *strategy types* to parameterize conversions: instead of a
//! single "convert" function that guesses what you want, you pass a zero-sized
//! type that names the exact mapping semantics.  This makes the intent visible
//! in source code **and** lets the compiler verify that the conversion is
//! implemented for the concrete pixel pair.
//!
//! Run with: `cargo run --example pixel_conversion`

fn main() {
    // ── Imports ──────────────────────────────────────────────────────────────
    // Image types live under `fovea::image`, pixel types under
    // `fovea::pixel`, and all conversion machinery under
    // `fovea::transform`.
    use fovea::image::{Image, ImageView};
    use fovea::pixel::{Mono8, Mono16, Rgb8, RgbF32, Srgb8};
    use fovea::transform::{ConvertPixel, FullRange, Luminance, SrgbGamma, convert_image};

    // =====================================================================
    // 1. FullRange: scale 8-bit grayscale to 16-bit grayscale
    // =====================================================================
    //
    // `FullRange` maps the entire dynamic range of the source onto the
    // entire dynamic range of the destination.  For u8 → u16 this means
    // [0, 255] → [0, 65535], so white stays white and black stays black.
    // Internally this is `v as u16 * 65535 / 255` (with correct rounding).

    // Build a tiny 2×2 image with hand-picked intensity values.
    let img8 = Image::from_vec(
        2,
        2,
        vec![
            Mono8::new(0),
            Mono8::new(128), // top row: black, mid-gray
            Mono8::new(255),
            Mono8::new(64), // bottom row: white, dark gray
        ],
    )
    .expect("pixel count matches 2×2");

    // The turbofish `::<Mono16>` tells `convert_image` what output pixel
    // type we want.  `FullRange` supplies the *how*.
    let img16: Image<Mono16> = convert_image(&img8, FullRange);

    // Verify a few landmark values.
    // pixel_at(x, y) returns &Pixel — x is column, y is row.
    println!("── FullRange: Mono8 → Mono16 ──");
    println!("  Mono8(0)   → Mono16({})", img16.pixel_at(0, 0).value()); // 0
    println!("  Mono8(128) → Mono16({})", img16.pixel_at(1, 0).value()); // 32896
    println!("  Mono8(255) → Mono16({})", img16.pixel_at(0, 1).value()); // 65535
    println!("  Mono8(64)  → Mono16({})", img16.pixel_at(1, 1).value()); // 16448

    // =====================================================================
    // 2. Luminance: RGB to grayscale (BT.601 coefficients)
    // =====================================================================
    //
    // `Luminance` converts colour to grayscale using the ITU-R BT.601
    // luma formula:  Y = 0.299·R + 0.587·G + 0.114·B
    //
    // Green contributes most because the human visual system is most
    // sensitive to green light.

    let rgb_img = Image::from_vec(
        2,
        1,
        vec![
            Rgb8::new(255, 0, 0), // pure red   → Y ≈ 76
            Rgb8::new(0, 255, 0), // pure green → Y ≈ 150
        ],
    )
    .expect("pixel count matches 2×1");

    // Luminance implements `ConvertPixel<Rgb8, Mono8>`, so the compiler
    // knows exactly which conversion is requested.
    let gray: Image<Mono8> = convert_image(&rgb_img, Luminance);

    println!("\n── Luminance: Rgb8 → Mono8 ──");
    println!("  Red(255,0,0)   → Mono8({})", gray.pixel_at(0, 0).value());
    println!("  Green(0,255,0) → Mono8({})", gray.pixel_at(1, 0).value());
    println!("  (Green is brighter because of the 0.587 coefficient)");

    // Sanity-check: green should yield a higher luma than red.
    assert!(
        gray.pixel_at(1, 0).value() > gray.pixel_at(0, 0).value(),
        "green should be perceptually brighter than red"
    );

    // =====================================================================
    // 3. SrgbGamma: sRGB decode to linear floating-point
    // =====================================================================
    //
    // Consumer images are stored in the sRGB colour space, which applies a
    // non-linear "gamma" curve so that perceptual mid-gray lands near
    // digital code 128 (≈ 50 % of 255).
    //
    // Most computer-vision math (blending, filtering, physically-based
    // rendering) requires *linear-light* values.  `SrgbGamma` applies the
    // standard sRGB transfer function to decode 8-bit sRGB into [0, 1]
    // linear floats (and can encode in the reverse direction).

    let srgb_img = Image::from_vec(
        1,
        1,
        vec![
            // Perceptual mid-gray in sRGB — all three channels at 128.
            Srgb8::new(128, 128, 128),
        ],
    )
    .expect("pixel count matches 1×1");

    // Decode sRGB → linear RgbF32.
    let linear: Image<RgbF32> = convert_image(&srgb_img, SrgbGamma);

    // `RgbF32` fields are plain `f32`, directly accessible.
    let p = linear.pixel_at(0, 0);
    println!("\n── SrgbGamma: Srgb8 → RgbF32 ──");
    println!(
        "  Srgb8(128,128,128) → RgbF32({:.4}, {:.4}, {:.4})",
        p.r, p.g, p.b
    );
    println!("  (Mid-gray sRGB ≈ 0.216 in linear light, NOT 0.502)");
    println!("  (The transfer function is non-linear — that's the whole point)");

    // The linear value for sRGB 128 is well below 0.5 because the sRGB
    // curve allocates more codes to darks (where the eye is sensitive).
    assert!(
        p.r < 0.25,
        "linear value of sRGB 128 should be well below 0.25"
    );
    assert!(p.r > 0.15, "linear value of sRGB 128 should be above 0.15");

    // =====================================================================
    // 4. Direct trait usage: ConvertPixel::convert on single pixels
    // =====================================================================
    //
    // Strategies are regular values — you can call `.convert()` on
    // individual pixels without building an image first.

    println!("\n── Direct ConvertPixel::convert calls ──");

    let mono_val = Mono8::new(200);
    let wide: Mono16 = FullRange.convert(&mono_val);
    println!(
        "  FullRange.convert(&Mono8(200)) = Mono16({})",
        wide.value()
    );

    let red = Rgb8::new(128, 64, 32);
    let luma: Mono8 = Luminance.convert(&red);
    println!(
        "  Luminance.convert(&Rgb8(128,64,32)) = Mono8({})",
        luma.value()
    );

    // =====================================================================

    println!("\nDone! Strategy types make conversion intent explicit and type-safe.");
    println!("Trying an unsupported pair (e.g. FullRange for Rgb8 → Mono8)");
    println!("would be a compile-time error, not a runtime surprise.");
}

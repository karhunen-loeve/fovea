//! # FMA Optimization
//!
//! Demonstrates how fused multiply-add (FMA) optimization works in fovea,
//! how to check whether it's enabled, and how to enable it.
//!
//! Run with: `cargo run --example fma_optimization`
//!
//! For FMA-optimized builds:
//! ```sh
//! RUSTFLAGS="-C target-cpu=native" cargo run --release --example fma_optimization
//! ```

fn main() {
    use fovea::border::Clamp;
    use fovea::image::{Image, ImageView, Neighborhood};
    use fovea::pixel::Mono8;
    use fovea::transform::convolve;

    // =====================================================================
    // 1. Check compile-time FMA status
    // =====================================================================
    //
    // fovea uses `#[cfg(target_feature = "fma")]` to gate FMA codegen.
    // This is a compile-time check — the compiler embeds either the FMA
    // instruction (`vfmadd213ps`) or separate multiply+add (`vmulps` +
    // `vaddps`) into the binary at build time.

    let fma_enabled = cfg!(target_feature = "fma");
    println!("=== fovea FMA Optimization ===\n");
    println!(
        "Compile-time FMA: {}",
        if fma_enabled {
            "ENABLED ✓"
        } else {
            "disabled"
        }
    );

    // =====================================================================
    // 2. Runtime CPU feature detection (x86-64 only)
    // =====================================================================
    //
    // Even if the binary was compiled without FMA, the CPU may support it.
    // This section detects that and suggests the correct recompile command.

    #[cfg(target_arch = "x86_64")]
    {
        let cpu_has_fma = is_x86_feature_detected!("fma");
        println!(
            "Runtime FMA support: {}",
            if cpu_has_fma { "YES" } else { "no" }
        );

        if cpu_has_fma && !fma_enabled {
            println!("\n╔══════════════════════════════════════════════════════╗");
            println!("║  FMA is available but not enabled!                   ║");
            println!("║  Recompile for ~10-15% faster convolution:           ║");
            println!("║                                                      ║");
            println!("║  Unix/macOS:                                         ║");
            println!("║    RUSTFLAGS=\"-C target-cpu=native\" \\                ║");
            println!("║      cargo build --release                           ║");
            println!("║                                                      ║");
            println!("║  Windows PowerShell:                                 ║");
            println!("║    $env:RUSTFLAGS=\"-C target-cpu=native\"             ║");
            println!("║    cargo build --release                             ║");
            println!("╚══════════════════════════════════════════════════════╝");
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        println!("Platform: AArch64 — FMA is always available (baseline ISA)");
    }

    #[cfg(target_arch = "wasm32")]
    {
        println!("Platform: WASM — FMA not available (relaxed-SIMD not yet standardized)");
    }

    // =====================================================================
    // 3. Run a small convolution to verify correctness
    // =====================================================================
    //
    // The same convolution code runs with or without FMA — only the
    // generated instructions differ. Results may vary by ±1 ULP due to
    // FMA's single rounding (vs. separate multiply and add rounding).

    println!("\n--- Convolution correctness check ---\n");

    // Create a 5×5 image with a known pattern.
    let img = Image::<Mono8>::generate(5, 5, |x, y| Mono8::new(((x + y * 5) * 10) as u8));

    println!("Input 5×5 image (values 0..240 in steps of 10):");
    for y in 0..img.height() {
        let row: Vec<u8> = (0..img.width())
            .map(|x| img.pixel_at(x, y).value())
            .collect();
        println!("  {:?}", row);
    }

    // Identity kernel: convolution with a single 1.0 weight at center
    // should reproduce the input exactly.
    let identity = Neighborhood::<f32, 1, 1>::new([1.0]);
    let result: Image<Mono8> = convolve(&img, &identity, &Clamp);

    println!("\nAfter identity convolution (should match input):");
    for y in 0..result.height() {
        let row: Vec<u8> = (0..result.width())
            .map(|x| result.pixel_at(x, y).value())
            .collect();
        println!("  {:?}", row);
    }

    // Verify identity convolution preserves all values.
    for y in 0..img.height() {
        for x in 0..img.width() {
            assert_eq!(
                result.pixel_at(x, y),
                img.pixel_at(x, y),
                "identity convolution must preserve pixel ({}, {})",
                x,
                y,
            );
        }
    }
    println!("\n✓ Identity convolution verified — all pixels match.");

    // 3×3 box blur for a more realistic test.
    let weight = 1.0 / 9.0;
    let box_kernel = Neighborhood::<f32, 3, 3>::new([weight; 9]);
    let blurred: Image<Mono8> = convolve(&img, &box_kernel, &Clamp);

    println!("\nAfter 3×3 box blur:");
    for y in 0..blurred.height() {
        let row: Vec<u8> = (0..blurred.width())
            .map(|x| blurred.pixel_at(x, y).value())
            .collect();
        println!("  {:?}", row);
    }

    // Center pixel (2,2) = average of the 3×3 neighborhood.
    // Input values at (1..=3, 1..=3): 60,70,80,110,120,130,160,170,180
    // Sum = 1080, average = 120.
    let center = blurred.pixel_at(2, 2).value();
    assert_eq!(center, 120, "center pixel should be the exact average");
    println!("✓ Center pixel = {} (expected 120)", center);

    // =====================================================================
    // 4. Assembly inspection instructions
    // =====================================================================

    println!("\n--- Assembly inspection ---\n");
    println!("To inspect generated assembly, install cargo-show-asm:");
    println!("  cargo install cargo-show-asm\n");
    println!("Then run:");
    println!(
        "  cargo asm -p fovea --example asm_inspect --target-cpu native fold_convolve_u8_hot\n"
    );
    println!("Look for:");
    if fma_enabled {
        println!("  vfmadd213ps / vfmadd231ps  — FMA is active ✓");
    } else {
        println!("  vmulps + vaddps            — separate multiply and add (no FMA)");
        println!("  Recompile with -C target-cpu=native to get vfmadd* instructions");
    }
}

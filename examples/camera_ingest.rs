//! # Camera Ingest
//!
//! Demonstrates zero-copy and low-copy ingestion from raw byte buffers,
//! as you would receive from an industrial camera SDK.
//!
//! Run with: `cargo run --example camera_ingest`

fn main() {
    use fovea::image::{Image, ImageRef, ImageView};
    use fovea::pixel::{Mono8, Mono16, PlainChannel, Rgb8};

    // ── Scenario 1: Zero-copy from a byte buffer (Mono8) ──────────────
    //
    // An industrial camera SDK typically gives you a raw byte buffer.
    // For byte-aligned pixel types (ALIGN == 1), you can convert this
    // directly into an Image without copying the pixel data.

    // Simulate a 4×3 camera frame (12 bytes, one byte per pixel).
    // In practice this would come from a camera SDK callback.
    let camera_buffer: Vec<u8> = (0..12).collect();
    println!("Camera buffer: {} bytes", camera_buffer.len());

    // from_raw_bytes takes ownership of the Vec<u8> and reinterprets it
    // as pixel data — no memcpy, no allocation.  This is only possible
    // because Mono8 has ALIGN == 1, which is enforced at compile time.
    let img = Image::<Mono8>::from_raw_bytes(4, 3, camera_buffer)
        .expect("buffer length must match width × height × pixel_size");

    // The image now owns the original allocation — zero copies were made.
    println!("Zero-copy image: {}×{}", img.width(), img.height());

    // pixel_at(x, y) gives a reference to the pixel at column x, row y.
    println!("  pixel(0,0) = {:?}", img.pixel_at(0, 0));
    println!("  pixel(3,2) = {:?}", img.pixel_at(3, 2));

    // ── Scenario 2: Zero-copy borrow of existing pixel data ───────────
    //
    // If you already have a &[T] (e.g., from a memory-mapped file or
    // a C library), ImageRef lets you wrap it without any copy.
    // The lifetime of the view is tied to the source data.

    let existing_pixels = [
        Mono8::new(10),
        Mono8::new(20),
        Mono8::new(30),
        Mono8::new(40),
    ];

    // ImageRef::new borrows the slice — the pixels stay where they are.
    let view =
        ImageRef::new(2, 2, &existing_pixels).expect("slice length must match width × height");

    println!("\nBorrowed view: {}×{}", view.width(), view.height());
    // Reading through the view accesses the original memory directly.
    println!("  pixel(0,0) = {:?}", view.pixel_at(0, 0));
    println!("  pixel(1,1) = {:?}", view.pixel_at(1, 1));

    // ── Scenario 3: Copy-based ingestion for aligned types ────────────
    //
    // Mono16 has ALIGN == 2, so from_raw_bytes won't compile (enforced
    // at compile time via a const-assert, not at runtime!).
    // Use from_bytes_copy instead, which copies the byte data into a
    // properly aligned allocation.

    // Simulate a 2×2 Mono16 frame (8 bytes: 2 bytes per pixel × 4 pixels).
    // Each pair of bytes is a little-endian u16 on most platforms.
    let camera_buffer_16: Vec<u8> = vec![0x00, 0x01, 0xFF, 0x00, 0x80, 0x00, 0x00, 0x10];

    // from_bytes_copy handles the alignment mismatch by copying bytes
    // into a fresh Vec<Mono16> with correct alignment.
    let img16 = Image::<Mono16>::from_bytes_copy(2, 2, &camera_buffer_16)
        .expect("buffer length must match width × height × pixel_size");

    println!(
        "\nCopy-based Mono16 image: {}×{}",
        img16.width(),
        img16.height()
    );
    // The pixel values depend on native endianness; the important thing
    // is that alignment is handled correctly and safely.
    println!("  pixel(0,0) = {:?}", img16.pixel_at(0, 0));
    println!("  pixel(1,0) = {:?}", img16.pixel_at(1, 0));

    // ── Scenario 4: Byte-level round-trip ─────────────────────────────
    //
    // PlainPixel::as_bytes() gives you the raw memory representation of
    // any pixel.  This is useful for sending data back to hardware,
    // writing to files, or handing off to a C library.

    let pixel = Rgb8::new(255, 128, 0);

    // as_bytes() returns a &[u8] view into the pixel's memory.
    // For Rgb8 (#[repr(C)]), bytes are always [R, G, B] — the memory
    // layout is guaranteed by the PlainPixel trait.
    let bytes = pixel.as_bytes();
    println!("\nRgb8(255, 128, 0) as bytes: {:?}", bytes);

    // Verify the layout is what we expect: three bytes, in R-G-B order.
    assert_eq!(bytes.len(), 3, "Rgb8 is exactly 3 bytes");
    assert_eq!(bytes[0], 255, "first byte is red");
    assert_eq!(bytes[1], 128, "second byte is green");
    assert_eq!(bytes[2], 0, "third byte is blue");

    // ── Scenario 5: Verifying compile-time guarantees ─────────────────
    //
    // These static assertions document the alignment invariants that
    // make from_raw_bytes safe for byte-aligned types.

    // Byte-aligned types (ALIGN == 1) — eligible for from_raw_bytes.
    assert_eq!(Mono8::ALIGN, 1, "Mono8 is byte-aligned");
    assert_eq!(Rgb8::ALIGN, 1, "Rgb8 is byte-aligned");

    // Multi-byte-aligned types — must use from_bytes_copy.
    assert_eq!(Mono16::ALIGN, 2, "Mono16 requires 2-byte alignment");

    println!("\nDone! Zero-copy ingestion keeps your pipeline allocation-free.");
}

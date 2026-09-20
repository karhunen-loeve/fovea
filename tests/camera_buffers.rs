//! Padded camera buffers: taking a row-strided plane without copying it.
//!
//! Camera SDKs hand over planes whose row pitch is wider than the image.
//! Android's `YUV_420_888` is the common case: hardware wants each row aligned,
//! so a 1920-wide frame arrives with a `rowStride` of 1984 and 64 bytes of
//! padding per row. Read back-to-back, every row starts late and the error
//! accumulates, which shows up as a sheared picture.
//!
//! These tests pin the two halves of the contract: the tightly packed
//! constructor refuses such a buffer outright, and there is a zero-copy route
//! that takes it correctly.

use fovea::border::Clamp;
use fovea::image::{Image, ImageRef, ImageView, SubView};
use fovea::pixel::{Mono8, PlainPixel};
use fovea::transform::gaussian_blur_3x3;
use fovea::{Rectangle, Size};

/// Image 6 px wide over a buffer with a row pitch of 8.
const WIDTH: usize = 6;
const HEIGHT: usize = 4;
const ROW_STRIDE: usize = 8;

/// Sample value at (x, y), chosen so a one-pixel shear is visible.
fn sample(x: usize, y: usize) -> u8 {
    (y * 10 + x + 1) as u8
}

/// A padded plane: real samples in the first `WIDTH` bytes of every row,
/// `0xFF` in the padding, so any leak is loud rather than plausible.
fn padded_plane() -> Vec<u8> {
    let mut bytes = vec![0u8; ROW_STRIDE * HEIGHT];
    for y in 0..HEIGHT {
        for x in 0..ROW_STRIDE {
            bytes[y * ROW_STRIDE + x] = if x < WIDTH { sample(x, y) } else { 0xFF };
        }
    }
    bytes
}

// ─── Test 1: the padded buffer is refused as a packed one ──────────────────

#[test]
fn from_raw_bytes_rejects_a_padded_plane() {
    let bytes = padded_plane();

    // The caller asks for the image dimensions, not the buffer dimensions.
    // The length does not match, so this fails at the boundary instead of
    // producing a sheared image that looks almost right.
    let result = Image::<Mono8>::from_raw_bytes(WIDTH, HEIGHT, bytes);
    assert!(
        result.is_err(),
        "a padded plane must not pass as WIDTH*HEIGHT"
    );
}

// ─── Test 2: the same buffer, taken zero-copy ──────────────────────────────

#[test]
fn padded_plane_survives_as_a_strided_roi() {
    let bytes = padded_plane();

    // 1. Reinterpret the bytes as pixels. Borrowed, nothing is copied.
    let pixels: &[Mono8] = Mono8::cast_slice(&bytes).expect("byte-aligned pixel type");

    // 2. View the buffer at the width it actually has, padding included.
    let padded = ImageRef::new(ROW_STRIDE, HEIGHT, pixels).expect("buffer is ROW_STRIDE*HEIGHT");

    // 3. Crop the padding away. The sub-view keeps the row pitch of its
    //    parent, which is exactly what a strided plane needs.
    let frame = padded
        .roi(Rectangle::new((0, 0), Size::new(WIDTH, HEIGHT)))
        .expect("the image fits inside its own buffer");

    assert_eq!(frame.size(), Size::new(WIDTH, HEIGHT));

    // Every sample is its own, so no row started late and no padding leaked.
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            assert_eq!(
                frame.pixel_at(x, y),
                Mono8::new(sample(x, y)),
                "at ({x}, {y})"
            );
        }
    }
}

// ─── Test 3: the strided view is usable, not merely readable ───────────────

#[test]
fn an_operation_on_the_strided_view_matches_the_packed_one() {
    let bytes = padded_plane();
    let pixels: &[Mono8] = Mono8::cast_slice(&bytes).unwrap();
    // The sub-view borrows its parent, so the padded view has to stay bound.
    let padded = ImageRef::new(ROW_STRIDE, HEIGHT, pixels).unwrap();
    let strided = padded
        .roi(Rectangle::new((0, 0), Size::new(WIDTH, HEIGHT)))
        .unwrap();

    // The same image with the padding removed up front.
    let packed = Image::<Mono8>::generate(WIDTH, HEIGHT, |x, y| Mono8::new(sample(x, y)));

    // A neighbourhood operation reaches sideways across rows, so it is the
    // one that would expose a wrong row pitch. Both must agree everywhere,
    // border included.
    let from_strided: Image<Mono8> = gaussian_blur_3x3(&strided, &Clamp);
    let from_packed: Image<Mono8> = gaussian_blur_3x3(&packed, &Clamp);

    assert_eq!(from_strided.size(), from_packed.size());
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            assert_eq!(
                from_strided.pixel_at(x, y),
                from_packed.pixel_at(x, y),
                "blur differs at ({x}, {y})"
            );
        }
    }
}

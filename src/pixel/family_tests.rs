//! Macro-generated structural test batteries for all pixel types.
//!
//! Each `test_pixel_family!` invocation generates ~19 tests that verify
//! PlainPixel and HomogeneousPixel invariants.

use super::*;
use std::num::Saturating;

macro_rules! test_pixel_family {
    ($mod_name:ident, $Pixel:ty, name: $name:expr,
     sample: $sample:expr, different: $diff:expr,
     channels: [$($ch:expr),+ $(,)?], other_ch: $other_ch:expr,
     align: $align:expr) => {
        mod $mod_name {
            use super::*;

            #[test]
            fn zero_has_all_zero_bytes() {
                let z = <$Pixel as ZeroablePixel>::zero();
                for &b in z.as_bytes() {
                    assert_eq!(b, 0, "ZeroablePixel::zero() must produce all-zero bytes");
                }
            }

            #[test]
            fn as_bytes_from_bytes_roundtrip() {
                let sample: $Pixel = $sample;
                let bytes = sample.as_bytes();
                let reconstructed = <$Pixel as PlainChannel>::from_bytes(bytes)
                    .expect("from_bytes should succeed for bytes produced by as_bytes");
                assert_eq!(reconstructed, sample);
            }

            #[test]
            fn as_bytes_le_len() {
                let sample: $Pixel = $sample;
                assert_eq!(
                    sample.as_bytes_le().len(),
                    <$Pixel as PlainChannel>::SIZE,
                    "as_bytes_le length must equal SIZE"
                );
            }

            #[test]
            fn as_bytes_be_len() {
                let sample: $Pixel = $sample;
                assert_eq!(
                    sample.as_bytes_be().len(),
                    <$Pixel as PlainChannel>::SIZE,
                    "as_bytes_be length must equal SIZE"
                );
            }

            #[test]
            fn as_mut_bytes_roundtrip() {
                let sample: $Pixel = $sample;
                let mut pixel = <$Pixel as ZeroablePixel>::zero();
                pixel.as_mut_bytes().copy_from_slice(sample.as_bytes());
                assert_eq!(pixel, sample);
            }

            #[test]
            fn copy_clone_eq() {
                let sample: $Pixel = $sample;
                let copied = sample;
                let cloned = sample.clone();
                assert_eq!(copied, sample);
                assert_eq!(cloned, sample);
            }

            #[test]
            fn ne_different_values() {
                let sample: $Pixel = $sample;
                let different: $Pixel = $diff;
                assert_ne!(sample, different);
            }

            #[test]
            fn debug_contains_type_name() {
                let sample: $Pixel = $sample;
                let s = format!("{:?}", sample);
                assert!(
                    s.contains($name),
                    "Debug output {:?} should contain {:?}",
                    s,
                    $name
                );
            }

            #[test]
            fn align_value() {
                assert_eq!(
                    <$Pixel as PlainChannel>::ALIGN,
                    $align,
                    "ALIGN mismatch"
                );
            }

            #[test]
            fn size_equals_channels_sum() {
                assert_eq!(
                    <$Pixel as PlainChannel>::SIZE,
                    <$Pixel as PlainPixel>::CHANNELS.iter().sum::<usize>(),
                    "SIZE must equal sum of CHANNELS"
                );
            }

            #[test]
            fn uniform_channel_access() {
                let sample: $Pixel = $sample;
                let expected = [$($ch),+];
                for (i, &exp) in expected.iter().enumerate() {
                    assert_eq!(
                        <$Pixel as HomogeneousPixel>::channel(&sample, i),
                        exp,
                        "channel({}) mismatch",
                        i
                    );
                }
            }

            #[test]
            fn uniform_to_channels() {
                let sample: $Pixel = $sample;
                let channels = <$Pixel as HomogeneousPixel>::to_channels(&sample);
                let expected = [$($ch),+];
                let ch_ref: &[<$Pixel as HomogeneousPixel>::Channel] = channels.as_ref();
                assert_eq!(ch_ref, &expected[..]);
            }

            #[test]
            fn uniform_from_channels_roundtrip() {
                let sample: $Pixel = $sample;
                let channels = <$Pixel as HomogeneousPixel>::to_channels(&sample);
                let ch_slice: &[<$Pixel as HomogeneousPixel>::Channel] = channels.as_ref();
                let reconstructed = <$Pixel as HomogeneousPixel>::from_channels(ch_slice);
                assert_eq!(reconstructed, sample);
            }

            #[test]
            fn uniform_set_channel() {
                let sample: $Pixel = $sample;
                let mut pixel = sample;
                let other_ch: <$Pixel as HomogeneousPixel>::Channel = $other_ch;
                <$Pixel as HomogeneousPixel>::set_channel(&mut pixel, 0, other_ch);
                assert_eq!(
                    <$Pixel as HomogeneousPixel>::channel(&pixel, 0),
                    other_ch
                );
            }

            #[test]
            fn uniform_size_assert() {
                assert_eq!(
                    <$Pixel as PlainChannel>::SIZE,
                    <$Pixel as HomogeneousPixel>::CHANNEL_COUNT
                        * core::mem::size_of::<<$Pixel as HomogeneousPixel>::Channel>(),
                    "SIZE must equal CHANNEL_COUNT * size_of::<Channel>()"
                );
            }

            #[test]
            fn uniform_dim_equals_channel_count() {
                assert_eq!(
                    <$Pixel as PlainPixel>::DIM,
                    <$Pixel as HomogeneousPixel>::CHANNEL_COUNT,
                    "DIM must equal CHANNEL_COUNT"
                );
            }

            #[test]
            #[should_panic]
            fn uniform_channel_out_of_bounds() {
                let sample: $Pixel = $sample;
                let _ = <$Pixel as HomogeneousPixel>::channel(
                    &sample,
                    <$Pixel as HomogeneousPixel>::CHANNEL_COUNT,
                );
            }

            #[test]
            #[should_panic]
            fn uniform_set_channel_out_of_bounds() {
                let mut sample: $Pixel = $sample;
                let other_ch: <$Pixel as HomogeneousPixel>::Channel = $other_ch;
                <$Pixel as HomogeneousPixel>::set_channel(
                    &mut sample,
                    <$Pixel as HomogeneousPixel>::CHANNEL_COUNT,
                    other_ch,
                );
            }

            #[test]
            #[should_panic]
            fn uniform_from_channels_wrong_count() {
                let sample: $Pixel = $sample;
                let channels = <$Pixel as HomogeneousPixel>::to_channels(&sample);
                let ch_slice: &[<$Pixel as HomogeneousPixel>::Channel] = channels.as_ref();
                let mut vec = ch_slice.to_vec();
                vec.push($other_ch);
                let _ = <$Pixel as HomogeneousPixel>::from_channels(&vec);
            }
        }
    };
}

// ── Mono integer ────────────────────────────────────────────────────────────

test_pixel_family!(mono8, Mono8, name: "Mono8",
    sample: Mono8::new(42), different: Mono8::new(100),
    channels: [Saturating(42u8)], other_ch: Saturating(99u8), align: 1);

test_pixel_family!(mono10, Mono10, name: "Mono",
    sample: Mono10::new(500), different: Mono10::new(100),
    channels: [Saturating(500u16)], other_ch: Saturating(99u16), align: 2);

test_pixel_family!(mono12, Mono12, name: "Mono",
    sample: Mono12::new(2000), different: Mono12::new(100),
    channels: [Saturating(2000u16)], other_ch: Saturating(99u16), align: 2);

test_pixel_family!(mono14, Mono14, name: "Mono",
    sample: Mono14::new(8000), different: Mono14::new(100),
    channels: [Saturating(8000u16)], other_ch: Saturating(99u16), align: 2);

test_pixel_family!(mono16, Mono16, name: "Mono16",
    sample: Mono16::new(12345), different: Mono16::new(100),
    channels: [Saturating(12345u16)], other_ch: Saturating(99u16), align: 2);

test_pixel_family!(mono32, Mono32, name: "Mono32",
    sample: Mono32::new(100000), different: Mono32::new(42),
    channels: [Saturating(100000u32)], other_ch: Saturating(99u32), align: 4);

test_pixel_family!(mono64, Mono64, name: "Mono64",
    sample: Mono64::new(10000000000), different: Mono64::new(42),
    channels: [Saturating(10000000000u64)], other_ch: Saturating(99u64), align: 8);

// ── Mono float ──────────────────────────────────────────────────────────────

test_pixel_family!(monof32, MonoF32, name: "MonoF32",
    sample: MonoF32::new(0.42), different: MonoF32::new(0.99),
    channels: [0.42f32], other_ch: 0.99f32, align: 4);

test_pixel_family!(monof64, MonoF64, name: "MonoF64",
    sample: MonoF64::new(0.42), different: MonoF64::new(0.99),
    channels: [0.42f64], other_ch: 0.99f64, align: 8);

// ── MonoA integer ───────────────────────────────────────────────────────────

test_pixel_family!(monoa8, MonoA8, name: "MonoA8",
    sample: MonoA8::new(100, 200), different: MonoA8::new(50, 60),
    channels: [Saturating(100u8), Saturating(200u8)], other_ch: Saturating(99u8), align: 1);

test_pixel_family!(monoa16, MonoA16, name: "MonoA16",
    sample: MonoA16::new(1000, 2000), different: MonoA16::new(50, 60),
    channels: [Saturating(1000u16), Saturating(2000u16)], other_ch: Saturating(99u16), align: 2);

test_pixel_family!(monoa32, MonoA32, name: "MonoA32",
    sample: MonoA32::new(100000, 200000), different: MonoA32::new(50, 60),
    channels: [Saturating(100000u32), Saturating(200000u32)], other_ch: Saturating(99u32), align: 4);

test_pixel_family!(monoa64, MonoA64, name: "MonoA64",
    sample: MonoA64::new(1000000000, 2000000000), different: MonoA64::new(50, 60),
    channels: [Saturating(1000000000u64), Saturating(2000000000u64)], other_ch: Saturating(99u64), align: 8);

// ── MonoA float ─────────────────────────────────────────────────────────────

test_pixel_family!(monoaf32, MonoAF32, name: "MonoAF32",
    sample: MonoAF32::new(0.5, 0.8), different: MonoAF32::new(0.1, 0.2),
    channels: [0.5f32, 0.8f32], other_ch: 0.99f32, align: 4);

test_pixel_family!(monoaf64, MonoAF64, name: "MonoAF64",
    sample: MonoAF64::new(0.5, 0.8), different: MonoAF64::new(0.1, 0.2),
    channels: [0.5f64, 0.8f64], other_ch: 0.99f64, align: 8);

// ── RGB integer ─────────────────────────────────────────────────────────────

test_pixel_family!(rgb8, Rgb8, name: "Rgb8",
    sample: Rgb8::new(10, 20, 30), different: Rgb8::new(100, 200, 150),
    channels: [Saturating(10u8), Saturating(20u8), Saturating(30u8)], other_ch: Saturating(99u8), align: 1);

test_pixel_family!(rgb16, Rgb16, name: "Rgb16",
    sample: Rgb16::new(1000, 2000, 3000), different: Rgb16::new(100, 200, 300),
    channels: [Saturating(1000u16), Saturating(2000u16), Saturating(3000u16)], other_ch: Saturating(99u16), align: 2);

test_pixel_family!(rgb32, Rgb32, name: "Rgb32",
    sample: Rgb32::new(100000, 200000, 300000), different: Rgb32::new(1, 2, 3),
    channels: [Saturating(100000u32), Saturating(200000u32), Saturating(300000u32)], other_ch: Saturating(99u32), align: 4);

test_pixel_family!(rgb64, Rgb64, name: "Rgb64",
    sample: Rgb64::new(10000000000, 20000000000, 30000000000), different: Rgb64::new(1, 2, 3),
    channels: [Saturating(10000000000u64), Saturating(20000000000u64), Saturating(30000000000u64)], other_ch: Saturating(99u64), align: 8);

// ── RGB Mono<N> channels ────────────────────────────────────────────────────

test_pixel_family!(rgb10, Rgb10, name: "Rgb",
    sample: Rgb10::new(100, 200, 300), different: Rgb10::new(1, 2, 3),
    channels: [Mono10::new(100), Mono10::new(200), Mono10::new(300)], other_ch: Mono10::new(99), align: 2);

test_pixel_family!(rgb12, Rgb12, name: "Rgb",
    sample: Rgb12::new(1000, 2000, 3000), different: Rgb12::new(1, 2, 3),
    channels: [Mono12::new(1000), Mono12::new(2000), Mono12::new(3000)], other_ch: Mono12::new(99), align: 2);

test_pixel_family!(rgb14, Rgb14, name: "Rgb",
    sample: Rgb14::new(1000, 2000, 3000), different: Rgb14::new(1, 2, 3),
    channels: [Mono14::new(1000), Mono14::new(2000), Mono14::new(3000)], other_ch: Mono14::new(99), align: 2);

// ── RGB float ───────────────────────────────────────────────────────────────

test_pixel_family!(rgbf32, RgbF32, name: "RgbF32",
    sample: RgbF32::new(0.25, 0.5, 0.75), different: RgbF32::new(0.1, 0.2, 0.3),
    channels: [0.25f32, 0.5f32, 0.75f32], other_ch: 0.99f32, align: 4);

test_pixel_family!(rgbf64, RgbF64, name: "RgbF64",
    sample: RgbF64::new(0.1, 0.2, 0.3), different: RgbF64::new(0.4, 0.5, 0.6),
    channels: [0.1f64, 0.2f64, 0.3f64], other_ch: 0.99f64, align: 8);

// ── RGBA integer ────────────────────────────────────────────────────────────

test_pixel_family!(rgba8, Rgba8, name: "Rgba8",
    sample: Rgba8::new(10, 20, 30, 40), different: Rgba8::new(100, 200, 150, 250),
    channels: [Saturating(10u8), Saturating(20u8), Saturating(30u8), Saturating(40u8)], other_ch: Saturating(99u8), align: 1);

test_pixel_family!(rgba16, Rgba16, name: "Rgba16",
    sample: Rgba16::new(1000, 2000, 3000, 4000), different: Rgba16::new(100, 200, 300, 400),
    channels: [Saturating(1000u16), Saturating(2000u16), Saturating(3000u16), Saturating(4000u16)], other_ch: Saturating(99u16), align: 2);

test_pixel_family!(rgba32, Rgba32, name: "Rgba32",
    sample: Rgba32::new(100000, 200000, 300000, 400000), different: Rgba32::new(1, 2, 3, 4),
    channels: [Saturating(100000u32), Saturating(200000u32), Saturating(300000u32), Saturating(400000u32)], other_ch: Saturating(99u32), align: 4);

test_pixel_family!(rgba64, Rgba64, name: "Rgba64",
    sample: Rgba64::new(10000000000, 20000000000, 30000000000, 40000000000), different: Rgba64::new(1, 2, 3, 4),
    channels: [Saturating(10000000000u64), Saturating(20000000000u64), Saturating(30000000000u64), Saturating(40000000000u64)], other_ch: Saturating(99u64), align: 8);

// ── RGBA Mono<N> channels ───────────────────────────────────────────────────

test_pixel_family!(rgba10, Rgba10, name: "Rgba",
    sample: Rgba10::new(100, 200, 300, 400), different: Rgba10::new(1, 2, 3, 4),
    channels: [Mono10::new(100), Mono10::new(200), Mono10::new(300), Mono10::new(400)], other_ch: Mono10::new(99), align: 2);

test_pixel_family!(rgba12, Rgba12, name: "Rgba",
    sample: Rgba12::new(1000, 2000, 3000, 500), different: Rgba12::new(1, 2, 3, 4),
    channels: [Mono12::new(1000), Mono12::new(2000), Mono12::new(3000), Mono12::new(500)], other_ch: Mono12::new(99), align: 2);

test_pixel_family!(rgba14, Rgba14, name: "Rgba",
    sample: Rgba14::new(1000, 2000, 3000, 500), different: Rgba14::new(1, 2, 3, 4),
    channels: [Mono14::new(1000), Mono14::new(2000), Mono14::new(3000), Mono14::new(500)], other_ch: Mono14::new(99), align: 2);

// ── RGBA float ──────────────────────────────────────────────────────────────

test_pixel_family!(rgbaf32, RgbaF32, name: "RgbaF32",
    sample: RgbaF32::new(0.25, 0.5, 0.75, 0.9), different: RgbaF32::new(0.1, 0.2, 0.3, 0.4),
    channels: [0.25f32, 0.5f32, 0.75f32, 0.9f32], other_ch: 0.99f32, align: 4);

test_pixel_family!(rgbaf64, RgbaF64, name: "RgbaF64",
    sample: RgbaF64::new(0.1, 0.2, 0.3, 0.4), different: RgbaF64::new(0.5, 0.6, 0.7, 0.8),
    channels: [0.1f64, 0.2f64, 0.3f64, 0.4f64], other_ch: 0.99f64, align: 8);

// ── BGR integer ─────────────────────────────────────────────────────────────

test_pixel_family!(bgr8, Bgr8, name: "Bgr8",
    sample: Bgr8::new(10, 20, 30), different: Bgr8::new(100, 200, 150),
    channels: [Saturating(10u8), Saturating(20u8), Saturating(30u8)], other_ch: Saturating(99u8), align: 1);

test_pixel_family!(bgr16, Bgr16, name: "Bgr16",
    sample: Bgr16::new(1000, 2000, 3000), different: Bgr16::new(100, 200, 300),
    channels: [Saturating(1000u16), Saturating(2000u16), Saturating(3000u16)], other_ch: Saturating(99u16), align: 2);

test_pixel_family!(bgr32, Bgr32, name: "Bgr32",
    sample: Bgr32::new(100000, 200000, 300000), different: Bgr32::new(1, 2, 3),
    channels: [Saturating(100000u32), Saturating(200000u32), Saturating(300000u32)], other_ch: Saturating(99u32), align: 4);

test_pixel_family!(bgr64, Bgr64, name: "Bgr64",
    sample: Bgr64::new(10000000000, 20000000000, 30000000000), different: Bgr64::new(1, 2, 3),
    channels: [Saturating(10000000000u64), Saturating(20000000000u64), Saturating(30000000000u64)], other_ch: Saturating(99u64), align: 8);

// ── BGR Mono<N> channels ────────────────────────────────────────────────────

test_pixel_family!(bgr10, Bgr10, name: "Bgr",
    sample: Bgr10::new(100, 200, 300), different: Bgr10::new(1, 2, 3),
    channels: [Mono10::new(100), Mono10::new(200), Mono10::new(300)], other_ch: Mono10::new(99), align: 2);

test_pixel_family!(bgr12, Bgr12, name: "Bgr",
    sample: Bgr12::new(1000, 2000, 3000), different: Bgr12::new(1, 2, 3),
    channels: [Mono12::new(1000), Mono12::new(2000), Mono12::new(3000)], other_ch: Mono12::new(99), align: 2);

test_pixel_family!(bgr14, Bgr14, name: "Bgr",
    sample: Bgr14::new(1000, 2000, 3000), different: Bgr14::new(1, 2, 3),
    channels: [Mono14::new(1000), Mono14::new(2000), Mono14::new(3000)], other_ch: Mono14::new(99), align: 2);

// ── BGR float ───────────────────────────────────────────────────────────────

test_pixel_family!(bgrf32, BgrF32, name: "BgrF32",
    sample: BgrF32::new(0.25, 0.5, 0.75), different: BgrF32::new(0.1, 0.2, 0.3),
    channels: [0.25f32, 0.5f32, 0.75f32], other_ch: 0.99f32, align: 4);

test_pixel_family!(bgrf64, BgrF64, name: "BgrF64",
    sample: BgrF64::new(0.1, 0.2, 0.3), different: BgrF64::new(0.4, 0.5, 0.6),
    channels: [0.1f64, 0.2f64, 0.3f64], other_ch: 0.99f64, align: 8);

// ── BGRA integer ────────────────────────────────────────────────────────────

test_pixel_family!(bgra8, Bgra8, name: "Bgra8",
    sample: Bgra8::new(10, 20, 30, 40), different: Bgra8::new(100, 200, 150, 250),
    channels: [Saturating(10u8), Saturating(20u8), Saturating(30u8), Saturating(40u8)], other_ch: Saturating(99u8), align: 1);

test_pixel_family!(bgra16, Bgra16, name: "Bgra16",
    sample: Bgra16::new(1000, 2000, 3000, 4000), different: Bgra16::new(100, 200, 300, 400),
    channels: [Saturating(1000u16), Saturating(2000u16), Saturating(3000u16), Saturating(4000u16)], other_ch: Saturating(99u16), align: 2);

test_pixel_family!(bgra32, Bgra32, name: "Bgra32",
    sample: Bgra32::new(100000, 200000, 300000, 400000), different: Bgra32::new(1, 2, 3, 4),
    channels: [Saturating(100000u32), Saturating(200000u32), Saturating(300000u32), Saturating(400000u32)], other_ch: Saturating(99u32), align: 4);

test_pixel_family!(bgra64, Bgra64, name: "Bgra64",
    sample: Bgra64::new(10000000000, 20000000000, 30000000000, 40000000000), different: Bgra64::new(1, 2, 3, 4),
    channels: [Saturating(10000000000u64), Saturating(20000000000u64), Saturating(30000000000u64), Saturating(40000000000u64)], other_ch: Saturating(99u64), align: 8);

// ── BGRA Mono<N> channels ───────────────────────────────────────────────────

test_pixel_family!(bgra10, Bgra10, name: "Bgra",
    sample: Bgra10::new(100, 200, 300, 400), different: Bgra10::new(1, 2, 3, 4),
    channels: [Mono10::new(100), Mono10::new(200), Mono10::new(300), Mono10::new(400)], other_ch: Mono10::new(99), align: 2);

test_pixel_family!(bgra12, Bgra12, name: "Bgra",
    sample: Bgra12::new(1000, 2000, 3000, 500), different: Bgra12::new(1, 2, 3, 4),
    channels: [Mono12::new(1000), Mono12::new(2000), Mono12::new(3000), Mono12::new(500)], other_ch: Mono12::new(99), align: 2);

test_pixel_family!(bgra14, Bgra14, name: "Bgra",
    sample: Bgra14::new(1000, 2000, 3000, 500), different: Bgra14::new(1, 2, 3, 4),
    channels: [Mono14::new(1000), Mono14::new(2000), Mono14::new(3000), Mono14::new(500)], other_ch: Mono14::new(99), align: 2);

// ── BGRA float ──────────────────────────────────────────────────────────────

test_pixel_family!(bgraf32, BgraF32, name: "BgraF32",
    sample: BgraF32::new(0.25, 0.5, 0.75, 0.9), different: BgraF32::new(0.1, 0.2, 0.3, 0.4),
    channels: [0.25f32, 0.5f32, 0.75f32, 0.9f32], other_ch: 0.99f32, align: 4);

test_pixel_family!(bgraf64, BgraF64, name: "BgraF64",
    sample: BgraF64::new(0.1, 0.2, 0.3, 0.4), different: BgraF64::new(0.5, 0.6, 0.7, 0.8),
    channels: [0.1f64, 0.2f64, 0.3f64, 0.4f64], other_ch: 0.99f64, align: 8);

// ── sRGB ────────────────────────────────────────────────────────────────────

test_pixel_family!(srgb8, Srgb8, name: "Srgb8",
    sample: Srgb8::new(100, 150, 200), different: Srgb8::new(10, 20, 30),
    channels: [Saturating(100u8), Saturating(150u8), Saturating(200u8)], other_ch: Saturating(99u8), align: 1);

test_pixel_family!(srgba8, Srgba8, name: "Srgba8",
    sample: Srgba8::new(100, 150, 200, 220), different: Srgba8::new(10, 20, 30, 40),
    channels: [Saturating(100u8), Saturating(150u8), Saturating(200u8), Saturating(220u8)], other_ch: Saturating(99u8), align: 1);

test_pixel_family!(srgb_mono8, SrgbMono8, name: "SrgbMono8",
    sample: SrgbMono8::new(42), different: SrgbMono8::new(100),
    channels: [Saturating(42u8)], other_ch: Saturating(99u8), align: 1);

test_pixel_family!(srgb_mono_a8, SrgbMonoA8, name: "SrgbMonoA8",
    sample: SrgbMonoA8::new(100, 200), different: SrgbMonoA8::new(10, 20),
    channels: [Saturating(100u8), Saturating(200u8)], other_ch: Saturating(99u8), align: 1);

test_pixel_family!(srgb16, Srgb16, name: "Srgb16",
    sample: Srgb16::new(1000, 2000, 3000), different: Srgb16::new(100, 200, 300),
    channels: [Saturating(1000u16), Saturating(2000u16), Saturating(3000u16)], other_ch: Saturating(99u16), align: 2);

test_pixel_family!(srgba16, Srgba16, name: "Srgba16",
    sample: Srgba16::new(1000, 2000, 3000, 4000), different: Srgba16::new(100, 200, 300, 400),
    channels: [Saturating(1000u16), Saturating(2000u16), Saturating(3000u16), Saturating(4000u16)], other_ch: Saturating(99u16), align: 2);

test_pixel_family!(srgb_mono16, SrgbMono16, name: "SrgbMono16",
    sample: SrgbMono16::new(12345), different: SrgbMono16::new(100),
    channels: [Saturating(12345u16)], other_ch: Saturating(99u16), align: 2);

test_pixel_family!(srgb_mono_a16, SrgbMonoA16, name: "SrgbMonoA16",
    sample: SrgbMonoA16::new(1000, 2000), different: SrgbMonoA16::new(100, 200),
    channels: [Saturating(1000u16), Saturating(2000u16)], other_ch: Saturating(99u16), align: 2);

// ── Indexed ─────────────────────────────────────────────────────────────────

test_pixel_family!(indexed8, Indexed8, name: "Indexed8",
    sample: Indexed8(42), different: Indexed8(100),
    channels: [42u8], other_ch: 99u8, align: 1);

// ── OriginInvariantPixel coverage (ADR-0051) ──────────────────────────────────
//
// Compile-time proof that every shipped pixel family implements the
// `OriginInvariantPixel` marker, keeping ordinary `SubView` ROI / tiling /
// sliding windows available for them. If a family ever loses its impl — or a
// new family forgets to add one — this fails to compile.
//
// `bool` is included because `BinaryImage = Image<bool>` relies on the marker
// for ROI. Raw channel primitives (`u8`, `u16`, `f32`, …) are deliberately
// absent — they are channels, not pixels (Philosophy §9) — and a
// coordinate-dependent Bayer CFA pixel would be absent too (ADR-0037).
#[test]
fn origin_invariant_marker_covers_all_families() {
    fn assert_marker<P: OriginInvariantPixel>() {}

    macro_rules! assert_all {
        ($($t:ty),+ $(,)?) => {{ $( assert_marker::<$t>(); )+ }};
    }

    assert_all!(
        Mono8, Mono16, Mono32, Mono64, MonoF32, MonoF64, Mono10, Mono12, Mono14,
        MonoA8, MonoA16, MonoA32, MonoA64, MonoAF32, MonoAF64,
        Rgb8, Rgb16, Rgb32, Rgb64, RgbF32, RgbF64, Rgb10, Rgb12, Rgb14,
        Rgba8, Rgba16, Rgba32, Rgba64, RgbaF32, RgbaF64, Rgba10, Rgba12, Rgba14,
        Bgr8, Bgr16, Bgr32, Bgr64, BgrF32, BgrF64, Bgr10, Bgr12, Bgr14,
        Bgra8, Bgra16, Bgra32, Bgra64, BgraF32, BgraF64, Bgra10, Bgra12, Bgra14,
        Srgb8, Srgba8, SrgbMono8, SrgbMonoA8, Srgb16, Srgba16, SrgbMono16, SrgbMonoA16,
        Indexed8, Label32, bool,
    );
}

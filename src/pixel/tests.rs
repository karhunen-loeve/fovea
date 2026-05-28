use super::*;
use std::hash::{Hash, Hasher};
use std::num::Saturating;

#[test]
fn test_mono_add() {
    assert_eq!(Mono::<10>::MAX, 1023);
    assert_eq!(Mono::<10>::new(2048), Mono::<10>::new(1023));

    let a = Mono::<10>::new(512);
    let b = Mono::<10>::new(600);
    let c = a + b;
    assert_eq!(c, Mono::<10>::new(1023));

    let mut d = Mono::<10>::new(400);
    d += Mono::<10>::new(700);
    assert_eq!(d, Mono::<10>::new(1023));

    let mut e = Mono::<10>::new(400);
    e += &700;
    assert_eq!(e, Mono::<10>::new(1023));
}

#[test]
fn test_mono_sub() {
    let a = Mono::<10>::new(800);
    let b = Mono::<10>::new(600);
    let c = a - b;
    assert_eq!(c, Mono::<10>::new(200));

    let mut d = Mono::<10>::new(400);
    d -= Mono::<10>::new(500);
    assert_eq!(d, Mono::<10>::new(0));

    let mut e = Mono::<10>::new(400);
    e -= &500;
    assert_eq!(e, Mono::<10>::new(0));
}

#[test]
fn test_mono_mul() {
    let a = Mono::<10>::new(32);
    let b = Mono::<10>::new(16);
    let c = a * b;
    assert_eq!(c, Mono::<10>::new(512));

    let mut d = Mono::<10>::new(32);
    d *= Mono::<10>::new(40);
    assert_eq!(d, Mono::<10>::new(1023));

    let mut e = Mono::<10>::new(32);
    e *= &40;
    assert_eq!(e, Mono::<10>::new(1023));
}

#[test]
fn test_mono_mul_overflow_mul_assign_u16() {
    // Regression: MulAssign<&u16> previously used raw u16 arithmetic which
    // panicked in debug builds and wrapped silently in release builds.
    // Mono10 MAX = 1023. 1000 * 1000 = 1_000_000 overflows u16.
    let mut m = Mono::<10>::new(1000);
    m *= &1000u16;
    assert_eq!(m.value(), 1023); // must clamp to MAX, not panic or wrap
}

#[test]
fn test_mono_mul_overflow_mono_times_mono() {
    // Mono12 MAX = 4095. 4095 * 4095 = 16_769_025 overflows u16.
    let a = Mono::<12>::new(4095);
    let b = Mono::<12>::new(4095);
    let c = a * b;
    assert_eq!(c.value(), 4095); // clamped to MAX

    // Same via reference
    let d = &a * &b;
    assert_eq!(d.value(), 4095);
}

#[test]
fn test_mono_mul_overflow_mul_assign_mono() {
    // Mono14 MAX = 16383. 16383 * 16383 = 268_402_689 overflows u16.
    let mut m = Mono::<14>::new(16383);
    m *= Mono::<14>::new(16383);
    assert_eq!(m.value(), 16383); // clamped to MAX

    let mut m2 = Mono::<14>::new(16383);
    m2 *= &Mono::<14>::new(16383);
    assert_eq!(m2.value(), 16383);
}

#[test]
fn test_mono_mul_no_overflow_small_values() {
    // Verify small multiplications still produce exact results.
    let a = Mono::<10>::new(10);
    let b = Mono::<10>::new(5);
    assert_eq!((a * b).value(), 50);

    let mut c = Mono::<12>::new(100);
    c *= &3u16;
    assert_eq!(c.value(), 300);
}

#[test]
fn test_mono_mul_overflow_just_above_max() {
    // Product exceeds MAX but fits in u16: should clamp to MAX.
    // Mono10 MAX = 1023. 32 * 40 = 1280 > 1023 but < 65535.
    let mut m = Mono::<10>::new(32);
    m *= Mono::<10>::new(40);
    assert_eq!(m.value(), 1023);
}

#[test]
fn test_mono_div() {
    let a = Mono::<10>::new(512);
    let b = Mono::<10>::new(16);
    let c = a / b;
    assert_eq!(c, Mono::<10>::new(32));

    let mut d = Mono::<10>::new(512);
    d /= Mono::<10>::new(2);
    assert_eq!(d, Mono::<10>::new(256));

    let mut e = Mono::<10>::new(512);
    e /= &2;
    assert_eq!(e, Mono::<10>::new(256));
}

#[test]
fn test_rgb8_zero() {
    assert_eq!(
        Rgb8::zero(),
        Rgb8 {
            r: Saturating(0),
            g: Saturating(0),
            b: Saturating(0)
        }
    );
}

macro_rules! test_pixel_size {
    ($name:ident, $pixel:ty) => {
        #[test]
        fn $name() {
            assert_eq!(
                size_of::<$pixel>(),
                <$pixel as PlainPixel>::CHANNELS.iter().sum::<usize>()
            );
        }
    };
}

test_pixel_size!(test_size_u8, u8);
test_pixel_size!(test_size_i8, i8);
test_pixel_size!(test_size_u16, u16);
test_pixel_size!(test_size_i16, i16);
test_pixel_size!(test_size_u32, u32);
test_pixel_size!(test_size_i32, i32);
test_pixel_size!(test_size_u64, u64);
test_pixel_size!(test_size_i64, i64);
// `test_size_f32` / `test_size_f64` removed (ADR-0044 Phase E):
// `f32` / `f64` are no longer pixels. The equivalent coverage now
// lives on `MonoF32` / `MonoF64` via the `family_tests.rs` battery.
test_pixel_size!(test_size_mono8, Mono8);
test_pixel_size!(test_size_mono10, Mono10);
test_pixel_size!(test_size_mono12, Mono12);
test_pixel_size!(test_size_mono14, Mono14);
test_pixel_size!(test_size_mono16, Mono16);
test_pixel_size!(test_size_mono32, Mono32);
test_pixel_size!(test_size_mono64, Mono64);
test_pixel_size!(test_size_rgb8, Rgb8);
test_pixel_size!(test_size_rgba8, Rgba8);
test_pixel_size!(test_size_rgb10, Rgb10);
test_pixel_size!(test_size_rgba10, Rgba10);
test_pixel_size!(test_size_rgb12, Rgb12);
test_pixel_size!(test_size_rgba12, Rgba12);
test_pixel_size!(test_size_rgb16, Rgb16);
test_pixel_size!(test_size_rgba16, Rgba16);
test_pixel_size!(test_size_rgb32, Rgb32);
test_pixel_size!(test_size_rgba32, Rgba32);
test_pixel_size!(test_size_rgb64, Rgb64);
test_pixel_size!(test_size_rgba64, Rgba64);
test_pixel_size!(test_size_rgbf32, RgbF32);
test_pixel_size!(test_size_rgbf64, RgbF64);
test_pixel_size!(test_size_rgbaf32, RgbaF32);
test_pixel_size!(test_size_rgbaf64, RgbaF64);
test_pixel_size!(test_size_bgr8, Bgr8);
test_pixel_size!(test_size_bgr10, Bgr10);
test_pixel_size!(test_size_bgr12, Bgr12);
test_pixel_size!(test_size_bgr16, Bgr16);
test_pixel_size!(test_size_bgr32, Bgr32);
test_pixel_size!(test_size_bgr64, Bgr64);
test_pixel_size!(test_size_bgra8, Bgra8);
test_pixel_size!(test_size_bgra10, Bgra10);
test_pixel_size!(test_size_bgra12, Bgra12);
test_pixel_size!(test_size_bgra16, Bgra16);
test_pixel_size!(test_size_bgra32, Bgra32);
test_pixel_size!(test_size_bgra64, Bgra64);
test_pixel_size!(test_size_bgrf32, BgrF32);
test_pixel_size!(test_size_bgrf64, BgrF64);
test_pixel_size!(test_size_bgraf32, BgraF32);
test_pixel_size!(test_size_bgraf64, BgraF64);
test_pixel_size!(test_size_indexed8, Indexed8);

#[test]
fn test_mono_as_bytes() {
    let m = Mono::<10>::new(512);
    let bytes = m.as_bytes();
    assert_eq!(bytes, &[0, 2]);
    let bytes_le = m.as_bytes_le();
    assert_eq!(&*bytes_le, &[0, 2]);
    let bytes_be = m.as_bytes_be();
    assert_eq!(&*bytes_be, &[2, 0]);
}

#[test]
fn does_not_compile_for_strange_mono_pixel() {
    // The following line should not compile, uncomment to test
    // let _ = Mono::<11>::new(0);
    // let _ = Mono::<20>::new(0);
}

#[test]
fn test_mono10_from_u16() {
    let m: Mono<10> = 512u16.into();
    assert_eq!(m, Mono::<10>::new(512));
    let v: u16 = m.into();
    assert_eq!(v, 512u16);
}

#[test]
fn test_mono10_from_saturating_u16() {
    let m: Mono<10> = Saturating(512u16).into();
    assert_eq!(m, Mono::<10>::new(512));
    let v: Saturating<u16> = m.into();
    assert_eq!(v, Saturating(512u16));
}

#[test]
fn test_generic_rgb_pixel() {
    let r: Rgb<10> = Rgb::new(2000, 256, 128);
    assert_eq!(r.r, Mono::<10>::new(1023));
    assert_eq!(r.g, Mono::<10>::new(256));
    assert_eq!(r.b, Mono::<10>::new(128));
}

#[test]
fn test_mono12_operations() {
    let a = Mono::<12>::new(2048);
    let b = Mono::<12>::new(1000);
    assert_eq!(a.value(), 2048);
    assert_eq!(b.value(), 1000);

    let c = a + b;
    assert_eq!(c.value(), 3048);

    let d = a - b;
    assert_eq!(d.value(), 1048);
}

#[test]
fn test_mono14_operations() {
    let a = Mono::<14>::new(8192);
    let b = Mono::<14>::new(4000);
    assert_eq!(a.value(), 8192);

    let c = a + b;
    assert_eq!(c.value(), 12192);
}

#[test]
fn test_mono_zero() {
    assert_eq!(Mono8::zero(), Mono8::new(0));
    assert_eq!(Mono10::zero(), Mono10::new(0));
    assert_eq!(Mono12::zero(), Mono12::new(0));
    assert_eq!(Mono14::zero(), Mono14::new(0));
    assert_eq!(Mono16::zero(), Mono16::new(0));
    assert_eq!(Mono32::zero(), Mono32::new(0));
    assert_eq!(Mono64::zero(), Mono64::new(0));
}

#[test]
fn test_rgb_zero() {
    assert_eq!(Rgb10::zero().r.value(), 0);
    assert_eq!(Rgb10::zero().g.value(), 0);
    assert_eq!(Rgb10::zero().b.value(), 0);

    assert_eq!(Rgb12::zero().r.value(), 0);
    assert_eq!(Rgb16::zero().r, Saturating(0));
    assert_eq!(Rgb32::zero().r, Saturating(0));
    assert_eq!(Rgb64::zero().r, Saturating(0));
}

#[test]
fn test_rgba_zero() {
    let rgba8 = Rgba8::zero();
    assert_eq!(rgba8.r, Saturating(0));
    assert_eq!(rgba8.g, Saturating(0));
    assert_eq!(rgba8.b, Saturating(0));
    assert_eq!(rgba8.a, Saturating(0));

    let rgba16 = Rgba16::zero();
    assert_eq!(rgba16.r, Saturating(0));
    assert_eq!(rgba16.a, Saturating(0));
}

#[test]
fn test_bgr_zero() {
    let bgr8 = Bgr8::zero();
    assert_eq!(bgr8.b, Saturating(0));
    assert_eq!(bgr8.g, Saturating(0));
    assert_eq!(bgr8.r, Saturating(0));

    assert_eq!(Bgr10::zero().b.value(), 0);
    assert_eq!(Bgr12::zero().b.value(), 0);
    assert_eq!(Bgr16::zero().b, Saturating(0));
    assert_eq!(Bgr32::zero().b, Saturating(0));
    assert_eq!(Bgr64::zero().b, Saturating(0));
}

#[test]
fn test_bgra_zero() {
    let bgra8 = Bgra8::zero();
    assert_eq!(bgra8.b, Saturating(0));
    assert_eq!(bgra8.g, Saturating(0));
    assert_eq!(bgra8.r, Saturating(0));
    assert_eq!(bgra8.a, Saturating(0));

    let bgra16 = Bgra16::zero();
    assert_eq!(bgra16.b, Saturating(0));
    assert_eq!(bgra16.a, Saturating(0));
}

#[test]
fn test_rgb_float_zero() {
    let rgbf32 = RgbF32::zero();
    assert_eq!(rgbf32.r, 0.0);
    assert_eq!(rgbf32.g, 0.0);
    assert_eq!(rgbf32.b, 0.0);

    let rgbf64 = RgbF64::zero();
    assert_eq!(rgbf64.r, 0.0);
    assert_eq!(rgbf64.g, 0.0);
    assert_eq!(rgbf64.b, 0.0);
}

#[test]
fn test_rgba_float_zero() {
    let rgbaf32 = RgbaF32::zero();
    assert_eq!(rgbaf32.r, 0.0);
    assert_eq!(rgbaf32.g, 0.0);
    assert_eq!(rgbaf32.b, 0.0);
    assert_eq!(rgbaf32.a, 0.0);

    let rgbaf64 = RgbaF64::zero();
    assert_eq!(rgbaf64.r, 0.0);
    assert_eq!(rgbaf64.a, 0.0);
}

#[test]
fn test_bgr_float_zero() {
    let bgrf32 = BgrF32::zero();
    assert_eq!(bgrf32.b, 0.0);
    assert_eq!(bgrf32.g, 0.0);
    assert_eq!(bgrf32.r, 0.0);

    let bgrf64 = BgrF64::zero();
    assert_eq!(bgrf64.b, 0.0);
    assert_eq!(bgrf64.r, 0.0);
}

#[test]
fn test_bgra_float_zero() {
    let bgraf32 = BgraF32::zero();
    assert_eq!(bgraf32.b, 0.0);
    assert_eq!(bgraf32.g, 0.0);
    assert_eq!(bgraf32.r, 0.0);
    assert_eq!(bgraf32.a, 0.0);

    let bgraf64 = BgraF64::zero();
    assert_eq!(bgraf64.b, 0.0);
    assert_eq!(bgraf64.a, 0.0);
}

#[test]
fn test_primitive_zero() {
    assert_eq!(u8::zero(), 0u8);
    assert_eq!(i8::zero(), 0i8);
    assert_eq!(u16::zero(), 0u16);
    assert_eq!(i16::zero(), 0i16);
    assert_eq!(u32::zero(), 0u32);
    assert_eq!(i32::zero(), 0i32);
    assert_eq!(u64::zero(), 0u64);
    assert_eq!(i64::zero(), 0i64);
    // ADR-0044 Phase E: `f32` / `f64` are no longer `ZeroablePixel`.
}

#[test]
fn test_rgb8_new() {
    let rgb = Rgb8::new(255, 128, 64);
    assert_eq!(rgb.r, Saturating(255));
    assert_eq!(rgb.g, Saturating(128));
    assert_eq!(rgb.b, Saturating(64));
}

#[test]
fn test_rgba8_new() {
    let rgba = Rgba8::new(255, 128, 64, 200);
    assert_eq!(rgba.r, Saturating(255));
    assert_eq!(rgba.g, Saturating(128));
    assert_eq!(rgba.b, Saturating(64));
    assert_eq!(rgba.a, Saturating(200));
}

#[test]
fn test_bgr8_new() {
    let bgr = Bgr8::new(255, 128, 64);
    assert_eq!(bgr.b, Saturating(255));
    assert_eq!(bgr.g, Saturating(128));
    assert_eq!(bgr.r, Saturating(64));
}

#[test]
fn test_bgra8_new() {
    let bgra = Bgra8::new(255, 128, 64, 200);
    assert_eq!(bgra.b, Saturating(255));
    assert_eq!(bgra.g, Saturating(128));
    assert_eq!(bgra.r, Saturating(64));
    assert_eq!(bgra.a, Saturating(200));
}

#[test]
fn test_rgb16_new() {
    let rgb = Rgb16::new(65535, 32768, 16384);
    assert_eq!(rgb.r, Saturating(65535));
    assert_eq!(rgb.g, Saturating(32768));
    assert_eq!(rgb.b, Saturating(16384));
}

#[test]
fn test_rgb_float_new() {
    let rgb = RgbF32::new(1.0, 0.5, 0.25);
    assert_eq!(rgb.r, 1.0);
    assert_eq!(rgb.g, 0.5);
    assert_eq!(rgb.b, 0.25);

    let rgb64 = RgbF64::new(1.0, 0.5, 0.25);
    assert_eq!(rgb64.r, 1.0);
    assert_eq!(rgb64.g, 0.5);
    assert_eq!(rgb64.b, 0.25);
}

#[test]
fn test_rgba_float_new() {
    let rgba = RgbaF32::new(1.0, 0.5, 0.25, 0.8);
    assert_eq!(rgba.r, 1.0);
    assert_eq!(rgba.g, 0.5);
    assert_eq!(rgba.b, 0.25);
    assert_eq!(rgba.a, 0.8);
}

#[test]
fn test_bgr_float_new() {
    let bgr = BgrF32::new(1.0, 0.5, 0.25);
    assert_eq!(bgr.b, 1.0);
    assert_eq!(bgr.g, 0.5);
    assert_eq!(bgr.r, 0.25);
}

#[test]
fn test_bgra_float_new() {
    let bgra = BgraF32::new(1.0, 0.5, 0.25, 0.8);
    assert_eq!(bgra.b, 1.0);
    assert_eq!(bgra.g, 0.5);
    assert_eq!(bgra.r, 0.25);
    assert_eq!(bgra.a, 0.8);
}

#[test]
fn test_mono_add_ref() {
    let a = Mono::<10>::new(400);
    let b = Mono::<10>::new(300);
    let c = &a + &b;
    assert_eq!(c.value(), 700);
}

#[test]
fn test_mono_sub_ref() {
    let a = Mono::<10>::new(800);
    let b = Mono::<10>::new(300);
    let c = &a - &b;
    assert_eq!(c.value(), 500);
}

#[test]
fn test_mono_mul_ref() {
    let a = Mono::<10>::new(10);
    let b = Mono::<10>::new(20);
    let c = &a * &b;
    assert_eq!(c.value(), 200);
}

#[test]
fn test_mono_div_ref() {
    let a = Mono::<10>::new(100);
    let b = Mono::<10>::new(5);
    let c = &a / &b;
    assert_eq!(c.value(), 20);
}

#[test]
fn test_rgb10_new() {
    let rgb = Rgb10::new(1023, 512, 256);
    assert_eq!(rgb.r.value(), 1023);
    assert_eq!(rgb.g.value(), 512);
    assert_eq!(rgb.b.value(), 256);
}

#[test]
fn test_rgb12_new() {
    let rgb = Rgb12::new(4095, 2048, 1024);
    assert_eq!(rgb.r.value(), 4095);
    assert_eq!(rgb.g.value(), 2048);
    assert_eq!(rgb.b.value(), 1024);
}

#[test]
fn test_rgba10_new() {
    let rgba = Rgba10::new(1023, 512, 256, 128);
    assert_eq!(rgba.r.value(), 1023);
    assert_eq!(rgba.g.value(), 512);
    assert_eq!(rgba.b.value(), 256);
    assert_eq!(rgba.a.value(), 128);
}

#[test]
fn test_bgr10_new() {
    let bgr = Bgr10::new(1023, 512, 256);
    assert_eq!(bgr.b.value(), 1023);
    assert_eq!(bgr.g.value(), 512);
    assert_eq!(bgr.r.value(), 256);
}

#[test]
fn test_bgra10_new() {
    let bgra = Bgra10::new(1023, 512, 256, 128);
    assert_eq!(bgra.b.value(), 1023);
    assert_eq!(bgra.g.value(), 512);
    assert_eq!(bgra.r.value(), 256);
    assert_eq!(bgra.a.value(), 128);
}

#[test]
fn test_rgb32_new() {
    let rgb = Rgb32::new(1000000, 500000, 250000);
    assert_eq!(rgb.r, Saturating(1000000));
    assert_eq!(rgb.g, Saturating(500000));
    assert_eq!(rgb.b, Saturating(250000));
}

#[test]
fn test_rgb64_new() {
    let rgb = Rgb64::new(10000000000, 5000000000, 2500000000);
    assert_eq!(rgb.r, Saturating(10000000000));
    assert_eq!(rgb.g, Saturating(5000000000));
    assert_eq!(rgb.b, Saturating(2500000000));
}

#[test]
fn test_bgr16_new() {
    let bgr = Bgr16::new(65535, 32768, 16384);
    assert_eq!(bgr.b, Saturating(65535));
    assert_eq!(bgr.g, Saturating(32768));
    assert_eq!(bgr.r, Saturating(16384));
}

#[test]
fn test_bgr32_new() {
    let bgr = Bgr32::new(1000000, 500000, 250000);
    assert_eq!(bgr.b, Saturating(1000000));
    assert_eq!(bgr.g, Saturating(500000));
    assert_eq!(bgr.r, Saturating(250000));
}

#[test]
fn test_bgr64_new() {
    let bgr = Bgr64::new(10000000000, 5000000000, 2500000000);
    assert_eq!(bgr.b, Saturating(10000000000));
    assert_eq!(bgr.g, Saturating(5000000000));
    assert_eq!(bgr.r, Saturating(2500000000));
}

#[test]
fn test_bgra16_new() {
    let bgra = Bgra16::new(65535, 32768, 16384, 8192);
    assert_eq!(bgra.b, Saturating(65535));
    assert_eq!(bgra.g, Saturating(32768));
    assert_eq!(bgra.r, Saturating(16384));
    assert_eq!(bgra.a, Saturating(8192));
}

#[test]
fn test_mono8_new() {
    let m = Mono8::new(255);
    assert_eq!(m, Mono8::new(255));
}

#[test]
fn test_mono16_new() {
    let m = Mono16::new(65535);
    assert_eq!(m, Mono16::new(65535));
}

#[test]
fn test_mono32_new() {
    let m = Mono32::new(1000000);
    assert_eq!(m, Mono32::new(1000000));
}

#[test]
fn test_mono64_new() {
    let m = Mono64::new(10000000000);
    assert_eq!(m, Mono64::new(10000000000));
}

#[test]
fn test_rgba16_new() {
    let rgba = Rgba16::new(65535, 32768, 16384, 8192);
    assert_eq!(rgba.r, Saturating(65535));
    assert_eq!(rgba.g, Saturating(32768));
    assert_eq!(rgba.b, Saturating(16384));
    assert_eq!(rgba.a, Saturating(8192));
}

#[test]
fn test_rgba32_new() {
    let rgba = Rgba32::new(1000000, 500000, 250000, 125000);
    assert_eq!(rgba.r, Saturating(1000000));
    assert_eq!(rgba.g, Saturating(500000));
    assert_eq!(rgba.b, Saturating(250000));
    assert_eq!(rgba.a, Saturating(125000));
}

#[test]
fn test_rgba64_new() {
    let rgba = Rgba64::new(10000000000, 5000000000, 2500000000, 1250000000);
    assert_eq!(rgba.r, Saturating(10000000000));
    assert_eq!(rgba.g, Saturating(5000000000));
    assert_eq!(rgba.b, Saturating(2500000000));
    assert_eq!(rgba.a, Saturating(1250000000));
}

#[test]
fn test_bgra32_new() {
    let bgra = Bgra32::new(1000000, 500000, 250000, 125000);
    assert_eq!(bgra.b, Saturating(1000000));
    assert_eq!(bgra.g, Saturating(500000));
    assert_eq!(bgra.r, Saturating(250000));
    assert_eq!(bgra.a, Saturating(125000));
}

#[test]
fn test_bgra64_new() {
    let bgra = Bgra64::new(10000000000, 5000000000, 2500000000, 1250000000);
    assert_eq!(bgra.b, Saturating(10000000000));
    assert_eq!(bgra.g, Saturating(5000000000));
    assert_eq!(bgra.r, Saturating(2500000000));
    assert_eq!(bgra.a, Saturating(1250000000));
}

#[test]
fn test_rgb12_as_bytes() {
    let rgb = Rgb12::new(4095, 2048, 1024);
    let bytes = rgb.as_bytes();
    assert_eq!(bytes.len(), 6);
}

#[test]
fn test_rgba10_as_bytes() {
    let rgba = Rgba10::new(1023, 512, 256, 128);
    let bytes = rgba.as_bytes();
    assert_eq!(bytes.len(), 8);
}

#[test]
fn test_bgr12_new() {
    let bgr = Bgr12::new(4095, 2048, 1024);
    assert_eq!(bgr.b.value(), 4095);
    assert_eq!(bgr.g.value(), 2048);
    assert_eq!(bgr.r.value(), 1024);
}

#[test]
fn test_bgra12_new() {
    let bgra = Bgra12::new(4095, 2048, 1024, 512);
    assert_eq!(bgra.b.value(), 4095);
    assert_eq!(bgra.g.value(), 2048);
    assert_eq!(bgra.r.value(), 1024);
    assert_eq!(bgra.a.value(), 512);
}

#[test]
fn test_rgba12_new() {
    let rgba = Rgba12::new(4095, 2048, 1024, 512);
    assert_eq!(rgba.r.value(), 4095);
    assert_eq!(rgba.g.value(), 2048);
    assert_eq!(rgba.b.value(), 1024);
    assert_eq!(rgba.a.value(), 512);
}

#[test]
fn test_rgbaf64_new() {
    let rgba = RgbaF64::new(1.0, 0.5, 0.25, 0.8);
    assert_eq!(rgba.r, 1.0);
    assert_eq!(rgba.g, 0.5);
    assert_eq!(rgba.b, 0.25);
    assert_eq!(rgba.a, 0.8);
}

#[test]
fn test_bgrf64_new() {
    let bgr = BgrF64::new(1.0, 0.5, 0.25);
    assert_eq!(bgr.b, 1.0);
    assert_eq!(bgr.g, 0.5);
    assert_eq!(bgr.r, 0.25);
}

#[test]
fn test_bgraf64_new() {
    let bgra = BgraF64::new(1.0, 0.5, 0.25, 0.8);
    assert_eq!(bgra.b, 1.0);
    assert_eq!(bgra.g, 0.5);
    assert_eq!(bgra.r, 0.25);
    assert_eq!(bgra.a, 0.8);
}

#[test]
fn test_linear_pixel_u8() {
    let pixel = 100u8;
    // Post-ADR-0045: `u8` is a channel, not a pixel; the scale impl
    // lives on `LinearChannel<f32>`. The test name is kept for
    // historical continuity.
    let scaled = <u8 as LinearChannel<f32>>::scale(&pixel, 0.5);
    assert_eq!(scaled, 50.0f32);

    let scaled = <u8 as LinearChannel<f32>>::scale(&pixel, 2.0);
    assert_eq!(scaled, 200.0f32);
}

#[test]
fn test_linear_pixel_u16() {
    let pixel = 1000u16;
    let scaled = <u16 as LinearChannel<f32>>::scale(&pixel, 0.5);
    assert_eq!(scaled, 500.0f32);
}

#[test]
fn test_linear_pixel_u32() {
    let pixel = 100000u32;
    // Post-ADR-0045: `u32` is a channel with both `LinearChannel<f32>`
    // and `LinearChannel<f64>` impls (PLAN §3.4). Pin to f32 here; the
    // f64-scalar path has its own tests below.
    let scaled = <u32 as LinearChannel<f32>>::scale(&pixel, 0.1);
    assert!((scaled - 10000.0f64).abs() < 1.0);
}

#[test]
fn test_linear_pixel_u64() {
    let pixel = 1000000u64;
    let scaled = <u64 as LinearChannel<f32>>::scale(&pixel, 0.001);
    assert!((scaled - 1000.0f64).abs() < 1.0);
}

#[test]
fn test_linear_pixel_i8() {
    let pixel = 50i8;
    let scaled = <i8 as LinearChannel<f32>>::scale(&pixel, 2.0);
    assert_eq!(scaled, 100.0f32);
}

#[test]
fn test_linear_pixel_i16() {
    let pixel = 500i16;
    let scaled = <i16 as LinearChannel<f32>>::scale(&pixel, 0.5);
    assert_eq!(scaled, 250.0f32);
}

#[test]
fn test_linear_pixel_i32() {
    let pixel = 10000i32;
    let scaled = <i32 as LinearChannel<f32>>::scale(&pixel, 0.1);
    assert!((scaled - 1000.0f64).abs() < 1.0);
}

#[test]
fn test_linear_pixel_i64() {
    let pixel = 100000i64;
    let scaled = <i64 as LinearChannel<f32>>::scale(&pixel, 0.01);
    assert!((scaled - 1000.0f64).abs() < 1.0);
}

// ADR-0044 Phase E: `f32` / `f64` no longer implement `LinearPixel`.
// The former `test_linear_pixel_f32` / `test_linear_pixel_f64` pixel-role
// tests have been removed; the arithmetic is covered via
// `LinearChannel` (see `test_linear_channel_f32` and friends).

#[test]
fn test_linear_pixel_saturating_u8() {
    let pixel = Saturating(200u8);
    let scaled = <Saturating<u8> as LinearChannel<f32>>::scale(&pixel, 0.5);
    assert_eq!(scaled, 100.0f32);
}

#[test]
fn test_linear_pixel_saturating_u16() {
    let pixel = Saturating(1000u16);
    let scaled = <Saturating<u16> as LinearChannel<f32>>::scale(&pixel, 2.0);
    assert_eq!(scaled, 2000.0f32);
}

#[test]
fn test_linear_pixel_saturating_u32() {
    let pixel = Saturating(50000u32);
    let scaled = <Saturating<u32> as LinearChannel<f32>>::scale(&pixel, 0.5);
    assert_eq!(scaled, 25000.0f64);
}

#[test]
fn test_linear_pixel_saturating_u64() {
    let pixel = Saturating(1000000u64);
    let scaled = <Saturating<u64> as LinearChannel<f32>>::scale(&pixel, 0.1);
    assert!((scaled - 100000.0f64).abs() < 1.0);
}

#[test]
fn test_linear_pixel_mono10() {
    let pixel = Mono10::new(512);
    let scaled = pixel.scale(0.5);
    assert_eq!(scaled, MonoF32(256.0));
}

#[test]
fn test_linear_pixel_mono12() {
    let pixel = Mono12::new(2048);
    let scaled = pixel.scale(0.5);
    assert_eq!(scaled, MonoF32(1024.0));
}

#[test]
fn test_linear_pixel_mono14() {
    let pixel = Mono14::new(8192);
    let scaled = pixel.scale(0.5);
    assert_eq!(scaled, MonoF32(4096.0));
}

#[test]
fn test_linear_pixel_mono8() {
    let pixel = Mono8::new(200);
    let scaled = pixel.scale(0.5);
    assert_eq!(scaled, MonoF32(100.0));
}

#[test]
fn test_linear_pixel_mono16() {
    let pixel = Mono16::new(10000);
    let scaled = pixel.scale(0.5);
    assert_eq!(scaled, MonoF32(5000.0));
}

#[test]
fn test_linear_pixel_mono32() {
    let pixel = Mono32::new(100000);
    // Mono32 now has both `LinearPixel<f32>` and `LinearPixel<f64>` impls
    // (PLAN §3.4). Pin the scalar to f32 to preserve this test's original
    // intent; the f64-scalar path has dedicated coverage elsewhere.
    let scaled = <Mono32 as LinearPixel<f32>>::scale(&pixel, 0.1);
    assert!((scaled - MonoF64(10000.0)).abs().0 < 1.0);
}

#[test]
fn test_linear_pixel_mono64() {
    let pixel = Mono64::new(1000000);
    let scaled = <Mono64 as LinearPixel<f32>>::scale(&pixel, 0.01);
    assert!((scaled - MonoF64(10000.0)).abs().0 < 1.0);
}

#[test]
fn test_linear_pixel_rgb8() {
    let pixel = Rgb8::new(100, 150, 200);
    let scaled = pixel.scale(0.5);
    assert_eq!(scaled.r, 50.0);
    assert_eq!(scaled.g, 75.0);
    assert_eq!(scaled.b, 100.0);
}

#[test]
fn test_linear_pixel_rgba8() {
    let pixel = Rgba8::new(100, 150, 200, 255);
    let scaled = pixel.scale(0.5);
    assert_eq!(scaled.r, 50.0);
    assert_eq!(scaled.g, 75.0);
    assert_eq!(scaled.b, 100.0);
    assert_eq!(scaled.a, 127.5);
}

#[test]
fn test_linear_pixel_bgr8() {
    let pixel = Bgr8::new(100, 150, 200);
    let scaled = pixel.scale(0.5);
    assert_eq!(scaled.b, 50.0);
    assert_eq!(scaled.g, 75.0);
    assert_eq!(scaled.r, 100.0);
}

#[test]
fn test_linear_pixel_bgra8() {
    let pixel = Bgra8::new(100, 150, 200, 255);
    let scaled = pixel.scale(0.5);
    assert_eq!(scaled.b, 50.0);
    assert_eq!(scaled.g, 75.0);
    assert_eq!(scaled.r, 100.0);
    assert_eq!(scaled.a, 127.5);
}

#[test]
fn test_linear_pixel_rgbf32() {
    let pixel = RgbF32::new(1.0, 0.5, 0.25);
    let scaled = pixel.scale(2.0);
    assert_eq!(scaled.r, 2.0);
    assert_eq!(scaled.g, 1.0);
    assert_eq!(scaled.b, 0.5);
}

#[test]
fn test_linear_pixel_rgbaf32() {
    let pixel = RgbaF32::new(1.0, 0.5, 0.25, 0.8);
    let scaled = pixel.scale(2.0);
    assert_eq!(scaled.r, 2.0);
    assert_eq!(scaled.g, 1.0);
    assert_eq!(scaled.b, 0.5);
    assert_eq!(scaled.a, 1.6);
}

// `test_blend_u8` removed (ADR-0045 Phase S4): `u8` is a channel, not
// a pixel, so it no longer implements `LinearPixel` / `LinearSpace`
// and cannot be blended directly. Use `Mono8` for the pixel role
// (see `test_blend_mono8` below or the Mono family blend coverage).

// `test_blend_f32` removed (ADR-0044 Phase E): `f32` is no longer a
// pixel and no longer implements `LinearPixel` / `LinearSpace`. The
// equivalent coverage now lives on `MonoF32` (see the Mono family
// blend tests below).

#[test]
fn test_blend_mono10() {
    use crate::pixel::blend;
    let pixel1 = Mono10::new(0);
    let pixel2 = Mono10::new(1000);
    let blended = blend(&pixel1, &pixel2, 0.5);
    assert_eq!(blended, MonoF32(500.0));
}

#[test]
fn test_blend_rgb8() {
    use crate::pixel::blend;
    let pixel1 = Rgb8::new(0, 0, 0);
    let pixel2 = Rgb8::new(100, 200, 255);
    let blended = blend(&pixel1, &pixel2, 0.5);
    assert_eq!(blended.r, 50.0);
    assert_eq!(blended.g, 100.0);
    assert_eq!(blended.b, 127.5);
}

#[test]
fn test_blend_rgbf32() {
    use crate::pixel::blend;
    let pixel1 = RgbF32::new(0.0, 0.0, 0.0);
    let pixel2 = RgbF32::new(1.0, 1.0, 1.0);
    let blended = blend(&pixel1, &pixel2, 0.3);
    assert!((blended.r - 0.3).abs() < 0.001);
    assert!((blended.g - 0.3).abs() < 0.001);
    assert!((blended.b - 0.3).abs() < 0.001);
}

#[test]
fn test_plain_pixel_as_mut_bytes() {
    let mut pixel = Rgb8::new(10, 20, 30);
    {
        let bytes = pixel.as_mut_bytes();
        bytes[0] = 100;
        bytes[1] = 200;
    }
    assert_eq!(pixel.r, Saturating(100));
    assert_eq!(pixel.g, Saturating(200));
}

#[test]
fn test_plain_pixel_mono10_as_bytes_le() {
    let pixel = Mono10::new(512);
    let bytes_le = pixel.as_bytes_le();
    assert_eq!(&*bytes_le, &[0, 2]);
}

#[test]
fn test_plain_pixel_mono10_as_bytes_be() {
    let pixel = Mono10::new(512);
    let bytes_be = pixel.as_bytes_be();
    assert_eq!(&*bytes_be, &[2, 0]);
}

// ==================== from_bytes tests ====================

#[test]
fn test_mono8_from_bytes() {
    let bytes = [42u8];
    let pixel = Mono8::from_bytes(&bytes).unwrap();
    assert_eq!(pixel.value(), 42);
}

#[test]
fn test_mono10_from_bytes() {
    let pixel = Mono10::new(512);
    let bytes = pixel.as_bytes();
    let reconstructed = Mono10::from_bytes(bytes).unwrap();
    assert_eq!(reconstructed.value(), 512);
}

#[test]
fn test_rgb8_from_bytes() {
    let bytes = [10u8, 20, 30];
    let pixel = Rgb8::from_bytes(&bytes).unwrap();
    assert_eq!(pixel.r.0, 10);
    assert_eq!(pixel.g.0, 20);
    assert_eq!(pixel.b.0, 30);
}

#[test]
fn test_rgba8_from_bytes() {
    let bytes = [10u8, 20, 30, 40];
    let pixel = Rgba8::from_bytes(&bytes).unwrap();
    assert_eq!(pixel.r.0, 10);
    assert_eq!(pixel.g.0, 20);
    assert_eq!(pixel.b.0, 30);
    assert_eq!(pixel.a.0, 40);
}

#[test]
fn test_rgb16_from_bytes() {
    let pixel = Rgb16::new(1000, 2000, 3000);
    let bytes = pixel.as_bytes();
    let reconstructed = Rgb16::from_bytes(bytes).unwrap();
    assert_eq!(reconstructed.r.0, 1000);
    assert_eq!(reconstructed.g.0, 2000);
    assert_eq!(reconstructed.b.0, 3000);
}

#[test]
fn test_rgba16_from_bytes() {
    let pixel = Rgba16::new(1000, 2000, 3000, 4000);
    let bytes = pixel.as_bytes();
    let reconstructed = Rgba16::from_bytes(bytes).unwrap();
    assert_eq!(reconstructed.r.0, 1000);
    assert_eq!(reconstructed.g.0, 2000);
    assert_eq!(reconstructed.b.0, 3000);
    assert_eq!(reconstructed.a.0, 4000);
}

// ==================== from_bytes_le tests ====================

#[test]
fn test_mono8_from_bytes_le() {
    // Single byte - endianness doesn't matter
    let bytes = [42u8];
    let pixel = Mono8::from_bytes_le(&bytes).unwrap();
    assert_eq!(pixel.value(), 42);
}

#[test]
fn test_mono10_from_bytes_le() {
    // 512 = 0x0200, little-endian: [0x00, 0x02]
    let bytes_le = [0x00u8, 0x02];
    let pixel = Mono10::from_bytes_le(&bytes_le).unwrap();
    assert_eq!(pixel.value(), 512);
}

#[test]
fn test_rgb8_from_bytes_le() {
    // Single byte channels - endianness doesn't matter
    let bytes = [10u8, 20, 30];
    let pixel = Rgb8::from_bytes_le(&bytes).unwrap();
    assert_eq!(pixel.r.0, 10);
    assert_eq!(pixel.g.0, 20);
    assert_eq!(pixel.b.0, 30);
}

#[test]
fn test_rgb16_from_bytes_le() {
    // 1000 = 0x03E8, little-endian: [0xE8, 0x03]
    // 2000 = 0x07D0, little-endian: [0xD0, 0x07]
    // 3000 = 0x0BB8, little-endian: [0xB8, 0x0B]
    let bytes_le = [0xE8u8, 0x03, 0xD0, 0x07, 0xB8, 0x0B];
    let pixel = Rgb16::from_bytes_le(&bytes_le).unwrap();
    assert_eq!(pixel.r.0, 1000);
    assert_eq!(pixel.g.0, 2000);
    assert_eq!(pixel.b.0, 3000);
}

#[test]
fn test_rgba16_from_bytes_le() {
    // 1000 = 0x03E8, little-endian: [0xE8, 0x03]
    // 2000 = 0x07D0, little-endian: [0xD0, 0x07]
    // 3000 = 0x0BB8, little-endian: [0xB8, 0x0B]
    // 4000 = 0x0FA0, little-endian: [0xA0, 0x0F]
    let bytes_le = [0xE8u8, 0x03, 0xD0, 0x07, 0xB8, 0x0B, 0xA0, 0x0F];
    let pixel = Rgba16::from_bytes_le(&bytes_le).unwrap();
    assert_eq!(pixel.r.0, 1000);
    assert_eq!(pixel.g.0, 2000);
    assert_eq!(pixel.b.0, 3000);
    assert_eq!(pixel.a.0, 4000);
}

// ==================== from_bytes_be tests ====================

#[test]
fn test_mono8_from_bytes_be() {
    // Single byte - endianness doesn't matter
    let bytes = [42u8];
    let pixel = Mono8::from_bytes_be(&bytes).unwrap();
    assert_eq!(pixel.value(), 42);
}

#[test]
fn test_mono10_from_bytes_be() {
    // 512 = 0x0200, big-endian: [0x02, 0x00]
    let bytes_be = [0x02u8, 0x00];
    let pixel = Mono10::from_bytes_be(&bytes_be).unwrap();
    assert_eq!(pixel.value(), 512);
}

#[test]
fn test_rgb8_from_bytes_be() {
    // Single byte channels - endianness doesn't matter
    let bytes = [10u8, 20, 30];
    let pixel = Rgb8::from_bytes_be(&bytes).unwrap();
    assert_eq!(pixel.r.0, 10);
    assert_eq!(pixel.g.0, 20);
    assert_eq!(pixel.b.0, 30);
}

#[test]
fn test_rgb16_from_bytes_be() {
    // 1000 = 0x03E8, big-endian: [0x03, 0xE8]
    // 2000 = 0x07D0, big-endian: [0x07, 0xD0]
    // 3000 = 0x0BB8, big-endian: [0x0B, 0xB8]
    let bytes_be = [0x03u8, 0xE8, 0x07, 0xD0, 0x0B, 0xB8];
    let pixel = Rgb16::from_bytes_be(&bytes_be).unwrap();
    assert_eq!(pixel.r.0, 1000);
    assert_eq!(pixel.g.0, 2000);
    assert_eq!(pixel.b.0, 3000);
}

#[test]
fn test_rgba16_from_bytes_be() {
    // 1000 = 0x03E8, big-endian: [0x03, 0xE8]
    // 2000 = 0x07D0, big-endian: [0x07, 0xD0]
    // 3000 = 0x0BB8, big-endian: [0x0B, 0xB8]
    // 4000 = 0x0FA0, big-endian: [0x0F, 0xA0]
    let bytes_be = [0x03u8, 0xE8, 0x07, 0xD0, 0x0B, 0xB8, 0x0F, 0xA0];
    let pixel = Rgba16::from_bytes_be(&bytes_be).unwrap();
    assert_eq!(pixel.r.0, 1000);
    assert_eq!(pixel.g.0, 2000);
    assert_eq!(pixel.b.0, 3000);
    assert_eq!(pixel.a.0, 4000);
}

// ==================== roundtrip tests ====================

#[test]
fn test_mono10_roundtrip_le() {
    let original = Mono10::new(512);
    let bytes = original.as_bytes_le();
    let reconstructed = Mono10::from_bytes_le(&bytes).unwrap();
    assert_eq!(original.value(), reconstructed.value());
}

#[test]
fn test_mono10_roundtrip_be() {
    let original = Mono10::new(512);
    let bytes = original.as_bytes_be();
    let reconstructed = Mono10::from_bytes_be(&bytes).unwrap();
    assert_eq!(original.value(), reconstructed.value());
}

#[test]
fn test_rgb16_roundtrip_le() {
    let original = Rgb16::new(1000, 2000, 3000);
    let bytes = original.as_bytes_le();
    let reconstructed = Rgb16::from_bytes_le(&bytes).unwrap();
    assert_eq!(original, reconstructed);
}

#[test]
fn test_rgb16_roundtrip_be() {
    let original = Rgb16::new(1000, 2000, 3000);
    let bytes = original.as_bytes_be();
    let reconstructed = Rgb16::from_bytes_be(&bytes).unwrap();
    assert_eq!(original, reconstructed);
}

#[test]
fn test_rgba16_roundtrip_le() {
    let original = Rgba16::new(1000, 2000, 3000, 4000);
    let bytes = original.as_bytes_le();
    let reconstructed = Rgba16::from_bytes_le(&bytes).unwrap();
    assert_eq!(original, reconstructed);
}

#[test]
fn test_rgba16_roundtrip_be() {
    let original = Rgba16::new(1000, 2000, 3000, 4000);
    let bytes = original.as_bytes_be();
    let reconstructed = Rgba16::from_bytes_be(&bytes).unwrap();
    assert_eq!(original, reconstructed);
}

// ==================== ALIGN const tests ====================

#[test]
fn test_align_mono8() {
    assert_eq!(Mono8::ALIGN, 1);
}

#[test]
fn test_align_mono16() {
    assert_eq!(Mono16::ALIGN, 2);
}

#[test]
fn test_align_mono32() {
    assert_eq!(Mono32::ALIGN, 4);
}

#[test]
fn test_align_rgb8() {
    assert_eq!(Rgb8::ALIGN, 1);
}

#[test]
fn test_align_rgb16() {
    assert_eq!(Rgb16::ALIGN, 2);
}

#[test]
fn test_align_rgba8() {
    assert_eq!(Rgba8::ALIGN, 1);
}

#[test]
fn test_align_rgba16() {
    assert_eq!(Rgba16::ALIGN, 2);
}

// `test_align_f32` / `test_align_f64` removed (ADR-0044 Phase E):
// `f32` / `f64` no longer implement `PlainPixel`. They still carry
// byte-layout via `PlainChannel` (ADR-0046) — see the
// `plain_channel_inventory` test. Pixel-role alignment coverage
// for floats lives on `MonoF32` / `MonoF64`.

// ==================== Channel sum validation tests ====================
// These tests verify that SIZE == sum(CHANNELS) for various pixel types.
// The compile-time assertion _ASSERT_CHANNELS ensures this invariant,
// but we also verify it at runtime for documentation purposes.

#[test]
fn test_channel_sum_mono8() {
    let sum: usize = Mono8::CHANNELS.iter().sum();
    assert_eq!(Mono8::SIZE, sum);
}

#[test]
fn test_channel_sum_mono16() {
    let sum: usize = Mono16::CHANNELS.iter().sum();
    assert_eq!(Mono16::SIZE, sum);
}

#[test]
fn test_channel_sum_rgb8() {
    let sum: usize = Rgb8::CHANNELS.iter().sum();
    assert_eq!(Rgb8::SIZE, sum);
}

#[test]
fn test_channel_sum_rgb16() {
    let sum: usize = Rgb16::CHANNELS.iter().sum();
    assert_eq!(Rgb16::SIZE, sum);
}

#[test]
fn test_channel_sum_rgba8() {
    let sum: usize = Rgba8::CHANNELS.iter().sum();
    assert_eq!(Rgba8::SIZE, sum);
}

#[test]
fn test_channel_sum_rgba16() {
    let sum: usize = Rgba16::CHANNELS.iter().sum();
    assert_eq!(Rgba16::SIZE, sum);
}

#[test]
fn test_channel_sum_rgbf32() {
    let sum: usize = RgbF32::CHANNELS.iter().sum();
    assert_eq!(RgbF32::SIZE, sum);
}

#[test]
fn test_channel_sum_rgbaf32() {
    let sum: usize = RgbaF32::CHANNELS.iter().sum();
    assert_eq!(RgbaF32::SIZE, sum);
}

// ==================== HomogeneousPixel tests ====================
// These tests verify that HomogeneousPixel is correctly implemented
// for all standard pixel types via derive or manual impls.

// --- Channel count ---

#[test]
fn test_uniform_channel_count_primitives() {
    assert_eq!(<u8 as HomogeneousPixel>::CHANNEL_COUNT, 1);
    assert_eq!(<u16 as HomogeneousPixel>::CHANNEL_COUNT, 1);
    assert_eq!(<u32 as HomogeneousPixel>::CHANNEL_COUNT, 1);
    assert_eq!(<u64 as HomogeneousPixel>::CHANNEL_COUNT, 1);
    assert_eq!(<i8 as HomogeneousPixel>::CHANNEL_COUNT, 1);
    assert_eq!(<i16 as HomogeneousPixel>::CHANNEL_COUNT, 1);
    assert_eq!(<i32 as HomogeneousPixel>::CHANNEL_COUNT, 1);
    assert_eq!(<i64 as HomogeneousPixel>::CHANNEL_COUNT, 1);
    // `f32` / `f64` no longer implement `HomogeneousPixel` (ADR-0044
    // Phase E). Equivalent coverage lives on `MonoF32` / `MonoF64`.
}

#[test]
fn test_uniform_channel_count_mono() {
    assert_eq!(Mono8::CHANNEL_COUNT, 1);
    assert_eq!(Mono16::CHANNEL_COUNT, 1);
    assert_eq!(Mono32::CHANNEL_COUNT, 1);
    assert_eq!(Mono64::CHANNEL_COUNT, 1);
    assert_eq!(Mono10::CHANNEL_COUNT, 1);
    assert_eq!(Mono12::CHANNEL_COUNT, 1);
    assert_eq!(Mono14::CHANNEL_COUNT, 1);
}

#[test]
fn test_uniform_channel_count_rgb_family() {
    assert_eq!(Rgb8::CHANNEL_COUNT, 3);
    assert_eq!(Rgb16::CHANNEL_COUNT, 3);
    assert_eq!(Rgb32::CHANNEL_COUNT, 3);
    assert_eq!(Rgb64::CHANNEL_COUNT, 3);
    assert_eq!(RgbF32::CHANNEL_COUNT, 3);
    assert_eq!(RgbF64::CHANNEL_COUNT, 3);
    assert_eq!(Rgb10::CHANNEL_COUNT, 3);
    assert_eq!(Rgb12::CHANNEL_COUNT, 3);
    assert_eq!(Rgb14::CHANNEL_COUNT, 3);
}

#[test]
fn test_uniform_channel_count_rgba_family() {
    assert_eq!(Rgba8::CHANNEL_COUNT, 4);
    assert_eq!(Rgba16::CHANNEL_COUNT, 4);
    assert_eq!(Rgba32::CHANNEL_COUNT, 4);
    assert_eq!(Rgba64::CHANNEL_COUNT, 4);
    assert_eq!(RgbaF32::CHANNEL_COUNT, 4);
    assert_eq!(RgbaF64::CHANNEL_COUNT, 4);
    assert_eq!(Rgba10::CHANNEL_COUNT, 4);
    assert_eq!(Rgba12::CHANNEL_COUNT, 4);
    assert_eq!(Rgba14::CHANNEL_COUNT, 4);
}

#[test]
fn test_uniform_channel_count_bgr_family() {
    assert_eq!(Bgr8::CHANNEL_COUNT, 3);
    assert_eq!(Bgr16::CHANNEL_COUNT, 3);
    assert_eq!(Bgr32::CHANNEL_COUNT, 3);
    assert_eq!(Bgr64::CHANNEL_COUNT, 3);
    assert_eq!(BgrF32::CHANNEL_COUNT, 3);
    assert_eq!(BgrF64::CHANNEL_COUNT, 3);
    assert_eq!(Bgr10::CHANNEL_COUNT, 3);
    assert_eq!(Bgr12::CHANNEL_COUNT, 3);
    assert_eq!(Bgr14::CHANNEL_COUNT, 3);
}

#[test]
fn test_uniform_channel_count_bgra_family() {
    assert_eq!(Bgra8::CHANNEL_COUNT, 4);
    assert_eq!(Bgra16::CHANNEL_COUNT, 4);
    assert_eq!(Bgra32::CHANNEL_COUNT, 4);
    assert_eq!(Bgra64::CHANNEL_COUNT, 4);
    assert_eq!(BgraF32::CHANNEL_COUNT, 4);
    assert_eq!(BgraF64::CHANNEL_COUNT, 4);
    assert_eq!(Bgra10::CHANNEL_COUNT, 4);
    assert_eq!(Bgra12::CHANNEL_COUNT, 4);
    assert_eq!(Bgra14::CHANNEL_COUNT, 4);
}

// --- Size assertions (force const evaluation) ---

#[test]
fn test_uniform_size_assert_all() {
    // Primitives
    let _ = <u8 as HomogeneousPixel>::_SIZE_ASSERT;
    let _ = <u16 as HomogeneousPixel>::_SIZE_ASSERT;
    let _ = <u32 as HomogeneousPixel>::_SIZE_ASSERT;
    let _ = <u64 as HomogeneousPixel>::_SIZE_ASSERT;
    let _ = <i8 as HomogeneousPixel>::_SIZE_ASSERT;
    let _ = <i16 as HomogeneousPixel>::_SIZE_ASSERT;
    let _ = <i32 as HomogeneousPixel>::_SIZE_ASSERT;
    let _ = <i64 as HomogeneousPixel>::_SIZE_ASSERT;
    // `f32` / `f64` no longer implement `HomogeneousPixel` (ADR-0044
    // Phase E); their size assertion role migrated to
    // `MonoF32` / `MonoF64`.
    // Mono
    let _ = <Mono8 as HomogeneousPixel>::_SIZE_ASSERT;
    let _ = <Mono16 as HomogeneousPixel>::_SIZE_ASSERT;
    let _ = <Mono32 as HomogeneousPixel>::_SIZE_ASSERT;
    let _ = <Mono64 as HomogeneousPixel>::_SIZE_ASSERT;
    let _ = <Mono10 as HomogeneousPixel>::_SIZE_ASSERT;
    let _ = <Mono12 as HomogeneousPixel>::_SIZE_ASSERT;
    // Rgb
    let _ = <Rgb8 as HomogeneousPixel>::_SIZE_ASSERT;
    let _ = <Rgb16 as HomogeneousPixel>::_SIZE_ASSERT;
    let _ = <Rgb32 as HomogeneousPixel>::_SIZE_ASSERT;
    let _ = <Rgb64 as HomogeneousPixel>::_SIZE_ASSERT;
    let _ = <RgbF32 as HomogeneousPixel>::_SIZE_ASSERT;
    let _ = <RgbF64 as HomogeneousPixel>::_SIZE_ASSERT;
    let _ = <Rgb10 as HomogeneousPixel>::_SIZE_ASSERT;
    // Rgba
    let _ = <Rgba8 as HomogeneousPixel>::_SIZE_ASSERT;
    let _ = <Rgba16 as HomogeneousPixel>::_SIZE_ASSERT;
    let _ = <Rgba32 as HomogeneousPixel>::_SIZE_ASSERT;
    let _ = <Rgba64 as HomogeneousPixel>::_SIZE_ASSERT;
    let _ = <RgbaF32 as HomogeneousPixel>::_SIZE_ASSERT;
    let _ = <RgbaF64 as HomogeneousPixel>::_SIZE_ASSERT;
    let _ = <Rgba10 as HomogeneousPixel>::_SIZE_ASSERT;
    // Bgr
    let _ = <Bgr8 as HomogeneousPixel>::_SIZE_ASSERT;
    let _ = <Bgr16 as HomogeneousPixel>::_SIZE_ASSERT;
    let _ = <Bgr32 as HomogeneousPixel>::_SIZE_ASSERT;
    let _ = <Bgr64 as HomogeneousPixel>::_SIZE_ASSERT;
    let _ = <BgrF32 as HomogeneousPixel>::_SIZE_ASSERT;
    let _ = <BgrF64 as HomogeneousPixel>::_SIZE_ASSERT;
    let _ = <Bgr10 as HomogeneousPixel>::_SIZE_ASSERT;
    // Bgra
    let _ = <Bgra8 as HomogeneousPixel>::_SIZE_ASSERT;
    let _ = <Bgra16 as HomogeneousPixel>::_SIZE_ASSERT;
    let _ = <Bgra32 as HomogeneousPixel>::_SIZE_ASSERT;
    let _ = <Bgra64 as HomogeneousPixel>::_SIZE_ASSERT;
    let _ = <BgraF32 as HomogeneousPixel>::_SIZE_ASSERT;
    let _ = <BgraF64 as HomogeneousPixel>::_SIZE_ASSERT;
    let _ = <Bgra10 as HomogeneousPixel>::_SIZE_ASSERT;
}

// --- channel() / to_channels() / from_channels() round-trips ---

#[test]
fn test_uniform_rgb8_channel_access() {
    let px = Rgb8::new(10, 20, 30);
    assert_eq!(px.channel(0), Saturating(10u8));
    assert_eq!(px.channel(1), Saturating(20u8));
    assert_eq!(px.channel(2), Saturating(30u8));
}

#[test]
fn test_uniform_rgba8_channel_access() {
    let px = Rgba8::new(1, 2, 3, 4);
    assert_eq!(px.channel(0), Saturating(1u8));
    assert_eq!(px.channel(1), Saturating(2u8));
    assert_eq!(px.channel(2), Saturating(3u8));
    assert_eq!(px.channel(3), Saturating(4u8));
}

#[test]
fn test_uniform_bgr8_channel_access() {
    let px = Bgr8::new(10, 20, 30);
    // Memory order is b, g, r
    assert_eq!(px.channel(0), Saturating(10u8)); // b
    assert_eq!(px.channel(1), Saturating(20u8)); // g
    assert_eq!(px.channel(2), Saturating(30u8)); // r
}

#[test]
fn test_uniform_bgra8_channel_access() {
    let px = Bgra8::new(10, 20, 30, 40);
    assert_eq!(px.channel(0), Saturating(10u8)); // b
    assert_eq!(px.channel(1), Saturating(20u8)); // g
    assert_eq!(px.channel(2), Saturating(30u8)); // r
    assert_eq!(px.channel(3), Saturating(40u8)); // a
}

#[test]
fn test_uniform_rgb16_channel_access() {
    let px = Rgb16::new(1000, 2000, 3000);
    assert_eq!(px.channel(0), Saturating(1000u16));
    assert_eq!(px.channel(1), Saturating(2000u16));
    assert_eq!(px.channel(2), Saturating(3000u16));
}

#[test]
fn test_uniform_rgbf32_channel_access() {
    let px = RgbF32::new(0.25, 0.5, 0.75);
    assert_eq!(px.channel(0), 0.25f32);
    assert_eq!(px.channel(1), 0.5f32);
    assert_eq!(px.channel(2), 0.75f32);
}

#[test]
fn test_uniform_rgbf64_channel_access() {
    let px = RgbF64::new(0.1, 0.2, 0.3);
    assert_eq!(px.channel(0), 0.1f64);
    assert_eq!(px.channel(1), 0.2f64);
    assert_eq!(px.channel(2), 0.3f64);
}

#[test]
fn test_uniform_mono8_channel_access() {
    let px = Mono8::new(42);
    assert_eq!(px.channel(0), Saturating(42u8));
}

#[test]
fn test_uniform_mono16_channel_access() {
    let px = Mono16::new(12345);
    assert_eq!(px.channel(0), Saturating(12345u16));
}

#[test]
fn test_uniform_mono10_channel_access() {
    let px = Mono10::new(500);
    assert_eq!(px.channel(0), Saturating(500u16));
}

#[test]
fn test_uniform_rgb10_channel_access() {
    let px = Rgb10::new(100, 200, 300);
    assert_eq!(px.channel(0), Mono::<10>::new(100));
    assert_eq!(px.channel(1), Mono::<10>::new(200));
    assert_eq!(px.channel(2), Mono::<10>::new(300));
}

#[test]
fn test_uniform_bgr10_channel_access() {
    let px = Bgr10::new(100, 200, 300);
    assert_eq!(px.channel(0), Mono::<10>::new(100));
    assert_eq!(px.channel(1), Mono::<10>::new(200));
    assert_eq!(px.channel(2), Mono::<10>::new(300));
}

#[test]
fn test_uniform_primitive_u8_channel() {
    let val: u8 = 99;
    assert_eq!(val.channel(0), 99u8);
    assert_eq!(val.to_channels(), [99u8]);
    assert_eq!(u8::from_channels(&[42]), 42u8);
}

// `test_uniform_primitive_f32_channel` / `..._f64_...` removed
// (ADR-0044 Phase E): raw floats are channels, not pixels, so they
// no longer implement `HomogeneousPixel`. Equivalent coverage lives
// on `MonoF32` / `MonoF64` via `family_tests.rs`.

// --- to_channels round-trips ---

#[test]
fn test_uniform_rgb8_to_channels() {
    let px = Rgb8::new(10, 20, 30);
    let ch = px.to_channels();
    assert_eq!(ch, [Saturating(10u8), Saturating(20), Saturating(30)]);
}

#[test]
fn test_uniform_rgba8_to_channels() {
    let px = Rgba8::new(1, 2, 3, 4);
    let ch = px.to_channels();
    assert_eq!(
        ch,
        [Saturating(1u8), Saturating(2), Saturating(3), Saturating(4)]
    );
}

#[test]
fn test_uniform_bgr8_to_channels() {
    let px = Bgr8::new(10, 20, 30);
    let ch = px.to_channels();
    assert_eq!(ch, [Saturating(10u8), Saturating(20), Saturating(30)]);
}

#[test]
fn test_uniform_rgbf32_to_channels() {
    let px = RgbF32::new(0.1, 0.2, 0.3);
    let ch = px.to_channels();
    assert_eq!(ch, [0.1f32, 0.2, 0.3]);
}

#[test]
fn test_uniform_rgb10_to_channels() {
    let px = Rgb10::new(100, 200, 300);
    let ch = px.to_channels();
    assert_eq!(
        ch,
        [
            Mono::<10>::new(100),
            Mono::<10>::new(200),
            Mono::<10>::new(300)
        ]
    );
}

// --- from_channels ---

#[test]
fn test_uniform_rgb8_from_channels() {
    let ch = [Saturating(10u8), Saturating(20), Saturating(30)];
    let px = Rgb8::from_channels(&ch);
    assert_eq!(px, Rgb8::new(10, 20, 30));
}

#[test]
fn test_uniform_rgba16_from_channels() {
    let ch = [
        Saturating(100u16),
        Saturating(200),
        Saturating(300),
        Saturating(400),
    ];
    let px = Rgba16::from_channels(&ch);
    assert_eq!(px, Rgba16::new(100, 200, 300, 400));
}

#[test]
fn test_uniform_bgrf32_from_channels() {
    let ch = [1.0f32, 2.0, 3.0];
    let px = BgrF32::from_channels(&ch);
    assert_eq!(px, BgrF32::new(1.0, 2.0, 3.0));
}

#[test]
fn test_uniform_rgb10_from_channels() {
    let ch = [
        Mono::<10>::new(100),
        Mono::<10>::new(200),
        Mono::<10>::new(300),
    ];
    let px = Rgb10::from_channels(&ch);
    assert_eq!(px, Rgb10::new(100, 200, 300));
}

// --- set_channel ---

#[test]
fn test_uniform_rgb8_set_channel() {
    let mut px = Rgb8::new(0, 0, 0);
    px.set_channel(0, Saturating(10));
    px.set_channel(1, Saturating(20));
    px.set_channel(2, Saturating(30));
    assert_eq!(px, Rgb8::new(10, 20, 30));
}

#[test]
fn test_uniform_rgba8_set_channel() {
    let mut px = Rgba8::new(0, 0, 0, 0);
    px.set_channel(3, Saturating(255));
    assert_eq!(px.a, Saturating(255u8));
}

#[test]
fn test_uniform_rgbf64_set_channel() {
    let mut px = RgbF64::new(0.0, 0.0, 0.0);
    px.set_channel(1, 1.5);
    assert_eq!(px.g, 1.5);
}

#[test]
fn test_uniform_bgra16_set_channel() {
    let mut px = Bgra16::new(0, 0, 0, 0);
    px.set_channel(2, Saturating(1000));
    assert_eq!(px.r, Saturating(1000u16));
}

// --- Full round-trips: from_channels(to_channels()) == identity ---

#[test]
fn test_uniform_roundtrip_rgb8() {
    let original = Rgb8::new(11, 22, 33);
    let reconstructed = Rgb8::from_channels(original.to_channels().as_ref());
    assert_eq!(original, reconstructed);
}

#[test]
fn test_uniform_roundtrip_rgba8() {
    let original = Rgba8::new(1, 2, 3, 4);
    let reconstructed = Rgba8::from_channels(original.to_channels().as_ref());
    assert_eq!(original, reconstructed);
}

#[test]
fn test_uniform_roundtrip_rgb16() {
    let original = Rgb16::new(1000, 2000, 3000);
    let reconstructed = Rgb16::from_channels(original.to_channels().as_ref());
    assert_eq!(original, reconstructed);
}

#[test]
fn test_uniform_roundtrip_rgba32() {
    let original = Rgba32::new(100_000, 200_000, 300_000, 400_000);
    let reconstructed = Rgba32::from_channels(original.to_channels().as_ref());
    assert_eq!(original, reconstructed);
}

#[test]
fn test_uniform_roundtrip_rgb64() {
    let original = Rgb64::new(1, 2, 3);
    let reconstructed = Rgb64::from_channels(original.to_channels().as_ref());
    assert_eq!(original, reconstructed);
}

#[test]
fn test_uniform_roundtrip_rgbf32() {
    let original = RgbF32::new(0.1, 0.2, 0.3);
    let reconstructed = RgbF32::from_channels(original.to_channels().as_ref());
    assert_eq!(original, reconstructed);
}

#[test]
fn test_uniform_roundtrip_rgbaf64() {
    let original = RgbaF64::new(0.1, 0.2, 0.3, 0.4);
    let reconstructed = RgbaF64::from_channels(original.to_channels().as_ref());
    assert_eq!(original, reconstructed);
}

#[test]
fn test_uniform_roundtrip_bgr8() {
    let original = Bgr8::new(10, 20, 30);
    let reconstructed = Bgr8::from_channels(original.to_channels().as_ref());
    assert_eq!(original, reconstructed);
}

#[test]
fn test_uniform_roundtrip_bgra8() {
    let original = Bgra8::new(1, 2, 3, 4);
    let reconstructed = Bgra8::from_channels(original.to_channels().as_ref());
    assert_eq!(original, reconstructed);
}

#[test]
fn test_uniform_roundtrip_bgr16() {
    let original = Bgr16::new(100, 200, 300);
    let reconstructed = Bgr16::from_channels(original.to_channels().as_ref());
    assert_eq!(original, reconstructed);
}

#[test]
fn test_uniform_roundtrip_bgra32() {
    let original = Bgra32::new(10, 20, 30, 40);
    let reconstructed = Bgra32::from_channels(original.to_channels().as_ref());
    assert_eq!(original, reconstructed);
}

#[test]
fn test_uniform_roundtrip_bgrf32() {
    let original = BgrF32::new(0.5, 0.6, 0.7);
    let reconstructed = BgrF32::from_channels(original.to_channels().as_ref());
    assert_eq!(original, reconstructed);
}

#[test]
fn test_uniform_roundtrip_bgraf64() {
    let original = BgraF64::new(0.1, 0.2, 0.3, 0.4);
    let reconstructed = BgraF64::from_channels(original.to_channels().as_ref());
    assert_eq!(original, reconstructed);
}

#[test]
fn test_uniform_roundtrip_mono8() {
    let original = Mono8::new(42);
    let reconstructed = Mono8::from_channels(original.to_channels().as_ref());
    assert_eq!(original, reconstructed);
}

#[test]
fn test_uniform_roundtrip_mono16() {
    let original = Mono16::new(12345);
    let reconstructed = Mono16::from_channels(original.to_channels().as_ref());
    assert_eq!(original, reconstructed);
}

#[test]
fn test_uniform_roundtrip_mono32() {
    let original = Mono32::new(100_000);
    let reconstructed = Mono32::from_channels(original.to_channels().as_ref());
    assert_eq!(original, reconstructed);
}

#[test]
fn test_uniform_roundtrip_mono64() {
    let original = Mono64::new(999_999_999);
    let reconstructed = Mono64::from_channels(original.to_channels().as_ref());
    assert_eq!(original, reconstructed);
}

#[test]
fn test_uniform_roundtrip_mono10() {
    let original = Mono10::new(500);
    let reconstructed = Mono10::from_channels(original.to_channels().as_ref());
    assert_eq!(original, reconstructed);
}

#[test]
fn test_uniform_roundtrip_rgb10() {
    let original = Rgb10::new(100, 200, 300);
    let reconstructed = Rgb10::from_channels(original.to_channels().as_ref());
    assert_eq!(original, reconstructed);
}

#[test]
fn test_uniform_roundtrip_rgba12() {
    let original = Rgba12::new(100, 200, 300, 400);
    let reconstructed = Rgba12::from_channels(original.to_channels().as_ref());
    assert_eq!(original, reconstructed);
}

#[test]
fn test_uniform_roundtrip_bgr14() {
    let original = Bgr14::new(1000, 2000, 3000);
    let reconstructed = Bgr14::from_channels(original.to_channels().as_ref());
    assert_eq!(original, reconstructed);
}

#[test]
fn test_uniform_roundtrip_bgra10() {
    let original = Bgra10::new(100, 200, 300, 400);
    let reconstructed = Bgra10::from_channels(original.to_channels().as_ref());
    assert_eq!(original, reconstructed);
}

// --- DIM == CHANNEL_COUNT consistency ---

#[test]
fn test_uniform_dim_equals_channel_count() {
    assert_eq!(Rgb8::DIM, Rgb8::CHANNEL_COUNT);
    assert_eq!(Rgba8::DIM, Rgba8::CHANNEL_COUNT);
    assert_eq!(Bgr8::DIM, Bgr8::CHANNEL_COUNT);
    assert_eq!(Bgra8::DIM, Bgra8::CHANNEL_COUNT);
    assert_eq!(Rgb16::DIM, Rgb16::CHANNEL_COUNT);
    assert_eq!(Rgba16::DIM, Rgba16::CHANNEL_COUNT);
    assert_eq!(Rgb32::DIM, Rgb32::CHANNEL_COUNT);
    assert_eq!(Rgba32::DIM, Rgba32::CHANNEL_COUNT);
    assert_eq!(Rgb64::DIM, Rgb64::CHANNEL_COUNT);
    assert_eq!(Rgba64::DIM, Rgba64::CHANNEL_COUNT);
    assert_eq!(RgbF32::DIM, RgbF32::CHANNEL_COUNT);
    assert_eq!(RgbaF32::DIM, RgbaF32::CHANNEL_COUNT);
    assert_eq!(RgbF64::DIM, RgbF64::CHANNEL_COUNT);
    assert_eq!(RgbaF64::DIM, RgbaF64::CHANNEL_COUNT);
    assert_eq!(BgrF32::DIM, BgrF32::CHANNEL_COUNT);
    assert_eq!(BgraF32::DIM, BgraF32::CHANNEL_COUNT);
    assert_eq!(BgrF64::DIM, BgrF64::CHANNEL_COUNT);
    assert_eq!(BgraF64::DIM, BgraF64::CHANNEL_COUNT);
}

// --- out-of-bounds panics ---

#[test]
#[should_panic]
fn test_uniform_rgb8_channel_out_of_bounds() {
    let px = Rgb8::new(0, 0, 0);
    let _ = px.channel(3);
}

#[test]
#[should_panic]
fn test_uniform_rgba8_channel_out_of_bounds() {
    let px = Rgba8::new(0, 0, 0, 0);
    let _ = px.channel(4);
}

#[test]
#[should_panic]
fn test_uniform_mono8_channel_out_of_bounds() {
    let px = Mono8::new(0);
    let _ = px.channel(1);
}

#[test]
#[should_panic]
fn test_uniform_rgb8_set_channel_out_of_bounds() {
    let mut px = Rgb8::new(0, 0, 0);
    px.set_channel(3, Saturating(0));
}

#[test]
#[should_panic]
fn test_uniform_from_channels_wrong_count() {
    let ch = [Saturating(1u8), Saturating(2)];
    let _ = Rgb8::from_channels(&ch);
}

// -----------------------------------------------------------------------
// Mono AddAssign / SubAssign / MulAssign / DivAssign with references
// -----------------------------------------------------------------------

#[test]
fn test_mono_add_assign_ref() {
    let mut a = Mono10::new(100);
    let b = Mono10::new(200);
    a += &b;
    assert_eq!(a.value(), 300);
}

#[test]
fn test_mono_add_assign_u16_ref() {
    let mut a = Mono10::new(100);
    a += &200u16;
    assert_eq!(a.value(), 300);
}

#[test]
fn test_mono_sub_assign_ref() {
    let mut a = Mono10::new(300);
    let b = Mono10::new(100);
    a -= &b;
    assert_eq!(a.value(), 200);
}

#[test]
fn test_mono_sub_assign_u16_ref() {
    let mut a = Mono10::new(300);
    a -= &100u16;
    assert_eq!(a.value(), 200);
}

#[test]
fn test_mono_mul_assign_ref() {
    let mut a = Mono10::new(10);
    let b = Mono10::new(5);
    a *= &b;
    assert_eq!(a.value(), 50);
}

#[test]
fn test_mono_mul_assign_u16_ref() {
    let mut a = Mono10::new(10);
    a *= &5u16;
    assert_eq!(a.value(), 50);
}

#[test]
fn test_mono_div_assign_ref() {
    let mut a = Mono10::new(100);
    let b = Mono10::new(5);
    a /= &b;
    assert_eq!(a.value(), 20);
}

#[test]
fn test_mono_div_assign_u16_ref() {
    let mut a = Mono10::new(100);
    a /= &5u16;
    assert_eq!(a.value(), 20);
}

// -----------------------------------------------------------------------
// Mono8 Mul<f32>
// -----------------------------------------------------------------------

#[test]
fn test_mono8_mul_f32() {
    let a = Mono8::new(100);
    let b = a * 0.5f32;
    assert_eq!(b, Mono8::new(50));
}

#[test]
fn test_mono8_ref_mul_f32() {
    let a = Mono8::new(100);
    let b = &a * 0.5f32;
    assert_eq!(b, Mono8::new(50));
}

// -----------------------------------------------------------------------
// Generic Rgba<BITS> Add and LinearPixel::scale
// -----------------------------------------------------------------------

#[test]
fn test_rgba10_add() {
    let a = Rgba10::new(10, 20, 30, 40);
    let b = Rgba10::new(1, 2, 3, 4);
    let c = a + b;
    assert_eq!(c.r.value(), 11);
    assert_eq!(c.g.value(), 22);
    assert_eq!(c.b.value(), 33);
    assert_eq!(c.a.value(), 44);
}

#[test]
fn test_rgba10_linear_pixel_scale() {
    use crate::pixel::LinearPixel;
    let a = Rgba10::new(100, 200, 300, 400);
    let scaled = a.scale(0.5);
    assert_eq!(scaled.r, 50.0);
    assert_eq!(scaled.g, 100.0);
    assert_eq!(scaled.b, 150.0);
    assert_eq!(scaled.a, 200.0);
}

// ═══════════════════════════════════════════════════════════════════════
// MonoA pixel family tests
// ═══════════════════════════════════════════════════════════════════════

// ── constructors ────────────────────────────────────────────────────────

#[test]
fn test_monoa8_new() {
    let p = MonoA8::new(100, 200);
    assert_eq!(p.v, Saturating(100u8));
    assert_eq!(p.a, Saturating(200u8));
}

#[test]
fn test_monoa16_new() {
    let p = MonoA16::new(1000, 2000);
    assert_eq!(p.v, Saturating(1000u16));
    assert_eq!(p.a, Saturating(2000u16));
}

#[test]
fn test_monoa32_new() {
    let p = MonoA32::new(100_000, 200_000);
    assert_eq!(p.v, Saturating(100_000u32));
    assert_eq!(p.a, Saturating(200_000u32));
}

#[test]
fn test_monoa64_new() {
    let p = MonoA64::new(1_000_000, 2_000_000);
    assert_eq!(p.v, Saturating(1_000_000u64));
    assert_eq!(p.a, Saturating(2_000_000u64));
}

#[test]
fn test_monoaf32_new() {
    let p = MonoAF32::new(0.5, 1.0);
    assert_eq!(p.v, 0.5);
    assert_eq!(p.a, 1.0);
}

#[test]
fn test_monoaf64_new() {
    let p = MonoAF64::new(0.25, 0.75);
    assert_eq!(p.v, 0.25);
    assert_eq!(p.a, 0.75);
}

// ── zero ────────────────────────────────────────────────────────────────

#[test]
fn test_monoa8_zero() {
    let z = MonoA8::zero();
    assert_eq!(z.v, Saturating(0u8));
    assert_eq!(z.a, Saturating(0u8));
}

#[test]
fn test_monoa16_zero() {
    let z = MonoA16::zero();
    assert_eq!(z.v, Saturating(0u16));
    assert_eq!(z.a, Saturating(0u16));
}

#[test]
fn test_monoa32_zero() {
    let z = MonoA32::zero();
    assert_eq!(z.v, Saturating(0u32));
    assert_eq!(z.a, Saturating(0u32));
}

#[test]
fn test_monoa64_zero() {
    let z = MonoA64::zero();
    assert_eq!(z.v, Saturating(0u64));
    assert_eq!(z.a, Saturating(0u64));
}

#[test]
fn test_monoaf32_zero() {
    let z = MonoAF32::zero();
    assert_eq!(z.v, 0.0f32);
    assert_eq!(z.a, 0.0f32);
}

#[test]
fn test_monoaf64_zero() {
    let z = MonoAF64::zero();
    assert_eq!(z.v, 0.0f64);
    assert_eq!(z.a, 0.0f64);
}

// ── PlainPixel size / as_bytes ──────────────────────────────────────────

test_pixel_size!(test_monoa8_size, MonoA8);
test_pixel_size!(test_monoa16_size, MonoA16);
test_pixel_size!(test_monoa32_size, MonoA32);
test_pixel_size!(test_monoa64_size, MonoA64);
test_pixel_size!(test_monoaf32_size, MonoAF32);
test_pixel_size!(test_monoaf64_size, MonoAF64);

#[test]
fn test_monoa8_as_bytes() {
    let p = MonoA8::new(0xAB, 0xCD);
    assert_eq!(p.as_bytes(), &[0xAB, 0xCD]);
}

#[test]
fn test_monoa16_as_bytes() {
    let p = MonoA16::new(0x1234, 0x5678);
    let bytes = p.as_bytes();
    assert_eq!(bytes.len(), 4);
    // Verify round-trip
    let q = MonoA16::from_bytes(bytes).unwrap();
    assert_eq!(p, q);
}

#[test]
fn test_monoa8_from_bytes() {
    let p = MonoA8::from_bytes(&[42, 128]).unwrap();
    assert_eq!(p, MonoA8::new(42, 128));
}

#[test]
fn test_monoa16_from_bytes_roundtrip() {
    let orig = MonoA16::new(1000, 60000);
    let bytes = orig.as_bytes().to_vec();
    let restored = MonoA16::from_bytes(&bytes).unwrap();
    assert_eq!(orig, restored);
}

#[test]
fn test_monoa32_from_bytes_roundtrip() {
    let orig = MonoA32::new(123456, u32::MAX);
    let bytes = orig.as_bytes().to_vec();
    let restored = MonoA32::from_bytes(&bytes).unwrap();
    assert_eq!(orig, restored);
}

#[test]
fn test_monoa64_from_bytes_roundtrip() {
    let orig = MonoA64::new(u64::MAX / 2, u64::MAX);
    let bytes = orig.as_bytes().to_vec();
    let restored = MonoA64::from_bytes(&bytes).unwrap();
    assert_eq!(orig, restored);
}

#[test]
fn test_monoaf32_from_bytes_roundtrip() {
    let orig = MonoAF32::new(0.123, 0.987);
    let bytes = orig.as_bytes().to_vec();
    let restored = MonoAF32::from_bytes(&bytes).unwrap();
    assert_eq!(orig, restored);
}

#[test]
fn test_monoaf64_from_bytes_roundtrip() {
    let orig = MonoAF64::new(0.123456789, 0.987654321);
    let bytes = orig.as_bytes().to_vec();
    let restored = MonoAF64::from_bytes(&bytes).unwrap();
    assert_eq!(orig, restored);
}

#[test]
fn test_monoa8_as_bytes_le() {
    let p = MonoA8::new(10, 20);
    assert_eq!(&*p.as_bytes_le(), &[10, 20]);
}

#[test]
fn test_monoa8_as_bytes_be() {
    let p = MonoA8::new(10, 20);
    assert_eq!(&*p.as_bytes_be(), &[10, 20]);
}

// ── as_mut_bytes ────────────────────────────────────────────────────────

#[test]
fn test_monoa8_as_mut_bytes() {
    let mut p = MonoA8::new(0, 0);
    let bytes = p.as_mut_bytes();
    bytes[0] = 42;
    bytes[1] = 128;
    assert_eq!(p, MonoA8::new(42, 128));
}

// ── HomogeneousPixel ────────────────────────────────────────────────────────

#[test]
fn test_uniform_channel_count_monoa() {
    assert_eq!(MonoA8::CHANNEL_COUNT, 2);
    assert_eq!(MonoA16::CHANNEL_COUNT, 2);
    assert_eq!(MonoA32::CHANNEL_COUNT, 2);
    assert_eq!(MonoA64::CHANNEL_COUNT, 2);
    assert_eq!(MonoAF32::CHANNEL_COUNT, 2);
    assert_eq!(MonoAF64::CHANNEL_COUNT, 2);
}

#[test]
fn test_uniform_monoa8_channel_access() {
    let p = MonoA8::new(100, 200);
    assert_eq!(p.channel(0), Saturating(100u8));
    assert_eq!(p.channel(1), Saturating(200u8));
}

#[test]
fn test_uniform_monoa16_channel_access() {
    let p = MonoA16::new(1000, 2000);
    assert_eq!(p.channel(0), Saturating(1000u16));
    assert_eq!(p.channel(1), Saturating(2000u16));
}

#[test]
fn test_uniform_monoaf32_channel_access() {
    let p = MonoAF32::new(0.5, 1.0);
    assert_eq!(p.channel(0), 0.5f32);
    assert_eq!(p.channel(1), 1.0f32);
}

#[test]
fn test_uniform_monoaf64_channel_access() {
    let p = MonoAF64::new(0.25, 0.75);
    assert_eq!(p.channel(0), 0.25f64);
    assert_eq!(p.channel(1), 0.75f64);
}

#[test]
fn test_uniform_monoa8_to_channels() {
    let p = MonoA8::new(10, 20);
    let ch = p.to_channels();
    assert_eq!(ch, [Saturating(10u8), Saturating(20u8)]);
}

#[test]
fn test_uniform_monoaf32_to_channels() {
    let p = MonoAF32::new(0.1, 0.9);
    let ch = p.to_channels();
    assert_eq!(ch, [0.1f32, 0.9f32]);
}

#[test]
fn test_uniform_monoa8_from_channels() {
    let ch = [Saturating(42u8), Saturating(128u8)];
    let p = MonoA8::from_channels(&ch);
    assert_eq!(p, MonoA8::new(42, 128));
}

#[test]
fn test_uniform_monoa16_from_channels() {
    let ch = [Saturating(1000u16), Saturating(60000u16)];
    let p = MonoA16::from_channels(&ch);
    assert_eq!(p, MonoA16::new(1000, 60000));
}

#[test]
fn test_uniform_monoaf32_from_channels() {
    let ch = [0.5f32, 1.0f32];
    let p = MonoAF32::from_channels(&ch);
    assert_eq!(p, MonoAF32::new(0.5, 1.0));
}

#[test]
fn test_uniform_monoa8_set_channel() {
    let mut p = MonoA8::new(0, 0);
    p.set_channel(0, Saturating(42u8));
    p.set_channel(1, Saturating(200u8));
    assert_eq!(p, MonoA8::new(42, 200));
}

#[test]
fn test_uniform_monoaf64_set_channel() {
    let mut p = MonoAF64::new(0.0, 0.0);
    p.set_channel(0, 0.5f64);
    p.set_channel(1, 0.75f64);
    assert_eq!(p, MonoAF64::new(0.5, 0.75));
}

#[test]
fn test_uniform_roundtrip_monoa8() {
    let orig = MonoA8::new(42, 200);
    let ch = orig.to_channels();
    assert_eq!(MonoA8::from_channels(&ch), orig);
}

#[test]
fn test_uniform_roundtrip_monoa16() {
    let orig = MonoA16::new(1000, 60000);
    let ch = orig.to_channels();
    assert_eq!(MonoA16::from_channels(&ch), orig);
}

#[test]
fn test_uniform_roundtrip_monoa32() {
    let orig = MonoA32::new(100_000, u32::MAX);
    let ch = orig.to_channels();
    assert_eq!(MonoA32::from_channels(&ch), orig);
}

#[test]
fn test_uniform_roundtrip_monoa64() {
    let orig = MonoA64::new(u64::MAX / 2, u64::MAX);
    let ch = orig.to_channels();
    assert_eq!(MonoA64::from_channels(&ch), orig);
}

#[test]
fn test_uniform_roundtrip_monoaf32() {
    let orig = MonoAF32::new(0.123, 0.987);
    let ch = orig.to_channels();
    assert_eq!(MonoAF32::from_channels(&ch), orig);
}

#[test]
fn test_uniform_roundtrip_monoaf64() {
    let orig = MonoAF64::new(0.123456789, 0.987654321);
    let ch = orig.to_channels();
    assert_eq!(MonoAF64::from_channels(&ch), orig);
}

#[test]
#[should_panic]
fn test_uniform_monoa8_channel_out_of_bounds() {
    let p = MonoA8::new(0, 0);
    let _ = p.channel(2);
}

#[test]
#[should_panic]
fn test_uniform_monoa8_set_channel_out_of_bounds() {
    let mut p = MonoA8::new(0, 0);
    p.set_channel(2, Saturating(0u8));
}

#[test]
#[should_panic]
fn test_uniform_monoa8_from_channels_wrong_count() {
    let ch = [Saturating(1u8)];
    let _ = MonoA8::from_channels(&ch);
}

// ── HomogeneousPixel size assertions ────────────────────────────────────────

#[test]
fn test_uniform_size_assert_monoa() {
    // HomogeneousPixel requires size == channel_size * channel_count.
    // These will fail to compile if the assertion is violated, but
    // we verify at runtime too.
    assert_eq!(MonoA8::SIZE, 2);
    assert_eq!(MonoA16::SIZE, 4);
    assert_eq!(MonoA32::SIZE, 8);
    assert_eq!(MonoA64::SIZE, 16);
    assert_eq!(MonoAF32::SIZE, 8);
    assert_eq!(MonoAF64::SIZE, 16);
}

// ── DIM (channel count from PlainPixel) ─────────────────────────────────

#[test]
fn test_monoa_dim() {
    assert_eq!(MonoA8::DIM, 2);
    assert_eq!(MonoA16::DIM, 2);
    assert_eq!(MonoA32::DIM, 2);
    assert_eq!(MonoA64::DIM, 2);
    assert_eq!(MonoAF32::DIM, 2);
    assert_eq!(MonoAF64::DIM, 2);
}

// ── alignment ───────────────────────────────────────────────────────────

#[test]
fn test_align_monoa8() {
    assert_eq!(MonoA8::ALIGN, 1);
}

#[test]
fn test_align_monoa16() {
    assert_eq!(MonoA16::ALIGN, 2);
}

#[test]
fn test_align_monoa32() {
    assert_eq!(MonoA32::ALIGN, 4);
}

#[test]
fn test_align_monoaf32() {
    assert_eq!(MonoAF32::ALIGN, 4);
}

#[test]
fn test_align_monoaf64() {
    assert_eq!(MonoAF64::ALIGN, 8);
}

// ── LinearPixel ─────────────────────────────────────────────────────────

#[test]
fn test_linear_pixel_monoa8() {
    let p = MonoA8::new(200, 100);
    let scaled = p.scale(0.5);
    assert_eq!(scaled.v, 100.0);
    assert_eq!(scaled.a, 50.0);
}

#[test]
fn test_linear_pixel_monoa16() {
    let p = MonoA16::new(1000, 500);
    let scaled = p.scale(0.5);
    assert_eq!(scaled.v, 500.0);
    assert_eq!(scaled.a, 250.0);
}

#[test]
fn test_linear_pixel_monoa32() {
    let p = MonoA32::new(1000, 500);
    let scaled = p.scale(0.5);
    assert_eq!(scaled.v, 500.0);
    assert_eq!(scaled.a, 250.0);
}

#[test]
fn test_linear_pixel_monoa64() {
    let p = MonoA64::new(1000, 500);
    let scaled = p.scale(0.5);
    assert_eq!(scaled.v, 500.0);
    assert_eq!(scaled.a, 250.0);
}

#[test]
fn test_linear_pixel_monoaf32() {
    let p = MonoAF32::new(0.8, 0.4);
    let scaled = p.scale(0.5);
    assert!((scaled.v - 0.4).abs() < 1e-6);
    assert!((scaled.a - 0.2).abs() < 1e-6);
}

#[test]
fn test_linear_pixel_monoaf64() {
    let p = MonoAF64::new(0.8, 0.4);
    let scaled = p.scale(0.5);
    assert!((scaled.v - 0.4).abs() < 1e-12);
    assert!((scaled.a - 0.2).abs() < 1e-12);
}

// ── blend ───────────────────────────────────────────────────────────────

#[test]
fn test_blend_monoa8() {
    use crate::pixel::blend;
    let a = MonoA8::new(0, 0);
    let b = MonoA8::new(200, 100);
    let mid = blend(&a, &b, 0.5);
    assert!((mid.v - 100.0).abs() < 1.0);
    assert!((mid.a - 50.0).abs() < 1.0);
}

#[test]
fn test_blend_monoaf32() {
    use crate::pixel::blend;
    let a = MonoAF32::new(0.0, 0.0);
    let b = MonoAF32::new(1.0, 0.5);
    let mid = blend(&a, &b, 0.5);
    assert!((mid.v - 0.5).abs() < 1e-6);
    assert!((mid.a - 0.25).abs() < 1e-6);
}

// ── Copy / Clone / Debug / PartialEq ────────────────────────────────────

#[test]
fn test_monoa8_copy_clone_eq() {
    let p = MonoA8::new(42, 128);
    let q = p; // Copy
    let r = p.clone(); // Clone
    assert_eq!(p, q);
    assert_eq!(p, r);
}

#[test]
fn test_monoa8_ne() {
    assert_ne!(MonoA8::new(1, 2), MonoA8::new(1, 3));
    assert_ne!(MonoA8::new(1, 2), MonoA8::new(2, 2));
}

#[test]
fn test_monoa8_debug() {
    let p = MonoA8::new(10, 20);
    let s = format!("{:?}", p);
    assert!(s.contains("MonoA8"));
}

#[test]
fn test_monoaf32_copy_clone_eq() {
    let p = MonoAF32::new(0.5, 1.0);
    let q = p;
    let r = p.clone();
    assert_eq!(p, q);
    assert_eq!(p, r);
}

#[test]
fn test_monoaf64_debug() {
    let p = MonoAF64::new(0.1, 0.2);
    let s = format!("{:?}", p);
    assert!(s.contains("MonoAF64"));
}

// ── channel_sum (via HomogeneousPixel) ──────────────────────────────────────

#[test]
fn test_channel_sum_monoa8() {
    let p = MonoA8::new(100, 200);
    let sum: u16 = (0..MonoA8::CHANNEL_COUNT)
        .map(|i| p.channel(i).0 as u16)
        .sum();
    assert_eq!(sum, 300);
}

#[test]
fn test_channel_sum_monoaf32() {
    let p = MonoAF32::new(0.3, 0.7);
    let sum: f32 = (0..MonoAF32::CHANNEL_COUNT).map(|i| p.channel(i)).sum();
    assert!((sum - 1.0).abs() < 1e-6);
}

// ── Indexed8 ────────────────────────────────────────────────────────────

#[test]
fn test_indexed8_new_and_value() {
    let p = Indexed8(42);
    assert_eq!(p.0, 42);
}

#[test]
fn test_indexed8_zero() {
    let p = Indexed8::zero();
    assert_eq!(p, Indexed8(0));
}

#[test]
fn test_indexed8_plain_pixel_size() {
    assert_eq!(Indexed8::SIZE, 1);
    assert_eq!(Indexed8::DIM, 1);
    assert_eq!(Indexed8::CHANNELS, &[1]);
}

#[test]
fn test_indexed8_as_bytes_roundtrip() {
    let p = Indexed8(200);
    let bytes = p.as_bytes();
    assert_eq!(bytes, &[200]);
    let q = Indexed8::from_bytes(bytes).unwrap();
    assert_eq!(p, q);
}

#[test]
fn test_indexed8_as_bytes_le() {
    let p = Indexed8(42);
    assert_eq!(&*p.as_bytes_le(), &[42]);
}

#[test]
fn test_indexed8_as_bytes_be() {
    let p = Indexed8(42);
    assert_eq!(&*p.as_bytes_be(), &[42]);
}

#[test]
fn test_indexed8_as_mut_bytes() {
    let mut p = Indexed8(10);
    p.as_mut_bytes()[0] = 99;
    assert_eq!(p, Indexed8(99));
}

#[test]
fn test_indexed8_uniform_channels() {
    let p = Indexed8(42);
    assert_eq!(Indexed8::CHANNEL_COUNT, 1);
    assert_eq!(p.channel(0), 42u8);
}

#[test]
fn test_indexed8_uniform_to_channels() {
    let p = Indexed8(77);
    let ch = p.to_channels();
    assert_eq!(ch, [77u8]);
}

#[test]
fn test_indexed8_uniform_from_channels() {
    let p = Indexed8::from_channels(&[55u8]);
    assert_eq!(p, Indexed8(55));
}

#[test]
fn test_indexed8_uniform_set_channel() {
    let mut p = Indexed8(10);
    p.set_channel(0, 200u8);
    assert_eq!(p, Indexed8(200));
}

#[test]
fn test_indexed8_uniform_roundtrip() {
    let p = Indexed8(123);
    let ch = p.to_channels();
    let q = Indexed8::from_channels(ch.as_ref());
    assert_eq!(p, q);
}

#[test]
#[should_panic]
fn test_indexed8_channel_out_of_bounds() {
    let p = Indexed8(0);
    let _ = p.channel(1);
}

#[test]
#[should_panic]
fn test_indexed8_set_channel_out_of_bounds() {
    let mut p = Indexed8(0);
    p.set_channel(1, 0u8);
}

#[test]
#[should_panic]
fn test_indexed8_from_channels_wrong_count() {
    let _ = Indexed8::from_channels(&[1u8, 2u8]);
}

#[test]
fn test_indexed8_copy_clone_eq() {
    let p = Indexed8(42);
    let q = p; // Copy
    let r = p.clone(); // Clone
    assert_eq!(p, q);
    assert_eq!(p, r);
}

#[test]
fn test_indexed8_ne() {
    assert_ne!(Indexed8(0), Indexed8(1));
    assert_ne!(Indexed8(255), Indexed8(0));
}

#[test]
fn test_indexed8_debug() {
    let p = Indexed8(42);
    let s = format!("{:?}", p);
    assert!(s.contains("Indexed8"));
}

#[test]
fn test_indexed8_extremes() {
    let lo = Indexed8(0);
    let hi = Indexed8(255);
    assert_eq!(lo.0, 0);
    assert_eq!(hi.0, 255);
}

#[test]
fn test_align_indexed8() {
    assert_eq!(Indexed8::ALIGN, 1);
}

#[test]
fn test_channel_sum_indexed8() {
    let p = Indexed8(42);
    let sum: u8 = (0..Indexed8::CHANNEL_COUNT).map(|i| p.channel(i)).sum();
    assert_eq!(sum, 42);
}

#[test]
fn test_indexed8_uniform_size_assert() {
    // SIZE must equal CHANNEL_COUNT * size_of::<Channel>()
    assert_eq!(
        Indexed8::SIZE,
        Indexed8::CHANNEL_COUNT * std::mem::size_of::<u8>()
    );
}

#[test]
fn test_indexed8_dim_equals_channel_count() {
    assert_eq!(Indexed8::DIM, Indexed8::CHANNEL_COUNT);
}

// ─── SrgbMono8 ──────────────────────────────────────────────────────────

#[test]
fn test_srgb_mono8_new() {
    let p = SrgbMono8::new(128);
    assert_eq!(p.0.0, 128);
}

#[test]
fn test_srgb_mono8_new_extremes() {
    assert_eq!(SrgbMono8::new(0).0.0, 0);
    assert_eq!(SrgbMono8::new(255).0.0, 255);
}

#[test]
fn test_srgb_mono8_zero() {
    let z = SrgbMono8::zero();
    assert_eq!(z.0.0, 0);
}

#[test]
fn test_srgb_mono8_plain_pixel_size() {
    assert_eq!(SrgbMono8::SIZE, 1);
    assert_eq!(SrgbMono8::ALIGN, 1);
}

#[test]
fn test_srgb_mono8_as_bytes_roundtrip() {
    let p = SrgbMono8::new(42);
    let bytes = p.as_bytes();
    assert_eq!(bytes, &[42]);
    let q = SrgbMono8::from_bytes(bytes).unwrap();
    assert_eq!(p, q);
}

#[test]
fn test_srgb_mono8_as_bytes_le() {
    let p = SrgbMono8::new(200);
    assert_eq!(&*p.as_bytes_le(), &[200]);
}

#[test]
fn test_srgb_mono8_as_bytes_be() {
    let p = SrgbMono8::new(200);
    assert_eq!(&*p.as_bytes_be(), &[200]);
}

#[test]
fn test_srgb_mono8_as_mut_bytes() {
    let mut p = SrgbMono8::new(10);
    p.as_mut_bytes()[0] = 99;
    assert_eq!(p.0.0, 99);
}

#[test]
fn test_srgb_mono8_uniform_channels() {
    assert_eq!(SrgbMono8::CHANNEL_COUNT, 1);
    let p = SrgbMono8::new(77);
    assert_eq!(p.channel(0), Saturating(77u8));
}

#[test]
fn test_srgb_mono8_uniform_to_channels() {
    let p = SrgbMono8::new(55);
    assert_eq!(p.to_channels(), [Saturating(55u8)]);
}

#[test]
fn test_srgb_mono8_uniform_from_channels() {
    let p = SrgbMono8::from_channels(&[Saturating(88u8)]);
    assert_eq!(p.0.0, 88);
}

#[test]
fn test_srgb_mono8_uniform_set_channel() {
    let mut p = SrgbMono8::new(0);
    p.set_channel(0, Saturating(123u8));
    assert_eq!(p.0.0, 123);
}

#[test]
fn test_srgb_mono8_uniform_roundtrip() {
    let orig = SrgbMono8::new(200);
    let channels = orig.to_channels();
    let restored = SrgbMono8::from_channels(&channels);
    assert_eq!(orig, restored);
}

#[test]
fn test_srgb_mono8_uniform_size_assert() {
    assert_eq!(
        SrgbMono8::SIZE,
        SrgbMono8::CHANNEL_COUNT * std::mem::size_of::<u8>()
    );
}

#[test]
fn test_srgb_mono8_dim_equals_channel_count() {
    assert_eq!(SrgbMono8::DIM, SrgbMono8::CHANNEL_COUNT);
}

#[test]
#[should_panic]
fn test_srgb_mono8_channel_out_of_bounds() {
    SrgbMono8::new(0).channel(1);
}

#[test]
#[should_panic]
fn test_srgb_mono8_set_channel_out_of_bounds() {
    let mut p = SrgbMono8::new(0);
    p.set_channel(1, Saturating(0u8));
}

#[test]
#[should_panic]
fn test_srgb_mono8_from_channels_wrong_count() {
    SrgbMono8::from_channels(&[Saturating(1u8), Saturating(2u8)]);
}

#[test]
fn test_srgb_mono8_copy_clone_eq() {
    let a = SrgbMono8::new(42);
    let b = a;
    let c = a.clone();
    assert_eq!(a, b);
    assert_eq!(a, c);
}

#[test]
fn test_srgb_mono8_ne() {
    assert_ne!(SrgbMono8::new(0), SrgbMono8::new(1));
}

#[test]
fn test_srgb_mono8_debug() {
    let p = SrgbMono8::new(42);
    let s = format!("{:?}", p);
    assert!(s.contains("SrgbMono8"));
}

#[test]
fn test_align_srgb_mono8() {
    assert_eq!(SrgbMono8::ALIGN, 1);
}

#[test]
fn test_channel_sum_srgb_mono8() {
    let sum: usize = SrgbMono8::CHANNELS.iter().sum();
    assert_eq!(SrgbMono8::SIZE, sum);
}

// ─── SrgbMonoA8 ─────────────────────────────────────────────────────────

#[test]
fn test_srgb_mono_a8_new() {
    let p = SrgbMonoA8::new(100, 200);
    assert_eq!(p.v.0, 100);
    assert_eq!(p.a.0, 200);
}

#[test]
fn test_srgb_mono_a8_new_extremes() {
    let lo = SrgbMonoA8::new(0, 0);
    let hi = SrgbMonoA8::new(255, 255);
    assert_eq!(lo.v.0, 0);
    assert_eq!(lo.a.0, 0);
    assert_eq!(hi.v.0, 255);
    assert_eq!(hi.a.0, 255);
}

#[test]
fn test_srgb_mono_a8_zero() {
    let z = SrgbMonoA8::zero();
    assert_eq!(z.v.0, 0);
    assert_eq!(z.a.0, 0);
}

#[test]
fn test_srgb_mono_a8_plain_pixel_size() {
    assert_eq!(SrgbMonoA8::SIZE, 2);
    assert_eq!(SrgbMonoA8::ALIGN, 1);
}

#[test]
fn test_srgb_mono_a8_as_bytes_roundtrip() {
    let p = SrgbMonoA8::new(42, 99);
    let bytes = p.as_bytes();
    assert_eq!(bytes, &[42, 99]);
    let q = SrgbMonoA8::from_bytes(bytes).unwrap();
    assert_eq!(p, q);
}

#[test]
fn test_srgb_mono_a8_as_bytes_le() {
    let p = SrgbMonoA8::new(100, 200);
    assert_eq!(&*p.as_bytes_le(), &[100, 200]);
}

#[test]
fn test_srgb_mono_a8_as_bytes_be() {
    let p = SrgbMonoA8::new(100, 200);
    assert_eq!(&*p.as_bytes_be(), &[100, 200]);
}

#[test]
fn test_srgb_mono_a8_as_mut_bytes() {
    let mut p = SrgbMonoA8::new(10, 20);
    p.as_mut_bytes()[0] = 99;
    p.as_mut_bytes()[1] = 88;
    assert_eq!(p.v.0, 99);
    assert_eq!(p.a.0, 88);
}

#[test]
fn test_srgb_mono_a8_uniform_channels() {
    assert_eq!(SrgbMonoA8::CHANNEL_COUNT, 2);
    let p = SrgbMonoA8::new(77, 33);
    assert_eq!(p.channel(0), Saturating(77u8));
    assert_eq!(p.channel(1), Saturating(33u8));
}

#[test]
fn test_srgb_mono_a8_uniform_to_channels() {
    let p = SrgbMonoA8::new(55, 44);
    assert_eq!(p.to_channels(), [Saturating(55u8), Saturating(44u8)]);
}

#[test]
fn test_srgb_mono_a8_uniform_from_channels() {
    let p = SrgbMonoA8::from_channels(&[Saturating(88u8), Saturating(77u8)]);
    assert_eq!(p.v.0, 88);
    assert_eq!(p.a.0, 77);
}

#[test]
fn test_srgb_mono_a8_uniform_set_channel() {
    let mut p = SrgbMonoA8::new(0, 0);
    p.set_channel(0, Saturating(123u8));
    p.set_channel(1, Saturating(45u8));
    assert_eq!(p.v.0, 123);
    assert_eq!(p.a.0, 45);
}

#[test]
fn test_srgb_mono_a8_uniform_roundtrip() {
    let orig = SrgbMonoA8::new(200, 150);
    let channels = orig.to_channels();
    let restored = SrgbMonoA8::from_channels(&channels);
    assert_eq!(orig, restored);
}

#[test]
fn test_srgb_mono_a8_uniform_size_assert() {
    assert_eq!(
        SrgbMonoA8::SIZE,
        SrgbMonoA8::CHANNEL_COUNT * std::mem::size_of::<u8>()
    );
}

#[test]
fn test_srgb_mono_a8_dim_equals_channel_count() {
    assert_eq!(SrgbMonoA8::DIM, SrgbMonoA8::CHANNEL_COUNT);
}

#[test]
#[should_panic]
fn test_srgb_mono_a8_channel_out_of_bounds() {
    SrgbMonoA8::new(0, 0).channel(2);
}

#[test]
#[should_panic]
fn test_srgb_mono_a8_set_channel_out_of_bounds() {
    let mut p = SrgbMonoA8::new(0, 0);
    p.set_channel(2, Saturating(0u8));
}

#[test]
#[should_panic]
fn test_srgb_mono_a8_from_channels_wrong_count() {
    SrgbMonoA8::from_channels(&[Saturating(1u8)]);
}

#[test]
fn test_srgb_mono_a8_copy_clone_eq() {
    let a = SrgbMonoA8::new(42, 99);
    let b = a;
    let c = a.clone();
    assert_eq!(a, b);
    assert_eq!(a, c);
}

#[test]
fn test_srgb_mono_a8_ne() {
    assert_ne!(SrgbMonoA8::new(0, 0), SrgbMonoA8::new(1, 0));
    assert_ne!(SrgbMonoA8::new(0, 0), SrgbMonoA8::new(0, 1));
}

#[test]
fn test_srgb_mono_a8_debug() {
    let p = SrgbMonoA8::new(42, 99);
    let s = format!("{:?}", p);
    assert!(s.contains("SrgbMonoA8"));
}

#[test]
fn test_align_srgb_mono_a8() {
    assert_eq!(SrgbMonoA8::ALIGN, 1);
}

#[test]
fn test_channel_sum_srgb_mono_a8() {
    let sum: usize = SrgbMonoA8::CHANNELS.iter().sum();
    assert_eq!(SrgbMonoA8::SIZE, sum);
}

// ─── Srgb16 ─────────────────────────────────────────────────────────

#[test]
fn test_srgb16_new() {
    let p = Srgb16::new(1000, 2000, 3000);
    assert_eq!(p.r.0, 1000);
    assert_eq!(p.g.0, 2000);
    assert_eq!(p.b.0, 3000);
}

#[test]
fn test_srgb16_new_extremes() {
    let lo = Srgb16::new(0, 0, 0);
    let hi = Srgb16::new(65535, 65535, 65535);
    assert_eq!(lo.r.0, 0);
    assert_eq!(hi.r.0, 65535);
    assert_eq!(hi.g.0, 65535);
    assert_eq!(hi.b.0, 65535);
}

#[test]
fn test_srgb16_zero() {
    let z = Srgb16::zero();
    assert_eq!(z.r.0, 0);
    assert_eq!(z.g.0, 0);
    assert_eq!(z.b.0, 0);
}

#[test]
fn test_srgb16_plain_pixel_size() {
    assert_eq!(Srgb16::SIZE, 6);
    assert_eq!(Srgb16::DIM, 3);
    assert_eq!(Srgb16::CHANNELS, &[2, 2, 2]);
}

#[test]
fn test_srgb16_as_bytes_roundtrip() {
    let p = Srgb16::new(0x1234, 0x5678, 0x9ABC);
    let bytes = p.as_bytes();
    assert_eq!(bytes.len(), 6);
    let q = Srgb16::from_bytes(bytes).unwrap();
    assert_eq!(p, q);
}

#[test]
fn test_srgb16_as_bytes_le() {
    let p = Srgb16::new(0x0102, 0x0304, 0x0506);
    let le = p.as_bytes_le();
    assert_eq!(le.len(), 6);
}

#[test]
fn test_srgb16_as_bytes_be() {
    let p = Srgb16::new(0x0102, 0x0304, 0x0506);
    let be = p.as_bytes_be();
    assert_eq!(be.len(), 6);
}

#[test]
fn test_srgb16_as_mut_bytes() {
    let mut p = Srgb16::new(0, 0, 0);
    let bytes = p.as_mut_bytes();
    // Write native-endian u16 for r channel
    let val: u16 = 42;
    bytes[0..2].copy_from_slice(&val.to_ne_bytes());
    assert_eq!(p.r.0, 42);
}

#[test]
fn test_srgb16_uniform_channels() {
    assert_eq!(Srgb16::CHANNEL_COUNT, 3);
    let p = Srgb16::new(100, 200, 300);
    assert_eq!(p.channel(0), Saturating(100u16));
    assert_eq!(p.channel(1), Saturating(200u16));
    assert_eq!(p.channel(2), Saturating(300u16));
}

#[test]
fn test_srgb16_uniform_to_channels() {
    let p = Srgb16::new(10, 20, 30);
    assert_eq!(
        p.to_channels(),
        [Saturating(10u16), Saturating(20u16), Saturating(30u16)]
    );
}

#[test]
fn test_srgb16_uniform_from_channels() {
    let p = Srgb16::from_channels(&[Saturating(11u16), Saturating(22u16), Saturating(33u16)]);
    assert_eq!(p.r.0, 11);
    assert_eq!(p.g.0, 22);
    assert_eq!(p.b.0, 33);
}

#[test]
fn test_srgb16_uniform_set_channel() {
    let mut p = Srgb16::new(0, 0, 0);
    p.set_channel(0, Saturating(111u16));
    p.set_channel(1, Saturating(222u16));
    p.set_channel(2, Saturating(333u16));
    assert_eq!(p, Srgb16::new(111, 222, 333));
}

#[test]
fn test_srgb16_uniform_roundtrip() {
    let orig = Srgb16::new(12345, 23456, 34567);
    let ch = orig.to_channels();
    let restored = Srgb16::from_channels(&ch);
    assert_eq!(orig, restored);
}

#[test]
fn test_srgb16_uniform_size_assert() {
    assert_eq!(
        Srgb16::SIZE,
        Srgb16::CHANNEL_COUNT * std::mem::size_of::<u16>()
    );
}

#[test]
fn test_srgb16_dim_equals_channel_count() {
    assert_eq!(Srgb16::DIM, Srgb16::CHANNEL_COUNT);
}

#[test]
#[should_panic]
fn test_srgb16_channel_out_of_bounds() {
    Srgb16::new(0, 0, 0).channel(3);
}

#[test]
#[should_panic]
fn test_srgb16_set_channel_out_of_bounds() {
    let mut p = Srgb16::new(0, 0, 0);
    p.set_channel(3, Saturating(0u16));
}

#[test]
#[should_panic]
fn test_srgb16_from_channels_wrong_count() {
    Srgb16::from_channels(&[Saturating(1u16), Saturating(2u16)]);
}

#[test]
fn test_srgb16_copy_clone_eq() {
    let a = Srgb16::new(100, 200, 300);
    let b = a;
    let c = a.clone();
    assert_eq!(a, b);
    assert_eq!(a, c);
}

#[test]
fn test_srgb16_ne() {
    assert_ne!(Srgb16::new(0, 0, 0), Srgb16::new(1, 0, 0));
    assert_ne!(Srgb16::new(0, 0, 0), Srgb16::new(0, 1, 0));
    assert_ne!(Srgb16::new(0, 0, 0), Srgb16::new(0, 0, 1));
}

#[test]
fn test_srgb16_debug() {
    let p = Srgb16::new(1, 2, 3);
    let s = format!("{:?}", p);
    assert!(s.contains("Srgb16"));
}

#[test]
fn test_align_srgb16() {
    assert_eq!(Srgb16::ALIGN, 2);
}

#[test]
fn test_channel_sum_srgb16() {
    let sum: usize = Srgb16::CHANNELS.iter().sum();
    assert_eq!(Srgb16::SIZE, sum);
}

// ─── Srgba16 ────────────────────────────────────────────────────────

#[test]
fn test_srgba16_new() {
    let p = Srgba16::new(1000, 2000, 3000, 4000);
    assert_eq!(p.r.0, 1000);
    assert_eq!(p.g.0, 2000);
    assert_eq!(p.b.0, 3000);
    assert_eq!(p.a.0, 4000);
}

#[test]
fn test_srgba16_new_extremes() {
    let lo = Srgba16::new(0, 0, 0, 0);
    let hi = Srgba16::new(65535, 65535, 65535, 65535);
    assert_eq!(lo.r.0, 0);
    assert_eq!(lo.a.0, 0);
    assert_eq!(hi.r.0, 65535);
    assert_eq!(hi.g.0, 65535);
    assert_eq!(hi.b.0, 65535);
    assert_eq!(hi.a.0, 65535);
}

#[test]
fn test_srgba16_zero() {
    let z = Srgba16::zero();
    assert_eq!(z.r.0, 0);
    assert_eq!(z.g.0, 0);
    assert_eq!(z.b.0, 0);
    assert_eq!(z.a.0, 0);
}

#[test]
fn test_srgba16_plain_pixel_size() {
    assert_eq!(Srgba16::SIZE, 8);
    assert_eq!(Srgba16::DIM, 4);
    assert_eq!(Srgba16::CHANNELS, &[2, 2, 2, 2]);
}

#[test]
fn test_srgba16_as_bytes_roundtrip() {
    let p = Srgba16::new(0x1234, 0x5678, 0x9ABC, 0xDEF0);
    let bytes = p.as_bytes();
    assert_eq!(bytes.len(), 8);
    let q = Srgba16::from_bytes(bytes).unwrap();
    assert_eq!(p, q);
}

#[test]
fn test_srgba16_as_bytes_le() {
    let p = Srgba16::new(1, 2, 3, 4);
    let le = p.as_bytes_le();
    assert_eq!(le.len(), 8);
}

#[test]
fn test_srgba16_as_bytes_be() {
    let p = Srgba16::new(1, 2, 3, 4);
    let be = p.as_bytes_be();
    assert_eq!(be.len(), 8);
}

#[test]
fn test_srgba16_as_mut_bytes() {
    let mut p = Srgba16::new(0, 0, 0, 0);
    let bytes = p.as_mut_bytes();
    let val: u16 = 42;
    bytes[0..2].copy_from_slice(&val.to_ne_bytes());
    assert_eq!(p.r.0, 42);
}

#[test]
fn test_srgba16_uniform_channels() {
    assert_eq!(Srgba16::CHANNEL_COUNT, 4);
    let p = Srgba16::new(10, 20, 30, 40);
    assert_eq!(p.channel(0), Saturating(10u16));
    assert_eq!(p.channel(1), Saturating(20u16));
    assert_eq!(p.channel(2), Saturating(30u16));
    assert_eq!(p.channel(3), Saturating(40u16));
}

#[test]
fn test_srgba16_uniform_to_channels() {
    let p = Srgba16::new(1, 2, 3, 4);
    assert_eq!(
        p.to_channels(),
        [
            Saturating(1u16),
            Saturating(2u16),
            Saturating(3u16),
            Saturating(4u16)
        ]
    );
}

#[test]
fn test_srgba16_uniform_from_channels() {
    let p = Srgba16::from_channels(&[
        Saturating(11u16),
        Saturating(22u16),
        Saturating(33u16),
        Saturating(44u16),
    ]);
    assert_eq!(p.r.0, 11);
    assert_eq!(p.g.0, 22);
    assert_eq!(p.b.0, 33);
    assert_eq!(p.a.0, 44);
}

#[test]
fn test_srgba16_uniform_set_channel() {
    let mut p = Srgba16::new(0, 0, 0, 0);
    p.set_channel(0, Saturating(111u16));
    p.set_channel(1, Saturating(222u16));
    p.set_channel(2, Saturating(333u16));
    p.set_channel(3, Saturating(444u16));
    assert_eq!(p, Srgba16::new(111, 222, 333, 444));
}

#[test]
fn test_srgba16_uniform_roundtrip() {
    let orig = Srgba16::new(12345, 23456, 34567, 45678);
    let ch = orig.to_channels();
    let restored = Srgba16::from_channels(&ch);
    assert_eq!(orig, restored);
}

#[test]
fn test_srgba16_uniform_size_assert() {
    assert_eq!(
        Srgba16::SIZE,
        Srgba16::CHANNEL_COUNT * std::mem::size_of::<u16>()
    );
}

#[test]
fn test_srgba16_dim_equals_channel_count() {
    assert_eq!(Srgba16::DIM, Srgba16::CHANNEL_COUNT);
}

#[test]
#[should_panic]
fn test_srgba16_channel_out_of_bounds() {
    Srgba16::new(0, 0, 0, 0).channel(4);
}

#[test]
#[should_panic]
fn test_srgba16_set_channel_out_of_bounds() {
    let mut p = Srgba16::new(0, 0, 0, 0);
    p.set_channel(4, Saturating(0u16));
}

#[test]
#[should_panic]
fn test_srgba16_from_channels_wrong_count() {
    Srgba16::from_channels(&[Saturating(1u16), Saturating(2u16), Saturating(3u16)]);
}

#[test]
fn test_srgba16_copy_clone_eq() {
    let a = Srgba16::new(10, 20, 30, 40);
    let b = a;
    let c = a.clone();
    assert_eq!(a, b);
    assert_eq!(a, c);
}

#[test]
fn test_srgba16_ne() {
    assert_ne!(Srgba16::new(0, 0, 0, 0), Srgba16::new(1, 0, 0, 0));
    assert_ne!(Srgba16::new(0, 0, 0, 0), Srgba16::new(0, 0, 0, 1));
}

#[test]
fn test_srgba16_debug() {
    let p = Srgba16::new(1, 2, 3, 4);
    let s = format!("{:?}", p);
    assert!(s.contains("Srgba16"));
}

#[test]
fn test_align_srgba16() {
    assert_eq!(Srgba16::ALIGN, 2);
}

#[test]
fn test_channel_sum_srgba16() {
    let sum: usize = Srgba16::CHANNELS.iter().sum();
    assert_eq!(Srgba16::SIZE, sum);
}

// ─── SrgbMono16 ─────────────────────────────────────────────────────

#[test]
fn test_srgb_mono16_new() {
    let p = SrgbMono16::new(32768);
    assert_eq!(p.0.0, 32768);
}

#[test]
fn test_srgb_mono16_new_extremes() {
    assert_eq!(SrgbMono16::new(0).0.0, 0);
    assert_eq!(SrgbMono16::new(65535).0.0, 65535);
}

#[test]
fn test_srgb_mono16_zero() {
    let z = SrgbMono16::zero();
    assert_eq!(z.0.0, 0);
}

#[test]
fn test_srgb_mono16_plain_pixel_size() {
    assert_eq!(SrgbMono16::SIZE, 2);
    assert_eq!(SrgbMono16::DIM, 1);
    assert_eq!(SrgbMono16::CHANNELS, &[2]);
    assert_eq!(SrgbMono16::ALIGN, 2);
}

#[test]
fn test_srgb_mono16_as_bytes_roundtrip() {
    let p = SrgbMono16::new(0x1234);
    let bytes = p.as_bytes();
    assert_eq!(bytes.len(), 2);
    let q = SrgbMono16::from_bytes(bytes).unwrap();
    assert_eq!(p, q);
}

#[test]
fn test_srgb_mono16_as_bytes_le() {
    let p = SrgbMono16::new(0x0102);
    let le = p.as_bytes_le();
    assert_eq!(&*le, &[0x02, 0x01]);
}

#[test]
fn test_srgb_mono16_as_bytes_be() {
    let p = SrgbMono16::new(0x0102);
    let be = p.as_bytes_be();
    assert_eq!(&*be, &[0x01, 0x02]);
}

#[test]
fn test_srgb_mono16_as_mut_bytes() {
    let mut p = SrgbMono16::new(0);
    let bytes = p.as_mut_bytes();
    let val: u16 = 999;
    bytes.copy_from_slice(&val.to_ne_bytes());
    assert_eq!(p.0.0, 999);
}

#[test]
fn test_srgb_mono16_uniform_channels() {
    assert_eq!(SrgbMono16::CHANNEL_COUNT, 1);
    let p = SrgbMono16::new(4444);
    assert_eq!(p.channel(0), Saturating(4444u16));
}

#[test]
fn test_srgb_mono16_uniform_to_channels() {
    let p = SrgbMono16::new(5555);
    assert_eq!(p.to_channels(), [Saturating(5555u16)]);
}

#[test]
fn test_srgb_mono16_uniform_from_channels() {
    let p = SrgbMono16::from_channels(&[Saturating(8888u16)]);
    assert_eq!(p.0.0, 8888);
}

#[test]
fn test_srgb_mono16_uniform_set_channel() {
    let mut p = SrgbMono16::new(0);
    p.set_channel(0, Saturating(12345u16));
    assert_eq!(p.0.0, 12345);
}

#[test]
fn test_srgb_mono16_uniform_roundtrip() {
    let orig = SrgbMono16::new(54321);
    let channels = orig.to_channels();
    let restored = SrgbMono16::from_channels(&channels);
    assert_eq!(orig, restored);
}

#[test]
fn test_srgb_mono16_uniform_size_assert() {
    assert_eq!(
        SrgbMono16::SIZE,
        SrgbMono16::CHANNEL_COUNT * std::mem::size_of::<u16>()
    );
}

#[test]
fn test_srgb_mono16_dim_equals_channel_count() {
    assert_eq!(SrgbMono16::DIM, SrgbMono16::CHANNEL_COUNT);
}

#[test]
#[should_panic]
fn test_srgb_mono16_channel_out_of_bounds() {
    SrgbMono16::new(0).channel(1);
}

#[test]
#[should_panic]
fn test_srgb_mono16_set_channel_out_of_bounds() {
    let mut p = SrgbMono16::new(0);
    p.set_channel(1, Saturating(0u16));
}

#[test]
#[should_panic]
fn test_srgb_mono16_from_channels_wrong_count() {
    SrgbMono16::from_channels(&[Saturating(1u16), Saturating(2u16)]);
}

#[test]
fn test_srgb_mono16_copy_clone_eq() {
    let a = SrgbMono16::new(42);
    let b = a;
    let c = a.clone();
    assert_eq!(a, b);
    assert_eq!(a, c);
}

#[test]
fn test_srgb_mono16_ne() {
    assert_ne!(SrgbMono16::new(0), SrgbMono16::new(1));
    assert_ne!(SrgbMono16::new(65535), SrgbMono16::new(0));
}

#[test]
fn test_srgb_mono16_debug() {
    let p = SrgbMono16::new(42);
    let s = format!("{:?}", p);
    assert!(s.contains("SrgbMono16"));
}

#[test]
fn test_align_srgb_mono16() {
    assert_eq!(SrgbMono16::ALIGN, 2);
}

#[test]
fn test_channel_sum_srgb_mono16() {
    let sum: usize = SrgbMono16::CHANNELS.iter().sum();
    assert_eq!(SrgbMono16::SIZE, sum);
}

// ─── SrgbMonoA16 ────────────────────────────────────────────────────

#[test]
fn test_srgb_mono_a16_new() {
    let p = SrgbMonoA16::new(1000, 2000);
    assert_eq!(p.v.0, 1000);
    assert_eq!(p.a.0, 2000);
}

#[test]
fn test_srgb_mono_a16_new_extremes() {
    let lo = SrgbMonoA16::new(0, 0);
    let hi = SrgbMonoA16::new(65535, 65535);
    assert_eq!(lo.v.0, 0);
    assert_eq!(lo.a.0, 0);
    assert_eq!(hi.v.0, 65535);
    assert_eq!(hi.a.0, 65535);
}

#[test]
fn test_srgb_mono_a16_zero() {
    let z = SrgbMonoA16::zero();
    assert_eq!(z.v.0, 0);
    assert_eq!(z.a.0, 0);
}

#[test]
fn test_srgb_mono_a16_plain_pixel_size() {
    assert_eq!(SrgbMonoA16::SIZE, 4);
    assert_eq!(SrgbMonoA16::DIM, 2);
    assert_eq!(SrgbMonoA16::CHANNELS, &[2, 2]);
}

#[test]
fn test_srgb_mono_a16_as_bytes_roundtrip() {
    let p = SrgbMonoA16::new(0x1234, 0x5678);
    let bytes = p.as_bytes();
    assert_eq!(bytes.len(), 4);
    let q = SrgbMonoA16::from_bytes(bytes).unwrap();
    assert_eq!(p, q);
}

#[test]
fn test_srgb_mono_a16_as_bytes_le() {
    let p = SrgbMonoA16::new(0x0102, 0x0304);
    let le = p.as_bytes_le();
    assert_eq!(le.len(), 4);
}

#[test]
fn test_srgb_mono_a16_as_bytes_be() {
    let p = SrgbMonoA16::new(0x0102, 0x0304);
    let be = p.as_bytes_be();
    assert_eq!(be.len(), 4);
}

#[test]
fn test_srgb_mono_a16_as_mut_bytes() {
    let mut p = SrgbMonoA16::new(0, 0);
    let bytes = p.as_mut_bytes();
    let val: u16 = 42;
    bytes[0..2].copy_from_slice(&val.to_ne_bytes());
    assert_eq!(p.v.0, 42);
}

#[test]
fn test_srgb_mono_a16_uniform_channels() {
    assert_eq!(SrgbMonoA16::CHANNEL_COUNT, 2);
    let p = SrgbMonoA16::new(100, 200);
    assert_eq!(p.channel(0), Saturating(100u16));
    assert_eq!(p.channel(1), Saturating(200u16));
}

#[test]
fn test_srgb_mono_a16_uniform_to_channels() {
    let p = SrgbMonoA16::new(10, 20);
    assert_eq!(p.to_channels(), [Saturating(10u16), Saturating(20u16)]);
}

#[test]
fn test_srgb_mono_a16_uniform_from_channels() {
    let p = SrgbMonoA16::from_channels(&[Saturating(11u16), Saturating(22u16)]);
    assert_eq!(p.v.0, 11);
    assert_eq!(p.a.0, 22);
}

#[test]
fn test_srgb_mono_a16_uniform_set_channel() {
    let mut p = SrgbMonoA16::new(0, 0);
    p.set_channel(0, Saturating(111u16));
    p.set_channel(1, Saturating(222u16));
    assert_eq!(p, SrgbMonoA16::new(111, 222));
}

#[test]
fn test_srgb_mono_a16_uniform_roundtrip() {
    let orig = SrgbMonoA16::new(12345, 54321);
    let ch = orig.to_channels();
    let restored = SrgbMonoA16::from_channels(&ch);
    assert_eq!(orig, restored);
}

#[test]
fn test_srgb_mono_a16_uniform_size_assert() {
    assert_eq!(
        SrgbMonoA16::SIZE,
        SrgbMonoA16::CHANNEL_COUNT * std::mem::size_of::<u16>()
    );
}

#[test]
fn test_srgb_mono_a16_dim_equals_channel_count() {
    assert_eq!(SrgbMonoA16::DIM, SrgbMonoA16::CHANNEL_COUNT);
}

#[test]
#[should_panic]
fn test_srgb_mono_a16_channel_out_of_bounds() {
    SrgbMonoA16::new(0, 0).channel(2);
}

#[test]
#[should_panic]
fn test_srgb_mono_a16_set_channel_out_of_bounds() {
    let mut p = SrgbMonoA16::new(0, 0);
    p.set_channel(2, Saturating(0u16));
}

#[test]
#[should_panic]
fn test_srgb_mono_a16_from_channels_wrong_count() {
    SrgbMonoA16::from_channels(&[Saturating(1u16)]);
}

#[test]
fn test_srgb_mono_a16_copy_clone_eq() {
    let a = SrgbMonoA16::new(42, 99);
    let b = a;
    let c = a.clone();
    assert_eq!(a, b);
    assert_eq!(a, c);
}

#[test]
fn test_srgb_mono_a16_ne() {
    assert_ne!(SrgbMonoA16::new(0, 0), SrgbMonoA16::new(1, 0));
    assert_ne!(SrgbMonoA16::new(0, 0), SrgbMonoA16::new(0, 1));
}

#[test]
fn test_srgb_mono_a16_debug() {
    let p = SrgbMonoA16::new(42, 99);
    let s = format!("{:?}", p);
    assert!(s.contains("SrgbMonoA16"));
}

#[test]
fn test_align_srgb_mono_a16() {
    assert_eq!(SrgbMonoA16::ALIGN, 2);
}

#[test]
fn test_channel_sum_srgb_mono_a16() {
    let sum: usize = SrgbMonoA16::CHANNELS.iter().sum();
    assert_eq!(SrgbMonoA16::SIZE, sum);
}

// ─── 16-bit sRGB types do NOT have LinearSpace ──────────────────────

#[test]
fn test_srgb16_not_linear_space() {
    // These types must NOT implement LinearSpace or LinearPixel.
    // We verify by checking they have the same memory layout as their
    // linear counterparts but are distinct types.
    assert_eq!(Srgb16::SIZE, Rgb16::SIZE);
    assert_eq!(Srgba16::SIZE, Rgba16::SIZE);
    assert_eq!(SrgbMono16::SIZE, Mono16::SIZE);
    assert_eq!(SrgbMonoA16::SIZE, MonoA16::SIZE);
}

// ─── FromLinear coverage for primitive and wrapper types ─────────────

#[test]
fn test_from_linear_u8() {
    assert_eq!(u8::from_linear(0.0f32), 0u8);
    assert_eq!(u8::from_linear(255.0f32), 255u8);
    assert_eq!(u8::from_linear(127.6f32), 128u8);
    // Clamping
    assert_eq!(u8::from_linear(-10.0f32), 0u8);
    assert_eq!(u8::from_linear(300.0f32), 255u8);
}

#[test]
fn test_from_linear_u16() {
    assert_eq!(u16::from_linear(0.0f32), 0u16);
    assert_eq!(u16::from_linear(65535.0f32), 65535u16);
    assert_eq!(u16::from_linear(-1.0f32), 0u16);
    assert_eq!(u16::from_linear(70000.0f32), 65535u16);
}

#[test]
fn test_from_linear_u32() {
    assert_eq!(u32::from_linear(0.0f64), 0u32);
    assert_eq!(u32::from_linear(1000.0f64), 1000u32);
    assert_eq!(u32::from_linear(-1.0f64), 0u32);
}

#[test]
fn test_from_linear_u64() {
    assert_eq!(u64::from_linear(0.0f64), 0u64);
    assert_eq!(u64::from_linear(1000.0f64), 1000u64);
    assert_eq!(u64::from_linear(-1.0f64), 0u64);
}

// `test_from_linear_i8` / `_i16` / `_i32` / `_i64` removed
// (ADR-0045 Phase S4.2): the signed-integer `FromLinear<f32|f64>`
// impls were speculative (no library-shipping pixel uses them,
// parallel to ADR-0043's signed-`BoundedChannel` removal) and have
// been deleted alongside the `LinearPixel` impls on signed
// primitives.

#[test]
fn test_from_linear_saturating_u8() {
    assert_eq!(Saturating::<u8>::from_linear(0.0f32), Saturating(0u8));
    assert_eq!(Saturating::<u8>::from_linear(255.0f32), Saturating(255u8));
    assert_eq!(Saturating::<u8>::from_linear(-10.0f32), Saturating(0u8));
    assert_eq!(Saturating::<u8>::from_linear(300.0f32), Saturating(255u8));
}

#[test]
fn test_from_linear_saturating_u16() {
    assert_eq!(Saturating::<u16>::from_linear(0.0f32), Saturating(0u16));
    assert_eq!(
        Saturating::<u16>::from_linear(65535.0f32),
        Saturating(65535u16)
    );
}

#[test]
fn test_from_linear_saturating_u32() {
    assert_eq!(Saturating::<u32>::from_linear(0.0f64), Saturating(0u32));
    assert_eq!(
        Saturating::<u32>::from_linear(1000.0f64),
        Saturating(1000u32)
    );
    assert_eq!(Saturating::<u32>::from_linear(-1.0f64), Saturating(0u32));
}

#[test]
fn test_from_linear_saturating_u64() {
    assert_eq!(Saturating::<u64>::from_linear(0.0f64), Saturating(0u64));
    assert_eq!(
        Saturating::<u64>::from_linear(1000.0f64),
        Saturating(1000u64)
    );
    assert_eq!(Saturating::<u64>::from_linear(-1.0f64), Saturating(0u64));
}

#[test]
fn test_from_linear_mono10() {
    let m = Mono::<10>::from_linear(0.0f32);
    assert_eq!(m.value(), 0);
    let m = Mono::<10>::from_linear(512.0f32);
    assert_eq!(m.value(), 512);
    let m = Mono::<10>::from_linear(2000.0f32);
    assert_eq!(m.value(), 1023); // clamped to max
    let m = Mono::<10>::from_linear(-1.0f32);
    assert_eq!(m.value(), 0);
}

#[test]
fn test_from_linear_mono12() {
    let m = Mono::<12>::from_linear(4095.0f32);
    assert_eq!(m.value(), 4095);
    let m = Mono::<12>::from_linear(5000.0f32);
    assert_eq!(m.value(), 4095);
}

// ─── blend coverage for additional types ────────────────────────────

// `test_blend_u16` / `test_blend_u32` / `test_blend_u64` /
// `test_blend_i8` / `test_blend_i16` / `test_blend_i32` /
// `test_blend_i64` / `test_blend_saturating_u32` /
// `test_blend_saturating_u64` removed (ADR-0045 Phase S4).
//
// Channel primitives no longer implement `LinearPixel` / `LinearSpace`
// and can therefore no longer be passed to `blend`. The corresponding
// pixel-role coverage lives in `test_blend_mono8` / `test_blend_mono16`
// / `test_blend_mono32` / `test_blend_mono64` and the `Rgb*` / `MonoA*`
// blend tests. `f64` is kept below until ADR-0044 Phase E lands.

// `test_blend_f64` removed (ADR-0044 Phase E): `f64` is no longer a
// pixel; pixel-role blend coverage lives in the `Mono*` / `Rgb*` /
// `MonoA*` blend tests.

#[test]
fn test_blend_mono10_pixel() {
    use crate::pixel::blend;
    let a = Mono::<10>::new(0);
    let b = Mono::<10>::new(1000);
    let mid = blend(&a, &b, 0.5);
    assert!((mid - MonoF32(500.0)).abs().0 < 1.0);
}

// ─── ZeroablePixel for Saturating wrappers ──────────────────────────

#[test]
fn test_zero_saturating_u8() {
    assert_eq!(Saturating::<u8>::zero(), Saturating(0u8));
}

#[test]
fn test_zero_saturating_u16() {
    assert_eq!(Saturating::<u16>::zero(), Saturating(0u16));
}

#[test]
fn test_zero_saturating_u32() {
    assert_eq!(Saturating::<u32>::zero(), Saturating(0u32));
}

#[test]
fn test_zero_saturating_u64() {
    assert_eq!(Saturating::<u64>::zero(), Saturating(0u64));
}

// `test_zero_f32` / `test_zero_f64` removed (ADR-0044 Phase E):
// `f32` / `f64` are no longer `ZeroablePixel`.

#[test]
fn test_zero_mono10() {
    let z = Mono::<10>::zero();
    assert_eq!(z.value(), 0);
}

#[test]
fn test_zero_mono12() {
    let z = Mono::<12>::zero();
    assert_eq!(z.value(), 0);
}

// ─── LinearPixel::scale for Saturating<u32>, Saturating<u64>, Mono ──

#[test]
fn test_linear_pixel_saturating_u32_scale() {
    let pixel = Saturating(50000u32);
    let scaled = <Saturating<u32> as LinearChannel<f32>>::scale(&pixel, 0.5);
    assert_eq!(scaled, 25000.0f64);
}

#[test]
fn test_linear_pixel_saturating_u64_scale() {
    let pixel = Saturating(1000000u64);
    // Post-ADR-0045: channel primitives bind on `LinearChannel`.
    let scaled = <Saturating<u64> as LinearChannel<f32>>::scale(&pixel, 0.1);
    assert!((scaled - 100000.0f64).abs() < 1.0);
}

#[test]
fn test_linear_pixel_mono10_scale() {
    let pixel = Mono::<10>::new(500);
    let scaled = pixel.scale(2.0);
    assert!((scaled - MonoF32(1000.0)).abs().0 < 1.0);
}

#[test]
fn test_linear_pixel_mono12_scale() {
    let pixel = Mono::<12>::new(2048);
    let scaled = pixel.scale(0.5);
    assert!((scaled - MonoF32(1024.0)).abs().0 < 1.0);
}

// ─── HomogeneousPixel for primitive scalars ──────────────────────────────

#[test]
fn test_uniform_u8_channel_access() {
    let p = 42u8;
    assert_eq!(p.channel(0), 42u8);
    assert_eq!(p.to_channels(), [42u8]);
    assert_eq!(u8::from_channels(&[99u8]), 99u8);
}

#[test]
fn test_uniform_u16_channel_access() {
    let p = 1000u16;
    assert_eq!(p.channel(0), 1000u16);
    assert_eq!(p.to_channels(), [1000u16]);
    assert_eq!(u16::from_channels(&[500u16]), 500u16);
}

#[test]
fn test_uniform_u32_channel_access() {
    let p = 100000u32;
    assert_eq!(p.channel(0), 100000u32);
    assert_eq!(u32::from_channels(&[99u32]), 99u32);
}

#[test]
fn test_uniform_u64_channel_access() {
    let p = 100000u64;
    assert_eq!(p.channel(0), 100000u64);
    assert_eq!(u64::from_channels(&[99u64]), 99u64);
}

#[test]
fn test_uniform_i8_channel_access() {
    let p = -42i8;
    assert_eq!(p.channel(0), -42i8);
    assert_eq!(i8::from_channels(&[-10i8]), -10i8);
}

#[test]
fn test_uniform_i16_channel_access() {
    let p = -1000i16;
    assert_eq!(p.channel(0), -1000i16);
    assert_eq!(i16::from_channels(&[500i16]), 500i16);
}

#[test]
fn test_uniform_i32_channel_access() {
    let p = -100000i32;
    assert_eq!(p.channel(0), -100000i32);
    assert_eq!(i32::from_channels(&[99i32]), 99i32);
}

#[test]
fn test_uniform_i64_channel_access() {
    let p = -100000i64;
    assert_eq!(p.channel(0), -100000i64);
    assert_eq!(i64::from_channels(&[99i64]), 99i64);
}

// `test_uniform_f32_channel_access_scalar` / `..._f64_...` removed
// (ADR-0044 Phase E): raw floats no longer implement
// `HomogeneousPixel`. Mono-pixel equivalents are covered in
// `family_tests.rs` for `MonoF32` / `MonoF64`.

// ─── PlainPixel CHANNELS / SIZE for primitives ──────────────────────

#[test]
fn test_plain_pixel_size_primitives() {
    assert_eq!(u8::SIZE, 1);
    assert_eq!(u16::SIZE, 2);
    assert_eq!(u32::SIZE, 4);
    assert_eq!(u64::SIZE, 8);
    assert_eq!(i8::SIZE, 1);
    assert_eq!(i16::SIZE, 2);
    assert_eq!(i32::SIZE, 4);
    assert_eq!(i64::SIZE, 8);
    assert_eq!(f32::SIZE, 4);
    assert_eq!(f64::SIZE, 8);
}

#[test]
fn test_plain_pixel_size_saturating() {
    assert_eq!(Saturating::<u8>::SIZE, 1);
    assert_eq!(Saturating::<u16>::SIZE, 2);
    assert_eq!(Saturating::<u32>::SIZE, 4);
    assert_eq!(Saturating::<u64>::SIZE, 8);
}

// ─── LinearSpace marker trait is inhabited ──────────────────────────

#[test]
fn test_linear_space_marker_primitives() {
    // Post-ADR-0044 Phase E + ADR-0045 Phase S4: `LinearSpace` is
    // pixel-only. Neither channel primitives (u8..u64, i8..i64,
    // `Saturating<_>`) nor raw floats (`f32`, `f64`) implement
    // `LinearPixel` / `LinearSpace` — use `MonoF32` / `MonoF64` for
    // the pixel role.
    fn assert_linear_space<T: LinearSpace>() {}
    assert_linear_space::<Mono<10>>();
    assert_linear_space::<MonoF32>();
    assert_linear_space::<MonoF64>();
}

// ─── MonoF32 tests ─────────────────────────────────────────────────

#[test]
fn test_monof32_new() {
    let p = MonoF32::new(0.5);
    assert_eq!(p.value(), 0.5);
    assert_eq!(p.0, 0.5);
}

#[test]
fn test_monof32_zero() {
    let p = MonoF32::zero();
    assert_eq!(p.value(), 0.0);
}

#[test]
fn test_monof32_from_f32() {
    let p = MonoF32::from(0.75f32);
    assert_eq!(p.value(), 0.75);
}

#[test]
fn test_monof32_into_f32() {
    let p = MonoF32::new(0.25);
    let v: f32 = p.into();
    assert_eq!(v, 0.25);
}

#[test]
fn test_monof32_plain_pixel_size() {
    assert_eq!(MonoF32::SIZE, 4);
    assert_eq!(MonoF32::DIM, 1);
    assert_eq!(MonoF32::CHANNELS, &[4]);
}

#[test]
fn test_monof32_align() {
    assert_eq!(MonoF32::ALIGN, 4);
}

#[test]
fn test_monof32_as_bytes_roundtrip() {
    let p = MonoF32::new(1.0);
    let bytes = p.as_bytes();
    assert_eq!(bytes.len(), 4);
    let back = MonoF32::from_bytes(bytes).unwrap();
    assert_eq!(back, p);
}

#[test]
fn test_monof32_copy_clone_eq() {
    let a = MonoF32::new(0.5);
    let b = a;
    let c = a.clone();
    assert_eq!(a, b);
    assert_eq!(a, c);
}

#[test]
fn test_monof32_ne() {
    assert_ne!(MonoF32::new(0.1), MonoF32::new(0.2));
}

#[test]
fn test_monof32_debug() {
    let p = MonoF32::new(0.5);
    let s = format!("{:?}", p);
    assert!(s.contains("MonoF32"));
}

#[test]
fn test_monof32_linear_pixel_scale() {
    let p = MonoF32::new(0.5);
    let scaled = p.scale(2.0);
    assert_eq!(scaled, MonoF32::new(1.0));
}

#[test]
fn test_monof32_add() {
    let a = MonoF32::new(0.3);
    let b = MonoF32::new(0.7);
    let c = a + b;
    assert!((c.value() - 1.0).abs() < 1e-6);
}

#[test]
fn test_monof32_sub() {
    let a = MonoF32::new(1.0);
    let b = MonoF32::new(0.3);
    let c = a - b;
    assert!((c.value() - 0.7).abs() < 1e-6);
}

#[test]
fn test_monof32_mul() {
    let a = MonoF32::new(2.0);
    let b = MonoF32::new(3.0);
    assert_eq!(a * b, MonoF32::new(6.0));
    assert_eq!(MonoF32::new(0.5) * MonoF32::new(0.5), MonoF32::new(0.25));
    assert_eq!(MonoF32::new(0.0) * MonoF32::new(999.0), MonoF32::new(0.0));
}

#[test]
fn test_monof32_blend() {
    use crate::pixel::blend;
    let a = MonoF32::new(0.0);
    let b = MonoF32::new(1.0);
    let mid = blend(&a, &b, 0.5);
    assert!((mid.value() - 0.5).abs() < 1e-6);
}

#[test]
fn test_monof32_uniform_channels() {
    use crate::pixel::HomogeneousPixel;
    let p = MonoF32::new(0.42);
    assert_eq!(p.channel(0), 0.42);
}

#[test]
fn test_monof32_uniform_to_channels() {
    use crate::pixel::HomogeneousPixel;
    let p = MonoF32::new(0.42);
    let ch = p.to_channels();
    assert_eq!(ch.as_ref(), &[0.42]);
}

#[test]
fn test_monof32_uniform_from_channels() {
    use crate::pixel::HomogeneousPixel;
    let p = MonoF32::from_channels(&[0.99]);
    assert_eq!(p.value(), 0.99);
}

#[test]
fn test_monof32_uniform_set_channel() {
    use crate::pixel::HomogeneousPixel;
    let mut p = MonoF32::new(0.0);
    p.set_channel(0, 0.77);
    assert_eq!(p.value(), 0.77);
}

#[test]
fn test_monof32_uniform_roundtrip() {
    use crate::pixel::HomogeneousPixel;
    let original = MonoF32::new(0.123);
    let channels = original.to_channels();
    let rebuilt = MonoF32::from_channels(channels.as_ref());
    assert_eq!(original, rebuilt);
}

#[test]
#[should_panic]
fn test_monof32_channel_out_of_bounds() {
    use crate::pixel::HomogeneousPixel;
    let p = MonoF32::new(0.5);
    let _ = p.channel(1);
}

#[test]
fn test_monof32_channel_sum() {
    assert_eq!(MonoF32::SIZE, MonoF32::CHANNELS.iter().sum::<usize>());
}

// ─── MonoF64 tests ─────────────────────────────────────────────────

#[test]
fn test_monof64_new() {
    let p = MonoF64::new(0.5);
    assert_eq!(p.value(), 0.5);
    assert_eq!(p.0, 0.5);
}

#[test]
fn test_monof64_zero() {
    let p = MonoF64::zero();
    assert_eq!(p.value(), 0.0);
}

#[test]
fn test_monof64_from_f64() {
    let p = MonoF64::from(0.75f64);
    assert_eq!(p.value(), 0.75);
}

#[test]
fn test_monof64_into_f64() {
    let p = MonoF64::new(0.25);
    let v: f64 = p.into();
    assert_eq!(v, 0.25);
}

#[test]
fn test_monof64_plain_pixel_size() {
    assert_eq!(MonoF64::SIZE, 8);
    assert_eq!(MonoF64::DIM, 1);
    assert_eq!(MonoF64::CHANNELS, &[8]);
}

#[test]
fn test_monof64_align() {
    assert_eq!(MonoF64::ALIGN, 8);
}

#[test]
fn test_monof64_as_bytes_roundtrip() {
    let p = MonoF64::new(1.0);
    let bytes = p.as_bytes();
    assert_eq!(bytes.len(), 8);
    let back = MonoF64::from_bytes(bytes).unwrap();
    assert_eq!(back, p);
}

#[test]
fn test_monof64_copy_clone_eq() {
    let a = MonoF64::new(0.5);
    let b = a;
    let c = a.clone();
    assert_eq!(a, b);
    assert_eq!(a, c);
}

#[test]
fn test_monof64_ne() {
    assert_ne!(MonoF64::new(0.1), MonoF64::new(0.2));
}

#[test]
fn test_monof64_debug() {
    let p = MonoF64::new(0.5);
    let s = format!("{:?}", p);
    assert!(s.contains("MonoF64"));
}

#[test]
fn test_monof64_linear_pixel_scale() {
    let p = MonoF64::new(0.5);
    let scaled = p.scale(2.0);
    assert_eq!(scaled, MonoF64::new(1.0));
}

#[test]
fn test_monof64_add() {
    let a = MonoF64::new(0.3);
    let b = MonoF64::new(0.7);
    let c = a + b;
    assert!((c.value() - 1.0).abs() < 1e-12);
}

#[test]
fn test_monof64_sub() {
    let a = MonoF64::new(1.0);
    let b = MonoF64::new(0.3);
    let c = a - b;
    assert!((c.value() - 0.7).abs() < 1e-12);
}

#[test]
fn test_monof64_mul() {
    let a = MonoF64::new(2.0);
    let b = MonoF64::new(3.0);
    assert_eq!(a * b, MonoF64::new(6.0));
    assert_eq!(MonoF64::new(0.5) * MonoF64::new(0.5), MonoF64::new(0.25));
    assert_eq!(MonoF64::new(0.0) * MonoF64::new(999.0), MonoF64::new(0.0));
}

#[test]
fn test_monof64_blend() {
    use crate::pixel::blend;
    let a = MonoF64::new(0.0);
    let b = MonoF64::new(1.0);
    let mid = blend(&a, &b, 0.5);
    assert!((mid.value() - 0.5).abs() < 1e-6);
}

#[test]
fn test_monof64_uniform_channels() {
    use crate::pixel::HomogeneousPixel;
    let p = MonoF64::new(0.42);
    assert_eq!(p.channel(0), 0.42);
}

#[test]
fn test_monof64_uniform_to_channels() {
    use crate::pixel::HomogeneousPixel;
    let p = MonoF64::new(0.42);
    let ch = p.to_channels();
    assert_eq!(ch.as_ref(), &[0.42]);
}

#[test]
fn test_monof64_uniform_from_channels() {
    use crate::pixel::HomogeneousPixel;
    let p = MonoF64::from_channels(&[0.99]);
    assert_eq!(p.value(), 0.99);
}

#[test]
fn test_monof64_uniform_set_channel() {
    use crate::pixel::HomogeneousPixel;
    let mut p = MonoF64::new(0.0);
    p.set_channel(0, 0.77);
    assert_eq!(p.value(), 0.77);
}

#[test]
fn test_monof64_uniform_roundtrip() {
    use crate::pixel::HomogeneousPixel;
    let original = MonoF64::new(0.123);
    let channels = original.to_channels();
    let rebuilt = MonoF64::from_channels(channels.as_ref());
    assert_eq!(original, rebuilt);
}

#[test]
#[should_panic]
fn test_monof64_channel_out_of_bounds() {
    use crate::pixel::HomogeneousPixel;
    let p = MonoF64::new(0.5);
    let _ = p.channel(1);
}

#[test]
fn test_monof64_channel_sum() {
    assert_eq!(MonoF64::SIZE, MonoF64::CHANNELS.iter().sum::<usize>());
}

// ─── MonoF32 / MonoF64 repr(transparent) guarantees ─────────────────

#[test]
fn test_monof32_repr_transparent() {
    assert_eq!(std::mem::size_of::<MonoF32>(), std::mem::size_of::<f32>());
    assert_eq!(std::mem::align_of::<MonoF32>(), std::mem::align_of::<f32>());
}

#[test]
fn test_monof64_repr_transparent() {
    assert_eq!(std::mem::size_of::<MonoF64>(), std::mem::size_of::<f64>());
    assert_eq!(std::mem::align_of::<MonoF64>(), std::mem::align_of::<f64>());
}

#[test]
fn test_monof32_cast_slice() {
    let data = [0.0f32, 0.5, 1.0];
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(&data))
    };
    let pixels = MonoF32::cast_slice(bytes).unwrap();
    assert_eq!(pixels.len(), 3);
    assert_eq!(pixels[0], MonoF32::new(0.0));
    assert_eq!(pixels[1], MonoF32::new(0.5));
    assert_eq!(pixels[2], MonoF32::new(1.0));
}

#[test]
fn test_monof64_cast_slice() {
    let data = [0.0f64, 0.5, 1.0];
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(&data))
    };
    let pixels = MonoF64::cast_slice(bytes).unwrap();
    assert_eq!(pixels.len(), 3);
    assert_eq!(pixels[0], MonoF64::new(0.0));
    assert_eq!(pixels[1], MonoF64::new(0.5));
    assert_eq!(pixels[2], MonoF64::new(1.0));
}

// ───────────────────────────────────────────────────────────────────
// From conversions: Mono pixel types ↔ primitives
// ───────────────────────────────────────────────────────────────────

#[test]
fn test_mono8_from_u8() {
    let m: Mono8 = Mono8::from(42u8);
    assert_eq!(m.value(), 42);
    let m: Mono8 = 0u8.into();
    assert_eq!(m.value(), 0);
    let m: Mono8 = 255u8.into();
    assert_eq!(m.value(), 255);
}

#[test]
fn test_u8_from_mono8() {
    let v: u8 = Mono8::new(42).into();
    assert_eq!(v, 42);
    let v: u8 = u8::from(Mono8::new(0));
    assert_eq!(v, 0);
    let v: u8 = u8::from(Mono8::new(255));
    assert_eq!(v, 255);
}

#[test]
fn test_mono16_from_u16() {
    let m: Mono16 = Mono16::from(1000u16);
    assert_eq!(m.value(), 1000);
    let m: Mono16 = 0u16.into();
    assert_eq!(m.value(), 0);
    let m: Mono16 = 65535u16.into();
    assert_eq!(m.value(), 65535);
}

#[test]
fn test_u16_from_mono16() {
    let v: u16 = Mono16::new(1000).into();
    assert_eq!(v, 1000);
    let v: u16 = u16::from(Mono16::new(0));
    assert_eq!(v, 0);
    let v: u16 = u16::from(Mono16::new(65535));
    assert_eq!(v, 65535);
}

#[test]
fn test_mono32_from_u32() {
    let m: Mono32 = Mono32::from(100_000u32);
    assert_eq!(m.value(), 100_000);
    let m: Mono32 = 0u32.into();
    assert_eq!(m.value(), 0);
    let m: Mono32 = u32::MAX.into();
    assert_eq!(m.value(), u32::MAX);
}

#[test]
fn test_u32_from_mono32() {
    let v: u32 = Mono32::new(100_000).into();
    assert_eq!(v, 100_000);
    let v: u32 = u32::from(Mono32::new(0));
    assert_eq!(v, 0);
    let v: u32 = u32::from(Mono32::new(u32::MAX));
    assert_eq!(v, u32::MAX);
}

#[test]
fn test_mono64_from_u64() {
    let m: Mono64 = Mono64::from(1_000_000u64);
    assert_eq!(m.value(), 1_000_000);
    let m: Mono64 = 0u64.into();
    assert_eq!(m.value(), 0);
    let m: Mono64 = u64::MAX.into();
    assert_eq!(m.value(), u64::MAX);
}

#[test]
fn test_u64_from_mono64() {
    let v: u64 = Mono64::new(1_000_000).into();
    assert_eq!(v, 1_000_000);
    let v: u64 = u64::from(Mono64::new(0));
    assert_eq!(v, 0);
    let v: u64 = u64::from(Mono64::new(u64::MAX));
    assert_eq!(v, u64::MAX);
}

#[test]
fn test_mono_from_roundtrip() {
    // Verify From roundtrips for all Mono widths
    for v in [0u8, 1, 127, 255] {
        let rt: u8 = Mono8::from(v).into();
        assert_eq!(rt, v);
    }
    for v in [0u16, 1, 32768, 65535] {
        let rt: u16 = Mono16::from(v).into();
        assert_eq!(rt, v);
    }
    for v in [0u32, 1, u32::MAX / 2, u32::MAX] {
        let rt: u32 = Mono32::from(v).into();
        assert_eq!(rt, v);
    }
    for v in [0u64, 1, u64::MAX / 2, u64::MAX] {
        let rt: u64 = Mono64::from(v).into();
        assert_eq!(rt, v);
    }
}

#[test]
fn test_mono10_mul_overflow_clamps_to_max() {
    // Mono10::MAX = 1023
    let mut a = Mono::<10>::new(1023);
    a *= Mono::<10>::new(2);
    assert_eq!(a.value(), 1023); // clamped, not wrapped

    let mut b = Mono::<10>::new(500);
    b *= Mono::<10>::new(500);
    // 500 * 500 = 250_000, far exceeds 1023
    assert_eq!(b.value(), 1023);
}

#[test]
fn test_mono10_mul_ref_overflow_clamps_to_max() {
    let mut a = Mono::<10>::new(1023);
    a *= &Mono::<10>::new(1023);
    assert_eq!(a.value(), 1023);
}

#[test]
fn test_mono10_mul_u16_ref_overflow_clamps_to_max() {
    let mut a = Mono::<10>::new(1000);
    a *= &65535u16;
    assert_eq!(a.value(), 1023);
}

#[test]
fn test_rgb16_as_bytes_le_be_roundtrip() {
    use crate::pixel::PlainPixel;
    let pixel = Rgb16::new(0x1234, 0x5678, 0x9ABC);
    let le = pixel.as_bytes_le();
    let be = pixel.as_bytes_be();

    // LE: each u16 channel stored little-endian
    assert_eq!(le[0..2], [0x34, 0x12]); // R
    assert_eq!(le[2..4], [0x78, 0x56]); // G
    assert_eq!(le[4..6], [0xBC, 0x9A]); // B

    // BE: each u16 channel stored big-endian
    assert_eq!(be[0..2], [0x12, 0x34]); // R
    assert_eq!(be[2..4], [0x56, 0x78]); // G
    assert_eq!(be[4..6], [0x9A, 0xBC]); // B
}

#[test]
fn test_rgba16_as_bytes_le_be_roundtrip() {
    use crate::pixel::PlainPixel;
    let pixel = Rgba16::new(0x0102, 0x0304, 0x0506, 0x0708);
    let le = pixel.as_bytes_le();
    let be = pixel.as_bytes_be();

    assert_eq!(le[0..2], [0x02, 0x01]);
    assert_eq!(be[0..2], [0x01, 0x02]);
    // Verify all 8 bytes are present
    assert_eq!(le.len(), 8);
    assert_eq!(be.len(), 8);
}

// ═══════════════════════════════════════════════════════════════════════
// Hash / Ord / PartialOrd tests  (B10 — ADR-0026)
// ═══════════════════════════════════════════════════════════════════════

/// Helper: hash a value and return the u64 result.
fn hash_of<T: Hash>(val: &T) -> u64 {
    use std::hash::DefaultHasher;
    let mut h = DefaultHasher::new();
    val.hash(&mut h);
    h.finish()
}

// -- Hash consistency: equal values must hash equally ----------------

#[test]
fn test_hash_mono8_equal_values() {
    let a = Mono8::new(42);
    let b = Mono8::new(42);
    assert_eq!(a, b);
    assert_eq!(hash_of(&a), hash_of(&b));
}

#[test]
fn test_hash_mono8_different_values() {
    let a = Mono8::new(0);
    let b = Mono8::new(255);
    // Different values *may* collide, but almost certainly don't.
    assert_ne!(a, b);
    // We don't assert hash inequality (collisions are legal), but
    // we exercise the code path.
    let _ = hash_of(&a);
    let _ = hash_of(&b);
}

#[test]
fn test_hash_mono16() {
    let a = Mono16::new(1000);
    let b = Mono16::new(1000);
    assert_eq!(hash_of(&a), hash_of(&b));
}

#[test]
fn test_hash_mono32() {
    let a = Mono32::new(123456);
    let b = Mono32::new(123456);
    assert_eq!(hash_of(&a), hash_of(&b));
}

#[test]
fn test_hash_mono64() {
    let a = Mono64::new(99999999);
    let b = Mono64::new(99999999);
    assert_eq!(hash_of(&a), hash_of(&b));
}

#[test]
fn test_hash_mono10() {
    let a = Mono10::new(512);
    let b = Mono10::new(512);
    assert_eq!(hash_of(&a), hash_of(&b));
}

#[test]
fn test_hash_indexed8() {
    let a = Indexed8(7);
    let b = Indexed8(7);
    assert_eq!(hash_of(&a), hash_of(&b));
}

#[test]
fn test_hash_srgb_mono8() {
    let a = SrgbMono8::new(128);
    let b = SrgbMono8::new(128);
    assert_eq!(hash_of(&a), hash_of(&b));
}

#[test]
fn test_hash_srgb_mono16() {
    let a = SrgbMono16::new(32768);
    let b = SrgbMono16::new(32768);
    assert_eq!(hash_of(&a), hash_of(&b));
}

// -- Hash for single-channel floats: ±0 and NaN ---------------------

#[test]
fn test_hash_monof32_equal() {
    let a = MonoF32::new(1.5);
    let b = MonoF32::new(1.5);
    assert_eq!(hash_of(&a), hash_of(&b));
}

#[test]
fn test_hash_monof32_pos_neg_zero() {
    let pos = MonoF32::new(0.0);
    let neg = MonoF32::new(-0.0);
    // PartialEq: 0.0 == -0.0
    assert_eq!(pos, neg);
    // Hash contract: equal values must hash equally.
    assert_eq!(hash_of(&pos), hash_of(&neg));
}

#[test]
fn test_hash_monof32_nan_consistent() {
    let a = MonoF32::new(f32::NAN);
    let b = MonoF32::new(-f32::NAN);
    // NaN != NaN under PartialEq, so no hash-equality constraint,
    // but our canonicalization hashes all NaNs to the same value.
    assert_eq!(hash_of(&a), hash_of(&b));
}

#[test]
fn test_hash_monof64_pos_neg_zero() {
    let pos = MonoF64::new(0.0);
    let neg = MonoF64::new(-0.0);
    assert_eq!(pos, neg);
    assert_eq!(hash_of(&pos), hash_of(&neg));
}

#[test]
fn test_hash_monof64_nan_consistent() {
    let a = MonoF64::new(f64::NAN);
    let b = MonoF64::new(-f64::NAN);
    assert_eq!(hash_of(&a), hash_of(&b));
}

#[test]
fn test_hash_monof64_equal() {
    let a = MonoF64::new(3.14);
    let b = MonoF64::new(3.14);
    assert_eq!(hash_of(&a), hash_of(&b));
}

// -- Hash for multi-channel integer pixels --------------------------

#[test]
fn test_hash_rgb8() {
    let a = Rgb8::new(10, 20, 30);
    let b = Rgb8::new(10, 20, 30);
    assert_eq!(hash_of(&a), hash_of(&b));
}

#[test]
fn test_hash_rgba8() {
    let a = Rgba8::new(1, 2, 3, 4);
    let b = Rgba8::new(1, 2, 3, 4);
    assert_eq!(hash_of(&a), hash_of(&b));
}

#[test]
fn test_hash_bgr8() {
    let a = Bgr8::new(10, 20, 30);
    let b = Bgr8::new(10, 20, 30);
    assert_eq!(hash_of(&a), hash_of(&b));
}

#[test]
fn test_hash_bgra8() {
    let a = Bgra8::new(5, 6, 7, 8);
    let b = Bgra8::new(5, 6, 7, 8);
    assert_eq!(hash_of(&a), hash_of(&b));
}

#[test]
fn test_hash_rgb16() {
    let a = Rgb16::new(1000, 2000, 3000);
    let b = Rgb16::new(1000, 2000, 3000);
    assert_eq!(hash_of(&a), hash_of(&b));
}

#[test]
fn test_hash_monoa8() {
    let a = MonoA8::new(100, 200);
    let b = MonoA8::new(100, 200);
    assert_eq!(hash_of(&a), hash_of(&b));
}

#[test]
fn test_hash_srgb8() {
    let a = Srgb8::new(10, 20, 30);
    let b = Srgb8::new(10, 20, 30);
    assert_eq!(hash_of(&a), hash_of(&b));
}

#[test]
fn test_hash_srgba8() {
    let a = Srgba8::new(10, 20, 30, 40);
    let b = Srgba8::new(10, 20, 30, 40);
    assert_eq!(hash_of(&a), hash_of(&b));
}

#[test]
fn test_hash_srgb_mono_a8() {
    let a = SrgbMonoA8::new(10, 20);
    let b = SrgbMonoA8::new(10, 20);
    assert_eq!(hash_of(&a), hash_of(&b));
}

#[test]
fn test_hash_rgb10() {
    let a = Rgb10::new(100, 200, 300);
    let b = Rgb10::new(100, 200, 300);
    assert_eq!(hash_of(&a), hash_of(&b));
}

#[test]
fn test_hash_bgr10() {
    let a = Bgr10::new(100, 200, 300);
    let b = Bgr10::new(100, 200, 300);
    assert_eq!(hash_of(&a), hash_of(&b));
}

// -- Hash for multi-channel float pixels (canonicalized) ------------

#[test]
fn test_hash_rgbf32_equal() {
    let a = RgbF32::new(1.0, 2.0, 3.0);
    let b = RgbF32::new(1.0, 2.0, 3.0);
    assert_eq!(hash_of(&a), hash_of(&b));
}

#[test]
fn test_hash_rgbf32_pos_neg_zero() {
    let a = RgbF32::new(0.0, -0.0, 0.0);
    let b = RgbF32::new(-0.0, 0.0, -0.0);
    assert_eq!(a, b);
    assert_eq!(hash_of(&a), hash_of(&b));
}

#[test]
fn test_hash_rgbaf32_equal() {
    let a = RgbaF32::new(0.1, 0.2, 0.3, 0.4);
    let b = RgbaF32::new(0.1, 0.2, 0.3, 0.4);
    assert_eq!(hash_of(&a), hash_of(&b));
}

#[test]
fn test_hash_rgbf64_pos_neg_zero() {
    let a = RgbF64::new(0.0, -0.0, 0.0);
    let b = RgbF64::new(-0.0, 0.0, -0.0);
    assert_eq!(a, b);
    assert_eq!(hash_of(&a), hash_of(&b));
}

#[test]
fn test_hash_bgrf32_equal() {
    let a = BgrF32::new(1.0, 2.0, 3.0);
    let b = BgrF32::new(1.0, 2.0, 3.0);
    assert_eq!(hash_of(&a), hash_of(&b));
}

#[test]
fn test_hash_bgraf32_pos_neg_zero() {
    let a = BgraF32::new(0.0, -0.0, 0.0, -0.0);
    let b = BgraF32::new(-0.0, 0.0, -0.0, 0.0);
    assert_eq!(a, b);
    assert_eq!(hash_of(&a), hash_of(&b));
}

#[test]
fn test_hash_bgrf64_equal() {
    let a = BgrF64::new(1.0, 2.0, 3.0);
    let b = BgrF64::new(1.0, 2.0, 3.0);
    assert_eq!(hash_of(&a), hash_of(&b));
}

#[test]
fn test_hash_bgraf64_equal() {
    let a = BgraF64::new(1.0, 2.0, 3.0, 4.0);
    let b = BgraF64::new(1.0, 2.0, 3.0, 4.0);
    assert_eq!(hash_of(&a), hash_of(&b));
}

#[test]
fn test_hash_monoaf32_pos_neg_zero() {
    let a = MonoAF32::new(0.0, -0.0);
    let b = MonoAF32::new(-0.0, 0.0);
    assert_eq!(a, b);
    assert_eq!(hash_of(&a), hash_of(&b));
}

#[test]
fn test_hash_monoaf64_pos_neg_zero() {
    let a = MonoAF64::new(0.0, -0.0);
    let b = MonoAF64::new(-0.0, 0.0);
    assert_eq!(a, b);
    assert_eq!(hash_of(&a), hash_of(&b));
}

// -- Hash in collections: exercise HashMap --------------------------

#[test]
fn test_hash_mono8_in_hashmap() {
    use std::collections::HashMap;
    let mut map = HashMap::new();
    map.insert(Mono8::new(10), "ten");
    map.insert(Mono8::new(20), "twenty");
    assert_eq!(map[&Mono8::new(10)], "ten");
    assert_eq!(map[&Mono8::new(20)], "twenty");
    assert_eq!(map.len(), 2);
}

#[test]
fn test_hash_rgb8_in_hashset() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(Rgb8::new(1, 2, 3));
    set.insert(Rgb8::new(1, 2, 3)); // duplicate
    set.insert(Rgb8::new(4, 5, 6));
    assert_eq!(set.len(), 2);
}

#[test]
fn test_hash_indexed8_in_hashmap() {
    use std::collections::HashMap;
    let mut map = HashMap::new();
    map.insert(Indexed8(1), "one");
    map.insert(Indexed8(2), "two");
    assert_eq!(map[&Indexed8(1)], "one");
    assert_eq!(map.len(), 2);
}

#[test]
fn test_hash_mono16_in_hashset() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(Mono16::new(100));
    set.insert(Mono16::new(100)); // duplicate
    set.insert(Mono16::new(200));
    assert_eq!(set.len(), 2);
}

// -- Ord for single-channel integer pixels --------------------------

#[test]
fn test_ord_mono8() {
    let a = Mono8::new(10);
    let b = Mono8::new(20);
    let c = Mono8::new(10);
    assert!(a < b);
    assert!(b > a);
    assert!(a <= c);
    assert!(a >= c);
    assert_eq!(a.cmp(&c), std::cmp::Ordering::Equal);
    assert_eq!(a.cmp(&b), std::cmp::Ordering::Less);
}

#[test]
fn test_ord_mono16() {
    assert!(Mono16::new(100) < Mono16::new(200));
    assert_eq!(
        Mono16::new(50).cmp(&Mono16::new(50)),
        std::cmp::Ordering::Equal
    );
}

#[test]
fn test_ord_mono32() {
    assert!(Mono32::new(0) < Mono32::new(1));
}

#[test]
fn test_ord_mono64() {
    assert!(Mono64::new(0) < Mono64::new(u64::MAX));
}

#[test]
fn test_ord_mono10() {
    let a = Mono10::new(100);
    let b = Mono10::new(500);
    assert!(a < b);
    assert_eq!(a.cmp(&a), std::cmp::Ordering::Equal);
}

#[test]
fn test_ord_mono12() {
    let a = Mono12::new(1000);
    let b = Mono12::new(2000);
    assert!(a < b);
}

#[test]
fn test_ord_indexed8() {
    let a = Indexed8(5);
    let b = Indexed8(10);
    assert!(a < b);
    assert_eq!(a.cmp(&a), std::cmp::Ordering::Equal);
}

#[test]
fn test_ord_srgb_mono8() {
    assert!(SrgbMono8::new(0) < SrgbMono8::new(255));
}

#[test]
fn test_ord_srgb_mono16() {
    assert!(SrgbMono16::new(0) < SrgbMono16::new(65535));
}

#[test]
fn test_ord_mono8_in_btreeset() {
    use std::collections::BTreeSet;
    let mut set = BTreeSet::new();
    set.insert(Mono8::new(30));
    set.insert(Mono8::new(10));
    set.insert(Mono8::new(20));
    let sorted: Vec<_> = set.iter().copied().collect();
    assert_eq!(sorted, vec![Mono8::new(10), Mono8::new(20), Mono8::new(30)]);
}

#[test]
fn test_ord_mono8_sort() {
    let mut pixels = vec![
        Mono8::new(200),
        Mono8::new(50),
        Mono8::new(100),
        Mono8::new(0),
        Mono8::new(255),
    ];
    pixels.sort();
    assert_eq!(
        pixels,
        vec![
            Mono8::new(0),
            Mono8::new(50),
            Mono8::new(100),
            Mono8::new(200),
            Mono8::new(255),
        ]
    );
}

#[test]
fn test_ord_mono10_sort() {
    let mut pixels = vec![Mono10::new(1023), Mono10::new(0), Mono10::new(512)];
    pixels.sort();
    assert_eq!(
        pixels,
        vec![Mono10::new(0), Mono10::new(512), Mono10::new(1023)]
    );
}

// -- PartialOrd for single-channel float pixels ---------------------

#[test]
fn test_partial_ord_monof32() {
    let a = MonoF32::new(1.0);
    let b = MonoF32::new(2.0);
    assert!(a < b);
    assert!(b > a);
    assert_eq!(a.partial_cmp(&a), Some(std::cmp::Ordering::Equal));
    assert_eq!(a.partial_cmp(&b), Some(std::cmp::Ordering::Less));
}

#[test]
fn test_partial_ord_monof32_nan() {
    let a = MonoF32::new(1.0);
    let nan = MonoF32::new(f32::NAN);
    // NaN is not comparable to anything.
    assert_eq!(a.partial_cmp(&nan), None);
    assert_eq!(nan.partial_cmp(&a), None);
    assert_eq!(nan.partial_cmp(&nan), None);
    // NaN is not less than, greater than, or equal to anything.
    assert!(!(nan < a));
    assert!(!(nan > a));
    assert!(!(nan <= a));
    assert!(!(nan >= a));
    assert!(!(nan == a));
}

#[test]
fn test_partial_ord_monof32_negative_values() {
    let neg = MonoF32::new(-1.0);
    let pos = MonoF32::new(1.0);
    let zero = MonoF32::new(0.0);
    assert!(neg < zero);
    assert!(zero < pos);
    assert!(neg < pos);
}

#[test]
fn test_partial_ord_monof64() {
    let a = MonoF64::new(1.0);
    let b = MonoF64::new(2.0);
    assert!(a < b);
    assert_eq!(a.partial_cmp(&a), Some(std::cmp::Ordering::Equal));
}

#[test]
fn test_partial_ord_monof64_nan() {
    let a = MonoF64::new(1.0);
    let nan = MonoF64::new(f64::NAN);
    assert_eq!(a.partial_cmp(&nan), None);
    assert_eq!(nan.partial_cmp(&nan), None);
}

#[test]
fn test_partial_ord_monof32_infinity() {
    let a = MonoF32::new(1.0);
    let inf = MonoF32::new(f32::INFINITY);
    let neg_inf = MonoF32::new(f32::NEG_INFINITY);
    assert!(neg_inf < a);
    assert!(a < inf);
    assert!(neg_inf < inf);
}

// -- Canonicalize helpers -------------------------------------------

#[test]
fn test_canonicalize_f32_normal() {
    assert_eq!(canonicalize_f32(1.0), 1.0f32.to_bits());
    assert_eq!(canonicalize_f32(-1.0), (-1.0f32).to_bits());
}

#[test]
fn test_canonicalize_f32_zero() {
    // Both +0 and -0 must map to the same value.
    assert_eq!(canonicalize_f32(0.0), canonicalize_f32(-0.0));
    assert_eq!(canonicalize_f32(0.0), 0u32);
}

#[test]
fn test_canonicalize_f32_nan() {
    // All NaN bit patterns must map to the same canonical value.
    let nan1 = f32::NAN;
    let nan2 = -f32::NAN;
    let nan3 = f32::from_bits(0x7F80_0001); // signaling NaN
    assert_eq!(canonicalize_f32(nan1), canonicalize_f32(nan2));
    assert_eq!(canonicalize_f32(nan1), canonicalize_f32(nan3));
}

#[test]
fn test_canonicalize_f64_zero() {
    assert_eq!(canonicalize_f64(0.0), canonicalize_f64(-0.0));
    assert_eq!(canonicalize_f64(0.0), 0u64);
}

#[test]
fn test_canonicalize_f64_nan() {
    let nan1 = f64::NAN;
    let nan2 = -f64::NAN;
    assert_eq!(canonicalize_f64(nan1), canonicalize_f64(nan2));
}

#[test]
fn test_canonicalize_f64_normal() {
    assert_eq!(canonicalize_f64(3.14), 3.14f64.to_bits());
}

// -- Exhaustive: Hash consistency across all float pixel types ------

#[test]
fn test_hash_rgbaf64_pos_neg_zero() {
    let a = RgbaF64::new(0.0, -0.0, 0.0, -0.0);
    let b = RgbaF64::new(-0.0, 0.0, -0.0, 0.0);
    assert_eq!(a, b);
    assert_eq!(hash_of(&a), hash_of(&b));
}

// -- Verify all pixel types are Hash (compilation check) ------------
// If any type doesn't implement Hash, this won't compile.

#[test]
fn test_all_pixel_types_are_hash() {
    fn assert_hash<T: Hash>() {}
    // Single-channel integer
    assert_hash::<Mono8>();
    assert_hash::<Mono16>();
    assert_hash::<Mono32>();
    assert_hash::<Mono64>();
    assert_hash::<Mono10>();
    assert_hash::<Mono12>();
    assert_hash::<Mono14>();
    assert_hash::<Indexed8>();
    assert_hash::<SrgbMono8>();
    assert_hash::<SrgbMono16>();
    // Single-channel float
    assert_hash::<MonoF32>();
    assert_hash::<MonoF64>();
    // Multi-channel integer
    assert_hash::<Rgb8>();
    assert_hash::<Rgba8>();
    assert_hash::<Rgb16>();
    assert_hash::<Rgba16>();
    assert_hash::<Rgb32>();
    assert_hash::<Rgba32>();
    assert_hash::<Rgb64>();
    assert_hash::<Rgba64>();
    assert_hash::<Rgb10>();
    assert_hash::<Rgb12>();
    assert_hash::<Rgba10>();
    assert_hash::<Rgba12>();
    assert_hash::<Bgr8>();
    assert_hash::<Bgra8>();
    assert_hash::<Bgr16>();
    assert_hash::<Bgra16>();
    assert_hash::<Bgr32>();
    assert_hash::<Bgra32>();
    assert_hash::<Bgr64>();
    assert_hash::<Bgra64>();
    assert_hash::<Bgr10>();
    assert_hash::<Bgr12>();
    assert_hash::<Bgra10>();
    assert_hash::<Bgra12>();
    assert_hash::<MonoA8>();
    assert_hash::<MonoA16>();
    assert_hash::<MonoA32>();
    assert_hash::<MonoA64>();
    assert_hash::<MonoAF32>();
    assert_hash::<MonoAF64>();
    assert_hash::<Srgb8>();
    assert_hash::<Srgba8>();
    assert_hash::<SrgbMonoA8>();
    assert_hash::<Srgb16>();
    assert_hash::<Srgba16>();
    assert_hash::<SrgbMonoA16>();
    // Multi-channel float
    assert_hash::<RgbF32>();
    assert_hash::<RgbaF32>();
    assert_hash::<RgbF64>();
    assert_hash::<RgbaF64>();
    assert_hash::<BgrF32>();
    assert_hash::<BgraF32>();
    assert_hash::<BgrF64>();
    assert_hash::<BgraF64>();
}

// -- Verify Ord for single-channel integer types --------------------

#[test]
fn test_all_single_channel_int_are_ord() {
    fn assert_ord<T: Ord>() {}
    assert_ord::<Mono8>();
    assert_ord::<Mono16>();
    assert_ord::<Mono32>();
    assert_ord::<Mono64>();
    assert_ord::<Mono10>();
    assert_ord::<Mono12>();
    assert_ord::<Mono14>();
    assert_ord::<Indexed8>();
    assert_ord::<SrgbMono8>();
    assert_ord::<SrgbMono16>();
}

// -- Verify PartialOrd for single-channel float types ---------------

#[test]
fn test_all_single_channel_float_are_partial_ord() {
    fn assert_partial_ord<T: PartialOrd>() {}
    assert_partial_ord::<MonoF32>();
    assert_partial_ord::<MonoF64>();
}

// ─── LinearPixel::uniform on composite / derived pixel types (PLAN §3.4) ───
//
// The derive macro emits `uniform(scalar) -> Accumulator` by delegating to
// each field's `LinearPixel::uniform`. These tests verify that the
// scalar is correctly broadcast across every channel of the accumulator,
// for each channel shape (single-field tuple wrapper, multi-channel
// struct with named fields, alpha-bearing variant, BGR ordering, etc.).
//
// Zero-sized-type strategies like `BrightnessContrast` use this method to
// build the additive "brightness" term without allocating an intermediate
// pixel — so any regression here would silently break the hot loop.

#[test]
fn test_uniform_mono8_equals_scalar() {
    use crate::pixel::LinearPixel;
    // Mono8 is a single-field tuple struct over Saturating<u8>, whose
    // accumulator is now `MonoF32` (post-ADR-0045 Phase B). The derive
    // collapses the tuple to a direct return of the field's uniform,
    // wrapped in `MonoF32`.
    assert_eq!(<Mono8 as LinearPixel>::uniform(0.5), MonoF32(0.5));
    assert_eq!(<Mono8 as LinearPixel>::uniform(0.0), MonoF32(0.0));
    assert_eq!(<Mono8 as LinearPixel>::uniform(255.0), MonoF32(255.0));
}

#[test]
fn test_uniform_mono16_equals_scalar() {
    use crate::pixel::LinearPixel;
    assert_eq!(<Mono16 as LinearPixel>::uniform(1.5), MonoF32(1.5));
}

#[test]
fn test_uniform_monof32_wraps_scalar() {
    use crate::pixel::LinearPixel;
    // MonoF32's accumulator is Self (MonoF32), so `uniform` wraps the
    // scalar back into the pixel type.
    assert_eq!(<MonoF32 as LinearPixel>::uniform(0.5), MonoF32(0.5));
}

#[test]
fn test_uniform_monof64_wraps_scalar() {
    use crate::pixel::LinearPixel;
    // MonoF64's accumulator is Self; the scalar is widened to f64.
    assert_eq!(<MonoF64 as LinearPixel>::uniform(0.5), MonoF64(0.5));
}

#[test]
fn test_uniform_rgb8_broadcasts_to_all_channels() {
    use crate::pixel::LinearPixel;
    // Rgb8 -> RgbF32 accumulator. Every channel should carry the same
    // scalar value.
    let acc: RgbF32 = <Rgb8 as LinearPixel>::uniform(1.0);
    assert_eq!(acc.r, 1.0f32);
    assert_eq!(acc.g, 1.0f32);
    assert_eq!(acc.b, 1.0f32);
}

#[test]
fn test_uniform_rgb8_zero() {
    use crate::pixel::LinearPixel;
    let acc: RgbF32 = <Rgb8 as LinearPixel>::uniform(0.0);
    assert_eq!(acc, RgbF32::new(0.0, 0.0, 0.0));
}

#[test]
fn test_uniform_rgba8_broadcasts_including_alpha() {
    use crate::pixel::LinearPixel;
    // Alpha is a channel; `uniform` treats every channel the same. Users
    // who want different behaviour should not use `uniform`.
    let acc: RgbaF32 = <Rgba8 as LinearPixel>::uniform(0.25);
    assert_eq!(acc, RgbaF32::new(0.25, 0.25, 0.25, 0.25));
}

#[test]
fn test_uniform_bgr8_preserves_channel_order_semantics() {
    use crate::pixel::LinearPixel;
    // BGR ordering is irrelevant for a uniform broadcast — every field
    // receives the same scalar regardless of semantic position.
    let acc: BgrF32 = <Bgr8 as LinearPixel>::uniform(0.5);
    assert_eq!(acc, BgrF32::new(0.5, 0.5, 0.5));
}

#[test]
fn test_uniform_rgb16_broadcasts() {
    use crate::pixel::LinearPixel;
    let acc: RgbF32 = <Rgb16 as LinearPixel>::uniform(2.5);
    assert_eq!(acc, RgbF32::new(2.5, 2.5, 2.5));
}

#[test]
fn test_uniform_monoa8_broadcasts() {
    use crate::pixel::LinearPixel;
    // Grayscale-with-alpha: value + alpha both receive the scalar.
    let acc: MonoAF32 = <MonoA8 as LinearPixel>::uniform(0.75);
    assert_eq!(acc, MonoAF32::new(0.75, 0.75));
}

#[test]
fn test_uniform_rgbf32_identity() {
    use crate::pixel::LinearPixel;
    // Self-accumulator: uniform wraps the scalar into every field.
    let acc: RgbF32 = <RgbF32 as LinearPixel>::uniform(0.1);
    assert_eq!(acc, RgbF32::new(0.1, 0.1, 0.1));
}

#[test]
fn test_uniform_rgbaf32_identity() {
    use crate::pixel::LinearPixel;
    let acc: RgbaF32 = <RgbaF32 as LinearPixel>::uniform(0.1);
    assert_eq!(acc, RgbaF32::new(0.1, 0.1, 0.1, 0.1));
}

#[test]
fn test_uniform_then_add_is_scale_add_brightness_model() {
    use crate::pixel::LinearPixel;
    // This is the exact composition that `BrightnessContrast` will use
    // (PLAN §3.5): `scale_add(pixel, contrast, uniform(brightness))`.
    // Verify the arithmetic identity on a concrete sample.
    let pixel = Rgb8::new(100, 50, 200);
    let brightness = 10.0f32;
    let contrast = 2.0f32;
    let addend: RgbF32 = <Rgb8 as LinearPixel>::uniform(brightness);
    let result: RgbF32 = pixel.scale_add(contrast, addend);
    // Expected per channel: channel * contrast + brightness
    assert!((result.r - (100.0 * 2.0 + 10.0)).abs() < 1e-4);
    assert!((result.g - (50.0 * 2.0 + 10.0)).abs() < 1e-4);
    assert!((result.b - (200.0 * 2.0 + 10.0)).abs() < 1e-4);
}

// ─── LinearPixel<f64> on f64-accumulator pixel types (PLAN §3.4) ────────────
//
// `Mono32`, `Mono64`, and `MonoF64` have `Accumulator = f64` (or `Self`
// for `MonoF64`) and carry a dedicated `LinearPixel<f64>` impl alongside
// the derive-generated `LinearPixel<f32>`. This lets scalar-parameterized
// strategies like `BrightnessContrast::<f64>` pick up a native-f64 path
// via trait resolution, avoiding an `f32 → f64` widening in the hot loop.

#[test]
fn test_mono32_linear_pixel_f64_scalar() {
    use crate::pixel::LinearPixel;
    let p = Mono32::new(1_000_000);
    assert_eq!(
        <Mono32 as LinearPixel<f64>>::to_accumulator(&p),
        MonoF64(1_000_000.0)
    );
    assert_eq!(
        <Mono32 as LinearPixel<f64>>::scale(&p, 0.5),
        MonoF64(500_000.0)
    );
    assert_eq!(
        <Mono32 as LinearPixel<f64>>::scale_add(&p, 0.5, MonoF64(7.0)),
        MonoF64(500_007.0)
    );
    assert_eq!(<Mono32 as LinearPixel<f64>>::uniform(0.25), MonoF64(0.25));
}

#[test]
fn test_mono64_linear_pixel_f64_scalar() {
    use crate::pixel::LinearPixel;
    let p = Mono64::new(2_000_000);
    assert_eq!(
        <Mono64 as LinearPixel<f64>>::to_accumulator(&p),
        MonoF64(2_000_000.0)
    );
    assert_eq!(
        <Mono64 as LinearPixel<f64>>::scale(&p, 0.25),
        MonoF64(500_000.0)
    );
    assert_eq!(
        <Mono64 as LinearPixel<f64>>::scale_add(&p, 0.25, MonoF64(-5.0)),
        MonoF64(499_995.0)
    );
    assert_eq!(<Mono64 as LinearPixel<f64>>::uniform(-7.5), MonoF64(-7.5));
}

#[test]
fn test_monof64_linear_pixel_f64_scalar() {
    use crate::pixel::LinearPixel;
    // MonoF64's accumulator is Self; the f64-scalar path stays entirely
    // in f64 with no widening.
    let p = MonoF64(3.5);
    assert_eq!(
        <MonoF64 as LinearPixel<f64>>::to_accumulator(&p),
        MonoF64(3.5)
    );
    assert_eq!(<MonoF64 as LinearPixel<f64>>::scale(&p, 2.0), MonoF64(7.0));
    assert_eq!(
        <MonoF64 as LinearPixel<f64>>::scale_add(&p, 2.0, MonoF64(1.0)),
        MonoF64(8.0)
    );
    assert_eq!(<MonoF64 as LinearPixel<f64>>::uniform(0.5), MonoF64(0.5));
}

#[test]
fn test_f64_scalar_matches_f32_scalar_for_exact_values() {
    use crate::pixel::LinearPixel;
    // For values representable exactly in both f32 and f64, the two
    // scalar paths must agree — this guards against the f64-scalar impl
    // diverging from the default f32-scalar path on values where no
    // precision is actually at stake.
    let p = Mono32::new(10_000);
    let via_f32 = <Mono32 as LinearPixel<f32>>::scale(&p, 0.5);
    let via_f64 = <Mono32 as LinearPixel<f64>>::scale(&p, 0.5);
    assert_eq!(via_f32, MonoF64(5_000.0));
    assert_eq!(via_f64, MonoF64(5_000.0));
    assert_eq!(via_f32, via_f64);
}

// ─── ADR-0045 Phase A — named-float-accumulator FromLinear siblings ──────
//
// Every pixel that has `FromLinear<f32|f64>` must, post-Phase-A, also
// have a `FromLinear<MonoF32|MonoF64>` sibling whose output agrees
// bit-for-bit with the bare-float body. The siblings exist so that
// Phase B can flip `Mono8` / `Mono16` / … accumulators from raw `f32`
// to `MonoF32` without breaking any downstream `FromLinear` bound.

#[test]
fn test_from_linear_monof32_matches_f32_u8() {
    for acc in [0.0f32, 12.3, 127.5, 200.0, 255.0, 300.0, -1.0] {
        let via_f32 = <u8 as FromLinear<f32>>::from_linear(acc);
        let via_named = <u8 as FromLinear<MonoF32>>::from_linear(MonoF32(acc));
        assert_eq!(via_f32, via_named);
    }
}

#[test]
fn test_from_linear_monof32_matches_f32_u16() {
    for acc in [0.0f32, 1000.0, 32767.5, 65535.0, 70000.0, -1.0] {
        let via_f32 = <u16 as FromLinear<f32>>::from_linear(acc);
        let via_named = <u16 as FromLinear<MonoF32>>::from_linear(MonoF32(acc));
        assert_eq!(via_f32, via_named);
    }
}

#[test]
fn test_from_linear_monof64_matches_f64_u32() {
    for acc in [0.0f64, 1_000_000.0, 2_147_483_647.5, 4_294_967_295.0, -1.0] {
        let via_f64 = <u32 as FromLinear<f64>>::from_linear(acc);
        let via_named = <u32 as FromLinear<MonoF64>>::from_linear(MonoF64(acc));
        assert_eq!(via_f64, via_named);
    }
}

#[test]
fn test_from_linear_monof64_matches_f64_u64() {
    for acc in [0.0f64, 1_000_000.0, 1e18, -1.0] {
        let via_f64 = <u64 as FromLinear<f64>>::from_linear(acc);
        let via_named = <u64 as FromLinear<MonoF64>>::from_linear(MonoF64(acc));
        assert_eq!(via_f64, via_named);
    }
}

#[test]
fn test_from_linear_monof32_matches_f32_saturating_u8() {
    for acc in [0.0f32, 50.0, 200.0, 255.0, 300.0, -1.0] {
        let via_f32 = <Saturating<u8> as FromLinear<f32>>::from_linear(acc);
        let via_named = <Saturating<u8> as FromLinear<MonoF32>>::from_linear(MonoF32(acc));
        assert_eq!(via_f32, via_named);
    }
}

#[test]
fn test_from_linear_monof32_matches_f32_saturating_u16() {
    for acc in [0.0f32, 10_000.0, 65535.0, 70_000.0, -1.0] {
        let via_f32 = <Saturating<u16> as FromLinear<f32>>::from_linear(acc);
        let via_named = <Saturating<u16> as FromLinear<MonoF32>>::from_linear(MonoF32(acc));
        assert_eq!(via_f32, via_named);
    }
}

#[test]
fn test_from_linear_monof64_matches_f64_saturating_u32() {
    for acc in [0.0f64, 1_000_000.0, 4_294_967_295.0, 5e9, -1.0] {
        let via_f64 = <Saturating<u32> as FromLinear<f64>>::from_linear(acc);
        let via_named = <Saturating<u32> as FromLinear<MonoF64>>::from_linear(MonoF64(acc));
        assert_eq!(via_f64, via_named);
    }
}

#[test]
fn test_from_linear_monof64_matches_f64_saturating_u64() {
    for acc in [0.0f64, 1_000_000.0, 1e18, -1.0] {
        let via_f64 = <Saturating<u64> as FromLinear<f64>>::from_linear(acc);
        let via_named = <Saturating<u64> as FromLinear<MonoF64>>::from_linear(MonoF64(acc));
        assert_eq!(via_f64, via_named);
    }
}

#[test]
fn test_from_linear_monof32_matches_f32_mono_bits() {
    // Mono<BITS> clamps to `(1 << BITS) - 1`; verify the MonoF32
    // sibling hits the same branch for representative in-range,
    // at-max, and out-of-range accumulators across the BITS values
    // the library instantiates (10 / 12 / 14 — see `_ASSERT_BITS`).
    for acc in [0.0f32, 100.0, 511.0, 1023.0, 2048.0, -1.0] {
        let v10 = <Mono<10> as FromLinear<f32>>::from_linear(acc);
        let v10n = <Mono<10> as FromLinear<MonoF32>>::from_linear(MonoF32(acc));
        assert_eq!(v10, v10n);

        let v12 = <Mono<12> as FromLinear<f32>>::from_linear(acc);
        let v12n = <Mono<12> as FromLinear<MonoF32>>::from_linear(MonoF32(acc));
        assert_eq!(v12, v12n);

        let v14 = <Mono<14> as FromLinear<f32>>::from_linear(acc);
        let v14n = <Mono<14> as FromLinear<MonoF32>>::from_linear(MonoF32(acc));
        assert_eq!(v14, v14n);
    }
}

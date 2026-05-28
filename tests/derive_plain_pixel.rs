use irys_cv::PlainPixel;
use irys_cv::pixel::PlainChannel;
use irys_cv::pixel::PlainPixel as PlainPixelTrait;
use std::num::Saturating;

#[derive(Clone, Copy, Debug, PartialEq, PlainPixel)]
#[repr(C)]
struct TestRgb {
    r: Saturating<u8>,
    g: Saturating<u8>,
    b: Saturating<u8>,
}

#[derive(Clone, Copy, PlainPixel)]
#[repr(C)]
struct TestRgba {
    r: Saturating<u8>,
    g: Saturating<u8>,
    b: Saturating<u8>,
    a: Saturating<u8>,
}

#[derive(Clone, Copy, PlainPixel)]
#[repr(C)]
struct TestMono16 {
    value: Saturating<u16>,
}

#[derive(Clone, Copy, PlainPixel)]
#[repr(transparent)]
struct TestWrappedMono(Saturating<u8>);

#[derive(Clone, Copy, PlainPixel)]
#[repr(C)]
struct TestTupleReprC(Saturating<u8>, Saturating<u8>);

#[test]
fn repr_c_channels_match_field_sizes() {
    assert_eq!(TestRgb::CHANNELS, &[1, 1, 1]);
    assert_eq!(TestRgb::DIM, 3);
    assert_eq!(TestRgb::SIZE, 3);
}

#[test]
fn repr_c_rgba_channels() {
    assert_eq!(TestRgba::CHANNELS, &[1, 1, 1, 1]);
    assert_eq!(TestRgba::DIM, 4);
    assert_eq!(TestRgba::SIZE, 4);
}

#[test]
fn repr_c_mono16_channels() {
    assert_eq!(TestMono16::CHANNELS, &[2]);
    assert_eq!(TestMono16::DIM, 1);
    assert_eq!(TestMono16::SIZE, 2);
}

#[test]
fn repr_transparent_inherits_channels() {
    assert_eq!(TestWrappedMono::CHANNELS, &[1]);
    assert_eq!(TestWrappedMono::DIM, 1);
    assert_eq!(TestWrappedMono::SIZE, 1);
}

#[test]
fn repr_c_tuple_struct() {
    assert_eq!(TestTupleReprC::CHANNELS, &[1, 1]);
    assert_eq!(TestTupleReprC::DIM, 2);
    assert_eq!(TestTupleReprC::SIZE, 2);
}

#[test]
fn as_bytes_roundtrip() {
    let pixel = TestRgb {
        r: Saturating(10),
        g: Saturating(20),
        b: Saturating(30),
    };
    let bytes = pixel.as_bytes();
    assert_eq!(bytes, &[10, 20, 30]);

    let restored = TestRgb::from_bytes(bytes).unwrap();
    assert_eq!(restored, pixel);
}

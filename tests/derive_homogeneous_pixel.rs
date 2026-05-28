use fovea::pixel::{HomogeneousPixel, PlainChannel, PlainPixel};
use fovea::{HomogeneousPixel as HomogeneousPixelDerive, PlainPixel as PlainPixelDerive};
use std::num::Saturating;

// ---------------------------------------------------------------------------
// Test structs — repr(C) named fields
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, PlainPixelDerive, HomogeneousPixelDerive)]
#[repr(C)]
struct TestRgb {
    r: Saturating<u8>,
    g: Saturating<u8>,
    b: Saturating<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PlainPixelDerive, HomogeneousPixelDerive)]
#[repr(C)]
struct TestRgba {
    r: Saturating<u8>,
    g: Saturating<u8>,
    b: Saturating<u8>,
    a: Saturating<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PlainPixelDerive, HomogeneousPixelDerive)]
#[repr(C)]
struct TestMono16 {
    value: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PlainPixelDerive, HomogeneousPixelDerive)]
#[repr(C)]
struct TestRgb16 {
    r: Saturating<u16>,
    g: Saturating<u16>,
    b: Saturating<u16>,
}

// ---------------------------------------------------------------------------
// Test structs — repr(C) tuple fields
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, PlainPixelDerive, HomogeneousPixelDerive)]
#[repr(C)]
struct TestTupleRgb(Saturating<u8>, Saturating<u8>, Saturating<u8>);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PlainPixelDerive, HomogeneousPixelDerive)]
#[repr(C)]
struct TestTuplePair(u32, u32);

// ---------------------------------------------------------------------------
// Test structs — repr(transparent)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, PlainPixelDerive, HomogeneousPixelDerive)]
#[repr(transparent)]
struct TestWrappedU8 {
    value: Saturating<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PlainPixelDerive, HomogeneousPixelDerive)]
#[repr(transparent)]
struct TestWrappedU16(u16);

// ===========================================================================
// Associated constant tests
// ===========================================================================

#[test]
fn channel_count_rgb() {
    assert_eq!(TestRgb::CHANNEL_COUNT, 3);
}

#[test]
fn channel_count_rgba() {
    assert_eq!(TestRgba::CHANNEL_COUNT, 4);
}

#[test]
fn channel_count_mono16() {
    assert_eq!(TestMono16::CHANNEL_COUNT, 1);
}

#[test]
fn channel_count_rgb16() {
    assert_eq!(TestRgb16::CHANNEL_COUNT, 3);
}

#[test]
fn channel_count_tuple_rgb() {
    assert_eq!(TestTupleRgb::CHANNEL_COUNT, 3);
}

#[test]
fn channel_count_tuple_pair() {
    assert_eq!(TestTuplePair::CHANNEL_COUNT, 2);
}

#[test]
fn channel_count_transparent_named() {
    assert_eq!(TestWrappedU8::CHANNEL_COUNT, 1);
}

#[test]
fn channel_count_transparent_tuple() {
    assert_eq!(TestWrappedU16::CHANNEL_COUNT, 1);
}

// ===========================================================================
// Size assertion (exercises the trait-level `_SIZE_ASSERT`)
// ===========================================================================

#[test]
fn size_assert_rgb() {
    // Force the const to be evaluated.
    let _ = <TestRgb as HomogeneousPixel>::_SIZE_ASSERT;
    assert_eq!(TestRgb::SIZE, std::mem::size_of::<Saturating<u8>>() * 3);
}

#[test]
fn size_assert_rgba() {
    let _ = <TestRgba as HomogeneousPixel>::_SIZE_ASSERT;
    assert_eq!(TestRgba::SIZE, std::mem::size_of::<Saturating<u8>>() * 4);
}

#[test]
fn size_assert_mono16() {
    let _ = <TestMono16 as HomogeneousPixel>::_SIZE_ASSERT;
    assert_eq!(TestMono16::SIZE, std::mem::size_of::<u16>());
}

#[test]
fn size_assert_rgb16() {
    let _ = <TestRgb16 as HomogeneousPixel>::_SIZE_ASSERT;
    assert_eq!(TestRgb16::SIZE, std::mem::size_of::<Saturating<u16>>() * 3);
}

#[test]
fn size_assert_transparent_named() {
    let _ = <TestWrappedU8 as HomogeneousPixel>::_SIZE_ASSERT;
    assert_eq!(TestWrappedU8::SIZE, std::mem::size_of::<Saturating<u8>>());
}

#[test]
fn size_assert_transparent_tuple() {
    let _ = <TestWrappedU16 as HomogeneousPixel>::_SIZE_ASSERT;
    assert_eq!(TestWrappedU16::SIZE, std::mem::size_of::<u16>());
}

// ===========================================================================
// channel() — read individual channels
// ===========================================================================

#[test]
fn channel_read_rgb() {
    let px = TestRgb {
        r: Saturating(10),
        g: Saturating(20),
        b: Saturating(30),
    };
    assert_eq!(px.channel(0), Saturating(10u8));
    assert_eq!(px.channel(1), Saturating(20u8));
    assert_eq!(px.channel(2), Saturating(30u8));
}

#[test]
fn channel_read_rgba() {
    let px = TestRgba {
        r: Saturating(1),
        g: Saturating(2),
        b: Saturating(3),
        a: Saturating(4),
    };
    assert_eq!(px.channel(0), Saturating(1u8));
    assert_eq!(px.channel(1), Saturating(2u8));
    assert_eq!(px.channel(2), Saturating(3u8));
    assert_eq!(px.channel(3), Saturating(4u8));
}

#[test]
fn channel_read_mono16() {
    let px = TestMono16 { value: 0xABCD };
    assert_eq!(px.channel(0), 0xABCDu16);
}

#[test]
fn channel_read_rgb16() {
    let px = TestRgb16 {
        r: Saturating(100),
        g: Saturating(200),
        b: Saturating(300),
    };
    assert_eq!(px.channel(0), Saturating(100u16));
    assert_eq!(px.channel(1), Saturating(200u16));
    assert_eq!(px.channel(2), Saturating(300u16));
}

#[test]
fn channel_read_tuple() {
    let px = TestTupleRgb(Saturating(11), Saturating(22), Saturating(33));
    assert_eq!(px.channel(0), Saturating(11u8));
    assert_eq!(px.channel(1), Saturating(22u8));
    assert_eq!(px.channel(2), Saturating(33u8));
}

#[test]
fn channel_read_transparent() {
    let px = TestWrappedU8 {
        value: Saturating(42),
    };
    assert_eq!(px.channel(0), Saturating(42u8));
}

#[test]
#[should_panic]
fn channel_read_out_of_bounds() {
    let px = TestRgb {
        r: Saturating(0),
        g: Saturating(0),
        b: Saturating(0),
    };
    let _ = px.channel(3); // only 0..3 valid
}

// ===========================================================================
// set_channel() — write individual channels
// ===========================================================================

#[test]
fn set_channel_rgb() {
    let mut px = TestRgb {
        r: Saturating(0),
        g: Saturating(0),
        b: Saturating(0),
    };
    px.set_channel(0, Saturating(10));
    px.set_channel(1, Saturating(20));
    px.set_channel(2, Saturating(30));
    assert_eq!(px.r, Saturating(10));
    assert_eq!(px.g, Saturating(20));
    assert_eq!(px.b, Saturating(30));
}

#[test]
fn set_channel_rgba() {
    let mut px = TestRgba {
        r: Saturating(0),
        g: Saturating(0),
        b: Saturating(0),
        a: Saturating(0),
    };
    px.set_channel(3, Saturating(255));
    assert_eq!(px.a, Saturating(255));
}

#[test]
fn set_channel_mono16() {
    let mut px = TestMono16 { value: 0 };
    px.set_channel(0, 0xBEEF);
    assert_eq!(px.value, 0xBEEF);
}

#[test]
#[should_panic]
fn set_channel_out_of_bounds() {
    let mut px = TestRgb {
        r: Saturating(0),
        g: Saturating(0),
        b: Saturating(0),
    };
    px.set_channel(3, Saturating(99));
}

// ===========================================================================
// to_channels() — decompose to array
// ===========================================================================

#[test]
fn to_channels_rgb() {
    let px = TestRgb {
        r: Saturating(10),
        g: Saturating(20),
        b: Saturating(30),
    };
    let ch = px.to_channels();
    assert_eq!(ch, [Saturating(10u8), Saturating(20), Saturating(30)]);
}

#[test]
fn to_channels_rgba() {
    let px = TestRgba {
        r: Saturating(1),
        g: Saturating(2),
        b: Saturating(3),
        a: Saturating(4),
    };
    let ch = px.to_channels();
    assert_eq!(
        ch,
        [Saturating(1u8), Saturating(2), Saturating(3), Saturating(4)]
    );
}

#[test]
fn to_channels_mono16() {
    let px = TestMono16 { value: 12345 };
    let ch = px.to_channels();
    assert_eq!(ch, [12345u16]);
}

#[test]
fn to_channels_rgb16() {
    let px = TestRgb16 {
        r: Saturating(100),
        g: Saturating(200),
        b: Saturating(300),
    };
    let ch = px.to_channels();
    assert_eq!(ch, [Saturating(100u16), Saturating(200), Saturating(300)]);
}

#[test]
fn to_channels_tuple() {
    let px = TestTuplePair(0xAAAA, 0xBBBB);
    let ch = px.to_channels();
    assert_eq!(ch, [0xAAAAu32, 0xBBBBu32]);
}

#[test]
fn to_channels_transparent() {
    let px = TestWrappedU16(0xDEAD);
    let ch = px.to_channels();
    assert_eq!(ch, [0xDEADu16]);
}

// ===========================================================================
// from_channels() — construct from channel slice
// ===========================================================================

#[test]
fn from_channels_rgb() {
    let ch = [Saturating(10u8), Saturating(20), Saturating(30)];
    let px = TestRgb::from_channels(&ch);
    assert_eq!(
        px,
        TestRgb {
            r: Saturating(10),
            g: Saturating(20),
            b: Saturating(30),
        }
    );
}

#[test]
fn from_channels_rgba() {
    let ch = [Saturating(1u8), Saturating(2), Saturating(3), Saturating(4)];
    let px = TestRgba::from_channels(&ch);
    assert_eq!(
        px,
        TestRgba {
            r: Saturating(1),
            g: Saturating(2),
            b: Saturating(3),
            a: Saturating(4),
        }
    );
}

#[test]
fn from_channels_mono16() {
    let ch = [54321u16];
    let px = TestMono16::from_channels(&ch);
    assert_eq!(px, TestMono16 { value: 54321 });
}

#[test]
fn from_channels_tuple_pair() {
    let ch = [0x1111u32, 0x2222];
    let px = TestTuplePair::from_channels(&ch);
    assert_eq!(px, TestTuplePair(0x1111, 0x2222));
}

#[test]
fn from_channels_transparent() {
    let ch = [Saturating(77u8)];
    let px = TestWrappedU8::from_channels(&ch);
    assert_eq!(
        px,
        TestWrappedU8 {
            value: Saturating(77)
        }
    );
}

#[test]
#[should_panic]
fn from_channels_wrong_count_too_few() {
    let ch = [Saturating(1u8), Saturating(2)];
    let _ = TestRgb::from_channels(&ch); // expects 3
}

#[test]
#[should_panic]
fn from_channels_wrong_count_too_many() {
    let ch = [Saturating(1u8), Saturating(2), Saturating(3), Saturating(4)];
    let _ = TestRgb::from_channels(&ch); // expects 3
}

// ===========================================================================
// Roundtrip: from_channels(to_channels) == identity
// ===========================================================================

#[test]
fn roundtrip_rgb() {
    let original = TestRgb {
        r: Saturating(100),
        g: Saturating(150),
        b: Saturating(200),
    };
    let channels = original.to_channels();
    let reconstructed = TestRgb::from_channels(channels.as_ref());
    assert_eq!(original, reconstructed);
}

#[test]
fn roundtrip_rgba() {
    let original = TestRgba {
        r: Saturating(10),
        g: Saturating(20),
        b: Saturating(30),
        a: Saturating(40),
    };
    let channels = original.to_channels();
    let reconstructed = TestRgba::from_channels(channels.as_ref());
    assert_eq!(original, reconstructed);
}

#[test]
fn roundtrip_rgb16() {
    let original = TestRgb16 {
        r: Saturating(1000),
        g: Saturating(2000),
        b: Saturating(3000),
    };
    let channels = original.to_channels();
    let reconstructed = TestRgb16::from_channels(channels.as_ref());
    assert_eq!(original, reconstructed);
}

#[test]
fn roundtrip_mono16() {
    let original = TestMono16 { value: 0xFACE };
    let channels = original.to_channels();
    let reconstructed = TestMono16::from_channels(channels.as_ref());
    assert_eq!(original, reconstructed);
}

#[test]
fn roundtrip_transparent() {
    let original = TestWrappedU16(0xCAFE);
    let channels = original.to_channels();
    let reconstructed = TestWrappedU16::from_channels(channels.as_ref());
    assert_eq!(original, reconstructed);
}

// ===========================================================================
// Roundtrip with set_channel: write channels back one by one
// ===========================================================================

#[test]
fn set_channel_roundtrip_rgb() {
    let original = TestRgb {
        r: Saturating(11),
        g: Saturating(22),
        b: Saturating(33),
    };
    let mut copy = TestRgb {
        r: Saturating(0),
        g: Saturating(0),
        b: Saturating(0),
    };
    for i in 0..TestRgb::CHANNEL_COUNT {
        copy.set_channel(i, original.channel(i));
    }
    assert_eq!(original, copy);
}

// ===========================================================================
// Interaction with PlainPixel trait
// ===========================================================================

#[test]
fn plain_pixel_channels_consistent_with_uniform_rgb() {
    // PlainPixel::DIM and HomogeneousPixel::CHANNEL_COUNT should agree
    // when all channels are the same size as Channel.
    assert_eq!(TestRgb::DIM, TestRgb::CHANNEL_COUNT);
}

#[test]
fn plain_pixel_channels_consistent_with_uniform_rgba() {
    assert_eq!(TestRgba::DIM, TestRgba::CHANNEL_COUNT);
}

#[test]
fn plain_pixel_channels_consistent_with_uniform_mono16() {
    assert_eq!(TestMono16::DIM, TestMono16::CHANNEL_COUNT);
}

#[test]
fn as_bytes_matches_channel_bytes() {
    let px = TestRgb {
        r: Saturating(10),
        g: Saturating(20),
        b: Saturating(30),
    };
    assert_eq!(px.as_bytes(), &[10, 20, 30]);
}

// ===========================================================================
// Test with the standard pixel types (Rgb8, u8)
// ===========================================================================

#[test]
fn standard_u8_uniform_pixel() {
    use fovea::pixel::Rgb8;

    // u8 has a manual HomogeneousPixel impl
    assert_eq!(<u8 as HomogeneousPixel>::CHANNEL_COUNT, 1);
    let val: u8 = 42;
    assert_eq!(val.channel(0), 42u8);
    assert_eq!(val.to_channels(), [42u8]);
    assert_eq!(u8::from_channels(&[99]), 99u8);

    // Rgb8 has a manual HomogeneousPixel impl
    assert_eq!(<Rgb8 as HomogeneousPixel>::CHANNEL_COUNT, 3);
    let px = Rgb8::new(10, 20, 30);
    let ch = px.to_channels();
    assert_eq!(ch, [Saturating(10u8), Saturating(20), Saturating(30)]);
}

// ===========================================================================
// Generic function test — ensures the derive generates proper trait impls
// ===========================================================================

fn swap_first_last<P: HomogeneousPixel>(pixel: &P) -> P
where
    P::Channel: Copy,
{
    let mut channels: Vec<P::Channel> = Vec::new();
    for i in 0..P::CHANNEL_COUNT {
        channels.push(pixel.channel(i));
    }
    channels.swap(0, P::CHANNEL_COUNT - 1);
    P::from_channels(&channels)
}

#[test]
fn generic_swap_first_last_rgb() {
    let px = TestRgb {
        r: Saturating(10),
        g: Saturating(20),
        b: Saturating(30),
    };
    let swapped = swap_first_last(&px);
    assert_eq!(
        swapped,
        TestRgb {
            r: Saturating(30),
            g: Saturating(20),
            b: Saturating(10),
        }
    );
}

#[test]
fn generic_swap_first_last_rgba() {
    let px = TestRgba {
        r: Saturating(1),
        g: Saturating(2),
        b: Saturating(3),
        a: Saturating(4),
    };
    let swapped = swap_first_last(&px);
    assert_eq!(
        swapped,
        TestRgba {
            r: Saturating(4),
            g: Saturating(2),
            b: Saturating(3),
            a: Saturating(1),
        }
    );
}

#[test]
fn generic_swap_first_last_mono_is_identity() {
    let px = TestMono16 { value: 999 };
    let swapped = swap_first_last(&px);
    assert_eq!(swapped, px);
}

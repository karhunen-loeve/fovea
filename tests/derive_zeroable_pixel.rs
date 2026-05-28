use irys_cv::ZeroablePixel;
use irys_cv::pixel::ZeroablePixel as ZeroablePixelTrait;
use std::num::Saturating;

// ---------------------------------------------------------------------------
// Basic derive (no attributes) — named struct
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, ZeroablePixel)]
#[repr(C)]
struct TestRgb {
    r: Saturating<u8>,
    g: Saturating<u8>,
    b: Saturating<u8>,
}

#[test]
fn named_struct_zero_all_fields_are_zero() {
    let z = TestRgb::zero();
    assert_eq!(z.r, Saturating(0));
    assert_eq!(z.g, Saturating(0));
    assert_eq!(z.b, Saturating(0));
}

// ---------------------------------------------------------------------------
// Basic derive (no attributes) — tuple struct
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, ZeroablePixel)]
#[repr(transparent)]
struct TestMono(Saturating<u16>);

#[test]
fn tuple_struct_zero() {
    let z = TestMono::zero();
    assert_eq!(z.0, Saturating(0));
}

// ---------------------------------------------------------------------------
// Multi-field tuple struct
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, ZeroablePixel)]
#[repr(C)]
struct TestTuple(Saturating<u8>, Saturating<u8>, Saturating<u8>);

#[test]
fn tuple_struct_multi_field_zero() {
    let z = TestTuple::zero();
    assert_eq!(z.0, Saturating(0));
    assert_eq!(z.1, Saturating(0));
    assert_eq!(z.2, Saturating(0));
}

// ---------------------------------------------------------------------------
// Helper types for attribute tests
// ---------------------------------------------------------------------------

/// A type that implements both `ZeroablePixel` and `Default` (with different values)
/// so we can verify which strategy was actually used.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
struct Channel(u8);

impl ZeroablePixelTrait for Channel {
    fn zero() -> Self {
        Channel(0)
    }
}

impl Default for Channel {
    fn default() -> Self {
        // Intentionally different from zero() so we can distinguish them.
        Channel(42)
    }
}

/// A type that only implements `Default`, not `ZeroablePixel`.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
#[repr(C)]
struct Meta {
    tag: u32,
}

// ---------------------------------------------------------------------------
// #[zero(default)] on named struct
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, ZeroablePixel)]
#[repr(C)]
struct WithDefault {
    ch: Channel,
    #[zero(default)]
    ch_via_default: Channel,
}

#[test]
fn named_struct_zero_default_attr() {
    let z = WithDefault::zero();
    // `ch` uses ZeroablePixel::zero() → Channel(0)
    assert_eq!(z.ch, Channel(0));
    // `ch_via_default` uses Default::default() → Channel(42)
    assert_eq!(z.ch_via_default, Channel(42));
}

// ---------------------------------------------------------------------------
// #[zero(default)] for a type that only implements Default
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, ZeroablePixel)]
#[repr(C)]
struct WithMetaDefault {
    ch: Channel,
    #[zero(default)]
    meta: Meta,
}

#[test]
fn named_struct_zero_default_only_type() {
    let z = WithMetaDefault::zero();
    assert_eq!(z.ch, Channel(0));
    assert_eq!(z.meta, Meta { tag: 0 }); // Default for u32 is 0
}

// ---------------------------------------------------------------------------
// #[zero(<expr>)] with a literal
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, ZeroablePixel)]
#[repr(C)]
struct WithExprLiteral {
    ch: Channel,
    #[zero(99u16)]
    value: u16,
}

#[test]
fn named_struct_zero_expr_literal() {
    let z = WithExprLiteral::zero();
    assert_eq!(z.ch, Channel(0));
    assert_eq!(z.value, 99u16);
}

// ---------------------------------------------------------------------------
// #[zero(<expr>)] with a constructor call
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, ZeroablePixel)]
#[repr(C)]
struct WithExprConstructor {
    ch: Channel,
    #[zero(Channel(7))]
    special: Channel,
}

#[test]
fn named_struct_zero_expr_constructor() {
    let z = WithExprConstructor::zero();
    assert_eq!(z.ch, Channel(0));
    assert_eq!(z.special, Channel(7));
}

// ---------------------------------------------------------------------------
// #[zero(<expr>)] with a block expression
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, ZeroablePixel)]
#[repr(C)]
struct WithExprBlock {
    #[zero({ let v: u8 = 1 + 2; Saturating(v) })]
    value: Saturating<u8>,
}

#[test]
fn named_struct_zero_expr_block() {
    let z = WithExprBlock::zero();
    assert_eq!(z.value, Saturating(3));
}

// ---------------------------------------------------------------------------
// Mixed strategies on a single struct
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, ZeroablePixel)]
#[repr(C)]
struct MixedNamed {
    /// Uses default ZeroablePixel::zero() strategy
    r: Channel,
    /// Uses Default::default() strategy
    #[zero(default)]
    meta: Meta,
    /// Uses a literal expression
    #[zero(42u16)]
    value: u16,
    /// Uses a constructor expression
    #[zero(Channel(100))]
    special: Channel,
}

#[test]
fn named_struct_mixed_strategies() {
    let z = MixedNamed::zero();
    assert_eq!(z.r, Channel(0));
    assert_eq!(z.meta, Meta { tag: 0 });
    assert_eq!(z.value, 42u16);
    assert_eq!(z.special, Channel(100));
}

// ---------------------------------------------------------------------------
// Tuple struct with #[zero(default)]
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, ZeroablePixel)]
#[repr(transparent)]
struct TupleDefault(#[zero(default)] Channel);

#[test]
fn tuple_struct_zero_default_attr() {
    let z = TupleDefault::zero();
    // Default::default() → Channel(42)
    assert_eq!(z.0, Channel(42));
}

// ---------------------------------------------------------------------------
// Tuple struct with #[zero(<expr>)]
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, ZeroablePixel)]
#[repr(transparent)]
struct TupleExpr(#[zero(255u8)] u8);

#[test]
fn tuple_struct_zero_expr_attr() {
    let z = TupleExpr::zero();
    assert_eq!(z.0, 255u8);
}

// ---------------------------------------------------------------------------
// Tuple struct with mixed strategies
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, ZeroablePixel)]
#[repr(C)]
struct TupleMixed(
    Channel,
    #[zero(default)] Channel,
    #[zero(Channel(50))] Channel,
);

#[test]
fn tuple_struct_mixed_strategies() {
    let z = TupleMixed::zero();
    assert_eq!(z.0, Channel(0)); // ZeroablePixel::zero()
    assert_eq!(z.1, Channel(42)); // Default::default()
    assert_eq!(z.2, Channel(50)); // Channel(50)
}

// ---------------------------------------------------------------------------
// Derive with primitive types (no attributes needed)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, ZeroablePixel)]
#[repr(C)]
struct PrimitiveStruct {
    a: u8,
    b: u16,
    c: u32,
    // ADR-0044 Phase E: `f32` no longer implements `ZeroablePixel`
    // (it is a channel, not a pixel). Use `#[zero(default)]` to
    // zero-initialize via `<f32 as Default>::default()` instead of
    // the (now-removed) `<f32 as ZeroablePixel>::zero()`.
    #[zero(default)]
    d: f32,
}

#[test]
fn primitive_fields_zero() {
    let z = PrimitiveStruct::zero();
    assert_eq!(z.a, 0u8);
    assert_eq!(z.b, 0u16);
    assert_eq!(z.c, 0u32);
    assert_eq!(z.d, 0.0f32);
}

// ---------------------------------------------------------------------------
// Ensure derived zero() is consistent with manual impl
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, ZeroablePixel)]
#[repr(C)]
struct DerivedRgba {
    r: Saturating<u8>,
    g: Saturating<u8>,
    b: Saturating<u8>,
    a: Saturating<u8>,
}

#[test]
fn derived_matches_manual_zero() {
    let derived = DerivedRgba::zero();
    let manual = DerivedRgba {
        r: Saturating(0),
        g: Saturating(0),
        b: Saturating(0),
        a: Saturating(0),
    };
    assert_eq!(derived, manual);
}

// ---------------------------------------------------------------------------
// Default impl equals zero
// ---------------------------------------------------------------------------

#[test]
fn zeroable_pixel_default_equals_zero() {
    use irys_cv::pixel::{Mono8, Rgb8, ZeroablePixel};
    assert_eq!(Rgb8::default(), Rgb8::zero());
    assert_eq!(Mono8::default(), Mono8::zero());
}

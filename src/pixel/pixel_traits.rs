use std::mem::{align_of, size_of};
use std::ops::Add;

use std::borrow::Cow;

/// Maximum pixel size in bytes supported by stack-buffer operations.
///
/// This must be at least as large as the largest pixel type in the library
/// (currently `RgbaF64` at 32 bytes). The value of 64 provides a comfortable
/// margin for custom pixel types.
pub(crate) const MAX_PIXEL_SIZE: usize = 64;

mod sealed {
    pub trait SealedArray<T> {}
    impl<T, const N: usize> SealedArray<T> for [T; N] {}
}

pub trait Array<T>: sealed::SealedArray<T> + AsRef<[T]> + AsMut<[T]> {
    const LEN: usize;
    type Map<U>: Array<U>;
    fn from_fn<F: FnMut(usize) -> T>(f: F) -> Self;
}

impl<T, const N: usize> Array<T> for [T; N] {
    const LEN: usize = N;
    type Map<U> = [U; N];
    fn from_fn<F: FnMut(usize) -> T>(f: F) -> Self {
        std::array::from_fn(f)
    }
}

/// Computes the sum of channel sizes at compile time.
const fn sum_channels(channels: &[usize]) -> usize {
    let mut sum = 0;
    let mut i = 0;
    while i < channels.len() {
        sum += channels[i];
        i += 1;
    }
    sum
}

/// A fixed-layout byte-addressable primitive suitable to appear as a
/// channel of a [`HomogeneousPixel`].
///
/// `PlainChannel` is the channel-role sibling of [`PlainPixel`]: it
/// describes the byte-layout-only subset of `PlainPixel`'s contract
/// — what you need to round-trip a value through `&[u8]` — without
/// the pixel-role items (`CHANNELS`, `DIM`, `cast_slice`, endian
/// helpers) that only make sense for a semantic pixel unit.
///
/// Role split (ADR-0046, structural companion to ADR-0045):
///
/// - **Channel role** (`PlainChannel`): a scalar component with a
///   stable byte layout. Implemented by primitives `u{8,16,32,64}`,
///   `i{8,16,32,64}`, and — crucially — `f32` / `f64`. Raw floats
///   are first-class channels but never first-class pixels
///   (ADR-0044).
/// - **Pixel role** (`PlainPixel`): a semantic pixel unit. Extends
///   `PlainChannel` because byte layout is role-blind — a pixel's
///   bytes are pixel bytes whether you query them as the pixel or
///   as a one-element channel stream. See `PHILOSOPHY.md` §9.
///
/// # Safety
///
/// 1. **Stable memory layout.** `size_of::<Self>() == SIZE` and the
///    bit pattern of `&self` is a contiguous `SIZE`-byte region with
///    no padding.
/// 2. **Valid for any bit pattern.** Any `[u8; SIZE]` is a valid
///    value of `Self`. (`f32` NaN bit patterns are valid — no UB to
///    construct — even though they may not represent meaningful
///    channel intensities.)
/// 3. **Round-trippable through `&[u8]`.**
///    `from_bytes(self.as_bytes()) == Some(self)` for every
///    `self: Self`.
pub unsafe trait PlainChannel: Sized + Copy {
    /// The total size in bytes of the channel.
    const SIZE: usize = size_of::<Self>();
    /// The alignment requirement of the channel in bytes.
    const ALIGN: usize = align_of::<Self>();

    /// Compile-time assertion: the overrideable `SIZE` / `ALIGN`
    /// constants must match the actual memory layout of `Self`, and
    /// `Self` must not be a ZST.
    ///
    /// `cast_slice`, `cast_slice_mut`, byte constructors, and the endian
    /// helpers all consult `Self::SIZE` and `Self::ALIGN`. Because both
    /// constants are defaulted but overrideable, a hand-written or
    /// derive-emitted impl that reports the wrong value would silently
    /// invalidate every unsafe byte path in the crate. This assertion is
    /// referenced from the byte helpers so it is forced for every concrete
    /// `Self` that takes any layout-dependent code path.
    const _ASSERT_SIZE: () = {
        assert!(size_of::<Self>() > 0, "SIZE must be > 0");
        assert!(
            Self::SIZE == size_of::<Self>(),
            "PlainChannel::SIZE must equal size_of::<Self>()"
        );
        assert!(
            Self::ALIGN == align_of::<Self>(),
            "PlainChannel::ALIGN must equal align_of::<Self>()"
        );
    };

    /// Convert the channel to a byte slice (native endianness).
    fn as_bytes(&self) -> &[u8] {
        // Force the compile-time layout assertion for every concrete `Self`
        // that touches any byte-level helper.
        let () = Self::_ASSERT_SIZE;
        unsafe { crate::internal::as_bytes(std::slice::from_ref(self)) }
    }

    /// Convert a byte slice to a channel (native endianness).
    ///
    /// Returns `None` if `bytes.len()` does not equal [`Self::SIZE`].
    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let () = Self::_ASSERT_SIZE;
        crate::internal::from_bytes(bytes)
    }
}

/// The `PlainPixel` trait defines a pixel type with a fixed number of channels
/// designed for byte conversion and storage.
///
/// `PlainPixel` is the pixel-role sibling of [`PlainChannel`]: it
/// extends the byte-layout channel contract with the pixel-level
/// items (`CHANNELS`, `DIM`, endian helpers, `cast_slice`) that
/// only make sense for a semantic pixel unit. The supertrait
/// relation is correct because byte layout is role-blind — see
/// `PHILOSOPHY.md` §9 and ADR-0046.
///
/// # Safety
///
/// Implementers of this trait must guarantee the following invariants:
///
/// 1. **Memory Layout**: The type must have a well-defined, stable memory layout
///    that is compatible with byte-level operations. This typically means:
///    - No padding bytes between fields (use `#[repr(C)]` or `#[repr(packed)]`)
///    - All fields must be `Copy` types with predictable byte representations
///
/// 2. **Channel Consistency**: The `DIM` constant must accurately reflect the number
///    of logical channels in the pixel, and `channel_sizes()` must return a vector
///    with exactly `DIM` elements where each element represents the size in bytes
///    of the corresponding channel.
///
/// 3. **Byte Conversion Safety**: The type must be safe to transmute to/from byte
///    arrays. This means:
///    - No invalid bit patterns that would cause undefined behavior
///    - All possible byte combinations must be values the type can hold without UB
///      (e.g. `f32` NaN bit patterns are valid — no UB to construct — even though
///      they may not represent meaningful pixel intensities)
///    - The size of the type must equal the sum of all channel sizes
///
/// 4. **Endianness Handling**: The type must handle endianness conversion correctly
///    in `as_bytes_le()` and `as_bytes_be()` methods, ensuring data integrity
///    across different architectures.
///
/// Violating these invariants may result in undefined behavior, data corruption,
/// or memory safety issues.
pub unsafe trait PlainPixel: PlainChannel {
    /// The structural dimension of the pixel (number of channels and their sizes).
    const CHANNELS: &'static [usize];
    /// The total number of channels in the pixel.
    const DIM: usize = Self::CHANNELS.len();
    // `SIZE`, `ALIGN`, `_ASSERT_SIZE`, `as_bytes`, and `from_bytes`
    // are inherited from `PlainChannel` (ADR-0046) — call sites
    // resolve them through `<T as PlainChannel>::...` or bring
    // `PlainChannel` into scope alongside `PlainPixel`.

    /// Compile-time assertion: size must equal sum of channel sizes.
    const _ASSERT_CHANNELS: () = assert!(
        size_of::<Self>() == sum_channels(Self::CHANNELS),
        "SIZE must equal sum of CHANNELS"
    );

    // Convert the pixel to a mutable byte slice (native endianness)
    fn as_mut_bytes(&mut self) -> &mut [u8] {
        unsafe { crate::internal::as_mut_bytes(std::slice::from_mut(self)) }
    }
    /// Convert the pixel to bytes in little-endian format.
    ///
    /// On little-endian platforms this borrows the underlying memory (zero-copy).
    /// On big-endian platforms this allocates a new `Vec` with swapped bytes.
    fn as_bytes_le(&self) -> Cow<'_, [u8]> {
        unsafe { crate::internal::as_bytes_le(std::slice::from_ref(self)) }
    }
    /// Convert the pixel to bytes in big-endian format.
    ///
    /// On big-endian platforms this borrows the underlying memory (zero-copy).
    /// On little-endian platforms this allocates a new `Vec` with swapped bytes.
    fn as_bytes_be(&self) -> Cow<'_, [u8]> {
        unsafe { crate::internal::as_bytes_be(std::slice::from_ref(self)) }
    }
    /// Convert a byte slice to a pixel (big-endian format).
    ///
    /// Returns `None` if `bytes.len()` does not equal [`PlainChannel::SIZE`].
    fn from_bytes_be(bytes: &[u8]) -> Option<Self> {
        crate::internal::from_bytes_be(bytes)
    }
    /// Convert a byte slice to a pixel (little-endian format).
    ///
    /// Returns `None` if `bytes.len()` does not equal [`PlainChannel::SIZE`].
    fn from_bytes_le(bytes: &[u8]) -> Option<Self> {
        crate::internal::from_bytes_le(bytes)
    }

    /// Reinterpret a byte slice as a slice of pixels (zero-copy).
    ///
    /// Returns `None` if `bytes.len()` is not a multiple of [`PlainChannel::SIZE`],
    /// or if the pointer is not aligned to [`PlainChannel::ALIGN`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use irys_cv::pixel::{PlainPixel, Mono8};
    /// let raw = [10u8, 20, 30];
    /// let pixels: &[Mono8] = Mono8::cast_slice(&raw).unwrap();
    /// assert_eq!(pixels.len(), 3);
    /// ```
    fn cast_slice(bytes: &[u8]) -> Option<&[Self]> {
        let () = <Self as PlainChannel>::_ASSERT_SIZE;
        if bytes.len() % <Self as PlainChannel>::SIZE != 0 {
            return None;
        }
        if (bytes.as_ptr() as usize) % <Self as PlainChannel>::ALIGN != 0 {
            return None;
        }
        let count = bytes.len() / <Self as PlainChannel>::SIZE;
        // SAFETY: PlainPixel guarantees Self is valid for any bit pattern.
        // We verified alignment and that the byte length is an exact multiple of SIZE.
        Some(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const Self, count) })
    }

    /// Reinterpret a mutable byte slice as a mutable slice of pixels (zero-copy).
    ///
    /// Returns `None` if `bytes.len()` is not a multiple of [`PlainChannel::SIZE`],
    /// or if the pointer is not aligned to [`PlainChannel::ALIGN`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use irys_cv::pixel::{PlainPixel, Mono8};
    /// let mut raw = [10u8, 20, 30];
    /// let pixels: &mut [Mono8] = Mono8::cast_slice_mut(&mut raw).unwrap();
    /// assert_eq!(pixels.len(), 3);
    /// ```
    fn cast_slice_mut(bytes: &mut [u8]) -> Option<&mut [Self]> {
        let () = <Self as PlainChannel>::_ASSERT_SIZE;
        if bytes.len() % <Self as PlainChannel>::SIZE != 0 {
            return None;
        }
        if (bytes.as_mut_ptr() as usize) % <Self as PlainChannel>::ALIGN != 0 {
            return None;
        }
        let count = bytes.len() / <Self as PlainChannel>::SIZE;
        // SAFETY: PlainPixel guarantees Self is valid for any bit pattern.
        // We verified alignment and that the byte length is an exact multiple of SIZE.
        Some(unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut Self, count) })
    }
}

pub trait ZeroablePixel: Sized + Copy {
    fn zero() -> Self;
}

// ──────────────────────────────────────────────────────────────────────────
// Label pixel role (ADR-0047)
// ──────────────────────────────────────────────────────────────────────────

/// A pixel type whose values name connected-component labels.
///
/// `LabelPixel` is the pixel-role trait that gates label-image
/// producers and consumers in this crate (notably
/// [`connected_components`](crate::analyze::components::connected_components)
/// and [`Labeling`](crate::analyze::components::Labeling)).
///
/// # Contract
///
/// - [`Self::zero()`](ZeroablePixel::zero) is the canonical
///   **background** label — the value the labeling engine writes to
///   every non-foreground pixel.
/// - The set of *foreground* labels is `1 ..= MAX_LABEL`. The labeling
///   engine never produces a label outside that range; if pass 1 would
///   need to, it returns [`Error::LabelOverflow`](crate::Error::LabelOverflow).
/// - [`from_label_index`](Self::from_label_index) and
///   [`to_label_index`](Self::to_label_index) must round-trip the entire
///   foreground range: for every `i in 1..=MAX_LABEL`,
///   `from_label_index(i).unwrap().to_label_index() == i`.
///
/// # Safe trait
///
/// `LabelPixel` is a **safe** trait. A wrong impl produces numerically
/// wrong labels or a wrong [`LabelOverflow`](crate::Error::LabelOverflow)
/// boundary, never undefined behaviour. Compare [`PlainPixel`], which is
/// `unsafe` because a wrong impl reinterprets bytes. Philosophy §11
/// ("if it can be written without unsafe, it must be") therefore keeps
/// the trait safe and pushes the correctness obligation to the
/// implementor's documentation.
///
/// # Deliberate non-extension
///
/// `LabelPixel` does **not** extend [`LinearPixel`], [`BoundedChannel`],
/// [`WhiteChannel`], [`LinearChannel`], [`LinearSpace`], or
/// [`FromLinear`]. Labels are not intensities — averaging two labels,
/// gamma-converting them, inverting them, or thresholding them is
/// meaningless. Excluding those traits is the type-level fence that
/// makes such operations *fail to compile* on label images
/// (Philosophy §1). Label-image capacity is exposed via
/// [`MAX_LABEL`](Self::MAX_LABEL) instead — a different,
/// label-specific concept from `BoundedChannel::MAX`.
pub trait LabelPixel: Copy + Eq + Ord + core::hash::Hash + ZeroablePixel {
    /// The largest distinct foreground label this pixel type can
    /// encode, expressed as `u64` for uniform capacity arithmetic
    /// across all shipped and future label widths.
    ///
    /// Background (= [`Self::zero()`](ZeroablePixel::zero)) is **not**
    /// counted; the valid foreground range is `1 ..= MAX_LABEL`.
    const MAX_LABEL: u64;

    /// Construct a label from a 1-based foreground index.
    ///
    /// Returns `None` if `index == 0` (background is not produced by
    /// this constructor; use [`Self::zero()`](ZeroablePixel::zero)
    /// instead) or `index > MAX_LABEL`.
    fn from_label_index(index: u64) -> Option<Self>;

    /// Read the label back as a `u64`. [`Self::zero()`](ZeroablePixel::zero)
    /// returns `0`; any foreground label `i` returns `i`.
    fn to_label_index(self) -> u64;
}

// ─────────────────────────────────────────────────────────────────────────────
// Integral-image source/accumulator gating (ADR-0032)
// ─────────────────────────────────────────────────────────────────────────────
//
// These traits gate the *valid combinations* of source pixel `Self` and
// accumulator pixel `A` for the summed-area-table engines in
// [`crate::analyze::integral`]. Both traits are **safe** — see the rationale
// in the [module-level doc on `IntegralPixel`](IntegralPixel).
//
// The pre-flight overflow check in `analyze::integral::preflight` reads
// `max_integral_value()` / `max_integral_squared_value()` to decide whether
// the chosen accumulator can hold the worst-case sum for the given image
// dimensions (ADR-0032 §3). The check is `O(1)`; the inner loop has zero
// per-pixel overhead.

/// Connects a pixel type to a valid integral-image accumulator pixel.
///
/// Implementing this trait declares that `Self` can be losslessly projected
/// into an accumulator pixel of type `A` for the purposes of computing a
/// summed-area table.
///
/// # Correctness contract — not a soundness contract
///
/// Unlike [`PlainPixel`] (which is `unsafe` because a wrong impl
/// reinterprets bytes), `IntegralPixel` is a *safe* trait. A wrong
/// [`max_integral_value`](Self::max_integral_value) does not cause UB; it
/// causes the pre-flight overflow check
/// ([`analyze::integral`](crate::analyze::integral)) to return a
/// **numerically wrong** answer (the check may pass when the accumulator
/// would actually saturate, producing a silently incorrect integral image).
/// Implementors must therefore guarantee that `max_integral_value()` is a
/// **tight upper bound** on every value `to_integral()` can produce.
///
/// Philosophy §11 — "if it can be written without unsafe, it must be."
/// ADR-0032 §2 originally specified an `unsafe` trait; the implementation
/// is safe because the contract is a numerical bound, not a layout claim.
///
/// # Float convention
///
/// For float-source pixels (e.g. `MonoF32`, `RgbF32`) the implementation
/// assumes the conventional `[0.0, 1.0]` value range. `max_integral_value`
/// therefore returns the accumulator-pixel value `1.0` on each channel.
/// Callers whose float data lies outside `[0, 1]` must either rescale
/// before calling the integral-image engine, or implement the trait on a
/// newtype that documents its own range. The library reports the
/// convention; it does not silently rescale (Philosophy §8).
///
/// # See also
///
/// - [ADR-0032](https://github.com/karhunen-loeve/irys-cv/blob/main/docs/adr/0032-integral-image-design.md)
/// - [`IntegralSquaredPixel`] — parallel trait for sum-of-squares.
pub trait IntegralPixel<A: Copy>: Copy {
    /// Convert a single source pixel to the accumulator representation.
    ///
    /// This is the only conversion path the engine uses on each input
    /// pixel; it must be branchless and inexpensive.
    fn to_integral(self) -> A;

    /// Tight upper bound on every value [`to_integral`](Self::to_integral)
    /// can produce.
    ///
    /// For integer sources this is a fixed numerical maximum
    /// (e.g. `Mono8 → Mono32` returns `Mono32::new(255)`).
    /// For float sources it returns the accumulator-pixel value `1.0`
    /// under the conventional `[0, 1]` range — see the trait-level docs.
    fn max_integral_value() -> A;
}

/// Connects a pixel type to a valid *squared* integral-image accumulator.
///
/// Parallel to [`IntegralPixel`]; gates the sum-of-squares engine
/// ([`integral_squared_image`](crate::analyze::integral::integral_squared_image)).
/// The per-pixel range is the squared range of the source (e.g.
/// `255² = 65_025` for `Mono8`), so the accumulator-pixel set is
/// strictly tighter than for the non-squared trait — there is no
/// `Mono8 → Mono32` impl here, only `Mono8 → Mono64` and
/// `Mono8 → MonoF64`.
///
/// The correctness contract and float convention match those of
/// [`IntegralPixel`]; see that trait's documentation.
pub trait IntegralSquaredPixel<A: Copy>: Copy {
    /// Convert a single source pixel to its squared accumulator representation.
    fn to_integral_squared(self) -> A;

    /// Tight upper bound on every value
    /// [`to_integral_squared`](Self::to_integral_squared) can produce.
    fn max_integral_squared_value() -> A;
}

/// Pixel-channel types that carry an intrinsic, type-defined maximum
/// representable value.
///
/// This is the counterpart of [`ZeroablePixel`]: where `ZeroablePixel::zero`
/// provides the *zero* of a channel, `BoundedChannel::MAX` provides the
/// *saturated-bright* end of the channel's range (i.e. the value the
/// library treats as "fully saturated / white / opaque").
///
/// Implemented for every integer channel type the library ships
/// (`u8`/`u16`/`u32`/`u64`, `i8`/`i16`/`i32`/`i64`, and their
/// `Saturating<_>` wrappers).
///
/// # Not Implemented for Floats
///
/// `f32` and `f64` deliberately do **not** implement `BoundedChannel`.
/// Floating-point pixels do not have an intrinsic maximum in this
/// library — the convention that "1.0 is white" is a downstream
/// assumption that belongs at the call site, not in the type system.
/// Operations that require a channel-level maximum (e.g. [`Invert`] on a
/// homogeneous pixel type) therefore refuse to compile for float-channel
/// pixels. Users who want float inversion must name their range
/// assumption explicitly (see `PixelMap`).
///
/// [`Invert`]: crate::transform::Invert
///
/// # Example
///
/// ```
/// # use irys_cv::pixel::BoundedChannel;
/// assert_eq!(<u8 as BoundedChannel>::MAX, 255);
/// assert_eq!(<u16 as BoundedChannel>::MAX, u16::MAX);
/// ```
///
/// # Rationale
///
/// See ADR-0042 for the full rationale — briefly, the alternative of
/// macro-enumerating every concrete pixel type (to scatter `MAX`
/// constants inline in per-type impls) would close the operation off to
/// user pixel types and duplicate a property the type system can carry
/// in a single place.
pub trait BoundedChannel: Copy {
    /// The largest value representable by this channel type.
    const MAX: Self;
}

/// Pixel types that declare, for every channel slot, the value that
/// represents "fully saturated" under this pixel type's invariant.
///
/// This is the pixel-level counterpart of [`BoundedChannel`]: where
/// `BoundedChannel::MAX` is a property of a **channel data type**
/// (`Saturating<u16>::MAX = 65535`), `WhiteChannel::white_channel()` is
/// a property of the **pixel type** — which, for reduced-range pixels
/// like [`Mono<BITS>`](crate::pixel::Mono), may be strictly less than
/// the channel type's storage maximum.
///
/// Operations that write a "saturated" channel value back into a
/// homogeneous pixel (`Invert`, `BinaryThreshold`, `BinaryThresholdInv`)
/// bind on this trait to preserve the pixel's invariant.
///
/// # Why this is distinct from `BoundedChannel`
///
/// `Mono<BITS>` uses `Saturating<u16>` as its channel type, but
/// maintains the stronger invariant that the raw channel value fits in
/// `BITS` bits. `<Saturating<u16> as BoundedChannel>::MAX` is `65535`;
/// the pixel-level "white" for `Mono<10>` is `1023`. Writing `65535`
/// back into a `Mono<10>` via `from_channels` (a layout-only primitive)
/// would silently violate the pixel's invariant. `WhiteChannel` exists
/// so strategies can ask "what does *this pixel* consider white" rather
/// than "what is the channel type's storage maximum" — and so the
/// reduced-range pixel can override the answer.
///
/// # Not implemented for float-channel pixels
///
/// Float-channel pixels (`MonoF32`, `RgbF32`, …) deliberately do not
/// implement `WhiteChannel`. Floating-point pixels have no intrinsic
/// "white" in this library — the `[0.0, 1.0]` convention is a
/// downstream assumption that belongs at the call site, not in the type
/// system (Philosophy §8 — "Surface information, don't decide").
///
/// # Rationale
///
/// See ADR-0043 for the full rationale and the concrete `Mono<BITS>`
/// invariant bug that motivated this trait.
pub trait WhiteChannel: HomogeneousPixel {
    /// The value the pixel treats as "fully saturated" on every
    /// channel slot.
    ///
    /// For every homogeneous pixel currently in the library this is
    /// the same value across all channels. The zero-argument shape
    /// reflects that; a future heterogeneous-per-channel "white"
    /// (e.g. a pixel whose alpha has a different saturated value than
    /// its color channels) would warrant a separate trait.
    fn white_channel() -> Self::Channel;
}

/// A pixel type that supports linear interpolation.
///
/// Mathematically, this trait defines a linear map from `Self` into `Output`:
/// - `scale` performs scalar multiplication: `s · v`
/// - `Output` supports addition (forms a vector space with scalar multiplication)
/// - `blend` computes affine combinations: `(1-α)·a + α·b`
///
/// # Linear Space Assumption
///
/// This trait **assumes that linear interpolation is mathematically meaningful**
/// for the pixel values. This is true for:
/// - Linear RGB/RGBA values
/// - Raw sensor intensity values (Mono8, Mono16, etc.)
/// - Floating-point linear light values
///
/// This is **NOT valid** for:
/// - Hue values (cyclic space: blending 10° and 350° should give ~0°, not 180°)
/// - Gamma-encoded sRGB (convert to linear first)
/// - Any cyclical or non-Euclidean value space
///
/// Users working with non-linear spaces are responsible for converting to a
/// linear representation before using `blend`.
///
/// # Output Type
///
/// The `Output` type may differ from `Self` to allow intermediate precision.
/// For example, scaling a `u8` by 0.5 produces a fractional result that cannot
/// be represented in `u8`. Implementations may use a higher-precision type
/// (like `f32`) for `Output`, or round back to the original type.
pub trait LinearPixel<S = f32>: Sized + Copy {
    /// The accumulator type for scaled values. Must support addition.
    /// May be `Self` for types closed under scaling (e.g., `f32`),
    /// or a higher-precision type for integer pixels.
    type Accumulator: Sized + Copy + Add<Output = Self::Accumulator>;

    /// Project this value into the accumulator space without scaling.
    ///
    /// For integer types this performs a widening conversion (e.g. `u8 → f32`).
    /// For float types where `Self::Accumulator = Self`, this is identity.
    ///
    /// This is the conversion half of [`scale`](Self::scale) without the
    /// multiplication, and the inverse direction of
    /// [`FromLinear::from_linear`](crate::pixel::FromLinear::from_linear).
    fn to_accumulator(&self) -> Self::Accumulator;

    /// Scalar multiplication: `scalar · self`
    fn scale(&self, scalar: S) -> Self::Accumulator;

    /// Fused scale-and-accumulate: `self.scale(scalar) + addend`.
    ///
    /// The default performs a separate multiply and add. Implementations
    /// for types whose accumulator supports fused multiply-add (f32, f64)
    /// override this to emit a single FMA instruction when the target
    /// supports it (`vfmadd231ps` on AVX2+FMA3), improving throughput.
    #[inline(always)]
    fn scale_add(&self, scalar: S, addend: Self::Accumulator) -> Self::Accumulator {
        self.scale(scalar) + addend
    }

    /// Broadcast a scalar into the accumulator space: every channel of the
    /// resulting accumulator equals `scalar` (cast to the accumulator's
    /// per-channel type).
    ///
    /// This is the "uniform vector" of the accumulator type — the additive
    /// identity for a given `scalar` in every channel. It is the
    /// counterpart of [`to_accumulator`](Self::to_accumulator) without the
    /// channel-specific input; together with [`scale_add`](Self::scale_add)
    /// it expresses affine per-pixel transforms like brightness + contrast
    /// (`contrast · pixel + brightness`) in a single, vectorizable
    /// expression.
    ///
    /// Channel-level arithmetic is expressed by [`LinearChannel`]. A
    /// pixel's accumulator is a pixel; a channel's accumulator is a scalar.
    /// See ADR-0045 for the channel/pixel split.
    ///
    /// For single-channel pixels (`Mono8`, `MonoF32`, …) this is the
    /// scalar itself (cast to the accumulator's numeric type).
    ///
    /// For multi-channel pixels (`Rgb8`, `Rgba16`, …) the derive macro
    /// delegates to each field's `LinearPixel::uniform`, producing an
    /// accumulator value with the scalar replicated across every channel.
    ///
    /// # Example
    ///
    /// ```
    /// # use irys_cv::pixel::{LinearPixel, MonoF32};
    /// // ADR-0044 Phase E: `f32` is no longer a pixel; the pixel-role
    /// // float type is `MonoF32`. `MonoF32::uniform` returns the
    /// // accumulator (`MonoF32`) with the scalar broadcast across
    /// // every channel — for this single-channel pixel, that's just
    /// // `MonoF32(0.5)`.
    /// let x = <MonoF32 as LinearPixel>::uniform(0.5);
    /// assert_eq!(x, MonoF32(0.5));
    /// ```
    fn uniform(scalar: S) -> Self::Accumulator;
}

/// A channel — one numeric axis of a pixel — that supports linear
/// combinations over the scalar field `S`.
///
/// `LinearChannel` is the arithmetic substrate the derive macro uses
/// when composing a pixel's accumulator from its channel fields. User
/// code that operates on whole pixels binds on [`LinearPixel`], not
/// `LinearChannel`.
///
/// Implemented by every integer channel primitive (`u8`…`u64`,
/// `i8`…`i64`), their `Saturating<_>` wrappers, and the float
/// primitives (`f32`, `f64`). Pixel-named types (`Mono8`, `Rgb8`,
/// `MonoF32`, …) implement [`LinearPixel`], not `LinearChannel`.
///
/// See ADR-0045 for the channel/pixel trait split rationale.
///
/// # Example
///
/// ```
/// # use irys_cv::pixel::LinearChannel;
/// let x = <u8 as LinearChannel<f32>>::scale(&100u8, 0.5);
/// assert_eq!(x, 50.0f32);
/// ```
pub trait LinearChannel<S = f32>: Sized + Copy {
    /// The scalar-space accumulator for this channel.
    ///
    /// For integer channels this is a floating-point widening (e.g.
    /// `u8::Accumulator = f32`). For float channels it is typically
    /// `Self`.
    type Accumulator: Sized + Copy + Add<Output = Self::Accumulator>;

    /// Project this channel value into the accumulator space without
    /// scaling.
    fn to_accumulator(&self) -> Self::Accumulator;

    /// Scalar multiplication: `scalar · self`.
    fn scale(&self, scalar: S) -> Self::Accumulator;

    /// Fused scale-and-accumulate: `self.scale(scalar) + addend`.
    ///
    /// The default performs a separate multiply and add.
    /// Implementations for types whose accumulator supports fused
    /// multiply-add (f32, f64) may override this to emit a single
    /// FMA instruction when the target supports it.
    #[inline(always)]
    fn scale_add(&self, scalar: S, addend: Self::Accumulator) -> Self::Accumulator {
        self.scale(scalar) + addend
    }

    /// Broadcast a scalar into the accumulator space.
    fn uniform(scalar: S) -> Self::Accumulator;
}

/// Converts a linear-space accumulator value back to a storage pixel,
/// applying rounding and clamping as appropriate.
///
/// This trait replaces `Into<P>` in algorithm bounds (like bilinear resize)
/// because the conversion from accumulator to storage pixel is intentionally
/// lossy (rounding, clamping) and the orphan rule prevents implementing
/// `From<f32> for u8`.
pub trait FromLinear<A> {
    fn from_linear(acc: A) -> Self;
}

/// Blanket identity: any type can be "converted" from itself.
impl<T> FromLinear<T> for T {
    #[inline(always)]
    fn from_linear(acc: T) -> Self {
        acc
    }
}

/// Marker trait asserting that a pixel type's values live in a linear mathematical
/// space where interpolation (affine combination) is meaningful.
///
/// This is true for:
/// - Linear RGB/RGBA values
/// - Raw sensor intensity values (Mono8, Mono16, etc.)
/// - Floating-point linear light values
///
/// This is **NOT valid** for:
/// - Hue values (cyclic space: blending 10° and 350° should give ~0°, not 180°)
/// - Gamma-encoded sRGB (convert to linear first)
/// - Any cyclical or non-Euclidean value space
///
/// # Status
///
/// This trait is **required by interpolation algorithms** — notably the
/// [`Bilinear`](crate::transform::Bilinear) resize strategy and the
/// [`blend`] helper above. Algorithms that need to mix pixel values
/// across positions (resize, optical flow warps, alpha compositing in a
/// non-linear-aware path) should bound on `LinearSpace`, not the broader
/// [`LinearPixel`].
///
/// All standard pixel types that derive or implement `LinearPixel` also implement
/// `LinearSpace`. If you have a non-linear pixel type that needs `scale()` for
/// internal use but should *not* be passed to interpolation algorithms, implement
/// `LinearPixel` manually without implementing `LinearSpace`.
pub trait LinearSpace: LinearPixel {}

/// Affine combination (linear interpolation): `(1-alpha)·a + alpha·b`
///
/// When `alpha = 0`, returns `a` (scaled to `Accumulator`).
/// When `alpha = 1`, returns `b` (scaled to `Accumulator`).
/// When `alpha = 0.5`, returns the midpoint.
#[inline(always)]
pub fn blend<T: LinearPixel + LinearSpace>(a: &T, b: &T, alpha: f32) -> T::Accumulator {
    b.scale_add(alpha, a.scale(1.0 - alpha))
}

/// # Safety
///
/// Implementers must guarantee, in addition to `PlainPixel` requirements:
///
/// 1. **Uniform Channels**: The pixel must genuinely consist of `CHANNEL_COUNT`
///    channels, all of type `Channel`. You cannot "reinterpret" the bytes as
///    a different channel decomposition (e.g., treating Rgba8 as two u32s).
///
/// 2. **Exact Memory Layout**: The pixel's memory layout must be exactly
///    `[Channel; CHANNEL_COUNT]` with no padding between channels.
///
/// 3. **Channel Ordering**: The channel order in memory must match the semantic
///    order implied by the pixel type (e.g., Rgb8 stores R at offset 0, G at 1, B at 2).
///
/// The compile-time size assertion catches size mismatches but cannot verify
/// semantic correctness. Incorrect implementations may cause:
/// - Wrong values returned from `channel()` / `to_channels()`
/// - Data corruption in planar ↔ interleaved conversions
/// - Undefined behavior in downstream code relying on channel semantics
pub unsafe trait HomogeneousPixel: PlainPixel {
    type Channel: PlainChannel + Copy;

    /// The channel array type, e.g. `[u8; 3]` for Rgb8.
    type Channels: Array<Self::Channel>;

    const CHANNEL_COUNT: usize = <Self::Channels as Array<Self::Channel>>::LEN;

    const _SIZE_ASSERT: () =
        assert!(size_of::<Self>() == size_of::<Self::Channel>() * Self::CHANNEL_COUNT);

    fn channel(&self, index: usize) -> Self::Channel {
        assert!(index < Self::CHANNEL_COUNT);
        let size = size_of::<Self::Channel>();
        let start = index * size;
        // SAFETY: index is bounds-checked above, so the slice is exactly `size` bytes
        <Self::Channel as PlainChannel>::from_bytes(
            &<Self as PlainChannel>::as_bytes(self)[start..start + size],
        )
        .expect("internal error: channel size mismatch")
    }

    fn set_channel(&mut self, index: usize, value: Self::Channel) {
        assert!(index < Self::CHANNEL_COUNT);
        let size = size_of::<Self::Channel>();
        let start = index * size;
        self.as_mut_bytes()[start..start + size]
            .copy_from_slice(<Self::Channel as PlainChannel>::as_bytes(&value));
    }

    fn from_channels(channels: &[Self::Channel]) -> Self {
        assert_eq!(channels.len(), Self::CHANNEL_COUNT);
        let size = size_of::<Self::Channel>();
        assert!(
            <Self as PlainChannel>::SIZE <= MAX_PIXEL_SIZE,
            "pixel type is larger than the stack buffer ({} > {})",
            <Self as PlainChannel>::SIZE,
            MAX_PIXEL_SIZE
        );
        let mut buf = [0u8; MAX_PIXEL_SIZE];
        let bytes = &mut buf[..<Self as PlainChannel>::SIZE];
        for (i, ch) in channels.iter().enumerate() {
            bytes[i * size..(i + 1) * size]
                .copy_from_slice(<Self::Channel as PlainChannel>::as_bytes(ch));
        }
        // SAFETY: bytes is constructed to be exactly Self::SIZE
        <Self as PlainChannel>::from_bytes(bytes)
            .expect("internal error: constructed byte buf size mismatch")
    }

    fn to_channels(&self) -> Self::Channels {
        Self::Channels::from_fn(|i| self.channel(i))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_channels_empty() {
        assert_eq!(sum_channels(&[]), 0);
    }

    #[test]
    fn test_sum_channels_single() {
        assert_eq!(sum_channels(&[4]), 4);
    }

    #[test]
    fn test_sum_channels_multiple() {
        assert_eq!(sum_channels(&[1, 2, 3]), 6);
    }

    #[test]
    fn test_array_from_fn() {
        let arr: [usize; 4] = <[usize; 4] as Array<usize>>::from_fn(|i| i * 2);
        assert_eq!(arr, [0, 2, 4, 6]);
    }

    #[test]
    fn test_array_len() {
        assert_eq!(<[u8; 3] as Array<u8>>::LEN, 3);
        assert_eq!(<[u8; 1] as Array<u8>>::LEN, 1);
    }

    #[test]
    fn test_cast_slice_mono8() {
        use crate::pixel::Mono8;
        let raw = [10u8, 20, 30];
        let pixels = Mono8::cast_slice(&raw).unwrap();
        assert_eq!(pixels.len(), 3);
        assert_eq!(pixels[0], Mono8::new(10));
        assert_eq!(pixels[1], Mono8::new(20));
        assert_eq!(pixels[2], Mono8::new(30));
    }

    #[test]
    fn test_cast_slice_wrong_length() {
        use crate::pixel::Rgb8;
        // 4 bytes is not a multiple of 3 (Rgb8::SIZE)
        let raw = [1u8, 2, 3, 4];
        assert!(Rgb8::cast_slice(&raw).is_none());
    }

    #[test]
    fn test_cast_slice_empty() {
        use crate::pixel::Mono8;
        let raw: [u8; 0] = [];
        let pixels = Mono8::cast_slice(&raw).unwrap();
        assert_eq!(pixels.len(), 0);
    }

    #[test]
    fn test_cast_slice_mut_mono8() {
        use crate::pixel::Mono8;
        let mut raw = [10u8, 20, 30];
        let pixels = Mono8::cast_slice_mut(&mut raw).unwrap();
        assert_eq!(pixels.len(), 3);
        assert_eq!(pixels[0], Mono8::new(10));
    }

    #[test]
    fn test_cast_slice_rgb8() {
        use crate::pixel::Rgb8;
        let raw = [10u8, 20, 30, 40, 50, 60];
        let pixels = Rgb8::cast_slice(&raw).unwrap();
        assert_eq!(pixels.len(), 2);
        assert_eq!(pixels[0], Rgb8::new(10, 20, 30));
        assert_eq!(pixels[1], Rgb8::new(40, 50, 60));
    }

    #[test]
    fn test_cast_slice_alignment_failure() {
        use crate::pixel::Rgb16;
        // Rgb16 has ALIGN == 2. Create a byte buffer that is misaligned
        // by embedding it inside a larger buffer and taking an odd-offset subslice.
        let raw = [0u8; 16];
        // Branchless: flip the lowest bit so the pointer is guaranteed misaligned for ALIGN=2
        let ptr = raw.as_ptr() as usize;
        let offset = 1 - (ptr % 2);
        let misaligned = &raw[offset..offset + 6]; // 6 bytes = 1 Rgb16 pixel
        assert!(Rgb16::cast_slice(misaligned).is_none());
    }

    #[test]
    fn test_cast_slice_mut_wrong_length() {
        use crate::pixel::Rgb8;
        // 4 bytes is not a multiple of 3 (Rgb8::SIZE)
        let mut raw = [1u8, 2, 3, 4];
        assert!(Rgb8::cast_slice_mut(&mut raw).is_none());
    }

    #[test]
    fn test_cast_slice_mut_alignment_failure() {
        use crate::pixel::Rgb16;
        // Rgb16 has ALIGN == 2. Create a misaligned mutable slice.
        let mut raw = [0u8; 16];
        let ptr = raw.as_mut_ptr() as usize;
        let offset = 1 - (ptr % 2);
        let misaligned = &mut raw[offset..offset + 6];
        assert!(Rgb16::cast_slice_mut(misaligned).is_none());
    }

    #[test]
    fn test_cast_slice_mut_empty() {
        use crate::pixel::Mono8;
        let mut raw: [u8; 0] = [];
        let pixels = Mono8::cast_slice_mut(&mut raw).unwrap();
        assert_eq!(pixels.len(), 0);
    }

    #[test]
    fn test_cast_slice_mut_modify() {
        use crate::pixel::Mono8;
        let mut raw = [10u8, 20, 30];
        {
            let pixels = Mono8::cast_slice_mut(&mut raw).unwrap();
            pixels[1] = Mono8::new(99);
        }
        assert_eq!(raw[1], 99);
    }

    #[test]
    fn test_from_bytes_wrong_length() {
        use crate::pixel::Mono8;
        use crate::pixel::Rgb8;
        use crate::pixel::Rgba8;

        // Too many bytes
        assert!(Mono8::from_bytes(&[1, 2]).is_none());
        // Too few bytes
        assert!(Rgb8::from_bytes(&[1, 2]).is_none());
        // Too many bytes
        assert!(Rgb8::from_bytes(&[1, 2, 3, 4]).is_none());
        // Empty slice
        assert!(Rgba8::from_bytes(&[]).is_none());
        // Correct length succeeds
        assert!(Mono8::from_bytes(&[42]).is_some());
        assert!(Rgb8::from_bytes(&[1, 2, 3]).is_some());
        assert!(Rgba8::from_bytes(&[1, 2, 3, 4]).is_some());
    }

    #[test]
    fn test_from_bytes_le_wrong_length() {
        use crate::pixel::Mono8;
        use crate::pixel::Rgb16;

        assert!(Mono8::from_bytes_le(&[]).is_none());
        assert!(Mono8::from_bytes_le(&[1, 2]).is_none());
        assert!(Rgb16::from_bytes_le(&[1, 2, 3]).is_none());
        // Correct length succeeds
        assert!(Mono8::from_bytes_le(&[42]).is_some());
        assert!(Rgb16::from_bytes_le(&[1, 2, 3, 4, 5, 6]).is_some());
    }

    #[test]
    fn test_from_bytes_be_wrong_length() {
        use crate::pixel::Mono8;
        use crate::pixel::Rgb16;

        assert!(Mono8::from_bytes_be(&[]).is_none());
        assert!(Mono8::from_bytes_be(&[1, 2]).is_none());
        assert!(Rgb16::from_bytes_be(&[1, 2, 3]).is_none());
        // Correct length succeeds
        assert!(Mono8::from_bytes_be(&[42]).is_some());
        assert!(Rgb16::from_bytes_be(&[1, 2, 3, 4, 5, 6]).is_some());
    }

    #[test]
    fn test_array_as_ref_as_mut() {
        let arr: [u8; 3] = [10, 20, 30];
        let slice: &[u8] = arr.as_ref();
        assert_eq!(slice, &[10, 20, 30]);

        let mut arr2: [u8; 2] = [1, 2];
        let slice_mut: &mut [u8] = arr2.as_mut();
        slice_mut[0] = 99;
        assert_eq!(arr2, [99, 2]);
    }
}

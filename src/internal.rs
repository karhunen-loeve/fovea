use std::borrow::Cow;

use crate::pixel::{MAX_PIXEL_SIZE, PlainChannel, PlainPixel};

// ADR-0046: byte-level helpers that only require the byte-layout
// role are bounded on `PlainChannel`. Endian / channel-walking
// helpers still require `PlainPixel` because they consult
// `CHANNELS`.
pub(crate) unsafe fn as_bytes<P: PlainChannel>(data: &[P]) -> &[u8] {
    let byte_size = std::mem::size_of_val(data);
    let ptr = data.as_ptr() as *const u8;
    unsafe { std::slice::from_raw_parts(ptr, byte_size) }
}

pub(crate) unsafe fn as_mut_bytes<P: PlainPixel>(data: &mut [P]) -> &mut [u8] {
    let byte_size = std::mem::size_of_val(data);
    let ptr = data.as_mut_ptr() as *mut u8;
    unsafe { std::slice::from_raw_parts_mut(ptr, byte_size) }
}

#[cfg(target_endian = "little")]
pub(crate) unsafe fn as_bytes_le<P: PlainPixel>(data: &[P]) -> Cow<'_, [u8]> {
    Cow::Borrowed(unsafe { as_bytes(data) })
}
#[cfg(target_endian = "big")]
pub(crate) unsafe fn as_bytes_be<P: PlainPixel>(data: &[P]) -> Cow<'_, [u8]> {
    Cow::Borrowed(unsafe { as_bytes(data) })
}

#[cfg(target_endian = "little")]
pub(crate) unsafe fn as_bytes_be<P: PlainPixel>(data: &[P]) -> Cow<'_, [u8]> {
    Cow::Owned(as_bytes_changed_endian(data))
}
#[cfg(target_endian = "big")]
pub(crate) unsafe fn as_bytes_le<P: PlainPixel>(data: &[P]) -> Cow<'_, [u8]> {
    Cow::Owned(as_bytes_changed_endian(data))
}

fn as_bytes_changed_endian<P: PlainPixel>(data: &[P]) -> Vec<u8> {
    let bytes = unsafe { as_bytes(data) };
    let pixel_size = <P as PlainChannel>::SIZE;
    //let num_pixels = P::DIM;
    let mut be_bytes = Vec::with_capacity(bytes.len());

    for i in 0..data.len() {
        let mut offset = i * pixel_size;
        for &size in P::CHANNELS {
            let channel_bytes = &bytes[offset..offset + size];
            be_bytes.extend(channel_bytes.iter().rev());
            offset += size;
        }
    }
    be_bytes
}

/// Converts a byte slice to a value of type `T` (native endianness).
///
/// Returns `None` if `bytes.len()` does not equal `size_of::<T>()`.
///
/// # Safety
/// - `T` must be valid for any bit pattern (guaranteed by `PlainChannel`)
pub(crate) fn from_bytes<T: PlainChannel>(bytes: &[u8]) -> Option<T> {
    if bytes.len() != std::mem::size_of::<T>() {
        return None;
    }
    // SAFETY: We verified the length. The caller guarantees T is valid for any bit pattern.
    Some(unsafe { std::ptr::read_unaligned(bytes.as_ptr() as *const T) })
}

#[cfg(target_endian = "little")]
pub(crate) fn from_bytes_le<T: PlainPixel>(bytes: &[u8]) -> Option<T> {
    from_bytes(bytes)
}

#[cfg(target_endian = "big")]
pub(crate) fn from_bytes_be<T: PlainPixel>(bytes: &[u8]) -> Option<T> {
    from_bytes(bytes)
}

#[cfg(target_endian = "little")]
pub(crate) fn from_bytes_be<T: PlainPixel>(bytes: &[u8]) -> Option<T> {
    from_bytes_changed_endian(bytes)
}

#[cfg(target_endian = "big")]
pub(crate) fn from_bytes_le<T: PlainPixel>(bytes: &[u8]) -> Option<T> {
    from_bytes_changed_endian(bytes)
}

/// Converts a byte slice to a value of type `T`, reversing the endianness of each channel.
///
/// Returns `None` if `bytes.len()` does not equal `T::SIZE`.
fn from_bytes_changed_endian<T: PlainPixel>(bytes: &[u8]) -> Option<T> {
    if bytes.len() != <T as PlainChannel>::SIZE {
        return None;
    }
    assert!(
        <T as PlainChannel>::SIZE <= MAX_PIXEL_SIZE,
        "pixel type is larger than the stack buffer ({} > {})",
        <T as PlainChannel>::SIZE,
        MAX_PIXEL_SIZE
    );
    let mut buf = [0u8; MAX_PIXEL_SIZE];
    let swapped = &mut buf[..<T as PlainChannel>::SIZE];
    let mut offset = 0;
    for &size in T::CHANNELS {
        let channel_bytes = &bytes[offset..offset + size];
        swapped[offset..offset + size].copy_from_slice(channel_bytes);
        swapped[offset..offset + size].reverse();
        offset += size;
    }
    from_bytes(swapped)
}

#[inline]
pub(crate) fn in_bounds(size: &crate::Size, x: usize, y: usize) -> bool {
    x < size.width && y < size.height
}

/// Strict pixel-index computation used by Tier-3 accessors
/// (`ImageView::pixel_at`, `ImageViewMut::pixel_at_mut`, `Index<(x, y)>`).
///
/// Returns the flat row-major index `y * size.width + x` after validating
/// the precondition `x < size.width && y < size.height` **and** that the
/// multiplication/addition do not wrap.
///
/// # Panics
///
/// - if `x >= size.width` or `y >= size.height` (out-of-bounds);
/// - if `y * size.width + x` overflows `usize` (which would otherwise wrap
///   silently in release and return the wrong pixel).
///
/// Both panics are Tier-3 (programmer-bug) per the error-handling
/// convention in `AGENTS.md`. The cost over the previous unchecked path
/// is two `checked_*` calls plus an unlikely-branch; under `opt-level=3`
/// it optimises to the same code path as the unchecked version for
/// inputs that satisfy the precondition.
#[inline]
pub(crate) fn checked_index_or_panic(size: &crate::Size, x: usize, y: usize) -> usize {
    assert!(
        x < size.width && y < size.height,
        "pixel index out of bounds: ({x}, {y}) is not inside {}x{}",
        size.width,
        size.height
    );
    y.checked_mul(size.width)
        .and_then(|row_start| row_start.checked_add(x))
        .unwrap_or_else(|| {
            panic!(
                "pixel index arithmetic overflowed usize: ({x}, {y}) on {}x{} image",
                size.width, size.height
            )
        })
}

/// Strict row-range computation used by Tier-3 accessors
/// (`RasterImage::row`, `RasterImageMut::row_mut`).
///
/// Returns the `(start, end)` slice bounds for row `y`. See
/// [`checked_index_or_panic`] for the rationale; the same Tier-3
/// guarantees apply here.
///
/// # Panics
///
/// - if `y >= size.height`;
/// - if `y * size.width` or `y * size.width + size.width` overflows.
#[inline]
pub(crate) fn checked_row_range_or_panic(size: &crate::Size, y: usize) -> (usize, usize) {
    assert!(
        y < size.height,
        "row index out of bounds: {y} is not less than height {}",
        size.height
    );
    let start = y.checked_mul(size.width).unwrap_or_else(|| {
        panic!(
            "row index arithmetic overflowed usize: y={y} on width={}",
            size.width
        )
    });
    let end = start.checked_add(size.width).unwrap_or_else(|| {
        panic!(
            "row index arithmetic overflowed usize: y={y} on width={}",
            size.width
        )
    });
    (start, end)
}

/// Unchecked, internal-only flat index. Inner loops that have already
/// proven `x < width && y < height` and that `y * width + x` cannot wrap
/// (e.g. because the image's `data.len() == width * height` is a `usize`
/// and was already constructed without overflow) may use this directly.
///
/// **Not exposed publicly.** Public Tier-3 accessors must go through
/// [`checked_index_or_panic`].
#[inline]
pub(crate) fn index(size: &crate::Size, x: usize, y: usize) -> usize {
    y * size.width + x
}

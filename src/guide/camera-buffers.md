# Camera buffers

Camera SDKs usually hand you bytes plus metadata: width, height, stride, bit depth, channel order, endian-ness, and sometimes padding. fovea's job is to turn that boundary into a typed image as early as possible.

## 8-bit packed buffers

For alignment-1 pixel types such as `Mono8`, `Srgb8`, `Rgb8`, and `Bgr8`, `Image::from_raw_bytes` can take ownership of the byte allocation without copying.

```rust
use fovea::image::{Image, ImageView};
use fovea::pixel::Mono8;

let raw = vec![0u8, 64, 128, 255];
let img: Image<Mono8> = Image::from_raw_bytes(2, 2, raw)?;

assert_eq!(img.width(), 2);
assert_eq!(img.pixel_at(1, 1), Mono8::new(255));
# Ok::<(), fovea::Error>(())
```

Use the pixel type that matches the buffer. If the SDK says BGR, use `Bgr8`, not `Rgb8`.

## Aligned pixels: copy into owned storage

For pixel types whose alignment is greater than 1, such as `Mono16`, use `Image::from_bytes_copy`. It copies bytes into an allocation with the correct alignment.

```rust
use fovea::image::{Image, ImageView};
use fovea::pixel::Mono16;

let bytes = [0u8, 1, 0, 2];
let img: Image<Mono16> = Image::from_bytes_copy(2, 1, &bytes)?;

assert_eq!(img.size(), fovea::Size::new(2, 1));
# Ok::<(), fovea::Error>(())
```

Be explicit about endian-ness at the camera boundary. `from_bytes_copy` uses the pixel's native byte interpretation; if your SDK delivers a fixed endian format, normalize the bytes before constructing the image or use the endian helpers on `PlainPixel` for individual pixels.

## Borrowing existing storage

If another owner controls the allocation lifetime, borrow it with `ImageRef` or `ImageRefMut` rather than taking ownership.

```rust
use fovea::image::{ImageRef, ImageView};
use fovea::pixel::Mono8;

let sdk_pixels = [Mono8::new(10), Mono8::new(20), Mono8::new(30), Mono8::new(40)];
let view = ImageRef::new(2, 2, &sdk_pixels)?;

assert_eq!(view.pixel_at(0, 1), Mono8::new(30));
# Ok::<(), fovea::Error>(())
```

For a padded or strided SDK buffer, build a borrowed strided view if the storage is already typed, or copy row-by-row into an owned `Image<P>` if the source is only bytes.

## Reinterpreting byte slices

`PlainPixel::cast_slice` is useful when you need a borrowed pixel slice over raw bytes. It checks length and alignment before returning a typed slice.

```rust
use fovea::pixel::{Mono8, PlainPixel};

let raw = [1u8, 2, 3, 4];
let pixels: &[Mono8] = Mono8::cast_slice(&raw).unwrap();

assert_eq!(pixels.len(), 4);
assert_eq!(pixels[2], Mono8::new(3));
```

If `cast_slice` returns `None`, do not force it with `unsafe`. The buffer length, alignment, or pixel type is wrong for zero-copy reinterpretation.

## Common mistakes

- **Treating BGR as RGB.** Use `Bgr8` / `Bgr16` at the boundary, then convert with `ColorSwap` only when you mean to.
- **Ignoring stride.** A region in a padded frame is not always contiguous. Use row access or a strided view.
- **Using `Srgb8` for linear camera data.** sRGB means a transfer function. Most raw camera data is linear mono or linear RGB/BGR.
- **Inventing runtime flags for layout.** Prefer distinct pixel types. The type should say what the bytes mean.

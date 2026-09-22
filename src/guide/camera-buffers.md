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

A padded buffer, whose rows sit further apart than the image is wide, still borrows without a copy. That is [the next section](#padded-rows-a-rowstride-wider-than-the-image).

## Padded rows: a rowStride wider than the image

Android's CameraX delivers `YUV_420_888` planes whose `rowStride` is routinely wider than the frame, because the hardware wants every row aligned; a 1920-wide plane commonly arrives with a pitch of 1984. Read back to back, every row starts late, the error accumulates, and the picture shears.

`Image::from_raw_bytes` will not do that to you. It asks for the image dimensions, the buffer length does not match them, and the call fails at the boundary instead of producing a frame that looks almost right. The route that does work costs no copy: view the buffer at the width it actually has, then crop the padding away.

```rust
use fovea::image::{ImageRef, ImageView, SubView};
use fovea::pixel::{Mono8, PlainPixel};
use fovea::{Rectangle, Size};

// A plane whose rows are 8 bytes apart but only 6 wide.
let (width, height, row_stride) = (6, 4, 8);
# let bytes: Vec<u8> = (0..row_stride * height)
#     .map(|i| if i % row_stride < width { 1 } else { 0xFF })
#     .collect();

let pixels: &[Mono8] = Mono8::cast_slice(&bytes).unwrap();
let padded = ImageRef::new(row_stride, height, pixels)?;
let frame = padded
    .roi(Rectangle::new((0, 0), Size::new(width, height)))
    .unwrap();

assert_eq!(frame.size(), Size::new(width, height));
// No padding leaked in: the last column is image data, not 0xFF.
assert_eq!(frame.pixel_at(width - 1, height - 1), Mono8::new(1));
# Ok::<(), fovea::Error>(())
```

The surprise is that no stride is passed anywhere. `ImageRef` keeps the row pitch of the buffer it was built from, and a sub-view inherits it, so viewing the plane at its own width and cropping to the image is the whole technique.

The sub-view borrows its parent, so both `let` bindings are load bearing. Written as one chained expression the same code fails to compile with `E0716`: the `ImageRef` would be a temporary, dropped at the end of the statement while `frame` still points into it.

`roi` is available here because `Mono8` implements [`OriginInvariantPixel`](crate::pixel::OriginInvariantPixel). A pixel type you define yourself needs `impl OriginInvariantPixel for MyPixel {}` before it can be cropped, and the trait's own documentation says when that marker is true and when a coordinate-dependent pixel should withhold it.

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

This is also how a raw Bayer frame enters the type system. The CFA types are `#[repr(transparent)]` over their sample, so a camera buffer reported as `BayerRG8` needs no copy — only the right type name:

```rust
use fovea::pixel::PlainPixel;
use fovea::pixel::bayer::BayerRggb8;

let raw = [1u8, 2, 3, 4];
let pixels: &[BayerRggb8] = BayerRggb8::cast_slice(&raw).unwrap();

assert_eq!(pixels[2].value(), 3);
```

Match the SDK's format string to the type once, at the boundary, and everything downstream knows the pattern and the depth. Note that SFNC abbreviates the tile to its first row: `BayerRG8` is `BayerRggb8`, `BayerGB12` is `BayerGbrg12`.

From there the pipeline is two named steps — the camera's balance ratios on the mosaic, then interpolation to RGB at the same depth:

```rust
use fovea::image::Image;
use fovea::pixel::{Rgb12, bayer::BayerRggb12};
use fovea::transform::{BayerGains, MalvarHeCutler, demosaic, white_balance};

let raw = Image::fill(64, 48, BayerRggb12::new(2048));

let balanced = white_balance(&raw, BayerGains::new(1.9, 1.0, 1.6).unwrap());
let rgb: Image<Rgb12> = demosaic(&balanced, MalvarHeCutler);
```

The output type follows from the sample type (`BayerRggb12` → `Rgb12`), so a twelve-bit frame cannot silently become eight-bit here. Neither function takes a border policy: only reflection without edge duplication preserves a CFA sample's colour at the frame edge, so that treatment is pinned into the contract.

## Common mistakes

- **Treating BGR as RGB.** Use `Bgr8` / `Bgr16` at the boundary, then convert with `ColorSwap` only when you mean to.
- **Ignoring stride.** A region in a padded frame is not contiguous. See [Padded rows](#padded-rows-a-rowstride-wider-than-the-image) for the zero-copy route.
- **Using `Srgb8` for linear camera data.** sRGB means a transfer function. Most raw camera data is linear mono or linear RGB/BGR.
- **Inventing runtime flags for layout.** Prefer distinct pixel types. The type should say what the bytes mean.
- **Typing a CFA frame as `Mono<N>`.** It loses the mosaic pattern, and with it every guard against blurring, resizing, or odd-origin-cropping across colour channels. Use the `bayer` types.

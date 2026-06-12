# FAQ

These are the questions people usually ask after the first example compiles.

## What is the difference between `Image`, `ImageRef`, `ImageRefMut`, and `ImageArray`?

Use `Image<P>` when you own heap-allocated pixels with runtime dimensions. Use `ImageRef<'a, P>` or `ImageRefMut<'a, P>` when you are borrowing storage owned by something else, including a region of interest. Use `ImageArray<P, W, H>` when the dimensions are compile-time constants.

```rust
use fovea::image::{ContiguousImage, Image, ImageArray, ImageRef, ImageView};

let owned = Image::fill(3, 2, 7u8);
let borrowed = ImageRef::new(3, 2, owned.as_slice())?;
let fixed: ImageArray<u8, 3, 2> = ImageArray::generate(|x, y| (x + y * 3) as u8);

assert_eq!(borrowed.size(), owned.size());
assert_eq!(fixed.pixel_at(2, 1), 5);
# Ok::<(), fovea::Error>(())
```

See also: `ImageView`, `RasterImage`, `ContiguousImage`, and `SubView` in the `image` module.

## What is the difference between `ImageView` and `RasterImage`?

`ImageView` means random pixel access: `pixel_at(x, y)`. `RasterImage` adds row-slice access: `row(y) -> &[P]`. Prefer `RasterImage` for algorithms that scan rows, because row slices are cache-friendly and avoid repeated coordinate indexing.

```rust
use fovea::image::{Image, RasterImage};

let img = Image::generate(4, 2, |x, y| (x + y * 4) as u8);
assert_eq!(img.row(1), &[4, 5, 6, 7]);
```

## Where do I start if I just want to load a PNG and resize it?

Use `fovea-io` to decode the file, match the returned per-codec image enum once, then run typed fovea operations. For bilinear resize on sRGB images, decode to linear pixels first.

```rust,ignore
use fovea::Size;
use fovea::image::Image;
use fovea::pixel::{RgbF32, Srgb8};
use fovea::transform::{Bilinear, SrgbGamma, convert_image, resize};
use fovea_io::png::{self, PngImage, PngEncodeOptions};

let bytes = std::fs::read("input.png")?;
let decoded = png::decode(&bytes)?;

let srgb: Image<Srgb8> = match decoded.image {
    PngImage::Srgb8(img) => img,
    other => return Err(format!("expected 8-bit sRGB PNG, got {other:?}").into()),
};

let linear: Image<RgbF32> = convert_image(&srgb, SrgbGamma);
let resized: Image<RgbF32> = resize(&linear, Size::new(800, 600), Bilinear);
let output: Image<Srgb8> = convert_image(&resized, SrgbGamma);

let encoded = png::encode(&output, &PngEncodeOptions::default())?;
std::fs::write("output.png", encoded)?;
```

The important pattern is `match once, then work with concrete image types`.

## What pixel type should I use for a 10/12/14-bit grayscale camera?

Use `Mono<10>`, `Mono<12>`, or `Mono<14>` when the bit depth itself is meaningful and you want the pixel type to say so. Use `Mono16` when the SDK already presents a 16-bit lane and you want the simplest representation for downstream tools.

```rust
use fovea::image::{Image, ImageView};
use fovea::pixel::{Mono, Mono12};

let img = Image::fill(2, 2, Mono12::new(4095));
assert_eq!(img.pixel_at(0, 0), Mono::<12>::new(4095));
```

`Mono<N>::new` clamps to the representable range for `N`, so a 12-bit pixel cannot accidentally carry a 16-bit value.

## What is the difference between `Srgb8` and `Rgb8`?

`Srgb8` is gamma-encoded display or file data. `Rgb8` is linear-light RGB with 8-bit channels. They have similar storage but different meaning, so they are different types. Algorithms that copy samples may accept both; algorithms that blend samples can require linear pixels.

```compile_fail
use fovea::Size;
use fovea::image::Image;
use fovea::pixel::Srgb8;
use fovea::transform::{Bilinear, resize};

let srgb = Image::fill(4, 4, Srgb8::new(128, 128, 128));
let _ = resize(&srgb, Size::new(8, 8), Bilinear);
```

Use `SrgbGamma` to move between sRGB and linear-light pixels.

## How do I convert between pixel types? What is a strategy?

A strategy names the semantics of the conversion. `SrgbGamma` applies the sRGB transfer function. `Luminance` chooses a grayscale formula. `FullRange` maps the full numeric range. `Narrow` preserves numeric values and clamps when narrowing.

```rust
use fovea::image::{Image, ImageView};
use fovea::pixel::{Mono8, Rgb8};
use fovea::transform::{Broadcast, Luminance, convert_image};

let gray = Image::fill(2, 1, Mono8::new(128));
let rgb: Image<Rgb8> = convert_image(&gray, Broadcast);
assert_eq!(rgb.pixel_at(0, 0), Rgb8::new(128, 128, 128));

let back: Image<Mono8> = convert_image(&rgb, Luminance);
assert_eq!(back.pixel_at(0, 0), Mono8::new(128));
```

## How do I go from raw bytes to a typed image?

For 8-bit pixels, `Image::from_raw_bytes` can take ownership of the byte allocation without copying. For aligned pixels such as `Mono16`, use `Image::from_bytes_copy` to copy into correctly aligned storage.

```rust
use fovea::image::{Image, ImageView};
use fovea::pixel::{Mono8, Mono16};

let img8: Image<Mono8> = Image::from_raw_bytes(2, 1, vec![10, 20])?;
assert_eq!(img8.pixel_at(1, 0), Mono8::new(20));

let bytes16 = [0u8, 1, 0, 2];
let img16: Image<Mono16> = Image::from_bytes_copy(2, 1, &bytes16)?;
assert_eq!(img16.width(), 2);
# Ok::<(), fovea::Error>(())
```

## How do I crop a region of interest without allocating?

Use `SubView::roi`. It returns an `Option` because out-of-bounds ROIs are normal caller-supplied data, not a programmer bug.

```rust
use fovea::{Rectangle, Size};
use fovea::image::{Image, ImageView, SubView};
use fovea::pixel::Mono8;

let img = Image::generate(5, 5, |x, y| Mono8::new((x + y * 5) as u8));
let roi = img.roi(Rectangle::new((1, 1), Size::new(3, 3))).unwrap();

assert_eq!(roi.size(), Size::new(3, 3));
assert_eq!(roi.pixel_at(0, 0), img.pixel_at(1, 1));
```

## How do I parallel-process my very large image?

fovea gives you safe pieces; your application chooses the execution policy. For per-pixel work, use contiguous slices or rows. For region-local work, split into tiles. For in-place tile mutation, use `into_tiles_mut`, which yields disjoint mutable tiles.

See [`large_images`] for the decision table and Rayon-shaped examples.

## How do I define a custom pixel type?

Most custom pixels should derive the fovea traits. Use an explicit representation (`#[repr(C)]` or a transparent wrapper), make layout obvious, and derive only the semantics the type truly supports.

```rust,ignore
use fovea::{HomogeneousPixel, PlainPixel, ZeroablePixel};
use std::num::Saturating;

#[derive(Clone, Copy, PlainPixel, HomogeneousPixel, ZeroablePixel)]
#[repr(C)]
pub struct MyBayer8 {
    pub value: Saturating<u8>,
}
```

Only implement `LinearPixel` / `LinearSpace` if interpolation and blending are meaningful for the type. Likewise, only implement `OriginInvariantPixel` if cropping preserves the pixel's meaning. A Bayer CFA pixel such as `MyBayer8` deliberately omits it: an ROI at an odd origin shifts the 2×2 mosaic phase, so the compiler rejects ordinary `roi`/`tiles`/`sliding_windows` and steers callers to a phase-aware API instead (see ADR-0051).

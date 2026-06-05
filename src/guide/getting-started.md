# Getting started

Start with the shape of the pipeline:

```text
file or camera bytes → typed Image<P> → explicit conversions → transforms → output
```

The important step is the middle one. Once pixels have a type, fovea can reject operations that do not make sense for that pixel type.

## Your first typed image

```rust
use fovea::image::{Image, ImageView};
use fovea::pixel::Srgb8;

let img = Image::generate(3, 2, |x, y| {
    Srgb8::new((x * 80) as u8, (y * 120) as u8, 64)
});

assert_eq!(img.width(), 3);
assert_eq!(img.height(), 2);
assert_eq!(img.pixel_at(2, 1), Srgb8::new(160, 120, 64));
```

`Image<P>` is the owned, heap-allocated image type. The `P` is not decoration: it says what every pixel means.

## Convert before you process

Most files and displays use gamma-encoded sRGB. Most operations that blend, filter, or interpolate should run in linear light. fovea makes that conversion explicit:

```rust
use fovea::image::{Image, ImageView};
use fovea::pixel::{RgbF32, Srgb8};
use fovea::transform::{SrgbGamma, convert_image};

let srgb = Image::fill(4, 4, Srgb8::new(128, 64, 32));
let linear: Image<RgbF32> = convert_image(&srgb, SrgbGamma);

assert_eq!(linear.size(), srgb.size());

let encoded: Image<Srgb8> = convert_image(&linear, SrgbGamma);
assert_eq!(encoded.size(), srgb.size());
```

The conversion strategy is part of the call. There is no hidden default gamma curve, luminance formula, or narrowing rule.

## Resize correctly

```rust
use fovea::Size;
use fovea::image::{Image, ImageView};
use fovea::pixel::{RgbF32, Srgb8};
use fovea::transform::{Bilinear, NearestNeighbor, SrgbGamma, convert_image, resize};

let srgb = Image::fill(8, 6, Srgb8::new(128, 64, 32));

// Nearest-neighbor copies samples, so it works directly on sRGB.
let quick: Image<Srgb8> = resize(&srgb, Size::new(4, 3), NearestNeighbor);
assert_eq!(quick.size(), Size::new(4, 3));

// Bilinear interpolation blends samples, so use linear pixels.
let linear: Image<RgbF32> = convert_image(&srgb, SrgbGamma);
let smooth: Image<RgbF32> = resize(&linear, Size::new(4, 3), Bilinear);
assert_eq!(smooth.size(), Size::new(4, 3));
```

This is the main mental model: if an algorithm needs a property, that property appears in the type bounds.

## Crop without allocating

ROIs are borrowed views into existing image memory. Use `roi` when you want to look at a region without copying pixels.

```rust
use fovea::{Coordinate, Rectangle, Size};
use fovea::image::{Image, ImageView, SubView};
use fovea::pixel::Mono8;

let img = Image::generate(6, 4, |x, y| Mono8::new((x + y * 6) as u8));
let rect = Rectangle::new(Coordinate::new(2, 1), Size::new(3, 2));
let roi = img.roi(rect).unwrap();

assert_eq!(roi.size(), Size::new(3, 2));
assert_eq!(roi.pixel_at(0, 0), img.pixel_at(2, 1));
```

## Work from camera bytes

For byte-addressable camera output, choose a pixel type that states the layout. `from_raw_bytes` is zero-copy for alignment-1 pixels such as `Mono8`; `from_bytes_copy` works for aligned pixels such as `Mono16`.

```rust
use fovea::image::{Image, ImageView};
use fovea::pixel::Mono8;

let raw = vec![0u8, 64, 128, 255];
let img: Image<Mono8> = Image::from_raw_bytes(2, 2, raw)?;

assert_eq!(img.pixel_at(1, 1), Mono8::new(255));
# Ok::<(), fovea::Error>(())
```

## Where next?

- Pixel choice: `guide::pixel_types`
- Camera SDK buffers: `guide::camera_buffers`
- Large-image processing: `guide::large_images`
- Common questions: `guide::faq`

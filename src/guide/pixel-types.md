# Pixel types

Pixel types are the vocabulary of a fovea pipeline. Choose the type that says what the bytes mean, not just how many bytes they occupy.

## Quick choice table

| You have | Start with | Why |
|---|---|---|
| 8-bit display/file RGB | `Srgb8` | This is gamma-encoded sRGB, not linear light. |
| Linear RGB samples | `Rgb8`, `Rgb16`, `RgbF32` | These can participate in linear-light operations. |
| 8-bit grayscale display/file data | `SrgbMono8` | Gamma-encoded grayscale. Decode before interpolation. |
| Scientific grayscale intensity | `Mono8`, `Mono16`, `MonoF32` | Linear mono pixels. |
| 10/12/14-bit camera samples | `Mono<10>`, `Mono<12>`, `Mono<14>` | The bit depth is part of the type. |
| BGR camera or SDK buffer | `Bgr8`, `Bgr16`, `BgrF32` | Channel order is explicit. |
| Segmentation mask | `bool` / `BinaryImage` | Native binary-image representation. |
| Connected-component labels | `Label32` | Label pixels are not grayscale pixels. |

## `Srgb8` vs `Rgb8`

This is the most important distinction in the crate.

- `Srgb8` means gamma-encoded sRGB samples, usually from a file or display path.
- `Rgb8` means linear-light RGB samples stored in 8-bit channels.

They are not interchangeable, even if both have three `u8` channels. A resize that copies pixels can accept `Srgb8`; a resize that blends pixels requires linear-space pixels.

```rust
use fovea::image::{Image, ImageView};
use fovea::pixel::{RgbF32, Srgb8};
use fovea::transform::{SrgbGamma, convert_image};

let srgb = Image::fill(2, 2, Srgb8::new(128, 64, 32));
let linear: Image<RgbF32> = convert_image(&srgb, SrgbGamma);

assert_eq!(linear.size(), srgb.size());
```

## Mono camera bit depths

`Mono<N>` stores a monochrome value with an explicit bit depth. The constructor clamps to the range for `N`.

```rust
use fovea::pixel::Mono;

let px = Mono::<12>::new(5000);
assert_eq!(px.value(), 4095);
```

Use `Mono16` instead when your camera SDK already expands samples to a full 16-bit lane and the exact sensor bit depth is documented elsewhere in your pipeline.

## Floating-point pixels

Raw `f32` and `f64` are channels, not pixels. Use `MonoF32`, `RgbF32`, `RgbaF32`, and their `f64` counterparts when an image stores pixel data.

```rust
use fovea::image::{Image, ImageView};
use fovea::pixel::MonoF32;

let img = Image::fill(2, 2, MonoF32::new(0.5));
assert_eq!(img.pixel_at(0, 0).value(), 0.5);
```

This is not ceremony. The wrapper states the role of the scalar: it is now a pixel intensity, not an arbitrary number.

## Conversion strategies

When a conversion can lose information or change meaning, name the strategy.

| Strategy | Meaning |
|---|---|
| `SrgbGamma` | Decode or encode the sRGB transfer function. |
| `Luminance` | Convert color to grayscale using BT.601 luminance. |
| `Broadcast` | Copy mono into every color channel. |
| `ColorSwap` | Swap RGB and BGR channel order. |
| `FullRange` | Map the full source numeric range to the full destination range. |
| `Narrow` | Preserve numeric values and clamp when narrowing. |

```rust
use fovea::image::{Image, ImageView};
use fovea::pixel::{Mono8, Rgb8};
use fovea::transform::{Broadcast, convert_image};

let mono = Image::fill(1, 1, Mono8::new(200));
let rgb: Image<Rgb8> = convert_image(&mono, Broadcast);
assert_eq!(rgb.pixel_at(0, 0), Rgb8::new(200, 200, 200));
```

For the full implementation matrix, see the repository's pixel conversion matrix.

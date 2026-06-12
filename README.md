# fovea

[![Crates.io](https://img.shields.io/crates/v/fovea.svg)](https://crates.io/crates/fovea)
[![Documentation](https://docs.rs/fovea/badge.svg)](https://docs.rs/fovea)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/karhunen-loeve/fovea/blob/main/LICENSE)

`fovea` is image processing for Rust where the compiler catches colorspace mistakes, channel-order bugs, and lossy conversions before your code runs.

If you have ever shipped a bug because someone passed BGR where RGB was expected, resized gamma-encoded pixels with bilinear interpolation, or treated a camera SDK byte buffer as "probably u16 grayscale", `fovea` was built for you.

In fovea, BGR and RGB are different types. Blending gamma-encoded pixels is a compile error. Byte layout is a contract, not a comment.

## What fovea prevents

- **BGR/RGB confusion** — `Bgr8` and `Rgb8` are distinct types. Passing one where the other is expected is a compile error.
- **Gamma-incorrect interpolation** — `Bilinear` resize requires pixels in `LinearSpace`. It will not compile for `Srgb8`; linearize first, explicitly.
- **Silent data loss** — there is no implicit "just convert this image" path. Lossy conversions are named strategies: `Luminance`, `Narrow`, `FullRange`, `SrgbGamma`, `Clamp`.
- **Raw byte misuse** — `PlainPixel` is an unsafe layout contract. Once a pixel type implements it, byte-level access is guaranteed by the type system.
- **Over-promising APIs** — traits describe exactly what an operation needs: random access, row access, contiguous storage, byte layout, or linear arithmetic.

## The compiler catches the image bug

Nearest-neighbor resize copies pixels, so it works for gamma-encoded sRGB. Bilinear resize blends neighboring samples, so fovea requires linear-light pixels first.

```rust
use fovea::Size;
use fovea::image::{Image, ImageView};
use fovea::pixel::{RgbF32, Srgb8};
use fovea::transform::{Bilinear, NearestNeighbor, SrgbGamma, convert_image, resize};

let srgb = Image::generate(4, 3, |x, y| {
    Srgb8::new((x * 40) as u8, (y * 60) as u8, 128)
});

// ✓ This compiles: nearest-neighbor copies samples without blending them.
let preview: Image<Srgb8> = resize(&srgb, Size::new(8, 6), NearestNeighbor);
assert_eq!(preview.size(), Size::new(8, 6));

// ✓ This compiles: explicit sRGB decode before interpolation.
let linear: Image<RgbF32> = convert_image(&srgb, SrgbGamma);
let resized: Image<RgbF32> = resize(&linear, Size::new(8, 6), Bilinear);
assert_eq!(resized.size(), Size::new(8, 6));
```

The version below does **not** compile, and that is the point:

```compile_fail
use fovea::Size;
use fovea::image::Image;
use fovea::pixel::Srgb8;
use fovea::transform::{Bilinear, resize};

let srgb = Image::fill(4, 3, Srgb8::new(128, 64, 32));

// ✗ Bilinear interpolation blends samples. Srgb8 is gamma-encoded.
let _: Image<Srgb8> = resize(&srgb, Size::new(8, 6), Bilinear);
```

Linearize first with `SrgbGamma`, resize in `RgbF32` or `MonoF32`, then encode back if you need an sRGB output image.

## When NOT to use fovea

- You need real-time video decode: use FFmpeg bindings.
- You need deep-learning tensor pipelines: use `candle`, `ort`, or your tensor runtime of choice.
- You want the broadest possible codec support with the simplest API: use the `image` crate.
- You only need to resize a JPEG in a web app and do not care about pixel semantics: use the `image` crate.

## When fovea is the right tool

- You read pixel data directly from industrial, scientific, or machine-vision cameras.
- Correctness matters more than convenience: inspection, metrology, robotics, medical, lab automation.
- You want the compiler to enforce colorspace and pixel-format discipline across a team.
- You need guaranteed byte layout for camera SDK buffers, memory-mapped images, GPU upload, or FFI.
- You want algorithms to state their real requirements in trait bounds instead of runtime checks.

## Install

```sh
cargo add fovea
```

For PNG, JPEG, or BMP I/O, add `fovea-io` with the codec features you need:

```sh
cargo add fovea
cargo add fovea-io --features png
```

## Getting started

Create an image, decode sRGB samples into linear light, modify pixels through the contiguous slice, and encode back to sRGB:

```rust
use fovea::image::{ContiguousImageMut, Image, ImageView};
use fovea::pixel::{RgbF32, Srgb8};
use fovea::transform::{SrgbGamma, convert_image};

let srgb = Image::generate(2, 2, |x, y| {
    Srgb8::new((x * 120) as u8, (y * 120) as u8, 64)
});

let mut linear: Image<RgbF32> = convert_image(&srgb, SrgbGamma);
for px in linear.as_mut_slice() {
    px.r = (px.r * 1.2).min(1.0);
}

let display: Image<Srgb8> = convert_image(&linear, SrgbGamma);
assert_eq!(display.size(), srgb.size());
```

For a longer first pass, start with the docs.rs guide: [`fovea::guide`](crate::guide).

## Core types

| Type or trait | Use it when |
|---|---|
| `Image<P>` | You own a heap-allocated image with runtime dimensions. |
| `ImageArray<P, W, H>` | Width and height are compile-time constants. |
| `ImageRef<'a, P>` / `ImageRefMut<'a, P>` | You want a borrowed view over existing storage, including strided ROIs. |
| `ImageView` / `ImageViewMut` | An algorithm only needs random pixel access. |
| `RasterImage` / `RasterImageMut` | An algorithm should process row slices efficiently. |
| `ContiguousImage` / `ContiguousImageMut` | The whole image is one dense pixel slice. |
| `PlainImage` / `PlainImageMut` | You need byte access to contiguous `PlainPixel` storage. |
| `SubView` / `SubViewMut` | You need zero-copy regions of interest, tiles, or sliding windows. |

The image traits intentionally mirror Rust's slice model: borrow views when you can, allocate only when you mean to.

## Pixel types

| Family | Examples | Meaning |
|---|---|---|
| Mono | `Mono8`, `Mono16`, `MonoF32`, `Mono<12>` | One-channel intensity pixels. |
| RGB/BGR | `Rgb8`, `Bgr8`, `RgbF32` | Linear color pixels with explicit channel order. |
| sRGB | `Srgb8`, `Srgba8`, `SrgbMono8` | Gamma-encoded display/file pixels. Not linear-light. |
| Alpha | `Rgba8`, `Bgra8`, `MonoA16` | Pixels with explicit alpha channels. |
| Indexed/labels | `Indexed8`, `Label32`, `bool` | Palette indices, connected-component labels, binary masks. |

Important distinction: `Rgb8` and `Srgb8` may both store three `u8` channels, but they do not mean the same thing. `Rgb8` is linear-light RGB. `Srgb8` is gamma-encoded sRGB. Algorithms that blend pixels can require the former and reject the latter.

## Modules

| Module | Start here | One job |
|---|---|---|
| `image` | `Image`, `ImageView`, `SubView` | Storage, views, rows, ROIs, tiles, and neighborhoods. |
| `pixel` | `Srgb8`, `RgbF32`, `PlainPixel`, `LinearSpace` | Pixel vocabulary and the traits that make illegal operations unrepresentable. |
| `transform` | `convert_image`, `resize`, `combine_images` | Image-producing operations: conversion, resize, geometry, convolution, morphology. |
| `analyze` | `histogram`, `integral_image`, `connected_components` | Image analysis that produces data about an image. |
| `border` | `Clamp`, `Mirror`, `Skip` | Boundary behavior for neighborhood operations. |
| `guide` | `guide::faq`, `guide::pixel_types` | Task-oriented docs.rs pages for common questions. |

## FAQ

**Where do I start if I just want to load a PNG and resize it?**
Use `fovea-io` to decode, match the returned pixel enum once, convert sRGB images to linear pixels with `SrgbGamma`, call `resize(..., Bilinear)`, then encode.

**Why does `Bilinear` fail for `Srgb8`?**
Because bilinear interpolation blends samples, and blending gamma-encoded samples is physically wrong. Use `NearestNeighbor` if you are only copying samples; otherwise linearize first.

**What type should I use for a 12-bit monochrome camera?**
Use `Mono<12>` when you want the bit depth represented in the pixel type. Use `Mono16` when the camera SDK already expands samples to full 16-bit storage and you want simpler integration.

**How do I process a large image in parallel?**
For contiguous per-pixel work, use `as_slice()` / `as_mut_slice()` and choose your own parallel runtime. For region-local work, split into tiles. For in-place mutation, `into_tiles_mut()` yields disjoint mutable tiles.

More answers are in [`fovea::guide::faq`](https://docs.rs/fovea/latest/fovea/guide/faq/index.html).

## Crate ecosystem

| Crate | Published? | Purpose |
|---|---:|---|
| `fovea` | crates.io + docs.rs | Core image types, pixels, analysis, and transforms. |
| `fovea-io` | crates.io + docs.rs | Feature-gated PNG, JPEG, and BMP codecs. |
| `fovea-display` | crates.io + docs.rs | Display conversion strategies, texture metadata, and debug windows. |
| `fovea-derive` | crates.io + docs.rs | Derive macros re-exported by `fovea`. |
| `fovea-examples` | repo only | End-to-end programs that combine the crates. |

## Design principles

fovea is designed around a small set of explicit principles: types are the spec, concerns are orthogonal, traits layer progressively, conversions are named, and layout is a contract. The short version is this:

> The compiler is the first reviewer.

## License

Licensed under the [MIT License](https://github.com/karhunen-loeve/fovea/blob/main/LICENSE).

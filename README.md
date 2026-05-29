# fovea

[![Crates.io](https://img.shields.io/crates/v/fovea.svg)](https://crates.io/crates/fovea)
[![Documentation](https://docs.rs/fovea/badge.svg)](https://docs.rs/fovea)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/karhunen-loeve/fovea/blob/main/LICENSE)

`fovea` is a high-precision, type-safe computer vision library for Rust, built around explicit pixel layout, colour-space correctness, and compile-time guarantees.

```toml
[dependencies]
fovea = "0.1.1"
```

## Quick start

Create an image, convert sRGB samples into linear-light values, and inspect the typed result:

```rust
use fovea::image::{Image, ImageView};
use fovea::pixel::{MonoF32, SrgbMono8};
use fovea::transform::{convert_image, SrgbGamma};

let srgb = Image::generate(4, 3, |x, y| {
    SrgbMono8::new(((x + y) * 32) as u8)
});

let linear: Image<MonoF32> = convert_image(&srgb, SrgbGamma);

assert_eq!(linear.width(), 4);
assert_eq!(linear.height(), 3);
assert!(linear.pixel_at(3, 2).value() > 0.0);
```

For a more tutorial-style first program, see `examples/hello_image.rs` in the crate repository.

## Why fovea?

- **Compile-time pixel correctness.** Pixel types encode semantic meaning: `Srgb8` and `Rgb8` have similar storage but different colour-space guarantees, so algorithms can reject invalid inputs at compile time.
- **Explicit conversions.** Lossy or semantic conversions are named through strategy types such as `SrgbGamma`, `Luminance`, and `Clamp`; data loss is never hidden in a default conversion.
- **Layout-aware image access.** `PlainPixel`, `ImageView`, `RasterImage`, and `PlainImage` make byte layout, random access, row access, and contiguous storage separate capabilities with separate trait bounds.

## Design principles

These numbered principles are the public digest for `§N` references in the fovea crate ecosystem:

1. **Types are the spec**
2. **Concerns are orthogonal**
3. **Traits layer progressively**
4. **Conversions are named**
5. **Layout is a contract**
6. **Images are slices**
7. **I/O is a boundary, not a pipeline**
8. **Surface information, don't decide**
9. **Channels and pixels are different roles**
10. **Extension by addition**
11. **Unsafe is local and proven**
12. **Errors are layered**

## Crate ecosystem

| Crate | Purpose | Links |
|---|---|---|
| `fovea` | Core image types, pixels, analysis, and transforms | [docs.rs](https://docs.rs/fovea) |
| `fovea-io` | PNG, JPEG, and BMP codecs | [GitHub](https://github.com/karhunen-loeve/fovea-io) · [docs.rs](https://docs.rs/fovea-io) |
| `fovea-display` | Display strategies and debug windows | [GitHub](https://github.com/karhunen-loeve/fovea-display) · [docs.rs](https://docs.rs/fovea-display) |
| `fovea-derive` | Derive macros re-exported by `fovea` | [GitHub](https://github.com/karhunen-loeve/fovea-derive) · [docs.rs](https://docs.rs/fovea-derive) |
| `fovea-examples` | End-to-end example programs | [GitHub](https://github.com/karhunen-loeve/fovea-examples) |

## License

Licensed under the [MIT License](https://github.com/karhunen-loeve/fovea/blob/main/LICENSE).

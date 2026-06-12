# Pixel conversions

All pixel-type changes in fovea go through explicit, named strategies.
There is no implicit coercion or default path — every conversion is a
deliberate choice that the compiler can check and the reader can see.

## Quick strategy reference

| Strategy | What it does | Common use |
|---|---|---|
| `FullRange` | Maps the full dynamic range of source to destination | `Rgb8 → Rgb16`, `Mono16 → Mono8` |
| `Narrow` | Preserves numeric values; clamps on narrowing | `Rgb16 → Rgb8` preserving raw counts |
| `SrgbGamma` | Applies / removes the IEC 61966-2-1 sRGB gamma curve | `Srgb8 ↔ RgbF32` (linearise / re-encode) |
| `Luminance` | Color → grayscale via BT.601 weighted sum | `Rgb8 → Mono8` |
| `Broadcast` | Grayscale → color (copy value to every channel) | `Mono8 → Rgb8` |
| `ColorSwap` | Swaps R ↔ B channels, leaves G (and alpha) unchanged | `Rgb8 ↔ Bgr8`, `Srgb8 ↔ SrgbBgr8` (OpenCV) |
| `AddAlpha` | Adds a fully-opaque alpha channel (max value) | `Rgb8 → Rgba8` |
| `PixelMap` | Arbitrary `Fn(&Src) -> Dst` closure | Any custom per-pixel logic |
| `Depalettize` | `Indexed8` → `P` via a 256-entry palette | PNG indexed images |

All built-in strategies except `PixelMap` are zero-sized types (ZSTs) — they
compile down to zero overhead.

> **`FullRange` and float pixel types:** `FullRange` is implemented for
> `MonoF32 ↔ integer` (and `RgbF32 ↔ integer`, etc.) conversions, but it
> assumes `[0.0, 1.0]` as the float "full range". `MonoF32` has no defined
> maximum value, so this is an implicit convention — HDR values above `1.0`
> silently clamp to the integer maximum. Use `PixelMap` with an explicit
> normalization factor whenever your float data may be outside `[0.0, 1.0]`.

## Common paths

```rust
use fovea::image::{Image, ImageView};
use fovea::pixel::{Mono8, Rgb8, RgbF32, Srgb8};
use fovea::transform::{Broadcast, FullRange, Luminance, SrgbGamma, convert_image};

let srgb = Image::fill(4, 4, Srgb8::new(128, 64, 32));

// Decode sRGB gamma → linear float (required before blending or filtering).
let linear: Image<RgbF32> = convert_image(&srgb, SrgbGamma);

// Re-encode linear float → sRGB (for display or file output).
let back: Image<Srgb8> = convert_image(&linear, SrgbGamma);
assert_eq!(back.size(), srgb.size());

// Color → grayscale.
let rgb_img = Image::fill(4, 4, Rgb8::new(100, 150, 200));
let gray: Image<Mono8> = convert_image(&rgb_img, Luminance);
assert_eq!(gray.size(), rgb_img.size());

// Grayscale → color (same value broadcast to every channel).
let mono_img = Image::fill(4, 4, Mono8::new(128));
let rgb_out: Image<Rgb8> = convert_image(&mono_img, Broadcast);
assert_eq!(rgb_out.size(), mono_img.size());

// Widen bit depth.
let wide: Image<fovea::pixel::Rgb16> = convert_image(&rgb_img, FullRange);
assert_eq!(wide.size(), rgb_img.size());
```

## Chaining with `.then()`

Two strategies can be fused into a single pass with no intermediate image
using the `.then::<Mid, _>(next)` combinator:

```rust
use fovea::image::{Image, ImageView};
use fovea::pixel::{Bgr8, Bgr16, Rgb8};
use fovea::transform::{ColorSwap, ConvertPixelExt, FullRange, convert_image};

let img = Image::fill(4, 4, Rgb8::new(200, 100, 50));

// Rgb8 → Bgr8 → Bgr16 in one pass.
let out: Image<Bgr16> = convert_image(&img, ColorSwap.then::<Bgr8, _>(FullRange));
assert_eq!(out.size(), img.size());
```

The turbofish `::<Bgr8, _>` names the intermediate type. Rust cannot infer it
because `FullRange` implements conversions for many type pairs — naming it makes
the pipeline self-documenting and consistent with the crate's "explicit data
layout" principle.

## The sRGB / linear boundary

| Pixel family | Encoding | `LinearSpace`? |
|---|---|---|
| `Srgb8`, `SrgbMono8`, `SrgbBgr8`, … | Gamma-encoded (display / file) | No |
| `Rgb8`, `RgbF32`, `Mono8`, `BgrF32`, … | Linear-light | Yes |

Operations that blend samples — `Bilinear` resize, Gaussian blur, `LinearCombine`
— require `LinearSpace` pixels. Passing gamma-encoded pixels is a compile
error. The fix is always the same: decode with `SrgbGamma` first, run the
algorithm in linear light, then re-encode if you need gamma-encoded output.

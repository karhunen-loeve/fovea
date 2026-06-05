# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `OriginInvariantPixel`: a safe marker trait for pixel types whose
  semantic meaning is invariant under translation of the image origin. It
  is implemented for every shipped pixel family (`Mono*`, `MonoA*`,
  `Rgb*` / `Bgr*`, `Srgb*`, `Indexed8`, `Label32`) and for `bool` (the
  pixel type of `BinaryImage`). See [ADR-0051](https://github.com/karhunen-loeve/fovea/blob/main/docs/adr/0051-origin-invariant-pixel.md).

### Changed

- **Breaking:** `SubView` / `SubViewMut` — and therefore `roi`, `roi_mut`,
  `tiles`, and `sliding_windows` — are now gated on
  `T: OriginInvariantPixel` instead of `T: Copy`. Ordinary same-pixel-type
  ROI and tiling is available only for pixel types whose meaning survives
  an origin shift, so coordinate-dependent pixels (e.g. future Bayer CFA
  types) can no longer silently produce a phase-shifted view. `ImageView`,
  `ImageViewMut`, `RasterImage`, `ContiguousImage`, and the (ungated)
  `IntoTilesMut` are unchanged and remain available for any `T: Copy`.
  Code that used these APIs on raw channel images such as `Image<u8>`
  should switch to a real pixel type such as `Mono8` (or `bool` for binary
  images). See [ADR-0051](https://github.com/karhunen-loeve/fovea/blob/main/docs/adr/0051-origin-invariant-pixel.md).
- **Breaking:** Renamed `SubView::into_tiles` to `SubView::tiles`. The
  method borrows `&self` and returns a borrowing iterator, so the
  `into_*` prefix (which by convention signals a consuming `self`-by-value
  conversion) was misleading and inconsistent with the sibling
  `SubView::sliding_windows`. There is no deprecated alias; update call
  sites from `img.into_tiles(size)` to `img.tiles(size)`.

## [0.1.1] — 2026-05-29

First real public release. `0.1.0` was a name-reservation placeholder
published from an empty source tree; this is the first version with
actual functionality.

### Added

- Initial public release of the `fovea` computer-vision library.
- Core image types: `Image`, `ImageRef`, `ImageRefMut`, `ImageArray`,
  `ContiguousImage`, `PlainImage`, `PlainImageMut`, `SubView`,
  `SubViewMut`, `Neighborhood`, `Kernel`, `ImagePlanes`.
- Trait-based access via `ImageView` and `ImageViewMut`.
- Pixel types with explicit colour-space and channel semantics
  (`Srgb8`, `Srgba8`, `Rgb8`, `Mono<BITS>`, `MonoF32`, `RgbF32`, …).
- Derive macros (`PlainPixel`, `HomogeneousPixel`, `LinearPixel`,
  `ZeroablePixel`) re-exported from `fovea-derive`.
- `transform` module:
  - Unary pixel transforms (`convert_image` with strategies like
    `Luminance`, `SrgbGamma`, `Narrow`, `Invert`, `Clamp`, `Lut`).
  - Binary pixel transforms (`combine_images` with strategies like
    `PixelAdd`, `AbsDiff`, `Blend`).
  - Neighbourhood transforms (`fold_neighborhood`,
    `map_neighborhood`) for convolution, separable filters,
    morphology, and median filtering.
  - Geometric transforms (resize, flip, rotate).
- `analyze` module: histograms, integral images / summed-area tables,
  connected components, statistics.
- `border` module: explicit border policies for neighbourhood
  operations.
- Three-tier error convention: `Option` for absence,
  `Result<T, Error>` for caller-data failures, `panic!` for
  programmer bugs.

[0.1.1]: https://github.com/karhunen-loeve/fovea/releases/tag/v0.1.1

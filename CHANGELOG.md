# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

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

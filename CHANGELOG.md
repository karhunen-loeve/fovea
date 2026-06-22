# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `analyze::threshold::hysteresis_threshold` (and `_into`): double-threshold
  segmentation that keeps a **weak** pixel (`value >= low`) only when its
  8-connected component contains a **strong** pixel (`value >= high`),
  returning a `BinaryImage`. Both comparisons are inclusive (matching the
  Canny literature / OpenCV, and intentionally differing from Otsu's
  exclusive `>`). Accepts any single-channel pixel, including the `MonoF32`
  gradient-magnitude image of a Canny pipeline; built by composition over
  `connected_components` rather than new machinery. The strong mask is never
  materialized. Panics (Tier 3) on a multi-channel pixel, an `out`/input size
  mismatch, or `!(low <= high)`.
- Parameterized Gaussian blur: `gaussian_blur(image, sigma, border)` and
  `gaussian_blur_with(image, sigma, truncate, border)` (plus `_into`
  variants writing to a caller-owned output) derive a normalized separable
  kernel from `sigma`, matching the SciPy / scikit-image convention
  (radius `= round(truncate · sigma)`, default `truncate = 4.0`). The kernel
  is generated allocation-free into a bounded stack buffer; a `sigma` whose
  radius exceeds `MAX_RADIUS` (64, i.e. `sigma > MAX_RADIUS / truncate`)
  panics, as does `sigma <= 0`. The kernel generator is exposed at the image
  layer as `gaussian_kernel_1d`, `gaussian_kernel_size`, `GaussianKernel1D`,
  and the `MAX_RADIUS` bound. The fixed `gaussian_blur_3x3` /
  `gaussian_blur_5x5` paths remain as fast const-sized convenience
  functions.
- `FullRange` conversions completing the const-generic `Mono<N>`
  (`Mono10` / `Mono12` / `Mono14`) coverage: `Mono8/16/32/64 → Mono<N>`
  (e.g. padding an 8-bit reference up to 12-bit) and `Mono<N1> → Mono<N2>`
  (e.g. `Mono10 → Mono12` for mixed-camera pipelines, including the
  equal-depth identity case). The previously shipped
  `Mono<N> → Mono8/16/32/64` direction is unchanged. Rounding is symmetric,
  so a widen-then-narrow round-trip is lossless when the wider depth is a
  superset of the narrower one.

### Changed

- **Breaking:** `gaussian_blur_3x3` and `gaussian_blur_5x5` are now
  **normalized** (kernel sums to 1) and therefore **preserve brightness**,
  matching `box_blur_3x3` / `box_blur_5x5` and every mainstream library.
  Previously they convolved with the raw integer kernels `[1, 2, 1]`
  (sum 16) and `[1, 4, 6, 4, 1]` (sum 256), scaling output brightness by
  ×16 / ×256 and silently saturating into integer output types. Callers
  that relied on the old scaling should divide by 16 / 256, or convolve
  directly with `Neighborhood::gaussian_3x3` / `gaussian_5x5` (the raw
  integer kernels are unchanged and remain available at that layer, where
  the caller owns the scale). The `SeparableKernel::gaussian_3` /
  `gaussian_5` factories are likewise normalized now (weights
  `[0.25, 0.5, 0.25]` and `[0.0625, 0.25, 0.375, 0.25, 0.0625]`).
- Internal (no public behaviour change): the separable convolution path no
  longer materializes its 1-D kernel weights onto the heap. Weights are now
  borrowed as `ImageRef` views and fed to a new no-flip correlation core, with
  true convolution flipping the kernel on the stack via
  `SeparableKernel::flipped`. This removes ~6–8 small per-call allocations from
  `gaussian_blur*`, `box_blur_*`, and `convolve_separable*`, closing the
  kernel-allocation half of the ADR-0023 deviation. The image-sized working-set
  buffers (intermediate + accumulator + output) are unchanged; reusing those
  across calls is the deferred follow-up (see ADR-0054 / OPT-004). The
  `SeparableKernel::to_h_image` / `to_v_image` helpers were removed.

## [0.2.0] — 2026-06-12

### Added

- `OriginInvariantPixel`: a safe marker trait for pixel types whose
  semantic meaning is invariant under translation of the image origin. It
  is implemented for every shipped pixel family (`Mono*`, `MonoA*`,
  `Rgb*` / `Bgr*`, `Srgb*`, `Indexed8`, `Label32`) and for `bool` (the
  pixel type of `BinaryImage`).
- `SrgbBgr` pixel family: `SrgbBgr8`, `SrgbBgr16`, `SrgbBgra8`,
  `SrgbBgra16` — gamma-encoded sRGB values in BGR channel order, for
  zero-copy interop with OpenCV `Mat` buffers and similar BGR-native
  camera SDKs. These types implement `PlainPixel`, `OriginInvariantPixel`,
  and all relevant conversion paths but deliberately omit `LinearSpace` so
  that bilinear interpolation on gamma-encoded data remains a compile error.
- `transform::resize` module documentation: method selection table
  (`NearestNeighbor` vs `Bilinear`), working doctest showing the correct
  linearize-then-resize pipeline, and a guide to implementing custom resize
  strategies.

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
  images).
- **Breaking:** Renamed `SubView::into_tiles` to `SubView::tiles`. The
  method borrows `&self` and returns a borrowing iterator, so the
  `into_*` prefix (which by convention signals a consuming `self`-by-value
  conversion) was misleading and inconsistent with the sibling
  `SubView::sliding_windows`. There is no deprecated alias; update call
  sites from `img.into_tiles(size)` to `img.tiles(size)`.
- `#![warn(missing_docs)]` promoted to `#![deny(missing_docs)]`. All
  public API items now have documentation; the deny lint enforces that
  new public items ship with docs.
- `transform` module overview rewritten with task-oriented section headings
  ("Geometry and flips", "Pixel conversion", "Image arithmetic",
  "Neighbourhood transforms") replacing the opaque "Level 0–3" numbering.
  The quick-start lookup table and submodule links are unchanged.
- README install instructions changed from a pinned `[dependencies]` TOML
  snippet to `cargo add fovea` so the snippet stays accurate across
  releases without manual maintenance.

### Fixed

- Renamed `analyze/histogram/histogram.rs` → `analyze/histogram/engine.rs`
  to eliminate the `clippy::module_inception` warning (module named the
  same as its parent). Follows the existing `analyze/integral/engine.rs`
  convention. The public API is unchanged — `Histogram` is still re-exported
  at `fovea::analyze::histogram::Histogram`.
- Eliminated all remaining `cargo clippy` warnings:
  - Five `needless_range_loop` instances in `image/sequential.rs` tests
    refactored to `for (x, &pixel) in row.iter().enumerate()`.
  - One `needless_range_loop` in `benches/geometry.rs` refactored to
    `for (x_out, pixel) in dst.iter_mut().enumerate()`.
  - `drop_non_drop` at `sequential.rs` — suppressed with a scoped
    `#[allow]`; the explicit `drop(view)` is intentional (it ends the
    borrow, making `data` readable on the next line).
  - Four `neg_cmp_op_on_partial_ord` instances in `pixel/tests.rs` —
    suppressed with a scoped `#[allow]`; these tests deliberately assert
    that `NaN` comparisons return `false`, which is the correct IEEE 754
    behaviour the tests are verifying.

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

[0.2.0]: https://github.com/karhunen-loeve/fovea/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/karhunen-loeve/fovea/releases/tag/v0.1.1

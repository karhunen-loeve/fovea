# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Image pyramids. `image::Pyramid<L>` is a multi-resolution container
  generic over its level type, never empty, with `depth`, `level`/`get`,
  `finest`/`coarsest`, `iter`, and a `try_from_levels` constructor for
  custom builders. The constructor **validates** its input instead of
  trusting it: an empty list is `Error::EmptyPyramid`, and levels that
  grow along either axis (they must be ordered finest → coarsest;
  equal sizes are allowed) are `Error::PyramidLevelOrder` naming the
  first offending index — levels are never silently re-sorted.
  `image::PyramidLevel` is the base level trait; `Image<P>`
  implements it directly, so a Gaussian pyramid is `Pyramid<Image<P>>`
  (aliased `image::GaussianPyramid<P>`) with no wrapper cost.
- `transform::pyr_down` / `transform::pyr_up`: the standard
  resolution-halving and -doubling primitives. Both use the pinned binomial
  5-tap kernel `[1, 4, 6, 4, 1] / 16` per axis (effective σ exactly 1.0)
  with reflect-without-edge-duplication borders, matching OpenCV's
  `pyrDown`/`pyrUp`. `pyr_down` output size is ceiling division
  (`(n + 1) / 2`, even-sample decimation); `pyr_up` takes an **explicit
  target size** so odd-sized parents reconstruct exactly instead of
  guessing between `2·n` and `2·n − 1`, and returns
  `Result<Image<P>, Error>` — a target that cannot be the parent of the
  input is reported as the new `Error::InvalidPyrUpTarget` (a relation
  between two runtime sizes is a recoverable data error, not a panic).
  Both require `LinearSpace` (linearize sRGB first; Bayer CFA data is
  rejected at compile time).
- `transform::PyramidMethod<P>`: pyramid construction strategy trait,
  consumed at build time and not stored in the result. `transform::Gaussian`
  is the first strategy: repeated `pyr_down`, level 0 a copy of the input.
  `max_depth` is an upper bound — builds clamp at the minimum usable level
  size (1×1) instead of erroring, and always contain at least one level.
- Scale capability traits for pyramid levels: `image::Decimated` exposes
  the full affine level→base mapping (`pixel_distance`, `origin_offset`,
  `to_base`) in the pixel-center convention, so keypoints found on a coarse
  level lift exactly into base-image coordinates instead of hand-rolled
  `x · 2^level` math; `image::ScaleLevel` exposes the absolute Gaussian σ
  in base-image pixels. A plain `Pyramid<Image<P>>` implements neither and
  pays nothing; the new `image::ScaledImage<P>` wrapper carries the
  metadata for callers who need it. Its constructor is **total**: it takes
  the invariant-carrying `PixelDistance` and `Sigma` parameter types (see
  below), so an invalid value is caught where it is constructed, not where
  the level is wrapped. `Decimated::pixel_distance()` and
  `ScaleLevel::sigma()` return the same types, so metadata flows through
  chains without re-validation.
- `Sigma` and `PixelDistance`: **invariant-carrying parameter types**
  (finite and strictly positive), the `std::num::NonZeroUsize` pattern
  applied to algorithm parameters. Literals use the `const fn new`
  (in a `const` context an invalid literal **fails to compile**; at
  runtime it panics deterministically on first execution); values
  computed from data use `try_new`, which returns the new
  `Error::InvalidParameter` so a NaN from an estimator or a formula chain
  is a value, not a crash. Functions taking these types are total in
  them: `gaussian_blur` / `gaussian_blur_with` (+ `_into` variants),
  `gaussian_kernel_1d` / `gaussian_kernel_size`, and `canny` now take
  `Sigma` instead of a raw `f32` and no longer document a `sigma <= 0`
  panic. (`truncate` stays a plain `f32` — a structural kernel-shape
  constant — and the `MAX_RADIUS` capacity bound remains a documented
  panic, testable up front via `gaussian_kernel_size`.)
- New `features` module: the keypoint data model feature detectors produce
  and descriptors consume. Each property a keypoint may carry is one trait
  adding one guarantee — `features::HasPosition` (sub-pixel location),
  `features::HasResponse` (detector strength, ranking only),
  `features::HasScale` (characteristic σ in base-image pixels) and
  `features::HasOrientation` (a directed `Orientation`) — instead of one struct
  whose fields are meaningless for half of its producers. Detectors return
  the precise type they can justify, consumers bind the minimum they
  require, and the mismatch is a **compile error**: `features::Corner`
  (position + response, for single-resolution detectors such as Harris,
  Shi-Tomasi and FAST) does not implement `HasScale`, so it cannot reach an
  operation that sizes a patch from a detected scale.
  `features::ScaleKeypoint` (position + response + scale) is the
  scale-selecting counterpart, carrying its σ as the invariant-carrying
  `Sigma` so it flows on — into a `gaussian_blur`, into a patch size —
  without re-validation. `HasOrientation` is defined but has no implementor
  yet: it is the capability rotation-invariant descriptors will bind
  against, and nothing in this release can compute an orientation
  honestly.
- `features::Corner::from_level` / `features::ScaleKeypoint::from_level`:
  the named level→base lift. A keypoint detected on a coarse pyramid level
  is *reported* in base-image coordinates, so keypoints from different
  levels are comparable, and the conversion happens in exactly one place —
  `image::Decimated::to_base` — rather than being re-derived as
  `x · 2^level` per detector, which drops the grid-origin term and drifts
  half a pixel per octave. `ScaleKeypoint::from_level` additionally takes
  the level's σ from `image::ScaleLevel::sigma`, which is already absolute
  in base-image pixels and so is *not* rescaled. Both are total.
- `image::Decimated::to_local`: the inverse of `to_base`
  (`local = (base − origin_offset) / pixel_distance`), completing the
  affine level↔base map. Detection lifts out of a level; anything sampling
  back into one — a descriptor reading a patch around a base-frame
  keypoint — comes back through here, so neither direction is hand-rolled
  at a call site. Total: a `PixelDistance` cannot be zero.
- `features::retain_top_n`, `features::sort_by_response` and the
  `features::by_response_then_position` comparator: **deterministic**
  keypoint selection. Ordering is by response descending with a tie-break
  on `(y, x)` ascending, because exact response ties are the rule rather
  than the exception on synthetic images — without the tie-break, "the
  strongest 50 corners" would depend on the order the detector happened to
  visit pixels in, and could not be asserted in a test. Comparisons use
  `total_cmp`, so a NaN response from a misbehaving detector still yields a
  consistent total order instead of a sort that silently loses keypoints;
  the sort is stable, so keypoints identical in response *and* position
  keep the order they were produced in.
- `features::detect`: the first corner detectors, and the first producers of
  the keypoint model above. `features::detect::Harris` (the classical
  `det(M) − k·tr(M)²`) and `features::detect::ShiTomasi` (`λ_min(M)`, no `k`
  to tune) are two **response strategies** over one shared pipeline rather
  than two detectors: `features::detect::detect_corners(&image, &method,
  params)` runs Sobel gradients → gradient products → Gaussian window →
  response → threshold and peak selection, and returns `Vec<Corner>` in
  raster order. A third measure is an implementation of the open
  `features::detect::CornerResponse` trait, not a copied pipeline; the
  arithmetic is generic over `f32` and `f64` accumulators through the sealed
  `features::detect::CornerResponseChannel`, the same two-trait split
  `Magnitude` / `MagnitudeChannel` uses. Single-channel input is a
  compile-time requirement (`pixel::SingleChannel`), not a runtime check.
- `features::detect::Harris` and `features::detect::CornerParams` are
  **invariant-carrying parameter types** (the `Sigma` pattern, ADR-0025
  category E): `const fn new` for literals, `try_new` returning
  `Error::InvalidParameter` for computed values. `Harris` owns its own
  sensitivity, and its domain `0 < k < 0.25` is a consequence rather than a
  convention — `det ≤ tr²/4` for a symmetric 2×2 matrix, so at `k ≥ 0.25`
  the response is non-positive for *every* tensor and the detector can never
  fire, while at `k ≤ 0` the edge penalty becomes an edge reward. Both ends
  fail silently, which is why they are rejected at construction.
  `CornerParams` carries the window σ, the absolute response threshold
  (finite — a NaN threshold would reject every pixel and look like an empty
  image) and the suppression radius (at least 1). There is deliberately no
  `Default`: a default σ and threshold would be a claim about *your* images.
- `features::detect::StructureTensor` and
  `features::detect::corner_response_map`: the detector's stages, public in
  their own right the way `canny`'s are. `StructureTensor::from_gradients`
  takes **your** gradient images (so Scharr or Prewitt instead of Sobel is a
  choice, not a fork) and windows their products with a Gaussian;
  `StructureTensor::from_smoothed` takes the three already-windowed products,
  so a box window computed from an integral image needs no new API. Both
  report a `Error::SizeMismatch` rather than panicking when the inputs
  disagree. `corner_response_map` returns the cornerness image itself — for
  visualization, for a different thresholding rule, and for calibrating a
  threshold, which the documentation recommends over guessing: the response
  scales with the gradient operator's gain and the image contrast raised to
  the measure's own power (squared for Shi-Tomasi, *fourth* for Harris), so
  the same picture as `Mono8` rather than `MonoF32` scores ≈ 255⁴ higher.
- `features::detect::corner_peaks`: threshold-and-local-maximum selection
  over any response map, shared by every detector in the module. The radius
  is the minimum separation between two reported corners; the comparison
  window is **clipped** at the image border rather than skipped, so a corner
  against the frame edge is still reported. Ties are resolved
  asymmetrically — strictly greater than neighbours earlier in raster order,
  greater or equal to later ones — so a flat plateau yields exactly one
  corner (its raster-first pixel) instead of all of them under `>=` or none
  under `>`. A `NaN` compares false against everything, so it neither wins a
  plateau nor survives its own threshold test.
- `features::detect::detect_corners_in_level`: the same detector on a
  `image::Decimated` pyramid level, with every position lifted into the
  base-image frame through `Corner::from_level`, so results from different
  levels are comparable and concatenate. Multi-resolution search
  deliberately still returns `Corner` and **not** `ScaleKeypoint`: the
  level's σ was imposed by whoever built the pyramid, not selected by the
  detector, and a scale the detector did not choose is exactly the
  conditionally-valid field the capability traits exist to prevent.
  Cross-level duplicate suppression is the caller's policy — the same
  physical corner legitimately appears once per level.
- `Orientation` and `AxialOrientation`: **angle vocabulary types** that state
  the one thing a bare float cannot — the **modulus**. `Orientation` is a
  *directed* angle mod 2π, canonicalized to (−π, π] (a gradient direction, a
  dominant feature orientation); `AxialOrientation` is an *undirected* axis
  mod π, canonicalized to (−π/2, π/2] (a region's major axis, which has no
  head or tail, so +80° and −100° are the same axis). They are separate
  types because they are not interchangeable: they wrap at different moduli,
  so subtracting one from the other is meaningless, and only the type system
  can say so. Both own the operation that hand-rolled float arithmetic gets
  wrong — `signed_difference`, which wraps across the seam: two directions at
  179° and −179° are 2° apart, not 358°, and two axes at 80° and −80° are 20°
  apart, not 160°. Construction **normalizes** rather than rejects (unlike
  `Sigma`, whose invariant makes σ ≤ 0 meaningless, 7.0 radians is not an
  invalid angle — it is 0.717): `from_radians` wraps and returns
  `Error::InvalidParameter` only for NaN/∞, while `Orientation::from_atan2`
  and `AxialOrientation::from_half_atan2` are **total** — `atan2`'s range is
  already canonical, so the hot path (a per-sample gradient angle) does no
  wrapping at all. `Orientation::to_axial` folds a direction onto its axis;
  the reverse is deliberately absent, since an axis names two opposite
  directions. Note that canonicalizing makes `==` *meaningful* (the same
  direction compares equal) but not bit-exact — wrapping rounds — so compare
  with `signed_difference(…).abs() < tolerance`. Neither type implements
  `PartialOrd`: on a circle there is no least angle, and `a < b` would invite
  reading "counter-clockwise of", which it is not.

### Changed

- **Breaking:** the error-handling convention was sharpened: a `panic!` is
  reserved for contracts that are locally decidable at the call site
  (indexed access with a `get` alternative, caller-allocated `_into`
  output buffers, structural constants, internal invariant backstops);
  everything whose validity depends on data — validating constructors and
  relations between separately obtained runtime values — returns
  `Result<_, Error>`. Three previously panicking sites move accordingly:
  - `match_template` / `match_template_into` report a zero-width or
    zero-height template as the new `Error::EmptyTemplate` instead of
    panicking — the template is data (typically a crop or a file), and its
    other data failure (`TemplateTooLarge`) was already an error.
  - `ImagePlanes::replace_plane` returns `Result<Image<_>, Error>`:
    a size-mismatched replacement plane is `Error::SizeMismatch` (the
    plane is data); an out-of-range plane *index* still panics, the same
    data-vs-constant split as `i32::from_str_radix` (`Err` for the
    string, panic for the radix).
  - `otsu_binary_mask` is bound on `pixel::SingleChannel` instead of
    asserting `CHANNEL_COUNT == 1` at runtime — a multi-channel pixel
    type is now a compile error, matching `hysteresis_threshold`.
  - `non_maximum_suppression` returns `Result<Image<P>, Error>`: a
    magnitude/direction size mismatch is `Error::SizeMismatch` rather than
    a panic, since it is a relation between two separately obtained runtime
    sizes.
- **Breaking:** `BlobMeasurements::orientation` returns `AxialOrientation`
  instead of a raw `f64` (see *Added*). The value, range and y-down sign
  convention are unchanged — `½·atan2(2·μ11, μ20 − μ02)` in (−π/2, π/2] — but
  the type now states that the angle is an **axis**, not a direction, which a
  bare `f64` could not: nothing previously stopped a blob's orientation being
  compared against, or subtracted from, a mod-2π feature orientation, and the
  result would have been silently wrong near the seam. Call `.radians()` for
  the bare angle, or prefer `.signed_difference(other)` when comparing two
  blobs' axes, which wraps at π instead of reading 80° and −80° as 160° apart.
- **Breaking:** `gaussian_blur` / `gaussian_blur_with` (+ `_into`
  variants), `gaussian_kernel_1d` / `gaussian_kernel_size`, and `canny`
  take the new `Sigma` parameter type instead of a raw `f32` σ (see
  *Added*). Wrap literals in `Sigma::new(…)`; validate computed values
  with `Sigma::try_new(…)?` where they are produced.

### Fixed

- `canny`'s documentation listed the wrong accumulator for `Mono16`: it
  claimed `Mono16` widens to `MonoF64` like `Mono32` and `Mono64`, when in
  fact `Mono16` — and `Mono<BITS>` — accumulate in `MonoF32`. Documentation
  only; the bound the compiler enforces never changed, so no behaviour or
  signature moves with this. Corrected while writing the same sentence for
  the corner detectors, which do widen `Mono32` and up.

## [0.3.0] — 2026-07-27

### Added

- `pixel::SingleChannel`: a sealed marker trait for pixel types with exactly
  one channel (`Mono8/16/32/64`, `Mono<N>`, `MonoF32`/`MonoF64`, `Label32`,
  `Indexed8`, and the primitive scalar pixels). It turns "this operation is
  monochrome-only" into a bound the compiler checks;
  `hysteresis_threshold` now uses it in place of a runtime
  `CHANNEL_COUNT` assertion.
- `transform::MagnitudeHypot`: an overflow-safe `CombinePixels` sibling of
  `Magnitude`, computing `sqrt(a² + b²)` via `f32::hypot` / `f64::hypot`.
  Use it when inputs can approach the limits of the float range; `Magnitude`
  itself now takes the direct, vectorizable route (see *Changed*). The
  sealed `MagnitudeChannel` trait gained a matching `magnitude_hypot`
  method beside `magnitude`.
- `analyze::components::connected_components_with_measurements`: opt-in blob
  shape analysis. Returns one `BlobMeasurements` per component alongside the
  labeling — area, bounding box, centroid sums, the raw second-order moment
  sums (`sum_x2`/`sum_y2`/`sum_xy`), and a 4-connected boundary-pixel
  `perimeter` — with derived `f64` descriptors (`centroid`,
  `equivalent_diameter`, `orientation`, `eccentricity`, `circularity`,
  `central_moments`) computed on demand. Everything accumulates in the same
  single pass 2 as the labeling; no separate contour extraction. The perimeter
  boundary test is fixed at 4-connectivity independent of the labeling
  `Connectivity`, and all measurements are view-relative (a blob clipped by the
  view edge is measured as clipped). The cheap `connected_components_with_stats`
  path is unchanged: the extra moment + boundary work is gated behind a
  monomorphised sink and compiles away when not requested. Note the 4-connected
  boundary-pixel count undercounts diagonal outline, so `circularity` of a
  rasterised disc reads ≈1.25 (above 1) while a square reads ≈π/4 — treat it as
  a relative shape score within a tolerance band.
- `CoordinateF64`: sub-pixel `f64` companion to `Coordinate`, with
  `From<Coordinate>`/`From<(f64, f64)>`. `BlobMeasurements::centroid`
  returns it, and `ComponentStats::centroid` was changed to return it (see
  *Changed → Breaking*).
- `analyze::edge::canny`: single-scale Canny edge detector composing the full
  pipeline — `gaussian_blur(sigma)` → Scharr `Gx`/`Gy` → gradient magnitude +
  direction → non-maximum suppression → `hysteresis_threshold` — and returning
  a `BinaryImage`. `sigma` is a true Gaussian standard deviation; `low`/`high`
  are absolute, kernel-independent gradient-magnitude thresholds (stable
  because the blur preserves brightness). Generic over any single-channel
  input whose linear accumulator is a float pixel: `Mono8`/`MonoF32` accumulate
  in `MonoF32`, `Mono16`/`Mono32`/`Mono64`/`MonoF64` in `MonoF64`. Every stage
  is a public function, so callers can swap operators or inspect intermediates
  by composing the pipeline by hand. Demo: `fovea-examples/src/canny.rs`.
- `transform::gradient_magnitude` and `transform::gradient_direction`: named
  wrappers over `combine_images` with the `Magnitude` (L2, `hypot`) and the new
  `Direction` (`atan2`) strategies, fusing an `Gx`/`Gy` pair into an
  edge-strength map or a gradient-angle map. Both are generic over the
  float-channel pixel types the strategies support (`MonoF32`, `MonoF64`, …)
  and return `Err(Error::SizeMismatch)` on differing input sizes.
- `transform::non_maximum_suppression`: thins a gradient-magnitude ridge to
  single-pixel width by zeroing any pixel that is not a local maximum along its
  quantised gradient direction (sectors 0°/45°/90°/135°). Ties are kept
  (inclusive `>=`, OpenCV-compatible); border pixels whose along-gradient
  neighbour is out of bounds are suppressed. Generic over single-channel float
  pixels; panics on a magnitude/direction size mismatch.
- `transform::Direction` (and the sealed `transform::DirectionChannel` trait):
  a `CombinePixels` strategy computing channel-wise `atan2(b, a)` in radians on
  `(-π, π]`, the directional companion to the existing `Magnitude` strategy.
  Defined for `f32` / `f64` channels.
- `analyze::threshold::adaptive_threshold` (and `_into`): local-mean
  adaptive thresholding for uneven illumination — a pixel is foreground iff
  `pixel > local_mean(window) − bias`, returning a `BinaryImage`. Built by
  composition over the integral-image engine, so the per-pixel window mean
  is `O(1)` and the whole pass is `O(n)` regardless of `window` size. The
  accumulator is named explicitly (`Mono32` / `Mono64` / `MonoF64`) exactly
  as for `integral_image`, and fixes the offset domain via the new `Bias<A>`
  newtype (`i64` for integer accumulators, `f64` for `MonoF64`); positive
  bias biases toward foreground. The decision `(pixel + offset) · area > sum`
  is evaluated in an exact `i128` (integer) / `f64` (float) domain — no
  per-pixel division or rounding. Edges use a **clipped** window (exact,
  allocation-free; matches scikit-image's `threshold_local`, differs from
  OpenCV's replicate border). The boundary is strict `>` (equality is
  background, matching Otsu). Single-channel-ness is enforced at compile time
  (the accepted accumulators are only valid for monochrome sources).
  Inherits `Error::AccumulatorOverflow` (Tier 2) from the integral
  pre-flight; panics (Tier 3) on an even/zero `window` or an `out`/input size
  mismatch. New public items: `adaptive_threshold`, `adaptive_threshold_into`,
  `Bias`, and the sealed `AdaptiveAccumulator` trait.
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

- **Breaking:** `ComponentStats::centroid` returns `CoordinateF64` instead
  of `(f64, f64)`. Destructuring call sites (`let (cx, cy) =
  stats.centroid();`) no longer compile; use `let c = stats.centroid();`
  with `c.x` / `c.y`, or `let (cx, cy) = stats.centroid().into();`. The
  sibling `BlobMeasurements::centroid` is new in this release and returns
  the same type, so the two agree.
- **Breaking:** `hysteresis_threshold` / `hysteresis_threshold_into` are
  bound on the new `pixel::SingleChannel` marker instead of
  `HomogeneousPixel`. Passing a multi-channel pixel is now a compile error
  rather than a runtime panic; every single-channel type that previously
  worked still works unchanged.
- **Breaking (numerical, narrow):** the `Magnitude` combine strategy — and
  therefore `gradient_magnitude` and the magnitude stage of `canny` —
  computes `sqrt(a² + b²)` directly instead of calling
  `f32::hypot` / `f64::hypot`. The results are identical for any input
  whose square is representable, which covers gradients of real image
  data; the direct form inlines and autovectorizes where the libm call did
  neither. Inputs large enough to overflow the square (above ≈1.8·10³⁸ for
  `f32`) now yield `inf` where they previously yielded a finite value —
  use the new `MagnitudeHypot` strategy if that matters for your data.
- `canny` no longer materialises an `atan2` gradient-direction image.
  Non-maximum suppression reads the sector it needs straight from
  `gx`/`gy`, removing one transcendental call and one full-image
  allocation per invocation. The output mask is unchanged, and the staged
  `gradient_direction` → `non_maximum_suppression` composition stays
  public for callers who want to inspect the angle map.
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
  kernel-allocation half of the previously documented separable-convolution
  allocation deviation. The image-sized working-set buffers (intermediate +
  accumulator + output) are unchanged; reusing those across calls is the
  deferred follow-up (tracked in OPT-004). The
  `SeparableKernel::to_h_image` / `to_v_image` helpers were removed.
- `hysteresis_threshold_into` builds its weak mask by row iteration rather
  than a per-pixel `pixel_at`, and `non_maximum_suppression` walks
  previous/current/next row slices instead of recomputing a bounds-checked
  index per neighbour. Behaviour is unchanged.

### Fixed

- `BlobMeasurements::eccentricity` documented its range as `[0, 1)`, but a
  straight axis-aligned blob has `λ₂ = 0` and returns exactly `1.0`. The
  range is `[0, 1]`; the doc and its test now say so.
- `gaussian_kernel_size` / the internal radius helper now document that
  they deliberately do **not** enforce `MAX_RADIUS`, unlike
  `gaussian_kernel_1d` and the `gaussian_blur*` family, which panic above
  it. Keeping the size query total is what lets a caller test whether a
  `sigma` is admissible instead of catching a panic.
- Shipped documentation no longer cites `PHILOSOPHY.md` sections or
  internal plan "Decision N" tags. Those files are not part of the
  published crate, so the references dangled on docs.rs; each is replaced
  by the reasoning it stood for.

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

[0.3.0]: https://github.com/karhunen-loeve/fovea/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/karhunen-loeve/fovea/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/karhunen-loeve/fovea/releases/tag/v0.1.1

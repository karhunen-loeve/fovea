# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `OriginOffset`: the invariant carrier for a pyramid level's origin,
  finite along both axes, with a const `ZERO` for the unshifted case that
  is what `pyr_down` levels have. `ScaledImage::new` takes it in place of
  a raw `CoordinateF64` (**breaking**), which makes the constructor's
  documented totality true: a NaN or infinite origin used to poison every
  `to_base` lift while the docs said nothing could fail.
- `CoordinateI32`: the signed member of the coordinate family, for
  whole-pixel positions that may legitimately lie off-frame. The five
  `draw` shapes carry it in their fields (**breaking** for struct
  literals: add `.into()`), while the free functions take
  `impl Into<CoordinateI32>`, so tuple call sites keep compiling.
  `TryFrom<Coordinate>` bridges from contour and component output, and a
  `(y, x)` transposition is now visible wherever shapes are stored or
  built from data.
- The seven public parameter and angle newtypes (`Sigma`, `PixelDistance`,
  `Tolerance`, `OddWindowSide`, `Orientation`, `AxialOrientation`,
  `PeakValue`) and `NmsRadius` are `#[repr(transparent)]`, making the
  wrapper-equals-inner layout a guarantee rather than an accident.
- `Pyramid` and `ScaledImage` derive `Debug` (alongside their existing
  `Clone`), matching the rest of the crate's types.
- `Extremum` lives in the crate root as shared vocabulary: it always served
  both `analyze::peak` and `transform`'s template matching, and now it is
  defined beside `Coordinate` and `Sigma`. `analyze::peak::Extremum`
  re-exports it, so existing imports keep compiling.
- The docs.rs guide gained four FAQ entries covering the release's new
  arcs: finding corners, getting geometry out of a binary mask, comparing
  two images, and drawing results onto an image. The README module table
  now lists `draw` and the new `analyze` and `pixel` members.

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
  applied to algorithm parameters. Literals use the `sigma!` /
  `pixel_distance!` macros, which are inline `const { }` blocks, so an
  invalid literal **fails to compile** wherever it is written; values
  computed from data use `try_new`, which returns the new
  `Error::InvalidParameter` so a NaN from an estimator or a formula chain
  is a value, not a crash. `new` is the checked `const fn` under the
  macros and returns `Option<Self>`; there is no panicking constructor. Functions taking these types are total in
  them: `gaussian_blur` (+ its `_into` variant),
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
  than two detectors: `features::detect::detect_corners(&image, method,
  params)` runs Sobel gradients → gradient products → Gaussian window →
  response → threshold and peak selection, and returns `Vec<Corner>` in
  raster order. A third measure is an implementation of the open
  `features::detect::CornerResponse` trait, not a copied pipeline; the
  arithmetic is generic over `f32` and `f64` accumulators through the sealed
  `features::detect::CornerResponseChannel`, the same two-trait split
  `Magnitude` / `MagnitudeChannel` uses. Single-channel input is a
  compile-time requirement (`pixel::SingleChannel`), not a runtime check.
- `features::detect::Harris` and `features::detect::CornerParams` are
  **invariant-carrying parameter types** (the same discipline as `Sigma` and
  `std::num::NonZeroUsize`): `const fn new` returning `Option<Self>`, plus
  the `harris!` literal macro for the single-scalar `Harris`, and `try_new`
  returning `Error::InvalidParameter` for computed values. `Harris` owns its own
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
- **FAST**, the segment-test detector, as a second family in
  `features::detect` sharing the first's keypoint type and peak stage
  unchanged. `features::detect::fast(&image, params, &Skip)` scores every
  pixel against the 16-pixel radius-3 ring and returns `Vec<Corner>` in
  raster order; `fast_in_level` is the base-frame pyramid variant, and stays
  `Corner` for the same reason `detect_corners_in_level` does. That two
  detectors this unalike — one averaging gradients over a Gaussian window,
  one reading seventeen raw samples — need no change to `Corner`,
  `HasResponse`, `corner_peaks` or `retain_top_n` is the evidence the
  keypoint model was not shaped around Harris.
- `features::detect::SegmentTest` and `features::detect::FastParams` are
  **invariant-carrying parameter types**, matching `Harris` / `CornerParams`:
  `const fn new` returning `Option<Self>`, and `try_new` returning
  `Error::InvalidParameter` for computed values. Neither takes a literal
  macro: their argument names are the information. `SegmentTest` owns the threshold and the arc length
  because neither decides anything alone, and both its domains are
  consequences rather than conventions. The threshold must be strictly
  positive — at `t = 0` every pixel of a flat field passes on all 16 ring
  positions, so the detector would report the whole image. The arc length
  must satisfy `9 <= n <= 16`: above 16 no arc exists and the detector is
  dead, and at `n <= 8` a straight step edge (which puts up to 8 contiguous
  ring pixels on one side) passes, so the detector stops distinguishing
  corners from edges. As with `Harris`, both failures are silent, which is
  why they are rejected at construction.
- The **threshold is in intensity units**, unlike the structure tensor's:
  `20.0` on a `Mono8` image means twenty grey levels and `0.08` on a
  `MonoF32` image in `0.0..=1.0` means eight per cent contrast. No operator
  gain and no squaring enter it, so it can be reasoned about — from a noise
  estimate, say — instead of calibrated against a response map. Widening
  uses `LinearPixel::to_accumulator`, which does not rescale, so the score is
  reported in those same units.
- `features::detect::fast_score_map` and `features::detect::fast_score_at`:
  the detector's stages, public the way `corner_response_map` is. The score
  is the **largest threshold at which the pixel still passes**, so a pixel is
  a corner at `t` exactly when its score is at least `t` — one number serves
  as both the segment test's threshold and the peak stage's, and raising the
  threshold on an already-computed map is exact. The map is always the
  input's size (so a position in it is a position in the image) and is
  `Image<MonoF32>` whatever the input is, because `HasResponse::response`
  is `f32` and precision the keypoint cannot carry would be precision the map
  only pretends to have. `fast_score_at` returns `Option<f32>`: `None` is
  "the border policy does not score this position", which is a different
  answer from `Some(0.0)`, "scored, and not a corner".
- Border handling reuses the crate's ordinary `BorderPolicy` vocabulary
  rather than a FAST-specific rule. `border::Skip` is the natural choice and
  declines the 3-pixel margin where the ring does not fit — a detection there
  would be made from invented samples — while any full-frame policy
  (`Clamp`, `Mirror`, `Constant`) extends the image and reports corners
  against the frame edge. Declined positions are written as `0.0` in the map,
  which is what every other non-corner reads.
- `features::detect::FAST_RING` and `FAST_RING_RADIUS` are public: the ring
  *is* the detector's geometry, its clockwise order is what "contiguous"
  means, and drawing it over an image is how a score is explained.
- The classical four-point early rejection ships as an **optimization behind
  a benchmark** (`benches/features.rs`), generalised from the textbook
  "three of four cardinals for FAST-12" to `arc_length / 4` of them for any
  arc length — a window of `n` consecutive ring positions covers at least
  that many of the four, however it is placed. It cannot change an answer
  (a test asserts the shipped path matches the plain scan pixel for pixel),
  and it is worth 1.61× at `n = 9`, 5.93× at 12 and 14.6× at 16 on a 512×512
  `Mono8` texture. Its one visible consequence is documented: the score map
  is floored at the test's own threshold, so a corner too faint for the test
  reads `0.0` rather than its true margin.
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

- `analyze::contours`: **contour extraction**, giving blobs geometry where
  the component measurements give them aggregates.
  `contours::extract_contours(&binary)` labels the foreground with the
  caller's connectivity and the background with its **dual** (8-connected
  foreground pairs with 4-connected background and vice versa — the
  pairing that keeps "hole" well defined on the grid), classifies each
  background region as outside or hole, and Moore-traces every component's
  outer border plus one inner border per hole. It returns the `Labeling`
  alongside a `contours::ContourHierarchy` indexed the same way (component
  `i` ↔ label `i + 1`), so contours join the stats and measurements tables
  with no translation. Nesting is explicit and unbounded:
  `ComponentContour::enclosing` names the component inside whose hole this
  one sits, and `ComponentContour::euler_number` is `1 − holes`. Thin
  structures trace out-and-back and single pixels yield one-point
  contours — both terminate, which is exactly the case the textbook
  stopping shorthand gets wrong.
- `contours::Contour`: a **certified traced border** — a closed chain of
  integer border pixels in trace order, consecutive points 8-adjacent by
  construction, outer borders clockwise on screen and hole borders
  counterclockwise (a documented property; the outer/hole distinction is
  the explicit `ContourKind`, never decoded from winding). Shape
  descriptors are derived on demand in `f64` and degenerate cases are
  `Option`, not NaN: `area` (shoelace, through pixel *centers* — a 6×6-px
  square measures 25.0, a different question than the pixel-count 36),
  `perimeter` (exact polygon length, diagonals √2), `centroid`,
  `circularity`, `convex_hull`, `solidity`, `chain_code`. There is no
  public constructor — arbitrary vertex lists use the free polygon
  functions instead.
- The contour-side `circularity` is the **geometric** score the
  boundary-pixel count cannot be: bounded by ~1 instead of reading ≈1.25
  for a rasterised disc. It is not bias-free — the traced chain carries
  the 8-connected staircase, so a disc of radius 20 measures ≈0.87 raw
  and ≈0.94 after Douglas–Peucker at ε = 0.8 — and the residual is
  documented with numbers rather than rounded away. No estimator with
  fitted weights is applied silently; if the underlying outline is smooth,
  simplifying first is the caller's named step.
- `contours::polygon_area` / `polygon_perimeter` / `polygon_centroid` /
  `convex_hull` / `approximate_polygon`: **free polygon functions** over
  any `&[Coordinate]` treated as a closed polygon, shared by `Contour`'s
  methods and usable on simplified vertex lists. The shoelace and hull
  arithmetic is exact in integers (`i128` accumulation, no epsilon);
  `convex_hull` is Andrew's monotone chain returning strict corners in
  clockwise-on-screen order, deterministic and input-order-invariant.
  `approximate_polygon` is Douglas–Peucker over a closed polygon with an
  **explicit ε** — nothing in the crate ever simplifies a contour
  implicitly, because the right tolerance is a claim about *your* images.
- `contours::ChainCode` / `contours::ChainDirection`: the compact
  encoding — a start pixel plus one byte-sized Freeman direction per
  border step, with the y-down offsets stated on each variant.
  `ChainCode::from_contour` is total (8-adjacency is certified by
  `Contour`) and `to_points` round-trips exactly.
- `Tolerance`: a third **invariant-carrying parameter type** beside
  `Sigma` and `PixelDistance` — a geometric tolerance in pixels, finite
  and non-negative. Zero is deliberately valid (ε = 0 removes exactly the
  collinear vertices), which is why the strictly-positive `PixelDistance`
  was not reused. The `tolerance!` macro for literals, `try_new` returning
  `Error::InvalidParameter` for computed values, and a `const fn new`
  returning `Option<Self>` underneath.
- `OddWindowSide`: a fourth **invariant-carrying parameter type** beside
  `Sigma`, `PixelDistance` and `Tolerance`, holding the side length of a
  square neighbourhood centred on the pixel being processed, odd and
  non-zero.
  The `window!` macro for literals (an even literal **fails to compile**,
  wherever it is written), `try_new` returning `Error::InvalidParameter` for
  values computed from data, and `radius()` for the half-width, which is exact
  because the side is odd. `adaptive_threshold` / `adaptive_threshold_into`
  take it instead of a bare `usize` (see *Changed*), which removes their
  "window must be odd and non-zero" panic entirely.
- `analyze::threshold::HysteresisThresholds<C>`: the `low <= high`
  threshold pair as one value. The invariant is a *relation*, so neither
  number is checkable on its own and the pair is what gets validated, once,
  where it is born. `try_new` (returning `Error::InvalidParameter`) is the
  only constructor, for literals and for thresholds derived from data alike,
  such as fractions of a measured magnitude peak. `C` is the comparison channel,
  and only `PartialOrd` is required, so the float case works: `!(low <=
  high)` is also exactly the test that rejects a **NaN** threshold, which
  would otherwise pass silently and return an empty mask, since every
  comparison against NaN is false. Unlike the other parameter types there
  is no `const fn new` and no literal macro, because the comparison goes
  through `PartialOrd` on a generic channel and trait methods cannot be
  called in a `const fn`. With no compile-time tier to protect, an
  `Option`-returning `new` would differ from `try_new` only by discarding
  the reason, so it does not exist.
- `transform::Clamp::try_new`: the validating constructor, returning
  `Error::InvalidParameter` and naming the first channel where `lo > hi`. A
  clip range derived from image data (a histogram percentile, an exposure
  estimate) can come out inverted for reasons that are not a programmer bug.
  It replaces the panicking `Clamp::new`, which is **removed** (see
  *Changed*): `Ord` on the channel puts a `const` constructor out of reach,
  so there is no compile-time tier for a second constructor to preserve.
- `analyze::components::Connectivity::Dual`: each connectivity now names
  the connectivity the background must be labeled with when the foreground
  uses it (`Connectivity8::Dual = Connectivity4` and vice versa). Additive:
  the trait is sealed, so no external implementor can break.
- **The Bayer pixel family** — `pixel::bayer`, twenty
  `#[repr(transparent)]` colour-filter-array types: four patterns (RGGB,
  BGGR, GRBG, GBRG) × five depths (8, 10, 12, 14, 16). The sub-word depths
  are aliases of a const-generic `BayerRggb<BITS>` over `Mono<BITS>`, so
  they clamp the same way; the fixed widths wrap `Saturating<u8>` /
  `Saturating<u16>`. Raw sensor data stops being indistinguishable from a
  monochrome frame: `Image<BayerRggb12>` and `Image<BayerBggr12>` are
  different types, so two cameras' frames cannot be mixed.
  **What these types withhold is the point.** They implement `LinearPixel`
  — a weighted sum of raw samples is real work (defect-pixel
  interpolation, local noise estimation, hot-pixel detection), and its
  accumulator is `MonoF32` — but **not `LinearSpace`**, so `blend()` and
  `Bilinear` resize are compile errors; and **not `OriginInvariantPixel`**,
  so `roi`, `tiles`, and `sliding_windows` do not exist for them. Both
  operations mix or re-label neighbouring samples, which in a mosaic means
  mixing or re-labelling colours. `Ord`, `PlainPixel`, `HomogeneousPixel`,
  `SingleChannel`, `WhiteChannel`, and `ZeroablePixel` are all present, so
  thresholding, histograms, min/max, convolution, and zero-copy
  `cast_slice` from a camera buffer all work.
- `pixel::bayer::BayerPixel`: the generic-dispatch trait, carrying the tile
  arrangement as the compile-time constant `PATTERN` and the demosaic
  result type as `RgbOutput` (depth-preserving: `BayerRggb12` →
  `Rgb12`). Not sealed — a downstream float CFA type is a legitimate
  implementor.
- `pixel::bayer::BayerPattern` and `pixel::bayer::CfaColor`: the pattern
  vocabulary. `BayerPattern::tile()` gives the 2×2 arrangement and
  `color_at(x, y)` resolves an image coordinate to the colour that site
  sampled; both are `const fn`, so a `B::PATTERN`-driven lookup folds to a
  pair of parity tests. The SFNC mapping is documented on the enum —
  `BayerRG12` is RGGB, since SFNC abbreviates the tile to its first row.
- `image::BayerSubView` / `image::BayerSubViewMut`: phase-preserving region
  access, the named replacement for the `SubView` methods Bayer images do
  not have. `aligned_bayer_roi` / `aligned_bayer_roi_mut` return `None`
  for an odd `left` or `top` — the crop would silently re-label every
  sample — and `None` out of bounds, the same Tier 1 answer `roi` gives.
  Odd *width* and *height* are fine: truncating mid-tile drops samples but
  does not move the ones that remain. Implemented for every container that
  offers ordinary ROI — `Image<B>`, `ImageArray<B, W, H>`,
  `ImageRef<'_, B>`, and `ImageRefMut<'_, B>` — so the Bayer path has no
  coverage gap relative to `SubView`.
- `transform::BayerToMono`: the named escape hatch out of the CFA family,
  `BayerRggb12 → Mono12` and so on at every depth. Losing the colour a
  sample carries is a real loss, so — like every lossy conversion in this
  crate — it has to be named; there is no `From<BayerRggb12> for Mono12`.
- **Demosaicing** — `transform::demosaic` / `transform::demosaic_into`
  turn a CFA mosaic into RGB at the depth the sensor sampled
  (`Image<BayerRggb12> → Image<Rgb12>`, from `BayerPixel::RgbOutput`; the
  output type is not a call-site choice). Two strategies:
  `transform::BayerBilinear`, the reference algorithm — the average of the
  nearest sites of each missing colour — and `transform::MalvarHeCutler`,
  the quality path, four fixed 5×5 kernels that correct the bilinear
  estimate with a second difference read from the colour that *was*
  sampled at the site. Both are **exact at the sampled sites** (a measured
  sample is never mixed away) and exact wherever the channels vary
  linearly; Malvar–He–Cutler additionally clips rather than wraps where its
  negative weights overshoot.
  **These take no border policy, and that is deliberate.** A CFA sample's
  colour is a function of its coordinate parity, and reflection *without*
  edge duplication is the only policy in the crate that maps a coordinate
  to another of the same parity — `Clamp` duplicates the edge sample and so
  reads red where the kernel expects green, `Wrap` is safe only for even
  dimensions, `Constant` injects a value with no CFA colour at all. The
  reflection is therefore pinned into the contract, as `pyr_down` pins its
  kernel and border. Demosaicing an `aligned_bayer_roi` sub-view is
  well-defined for the same reason its origin must be even.
- `transform::DemosaicMethod<B>`: the CFA interpolation strategy trait. An
  implementation receives the site coordinate and an accessor for the
  border-resolved samples around it (`RADIUS` declares how far it reads)
  and returns the `RgbF32` triple; the engine owns the traversal, the
  interior/boundary split, and the pinned reflection. It is a *site*
  coordinate rather than a fixed weight grid because `FoldOp` applies one
  grid to the whole image and is deliberately blind to position, while a
  demosaic kernel is selected by `(x % 2, y % 2)`.
- **White balance** — `transform::white_balance` /
  `transform::white_balance_into` scale each raw sample by the gain of the
  colour its site sampled, returning the same Bayer type, so the result
  keeps its CFA phase and feeds straight into `demosaic`. Gains are
  `transform::BayerGains`, an invariant-carrying parameter type (finite and
  non-negative; `const fn new` returning `Option<Self>`, `try_new` →
  `Error::InvalidParameter` for ratios estimated from data), keyed by
  `CfaColor` with one gain shared by both green sites. Balancing the mosaic
  before interpolation is the industrial order and the one that matters for
  a channel-mixing algorithm like Malvar–He–Cutler; the cost is that gains
  above `1.0` clip at the sample depth. This is **not** a `ConvertPixel`
  strategy and cannot be one — a per-pixel conversion is blind to position,
  and a CFA sample's colour is its position.
- `transform::SeparableScratch<Acc>`: a caller-owned, reusable working set
  for separable convolution. A two-pass separable convolution needs an
  inter-pass intermediate image, a per-row accumulator and a kernel-position
  list; the one-shot functions allocate them per call, while a scratch owns
  them across calls. The reusing entry points are **methods on the scratch**,
  named exactly like their allocating free-function counterparts:
  `scratch.convolve_separable_into(&src, &kernel, &border, &mut out)` and
  `scratch.gaussian_blur_into(&src, sigma, &border, &mut out)`. The receiver
  expresses the reuse, so no name has to.
  With a caller-owned output as well, a blur in a hot loop — video frames,
  pyramid levels, scale-space octaves — performs **zero heap allocations
  after the first call**, which is asserted directly by a counting allocator
  in the test suite rather than claimed. Buffers grow to fit and are never
  shrunk, so a smaller frame after a larger one reuses the larger storage;
  only capacity carries over between calls, never contents, so one scratch
  can serve different images, kernels, σ values and border policies. Reuse
  stays explicit — there is no hidden pool and no global state, and the
  one-shot free functions are unchanged and remain the default. Two notes
  for callers. `Acc` is the accumulator pixel type (`MonoF32` for `Mono8`),
  so one scratch serves one accumulator type; input and output types may
  differ from it. And because the second pass now reads a borrowed view of
  the scratch, these methods bind
  `for<'r> BorderPolicy<ImageRef<'r, Acc>>` where the free functions bind
  `BorderPolicy<Image<Acc>>` — every built-in policy satisfies both, but a
  custom policy implemented only for `Image<T>` will need the wider impl.
- `image::SeparableWeights`: the trait the separable convolution engine
  consumes — two 1-D weight slices plus their anchors, and a stack-based
  `flipped()` for true convolution. `SeparableKernel<HK, VK>` (compile-time
  tap counts) and `GaussianKernel1D` (σ-derived tap count, symmetric, so its
  `flipped()` returns itself) both implement it, and every separable entry
  point is generic over it. This is what makes a σ-derived kernel a
  first-class argument:
  `convolve_separable(&src, &gaussian_kernel_1d(sigma, 3.0), &Clamp)`
  replaces the removed `gaussian_blur_with` (see *Changed*), and anything
  that needs a σ-shaped separable kernel — scale space, DoG — can now apply
  one directly instead of going through a named blur. Callers may implement
  the trait for their own kernel types; the contract is non-empty axes,
  in-bounds anchors, and a non-allocating `flipped()`.
- `fovea::draw`: drawing primitives that burn annotations into image
  pixels — the workflow behind self-contained inspection records and
  rejection-image archives, and the piece that lets detected keypoints and
  traced contours be *seen*. Five shapes as storable structs with one-shot
  free-function wrappers: `Line` / `draw_line` (Bresenham), `Rect` /
  `draw_rect` and `Circle` / `draw_circle` (midpoint), each outlined or
  filled, `Polyline` / `draw_polyline` (open chain or closed polygon —
  pass a traced contour's vertices), and `Crosshair` / `draw_crosshair`
  (the keypoint marker). All of them implement the new `draw::Drawable<P>`
  trait, the module's extension point: a user-defined marker implements
  `Drawable` and is drawable everywhere the built-ins are. Drawing
  positions are **signed** `(i32, i32)` — a shape centred near the image
  edge legitimately extends past it — and every primitive silently clips
  to the image bounds: no error, no panic. The only bound is `P: Copy`,
  so any pixel type can be drawn onto, and rendering is deliberately
  crisp (hard single-pixel strokes, no anti-aliasing or blending), which
  survives JPEG compression without smearing. Invalid geometry is
  unrepresentable rather than documented away: `Circle::radius` and
  `Crosshair::arm_length` are `u32`, and degenerate polylines (fewer than
  two points) are no-ops so partially built shapes can be handled safely.
  Text rendering and non-destructive display overlays are deferred.
- `analyze::statistics`: whole-image statistics, in two entry points that need
  no configuration and allocate nothing.
  `analyze::statistics::image_statistics` reports minimum, maximum, mean,
  variance and standard deviation per channel as `ChannelStatistics<C>`, with
  the output shape chosen by the caller's annotation exactly as `histogram`
  does (`ChannelStatistics<_>` for single-channel input,
  `[ChannelStatistics<_>; N]`, or `Vec<_>`). Three deliberate choices are
  visible in the signature. **Every accessor returns `Option`:** an image with
  no pixels, or a float channel every sample of which is `NaN`, has no mean and
  says so rather than returning `0.0`. **`min` and `max` come back in the
  channel's own type, not `f64`,** because they are selections rather than
  sums, so a `u64` channel value above `2^53` stays exact where a widened one
  would not. **Both variance conventions ship:** `variance` and `std_dev`
  divide by `n` (an image is the whole population, and this is what imaging
  libraries report), `sample_variance` and `sample_std_dev` divide by `n − 1`
  and are `None` below two samples. Mean and variance use Welford's
  recurrence rather than `Σx² / n − mean²`, which loses catastrophically on
  the ordinary industrial case of 16-bit data with a small spread about a large
  pedestal; accumulation is in `f64` whatever the input width, since a pixel's
  own `f32` accumulator cannot carry the running sum of a multi-megapixel
  frame. `NaN` samples are counted in `nan_count` and excluded from every
  statistic, so `count + nan_count` is the pixel count and a partly-invalid
  image is distinguishable from a clean one. Channels are admitted by the new
  sealed `StatisticsChannel` trait.
- `analyze::statistics::image_moments`: intensity-weighted image moments and
  the invariants derived from them, as a chain of named stages rather than one
  wide struct: `ImageMoments` (raw `m_pq` for `p + q ≤ 3`) →
  `CentralMoments` (translation-invariant, via `central_moments()`) →
  `NormalizedMoments` (translation- and scale-invariant, via `normalized()`) →
  `NormalizedMoments::hu()` (Hu's seven invariants, which add rotation
  invariance). Each step returns `Option` because each can genuinely not exist:
  an image with zero total intensity has no centroid to take moments about, and
  one with non-positive total intensity has no real scale normalisation.
  `CentralMoments` also exposes `orientation()` (an `AxialOrientation`, since an
  ellipse's major axis has no head or tail) and `eccentricity()`. Single-channel
  input is a compile-time bound rather than a documented precondition: "the
  centre of brightness of an RGB image" has no one meaning, so convert with a
  named strategy first. Unlike the summaries above, a `NaN` sample
  **propagates** to every moment: a moment is a sum over every pixel, so
  skipping one would report a figure for an image that was not measured.
  Note the convention difference from `BlobMeasurements::central_moments`,
  which divides by area; the `μ_pq` here are unnormalized sums, which is the
  standard definition and what the Hu invariants are built on.
- `analyze::peak`: peak interpolation, one quadratic fit shared by every
  operation in the crate that reports a position as a pixel index. A corner
  peak, a template-match score, an edge on a gradient ridge and a traced
  contour vertex are all quantized to the grid and so all carry up to half a
  pixel of error along each axis, systematically. Fitting a quadratic to the
  samples around the winner and reporting the fitted vertex removes it.
  `analyze::peak::interpolate_peak` is the 2-D fit over a 3×3 window,
  `interpolate_peak_along` the 1-D fit across a gradient ridge,
  `interpolate_ridge_points` that fit over a list of sites, and
  `parabola_vertex` the 1-D arithmetic on its own for three samples that did
  not come from an image. The 2-D fit **includes the xy cross term**: two
  independent 1-D fits are cheaper and are wrong for a peak whose iso-contours
  lie at an angle to the pixel axes, which is exactly what a corner response
  and a match score surface produce.
  `analyze::peak::Extremum` names which stationary point the caller expects,
  because `transform::SSD` and `SAD` are *minimized* at the best match while
  `NCC` and every corner response are maximized; naming it is what lets the
  fit refuse a surface that curves the other way instead of returning the
  wrong stationary point silently. Every entry point returns `Option`: a flat
  plateau has no unique vertex, a saddle is not a maximum, and a site on the
  image border has no neighbour on one side.
- `analyze::edge::interpolate_edge_points` and
  `features::detect::interpolate_corners`: the two named call sites of the
  above. `interpolate_edge_points` turns a `canny` mask plus the magnitude and
  gradient stages into a point list, one interpolated position per kept pixel.
  `interpolate_corners` takes `&mut [Corner]` and moves each corner to the
  interpolated peak of the response map, composing after the detector like
  `features::retain_top_n` does, and returns how many were interpolated. Two
  requirements
  neither type system can express, so both are documented and
  regression-tested: `interpolate_edge_points` needs the **unthinned**
  magnitude (suppression zeroes exactly the two neighbours each fit reads, so
  a thinned map leaves every point on its pixel centre, which is a plausible
  wrong answer), and `interpolate_corners` needs corners in the *response map's*
  frame, so it runs on `corner_peaks` output before any level→base lift.
  For contours, `interpolate_ridge_points` takes `Contour::points` as its
  sites and returns one entry per vertex **in vertex order**, `None` where the
  fit was refused rather than dropping it, because a contour is a sequence and
  removing a vertex splices two unrelated parts of the outline together.
- **Naming, and what interpolation does not do.** Nothing added here is called
  "sub-pixel" or "refinement". Interpolation removes the ≤0.5 px *grid
  quantization*; it leaves the surface's own *localization bias* untouched,
  because it locates the extremum of the surface it is handed. A
  structure-tensor response peak drifts inward from the corner as its window
  grows (measured and regression-tested since the detectors landed), and
  interpolating a displaced peak yields a precise displaced peak. Removing
  that needs a different computation over a different input, which is what
  `refine_corners` (below) does for corners; "refine" is reserved for that
  step. The distinction is stated in the `analyze::peak` module docs
  with both error magnitudes, because every surveyed library ships the two
  concepts under the single word "sub-pixel", which is how a position that is
  still a pixel off comes to be described as sub-pixel refined.
- `features::detect::refine_corners`: the corner **accuracy** step, the half
  of corner localization that interpolation cannot do. Per corner it solves
  the Förstner gradient-orthogonality least squares over a window of the
  caller's gradient images (`radius` of about 2σ for the tensor family, 3
  for the segment test): near a corner every pixel's gradient is
  perpendicular to the edge line through that pixel, so the corner is the
  least-squares intersection of the edge lines the window's gradients imply.
  It reads the gradient field and never the response map, which is what lets
  it remove both detector families' localization biases: the
  structure-tensor peak's window-induced inward drift (one full pixel per
  axis at σ = 1.6 on the regression fixture) and the segment test's
  raster-first plateau bias (two full pixels on the same fixture). Refined
  positions land within 0.05 px of the geometric corner for every window σ
  from 0.8 to 2.0, regression-tested; the remaining few hundredths are the
  gradient operator's own support mixing the two edges near the apex, the
  method's noise floor. Composes in place over `&mut [Corner]` like
  `interpolate_corners` and returns how many corners were refined; a corner
  whose fit is refused (incomplete window, rank-deficient gradients, a
  solution outside its own window) keeps its position. Returns `Result`:
  `gx` and `gy` are two separately produced images (`Error::SizeMismatch`),
  and a zero radius is rejected as `Error::InvalidParameter` because a
  single-pixel window would silently refuse every corner. Responses are left
  alone; the output type stays `Corner`, with the changed error model
  documented rather than encoded in a new type.
- `analyze::quality`: image quality metrics, the first operations in the crate
  that take **two** images and report how far apart they are.
  `squared_error(&a, &b)` makes one pass over both and returns a
  `SquaredError` holding a `ChannelSquaredError` per channel; every value
  metric is an accessor on it — `sum_squared_error`, `mean_squared_error`,
  `root_mean_squared_error`, `max_absolute_error` and
  `peak_signal_to_noise_ratio` — and `SquaredError::pooled()` returns the same
  record with the channels' accumulators summed. Per channel rather than
  pooled by default because "the MSE of a colour image" is three different
  numbers in the literature and a demosaic regression cares which; note that
  the pooled PSNR is the PSNR of the pooled MSE, the figure other libraries
  report, and is *not* the mean of the per-channel PSNRs. Accessors return
  `Option` (an empty pair, or a float pair whose every difference is `NaN`,
  has no mean error), a `NaN` difference is counted in `nan_count` and
  excluded, and an MSE of zero gives `f64::INFINITY` for the PSNR rather than
  an error, because that is what the definition says. Both images must be the
  same pixel type — an error between `Mono8` and `MonoF32` needs a range
  convention this crate deliberately does not have — while the two *image*
  types stay independent, so a region of view compares against an owned
  reference. `max_absolute_error` ships beyond the three planned metrics
  because it is the assertion a regression test actually wants: a small mean
  hides one catastrophic pixel.
- `analyze::quality::PeakValue`: the full-scale value (`L`) PSNR divides by and
  SSIM's `C1` / `C2` are fractions of, as an invariant-carrying parameter type
  alongside `Sigma`, `PixelDistance` and `Tolerance`. `PeakValue::of_pixel::<P>()`
  reads the pixel type's own `WhiteChannel`, so `Mono<10>` reports **1023, not
  65535** — a PSNR built on the channel type's storage maximum is 36 dB
  optimistic on 10-bit sensor data. Float-channel pixels do not implement
  `WhiteChannel` and therefore **fail to compile** through that constructor
  rather than silently assuming `1.0`; a float caller writes
  `peak!(1.0)` and owns the assumption. The `peak!` macro for literals,
  `try_new` for computed values, and a `const fn new` returning
  `Option<Self>` underneath. The rustdoc states what it is not: the range
  of the representation, not the largest value the data happens to contain.
- `analyze::quality::ssim` / `ssim_map`: structural similarity, one score or
  the per-position map it averages. `SsimParams::reference(peak)` builds the
  published parameters of Wang et al. (2004) — an 11-tap Gaussian window at
  σ = 1.5, `K1 = 0.01`, `K2 = 0.03` — so a score is comparable with MATLAB's
  `ssim`, OpenCV's sample and scikit-image's `gaussian_weights=True` path;
  `SsimParams::TRUNCATE` is pinned at 3.0 rather than the crate's blur default
  of 4.0 precisely to derive 11 taps from σ = 1.5 instead of 13. Parameters
  are a value, not a `_with` suffix. Implemented as five separable Gaussian
  convolutions with the `Skip` border, **not** on the integral image: a
  summed-area table has no product accumulator for the covariance term and
  computes a uniform window, which is not the window any published figure
  uses. Each image is globally mean-centred before the moments are formed,
  which is load-bearing rather than tidy — the separable engine's kernel taps
  are `f32`, so an uncentred moment on a narrow signal riding a large pedestal
  carries enough absolute error to drive the window variance negative; on the
  regression fixture the uncentred score is 0.703 where the correct answer is
  0.967, and with the peak named for the signal it reaches −30, outside SSIM's
  range. Single-channel by a compile-time bound, because `C1` and `C2` are
  fractions of one dynamic range and colour SSIM has no settled definition:
  convert with `Luminance` and name the choice. The map is the
  `(w − 2r) × (h − 2r)` block where the window lies wholly inside the frame,
  with map `(x, y)` reporting input `(x + r, y + r)`, the convention
  `match_template` already uses; an image the window does not fit is
  `Error::InvalidParameter` rather than a clipped window, since a value
  extrapolated past the edge is not a measurement. `NaN` propagates here,
  where `squared_error` excludes it, because a window statistic has no
  per-position count to record an exclusion in.
- Literal macros for the invariant-carrying parameter types: `sigma!`,
  `pixel_distance!`, `tolerance!`, `window!`, `peak!` and `harris!`. Each
  expands to an inline `const { }` block around the type's checked
  constructor, so an invalid literal is a **compile error** wherever it is
  written, not only inside a `const` item:

  ```rust
  let blurred: Image<MonoF32> = gaussian_blur(&img, sigma!(1.4), &Clamp);
  const WINDOW: OddWindowSide = window!(31);
  // error[E0080]: evaluation panicked: sigma must be finite and
  //               strictly positive
  let bad = sigma!(-1.4);
  ```

  A value that is not a constant expression does not compile through a
  macro (`error[E0435]`); that is what `try_new` is for, and it reports a
  reason the caller can act on. The macros are the reason no parameter type
  needs a panicking constructor: the compile-time guarantee that a `const fn`
  only *sometimes* delivers is unconditional here.
- `Offset`: the grid-displacement vocabulary type, `Offset { dx: i32, dy:
  i32 }` in `common` and re-exported at the crate root beside `Coordinate`.
  One named type for "a signed step on the pixel grid" — a connectivity
  neighbourhood, a detector's sampling ring, a chain-code direction, a
  filter tap — which the crate previously spelled as bare pairs in four
  different widths. It carries **no invariant** (every `(dx, dy)` is a
  meaningful step, so there is no `try_new` and no validation); what it
  carries is the field names, which is what makes a transposed step fail to
  compile instead of silently answering a different question. Deliberately
  not an arithmetic type, and deliberately **without** `From<(i32, i32)>`:
  the pair is the shape the transposition slips through, pinned by a
  `compile_fail` doctest.
- `Coordinate::checked_add(Offset) -> Option<Coordinate>` and
  `Coordinate::offset_to(Coordinate) -> Offset`. `checked_add` is the
  crate's one neighbourhood bounds check, replacing four open-coded
  versions that each handled the negative half differently (a widening cast
  to `i64` and a four-way range test in the labeling engine, an
  off-view-reports-label-0 helper in the contour tracer, a `debug_assert!`
  in the chain decoder, and a pair of `checked_add_signed` calls in the
  peak fitter). It covers the negative half only, because the far
  edge already has an answer: `ImageView::get` returns `Option`, so
  `c.checked_add(off).and_then(|p| img.get(p.x, p.y))` is the whole check,
  composed from two operations that each say what they mean.
  `offset_to` is the inverse, saturating rather than wrapping for
  separations beyond `i32`.

### Changed

- **Breaking:** `BlobMeasurements::perimeter` is renamed to
  **`boundary_pixels`**. The value is unchanged — the count of 4-connected
  boundary pixels — but a pixel *count* is not a geometric length, and a
  field named `perimeter` invited using it as one (the documented
  `circularity ≈ 1.25` artifact is that misuse, baked in). The honest
  perimeter is now available where a length is meant:
  `Contour::perimeter` from `analyze::contours` (see *Added*).
  `BlobMeasurements::circularity()` keeps its cheap single-pass semantics
  and its documented bias, and now points at `Contour::circularity` for
  the geometric score.
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
- **Breaking:** `analyze::threshold::hysteresis_threshold` (+ its `_into`
  variant) and `analyze::edge::canny` take one `HysteresisThresholds`
  argument instead of two bare `low` / `high` values (see *Added*). Build
  the pair with `HysteresisThresholds::try_new(low, high)?`, whether the two
  numbers are literals or computed. Both
  functions are now **total in their thresholds** and their
  `!(low <= high)` panic is gone; `canny`'s only remaining panic is
  `gaussian_blur`'s `MAX_RADIUS` capacity bound. `canny` still takes its
  pair in `f32` and widens it to the accumulator channel, which is `f32` or
  `f64` and nothing else, since `MagnitudeChannel` is sealed over exactly
  those two, so the widening is order-preserving and the pair is re-typed
  rather than re-validated.
- **Breaking:** `analyze::threshold::adaptive_threshold` (+ its `_into`
  variant) takes `OddWindowSide` instead of `window: usize` (see *Added*).
  Write `window!(31)` for a literal, `OddWindowSide::try_new(side)?` for
  a side computed from data. The value's meaning is unchanged: it is still
  the window's side length, not its radius, matching OpenCV's `blockSize`.
  The "window must be odd and non-zero" panic is gone from both
  functions.
- **Breaking:** `gaussian_blur` (+ its `_into` variant),
  `gaussian_kernel_1d` / `gaussian_kernel_size`, and `canny`
  take the new `Sigma` parameter type instead of a raw `f32` σ (see
  *Added*). Write `sigma!(…)` for literals; validate computed values
  with `Sigma::try_new(…)?` where they are produced.
- **Breaking:** `gaussian_blur_with` and `gaussian_blur_with_into` are
  **removed**. A `truncate` other than the default is not a variant of
  "blur" — it is a different kernel, and kernels are values here. Build one
  with `gaussian_kernel_1d(sigma, truncate)` and hand it to
  `convolve_separable`, which now accepts any `SeparableWeights` value:

  ```rust
  // before
  let out: Image<MonoF32> = gaussian_blur_with(&src, sigma, 3.0, &Clamp);
  gaussian_blur_with_into(&src, sigma, 3.0, &Clamp, &mut dst);

  // after
  let kernel = gaussian_kernel_1d(sigma, 3.0);
  let out: Image<MonoF32> = convolve_separable(&src, &kernel, &Clamp);
  convolve_separable_into(&src, &kernel, &Clamp, &mut dst);
  ```

  Results are identical — both paths correlate the same normalized taps,
  pinned by `gaussian_blur_equals_convolve_with_its_own_kernel`.
  `gaussian_blur` and `gaussian_blur_into` are **unchanged**: σ with the
  default `truncate` stays a one-call operation.

  Why it existed at all: a σ-derived kernel had no way into the separable
  engine, so the only place left for the parameter was the function name —
  where `_with` already meant something else in this crate
  (`connected_components_with_stats` returns extra data;
  `debug_histogram_with` in `fovea-display` takes an options struct). The
  rule going forward is the one the rest of the crate already follows:
  **a variant is a value**, as with `resize` + `Bilinear`, `demosaic` +
  `MalvarHeCutler`, `convert_image` + `Luminance`, `combine_images` +
  `AbsDiff`.
- **Breaking:** `convolve_separable` and `convolve_separable_into` are now
  generic over `image::SeparableWeights` instead of taking
  `&SeparableKernel<HK, VK>` concretely. Ordinary calls are unaffected —
  `SeparableKernel` implements the trait — but code that spelled the const
  generic parameters explicitly through a turbofish, or stored a function
  pointer to either function, must drop the turbofish or re-infer the type.
- **Breaking:** the label-index vocabulary narrows from `u64` to `u32`:
  `LabelPixel::MAX_LABEL`, `LabelPixel::from_label_index` /
  `to_label_index`, `Labeling::label_count`, the count
  `connected_components_into` returns,
  `Error::LabelOverflow::label_capacity`, and the label
  `ContourHierarchy::component_for_label` takes. The engine's
  provisional-label and compaction buffers (and its union-find) move to
  `u32` with them, halving the width of the working set that both
  full-image labeling passes read and write, the passes hysteresis
  thresholding and blob measurements ride on. This is a buffer-width
  result, not a benchmark: no wall-clock claim is made. The narrowing is
  also what makes the types honest about capacity: those buffers are what
  bounds a labeling pass, so a `u64` index advertised a range no pass
  could deliver, and a hypothetical `Label64` could never have carried a
  label that `Label32` cannot. Labels are bounded by the image's pixel
  count, so nothing reachable changes below a 4-gigapixel input; an image
  that would need the `(u32::MAX + 1)`-th provisional label now returns
  `Error::LabelOverflow` when the engine's label space is spent, instead
  of a capacity the types promised and the buffers did not have.
- **Breaking (for implementors):** `MatchMethod` gains the non-generic
  supertrait `transform::ScorePolarity`, whose associated const
  `EXTREMUM` states where the method's score map marks the best match:
  `SAD` and `SSD` carry `Extremum::Minimum`, `NCC` carries
  `Extremum::Maximum`. The peak fit is now asked for the method's own
  polarity, `interpolate_peak(&scores, best, SSD::EXTREMUM)`, instead of
  a convention the caller has to remember and can invert without any
  diagnostic. Callers of `match_template` are unaffected; an external
  `MatchMethod` implementor must add the one-line `ScorePolarity` impl.
  The const lives on a supertrait rather than on `MatchMethod` itself
  because an associated const on a trait with three type parameters
  cannot be read without naming all three: `SSD::EXTREMUM` compiles only
  from a non-generic trait.

- **Breaking:** no invariant-carrying parameter type has a panicking
  constructor any more. `new` on `Sigma`, `PixelDistance`, `Tolerance`,
  `OddWindowSide`, `PeakValue`, `Harris`, `SegmentTest`, `FastParams`,
  `CornerParams` and `BayerGains` returns **`Option<Self>`** instead of
  `Self`, and `Clamp::new` is **removed** in favour of `Clamp::try_new`.
  The old shape rested on a mistaken premise: a `const fn` whose body
  `assert!`s is a compile error only when the compiler happens to evaluate
  it at compile time, and a runtime abort everywhere else, with nothing at
  the call site to say which applies. `Sigma::new(detail_estimate(&image))`
  compiled and aborted on a flat frame, which is the per-function panic the
  parameter types exist to remove, relocated one call earlier rather than
  eliminated.

  The compile-time guarantee moved to the literal macros (see *Added*),
  which deliver it unconditionally. Migration is mechanical:

  ```rust
  // before                              // after
  Sigma::new(1.4)                         sigma!(1.4)
  OddWindowSide::new(31)                  window!(31)
  Harris::new(0.04)                       harris!(0.04)
  Sigma::new(computed)                    Sigma::try_new(computed)?
  Clamp::new(lo, hi)                      Clamp::try_new(lo, hi)?
  SegmentTest::new(0.08, 9)               SegmentTest::new(0.08, 9).unwrap()
  ```

  The four composites (`SegmentTest`, `FastParams`, `CornerParams`,
  `BayerGains`) get no macro, because their argument *names* are the
  information and `harris!(1.4, 0.01, 3)` reads worse than the named
  constructor; bind them to a `const` item and `.unwrap()` is checked at
  compile time as before. `HysteresisThresholds` and `Clamp` get neither a
  macro nor an `Option` constructor: both compare through a trait method
  (`PartialOrd`, `Ord`), which a `const fn` cannot call on stable Rust, so
  there is no compile-time tier for a second constructor to preserve and
  `try_new` alone carries them.
- **Breaking:** the four public places that spelled a grid displacement as
  a bare pair now take or return the new `Offset` (see *Added*):
  `analyze::components::Connectivity::OFFSETS` is `&'static [Offset]` (was
  `&'static [(i32, i32)]`), `features::detect::FAST_RING` is `[Offset; 16]`
  (was `[(isize, isize); 16]`), `analyze::contours::ChainDirection::offset`
  returns `Offset` and `ChainDirection::from_offset` takes one argument
  instead of two, and `transform::DemosaicMethod::interpolate` takes
  `at: Coordinate` with `S: Fn(Offset) -> f32` (was `x: usize, y: usize`
  with `S: Fn(isize, isize) -> f32`).

  Three widths for one concept was the symptom; the unshared bounds check
  was the problem, and it is now `Coordinate::checked_add`. The
  `DemosaicMethod` half is where the type-level argument bites hardest:
  `MalvarHeCutler`'s row and column kernels are each other's transpose, so
  a tap written `(dy, dx)` is a *different* kernel that still compiles, and
  until now only a numeric round-trip test stood between that and a wrong
  colour. Migration is mechanical — `(dx, dy)` becomes `Offset::new(dx,
  dy)`, `(0, 0)` becomes `Offset::ZERO`, and a destructuring `let (dx, dy)
  = …` becomes field access.
- **Breaking:** the four structure-tensor detector entry points take their
  response strategy by value: `StructureTensor::response(&self, method: M)`,
  `corner_response_map(&image, method, window)`, `detect_corners` and
  `detect_corners_in_level` likewise (was `&M` everywhere). The demosaic,
  template-matching, resize and image-combining engines already took their
  strategies by value, so the detect family was the one place a caller had
  to borrow a `Copy`-sized marker, and the inconsistency was between
  shipped siblings. Migration: delete the `&`.
- **Breaking:** the non-maximum-suppression radius is a validated type,
  `features::detect::NmsRadius` (at least 1, with `const fn new -> Option`,
  `try_new -> Result` and `get`), replacing three hand-written "must be at
  least 1" validations with three message phrasings. `CornerParams` and
  `FastParams` carry it as a field and their `nms_radius()` accessors
  return it; `refine_corners` takes it for the fitting window, whose
  at-least-one-pixel invariant is the same one (a single-pixel window has
  a rank-one normal matrix, so every fit would be refused), and is
  therefore total in its radius. `FastParams::new` is total now that both
  fields carry their own invariants, and `FastParams::try_new` is removed.
  `corner_peaks` deliberately keeps its raw `usize`: it is the permissive
  primitive, where radius 0 degenerates to "every pixel above the
  threshold". Migration: `NmsRadius::new(3).unwrap()` for a literal
  (checkable in a `const` item), `NmsRadius::try_new(r)?` for a computed
  radius.
- **Breaking** for downstream implementors: `StatisticsOutput<C>` gained
  the pixel type as a parameter, `StatisticsOutput<C, P>`, and the
  single-record `ChannelStatistics<C>` shape is implemented only for
  `P: SingleChannel`. Binding a single record on a colour image is now a
  compile error carrying the former panic message's advice, instead of a
  run-time panic. Call sites of `image_statistics` are unchanged unless
  they relied on the panic, which no longer compiles; the array shape's
  length check stays at run time.
- **Breaking:** `Depalettize::from_slice` is replaced by
  `Depalettize::try_from_slice -> Result`. A partial palette routinely
  arrives from a decoded file, so its length is data, and data failures
  are errors, not panics; this was the one value-certifying constructor
  in the crate that still aborted. The full-array
  `Depalettize::new([P; 256])` is unchanged and total.
- **Breaking** for implementors: `PyramidMethod::build` accepts any
  `RasterImage` view (`fn build<I: RasterImage<Pixel = P>>(&self, image:
  &I, ...)`) instead of binding concrete `&Image<P>`, so a pyramid can be
  built straight from a borrowed buffer or ROI. Call sites are unaffected;
  an owned `Image<P>` still satisfies the bound. Custom implementations add
  the generic parameter.
- **Breaking** in name only: `AdaptiveAccumulator::Offset` is renamed
  `BiasValue`. The trait is sealed, so no downstream implementations exist;
  the associated type was nameable, hence the label. The crate reserves
  `Offset` for grid displacements, and this was the one surface where the
  same name meant a signed brightness bias.
- A `SeparableWeights` implementation returning an empty weight slice or
  an out-of-bounds anchor now fails at the engine's boundary with a
  message naming the trait and the offending method, instead of
  underflowing the interior-region arithmetic several frames deeper in a
  panic that named neither. Correct implementations are unaffected; the
  checks run once per call, not per pixel.

### Fixed

- `canny`'s documentation listed the wrong accumulator for `Mono16`: it
  claimed `Mono16` widens to `MonoF64` like `Mono32` and `Mono64`, when in
  fact `Mono16` — and `Mono<BITS>` — accumulate in `MonoF32`. Documentation
  only; the bound the compiler enforces never changed, so no behaviour or
  signature moves with this. Corrected while writing the same sentence for
  the corner detectors, which do widen `Mono32` and up.
- `connected_components` (and through it `_with_stats`,
  `_with_measurements` and `extract_contours`) reported `LabelOverflow`
  against the engine's *provisional* label count, so an ordinary image
  whose components merge through many provisional labels could fail to
  label into a narrow `LabelPixel` type: a one-component 7x2 comb-over-bar
  image overflowed a `MAX_LABEL = 3` label type. The capacity check now
  runs in pass 2 against the final compacted count, matching the
  documented contract; exhausting the engine's own `u32` provisional
  space, which takes more than `u32::MAX` disconnected foreground sites,
  is reported with `label_capacity == u32::MAX`. On an `Err`,
  `connected_components_into` may leave a prefix of the output buffer
  written, now documented.
- `parabola_vertex` and `interpolate_peak` returned `Some(NaN)` for an
  infinite sample, outside their documented ranges: infinite curvature
  passes the positive definiteness guards and the vertex quotient of two
  infinities is NaN, which the negatively-written containment guard let
  through, so `interpolate_corners` then counted a NaN position as a
  successful refinement. Both guards are rewritten in the positive, and an
  infinite sample is refused exactly like a NaN one.
- The line and circle walks are clipped to the frame. Painted pixels were
  always correct, but the walks stepped through the *ideal* shape, so
  `draw_line(&mut img, (i32::MIN, 0), (i32::MAX, 0), color)` cost roughly
  twenty seconds to paint four pixels and a `u32::MAX` circle radius ran
  the octant walk about three billion times. Both walks now fast-forward
  their exact internal state past invisible iterations, so the cost
  follows the visible portion; the painted output is unchanged, pinned
  pixel-for-pixel against the unclipped walks by exhaustive sweeps.
- `Orientation::from_atan2` and `AxialOrientation::from_half_atan2` could
  return exactly `-π` (respectively `-π/2`), outside their documented
  canonical ranges: `atan2` returns the closed `[-π, π]`, and a
  negative-zero `y` beside a negative `x` lands on the bottom endpoint.
  That single boundary value now folds to the top of the range, the same
  angle spelled canonically.
- `pyr_up` doubled the brightness of a length-1 axis: the doubled-weight
  interpolation kernel compensates for zero-inserted samples, and a
  dimension-1 source upsampled to a dimension-1 target (the only case where
  an axis stays its own length) has none, so a flat 1xN bar came back at
  twice its value. Such an axis now takes the normalized weights, and a
  flat field stays flat for every valid source/target pair. Found by the
  degenerate-size fixtures the v0.4.0 review asked for.
- `pyr_up`'s target validation could overflow and abort in debug builds: a
  zero-area image can legally carry a dimension past `usize::MAX / 2`, and
  any `Size` can be named as the target. The validation now uses checked
  arithmetic and rejects such targets as invalid.
- `eccentricity` (on `CentralMoments` and `BlobMeasurements`) could exceed
  its documented `[0, 1]` for a near-collinear shape at a large coordinate
  offset, where cancellation pushes the smaller eigenvalue slightly below
  zero (1.000 000 19 measured at x = 65 535). The guarding clamp is now
  two-sided.
- `Contour::solidity` returned `Some(0.0)`, outside its documented
  `(0, 1]`, for a bent one-pixel out-and-back trace: the trace encloses no
  area while its point set's hull does, so the degenerate-hull guard did
  not fire. Zero enclosed area now yields `None`, symmetric with
  `centroid`. `circularity` is unchanged: its `Some(0.0)` for a line-like
  chain is a meaningful zero-roundness score over a nonzero perimeter.
- `Rect` painted a span to the *left* of its anchor for a width or height
  past `i64::MAX`: the plain cast wrapped (`usize::MAX as i64` is `-1`)
  and `hspan` accepts either argument order, so the output was silently
  wrong rather than clipped. The extents now saturate, and a pathological
  `Size` clips like any other off-image extent.
- `ssim` / `ssim_map` clamp the windowed covariance into the
  Cauchy-Schwarz bound over the (already clamped) variances. Without it,
  an image compared against itself scored a hair off exactly 1.0 wherever
  its raw windowed variance rounded a step below zero, contradicting the
  documented identity; and a residual covariance over clamped-to-zero
  variances could push a map value marginally past 1.0. Epsilon-scale on
  real data, and NaN samples still propagate.
- The detect module documentation claimed "no parameter moves a detection"
  for the segment test. True for the threshold, which only filters; false
  for the arc length, which changes the score map itself, so detections
  can move or vanish between arc lengths (the suite's own right-angle
  fixture vanishes outright between 9 and 12). The claim is narrowed to
  the threshold and pinned by a mixed-contrast filtering test.

### Performance

- `fast_score_map` splits each scan line into a hot span whose whole ring is
  guaranteed inside the frame and cold strips that keep the border-policy path,
  the same interior/boundary split `fold_neighborhood` already uses. The hot
  span reads the seven scan lines the ring spans once per line instead of once
  per sample, through a const table that rewrites each ring offset as
  `(row, dx)`, and pays no per-sample bounds test. Under `Skip`, the documented
  default, the cold strips are empty and the border policy is never consulted.
  Measured **1.11× / 1.15× / 1.18×** at arc length 9 / 12 / 16 on a 512×512
  `Mono8` texture. Behaviour is unchanged and pinned as such: a test compares
  the map against the per-pixel `fast_score_at`, which always takes the general
  path, over every pixel under both `Clamp` and `Skip`. Also folded in: the
  twelve non-cardinal ring positions are a constant rather than a linear search
  per pixel.
- Documented in `segment_score`, for the reader who has the same idea: the
  `O(16 · arc_length)` arc scan was replaced with an `O(16 + arc_length)`
  sliding-window minimum, measured, and reverted. Sixteen ring positions are too
  few for the asymptotics to pay for the scratch arrays they need, and the
  replacement was ≈6 % *slower* at arc length 9, which is the variant ORB uses.

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

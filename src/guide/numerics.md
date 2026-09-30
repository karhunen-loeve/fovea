# Numerics

This page says what fovea promises about floating-point results: which bits
you can rely on, how accuracy is documented, and how an operation chooses
between equal candidates.

## Bit patterns

The last bits of a floating-point result can differ between builds of the
same code. They depend on the target architecture, on the target features the
build enables, on the compiler version and, once fovea uses them, on the
number of threads and the SIMD width. fovea promises no bit pattern across any
of these.

FMA is the case that exists today. With `-C target-feature=+fma`, which
`-C target-cpu=native` turns on for most current x86_64 processors,
convolution, resizing and sampling compute a multiply and an add with one
rounding; without it they round twice. Both results are correct, and they can differ in the
last bit.

If you need identical bits, for a validated inspection system or a comparison
against golden images, validate the binary you ship and keep it. A rebuild
with another compiler, other target features or another fovea version is a
new configuration to validate.

## Accuracy

Where the accuracy of an operation depends on a choice it makes, its
documentation names the choice and what follows from it. Two examples:
template matching accumulates in `f64`, so a large template loses no
precision, and the image statistics compute the variance with Welford's
recurrence, which avoids the cancellation of `E[x²] − E[x]²` on data with a
large offset. Such a choice holds for every input the operation accepts, not
only for typical image sizes.

This describes the behaviour. fovea guarantees no numeric precision, and a
figure in the documentation, where there is one, is information for you. If
an operation switches to a more accurate method, its results change and the
changelog says so. Switching to a less accurate one is a breaking change.

## Ties

When an operation has to choose between equal candidates, it follows a stated
rule, and the rule is part of the API. It holds on every build, and a faster
implementation keeps it, because it constrains the order of a choice and not
the arithmetic.

Two defaults hold unless an operation documents otherwise:

- **The earlier candidate wins.** Earlier means earlier in raster order (rows
  top to bottom, each row left to right), or earlier in the sequence the
  operation walks, such as the points of a contour.
- **NaN never wins.** A NaN candidate loses the comparison, so it is never
  selected as a maximum, a peak or a threshold.

Where the rules are applied:

| Operation | Rule |
|---|---|
| [`corner_peaks`](crate::features::detect::corner_peaks) | a plateau of equal responses yields its raster-first pixel; a NaN suppresses rather than wins |
| [`sort_by_response`](crate::features::sort_by_response), [`retain_top_n`](crate::features::retain_top_n) | strongest response first and a NaN response last; equal responses by position, smaller `y` first, then smaller `x` |
| [`connected_components`](crate::analyze::components::connected_components) | components are numbered in raster order of their first pixel |
| [`otsu_threshold`](crate::analyze::histogram::otsu_threshold) | the lowest of several equally good thresholds |
| [`approximate_polygon`](crate::analyze::contours::approximate_polygon) | the earliest of several vertices equally far from the first |

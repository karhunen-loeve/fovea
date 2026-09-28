# Scales of measurement

A pixel type says how the bytes are laid out and what they mean. This page is
about the second half: which operations a value admits. Two images with the
same layout can admit very different operations. A `Mono8` intensity and a
`Label32` component ID are both one unsigned integer per pixel, yet averaging
two intensities is meaningful and averaging two IDs is not.

The systematic answer comes from measurement theory. S. S. Stevens ("On the
Theory of Scales of Measurement", *Science* 103, 1946) defined the level of a
scale by the transformations that leave its meaning intact. What survives
every such transformation is meaningful; what does not is an artefact of an
arbitrary choice.

This page describes three families of scales and how fovea's types map onto
them.

## 1. Linear scales

For a single real quantity, Stevens' levels form a chain. Each level admits
everything the one before it admits.

| Level | Admissible transformations | Meaningful operations | Examples in fovea |
|---|---|---|---|
| Nominal | any renaming (a bijection) | equal or not equal, counting | `Label32`, `Indexed8` |
| Ordinal | any strictly increasing function | order: minimum, maximum, thresholds, median | |
| Interval | `x → a·x + b`, with `a > 0` | differences, means | gamma-encoded values such as `Srgb8`, raw sensor samples before the black level is subtracted |
| Ratio | `x → a·x`, with `a > 0` | ratios and scaling as well | linear intensities: `Mono8`, `MonoF32`, `Rgb8`, `RgbF32` |

Two consequences run through the crate:

- **Linearity is its own question.** A gamma-encoded `Srgb8` channel is an
  ordered quantity whose differences mean something in encoded space, but its
  mean is not the mean of the light. That is why `Srgb8` does not implement
  `LinearSpace` and interpolation on it does not compile. The level of a
  channel and the linearity of its encoding are separate properties.
- **An ID is nominal.** A component label or a palette index can be compared
  for equality and counted, and nothing else. `Label32` and `Indexed8`
  implement none of the arithmetic traits for this reason. Counting pixels per
  label is what `connected_components_with_stats` reports.

Per-channel operations, such as a channel-wise maximum or a gradient
magnitude, apply a scalar operation to each channel on its own. They are
meaningful only when each channel on its own is a quantity on one of these
levels. **Current limitation:** these operations require `HomogeneousPixel`,
which guarantees the layout (every channel has the same type) and does not
yet check the level. The channel-wise maximum of two `Label32` images
therefore compiles, although it has no meaning.

## 2. Circular scales

Angles and directions are not on a line. The difference between 179° and
−179° is 2°, not 358°, and there is no smallest angle. Stevens' chain does not
cover them.

fovea has two types for them, and they differ in their modulus:

- `Orientation`: a direction, modulo 2π. A gradient direction distinguishes
  a direction from its opposite.
- `AxialOrientation`: an axis, modulo π. The orientation of a line or of a
  stripe pattern does not.

Both take differences on the circle through `signed_difference`, and neither
implements an ordering.

```rust
use fovea::{AxialOrientation, Orientation};

// Directions: 179° and -179° are 2° apart.
let east = Orientation::from_radians(179_f32.to_radians())?;
let west = Orientation::from_radians((-179_f32).to_radians())?;
assert!((east.signed_difference(west).abs().to_degrees() - 2.0).abs() < 1e-3);

// Axes: 10° and 190° are the same axis.
let a = AxialOrientation::from_radians(10_f64.to_radians())?;
let b = AxialOrientation::from_radians(190_f64.to_radians())?;
assert!(a.signed_difference(b).abs() < 1e-9);
# Ok::<(), fovea::Error>(())
```

## 3. Vector quantities

Some values have several components that only mean something together. The
standard example is a complex amplitude, such as a bin of a Fourier spectrum.
Its real and imaginary parts are coordinates in a basis, but the basis has no
physical meaning: rotating every value by the same phase changes nothing that
can be observed. The admissible transformations are `z → c·z` for any
non-zero complex `c`, which scales and rotates at once.

What survives them is meaningful:

- sums, differences and means, and every linear operation,
- the magnitude `|z|`, which is an ordinary ratio-scale quantity,
- quotients `z₁ / z₂`, which carry a magnitude ratio and a phase difference
  (phase correlation relies on exactly this).

What does not survive is not:

- order: no ordering of complex numbers is compatible with rotation,
- a single component on its own, such as the real part,
- an absolute phase, and anything computed per component nonlinearly.

Complex values therefore sit outside Stevens' chain. They behave like a ratio
scale without being ordinal, which the one-dimensional chain assumes cannot
happen. The principle behind the chain still applies: a value's meaningful
operations are the ones that are invariant under its admissible
transformations. Derived quantities fall back into the families above: the
magnitude is a ratio scale, the phase is a circular scale.

## Summary

| Family | Defined by | Examples |
|---|---|---|
| Linear | Stevens' chain: nominal, ordinal, interval, ratio | intensities, encoded values, IDs |
| Circular | a modulus: 2π for directions, π for axes | `Orientation`, `AxialOrientation` |
| Vector | the group of admissible transformations | complex amplitudes |

When an operation looks questionable on some pixel type, the useful question
is which family and level the value belongs to, and whether the operation
survives that level's transformations.

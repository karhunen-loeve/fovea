use crate::common::Size;
use core::fmt;

/// Errors returned by fallible image operations.
///
/// This type represents data-dependent failures — situations where the
/// operation is well-formed but the supplied data doesn't meet the
/// requirements. The crate uses a three-tier error handling convention:
/// `Option` for absence, `Result<T, Error>` for data-dependent failure,
/// and `panic!` for programmer bugs.
///
/// # Tier summary
///
/// | Tier | Type | When |
/// |------|------|------|
/// | 1 | `Option` | Absence — query found nothing (e.g. `get()` out of bounds) |
/// | 2 | `Result<T, Error>` | Data failure — caller-supplied data doesn't fit |
/// | 3 | `panic!` | Programmer bug — violated precondition (e.g. output size mismatch) |
///
/// # Examples
///
/// ```
/// use fovea::Error;
/// use fovea::Size;
///
/// let err = Error::LengthMismatch { expected: 100, actual: 50 };
/// assert_eq!(
///     err.to_string(),
///     "length mismatch: expected 100 elements, got 50"
/// );
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Two images that must have identical dimensions do not.
    ///
    /// Returned by [`combine_images`](crate::transform::combine_images),
    /// [`zip_pixels`](crate::image::zip_pixels), and similar functions
    /// that operate on image pairs.
    SizeMismatch {
        /// The dimensions of the first / reference image.
        expected: Size,
        /// The dimensions of the second image that does not match.
        actual: Size,
    },

    /// A data buffer's element count does not match the required
    /// dimensions.
    ///
    /// Returned by [`Image::from_vec`](crate::image::sequential::Image::from_vec),
    /// [`ImageRef::new`](crate::image::sequential::ImageRef::new), and similar
    /// constructors where `data.len() != width * height`, and by
    /// [`ImageRef::from_strided`](crate::image::sequential::ImageRef::from_strided)
    /// and its mutable twin where `data` is shorter than the view needs.
    LengthMismatch {
        /// The number of elements required (`width * height`, or
        /// `width * height * pixel_size` for byte constructors). For a
        /// strided view, the least length it needs,
        /// `(height - 1) * row_stride + width`.
        expected: usize,
        /// The number of elements actually provided.
        actual: usize,
    },

    /// The number of image planes does not match the pixel type's
    /// channel count.
    ///
    /// Returned by [`ImagePlanes::try_from_planes`](crate::image::ImagePlanes::try_from_planes).
    ChannelCountMismatch {
        /// The channel count required by the pixel type.
        expected: usize,
        /// The number of planes actually provided.
        actual: usize,
    },

    /// The requested `pyr_up` target is not a size whose `pyr_down`
    /// result is the source image's size.
    ///
    /// Returned by [`pyr_up`](crate::transform::pyr_up) when
    /// `target.width ∉ {2·w − 1, 2·w}` or
    /// `target.height ∉ {2·h − 1, 2·h}` for a `w`×`h` source image.
    /// Because `pyr_down` uses ceiling division, both the odd and the
    /// even parent dimension are valid targets — anything else cannot
    /// be the parent of this image.
    InvalidPyrUpTarget {
        /// The dimensions of the source image being upsampled.
        source: Size,
        /// The rejected target dimensions.
        target: Size,
    },

    /// A pyramid was constructed from an empty level list.
    ///
    /// Returned by
    /// [`LevelChain::try_from_levels`](crate::image::LevelChain::try_from_levels) —
    /// a pyramid always contains at least one level.
    EmptyPyramid,

    /// Pyramid levels are not ordered finest to coarsest.
    ///
    /// Returned by
    /// [`LevelChain::try_from_levels`](crate::image::LevelChain::try_from_levels)
    /// when a level is larger than its predecessor along either axis.
    /// Levels must be non-increasing in both width and height (equal sizes
    /// are allowed — same-size levels occur in scale stacks and sub-band
    /// decompositions). Levels are never reordered automatically: a wrong
    /// order is reported, not silently normalized.
    PyramidLevelOrder {
        /// Index of the first level that violates the ordering.
        index: usize,
        /// The dimensions of the preceding level.
        previous: Size,
        /// The dimensions of the offending level.
        current: Size,
    },

    /// Pyramid levels do not halve from one level to the next.
    ///
    /// Returned by [`Dyadic::try_new`](crate::image::Dyadic::try_new) when a
    /// level's size is not its predecessor's ceiling-halved size along both
    /// axes, which is the relation [`pyr_down`](crate::transform::pyr_down)
    /// produces. A chain may be perfectly well ordered and still not be
    /// dyadic: equal-size neighbours and a chain that shrinks by some other
    /// factor both pass
    /// [`LevelChain::try_from_levels`](crate::image::LevelChain::try_from_levels)
    /// on purpose. Because the relation holds between two runtime sizes,
    /// this is a recoverable error, not a panic.
    NotDyadic {
        /// Index of the first level that does not halve its predecessor.
        index: usize,
        /// The dimensions of the preceding level.
        parent: Size,
        /// The dimensions of the offending level.
        child: Size,
    },

    /// A value violates a parameter's invariant.
    ///
    /// Returned by the `try_new` constructors of the invariant-carrying
    /// parameter types — [`Sigma`](crate::Sigma),
    /// [`PixelDistance`](crate::PixelDistance),
    /// [`Tolerance`](crate::Tolerance),
    /// [`OddWindowSide`](crate::OddWindowSide),
    /// [`HysteresisThresholds`](crate::analyze::threshold::HysteresisThresholds),
    /// [`Clamp`](crate::transform::Clamp),
    /// [`Harris`](crate::features::detect::Harris),
    /// [`SegmentTest`](crate::features::detect::SegmentTest),
    /// [`NmsRadius`](crate::features::detect::NmsRadius),
    /// [`PeakValue`](crate::analyze::quality::PeakValue),
    /// [`BayerGains`](crate::transform::BayerGains) and their kin — by
    /// validating functions whose parameter is a plain value, and by a
    /// [`BinningStrategy`](crate::analyze::histogram::BinningStrategy)
    /// whose own configuration is invalid.
    ///
    /// The [`ParameterError`] says which parameter was rejected, which
    /// [`Requirement`] it broke and which [`Value`] was received, so a caller
    /// can react to the failure without reading the message:
    ///
    /// ```
    /// use fovea::{Error, Sigma};
    /// use fovea::error::{Requirement, Value};
    ///
    /// let Err(Error::InvalidParameter(e)) = Sigma::try_new(-1.0) else {
    ///     unreachable!("a negative sigma is rejected");
    /// };
    /// assert_eq!(e.requirement(), Requirement::FinitePositive);
    /// assert_eq!(e.value(), Value::F32(-1.0));
    /// ```
    ///
    /// This is the *computed-value* path, for parameters derived from data
    /// at run time. A literal parameter does not need it: the types carry
    /// `const fn new -> Option` constructors, and where a literal is the
    /// normal input, a matching literal macro ([`sigma!`](crate::sigma),
    /// [`pixel_distance!`](crate::pixel_distance),
    /// [`tolerance!`](crate::tolerance), [`window!`](crate::window),
    /// [`harris!`](crate::harris), [`peak!`](crate::peak)) that rejects a
    /// bad literal at compile time, with the same wording.
    InvalidParameter(ParameterError),

    /// The template is larger than the image in one or both dimensions.
    ///
    /// Returned by [`match_template`](crate::transform::match_template) when
    /// the template does not fit inside the image.
    TemplateTooLarge {
        /// The dimensions of the source image.
        image_size: Size,
        /// The dimensions of the template that does not fit.
        template_size: Size,
    },

    /// The template has zero width or height.
    ///
    /// Returned by [`match_template`](crate::transform::match_template) and
    /// [`match_template_into`](crate::transform::match_template_into) —
    /// an empty template (for example a degenerate user crop) has no
    /// defined score.
    EmptyTemplate {
        /// The dimensions of the degenerate template.
        template_size: Size,
    },

    /// A sliding window is larger than the image in one or both
    /// dimensions, so no position has the whole window inside the frame.
    ///
    /// Returned by [`ssim`](crate::analyze::quality::ssim) and
    /// [`ssim_map`](crate::analyze::quality::ssim_map), whose window size
    /// follows from the σ in their parameters.
    WindowLargerThanImage {
        /// The dimensions of the window.
        window: Size,
        /// The dimensions of the image.
        image: Size,
    },

    /// A caliper's footprint leaves the image, and the border policy does
    /// not extend it.
    ///
    /// Returned by [`profile`](crate::measure::profile) under
    /// [`Skip`](crate::border::Skip), where an interpolation tap outside
    /// the image has no value. Where a part lies is data, so this is an
    /// error and not a profile with gaps.
    CaliperOutsideImage {
        /// The first position along the path, counted in sampling steps
        /// from its start, at which a tap falls outside the image.
        sample: usize,
    },

    /// A fit has fewer points than its estimator needs to determine an
    /// element: 2 for a line, 3 for a circle, 5 for an ellipse.
    ///
    /// Returned by [`try_fit`](crate::measure::try_fit), both when the input
    /// is too short and when the outlier handling leaves too few points to
    /// shape the element.
    TooFewPoints {
        /// The estimator's minimum.
        required: usize,
        /// The number of points there were, or that the outlier handling
        /// kept.
        actual: usize,
    },

    /// The points determine no unique element of the kind being fitted.
    ///
    /// Returned by [`try_fit`](crate::measure::try_fit) when the points
    /// coincide or spread equally in every direction (a line), lie on one
    /// line (a circle or an ellipse), or admit no ellipse. Where the points
    /// lie is data, so this is an error and not a panic.
    DegeneratePoints,

    /// An iterative inverse did not reach its tolerance.
    ///
    /// Returned by
    /// [`BrownConrady::undistort_point`](crate::geometry::BrownConrady::undistort_point)
    /// when no ideal point distorts onto the measured one within the
    /// tolerance, which in practice means the point lies outside the field
    /// the lens model was calibrated over, where the model folds back on
    /// itself.
    DidNotConverge {
        /// The number of steps taken before giving up.
        steps: usize,
    },

    /// An image without pixels was asked to fill a result that has some.
    ///
    /// Returned by [`pad`](crate::transform::pad),
    /// [`resize`](crate::transform::resize),
    /// [`resize_into`](crate::transform::resize_into),
    /// [`remap`](crate::transform::remap) and
    /// [`DestToSourceTable::remap`](crate::transform::DestToSourceTable::remap)
    /// when the source is empty, its border policy copies from the image
    /// (`Clamp`, `Mirror`, `Wrap`) or there is none, and the result is not
    /// empty. Under [`Constant`](crate::border::Constant), which needs no
    /// pixel, `pad` and `remap` fill the result with its value instead.
    EmptySource {
        /// The size of the result that could not be filled.
        target: Size,
    },

    /// A DFT method does not transform images of this size.
    ///
    /// Returned by [`dft`](crate::frequency::dft) and
    /// [`Spectrum::inverse`](crate::frequency::Spectrum::inverse) under
    /// [`Radix2`](crate::frequency::Radix2) when a side is not a power of
    /// two. The size is data, so this is an error; padding to the method's
    /// [`next_size`](crate::frequency::Radix2::next_size) first, or a
    /// method that accepts every size, avoids it.
    UnsupportedDftSize {
        /// The method's name.
        method: &'static str,
        /// The size of the image or of the spectrum's source.
        size: Size,
    },

    /// [`pad`](crate::transform::pad) was asked for a target smaller than
    /// the image along a side.
    ///
    /// Padding never crops: an image larger than the target is reported,
    /// not cut down.
    PadTargetTooSmall {
        /// The size of the image.
        source: Size,
        /// The rejected target size.
        target: Size,
    },

    /// The chosen accumulator type cannot hold the worst-case sum for an
    /// image of this size.
    ///
    /// Returned by
    /// [`integral_image`](crate::analyze::integral::integral_image),
    /// [`integral_image_into`](crate::analyze::integral::integral_image_into),
    /// [`integral_squared_image`](crate::analyze::integral::integral_squared_image),
    /// and
    /// [`integral_squared_image_into`](crate::analyze::integral::integral_squared_image_into)
    /// when the O(1) pre-flight overflow check fails.
    ///
    /// `required_capacity` is the theoretical worst-case sum given the
    /// source image dimensions and pixel type. `accumulator_capacity` is
    /// the maximum value the accumulator pixel can hold (per channel,
    /// for multi-channel accumulators). Both are expressed as `u128`
    /// for a uniform representation across integer and floating-point
    /// accumulators (for floats, the capacity is the exact-integer range
    /// of the underlying float type, e.g. `2^53` for `f64`).
    AccumulatorOverflow {
        /// Worst-case sum the chosen accumulator would have to hold,
        /// expressed as `u128`. Set to `u128::MAX` if the worst-case
        /// computation itself overflowed `u128`.
        required_capacity: u128,
        /// Maximum value the accumulator type can hold, as `u128`.
        accumulator_capacity: u128,
    },

    /// The binary image contains more connected components than the
    /// chosen [`LabelPixel`](crate::pixel::LabelPixel) type can encode.
    ///
    /// Returned by
    /// [`connected_components`](crate::analyze::components::connected_components)
    /// and
    /// [`connected_components_into`](crate::analyze::components::connected_components_into)
    /// when pass 1 would allocate the `(label_capacity + 1)`-th
    /// provisional label. This is a Tier 2 / data-dependent error:
    /// a pre-flight check is impossible without running the labeling pass.
    ///
    /// `label_capacity` is `L::MAX_LABEL` for the chosen label type — the
    /// largest distinct foreground label it can represent. Callers can
    /// retry with a wider label type (e.g. `Label32` if a hypothetical
    /// narrower `Label16` overflowed).
    LabelOverflow {
        /// `MAX_LABEL` of the chosen label pixel type — the maximum
        /// foreground label the type can represent.
        label_capacity: u32,
    },
}

/// Which parameter was rejected, which rule it broke, and what was
/// received.
///
/// The payload of [`Error::InvalidParameter`]. The three parts are data
/// rather than text, so a caller or a test matches on
/// [`requirement`](Self::requirement) and [`value`](Self::value) instead of
/// searching the message. The [`parameter`](Self::parameter) name is for
/// the message.
///
/// A [`BinningStrategy`](crate::analyze::histogram::BinningStrategy)
/// implemented outside this crate builds one with [`new`](Self::new) to
/// report its own invalid configuration.
///
/// # Example
///
/// ```
/// use fovea::Error;
/// use fovea::error::{ParameterError, Requirement, Value};
///
/// let e = ParameterError::new("bin width", Requirement::FinitePositive, Value::F64(0.0));
/// assert_eq!(
///     Error::from(e).to_string(),
///     "invalid parameter: bin width must be finite and strictly positive, got 0"
/// );
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParameterError {
    parameter: &'static str,
    requirement: Requirement,
    value: Value,
    index: Option<usize>,
}

impl ParameterError {
    /// Creates the payload for `parameter`, which broke `requirement` with
    /// `value`.
    #[must_use]
    pub const fn new(parameter: &'static str, requirement: Requirement, value: Value) -> Self {
        Self {
            parameter,
            requirement,
            value,
            index: None,
        }
    }

    /// Records where in a sequence the rejected element sits: the channel
    /// of a pixel, the position in a list of edges.
    #[must_use]
    pub const fn at(self, index: usize) -> Self {
        Self {
            index: Some(index),
            ..self
        }
    }

    /// The name of the rejected parameter, as the message prints it.
    #[must_use]
    pub const fn parameter(&self) -> &'static str {
        self.parameter
    }

    /// The rule the value broke.
    #[must_use]
    pub const fn requirement(&self) -> Requirement {
        self.requirement
    }

    /// The value that was received.
    #[must_use]
    pub const fn value(&self) -> Value {
        self.value
    }

    /// Where in a sequence the rejected element sits, if the parameter is
    /// one element of several.
    #[must_use]
    pub const fn index(&self) -> Option<usize> {
        self.index
    }
}

impl From<ParameterError> for Error {
    #[inline]
    fn from(e: ParameterError) -> Self {
        Error::InvalidParameter(e)
    }
}

impl fmt::Display for ParameterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.parameter)?;
        if let Some(index) = self.index {
            write!(f, " {index}")?;
        }
        write!(f, " {}", self.requirement)?;
        if self.value != Value::NotRecorded {
            write!(f, ", got {}", self.value)?;
        }
        Ok(())
    }
}

/// The rule a rejected parameter broke.
///
/// Each variant is a whole invariant as a type documents it, so a NaN σ
/// and a negative σ both report [`FinitePositive`](Self::FinitePositive).
///
/// Equality compares the float bounds of
/// [`OpenInterval`](Self::OpenInterval) by bit pattern, which keeps `Eq`
/// lawful.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum Requirement {
    /// Neither NaN nor infinite.
    Finite,
    /// Finite and greater than zero.
    FinitePositive,
    /// Finite and not below zero.
    FiniteNonNegative,
    /// Finite and not zero. A subnormal value counts as zero, because its
    /// reciprocal overflows.
    FiniteNonZero,
    /// An odd integer (which excludes zero).
    Odd,
    /// An integer not below the bound.
    AtLeast(usize),
    /// An integer not above the bound.
    AtMost(usize),
    /// An integer in `min..=max`.
    InRange {
        /// The smallest accepted value.
        min: usize,
        /// The largest accepted value.
        max: usize,
    },
    /// A number strictly between the bounds, both excluded.
    OpenInterval {
        /// The excluded lower bound.
        low: f64,
        /// The excluded upper bound.
        high: f64,
    },
    /// A pair with `low <= high`.
    Ordered,
    /// A pair with `low < high`.
    StrictlyOrdered,
    /// Exactly zero: a value the model does not use, accepted only where it
    /// changes nothing.
    Zero,
    /// One of the listed counts.
    OneOf(&'static [usize]),
    /// A whole multiple of the bound, such as a byte count that must hold
    /// whole pixels.
    MultipleOf(usize),
}

impl Requirement {
    /// The wording of the requirement, without its bounds.
    ///
    /// `const`, so the literal macros can use it in their compile-time
    /// check and a rejected literal reads the same as a rejected runtime
    /// value. [`Display`](fmt::Display) appends the bounds of the variants
    /// that carry them.
    #[must_use]
    pub const fn text(self) -> &'static str {
        match self {
            Requirement::Finite => "must be finite",
            Requirement::FinitePositive => "must be finite and strictly positive",
            Requirement::FiniteNonNegative => "must be finite and non-negative",
            Requirement::FiniteNonZero => "must be finite and non-zero",
            Requirement::Odd => "must be odd",
            Requirement::AtLeast(_) => "must be at least",
            Requirement::AtMost(_) => "must be at most",
            Requirement::InRange { .. } => "must lie in the inclusive range",
            Requirement::OpenInterval { .. } => "must lie strictly between",
            Requirement::Ordered => "must satisfy low <= high",
            Requirement::StrictlyOrdered => "must satisfy low < high",
            Requirement::Zero => "must be zero",
            Requirement::OneOf(_) => "must be one of",
            Requirement::MultipleOf(_) => "must be a multiple of",
        }
    }
}

impl PartialEq for Requirement {
    fn eq(&self, other: &Self) -> bool {
        use Requirement::*;
        match (*self, *other) {
            (Finite, Finite)
            | (FinitePositive, FinitePositive)
            | (FiniteNonNegative, FiniteNonNegative)
            | (FiniteNonZero, FiniteNonZero)
            | (Odd, Odd)
            | (Ordered, Ordered)
            | (StrictlyOrdered, StrictlyOrdered)
            | (Zero, Zero) => true,
            (AtLeast(a), AtLeast(b)) | (AtMost(a), AtMost(b)) | (MultipleOf(a), MultipleOf(b)) => {
                a == b
            }
            (OneOf(a), OneOf(b)) => a == b,
            (InRange { min: a, max: b }, InRange { min: c, max: d }) => a == c && b == d,
            (OpenInterval { low: a, high: b }, OpenInterval { low: c, high: d }) => {
                a.to_bits() == c.to_bits() && b.to_bits() == d.to_bits()
            }
            _ => false,
        }
    }
}

impl Eq for Requirement {}

impl fmt::Display for Requirement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.text())?;
        match *self {
            Requirement::AtLeast(bound)
            | Requirement::AtMost(bound)
            | Requirement::MultipleOf(bound) => write!(f, " {bound}"),
            Requirement::InRange { min, max } => write!(f, " {min}..={max}"),
            Requirement::OpenInterval { low, high } => write!(f, " {low} and {high}"),
            Requirement::OneOf(counts) => {
                for (i, c) in counts.iter().enumerate() {
                    write!(f, "{}{c}", if i == 0 { " " } else { ", " })?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
}

/// The value a rejected parameter received.
///
/// Equality compares floats by bit pattern, so a `Value` holding NaN equals
/// itself and `assert_eq!` works for every input a constructor rejects. The
/// flip side: `0.0` and `-0.0` are different values here.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum Value {
    /// A single `f32`.
    F32(f32),
    /// A single `f64`.
    F64(f64),
    /// A single count, length or index.
    Usize(usize),
    /// Two `f64` that are checked together, such as a range or an offset.
    F64Pair(f64, f64),
    /// The value has a generic type the error cannot hold; the caller
    /// passed it and still has it.
    NotRecorded,
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (*self, *other) {
            (Value::F32(a), Value::F32(b)) => a.to_bits() == b.to_bits(),
            (Value::F64(a), Value::F64(b)) => a.to_bits() == b.to_bits(),
            (Value::Usize(a), Value::Usize(b)) => a == b,
            (Value::F64Pair(a, b), Value::F64Pair(c, d)) => {
                a.to_bits() == c.to_bits() && b.to_bits() == d.to_bits()
            }
            (Value::NotRecorded, Value::NotRecorded) => true,
            _ => false,
        }
    }
}

impl Eq for Value {}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Value::F32(v) => write!(f, "{v}"),
            Value::F64(v) => write!(f, "{v}"),
            Value::Usize(v) => write!(f, "{v}"),
            Value::F64Pair(a, b) => write!(f, "({a}, {b})"),
            Value::NotRecorded => f.write_str("not recorded"),
        }
    }
}

// `std::error::Error` is implemented manually (not via `thiserror`) to
// avoid pulling in a derive dependency for the core crate. The default
// blanket `source()` (returns `None`) is correct for every variant: no
// `Error` value wraps another `Error`. If we ever add a wrapping variant
// we must override `source` for it.
impl std::error::Error for Error {}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::SizeMismatch { expected, actual } => {
                write!(
                    f,
                    "size mismatch: expected {}x{}, got {}x{}",
                    expected.width, expected.height, actual.width, actual.height
                )
            }
            Error::LengthMismatch { expected, actual } => {
                write!(
                    f,
                    "length mismatch: expected {} elements, got {}",
                    expected, actual
                )
            }
            Error::ChannelCountMismatch { expected, actual } => {
                write!(
                    f,
                    "channel count mismatch: expected {} channels, got {}",
                    expected, actual
                )
            }
            Error::InvalidPyrUpTarget { source, target } => {
                write!(
                    f,
                    "invalid pyr_up target: {}x{} is not a size whose pyr_down is {}x{}",
                    target.width, target.height, source.width, source.height
                )
            }
            Error::EmptyPyramid => {
                write!(
                    f,
                    "empty pyramid: a pyramid must contain at least one level"
                )
            }
            Error::PyramidLevelOrder {
                index,
                previous,
                current,
            } => {
                write!(
                    f,
                    "pyramid level order: level {} is {}x{}, larger than its \
                     predecessor {}x{} (levels must be finest to coarsest)",
                    index, current.width, current.height, previous.width, previous.height
                )
            }
            Error::NotDyadic {
                index,
                parent,
                child,
            } => {
                write!(
                    f,
                    "not dyadic: level {} is {}x{}, but its predecessor {}x{} halves to \
                     {}x{} (every level must be ceil(parent / 2) along both axes)",
                    index,
                    child.width,
                    child.height,
                    parent.width,
                    parent.height,
                    parent.width / 2 + parent.width % 2,
                    parent.height / 2 + parent.height % 2
                )
            }
            Error::InvalidParameter(e) => {
                write!(f, "invalid parameter: {e}")
            }
            Error::EmptyTemplate { template_size } => {
                write!(
                    f,
                    "empty template: {}x{} has zero width or height",
                    template_size.width, template_size.height
                )
            }
            Error::TemplateTooLarge {
                image_size,
                template_size,
            } => {
                write!(
                    f,
                    "template {}x{} is larger than image {}x{}",
                    template_size.width, template_size.height, image_size.width, image_size.height
                )
            }
            Error::WindowLargerThanImage { window, image } => {
                write!(
                    f,
                    "window {}x{} is larger than image {}x{}",
                    window.width, window.height, image.width, image.height
                )
            }
            Error::CaliperOutsideImage { sample } => {
                write!(
                    f,
                    "caliper footprint leaves the image at sample {sample} along its path"
                )
            }
            Error::TooFewPoints { required, actual } => {
                write!(
                    f,
                    "too few points: the fit needs at least {required}, got {actual}"
                )
            }
            Error::DegeneratePoints => {
                write!(
                    f,
                    "degenerate points: they determine no unique element of this kind"
                )
            }
            Error::DidNotConverge { steps } => {
                write!(
                    f,
                    "did not converge: no solution within the tolerance after {steps} steps"
                )
            }
            Error::EmptySource { target } => {
                write!(
                    f,
                    "empty source: the image has no pixel to fill the {}x{} result from",
                    target.width, target.height
                )
            }
            Error::UnsupportedDftSize { method, size } => {
                write!(
                    f,
                    "unsupported DFT size: {method} does not transform a {}x{} image",
                    size.width, size.height
                )
            }
            Error::PadTargetTooSmall { source, target } => {
                write!(
                    f,
                    "pad target too small: {}x{} is smaller than the {}x{} image along a side",
                    target.width, target.height, source.width, source.height
                )
            }
            Error::AccumulatorOverflow {
                required_capacity,
                accumulator_capacity,
            } => {
                write!(
                    f,
                    "accumulator overflow: image requires capacity for {}, \
                     but accumulator can hold at most {}",
                    required_capacity, accumulator_capacity
                )
            }
            Error::LabelOverflow { label_capacity } => {
                write!(
                    f,
                    "label overflow: image contains more components than the \
                     chosen label type can represent (capacity = {})",
                    label_capacity
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_size_mismatch() {
        let err = Error::SizeMismatch {
            expected: Size::new(640, 480),
            actual: Size::new(320, 240),
        };
        assert_eq!(
            err.to_string(),
            "size mismatch: expected 640x480, got 320x240"
        );
    }

    #[test]
    fn display_caliper_outside_image() {
        let err = Error::CaliperOutsideImage { sample: 17 };
        assert_eq!(
            err.to_string(),
            "caliper footprint leaves the image at sample 17 along its path"
        );
    }

    #[test]
    fn display_fit_errors() {
        let err = Error::TooFewPoints {
            required: 5,
            actual: 4,
        };
        assert_eq!(
            err.to_string(),
            "too few points: the fit needs at least 5, got 4"
        );
        assert_eq!(
            Error::DegeneratePoints.to_string(),
            "degenerate points: they determine no unique element of this kind"
        );
        assert_eq!(
            Error::DidNotConverge { steps: 20 }.to_string(),
            "did not converge: no solution within the tolerance after 20 steps"
        );
    }

    #[test]
    fn display_zero_and_one_of() {
        let e =
            ParameterError::new("distortion coefficient", Requirement::Zero, Value::F64(0.5)).at(5);
        assert_eq!(
            e.to_string(),
            "distortion coefficient 5 must be zero, got 0.5"
        );
        let e = ParameterError::new(
            "distortion coefficient count",
            Requirement::OneOf(&[4, 5, 8]),
            Value::Usize(6),
        );
        assert_eq!(
            e.to_string(),
            "distortion coefficient count must be one of 4, 5, 8, got 6"
        );
        assert_eq!(Requirement::OneOf(&[4, 5]), Requirement::OneOf(&[4, 5]));
        assert_ne!(Requirement::OneOf(&[4, 5]), Requirement::OneOf(&[4]));
        assert_ne!(Requirement::Zero, Requirement::Finite);
    }

    #[test]
    fn display_empty_source() {
        let err = Error::EmptySource {
            target: Size::new(64, 48),
        };
        assert_eq!(
            err.to_string(),
            "empty source: the image has no pixel to fill the 64x48 result from"
        );
    }

    #[test]
    fn display_dft_and_pad_errors() {
        let err = Error::UnsupportedDftSize {
            method: "Radix2",
            size: Size::new(1920, 1080),
        };
        assert_eq!(
            err.to_string(),
            "unsupported DFT size: Radix2 does not transform a 1920x1080 image"
        );
        let err = Error::PadTargetTooSmall {
            source: Size::new(640, 480),
            target: Size::new(512, 512),
        };
        assert_eq!(
            err.to_string(),
            "pad target too small: 512x512 is smaller than the 640x480 image along a side"
        );
    }

    #[test]
    fn display_length_mismatch() {
        let err = Error::LengthMismatch {
            expected: 100,
            actual: 50,
        };
        assert_eq!(
            err.to_string(),
            "length mismatch: expected 100 elements, got 50"
        );
    }

    #[test]
    fn display_channel_count_mismatch() {
        let err = Error::ChannelCountMismatch {
            expected: 3,
            actual: 2,
        };
        assert_eq!(
            err.to_string(),
            "channel count mismatch: expected 3 channels, got 2"
        );
    }

    #[test]
    fn error_is_clone() {
        let err = Error::LengthMismatch {
            expected: 10,
            actual: 5,
        };
        let cloned = err.clone();
        assert_eq!(err, cloned);
    }

    #[test]
    fn error_is_debug() {
        let err = Error::SizeMismatch {
            expected: Size::new(10, 10),
            actual: Size::new(5, 5),
        };
        let debug = format!("{:?}", err);
        assert!(debug.contains("SizeMismatch"));
    }

    #[test]
    fn error_equality() {
        let a = Error::LengthMismatch {
            expected: 100,
            actual: 50,
        };
        let b = Error::LengthMismatch {
            expected: 100,
            actual: 50,
        };
        let c = Error::LengthMismatch {
            expected: 100,
            actual: 99,
        };
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn display_invalid_pyr_up_target() {
        let err = Error::InvalidPyrUpTarget {
            source: Size::new(4, 4),
            target: Size::new(9, 8),
        };
        assert_eq!(
            err.to_string(),
            "invalid pyr_up target: 9x8 is not a size whose pyr_down is 4x4"
        );
    }

    #[test]
    fn invalid_pyr_up_target_equality_and_clone() {
        let a = Error::InvalidPyrUpTarget {
            source: Size::new(4, 4),
            target: Size::new(9, 8),
        };
        let b = a.clone();
        let c = Error::InvalidPyrUpTarget {
            source: Size::new(4, 4),
            target: Size::new(6, 8),
        };
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn display_empty_pyramid() {
        assert_eq!(
            Error::EmptyPyramid.to_string(),
            "empty pyramid: a pyramid must contain at least one level"
        );
    }

    #[test]
    fn display_pyramid_level_order() {
        let err = Error::PyramidLevelOrder {
            index: 2,
            previous: Size::new(4, 3),
            current: Size::new(8, 6),
        };
        assert_eq!(
            err.to_string(),
            "pyramid level order: level 2 is 8x6, larger than its \
             predecessor 4x3 (levels must be finest to coarsest)"
        );
    }

    #[test]
    fn display_not_dyadic() {
        let err = Error::NotDyadic {
            index: 1,
            parent: Size::new(101, 68),
            child: Size::new(30, 34),
        };
        assert_eq!(
            err.to_string(),
            "not dyadic: level 1 is 30x34, but its predecessor 101x68 halves to \
             51x34 (every level must be ceil(parent / 2) along both axes)"
        );
    }

    #[test]
    fn display_invalid_parameter() {
        let err = Error::from(ParameterError::new(
            "sigma",
            Requirement::FinitePositive,
            Value::F32(-1.0),
        ));
        assert_eq!(
            err.to_string(),
            "invalid parameter: sigma must be finite and strictly positive, got -1"
        );
    }

    #[test]
    fn display_invalid_parameter_with_bounds_and_index() {
        let cases = [
            (
                ParameterError::new("radius", Requirement::AtLeast(1), Value::Usize(0)),
                "radius must be at least 1, got 0",
            ),
            (
                ParameterError::new(
                    "palette length",
                    Requirement::AtMost(256),
                    Value::Usize(257),
                ),
                "palette length must be at most 256, got 257",
            ),
            (
                ParameterError::new("row stride", Requirement::MultipleOf(2), Value::Usize(15)),
                "row stride must be a multiple of 2, got 15",
            ),
            (
                ParameterError::new(
                    "arc length",
                    Requirement::InRange { min: 9, max: 16 },
                    Value::Usize(8),
                ),
                "arc length must lie in the inclusive range 9..=16, got 8",
            ),
            (
                ParameterError::new(
                    "k",
                    Requirement::OpenInterval {
                        low: 0.0,
                        high: 0.25,
                    },
                    Value::F32(0.3),
                ),
                "k must lie strictly between 0 and 0.25, got 0.3",
            ),
            (
                ParameterError::new(
                    "range",
                    Requirement::StrictlyOrdered,
                    Value::F64Pair(2.0, 1.0),
                ),
                "range must satisfy low < high, got (2, 1)",
            ),
            (
                ParameterError::new("clamp channel", Requirement::Ordered, Value::NotRecorded)
                    .at(1),
                "clamp channel 1 must satisfy low <= high",
            ),
        ];
        for (e, expected) in cases {
            assert_eq!(e.to_string(), expected);
        }
    }

    #[test]
    fn parameter_error_accessors() {
        let e = ParameterError::new("edge", Requirement::Finite, Value::F64(f64::NAN)).at(3);
        assert_eq!(e.parameter(), "edge");
        assert_eq!(e.requirement(), Requirement::Finite);
        assert_eq!(e.value(), Value::F64(f64::NAN));
        assert_eq!(e.index(), Some(3));
        assert_eq!(
            ParameterError::new("edge", Requirement::Finite, Value::F64(1.0)).index(),
            None
        );
    }

    #[test]
    fn a_nan_payload_equals_itself() {
        // The reason floats compare by bit pattern: under IEEE equality this
        // error would be unequal to itself, and `assert_eq!` would fail for
        // exactly the inputs the constructors exist to reject.
        let nan = ParameterError::new("sigma", Requirement::FinitePositive, Value::F32(f32::NAN));
        assert_eq!(nan, nan);
        assert_eq!(Error::from(nan), Error::from(nan));
        assert_eq!(Value::F64Pair(f64::NAN, 1.0), Value::F64Pair(f64::NAN, 1.0));
        assert_ne!(Value::F64Pair(f64::NAN, 1.0), Value::F64Pair(1.0, f64::NAN));
    }

    #[test]
    fn value_equality_is_by_bit_pattern_and_by_type() {
        assert_ne!(Value::F64(0.0), Value::F64(-0.0));
        assert_ne!(Value::F32(1.0), Value::F64(1.0));
        assert_ne!(Value::Usize(1), Value::F64(1.0));
        assert_eq!(Value::NotRecorded, Value::NotRecorded);
    }

    #[test]
    fn requirement_equality_compares_bounds() {
        assert_eq!(Requirement::AtLeast(3), Requirement::AtLeast(3));
        assert_ne!(Requirement::AtLeast(3), Requirement::AtMost(3));
        assert_eq!(Requirement::MultipleOf(2), Requirement::MultipleOf(2));
        assert_ne!(Requirement::MultipleOf(2), Requirement::AtLeast(2));
        assert_ne!(
            Requirement::InRange { min: 9, max: 16 },
            Requirement::InRange { min: 9, max: 15 }
        );
        let open = Requirement::OpenInterval {
            low: 0.0,
            high: 0.25,
        };
        assert_eq!(open, open);
        assert_ne!(
            open,
            Requirement::OpenInterval {
                low: 0.0,
                high: 0.5
            }
        );
        assert_ne!(Requirement::Ordered, Requirement::StrictlyOrdered);
    }

    #[test]
    fn display_value_variants() {
        assert_eq!(Value::F32(1.5).to_string(), "1.5");
        assert_eq!(Value::F64(f64::INFINITY).to_string(), "inf");
        assert_eq!(Value::Usize(7).to_string(), "7");
        assert_eq!(Value::F64Pair(0.5, f64::NAN).to_string(), "(0.5, NaN)");
        assert_eq!(Value::NotRecorded.to_string(), "not recorded");
    }

    #[test]
    fn requirement_text_is_const() {
        // The literal macros read the wording in a `const` block.
        const TEXT: &str = Requirement::FinitePositive.text();
        assert_eq!(TEXT, "must be finite and strictly positive");
    }

    #[test]
    fn display_empty_template() {
        let err = Error::EmptyTemplate {
            template_size: Size::new(0, 5),
        };
        assert_eq!(
            err.to_string(),
            "empty template: 0x5 has zero width or height"
        );
    }

    #[test]
    fn display_template_too_large() {
        let err = Error::TemplateTooLarge {
            image_size: Size::new(10, 10),
            template_size: Size::new(20, 15),
        };
        assert_eq!(err.to_string(), "template 20x15 is larger than image 10x10");
    }

    #[test]
    fn display_window_larger_than_image() {
        let err = Error::WindowLargerThanImage {
            window: Size::new(11, 11),
            image: Size::new(8, 20),
        };
        assert_eq!(err.to_string(), "window 11x11 is larger than image 8x20");
    }

    #[test]
    fn different_variants_not_equal() {
        let size_err = Error::SizeMismatch {
            expected: Size::new(10, 10),
            actual: Size::new(5, 5),
        };
        let length_err = Error::LengthMismatch {
            expected: 100,
            actual: 25,
        };
        assert_ne!(size_err, length_err);
    }

    #[test]
    fn display_accumulator_overflow() {
        let err = Error::AccumulatorOverflow {
            required_capacity: 4_278_190_080,
            accumulator_capacity: 4_294_967_295,
        };
        assert_eq!(
            err.to_string(),
            "accumulator overflow: image requires capacity for 4278190080, \
             but accumulator can hold at most 4294967295"
        );
    }

    #[test]
    fn accumulator_overflow_equality_and_clone() {
        let a = Error::AccumulatorOverflow {
            required_capacity: 100,
            accumulator_capacity: 50,
        };
        let b = a.clone();
        let c = Error::AccumulatorOverflow {
            required_capacity: 100,
            accumulator_capacity: 51,
        };
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn display_label_overflow() {
        let err = Error::LabelOverflow {
            label_capacity: u32::MAX,
        };
        assert_eq!(
            err.to_string(),
            "label overflow: image contains more components than the chosen label type \
             can represent (capacity = 4294967295)"
        );
    }

    #[test]
    fn label_overflow_equality_and_clone() {
        let a = Error::LabelOverflow {
            label_capacity: 255,
        };
        let b = a.clone();
        let c = Error::LabelOverflow {
            label_capacity: 65_535,
        };
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn error_implements_std_error_trait() {
        // P1-4: `Error` must integrate with the std error ecosystem so
        // it can be boxed into `Box<dyn std::error::Error>` and used with
        // `?` against `Box<dyn Error + Send + Sync>` sinks.
        fn assert_error<E: std::error::Error>() {}
        assert_error::<Error>();

        let err: Box<dyn std::error::Error> = Box::new(Error::LengthMismatch {
            expected: 10,
            actual: 5,
        });
        // Display reachable through the trait object.
        assert!(err.to_string().contains("length mismatch"));
        // No wrapped source (no Error variant wraps another error today).
        assert!(err.source().is_none());
    }
}

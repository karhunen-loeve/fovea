use crate::common::Size;
use core::fmt;

/// Errors returned by fallible image operations.
///
/// This type represents data-dependent failures — situations where the
/// operation is well-formed but the supplied data doesn't meet the
/// requirements. See [ADR-0025] for the three-tier error handling
/// convention used throughout this crate.
///
/// [ADR-0025]: https://github.com/…/docs/adr/0025-error-handling-conventions.md
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
/// use irys_cv::Error;
/// use irys_cv::Size;
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
    /// constructors where `data.len() != width * height`.
    LengthMismatch {
        /// The number of elements required (`width * height`, or
        /// `width * height * pixel_size` for byte constructors).
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

    /// A caller-supplied binning strategy contains invalid parameters.
    ///
    /// Returned by [`histogram`](crate::analyze::histogram::histogram()) when
    /// the strategy's `validate()` rejects its own configuration — for
    /// example, `LinearBins` with `min >= max`, non-finite bounds, a
    /// `bin_count` of zero, or `CustomBins` whose edges are not strictly
    /// increasing.
    ///
    /// The contained string describes the specific reason. Treat it as
    /// human-readable diagnostic text, not as a stable machine-readable
    /// tag.
    InvalidBinningStrategy(String),

    /// The chosen accumulator type cannot hold the worst-case sum for an
    /// image of this size.
    ///
    /// Returned by
    /// [`integral_image`](crate::analyze::integral::integral_image),
    /// [`integral_image_into`](crate::analyze::integral::integral_image_into),
    /// [`integral_squared_image`](crate::analyze::integral::integral_squared_image),
    /// and
    /// [`integral_squared_image_into`](crate::analyze::integral::integral_squared_image_into)
    /// when the O(1) pre-flight overflow check fails (ADR-0032 §3, §6).
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
    /// provisional label (ADR-0047 §6). This is a Tier 2 / data-dependent
    /// error per [ADR-0025](https://github.com/karhunen-loeve/irys-cv/blob/main/docs/adr/0025-error-handling-conventions.md):
    /// a pre-flight check is impossible without running the labeling pass.
    ///
    /// `label_capacity` is `L::MAX_LABEL` for the chosen label type — the
    /// largest distinct foreground label it can represent. Callers can
    /// retry with a wider label type (e.g. `Label32` if a hypothetical
    /// narrower `Label16` overflowed).
    LabelOverflow {
        /// `MAX_LABEL` of the chosen label pixel type — the maximum
        /// foreground label the type can represent.
        label_capacity: u64,
    },
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
            Error::InvalidBinningStrategy(reason) => {
                write!(f, "invalid binning strategy: {}", reason)
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
    fn display_template_too_large() {
        let err = Error::TemplateTooLarge {
            image_size: Size::new(10, 10),
            template_size: Size::new(20, 15),
        };
        assert_eq!(err.to_string(), "template 20x15 is larger than image 10x10");
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
    fn display_invalid_binning_strategy() {
        let err = Error::InvalidBinningStrategy("min >= max".to_string());
        assert_eq!(err.to_string(), "invalid binning strategy: min >= max");
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
            label_capacity: u32::MAX as u64,
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
    fn invalid_binning_strategy_equality_and_clone() {
        let a = Error::InvalidBinningStrategy("bin_count == 0".to_string());
        let b = a.clone();
        let c = Error::InvalidBinningStrategy("non-finite edge".to_string());
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

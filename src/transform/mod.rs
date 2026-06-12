//! Image-producing operations: conversion, resize, geometry, filters, morphology, and template matching.
//!
//! Start with [`convert_image`](crate::transform::convert_image) when the pixel type changes, [`resize`](crate::transform::resize) when
//! dimensions change, [`combine_images`](crate::transform::combine_images) when two same-sized images become one,
//! and the filter/convolution APIs when each output pixel depends on a
//! neighborhood.
//!
//! ## Common choices
//!
//! | Task | Start with | Important constraint |
//! |---|---|---|
//! | Convert sRGB to linear light | [`convert_image`](crate::transform::convert_image) + [`SrgbGamma`](crate::transform::SrgbGamma) | The strategy names the transfer function. |
//! | Resize by copying samples | [`resize`](crate::transform::resize) + [`NearestNeighbor`](crate::transform::NearestNeighbor) | Works for gamma-encoded pixels because no blending occurs. |
//! | Resize smoothly | [`resize`](crate::transform::resize) + [`Bilinear`](crate::transform::Bilinear) | Requires [`crate::pixel::LinearSpace`]. Linearize sRGB first. |
//! | Combine two images pixel-wise | [`combine_images`](crate::transform::combine_images) | Inputs must have the same size. |
//! | Apply a convolution/filter | [`convolve`](crate::transform::convolve) or named filters | Choose an explicit border policy. |
//! | Erode/dilate/median | [`map_neighborhood`](crate::transform::map_neighborhood) or morphology helpers | Use masks for active neighborhood positions. |
//!
//! ## Geometry and flips
//!
//! Pixel **positions** are rearranged without modifying pixel **values**.
//! [`flip_h`](crate::transform::flip_h), [`flip_v`](crate::transform::flip_v),
//! [`rotate_90`](crate::transform::rotate_90),
//! [`rotate_180`](crate::transform::rotate_180),
//! [`rotate_270`](crate::transform::rotate_270),
//! and [`transpose`](crate::transform::transpose) each return a fresh,
//! fully-fledged [`Image`](crate::image::Image) (or write into a caller-supplied
//! [`RasterImageMut`](crate::image::RasterImageMut)).
//! Because the operation is purely a coordinate permutation, the pixel
//! bound is only [`Copy`] — every pixel type the library defines is
//! supported (including [`crate::pixel::Srgb8`] and
//! [`crate::pixel::Indexed8`]). These are physical operations rather
//! than view wrappers, so the result is a normal image.
//!
//! ## Pixel conversion
//!
//! Apply a function to every pixel independently, potentially changing the
//! pixel type.  The workhorse here is [`convert_image`](crate::transform::convert_image)
//! (and the in-place [`convert_image_into`](crate::transform::convert_image_into)).
//! Named strategies such as [`Luminance`](crate::transform::Luminance),
//! [`SrgbGamma`](crate::transform::SrgbGamma), and
//! [`Clamp`](crate::transform::Clamp) cover the most common conversions; the
//! [`PixelMap`](crate::transform::PixelMap) wrapper lets you pass an arbitrary closure.
//!
//! ## Image arithmetic
//!
//! Combine two images of the same size pixel-wise, producing a third image.
//! [`combine_images`](crate::transform::combine_images) is the generic engine;
//! named strategies ([`PixelAdd`](crate::transform::PixelAdd),
//! [`PixelSubtract`](crate::transform::PixelSubtract),
//! [`PixelMultiply`](crate::transform::PixelMultiply),
//! [`AbsDiff`](crate::transform::AbsDiff),
//! [`Min`](crate::transform::Min),
//! [`Max`](crate::transform::Max),
//! [`LinearCombine`](crate::transform::LinearCombine),
//! [`Blend`](crate::transform::Blend),
//! [`Magnitude`](crate::transform::Magnitude)) cover everyday arithmetic.
//! Thin free-function wrappers ([`add`](crate::transform::add),
//! [`subtract`](crate::transform::subtract),
//! [`abs_diff`](crate::transform::abs_diff),
//! [`image_min`](crate::transform::image_min),
//! [`image_max`](crate::transform::image_max)) are provided as
//! discoverability shortcuts.
//!
//! ## Neighbourhood transforms
//!
//! Each output pixel is computed from a window of input pixels centred at
//! that position.  Two complementary primitives are provided:
//!
//! * **[`fold_neighborhood`](crate::transform::fold_neighborhood)** — fixed,
//!   weighted aggregation (convolution, separable filters, …).
//!   The [`FoldOp`](crate::transform::FoldOp) trait +
//!   [`ClosureFold`](crate::transform::ClosureFold) wrapper
//!   give a zero-overhead, monomorphised hot path.
//!
//! * **[`map_neighborhood`](crate::transform::map_neighborhood)** — non-linear,
//!   data-dependent transforms (erosion/dilation, median, bilateral, …).
//!   The [`MapOp`](crate::transform::MapOp) trait +
//!   [`ClosureMap`](crate::transform::ClosureMap) wrapper follow the same
//!   design; a boolean [`Mask`](crate::image::Mask) selects which neighbours
//!   are active.
//!
//! Both engines use an interior/boundary split: the interior pixels are
//! processed with fast direct indexing; only the thin boundary strip goes
//! through the [`crate::border::BorderPolicy`] accessor.

mod combine;
mod convert;
mod convolve;
mod convolve_separable;
mod filters;
mod fold;
mod geometry;
mod map_neighborhood;
mod morphology;
mod resize;
mod template_match;

pub use crate::pixel::blend;
pub use combine::{
    AbsDiff, Blend, ClosureCombine, CombinePixels, LinearCombine, Magnitude, MagnitudeChannel, Max,
    Min, PixelAdd, PixelMultiply, PixelSubtract, abs_diff, add, combine_images, combine_images_fn,
    combine_images_fn_into, combine_images_into, image_max, image_min, subtract,
};
pub use convert::{
    AddAlpha, BinaryMask, BinaryThreshold, BinaryThresholdInv, BrightnessContrast, Broadcast,
    ChannelLut, Clamp, ColorSwap, ConvertPixel, ConvertPixelExt, Depalettize, FullRange, Invert,
    Luminance, Lut, Narrow, PixelMap, SrgbGamma, Then, ToZeroThreshold, ToZeroThresholdInv,
    TruncateThreshold, convert_image, convert_image_into,
};
pub use convolve::{convolve, convolve_into, correlate, correlate_into};
pub use convolve_separable::{convolve_separable, convolve_separable_into};
pub use filters::{
    box_blur_3x3, box_blur_5x5, emboss, gaussian_blur_3x3, gaussian_blur_5x5, laplacian,
    laplacian_8, prewitt_x, prewitt_y, scharr_x, scharr_y, sharpen, sobel_x, sobel_y,
};
pub use fold::{
    ClosureFold, FoldItem, FoldOp, fold_neighborhood, fold_neighborhood_fn,
    fold_neighborhood_fn_into, fold_neighborhood_into,
};
pub use geometry::{
    flip_h, flip_h_into, flip_v, flip_v_into, rotate_90, rotate_90_into, rotate_180,
    rotate_180_into, rotate_270, rotate_270_into, transpose, transpose_into,
};
pub use map_neighborhood::{
    ClosureMap, MapItem, MapOp, map_neighborhood, map_neighborhood_fn, map_neighborhood_fn_into,
    map_neighborhood_into,
};
pub use morphology::{
    black_hat, closing, closing_into, dilate, dilate_into, erode, erode_into, median_filter,
    morphological_gradient, opening, opening_into, top_hat,
};
pub use resize::{Bilinear, NearestNeighbor, ResizeMethod, resize, resize_into};
pub use template_match::{MatchMethod, NCC, SAD, SSD, match_template, match_template_into};

//! Multi-resolution image pyramids.
//!
//! A [`Pyramid<L>`] is a chain of levels at decreasing resolution, generic
//! over the level type `L` rather than over the construction method: the
//! structural truth ("what does a level contain?") lives in the type, while
//! the construction strategy
//! ([`PyramidMethod`](crate::transform::PyramidMethod)) is consumed at build
//! time and not stored — the same pattern as
//! [`ResizeMethod`](crate::transform::ResizeMethod) and
//! [`ConvertPixel`](crate::transform::ConvertPixel).
//!
//! [`Image<P>`] implements [`PyramidLevel`] directly, so a Gaussian pyramid
//! is simply `Pyramid<Image<P>>` (aliased as [`GaussianPyramid<P>`]) with no
//! wrapper cost. Levels that carry scale metadata opt in via the
//! [`Decimated`] / [`ScaleLevel`] capability traits, implemented by the thin
//! [`ScaledImage<P>`] wrapper.

use crate::error::Error;
use crate::image::{Image, ImageView};
use crate::{CoordinateF64, PixelDistance, Sigma, Size};

// ─── PyramidLevel ────────────────────────────────────────────────────────────

/// Base trait for all pyramid level types.
///
/// Every level provides access to itself as an image — the minimum contract
/// that all pyramid-consuming algorithms can rely on; the level's spatial
/// size is `as_image().size()`. Capability traits ([`Decimated`],
/// [`ScaleLevel`]) extend this base with additional per-level guarantees; a
/// function that needs any pyramid binds on `PyramidLevel`, a function that
/// needs scale metadata binds on the capability it actually uses.
///
/// [`Image<P>`] implements `PyramidLevel` trivially (`as_image` returns
/// `&self`), so plain images are levels with no wrapper type. (This is also
/// why the trait deliberately has no `size` method of its own: `Image`
/// already has [`ImageView::size`], and a same-named provided method would
/// make every plain `img.size()` call ambiguous for code that imports both
/// traits.)
///
/// # Example
///
/// ```
/// use fovea::Size;
/// use fovea::image::{Image, ImageView, PyramidLevel};
/// use fovea::pixel::Mono8;
///
/// let img: Image<Mono8> = Image::zero(8, 6);
/// // An `Image` is its own pyramid level.
/// assert_eq!(img.as_image().size(), Size::new(8, 6));
/// ```
pub trait PyramidLevel {
    /// The pixel type of this level's image.
    type Pixel: Copy;

    /// Returns this level as an image reference.
    ///
    /// The level's spatial size is `as_image().size()`.
    fn as_image(&self) -> &Image<Self::Pixel>;
}

impl<P: Copy> PyramidLevel for Image<P> {
    type Pixel = P;

    #[inline]
    fn as_image(&self) -> &Image<P> {
        self
    }
}

// ─── Pyramid ────────────────────────────────────────────────────────────────

/// A multi-resolution image pyramid: a chain of levels from finest to
/// coarsest.
///
/// `Pyramid<L>` is a thin container over `Vec<L>` where **index 0 is the
/// finest (largest) level**. It is generic over the level type, not the
/// construction method — a Gaussian-built pyramid and a custom-built pyramid
/// with the same level type are interchangeable downstream.
///
/// A `Pyramid` is **never empty**: every constructor guarantees at least one
/// level, so [`finest`](Self::finest) and [`coarsest`](Self::coarsest)
/// cannot fail in correct code — their documented panics are the backstop
/// for a violated invariant, not an expected path.
///
/// # Example
///
/// ```
/// use fovea::Size;
/// use fovea::image::{Image, ImageView, Pyramid};
/// use fovea::pixel::MonoF32;
/// use fovea::transform::{Gaussian, PyramidMethod};
///
/// let img = Image::fill(16, 16, MonoF32::new(0.5));
/// let pyramid = Gaussian.build(&img, 3);
///
/// assert_eq!(pyramid.depth(), 3);
/// assert_eq!(pyramid.finest().size(), Size::new(16, 16));
/// assert_eq!(pyramid.coarsest().size(), Size::new(4, 4));
/// ```
#[derive(Clone)]
pub struct Pyramid<L: PyramidLevel> {
    levels: Vec<L>,
}

impl<L: PyramidLevel> Pyramid<L> {
    /// Creates a pyramid from pre-built levels, finest (index 0) to
    /// coarsest.
    ///
    /// This is the constructor custom
    /// [`PyramidMethod`](crate::transform::PyramidMethod) implementations
    /// use to assemble their result.
    ///
    /// The levels are validated, never reordered: they must already be
    /// sorted finest to coarsest, meaning each level's width and height are
    /// less than or equal to its predecessor's. Equal sizes are allowed —
    /// same-size levels occur in scale stacks and sub-band decompositions —
    /// so an automatic sort would be ambiguous; a wrong order is reported
    /// as an error instead. What the ordering *means* (which decomposition
    /// produced the levels) remains the builder's responsibility.
    ///
    /// # Errors
    ///
    /// - [`Error::EmptyPyramid`] if `levels` is empty — a pyramid always
    ///   contains at least one level.
    /// - [`Error::PyramidLevelOrder`] if a level is larger than its
    ///   predecessor along either axis; the error names the first
    ///   offending index and both sizes.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::image::{Image, Pyramid};
    /// use fovea::pixel::Mono8;
    ///
    /// let levels = vec![Image::<Mono8>::zero(8, 8), Image::<Mono8>::zero(4, 4)];
    /// let pyramid = Pyramid::try_from_levels(levels)?;
    /// assert_eq!(pyramid.depth(), 2);
    /// # Ok::<(), fovea::Error>(())
    /// ```
    pub fn try_from_levels(levels: Vec<L>) -> Result<Self, Error> {
        if levels.is_empty() {
            return Err(Error::EmptyPyramid);
        }
        for (index, pair) in levels.windows(2).enumerate() {
            let previous = pair[0].as_image().size();
            let current = pair[1].as_image().size();
            if current.width > previous.width || current.height > previous.height {
                return Err(Error::PyramidLevelOrder {
                    index: index + 1,
                    previous,
                    current,
                });
            }
        }
        Ok(Self { levels })
    }

    /// Returns the number of levels in the pyramid.
    pub fn depth(&self) -> usize {
        self.levels.len()
    }

    /// Returns a reference to the level at the given index.
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.depth()` (programmer bug). Use
    /// [`get`](Self::get) for a non-panicking lookup.
    pub fn level(&self, index: usize) -> &L {
        &self.levels[index]
    }

    /// Returns a reference to the level at the given index, or `None` if
    /// the index is out of bounds.
    pub fn get(&self, index: usize) -> Option<&L> {
        self.levels.get(index)
    }

    /// Returns a reference to the finest (largest) level.
    ///
    /// # Panics
    ///
    /// Panics if the pyramid is empty — impossible for pyramids built by
    /// the provided constructors, which guarantee at least one level.
    pub fn finest(&self) -> &L {
        &self.levels[0]
    }

    /// Returns a reference to the coarsest (smallest) level.
    ///
    /// # Panics
    ///
    /// Panics if the pyramid is empty — impossible for pyramids built by
    /// the provided constructors, which guarantee at least one level.
    pub fn coarsest(&self) -> &L {
        self.levels.last().expect("pyramid is empty")
    }

    /// Returns an iterator over levels from finest to coarsest.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::image::{Image, ImageView, Pyramid};
    /// use fovea::pixel::MonoF32;
    /// use fovea::transform::{Gaussian, PyramidMethod};
    ///
    /// let img = Image::fill(8, 8, MonoF32::new(1.0));
    /// let pyramid = Gaussian.build(&img, 3);
    ///
    /// let widths: Vec<usize> = pyramid.iter().map(|l| l.size().width).collect();
    /// assert_eq!(widths, [8, 4, 2]);
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = &L> {
        self.levels.iter()
    }
}

/// A Gaussian pyramid: plain images as levels, no wrapper type.
///
/// This alias exists for discoverability — the underlying type is an
/// ordinary [`Pyramid`] whose levels are [`Image<P>`], produced by the
/// [`Gaussian`](crate::transform::Gaussian) construction strategy.
pub type GaussianPyramid<P> = Pyramid<Image<P>>;

// ─── Scale capability traits ────────────────────────────────────────────────

/// A level sampled on a different grid than the base (finest) image.
///
/// Resolution and scale are orthogonal axes, so this trait carries only the
/// **sampling geometry**: how far apart this level's samples sit in the base
/// image, and where its grid origin lies. Together they form the affine
/// level→base coordinate map [`to_base`](Self::to_base) — the single place
/// the conversion is defined, so callers never hand-roll `x * 2^level`
/// lifting (which silently drops the grid-alignment offset).
///
/// All coordinates use the **pixel-center convention**: coordinate
/// `(0.0, 0.0)` is the *center* of pixel `(0, 0)`.
///
/// A plain `Pyramid<Image<P>>` used only for multi-resolution work
/// implements neither this trait nor [`ScaleLevel`] — it carries no scale
/// metadata and pays nothing for it. Level types that guarantee the
/// metadata (such as [`ScaledImage`]) opt in.
///
/// # Example
///
/// ```
/// use fovea::{CoordinateF64, PixelDistance, Sigma};
/// use fovea::image::{Decimated, Image, ScaledImage};
/// use fovea::pixel::MonoF32;
///
/// // Level 1 of a 2× pyramid built by even-sample decimation:
/// // adjacent samples are 2 base pixels apart, grid origin unshifted.
/// let level = ScaledImage::new(
///     Image::<MonoF32>::zero(4, 4),
///     PixelDistance::new(2.0),
///     CoordinateF64::new(0.0, 0.0),
///     Sigma::new(1.0),
/// );
///
/// let base = level.to_base(CoordinateF64::new(1.5, 3.0));
/// assert_eq!(base, CoordinateF64::new(3.0, 6.0));
/// ```
pub trait Decimated: PyramidLevel {
    /// Distance between two adjacent samples of this level, measured in
    /// base-image pixels. `2.0` for octave 1 of a 2× pyramid, `0.5` for an
    /// upsampled octave −1.
    ///
    /// Returned as the invariant-carrying [`PixelDistance`], so the value
    /// can flow into further constructors without re-validation; use
    /// [`PixelDistance::get`] for arithmetic.
    fn pixel_distance(&self) -> PixelDistance;

    /// Position of this level's pixel-(0,0) center in base-image
    /// coordinates. `(0.0, 0.0)` for even-sample decimation (the
    /// [`pyr_down`](crate::transform::pyr_down) convention: coarse pixel
    /// `k` samples fine pixel `2k`); `(0.5, 0.5)` for an area-averaging 2×
    /// reduction, whose coarse pixel centers sit between fine ones.
    fn origin_offset(&self) -> CoordinateF64;

    /// Lifts a level-local sub-pixel point into the base-image frame:
    /// `base = origin_offset + pixel_distance · local`.
    ///
    /// This is the mapping a detector uses to report keypoints found on
    /// this level in base-image coordinates.
    fn to_base(&self, local: CoordinateF64) -> CoordinateF64 {
        let d = self.pixel_distance().get();
        let o = self.origin_offset();
        CoordinateF64::new(o.x + d * local.x, o.y + d * local.y)
    }
}

/// A level produced by Gaussian smoothing at a known absolute scale.
///
/// This is deliberately separate from [`Decimated`]: resolution and scale
/// coincide in a textbook Gaussian pyramid but are genuinely orthogonal — a
/// band-pass residual has no meaningful σ, and a constant-resolution scale
/// stack varies σ without decimating. Each trait adds exactly one
/// guarantee.
///
/// # Example
///
/// ```
/// use fovea::{CoordinateF64, PixelDistance, Sigma};
/// use fovea::image::{Image, ScaledImage, ScaleLevel};
/// use fovea::pixel::MonoF32;
///
/// let level = ScaledImage::new(
///     Image::<MonoF32>::zero(8, 8),
///     PixelDistance::new(1.0),
///     CoordinateF64::new(0.0, 0.0),
///     Sigma::new(1.6),
/// );
/// assert_eq!(level.sigma().get(), 1.6);
/// ```
pub trait ScaleLevel: PyramidLevel {
    /// Absolute Gaussian σ, expressed in *base-image* pixels.
    ///
    /// Returned as the invariant-carrying [`Sigma`], so the value can flow
    /// into further constructors (or a
    /// [`gaussian_blur`](crate::transform::gaussian_blur)) without
    /// re-validation; use [`Sigma::get`] for arithmetic.
    fn sigma(&self) -> Sigma;
}

// ─── ScaledImage ────────────────────────────────────────────────────────────

/// An image level that carries its scale metadata: sampling geometry
/// ([`Decimated`]) and absolute Gaussian σ ([`ScaleLevel`]).
///
/// This is the thin wrapper for callers who need scale-aware pyramid
/// levels — for example to lift feature positions detected on a coarse
/// level back into base-image coordinates. A plain `Pyramid<Image<P>>`
/// carries none of this metadata and pays nothing for it.
///
/// The metadata is supplied explicitly at construction: whoever builds the
/// level states its sampling convention instead of leaving it implicit in
/// the resize kernel.
///
/// # Example
///
/// ```
/// use fovea::{CoordinateF64, PixelDistance, Sigma};
/// use fovea::image::{Decimated, Image, ImageView, ScaledImage, ScaleLevel};
/// use fovea::pixel::MonoF32;
/// use fovea::transform::pyr_down;
///
/// let base = Image::fill(16, 16, MonoF32::new(1.0));
/// let coarse: Image<MonoF32> = pyr_down(&base);
///
/// // pyr_down keeps even samples: distance 2, origin unshifted, σ = 1.
/// let level = ScaledImage::new(
///     coarse,
///     PixelDistance::new(2.0),
///     CoordinateF64::new(0.0, 0.0),
///     Sigma::new(1.0),
/// );
///
/// assert_eq!(level.size().width, 8);
/// assert_eq!(level.pixel_distance().get(), 2.0);
/// assert_eq!(level.sigma().get(), 1.0);
/// assert_eq!(
///     level.to_base(CoordinateF64::new(3.0, 4.0)),
///     CoordinateF64::new(6.0, 8.0),
/// );
/// ```
#[derive(Clone)]
pub struct ScaledImage<P: Copy> {
    image: Image<P>,
    pixel_distance: PixelDistance,
    origin_offset: CoordinateF64,
    sigma: Sigma,
}

impl<P: Copy> ScaledImage<P> {
    /// Wraps an image with its scale metadata.
    ///
    /// - `pixel_distance` — distance between adjacent samples of this
    ///   level, in base-image pixels (see [`Decimated::pixel_distance`]).
    /// - `origin_offset` — position of this level's pixel-(0,0) center in
    ///   base-image coordinates (see [`Decimated::origin_offset`]).
    /// - `sigma` — absolute Gaussian σ in base-image pixels (see
    ///   [`ScaleLevel::sigma`]).
    ///
    /// This constructor is **total**: the parameter invariants live in
    /// [`PixelDistance`] and [`Sigma`] and were checked when those values
    /// were constructed — literals via their const `new`, computed values
    /// via their `try_new`. Nothing can fail here.
    pub fn new(
        image: Image<P>,
        pixel_distance: PixelDistance,
        origin_offset: CoordinateF64,
        sigma: Sigma,
    ) -> Self {
        Self {
            image,
            pixel_distance,
            origin_offset,
            sigma,
        }
    }

    /// Returns the wrapped image.
    pub fn image(&self) -> &Image<P> {
        &self.image
    }

    /// Returns the spatial dimensions of this level.
    pub fn size(&self) -> Size {
        self.image.size()
    }

    /// Unwraps the level, discarding the scale metadata.
    pub fn into_image(self) -> Image<P> {
        self.image
    }
}

impl<P: Copy> PyramidLevel for ScaledImage<P> {
    type Pixel = P;

    #[inline]
    fn as_image(&self) -> &Image<P> {
        &self.image
    }
}

impl<P: Copy> Decimated for ScaledImage<P> {
    #[inline]
    fn pixel_distance(&self) -> PixelDistance {
        self.pixel_distance
    }

    #[inline]
    fn origin_offset(&self) -> CoordinateF64 {
        self.origin_offset
    }
}

impl<P: Copy> ScaleLevel for ScaledImage<P> {
    #[inline]
    fn sigma(&self) -> Sigma {
        self.sigma
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pixel::{Mono8, MonoF32};

    fn two_level_pyramid() -> Pyramid<Image<Mono8>> {
        Pyramid::try_from_levels(vec![
            Image::fill(8, 6, Mono8::new(10)),
            Image::fill(4, 3, Mono8::new(20)),
        ])
        .unwrap()
    }

    // ── PyramidLevel ────────────────────────────────────────────────────

    #[test]
    fn image_is_its_own_level() {
        let img: Image<Mono8> = Image::fill(5, 4, Mono8::new(7));
        let view: &Image<Mono8> = img.as_image();
        assert_eq!(view.size(), Size::new(5, 4));
        assert_eq!(view.pixel_at(0, 0), Mono8::new(7));
    }

    // ── Pyramid container ───────────────────────────────────────────────

    #[test]
    fn from_levels_and_depth() {
        let p = two_level_pyramid();
        assert_eq!(p.depth(), 2);
    }

    #[test]
    fn try_from_levels_empty_is_error() {
        let result = Pyramid::<Image<Mono8>>::try_from_levels(vec![]);
        assert_eq!(result.err().unwrap(), Error::EmptyPyramid);
    }

    #[test]
    fn try_from_levels_rejects_growing_levels() {
        // Coarsest-first is the realistic builder bug: reported, not sorted.
        let result = Pyramid::try_from_levels(vec![
            Image::fill(4, 3, Mono8::new(0)),
            Image::fill(8, 6, Mono8::new(0)),
        ]);
        assert_eq!(
            result.err().unwrap(),
            Error::PyramidLevelOrder {
                index: 1,
                previous: Size::new(4, 3),
                current: Size::new(8, 6),
            }
        );
    }

    #[test]
    fn try_from_levels_rejects_single_growing_axis() {
        // Width shrinks but height grows — still not a coarser level.
        let result = Pyramid::try_from_levels(vec![
            Image::fill(8, 6, Mono8::new(0)),
            Image::fill(4, 7, Mono8::new(0)),
        ]);
        assert_eq!(
            result.err().unwrap(),
            Error::PyramidLevelOrder {
                index: 1,
                previous: Size::new(8, 6),
                current: Size::new(4, 7),
            }
        );
    }

    #[test]
    fn try_from_levels_allows_equal_sizes() {
        // Same-size levels are legitimate (scale stacks, sub-bands).
        let p = Pyramid::try_from_levels(vec![
            Image::fill(8, 8, Mono8::new(1)),
            Image::fill(8, 8, Mono8::new(2)),
            Image::fill(4, 4, Mono8::new(3)),
        ])
        .unwrap();
        assert_eq!(p.depth(), 3);
    }

    #[test]
    fn level_returns_by_index() {
        let p = two_level_pyramid();
        assert_eq!(p.level(0).size(), Size::new(8, 6));
        assert_eq!(p.level(1).size(), Size::new(4, 3));
    }

    #[test]
    #[should_panic]
    fn level_out_of_bounds_panics() {
        let p = two_level_pyramid();
        let _ = p.level(2);
    }

    #[test]
    fn get_returns_option() {
        let p = two_level_pyramid();
        assert!(p.get(0).is_some());
        assert!(p.get(1).is_some());
        assert!(p.get(2).is_none());
    }

    #[test]
    fn finest_and_coarsest() {
        let p = two_level_pyramid();
        assert_eq!(p.finest().size(), Size::new(8, 6));
        assert_eq!(p.coarsest().size(), Size::new(4, 3));
    }

    #[test]
    fn finest_equals_coarsest_for_single_level() {
        let p = Pyramid::try_from_levels(vec![Image::fill(3, 3, Mono8::new(1))]).unwrap();
        assert_eq!(p.finest().size(), p.coarsest().size());
        assert_eq!(p.depth(), 1);
    }

    #[test]
    fn iter_goes_finest_to_coarsest() {
        let p = two_level_pyramid();
        let sizes: Vec<Size> = p.iter().map(|l| l.size()).collect();
        assert_eq!(sizes, [Size::new(8, 6), Size::new(4, 3)]);
    }

    #[test]
    fn pyramid_is_clone() {
        let p = two_level_pyramid();
        let q = p.clone();
        assert_eq!(q.depth(), p.depth());
        assert_eq!(q.level(1).pixel_at(0, 0), Mono8::new(20));
    }

    #[test]
    fn gaussian_pyramid_alias_is_plain_pyramid() {
        let p: GaussianPyramid<Mono8> = two_level_pyramid();
        assert_eq!(p.depth(), 2);
    }

    // ── ScaledImage / capability traits ─────────────────────────────────

    #[test]
    fn scaled_image_accessors() {
        let level = ScaledImage::new(
            Image::fill(4, 4, MonoF32::new(0.5)),
            PixelDistance::new(2.0),
            CoordinateF64::new(0.0, 0.0),
            Sigma::new(1.0),
        );
        assert_eq!(level.size(), Size::new(4, 4));
        assert_eq!(level.image().pixel_at(1, 1), MonoF32::new(0.5));
        assert_eq!(level.pixel_distance(), PixelDistance::new(2.0));
        assert_eq!(level.origin_offset(), CoordinateF64::new(0.0, 0.0));
        assert_eq!(level.sigma(), Sigma::new(1.0));
        let img = level.into_image();
        assert_eq!(img.size(), Size::new(4, 4));
    }

    #[test]
    fn to_base_even_sample_convention() {
        // pyr_down convention: coarse pixel k sits at base pixel 2k.
        let level = ScaledImage::new(
            Image::<MonoF32>::zero(4, 4),
            PixelDistance::new(2.0),
            CoordinateF64::new(0.0, 0.0),
            Sigma::new(1.0),
        );
        assert_eq!(
            level.to_base(CoordinateF64::new(0.0, 0.0)),
            CoordinateF64::new(0.0, 0.0)
        );
        assert_eq!(
            level.to_base(CoordinateF64::new(1.5, 3.0)),
            CoordinateF64::new(3.0, 6.0)
        );
    }

    #[test]
    fn to_base_area_average_convention() {
        // Area-averaging 2× reduction: coarse pixel centers sit between
        // fine ones — the offset is the whole point of the affine map.
        let level = ScaledImage::new(
            Image::<MonoF32>::zero(4, 4),
            PixelDistance::new(2.0),
            CoordinateF64::new(0.5, 0.5),
            Sigma::new(1.0),
        );
        assert_eq!(
            level.to_base(CoordinateF64::new(0.0, 0.0)),
            CoordinateF64::new(0.5, 0.5)
        );
        assert_eq!(
            level.to_base(CoordinateF64::new(2.0, 1.0)),
            CoordinateF64::new(4.5, 2.5)
        );
    }

    #[test]
    fn to_base_upsampled_level() {
        // An upsampled "octave −1" is an ordinary level with distance 0.5.
        let level = ScaledImage::new(
            Image::<MonoF32>::zero(16, 16),
            PixelDistance::new(0.5),
            CoordinateF64::new(0.0, 0.0),
            Sigma::new(0.8),
        );
        assert_eq!(
            level.to_base(CoordinateF64::new(6.0, 10.0)),
            CoordinateF64::new(3.0, 5.0)
        );
    }

    #[test]
    fn scaled_pyramid_composes() {
        // A pyramid of ScaledImage levels: capability metadata per level.
        let levels = vec![
            ScaledImage::new(
                Image::<MonoF32>::zero(8, 8),
                PixelDistance::new(1.0),
                CoordinateF64::new(0.0, 0.0),
                Sigma::new(0.5),
            ),
            ScaledImage::new(
                Image::<MonoF32>::zero(4, 4),
                PixelDistance::new(2.0),
                CoordinateF64::new(0.0, 0.0),
                Sigma::new(1.0),
            ),
        ];
        let p = Pyramid::try_from_levels(levels).unwrap();
        assert_eq!(p.depth(), 2);
        assert_eq!(p.level(1).pixel_distance(), PixelDistance::new(2.0));
        assert_eq!(p.level(1).sigma(), Sigma::new(1.0));
    }
}

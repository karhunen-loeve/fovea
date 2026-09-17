//! Multi-resolution image pyramids.
//!
//! [`Pyramid`] is the trait a chain of levels at decreasing resolution
//! implements, and [`LevelChain<L>`] is its ordinary container: generic over
//! the level type `L` rather than over the construction method, so the
//! structural truth ("what does a level contain?") lives in the type, while
//! the construction strategy
//! ([`PyramidMethod`](crate::transform::PyramidMethod)) is consumed at build
//! time and not stored — the same pattern as
//! [`ResizeMethod`](crate::transform::ResizeMethod) and
//! [`ConvertPixel`](crate::transform::ConvertPixel).
//!
//! [`Dyadic<C>`] wraps any container whose neighbouring levels halve and is
//! what turns the lift back up into
//! [`expand`](crate::image::Dyadic::expand), which needs no target size
//! because the container already holds one.
//!
//! [`Image<P>`] implements [`PyramidLevel`] directly, so a level can be a
//! plain image with no wrapper cost. Levels that answer for their own
//! geometry opt in via the [`Decimated`] / [`ScaleLevel`] capability traits:
//! [`PlacedImage<P>`] knows where its samples sit, [`ScaledImage<P>`] knows
//! that and its absolute σ. The two aliases [`PlacedPyramid<P>`] and
//! [`ScaledPyramid<P>`] name the chains the two Gaussian builders return.

use crate::error::Error;
use crate::image::{Image, ImageView, RasterImage};
use crate::{CoordinateF64, PixelDistance, Sigma, Size};

// ─── PyramidLevel ────────────────────────────────────────────────────────────

/// Base trait for all pyramid level types.
///
/// Every level is a [`RasterImage`] and can additionally hand out the owned
/// image behind it. That supertrait is the whole reason a pyramid level can
/// be passed to code that never heard of pyramids: `level.size()`,
/// `level.pixel_at(x, y)` and `pyr_down(level)` all work directly, because
/// the level really is an image with extra facts attached.
///
/// Capability traits ([`Decimated`], [`ScaleLevel`]) extend this base with
/// additional per-level guarantees; a function that needs any pyramid binds
/// on `PyramidLevel`, a function that needs scale metadata binds on the
/// capability it actually uses.
///
/// [`Image<P>`] implements `PyramidLevel` trivially (`as_image` returns
/// `&self`), so plain images are levels with no wrapper type.
///
/// The supertrait **adds no requirement**: `as_image` already forces an
/// owned, contiguous, host-memory `Image<P>`, so every level that can exist
/// is a `RasterImage` already. It writes an existing constraint down rather
/// than imposing a new one, and in exchange there is exactly one `Pixel`
/// associated type instead of two that generic callers had to prove equal
/// by hand.
///
/// # Example
///
/// ```
/// use fovea::Size;
/// use fovea::image::{Image, ImageView, PyramidLevel};
/// use fovea::pixel::Mono8;
///
/// let img: Image<Mono8> = Image::zero(8, 6);
/// // An `Image` is its own pyramid level, and answers as an image.
/// assert_eq!(img.size(), Size::new(8, 6));
/// assert_eq!(img.as_image().size(), Size::new(8, 6));
/// ```
pub trait PyramidLevel: RasterImage {
    /// Returns this level as an image reference.
    ///
    /// Use this where the owned `Image<P>` itself is needed. For the
    /// level's size and pixels, the [`ImageView`] methods inherited through
    /// [`RasterImage`] answer directly.
    fn as_image(&self) -> &Image<Self::Pixel>;
}

impl<P: Copy> PyramidLevel for Image<P> {
    #[inline]
    fn as_image(&self) -> &Image<P> {
        self
    }
}

// ─── Pyramid ────────────────────────────────────────────────────────────────

/// A multi-resolution image pyramid: a chain of levels from finest to
/// coarsest, where **index 0 is the finest (largest) level**.
///
/// `Pyramid` is a trait, not a container: [`LevelChain<L>`] is the ordinary
/// `Vec`-backed implementation, [`Dyadic<C>`] wraps any implementor and adds
/// the halving guarantee, and a backend that pages levels in from disk or
/// decides lazily which levels exist implements the same two methods without
/// inheriting either representation. It is the same split as [`ImageView`]
/// and [`Image<P>`] one level down.
///
/// Implementors owe [`depth`](Self::depth) and [`get`](Self::get); the four
/// accessors below are provided in terms of them and may be overridden where
/// a container has a faster path.
///
/// # Contract
///
/// Implementors must uphold all of the following. These are logical
/// requirements, not memory-safety ones: breaking them makes the provided
/// methods panic or return misleading values, and the panic messages name
/// the clause that was broken.
///
/// 1. **Non-empty.** `depth() >= 1`. A pyramid always has at least its base
///    level, which is what makes [`finest`](Self::finest) and
///    [`coarsest`](Self::coarsest) total.
/// 2. **Agreement.** `get(index).is_some()` if and only if `index < depth()`.
/// 3. **Order.** Index `0` is the finest level, and neither dimension grows
///    as the index increases. Equal sizes are legal and deliberate: scale
///    stacks and sub-band decompositions need them.
/// 4. **Stability.** Repeated `get` calls with the same index observe the
///    same level. A container that computes levels on demand must cache
///    them; the signature already forces this, since a reference cannot
///    outlive a temporary.
///
/// The constructors this crate provides ([`LevelChain::try_from_levels`],
/// [`Dyadic::try_new`]) uphold every clause, so the panics are the backstop
/// for a violated invariant, not an expected path.
///
/// Note that `iter` returns `impl Iterator`, so `Pyramid` is deliberately
/// **not** object-safe: there is no `dyn Pyramid`.
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
pub trait Pyramid {
    /// The level type this pyramid hands out.
    type Level: PyramidLevel;

    /// Returns the number of levels in the pyramid.
    ///
    /// At least 1, per contract clause 1.
    fn depth(&self) -> usize;

    /// Returns a reference to the level at the given index, or `None` if
    /// the index is out of bounds.
    ///
    /// `Some` exactly for `index < depth()`, per contract clause 2.
    fn get(&self, index: usize) -> Option<&Self::Level>;

    /// Returns a reference to the level at the given index.
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.depth()` (programmer bug). Use
    /// [`get`](Self::get) for a non-panicking lookup. A panic for an index
    /// below `depth` means the implementor broke contract clause 2.
    fn level(&self, index: usize) -> &Self::Level {
        let depth = self.depth();
        self.get(index).unwrap_or_else(|| {
            panic!("no pyramid level {index} at depth {depth} (contract clause 2: agreement)")
        })
    }

    /// Returns a reference to the finest (largest) level.
    ///
    /// # Panics
    ///
    /// Panics if the pyramid is empty, which contract clause 1 forbids.
    fn finest(&self) -> &Self::Level {
        self.get(0)
            .expect("pyramid is empty (contract clause 1: non-empty)")
    }

    /// Returns a reference to the coarsest (smallest) level.
    ///
    /// # Panics
    ///
    /// Panics if the pyramid is empty, which contract clause 1 forbids.
    fn coarsest(&self) -> &Self::Level {
        let last = self
            .depth()
            .checked_sub(1)
            .expect("pyramid is empty (contract clause 1: non-empty)");
        self.get(last)
            .expect("coarsest index is below depth (contract clause 2: agreement)")
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
    fn iter(&self) -> impl Iterator<Item = &Self::Level> {
        (0..self.depth()).map(move |index| self.level(index))
    }
}

// ─── LevelChain ─────────────────────────────────────────────────────────────

/// The ordinary pyramid container: a `Vec` of levels, finest first.
///
/// `LevelChain<L>` is generic over the level type, not the construction
/// method, so a Gaussian-built chain and a custom-built chain with the same
/// level type are interchangeable downstream. It is the [`Pyramid`]
/// implementation that owns its levels contiguously; the accessors come from
/// that trait, so callers import it.
///
/// A `LevelChain` is **never empty** and never grows from one level to the
/// next: its only constructor validates both, which is what makes contract
/// clauses 1 and 3 hold for it by construction.
#[derive(Clone, Debug)]
pub struct LevelChain<L: PyramidLevel> {
    levels: Vec<L>,
}

impl<L: PyramidLevel> LevelChain<L> {
    /// Creates a level chain from pre-built levels, finest (index 0) to
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
    /// The chain is deliberately **not** checked for halving: that is a
    /// stricter relation, and the type that carries it is [`Dyadic`].
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
    /// use fovea::image::{Image, LevelChain, Pyramid};
    /// use fovea::pixel::Mono8;
    ///
    /// let levels = vec![Image::<Mono8>::zero(8, 8), Image::<Mono8>::zero(4, 4)];
    /// let pyramid = LevelChain::try_from_levels(levels)?;
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
}

impl<L: PyramidLevel> Pyramid for LevelChain<L> {
    type Level = L;

    fn depth(&self) -> usize {
        self.levels.len()
    }

    fn get(&self, index: usize) -> Option<&L> {
        self.levels.get(index)
    }

    // Overridden: the levels are contiguous, so the slice iterator beats
    // indexing through `get` once per level.
    fn iter(&self) -> impl Iterator<Item = &L> {
        self.levels.iter()
    }
}

// ─── Dyadic ─────────────────────────────────────────────────────────────────

/// A pyramid whose neighbouring levels halve: every level's size is its
/// predecessor's ceiling-halved size, the relation
/// [`pyr_down`](crate::transform::pyr_down) produces.
///
/// `Dyadic<C>` is an adapter over any [`Pyramid`], not a container of its
/// own, so a memory-mapped or lazily paged container gains the guarantee by
/// being wrapped rather than by being reimplemented. It is itself a
/// [`Pyramid`], one with an extra guarantee, so it passes everywhere a
/// `C: Pyramid` bound does.
///
/// That guarantee is what makes [`expand`](Self::expand) total: the parent's
/// size is already in the container, so the lift back up cannot be handed a
/// size this level is not the reduction of. There is no way to hold a `Dyadic`
/// without the relation having been established, because
/// [`try_new`](Self::try_new) is the only public constructor and it checks.
///
/// "Dyadic" is the standard term for the factor of two (dyadic pyramid,
/// dyadic wavelet transform); it names the relation between levels rather
/// than an operation performed on them, which is why an upsampled level
/// chain can still be dyadic.
///
/// # Example
///
/// ```
/// use fovea::image::{Dyadic, Image, ImageView, LevelChain, Pyramid};
/// use fovea::pixel::MonoF32;
///
/// let levels = vec![
///     Image::fill(9, 7, MonoF32::new(0.25)),
///     Image::fill(5, 4, MonoF32::new(0.25)),
/// ];
/// let pyramid = Dyadic::try_new(LevelChain::try_from_levels(levels)?)?;
///
/// // The odd parent size is recovered without the caller naming it.
/// let raised: Image<MonoF32> = pyramid.expand(1).expect("level 1 has a parent");
/// assert_eq!(raised.size(), pyramid.level(0).size());
/// # Ok::<(), fovea::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct Dyadic<C: Pyramid>(C);

impl<C: Pyramid> Dyadic<C> {
    /// Wraps a pyramid of unknown provenance, checking the halving relation.
    ///
    /// Every neighbouring pair must satisfy `child = ceil(parent / 2)` along
    /// both axes. That is the relation `pyr_down` produces, under which an
    /// odd and an even parent dimension map onto the same child. A
    /// single-level pyramid has no pair to violate and is always dyadic.
    ///
    /// # Errors
    ///
    /// [`Error::NotDyadic`] if two neighbours do not halve; the error names
    /// the first offending index and both sizes.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::image::{Dyadic, Image, LevelChain};
    /// use fovea::pixel::Mono8;
    ///
    /// // Non-growing, so the chain is legal, but 30 is not half of 100.
    /// let levels = vec![Image::<Mono8>::zero(100, 68), Image::<Mono8>::zero(30, 34)];
    /// let chain = LevelChain::try_from_levels(levels)?;
    /// assert!(Dyadic::try_new(chain).is_err());
    /// # Ok::<(), fovea::Error>(())
    /// ```
    pub fn try_new(inner: C) -> Result<Self, Error> {
        for index in 1..inner.depth() {
            let parent = inner.level(index - 1).as_image().size();
            let child = inner.level(index).as_image().size();
            if child.width != halved(parent.width) || child.height != halved(parent.height) {
                return Err(Error::NotDyadic {
                    index,
                    parent,
                    child,
                });
            }
        }
        Ok(Self(inner))
    }

    /// Wraps a pyramid whose halving the caller has already established.
    ///
    /// Crate-internal on purpose. An unchecked constructor is warranted only
    /// where the caller has already established the invariant: the builders
    /// that compose `pyr_down` produce the halving relation by construction,
    /// so re-deriving it from the sizes afterwards would check the arithmetic
    /// of the very function that produced them. Callers outside the crate
    /// have no such proof and go through [`try_new`](Self::try_new).
    pub(crate) fn new_unchecked(inner: C) -> Self {
        Self(inner)
    }

    /// Returns the wrapped pyramid, dropping the halving guarantee.
    pub fn into_inner(self) -> C {
        self.0
    }
}

impl<C: Pyramid> Pyramid for Dyadic<C> {
    type Level = C::Level;

    fn depth(&self) -> usize {
        self.0.depth()
    }

    fn get(&self, index: usize) -> Option<&C::Level> {
        self.0.get(index)
    }

    fn iter(&self) -> impl Iterator<Item = &C::Level> {
        self.0.iter()
    }
}

/// The dyadic relation between neighbouring level sizes.
///
/// `pyr_down` maps a dimension onto `ceil(dim / 2)`, so an odd and an even
/// parent dimension land on the same child. That is why the way back up
/// needs a target, and why a container that already holds one does not.
fn halved(dim: usize) -> usize {
    dim / 2 + dim % 2
}

/// A dyadic pyramid of levels that know their sampling grid.
///
/// What [`Gaussian`](crate::transform::Gaussian) builds. The name states a
/// capability rather than a construction method, which is why it is also
/// the right name for an imported GPU mipmap: both know where their samples
/// sit, and neither claims a blur.
pub type PlacedPyramid<P> = Dyadic<LevelChain<PlacedImage<P>>>;

/// A dyadic pyramid of levels that know their sampling grid **and** their
/// absolute σ.
///
/// What [`ScaledGaussian`](crate::transform::ScaledGaussian) builds, reached
/// through
/// [`Gaussian::assuming_input_sigma`](crate::transform::Gaussian::assuming_input_sigma).
/// The extra capability is [`ScaleLevel`], so a function that binds it
/// accepts this and not [`PlacedPyramid`].
pub type ScaledPyramid<P> = Dyadic<LevelChain<ScaledImage<P>>>;

// ─── Scale capability traits ────────────────────────────────────────────────

/// A level sampled on a different grid than the base (finest) image.
///
/// Resolution and scale are orthogonal axes, so this trait carries only the
/// **sampling geometry**: how far apart this level's samples sit in the base
/// image, and where its grid origin lies. Together they form the affine
/// level↔base coordinate map — [`to_base`](Self::to_base) out of the level,
/// [`to_local`](Self::to_local) back into it — the single place the
/// conversion is defined, so callers never hand-roll `x * 2^level` lifting
/// (which silently drops the grid-alignment offset).
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
/// use fovea::CoordinateF64;
/// use fovea::image::{Decimated, Image, OriginOffset, ScaledImage};
/// use fovea::pixel::MonoF32;
/// use fovea::{pixel_distance, sigma};
///
/// // Level 1 of a 2× pyramid built by even-sample decimation:
/// // adjacent samples are 2 base pixels apart, grid origin unshifted.
/// let level = ScaledImage::new(
///     Image::<MonoF32>::zero(4, 4),
///     pixel_distance!(2.0),
///     OriginOffset::ZERO,
///     sigma!(1.0),
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

    /// Projects a base-image point back into this level's local
    /// coordinates: `local = (base − origin_offset) / pixel_distance`.
    ///
    /// The exact inverse of [`to_base`](Self::to_base), and the other half
    /// of the same affine map. Detection lifts *out* of a level; anything
    /// that samples *into* one — a descriptor reading a patch around a
    /// keypoint whose position is in base-image coordinates — comes back
    /// through here, so neither direction is re-derived at a call site.
    ///
    /// The division is total: [`PixelDistance`] cannot be zero.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::CoordinateF64;
    /// use fovea::image::{Decimated, Image, OriginOffset, ScaledImage};
    /// use fovea::pixel::MonoF32;
    /// use fovea::{pixel_distance, sigma};
    ///
    /// let level = ScaledImage::new(
    ///     Image::<MonoF32>::zero(4, 4),
    ///     pixel_distance!(2.0),
    ///     OriginOffset::new(0.5, 0.5).unwrap(),
    ///     sigma!(1.0),
    /// );
    ///
    /// let local = CoordinateF64::new(1.5, 3.0);
    /// assert_eq!(level.to_local(level.to_base(local)), local);
    /// ```
    fn to_local(&self, base: CoordinateF64) -> CoordinateF64 {
        let d = self.pixel_distance().get();
        let o = self.origin_offset();
        CoordinateF64::new((base.x - o.x) / d, (base.y - o.y) / d)
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
/// use fovea::CoordinateF64;
/// use fovea::image::{Image, OriginOffset, ScaleLevel, ScaledImage};
/// use fovea::pixel::MonoF32;
/// use fovea::{pixel_distance, sigma};
///
/// let level = ScaledImage::new(
///     Image::<MonoF32>::zero(8, 8),
///     pixel_distance!(1.0),
///     OriginOffset::ZERO,
///     sigma!(1.6),
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
/// use fovea::CoordinateF64;
/// use fovea::image::{Decimated, Image, ImageView, OriginOffset, ScaleLevel, ScaledImage};
/// use fovea::pixel::MonoF32;
/// use fovea::{pixel_distance, sigma};
/// use fovea::transform::pyr_down;
///
/// let base = Image::fill(16, 16, MonoF32::new(1.0));
/// let coarse: Image<MonoF32> = pyr_down(&base);
///
/// // pyr_down keeps even samples: distance 2, origin unshifted, σ = 1.
/// let level = ScaledImage::new(
///     coarse,
///     pixel_distance!(2.0),
///     OriginOffset::ZERO,
///     sigma!(1.0),
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
#[derive(Clone, Debug)]
pub struct ScaledImage<P: Copy> {
    image: Image<P>,
    pixel_distance: PixelDistance,
    origin_offset: OriginOffset,
    sigma: Sigma,
}

/// A level's pixel-(0,0) center in base-image coordinates: **finite** along
/// both axes.
///
/// The invariant carrier for [`ScaledImage`]'s origin, in the same family
/// as [`PixelDistance`] and [`Sigma`](crate::Sigma): a NaN or infinite
/// origin would poison every [`Decimated::to_base`] lift while the
/// constructor's totality claim promised nothing can fail, so the claim is
/// carried by the type instead. Negative offsets are valid — a level padded
/// past its base's origin sits at one.
///
/// [`ZERO`](Self::ZERO) is the unshifted origin, which is what `pyr_down`
/// levels have and most call sites want.
///
/// # Example
///
/// ```
/// use fovea::image::OriginOffset;
///
/// const UNSHIFTED: OriginOffset = OriginOffset::ZERO;
/// assert_eq!(UNSHIFTED.get().x, 0.0);
///
/// // Literals: checked at compile time in a const context.
/// const HALF: OriginOffset = OriginOffset::new(0.5, 0.5).unwrap();
/// assert_eq!(HALF.get().y, 0.5);
///
/// assert!(OriginOffset::new(f64::NAN, 0.0).is_none());
/// assert!(OriginOffset::try_new(f64::INFINITY, 0.0).is_err());
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OriginOffset(CoordinateF64);

impl OriginOffset {
    /// The unshifted origin: this level's pixel (0, 0) sits exactly on the
    /// base image's.
    pub const ZERO: Self = Self(CoordinateF64 { x: 0.0, y: 0.0 });

    /// Creates an origin offset, returning `None` if either component is
    /// NaN or infinite.
    ///
    /// `const`, so binding the result to a `const` item checks literals at
    /// compile time. For computed values use [`try_new`](Self::try_new).
    #[must_use]
    pub const fn new(x: f64, y: f64) -> Option<Self> {
        if x.is_finite() && y.is_finite() {
            Some(Self(CoordinateF64 { x, y }))
        } else {
            None
        }
    }

    /// Creates an origin offset from computed values, validating them.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidParameter`] if either component is NaN or
    /// infinite.
    pub fn try_new(x: f64, y: f64) -> Result<Self, Error> {
        Self::new(x, y).ok_or_else(|| {
            Error::InvalidParameter(format!(
                "origin offset must be finite along both axes, got ({x}, {y})"
            ))
        })
    }

    /// Returns the offset as a coordinate in base-image units.
    #[inline]
    #[must_use]
    pub const fn get(self) -> CoordinateF64 {
        self.0
    }
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
    /// This constructor is **total**: every parameter invariant lives in
    /// its type — [`PixelDistance`], [`OriginOffset`] and [`Sigma`] — and
    /// was checked when those values were constructed, literals via their
    /// const `new`, computed values via their `try_new`. Nothing can fail
    /// here.
    pub fn new(
        image: Image<P>,
        pixel_distance: PixelDistance,
        origin_offset: OriginOffset,
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

    /// Unwraps the level, discarding the scale metadata.
    pub fn into_image(self) -> Image<P> {
        self.image
    }
}

impl<P: Copy> ImageView for ScaledImage<P> {
    type Pixel = P;

    #[inline]
    fn size(&self) -> Size {
        self.image.size()
    }

    #[inline]
    fn pixel_at(&self, x: usize, y: usize) -> P {
        self.image.pixel_at(x, y)
    }
}

impl<P: Copy> RasterImage for ScaledImage<P> {
    #[inline]
    fn row(&self, y: usize) -> &[P] {
        self.image.row(y)
    }
}

impl<P: Copy> PyramidLevel for ScaledImage<P> {
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
        self.origin_offset.get()
    }
}

impl<P: Copy> ScaleLevel for ScaledImage<P> {
    #[inline]
    fn sigma(&self) -> Sigma {
        self.sigma
    }
}

// ─── PlacedImage ────────────────────────────────────────────────────────────

/// An image level that knows **where it sits** but not how blurred it is:
/// sampling geometry ([`Decimated`]) without an absolute σ.
///
/// This is the level for the case the library could not express before: a
/// grid that is known and a blur that is not. An imported GPU mipmap is
/// exactly that, and so is the output of [`Gaussian`](crate::transform::Gaussian),
/// which halves on a known grid but is told nothing about how sharp its
/// input was. Both used to force an invented `sigma!(1.0)`.
///
/// It pairs with [`ScaledImage<P>`] as a strict extension. `PlacedImage`
/// knows where it sits; `ScaledImage` knows where it sits *and* how blurred
/// it is, and its constructor is this one plus a σ:
///
/// | Caller knows | level type |
/// |---|---|
/// | pixels only | [`Image<P>`] |
/// | pixels + grid | `PlacedImage<P>` |
/// | pixels + grid + blur | [`ScaledImage<P>`] |
///
/// Deliberately **not** [`ScaleLevel`], and deliberately not an
/// `Option<Sigma>` on one shared type. A function that binds
/// `L: ScaleLevel` is then *unreachable* from a pyramid nobody named an
/// assumption for: asking for a σ that was never established is a compile
/// error rather than a plausible `None` one `unwrap_or` away from the very
/// failure this distinction exists to remove.
///
/// # Example
///
/// ```
/// use fovea::CoordinateF64;
/// use fovea::image::{Decimated, Image, ImageView, OriginOffset, PlacedImage};
/// use fovea::pixel::MonoF32;
/// use fovea::pixel_distance;
/// use fovea::transform::pyr_down;
///
/// let base = Image::fill(16, 16, MonoF32::new(1.0));
/// let coarse: Image<MonoF32> = pyr_down(&base);
///
/// // pyr_down keeps even samples: distance 2, origin unshifted.
/// let level = PlacedImage::new(coarse, pixel_distance!(2.0), OriginOffset::ZERO);
///
/// assert_eq!(level.size().width, 8);
/// assert_eq!(level.pixel_distance().get(), 2.0);
/// assert_eq!(
///     level.to_base(CoordinateF64::new(3.0, 4.0)),
///     CoordinateF64::new(6.0, 8.0),
/// );
/// ```
#[derive(Clone, Debug)]
pub struct PlacedImage<P: Copy> {
    image: Image<P>,
    pixel_distance: PixelDistance,
    origin_offset: OriginOffset,
}

impl<P: Copy> PlacedImage<P> {
    /// Wraps an image with its sampling geometry.
    ///
    /// - `pixel_distance`: distance between adjacent samples of this level,
    ///   in base-image pixels (see [`Decimated::pixel_distance`]).
    /// - `origin_offset`: position of this level's pixel-(0,0) center in
    ///   base-image coordinates (see [`Decimated::origin_offset`]).
    ///
    /// This constructor is **total**, for the same reason
    /// [`ScaledImage::new`] is: every parameter invariant lives in its type
    /// and was checked when that value was constructed. Nothing can fail
    /// here.
    pub fn new(
        image: Image<P>,
        pixel_distance: PixelDistance,
        origin_offset: OriginOffset,
    ) -> Self {
        Self {
            image,
            pixel_distance,
            origin_offset,
        }
    }

    /// Returns the wrapped image.
    pub fn image(&self) -> &Image<P> {
        &self.image
    }

    /// Unwraps the level, discarding the sampling geometry.
    pub fn into_image(self) -> Image<P> {
        self.image
    }

    /// Adds an absolute σ, turning this level into a [`ScaledImage`].
    ///
    /// The geometry carries over unchanged; only the claim about blur is
    /// new, which is why it has to be supplied here rather than derived.
    pub fn with_sigma(self, sigma: Sigma) -> ScaledImage<P> {
        ScaledImage::new(self.image, self.pixel_distance, self.origin_offset, sigma)
    }
}

impl<P: Copy> ImageView for PlacedImage<P> {
    type Pixel = P;

    #[inline]
    fn size(&self) -> Size {
        self.image.size()
    }

    #[inline]
    fn pixel_at(&self, x: usize, y: usize) -> P {
        self.image.pixel_at(x, y)
    }
}

impl<P: Copy> RasterImage for PlacedImage<P> {
    #[inline]
    fn row(&self, y: usize) -> &[P] {
        self.image.row(y)
    }
}

impl<P: Copy> PyramidLevel for PlacedImage<P> {
    #[inline]
    fn as_image(&self) -> &Image<P> {
        &self.image
    }
}

impl<P: Copy> Decimated for PlacedImage<P> {
    #[inline]
    fn pixel_distance(&self) -> PixelDistance {
        self.pixel_distance
    }

    #[inline]
    fn origin_offset(&self) -> CoordinateF64 {
        self.origin_offset.get()
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pixel::{Mono8, MonoF32};
    use crate::{pixel_distance, sigma};

    fn two_level_pyramid() -> LevelChain<Image<Mono8>> {
        LevelChain::try_from_levels(vec![
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
        let result = LevelChain::<Image<Mono8>>::try_from_levels(vec![]);
        assert_eq!(result.err().unwrap(), Error::EmptyPyramid);
    }

    #[test]
    fn try_from_levels_rejects_growing_levels() {
        // Coarsest-first is the realistic builder bug: reported, not sorted.
        let result = LevelChain::try_from_levels(vec![
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
        let result = LevelChain::try_from_levels(vec![
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
        let p = LevelChain::try_from_levels(vec![
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
    #[should_panic(expected = "no pyramid level 2 at depth 2")]
    fn level_out_of_bounds_panics() {
        let p = two_level_pyramid();
        let _ = p.level(2);
    }

    /// A deliberately contract-breaking implementor: it claims no levels at
    /// all, which clause 1 forbids. The provided accessors are supposed to
    /// panic, and to say which clause was broken while they do it.
    struct NoLevels;

    impl Pyramid for NoLevels {
        type Level = Image<Mono8>;

        fn depth(&self) -> usize {
            0
        }

        fn get(&self, _index: usize) -> Option<&Image<Mono8>> {
            None
        }
    }

    #[test]
    #[should_panic(expected = "contract clause 1")]
    fn finest_names_the_clause_an_empty_implementor_broke() {
        let _ = NoLevels.finest();
    }

    #[test]
    #[should_panic(expected = "contract clause 1")]
    fn coarsest_names_the_clause_an_empty_implementor_broke() {
        let _ = NoLevels.coarsest();
    }

    /// The smallest legal implementor: `depth` and `get` only, so the four
    /// provided accessors are the ones under test, including `iter`, whose
    /// `impl Iterator` return borrows `&self` and is the one shape D2e
    /// wanted compiled rather than assumed.
    struct MinimalPyramid(Vec<Image<Mono8>>);

    impl Pyramid for MinimalPyramid {
        type Level = Image<Mono8>;

        fn depth(&self) -> usize {
            self.0.len()
        }

        fn get(&self, index: usize) -> Option<&Image<Mono8>> {
            self.0.get(index)
        }
    }

    #[test]
    fn the_provided_accessors_derive_from_depth_and_get() {
        let p = MinimalPyramid(vec![
            Image::fill(8, 6, Mono8::new(10)),
            Image::fill(4, 3, Mono8::new(20)),
        ]);
        assert_eq!(p.level(1).size(), Size::new(4, 3));
        assert_eq!(p.finest().size(), Size::new(8, 6));
        assert_eq!(p.coarsest().size(), Size::new(4, 3));
        let widths: Vec<usize> = p.iter().map(|l| l.size().width).collect();
        assert_eq!(widths, [8, 4]);
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
        let p = LevelChain::try_from_levels(vec![Image::fill(3, 3, Mono8::new(1))]).unwrap();
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
    fn the_capability_aliases_name_dyadic_chains() {
        let placed: PlacedPyramid<Mono8> = Dyadic::try_new(
            LevelChain::try_from_levels(vec![
                PlacedImage::new(
                    Image::fill(8, 6, Mono8::new(10)),
                    pixel_distance!(1.0),
                    OriginOffset::ZERO,
                ),
                PlacedImage::new(
                    Image::fill(4, 3, Mono8::new(20)),
                    pixel_distance!(2.0),
                    OriginOffset::ZERO,
                ),
            ])
            .unwrap(),
        )
        .unwrap();
        assert_eq!(placed.depth(), 2);
        assert_eq!(placed.level(1).pixel_distance().get(), 2.0);

        let scaled: ScaledPyramid<Mono8> = Dyadic::try_new(
            LevelChain::try_from_levels(vec![ScaledImage::new(
                Image::fill(8, 6, Mono8::new(10)),
                pixel_distance!(1.0),
                OriginOffset::ZERO,
                sigma!(0.5),
            )])
            .unwrap(),
        )
        .unwrap();
        assert_eq!(scaled.finest().sigma().get(), 0.5);
    }

    // ── Dyadic ──────────────────────────────────────────────────────────

    #[test]
    fn try_new_accepts_a_halving_chain() {
        let pyramid = Dyadic::try_new(two_level_pyramid()).unwrap();
        assert_eq!(pyramid.depth(), 2);
        assert_eq!(pyramid.level(1).size(), Size::new(4, 3));
    }

    #[test]
    fn try_new_accepts_an_odd_parent() {
        // 9 and 10 both reduce to 5: ceiling halving, not exact division.
        let chain = LevelChain::try_from_levels(vec![
            Image::fill(9, 7, Mono8::new(1)),
            Image::fill(5, 4, Mono8::new(1)),
            Image::fill(3, 2, Mono8::new(1)),
        ])
        .unwrap();
        assert!(Dyadic::try_new(chain).is_ok());
    }

    #[test]
    fn try_new_accepts_a_single_level() {
        let chain = LevelChain::try_from_levels(vec![Image::fill(7, 5, Mono8::new(1))]).unwrap();
        assert!(Dyadic::try_new(chain).is_ok());
    }

    #[test]
    fn try_new_rejects_equal_size_neighbours() {
        // A legal chain that is nevertheless not dyadic: `try_from_levels`
        // admits equal sizes on purpose.
        let chain = LevelChain::try_from_levels(vec![
            Image::fill(8, 6, Mono8::new(1)),
            Image::fill(8, 6, Mono8::new(1)),
        ])
        .unwrap();
        assert_eq!(
            Dyadic::try_new(chain).err().unwrap(),
            Error::NotDyadic {
                index: 1,
                parent: Size::new(8, 6),
                child: Size::new(8, 6),
            }
        );
    }

    #[test]
    fn try_new_rejects_a_chain_that_shrinks_without_halving() {
        let chain = LevelChain::try_from_levels(vec![
            Image::fill(100, 68, Mono8::new(1)),
            Image::fill(30, 34, Mono8::new(1)),
        ])
        .unwrap();
        assert!(matches!(
            Dyadic::try_new(chain),
            Err(Error::NotDyadic { index: 1, .. })
        ));
    }

    #[test]
    fn try_new_names_the_first_offending_index() {
        let chain = LevelChain::try_from_levels(vec![
            Image::fill(16, 16, Mono8::new(1)),
            Image::fill(8, 8, Mono8::new(1)),
            Image::fill(3, 4, Mono8::new(1)),
        ])
        .unwrap();
        assert!(matches!(
            Dyadic::try_new(chain),
            Err(Error::NotDyadic { index: 2, .. })
        ));
    }

    #[test]
    fn dyadic_forwards_the_pyramid_accessors() {
        let pyramid = Dyadic::try_new(two_level_pyramid()).unwrap();
        assert_eq!(pyramid.finest().size(), Size::new(8, 6));
        assert_eq!(pyramid.coarsest().size(), Size::new(4, 3));
        assert!(pyramid.get(2).is_none());
        let widths: Vec<usize> = pyramid.iter().map(|l| l.size().width).collect();
        assert_eq!(widths, [8, 4]);
    }

    #[test]
    fn into_inner_returns_the_wrapped_container() {
        let pyramid = Dyadic::try_new(two_level_pyramid()).unwrap();
        let chain: LevelChain<Image<Mono8>> = pyramid.into_inner();
        assert_eq!(chain.depth(), 2);
    }
    // ── ScaledImage / capability traits ─────────────────────────────────

    #[test]
    fn scaled_image_accessors() {
        let level = ScaledImage::new(
            Image::fill(4, 4, MonoF32::new(0.5)),
            pixel_distance!(2.0),
            OriginOffset::ZERO,
            sigma!(1.0),
        );
        assert_eq!(level.size(), Size::new(4, 4));
        assert_eq!(level.image().pixel_at(1, 1), MonoF32::new(0.5));
        assert_eq!(level.pixel_distance(), pixel_distance!(2.0));
        assert_eq!(level.origin_offset(), CoordinateF64::new(0.0, 0.0));
        assert_eq!(level.sigma(), sigma!(1.0));
        let img = level.into_image();
        assert_eq!(img.size(), Size::new(4, 4));
    }

    #[test]
    fn to_base_even_sample_convention() {
        // pyr_down convention: coarse pixel k sits at base pixel 2k.
        let level = ScaledImage::new(
            Image::<MonoF32>::zero(4, 4),
            pixel_distance!(2.0),
            OriginOffset::ZERO,
            sigma!(1.0),
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
            pixel_distance!(2.0),
            OriginOffset::new(0.5, 0.5).unwrap(),
            sigma!(1.0),
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
            pixel_distance!(0.5),
            OriginOffset::ZERO,
            sigma!(0.8),
        );
        assert_eq!(
            level.to_base(CoordinateF64::new(6.0, 10.0)),
            CoordinateF64::new(3.0, 5.0)
        );
    }

    #[test]
    fn to_local_inverts_to_base() {
        // Round-trip on the offset convention, where a ratio-only inverse
        // would be wrong: the offset must be subtracted before dividing.
        let level = ScaledImage::new(
            Image::<MonoF32>::zero(4, 4),
            pixel_distance!(2.0),
            OriginOffset::new(0.5, 0.5).unwrap(),
            sigma!(1.0),
        );
        for local in [
            CoordinateF64::new(0.0, 0.0),
            CoordinateF64::new(1.5, 3.0),
            CoordinateF64::new(3.25, 0.75),
        ] {
            assert_eq!(level.to_local(level.to_base(local)), local);
        }
    }

    #[test]
    fn to_local_projects_base_coordinates_into_the_level() {
        let level = ScaledImage::new(
            Image::<MonoF32>::zero(4, 4),
            pixel_distance!(2.0),
            OriginOffset::ZERO,
            sigma!(1.0),
        );
        assert_eq!(
            level.to_local(CoordinateF64::new(6.0, 8.0)),
            CoordinateF64::new(3.0, 4.0)
        );
        // Base points between this level's samples land on fractions.
        assert_eq!(
            level.to_local(CoordinateF64::new(3.0, 1.0)),
            CoordinateF64::new(1.5, 0.5)
        );
    }

    #[test]
    fn to_local_on_an_upsampled_level() {
        let level = ScaledImage::new(
            Image::<MonoF32>::zero(16, 16),
            pixel_distance!(0.5),
            OriginOffset::ZERO,
            sigma!(0.8),
        );
        assert_eq!(
            level.to_local(CoordinateF64::new(3.0, 5.0)),
            CoordinateF64::new(6.0, 10.0)
        );
    }

    #[test]
    fn scaled_pyramid_composes() {
        // A pyramid of ScaledImage levels: capability metadata per level.
        let levels = vec![
            ScaledImage::new(
                Image::<MonoF32>::zero(8, 8),
                pixel_distance!(1.0),
                OriginOffset::ZERO,
                sigma!(0.5),
            ),
            ScaledImage::new(
                Image::<MonoF32>::zero(4, 4),
                pixel_distance!(2.0),
                OriginOffset::ZERO,
                sigma!(1.0),
            ),
        ];
        let p = LevelChain::try_from_levels(levels).unwrap();
        assert_eq!(p.depth(), 2);
        assert_eq!(p.level(1).pixel_distance(), pixel_distance!(2.0));
        assert_eq!(p.level(1).sigma(), sigma!(1.0));
    }

    // ── PlacedImage ─────────────────────────────────────────────────────

    fn placed_level() -> PlacedImage<MonoF32> {
        PlacedImage::new(
            Image::generate(4, 3, |x, y| MonoF32::new((y * 4 + x) as f32)),
            pixel_distance!(2.0),
            OriginOffset::ZERO,
        )
    }

    #[test]
    fn placed_image_carries_its_grid_and_no_sigma() {
        let level = placed_level();
        assert_eq!(level.pixel_distance(), pixel_distance!(2.0));
        assert_eq!(level.origin_offset(), CoordinateF64::new(0.0, 0.0));
        assert_eq!(
            level.to_base(CoordinateF64::new(3.0, 4.0)),
            CoordinateF64::new(6.0, 8.0)
        );
        assert_eq!(
            level.to_local(CoordinateF64::new(6.0, 8.0)),
            CoordinateF64::new(3.0, 4.0)
        );
    }

    #[test]
    fn placed_image_answers_as_an_image() {
        // The delegation is what keeps the break soft: code that never heard
        // of pyramids sees an ordinary raster image.
        let level = placed_level();
        assert_eq!(level.size(), Size::new(4, 3));
        assert_eq!(level.width(), 4);
        assert_eq!(level.pixel_at(2, 1), MonoF32::new(6.0));
        assert_eq!(level.row(1)[2], MonoF32::new(6.0));
        assert_eq!(level.as_image().size(), Size::new(4, 3));
    }

    #[test]
    fn scaled_image_answers_as_an_image_too() {
        let level = ScaledImage::new(
            Image::generate(4, 3, |x, y| MonoF32::new((y * 4 + x) as f32)),
            pixel_distance!(2.0),
            OriginOffset::ZERO,
            sigma!(1.0),
        );
        assert_eq!(level.size(), Size::new(4, 3));
        assert_eq!(level.pixel_at(2, 1), MonoF32::new(6.0));
        assert_eq!(level.row(1)[2], MonoF32::new(6.0));
    }

    #[test]
    fn with_sigma_carries_the_geometry_over() {
        let scaled = placed_level().with_sigma(sigma!(1.5));
        assert_eq!(scaled.pixel_distance(), pixel_distance!(2.0));
        assert_eq!(scaled.origin_offset(), CoordinateF64::new(0.0, 0.0));
        assert_eq!(scaled.sigma(), sigma!(1.5));
        assert_eq!(scaled.size(), Size::new(4, 3));
    }

    #[test]
    fn into_image_drops_the_geometry() {
        let image = placed_level().into_image();
        assert_eq!(image.size(), Size::new(4, 3));
    }
}

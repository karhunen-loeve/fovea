//! Composition of mappings, and the direction a remap reads them in.

use super::homography::Homography;
use super::lens::BrownConrady;
use super::map::{AffineMap, PlaneMap};
use super::point::Point;
use super::units::{IdealPixels, LengthUnit, PixelSpace};

/// The composition of one mapping with another: `self` first, then `next`.
///
/// The output is the smallest class of mapping that contains the
/// composition. Called through [`PlaneMap::then`].
pub trait Compose<B> {
    /// The mapping the composition is.
    type Output;

    /// `self` first, then `next`.
    fn compose(self, next: B) -> Self::Output;
}

/// Two mappings applied one after the other, as one [`PlaneMap`].
///
/// The composition of a mapping with a lens model has no closed form and
/// converts points only. Its point has no image where either mapping has
/// none.
///
/// # Example
///
/// ```
/// use fovea::{IdealPixels, Pixels, Point};
/// use fovea::geometry::{BrownConrady, CameraMatrix, FocalLength, PlaneMap, UniformScale};
///
/// let cam = CameraMatrix::try_new(FocalLength { x: 1000.0, y: 1000.0 }, Point::new(640.0, 480.0))?;
/// let lens = BrownConrady::from_opencv(cam, &[-0.2, 0.0, 0.0, 0.0])?;
///
/// // An undistorted view at half the resolution: a pixel of it is two of
/// // the ideal image, which the lens then distorts.
/// let half: UniformScale<Pixels, IdealPixels> = fovea::uniform_scale!(2.0);
/// let chain = half.then(lens);
/// let p = Point::new(100.0, 50.0);
/// assert_eq!(chain.try_map_point(p), lens.try_map_point(Point::new(200.0, 100.0)));
/// # Ok::<(), fovea::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Chain<A, B> {
    first: A,
    second: B,
}

impl<A, B> Chain<A, B> {
    /// The mapping applied first.
    #[must_use]
    pub fn first(&self) -> &A {
        &self.first
    }

    /// The mapping applied second.
    #[must_use]
    pub fn second(&self) -> &B {
        &self.second
    }
}

impl<A, B> PlaneMap for Chain<A, B>
where
    A: PlaneMap,
    B: PlaneMap<Domain = A::Codomain>,
{
    type Domain = A::Domain;
    type Codomain = B::Codomain;

    fn try_map_point(&self, p: Point<A::Domain>) -> Option<Point<B::Codomain>> {
        self.first
            .try_map_point(p)
            .and_then(|q| self.second.try_map_point(q))
    }
}

/// Any mapping ending in the ideal image, followed by a lens model: a
/// rectification read through the lens in one step, or the undistorted
/// image given a camera matrix of its own.
impl<A> Compose<BrownConrady> for A
where
    A: PlaneMap<Codomain = IdealPixels>,
{
    type Output = Chain<A, BrownConrady>;

    fn compose(self, next: BrownConrady) -> Chain<A, BrownConrady> {
        Chain {
            first: self,
            second: next,
        }
    }
}

/// A mapping whose inverse is known in closed form: every [`AffineMap`]
/// and the homography.
///
/// [`SourceToDest`] needs one. Every `AffineMap` is `Invertible`, with its
/// own [`inverse`](AffineMap::inverse) as [`invert`](Self::invert), so an
/// affine mapping of your own needs no second impl and both traits name the
/// same inverse. In generic code, bound on `AffineMap` alone: it brings
/// `Invertible` with it, and `invert()` then returns an `AffineMap`. A bound
/// on both, `M: AffineMap + Invertible`, makes the compiler take the two
/// `Inverse` types from the bounds, where it cannot see that they are one.
/// A lens model has no closed-form inverse, so it is not `Invertible`: its
/// points are corrected one by one with [`BrownConrady::undistort_point`].
///
/// # Example
///
/// ```
/// use fovea::{Pixels, Point};
/// use fovea::geometry::{AffineMap, Invertible, SourceLookup, SourceToDest, Similarity};
///
/// // An `AffineMap` bound is enough for both.
/// fn back<M: AffineMap>(m: &M, p: Point<M::Codomain>) -> Point<M::Domain> {
///     m.invert().map_point(p)
/// }
///
/// let s: Similarity<Pixels, Pixels> = Similarity::try_linear(2.0, 0.0)?;
/// assert_eq!(back(&s, Point::new(4.0, 6.0)), Point::new(2.0, 3.0));
/// assert_eq!(SourceToDest::new(s).source_of(Point::new(4.0, 6.0)), Some(Point::new(2.0, 3.0)));
/// # Ok::<(), fovea::Error>(())
/// ```
pub trait Invertible: PlaneMap {
    /// The inverse mapping's type.
    type Inverse: PlaneMap<Domain = Self::Codomain, Codomain = Self::Domain>;

    /// The inverse mapping.
    fn invert(&self) -> Self::Inverse;
}

/// One inverse for the affine family: the one `AffineMap` computes, so the
/// two traits cannot disagree.
impl<M: AffineMap> Invertible for M {
    type Inverse = <M as AffineMap>::Inverse;

    fn invert(&self) -> Self::Inverse {
        self.inverse()
    }
}

impl<D: LengthUnit, C: LengthUnit> Invertible for Homography<D, C> {
    type Inverse = Homography<C, D>;
    fn invert(&self) -> Homography<C, D> {
        self.inverse()
    }
}

mod sealed {
    pub trait Sealed {}
}

/// Where the source of each destination pixel lies: what a remap reads.
///
/// Implemented by the two direction types, [`DestToSource`] and
/// [`SourceToDest`], and by nothing else, so a remap always says in its
/// type which way its mapping runs. Sealed.
///
/// Both ends are spaces measured in pixels: a remap reads an image in
/// [`Source`](Self::Source) and returns one in [`Dest`](Self::Dest). A
/// mapping into a unit of the world, a calibration, is no lookup, so a
/// remap through it does not compile.
pub trait SourceLookup: sealed::Sealed {
    /// The space of the destination, the image a remap returns.
    type Dest: PixelSpace;
    /// The space of the source, the image a remap reads.
    type Source: PixelSpace;

    /// The position in the source image of the destination pixel `dest`, or
    /// `None` where the mapping has none.
    fn source_of(&self, dest: Point<Self::Dest>) -> Option<Point<Self::Source>>;
}

/// A mapping from destination pixels to source pixels, used as it is.
///
/// This is how a remap works: it visits every pixel of the destination and
/// reads the source where the mapping sends it. A lens model is used this
/// way, because it maps the ideal image's pixels, the destination of an
/// undistortion, to the camera image's.
///
/// # Example
///
/// ```
/// use fovea::{Pixels, Point};
/// use fovea::geometry::{DestToSource, SourceLookup, UniformScale};
///
/// // Each destination pixel reads the source at twice its position.
/// let twice: UniformScale<Pixels, Pixels> = fovea::uniform_scale!(2.0);
/// let shrink = DestToSource(twice);
/// assert_eq!(shrink.source_of(Point::new(3.0, 4.0)), Some(Point::new(6.0, 8.0)));
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DestToSource<M>(pub M);

/// A mapping from source pixels to destination pixels, inverted once at
/// construction so a remap can read it from the destination.
///
/// For a mapping given the way a caller thinks of it: where a source pixel
/// ends up. Only an [`Invertible`] mapping qualifies; every one of them
/// checked at its own construction that it can be inverted, so this cannot
/// fail.
///
/// # Example
///
/// ```
/// use fovea::{Pixels, Point};
/// use fovea::geometry::{SourceLookup, SourceToDest, UniformScale};
///
/// // Each source pixel moves to twice its position: the image doubles.
/// let twice: UniformScale<Pixels, Pixels> = fovea::uniform_scale!(2.0);
/// let grow = SourceToDest::new(twice);
/// assert_eq!(grow.source_of(Point::new(6.0, 8.0)), Some(Point::new(3.0, 4.0)));
/// ```
///
/// A lens model has no closed-form inverse:
///
/// ```compile_fail
/// use fovea::Point;
/// use fovea::geometry::{BrownConrady, CameraMatrix, FocalLength, SourceToDest};
///
/// let cam = CameraMatrix::try_new(FocalLength { x: 1.0, y: 1.0 }, Point::new(0.0, 0.0)).unwrap();
/// let lens = BrownConrady::from_opencv(cam, &[0.0; 4]).unwrap();
/// let _ = SourceToDest::new(lens);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SourceToDest<M: Invertible> {
    inverse: M::Inverse,
}

impl<M: Invertible> SourceToDest<M> {
    /// The source-to-destination mapping `map`, stored as its inverse.
    #[must_use]
    pub fn new(map: M) -> Self {
        Self {
            inverse: map.invert(),
        }
    }
}

impl<M> sealed::Sealed for DestToSource<M> {}
impl<M: Invertible> sealed::Sealed for SourceToDest<M> {}

impl<M> SourceLookup for DestToSource<M>
where
    M: PlaneMap,
    M::Domain: PixelSpace,
    M::Codomain: PixelSpace,
{
    type Dest = M::Domain;
    type Source = M::Codomain;

    fn source_of(&self, dest: Point<M::Domain>) -> Option<Point<M::Codomain>> {
        self.0.try_map_point(dest)
    }
}

impl<M> SourceLookup for SourceToDest<M>
where
    M: Invertible,
    M::Domain: PixelSpace,
    M::Codomain: PixelSpace,
{
    type Dest = M::Codomain;
    type Source = M::Domain;

    fn source_of(&self, dest: Point<M::Codomain>) -> Option<Point<M::Domain>> {
        self.inverse.try_map_point(dest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{Affine, CameraMatrix, FocalLength, Similarity, Vector};
    use crate::{IdealPixels, Pixels};

    fn lens() -> BrownConrady {
        let cam =
            CameraMatrix::try_new(FocalLength { x: 800.0, y: 800.0 }, Point::new(320.0, 240.0))
                .unwrap();
        BrownConrady::from_opencv(cam, &[-0.25, 0.08, 0.001, 0.0005, 0.0]).unwrap()
    }

    #[test]
    fn a_chain_applies_its_mappings_in_order() {
        let shift: Affine<Pixels, IdealPixels> =
            Affine::try_new([[1.0, 0.0], [0.0, 1.0]], Vector::new(10.0, -5.0)).unwrap();
        let chain = shift.then(lens());
        let p = Point::new(100.0, 200.0);
        assert_eq!(
            chain.try_map_point(p),
            lens().try_map_point(Point::new(110.0, 195.0))
        );
        assert_eq!(chain.first(), &shift);
        assert_eq!(chain.second(), &lens());
    }

    #[test]
    fn a_chain_has_no_image_where_its_first_mapping_has_none() {
        let h: Homography<Pixels, IdealPixels> =
            Homography::try_new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.5, 1.0]]).unwrap();
        let chain = h.then(lens());
        assert_eq!(chain.try_map_point(Point::new(3.0, -2.0)), None);
        assert!(chain.try_map_point(Point::new(3.0, 0.0)).is_some());
    }

    #[test]
    fn source_to_dest_reads_through_the_inverse() {
        let h: Homography<Pixels, Pixels> =
            Homography::try_new([[1.1, 0.05, 4.0], [0.02, 0.95, -3.0], [1e-4, 5e-5, 1.0]]).unwrap();
        let lookup = SourceToDest::new(h);
        for p in [Point::new(0.0, 0.0), Point::new(300.0, 200.0)] {
            let dest = h.try_map_point(p).unwrap();
            let back = lookup.source_of(dest).unwrap();
            assert!(back.distance(p).get() < 1e-9);
        }
        let s: Similarity<Pixels, Pixels> = Similarity::try_linear(2.0, 0.5).unwrap();
        let lookup = SourceToDest::new(s);
        let dest = s.try_map_point(Point::new(5.0, 7.0)).unwrap();
        assert!(
            lookup
                .source_of(dest)
                .unwrap()
                .distance(Point::new(5.0, 7.0))
                .get()
                < 1e-12
        );
    }

    /// A translation of a caller's own, written as an `AffineMap` only.
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Shift(f64, f64);

    impl PlaneMap for Shift {
        type Domain = Pixels;
        type Codomain = Pixels;
        fn try_map_point(&self, p: Point<Pixels>) -> Option<Point<Pixels>> {
            Some(self.map_point(p))
        }
    }

    impl AffineMap for Shift {
        type Inverse = Shift;
        fn map_point(&self, p: Point<Pixels>) -> Point<Pixels> {
            Point::new(p.x + self.0, p.y + self.1)
        }
        fn map_vector(&self, v: Vector<Pixels>) -> Vector<Pixels> {
            v
        }
        fn inverse(&self) -> Shift {
            Shift(-self.0, -self.1)
        }
        fn reverses_orientation(&self) -> bool {
            false
        }
    }

    #[test]
    fn an_affine_map_of_ones_own_is_invertible_with_its_own_inverse() {
        // The v0.6.0 review's P4: the two traits named two inverses, and
        // an affine mapping of a caller's own needed a second impl to enter
        // `SourceToDest`.
        let shift = Shift(3.0, -2.0);
        assert_eq!(shift.invert(), shift.inverse());
        let lookup = SourceToDest::new(shift);
        assert_eq!(
            lookup.source_of(Point::new(10.0, 10.0)),
            Some(Point::new(7.0, 12.0))
        );
        // The affine family goes through the same impl.
        let s: Similarity<Pixels, Pixels> = Similarity::try_linear(2.0, 0.5).unwrap();
        assert_eq!(s.invert(), s.inverse());
        let a: Affine<Pixels, IdealPixels> =
            Affine::try_new([[2.0, 0.5], [0.0, 1.0]], Vector::new(1.0, 2.0)).unwrap();
        assert_eq!(a.invert(), a.inverse());
    }

    /// Generic over `AffineMap` alone, the inverse is an `AffineMap` again.
    fn back<M: AffineMap>(m: &M, p: Point<M::Codomain>) -> Point<M::Domain> {
        m.invert().map_point(p)
    }

    #[test]
    fn an_affine_bound_alone_gives_the_affine_inverse() {
        assert_eq!(
            back(&Shift(1.0, 1.0), Point::new(4.0, 6.0)),
            Point::new(3.0, 5.0)
        );
    }
}

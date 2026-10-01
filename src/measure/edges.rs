//! Edges on a profile: where they are, how strong, and how they pair.

use core::f64::consts::TAU;

use super::caliper::Profile;
use crate::Error;
use crate::Sigma;
use crate::analyze::peak::{Extremum, parabola_vertex};
use crate::error::{ParameterError, Requirement, Value};
use crate::geometry::{Length, Pixels, Point};

/// The direction of an edge, read along the caliper's path.
///
/// # Example
///
/// ```
/// use fovea::measure::Polarity;
///
/// assert_eq!(Polarity::DarkToLight.opposite(), Polarity::LightToDark);
/// assert_eq!(Polarity::Either.opposite(), Polarity::Either);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Polarity {
    /// The intensity rises along the path.
    DarkToLight,
    /// The intensity falls along the path.
    LightToDark,
    /// Either direction.
    Either,
}

impl Polarity {
    /// The other direction; `Either` stays `Either`.
    #[must_use]
    pub const fn opposite(self) -> Self {
        match self {
            Polarity::DarkToLight => Polarity::LightToDark,
            Polarity::LightToDark => Polarity::DarkToLight,
            Polarity::Either => Polarity::Either,
        }
    }

    fn admits(self, edge: &Edge) -> bool {
        match self {
            Polarity::DarkToLight => edge.contrast > 0.0,
            Polarity::LightToDark => edge.contrast < 0.0,
            Polarity::Either => true,
        }
    }
}

/// The smallest contrast an edge must have to be reported, in the pixel's
/// own units: finite and positive.
///
/// The contrast of an ideal step is its height, so `min_contrast!(20.0)` on
/// an 8-bit image keeps edges with a step of at least 20 grey levels. Same
/// construction discipline as the crate's other parameter types: the
/// [`min_contrast!`](crate::min_contrast) macro for literals,
/// [`MinContrast::try_new`] for computed values, and [`MinContrast::new`]
/// as the checked `const fn` under the macro.
///
/// # Example
///
/// ```
/// use fovea::measure::MinContrast;
///
/// const THRESHOLD: MinContrast = fovea::min_contrast!(20.0);
/// assert_eq!(THRESHOLD.get(), 20.0);
/// assert!(MinContrast::try_new(0.0).is_err());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct MinContrast(f64);

impl MinContrast {
    /// Creates a `MinContrast`, returning `None` unless `value` is finite and
    /// positive.
    #[must_use]
    pub const fn new(value: f64) -> Option<Self> {
        if value.is_finite() && value > 0.0 {
            Some(Self(value))
        } else {
            None
        }
    }

    /// Creates a `MinContrast` from a computed value, validating it.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidParameter`] if `value` is zero, negative, NaN or
    /// infinite.
    pub fn try_new(value: f64) -> Result<Self, Error> {
        Self::new(value).ok_or_else(|| {
            ParameterError::new(
                "min contrast",
                Requirement::FinitePositive,
                Value::F64(value),
            )
            .into()
        })
    }

    /// Returns the raw value.
    #[must_use]
    pub const fn get(self) -> f64 {
        self.0
    }
}

/// A [`MinContrast`](crate::measure::MinContrast) literal, checked at
/// compile time.
///
/// A value that is not a constant expression does not compile
/// (`error[E0435]`); use
/// [`MinContrast::try_new`](crate::measure::MinContrast::try_new) there.
///
/// # Example
///
/// ```
/// let threshold = fovea::min_contrast!(15.0);
/// assert_eq!(threshold.get(), 15.0);
/// ```
///
/// ```compile_fail
/// // ERROR: evaluation panicked: must be finite and strictly positive
/// let _ = fovea::min_contrast!(0.0);
/// ```
#[macro_export]
macro_rules! min_contrast {
    ($value:expr) => {
        const {
            $crate::measure::MinContrast::new($value)
                .expect($crate::error::Requirement::FinitePositive.text())
        }
    };
}

/// An edge found on a profile.
///
/// `contrast` is signed: positive where the intensity rises along the
/// path, negative where it falls. Its magnitude is the height of an ideal
/// step, in the pixel's units; an edge blurred by the optics reads lower.
///
/// # Example
///
/// ```
/// use fovea::{Length, Point};
/// use fovea::measure::{Edge, Polarity};
///
/// let e = Edge { position: Point::new(20.3, 16.0), along: Length::new(10.3), contrast: -180.0 };
/// assert_eq!(e.polarity(), Polarity::LightToDark);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Edge {
    /// The edge's position in the image, on the caliper's path.
    pub position: Point<Pixels>,
    /// The distance from the start of the path along it.
    pub along: Length<Pixels>,
    /// The signed contrast, the step height for an ideal step.
    pub contrast: f64,
}

impl Edge {
    /// `DarkToLight` for a positive contrast, `LightToDark` otherwise.
    #[must_use]
    pub fn polarity(&self) -> Polarity {
        if self.contrast > 0.0 {
            Polarity::DarkToLight
        } else {
            Polarity::LightToDark
        }
    }
}

/// The edges of a profile, sorted along the path.
///
/// Every edge above the threshold is kept; choosing one is a named step:
/// [`first`](Edges::first), [`last`](Edges::last),
/// [`strongest`](Edges::strongest), [`nearest_to`](Edges::nearest_to), or
/// [`pairs`](Edges::pairs) for edge pairs. Where two candidates tie, the
/// earlier one along the path wins, and an edge with a NaN contrast never
/// does.
///
/// # Example
///
/// ```
/// use fovea::{Length, Point};
/// use fovea::border::Skip;
/// use fovea::image::Image;
/// use fovea::measure::{Caliper, Polarity, profile};
/// use fovea::pixel::MonoF32;
/// use fovea::transform::CatmullRom;
///
/// // Dark left of x = 20.5, light right of it.
/// let img = Image::generate(48, 16, |x, _| MonoF32::new(if x <= 20 { 30.0 } else { 230.0 }));
/// let cal = Caliper::try_segment(
///     Point::new(4.0, 8.0), Point::new(40.0, 8.0), Length::new(4.0), Length::new(0.25),
/// )?;
/// let edges = profile(&img, &cal, CatmullRom, &Skip)?
///     .edges(Polarity::Either, fovea::sigma!(1.5), fovea::min_contrast!(50.0));
/// assert_eq!(edges.len(), 1);
/// let edge = edges.first().unwrap();
/// assert!((edge.position.x - 20.5).abs() < 1e-6);
/// assert!(edge.contrast > 190.0);
/// # Ok::<(), fovea::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Edges {
    edges: Vec<Edge>,
}

/// The index of the strongest of `edges` by contrast magnitude: the first
/// of several equal ones, and never one with a NaN contrast.
fn strongest_index(edges: &[Edge]) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (i, e) in edges.iter().enumerate() {
        let m = e.contrast.abs();
        if m.is_nan() {
            continue;
        }
        if best.is_none_or(|(_, b)| m > b) {
            best = Some((i, m));
        }
    }
    best.map(|(i, _)| i)
}

impl Edges {
    /// The edges, sorted along the path.
    #[must_use]
    pub fn as_slice(&self) -> &[Edge] {
        &self.edges
    }

    /// The number of edges.
    #[must_use]
    pub fn len(&self) -> usize {
        self.edges.len()
    }

    /// Whether no edge passed the threshold.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    /// An iterator over the edges, along the path.
    pub fn iter(&self) -> core::slice::Iter<'_, Edge> {
        self.edges.iter()
    }

    /// The edge nearest the start of the path.
    #[must_use]
    pub fn first(&self) -> Option<Edge> {
        self.edges.first().copied()
    }

    /// The edge nearest the end of the path.
    #[must_use]
    pub fn last(&self) -> Option<Edge> {
        self.edges.last().copied()
    }

    /// The edge of the largest contrast magnitude; of several equal ones,
    /// the earliest.
    #[must_use]
    pub fn strongest(&self) -> Option<Edge> {
        strongest_index(&self.edges).map(|i| self.edges[i])
    }

    /// The edge nearest the distance `along` from the start of the path; of
    /// two equally near, the earlier.
    #[must_use]
    pub fn nearest_to(&self, along: Length<Pixels>) -> Option<Edge> {
        let mut best: Option<(Edge, f64)> = None;
        for e in &self.edges {
            let d = (e.along.get() - along.get()).abs();
            if d.is_nan() {
                continue;
            }
            if best.is_none_or(|(_, b)| d < b) {
                best = Some((*e, d));
            }
        }
        best.map(|(e, _)| e)
    }

    /// The edge pairs `rule` forms, each starting with an edge of polarity
    /// `first`.
    ///
    /// With `Polarity::Either`, the first edge on the path sets the order,
    /// so a light bar on a dark background pairs dark-to-light with
    /// light-to-dark, and a dark bar the reverse.
    ///
    /// # Example
    ///
    /// ```
    /// use fovea::{Length, Point};
    /// use fovea::measure::{Edge, Neighbors, PairRule, Polarity};
    ///
    /// let at = |s: f64, c: f64| Edge {
    ///     position: Point::new(s, 0.0), along: Length::new(s), contrast: c,
    /// };
    /// // A light bar from 10 to 25 with an inner step at 14.
    /// let edges = [at(10.0, 60.0), at(14.0, 120.0), at(25.0, -180.0)];
    /// let pairs = Neighbors.pairs(&edges, Polarity::DarkToLight);
    /// assert_eq!(pairs.len(), 1);
    /// assert_eq!(pairs[0].width().get(), 11.0);
    /// ```
    #[must_use]
    pub fn pairs(&self, first: Polarity, rule: impl PairRule) -> Vec<EdgePair> {
        rule.pairs(&self.edges, first)
    }

    /// The edges as a vector, sorted along the path.
    #[must_use]
    pub fn into_vec(self) -> Vec<Edge> {
        self.edges
    }
}

impl<'a> IntoIterator for &'a Edges {
    type Item = &'a Edge;
    type IntoIter = core::slice::Iter<'a, Edge>;
    fn into_iter(self) -> Self::IntoIter {
        self.edges.iter()
    }
}

impl IntoIterator for Edges {
    type Item = Edge;
    type IntoIter = std::vec::IntoIter<Edge>;
    fn into_iter(self) -> Self::IntoIter {
        self.edges.into_iter()
    }
}

/// Two edges that bound a feature: a bar, a gap, a slot.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EdgePair {
    /// The edge nearer the start of the path.
    pub first: Edge,
    /// The edge nearer its end.
    pub second: Edge,
}

impl EdgePair {
    /// The distance between the two edges along the path: on an arc, the
    /// arc length.
    ///
    /// The straight distance between the two positions is
    /// `first.position.distance(second.position)`. Only that one converts to
    /// world units under every mapping, by converting the two positions; a
    /// width along the path converts under a conformal mapping.
    #[must_use]
    pub fn width(&self) -> Length<Pixels> {
        self.second.along - self.first.along
    }
}

/// A rule that forms edge pairs from the edges of a profile.
///
/// A value passed to [`Edges::pairs`], with no default, so the caller
/// writes down which rule pairs the edges. `edges` is sorted along the
/// path. The trait is open: a rule of one's own needs no change in fovea.
///
/// # Example
///
/// ```
/// use fovea::{Length, Point};
/// use fovea::measure::{Edge, EdgePair, PairRule, Polarity};
///
/// /// The outermost pair: the first edge and the last.
/// struct Outermost;
/// impl PairRule for Outermost {
///     fn pairs(&self, edges: &[Edge], _first: Polarity) -> Vec<EdgePair> {
///         match (edges.first(), edges.last()) {
///             (Some(&first), Some(&second)) if edges.len() > 1 => vec![EdgePair { first, second }],
///             _ => Vec::new(),
///         }
///     }
/// }
///
/// let at = |s: f64, c: f64| Edge {
///     position: Point::new(s, 0.0), along: Length::new(s), contrast: c,
/// };
/// let pairs = Outermost.pairs(&[at(2.0, 50.0), at(5.0, 70.0), at(9.0, -90.0)], Polarity::Either);
/// assert_eq!(pairs[0].width().get(), 7.0);
/// ```
pub trait PairRule {
    /// The pairs formed from `edges`, each starting with an edge of
    /// polarity `first`.
    fn pairs(&self, edges: &[Edge], first: Polarity) -> Vec<EdgePair>;
}

/// Pairs two edges that are adjacent along the path, the first of the
/// chosen polarity and the second of the opposite one.
///
/// In a run of edges of one polarity, only the one next to the opposite
/// edge pairs: for a light bar with an inner step, edges ↑1 ↑2 ↓, it pairs
/// ↑2 with ↓. Each edge belongs to one pair at most.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Neighbors;

/// Lets the strongest edge of each run of one polarity stand for the run,
/// then pairs neighbours.
///
/// For a light bar with an inner step, edges ↑1 ↑2 ↓, it pairs whichever
/// of ↑1 and ↑2 has the larger contrast with ↓. Of equal contrasts the
/// earlier edge wins.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StrongestOfRun;

/// `first`, with `Either` resolved by the first edge on the path.
fn resolve(first: Polarity, edges: &[Edge]) -> Option<Polarity> {
    match first {
        Polarity::Either => edges.first().map(Edge::polarity),
        p => Some(p),
    }
}

fn neighbour_pairs(edges: &[Edge], first: Polarity) -> Vec<EdgePair> {
    let Some(first) = resolve(first, edges) else {
        return Vec::new();
    };
    let second = first.opposite();
    let mut pairs = Vec::new();
    let mut i = 0;
    while i + 1 < edges.len() {
        if first.admits(&edges[i]) && second.admits(&edges[i + 1]) {
            pairs.push(EdgePair {
                first: edges[i],
                second: edges[i + 1],
            });
            i += 2;
        } else {
            i += 1;
        }
    }
    pairs
}

impl PairRule for Neighbors {
    fn pairs(&self, edges: &[Edge], first: Polarity) -> Vec<EdgePair> {
        neighbour_pairs(edges, first)
    }
}

impl PairRule for StrongestOfRun {
    fn pairs(&self, edges: &[Edge], first: Polarity) -> Vec<EdgePair> {
        let strongest: Vec<Edge> = edges
            .chunk_by(|a, b| a.polarity() == b.polarity())
            .filter_map(|run| strongest_index(run).map(|i| run[i]))
            .collect();
        neighbour_pairs(&strongest, first)
    }
}

// ── Finding the edges ────────────────────────────────────────────────────────

/// The derivative-of-Gaussian kernel for a standard deviation of `sigma`
/// samples, reaching `radius` samples each side, scaled so that a ramp
/// rising by one per sample has the derivative one.
fn derivative_kernel(sigma: f64, radius: usize) -> Vec<f64> {
    let r = radius as isize;
    let raw: Vec<f64> = (-r..=r)
        .map(|k| {
            let k = k as f64;
            k * (-k * k / (2.0 * sigma * sigma)).exp()
        })
        .collect();
    let norm: f64 = (-r..=r).zip(&raw).map(|(k, w)| k as f64 * w).sum();
    raw.iter().map(|w| w / norm).collect()
}

impl Profile {
    /// The edges of the given polarity whose contrast reaches
    /// `min_contrast`, sorted along the path.
    ///
    /// The profile is differentiated with a derivative of a Gaussian of
    /// standard deviation `sigma`, in pixels along the path. An edge is a
    /// local extreme of that derivative. Its contrast is the extreme value,
    /// refined by a parabola through it and its neighbours, times
    /// `σ·√(2π)`, which makes the contrast of an ideal step its height. Its
    /// position is the centroid of the derivative within `±4σ` of the
    /// extreme, ending early where the derivative changes sign: unlike the
    /// vertex of the parabola, the centroid is not pulled towards pixel
    /// boundaries.
    ///
    /// Choose `sigma` at least the sampling step, and comparable to the
    /// blur of the edges: a smaller σ separates close edges, a larger one
    /// suppresses noise. Two edges of the same polarity closer than about
    /// `4σ` pull each other's positions; use a smaller σ there. An edge
    /// whose derivative lobe runs into either end of the profile is not
    /// reported, because its position cannot be measured without bias;
    /// extend the caliper about `8σ` beyond the edges it should find.
    ///
    /// See [`Edges`] for an example.
    #[must_use]
    pub fn edges(&self, polarity: Polarity, sigma: Sigma, min_contrast: MinContrast) -> Edges {
        let caliper = self.caliper();
        let step = caliper.step().get();
        let values = self.values();
        let n = values.len();
        let s = f64::from(sigma.get()) / step;
        let radius = ((4.0 * s).ceil() as usize).max(1);
        if n < 2 * radius + 3 {
            return Edges::default();
        }
        let kernel = derivative_kernel(s, radius);
        let scale = s * TAU.sqrt();
        // Contrast at each position whose kernel lies inside the profile;
        // the valid positions are `radius..n - radius`.
        let contrast: Vec<f64> = (radius..n - radius)
            .map(|i| {
                let window = &values[i - radius..=i + radius];
                window.iter().zip(&kernel).map(|(v, w)| v * w).sum::<f64>() * scale
            })
            .collect();

        let reach = (4.0 * s).ceil() as usize;
        let mut edges = Vec::new();
        for i in 1..contrast.len().saturating_sub(1) {
            let (a, b, c) = (contrast[i - 1], contrast[i], contrast[i + 1]);
            // Strict towards the earlier neighbour, inclusive towards the
            // later one, so a plateau reports its first sample; NaN fails
            // every comparison and is never an edge.
            let kind = if b > 0.0 && b > a && b >= c {
                Extremum::Maximum
            } else if b < 0.0 && b < a && b <= c {
                Extremum::Minimum
            } else {
                continue;
            };
            let rising = kind == Extremum::Maximum;
            if (polarity == Polarity::DarkToLight && !rising)
                || (polarity == Polarity::LightToDark && rising)
            {
                continue;
            }
            let peak = match parabola_vertex(a, b, c, kind) {
                Some(d) => b + d * (c - a) / 2.0 + d * d * (a + c - 2.0 * b) / 2.0,
                None => b,
            };
            if peak.abs() < min_contrast.get() {
                continue;
            }
            // The centroid of the lobe, which must end inside the profile.
            let same_sign = |v: f64| if rising { v > 0.0 } else { v < 0.0 };
            let mut lo = i;
            while lo > 0 && i - lo < reach && same_sign(contrast[lo - 1]) {
                lo -= 1;
            }
            let mut hi = i;
            while hi + 1 < contrast.len() && hi - i < reach && same_sign(contrast[hi + 1]) {
                hi += 1;
            }
            let truncated = (lo == 0 && same_sign(contrast[0]) && i - lo < reach)
                || (hi + 1 == contrast.len() && hi - i < reach);
            if truncated {
                continue;
            }
            let (mut weight, mut moment) = (0.0, 0.0);
            for (j, &v) in contrast.iter().enumerate().take(hi + 1).skip(lo) {
                weight += v;
                moment += v * j as f64;
            }
            let index = radius as f64 + moment / weight;
            let along = Length::new(index * step);
            edges.push(Edge {
                position: caliper.point_at(along),
                along,
                contrast: peak,
            });
        }
        Edges { edges }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::border::Skip;
    use crate::image::Image;
    use crate::measure::{Caliper, profile};
    use crate::pixel::MonoF32;
    use crate::sigma;
    use crate::transform::CatmullRom;

    fn l(v: f64) -> Length<Pixels> {
        Length::new(v)
    }

    fn at(s: f64, c: f64) -> Edge {
        Edge {
            position: Point::new(s, 0.0),
            along: l(s),
            contrast: c,
        }
    }

    /// The fraction of pixel `i`, which covers `[i - 0.5, i + 0.5]`, inside
    /// `[x0, x1]`.
    fn coverage(i: usize, x0: f64, x1: f64) -> f64 {
        let (lo, hi) = ((i as f64 - 0.5).max(x0), (i as f64 + 0.5).min(x1));
        (hi - lo).max(0.0)
    }

    /// A light vertical bar from `x0` to `x1` on a dark background, each
    /// pixel the area average of the ideal scene.
    fn bar(x0: f64, x1: f64) -> Image<MonoF32> {
        Image::generate(64, 32, |x, _| {
            MonoF32::new((20.0 + 200.0 * coverage(x, x0, x1)) as f32)
        })
    }

    fn horizontal(y: f64, from: f64, to: f64, width: f64, step: f64) -> Caliper {
        Caliper::try_segment(Point::new(from, y), Point::new(to, y), l(width), l(step)).unwrap()
    }

    #[test]
    fn an_edge_pair_measures_a_bar_to_a_few_thousandths_of_a_pixel() {
        // The item's acceptance test: a bar of known sub-pixel width, built
        // from its area coverage, measured by an edge pair. At σ = 1 px and
        // a step of 0.25 px the worst position error over every pixel and
        // sampling phase is about 0.001 px (simulated 2026-10-01).
        for (x0, x1) in [(20.3, 35.8), (20.13, 41.71), (21.0, 33.5), (19.77, 30.02)] {
            let img = bar(x0, x1);
            let cal = horizontal(16.0, 5.0, 55.0, 4.0, 0.25);
            let prof = profile(&img, &cal, CatmullRom, &Skip).unwrap();
            let edges = prof.edges(Polarity::Either, sigma!(1.0), min_contrast!(50.0));
            let pairs = edges.pairs(Polarity::DarkToLight, Neighbors);
            assert_eq!(pairs.len(), 1, "{x0}..{x1}: {edges:?}");
            let pair = pairs[0];
            let width = pair.width().get();
            assert!(
                (width - (x1 - x0)).abs() < 2.5e-3,
                "{x0}..{x1}: width {width}"
            );
            assert!((pair.first.position.x - x0).abs() < 1.5e-3);
            assert!((pair.second.position.x - x1).abs() < 1.5e-3);
            assert_eq!(pair.first.position.y, 16.0);
        }
    }

    #[test]
    fn the_contrast_of_a_step_is_close_to_its_height() {
        // σ = 2 needs about 16 px between each edge and the profile's ends.
        let img = bar(20.3, 40.6);
        let cal = horizontal(16.0, 2.0, 60.0, 0.0, 0.25);
        let prof = profile(&img, &cal, CatmullRom, &Skip).unwrap();
        let edges = prof.edges(Polarity::Either, sigma!(2.0), min_contrast!(50.0));
        assert_eq!(edges.len(), 2);
        let (rise, fall) = (edges.as_slice()[0], edges.as_slice()[1]);
        assert!(rise.contrast > 0.0 && fall.contrast < 0.0);
        // The pixel's area integration blurs the step a little, so the
        // contrast reads slightly below 200.
        for c in [rise.contrast, -fall.contrast] {
            assert!((c - 200.0).abs() < 0.03 * 200.0, "{c}");
        }
    }

    #[test]
    fn polarity_and_threshold_filter_the_edges() {
        let img = bar(20.3, 35.8);
        let cal = horizontal(16.0, 5.0, 55.0, 2.0, 0.25);
        let prof = profile(&img, &cal, CatmullRom, &Skip).unwrap();
        let rising = prof.edges(Polarity::DarkToLight, sigma!(1.0), min_contrast!(50.0));
        let falling = prof.edges(Polarity::LightToDark, sigma!(1.0), min_contrast!(50.0));
        assert_eq!(rising.len(), 1);
        assert_eq!(falling.len(), 1);
        assert!(rising.first().unwrap().contrast > 0.0);
        assert!(falling.first().unwrap().contrast < 0.0);
        let none = prof.edges(Polarity::Either, sigma!(1.0), min_contrast!(250.0));
        assert!(none.is_empty());
    }

    #[test]
    fn an_edge_too_close_to_the_end_is_not_reported() {
        let img = bar(20.3, 60.0);
        // The edge sits 3 px from the start, inside the derivative kernel
        // and its lobe of 4σ each.
        let cal = horizontal(16.0, 17.3, 40.0, 0.0, 0.25);
        let prof = profile(&img, &cal, CatmullRom, &Skip).unwrap();
        assert!(
            prof.edges(Polarity::Either, sigma!(1.0), min_contrast!(50.0))
                .is_empty()
        );
        let longer = horizontal(16.0, 11.0, 40.0, 0.0, 0.25);
        let prof = profile(&img, &longer, CatmullRom, &Skip).unwrap();
        assert_eq!(
            prof.edges(Polarity::Either, sigma!(1.0), min_contrast!(50.0))
                .len(),
            1
        );
    }

    #[test]
    fn an_arc_caliper_measures_along_the_arc() {
        // A light wedge between the angles -0.3 and 0.3 about (8, 32), each
        // pixel the average of 8 × 8 sub-samples.
        let (cx, cy) = (8.0, 32.0);
        let inside = |x: f64, y: f64| (y - cy).atan2(x - cx).abs() <= 0.3;
        let img = Image::generate(64, 64, |x, y| {
            let mut hits = 0u32;
            for j in 0..8 {
                for i in 0..8 {
                    let sx = x as f64 - 0.5 + (i as f64 + 0.5) / 8.0;
                    let sy = y as f64 - 0.5 + (j as f64 + 0.5) / 8.0;
                    hits += u32::from(inside(sx, sy));
                }
            }
            MonoF32::new(20.0 + 200.0 * hits as f32 / 64.0)
        });
        let radius = 40.0;
        let cal =
            Caliper::try_arc(Point::new(cx, cy), l(radius), -0.8, 1.6, l(2.0), l(0.25)).unwrap();
        let prof = profile(&img, &cal, CatmullRom, &Skip).unwrap();
        let edges = prof.edges(Polarity::Either, sigma!(1.0), min_contrast!(50.0));
        let pairs = edges.pairs(Polarity::Either, Neighbors);
        assert_eq!(pairs.len(), 1, "{edges:?}");
        let expected = radius * 0.6;
        let width = pairs[0].width().get();
        assert!((width - expected).abs() < 0.05, "arc {width} vs {expected}");
        let chord = pairs[0].first.position.distance(pairs[0].second.position);
        assert!(chord.get() < width, "the chord is shorter than the arc");
    }

    #[test]
    fn neighbours_and_strongest_of_run_pair_an_inner_step_differently() {
        let edges = [at(10.0, 60.0), at(14.0, 120.0), at(25.0, -180.0)];
        let n = Neighbors.pairs(&edges, Polarity::DarkToLight);
        assert_eq!(
            n,
            vec![EdgePair {
                first: edges[1],
                second: edges[2]
            }]
        );
        let s = StrongestOfRun.pairs(&edges, Polarity::DarkToLight);
        assert_eq!(
            s,
            vec![EdgePair {
                first: edges[1],
                second: edges[2]
            }]
        );

        let outer_stronger = [at(10.0, 150.0), at(14.0, 120.0), at(25.0, -180.0)];
        let s = StrongestOfRun.pairs(&outer_stronger, Polarity::DarkToLight);
        assert_eq!(s[0].first, outer_stronger[0]);
        assert_eq!(s[0].width().get(), 15.0);

        let tied = [at(10.0, 120.0), at(14.0, 120.0), at(25.0, -180.0)];
        let s = StrongestOfRun.pairs(&tied, Polarity::DarkToLight);
        assert_eq!(s[0].first, tied[0], "of equal contrasts the earlier wins");
    }

    #[test]
    fn pairs_follow_the_requested_order_and_either_resolves_by_the_first_edge() {
        // A dark bar on light: falling, then rising.
        let edges = [
            at(5.0, -100.0),
            at(9.0, 100.0),
            at(15.0, -100.0),
            at(19.0, 100.0),
        ];
        assert_eq!(Neighbors.pairs(&edges, Polarity::LightToDark).len(), 2);
        let rising_first = Neighbors.pairs(&edges, Polarity::DarkToLight);
        assert_eq!(rising_first.len(), 1);
        assert_eq!(rising_first[0].width().get(), 6.0);
        let either = Neighbors.pairs(&edges, Polarity::Either);
        assert_eq!(either.len(), 2);
        assert_eq!(either[0].first.contrast, -100.0);
        assert!(Neighbors.pairs(&[], Polarity::Either).is_empty());
    }

    #[test]
    fn named_selections_break_ties_towards_the_earlier_edge() {
        let edges = Edges {
            edges: vec![
                at(2.0, 50.0),
                at(6.0, -90.0),
                at(10.0, 90.0),
                at(14.0, f64::NAN),
            ],
        };
        assert_eq!(edges.first(), Some(edges.as_slice()[0]));
        assert_eq!(edges.last().unwrap().along, l(14.0));
        assert_eq!(edges.strongest().unwrap().along, l(6.0));
        assert_eq!(edges.nearest_to(l(8.0)).unwrap().along, l(6.0));
        assert_eq!(edges.nearest_to(l(9.0)).unwrap().along, l(10.0));
        assert_eq!(edges.iter().count(), 4);
        assert_eq!((&edges).into_iter().count(), 4);
        assert_eq!(edges.clone().into_vec().len(), 4);
        assert!(Edges::default().strongest().is_none());
    }

    #[test]
    fn min_contrast_validates() {
        assert_eq!(MinContrast::new(1.5).map(MinContrast::get), Some(1.5));
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(MinContrast::new(bad).is_none());
            match MinContrast::try_new(bad) {
                Err(Error::InvalidParameter(e)) => {
                    assert_eq!(e.requirement(), Requirement::FinitePositive);
                }
                other => panic!("{other:?}"),
            }
        }
    }

    #[test]
    fn the_derivative_kernel_measures_a_ramp_exactly() {
        let k = derivative_kernel(2.0, 8);
        let slope: f64 = k.iter().enumerate().map(|(i, w)| w * i as f64).sum();
        assert!((slope - 1.0).abs() < 1e-12);
        let flat: f64 = k.iter().sum();
        assert!(flat.abs() < 1e-12);
    }
}

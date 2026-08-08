//! Per-component statistics for connected-component labeling.
//!
//! See [`ComponentStats`]. Accumulated inline during pass 2 of the
//! engine when callers invoke
//! [`connected_components_with_stats`](super::connected_components_with_stats).
//!
//! The v1 scope is deliberately small: `area`, axis-aligned bounding
//! box (as inclusive min/max coordinates), and `sum_x` / `sum_y` for
//! the integer centroid. Higher-order moments, boundary pixels, holes,
//! Euler number, orientation, convexity, and intensity measurements
//! are deferred to follow-up work.

use crate::{Coordinate, CoordinateF64, Rectangle, Size};

/// Bounded per-component statistics, accumulated inline during pass 2.
///
/// Field semantics:
///
/// - `area` \u2014 pixel count of the component.
/// - `bbox_min` / `bbox_max_inclusive` \u2014 inclusive corners of the
///   tightest axis-aligned bounding box that contains every pixel in
///   the component. Both are valid pixel coordinates *inside* the
///   image; converting to a half-open [`Rectangle`] adds 1 to width
///   and height.
/// - `sum_x` / `sum_y` \u2014 sums of `x` and `y` pixel coordinates,
///   sufficient to compute the centroid as `(sum_x / area, sum_y /
///   area)`. Stored as `u64` to accommodate large images without
///   overflow (`u32::MAX * u32::MAX > u64::MAX / 4` only at extreme
///   sizes).
///
/// All fields are public; the helper methods are derived quantities
/// only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComponentStats {
    /// Number of pixels in the component.
    pub area: u64,
    /// Inclusive lower-right-most-top-left corner of the bounding box.
    pub bbox_min: Coordinate,
    /// Inclusive bottom-right corner of the bounding box.
    pub bbox_max_inclusive: Coordinate,
    /// Sum of `x` coordinates of every pixel in the component.
    pub sum_x: u64,
    /// Sum of `y` coordinates of every pixel in the component.
    pub sum_y: u64,
}

impl ComponentStats {
    /// Seed a new stats record from the first foreground pixel of a
    /// component.
    #[inline]
    pub(super) fn from_seed(at: Coordinate) -> Self {
        Self {
            area: 1,
            bbox_min: at,
            bbox_max_inclusive: at,
            sum_x: at.x as u64,
            sum_y: at.y as u64,
        }
    }

    /// Extend an existing stats record with another pixel of the same
    /// component.
    #[inline]
    pub(super) fn extend(&mut self, at: Coordinate) {
        self.area += 1;
        if at.x < self.bbox_min.x {
            self.bbox_min.x = at.x;
        }
        if at.y < self.bbox_min.y {
            self.bbox_min.y = at.y;
        }
        if at.x > self.bbox_max_inclusive.x {
            self.bbox_max_inclusive.x = at.x;
        }
        if at.y > self.bbox_max_inclusive.y {
            self.bbox_max_inclusive.y = at.y;
        }
        self.sum_x += at.x as u64;
        self.sum_y += at.y as u64;
    }

    /// Centroid (centre of mass) as a sub-pixel [`CoordinateF64`].
    ///
    /// Returns `(sum_x / area, sum_y / area)`. `area` is always `>= 1`
    /// for stats produced by the engine, so the division is safe.
    pub fn centroid(&self) -> CoordinateF64 {
        let inv = 1.0 / self.area as f64;
        CoordinateF64::new(self.sum_x as f64 * inv, self.sum_y as f64 * inv)
    }

    /// Axis-aligned bounding box as a half-open [`Rectangle`].
    ///
    /// The returned rectangle's width and height are `bbox_max_inclusive
    /// - bbox_min + 1` on each axis.
    pub fn bbox(&self) -> Rectangle {
        let w = self.bbox_max_inclusive.x - self.bbox_min.x + 1;
        let h = self.bbox_max_inclusive.y - self.bbox_min.y + 1;
        Rectangle::new(self.bbox_min, Size::new(w, h))
    }

    /// Aspect ratio of the bounding box: `width / height`. Returns
    /// `f64::INFINITY` for a zero-height bbox (which the engine
    /// never produces).
    pub fn aspect_ratio(&self) -> f64 {
        let r = self.bbox();
        r.size.width as f64 / r.size.height as f64
    }
}

// \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
// StatsSink \u2014 monomorphised, branchless stats accumulation gate.
// \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
//
// The engine's pass-2 loop accumulates `ComponentStats` only when the
// caller has asked for them. We express that via a trait
// (`StatsSink`) with two implementations:
//
//   * `NoStats`  \u2014 zero-sized; all methods are no-ops the compiler
//                  elides entirely.
//   * `&mut Vec<ComponentStats>`
//                \u2014 records (seed | extend) per pixel.
//
// Because the engine is generic over `S: StatsSink`, each entry point
// monomorphises into a specialised function: `connected_components_into`
// generates a path with no stats overhead, while
// `connected_components_with_stats` generates a path that calls the
// vector-backed accumulator. There is no per-pixel runtime branch.

pub(super) mod sink {
    use super::super::measurements::BlobMeasurements;
    use super::ComponentStats;
    use crate::Coordinate;

    /// Sealed sink trait \u2014 monomorphisation gate for stats
    /// accumulation. See module docs.
    pub(crate) trait StatsSink {
        /// When `false`, the engine skips the per-pixel boundary
        /// neighbour-check entirely and always passes `is_boundary =
        /// false` to [`record`](StatsSink::record). Because this is an
        /// associated `const`, the check is const-folded away for sinks
        /// that do not need it (`NoStats`, `WithStats`), leaving their
        /// generated code identical to before the parameter existed.
        const NEEDS_BOUNDARY: bool = false;

        /// Called once per foreground pixel at coordinate `at`.
        /// `compact_label` is the component's compact label
        /// (`1..=label_count`); `first` indicates whether this is the
        /// first pixel ever seen of that component (so the sink should
        /// seed rather than extend). `is_boundary` is the 4-connected
        /// boundary flag for this pixel, meaningful only when
        /// [`NEEDS_BOUNDARY`](StatsSink::NEEDS_BOUNDARY) is `true`
        /// (otherwise always `false`).
        fn record(&mut self, compact_label: u64, first: bool, at: Coordinate, is_boundary: bool);
    }

    /// Sink that drops every record. Compiles down to no work.
    pub(crate) struct NoStats;

    impl StatsSink for NoStats {
        #[inline(always)]
        fn record(&mut self, _compact_label: u64, _first: bool, _at: Coordinate, _is_boundary: bool) {}
    }

    /// Sink that accumulates per-component stats into a `Vec` indexed
    /// by `compact_label - 1`.
    pub(crate) struct WithStats<'a> {
        pub(crate) out: &'a mut Vec<ComponentStats>,
    }

    impl StatsSink for WithStats<'_> {
        // `NEEDS_BOUNDARY` stays at its `false` default: area/bbox/centroid
        // do not need the boundary flag, so the engine's neighbour-check is
        // const-folded away and this path is unchanged.
        #[inline]
        fn record(&mut self, compact_label: u64, first: bool, at: Coordinate, _is_boundary: bool) {
            if first {
                // Compact labels are dense `1..=label_count`, so each
                // new label appears exactly once with `first = true`
                // and `compact_label == out.len() + 1`.
                debug_assert_eq!(self.out.len() as u64, compact_label - 1);
                self.out.push(ComponentStats::from_seed(at));
            } else {
                self.out[(compact_label - 1) as usize].extend(at);
            }
        }
    }

    /// Sink that accumulates per-component [`BlobMeasurements`] (moments
    /// and boundary count) into a `Vec` indexed by `compact_label - 1`. Sets
    /// `NEEDS_BOUNDARY = true` so the engine runs the per-pixel boundary
    /// check that feeds the boundary count.
    pub(crate) struct WithMeasurements<'a> {
        pub(crate) out: &'a mut Vec<BlobMeasurements>,
    }

    impl StatsSink for WithMeasurements<'_> {
        const NEEDS_BOUNDARY: bool = true;

        #[inline]
        fn record(&mut self, compact_label: u64, first: bool, at: Coordinate, is_boundary: bool) {
            if first {
                debug_assert_eq!(self.out.len() as u64, compact_label - 1);
                self.out.push(BlobMeasurements::from_seed(at, is_boundary));
            } else {
                self.out[(compact_label - 1) as usize].extend(at, is_boundary);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_seed_single_pixel() {
        let s = ComponentStats::from_seed(Coordinate::new(3, 5));
        assert_eq!(s.area, 1);
        assert_eq!(s.bbox_min, Coordinate::new(3, 5));
        assert_eq!(s.bbox_max_inclusive, Coordinate::new(3, 5));
        assert_eq!(s.sum_x, 3);
        assert_eq!(s.sum_y, 5);
        assert_eq!(s.centroid(), CoordinateF64::new(3.0, 5.0));
        assert_eq!(
            s.bbox(),
            Rectangle::new(Coordinate::new(3, 5), Size::new(1, 1))
        );
        assert_eq!(s.aspect_ratio(), 1.0);
    }

    #[test]
    fn extend_grows_area_and_bbox() {
        let mut s = ComponentStats::from_seed(Coordinate::new(2, 2));
        s.extend(Coordinate::new(5, 4));
        s.extend(Coordinate::new(3, 1));
        assert_eq!(s.area, 3);
        assert_eq!(s.bbox_min, Coordinate::new(2, 1));
        assert_eq!(s.bbox_max_inclusive, Coordinate::new(5, 4));
        assert_eq!(s.sum_x, 2 + 5 + 3);
        assert_eq!(s.sum_y, 2 + 4 + 1);
    }

    #[test]
    fn centroid_of_centred_square() {
        // 3x3 square at (1..=3, 1..=3), 9 pixels.
        let mut s = ComponentStats::from_seed(Coordinate::new(1, 1));
        for y in 1..=3usize {
            for x in 1..=3usize {
                if (x, y) != (1, 1) {
                    s.extend(Coordinate::new(x, y));
                }
            }
        }
        assert_eq!(s.area, 9);
        let c = s.centroid();
        assert!((c.x - 2.0).abs() < 1e-12);
        assert!((c.y - 2.0).abs() < 1e-12);
    }

    #[test]
    fn bbox_is_half_open_rectangle() {
        let mut s = ComponentStats::from_seed(Coordinate::new(2, 3));
        s.extend(Coordinate::new(7, 9));
        let r = s.bbox();
        assert_eq!(r.offset, Coordinate::new(2, 3));
        assert_eq!(r.size, Size::new(6, 7));
    }

    #[test]
    fn aspect_ratio_wide_versus_tall() {
        // 6x2 box
        let mut s = ComponentStats::from_seed(Coordinate::new(0, 0));
        s.extend(Coordinate::new(5, 1));
        assert!((s.aspect_ratio() - 3.0).abs() < 1e-12);

        // 2x6 box
        let mut t = ComponentStats::from_seed(Coordinate::new(0, 0));
        t.extend(Coordinate::new(1, 5));
        assert!((t.aspect_ratio() - (2.0 / 6.0)).abs() < 1e-12);
    }
}

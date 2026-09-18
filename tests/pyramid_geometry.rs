//! What a pyramid level knows about itself, checked from outside the crate.
//!
//! An integration test reaches only the public API, so whatever it proves
//! here an external caller can also rely on. That is the point: v0.5.0 makes
//! four claims about geometry, and each of them is only worth making if a
//! caller can depend on it rather than on an implementation detail that
//! happens to hold today.
//!
//! The precedent is ADR-0031, which mandated an odd-size round-trip property
//! test for Laplacian reconstruction and gave its reason: a major library
//! shipped a pyramid for years that could not reconstruct at all, because no
//! such test existed.

use fovea::border::Skip;
use fovea::features::detect::{FastParams, NmsRadius, SegmentTest, fast_in_level};
use fovea::image::{
    Decimated, Dyadic, Image, ImageView, LevelChain, OriginOffset, PlacedImage, PlacedPyramid,
    Pyramid, ScaleLevel, ScaledImage, ScaledPyramid,
};
use fovea::pixel::{Mono8, MonoF32};
use fovea::transform::{Gaussian, PyramidMethod, pyr_down};
use fovea::{pixel_distance, sigma};

/// A flat test image. The geometry claims under test are independent of the
/// pixel values, so the cheapest content that exercises every size is right.
fn flat(width: usize, height: usize) -> Image<MonoF32> {
    Image::fill(width, height, MonoF32::new(0.5))
}

// ─── 1. `expand` is total ───────────────────────────────────────────────────

/// Every index that has a parent produces that parent's size, and the only
/// two indices that do not have one return `None`.
///
/// The sweep is the one the prototype ran before any of this was designed:
/// every width from 1 to 140 at five heights, at every depth. It covers both
/// parities on both axes, the degenerate 1-wide and 1-tall cases, and the
/// point where a level stops shrinking. Without totality over that range,
/// `expand`'s `expect` would be a panic waiting for a particular image size.
#[test]
fn expand_reproduces_every_parent_size() {
    for width in 1..=140 {
        for height in [1, 2, 3, 67, 68] {
            let pyramid: PlacedPyramid<MonoF32> = Gaussian.build(&flat(width, height), 8);

            for child in 1..pyramid.depth() {
                let raised: Image<MonoF32> = pyramid
                    .expand(child)
                    .expect("every level below the finest has a parent");
                assert_eq!(
                    raised.size(),
                    pyramid.level(child - 1).size(),
                    "{width}x{height}, level {child}"
                );
            }
        }
    }
}

/// The finest level has no parent, and neither does an index past the depth.
/// That is the same question `get` answers, and after the halving guarantee
/// it is the only one left.
#[test]
fn expand_returns_none_exactly_where_there_is_no_parent() {
    for width in 1..=140 {
        for height in [1, 2, 3, 67, 68] {
            let pyramid: PlacedPyramid<MonoF32> = Gaussian.build(&flat(width, height), 8);
            let depth = pyramid.depth();

            assert!(
                pyramid.expand::<MonoF32, _>(0).is_none(),
                "{width}x{height}: the finest level has no parent"
            );
            assert!(
                pyramid.expand::<MonoF32, _>(depth).is_none(),
                "{width}x{height}: there is no level {depth}"
            );
        }
    }
}

// ─── 2. `Dyadic::try_new` rejects what it must ──────────────────────────────

/// A chain that `LevelChain::try_from_levels` admits on purpose, and that is
/// nevertheless not dyadic.
///
/// Without this, the totality test above proves nothing about the invariant,
/// only about the builder: a `Dyadic` that accepted anything would still pass
/// it, because the builder never produces a non-halving chain.
#[test]
fn a_chain_with_equal_size_neighbours_is_not_dyadic() {
    let levels = vec![Image::<Mono8>::zero(8, 6), Image::<Mono8>::zero(8, 6)];
    // Legal as a level chain: equal sizes are what scale stacks and sub-band
    // decompositions need, so the chain constructor allows them.
    let chain = LevelChain::try_from_levels(levels).expect("non-growing is a legal chain");
    assert!(Dyadic::try_new(chain).is_err());
}

/// A chain that shrinks, but not by halving.
#[test]
fn a_chain_that_shrinks_without_halving_is_not_dyadic() {
    let levels = vec![Image::<Mono8>::zero(100, 68), Image::<Mono8>::zero(30, 34)];
    let chain = LevelChain::try_from_levels(levels).expect("non-growing is a legal chain");
    assert!(Dyadic::try_new(chain).is_err());
}

/// What `try_new` must accept, so the rejections above are not vacuous: a
/// chain whose parents are odd, where `ceil` rather than exact division is
/// the relation.
#[test]
fn an_odd_parent_still_halves() {
    let levels = vec![
        Image::<Mono8>::zero(9, 7),
        Image::<Mono8>::zero(5, 4),
        Image::<Mono8>::zero(3, 2),
    ];
    let chain = LevelChain::try_from_levels(levels).expect("non-growing is a legal chain");
    assert!(Dyadic::try_new(chain).is_ok());
}

// ─── 3. The σ ladder ────────────────────────────────────────────────────────

/// The published table, against the builder.
///
/// `σ_k² = σ_in² + (4^k − 1)/3`, in base-image pixels. The row for a sharp
/// input is a **limit**, not a reachable argument: `Sigma` is strictly
/// positive, so `σ_in = 0` cannot be named, which is itself part of the
/// design. σ_in = 0.001 is as close as the API allows.
#[test]
fn the_sigma_ladder_matches_the_published_numbers() {
    let image = flat(64, 64);

    let sharp: ScaledPyramid<MonoF32> = Gaussian
        .assuming_input_sigma(sigma!(0.001))
        .build(&image, 4);
    let lowe: ScaledPyramid<MonoF32> = Gaussian.assuming_input_sigma(sigma!(0.5)).build(&image, 4);

    // Level, σ for a sharp input, σ under Lowe's σ_in = 0.5.
    let table = [(1, 1.000, 1.118), (2, 2.236, 2.291), (3, 4.583, 4.610)];

    for (level, sharp_sigma, lowe_sigma) in table {
        assert!(
            (sharp.level(level).sigma().get() - sharp_sigma).abs() < 1e-3,
            "sharp level {level}: {}",
            sharp.level(level).sigma().get()
        );
        assert!(
            (lowe.level(level).sigma().get() - lowe_sigma).abs() < 1e-3,
            "level {level} at σ_in = 0.5: {}",
            lowe.level(level).sigma().get()
        );
    }
}

/// Level 0 carries the assumption itself, unchanged: no `pyr_down` has run
/// yet, so no variance has been added.
#[test]
fn the_base_level_carries_the_assumed_input_sigma() {
    let pyramid: ScaledPyramid<MonoF32> = Gaussian
        .assuming_input_sigma(sigma!(0.5))
        .build(&flat(32, 32), 3);
    assert_eq!(pyramid.finest().sigma(), sigma!(0.5));
}

// ─── 4. The delegation still binds ──────────────────────────────────────────

/// A pyramid level is accepted by code that never heard of pyramids, for
/// both level types.
///
/// This is the property that turns a hard break into a soft one, and it
/// fails at **compile time**, which is the point: if `PyramidLevel` ever
/// stops implying `RasterImage`, this file stops building rather than
/// producing a wrong number.
#[test]
fn a_level_is_accepted_wherever_an_image_is() {
    let base = Image::<MonoF32>::generate(48, 48, |x, y| {
        let inside = (16..32).contains(&x) && (16..32).contains(&y);
        MonoF32::new(if inside { 1.0 } else { 0.0 })
    });

    let placed: PlacedPyramid<MonoF32> = Gaussian.build(&base, 3);
    let scaled: ScaledPyramid<MonoF32> = Gaussian.assuming_input_sigma(sigma!(0.5)).build(&base, 3);

    // `pyr_down` takes any `RasterImage`, and a level is one.
    let from_placed: Image<MonoF32> = pyr_down(placed.level(1));
    let from_scaled: Image<MonoF32> = pyr_down(scaled.level(1));
    assert_eq!(from_placed.size(), placed.level(2).size());
    assert_eq!(from_scaled.size(), scaled.level(2).size());

    // A detector that binds `Decimated` accepts both, and reports in the
    // base frame either way.
    let params = FastParams::new(
        SegmentTest::new(0.15, 9).expect("a valid segment test"),
        NmsRadius::new(2).expect("a non-zero radius"),
    );
    let a = fast_in_level(placed.level(1), params, &Skip);
    let b = fast_in_level(scaled.level(1), params, &Skip);
    assert_eq!(
        a, b,
        "the σ a level carries must not change where it detects"
    );

    // The `ImageView` methods answer directly, with no `as_image()` between.
    assert_eq!(placed.level(1).size(), scaled.level(1).size());
    assert_eq!(placed.level(1).width(), 24);
    assert_eq!(
        placed.level(1).pixel_at(0, 0),
        scaled.level(1).pixel_at(0, 0)
    );
}

/// The capability split is a compile-time fence, and this is the readable
/// half of it: a function binding `ScaleLevel` takes a `ScaledImage` and
/// there is no `PlacedImage` it could be handed instead.
///
/// The other half cannot be written as a test, because it is the code that
/// does *not* compile: `needs_a_sigma(placed.level(1))` is a type error, and
/// that is the guarantee.
#[test]
fn only_a_level_that_was_told_a_sigma_can_answer_for_one() {
    fn needs_a_sigma(level: &impl ScaleLevel) -> f32 {
        level.sigma().get()
    }

    let scaled: ScaledPyramid<MonoF32> = Gaussian
        .assuming_input_sigma(sigma!(0.5))
        .build(&flat(32, 32), 2);
    assert!((needs_a_sigma(scaled.level(1)) - 1.118).abs() < 1e-3);

    // Both level types answer the geometry question, which is the one a lift
    // actually needs.
    let placed: PlacedPyramid<MonoF32> = Gaussian.build(&flat(32, 32), 2);
    fn needs_a_grid(level: &impl Decimated) -> f64 {
        level.pixel_distance().get()
    }
    assert_eq!(needs_a_grid(placed.level(1)), 2.0);
    assert_eq!(needs_a_grid(scaled.level(1)), 2.0);
}

/// A level of unknown provenance still reaches both capability traits, so
/// the fence is about what was established and not about who built it.
#[test]
fn an_imported_level_carries_the_same_capabilities() {
    let coarse = pyr_down(&flat(16, 16));

    let placed = PlacedImage::new(coarse.clone(), pixel_distance!(2.0), OriginOffset::ZERO);
    assert_eq!(placed.pixel_distance().get(), 2.0);

    let scaled = ScaledImage::new(
        coarse,
        pixel_distance!(2.0),
        OriginOffset::ZERO,
        sigma!(1.118),
    );
    assert_eq!(scaled.sigma(), sigma!(1.118));
    assert_eq!(scaled.pixel_distance(), placed.pixel_distance());
}

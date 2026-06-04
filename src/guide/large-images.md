# Large images and parallel work

fovea does not own your parallel runtime. The core crate has no Rayon dependency by design. Instead, it gives you safe decomposition primitives: slices, rows, ROIs, tiles, mutable disjoint tiles, sliding windows, and built-in neighborhood transforms.

## Decision table

| Work shape | Recommended API | Why |
|---|---|---|
| Independent per-pixel work on a contiguous image | `ContiguousImage::as_slice` / `ContiguousImageMut::as_mut_slice` | Fastest path; caller chooses Rayon, scoped threads, SIMD, GPU upload, or a plain loop. |
| Row-local work | `RasterImage::row` / `RasterImageMut::row_mut` | Keeps row boundaries explicit and avoids `pixel_at` in the inner loop. |
| Independent rectangular analysis | `SubView::tiles` | Zero-copy immutable views over non-overlapping regions. |
| Independent in-place tile mutation | `(&mut image).into_tiles_mut(size)` | Tiles are disjoint mutable views, so each worker can own one tile. |
| Built-in neighborhood filter | `convolve`, `filters`, `map_neighborhood`, `morphology`, `template_match` | Border handling and algorithm constraints are already encoded. |
| Custom overlapping-window analysis | `sliding_windows` / `SlidingWindow` | Use when windows overlap and you are analyzing rather than independently mutating. |
| Advanced tiled neighborhood processing | Tiles plus explicit halo/read margin | Necessary for custom scheduling or out-of-core processing; handle seams deliberately. |

## Per-pixel work: use slices or rows

```rust
use fovea::image::{ContiguousImageMut, Image, ImageView};
use fovea::pixel::Mono8;

let mut img = Image::fill(4, 2, Mono8::new(10));

for px in img.as_mut_slice() {
    *px = Mono8::new(255 - px.value());
}

assert_eq!(img.pixel_at(0, 0), Mono8::new(245));
```

If the image might be strided, use row access instead:

```rust
use fovea::image::{Image, ImageView, RasterImageMut};

let mut img = Image::generate(4, 2, |x, y| (x + y * 4) as u8);
for y in 0..img.height() {
    img.row_mut(y).reverse();
}

assert_eq!(img.pixel_at(0, 0), 3);
assert_eq!(img.pixel_at(0, 1), 7);
```

## Region-local work: use tiles

```rust
use fovea::Size;
use fovea::image::{Image, ImageView, SubView};

let img = Image::generate(6, 4, |x, y| (x + y * 6) as u8);
let tiles: Vec<_> = img.tiles(Size::new(3, 2)).collect();

assert_eq!(tiles.len(), 4);
assert!(tiles.iter().all(|tile| tile.size() == Size::new(3, 2)));
```

For in-place tile mutation, use `into_tiles_mut` on a mutable image reference:

```rust
use fovea::Size;
use fovea::image::{Image, ImageView, ImageViewMut, IntoTilesMut};

let mut img = Image::fill(4, 4, 0u8);
for mut tile in (&mut img).into_tiles_mut(Size::new(2, 2)) {
    *tile.pixel_at_mut(0, 0) = 255;
}

assert_eq!(img.pixel_at(0, 0), 255);
assert_eq!(img.pixel_at(2, 0), 255);
assert_eq!(img.pixel_at(0, 2), 255);
assert_eq!(img.pixel_at(2, 2), 255);
```

## Rayon-shaped example

Keep Rayon in your application or example crate, not in `fovea` core:

```rust,ignore
use fovea::Size;
use fovea::image::{ImageView, ImageViewMut, IntoTilesMut, RasterImageMut};
use rayon::prelude::*;

(&mut image)
    .into_tiles_mut(Size::new(512, 512))
    .par_bridge()
    .for_each(|mut tile| {
        for y in 0..tile.height() {
            for pixel in tile.row_mut(y) {
                *pixel = process(*pixel);
            }
        }
    });
```

## Neighborhood algorithms: beware halos

Do not hand-roll tiled convolution first. Prefer the existing transform APIs: `convolve`, `fold_neighborhood`, `map_neighborhood`, `morphology`, filters, and template matching. They already encode border behavior.

If you deliberately tile a neighborhood algorithm, each tile usually needs a read margin outside its write region. That margin is the halo. Without it, seams appear at tile boundaries.

## Inner-loop rule

Do not use `pixel_at` as the inner loop for a large contiguous image when row or slice access is available. Use `row`, `row_mut`, `as_slice`, or `as_mut_slice` for cache-friendly processing.

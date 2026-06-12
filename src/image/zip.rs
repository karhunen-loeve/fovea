use crate::error::Error;
use crate::image::ImageView;

/// An iterator that pairs up pixels from two [`ImageView`]s of the same size.
///
/// Iterates in **row-major order** (left-to-right, top-to-bottom), yielding
/// `(A::Pixel, B::Pixel)` for each pixel position. All pixel types implement
/// `Copy`, so returning by value is zero-cost.
///
/// Created by the [`zip_pixels`] free function.
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageView, zip_pixels};
///
/// let a = Image::generate(3, 2, |x, y| (x + y) as u8);
/// let b = Image::generate(3, 2, |x, y| (x * y) as u8);
///
/// let pairs: Vec<_> = zip_pixels(&a, &b).unwrap().collect();
/// assert_eq!(pairs.len(), 6);
/// assert_eq!((pairs[0].0, pairs[0].1), (0, 0)); // (0+0, 0*0)
/// assert_eq!((pairs[1].0, pairs[1].1), (1, 0)); // (1+0, 1*0)
/// ```
#[derive(Clone, Debug)]
pub struct ZipPixelsIter<'a, A: ImageView, B: ImageView> {
    a: &'a A,
    b: &'a B,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
}

impl<'a, A, B> ZipPixelsIter<'a, A, B>
where
    A: ImageView,
    B: ImageView,
{
    fn new(a: &'a A, b: &'a B) -> Self {
        let width = a.width();
        let height = a.height();
        Self {
            a,
            b,
            x: 0,
            y: 0,
            width,
            height,
        }
    }
}

impl<'a, A, B> Iterator for ZipPixelsIter<'a, A, B>
where
    A: ImageView,
    B: ImageView,
{
    type Item = (A::Pixel, B::Pixel);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.y >= self.height {
            return None;
        }

        let pa = self.a.pixel_at(self.x, self.y);
        let pb = self.b.pixel_at(self.x, self.y);

        self.x += 1;
        if self.x >= self.width {
            self.x = 0;
            self.y += 1;
        }

        Some((pa, pb))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = if self.y >= self.height {
            0
        } else {
            (self.height - self.y - 1) * self.width + (self.width - self.x)
        };
        (remaining, Some(remaining))
    }
}

impl<'a, A, B> ExactSizeIterator for ZipPixelsIter<'a, A, B>
where
    A: ImageView,
    B: ImageView,
{
}

/// Pairs up pixels from two [`ImageView`]s of the **same size**, yielding
/// `(A::Pixel, B::Pixel)` in row-major order.
///
/// Returns `Err(Error::SizeMismatch)` if the two images have different
/// dimensions.
///
/// The returned iterator is agnostic about the concrete image types — the
/// inputs can be any combination of [`Image`](crate::image::Image),
/// [`ImageArray`](crate::image::ImageArray), [`ImageRef`](crate::image::ImageRef),
/// or any user type implementing [`ImageView`].
///
/// # Example
///
/// ```
/// use fovea::image::{Image, ImageArray, ImageView, zip_pixels};
///
/// let img = Image::generate(4, 4, |x, y| (x + y * 4) as u8);
/// let arr: ImageArray<u8, 4, 4> = ImageArray::generate(|x, y| (x * y) as u8);
///
/// // Works with mixed types
/// let sum: u32 = zip_pixels(&img, &arr)
///     .unwrap()
///     .map(|(a, b)| a as u32 + b as u32)
///     .sum();
/// assert!(sum > 0);
/// ```
///
/// ```
/// use fovea::image::{Image, ImageView, zip_pixels};
///
/// let a = Image::fill(3, 3, 1u8);
/// let b = Image::fill(4, 3, 1u8);
///
/// // Size mismatch → Err
/// assert!(zip_pixels(&a, &b).is_err());
/// ```
pub fn zip_pixels<'a, A, B>(a: &'a A, b: &'a B) -> Result<ZipPixelsIter<'a, A, B>, Error>
where
    A: ImageView,
    B: ImageView,
{
    if a.size() != b.size() {
        return Err(Error::SizeMismatch {
            expected: a.size(),
            actual: b.size(),
        });
    }

    // Handle zero-sized images: valid, but the iterator will yield nothing.
    Ok(ZipPixelsIter::new(a, b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Size;
    use crate::image::{Image, ImageArray, SubView};
    use crate::pixel::Mono8;

    // ── Basic functionality ──────────────────────────────────────────

    #[test]
    fn test_zip_pixels_same_size() {
        let a = Image::generate(3, 2, |x, y| (x + y) as u8);
        let b = Image::generate(3, 2, |x, y| (x * y) as u8);

        let iter = zip_pixels(&a, &b);
        assert!(iter.is_ok());
        let pairs: Vec<_> = iter.unwrap().collect();
        assert_eq!(pairs.len(), 6);
    }

    #[test]
    fn test_zip_pixels_size_mismatch_returns_none() {
        let a = Image::fill(3, 3, 0u8);
        let b = Image::fill(4, 3, 0u8);
        assert!(zip_pixels(&a, &b).is_err());
    }

    #[test]
    fn test_zip_pixels_width_mismatch() {
        let a = Image::fill(5, 3, 0u8);
        let b = Image::fill(3, 3, 0u8);
        assert!(zip_pixels(&a, &b).is_err());
    }

    #[test]
    fn test_zip_pixels_height_mismatch() {
        let a = Image::fill(3, 5, 0u8);
        let b = Image::fill(3, 3, 0u8);
        assert!(zip_pixels(&a, &b).is_err());
    }

    // ── Pixel pairing correctness ────────────────────────────────────

    #[test]
    fn test_zip_pixels_pairing_correctness() {
        let a = Image::generate(4, 3, |x, y| (x + y * 4) as u8);
        let b = Image::generate(4, 3, |x, y| (x * 10 + y) as u8);

        let pairs: Vec<_> = zip_pixels(&a, &b).unwrap().collect();

        // Verify each pair matches the expected pixel values
        let mut idx = 0;
        for y in 0..3 {
            for x in 0..4 {
                let (pa, pb) = pairs[idx];
                assert_eq!(pa, (x + y * 4) as u8, "pixel a mismatch at ({x}, {y})");
                assert_eq!(pb, (x * 10 + y) as u8, "pixel b mismatch at ({x}, {y})");
                idx += 1;
            }
        }
        assert_eq!(idx, 12);
    }

    // ── Row-major iteration order ────────────────────────────────────

    #[test]
    fn test_zip_pixels_row_major_order() {
        // Create an image where pixel value encodes position: y * width + x
        let a = Image::generate(3, 4, |x, y| (y * 3 + x) as u8);
        let b = Image::fill(3, 4, 0u8);

        let values: Vec<u8> = zip_pixels(&a, &b).unwrap().map(|(pa, _)| pa).collect();

        // Should be [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        let expected: Vec<u8> = (0..12).collect();
        assert_eq!(values, expected);
    }

    // ── Works with ImageRef inputs ───────────────────────────────

    #[test]
    fn test_zip_pixels_with_sequential_roi() {
        let img_a = Image::generate(6, 6, |x, y| Mono8::new((x + y * 6) as u8));
        let img_b = Image::generate(6, 6, |x, y| Mono8::new(((x + y) * 2) as u8));

        // Take 3×3 ROIs from each
        let roi_a = img_a.roi(crate::Rectangle::new((1, 1), (3, 3))).unwrap();
        let roi_b = img_b.roi(crate::Rectangle::new((2, 2), (3, 3))).unwrap();

        let pairs: Vec<_> = zip_pixels(&roi_a, &roi_b).unwrap().collect();
        assert_eq!(pairs.len(), 9);

        // Verify first pixel: roi_a starts at (1,1), roi_b starts at (2,2)
        assert_eq!(pairs[0].0, Mono8::new((1 + 6) as u8)); // img_a[1,1] = 7
        assert_eq!(pairs[0].1, Mono8::new(((2 + 2) * 2) as u8)); // img_b[2,2] = 8
    }

    #[test]
    fn test_zip_pixels_roi_size_mismatch() {
        let img = Image::generate(6, 6, |x, y| Mono8::new((x + y) as u8));
        let roi_a = img.roi(crate::Rectangle::new((0, 0), (3, 3))).unwrap();
        let roi_b = img.roi(crate::Rectangle::new((0, 0), (4, 3))).unwrap();

        assert!(zip_pixels(&roi_a, &roi_b).is_err());
    }

    // ── Works with mixed types (Image + ImageArray) ──────────────────

    #[test]
    fn test_zip_pixels_image_with_imagearray() {
        let img = Image::generate(3, 3, |x, y| (x + y) as u8);
        let arr: ImageArray<u8, 3, 3> = ImageArray::generate(|x, y| (x * y) as u8);

        let pairs: Vec<_> = zip_pixels(&img, &arr).unwrap().collect();
        assert_eq!(pairs.len(), 9);

        // Spot-check: position (2, 1)
        // img: 2+1=3, arr: 2*1=2
        // In row-major: index = 1*3 + 2 = 5
        assert_eq!(pairs[5].0, 3u8);
        assert_eq!(pairs[5].1, 2u8);
    }

    #[test]
    fn test_zip_pixels_imagearray_with_image() {
        let arr: ImageArray<u8, 4, 2> = ImageArray::generate(|x, y| (x + y * 10) as u8);
        let img = Image::generate(4, 2, |x, y| (x * 3 + y) as u8);

        let pairs: Vec<_> = zip_pixels(&arr, &img).unwrap().collect();
        assert_eq!(pairs.len(), 8);
    }

    #[test]
    fn test_zip_pixels_imagearray_size_mismatch() {
        let a: ImageArray<u8, 3, 3> = ImageArray::generate(|_, _| 0u8);
        let b: ImageArray<u8, 4, 3> = ImageArray::generate(|_, _| 0u8);

        assert!(zip_pixels(&a, &b).is_err());
    }

    // ── Works with ImageArray + ImageRef ─────────────────────────

    #[test]
    fn test_zip_pixels_imagearray_with_roi() {
        let arr: ImageArray<Mono8, 3, 3> = ImageArray::generate(|x, y| Mono8::new((x + y) as u8));
        let img = Image::generate(6, 6, |x, y| Mono8::new((x + y * 6) as u8));
        let roi = img.roi(crate::Rectangle::new((0, 0), (3, 3))).unwrap();

        let pairs: Vec<_> = zip_pixels(&arr, &roi).unwrap().collect();
        assert_eq!(pairs.len(), 9);

        // arr[0,0]=0, roi[0,0]=img[0,0]=0
        assert_eq!(pairs[0].0, Mono8::new(0));
        assert_eq!(pairs[0].1, Mono8::new(0));

        // arr[2,2]=4, roi[2,2]=img[2,2]=2+2*6=14
        assert_eq!(pairs[8].0, Mono8::new(4));
        assert_eq!(pairs[8].1, Mono8::new(14));
    }

    // ── ExactSizeIterator ────────────────────────────────────────────

    #[test]
    fn test_zip_pixels_exact_size() {
        let a = Image::fill(5, 4, 1u8);
        let b = Image::fill(5, 4, 2u8);

        let mut iter = zip_pixels(&a, &b).unwrap();
        assert_eq!(iter.len(), 20);

        iter.next();
        assert_eq!(iter.len(), 19);

        // Consume 9 more (10 total consumed)
        for _ in 0..9 {
            iter.next();
        }
        assert_eq!(iter.len(), 10);
    }

    #[test]
    fn test_zip_pixels_exact_size_at_exhaustion() {
        let a = Image::fill(2, 2, 0u8);
        let b = Image::fill(2, 2, 0u8);

        let mut iter = zip_pixels(&a, &b).unwrap();
        assert_eq!(iter.len(), 4);

        for expected_remaining in (0..4).rev() {
            iter.next();
            assert_eq!(iter.len(), expected_remaining);
        }

        assert!(iter.next().is_none());
        assert_eq!(iter.len(), 0);
    }

    // ── Edge cases ───────────────────────────────────────────────────

    #[test]
    fn test_zip_pixels_1x1_images() {
        let a = Image::fill(1, 1, 42u8);
        let b = Image::fill(1, 1, 99u8);

        let pairs: Vec<_> = zip_pixels(&a, &b).unwrap().collect();
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].0, 42u8);
        assert_eq!(pairs[0].1, 99u8);
    }

    #[test]
    fn test_zip_pixels_single_row() {
        let a = Image::generate(5, 1, |x, _| x as u8);
        let b = Image::generate(5, 1, |x, _| (x * 2) as u8);

        let pairs: Vec<_> = zip_pixels(&a, &b).unwrap().collect();
        assert_eq!(pairs.len(), 5);
        for (i, pair) in pairs.iter().enumerate().take(5) {
            assert_eq!(pair.0, i as u8);
            assert_eq!(pair.1, (i * 2) as u8);
        }
    }

    #[test]
    fn test_zip_pixels_single_column() {
        let a = Image::generate(1, 5, |_, y| y as u8);
        let b = Image::generate(1, 5, |_, y| (y * 3) as u8);

        let pairs: Vec<_> = zip_pixels(&a, &b).unwrap().collect();
        assert_eq!(pairs.len(), 5);
        for (i, pair) in pairs.iter().enumerate().take(5) {
            assert_eq!(pair.0, i as u8);
            assert_eq!(pair.1, (i * 3) as u8);
        }
    }

    #[test]
    fn test_zip_pixels_does_not_consume_images() {
        let a = Image::fill(3, 3, 1u8);
        let b = Image::fill(3, 3, 2u8);

        // First zip
        let count1 = zip_pixels(&a, &b).unwrap().count();
        // Second zip — images are not consumed
        let count2 = zip_pixels(&a, &b).unwrap().count();
        assert_eq!(count1, 9);
        assert_eq!(count2, 9);

        // Images are still accessible
        assert_eq!(a.pixel_at(0, 0), 1u8);
        assert_eq!(b.pixel_at(0, 0), 2u8);
    }

    #[test]
    fn test_zip_pixels_large_image() {
        let a = Image::generate(100, 100, |x, y| ((x + y) % 256) as u8);
        let b = Image::generate(100, 100, |x, y| ((x * y) % 256) as u8);

        let mut iter = zip_pixels(&a, &b).unwrap();
        assert_eq!(iter.len(), 10_000);

        let pairs: Vec<_> = iter.by_ref().collect();
        assert_eq!(pairs.len(), 10_000);
        assert_eq!(iter.len(), 0);
    }

    // ── Composability: map / sum / fold ───────────────────────────────

    #[test]
    fn test_zip_pixels_sum_of_products() {
        // Dot product of two 2×2 images
        let a = Image::generate(2, 2, |x, y| (x + y + 1) as u32);
        let b = Image::generate(2, 2, |x, y| (x + y + 1) as u32);

        // a = [1, 2, 2, 3], b = [1, 2, 2, 3]
        // dot = 1*1 + 2*2 + 2*2 + 3*3 = 1 + 4 + 4 + 9 = 18
        let dot: u32 = zip_pixels(&a, &b).unwrap().map(|(pa, pb)| pa * pb).sum();
        assert_eq!(dot, 18);
    }

    #[test]
    fn test_zip_pixels_abs_diff() {
        let a = Image::generate(3, 3, |_, _| 100u8);
        let b = Image::generate(3, 3, |_, _| 60u8);

        let diffs: Vec<u8> = zip_pixels(&a, &b)
            .unwrap()
            .map(|(pa, pb)| pa.abs_diff(pb))
            .collect();

        assert!(diffs.iter().all(|&d| d == 40));
    }

    #[test]
    fn test_zip_pixels_filter() {
        let a = Image::generate(4, 4, |x, y| (x + y) as u8);
        let b = Image::fill(4, 4, 3u8);

        // Count positions where a > b
        let count = zip_pixels(&a, &b)
            .unwrap()
            .filter(|&(pa, pb)| pa > pb)
            .count();

        // a values: row 0: [0,1,2,3], row 1: [1,2,3,4], row 2: [2,3,4,5], row 3: [3,4,5,6]
        // Values > 3: 4, 4,5, 4,5,6 → 6 positions
        assert_eq!(count, 6);
    }

    // ── Different pixel types ────────────────────────────────────────

    #[test]
    fn test_zip_pixels_u16_images() {
        let a = Image::generate(3, 3, |x, y| (x + y * 100) as u16);
        let b = Image::generate(3, 3, |x, y| (x * 1000 + y) as u16);

        let pairs: Vec<_> = zip_pixels(&a, &b).unwrap().collect();
        assert_eq!(pairs.len(), 9);
        assert_eq!(pairs[0].0, 0u16); // (0 + 0*100)
        assert_eq!(pairs[0].1, 0u16); // (0*1000 + 0)
        assert_eq!(pairs[4].0, 101u16); // (1 + 1*100) at (1,1), index 4
        assert_eq!(pairs[4].1, 1001u16); // (1*1000 + 1)
    }

    #[test]
    fn test_zip_pixels_f32_images() {
        use crate::pixel::MonoF32;
        let a = Image::generate(2, 2, |x, y| MonoF32::new((x + y) as f32));
        let b = Image::fill(2, 2, MonoF32::new(0.5));

        let sum: f32 = zip_pixels(&a, &b)
            .unwrap()
            .map(|(pa, pb)| pa.0 + pb.0)
            .sum();

        // a = [0.0, 1.0, 1.0, 2.0], each + 0.5 → [0.5, 1.5, 1.5, 2.5] → sum = 6.0
        assert!((sum - 6.0).abs() < f32::EPSILON);
    }

    // ── Different pixel types between A and B ────────────────────────

    #[test]
    fn test_zip_pixels_different_pixel_types() {
        let a = Image::generate(3, 3, |x, y| (x + y) as u8);
        let b = Image::generate(3, 3, |x, y| (x + y) as f32);

        let results: Vec<f32> = zip_pixels(&a, &b)
            .unwrap()
            .map(|(pa, pb)| pa as f32 + pb)
            .collect();

        assert_eq!(results.len(), 9);
        // (0,0): 0 + 0.0 = 0.0
        assert!((results[0] - 0.0).abs() < f32::EPSILON);
        // (1,0): 1 + 1.0 = 2.0
        assert!((results[1] - 2.0).abs() < f32::EPSILON);
    }

    // ── Iterator trait consistency ────────────────────────────────────

    #[test]
    fn test_zip_pixels_size_hint_consistency() {
        let a = Image::fill(4, 3, 0u8);
        let b = Image::fill(4, 3, 0u8);

        let mut iter = zip_pixels(&a, &b).unwrap();

        for expected in (0..=12).rev() {
            assert_eq!(iter.size_hint(), (expected, Some(expected)));
            assert_eq!(iter.len(), expected);
            if expected > 0 {
                iter.next();
            }
        }
    }

    #[test]
    fn test_zip_pixels_fused_after_exhaustion() {
        let a = Image::fill(1, 1, 0u8);
        let b = Image::fill(1, 1, 0u8);

        let mut iter = zip_pixels(&a, &b).unwrap();
        assert!(iter.next().is_some());
        assert!(iter.next().is_none());
        // Should keep returning None
        assert!(iter.next().is_none());
        assert!(iter.next().is_none());
    }

    // ── Non-square images ────────────────────────────────────────────

    #[test]
    fn test_zip_pixels_wide_image() {
        let a = Image::generate(10, 2, |x, _| x as u8);
        let b = Image::generate(10, 2, |x, _| (9 - x) as u8);

        let sum: u16 = zip_pixels(&a, &b)
            .unwrap()
            .map(|(pa, pb)| pa as u16 + pb as u16)
            .sum();

        // Each row: pairs sum to 9, 10 pairs → 90 per row, 2 rows → 180
        assert_eq!(sum, 180);
    }

    #[test]
    fn test_zip_pixels_tall_image() {
        let a = Image::generate(2, 10, |_, y| y as u8);
        let b = Image::fill(2, 10, 1u8);

        let count = zip_pixels(&a, &b).unwrap().count();
        assert_eq!(count, 20);
    }

    // ── Sliding window + zip_pixels composition ──────────────────────

    #[test]
    fn test_zip_pixels_with_sliding_windows() {
        let img = Image::generate(5, 5, |x, y| Mono8::new((x + y * 5) as u8));
        let template: ImageArray<Mono8, 3, 3> =
            ImageArray::generate(|x, y| Mono8::new((x + y) as u8));

        // Compute SAD (sum of absolute differences) for each window position
        let sads: Vec<u32> = img
            .sliding_windows(Size::new(3, 3))
            .map(|window| {
                zip_pixels(&window, &template)
                    .unwrap()
                    .map(|(a, b)| a.value().abs_diff(b.value()) as u32)
                    .sum()
            })
            .collect();

        assert_eq!(sads.len(), 9); // (5-3+1)² = 9
    }
}

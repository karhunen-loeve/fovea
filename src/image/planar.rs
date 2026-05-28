use crate::image::{Image, ImageView};
use crate::{
    Size,
    error::Error,
    pixel::{Array, HomogeneousPixel, MAX_PIXEL_SIZE, PlainChannel, ZeroablePixel},
};
use std::mem::size_of;

/// Planar image storage — each channel is stored as a separate [`Image`].
///
/// `ImagePlanes<P>` decomposes a pixel type `P` into its constituent channels,
/// storing each as an independent `Image<P::Channel>`. This enables per-channel
/// processing using the full [`ImageView`] / [`ImageViewMut`](crate::image::ImageViewMut) API on each plane.
///
/// # Split → Process → Merge workflow
///
/// ```
/// # use irys_cv::image::{Image, ImageView, ImagePlanes};
/// # use irys_cv::pixel::Rgb8;
/// # use std::num::Saturating;
/// // 1. Start with an interleaved RGB image
/// let original = Image::generate(4, 4, |x, y| {
///     Rgb8::new((x * 60) as u8, (y * 60) as u8, 128)
/// });
///
/// // 2. Split into per-channel planes
/// let mut planes = ImagePlanes::from_interleaved(&original);
///
/// // 3. Process one channel independently
/// let r = planes.plane(0).unwrap();
/// let inverted_r = Image::generate(r.width(), r.height(), |x, y| {
///     Saturating(255u8 - r.pixel_at(x, y).0)
/// });
///
/// // 4. Replace the processed plane
/// let _old = planes.replace_plane(0, inverted_r);
///
/// // 5. Merge back to interleaved
/// let result = planes.to_interleaved();
///
/// // R channel is inverted, G and B unchanged
/// assert_eq!(result.pixel_at(0, 0).r, Saturating(255));
/// assert_eq!(result.pixel_at(0, 0).g, original.pixel_at(0, 0).g);
/// assert_eq!(result.pixel_at(0, 0).b, original.pixel_at(0, 0).b);
/// ```
pub struct ImagePlanes<P: HomogeneousPixel> {
    size: Size,
    planes: <P::Channels as Array<P::Channel>>::Map<Image<P::Channel>>,
}

impl<P: HomogeneousPixel> ImagePlanes<P> {
    // Accessors

    /// Returns the size of the image.
    #[inline]
    pub fn size(&self) -> Size {
        self.size
    }

    /// Returns the width of the image.
    #[inline]
    pub fn width(&self) -> usize {
        self.size.width
    }

    /// Returns the height of the image.
    #[inline]
    pub fn height(&self) -> usize {
        self.size.height
    }

    /// Returns the number of channels (planes) in the image.
    #[inline]
    pub fn channel_count(&self) -> usize {
        P::CHANNEL_COUNT
    }

    // Creators

    /// Generates an `ImagePlanes` by calling `f(x, y)` for each pixel position,
    /// then decomposing the returned pixel into its channel planes.
    ///
    /// This signature matches `Image::generate`, making it easy to switch between
    /// interleaved and planar storage.
    ///
    /// # Example
    /// ```
    /// # use irys_cv::image::ImagePlanes;
    /// # use irys_cv::pixel::Rgb8;
    /// // Create a gradient image
    /// let planes: ImagePlanes<Rgb8> = ImagePlanes::generate(4, 4, |x, y| {
    ///     Rgb8::new((x * 64) as u8, (y * 64) as u8, 128)
    /// });
    /// ```
    pub fn generate(width: usize, height: usize, f: impl Fn(usize, usize) -> P) -> Self {
        let size = Size::new(width, height);
        // First generate all pixels, then transpose into planes
        let mut pixels = Vec::with_capacity(size.area());
        for y in 0..height {
            for x in 0..width {
                pixels.push(f(x, y));
            }
        }

        let planes = <P::Channels as Array<P::Channel>>::Map::from_fn(|channel_index| {
            let data: Vec<_> = pixels.iter().map(|p| p.channel(channel_index)).collect();
            Image::from_vec(width, height, data).expect("data length matches size")
        });
        Self { size, planes }
    }

    // Plane accessors

    /// Returns an immutable reference to the plane at `index`, or `None` if out of bounds.
    pub fn plane(&self, index: usize) -> Option<&Image<P::Channel>> {
        if index < P::CHANNEL_COUNT {
            Some(&self.planes.as_ref()[index])
        } else {
            None
        }
    }

    /// Returns a mutable reference to the plane at `index`, or `None` if out of bounds.
    pub fn plane_mut(&mut self, index: usize) -> Option<&mut Image<P::Channel>> {
        if index < P::CHANNEL_COUNT {
            Some(&mut self.planes.as_mut()[index])
        } else {
            None
        }
    }

    /// Returns an iterator over immutable references to each plane.
    pub fn planes(&self) -> impl Iterator<Item = &Image<P::Channel>> {
        self.planes.as_ref().iter()
    }

    /// Returns an iterator over mutable references to each plane.
    pub fn planes_mut(&mut self) -> impl Iterator<Item = &mut Image<P::Channel>> {
        self.planes.as_mut().iter_mut()
    }

    /// Replaces the plane at `index` with `new_plane`, returning the old plane.
    ///
    /// # Panics
    ///
    /// Panics if `index >= P::CHANNEL_COUNT` or if `new_plane.size() != self.size()`.
    ///
    /// # Example
    /// ```
    /// # use irys_cv::image::{Image, ImageView, ImagePlanes};
    /// # use irys_cv::pixel::Rgb8;
    /// let mut planes = ImagePlanes::<Rgb8>::zero(4, 4);
    /// let new_red = Image::fill(4, 4, std::num::Saturating(255u8));
    /// let old_red = planes.replace_plane(0, new_red);
    /// // old_red is the previous R plane (all zeros)
    /// // planes now has a saturated R channel
    /// ```
    pub fn replace_plane(
        &mut self,
        index: usize,
        new_plane: Image<P::Channel>,
    ) -> Image<P::Channel> {
        assert!(
            index < P::CHANNEL_COUNT,
            "plane index {index} out of bounds for {} channels",
            P::CHANNEL_COUNT
        );
        assert_eq!(
            new_plane.size(),
            self.size,
            "new plane has size {:?}, expected {:?}",
            new_plane.size(),
            self.size
        );
        let slot = &mut self.planes.as_mut()[index];
        std::mem::replace(slot, new_plane)
    }

    /// Constructs an `ImagePlanes` from an array of per-channel images.
    ///
    /// The array shape is enforced at compile time via the `Map` associated type
    /// (e.g., `[Image<Saturating<u8>>; 3]` for `Rgb8`).
    ///
    /// # Panics
    ///
    /// Panics if any plane's size differs from the first plane's size.
    ///
    /// # Example
    /// ```
    /// # use irys_cv::image::{Image, ImageView, ImagePlanes};
    /// # use irys_cv::pixel::Rgb8;
    /// # use std::num::Saturating;
    /// let r = Image::fill(4, 4, Saturating(255u8));
    /// let g = Image::fill(4, 4, Saturating(0u8));
    /// let b = Image::fill(4, 4, Saturating(0u8));
    /// let planes = ImagePlanes::<Rgb8>::from_planes([r, g, b]);
    /// assert_eq!(planes.plane(0).unwrap().pixel_at(0, 0), Saturating(255u8));
    /// ```
    pub fn from_planes(planes: <P::Channels as Array<P::Channel>>::Map<Image<P::Channel>>) -> Self {
        let slices = planes.as_ref();
        assert!(!slices.is_empty(), "pixel type has zero channels");
        let size = slices[0].size();
        for (i, plane) in slices.iter().enumerate().skip(1) {
            assert_eq!(
                plane.size(),
                size,
                "plane {i} has size {:?}, expected {size:?}",
                plane.size()
            );
        }
        Self { size, planes }
    }

    /// Constructs an `ImagePlanes` from a fixed-size array of per-channel
    /// images, returning an `Error` instead of panicking when plane sizes
    /// disagree.
    ///
    /// This is the fallible counterpart to [`from_planes`](Self::from_planes).
    /// Per ADR-0025, plane-size mismatch is caller-supplied data and is
    /// therefore a Tier-2 condition; prefer this constructor whenever plane
    /// sizes originate outside the caller (decoders, IO, FFI).
    ///
    /// # Errors
    ///
    /// Returns [`Error::SizeMismatch`] if any plane has a different size
    /// than the first.
    ///
    /// # Example
    ///
    /// ```
    /// # use irys_cv::image::{Image, ImageView, ImagePlanes};
    /// # use irys_cv::pixel::Rgb8;
    /// # use std::num::Saturating;
    /// let r = Image::fill(4, 4, Saturating(255u8));
    /// let g = Image::fill(4, 4, Saturating(0u8));
    /// let b = Image::fill(4, 4, Saturating(0u8));
    /// let planes = ImagePlanes::<Rgb8>::try_from_array([r, g, b]).unwrap();
    /// assert_eq!(planes.plane(0).unwrap().pixel_at(0, 0), Saturating(255u8));
    /// ```
    pub fn try_from_array(
        planes: <P::Channels as Array<P::Channel>>::Map<Image<P::Channel>>,
    ) -> Result<Self, Error> {
        let slices = planes.as_ref();
        assert!(!slices.is_empty(), "pixel type has zero channels");
        let size = slices[0].size();
        for plane in slices.iter().skip(1) {
            if plane.size() != size {
                return Err(Error::SizeMismatch {
                    expected: size,
                    actual: plane.size(),
                });
            }
        }
        Ok(Self { size, planes })
    }

    /// Constructs an `ImagePlanes` from a `Vec` of per-channel images.
    ///
    /// Returns `Err` if:
    /// - `planes.len() != P::CHANNEL_COUNT` → [`Error::ChannelCountMismatch`]
    /// - Any plane has a different size than the first → [`Error::SizeMismatch`]
    ///
    /// # Example
    /// ```
    /// # use irys_cv::image::{Image, ImageView, ImagePlanes};
    /// # use irys_cv::pixel::Rgb8;
    /// # use std::num::Saturating;
    /// let planes_vec = vec![
    ///     Image::fill(4, 4, Saturating(255u8)),
    ///     Image::fill(4, 4, Saturating(128u8)),
    ///     Image::fill(4, 4, Saturating(0u8)),
    /// ];
    /// let planes = ImagePlanes::<Rgb8>::try_from_planes(planes_vec).unwrap();
    /// assert_eq!(planes.channel_count(), 3);
    /// ```
    pub fn try_from_planes(planes_vec: Vec<Image<P::Channel>>) -> Result<Self, Error> {
        if planes_vec.len() != P::CHANNEL_COUNT {
            return Err(Error::ChannelCountMismatch {
                expected: P::CHANNEL_COUNT,
                actual: planes_vec.len(),
            });
        }
        let size = planes_vec[0].size(); // safe: CHANNEL_COUNT >= 1
        if let Some(bad) = planes_vec.iter().skip(1).find(|p| p.size() != size) {
            return Err(Error::SizeMismatch {
                expected: size,
                actual: bad.size(),
            });
        }
        let mut iter = planes_vec.into_iter();
        let planes = <P::Channels as Array<P::Channel>>::Map::from_fn(|_| {
            iter.next().expect("length already checked")
        });
        Ok(Self { size, planes })
    }

    // Conversions

    /// Creates an `ImagePlanes` from an interleaved `Image` by splitting pixels into planes.
    pub fn from_interleaved(image: &impl ImageView<Pixel = P>) -> Self {
        let size = image.size();
        let planes = <P::Channels as Array<P::Channel>>::Map::from_fn(|channel_index| {
            Image::generate(size.width, size.height, |x, y| {
                image.pixel_at(x, y).channel(channel_index)
            })
        });
        Self { size, planes }
    }

    /// Converts this `ImagePlanes` back to an interleaved `Image`.
    pub fn to_interleaved(&self) -> Image<P> {
        // Assert once outside the hot loop.
        assert!(
            size_of::<P>() <= MAX_PIXEL_SIZE,
            "pixel type is larger than the stack buffer ({} > {})",
            size_of::<P>(),
            MAX_PIXEL_SIZE
        );
        let planes_ref = self.planes.as_ref();
        Image::generate(self.size.width, self.size.height, |x, y| {
            let mut buf = [0u8; MAX_PIXEL_SIZE];
            let bytes = &mut buf[..size_of::<P>()];
            let channel_size = size_of::<P::Channel>();
            for (i, plane) in planes_ref.iter().enumerate() {
                let px = plane.pixel_at(x, y);
                let channel_bytes = px.as_bytes();
                bytes[i * channel_size..(i + 1) * channel_size].copy_from_slice(channel_bytes);
            }
            <P as PlainChannel>::from_bytes(bytes)
                .expect("internal error: constructed byte buf size mismatch")
        })
    }
}

impl<P: HomogeneousPixel> ImagePlanes<P>
where
    P::Channel: ZeroablePixel,
{
    // Creators
    pub fn zero(width: usize, height: usize) -> Self {
        let size = Size::new(width, height);
        let planes =
            <P::Channels as Array<P::Channel>>::Map::from_fn(|_| Image::zero(width, height));
        Self { size, planes }
    }

    /// Creates an `ImagePlanes` filled with the given pixel value.
    pub fn fill(width: usize, height: usize, value: P) -> Self {
        let size = Size::new(width, height);
        let channels = value.to_channels();
        let planes = <P::Channels as Array<P::Channel>>::Map::from_fn(|i| {
            Image::fill(width, height, channels.as_ref()[i])
        });
        Self { size, planes }
    }
}

// ===========================================
// IntoIterator implementations
// ===========================================

/// Iterator that consumes `ImagePlanes` and yields `Image<P::Channel>` for each plane.
pub struct IntoPlanesIter<P: HomogeneousPixel> {
    iter: std::vec::IntoIter<Image<P::Channel>>,
}

impl<P: HomogeneousPixel> Iterator for IntoPlanesIter<P> {
    type Item = Image<P::Channel>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<P: HomogeneousPixel> ExactSizeIterator for IntoPlanesIter<P> {}

/// Consumes `ImagePlanes` and yields `Image<P::Channel>` for each plane.
impl<P: HomogeneousPixel> IntoIterator for ImagePlanes<P> {
    type Item = Image<P::Channel>;
    type IntoIter = IntoPlanesIter<P>;

    fn into_iter(self) -> Self::IntoIter {
        let mut planes = self.planes;
        let mut vec = Vec::with_capacity(P::CHANNEL_COUNT);
        for slot in planes.as_mut().iter_mut() {
            let dummy = Image::from_vec(0, 0, vec![]).expect("empty image is valid");
            vec.push(std::mem::replace(slot, dummy));
        }
        IntoPlanesIter {
            iter: vec.into_iter(),
        }
    }
}

/// Iterator over `&ImagePlanes` that yields `&Image<P::Channel>` for each plane.
pub struct PlanesIter<'a, P: HomogeneousPixel> {
    iter: std::slice::Iter<'a, Image<P::Channel>>,
}

impl<'a, P: HomogeneousPixel> Iterator for PlanesIter<'a, P> {
    type Item = &'a Image<P::Channel>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<P: HomogeneousPixel> ExactSizeIterator for PlanesIter<'_, P> {}

/// Borrows `&ImagePlanes` and yields `&Image<P::Channel>` for each plane.
impl<'a, P: HomogeneousPixel> IntoIterator for &'a ImagePlanes<P> {
    type Item = &'a Image<P::Channel>;
    type IntoIter = PlanesIter<'a, P>;

    fn into_iter(self) -> Self::IntoIter {
        PlanesIter {
            iter: self.planes.as_ref().iter(),
        }
    }
}

/// Iterator over `&mut ImagePlanes` that yields `&mut Image<P::Channel>` for each plane.
pub struct PlanesIterMut<'a, P: HomogeneousPixel> {
    iter: std::slice::IterMut<'a, Image<P::Channel>>,
}

impl<'a, P: HomogeneousPixel> Iterator for PlanesIterMut<'a, P> {
    type Item = &'a mut Image<P::Channel>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<P: HomogeneousPixel> ExactSizeIterator for PlanesIterMut<'_, P> {}

/// Mutably borrows `&mut ImagePlanes` and yields `&mut Image<P::Channel>` for each plane.
impl<'a, P: HomogeneousPixel> IntoIterator for &'a mut ImagePlanes<P> {
    type Item = &'a mut Image<P::Channel>;
    type IntoIter = PlanesIterMut<'a, P>;

    fn into_iter(self) -> Self::IntoIter {
        PlanesIterMut {
            iter: self.planes.as_mut().iter_mut(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::ImageViewMut;
    use crate::pixel::Rgb8;
    use std::num::Saturating;

    // ===========================================
    // ImagePlanes tests (for u8 - single channel)
    // ===========================================

    #[test]
    fn image_planes_u8_plane_returns_correct_size() {
        let planes: ImagePlanes<u8> = ImagePlanes {
            size: Size::new(3, 2),
            planes: [Image::from_vec(3, 2, vec![1u8, 2, 3, 4, 5, 6]).unwrap()],
        };

        let plane = planes.plane(0).unwrap();
        assert_eq!(plane.size(), Size::new(3, 2));
        assert_eq!(plane.width(), 3);
        assert_eq!(plane.height(), 2);
    }

    #[test]
    fn image_planes_u8_plane_pixel_access() {
        let planes: ImagePlanes<u8> = ImagePlanes {
            size: Size::new(2, 2),
            planes: [Image::from_vec(2, 2, vec![10u8, 20, 30, 40]).unwrap()],
        };

        let plane = planes.plane(0).unwrap();
        assert_eq!(plane.pixel_at(0, 0), 10);
        assert_eq!(plane.pixel_at(1, 0), 20);
        assert_eq!(plane.pixel_at(0, 1), 30);
        assert_eq!(plane.pixel_at(1, 1), 40);
    }

    #[test]
    fn image_planes_u8_plane_mut_modify() {
        let mut planes: ImagePlanes<u8> = ImagePlanes {
            size: Size::new(2, 2),
            planes: [Image::from_vec(2, 2, vec![1u8, 2, 3, 4]).unwrap()],
        };

        {
            let plane = planes.plane_mut(0).unwrap();
            *plane.pixel_at_mut(0, 0) = 100;
            *plane.pixel_at_mut(1, 1) = 200;
        }

        let plane = planes.plane(0).unwrap();
        assert_eq!(plane.pixel_at(0, 0), 100);
        assert_eq!(plane.pixel_at(1, 1), 200);
    }

    #[test]
    fn image_planes_u8_planes_iterator() {
        let planes: ImagePlanes<u8> = ImagePlanes {
            size: Size::new(2, 2),
            planes: [Image::from_vec(2, 2, vec![1u8, 2, 3, 4]).unwrap()],
        };

        let plane_views: Vec<_> = planes.planes().collect();
        assert_eq!(plane_views.len(), 1);
        assert_eq!(plane_views[0].size(), Size::new(2, 2));
    }

    #[test]
    fn image_planes_u8_planes_mut_iterator() {
        let mut planes: ImagePlanes<u8> = ImagePlanes {
            size: Size::new(2, 2),
            planes: [Image::from_vec(2, 2, vec![0u8, 0, 0, 0]).unwrap()],
        };

        for plane in planes.planes_mut() {
            *plane.pixel_at_mut(0, 0) = 255;
        }

        let plane = planes.plane(0).unwrap();
        assert_eq!(plane.pixel_at(0, 0), 255);
    }

    // ===========================================
    // ImagePlanes tests (for Rgb8 - three channels)
    // ===========================================

    #[test]
    fn image_planes_rgb8_has_three_planes() {
        let planes: ImagePlanes<Rgb8> = ImagePlanes {
            size: Size::new(2, 2),
            planes: [
                Image::from_vec(
                    2,
                    2,
                    vec![
                        Saturating(10u8),
                        Saturating(20),
                        Saturating(30),
                        Saturating(40),
                    ],
                )
                .unwrap(), // R
                Image::from_vec(
                    2,
                    2,
                    vec![
                        Saturating(50u8),
                        Saturating(60),
                        Saturating(70),
                        Saturating(80),
                    ],
                )
                .unwrap(), // G
                Image::from_vec(
                    2,
                    2,
                    vec![
                        Saturating(90u8),
                        Saturating(100),
                        Saturating(110),
                        Saturating(120),
                    ],
                )
                .unwrap(), // B
            ],
        };

        let plane_views: Vec<_> = planes.planes().collect();
        assert_eq!(plane_views.len(), 3);
    }

    #[test]
    fn image_planes_rgb8_plane_access_by_index() {
        let planes: ImagePlanes<Rgb8> = ImagePlanes {
            size: Size::new(2, 2),
            planes: [
                Image::from_vec(
                    2,
                    2,
                    vec![Saturating(1u8), Saturating(2), Saturating(3), Saturating(4)],
                )
                .unwrap(), // R
                Image::from_vec(
                    2,
                    2,
                    vec![Saturating(5u8), Saturating(6), Saturating(7), Saturating(8)],
                )
                .unwrap(), // G
                Image::from_vec(
                    2,
                    2,
                    vec![
                        Saturating(9u8),
                        Saturating(10),
                        Saturating(11),
                        Saturating(12),
                    ],
                )
                .unwrap(), // B
            ],
        };

        // Access R plane
        let r_plane = planes.plane(0).unwrap();
        assert_eq!(r_plane.pixel_at(0, 0), Saturating(1));
        assert_eq!(r_plane.pixel_at(1, 1), Saturating(4));

        // Access G plane
        let g_plane = planes.plane(1).unwrap();
        assert_eq!(g_plane.pixel_at(0, 0), Saturating(5));
        assert_eq!(g_plane.pixel_at(1, 1), Saturating(8));

        // Access B plane
        let b_plane = planes.plane(2).unwrap();
        assert_eq!(b_plane.pixel_at(0, 0), Saturating(9));
        assert_eq!(b_plane.pixel_at(1, 1), Saturating(12));
    }

    #[test]
    fn image_planes_rgb8_modify_individual_channels() {
        let mut planes: ImagePlanes<Rgb8> = ImagePlanes {
            size: Size::new(2, 2),
            planes: [
                Image::from_vec(2, 2, vec![Saturating(0u8); 4]).unwrap(), // R
                Image::from_vec(2, 2, vec![Saturating(0u8); 4]).unwrap(), // G
                Image::from_vec(2, 2, vec![Saturating(0u8); 4]).unwrap(), // B
            ],
        };

        // Modify R channel
        {
            let r_plane = planes.plane_mut(0).unwrap();
            *r_plane.pixel_at_mut(0, 0) = Saturating(255);
        }

        // Modify G channel
        {
            let g_plane = planes.plane_mut(1).unwrap();
            *g_plane.pixel_at_mut(1, 0) = Saturating(128);
        }

        // Modify B channel
        {
            let b_plane = planes.plane_mut(2).unwrap();
            *b_plane.pixel_at_mut(1, 1) = Saturating(64);
        }

        assert_eq!(planes.plane(0).unwrap().pixel_at(0, 0), Saturating(255));
        assert_eq!(planes.plane(1).unwrap().pixel_at(1, 0), Saturating(128));
        assert_eq!(planes.plane(2).unwrap().pixel_at(1, 1), Saturating(64));
    }

    #[test]
    fn image_planes_rgb8_planes_mut_modify_all() {
        let mut planes: ImagePlanes<Rgb8> = ImagePlanes {
            size: Size::new(2, 2),
            planes: [
                Image::from_vec(2, 2, vec![Saturating(0u8); 4]).unwrap(),
                Image::from_vec(2, 2, vec![Saturating(0u8); 4]).unwrap(),
                Image::from_vec(2, 2, vec![Saturating(0u8); 4]).unwrap(),
            ],
        };

        // Set all pixels in all planes to different values based on plane index
        for (i, plane) in planes.planes_mut().enumerate() {
            let value = Saturating((i as u8 + 1) * 50);
            for y in 0..2 {
                for x in 0..2 {
                    *plane.pixel_at_mut(x, y) = value;
                }
            }
        }

        // R plane should be 50, G plane should be 100, B plane should be 150
        assert_eq!(planes.plane(0).unwrap().pixel_at(0, 0), Saturating(50));
        assert_eq!(planes.plane(1).unwrap().pixel_at(0, 0), Saturating(100));
        assert_eq!(planes.plane(2).unwrap().pixel_at(0, 0), Saturating(150));
    }

    #[test]
    fn image_planes_each_plane_is_image_view() {
        let planes: ImagePlanes<Rgb8> = ImagePlanes {
            size: Size::new(3, 3),
            planes: [
                Image::from_vec(3, 3, vec![Saturating(0u8); 9]).unwrap(),
                Image::from_vec(3, 3, vec![Saturating(0u8); 9]).unwrap(),
                Image::from_vec(3, 3, vec![Saturating(0u8); 9]).unwrap(),
            ],
        };

        // Each plane is &Image, which implements ImageView, so we can use ImageView methods
        fn use_image_view<V: ImageView>(view: &V) -> usize {
            view.width() * view.height()
        }

        let plane = planes.plane(0).unwrap();
        assert_eq!(use_image_view(plane), 9);
    }

    #[test]
    fn image_planes_plane_mut_is_image_view_mut() {
        let mut planes: ImagePlanes<u8> = ImagePlanes {
            size: Size::new(2, 2),
            planes: [Image::from_vec(2, 2, vec![0u8; 4]).unwrap()],
        };

        // &mut Image implements ImageViewMut
        fn modify_image<V: ImageViewMut<Pixel = u8>>(view: &mut V) {
            *view.pixel_at_mut(0, 0) = 42;
        }

        {
            let plane = planes.plane_mut(0).unwrap();
            modify_image(plane);
        }

        assert_eq!(planes.plane(0).unwrap().pixel_at(0, 0), 42);
    }

    // ===========================================
    // ImagePlanes constructor tests (zero/fill)
    // ===========================================

    #[test]
    fn image_planes_u8_zero_creates_correct_size() {
        let planes: ImagePlanes<u8> = ImagePlanes::zero(4, 3);
        let plane = planes.plane(0).unwrap();
        assert_eq!(plane.size(), Size::new(4, 3));
        assert_eq!(plane.width(), 4);
        assert_eq!(plane.height(), 3);
    }

    #[test]
    fn image_planes_u8_zero_all_pixels_are_zero() {
        let planes: ImagePlanes<u8> = ImagePlanes::zero(3, 3);
        let plane = planes.plane(0).unwrap();
        for y in 0..3 {
            for x in 0..3 {
                assert_eq!(plane.pixel_at(x, y), 0);
            }
        }
    }

    #[test]
    fn image_planes_u8_fill_all_pixels_have_value() {
        let planes: ImagePlanes<u8> = ImagePlanes::fill(2, 2, 42u8);
        let plane = planes.plane(0).unwrap();
        assert_eq!(plane.pixel_at(0, 0), 42);
        assert_eq!(plane.pixel_at(1, 0), 42);
        assert_eq!(plane.pixel_at(0, 1), 42);
        assert_eq!(plane.pixel_at(1, 1), 42);
    }

    #[test]
    fn image_planes_rgb8_zero_creates_three_planes() {
        let planes: ImagePlanes<Rgb8> = ImagePlanes::zero(2, 2);
        let plane_views: Vec<_> = planes.planes().collect();
        assert_eq!(plane_views.len(), 3);
    }

    #[test]
    fn image_planes_rgb8_zero_all_channels_are_zero() {
        let planes: ImagePlanes<Rgb8> = ImagePlanes::zero(2, 2);

        for plane_idx in 0..3 {
            let plane = planes.plane(plane_idx).unwrap();
            for y in 0..2 {
                for x in 0..2 {
                    assert_eq!(plane.pixel_at(x, y), Saturating(0u8));
                }
            }
        }
    }

    #[test]
    fn image_planes_rgb8_fill_all_channels_have_value() {
        let planes: ImagePlanes<Rgb8> = ImagePlanes::fill(2, 2, Rgb8::new(128, 128, 128));

        for plane_idx in 0..3 {
            let plane = planes.plane(plane_idx).unwrap();
            for y in 0..2 {
                for x in 0..2 {
                    assert_eq!(plane.pixel_at(x, y), Saturating(128u8));
                }
            }
        }
    }

    #[test]
    fn image_planes_zero_then_modify() {
        let mut planes: ImagePlanes<u8> = ImagePlanes::zero(3, 3);

        // Modify center pixel
        {
            let plane = planes.plane_mut(0).unwrap();
            *plane.pixel_at_mut(1, 1) = 255;
        }

        // Verify only center pixel changed
        let plane = planes.plane(0).unwrap();
        assert_eq!(plane.pixel_at(0, 0), 0);
        assert_eq!(plane.pixel_at(1, 1), 255);
        assert_eq!(plane.pixel_at(2, 2), 0);
    }

    // ===========================================
    // ImagePlanes generate tests
    // ===========================================

    #[test]
    fn image_planes_u8_generate_constant() {
        let planes: ImagePlanes<u8> = ImagePlanes::generate(3, 3, |_, _| 42);

        let plane = planes.plane(0).unwrap();
        for y in 0..3 {
            for x in 0..3 {
                assert_eq!(plane.pixel_at(x, y), 42);
            }
        }
    }

    #[test]
    fn image_planes_u8_generate_with_coordinates() {
        // Generate values based on position: value = x + y * width
        let planes: ImagePlanes<u8> = ImagePlanes::generate(3, 2, |x, y| (x + y * 3) as u8);

        let plane = planes.plane(0).unwrap();
        assert_eq!(plane.pixel_at(0, 0), 0);
        assert_eq!(plane.pixel_at(1, 0), 1);
        assert_eq!(plane.pixel_at(2, 0), 2);
        assert_eq!(plane.pixel_at(0, 1), 3);
        assert_eq!(plane.pixel_at(1, 1), 4);
        assert_eq!(plane.pixel_at(2, 1), 5);
    }

    #[test]
    fn image_planes_rgb8_generate_per_channel() {
        // R = x, G = y, B = constant 100
        let planes: ImagePlanes<Rgb8> =
            ImagePlanes::generate(4, 4, |x, y| Rgb8::new(x as u8, y as u8, 100));

        // Check R plane (channel 0)
        let r_plane = planes.plane(0).unwrap();
        assert_eq!(r_plane.pixel_at(0, 0), Saturating(0));
        assert_eq!(r_plane.pixel_at(3, 0), Saturating(3));
        assert_eq!(r_plane.pixel_at(2, 2), Saturating(2));

        // Check G plane (channel 1)
        let g_plane = planes.plane(1).unwrap();
        assert_eq!(g_plane.pixel_at(0, 0), Saturating(0));
        assert_eq!(g_plane.pixel_at(0, 3), Saturating(3));
        assert_eq!(g_plane.pixel_at(2, 2), Saturating(2));

        // Check B plane (channel 2)
        let b_plane = planes.plane(2).unwrap();
        assert_eq!(b_plane.pixel_at(0, 0), Saturating(100));
        assert_eq!(b_plane.pixel_at(3, 3), Saturating(100));
    }

    #[test]
    fn image_planes_rgb8_generate_gradient() {
        // Each channel gets a different gradient based on pixel construction
        let planes: ImagePlanes<Rgb8> = ImagePlanes::generate(2, 2, |x, y| {
            Rgb8::new((x + y) as u8, (x + y + 50) as u8, (x + y + 100) as u8)
        });

        // R plane: base 0
        assert_eq!(planes.plane(0).unwrap().pixel_at(0, 0), Saturating(0));
        assert_eq!(planes.plane(0).unwrap().pixel_at(1, 1), Saturating(2));

        // G plane: base 50
        assert_eq!(planes.plane(1).unwrap().pixel_at(0, 0), Saturating(50));
        assert_eq!(planes.plane(1).unwrap().pixel_at(1, 1), Saturating(52));

        // B plane: base 100
        assert_eq!(planes.plane(2).unwrap().pixel_at(0, 0), Saturating(100));
        assert_eq!(planes.plane(2).unwrap().pixel_at(1, 1), Saturating(102));
    }

    #[test]
    fn image_planes_generate_correct_size() {
        let planes: ImagePlanes<Rgb8> = ImagePlanes::generate(5, 7, |_, _| Rgb8::new(0, 0, 0));

        assert_eq!(planes.plane(0).unwrap().size(), Size::new(5, 7));
        assert_eq!(planes.plane(1).unwrap().size(), Size::new(5, 7));
        assert_eq!(planes.plane(2).unwrap().size(), Size::new(5, 7));
    }

    #[test]
    fn image_planes_generate_then_modify() {
        let mut planes: ImagePlanes<u8> = ImagePlanes::generate(3, 3, |x, y| (x + y) as u8);

        // Modify a pixel
        {
            let plane = planes.plane_mut(0).unwrap();
            *plane.pixel_at_mut(1, 1) = 255;
        }

        // Original generated values should still be there except for modified pixel
        let plane = planes.plane(0).unwrap();
        assert_eq!(plane.pixel_at(0, 0), 0); // 0 + 0
        assert_eq!(plane.pixel_at(2, 0), 2); // 2 + 0
        assert_eq!(plane.pixel_at(1, 1), 255); // modified
        assert_eq!(plane.pixel_at(2, 2), 4); // 2 + 2
    }

    // ===========================================
    // IntoIterator tests
    // ===========================================

    #[test]
    fn image_planes_into_iter_u8_yields_one_image() {
        let planes: ImagePlanes<u8> = ImagePlanes::generate(3, 2, |x, y| (x + y) as u8);

        let images: Vec<Image<u8>> = planes.into_iter().collect();
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].size(), Size::new(3, 2));
        assert_eq!(images[0].pixel_at(0, 0), 0);
        assert_eq!(images[0].pixel_at(2, 1), 3);
    }

    #[test]
    fn image_planes_into_iter_rgb8_yields_three_images() {
        let planes: ImagePlanes<Rgb8> =
            ImagePlanes::generate(2, 2, |x, y| Rgb8::new((x * 10) as u8, (y * 20) as u8, 100));

        let images: Vec<Image<Saturating<u8>>> = planes.into_iter().collect();
        assert_eq!(images.len(), 3);

        // R plane
        assert_eq!(images[0].size(), Size::new(2, 2));
        assert_eq!(images[0].pixel_at(0, 0), Saturating(0));
        assert_eq!(images[0].pixel_at(1, 0), Saturating(10));

        // G plane
        assert_eq!(images[1].pixel_at(0, 0), Saturating(0));
        assert_eq!(images[1].pixel_at(0, 1), Saturating(20));

        // B plane
        assert_eq!(images[2].pixel_at(0, 0), Saturating(100));
        assert_eq!(images[2].pixel_at(1, 1), Saturating(100));
    }

    #[test]
    fn image_planes_into_iter_for_loop_syntax() {
        let planes: ImagePlanes<Rgb8> = ImagePlanes::zero(2, 2);

        let mut count = 0;
        for image in planes {
            assert_eq!(image.size(), Size::new(2, 2));
            count += 1;
        }
        assert_eq!(count, 3);
    }

    #[test]
    fn image_planes_ref_into_iter_yields_image_refs() {
        let planes: ImagePlanes<Rgb8> =
            ImagePlanes::generate(2, 2, |x, y| Rgb8::new(x as u8, y as u8, 50));

        let views: Vec<_> = (&planes).into_iter().collect();
        assert_eq!(views.len(), 3);

        // Can still use planes after iteration (it was borrowed, not consumed)
        assert_eq!(planes.plane(0).unwrap().size(), Size::new(2, 2));
    }

    #[test]
    fn image_planes_ref_for_loop_syntax() {
        let planes: ImagePlanes<Rgb8> = ImagePlanes::zero(3, 3);

        let mut count = 0;
        for plane in &planes {
            assert_eq!(plane.size(), Size::new(3, 3));
            count += 1;
        }
        assert_eq!(count, 3);

        // planes still usable
        assert_eq!(planes.plane(0).unwrap().width(), 3);
    }

    #[test]
    fn image_planes_mut_ref_into_iter_yields_image_ref_muts() {
        let mut planes: ImagePlanes<Rgb8> = ImagePlanes::zero(2, 2);

        for plane in &mut planes {
            *plane.pixel_at_mut(0, 0) = Saturating(255);
        }

        assert_eq!(planes.plane(0).unwrap().pixel_at(0, 0), Saturating(255));
        assert_eq!(planes.plane(1).unwrap().pixel_at(0, 0), Saturating(255));
        assert_eq!(planes.plane(2).unwrap().pixel_at(0, 0), Saturating(255));
    }

    #[test]
    fn image_planes_mut_ref_for_loop_modify_each_plane_differently() {
        let mut planes: ImagePlanes<Rgb8> = ImagePlanes::zero(2, 2);

        for (i, plane) in (&mut planes).into_iter().enumerate() {
            let value = Saturating((i as u8 + 1) * 50);
            *plane.pixel_at_mut(1, 1) = value;
        }

        assert_eq!(planes.plane(0).unwrap().pixel_at(1, 1), Saturating(50));
        assert_eq!(planes.plane(1).unwrap().pixel_at(1, 1), Saturating(100));
        assert_eq!(planes.plane(2).unwrap().pixel_at(1, 1), Saturating(150));
    }

    #[test]
    fn image_planes_into_iter_exact_size() {
        let planes: ImagePlanes<Rgb8> = ImagePlanes::zero(2, 2);
        let iter = planes.into_iter();
        assert_eq!(iter.len(), 3);
    }

    #[test]
    fn image_planes_ref_into_iter_exact_size() {
        let planes: ImagePlanes<Rgb8> = ImagePlanes::zero(2, 2);
        let iter = (&planes).into_iter();
        assert_eq!(iter.len(), 3);
    }

    #[test]
    fn image_planes_mut_ref_into_iter_exact_size() {
        let mut planes: ImagePlanes<Rgb8> = ImagePlanes::zero(2, 2);
        let iter = (&mut planes).into_iter();
        assert_eq!(iter.len(), 3);
    }

    // ===========================================
    // Accessor tests
    // ===========================================

    #[test]
    fn image_planes_size_accessor() {
        let planes: ImagePlanes<Rgb8> = ImagePlanes::zero(5, 3);
        assert_eq!(planes.size(), Size::new(5, 3));
    }

    #[test]
    fn image_planes_width_accessor() {
        let planes: ImagePlanes<Rgb8> = ImagePlanes::zero(7, 4);
        assert_eq!(planes.width(), 7);
    }

    #[test]
    fn image_planes_height_accessor() {
        let planes: ImagePlanes<Rgb8> = ImagePlanes::zero(7, 4);
        assert_eq!(planes.height(), 4);
    }

    #[test]
    fn image_planes_channel_count_u8() {
        let planes: ImagePlanes<u8> = ImagePlanes::zero(2, 2);
        assert_eq!(planes.channel_count(), 1);
    }

    #[test]
    fn image_planes_channel_count_rgb8() {
        let planes: ImagePlanes<Rgb8> = ImagePlanes::zero(2, 2);
        assert_eq!(planes.channel_count(), 3);
    }

    #[test]
    fn image_planes_plane_out_of_bounds_returns_none() {
        let planes: ImagePlanes<Rgb8> = ImagePlanes::zero(2, 2);
        assert!(planes.plane(0).is_some());
        assert!(planes.plane(1).is_some());
        assert!(planes.plane(2).is_some());
        assert!(planes.plane(3).is_none());
        assert!(planes.plane(100).is_none());
    }

    #[test]
    fn image_planes_plane_mut_out_of_bounds_returns_none() {
        let mut planes: ImagePlanes<Rgb8> = ImagePlanes::zero(2, 2);
        assert!(planes.plane_mut(0).is_some());
        assert!(planes.plane_mut(1).is_some());
        assert!(planes.plane_mut(2).is_some());
        assert!(planes.plane_mut(3).is_none());
        assert!(planes.plane_mut(100).is_none());
    }

    // ===========================================
    // from_interleaved / to_interleaved tests
    // ===========================================

    #[test]
    fn image_planes_from_interleaved_u8() {
        let image = Image::generate(3, 2, |x, y| (x + y * 3) as u8);
        let planes = ImagePlanes::from_interleaved(&image);

        assert_eq!(planes.size(), Size::new(3, 2));
        assert_eq!(planes.channel_count(), 1);

        let plane = planes.plane(0).unwrap();
        assert_eq!(plane.pixel_at(0, 0), 0);
        assert_eq!(plane.pixel_at(1, 0), 1);
        assert_eq!(plane.pixel_at(2, 1), 5);
    }

    #[test]
    fn image_planes_from_interleaved_rgb8() {
        let image = Image::generate(2, 2, |x, y| Rgb8::new((x * 10) as u8, (y * 20) as u8, 100));
        let planes = ImagePlanes::from_interleaved(&image);

        assert_eq!(planes.size(), Size::new(2, 2));
        assert_eq!(planes.channel_count(), 3);

        // R plane
        assert_eq!(planes.plane(0).unwrap().pixel_at(0, 0), Saturating(0));
        assert_eq!(planes.plane(0).unwrap().pixel_at(1, 0), Saturating(10));

        // G plane
        assert_eq!(planes.plane(1).unwrap().pixel_at(0, 0), Saturating(0));
        assert_eq!(planes.plane(1).unwrap().pixel_at(0, 1), Saturating(20));

        // B plane
        assert_eq!(planes.plane(2).unwrap().pixel_at(0, 0), Saturating(100));
        assert_eq!(planes.plane(2).unwrap().pixel_at(1, 1), Saturating(100));
    }

    #[test]
    fn image_planes_to_interleaved_u8() {
        let planes: ImagePlanes<u8> = ImagePlanes::generate(3, 2, |x, y| (x + y * 3) as u8);
        let image = planes.to_interleaved();

        assert_eq!(image.size(), Size::new(3, 2));
        assert_eq!(image.pixel_at(0, 0), 0);
        assert_eq!(image.pixel_at(1, 0), 1);
        assert_eq!(image.pixel_at(2, 1), 5);
    }

    #[test]
    fn image_planes_to_interleaved_rgb8() {
        let planes: ImagePlanes<Rgb8> =
            ImagePlanes::generate(2, 2, |x, y| Rgb8::new((x * 10) as u8, (y * 20) as u8, 100));
        let image = planes.to_interleaved();

        assert_eq!(image.size(), Size::new(2, 2));
        assert_eq!(image.pixel_at(0, 0), Rgb8::new(0, 0, 100));
        assert_eq!(image.pixel_at(1, 0), Rgb8::new(10, 0, 100));
        assert_eq!(image.pixel_at(0, 1), Rgb8::new(0, 20, 100));
        assert_eq!(image.pixel_at(1, 1), Rgb8::new(10, 20, 100));
    }

    #[test]
    fn image_planes_roundtrip_from_to_interleaved() {
        let original = Image::generate(4, 3, |x, y| {
            Rgb8::new((x * 20) as u8, (y * 30) as u8, ((x + y) * 10) as u8)
        });

        let planes = ImagePlanes::from_interleaved(&original);
        let reconstructed = planes.to_interleaved();

        assert_eq!(original.size(), reconstructed.size());
        for y in 0..3 {
            for x in 0..4 {
                assert_eq!(original.pixel_at(x, y), reconstructed.pixel_at(x, y),);
            }
        }
    }

    #[test]
    fn image_planes_fill_with_pixel_value() {
        let pixel = Rgb8::new(10, 20, 30);
        let planes: ImagePlanes<Rgb8> = ImagePlanes::fill(2, 2, pixel);

        // Each plane should have the corresponding channel value
        assert_eq!(planes.plane(0).unwrap().pixel_at(0, 0), Saturating(10)); // R
        assert_eq!(planes.plane(1).unwrap().pixel_at(0, 0), Saturating(20)); // G
        assert_eq!(planes.plane(2).unwrap().pixel_at(0, 0), Saturating(30)); // B

        // All pixels should have the same values
        assert_eq!(planes.plane(0).unwrap().pixel_at(1, 1), Saturating(10));
        assert_eq!(planes.plane(1).unwrap().pixel_at(1, 1), Saturating(20));
        assert_eq!(planes.plane(2).unwrap().pixel_at(1, 1), Saturating(30));
    }

    // ===========================================
    // replace_plane tests
    // ===========================================

    #[test]
    fn image_planes_replace_plane_returns_old() {
        let mut planes = ImagePlanes::<Rgb8>::fill(2, 2, Rgb8::new(10, 20, 30));
        let new_r = Image::fill(2, 2, Saturating(255u8));
        let old_r = planes.replace_plane(0, new_r);

        // Old plane should have the original R value
        assert_eq!(old_r.pixel_at(0, 0), Saturating(10));
        assert_eq!(old_r.pixel_at(1, 1), Saturating(10));

        // New plane should be in place
        assert_eq!(planes.plane(0).unwrap().pixel_at(0, 0), Saturating(255));
        assert_eq!(planes.plane(0).unwrap().pixel_at(1, 1), Saturating(255));

        // Other planes unchanged
        assert_eq!(planes.plane(1).unwrap().pixel_at(0, 0), Saturating(20));
        assert_eq!(planes.plane(2).unwrap().pixel_at(0, 0), Saturating(30));
    }

    #[test]
    #[should_panic(expected = "new plane has size")]
    fn image_planes_replace_plane_wrong_size_panics() {
        let mut planes = ImagePlanes::<Rgb8>::zero(4, 4);
        let wrong_size = Image::fill(3, 3, Saturating(255u8));
        planes.replace_plane(0, wrong_size);
    }

    #[test]
    #[should_panic(expected = "plane index 3 out of bounds")]
    fn image_planes_replace_plane_out_of_bounds_panics() {
        let mut planes = ImagePlanes::<Rgb8>::zero(2, 2);
        let new_plane = Image::fill(2, 2, Saturating(0u8));
        planes.replace_plane(3, new_plane);
    }

    #[test]
    fn image_planes_replace_plane_then_to_interleaved() {
        let mut planes = ImagePlanes::<Rgb8>::fill(2, 2, Rgb8::new(10, 20, 30));
        let new_r = Image::fill(2, 2, Saturating(100u8));
        planes.replace_plane(0, new_r);

        let image = planes.to_interleaved();
        assert_eq!(image.pixel_at(0, 0), Rgb8::new(100, 20, 30));
        assert_eq!(image.pixel_at(1, 1), Rgb8::new(100, 20, 30));
    }

    // ===========================================
    // from_planes tests
    // ===========================================

    #[test]
    fn image_planes_from_planes_rgb8() {
        let r = Image::fill(3, 3, Saturating(100u8));
        let g = Image::fill(3, 3, Saturating(150u8));
        let b = Image::fill(3, 3, Saturating(200u8));
        let planes = ImagePlanes::<Rgb8>::from_planes([r, g, b]);

        assert_eq!(planes.size(), Size::new(3, 3));
        assert_eq!(planes.plane(0).unwrap().pixel_at(0, 0), Saturating(100));
        assert_eq!(planes.plane(1).unwrap().pixel_at(0, 0), Saturating(150));
        assert_eq!(planes.plane(2).unwrap().pixel_at(0, 0), Saturating(200));
    }

    #[test]
    fn image_planes_from_planes_u8() {
        let plane = Image::from_vec(2, 3, vec![1u8, 2, 3, 4, 5, 6]).unwrap();
        let planes = ImagePlanes::<u8>::from_planes([plane]);

        assert_eq!(planes.size(), Size::new(2, 3));
        assert_eq!(planes.plane(0).unwrap().pixel_at(1, 2), 6);
    }

    #[test]
    #[should_panic(expected = "plane 1 has size")]
    fn image_planes_from_planes_mismatched_sizes_panics() {
        let r = Image::fill(3, 3, Saturating(0u8));
        let g = Image::fill(2, 3, Saturating(0u8)); // wrong width
        let b = Image::fill(3, 3, Saturating(0u8));
        let _ = ImagePlanes::<Rgb8>::from_planes([r, g, b]);
    }

    // ===========================================
    // try_from_planes tests
    // ===========================================

    #[test]
    fn image_planes_try_from_planes_success() {
        let planes_vec = vec![
            Image::fill(4, 4, Saturating(10u8)),
            Image::fill(4, 4, Saturating(20u8)),
            Image::fill(4, 4, Saturating(30u8)),
        ];
        let planes = ImagePlanes::<Rgb8>::try_from_planes(planes_vec).unwrap();
        assert_eq!(planes.size(), Size::new(4, 4));
        assert_eq!(planes.channel_count(), 3);
        assert_eq!(planes.plane(0).unwrap().pixel_at(0, 0), Saturating(10));
    }

    #[test]
    fn image_planes_try_from_planes_wrong_count() {
        let planes_vec = vec![
            Image::fill(4, 4, Saturating(0u8)),
            Image::fill(4, 4, Saturating(0u8)),
        ];
        assert!(ImagePlanes::<Rgb8>::try_from_planes(planes_vec).is_err());
    }

    #[test]
    fn image_planes_try_from_planes_mismatched_sizes() {
        let planes_vec = vec![
            Image::fill(4, 4, Saturating(0u8)),
            Image::fill(3, 4, Saturating(0u8)),
            Image::fill(4, 4, Saturating(0u8)),
        ];
        assert!(ImagePlanes::<Rgb8>::try_from_planes(planes_vec).is_err());
    }

    // ===========================================
    // Split -> Process -> Merge workflow test
    // ===========================================

    #[test]
    fn image_planes_split_process_merge_workflow() {
        // 1. Create an interleaved RGB image
        let original = Image::generate(4, 4, |x, y| Rgb8::new((x * 60) as u8, (y * 60) as u8, 128));

        // 2. Split into planes
        let mut planes = ImagePlanes::from_interleaved(&original);

        // 3. Extract a plane, process it independently (e.g., invert the R channel)
        let r_plane = planes.plane(0).unwrap();
        let inverted_r = Image::generate(r_plane.width(), r_plane.height(), |x, y| {
            let ch = r_plane.pixel_at(x, y);
            Saturating(255u8 - ch.0)
        });

        // 4. Replace the processed plane back
        let _old_r = planes.replace_plane(0, inverted_r);

        // 5. Merge back to interleaved
        let result = planes.to_interleaved();

        // 6. Verify: R channel is inverted, G and B unchanged
        for y in 0..4 {
            for x in 0..4 {
                let orig = original.pixel_at(x, y);
                let res = result.pixel_at(x, y);
                // R should be inverted
                assert_eq!(res.r, Saturating(255 - orig.r.0));
                // G and B should be unchanged
                assert_eq!(res.g, orig.g);
                assert_eq!(res.b, orig.b);
            }
        }
    }

    // ===========================================
    // Plane interop with transforms test
    // ===========================================

    #[test]
    fn image_planes_plane_works_with_image_view_functions() {
        fn sum_pixels(img: &impl ImageView<Pixel = Saturating<u8>>) -> u32 {
            let mut sum = 0u32;
            for y in 0..img.height() {
                for x in 0..img.width() {
                    sum += img.pixel_at(x, y).0 as u32;
                }
            }
            sum
        }

        let planes = ImagePlanes::<Rgb8>::fill(2, 2, Rgb8::new(10, 20, 30));
        assert_eq!(sum_pixels(planes.plane(0).unwrap()), 40); // 10 * 4
        assert_eq!(sum_pixels(planes.plane(1).unwrap()), 80); // 20 * 4
        assert_eq!(sum_pixels(planes.plane(2).unwrap()), 120); // 30 * 4
    }

    // ===========================================
    // Clone for Image<T> test
    // ===========================================

    #[test]
    fn image_clone() {
        let original = Image::generate(3, 3, |x, y| (x + y) as u8);
        let cloned = original.clone();

        assert_eq!(original.size(), cloned.size());
        for y in 0..3 {
            for x in 0..3 {
                assert_eq!(original.pixel_at(x, y), cloned.pixel_at(x, y));
            }
        }
    }

    // ── M4: try_from_array ──

    #[test]
    fn try_from_array_returns_size_mismatch_for_uneven_planes() {
        let r = Image::fill(4, 4, Saturating(255u8));
        let g = Image::fill(3, 4, Saturating(0u8)); // wrong size
        let b = Image::fill(4, 4, Saturating(0u8));
        let res = ImagePlanes::<Rgb8>::try_from_array([r, g, b]);
        assert!(matches!(res, Err(crate::Error::SizeMismatch { .. })));
    }

    #[test]
    fn try_from_array_accepts_consistent_planes() {
        let r = Image::fill(4, 4, Saturating(10u8));
        let g = Image::fill(4, 4, Saturating(20u8));
        let b = Image::fill(4, 4, Saturating(30u8));
        let planes = ImagePlanes::<Rgb8>::try_from_array([r, g, b]).unwrap();
        assert_eq!(planes.channel_count(), 3);
        assert_eq!(planes.plane(1).unwrap().pixel_at(0, 0), Saturating(20u8));
    }
}

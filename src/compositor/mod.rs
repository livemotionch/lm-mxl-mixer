//! The compositor: the mixer's pixel engine.
//!
//! The 8-bit engine reads each source in its own format
//! (UYVY, YUY2, v210, NV12, I420, P010, BGRA, BGRx, RGBA, RGBx, AYUV),
//! crops, scales (area filter down, bilinear up), converts and blends
//! directly into the output frame (NV12 or I420; premultiplied BGRA for a
//! transparent canvas), and only for the pixels that are seen. The canvas is
//! drawn in horizontal strips on a small thread pool. Here it draws the
//! preview mosaic.
//!
//! The 10-bit engine ([`v210`], [`render10`], [`kernels10`]) uses the same plan (layers, visibility, occlusion, clips, fills),
//! filter tables and strip pool, drawn into a v210 canvas at 10 bits, which
//! is what MXL carries.
//!
//! - [`layout`]: the layout value and what to draw in a frame (visibility)
//! - [`render`]: drawing an 8-bit frame
//! - [`formats`], [`scale`], [`color`], [`kernels`]: the pieces
//! - [`config`]: rectangles, layers, crop, borders 

pub mod color;
pub mod config;
pub mod formats;
pub mod kernels;
pub mod kernels10;
pub mod layout;
pub mod mixer10;
pub mod pool;
pub mod render;
pub mod render10;
pub mod scale;
pub mod scale10;
pub mod simd10;
#[doc(hidden)]
pub mod testing;
pub mod v210;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests10;

pub use layout::{MixerLayer, MixerLayout};

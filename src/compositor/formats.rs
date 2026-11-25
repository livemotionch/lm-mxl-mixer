//! Source formats the mixer reads directly, described as byte layouts.
//!
//! Every format is read as 8-bit samples in up to three planes. Most are
//! read straight from the frame; two need a row converted first (into a
//! scratch row, only the bytes needed):
//!
//! - v210 is unpacked to UYVY (10 → 8 bits, rounded);
//! - P010 is rounded to NV12.
//!
//! Formats with alpha (BGRA, RGBA, AYUV) are read as they are: ports carry
//! premultiplied alpha (see `crate::pipeline::alpha`), which is what
//! filtering and blending need (filtering straight alpha darkens edges).

use super::color::Model;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SrcFormat {
    Uyvy,
    Yuy2,
    V210,
    Nv12,
    I420,
    P010,
    Bgra,
    Bgrx,
    Rgba,
    Rgbx,
    Ayuv,
}

/// Where one channel's samples are in a (loaded) plane row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Chan {
    pub plane: usize,
    /// Byte of the first sample
    pub offset: usize,
    /// Bytes from one sample to the next
    pub step: usize,
    /// Horizontal subsampling (1 or 2)
    pub hsub: u32,
}

/// Row conversion before reading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Loader {
    /// Read the frame's bytes as they are.
    Direct,
    /// v210 → UYVY
    V210,
    /// 16-bit little-endian samples (10 bits in the top) → 8 bits
    Msb16,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FormatSpec {
    pub planes: usize,
    /// Vertical subsampling of each plane
    pub vsub: [u32; 3],
    /// Bytes from one aligned pixel group to the next, per (loaded) plane;
    /// byte ranges are aligned to it.
    pub group: [usize; 3],
    /// Channels 0–2 (Y, Cb, Cr or R, G, B) and alpha.
    pub chans: [Chan; 3],
    pub alpha: Option<Chan>,
    pub loader: Loader,
    pub rgb: bool,
}

const fn ch(plane: usize, offset: usize, step: usize, hsub: u32) -> Chan {
    Chan {
        plane,
        offset,
        step,
        hsub,
    }
}

impl SrcFormat {
    pub const ALL: [SrcFormat; 11] = [
        SrcFormat::Uyvy,
        SrcFormat::Yuy2,
        SrcFormat::V210,
        SrcFormat::Nv12,
        SrcFormat::I420,
        SrcFormat::P010,
        SrcFormat::Bgra,
        SrcFormat::Bgrx,
        SrcFormat::Rgba,
        SrcFormat::Rgbx,
        SrcFormat::Ayuv,
    ];

    pub fn spec(self) -> FormatSpec {
        let packed422 = |y: usize, u: usize, v: usize, loader| FormatSpec {
            planes: 1,
            vsub: [1, 1, 1],
            group: [if loader == Loader::V210 { 12 } else { 4 }, 0, 0],
            chans: [ch(0, y, 2, 1), ch(0, u, 4, 2), ch(0, v, 4, 2)],
            alpha: None,
            loader,
            rgb: false,
        };
        let semi = |loader| FormatSpec {
            planes: 2,
            vsub: [1, 2, 1],
            group: [1, 2, 0],
            chans: [ch(0, 0, 1, 1), ch(1, 0, 2, 2), ch(1, 1, 2, 2)],
            alpha: None,
            loader,
            rgb: false,
        };
        let packed4 = |c: [usize; 3], a: Option<usize>, rgb: bool| FormatSpec {
            planes: 1,
            vsub: [1, 1, 1],
            group: [4, 0, 0],
            chans: [ch(0, c[0], 4, 1), ch(0, c[1], 4, 1), ch(0, c[2], 4, 1)],
            alpha: a.map(|a| ch(0, a, 4, 1)),
            // Premultiplied already: read as they are.
            loader: Loader::Direct,
            rgb,
        };
        match self {
            SrcFormat::Uyvy => packed422(1, 0, 2, Loader::Direct),
            SrcFormat::V210 => packed422(1, 0, 2, Loader::V210),
            SrcFormat::Yuy2 => packed422(0, 1, 3, Loader::Direct),
            SrcFormat::Nv12 => semi(Loader::Direct),
            SrcFormat::P010 => semi(Loader::Msb16),
            SrcFormat::I420 => FormatSpec {
                planes: 3,
                vsub: [1, 2, 2],
                group: [1, 1, 1],
                chans: [ch(0, 0, 1, 1), ch(1, 0, 1, 2), ch(2, 0, 1, 2)],
                alpha: None,
                loader: Loader::Direct,
                rgb: false,
            },
            SrcFormat::Bgra => packed4([2, 1, 0], Some(3), true),
            SrcFormat::Bgrx => packed4([2, 1, 0], None, true),
            SrcFormat::Rgba => packed4([0, 1, 2], Some(3), true),
            SrcFormat::Rgbx => packed4([0, 1, 2], None, true),
            SrcFormat::Ayuv => packed4([1, 2, 3], Some(0), false),
        }
    }

    /// The same layout without alpha, for frames that are all opaque.
    pub fn without_alpha(self) -> Option<SrcFormat> {
        match self {
            SrcFormat::Bgra => Some(SrcFormat::Bgrx),
            SrcFormat::Rgba => Some(SrcFormat::Rgbx),
            _ => None,
        }
    }

    pub fn has_alpha(self) -> bool {
        self.spec().alpha.is_some()
    }

    /// Bytes of a loaded row of `plane` for a frame `width` pixels wide.
    pub fn loaded_row_bytes(self, plane: usize, width: usize) -> usize {
        let cw = width.div_ceil(2);
        match (self, plane) {
            // Whole 6-pixel groups: v210 rows are padded to them.
            (SrcFormat::V210, _) => width.div_ceil(6) * 12,
            (SrcFormat::Uyvy | SrcFormat::Yuy2, _) => cw * 4,
            (SrcFormat::Nv12 | SrcFormat::P010, 0) => width,
            (SrcFormat::Nv12 | SrcFormat::P010, _) => cw * 2,
            (SrcFormat::I420, 0) => width,
            (SrcFormat::I420, _) => cw,
            _ => width * 4,
        }
    }

    /// Model of a YUV format with the given matrix and range, or RGB.
    pub fn model(self, yuv: (f64, f64), full_range: bool) -> Model {
        if self.spec().rgb {
            Model::Rgb
        } else {
            Model::Yuv {
                kr: yuv.0,
                kb: yuv.1,
                full_range,
            }
        }
    }
}

/// `round(x / 255)` for `x` in `0..=65535`.
#[inline(always)]
pub fn div255(x: u32) -> u32 {
    let x = x + 128;
    (x + (x >> 8)) >> 8
}

/// Convert bytes `b0..b1` of a loaded row (`b0`/`b1` aligned to the
/// format's pixel group, `b1` within the loaded row) from the frame's row
/// `src` into `out[..b1 - b0]`.
pub fn load(loader: Loader, src: &[u8], b0: usize, b1: usize, out: &mut [u8]) {
    let out = &mut out[..b1 - b0];
    match loader {
        Loader::Direct => out.copy_from_slice(&src[b0..b1]),
        Loader::Msb16 => {
            let src = &src[b0 * 2..b1 * 2];
            for (o, s) in out.iter_mut().zip(src.as_chunks::<2>().0.iter()) {
                let v = u16::from_le_bytes([s[0], s[1]]) as u32;
                *o = ((v + 128) >> 8).min(255) as u8;
            }
        }
        Loader::V210 => {
            // 12 UYVY bytes (6 pixels) per 16 v210 bytes, each sample
            // rounded to 8 bits.
            let g0 = b0 / 12;
            let groups = (b1 - b0) / 12;
            let src = &src[g0 * 16..g0 * 16 + groups * 16];
            // Straight to bytes in one pass where the CPU can (SSSE3), else
            // through 10-bit samples.
            #[cfg(target_arch = "x86_64")]
            // SAFETY: the CPU has the features (checked at run time).
            let done = match super::kernels::level() {
                2 => unsafe { super::simd10::unpack_words_u8_avx2(src, out) },
                1 => unsafe { super::simd10::unpack_words_u8_ssse3(src, out) },
                _ => 0,
            };
            #[cfg(not(target_arch = "x86_64"))]
            let done = 0;
            let src = &src[4 * done..];
            let rest = &mut out[3 * done..groups * 12];
            V210_ROW.with(|buf| {
                let buf = &mut *buf.borrow_mut();
                if buf.len() < rest.len() {
                    buf.resize(rest.len(), 0);
                }
                let tmp = &mut buf[..rest.len()];
                super::v210::unpack_words(src, tmp);
                ten_to_eight(tmp, rest);
            });
        }
    }
}

thread_local! {
    static V210_ROW: std::cell::RefCell<Vec<u16>> = const { std::cell::RefCell::new(Vec::new()) };
}

super::kernels::multiversion! {
    /// 10-bit samples to 8 bits, rounded: `min((v + 2) >> 2, 255)`.
    pub fn ten_to_eight(src: &[u16], out: &mut [u8]) {
        for (o, &v) in out.iter_mut().zip(src) {
            *o = ((v as u32 + 2) >> 2).min(255) as u8;
        }
    }
}

/// Whether every 4-byte pixel of `row` has alpha 255 at byte `alpha`.
pub fn all_opaque(row: &[u8], alpha: usize) -> bool {
    let mask: u64 = 0xFF_u64 << (8 * alpha) | 0xFF_u64 << (8 * (alpha + 4));
    let (blocks, tail) = row.as_chunks::<64>();
    for b in blocks {
        let mut and = u64::MAX;
        for w in b.as_chunks::<8>().0 {
            and &= u64::from_le_bytes(*w);
        }
        if and & mask != mask {
            return false;
        }
    }
    tail.as_chunks::<4>().0.iter().all(|px| px[alpha] == 255)
}

#[cfg(test)]
mod tests {

    #[test]
    fn v210_rows_load_like_the_scalar_rounding() {
        // Long rows take the SIMD path; every length up to a few steps.
        let mut x = 99u32;
        for groups in [1usize, 2, 3, 5, 8, 33, 320] {
            let mut row = vec![0u8; 16 * groups];
            let mut want = vec![0u8; 12 * groups];
            for (w, chunk) in row.chunks_exact_mut(4).enumerate() {
                let mut v = 0u32;
                for f in 0..3 {
                    x ^= x << 13;
                    x ^= x >> 17;
                    x ^= x << 5;
                    let s = x % 1024;
                    v |= s << (10 * f);
                    want[3 * w + f as usize] = ((s + 2) >> 2).min(255) as u8;
                }
                chunk.copy_from_slice(&v.to_le_bytes());
            }
            let mut out = vec![0u8; 12 * groups];
            load(Loader::V210, &row, 0, 12 * groups, &mut out);
            assert_eq!(out, want, "{groups} groups");
        }
    }

    use super::*;

    #[test]
    fn div255_is_exact() {
        for x in 0..=65535u32 {
            assert_eq!(div255(x), (x as f64 / 255.0).round() as u32, "{x}");
        }
    }

    #[test]
    fn opaque_rows() {
        let mut row = vec![255u8; 4 * 40];
        assert!(all_opaque(&row, 3));
        row[4 * 39 + 3] = 254;
        assert!(!all_opaque(&row, 3));
        row[4 * 39 + 3] = 255;
        row[4 * 3 + 3] = 0;
        assert!(!all_opaque(&row, 3));
        assert!(all_opaque(&row, 0));
    }

    #[test]
    fn v210_unpacks_to_uyvy() {
        // Six pixels: Cb=512 Y=64.. etc.
        let pack = |a: u32, b: u32, c: u32| (a | (b << 10) | (c << 20)).to_le_bytes();
        let mut src = Vec::new();
        src.extend(pack(512, 64, 520)); // Cb0 Y0 Cr0
        src.extend(pack(940, 400, 100)); // Y1 Cb2 Y2
        src.extend(pack(600, 1019, 4)); // Cr2 Y3 Cb4
        src.extend(pack(1, 1023, 3)); // Y4 Cr4 Y5
        let mut out = [0u8; 12];
        load(Loader::V210, &src, 0, 12, &mut out);
        assert_eq!(out, [128, 16, 130, 235, 100, 25, 150, 255, 1, 0, 255, 1]);
    }

    #[test]
    fn msb16_rounds() {
        let src: Vec<u8> = [0u16, 0xFFC0, 0x8000, 0x7F80]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let mut out = [0u8; 4];
        load(Loader::Msb16, &src, 0, 4, &mut out);
        assert_eq!(out, [0, 255, 128, 128]);
    }
}

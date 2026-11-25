//! 10-bit kernels on unpacked samples (UYVY order, one `u16` per sample, see
//! [`super::v210`]): blends with a level per sample, keys, and the colour
//! of fills.
//!
//! Levels (alpha, opacity, mix) are Q15: `0..=32768`, 32768 = opaque.

// Index loops over several slices at once are intentional here.
#![allow(clippy::needless_range_loop)]

use super::kernels::{level, multiversion};

/// Q15 one.
pub const ONE: u32 = 32768;

/// `dst = dst + (src − dst) · a / 32768`, one level for all samples.
pub fn mix16(dst: &mut [u16], src: &[u16], a: u32) {
    #[allow(unused_assignments, unused_mut)]
    let mut done = 0;
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: the CPU has the features (checked at run time).
        done = match level() {
            2 => unsafe { super::simd10::mix16_avx2(dst, src, a) },
            1 => unsafe { super::simd10::mix16_ssse3(dst, src, a) },
            _ => 0,
        };
    }
    mix16_scalar(&mut dst[done..], &src[done.min(src.len())..], a)
}

pub fn mix16_scalar(dst: &mut [u16], src: &[u16], a: u32) {
    let n = dst.len().min(src.len());
    let a = a.min(ONE) as i32;
    for i in 0..n {
        let d = dst[i] as i32;
        dst[i] = (d + (((src[i] as i32 - d) * a + 16384) >> 15)) as u16;
    }
}

/// `dst = dst + (src − dst) · alpha / 32768`, a level per sample.
pub fn blend16(dst: &mut [u16], src: &[u16], alpha: &[u16]) {
    #[allow(unused_assignments, unused_mut)]
    let mut done = 0;
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: the CPU has the features (checked at run time).
        done = match level() {
            2 => unsafe { super::simd10::blend16_avx2(dst, src, alpha) },
            1 => unsafe { super::simd10::blend16_ssse3(dst, src, alpha) },
            _ => 0,
        };
    }
    let n = dst.len().min(src.len()).min(alpha.len());
    blend16_scalar(&mut dst[done..n], &src[done..n], &alpha[done..n])
}

pub fn blend16_scalar(dst: &mut [u16], src: &[u16], alpha: &[u16]) {
    let n = dst.len().min(src.len()).min(alpha.len());
    for i in 0..n {
        let d = dst[i] as i32;
        let a = alpha[i].min(ONE as u16) as i32;
        dst[i] = (d + (((src[i] as i32 - d) * a + 16384) >> 15)) as u16;
    }
}

multiversion! {
    /// Levels per sample from levels per pixel (`px` covers whole pixel
    /// pairs): a luma sample takes its pixel's level, the pair's chroma the
    /// mean of the two.
    pub fn alpha_to_samples(px: &[u16], out: &mut [u16]) {
        for (p, o) in px.chunks_exact(2).zip(out.chunks_exact_mut(4)) {
            let (a0, a1) = (p[0], p[1]);
            let c = ((a0 as u32 + a1 as u32 + 1) / 2) as u16;
            o[0] = c;
            o[1] = a0;
            o[2] = c;
            o[3] = a1;
        }
    }
}

multiversion! {
    /// Scale levels by `opacity` (Q15).
    pub fn scale_alpha(alpha: &mut [u16], opacity: u32) {
        if opacity >= ONE {
            return;
        }
        for a in alpha.iter_mut() {
            *a = ((*a as u32 * opacity + 16384) >> 15) as u16;
        }
    }
}

multiversion! {
    /// `a = a · m` (Q15 levels).
    pub fn mul_alpha(alpha: &mut [u16], m: &[u16]) {
        for (a, &m) in alpha.iter_mut().zip(m) {
            *a = ((*a as u32 * m as u32 + 16384) >> 15) as u16;
        }
    }
}

multiversion! {
    /// Whether every level is 0 (`Some(false)`), every level is one
    /// (`Some(true)`), or neither (`None`).
    pub fn uniform(alpha: &[u16]) -> Option<bool> {
        let (mut or, mut and) = (0u16, u16::MAX);
        for &a in alpha {
            or |= a;
            // Levels are at most 32768: bit 15 is set only for "one".
            and &= a;
        }
        if or == 0 {
            Some(false)
        } else if and & 0x8000 != 0 {
            Some(true)
        } else {
            None
        }
    }
}

/// How a key level is made from a signal (a key plane or the fill's luma),
/// as 10-bit levels: `a = clamp((v − lo) · gain / span)`, inverted if asked.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KeyShape {
    /// Signal level where the key starts (10-bit)
    pub lo: f32,
    /// Signal range from transparent to opaque (10-bit, > 0)
    pub span: f32,
    pub invert: bool,
}

impl KeyShape {
    /// A key plane in video range (64 → 0, 940 → 1), with `clip` (0–1:
    /// raises the level where the key starts) and `gain` (≥ 1: steeper).
    pub fn linear(clip: f32, gain: f32, invert: bool) -> KeyShape {
        let clip = clip.clamp(0.0, 1.0);
        let gain = gain.max(0.01);
        let lo = 64.0 + clip * 876.0;
        KeyShape {
            lo,
            span: ((940.0 - lo) / gain).max(1.0),
            invert,
        }
    }

    /// A luma key: transparent below `clip`, opaque `soft` above it (both
    /// fractions of the video range).
    pub fn luma(clip: f32, soft: f32, invert: bool) -> KeyShape {
        KeyShape {
            lo: 64.0 + clip.clamp(0.0, 1.0) * 876.0,
            span: (soft.clamp(0.0, 1.0) * 876.0).max(1.0),
            invert,
        }
    }

    /// Q15 level of a signal value.
    #[inline]
    pub fn level(&self, v: u16) -> u16 {
        let a = ((v as f32 - self.lo) / self.span).clamp(0.0, 1.0);
        let a = if self.invert { 1.0 - a } else { a };
        (a * ONE as f32).round() as u16
    }

    /// A lookup table over all 10-bit values.
    pub fn table(&self) -> Box<[u16; 1024]> {
        let mut t = Box::new([0u16; 1024]);
        for (v, a) in t.iter_mut().enumerate() {
            *a = self.level(v as u16);
        }
        t
    }
}

multiversion! {
    /// Per-pixel levels of a luma key from unpacked samples (UYVY order).
    pub fn luma_key(samples: &[u16], table: &[u16; 1024], out: &mut [u16]) {
        for (p, o) in samples.chunks_exact(2).zip(out.iter_mut()) {
            *o = table[(p[1] & 0x3FF) as usize];
        }
    }
}

multiversion! {
    /// Per-pixel levels from key samples.
    pub fn key_levels(key: &[u16], table: &[u16; 1024], out: &mut [u16]) {
        for (k, o) in key.iter().zip(out.iter_mut()) {
            *o = table[(*k & 0x3FF) as usize];
        }
    }
}

/// BT.709 video-range 10-bit `[Y, Cb, Cr]` of an ARGB colour.
pub fn argb_to_ycbcr10(argb: u32) -> [u16; 3] {
    let r = ((argb >> 16) & 0xFF) as f64 / 255.0;
    let g = ((argb >> 8) & 0xFF) as f64 / 255.0;
    let b = (argb & 0xFF) as f64 / 255.0;
    let (kr, kb) = (0.2126, 0.0722);
    let y = kr * r + (1.0 - kr - kb) * g + kb * b;
    let cb = (b - y) / (2.0 * (1.0 - kb));
    let cr = (r - y) / (2.0 * (1.0 - kr));
    [
        (64.0 + 876.0 * y).round().clamp(4.0, 1019.0) as u16,
        (512.0 + 896.0 * cb).round().clamp(4.0, 1019.0) as u16,
        (512.0 + 896.0 * cr).round().clamp(4.0, 1019.0) as u16,
    ]
}

/// Black in 10-bit video range.
pub const BLACK10: [u16; 3] = [64, 512, 512];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colours() {
        assert_eq!(argb_to_ycbcr10(0xFF00_0000), BLACK10);
        assert_eq!(argb_to_ycbcr10(0xFFFF_FFFF), [940, 512, 512]);
        // 100% red in BT.709 10-bit: Y 250, Cb 409, Cr 960
        assert_eq!(argb_to_ycbcr10(0xFFFF_0000), [250, 409, 960]);
    }

    #[test]
    fn keys() {
        let k = KeyShape::linear(0.0, 1.0, false);
        assert_eq!(k.level(64), 0);
        assert_eq!(k.level(940), ONE as u16);
        assert_eq!(k.level(1023), ONE as u16);
        assert_eq!(k.level(0), 0);
        assert!((k.level(502) as i32 - 16384).abs() <= 1);
        let inv = KeyShape::linear(0.0, 1.0, true);
        assert_eq!(inv.level(64), ONE as u16);
        let l = KeyShape::luma(0.5, 0.1, false);
        assert_eq!(l.level(64 + 438), 0);
        assert_eq!(l.level(64 + 438 + 88), ONE as u16);
    }

    #[test]
    fn blends() {
        let mut d = vec![100u16, 200, 900, 64];
        blend16(&mut d, &[900, 200, 100, 940], &[0, 16384, 32768, 16384]);
        assert_eq!(d, vec![100, 200, 100, 502]);
        let mut s = [0u16; 8];
        alpha_to_samples(&[0, 32768, 100, 300], &mut s);
        assert_eq!(s, [16384, 0, 16384, 32768, 200, 100, 200, 300]);
        assert_eq!(uniform(&[0, 0]), Some(false));
        assert_eq!(uniform(&[32768, 32768]), Some(true));
        assert_eq!(uniform(&[0, 32768]), None);
    }

    #[test]
    fn simd_matches_scalar() {
        let mut x = 12345u32;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x
        };
        let n = 1000;
        let d: Vec<u16> = (0..n).map(|_| (next() % 1024) as u16).collect();
        let s: Vec<u16> = (0..n).map(|_| (next() % 1024) as u16).collect();
        let a: Vec<u16> = (0..n).map(|i| [0, 32768, (next() % 32769) as u16][i % 3]).collect();
        let (mut d1, mut d2) = (d.clone(), d.clone());
        blend16(&mut d1, &s, &a);
        blend16_scalar(&mut d2, &s, &a);
        assert_eq!(d1, d2);
        for level in [0u32, 1, 16384, 32767, 32768] {
            let (mut d1, mut d2) = (d.clone(), d.clone());
            mix16(&mut d1, &s, level);
            mix16_scalar(&mut d2, &s, level);
            assert_eq!(d1, d2, "{level}");
        }
    }
}

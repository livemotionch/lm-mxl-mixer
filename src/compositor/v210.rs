//! v210 (10-bit 4:2:2, MXL's `video/v210`) and the key plane of `video/v210a`.
//!
//! A v210 row is a run of little-endian 32-bit words, each holding three
//! 10-bit samples (bits 0–9, 10–19, 20–29). The samples are in UYVY order:
//! `Cb0 Y0 Cr0 Y1 Cb2 Y2 Cr2 Y3 ...`, so 6 pixels (12 samples) fill one
//! 16-byte block, and a row is padded to 128 bytes (48 pixels).
//!
//! Everything here works on *samples* in that order ("UYVY16": one `u16`
//! per sample): pixel pair `q` is samples `4q..4q+4` (Cb, Y, Cr, Y), pixel
//! `x` (even) starts at sample `2x`. Most operations of a vision mixer don't
//! care which component a sample is: a cut copies them, a mix or a key with
//! one level blends every sample the same way. So they run on whole words
//! straight in the v210 rows (no unpacking), and only the words a span
//! starts or ends in (a span on a pixel that isn't a multiple of 3) are
//! unpacked and packed again. See `docs/v210.md` for the measurements that
//! led here.
//!
//! The key plane of v210a: three 10-bit samples per word, one per pixel,
//! rows 4-byte aligned. Keys have the fill's range (64–940 for video range).

// Index loops over several slices at once are intentional here.
#![allow(clippy::needless_range_loop)]

use super::kernels::level;
#[cfg(target_arch = "x86_64")]
use super::simd10;

/// Bytes of a v210 row of `width` pixels (128-byte aligned).
pub fn line_bytes(width: usize) -> usize {
    width.div_ceil(48) * 128
}

/// Bytes of a v210a key row of `width` pixels (3 samples per word).
pub fn key_line_bytes(width: usize) -> usize {
    width.div_ceil(3) * 4
}

/// Bytes of a v210 frame, and of a v210a frame (fill, then key).
pub fn frame_bytes(width: usize, height: usize, with_key: bool) -> usize {
    line_bytes(width) * height + if with_key { key_line_bytes(width) * height } else { 0 }
}

const MASK: u32 = 0x3FF;

#[inline(always)]
fn get_word(row: &[u8], w: usize) -> u32 {
    u32::from_le_bytes([row[4 * w], row[4 * w + 1], row[4 * w + 2], row[4 * w + 3]])
}

#[inline(always)]
fn put_word(row: &mut [u8], w: usize, v: u32) {
    row[4 * w..4 * w + 4].copy_from_slice(&v.to_le_bytes());
}

#[inline(always)]
fn field(v: u32, f: usize) -> u16 {
    ((v >> (10 * f)) & MASK) as u16
}

#[inline(always)]
fn with_field(v: u32, f: usize, s: u16) -> u32 {
    (v & !(MASK << (10 * f))) | ((s as u32 & MASK) << (10 * f))
}

/// Sample `s` of a v210 row.
#[inline]
pub fn sample(row: &[u8], s: usize) -> u16 {
    field(get_word(row, s / 3), s % 3)
}

/// Set sample `s` of a v210 row.
#[inline]
pub fn set_sample(row: &mut [u8], s: usize, v: u16) {
    let w = s / 3;
    put_word(row, w, with_field(get_word(row, w), s % 3, v));
}

/// The whole words of the sample range `s0..s1`: `(first word, end word)`,
/// and the samples before and after them that sit in partly covered words.
#[inline]
fn whole_words(s0: usize, s1: usize) -> (usize, usize) {
    let w0 = s0.div_ceil(3);
    let w1 = s1 / 3;
    if w0 >= w1 {
        (w0, w0)
    } else {
        (w0, w1)
    }
}

/// Apply `f(sample index, value) -> value` to the samples of `s0..s1` that
/// are not in whole words of the range (at most 2 at each end).
#[inline]
fn edges(row: &mut [u8], s0: usize, s1: usize, mut f: impl FnMut(usize, u16) -> u16) {
    let (w0, w1) = whole_words(s0, s1);
    let head_end = if w0 == w1 { s1 } else { (w0 * 3).min(s1) };
    for s in s0..head_end {
        let v = f(s, sample(row, s));
        set_sample(row, s, v);
    }
    if w0 != w1 {
        for s in (w1 * 3).max(head_end)..s1 {
            let v = f(s, sample(row, s));
            set_sample(row, s, v);
        }
    }
}

/// Unpack samples `s0..s0 + out.len()` of a v210 row.
pub fn unpack(row: &[u8], s0: usize, out: &mut [u16]) {
    let s1 = s0 + out.len();
    let (w0, w1) = whole_words(s0, s1);
    let head_end = if w0 == w1 { s1 } else { (w0 * 3).min(s1) };
    for s in s0..head_end {
        out[s - s0] = sample(row, s);
    }
    if w0 != w1 {
        let o = w0 * 3 - s0;
        unpack_words(&row[4 * w0..4 * w1], &mut out[o..o + 3 * (w1 - w0)]);
        for s in (w1 * 3)..s1 {
            out[s - s0] = sample(row, s);
        }
    }
}

/// Pack `inp` into samples `s0..s0 + inp.len()` of a v210 row (the other
/// samples of the words at the ends are kept).
pub fn pack(row: &mut [u8], s0: usize, inp: &[u16]) {
    let s1 = s0 + inp.len();
    edges(row, s0, s1, |s, _| inp[s - s0]);
    let (w0, w1) = whole_words(s0, s1);
    if w0 != w1 {
        let o = w0 * 3 - s0;
        pack_words(&inp[o..o + 3 * (w1 - w0)], &mut row[4 * w0..4 * w1]);
    }
}

/// Copy samples `s0..s1` from `src` to `dst` (the same positions).
pub fn copy(dst: &mut [u8], src: &[u8], s0: usize, s1: usize) {
    if s0 >= s1 {
        return;
    }
    edges(dst, s0, s1, |s, _| sample(src, s));
    let (w0, w1) = whole_words(s0, s1);
    dst[4 * w0..4 * w1].copy_from_slice(&src[4 * w0..4 * w1]);
}

/// Fill samples `s0..s1` with a colour (`[Cb, Y, Cr]`).
pub fn fill(dst: &mut [u8], s0: usize, s1: usize, ycbcr: [u16; 3]) {
    if s0 >= s1 {
        return;
    }
    let at = |s: usize| match s % 4 {
        0 => ycbcr[1],
        1 | 3 => ycbcr[0],
        _ => ycbcr[2],
    };
    edges(dst, s0, s1, |s, _| at(s));
    let (w0, w1) = whole_words(s0, s1);
    // Four words (12 samples) repeat.
    let words: [u32; 4] = std::array::from_fn(|k| {
        let w = w0 + k;
        (0..3).fold(0u32, |v, f| v | ((at(3 * w + f) as u32) << (10 * f)))
    });
    let mut block = [0u8; 16];
    for k in 0..4 {
        block[4 * k..4 * k + 4].copy_from_slice(&words[k].to_le_bytes());
    }
    let bytes = &mut dst[4 * w0..4 * w1];
    let mut chunks = bytes.chunks_exact_mut(16);
    for c in &mut chunks {
        c.copy_from_slice(&block);
    }
    let rest = chunks.into_remainder();
    let n = rest.len();
    rest.copy_from_slice(&block[..n]);
}

/// `dst = dst + (src − dst) · a / 32768` for samples `s0..s1`, `a` in
/// `0..=32768` (0: keep `dst`, 32768: copy `src`).
pub fn mix(dst: &mut [u8], src: &[u8], s0: usize, s1: usize, a: u32) {
    if s0 >= s1 || a == 0 {
        return;
    }
    if a >= 32768 {
        return copy(dst, src, s0, s1);
    }
    edges(dst, s0, s1, |s, d| lerp(d, sample(src, s), a));
    let (w0, w1) = whole_words(s0, s1);
    mix_words(&mut dst[4 * w0..4 * w1], &src[4 * w0..4 * w1], a);
}

/// `dst = dst + (colour − dst) · a / 32768` for samples `s0..s1`.
pub fn mix_color(dst: &mut [u8], s0: usize, s1: usize, ycbcr: [u16; 3], a: u32) {
    if s0 >= s1 || a == 0 {
        return;
    }
    if a >= 32768 {
        return fill(dst, s0, s1, ycbcr);
    }
    // A row of the colour, mixed in like a source.
    let mut tmp = vec![0u8; 4 * (s1.div_ceil(3) + 1)];
    fill(&mut tmp, s0, s1, ycbcr);
    mix(dst, &tmp, s0, s1, a);
}

/// One sample of a mix: `d + (s − d) · a / 32768`, rounded.
#[inline(always)]
pub fn lerp(d: u16, s: u16, a: u32) -> u16 {
    let diff = s as i32 - d as i32;
    (d as i32 + ((diff * a as i32 + 16384) >> 15)) as u16
}

// --- whole-word kernels -------------------------------------------------

/// Unpack whole words: 3 samples per word.
pub fn unpack_words(src: &[u8], out: &mut [u16]) {
    #[allow(unused_assignments, unused_mut)]
    let mut done = 0;
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: the CPU has the features (checked at run time).
        done = match level() {
            2 => unsafe { simd10::unpack_words_avx2(src, out) },
            1 => unsafe { simd10::unpack_words_ssse3(src, out) },
            _ => 0,
        };
    }
    unpack_words_scalar(&src[4 * done..], &mut out[3 * done..])
}

pub fn unpack_words_scalar(src: &[u8], out: &mut [u16]) {
    for (w, o) in src.chunks_exact(4).zip(out.chunks_exact_mut(3)) {
        let v = u32::from_le_bytes([w[0], w[1], w[2], w[3]]);
        o[0] = (v & MASK) as u16;
        o[1] = ((v >> 10) & MASK) as u16;
        o[2] = ((v >> 20) & MASK) as u16;
    }
}

/// Pack whole words: 3 samples per word (samples are masked to 10 bits).
pub fn pack_words(inp: &[u16], dst: &mut [u8]) {
    #[allow(unused_assignments, unused_mut)]
    let mut done = 0;
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: the CPU has the features (checked at run time).
        done = match level() {
            2 => unsafe { simd10::pack_words_avx2(inp, dst) },
            1 => unsafe { simd10::pack_words_ssse3(inp, dst) },
            _ => 0,
        };
    }
    pack_words_scalar(&inp[3 * done..], &mut dst[4 * done..])
}

pub fn pack_words_scalar(inp: &[u16], dst: &mut [u8]) {
    for (i, w) in inp.chunks_exact(3).zip(dst.chunks_exact_mut(4)) {
        let v = (i[0] as u32 & MASK) | ((i[1] as u32 & MASK) << 10) | ((i[2] as u32 & MASK) << 20);
        w.copy_from_slice(&v.to_le_bytes());
    }
}

/// Mix whole words with one level `a` (`0..32768`).
pub fn mix_words(dst: &mut [u8], src: &[u8], a: u32) {
    #[allow(unused_assignments, unused_mut)]
    let mut done = 0;
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: the CPU has the features (checked at run time).
        done = match level() {
            2 => unsafe { simd10::mix_words_avx2(dst, src, a) },
            1 => unsafe { simd10::mix_words_ssse3(dst, src, a) },
            _ => 0,
        };
    }
    mix_words_scalar(&mut dst[4 * done..], &src[4 * done..], a)
}

pub fn mix_words_scalar(dst: &mut [u8], src: &[u8], a: u32) {
    for (d, s) in dst.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
        let dv = u32::from_le_bytes([d[0], d[1], d[2], d[3]]);
        let sv = u32::from_le_bytes([s[0], s[1], s[2], s[3]]);
        let mut out = 0u32;
        for f in 0..3 {
            out |= (lerp(field(dv, f), field(sv, f), a) as u32) << (10 * f);
        }
        d.copy_from_slice(&out.to_le_bytes());
    }
}

/// Unpack samples `s0..s0 + out.len()` of a v210a key row: one sample per
/// pixel, 3 per word.
pub fn unpack_key(row: &[u8], x0: usize, out: &mut [u16]) {
    let x1 = x0 + out.len();
    let mut x = x0;
    while x < x1 && x % 3 != 0 {
        out[x - x0] = sample(row, x);
        x += 1;
    }
    let whole = (x1 - x) / 3;
    if whole > 0 {
        unpack_words(&row[4 * (x / 3)..4 * (x / 3 + whole)], &mut out[x - x0..x - x0 + 3 * whole]);
        x += 3 * whole;
    }
    while x < x1 {
        out[x - x0] = sample(row, x);
        x += 1;
    }
}

/// Unpack pixels `p0..p0 + y.len()` (`p0` and the count even) of a v210 row
/// straight into planes: Y (`n`), Cb and Cr (`n/2`). Whole 6-pixel blocks
/// go through SSSE3 shuffles (no UYVY row in between); `y`, `cb`, `cr` need
/// 2, 1 and 1 samples of slack after their ends (the stores overlap).
pub fn unpack_planar(row: &[u8], p0: usize, y: &mut [u16], cb: &mut [u16], cr: &mut [u16]) {
    let n = y.len().min(2 * cb.len()).min(2 * cr.len()) & !1;
    let n = n.min(y.len().saturating_sub(2));
    let p1 = p0 + n;
    let mut p = p0;
    let pair = |row: &[u8], p: usize, y: &mut [u16], cb: &mut [u16], cr: &mut [u16]| {
        let s = 2 * p;
        let o = p - p0;
        cb[o / 2] = sample(row, s);
        y[o] = sample(row, s + 1);
        cr[o / 2] = sample(row, s + 2);
        y[o + 1] = sample(row, s + 3);
    };
    while p < p1 && p % 6 != 0 {
        pair(row, p, y, cb, cr);
        p += 2;
    }
    #[cfg(target_arch = "x86_64")]
    if level() >= 1 {
        let blocks = ((p1 - p) / 6).min((row.len() / 16).saturating_sub(p / 6));
        if blocks > 0 {
            let o = p - p0;
            // SAFETY: SSSE3 checked; reads `blocks` whole blocks of the row,
            // writes 8/4/4 samples at each step (slack given by the caller).
            unsafe {
                super::simd10::unpack_planar_ssse3(
                    &row[16 * (p / 6)..16 * (p / 6 + blocks)],
                    &mut y[o..],
                    &mut cb[o / 2..],
                    &mut cr[o / 2..],
                    blocks,
                )
            };
            p += 6 * blocks;
        }
    }
    while p < p1 {
        pair(row, p, y, cb, cr);
        p += 2;
    }
}

/// The largest key sample of pixels `x0..x1` of a v210a key row.
pub fn max_key(row: &[u8], x0: usize, x1: usize) -> u16 {
    let mut m = 0u16;
    let mut x = x0;
    while x < x1 && x % 3 != 0 {
        m = m.max(sample(row, x));
        x += 1;
    }
    let (w0, w1) = (x / 3, x1 / 3);
    if w1 > w0 {
        let mut mw = [0u32; 3];
        for w in row[4 * w0..4 * w1].chunks_exact(4) {
            let v = u32::from_le_bytes([w[0], w[1], w[2], w[3]]);
            mw[0] = mw[0].max(v & MASK);
            mw[1] = mw[1].max((v >> 10) & MASK);
            mw[2] = mw[2].max((v >> 20) & MASK);
        }
        m = m.max(mw[0].max(mw[1]).max(mw[2]) as u16);
        x = 3 * w1;
    }
    while x < x1 {
        m = m.max(sample(row, x));
        x += 1;
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row_of(samples: &[u16]) -> Vec<u8> {
        let mut row = vec![0u8; 4 * samples.len().div_ceil(3) + 64];
        for (s, &v) in samples.iter().enumerate() {
            set_sample(&mut row, s, v);
        }
        row
    }

    fn samples(n: usize, seed: u32) -> Vec<u16> {
        let mut x = seed.wrapping_mul(2654435761).wrapping_add(1);
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                (x % 1024) as u16
            })
            .collect()
    }

    #[test]
    fn geometry() {
        assert_eq!(line_bytes(1920), 5120);
        assert_eq!(line_bytes(1280), 3456);
        assert_eq!(line_bytes(3840), 10240);
        assert_eq!(key_line_bytes(1920), 2560);
        assert_eq!(key_line_bytes(1280), 1708);
        assert_eq!(frame_bytes(1920, 1080, false), 5_529_600);
    }

    #[test]
    fn unpack_pack_any_range() {
        let n = 200;
        let src = samples(n, 1);
        let row = row_of(&src);
        for s0 in 0..12 {
            for len in 0..60 {
                let mut out = vec![0u16; len];
                unpack(&row, s0, &mut out);
                assert_eq!(out, src[s0..s0 + len], "unpack {s0} {len}");
                let mut dst = row_of(&samples(n, 2));
                let before: Vec<u16> = (0..n).map(|s| sample(&dst, s)).collect();
                pack(&mut dst, s0, &out);
                for s in 0..n {
                    let want = if s >= s0 && s < s0 + len { src[s] } else { before[s] };
                    assert_eq!(sample(&dst, s), want, "pack {s0} {len} at {s}");
                }
            }
        }
    }

    #[test]
    fn copy_fill_mix_keep_samples_outside() {
        let n = 300;
        let a = samples(n, 3);
        let b = samples(n, 4);
        for s0 in 0..13 {
            for s1 in [s0, s0 + 1, s0 + 2, s0 + 5, s0 + 37, s0 + 200] {
                let mut dst = row_of(&a);
                copy(&mut dst, &row_of(&b), s0, s1);
                for s in 0..n {
                    let want = if (s0..s1).contains(&s) { b[s] } else { a[s] };
                    assert_eq!(sample(&dst, s), want);
                }
                let mut dst = row_of(&a);
                fill(&mut dst, s0, s1, [940, 512, 600]);
                for s in 0..n {
                    let want = if (s0..s1).contains(&s) {
                        [512, 940, 600, 940][s % 4]
                    } else {
                        a[s]
                    };
                    assert_eq!(sample(&dst, s), want, "fill {s0} {s1} {s}");
                }
                for level in [1u32, 8192, 16384, 32767] {
                    let mut dst = row_of(&a);
                    mix(&mut dst, &row_of(&b), s0, s1, level);
                    for s in 0..n {
                        let want = if (s0..s1).contains(&s) {
                            let f = a[s] as f64 + (b[s] as f64 - a[s] as f64) * level as f64 / 32768.0;
                            f.round() as i32
                        } else {
                            a[s] as i32
                        };
                        assert!(
                            (sample(&dst, s) as i32 - want).abs() <= 1,
                            "mix {level} {s}: {} vs {want}",
                            sample(&dst, s)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn simd_matches_scalar() {
        let n = 3 * 1000;
        let a = row_of(&samples(n, 5));
        let b = row_of(&samples(n, 6));
        let words = n / 3;
        let mut x = vec![0u16; n];
        let mut y = vec![0u16; n];
        unpack_words(&a[..4 * words], &mut x);
        unpack_words_scalar(&a[..4 * words], &mut y);
        assert_eq!(x, y);
        let mut p = vec![0u8; 4 * words];
        let mut q = vec![0u8; 4 * words];
        pack_words(&x, &mut p);
        pack_words_scalar(&x, &mut q);
        assert_eq!(p, q);
        for level in [0u32, 3, 12345, 32767] {
            let mut d1 = a[..4 * words].to_vec();
            let mut d2 = d1.clone();
            mix_words(&mut d1, &b[..4 * words], level);
            mix_words_scalar(&mut d2, &b[..4 * words], level);
            assert_eq!(d1, d2, "level {level}");
        }
    }

    #[test]
    fn planar_unpack_any_start() {
        let n = 600;
        let src = samples(n, 21);
        let row = row_of(&src);
        for p0 in [0usize, 2, 4, 6, 8, 10, 12] {
            for len in [0usize, 2, 4, 6, 10, 12, 14, 36, 100] {
                let (mut y, mut cb, mut cr) = (vec![0; len + 2], vec![0; len / 2 + 1], vec![0; len / 2 + 1]);
                // As callers pass them: exactly len + slack.
                unpack_planar(&row, p0, &mut y, &mut cb, &mut cr);
                for q in 0..len / 2 {
                    let s = 2 * p0 + 4 * q;
                    assert_eq!((cb[q], y[2 * q], cr[q], y[2 * q + 1]), (src[s], src[s + 1], src[s + 2], src[s + 3]), "{p0} {len} {q}");
                }
            }
        }
    }

    #[test]
    fn key_max() {
        let src = samples(100, 11);
        let row = row_of(&src);
        for x0 in 0..7 {
            for x1 in x0..60 {
                let want = src[x0..x1].iter().copied().max().unwrap_or(0);
                assert_eq!(max_key(&row, x0, x1), want, "{x0} {x1}");
            }
        }
    }

    #[test]
    fn key_rows() {
        let src = samples(100, 9);
        let row = row_of(&src);
        for x0 in 0..7 {
            for len in 0..40 {
                let mut out = vec![0u16; len];
                unpack_key(&row, x0, &mut out);
                assert_eq!(out, src[x0..x0 + len]);
            }
        }
    }
}

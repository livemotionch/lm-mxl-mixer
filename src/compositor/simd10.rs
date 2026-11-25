//! Hand-written SSSE3/SSE4.1 and AVX2 kernels for v210 words and 16-bit
//! samples. Each has a plain-Rust twin in [`super::v210`] /
//! [`super::kernels10`] that the tests compare it with.
//!
//! - unpack: 4 words → 12 samples per 128-bit lane: the three 10-bit fields
//!   are masked and shifted into 32-bit lanes, paired into 16-bit lanes and
//!   put in order with byte shuffles.
//! - pack: the reverse.
//! - mix and blend: `d + ((s − d) · a + 2^14) >> 15` is exactly `pmulhrsw`
//!   for |s − d| ≤ 1023 and a ≤ 32767 (an opaque level 32768 is used as
//!   32767, which gives the same result for 10-bit differences).
//! - mix of words in place: fields 0 and 2 of a word in the two 16-bit
//!   halves of its lane, field 1 in another register; no unpacking.

#![cfg(target_arch = "x86_64")]
#![allow(clippy::missing_safety_doc)]

use std::arch::x86_64::*;

const M: i32 = 0x3FF;

// Byte shuffles for unpack (per 128-bit lane). `lo`: fields 0 and 1 of
// words a b c d as u16 [a0 a1 b0 b1 c0 c1 d0 d1]; `hi`: field 2 [a2 _ b2 _
// c2 _ d2 _]. Out: [a0 a1 a2 b0 b1 b2 c0 c1] and [c2 d0 d1 d2].
const Z: i8 = -128;
#[rustfmt::skip]
const U_LO0: [i8; 16] = [0, 1, 2, 3, Z, Z, 4, 5, 6, 7, Z, Z, 8, 9, 10, 11];
#[rustfmt::skip]
const U_HI0: [i8; 16] = [Z, Z, Z, Z, 0, 1, Z, Z, Z, Z, 4, 5, Z, Z, Z, Z];
#[rustfmt::skip]
const U_LO1: [i8; 16] = [Z, Z, 12, 13, 14, 15, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z];
#[rustfmt::skip]
const U_HI1: [i8; 16] = [8, 9, Z, Z, Z, Z, 12, 13, Z, Z, Z, Z, Z, Z, Z, Z];

// Pack: from in0 = [a0 a1 a2 b0 b1 b2 c0 c1] and in1 = [c2 d0 d1 d2] to the
// fields in 32-bit lanes.
#[rustfmt::skip]
const P_F0_0: [i8; 16] = [0, 1, Z, Z, 6, 7, Z, Z, 12, 13, Z, Z, Z, Z, Z, Z];
#[rustfmt::skip]
const P_F0_1: [i8; 16] = [Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, 2, 3, Z, Z];
#[rustfmt::skip]
const P_F1_0: [i8; 16] = [2, 3, Z, Z, 8, 9, Z, Z, 14, 15, Z, Z, Z, Z, Z, Z];
#[rustfmt::skip]
const P_F1_1: [i8; 16] = [Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z, 4, 5, Z, Z];
#[rustfmt::skip]
const P_F2_0: [i8; 16] = [4, 5, Z, Z, 10, 11, Z, Z, Z, Z, Z, Z, Z, Z, Z, Z];
#[rustfmt::skip]
const P_F2_1: [i8; 16] = [Z, Z, Z, Z, Z, Z, Z, Z, 0, 1, Z, Z, 6, 7, Z, Z];

#[inline(always)]
unsafe fn ld(m: &[i8; 16]) -> __m128i {
    unsafe { _mm_loadu_si128(m.as_ptr() as *const __m128i) }
}

#[inline(always)]
unsafe fn ld2(m: &[i8; 16]) -> __m256i {
    unsafe { _mm256_broadcastsi128_si256(ld(m)) }
}

/// Unpack whole words (3 samples each). Returns the words done.
#[target_feature(enable = "ssse3,sse4.1")]
pub unsafe fn unpack_words_ssse3(src: &[u8], out: &mut [u16]) -> usize {
    let words = (src.len() / 4).min(out.len() / 3);
    // The second store writes 8 bytes: keep a whole group in reserve.
    let rounds = words / 4;
    unsafe {
        let m = _mm_set1_epi32(M);
        let (lo0, hi0, lo1, hi1) = (ld(&U_LO0), ld(&U_HI0), ld(&U_LO1), ld(&U_HI1));
        for r in 0..rounds {
            let v = _mm_loadu_si128(src.as_ptr().add(16 * r) as *const __m128i);
            let f0 = _mm_and_si128(v, m);
            let f1 = _mm_and_si128(_mm_srli_epi32(v, 10), m);
            let f2 = _mm_and_si128(_mm_srli_epi32(v, 20), m);
            let lo = _mm_or_si128(f0, _mm_slli_epi32(f1, 16));
            let o0 = _mm_or_si128(_mm_shuffle_epi8(lo, lo0), _mm_shuffle_epi8(f2, hi0));
            let o1 = _mm_or_si128(_mm_shuffle_epi8(lo, lo1), _mm_shuffle_epi8(f2, hi1));
            let o = out.as_mut_ptr().add(12 * r);
            _mm_storeu_si128(o as *mut __m128i, o0);
            _mm_storel_epi64(o.add(8) as *mut __m128i, o1);
        }
    }
    4 * rounds
}

#[target_feature(enable = "avx2")]
pub unsafe fn unpack_words_avx2(src: &[u8], out: &mut [u16]) -> usize {
    let words = (src.len() / 4).min(out.len() / 3);
    let rounds = words / 8;
    unsafe {
        let m = _mm256_set1_epi32(M);
        let (lo0, hi0, lo1, hi1) = (ld2(&U_LO0), ld2(&U_HI0), ld2(&U_LO1), ld2(&U_HI1));
        for r in 0..rounds {
            let v = _mm256_loadu_si256(src.as_ptr().add(32 * r) as *const __m256i);
            let f0 = _mm256_and_si256(v, m);
            let f1 = _mm256_and_si256(_mm256_srli_epi32(v, 10), m);
            let f2 = _mm256_and_si256(_mm256_srli_epi32(v, 20), m);
            let lo = _mm256_or_si256(f0, _mm256_slli_epi32(f1, 16));
            let o0 = _mm256_or_si256(_mm256_shuffle_epi8(lo, lo0), _mm256_shuffle_epi8(f2, hi0));
            let o1 = _mm256_or_si256(_mm256_shuffle_epi8(lo, lo1), _mm256_shuffle_epi8(f2, hi1));
            let o = out.as_mut_ptr().add(24 * r);
            _mm_storeu_si128(o as *mut __m128i, _mm256_castsi256_si128(o0));
            _mm_storel_epi64(o.add(8) as *mut __m128i, _mm256_castsi256_si128(o1));
            _mm_storeu_si128(o.add(12) as *mut __m128i, _mm256_extracti128_si256(o0, 1));
            _mm_storel_epi64(o.add(20) as *mut __m128i, _mm256_extracti128_si256(o1, 1));
        }
    }
    8 * rounds
}

/// Pack whole words. Returns the words done.
#[target_feature(enable = "ssse3,sse4.1")]
pub unsafe fn pack_words_ssse3(inp: &[u16], dst: &mut [u8]) -> usize {
    let words = (dst.len() / 4).min(inp.len() / 3);
    let rounds = words / 4;
    unsafe {
        let m = _mm_set1_epi32(M);
        let (a0, a1, b0, b1, c0, c1) =
            (ld(&P_F0_0), ld(&P_F0_1), ld(&P_F1_0), ld(&P_F1_1), ld(&P_F2_0), ld(&P_F2_1));
        for r in 0..rounds {
            let p = inp.as_ptr().add(12 * r);
            let i0 = _mm_loadu_si128(p as *const __m128i);
            let i1 = _mm_loadl_epi64(p.add(8) as *const __m128i);
            let f0 = _mm_or_si128(_mm_shuffle_epi8(i0, a0), _mm_shuffle_epi8(i1, a1));
            let f1 = _mm_or_si128(_mm_shuffle_epi8(i0, b0), _mm_shuffle_epi8(i1, b1));
            let f2 = _mm_or_si128(_mm_shuffle_epi8(i0, c0), _mm_shuffle_epi8(i1, c1));
            let w = _mm_or_si128(
                _mm_or_si128(_mm_and_si128(f0, m), _mm_slli_epi32(_mm_and_si128(f1, m), 10)),
                _mm_slli_epi32(_mm_and_si128(f2, m), 20),
            );
            _mm_storeu_si128(dst.as_mut_ptr().add(16 * r) as *mut __m128i, w);
        }
    }
    4 * rounds
}

#[target_feature(enable = "avx2")]
pub unsafe fn pack_words_avx2(inp: &[u16], dst: &mut [u8]) -> usize {
    let words = (dst.len() / 4).min(inp.len() / 3);
    let rounds = words / 8;
    unsafe {
        let m = _mm256_set1_epi32(M);
        let (a0, a1, b0, b1, c0, c1) =
            (ld2(&P_F0_0), ld2(&P_F0_1), ld2(&P_F1_0), ld2(&P_F1_1), ld2(&P_F2_0), ld2(&P_F2_1));
        for r in 0..rounds {
            let p = inp.as_ptr().add(24 * r);
            let i0 = _mm256_set_m128i(
                _mm_loadu_si128(p.add(12) as *const __m128i),
                _mm_loadu_si128(p as *const __m128i),
            );
            let i1 = _mm256_set_m128i(
                _mm_loadl_epi64(p.add(20) as *const __m128i),
                _mm_loadl_epi64(p.add(8) as *const __m128i),
            );
            let f0 = _mm256_or_si256(_mm256_shuffle_epi8(i0, a0), _mm256_shuffle_epi8(i1, a1));
            let f1 = _mm256_or_si256(_mm256_shuffle_epi8(i0, b0), _mm256_shuffle_epi8(i1, b1));
            let f2 = _mm256_or_si256(_mm256_shuffle_epi8(i0, c0), _mm256_shuffle_epi8(i1, c1));
            let w = _mm256_or_si256(
                _mm256_or_si256(
                    _mm256_and_si256(f0, m),
                    _mm256_slli_epi32(_mm256_and_si256(f1, m), 10),
                ),
                _mm256_slli_epi32(_mm256_and_si256(f2, m), 20),
            );
            _mm256_storeu_si256(dst.as_mut_ptr().add(32 * r) as *mut __m256i, w);
        }
    }
    8 * rounds
}

/// Mix whole words in place with one level (`a` < 32768). Returns the
/// words done.
#[target_feature(enable = "ssse3,sse4.1")]
pub unsafe fn mix_words_ssse3(dst: &mut [u8], src: &[u8], a: u32) -> usize {
    let words = (dst.len() / 4).min(src.len() / 4);
    let rounds = words / 4;
    unsafe {
        let m = _mm_set1_epi32(M);
        let mhi = _mm_set1_epi32(M << 16);
        let av = _mm_set1_epi16(a.min(32767) as i16);
        for r in 0..rounds {
            let dp = dst.as_mut_ptr().add(16 * r) as *mut __m128i;
            let d = _mm_loadu_si128(dp);
            let s = _mm_loadu_si128(src.as_ptr().add(16 * r) as *const __m128i);
            // Fields 0 and 2 in the 16-bit halves; field 1 alone.
            let d02 = _mm_or_si128(_mm_and_si128(d, m), _mm_and_si128(_mm_srli_epi32(d, 4), mhi));
            let s02 = _mm_or_si128(_mm_and_si128(s, m), _mm_and_si128(_mm_srli_epi32(s, 4), mhi));
            let d1 = _mm_and_si128(_mm_srli_epi32(d, 10), m);
            let s1 = _mm_and_si128(_mm_srli_epi32(s, 10), m);
            let r02 = _mm_add_epi16(d02, _mm_mulhrs_epi16(_mm_sub_epi16(s02, d02), av));
            let r1 = _mm_add_epi16(d1, _mm_mulhrs_epi16(_mm_sub_epi16(s1, d1), av));
            let lo = _mm_and_si128(r02, m);
            let hi = _mm_slli_epi32(_mm_srli_epi32(r02, 16), 20);
            let w = _mm_or_si128(_mm_or_si128(lo, hi), _mm_slli_epi32(_mm_and_si128(r1, m), 10));
            _mm_storeu_si128(dp, w);
        }
    }
    4 * rounds
}

#[target_feature(enable = "avx2")]
pub unsafe fn mix_words_avx2(dst: &mut [u8], src: &[u8], a: u32) -> usize {
    let words = (dst.len() / 4).min(src.len() / 4);
    let rounds = words / 8;
    unsafe {
        let m = _mm256_set1_epi32(M);
        let mhi = _mm256_set1_epi32(M << 16);
        let av = _mm256_set1_epi16(a.min(32767) as i16);
        for r in 0..rounds {
            let dp = dst.as_mut_ptr().add(32 * r) as *mut __m256i;
            let d = _mm256_loadu_si256(dp);
            let s = _mm256_loadu_si256(src.as_ptr().add(32 * r) as *const __m256i);
            let d02 = _mm256_or_si256(
                _mm256_and_si256(d, m),
                _mm256_and_si256(_mm256_srli_epi32(d, 4), mhi),
            );
            let s02 = _mm256_or_si256(
                _mm256_and_si256(s, m),
                _mm256_and_si256(_mm256_srli_epi32(s, 4), mhi),
            );
            let d1 = _mm256_and_si256(_mm256_srli_epi32(d, 10), m);
            let s1 = _mm256_and_si256(_mm256_srli_epi32(s, 10), m);
            let r02 = _mm256_add_epi16(d02, _mm256_mulhrs_epi16(_mm256_sub_epi16(s02, d02), av));
            let r1 = _mm256_add_epi16(d1, _mm256_mulhrs_epi16(_mm256_sub_epi16(s1, d1), av));
            let lo = _mm256_and_si256(r02, m);
            let hi = _mm256_slli_epi32(_mm256_srli_epi32(r02, 16), 20);
            let w = _mm256_or_si256(
                _mm256_or_si256(lo, hi),
                _mm256_slli_epi32(_mm256_and_si256(r1, m), 10),
            );
            _mm256_storeu_si256(dp, w);
        }
    }
    8 * rounds
}

/// `dst += (src − dst) · alpha / 32768` per sample. Returns the samples done.
#[target_feature(enable = "ssse3,sse4.1")]
pub unsafe fn blend16_ssse3(dst: &mut [u16], src: &[u16], alpha: &[u16]) -> usize {
    let n = dst.len().min(src.len()).min(alpha.len());
    let rounds = n / 8;
    unsafe {
        let cap = _mm_set1_epi16(32767);
        for r in 0..rounds {
            let dp = dst.as_mut_ptr().add(8 * r) as *mut __m128i;
            let d = _mm_loadu_si128(dp);
            let s = _mm_loadu_si128(src.as_ptr().add(8 * r) as *const __m128i);
            let a = _mm_min_epu16(_mm_loadu_si128(alpha.as_ptr().add(8 * r) as *const __m128i), cap);
            _mm_storeu_si128(dp, _mm_add_epi16(d, _mm_mulhrs_epi16(_mm_sub_epi16(s, d), a)));
        }
    }
    8 * rounds
}

#[target_feature(enable = "avx2")]
pub unsafe fn blend16_avx2(dst: &mut [u16], src: &[u16], alpha: &[u16]) -> usize {
    let n = dst.len().min(src.len()).min(alpha.len());
    let rounds = n / 16;
    unsafe {
        let cap = _mm256_set1_epi16(32767);
        for r in 0..rounds {
            let dp = dst.as_mut_ptr().add(16 * r) as *mut __m256i;
            let d = _mm256_loadu_si256(dp);
            let s = _mm256_loadu_si256(src.as_ptr().add(16 * r) as *const __m256i);
            let a = _mm256_min_epu16(
                _mm256_loadu_si256(alpha.as_ptr().add(16 * r) as *const __m256i),
                cap,
            );
            _mm256_storeu_si256(dp, _mm256_add_epi16(d, _mm256_mulhrs_epi16(_mm256_sub_epi16(s, d), a)));
        }
    }
    16 * rounds
}

/// `dst += (src − dst) · a / 32768` per sample, one level.
#[target_feature(enable = "avx2")]
pub unsafe fn mix16_avx2(dst: &mut [u16], src: &[u16], a: u32) -> usize {
    let n = dst.len().min(src.len());
    let rounds = n / 16;
    unsafe {
        let av = _mm256_set1_epi16(a.min(32767) as i16);
        for r in 0..rounds {
            let dp = dst.as_mut_ptr().add(16 * r) as *mut __m256i;
            let d = _mm256_loadu_si256(dp);
            let s = _mm256_loadu_si256(src.as_ptr().add(16 * r) as *const __m256i);
            _mm256_storeu_si256(dp, _mm256_add_epi16(d, _mm256_mulhrs_epi16(_mm256_sub_epi16(s, d), av)));
        }
    }
    16 * rounds
}

#[target_feature(enable = "ssse3,sse4.1")]
pub unsafe fn mix16_ssse3(dst: &mut [u16], src: &[u16], a: u32) -> usize {
    let n = dst.len().min(src.len());
    let rounds = n / 8;
    unsafe {
        let av = _mm_set1_epi16(a.min(32767) as i16);
        for r in 0..rounds {
            let dp = dst.as_mut_ptr().add(8 * r) as *mut __m128i;
            let d = _mm_loadu_si128(dp);
            let s = _mm_loadu_si128(src.as_ptr().add(8 * r) as *const __m128i);
            _mm_storeu_si128(dp, _mm_add_epi16(d, _mm_mulhrs_epi16(_mm_sub_epi16(s, d), av)));
        }
    }
    8 * rounds
}

/// One 6-pixel block per step: the three fields of 4 words in 32-bit lanes,
/// `f0 | f1 << 16` and `f2`, shuffled straight into Y (6) and Cb, Cr (3 each).
#[target_feature(enable = "ssse3,sse4.1")]
pub unsafe fn unpack_planar_ssse3(src: &[u8], y: &mut [u16], cb: &mut [u16], cr: &mut [u16], blocks: usize) {
    // lo (u16): [cb0 y0 y1 cb2 cr2 y3 y4 cr4]; hi: [cr0 _ y2 _ cb4 _ y5 _]
    #[rustfmt::skip]
    const Y_LO: [i8; 16] = [2, 3, 4, 5, Z, Z, 10, 11, 12, 13, Z, Z, Z, Z, Z, Z];
    #[rustfmt::skip]
    const Y_HI: [i8; 16] = [Z, Z, Z, Z, 4, 5, Z, Z, Z, Z, 12, 13, Z, Z, Z, Z];
    // [cb0 cb2 cb4 _ cr0 cr2 cr4 _]
    #[rustfmt::skip]
    const C_LO: [i8; 16] = [0, 1, 6, 7, Z, Z, Z, Z, Z, Z, 8, 9, 14, 15, Z, Z];
    #[rustfmt::skip]
    const C_HI: [i8; 16] = [Z, Z, Z, Z, 8, 9, Z, Z, 0, 1, Z, Z, Z, Z, Z, Z];
    unsafe {
        let m = _mm_set1_epi32(M);
        let (ylo, yhi, clo, chi) = (ld(&Y_LO), ld(&Y_HI), ld(&C_LO), ld(&C_HI));
        let (yp, bp, rp) = (y.as_mut_ptr(), cb.as_mut_ptr(), cr.as_mut_ptr());
        for b in 0..blocks {
            let v = _mm_loadu_si128(src.as_ptr().add(16 * b) as *const __m128i);
            let f0 = _mm_and_si128(v, m);
            let f1 = _mm_and_si128(_mm_srli_epi32(v, 10), m);
            let f2 = _mm_and_si128(_mm_srli_epi32(v, 20), m);
            let lo = _mm_or_si128(f0, _mm_slli_epi32(f1, 16));
            let yv = _mm_or_si128(_mm_shuffle_epi8(lo, ylo), _mm_shuffle_epi8(f2, yhi));
            let cv = _mm_or_si128(_mm_shuffle_epi8(lo, clo), _mm_shuffle_epi8(f2, chi));
            _mm_storeu_si128(yp.add(6 * b) as *mut __m128i, yv);
            _mm_storel_epi64(bp.add(3 * b) as *mut __m128i, cv);
            _mm_storel_epi64(rp.add(3 * b) as *mut __m128i, _mm_unpackhi_epi64(cv, cv));
        }
    }
}

/// v210 words straight to 8-bit samples (`min((v + 2) >> 2, 255)`): per
/// 4 words, the fields in 16-bit lanes (as the unpack), rounded and shifted
/// there, saturated to bytes by `packuswb`, ordered by one byte shuffle: 12
/// bytes, stored 16 at a time (each store stays inside `out`; the next one
/// overwrites its last 4 bytes). Returns the words done.
#[target_feature(enable = "ssse3,sse4.1")]
pub unsafe fn unpack_words_u8_ssse3(src: &[u8], out: &mut [u8]) -> usize {
    let words = (src.len() / 4).min(out.len() / 3);
    let steps = if out.len() >= 16 { ((out.len() - 16) / 12 + 1).min(words / 4) } else { 0 };
    unsafe {
        let m = _mm_set1_epi32(M);
        let two = _mm_set1_epi16(2);
        let order = ld(&U8_ORDER);
        for r in 0..steps {
            let v = _mm_loadu_si128(src.as_ptr().add(16 * r) as *const __m128i);
            let lo = _mm_or_si128(_mm_and_si128(v, m), _mm_slli_epi32(_mm_and_si128(_mm_srli_epi32(v, 10), m), 16));
            let hi = _mm_and_si128(_mm_srli_epi32(v, 20), m);
            let lo = _mm_srli_epi16(_mm_add_epi16(lo, two), 2);
            let hi = _mm_srli_epi16(_mm_add_epi16(hi, two), 2);
            let b = _mm_shuffle_epi8(_mm_packus_epi16(lo, hi), order);
            _mm_storeu_si128(out.as_mut_ptr().add(12 * r) as *mut __m128i, b);
        }
    }
    4 * steps
}

/// `packuswb` of `[a0 a1 b0 b1 c0 c1 d0 d1]` and `[a2 _ b2 _ c2 _ d2 _]`
/// to `a0 a1 a2 b0 b1 b2 …`.
#[rustfmt::skip]
const U8_ORDER: [i8; 16] = [0, 1, 8, 2, 3, 10, 4, 5, 12, 6, 7, 14, Z, Z, Z, Z];

/// [`unpack_words_u8_ssse3`] with 8 words per round (two 12-byte stores).
#[target_feature(enable = "avx2")]
pub unsafe fn unpack_words_u8_avx2(src: &[u8], out: &mut [u8]) -> usize {
    let words = (src.len() / 4).min(out.len() / 3);
    // Round r writes 16 bytes at 24 r and at 24 r + 12.
    let rounds = if out.len() >= 28 { ((out.len() - 28) / 24 + 1).min(words / 8) } else { 0 };
    unsafe {
        let m = _mm256_set1_epi32(M);
        let two = _mm256_set1_epi16(2);
        let order = ld2(&U8_ORDER);
        for r in 0..rounds {
            let v = _mm256_loadu_si256(src.as_ptr().add(32 * r) as *const __m256i);
            let lo = _mm256_or_si256(
                _mm256_and_si256(v, m),
                _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(v, 10), m), 16),
            );
            let hi = _mm256_and_si256(_mm256_srli_epi32(v, 20), m);
            let lo = _mm256_srli_epi16(_mm256_add_epi16(lo, two), 2);
            let hi = _mm256_srli_epi16(_mm256_add_epi16(hi, two), 2);
            let b = _mm256_shuffle_epi8(_mm256_packus_epi16(lo, hi), order);
            let o = out.as_mut_ptr().add(24 * r);
            _mm_storeu_si128(o as *mut __m128i, _mm256_castsi256_si128(b));
            _mm_storeu_si128(o.add(12) as *mut __m128i, _mm256_extracti128_si256(b, 1));
        }
    }
    8 * rounds
}

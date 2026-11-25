//! Scaling at 10 bits: the steps of a scaled layer's row, on planar samples.
//!
//! The first version (render10's `filter_span`) summed interleaved UYVY
//! rows and filtered horizontally per output with the table's weights in a
//! plain loop. perf on the test VM showed that the inner
//! loop compiles to scalar code (indexed loads of the taps), and it was 55%
//! of a fractional-size PiP. Now:
//!
//! 1. **Rows to planes**: each source row's span is unpacked (SIMD v210
//!    unpack) and split into Y, Cb, Cr rows ([`deinterleave`]).
//! 2. **Vertical, one row after the other** into rows of `u32` sums that stay
//!    in L1 (`acc_rows`, vectorised).
//! 3. **Normalised** to 16-bit Q5 (sample × 32): [`normalize`].
//! 4. **Horizontal with SIMD** ([`hfilter`]): 8 outputs at a time, two taps
//!    per step: a 32-bit gather fetches the two neighbouring Q5 samples of
//!    each output, `pmaddwd` multiplies them by the pair of weights and adds
//!    (exact: Q5 ≤ 32 736 and weights ≤ 1024 fit in i16), the sum is scaled
//!    in floating point. Boxes are tables with weights of 1.
//! 5. **Interleaved** back to UYVY ([`interleave`]).
//!
//! Strong vertical downscales (7.5:1 or more) read only every n-th row
//! (`render::vertical_step`).

use super::kernels::{level, multiversion};
use super::scale::Table;

/// A horizontal table in the form [`hfilter`] wants.
#[derive(Debug, Clone)]
pub struct HTab {
    pub len: usize,
    /// Tap pairs per output
    pub pairs: usize,
    /// First source sample of each output (and 8 of slack)
    pub start: Vec<i32>,
    /// `pairs × len` weight pairs, pair-major: `(w[2p] | w[2p+1] << 16)`
    /// of output `j` at `p·len + j` (and 8 of slack per pair)
    pub weights: Vec<i32>,
    /// `1 / (total · 32)`: Q5 sums to samples
    pub inv: f32,
    /// Source samples read past the last start
    pub reach: usize,
}

impl HTab {
    pub fn new(t: &Table) -> HTab {
        let pairs = t.taps.div_ceil(2).max(1);
        let len = t.len;
        let stride = len + 8;
        let mut weights = vec![0i32; pairs * stride];
        for j in 0..len {
            let w = t.weights(j);
            for p in 0..pairs {
                let lo = w.get(2 * p).copied().unwrap_or(0) as i32;
                let hi = w.get(2 * p + 1).copied().unwrap_or(0) as i32;
                weights[p * stride + j] = lo | (hi << 16);
            }
        }
        let mut start: Vec<i32> = t.start.iter().map(|&s| s as i32).collect();
        start.extend(std::iter::repeat_n(*start.last().unwrap_or(&0), 8));
        HTab {
            len,
            pairs,
            start,
            weights,
            inv: 1.0 / (t.total as f32 * 32.0),
            reach: 2 * pairs + 1,
        }
    }

    fn stride(&self) -> usize {
        self.len + 8
    }
}

/// Split UYVY samples (`2·n` for `n` pixels, `n` even) into Y (`n`), Cb and
/// Cr (`n/2` each).
pub fn deinterleave(src: &[u16], y: &mut [u16], cb: &mut [u16], cr: &mut [u16]) {
    let n = (src.len() / 2).min(y.len()).min(2 * cb.len()).min(2 * cr.len()) & !1;
    #[allow(unused_assignments, unused_mut)]
    let mut done = 0;
    #[cfg(target_arch = "x86_64")]
    if level() >= 1 {
        // SAFETY: SSSE3 checked; reads src[..2 done], writes within the
        // first `done` / `done / 2` samples.
        done = unsafe { simd::deinterleave_ssse3(&src[..2 * n], y, cb, cr) };
    }
    for q in done / 2..n / 2 {
        cb[q] = src[4 * q];
        y[2 * q] = src[4 * q + 1];
        cr[q] = src[4 * q + 2];
        y[2 * q + 1] = src[4 * q + 3];
    }
}

/// UYVY samples from Y (`n`), Cb and Cr (`n/2`).
pub fn interleave(y: &[u16], cb: &[u16], cr: &[u16], out: &mut [u16]) {
    let n = (out.len() / 2).min(y.len()).min(2 * cb.len()).min(2 * cr.len()) & !1;
    #[allow(unused_assignments, unused_mut)]
    let mut done = 0;
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: SSE2 is baseline on x86_64.
        done = unsafe { simd::interleave_sse2(&y[..n], &cb[..n / 2], &cr[..n / 2], &mut out[..2 * n]) };
    }
    for q in done / 2..n / 2 {
        out[4 * q] = cb[q];
        out[4 * q + 1] = y[2 * q];
        out[4 * q + 2] = cr[q];
        out[4 * q + 3] = y[2 * q + 1];
    }
}

multiversion! {
    /// `acc = row · w` (the first row of a vertical sum).
    pub fn acc_first(acc: &mut [u32], row: &[u16], w: u32) {
        for (a, &v) in acc.iter_mut().zip(row) {
            *a = v as u32 * w;
        }
    }
}

multiversion! {
    /// `acc += row · w`
    pub fn acc_add(acc: &mut [u32], row: &[u16], w: u32) {
        for (a, &v) in acc.iter_mut().zip(row) {
            *a += v as u32 * w;
        }
    }
}

multiversion! {
    /// Vertical sums (total weight `total`) to Q5 samples: `round(a · 32 /
    /// total)`; exact in f32 (sums are below 2^20).
    pub fn normalize(out: &mut [u16], acc: &[u32], total: u32) {
        let scale = 32.0 / total.max(1) as f32;
        for (o, &a) in out.iter_mut().zip(acc) {
            // In range (0 ≤ x < 32 768): the unchecked conversion is one
            // `cvttps2dq`; Rust's saturating `as` kept this loop scalar.
            let x = (a as i32 as f32 * scale + 0.5).min(32767.0);
            // SAFETY: x is finite and within i32.
            *o = unsafe { x.to_int_unchecked::<i32>() } as u16;
        }
    }
}

/// Filter Q5 samples `q` (source sample `s` at `q[s − base]`) to outputs
/// `j0..j0 + out.len()` of `tab`. `q` must reach `tab.reach` samples past
/// the last start used (zeros).
pub fn hfilter(out: &mut [u16], q: &[u16], tab: &HTab, j0: usize, base: usize) {
    hfilter_div(out, q, tab, j0, base, tab.inv)
}

/// [`hfilter`] of samples that aren't Q5 (box sums): the sums are scaled by
/// `inv` instead.
pub fn hfilter_div(out: &mut [u16], q: &[u16], tab: &HTab, j0: usize, base: usize, inv: f32) {
    let n = out.len().min(tab.len.saturating_sub(j0));
    #[allow(unused_assignments, unused_mut)]
    let mut done = 0;
    #[cfg(target_arch = "x86_64")]
    if level() >= 2 {
        // SAFETY: AVX2 checked; the gathers read q at start − base + 2p + 1
        // at most, which the caller keeps inside `q`.
        done = unsafe { simd::hfilter_avx2(&mut out[..n], q, tab, j0, base as i32, inv) };
    }
    hfilter_scalar_div(&mut out[done..n], q, tab, j0 + done, base, inv);
}

/// The plain twin of the AVX2 kernel (same arithmetic).
pub fn hfilter_scalar(out: &mut [u16], q: &[u16], tab: &HTab, j0: usize, base: usize) {
    hfilter_scalar_div(out, q, tab, j0, base, tab.inv)
}

fn hfilter_scalar_div(out: &mut [u16], q: &[u16], tab: &HTab, j0: usize, base: usize, inv: f32) {
    let stride = tab.stride();
    for (o, out) in out.iter_mut().enumerate() {
        let j = j0 + o;
        let s = tab.start[j] as usize - base;
        let mut sum = 0i32;
        for p in 0..tab.pairs {
            let w = tab.weights[p * stride + j];
            let (w0, w1) = (w as i16 as i32, (w >> 16) as i16 as i32);
            sum += q[s + 2 * p] as i32 * w0 + q[s + 2 * p + 1] as i32 * w1;
        }
        *out = ((sum as f32 * inv + 0.5) as i32).clamp(0, 1023) as u16;
    }
}

/// Box sums of one plane: `out[j] = round(Σ sums[j·k .. (j+1)·k] / div)`,
/// `sums` vertical box sums (`u16`, each ≤ 32 767 / k). SSSE3 `phaddw`
/// folds for k = 1, 2, 4, 8 the rest plain.
pub fn hbox(out: &mut [u16], sums: &[u16], k: usize, div: u32) {
    let n = out.len().min(sums.len() / k.max(1));
    #[allow(unused_assignments, unused_mut)]
    let mut done = 0;
    #[cfg(target_arch = "x86_64")]
    if level() >= 1 && matches!(k, 1 | 2 | 4 | 8) && div > 1 {
        // SAFETY: SSSE3 checked; reads sums[..k n].
        done = unsafe { simd::hbox_ssse3(&mut out[..n], &sums[..k * n], k, div) };
    }
    hbox_scalar(&mut out[done..n], &sums[k * done..], k, div);
}

pub fn hbox_scalar(out: &mut [u16], sums: &[u16], k: usize, div: u32) {
    if div <= 1 {
        for (o, &v) in out.iter_mut().zip(sums) {
            *o = v.min(1023);
        }
        return;
    }
    let m = hbox_mul(div);
    for (j, o) in out.iter_mut().enumerate() {
        let s: u32 = sums[j * k..(j + 1) * k].iter().map(|&v| v as u32).sum();
        *o = (((s * m as u32 + (1 << 14)) >> 15) as u16).min(1023);
    }
}

/// `round(32768 / div)`: `(s · m + 2^14) >> 15` is `pmulhrsw`.
fn hbox_mul(div: u32) -> i16 {
    ((32768.0 / div.max(2) as f64).round() as i32).min(i16::MAX as i32) as i16
}

#[cfg(target_arch = "x86_64")]
mod simd {
    use super::HTab;
    use std::arch::x86_64::*;

    /// 8 pixels per round: `[cb y cr y]×4` → 8 Y, 4 Cb, 4 Cr.
    #[target_feature(enable = "ssse3")]
    pub unsafe fn deinterleave_ssse3(src: &[u16], y: &mut [u16], cb: &mut [u16], cr: &mut [u16]) -> usize {
        let n = src.len() / 2;
        let rounds = n / 8;
        unsafe {
            // Per 8 samples (4 pixels): Y Y Y Y | Cb Cb | Cr Cr
            let sh = _mm_setr_epi8(2, 3, 6, 7, 10, 11, 14, 15, 0, 1, 8, 9, 4, 5, 12, 13);
            for r in 0..rounds {
                let p = src.as_ptr().add(16 * r);
                let a = _mm_shuffle_epi8(_mm_loadu_si128(p as *const __m128i), sh);
                let b = _mm_shuffle_epi8(_mm_loadu_si128(p.add(8) as *const __m128i), sh);
                // a = Ya | Cba Cra (as 32-bit pairs); b likewise.
                _mm_storeu_si128(y.as_mut_ptr().add(8 * r) as *mut __m128i, _mm_unpacklo_epi64(a, b));
                let c = _mm_unpackhi_epi32(a, b); // Cba Cbb Cra Crb (pairs)
                _mm_storel_epi64(cb.as_mut_ptr().add(4 * r) as *mut __m128i, c);
                _mm_storel_epi64(cr.as_mut_ptr().add(4 * r) as *mut __m128i, _mm_unpackhi_epi64(c, c));
            }
        }
        8 * rounds
    }

    /// 8 pixels per round.
    #[target_feature(enable = "sse2")]
    pub unsafe fn interleave_sse2(y: &[u16], cb: &[u16], cr: &[u16], out: &mut [u16]) -> usize {
        let n = y.len();
        let rounds = n / 8;
        unsafe {
            for r in 0..rounds {
                let yv = _mm_loadu_si128(y.as_ptr().add(8 * r) as *const __m128i);
                let b = _mm_loadl_epi64(cb.as_ptr().add(4 * r) as *const __m128i);
                let c = _mm_loadl_epi64(cr.as_ptr().add(4 * r) as *const __m128i);
                let bc = _mm_unpacklo_epi16(b, c); // cb0 cr0 cb1 cr1 ...
                // cb y cr y: pairs (cb, cr) and pairs (y, y) interleaved as 16-bit.
                let o = out.as_mut_ptr().add(16 * r);
                _mm_storeu_si128(o as *mut __m128i, _mm_unpacklo_epi16(bc, yv));
                _mm_storeu_si128(o.add(8) as *mut __m128i, _mm_unpackhi_epi16(bc, yv));
            }
        }
        8 * rounds
    }

    /// 8 outputs per round: `k` loads folded with `phaddw` (k − 1 adds per
    /// output), then `pmulhrsw` by `round(32768 / div)`.
    #[target_feature(enable = "ssse3")]
    pub unsafe fn hbox_ssse3(out: &mut [u16], sums: &[u16], k: usize, div: u32) -> usize {
        let n = out.len() / 8 * 8;
        unsafe {
            let m = _mm_set1_epi16(super::hbox_mul(div));
            let max = _mm_set1_epi16(1023);
            let p = sums.as_ptr();
            let ld = |i: usize| _mm_loadu_si128(p.add(8 * i) as *const __m128i);
            let mut j = 0;
            while j < n {
                let base = j * k / 8;
                let v = match k {
                    1 => ld(base),
                    2 => _mm_hadd_epi16(ld(base), ld(base + 1)),
                    4 => _mm_hadd_epi16(
                        _mm_hadd_epi16(ld(base), ld(base + 1)),
                        _mm_hadd_epi16(ld(base + 2), ld(base + 3)),
                    ),
                    _ => {
                        let h = |i: usize| _mm_hadd_epi16(ld(base + i), ld(base + i + 1));
                        _mm_hadd_epi16(_mm_hadd_epi16(h(0), h(2)), _mm_hadd_epi16(h(4), h(6)))
                    }
                };
                let r = _mm_min_epi16(_mm_mulhrs_epi16(v, m), max);
                _mm_storeu_si128(out.as_mut_ptr().add(j) as *mut __m128i, r);
                j += 8;
            }
        }
        n
    }

    /// 8 outputs per round, two taps per step (gather + `pmaddwd`).
    #[target_feature(enable = "avx2")]
    pub unsafe fn hfilter_avx2(out: &mut [u16], q: &[u16], tab: &HTab, j0: usize, base: i32, inv: f32) -> usize {
        let n = out.len();
        let rounds = n / 8;
        let stride = tab.len + 8;
        unsafe {
            let basev = _mm256_set1_epi32(base);
            let inv = _mm256_set1_ps(inv);
            let half = _mm256_set1_ps(0.5);
            let max = _mm256_set1_epi32(1023);
            let qp = q.as_ptr() as *const i32;
            for r in 0..rounds {
                let j = j0 + 8 * r;
                let mut idx = _mm256_sub_epi32(_mm256_loadu_si256(tab.start.as_ptr().add(j) as *const __m256i), basev);
                let mut sum = _mm256_setzero_si256();
                for p in 0..tab.pairs {
                    // Scale 2: the i32 at q + 2·idx is samples idx, idx + 1.
                    let v = _mm256_i32gather_epi32::<2>(qp, idx);
                    let w = _mm256_loadu_si256(tab.weights.as_ptr().add(p * stride + j) as *const __m256i);
                    sum = _mm256_add_epi32(sum, _mm256_madd_epi16(v, w));
                    idx = _mm256_add_epi32(idx, _mm256_set1_epi32(2));
                }
                let f = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(sum), inv), half);
                let i = _mm256_min_epi32(_mm256_max_epi32(_mm256_cvttps_epi32(f), _mm256_setzero_si256()), max);
                // 8 × i32 → 8 × u16.
                let packed = _mm256_packus_epi32(i, i);
                let packed = _mm256_permute4x64_epi64::<0b1000>(packed);
                _mm_storeu_si128(out.as_mut_ptr().add(8 * r) as *mut __m128i, _mm256_castsi256_si128(packed));
            }
        }
        8 * rounds
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rnd(n: usize, seed: u32, max: u32) -> Vec<u16> {
        let mut x = seed.wrapping_mul(2654435761).wrapping_add(1);
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                (x % max) as u16
            })
            .collect()
    }

    #[test]
    fn planes_round_trip() {
        for n in [0usize, 2, 6, 8, 14, 16, 30, 1920] {
            let src = rnd(2 * n, n as u32, 1024);
            let (mut y, mut cb, mut cr) = (vec![0; n], vec![0; n / 2], vec![0; n / 2]);
            deinterleave(&src, &mut y, &mut cb, &mut cr);
            for q in 0..n / 2 {
                assert_eq!((cb[q], y[2 * q], cr[q], y[2 * q + 1]), (src[4 * q], src[4 * q + 1], src[4 * q + 2], src[4 * q + 3]));
            }
            let mut back = vec![0; 2 * n];
            interleave(&y, &cb, &cr, &mut back);
            assert_eq!(back, src, "{n}");
        }
    }

    #[test]
    fn simd_filter_matches_scalar_and_reference() {
        for (a, b, n, limit) in [
            (0.0, 1920.0, 672, 1920),
            (0.0, 1920.0, 480, 1920),
            (10.0, 970.0, 333, 1920),
            (0.0, 100.0, 333, 100),
            (3.0, 17.0, 5, 17),
            (0.0, 1920.0, 160, 1920),
        ] {
            for total in [256u32, 1024] {
                let t = Table::with_total(a, b, n, limit, total);
                let tab = HTab::new(&t);
                let total_w = t.total;
                let q = {
                    let mut v: Vec<u16> = rnd(limit, n as u32, 1024).iter().map(|&s| s * 32).collect();
                    v.resize(limit + tab.reach + 16, 0);
                    v
                };
                let (mut x, mut y) = (vec![0u16; n], vec![0u16; n]);
                hfilter(&mut x, &q, &tab, 0, 0);
                hfilter_scalar(&mut y, &q, &tab, 0, 0);
                assert_eq!(x, y, "{a} {b} {n}");
                for j in 0..n {
                    let s: f64 = t
                        .weights(j)
                        .iter()
                        .enumerate()
                        .map(|(k, &w)| w as f64 * q[t.start[j] as usize + k] as f64)
                        .sum();
                    let want = (s / (total_w as f64 * 32.0)).round();
                    assert!((x[j] as f64 - want).abs() <= 1.0, "{j}: {} vs {want}", x[j]);
                }
                // From an offset.
                let mut z = vec![0u16; n / 2];
                hfilter(&mut z, &q, &tab, n / 3, 0);
                assert_eq!(z[..], x[n / 3..n / 3 + n / 2]);
            }
        }
    }

    #[test]
    fn box_folds_match_scalar_and_exact() {
        for k in [1usize, 2, 3, 4, 8] {
            for ky in [1u32, 2, 3, 4] {
                if k as u32 * ky * 1023 > 32767 {
                    continue;
                }
                let n = 101;
                let sums: Vec<u16> = rnd(k * n, k as u32 * 7 + ky, 1024 * ky).iter().map(|&v| v.min(1023 * ky as u16)).collect();
                let div = k as u32 * ky;
                let (mut a, mut b) = (vec![0u16; n], vec![0u16; n]);
                hbox(&mut a, &sums, k, div);
                hbox_scalar(&mut b, &sums, k, div);
                assert_eq!(a, b, "k {k} ky {ky}");
                for j in 0..n {
                    let s: u32 = sums[j * k..(j + 1) * k].iter().map(|&v| v as u32).sum();
                    let want = (s as f64 / div as f64).round();
                    assert!((a[j] as f64 - want).abs() <= 1.0, "k {k} ky {ky} j {j}");
                }
            }
        }
    }

    #[test]
    fn normalize_rounds() {
        let acc = vec![0u32, 1023 * 1024, 512 * 1024 + 16, 3 * 1023];
        let mut out = vec![0u16; 4];
        normalize(&mut out, &acc, 1024);
        assert_eq!(out[..3], [0, 32736, 16384 + 1]);
        normalize(&mut out[3..], &acc[3..], 3);
        assert_eq!(out[3], 32736);
    }
}

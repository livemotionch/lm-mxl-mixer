//! Inner loops. Each is plain Rust written so the compiler vectorises it
//! (fixed-width chunks, no bounds checks in the loop, integer fixed point),
//! compiled twice: for the baseline target and for AVX2, picked at run
//! time ([`avx2`]).
//!
//! Sample precision between stages:
//! - vertical sums: `u16`, sample × weight with weights adding up to ≤ 256;
//! - horizontal results: `u16` in Q6 (value × 64);
//! - canvas: `u8`, premultiplied where there is alpha.

// Index loops over several slices at once are intentional here.
#![allow(clippy::needless_range_loop)]

use super::color::Transform;
use super::formats::div255;
use super::scale::Table;

/// SIMD level of the kernels: 2 = AVX2, 1 = SSE4.1, 0 = baseline. Checked
/// once; `MIXER_SIMD=avx2|sse41|none` lowers it (for comparisons).
#[inline]
pub fn level() -> u8 {
    #[cfg(target_arch = "x86_64")]
    {
        use std::sync::OnceLock;
        static LEVEL: OnceLock<u8> = OnceLock::new();
        *LEVEL.get_or_init(|| {
            let cpu = if std::is_x86_feature_detected!("avx2") {
                2
            } else if std::is_x86_feature_detected!("sse4.1") {
                1
            } else {
                0
            };
            let cap = match std::env::var("MIXER_SIMD").as_deref() {
                Ok("none") => 0,
                Ok("sse41") => 1,
                _ => 2,
            };
            cpu.min(cap)
        })
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        0
    }
}

/// Name of [`level`].
pub fn simd_level() -> &'static str {
    ["baseline", "SSE4.1", "AVX2"][level() as usize]
}

/// Define `pub fn $name` that runs `$body` compiled for AVX2 or SSE4.1 when
/// the CPU has it, and for the baseline otherwise.
macro_rules! multiversion {
    ($(#[$m:meta])* pub fn $name:ident$(<$($g:ident: $bound:path),*>)?($($arg:ident: $ty:ty),* $(,)?) $(-> $ret:ty)? $body:block) => {
        $(#[$m])*
        #[inline]
        #[allow(clippy::too_many_arguments)]
        pub fn $name$(<$($g: $bound),*>)?($($arg: $ty),*) $(-> $ret)? {
            #[inline(always)]
            #[allow(clippy::too_many_arguments)]
            fn imp$(<$($g: $bound),*>)?($($arg: $ty),*) $(-> $ret)? $body
            #[cfg(target_arch = "x86_64")]
            {
                #[target_feature(enable = "avx2")]
                #[allow(clippy::too_many_arguments)]
                unsafe fn avx$(<$($g: $bound),*>)?($($arg: $ty),*) $(-> $ret)? { imp($($arg),*) }
                #[target_feature(enable = "sse4.1")]
                #[allow(clippy::too_many_arguments)]
                unsafe fn sse$(<$($g: $bound),*>)?($($arg: $ty),*) $(-> $ret)? { imp($($arg),*) }
                match $crate::compositor::kernels::level() {
                    // SAFETY: the CPU has the feature (checked at run time).
                    2 => return unsafe { avx($($arg),*) },
                    1 => return unsafe { sse($($arg),*) },
                    _ => {}
                }
            }
            imp($($arg),*)
        }
    };
}
pub(crate) use multiversion;

/// Evaluate `$body` with `$q` bound to the (inlined) Q6 conversion for
/// `$scale`.
macro_rules! with_q {
    ($scale:expr, $q:ident, $body:expr) => {
        match $scale {
            Scale::Shift(k) if k >= 0 => {
                let round = if k > 0 { 1u32 << (k - 1) } else { 0 };
                let $q = move |v: u32| ((v + round) >> k) as u16;
                $body
            }
            Scale::Shift(k) => {
                let $q = move |v: u32| (v << -k) as u16;
                $body
            }
            Scale::Mul(m) => {
                let $q = move |v: u32| ((v as u64 * m + (1 << 31)) >> 32) as u16;
                $body
            }
        }
    };
}

multiversion! {
    /// `acc = row · w`
    pub fn vacc_first(acc: &mut [u16], row: &[u8], w: u16) {
        let n = acc.len().min(row.len());
        let (acc, row) = (&mut acc[..n], &row[..n]);
        if w == 1 {
            for (a, &s) in acc.iter_mut().zip(row) {
                *a = s as u16;
            }
        } else {
            for (a, &s) in acc.iter_mut().zip(row) {
                *a = s as u16 * w;
            }
        }
    }
}

multiversion! {
    /// `acc += row · w`
    pub fn vacc_add(acc: &mut [u16], row: &[u8], w: u16) {
        let n = acc.len().min(row.len());
        let (acc, row) = (&mut acc[..n], &row[..n]);
        if w == 1 {
            for (a, &s) in acc.iter_mut().zip(row) {
                *a = a.wrapping_add(s as u16);
            }
        } else {
            for (a, &s) in acc.iter_mut().zip(row) {
                *a = a.wrapping_add(s as u16 * w);
            }
        }
    }
}

/// `acc = Σ rows[t] · weights[t]`, one row after the other: `acc` (a row
/// of u16, a few KB) stays in L1 while each source row is read straight
/// through, which the hardware prefetchers follow. Reading all rows in one
/// pass, 32 bytes of each in turn, was 1.6× slower for sources in DRAM
/// (multiview thumbnails of live 1080p sources), even with software
/// prefetching. Weights add up to at most 256: the u16 sums can't overflow.
pub fn vacc_rows(acc: &mut [u16], rows: &[&[u8]], weights: &[u16]) {
    let n = rows.iter().fold(acc.len(), |n, r| n.min(r.len()));
    let acc = &mut acc[..n];
    let mut pairs = rows.iter().zip(weights);
    match pairs.next() {
        Some((r, &w)) => vacc_first(acc, r, w),
        None => acc.fill(0),
    }
    for (r, &w) in pairs {
        vacc_add(acc, r, w);
    }
}

/// How a sum with total weight `divisor` becomes Q6 (value × 64).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scale {
    /// Power-of-two divisor: `(sum + round) >> shift` (or `<<` for small ones)
    Shift(i32),
    /// `(sum · m + 2^31) >> 32`
    Mul(u64),
}

pub fn q6_scale(divisor: u64) -> Scale {
    if divisor.is_power_of_two() {
        Scale::Shift(divisor.trailing_zeros() as i32 - 6)
    } else {
        Scale::Mul(((64u64 << 32) + divisor / 2) / divisor)
    }
}

/// Sample type of a summed row: `u16` (one row) or `u32` (two rows added).
pub trait Sample: Copy + Default + Into<u32> + Send + Sync + 'static {}
impl Sample for u16 {}
impl Sample for u32 {}

multiversion! {
/// Horizontal filter of one channel: for outputs `j0..j0 + out.len()`,
/// filter the samples `acc[base + s · step]` (`s` a source index of
/// `table`). `tmp` holds the deinterleaved samples.
pub fn extract<T: Sample>(
    out: &mut [u16],
    acc: &[T],
    base: isize,
    step: usize,
    table: &Table,
    j0: usize,
    scale: Scale,
    tmp: &mut Vec<T>,
) {
    let n = out.len();
    if n == 0 {
        return;
    }
    let (s_lo, s_hi) = table.source_range(j0, j0 + n);
    let first = (base + (s_lo * step) as isize) as usize;
    let m = s_hi - s_lo;
    // Box filters on interleaved samples: sum straight from the strided
    // row, without deinterleaving first.
    if let Some(k) = table.box_k {
        let src = &acc[first..];
        macro_rules! go {
            ($k:literal, $s:literal, $c:literal) => {
                if k == $k && step == $s {
                    with_q!(scale, q, box_strided::<T, _, $k, $s, $c>(out, src, q));
                    return;
                }
            };
        }
        go!(1, 2, 2);
        go!(1, 4, 4);
        go!(2, 2, 4);
        go!(2, 4, 8);
        go!(3, 2, 6);
        go!(3, 4, 12);
        go!(4, 2, 8);
        go!(4, 4, 16);
    }
    let src: &[T] = if step == 1 {
        &acc[first..first + m]
    } else {
        if tmp.len() < m {
            tmp.resize(m, T::default());
        }
        gather(&mut tmp[..m], &acc[first..], step);
        &tmp[..m]
    };
    match scale {
        Scale::Shift(k) if k >= 0 => {
            let round = if k > 0 { 1u32 << (k - 1) } else { 0 };
            filter(out, src, table, j0, s_lo, move |v: u32| ((v + round) >> k) as u16)
        }
        Scale::Shift(k) => filter(out, src, table, j0, s_lo, move |v: u32| (v << -k) as u16),
        Scale::Mul(m) => filter(out, src, table, j0, s_lo, move |v: u32| {
            ((v as u64 * m + (1 << 31)) >> 32) as u16
        }),
    }
}
}

multiversion! {
    /// [`extract`] of the sum of two rows (4:2:0 chroma of a packed source),
    /// adding only the samples it reads.
    pub fn extract_pair(
        out: &mut [u16],
        a: &[u16],
        b: &[u16],
        base: isize,
        step: usize,
        table: &Table,
        j0: usize,
        scale: Scale,
        tmp: &mut Vec<u32>,
    ) {
        let n = out.len();
        if n == 0 {
            return;
        }
        let (s_lo, s_hi) = table.source_range(j0, j0 + n);
        let first = (base + (s_lo * step) as isize) as usize;
        let m = s_hi - s_lo;
        let (a, b) = (&a[first..], &b[first..]);
        if let Some(k) = table.box_k {
            macro_rules! go {
                ($k:literal, $s:literal, $c:literal) => {
                    if k == $k && step == $s {
                        with_q!(scale, q, box_strided2::<_, $k, $s, $c>(out, a, b, q));
                        return;
                    }
                };
            }
            go!(1, 1, 1);
            go!(2, 1, 2);
            go!(3, 1, 3);
            go!(4, 1, 4);
            go!(1, 2, 2);
            go!(1, 4, 4);
            go!(2, 2, 4);
            go!(2, 4, 8);
            go!(3, 2, 6);
            go!(3, 4, 12);
            go!(4, 2, 8);
            go!(4, 4, 16);
        }
        if tmp.len() < m {
            tmp.resize(m, 0);
        }
        let full = (a.len().min(b.len()) / step).min(m);
        match step {
            2 => {
                for ((d, x), y) in tmp[..full].iter_mut().zip(a.as_chunks::<2>().0).zip(b.as_chunks::<2>().0) {
                    *d = x[0] as u32 + y[0] as u32;
                }
            }
            4 => {
                for ((d, x), y) in tmp[..full].iter_mut().zip(a.as_chunks::<4>().0).zip(b.as_chunks::<4>().0) {
                    *d = x[0] as u32 + y[0] as u32;
                }
            }
            _ => {
                for i in 0..full {
                    tmp[i] = a[i * step] as u32 + b[i * step] as u32;
                }
            }
        }
        for i in full..m {
            tmp[i] = a[i * step] as u32 + b[i * step] as u32;
        }
        let src = &tmp[..m];
        with_q!(scale, q, filter(out, src, table, j0, s_lo, q));
    }
}

multiversion! {
    /// `dst[i] = src[i · step]`
    pub fn gather<T: Sample>(dst: &mut [T], src: &[T], step: usize) {
        let n = dst.len();
        match step {
            2 => {
                for (d, s) in dst.iter_mut().zip(src.as_chunks::<2>().0.iter()) {
                    *d = s[0];
                }
                if src.len() < n * 2 {
                    dst[n - 1] = src[(n - 1) * 2];
                }
            }
            4 => {
                let full = (src.len() / 4).min(n);
                for (d, s) in dst[..full].iter_mut().zip(src.as_chunks::<4>().0.iter()) {
                    *d = s[0];
                }
                for i in full..n {
                    dst[i] = src[i * 4];
                }
            }
            _ => {
                for (i, d) in dst.iter_mut().enumerate() {
                    *d = src[i * step];
                }
            }
        }
    }
}

multiversion! {
    /// `dst[i] = src[i · step]` for bytes.
    pub fn gather8(dst: &mut [u8], src: &[u8], step: usize) {
        let n = dst.len();
        match step {
            1 => dst.copy_from_slice(&src[..n]),
            2 => {
                let full = (src.len() / 2).min(n);
                for (d, s) in dst[..full].iter_mut().zip(src.as_chunks::<2>().0.iter()) {
                    *d = s[0];
                }
                for i in full..n {
                    dst[i] = src[i * 2];
                }
            }
            4 => {
                let full = (src.len() / 4).min(n);
                for (d, s) in dst[..full].iter_mut().zip(src.as_chunks::<4>().0.iter()) {
                    *d = s[0];
                }
                for i in full..n {
                    dst[i] = src[i * 4];
                }
            }
            _ => {
                for (i, d) in dst.iter_mut().enumerate() {
                    *d = src[i * step];
                }
            }
        }
    }
}

/// `out[i] = q(Σ_{t<K} src[(i·K + t)·S])`, `C = K·S`. `src` may end
/// right after the last sample read.
#[inline(always)]
fn box_strided<T: Sample, Q: Fn(u32) -> u16, const K: usize, const S: usize, const C: usize>(
    out: &mut [u16],
    src: &[T],
    q: Q,
) {
    let full = (src.len() / C).min(out.len());
    let (head, tail) = out.split_at_mut(full);
    for (o, c) in head.iter_mut().zip(src.as_chunks::<C>().0) {
        let mut sum = 0u32;
        for t in 0..K {
            sum += c[t * S].into();
        }
        *o = q(sum);
    }
    for (i, o) in tail.iter_mut().enumerate() {
        let b = (full + i) * C;
        let mut sum = 0u32;
        for t in 0..K {
            sum += src[b + t * S].into();
        }
        *o = q(sum);
    }
}

/// [`box_strided`] of the sum of two rows.
#[inline(always)]
fn box_strided2<Q: Fn(u32) -> u16, const K: usize, const S: usize, const C: usize>(
    out: &mut [u16],
    a: &[u16],
    b: &[u16],
    q: Q,
) {
    let full = (a.len().min(b.len()) / C).min(out.len());
    let (head, tail) = out.split_at_mut(full);
    for ((o, x), y) in head
        .iter_mut()
        .zip(a.as_chunks::<C>().0)
        .zip(b.as_chunks::<C>().0)
    {
        let mut sum = 0u32;
        for t in 0..K {
            sum += x[t * S] as u32 + y[t * S] as u32;
        }
        *o = q(sum);
    }
    for (i, o) in tail.iter_mut().enumerate() {
        let base = (full + i) * C;
        let mut sum = 0u32;
        for t in 0..K {
            sum += a[base + t * S] as u32 + b[base + t * S] as u32;
        }
        *o = q(sum);
    }
}

#[inline(always)]
fn filter<T: Sample, Q: Fn(u32) -> u16 + Copy>(
    out: &mut [u16],
    src: &[T],
    table: &Table,
    j0: usize,
    s_lo: usize,
    q: Q,
) {
    let n = out.len();
    match table.box_k {
        Some(1) => box1(out, &src[..n], q),
        Some(2) => box_n::<T, Q, 2>(out, &src[..n * 2], q),
        Some(3) => box_n::<T, Q, 3>(out, &src[..n * 3], q),
        Some(4) => box_n::<T, Q, 4>(out, &src[..n * 4], q),
        Some(k) => {
            for (o, c) in out.iter_mut().zip(src.chunks_exact(k as usize)) {
                *o = q(c.iter().map(|&v| v.into()).sum());
            }
        }
        None => {
            let starts = &table.start[j0..j0 + n];
            let taps = table.taps;
            let weights = &table.weight[j0 * taps..(j0 + n) * taps];
            match taps {
                1 => taps_n::<T, Q, 1>(out, src, starts, weights, s_lo, q),
                2 => taps_n::<T, Q, 2>(out, src, starts, weights, s_lo, q),
                3 => taps_n::<T, Q, 3>(out, src, starts, weights, s_lo, q),
                4 => taps_n::<T, Q, 4>(out, src, starts, weights, s_lo, q),
                5 => taps_n::<T, Q, 5>(out, src, starts, weights, s_lo, q),
                6 => taps_n::<T, Q, 6>(out, src, starts, weights, s_lo, q),
                7 => taps_n::<T, Q, 7>(out, src, starts, weights, s_lo, q),
                8 => taps_n::<T, Q, 8>(out, src, starts, weights, s_lo, q),
                _ => {
                    for (i, o) in out.iter_mut().enumerate() {
                        let s = starts[i] as usize - s_lo;
                        let w = &weights[i * taps..(i + 1) * taps];
                        let x = &src[s..s + taps];
                        let sum: u32 = w.iter().zip(x).map(|(&w, &x)| w as u32 * x.into()).sum();
                        *o = q(sum);
                    }
                }
            }
        }
    }
}

#[inline(always)]
fn box1<T: Sample, Q: Fn(u32) -> u16>(out: &mut [u16], src: &[T], q: Q) {
    for (o, &s) in out.iter_mut().zip(src) {
        *o = q(s.into());
    }
}

#[inline(always)]
fn box_n<T: Sample, Q: Fn(u32) -> u16, const K: usize>(out: &mut [u16], src: &[T], q: Q) {
    for (o, c) in out.iter_mut().zip(src.as_chunks::<K>().0) {
        let mut sum = 0u32;
        for &v in c {
            sum += v.into();
        }
        *o = q(sum);
    }
}

#[inline(always)]
fn taps_n<T: Sample, Q: Fn(u32) -> u16, const N: usize>(
    out: &mut [u16],
    src: &[T],
    starts: &[u32],
    weights: &[u16],
    s_lo: usize,
    q: Q,
) {
    for ((o, &start), w) in out.iter_mut().zip(starts).zip(weights.as_chunks::<N>().0) {
        let s = start as usize - s_lo;
        let x: &[T; N] = src[s..s + N].try_into().unwrap();
        let mut sum = 0u32;
        for t in 0..N {
            sum += w[t] as u32 * x[t].into();
        }
        *o = q(sum);
    }
}

multiversion! {
    /// Q6 → u8, rounded.
    pub fn round_q6(out: &mut [u8], q: &[u16]) {
        for (o, &v) in out.iter_mut().zip(q) {
            *o = ((v as u32 + 32) >> 6).min(255) as u8;
        }
    }
}

/// Apply `t`'s output channel `k` to Q6 inputs (premultiplied by `alpha`,
/// Q6, if given), writing premultiplied u8.
///
/// Fixed point: coefficients in Q12, so sums are Q18 and fit in `i32`
/// (|m| < 2.2, inputs < 2^14).
pub fn transform_row(
    out: &mut [u8],
    t: &Transform,
    k: usize,
    q: [&[u16]; 3],
    alpha: Option<&[u16]>,
) {
    let m: [i32; 3] = std::array::from_fn(|j| (t.m[k][j] as f64 * 4096.0).round() as i32);
    let o = t.o[k] as f64;
    match alpha {
        None => transform_opaque(
            out,
            m,
            (o * 262144.0 + 131072.0).round() as i32,
            q[0],
            q[1],
            q[2],
        ),
        Some(a) => transform_alpha(
            out,
            m,
            (o * 262144.0 / (255.0 * 64.0)).round() as i32,
            q[0],
            q[1],
            q[2],
            a,
        ),
    }
}

multiversion! {
    pub fn transform_opaque(out: &mut [u8], m: [i32; 3], o: i32, c0: &[u16], c1: &[u16], c2: &[u16]) {
        let n = out.len();
        let (c0, c1, c2) = (&c0[..n], &c1[..n], &c2[..n]);
        for i in 0..n {
            let v = m[0] * c0[i] as i32 + m[1] * c1[i] as i32 + m[2] * c2[i] as i32 + o;
            out[i] = (v >> 18).clamp(0, 255) as u8;
        }
    }
}

multiversion! {
    pub fn transform_alpha(
        out: &mut [u8],
        m: [i32; 3],
        oa: i32,
        c0: &[u16],
        c1: &[u16],
        c2: &[u16],
        alpha: &[u16],
    ) {
        let n = out.len();
        let (c0, c1, c2, alpha) = (&c0[..n], &c1[..n], &c2[..n], &alpha[..n]);
        for i in 0..n {
            let a = alpha[i] as i32;
            let v = m[0] * c0[i] as i32 + m[1] * c1[i] as i32 + m[2] * c2[i] as i32 + oa * a + (1 << 17);
            // Premultiplied: never above the alpha.
            let a8 = ((a + 32) >> 6).min(255);
            out[i] = (v >> 18).clamp(0, a8) as u8;
        }
    }
}

/// One output channel of an affine map from the 4 bytes of a pixel, in
/// fixed point: `(Σ c[i] · byte[i] · n + o) >> shift`, where `n` is the
/// number of pixels summed (1 for luma, 4 for 2×2 chroma).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PixelMap {
    /// Q14 coefficient per byte of the pixel
    pub c: [i32; 4],
    /// Offset in the result's fixed point, rounding included
    pub o: i32,
}

impl PixelMap {
    /// Output channel `k` of `t` for pixels whose bytes `bytes[j]` hold
    /// input channel `j` (0–2), summing `n` (1 or 4) pixels.
    pub fn new(t: &Transform, k: usize, bytes: [usize; 3], n: u32) -> PixelMap {
        let mut c = [0i32; 4];
        for j in 0..3 {
            c[bytes[j]] = (t.m[k][j] as f64 * 16384.0).round() as i32;
        }
        let shift = 14 + n.trailing_zeros();
        PixelMap {
            c,
            o: (t.o[k] as f64 * (1u64 << shift) as f64).round() as i32 + (1 << (shift - 1)),
        }
    }
}

impl PixelMap {
    /// [`PixelMap::new`] for premultiplied pixels with alpha in byte
    /// `alpha`: the map is linear in premultiplied colour, its offset
    /// scaled by the alpha (`o · a / 255`), which becomes the alpha byte's
    /// coefficient. The result is premultiplied.
    pub fn premultiplied(
        t: &Transform,
        k: usize,
        bytes: [usize; 3],
        alpha: usize,
        n: u32,
    ) -> PixelMap {
        let mut m = PixelMap::new(t, k, bytes, n);
        let shift = 14 + n.trailing_zeros();
        m.c[alpha] = (t.o[k] as f64 * 16384.0 / 255.0).round() as i32;
        m.o = 1 << (shift - 1);
        m
    }

    /// Whether the coefficients fit the 16-bit SIMD kernels.
    pub fn fits_i16(&self) -> bool {
        self.c.iter().all(|&c| (-32768..32768).contains(&c))
    }
}

/// One canvas channel of same-size 4-byte pixels without alpha (RGBx,
/// BGRx): `out[i] = map(src pixel i)`. The coefficients must fit `i16`.
pub fn pixels_map(out: &mut [u8], src: &[u8], map: PixelMap) {
    let n = out.len().min(src.len() / 4);
    // SAFETY: the CPU has the feature (checked at run time); the kernels
    // stay within `out[..n]` and `src[..4 n]`.
    #[cfg(target_arch = "x86_64")]
    let done = unsafe {
        match level() {
            2 => simd::pixels_map_avx2(&mut out[..n], &src[..n * 4], map),
            1 => simd::pixels_map_ssse3(&mut out[..n], &src[..n * 4], map),
            _ => 0,
        }
    };
    #[cfg(not(target_arch = "x86_64"))]
    let done = 0;
    let c = map.c;
    for (d, p) in out[done..n]
        .iter_mut()
        .zip(src[done * 4..].as_chunks::<4>().0)
    {
        let v = c[0] * p[0] as i32
            + c[1] * p[1] as i32
            + c[2] * p[2] as i32
            + c[3] * p[3] as i32
            + map.o;
        *d = (v >> 14).clamp(0, 255) as u8;
    }
}

/// Two canvas channels (Cb, Cr) of 4:2:0 chroma from same-size 4-byte
/// pixels without alpha: each output from the 2×2 pixels of rows `a` and
/// `b` (maps made with `n = 4`; coefficients within `i16`).
pub fn pixels_map_420(u: &mut [u8], v: &mut [u8], a: &[u8], b: &[u8], mu: PixelMap, mv: PixelMap) {
    let n = u.len().min(v.len()).min(a.len() / 8).min(b.len() / 8);
    // SAFETY: as in `pixels_map`.
    #[cfg(target_arch = "x86_64")]
    let done = unsafe {
        let (u, v, a, b) = (&mut u[..n], &mut v[..n], &a[..n * 8], &b[..n * 8]);
        match level() {
            2 => simd::pixels_map_420_avx2(u, v, a, b, mu, mv),
            1 => simd::pixels_map_420_ssse3(u, v, a, b, mu, mv),
            _ => 0,
        }
    };
    #[cfg(not(target_arch = "x86_64"))]
    let done = 0;
    let (a, b) = (a.as_chunks::<8>().0, b.as_chunks::<8>().0);
    for i in done..n {
        let (x, y) = (&a[i], &b[i]);
        let s: [i32; 4] =
            std::array::from_fn(|c| x[c] as i32 + x[c + 4] as i32 + y[c] as i32 + y[c + 4] as i32);
        let su = mu.c[0] * s[0] + mu.c[1] * s[1] + mu.c[2] * s[2] + mu.c[3] * s[3] + mu.o;
        let sv = mv.c[0] * s[0] + mv.c[1] * s[1] + mv.c[2] * s[2] + mv.c[3] * s[3] + mv.o;
        u[i] = (su >> 16).clamp(0, 255) as u8;
        v[i] = (sv >> 16).clamp(0, 255) as u8;
    }
}

/// Horizontal box filter of packed 4:2:2 (UYVY, YUY2) vertical sums, to u8.
///
/// `acc` holds the sums (u16, one per byte of the source layout) starting
/// at the first pixel pair of the first output; `k` (2 or 4) pixels per
/// output, divided by `2^shift` (the box's area) with rounding. `y_odd`:
/// luma is in the odd bytes (UYVY) rather than the even ones (YUY2).
pub fn packed422_luma(out: &mut [u8], acc: &[u16], k: usize, shift: u32, y_odd: bool) {
    let n = out.len().min(acc.len() / (2 * k));
    #[cfg(target_arch = "x86_64")]
    // SAFETY: the CPU has the feature (checked at run time); reads stay
    // within `acc[..2 k n]`.
    let done = unsafe {
        match level() {
            2 => simd::packed422_luma_avx2(&mut out[..n], &acc[..2 * k * n], k, shift, y_odd),
            1 => simd::packed422_luma_sse41(&mut out[..n], &acc[..2 * k * n], k, shift, y_odd),
            _ => 0,
        }
    };
    #[cfg(not(target_arch = "x86_64"))]
    let done = 0;
    let (y0, round) = (y_odd as usize, 1u32 << shift >> 1);
    for j in done..n {
        let s: u32 = (0..k).map(|t| acc[2 * (j * k + t) + y0] as u32).sum();
        out[j] = ((s + round) >> shift).min(255) as u8;
    }
}

/// 4:2:0 chroma of packed 4:2:2 vertical sums: rows `a` and `b` added, `k`
/// (2 or 4) chroma samples (pairs of pixels) per output, divided by
/// `2^shift`. Same layout as [`packed422_luma`]; Cb is the first chroma
/// byte of a pair, Cr the second.
pub fn packed422_chroma(
    u: &mut [u8],
    v: &mut [u8],
    a: &[u16],
    b: &[u16],
    k: usize,
    shift: u32,
    y_odd: bool,
) {
    let n = u
        .len()
        .min(v.len())
        .min(a.len() / (4 * k))
        .min(b.len() / (4 * k));
    #[cfg(target_arch = "x86_64")]
    // SAFETY: as in `packed422_luma`.
    let done = unsafe {
        let (u, v, a, b) = (&mut u[..n], &mut v[..n], &a[..4 * k * n], &b[..4 * k * n]);
        match level() {
            2 => simd::packed422_chroma_avx2(u, v, a, b, k, shift, y_odd),
            1 => simd::packed422_chroma_sse41(u, v, a, b, k, shift, y_odd),
            _ => 0,
        }
    };
    #[cfg(not(target_arch = "x86_64"))]
    let done = 0;
    let (c0, round) = (!y_odd as usize, 1u32 << shift >> 1);
    for j in done..n {
        let sum = |o: usize| -> u32 {
            (0..k)
                .map(|t| {
                    let i = 4 * (j * k + t) + c0 + o;
                    a[i] as u32 + b[i] as u32
                })
                .sum()
        };
        u[j] = ((sum(0) + round) >> shift).min(255) as u8;
        v[j] = ((sum(2) + round) >> shift).min(255) as u8;
    }
}

/// 4:2:0 chroma of same-size packed 4:2:2 rows `a` and `b`: each output
/// the rounded average of the two rows' sample (`y_odd` as in
/// [`packed422_luma`]).
pub fn packed422_chroma_avg(u: &mut [u8], v: &mut [u8], a: &[u8], b: &[u8], y_odd: bool) {
    let n = u.len().min(v.len()).min(a.len() / 4).min(b.len() / 4);
    #[cfg(target_arch = "x86_64")]
    // SAFETY: as in `packed422_luma`.
    let done = unsafe {
        let (u, v, a, b) = (&mut u[..n], &mut v[..n], &a[..4 * n], &b[..4 * n]);
        match level() {
            2 => simd::packed422_chroma_avg_avx2(u, v, a, b, y_odd),
            1 => simd::packed422_chroma_avg_sse41(u, v, a, b, y_odd),
            _ => 0,
        }
    };
    #[cfg(not(target_arch = "x86_64"))]
    let done = 0;
    let c0 = !y_odd as usize;
    for j in done..n {
        let i = 4 * j + c0;
        u[j] = ((a[i] as u16 + b[i] as u16 + 1) >> 1) as u8;
        v[j] = ((a[i + 2] as u16 + b[i + 2] as u16 + 1) >> 1) as u8;
    }
}

/// Sums of 4-byte pixels (u16 per byte, each below 2^15) → one u8 pixel
/// per `k` of them, divided by `div` (the box's area) with rounding.
pub fn box_reduce4(out: &mut [u8], acc: &[u16], k: usize, div: u32) {
    let n = (out.len() / 4).min(acc.len() / (4 * k));
    // (s · m + 2^14) >> 15 with m = 2^15 / div: exact for powers of two,
    // within 0.07 of s / div otherwise (s < 2^12).
    let m = ((1u32 << 15) + div / 2) / div;
    #[cfg(target_arch = "x86_64")]
    // SAFETY: the CPU has the feature (checked at run time); reads stay
    // within `acc[..4 k n]`.
    let done = unsafe {
        match level() {
            2 => simd::box_reduce4_avx2(&mut out[..4 * n], &acc[..4 * k * n], k, m as i16),
            1 => simd::box_reduce4_sse41(&mut out[..4 * n], &acc[..4 * k * n], k, m as i16),
            _ => 0,
        }
    };
    #[cfg(not(target_arch = "x86_64"))]
    let done = 0;
    for j in done..n {
        for c in 0..4 {
            let s: u32 = (0..k).map(|t| acc[4 * (j * k + t) + c] as u32).sum();
            out[4 * j + c] = ((s * m + (1 << 14)) >> 15).min(255) as u8;
        }
    }
}

/// Scaling of [`hbox`] and [`hbox_pairs`], for `k` vertical sums per
/// output whose weights add up to `vtotal` (a box of rows: their number;
/// an area filter: 256):
///
/// - each sum is shifted right by `shift` with rounding, the least that
///   keeps `k` of them below 2^15 (box sums of few rows: none; an area
///   filter's: to Q4 for k = 8);
/// - the `k` are added and divided by `div = k · vtotal / 2^shift`, as
///   `(s · m + 2^14) >> 15` (`_mm_mulhrs_epi16`) with `m = 2^(15 + e) / div`,
///   then shifted right by `e` with rounding. Areas that are powers of two
///   have `e = 0` and are exact; for the others `e` keeps 15 bits in `m`,
///   which puts the result within `0.5 + 2^-(e + 1) + 0.01` of the exact
///   quotient (`e` ≥ 5 for every area above 48).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HBox {
    pub shift: u32,
    m: i16,
    e: u32,
    /// `div` is 1: no scaling
    identity: bool,
}

impl HBox {
    pub fn new(k: usize, vtotal: u32) -> HBox {
        let k = k.max(1) as u64;
        let max = 255 * vtotal as u64;
        let mut shift = 0;
        while k * ((max >> shift) + 1) > i16::MAX as u64 {
            shift += 1;
        }
        let div = (k * vtotal as u64) as f64 / (1u64 << shift) as f64;
        let pow2 = div.fract() == 0.0 && (div as u64).is_power_of_two();
        let e = if pow2 {
            0
        } else {
            // The most bits that keep m and the result in i16.
            ((i16::MAX as f64 * div / 32768.0).log2().floor() as u32).min(7)
        };
        let m = ((1u64 << (15 + e)) as f64 / div)
            .round()
            .min(i16::MAX as f64) as i16;
        HBox {
            shift,
            m,
            e,
            identity: div == 1.0,
        }
    }

    /// The divisor (tests).
    #[doc(hidden)]
    pub fn div(self) -> f64 {
        if self.identity {
            1.0
        } else {
            (1u64 << (15 + self.e)) as f64 / self.m as f64
        }
    }

    /// The `e` above (tests).
    #[doc(hidden)]
    pub fn extra_bits(self) -> u32 {
        self.e
    }

    #[inline(always)]
    fn pre(self, v: u16) -> u32 {
        if self.shift == 0 {
            v as u32
        } else {
            (v as u32 + (1 << (self.shift - 1))) >> self.shift
        }
    }

    #[inline(always)]
    fn scale(self, s: u32) -> u8 {
        if self.identity {
            return s.min(255) as u8;
        }
        let x = (s * self.m as u32 + (1 << 14)) >> 15;
        let x = if self.e == 0 {
            x
        } else {
            (x + (1 << (self.e - 1))) >> self.e
        };
        x.min(255) as u8
    }
}

/// Horizontal box of one channel's vertical sums, to bytes: `out[j]` from
/// `acc[j·k .. (j + 1)·k]` ([`HBox`]). Planar luma and I420 chroma.
pub fn hbox(out: &mut [u8], acc: &[u16], k: usize, hb: HBox) {
    let n = out.len().min(acc.len() / k.max(1));
    #[cfg(target_arch = "x86_64")]
    // SAFETY: the CPU has the feature (checked at run time); reads stay
    // within `acc[..k n]`.
    let done = unsafe {
        match level() {
            2 => simd::hbox_avx2(&mut out[..n], &acc[..k * n], k, hb),
            1 => simd::hbox_sse41(&mut out[..n], &acc[..k * n], k, hb),
            _ => 0,
        }
    };
    #[cfg(not(target_arch = "x86_64"))]
    let done = 0;
    for j in done..n {
        let s: u32 = acc[j * k..(j + 1) * k].iter().map(|&v| hb.pre(v)).sum();
        out[j] = hb.scale(s);
    }
}

/// [`hbox`] of interleaved pairs (NV12 chroma: `acc` is `[Cb Cr Cb Cr ...]`)
/// into separate `u` and `v` rows.
pub fn hbox_pairs(u: &mut [u8], v: &mut [u8], acc: &[u16], k: usize, hb: HBox) {
    let n = u.len().min(v.len()).min(acc.len() / (2 * k.max(1)));
    #[cfg(target_arch = "x86_64")]
    // SAFETY: as in `hbox`; reads stay within `acc[..2 k n]`.
    let done = unsafe {
        let (u, v, acc) = (&mut u[..n], &mut v[..n], &acc[..2 * k * n]);
        match level() {
            2 => simd::hbox_pairs_avx2(u, v, acc, k, hb),
            1 => simd::hbox_pairs_sse41(u, v, acc, k, hb),
            _ => 0,
        }
    };
    #[cfg(not(target_arch = "x86_64"))]
    let done = 0;
    for j in done..n {
        let (mut a, mut b) = (0u32, 0u32);
        for t in 0..k {
            a += hb.pre(acc[2 * (j * k + t)]);
            b += hb.pre(acc[2 * (j * k + t) + 1]);
        }
        u[j] = hb.scale(a);
        v[j] = hb.scale(b);
    }
}

/// Ask for `bytes` to be loaded into the caches (a row read next), one
/// prefetch per cache line; nothing off x86_64.
#[inline]
pub fn prefetch(bytes: &[u8]) {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
        let p = bytes.as_ptr();
        let mut i = 0;
        while i < bytes.len() {
            // SAFETY: prefetching never faults; the address is in `bytes`.
            unsafe { _mm_prefetch::<_MM_HINT_T0>(p.add(i) as *const i8) };
            i += 64;
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = bytes;
}

/// Hand-written kernels. Each handles a multiple of its block size and
/// returns how many outputs it wrote; the caller does the rest.
#[cfg(target_arch = "x86_64")]
mod simd {
    use super::PixelMap;
    use std::arch::x86_64::*;

    /// [`super::HBox`]'s shift of 8 vertical sums.
    #[inline(always)]
    unsafe fn hbox_pre(x: __m128i, hb: super::HBox) -> __m128i {
        if hb.shift == 0 {
            x
        } else {
            let r = _mm_set1_epi16(1 << (hb.shift - 1));
            _mm_srl_epi16(_mm_add_epi16(x, r), _mm_cvtsi32_si128(hb.shift as i32))
        }
    }

    /// [`super::HBox`]'s division of 8 sums (i16).
    #[inline(always)]
    unsafe fn hbox_scale(x: __m128i, hb: super::HBox) -> __m128i {
        if hb.identity {
            return x;
        }
        let x = _mm_mulhrs_epi16(x, _mm_set1_epi16(hb.m));
        if hb.e == 0 {
            x
        } else {
            let r = _mm_set1_epi16(1 << (hb.e - 1));
            _mm_srl_epi16(_mm_add_epi16(x, r), _mm_cvtsi32_si128(hb.e as i32))
        }
    }

    /// 8 outputs of [`super::hbox`] (`K` samples each) at `p`, as i16 sums.
    #[inline(always)]
    unsafe fn hbox8<const K: usize>(p: *const u16, hb: super::HBox) -> __m128i {
        let ld = |i: usize| hbox_pre(_mm_loadu_si128(p.add(8 * i) as *const __m128i), hb);
        match K {
            1 => ld(0),
            2 => _mm_hadd_epi16(ld(0), ld(1)),
            4 => _mm_hadd_epi16(_mm_hadd_epi16(ld(0), ld(1)), _mm_hadd_epi16(ld(2), ld(3))),
            _ => {
                let a = _mm_hadd_epi16(_mm_hadd_epi16(ld(0), ld(1)), _mm_hadd_epi16(ld(2), ld(3)));
                let b = _mm_hadd_epi16(_mm_hadd_epi16(ld(4), ld(5)), _mm_hadd_epi16(ld(6), ld(7)));
                _mm_hadd_epi16(a, b)
            }
        }
    }

    #[inline(always)]
    unsafe fn hbox_k<const K: usize>(out: &mut [u8], acc: &[u16], hb: super::HBox) -> usize {
        let n = out.len().min(acc.len() / K) / 8 * 8;
        let (p, o) = (acc.as_ptr(), out.as_mut_ptr());
        let mut j = 0;
        while j + 16 <= n {
            let a = hbox_scale(hbox8::<K>(p.add(K * j), hb), hb);
            let b = hbox_scale(hbox8::<K>(p.add(K * (j + 8)), hb), hb);
            _mm_storeu_si128(o.add(j) as *mut __m128i, _mm_packus_epi16(a, b));
            j += 16;
        }
        if j < n {
            let a = hbox_scale(hbox8::<K>(p.add(K * j), hb), hb);
            _mm_storel_epi64(o.add(j) as *mut __m128i, _mm_packus_epi16(a, a));
            j += 8;
        }
        j
    }

    #[inline(always)]
    unsafe fn hbox_x86(out: &mut [u8], acc: &[u16], k: usize, hb: super::HBox) -> usize {
        match k {
            1 => hbox_k::<1>(out, acc, hb),
            2 => hbox_k::<2>(out, acc, hb),
            4 => hbox_k::<4>(out, acc, hb),
            8 => hbox_k::<8>(out, acc, hb),
            _ => 0,
        }
    }

    #[target_feature(enable = "sse4.1")]
    pub unsafe fn hbox_sse41(out: &mut [u8], acc: &[u16], k: usize, hb: super::HBox) -> usize {
        hbox_x86(out, acc, k, hb)
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn hbox_avx2(out: &mut [u8], acc: &[u16], k: usize, hb: super::HBox) -> usize {
        hbox_x86(out, acc, k, hb)
    }

    /// Pairs `[x0 y0 x1 y1 ...]` of `a` and `b` (4 each) added in twos:
    /// `[x0+x1, y0+y1, x2+x3, y2+y3]` of `a`, then of `b`.
    #[inline(always)]
    unsafe fn fold_pairs(a: __m128i, b: __m128i) -> __m128i {
        let (a, b) = (_mm_castsi128_ps(a), _mm_castsi128_ps(b));
        _mm_add_epi16(
            _mm_castps_si128(_mm_shuffle_ps::<0b10_00_10_00>(a, b)),
            _mm_castps_si128(_mm_shuffle_ps::<0b11_01_11_01>(a, b)),
        )
    }

    /// 4 output pairs of [`super::hbox_pairs`] (`K` pairs each) at `p`, as
    /// i16 sums `[u0 v0 u1 v1 u2 v2 u3 v3]`.
    #[inline(always)]
    unsafe fn hbox_pairs4<const K: usize>(p: *const u16, hb: super::HBox) -> __m128i {
        let ld = |i: usize| hbox_pre(_mm_loadu_si128(p.add(8 * i) as *const __m128i), hb);
        match K {
            1 => ld(0),
            2 => fold_pairs(ld(0), ld(1)),
            4 => fold_pairs(fold_pairs(ld(0), ld(1)), fold_pairs(ld(2), ld(3))),
            _ => {
                let a = fold_pairs(fold_pairs(ld(0), ld(1)), fold_pairs(ld(2), ld(3)));
                let b = fold_pairs(fold_pairs(ld(4), ld(5)), fold_pairs(ld(6), ld(7)));
                fold_pairs(a, b)
            }
        }
    }

    #[inline(always)]
    unsafe fn hbox_pairs_k<const K: usize>(
        u: &mut [u8],
        v: &mut [u8],
        acc: &[u16],
        hb: super::HBox,
    ) -> usize {
        let n = u.len().min(v.len()).min(acc.len() / (2 * K)) / 8 * 8;
        let p = acc.as_ptr();
        // [u0 v0 u1 v1 ...] → [u0 u1 ... u7 v0 v1 ... v7]
        let split = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
        let mut j = 0;
        while j < n {
            let a = hbox_scale(hbox_pairs4::<K>(p.add(2 * K * j), hb), hb);
            let b = hbox_scale(hbox_pairs4::<K>(p.add(2 * K * (j + 4)), hb), hb);
            let w = _mm_shuffle_epi8(_mm_packus_epi16(a, b), split);
            _mm_storel_epi64(u.as_mut_ptr().add(j) as *mut __m128i, w);
            _mm_storel_epi64(
                v.as_mut_ptr().add(j) as *mut __m128i,
                _mm_unpackhi_epi64(w, w),
            );
            j += 8;
        }
        n
    }

    #[inline(always)]
    unsafe fn hbox_pairs_x86(
        u: &mut [u8],
        v: &mut [u8],
        acc: &[u16],
        k: usize,
        hb: super::HBox,
    ) -> usize {
        match k {
            1 => hbox_pairs_k::<1>(u, v, acc, hb),
            2 => hbox_pairs_k::<2>(u, v, acc, hb),
            4 => hbox_pairs_k::<4>(u, v, acc, hb),
            8 => hbox_pairs_k::<8>(u, v, acc, hb),
            _ => 0,
        }
    }

    #[target_feature(enable = "sse4.1")]
    pub unsafe fn hbox_pairs_sse41(
        u: &mut [u8],
        v: &mut [u8],
        acc: &[u16],
        k: usize,
        hb: super::HBox,
    ) -> usize {
        hbox_pairs_x86(u, v, acc, k, hb)
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn hbox_pairs_avx2(
        u: &mut [u8],
        v: &mut [u8],
        acc: &[u16],
        k: usize,
        hb: super::HBox,
    ) -> usize {
        hbox_pairs_x86(u, v, acc, k, hb)
    }

    /// [`super::box_reduce4`]: 4 outputs (16 bytes) per step; returns
    /// outputs done. `k` of 2 or 4 with whole-register loads, any other `k`
    /// with one 64-bit load (a pixel's 4 sums) per source pixel.
    #[inline(always)]
    unsafe fn box_reduce4_x86(out: &mut [u8], acc: &[u16], k: usize, m: i16) -> usize {
        let n = (out.len() / 4).min(acc.len() / (4 * k)) / 4 * 4;
        let mm = _mm_set1_epi16(m);
        let p = acc.as_ptr();
        if k != 2 && k != 4 {
            // Sum of output `j`'s k pixels, in the low 64 bits.
            let one = |j: usize| {
                let mut v = _mm_loadl_epi64(p.add(4 * j * k) as *const __m128i);
                for t in 1..k {
                    v = _mm_add_epi16(v, _mm_loadl_epi64(p.add(4 * (j * k + t)) as *const __m128i));
                }
                v
            };
            let mut j = 0;
            while j < n {
                let a = _mm_mulhrs_epi16(_mm_unpacklo_epi64(one(j), one(j + 1)), mm);
                let b = _mm_mulhrs_epi16(_mm_unpacklo_epi64(one(j + 2), one(j + 3)), mm);
                _mm_storeu_si128(
                    out.as_mut_ptr().add(4 * j) as *mut __m128i,
                    _mm_packus_epi16(a, b),
                );
                j += 4;
            }
            return n;
        }
        let ld = |i: usize| _mm_loadu_si128(p.add(8 * i) as *const __m128i);
        // Two outputs: [o0 | o1] (4 channels each, u16)
        let two = |i: usize| {
            if k == 2 {
                let (a, b) = (ld(i), ld(i + 1));
                _mm_add_epi16(_mm_unpacklo_epi64(a, b), _mm_unpackhi_epi64(a, b))
            } else {
                let x = _mm_add_epi16(ld(i), ld(i + 1));
                let y = _mm_add_epi16(ld(i + 2), ld(i + 3));
                _mm_add_epi16(_mm_unpacklo_epi64(x, y), _mm_unpackhi_epi64(x, y))
            }
        };
        let regs = k / 2; // loads per output
        let mut j = 0;
        while j < n {
            let a = _mm_mulhrs_epi16(two(j * regs), mm);
            let b = _mm_mulhrs_epi16(two((j + 2) * regs), mm);
            _mm_storeu_si128(
                out.as_mut_ptr().add(4 * j) as *mut __m128i,
                _mm_packus_epi16(a, b),
            );
            j += 4;
        }
        n
    }

    #[target_feature(enable = "sse4.1")]
    pub unsafe fn box_reduce4_sse41(out: &mut [u8], acc: &[u16], k: usize, m: i16) -> usize {
        box_reduce4_x86(out, acc, k, m)
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn box_reduce4_avx2(out: &mut [u8], acc: &[u16], k: usize, m: i16) -> usize {
        box_reduce4_x86(out, acc, k, m)
    }

    /// 16-bit lanes `[1, 0, 1, 0, ...]` (`odd`: `[0, 1, ...]`): with
    /// `madd`, picks every other sample as i32.
    #[inline(always)]
    unsafe fn pick(odd: bool) -> __m128i {
        if odd {
            _mm_set1_epi32(1 << 16)
        } else {
            _mm_set1_epi32(1)
        }
    }

    /// The luma sums of 4 pixels (8 samples at `p`) as i32.
    #[inline(always)]
    unsafe fn luma_of4(p: *const u16, y_odd: bool) -> __m128i {
        let v = _mm_loadu_si128(p as *const __m128i);
        if y_odd {
            _mm_srli_epi32::<16>(v)
        } else {
            _mm_and_si128(v, _mm_set1_epi32(0xFFFF))
        }
    }

    /// 4 luma outputs of `K` pixels each, as i32 sums: pixels packed to
    /// i16 and added in pairs with `madd` (sums stay below 2^15).
    #[inline(always)]
    unsafe fn luma4<const K: usize>(p: *const u16, y_odd: bool) -> __m128i {
        let ones = _mm_set1_epi16(1);
        let pairs = |i: usize| {
            _mm_madd_epi16(
                _mm_packs_epi32(
                    luma_of4(p.add(16 * i), y_odd),
                    luma_of4(p.add(16 * i + 8), y_odd),
                ),
                ones,
            )
        };
        if K == 2 {
            pairs(0)
        } else {
            _mm_madd_epi16(_mm_packs_epi32(pairs(0), pairs(1)), ones)
        }
    }

    #[inline(always)]
    unsafe fn packed422_luma_k<const K: usize>(
        out: &mut [u8],
        acc: &[u16],
        shift: u32,
        y_odd: bool,
    ) -> usize {
        let n = out.len().min(acc.len() / (2 * K)) / 4 * 4;
        let round = _mm_set1_epi32((1 << shift) >> 1);
        let sh = _mm_cvtsi32_si128(shift as i32);
        let f = |x: __m128i| _mm_sra_epi32(_mm_add_epi32(x, round), sh);
        let (p, o) = (acc.as_ptr(), out.as_mut_ptr());
        let mut j = 0;
        while j + 16 <= n {
            let r0 = f(luma4::<K>(p.add(2 * K * j), y_odd));
            let r1 = f(luma4::<K>(p.add(2 * K * (j + 4)), y_odd));
            let r2 = f(luma4::<K>(p.add(2 * K * (j + 8)), y_odd));
            let r3 = f(luma4::<K>(p.add(2 * K * (j + 12)), y_odd));
            let w = _mm_packus_epi16(_mm_packs_epi32(r0, r1), _mm_packs_epi32(r2, r3));
            _mm_storeu_si128(o.add(j) as *mut __m128i, w);
            j += 16;
        }
        while j < n {
            let r = f(luma4::<K>(p.add(2 * K * j), y_odd));
            let w = _mm_packus_epi16(_mm_packs_epi32(r, r), r);
            (o.add(j) as *mut i32).write_unaligned(_mm_cvtsi128_si32(w));
            j += 4;
        }
        n
    }

    #[inline(always)]
    unsafe fn packed422_luma_x86(
        out: &mut [u8],
        acc: &[u16],
        k: usize,
        shift: u32,
        y_odd: bool,
    ) -> usize {
        match k {
            2 => packed422_luma_k::<2>(out, acc, shift, y_odd),
            4 => packed422_luma_k::<4>(out, acc, shift, y_odd),
            _ => 0,
        }
    }

    #[target_feature(enable = "sse4.1")]
    pub unsafe fn packed422_luma_sse41(
        out: &mut [u8],
        acc: &[u16],
        k: usize,
        shift: u32,
        y_odd: bool,
    ) -> usize {
        packed422_luma_x86(out, acc, k, shift, y_odd)
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn packed422_luma_avx2(
        out: &mut [u8],
        acc: &[u16],
        k: usize,
        shift: u32,
        y_odd: bool,
    ) -> usize {
        packed422_luma_x86(out, acc, k, shift, y_odd)
    }

    /// 4 chroma outputs `[U0 V0 U1 V1 U2 V2 U3 V3]` as i16 (after the
    /// shift).
    #[inline(always)]
    unsafe fn chroma4(
        a: *const u16,
        b: *const u16,
        k: usize,
        sel: __m128i,
        round: __m128i,
        shift: __m128i,
    ) -> __m128i {
        // Each load: 2 chroma samples → [U0 U1 V0 V1] as i32.
        let ld = |i: usize| {
            let s = _mm_add_epi16(
                _mm_loadu_si128(a.add(8 * i) as *const __m128i),
                _mm_loadu_si128(b.add(8 * i) as *const __m128i),
            );
            _mm_shuffle_epi32::<0b11_01_10_00>(_mm_madd_epi16(s, sel))
        };
        // Two outputs: [U V U V] (i32)
        let two = |i: usize| {
            if k == 2 {
                _mm_hadd_epi32(ld(i), ld(i + 1))
            } else {
                let x =
                    _mm_shuffle_epi32::<0b11_01_10_00>(_mm_hadd_epi32(ld(2 * i), ld(2 * i + 1)));
                let y = _mm_shuffle_epi32::<0b11_01_10_00>(_mm_hadd_epi32(
                    ld(2 * i + 2),
                    ld(2 * i + 3),
                ));
                _mm_hadd_epi32(x, y)
            }
        };
        let f = |x: __m128i| _mm_sra_epi32(_mm_add_epi32(x, round), shift);
        _mm_packs_epi32(f(two(0)), f(two(2)))
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn packed422_chroma_x86(
        u: &mut [u8],
        v: &mut [u8],
        a: &[u16],
        b: &[u16],
        k: usize,
        shift: u32,
        y_odd: bool,
    ) -> usize {
        if k != 2 && k != 4 {
            return 0;
        }
        let n = u
            .len()
            .min(v.len())
            .min(a.len() / (4 * k))
            .min(b.len() / (4 * k))
            / 4
            * 4;
        let sel = pick(!y_odd);
        let round = _mm_set1_epi32((1 << shift) >> 1);
        let sh = _mm_cvtsi32_si128(shift as i32);
        // [U0 V0 U1 V1 ...] → [U0 U1 U2 ... | V0 V1 V2 ...]
        let split = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let (pu, pv) = (u.as_mut_ptr(), v.as_mut_ptr());
        let mut j = 0;
        while j + 8 <= n {
            let x = chroma4(pa.add(4 * k * j), pb.add(4 * k * j), k, sel, round, sh);
            let y = chroma4(
                pa.add(4 * k * (j + 4)),
                pb.add(4 * k * (j + 4)),
                k,
                sel,
                round,
                sh,
            );
            let w = _mm_shuffle_epi8(_mm_packus_epi16(x, y), split);
            _mm_storel_epi64(pu.add(j) as *mut __m128i, w);
            _mm_storel_epi64(pv.add(j) as *mut __m128i, _mm_srli_si128::<8>(w));
            j += 8;
        }
        while j < n {
            let x = chroma4(pa.add(4 * k * j), pb.add(4 * k * j), k, sel, round, sh);
            let w = _mm_shuffle_epi8(_mm_packus_epi16(x, x), split);
            (pu.add(j) as *mut i32).write_unaligned(_mm_cvtsi128_si32(w));
            (pv.add(j) as *mut i32).write_unaligned(_mm_cvtsi128_si32(_mm_srli_si128::<8>(w)));
            j += 4;
        }
        n
    }

    #[inline(always)]
    unsafe fn packed422_chroma_avg_x86(
        u: &mut [u8],
        v: &mut [u8],
        a: &[u8],
        b: &[u8],
        y_odd: bool,
    ) -> usize {
        let n = u.len().min(v.len()).min(a.len() / 4).min(b.len() / 4) / 8 * 8;
        // [U0 V0 U1 V1 ...] of 4 samples → [U0 U1 U2 U3 V0 V1 V2 V3]
        let split = if y_odd {
            _mm_setr_epi8(0, 4, 8, 12, 2, 6, 10, 14, -1, -1, -1, -1, -1, -1, -1, -1)
        } else {
            _mm_setr_epi8(1, 5, 9, 13, 3, 7, 11, 15, -1, -1, -1, -1, -1, -1, -1, -1)
        };
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let (pu, pv) = (u.as_mut_ptr(), v.as_mut_ptr());
        let mut j = 0;
        while j < n {
            let m = |o: usize| {
                let x = _mm_loadu_si128(pa.add(4 * j + o) as *const __m128i);
                let y = _mm_loadu_si128(pb.add(4 * j + o) as *const __m128i);
                _mm_shuffle_epi8(_mm_avg_epu8(x, y), split)
            };
            // [U0-3 V0-3] and [U4-7 V4-7] → [U0-7 V0-7]
            let w = _mm_unpacklo_epi32(m(0), m(16));
            _mm_storel_epi64(pu.add(j) as *mut __m128i, w);
            _mm_storel_epi64(pv.add(j) as *mut __m128i, _mm_srli_si128::<8>(w));
            j += 8;
        }
        n
    }

    #[target_feature(enable = "sse4.1")]
    pub unsafe fn packed422_chroma_avg_sse41(
        u: &mut [u8],
        v: &mut [u8],
        a: &[u8],
        b: &[u8],
        y_odd: bool,
    ) -> usize {
        packed422_chroma_avg_x86(u, v, a, b, y_odd)
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn packed422_chroma_avg_avx2(
        u: &mut [u8],
        v: &mut [u8],
        a: &[u8],
        b: &[u8],
        y_odd: bool,
    ) -> usize {
        packed422_chroma_avg_x86(u, v, a, b, y_odd)
    }

    #[target_feature(enable = "sse4.1")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn packed422_chroma_sse41(
        u: &mut [u8],
        v: &mut [u8],
        a: &[u16],
        b: &[u16],
        k: usize,
        shift: u32,
        y_odd: bool,
    ) -> usize {
        packed422_chroma_x86(u, v, a, b, k, shift, y_odd)
    }

    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn packed422_chroma_avx2(
        u: &mut [u8],
        v: &mut [u8],
        a: &[u16],
        b: &[u16],
        k: usize,
        shift: u32,
        y_odd: bool,
    ) -> usize {
        packed422_chroma_x86(u, v, a, b, k, shift, y_odd)
    }

    /// `[c0, c1, c2, c3]` repeated, as 16-bit lanes.
    #[inline(always)]
    unsafe fn coeffs128(c: [i32; 4]) -> __m128i {
        let c = c.map(|v| v as i16);
        _mm_setr_epi16(c[0], c[1], c[2], c[3], c[0], c[1], c[2], c[3])
    }

    /// 4 pixels (16 bytes) → 4 × i32 `Σ c·byte`: bytes widened, `madd`
    /// sums byte pairs, `hadd` the two pairs of a pixel.
    #[inline(always)]
    unsafe fn dot4(v: __m128i, c: __m128i) -> __m128i {
        let z = _mm_setzero_si128();
        let lo = _mm_madd_epi16(_mm_unpacklo_epi8(v, z), c);
        let hi = _mm_madd_epi16(_mm_unpackhi_epi8(v, z), c);
        _mm_hadd_epi32(lo, hi)
    }

    #[target_feature(enable = "ssse3")]
    pub unsafe fn pixels_map_ssse3(out: &mut [u8], src: &[u8], map: PixelMap) -> usize {
        let n = out.len().min(src.len() / 4) / 16 * 16;
        let c = coeffs128(map.c);
        let o = _mm_set1_epi32(map.o);
        let (sp, op) = (src.as_ptr(), out.as_mut_ptr());
        let mut i = 0;
        while i < n {
            let r: [__m128i; 4] = std::array::from_fn(|k| {
                let v = _mm_loadu_si128(sp.add((i + 4 * k) * 4) as *const __m128i);
                _mm_srai_epi32::<14>(_mm_add_epi32(dot4(v, c), o))
            });
            let w = _mm_packus_epi16(_mm_packs_epi32(r[0], r[1]), _mm_packs_epi32(r[2], r[3]));
            _mm_storeu_si128(op.add(i) as *mut __m128i, w);
            i += 16;
        }
        n
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn pixels_map_avx2(out: &mut [u8], src: &[u8], map: PixelMap) -> usize {
        let n = out.len().min(src.len() / 4) / 32 * 32;
        let c = _mm256_broadcastsi128_si256(coeffs128(map.c));
        let o = _mm256_set1_epi32(map.o);
        let z = _mm256_setzero_si256();
        // Undoes the in-lane packs: dwords 0 4 1 5 2 6 3 7.
        let order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        let (sp, op) = (src.as_ptr(), out.as_mut_ptr());
        let mut i = 0;
        while i < n {
            // In-lane unpacks keep the pixel order within each 8-pixel load.
            let r: [__m256i; 4] = std::array::from_fn(|k| {
                let v = _mm256_loadu_si256(sp.add((i + 8 * k) * 4) as *const __m256i);
                let lo = _mm256_madd_epi16(_mm256_unpacklo_epi8(v, z), c);
                let hi = _mm256_madd_epi16(_mm256_unpackhi_epi8(v, z), c);
                _mm256_srai_epi32::<14>(_mm256_add_epi32(_mm256_hadd_epi32(lo, hi), o))
            });
            let w = _mm256_packus_epi16(
                _mm256_packs_epi32(r[0], r[1]),
                _mm256_packs_epi32(r[2], r[3]),
            );
            let w = _mm256_permutevar8x32_epi32(w, order);
            _mm256_storeu_si256(op.add(i) as *mut __m256i, w);
            i += 32;
        }
        n
    }

    /// Per 4 pixels of rows `a` and `b` (16 bytes each): channel sums of
    /// the two 2×2 blocks → `[U01, U23, V01, V23]` (i32, before the shift).
    #[inline(always)]
    unsafe fn uv4(va: __m128i, vb: __m128i, cu: __m128i, cv: __m128i) -> __m128i {
        let z = _mm_setzero_si128();
        let lo = _mm_add_epi16(_mm_unpacklo_epi8(va, z), _mm_unpacklo_epi8(vb, z));
        let hi = _mm_add_epi16(_mm_unpackhi_epi8(va, z), _mm_unpackhi_epi8(vb, z));
        let s = _mm_add_epi16(_mm_unpacklo_epi64(lo, hi), _mm_unpackhi_epi64(lo, hi));
        _mm_hadd_epi32(_mm_madd_epi16(s, cu), _mm_madd_epi16(s, cv))
    }

    #[target_feature(enable = "ssse3")]
    pub unsafe fn pixels_map_420_ssse3(
        u: &mut [u8],
        v: &mut [u8],
        a: &[u8],
        b: &[u8],
        mu: PixelMap,
        mv: PixelMap,
    ) -> usize {
        let n = u.len().min(v.len()).min(a.len() / 8).min(b.len() / 8) / 8 * 8;
        let (cu, cv) = (coeffs128(mu.c), coeffs128(mv.c));
        let o = _mm_setr_epi32(mu.o, mu.o, mv.o, mv.o);
        // [U0 U1 V0 V1 U2 U3 V2 V3 ...] → [U0..U7 | V0..V7]
        let split = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);
        let mut i = 0;
        while i < n {
            let r: [__m128i; 4] = std::array::from_fn(|k| {
                let off = (2 * i + 4 * k) * 4;
                let va = _mm_loadu_si128(a.as_ptr().add(off) as *const __m128i);
                let vb = _mm_loadu_si128(b.as_ptr().add(off) as *const __m128i);
                _mm_srai_epi32::<16>(_mm_add_epi32(uv4(va, vb, cu, cv), o))
            });
            let w = _mm_packus_epi16(_mm_packs_epi32(r[0], r[1]), _mm_packs_epi32(r[2], r[3]));
            let w = _mm_shuffle_epi8(w, split);
            _mm_storel_epi64(u.as_mut_ptr().add(i) as *mut __m128i, w);
            _mm_storel_epi64(
                v.as_mut_ptr().add(i) as *mut __m128i,
                _mm_srli_si128::<8>(w),
            );
            i += 8;
        }
        n
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn pixels_map_420_avx2(
        u: &mut [u8],
        v: &mut [u8],
        a: &[u8],
        b: &[u8],
        mu: PixelMap,
        mv: PixelMap,
    ) -> usize {
        let n = u.len().min(v.len()).min(a.len() / 8).min(b.len() / 8) / 16 * 16;
        let cu = _mm256_broadcastsi128_si256(coeffs128(mu.c));
        let cv = _mm256_broadcastsi128_si256(coeffs128(mv.c));
        let o = _mm256_setr_epi32(mu.o, mu.o, mv.o, mv.o, mu.o, mu.o, mv.o, mv.o);
        let z = _mm256_setzero_si256();
        let split = _mm256_broadcastsi128_si256(_mm_setr_epi8(
            0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15,
        ));
        let mut i = 0;
        while i < n {
            // Load k (8 pixels): lane 0 gives outputs 4k, 4k+1; lane 1
            // 4k+2, 4k+3, as [U U V V].
            let r: [__m256i; 4] = std::array::from_fn(|k| {
                let off = (2 * i + 8 * k) * 4;
                let va = _mm256_loadu_si256(a.as_ptr().add(off) as *const __m256i);
                let vb = _mm256_loadu_si256(b.as_ptr().add(off) as *const __m256i);
                let lo = _mm256_add_epi16(_mm256_unpacklo_epi8(va, z), _mm256_unpacklo_epi8(vb, z));
                let hi = _mm256_add_epi16(_mm256_unpackhi_epi8(va, z), _mm256_unpackhi_epi8(vb, z));
                let s =
                    _mm256_add_epi16(_mm256_unpacklo_epi64(lo, hi), _mm256_unpackhi_epi64(lo, hi));
                let uv = _mm256_hadd_epi32(_mm256_madd_epi16(s, cu), _mm256_madd_epi16(s, cv));
                _mm256_srai_epi32::<16>(_mm256_add_epi32(uv, o))
            });
            // Lane 0: U0 U1 V0 V1 U4 U5 V4 V5 U8 U9 .. U12 U13 ..;
            // lane 1: the same for 2 3, 6 7, 10 11, 14 15.
            let w = _mm256_packus_epi16(
                _mm256_packs_epi32(r[0], r[1]),
                _mm256_packs_epi32(r[2], r[3]),
            );
            // Lane 0: U01 U45 U89 U1213 | V...; lane 1: U23 U67 ... | V...
            let w = _mm256_shuffle_epi8(w, split);
            // [U(lane 0), U(lane 1)] and [V(lane 0), V(lane 1)]
            let w = _mm256_permute4x64_epi64::<0b11_01_10_00>(w);
            let (us, vs) = (_mm256_castsi256_si128(w), _mm256_extracti128_si256::<1>(w));
            // Interleave the pairs of the two lanes: U01 U23 U45 ...
            let us = _mm_unpacklo_epi16(us, _mm_srli_si128::<8>(us));
            let vs = _mm_unpacklo_epi16(vs, _mm_srli_si128::<8>(vs));
            _mm_storeu_si128(u.as_mut_ptr().add(i) as *mut __m128i, us);
            _mm_storeu_si128(v.as_mut_ptr().add(i) as *mut __m128i, vs);
            i += 16;
        }
        n
    }
}

/// Blend premultiplied `p` with alpha `a` over `d`, layer opacity `o`.
#[inline(always)]
fn over(d: u8, p: u8, a: u8, o: u32) -> u8 {
    let o = o as u16;
    let (p, a) = if o == 255 {
        (p as u16, a as u16)
    } else {
        (div255_16(p as u16 * o), div255_16(a as u16 * o))
    };
    (p + div255_16(d as u16 * (255 - a))).min(255) as u8
}

/// `round(x / 255)` for `x` in `0..=65025`, in 16-bit lanes.
#[inline(always)]
pub fn div255_16(x: u16) -> u16 {
    let x = x + 128;
    (x + (x >> 8)) >> 8
}

/// What a 16-pixel block of alpha is.
#[inline(always)]
fn block_kind(a: &[u8]) -> u8 {
    let (mut and, mut or) = (0xFFu8, 0u8);
    for &v in a {
        and &= v;
        or |= v;
    }
    if or == 0 {
        0 // transparent
    } else if and == 0xFF {
        2 // opaque
    } else {
        1
    }
}

/// `d = p·o + d·(1 − a·o)` over a run (premultiplied `p`, alpha `a`,
/// opacity `o`), in 16-bit lanes so it vectorises.
#[inline(always)]
fn blend_run(d: &mut [u8], p: &[u8], a: &[u8], o: u8) {
    let n = d.len().min(p.len()).min(a.len());
    let (d, p, a) = (&mut d[..n], &p[..n], &a[..n]);
    if o == 255 {
        for i in 0..n {
            d[i] = (p[i] as u16 + div255_16(d[i] as u16 * (255 - a[i] as u16))).min(255) as u8;
        }
    } else {
        let o = o as u16;
        for i in 0..n {
            let (pp, aa) = (div255_16(p[i] as u16 * o), div255_16(a[i] as u16 * o));
            d[i] = (pp + div255_16(d[i] as u16 * (255 - aa))).min(255) as u8;
        }
    }
}

multiversion! {
    /// Write one channel into a canvas plane row: copy if opaque, else blend.
    pub fn write_plane(dst: &mut [u8], src: &[u8], alpha: Option<&[u8]>, opacity: u8) {
        let n = dst.len().min(src.len());
        let (dst, src) = (&mut dst[..n], &src[..n]);
        match alpha {
            None if opacity == 255 => dst.copy_from_slice(src),
            None => {
                let o = opacity as u16;
                let inv = 255 - o;
                for (d, &s) in dst.iter_mut().zip(src) {
                    *d = (div255_16(s as u16 * o) + div255_16(*d as u16 * inv)).min(255) as u8;
                }
            }
            Some(alpha) => {
                let alpha = &alpha[..n];
                let mut i = 0;
                while i < n {
                    let end = (i + 16).min(n);
                    match block_kind(&alpha[i..end]) {
                        0 => {}
                        2 if opacity == 255 => dst[i..end].copy_from_slice(&src[i..end]),
                        _ => blend_run(&mut dst[i..end], &src[i..end], &alpha[i..end], opacity),
                    }
                    i = end;
                }
            }
        }
    }
}

multiversion! {
    /// Write two channels into an interleaved canvas row (NV12 chroma).
    pub fn write_pairs(dst: &mut [u8], u: &[u8], v: &[u8], alpha: Option<&[u8]>, opacity: u8) {
        let n = (dst.len() / 2).min(u.len()).min(v.len());
        let (dst, u, v) = (&mut dst[..n * 2], &u[..n], &v[..n]);
        match alpha {
            None if opacity == 255 => {
                for ((d, &a), &b) in dst.as_chunks_mut::<2>().0.iter_mut().zip(u).zip(v) {
                    d[0] = a;
                    d[1] = b;
                }
            }
            _ => {
                let mut i = 0;
                while i < n {
                    let end = (i + 16).min(n);
                    let kind = alpha.map_or(2, |a| block_kind(&a[i..end]));
                    if kind != 0 {
                        let len = end - i;
                        let mut p = [0u8; 32];
                        let mut a2 = [255u8; 32];
                        for k in 0..len {
                            p[2 * k] = u[i + k];
                            p[2 * k + 1] = v[i + k];
                        }
                        if kind == 1 {
                            let alpha = &alpha.unwrap()[i..end];
                            for k in 0..len {
                                a2[2 * k] = alpha[k];
                                a2[2 * k + 1] = alpha[k];
                            }
                        }
                        blend_run(&mut dst[2 * i..2 * end], &p[..2 * len], &a2[..2 * len], opacity);
                    }
                    i = end;
                }
            }
        }
    }
}

multiversion! {
    /// Write premultiplied R, G, B (and alpha) into a premultiplied BGRA row.
    pub fn write_bgra(dst: &mut [u8], r: &[u8], g: &[u8], b: &[u8], alpha: Option<&[u8]>, opacity: u8) {
        let n = (dst.len() / 4).min(r.len()).min(g.len()).min(b.len());
        let dst = &mut dst[..n * 4];
        let (r, g, b) = (&r[..n], &g[..n], &b[..n]);
        let mut i = 0;
        while i < n {
            let end = (i + 16).min(n);
            let len = end - i;
            let kind = alpha.map_or(2, |a| block_kind(&a[i..end]));
            if kind == 0 {
                i = end;
                continue;
            }
            let mut p = [255u8; 64];
            for k in 0..len {
                p[4 * k] = b[i + k];
                p[4 * k + 1] = g[i + k];
                p[4 * k + 2] = r[i + k];
            }
            let d = &mut dst[4 * i..4 * end];
            if kind == 2 && opacity == 255 {
                d.copy_from_slice(&p[..4 * len]);
            } else {
                let mut a4 = [255u8; 64];
                if kind == 1 {
                    let alpha = &alpha.unwrap()[i..end];
                    for k in 0..len {
                        a4[4 * k..4 * k + 4].fill(alpha[k]);
                        p[4 * k + 3] = alpha[k];
                    }
                }
                blend_run(d, &p[..4 * len], &a4[..4 * len], opacity);
            }
            i = end;
        }
    }
}

multiversion! {
    /// Draw premultiplied BGRA (`alpha`) or BGRx pixels `src` over
    /// premultiplied BGRA `dst` at layer opacity `opacity`, 16 pixels at a
    /// time: fully transparent blocks are skipped, opaque ones copied, the
    /// rest blended (`d = p + d · (1 − a)`).
    pub fn over_bgra(dst: &mut [u8], src: &[u8], alpha: bool, opacity: u8) {
        let n = dst.len().min(src.len()) / 64 * 64;
        let (blocks, tail) = src[..src.len().min(dst.len())].split_at(n);
        let (dblocks, dtail) = dst.split_at_mut(n);
        let mask: u64 = 0xFF00_0000_FF00_0000;
        let mut p = [0u8; 64];
        for (s, d) in blocks.as_chunks::<64>().0.iter().zip(dblocks.as_chunks_mut::<64>().0) {
            if !alpha {
                p.copy_from_slice(s);
                for px in p.as_chunks_mut::<4>().0 {
                    px[3] = 255;
                }
                if opacity == 255 {
                    d.copy_from_slice(&p);
                } else {
                    over_pixels(d, &p, opacity);
                }
                continue;
            }
            let (mut and, mut or) = (u64::MAX, 0u64);
            for w in s.as_chunks::<8>().0 {
                let v = u64::from_le_bytes(*w);
                and &= v;
                or |= v;
            }
            if or & mask == 0 {
                continue; // transparent
            }
            if and & mask == mask && opacity == 255 {
                d.copy_from_slice(s);
                continue;
            }
            over_pixels(d, s, opacity);
        }
        if !dtail.is_empty() {
            let t = &tail[..dtail.len()];
            if alpha {
                over_pixels(dtail, t, opacity);
            } else {
                let mut p = [0u8; 64];
                p[..t.len()].copy_from_slice(t);
                for px in p[..t.len()].as_chunks_mut::<4>().0 {
                    px[3] = 255;
                }
                over_pixels(dtail, &p[..t.len()], opacity);
            }
        }
    }
}

/// Premultiplied BGRA `p` over `d`, with opacity.
#[inline(always)]
fn over_pixels(d: &mut [u8], p: &[u8], opacity: u8) {
    let o = opacity as u16;
    for (d, p) in d
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(p.as_chunks::<4>().0)
    {
        let pa = if o == 255 {
            p[3] as u16
        } else {
            div255_16(p[3] as u16 * o)
        };
        let inv = 255 - pa;
        for c in 0..4 {
            let pc = if o == 255 {
                p[c] as u16
            } else {
                div255_16(p[c] as u16 * o)
            };
            d[c] = (pc + div255_16(d[c] as u16 * inv)).min(255) as u8;
        }
    }
}

/// Fill `dst` with `value` at opacity `a` (straight colour).
pub fn fill_plane(dst: &mut [u8], value: u8, a: u8) {
    if a == 255 {
        dst.fill(value);
    } else {
        let p = div255(value as u32 * a as u32);
        let inv = 255 - a as u32;
        for d in dst {
            *d = (p + div255(*d as u32 * inv)).min(255) as u8;
        }
    }
}

/// Fill interleaved pairs with `(u, v)` at opacity `a`.
pub fn fill_pairs(dst: &mut [u8], u: u8, v: u8, a: u8) {
    if a == 255 {
        // Opaque (the background): plain stores of a 64-byte pattern, not
        // a blend per byte (a quarter of a sparse composition's frame).
        let mut pattern = [0u8; 64];
        for p in pattern.as_chunks_mut::<2>().0.iter_mut() {
            *p = [u, v];
        }
        let (blocks, tail) = dst.as_chunks_mut::<64>();
        for b in blocks {
            *b = pattern;
        }
        for p in tail.as_chunks_mut::<2>().0.iter_mut() {
            *p = [u, v];
        }
        return;
    }
    for d in dst.as_chunks_mut::<2>().0.iter_mut() {
        d[0] = over(d[0], div255(u as u32 * a as u32) as u8, a, 255);
        d[1] = over(d[1], div255(v as u32 * a as u32) as u8, a, 255);
    }
}

/// Fill premultiplied BGRA pixels with straight `[b, g, r]` at opacity `a`.
pub fn fill_bgra(dst: &mut [u8], bgr: [u8; 3], a: u8) {
    let p: [u8; 4] = [
        div255(bgr[0] as u32 * a as u32) as u8,
        div255(bgr[1] as u32 * a as u32) as u8,
        div255(bgr[2] as u32 * a as u32) as u8,
        a,
    ];
    if a == 255 {
        for d in dst.as_chunks_mut::<4>().0.iter_mut() {
            d.copy_from_slice(&p);
        }
    } else {
        for d in dst.as_chunks_mut::<4>().0.iter_mut() {
            for c in 0..4 {
                d[c] = over(d[c], p[c], a, 255);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fill_pairs_opaque_and_blended() {
        for len in [0usize, 2, 6, 64, 130, 3840] {
            let base: Vec<u8> = (0..len).map(|i| (i * 37 % 251) as u8).collect();
            for a in [0u8, 1, 128, 254, 255] {
                let mut fast = base.clone();
                fill_pairs(&mut fast, 90, 240, a);
                let want: Vec<u8> = base
                    .iter()
                    .enumerate()
                    .map(|(i, &d)| {
                        let c = if i % 2 == 0 { 90u32 } else { 240 };
                        over(d, div255(c * a as u32) as u8, a, 255)
                    })
                    .collect();
                assert_eq!(fast, want, "len {len}, a {a}");
            }
        }
    }

    #[test]
    fn vertical_sums() {
        let mut acc = vec![0u16; 40];
        let row: Vec<u8> = (0..40).map(|i| (i * 6) as u8).collect();
        vacc_first(&mut acc, &row, 3);
        vacc_add(&mut acc, &row, 1);
        assert!(acc.iter().zip(&row).all(|(&a, &r)| a == r as u16 * 4));
    }

    /// The horizontal boxes of the planar kernels against exact quotients
    /// (within [`HBox`]'s bound; exact for areas that are powers of two):
    /// widths 1 to 9 and some wide ones (SIMD for 1, 2, 4, 8), box and area
    /// filter sums, lengths with tails.
    #[test]
    fn hbox_kernels_match_plain_sums() {
        let mut seed = 3u32;
        let mut rnd = |m: u32| {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            (seed >> 8) % m
        };
        for k in (1..=9usize).chain([12, 16, 24, 32]) {
            // Box sums of 1, 2, 3, 5, 16 rows; area filters (weights adding
            // up to 256).
            for vtotal in [1u32, 2, 3, 5, 16, 256] {
                let hb = HBox::new(k, vtotal);
                let pre = |v: u16| -> u32 {
                    if hb.shift == 0 {
                        v as u32
                    } else {
                        (v as u32 + (1 << (hb.shift - 1))) >> hb.shift
                    }
                };
                let div = (k as u32 * vtotal) as f64 / (1u32 << hb.shift) as f64;
                assert!(
                    (hb.div() - div).abs() / div < 1e-4,
                    "k {k} v {vtotal}: div {}",
                    hb.div()
                );
                let bound = 0.5 + 0.5 / (1u32 << hb.extra_bits()) as f64 + 0.01;
                for n in (0..40).chain([67, 129]) {
                    let acc: Vec<u16> = (0..2 * k * n + 5)
                        .map(|_| rnd(255 * vtotal + 1) as u16)
                        .collect();
                    let check = |got: u8, s: u32, what: &str| {
                        let exact = (s as f64 / div).min(255.0);
                        assert!(
                            (got as f64 - exact).abs() <= bound,
                            "{what} k {k} v {vtotal} n {n}: {got} vs {exact}"
                        );
                        if hb.extra_bits() == 0 {
                            // Powers of two: exact (halves round up).
                            assert_eq!(got as f64, (exact + 0.5).floor().min(255.0), "{what}");
                        }
                    };
                    let mut out = vec![0u8; n];
                    hbox(&mut out, &acc, k, hb);
                    for j in 0..n {
                        check(out[j], (0..k).map(|t| pre(acc[j * k + t])).sum(), "hbox");
                        // SIMD lanes and the plain tail agree.
                        let mut one = [0u8; 1];
                        hbox(&mut one, &acc[j * k..(j + 1) * k], k, hb);
                        assert_eq!(out[j], one[0], "hbox lane k {k} j {j}");
                    }
                    let (mut u, mut v) = (vec![0u8; n], vec![0u8; n]);
                    hbox_pairs(&mut u, &mut v, &acc, k, hb);
                    for j in 0..n {
                        let (mut one, mut other) = ([0u8; 1], [0u8; 1]);
                        hbox_pairs(
                            &mut one,
                            &mut other,
                            &acc[2 * j * k..2 * (j + 1) * k],
                            k,
                            hb,
                        );
                        for (got, c, lane) in [(u[j], 0, one[0]), (v[j], 1, other[0])] {
                            let s = (0..k).map(|t| pre(acc[2 * (j * k + t) + c])).sum();
                            check(got, s, "pairs");
                            assert_eq!(got, lane, "pairs lane k {k} j {j} c {c}");
                        }
                    }
                }
            }
        }
    }

    /// The vertical sums against a plain sum: boxes and weights adding up
    /// to 256, 1 to 17 rows, every length up to a few blocks, rows longer
    /// than `acc` and the other way round.
    #[test]
    fn vacc_rows_match_plain_sums() {
        let mut seed = 7u32;
        let mut rnd = |m: u32| {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            (seed >> 8) % m
        };
        for count in 1..=17usize {
            let data: Vec<Vec<u8>> = (0..count)
                .map(|_| (0..140).map(|_| rnd(256) as u8).collect())
                .collect();
            let boxed = vec![1u16; count];
            // Weights adding up to 256, like an area filter's.
            let mut weighted: Vec<u16> =
                (0..count).map(|_| rnd(256 / count as u32) as u16).collect();
            weighted[0] += 256 - weighted.iter().sum::<u16>();
            for weights in [&boxed, &weighted] {
                for n in (0..100).chain([129, 140]) {
                    let rows: Vec<&[u8]> = data.iter().map(|r| &r[..n.min(r.len())]).collect();
                    let mut acc = vec![0xAAAAu16; n + 3];
                    vacc_rows(&mut acc[..n], &rows, weights);
                    for k in 0..n {
                        let want: u32 = (0..count)
                            .map(|t| rows[t][k] as u32 * weights[t] as u32)
                            .sum();
                        assert_eq!(acc[k] as u32, want, "count {count} n {n} k {k}");
                    }
                    assert!(acc[n..].iter().all(|&a| a == 0xAAAA), "wrote past n");
                }
            }
        }
    }

    #[test]
    fn extract_box_and_generic_agree() {
        let acc: Vec<u16> = (0..64).map(|i| (i * 37 % 255) as u16).collect();
        let boxed = Table::new(0.0, 32.0, 8, 32);
        assert_eq!(boxed.box_k, Some(4));
        let mut generic = boxed.clone();
        generic.box_k = None;
        for (step, base) in [(1usize, 0isize), (2, 1)] {
            let mut a = vec![0u16; 8];
            let mut b = vec![0u16; 8];
            let mut tmp = Vec::new();
            extract(&mut a, &acc, base, step, &boxed, 0, q6_scale(4), &mut tmp);
            extract(&mut b, &acc, base, step, &generic, 0, q6_scale(4), &mut tmp);
            let mut c = vec![0u16; 8];
            extract(&mut c, &acc, base, step, &boxed, 0, q6_scale(3), &mut tmp);
            assert_eq!(c[0], ((a[0] as f64) * 4.0 / 3.0).round() as u16);
            assert_eq!(a, b);
            let want = (acc[base as usize]
                + acc[base as usize + step]
                + acc[base as usize + 2 * step]
                + acc[base as usize + 3 * step]) as f64
                / 4.0;
            assert_eq!(a[0], (want * 64.0).round() as u16);
        }
    }

    /// The SIMD kernels against plain loops, for every length up to a few
    /// blocks (tails included).
    #[test]
    fn simd_kernels_match_plain_loops() {
        let mut seed = 1u32;
        let mut rnd = |m: u32| {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            (seed >> 8) % m
        };
        for n in 0..70 {
            for (k, kv) in [(2usize, 1u32), (2, 4), (4, 1), (4, 2), (4, 4), (4, 8)] {
                let shift = (k as u32 * kv).trailing_zeros();
                let a: Vec<u16> = (0..4 * k * n + 8)
                    .map(|_| rnd(255 * kv + 1) as u16)
                    .collect();
                let b: Vec<u16> = (0..4 * k * n + 8)
                    .map(|_| rnd(255 * kv + 1) as u16)
                    .collect();
                for y_odd in [false, true] {
                    let mut out = vec![0u8; n];
                    packed422_luma(&mut out, &a, k, shift, y_odd);
                    for j in 0..n {
                        let s: u32 = (0..k)
                            .map(|t| a[2 * (j * k + t) + y_odd as usize] as u32)
                            .sum();
                        let want = (s as f64 / (1 << shift) as f64 + 0.5).floor().min(255.0) as u8;
                        assert_eq!(out[j], want, "luma n={n} k={k} kv={kv} j={j}");
                    }
                    let (mut u, mut v) = (vec![0u8; n], vec![0u8; n]);
                    packed422_chroma(&mut u, &mut v, &a, &b, k, shift + 1, y_odd);
                    for j in 0..n {
                        for (got, o) in [(u[j], 0), (v[j], 2)] {
                            let s: u32 = (0..k)
                                .map(|t| {
                                    let i = 4 * (j * k + t) + !y_odd as usize + o;
                                    a[i] as u32 + b[i] as u32
                                })
                                .sum();
                            let want =
                                (s as f64 / (2 << shift) as f64 + 0.5).floor().min(255.0) as u8;
                            assert_eq!(got, want, "chroma n={n} k={k} kv={kv} j={j} o={o}");
                        }
                    }
                }
            }
            for y_odd in [false, true] {
                let ra: Vec<u8> = (0..4 * n + 4).map(|_| rnd(256) as u8).collect();
                let rb: Vec<u8> = (0..4 * n + 4).map(|_| rnd(256) as u8).collect();
                let (mut u, mut v) = (vec![0u8; n], vec![0u8; n]);
                packed422_chroma_avg(&mut u, &mut v, &ra, &rb, y_odd);
                for j in 0..n {
                    let i = 4 * j + !y_odd as usize;
                    assert_eq!(
                        u[j] as u32,
                        (ra[i] as u32 + rb[i] as u32).div_ceil(2),
                        "avg n={n} j={j}"
                    );
                    assert_eq!(
                        v[j] as u32,
                        (ra[i + 2] as u32 + rb[i + 2] as u32).div_ceil(2),
                        "avg n={n} j={j}"
                    );
                }
            }
            for (k, div) in [
                (2usize, 2u32),
                (2, 4),
                (4, 16),
                (4, 12),
                (3, 9),
                (2, 6),
                (5, 25),
                (6, 36),
                (8, 64),
            ] {
                let acc: Vec<u16> = (0..4 * k * n + 8)
                    .map(|_| rnd(255 * (div / k as u32) + 1) as u16)
                    .collect();
                let mut out = vec![0u8; 4 * n];
                box_reduce4(&mut out, &acc, k, div);
                for j in 0..n {
                    for c in 0..4 {
                        let s: u32 = (0..k).map(|t| acc[4 * (j * k + t) + c] as u32).sum();
                        let want = s as f64 / div as f64;
                        assert!(
                            (out[4 * j + c] as f64 - want).abs() <= 0.57,
                            "reduce n={n} k={k} div={div}"
                        );
                    }
                }
            }
            let t = Transform::new(
                super::super::color::Model::Rgb,
                super::super::color::Model::yuv(super::super::color::BT709),
            );
            let px: Vec<u8> = (0..8 * n + 8).map(|_| rnd(256) as u8).collect();
            let px2: Vec<u8> = (0..8 * n + 8).map(|_| rnd(256) as u8).collect();
            let bytes = [2, 1, 0];
            for k in 0..3 {
                let m = PixelMap::new(&t, k, bytes, 1);
                let mut out = vec![0u8; 2 * n];
                pixels_map(&mut out, &px, m);
                for i in 0..2 * n {
                    let p = &px[4 * i..];
                    let v: f64 = (0..3)
                        .map(|j| t.m[k][j] as f64 * p[bytes[j]] as f64)
                        .sum::<f64>()
                        + t.o[k] as f64;
                    assert!(
                        (out[i] as f64 - v.clamp(0.0, 255.0)).abs() <= 0.51,
                        "rgb n={n} k={k} i={i}"
                    );
                }
            }
            let (mu, mv) = (
                PixelMap::new(&t, 1, bytes, 4),
                PixelMap::new(&t, 2, bytes, 4),
            );
            let (mut u, mut v) = (vec![0u8; n], vec![0u8; n]);
            pixels_map_420(&mut u, &mut v, &px, &px2, mu, mv);
            for i in 0..n {
                for (got, k) in [(u[i], 1), (v[i], 2)] {
                    let c = |j: usize| {
                        let o = 8 * i + bytes[j];
                        (px[o] as f64 + px[o + 4] as f64 + px2[o] as f64 + px2[o + 4] as f64) / 4.0
                    };
                    let want: f64 =
                        (0..3).map(|j| t.m[k][j] as f64 * c(j)).sum::<f64>() + t.o[k] as f64;
                    assert!(
                        (got as f64 - want.clamp(0.0, 255.0)).abs() <= 0.51,
                        "rgb 420 n={n} i={i}"
                    );
                }
            }
        }
    }

    #[test]
    fn writes_blend_and_skip() {
        let mut dst = vec![100u8; 40];
        let src = [50u8; 40];
        let mut alpha = vec![0u8; 40];
        alpha[16..32].fill(255);
        alpha[32..].fill(128);
        let premult: Vec<u8> = src
            .iter()
            .zip(&alpha)
            .map(|(&s, &a)| div255(s as u32 * a as u32) as u8)
            .collect();
        write_plane(&mut dst, &premult, Some(&alpha), 255);
        assert_eq!(dst[0], 100);
        assert_eq!(dst[20], 50);
        assert_eq!(dst[35], 75);
        let mut dst = vec![100u8; 4];
        write_plane(&mut dst, &[200; 4], None, 128);
        assert_eq!(dst[0], 150);
    }

    #[test]
    fn transform_rounds_and_clamps() {
        let t = Transform {
            m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            o: [10.0, 0.0, 0.0],
            identity: false,
        };
        let c = vec![64u16 * 250; 3];
        let mut out = vec![0u8; 3];
        transform_row(&mut out, &t, 0, [&c, &c, &c], None);
        assert_eq!(out[0], 255);
        let a = vec![64u16 * 100; 3];
        let c = vec![64u16 * 50; 3];
        transform_row(&mut out, &t, 0, [&c, &c, &c], Some(&a));
        // 50 + 10 · 100/255 = 53.9
        assert_eq!(out[0], 54);
    }
}

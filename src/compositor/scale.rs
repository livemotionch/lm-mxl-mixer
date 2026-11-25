//! Filter tables: which source samples, with which integer weights, make
//! each output sample along one axis.
//!
//! - Downscaling uses an area filter: an output sample averages the source
//!   interval it covers, with fractional weights at the edges. For integer
//!   ratios (2:1, 4:1, ...) it's an exact box with weights of 1.
//! - Upscaling, and downscaling by less than 2:1, is bilinear (sample
//!   centres aligned).
//!
//! Weights are integers that add up to the same `total` for every output
//! (the box size, or 256), so a sum over a table never needs a division
//! until the very end.

/// Weights for one axis.
#[derive(Debug, Clone, PartialEq)]
pub struct Table {
    /// Number of outputs
    pub len: usize,
    /// Weights per output (zero-padded)
    pub taps: usize,
    /// Sum of the weights of every output (≤ 256, or the `max_total` of
    /// [`Table::with_total`])
    pub total: u32,
    /// First source index of each output
    pub start: Vec<u32>,
    /// `len × taps` weights
    pub weight: Vec<u16>,
    /// Every output is `k` consecutive samples with weight 1, starting at
    /// `start[0] + j·k`.
    pub box_k: Option<u32>,
}

const MAX_TOTAL: u32 = 256;

impl Table {
    /// Map the source interval `[a, b)` (in samples; fractional for chroma
    /// of an odd crop) onto `n` outputs. Source indices are clamped to
    /// `0..limit`.
    pub fn new(a: f64, b: f64, n: usize, limit: usize) -> Table {
        Table::with_total(a, b, n, limit, MAX_TOTAL)
    }

    /// [`Table::new`] with fractional weights quantised to `max_total`
    /// instead of 256 (the 10-bit renderer uses 1024: 1/256 steps are
    /// visible at 10 bits).
    pub fn with_total(a: f64, b: f64, n: usize, limit: usize, max_total: u32) -> Table {
        if n == 0 || limit == 0 || b <= a {
            return Table {
                len: 0,
                taps: 1,
                total: 1,
                start: Vec::new(),
                weight: Vec::new(),
                box_k: None,
            };
        }
        let s = (b - a) / n as f64;
        let last = (limit - 1) as i64;

        // Exact box: integer ratio, integer start, all inside the source.
        let k = s.round();
        if (s - k).abs() < 1e-9
            && k >= 1.0
            && k <= max_total as f64
            && (a - a.round()).abs() < 1e-9
            && a >= 0.0
            && a.round() as i64 + (n as i64) * (k as i64) - 1 <= last
        {
            let (k, a) = (k as u32, a.round() as u32);
            return Table {
                len: n,
                taps: k as usize,
                total: k,
                start: (0..n as u32).map(|j| a + j * k).collect(),
                weight: vec![1; n * k as usize],
                box_k: Some(k),
            };
        }

        // Real-valued weights per output, as (index, weight) runs.
        let mut rows: Vec<(i64, Vec<f64>)> = Vec::with_capacity(n);
        for j in 0..n {
            let mut w: Vec<(i64, f64)> = Vec::new();
            // Area filter from 2:1 down; above, bilinear (as the legacy
            // pipeline's single `videoscale` step): every source sample
            // still counts, and the passband stays sharper than with an
            // area filter of less than two samples (zone plate, 1.33:1:
            // 30.7 against 28.9 dB to a Lanczos-3 reference).
            if s >= 2.0 - 1e-9 {
                let lo = a + j as f64 * s;
                let hi = lo + s;
                let mut i = lo.floor() as i64;
                while (i as f64) < hi {
                    let overlap = hi.min(i as f64 + 1.0) - lo.max(i as f64);
                    if overlap > 1e-12 {
                        w.push((i.clamp(0, last), overlap));
                    }
                    i += 1;
                }
            } else {
                let lo_i = a.floor().max(0.0) as i64;
                let hi_i = ((b.ceil() as i64) - 1).min(last).max(lo_i);
                let p = a + (j as f64 + 0.5) * s - 0.5;
                let i0 = p.floor();
                let f = p - i0;
                let i0 = i0 as i64;
                w.push((i0.clamp(lo_i, hi_i), 1.0 - f));
                w.push(((i0 + 1).clamp(lo_i, hi_i), f));
            }
            // Merge clamped duplicates (indices are non-decreasing).
            let first = w[0].0;
            let span = (w.last().unwrap().0 - first + 1) as usize;
            let mut dense = vec![0.0; span];
            for (i, v) in w {
                dense[(i - first) as usize] += v;
            }
            rows.push((first, dense));
        }

        // Quantize to integers summing to max_total (largest remainder).
        let mut quantized: Vec<(i64, Vec<u16>)> = Vec::with_capacity(n);
        for (first, dense) in rows {
            let sum: f64 = dense.iter().sum();
            let exact: Vec<f64> = dense.iter().map(|v| v / sum * max_total as f64).collect();
            let mut q: Vec<u16> = exact.iter().map(|v| v.floor() as u16).collect();
            let mut missing = max_total as i64 - q.iter().map(|&v| v as i64).sum::<i64>();
            let mut order: Vec<usize> = (0..q.len()).collect();
            order.sort_by(|&x, &y| {
                let fx = exact[x] - exact[x].floor();
                let fy = exact[y] - exact[y].floor();
                fy.partial_cmp(&fx).unwrap()
            });
            let mut idx = 0;
            while missing > 0 {
                q[order[idx % order.len()]] += 1;
                missing -= 1;
                idx += 1;
            }
            // Trim zero weights at both ends.
            let lead = q.iter().take_while(|&&v| v == 0).count();
            let trail = q.iter().rev().take_while(|&&v| v == 0).count();
            let q = q[lead..q.len() - trail].to_vec();
            quantized.push((first + lead as i64, q));
        }
        let taps = quantized.iter().map(|(_, q)| q.len()).max().unwrap_or(1);
        let mut start = Vec::with_capacity(n);
        let mut weight = Vec::with_capacity(n * taps);
        for (first, q) in quantized {
            // Padding weights past the end of the source are zero; readers
            // keep `taps` samples of slack after the last used one.
            start.push(first as u32);
            weight.extend_from_slice(&q);
            weight.extend(std::iter::repeat_n(0, taps - q.len()));
        }
        Table {
            len: n,
            taps,
            total: max_total,
            start,
            weight,
            box_k: None,
        }
    }

    /// Source indices read for outputs `j0..j1`, including zero-weight
    /// padding (which may reach past the end of the source).
    pub fn source_range(&self, j0: usize, j1: usize) -> (usize, usize) {
        if j0 >= j1 {
            return (0, 0);
        }
        // Starts never decrease.
        (
            self.start[j0] as usize,
            self.start[j1 - 1] as usize + self.taps,
        )
    }

    /// Weights of output `j`.
    #[inline]
    pub fn weights(&self, j: usize) -> &[u16] {
        &self.weight[j * self.taps..(j + 1) * self.taps]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_sums(t: &Table) {
        for j in 0..t.len {
            let sum: u32 = t.weights(j).iter().map(|&w| w as u32).sum();
            assert_eq!(sum, t.total, "output {j}");
        }
    }

    #[test]
    fn integer_ratio_is_a_box() {
        let t = Table::new(0.0, 1920.0, 480, 1920);
        assert_eq!(t.box_k, Some(4));
        assert_eq!(t.total, 4);
        assert_eq!(t.start[1], 4);
        let t = Table::new(0.0, 1080.0, 1080, 1080);
        assert_eq!(t.box_k, Some(1));
        let t = Table::new(10.0, 30.0, 10, 40);
        assert_eq!(t.box_k, Some(2));
        assert_eq!(t.start[0], 10);
    }

    #[test]
    fn fractional_area_weights() {
        // 5 → 2: each output covers 2.5 samples: (1, 1, 0.5), (0.5, 1, 1).
        let t = Table::new(0.0, 5.0, 2, 5);
        assert_eq!(t.box_k, None);
        check_sums(&t);
        assert_eq!(t.start, vec![0, 2]);
        // (102.4, 102.4, 51.2) rounded to integers summing to 256
        for (w, want) in t.weights(0)[..3].iter().zip([102.4, 102.4, 51.2]) {
            assert!((*w as f64 - want).abs() < 1.0);
        }
        for (w, want) in t.weights(1)[..3].iter().zip([51.2, 102.4, 102.4]) {
            assert!((*w as f64 - want).abs() < 1.0);
        }
        // 3 → 2 (less than 2:1) is bilinear: centres at 0.25 and 1.75.
        let t = Table::new(0.0, 3.0, 2, 3);
        check_sums(&t);
        assert_eq!(t.start, vec![0, 1]);
        assert_eq!(&t.weights(0)[..2], &[192, 64]);
        assert_eq!(&t.weights(1)[..2], &[64, 192]);
        // Odd chroma crop: half-sample start.
        let t = Table::new(0.5, 10.5, 5, 11);
        check_sums(&t);
        assert_eq!(t.start[0], 0);
    }

    #[test]
    fn upscale_is_bilinear_and_clamped() {
        let t = Table::new(0.0, 2.0, 4, 2);
        check_sums(&t);
        // Centres at -0.25, 0.25, 0.75, 1.25 → clamped at both ends.
        assert_eq!(t.weights(0).iter().map(|&w| w as u32).sum::<u32>(), 256);
        assert_eq!(t.start[0], 0);
    }

    #[test]
    fn windows_stay_inside_the_source() {
        for (a, b, n, limit) in [
            (0.0, 1920.0, 1000, 1920),
            (0.0, 1080.0, 1079, 1080),
            (3.0, 17.0, 5, 17),
            (0.0, 5.0, 17, 5),
            (0.0, 960.0, 241, 960),
        ] {
            let t = Table::new(a, b, n, limit);
            check_sums(&t);
            assert!(t.start.windows(2).all(|w| w[0] <= w[1]));
            for j in 0..t.len {
                let used = t.weights(j).iter().rposition(|&w| w != 0).unwrap();
                assert!(
                    t.start[j] as usize + used < limit,
                    "{a} {b} {n} {limit} j={j}"
                );
            }
        }
    }
}

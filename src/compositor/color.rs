//! Colour models and the affine transform from a source's samples to the
//! canvas's.
//!
//! Every conversion the mixer does (RGB → YUV, YUV → RGB, bt601 → bt709,
//! full → limited range) is one affine map on three channels, applied after
//! filtering, so it runs once per output pixel, not per source pixel.
//! Filtering commutes with an affine map, so the result is the same as
//! converting first (up to rounding).

/// How a source or the canvas encodes colour, in 8-bit code values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Model {
    /// Y'CbCr with the matrix's `kr`/`kb`; limited range (16–235/240) or full.
    Yuv { kr: f64, kb: f64, full_range: bool },
    /// R'G'B', full range.
    Rgb,
}

pub const BT709: (f64, f64) = (0.2126, 0.0722);
pub const BT601: (f64, f64) = (0.299, 0.114);
pub const BT2020: (f64, f64) = (0.2627, 0.0593);

impl Model {
    pub fn yuv((kr, kb): (f64, f64)) -> Model {
        Model::Yuv {
            kr,
            kb,
            full_range: false,
        }
    }

    /// 4×4 affine matrix from this model's code values (c0, c1, c2, 1) to
    /// normalized R'G'B' (0–1).
    fn decode(self) -> [[f64; 4]; 4] {
        match self {
            Model::Rgb => {
                let s = 1.0 / 255.0;
                [
                    [s, 0.0, 0.0, 0.0],
                    [0.0, s, 0.0, 0.0],
                    [0.0, 0.0, s, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            }
            Model::Yuv { kr, kb, full_range } => {
                let (ys, yo, cs) = if full_range {
                    (1.0 / 255.0, 0.0, 1.0 / 255.0)
                } else {
                    (1.0 / 219.0, -16.0 / 219.0, 1.0 / 224.0)
                };
                let kg = 1.0 - kr - kb;
                // y = ys*Y + yo; cb = cs*(U-128); cr = cs*(V-128)
                // R = y + 2(1-kr) cr; B = y + 2(1-kb) cb; G = (y - kr R - kb B)/kg
                let rcr = 2.0 * (1.0 - kr);
                let bcb = 2.0 * (1.0 - kb);
                let r = [ys, 0.0, rcr * cs, yo - rcr * cs * 128.0];
                let b = [ys, bcb * cs, 0.0, yo - bcb * cs * 128.0];
                let y = [ys, 0.0, 0.0, yo];
                let g: [f64; 4] = std::array::from_fn(|i| (y[i] - kr * r[i] - kb * b[i]) / kg);
                [r, g, b, [0.0, 0.0, 0.0, 1.0]]
            }
        }
    }

    /// 4×4 affine matrix from normalized R'G'B' to this model's code values.
    fn encode(self) -> [[f64; 4]; 4] {
        match self {
            Model::Rgb => [
                [255.0, 0.0, 0.0, 0.0],
                [0.0, 255.0, 0.0, 0.0],
                [0.0, 0.0, 255.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            Model::Yuv { kr, kb, full_range } => {
                let kg = 1.0 - kr - kb;
                let (ys, yo, cs) = if full_range {
                    (255.0, 0.0, 255.0)
                } else {
                    (219.0, 16.0, 224.0)
                };
                let y = [kr, kg, kb];
                // cb = (B - y) / (2(1-kb)), cr = (R - y) / (2(1-kr))
                let cb: [f64; 3] =
                    std::array::from_fn(|i| ([0.0, 0.0, 1.0][i] - y[i]) / (2.0 * (1.0 - kb)));
                let cr: [f64; 3] =
                    std::array::from_fn(|i| ([1.0, 0.0, 0.0][i] - y[i]) / (2.0 * (1.0 - kr)));
                [
                    [ys * y[0], ys * y[1], ys * y[2], yo],
                    [cs * cb[0], cs * cb[1], cs * cb[2], 128.0],
                    [cs * cr[0], cs * cr[1], cs * cr[2], 128.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            }
        }
    }
}

fn mul(a: &[[f64; 4]; 4], b: &[[f64; 4]; 4]) -> [[f64; 4]; 4] {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..4).map(|k| a[i][k] * b[k][j]).sum()))
}

/// `out_k = Σ_j m[k][j] · c_j + o[k] · a`, with `c` premultiplied by the
/// alpha `a` (0–1; 1 for opaque sources), so the same map works for
/// premultiplied samples.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform {
    pub m: [[f32; 3]; 3],
    pub o: [f32; 3],
    /// `m` is the identity and `o` zero: samples are copied.
    pub identity: bool,
}

impl Transform {
    pub fn new(source: Model, canvas: Model) -> Transform {
        let t = mul(&canvas.encode(), &source.decode());
        let mut m = [[0f32; 3]; 3];
        let mut o = [0f32; 3];
        let mut identity = true;
        for k in 0..3 {
            for j in 0..3 {
                let v = t[k][j];
                let want = if k == j { 1.0 } else { 0.0 };
                if (v - want).abs() > 1e-6 {
                    identity = false;
                }
                m[k][j] = if v.abs() < 1e-9 { 0.0 } else { v as f32 };
            }
            if t[k][3].abs() > 1e-6 {
                identity = false;
            }
            o[k] = if t[k][3].abs() < 1e-9 {
                0.0
            } else {
                t[k][3] as f32
            };
        }
        Transform { m, o, identity }
    }

    /// Whether output channel `k` depends on input channel `j`.
    pub fn uses(&self, k: usize, j: usize) -> bool {
        if self.identity {
            k == j
        } else {
            self.m[k][j] != 0.0
        }
    }
}

/// An ARGB colour (`0xAARRGGBB`, alpha ignored) in the canvas's code values.
pub fn argb_to(model: Model, argb: u32) -> [u8; 3] {
    let rgb = [
        ((argb >> 16) & 0xFF) as f64,
        ((argb >> 8) & 0xFF) as f64,
        (argb & 0xFF) as f64,
    ];
    let t = mul(&model.encode(), &Model::Rgb.decode());
    std::array::from_fn(|k| {
        let v = t[k][0] * rgb[0] + t[k][1] * rgb[1] + t[k][2] * rgb[2] + t[k][3];
        v.round().clamp(0.0, 255.0) as u8
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apply(t: &Transform, c: [f64; 3]) -> [f64; 3] {
        std::array::from_fn(|k| {
            (0..3).map(|j| t.m[k][j] as f64 * c[j]).sum::<f64>() + t.o[k] as f64
        })
    }

    #[test]
    fn identity_is_detected() {
        assert!(Transform::new(Model::yuv(BT709), Model::yuv(BT709)).identity);
        assert!(Transform::new(Model::Rgb, Model::Rgb).identity);
        assert!(!Transform::new(Model::yuv(BT601), Model::yuv(BT709)).identity);
        assert!(!Transform::new(Model::Rgb, Model::yuv(BT709)).identity);
    }

    #[test]
    fn yuv_to_rgb_is_exact_at_the_extremes() {
        let t = Transform::new(Model::yuv(BT709), Model::Rgb);
        let white = apply(&t, [235.0, 128.0, 128.0]);
        assert!(white.iter().all(|v| (v - 255.0).abs() < 1e-3), "{white:?}");
        let black = apply(&t, [16.0, 128.0, 128.0]);
        assert!(black.iter().all(|v| v.abs() < 1e-3), "{black:?}");
    }

    #[test]
    fn rgb_to_yuv_round_trips() {
        let to = Transform::new(Model::Rgb, Model::yuv(BT709));
        let back = Transform::new(Model::yuv(BT709), Model::Rgb);
        for c in [[255.0, 0.0, 0.0], [0.0, 255.0, 0.0], [12.0, 200.0, 99.0]] {
            let rgb = apply(&back, apply(&to, c));
            assert!(rgb.iter().zip(c).all(|(a, b)| (a - b).abs() < 1e-3));
        }
        assert_eq!(argb_to(Model::yuv(BT709), 0xFFFF_0000), [63, 102, 240]);
        assert_eq!(argb_to(Model::yuv(BT709), 0xFFFF_FFFF), [235, 128, 128]);
    }

    #[test]
    fn matrix_change_keeps_chroma_independent_of_luma() {
        let t = Transform::new(Model::yuv(BT601), Model::yuv(BT709));
        assert!(!t.uses(1, 0) && !t.uses(2, 0));
        assert!(t.uses(0, 1) && t.uses(0, 2));
        // Grey stays grey.
        let grey = apply(&t, [100.0, 128.0, 128.0]);
        assert!((grey[0] - 100.0).abs() < 1e-3 && (grey[1] - 128.0).abs() < 1e-3);
    }
}

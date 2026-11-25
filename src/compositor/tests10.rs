//! The 10-bit renderer against a slow reference: every layer drawn whole,
//! bottom-up, with floating-point filter weights, straight from the
//! definitions (no plan, no occlusion, no word tricks).

// Index loops over several slices at once are intentional here.
#![allow(clippy::needless_range_loop)]

use super::config::{BorderPosition, CompositionLayer, LayerBorder, LayerCrop, LayerFit, PixelRect};
use super::kernels10::{argb_to_ycbcr10, KeyShape, BLACK10, ONE};
use super::layout::crop_rect;
use super::mixer10::{Layer10, Mixer10};
use super::pool::StripPool;
use super::render10::{Keying, Source10, WipeMask, WipePattern};
use super::testing::Lcg;
use super::v210;

/// A v210 (or v210a) frame with known samples.
struct Frame {
    width: usize,
    height: usize,
    /// UYVY16 samples per row
    samples: Vec<Vec<u16>>,
    key: Option<Vec<Vec<u16>>>,
    bytes: Vec<u8>,
}

impl Frame {
    fn random(width: usize, height: usize, with_key: bool, rng: &mut Lcg) -> Frame {
        let mut smooth = |n: usize, lo: u32, hi: u32| -> Vec<u16> {
            // Random walks: pictures with edges and flat parts.
            let mut v = lo + rng.next_u32() % (hi - lo);
            (0..n)
                .map(|_| {
                    if rng.next_u32() % 8 == 0 {
                        v = lo + rng.next_u32() % (hi - lo);
                    }
                    v as u16
                })
                .collect()
        };
        let samples: Vec<Vec<u16>> = (0..height).map(|_| smooth(2 * width, 64, 960)).collect();
        let key = with_key.then(|| (0..height).map(|_| smooth(width, 0, 1023)).collect::<Vec<_>>());
        Frame::from_samples(width, height, samples, key)
    }

    fn from_samples(
        width: usize,
        height: usize,
        samples: Vec<Vec<u16>>,
        key: Option<Vec<Vec<u16>>>,
    ) -> Frame {
        let mut bytes = vec![0u8; v210::frame_bytes(width, height, key.is_some())];
        let stride = v210::line_bytes(width);
        for y in 0..height {
            v210::pack(&mut bytes[y * stride..(y + 1) * stride], 0, &samples[y]);
        }
        if let Some(key) = &key {
            let ks = v210::key_line_bytes(width);
            let base = stride * height;
            for y in 0..height {
                let row = &mut bytes[base + y * ks..base + (y + 1) * ks];
                for x in 0..width {
                    v210::set_sample(row, x, key[y][x]);
                }
            }
        }
        Frame {
            width,
            height,
            samples,
            key,
            bytes,
        }
    }

    fn source(&self) -> Source10<'_> {
        Source10::new(&self.bytes, self.width, self.height, self.key.is_some()).unwrap()
    }
}

fn read_canvas(bytes: &[u8], width: usize, height: usize) -> Vec<Vec<u16>> {
    let stride = v210::line_bytes(width);
    (0..height)
        .map(|y| {
            let mut r = vec![0u16; 2 * width];
            v210::unpack(&bytes[y * stride..], 0, &mut r);
            r
        })
        .collect()
}

/// Float weights of output `j` of `[a, b)` → `n` (the tables' definition,
/// without quantization).
fn ref_weights(a: f64, b: f64, n: usize, limit: usize, j: usize) -> Vec<(usize, f64)> {
    let s = (b - a) / n as f64;
    let last = limit as i64 - 1;
    let mut w = Vec::new();
    if s >= 2.0 - 1e-9 {
        let lo = a + j as f64 * s;
        let hi = lo + s;
        let mut i = lo.floor() as i64;
        while (i as f64) < hi {
            let o = hi.min(i as f64 + 1.0) - lo.max(i as f64);
            if o > 1e-12 {
                w.push((i.clamp(0, last) as usize, o / s));
            }
            i += 1;
        }
    } else {
        let lo_i = a.floor().max(0.0) as i64;
        let hi_i = ((b.ceil() as i64) - 1).min(last).max(lo_i);
        let p = a + (j as f64 + 0.5) * s - 0.5;
        let f = p - p.floor();
        let i0 = p.floor() as i64;
        w.push((i0.clamp(lo_i, hi_i) as usize, 1.0 - f));
        w.push(((i0 + 1).clamp(lo_i, hi_i) as usize, f));
    }
    w
}

fn snap_x(r: PixelRect) -> PixelRect {
    let round = |v: i32| v + v.rem_euclid(2);
    let (x0, x1) = (round(r.x), round(r.x + r.width));
    PixelRect {
        x: x0,
        width: x1 - x0,
        ..r
    }
}

fn intersect(a: &PixelRect, b: &PixelRect) -> PixelRect {
    let x0 = a.x.max(b.x);
    let y0 = a.y.max(b.y);
    let x1 = (a.x + a.width).min(b.x + b.width);
    let y1 = (a.y + a.height).min(b.y + b.height);
    PixelRect {
        x: x0,
        y: y0,
        width: (x1 - x0).max(0),
        height: (y1 - y0).max(0),
    }
}

/// Mask level (the mask's definition is `WipeMask::level`).
fn mask_level(m: &WipeMask, x: usize, y: usize, w: usize, h: usize) -> f64 {
    m.level(x, y, w, h) as f64 / ONE as f64
}

fn key_level(shape: &KeyShape, v: f64) -> f64 {
    shape.level(v.round().clamp(0.0, 1023.0) as u16) as f64 / ONE as f64
}

/// The reference: every layer whole, bottom-up, rounded after each layer.
fn reference(
    layers: &[(Layer10, Option<&Frame>)],
    width: usize,
    height: usize,
) -> Vec<Vec<u16>> {
    let mut canvas: Vec<Vec<u16>> = (0..height)
        .map(|_| {
            (0..2 * width)
                .map(|s| match s % 4 {
                    0 | 2 => BLACK10[1],
                    _ => BLACK10[0],
                })
                .collect()
        })
        .collect();
    let canvas_rect = PixelRect {
        x: 0,
        y: 0,
        width: width as i32,
        height: height as i32,
    };
    for (l, frame) in layers {
        if !l.layer.visible || l.layer.alpha <= 0.0 {
            continue;
        }
        let opacity = (l.layer.alpha.clamp(0.0, 1.0) * 255.0).round() / 255.0;
        // Picture rectangle and the source's samples per canvas pixel.
        let (picture, fitted) = if let Some(argb) = l.fill {
            let rect = snap_x(l.layer.pixel_rect(width as u32, height as u32));
            let c = argb_to_ycbcr10(argb);
            let a = ((argb >> 24) as f64 * l.layer.alpha.clamp(0.0, 1.0)).round() / 255.0;
            let mut area = intersect(&rect, &canvas_rect);
            if let Some(clip) = &l.clip {
                area = intersect(&area, &snap_x(*clip));
            }
            for y in area.y..area.y + area.height {
                for x in area.x..area.x + area.width {
                    let (x, y) = (x as usize, y as usize);
                    let row = &mut canvas[y];
                    let chroma = if x % 2 == 0 { 3 } else { 1 };
                    for (s, v) in [(2 * x + 1, c[0]), (4 * (x / 2), c[1]), (4 * (x / 2) + 2, c[2])].into_iter().take(chroma) {
                        let d = row[s] as f64;
                        row[s] = (d + (v as f64 - d) * a).round() as u16;
                    }
                }
            }
            (rect, l.layer.clone())
        } else {
            let frame = frame.expect("frame");
            let (x0, y0, x1, y1) = crop_rect(&l.layer.crop, frame.width as u32, frame.height as u32);
            let aspect = (x1 - x0) as f64 / (y1 - y0) as f64;
            let fitted = super::config::picture_box(&l.layer, width as u32, height as u32, Some(aspect));
            let picture = snap_x(fitted.pixel_rect(width as u32, height as u32));
            let (pw, ph) = (picture.width as usize, picture.height as usize);
            let unscaled = (x1 - x0) as usize == pw && (y1 - y0) as usize == ph && x0 % 2 == 0;
            let mut area = intersect(&picture, &canvas_rect);
            if let Some(clip) = &l.clip {
                area = intersect(&area, &snap_x(*clip));
            }
            // Filtered samples of a picture pixel pair (Cb, Y0, Cr, Y1) and
            // the key of each pixel.
            let filtered = |px: usize, py: usize| -> ([f64; 4], [f64; 2]) {
                if unscaled {
                    let sy = y0 as usize + py;
                    let sx = x0 as usize + px;
                    let r = &frame.samples[sy];
                    let k = frame.key.as_ref().map(|k| [k[sy][sx] as f64, k[sy][sx + 1] as f64]);
                    return (
                        [r[2 * sx] as f64, r[2 * sx + 1] as f64, r[2 * sx + 2] as f64, r[2 * sx + 3] as f64],
                        k.unwrap_or([940.0, 940.0]),
                    );
                }
                let wy = ref_weights(y0 as f64, y1 as f64, ph, frame.height, py);
                let luma = |ox: usize| -> (f64, f64) {
                    let wx = ref_weights(x0 as f64, x1 as f64, pw, frame.width, ox);
                    let (mut v, mut k) = (0.0, 0.0);
                    for &(sy, a) in &wy {
                        for &(sx, b) in &wx {
                            v += a * b * frame.samples[sy][2 * sx + 1] as f64;
                            k += a * b * frame.key.as_ref().map_or(940.0, |kk| kk[sy][sx] as f64);
                        }
                    }
                    (v, k)
                };
                let wc = ref_weights(x0 as f64 / 2.0, x1 as f64 / 2.0, pw / 2, frame.width.div_ceil(2), px / 2);
                let (mut cb, mut cr) = (0.0, 0.0);
                for &(sy, a) in &wy {
                    for &(sq, b) in &wc {
                        cb += a * b * frame.samples[sy][4 * sq] as f64;
                        cr += a * b * frame.samples[sy][4 * sq + 2] as f64;
                    }
                }
                let (l0, k0) = luma(px);
                let (l1, k1) = luma(px + 1);
                ([cb, l0, cr, l1], [k0, k1])
            };
            for y in area.y..area.y + area.height {
                let mut x = area.x;
                while x < area.x + area.width {
                    let (px, py) = ((x - picture.x) as usize, (y - picture.y) as usize);
                    let (v, k) = filtered(px, py);
                    let a: [f64; 2] = std::array::from_fn(|i| {
                        let key = match &l.keying {
                            Keying::None => 1.0,
                            Keying::Luma(shape) => key_level(shape, v[1 + 2 * i].round()),
                            Keying::Linear(shape) => {
                                if frame.key.is_some() {
                                    key_level(shape, k[i].round())
                                } else {
                                    1.0
                                }
                            }
                        };
                        let m = l
                            .mask
                            .as_ref()
                            .map_or(1.0, |m| mask_level(m, x as usize + i, y as usize, width, height));
                        // Levels are Q15 in the renderer.
                        let q = |f: f64| (f * ONE as f64).round() / ONE as f64;
                        q(q(q(key) * m) * q(opacity))
                    });
                    let ac = (a[0] + a[1]) / 2.0;
                    let row = &mut canvas[y as usize];
                    let base = 2 * x as usize;
                    for (i, alpha) in [ac, a[0], ac, a[1]].into_iter().enumerate() {
                        let d = row[base + i] as f64;
                        let sv = v[i].round().min(1023.0);
                        row[base + i] = (d + (sv - d) * alpha).round() as u16;
                    }
                    x += 2;
                }
            }
            (picture, fitted)
        };
        // Border, as the renderer draws it (around the snapped picture).
        if !l.layer.border.is_none() {
            let argb = l.layer.border.argb();
            let c = argb_to_ycbcr10(argb);
            let a = ((argb >> 24) as f64 * l.layer.alpha).round() / 255.0;
            let mut around = fitted.clone();
            around.x = picture.x as f64 / width as f64;
            around.y = picture.y as f64 / height as f64;
            around.w = picture.width as f64 / width as f64;
            around.h = picture.height as f64 / height as f64;
            for strip in around.border_strips(width as u32, height as u32) {
                let r = {
                    let s = strip.clip(width as u32, height as u32);
                    let x0 = s.x & !1;
                    let x1 = ((s.x + s.width + 1) & !1).min(width as i32 & !1);
                    PixelRect { x: x0, width: (x1 - x0).max(0), ..s }
                };
                for y in r.y..r.y + r.height {
                    for x in r.x..r.x + r.width {
                        let (x, y) = (x as usize, y as usize);
                        let row = &mut canvas[y];
                        let chroma = if x % 2 == 0 { 3 } else { 1 };
                        for (s, v) in [(2 * x + 1, c[0]), (4 * (x / 2), c[1]), (4 * (x / 2) + 2, c[2])].into_iter().take(chroma) {
                            let d = row[s] as f64;
                            row[s] = (d + (v as f64 - d) * a).round() as u16;
                        }
                    }
                }
            }
        }
        let _ = opacity;
    }
    canvas
}

fn render(layers: &[(Layer10, Option<&Frame>)], width: usize, height: usize, pool: Option<&StripPool>) -> Vec<u8> {
    let mut out = vec![0x55u8; v210::frame_bytes(width, height, false)];
    let list: Vec<Layer10> = layers.iter().map(|(l, _)| l.clone()).collect();
    let mut mixer = Mixer10::new();
    mixer.force_parallel = Some(pool.is_some());
    // The reference reads every row.
    mixer.decimate = Some(false);
    mixer.draw(&mut out, width, height, &list, Some(BLACK10), pool, 0.02);
    out
}

fn with_source<'a>(f: &'a Frame, layer: CompositionLayer) -> (Layer10<'a>, Option<&'a Frame>) {
    (
        Layer10 {
            source: Some(f.source()),
            layer,
            fill: None,
            clip: None,
            keying: Keying::None,
            mask: None,
        },
        Some(f),
    )
}

fn compare(got: &[Vec<u16>], want: &[Vec<u16>], tolerance: i32, what: &str) {
    let mut worst = (0, 0, 0);
    for (y, (g, w)) in got.iter().zip(want).enumerate() {
        for (s, (&a, &b)) in g.iter().zip(w).enumerate() {
            let d = (a as i32 - b as i32).abs();
            if d > worst.0 {
                worst = (d, y, s);
            }
        }
    }
    assert!(
        worst.0 <= tolerance,
        "{what}: off by {} at row {} sample {} (got {}, want {})",
        worst.0,
        worst.1,
        worst.2,
        got[worst.1][worst.2],
        want[worst.1][worst.2]
    );
}

#[test]
fn a_cut_is_the_source_bytes() {
    let mut rng = Lcg(1);
    for (w, h) in [(1920, 8), (1280, 4), (100, 6), (52, 3)] {
        let f = Frame::random(w, h, false, &mut rng);
        let out = render(&[with_source(&f, CompositionLayer::default())], w, h, None);
        let stride = v210::line_bytes(w);
        for y in 0..h {
            // The whole row, padding included: a cut copies the row.
            let (a, b) = (&out[y * stride..(y + 1) * stride], &f.bytes[y * stride..(y + 1) * stride]);
            assert!(a == b, "{w}x{h} row {y}");
        }
    }
}

#[test]
fn mixes_and_fills_match_the_reference() {
    let mut rng = Lcg(2);
    let (w, h) = (100, 6);
    let a = Frame::random(w, h, false, &mut rng);
    let b = Frame::random(w, h, false, &mut rng);
    for alpha in [0.0, 0.01, 0.25, 0.5, 0.999, 1.0] {
        let layers = vec![
            with_source(&a, CompositionLayer::default()),
            with_source(&b, CompositionLayer { alpha, ..Default::default() }),
        ];
        let got = read_canvas(&render(&layers, w, h, None), w, h);
        compare(&got, &reference(&layers, w, h), 1, &format!("mix {alpha}"));
        // Dip: a colour over a.
        let layers = vec![
            with_source(&a, CompositionLayer::default()),
            (Layer10::color(0xFF20_4080, alpha), None),
        ];
        let got = read_canvas(&render(&layers, w, h, None), w, h);
        compare(&got, &reference(&layers, w, h), 1, &format!("dip {alpha}"));
    }
}

fn random_layer<'a>(rng: &mut Lcg, frames: &'a [Frame], round: usize, same_size: bool) -> (Layer10<'a>, Option<&'a Frame>) {
    let f = |rng: &mut Lcg, lo: f64, hi: f64| lo + (rng.next_u32() % 10_000) as f64 / 10_000.0 * (hi - lo);
    if rng.next_u32() % 8 == 0 {
        let argb = 0xFF00_0000 | (rng.next_u32() & 0xFF_FFFF);
        let mut l = Layer10::color(argb, f(rng, 0.2, 1.0));
        l.layer.x = f(rng, -0.2, 0.8);
        l.layer.y = f(rng, -0.2, 0.8);
        l.layer.w = f(rng, 0.1, 0.8);
        l.layer.h = f(rng, 0.1, 0.8);
        return (l, None);
    }
    let frame = &frames[rng.next_u32() as usize % frames.len()];
    let scaled = round % 3 != 0;
    let mut layer = CompositionLayer {
        fit: LayerFit::Stretch,
        alpha: if rng.next_u32() % 3 == 0 { f(rng, 0.1, 1.0) } else { 1.0 },
        ..Default::default()
    };
    if scaled {
        layer.x = f(rng, -0.3, 0.9);
        layer.y = f(rng, -0.3, 0.9);
        layer.w = f(rng, 0.05, 1.2);
        layer.h = f(rng, 0.05, 1.2);
        if rng.next_u32() % 2 == 0 {
            layer.crop = LayerCrop {
                left: f(rng, 0.0, 0.3),
                right: f(rng, 0.0, 0.3),
                top: f(rng, 0.0, 0.3),
                bottom: f(rng, 0.0, 0.3),
            };
        }
    } else {
        // Unscaled: whole pixels, moved by any even number of pixels (also
        // ones that don't line up with v210 words).
        let dx = (rng.next_u32() % 41) as i32 - 20;
        let dy = (rng.next_u32() % 5) as i32 - 2;
        layer.x = (2 * dx) as f64 / 100.0;
        layer.y = dy as f64 / 6.0;
    }
    if rng.next_u32() % 4 == 0 {
        layer.border = LayerBorder {
            width: 1 + rng.next_u32() % 3,
            color: "#ff8000".into(),
            position: [BorderPosition::Inside, BorderPosition::Outside, BorderPosition::Center]
                [rng.next_u32() as usize % 3],
        };
    }
    let keying = match rng.next_u32() % 4 {
        0 => Keying::Luma(KeyShape::luma(f(rng, 0.0, 0.6) as f32, f(rng, 0.05, 0.5) as f32, rng.next_u32() % 2 == 0)),
        1 => Keying::Linear(KeyShape::linear(f(rng, 0.0, 0.3) as f32, f(rng, 1.0, 2.0) as f32, false)),
        _ => Keying::None,
    };
    let clip = (rng.next_u32() % 4 == 0).then(|| PixelRect {
        x: (rng.next_u32() % 60) as i32,
        y: (rng.next_u32() % 4) as i32,
        width: (rng.next_u32() % 60) as i32,
        height: (rng.next_u32() % 6) as i32,
    });
    let mask = (rng.next_u32() % 4 == 0).then(|| WipeMask {
        pattern: WipePattern::ALL[rng.next_u32() as usize % WipePattern::ALL.len()],
        progress: f(rng, 0.0, 1.0) as f32,
        soft: if rng.next_u32() % 2 == 0 { 0.0 } else { f(rng, 0.01, 0.5) as f32 },
        reverse: rng.next_u32() % 2 == 0,
    });
    // Luma keys of scaled pictures are only checked within a tolerance that
    // the key's steepness amplifies; keep them to unscaled layers here.
    let keying = if (scaled || !same_size) && matches!(keying, Keying::Luma(_)) { Keying::None } else { keying };
    (
        Layer10 {
            source: Some(frame.source()),
            layer,
            fill: None,
            clip,
            keying,
            mask,
        },
        Some(frame),
    )
}

#[test]
fn random_layouts_match_the_reference() {
    let mut rng = Lcg(3);
    let pool = StripPool::new(3);
    for round in 0..400 {
        let (w, h) = [(100, 6), (96, 8), (52, 5), (148, 7)][round % 4];
        let (sw, sh) = [(w, h), (w, h), (64, 9), (200, 12)][(round / 4) % 4];
        let frames = vec![
            Frame::random(sw, sh, false, &mut rng),
            Frame::random(sw, sh, true, &mut rng),
        ];
        let n = 1 + rng.next_u32() as usize % 4;
        let layers: Vec<_> = (0..n).map(|_| random_layer(&mut rng, &frames, round, (sw, sh) == (w, h))).collect();
        let scaled = round % 3 != 0 || (sw, sh) != (w, h);
        let want = reference(&layers, w, h);
        let bytes = render(&layers, w, h, None);
        let got = read_canvas(&bytes, w, h);
        if std::env::var("DEBUG_ROUND").ok() == Some(round.to_string()) {
            for (l, _) in &layers {
                eprintln!("layer {:?} fill {:?} clip {:?} mask {:?} keyed {} src {:?}", l.layer, l.fill, l.clip, l.mask, !matches!(l.keying, Keying::None), l.source.map(|s| (s.width, s.height, s.key.is_some())));
            }
        }
        // A scaled key's level comes from integer filter weights; a steep
        // key (gain) amplifies their rounding.
        let keyed = layers.iter().any(|(l, _)| matches!(l.keying, Keying::Linear(_)));
        let tolerance = match (scaled, keyed) {
            (false, _) => 1,
            (true, false) => 2,
            (true, true) => 4,
        };
        compare(&got, &want, tolerance, &format!("round {round}"));
        // Strips on the pool draw the same bytes.
        assert!(render(&layers, w, h, Some(&pool)) == bytes, "round {round}: parallel");
    }
}

#[test]
fn box_downscales_match_the_reference() {
    let mut rng = Lcg(4);
    let (w, h) = (96, 8);
    for k in 1..=5usize {
        let f = Frame::random(w * k, h * k, false, &mut rng);
        // Whole canvas, and a box at an offset (also not word aligned).
        for (x, y, size) in [(0.0, 0.0, 1.0), (10.0 / 96.0, 2.0 / 8.0, 0.5), (22.0 / 96.0, 0.0, 0.25)] {
            let layer = CompositionLayer {
                x,
                y,
                w: size,
                h: size,
                fit: LayerFit::Stretch,
                ..Default::default()
            };
            let layers = vec![with_source(&f, layer)];
            let got = read_canvas(&render(&layers, w, h, None), w, h);
            compare(&got, &reference(&layers, w, h), 1, &format!("box {k} at {x} size {size}"));
        }
    }
}

/// Decimated downscales (7.5:1 and more): the same picture as the reference
/// drawn from the rows read (every n-th, from the middle of each block).
#[test]
fn decimated_downscales_read_every_nth_row() {
    use super::render::vertical_step;
    let mut rng = Lcg(9);
    let (w, h) = (480, 270);
    let f = Frame::random(1920, 1080, true, &mut rng);
    for (bw, bh, vstep) in [(240.0, 134.0, 2usize), (192.0, 108.0, 2), (120.0, 66.0, 4)] {
        assert_eq!(vertical_step(1080.0 / bh), vstep);
        let phase = (vstep - 1) / 2;
        let rows: Vec<usize> = (0..1080 / vstep).map(|i| i * vstep + phase).collect();
        let sub = Frame::from_samples(
            1920,
            rows.len(),
            rows.iter().map(|&r| f.samples[r].clone()).collect(),
            Some(rows.iter().map(|&r| f.key.as_ref().unwrap()[r].clone()).collect()),
        );
        let layer = CompositionLayer {
            x: 0.1,
            y: 0.1,
            w: bw / w as f64,
            h: bh / h as f64,
            fit: LayerFit::Stretch,
            ..Default::default()
        };
        for keyed in [false, true] {
            let mk = |fr: &'static Frame| {
                let (mut l, _) = with_source(fr, layer.clone());
                if keyed {
                    l.keying = Keying::Linear(KeyShape::linear(0.0, 1.0, false));
                }
                l
            };
            let fs: &'static Frame = Box::leak(Box::new(Frame::from_samples(f.width, f.height, f.samples.clone(), f.key.clone())));
            let ss: &'static Frame = Box::leak(Box::new(Frame::from_samples(sub.width, sub.height, sub.samples.clone(), sub.key.clone())));
            let mut a = vec![0u8; v210::frame_bytes(w, h, false)];
            let mut b = a.clone();
            let mut m = Mixer10::new();
            m.force_parallel = Some(false);
            m.decimate = Some(true);
            m.draw(&mut a, w, h, &[mk(fs)], Some(BLACK10), None, 0.02);
            let mut m = Mixer10::new();
            m.force_parallel = Some(false);
            m.decimate = Some(false);
            m.draw(&mut b, w, h, &[mk(ss)], Some(BLACK10), None, 0.02);
            compare(&read_canvas(&a, w, h), &read_canvas(&b, w, h), 1, &format!("{bw}x{bh} keyed {keyed}"));
        }
    }
}


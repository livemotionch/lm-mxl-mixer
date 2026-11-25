//! The renderer against a slow floating-point reference: every source
//! format on every canvas format, with downscaling (box and fractional),
//! upscaling, odd crops, layers partly outside the canvas, opacity, and
//! whole layouts with occlusion and borders.

// Index loops over several slices at once are intentional here.
#![allow(clippy::needless_range_loop)]

use super::config::{
    BorderPosition, CompositionBackground, CompositionLayer, LayerBorder, LayerCrop, LayerFit,
    PixelRect,
};
use super::color::{Transform, argb_to};
use super::formats::SrcFormat;
use super::layout::{Item, MixerLayer, MixerLayout, Plan, SourceGeom, plan};
use super::render::{CacheMode, CanvasFormat, Frame, LayerInput, PixelCache, Prepared, render};
use super::testing::{Lcg, OwnedCanvas, OwnedSource, draw_one, key_for};

/// Float weights of output `j` of `[a, b)` → `n` (same definition as the
/// tables, without quantization).
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

/// Float canvas: Y, U, V (opaque) or premultiplied R, G, B, A.
struct RefCanvas {
    format: CanvasFormat,
    w: usize,
    h: usize,
    planes: Vec<Vec<f64>>,
}

impl RefCanvas {
    fn new(format: CanvasFormat, w: usize, h: usize, bg: Option<[u8; 3]>) -> RefCanvas {
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let planes = match (format, bg) {
            (CanvasFormat::Bgra, _) => vec![vec![0.0; w * h]; 4],
            (_, bg) => {
                let bg = bg.unwrap();
                vec![
                    vec![bg[0] as f64; w * h],
                    vec![bg[1] as f64; cw * ch],
                    vec![bg[2] as f64; cw * ch],
                ]
            }
        };
        RefCanvas {
            format,
            w,
            h,
            planes,
        }
    }

    /// Draw a whole layer (no occlusion) with `opacity` 0–255.
    fn draw(&mut self, src: &OwnedSource, crop: (u32, u32, u32, u32), pic: PixelRect, opacity: u8) {
        let spec = src.format.spec();
        let canvas_model = if self.format == CanvasFormat::Bgra {
            super::color::Model::Rgb
        } else {
            super::color::Model::yuv(super::color::BT709)
        };
        let t = Transform::new(src.model, canvas_model);
        let groups: Vec<(usize, Vec<usize>)> = if self.format == CanvasFormat::Bgra {
            vec![(1, vec![0, 1, 2])]
        } else {
            vec![(1, vec![0]), (2, vec![1, 2])]
        };
        let (cx0, cy0, cx1, cy1) = crop;
        let o = opacity as f64 / 255.0;
        for (gi, (sub, outputs)) in groups.iter().enumerate() {
            let (gw, gh) = (pic.width as usize / sub, pic.height as usize / sub);
            let (cw, chh) = (self.w.div_ceil(*sub), self.h.div_ceil(*sub));
            for gy in 0..gh {
                let y = pic.y / *sub as i32 + gy as i32;
                if y < 0 || y as usize >= chh || (*sub == 1 && y as usize >= self.h) {
                    continue;
                }
                for gx in 0..gw {
                    let x = pic.x / *sub as i32 + gx as i32;
                    if x < 0 || x as usize >= cw {
                        continue;
                    }
                    // Filtered premultiplied channels and alpha.
                    let mut val = [0.0f64; 4];
                    for c in 0..4 {
                        if c == 3 && spec.alpha.is_none() {
                            val[3] = 255.0;
                            continue;
                        }
                        let ch = if c == 3 {
                            spec.alpha.unwrap()
                        } else {
                            spec.chans[c]
                        };
                        let (hs, vs) = (ch.hsub as f64, spec.vsub[ch.plane] as f64);
                        let (nw, nh) = src.native_dims[c];
                        let hw = ref_weights(cx0 as f64 / hs, cx1 as f64 / hs, gw, nw, gx);
                        let vw: Vec<(usize, f64)> = if gi == 1 && vs == 1.0 {
                            // 4:2:0 chroma of a full-height plane: the two luma rows.
                            let lh = pic.height as usize;
                            let mut v = ref_weights(cy0 as f64, cy1 as f64, lh, nh, 2 * gy);
                            v.extend(ref_weights(cy0 as f64, cy1 as f64, lh, nh, 2 * gy + 1));
                            v.iter().map(|&(i, w)| (i, w / 2.0)).collect()
                        } else {
                            ref_weights(cy0 as f64 / vs, cy1 as f64 / vs, gh, nh, gy)
                        };
                        let mut sum = 0.0;
                        for &(sy, wy) in &vw {
                            for &(sx, wx) in &hw {
                                // Premultiplied, as stored.
                                sum += wy * wx * src.premul8(c, sx, sy);
                            }
                        }
                        val[c] = sum;
                    }
                    let a = val[3] / 255.0;
                    let mut out = [0.0f64; 3];
                    for k in 0..3 {
                        out[k] = (0..3).map(|j| t.m[k][j] as f64 * val[j]).sum::<f64>()
                            + t.o[k] as f64 * a;
                        // Out-of-gamut colours are clipped before blending.
                        out[k] = out[k].clamp(0.0, val[3]);
                    }
                    let i = y as usize * cw + x as usize;
                    for (slot, &k) in outputs.iter().enumerate() {
                        let plane = if self.format == CanvasFormat::Bgra {
                            k
                        } else {
                            gi + slot
                        };
                        let d = &mut self.planes[plane][i];
                        *d = out[k] * o + *d * (1.0 - a * o);
                    }
                    if self.format == CanvasFormat::Bgra {
                        let d = &mut self.planes[3][i];
                        *d = val[3] * o + *d * (1.0 - a * o);
                    }
                }
            }
        }
    }

    fn fill(&mut self, r: PixelRect, argb: u32, alpha: u8) {
        let c = argb_to(
            if self.format == CanvasFormat::Bgra {
                super::color::Model::Rgb
            } else {
                super::color::Model::yuv(super::color::BT709)
            },
            argb,
        );
        let a = alpha as f64 / 255.0;
        let r = r.clip(self.w as u32, self.h as u32);
        let (x0, x1, y0, y1) = (
            r.x as usize,
            (r.x + r.width) as usize,
            r.y as usize,
            (r.y + r.height) as usize,
        );
        if self.format == CanvasFormat::Bgra {
            for y in y0..y1 {
                for x in x0..x1 {
                    let i = y * self.w + x;
                    for k in 0..3 {
                        self.planes[k][i] = c[k] as f64 * a + self.planes[k][i] * (1.0 - a);
                    }
                    self.planes[3][i] = 255.0 * a + self.planes[3][i] * (1.0 - a);
                }
            }
        } else {
            for y in y0..y1 {
                for x in x0..x1 {
                    let d = &mut self.planes[0][y * self.w + x];
                    *d = c[0] as f64 * a + *d * (1.0 - a);
                }
            }
            let cw = self.w.div_ceil(2);
            for y in y0 / 2..y1.div_ceil(2) {
                for x in x0 / 2..x1.div_ceil(2) {
                    for k in 1..3 {
                        let d = &mut self.planes[k][y * cw + x];
                        *d = c[k] as f64 * a + *d * (1.0 - a);
                    }
                }
            }
        }
    }

    /// Largest difference to `canvas` (premultiplied colour on BGRA).
    fn max_error(&self, canvas: &OwnedCanvas) -> f64 {
        let mut worst = 0.0f64;
        match self.format {
            CanvasFormat::Bgra => {
                for y in 0..self.h {
                    for x in 0..self.w {
                        let i = y * self.w + x;
                        let px = &canvas.planes[0][y * canvas.strides[0] + x * 4..][..4];
                        let a = self.planes[3][i];
                        worst = worst.max((px[3] as f64 - a).abs());
                        if a < 0.5 {
                            // (Alpha 1 is rounding at the threshold.)
                            assert!(
                                px == [0, 0, 0, 0] || (px[3] == 1 && a > 0.0),
                                "transparent pixel at {x},{y}: {px:?}"
                            );
                            continue;
                        }
                        for (k, b) in [(0, 2), (1, 1), (2, 0)] {
                            let want = self.planes[k][i];
                            let e = (px[b] as f64 - want).abs();
                            if e > 3.0 && e > worst {
                                eprintln!(
                                    "({x},{y}) channel {k}: got {px:?}, want {want:.1} a {a:.1}"
                                );
                            }
                            worst = worst.max(e);
                        }
                    }
                }
            }
            format => {
                let (cw, ch) = (self.w.div_ceil(2), self.h.div_ceil(2));
                for y in 0..self.h {
                    for x in 0..self.w {
                        let got = canvas.planes[0][y * canvas.strides[0] + x] as f64;
                        worst = worst.max((got - self.planes[0][y * self.w + x]).abs());
                    }
                }
                for y in 0..ch {
                    for x in 0..cw {
                        let (u, v) = if format == CanvasFormat::Nv12 {
                            let row = &canvas.planes[1][y * canvas.strides[1]..];
                            (row[2 * x], row[2 * x + 1])
                        } else {
                            (
                                canvas.planes[1][y * canvas.strides[1] + x],
                                canvas.planes[2][y * canvas.strides[2] + x],
                            )
                        };
                        worst = worst.max((u as f64 - self.planes[1][y * cw + x]).abs());
                        worst = worst.max((v as f64 - self.planes[2][y * cw + x]).abs());
                    }
                }
            }
        }
        worst
    }
}

const CANVASES: [CanvasFormat; 3] = [CanvasFormat::Nv12, CanvasFormat::I420, CanvasFormat::Bgra];

fn bg_of(format: CanvasFormat) -> Option<[u8; 3]> {
    (format != CanvasFormat::Bgra).then_some([16, 128, 128])
}

fn fresh(format: CanvasFormat, w: usize, h: usize) -> OwnedCanvas {
    let mut c = OwnedCanvas::new(format, w, h);
    if format != CanvasFormat::Bgra {
        c.planes[0].fill(16);
        c.planes[1].fill(128);
        c.planes[2].fill(128);
    }
    c
}

/// One layer of each format on each canvas, compared to the reference.
fn check_layer(
    sw: usize,
    sh: usize,
    crop: (u32, u32, u32, u32),
    pic: PixelRect,
    opacity: u8,
    tolerance: f64,
) {
    let (cw, ch) = (64, 48);
    for format in SrcFormat::ALL {
        let src = OwnedSource::new(format, sw, sh, 7);
        for canvas_format in CANVASES {
            let mut canvas = fresh(canvas_format, cw, ch);
            let prepared = Prepared::new(key_for(&src, &canvas, crop, pic));
            draw_one(&src, &mut canvas, &prepared, opacity, None);
            let mut reference = RefCanvas::new(canvas_format, cw, ch, bg_of(canvas_format));
            reference.draw(&src, crop, pic, opacity);
            let err = reference.max_error(&canvas);
            // Premultiplied 8-bit YUV → RGB amplifies the chroma rounding
            // (×1.8 for Cr → R).
            let tolerance = if format == SrcFormat::Ayuv && canvas_format == CanvasFormat::Bgra {
                tolerance + 1.0
            } else {
                tolerance
            };
            assert!(
                err <= tolerance,
                "{format:?} on {canvas_format:?}, {sw}x{sh} crop {crop:?} -> {pic:?}: error {err:.2}"
            );
        }
    }
}

fn rect(x: i32, y: i32, width: i32, height: i32) -> PixelRect {
    PixelRect {
        x,
        y,
        width,
        height,
    }
}

#[test]
fn same_size_is_exact() {
    check_layer(48, 32, (0, 0, 48, 32), rect(8, 8, 48, 32), 255, 1.5);
}

#[test]
fn box_downscale() {
    // 4:1 and 2:1
    check_layer(96, 64, (0, 0, 96, 64), rect(4, 6, 24, 16), 255, 1.5);
    check_layer(96, 64, (0, 0, 96, 64), rect(0, 0, 48, 32), 255, 1.5);
}

#[test]
fn fractional_downscale_and_crop() {
    check_layer(96, 72, (0, 0, 96, 72), rect(2, 2, 34, 26), 255, 1.5);
    // Odd crop: chroma starts half a sample in.
    check_layer(96, 72, (3, 5, 91, 69), rect(10, 4, 30, 22), 255, 1.5);
}

#[test]
fn upscale() {
    check_layer(24, 16, (0, 0, 24, 16), rect(0, 0, 60, 40), 255, 1.5);
    check_layer(24, 16, (1, 1, 23, 15), rect(4, 2, 40, 44), 255, 1.5);
}

#[test]
fn partly_outside_the_canvas() {
    check_layer(96, 64, (0, 0, 96, 64), rect(-20, -10, 48, 32), 255, 1.5);
    check_layer(96, 64, (0, 0, 96, 64), rect(40, 30, 48, 32), 255, 1.5);
}

#[test]
fn opacity() {
    check_layer(96, 64, (0, 0, 96, 64), rect(4, 4, 40, 30), 128, 1.5);
}

#[test]
fn transparent_regions_are_skipped_exactly() {
    let src = super::testing::lower_third(192, 108);
    for canvas_format in CANVASES {
        for (pic, op) in [
            (rect(0, 0, 96, 54), 255u8),
            (rect(-10, 4, 192, 108), 255),
            (rect(6, 2, 70, 40), 180),
        ] {
            let mut canvas = fresh(canvas_format, 96, 64);
            // Something below, so skipping wrongly would show.
            canvas.planes[0].fill(90);
            let mut reference = RefCanvas::new(canvas_format, 96, 64, bg_of(canvas_format));
            reference.planes[0].fill(90.0);
            if canvas_format == CanvasFormat::Bgra {
                canvas.planes[0].fill(90);
                for k in 0..4 {
                    reference.planes[k].fill(90.0);
                }
            }
            let prepared = Prepared::new(key_for(&src, &canvas, (0, 0, 192, 108), pic));
            draw_one(&src, &mut canvas, &prepared, op, None);
            reference.draw(&src, (0, 0, 192, 108), pic, op);
            let err = reference.max_error(&canvas);
            assert!(err <= 1.5, "{canvas_format:?} {pic:?}: error {err:.2}");

            // Through the cache (built over stale contents).
            let mut cache = PixelCache::new(&prepared, 96, 64);
            for v in cache_planes(&mut cache) {
                v.fill(77);
            }
            let view = super::render::cache_view(&mut cache);
            let mut cached = fresh(canvas_format, 96, 64);
            cached.planes[0].fill(90);
            let before = cached.planes.clone();
            let mut direct = fresh(canvas_format, 96, 64);
            direct.planes = before;
            draw_one(&src, &mut direct, &prepared, op, None);
            let plan = Plan {
                background: Vec::new(),
                items: vec![Item::Layer {
                    index: 0,
                    picture: pic,
                    visible: vec![pic.clip(96, 64)],
                    opacity: op,
                }],
            };
            let c = cached.canvas();
            let frame = Frame {
                canvas: &c,
                plan: &plan,
                layers: vec![Some(LayerInput {
                    source: src.frame(),
                    prepared: &prepared,
                    cache: CacheMode::Build(&view),
                })],
                background: None,
            };
            render(&frame, None);
            drop(frame);
            assert_eq!(
                direct.planes, cached.planes,
                "{canvas_format:?} {pic:?} cache"
            );
        }
    }
}

fn cache_planes(cache: &mut PixelCache) -> Vec<&mut Vec<u8>> {
    cache.planes_mut()
}

#[test]
fn strips_and_cache_give_the_same_picture() {
    let pool = super::pool::StripPool::new(3);
    for format in [SrcFormat::Uyvy, SrcFormat::Bgra, SrcFormat::Nv12] {
        let src = OwnedSource::new(format, 200, 120, 3);
        for canvas_format in CANVASES {
            let (cw, ch) = (160, 128);
            let pic = rect(10, 6, 140, 110);
            let mut one = fresh(canvas_format, cw, ch);
            let prepared = Prepared::new(key_for(&src, &one, (0, 0, 200, 120), pic));
            draw_one(&src, &mut one, &prepared, 200, None);
            let mut many = fresh(canvas_format, cw, ch);
            draw_one(&src, &mut many, &prepared, 200, Some(&pool));
            assert_eq!(
                one.planes, many.planes,
                "{format:?} {canvas_format:?} strips"
            );

            // Build the cache, then draw from it.
            let mut cache = PixelCache::new(&prepared, cw, ch);
            let view = super::render::cache_view(&mut cache);
            for build in [true, false] {
                let mut cached = fresh(canvas_format, cw, ch);
                let plan = Plan {
                    background: Vec::new(),
                    items: vec![Item::Layer {
                        index: 0,
                        picture: pic,
                        visible: vec![pic.clip(cw as u32, ch as u32)],
                        opacity: 200,
                    }],
                };
                let c = cached.canvas();
                let frame = Frame {
                    canvas: &c,
                    plan: &plan,
                    layers: vec![Some(LayerInput {
                        source: src.frame(),
                        prepared: &prepared,
                        cache: if build {
                            CacheMode::Build(&view)
                        } else {
                            CacheMode::Use(&view)
                        },
                    })],
                    background: None,
                };
                render(&frame, Some(&pool));
                drop(frame);
                assert_eq!(
                    one.planes, cached.planes,
                    "{format:?} {canvas_format:?} cache {build}"
                );
            }
        }
    }
}

/// Random layouts: the renderer (with occlusion) against the reference
/// drawing every layer whole, bottom-up.
/// A layer moved within the canvas keeps its tables and cache: drawn from
/// the cache at the new place, it's the same as drawn there afresh.
#[test]
fn moved_layer_keeps_tables_and_cache() {
    for format in [SrcFormat::Uyvy, SrcFormat::Bgra] {
        let src = OwnedSource::new(format, 200, 120, 4);
        for canvas_format in CANVASES {
            let (cw, ch) = (160, 128);
            let (from, to) = (rect(10, 6, 100, 60), rect(40, 50, 100, 60));
            let canvas = fresh(canvas_format, cw, ch);
            let mut prepared = Prepared::new(key_for(&src, &canvas, (0, 0, 200, 120), from));
            let mut cache = PixelCache::new(&prepared, cw, ch);
            let draw = |prepared: &Prepared,
                        pic: PixelRect,
                        mode: Option<(bool, &super::render::CacheView)>| {
                let mut out = fresh(canvas_format, cw, ch);
                let plan = Plan {
                    background: Vec::new(),
                    items: vec![Item::Layer {
                        index: 0,
                        picture: pic,
                        visible: vec![pic.clip(cw as u32, ch as u32)],
                        opacity: 255,
                    }],
                };
                let c = out.canvas();
                let frame = Frame {
                    canvas: &c,
                    plan: &plan,
                    layers: vec![Some(LayerInput {
                        source: src.frame(),
                        prepared,
                        cache: match mode {
                            None => CacheMode::Off,
                            Some((true, v)) => CacheMode::Build(v),
                            Some((false, v)) => CacheMode::Use(v),
                        },
                    })],
                    background: None,
                };
                render(&frame, None);
                drop(frame);
                out.planes
            };
            {
                let view = super::render::cache_view(&mut cache);
                draw(&prepared, from, Some((true, &view)));
            }
            let moved = Prepared::new(key_for(&src, &canvas, (0, 0, 200, 120), to));
            assert!(prepared.key.same_tables(&moved.key));
            prepared.set_picture(to);
            assert!(
                cache.fits(&prepared, cw, ch),
                "{format:?} {canvas_format:?}"
            );
            let view = super::render::cache_view(&mut cache);
            assert_eq!(
                draw(&prepared, to, Some((false, &view))),
                draw(&moved, to, None),
                "{format:?} {canvas_format:?}"
            );
            // Partly off the canvas: the cache no longer fits.
            prepared.set_picture(rect(100, 90, 100, 60));
            assert!(!cache.fits(&prepared, cw, ch));
        }
    }
}

/// Same-size layers with and without alpha over another layer, on every
/// canvas (the dedicated paths blend into what's there). Two 8-bit blends
/// in a row: within 2 levels of the float reference.
#[test]
fn same_size_layers_blend_over_others() {
    let (cw, ch) = (64, 48);
    let below = OwnedSource::new(SrcFormat::Uyvy, cw, ch, 9);
    for format in [
        SrcFormat::Bgra,
        SrcFormat::Bgrx,
        SrcFormat::Rgba,
        SrcFormat::Uyvy,
    ] {
        let src = OwnedSource::new(format, 48, 32, 10);
        for canvas_format in CANVASES {
            for opacity in [255, 140] {
                let mut canvas = fresh(canvas_format, cw, ch);
                let mut reference = RefCanvas::new(canvas_format, cw, ch, bg_of(canvas_format));
                let full = rect(0, 0, cw as i32, ch as i32);
                let pic = rect(8, 6, 48, 32);
                let pb =
                    Prepared::new(key_for(&below, &canvas, (0, 0, cw as u32, ch as u32), full));
                let p = Prepared::new(key_for(&src, &canvas, (0, 0, 48, 32), pic));
                reference.draw(&below, (0, 0, cw as u32, ch as u32), full, 200);
                reference.draw(&src, (0, 0, 48, 32), pic, opacity);
                let plan = Plan {
                    background: Vec::new(),
                    items: vec![
                        Item::Layer {
                            index: 0,
                            picture: full,
                            visible: vec![full],
                            opacity: 200,
                        },
                        Item::Layer {
                            index: 1,
                            picture: pic,
                            visible: vec![pic],
                            opacity,
                        },
                    ],
                };
                let c = canvas.canvas();
                let frame = Frame {
                    canvas: &c,
                    plan: &plan,
                    layers: vec![
                        Some(LayerInput {
                            source: below.frame(),
                            prepared: &pb,
                            cache: CacheMode::Off,
                        }),
                        Some(LayerInput {
                            source: src.frame(),
                            prepared: &p,
                            cache: CacheMode::Off,
                        }),
                    ],
                    background: None,
                };
                render(&frame, None);
                drop(frame);
                let err = reference.max_error(&canvas);
                // Alpha over a half-transparent BGRA canvas: two rounded
                // premultiplied blends.
                let tolerance = if format.has_alpha() && canvas_format == CanvasFormat::Bgra {
                    2.5
                } else {
                    2.0
                };
                assert!(
                    err <= tolerance,
                    "{format:?} on {canvas_format:?} at {opacity}: error {err:.2}"
                );
            }
        }
    }
}

#[test]
fn random_layouts_match_the_reference() {
    let mut rng = Lcg(42);
    for round in 0..48 {
        // Odd canvas sizes in every other round.
        let (cw, ch) = if round < 24 { (96, 64) } else { (95, 63) };
        let canvas_format = CANVASES[round % 3];
        let n = 1 + rng.next_u32() as usize % 4;
        let mut sources = Vec::new();
        let mut layers = Vec::new();
        for i in 0..n {
            let format = SrcFormat::ALL[rng.next_u32() as usize % SrcFormat::ALL.len()];
            let (sw, sh) = [(96, 64), (48, 36), (30, 20)][rng.next_u32() as usize % 3];
            sources.push(OwnedSource::new(format, sw, sh, rng.next_u32() as u64));
            let f = |r: &mut Lcg, m: u32| (r.next_u32() % m) as f64 / 100.0;
            layers.push(MixerLayer {
                pad: format!("sink_{i}"),
                shown: true,
                layer: CompositionLayer {
                    id: format!("l{i}"),
                    x: f(&mut rng, 80) - 0.1,
                    y: f(&mut rng, 80) - 0.1,
                    w: 0.2 + f(&mut rng, 90),
                    h: 0.2 + f(&mut rng, 90),
                    fit: if rng.next_u32().is_multiple_of(2) {
                        LayerFit::Stretch
                    } else {
                        LayerFit::Letterbox
                    },
                    crop: if rng.next_u32().is_multiple_of(3) {
                        LayerCrop {
                            left: 0.1,
                            top: 0.05,
                            ..Default::default()
                        }
                    } else {
                        LayerCrop::default()
                    },
                    alpha: if rng.next_u32().is_multiple_of(4) {
                        0.6
                    } else {
                        1.0
                    },
                    border: if rng.next_u32().is_multiple_of(3) {
                        LayerBorder {
                            width: 3,
                            color: "#ff8000".into(),
                            position: BorderPosition::Outside,
                        }
                    } else {
                        LayerBorder::default()
                    },
                    ..Default::default()
                },
                ..Default::default()
            });
        }
        let layout = MixerLayout {
            layers,
            background: if canvas_format == CanvasFormat::Bgra {
                CompositionBackground::Transparent
            } else {
                CompositionBackground::Black
            },
        };
        let geoms: Vec<Option<SourceGeom>> = sources
            .iter()
            .map(|s| {
                Some(SourceGeom {
                    width: s.width as u32,
                    height: s.height as u32,
                    par: 1.0,
                    has_alpha: s.format.has_alpha(),
                })
            })
            .collect();
        let p = plan(
            &layout,
            &geoms,
            cw as u32,
            ch as u32,
            canvas_format.is_420(),
        );

        let (canvas, reference) = draw_both(&p, &layout, &sources, canvas_format, cw, ch);
        let err = reference.max_error(&canvas);
        if err > 3.0 {
            for (i, s) in sources.iter().enumerate() {
                eprintln!("source {i}: {:?} {}x{}", s.format, s.width, s.height);
            }
            eprintln!("{:#?}", p.items);
            for k in 1..=p.items.len() {
                let part = Plan {
                    background: p.background.clone(),
                    items: p.items[..k].to_vec(),
                };
                let (c, r) = draw_both(&part, &layout, &sources, canvas_format, cw, ch);
                eprintln!("first {k} items: error {:.2}", r.max_error(&c));
                let part = Plan {
                    background: p.background.clone(),
                    items: p.items[k - 1..k].to_vec(),
                };
                let (c, r) = draw_both(&part, &layout, &sources, canvas_format, cw, ch);
                eprintln!("only item {k}: error {:.2}", r.max_error(&c));
            }
        }
        assert!(
            err <= 3.0,
            "round {round} ({canvas_format:?}): error {err:.2}, layout {layout:#?}"
        );
    }
}

/// Draw `items` of `p` with both renderers; returns (renderer, reference).
fn draw_both(
    p: &Plan,
    layout: &MixerLayout,
    sources: &[OwnedSource],
    canvas_format: CanvasFormat,
    cw: usize,
    ch: usize,
) -> (OwnedCanvas, RefCanvas) {
    let n = sources.len();
    let mut canvas = OwnedCanvas::new(canvas_format, cw, ch);
    let mut reference = RefCanvas::new(canvas_format, cw, ch, bg_of(canvas_format));
    let prepared: Vec<Option<Prepared>> = (0..n)
        .map(|i| {
            p.items.iter().find_map(|item| match item {
                Item::Layer { index, picture, .. } if *index == i => {
                    let s = &sources[i];
                    let crop = super::layout::crop_rect(
                        &layout.layers[i].layer.crop,
                        s.width as u32,
                        s.height as u32,
                    );
                    Some(Prepared::new(key_for(s, &canvas, crop, *picture)))
                }
                _ => None,
            })
        })
        .collect();
    for item in &p.items {
        match item {
            Item::Layer {
                index,
                picture,
                opacity,
                ..
            } => {
                let s = &sources[*index];
                reference.draw(
                    s,
                    prepared[*index].as_ref().unwrap().key.crop,
                    *picture,
                    *opacity,
                )
            }
            Item::Fill { rect, argb, alpha } => reference.fill(*rect, *argb, *alpha),
        }
    }
    let c = canvas.canvas();
    let frame = Frame {
        canvas: &c,
        plan: p,
        layers: prepared
            .iter()
            .enumerate()
            .map(|(i, pr)| {
                pr.as_ref().map(|pr| LayerInput {
                    source: sources[i].frame(),
                    prepared: pr,
                    cache: CacheMode::Off,
                })
            })
            .collect(),
        background: bg_of(canvas_format),
    };
    render(&frame, None);
    drop(frame);
    (canvas, reference)
}

#[test]
fn colours_are_exact() {
    // 100% white in YUV becomes 255 in BGRA (no 8-bit bias)...
    let mut white = OwnedSource::new(SrcFormat::Uyvy, 16, 8, 1);
    for (c, v) in [(0, 235u16), (1, 128), (2, 128)] {
        white.native[c].fill(v);
    }
    white.pack();
    let mut canvas = OwnedCanvas::new(CanvasFormat::Bgra, 16, 8);
    let prepared = Prepared::new(key_for(&white, &canvas, (0, 0, 16, 8), rect(0, 0, 16, 8)));
    draw_one(&white, &mut canvas, &prepared, 255, None);
    assert_eq!(&canvas.planes[0][..4], &[255, 255, 255, 255]);

    // ...and RGB red is the exact bt709 red in NV12.
    let mut red = OwnedSource::new(SrcFormat::Bgrx, 16, 8, 1);
    red.native[0].fill(255);
    red.native[1].fill(0);
    red.native[2].fill(0);
    red.pack();
    let mut canvas = fresh(CanvasFormat::Nv12, 16, 8);
    let prepared = Prepared::new(key_for(&red, &canvas, (0, 0, 16, 8), rect(0, 0, 16, 8)));
    draw_one(&red, &mut canvas, &prepared, 255, None);
    assert_eq!(canvas.planes[0][0], 63);
    assert_eq!(&canvas.planes[1][..2], &[102, 240]);
}



/// `src`'s planes and strides as they are.
fn tight(src: &OwnedSource) -> (Vec<Vec<u8>>, [usize; 3]) {
    let mut planes: Vec<Vec<u8>> = src.planes.iter().cloned().collect();
    planes.resize(3, Vec::new());
    (planes, src.strides)
}

/// Odd sizes, tiny pictures and crops, tight strides: no panic, and the
/// same picture as the reference.
#[test]
fn odd_sizes_tiny_layers_and_tight_buffers() {
    let mut rng = Lcg(7);
    let pool = super::pool::StripPool::new(3);
    for round in 0..1500 {
        // Odd canvas sizes too (any size can be configured).
        let (cw, ch) = [(40, 30), (41, 31), (40, 33)][round % 3];
        let format = SrcFormat::ALL[round % SrcFormat::ALL.len()];
        let canvas_format = CANVASES[(round / SrcFormat::ALL.len()) % 3];
        let big = round % 7 == 0;
        let sw = 1 + rng.next_u32() as usize % 13 + if round % 5 == 0 { 40 } else { 0 };
        let sh = 1 + rng.next_u32() as usize % 11;
        // Large ratios too (beyond the 256-sample box limit).
        let (sw, sh) = if big { (sw * 61, sh * 53) } else { (sw, sh) };
        let src = OwnedSource::new(format, sw, sh, round as u64);
        let (planes, strides) = tight(&src);
        let mut frame = src.frame();
        frame.planes = [&planes[0], &planes[1], &planes[2]];
        frame.strides = strides;
        // A crop of at least one pixel.
        let x0 = rng.next_u32() % sw as u32;
        let x1 = x0 + 1 + rng.next_u32() % (sw as u32 - x0);
        let y0 = rng.next_u32() % sh as u32;
        let y1 = y0 + 1 + rng.next_u32() % (sh as u32 - y0);
        let crop = (x0, y0, x1, y1);
        let even = canvas_format.is_420();
        let mut size = |m: u32| {
            let v = 1 + (rng.next_u32() % m) as i32;
            if even { (v + 1) & !1 } else { v }
        };
        let (w, h) = (size(50), size(36));
        let mut pos = |m: i32| {
            let v = (rng.next_u32() % (m as u32 + 20)) as i32 - 10;
            if even { v & !1 } else { v }
        };
        let pic = rect(pos(cw as i32), pos(ch as i32), w, h);
        let opacity = [255u8, 255, 128][round % 3];

        let mut canvas = fresh(canvas_format, cw, ch);
        let prepared = Prepared::new(key_for(&src, &canvas, crop, pic));
        let visible = pic.clip(cw as u32, ch as u32);
        let plan = Plan {
            background: Vec::new(),
            items: if visible.is_empty() {
                vec![]
            } else {
                vec![Item::Layer {
                    index: 0,
                    picture: pic,
                    visible: vec![visible],
                    opacity,
                }]
            },
        };
        let c = canvas.canvas();
        let f = Frame {
            canvas: &c,
            plan: &plan,
            layers: vec![Some(LayerInput {
                source: frame,
                prepared: &prepared,
                cache: CacheMode::Off,
            })],
            background: None,
        };
        render(&f, (round % 2 == 0).then_some(&pool));
        drop(f);
        let mut reference = RefCanvas::new(canvas_format, cw, ch, bg_of(canvas_format));
        reference.draw(&src, crop, pic, opacity);
        let err = reference.max_error(&canvas);
        let tolerance = if format == SrcFormat::Ayuv && canvas_format == CanvasFormat::Bgra {
            2.5
        } else {
            2.0
        };
        // Beyond ~16:1 the 1/256 weights are coarse against the float
        // area filter (noise sources); this checks they stay sane.
        let tolerance = if big { 12.0 } else { tolerance };
        assert!(
            err <= tolerance,
            "round {round}: {format:?} {sw}x{sh} crop {crop:?} -> {pic:?} on {canvas_format:?}: error {err:.2}"
        );

        // Through the cache (built over stale contents, then used): the same.
        let mut cache = PixelCache::new(&prepared, cw, ch);
        for v in cache_planes(&mut cache) {
            v.fill(77);
        }
        let view = super::render::cache_view(&mut cache);
        for build in [true, false] {
            let mut cached = fresh(canvas_format, cw, ch);
            let c = cached.canvas();
            let f = Frame {
                canvas: &c,
                plan: &plan,
                layers: vec![Some(LayerInput {
                    source: frame,
                    prepared: &prepared,
                    cache: if build {
                        CacheMode::Build(&view)
                    } else {
                        CacheMode::Use(&view)
                    },
                })],
                background: None,
            };
            render(&f, (round % 4 < 2).then_some(&pool));
            drop(f);
            assert!(
                cached.planes == canvas.planes,
                "round {round}: cache (build {build}) differs: {format:?} {sw}x{sh} crop {crop:?} -> {pic:?} on {canvas_format:?} {cw}x{ch}"
            );
        }
    }
}

/// A layer zoomed far beyond the canvas caches only what's on the canvas
/// (its whole picture would be gigabytes), and draws the same from it.
#[test]
fn cache_holds_only_the_part_on_the_canvas() {
    let (cw, ch) = (96, 64);
    let src = OwnedSource::new(SrcFormat::Bgra, 48, 32, 5);
    for canvas_format in CANVASES {
        let pic = rect(-60_000, -40_000, 120_000, 80_000);
        let canvas = fresh(canvas_format, cw, ch);
        let prepared = Prepared::new(key_for(&src, &canvas, (0, 0, 48, 32), pic));
        let mut cache = PixelCache::new(&prepared, cw, ch);
        assert!(
            cache.bytes() <= cw * ch * 4 * 2,
            "{canvas_format:?}: {} bytes",
            cache.bytes()
        );
        let plan = Plan {
            background: Vec::new(),
            items: vec![Item::Layer {
                index: 0,
                picture: pic,
                visible: vec![pic.clip(cw as u32, ch as u32)],
                opacity: 255,
            }],
        };
        let mut direct = fresh(canvas_format, cw, ch);
        draw_one(&src, &mut direct, &prepared, 255, None);
        let view = super::render::cache_view(&mut cache);
        for build in [true, false] {
            let mut cached = fresh(canvas_format, cw, ch);
            let c = cached.canvas();
            let f = Frame {
                canvas: &c,
                plan: &plan,
                layers: vec![Some(LayerInput {
                    source: src.frame(),
                    prepared: &prepared,
                    cache: if build {
                        CacheMode::Build(&view)
                    } else {
                        CacheMode::Use(&view)
                    },
                })],
                background: None,
            };
            render(&f, None);
            drop(f);
            assert!(
                direct.planes == cached.planes,
                "{canvas_format:?} build {build}"
            );
        }
    }
}





/// Every size the composition allows for a source with transparency
/// (`config::alpha_sizes`) has a dedicated kernel, on both canvases.
#[test]
fn allowed_alpha_sizes_take_dedicated_kernels() {
    for (sw, sh) in [(1920, 1080), (1280, 720)] {
        for format in [SrcFormat::Bgra, SrcFormat::Rgba, SrcFormat::Bgrx] {
            let src = OwnedSource::new(format, sw, sh, 1);
            for cf in [CanvasFormat::Nv12, CanvasFormat::Bgra] {
                for (_, w, h) in super::config::alpha_sizes(sw as u32, sh as u32) {
                    let canvas = OwnedCanvas::new(cf, 1920, 1080);
                    let pic = rect(2, 4, w as i32, h as i32);
                    let p =
                        Prepared::new(key_for(&src, &canvas, (0, 0, sw as u32, sh as u32), pic));
                    // RGBA on the BGRA canvas swaps channels: no kernel for that.
                    if format == SrcFormat::Rgba && cf == CanvasFormat::Bgra {
                        continue;
                    }
                    assert_ne!(
                        p.path(),
                        "general",
                        "{format:?} {sw}x{sh} -> {w}x{h} on {cf:?}"
                    );
                }
            }
        }
    }
}

/// Largest difference between two canvases of the same format and size.
fn max_diff(a: &OwnedCanvas, b: &OwnedCanvas) -> u8 {
    a.planes
        .iter()
        .zip(&b.planes)
        .flat_map(|(x, y)| x.iter().zip(y).map(|(&p, &q)| p.abs_diff(q)))
        .max()
        .unwrap_or(0)
}

/// Layers of planar 4:2:0 sources (source size, crop, picture on a 1080p
/// canvas, kernel expected): copies, crops, whole-number downscales with a
/// box or an area filter vertically, vertical-only scaling, a vertical
/// upscale, pictures partly outside the canvas. Multiview sizes included.
fn planar_cases() -> Vec<(
    (usize, usize),
    (u32, u32, u32, u32),
    PixelRect,
    &'static str,
)> {
    let full = (0, 0, 1920, 1080);
    vec![
        ((1920, 1080), full, rect(0, 0, 1920, 1080), "planar420-same"),
        (
            (1920, 1080),
            full,
            rect(-300, 200, 1920, 1080),
            "planar420-same",
        ),
        (
            (1920, 1080),
            (100, 60, 1820, 1020),
            rect(98, 60, 1720, 960),
            "planar420-same",
        ),
        ((1920, 1080), full, rect(960, 0, 960, 540), "planar420"),
        ((1920, 1080), full, rect(0, 540, 480, 270), "planar420"),
        // Multiview thumbnails: 8:1 across, 1080 → 134 or 136 lines.
        ((1920, 1080), full, rect(240, 744, 240, 134), "planar420"),
        ((1920, 1080), full, rect(1680, 946, 240, 136), "planar420"),
        // 1:3 and 1:6 (box widths without a SIMD kernel), 1:5 by an area
        // filter vertically.
        ((1920, 1080), full, rect(0, 0, 640, 360), "planar420"),
        ((1920, 1080), full, rect(640, 360, 320, 180), "planar420"),
        ((1920, 1080), full, rect(100, 100, 384, 210), "planar420"),
        // Vertical only, and a vertical upscale.
        ((1920, 1080), full, rect(0, 270, 1920, 540), "planar420"),
        (
            (1280, 720),
            (0, 0, 1280, 720),
            rect(0, 0, 640, 1000),
            "planar420",
        ),
        // Crop and scale, partly outside the canvas.
        (
            (1920, 1080),
            (240, 136, 1680, 944),
            rect(1500, 800, 720, 404),
            "planar420",
        ),
        ((1920, 1080), full, rect(-200, -100, 480, 270), "planar420"),
        // An odd crop puts chroma between samples: general path.
        (
            (1920, 1080),
            (241, 0, 1681, 1080),
            rect(0, 0, 720, 540),
            "general",
        ),
    ]
}

/// The planar 4:2:0 kernels against the general path (without decimation),
/// NV12 and I420 onto both 4:2:0 canvases, opaque and at half opacity.
///
/// Tolerance: 1 level. Box filters of power-of-two areas are exact on both;
/// otherwise the general path rounds its horizontal sums to Q6, while the
/// kernels round an area filter's vertical sums to Q4 and divide by
/// multiplying with a 15-bit reciprocal (within 0.07 of the exact quotient).
#[test]
fn planar420_kernels_match_the_general_path() {
    for format in [SrcFormat::Nv12, SrcFormat::I420] {
        let mut sources = std::collections::HashMap::new();
        for (size, crop, pic, path) in planar_cases() {
            let src = sources
                .entry(size)
                .or_insert_with(|| OwnedSource::new(format, size.0, size.1, 11));
            for canvas_format in [CanvasFormat::Nv12, CanvasFormat::I420] {
                for opacity in [255, 128] {
                    let mut fast = fresh(canvas_format, 1920, 1080);
                    let key = key_for(src, &fast, crop, pic);
                    let kernel = Prepared::with_options(key.clone(), true, false);
                    assert_eq!(kernel.path(), path, "{format:?} {crop:?} -> {pic:?}");
                    draw_one(src, &mut fast, &kernel, opacity, None);
                    let mut general = fresh(canvas_format, 1920, 1080);
                    draw_one(src, &mut general, &Prepared::general(key), opacity, None);
                    let d = max_diff(&fast, &general);
                    assert!(
                        d <= 1,
                        "{format:?} on {canvas_format:?} {crop:?} -> {pic:?} at {opacity}: differs by {d}"
                    );
                }
            }
        }
    }
}

/// Decimated vertical downscales read every `vstep`-th row (the middle one
/// of each block): the same picture as the general path drawing a source
/// made of just those rows, within the tolerance above.
#[test]
fn decimated_downscales_read_every_nth_row() {
    use super::render::vertical_step;
    // 1080 lines → thumbnails (8:1, vstep 2), 108 lines (10:1, 2), 66 (16:1, 4)
    for (pic, vstep) in [
        (rect(0, 0, 240, 134), 2),
        (rect(240, 136, 240, 136), 2),
        (rect(0, 0, 192, 108), 2),
        (rect(64, 64, 120, 66), 4),
    ] {
        assert_eq!(vertical_step(1080.0 / pic.height as f64), vstep);
        for format in [SrcFormat::Nv12, SrcFormat::I420] {
            let src = OwnedSource::new(format, 1920, 1080, 5);
            // The rows read, as a frame of their own.
            let mut rows = src.copy_frame();
            let phase = (vstep - 1) / 2;
            rows.height = 1080 / vstep;
            let spec = format.spec();
            for p in 0..spec.planes {
                let stride = src.strides[p];
                let n = src.height.div_ceil(spec.vsub[p] as usize) / vstep;
                rows.planes[p] = (0..n)
                    .flat_map(|i| {
                        let r = i * vstep + phase;
                        src.planes[p][r * stride..(r + 1) * stride].iter().copied()
                    })
                    .collect();
            }
            for canvas_format in [CanvasFormat::Nv12, CanvasFormat::I420] {
                let mut fast = fresh(canvas_format, 1920, 1080);
                let key = key_for(&src, &fast, (0, 0, 1920, 1080), pic);
                let kernel = Prepared::with_options(key, true, true);
                assert_eq!(kernel.path(), "planar420");
                draw_one(&src, &mut fast, &kernel, 255, None);
                let mut want = fresh(canvas_format, 1920, 1080);
                let key = key_for(&rows, &want, (0, 0, 1920, rows.height as u32), pic);
                draw_one(&rows, &mut want, &Prepared::general(key), 255, None);
                let d = max_diff(&fast, &want);
                assert!(
                    d <= 1,
                    "{format:?} on {canvas_format:?} -> {pic:?}: differs by {d}"
                );
            }
        }
    }
}

/// Strips, the pixel cache and covered parts (several visible rectangles)
/// with the planar kernels: the same picture as one plain draw.
#[test]
fn planar420_strips_cache_and_occlusion() {
    let pool = super::pool::StripPool::new(3);
    for format in [SrcFormat::Nv12, SrcFormat::I420] {
        let src = OwnedSource::new(format, 1920, 1080, 9);
        for (crop, pic) in [
            ((0, 0, 1920, 1080), rect(100, 50, 1920, 1080)),
            ((0, 0, 1920, 1080), rect(10, 6, 480, 270)),
            ((0, 0, 1920, 1080), rect(10, 6, 240, 134)),
        ] {
            for canvas_format in [CanvasFormat::Nv12, CanvasFormat::I420] {
                let (cw, ch) = (1280, 720);
                let prepared =
                    Prepared::new(key_for(&src, &fresh(canvas_format, cw, ch), crop, pic));
                assert_ne!(prepared.path(), "general");
                let mut one = fresh(canvas_format, cw, ch);
                draw_one(&src, &mut one, &prepared, 200, None);
                // A covered middle: the visible rectangles around it.
                let whole = pic.clip(cw as u32, ch as u32);
                let hole = rect(
                    whole.x + 40,
                    whole.y + 20,
                    (whole.width / 2) & !1,
                    (whole.height / 2) & !1,
                );
                let visible = whole.subtract(&hole);
                assert!(visible.len() > 1);
                let mut cache = PixelCache::new(&prepared, cw, ch);
                let view = super::render::cache_view(&mut cache);
                for mode in 0..3 {
                    let mut out = fresh(canvas_format, cw, ch);
                    let plan = Plan {
                        background: Vec::new(),
                        items: vec![Item::Layer {
                            index: 0,
                            picture: pic,
                            visible: visible.clone(),
                            opacity: 200,
                        }],
                    };
                    let c = out.canvas();
                    let frame = Frame {
                        canvas: &c,
                        plan: &plan,
                        layers: vec![Some(LayerInput {
                            source: src.frame(),
                            prepared: &prepared,
                            cache: match mode {
                                0 => CacheMode::Off,
                                1 => CacheMode::Build(&view),
                                _ => CacheMode::Use(&view),
                            },
                        })],
                        background: None,
                    };
                    render(&frame, Some(&pool));
                    drop(frame);
                    // Outside the hole as drawn whole; the hole untouched.
                    let mut want = fresh(canvas_format, cw, ch);
                    let hy = hole.y as usize..(hole.y + hole.height) as usize;
                    let hx = hole.x as usize..(hole.x + hole.width) as usize;
                    for y in 0..ch {
                        for x in 0..cw {
                            let i = y * want.strides[0] + x;
                            if !(hy.contains(&y) && hx.contains(&x)) {
                                want.planes[0][i] = one.planes[0][i];
                            }
                        }
                    }
                    assert_eq!(
                        out.planes[0], want.planes[0],
                        "{format:?} on {canvas_format:?} -> {pic:?} mode {mode}: luma"
                    );
                    // Chroma: the same outside the hole's 2×2 blocks.
                    for p in 1..3 {
                        if out.planes[p].is_empty() {
                            continue;
                        }
                        let stride = out.strides[p];
                        let per = if canvas_format == CanvasFormat::Nv12 {
                            2
                        } else {
                            1
                        };
                        for cy in 0..ch / 2 {
                            for b in 0..stride {
                                let cx = b / per;
                                let inside = hy.contains(&(2 * cy)) && hx.contains(&(2 * cx));
                                let i = cy * stride + b;
                                let expect = if inside { 128 } else { one.planes[p][i] };
                                if b < cw / 2 * per {
                                    assert_eq!(
                                        out.planes[p][i], expect,
                                        "{format:?} on {canvas_format:?} -> {pic:?} mode {mode}: plane {p} ({b},{cy})"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Planar 4:2:0 sources at the sizes compositions draw them (full screen,
/// halves, quarters, thirds, multiview thumbnails) take a dedicated kernel
/// on both 4:2:0 canvases; packed and RGB sources keep theirs.
#[test]
fn planar_sources_take_dedicated_kernels() {
    for (sw, sh) in [(1920, 1080), (1280, 720)] {
        for format in [SrcFormat::Nv12, SrcFormat::I420] {
            let src = OwnedSource::new(format, sw, sh, 1);
            for cf in [CanvasFormat::Nv12, CanvasFormat::I420] {
                let canvas = OwnedCanvas::new(cf, 1920, 1080);
                // (A third of 1280 isn't whole: 720p has no thirds.)
                let thirds = if sw == 1920 { &[(640, 360)][..] } else { &[] };
                let sizes = [(1920, 1080), (960, 540), (480, 270), (240, 134), (240, 136)];
                for &(w, h) in sizes.iter().chain(thirds) {
                    let (w, h) = (w * sw / 1920, (h * sh / 1080) & !1);
                    let p = Prepared::new(key_for(
                        &src,
                        &canvas,
                        (0, 0, sw as u32, sh as u32),
                        rect(0, 0, w as i32, h as i32),
                    ));
                    assert_ne!(
                        p.path(),
                        "general",
                        "{format:?} {sw}x{sh} -> {w}x{h} on {cf:?}"
                    );
                }
            }
        }
    }
    let uyvy = OwnedSource::new(SrcFormat::Uyvy, 1920, 1080, 1);
    let canvas = OwnedCanvas::new(CanvasFormat::Nv12, 1920, 1080);
    let p = Prepared::new(key_for(
        &uyvy,
        &canvas,
        (0, 0, 1920, 1080),
        rect(0, 0, 480, 270),
    ));
    assert_eq!(p.path(), "packed422");
}


//! Test and benchmark helpers: synthetic source frames in every format,
//! owned canvases, and a micro-benchmark (`cargo run --release --example
//! mixer_bench`).

// Index loops over several slices at once are intentional here.
#![allow(clippy::needless_range_loop)]

use std::time::Instant;

use super::config::PixelRect;
use super::color::{BT709, Model, argb_to};
use super::formats::SrcFormat;
use super::layout::{Item, Plan};
use super::pool::StripPool;
use super::render::{
    CacheMode, Canvas, CanvasFormat, Frame, LayerInput, LayerKey, Prepared, SourceFrame, render,
};

/// Native sample values of a frame (8-bit, or 10-bit for v210/P010) and
/// the frame packed in its format.
pub struct OwnedSource {
    pub format: SrcFormat,
    pub width: usize,
    pub height: usize,
    /// Channels 0–2 at their own resolution, then alpha (full resolution;
    /// 255 when the format has none)
    pub native: [Vec<u16>; 4],
    pub native_dims: [(usize, usize); 4],
    pub planes: [Vec<u8>; 3],
    pub strides: [usize; 3],
    pub model: Model,
}

/// Small deterministic generator.
pub struct Lcg(pub u64);

impl Lcg {
    pub fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
}

impl OwnedSource {
    /// A frame with gradients, noise and (for alpha formats) blocks of fully
    /// transparent, fully opaque and partly transparent pixels.
    pub fn new(format: SrcFormat, width: usize, height: usize, seed: u64) -> OwnedSource {
        let spec = format.spec();
        let ten = matches!(format, SrcFormat::V210 | SrcFormat::P010);
        let max: u32 = if ten { 1023 } else { 255 };
        let mut rng = Lcg(seed);
        let dims: [(usize, usize); 4] = std::array::from_fn(|c| {
            if c == 3 {
                (width, height)
            } else {
                let ch = spec.chans[c];
                let vs = spec.vsub[ch.plane] as usize;
                (width.div_ceil(ch.hsub as usize), height.div_ceil(vs))
            }
        });
        let native: [Vec<u16>; 4] = std::array::from_fn(|c| {
            let (w, h) = dims[c];
            let mut v = Vec::with_capacity(w * h);
            for y in 0..h {
                for x in 0..w {
                    let val = if c == 3 {
                        if spec.alpha.is_none() {
                            255
                        } else {
                            match (x / 8 + y / 8) % 4 {
                                0 => 0,
                                1 => 255,
                                _ => rng.next_u32() % 256,
                            }
                        }
                    } else {
                        let grad = (x * 7 + y * 3 + c * 50) as u32 % (max + 1);
                        let noise = rng.next_u32() % (max / 4 + 1);
                        (grad / 2 + noise + max / 8).min(max)
                    };
                    v.push(val as u16);
                }
            }
            v
        });
        let mut src = OwnedSource {
            format,
            width,
            height,
            native,
            native_dims: dims,
            planes: Default::default(),
            strides: [0; 3],
            model: format.model(BT709, false),
        };
        src.pack();
        src
    }

    fn get(&self, c: usize, x: usize, y: usize) -> u16 {
        let (w, h) = self.native_dims[c];
        self.native[c][y.min(h - 1) * w + x.min(w - 1)]
    }

    /// Pack `native` into `planes` again (after changing samples).
    pub fn pack(&mut self) {
        let (w, h) = (self.width, self.height);
        let pad = 16;
        let cw = w.div_ceil(2);
        let ch = h.div_ceil(2);
        let b = |v: u16| v as u8;
        match self.format {
            SrcFormat::Uyvy | SrcFormat::Yuy2 => {
                let stride = cw * 4 + pad;
                let mut p = vec![0u8; stride * h];
                for y in 0..h {
                    for i in 0..cw {
                        let (u, v) = (b(self.get(1, i, y)), b(self.get(2, i, y)));
                        let (y0, y1) = (b(self.get(0, 2 * i, y)), b(self.get(0, 2 * i + 1, y)));
                        let o = y * stride + i * 4;
                        let bytes = if self.format == SrcFormat::Uyvy {
                            [u, y0, v, y1]
                        } else {
                            [y0, u, y1, v]
                        };
                        p[o..o + 4].copy_from_slice(&bytes);
                    }
                }
                self.planes[0] = p;
                self.strides[0] = stride;
            }
            SrcFormat::V210 => {
                let groups = w.div_ceil(6);
                let stride = (groups * 16).div_ceil(128) * 128;
                let mut p = vec![0u8; stride * h];
                for y in 0..h {
                    for g in 0..groups {
                        let yv = |i: usize| self.get(0, g * 6 + i, y) as u32;
                        let u = |i: usize| self.get(1, g * 3 + i, y) as u32;
                        let v = |i: usize| self.get(2, g * 3 + i, y) as u32;
                        let words = [
                            u(0) | yv(0) << 10 | v(0) << 20,
                            yv(1) | u(1) << 10 | yv(2) << 20,
                            v(1) | yv(3) << 10 | u(2) << 20,
                            yv(4) | v(2) << 10 | yv(5) << 20,
                        ];
                        for (k, word) in words.iter().enumerate() {
                            let o = y * stride + g * 16 + k * 4;
                            p[o..o + 4].copy_from_slice(&word.to_le_bytes());
                        }
                    }
                }
                self.planes[0] = p;
                self.strides[0] = stride;
            }
            SrcFormat::Nv12 | SrcFormat::P010 => {
                let bps = if self.format == SrcFormat::P010 { 2 } else { 1 };
                let put = |p: &mut [u8], o: usize, v: u16| {
                    if bps == 2 {
                        p[o..o + 2].copy_from_slice(&(v << 6).to_le_bytes());
                    } else {
                        p[o] = v as u8;
                    }
                };
                let s0 = w * bps + pad;
                let mut p0 = vec![0u8; s0 * h];
                for y in 0..h {
                    for x in 0..w {
                        put(&mut p0, y * s0 + x * bps, self.get(0, x, y));
                    }
                }
                let s1 = cw * 2 * bps + pad;
                let mut p1 = vec![0u8; s1 * ch];
                for y in 0..ch {
                    for i in 0..cw {
                        put(&mut p1, y * s1 + i * 2 * bps, self.get(1, i, y));
                        put(&mut p1, y * s1 + (i * 2 + 1) * bps, self.get(2, i, y));
                    }
                }
                self.planes = [p0, p1, Vec::new()];
                self.strides = [s0, s1, 0];
            }
            SrcFormat::I420 => {
                let s0 = w + pad;
                let s1 = cw + pad;
                let mut planes = [vec![0u8; s0 * h], vec![0u8; s1 * ch], vec![0u8; s1 * ch]];
                for y in 0..h {
                    for x in 0..w {
                        planes[0][y * s0 + x] = b(self.get(0, x, y));
                    }
                }
                for c in 1..3 {
                    for y in 0..ch {
                        for x in 0..cw {
                            planes[c][y * s1 + x] = b(self.get(c, x, y));
                        }
                    }
                }
                self.planes = planes;
                self.strides = [s0, s1, s1];
            }
            _ => {
                let spec = self.format.spec();
                let stride = w * 4 + pad;
                let mut p = vec![0u8; stride * h];
                let mut rng = Lcg(99);
                for y in 0..h {
                    for x in 0..w {
                        let o = y * stride + x * 4;
                        // Padding byte of the x formats: junk, never read.
                        p[o..o + 4].fill(rng.next_u32() as u8);
                        // Ports carry premultiplied alpha.
                        for c in 0..3 {
                            p[o + spec.chans[c].offset] = self.premul8(c, x, y) as u8;
                        }
                        if let Some(a) = spec.alpha {
                            p[o + a.offset] = b(self.get(3, x, y));
                        }
                    }
                }
                self.planes[0] = p;
                self.strides[0] = stride;
            }
        }
    }

    /// A copy of the packed frame only (in new memory; no samples).
    pub fn copy_frame(&self) -> OwnedSource {
        OwnedSource {
            format: self.format,
            width: self.width,
            height: self.height,
            native: Default::default(),
            native_dims: self.native_dims,
            planes: self.planes.clone(),
            strides: self.strides,
            model: self.model,
        }
    }

    /// 8-bit value of channel `c` (rounded from 10 bits).
    pub fn value8(&self, c: usize, x: usize, y: usize) -> f64 {
        let v = self.get(c, x, y) as u32;
        if matches!(self.format, SrcFormat::V210 | SrcFormat::P010) && c < 3 {
            ((v + 2) >> 2).min(255) as f64
        } else {
            v as f64
        }
    }

    /// Sample of channel `c` as stored in the frame: premultiplied by the
    /// alpha (`round(c · a / 255)`) for formats with alpha.
    pub fn premul8(&self, c: usize, x: usize, y: usize) -> f64 {
        let v = self.value8(c, x, y);
        if c < 3 && self.format.has_alpha() {
            (v * self.value8(3, x, y) / 255.0).round()
        } else {
            v
        }
    }

    pub fn frame(&self) -> SourceFrame<'_> {
        SourceFrame {
            format: self.format,
            width: self.width,
            height: self.height,
            planes: [&self.planes[0], &self.planes[1], &self.planes[2]],
            strides: self.strides,
            model: self.model,
        }
    }
}

/// An output frame owned by the test.
pub struct OwnedCanvas {
    pub format: CanvasFormat,
    pub width: usize,
    pub height: usize,
    pub planes: [Vec<u8>; 3],
    pub strides: [usize; 3],
    pub model: Model,
}

impl OwnedCanvas {
    pub fn new(format: CanvasFormat, width: usize, height: usize) -> OwnedCanvas {
        let (cw, ch) = (width.div_ceil(2), height.div_ceil(2));
        let (planes, strides) = match format {
            CanvasFormat::Nv12 => (
                [vec![0; width * height], vec![0; cw * 2 * ch], Vec::new()],
                [width, cw * 2, 0],
            ),
            CanvasFormat::I420 => (
                [vec![0; width * height], vec![0; cw * ch], vec![0; cw * ch]],
                [width, cw, cw],
            ),
            CanvasFormat::Bgra => (
                [vec![0; width * 4 * height], Vec::new(), Vec::new()],
                [width * 4, 0, 0],
            ),
        };
        OwnedCanvas {
            format,
            width,
            height,
            planes,
            strides,
            model: if format == CanvasFormat::Bgra {
                Model::Rgb
            } else {
                Model::yuv(BT709)
            },
        }
    }

    pub fn canvas(&mut self) -> Canvas {
        let ptrs = [0, 1, 2].map(|i| (self.planes[i].as_mut_ptr(), self.planes[i].len()));
        // SAFETY: the planes outlive the canvas in every use here.
        unsafe {
            Canvas::new(
                self.format,
                self.width,
                self.height,
                self.model,
                ptrs,
                self.strides,
            )
        }
    }
}

/// The key of a layer drawn at `picture` with the whole source.
pub fn key_for(
    src: &OwnedSource,
    canvas: &OwnedCanvas,
    crop: (u32, u32, u32, u32),
    picture: PixelRect,
) -> LayerKey {
    LayerKey {
        format: src.format,
        width: src.width,
        height: src.height,
        model: src.model,
        crop,
        picture,
        canvas: canvas.format,
        canvas_model: canvas.model,
    }
}

/// Draw one layer (no background) and return the time it took.
pub fn draw_one(
    src: &OwnedSource,
    canvas: &mut OwnedCanvas,
    prepared: &Prepared,
    opacity: u8,
    pool: Option<&StripPool>,
) -> std::time::Duration {
    let picture = prepared.key.picture;
    let plan = Plan {
        background: Vec::new(),
        items: vec![Item::Layer {
            index: 0,
            picture,
            visible: vec![picture.clip(canvas.width as u32, canvas.height as u32)],
            opacity,
        }],
    };
    let c = canvas.canvas();
    let frame = Frame {
        canvas: &c,
        plan: &plan,
        layers: vec![Some(LayerInput {
            source: src.frame(),
            prepared,
            cache: CacheMode::Off,
        })],
        background: Some(argb_to(c.model, 0xFF00_0000)),
    };
    let t = Instant::now();
    render(&frame, pool);
    t.elapsed()
}

/// Milliseconds per layer, single-threaded, for each source format:
/// `source` → `target` on a 1080p NV12 canvas (BGRA with
/// `MIXER_BENCH_CANVAS=bgra`). Prints a table.
pub fn bench(
    source: (usize, usize),
    target: (usize, usize),
    frames: usize,
    filter: &str,
) -> Vec<(SrcFormat, f64)> {
    let mut results = Vec::new();
    for format in SrcFormat::ALL {
        if !format!("{format:?}").to_lowercase().contains(filter) {
            continue;
        }
        let src = OwnedSource::new(format, source.0, source.1, 1);
        // MIXER_BENCH_CANVAS=bgra: the transparent canvas.
        let canvas_format = if std::env::var("MIXER_BENCH_CANVAS").as_deref() == Ok("bgra") {
            CanvasFormat::Bgra
        } else {
            CanvasFormat::Nv12
        };
        let mut canvas = OwnedCanvas::new(canvas_format, 1920, 1080);
        let picture = PixelRect {
            x: 0,
            y: 0,
            width: target.0 as i32,
            height: target.1 as i32,
        };
        let prepared = Prepared::new(key_for(
            &src,
            &canvas,
            (0, 0, source.0 as u32, source.1 as u32),
            picture,
        ));
        draw_one(&src, &mut canvas, &prepared, 255, None);
        let mut best = f64::MAX;
        let mut total = 0.0;
        for _ in 0..frames {
            let ms = draw_one(&src, &mut canvas, &prepared, 255, None).as_secs_f64() * 1000.0;
            best = best.min(ms);
            total += ms;
        }
        println!(
            "{:>5?} {}x{} -> {}x{}: {:.3} ms avg, {:.3} ms best",
            format,
            source.0,
            source.1,
            target.0,
            target.1,
            total / frames as f64,
            best
        );
        results.push((format, total / frames as f64));
    }
    results
}

/// A web page as a 1080p BGRA frame: transparent, with a lower third that
/// is 60% opaque (about 7% of the pixels), like the measurements in
/// docs/VIDEO_COMPOSITION.md.
pub fn lower_third(width: usize, height: usize) -> OwnedSource {
    let mut src = OwnedSource::new(SrcFormat::Bgra, width, height, 5);
    let (w, h) = src.native_dims[3];
    for y in 0..h {
        for x in 0..w {
            let inside = y > h * 3 / 4 && y < h * 3 / 4 + h / 8 && x > w / 10 && x < w * 7 / 10;
            src.native[3][y * w + x] = if inside { 153 } else { 0 };
        }
    }
    src.pack();
    src
}

/// Whole frames: `layers` (source, layer box as fractions, opacity) on a
/// 1080p canvas, with occlusion; prints ms per frame single-threaded and
/// on `pool`. `static_last`: the last layer's frame doesn't change (drawn
/// from the cache after the second frame).
pub fn bench_scenario(
    name: &str,
    layers: &[(&OwnedSource, [f64; 4], f64)],
    canvas_format: CanvasFormat,
    static_last: bool,
    frames: usize,
    pool: &StripPool,
) {
    use super::config::{CompositionBackground, CompositionLayer, LayerFit};
    use super::layout::{MixerLayer, MixerLayout, SourceGeom, plan};
    use super::render::{PixelCache, cache_view};

    let (cw, ch) = (1920usize, 1080usize);
    let layout = MixerLayout {
        layers: layers
            .iter()
            .enumerate()
            .map(|(i, (_, b, alpha))| MixerLayer {
                pad: format!("sink_{i}"),
                shown: true,
                layer: CompositionLayer {
                    id: format!("l{i}"),
                    x: b[0],
                    y: b[1],
                    w: b[2],
                    h: b[3],
                    alpha: *alpha,
                    fit: LayerFit::Stretch,
                    ..Default::default()
                },
                ..Default::default()
            })
            .collect(),
        background: if canvas_format == CanvasFormat::Bgra {
            CompositionBackground::Transparent
        } else {
            CompositionBackground::Black
        },
    };
    let geoms: Vec<Option<SourceGeom>> = layers
        .iter()
        .map(|(s, _, _)| {
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
    let mut canvas = OwnedCanvas::new(canvas_format, cw, ch);
    let prepared: Vec<Option<Prepared>> = (0..layers.len())
        .map(|i| {
            p.items.iter().find_map(|item| match item {
                Item::Layer { index, picture, .. } if *index == i => {
                    let s = layers[i].0;
                    Some(Prepared::new(key_for(
                        s,
                        &canvas,
                        (0, 0, s.width as u32, s.height as u32),
                        *picture,
                    )))
                }
                _ => None,
            })
        })
        .collect();
    let last = layers.len() - 1;
    let mut cache = prepared[last]
        .as_ref()
        .map(|p| PixelCache::new(p, canvas.width, canvas.height));
    let mut run = |pool: Option<&StripPool>| {
        let mut total = 0.0;
        for f in 0..frames + 2 {
            let view = cache.as_mut().map(cache_view);
            let c = canvas.canvas();
            let frame = Frame {
                canvas: &c,
                plan: &p,
                layers: prepared
                    .iter()
                    .enumerate()
                    .map(|(i, pr)| {
                        pr.as_ref().map(|pr| LayerInput {
                            source: layers[i].0.frame(),
                            prepared: pr,
                            cache: match (&view, static_last && i == last, f) {
                                (Some(v), true, 1) => CacheMode::Build(v),
                                (Some(v), true, 2..) => CacheMode::Use(v),
                                _ => CacheMode::Off,
                            },
                        })
                    })
                    .collect(),
                background: (canvas_format != CanvasFormat::Bgra)
                    .then(|| argb_to(c.model, 0xFF00_0000)),
            };
            let t = Instant::now();
            render(&frame, pool);
            if f >= 2 {
                total += t.elapsed().as_secs_f64() * 1000.0;
            }
        }
        total / frames as f64
    };
    let single = run(None);
    let threaded = run(Some(pool));
    println!(
        "{name}: {single:.2} ms/frame on one thread, {threaded:.2} ms on {} threads",
        pool.threads()
    );
}

/// How [`bench_scene`] feeds its sources.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Inputs {
    /// The same buffers every frame (they stay in the caches)
    Warm,
    /// Rotating through copies of the sources, more than the L3 cache in
    /// total: every frame reads buffers it last saw several frames ago
    Cold,
    /// New buffers every frame, allocated and written outside the timing
    /// (like a source that allocates each frame); the previous frame's are
    /// freed inside it, timed separately (like the mixer dropping the
    /// buffer it held)
    Fresh,
}

/// Result of [`bench_scene`] (ms per frame).
#[derive(Debug, Clone, Copy)]
pub struct SceneResult {
    pub avg: f64,
    pub p50: f64,
    pub p90: f64,
    pub best: f64,
    /// Freeing the previous frame's buffers ([`Inputs::Fresh`])
    pub free: f64,
    /// Effective clock (GHz) right before and right after drawing
    pub ghz_before: f64,
    pub ghz_after: f64,
}

/// Effective core clock in GHz: a loop of one macro-fused `dec`/`jnz` per
/// cycle, ~20 µs (0 off x86_64).
pub fn effective_ghz() -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        let n: u64 = 80_000;
        let t = Instant::now();
        // SAFETY: a register-only loop.
        unsafe {
            std::arch::asm!("2:", "dec {0}", "jnz 2b", inout(reg) n => _, options(nomem, nostack));
        }
        n as f64 / t.elapsed().as_secs_f64() / 1e9
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        0.0
    }
}

/// Whole frames like the app draws them: `layers` are (index into
/// `sources`, layer box as fractions) on a 1080p canvas with a black (or,
/// for BGRA, transparent) background, drawn on one thread. `canvases`
/// output frames are used in turn (like the output buffer pool),
/// `paced`: wait for the next tick of this interval between frames (like a
/// live 50 fps output, with the thread idle in between).
pub fn bench_scene(
    sources: &[OwnedSource],
    layers: &[(usize, [f64; 4])],
    canvas_format: CanvasFormat,
    frames: usize,
    inputs: Inputs,
    canvases: usize,
    paced: Option<std::time::Duration>,
) -> SceneResult {
    use super::config::{CompositionBackground, CompositionLayer, LayerFit};
    use super::layout::{MixerLayer, MixerLayout, SourceGeom, plan};

    let (cw, ch) = (1920usize, 1080usize);
    let layout = MixerLayout {
        layers: layers
            .iter()
            .enumerate()
            .map(|(i, (_, b))| MixerLayer {
                pad: format!("sink_{i}"),
                shown: true,
                layer: CompositionLayer {
                    id: format!("l{i}"),
                    x: b[0],
                    y: b[1],
                    w: b[2],
                    h: b[3],
                    fit: LayerFit::Stretch,
                    ..Default::default()
                },
                ..Default::default()
            })
            .collect(),
        background: if canvas_format == CanvasFormat::Bgra {
            CompositionBackground::Transparent
        } else {
            CompositionBackground::Black
        },
    };
    let geoms: Vec<Option<SourceGeom>> = layers
        .iter()
        .map(|(s, _)| {
            let s = &sources[*s];
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
    let mut outs: Vec<OwnedCanvas> = (0..canvases.max(1))
        .map(|_| OwnedCanvas::new(canvas_format, cw, ch))
        .collect();
    let prepared: Vec<Option<Prepared>> = (0..layers.len())
        .map(|i| {
            p.items.iter().find_map(|item| match item {
                Item::Layer { index, picture, .. } if *index == i => {
                    let s = &sources[layers[i].0];
                    Some(Prepared::new(key_for(
                        s,
                        &outs[0],
                        (0, 0, s.width as u32, s.height as u32),
                        *picture,
                    )))
                }
                _ => None,
            })
        })
        .collect();
    // Copies of the whole source set: more than 32 MB for `Cold`.
    let set_bytes: usize = sources
        .iter()
        .map(|s| s.planes.iter().map(Vec::len).sum::<usize>())
        .sum();
    let copies = match inputs {
        Inputs::Cold => (32 << 20) / set_bytes.max(1) + 2,
        _ => 1,
    };
    let copy_set = || {
        sources
            .iter()
            .map(OwnedSource::copy_frame)
            .collect::<Vec<_>>()
    };
    let mut sets: Vec<Vec<OwnedSource>> = (0..copies).map(|_| copy_set()).collect();
    let mut previous: Option<Vec<OwnedSource>> = None;
    let mut times = Vec::with_capacity(frames);
    let (mut free, mut ghz_b, mut ghz_a) = (0.0, 0.0, 0.0);
    let start = Instant::now();
    for f in 0..frames + 2 {
        if let Some(iv) = paced {
            let next = iv * (f as u32 + 1);
            let now = start.elapsed();
            if next > now {
                std::thread::sleep(next - now);
            }
        }
        let set = match inputs {
            Inputs::Fresh => {
                // New buffers, written now (another thread in the app).
                let fresh = copy_set();
                previous = Some(std::mem::replace(&mut sets[0], fresh));
                0
            }
            _ => f % copies,
        };
        let gb = effective_ghz();
        let t = Instant::now();
        if let Some(old) = previous.take() {
            drop(old);
        }
        let t_free = t.elapsed().as_secs_f64() * 1000.0;
        let canvas = &mut outs[f % canvases.max(1)];
        let c = canvas.canvas();
        let frame = Frame {
            canvas: &c,
            plan: &p,
            layers: prepared
                .iter()
                .enumerate()
                .map(|(i, pr)| {
                    pr.as_ref().map(|pr| LayerInput {
                        source: sets[set][layers[i].0].frame(),
                        prepared: pr,
                        cache: CacheMode::Off,
                    })
                })
                .collect(),
            background: (canvas_format != CanvasFormat::Bgra)
                .then(|| argb_to(c.model, 0xFF00_0000)),
        };
        let t = Instant::now();
        render(&frame, None);
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        let ga = effective_ghz();
        if f >= 2 {
            times.push(ms);
            free += t_free;
            ghz_b += gb;
            ghz_a += ga;
        }
    }
    let n = frames as f64;
    let avg = times.iter().sum::<f64>() / n;
    times.sort_by(|a, b| a.total_cmp(b));
    SceneResult {
        avg,
        p50: times[times.len() / 2],
        p90: times[times.len() * 9 / 10],
        best: times[0],
        free: free / n,
        ghz_before: ghz_b / n,
        ghz_after: ghz_a / n,
    }
}

/// Result of [`bench_compositions`] for one composition (ms per frame).
#[derive(Debug, Clone)]
pub struct CompositionResult {
    pub name: String,
    pub avg: f64,
    pub p90: f64,
    /// Layers per kernel ([`Prepared::path`]), e.g. `[("general", 16)]`
    pub paths: Vec<(&'static str, usize)>,
    /// Picture sizes drawn, with how many layers have each
    pub sizes: Vec<((i32, i32), usize)>,
}

/// A composition for [`bench_compositions`]: a name and its layers, as
/// (index into the sources, layer box as fractions).
pub type BenchComposition<'a> = (&'a str, Vec<(usize, [f64; 4])>);

/// CPU time of the whole process so far, in ms (all threads; Linux only,
/// 0 elsewhere).
pub fn process_cpu_ms() -> f64 {
    #[cfg(target_os = "linux")]
    {
        // utime and stime, fields 14 and 15, in clock ticks (100 per
        // second on every Linux we run on).
        let stat = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
        let after = stat.rsplit(')').next().unwrap_or("");
        let f: Vec<f64> = after
            .split_whitespace()
            .map(|v| v.parse().unwrap_or(0.0))
            .collect();
        if f.len() > 13 {
            return (f[11] + f[12]) * 10.0;
        }
        0.0
    }
    #[cfg(not(target_os = "linux"))]
    {
        0.0
    }
}

/// Several compositions drawn one after the other per output frame, like
/// the app does (each its own canvas), from live inputs: `sources` (all
/// distinct frames, the same size and format) rotated through copies of
/// more than 64 MB in total, so every frame reads its sources from DRAM
/// like the app's live sources at many inputs. Layers are (index into
/// `sources`, layer box as fractions) with the app's default fit,
/// letterbox, on a 1080p canvas with a black background. On `pool`
/// (strips) if given, else on one thread. Returns the ms per frame of
/// each composition and the process CPU time per frame (ms).
pub fn bench_compositions(
    sources: &[OwnedSource],
    comps: &[BenchComposition],
    canvas_format: CanvasFormat,
    frames: usize,
    pool: Option<&StripPool>,
) -> (Vec<CompositionResult>, f64) {
    use super::config::{CompositionBackground, CompositionLayer, LayerFit};
    use super::layout::{MixerLayer, MixerLayout, SourceGeom, plan};

    let (cw, ch) = (1920usize, 1080usize);
    let mut canvases: Vec<OwnedCanvas> = comps
        .iter()
        .map(|_| OwnedCanvas::new(canvas_format, cw, ch))
        .collect();
    let scenes: Vec<(Plan, Vec<Option<Prepared>>)> = comps
        .iter()
        .zip(&canvases)
        .map(|((_, layers), canvas)| {
            let layout = MixerLayout {
                layers: layers
                    .iter()
                    .enumerate()
                    .map(|(i, (_, b))| MixerLayer {
                        pad: format!("sink_{i}"),
                        shown: true,
                        layer: CompositionLayer {
                            id: format!("l{i}"),
                            x: b[0],
                            y: b[1],
                            w: b[2],
                            h: b[3],
                            fit: LayerFit::Letterbox,
                            ..Default::default()
                        },
                        ..Default::default()
                    })
                    .collect(),
                background: CompositionBackground::Black,
            };
            let geoms: Vec<Option<SourceGeom>> = layers
                .iter()
                .map(|(s, _)| {
                    let s = &sources[*s];
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
            let prepared = (0..layers.len())
                .map(|i| {
                    p.items.iter().find_map(|item| match item {
                        Item::Layer { index, picture, .. } if *index == i => {
                            let s = &sources[layers[i].0];
                            Some(Prepared::new(key_for(
                                s,
                                canvas,
                                (0, 0, s.width as u32, s.height as u32),
                                *picture,
                            )))
                        }
                        _ => None,
                    })
                })
                .collect();
            (p, prepared)
        })
        .collect();
    let set_bytes: usize = sources
        .iter()
        .map(|s| s.planes.iter().map(Vec::len).sum::<usize>())
        .sum();
    let copies = (64 << 20) / set_bytes.max(1) + 2;
    let sets: Vec<Vec<OwnedSource>> = (0..copies)
        .map(|_| sources.iter().map(OwnedSource::copy_frame).collect())
        .collect();
    let mut times: Vec<Vec<f64>> = vec![Vec::with_capacity(frames); comps.len()];
    let mut cpu0 = 0.0;
    for f in 0..frames + 2 {
        if f == 2 {
            cpu0 = process_cpu_ms();
        }
        let set = &sets[f % copies];
        for (ci, ((p, prepared), canvas)) in scenes.iter().zip(canvases.iter_mut()).enumerate() {
            let c = canvas.canvas();
            let frame = Frame {
                canvas: &c,
                plan: p,
                layers: prepared
                    .iter()
                    .enumerate()
                    .map(|(i, pr)| {
                        pr.as_ref().map(|pr| LayerInput {
                            source: set[comps[ci].1[i].0].frame(),
                            prepared: pr,
                            cache: CacheMode::Off,
                        })
                    })
                    .collect(),
                background: Some(argb_to(c.model, 0xFF00_0000)),
            };
            let t = Instant::now();
            render(&frame, pool);
            if f >= 2 {
                times[ci].push(t.elapsed().as_secs_f64() * 1000.0);
            }
        }
    }
    let cpu = (process_cpu_ms() - cpu0) / frames as f64;
    let results = comps
        .iter()
        .zip(&scenes)
        .zip(times)
        .map(|(((name, _), (_, prepared)), mut t)| {
            let avg = t.iter().sum::<f64>() / t.len() as f64;
            t.sort_by(|a, b| a.total_cmp(b));
            let mut paths: Vec<(&'static str, usize)> = Vec::new();
            let mut sizes: Vec<((i32, i32), usize)> = Vec::new();
            for pr in prepared.iter().flatten() {
                match paths.iter_mut().find(|(p, _)| *p == pr.path()) {
                    Some((_, n)) => *n += 1,
                    None => paths.push((pr.path(), 1)),
                }
                let size = (pr.key.picture.width, pr.key.picture.height);
                match sizes.iter_mut().find(|(s, _)| *s == size) {
                    Some((_, n)) => *n += 1,
                    None => sizes.push((size, 1)),
                }
            }
            CompositionResult {
                name: name.to_string(),
                avg,
                p90: t[t.len() * 9 / 10],
                paths,
                sizes,
            }
        })
        .collect();
    (results, cpu)
}

//! Drawing a frame into a v210 canvas at 10 bits.
//!
//! The same plan as the 8-bit renderer ([`super::layout::plan_snapped`],
//! even pixels horizontally): background rectangles, then per layer
//! (bottom-up) the parts not covered by opaque layers above, fills and
//! borders in the z-order. It is drawn per canvas row, in horizontal strips
//! on the [`StripPool`], straight into the v210 output (an MXL grain).
//!
//! Per row, each item writes only its own span of the row, picking the
//! cheapest way:
//!
//! - **Opaque, unscaled source**: the v210 words are copied
//!   ([`v210::copy`]); a cut is a `memcpy` of the grain.
//! - **Unscaled source at one level** (a mix, a dissolving keyer): the
//!   words are mixed in place ([`v210::mix`]), 3 samples per word, without
//!   unpacking.
//! - **Fills** (background, dips, borders, colour mattes): the 16-byte block
//!   of the colour is repeated, or mixed in.
//! - **Everything else** (keys with a level per pixel, soft wipes, scaled
//!   boxes, a picture moved by a number of pixels that isn't a multiple of
//!   3): the source span is unpacked to 16-bit samples (or filtered, for a
//!   scaled box), its levels made, and only the runs of 48 pixels that
//!   aren't fully transparent are unpacked from the canvas, blended and
//!   packed again; fully opaque runs are just packed.
//!
//! Scaling uses the filter tables ([`Table`]: area filter down,
//! bilinear up), vertical first (the source rows of an output row summed
//! with the vertical weights), then horizontal per channel.

// Index loops over several slices at once are intentional here.
#![allow(clippy::needless_range_loop)]

use std::cell::RefCell;

use super::config::PixelRect;
use super::kernels::{self, multiversion};
use super::scale10::{self, HTab};
use super::kernels10::{self, KeyShape, ONE};
use super::layout::{Item, Plan};
use super::pool::StripPool;
use super::scale::Table;
use super::v210;

/// A v210 (or v210a) source frame.
#[derive(Clone, Copy)]
pub struct Source10<'a> {
    pub data: &'a [u8],
    pub stride: usize,
    pub width: usize,
    pub height: usize,
    /// The key plane of a v210a frame: data and stride
    pub key: Option<(&'a [u8], usize)>,
}

impl<'a> Source10<'a> {
    /// A v210 frame (`with_key`: v210a, the key plane after the fill).
    pub fn new(data: &'a [u8], width: usize, height: usize, with_key: bool) -> Option<Self> {
        let stride = v210::line_bytes(width);
        if data.len() < v210::frame_bytes(width, height, with_key) {
            return None;
        }
        let key = with_key.then(|| (&data[stride * height..], v210::key_line_bytes(width)));
        Some(Self {
            data,
            stride,
            width,
            height,
            key,
        })
    }

    #[inline]
    pub fn row(&self, y: usize) -> &'a [u8] {
        &self.data[y * self.stride..(y + 1) * self.stride]
    }

    #[inline]
    fn key_row(&self, y: usize) -> Option<&'a [u8]> {
        self.key.map(|(k, s)| &k[y * s..(y + 1) * s])
    }
}

/// The output frame: v210 rows.
pub struct Canvas10 {
    pub width: usize,
    pub height: usize,
    pub stride: usize,
    ptr: *mut u8,
    len: usize,
}

// SAFETY: strips write disjoint rows only (see `row`).
unsafe impl Send for Canvas10 {}
unsafe impl Sync for Canvas10 {}

impl Canvas10 {
    /// A canvas over `data` (at least `v210::frame_bytes(width, height)`).
    pub fn new(data: &mut [u8], width: usize, height: usize) -> Canvas10 {
        let stride = v210::line_bytes(width);
        assert!(data.len() >= stride * height, "v210 canvas too small");
        Canvas10 {
            width,
            height,
            stride,
            ptr: data.as_mut_ptr(),
            len: data.len(),
        }
    }

    /// # Safety
    /// No two threads may hold the same row at once.
    #[allow(clippy::mut_from_ref)]
    unsafe fn row(&self, y: usize) -> &mut [u8] {
        let start = y * self.stride;
        assert!(start + self.stride <= self.len);
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(start), self.stride) }
    }
}

/// How a layer is keyed.
#[derive(Clone)]
pub enum Keying {
    /// Opaque (apart from the layer's opacity)
    None,
    /// The key plane of a v210a source
    Linear(KeyShape),
    /// A self key from the source's luma
    Luma(KeyShape),
}

/// A wipe's edge as a level per pixel: the layer shows where the pattern's
/// value is below the progress.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WipeMask {
    pub pattern: WipePattern,
    /// 0 – 1
    pub progress: f32,
    /// Width of the soft edge, as a fraction of the pattern's travel
    pub soft: f32,
    /// Grows from the other side
    pub reverse: bool,
}

/// Wipe shapes as a value per canvas position (0 where the wipe starts, 1
/// where it ends).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum WipePattern {
    /// Left to right
    #[default]
    Horizontal,
    /// Top to bottom
    Vertical,
    /// Two edges from the centre outwards, left and right
    BarnDoorVertical,
    /// Two edges from the centre outwards, up and down
    BarnDoorHorizontal,
    /// A box from the centre
    Box,
    /// A circle from the centre
    Iris,
    /// Top left to bottom right
    Diagonal,
}

impl WipePattern {
    pub const ALL: [WipePattern; 7] = [
        WipePattern::Horizontal,
        WipePattern::Vertical,
        WipePattern::BarnDoorVertical,
        WipePattern::BarnDoorHorizontal,
        WipePattern::Box,
        WipePattern::Iris,
        WipePattern::Diagonal,
    ];

    /// The pattern's value at canvas position (`u`, `v`) in 0–1.
    #[inline]
    pub fn value(self, u: f32, v: f32, aspect: f32) -> f32 {
        match self {
            WipePattern::Horizontal => u,
            WipePattern::Vertical => v,
            WipePattern::BarnDoorVertical => (u - 0.5).abs() * 2.0,
            WipePattern::BarnDoorHorizontal => (v - 0.5).abs() * 2.0,
            WipePattern::Box => ((u - 0.5).abs() * 2.0).max((v - 0.5).abs() * 2.0),
            WipePattern::Iris => {
                let (dx, dy) = ((u - 0.5) * aspect, v - 0.5);
                (dx * dx + dy * dy).sqrt() / (0.25 * aspect * aspect + 0.25).sqrt()
            }
            WipePattern::Diagonal => (u + v) * 0.5,
        }
    }

    /// The part of the canvas a hard wipe of this pattern shows at
    /// `progress`, if it is one rectangle (drawn with a clip instead of a
    /// mask: no per-pixel levels).
    pub fn clip_rect(self, progress: f32, reverse: bool, width: u32, height: u32) -> Option<PixelRect> {
        let p = progress.clamp(0.0, 1.0) as f64;
        let (w, h) = (width as f64, height as f64);
        let r = |x0: f64, y0: f64, x1: f64, y1: f64| PixelRect {
            x: x0.round() as i32,
            y: y0.round() as i32,
            width: (x1.round() - x0.round()) as i32,
            height: (y1.round() - y0.round()) as i32,
        };
        Some(match (self, reverse) {
            (WipePattern::Horizontal, false) => r(0.0, 0.0, w * p, h),
            (WipePattern::Horizontal, true) => r(w * (1.0 - p), 0.0, w, h),
            (WipePattern::Vertical, false) => r(0.0, 0.0, w, h * p),
            (WipePattern::Vertical, true) => r(0.0, h * (1.0 - p), w, h),
            (WipePattern::BarnDoorVertical, false) => {
                r(w * (0.5 - p / 2.0), 0.0, w * (0.5 + p / 2.0), h)
            }
            (WipePattern::BarnDoorHorizontal, false) => {
                r(0.0, h * (0.5 - p / 2.0), w, h * (0.5 + p / 2.0))
            }
            (WipePattern::Box, false) => r(
                w * (0.5 - p / 2.0),
                h * (0.5 - p / 2.0),
                w * (0.5 + p / 2.0),
                h * (0.5 + p / 2.0),
            ),
            _ => return None,
        })
    }
}

impl WipeMask {
    /// Q15 level at canvas pixel (`x`, `y`).
    pub fn level(&self, x: usize, y: usize, width: usize, height: usize) -> u16 {
        let mut out = [0u16; 1];
        self.row_levels(y, x, &mut out, width, height);
        out[0]
    }

    /// Q15 levels of pixels `x0..x0 + out.len()` of row `y`. One loop per
    /// pattern, so that each vectorises.
    pub fn row_levels(&self, y: usize, x0: usize, out: &mut [u16], width: usize, height: usize) {
        let (wf, hf) = (width as f32, height as f32);
        let (iw, ih) = (1.0 / wf, 1.0 / hf);
        let v = (y as f32 + 0.5) * ih;
        let aspect = wf * ih;
        let s = self.soft.max(0.0);
        let p = self.progress.clamp(0.0, 1.0);
        let top = p * (1.0 + s);
        let inv_s = if s > 0.0 { 1.0 / s } else { 0.0 };
        let reverse = self.reverse;
        // Pattern values first (one plain loop per pattern, vectorised),
        // then the levels.
        thread_local! {
            static VALUES: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
        }
        VALUES.with(|vals| {
            let vals = &mut *vals.borrow_mut();
            if vals.len() < out.len() {
                vals.resize(out.len(), 0.0);
            }
            let vals = &mut vals[..out.len()];
            let u0 = (x0 as f32 + 0.5) * iw;
            match self.pattern {
                WipePattern::Horizontal => mask_values(vals, u0, iw, 0, 0.0, 0.0, 0.0),
                WipePattern::Vertical => vals.fill(v),
                WipePattern::BarnDoorVertical => mask_values(vals, u0, iw, 1, 0.0, 0.0, 0.0),
                WipePattern::BarnDoorHorizontal => vals.fill((v - 0.5).abs() * 2.0),
                WipePattern::Box => mask_values(vals, u0, iw, 2, (v - 0.5).abs() * 2.0, 0.0, 0.0),
                WipePattern::Iris => {
                    let norm = 1.0 / (0.25 * aspect * aspect + 0.25).sqrt();
                    mask_values(vals, u0, iw, 3, (v - 0.5) * (v - 0.5), aspect, norm)
                }
                WipePattern::Diagonal => mask_values(vals, u0, iw, 4, v, 0.0, 0.0),
            }
            mask_levels(out, vals, reverse, s, p, top, inv_s);
        });
    }
}

multiversion! {
    /// Wipe pattern values of a row: `u = u0 + i·iw`; `kind` 0 horizontal,
    /// 1 barn door, 2 box (`c`: the row's part), 3 iris (`c`: dy²), 4
    /// diagonal (`c`: v).
    pub fn mask_values(vals: &mut [f32], u0: f32, iw: f32, kind: u8, c: f32, aspect: f32, norm: f32) {
        const LANES: [f32; 8] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mut base = u0;
        let step = 8.0 * iw;
        macro_rules! each {
            (|$u:ident| $e:expr) => {{
                let mut chunks = vals.chunks_exact_mut(8);
                for ch in &mut chunks {
                    for k in 0..8 {
                        let $u = base + LANES[k] * iw;
                        ch[k] = $e;
                    }
                    base += step;
                }
                for (k, o) in chunks.into_remainder().iter_mut().enumerate() {
                    let $u = base + LANES[k] * iw;
                    *o = $e;
                }
            }};
        }
        match kind {
            0 => each!(|u| u),
            1 => each!(|u| (u - 0.5).abs() * 2.0),
            2 => each!(|u| ((u - 0.5).abs() * 2.0).max(c)),
            3 => each!(|u| {
                let dx = (u - 0.5) * aspect;
                (dx * dx + c).sqrt() * norm
            }),
            _ => each!(|u| (u + c) * 0.5),
        }
    }
}

multiversion! {
    /// Q15 levels of wipe pattern values.
    pub fn mask_levels(out: &mut [u16], vals: &mut [f32], reverse: bool, s: f32, p: f32, top: f32, inv_s: f32) {
        if reverse {
            for v in vals.iter_mut() {
                *v = 1.0 - *v;
            }
        }
        if s <= 0.0 {
            for (o, &v) in out.iter_mut().zip(vals.iter()) {
                *o = if v < p { ONE as u16 } else { 0 };
            }
        } else {
            for (o, &v) in out.iter_mut().zip(vals.iter()) {
                let a = ((top - v) * inv_s).max(0.0).min(1.0);
                *o = (a * ONE as f32 + 0.5) as i32 as u16;
            }
        }
    }
}

multiversion! {
    /// `acc += row · w`
    pub fn acc_rows(acc: &mut [u32], row: &[u16], w: u32) {
        for (a, &v) in acc.iter_mut().zip(row.iter()) {
            *a += v as u32 * w;
        }
    }
}

multiversion! {
    /// `acc += row`
    pub fn add_rows(acc: &mut [u16], row: &[u16]) {
        for (a, &v) in acc.iter_mut().zip(row.iter()) {
            *a = a.wrapping_add(v);
        }
    }
}

/// What makes a layer's tables valid.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerKey10 {
    pub width: usize,
    pub height: usize,
    /// Source pixels `(x0, y0, x1, y1)`
    pub crop: (u32, u32, u32, u32),
    /// Picture size in canvas pixels
    pub size: (i32, i32),
}

/// Sum of the fractional filter weights.
const WEIGHTS: u32 = 1024;

/// A layer's filter tables (scaled pictures only).
pub struct Prepared10 {
    pub key: LayerKey10,
    /// Unscaled: the source is drawn 1:1 (no tables)
    pub unscaled: bool,
    luma_x: HTab,
    chroma_x: HTab,
    /// Vertical table, over the rows read (every `vstep`-th from `phase`)
    y: Table,
    vstep: usize,
    phase: usize,
    /// Whole-number box both ways `(kx, ky)`, small enough for 16-bit sums
    boxed: Option<(usize, usize)>,
}

/// Whether strong vertical downscales read only some rows
/// (`MIXER_DECIMATE`; `0` turns it off).
pub fn decimate() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("MIXER_DECIMATE").as_deref() != Ok("0"))
}

impl Prepared10 {
    pub fn new(key: LayerKey10) -> Prepared10 {
        Self::with_options(key, decimate())
    }

    /// With or without decimation (tests compare both).
    pub fn with_options(key: LayerKey10, decimate: bool) -> Prepared10 {
        let (x0, y0, x1, y1) = key.crop;
        let (pw, ph) = (key.size.0.max(0) as usize, key.size.1.max(0) as usize);
        let unscaled = (x1 - x0) as usize == pw && (y1 - y0) as usize == ph && x0 % 2 == 0;
        let empty = || Table::new(0.0, 0.0, 0, 0);
        if unscaled {
            return Prepared10 {
                key,
                unscaled,
                luma_x: HTab::new(&empty()),
                chroma_x: HTab::new(&empty()),
                y: empty(),
                vstep: 1,
                phase: 0,
                boxed: None,
            };
        }
        // 1/1024 weights: vertical sums stay below 2^20, Q5 below 2^15.
        let luma_x = Table::with_total(x0 as f64, x1 as f64, pw, key.width, WEIGHTS);
        let chroma_x = Table::with_total(
            x0 as f64 / 2.0,
            x1 as f64 / 2.0,
            pw / 2,
            key.width.div_ceil(2),
            WEIGHTS,
        );
        let vstep = if decimate && ph > 0 {
            super::render::vertical_step((y1 - y0) as f64 / ph as f64)
        } else {
            1
        };
        let y = Table::with_total(
            y0 as f64 / vstep as f64,
            y1 as f64 / vstep as f64,
            ph,
            key.height.div_ceil(vstep),
            WEIGHTS,
        );
        let boxed = match (luma_x.box_k, chroma_x.box_k, y.box_k) {
            (Some(kx), Some(kc), Some(ky)) if kx == kc && kx * ky <= 32 => Some((kx as usize, ky as usize)),
            _ => None,
        };
        Prepared10 {
            key,
            unscaled,
            luma_x: HTab::new(&luma_x),
            chroma_x: HTab::new(&chroma_x),
            y,
            vstep,
            phase: (vstep - 1) / 2,
            boxed,
        }
    }

    /// Source rows read per output row (1: all).
    pub fn vstep(&self) -> usize {
        self.vstep
    }
}

/// One layer to draw this frame.
pub struct LayerInput10<'a> {
    pub source: Source10<'a>,
    pub prepared: &'a Prepared10,
    pub keying: Keying,
    pub mask: Option<WipeMask>,
}

/// Everything needed to draw a frame.
pub struct Frame10<'a> {
    pub canvas: &'a Canvas10,
    pub plan: &'a Plan,
    /// By layout index
    pub layers: Vec<Option<LayerInput10<'a>>>,
    /// Background colour `[Y, Cb, Cr]`; `None`: draw over what the canvas
    /// holds
    pub background: Option<[u16; 3]>,
}

/// Draw `frame`, on `pool` in strips if given.
pub fn render10(frame: &Frame10, pool: Option<&StripPool>) {
    let h = frame.canvas.height;
    let strips = match pool {
        Some(pool) if pool.threads() > 1 => (pool.threads() * 4).min(h / 8).max(1),
        _ => 1,
    };
    let edges: Vec<usize> = (0..=strips).map(|i| i * h / strips).collect();
    match pool {
        Some(pool) if strips > 1 => pool.run(strips, &|i| render_strip(frame, edges[i], edges[i + 1])),
        _ => render_strip(frame, 0, h),
    }
}

/// Per-thread buffers.
#[derive(Default)]
struct Scratch {
    /// The layer's samples for the span
    src: Vec<u16>,
    /// The canvas's samples of a run
    dst: Vec<u16>,
    /// Levels per pixel, then per sample
    alpha_px: Vec<u16>,
    alpha: Vec<u16>,
    /// Key plane samples
    key: Vec<u16>,
    /// A source row's span (UYVY), its planes (Y, Cb, Cr, key), their
    /// vertical sums, the sums as Q5, the filtered planes
    row: Vec<u16>,
    key_row: Vec<u16>,
    planes: [Vec<u16>; 4],
    acc: [Vec<u32>; 4],
    q5: [Vec<u16>; 4],
    out: [Vec<u16>; 3],
    tables: Vec<(KeyShape, Box<[u16; 1024]>)>,
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::default());
}

fn grow<T: Default + Clone>(v: &mut Vec<T>, n: usize) {
    if v.len() < n {
        v.resize(n, T::default());
    }
}

impl Scratch {
    fn table(&mut self, shape: &KeyShape) -> usize {
        if let Some(i) = self.tables.iter().position(|(s, _)| s == shape) {
            return i;
        }
        if self.tables.len() > 16 {
            self.tables.clear();
        }
        self.tables.push((*shape, shape.table()));
        self.tables.len() - 1
    }
}

/// The span of row `y` inside `r`, as pixels `(x0, x1)`: even, inside the
/// canvas (rectangles are on even pixels already; this keeps them there).
fn span(r: &PixelRect, y: usize, width: usize) -> Option<(usize, usize)> {
    let yi = y as i32;
    if r.is_empty() || yi < r.y || yi >= r.y + r.height {
        return None;
    }
    let w = (width & !1) as i32;
    let x0 = (r.x.clamp(0, w) & !1) as usize;
    let x1 = (((r.x + r.width).clamp(0, w) + 1) & !1).min(w) as usize;
    (x0 < x1).then_some((x0, x1))
}

/// Draw rows `y0..y1` of the frame.
pub fn render_strip(frame: &Frame10, y0: usize, y1: usize) {
    let canvas = frame.canvas;
    let width = canvas.width;
    SCRATCH.with(|s| {
        let s = &mut *s.borrow_mut();
        for y in y0..y1 {
            // SAFETY: rows of this strip only.
            let row = unsafe { canvas.row(y) };
            if let Some(bg) = frame.background {
                for r in &frame.plan.background {
                    if let Some((x0, x1)) = span(r, y, width) {
                        v210::fill(row, 2 * x0, 2 * x1, bg);
                    }
                }
            }
            for item in &frame.plan.items {
                match item {
                    Item::Fill { rect, argb, alpha } => {
                        if let Some((x0, x1)) = span(rect, y, width) {
                            let c = kernels10::argb_to_ycbcr10(*argb);
                            let a = (*alpha as u32 * ONE + 127) / 255;
                            v210::mix_color(row, 2 * x0, 2 * x1, c, a);
                        }
                    }
                    Item::Layer {
                        index,
                        picture,
                        visible,
                        opacity,
                    } => {
                        let Some(Some(input)) = frame.layers.get(*index) else {
                            continue;
                        };
                        let opacity = (*opacity as u32 * ONE + 127) / 255;
                        for r in visible {
                            if let Some((x0, x1)) = span(r, y, width) {
                                draw_span(s, row, canvas, input, picture, opacity, y, x0, x1);
                            }
                        }
                    }
                }
            }
        }
    });
}

/// Draw pixels `x0..x1` (even) of canvas row `y` of a layer.
#[allow(clippy::too_many_arguments)]
fn draw_span(
    s: &mut Scratch,
    row: &mut [u8],
    canvas: &Canvas10,
    input: &LayerInput10,
    picture: &PixelRect,
    opacity: u32,
    y: usize,
    x0: usize,
    x1: usize,
) {
    let prep = input.prepared;
    let src = &input.source;
    let (cx0, cy0, _, _) = prep.key.crop;
    let j = (y as i32 - picture.y) as usize;
    let n = x1 - x0;
    let plain = matches!(input.keying, Keying::None) && input.mask.is_none();

    if !prep.unscaled {
        if !filter_span(s, input, picture, j, x0, x1) {
            return;
        }
        if plain {
            put_filtered(s, row, x0, n, opacity);
            return;
        }
        grow(&mut s.alpha_px, n);
        match &input.keying {
            Keying::None => s.alpha_px[..n].fill(ONE as u16),
            Keying::Luma(shape) => {
                let t = s.table(shape);
                let (tables, src16, alpha) = (&s.tables, &s.src, &mut s.alpha_px);
                kernels10::luma_key(&src16[..2 * n], &tables[t].1, &mut alpha[..n]);
            }
            Keying::Linear(shape) if src.key.is_some() => {
                let t = s.table(shape);
                let (tables, key, alpha) = (&s.tables, &s.key, &mut s.alpha_px);
                kernels10::key_levels(&key[..n], &tables[t].1, &mut alpha[..n]);
            }
            Keying::Linear(_) => s.alpha_px[..n].fill(ONE as u16),
        }
        apply_mask(s, input, canvas, y, x0, n);
        kernels10::scale_alpha(&mut s.alpha_px[..n], opacity);
        blend_runs(s, row, x0, n, None);
        return;
    }

    // Unscaled: source pixel of canvas pixel x is sx0 + (x − x0).
    let sy = cy0 as usize + j;
    if sy >= src.height {
        return;
    }
    let sx0 = (cx0 as i64 + x0 as i64 - picture.x as i64) as usize;
    let srow = src.row(sy);
    let aligned = (2 * x0) % 3 == (2 * sx0) % 3;
    if plain {
        put_unscaled(s, row, srow, canvas.width, x0, sx0, n, opacity, aligned);
        return;
    }
    // Levels first: transparent runs never read the fill.
    grow(&mut s.alpha_px, n);
    let mut have_fill = false;
    match &input.keying {
        Keying::None => s.alpha_px[..n].fill(ONE as u16),
        Keying::Luma(shape) => {
            grow(&mut s.src, 2 * n);
            v210::unpack(srow, 2 * sx0, &mut s.src[..2 * n]);
            have_fill = true;
            let t = s.table(shape);
            let (tables, src16, alpha) = (&s.tables, &s.src, &mut s.alpha_px);
            kernels10::luma_key(&src16[..2 * n], &tables[t].1, &mut alpha[..n]);
        }
        Keying::Linear(shape) => match src.key_row(sy) {
            Some(krow) => {
                // A key row at or below the key's start is transparent.
                if !shape.invert && v210::max_key(krow, sx0, sx0 + n) as f32 <= shape.lo {
                    return;
                }
                grow(&mut s.key, n);
                v210::unpack_key(krow, sx0, &mut s.key[..n]);
                let t = s.table(shape);
                kernels10::key_levels(&s.key[..n], &s.tables[t].1, &mut s.alpha_px[..n]);
            }
            None => s.alpha_px[..n].fill(ONE as u16),
        },
    }
    apply_mask(s, input, canvas, y, x0, n);
    kernels10::scale_alpha(&mut s.alpha_px[..n], opacity);
    let fill = if have_fill { None } else { Some((srow, sx0, aligned)) };
    blend_runs(s, row, x0, n, fill);
}

/// Multiply the levels by the layer's wipe mask.
fn apply_mask(s: &mut Scratch, input: &LayerInput10, canvas: &Canvas10, y: usize, x0: usize, n: usize) {
    if let Some(mask) = &input.mask {
        grow(&mut s.key_row, n);
        mask.row_levels(y, x0, &mut s.key_row[..n], canvas.width, canvas.height);
        kernels10::mul_alpha(&mut s.alpha_px[..n], &s.key_row[..n]);
    }
}

/// An unscaled, unkeyed span: copied or mixed as words when the source's
/// words line up with the canvas's, else through 16-bit samples.
#[allow(clippy::too_many_arguments)]
fn put_unscaled(
    s: &mut Scratch,
    row: &mut [u8],
    srow: &[u8],
    width: usize,
    x0: usize,
    sx0: usize,
    n: usize,
    opacity: u32,
    aligned: bool,
) {
    let (d0, s0) = (2 * x0, 2 * sx0);
    if aligned {
        if opacity >= ONE && d0 == s0 && x0 == 0 && n == (width & !1) && srow.len() == row.len() {
            // A whole row in place: the bytes, padding included.
            row.copy_from_slice(srow);
            return;
        }
        let shift = (s0 as isize - d0 as isize) / 3 * 4;
        let srow = shifted(srow, shift, row.len());
        if opacity >= ONE {
            v210::copy(row, &srow, d0, d0 + 2 * n);
        } else {
            v210::mix(row, &srow, d0, d0 + 2 * n, opacity);
        }
        return;
    }
    grow(&mut s.src, 2 * n);
    v210::unpack(srow, s0, &mut s.src[..2 * n]);
    put_filtered(s, row, x0, n, opacity);
}

/// `s.src` (2·n samples) into the row at pixel `x0` with one level.
fn put_filtered(s: &mut Scratch, row: &mut [u8], x0: usize, n: usize, opacity: u32) {
    if opacity >= ONE {
        v210::pack(row, 2 * x0, &s.src[..2 * n]);
    } else {
        grow(&mut s.dst, 2 * n);
        v210::unpack(row, 2 * x0, &mut s.dst[..2 * n]);
        kernels10::mix16(&mut s.dst[..2 * n], &s.src[..2 * n], opacity);
        v210::pack(row, 2 * x0, &s.dst[..2 * n]);
    }
}

/// `row` offset by `shift` bytes (whole words), padded so that indices up to
/// `len` are valid. Borrowed when the shift is 0 or positive.
fn shifted(row: &[u8], shift: isize, len: usize) -> std::borrow::Cow<'_, [u8]> {
    if shift >= 0 {
        let s = shift as usize;
        if s + len <= row.len() {
            return std::borrow::Cow::Borrowed(&row[s..s + len]);
        }
        let mut v = vec![0u8; len];
        let n = row.len().saturating_sub(s).min(len);
        v[..n].copy_from_slice(&row[s..s + n]);
        std::borrow::Cow::Owned(v)
    } else {
        let s = (-shift) as usize;
        let mut v = vec![0u8; len];
        let n = len.saturating_sub(s).min(row.len());
        v[s..s + n].copy_from_slice(&row[..n]);
        std::borrow::Cow::Owned(v)
    }
}

/// Blend a layer's span (pixel `x0`, `n` pixels) into the row with the
/// levels in `s.alpha_px`, in runs of 48 pixels: transparent runs are
/// skipped, opaque runs written, the rest blended. The layer's samples are
/// `s.src`, or (`fill`) read here from the source row (`srow`, its first
/// pixel, whether its words line up) only for the runs that show.
fn blend_runs(s: &mut Scratch, row: &mut [u8], x0: usize, n: usize, fill: Option<(&[u8], usize, bool)>) {
    const RUN: usize = 48;
    grow(&mut s.alpha, 2 * n);
    grow(&mut s.dst, 2 * n);
    grow(&mut s.src, 2 * n);
    let mut i = 0;
    while i < n {
        let m = RUN.min(n - i);
        let (a0, a1) = (2 * i, 2 * (i + m));
        match kernels10::uniform(&s.alpha_px[i..i + m]) {
            Some(false) => {}
            Some(true) => match fill {
                Some((srow, sx0, true)) => {
                    let (d0, s0) = (2 * (x0 + i), 2 * (sx0 + i));
                    let shift = (s0 as isize - d0 as isize) / 3 * 4;
                    let srow = shifted(srow, shift, row.len());
                    v210::copy(row, &srow, d0, d0 + 2 * m);
                }
                Some((srow, sx0, false)) => {
                    v210::unpack(srow, 2 * (sx0 + i), &mut s.src[a0..a1]);
                    v210::pack(row, 2 * (x0 + i), &s.src[a0..a1]);
                }
                None => v210::pack(row, 2 * (x0 + i), &s.src[a0..a1]),
            },
            None => {
                if let Some((srow, sx0, _)) = fill {
                    v210::unpack(srow, 2 * (sx0 + i), &mut s.src[a0..a1]);
                }
                kernels10::alpha_to_samples(&s.alpha_px[i..i + m], &mut s.alpha[a0..a1]);
                let d = &mut s.dst[a0..a1];
                v210::unpack(row, 2 * (x0 + i), d);
                kernels10::blend16(d, &s.src[a0..a1], &s.alpha[a0..a1]);
                v210::pack(row, 2 * (x0 + i), d);
            }
        }
        i += m;
    }
}

/// Filter the picture's pixels `x0..x1` of canvas row `y = picture.y + j`
/// into `s.src` (and the key into `s.key`). `false`: nothing to draw. See
/// [`super::scale10`] for the steps.
fn filter_span(
    s: &mut Scratch,
    input: &LayerInput10,
    picture: &PixelRect,
    j: usize,
    x0: usize,
    x1: usize,
) -> bool {
    let prep = input.prepared;
    let src = &input.source;
    if j >= prep.y.len {
        return false;
    }
    let n = x1 - x0;
    let ox0 = (x0 as i64 - picture.x as i64) as usize;
    let ox1 = ox0 + n;
    if ox1 > prep.luma_x.len {
        return false;
    }
    let (lt, ct) = (&prep.luma_x, &prep.chroma_x);
    let (oc0, oc1) = (ox0 / 2, ox1.div_ceil(2).min(ct.len));
    // Source pixels read: even start, both channels' windows.
    let lx0 = lt.start[ox0] as usize;
    let lx1 = lt.start[ox1 - 1] as usize + 2 * lt.pairs;
    let cq0 = ct.start[oc0] as usize;
    let cq1 = if oc1 > oc0 { ct.start[oc1 - 1] as usize + 2 * ct.pairs } else { cq0 };
    let p0 = lx0.min(2 * cq0) & !1;
    let p1 = ((lx1.max(2 * cq1) + 1) & !1).min(src.width & !1).max(p0);
    let np = p1 - p0;
    let key = matches!(input.keying, Keying::Linear(_)) && src.key.is_some();
    let nplanes = if key { 4 } else { 3 };
    let len = |c: usize| if c == 1 || c == 2 { np / 2 } else { np };
    // Slack for the filter's reach past the last sample read (zeros).
    let pad = 2 * (lt.reach.max(ct.reach) + 8);
    grow(&mut s.row, 2 * np);
    grow(&mut s.key_row, np);
    for c in 0..nplanes {
        grow(&mut s.planes[c], np + pad);
        grow(&mut s.acc[c], np + pad);
        grow(&mut s.q5[c], np + pad);
    }
    if let Some((kx, ky)) = prep.boxed {
        return filter_box(s, input, j, ox0, n, kx, ky, key, p0, np, pad);
    }
    // Vertical: the source rows of output row j, one after the other.
    let w = prep.y.weights(j);
    let start = prep.y.start[j] as usize;
    let mut first = true;
    for (t, &wt) in w.iter().enumerate() {
        if wt == 0 {
            continue;
        }
        let sy = ((start + t) * prep.vstep + prep.phase).min(src.height - 1);
        // The next source row is read next: ask for it now (16 bytes per
        // 6 pixels).
        if w.get(t + 1).is_some_and(|&w| w != 0) {
            let ny = ((start + t + 1) * prep.vstep + prep.phase).min(src.height - 1);
            let row = src.row(ny);
            let b0 = (8 * p0 / 3).min(row.len());
            kernels::prefetch(&row[b0..(b0 + 8 * np / 3 + 16).min(row.len())]);
        }
        {
            let [py, pcb, pcr, _] = &mut s.planes;
            v210::unpack_planar(src.row(sy), p0, &mut py[..np + 2], &mut pcb[..np / 2 + 1], &mut pcr[..np / 2 + 1]);
        }
        if key {
            let kr = &mut s.planes[3][..np];
            v210::unpack_key(src.key_row(sy).unwrap(), p0, kr);
        }
        for c in 0..nplanes {
            let (acc, row) = (&mut s.acc[c][..len(c)], &s.planes[c][..len(c)]);
            if first {
                scale10::acc_first(acc, row, wt as u32);
            } else {
                scale10::acc_add(acc, row, wt as u32);
            }
        }
        first = false;
    }
    if first {
        return false;
    }
    for c in 0..nplanes {
        let l = len(c);
        scale10::normalize(&mut s.q5[c][..l], &s.acc[c][..l], prep.y.total);
        s.q5[c][l..l + pad].fill(0);
    }
    // Horizontal, per plane.
    for c in 0..3 {
        grow(&mut s.out[c], n);
    }
    let [oy, ocb, ocr] = &mut s.out;
    scale10::hfilter(&mut oy[..n], &s.q5[0], lt, ox0, p0);
    scale10::hfilter(&mut ocb[..n / 2], &s.q5[1], ct, oc0, p0 / 2);
    scale10::hfilter(&mut ocr[..n / 2], &s.q5[2], ct, oc0, p0 / 2);
    grow(&mut s.src, 2 * n);
    scale10::interleave(&oy[..n], &ocb[..n / 2], &ocr[..n / 2], &mut s.src[..2 * n]);
    if key {
        grow(&mut s.key, n);
        scale10::hfilter(&mut s.key[..n], &s.q5[3], lt, ox0, p0);
    }
    true
}

/// Whole-number boxes both ways: the source rows' planes summed as `u16`
/// one row after the other, then [`scale10::hbox`] (or the filter with
/// weights of 1 for box widths other than 1, 2, 4, 8).
#[allow(clippy::too_many_arguments)]
fn filter_box(
    s: &mut Scratch,
    input: &LayerInput10,
    j: usize,
    ox0: usize,
    n: usize,
    kx: usize,
    ky: usize,
    key: bool,
    p0: usize,
    np: usize,
    pad: usize,
) -> bool {
    let prep = input.prepared;
    let src = &input.source;
    let nplanes = if key { 4 } else { 3 };
    let len = |c: usize| if c == 1 || c == 2 { np / 2 } else { np };
    let start = prep.y.start[j] as usize;
    for t in 0..ky {
        let sy = ((start + t) * prep.vstep + prep.phase).min(src.height - 1);
        if t + 1 < ky {
            let ny = ((start + t + 1) * prep.vstep + prep.phase).min(src.height - 1);
            let row = src.row(ny);
            let b0 = (8 * p0 / 3).min(row.len());
            kernels::prefetch(&row[b0..(b0 + 8 * np / 3 + 16).min(row.len())]);
        }
        // The first row's planes are the sums; the next rows are added.
        let target = if t == 0 { &mut s.q5 } else { &mut s.planes };
        {
            let [py, pcb, pcr, _] = target;
            v210::unpack_planar(src.row(sy), p0, &mut py[..np + 2], &mut pcb[..np / 2 + 1], &mut pcr[..np / 2 + 1]);
        }
        if key {
            v210::unpack_key(src.key_row(sy).unwrap(), p0, &mut target[3][..np]);
        }
        if t > 0 {
            for c in 0..nplanes {
                let l = len(c);
                add_rows(&mut s.q5[c][..l], &s.planes[c][..l]);
            }
        }
    }
    for c in 0..nplanes {
        let l = len(c);
        s.q5[c][l..l + pad].fill(0);
    }
    for c in 0..3 {
        grow(&mut s.out[c], n);
    }
    let div = (kx * ky) as u32;
    let (lt, ct) = (&prep.luma_x, &prep.chroma_x);
    let plane = |out: &mut [u16], sums: &[u16], tab: &HTab, j0: usize, base: usize| {
        if matches!(kx, 1 | 2 | 4 | 8) {
            // Box outputs start at whole boxes from the window's start.
            let first = tab.start[j0] as usize - base;
            scale10::hbox(out, &sums[first..], kx, div);
        } else {
            scale10::hfilter_div(out, sums, tab, j0, base, 1.0 / div as f32);
        }
    };
    let [oy, ocb, ocr] = &mut s.out;
    plane(&mut oy[..n], &s.q5[0], lt, ox0, p0);
    plane(&mut ocb[..n / 2], &s.q5[1], ct, ox0 / 2, p0 / 2);
    plane(&mut ocr[..n / 2], &s.q5[2], ct, ox0 / 2, p0 / 2);
    grow(&mut s.src, 2 * n);
    scale10::interleave(&oy[..n], &ocb[..n / 2], &ocr[..n / 2], &mut s.src[..2 * n]);
    if key {
        grow(&mut s.key, n);
        plane(&mut s.key[..n], &s.q5[3], lt, ox0, p0);
    }
    true
}

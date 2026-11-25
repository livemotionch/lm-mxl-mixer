//! Drawing a frame: background, layers and borders, in horizontal strips.
//!
//! A layer is drawn per output row (pairs of rows on a 4:2:0 canvas), and
//! only for the parts the plan says are visible:
//!
//! 1. **Vertical**: the source rows that make the output row are summed with
//!    the vertical filter's weights, as raw bytes of the source's layout
//!    (UYVY, BGRA, an NV12 plane...), over the byte range the visible span
//!    needs. Downscaling vertically first means the horizontal filter runs
//!    once per *output* row. On a 4:2:0 canvas, the chroma of a packed
//!    source is the sum of the two luma rows' sums, so the source is read
//!    once.
//! 2. **Horizontal**: each channel is filtered out of the summed row
//!    (deinterleaving at the same time), to Q6.
//! 3. **Transform**: to the canvas's colour model (or a plain rounding).
//! 4. **Write**: copied into the canvas, or blended (premultiplied alpha,
//!    layer opacity), skipping blocks of 16 fully transparent pixels.
//!
//! Strips are drawn in parallel and never touch the same rows, which is
//! what makes the raw-pointer canvas access below sound.

// Index loops over several slices at once are intentional here.
#![allow(clippy::needless_range_loop)]

use std::cell::RefCell;

use super::config::PixelRect;
use super::color::{Model, Transform, argb_to};
use super::formats::{FormatSpec, Loader, SrcFormat, load};
use super::kernels::{self, Scale, q6_scale};
use super::layout::{Item, Plan};
use super::pool::StripPool;
use super::scale::Table;

/// A layer's current source frame.
#[derive(Clone, Copy)]
pub struct SourceFrame<'a> {
    pub format: SrcFormat,
    pub width: usize,
    pub height: usize,
    pub planes: [&'a [u8]; 3],
    pub strides: [usize; 3],
    pub model: Model,
}

impl<'a> SourceFrame<'a> {
    fn row(&self, plane: usize, row: usize) -> &'a [u8] {
        let start = row * self.strides[plane];
        let data = self.planes[plane];
        &data[start.min(data.len())..(start + self.strides[plane]).min(data.len())]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CanvasFormat {
    Nv12,
    I420,
    /// Premultiplied BGRA, the ports' convention for alpha
    /// (`crate::pipeline::alpha`)
    Bgra,
}

impl CanvasFormat {
    pub fn is_420(self) -> bool {
        self != CanvasFormat::Bgra
    }
}

/// The output frame being drawn.
pub struct Canvas {
    pub format: CanvasFormat,
    pub width: usize,
    pub height: usize,
    pub model: Model,
    planes: [*mut u8; 3],
    strides: [usize; 3],
    lens: [usize; 3],
}

// SAFETY: strips access disjoint rows only (see `row`).
unsafe impl Send for Canvas {}
unsafe impl Sync for Canvas {}

impl Canvas {
    /// # Safety
    /// Each `(pointer, length)` must be a writable plane that stays valid
    /// and unaliased while the canvas is used.
    pub unsafe fn new(
        format: CanvasFormat,
        width: usize,
        height: usize,
        model: Model,
        planes: [(*mut u8, usize); 3],
        strides: [usize; 3],
    ) -> Canvas {
        Canvas {
            format,
            width,
            height,
            model,
            planes: planes.map(|(p, _)| p),
            strides,
            lens: planes.map(|(_, l)| l),
        }
    }

    /// Row `y` of `plane`.
    ///
    /// # Safety
    /// No two threads may hold the same row at once: each strip only asks
    /// for rows inside its own range.
    #[allow(clippy::mut_from_ref)]
    unsafe fn row(&self, plane: usize, y: usize) -> &mut [u8] {
        let start = y * self.strides[plane];
        let end = (start + self.strides[plane]).min(self.lens[plane]);
        assert!(start <= end);
        unsafe { std::slice::from_raw_parts_mut(self.planes[plane].add(start), end - start) }
    }
}

/// What makes a layer's tables valid.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerKey {
    pub format: SrcFormat,
    pub width: usize,
    pub height: usize,
    pub model: Model,
    /// Source pixels `(x0, y0, x1, y1)`
    pub crop: (u32, u32, u32, u32),
    pub picture: PixelRect,
    pub canvas: CanvasFormat,
    pub canvas_model: Model,
}

impl LayerKey {
    /// Whether the tables (and a cache's pixels) made for `self` also do for
    /// `other`: everything but the picture's position is the same, so a
    /// layer being moved keeps them.
    pub fn same_tables(&self, other: &LayerKey) -> bool {
        let size = |k: &LayerKey| (k.picture.width, k.picture.height);
        self.format == other.format
            && self.width == other.width
            && self.height == other.height
            && self.model == other.model
            && self.crop == other.crop
            && size(self) == size(other)
            && self.canvas == other.canvas
            && self.canvas_model == other.canvas_model
    }
}

/// One set of canvas planes at one resolution: luma and chroma of a 4:2:0
/// canvas, or the R, G, B of a BGRA canvas.
#[derive(Debug)]
struct Group {
    /// Canvas pixels per group sample (1 or 2)
    sub: usize,
    w: usize,
    h: usize,
    /// Transform rows (canvas channels) this group writes
    outputs: Vec<usize>,
    /// Source channels read (0–2, 3 = alpha)
    needed: [bool; 4],
    htab: [Option<Table>; 4],
    /// Per channel Q6 scale
    scale: [Scale; 4],
    /// Per plane: plane rows → group rows (`None` if unused or fused)
    vtab: [Option<Table>; 3],
    /// Per plane: sums of the luma group's two rows (4:2:0 chroma of a plane
    /// with full vertical resolution)
    fused: [bool; 3],
    /// Same size, same colours, no alpha: the source's bytes are copied
    /// without filtering.
    direct: bool,
}

/// Layers drawn by a dedicated kernel instead of the general
/// sum-filter-transform steps.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Fast {
    None,
    /// Same-size RGB(x) source (no alpha) on a 4:2:0 canvas: each canvas
    /// sample straight from the source pixels.
    Rgb420 {
        y: kernels::PixelMap,
        u: kernels::PixelMap,
        v: kernels::PixelMap,
        /// Byte of the alpha (premultiplied first; maps for premultiplied
        /// pixels)
        alpha: Option<usize>,
    },
    /// Packed 4:2:2 (UYVY, YUY2) without conversion, downscaled
    /// horizontally by `k` (2 or 4) with an exact box whose area is a power
    /// of two, on a 4:2:0 canvas: the horizontal pass straight to bytes.
    Packed422 {
        k: usize,
        shift_y: u32,
        shift_c: u32,
        y_odd: bool,
    },
    /// Same-size packed 4:2:2 without conversion on a 4:2:0 canvas: luma
    /// copied, chroma the average of two rows.
    Packed422Same {
        y_odd: bool,
    },
    /// RGB(x)/RGBA downscaled by an exact box on a 4:2:0 canvas: the
    /// vertical sums (premultiplied for alpha) reduced to one pixel per
    /// output (`k` source pixels, divided by the box's area `div`), then
    /// the maps of [`Fast::Rgb420`].
    /// Same-size BGRA/BGRx on the (premultiplied BGRA) transparent canvas:
    /// pixels copied, or premultiplied and blended, 16 at a time.
    Bgra {
        alpha: bool,
    },
    RgbBox420 {
        k: usize,
        div: u32,
        y: kernels::PixelMap,
        u: kernels::PixelMap,
        v: kernels::PixelMap,
        alpha: Option<usize>,
    },
    /// BGRA/BGRx downscaled by an exact box onto the (premultiplied BGRA)
    /// transparent canvas: the vertical sums reduced to one pixel per
    /// output (`k` source pixels, divided by the box's area `div`), then
    /// copied or blended as [`Fast::Bgra`].
    BgraBox {
        k: usize,
        div: u32,
        alpha: bool,
    },
    /// Same-size planar 4:2:0 (NV12, I420; `nv12`: chroma interleaved)
    /// without conversion on a 4:2:0 canvas: rows copied or blended
    /// straight from the source planes.
    Planar420Same {
        nv12: bool,
    },
    /// Planar 4:2:0 without conversion on a 4:2:0 canvas, scaled
    /// horizontally by a whole number: per group (luma, chroma) the
    /// vertical sums (any filter table, over every `vstep`-th source row
    /// for strong downscales), then a box of `k` samples straight to bytes.
    Planar420 {
        nv12: bool,
        luma: Planar,
        chroma: Planar,
    },
}

/// One group of [`Fast::Planar420`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Planar {
    /// Source samples per output (horizontal box)
    k: usize,
    /// Read every `vstep`-th source row (the vertical table is over those)
    vstep: usize,
    /// Of each block of `vstep` rows, the one read (its middle)
    phase: usize,
    /// The horizontal kernel's scaling (vertical sums to bytes)
    hbox: kernels::HBox,
}

/// Whether strong vertical downscales of planar sources read only some of
/// their rows ([`Fast::Planar420`]); `MIXER_DECIMATE=0` turns it off
/// (for comparisons).
fn decimate() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("MIXER_DECIMATE").as_deref() != Ok("0"))
}

/// Rows read per output row from a vertical downscale of `ratio` (source
/// rows per output row): from 7.5:1 down, every `vstep`-th row, keeping
/// about four rows per output. Fewer rows read is what makes multiview
/// thumbnails (1080 → 134 lines, 8:1) cheap: sources come from DRAM and
/// their reads are most of a thumbnail's cost. An area filter over every
/// other row still averages four rows per output.
pub fn vertical_step(ratio: f64) -> usize {
    if ratio >= 7.5 {
        ((ratio + 0.5) / 4.0).floor() as usize
    } else {
        1
    }
}

/// Tables for drawing one layer; rebuilt when its [`LayerKey`] changes.
#[derive(Debug)]
pub struct Prepared {
    pub key: LayerKey,
    spec: FormatSpec,
    transform: Transform,
    groups: Vec<Group>,
    fast: Fast,
}

impl Prepared {
    pub fn new(key: LayerKey) -> Prepared {
        Self::with_options(key, true, decimate())
    }

    /// Tables for the general steps only, never a dedicated kernel (tests
    /// and benchmarks compare the kernels with it).
    #[doc(hidden)]
    pub fn general(key: LayerKey) -> Prepared {
        Self::with_options(key, false, false)
    }

    /// With or without the dedicated `kernels`, and with or without
    /// [`Fast::Planar420`]'s `decimate` (tests).
    #[doc(hidden)]
    pub fn with_options(key: LayerKey, kernels: bool, decimate: bool) -> Prepared {
        let spec = key.format.spec();
        let transform = Transform::new(key.model, key.canvas_model);
        let has_alpha = spec.alpha.is_some();
        let pic = key.picture;
        let (cx0, cy0, cx1, cy1) = key.crop;
        let chan = |c: usize| {
            if c == 3 {
                spec.alpha.unwrap()
            } else {
                spec.chans[c]
            }
        };

        let layouts: Vec<(usize, Vec<usize>)> = if key.canvas.is_420() {
            vec![(1, vec![0]), (2, vec![1, 2])]
        } else {
            vec![(1, vec![0, 1, 2])]
        };
        let mut groups: Vec<Group> = Vec::new();
        for (gi, (sub, outputs)) in layouts.into_iter().enumerate() {
            let w = (pic.width.max(0) as usize) / sub;
            let h = (pic.height.max(0) as usize) / sub;
            let mut needed = [false; 4];
            for j in 0..3 {
                needed[j] = outputs.iter().any(|&k| transform.uses(k, j));
            }
            needed[3] = has_alpha;
            let mut htab: [Option<Table>; 4] = Default::default();
            for c in 0..4 {
                if needed[c] {
                    let ch = chan(c);
                    let hs = ch.hsub as f64;
                    htab[c] = Some(Table::new(
                        cx0 as f64 / hs,
                        cx1 as f64 / hs,
                        w,
                        key.width.div_ceil(ch.hsub as usize),
                    ));
                }
            }
            let mut fused = [false; 3];
            let mut vtab: [Option<Table>; 3] = Default::default();
            for p in 0..spec.planes {
                if !(0..4).any(|c| needed[c] && chan(c).plane == p) {
                    continue;
                }
                let vs = spec.vsub[p];
                // Chroma from the luma rows' sums, unless the luma group
                // doesn't sum (copies) anyway.
                if gi == 1 && vs == 1 && !groups[0].direct {
                    fused[p] = true;
                    continue;
                }
                vtab[p] = Some(Table::new(
                    cy0 as f64 / vs as f64,
                    cy1 as f64 / vs as f64,
                    h,
                    key.height.div_ceil(vs as usize),
                ));
            }
            let direct = transform.identity
                && spec.loader == Loader::Direct
                && !has_alpha
                && (0..3).filter(|&c| needed[c]).all(|c| {
                    htab[c].as_ref().is_some_and(|t| t.box_k == Some(1))
                        && vtab[chan(c).plane]
                            .as_ref()
                            .is_some_and(|t| t.box_k == Some(1))
                });
            groups.push(Group {
                sub,
                w,
                h,
                outputs,
                needed,
                htab,
                scale: [Scale::Shift(0); 4],
                vtab,
                fused,
                direct,
            });
        }
        // The luma group also sums the planes its chroma group fuses.
        if groups.len() == 2 {
            for p in 0..spec.planes {
                if groups[1].fused[p] && groups[0].vtab[p].is_none() {
                    groups[0].vtab[p] =
                        Some(Table::new(cy0 as f64, cy1 as f64, groups[0].h, key.height));
                }
            }
        }
        // Scales: vertical total × horizontal total (× 2 when fused).
        for gi in 0..groups.len() {
            for c in 0..4 {
                let Some(ht) = &groups[gi].htab[c] else {
                    continue;
                };
                let p = chan(c).plane;
                let (vt, mult) = if groups[gi].fused[p] {
                    (groups[0].vtab[p].as_ref().unwrap(), 2)
                } else {
                    (groups[gi].vtab[p].as_ref().unwrap(), 1)
                };
                groups[gi].scale[c] = q6_scale(vt.total as u64 * ht.total as u64 * mult);
            }
        }
        let fast = if !kernels {
            Fast::None
        } else if let Some(fast) = Self::planar420(&key, &spec, &transform, &mut groups, decimate) {
            fast
        } else {
            Self::fast(&spec, &transform, &groups)
        };
        Prepared {
            key,
            spec,
            transform,
            groups,
            fast,
        }
    }

    /// [`Fast::Planar420Same`] or [`Fast::Planar420`] if they draw this
    /// layer. For the latter, a strong vertical downscale gets vertical
    /// tables over every `vstep`-th source row ([`vertical_step`]).
    fn planar420(
        key: &LayerKey,
        spec: &FormatSpec,
        transform: &Transform,
        groups: &mut [Group],
        decimate: bool,
    ) -> Option<Fast> {
        let planar = spec.loader == Loader::Direct
            && !spec.rgb
            && spec.alpha.is_none()
            && spec.planes >= 2
            && spec.vsub[1] == 2
            && transform.identity
            && groups.len() == 2;
        if !planar {
            return None;
        }
        let nv12 = spec.planes == 2;
        if groups[0].direct && groups[1].direct {
            return Some(Fast::Planar420Same { nv12 });
        }
        let (_, cy0, _, cy1) = key.crop;
        let mut planars: [Option<(Planar, Table)>; 2] = [None, None];
        for gi in 0..2 {
            let g = &groups[gi];
            let channels: &[usize] = if gi == 0 { &[0] } else { &[1, 2] };
            let k = g.htab[channels[0]].as_ref()?.box_k? as usize;
            if channels
                .iter()
                .any(|&c| g.htab[c].as_ref().and_then(|t| t.box_k) != Some(k as u32))
            {
                return None;
            }
            // Rows of the planes this group reads (chroma: half the luma's).
            let vs = if gi == 0 { 1 } else { 2 };
            let (a, b) = (cy0 as f64 / vs as f64, cy1 as f64 / vs as f64);
            let limit = key.height.div_ceil(vs);
            let vstep = if decimate && g.h > 0 {
                vertical_step((b - a) / g.h as f64)
            } else {
                1
            };
            let table = if vstep > 1 {
                Table::new(
                    a / vstep as f64,
                    b / vstep as f64,
                    g.h,
                    limit.div_ceil(vstep),
                )
            } else {
                g.vtab[spec.chans[channels[0]].plane].clone()?
            };
            if k > 64 {
                return None;
            }
            planars[gi] = Some((
                Planar {
                    k,
                    vstep,
                    phase: (vstep - 1) / 2,
                    hbox: kernels::HBox::new(k, table.total),
                },
                table,
            ));
        }
        let [Some((luma, lt)), Some((chroma, ct))] = planars else {
            return None;
        };
        groups[0].vtab[0] = Some(lt);
        for p in 1..spec.planes {
            groups[1].vtab[p] = Some(ct.clone());
        }
        Some(Fast::Planar420 { nv12, luma, chroma })
    }

    /// Whether a dedicated kernel draws this layer.
    fn fast(spec: &FormatSpec, transform: &Transform, groups: &[Group]) -> Fast {
        let boxed = |t: &Option<Table>, k: u32| t.as_ref().is_some_and(|t| t.box_k == Some(k));
        // Pixels read as they are (premultiplied where there is alpha).
        let alpha = (spec.loader == Loader::Direct).then(|| spec.alpha.map(|a| a.offset));
        if let Some(alpha) = alpha.filter(|_| {
            spec.rgb
                && groups.len() == 2
                && (0..3).all(|c| boxed(&groups[0].htab[c], 1) && boxed(&groups[1].htab[c], 2))
                && boxed(&groups[0].vtab[0], 1)
                && groups[1].fused[0]
        }) {
            let bytes = spec.chans.map(|c| c.offset);
            let map = |k: usize, n: u32| match alpha {
                None => kernels::PixelMap::new(transform, k, bytes, n),
                Some(a) => kernels::PixelMap::premultiplied(transform, k, bytes, a, n),
            };
            let (y, u, v) = (map(0, 1), map(1, 4), map(2, 4));
            if y.fits_i16() && u.fits_i16() && v.fits_i16() {
                return Fast::Rgb420 { y, u, v, alpha };
            }
        }
        let box_k = |t: &Option<Table>| t.as_ref().and_then(|t| t.box_k).unwrap_or(0) as usize;
        if spec.rgb
            && groups.len() == 1
            && transform.identity
            && spec.chans.map(|c| c.offset) == [2, 1, 0]
            && matches!(
                spec.alpha,
                None | Some(super::formats::Chan { offset: 3, .. })
            )
            && (0..3).all(|c| box_k(&groups[0].htab[c]) == 1)
            && box_k(&groups[0].vtab[0]) == 1
        {
            return Fast::Bgra {
                alpha: spec.alpha.is_some(),
            };
        }
        if spec.rgb
            && spec.loader == Loader::Direct
            && groups.len() == 1
            && transform.identity
            && spec.chans.map(|c| c.offset) == [2, 1, 0]
            && matches!(
                spec.alpha,
                None | Some(super::formats::Chan { offset: 3, .. })
            )
        {
            let k = box_k(&groups[0].htab[0]);
            let kv = box_k(&groups[0].vtab[0]);
            let channels = if spec.alpha.is_some() { 4 } else { 3 };
            if k >= 2
                && kv >= 1
                && k * kv <= 64
                && (0..channels).all(|c| box_k(&groups[0].htab[c]) == k)
            {
                return Fast::BgraBox {
                    k,
                    div: (k * kv) as u32,
                    alpha: spec.alpha.is_some(),
                };
            }
        }
        if let Some(alpha) = alpha {
            let k = box_k(&groups[0].htab[0]);
            let kv = box_k(&groups[0].vtab[0]);
            let channels = if alpha.is_some() { 4 } else { 3 };
            if spec.rgb
                && groups.len() == 2
                && k >= 2
                && kv >= 1
                && k * kv <= 64
                && (0..channels)
                    .all(|c| box_k(&groups[0].htab[c]) == k && box_k(&groups[1].htab[c]) == 2 * k)
                && groups[1].fused[0]
            {
                let bytes = spec.chans.map(|c| c.offset);
                let map = |k: usize, n: u32| match alpha {
                    None => kernels::PixelMap::new(transform, k, bytes, n),
                    Some(a) => kernels::PixelMap::premultiplied(transform, k, bytes, a, n),
                };
                let (y, u, v) = (map(0, 1), map(1, 4), map(2, 4));
                if y.fits_i16() && u.fits_i16() && v.fits_i16() {
                    return Fast::RgbBox420 {
                        k,
                        div: (k * kv) as u32,
                        y,
                        u,
                        v,
                        alpha,
                    };
                }
            }
        }
        let [y, u, v] = spec.chans;
        let k = box_k(&groups[0].htab[0]);
        let kv = box_k(&groups[0].vtab[0]);
        let packed422 = spec.loader == Loader::Direct
            && spec.alpha.is_none()
            && !spec.rgb
            && spec.planes == 1
            && y.step == 2
            && u.step == 4
            && v.offset == u.offset + 2
            && transform.identity
            && groups.len() == 2;
        if packed422
            && groups[0].direct
            && box_k(&groups[1].htab[1]) == 1
            && box_k(&groups[1].htab[2]) == 1
            && box_k(&groups[1].vtab[0]) == 2
        {
            return Fast::Packed422Same {
                y_odd: y.offset == 1,
            };
        }
        if packed422
            && groups[1].fused[0]
            && (k == 2 || k == 4)
            && groups[0].htab[0]
                .as_ref()
                .is_some_and(|t| t.start[0] % 2 == 0)
            && box_k(&groups[1].htab[1]) == k
            && box_k(&groups[1].htab[2]) == k
            && matches!(kv, 1 | 2 | 4 | 8)
        {
            let shift_y = (k * kv).trailing_zeros();
            return Fast::Packed422 {
                k,
                shift_y,
                shift_c: shift_y + 1,
                y_odd: y.offset == 1,
            };
        }
        Fast::None
    }

    /// Name of the dedicated kernel that draws this layer, or `"general"`
    /// for the general sum-filter-transform steps.
    pub fn path(&self) -> &'static str {
        match self.fast {
            Fast::None => "general",
            Fast::Rgb420 { .. } => "rgb420",
            Fast::Packed422 { .. } => "packed422",
            Fast::Packed422Same { .. } => "packed422-same",
            Fast::Bgra { .. } => "bgra",
            Fast::RgbBox420 { .. } => "rgb-box420",
            Fast::BgraBox { .. } => "bgra-box",
            Fast::Planar420Same { .. } => "planar420-same",
            Fast::Planar420 { .. } => "planar420",
        }
    }

    /// Move the picture (tables don't depend on its position).
    pub fn set_picture(&mut self, picture: PixelRect) {
        debug_assert_eq!(
            (picture.width, picture.height),
            (self.key.picture.width, self.key.picture.height)
        );
        self.key.picture = picture;
    }

    fn chan(&self, c: usize) -> super::formats::Chan {
        if c == 3 {
            self.spec.alpha.unwrap()
        } else {
            self.spec.chans[c]
        }
    }
}

/// A layer's filtered pixels, kept while its source doesn't change: the
/// part of its picture inside the canvas, in group samples.
pub struct PixelCache {
    pub key: LayerKey,
    /// Canvas size it was made for
    pub canvas: (usize, usize),
    /// The part of the picture it holds (inside the canvas), relative to
    /// the picture
    inside: PixelRect,
    groups: Vec<CacheGroup>,
}

/// Columns `x0..x0 + w` and rows `y0..y0 + h` of a group's picture: one
/// plane per output channel, then alpha.
struct CacheGroup {
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    planes: Vec<Vec<u8>>,
}

impl PixelCache {
    pub fn new(prepared: &Prepared, canvas_width: usize, canvas_height: usize) -> PixelCache {
        let alpha = prepared.spec.alpha.is_some() as usize;
        let pic = prepared.key.picture;
        let clip = pic.clip(canvas_width as u32, canvas_height as u32);
        PixelCache {
            key: prepared.key.clone(),
            canvas: (canvas_width, canvas_height),
            inside: Self::inside(prepared, canvas_width, canvas_height),
            groups: prepared
                .groups
                .iter()
                .map(|g| {
                    // Rounded out like `draw_rect` (odd canvas edges).
                    let span = |start: i32, len: i32, n: usize| {
                        if len <= 0 {
                            return (0, 0);
                        }
                        let (start, end) = (start as usize, (start + len) as usize);
                        let (a, b) = (start / g.sub, end.div_ceil(g.sub).min(n));
                        (a.min(b), b.saturating_sub(a))
                    };
                    let (x0, w) = span(clip.x - pic.x, clip.width, g.w);
                    let (y0, h) = span(clip.y - pic.y, clip.height, g.h);
                    CacheGroup {
                        x0,
                        y0,
                        w,
                        h,
                        planes: vec![vec![0u8; w * h]; g.outputs.len() + alpha],
                    }
                })
                .collect(),
        }
    }

    fn inside(prepared: &Prepared, canvas_width: usize, canvas_height: usize) -> PixelRect {
        let pic = prepared.key.picture;
        let clip = pic.clip(canvas_width as u32, canvas_height as u32);
        PixelRect {
            x: clip.x - pic.x,
            y: clip.y - pic.y,
            ..clip
        }
    }

    /// Whether this cache can hold `prepared`'s layer on this canvas: same
    /// tables and the same part of the picture on the canvas (a layer moved
    /// within the canvas keeps its cache).
    pub fn fits(&self, prepared: &Prepared, canvas_width: usize, canvas_height: usize) -> bool {
        self.canvas == (canvas_width, canvas_height)
            && self.key.same_tables(&prepared.key)
            && self.inside == Self::inside(prepared, canvas_width, canvas_height)
    }

    /// Bytes held.
    pub fn bytes(&self) -> usize {
        self.groups
            .iter()
            .flat_map(|g| &g.planes)
            .map(Vec::len)
            .sum()
    }

    /// The cache's planes (tests).
    #[doc(hidden)]
    pub fn planes_mut(&mut self) -> Vec<&mut Vec<u8>> {
        self.groups
            .iter_mut()
            .flat_map(|g| g.planes.iter_mut())
            .collect()
    }

    fn view(&mut self) -> CacheView {
        CacheView {
            groups: self
                .groups
                .iter_mut()
                .map(|g| CacheGroupView {
                    x0: g.x0,
                    y0: g.y0,
                    w: g.w,
                    h: g.h,
                    planes: g
                        .planes
                        .iter_mut()
                        .map(|p| (p.as_mut_ptr(), p.len()))
                        .collect(),
                })
                .collect(),
        }
    }
}

struct CacheGroupView {
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    planes: Vec<(*mut u8, usize)>,
}

/// Raw access to a [`PixelCache`] from the strips (disjoint rows).
pub struct CacheView {
    groups: Vec<CacheGroupView>,
}

unsafe impl Send for CacheView {}
unsafe impl Sync for CacheView {}

impl CacheView {
    /// Samples `j0..j0 + n` of group row `y` (picture coordinates).
    ///
    /// # Safety
    /// As for [`Canvas::row`]: only rows of the caller's strip.
    #[allow(clippy::mut_from_ref)]
    unsafe fn span(&self, group: usize, plane: usize, y: usize, j0: usize, n: usize) -> &mut [u8] {
        let g = &self.groups[group];
        let (ptr, len) = g.planes[plane];
        assert!(y >= g.y0 && y < g.y0 + g.h && j0 >= g.x0 && j0 + n <= g.x0 + g.w);
        let start = (y - g.y0) * g.w + (j0 - g.x0);
        assert!(start + n <= len);
        unsafe { std::slice::from_raw_parts_mut(ptr.add(start), n) }
    }
}

/// How a layer uses its pixel cache this frame.
pub enum CacheMode<'a> {
    Off,
    /// Draw the whole picture into the cache, then from it
    Build(&'a CacheView),
    /// Draw from the cache
    Use(&'a CacheView),
}

/// One layer to draw this frame.
pub struct LayerInput<'a> {
    pub source: SourceFrame<'a>,
    pub prepared: &'a Prepared,
    pub cache: CacheMode<'a>,
}

/// Everything needed to draw a frame.
pub struct Frame<'a> {
    pub canvas: &'a Canvas,
    pub plan: &'a Plan,
    /// By layout index
    pub layers: Vec<Option<LayerInput<'a>>>,
    /// Background colour in the canvas's model; `None`: transparent
    pub background: Option<[u8; 3]>,
}

/// Turn a cache into a view for one frame.
pub fn cache_view(cache: &mut PixelCache) -> CacheView {
    cache.view()
}

/// Draw `frame`, on `pool` in strips if given.
pub fn render(frame: &Frame, pool: Option<&StripPool>) {
    let h = frame.canvas.height;
    let strips = match pool {
        Some(pool) if pool.threads() > 1 => (pool.threads() * 4).min(h / 16).max(1),
        _ => 1,
    };
    let edges: Vec<usize> = (0..=strips)
        .map(|i| {
            let e = i * h / strips;
            if i == strips { h } else { e & !1 }
        })
        .collect();
    match pool {
        Some(pool) if strips > 1 => {
            pool.run(strips, &|i| render_strip(frame, edges[i], edges[i + 1]))
        }
        _ => render_strip(frame, 0, h),
    }
}

fn rows_in(r: &PixelRect, y0: usize, y1: usize) -> (usize, usize) {
    let a = (r.y.max(0) as usize).max(y0);
    let b = ((r.y + r.height).max(0) as usize).min(y1);
    (a, b.max(a))
}

/// Draw rows `y0..y1` (even bounds) of the frame.
pub fn render_strip(frame: &Frame, y0: usize, y1: usize) {
    let canvas = frame.canvas;
    if y0 >= y1 {
        return;
    }
    for r in &frame.plan.background {
        fill(canvas, r, y0, y1, frame.background, 255);
    }
    SCRATCH.with(|s| {
        let s = &mut *s.borrow_mut();
        for item in &frame.plan.items {
            match item {
                Item::Fill { rect, argb, alpha } => {
                    let bgr_or_yuv = argb_to(canvas.model, *argb);
                    fill(canvas, rect, y0, y1, Some(bgr_or_yuv), *alpha);
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
                    let job = Job {
                        canvas,
                        src: &input.source,
                        prep: input.prepared,
                        pic: *picture,
                        opacity: *opacity,
                    };
                    match input.cache {
                        CacheMode::Off => {
                            for r in visible {
                                draw_rect(&job, s, r, y0, y1, Sink::Canvas);
                            }
                        }
                        CacheMode::Build(cache) => {
                            let whole = picture.clip(canvas.width as u32, canvas.height as u32);
                            draw_rect(&job, s, &whole, y0, y1, Sink::Cache(cache));
                            for r in visible {
                                from_cache(&job, cache, r, y0, y1);
                            }
                        }
                        CacheMode::Use(cache) => {
                            for r in visible {
                                from_cache(&job, cache, r, y0, y1);
                            }
                        }
                    }
                }
            }
        }
    });
    // A BGRA canvas stays premultiplied: that's what ports carry.
}

/// Fill `r` (rows within `y0..y1`) with a colour in the canvas's model at
/// opacity `alpha`; `None` is transparent (BGRA canvas only).
fn fill(canvas: &Canvas, r: &PixelRect, y0: usize, y1: usize, color: Option<[u8; 3]>, alpha: u8) {
    let r = r.clip(canvas.width as u32, canvas.height as u32);
    if r.is_empty() || alpha == 0 {
        return;
    }
    let (x0, x1) = (r.x as usize, (r.x + r.width) as usize);
    let (ra, rb) = rows_in(&r, y0, y1);
    match canvas.format {
        CanvasFormat::Bgra => {
            for y in ra..rb {
                // SAFETY: row of this strip.
                let row = unsafe { canvas.row(0, y) };
                let dst = &mut row[x0 * 4..x1 * 4];
                match color {
                    // Model is RGB: [r, g, b].
                    Some(c) => kernels::fill_bgra(dst, [c[2], c[1], c[0]], alpha),
                    None => dst.fill(0),
                }
            }
        }
        format => {
            let c = color.unwrap_or([16, 128, 128]);
            for y in ra..rb {
                // SAFETY: row of this strip.
                kernels::fill_plane(&mut unsafe { canvas.row(0, y) }[x0..x1], c[0], alpha);
            }
            // Chroma of every 2×2 block the rectangle touches.
            let (cx0, cx1) = (x0 / 2, x1.div_ceil(2));
            for cy in ra / 2..rb.div_ceil(2) {
                if format == CanvasFormat::Nv12 {
                    // SAFETY: chroma row of this strip (strips are even).
                    let row = unsafe { canvas.row(1, cy) };
                    kernels::fill_pairs(&mut row[cx0 * 2..cx1 * 2], c[1], c[2], alpha);
                } else {
                    for (p, v) in [(1, c[1]), (2, c[2])] {
                        // SAFETY: as above.
                        kernels::fill_plane(&mut unsafe { canvas.row(p, cy) }[cx0..cx1], v, alpha);
                    }
                }
            }
        }
    }
}

struct Job<'a> {
    canvas: &'a Canvas,
    src: &'a SourceFrame<'a>,
    prep: &'a Prepared,
    pic: PixelRect,
    opacity: u8,
}

#[derive(Clone, Copy)]
enum Sink<'a> {
    Canvas,
    Cache(&'a CacheView),
}

/// Per-thread buffers.
#[derive(Default)]
struct Scratch {
    /// Per plane, the vertical sums of the (up to) two luma rows
    acc: [[Vec<u16>; 2]; 3],
    /// First loaded byte and length of `acc` per plane
    acc_b0: [usize; 3],
    acc_len: [usize; 3],
    /// Per plane, the vertical sum of the chroma row (planes with their own
    /// chroma rows)
    accc: [Vec<u16>; 3],
    accc_b0: [usize; 3],
    accc_len: [usize; 3],
    /// Per 64 loaded bytes of an alpha source's rows: anything visible
    mask: Vec<u8>,
    runs: Vec<(usize, usize)>,
    tmp16: Vec<u16>,
    tmp32: Vec<u32>,
    q: [Vec<u16>; 4],
    zero: Vec<u16>,
    out: [Vec<u8>; 4],
    loaded: Vec<u8>,
    /// Two rows reduced to output pixels ([`Fast::RgbBox420`])
    red: [Vec<u8>; 2],
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::default());
}

fn grow<T: Default + Clone>(v: &mut Vec<T>, n: usize) {
    if v.len() < n {
        v.resize(n, T::default());
    }
}

/// Byte range `[b0, b1)` of plane `p` that channels of `group` read for
/// group columns `j0..j1`, or `None`.
fn byte_range(
    prep: &Prepared,
    group: &Group,
    p: usize,
    j0: usize,
    j1: usize,
) -> Option<(usize, usize)> {
    let mut range: Option<(usize, usize)> = None;
    for c in 0..4 {
        if !group.needed[c] || prep.chan(c).plane != p {
            continue;
        }
        let ch = prep.chan(c);
        let (s0, s1) = group.htab[c].as_ref().unwrap().source_range(j0, j1);
        if s1 <= s0 {
            continue;
        }
        let b0 = s0 * ch.step + ch.offset;
        let b1 = (s1 - 1) * ch.step + ch.offset + 1;
        range = Some(match range {
            Some((a, b)) => (a.min(b0), b.max(b1)),
            None => (b0, b1),
        });
    }
    range
}

fn union(a: Option<(usize, usize)>, b: Option<(usize, usize)>) -> Option<(usize, usize)> {
    match (a, b) {
        (Some((a0, a1)), Some((b0, b1))) => Some((a0.min(b0), a1.max(b1))),
        (x, None) | (None, x) => x,
    }
}

/// Sum source rows of plane `p` for group row `gy` of table `vt` into `acc`
/// over loaded bytes `[b0, b1)` (aligned; `b1` may pass the row end: the
/// rest is zero).
#[allow(clippy::too_many_arguments)]
fn vertical(
    src: &SourceFrame,
    spec: &FormatSpec,
    p: usize,
    vt: &Table,
    gy: usize,
    b0: usize,
    b1: usize,
    acc: &mut Vec<u16>,
    loaded: &mut Vec<u8>,
    mut mask: Option<&mut [u8]>,
) {
    let row_bytes = src.format.loaded_row_bytes(p, src.width);
    let end = b1.min(row_bytes);
    let n = b1 - b0;
    grow(acc, n);
    let acc = &mut acc[..n];
    let live = end.saturating_sub(b0);
    let start = vt.start[gy] as usize;
    // Rows read as they are, straight from the frame.
    if spec.loader == Loader::Direct && mask.is_none() && vt.taps <= 16 {
        let mut rows: [&[u8]; 16] = [&[]; 16];
        let mut weights = [0u16; 16];
        let mut count = 0;
        for (t, &w) in vt.weights(gy).iter().enumerate() {
            if w != 0 {
                rows[count] = &src.row(p, start + t)[b0..end];
                weights[count] = w;
                count += 1;
            }
        }
        kernels::vacc_rows(&mut acc[..live], &rows[..count], &weights[..count]);
        acc[live..].fill(0);
        return;
    }
    let mut first = true;
    for (t, &w) in vt.weights(gy).iter().enumerate() {
        if w == 0 {
            continue;
        }
        let row = src.row(p, start + t);
        let bytes: &[u8] = if spec.loader == Loader::Direct {
            &row[b0..end]
        } else {
            grow(loaded, live);
            load(spec.loader, row, b0, end, loaded);
            &loaded[..live]
        };
        if let Some(mask) = mask.as_deref_mut() {
            // Premultiplied: a block with any visible pixel has a non-zero byte.
            for (m, chunk) in mask.iter_mut().zip(bytes.chunks(64)) {
                if *m == 0 {
                    let (words, rest) = chunk.as_chunks::<8>();
                    let any = words.iter().fold(0u64, |a, w| a | u64::from_ne_bytes(*w))
                        | rest.iter().fold(0u64, |a, &b| a | b as u64);
                    *m = (any != 0) as u8;
                }
            }
        }
        if first {
            kernels::vacc_first(&mut acc[..live], bytes, w);
            first = false;
        } else {
            kernels::vacc_add(&mut acc[..live], bytes, w);
        }
    }
    if first {
        acc[..live].fill(0);
    }
    acc[live..].fill(0);
}

/// Align a byte range to the plane's pixel groups.
fn align(spec: &FormatSpec, p: usize, (b0, b1): (usize, usize)) -> (usize, usize) {
    let g = spec.group[p].max(1);
    (b0 / g * g, b1.div_ceil(g) * g)
}

/// Draw the part of `r` (canvas pixels; even on a 4:2:0 canvas) within rows
/// `y0..y1` into `sink`.
fn draw_rect(job: &Job, s: &mut Scratch, r: &PixelRect, y0: usize, y1: usize, sink: Sink) {
    let prep = job.prep;
    let spec = &prep.spec;
    let pic = job.pic;
    let (ra, rb) = rows_in(r, y0, y1);
    if ra >= rb || r.width <= 0 {
        return;
    }
    let luma = &prep.groups[0];
    let chroma = prep.groups.get(1);
    let step = if chroma.is_some() { 2 } else { 1 };
    // Group columns of the span.
    let gx0 = (r.x - pic.x) as usize;
    let gx1 = gx0 + r.width as usize;
    let n = gx1 - gx0;
    // Rectangles are even except at the right and bottom edge of an odd
    // sized canvas: the last chroma column and row then cover one luma
    // column or row outside the canvas (inside the picture, which is even).
    let (cx0, cx1) = (gx0 / 2, chroma.map_or(0, |c| gx1.div_ceil(2).min(c.w)));

    // Byte ranges per plane.
    let mut luma_range: [Option<(usize, usize)>; 3] = [None; 3];
    let mut chroma_range: [Option<(usize, usize)>; 3] = [None; 3];
    for p in 0..spec.planes {
        let mut lr = byte_range(prep, luma, p, gx0, gx1);
        if let Some(c) = chroma {
            let cr = byte_range(prep, c, p, cx0, cx1);
            if c.fused[p] {
                lr = union(lr, cr);
            } else {
                chroma_range[p] = cr.map(|x| align(spec, p, x));
            }
        }
        if luma.vtab[p].is_some() && !luma.direct {
            luma_range[p] = lr.map(|x| align(spec, p, x));
        }
    }
    grow(&mut s.zero, n);
    for q in &mut s.q {
        grow(q, n);
    }
    for o in &mut s.out {
        grow(o, n);
    }
    if let (Fast::Bgra { alpha }, Sink::Canvas) = (prep.fast, sink) {
        let luma = &prep.groups[0];
        let vt = luma.vtab[0].as_ref().unwrap();
        let b = luma.htab[0].as_ref().unwrap().start[gx0] as usize * 4;
        let x = (pic.x + gx0 as i32) as usize;
        for y in ra..rb {
            let gy = (y as i32 - pic.y) as usize;
            let src = &job.src.row(0, vt.start[gy] as usize)[b..b + 4 * n];
            // SAFETY: row of this strip.
            let dst = &mut unsafe { job.canvas.row(0, y) }[x * 4..(x + n) * 4];
            kernels::over_bgra(dst, src, alpha, job.opacity);
        }
        return;
    }
    if let Fast::Rgb420 { y, u, v, alpha } = prep.fast {
        match alpha {
            None => draw_rgb420(job, s, (ra, rb), (gx0, gx1), (cx0, cx1), sink, [y, u, v]),
            Some(a) => draw_rgba420(job, s, (ra, rb), (gx0, gx1), (cx0, cx1), sink, [y, u, v], a),
        }
        return;
    }
    if let Fast::Planar420Same { nv12 } = prep.fast {
        draw_planar420_same(job, s, (ra, rb), (gx0, gx1), (cx0, cx1), sink, nv12);
        return;
    }
    if let Fast::Planar420 { nv12, luma, chroma } = prep.fast {
        draw_planar420(
            job,
            s,
            (ra, rb),
            (gx0, gx1),
            (cx0, cx1),
            sink,
            nv12,
            [luma, chroma],
        );
        return;
    }

    // Sources with alpha: skip output columns whose source pixels are all
    // fully transparent.
    // (The box path for RGBA finds transparent blocks after reducing.)
    let bgra_box = matches!((prep.fast, sink), (Fast::BgraBox { .. }, Sink::Canvas));
    let masked = spec.alpha.is_some()
        && luma_range[0].is_some()
        && !matches!(prep.fast, Fast::RgbBox420 { .. })
        && !bgra_box;

    let mut y = ra;
    while y < rb {
        let gy = (y as i32 - pic.y) as usize;
        let rows_here = step.min(rb - y);
        // Rows summed: both rows of a chroma pair, also when the second is
        // below an odd canvas (see above).
        let rows_sum = step.min(luma.h.saturating_sub(gy));
        if masked {
            let (b0, b1) = luma_range[0].unwrap();
            s.mask.clear();
            s.mask.resize((b1 - b0).div_ceil(64), 0);
        }
        // Vertical sums.
        for p in 0..spec.planes {
            if let (Some((b0, b1)), Some(vt)) = (luma_range[p], &luma.vtab[p]) {
                s.acc_b0[p] = b0;
                s.acc_len[p] = b1 - b0;
                for k in 0..rows_sum {
                    if gy + k < luma.h {
                        let mask = masked.then_some(&mut s.mask[..]);
                        vertical(
                            job.src,
                            spec,
                            p,
                            vt,
                            gy + k,
                            b0,
                            b1,
                            &mut s.acc[p][k],
                            &mut s.loaded,
                            mask,
                        );
                    }
                }
            }
            if let (Some(c), Some((b0, b1))) = (chroma, chroma_range[p]) {
                if c.direct || matches!(prep.fast, Fast::Packed422Same { .. }) {
                    continue;
                }
                if let Some(vt) = &c.vtab[p] {
                    s.accc_b0[p] = b0;
                    s.accc_len[p] = b1 - b0;
                    if gy / 2 < c.h {
                        vertical(
                            job.src,
                            spec,
                            p,
                            vt,
                            gy / 2,
                            b0,
                            b1,
                            &mut s.accc[p],
                            &mut s.loaded,
                            None,
                        );
                    }
                }
            }
        }
        if let (Fast::BgraBox { k: kh, div, alpha }, true) = (prep.fast, bgra_box) {
            let first = luma.htab[0].as_ref().unwrap().start[gx0] as usize * 4 - s.acc_b0[0];
            let red = &mut s.red[0];
            grow(red, 4 * n);
            kernels::box_reduce4(
                &mut red[..4 * n],
                &s.acc[0][0][first..s.acc_len[0]],
                kh,
                div,
            );
            let x = (pic.x + gx0 as i32) as usize;
            // SAFETY: row of this strip.
            let dst = &mut unsafe { job.canvas.row(0, y) }[x * 4..(x + n) * 4];
            kernels::over_bgra(dst, &red[..4 * n], alpha, job.opacity);
            y += step;
            continue;
        }
        if let Fast::RgbBox420 {
            k: kh,
            div,
            y: my,
            u: mu,
            v: mv,
            alpha,
        } = prep.fast
        {
            // Both rows reduced to output pixels (4 bytes each).
            let first = luma.htab[0].as_ref().unwrap().start[gx0] as usize * 4 - s.acc_b0[0];
            let mut red = std::mem::take(&mut s.red);
            for k in 0..2 {
                grow(&mut red[k], 4 * n);
            }
            for k in 0..rows_sum {
                kernels::box_reduce4(
                    &mut red[k][..4 * n],
                    &s.acc[0][k][first..s.acc_len[0]],
                    kh,
                    div,
                );
            }
            if rows_sum == 1 {
                let (a, b) = red.split_at_mut(1);
                b[0][..4 * n].copy_from_slice(&a[0][..4 * n]);
            }
            let rows = [&red[0][..4 * n], &red[1][..4 * n]];
            emit_rgb_pair(
                job,
                s,
                rows,
                (gy, rows_here),
                (gx0, cx0, cx1),
                sink,
                [my, mu, mv],
                alpha,
            );
            s.red = red;
            y += step;
            continue;
        }
        // Luma (or RGB) rows.
        for k in 0..rows_here {
            if gy + k >= luma.h {
                break;
            }
            if luma.direct {
                copy_direct(job, s, luma, gy + k, gx0, n);
                emit(job, s, 0, luma, gy + k, gx0, n, sink);
                continue;
            }
            if let Fast::Packed422 {
                k: kh,
                shift_y,
                y_odd,
                ..
            } = prep.fast
            {
                let first = luma.htab[0].as_ref().unwrap().start[gx0] as usize * 2 - s.acc_b0[0];
                kernels::packed422_luma(
                    &mut s.out[0][..n],
                    &s.acc[0][k][first..s.acc_len[0]],
                    kh,
                    shift_y,
                    y_odd,
                );
                emit(job, s, 0, luma, gy + k, gx0, n, sink);
                continue;
            }
            visible_runs(prep, luma, s, masked, gx0, gx1, sink, 0, gy + k);
            for ri in 0..s.runs.len() {
                let (j0, j1) = s.runs[ri];
                luma_row(job, s, luma, k, j0, j1 - j0);
                emit(job, s, 0, luma, gy + k, j0, j1 - j0, sink);
            }
        }
        // Chroma row of a 4:2:0 canvas.
        if let Some(c) = chroma {
            let cy = gy / 2;
            if cy < c.h && cx1 > cx0 && c.direct {
                copy_direct(job, s, c, cy, cx0, cx1 - cx0);
                emit(job, s, 1, c, cy, cx0, cx1 - cx0, sink);
            } else if let (true, Fast::Packed422Same { y_odd }) = (cy < c.h && cx1 > cx0, prep.fast)
            {
                let cn = cx1 - cx0;
                let vt = c.vtab[0].as_ref().unwrap();
                let first = c.htab[1].as_ref().unwrap().start[cx0] as usize * 4;
                let row = |t: usize| &job.src.row(0, vt.start[cy] as usize + t)[first..];
                let [u, v, ..] = &mut s.out;
                kernels::packed422_chroma_avg(&mut u[..cn], &mut v[..cn], row(0), row(1), y_odd);
                emit(job, s, 1, c, cy, cx0, cn, sink);
            } else if let (
                true,
                Fast::Packed422 {
                    k: kh,
                    shift_c,
                    y_odd,
                    ..
                },
            ) = (cy < c.h && cx1 > cx0, prep.fast)
            {
                let cn = cx1 - cx0;
                let first = c.htab[1].as_ref().unwrap().start[cx0] as usize * 4 - s.acc_b0[0];
                let len = s.acc_len[0];
                let [u, v, ..] = &mut s.out;
                kernels::packed422_chroma(
                    &mut u[..cn],
                    &mut v[..cn],
                    &s.acc[0][0][first..len],
                    &s.acc[0][1][first..len],
                    kh,
                    shift_c,
                    y_odd,
                );
                emit(job, s, 1, c, cy, cx0, cn, sink);
            } else if cy < c.h && cx1 > cx0 {
                visible_runs(prep, c, s, masked, cx0, cx1, sink, 1, cy);
                for ri in 0..s.runs.len() {
                    let (j0, j1) = s.runs[ri];
                    chroma_row(job, s, c, j0, j1 - j0);
                    emit(job, s, 1, c, cy, j0, j1 - j0, sink);
                }
            }
        }
        y += step;
    }
}

/// [`Fast::Rgb420`]: rows `ra..rb`, luma columns `gx0..gx1` and chroma
/// columns `cx0..cx1` of the picture, straight from the source pixels.
fn draw_rgb420(
    job: &Job,
    s: &mut Scratch,
    (ra, rb): (usize, usize),
    (gx0, gx1): (usize, usize),
    (cx0, cx1): (usize, usize),
    sink: Sink,
    [my, mu, mv]: [kernels::PixelMap; 3],
) {
    let prep = job.prep;
    let (luma, chroma) = (&prep.groups[0], &prep.groups[1]);
    let vt = luma.vtab[0].as_ref().unwrap();
    let ht = luma.htab[0].as_ref().unwrap();
    let ct = chroma.htab[0].as_ref().unwrap();
    let n = gx1 - gx0;
    let row = |gy: usize| job.src.row(0, vt.start[gy.min(luma.h - 1)] as usize);
    let mut y = ra;
    while y < rb {
        let gy = (y as i32 - job.pic.y) as usize;
        for k in 0..2.min(rb - y) {
            if gy + k >= luma.h {
                break;
            }
            let b = ht.start[gx0] as usize * 4;
            kernels::pixels_map(&mut s.out[0][..n], &row(gy + k)[b..], my);
            emit(job, s, 0, luma, gy + k, gx0, n, sink);
        }
        let cy = gy / 2;
        if cy < chroma.h && cx1 > cx0 {
            let cn = cx1 - cx0;
            let b = ct.start[cx0] as usize * 4;
            let [u, v, ..] = &mut s.out;
            kernels::pixels_map_420(
                &mut u[..cn],
                &mut v[..cn],
                &row(2 * cy)[b..],
                &row(2 * cy + 1)[b..],
                mu,
                mv,
            );
            emit(job, s, 1, chroma, cy, cx0, cn, sink);
        }
        y += 2;
    }
}

/// Runs of 16-pixel blocks of `row` (4-byte pixels, `n` of them) with a
/// visible pixel (alpha byte `alpha` not 0), as pixel ranges, into `runs`.
fn alpha_runs(row: &[u8], n: usize, alpha: usize, runs: &mut Vec<(usize, usize)>) {
    runs.clear();
    let mask = 0xFF_u64 << (8 * alpha) | 0xFF_u64 << (8 * (alpha + 4));
    let mut p = 0;
    while p < n {
        let end = (p + 16).min(n);
        let any = row[p * 4..end * 4]
            .as_chunks::<8>()
            .0
            .iter()
            .fold(0u64, |a, w| a | u64::from_ne_bytes(*w))
            & mask
            != 0
            || (end - p) % 2 == 1 && row[(end - 1) * 4 + alpha] != 0;
        if any {
            match runs.last_mut() {
                Some(last) if last.1 == p => last.1 = end,
                _ => runs.push((p, end)),
            }
        }
        p = end;
    }
}

/// [`draw_rgb420`] for a source with alpha (premultiplied): each pair of
/// source rows straight through [`emit_rgb_pair`], which skips blocks of 16
/// fully transparent pixels.
#[allow(clippy::too_many_arguments)]
fn draw_rgba420(
    job: &Job,
    s: &mut Scratch,
    (ra, rb): (usize, usize),
    (gx0, gx1): (usize, usize),
    (cx0, cx1): (usize, usize),
    sink: Sink,
    maps: [kernels::PixelMap; 3],
    alpha: usize,
) {
    let luma = &job.prep.groups[0];
    let vt = luma.vtab[0].as_ref().unwrap();
    let b = luma.htab[0].as_ref().unwrap().start[gx0] as usize * 4;
    let n = gx1 - gx0;
    let row = |gy: usize| &job.src.row(0, vt.start[gy.min(luma.h - 1)] as usize)[b..b + 4 * n];
    let mut y = ra;
    while y < rb {
        let gy = (y as i32 - job.pic.y) as usize;
        emit_rgb_pair(
            job,
            s,
            [row(gy), row(gy + 1)],
            (gy, 2.min(rb - y)),
            (gx0, cx0, cx1),
            sink,
            maps,
            Some(alpha),
        );
        y += 2;
    }
}

/// Write a pair of canvas rows (`gy`, `gy + 1` of the luma group; chroma
/// row `gy / 2`) from two rows of 4-byte pixels, one per output pixel
/// (premultiplied if `alpha`), with [`Fast::Rgb420`]'s maps. With alpha,
/// blocks of 16 fully transparent pixels are skipped.
#[allow(clippy::too_many_arguments)]
fn emit_rgb_pair(
    job: &Job,
    s: &mut Scratch,
    rows: [&[u8]; 2],
    (gy, rows_here): (usize, usize),
    (gx0, cx0, cx1): (usize, usize, usize),
    sink: Sink,
    [my, mu, mv]: [kernels::PixelMap; 3],
    alpha: Option<usize>,
) {
    let prep = job.prep;
    let (luma, chroma) = (&prep.groups[0], &prep.groups[1]);
    let n = rows[0].len() / 4;
    let mut runs: [Vec<(usize, usize)>; 2] = Default::default();
    for k in 0..2 {
        match alpha {
            Some(a) => alpha_runs(rows[k], n, a, &mut runs[k]),
            None => runs[k].push((0, n)),
        }
    }
    for k in 0..rows_here.min(2) {
        if gy + k >= luma.h {
            break;
        }
        if let (Sink::Cache(cache), Some(_)) = (sink, alpha) {
            for pl in 0..2 {
                // SAFETY: cache rows of this strip.
                unsafe { cache.span(0, pl, gy + k, gx0, n) }.fill(0);
            }
        }
        for &(p0, p1) in &runs[k] {
            let len = p1 - p0;
            let px = &rows[k][p0 * 4..p1 * 4];
            kernels::pixels_map(&mut s.out[0][..len], px, my);
            if let Some(a) = alpha {
                kernels::gather8(&mut s.out[1][..len], &px[a..], 4);
            }
            emit(job, s, 0, luma, gy + k, gx0 + p0, len, sink);
        }
    }
    let cy = gy / 2;
    if cy >= chroma.h || cx1 <= cx0 {
        return;
    }
    let cn = cx1 - cx0;
    if let (Sink::Cache(cache), Some(_)) = (sink, alpha) {
        for pl in 0..3 {
            // SAFETY: as above.
            unsafe { cache.span(1, pl, cy, cx0, cn) }.fill(0);
        }
    }
    let mut merged = Vec::new();
    merge_runs(&runs[0], &runs[1], &mut merged);
    for (p0, p1) in merged {
        let (c0, c1) = (p0 / 2, p1.div_ceil(2).min(cn));
        if c1 <= c0 {
            continue;
        }
        let len = c1 - c0;
        let (a, b) = (&rows[0][c0 * 8..], &rows[1][c0 * 8..]);
        let [u, v, al, ..] = &mut s.out;
        kernels::pixels_map_420(&mut u[..len], &mut v[..len], a, b, mu, mv);
        if let Some(ab) = alpha {
            for i in 0..len {
                let q = 8 * i + ab;
                let sum = a[q] as u32 + a[q + 4] as u32 + b[q] as u32 + b[q + 4] as u32;
                al[i] = ((sum + 2) >> 2) as u8;
            }
        }
        emit(job, s, 1, chroma, cy, cx0 + c0, len, sink);
    }
}

/// Union of two sorted run lists.
fn merge_runs(a: &[(usize, usize)], b: &[(usize, usize)], out: &mut Vec<(usize, usize)>) {
    out.clear();
    let (mut i, mut j) = (0, 0);
    while i < a.len() || j < b.len() {
        let r = if j >= b.len() || (i < a.len() && a[i].0 <= b[j].0) {
            i += 1;
            a[i - 1]
        } else {
            j += 1;
            b[j - 1]
        };
        match out.last_mut() {
            Some(last) if last.1 >= r.0 => last.1 = last.1.max(r.1),
            _ => out.push(r),
        }
    }
}

/// Runs of group columns `j0..j1` to draw into `s.runs`: all of them, or
/// (`masked`) those with a visible source pixel. When building a cache,
/// the skipped columns are cleared in it.
#[allow(clippy::too_many_arguments)]
fn visible_runs(
    prep: &Prepared,
    g: &Group,
    s: &mut Scratch,
    masked: bool,
    j0: usize,
    j1: usize,
    sink: Sink,
    gi: usize,
    gy: usize,
) {
    s.runs.clear();
    if !masked {
        s.runs.push((j0, j1));
        return;
    }
    let ch = prep.spec.alpha.unwrap();
    let t = g.htab[3].as_ref().unwrap();
    let b0 = s.acc_b0[ch.plane];
    let starts = &t.start[j0..j1];
    let taps = t.taps;
    // Runs of visible 64-byte blocks → the columns whose windows touch them
    // (starts never decrease, so two binary searches per run).
    let mut c = 0;
    while c < s.mask.len() {
        if s.mask[c] == 0 {
            c += 1;
            continue;
        }
        let ca = c;
        while c < s.mask.len() && s.mask[c] != 0 {
            c += 1;
        }
        let (bytes_a, bytes_b) = (ca * 64 + b0, c * 64 + b0);
        let sa = bytes_a.saturating_sub(ch.offset) / ch.step;
        let sb = (bytes_b.saturating_sub(ch.offset)).div_ceil(ch.step);
        let ja = j0 + starts.partition_point(|&st| st as usize + taps <= sa);
        let jb = j0 + starts.partition_point(|&st| (st as usize) < sb);
        if ja < jb {
            match s.runs.last_mut() {
                Some(last) if last.1 >= ja => last.1 = last.1.max(jb),
                _ => s.runs.push((ja, jb)),
            }
        }
    }
    if let Sink::Cache(cache) = sink {
        let planes = g.outputs.len() + 1;
        for pl in 0..planes {
            // SAFETY: cache rows of this strip.
            unsafe { cache.span(gi, pl, gy, j0, j1 - j0) }.fill(0);
        }
    }
}

/// [`Fast::Planar420Same`]: rows `ra..rb`, luma columns `gx0..gx1` and
/// chroma columns `cx0..cx1` of the picture, straight from the source
/// planes (NV12 chroma onto the NV12 canvas as it is, interleaved).
fn draw_planar420_same(
    job: &Job,
    s: &mut Scratch,
    (ra, rb): (usize, usize),
    (gx0, gx1): (usize, usize),
    (cx0, cx1): (usize, usize),
    sink: Sink,
    nv12: bool,
) {
    let prep = job.prep;
    let pic = job.pic;
    let (luma, chroma) = (&prep.groups[0], &prep.groups[1]);
    let n = gx1 - gx0;
    let sx = luma.htab[0].as_ref().unwrap().start[gx0] as usize;
    let lv = luma.vtab[0].as_ref().unwrap();
    let cv = chroma.vtab[1].as_ref().unwrap();
    let csx = chroma.htab[1].as_ref().unwrap().start[cx0.min(chroma.w.saturating_sub(1))] as usize;
    let cn = cx1.saturating_sub(cx0);
    let mut y = ra;
    while y < rb {
        let gy = (y as i32 - pic.y) as usize;
        for k in 0..2.min(rb - y) {
            if gy + k >= luma.h {
                break;
            }
            let row = &job.src.row(0, lv.start[gy + k] as usize)[sx..sx + n];
            emit_rows(job, 0, luma, gy + k, gx0, &[row], sink);
        }
        let cy = gy / 2;
        if cy < chroma.h && cn > 0 {
            let r = cv.start[cy] as usize;
            if nv12 {
                let row = &job.src.row(1, r)[2 * csx..2 * (csx + cn)];
                if let (Sink::Canvas, CanvasFormat::Nv12) = (sink, job.canvas.format) {
                    // Interleaved as on the canvas: one write per row.
                    let x = (pic.x / 2) as usize + cx0;
                    // SAFETY: chroma row of this strip.
                    let dst = &mut unsafe { job.canvas.row(1, (pic.y / 2) as usize + cy) }
                        [2 * x..2 * (x + cn)];
                    kernels::write_plane(dst, row, None, job.opacity);
                } else {
                    let [u, v, ..] = &mut s.out;
                    grow(u, cn);
                    grow(v, cn);
                    kernels::gather8(&mut u[..cn], row, 2);
                    kernels::gather8(&mut v[..cn], &row[1..], 2);
                    emit_rows(job, 1, chroma, cy, cx0, &[&u[..cn], &v[..cn]], sink);
                }
            } else {
                let u = &job.src.row(1, r)[csx..csx + cn];
                let v = &job.src.row(2, r)[csx..csx + cn];
                emit_rows(job, 1, chroma, cy, cx0, &[u, v], sink);
            }
        }
        y += 2;
    }
}

/// Vertical sums of plane `p` for group row `gy` of [`Fast::Planar420`]:
/// table `vt` over every `vstep`-th row, bytes `b0..b1`, into `acc`.
#[allow(clippy::too_many_arguments)]
fn planar_vsum(
    src: &SourceFrame,
    p: usize,
    vt: &Table,
    gy: usize,
    planar: &Planar,
    rows_in_plane: usize,
    (b0, b1): (usize, usize),
    acc: &mut Vec<u16>,
) {
    grow(acc, b1 - b0);
    let acc = &mut acc[..b1 - b0];
    let row_of = |i: usize| (i * planar.vstep + planar.phase).min(rows_in_plane - 1);
    let start = vt.start[gy] as usize;
    let weights = vt.weights(gy);
    let last = weights.iter().rposition(|&w| w != 0);
    let mut first = true;
    for (t, &w) in weights.iter().enumerate() {
        if w == 0 {
            continue;
        }
        // The row after this one, while this one is summed: rows are short
        // (a thumbnail's 1920 bytes) and, with `vstep`, not next to each
        // other, so the hardware prefetchers ramp up too late.
        let next = if Some(t) == last {
            (gy + 1 < vt.len).then(|| vt.start[gy + 1] as usize)
        } else {
            Some(start + t + 1)
        };
        if let Some(next) = next {
            kernels::prefetch(&src.row(p, row_of(next))[b0..b1]);
        }
        let row = &src.row(p, row_of(start + t))[b0..b1];
        if first {
            kernels::vacc_first(acc, row, w);
            first = false;
        } else {
            kernels::vacc_add(acc, row, w);
        }
    }
    if first {
        acc.fill(0);
    }
}

/// [`Fast::Planar420`]: rows `ra..rb`, luma columns `gx0..gx1` and chroma
/// columns `cx0..cx1` of the picture.
#[allow(clippy::too_many_arguments)]
fn draw_planar420(
    job: &Job,
    s: &mut Scratch,
    (ra, rb): (usize, usize),
    (gx0, gx1): (usize, usize),
    (cx0, cx1): (usize, usize),
    sink: Sink,
    nv12: bool,
    [pl, pc]: [Planar; 2],
) {
    let prep = job.prep;
    let pic = job.pic;
    let src = job.src;
    let (luma, chroma) = (&prep.groups[0], &prep.groups[1]);
    let n = gx1 - gx0;
    let cn = cx1.saturating_sub(cx0);
    let lv = luma.vtab[0].as_ref().unwrap();
    let cv = chroma.vtab[1].as_ref().unwrap();
    // Source bytes of the span: whole boxes, inside the source (box tables
    // are).
    let sx = luma.htab[0].as_ref().unwrap().start[gx0] as usize;
    let luma_bytes = (sx, sx + n * pl.k);
    let csx = if cn > 0 {
        chroma.htab[1].as_ref().unwrap().start[cx0] as usize
    } else {
        0
    };
    let (lh, ch) = (src.height, src.height.div_ceil(2));
    for o in &mut s.out[..2] {
        grow(o, n.max(cn));
    }
    let mut y = ra;
    while y < rb {
        let gy = (y as i32 - pic.y) as usize;
        for k in 0..2.min(rb - y) {
            if gy + k >= luma.h {
                break;
            }
            planar_vsum(src, 0, lv, gy + k, &pl, lh, luma_bytes, &mut s.acc[0][0]);
            kernels::hbox(&mut s.out[0][..n], &s.acc[0][0], pl.k, pl.hbox);
            emit(job, s, 0, luma, gy + k, gx0, n, sink);
        }
        let cy = gy / 2;
        if cy < chroma.h && cn > 0 {
            let [u, v, ..] = &mut s.out;
            if nv12 {
                let bytes = (2 * csx, 2 * (csx + cn * pc.k));
                planar_vsum(src, 1, cv, cy, &pc, ch, bytes, &mut s.acc[1][0]);
                kernels::hbox_pairs(&mut u[..cn], &mut v[..cn], &s.acc[1][0], pc.k, pc.hbox);
            } else {
                let bytes = (csx, csx + cn * pc.k);
                planar_vsum(src, 1, cv, cy, &pc, ch, bytes, &mut s.acc[1][0]);
                planar_vsum(src, 2, cv, cy, &pc, ch, bytes, &mut s.acc[2][0]);
                kernels::hbox(&mut u[..cn], &s.acc[1][0], pc.k, pc.hbox);
                kernels::hbox(&mut v[..cn], &s.acc[2][0], pc.k, pc.hbox);
            }
            emit(job, s, 1, chroma, cy, cx0, cn, sink);
        }
        y += 2;
    }
}

/// Filter luma (or RGB) row `k` of the summed rows, columns `j0..j0 + n`,
/// into `s.out`.
fn luma_row(job: &Job, s: &mut Scratch, g: &Group, k: usize, j0: usize, n: usize) {
    let prep = job.prep;
    for c in 0..4 {
        if !g.needed[c] {
            continue;
        }
        let ch = prep.chan(c);
        let base = ch.offset as isize - s.acc_b0[ch.plane] as isize;
        kernels::extract(
            &mut s.q[c][..n],
            &s.acc[ch.plane][k],
            base,
            ch.step,
            g.htab[c].as_ref().unwrap(),
            j0,
            g.scale[c],
            &mut s.tmp16,
        );
    }
    finish(job, s, 0, g, n);
}

/// Filter the chroma row (fused planes from `s.sum`), columns
/// `j0..j0 + n`, into `s.out`.
fn chroma_row(job: &Job, s: &mut Scratch, c: &Group, j0: usize, n: usize) {
    let prep = job.prep;
    for ci in 0..4 {
        if !c.needed[ci] {
            continue;
        }
        let ch = prep.chan(ci);
        let p = ch.plane;
        if c.fused[p] {
            let base = ch.offset as isize - s.acc_b0[p] as isize;
            let len = s.acc_len[p];
            kernels::extract_pair(
                &mut s.q[ci][..n],
                &s.acc[p][0][..len],
                &s.acc[p][1][..len],
                base,
                ch.step,
                c.htab[ci].as_ref().unwrap(),
                j0,
                c.scale[ci],
                &mut s.tmp32,
            );
        } else {
            let base = ch.offset as isize - s.accc_b0[p] as isize;
            kernels::extract(
                &mut s.q[ci][..n],
                &s.accc[p],
                base,
                ch.step,
                c.htab[ci].as_ref().unwrap(),
                j0,
                c.scale[ci],
                &mut s.tmp16,
            );
        }
    }
    finish(job, s, 1, c, n);
}

/// A direct group's row: the source bytes, deinterleaved, into `s.out`.
fn copy_direct(job: &Job, s: &mut Scratch, g: &Group, gy: usize, j0: usize, n: usize) {
    for (i, &c) in g.outputs.iter().enumerate() {
        let ch = job.prep.chan(c);
        let vt = g.vtab[ch.plane].as_ref().unwrap();
        let row = job.src.row(ch.plane, vt.start[gy] as usize);
        let first = g.htab[c].as_ref().unwrap().start[j0] as usize * ch.step + ch.offset;
        kernels::gather8(&mut s.out[i][..n], &row[first..], ch.step);
    }
}

/// Q6 channels → output channels (u8, premultiplied) and alpha in `s.out`.
fn finish(job: &Job, s: &mut Scratch, _gi: usize, g: &Group, n: usize) {
    let t = &job.prep.transform;
    let alpha = g.needed[3];
    if t.identity {
        for (i, &k) in g.outputs.iter().enumerate() {
            kernels::round_q6(&mut s.out[i][..n], &s.q[k][..n]);
        }
    } else {
        let zero = &s.zero[..n];
        let q: [&[u16]; 3] = std::array::from_fn(|j| if g.needed[j] { &s.q[j][..n] } else { zero });
        let a = alpha.then(|| &s.q[3][..n]);
        for (i, &k) in g.outputs.iter().enumerate() {
            kernels::transform_row(&mut s.out[i][..n], t, k, q, a);
        }
    }
    if alpha {
        let slot = g.outputs.len();
        kernels::round_q6(&mut s.out[slot][..n], &s.q[3][..n]);
    }
}

/// Write the group's finished row (`s.out`) at group row `gy`, columns
/// `j0..j0 + n`.
#[allow(clippy::too_many_arguments)]
fn emit(job: &Job, s: &Scratch, gi: usize, g: &Group, gy: usize, j0: usize, n: usize, sink: Sink) {
    let count = g.outputs.len() + g.needed[3] as usize;
    let outs: [&[u8]; 4] = std::array::from_fn(|i| &s.out[i][..n]);
    emit_rows(job, gi, g, gy, j0, &outs[..count], sink);
}

/// [`emit`] of given rows (one per output channel, then alpha).
fn emit_rows(job: &Job, gi: usize, g: &Group, gy: usize, j0: usize, outs: &[&[u8]], sink: Sink) {
    let n = outs[0].len();
    match sink {
        Sink::Canvas => write_canvas(job, gi, g, gy, j0, outs),
        Sink::Cache(cache) => {
            for (pl, data) in outs.iter().enumerate() {
                // SAFETY: cache rows of this strip's canvas rows.
                unsafe { cache.span(gi, pl, gy, j0, n) }.copy_from_slice(data);
            }
        }
    }
}

/// Write output channels (and alpha last, if the group has it) of group
/// row `gy`, columns `j0..`, into the canvas.
fn write_canvas(job: &Job, gi: usize, g: &Group, gy: usize, j0: usize, outs: &[&[u8]]) {
    let canvas = job.canvas;
    let alpha = g.needed[3].then(|| outs[g.outputs.len()]);
    let n = outs[0].len();
    // Pictures start at even pixels on a 4:2:0 canvas: exact division.
    let y = (job.pic.y / g.sub as i32 + gy as i32) as usize;
    let x = (job.pic.x / g.sub as i32 + j0 as i32) as usize;
    let o = job.opacity;
    // SAFETY (all rows below): rows of this strip.
    match (canvas.format, gi) {
        (CanvasFormat::Bgra, _) => {
            let row = unsafe { canvas.row(0, y) };
            kernels::write_bgra(
                &mut row[x * 4..(x + n) * 4],
                outs[0],
                outs[1],
                outs[2],
                alpha,
                o,
            );
        }
        (_, 0) => {
            let row = unsafe { canvas.row(0, y) };
            kernels::write_plane(&mut row[x..x + n], outs[0], alpha, o);
        }
        (CanvasFormat::Nv12, _) => {
            let row = unsafe { canvas.row(1, y) };
            kernels::write_pairs(&mut row[x * 2..(x + n) * 2], outs[0], outs[1], alpha, o);
        }
        (CanvasFormat::I420, _) => {
            for (p, data) in [(1, outs[0]), (2, outs[1])] {
                let row = unsafe { canvas.row(p, y) };
                kernels::write_plane(&mut row[x..x + n], data, alpha, o);
            }
        }
    }
}

/// Write the visible rectangle `r` from the layer's cache.
fn from_cache(job: &Job, cache: &CacheView, r: &PixelRect, y0: usize, y1: usize) {
    let (ra, rb) = rows_in(r, y0, y1);
    if ra >= rb || r.width <= 0 {
        return;
    }
    let pic = job.pic;
    for (gi, g) in job.prep.groups.iter().enumerate() {
        // Rounded out: the last chroma column and row of an odd canvas (see
        // `draw_rect`).
        let j0 = (r.x - pic.x) as usize / g.sub;
        let j1 = ((r.x + r.width - pic.x) as usize).div_ceil(g.sub).min(g.w);
        if j1 <= j0 {
            continue;
        }
        let n = j1 - j0;
        let planes = g.outputs.len() + g.needed[3] as usize;
        let (ga, gb) = (
            (ra as i32 - pic.y) as usize / g.sub,
            ((rb as i32 - pic.y) as usize).div_ceil(g.sub),
        );
        for gy in ga..gb.min(g.h) {
            let mut outs: [&[u8]; 4] = [&[]; 4];
            for (pl, out) in outs.iter_mut().enumerate().take(planes) {
                // SAFETY: cache rows of this strip, only read here.
                *out = unsafe { cache.span(gi, pl, gy, j0, n) };
            }
            write_canvas(job, gi, g, gy, j0, &outs[..planes]);
        }
    }
}

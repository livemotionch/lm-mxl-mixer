//! The 10-bit mixer: a list of layers in, a v210 frame out. Per output
//! frame: layout → plan → tables → draw; the caller hands the current frame of
//! each layer's source.
//!
//! Tables are kept per layer slot and reused while the source size, crop
//! and picture size stay (a box being moved keeps them). Drawing is on one
//! thread while that takes under 70% of the frame interval, in parallel
//! strips only to keep up ([`Threading`]).

use std::time::Instant;

use super::config::{CompositionBackground, CompositionLayer, PixelRect};
use super::layout::{crop_rect, plan_snapped, Item, MixerLayer, MixerLayout, Plan, SourceGeom};
use super::pool::StripPool;
use super::render10::{
    render10, Canvas10, Frame10, Keying, LayerInput10, LayerKey10, Prepared10, Source10, WipeMask,
};

/// One layer of a frame, bottom first.
#[derive(Clone)]
pub struct Layer10<'a> {
    /// The source's current frame; `None` with `fill`, or nothing to show
    pub source: Option<Source10<'a>>,
    /// Box, crop, fit, opacity and border (canvas fractions)
    pub layer: CompositionLayer,
    /// A solid colour (ARGB) over the box instead of a source
    pub fill: Option<u32>,
    /// Only this part of the canvas (pixels) is drawn: hard wipes
    pub clip: Option<PixelRect>,
    pub keying: Keying,
    /// Soft or shaped wipes: a level per pixel
    pub mask: Option<WipeMask>,
}

impl<'a> Layer10<'a> {
    /// A source over the whole canvas.
    pub fn full(source: Source10<'a>) -> Self {
        Self {
            source: Some(source),
            layer: CompositionLayer {
                fit: super::config::LayerFit::Stretch,
                ..Default::default()
            },
            fill: None,
            clip: None,
            keying: Keying::None,
            mask: None,
        }
    }

    /// A colour over the whole canvas.
    pub fn color(argb: u32, alpha: f64) -> Self {
        Self {
            source: None,
            layer: CompositionLayer {
                alpha,
                ..Default::default()
            },
            fill: Some(argb),
            clip: None,
            keying: Keying::None,
            mask: None,
        }
    }

    pub fn opacity(mut self, alpha: f64) -> Self {
        self.layer.alpha = alpha;
        self
    }

    fn has_alpha(&self) -> bool {
        !matches!(self.keying, Keying::None) || self.mask.is_some()
    }
}

/// When to draw in parallel strips: on one
/// thread while a frame takes under 70% of the interval; in parallel after
/// 3 slower frames in a row; back to one thread when a probe frame (every
/// 250th) takes under 50%.
#[derive(Debug, Default)]
pub struct Threading {
    parallel: bool,
    frames: u32,
    slow: u32,
}

const PROBE_EVERY: u32 = 250;
const SLOW_IN_A_ROW: u32 = 3;

impl Threading {
    pub fn parallel(&mut self) -> bool {
        self.frames = self.frames.wrapping_add(1);
        self.parallel && !self.frames.is_multiple_of(PROBE_EVERY)
    }

    pub fn took(&mut self, parallel: bool, seconds: f64, interval: f64) {
        if !parallel {
            self.parallel = if self.parallel {
                seconds >= 0.5 * interval
            } else {
                self.slow = if seconds > 0.7 * interval { self.slow + 1 } else { 0 };
                self.slow >= SLOW_IN_A_ROW
            };
            if self.parallel {
                self.slow = 0;
            }
        }
    }

    pub fn is_parallel(&self) -> bool {
        self.parallel
    }
}

/// What a drawn frame cost.
#[derive(Debug, Clone, Copy, Default)]
pub struct DrawStats {
    pub seconds: f64,
    pub parallel: bool,
    pub layers: u32,
    pub tables_built: u32,
}

/// A 10-bit mixer for one output: tables per layer slot and the threading
/// decision.
#[derive(Default)]
pub struct Mixer10 {
    prepared: Vec<Option<Box<Prepared10>>>,
    pub threading: Threading,
    /// Always draw on one thread (`Some(false)`) or in strips (`Some(true)`)
    pub force_parallel: Option<bool>,
    /// Decimate strong vertical downscales (`None`: `MIXER_DECIMATE`,
    /// on by default)
    pub decimate: Option<bool>,
}

impl Mixer10 {
    pub fn new() -> Self {
        Self::default()
    }

    /// The plan for `layers` on a `width × height` canvas.
    pub fn plan(layers: &[Layer10], width: usize, height: usize) -> Plan {
        let layout = MixerLayout {
            layers: layers
                .iter()
                .map(|l| MixerLayer {
                    pad: String::new(),
                    layer: l.layer.clone(),
                    shown: l.layer.visible
                        && l.layer.alpha > 0.0
                        && (l.source.is_some() || l.fill.is_some()),
                    fill: l.fill,
                    clip: l.clip,
                })
                .collect(),
            background: CompositionBackground::Black,
        };
        let geoms: Vec<Option<SourceGeom>> = layers
            .iter()
            .map(|l| {
                l.source.map(|s| SourceGeom {
                    width: s.width as u32,
                    height: s.height as u32,
                    par: 1.0,
                    has_alpha: l.has_alpha(),
                })
            })
            .collect();
        plan_snapped(&layout, &geoms, width as u32, height as u32, true, false)
    }

    /// Draw `layers` into `out` (a v210 frame of `width × height`) on
    /// `background` (`None`: over what `out` holds). `interval`: the frame
    /// interval in seconds, for the threading decision.
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &mut self,
        out: &mut [u8],
        width: usize,
        height: usize,
        layers: &[Layer10],
        background: Option<[u16; 3]>,
        pool: Option<&StripPool>,
        interval: f64,
    ) -> DrawStats {
        let started = Instant::now();
        let plan = Self::plan(layers, width, height);
        if self.prepared.len() < layers.len() {
            self.prepared.resize_with(layers.len(), || None);
        }
        let mut stats = DrawStats::default();
        for item in &plan.items {
            let Item::Layer { index, picture, .. } = item else {
                continue;
            };
            let Some(src) = &layers[*index].source else {
                continue;
            };
            let crop = crop_rect(&layers[*index].layer.crop, src.width as u32, src.height as u32);
            let key = LayerKey10 {
                width: src.width,
                height: src.height,
                crop,
                size: (picture.width, picture.height),
            };
            stats.layers += 1;
            let slot = &mut self.prepared[*index];
            if slot.as_ref().is_none_or(|p| p.key != key) {
                *slot = Some(Box::new(Prepared10::with_options(key, self.decimate.unwrap_or_else(super::render10::decimate))));
                stats.tables_built += 1;
            }
        }
        let inputs: Vec<Option<LayerInput10>> = layers
            .iter()
            .enumerate()
            .map(|(i, l)| {
                Some(LayerInput10 {
                    source: l.source?,
                    prepared: self.prepared.get(i)?.as_deref()?,
                    keying: l.keying.clone(),
                    mask: l.mask,
                })
            })
            .collect();
        let canvas = Canvas10::new(out, width, height);
        let frame = Frame10 {
            canvas: &canvas,
            plan: &plan,
            layers: inputs,
            background,
        };
        let parallel = match self.force_parallel {
            Some(p) => p,
            None => self.threading.parallel(),
        };
        render10(&frame, if parallel { pool } else { None });
        stats.seconds = started.elapsed().as_secs_f64();
        stats.parallel = parallel;
        if self.force_parallel.is_none() {
            self.threading.took(parallel, stats.seconds, interval);
        }
        stats
    }
}

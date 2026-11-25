//! Keyer settings and the step from an ME's scene ([`SceneLayer`]s) to
//! the compositor's layers ([`Layer10`]) for one output frame.

use serde::{Deserialize, Serialize};

use crate::compositor::config::{BorderPosition, CompositionLayer, LayerBorder, LayerCrop, LayerFit, PixelRect};
use crate::compositor::kernels10::KeyShape;
use crate::compositor::mixer10::Layer10;
use crate::compositor::render10::{Keying, Source10};
use crate::vision::transition::SceneLayer;

/// How a keyer cuts its source out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum KeyMode {
    /// The source's luma: transparent below `clip`, opaque `soft` above
    #[default]
    Luma,
    /// The key plane of a v210a source (fill + key)
    Linear,
    /// The source opaque in a box (picture in picture)
    Pip,
}

/// Crop of a box's source, fractions of the source.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct Crop {
    pub left: f64,
    pub right: f64,
    pub top: f64,
    pub bottom: f64,
}

/// A box's border.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Border {
    /// Pixels; 0 = none
    pub width: u32,
    /// `#RRGGBB` or `#RRGGBBAA`
    pub color: String,
}

impl Default for Border {
    fn default() -> Self {
        Self {
            width: 0,
            color: "#ffffff".into(),
        }
    }
}

/// A DME box: where an upstream keyer's picture goes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DmeBox {
    pub enabled: bool,
    /// Left and top edge, fractions of the canvas
    pub x: f64,
    pub y: f64,
    /// Width and height, fractions of the canvas
    pub w: f64,
    pub h: f64,
    pub crop: Crop,
    pub border: Border,
}

impl Default for DmeBox {
    /// A third of the canvas in the top right: whole-number sizes (1/2,
    /// 1/3, 1/4 of the source) take the renderer's box filter, about 2–3×
    /// faster than other sizes.
    fn default() -> Self {
        Self {
            enabled: false,
            x: 0.625,
            y: 0.0556,
            w: 1.0 / 3.0,
            h: 1.0 / 3.0,
            crop: Crop::default(),
            border: Border::default(),
        }
    }
}

/// An upstream or downstream keyer's settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct KeyerConfig {
    pub label: String,
    pub mode: KeyMode,
    /// Source id of the fill (and of the key, for a linear key)
    pub source: Option<String>,
    /// Luma: level where the key starts (0–1 of the video range).
    /// Linear: lifts the key's start.
    pub clip: f32,
    /// Luma: width of the soft edge (0–1)
    pub soft: f32,
    /// Linear: steepness (≥ 1)
    pub gain: f32,
    pub invert: bool,
    /// Upstream keyers only: a box (always one for a PiP)
    #[serde(rename = "dme")]
    pub dme: DmeBox,
    /// Fade time of AUTO, ms
    pub rate_ms: u32,
}

impl Default for KeyerConfig {
    fn default() -> Self {
        Self {
            label: String::new(),
            mode: KeyMode::Luma,
            source: None,
            clip: 0.5,
            soft: 0.1,
            gain: 1.0,
            invert: false,
            dme: DmeBox::default(),
            rate_ms: 500,
        }
    }
}

impl KeyerConfig {
    fn keying(&self, has_key: bool) -> Keying {
        match self.mode {
            KeyMode::Luma => Keying::Luma(KeyShape::luma(self.clip, self.soft, self.invert)),
            KeyMode::Linear if has_key => Keying::Linear(KeyShape::linear(self.clip, self.gain, self.invert)),
            // A linear key on a source without a key plane: opaque.
            KeyMode::Linear | KeyMode::Pip => Keying::None,
        }
    }

    /// The keyer's layer at `alpha`, over `source`'s current frame.
    pub fn layer<'a>(&self, source: Source10<'a>, alpha: f64, upstream: bool) -> Layer10<'a> {
        let boxed = upstream && (self.mode == KeyMode::Pip || self.dme.enabled);
        let mut layer = CompositionLayer {
            fit: LayerFit::Stretch,
            alpha,
            ..Default::default()
        };
        if boxed {
            let d = &self.dme;
            layer.x = d.x;
            layer.y = d.y;
            layer.w = d.w.max(0.01);
            layer.h = d.h.max(0.01);
            layer.crop = LayerCrop {
                left: d.crop.left,
                right: d.crop.right,
                top: d.crop.top,
                bottom: d.crop.bottom,
            };
            if d.border.width > 0 {
                layer.border = LayerBorder {
                    width: d.border.width,
                    color: d.border.color.clone(),
                    position: BorderPosition::Outside,
                };
            }
        }
        Layer10 {
            source: Some(source),
            layer,
            fill: None,
            clip: None,
            keying: self.keying(source.key.is_some()),
            mask: None,
        }
    }
}

/// The frame of each source this output frame (by source index).
pub type Frames<'a> = dyn Fn(usize) -> Option<Source10<'a>> + 'a;

/// Compositor layers for a scene. `keyers`/`dsks`: the ME's keyer settings
/// and each one's source index.
pub fn layers<'a>(
    scene: &[SceneLayer],
    frames: &Frames<'a>,
    keyers: &[(KeyerConfig, Option<usize>)],
    dsks: &[(KeyerConfig, Option<usize>)],
    width: u32,
    height: u32,
) -> Vec<Layer10<'a>> {
    let mut out = Vec::with_capacity(scene.len());
    for l in scene {
        match l {
            SceneLayer::Input {
                input,
                rect,
                alpha,
                clip,
                mask,
            } => {
                let Some(src) = frames(*input) else { continue };
                let layer = CompositionLayer {
                    x: rect.x,
                    y: rect.y,
                    w: rect.w,
                    h: rect.h,
                    fit: LayerFit::Stretch,
                    alpha: *alpha,
                    ..Default::default()
                };
                let clip = clip.map(|c| PixelRect {
                    x: (c.x * width as f64).round() as i32,
                    y: (c.y * height as f64).round() as i32,
                    width: ((c.x + c.w) * width as f64).round() as i32 - (c.x * width as f64).round() as i32,
                    height: ((c.y + c.h) * height as f64).round() as i32 - (c.y * height as f64).round() as i32,
                });
                out.push(Layer10 {
                    source: Some(src),
                    layer,
                    fill: None,
                    clip,
                    keying: Keying::None,
                    mask: *mask,
                });
            }
            SceneLayer::Fill { argb, alpha } => out.push(Layer10::color(*argb, *alpha)),
            SceneLayer::Keyer { keyer, alpha } => {
                if let Some((cfg, Some(s))) = keyers.get(*keyer) {
                    if let Some(src) = frames(*s) {
                        out.push(cfg.layer(src, *alpha, true));
                    }
                }
            }
            SceneLayer::Dsk { dsk, alpha } => {
                if let Some((cfg, Some(s))) = dsks.get(*dsk) {
                    if let Some(src) = frames(*s) {
                        out.push(cfg.layer(src, *alpha, false));
                    }
                }
            }
        }
    }
    out
}

/// The sources a scene draws.
pub fn sources_of(
    scene: &[SceneLayer],
    keyers: &[(KeyerConfig, Option<usize>)],
    dsks: &[(KeyerConfig, Option<usize>)],
    out: &mut Vec<usize>,
) {
    for l in scene {
        let s = match l {
            SceneLayer::Input { input, .. } => Some(*input),
            SceneLayer::Keyer { keyer, .. } => keyers.get(*keyer).and_then(|k| k.1),
            SceneLayer::Dsk { dsk, .. } => dsks.get(*dsk).and_then(|k| k.1),
            SceneLayer::Fill { .. } => None,
        };
        if let Some(s) = s {
            if !out.contains(&s) {
                out.push(s);
            }
        }
    }
}

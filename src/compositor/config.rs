//! Layout types the compositor uses: rectangles, layers, crop, borders, fit.

use serde::{Deserialize, Serialize};


/// Canvas background.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum CompositionBackground {
    #[default]
    Black,
    White,
    /// Transparent canvas. The output carries alpha (BGRA), so the
    /// composition can be used as an overlay layer in another composition.
    Transparent,
}

impl CompositionBackground {
    pub fn has_alpha(self) -> bool {
        self == CompositionBackground::Transparent
    }
}

/// How a layer's source is fitted into the layer's box.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum LayerFit {
    /// Keep the source aspect ratio; fill the rest of the box with nothing
    #[default]
    Letterbox,
    /// Scale the source to the box, ignoring its aspect ratio
    Stretch,
}

/// Crop applied to a layer's source, as fractions of the source size.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub struct LayerCrop {
    #[serde(default)]
    pub left: f64,
    #[serde(default)]
    pub right: f64,
    #[serde(default)]
    pub top: f64,
    #[serde(default)]
    pub bottom: f64,
}

impl LayerCrop {
    pub fn is_zero(&self) -> bool {
        self.left == 0.0 && self.right == 0.0 && self.top == 0.0 && self.bottom == 0.0
    }

    /// Crop in source pixels. Always leaves at least one pixel.
    pub fn to_pixels(&self, width: u32, height: u32) -> (u32, u32, u32, u32) {
        let px = |f: f64, size: u32| (f.clamp(0.0, 1.0) * size as f64).round() as u32;
        let (mut left, mut right) = (px(self.left, width), px(self.right, width));
        let (mut top, mut bottom) = (px(self.top, height), px(self.bottom, height));
        if left + right >= width {
            left = 0;
            right = 0;
        }
        if top + bottom >= height {
            top = 0;
            bottom = 0;
        }
        (left, right, top, bottom)
    }
}

/// Where a border is drawn relative to the layer's box.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum BorderPosition {
    /// Inside the box, covering the picture's edge
    #[default]
    Inside,
    /// Outside the box, around the picture
    Outside,
    /// Centred on the box edge
    Center,
}

/// A solid border around a layer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LayerBorder {
    /// Width in output pixels; 0 = no border
    #[serde(default)]
    pub width: u32,
    /// Colour as `#RRGGBB` or `#RRGGBBAA`
    #[serde(default = "default_border_color")]
    pub color: String,
    #[serde(default)]
    pub position: BorderPosition,
}

impl Default for LayerBorder {
    fn default() -> Self {
        Self {
            width: 0,
            color: default_border_color(),
            position: BorderPosition::default(),
        }
    }
}

impl LayerBorder {
    pub fn is_none(&self) -> bool {
        self.width == 0
    }

    /// Colour as big-endian ARGB, as `videotestsrc foreground-color` wants it.
    pub fn argb(&self) -> u32 {
        parse_color(&self.color).unwrap_or(0xFFFF_FFFF)
    }
}

fn default_border_color() -> String {
    "#ffffff".to_string()
}

/// Parse `#RRGGBB` or `#RRGGBBAA` into ARGB.
pub fn parse_color(color: &str) -> Option<u32> {
    let hex = color.trim().strip_prefix('#')?;
    let value = u32::from_str_radix(hex, 16).ok()?;
    match hex.len() {
        6 => Some(0xFF00_0000 | value),
        8 => Some((value >> 8) | ((value & 0xFF) << 24)),
        _ => None,
    }
}

/// One layer of a composition.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompositionLayer {
    /// Stable id (letters, digits, `_`). Assigned by `normalize` if empty.
    #[serde(default)]
    pub id: String,

    /// Display name
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub name: String,

    /// Left edge, fraction of the canvas width
    #[serde(default)]
    pub x: f64,
    /// Top edge, fraction of the canvas height
    #[serde(default)]
    pub y: f64,
    /// Width, fraction of the canvas width
    #[serde(default = "one")]
    pub w: f64,
    /// Height, fraction of the canvas height
    #[serde(default = "one")]
    pub h: f64,

    #[serde(default)]
    pub fit: LayerFit,

    #[serde(default, skip_serializing_if = "LayerCrop::is_zero")]
    pub crop: LayerCrop,

    /// Opacity, 0.0–1.0
    #[serde(default = "one")]
    pub alpha: f64,

    #[serde(default = "yes")]
    pub visible: bool,

    /// Locked layers can't be moved in the editor
    #[serde(default, skip_serializing_if = "is_false")]
    pub locked: bool,

    #[serde(default, skip_serializing_if = "LayerBorder::is_none")]
    pub border: LayerBorder,
}

impl Default for CompositionLayer {
    fn default() -> Self {
        Self {
            id: String::new(),
            name: String::new(),
            x: 0.0,
            y: 0.0,
            w: 1.0,
            h: 1.0,
            fit: LayerFit::default(),
            crop: LayerCrop::default(),
            alpha: 1.0,
            visible: true,
            locked: false,
            border: LayerBorder::default(),
        }
    }
}

fn one() -> f64 {
    1.0
}
fn yes() -> bool {
    true
}
fn is_false(value: &bool) -> bool {
    !value
}

/// A rectangle in output pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PixelRect {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

impl PixelRect {
    pub fn is_empty(&self) -> bool {
        self.width <= 0 || self.height <= 0
    }

    /// The part of this rectangle inside `0..width` × `0..height`.
    pub fn clip(&self, width: u32, height: u32) -> PixelRect {
        let x0 = self.x.max(0);
        let y0 = self.y.max(0);
        let x1 = (self.x + self.width).min(width as i32);
        let y1 = (self.y + self.height).min(height as i32);
        PixelRect {
            x: x0,
            y: y0,
            width: (x1 - x0).max(0),
            height: (y1 - y0).max(0),
        }
    }

    /// This rectangle minus `other`, as up to four rectangles.
    pub fn subtract(&self, other: &PixelRect) -> Vec<PixelRect> {
        let (ax1, ay1) = (self.x + self.width, self.y + self.height);
        let (bx1, by1) = (other.x + other.width, other.y + other.height);
        if self.is_empty()
            || other.is_empty()
            || other.x >= ax1
            || bx1 <= self.x
            || other.y >= ay1
            || by1 <= self.y
        {
            return if self.is_empty() { vec![] } else { vec![*self] };
        }
        let mut out = Vec::new();
        // Above and below the overlap, full width; left and right of it, overlap height.
        let (top, bottom) = (other.y.max(self.y), by1.min(ay1));
        if top > self.y {
            out.push(PixelRect {
                x: self.x,
                y: self.y,
                width: self.width,
                height: top - self.y,
            });
        }
        if bottom < ay1 {
            out.push(PixelRect {
                x: self.x,
                y: bottom,
                width: self.width,
                height: ay1 - bottom,
            });
        }
        if other.x > self.x {
            out.push(PixelRect {
                x: self.x,
                y: top,
                width: other.x - self.x,
                height: bottom - top,
            });
        }
        if bx1 < ax1 {
            out.push(PixelRect {
                x: bx1,
                y: top,
                width: ax1 - bx1,
                height: bottom - top,
            });
        }
        out
    }
}

impl CompositionLayer {
    /// The layer's box in output pixels.
    pub fn pixel_rect(&self, canvas_width: u32, canvas_height: u32) -> PixelRect {
        let (cw, ch) = (canvas_width as f64, canvas_height as f64);
        PixelRect {
            x: (self.x * cw).round() as i32,
            y: (self.y * ch).round() as i32,
            width: ((self.w * cw).round() as i32).max(1),
            height: ((self.h * ch).round() as i32).max(1),
        }
    }

    /// The four border strips (top, bottom, left, right) in output pixels.
    /// Strips that would be empty have zero width or height.
    pub fn border_strips(&self, canvas_width: u32, canvas_height: u32) -> [PixelRect; 4] {
        let rect = self.pixel_rect(canvas_width, canvas_height);
        let b = self.border.width as i32;
        let grow = match self.border.position {
            BorderPosition::Inside => 0,
            BorderPosition::Outside => b,
            BorderPosition::Center => b / 2,
        };
        let outer = PixelRect {
            x: rect.x - grow,
            y: rect.y - grow,
            width: rect.width + 2 * grow,
            height: rect.height + 2 * grow,
        };
        let b = b.min(outer.width).min(outer.height);
        let side_height = (outer.height - 2 * b).max(0);
        [
            PixelRect {
                x: outer.x,
                y: outer.y,
                width: outer.width,
                height: b,
            },
            PixelRect {
                x: outer.x,
                y: outer.y + outer.height - b,
                width: outer.width,
                height: b,
            },
            PixelRect {
                x: outer.x,
                y: outer.y + b,
                width: b,
                height: side_height,
            },
            PixelRect {
                x: outer.x + outer.width - b,
                y: outer.y + b,
                width: b,
                height: side_height,
            },
        ]
    }
}

/// Largest whole-number downscale of a source with transparency (the CPU
/// mixer's box kernels; `k × k ≤ 64`).
pub const MAX_ALPHA_SCALE: u32 = 8;

/// Picture sizes a source with transparency (cropped to `width × height`)
/// may be drawn at, as `(k, width / k, height / k)`: the CPU mixer has
/// dedicated kernels only for whole-number downscales (1:1, 1:2, 1:3, …,
/// up to [`MAX_ALPHA_SCALE`]); any other size takes its general path, 6–7×
/// slower per pixel. Sizes must divide exactly and, below 1:1, be even (the
/// 4:2:0 canvas puts pictures on even pixels).
pub fn alpha_sizes(width: u32, height: u32) -> Vec<(u32, u32, u32)> {
    let mut sizes = vec![(1, width, height)];
    for k in 2..=MAX_ALPHA_SCALE {
        if width % k == 0 && height % k == 0 && (width / k) % 2 == 0 && (height / k) % 2 == 0 {
            sizes.push((k, width / k, height / k));
        }
    }
    sizes
}

impl CompositionLayer {
    /// The size `self` draws a source of `src` pixels at on a `cw × ch`
    /// canvas (after crop and fit) and that source's cropped size.
    pub fn picture_size(&self, cw: u32, ch: u32, src: (u32, u32)) -> ((u32, u32), (u32, u32)) {
        let (left, right, top, bottom) = self.crop.to_pixels(src.0.max(1), src.1.max(1));
        let cropped = (src.0 - left - right, src.1 - top - bottom);
        let aspect = cropped.0 as f64 / cropped.1.max(1) as f64;
        let rect = picture_box(self, cw, ch, Some(aspect)).pixel_rect(cw, ch);
        (
            (rect.width.max(0) as u32, rect.height.max(0) as u32),
            cropped,
        )
    }

    /// Why `self` can't show a source with transparency of `src` pixels at
    /// its size, or `None` if it can (see [`alpha_sizes`]).
    pub fn alpha_size_error(&self, cw: u32, ch: u32, src: (u32, u32)) -> Option<String> {
        let (size, cropped) = self.picture_size(cw, ch, src);
        let sizes = alpha_sizes(cropped.0, cropped.1);
        if sizes.iter().any(|&(_, w, h)| (w, h) == size) {
            return None;
        }
        let list: Vec<String> = sizes.iter().map(|(_, w, h)| format!("{w}×{h}")).collect();
        let name = if self.name.is_empty() {
            &self.id
        } else {
            &self.name
        };
        Some(format!(
            "Layer '{}' shows a source with transparency ({}×{}{}) at {}×{}. Sources with \
             transparency can only be shown at their own size divided by a whole number: {}. \
             For another size, set the source itself to that size (e.g. a web source's width \
             and height).",
            name,
            cropped.0,
            cropped.1,
            if self.crop.is_zero() {
                ""
            } else {
                " after crop"
            },
            size.0,
            size.1,
            list.join(", "),
        ))
    }
}

/// The part of a layer's box the picture covers: the whole box when
/// stretched, the centred letterboxed area otherwise.
pub fn picture_box(
    layer: &CompositionLayer,
    cw: u32,
    ch: u32,
    aspect: Option<f64>,
) -> CompositionLayer {
    let mut fitted = layer.clone();
    let Some(aspect) = aspect.filter(|_| layer.fit == LayerFit::Letterbox) else {
        return fitted;
    };
    let (bw, bh) = (layer.w * cw as f64, layer.h * ch as f64);
    if bh <= 0.0 {
        return fitted;
    }
    if bw / bh > aspect {
        let w = bh * aspect;
        fitted.x += (bw - w) / 2.0 / cw as f64;
        fitted.w = w / cw as f64;
    } else {
        let h = bw / aspect;
        fitted.y += (bh - h) / 2.0 / ch as f64;
        fitted.h = h / ch as f64;
    }
    fitted
}

/// Intermediate sizes for scaling `source` to `target` in steps of at most
/// 2:1 per axis (the legacy GStreamer pipeline; the quality test compares
/// against it).
pub fn cascade(source: Option<(i32, i32)>, target: (i32, i32)) -> [(i32, i32); 2] {
    let Some((sw, sh)) = source else {
        return [target, target];
    };
    let step = |src: i32, dst: i32, factor: i32| (dst * factor).min(src).max(dst);
    let mid1 = (step(sw, target.0, 4), step(sh, target.1, 4));
    let mid2 = (step(mid1.0, target.0, 2), step(mid1.1, target.1, 2));
    [mid1, mid2]
}

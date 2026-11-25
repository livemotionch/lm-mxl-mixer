//! The mixer's layout and what to draw in a frame.
//!
//! [`MixerLayout`] is the whole layout as one immutable value: the running
//! element holds an `Arc` of it and swaps it in one step, so a frame never
//! sees half an update. [`plan`] turns it, with the size and alpha of each
//! layer's current source frame, into the list of things to draw:
//! background rectangles, and per layer (bottom-up) the parts of its
//! picture not covered by opaque layers above, followed by its border.

use super::config::{
    CompositionBackground, CompositionLayer, LayerCrop, PixelRect,
};
use super::config::picture_box;

/// One layer of a [`MixerLayout`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct MixerLayer {
    /// Name of the mixer sink pad that carries the layer's source
    pub pad: String,
    pub layer: CompositionLayer,
    /// Visible, not fully transparent, and with a source (or a fill)
    pub shown: bool,
    /// A solid colour (ARGB) over the layer's box instead of a source: the
    /// layer has no pad. Opaque (alpha 255 and layer opacity 1) fills hide
    /// what is below them like opaque pictures.
    pub fill: Option<u32>,
    /// Only this part of the canvas (pixels) is drawn and hides layers
    /// below: wipes. `None`: the whole picture.
    pub clip: Option<PixelRect>,
}

/// Everything the mixer draws, bottom layer first.
#[derive(Debug, Clone, PartialEq)]
pub struct MixerLayout {
    pub layers: Vec<MixerLayer>,
    pub background: CompositionBackground,
}

impl Default for MixerLayout {
    fn default() -> Self {
        Self {
            layers: Vec::new(),
            background: CompositionBackground::Black,
        }
    }
}

/// What the plan needs to know about a layer's current source frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SourceGeom {
    pub width: u32,
    pub height: u32,
    /// Pixel aspect ratio
    pub par: f64,
    pub has_alpha: bool,
}

/// Crop in source pixels: `(x0, y0, x1, y1)`.
pub fn crop_rect(crop: &LayerCrop, width: u32, height: u32) -> (u32, u32, u32, u32) {
    let (left, right, top, bottom) = crop.to_pixels(width.max(1), height.max(1));
    (left, top, width - right, height - bottom)
}

/// One thing to draw.
#[derive(Debug, Clone, PartialEq)]
pub enum Item {
    Layer {
        /// Index into the layout's layers
        index: usize,
        /// Where the whole (cropped, fitted) picture goes, in canvas pixels;
        /// may reach outside the canvas
        picture: PixelRect,
        /// The parts to draw: inside the canvas, not under opaque layers above
        visible: Vec<PixelRect>,
        /// 0–255
        opacity: u8,
    },
    Fill {
        rect: PixelRect,
        argb: u32,
        /// 0–255
        alpha: u8,
    },
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Plan {
    /// Canvas areas not covered by any opaque layer
    pub background: Vec<PixelRect>,
    pub items: Vec<Item>,
}

#[allow(dead_code)]
fn snap_even(rect: PixelRect) -> PixelRect {
    snap_axes(rect, true, true)
}

/// `rect` with its edges moved to even pixels on the axes asked for.
fn snap_axes(rect: PixelRect, x: bool, y: bool) -> PixelRect {
    let round = |v: i32| v + v.rem_euclid(2);
    let (rx, ry) = (
        |v: i32| if x { round(v) } else { v },
        |v: i32| if y { round(v) } else { v },
    );
    let (x0, y0) = (rx(rect.x), ry(rect.y));
    let (x1, y1) = (rx(rect.x + rect.width), ry(rect.y + rect.height));
    PixelRect {
        x: x0,
        y: y0,
        width: x1 - x0,
        height: y1 - y0,
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

fn subtract_all(rects: Vec<PixelRect>, cut: &[PixelRect]) -> Vec<PixelRect> {
    let mut rects = rects;
    for c in cut {
        rects = rects.iter().flat_map(|r| r.subtract(c)).collect();
    }
    rects.retain(|r| !r.is_empty());
    rects
}

/// What to draw for `sources` (the current frame of each layout layer, if
/// any) on a `width × height` canvas. With `even`, rectangles are moved to
/// even pixels (4:2:0 canvas: a layer's chroma then lines up with its luma).
pub fn plan(
    layout: &MixerLayout,
    sources: &[Option<SourceGeom>],
    width: u32,
    height: u32,
    even: bool,
) -> Plan {
    plan_snapped(layout, sources, width, height, even, even)
}

/// [`plan`] with even snapping per axis: a 4:2:2 canvas (v210) needs even
/// pixels horizontally only.
pub fn plan_snapped(
    layout: &MixerLayout,
    sources: &[Option<SourceGeom>],
    width: u32,
    height: u32,
    even_x: bool,
    even_y: bool,
) -> Plan {
    let even = even_x || even_y;
    let canvas = PixelRect {
        x: 0,
        y: 0,
        width: width as i32,
        height: height as i32,
    };
    let snap = |r: PixelRect| snap_axes(r, even_x, even_y);

    // Picture rectangles of the layers that draw something.
    let pictures: Vec<Option<(PixelRect, CompositionLayer)>> = layout
        .layers
        .iter()
        .zip(sources)
        .map(|(l, src)| {
            if !l.shown {
                return None;
            }
            if l.fill.is_some() {
                // A fill covers its box, whatever the fit.
                let rect = snap(l.layer.pixel_rect(width, height));
                return Some((rect, l.layer.clone()));
            }
            let src = src.as_ref()?;
            if src.width == 0 || src.height == 0 {
                return None;
            }
            let (x0, y0, x1, y1) = crop_rect(&l.layer.crop, src.width, src.height);
            let aspect = (x1 - x0) as f64 * src.par / (y1 - y0) as f64;
            let fitted = picture_box(&l.layer, width, height, Some(aspect));
            let rect = snap(fitted.pixel_rect(width, height));
            Some((rect, fitted))
        })
        .collect();
    let opaque: Vec<Option<PixelRect>> = layout
        .layers
        .iter()
        .zip(sources)
        .zip(&pictures)
        .map(|((l, src), pic)| {
            let (rect, _) = pic.as_ref()?;
            let rect = match &l.clip {
                Some(clip) => intersect(rect, &snap(*clip)),
                None => *rect,
            };
            let opaque_source = match l.fill {
                Some(argb) => argb >> 24 == 0xFF,
                None => !src.as_ref()?.has_alpha,
            };
            (l.layer.alpha >= 1.0 && opaque_source && !rect.is_empty()).then_some(rect)
        })
        .collect();

    let mut out = Plan {
        background: subtract_all(
            vec![canvas],
            &opaque.iter().flatten().copied().collect::<Vec<_>>(),
        ),
        items: Vec::new(),
    };

    for (index, pic) in pictures.iter().enumerate() {
        let Some((picture, fitted)) = pic else {
            continue;
        };
        let above: Vec<PixelRect> = opaque[index + 1..].iter().flatten().copied().collect();
        let mixer_layer = &layout.layers[index];
        let mut shown_part = picture.clip(width, height);
        if let Some(clip) = &mixer_layer.clip {
            shown_part = intersect(&shown_part, &snap(*clip));
        }
        let visible = subtract_all(vec![shown_part], &above);
        let layer = &mixer_layer.layer;
        let opacity = (layer.alpha.clamp(0.0, 1.0) * 255.0).round() as u8;
        if let Some(argb) = mixer_layer.fill {
            let alpha = ((argb >> 24) as f64 * layer.alpha.clamp(0.0, 1.0)).round() as u8;
            if alpha > 0 {
                for rect in visible {
                    out.items.push(Item::Fill { rect, argb, alpha });
                }
            }
        } else if !visible.is_empty() && opacity > 0 {
            out.items.push(Item::Layer {
                index,
                picture: *picture,
                visible,
                opacity,
            });
        }
        if !layer.border.is_none() {
            let argb = layer.border.argb();
            let alpha = ((argb >> 24) as f64 * layer.alpha).round() as u8;
            // Borders follow the (snapped) picture.
            let mut around = fitted.clone();
            if even {
                around.x = picture.x as f64 / width as f64;
                around.y = picture.y as f64 / height as f64;
                around.w = picture.width as f64 / width as f64;
                around.h = picture.height as f64 / height as f64;
            }
            for strip in around.border_strips(width, height) {
                for rect in subtract_all(vec![strip.clip(width, height)], &above) {
                    out.items.push(Item::Fill { rect, argb, alpha });
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::config::{BorderPosition, LayerBorder, LayerFit};
    use super::*;

    fn layer(pad: &str, x: f64, y: f64, w: f64, h: f64) -> MixerLayer {
        MixerLayer {
            pad: pad.into(),
            shown: true,
            layer: CompositionLayer {
                id: pad.into(),
                x,
                y,
                w,
                h,
                fit: LayerFit::Stretch,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn src(alpha: bool) -> Option<SourceGeom> {
        Some(SourceGeom {
            width: 1920,
            height: 1080,
            par: 1.0,
            has_alpha: alpha,
        })
    }

    fn area(rects: &[PixelRect]) -> i32 {
        rects.iter().map(|r| r.width * r.height).sum()
    }

    #[test]
    fn covered_layers_and_background_are_skipped() {
        let layout = MixerLayout {
            layers: vec![
                layer("a", 0.0, 0.0, 1.0, 1.0),
                layer("b", 0.0, 0.0, 1.0, 1.0),
                layer("c", 0.5, 0.5, 0.25, 0.25),
            ],
            background: CompositionBackground::Black,
        };
        let p = plan(
            &layout,
            &[src(false), src(false), src(false)],
            100,
            100,
            true,
        );
        assert!(p.background.is_empty());
        // a is fully under b; b is drawn around c; c whole.
        let layers: Vec<_> = p
            .items
            .iter()
            .map(|i| match i {
                Item::Layer { index, visible, .. } => (*index, area(visible)),
                _ => unreachable!(),
            })
            .collect();
        assert_eq!(layers, vec![(1, 100 * 100 - 26 * 26), (2, 26 * 26)]);

        // A see-through layer on top doesn't hide anything.
        let p = plan(
            &layout,
            &[src(false), src(true), src(false)],
            100,
            100,
            true,
        );
        assert_eq!(p.items.len(), 3);
        // Without a frame, a layer neither draws nor hides.
        let p = plan(&layout, &[src(false), None, None], 100, 100, true);
        assert_eq!(p.items.len(), 1);
        assert!(p.background.is_empty());
        let p = plan(&layout, &[None, None, None], 100, 100, true);
        assert_eq!(area(&p.background), 100 * 100);
    }

    #[test]
    fn letterbox_and_even_snapping() {
        let mut l = layer("a", 0.0, 0.0, 1.0, 1.0);
        l.layer.fit = LayerFit::Letterbox;
        let layout = MixerLayout {
            layers: vec![l],
            background: CompositionBackground::Black,
        };
        let four_three = Some(SourceGeom {
            width: 640,
            height: 480,
            par: 1.0,
            has_alpha: false,
        });
        let p = plan(&layout, &[four_three], 1920, 1080, true);
        let Item::Layer { picture, .. } = &p.items[0] else {
            panic!()
        };
        assert_eq!(
            *picture,
            PixelRect {
                x: 240,
                y: 0,
                width: 1440,
                height: 1080
            }
        );
        assert_eq!(area(&p.background), 480 * 1080);
        // Odd positions snap to even ones.
        let layout = MixerLayout {
            layers: vec![layer("a", 0.0, 0.0, 0.5, 0.5)],
            background: CompositionBackground::Black,
        };
        let p = plan(&layout, &[src(false)], 102, 50, true);
        let Item::Layer { picture, .. } = &p.items[0] else {
            panic!()
        };
        assert_eq!(
            *picture,
            PixelRect {
                x: 0,
                y: 0,
                width: 52,
                height: 26
            }
        );
        let p = plan(&layout, &[src(false)], 102, 50, false);
        let Item::Layer { picture, .. } = &p.items[0] else {
            panic!()
        };
        assert_eq!(
            *picture,
            PixelRect {
                x: 0,
                y: 0,
                width: 51,
                height: 25
            }
        );
    }

    #[test]
    fn borders_follow_the_z_order() {
        let mut low = layer("a", 0.0, 0.0, 0.5, 0.5);
        low.layer.border = LayerBorder {
            width: 2,
            color: "#ff0000".into(),
            position: BorderPosition::Inside,
        };
        let layout = MixerLayout {
            layers: vec![low, layer("b", 0.25, 0.0, 0.5, 1.0)],
            background: CompositionBackground::Black,
        };
        let p = plan(&layout, &[src(false), src(false)], 100, 100, true);
        let fills: Vec<PixelRect> = p
            .items
            .iter()
            .filter_map(|i| match i {
                Item::Fill { rect, .. } => Some(*rect),
                _ => None,
            })
            .collect();
        // "b" starts at x = 25, snapped to 26.
        assert!(fills.iter().all(|r| r.x + r.width <= 26));
        // Fills come right after their layer, before the layer above.
        assert!(matches!(p.items.last(), Some(Item::Layer { index: 1, .. })));
    }
}

//! Transition maths: what the programme shows at a given progress.
//!
//! Wipes have patterns, a soft edge
//! and a direction ([`WipeParams`]): hard wipes of a rectangular pattern are
//! clip rectangles (copies only), the others a mask per pixel.
//!
//! A transition goes from the programme input `a` to the preview input `b`.
//! [`layers`] turns its kind and progress (0.0 = all `a`, 1.0 = all `b`)
//! into a bottom-up list of [`SceneLayer`]s: inputs with a box, opacity and
//! clip, and solid fills. The production's mixer driver maps them to the
//! mixer's layout once per output frame, so a transition is exact to the
//! frame and costs no layout updates from outside.
//!
//! Boxes and clips are fractions of the canvas, like composition layers.

use serde::{Deserialize, Serialize};

use crate::compositor::render10::{WipeMask, WipePattern};

/// How the programme changes from one input to the next.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransitionKind {
    /// Straight cut (an "auto" with this kind cuts)
    Cut,
    /// Cross-dissolve
    #[default]
    Mix,
    /// Fade to a colour, then from it
    Dip,
    /// A hard edge moves across and reveals the next input
    Wipe,
    /// The next input pushes the current one out
    Push,
    /// The next input slides in over the current one
    Slide,
}

impl TransitionKind {
    pub const ALL: [TransitionKind; 6] = [
        TransitionKind::Cut,
        TransitionKind::Mix,
        TransitionKind::Dip,
        TransitionKind::Wipe,
        TransitionKind::Push,
        TransitionKind::Slide,
    ];
}

/// Which way the edge (wipe) or the pictures (push, slide) move.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    #[default]
    LeftToRight,
    RightToLeft,
    TopToBottom,
    BottomToTop,
}

impl Direction {
    /// Unit vector of the movement, in canvas fractions.
    fn vector(self) -> (f64, f64) {
        match self {
            Direction::LeftToRight => (1.0, 0.0),
            Direction::RightToLeft => (-1.0, 0.0),
            Direction::TopToBottom => (0.0, 1.0),
            Direction::BottomToTop => (0.0, -1.0),
        }
    }
}

/// A rectangle in canvas fractions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

impl Rect {
    pub const FULL: Rect = Rect {
        x: 0.0,
        y: 0.0,
        w: 1.0,
        h: 1.0,
    };

    fn moved(self, dx: f64, dy: f64) -> Rect {
        Rect {
            x: self.x + dx,
            y: self.y + dy,
            ..self
        }
    }
}

/// One layer of what the programme shows, bottom first.
#[derive(Debug, Clone, PartialEq)]
pub enum SceneLayer {
    /// A source (by index) in a box
    Input {
        input: usize,
        rect: Rect,
        alpha: f64,
        /// Only this part of the canvas is drawn
        clip: Option<Rect>,
        /// Soft or shaped wipe edge
        mask: Option<WipeMask>,
    },
    /// A solid colour over the whole canvas (ARGB; alpha also scales it)
    Fill { argb: u32, alpha: f64 },
    /// An upstream keyer (by keyer index) at its level
    Keyer { keyer: usize, alpha: f64 },
    /// A downstream keyer (by index) at its level
    Dsk { dsk: usize, alpha: f64 },
}

/// How a wipe looks.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WipeParams {
    pub pattern: WipePattern,
    /// Soft edge, a fraction of the pattern's travel (0: hard)
    pub soft: f32,
    /// From the other side
    pub reverse: bool,
}

impl Default for WipeParams {
    fn default() -> Self {
        Self {
            pattern: WipePattern::Horizontal,
            soft: 0.0,
            reverse: false,
        }
    }
}

/// Smooth start and end (for moves): 3p² − 2p³.
pub fn ease(p: f64) -> f64 {
    let p = p.clamp(0.0, 1.0);
    p * p * (3.0 - 2.0 * p)
}

fn input(input: Option<usize>, rect: Rect, alpha: f64, clip: Option<Rect>) -> Option<SceneLayer> {
    let input = input?;
    (alpha > 0.0).then_some(SceneLayer::Input {
        input,
        rect,
        alpha: alpha.min(1.0),
        clip,
        mask: None,
    })
}

/// `b` revealed by a wipe at `p`: a clip rectangle when the edge is hard and
/// the pattern rectangular, else a mask.
fn wiped(b: Option<usize>, wipe: &WipeParams, p: f64) -> Option<SceneLayer> {
    let b = b?;
    if wipe.soft <= 0.0 {
        if let Some(r) = wipe.pattern.clip_rect(p as f32, wipe.reverse, 10_000, 10_000) {
            let f = |v: i32| v as f64 / 10_000.0;
            return Some(SceneLayer::Input {
                input: b,
                rect: Rect::FULL,
                alpha: 1.0,
                clip: Some(Rect {
                    x: f(r.x),
                    y: f(r.y),
                    w: f(r.width),
                    h: f(r.height),
                }),
                mask: None,
            });
        }
    }
    Some(SceneLayer::Input {
        input: b,
        rect: Rect::FULL,
        alpha: 1.0,
        clip: None,
        mask: Some(WipeMask {
            pattern: wipe.pattern,
            progress: p as f32,
            soft: wipe.soft,
            reverse: wipe.reverse,
        }),
    })
}

/// What the programme shows during a transition from `a` to `b` at
/// `progress` (clamped to 0–1). `None` inputs are black. `dip_argb` is the
/// colour of a dip.
pub fn layers(
    kind: TransitionKind,
    direction: Direction,
    wipe: &WipeParams,
    dip_argb: u32,
    a: Option<usize>,
    b: Option<usize>,
    progress: f64,
) -> Vec<SceneLayer> {
    let p = progress.clamp(0.0, 1.0);
    if p <= 0.0 {
        return input(a, Rect::FULL, 1.0, None).into_iter().collect();
    }
    if p >= 1.0 {
        return input(b, Rect::FULL, 1.0, None).into_iter().collect();
    }
    let (dx, dy) = direction.vector();
    let out: Vec<Option<SceneLayer>> = match kind {
        TransitionKind::Cut => vec![input(a, Rect::FULL, 1.0, None)],
        // `b` blended over `a`: a·(1 − p) + b·p. Without `a`, over black.
        TransitionKind::Mix => vec![
            input(a, Rect::FULL, 1.0, None),
            input(b, Rect::FULL, p, None),
        ],
        // First half: `a` fades to the colour; second half: `b` from it.
        TransitionKind::Dip => {
            let fill = Some(SceneLayer::Fill {
                argb: dip_argb | 0xFF00_0000,
                alpha: 1.0,
            });
            if p < 0.5 {
                vec![fill, input(a, Rect::FULL, 1.0 - 2.0 * p, None)]
            } else {
                vec![fill, input(b, Rect::FULL, 2.0 * p - 1.0, None)]
            }
        }
        TransitionKind::Wipe => vec![input(a, Rect::FULL, 1.0, None), wiped(b, wipe, p)],
        TransitionKind::Push => {
            let e = ease(p);
            vec![
                input(a, Rect::FULL.moved(dx * e, dy * e), 1.0, None),
                input(
                    b,
                    Rect::FULL.moved(dx * (e - 1.0), dy * (e - 1.0)),
                    1.0,
                    None,
                ),
            ]
        }
        TransitionKind::Slide => {
            let e = ease(p);
            vec![
                input(a, Rect::FULL, 1.0, None),
                input(
                    b,
                    Rect::FULL.moved(dx * (e - 1.0), dy * (e - 1.0)),
                    1.0,
                    None,
                ),
            ]
        }
    };
    out.into_iter().flatten().collect()
}

/// How loud `a` and `b` are at `progress` for audio-follow-video:
/// equal-power for dissolves and moves (no dip in loudness), a dip through
/// silence for a dip, a switch for a cut.
pub fn audio_weights(kind: TransitionKind, progress: f64) -> (f64, f64) {
    let p = progress.clamp(0.0, 1.0);
    match kind {
        TransitionKind::Cut => {
            if p >= 1.0 {
                (0.0, 1.0)
            } else {
                (1.0, 0.0)
            }
        }
        TransitionKind::Dip => ((1.0 - 2.0 * p).max(0.0), (2.0 * p - 1.0).max(0.0)),
        _ => {
            let angle = p * std::f64::consts::FRAC_PI_2;
            (angle.cos(), angle.sin())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BLACK: u32 = 0xFF00_0000;
    const W: WipeParams = WipeParams {
        pattern: WipePattern::Horizontal,
        soft: 0.0,
        reverse: false,
    };

    fn alphas(layers: &[SceneLayer]) -> Vec<(Option<usize>, f64)> {
        layers
            .iter()
            .map(|l| match l {
                SceneLayer::Input { input, alpha, .. } => (Some(*input), *alpha),
                SceneLayer::Fill { alpha, .. } => (None, *alpha),
                SceneLayer::Keyer { alpha, .. } => (None, *alpha),
                SceneLayer::Dsk { alpha, .. } => (None, *alpha),
            })
            .collect()
    }

    #[test]
    fn every_kind_starts_on_a_and_ends_on_b() {
        for kind in TransitionKind::ALL {
            for dir in [
                Direction::LeftToRight,
                Direction::RightToLeft,
                Direction::TopToBottom,
                Direction::BottomToTop,
            ] {
                let start = layers(kind, dir, &W, BLACK, Some(0), Some(1), 0.0);
                assert_eq!(alphas(&start), vec![(Some(0), 1.0)], "{kind:?}");
                let end = layers(kind, dir, &W, BLACK, Some(0), Some(1), 1.0);
                assert_eq!(alphas(&end), vec![(Some(1), 1.0)], "{kind:?}");
                // Out of range is clamped.
                assert_eq!(layers(kind, dir, &W, BLACK, Some(0), Some(1), 7.0), end);
            }
        }
    }

    #[test]
    fn mix_blends_b_over_a() {
        let l = layers(
            TransitionKind::Mix,
            Direction::LeftToRight,
            &W,
            BLACK,
            Some(2),
            Some(5),
            0.25,
        );
        assert_eq!(alphas(&l), vec![(Some(2), 1.0), (Some(5), 0.25)]);
        // From black: only b, fading in.
        let l = layers(
            TransitionKind::Mix,
            Direction::LeftToRight,
            &W,
            BLACK,
            None,
            Some(5),
            0.25,
        );
        assert_eq!(alphas(&l), vec![(Some(5), 0.25)]);
    }

    #[test]
    fn dip_goes_through_the_colour() {
        let at = |p| {
            layers(
                TransitionKind::Dip,
                Direction::LeftToRight,
                &W,
                0x00FF_FFFF,
                Some(0),
                Some(1),
                p,
            )
        };
        let l = at(0.25);
        assert_eq!(
            l[0],
            SceneLayer::Fill {
                argb: 0xFFFF_FFFF,
                alpha: 1.0
            }
        );
        assert_eq!(alphas(&l)[1], (Some(0), 0.5));
        // Halfway: only the colour.
        assert_eq!(alphas(&at(0.5)), vec![(None, 1.0)]);
        assert_eq!(alphas(&at(0.75))[1], (Some(1), 0.5));
    }

    #[test]
    fn wipes_clip_hard_rectangles_and_mask_the_rest() {
        let wipe = |pattern, soft, reverse, p| {
            let w = WipeParams { pattern, soft, reverse };
            layers(TransitionKind::Wipe, Direction::LeftToRight, &w, BLACK, Some(0), Some(1), p)
        };
        let clip_of = |l: &[SceneLayer]| match &l[1] {
            SceneLayer::Input { input: 1, clip, mask, rect, .. } => {
                assert_eq!(*rect, Rect::FULL);
                (*clip, *mask)
            }
            other => panic!("{other:?}"),
        };
        let (c, m) = clip_of(&wipe(WipePattern::Horizontal, 0.0, false, 0.3));
        let c = c.unwrap();
        assert!(m.is_none());
        assert_eq!((c.x, c.y, c.h), (0.0, 0.0, 1.0));
        assert!((c.w - 0.3).abs() < 1e-9);
        let (c, _) = clip_of(&wipe(WipePattern::Horizontal, 0.0, true, 0.3));
        let c = c.unwrap();
        assert!((c.x - 0.7).abs() < 1e-9 && (c.w - 0.3).abs() < 1e-9);
        let (c, _) = clip_of(&wipe(WipePattern::Vertical, 0.0, true, 0.4));
        let c = c.unwrap();
        assert!((c.y - 0.6).abs() < 1e-9 && (c.h - 0.4).abs() < 1e-9);
        // Soft edges and round patterns are masks.
        let (c, m) = clip_of(&wipe(WipePattern::Horizontal, 0.1, false, 0.3));
        assert!(c.is_none() && m.unwrap().soft == 0.1);
        let (c, m) = clip_of(&wipe(WipePattern::Iris, 0.0, false, 0.3));
        assert!(c.is_none() && m.unwrap().pattern == WipePattern::Iris);
    }

    #[test]
    fn push_moves_both_and_keeps_them_touching() {
        for p in [0.1, 0.5, 0.9] {
            let l = layers(
                TransitionKind::Push,
                Direction::RightToLeft,
                &W,
                BLACK,
                Some(0),
                Some(1),
                p,
            );
            let rects: Vec<Rect> = l
                .iter()
                .map(|l| match l {
                    SceneLayer::Input { rect, .. } => *rect,
                    _ => unreachable!(),
                })
                .collect();
            // a moves left by ease(p), b follows right behind it.
            assert!((rects[0].x + ease(p)).abs() < 1e-12);
            assert!((rects[1].x - rects[0].x - 1.0).abs() < 1e-12);
            assert_eq!(rects[0].y, 0.0);
        }
        // Slide: a stays.
        let l = layers(
            TransitionKind::Slide,
            Direction::TopToBottom,
            &W,
            BLACK,
            Some(0),
            Some(1),
            0.5,
        );
        match (&l[0], &l[1]) {
            (SceneLayer::Input { rect: a, .. }, SceneLayer::Input { rect: b, .. }) => {
                assert_eq!(*a, Rect::FULL);
                assert!((b.y + 0.5).abs() < 1e-12);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn ease_is_monotonic_and_symmetric() {
        let mut last = 0.0;
        for i in 0..=100 {
            let p = i as f64 / 100.0;
            let e = ease(p);
            assert!(e >= last);
            assert!((ease(1.0 - p) - (1.0 - e)).abs() < 1e-12);
            last = e;
        }
        assert_eq!(ease(0.0), 0.0);
        assert_eq!(ease(1.0), 1.0);
    }

    #[test]
    fn audio_weights_keep_power_or_dip() {
        for i in 0..=10 {
            let p = i as f64 / 10.0;
            let (a, b) = audio_weights(TransitionKind::Mix, p);
            assert!((a * a + b * b - 1.0).abs() < 1e-9);
            let (a, b) = audio_weights(TransitionKind::Dip, p);
            assert!(a == 0.0 || b == 0.0);
        }
        assert_eq!(audio_weights(TransitionKind::Mix, 0.0), (1.0, 0.0));
        let (a, b) = audio_weights(TransitionKind::Wipe, 1.0);
        assert!(a.abs() < 1e-12 && (b - 1.0).abs() < 1e-12);
        assert_eq!(audio_weights(TransitionKind::Cut, 0.5), (1.0, 0.0));
    }
}

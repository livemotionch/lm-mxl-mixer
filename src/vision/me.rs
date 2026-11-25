//! One mix/effects bank (ME): programme and preview buses, auto and manual
//! (T-bar) transitions, upstream keyers that a transition can take along
//! (ties), downstream keyers and fade to black.
//!
//! Ramps run over whole frames (fades, auto, T-bar, cut, FTB). The next
//! transition's elements are chosen as on a broadcast switcher (background
//! and tied upstream keyers); downstream keyers are separate from upstream
//! ones; the clean feed is the programme without DSKs and FTB.
//!
//! Time is the output frame's TAI time in nanoseconds: the engine calls
//! [`Me::frame`] once per output frame, and everything that moves is a
//! function of that time, counted in whole frames. A command only changes
//! the state; the next frame picks it up, so a transition's first frame is
//! the first one drawn after the command, and its last is exactly `to`.
//!
//! Sources are indices (the engine's source list: inputs, key inputs, other
//! MEs); `None` is black.

use super::transition::{self, Direction, SceneLayer, TransitionKind, WipeParams};

/// Transition settings: kind, length and the kind's parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransitionSpec {
    pub kind: TransitionKind,
    pub duration_ms: u32,
    /// Push and slide
    pub direction: Direction,
    pub wipe: WipeParams,
    /// ARGB of a dip
    pub dip_argb: u32,
}

impl Default for TransitionSpec {
    fn default() -> Self {
        Self {
            kind: TransitionKind::Mix,
            duration_ms: 1000,
            direction: Direction::LeftToRight,
            wipe: WipeParams::default(),
            dip_argb: 0xFF00_0000,
        }
    }
}

/// Whole frames a fade of `ms` takes at `frame_ns` per frame (the nearest
/// number, at least 1).
pub fn frames_for(ms: u32, frame_ns: u64) -> u32 {
    let frames = (ms as f64 * 1e6 / frame_ns.max(1) as f64).round();
    frames.clamp(1.0, u32::MAX as f64) as u32
}

/// Something that moves from one level to another over whole frames,
/// started by the first frame after it was asked to.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Ramp {
    from: f64,
    to: f64,
    frames: u32,
    /// Time of its first frame; `None` until that frame is drawn
    start: Option<u64>,
}

impl Ramp {
    /// Level at `time`: `from + (to − from)·k/frames` on the k-th frame
    /// (k = 1 on the first), so the last frame is exactly `to`.
    fn level(&self, time: u64, frame_ns: u64) -> f64 {
        let Some(start) = self.start else {
            return self.from;
        };
        let k = frame_index(start, time, frame_ns) + 1;
        let t = (k as f64 / self.frames.max(1) as f64).min(1.0);
        self.from + (self.to - self.from) * t
    }

    /// Like [`Ramp::level`], continuous in time (audio): 0 at the start.
    fn level_smooth(&self, time: u64, frame_ns: u64) -> f64 {
        let Some(start) = self.start else {
            return self.from;
        };
        let elapsed = time.saturating_sub(start) as f64 + frame_ns as f64;
        let t = (elapsed / (self.frames.max(1) as f64 * frame_ns as f64)).clamp(0.0, 1.0);
        self.from + (self.to - self.from) * t
    }

    fn done(&self, time: u64, frame_ns: u64) -> bool {
        self.start
            .is_some_and(|s| frame_index(s, time, frame_ns) + 1 >= self.frames as u64)
    }
}

/// Frames between `start` and `time`, rounded.
fn frame_index(start: u64, time: u64, frame_ns: u64) -> u64 {
    let frame_ns = frame_ns.max(1);
    (time.saturating_sub(start) + frame_ns / 2) / frame_ns
}

/// A level that fades between 0 and 1 (FTB, keyers).
#[derive(Debug, Clone, Copy, PartialEq)]
struct Fade {
    on: bool,
    ramp: Option<Ramp>,
    /// Level of the last frame drawn
    level: f64,
}

impl Default for Fade {
    fn default() -> Self {
        Self {
            on: false,
            ramp: None,
            level: 0.0,
        }
    }
}

impl Fade {
    fn set(&mut self, on: bool, frames: Option<u32>) {
        self.on = on;
        let to = if on { 1.0 } else { 0.0 };
        match frames {
            Some(frames) if (self.level - to).abs() > f64::EPSILON => {
                self.ramp = Some(Ramp {
                    from: self.level,
                    to,
                    frames,
                    start: None,
                })
            }
            _ => {
                self.ramp = None;
                self.level = to;
            }
        }
    }

    /// Level of the frame at `time`; finishes the ramp on its last frame.
    fn frame(&mut self, time: u64, frame_ns: u64) -> f64 {
        if let Some(ramp) = &mut self.ramp {
            ramp.start.get_or_insert(time);
            self.level = ramp.level(time, frame_ns);
            if ramp.done(time, frame_ns) {
                self.level = ramp.to;
                self.ramp = None;
            }
        }
        self.level
    }

    fn moving(&self) -> bool {
        self.ramp.is_some()
    }
}

/// How a running transition advances.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Drive {
    /// On its own, over whole frames from `ramp.from` to 1
    Auto(Ramp),
    /// Held by the T-bar at this progress
    Manual(f64),
}

#[derive(Debug, Clone, PartialEq)]
struct Active {
    from: Option<usize>,
    to: Option<usize>,
    /// The background changes (else: keys only)
    background: bool,
    /// Tied upstream keyers and whether each was on when it started
    keys: Vec<(usize, bool)>,
    spec: TransitionSpec,
    drive: Drive,
    /// Progress of the last frame drawn
    progress: f64,
}

/// Why a command was refused.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum VisionError {
    #[error("no source {0}")]
    NoInput(usize),
    #[error("no keyer {0}")]
    NoKeyer(usize),
    #[error("a transition is running")]
    InTransition,
    #[error("an auto transition is running")]
    AutoRunning,
    #[error("the next transition has nothing in it (no background, no tied keyer)")]
    NothingToTransition,
}

/// What a frame changed that others want to know.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FrameEvents {
    /// A transition finished (buses swapped) or a fade ended
    pub changed: bool,
}

/// The layers of one ME's outputs for a frame.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MeFrame {
    /// Background, upstream keyers, DSKs, FTB
    pub program: Vec<SceneLayer>,
    /// Background and upstream keyers
    pub clean: Vec<SceneLayer>,
    /// The preview bus
    pub preview: Vec<SceneLayer>,
    pub events: FrameEvents,
}

/// The state machine of one ME. See the module docs.
#[derive(Debug, Clone)]
pub struct Me {
    sources: usize,
    pub frame_ns: u64,
    pgm: Option<usize>,
    pvw: Option<usize>,
    pub spec: TransitionSpec,
    /// Next transition: the background changes
    pub next_background: bool,
    /// Next transition: upstream keyers it takes along
    pub ties: Vec<bool>,
    active: Option<Active>,
    /// T-bar lever position (0–1) and whether pushing it towards 0 is
    /// "forward" (after a manual transition ended at 1)
    tbar_position: f64,
    tbar_inverted: bool,
    ftb: Fade,
    pub ftb_ms: u32,
    keyers: Vec<Fade>,
    pub keyer_ms: Vec<u32>,
    dsks: Vec<Fade>,
    pub dsk_ms: Vec<u32>,
    last_time: Option<u64>,
}

impl Me {
    /// An ME over `sources` sources with `keyers` upstream and `dsks`
    /// downstream keyers at `fps_n/fps_d`. Programme on the first source,
    /// preview on the second, if there are.
    pub fn new(sources: usize, keyers: usize, dsks: usize, fps_n: u32, fps_d: u32) -> Self {
        let frame_ns = (1_000_000_000u64 * fps_d.max(1) as u64) / fps_n.max(1) as u64;
        Self {
            sources,
            frame_ns,
            pgm: (sources > 0).then_some(0),
            pvw: (sources > 1).then_some(1).or((sources > 0).then_some(0)),
            spec: TransitionSpec::default(),
            next_background: true,
            ties: vec![false; keyers],
            active: None,
            tbar_position: 0.0,
            tbar_inverted: false,
            ftb: Fade::default(),
            ftb_ms: 1000,
            keyers: vec![Fade::default(); keyers],
            keyer_ms: vec![500; keyers],
            dsks: vec![Fade::default(); dsks],
            dsk_ms: vec![500; dsks],
            last_time: None,
        }
    }

    pub fn sources(&self) -> usize {
        self.sources
    }

    pub fn program(&self) -> Option<usize> {
        self.pgm
    }

    pub fn preview(&self) -> Option<usize> {
        self.pvw
    }

    /// The sources a running transition goes between (from, to).
    pub fn transition_sources(&self) -> Option<(Option<usize>, Option<usize>)> {
        self.active.as_ref().map(|a| (a.from, a.to))
    }

    /// Progress of the running transition (of the last frame drawn).
    pub fn transition_progress(&self) -> Option<f64> {
        self.active.as_ref().map(|a| a.progress)
    }

    pub fn auto_running(&self) -> bool {
        matches!(
            self.active,
            Some(Active {
                drive: Drive::Auto(_),
                ..
            })
        )
    }

    /// The T-bar lever position the UI should show.
    pub fn tbar_position(&self) -> f64 {
        self.tbar_position
    }

    pub fn ftb_on(&self) -> bool {
        self.ftb.on
    }

    pub fn ftb_level(&self) -> f64 {
        self.ftb.level
    }

    pub fn keyers(&self) -> usize {
        self.keyers.len()
    }

    pub fn dsks(&self) -> usize {
        self.dsks.len()
    }

    pub fn keyer_on(&self, keyer: usize) -> bool {
        self.keyers.get(keyer).is_some_and(|k| k.on)
    }

    pub fn keyer_level(&self, keyer: usize) -> f64 {
        self.keyers.get(keyer).map_or(0.0, |k| k.level)
    }

    pub fn dsk_on(&self, dsk: usize) -> bool {
        self.dsks.get(dsk).is_some_and(|k| k.on)
    }

    pub fn dsk_level(&self, dsk: usize) -> f64 {
        self.dsks.get(dsk).map_or(0.0, |k| k.level)
    }

    fn check_input(&self, input: Option<usize>) -> Result<(), VisionError> {
        match input {
            Some(i) if i >= self.sources => Err(VisionError::NoInput(i)),
            _ => Ok(()),
        }
    }

    /// The lever position where the next transition starts.
    fn tbar_rest(&self) -> f64 {
        if self.tbar_inverted {
            1.0
        } else {
            0.0
        }
    }

    /// Put a source (`None`: black) on the preview bus. Refused while a
    /// transition runs.
    pub fn set_preview(&mut self, input: Option<usize>) -> Result<(), VisionError> {
        self.check_input(input)?;
        if self.active.is_some() {
            return Err(VisionError::InTransition);
        }
        self.pvw = input;
        Ok(())
    }

    /// Put a source on the programme straight away (a cut from the
    /// programme bus); ends a running transition.
    pub fn set_program(&mut self, input: Option<usize>) -> Result<(), VisionError> {
        self.check_input(input)?;
        self.active = None;
        self.tbar_position = self.tbar_rest();
        self.pgm = input;
        Ok(())
    }

    fn finish(&mut self, a: &Active) {
        if a.background {
            self.pgm = a.to;
            self.pvw = a.from;
        }
        for &(k, was_on) in &a.keys {
            if let Some(f) = self.keyers.get_mut(k) {
                f.set(!was_on, None);
            }
        }
        self.active = None;
    }

    fn start(&self, drive: Drive, progress: f64) -> Result<Active, VisionError> {
        let keys: Vec<(usize, bool)> = self
            .ties
            .iter()
            .enumerate()
            .filter(|(k, &t)| t && *k < self.keyers.len())
            .map(|(k, _)| (k, self.keyers[k].on))
            .collect();
        if !self.next_background && keys.is_empty() {
            return Err(VisionError::NothingToTransition);
        }
        Ok(Active {
            from: self.pgm,
            to: self.pvw,
            background: self.next_background,
            keys,
            spec: self.spec,
            drive,
            progress,
        })
    }

    /// Take the next transition now (background and tied keyers). During a
    /// transition: jump to its end.
    pub fn cut(&mut self) -> Result<(), VisionError> {
        let a = match self.active.take() {
            Some(a) => a,
            None => self.start(Drive::Manual(1.0), 1.0)?,
        };
        self.finish(&a);
        self.tbar_position = self.tbar_rest();
        Ok(())
    }

    /// Run the next transition (`spec`) from programme to preview. From a
    /// T-bar position, it carries on from there. An auto of kind `Cut`
    /// cuts.
    pub fn auto(&mut self) -> Result<(), VisionError> {
        match &self.active {
            Some(Active {
                drive: Drive::Auto(_),
                ..
            }) => Err(VisionError::AutoRunning),
            Some(a) => {
                let mut a = a.clone();
                let Drive::Manual(progress) = a.drive else {
                    unreachable!()
                };
                let left = 1.0 - progress;
                let frames = ((frames_for(a.spec.duration_ms, self.frame_ns) as f64) * left)
                    .ceil()
                    .max(1.0) as u32;
                a.drive = Drive::Auto(Ramp {
                    from: progress,
                    to: 1.0,
                    frames,
                    start: None,
                });
                self.active = Some(a);
                Ok(())
            }
            None if self.spec.kind == TransitionKind::Cut || self.spec.duration_ms == 0 => self.cut(),
            None => {
                let frames = frames_for(self.spec.duration_ms, self.frame_ns);
                let a = self.start(
                    Drive::Auto(Ramp {
                        from: 0.0,
                        to: 1.0,
                        frames,
                        start: None,
                    }),
                    0.0,
                )?;
                self.active = Some(a);
                Ok(())
            }
        }
    }

    /// Move the T-bar lever to `position` (0–1). Moving it away from its
    /// rest position runs the transition by hand; at the far end it
    /// finishes (and the far end becomes the rest position); back at rest it
    /// is cancelled. Refused while an auto runs.
    pub fn set_tbar(&mut self, position: f64) -> Result<(), VisionError> {
        if self.auto_running() {
            return Err(VisionError::AutoRunning);
        }
        let position = position.clamp(0.0, 1.0);
        let progress = if self.tbar_inverted {
            1.0 - position
        } else {
            position
        };
        if progress >= 1.0 {
            let a = match self.active.take() {
                Some(a) => a,
                None => self.start(Drive::Manual(1.0), 1.0)?,
            };
            self.finish(&a);
            self.tbar_inverted = !self.tbar_inverted;
            self.tbar_position = position;
        } else if progress <= 0.0 {
            self.active = None;
            self.tbar_position = position;
        } else {
            if self.active.is_none() {
                let mut a = self.start(Drive::Manual(progress), progress)?;
                // A cut has nothing to show halfway: the T-bar mixes.
                if a.spec.kind == TransitionKind::Cut {
                    a.spec.kind = TransitionKind::Mix;
                }
                self.active = Some(a);
            }
            if let Some(a) = &mut self.active {
                a.drive = Drive::Manual(progress);
            }
            self.tbar_position = position;
        }
        Ok(())
    }

    /// Fade to black, or back (`on`: `None` toggles).
    pub fn set_ftb(&mut self, on: Option<bool>, cut: bool) {
        let on = on.unwrap_or(!self.ftb.on);
        let frames = (!cut).then(|| frames_for(self.ftb_ms, self.frame_ns));
        self.ftb.set(on, frames);
    }

    /// Put an upstream keyer on air or take it off (`on`: `None` toggles),
    /// fading over its rate unless `cut`.
    pub fn set_keyer(&mut self, keyer: usize, on: Option<bool>, cut: bool) -> Result<(), VisionError> {
        let ms = *self.keyer_ms.get(keyer).ok_or(VisionError::NoKeyer(keyer))?;
        let frame_ns = self.frame_ns;
        let k = &mut self.keyers[keyer];
        let on = on.unwrap_or(!k.on);
        k.set(on, (!cut).then(|| frames_for(ms, frame_ns)));
        Ok(())
    }

    /// The same for a downstream keyer.
    pub fn set_dsk(&mut self, dsk: usize, on: Option<bool>, cut: bool) -> Result<(), VisionError> {
        let ms = *self.dsk_ms.get(dsk).ok_or(VisionError::NoKeyer(dsk))?;
        let frame_ns = self.frame_ns;
        let k = &mut self.dsks[dsk];
        let on = on.unwrap_or(!k.on);
        k.set(on, (!cut).then(|| frames_for(ms, frame_ns)));
        Ok(())
    }

    /// The ME's outputs at `time`: advances transitions and fades, finishes
    /// them on their last frame.
    pub fn frame(&mut self, time: Option<u64>) -> MeFrame {
        let time = time.or(self.last_time.map(|t| t + self.frame_ns)).unwrap_or(0);
        self.last_time = Some(time);
        let mut events = FrameEvents::default();
        let frame_ns = self.frame_ns;

        // Tied keyers' levels during a transition.
        let mut tied: Vec<(usize, f64)> = Vec::new();
        let mut layers = match self.active.clone() {
            None => transition::layers(
                TransitionKind::Cut,
                Direction::LeftToRight,
                &WipeParams::default(),
                0,
                self.pgm,
                None,
                0.0,
            ),
            Some(mut a) => {
                let progress = match &mut a.drive {
                    Drive::Manual(p) => *p,
                    Drive::Auto(ramp) => {
                        ramp.start.get_or_insert(time);
                        if ramp.done(time, frame_ns) {
                            1.0
                        } else {
                            ramp.level(time, frame_ns)
                        }
                    }
                };
                a.progress = progress;
                let layers = if a.background {
                    transition::layers(
                        a.spec.kind,
                        a.spec.direction,
                        &a.spec.wipe,
                        a.spec.dip_argb,
                        a.from,
                        a.to,
                        progress,
                    )
                } else {
                    transition::layers(
                        TransitionKind::Cut,
                        Direction::LeftToRight,
                        &WipeParams::default(),
                        0,
                        self.pgm,
                        None,
                        0.0,
                    )
                };
                for &(k, was_on) in &a.keys {
                    tied.push((k, if was_on { 1.0 - progress } else { progress }));
                }
                if progress >= 1.0 {
                    self.finish(&a);
                    self.tbar_position = self.tbar_rest();
                    events.changed = true;
                } else {
                    self.active = Some(a);
                }
                layers
            }
        };

        for (i, k) in self.keyers.iter_mut().enumerate() {
            let moving = k.moving();
            let mut level = k.frame(time, frame_ns);
            events.changed |= moving && !k.moving();
            if let Some(&(_, l)) = tied.iter().find(|(t, _)| *t == i) {
                if !events.changed || self.active.is_some() {
                    level = l;
                }
            }
            if level > 0.0 {
                layers.push(SceneLayer::Keyer { keyer: i, alpha: level });
            }
        }
        let clean = layers.clone();
        for (i, k) in self.dsks.iter_mut().enumerate() {
            let moving = k.moving();
            let level = k.frame(time, frame_ns);
            events.changed |= moving && !k.moving();
            if level > 0.0 {
                layers.push(SceneLayer::Dsk { dsk: i, alpha: level });
            }
        }
        let moving = self.ftb.moving();
        let ftb = self.ftb.frame(time, frame_ns);
        events.changed |= moving && !self.ftb.moving();
        if ftb > 0.0 {
            layers.push(SceneLayer::Fill {
                argb: 0xFF00_0000,
                alpha: ftb,
            });
        }
        let preview = transition::layers(
            TransitionKind::Cut,
            Direction::LeftToRight,
            &WipeParams::default(),
            0,
            self.pvw,
            None,
            0.0,
        );
        MeFrame {
            program: layers,
            clean,
            preview,
            events,
        }
    }

    /// How much of `input`'s sound the programme carries at `time`, for
    /// audio-follow-video: 1 on air, 0 off, in between during a transition.
    /// Doesn't change the state.
    pub fn program_weight(&self, input: usize, time: u64) -> f64 {
        let Some(a) = &self.active else {
            return if self.pgm == Some(input) { 1.0 } else { 0.0 };
        };
        if !a.background {
            return if self.pgm == Some(input) { 1.0 } else { 0.0 };
        }
        let progress = match &a.drive {
            Drive::Manual(p) => *p,
            Drive::Auto(r) => r.level_smooth(time, self.frame_ns),
        };
        let (wa, wb) = transition::audio_weights(a.spec.kind, progress);
        let mut w = 0.0;
        if a.from == Some(input) {
            w += wa;
        }
        if a.to == Some(input) {
            w += wb;
        }
        w.min(1.0)
    }

    /// How far the programme is faded to black at `time` (0–1).
    pub fn ftb_at(&self, time: u64) -> f64 {
        match &self.ftb.ramp {
            Some(r) => r.level_smooth(time, self.frame_ns),
            None => self.ftb.level,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAME: u64 = 20_000_000; // 50 fps

    fn me() -> Me {
        let mut m = Me::new(4, 2, 1, 50, 1);
        m.spec.duration_ms = 100; // 5 frames
        m
    }

    fn inputs(layers: &[SceneLayer]) -> Vec<(usize, f64)> {
        layers
            .iter()
            .filter_map(|l| match l {
                SceneLayer::Input { input, alpha, .. } => Some((*input, *alpha)),
                _ => None,
            })
            .collect()
    }

    fn keyers(layers: &[SceneLayer]) -> Vec<(usize, f64)> {
        layers
            .iter()
            .filter_map(|l| match l {
                SceneLayer::Keyer { keyer, alpha } => Some((*keyer, *alpha)),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn frames_round_up_to_whole_frames() {
        assert_eq!(frames_for(1000, FRAME), 50);
        assert_eq!(frames_for(1010, FRAME), 51);
        assert_eq!(frames_for(0, FRAME), 1);
        assert_eq!(frames_for(1001, 1_000_000_000 * 1001 / 30000), 30);
    }

    #[test]
    fn starts_on_the_first_two_sources() {
        let m = me();
        assert_eq!((m.program(), m.preview()), (Some(0), Some(1)));
        let one = Me::new(1, 0, 0, 50, 1);
        assert_eq!((one.program(), one.preview()), (Some(0), Some(0)));
    }

    #[test]
    fn auto_runs_frame_by_frame_and_swaps_the_buses() {
        let mut m = me();
        m.set_preview(Some(2)).unwrap();
        m.auto().unwrap();
        assert_eq!(m.transition_progress(), Some(0.0));
        let start = 1_000 * FRAME;
        let mut seen = Vec::new();
        for k in 0..5u64 {
            let f = m.frame(Some(start + k * FRAME));
            seen.push(inputs(&f.program));
            assert_eq!(f.events.changed, k == 4, "frame {k}");
        }
        assert_eq!(seen[0], vec![(0, 1.0), (2, 0.2)]);
        assert!((seen[3][1].1 - 0.8).abs() < 1e-9);
        assert_eq!(seen[4], vec![(2, 1.0)]);
        assert_eq!((m.program(), m.preview()), (Some(2), Some(0)));
        let f = m.frame(Some(start + 5 * FRAME));
        assert_eq!(inputs(&f.program), vec![(2, 1.0)]);
        assert_eq!(inputs(&f.preview), vec![(0, 1.0)]);
    }

    #[test]
    fn jittery_times_still_count_whole_frames() {
        let mut m = me();
        m.auto().unwrap();
        let start = 77 * FRAME + 3;
        let times = [start, start + FRAME - 40, start + 2 * FRAME + 90, start + 3 * FRAME];
        let p: Vec<f64> = times
            .iter()
            .map(|t| {
                m.frame(Some(*t));
                m.transition_progress().unwrap()
            })
            .collect();
        for (got, want) in p.iter().zip([0.2, 0.4, 0.6, 0.8]) {
            assert!((got - want).abs() < 1e-9, "{p:?}");
        }
    }

    #[test]
    fn preview_is_locked_during_a_transition() {
        let mut m = me();
        m.auto().unwrap();
        assert_eq!(m.set_preview(Some(3)), Err(VisionError::InTransition));
        assert_eq!(m.auto(), Err(VisionError::AutoRunning));
        assert_eq!(m.set_preview(Some(9)), Err(VisionError::NoInput(9)));
        m.cut().unwrap();
        assert_eq!((m.program(), m.preview()), (Some(1), Some(0)));
        m.set_preview(Some(3)).unwrap();
        m.set_preview(None).unwrap();
        m.cut().unwrap();
        assert_eq!(m.program(), None);
        assert!(inputs(&m.frame(Some(0)).program).is_empty());
    }

    #[test]
    fn cut_swaps_and_program_bus_takes_directly() {
        let mut m = me();
        m.cut().unwrap();
        assert_eq!((m.program(), m.preview()), (Some(1), Some(0)));
        m.set_program(Some(3)).unwrap();
        assert_eq!((m.program(), m.preview()), (Some(3), Some(0)));
        m.spec.kind = TransitionKind::Cut;
        m.auto().unwrap();
        assert_eq!((m.program(), m.preview()), (Some(0), Some(3)));
    }

    #[test]
    fn tbar_drives_by_hand_and_flips_at_the_end() {
        let mut m = me();
        m.set_tbar(0.3).unwrap();
        let f = m.frame(Some(0));
        assert_eq!(inputs(&f.program), vec![(0, 1.0), (1, 0.3)]);
        let f = m.frame(Some(50 * FRAME));
        assert_eq!(inputs(&f.program), vec![(0, 1.0), (1, 0.3)]);
        m.set_tbar(1.0).unwrap();
        assert_eq!((m.program(), m.preview()), (Some(1), Some(0)));
        assert_eq!(m.transition_progress(), None);
        m.set_tbar(0.75).unwrap();
        let f = m.frame(Some(0));
        assert_eq!(inputs(&f.program), vec![(1, 1.0), (0, 0.25)]);
        m.set_tbar(1.0).unwrap();
        assert_eq!((m.program(), m.preview()), (Some(1), Some(0)));
        m.set_tbar(0.0).unwrap();
        assert_eq!((m.program(), m.preview()), (Some(0), Some(1)));
        assert_eq!(m.tbar_position(), 0.0);
    }

    #[test]
    fn auto_carries_on_from_the_tbar() {
        let mut m = me();
        m.spec.duration_ms = 200; // 10 frames
        m.set_tbar(0.5).unwrap();
        m.auto().unwrap();
        assert_eq!(m.set_tbar(0.2), Err(VisionError::AutoRunning));
        let mut frames = 0;
        while m.transition_progress().is_some() {
            m.frame(Some(frames * FRAME));
            frames += 1;
        }
        assert_eq!(frames, 5);
        assert_eq!((m.program(), m.preview()), (Some(1), Some(0)));
        assert_eq!(m.tbar_position(), 0.0);
    }

    #[test]
    fn tied_keyers_go_with_the_transition() {
        let mut m = me();
        m.ties = vec![true, false];
        m.frame(Some(0));
        m.auto().unwrap();
        let mut levels = Vec::new();
        for k in 1..=5 {
            let f = m.frame(Some(k * FRAME));
            levels.push(keyers(&f.program));
        }
        // Keyer 1 comes on with the mix, keyer 2 isn't tied.
        assert_eq!(levels[0], vec![(0, 0.2)]);
        assert_eq!(levels[4], vec![(0, 1.0)]);
        assert!(m.keyer_on(0) && !m.keyer_on(1));
        assert_eq!(m.program(), Some(1));
        // Keys only: the background stays, the keyer goes off with a cut.
        m.next_background = false;
        m.cut().unwrap();
        assert!(!m.keyer_on(0));
        assert_eq!(m.program(), Some(1));
        m.ties = vec![false, false];
        assert_eq!(m.auto(), Err(VisionError::NothingToTransition));
    }

    #[test]
    fn clean_feed_has_no_dsk_and_no_ftb() {
        let mut m = me();
        m.set_keyer(0, Some(true), true).unwrap();
        m.set_dsk(0, Some(true), true).unwrap();
        m.set_ftb(Some(true), true);
        let f = m.frame(Some(0));
        assert!(f.program.iter().any(|l| matches!(l, SceneLayer::Dsk { .. })));
        assert!(matches!(f.program.last(), Some(SceneLayer::Fill { .. })));
        assert_eq!(keyers(&f.clean), vec![(0, 1.0)]);
        assert!(!f.clean.iter().any(|l| matches!(l, SceneLayer::Dsk { .. } | SceneLayer::Fill { .. })));
    }

    #[test]
    fn audio_follows_video_through_a_transition() {
        let mut m = me();
        m.spec.duration_ms = 1000;
        assert_eq!(m.program_weight(0, 0), 1.0);
        assert_eq!(m.program_weight(1, 0), 0.0);
        m.auto().unwrap();
        let start = 10 * FRAME;
        m.frame(Some(start));
        let half = start + 24 * FRAME;
        let (a, b) = (m.program_weight(0, half), m.program_weight(1, half));
        assert!((a - b).abs() < 1e-9 && (a * a + b * b - 1.0).abs() < 1e-9);
        assert!((m.program_weight(1, start + 60 * FRAME) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn ftb_and_keyers_fade_over_whole_frames() {
        let mut m = me();
        m.ftb_ms = 60; // 3 frames
        m.keyer_ms = vec![40, 40]; // 2 frames
        m.set_keyer(0, Some(true), false).unwrap();
        m.set_ftb(None, false);
        let mut levels = Vec::new();
        for k in 0..4 {
            let f = m.frame(Some(k * FRAME));
            let keyer = keyers(&f.program).first().map(|k| k.1);
            let ftb = f.program.iter().rev().find_map(|l| match l {
                SceneLayer::Fill { alpha, .. } => Some(*alpha),
                _ => None,
            });
            levels.push((keyer, ftb));
        }
        assert_eq!(levels[0].0, Some(0.5));
        assert_eq!(levels[1].0, Some(1.0));
        assert!((levels[0].1.unwrap() - 1.0 / 3.0).abs() < 1e-9);
        assert_eq!(levels[2].1, Some(1.0));
        m.set_ftb(Some(false), true);
        assert_eq!(m.ftb_level(), 0.0);
        assert_eq!(m.set_keyer(4, None, true), Err(VisionError::NoKeyer(4)));
    }
}

//! The operator's state: the sources, each ME's state machine and keyer
//! settings, tally. Shared by the API (commands) and the engine (one
//! [`Control::frame`] per output frame), behind one mutex held only briefly.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::config::Config;
use crate::engine::scene::{self, KeyerConfig};
use crate::compositor::config::parse_color;
use crate::vision::me::{Me, MeFrame, TransitionSpec, VisionError};
use crate::vision::transition::{Direction, TransitionKind, WipeParams};

/// What a source is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "index")]
pub enum SourceKind {
    /// Video input (0-based)
    Input(usize),
    /// Key input (fill + key), 0-based
    Key(usize),
    /// The programme of ME `index` (0-based; ME 2 is 1)
    Me(usize),
}

#[derive(Debug, Clone, Serialize)]
pub struct SourceDef {
    pub id: String,
    pub label: String,
    #[serde(flatten)]
    pub kind: SourceKind,
}

/// An error for the API: status and message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandError {
    pub status: u16,
    pub message: String,
}

impl CommandError {
    pub fn bad(message: impl Into<String>) -> Self {
        Self {
            status: 400,
            message: message.into(),
        }
    }
    pub fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: 404,
            message: message.into(),
        }
    }
    pub fn conflict(message: impl Into<String>) -> Self {
        Self {
            status: 409,
            message: message.into(),
        }
    }
}

impl From<VisionError> for CommandError {
    fn from(e: VisionError) -> Self {
        match e {
            VisionError::NoInput(_) | VisionError::NoKeyer(_) => CommandError::not_found(e.to_string()),
            _ => CommandError::conflict(e.to_string()),
        }
    }
}

pub type CmdResult = Result<(), CommandError>;

/// Transition settings as the API sees them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TransitionSettings {
    #[serde(rename = "type")]
    pub kind: TransitionKind,
    pub duration_ms: u32,
    pub direction: Direction,
    pub wipe: WipeParams,
    /// `#RRGGBB`
    pub dip_color: String,
}

impl Default for TransitionSettings {
    fn default() -> Self {
        Self {
            kind: TransitionKind::Mix,
            duration_ms: 1000,
            direction: Direction::LeftToRight,
            wipe: WipeParams::default(),
            dip_color: "#000000".into(),
        }
    }
}

impl TransitionSettings {
    fn spec(&self) -> Result<TransitionSpec, CommandError> {
        let dip = parse_color(&self.dip_color)
            .ok_or_else(|| CommandError::bad(format!("dip_color {:?}: expected #RRGGBB", self.dip_color)))?;
        if self.duration_ms > 60_000 {
            return Err(CommandError::bad("duration_ms: at most 60000"));
        }
        Ok(TransitionSpec {
            kind: self.kind,
            duration_ms: self.duration_ms,
            direction: self.direction,
            wipe: WipeParams {
                soft: self.wipe.soft.clamp(0.0, 1.0),
                ..self.wipe
            },
            dip_argb: dip | 0xFF00_0000,
        })
    }
}

/// Settings of one ME (saved, exported).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct MeSettings {
    pub program: Option<String>,
    pub preview: Option<String>,
    pub transition: TransitionSettings,
    pub background: bool,
    pub ties: Vec<bool>,
    pub ftb_ms: u32,
    pub keyers: Vec<KeyerConfig>,
    pub dsks: Vec<KeyerConfig>,
}

/// Everything saved in `state.json` and in the config export.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct Saved {
    pub labels: BTreeMap<String, String>,
    pub mes: Vec<MeSettings>,
    pub tally_targets: Vec<String>,
}

/// Tally of a source.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct Tally {
    /// On air on the programme of ME 1 (red)
    pub program: bool,
    /// On the preview of ME 1 (green)
    pub preview: bool,
}

/// One ME's frame: its scenes and the keyers they need.
pub struct MeOut {
    pub frame: MeFrame,
    pub keyers: Vec<(KeyerConfig, Option<usize>)>,
    pub dsks: Vec<(KeyerConfig, Option<usize>)>,
}

pub struct Control {
    pub sources: Vec<SourceDef>,
    pub mes: Vec<Me>,
    pub settings: Vec<MeSettings>,
    pub tally: Vec<Tally>,
    pub tally_targets: Vec<String>,
    /// Bumped on every change (the UI's WebSocket pushes then)
    pub revision: u64,
}

impl Control {
    pub fn new(config: &Config) -> Control {
        let mut sources = Vec::new();
        for i in 0..config.inputs {
            sources.push(SourceDef {
                id: format!("in-{}", i + 1),
                label: format!("In {}", i + 1),
                kind: SourceKind::Input(i),
            });
        }
        for i in 0..config.key_inputs {
            sources.push(SourceDef {
                id: format!("key-{}", i + 1),
                label: format!("Key {}", i + 1),
                kind: SourceKind::Key(i),
            });
        }
        for m in 1..config.mes {
            sources.push(SourceDef {
                id: format!("me-{}", m + 1),
                label: format!("ME {}", m + 1),
                kind: SourceKind::Me(m),
            });
        }
        let n = sources.len();
        let f = &config.format;
        let mes: Vec<Me> = (0..config.mes)
            .map(|_| Me::new(n, config.keyers_per_me, config.dsks, f.rate_n, f.rate_d))
            .collect();
        let settings = (0..config.mes)
            .map(|_| MeSettings {
                background: true,
                ties: vec![false; config.keyers_per_me],
                ftb_ms: 1000,
                keyers: (0..config.keyers_per_me)
                    .map(|k| KeyerConfig {
                        label: format!("Key {}", k + 1),
                        ..Default::default()
                    })
                    .collect(),
                dsks: (0..config.dsks)
                    .map(|k| KeyerConfig {
                        label: format!("DSK {}", k + 1),
                        mode: scene::KeyMode::Linear,
                        source: (k < config.key_inputs).then(|| format!("key-{}", k + 1)),
                        clip: 0.0,
                        ..Default::default()
                    })
                    .collect(),
                ..Default::default()
            })
            .collect();
        Control {
            tally: vec![Tally::default(); n],
            sources,
            mes,
            settings,
            tally_targets: config.tsl_targets.clone(),
            revision: 1,
        }
    }

    pub fn source_index(&self, id: &str) -> Option<usize> {
        self.sources.iter().position(|s| s.id == id)
    }

    fn me_index(&self, me: usize) -> Result<usize, CommandError> {
        if me >= 1 && me <= self.mes.len() {
            Ok(me - 1)
        } else {
            Err(CommandError::not_found(format!("no ME {me}")))
        }
    }

    /// A source an ME may take (`None`: black). ME `m` can't take its own
    /// programme or a lower ME's (re-entry goes from higher MEs down).
    fn bus_source(&self, m: usize, id: Option<&str>) -> Result<Option<usize>, CommandError> {
        let Some(id) = id.filter(|s| *s != "black") else {
            return Ok(None);
        };
        let i = self
            .source_index(id)
            .ok_or_else(|| CommandError::not_found(format!("no source {id:?}")))?;
        if let SourceKind::Me(j) = self.sources[i].kind {
            if j <= m {
                return Err(CommandError::bad(format!(
                    "ME {} can only take the programme of a higher ME",
                    m + 1
                )));
            }
        }
        Ok(Some(i))
    }

    fn changed(&mut self) {
        self.revision += 1;
    }

    fn sync_settings(&mut self, m: usize) {
        let me = &self.mes[m];
        let id = |i: Option<usize>| i.map(|i| self.sources[i].id.clone());
        let (p, v) = (id(me.program()), id(me.preview()));
        self.settings[m].program = p;
        self.settings[m].preview = v;
    }

    pub fn preview(&mut self, me: usize, source: Option<&str>) -> CmdResult {
        let m = self.me_index(me)?;
        let s = self.bus_source(m, source)?;
        self.mes[m].set_preview(s)?;
        self.sync_settings(m);
        self.changed();
        Ok(())
    }

    pub fn program(&mut self, me: usize, source: Option<&str>) -> CmdResult {
        let m = self.me_index(me)?;
        let s = self.bus_source(m, source)?;
        self.mes[m].set_program(s)?;
        self.sync_settings(m);
        self.changed();
        Ok(())
    }

    pub fn cut(&mut self, me: usize) -> CmdResult {
        let m = self.me_index(me)?;
        self.mes[m].cut()?;
        self.sync_settings(m);
        self.changed();
        Ok(())
    }

    pub fn auto(&mut self, me: usize) -> CmdResult {
        let m = self.me_index(me)?;
        self.mes[m].auto()?;
        self.sync_settings(m);
        self.changed();
        Ok(())
    }

    pub fn tbar(&mut self, me: usize, position: f64) -> CmdResult {
        let m = self.me_index(me)?;
        if !position.is_finite() {
            return Err(CommandError::bad("position must be 0–1"));
        }
        self.mes[m].set_tbar(position)?;
        self.sync_settings(m);
        self.changed();
        Ok(())
    }

    pub fn ftb(&mut self, me: usize, on: Option<bool>, cut: bool) -> CmdResult {
        let m = self.me_index(me)?;
        self.mes[m].set_ftb(on, cut);
        self.changed();
        Ok(())
    }

    pub fn set_ftb_rate(&mut self, me: usize, ms: u32) -> CmdResult {
        let m = self.me_index(me)?;
        if ms == 0 || ms > 60_000 {
            return Err(CommandError::bad("rate_ms: 1–60000"));
        }
        self.mes[m].ftb_ms = ms;
        self.settings[m].ftb_ms = ms;
        self.changed();
        Ok(())
    }

    pub fn set_transition(&mut self, me: usize, t: TransitionSettings) -> CmdResult {
        let m = self.me_index(me)?;
        self.mes[m].spec = t.spec()?;
        self.settings[m].transition = t;
        self.changed();
        Ok(())
    }

    pub fn set_ties(&mut self, me: usize, background: Option<bool>, ties: Option<Vec<bool>>) -> CmdResult {
        let m = self.me_index(me)?;
        let n = self.mes[m].keyers();
        let ties = ties.unwrap_or_else(|| self.mes[m].ties.clone());
        if ties.len() != n {
            return Err(CommandError::bad(format!("ties: {n} values expected")));
        }
        let background = background.unwrap_or(self.mes[m].next_background);
        if !background && !ties.iter().any(|&t| t) {
            return Err(CommandError::bad("the next transition needs the background or a keyer"));
        }
        self.mes[m].ties = ties.clone();
        self.mes[m].next_background = background;
        self.settings[m].ties = ties;
        self.settings[m].background = background;
        self.changed();
        Ok(())
    }

    fn check_keyer(&self, m: usize, cfg: &KeyerConfig, upstream: bool) -> CmdResult {
        if let Some(s) = &cfg.source {
            self.bus_source(m, Some(s))?;
        }
        if !upstream && cfg.mode == scene::KeyMode::Pip {
            return Err(CommandError::bad("a DSK can't be a PiP"));
        }
        if parse_color(&cfg.dme.border.color).is_none() {
            return Err(CommandError::bad("dme.border.color: expected #RRGGBB"));
        }
        let c = &cfg.dme.crop;
        if c.left + c.right >= 1.0 || c.top + c.bottom >= 1.0 || [c.left, c.right, c.top, c.bottom].iter().any(|v| *v < 0.0) {
            return Err(CommandError::bad("dme.crop: fractions, left + right and top + bottom below 1"));
        }
        Ok(())
    }

    /// Replace a keyer's settings (`dsk`: a downstream keyer).
    pub fn set_keyer(&mut self, me: usize, keyer: usize, dsk: bool, cfg: KeyerConfig) -> CmdResult {
        let m = self.me_index(me)?;
        self.check_keyer(m, &cfg, !dsk)?;
        let list = if dsk { &mut self.settings[m].dsks } else { &mut self.settings[m].keyers };
        let slot = list
            .get_mut(keyer)
            .ok_or_else(|| CommandError::not_found(format!("no keyer {}", keyer + 1)))?;
        *slot = cfg.clone();
        if dsk {
            self.mes[m].dsk_ms[keyer] = cfg.rate_ms;
        } else {
            self.mes[m].keyer_ms[keyer] = cfg.rate_ms;
        }
        self.changed();
        Ok(())
    }

    /// On, off or toggle a keyer, cut or faded (AUTO).
    pub fn key_onair(&mut self, me: usize, keyer: usize, dsk: bool, on: Option<bool>, cut: bool) -> CmdResult {
        let m = self.me_index(me)?;
        if dsk {
            self.mes[m].set_dsk(keyer, on, cut)?;
        } else {
            self.mes[m].set_keyer(keyer, on, cut)?;
        }
        self.changed();
        Ok(())
    }

    pub fn set_label(&mut self, source: &str, label: &str) -> CmdResult {
        let i = self
            .source_index(source)
            .ok_or_else(|| CommandError::not_found(format!("no source {source:?}")))?;
        let label = label.trim();
        if label.is_empty() || label.len() > 64 {
            return Err(CommandError::bad("label: 1–64 characters"));
        }
        self.sources[i].label = label.to_string();
        self.changed();
        Ok(())
    }

    /// Advance every ME to the frame at `time` (TAI ns). MEs whose
    /// programme another ME takes come first in the result's order (the
    /// last ME first); tally is updated.
    pub fn frame(&mut self, time: u64) -> Vec<MeOut> {
        let mut outs = Vec::with_capacity(self.mes.len());
        let mut changed = false;
        for m in 0..self.mes.len() {
            let frame = self.mes[m].frame(Some(time));
            changed |= frame.events.changed;
            let resolve = |cfgs: &[KeyerConfig]| {
                cfgs.iter()
                    .map(|c| (c.clone(), c.source.as_deref().and_then(|s| self.source_index(s))))
                    .collect::<Vec<_>>()
            };
            outs.push(MeOut {
                keyers: resolve(&self.settings[m].keyers),
                dsks: resolve(&self.settings[m].dsks),
                frame,
            });
        }
        if changed {
            for m in 0..self.mes.len() {
                self.sync_settings(m);
            }
        }
        // Tally of ME 1: what its programme shows, through re-entered MEs.
        let mut program = Vec::new();
        let mut todo = vec![0usize];
        let mut seen = vec![false; outs.len()];
        while let Some(m) = todo.pop() {
            if seen[m] {
                continue;
            }
            seen[m] = true;
            let o = &outs[m];
            let mut s = Vec::new();
            scene::sources_of(&o.frame.program, &o.keyers, &o.dsks, &mut s);
            for i in s {
                if let SourceKind::Me(j) = self.sources[i].kind {
                    todo.push(j);
                }
                if !program.contains(&i) {
                    program.push(i);
                }
            }
        }
        let preview = self.mes.first().and_then(|m| m.preview());
        let tally: Vec<Tally> = (0..self.sources.len())
            .map(|i| Tally {
                program: program.contains(&i),
                preview: preview == Some(i),
            })
            .collect();
        if tally != self.tally || changed {
            self.tally = tally;
            self.changed();
        }
        outs
    }

    /// Audio follows video: each video input's level in the programme of
    /// ME 1 at TAI `time` (equal-power through transitions, a dip through
    /// silence, FTB), through re-entered MEs.
    pub fn audio_weights(&self, time: u64, inputs: usize) -> Vec<f64> {
        let mut w = vec![0.0; inputs];
        self.add_weights(0, 1.0, time, &mut w, 0);
        w
    }

    fn add_weights(&self, m: usize, gain: f64, time: u64, w: &mut [f64], depth: usize) {
        let Some(me) = self.mes.get(m) else { return };
        if depth > self.mes.len() {
            return;
        }
        let gain = gain * (1.0 - me.ftb_at(time)).clamp(0.0, 1.0);
        if gain <= 0.0 {
            return;
        }
        for (i, s) in self.sources.iter().enumerate() {
            let pw = me.program_weight(i, time);
            if pw <= 0.0 {
                continue;
            }
            match s.kind {
                SourceKind::Input(k) => {
                    if let Some(v) = w.get_mut(k) {
                        *v += gain * pw;
                    }
                }
                SourceKind::Me(j) if j > m => self.add_weights(j, gain * pw, time, w, depth + 1),
                _ => {}
            }
        }
    }

    /// The saved state.
    pub fn saved(&self) -> Saved {
        Saved {
            labels: self.sources.iter().map(|s| (s.id.clone(), s.label.clone())).collect(),
            mes: self.settings.clone(),
            tally_targets: self.tally_targets.clone(),
        }
    }

    /// Apply a saved state (start-up, config import). Unknown sources and
    /// keyers beyond the configured counts are left out; what is invalid
    /// is reported.
    pub fn apply(&mut self, saved: &Saved) -> Vec<String> {
        let mut warnings = Vec::new();
        for (id, label) in &saved.labels {
            if let Err(e) = self.set_label(id, label) {
                warnings.push(format!("label {id}: {}", e.message));
            }
        }
        for (m, s) in saved.mes.iter().enumerate().take(self.mes.len()) {
            let me = m + 1;
            let mut warn = |what: &str, r: CmdResult| {
                if let Err(e) = r {
                    warnings.push(format!("ME {me} {what}: {}", e.message));
                }
            };
            warn("transition", self.set_transition(me, s.transition.clone()));
            for (k, cfg) in s.keyers.iter().enumerate().take(self.mes[m].keyers()) {
                warn("keyer", self.set_keyer(me, k, false, cfg.clone()));
            }
            for (k, cfg) in s.dsks.iter().enumerate().take(self.mes[m].dsks()) {
                warn("dsk", self.set_keyer(me, k, true, cfg.clone()));
            }
            let mut ties = s.ties.clone();
            ties.resize(self.mes[m].keyers(), false);
            warn("ties", self.set_ties(me, Some(s.background || !ties.iter().any(|&t| t)), Some(ties)));
            self.mes[m].ftb_ms = s.ftb_ms.max(1);
            self.settings[m].ftb_ms = self.mes[m].ftb_ms;
            warn("program", self.program(me, s.program.as_deref()));
            warn("preview", self.preview(me, s.preview.as_deref()));
        }
        self.tally_targets = saved.tally_targets.clone();
        self.changed();
        warnings
    }

    /// The whole state for the UI and `GET /api/v1/state`.
    pub fn snapshot(&self) -> Value {
        let name = |i: Option<usize>| i.map(|i| self.sources[i].id.clone()).unwrap_or_else(|| "black".into());
        let mes: Vec<Value> = self
            .mes
            .iter()
            .enumerate()
            .map(|(m, me)| {
                let s = &self.settings[m];
                json!({
                    "id": format!("me-{}", m + 1),
                    "program": name(me.program()),
                    "preview": name(me.preview()),
                    "transition": s.transition,
                    "transition_running": me.transition_progress().is_some(),
                    "transition_progress": me.transition_progress(),
                    "auto_running": me.auto_running(),
                    "tbar": me.tbar_position(),
                    "background": me.next_background,
                    "ties": me.ties,
                    "ftb": {"on": me.ftb_on(), "level": me.ftb_level(), "rate_ms": me.ftb_ms},
                    "keyers": s.keyers.iter().enumerate().map(|(k, c)| json!({
                        "id": format!("me-{}-key-{}", m + 1, k + 1),
                        "on": me.keyer_on(k), "level": me.keyer_level(k), "config": c,
                    })).collect::<Vec<_>>(),
                    "dsks": s.dsks.iter().enumerate().map(|(k, c)| json!({
                        "id": format!("me-{}-dsk-{}", m + 1, k + 1),
                        "on": me.dsk_on(k), "level": me.dsk_level(k), "config": c,
                    })).collect::<Vec<_>>(),
                })
            })
            .collect();
        json!({
            "revision": self.revision,
            "sources": self.sources.iter().zip(&self.tally).map(|(s, t)| json!({
                "id": s.id, "label": s.label, "kind": s.kind, "tally": t,
            })).collect::<Vec<_>>(),
            "mes": mes,
            "tally_targets": self.tally_targets,
        })
    }
}

/// Load `state.json` (missing or invalid: defaults, the invalid file kept
/// as `.bad`).
pub fn load_saved(path: &Path) -> Option<Saved> {
    let text = std::fs::read_to_string(path).ok()?;
    match serde_json::from_str(&text) {
        Ok(s) => Some(s),
        Err(e) => {
            tracing::error!("{} is not valid ({e}); starting with defaults", path.display());
            let _ = std::fs::rename(path, path.with_extension("json.bad"));
            None
        }
    }
}

/// Write `state.json` atomically.
pub fn store_saved(path: &Path, saved: &Saved) -> std::io::Result<()> {
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(saved)?)?;
    std::fs::rename(tmp, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> Config {
        // Defaults from an empty environment, with 2 MEs.
        let mut c = Config::from_env().unwrap();
        c.inputs = 4;
        c.key_inputs = 1;
        c.mes = 2;
        c.keyers_per_me = 2;
        c.dsks = 1;
        c
    }

    #[test]
    fn sources_and_buses() {
        let mut c = Control::new(&config());
        let ids: Vec<&str> = c.sources.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(ids, vec!["in-1", "in-2", "in-3", "in-4", "key-1", "me-2"]);
        c.preview(1, Some("in-3")).unwrap();
        c.cut(1).unwrap();
        assert_eq!(c.settings[0].program.as_deref(), Some("in-3"));
        // Re-entry only from higher MEs.
        c.preview(1, Some("me-2")).unwrap();
        assert_eq!(c.preview(2, Some("me-2")).unwrap_err().status, 400);
        assert_eq!(c.preview(1, Some("nope")).unwrap_err().status, 404);
        assert_eq!(c.cut(3).unwrap_err().status, 404);
        c.preview(1, Some("black")).unwrap();
        assert_eq!(c.mes[0].preview(), None);
    }

    #[test]
    fn tally_follows_re_entry() {
        let mut c = Control::new(&config());
        c.program(2, Some("in-4")).unwrap();
        c.program(1, Some("me-2")).unwrap();
        c.preview(1, Some("in-2")).unwrap();
        c.frame(0);
        let t: Vec<(bool, bool)> = c.tally.iter().map(|t| (t.program, t.preview)).collect();
        assert_eq!(t, vec![(false, false), (false, true), (false, false), (true, false), (false, false), (true, false)]);
    }

    #[test]
    fn audio_follows_video_through_re_entry_and_ftb() {
        let mut c = Control::new(&config());
        c.program(1, Some("in-2")).unwrap();
        assert_eq!(c.audio_weights(0, 4), vec![0.0, 1.0, 0.0, 0.0]);
        c.program(2, Some("in-3")).unwrap();
        c.program(1, Some("me-2")).unwrap();
        assert_eq!(c.audio_weights(0, 4), vec![0.0, 0.0, 1.0, 0.0]);
        // T-bar halfway in ME 1 from ME 2 (in-3) to in-1: equal power.
        c.preview(1, Some("in-1")).unwrap();
        c.tbar(1, 0.5).unwrap();
        let w = c.audio_weights(0, 4);
        assert!((w[0] - w[2]).abs() < 1e-9 && (w[0] * w[0] + w[2] * w[2] - 1.0).abs() < 1e-9);
        c.tbar(1, 0.0).unwrap();
        c.ftb(1, Some(true), true).unwrap();
        assert_eq!(c.audio_weights(0, 4), vec![0.0; 4]);
    }

    #[test]
    fn saved_state_round_trips() {
        let mut c = Control::new(&config());
        c.set_label("in-1", "Cam 1").unwrap();
        c.preview(1, Some("in-2")).unwrap();
        c.set_transition(
            1,
            TransitionSettings {
                kind: TransitionKind::Wipe,
                duration_ms: 500,
                ..Default::default()
            },
        )
        .unwrap();
        c.set_ties(1, Some(true), Some(vec![true, false])).unwrap();
        let saved = c.saved();
        let mut d = Control::new(&config());
        assert!(d.apply(&saved).is_empty());
        assert_eq!(d.saved(), saved);
        assert_eq!(d.sources[0].label, "Cam 1");
        assert!(c.set_ties(1, Some(false), Some(vec![false, false])).is_err());
        let bad = TransitionSettings {
            dip_color: "red".into(),
            ..Default::default()
        };
        assert_eq!(c.set_transition(1, bad).unwrap_err().status, 400);
    }
}

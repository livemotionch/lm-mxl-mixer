//! Programme audio: audio follows video (the audio of each source, weighted
//! by how much of it the programme shows, with equal-power
//! curves through transitions and a dip through silence), or one input
//! passed through untouched. A full audio mixer is mxl-audio-mixer's job.
//!
//! Per output frame `N` the samples of `[t(N), t(N+1))` (48 kHz) are written,
//! made from the inputs' samples one read offset earlier (the same latency
//! as the video). Weights are computed at both ends of the frame and ramped
//! over its samples.

use std::path::Path;

use crate::config::AudioMode;
use crate::engine::inputs::{Instances, Route};
use crate::mxlio::{self, flowdef, Api, AudioReader, Clock, Instance, RawWriter};

/// One audio input.
struct AudioIn {
    name: String,
    route: Option<Route>,
    reader: Option<AudioReader>,
    retry_at: u64,
    error: Option<String>,
    /// Samples read (per channel) for this frame
    buf: Vec<Vec<f32>>,
    ok: bool,
    /// Frames this input's samples were not there in time
    gaps: u64,
}

/// Counters for the API and metrics.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct AudioStats {
    pub mode: Option<AudioMode>,
    pub channels: usize,
    pub frames: u64,
    /// Input frames whose samples were not there (silence instead)
    pub input_gaps: u64,
    /// `input_gaps` per input, in the order of `inputs`
    pub input_gaps_by_input: Vec<u64>,
    pub errors: u64,
    pub inputs: Vec<(String, bool, Option<String>)>,
}

pub struct AudioPath {
    mode: AudioMode,
    inputs: Vec<AudioIn>,
    writer: RawWriter,
    out: Vec<Vec<f32>>,
    pub stats: AudioStats,
}

const RATE: (u32, u32) = (48_000, 1);

/// Receiver keys and labels of the audio inputs.
pub fn receivers(mode: AudioMode, inputs: usize, prefix: &str) -> Vec<(String, String)> {
    match mode {
        AudioMode::Off => Vec::new(),
        AudioMode::Afv => (0..inputs)
            .map(|i| (format!("in-{}-audio", i + 1), format!("{prefix} In {} Audio", i + 1)))
            .collect(),
        AudioMode::Passthrough => vec![("audio-in".into(), format!("{prefix} Audio In"))],
    }
}

/// The audio output's flow id.
pub fn flow_id(seed: &str) -> uuid::Uuid {
    mxl_nmos::ids::derive_id(seed, "pgm-audio/mxl-flow")
}

impl AudioPath {
    pub fn new(
        api: &Api,
        domain: &Path,
        mode: AudioMode,
        channels: usize,
        n_inputs: usize,
        seed: &str,
        label: &str,
        group: &str,
    ) -> Result<Option<AudioPath>, String> {
        if mode == AudioMode::Off {
            return Ok(None);
        }
        let inst = Instance::new(api, domain)?;
        let id = flow_id(seed);
        let def = flowdef::audio(id, label, "Programme audio of ME 1", group, channels);
        let (writer, _) = inst.writer(&def).map_err(|e| format!("audio output: {e}"))?;
        let names = receivers(mode, n_inputs, "");
        Ok(Some(AudioPath {
            mode,
            inputs: names
                .into_iter()
                .map(|(k, _)| AudioIn {
                    name: k,
                    route: None,
                    reader: None,
                    retry_at: 0,
                    error: None,
                    buf: Vec::new(),
                    ok: false,
                    gaps: 0,
                })
                .collect(),
            writer,
            out: vec![Vec::new(); channels],
            stats: AudioStats {
                mode: Some(mode),
                channels,
                ..Default::default()
            },
        }))
    }

    pub fn set_routes(&mut self, routes: &[Option<Route>]) {
        for (a, r) in self.inputs.iter_mut().zip(routes) {
            if a.route != *r {
                a.route = r.clone();
                a.reader = None;
                a.retry_at = 0;
            }
        }
    }

    fn open(&mut self, api: &Api, instances: &mut Instances, now: u64) {
        for a in &mut self.inputs {
            let Some(route) = &a.route else { continue };
            if a.reader.is_some() || now < a.retry_at {
                continue;
            }
            a.retry_at = now + 1_000_000_000;
            let r = instances
                .get(api, &route.domain)
                .and_then(|i| i.audio_reader(route.flow).map_err(|e| e.to_string()));
            match r {
                Ok(r) if r.rate.numerator == 48_000 => {
                    tracing::info!("{}: reading audio {} ({} channels)", a.name, route.flow, r.channels);
                    a.reader = Some(r);
                    a.error = None;
                }
                Ok(r) => a.error = Some(format!("{} Hz audio is not supported (48 kHz)", r.rate.numerator)),
                Err(e) => a.error = Some(e),
            }
        }
    }

    /// Write the samples of output frame `[t0, t1)`, read `offset_ns`
    /// earlier from the inputs. `weights(t)`: each input's level at TAI `t`
    /// (afv; ignored for passthrough). The window ends about now, so inputs
    /// get until `wait_ns` from now (all inputs together) to commit their
    /// samples, like video inputs; after that the input is silent this frame.
    #[allow(clippy::too_many_arguments)]
    pub fn frame(
        &mut self,
        api: &Api,
        instances: &mut Instances,
        clock: &Clock,
        t0: u64,
        t1: u64,
        offset_ns: u64,
        w0: &[f64],
        w1: &[f64],
        wait_ns: u64,
    ) {
        let rate = mxlio::rational(RATE.0, RATE.1);
        let now = clock.now();
        self.open(api, instances, now);
        let (s0, s1) = (clock.time_to_index(&rate, t0), clock.time_to_index(&rate, t1));
        let count = (s1 - s0) as usize;
        if count == 0 || count > 48_000 {
            return;
        }
        let end = clock.time_to_index(&rate, t1.saturating_sub(offset_ns));
        let deadline = now + wait_ns;
        for (i, a) in self.inputs.iter_mut().enumerate() {
            a.ok = false;
            let wanted = match self.mode {
                AudioMode::Afv => w0.get(i).copied().unwrap_or(0.0) > 0.0 || w1.get(i).copied().unwrap_or(0.0) > 0.0,
                _ => true,
            };
            let Some(r) = &a.reader else { continue };
            if !wanted {
                continue;
            }
            if a.buf.len() < r.channels {
                a.buf.resize(r.channels, Vec::new());
            }
            let timeout = deadline.saturating_sub(clock.now());
            match r.read(end, count, &mut a.buf, timeout) {
                Ok(true) => a.ok = a.buf.first().is_some_and(|b| b.len() == count),
                Ok(false) => {
                    self.stats.input_gaps += 1;
                    a.gaps += 1;
                }
                Err(e) => {
                    a.error = Some(e.to_string());
                    a.reader = None;
                }
            }
        }
        for o in &mut self.out {
            o.clear();
            o.resize(count, 0.0);
        }
        match self.mode {
            AudioMode::Passthrough => {
                if let Some(a) = self.inputs.first().filter(|a| a.ok) {
                    for (o, b) in self.out.iter_mut().zip(&a.buf) {
                        o.copy_from_slice(&b[..count]);
                    }
                }
            }
            AudioMode::Afv => {
                for (i, a) in self.inputs.iter().enumerate() {
                    if !a.ok {
                        continue;
                    }
                    let (g0, g1) = (w0.get(i).copied().unwrap_or(0.0) as f32, w1.get(i).copied().unwrap_or(0.0) as f32);
                    let step = (g1 - g0) / count as f32;
                    for (o, b) in self.out.iter_mut().zip(&a.buf) {
                        for (k, (o, &s)) in o.iter_mut().zip(b.iter()).enumerate() {
                            *o += s * (g0 + step * k as f32);
                        }
                    }
                }
            }
            AudioMode::Off => {}
        }
        let out = &self.out;
        let r = self.writer.write_samples(s1, count, |slices| {
            for (c, (a, b)) in slices.iter_mut().enumerate() {
                let src = out.get(c);
                let n = a.len();
                for (k, v) in a.iter_mut().enumerate() {
                    *v = src.map_or(0.0, |s| s[k]);
                }
                for (k, v) in b.iter_mut().enumerate() {
                    *v = src.map_or(0.0, |s| s[n + k]);
                }
            }
        });
        match r {
            Ok(()) => self.stats.frames += 1,
            Err(e) => {
                self.stats.errors += 1;
                if self.stats.errors % 500 == 1 {
                    tracing::warn!("audio output: {e}");
                }
            }
        }
        if self.stats.frames % 25 == 1 {
            self.stats.inputs = self.inputs.iter().map(|a| (a.name.clone(), a.reader.is_some(), a.error.clone())).collect();
            self.stats.input_gaps_by_input = self.inputs.iter().map(|a| a.gaps).collect();
        }
    }
}

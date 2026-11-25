//! The frame engine: one thread that writes output grain `N` at its TAI
//! time, every frame.
//!
//! Per frame: the MEs' state machines give the scenes ([`Control::frame`]);
//! only the inputs some output shows are read ([`inputs`]); each ME is drawn
//! by the compositor's 10-bit renderer straight into its output grains (the
//! highest ME first, so that lower MEs can take its programme the same
//! frame). A frame that can't be drawn in time is skipped, never waited for.

pub mod audio;
pub mod inputs;
pub mod outputs;
pub mod scene;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde::Serialize;

use crate::config::Config;
use crate::control::{Control, SourceKind};
use crate::compositor::kernels10::BLACK10;
use crate::compositor::mixer10::Mixer10;
use crate::compositor::pool::StripPool;
use crate::compositor::render10::Source10;
use crate::compositor::v210;
use crate::mxlio::{self, Api, Clock};
use inputs::{InputCounters, InputFrame, InputState, Instances, Route, VideoInput};
use outputs::{OutputKind, Outputs};

/// Time per drawn frame, as a histogram (Prometheus buckets, ms).
pub const BUCKETS_MS: [f64; 10] = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 16.0, 20.0, 40.0];

#[derive(Debug, Clone, Default, Serialize)]
pub struct Histogram {
    pub counts: [u64; BUCKETS_MS.len()],
    pub count: u64,
    pub sum_ms: f64,
    pub last_ms: f64,
    pub max_ms: f64,
}

impl Histogram {
    pub fn add(&mut self, ms: f64) {
        for (i, b) in BUCKETS_MS.iter().enumerate() {
            if ms <= *b {
                self.counts[i] += 1;
            }
        }
        self.count += 1;
        self.sum_ms += ms;
        self.last_ms = ms;
        self.max_ms = self.max_ms.max(ms);
    }
}

/// An input's state for the API and metrics.
#[derive(Debug, Clone, Serialize)]
pub struct InputStatus {
    pub id: String,
    pub state: InputState,
    pub route: Option<Route>,
    pub flow: Option<crate::mxlio::flowdef::InputFlow>,
    pub error: Option<String>,
    pub counters: InputCounters,
}

/// An output's state.
#[derive(Debug, Clone, Default, Serialize)]
pub struct OutputStatus {
    pub key: String,
    pub label: String,
    pub flow_id: String,
    pub frames: u64,
    pub errors: u64,
    pub render: Histogram,
    pub parallel: bool,
}

/// What the engine did, for `/metrics` and the UI.
#[derive(Debug, Clone, Default, Serialize)]
pub struct EngineStats {
    pub running: bool,
    pub format: String,
    pub frames: u64,
    /// Output grains not written: the engine was more than a frame late
    pub missed: u64,
    /// Frames finished after the next frame's time
    pub late: u64,
    /// The whole frame (all outputs)
    pub frame: Histogram,
    pub threads: usize,
    pub simd: &'static str,
    pub inputs: Vec<InputStatus>,
    pub outputs: Vec<OutputStatus>,
    pub index: u64,
}

/// What the engine and the rest share.
pub struct Shared {
    pub control: Mutex<Control>,
    /// Routes of the input and key receivers (by source index)
    pub routes: Mutex<Vec<Option<Route>>>,
    pub stats: Mutex<EngineStats>,
    pub stop: AtomicBool,
    /// Frames for the preview encoder
    pub preview: Mutex<Option<crate::preview::Tap>>,
    /// Routes of the audio receivers
    pub audio_routes: Mutex<Vec<Option<Route>>>,
    pub audio: Mutex<audio::AudioStats>,
}

impl Shared {
    pub fn new(control: Control, inputs: usize) -> Arc<Shared> {
        Arc::new(Shared {
            control: Mutex::new(control),
            routes: Mutex::new(vec![None; inputs]),
            stats: Mutex::new(EngineStats::default()),
            stop: AtomicBool::new(false),
            preview: Mutex::new(None),
            audio_routes: Mutex::new(vec![None; inputs]),
            audio: Mutex::new(audio::AudioStats::default()),
        })
    }

    pub fn control(&self) -> std::sync::MutexGuard<'_, Control> {
        self.control.lock().unwrap_or_else(|e| e.into_inner())
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Run the engine on this thread until `shared.stop`.
pub fn run(config: Config, api: Api, shared: Arc<Shared>, mut outputs: Outputs, mut audio: Option<audio::AudioPath>) {
    let clock = Clock::new(api.clone());
    let f = config.format;
    let rate = mxlio::rational(f.rate_n, f.rate_d);
    let (width, height) = (f.width as usize, f.height as usize);
    let frame_bytes = v210::frame_bytes(width, height, false);
    let interval = config.interval();
    let pool = StripPool::new(config.threads);
    let sources: Vec<SourceKind> = lock(&shared.control).sources.iter().map(|s| s.kind).collect();
    let n_inputs = config.inputs + config.key_inputs;
    let mut inputs: Vec<VideoInput> = lock(&shared.control).sources[..n_inputs]
        .iter()
        .map(|s| VideoInput::new(s.id.clone()))
        .collect();
    let mut instances = Instances::default();
    let mes = config.mes;
    // Programmes of MEs 2.. (re-entry and their outputs).
    let mut me_buf: Vec<Vec<u8>> = (0..mes).map(|m| if m == 0 { Vec::new() } else { vec![0u8; frame_bytes] }).collect();
    let mut mixers: Vec<Mixer10> = (0..outputs.list.len()).map(|_| Mixer10::new()).collect();
    let mut me_mixers: Vec<Mixer10> = (0..mes).map(|_| Mixer10::new()).collect();
    let hold_max_ns = config.hold_max.as_nanos() as u64;
    let wait_ns = if config.read_offset == 0 { config.input_wait_ns } else { 0 };
    {
        let mut st = lock(&shared.stats);
        st.running = true;
        st.format = f.name();
        st.threads = config.threads;
        st.simd = crate::compositor::kernels::simd_level();
        st.outputs = outputs
            .list
            .iter()
            .map(|o| OutputStatus {
                key: o.key.clone(),
                label: o.label.clone(),
                flow_id: o.flow_id.to_string(),
                ..Default::default()
            })
            .collect();
    }
    // One decision for the whole frame (on the time of
    // all drawings together): one thread while that takes under 70% of the
    // frame interval, parallel strips only to keep up.
    let mut threading = crate::compositor::mixer10::Threading::default();
    let drawings = 1.0;
    let mut n = clock.index_now(&rate) + 1;
    tracing::info!(
        "engine: {} on {} threads ({}), {} outputs",
        f.name(),
        config.threads,
        crate::compositor::kernels::simd_level(),
        outputs.list.len()
    );
    while !shared.stop.load(Ordering::Relaxed) {
        let t_n = clock.index_to_time(&rate, n);
        clock.sleep_until(t_n);
        let now_index = clock.index_now(&rate);
        if now_index > n + 1 {
            // More than a frame late: skip to now.
            let skipped = now_index - n - 1;
            lock(&shared.stats).missed += skipped;
            n = now_index;
            continue;
        }
        let started = Instant::now();
        let now = clock.now();
        let parallel = config.parallel.unwrap_or_else(|| threading.parallel());
        for m in mixers.iter_mut().chain(me_mixers.iter_mut()) {
            m.force_parallel = Some(parallel);
        }

        // Routes from IS-05.
        {
            let routes = lock(&shared.routes);
            for (i, input) in inputs.iter_mut().enumerate() {
                input.set_route(routes.get(i).cloned().flatten());
            }
        }
        let me_outs = lock(&shared.control).frame(t_n);

        // The sources the outputs draw.
        let mut needed: Vec<usize> = Vec::new();
        for (m, o) in me_outs.iter().enumerate() {
            scene::sources_of(&o.frame.program, &o.keyers, &o.dsks, &mut needed);
            if outputs.has(m, OutputKind::Preview) {
                scene::sources_of(&o.frame.preview, &o.keyers, &o.dsks, &mut needed);
            }
        }
        let input_time = clock.index_to_time(&rate, n.saturating_sub(config.read_offset));
        let frames: Vec<Option<InputFrame>> = inputs
            .iter_mut()
            .enumerate()
            .map(|(i, input)| {
                if needed.contains(&i) {
                    input.frame(&api, &mut instances, &clock, input_time, now, wait_ns, hold_max_ns)
                } else {
                    input.poll(&api, &mut instances, &clock, now);
                    None
                }
            })
            .collect();

        // Draw the MEs, highest first.
        for m in (0..mes).rev() {
            let o = &me_outs[m];
            let (lower, higher) = me_buf.split_at_mut(m + 1);
            let higher: &[Vec<u8>] = higher;
            let frames_ref = &frames;
            let sources_ref = &sources;
            let src = move |i: usize| -> Option<Source10<'_>> {
                match sources_ref.get(i)? {
                    SourceKind::Input(_) | SourceKind::Key(_) => {
                        let fr = frames_ref.get(i)?.as_ref()?;
                        // SAFETY: the grain's reader is open for this frame.
                        let data = unsafe { fr.grain.data() };
                        Source10::new(data, fr.width, fr.height, fr.has_key)
                    }
                    SourceKind::Me(j) => {
                        let buf = higher.get(j.checked_sub(m + 1)?)?;
                        Source10::new(buf, width, height, false)
                    }
                }
            };
            let pgm_layers = scene::layers(&o.frame.program, &src, &o.keyers, &o.dsks, f.width, f.height);
            let budget = interval / drawings;
            let mut out_stats: Vec<(usize, f64, bool, bool)> = Vec::new();
            if m > 0 {
                // An ME's programme: into its buffer (re-entry), then its output.
                let buf = &mut lower[m];
                let s = me_mixers[m].draw(buf, width, height, &pgm_layers, Some(BLACK10), Some(&pool), budget);
                if let Some(idx) = outputs.find(m, OutputKind::Program) {
                    let ok = outputs.list[idx].writer.write_grain(n, |p| copy_frame(p, buf)).is_ok();
                    out_stats.push((idx, s.seconds, s.parallel, ok));
                }
            } else {
                // ME 1: programme and clean feed straight into the grains.
                // The programme is the clean feed plus DSKs and FTB on top:
                // the clean feed is drawn first, and the programme is a copy
                // of it with only those layers drawn over it.
                let clean_idx = outputs.find(0, OutputKind::Clean);
                let pgm_idx = outputs.find(0, OutputKind::Program);
                let prog = &o.frame.program;
                let clean = &o.frame.clean;
                let on_top = clean_idx.is_some() && prog.len() >= clean.len() && prog[..clean.len()] == clean[..];
                let mut clean_grain: Option<(*const u8, usize)> = None;
                if let Some(ci) = clean_idx {
                    let layers = scene::layers(clean, &src, &o.keyers, &o.dsks, f.width, f.height);
                    let mixer = &mut mixers[ci];
                    let r = outputs.list[ci].writer.write_grain(n, |c| {
                        let s = mixer.draw(c, width, height, &layers, Some(BLACK10), Some(&pool), budget);
                        clean_grain = Some((c.as_ptr(), c.len()));
                        s
                    });
                    match r {
                        Ok(s) => out_stats.push((ci, s.seconds, s.parallel, true)),
                        Err(_) => out_stats.push((ci, 0.0, false, false)),
                    }
                }
                if let Some(pi) = pgm_idx {
                    let mixer = &mut mixers[pi];
                    let r = outputs.list[pi].writer.write_grain(n, |p| {
                        let s = match clean_grain.filter(|_| on_top) {
                            Some((ptr, len)) => {
                                let t = Instant::now();
                                // SAFETY: the clean grain was just committed to
                                // our own ring; it is overwritten a history later.
                                copy_frame(p, unsafe { std::slice::from_raw_parts(ptr, len) });
                                let top = scene::layers(&prog[clean.len()..], &src, &o.keyers, &o.dsks, f.width, f.height);
                                let mut s = mixer.draw(p, width, height, &top, None, Some(&pool), budget);
                                s.seconds = t.elapsed().as_secs_f64();
                                s
                            }
                            None => mixer.draw(p, width, height, &pgm_layers, Some(BLACK10), Some(&pool), budget),
                        };
                        if let Some(tap) = lock(&shared.preview).as_mut() {
                            tap.offer_program(p);
                        }
                        s
                    });
                    match r {
                        Ok(s) => out_stats.push((pi, s.seconds, s.parallel, true)),
                        Err(e) => {
                            tracing::warn!("PGM grain {n}: {e}");
                            out_stats.push((pi, 0.0, false, false));
                        }
                    }
                }
            }
            if let Some(vi) = outputs.find(m, OutputKind::Preview) {
                let pvw_layers = scene::layers(&o.frame.preview, &src, &o.keyers, &o.dsks, f.width, f.height);
                let mixer = &mut mixers[vi];
                let r = outputs.list[vi].writer.write_grain(n, |p| {
                    let s = mixer.draw(p, width, height, &pvw_layers, Some(BLACK10), Some(&pool), budget);
                    if m == 0 {
                        if let Some(tap) = lock(&shared.preview).as_mut() {
                            tap.offer_preview(p);
                        }
                    }
                    s
                });
                match r {
                    Ok(s) => out_stats.push((vi, s.seconds, s.parallel, true)),
                    Err(_) => out_stats.push((vi, 0.0, false, false)),
                }
            }
            let mut st = lock(&shared.stats);
            for (idx, secs, par, ok) in out_stats {
                let os = &mut st.outputs[idx];
                if ok {
                    os.frames += 1;
                    os.render.add(secs * 1e3);
                    os.parallel = par;
                } else {
                    os.errors += 1;
                }
            }
        }

        // Programme audio for this frame.
        if let Some(a) = audio.as_mut() {
            a.set_routes(&lock(&shared.audio_routes));
            let t1 = clock.index_to_time(&rate, n + 1);
            let (w0, w1) = {
                let c = lock(&shared.control);
                (c.audio_weights(t_n, config.inputs), c.audio_weights(t1, config.inputs))
            };
            let offset = config.read_offset * f.frame_ns();
            a.frame(&api, &mut instances, &clock, t_n, t1, offset, &w0, &w1, config.input_wait_ns);
            if a.stats.frames % 25 == 1 {
                *lock(&shared.audio) = a.stats.clone();
            }
        }

        // The preview mosaic: the sources' pictures.
        if let Some(tap) = lock(&shared.preview).as_mut() {
            if tap.wants_sources() {
                let pics: Vec<Option<(&[u8], usize, usize)>> = frames
                    .iter()
                    .map(|fr| fr.as_ref().map(|fr| (unsafe { fr.grain.data() }, fr.width, fr.height)))
                    .collect();
                tap.offer_sources(&pics);
            }
            tap.frame_done();
        }

        let secs = started.elapsed().as_secs_f64();
        if config.parallel.is_none() {
            threading.took(parallel, secs, interval);
        }
        let mut st = lock(&shared.stats);
        st.frames += 1;
        st.index = n;
        st.frame.add(secs * 1e3);
        if clock.now() > clock.index_to_time(&rate, n + 1) {
            st.late += 1;
        }
        if st.frames % 25 == 1 {
            st.inputs = inputs
                .iter()
                .map(|i| InputStatus {
                    id: i.name.clone(),
                    state: i.state,
                    route: i.route().cloned(),
                    flow: i.flow().cloned(),
                    error: i.error.clone(),
                    counters: i.counters,
                })
                .collect();
        }
        drop(st);
        n += 1;
    }
    lock(&shared.stats).running = false;
    tracing::info!("engine stopped");
}

fn copy_frame(dst: &mut [u8], src: &[u8]) {
    let n = dst.len().min(src.len());
    dst[..n].copy_from_slice(&src[..n]);
}


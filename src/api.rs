//! The web API (`/api/v1/...`), the WebSocket state push, Prometheus
//! `/metrics` and the embedded UI. Routes and bodies follow FlowXer's where
//! there is an equivalent (`/mixer/cut`, `/mixer/auto`, `/mixer/preview`,
//! `/mixer/take`, `/mixer/position`, `/mixer/fade-to-black`,
//! `/panels/{id}/transition`, `/keyers/{id}`, `/tally/receivers`,
//! `/config/export`, `/config/import`).

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path, State};
use axum::http::{header, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, patch, post, put};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::config::Config;
use crate::control::{CommandError, Control, Saved, TransitionSettings};
use crate::engine::{Shared, BUCKETS_MS};
use crate::preview::MosaicLayout;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
const UI: &str = include_str!(env!("MIXER_UI_HTML"));

#[derive(Clone)]
pub struct App {
    pub config: Arc<Config>,
    pub shared: Arc<Shared>,
    pub node: mxl_nmos::Node,
    pub mosaic: Option<Arc<MosaicLayout>>,
    pub started: Instant,
    pub tai_offset_s: i64,
}

type ApiResult = Result<Json<Value>, ApiError>;

#[derive(Debug)]
pub struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({"error": self.1}))).into_response()
    }
}

impl From<CommandError> for ApiError {
    fn from(e: CommandError) -> Self {
        ApiError(StatusCode::from_u16(e.status).unwrap_or(StatusCode::BAD_REQUEST), e.message)
    }
}

fn bad(msg: impl Into<String>) -> ApiError {
    ApiError(StatusCode::BAD_REQUEST, msg.into())
}

/// An ME from `me` (1-based) or `panel_id` (`me-1`); ME 1 by default.
#[derive(Deserialize, Default)]
struct MeRef {
    me: Option<usize>,
    panel_id: Option<String>,
}

impl MeRef {
    fn me(&self) -> Result<usize, ApiError> {
        if let Some(m) = self.me {
            return Ok(m);
        }
        match &self.panel_id {
            None => Ok(1),
            Some(p) => parse_me(p),
        }
    }
}

fn parse_me(p: &str) -> Result<usize, ApiError> {
    p.strip_prefix("me-")
        .unwrap_or(p)
        .parse()
        .map_err(|_| bad(format!("panel {p:?}: expected me-<n>")))
}

#[derive(Deserialize)]
struct SourceBody {
    #[serde(flatten)]
    me: MeRef,
    #[serde(alias = "input_id", alias = "source_id")]
    source: Option<String>,
}

#[derive(Deserialize)]
struct TakeBody {
    #[serde(flatten)]
    me: MeRef,
    #[serde(alias = "input_id", alias = "source_id")]
    source: Option<String>,
    /// `cut` (default) or a transition kind: put the source on preview and
    /// run that transition
    transition: Option<String>,
    duration_ms: Option<u32>,
}

#[derive(Deserialize)]
struct PositionBody {
    #[serde(flatten)]
    me: MeRef,
    position: f64,
}

#[derive(Deserialize)]
struct FtbBody {
    #[serde(flatten)]
    me: MeRef,
    on: Option<bool>,
    #[serde(default)]
    cut: bool,
}

#[derive(Deserialize)]
struct NextBody {
    background: Option<bool>,
    ties: Option<Vec<bool>>,
}

#[derive(Deserialize, Default)]
struct OnAir {
    on: Option<bool>,
}

#[derive(Deserialize)]
struct LabelBody {
    label: String,
}

#[derive(Deserialize)]
struct TallyTargets {
    #[serde(alias = "receivers")]
    targets: Vec<Value>,
}

fn me_json(app: &App) -> Json<Value> {
    Json(app.shared.control().snapshot())
}

/// A command on the control state; the answer is the new state.
fn command(app: &App, f: impl FnOnce(&mut Control) -> Result<(), CommandError>) -> ApiResult {
    let mut c = app.shared.control();
    f(&mut c)?;
    Ok(Json(c.snapshot()))
}

/// Keyer ids: `me-<n>-key-<k>`, `me-<n>-dsk-<k>`, `dsk-<k>` (ME 1).
fn parse_keyer(id: &str) -> Result<(usize, usize, bool), ApiError> {
    let fail = || bad(format!("keyer {id:?}: expected me-<n>-key-<k>, me-<n>-dsk-<k> or dsk-<k>"));
    let (me, rest) = match id.strip_prefix("me-") {
        Some(r) => {
            let (m, rest) = r.split_once('-').ok_or_else(fail)?;
            (m.parse::<usize>().map_err(|_| fail())?, rest)
        }
        None => (1, id),
    };
    let (dsk, k) = if let Some(k) = rest.strip_prefix("key-") {
        (false, k)
    } else if let Some(k) = rest.strip_prefix("dsk-") {
        (true, k)
    } else {
        return Err(fail());
    };
    let k: usize = k.parse().map_err(|_| fail())?;
    if k == 0 {
        return Err(fail());
    }
    Ok((me, k - 1, dsk))
}

pub fn router(app: App) -> Router {
    Router::new()
        .route("/", get(|| async { Html(UI) }))
        .route("/livez", get(|| async { "ok" }))
        .route("/readyz", get(readyz))
        .route("/metrics", get(metrics))
        .route("/api/v1/health", get(health))
        .route("/api/v1/state", get(state))
        .route("/api/v1/console", get(state))
        .route("/api/v1/ws", get(ws))
        .route("/api/v1/engine", get(engine))
        .route("/api/v1/inputs", get(inputs))
        .route("/api/v1/outputs", get(outputs))
        .route("/api/v1/preview", get(preview))
        .route("/api/v1/sources/{id}", patch(label))
        .route("/api/v1/mixer/preview", post(set_preview))
        .route("/api/v1/mixer/program", post(set_program))
        .route("/api/v1/mixer/take", post(take))
        .route("/api/v1/mixer/cut", post(cut))
        .route("/api/v1/mixer/auto", post(auto))
        .route("/api/v1/mixer/position", post(position))
        .route("/api/v1/mixer/fade-to-black", post(ftb))
        .route("/api/v1/panels/{id}/transition", patch(transition))
        .route("/api/v1/panels/{id}/next", patch(next))
        .route("/api/v1/panels/{id}/ftb", patch(ftb_rate))
        .route("/api/v1/keyers", get(keyers))
        .route("/api/v1/keyers/{id}", patch(keyer_patch))
        .route("/api/v1/keyers/{id}/auto", post(keyer_auto))
        .route("/api/v1/keyers/{id}/cut", post(keyer_cut))
        .route("/api/v1/tally", get(tally))
        .route("/api/v1/tally/receivers", put(tally_receivers))
        .route("/api/v1/config", get(config))
        .route("/api/v1/config/export", get(export))
        .route("/api/v1/config/import", post(import))
        .with_state(app)
}

async fn readyz(State(app): State<App>) -> impl IntoResponse {
    let running = app.shared.stats.lock().map(|s| s.running).unwrap_or(false);
    let registered = !app.node.registration_required() || app.node.is_registered();
    if running && registered {
        (StatusCode::OK, "ready")
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "not ready")
    }
}

async fn health(State(app): State<App>) -> Json<Value> {
    let st = app.shared.stats.lock().unwrap_or_else(|e| e.into_inner()).clone();
    Json(json!({
        "status": if st.running { "ok" } else { "starting" },
        "version": VERSION,
        "uptime_s": app.started.elapsed().as_secs(),
        "format": app.config.format.name(),
        "frames": st.frames,
        "missed": st.missed,
        "late": st.late,
        "nmos": {
            "node_id": app.node.node_id(),
            "registered": app.node.is_registered(),
            "host": app.node.host().to_string(),
            "port": app.config.nmos_port,
        },
        "tai_offset_s": app.tai_offset_s,
    }))
}

async fn state(State(app): State<App>) -> Json<Value> {
    let mut v = app.shared.control().snapshot();
    let st = app.shared.stats.lock().unwrap_or_else(|e| e.into_inner()).clone();
    v["engine"] = json!({
        "format": st.format, "frames": st.frames, "missed": st.missed, "late": st.late,
        "frame_ms_last": st.frame.last_ms,
        "frame_ms_avg": if st.frame.count > 0 { st.frame.sum_ms / st.frame.count as f64 } else { 0.0 },
        "threads": st.threads, "simd": st.simd,
        "outputs": st.outputs.iter().map(|o| json!({
            "key": o.key, "label": o.label, "frames": o.frames, "errors": o.errors,
            "render_ms_last": o.render.last_ms, "parallel": o.parallel,
        })).collect::<Vec<_>>(),
        "inputs": st.inputs,
    });
    v["preview"] = preview_json(&app);
    v["audio"] = json!(*app.shared.audio.lock().unwrap_or_else(|e| e.into_inner()));
    Json(v)
}

async fn engine(State(app): State<App>) -> Json<Value> {
    Json(serde_json::to_value(&*app.shared.stats.lock().unwrap_or_else(|e| e.into_inner())).unwrap_or_default())
}

async fn inputs(State(app): State<App>) -> Json<Value> {
    Json(json!(app.shared.stats.lock().unwrap_or_else(|e| e.into_inner()).inputs))
}

async fn outputs(State(app): State<App>) -> Json<Value> {
    Json(json!(app.shared.stats.lock().unwrap_or_else(|e| e.into_inner()).outputs))
}

fn preview_json(app: &App) -> Value {
    let c = &app.config;
    let path = format!("{}/mosaic", c.preview_path_prefix.trim_matches('/'));
    match (&app.mosaic, &c.preview_whep_url) {
        (Some(m), whep) => json!({
            "enabled": true,
            "path": path,
            "whep_url": whep.as_ref().map(|w| format!("{}/{path}/whep", w.trim_end_matches('/'))),
            "hls_url": c.preview_hls_url.as_ref().map(|h| format!("{}/{path}/index.m3u8", h.trim_end_matches('/'))),
            "fps": c.preview_fps,
            "layout": **m,
        }),
        (None, _) => json!({"enabled": false, "reason": "PREVIEW_PUBLISH_URL is not set"}),
    }
}

async fn preview(State(app): State<App>) -> Json<Value> {
    Json(preview_json(&app))
}

async fn ws(State(app): State<App>, upgrade: WebSocketUpgrade) -> Response {
    upgrade.on_upgrade(move |socket| push(app, socket))
}

/// Push the state on every change (checked every 40 ms) and at least once a
/// second (engine numbers).
async fn push(app: App, mut socket: WebSocket) {
    let mut last = 0u64;
    let mut sent = Instant::now() - Duration::from_secs(10);
    let mut tick = tokio::time::interval(Duration::from_millis(40));
    loop {
        tokio::select! {
            _ = tick.tick() => {}
            msg = socket.recv() => match msg {
                Some(Ok(Message::Close(_))) | None | Some(Err(_)) => return,
                _ => continue,
            },
        }
        let rev = app.shared.control().revision;
        if rev == last && sent.elapsed() < Duration::from_secs(1) {
            continue;
        }
        last = rev;
        sent = Instant::now();
        let Json(v) = state(State(app.clone())).await;
        if socket.send(Message::Text(v.to_string().into())).await.is_err() {
            return;
        }
    }
}

async fn label(State(app): State<App>, Path(id): Path<String>, Json(b): Json<LabelBody>) -> ApiResult {
    command(&app, |c| c.set_label(&id, &b.label))
}

async fn set_preview(State(app): State<App>, Json(b): Json<SourceBody>) -> ApiResult {
    let me = b.me.me()?;
    command(&app, |c| c.preview(me, b.source.as_deref()))
}

async fn set_program(State(app): State<App>, Json(b): Json<SourceBody>) -> ApiResult {
    let me = b.me.me()?;
    command(&app, |c| c.program(me, b.source.as_deref()))
}

async fn take(State(app): State<App>, Json(b): Json<TakeBody>) -> ApiResult {
    let me = b.me.me()?;
    command(&app, |c| match b.transition.as_deref() {
        None | Some("cut") => c.program(me, b.source.as_deref()),
        Some(kind) => {
            let kind: crate::vision::transition::TransitionKind = serde_json::from_value(json!(kind))
                .map_err(|_| CommandError::bad(format!("transition {kind:?}: cut, mix, dip, wipe, push or slide")))?;
            let mut t = c.settings.get(me.saturating_sub(1)).map(|s| s.transition.clone()).unwrap_or_default();
            t.kind = kind;
            if let Some(d) = b.duration_ms {
                t.duration_ms = d;
            }
            c.set_transition(me, t)?;
            if b.source.is_some() {
                c.preview(me, b.source.as_deref())?;
            }
            c.auto(me)
        }
    })
}

async fn cut(State(app): State<App>, b: Option<Json<MeRef>>) -> ApiResult {
    let me = b.map(|b| b.0).unwrap_or_default().me()?;
    command(&app, |c| c.cut(me))
}

async fn auto(State(app): State<App>, b: Option<Json<MeRef>>) -> ApiResult {
    let me = b.map(|b| b.0).unwrap_or_default().me()?;
    command(&app, |c| c.auto(me))
}

async fn position(State(app): State<App>, Json(b): Json<PositionBody>) -> ApiResult {
    let me = b.me.me()?;
    command(&app, |c| c.tbar(me, b.position))
}

async fn ftb(State(app): State<App>, b: Option<Json<FtbBody>>) -> ApiResult {
    let (me, on, cut) = match b {
        Some(Json(b)) => (b.me.me()?, b.on, b.cut),
        None => (1, None, false),
    };
    command(&app, |c| c.ftb(me, on, cut))
}

/// Merge a partial JSON object into a serialisable value.
fn merge<T: serde::Serialize + serde::de::DeserializeOwned>(current: &T, patch: Value) -> Result<T, ApiError> {
    let mut v = serde_json::to_value(current).map_err(|e| bad(e.to_string()))?;
    merge_value(&mut v, patch);
    serde_json::from_value(v).map_err(|e| bad(e.to_string()))
}

fn merge_value(base: &mut Value, patch: Value) {
    match (base, patch) {
        (Value::Object(b), Value::Object(p)) => {
            for (k, v) in p {
                merge_value(b.entry(k).or_insert(Value::Null), v);
            }
        }
        (b, p) => *b = p,
    }
}

async fn transition(State(app): State<App>, Path(id): Path<String>, Json(patch): Json<Value>) -> ApiResult {
    let me = parse_me(&id)?;
    let current: TransitionSettings = app
        .shared
        .control()
        .settings
        .get(me.saturating_sub(1))
        .map(|s| s.transition.clone())
        .ok_or_else(|| ApiError(StatusCode::NOT_FOUND, format!("no ME {me}")))?;
    let t = merge(&current, patch)?;
    command(&app, |c| c.set_transition(me, t))
}

async fn next(State(app): State<App>, Path(id): Path<String>, Json(b): Json<NextBody>) -> ApiResult {
    let me = parse_me(&id)?;
    command(&app, |c| c.set_ties(me, b.background, b.ties))
}

#[derive(Deserialize)]
struct RateBody {
    rate_ms: u32,
}

async fn ftb_rate(State(app): State<App>, Path(id): Path<String>, Json(b): Json<RateBody>) -> ApiResult {
    let me = parse_me(&id)?;
    command(&app, |c| c.set_ftb_rate(me, b.rate_ms))
}

async fn keyers(State(app): State<App>) -> Json<Value> {
    let v = app.shared.control().snapshot();
    let mut list = Vec::new();
    for me in v["mes"].as_array().into_iter().flatten() {
        for k in me["keyers"].as_array().into_iter().flatten().chain(me["dsks"].as_array().into_iter().flatten()) {
            list.push(k.clone());
        }
    }
    Json(json!(list))
}

async fn keyer_patch(State(app): State<App>, Path(id): Path<String>, Json(patch): Json<Value>) -> ApiResult {
    let (me, k, dsk) = parse_keyer(&id)?;
    let current = {
        let c = app.shared.control();
        let s = c.settings.get(me.saturating_sub(1)).ok_or_else(|| ApiError(StatusCode::NOT_FOUND, format!("no ME {me}")))?;
        let list = if dsk { &s.dsks } else { &s.keyers };
        list.get(k).cloned().ok_or_else(|| ApiError(StatusCode::NOT_FOUND, format!("no keyer {id}")))?
    };
    // `enabled` (FlowXer) puts the keyer on air or off.
    let onair = patch.get("enabled").and_then(|v| v.as_bool());
    let mut patch = patch;
    if let Some(o) = patch.as_object_mut() {
        o.remove("enabled");
    }
    let cfg = merge(&current, patch)?;
    command(&app, |c| {
        c.set_keyer(me, k, dsk, cfg)?;
        if let Some(on) = onair {
            c.key_onair(me, k, dsk, Some(on), true)?;
        }
        Ok(())
    })
}

async fn keyer_auto(State(app): State<App>, Path(id): Path<String>, b: Option<Json<OnAir>>) -> ApiResult {
    let (me, k, dsk) = parse_keyer(&id)?;
    let on = b.and_then(|b| b.0.on);
    command(&app, |c| c.key_onair(me, k, dsk, on, false))
}

async fn keyer_cut(State(app): State<App>, Path(id): Path<String>, b: Option<Json<OnAir>>) -> ApiResult {
    let (me, k, dsk) = parse_keyer(&id)?;
    let on = b.and_then(|b| b.0.on);
    command(&app, |c| c.key_onair(me, k, dsk, on, true))
}

async fn tally(State(app): State<App>) -> Json<Value> {
    let c = app.shared.control();
    Json(json!(c
        .sources
        .iter()
        .zip(&c.tally)
        .map(|(s, t)| json!({"id": s.id, "label": s.label, "program": t.program, "preview": t.preview}))
        .collect::<Vec<_>>()))
}

async fn tally_receivers(State(app): State<App>, Json(b): Json<TallyTargets>) -> ApiResult {
    let mut targets = Vec::new();
    for t in b.targets {
        let s = match &t {
            Value::String(s) => s.clone(),
            Value::Object(o) => match (o.get("host").and_then(|v| v.as_str()), o.get("port").and_then(|v| v.as_u64())) {
                (Some(h), Some(p)) => format!("{h}:{p}"),
                _ => return Err(bad("receiver: expected \"host:port\" or {host, port}")),
            },
            _ => return Err(bad("receiver: expected \"host:port\" or {host, port}")),
        };
        if s.parse::<std::net::SocketAddr>().is_err() && !s.contains(':') {
            return Err(bad(format!("receiver {s:?}: expected host:port")));
        }
        targets.push(s);
    }
    command(&app, |c| {
        c.tally_targets = targets;
        c.revision += 1;
        Ok(())
    })
}

async fn config(State(app): State<App>) -> Json<Value> {
    let c = &app.config;
    Json(json!({
        "version": VERSION,
        "format": c.format, "format_name": c.format.name(),
        "inputs": c.inputs, "key_inputs": c.key_inputs, "mes": c.mes,
        "keyers_per_me": c.keyers_per_me, "dsks": c.dsks,
        "clean_feed": c.clean_feed, "me_outputs": c.me_outputs,
        "audio": c.audio, "threads": c.threads,
        "read_offset_grains": c.read_offset, "hold_max_s": c.hold_max.as_secs_f64(),
        "web_port": c.web_port, "nmos_port": c.nmos_port,
        "output_domain": {"dir": c.output_domain_dir, "id": c.output_domain_id},
    }))
}

async fn export(State(app): State<App>) -> Response {
    let saved = app.shared.control().saved();
    (
        [
            (header::CONTENT_TYPE, "application/json"),
            (header::CONTENT_DISPOSITION, "attachment; filename=\"lm-mxl-mixer-config.json\""),
        ],
        serde_json::to_string_pretty(&saved).unwrap_or_default(),
    )
        .into_response()
}

async fn import(State(app): State<App>, Json(saved): Json<Saved>) -> ApiResult {
    let warnings = app.shared.control().apply(&saved);
    let mut v = me_json(&app).0;
    v["warnings"] = json!(warnings);
    Ok(Json(v))
}

async fn metrics(State(app): State<App>) -> Response {
    use std::fmt::Write;
    let st = app.shared.stats.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let mut m = String::new();
    let mut counter = |name: &str, help: &str, v: u64| {
        let _ = writeln!(m, "# HELP {name} {help}\n# TYPE {name} counter\n{name} {v}");
    };
    counter("lm_mixer_frames_total", "Output frames drawn", st.frames);
    counter("lm_mixer_frames_missed_total", "Output grains not written (the engine was more than a frame late)", st.missed);
    counter("lm_mixer_frames_late_total", "Frames finished after the next frame's time", st.late);
    let hist = |m: &mut String, name: &str, labels: &str, h: &crate::engine::Histogram| {
        for (i, b) in BUCKETS_MS.iter().enumerate() {
            let sep = if labels.is_empty() { "" } else { "," };
            let _ = writeln!(m, "{name}_bucket{{{labels}{sep}le=\"{}\"}} {}", b / 1e3, h.counts[i]);
        }
        let sep = if labels.is_empty() { "" } else { "," };
        let _ = writeln!(m, "{name}_bucket{{{labels}{sep}le=\"+Inf\"}} {}", h.count);
        let l = if labels.is_empty() { String::new() } else { format!("{{{labels}}}") };
        let _ = writeln!(m, "{name}_sum{l} {}", h.sum_ms / 1e3);
        let _ = writeln!(m, "{name}_count{l} {}", h.count);
    };
    let _ = writeln!(m, "# HELP lm_mixer_frame_seconds Time to draw all outputs of a frame\n# TYPE lm_mixer_frame_seconds histogram");
    hist(&mut m, "lm_mixer_frame_seconds", "", &st.frame);
    let _ = writeln!(m, "# HELP lm_mixer_mix_seconds Time to draw one output frame (mix time per frame)\n# TYPE lm_mixer_mix_seconds histogram");
    for o in &st.outputs {
        hist(&mut m, "lm_mixer_mix_seconds", &format!("output=\"{}\"", o.key), &o.render);
    }
    let _ = writeln!(m, "# HELP lm_mixer_output_errors_total Output grains that could not be written\n# TYPE lm_mixer_output_errors_total counter");
    for o in &st.outputs {
        let _ = writeln!(m, "lm_mixer_output_errors_total{{output=\"{}\"}} {}", o.key, o.errors);
    }
    let _ = writeln!(m, "# HELP lm_mixer_parallel Whether an output is drawn in parallel strips\n# TYPE lm_mixer_parallel gauge");
    for o in &st.outputs {
        let _ = writeln!(m, "lm_mixer_parallel{{output=\"{}\"}} {}", o.key, o.parallel as u8);
    }
    let _ = writeln!(m, "# HELP lm_mixer_threads Drawing threads (strip pool)\n# TYPE lm_mixer_threads gauge\nlm_mixer_threads {}", st.threads);
    for (name, help, f) in [
        ("lm_mixer_input_frames_late_total", "Frames drawn from an older grain than wanted (the writer was behind)", 0),
        ("lm_mixer_input_frames_held_total", "Frames drawn from the last grain again", 1),
        ("lm_mixer_input_frames_missed_total", "Frames with nothing to draw (black)", 2),
    ] {
        let _ = writeln!(m, "# HELP {name} {help}\n# TYPE {name} counter");
        for i in &st.inputs {
            let c = &i.counters;
            let v = [c.late, c.held, c.missed][f];
            let _ = writeln!(m, "{name}{{input=\"{}\"}} {v}", i.id);
        }
    }
    let _ = writeln!(m, "# HELP lm_mixer_input_ok Whether an input delivers grains\n# TYPE lm_mixer_input_ok gauge");
    for i in &st.inputs {
        let ok = matches!(i.state, crate::engine::inputs::InputState::Ok) as u8;
        let _ = writeln!(m, "lm_mixer_input_ok{{input=\"{}\"}} {ok}", i.id);
    }
    {
        let c = app.shared.control();
        let _ = writeln!(m, "# HELP lm_mixer_tally Tally of a source (1 program, 2 preview, 3 both)\n# TYPE lm_mixer_tally gauge");
        for (s, t) in c.sources.iter().zip(&c.tally) {
            let _ = writeln!(m, "lm_mixer_tally{{source=\"{}\"}} {}", s.id, t.program as u8 + 2 * t.preview as u8);
        }
    }
    if let Some(tap) = app.shared.preview.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
        let _ = writeln!(
            m,
            "# HELP lm_mixer_preview_frames_total Preview mosaic frames sent to the encoder\n# TYPE lm_mixer_preview_frames_total counter\nlm_mixer_preview_frames_total {}\n# HELP lm_mixer_preview_dropped_total Preview frames dropped (encoder busy)\n# TYPE lm_mixer_preview_dropped_total counter\nlm_mixer_preview_dropped_total {}\n# HELP lm_mixer_preview_draw_seconds_total Time spent drawing the preview mosaic\n# TYPE lm_mixer_preview_draw_seconds_total counter\nlm_mixer_preview_draw_seconds_total {}",
            tap.drawn, tap.dropped, tap.draw_ms / 1e3
        );
    }
    {
        let a = app.shared.audio.lock().unwrap_or_else(|e| e.into_inner()).clone();
        if a.mode.is_some() {
            let _ = writeln!(
                m,
                "# HELP lm_mixer_audio_frames_total Programme audio frames written\n# TYPE lm_mixer_audio_frames_total counter\nlm_mixer_audio_frames_total {}\n# HELP lm_mixer_audio_input_gaps_total Input audio not there in time (silence instead)\n# TYPE lm_mixer_audio_input_gaps_total counter\nlm_mixer_audio_input_gaps_total {}",
                a.frames, a.input_gaps
            );
            let _ = writeln!(
                m,
                "# HELP lm_mixer_audio_input_gaps_by_input_total Input audio not there in time, per input\n# TYPE lm_mixer_audio_input_gaps_by_input_total counter"
            );
            for ((name, _, _), gaps) in a.inputs.iter().zip(&a.input_gaps_by_input) {
                let input = name.trim_end_matches("-audio");
                let _ = writeln!(m, "lm_mixer_audio_input_gaps_by_input_total{{input=\"{input}\"}} {gaps}");
            }
        }
    }
    let _ = writeln!(
        m,
        "# HELP lm_mixer_nmos_registered Whether the node is registered\n# TYPE lm_mixer_nmos_registered gauge\nlm_mixer_nmos_registered {}",
        app.node.is_registered() as u8
    );
    ([(header::CONTENT_TYPE, "text/plain; version=0.0.4")], m).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keyer_ids() {
        assert_eq!(parse_keyer("me-1-key-2").ok(), Some((1, 1, false)));
        assert_eq!(parse_keyer("me-2-dsk-1").ok(), Some((2, 0, true)));
        assert_eq!(parse_keyer("dsk-1").ok(), Some((1, 0, true)));
        assert!(parse_keyer("me-1-key-0").is_err());
        assert!(parse_keyer("key").is_err());
        assert_eq!(parse_me("me-3").ok(), Some(3));
    }

    #[test]
    fn merging_keeps_what_is_not_patched() {
        let t = TransitionSettings::default();
        let m: TransitionSettings = merge(&t, json!({"wipe": {"soft": 0.2}, "type": "wipe"})).unwrap();
        assert_eq!(m.wipe.soft, 0.2);
        assert_eq!(m.duration_ms, t.duration_ms);
        assert!(merge::<TransitionSettings>(&t, json!({"type": "nope"})).is_err());
    }
}

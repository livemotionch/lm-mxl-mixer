//! Configuration from the environment (the mxl-stack conventions; see the
//! README's table). Invalid values are an error: the process exits with 78.

use std::path::PathBuf;
use std::time::Duration;

use uuid::Uuid;

use crate::mxlio::flowdef::VideoFormat;

/// Audio of the programme output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum AudioMode {
    /// No audio receivers or sender
    Off,
    /// The audio of whatever is on the programme, crossfaded with the
    /// transitions (equal power; a dip goes through silence)
    Afv,
    /// One audio receiver passed to the programme audio sender untouched
    Passthrough,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub format: VideoFormat,
    pub inputs: usize,
    pub key_inputs: usize,
    pub mes: usize,
    pub keyers_per_me: usize,
    pub dsks: usize,
    pub clean_feed: bool,
    pub me_outputs: bool,
    /// Prefix of every NMOS label of this instance
    pub label: String,
    pub nmos_label: String,
    pub nmos_port: u16,
    pub web_port: u16,
    pub output_domain_dir: PathBuf,
    pub scan_path: PathBuf,
    pub output_domain_id: Uuid,
    pub seed: String,
    pub cleanup_on_exit: bool,
    pub history_ns: u64,
    pub state_dir: PathBuf,
    pub shutdown_timeout: Duration,
    pub read_offset: u64,
    /// Wait for an input grain this long before falling back to an older
    /// one (only with a read offset of 0)
    pub input_wait_ns: u64,
    pub hold_max: Duration,
    pub threads: usize,
    /// `Some`: always (or never) draw in parallel strips
    pub parallel: Option<bool>,
    pub audio: AudioMode,
    pub audio_channels: usize,
    pub preview_publish_url: Option<String>,
    pub preview_path_prefix: String,
    pub preview_whep_url: Option<String>,
    pub preview_hls_url: Option<String>,
    pub mediamtx_api_url: Option<String>,
    pub preview_fps: u32,
    pub preview_kbps: u32,
    pub ffmpeg: String,
    pub tsl_targets: Vec<String>,
    pub tsl_screen: u16,
    pub log_level: String,
}

fn var(name: &str) -> Option<String> {
    std::env::var(name).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty())
}

fn parse<T: std::str::FromStr>(name: &str, default: T) -> Result<T, String> {
    match var(name) {
        None => Ok(default),
        Some(v) => v.parse().map_err(|_| format!("{name}={v:?} is not valid")),
    }
}

fn boolean(name: &str, default: bool) -> Result<bool, String> {
    match var(name).as_deref().map(str::to_ascii_lowercase).as_deref() {
        None => Ok(default),
        Some("1" | "true" | "yes" | "on") => Ok(true),
        Some("0" | "false" | "no" | "off") => Ok(false),
        Some(v) => Err(format!("{name}={v:?}: expected true or false")),
    }
}

fn ranged(name: &str, default: usize, lo: usize, hi: usize) -> Result<usize, String> {
    let v: usize = parse(name, default)?;
    if v < lo || v > hi {
        return Err(format!("{name}={v}: must be {lo}–{hi}"));
    }
    Ok(v)
}

impl Config {
    pub fn from_env() -> Result<Config, String> {
        let format = VideoFormat::parse(&var("MIXER_FORMAT").unwrap_or_else(|| "1080p50".into()))?;
        let scan_path = PathBuf::from(var("MXL_DOMAIN_SCAN_PATH").unwrap_or_else(|| "/Volumes/mxl".into()));
        let output_domain_dir = PathBuf::from(
            var("MXL_OUTPUT_DOMAIN_DIR").unwrap_or_else(|| scan_path.join("vision-mixer").to_string_lossy().into()),
        );
        let seed = var("NMOS_SEED").unwrap_or_else(|| "lm-mxl-mixer".into());
        let output_domain_id = mxl_nmos::ids::derive_id(&seed, "mxl-output-domain");
        let parallel = match var("MIXER_PARALLEL").as_deref() {
            None | Some("auto") => None,
            Some("on" | "true") => Some(true),
            Some("off" | "false") => Some(false),
            Some(v) => return Err(format!("MIXER_PARALLEL={v:?}: expected auto, on or off")),
        };
        let audio = match var("MIXER_AUDIO").as_deref() {
            None | Some("afv") => AudioMode::Afv,
            Some("passthrough") => AudioMode::Passthrough,
            Some("off") => AudioMode::Off,
            Some(v) => return Err(format!("MIXER_AUDIO={v:?}: expected afv, passthrough or off")),
        };
        let default_threads = (std::thread::available_parallelism().map_or(2, |n| n.get()) / 2).clamp(1, 4);
        let label = var("MIXER_LABEL_PREFIX").unwrap_or_else(|| "VM".into());
        Ok(Config {
            format,
            inputs: ranged("MIXER_INPUTS", 8, 1, 32)?,
            key_inputs: ranged("MIXER_KEY_INPUTS", 2, 0, 8)?,
            mes: ranged("MIXER_MES", 1, 1, 4)?,
            keyers_per_me: ranged("MIXER_KEYERS_PER_ME", 2, 0, 4)?,
            dsks: ranged("MIXER_DSKS", 2, 0, 4)?,
            clean_feed: boolean("MIXER_CLEAN_FEED", true)?,
            me_outputs: boolean("MIXER_ME_OUTPUTS", true)?,
            nmos_label: var("NMOS_LABEL").unwrap_or_else(|| "Vision mixer".into()),
            label,
            nmos_port: parse("NMOS_PORT", 3332)?,
            web_port: parse("WEB_PORT", 8180)?,
            output_domain_dir,
            scan_path,
            output_domain_id,
            seed,
            cleanup_on_exit: boolean("MXL_CLEANUP_ON_EXIT", false)?,
            history_ns: parse::<u64>("MXL_HISTORY_MS", 200)?.clamp(40, 10_000) * 1_000_000,
            state_dir: PathBuf::from(var("MIXER_STATE_DIR").unwrap_or_else(|| "/config".into())),
            shutdown_timeout: Duration::from_secs(parse("SHUTDOWN_TIMEOUT_S", 10)?),
            read_offset: parse("MIXER_READ_OFFSET_GRAINS", 1)?,
            input_wait_ns: parse::<u64>("MIXER_INPUT_WAIT_US", 5000)? * 1000,
            hold_max: Duration::from_secs_f64(parse("MIXER_HOLD_MAX_S", 10.0)?),
            threads: ranged("MIXER_THREADS", default_threads, 1, 64)?,
            parallel,
            audio,
            audio_channels: ranged("MIXER_AUDIO_CHANNELS", 2, 1, 64)?,
            preview_publish_url: var("PREVIEW_PUBLISH_URL"),
            preview_path_prefix: var("PREVIEW_PATH_PREFIX").unwrap_or_else(|| "vision-mixer".into()),
            preview_whep_url: var("PREVIEW_WHEP_URL"),
            preview_hls_url: var("PREVIEW_HLS_URL"),
            mediamtx_api_url: var("MEDIAMTX_API_URL"),
            preview_fps: ranged("PREVIEW_FPS", 25, 1, 60)? as u32,
            preview_kbps: ranged("PREVIEW_KBPS", 1500, 100, 20_000)? as u32,
            ffmpeg: var("PREVIEW_FFMPEG").unwrap_or_else(|| "ffmpeg".into()),
            tsl_targets: var("TSL_TARGETS")
                .map(|v| v.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect())
                .unwrap_or_default(),
            tsl_screen: parse("TSL_SCREEN", 0)?,
            log_level: var("LOG_LEVEL").unwrap_or_else(|| "info".into()),
        })
    }

    /// The NMOS node's settings (`NMOS_*`).
    pub fn nmos_settings(&self) -> Result<mxl_nmos::Settings, String> {
        let mut s = mxl_nmos::Settings::new(self.nmos_label.clone(), self.nmos_port);
        s.description = "CPU vision mixer (lm-mxl-mixer)".into();
        s.service_name_prefix = "lm-mxl-mixer".into();
        s.apply_env().map_err(|e| e.to_string())?;
        if s.seed.is_none() {
            s.seed = Some(self.seed.clone());
        }
        Ok(s)
    }

    /// Frame interval in seconds.
    pub fn interval(&self) -> f64 {
        self.format.frame_ns() as f64 / 1e9
    }
}

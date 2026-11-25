//! The mixer's output flows (writers in its own MXL domain).

use std::path::Path;

use uuid::Uuid;

use crate::config::Config;
use crate::mxlio::{flowdef, Api, Instance, RawWriter};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputKind {
    Program,
    Preview,
    /// The programme without DSKs and FTB
    Clean,
}

/// One output: a v210 flow and its NMOS sender.
pub struct Output {
    /// NMOS key: `pgm`, `pvw`, `clean`, `me2-pgm`, ...
    pub key: String,
    pub label: String,
    pub description: String,
    pub me: usize,
    pub kind: OutputKind,
    pub flow_id: Uuid,
    pub writer: RawWriter,
}

pub struct Outputs {
    pub list: Vec<Output>,
}

/// The outputs a configuration has: `(key, me, kind, label, description)`.
pub fn plan(config: &Config) -> Vec<(String, usize, OutputKind, String, String)> {
    let p = &config.label;
    let mut v = vec![
        ("pgm".to_string(), 0, OutputKind::Program, format!("{p} PGM Video"), "Programme of ME 1".to_string()),
        ("pvw".to_string(), 0, OutputKind::Preview, format!("{p} PVW Video"), "Preview of ME 1".to_string()),
    ];
    if config.clean_feed {
        v.push((
            "clean".into(),
            0,
            OutputKind::Clean,
            format!("{p} Clean Video"),
            "Clean feed of ME 1 (programme without DSKs and FTB)".into(),
        ));
    }
    if config.me_outputs {
        for m in 1..config.mes {
            v.push((
                format!("me{}-pgm", m + 1),
                m,
                OutputKind::Program,
                format!("{p} ME{} PGM Video", m + 1),
                format!("Programme of ME {}", m + 1),
            ));
            v.push((
                format!("me{}-pvw", m + 1),
                m,
                OutputKind::Preview,
                format!("{p} ME{} PVW Video", m + 1),
                format!("Preview of ME {}", m + 1),
            ));
        }
    }
    v
}

/// The MXL flow id of an output (stable: from the seed).
pub fn flow_id(seed: &str, key: &str) -> Uuid {
    mxl_nmos::ids::derive_id(seed, &format!("{key}/mxl-flow"))
}

impl Outputs {
    /// Create (or open) every output flow in the output domain.
    pub fn new(api: &Api, config: &Config, domain: &Path) -> Result<Outputs, String> {
        let instance = Instance::new(api, domain)?;
        let mut list = Vec::new();
        for (key, me, kind, label, description) in plan(config) {
            let id = flow_id(&config.seed, &key);
            let def = flowdef::video(id, &label, &description, &config.nmos_label, &config.format, false);
            let (writer, created) = instance
                .writer(&def)
                .map_err(|e| format!("output flow {key} ({id}): {e}"))?;
            tracing::info!("{} output {key}: flow {id} ({label})", if created { "created" } else { "opened" });
            list.push(Output {
                key,
                label,
                description,
                me,
                kind,
                flow_id: writer.id(),
                writer,
            });
        }
        Ok(Outputs { list })
    }

    pub fn find(&self, me: usize, kind: OutputKind) -> Option<usize> {
        self.list.iter().position(|o| o.me == me && o.kind == kind)
    }

    pub fn has(&self, me: usize, kind: OutputKind) -> bool {
        self.find(me, kind).is_some()
    }
}

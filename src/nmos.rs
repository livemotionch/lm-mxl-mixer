//! The NMOS node (IS-04, IS-05 with the MXL transport, BCP-007-03) on
//! LeeO86's `mxl-nmos`: one Device, a receiver per video input and key
//! input, a sender per output flow.

use std::path::PathBuf;
use std::sync::Arc;

use mxl_nmos::{
    media_types, ActiveConnection, Component, Device, Flow, Format, MxlDomain, Node, NodeImplementation, Rational,
    Receiver, ReceiverCaps, Role, Sender, Source, StagedConnection,
};
use serde_json::json;
use uuid::Uuid;

use crate::config::Config;
use crate::engine::inputs::Route;
use crate::engine::outputs;
use crate::engine::Shared;
use crate::mxlio::domain;

pub const DEVICE: &str = "mixer";

/// Receiver key of source index `i` (`in-1`…, `key-1`…).
pub fn receiver_keys(config: &Config) -> Vec<String> {
    (0..config.inputs)
        .map(|i| format!("in-{}", i + 1))
        .chain((0..config.key_inputs).map(|i| format!("key-{}", i + 1)))
        .collect()
}

fn receiver(config: &Config, key: &str, label: String, key_input: bool) -> Receiver {
    let types = vec![media_types::VIDEO_V210.to_string(), media_types::VIDEO_V210A.to_string()];
    let mut r = Receiver::mxl(key, DEVICE, Format::Video, types.clone());
    r.description = if key_input {
        format!("{label}: fill and key (video/v210a) for linear keys")
    } else {
        format!("{label}: video input (video/v210, or v210a for a linear key)")
    };
    r.label = label.clone();
    r.caps = ReceiverCaps {
        media_types: types,
        constraint_sets: vec![json!({
            "urn:x-nmos:cap:format:frame_width": {"minimum": 2, "maximum": 3840},
            "urn:x-nmos:cap:format:frame_height": {"minimum": 2, "maximum": 2160},
            "urn:x-nmos:cap:format:interlace_mode": {"enum": ["progressive"]},
        })],
        version: None,
    };
    mxl_nmos::add_group_hint(&mut r.tags, &config.nmos_label, &label);
    r
}

/// Build the node. Activations of receivers set the engine's routes; they
/// are accepted at once and never wait for media.
pub fn build(config: &Config, shared: Arc<Shared>, connections: PathBuf) -> Result<Node, String> {
    let scan = config.scan_path.clone();
    let own = config.output_domain_dir.clone();
    let keys = receiver_keys(config);
    let keys_cb = keys.clone();
    let audio_keys: Vec<String> = crate::engine::audio::receivers(config.audio, config.inputs, &config.label)
        .into_iter()
        .map(|(k, _)| k)
        .collect();
    let audio_cb = audio_keys.clone();
    let implementation = NodeImplementation::new(move |activation| {
        let shared = shared.clone();
        let keys = keys_cb.clone();
        let audio_keys = audio_cb.clone();
        Box::pin(async move {
            if activation.role == Role::Receiver {
                if let Some(i) = audio_keys.iter().position(|k| *k == activation.key) {
                    let route = match (activation.master_enable, activation.domain_path, activation.mxl_flow_id) {
                        (true, Some(domain), Some(flow)) => Some(Route { domain, flow }),
                        _ => None,
                    };
                    tracing::info!("IS-05: {} → {:?}", activation.key, route);
                    if let Some(r) = shared.audio_routes.lock().unwrap_or_else(|e| e.into_inner()).get_mut(i) {
                        *r = route;
                    }
                    return Ok(());
                }
                if let Some(i) = keys.iter().position(|k| *k == activation.key) {
                    let route = match (activation.master_enable, activation.domain_path, activation.mxl_flow_id) {
                        (true, Some(domain), Some(flow)) => Some(Route { domain, flow }),
                        _ => None,
                    };
                    tracing::info!("IS-05: {} → {:?}", activation.key, route);
                    let mut routes = shared.routes.lock().unwrap_or_else(|e| e.into_inner());
                    if let Some(r) = routes.get_mut(i) {
                        *r = route;
                    }
                    shared.control().revision += 1;
                }
            }
            Ok(())
        })
    })
    .domains(move || {
        domain::scan(&scan, &own)
            .into_iter()
            .map(|d| MxlDomain {
                id: d.id,
                label: d.label,
                path: d.path,
            })
            .collect()
    })
    .on_connections_changed(move |all| {
        if let Err(e) = std::fs::write(&connections, serde_json::to_vec_pretty(&all).unwrap_or_default()) {
            tracing::error!("saving the IS-05 connections failed: {e}");
        }
    });
    let node = Node::new(config.nmos_settings()?, implementation).map_err(|e| e.to_string())?;
    let err = |e: mxl_nmos::Error| e.to_string();
    node.insert_device(Device::new(DEVICE, "Vision Mixer").description("CPU vision mixer (lm-mxl-mixer)"))
        .map_err(err)?;
    for (i, key) in keys.iter().enumerate() {
        let (label, key_input) = if i < config.inputs {
            (format!("{} In {} Video", config.label, i + 1), false)
        } else {
            (format!("{} Key {} Video", config.label, i + 1 - config.inputs), true)
        };
        node.insert_receiver(receiver(config, key, label, key_input)).map_err(err)?;
    }
    for (key, label) in crate::engine::audio::receivers(config.audio, config.inputs, &config.label) {
        let types = vec![media_types::AUDIO_FLOAT32.to_string()];
        let mut r = Receiver::mxl(&key, DEVICE, Format::Audio, types.clone());
        r.description = format!("{label}: audio (float32, 48 kHz)");
        r.label = label.clone();
        r.caps = ReceiverCaps {
            media_types: types,
            constraint_sets: vec![json!({
                "urn:x-nmos:cap:format:channel_count": {"minimum": 1, "maximum": 64},
                "urn:x-nmos:cap:format:sample_rate": {"enum": [{"numerator": 48000, "denominator": 1}]},
                "urn:x-nmos:cap:format:sample_depth": {"enum": [32]}
            })],
            version: None,
        };
        mxl_nmos::add_group_hint(&mut r.tags, &config.nmos_label, &label);
        node.insert_receiver(r).map_err(err)?;
    }
    if config.audio != crate::config::AudioMode::Off {
        let key = "pgm-audio";
        let label = format!("{} PGM Audio", config.label);
        let flow_id = crate::engine::audio::flow_id(&config.seed);
        let channels: Vec<mxl_nmos::Channel> = (0..config.audio_channels)
            .map(|c| mxl_nmos::Channel::new(format!("Ch {}", c + 1)))
            .collect();
        let mut source = Source::audio(key, DEVICE, channels);
        source.label = label.clone();
        mxl_nmos::add_group_hint(&mut source.tags, &config.nmos_label, &label);
        node.insert_source(source).map_err(err)?;
        let mut flow = Flow::audio(key, key, DEVICE, media_types::AUDIO_FLOAT32, Rational::new(48_000, 1), Some(32));
        flow.label = label.clone();
        mxl_nmos::add_group_hint(&mut flow.tags, &config.nmos_label, &label);
        node.insert_flow(flow).map_err(err)?;
        let mut sender = Sender::mxl(key, key, DEVICE);
        sender.label = label.clone();
        sender.description = "Programme audio of ME 1 (audio follows video, or passed through)".into();
        sender.mxl_domain_id = Some(config.output_domain_id);
        sender.mxl_flow_id = Some(flow_id);
        mxl_nmos::add_group_hint(&mut sender.tags, &config.nmos_label, &label);
        node.insert_sender(sender).map_err(err)?;
        node.set_staged(
            Role::Sender,
            key,
            StagedConnection {
                master_enable: true,
                peer_id: None,
                mxl_domain_id: Some(config.output_domain_id).into(),
                mxl_flow_id: Some(flow_id).into(),
            },
        )
        .map_err(err)?;
        node.set_active(
            Role::Sender,
            key,
            ActiveConnection {
                master_enable: true,
                peer_id: None,
                mxl_domain_id: Some(config.output_domain_id),
                mxl_flow_id: Some(flow_id),
            },
        )
        .map_err(err)?;
    }
    let f = &config.format;
    for (key, _me, _kind, label, description) in outputs::plan(config) {
        let flow_id: Uuid = outputs::flow_id(&config.seed, &key);
        let mut source = Source::video(&key, DEVICE);
        source.label = label.clone();
        source.description = description.clone();
        source.grain_rate = Some(Rational::new(f.rate_n, f.rate_d));
        mxl_nmos::add_group_hint(&mut source.tags, &config.nmos_label, &label);
        node.insert_source(source).map_err(err)?;
        let mut flow = Flow::video(
            &key,
            &key,
            DEVICE,
            media_types::VIDEO_V210,
            Rational::new(f.rate_n, f.rate_d),
            f.width,
            f.height,
            Component::ycbcr_422(f.width, f.height, 10),
        );
        flow.label = label.clone();
        flow.description = description.clone();
        mxl_nmos::add_group_hint(&mut flow.tags, &config.nmos_label, &label);
        node.insert_flow(flow).map_err(err)?;
        let mut sender = Sender::mxl(&key, &key, DEVICE);
        sender.label = label.clone();
        sender.description = description;
        sender.mxl_domain_id = Some(config.output_domain_id);
        sender.mxl_flow_id = Some(flow_id);
        mxl_nmos::add_group_hint(&mut sender.tags, &config.nmos_label, &label);
        node.insert_sender(sender).map_err(err)?;
        // Senders are always active: the flow is written all the time.
        node.set_staged(
            Role::Sender,
            &key,
            StagedConnection {
                master_enable: true,
                peer_id: None,
                mxl_domain_id: Some(config.output_domain_id).into(),
                mxl_flow_id: Some(flow_id).into(),
            },
        )
        .map_err(err)?;
        node.set_active(
            Role::Sender,
            &key,
            ActiveConnection {
                master_enable: true,
                peer_id: None,
                mxl_domain_id: Some(config.output_domain_id),
                mxl_flow_id: Some(flow_id),
            },
        )
        .map_err(err)?;
    }
    Ok(node)
}

/// The saved IS-05 connections of the last run.
pub fn load_connections(path: &std::path::Path) -> Vec<mxl_nmos::SavedConnection> {
    std::fs::read(path)
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default()
}

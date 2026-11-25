//! A TAI-locked v210 test source for tests and benchmarks: `N` flows in one
//! MXL domain, each a different colour field with a moving bar and the
//! flow number as blocks, optionally v210a (fill + a key: a lower third).
//!
//! ```sh
//! mxl-pattern <domain dir> [--flows N] [--format 1080p50] [--key] [--audio] [--label Pattern] [--seed s]
//! ```
//!
//! With `--audio`, each video flow gets a 2-channel float32 48 kHz flow too,
//! holding the constant level `(n + 1) / 10` (n: the flow's number from 0):
//! easy to recognise in a mix.
//!
//! Prints `<flow id> <label>` for each flow, then writes until SIGTERM/ctrl-C.
//! Frames are made once (50 per flow) and copied into each grain.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use lm_mxl_mixer::compositor::kernels10::argb_to_ycbcr10;
use lm_mxl_mixer::compositor::v210;
use lm_mxl_mixer::mxlio::{self, domain, flowdef, Clock, Instance};

/// The colour of flow `i` (distinct hues).
pub fn colour(i: usize) -> u32 {
    const C: [u32; 8] = [
        0xFFC0_2020, 0xFF20_A020, 0xFF20_40C0, 0xFFC0_C020, 0xFF20_B0B0, 0xFFB0_20B0, 0xFFE0_8020, 0xFF80_8080,
    ];
    C[i % 8] ^ (((i / 8) as u32 * 0x30) << 8)
}

fn frame(i: usize, t: usize, frames: usize, w: usize, h: usize, key: bool) -> Vec<u8> {
    let mut bytes = vec![0u8; v210::frame_bytes(w, h, key)];
    let stride = v210::line_bytes(w);
    let bg = argb_to_ycbcr10(colour(i));
    let white = argb_to_ycbcr10(0xFFFF_FFFF);
    let bar_x = (t * w / frames) & !1;
    let bar_w = (w / 16) & !1;
    let mut row = vec![0u16; 2 * w];
    for y in 0..h {
        for x in (0..w).step_by(2) {
            // A moving white bar (rows h/10..h/5); flow number as blocks in
            // the top left. The rest is the flow's colour.
            let block = y < h / 10 && x < (i + 1) * w / 40 && (x / (w / 40)) % 2 == 0;
            let bar = y >= h / 10 && y < h / 5 && x >= bar_x && x < bar_x + bar_w;
            let c = if bar || block { white } else { bg };
            row[2 * x] = c[1];
            row[2 * x + 1] = c[0];
            row[2 * x + 2] = c[2];
            row[2 * x + 3] = c[0];
        }
        v210::pack(&mut bytes[y * stride..(y + 1) * stride], 0, &row);
    }
    if key {
        let ks = v210::key_line_bytes(w);
        let base = stride * h;
        for y in 0..h {
            let r = &mut bytes[base + y * ks..base + (y + 1) * ks];
            let on = y >= h * 3 / 4 && y < h * 9 / 10;
            for x in 0..w {
                v210::set_sample(r, x, if on && x >= w / 10 && x < w * 9 / 10 { 940 } else { 64 });
            }
        }
    }
    bytes
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut dir = None;
    let (mut flows, mut format, mut key, mut label, mut seed) = (1usize, "1080p50".to_string(), false, "Pattern".to_string(), None);
    let mut with_audio = false;
    let mut it = args.iter().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--flows" => flows = it.next().and_then(|v| v.parse().ok()).expect("--flows N"),
            "--format" => format = it.next().expect("--format").clone(),
            "--key" => key = true,
            "--audio" => with_audio = true,
            "--label" => label = it.next().expect("--label").clone(),
            "--seed" => seed = Some(it.next().expect("--seed").clone()),
            d => dir = Some(PathBuf::from(d)),
        }
    }
    let dir = dir.expect("usage: mxl-pattern <domain dir> [--flows N] [--format 1080p50] [--key] [--label L]");
    let f = flowdef::VideoFormat::parse(&format).expect("format");
    let seed = seed.unwrap_or_else(|| label.clone());
    let api = mxlio::load().expect("libmxl");
    let domain_id = mxl_nmos::ids::derive_id(&seed, "domain");
    domain::ensure_output_domain(&dir, domain_id, &label, 200_000_000).expect("domain");
    let inst = Instance::new(&api, &dir).expect("instance");
    let (w, h) = (f.width as usize, f.height as usize);
    let frames_per = 50;
    let mut writers = Vec::new();
    let mut audio_writers = Vec::new();
    let mut pictures = Vec::new();
    for i in 0..flows {
        let id = mxl_nmos::ids::derive_id(&seed, &format!("flow-{i}"));
        let l = format!("{label} {}", i + 1);
        let def = flowdef::video(id, &l, "mxl-pattern test source", &label, &f, key);
        let (wr, _) = inst.writer(&def).expect("writer");
        println!("{} {l}", wr.id());
        writers.push(wr);
        if with_audio {
            let aid = mxl_nmos::ids::derive_id(&seed, &format!("audio-{i}"));
            let al = format!("{label} {} Audio", i + 1);
            let def = flowdef::audio(aid, &al, "mxl-pattern test tone (constant level)", &label, 2);
            let (aw, _) = inst.writer(&def).expect("audio writer");
            println!("{} {al}", aw.id());
            audio_writers.push((aw, (i + 1) as f32 / 10.0));
        }
        pictures.push((0..frames_per).map(|t| frame(i, t, frames_per, w, h, key)).collect::<Vec<_>>());
    }
    // SIGTERM and SIGINT are blocked here and taken by a waiting thread
    // (which inherits the mask); the loop checks the flag every frame.
    let stop = Arc::new(AtomicBool::new(false));
    unsafe {
        let mut set: libc::sigset_t = std::mem::zeroed();
        libc::sigemptyset(&mut set);
        libc::sigaddset(&mut set, libc::SIGTERM);
        libc::sigaddset(&mut set, libc::SIGINT);
        libc::pthread_sigmask(libc::SIG_BLOCK, &set, std::ptr::null_mut());
        let s = stop.clone();
        let set_copy = set;
        std::thread::spawn(move || {
            let mut got = 0;
            libc::sigwait(&set_copy, &mut got);
            s.store(true, Ordering::Relaxed);
        });
    }
    let clock = Clock::new(api.clone());
    let rate = mxlio::rational(f.rate_n, f.rate_d);
    let mut n = clock.index_now(&rate) + 1;
    while !stop.load(Ordering::Relaxed) {
        clock.sleep_until(clock.index_to_time(&rate, n));
        for (i, wr) in writers.iter_mut().enumerate() {
            let pic = &pictures[i][(n % frames_per as u64) as usize];
            let _ = wr.write_grain(n, |p| {
                let k = p.len().min(pic.len());
                p[..k].copy_from_slice(&pic[..k]);
            });
        }
        // Audio: the samples of [t(n), t(n + 1)).
        if !audio_writers.is_empty() {
            let ar = mxlio::rational(48_000, 1);
            let s0 = clock.time_to_index(&ar, clock.index_to_time(&rate, n));
            let s1 = clock.time_to_index(&ar, clock.index_to_time(&rate, n + 1));
            for (aw, level) in audio_writers.iter_mut() {
                let level = *level;
                let _ = aw.write_samples(s1, (s1 - s0) as usize, |slices| {
                    for (a, b) in slices.iter_mut() {
                        a.fill(level);
                        b.fill(level);
                    }
                });
            }
        }
        n += 1;
        let now = clock.index_now(&rate);
        if now > n {
            n = now;
        }
    }
    drop(writers);
    let _ = domain::remove_output_domain(&dir, dir.parent().unwrap_or(&dir));
}

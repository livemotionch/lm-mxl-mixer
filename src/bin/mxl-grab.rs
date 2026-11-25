//! Read the newest complete grain of a v210 flow and print pixels, or save
//! the frame as a PPM picture (8-bit RGB, BT.709):
//!
//! ```sh
//! mxl-grab <domain dir> <flow id> x,y [x,y ...]   # prints "x,y Y Cb Cr" per pixel (10-bit)
//! mxl-grab <domain dir> <flow id> --ppm out.ppm [--scale 2]
//! mxl-grab <domain dir> <flow id> --watch 5        # grains per second and gaps over 5 s
//! mxl-grab <domain dir> <flow id> --audio [ms]     # an audio flow: mean and RMS per channel
//!                                                  # over the newest ms (default 20)
//! ```

use std::io::Write;
use std::path::Path;

use lm_mxl_mixer::compositor::v210;
use lm_mxl_mixer::mxlio::{self, flowdef::InputFlow, Clock, Instance};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: mxl-grab <domain dir> <flow id> (x,y ... | --ppm file [--scale k] | --watch secs)");
        std::process::exit(2);
    }
    let api = mxlio::load().expect("libmxl");
    let inst = Instance::new(&api, Path::new(&args[1])).expect("domain");
    let flow: uuid::Uuid = args[2].parse().expect("flow id");
    if args[3] == "--audio" {
        let ms: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(20);
        let r = inst.audio_reader(flow).expect("audio reader");
        let head = r.head().expect("head");
        let count = 48 * ms;
        let mut buf = vec![Vec::new(); r.channels];
        if r.read(head, count, &mut buf, 0).expect("read") {
            let means: Vec<String> = buf
                .iter()
                .map(|c| format!("{:.4}", c.iter().map(|v| *v as f64).sum::<f64>() / c.len().max(1) as f64))
                .collect();
            let rms: Vec<String> = buf
                .iter()
                .map(|c| {
                    let p = c.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / c.len().max(1) as f64;
                    format!("{:.1}", 10.0 * p.max(1e-12).log10())
                })
                .collect();
            println!("head {head} channels {} mean {} rms_dbfs {}", r.channels, means.join(" "), rms.join(" "));
        } else {
            println!("head {head}: no samples");
        }
        return;
    }
    let def = InputFlow::parse(&inst.flow_def(flow).expect("flow")).expect("v210 flow");
    let reader = inst.video_reader(flow).expect("reader");
    let clock = Clock::new(api.clone());
    let rate = reader.grain_rate().unwrap();
    if args[3] == "--watch" {
        let secs: f64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(5.0);
        let start = clock.index_now(&rate);
        let end = start + (secs * rate.numerator as f64 / rate.denominator as f64) as u64;
        let (mut got, mut missing, mut lat) = (0u64, Vec::new(), Vec::new());
        for i in start + 1..=end {
            match reader.wait_grain(i, 100_000_000) {
                Ok(Some(_)) => {
                    got += 1;
                    lat.push((clock.now() as f64 - clock.index_to_time(&rate, i) as f64) / 1e6);
                }
                _ => missing.push(i),
            }
        }
        lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p = |q: f64| lat.get(((lat.len() as f64 - 1.0) * q) as usize).copied().unwrap_or(f64::NAN);
        println!(
            "grains {got}/{} missing {} (first {:?}); complete after TAI time ms: median {:.2} p99 {:.2} max {:.2}",
            end - start,
            missing.len(),
            missing.first(),
            p(0.5),
            p(0.99),
            p(1.0)
        );
        return;
    }
    let head = reader.head().expect("head");
    let g = reader.grain(head).ok().flatten().or_else(|| reader.grain(head - 1).ok().flatten()).expect("grain");
    let data = unsafe { g.data() };
    let (w, h) = (def.width, def.height);
    let stride = v210::line_bytes(w);
    let mut row = vec![0u16; 2 * w];
    if args[3] == "--ppm" {
        let scale: usize = args.iter().position(|a| a == "--scale").and_then(|i| args.get(i + 1)?.parse().ok()).unwrap_or(1);
        let (ow, oh) = (w / scale, h / scale);
        let mut out = std::fs::File::create(&args[4]).expect("file");
        write!(out, "P6\n{ow} {oh}\n255\n").unwrap();
        let mut px = Vec::with_capacity(ow * oh * 3);
        for oy in 0..oh {
            v210::unpack(&data[oy * scale * stride..], 0, &mut row);
            for ox in 0..ow {
                let x = ox * scale;
                let (y, cb, cr) = (row[2 * x + 1] as f64, row[4 * (x / 2)] as f64, row[4 * (x / 2) + 2] as f64);
                let (y, cb, cr) = ((y - 64.0) / 876.0, (cb - 512.0) / 896.0, (cr - 512.0) / 896.0);
                let r = y + 1.5748 * cr;
                let g = y - 0.1873 * cb - 0.4681 * cr;
                let b = y + 1.8556 * cb;
                for v in [r, g, b] {
                    px.push((v.clamp(0.0, 1.0) * 255.0).round() as u8);
                }
            }
        }
        out.write_all(&px).unwrap();
        println!("grain {} {}x{} -> {} ({ow}x{oh})", g.index, w, h, args[4]);
        return;
    }
    println!("grain {}", g.index);
    for a in &args[3..] {
        let (x, y) = a.split_once(',').expect("x,y");
        let (x, y): (usize, usize) = (x.parse().unwrap(), y.parse().unwrap());
        v210::unpack(&data[y * stride..], 0, &mut row);
        println!("{x},{y} {} {} {}", row[2 * x + 1], row[4 * (x / 2)], row[4 * (x / 2) + 2]);
    }
}

//! Read a v210 flow and report when its grains complete relative to their
//! TAI time: `probe <domain dir> <flow id> [seconds]`.
use lm_mxl_mixer::mxlio::{self, flowdef::InputFlow, Clock, Instance};
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let api = mxlio::load().expect("libmxl");
    let inst = Instance::new(&api, Path::new(&args[1])).expect("instance");
    let flow: uuid::Uuid = args[2].parse().unwrap();
    let secs: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(2);
    let def = InputFlow::parse(&inst.flow_def(flow).unwrap()).unwrap();
    println!("{def:?}");
    let r = inst.video_reader(flow).unwrap();
    let rate = r.grain_rate().unwrap();
    let clock = Clock::new(api.clone());
    let now = clock.index_now(&rate);
    println!("rate {}/{} head {} now {}", rate.numerator, rate.denominator, r.head().unwrap(), now);
    // For each of the next grains: wait until it is complete, note the delay.
    let mut delays = Vec::new();
    let start = clock.index_now(&rate) + 1;
    for i in start..start + secs * rate.numerator as u64 / rate.denominator as u64 {
        let t0 = clock.index_to_time(&rate, i);
        let g = r.wait_grain(i, 200_000_000).unwrap();
        let done = clock.now();
        if let Some(g) = g {
            delays.push((done as i64 - t0 as i64) as f64 / 1e6);
            if i == start {
                println!("grain {} len {} flags {}", g.index, g.len, g.flags);
            }
        } else {
            println!("grain {i} missing");
        }
    }
    delays.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = delays.len();
    println!("complete after its TAI time (ms): min {:.2} median {:.2} p99 {:.2} max {:.2}", delays[0], delays[n / 2], delays[n * 99 / 100], delays[n - 1]);
}

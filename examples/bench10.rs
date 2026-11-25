//! Benchmarks of the 10-bit (v210) renderer at 1080p, one thread unless
//! said otherwise: what a vision mixer's outputs cost per frame.
//!
//! ```sh
//! cargo run --release --example bench10            # all scenarios
//! cargo run --release --example bench10 -- 500 mix # 500 frames, only "mix"
//! MIXER_SIMD=none cargo run --release --example bench10   # no AVX2
//! ```
//!
//! "native vs unpacked": a mix of two frames done on the v210 words in
//! place (the renderer's way) against unpacking both to 16-bit samples,
//! mixing and packing again (a 16-bit working format).

use std::hint::black_box;
use std::time::Instant;

use lm_mxl_mixer::compositor::config::{BorderPosition, CompositionLayer, LayerBorder, LayerFit};
use lm_mxl_mixer::compositor::kernels10::{self, KeyShape, BLACK10};
use lm_mxl_mixer::compositor::mixer10::{Layer10, Mixer10};
use lm_mxl_mixer::compositor::pool::StripPool;
use lm_mxl_mixer::compositor::render10::{Keying, Source10, WipeMask, WipePattern};
use lm_mxl_mixer::compositor::{kernels, v210};

const W: usize = 1920;
const H: usize = 1080;

fn frame(seed: u32, with_key: bool) -> Vec<u8> {
    let mut x = seed.wrapping_mul(2_654_435_761).wrapping_add(7);
    let mut bytes = vec![0u8; v210::frame_bytes(W, H, with_key)];
    let stride = v210::line_bytes(W);
    let mut row = vec![0u16; 2 * W];
    for y in 0..H {
        for (s, v) in row.iter_mut().enumerate() {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            // Gradients with noise: not compressible, like a camera.
            *v = (64 + ((s / 2 + y) % 800) as u32 + (x % 64)) as u16;
        }
        v210::pack(&mut bytes[y * stride..(y + 1) * stride], 0, &row);
    }
    if with_key {
        // A lower third: opaque box in rows 800–1000, soft edges, the rest 0.
        let ks = v210::key_line_bytes(W);
        let base = stride * H;
        for y in 0..H {
            let r = &mut bytes[base + y * ks..base + (y + 1) * ks];
            for xx in 0..W {
                let v = if (800..1000).contains(&y) && (100..1800).contains(&xx) {
                    940
                } else if (790..1010).contains(&y) && (90..1810).contains(&xx) {
                    500
                } else {
                    64
                };
                v210::set_sample(r, xx, v);
            }
        }
    }
    bytes
}

struct Timing {
    best: f64,
    avg: f64,
}

fn time(frames: usize, mut f: impl FnMut()) -> Timing {
    for _ in 0..3 {
        f();
    }
    let mut best = f64::MAX;
    let mut total = 0.0;
    for _ in 0..frames {
        let t = Instant::now();
        f();
        let s = t.elapsed().as_secs_f64();
        best = best.min(s);
        total += s;
    }
    Timing {
        best: best * 1e3,
        avg: total / frames as f64 * 1e3,
    }
}

fn print(name: &str, t: Timing) {
    // On several threads the wall time is not the CPU time: no % then.
    let share = if name.contains("wall") {
        String::new()
    } else {
        format!("  {:>5.1}% of a core at 50p", t.avg / 20.0 * 100.0)
    };
    println!("{name:<52} {:>7.3} ms best {:>7.3} ms avg{share}", t.best, t.avg);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let frames: usize = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(200);
    let only = args.get(2).cloned();
    let want = |name: &str| only.as_ref().is_none_or(|o| name.contains(o.as_str()));
    println!(
        "1080p50 v210, {frames} frames, SIMD {}, {} threads available",
        kernels::simd_level(),
        std::thread::available_parallelism().map_or(1, |n| n.get())
    );

    let a = frame(1, false);
    let b = frame(2, false);
    let k = frame(3, true);
    let sa = Source10::new(&a, W, H, false).unwrap();
    let sb = Source10::new(&b, W, H, false).unwrap();
    let sk = Source10::new(&k, W, H, true).unwrap();
    let mut out = vec![0u8; v210::frame_bytes(W, H, false)];
    let pool = StripPool::new(4);
    let stride = v210::line_bytes(W);

    // Kernels alone.
    if want("kernels") {
        let mut buf = vec![0u16; 2 * W];
        print(
            "kernel: unpack a frame to 16 bit",
            time(frames, || {
                for y in 0..H {
                    v210::unpack(&a[y * stride..], 0, &mut buf);
                    black_box(&buf);
                }
            }),
        );
        let mut o = out.clone();
        print(
            "kernel: pack a frame from 16 bit",
            time(frames, || {
                for y in 0..H {
                    v210::pack(&mut o[y * stride..(y + 1) * stride], 0, &buf);
                }
                black_box(&o);
            }),
        );
        print(
            "kernel: memcpy a frame",
            time(frames, || {
                o.copy_from_slice(&a);
                black_box(&o);
            }),
        );
    }

    // Native v210 mix against a 16-bit working format.
    if want("mix") {
        print(
            "mix 50%: native, words in place",
            time(frames, || {
                out.copy_from_slice(&a);
                for y in 0..H {
                    let r = &mut out[y * stride..(y + 1) * stride];
                    v210::mix(r, &b[y * stride..(y + 1) * stride], 0, 2 * W, 16384);
                }
                black_box(&out);
            }),
        );
        let (mut ua, mut ub) = (vec![0u16; 2 * W], vec![0u16; 2 * W]);
        print(
            "mix 50%: unpack both, mix 16 bit, pack",
            time(frames, || {
                for y in 0..H {
                    v210::unpack(&a[y * stride..], 0, &mut ua);
                    v210::unpack(&b[y * stride..], 0, &mut ub);
                    kernels10::mix16(&mut ua, &ub, 16384);
                    v210::pack(&mut out[y * stride..(y + 1) * stride], 0, &ua);
                }
                black_box(&out);
            }),
        );
    }

    let mut mixer = Mixer10::new();
    mixer.force_parallel = Some(false);
    let mut scene = |name: &str, layers: &[Layer10], parallel: bool| {
        if !want(name) {
            return;
        }
        mixer.force_parallel = Some(parallel);
        let t = time(frames, || {
            mixer.draw(&mut out, W, H, layers, Some(BLACK10), Some(&pool), 0.02);
        });
        let suffix = if parallel { " (4 threads, wall)" } else { "" };
        print(&format!("scene: {name}{suffix}"), t);
    };

    let full_a = Layer10::full(sa);
    let full_b = Layer10::full(sb);
    scene("cut", &[full_a.clone()], false);
    scene("mix", &[full_a.clone(), full_b.clone().opacity(0.5)], false);
    scene("dip", &[full_a.clone(), Layer10::color(0xFF00_0000, 0.4)], false);
    let mut wipe = full_b.clone();
    wipe.clip = Some(lm_mxl_mixer::compositor::config::PixelRect { x: 0, y: 0, width: 701, height: H as i32 });
    scene("wipe hard (clip)", &[full_a.clone(), wipe], false);
    let mut iris = full_b.clone();
    iris.mask = Some(WipeMask {
        pattern: WipePattern::Iris,
        progress: 0.5,
        soft: 0.1,
        reverse: false,
    });
    scene("wipe soft iris (mask)", &[full_a.clone(), iris], false);
    let mut push_b = full_b.clone();
    push_b.layer.x = -0.6;
    let mut push_a = full_a.clone();
    push_a.layer.x = 0.4;
    scene("push (moved 768 px)", &[push_a, push_b], false);
    let mut odd_a = full_a.clone();
    odd_a.layer.x = 0.3 + 2.0 / 1920.0;
    let mut odd_b = full_b.clone();
    odd_b.layer.x = -0.7 + 2.0 / 1920.0;
    scene("push (moved, not word aligned)", &[odd_b, odd_a], false);
    let mut dsk = Layer10::full(sk);
    dsk.keying = Keying::Linear(KeyShape::linear(0.0, 1.0, false));
    scene("cut + DSK lower third (v210a key)", &[full_a.clone(), dsk.clone()], false);
    let mut luma = full_b.clone();
    luma.keying = Keying::Luma(KeyShape::luma(0.5, 0.1, false));
    scene("cut + full-screen luma key", &[full_a.clone(), luma], false);
    let pip = |x: f64, y: f64, w: f64, src: Source10<'static>| {
        let mut l = Layer10::full(src);
        l.layer = CompositionLayer {
            x,
            y,
            w,
            h: w,
            fit: LayerFit::Stretch,
            border: LayerBorder {
                width: 4,
                color: "#ffffff".into(),
                position: BorderPosition::Outside,
            },
            ..Default::default()
        };
        l
    };
    // SAFETY of 'static: the frames live until the end of main.
    let sb_static: Source10<'static> = unsafe { std::mem::transmute(sb) };
    let sa_static: Source10<'static> = unsafe { std::mem::transmute(sa) };
    scene(
        "cut + PiP 1:4 with border",
        &[full_a.clone(), pip(0.7, 0.05, 0.25, sb_static)],
        false,
    );
    scene(
        "cut + PiP 0.3 (fractional) with border",
        &[full_a.clone(), pip(0.65, 0.05, 0.3, sb_static)],
        false,
    );
    let quad: Vec<Layer10> = (0..4)
        .map(|i| {
            let mut l = pip((i % 2) as f64 * 0.5, (i / 2) as f64 * 0.5, 0.5, if i % 2 == 0 { sa_static } else { sb_static });
            l.layer.border.width = 0;
            l
        })
        .collect();
    scene("quad split (4 × 1:2)", &quad, false);
    scene("quad split (4 × 1:2)", &quad, true);
    // The preview mosaic (the 8-bit engine): PVW, PGM and two source
    // tiles from 1080p v210 into the 1280×720 I420 mosaic, per preview frame.
    if want("preview") {
        for n in [8usize, 16] {
            let ids: Vec<String> = (0..n).map(|i| format!("in-{}", i + 1)).collect();
            let (tx, rx) = lm_mxl_mixer::preview::channel();
            let mut tap = lm_mxl_mixer::preview::Tap::new(lm_mxl_mixer::preview::layout(&ids), 1.0, 1, W, H, tx);
            tap.frame_done();
            let pics: Vec<Option<(&[u8], usize, usize)>> =
                (0..n).map(|i| Some((if i % 2 == 0 { &a[..] } else { &b[..] }, W, H))).collect();
            print(
                &format!("preview mosaic frame, {n} sources (PVW, PGM, 2 tiles)"),
                time(frames, || {
                    tap.offer_preview(&a);
                    tap.offer_program(&b);
                    tap.offer_sources(&pics);
                    tap.frame_done();
                    while rx.try_recv().is_ok() {}
                }),
            );
        }
    }
    // N distinct visible sources at once: grids of boxes (worst case for
    // the number of inputs; one frame each, so memory traffic is real).
    if want("grid") {
        let frames_n: Vec<Vec<u8>> = (0..16).map(|i| frame(100 + i, false)).collect();
        for (n, cols, k) in [(4usize, 2usize, 0.5f64), (8, 4, 0.25), (16, 4, 0.25)] {
            let rows = n.div_ceil(cols);
            let layers: Vec<Layer10> = (0..n)
                .map(|i| {
                    let src = Source10::new(&frames_n[i], W, H, false).unwrap();
                    let mut l = Layer10::full(src);
                    l.layer.x = (i % cols) as f64 * k;
                    l.layer.y = (i / cols) as f64 * (1.0 / rows.max(2) as f64);
                    l.layer.w = k;
                    l.layer.h = 1.0 / rows.max(2) as f64;
                    l
                })
                .collect();
            let mut gm = Mixer10::new();
            let mut gout = vec![0u8; v210::frame_bytes(W, H, false)];
            for parallel in [false, true] {
                gm.force_parallel = Some(parallel);
                let t = time(frames, || {
                    gm.draw(&mut gout, W, H, &layers, Some(BLACK10), Some(&pool), 0.02);
                });
                let suffix = if parallel { " (4 threads, wall)" } else { "" };
                print(&format!("grid: {n} different sources in boxes{suffix}"), t);
            }
        }
    }
    let mut heavy = vec![full_a.clone(), full_b.clone().opacity(0.5), pip(0.65, 0.05, 0.3, sb_static)];
    heavy.push(dsk.clone());
    scene("mix + PiP 0.3 + DSK", &heavy, false);
    scene("mix + PiP 0.3 + DSK", &heavy, true);
}

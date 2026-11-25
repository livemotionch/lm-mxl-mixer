//! Browser previews: one low-rate H.264 mosaic over RTSP to MediaMTX,
//! played in the UI over WebRTC (WHEP), as the other mxl-stack functions do.
//!
//! The mosaic is drawn by the compositor's 8-bit engine: PVW and PGM side by side on top, the sources
//! in tiles below, read straight from the v210 grains of the frame (area
//! filter, into I420). The UI crops each picture out of the one stream. An
//! `ffmpeg` child encodes it (x264, ultrafast, zerolatency) and publishes it
//! to `PREVIEW_PUBLISH_URL/PREVIEW_PATH_PREFIX/mosaic`.
//!
//! The engine offers its frames every output frame; the tap takes one every
//! `output rate / PREVIEW_FPS` frames, and only the source tiles of a few
//! sources per preview frame (round robin), so the cost stays small. Drawing
//! happens on the engine thread (the grains are only valid there), the
//! encoder's pipe is fed by another thread; a busy encoder drops frames.

use std::io::Write;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};
use std::time::{Duration, Instant};

use serde::Serialize;

use crate::compositor::color::{argb_to, Model, BT709};
use crate::compositor::config::PixelRect;
use crate::compositor::formats::SrcFormat;
use crate::compositor::layout::{Item, Plan};
use crate::compositor::render::{render, CacheMode, Canvas, CanvasFormat, Frame, LayerInput, LayerKey, Prepared, SourceFrame};
use crate::compositor::v210;

/// Where each picture is in the mosaic (pixels), for the UI.
#[derive(Debug, Clone, Serialize)]
pub struct Tile {
    /// `pvw`, `pgm` or a source id
    pub id: String,
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

#[derive(Debug, Clone, Serialize)]
pub struct MosaicLayout {
    pub width: usize,
    pub height: usize,
    pub tiles: Vec<Tile>,
}

/// The mosaic's layout for `sources` sources on a 1280×720 canvas: PVW and
/// PGM 640×360 on top, the sources in rows below (sizes even).
pub fn layout(source_ids: &[String]) -> MosaicLayout {
    let (w, h) = (1280usize, 720usize);
    let mut tiles = vec![
        Tile { id: "pvw".into(), x: 0, y: 0, w: 640, h: 360 },
        Tile { id: "pgm".into(), x: 640, y: 0, w: 640, h: 360 },
    ];
    let n = source_ids.len().max(1);
    let cols = [4usize, 6, 8].into_iter().find(|c| c * (c / 2) >= n).unwrap_or(8);
    let tw = (w / cols) & !1;
    let th = (tw * 9 / 16) & !1;
    for (i, id) in source_ids.iter().enumerate() {
        let (c, r) = (i % cols, i / cols);
        let y = 360 + r * th;
        if y + th > h {
            break;
        }
        tiles.push(Tile {
            id: id.clone(),
            x: (c * tw) as i32,
            y: y as i32,
            w: tw as i32,
            h: th as i32,
        });
    }
    MosaicLayout { width: w, height: h, tiles }
}

/// Draws the mosaic on the engine thread and hands frames to the encoder.
pub struct Tap {
    layout: MosaicLayout,
    canvas: [Vec<u8>; 3],
    /// Tables per tile
    prepared: Vec<Option<Prepared>>,
    every: u64,
    count: u64,
    /// Source tiles drawn per preview frame
    per_frame: usize,
    next_source: usize,
    take: bool,
    width: usize,
    height: usize,
    tx: SyncSender<Vec<u8>>,
    pub drawn: u64,
    pub dropped: u64,
    pub draw_ms: f64,
}

const BLACK: u32 = 0xFF00_0000;

impl Tap {
    pub fn new(layout: MosaicLayout, out_fps: f64, fps: u32, width: usize, height: usize, tx: SyncSender<Vec<u8>>) -> Tap {
        let (w, h) = (layout.width, layout.height);
        let model = Model::yuv(BT709);
        let black = argb_to(model, BLACK);
        let canvas = [vec![black[0]; w * h], vec![black[1]; w * h / 4], vec![black[2]; w * h / 4]];
        let tiles = layout.tiles.len();
        Tap {
            layout,
            canvas,
            prepared: (0..tiles).map(|_| None).collect(),
            every: (out_fps / fps.max(1) as f64).round().max(1.0) as u64,
            count: 0,
            per_frame: 2,
            next_source: 0,
            take: false,
            width,
            height,
            tx,
            drawn: 0,
            dropped: 0,
            draw_ms: 0.0,
        }
    }

    fn draw(&mut self, tile: usize, data: &[u8], width: usize, height: usize) {
        let t = &self.layout.tiles[tile];
        let picture = PixelRect { x: t.x, y: t.y, width: t.w, height: t.h };
        let stride = v210::line_bytes(width);
        if data.len() < stride * height {
            return;
        }
        // Thumbnails of 7.5:1 and more read every n-th row
        // (decimation): a view of the frame with a longer stride, starting
        // in the middle of the first block of rows.
        let vstep = if crate::compositor::render10::decimate() {
            crate::compositor::render::vertical_step(height as f64 / t.h.max(1) as f64)
        } else {
            1
        };
        let phase = (vstep - 1) / 2;
        let data = &data[phase * stride..];
        let (height, stride_read) = ((height - phase) / vstep, stride * vstep);
        let key = LayerKey {
            format: SrcFormat::V210,
            width,
            height,
            model: Model::yuv(BT709),
            crop: (0, 0, width as u32, height as u32),
            picture,
            canvas: CanvasFormat::I420,
            canvas_model: Model::yuv(BT709),
        };
        if self.prepared[tile].as_ref().is_none_or(|p| p.key != key) {
            self.prepared[tile] = Some(Prepared::new(key));
        }
        let prepared = self.prepared[tile].as_ref().unwrap();
        let plan = Plan {
            background: Vec::new(),
            items: vec![Item::Layer {
                index: 0,
                picture,
                visible: vec![picture],
                opacity: 255,
            }],
        };
        let (w, h) = (self.layout.width, self.layout.height);
        let [y, u, v] = &mut self.canvas;
        // SAFETY: the planes outlive the canvas, which is dropped below.
        let canvas = unsafe {
            Canvas::new(
                CanvasFormat::I420,
                w,
                h,
                Model::yuv(BT709),
                [(y.as_mut_ptr(), y.len()), (u.as_mut_ptr(), u.len()), (v.as_mut_ptr(), v.len())],
                [w, w / 2, w / 2],
            )
        };
        let frame = Frame {
            canvas: &canvas,
            plan: &plan,
            layers: vec![Some(LayerInput {
                source: SourceFrame {
                    format: SrcFormat::V210,
                    width,
                    height,
                    planes: [data, &[], &[]],
                    strides: [stride_read, 0, 0],
                    model: Model::yuv(BT709),
                },
                prepared,
                cache: CacheMode::Off,
            })],
            background: Some(argb_to(Model::yuv(BT709), BLACK)),
        };
        render(&frame, None);
    }

    fn black(&mut self, tile: usize) {
        let t = self.layout.tiles[tile].clone();
        let c = argb_to(Model::yuv(BT709), BLACK);
        let w = self.layout.width;
        for yy in t.y..t.y + t.h {
            let r = yy as usize * w;
            self.canvas[0][r + t.x as usize..r + (t.x + t.w) as usize].fill(c[0]);
        }
        for yy in t.y / 2..(t.y + t.h) / 2 {
            let r = yy as usize * w / 2;
            for p in 1..3 {
                self.canvas[p][r + t.x as usize / 2..r + (t.x + t.w) as usize / 2].fill(c[p]);
            }
        }
    }

    fn start(&mut self) -> Instant {
        Instant::now()
    }

    /// The programme of this output frame (a v210 frame of the house size).
    pub fn offer_program(&mut self, frame: &[u8]) {
        if self.take {
            let t = self.start();
            self.draw(1, frame, self.width, self.height);
            self.draw_ms += t.elapsed().as_secs_f64() * 1e3;
        }
    }

    pub fn offer_preview(&mut self, frame: &[u8]) {
        if self.take {
            let t = self.start();
            self.draw(0, frame, self.width, self.height);
            self.draw_ms += t.elapsed().as_secs_f64() * 1e3;
        }
    }

    pub fn wants_sources(&self) -> bool {
        self.take
    }

    /// The sources' current frames (by source index; `None`: black).
    pub fn offer_sources(&mut self, pics: &[Option<(&[u8], usize, usize)>]) {
        let t = self.start();
        let n = (self.layout.tiles.len() - 2).min(pics.len());
        for _ in 0..self.per_frame.min(n) {
            let i = self.next_source % n.max(1);
            self.next_source = self.next_source.wrapping_add(1);
            match pics[i] {
                Some((data, w, h)) => self.draw(2 + i, data, w, h),
                None => self.black(2 + i),
            }
        }
        self.draw_ms += t.elapsed().as_secs_f64() * 1e3;
    }

    /// End of an output frame: send the mosaic if this one was taken.
    pub fn frame_done(&mut self) {
        if self.take {
            let mut buf = Vec::with_capacity(self.canvas.iter().map(|p| p.len()).sum());
            for p in &self.canvas {
                buf.extend_from_slice(p);
            }
            match self.tx.try_send(buf) {
                Ok(()) => self.drawn += 1,
                Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => self.dropped += 1,
            }
        }
        self.count += 1;
        self.take = self.count % self.every == 0;
    }
}

/// The encoder: an ffmpeg child fed from a channel, restarted when it dies.
pub fn spawn_encoder(
    ffmpeg: String,
    url: String,
    width: usize,
    height: usize,
    fps: u32,
    kbps: u32,
    rx: Receiver<Vec<u8>>,
) -> std::thread::JoinHandle<()> {
    std::thread::Builder::new()
        .name("preview-encoder".into())
        .spawn(move || {
            let mut child: Option<Child> = None;
            let mut backoff = Duration::from_secs(1);
            let mut next_try = Instant::now();
            while let Ok(frame) = rx.recv() {
                if child.is_none() {
                    if Instant::now() < next_try {
                        continue;
                    }
                    match start_ffmpeg(&ffmpeg, &url, width, height, fps, kbps) {
                        Ok(c) => {
                            tracing::info!("preview: encoding {width}x{height}@{fps} to {url}");
                            child = Some(c);
                            backoff = Duration::from_secs(1);
                        }
                        Err(e) => {
                            tracing::warn!("preview: cannot start {ffmpeg}: {e}");
                            next_try = Instant::now() + backoff;
                            backoff = (backoff * 2).min(Duration::from_secs(30));
                            continue;
                        }
                    }
                }
                let ok = child
                    .as_mut()
                    .and_then(|c| c.stdin.as_mut())
                    .is_some_and(|stdin| stdin.write_all(&frame).is_ok());
                if !ok {
                    if let Some(mut c) = child.take() {
                        let _ = c.kill();
                        let _ = c.wait();
                    }
                    tracing::warn!("preview: the encoder stopped; restarting in {backoff:?}");
                    next_try = Instant::now() + backoff;
                    backoff = (backoff * 2).min(Duration::from_secs(30));
                }
            }
            if let Some(mut c) = child {
                let _ = c.kill();
                let _ = c.wait();
            }
        })
        .expect("preview thread")
}

fn start_ffmpeg(ffmpeg: &str, url: &str, w: usize, h: usize, fps: u32, kbps: u32) -> std::io::Result<Child> {
    let gop = (fps * 2).to_string();
    Command::new(ffmpeg)
        .args(["-hide_banner", "-loglevel", "error", "-nostats"])
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", &format!("{w}x{h}"), "-r", &fps.to_string()])
        .args(["-i", "pipe:0", "-an"])
        .args(["-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency", "-profile:v", "baseline"])
        .args(["-g", &gop, "-keyint_min", &gop, "-bf", "0", "-threads", "2"])
        .args(["-b:v", &format!("{kbps}k"), "-maxrate", &format!("{kbps}k"), "-bufsize", &format!("{}k", kbps / 2)])
        .args(["-f", "rtsp", "-rtsp_transport", "tcp", url])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .spawn()
}

/// A channel for the tap and the encoder.
pub fn channel() -> (SyncSender<Vec<u8>>, Receiver<Vec<u8>>) {
    sync_channel(2)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A flat grey source in every tile (decimated for the 16-source
    /// layout's thumbnails): the mosaic's luma is the grey's.
    #[test]
    fn mosaic_tiles_show_their_sources() {
        let (w, h) = (1920usize, 1080usize);
        let mut frame = vec![0u8; v210::frame_bytes(w, h, false)];
        let stride = v210::line_bytes(w);
        // Y 502 (8-bit 126), Cb Cr 512.
        let row: Vec<u16> = (0..2 * w).map(|s| if s % 2 == 1 { 502 } else { 512 }).collect();
        for y in 0..h {
            v210::pack(&mut frame[y * stride..(y + 1) * stride], 0, &row);
        }
        for n in [8usize, 16] {
            let ids: Vec<String> = (0..n).map(|i| format!("in-{i}")).collect();
            let layout = layout(&ids);
            let (tx, rx) = channel();
            let mut tap = Tap::new(layout.clone(), 1.0, 1, w, h, tx);
            tap.frame_done();
            tap.offer_preview(&frame);
            tap.offer_program(&frame);
            let pics: Vec<Option<(&[u8], usize, usize)>> = (0..n).map(|_| Some((&frame[..], w, h))).collect();
            for _ in 0..n {
                tap.offer_sources(&pics);
            }
            tap.frame_done();
            let mosaic = rx.try_recv().expect("a frame");
            for t in &layout.tiles {
                let (x, y) = ((t.x + t.w / 2) as usize, (t.y + t.h / 2) as usize);
                let v = mosaic[y * layout.width + x];
                assert!((v as i32 - 126).abs() <= 1, "{n} sources, tile {}: {v}", t.id);
            }
        }
    }

    #[test]
    fn layout_fits_the_sources() {
        for n in [1usize, 8, 10, 18, 24, 32] {
            let ids: Vec<String> = (0..n).map(|i| format!("in-{i}")).collect();
            let l = layout(&ids);
            assert_eq!(l.tiles.len(), 2 + n.min(32), "{n}");
            for t in &l.tiles {
                assert!(t.x % 2 == 0 && t.y % 2 == 0 && t.w % 2 == 0 && t.h % 2 == 0);
                assert!(t.x + t.w <= 1280 && t.y + t.h <= 720, "{n} {t:?}");
            }
        }
    }
}

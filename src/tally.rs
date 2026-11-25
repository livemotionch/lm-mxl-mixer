//! Tally over TSL UMD 5.0 (UDP), as the other mxl-stack functions speak it:
//! one display message per source (index = source number − 1, in the order
//! of the source list), red on the programme of ME 1, green on its preview,
//! the source's label as text. Sent on every change and once a second.

use std::net::UdpSocket;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::engine::Shared;

/// TSL 5.0 tally colours (2 bits): off, red, green, amber.
fn colour(program: bool, preview: bool) -> u16 {
    match (program, preview) {
        (true, false) => 1,
        (false, true) => 2,
        (true, true) => 1,
        (false, false) => 0,
    }
}

/// One TSL 5.0 packet with a display message per entry
/// (`(index, program, preview, text)`).
pub fn packet(screen: u16, entries: &[(u16, bool, bool, String)]) -> Vec<u8> {
    let mut body = Vec::new();
    body.push(0u8); // version
    body.push(0u8); // flags: ASCII text
    body.extend_from_slice(&screen.to_le_bytes());
    for (index, program, preview, text) in entries {
        let c = colour(*program, *preview);
        // Bits 0–1 RH tally, 2–3 text tally, 4–5 LH tally, 6–7 brightness (3).
        let control: u16 = c | (c << 2) | (c << 4) | (3 << 6);
        let text: Vec<u8> = text.bytes().filter(|b| b.is_ascii()).take(64).collect();
        body.extend_from_slice(&index.to_le_bytes());
        body.extend_from_slice(&control.to_le_bytes());
        body.extend_from_slice(&(text.len() as u16).to_le_bytes());
        body.extend_from_slice(&text);
    }
    let mut out = (body.len() as u16).to_le_bytes().to_vec();
    out.extend(body);
    out
}

/// Send tally to the configured targets until the process ends.
pub fn spawn(shared: Arc<Shared>, screen: u16) {
    std::thread::Builder::new()
        .name("tsl".into())
        .spawn(move || {
            let Ok(socket) = UdpSocket::bind("0.0.0.0:0") else {
                tracing::error!("TSL: cannot open a UDP socket");
                return;
            };
            let mut last: Option<(Vec<(u16, bool, bool, String)>, Vec<String>)> = None;
            let mut sent = Instant::now();
            while !shared.stop.load(std::sync::atomic::Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(20));
                let (entries, targets) = {
                    let c = shared.control();
                    let e: Vec<(u16, bool, bool, String)> = c
                        .sources
                        .iter()
                        .zip(&c.tally)
                        .enumerate()
                        .map(|(i, (s, t))| (i as u16, t.program, t.preview, s.label.clone()))
                        .collect();
                    (e, c.tally_targets.clone())
                };
                let now = (entries.clone(), targets.clone());
                if last.as_ref() == Some(&now) && sent.elapsed() < Duration::from_secs(1) {
                    continue;
                }
                if targets.is_empty() {
                    last = Some(now);
                    continue;
                }
                let p = packet(screen, &entries);
                for t in &targets {
                    if let Err(e) = socket.send_to(&p, t.as_str()) {
                        tracing::debug!("TSL to {t}: {e}");
                    }
                }
                last = Some(now);
                sent = Instant::now();
            }
        })
        .expect("tsl thread");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tsl5_packet() {
        let p = packet(0, &[(0, true, false, "Cam 1".into()), (1, false, true, "Cam 2".into())]);
        assert_eq!(u16::from_le_bytes([p[0], p[1]]) as usize, p.len() - 2);
        assert_eq!(&p[2..6], &[0, 0, 0, 0]);
        // First message: index 0, red everywhere, full brightness, "Cam 1".
        assert_eq!(&p[6..8], &[0, 0]);
        assert_eq!(u16::from_le_bytes([p[8], p[9]]), 1 | 1 << 2 | 1 << 4 | 3 << 6);
        assert_eq!(u16::from_le_bytes([p[10], p[11]]), 5);
        assert_eq!(&p[12..17], b"Cam 1");
        assert_eq!(u16::from_le_bytes([p[19], p[20]]), 2 | 2 << 2 | 2 << 4 | 3 << 6);
    }
}

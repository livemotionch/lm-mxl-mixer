# lm-mxl-mixer specification

What the mixer does, exactly. The README has the overview, the settings and the
measurements.

## 1. Media

### 1.1 House format

`MIXER_FORMAT`: `720p`, `1080p` or `2160p` at 23.98, 24, 25, 29.97, 30, 50,
59.94 or 60 (default `1080p50`). Progressive only. Every output has the house
format. Inputs may have other sizes and rates (§1.3).

### 1.2 Outputs

All outputs are v210 flows in the mixer's own MXL domain
(`MXL_OUTPUT_DOMAIN_DIR`), each with an NMOS sender that is always active:

| Key | Label (`MIXER_LABEL_PREFIX` = `VM`) | What |
| --- | --- | --- |
| `pgm` | `VM PGM Video` | Programme of ME 1: background, upstream keyers, DSKs, FTB |
| `pvw` | `VM PVW Video` | Preview of ME 1: the preview bus source |
| `clean` | `VM Clean Video` | Clean feed of ME 1: background and upstream keyers, no DSK, no FTB (`MIXER_CLEAN_FEED`) |
| `me<n>-pgm`, `me<n>-pvw` | `VM ME<n> PGM Video`, … | Programme and preview of ME n ≥ 2 (`MIXER_ME_OUTPUTS`) |
| `pgm-audio` | `VM PGM Audio` | Programme audio, float32 48 kHz, `MIXER_AUDIO_CHANNELS` channels (unless `MIXER_AUDIO=off`) |

Flow ids are UUIDv5 of `NMOS_SEED` and the key, so they stay across restarts.

### 1.3 Inputs

- `MIXER_INPUTS` video receivers (`in-1` …, `VM In n Video`) and
  `MIXER_KEY_INPUTS` key receivers (`key-1` …, `VM Key n Video`). Both accept
  `video/v210` and `video/v210a`; a v210a source can drive a linear key, a v210
  source on a key receiver is opaque for a linear key (or used as a luma key).
- With `MIXER_AUDIO=afv` an audio receiver per video input (`VM In n Audio`);
  with `passthrough` one (`VM Audio In`).
- An IS-05 activation (single, bulk, scheduled, restored) is accepted at once and
  never waits for media: a flow that doesn't exist yet is waited for.
- Inputs of another size are scaled to the house size; another rate is read at
  the input grain whose time is the output grain's time (a 25p source repeats
  every grain into 50p).
- Interlaced flows and flows that aren't v210/v210a are refused (state
  `unsupported`).

### 1.4 Timing

- The engine writes output grain `N` at its TAI time `t(N)`: it wakes at
  `t(N)`, draws every output and commits. Measured on the test VM: a PGM grain
  is complete 0.9–4.4 ms (median, depending on the scene) after its TAI time.
- For grain `N` it draws the input grains at the time of output grain
  `N − MIXER_READ_OFFSET_GRAINS` (default 1: one frame of latency; the inputs
  are always complete by then). With 0 it waits up to `MIXER_INPUT_WAIT_US` for
  the grain of the same time.
- **Late input**: the newest complete grain is drawn (counted as `late`).
  **No new grain**: the last one is held (`held`), for up to
  `MIXER_HOLD_MAX_S` (default 10 s), also when the writer stopped or its flow
  went away (the reader of the held grain stays open until a new grain comes).
  After that, or with nothing ever read: black (`missed`). A flow re-created by
  its writer (new inode) is opened again.
- The output never waits for an input. A frame the engine could not start in
  time (more than a grain late) is skipped (`lm_mixer_frames_missed_total`).

## 2. Mixing

### 2.1 MEs, buses, sources

`MIXER_MES` (1–4) MEs. Each has a programme and a preview bus. The sources are
the inputs (`in-n`), the key inputs (`key-n`), `black`, and the programmes of
higher MEs (`me-n`): ME m can take ME n for n > m (re-entry; drawn the same
frame, the highest ME first).

### 2.2 Transitions

Kinds: `cut`, `mix`, `dip` (through `dip_color`), `wipe`, `push`, `slide`;
`duration_ms`; `direction` for push and slide; for wipes `pattern`
(`horizontal`, `vertical`, `barn-door-vertical`, `barn-door-horizontal`, `box`,
`iris`, `diagonal`), `soft` (edge width, a fraction of the travel) and
`reverse`.

- **Cut** swaps programme and preview now.
- **Auto** runs the transition over whole frames: the k-th frame after the
  command shows progress k/frames, the last frame exactly the end; then the
  buses swap. An auto of kind `cut` (or 0 ms) cuts.
- **T-bar** (`position` 0–1): away from its rest end it holds the transition at
  that progress; at the other end the transition completes and that end becomes
  the rest (ping-pong); back at rest it is cancelled. Auto from a held position
  runs the rest in the remaining share of the duration.
- The preview bus is locked while a transition runs.
- **Next transition**: `background` (the buses swap) and `ties` (upstream
  keyers taken along: on → off, off → on, faded or wiped with the transition;
  cut with a cut). Background off is a key-only transition; with nothing in it,
  cut and auto are refused.
- **FTB** fades the whole programme to black (over `ftb.rate_ms`) and back;
  the clean feed stays.

Transitions are computed per output frame inside the engine (`vision::me`),
from TAI time counted in whole frames, so they are frame accurate and jitter of
the wake-up doesn't change them.

### 2.3 Keyers

Upstream keyers (`MIXER_KEYERS_PER_ME`, 0–4, ids `me-<m>-key-<k>`) are over the
background, under the DSKs. Downstream keyers (`MIXER_DSKS`, 0–4,
`me-<m>-dsk-<k>`) are over the upstream keyers, under FTB. Each has:

- `mode`: `luma` (a self key: transparent below `clip`, opaque `soft` above it,
  fractions of the video range), `linear` (the key plane of a v210a source,
  video range 64 → transparent, 940 → opaque; `clip` lifts the start, `gain`
  steepens it), `pip` (upstream only: the source opaque in a box); `invert`;
- `source`: a source id;
- `dme` (upstream only; always used by `pip`): `enabled`, the box `x`, `y`, `w`,
  `h` (canvas fractions), `crop` (`left`, `right`, `top`, `bottom`, source
  fractions) and an outside `border` (`width` px, `color`);
- `rate_ms`: the time of AUTO; ON/OFF (cut) is immediate.

Keys are straight (not premultiplied), as MXL's v210a defines them; a
chroma sample takes the mean level of its two pixels.

### 2.4 Pixels

The 10-bit renderer (`src/compositor/render10.rs`):

- Layers are placed on even pixels horizontally (4:2:2 chroma lines up).
- Scaling: tables (area filter for 2:1 and more, bilinear above),
  fractional weights in 1/1024, vertical then horizontal, chroma at chroma
  resolution, exact boxes for whole-number ratios. Vertical downscales of
  7.5:1 or more read every n-th source row (n = ⌊(ratio + 0.5) / 4⌋, the
  middle row of each block);
  `MIXER_DECIMATE=0` reads every row.
- Blends: `d + (s − d)·a` per sample in Q15, rounded; within ±1 of a float
  reference for unscaled layers, ±2 scaled (tests `tests10.rs`).
- Fills (background, dips, borders, FTB) are BT.709 video-range colours.

### 2.5 Tally

Per source: `program` (on the programme of ME 1, through any re-entered ME, as
background, during a transition, as a keyer's source while it is up) and
`preview` (the preview bus of ME 1). `GET /api/v1/tally`, `lm_mixer_tally`,
the WebSocket state, and TSL UMD 5.0 over UDP to `TSL_TARGETS` /
`PUT /api/v1/tally/receivers`: one display message per source (index = source
number − 1 in the source list), RH, text and LH tally red (programme) or green
(preview), brightness 3, the source's label as text; sent on change and once a
second.

### 2.6 Audio

`MIXER_AUDIO=afv`: each input's audio weighted by how much of it the programme
of ME 1 carries (1 on air, equal-power curves through transitions, a dip through silence, × (1 − FTB),
through re-entered MEs), the weights computed at both ends of each frame and
ramped over its samples. `passthrough`: the one audio input copied untouched.
Channel `c` of the output is channel `c` of each input; missing channels are
silent. The samples of output frame `N` are `[t(N), t(N+1))`, read from the
inputs one read offset earlier (the video's latency). An input without samples
in time is silent for that frame (`lm_mixer_audio_input_gaps_total`).

## 3. Control

### 3.1 REST API (`/api/v1`, port `WEB_PORT`)

Bodies are JSON. ME by `me` (1-based) or `panel_id` (`me-1`); ME 1 by default.
Every command answers the whole state (as `GET /state`) or `{error}` with 400
(invalid), 404 (no such source/ME/keyer) or 409 (refused now, e.g. the preview
during a transition).

| Method, path | Body | What |
| --- | --- | --- |
| `GET /state` (`/console`) | | Sources with tally, MEs (buses, transition, T-bar, keyers, DSKs, FTB), engine numbers, inputs, preview, audio |
| `GET /api/v1/ws` (WebSocket) | | The state on every change, at least once a second |
| `POST /mixer/preview` | `{me, source}` | Preview bus (`source`: an id or `black`) |
| `POST /mixer/program` | `{me, source}` | Programme bus (a cut from the programme bus) |
| `POST /mixer/take` | `{me, source, transition?, duration_ms?}` | FlowXer's take: without `transition` (or `cut`) like `program`; else source to preview and that transition |
| `POST /mixer/cut`, `/mixer/auto` | `{me}` | Cut, auto |
| `POST /mixer/position` | `{me, position}` | T-bar 0–1 |
| `POST /mixer/fade-to-black` | `{me, on?, cut?}` | FTB on/off (toggle without `on`), faded or cut |
| `PATCH /panels/{me-n}/transition` | partial `{type, duration_ms, direction, wipe: {pattern, soft, reverse}, dip_color}` | Next transition |
| `PATCH /panels/{me-n}/next` | `{background?, ties?}` | Elements of the next transition |
| `PATCH /panels/{me-n}/ftb` | `{rate_ms}` | FTB time |
| `GET /keyers` | | All keyers |
| `PATCH /keyers/{id}` | partial keyer settings (§2.3), `enabled` (FlowXer: on/off) | Keyer settings |
| `POST /keyers/{id}/auto`, `/keyers/{id}/cut` | `{on?}` | Fade or cut on/off (toggle without `on`) |
| `PATCH /sources/{id}` | `{label}` | Button and TSL label |
| `GET /tally`, `PUT /tally/receivers` | `{targets: ["host:port"]}` | Tally, TSL receivers |
| `GET /inputs`, `/outputs`, `/engine` | | Reader states and counters, output counters and mix times |
| `GET /preview` | | WHEP/HLS URLs of the mosaic and where each picture is in it |
| `GET /config`, `/config/export`, `POST /config/import` | | Settings in force; the operator state as one JSON document |
| `GET /health` | | Status, version, frames, NMOS registration, TAI offset |

`/livez` (process), `/readyz` (engine running and, with a registry, registered)
and `/metrics` (Prometheus) are at the root. The UI is at `/`.

### 3.2 Saved state

`MIXER_STATE_DIR/state.json` (written within a second of a change, and at
shutdown): source labels, per ME the buses, transition, ties, FTB time and the
keyer settings, the TSL receivers. Keyers come back off air.
`connections.json`: the IS-05 connections, re-applied at start.

### 3.3 NMOS

mxl-nmos (IS-04 v1.3 Node API, IS-05 v1.2, BCP-007-03) on `NMOS_PORT`: one
Device, the receivers of §1.3, a Source, Flow and Sender per output. Settings
from `NMOS_*` (README). Shutdown deletes every resource, the Node last.

## 4. Process

- One container, host network, uid 1000, the MXL root mounted at
  `/Volumes/mxl`.
- Exit codes: 0, 75 (a port can't be bound, libmxl can't be loaded), 78
  (invalid configuration), 143 (SIGTERM).
- Start: configuration, TAI offset check (logged as an error when the kernel
  offset isn't 37 s), scan path, state dir, libmxl, ports, output domain
  (created once, never overwritten), saved state, output flows, NMOS node,
  saved connections, web, engine, registration.
- Shutdown (SIGTERM, within `SHUTDOWN_TIMEOUT_S`): engine stopped (writers and
  readers released), NMOS resources deleted, state saved, own output domain
  removed with `MXL_CLEANUP_ON_EXIT`.

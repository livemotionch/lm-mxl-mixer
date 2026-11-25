#!/usr/bin/env python3
"""CPU and timing of a running lm-mxl-mixer over a window.

    python3 tests/measure.py [seconds] [web url]

Samples /proc for the mixer process (`lm-mxl-mixer`) and its ffmpeg preview
encoder, and diffs the mixer's /metrics counters (frames, missed, late, mix time
per output). CPU is in % of one core. Run it on the mixer's host.
"""
import json, os, re, sys, time, urllib.request

SECS = float(sys.argv[1]) if len(sys.argv) > 1 else 20
WEB = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8180"
TICK = os.sysconf("SC_CLK_TCK")


def pids(name):
    out = []
    for p in os.listdir("/proc"):
        if p.isdigit():
            try:
                comm = open(f"/proc/{p}/comm").read().strip()
                if comm == name:
                    out.append(int(p))
            except OSError:
                pass
    return out


def cpu(pid):
    try:
        f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
        return (int(f[11]) + int(f[12])) / TICK
    except OSError:
        return 0.0


def threads(pid):
    out = {}
    for t in os.listdir(f"/proc/{pid}/task"):
        try:
            name = open(f"/proc/{pid}/task/{t}/comm").read().strip()
            f = open(f"/proc/{pid}/task/{t}/stat").read().rsplit(")", 1)[1].split()
            out.setdefault(re.sub(r"[-_]?\d+$", "", name), 0.0)
            out[re.sub(r"[-_]?\d+$", "", name)] += (int(f[11]) + int(f[12])) / TICK
        except OSError:
            pass
    return out


def metrics():
    text = urllib.request.urlopen(f"{WEB}/metrics", timeout=5).read().decode()
    m = {}
    for line in text.splitlines():
        if line and not line.startswith("#"):
            k, v = line.rsplit(" ", 1)
            m[k] = float(v)
    return m


mixer = pids("lm-mxl-mixer")
ffmpeg = pids("ffmpeg")
if not mixer:
    sys.exit("no lm-mxl-mixer process")
m0, c0, t0 = metrics(), {p: cpu(p) for p in mixer + ffmpeg}, threads(mixer[0])
w0 = time.time()
time.sleep(SECS)
dt = time.time() - w0
m1, c1, t1 = metrics(), {p: cpu(p) for p in mixer + ffmpeg}, threads(mixer[0])
d = lambda k: m1.get(k, 0) - m0.get(k, 0)
res = {
    "seconds": round(dt, 1),
    "mixer_cpu_pct": round(sum(c1[p] - c0[p] for p in mixer) / dt * 100, 1),
    "preview_ffmpeg_cpu_pct": round(sum(c1[p] - c0[p] for p in ffmpeg) / dt * 100, 1),
    "mixer_threads_cpu_pct": {k: round((t1.get(k, 0) - t0.get(k, 0)) / dt * 100, 1) for k in t1 if t1.get(k, 0) - t0.get(k, 0) > 0.01 * dt},
    "frames": int(d("lm_mixer_frames_total")),
    "fps": round(d("lm_mixer_frames_total") / dt, 2),
    "missed": int(d("lm_mixer_frames_missed_total")),
    "late": int(d("lm_mixer_frames_late_total")),
    "frame_ms_avg": round(d("lm_mixer_frame_seconds_sum") / max(1, d("lm_mixer_frame_seconds_count")) * 1e3, 3),
    "mix_ms_avg": {},
    "input_late": int(sum(v for k, v in m1.items() if k.startswith("lm_mixer_input_frames_late_total")) - sum(v for k, v in m0.items() if k.startswith("lm_mixer_input_frames_late_total"))),
    "input_missed": int(sum(v for k, v in m1.items() if k.startswith("lm_mixer_input_frames_missed_total")) - sum(v for k, v in m0.items() if k.startswith("lm_mixer_input_frames_missed_total"))),
}
for k in m1:
    g = re.match(r'lm_mixer_mix_seconds_sum\{output="(.+)"\}', k)
    if g:
        o = g.group(1)
        n = d(f'lm_mixer_mix_seconds_count{{output="{o}"}}')
        res["mix_ms_avg"][o] = round(d(k) / max(1, n) * 1e3, 3)
print(json.dumps(res, indent=1))

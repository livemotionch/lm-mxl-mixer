#!/usr/bin/env python3
"""CPU of a running mixer with 4, 8 and 16 routed inputs, per scene.

    python3 tests/bench_live.py <pattern domain dir> [seconds per run]

Routes the mixer's inputs (its own IS-05 API on 127.0.0.1:3332) to the test
player's two outputs, the browser source and the `mxl-pattern` flows of the given
domain (registry Query API on 127.0.0.1:3211 for the stack's senders), then
measures with tests/measure.py for each scene:

  cut   PGM in-1, PVW in-2, nothing else
  show  a mix every 2 s between in-1 and in-2, two PiP boxes (in-3, in-4),
        a luma DSK (the browser source on key-1)
  max   a soft iris wipe every 2 s, four PiP boxes (the last four inputs),
        the DSK

The mixer needs MIXER_INPUTS=16 and MIXER_KEYERS_PER_ME=4.
"""
import json, os, subprocess, sys, threading, time, urllib.request

PATTERN = sys.argv[1]
SECS = sys.argv[2] if len(sys.argv) > 2 else "20"
API, NODE, QUERY = "http://127.0.0.1:8180/api/v1", "http://127.0.0.1:3332", "http://127.0.0.1:3211/x-nmos/query/v1.3"
HERE = os.path.dirname(os.path.abspath(__file__))


def call(method, url, body=None):
    req = urllib.request.Request(url, None if body is None else json.dumps(body).encode(),
                                 {"content-type": "application/json"}, method=method)
    with urllib.request.urlopen(req, timeout=5) as r:
        raw = r.read()
        return json.loads(raw) if raw[:1] in (b"{", b"[") else raw


def stack_flow(label):
    s = next(s for s in call("GET", f"{QUERY}/senders?paging.limit=1000") if s["label"] == label)
    dev = next(d for d in call("GET", f"{QUERY}/devices?paging.limit=1000") if d["id"] == s["device_id"])
    href = max((c["href"].rstrip("/") for c in dev["controls"] if "/x-nmos/connection/" in c["href"]),
               key=lambda h: tuple(int(x) for x in h.rsplit("/v", 1)[1].split(".")))
    tp = call("GET", f"{href}/single/senders/{s['id']}/active")["transport_params"][0]
    return tp["mxl_domain_id"], tp["mxl_flow_id"]


def pattern_flows():
    dom = json.load(open(f"{PATTERN}/domain_def.json"))["id"]
    flows = []
    for d in sorted(os.listdir(PATTERN)):
        if d.endswith(".mxl-flow"):
            label = json.load(open(f"{PATTERN}/{d}/flow_def.json"))["label"]
            flows.append((label, dom, d[:-len(".mxl-flow")]))
    flows.sort(key=lambda f: int(f[0].rsplit(" ", 1)[1]))
    return [(dom, f) for _, dom, f in flows]


receivers = {r["label"]: r["id"] for r in call("GET", f"{NODE}/x-nmos/node/v1.3/receivers")}


def route(label, target):
    body = {"master_enable": target is not None, "activation": {"mode": "activate_immediate"}}
    if target:
        body["transport_params"] = [{"mxl_domain_id": target[0], "mxl_flow_id": target[1]}]
    call("PATCH", f"{NODE}/x-nmos/connection/v1.2/single/receivers/{receivers[label]}/staged", body)


sources = [stack_flow("Out 1 Video"), stack_flow("Out 2 Video"), stack_flow("Browser Video")] + pattern_flows()
route("VM Key 1 Video", sources[2])
cfg = call("GET", f"{API}/config")
assert cfg["inputs"] >= 16 and cfg["keyers_per_me"] >= 4, "MIXER_INPUTS=16 and MIXER_KEYERS_PER_ME=4 needed"

stop = threading.Event()


def autos(kind):
    while not stop.wait(2.0):
        try:
            call("POST", f"{API}/mixer/auto", {"me": 1})
        except Exception:
            pass


def keyers(on, boxes):
    for k in range(4):
        call("POST", f"{API}/keyers/me-1-key-{k + 1}/cut", {"on": False})
    for k, src in enumerate(boxes):
        x, y = [(0.62, 0.04), (0.03, 0.04), (0.62, 0.6), (0.03, 0.6)][k]
        call("PATCH", f"{API}/keyers/me-1-key-{k + 1}", {"mode": "pip", "source": src,
                                                          "dme": {"x": x, "y": y, "w": 0.35, "h": 0.35, "border": {"width": 4}}})
        call("POST", f"{API}/keyers/me-1-key-{k + 1}/cut", {"on": True})
    call("PATCH", f"{API}/keyers/me-1-dsk-1", {"mode": "luma", "source": "key-1", "clip": 0.3, "soft": 0.2})
    call("POST", f"{API}/keyers/me-1-dsk-1/cut", {"on": on})


results = []
for n in (4, 8, 16):
    for i in range(16):
        route(f"VM In {i + 1} Video", sources[i] if i < n else None)
    time.sleep(2)
    for scene in ("cut", "show", "max"):
        call("POST", f"{API}/mixer/program", {"me": 1, "source": "in-1"})
        call("POST", f"{API}/mixer/preview", {"me": 1, "source": "in-2"})
        if scene == "cut":
            keyers(False, [])
            call("PATCH", f"{API}/panels/me-1/transition", {"type": "mix", "duration_ms": 1000})
        elif scene == "show":
            keyers(True, ["in-3", "in-4"])
            call("PATCH", f"{API}/panels/me-1/transition", {"type": "mix", "duration_ms": 1000})
        else:
            keyers(True, [f"in-{n - 3 + k}" for k in range(4)])
            call("PATCH", f"{API}/panels/me-1/transition",
                 {"type": "wipe", "duration_ms": 1000, "wipe": {"pattern": "iris", "soft": 0.1}})
        stop.clear()
        t = None
        if scene != "cut":
            t = threading.Thread(target=autos, args=(scene,), daemon=True)
            t.start()
        time.sleep(1)
        out = json.loads(subprocess.run([sys.executable, f"{HERE}/measure.py", SECS], capture_output=True, text=True).stdout)
        stop.set()
        if t:
            t.join()
        out.update({"inputs": n, "scene": scene})
        results.append(out)
        print(f"{n:>2} inputs  {scene:<5} mixer {out['mixer_cpu_pct']:>5.1f}%  ffmpeg {out['preview_ffmpeg_cpu_pct']:>4.1f}%  "
              f"fps {out['fps']}  missed {out['missed']}  late {out['late']}  frame {out['frame_ms_avg']} ms  "
              f"input late/missed {out['input_late']}/{out['input_missed']}  mix ms {out['mix_ms_avg']}", flush=True)
json.dump(results, open("/tmp/bench_live.json", "w"), indent=1)
keyers(False, [])

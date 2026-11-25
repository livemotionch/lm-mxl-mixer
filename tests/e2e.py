#!/usr/bin/env python3
"""End-to-end test with real MXL flows on a tmpfs.

Starts `mxl-pattern` (3 colour sources and one v210a fill+key source) and the
mixer in a scan directory on a tmpfs, routes the mixer's receivers over its own
IS-05 API, drives it over its REST API and checks the pixels of its PGM, PVW and
clean-feed flows with `mxl-grab`:

  cut, mix (T-bar halfway), hard wipe (T-bar), auto mix to the end, DSK (linear
  key from the v210a source) on PGM but not on the clean feed, FTB, the
  programme of a second ME re-entered into ME 1, a PiP box, no missed output
  grains, metrics, NMOS resources, config export/import.

    BIN=target/release LD_LIBRARY_PATH=/opt/mxl/lib python3 tests/e2e.py [/Volumes/mxl/lmtest]

Exit status 0 when every check passes. Needs libmxl (LD_LIBRARY_PATH) and a
tmpfs (the default is a directory below /dev/shm).
"""
import json, os, shutil, signal, subprocess, sys, time, urllib.request

BIN = os.environ.get("BIN", "target/release")
ROOT = sys.argv[1] if len(sys.argv) > 1 else "/dev/shm/lm-mxl-mixer-e2e"
WEB, NMOS = int(os.environ.get("E2E_WEB_PORT", "18181")), int(os.environ.get("E2E_NMOS_PORT", "13333"))
FMT = os.environ.get("E2E_FORMAT", "720p50")
W, H = {"720p50": (1280, 720), "1080p50": (1920, 1080)}[FMT]
API = f"http://127.0.0.1:{WEB}/api/v1"
procs, failures = [], []


def call(method, url, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data, {"content-type": "application/json"}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            raw = r.read()
            return json.loads(raw) if raw[:1] in (b"{", b"[") else raw.decode()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{method} {url}: {e.code} {e.read().decode()}")


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


def ycbcr(r, g, b):
    """BT.709 video range, 10 bit (as the mixer converts fills)."""
    r, g, b = r / 255, g / 255, b / 255
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return (round(64 + 876 * y), round(512 + 896 * (b - y) / 1.8556), round(512 + 896 * (r - y) / 1.5748))


COLOURS = [0xC02020, 0x20A020, 0x2040C0, 0xC0C020]
C = [ycbcr(c >> 16, (c >> 8) & 255, c & 255) for c in COLOURS]
BLACK = (64, 512, 512)


def grab(domain, flow, *points):
    out = subprocess.run([f"{BIN}/mxl-grab", domain, flow] + [f"{x},{y}" for x, y in points],
                         capture_output=True, text=True, timeout=10).stdout.split("\n")
    vals = [tuple(int(v) for v in l.split()[1:]) for l in out[1:] if l.strip()]
    return vals


def near(a, b, tol=3):
    return a is not None and b is not None and all(abs(x - y) <= tol for x, y in zip(a, b))


def mid(a, b):
    return tuple((x + y) / 2 for x, y in zip(a, b))


def settle():
    time.sleep(0.3)


def main():
    shutil.rmtree(ROOT, ignore_errors=True)
    os.makedirs(f"{ROOT}/state")
    env = dict(os.environ)
    pat = subprocess.Popen([f"{BIN}/mxl-pattern", f"{ROOT}/pattern", "--flows", "3", "--format", FMT, "--label", "E2E",
                            "--audio"], stdout=subprocess.PIPE, text=True)
    procs.append(pat)
    key = subprocess.Popen([f"{BIN}/mxl-pattern", f"{ROOT}/keysrc", "--flows", "1", "--format", FMT, "--key",
                            "--label", "E2E Key", "--seed", "e2e-key"], stdout=subprocess.PIPE, text=True)
    procs.append(key)
    lines = [pat.stdout.readline().split() for _ in range(6)]
    flows = [l[0] for l in lines if l[-1] != "Audio"]
    audio_flows = [l[0] for l in lines if l[-1] == "Audio"]
    key_flow = key.stdout.readline().split()[0]
    env.update({
        "MXL_DOMAIN_SCAN_PATH": ROOT, "MXL_OUTPUT_DOMAIN_DIR": f"{ROOT}/mixer", "MIXER_STATE_DIR": f"{ROOT}/state",
        "MIXER_FORMAT": FMT, "MIXER_INPUTS": "4", "MIXER_KEY_INPUTS": "1", "MIXER_MES": "2",
        "MIXER_KEYERS_PER_ME": "2", "MIXER_DSKS": "1", "NMOS_SEED": "e2e-mixer",
        "WEB_PORT": str(WEB), "NMOS_PORT": str(NMOS), "MIXER_THREADS": "2", "LOG_LEVEL": "info",
    })
    for k in ["NMOS_REGISTRY_ADDRESS", "NMOS_REGISTRY_PORT", "PREVIEW_PUBLISH_URL"]:
        env.pop(k, None)
    mixer = subprocess.Popen([f"{BIN}/lm-mxl-mixer"], env=env, stderr=open(f"{ROOT}/mixer.log", "w"))
    procs.append(mixer)
    for _ in range(50):
        try:
            call("GET", f"{API}/health")
            break
        except Exception:
            time.sleep(0.1)
    health = call("GET", f"{API}/health")
    check("mixer up", health["format"] == FMT, health["format"])

    # NMOS: receivers and senders of the node; route over IS-05.
    node = f"http://127.0.0.1:{NMOS}"
    receivers = {r["label"]: r for r in call("GET", f"{node}/x-nmos/node/v1.3/receivers")}
    senders = {s["label"]: s for s in call("GET", f"{node}/x-nmos/node/v1.3/senders")}
    check("NMOS receivers", {"VM In 1 Video", "VM In 4 Video", "VM Key 1 Video"} <= set(receivers), sorted(receivers))
    check("NMOS senders", {"VM PGM Video", "VM PVW Video", "VM Clean Video", "VM ME2 PGM Video"} <= set(senders),
          sorted(senders))
    pattern_domain = json.load(open(f"{ROOT}/pattern/domain_def.json"))["id"]
    key_domain = json.load(open(f"{ROOT}/keysrc/domain_def.json"))["id"]

    def route(label, domain, flow):
        r = receivers[label]
        call("PATCH", f"{node}/x-nmos/connection/v1.2/single/receivers/{r['id']}/staged", {
            "master_enable": True, "activation": {"mode": "activate_immediate"},
            "transport_params": [{"mxl_domain_id": domain, "mxl_flow_id": flow}]})

    for i in range(3):
        route(f"VM In {i + 1} Video", pattern_domain, flows[i])
    route("VM Key 1 Video", key_domain, key_flow)
    for i in range(3):
        route(f"VM In {i + 1} Audio", pattern_domain, audio_flows[i])
    out = {}
    for key_, label in [("pgm", "VM PGM Video"), ("pvw", "VM PVW Video"), ("clean", "VM Clean Video"),
                        ("audio", "VM PGM Audio")]:
        tp = call("GET", f"{node}/x-nmos/connection/v1.2/single/senders/{senders[label]['id']}/active")["transport_params"][0]
        out[key_] = tp["mxl_flow_id"]
    mixer_dir = f"{ROOT}/mixer"
    time.sleep(1.5)

    c = (W // 2, H // 2)
    left, right = (W // 8, H // 2), (W * 7 // 8, H // 2)
    pgm = lambda *p: grab(mixer_dir, out["pgm"], *p)
    pvw = lambda *p: grab(mixer_dir, out["pvw"], *p)
    clean = lambda *p: grab(mixer_dir, out["clean"], *p)

    def level():
        out_ = subprocess.run([f"{BIN}/mxl-grab", mixer_dir, out["audio"], "--audio", "20"], capture_output=True,
                              text=True, timeout=10).stdout.split()
        return [float(v) for v in out_[out_.index("mean") + 1:out_.index("rms_dbfs")]] if "mean" in out_ else None

    check("PGM shows input 1", near(pgm(c)[0], C[0]), f"{pgm(c)} vs {C[0]}")
    lv = level()
    check("PGM audio is input 1's", lv is not None and all(abs(v - 0.1) < 1e-3 for v in lv), str(lv))
    check("PVW shows input 2", near(pvw(c)[0], C[1]), f"{pvw(c)} vs {C[1]}")

    call("POST", f"{API}/mixer/cut", {"me": 1}); settle()
    check("cut: PGM input 2, PVW input 1", near(pgm(c)[0], C[1]) and near(pvw(c)[0], C[0]))
    lv = level()
    check("audio follows the cut", lv is not None and all(abs(v - 0.2) < 1e-3 for v in lv), str(lv))

    call("POST", f"{API}/mixer/preview", {"me": 1, "source": "in-3"})
    call("PATCH", f"{API}/panels/me-1/transition", {"type": "mix", "duration_ms": 400}); settle()
    call("POST", f"{API}/mixer/position", {"me": 1, "position": 0.5}); settle()
    check("T-bar halfway mixes 50/50", near(pgm(c)[0], mid(C[1], C[2]), 2), f"{pgm(c)} vs {mid(C[1], C[2])}")
    lv = level()
    want = 0.5 ** 0.5 * (0.2 + 0.3)
    check("audio halfway: equal power", lv is not None and all(abs(v - want) < 2e-3 for v in lv), f"{lv} vs {want:.4f}")
    call("POST", f"{API}/mixer/position", {"me": 1, "position": 0.0}); settle()
    check("T-bar back cancels", near(pgm(c)[0], C[1]))

    call("PATCH", f"{API}/panels/me-1/transition", {"type": "wipe", "wipe": {"pattern": "horizontal", "soft": 0}})
    call("POST", f"{API}/mixer/position", {"me": 1, "position": 0.5}); settle()
    l, r = pgm(left, right)
    check("hard wipe halfway: left new, right old", near(l, C[2]) and near(r, C[1]), f"{l} {r}")
    call("POST", f"{API}/mixer/position", {"me": 1, "position": 0.0}); settle()

    call("PATCH", f"{API}/panels/me-1/transition", {"type": "mix", "duration_ms": 400})
    t0 = time.time()
    call("POST", f"{API}/mixer/auto", {"me": 1})
    time.sleep(0.15)
    during = pgm(c)[0]
    time.sleep(0.6)
    st = call("GET", f"{API}/state")
    check("auto mix runs and ends on input 3", near(pgm(c)[0], C[2]) and st["mes"][0]["program"] == "in-3"
          and not near(during, C[1]) and not near(during, C[2]), f"during {during}")

    # DSK: linear key from the v210a source; on PGM, not on the clean feed.
    call("PATCH", f"{API}/keyers/me-1-dsk-1", {"mode": "linear", "source": "key-1", "clip": 0, "gain": 1})
    call("POST", f"{API}/keyers/me-1-dsk-1/cut", {"on": True}); settle()
    third = (W // 2, H * 4 // 5)
    p_third, p_c = pgm(third, c)
    cl_third = clean(third)[0]
    check("DSK keyed in the lower third", near(p_third, C[0]) and near(p_c, C[2]), f"{p_third} {p_c}")
    check("clean feed without the DSK", near(cl_third, C[2]), f"{cl_third}")
    call("POST", f"{API}/keyers/me-1-dsk-1/cut", {"on": False}); settle()

    # Upstream keyer as a PiP box (input 1, top right quarter).
    call("PATCH", f"{API}/keyers/me-1-key-1", {"mode": "pip", "source": "in-1",
                                                 "dme": {"x": 0.5, "y": 0.0, "w": 0.5, "h": 0.5, "border": {"width": 0}}})
    call("POST", f"{API}/keyers/me-1-key-1/cut", {"on": True}); settle()
    box, outside = pgm((W * 3 // 4, H // 4 + H // 10), (W // 4, H * 3 // 4))
    check("PiP box shows input 1 over the programme", near(box, C[0]) and near(outside, C[2]), f"{box} {outside}")
    call("POST", f"{API}/keyers/me-1-key-1/cut", {"on": False})

    # Re-entry: ME 2 programme (input 2) into ME 1.
    call("POST", f"{API}/mixer/program", {"me": 2, "source": "in-2"})
    call("POST", f"{API}/mixer/program", {"me": 1, "source": "me-2"}); settle()
    check("ME 2 re-entered into ME 1", near(pgm(c)[0], C[1]), f"{pgm(c)}")
    tally = {t["id"]: t for t in call("GET", f"{API}/tally")}
    check("tally follows re-entry", tally["in-2"]["program"] and tally["me-2"]["program"] and not tally["in-3"]["program"])

    call("POST", f"{API}/mixer/fade-to-black", {"me": 1, "on": True, "cut": True}); settle()
    check("FTB: black", near(pgm(c)[0], BLACK, 1), f"{pgm(c)}")
    lv = level()
    check("FTB: silence", lv is not None and all(abs(v) < 1e-4 for v in lv), str(lv))
    call("POST", f"{API}/mixer/fade-to-black", {"me": 1, "on": False, "cut": True})

    # Output timing: every grain of PGM over 3 s.
    watch = subprocess.run([f"{BIN}/mxl-grab", mixer_dir, out["pgm"], "--watch", "3"], capture_output=True, text=True).stdout
    check("PGM: no missing grains in 3 s", "missing 0 " in watch, watch.strip())
    metrics = call("GET", f"http://127.0.0.1:{WEB}/metrics")
    missed = [l for l in metrics.splitlines() if l.startswith("lm_mixer_frames_missed_total")]
    check("metrics: no missed frames", missed == ["lm_mixer_frames_missed_total 0"], str(missed))
    check("metrics: mix time per output", 'lm_mixer_mix_seconds_count{output="pgm"}' in metrics)

    exported = call("GET", f"{API}/config/export")
    imported = call("POST", f"{API}/config/import", exported)
    check("config export/import", imported.get("warnings") == [], str(imported.get("warnings")))

    # Restart: the routes (IS-05) and the state come back.
    mixer.send_signal(signal.SIGTERM)
    code = mixer.wait(15)
    check("SIGTERM exits 143", code == 143, str(code))
    mixer2 = subprocess.Popen([f"{BIN}/lm-mxl-mixer"], env=env, stderr=open(f"{ROOT}/mixer2.log", "w"))
    procs.append(mixer2)
    time.sleep(2.5)
    st = call("GET", f"{API}/state")
    check("restart restores state and routes", st["mes"][0]["program"] == "me-2" and near(pgm(c)[0], C[1]),
          f"{st['mes'][0]['program']} {pgm(c)}")


if __name__ == "__main__":
    try:
        main()
    finally:
        for p in procs:
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
        for p in procs:
            try:
                p.wait(10)
            except Exception:
                p.kill()
    print(f"{len(failures)} failure(s)" if failures else "all passed")
    sys.exit(1 if failures else 0)

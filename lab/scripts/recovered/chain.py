"""Feed the band-arm chain into the warden: <= 2 doppel entries (running + queued), each job after its deps are done."""
import json, subprocess, time, sys
from pathlib import Path

P = "/home/alliedtoasters/projects/chess/bot"
W = "/home/alliedtoasters/projects/clearwood/warden/.venv/bin/warden"
AUDIT = Path.home() / ".local/state/warden/audit.jsonl"
PY = ["uv", "run", "--project", f"{P}/lab", "python", "-m"]
TRAIN = ["doppel_lab.v0e_train", "train", "--w-con", "0.1", "--kl-per-player", "--res-lr-mult", "0.3",
         "--max-minutes", "90", "--gpu-ok"]
# name: (vram, max_runtime, deps, argv or 'cpu:'-prefixed local command)
JOBS = {
    "seg1_1100": "59b0b0d91955", "seg1_1900": "912de0bbdcf7",  # already queued
}
CHAIN = [
    ("seg2_1100", 20000, 6600, ["seg1_1100"], PY + TRAIN + ["--run", "v0g_1100", "--band", "1100"]),
    ("seg2_1900", 20000, 6600, ["seg1_1900"], PY + TRAIN + ["--run", "v0g_1900", "--band", "1900"]),
    ("vec_1100", 16000, 6000, [], PY + ["doppel_lab.v0e_vec", "run", "--split", "allie1100", "--band", "1100", "--gpu-ok"]),
    ("prep_1100", 0, 0, ["vec_1100"], "cpu"),
    ("vec_1900", 16000, 6000, [], PY + ["doppel_lab.v0e_vec", "run", "--split", "allie1900", "--band", "1900", "--gpu-ok"]),
    ("prep_1900", 0, 0, ["vec_1900"], "cpu"),
    ("eval_1100", 20000, 5400, ["seg2_1100", "prep_1100"],
     PY + ["doppel_lab.allie_v0e", "eval", "--run", "v0g_1100", "--band", "1100", "--gpu-ok"]),
    ("eval_1900", 20000, 5400, ["seg2_1900", "prep_1900"],
     PY + ["doppel_lab.allie_v0e", "eval", "--run", "v0g_1900", "--band", "1900", "--gpu-ok"]),
]
done_local = set()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def ended():
    out = {}
    for line in AUDIT.read_text().splitlines()[-20000:]:
        if '"job.' not in line:
            continue
        e = json.loads(line)
        if e["event"] in ("job.done", "job.failed", "job.timeout", "job.cancelled", "job.stopped"):
            out[e.get("job_id")] = e["event"]
    return out


def active():
    s = json.loads(subprocess.run([W, "status"], capture_output=True, text=True).stdout)
    b = s["batch"]
    labs = [j["label"] for j in ([b["running"]] if b["running"] else []) + b["queued"]]
    return sum(1 for l in labs if l and l.startswith("doppel:"))


while CHAIN:
    end = ended()
    for n, jid in JOBS.items():
        if end.get(jid, "job.done") != "job.done":
            log(f"STOP: {n} ({jid}) ended with {end[jid]}")
            sys.exit(1)
    for i, (name, vram, rt, deps, cmd) in enumerate(CHAIN):
        ok = all(d in done_local or end.get(JOBS.get(d)) == "job.done" for d in deps)
        if not ok:
            continue
        if cmd == "cpu":
            band = name.split("_")[1]
            log(f"prep {band}")
            r = subprocess.run(PY + ["doppel_lab.v0e_data", "prep", "--split", f"allie{band}", "--band", band],
                               cwd=P, capture_output=True, text=True)
            log(r.stdout[-400:], r.stderr[-400:])
            if r.returncode:
                log(f"STOP: prep {band} failed")
                sys.exit(1)
            done_local.add(name)
            CHAIN.pop(i)
            break
        if active() >= 2:
            break
        r = subprocess.run([W, "run", "--vram", str(vram), "--max-runtime", str(rt), "--label", f"doppel:v0g-{name}",
                            "--cwd", P, "--"] + cmd, capture_output=True, text=True)
        log(name, r.stdout.strip(), r.stderr.strip()[-300:])
        if r.returncode:
            sys.exit(1)
        JOBS[name] = json.loads(r.stdout)["job_id"]
        CHAIN.pop(i)
        break
    else:
        time.sleep(60)
        continue
    time.sleep(5)
# wait for the evals, then summaries
while True:
    end = ended()
    if all(end.get(JOBS[n]) for n in ("eval_1100", "eval_1900")):
        break
    time.sleep(120)
for b in ("1100", "1900"):
    r = subprocess.run(PY + ["doppel_lab.allie_v0e", "summary", "--run", f"v0g_{b}", "--band", b],
                       cwd=P, capture_output=True, text=True)
    Path(f"{P}/lab/results/allie_v0e_v0g_{b}.json").write_text(r.stdout)
    log(b, r.stdout[-600:], r.stderr[-300:])
log("ALL DONE")

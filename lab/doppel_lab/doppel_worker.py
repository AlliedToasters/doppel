"""bigbox's doppel trainer: pull jobs from the api, train each through the warden, push the adapter back.

    .venv/bin/python -m doppel_lab.doppel_worker [--api https://doppelbot.org] [--once] [--interval 30]

Nothing connects to bigbox: this polls POST /api/trainer/claim with the trainer token
(~/.config/doppel/trainer.token; the api holds only its sha256). Per job, one at a time:
  1. write job.json to data/doppel_jobs/<job>/ (mode 700; the user's games, no ids)
  2. warden run --wait --vram 8000 --max-runtime 900 --label doppel:tune -- doppel_lab.doppel_train ... --device cuda
  3. on success POST the adapter file to /api/trainer/jobs/<job>/adapter, else POST .../fail with the log tail
  4. delete the job directory either way (the games are not kept on bigbox)
A 410 from the api means the doppel was deleted or re-requested while training: the adapter is discarded.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WARDEN = Path.home() / "projects/clearwood/warden/.venv/bin/warden"
TOKEN = Path.home() / ".config/doppel/trainer.token"
JOBS = ROOT / "data/doppel_jobs"
UA = "doppel-trainer"


def call(api: str, path: str, token: str, body: bytes | None = None, ctype: str = "application/json"):
    req = urllib.request.Request(api + path, data=body if body is not None else b"{}", method="POST",
                                 headers={"authorization": f"Bearer {token}", "content-type": ctype, "user-agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")[:300]


def train(job: dict, d: Path) -> tuple[Path | None, str]:
    """Run doppel_train through the warden; (adapter path or None, log tail)."""
    (d / "job.json").write_text(json.dumps(job))
    out = d / "out"
    cmd = [str(WARDEN), "run", "--vram", "8000", "--max-runtime", "900", "--label", "doppel:tune", "--cwd", str(ROOT),
           "--wait", "--", "uv", "run", "--project", str(ROOT / "lab"), "python", "-m", "doppel_lab.doppel_train",
           "--job", str(d / "job.json"), "--out", str(out), "--device", "cuda"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
    end = json.loads(lines[-1]) if lines else {}
    log = ""
    if end.get("log") and Path(end["log"]).exists():
        log = Path(end["log"]).read_text(errors="replace")[-2000:]
    res = out / "result.json"
    if p.returncode != 0 or end.get("rc") != 0 or not res.exists():
        return None, f"warden rc={p.returncode} job={end.get('state')}/{end.get('rc')} {p.stderr[-300:]} {log[-1200:]}"
    r = json.loads(res.read_text())
    return out / f"{r['adapter']}.npz", json.dumps({k: r[k] for k in ("best_epoch", "games", "seconds")})


def once(api: str, token: str) -> bool:
    """One claim; True if a job was handled."""
    st, j = call(api, "/api/trainer/claim", token)
    if st != 200 or not isinstance(j, dict):
        print(f"claim: {st} {j}", file=sys.stderr, flush=True)
        return False
    if not j.get("job"):
        return False
    d = JOBS / j["job"]
    d.mkdir(parents=True, exist_ok=True)
    os.chmod(d, 0o700)
    try:
        path, note = train(j, d)
        if path is None:
            st, r = call(api, f"/api/trainer/jobs/{j['job']}/fail", token, json.dumps({"error": note[:480]}).encode())
            print(f"job {j['job'][:8]}: failed ({st}) {note[:200]}", flush=True)
        else:
            st, r = call(api, f"/api/trainer/jobs/{j['job']}/adapter", token, path.read_bytes(),
                         "application/octet-stream")
            print(f"job {j['job'][:8]}: {len(j['games'])} games -> {st} {r} {note}", flush=True)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="https://doppelbot.org")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=float, default=30)
    ap.add_argument("--token-file", type=Path, default=TOKEN)
    a = ap.parse_args()
    token = a.token_file.read_text().strip()
    while True:
        try:
            busy = once(a.api, token)
        except Exception as e:                     # keep polling through network blips
            print(f"worker: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            busy = False
        if a.once:
            return
        if not busy:
            time.sleep(a.interval)


if __name__ == "__main__":
    main()

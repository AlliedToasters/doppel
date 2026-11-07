"""Fetch public game sources into the lake (doppel_lab.lake). Resumable: finished outputs are skipped.

    python -m doppel_lab.lake_fetch twic [--first 920 --last 0]
    python -m doppel_lab.lake_fetch vintage [--month 03 --max-games 200000 --years 2013-2026]

twic:    weekly TWIC zips, one lake dir per issue (data/lake/twic/<issue>). Stops at the first missing issue.
vintage: the head of one Lichess month per year (data/lake/lichess/<YYYY-MM>). Streams; only the first
         --max-games games are downloaded, so a month's head is ~its first day (time-of-day/weekday bias).
Raw downloads never stay on disk. Requests are spaced out and carry a contact user-agent.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAKE = ROOT / "data" / "lake"
UA = "doppel-research (toast@doppelbot.org)"


def lake(src_cmd: str, out: Path, source: str, extra: str = "") -> dict | None:
    tmp = out.with_name(out.name + ".partial")
    subprocess.run(["rm", "-rf", str(tmp)], check=True)
    cmd = (f"set -o pipefail; {src_cmd} | {sys.executable} -m doppel_lab.lake - {tmp} --source {source} {extra}")
    p = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    last = [line for line in p.stdout.splitlines() if line.startswith("{")]
    # A --max-games stop closes the pipe early; curl/zstd then fail with SIGPIPE, which is expected.
    if not last or (p.returncode != 0 and "--max-games" not in extra):
        print(f"FAILED {out}: rc={p.returncode} {p.stderr[-500:]}", flush=True)
        return None
    r = json.loads(last[-1])
    if r["read"] == 0:
        print(f"FAILED {out}: no games read (rc={p.returncode}) {p.stderr[-300:]}", flush=True)
        return None
    tmp.rename(out)
    r["out"] = str(out.relative_to(ROOT))
    return r


def twic(a):
    d = LAKE / "twic"
    d.mkdir(parents=True, exist_ok=True)
    n, total = a.first, 0
    while not a.last or n <= a.last:
        out = d / str(n)
        if out.exists():
            n += 1
            continue
        url = f"https://theweekinchess.com/zips/twic{n}g.zip"
        with tempfile.TemporaryDirectory() as td:
            z = Path(td) / "t.zip"
            code = subprocess.run(["curl", "-sfL", "-A", UA, "-o", str(z), url]).returncode
            if code != 0:
                print(f"twic {n}: not available (curl {code}); stopping", flush=True)
                break
            r = lake(f"unzip -p {z}", out, "twic")
        if r:
            total += r["kept"]
            print(json.dumps({"issue": n, **r, "total": total}), flush=True)
        n += 1
        time.sleep(a.sleep)


def vintage(a):
    d = LAKE / "lichess"
    d.mkdir(parents=True, exist_ok=True)
    y0, y1 = (int(x) for x in a.years.split("-"))
    for y in range(y0, y1 + 1):
        tag = f"{y}-{a.month}"
        out = d / tag
        if out.exists():
            continue
        url = f"https://database.lichess.org/standard/lichess_db_standard_rated_{tag}.pgn.zst"
        r = lake(f"curl -sfL -A '{UA}' {url} | zstd -dc --long=31", out, "lichess", f"--max-games {a.max_games}")
        if r:
            print(json.dumps({"month": tag, **r}), flush=True)
        time.sleep(a.sleep)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("twic")
    t.add_argument("--first", type=int, default=920)
    t.add_argument("--last", type=int, default=0)
    t.add_argument("--sleep", type=float, default=3.0)
    v = sub.add_parser("vintage")
    v.add_argument("--month", default="03")
    v.add_argument("--years", default="2013-2026")
    v.add_argument("--max-games", type=int, default=200_000)
    v.add_argument("--sleep", type=float, default=5.0)
    a = ap.parse_args()
    {"twic": twic, "vintage": vintage}[a.cmd](a)

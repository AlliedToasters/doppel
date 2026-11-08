"""Book depth vs corpus size across months: does R0 / R1 book exit saturate? (CPU)

    # stream a full month (CC0), nothing raw on disk; corpus.py's filter (rated blitz, no BOT, Normal / Time
    # forfeit, both Elo 600-2800, clocks present):
    curl -sfL https://database.lichess.org/standard/lichess_db_standard_rated_2026-09.pgn.zst | zstd -dc --long=31 \
      | lab/poskey/target/release/poskey pgn --lichess-blitz --cap 80 --out data/lake/lichess_moves/2026-09
    python -m doppel_lab.posbook_months prep 2026-09        # drop games with a held-out player (split.py rule)
    python -m doppel_lab.posbook_months prep_heldout         # 2026-10 heldout minus the eval games ("2026-10h")
    python -m doppel_lab.posbook_months count 2026-10h 2026-09   # cumulative books: 2026-10 train + listed sets
    python -m doppel_lab.posbook_months analyze

Books: 2026-10 train, then 2026-10 train + each further month in turn, counted by position (R0) and by move-order
prefix (R1, `poskey count --key prefix`, equal to book.py's key). Probe: posbook.py's heldout eval games (10k per
band, 2026-10), exact counts at 9 nested subsample levels per book, so one curve spans N/256 .. N.
Output: lab/results/posbook_explore.json, section "depth_vs_corpus".
"""

from __future__ import annotations

import argparse
import json
import subprocess

import numpy as np
import polars as pl

from doppel_lab.posbook_common import BIN, CORPUS, DATA, V0, ply_index, save_section

c = pl.col
LAKE = DATA / "lake/lichess_moves"
OUT = V0 / "posbook/months"
LEVELS = 9
BANDS = (800, 1500, 2400)


def prep(month: str):
    from doppel_lab.split import held_out
    out = OUT / "books" / month
    out.mkdir(parents=True, exist_ok=True)
    n_in = n_out = 0
    for f in sorted((LAKE / month).glob("part-*.parquet")):
        g = pl.read_parquet(f, columns=["game_id", "white", "black", "band", "moves"])
        names = pl.concat([g["white"], g["black"]]).str.to_lowercase().unique()
        ho = set(n for n in names.to_list() if held_out(n))
        keep = g.filter(~c("white").str.to_lowercase().is_in(list(ho)) & ~c("black").str.to_lowercase().is_in(list(ho)))
        keep.select(game_id=pl.lit(month + ":") + c("game_id"), band="band", moves="moves").write_parquet(out / f.name)
        n_in, n_out = n_in + g.height, n_out + keep.height
        print(f"{month} {f.name}: {g.height:,} -> {keep.height:,}", flush=True)
    (out / "meta.json").write_text(json.dumps({"games_read": n_in, "games_kept": n_out}) + "\n")


def prep_heldout():
    """The 2026-10 heldout split minus every book.py eval game, as one more book increment ("2026-10h").
    Caveat: the probe players' other games are then in the book, which can only lengthen their book."""
    out = OUT / "books" / "2026-10h"
    out.mkdir(parents=True, exist_ok=True)
    ev = pl.read_parquet(V0 / "book/eval_games.parquet", columns=["game_id"])["game_id"]
    n = 0
    for f in sorted((CORPUS / "heldout").glob("*.parquet")):
        g = pl.read_parquet(f, columns=["game_id", "band", "moves"]).filter(~c("game_id").is_in(ev.implode()))
        g.with_columns(c("moves").list.head(80)).write_parquet(out / f.name)
        n += g.height
    (out / "meta.json").write_text(json.dumps({"games_kept": n}) + "\n")
    print(f"2026-10 heldout minus eval games: {n:,}", flush=True)


def count(months: list[str]):
    files = sorted(str(p) for p in (CORPUS / "train").glob("*.parquet"))
    sets = [("oct", [])] + [("oct+" + "+".join(months[:i + 1]), months[:i + 1]) for i in range(len(months))]
    for name, ms in sets:
        fs = files + [str(p) for m in ms for p in sorted((OUT / "books" / m).glob("part-*.parquet"))]
        for key in ("position", "prefix"):
            out = OUT / "runs" / f"{name}_{key}"
            if (out / "stats.json").exists():
                continue
            subprocess.run([str(BIN), "count", "--out", str(out), "--threads", "16", "--fpr", "0.01", "--key", key,
                            "--probe", str(V0 / "posbook/probe.parquet"), *fs], check=True, capture_output=True)
            for p in out.glob("table_*.npy"):     # only the probe counts are needed
                p.unlink()
            s = json.loads((out / "stats.json").read_text())
            print(f"{name} {key}: {s['games']:,} games, {s['seconds_total']:.0f}s, {s['peak_rss_gb']:.1f} GB", flush=True)


def analyze():
    pb = pl.read_parquet(V0 / "posbook/probe.parquet", columns=["band"])
    band = pb["band"].to_numpy()
    runs = sorted({p.name.rsplit("_", 1)[0] for p in (OUT / "runs").glob("*_position")}, key=len)
    lens = None
    pts = {"R0": [], "R1": []}
    per_run = {}
    for name in runs:
        per_run[name] = {}
        for key, reg in (("position", "R0"), ("prefix", "R1")):
            d = OUT / "runs" / f"{name}_{key}"
            if not (d / "stats.json").exists():
                continue
            s = json.loads((d / "stats.json").read_text())
            cnt = np.load(d / "probe_counts.npy")
            if lens is None:
                lens = np.load(d / "probe_lens.npy")
                gi, ply = ply_index(lens)
            rows = []
            for j in range(LEVELS):
                e = lens.astype(np.int64).copy()
                below = np.nonzero(cnt[:, j] < 1)[0]
                g, i = np.unique(gi[below], return_index=True)
                e[g] = ply[below[i]]
                r = {"games": s["games"] * 2.0 ** -j, "T1_mean": float(e.mean()), "T1_median": float(np.median(e)),
                     "T1_mean_by_band": {int(b): float(e[(band >= b) & (band < b + 100)].mean()) for b in BANDS}}
                rows.append(r)
                pts[reg].append((np.log2(r["games"]), r["T1_mean"], name, j))
            per_run[name][reg] = {"games": s["games"], "levels": rows}
    out = {"runs": per_run, "probe_games": int(len(lens)),
           "note": "probe = heldout 2026-10 eval games; extra months exclude games with a held-out player"}
    # slope (plies per doubling) by N range, from the largest book's levels plus the full points of smaller books
    for reg in ("R0", "R1"):
        p = sorted(pts[reg])
        x = np.array([a for a, *_ in p])
        y = np.array([b for _, b, *_ in p])
        segs = {}
        for lo, hi in ((16, 20), (20, 23), (23, 24.6), (24.6, 27)):
            m = (x >= lo) & (x < hi)
            if m.sum() >= 2 and np.ptp(x[m]) > 0.5:
                segs[f"2^{lo}-2^{hi}"] = float(np.polyfit(x[m], y[m], 1)[0])
        out[f"{reg}_plies_per_doubling"] = segs
        out[f"{reg}_curve"] = {"log2_games": x.tolist(), "T1_mean": y.tolist()}
    print(json.dumps({k: v for k, v in out.items() if "plies_per" in k}, indent=1))
    save_section("depth_vs_corpus", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "prep_heldout", "count", "analyze"])
    ap.add_argument("months", nargs="*")
    a = ap.parse_args()
    if a.cmd == "prep":
        for m in a.months:
            prep(m)
    elif a.cmd == "prep_heldout":
        prep_heldout()
    elif a.cmd == "count":
        count(a.months)
    else:
        analyze()

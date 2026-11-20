"""CPU pilots for backlog items 12d, 12e and 12f (tier 1) from saved outputs; no GPU, no new training.

    python -m doppel_lab.persona_pilots [--tag _ctx200]        -> lab/results/persona_pilots.json

Per band, per Allie test position: the ctx200 ttt hits (Maia-3, own adapter, control = the next side's adapter,
allie_ttt) joined with Maia-3's full policy over the legal moves (allie_knn test.npz, the same fp16 79M pass with
real Elos; its argmax is checked against the ttt Maia hit). Gains are top-1 points; "norm" = gain as a share of the
positions Maia-3 misses in that bin (headroom-free across bins and bands).
  12f  gain by Maia-3's margin (p1 - p2) and entropy, quintiles per band and fixed cut points across bands
  12d  per-input diversity (Maia-3 entropy, top-1 prob) vs across-input (random-swap cost = ctrl - Maia-3), per band
  12e  gain by ply bucket and by exact-repeat history (memory table `hist`: the side played this exact position
       before), the repertoire-vs-style split
Aggregate numbers only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
BANDS = (700, 1100, 1500, 1900, 2300)
OUT = ROOT / "lab/results/persona_pilots.json"
MARGIN_CUTS = (0.1, 0.25, 0.5, 0.75)          # fixed cut points on p1 - p2, comparable across bands
PLY_CUTS = (20, 40, 60)
c = pl.col


def maia_policy(band: int) -> pl.DataFrame:
    """(game_id, ply) -> Maia-3 margin, entropy, top-1 prob, argmax column, from knn_<band>/test.npz."""
    from doppel_lab.allie_knn import games, knn_dir
    d = knn_dir(band)
    tz = np.load(d / "test.npz")
    off = np.load(d / "off.npy")
    g = games(band).select("game_id").with_row_index("gi")
    pos = tz["pos"].astype(np.int64)
    gi = np.searchsorted(off, pos, side="right") - 1
    ply = pos - off[gi]
    lp = tz["logp"].astype(np.float32)                 # padding: legal -1, logp -inf
    mask = tz["legal"] >= 0
    n_legal = mask.sum(1)
    p = np.where(mask, np.exp(np.where(mask, lp, 0.0)), 0.0)
    p = p / p.sum(1, keepdims=True)
    srt = -np.sort(-p, 1)
    ent = -(np.where(p > 0, p * np.log(np.where(p > 0, p, 1)), 0)).sum(1)
    gid = g["game_id"].to_numpy()[gi]
    played = np.load(d / "played.npy")[pos].astype(np.int64)
    amax_hit = tz["legal"][np.arange(len(pos)), p.argmax(1)].astype(np.int64) == played
    return pl.DataFrame({"game_id": gid, "ply": ply, "p1": srt[:, 0], "margin": srt[:, 0] - srt[:, 1],
                         "entropy": ent, "n_legal": n_legal, "amax_hit": amax_hit})


def positions(band: int, tag: str) -> pl.DataFrame:
    z = np.load(ROOT / f"data/allie/ttt_{band}{tag}.npz")
    t = pl.DataFrame({k: z[k] for k in ("game_id", "ply", "maia", "own", "ctrl")})
    return t.join(maia_policy(band), on=["game_id", "ply"], how="left")


def gains(df: pl.DataFrame, by: str) -> list[dict]:
    out = []
    for r in (df.group_by(by).agg(n=pl.len(), maia=c("maia").mean(), own=c("own").mean(), ctrl=c("ctrl").mean())
              .sort(by).iter_rows(named=True)):
        miss = 1 - r["maia"]
        out.append({by: r[by], "n": r["n"], "maia_acc": round(100 * r["maia"], 2),
                    "own_gain": round(100 * (r["own"] - r["maia"]), 3),
                    "ctrl_gain": round(100 * (r["ctrl"] - r["maia"]), 3),
                    "own_norm": round(100 * (r["own"] - r["maia"]) / miss, 2) if miss > 0 else None,
                    "personal": round(100 * (r["own"] - r["ctrl"]), 3)})
    return out


def band_report(band: int, tag: str, mem: pl.DataFrame) -> dict:
    df = positions(band, tag)
    cov = df["margin"].is_not_null().mean()
    df = df.drop_nulls("margin")
    qs = np.quantile(df["margin"].to_numpy(), [0.2, 0.4, 0.6, 0.8])
    eq = np.quantile(df["entropy"].to_numpy(), [0.2, 0.4, 0.6, 0.8])
    df = df.with_columns(
        margin_q=pl.Series(np.searchsorted(qs, df["margin"].to_numpy(), side="right")),
        margin_bin=pl.Series(np.searchsorted(MARGIN_CUTS, df["margin"].to_numpy(), side="right")),
        entropy_q=pl.Series(np.searchsorted(eq, df["entropy"].to_numpy(), side="right")),
        ply_bin=pl.Series(np.searchsorted(PLY_CUTS, df["ply"].to_numpy(), side="right")))
    df = df.join(mem, on=["game_id", "ply"], how="left")
    m = df.select(c("maia").mean(), c("own").mean(), c("ctrl").mean()).row(0)
    return {
        "band": band, "positions": df.height, "policy_coverage": round(float(cov), 4),
        "argmax_agrees_maia_hit": round(float((df["amax_hit"] == df["maia"]).mean()), 4),
        "overall": {"maia_acc": round(100 * m[0], 2), "own_gain": round(100 * (m[1] - m[0]), 3),
                    "ctrl_gain": round(100 * (m[2] - m[0]), 3),
                    "own_norm": round(100 * (m[1] - m[0]) / (1 - m[0]), 2)},
        "12d": {"maia_entropy_mean": round(float(df["entropy"].mean()), 4),
                "maia_p1_mean": round(float(df["p1"].mean()), 4),
                "margin_median": round(float(df["margin"].median()), 4),
                "n_legal_mean": round(float(df["n_legal"].mean()), 2),
                "swap_cost": round(100 * (m[2] - m[0]), 3),
                "swap_cost_norm": round(100 * (m[2] - m[0]) / (1 - m[0]), 2)},
        "12f": {"margin_quintile_edges": [round(float(x), 4) for x in qs], "by_margin_quintile": gains(df, "margin_q"),
                "margin_cuts": list(MARGIN_CUTS), "by_margin_fixed": gains(df, "margin_bin"),
                "entropy_quintile_edges": [round(float(x), 4) for x in eq], "by_entropy_quintile": gains(df, "entropy_q")},
        "12e": {"ply_cuts": list(PLY_CUTS), "by_ply": gains(df, "ply_bin"),
                "hist_coverage": round(float(df["hist"].is_not_null().mean()), 4),
                "by_hist": gains(df.drop_nulls("hist"), "hist")},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_ctx200")
    ap.add_argument("--bands", default=",".join(map(str, BANDS)))
    a = ap.parse_args()
    mem = (pl.read_parquet(ROOT / "data/allie/memory_positions.parquet", columns=["game_id", "k", "hist"])
           .rename({"k": "ply"}).with_columns(c("ply").cast(pl.Int64)).unique(["game_id", "ply"]))
    rep = {"tag": a.tag, "bands": []}
    for b in map(int, a.bands.split(",")):
        r = band_report(b, a.tag, mem)
        rep["bands"].append(r)
        f = r["12f"]["by_margin_quintile"]
        print(f"{b}: n {r['positions']:,} cov {r['policy_coverage']:.3f} agree {r['argmax_agrees_maia_hit']:.3f} own {r['overall']['own_gain']:+.2f} "
              f"swap {r['12d']['swap_cost']:+.2f} H {r['12d']['maia_entropy_mean']:.3f} | margin-q norm gain "
              + " ".join(f"{x['own_norm']:+.1f}" for x in f) + " | personal "
              + " ".join(f"{x['personal']:+.2f}" for x in f), flush=True)
    OUT.write_text(json.dumps(rep, indent=1))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()

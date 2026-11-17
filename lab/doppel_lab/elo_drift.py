"""Do history methods hold up when the player's rating has moved? Allie 1400-1600, CPU only.

    python -m doppel_lab.elo_drift          # -> lab/results/elo_drift_1500.json

Per test side: drift = the side's Elo in the test game minus the median of its own Elo over its 40 context games
(allie_ttt.side_rows, the same games every method reads). Positions are binned by drift; per bin the top-1 gain over
Maia-3 of z, the lookup, z + lookup and the test-time adapter (ttt40), from combo.stack's saved hits. Game bootstrap.
Also: the adapter's gain by context span (days from the oldest context game to the test game).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
BAND = 1500
c = pl.col
DRIFT_BINS = ((-9999, -100), (-100, -40), (-40, 40), (40, 100), (100, 9999))
SPAN_BINS = ((0, 7), (7, 30), (30, 90), (90, 9999))
ARMS = ("z", "lookup", "z_lookup", "ttt40")


def sides() -> pl.DataFrame:
    from doppel_lab.allie_ttt import side_rows
    from doppel_lab.allie_knn import games
    rows, _ = side_rows(BAND)
    t_of = games(BAND).select("game_id", "t")
    rows = rows.join(t_of, on="game_id").with_columns(
        elo=pl.when(c("col") == 0).then(c("white_elo")).otherwise(c("black_elo")).cast(pl.Float64))
    ctx = rows.filter(c("role") != "test").group_by("pid").agg(
        ctx_elo=c("elo").median(), n_ctx=pl.len(), t0=c("t").min())
    tst = rows.filter(c("role") == "test").select("pid", "game_id", "col", test_elo="elo", tt="t")
    s = tst.join(ctx, on="pid", how="left").with_columns(drift=c("test_elo") - c("ctx_elo"))
    span = (c("tt") - c("t0"))
    if s.schema["tt"] in (pl.Datetime, pl.Datetime("us"), pl.Datetime("ns"), pl.Datetime("ms")):
        span = span.dt.total_seconds() / 86400
    else:
        span = span.cast(pl.Float64) / 86_400_000   # epoch milliseconds
    return s.with_columns(span_days=span)


def main():
    s = sides()
    h = np.load(ROOT / "data/allie/combo/stack_hits_1500.npz", allow_pickle=True)
    pos = pl.DataFrame({"game_id": h["game_id"].astype(str), "ply": h["ply"].astype(np.int64),
                        "i": np.arange(len(h["ply"]))}).with_columns(col=c("ply") % 2)
    pos = pos.join(s.select("game_id", "col", "drift", "span_days", "n_ctx"), on=["game_id", "col"], how="left")
    nohist = int(pos["drift"].null_count())            # sides with no context game: in no bin
    assert pos["i"].n_unique() == pos.height == len(h["ply"])
    pos = pos.sort("i").with_columns(c("drift").fill_null(np.nan), c("span_days").fill_null(np.nan))
    drift, span = pos["drift"].to_numpy(), pos["span_days"].to_numpy()
    gid = np.unique(pos["game_id"].to_numpy(), return_inverse=True)[1]
    m = h["maia"].astype(float)
    rng = np.random.default_rng(0)

    def gain(arm, sel):
        d = h[arm][sel].astype(float) - m[sel]
        g = np.unique(gid[sel], return_inverse=True)[1]
        gs, gn = np.bincount(g, d), np.bincount(g)
        b = rng.integers(0, len(gs), (1000, len(gs)))
        bs = 100 * gs[b].sum(1) / gn[b].sum(1)
        return {"pts": round(100 * float(d.mean()), 3),
                "ci95": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]}

    sd = s["drift"].drop_nulls().to_numpy()
    res = {"band": BAND, "positions": len(m), "positions_no_history": nohist, "sides": s.height,
           "side_drift_quantiles": {q: float(np.quantile(sd, q / 100)) for q in (5, 25, 50, 75, 95)},
           "by_drift": {}, "by_abs_drift": {}, "by_span_days": {}}
    for lo, hi in DRIFT_BINS:
        sel = (drift >= lo) & (drift < hi)
        res["by_drift"][f"{lo}..{hi}"] = {"positions": int(sel.sum()), "maia": round(float(m[sel].mean()), 4),
                                          **{a: gain(a, sel) for a in ARMS}}
    for lo, hi in ((0, 40), (40, 100), (100, 9999)):
        sel = (np.abs(drift) >= lo) & (np.abs(drift) < hi)
        res["by_abs_drift"][f"{lo}..{hi}"] = {"positions": int(sel.sum()), **{a: gain(a, sel) for a in ARMS}}
    for lo, hi in SPAN_BINS:
        sel = (span >= lo) & (span < hi)
        res["by_span_days"][f"{lo}..{hi}"] = {"positions": int(sel.sum()), **{a: gain(a, sel) for a in ARMS}}
    (ROOT / "lab/results/elo_drift_1500.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res["side_drift_quantiles"]))
    for k in ("by_drift", "by_abs_drift", "by_span_days"):
        print(k)
        for b, v in res[k].items():
            print(f"  {b:>12s} n={v['positions']:6d} " + "  ".join(f"{a} {v[a]['pts']:+.2f} {v[a]['ci95']}" for a in ARMS))


if __name__ == "__main__":
    main()

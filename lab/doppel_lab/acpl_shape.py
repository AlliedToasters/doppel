"""Why v0d fails the gate's variance check: per-game ACPL distributions, human vs Maia-3 vs Doppel (CPU).

    python -m doppel_lab.acpl_shape [--run v0d]

Same sources and scoring as v0_gate.game_checks (20k-node evals, metrics.game_stats). Quantiles of per-player-game
ACPL, the share of very clean (< 20) and very bad (> 120) games, and the ratio of spreads (IQR).
Output: lab/results/acpl_shape_<run>.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import polars as pl

from doppel_lab.v0_gate import player_stats, sources

QS = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v0d")
    ap.add_argument("--band", type=int, default=1500)
    a = ap.parse_args()
    res = {}
    for name, (g, e) in sources(a.run, a.band).items():
        games = pl.read_parquet(g)
        if "band" in games.columns:
            games = games.filter(pl.col("band") == a.band)
        x = player_stats(games, pl.read_parquet(e))[:, 0]
        x = x[np.isfinite(x)]
        q = np.quantile(x, QS)
        res[name] = {"n": len(x), "mean": float(x.mean()), "quantiles": dict(zip(map(str, QS), q.tolist())),
                     "iqr": float(q[4] - q[2]), "clean_lt20": float((x < 20).mean()), "bad_gt120": float((x > 120).mean()), "_x": x}
        print(f"{name:7s} n {len(x):5d}  mean {x.mean():5.1f}  q05..q95 " + " ".join(f"{v:5.1f}" for v in q)
              + f"  IQR {res[name]['iqr']:5.1f}  <20 {res[name]['clean_lt20']:.3f}  >120 {res[name]['bad_gt120']:.3f}", flush=True)
    # bootstrap the gate statistic: KS(doppel, human) - KS(maia3, human)
    from scipy.stats import ks_2samp
    rng = np.random.default_rng(0)
    xs = {k: v["_x"] for k, v in res.items()}
    diffs = []
    for _ in range(500):
        h = rng.choice(xs["human"], len(xs["human"]))
        diffs.append(ks_2samp(rng.choice(xs["doppel"], len(xs["doppel"])), h).statistic
                     - ks_2samp(rng.choice(xs["maia3"], len(xs["maia3"])), h).statistic)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    for v in res.values():
        v.pop("_x")
    res["ks_gap_doppel_minus_maia3_ci"] = [float(lo), float(hi)]
    print(f"KS(doppel) - KS(maia3) bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}]", flush=True)
    from doppel_lab.bench import RESULTS
    (RESULTS / f"acpl_shape_{a.run}.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()

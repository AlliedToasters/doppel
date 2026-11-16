"""Is z's backwards conformity real, and how does it arise? (CPU, cached style_fidelity pass B; ~2 min.)

    python -m doppel_lab.conformity_check     # -> lab/results/style_fidelity.json "conformity_check"

Per run, the choice-level conformity of style_fidelity.choice_level (plies >= 10): actual = log p_maia(played) -
E_maia[log p_maia]; model = E_model[log p_maia] - E_maia[log p_maia] (post: the player's z, prior: the mean z).
  bootstrap   players resampled 2,000 times: CI of rho(post, actual), rho(prior, actual) and of z_adds
  mechanism   zres = post - prior per player. E_post[log p_maia] - E_prior[log p_maia] < 0 whenever z moves
              mass off Maia's likely moves, so zres reads "how far z pushes away from Maia" as much as a
              direction. Report: share of players with zres < 0; rho(zres, actual); rho(|zres|, actual);
              rho(zres, steer zres) (does one z axis drive both?)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata

from doppel_lab.style_fidelity import HALF, MIN_PLY, RESULTS, targets

ROOT = Path(__file__).resolve().parents[2]
RUNS = (("v0f", 1500), ("v0f_half", 1500), ("v0f_quarter", 1500), ("v0f_1100", 1100), ("v0f_1900", 1900))
c = pl.col


def rho(a, b):
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def per_player(run: str, band: int) -> pl.DataFrame:
    _, T = targets(band, 0)
    key = T.select(c("g").alias("game"), "col", "player")
    Bm = pl.read_parquet(ROOT / f"data/v0e/style_{run}_B.parquet")
    return (Bm.filter(c("ply") >= MIN_PLY).with_columns(col=(c("ply") % 2).cast(pl.Int64)).join(key, on=["game", "col"])
            .group_by("player").agg(actual=(c("lp_played") - c("conf_maia")).mean(),
                                    post=(c("conf_post") - c("conf_maia")).mean(),
                                    prior=(c("conf_prior") - c("conf_maia")).mean(),
                                    s_post=(c("steer_post") - c("steer_maia")).mean(),
                                    s_prior=(c("steer_prior") - c("steer_maia")).mean(),
                                    n=pl.len())
            .sort("player"))


def main():
    rng = np.random.default_rng(0)
    out = {}
    for run, band in RUNS:
        P = per_player(run, band)
        y, po, pr = P["actual"].to_numpy(), P["post"].to_numpy(), P["prior"].to_numpy()
        zres = po - pr
        szres = P["s_post"].to_numpy() - P["s_prior"].to_numpy()
        bs = []
        for _ in range(2000):
            i = rng.integers(0, len(y), len(y))
            bs.append((rho(po[i], y[i]), rho(pr[i], y[i])))
        bs = np.array(bs)
        d = bs[:, 0] - bs[:, 1]
        out[run] = {"players": len(y), "post": round(rho(po, y), 3), "prior": round(rho(pr, y), 3),
                    "post_ci95": [round(float(v), 3) for v in np.percentile(bs[:, 0], [2.5, 97.5])],
                    "z_adds": round(rho(po, y) - rho(pr, y), 3),
                    "z_adds_ci95": [round(float(v), 3) for v in np.percentile(d, [2.5, 97.5])],
                    "p_z_adds_ge0": round(float((d >= 0).mean()), 4),
                    "zres_neg_share": round(float((zres < 0).mean()), 3),
                    "rho_zres_actual": round(rho(zres, y), 3), "rho_abs_zres_actual": round(rho(np.abs(zres), y), 3),
                    "rho_zres_steer_zres": round(rho(zres, szres), 3),
                    "rho_prior_actual_vs_post": round(rho(pr, po), 3)}
        print(run, out[run], flush=True)
    res = json.loads(RESULTS.read_text())
    res["conformity_check"] = out
    RESULTS.write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()

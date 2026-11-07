"""How well does the Elo curve fit Lichess blitz? Empirical expected score vs rating gap (CPU).

    python -m doppel_lab.elo_curve

2026-10 corpus (train + heldout, rated blitz). For each gap bin (higher-rated side's view): observed score
vs the Elo expectation 1 / (1 + 10^(-gap/400)). Two fits per rating level (mean rating band):
  scale     E = 1 / (1 + 10^(-gap/s)), s free (Elo's s = 400; a larger s means flatter, fatter tails)
  mixture   E = (1 - m) * Elo(gap) + m * 0.5: a share m of games behave like coin flips (mis-rated sides,
            disconnects, sandbagging), the rest follow Elo
Each fit is repeated without games where either side is rated 1500-1599 (*_no1500): new Lichess accounts
start at 1500, so that bin holds provisional players whose rating says little yet.
Output: lab/results/elo_curve.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "corpus" / "2026-10"
RESULTS = ROOT / "lab" / "results"
c = pl.col


def elo(gap, s=400.0):
    return 1 / (1 + 10 ** (-gap / s))


def fit(gap: np.ndarray, score: np.ndarray, n: np.ndarray) -> dict:
    """Weighted least squares on binned means (bins carry n)."""
    sse = lambda f: float((n * (score - f) ** 2).sum())
    s = minimize_scalar(lambda s: sse(elo(gap, s)), bounds=(200, 2000), method="bounded").x
    m = minimize_scalar(lambda m: sse((1 - m) * elo(gap) + m * 0.5), bounds=(0, 1), method="bounded").x
    return {"scale": float(s), "mixture_m": float(m), "sse_elo": sse(elo(gap)), "sse_scale": sse(elo(gap, s)),
            "sse_mixture": sse((1 - m) * elo(gap) + m * 0.5)}


def main():
    d = pl.scan_parquet(str(CORPUS / "*" / "*.parquet")).select("white_elo", "black_elo", "result", "base").collect()
    d = d.with_columns(sw=pl.when(c("result") == "1-0").then(1.0).when(c("result") == "0-1").then(0.0).otherwise(0.5),
                       g=(c("white_elo") - c("black_elo")).cast(pl.Float64),
                       lvl=((c("white_elo") + c("black_elo")) // 400 * 200).clip(800, 2400))
    # higher-rated side's view; white's first-move edge averages out because both colours are pooled
    d = d.with_columns(gap=c("g").abs(), s=pl.when(c("g") >= 0).then(c("sw")).otherwise(1 - c("sw")))
    d = d.filter(c("gap") > 0).with_columns(gb=(c("gap") // 25 * 25 + 12.5).clip(upper_bound=987.5))
    res = {"games": d.height, "levels": {}}
    no1500 = d.filter(~c("white_elo").is_between(1500, 1599) & ~c("black_elo").is_between(1500, 1599))
    for name, sub in [("all", d), ("all_no1500", no1500)] + [(f"{int(l)}_no1500", no1500.filter(c("lvl") == l))
                                                             for l in sorted(d["lvl"].unique())] + [(f"{int(l)}", d.filter(c("lvl") == l)) for l in sorted(d["lvl"].unique())]:
        b = sub.group_by("gb").agg(score=c("s").mean(), n=pl.len()).sort("gb").filter(c("n") >= 200)
        gap, score, n = (b[k].to_numpy().astype(float) for k in ("gb", "score", "n"))
        f = fit(gap, score, n)
        pick = {g_: (float(score[i]), float(elo(g_))) for i, g_ in enumerate(gap) if g_ in (112.5, 212.5, 312.5, 512.5, 712.5)}
        f["observed_vs_elo"] = pick
        f["games"] = int(n.sum())
        res["levels"][name] = f
        print(f"{name:>5s} n={f['games']:>9,d} scale={f['scale']:.0f} mix_m={f['mixture_m']:.3f}  "
              + "  ".join(f"gap{int(k)}: {v[0]:.3f} vs {v[1]:.3f}" for k, v in pick.items()), flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "elo_curve.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()

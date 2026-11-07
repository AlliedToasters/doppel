"""Opening repertoire breadth by rating, and whether a narrow repertoire pays off (2026-10 held-out, 3+0; CPU).

    python -m doppel_lab.repertoire

Held-out players with >= MIN_GAMES 3+0 games as each colour. Breadth = Shannon entropy (bits) of the player's
first 4 plies as White (their 2 moves and the opponent's replies vary, so the White line is the player's moves
1 and 2 only), and of their reply to 1.e4 / 1.d4 as Black. Rating bins of 100. Within each bin: Spearman of
breadth with score minus Elo expectation (does specialising help?), with the side's own book exit, and its
mean by bin. Entropy grows with sample size, so every player's breadth uses exactly MIN_GAMES games per colour.
Output: lab/results/repertoire.json
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from doppel_lab.persistence import CORPUS, RESULTS, held_out

MIN_GAMES = 30
c = pl.col


def entropy(xs) -> float:
    _, n = np.unique(np.asarray(xs, dtype=object).astype(str), return_counts=True)
    p = n / n.sum()
    return float(-(p * np.log2(p)).sum())


def main():
    g = pl.concat([pl.read_parquet(f, columns=["white", "black", "white_elo", "black_elo", "result", "moves", "base",
                                                "inc"]) for f in sorted(CORPUS.glob("part-*.parquet"))])
    g = g.filter((c("base") == 180) & (c("inc") == 0) & (c("moves").list.len() >= 4))
    sw = pl.when(c("result") == "1-0").then(1.0).when(c("result") == "0-1").then(0.0).otherwise(0.5)
    w = g.select(p=c("white").str.to_lowercase(), elo=c("white_elo"), op=c("black_elo"), score=sw,
                 line=c("moves").list.get(0) + " " + c("moves").list.get(2))
    b = g.filter(c("moves").list.get(0).is_in(["e2e4", "d2d4"])).select(
        p=c("black").str.to_lowercase(), elo=c("black_elo"), op=c("white_elo"), score=1 - sw,
        line=c("moves").list.get(0) + " " + c("moves").list.get(1))
    out = {}
    for name, d in (("white", w), ("black", b)):
        d = d.with_columns(k=pl.int_range(pl.len()).over("p"), n=pl.len().over("p")).filter(c("n") >= MIN_GAMES)
        d = d.filter(c("k") < MIN_GAMES)
        d = d.filter(pl.Series([held_out(x) for x in d["p"].to_list()]))
        d = d.with_columns(exp=1 / (1 + 10 ** ((c("op") - c("elo")) / 400)))
        per = d.group_by("p").agg(elo=c("elo").mean(), resid=(c("score") - c("exp")).mean(), lines=c("line"))
        per = per.with_columns(H=pl.Series([entropy(x) for x in per["lines"].to_list()])).drop("lines")
        per = per.with_columns(bin=(c("elo") // 100 * 100).clip(800, 2400))
        rows = []
        for bn, sub in sorted(per.group_by("bin"), key=lambda t: t[0][0]):
            if sub.height < 50:
                continue
            rho = spearmanr(sub["H"], sub["resid"]).statistic
            rows.append({"bin": int(bn[0]), "players": sub.height, "breadth_bits": float(sub["H"].mean()),
                         "rho_breadth_vs_resid": float(rho)})
        out[name] = {"players": per.height, "by_rating": rows,
                     "rho_breadth_rating": float(spearmanr(per["H"], per["elo"]).statistic)}
        print(f"{name}: {per.height:,} players; breadth vs rating rho {out[name]['rho_breadth_rating']:+.3f}", flush=True)
        for r in rows:
            print(f"   {r['bin']}: n {r['players']:5d}  breadth {r['breadth_bits']:.2f} bits  "
                  f"rho(breadth, score - expected) {r['rho_breadth_vs_resid']:+.3f}", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "repertoire.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()

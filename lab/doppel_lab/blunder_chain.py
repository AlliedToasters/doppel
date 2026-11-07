"""Do blunders cluster within a game? 1500-band train table (CPU).

    python -m doppel_lab.blunder_chain

Per own move at ply >= 10: was the player's own previous move a blunder; was the opponent's last move (the ply
just before) a blunder. A blunder leaves a worse position, and worse positions are harder, so rates are compared
inside cells of Maia-3 entropy decile x the mover's win chances before the move (wc_own, 5 bins) x clock
fraction (3 bins); effects are cell-size-weighted differences. Reported: own-after-own (does a blunder breed
another?) and own-after-opponent's (do players miss or relax after a gift?).
Output: lab/results/blunder_chain.json
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl

from doppel_lab.features import table

c = pl.col


def effect(t: pl.DataFrame, flag: str) -> dict:
    keys = ["dq", "wq", "cq"]
    g = t.group_by(keys + [flag]).agg(n=pl.len(), b=c("blunder").mean())
    j = g.filter(c(flag)).join(g.filter(~c(flag)), on=keys, suffix="_0").filter(c("n_0") >= 50)
    w = j["n"].to_numpy().astype(float)
    d = (j["b"] - j["b_0"]).to_numpy()
    base = float((w * j["b_0"].to_numpy()).sum() / w.sum())
    att = float((w * d).sum() / w.sum())
    se = float(np.sqrt((w ** 2 * (j["b"] * (1 - j["b"]) / j["n"] + j["b_0"] * (1 - j["b_0"]) / j["n_0"]).to_numpy()).sum()) / w.sum())
    raw1 = float(t.filter(c(flag))["blunder"].mean())
    raw0 = float(t.filter(~c(flag))["blunder"].mean())
    return {"treated": int(w.sum()), "matched_base_rate": base, "matched_effect": att, "se": se,
            "raw_rate_after": raw1, "raw_rate_otherwise": raw0}


def main():
    from doppel_lab.bench import RESULTS
    t = table.load("train", "T2", 1500, columns=["game_id", "mover", "ply", "maia_entropy", "wc_own", "clock_frac",
                                                   "blunder"]).sort("game_id", "ply")
    t = t.with_columns(opp_prev=c("blunder").shift(1).over("game_id"), own_prev=c("blunder").shift(2).over("game_id"))
    t = t.filter((c("ply") >= 10) & c("maia_entropy").is_finite() & c("wc_own").is_not_null()
                 & c("opp_prev").is_not_null() & c("own_prev").is_not_null())
    t = t.with_columns(dq=c("maia_entropy").qcut(10, labels=[str(i) for i in range(10)], allow_duplicates=True),
                       wq=c("wc_own").cut([-0.6, -0.2, 0.2, 0.6], labels=["lost", "worse", "equal", "better", "won"]),
                       cq=c("clock_frac").fill_null(0.5).cut([0.1, 0.3], labels=["<10%", "10-30%", ">30%"]),
                       own_prev=c("own_prev").cast(pl.Boolean), opp_prev=c("opp_prev").cast(pl.Boolean))
    res = {"rows": t.height, "own_after_own_blunder": effect(t, "own_prev"),
           "own_after_opponent_blunder": effect(t, "opp_prev")}
    for k in ("own_after_own_blunder", "own_after_opponent_blunder"):
        r = res[k]
        print(f"{k:28s} raw {r['raw_rate_after']:.4f} vs {r['raw_rate_otherwise']:.4f};  matched effect "
              f"{r['matched_effect']:+.4f} on a base of {r['matched_base_rate']:.4f} (z {r['matched_effect'] / r['se']:+.1f}, "
              f"n {r['treated']:,})", flush=True)
    (RESULTS / "blunder_chain.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()

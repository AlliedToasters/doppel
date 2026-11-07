"""Human noise floor: when a player reaches the same position again, how often do they repeat their move? (CPU)

    python -m doppel_lab.noise_floor

2026-10 held-out corpus, 3+0. A "position" is the move-order prefix (transpositions are not merged) at plies
PLIES. For each (player, prefix) the player reached >= 2 times as the side to move: the self-agreement rate,
sum n_i (n_i - 1) / (N (N - 1)) (the chance two random visits get the same move, unbiased). Compared with
the population's agreement at the same prefixes (two random different players), and with the best a model
could do: top-1 accuracy of the player's own modal move (leave-one-out). Rows pooled by ply.
Output: lab/results/noise_floor.json
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl

from doppel_lab.persistence import CORPUS, RESULTS

PLIES = (4, 6, 8, 10, 12, 14, 16, 20)
c = pl.col


def main():
    g = pl.concat([pl.read_parquet(f, columns=["white", "black", "white_elo", "black_elo", "moves", "base", "inc"])
                   for f in sorted(CORPUS.glob("part-*.parquet"))])
    g = g.filter((c("base") == 180) & (c("inc") == 0))
    print(f"{g.height:,} 3+0 games", flush=True)
    out = {"games": g.height, "plies": {}}
    for k in PLIES:
        x = g.filter(c("moves").list.len() > k).select(
            player=pl.when(k % 2 == 0).then(c("white")).otherwise(c("black")).str.to_lowercase(),
            pre=c("moves").list.head(k).list.join(" "), mv=c("moves").list.get(k))
        cnt = x.group_by("player", "pre", "mv").len().with_columns(c("len").cast(pl.Float64))
        per = cnt.group_by("player", "pre").agg(N=c("len").sum(), s2=(c("len") * (c("len") - 1)).sum(),
                                                top=c("len").max(), second=c("len").sort(descending=True).get(1, null_on_oob=True))
        per = per.filter(c("N") >= 2)
        self_agree = float(per["s2"].sum() / (per["N"] * (per["N"] - 1)).sum())
        # leave-one-out modal accuracy: a visit is predicted by the mode of the player's other visits
        # (ties broken toward the same move): visits of the top move are right if top-1 >= second, others never.
        per = per.with_columns(second=c("second").fill_null(0))
        loo = float(per.select((c("top") * ((c("top") - 1) >= c("second")).cast(pl.Float64)).sum()).item() / per["N"].sum())
        # population agreement at the same prefixes, between different players
        pop = cnt.group_by("pre", "mv").agg(n=c("len").sum(), own2=(c("len") * c("len")).sum())
        pre_tot = pop.group_by("pre").agg(N=c("n").sum(), sq=(c("n") ** 2).sum(), own=c("own2").sum())
        # pairs of visits by different players: (sum_m n_m^2 - sum_{p,m} n_pm^2) / (N^2 - sum_p N_p^2)
        pn = cnt.group_by("pre", "player").agg(Np=c("len").sum()).group_by("pre").agg(sqp=(c("Np") ** 2).sum())
        pre_tot = pre_tot.join(pn, on="pre").filter(c("pre").is_in(per["pre"].unique().implode()))
        pop_agree = float((pre_tot["sq"] - pre_tot["own"]).sum() / (pre_tot["N"] ** 2 - pre_tot["sqp"]).sum())
        r = {"repeat_pairs": int(per.height), "visits": int(per["N"].sum()), "self_agreement": self_agree,
             "population_agreement": pop_agree, "loo_modal_top1": loo}
        out["plies"][str(k)] = r
        print(f"ply {k:2d}: {r['repeat_pairs']:>9,} (player, position) repeats, {r['visits']:>10,} visits; "
              f"self-agreement {self_agree:.3f}, between players {pop_agree:.3f}, own-mode top-1 {loo:.3f}", flush=True)
    # by rating band (the mover's rating at that game), plies 8-16 pooled
    rows = []
    for k in (8, 10, 12, 14, 16):
        rows.append(g.filter(c("moves").list.len() > k).select(
            player=pl.when(k % 2 == 0).then(c("white")).otherwise(c("black")).str.to_lowercase(),
            elo=pl.when(k % 2 == 0).then(c("white_elo")).otherwise(c("black_elo")),
            pre=pl.lit(str(k) + ":") + c("moves").list.head(k).list.join(" "), mv=c("moves").list.get(k)))
    x = pl.concat(rows)
    pe = x.group_by("player").agg(e=c("elo").mean())
    cnt = x.group_by("player", "pre", "mv").len().with_columns(c("len").cast(pl.Float64))
    per = cnt.group_by("player", "pre").agg(N=c("len").sum(), s2=(c("len") * (c("len") - 1)).sum()).filter(c("N") >= 2)
    per = per.join(pe, on="player").with_columns(band=(c("e") // 200 * 200).clip(800, 2400))
    by = per.group_by("band").agg(pairs=pl.len(), self_agreement=c("s2").sum() / (c("N") * (c("N") - 1)).sum()).sort("band")
    out["by_band_plies_8_16"] = by.to_dicts()
    print("self-agreement by rating band (plies 8-16):", ", ".join(f"{int(r['band'])}: {r['self_agreement']:.3f}" for r in by.to_dicts()))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "noise_floor.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()

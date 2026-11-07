"""Rematches: when the same two players meet again within 60 minutes, does the previous winner beat the Elo
expectation? (Lichess heads 2018-2026, CPU)

    python -m doppel_lab.rematch

Pairs of consecutive games between the same two accounts (either colour) with the second starting within GAP_MIN
of the first. Outcome: the first game's winner's score in the rematch minus the expectation from the rematch's
own pre-game ratings (so the rating exchange from game 1 is already priced in). No fixed effects, so no
finite-sequence bias. Positive = the winner's edge persists beyond what the updated ratings say (a matchup edge,
or the loser tilting). Also by how game 1 ended (time vs other) and colour swap.
Output: lab/results/rematch.json
"""

from __future__ import annotations

import json

import polars as pl

from doppel_lab.tilt import GAP_MIN, LAKE, RESULTS, sides_of

c = pl.col


def main():
    s = pl.concat([sides_of(p) for p in sorted(LAKE.glob("????-??")) if int(p.name[:4]) >= 2018])
    w = s.filter(c("side") == 0).select("game_id", "t", "player", "score", "resid", "termination")
    b = s.filter(c("side") == 1).select("game_id", bp="player")
    g = w.join(b, on="game_id").with_columns(pair=pl.min_horizontal("player", "bp") + "|" + pl.max_horizontal("player", "bp"))
    g = g.sort("pair", "t").with_columns(
        nt=c("t").shift(-1).over("pair"), nw=c("player").shift(-1).over("pair"), ns=c("score").shift(-1).over("pair"),
        nr=c("resid").shift(-1).over("pair"))
    g = g.filter(c("nt").is_not_null() & ((c("nt") - c("t")).dt.total_seconds() / 60 <= GAP_MIN) & (c("score") != 0.5))
    # winner of game 1 (by name), their score and residual in game 2
    g = g.with_columns(winner=pl.when(c("score") == 1).then(c("player")).otherwise(c("bp")))
    g = g.with_columns(win_is_w2=c("winner") == c("nw"),
                       r2=pl.when(c("winner") == c("nw")).then(c("nr")).otherwise(-c("nr")),
                       s2=pl.when(c("winner") == c("nw")).then(c("ns")).otherwise(1 - c("ns")),
                       on_time=c("termination") == "Time forfeit")
    res = {"rematches": g.height, "winner_resid": float(g["r2"].mean()),
           "se": float(g["r2"].std() / g.height ** 0.5), "winner_score": float(g["s2"].mean())}
    print(f"{g.height:,} rematches within {GAP_MIN} min; game-1 winner's residual in the rematch "
          f"{res['winner_resid']:+.4f} (se {res['se']:.4f}), raw score {res['winner_score']:.3f}", flush=True)
    for k in ("on_time", "win_is_w2"):
        t = g.group_by(k).agg(n=pl.len(), r=c("r2").mean(), se=c("r2").std() / pl.len().sqrt()).sort(k)
        res[k] = t.to_dicts()
        print(f"  by {k}: " + "; ".join(f"{r[k]}: {r['r']:+.4f} ± {r['se']:.4f} (n {r['n']:,})" for r in t.to_dicts()), flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "rematch.json").write_text(json.dumps(res, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()

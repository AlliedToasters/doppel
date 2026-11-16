"""Is there game-level state a per-turn z update could track? (CPU, ~10 min.)

    python -m doppel_lab.ingame_state        # -> lab/results/ingame_state.json

Idea under test: update z every turn from the moves already played in this game (plies 0-9 included; the benchmark
scores plies >= 10 given the prefix). That only helps if a player's deviation from Maia has a component that is
shared within a game but differs between that player's games ("today's form", tilt after a blunder, a sharp game).

Signals per own move (forced moves dropped), each residualised globally on position / clock / rating covariates
(Maia entropy and top-p, ply bin, clock fraction, legal-move count, both Elos), then centred:
  surprise   maia_played_lp + maia_entropy       (0 in expectation if the player samples from Maia)
  offtop     maia_played_rank > 1
  think      think_frac
  loss       wc_loss (net win-chance loss)
Variance components from pairwise covariances (players with >= 16 games):
  player       cov of two moves from different games of the same player
  game         cov of two moves from the same game, minus player  (the ceiling for any in-game update)
  lag profile  same-game cov at own-move lag 1..30: decaying = local position similarity, plateau = game state
Prediction (the simplest recursive update): at each scored own move (ply >= 10), regress the residual on the
player's leave-this-game-out mean and on the mean of this game's earlier own moves; report the R^2 the prefix adds,
by prefix length.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "data/v0/features/moves_f1_T2net_1500_train"
GAMES = ROOT / "data/v0/games_1500_train.parquet"
RESULTS = ROOT / "lab/results/ingame_state.json"
MIN_GAMES = 16
SIGNALS = ("surprise", "offtop", "think", "loss")
c = pl.col


def load(parts: int | None = None) -> pl.DataFrame:
    files = sorted(TABLE.glob("part-*.parquet"))[:parts]
    t = (pl.scan_parquet(files)
         .select("game_id", "mover", "ply", "maia_played_lp", "maia_entropy", "maia_p_top", "maia_played_rank",
                 "think_frac", "wc_loss", "clock_frac", "n_legal", "elo_self", "elo_opp")
         .filter(c("n_legal") > 1)
         .with_columns(surprise=c("maia_played_lp") + c("maia_entropy"),
                       offtop=(c("maia_played_rank") > 1).cast(pl.Float32),
                       think=c("think_frac"), loss=c("wc_loss"))
         .collect())
    g = pl.read_parquet(GAMES, columns=["game_id", "white", "black"])
    t = t.join(g, on="game_id").with_columns(
        player=pl.when(c("mover") == 0).then(c("white")).otherwise(c("black")).str.to_lowercase()).drop("white", "black")
    ng = t.select("player", "game_id").unique().group_by("player").len()
    keep = ng.filter(c("len") >= MIN_GAMES)["player"]
    return t.filter(c("player").is_in(keep.implode()))


def residualise(t: pl.DataFrame) -> pl.DataFrame:
    def design(d):
        ply = d["ply"].to_numpy()
        cols = [np.ones(d.height), d["maia_entropy"].to_numpy(), d["maia_entropy"].to_numpy() ** 2,
                d["maia_p_top"].to_numpy(), np.nan_to_num(d["clock_frac"].to_numpy(), nan=0.5),
                np.nan_to_num(d["clock_frac"].to_numpy(), nan=0.5) ** 2, np.log(d["n_legal"].to_numpy()),
                (d["elo_self"].to_numpy() - 1500) / 100, (d["elo_opp"].to_numpy() - 1500) / 100]
        cols += [(np.minimum(ply // 10, 8) == b).astype(float) for b in range(1, 9)]
        return np.column_stack(cols).astype(np.float32)

    X = design(t)
    rng = np.random.default_rng(0)
    out = {}
    for s in SIGNALS:
        y = t[s].to_numpy().astype(np.float64)
        ok = np.isfinite(y) & np.all(np.isfinite(X), 1)
        fit = np.flatnonzero(ok)
        fit = rng.choice(fit, min(len(fit), 5_000_000), replace=False)
        w = np.linalg.lstsq(X[fit], y[fit], rcond=None)[0]
        r = np.where(ok, y - X @ w, np.nan)
        out[s] = r - np.nanmean(r)
    return t.select("game_id", "mover", "ply", "player").with_columns(**{s: pl.Series(v) for s, v in out.items()})


def components(t: pl.DataFrame, s: str) -> dict:
    d = t.filter(c(s).is_not_nan()).select("player", "game_id", "mover", x=c(s))
    var = float((d["x"] ** 2).mean())
    G = d.group_by("player", "game_id", "mover").agg(S=c("x").sum(), Q=(c("x") ** 2).sum(), n=pl.len())
    P = d.group_by("player").agg(S=c("x").sum(), Q=(c("x") ** 2).sum(), n=pl.len())
    sg = float(((G["S"] ** 2 - G["Q"]) / 2).sum())
    ng = float((G["n"] * (G["n"] - 1) / 2).sum())
    sp = float(((P["S"] ** 2 - P["Q"]) / 2).sum())
    npairs = float((P["n"] * (P["n"] - 1) / 2).sum())
    cov_game, cov_player = sg / ng, (sp - sg) / (npairs - ng)
    return {"var": var, "player": cov_player, "game": cov_game - cov_player, "move": var - cov_game,
            "player_share": cov_player / var, "game_share": (cov_game - cov_player) / var}


def lag_profile(t: pl.DataFrame, s: str, lags=(1, 2, 3, 5, 8, 12, 20, 30)) -> dict:
    d = (t.filter(c(s).is_not_nan()).select("player", "game_id", "mover", "ply", x=c(s))
         .sort("game_id", "mover", "ply"))
    pm = d.group_by("player").agg(pm=c("x").mean())
    d = d.join(pm, on="player").with_columns(xr=c("x") - c("pm"))   # remove the player's own mean
    var = float((d["xr"] ** 2).mean())
    out = {}
    for L in lags:
        z = d.with_columns(prev=c("xr").shift(L).over("game_id", "mover")).drop_nulls("prev")
        out[L] = float((z["xr"] * z["prev"]).mean() / var) if z.height > 1000 else None
    return out


def prediction(t: pl.DataFrame, s: str) -> dict:
    d = (t.filter(c(s).is_not_nan()).select("player", "game_id", "mover", "ply", x=c(s))
         .sort("game_id", "mover", "ply"))
    P = d.group_by("player").agg(PS=c("x").sum(), Pn=pl.len())
    G = d.group_by("game_id", "mover").agg(GS=c("x").sum(), Gn=pl.len())
    d = (d.join(P, on="player").join(G, on=["game_id", "mover"])
         .with_columns(plogo=(c("PS") - c("GS")) / (c("Pn") - c("Gn")),
                       k=pl.int_range(pl.len()).over("game_id", "mover"),
                       cs=c("x").cum_sum().over("game_id", "mover") - c("x"))
         .with_columns(prefix=c("cs") / c("k"),
                       last5=(c("cs") - c("cs").shift(5).over("game_id", "mover").fill_null(0.0))
                       / pl.min_horizontal(c("k"), 5))
         .filter((c("ply") >= 10) & (c("k") >= 3)))
    out = {}
    for lo, hi in ((3, 8), (8, 15), (15, 25), (25, 999)):
        b = d.filter((c("k") >= lo) & (c("k") < hi))
        y = b["x"].to_numpy()

        def r2(*cols):
            X = np.column_stack([np.ones(len(y))] + [b[k].to_numpy() for k in cols])
            w = np.linalg.lstsq(X, y, rcond=None)[0]
            return 1 - ((y - X @ w) ** 2).mean() / y.var(), w

        rp, _ = r2("plogo")
        rf, w = r2("plogo", "prefix")
        rl, _ = r2("plogo", "prefix", "last5")
        out[f"{lo}-{hi - 1 if hi < 999 else '+'}"] = {
            "n": len(y), "r2_player": rp, "r2_plus_prefix": rf, "prefix_adds": rf - rp,
            "last5_adds_beyond_prefix": rl - rf, "prefix_coef": float(w[2])}
    return out


def main():
    import sys
    parts = int(sys.argv[1]) if len(sys.argv) > 1 else None   # smoke: first N parts
    t = residualise(load(parts))
    print(f"{t.height:,} moves, {t['player'].n_unique():,} players", flush=True)
    res = {}
    for s in SIGNALS:
        comp = components(t, s)
        lag = lag_profile(t, s)
        pred = prediction(t, s)
        res[s] = {"components": comp, "lag_corr_within_player": lag, "prediction": pred}
        print(f"{s:9s} player {comp['player_share']:.4f}  game {comp['game_share']:.4f}  "
              f"lag {' '.join(f'{k}:{v:+.3f}' for k, v in lag.items())}", flush=True)
        for k, v in pred.items():
            print(f"          prefix k={k:6s} R2 player {v['r2_player']:.4f}  +prefix {v['prefix_adds']:+.4f}  "
                  f"+last5 {v['last5_adds_beyond_prefix']:+.4f}  n {v['n']:,}", flush=True)
    if parts is None:
        RESULTS.write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()

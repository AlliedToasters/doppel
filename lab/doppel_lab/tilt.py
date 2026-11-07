"""Tilt: does a player's next game change after a loss? Lichess month heads 2018-2026, within-player (CPU).

    python -m doppel_lab.tilt

Sessions: one player's games sorted by start time (UTCDate/UTCTime); a pair is (game, the player's next game)
when the next one starts within GAP_MIN minutes. Outcomes of the next game, from that player's side:
  resid        score minus the Elo expectation at the next game's own (pre-game) ratings
  think_mean   mean seconds per own move (clock differences + increment)
  plies, flag_loss, early_loss (lost in <= 40 plies), same_tc (kept the time control)
Continuation: does the player start another game within GAP_MIN, for games that start at least GAP_MIN before
the head's last game (so the window isn't cut off).
Estimates are within-player (player fixed effects): outcome and the previous-result dummies are demeaned per
player before least squares, so a player is compared with themselves, which removes "weak players lose more".
With a few games per session that estimator is biased (Miller & Sanjurjo's finite-sequence bias): a game's
score is both one pair's outcome and the next pair's regressor, and demeaning turns that overlap into a
negative dependence. So every coefficient is recomputed with game order permuted within each session (N_PERM
times; records intact, pairs rebuilt), and the reported effect is observed minus permuted. Fixed effects are
per session (a player's chain of games with gaps <= GAP_MIN).
Also after streaks (two losses / two wins in a row) and by how the previous game was lost (flag vs other).
Output: lab/results/tilt.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
LAKE = ROOT / "data" / "lake" / "lichess"
RESULTS = ROOT / "lab" / "results"
GAP_MIN = 60
N_PERM = 3
c = pl.col


def sides_of(p: Path) -> pl.DataFrame:
    d = pl.read_parquet(p / "*.parquet", columns=["game_id", "white", "black", "white_elo", "black_elo", "result",
                                                  "termination", "time_control", "utc", "plies", "clocks",
                                                  "white_title", "black_title", "fen"])
    d = d.filter(c("fen").is_null() & c("white_elo").is_not_null() & c("black_elo").is_not_null()
                 & c("time_control").str.contains(r"^\d+\+\d+$") & c("utc").is_not_null()
                 & ~c("white_title").fill_null("").is_in(["BOT"]) & ~c("black_title").fill_null("").is_in(["BOT"]))
    d = d.with_columns(t=c("utc").str.strptime(pl.Datetime, "%Y.%m.%d %H:%M:%S", strict=False),
                       inc=c("time_control").str.split("+").list.get(1).cast(pl.Float64),
                       sw=pl.when(c("result") == "1-0").then(1.0).when(c("result") == "0-1").then(0.0).otherwise(0.5))
    out = []
    for col, opp, k in (("white", "black", 0), ("black", "white", 1)):
        x = d.select("game_id", "t", "plies", "time_control", "termination", "clocks", "inc",
                     player=c(col).str.to_lowercase(), me=c(f"{col}_elo"), op=c(f"{opp}_elo"),
                     score=c("sw") if k == 0 else 1 - c("sw"), side=pl.lit(k))
        out.append(x)
    s = pl.concat(out)
    # mean think per own move: own clock at consecutive own moves, plus the increment
    th = []
    for clocks, side, inc in zip(s["clocks"].to_list(), s["side"].to_list(), s["inc"].to_list()):
        if clocks is None or len(clocks) < side + 6:
            th.append(None)
            continue
        a = np.asarray(clocks[side::2], float)
        th.append(float(np.clip(a[:-1] - a[1:] + inc, 0, 300)[1:].mean()))
    s = s.with_columns(think_mean=pl.Series(th, dtype=pl.Float64)).drop("clocks")
    s = s.with_columns(exp=1 / (1 + 10 ** ((c("op") - c("me")) / 400)), year=pl.lit(int(p.name[:4])),
                       head_end=c("t").max())
    return s.with_columns(resid=c("score") - c("exp"), flag_loss=(c("termination") == "Time forfeit") & (c("score") == 0),
                          early_loss=(c("plies") <= 40) & (c("score") == 0))


def within(df: pl.DataFrame, y: str, xs: list[str]) -> dict:
    """Least squares of y on xs after demeaning everything per player (fixed effects); SE clustered-free OLS."""
    d = df.select("player", y, *xs).drop_nulls()
    d = d.with_columns([(c(v).cast(pl.Float64) - c(v).cast(pl.Float64).mean().over("player")).alias(v) for v in [y] + xs])
    Y = d[y].to_numpy()
    X = d.select(xs).to_numpy()
    keep = np.abs(X).sum(1) > 0            # rows with no within-player variation carry no information
    Y, X = Y[keep], X[keep]
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    res = Y - X @ beta
    dof = max(1, len(Y) - X.shape[1])
    cov = np.linalg.pinv(X.T @ X) * (res @ res) / dof
    se = np.sqrt(np.diag(cov))
    return {x: {"coef": float(b), "se": float(s_), "z": float(b / s_) if s_ > 0 else 0.0} for x, b, s_ in zip(xs, beta, se)} | {"n": int(len(Y))}


def build_pairs(s: pl.DataFrame, order: str) -> pl.DataFrame:
    """Pairs (game, the player's previous game in `order`) within a session; previous-game variables attached."""
    o = s.sort("session", order)
    o = o.with_columns(prev_score=c("score").shift(1).over("session"), prev2_score=c("score").shift(2).over("session"),
                       prev_flag=c("flag_loss").shift(1).over("session"),
                       prev_tc=c("time_control").shift(1).over("session"),
                       k=pl.int_range(pl.len()).over("session"))
    return o.filter(c("k") >= 1).with_columns(
        after_loss=(c("prev_score") == 0).cast(pl.Float64), after_draw=(c("prev_score") == 0.5).cast(pl.Float64),
        after_flag_loss=c("prev_flag").cast(pl.Float64), same_tc=(c("time_control") == c("prev_tc")).cast(pl.Float64),
        has2=(c("k") >= 2),
        two_losses=((c("prev_score") == 0) & (c("prev2_score") == 0) & (c("k") >= 2)).cast(pl.Float64),
        two_wins=((c("prev_score") == 1) & (c("prev2_score") == 1) & (c("k") >= 2)).cast(pl.Float64))


def main():
    s = pl.concat([sides_of(p) for p in sorted(LAKE.glob("????-??")) if int(p.name[:4]) >= 2018])
    gap = lambda a, b: (a - b).dt.total_seconds() / 60
    # Sessions: a player's games chained by gaps <= GAP_MIN between consecutive starts.
    s = s.sort("player", "t").with_columns(prev_t=c("t").shift(1).over("player"), nxt_t=c("t").shift(-1).over("player"))
    s = s.with_columns(new=c("prev_t").is_null() | (gap(c("t"), c("prev_t")) > GAP_MIN))
    s = s.with_columns(session=c("new").cum_sum(),
                       cont=pl.when(gap(c("head_end"), c("t")) >= GAP_MIN)
                       .then(c("nxt_t").is_not_null() & (gap(c("nxt_t"), c("t")) <= GAP_MIN)))
    s = s.with_columns(n_sess=pl.len().over("session")).filter(c("n_sess") >= 2)
    pairs = build_pairs(s, "t")
    print(f"{s.height:,} sides in {s['session'].n_unique():,} sessions of 2+ games, {pairs.height:,} pairs", flush=True)
    # Null: permute game order within each session (each game's record intact), rebuild pairs. This keeps the
    # overlap that biases a within-session estimator (a game's score is one pair's outcome and the next pair's
    # regressor) and removes only the time order (Miller & Sanjurjo).
    rng = np.random.default_rng(0)
    nulls = [build_pairs(s.with_columns(r=pl.Series(rng.random(s.height))), "r") for _ in range(N_PERM)]
    res = {"sides": s.height, "pairs": pairs.height, "gap_min": GAP_MIN, "n_perm": N_PERM, "outcomes": {}}

    def est(fn, y, xs):
        obs = within(fn(pairs).with_columns(player=c("session")), y, xs)
        nul = [within(fn(p_).with_columns(player=c("session")), y, xs) for p_ in nulls]
        for x in xs:
            b0 = float(np.mean([n_[x]["coef"] for n_ in nul]))
            obs[x].update(null=b0, effect=obs[x]["coef"] - b0,
                          z_effect=(obs[x]["coef"] - b0) / obs[x]["se"] if obs[x]["se"] > 0 else 0.0)
        return obs

    for y in ["resid", "think_mean", "plies", "flag_loss", "early_loss", "same_tc"]:
        r = est(lambda f: f, y, ["after_loss", "after_draw"])
        r_fl = est(lambda f: f.filter(c("prev_score") == 0), y, ["after_flag_loss"])
        r_st = est(lambda f: f.filter(c("has2")), y, ["two_losses", "two_wins"])
        res["outcomes"][y] = {"vs_after_win": r, "flag_vs_other_loss": r_fl, "streaks_vs_mixed": r_st}
        f_ = lambda d, k: f"{d[k]['effect']:+.4f} (raw {d[k]['coef']:+.4f}, null {d[k]['null']:+.4f}, z {d[k]['z_effect']:+.1f})"
        print(f"{y:11s} after loss {f_(r, 'after_loss')} | flag-loss vs other {f_(r_fl, 'after_flag_loss')} | "
              f"2 losses {f_(r_st, 'two_losses')} | 2 wins {f_(r_st, 'two_wins')}", flush=True)
    # Continuation (no overlap problem: the regressor is this game's result, the outcome is whether a next game
    # starts): within-player, after a loss vs after a win.
    cs = s.filter(c("cont").is_not_null()).with_columns(lost=(c("score") == 0).cast(pl.Float64),
                                                        drew=(c("score") == 0.5).cast(pl.Float64),
                                                        cont=c("cont").cast(pl.Float64))
    rc = within(cs, "cont", ["lost", "drew"])
    raw = cs.group_by("score").agg(cont=c("cont").mean(), n=pl.len()).sort("score")
    res["continuation"] = {"within_player": rc, "raw": raw.to_dicts()}
    print(f"continue within {GAP_MIN} min: after loss {rc['lost']['coef']:+.4f} (z {rc['lost']['z']:+.1f}) vs after win; raw "
          + ", ".join(f"score {r['score']}: {r['cont']:.3f}" for r in raw.to_dicts()), flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "tilt.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()

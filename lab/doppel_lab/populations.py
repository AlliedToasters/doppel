"""Population contrasts on the Lichess lake, rating-matched (CPU, no engine).

    python -m doppel_lab.populations lichess

Contrasts (per side unless noted; all within blitz and rapid separately):
  arena    : side's game is an arena tournament game vs a lobby game (competitive-vs-casual proxy; the
             public dumps hold rated games only, so true casual games aren't available).
  titled   : titled human (GM..WNM, not LM/BOT) vs untitled human.
  bot      : BOT account vs human.
Matching: coarsened exact matching on (year, time control class, base+inc, 50-Elo bin of own rating,
100-Elo bin of opponent rating). Effect = treated mean minus the control mean of the same stratum,
averaged over treated rows; SE from the per-stratum variances (treated rows independent).
Features per side: score minus Elo expectation, flag-loss, think mean and CV (clocks, 2017-04 on),
time-trouble share (own clock < 10% of base), weird-move rate (own-era book, self-excluded), and the
game's own-era book exit (T1) and plies. Berserked arena games (start clock below base) are dropped,
since berserk halves the clock.
Output: lab/results/populations_lichess.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

from doppel_lab.book import move_ids, prefix_hashes
from doppel_lab.vintage import tc_class

ROOT = Path(__file__).resolve().parents[2]
LAKE = ROOT / "data" / "lake"
RESULTS = ROOT / "lab" / "results"
CAP = 40
HUMAN_TITLES = {"GM", "IM", "FM", "CM", "NM", "WGM", "WIM", "WFM", "WCM", "WNM"}
FEATS = ["score_minus_exp", "flag_loss", "think_mean", "think_cv", "trouble_share", "weird_rate", "t1", "plies"]


def side_clock_feats(clocks, base: int, inc: int):
    """(think mean, think cv, trouble share, berserk) for white and black."""
    nan4 = (np.nan, np.nan, np.nan)
    if clocks is None or len(clocks) < 6:
        return nan4, nan4, False
    a = np.asarray(clocks, float)
    berserk = a[0] < 0.75 * base or (len(a) > 1 and a[1] < 0.75 * base)
    out = []
    for s in (0, 1):
        c = a[s::2]
        spent = np.clip(c[:-1] - c[1:] + inc, 0, 300)[1:]
        if len(spent) < 3:
            out.append(nan4)
            continue
        m = spent.mean()
        out.append((m, spent.std() / m if m > 0 else np.nan, float((c < 0.1 * base).mean())))
    return out[0], out[1], berserk


def year_sides(path: Path) -> pl.DataFrame:
    df = pl.read_parquet(path / "*.parquet").filter(
        pl.col("fen").is_null() & pl.col("white_elo").is_not_null() & pl.col("black_elo").is_not_null()
        & pl.col("time_control").str.contains(r"^\d+\+\d+$") & pl.col("result").is_in(["1-0", "0-1", "1/2-1/2"]))
    tcs = pl.col("time_control").str.split("+")
    df = df.with_columns(base=tcs.list.get(0).cast(pl.Int32), inc=tcs.list.get(1).cast(pl.Int32))
    df = df.with_columns(tc=tc_class(pl.col("base"), pl.col("inc"))).filter(pl.col("tc").is_in(["blitz", "rapid"]))
    # own-era book, self-excluded; weird move per ply, then split by side (even ply = white)
    ids, off = move_ids(df["moves"])
    keys, cnt = prefix_hashes(ids, off, cap=CAP)
    _, inv, c = np.unique(keys, return_inverse=True, return_counts=True)
    other = c[inv] - 1
    o2 = np.zeros(len(cnt) + 1, np.int64)
    np.cumsum(cnt, out=o2[1:])
    start = o2[:-1]
    pos = np.arange(len(keys)) - np.repeat(start, cnt)
    big = np.iinfo(np.int64).max
    f = np.minimum.reduceat(np.where(other < 1, pos, big), start)
    t1 = np.where(f == big, cnt, f) + 1
    par = np.empty(len(keys))
    par[1:] = other[:-1]
    par[start] = len(cnt) - 1
    ok = par >= 100
    wd = ok & (other < 0.01 * par)
    white = pos % 2 == 0
    red = lambda m: np.add.reduceat(m.astype(np.int64), start)
    el = {0: red(ok & white), 1: red(ok & ~white)}
    wr = {0: red(wd & white), 1: red(wd & ~white)}
    year = int(path.name[:4])
    clk = [side_clock_feats(c_, b, i) for c_, b, i in zip(df["clocks"].to_list(), df["base"].to_list(),
                                                          df["inc"].to_list())]
    arena = df["event"].str.contains("tournament").to_numpy()
    res = df["result"].to_numpy()
    score_w = np.where(res == "1-0", 1.0, np.where(res == "0-1", 0.0, 0.5))
    term = df["termination"].to_numpy()
    rows = []
    for s, col, opp in ((0, "white", "black"), (1, "black", "white")):
        me, op = df[f"{col}_elo"].to_numpy(), df[f"{opp}_elo"].to_numpy()
        exp = 1 / (1 + 10 ** ((op - me) / 400))
        score = score_w if s == 0 else 1 - score_w
        cf = np.array([x[s] for x in clk], float)
        rows.append(pl.DataFrame({
            "year": np.full(df.height, year), "tc": df["tc"], "base": df["base"], "inc": df["inc"],
            "elo": me, "opp_elo": op, "title": df[f"{col}_title"], "opp_title": df[f"{opp}_title"],
            "arena": arena, "berserk": np.array([x[2] for x in clk]),
            "score_minus_exp": score - exp, "flag_loss": (term == "Time forfeit") & (score == 0),
            "think_mean": cf[:, 0], "think_cv": cf[:, 1], "trouble_share": cf[:, 2],
            "weird_rate": np.where(el[s] > 0, wr[s] / np.maximum(el[s], 1), np.nan),
            "t1": t1.astype(float), "plies": df["plies"].to_numpy().astype(float)}))
    return pl.concat(rows)


def cem(sides: pl.DataFrame, treat: pl.Expr, pool: pl.Expr, name: str) -> dict:
    """Coarsened exact matching; per feature: ATT, SE, treated n, matched strata."""
    d = sides.filter(pool).with_columns(t=treat, eb=(pl.col("elo") // 50), ob=(pl.col("opp_elo") // 100))
    keys = ["year", "tc", "base", "inc", "eb", "ob"]
    out = {"name": name}
    for tc in ("blitz", "rapid"):
        dd = d.filter(pl.col("tc") == tc)
        r = {}
        for f in FEATS:
            x = dd.with_columns(pl.col(f).cast(pl.Float64)).filter(pl.col(f).is_not_nan() & pl.col(f).is_not_null())
            g = x.group_by(keys + ["t"]).agg(m=pl.col(f).mean(), v=pl.col(f).var(), n=pl.len())
            tr, co = g.filter(pl.col("t")), g.filter(~pl.col("t"))
            j = tr.join(co, on=keys, suffix="_c").filter(pl.col("n_c") >= 2)
            if j.height == 0 or j["n"].sum() < 30:
                r[f] = None
                continue
            w = j["n"].to_numpy().astype(float)
            diff = (j["m"] - j["m_c"]).to_numpy()
            att = float((w * diff).sum() / w.sum())
            # Var of the weighted mean: treated var/n per stratum + control var/n_c, weighted by n^2.
            var_s = (j["v"].fill_null(0) / j["n"] + j["v_c"].fill_null(0) / j["n_c"]).to_numpy()
            se = float(np.sqrt((w ** 2 * var_s).sum()) / w.sum())
            ctrl = float((w * j["m_c"].to_numpy()).sum() / w.sum())
            r[f] = {"att": att, "se": se, "control_mean": ctrl, "treated_n": int(w.sum()), "strata": j.height}
        out[tc] = r
    return out


def lichess():
    sides = pl.concat([year_sides(p) for p in sorted((LAKE / "lichess").glob("????-??"))])
    sides = sides.filter(~(pl.col("arena") & pl.col("berserk")))
    human = ~pl.col("title").fill_null("").is_in(["BOT", "LM"])
    opp_human = ~pl.col("opp_title").fill_null("").is_in(["BOT"])
    res = {"sides": sides.height, "contrasts": [
        cem(sides, pl.col("arena"), human & opp_human, "arena_vs_lobby"),
        cem(sides, pl.col("title").fill_null("").is_in(list(HUMAN_TITLES)),
            human & opp_human & (pl.col("elo") >= 1900), "titled_vs_untitled"),
        cem(sides, pl.col("title") == "BOT", opp_human & (pl.col("title").fill_null("") != "LM"), "bot_vs_human"),
    ]}
    for c in res["contrasts"]:
        for tc in ("blitz", "rapid"):
            line = []
            for f, v in c[tc].items():
                if v:
                    z = v["att"] / v["se"] if v["se"] > 0 else 0
                    line.append(f"{f} {v['att']:+.3f} (ctrl {v['control_mean']:.3f}, z {z:+.1f}, n {v['treated_n']})")
            print(f"{c['name']} [{tc}]\n   " + "\n   ".join(line), flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "populations_lichess.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    {"lichess": lichess}[sys.argv[1]]()

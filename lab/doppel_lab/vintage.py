"""Vintage: has online play changed across Lichess eras, and does the change differ by tier? (CPU, no engine)

    python -m doppel_lab.vintage lichess

Input: data/lake/lichess/<YYYY-03> (lake_fetch vintage: the first ~200k rated games of each March).
Every year is subsampled to the same N games so book statistics (which grow with corpus size) compare.
Tier = within-year percentile of the game's mean rating (rating pools drift, so raw Elo doesn't compare
across years); raw medians are reported alongside.

Per game (surface features, no Stockfish):
  plies, decisive, flag (time forfeit), first move, think time (clocks exist from 2017-04).
  own-era book: prefix-hash counts in the same year's sample, self-excluded; T1/T10 exit ply,
    weird-move rate (q = c_k / c_{k-1} < 1% with parent >= 100 other games).
  modern book: T10 exit against the 2026-10 blitz table (data/v0/book/table_*), i.e. conformity to
    today's lines.
  opening diversity: entropy (bits) of the 6-ply line distribution.
Output: lab/results/vintage_lichess.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

from doppel_lab.book import move_ids, prefix_hashes

ROOT = Path(__file__).resolve().parents[2]
LAKE = ROOT / "data" / "lake"
BOOK = ROOT / "data" / "v0" / "book"
RESULTS = ROOT / "lab" / "results"
N = 150_000
CAP = 40
TIERS = [(0, 25, "p00-25"), (25, 75, "p25-75"), (75, 95, "p75-95"), (95, 101, "p95+")]


def tc_class(base: pl.Expr, inc: pl.Expr) -> pl.Expr:
    est = base + 40 * inc
    return (pl.when(est < 180).then(pl.lit("bullet")).when(est < 480).then(pl.lit("blitz"))
            .when(est < 1500).then(pl.lit("rapid")).otherwise(pl.lit("classical")))


def load_year(path: Path, rng) -> pl.DataFrame:
    df = pl.read_parquet(path / "*.parquet").filter(
        pl.col("fen").is_null() & pl.col("white_elo").is_not_null() & pl.col("black_elo").is_not_null()
        & pl.col("time_control").str.contains(r"^\d+\+\d+$"))
    if df.height > N:
        df = df[np.sort(rng.choice(df.height, N, replace=False))]
    tc = pl.col("time_control").str.split("+")
    df = df.with_columns(base=tc.list.get(0).cast(pl.Int32), inc=tc.list.get(1).cast(pl.Int32),
                         elo=(pl.col("white_elo") + pl.col("black_elo")) / 2)
    return df.with_columns(tc=tc_class(pl.col("base"), pl.col("inc")),
                           pct=pl.col("elo").rank("average") / pl.len() * 100)


def book_feats(keys: np.ndarray, cnt: np.ndarray, ref_keys: np.ndarray, ref_cnt: np.ndarray):
    """Per game: own-era T1/T10 exit, weird moves / eligible plies, modern-book T10 exit, 6-ply line key."""
    uk, inv, c = np.unique(keys, return_inverse=True, return_counts=True)
    other = c[inv] - 1                                   # self-excluded count of each (game, ply) key
    i = np.searchsorted(ref_keys, keys).clip(max=len(ref_keys) - 1)
    modern = np.where(ref_keys[i] == keys, ref_cnt[i], 0)
    off = np.zeros(len(cnt) + 1, np.int64)
    np.cumsum(cnt, out=off[1:])
    start = off[:-1]
    pos = np.arange(len(keys)) - np.repeat(start, cnt)       # ply index within its game
    big = np.iinfo(np.int64).max

    def first(mask):  # first ply index where mask holds, else the game's length
        f = np.minimum.reduceat(np.where(mask, pos, big), start)
        return np.where(f == big, cnt, f).astype(float)

    t1, t10, mt10 = first(other < 1), first(other < 10), first(modern < 10)
    par = np.empty(len(keys))
    par[1:] = other[:-1]
    par[start] = len(cnt) - 1                                 # first ply: parent is the start position (every other game)
    ok = par >= 100
    elig = np.add.reduceat(ok.astype(np.int64), start).astype(float)
    weird = np.add.reduceat((ok & (other < 0.01 * par)).astype(np.int64), start).astype(float)
    line6 = np.where(cnt >= 6, keys[np.minimum(start + 5, len(keys) - 1)], np.uint64(0))
    return t1 + 1, t10 + 1, mt10 + 1, weird, elig, line6


def entropy_bits(x: np.ndarray) -> float:
    _, c = np.unique(x, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def think_median(clocks: pl.Series, inc: np.ndarray) -> np.ndarray:
    """Median seconds spent per move (both sides pooled), from clocks; NaN without clocks."""
    out = np.full(len(clocks), np.nan)
    for g, c in enumerate(clocks):
        if c is None or len(c) < 4:
            continue
        a = np.asarray(c, float)
        spent = np.concatenate([a[:-2] - a[2:]]) + inc[g]   # same side, consecutive own moves
        out[g] = np.clip(spent[2:], 0, 120).mean() if len(spent) > 2 else np.nan  # clocks are whole seconds: mean, not median
    return out


def year_table(df: pl.DataFrame, ref_keys, ref_cnt) -> pl.DataFrame:
    ids, off = move_ids(df["moves"])
    keys, cnt = prefix_hashes(ids, off, cap=CAP)
    t1, t10, mt10, weird, elig, line6 = book_feats(keys, cnt, ref_keys, ref_cnt)
    tier = np.empty(df.height, object)
    pct = df["pct"].to_numpy()
    for lo, hi, name in TIERS:
        tier[(pct >= lo) & (pct < hi)] = name
    return df.select("elo", "pct", "tc", "plies", "result", "termination").with_columns(
        tier=pl.Series(tier.astype(str)), t1=t1, t10=t10, modern_t10=mt10, weird=weird, elig=elig,
        line6=pl.Series(line6), first=df["moves"].list.first(),
        think=think_median(df["clocks"], df["inc"].to_numpy()),
        decisive=pl.col("result").is_in(["1-0", "0-1"]), flag=pl.col("termination") == "Time forfeit")


def summarise(t: pl.DataFrame) -> dict:
    t = t.with_columns(pl.col("think").fill_nan(None))
    return {"games": t.height, "elo_median": float(t["elo"].median()),
            "plies_median": float(t["plies"].median()), "decisive": float(t["decisive"].mean()),
            "flag": float(t["flag"].mean()), "t1_mean": float(t["t1"].mean()), "t10_mean": float(t["t10"].mean()),
            "modern_t10_mean": float(t["modern_t10"].mean()),
            "weird_rate": float(t["weird"].sum() / max(1, t["elig"].sum())),
            "open6_entropy_bits": entropy_bits(t["line6"].to_numpy()[t["plies"].to_numpy() >= 6]),
            "e4": float((t["first"] == "e2e4").mean()), "d4": float((t["first"] == "d2d4").mean()),
            "think_mean_s": (float(t["think"].median()) if t["think"].null_count() < t.height else None),
            "clock_share": float(1 - t["think"].null_count() / t.height)}


def lichess():
    rng = np.random.default_rng(0)
    ref_keys, ref_cnt = np.load(BOOK / "table_keys.npy"), np.load(BOOK / "table_counts.npy")
    res = {"N_per_year": N, "years": {}}
    for p in sorted((LAKE / "lichess").glob("????-??")):
        df = load_year(p, rng)
        t = year_table(df, ref_keys, ref_cnt)
        y = {"all": summarise(t), "tc_mix": dict(t["tc"].value_counts().iter_rows())}
        # Blitz only, by tier: the tc mix moved a lot (early Lichess was slower), so tiers compare within blitz.
        b = t.filter(pl.col("tc") == "blitz")
        y["blitz"] = summarise(b)
        y["blitz_by_tier"] = {name: summarise(b.filter(pl.col("tier") == name)) for _, _, name in TIERS}
        res["years"][p.name] = y
        a, bl = y["all"], y["blitz"]
        print(f"{p.name} n={a['games']} elo~{a['elo_median']:.0f} blitz={bl['games']} T1={bl['t1_mean']:.2f} "
              f"T10={bl['t10_mean']:.2f} modernT10={bl['modern_t10_mean']:.2f} weird={bl['weird_rate']:.3f} "
              f"H6={bl['open6_entropy_bits']:.2f} e4={bl['e4']:.2f} plies={bl['plies_median']:.0f} "
              f"flag={bl['flag']:.3f} think={bl['think_mean_s']}", flush=True)
        print("   " + " | ".join(f"{k}: elo~{v['elo_median']:.0f} T1={v['t1_mean']:.2f} mT10={v['modern_t10_mean']:.2f} "
                                f"weird={v['weird_rate']:.3f} flag={v['flag']:.2f}"
                                for k, v in y["blitz_by_tier"].items()), flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "vintage_lichess.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    {"lichess": lichess}[sys.argv[1]]()

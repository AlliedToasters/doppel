"""Encoder benchmark, surface side: the scores a learned match encoder has to beat (SCIENCE §2, §8). CPU.

    python -m doppel_lab.bench build      # task table + surface features -> data/bench/
    python -m doppel_lab.bench surface    # 5-fold CV gradient-boosted trees -> lab/results/bench_surface.json

The rule: every model sees only the game's move and clock stream, never headers (ratings, titles, event,
dates). Surface features are the standard T0 side aggregates (game_science.AGG, from features.table.game_rows
without evals), White minus Black differences, plies, and book features against the fixed 2026-10 blitz table
(T1/T10 exit, per-side weird-move counts). An encoder (Maia embeddings, the Doppel z) is scored on the same
rows and the same folds, which bench build writes out with the labels.

Games: Lichess lake 2018-03 .. 2026-03 (every ply clocked), standard start, blitz and rapid, at least 20 plies.
Tasks (label: metric):
  rating_pct   within-year percentile of the mean rating          regression: MAE, R²
  elo_mean     raw mean rating                                    regression: MAE, R²
  year         2018..2026                                         regression: MAE, R²; and AUC of 2023+ vs <=2020
  top5         mean rating in the year's top 5%                   AUC
  arena        arena tournament game vs lobby game                AUC
  titled       a titled human plays (only games with rating_pct >= 90, where titled players are)   AUC
  rapid        rapid vs blitz (sanity check: the clock gives it away)   AUC
"""

from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import polars as pl

from doppel_lab.book import move_ids, prefix_hashes
from doppel_lab.vintage import tc_class

ROOT = Path(__file__).resolve().parents[2]
LAKE = ROOT / "data" / "lake" / "lichess"
BENCH = ROOT / "data" / "bench"
BOOK = ROOT / "data" / "v0" / "book"
RESULTS = ROOT / "lab" / "results"
PER_YEAR = 8_000
TWIC_N = 80_000
TITLED_PER_YEAR = 1_500
HUMAN_TITLES = ["GM", "IM", "FM", "CM", "NM", "WGM", "WIM", "WFM", "WCM", "WNM"]
SEED = 0
c = pl.col


def suite_dir(suite: str) -> Path:
    """lichess lives at data/bench/ (the first suite); others in data/bench/<suite>/."""
    return BENCH if suite == "lichess" else BENCH / suite


def sample_twic() -> pl.DataFrame:
    """TWIC 2012-2026 with both FIDE ratings, >= 20 plies; no clocks. One random sample, labels per task."""
    from doppel_lab.twic_science import load, twic_files
    rng = np.random.default_rng(SEED)
    df = load().filter(c("elo").is_not_null() & (c("plies") >= 20))
    df = df[np.sort(rng.choice(df.height, min(TWIC_N, df.height), replace=False))]
    df = df.with_columns(white=pl.lit(None, pl.String), black=pl.lit(None, pl.String))
    t = pl.read_parquet(twic_files(), columns=["game_id", "white", "black"]).unique("game_id")
    df = df.drop("white", "black").join(t, on="game_id", how="left")
    df = df.with_columns(clocks=pl.lit(None, pl.List(pl.Float32)), base=pl.lit(None, pl.Int32),
                         inc=pl.lit(None, pl.Int32), elo_mean=c("elo"), in_main=pl.lit(True),
                         fold=(c("white").fill_null("").str.to_lowercase().hash(SEED) % 5).cast(pl.Int8))
    return df.select("game_id", "year", "white", "black", "white_elo", "black_elo", "base", "inc", "moves", "clocks",
                     "elo_mean", "medium", "speed", "age", "in_main", "fold")


def sample() -> pl.DataFrame:
    rng = np.random.default_rng(SEED)
    out = []
    for p in sorted(LAKE.glob("????-??")):
        y = int(p.name[:4])
        if y < 2018:
            continue
        df = pl.read_parquet(p / "*.parquet").filter(
            c("fen").is_null() & c("white_elo").is_not_null() & c("black_elo").is_not_null()
            & c("clocks").is_not_null() & (c("plies") >= 20) & c("time_control").str.contains(r"^\d+\+\d+$")
            & ~c("white_title").fill_null("").is_in(["BOT"]) & ~c("black_title").fill_null("").is_in(["BOT"]))
        tcs = c("time_control").str.split("+")
        df = df.with_columns(base=tcs.list.get(0).cast(pl.Int32), inc=tcs.list.get(1).cast(pl.Int32),
                             elo_mean=(c("white_elo") + c("black_elo")) / 2)
        df = df.with_columns(tc=tc_class(c("base"), c("inc"))).filter(c("tc").is_in(["blitz", "rapid"]))
        df = df.with_columns(rating_pct=c("elo_mean").rank("average") / pl.len() * 100, year=pl.lit(y),
                             titled=c("white_title").fill_null("").is_in(HUMAN_TITLES)
                             | c("black_title").fill_null("").is_in(HUMAN_TITLES),
                             arena=c("event").str.contains("tournament"))
        main = df[np.sort(rng.choice(df.height, min(PER_YEAR, df.height), replace=False))].with_columns(
            in_main=pl.lit(True))
        hi = df.filter((c("rating_pct") >= 90) & ~c("game_id").is_in(main["game_id"].to_list()))
        tit = hi.filter(c("titled"))
        ctl = hi.filter(~c("titled"))
        extra = pl.concat([tit.head(TITLED_PER_YEAR), ctl[rng.choice(ctl.height, min(ctl.height, TITLED_PER_YEAR),
                                                                    replace=False)]]).with_columns(in_main=pl.lit(False))
        out.append(pl.concat([main, extra]))
    g = pl.concat(out)
    # Folds grouped by White's account, so one player's games never straddle train and test.
    g = g.with_columns(fold=(c("white").str.to_lowercase().hash(SEED) % 5).cast(pl.Int8))
    return g.select("game_id", "year", "white", "black", "white_elo", "black_elo", "base", "inc", "tc", "moves",
                    "clocks", "elo_mean", "rating_pct", "titled", "arena", "in_main", "fold")


def _task(row):
    from doppel_lab.features.table import game_rows
    return game_rows((row, None, None))


def book_cols(g: pl.DataFrame) -> pl.DataFrame:
    ref_k, ref_c = np.load(BOOK / "table_keys.npy"), np.load(BOOK / "table_counts.npy")
    ids, off = move_ids(g["moves"])
    keys, cnt = prefix_hashes(ids, off, cap=40)
    i = np.searchsorted(ref_k, keys).clip(max=len(ref_k) - 1)
    m = np.where(ref_k[i] == keys, ref_c[i], 0).astype(float)
    o = np.zeros(len(cnt) + 1, np.int64)
    np.cumsum(cnt, out=o[1:])
    start = o[:-1]
    pos = np.arange(len(keys)) - np.repeat(start, cnt)
    big = np.iinfo(np.int64).max
    first = lambda mask: np.where((f := np.minimum.reduceat(np.where(mask, pos, big), start)) == big, cnt, f) + 1
    par = np.empty(len(keys))
    par[1:] = m[:-1]
    par[start] = 22_700_000.0                       # the start position: every game in the reference corpus
    ok = par >= 100
    wd = ok & (m < 0.01 * par)
    red = lambda x: np.add.reduceat(x.astype(np.int64), start)
    white = pos % 2 == 0
    logq = np.where(ok, np.log10(np.maximum(m, 1) / par), 0.0)
    return pl.DataFrame({"game_id": g["game_id"], "book_t1": first(m < 2), "book_t10": first(m < 10),
                         "book_t100": first(m < 100), "w_weird": red(wd & white), "b_weird": red(wd & ~white),
                         "w_book_logq": np.add.reduceat(np.where(white, logq, 0), start),
                         "b_book_logq": np.add.reduceat(np.where(~white, logq, 0), start)})


def build(suite: str = "lichess"):
    from doppel_lab.features.table import _concat
    from doppel_lab.game_science import AGG
    t = time.time()
    g = sample() if suite == "lichess" else sample_twic()
    out = suite_dir(suite)
    print(f"{g.height:,} games sampled ({g['in_main'].sum():,} main) in {time.time() - t:.0f}s", flush=True)
    rows = g.select("game_id", "white_elo", "black_elo", "base", "inc", "moves", "clocks").to_dicts()
    with Pool(12) as p:
        parts = p.map(_task, rows, chunksize=64)
    ply = _concat(parts)
    ply = ply.with_columns([c(k).fill_nan(None) for k, ty in ply.schema.items() if ty.is_float()])
    # T0 aggregates whose source columns exist (TWIC has no clocks, so no think/clock columns)
    t0 = {k: e for k, (tier, e, _) in AGG.items() if tier == "T0" and set(e.meta.root_names()) <= set(ply.columns)}
    sides = (ply.sort("game_id", "ply").group_by("game_id", "mover", maintain_order=True)
             .agg([e.alias(k) for k, e in t0.items()])).with_columns([c(k).cast(pl.Float64) for k in t0])
    w = sides.filter(c("mover") == 0).select("game_id", *[c(k).alias(f"w_{k}") for k in t0])
    b = sides.filter(c("mover") == 1).select("game_id", *[c(k).alias(f"b_{k}") for k in t0])
    feats = (g.select("game_id", plies=c("moves").list.len()).join(w, on="game_id").join(b, on="game_id")
             .with_columns([(c(f"w_{k}") - c(f"b_{k}")).alias(f"d_{k}") for k in t0])
             .join(book_cols(g), on="game_id"))
    out.mkdir(parents=True, exist_ok=True)
    g.drop("moves", "clocks").write_parquet(out / "tasks.parquet")
    g.select("game_id", "moves", "clocks", "base", "inc").write_parquet(out / "games.parquet")
    feats.write_parquet(out / "surface.parquet")
    print(f"surface features: {feats.width - 1} columns for {feats.height:,} games, {time.time() - t:.0f}s", flush=True)


TASKS = {  # name: (filter on tasks, label expression, kind)
    "rating_pct": (c("in_main"), c("rating_pct"), "reg"),
    "elo_mean": (c("in_main"), c("elo_mean"), "reg"),
    "year": (c("in_main"), c("year").cast(pl.Float64), "reg"),
    "era_2023plus": (c("in_main") & ((c("year") >= 2023) | (c("year") <= 2020)), c("year") >= 2023, "clf"),
    "top5": (c("in_main"), c("rating_pct") >= 95, "clf"),
    "arena": (c("in_main"), c("arena"), "clf"),
    "titled": (c("rating_pct") >= 90, c("titled"), "clf"),
    "rapid": (c("in_main"), c("tc") == "rapid", "clf"),
}


TASKS_TWIC = {  # move-only: TWIC has no clocks
    "elo_fide": (c("in_main"), c("elo_mean"), "reg"),
    "year": (c("in_main"), c("year").cast(pl.Float64), "reg"),
    "era_2021plus": ((c("year") >= 2021) | (c("year") <= 2016), c("year") >= 2021, "clf"),
    "online_fast": (c("speed") == "fast", c("medium") == "online", "clf"),
    "junior": ((c("medium") == "otb") & (c("speed") == "classical") & (c("age") != "senior") & (c("elo_mean") < 2300),
               c("age") == "junior", "clf"),
    "fast_otb": ((c("medium") == "otb") & (c("age") == "open"), c("speed") == "fast", "clf"),
}
SUITES = {"lichess": TASKS, "twic": TASKS_TWIC}


def surface(suite: str = "lichess"):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
    tasks = pl.read_parquet(suite_dir(suite) / "tasks.parquet")
    feats = pl.read_parquet(suite_dir(suite) / "surface.parquet")
    cols = [k for k in feats.columns if k != "game_id" and feats[k].null_count() < feats.height]
    d = tasks.join(feats, on="game_id")
    res = {"features": cols, "tasks": {}}
    for name, (flt, lab, kind) in SUITES[suite].items():
        x = d.filter(flt).with_columns(y=lab)
        X, y, fold = x.select(cols).to_numpy().astype(float), x["y"].to_numpy().astype(float), x["fold"].to_numpy()
        pred = np.empty(len(y))
        for k in range(5):
            tr, te = fold != k, fold == k
            if kind == "reg":
                m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=SEED)
                pred[te] = m.fit(X[tr], y[tr]).predict(X[te])
            else:
                m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, random_state=SEED)
                pred[te] = m.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        if kind == "reg":
            r = {"n": len(y), "mae": float(mean_absolute_error(y, pred)), "r2": float(r2_score(y, pred)),
                 "mae_const": float(np.abs(y - np.median(y)).mean())}
            print(f"{name:13s} n={len(y):6d} MAE {r['mae']:.2f} (constant {r['mae_const']:.2f}) R² {r['r2']:.3f}", flush=True)
        else:
            r = {"n": len(y), "pos_rate": float(y.mean()), "auc": float(roc_auc_score(y, pred))}
            print(f"{name:13s} n={len(y):6d} pos {r['pos_rate']:.3f} AUC {r['auc']:.3f}", flush=True)
        res["tasks"][name] = r
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / ("bench_surface.json" if suite == "lichess" else f"bench_surface_{suite}.json")).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    {"build": build, "surface": surface}[sys.argv[1]](*(sys.argv[2:3] or ["lichess"]))

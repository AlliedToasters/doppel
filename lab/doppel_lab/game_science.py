"""Game-level science on the human corpus with the registry features: how features vary with rating, how they
relate to each other, and how much rating a simple estimator recovers from them (CPU only).

Data: data/human/games_2026-10.parquet (36,000 Lichess blitz games, 6,000 in each of six 100-Elo bands 1100..2100;
both players in the band) and data/evals/human.parquet (Stockfish 20k nodes for 2,000 games per band). T1 (Maia-3)
is not stored for this corpus, so the tiers here are T0 (moves, clocks) and T2 (Stockfish).

Per ply: features.table.game_rows, the corpus feature table's own row builder, so names mean what the registry
says. Per game side: aggregates of those channels (AGG below). Per game: both sides plus their differences.

Analyses (lab/results/game_science.json):
  trends       per band mean and 95% bootstrap CI of every side aggregate, Spearman rho with rating,
               and the same split by time control (rating and time control are confounded on Lichess)
  structure    Spearman correlation between aggregates, raw and within band (ranks taken inside each band, which
               removes the shared dependence on rating)
  rating       gradient-boosted regression, 5-fold CV by game:
                 game_mean  mean rating of the two players, from both sides' aggregates
                 side       one player's rating from their own side only
                 delta      white - black rating, from the side differences, with and without the result
               each at T0 (all 36k games, and the same 12k as T2 for a fair comparison) and T2
  importance   permutation importance for the T2 game_mean model

    python -m doppel_lab.game_science build      # -> data/v0/features/human_f1_T2.parquet (per ply)
    python -m doppel_lab.game_science run        # -> lab/results/game_science.json
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import polars as pl

from doppel_lab.features import registry
from doppel_lab.features.table import game_rows

ROOT = Path(__file__).resolve().parents[2]
PLY = ROOT / f"data/v0/features/human_{registry.VERSION}_T2.parquet"
RESULTS = ROOT / "lab/results"
SEED = 0

# Side aggregates: name -> (tier, polars expression over that side's plies, doc).
c = pl.col


def _agg():
    T0 = {
        "n_moves": (pl.len(), "moves played"),
        "think_frac_mean": (c("think_frac").mean(), "mean logit think fraction"),
        "think_frac_std": (c("think_frac").std(), "spread of logit think fraction"),
        "think_s_median": (c("think_s").median(), "median think time, s"),
        "premove_share": ((c("think_s") < 0.5).mean(), "share of moves under 0.5 s"),
        "long_think_share": ((c("think_frac") > np.log(0.1 / 0.9)).mean(), "share of moves using >10% of the clock"),
        "tempo_jitter": (c("think_frac_roll_std").mean(), "mean rolling std of think fraction (tempo volatility)"),
        "clock_end": (c("clock_frac").last(), "clock fraction at the side's last move"),
        "time_trouble_share": ((c("clock_frac") < 0.1).mean(), "share of moves with <10% of base time left"),
        "capture_share": (c("capture").mean(), "share of moves that capture"),
        "check_share": (c("gives_check").mean(), "share of moves that give check"),
        "castled": (c("castle").max(), "castled at all"),
        "castle_ply": (c("ply").filter(c("castle") > 0).first(), "ply of castling"),
        "trade_intensity": (c("material_own_delta").abs().mean(), "mean |material change| between own moves"),
        "n_legal_mean": (c("n_legal").mean(), "mean legal moves faced (position openness)"),
        "in_check_share": (c("in_check").mean(), "share of positions faced in check"),
    }
    T2 = {
        "acpl": (c("cpl").mean(), "average centipawn loss (clipped 1000)"),
        "cpl_std": (c("cpl").std(), "spread of centipawn loss"),
        "cpl_median": (c("cpl").median(), "median centipawn loss"),
        "wc_loss_mean": (c("wc_loss").mean(), "mean win-chance loss"),
        "wc_loss_max": (c("wc_loss").max(), "worst single move, win-chance loss"),
        "blunder_rate": (c("blunder").mean(), "share of moves with wc loss >= 0.3"),
        "blunder2_rate": (c("blunder2").mean(), "share of moves with wc loss >= 0.2"),
        "first_blunder_move": (c("ply").filter(c("blunder") > 0).first() // 2, "move number of the first blunder"),
        "accuracy_jitter": (c("cpl_roll_std").mean(), "mean rolling std of CPL (accuracy volatility)"),
        "volatility": (c("wc_w_volatility").mean(), "mean Lichess win% volatility at own decisions"),
        "max_drawdown": (c("wc_w_drawdown_own").min(), "deepest drop from own best win% (win% points, <= 0)"),
        "eval_abs_mean": (c("cp_own").abs().mean(), "mean |eval| faced, cp (how lopsided the game is)"),
        "wc_own_mean": (c("wc_own").mean(), "mean own win chances faced"),
        "blunder_in_tt": (c("blunder").filter(c("clock_frac") < 0.1).mean(), "blunder rate in time trouble"),
        "blunder_calm": (c("blunder").filter(c("clock_frac") >= 0.25).mean(), "blunder rate with >=25% clock"),
    }
    return ({k: ("T0", e, d) for k, (e, d) in T0.items()} | {k: ("T2", e, d) for k, (e, d) in T2.items()})


AGG = _agg()
GAME = {  # whole-game features (same for both sides)
    "plies": "game length, plies",
    "lead_changes": "sign changes of White's win chances (|wc| > 0.1 either side)",
    "decisive": "game not drawn",
    "time_forfeit": "ended on time",
}


# --- build -------------------------------------------------------------------------------------------------------

def _task(args):
    row, ev = args
    return game_rows((row, None, ev))


def build(workers: int):
    g = pl.read_parquet(ROOT / "data/human/games_2026-10.parquet")
    ev = dict(pl.read_parquet(ROOT / "data/evals/human.parquet").iter_rows())
    rows = g.select("game_id", "white_elo", "black_elo", "base", "inc", "moves", "clocks").iter_rows(named=True)
    tasks = [(r, ev.get(r["game_id"])) for r in rows if r["moves"]]
    t = time.time()
    with Pool(workers) as p:
        parts = p.map(_task, tasks, chunksize=64)
    from doppel_lab.features.table import _concat
    df = _concat(parts)
    PLY.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(PLY)
    print(f"{len(tasks):,} games, {df.height:,} plies, {df.width} columns, {time.time() - t:.0f}s -> {PLY}")


# --- tables ------------------------------------------------------------------------------------------------------

def tables() -> tuple[pl.DataFrame, pl.DataFrame]:
    """(sides, games): one row per (game, side) and one per game."""
    ply = pl.read_parquet(PLY)
    ply = ply.with_columns([c(k).fill_nan(None) for k, t in ply.schema.items() if t.is_float()])  # mean() skips null, not NaN
    meta = pl.read_parquet(ROOT / "data/human/games_2026-10.parquet",
                           columns=["game_id", "band", "base", "inc", "result", "termination", "white_elo",
                                    "black_elo"])
    has_ev = ply.group_by("game_id").agg(c("cpl").is_not_null().any().alias("has_eval"))
    sides = (ply.sort("game_id", "ply").group_by("game_id", "mover", maintain_order=True)
             .agg([e.alias(k) for k, (_, e, _) in AGG.items()]))
    sides = sides.with_columns([c(k).cast(pl.Float64) for k in AGG])
    # whole-game features from the White-view eval series
    wcw = ply.with_columns(pl.when(c("mover") == 0).then(c("wc_own")).otherwise(-c("wc_own")).alias("wcw"))
    lead = (wcw.filter(c("wcw").abs() > 0.1).sort("game_id", "ply").group_by("game_id")
            .agg((c("wcw").sign().diff().abs() > 0).sum().alias("lead_changes")))
    plies = ply.group_by("game_id").agg(pl.len().alias("plies"))
    games = (meta.join(plies, on="game_id").join(has_ev, on="game_id").join(lead, on="game_id", how="left")
             .with_columns(decisive=(c("result") != "1/2-1/2").cast(pl.Float64),
                           time_forfeit=(c("termination") == "Time forfeit").cast(pl.Float64),
                           score_w=pl.when(c("result") == "1-0").then(1.0)
                           .when(c("result") == "0-1").then(0.0).otherwise(0.5),
                           elo_mean=(c("white_elo") + c("black_elo")) / 2,
                           elo_delta=(c("white_elo") - c("black_elo")).cast(pl.Float64),
                           tc=pl.format("{}+{}", c("base"), c("inc"))))
    games = games.with_columns(pl.when(c("has_eval")).then(c("lead_changes").fill_null(0)).alias("lead_changes"))
    sides = sides.join(games.select("game_id", "band", "tc", "base", "inc", "has_eval", "white_elo", "black_elo",
                                    "score_w"), on="game_id")
    sides = sides.with_columns(elo=pl.when(c("mover") == 0).then(c("white_elo")).otherwise(c("black_elo")),
                               score=pl.when(c("mover") == 0).then(c("score_w")).otherwise(1 - c("score_w")))
    w = sides.filter(c("mover") == 0).select("game_id", *[c(k).alias(f"w_{k}") for k in AGG])
    b = sides.filter(c("mover") == 1).select("game_id", *[c(k).alias(f"b_{k}") for k in AGG])
    games = games.join(w, on="game_id", how="left").join(b, on="game_id", how="left")
    games = games.with_columns([(c(f"w_{k}") - c(f"b_{k}")).alias(f"d_{k}") for k in AGG])
    return sides, games


# --- analyses ----------------------------------------------------------------------------------------------------

def boot_ci(x: np.ndarray, n: int = 1000, rng=None) -> list[float]:
    rng = rng or np.random.default_rng(SEED)
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return [float("nan")] * 2
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    ok = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[ok], b[ok]).statistic) if ok.sum() > 30 else float("nan")


def trends(sides: pl.DataFrame, games: pl.DataFrame) -> dict:
    out = {}
    bands = sorted(sides["band"].unique().to_list())
    feats = [(k, AGG[k][0], AGG[k][2], sides) for k in AGG] + [(k, "T2" if k == "lead_changes" else "T0", d, games)
                                                               for k, d in GAME.items()]
    for k, tier, doc, df in feats:
        d = df.filter(c("has_eval")) if tier == "T2" else df
        rows = []
        for bd in bands:
            x = d.filter(c("band") == bd)[k].cast(pl.Float64).to_numpy()
            x = x[np.isfinite(x)]
            rows.append({"band": bd, "n": int(len(x)), "mean": float(x.mean()) if len(x) else None,
                         "ci": boot_ci(x), "median": float(np.median(x)) if len(x) else None})
        by_tc = {}
        for tc in ("180+0", "300+0", "180+2"):
            dt = d.filter(c("tc") == tc)
            by_tc[tc] = [{"band": bd, "mean": float(np.nanmean(dt.filter(c("band") == bd)[k].cast(pl.Float64)
                                                                    .to_numpy()))}
                         for bd in bands if dt.filter(c("band") == bd).height >= 100]
        rating = (d["elo"] if "elo" in d.columns else d["elo_mean"]).cast(pl.Float64).to_numpy()
        x = d[k].cast(pl.Float64).to_numpy()
        lo, hi = rows[0]["mean"], rows[-1]["mean"]
        out[k] = {"tier": tier, "doc": doc, "bands": rows, "by_tc": by_tc, "rho_rating": spearman(x, rating),
                  "rel_change_1100_to_2100": (hi - lo) / abs(lo) if lo else None}
    return out


def structure(sides: pl.DataFrame) -> dict:
    d = sides.filter(c("has_eval"))
    feats = [k for k in AGG if k not in ("castle_ply", "first_blunder_move", "blunder_in_tt", "castled")]
    ranked = d.with_columns([c(k).rank().over("band").alias(f"r_{k}") for k in feats + ["elo"]])

    def mat(cols):
        X = ranked.select([c(k).cast(pl.Float64) for k in cols]).to_numpy()
        n = X.shape[1]
        M = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                M[i, j] = M[j, i] = spearman(X[:, i], X[:, j])
        return M

    raw = mat(feats + ["elo"])
    within = mat([f"r_{k}" for k in feats] + ["r_elo"])
    # order by hierarchical clustering on the within-band matrix
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform
    W = np.nan_to_num(within[:-1, :-1])
    dist = squareform(1 - np.abs(W), checks=False)
    order = leaves_list(linkage(dist, "average")).tolist()
    names = feats + ["elo"]
    top = []
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            top.append((abs(within[i, j]), feats[i], feats[j], float(within[i, j]), float(raw[i, j])))
    top.sort(reverse=True)
    return {"features": names, "order": [feats[i] for i in order] + ["elo"], "raw": raw.round(4).tolist(),
            "within_band": within.round(4).tolist(), "n_sides": d.height,
            "top_pairs_within": [{"a": a, "b": b, "within": w, "raw": r} for _, a, b, w, r in top[:15]]}


def cv_reg(X: np.ndarray, y: np.ndarray, groups: np.ndarray, k: int = 5) -> np.ndarray:
    from sklearn.ensemble import HistGradientBoostingRegressor
    _, g = np.unique(groups, return_inverse=True)
    fold = np.random.default_rng(SEED).integers(0, k, g.max() + 1)[g]
    p = np.empty(len(y))
    for f in range(k):
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
                                          l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                                          random_state=SEED)
        m.fit(X[fold != f], y[fold != f])
        p[fold == f] = m.predict(X[fold == f])
    return p


def reg_scores(y, p, band=None) -> dict:
    r = {"n": int(len(y)), "mae": float(np.abs(y - p).mean()), "rmse": float(np.sqrt(((y - p) ** 2).mean())),
         "r2": float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()), "pearson": float(np.corrcoef(y, p)[0, 1]),
         "y_std": float(y.std()), "baseline_mae": float(np.abs(y - y.mean()).mean())}
    if band is not None:
        bands = np.array(sorted(set(band)))
        pb = bands[np.abs(p[:, None] - (bands[None] + 50)).argmin(1)]
        r["band_exact"] = float((pb == band).mean())
        r["band_within_1"] = float((np.abs(pb - band) <= 200).mean())
        r["confusion"] = {int(b): {int(q): int(((band == b) & (pb == q)).sum()) for q in bands} for b in bands}
        r["pred_by_band"] = {int(b): {"mean": float(p[band == b].mean()), "sd": float(p[band == b].std())}
                             for b in bands}
    return r


def tier_feats(tier: str, prefix: str) -> list[str]:
    return [f"{prefix}{k}" for k, (t, _, _) in AGG.items() if registry.TIERS.index(t) <= registry.TIERS.index(tier)]


def rating(sides: pl.DataFrame, games: pl.DataFrame) -> dict:
    out = {}
    tc = ["base", "inc"]
    ev = games.filter(c("has_eval"))
    gfeat = {"T0": ["plies", "decisive", "time_forfeit"], "T2": ["plies", "decisive", "time_forfeit", "lead_changes"]}
    jobs = {
        "game_mean/tc_only/all": (games, tc, "elo_mean"),
        "game_mean/T0/all": (games, tc + gfeat["T0"] + tier_feats("T0", "w_") + tier_feats("T0", "b_"), "elo_mean"),
        "game_mean/T0/eval_subset": (ev, tc + gfeat["T0"] + tier_feats("T0", "w_") + tier_feats("T0", "b_"),
                                     "elo_mean"),
        "game_mean/T2_only/eval_subset": (ev, tc + [f for f in tier_feats("T2", "w_") + tier_feats("T2", "b_")
                                                    if f not in tier_feats("T0", "w_") + tier_feats("T0", "b_")]
                                          + ["lead_changes"], "elo_mean"),
        "game_mean/T2/eval_subset": (ev, tc + gfeat["T2"] + tier_feats("T2", "w_") + tier_feats("T2", "b_"),
                                     "elo_mean"),
        "side/T0/all": (sides, tc + tier_feats("T0", ""), "elo"),
        "side/T2/eval_subset": (sides.filter(c("has_eval")), tc + tier_feats("T2", ""), "elo"),
        "delta/T0/all": (games, tc + tier_feats("T0", "d_"), "elo_delta"),
        "delta/T0+result/all": (games, tc + tier_feats("T0", "d_") + ["score_w"], "elo_delta"),
        "delta/T2/eval_subset": (ev, tc + tier_feats("T2", "d_"), "elo_delta"),
        "delta/T2+result/eval_subset": (ev, tc + tier_feats("T2", "d_") + ["score_w"], "elo_delta"),
        "delta/result_only/all": (games, ["score_w"], "elo_delta"),
    }
    preds = {}
    for name, (df, feats, tgt) in jobs.items():
        t = time.time()
        X = df.select([c(f).cast(pl.Float64) for f in feats]).to_numpy()
        y = df[tgt].cast(pl.Float64).to_numpy()
        p = cv_reg(X, y, df["game_id"].to_numpy())
        out[name] = reg_scores(y, p, df["band"].to_numpy() if tgt != "elo_delta" else None) | {"features": feats}
        if name in ("game_mean/T2/eval_subset", "game_mean/T0/all"):
            preds[name] = (df, p)
        print(f"{name:32} n {len(y):6}  mae {out[name]['mae']:6.1f}  r2 {out[name]['r2']:+.3f}"
              f"  (base mae {out[name]['baseline_mae']:.1f}; {time.time() - t:.0f}s)", flush=True)
    # residual within band: does the model rank players inside a 100-Elo band?
    for name, (df, p) in preds.items():
        y = df["elo_mean"].cast(pl.Float64).to_numpy()
        band = df["band"].to_numpy()
        out[name]["within_band_rho"] = {int(b): spearman(p[band == b], y[band == b]) for b in sorted(set(band))}
    return out


def importance(games: pl.DataFrame) -> dict:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    ev = games.filter(c("has_eval"))
    feats = ["base", "inc", "plies", "decisive", "time_forfeit", "lead_changes"] + tier_feats("T2", "w_") + \
        tier_feats("T2", "b_")
    X = ev.select([c(f).cast(pl.Float64) for f in feats]).to_numpy()
    y = ev["elo_mean"].cast(pl.Float64).to_numpy()
    idx = np.random.default_rng(SEED).permutation(len(y))
    tr, te = idx[: int(0.8 * len(y))], idx[int(0.8 * len(y)):]
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, min_samples_leaf=40, l2_regularization=1.0,
                                      early_stopping=True, random_state=SEED).fit(X[tr], y[tr])
    pi = permutation_importance(m, X[te], y[te], n_repeats=5, random_state=SEED, scoring="neg_mean_absolute_error")
    # pool white and black versions of a feature
    pooled = {}
    for f, v in zip(feats, pi.importances_mean):
        key = f[2:] if f[:2] in ("w_", "b_") else f
        pooled[key] = pooled.get(key, 0.0) + float(v)
    return {"metric": "MAE increase, Elo, when permuted (white + black summed)",
            "pooled": dict(sorted(pooled.items(), key=lambda kv: -kv[1]))}


def run():
    t = time.time()
    sides, games = tables()
    print(f"{games.height:,} games, {sides.height:,} sides, {games['has_eval'].sum():,} with evals "
          f"({time.time() - t:.0f}s)", flush=True)
    out = {"registry_version": registry.VERSION, "games": games.height, "games_with_evals": int(games["has_eval"].sum()),
           "aggregates": {k: {"tier": t_, "doc": d} for k, (t_, _, d) in AGG.items()} | {
               k: {"tier": "T0", "doc": d} for k, d in GAME.items()}}
    out["trends"] = trends(sides, games)
    (RESULTS / "game_science.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"trends done ({time.time() - t:.0f}s)", flush=True)
    out["structure"] = structure(sides)
    print(f"structure done ({time.time() - t:.0f}s)", flush=True)
    out["rating"] = rating(sides, games)
    out["importance"] = importance(games)
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "game_science.json"
    path.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"-> {path} ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    build(a.workers) if a.cmd == "build" else run()

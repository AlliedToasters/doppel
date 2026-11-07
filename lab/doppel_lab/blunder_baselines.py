"""Blunder baselines, step 3 of docs/SCIENCE.md §7: the prior and feature heuristics, on the corpus feature table.

Target: blunder (wc_loss >= 0.3) and blunder2 (>= 0.2), one row per ply, mover's view; every position is kept.
Feature sets come from the registry (doppel_lab.features.registry.select), so a name means one thing everywhere:

  prior        the training rate
  T0           pre-move T0 features (material, clocks, ratings, legal moves, check, tempo), point + trajectory
  T1           + Maia-3 entropy and top-move probability (cheap difficulty proxy), still pre-move
  T1+think     + this move's think time ("joint with the clock": not usable before moving; measures the link)
  T2           + Stockfish eval of the position before the move and lagged CPL/wc-loss trajectories (pre)

Model: gradient-boosted trees (they take NaN for missing clocks and empty lags). Fit modes:
  cv     5 folds on val, grouped by game (val players are held out from v0 training; use until train evals exist)
  train  fit on the train table, score on val (the final numbers)

Metrics (§7): AUC, average precision, log-loss gain over the prior, expected calibration error and a 10-bin
reliability curve, and rate match (mean predicted vs observed) by clock fraction, Maia-3 entropy and rating.

    python -m doppel_lab.features.table --split val --tier T2
    python -m doppel_lab.blunder_baselines --fit cv          # -> lab/results/blunder_baselines_1500_cv.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from doppel_lab.features import registry, table

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "lab/results"
TARGETS = ("blunder", "blunder2")
THINK = ["think_s", "think_frac"]
SLICES = {"clock_frac": [0, 0.1, 0.25, 0.5, 1.0, np.inf], "maia_entropy": [0, 0.5, 1.0, 1.5, 2.0, np.inf],
          "elo_self": [0, 1450, 1500, 1550, np.inf]}


def feature_sets() -> dict[str, list[str]]:
    t0 = registry.select(tier="T0", regime="R1")
    t1 = registry.select(tier="T1", regime="R1")
    t2 = registry.select(tier="T2", regime="R1")
    return {"T0": t0, "T1": t1, "T1+think": t1 + THINK, "T2": t2}


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> tuple[float, list[dict]]:
    """Equal-mass bins: ECE = sum_b (n_b / n) |mean p_b - mean y_b|."""
    order = np.argsort(p)
    curve, err = [], 0.0
    for idx in np.array_split(order, bins):
        mp, my = float(p[idx].mean()), float(y[idx].mean())
        curve.append({"n": int(len(idx)), "pred": mp, "obs": my})
        err += len(idx) / len(p) * abs(mp - my)
    return float(err), curve


def scores(y: np.ndarray, p: np.ndarray, prior: np.ndarray, df: pl.DataFrame) -> dict:
    from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
    p = np.clip(p, 1e-6, 1 - 1e-6)
    ll, ll0 = log_loss(y, p), log_loss(y, prior)
    e, curve = ece(y, p)
    r = {"auc": float(roc_auc_score(y, p)), "avg_precision": float(average_precision_score(y, p)),
         "logloss": float(ll), "logloss_gain": float(1 - ll / ll0), "ece": e,
         "mean_pred": float(p.mean()), "rate": float(y.mean()), "reliability": curve, "rate_match": {}}
    for col, edges in SLICES.items():
        x = df[col].to_numpy().astype(float)
        rows = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (x >= lo) & (x < hi)
            if m.sum() >= 200:
                rows.append({"lo": lo, "hi": None if np.isinf(hi) else hi, "n": int(m.sum()),
                             "obs": float(y[m].mean()), "pred": float(p[m].mean())})
        r["rate_match"][col] = rows
    return r


def fit_predict(Xtr, ytr, Xte, seed: int = 0) -> np.ndarray:
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=200,
                                         l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                                         random_state=seed)
    return clf.fit(Xtr, ytr).predict_proba(Xte)[:, 1]


def folds(game_ids: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    _, g = np.unique(game_ids, return_inverse=True)
    return np.random.default_rng(seed).integers(0, k, g.max() + 1)[g]


def run(fit: str, band: int, k: int, max_rows: int | None):
    have = set(pl.scan_parquet(table.out_dir("val", "T2", band) / "part-*.parquet").collect_schema().names())
    sets = {k: [f for f in v if f in have] for k, v in feature_sets().items()}
    absent = sorted(set().union(*feature_sets().values()) - have)  # e.g. Maia-3 W/D/L, not stored for v0
    cols = sorted(set().union(*sets.values()) | set(TARGETS) | set(SLICES) | {"game_id"})
    va = table.load("val", "T2", band, columns=cols)
    tr = table.load("train", "T2", band, columns=cols) if fit == "train" else None
    if max_rows:
        va = va.head(max_rows)
    out = {"registry_version": registry.VERSION, "band": band, "fit": fit, "rows_val": va.height,
           "games_val": va["game_id"].n_unique(), "feature_sets": sets, "absent_from_table": absent, "targets": {}}
    if tr is not None:
        out |= {"rows_train": tr.height, "games_train": tr["game_id"].n_unique()}
    fold = folds(va["game_id"].to_numpy(), k) if fit == "cv" else None
    for tgt in TARGETS:
        y = va[tgt].to_numpy().astype(int)
        if fit == "cv":
            prior = np.empty(len(y))
            for f in range(k):
                prior[fold == f] = y[fold != f].mean()
        else:
            prior = np.full(len(y), tr[tgt].to_numpy().mean())
        res = {"prior": scores(y, prior, prior, va)}
        for name, feats in sets.items():
            t = time.time()
            X = va.select(feats).to_numpy().astype(np.float32)
            if fit == "cv":
                p = np.empty(len(y))
                for f in range(k):
                    p[fold == f] = fit_predict(X[fold != f], y[fold != f], X[fold == f])
            else:
                p = fit_predict(tr.select(feats).to_numpy().astype(np.float32), tr[tgt].to_numpy().astype(int), X)
            res[name] = scores(y, p, prior, va)
            print(f"{tgt:9} {name:9} auc {res[name]['auc']:.3f}  ap {res[name]['avg_precision']:.3f}  "
                  f"ll gain {res[name]['logloss_gain']:+.3f}  ece {res[name]['ece']:.4f}  ({time.time() - t:.0f}s)",
                  flush=True)
        out["targets"][tgt] = res
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"blunder_baselines_{band}_{fit}.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"-> {path}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", choices=["cv", "train"], default="cv")
    ap.add_argument("--band", type=int, default=1500)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-rows", type=int, default=None, help="smoke test")
    a = ap.parse_args()
    run(a.fit, a.band, a.folds, a.max_rows)

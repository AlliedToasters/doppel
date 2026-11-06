"""Resignation labels for the v0 games, plus a cheap CPU baseline for a resign head.

Labels (one row per game, a sidecar to games_<band>_<split>.parquet and the teacher shards, joined on game_id):
  end        resign | mate | draw | flag | other
             "Normal" termination without checkmate and without a draw is a resignation by the loser.
  resigner   "w" / "b" for resignations, else null
  on_move    the resigner was the side to move in the final position (resigned instead of replying)
  plies      number of moves

Positions for a resign head are 0..plies (the position before each ply, plus the final one). Teacher rows cover
0..plies-1 only, so the head trains on its own positions, from the moves. The target at position i for the side
to move: 1 iff i == plies and that side resigned; 0 at every earlier position. Final positions of games that did
not end in resignation are dropped: no decision was made there (mate, draw) or it is ambiguous (flag).

Baseline: logistic regression on hand features (material, clocks, ratings, ply), so the neural head has a number
to beat. Material is a crude eval: the network sees the board and should do much better.

    python -m doppel_lab.resign label --split val          # CPU, ~1 min for val, ~15 min for train
    python -m doppel_lab.resign baseline --train-games 60000
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import chess
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
V0 = ROOT / "data/v0"
RESULTS = ROOT / "lab/results"
VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
FEATURES = ["material", "material_abs", "own_clock_frac", "opp_clock_frac", "log_own_clock", "clock_ratio",
            "elo_diff", "own_elo", "ply", "n_legal", "in_check", "losing_big"]


def classify(result: str, termination: str, board: chess.Board) -> tuple[str, str | None, bool]:
    if termination == "Time forfeit":
        return "flag", None, False
    if termination != "Normal":
        return "other", None, False
    if board.is_checkmate():
        return "mate", None, False
    if result == "1/2-1/2":
        return "draw", None, False
    resigner = "b" if result == "1-0" else "w" if result == "0-1" else None
    if resigner is None:
        return "other", None, False
    return "resign", resigner, (board.turn == chess.WHITE) == (resigner == "w")


def material(b: chess.Board) -> int:
    """Material balance from the side to move's point of view."""
    s = 0
    for pt, v in VALUES.items():
        s += v * (len(b.pieces(pt, b.turn)) - len(b.pieces(pt, not b.turn)))
    return s


def label_game(args):
    gid, moves, result, termination = args
    b = chess.Board()
    for u in moves:
        b.push_uci(u)
    end, who, on_move = classify(result, termination, b)
    return gid, end, who, on_move, len(moves)


def label(split: str, band: int, workers: int) -> Path:
    g = pl.read_parquet(V0 / f"games_{band}_{split}.parquet").select("game_id", "moves", "result", "termination")
    with mp.get_context("fork").Pool(workers) as pool:
        rows = pool.map(label_game, g.iter_rows(), chunksize=256)
    out = pl.DataFrame(rows, schema=["game_id", "end", "resigner", "on_move", "plies"], orient="row")
    path = V0 / f"resign_{band}_{split}.parquet"
    out.write_parquet(path)
    counts = out["end"].value_counts().sort("count", descending=True)
    r = out.filter(pl.col("end") == "resign")
    print(f"{split}: {out.height:,} games -> {path}")
    print("  " + ", ".join(f"{e} {n:,} ({n / out.height:.1%})" for e, n in counts.iter_rows()))
    print(f"  resigner on move in final position: {r['on_move'].mean():.1%} of resignations")
    return path


def game_features(args):
    """Per decision position of one game: hand features and the resign target for the side to move."""
    moves, clocks, we, be, base, inc, end, resigner = args
    b = chess.Board()
    n = len(moves)
    rows, ys = [], []
    for i in range(n + 1):
        if i == n and end != "resign":
            break
        white = b.turn == chess.WHITE
        own = clocks[i - 2] if i >= 2 else base
        opp = clocks[i - 1] if i >= 1 else base
        es, eo = (we, be) if white else (be, we)
        m = material(b)
        nl = b.legal_moves.count()
        rows.append([m, abs(m), own / max(base, 1), opp / max(base, 1), np.log1p(own), np.log((own + 1) / (opp + 1)),
                     (es - eo) / 400, (es - 1500) / 400, i / 100, nl / 40, float(b.is_check()), float(m <= -5)])
        ys.append(1 if (i == n and (resigner == "w") == white) else 0)
        if i < n:
            b.push_uci(moves[i])
    return rows, ys


def features(split: str, band: int, n_games: int | None, workers: int, seed: int = 0):
    g = pl.read_parquet(V0 / f"games_{band}_{split}.parquet")
    lab = pl.read_parquet(V0 / f"resign_{band}_{split}.parquet").select("game_id", "end", "resigner")
    g = g.join(lab, on="game_id")
    if n_games and n_games < g.height:
        g = g.sample(n_games, seed=seed)
    args = list(g.select("moves", "clocks", "white_elo", "black_elo", "base", "inc", "end", "resigner").iter_rows())
    with mp.get_context("fork").Pool(workers) as pool:
        out = pool.map(game_features, args, chunksize=64)
    X = np.array([r for rows, _ in out for r in rows], np.float32)
    y = np.array([v for _, ys in out for v in ys], np.int8)
    return X, y


def baseline(band: int, train_games: int, workers: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

    Xtr, ytr = features("train", band, train_games, workers)
    Xva, yva = features("val", band, None, workers)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr)
    p = clf.predict_proba(Xva)[:, 1]
    prior = np.full_like(p, ytr.mean())
    res = {
        "train_positions": int(len(ytr)), "val_positions": int(len(yva)), "val_positive_rate": float(yva.mean()),
        "auc": float(roc_auc_score(yva, p)), "avg_precision": float(average_precision_score(yva, p)),
        "logloss": float(log_loss(yva, p)), "logloss_prior": float(log_loss(yva, prior)),
        "coef": dict(zip(FEATURES, map(float, clf.coef_[0]))),
    }
    res["logloss_gain"] = 1 - res["logloss"] / res["logloss_prior"]
    # How often the true resignation is the model's highest-scoring position in the resigner's game? Proxy for
    # "would resign at the right moment": rank within game is below; here just top-k precision over all positions.
    k = int(yva.sum())
    top = np.argsort(-p)[:k]
    res["precision_at_n_resigns"] = float(yva[top].mean())
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"resign_baseline_{band}.json"
    path.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))
    print(f"-> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("label")
    s.add_argument("--split", choices=["train", "val"], default="val")
    s.add_argument("--band", type=int, default=1500)
    s.add_argument("--workers", type=int, default=24)
    b = sub.add_parser("baseline")
    b.add_argument("--band", type=int, default=1500)
    b.add_argument("--train-games", type=int, default=60_000)
    b.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    if a.cmd == "label":
        label(a.split, a.band, a.workers)
    else:
        baseline(a.band, a.train_games, a.workers)

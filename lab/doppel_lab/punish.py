"""Does Maia-3 fail to punish blunders as often as 1500 humans do? (CPU: Stockfish on every legal move, Maia-3 CPU)

    python -m doppel_lab.punish [--n 1500] [--workers 24]

Val positions (held-out players, 1500 band, ply >= 10). Treated: the opponent's last move was a blunder.
Control: no blunder by the opponent, drawn to match the treated cells (Maia entropy decile x win chances x
clock fraction, as blunder_chain.py). For every legal move: Stockfish at the label budget (50k nodes) after the
move; the move blunders if the mover's win chances drop >= 0.3 from the stored pre-move eval (the exact
label rule). Per position:
  human      the label of the move actually played
  maia_p     P(blunder) under Maia-3 79M at 1500 vs 1500 = sum of policy over blundering moves
  beta       share of legal moves that blunder (Anderson et al.'s blunder potential)
Output: lab/results/punish.json
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import chess
import chess.engine
import numpy as np
import polars as pl

from doppel_lab.features import table
from doppel_lab.features.channels import wc

ROOT = Path(__file__).resolve().parents[2]
V0 = ROOT / "data" / "v0"
RESULTS = ROOT / "lab" / "results"
NODES, MATE, THRESH, HIST = 50_000, 10_000, 0.3, 8
c = pl.col
_eng = None


def _init():
    global _eng
    _eng = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")
    _eng.configure({"Threads": 1, "Hash": 16})


def _cp(b: chess.Board) -> float:
    if b.is_checkmate():
        return -MATE if b.turn == chess.WHITE else MATE
    if b.is_stalemate() or b.is_insufficient_material() or b.can_claim_threefold_repetition():
        return 0
    return _eng.analyse(b, chess.engine.Limit(nodes=NODES))["score"].white().score(mate_score=MATE)


def _position(task):
    moves, k, cp_before = task
    b = chess.Board()
    for u in moves[:k]:
        b.push_uci(u)
    sign = 1.0 if b.turn == chess.WHITE else -1.0
    w0 = sign * wc(np.array([cp_before], float))[0]
    out = []
    for m in b.legal_moves:
        b.push(m)
        cp = _cp(b)
        b.pop()
        w1 = sign * wc(np.array([cp], float))[0]
        out.append((m.uci(), float(w0 - w1) >= THRESH))
    return out


def sample(n: int, rng) -> pl.DataFrame:
    t = table.load("val", "T2", 1500, columns=["game_id", "mover", "ply", "maia_entropy", "wc_own", "clock_frac",
                                                 "blunder"]).sort("game_id", "ply")
    t = t.with_columns(opp_prev=c("blunder").shift(1).over("game_id"))
    t = t.filter((c("ply") >= 10) & c("maia_entropy").is_finite() & c("wc_own").is_not_null() & c("opp_prev").is_not_null())
    t = t.with_columns(dq=c("maia_entropy").qcut(10, labels=[str(i) for i in range(10)], allow_duplicates=True),
                       wq=c("wc_own").cut([-0.6, -0.2, 0.2, 0.6], labels=["lost", "worse", "equal", "better", "won"]),
                       cq=c("clock_frac").fill_null(0.5).cut([0.1, 0.3], labels=["a", "b", "c"]),
                       opp_prev=c("opp_prev").cast(pl.Boolean))
    tr = t.filter(c("opp_prev"))
    tr = tr[np.sort(rng.choice(tr.height, min(n, tr.height), replace=False))]
    keys = ["dq", "wq", "cq"]
    need = tr.group_by(keys).len()
    ctl = []
    pool = t.filter(~c("opp_prev"))
    for row in need.iter_rows(named=True):
        cand = pool.filter((c("dq") == row["dq"]) & (c("wq") == row["wq"]) & (c("cq") == row["cq"]))
        if cand.height:
            ctl.append(cand[rng.choice(cand.height, min(row["len"], cand.height), replace=False)])
    return pl.concat([tr.with_columns(group=pl.lit("after_opp_blunder")),
                      pl.concat(ctl).with_columns(group=pl.lit("matched_control"))])


def maia_probs(rows: pl.DataFrame, games: dict) -> list[dict]:
    import torch

    from doppel_lab.engines import Maia3
    from doppel_lab.teacher import encode_game
    from doppel_model import MOVE_INDEX, mirror_uci
    model = Maia3("79m", "cpu").model.eval()
    out = []
    for r in rows.iter_rows(named=True):
        moves = games[r["game_id"]]
        k = r["ply"]
        planes = encode_game((moves[:k + 1], 1500, 1500))[0]          # (k+1, 64, 12) positions 0..k
        h = list(range(max(0, k - HIST + 1), k + 1))
        h = [h[0]] * (HIST - len(h)) + h
        x = torch.from_numpy(planes[h]).float().permute(1, 0, 2).reshape(64, HIST * 12)
        x = torch.cat([x, torch.zeros(64, 1)], 1)[None]
        with torch.no_grad():
            logits = model(x, torch.tensor([1500]), torch.tensor([1500]))[0][0].float()
        b = chess.Board()
        for u in moves[:k]:
            b.push_uci(u)
        white = b.turn == chess.WHITE
        legal = [m.uci() for m in b.legal_moves]
        idx = torch.tensor([MOVE_INDEX[u if white else mirror_uci(u)] for u in legal])
        p = torch.softmax(logits[idx], 0).numpy()
        out.append(dict(zip(legal, p.tolist())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    rows = sample(a.n, rng)
    games = dict(pl.read_parquet(V0 / "blunder_games_1500_val.parquet", columns=["game_id", "moves"]).iter_rows())
    evals = dict(pl.read_parquet(V0 / "evals_1500_val.parquet").iter_rows())
    tasks = [(games[r["game_id"]], r["ply"], evals[r["game_id"]][r["ply"]]) for r in rows.iter_rows(named=True)]
    print(f"{rows.height:,} positions ({rows.filter(c('group') == 'after_opp_blunder').height:,} treated)", flush=True)
    t0 = time.time()
    with Pool(a.workers, initializer=_init) as p:
        bl = p.map(_position, tasks, chunksize=4)
    print(f"stockfish on all legal moves: {time.time() - t0:.0f}s", flush=True)
    t0 = time.time()
    pol = maia_probs(rows, games)
    print(f"maia-3 on cpu: {time.time() - t0:.0f}s", flush=True)
    maia_p = [sum(pm.get(u, 0.0) for u, is_b in moves if is_b) for moves, pm in zip(bl, pol)]
    beta = [np.mean([is_b for _, is_b in moves]) if moves else np.nan for moves in bl]
    maia_top_b = [dict(moves).get(max(pm, key=pm.get), False) for moves, pm in zip(bl, pol)]
    any_safe = [not all(is_b for _, is_b in moves) for moves in bl]
    d = rows.with_columns(maia_p=pl.Series(maia_p), beta=pl.Series(beta), maia_top1_blunder=pl.Series(maia_top_b),
                          any_safe=pl.Series(any_safe))
    s = d.group_by("group").agg(n=pl.len(), human=c("blunder").cast(pl.Float64).mean(), maia_implied=c("maia_p").mean(),
                                maia_top1=c("maia_top1_blunder").cast(pl.Float64).mean(), beta=c("beta").mean(),
                                all_moves_blunder=(~c("any_safe")).cast(pl.Float64).mean()).sort("group")
    with pl.Config(float_precision=4, tbl_cols=10):
        print(s)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "punish.json").write_text(json.dumps({"nodes": NODES, "summary": s.to_dicts()}, indent=1) + "\n")


if __name__ == "__main__":
    main()

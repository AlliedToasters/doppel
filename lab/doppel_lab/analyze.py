"""Stockfish-annotate games: centipawn eval (White's view) of the start position and after every ply.

Fixed node budget per position so human and synthetic games get identical treatment. Mate scores
become +/-10000. Resumable: games already in the output are skipped (output is rewritten per chunk).

Usage: python -m doppel_lab.analyze <games.parquet> <evals.parquet> [--nodes 50000] [--workers 28]
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool

import chess
import chess.engine
import polars as pl

MATE = 10000
_engine = None
_nodes = None


def _init(nodes):
    global _engine, _nodes
    _engine = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")
    _engine.configure({"Threads": 1, "Hash": 16})
    _nodes = nodes


def _score(board):
    if board.is_checkmate():
        return -MATE if board.turn == chess.WHITE else MATE
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    info = _engine.analyse(board, chess.engine.Limit(nodes=_nodes))
    return info["score"].white().score(mate_score=MATE)


def _game(task):
    gid, moves = task
    b = chess.Board()
    evals = [_score(b)]
    for mv in moves:
        b.push_uci(mv)
        evals.append(_score(b))
    return gid, evals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--nodes", type=int, default=50000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    ap.add_argument("--chunk", type=int, default=500)
    args = ap.parse_args()

    games = pl.read_parquet(args.src, columns=["game_id", "moves"])
    done = pl.read_parquet(args.out) if os.path.exists(args.out) else pl.DataFrame(
        schema={"game_id": pl.String, "evals": pl.List(pl.Int64)})
    todo = games.filter(~pl.col("game_id").is_in(done["game_id"].implode()))
    print(f"{len(todo)} games to analyse ({len(done)} done) at {args.nodes} nodes, {args.workers} workers",
          file=sys.stderr, flush=True)
    rows = list(todo.iter_rows())
    t0 = time.time()
    with Pool(args.workers, initializer=_init, initargs=(args.nodes,)) as pool:
        for start in range(0, len(rows), args.chunk):
            res = pool.map(_game, rows[start:start + args.chunk], chunksize=4)
            done = pl.concat([done, pl.DataFrame(res, schema=done.schema, orient="row")])
            done.write_parquet(args.out + ".tmp")
            os.replace(args.out + ".tmp", args.out)
            print(f"{start + len(res)}/{len(rows)} games, {time.time() - t0:.0f}s", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()

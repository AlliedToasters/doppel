"""Batched self-play: N games at once, both sides at the band-centre Elo, time controls drawn from the
band's human games. Writes the same columns as doppel_lab.ingest plus `source`.

Termination: checkmate, stalemate, insufficient material, threefold repetition, 50-move rule,
flag fall (ChessMimic only: it has a clock model), or a 400-ply cap. Neither baseline resigns;
the metrics adjudicate every source the same way from Stockfish evals.

Usage: python -m doppel_lab.selfplay --engine {chessmimic,maia3} --band 1500 --games 1000 \
        --human data/human/games.parquet --out data/synth/... [--device cuda --gpu-ok]
"""

import argparse
import sys
import time

import chess
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from doppel_lab.engines import ChessMimic, GameState, Maia3

MAX_PLIES = 400


def ended(b: chess.Board):
    if b.is_checkmate():
        return ("1-0" if b.turn == chess.BLACK else "0-1"), "Normal"
    if b.is_stalemate() or b.is_insufficient_material() or b.halfmove_clock >= 100 or b.is_repetition(3):
        return "1/2-1/2", "Normal"
    if b.ply() >= MAX_PLIES:
        return "*", "Cap"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["chessmimic", "maia3"], required=True)
    ap.add_argument("--maia-size", default="79m")
    ap.add_argument("--band", type=int, required=True)
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--human", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--gpu-ok", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.device.startswith("cuda") and not args.gpu_ok:
        sys.exit("refusing to touch the GPU without --gpu-ok (run through the warden)")

    rng = np.random.default_rng(args.seed + args.band)
    import torch
    torch.manual_seed(args.seed + args.band)
    tcs = pl.read_parquet(args.human, columns=["band", "base", "inc"]).filter(pl.col("band") == args.band)
    tc_idx = rng.integers(0, len(tcs), args.games)
    elo = args.band + 50

    eng = ChessMimic(args.band, args.device) if args.engine == "chessmimic" else Maia3(args.maia_size, args.device)
    use_clock = args.engine == "chessmimic"

    states, clocks, results = [], [], [None] * args.games
    for i in range(args.games):
        base, inc = float(tcs["base"][int(tc_idx[i])]), float(tcs["inc"][int(tc_idx[i])])
        s = GameState(chess.Board(), elo, elo, base, inc, base, base)
        if not use_clock:
            s.maia_hist.append(Maia3.board_tokens(s.board))
        states.append(s)
        clocks.append([])
    active = list(range(args.games))
    t0, step = time.time(), 0
    while active:
        picks = eng.sample([states[i] for i in active], rng)
        still = []
        for i, (mv, think) in zip(active, picks):
            s = states[i]
            mover_white = s.board.turn == chess.WHITE
            if use_clock:
                left = (s.clock_white if mover_white else s.clock_black) - think
                if left <= 0:
                    results[i] = ("0-1" if mover_white else "1-0"), "Time forfeit"
                    continue
                left += s.inc
                if mover_white:
                    s.clock_white = left
                else:
                    s.clock_black = left
                clocks[i].append(round(left, 2))
            s.board.push(chess.Move.from_uci(mv))
            s.moves.append(mv)
            if not use_clock:
                s.maia_hist.append(Maia3.board_tokens(s.board))
                s.maia_hist = s.maia_hist[-8:]
            r = ended(s.board)
            if r:
                results[i] = r
            else:
                still.append(i)
        active = still
        step += 1
        if step % 20 == 0:
            print(f"ply {step}: {len(active)} games running, {time.time() - t0:.0f}s", file=sys.stderr, flush=True)

    src = f"{args.engine}" + (f"-{args.maia_size}" if args.engine == "maia3" else "")
    tbl = pa.table({
        "game_id": [f"{src}-{args.band}-{i}" for i in range(args.games)],
        "band": [args.band] * args.games,
        "base": [int(s.base) for s in states],
        "inc": [int(s.inc) for s in states],
        "result": [r[0] for r in results],
        "termination": [r[1] for r in results],
        "moves": [s.moves for s in states],
        "clocks": [c[:len(s.moves)] if use_clock else None for c, s in zip(clocks, states)],
        "source": [src] * args.games,
    })
    pq.write_table(tbl, args.out, compression="zstd")
    print(f"wrote {args.games} games to {args.out} in {time.time() - t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()

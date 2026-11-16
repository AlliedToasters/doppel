"""Teacher-forced move prediction on human games: top-1 accuracy and perplexity per band.

Uses the move model only (no lookup table, no sampling), both players' real Elo, and for ChessMimic the
real clock before each move. Plies before --skip-plies are excluded. --cut-clock S ends each game at the first ply
where the player to move has under S seconds before moving (the Allie / Maia-3 test protocol is --skip-plies 10
--cut-clock 30: 884,049 positions on the Allie 2022 test set). --save-positions writes one row per scored position
(game_id, ply, hit, logp of the played move) for paired comparisons.

Usage: python -m doppel_lab.top1 --engine {chessmimic,maia3} --human games.parquet [--bands 1500 ...]
        [--games-per-band 200] [--device cuda --gpu-ok]
"""

import argparse
import json
import math
import sys

import chess
import numpy as np
import polars as pl
import torch

from doppel_lab.engines import ChessMimic, GameState, Maia3


@torch.no_grad()
def logprobs(eng, states):
    """Return (log p over legal moves as dict uci->logp) for each state, via the engine's move model."""
    out = []
    if isinstance(eng, Maia3):
        from maia3.utils import mirror_move
        x = torch.stack([eng._input(s) for s in states]).to(eng.device)
        se = torch.tensor([s.elo_to_move for s in states], dtype=torch.long, device=eng.device)
        oe = torch.tensor([s.elo_opp for s in states], dtype=torch.long, device=eng.device)
        with torch.autocast("cuda", enabled=eng.device.startswith("cuda")):
            logits = eng.model(x, se, oe)[0].float()
        for r, s in enumerate(states):
            white = s.board.turn == chess.WHITE
            legal = [m.uci() for m in s.board.legal_moves]
            idx = [eng.move_idx[u if white else mirror_move(u)] for u in legal]
            lp = torch.log_softmax(logits[r, idx], 0).tolist()
            out.append(dict(zip(legal, lp)))
        return out
    cc, dev = eng.cc, eng.device

    def z(v, ms):
        return (v - ms[0]) / ms[1]
    recent = torch.tensor([eng._recent(s.moves) for s in states], dtype=torch.long, device=dev)
    board = torch.tensor(np.stack([cc.tokenize(s.board.fen()) for s in states]), dtype=torch.long, device=dev)
    rat = torch.tensor([z(s.elo_to_move, eng.m_rating) for s in states], device=dev)
    lt = torch.tensor([z(math.log(1 + s.clocks[0]), eng.m_time) for s in states], device=dev)
    logits = eng.move_model(recent, board, rat, lt).float()
    for r, s in enumerate(states):
        legal = [m.uci() for m in s.board.legal_moves]
        lp = torch.log_softmax(logits[r, [cc.MOVE_TO_ACTION[u] for u in legal]], 0).tolist()
        out.append(dict(zip(legal, lp)))
    return out


def positions(row, skip, with_maia_hist, cut_clock=None):
    b = chess.Board()
    s = GameState(b.copy(), row["white_elo"], row["black_elo"], row["base"], row["inc"], row["base"], row["base"])
    if with_maia_hist:
        s.maia_hist = [Maia3.board_tokens(s.board)]
    for ply, (mv, clk) in enumerate(zip(row["moves"], row["clocks"])):
        before = s.clock_white if s.board.turn == chess.WHITE else s.clock_black
        if cut_clock is not None and before < cut_clock:
            return
        if ply >= skip:
            snap = GameState(s.board.copy(stack=False), s.elo_white, s.elo_black, s.base, s.inc,
                             s.clock_white, s.clock_black, list(s.moves), list(s.maia_hist))
            yield snap, mv, (row["game_id"], ply)
        if s.board.turn == chess.WHITE:
            s.clock_white = clk
        else:
            s.clock_black = clk
        s.board.push_uci(mv)
        s.moves.append(mv)
        if with_maia_hist:
            s.maia_hist = (s.maia_hist + [Maia3.board_tokens(s.board)])[-8:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["chessmimic", "maia3"], required=True)
    ap.add_argument("--maia-size", default="79m")
    ap.add_argument("--human", required=True)
    ap.add_argument("--bands", type=int, nargs="*", default=[1100, 1300, 1500, 1700, 1900, 2100])
    ap.add_argument("--games-per-band", type=int, default=200)
    ap.add_argument("--skip-plies", type=int, default=8)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--gpu-ok", action="store_true")
    ap.add_argument("--cut-clock", type=float, default=None)
    ap.add_argument("--save-positions")
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.device.startswith("cuda") and not args.gpu_ok:
        sys.exit("refusing to touch the GPU without --gpu-ok (run through the warden)")

    df = pl.read_parquet(args.human)
    maia = Maia3(args.maia_size, args.device) if args.engine == "maia3" else None
    results = {}
    saved = []
    for band in args.bands:
        eng = maia or ChessMimic(band, args.device)
        rows = df.filter(pl.col("band") == band).head(args.games_per_band).iter_rows(named=True)
        hits = n = 0
        nll = 0.0
        buf = []

        def flush():
            nonlocal hits, n, nll
            for (s, mv, key), lp in zip(buf, logprobs(eng, [s for s, _, _ in buf])):
                h = max(lp, key=lp.get) == mv
                hits += h
                nll -= lp[mv]
                n += 1
                if args.save_positions:
                    saved.append((band, key[0], key[1], bool(h), lp[mv]))
            buf.clear()
        for row in rows:
            for item in positions(row, args.skip_plies, maia is not None, args.cut_clock):
                buf.append(item)
                if len(buf) >= args.batch:
                    flush()
        if buf:
            flush()
        results[band] = {"positions": n, "top1": hits / n, "perplexity": math.exp(nll / n)}
        print(band, results[band], file=sys.stderr, flush=True)
    if args.save_positions:
        pl.DataFrame(saved, schema=["band", "game_id", "ply", "hit", "logp"], orient="row").write_parquet(args.save_positions)
    if args.out:
        json.dump({"engine": args.engine if not maia else f"maia3-{args.maia_size}", "bands": results},
                  open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()

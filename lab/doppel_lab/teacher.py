"""Doppel v0 data: select one rating band from the corpus, then precompute Maia-3 79M teacher outputs.

Per position (the position before each ply, both sides), over legal moves only:
  top_idx (16,) int16 and top_lp (16,) float16   teacher top-16 moves (Maia vocab, mover's frame) and log-probs
  lse float32                                     logsumexp of the legal logits (to rebuild full log-probs)
  played_idx int16, played_lp float16             the move actually played and its teacher log-prob
  entropy float16                                 teacher entropy (nats)
Shards: <out>/shard-XXXX.npz with game_ids and per-game offsets into the position arrays.

    python -m doppel_lab.teacher select --band 1500 --games 700000      # CPU, after corpus.py
    python -m doppel_lab.teacher run --split train --gpu-ok              # GPU, via the warden
    python -m doppel_lab.teacher run --split val --gpu-ok
    # any games file (game_id, moves, white_elo, black_elo), optionally with both Elos fixed (a rating-free view)
    python -m doppel_lab.teacher run --games data/human/games_2026-10.parquet --out data/human/teacher_e1500 \
        --fixed-elo 1500 --gpu-ok
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path

import chess
import numpy as np
import polars as pl

from doppel_model import MOVE_INDEX, board_planes, mirror_uci

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data/corpus/2026-10"
V0 = ROOT / "data/v0"
HIST, TOPK, BLOCK = 8, 16, 20_000


def select(band: int, n_games: int, n_val: int, seed: int = 0):
    V0.mkdir(parents=True, exist_ok=True)
    for split, src, n in (("train", "train", n_games), ("val", "heldout", n_val)):
        files = sorted((CORPUS / src).glob("part-*.parquet"))
        d = pl.concat([pl.read_parquet(f).filter((pl.col("band") == band) & (pl.col("moves").list.len() >= 20))
                       for f in files])
        d = d.sample(min(n, d.height), seed=seed, shuffle=True)
        out = V0 / f"games_{band}_{split}.parquet"
        d.write_parquet(out)
        print(f"{split}: {d.height:,} games, {d['moves'].list.len().sum():,} positions -> {out}")


def encode_game(args):
    """CPU: board planes for every position, legal-move indices, played-move index, Elos per position."""
    moves, we, be = args
    b = chess.Board()
    planes = [board_planes(b)]
    legal, counts, played, e_self, e_opp = [], [], [], [], []
    for u in moves:
        white = b.turn == chess.WHITE
        idx = [MOVE_INDEX[m.uci() if white else mirror_uci(m.uci())] for m in b.legal_moves]
        legal += idx
        counts.append(len(idx))
        mv = b.parse_uci(u)
        played.append(MOVE_INDEX[mv.uci() if white else mirror_uci(mv.uci())])
        e_self.append(we if white else be)
        e_opp.append(be if white else we)
        b.push(mv)
        planes.append(board_planes(b))
    return (np.stack(planes[:-1]).astype(np.uint8), np.array(legal, np.int16), np.array(counts, np.int16),
            np.array(played, np.int16), np.array(e_self, np.int32), np.array(e_opp, np.int32))


def run(split: str, band: int, device: str, workers: int, batch: int, games_path: str | None = None,
        out: str | None = None, fixed_elo: int | None = None):
    import torch

    from doppel_lab.engines import Maia3
    src = ROOT / games_path if games_path else V0 / f"games_{band}_{split}.parquet"
    games = pl.read_parquet(src).select("game_id", "moves", "white_elo", "black_elo")
    if fixed_elo is not None:
        games = games.with_columns(white_elo=pl.lit(fixed_elo), black_elo=pl.lit(fixed_elo))
    out_dir = ROOT / out if out else V0 / f"teacher_{band}_{split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    eng = Maia3("79m", device)
    model = eng.model
    rows = list(games.iter_rows())
    blocks = [rows[i:i + BLOCK] for i in range(0, len(rows), BLOCK)]
    t0, done_pos = time.time(), 0
    with mp.get_context("fork").Pool(workers) as pool:
        nxt = pool.map_async(encode_game, [(m, we, be) for _, m, we, be in blocks[0]], chunksize=64)
        for bi, blk in enumerate(blocks):
            path = out_dir / f"shard-{bi:04d}.npz"
            enc = nxt.get()
            if bi + 1 < len(blocks):
                nxt = pool.map_async(encode_game, [(m, we, be) for _, m, we, be in blocks[bi + 1]], chunksize=64)
            if path.exists():
                continue  # resumable: a preempted job skips finished shards
            n_per = np.array([len(e[3]) for e in enc])
            offsets = np.r_[0, np.cumsum(n_per)]
            planes = torch.from_numpy(np.concatenate([e[0] for e in enc])).to(device)  # (P, 64, 12)
            # History window for position j of game g: positions max(start, j-7)..j, padded with the earliest.
            hidx = np.concatenate([
                np.clip(np.arange(n)[:, None] + np.arange(-HIST + 1, 1)[None, :], 0, None) + off
                for n, off in zip(n_per, offsets[:-1])])
            legal = np.concatenate([e[1] for e in enc]).astype(np.int64)
            lcount = np.concatenate([e[2] for e in enc]).astype(np.int64)
            lrow = np.repeat(np.arange(len(lcount)), lcount)
            lstart = np.r_[0, np.cumsum(lcount)]
            played = torch.from_numpy(np.concatenate([e[3] for e in enc]).astype(np.int64)).to(device)
            es = torch.from_numpy(np.concatenate([e[4] for e in enc])).to(device)
            eo = torch.from_numpy(np.concatenate([e[5] for e in enc])).to(device)
            P = len(hidx)
            top_idx = np.empty((P, TOPK), np.int16)
            top_lp = np.empty((P, TOPK), np.float16)
            lse = np.empty(P, np.float32)
            played_lp = np.empty(P, np.float16)
            ent = np.empty(P, np.float16)
            hidx_t = torch.from_numpy(hidx).to(device)
            for s in range(0, P, batch):
                e_ = min(P, s + batch)
                x = planes[hidx_t[s:e_]].float()  # (B, 8, 64, 12)
                x = x.permute(0, 2, 1, 3).reshape(e_ - s, 64, HIST * 12)
                x = torch.cat([x, torch.zeros(e_ - s, 64, 1, device=device)], 2)
                with torch.no_grad(), torch.autocast("cuda", enabled=device.startswith("cuda")):
                    logits = model(x, es[s:e_], eo[s:e_])[0].float()
                mask = torch.zeros_like(logits, dtype=torch.bool)
                a, b_ = lstart[s], lstart[e_]
                mask[torch.from_numpy(lrow[a:b_] - s).to(device), torch.from_numpy(legal[a:b_]).to(device)] = True
                lg = logits.masked_fill(~mask, float("-inf"))
                z = torch.logsumexp(lg, 1)
                lp = lg - z[:, None]
                k = min(TOPK, logits.shape[1])
                tv, ti = lp.topk(k, 1)
                p = lp.exp()
                h = -(p * lp.masked_fill(~mask, 0.0)).sum(1)
                top_idx[s:e_] = ti.cpu().numpy().astype(np.int16)
                top_lp[s:e_] = tv.clamp(min=-60).cpu().numpy().astype(np.float16)  # fewer than 16 legal -> -inf
                lse[s:e_] = z.cpu().numpy()
                played_lp[s:e_] = lp.gather(1, played[s:e_, None])[:, 0].clamp(min=-60).cpu().numpy().astype(np.float16)
                ent[s:e_] = h.cpu().numpy().astype(np.float16)
            tmp = path.with_suffix(".tmp.npz")
            np.savez(tmp, game_ids=np.array([r[0] for r in blk]), offsets=offsets, top_idx=top_idx, top_lp=top_lp,
                     lse=lse, played_idx=played.cpu().numpy().astype(np.int16), played_lp=played_lp, entropy=ent)
            tmp.rename(path)
            done_pos += P
            el = time.time() - t0
            print(f"shard {bi + 1}/{len(blocks)}: {done_pos:,} positions, {done_pos / el:,.0f}/s", flush=True)
    print("done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--band", type=int, default=1500)
    s.add_argument("--games", type=int, default=700_000)
    s.add_argument("--val", type=int, default=20_000)
    r = sub.add_parser("run")
    r.add_argument("--split", choices=["train", "val"], default="train")
    r.add_argument("--band", type=int, default=1500)
    r.add_argument("--workers", type=int, default=20)
    r.add_argument("--batch", type=int, default=2048)
    r.add_argument("--games", default=None, help="games parquet (repo-relative) instead of the v0 band split")
    r.add_argument("--out", default=None, help="output dir (repo-relative)")
    r.add_argument("--fixed-elo", type=int, default=None, help="feed this Elo for both sides")
    r.add_argument("--gpu-ok", action="store_true")
    a = ap.parse_args()
    if a.cmd == "select":
        select(a.band, a.games, a.val)
    else:
        run(a.split, a.band, "cuda" if a.gpu_ok else "cpu", a.workers, a.batch, a.games, a.out, a.fixed_elo)

"""Doppel v0e: per-game, per-side Maia-3 vectors for the multi-game z encoder (GPU once, via the warden).

    # GPU (warden): data/v0/games_1500_train.parquet -> data/v0e/vec_1500_train/
    python -m doppel_lab.v0e_vec run --split train --gpu-ok
    python -m doppel_lab.v0e_vec run --split valrich --gpu-ok     # after `v0e_data select-valrich`
    # CPU smoke (a few games, writes data/v0e/vec_1500_<split>_smoke/):
    python -m doppel_lab.v0e_vec run --split train --limit 40

Per (game, side), the same definitions as bench_embed.run / sweep, so the players suite (data/bench/players/
maia3.npz, maia3_sweep.npz) gives the same features without another GPU pass:
  emb   (G, 2, 1024) f16   mean pooled trunk output (last_ln) over the side's positions, Maia-3 at 1500 vs 1500
  scal  (G, 2, 5)    f32   played-move log-prob mean / std, entropy mean, top-1 agreement, mean log1p(rank), at 1500
  sweep (G, 2, 3)    f32   mean played-move log-prob at 1000, 1500, 2000 (both inputs at that rating) over the
                           side's even-numbered own moves (half of them: own move j with j % 2 == 0)

Budget (measured on the players suite: one forward pass is ~11-12k positions/s): the full band, 48.7M train
positions, at 9 sweep ratings would be ~11 h. Here: one pass at 1500 over every position (embedding and policy
scalars) plus 2 extra ratings on half the positions = 2 forward passes per position, ~2.4 h for train and ~6 min
for valrich. Half the moves give an unbiased, noisier estimate of the side's mean log-likelihood; the 1500 value
is taken on the same half, so the differences ll(1000) - ll(1500) and ll(2000) - ll(1500) are paired.

Resumable: each PART games are written to part-XXXX.npz and skipped on restart (a preempted job loses at most
one part); `merge` (run automatically at the end) concatenates them into .npy arrays for memory mapping.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import polars as pl

from doppel_lab.bench_embed import SCAL
from doppel_lab.teacher import HIST, encode_game

ROOT = Path(__file__).resolve().parents[2]
V0E = ROOT / "data/v0e"
BLOCK, PART = 4_000, 40_000
SWEEP = (1000, 1500, 2000)


def games_path(split: str, band: int) -> Path:
    if split == "players":                      # the players suite (consistency check against bench_embed)
        return ROOT / "data/bench/players/games.parquet"
    if split in ("train", "val"):
        return ROOT / "data/v0" / f"games_{band}_{split}.parquet"
    return V0E / f"games_{band}_{split}.parquet"


def vec_dir(split: str, band: int = 1500) -> Path:
    return V0E / f"vec_{band}_{split}"


def sweep_subset(n: int) -> np.ndarray:
    """Plies (0..n-1) whose position gets the extra sweep ratings: own move j = ply // 2 with j even."""
    p = np.arange(n)
    return (p // 2) % 2 == 0


def _block(model, pooled, dim, device, batch, enc):
    """Per-position outputs for one block of encoded games."""
    import torch
    n_per = np.array([len(e[3]) for e in enc])
    offsets = np.r_[0, np.cumsum(n_per)]
    planes = torch.from_numpy(np.concatenate([e[0] for e in enc])).to(device)
    hidx = np.concatenate([np.clip(np.arange(n)[:, None] + np.arange(-HIST + 1, 1)[None, :], 0, None) + off
                           for n, off in zip(n_per, offsets[:-1])])
    legal = np.concatenate([e[1] for e in enc]).astype(np.int64)
    lcount = np.concatenate([e[2] for e in enc]).astype(np.int64)
    lrow = np.repeat(np.arange(len(lcount)), lcount)
    lstart = np.r_[0, np.cumsum(lcount)]
    played = torch.from_numpy(np.concatenate([e[3] for e in enc]).astype(np.int64)).to(device)
    sub = torch.from_numpy(np.concatenate([sweep_subset(n) for n in n_per])).to(device)
    P = len(hidx)
    feat = torch.empty(P, dim, device=device)
    lp_p, ent, top1, rank = (torch.empty(P, device=device) for _ in range(4))
    lp_sw = torch.full((P, 2), float("nan"), device=device)       # at SWEEP[0], SWEEP[2]
    hidx_t = torch.from_numpy(hidx).to(device)
    for s in range(0, P, batch):
        e_ = min(P, s + batch)
        x = planes[hidx_t[s:e_]].float().permute(0, 2, 1, 3).reshape(e_ - s, 64, HIST * 12)
        x = torch.cat([x, torch.zeros(e_ - s, 64, 1, device=device)], 2)
        mask = torch.zeros(e_ - s, 4352, dtype=torch.bool, device=device)
        a, b_ = lstart[s], lstart[e_]
        mask[torch.from_numpy(lrow[a:b_] - s).to(device), torch.from_numpy(legal[a:b_]).to(device)] = True
        pl_idx = played[s:e_, None]

        def logp(elo, rows=None):
            xx = x if rows is None else x[rows]
            ev = torch.full((len(xx),), elo, device=device)
            with torch.no_grad(), torch.autocast("cuda", enabled=device.startswith("cuda")):
                logits = model(xx, ev, ev)[0].float()
            m = mask if rows is None else mask[rows]
            return torch.log_softmax(logits[:, :m.shape[1]].masked_fill(~m, float("-inf")), 1)

        lp = logp(SWEEP[1])
        px = pooled["x"].float()
        feat[s:e_] = px.mean(1) if px.dim() == 3 else px
        pl_ = lp.gather(1, pl_idx)[:, 0]
        lp_p[s:e_] = pl_.clamp(min=-30)
        ent[s:e_] = -(lp.exp() * lp.masked_fill(~mask, 0.0)).sum(1)
        top1[s:e_] = (lp.argmax(1) == played[s:e_]).float()
        rank[s:e_] = (lp > pl_[:, None]).sum(1).float()
        rows = torch.nonzero(sub[s:e_])[:, 0]
        if len(rows):
            for j, elo in enumerate((SWEEP[0], SWEEP[2])):
                lp_sw[s + rows, j] = logp(elo, rows).gather(1, pl_idx[rows])[:, 0].clamp(min=-30)
    out = [t.cpu().numpy() for t in (feat, lp_p, ent, top1, rank, lp_sw)]
    return n_per, offsets, out


def side_rows(feat, lp_p, ent, top1, rank, lp_sw, o, n):
    """Aggregate one game's per-position outputs into (emb (2, dim), scal (2, 5), sweep (2, 3))."""
    emb = np.zeros((2, feat.shape[1]), np.float32)
    scal = np.zeros((2, len(SCAL)), np.float32)
    sw = np.full((2, 3), np.nan, np.float32)
    sub = sweep_subset(n)
    for side in (0, 1):
        ix = np.arange(o + side, o + n, 2)
        if len(ix) == 0:
            continue
        emb[side] = feat[ix].mean(0)
        scal[side] = (lp_p[ix].mean(), lp_p[ix].std(), ent[ix].mean(), top1[ix].mean(), np.log1p(rank[ix]).mean())
        js = ix[sub[ix - o]]
        if len(js):
            sw[side] = (lp_sw[js, 0].mean(), lp_p[js].mean(), lp_sw[js, 1].mean())
    return emb, scal, sw


def run(split: str, band: int, device: str, workers: int, batch: int, limit: int = 0):
    import torch

    from doppel_lab.engines import Maia3
    games = pl.read_parquet(games_path(split, band), columns=["game_id", "moves"])
    if limit:
        games = games.head(limit)
    out = vec_dir(split + ("_smoke" if limit else ""), band)
    out.mkdir(parents=True, exist_ok=True)
    model = Maia3("79m", device).model.eval()
    pooled = {}
    model.last_ln.register_forward_hook(lambda m, i, o: pooled.__setitem__("x", o))
    dim = model.last_ln.normalized_shape[0]
    G = games.height
    rows = list(games.iter_rows())
    t0, done = time.time(), 0
    with mp.get_context("fork").Pool(workers) as pool:
        for p0 in range(0, G, PART):
            part = out / f"part-{p0 // PART:04d}.npz"
            if part.exists():
                continue
            prow = rows[p0:p0 + PART]
            emb = np.zeros((len(prow), 2, dim), np.float16)
            scal = np.zeros((len(prow), 2, len(SCAL)), np.float32)
            sw = np.zeros((len(prow), 2, 3), np.float32)
            for b0 in range(0, len(prow), BLOCK):
                blk = prow[b0:b0 + BLOCK]
                enc = pool.map(encode_game, [(m, 1500, 1500) for _, m in blk], chunksize=32)
                n_per, offsets, arrs = _block(model, pooled, dim, device, batch, enc)
                for j in range(len(blk)):
                    e, s_, w_ = side_rows(*arrs, offsets[j], n_per[j])
                    emb[b0 + j], scal[b0 + j], sw[b0 + j] = e, s_, w_
                done += int(n_per.sum())
                print(f"{p0 + b0 + len(blk):,}/{G:,} games, {done:,} positions this run, "
                      f"{done / (time.time() - t0):,.0f}/s", flush=True)
            tmp = part.with_suffix(".tmp.npz")
            np.savez(tmp, game_ids=np.array([g for g, _ in prow]), emb=emb, scal=scal, sweep=sw)
            tmp.rename(part)
    merge(out, G)


def merge(out: Path, G: int):
    parts = sorted(out.glob("part-*.npz"))
    zs = [np.load(p) for p in parts]
    gids = np.concatenate([z["game_ids"] for z in zs])
    assert len(gids) == G, (len(gids), G)
    for k in ("emb", "scal", "sweep"):
        np.save(out / f"{k}.npy", np.concatenate([z[k] for z in zs]))
    np.save(out / "game_ids.npy", gids)
    np.save(out / "elos.npy", np.array(SWEEP))
    print(f"done -> {out} ({G:,} games)", flush=True)


class Vecs:
    """Memory-mapped per-(game, side) vectors of one split, in games-parquet order."""

    def __init__(self, split: str, band: int = 1500, d: Path | None = None):
        d = d or vec_dir(split, band)
        self.game_ids = np.load(d / "game_ids.npy", allow_pickle=True)
        self.emb = np.load(d / "emb.npy", mmap_mode="r")
        self.scal = np.load(d / "scal.npy")
        self.sweep = np.load(d / "sweep.npy")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--split", default="train")
    r.add_argument("--band", type=int, default=1500)
    r.add_argument("--workers", type=int, default=12)
    r.add_argument("--batch", type=int, default=2048)
    r.add_argument("--gpu-ok", action="store_true")
    r.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    run(a.split, a.band, "cuda" if a.gpu_ok else "cpu", a.workers, a.batch, a.limit)

"""Doppel v0 data: a compact per-split index, and the training sample builder.

Index (CPU, once per split): data/v0/index_<band>_<split>.npz joins the games parquet, the resign sidecar and the
teacher shards (same game order, checked) into flat arrays:
  per game:      off (G+1) into positions, white_elo, black_elo, base, inc, end, resigner, on_move
  per position:  uci (S5), clock (mover's clock after the ply, s), played_lp, rank (of the played move in the
                 teacher's top 16; 16 = outside), entropy

Sample = (game, colour, context half). The encoder reads every ply of the context half (own moves scored, the
opponent's as context: SCIENCE.md / brief "z encoder reads the opponent's moves"). Targets are the colour's own
decision positions in the other half: up to K sampled positions, plus the final position when that colour
resigned on move (the only resign positive). Sampled positions carry weight n_candidates / K so the resign
hazard stays unbiased.

    python -m doppel_lab.v0_data prep --split val
    python -m doppel_lab.v0_data prep --split train
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import chess
import numpy as np
import polars as pl

from doppel_model import MOVE_INDEX, board_planes, mirror_uci

ROOT = Path(__file__).resolve().parents[2]
V0 = ROOT / "data/v0"
HIST = 8
END = {"resign": 0, "mate": 1, "flag": 2, "draw": 3, "other": 4}
ENC_FEATS = 8   # per context ply, see encoder_tokens
CLK_FEATS = 7   # see clock_feats
TEMPO_FEATS = 6
MAX_LEGAL = 224


def prep(split: str, band: int = 1500) -> Path:
    g = pl.read_parquet(V0 / f"games_{band}_{split}.parquet")
    lab = pl.read_parquet(V0 / f"resign_{band}_{split}.parquet")
    g = g.join(lab, on="game_id", how="left", maintain_order="left")
    assert g["end"].null_count() == 0
    shards = sorted((V0 / f"teacher_{band}_{split}").glob("shard-*.npz"))
    gids, plp, rank, ent = [], [], [], []
    for s in shards:
        z = np.load(s)
        gids.append(z["game_ids"])
        plp.append(z["played_lp"])
        ent.append(z["entropy"])
        hit = z["top_idx"] == z["played_idx"][:, None]
        rank.append(np.where(hit.any(1), hit.argmax(1), 16).astype(np.uint8))
    gids = np.concatenate(gids)
    assert (gids == g["game_id"].to_numpy()).all(), "teacher shards are not in games parquet order"
    n = g["moves"].list.len().to_numpy()
    off = np.r_[0, np.cumsum(n)].astype(np.int64)
    uci = np.array(g["moves"].explode().to_list(), dtype="S5")
    clock = g["clocks"].explode().cast(pl.Float32).fill_null(float("nan")).to_numpy()
    assert len(uci) == off[-1] == len(clock) == sum(map(len, plp))
    out = V0 / f"index_{band}_{split}.npz"
    np.savez(out, game_id=gids, off=off, white_elo=g["white_elo"].to_numpy().astype(np.int32),
             black_elo=g["black_elo"].to_numpy().astype(np.int32), base=g["base"].to_numpy().astype(np.int32),
             inc=g["inc"].to_numpy().astype(np.int32), end=g["end"].replace_strict(END).to_numpy().astype(np.int8),
             resigner=g["resigner"].replace_strict({"w": 1, "b": 2}, default=0).to_numpy().astype(np.int8),
             on_move=g["on_move"].to_numpy().astype(bool), uci=uci, clock=clock,
             played_lp=np.concatenate(plp), rank=np.concatenate(rank), entropy=np.concatenate(ent))
    print(f"{split}: {len(gids):,} games, {off[-1]:,} positions -> {out}")
    return out


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def think_seconds(clock: np.ndarray, k: int, base: float, inc: float) -> tuple[float, float]:
    """(think s, mover's remaining before the move) for ply k; NaN when unlabelled (each side's first move)."""
    if k < 2 or not (np.isfinite(clock[k]) and np.isfinite(clock[k - 2])):
        return math.nan, float(clock[k - 2]) if k >= 2 and np.isfinite(clock[k - 2]) else float(base)
    before = float(clock[k - 2])
    return max(0.0, before - float(clock[k]) + inc), before


def remaining(clock: np.ndarray, k: int, base: float) -> tuple[float, float]:
    """(mover's, opponent's) clock in the position before ply k."""
    own = clock[k - 2] if k >= 2 else base
    opp = clock[k - 1] if k >= 1 else base
    own = float(own) if np.isfinite(own) else float(base)
    opp = float(opp) if np.isfinite(opp) else float(base)
    return own, opp


def clock_feats(own: float, opp: float, base: float, inc: float, k: int) -> list[float]:
    b = max(base, 1.0)
    return [own / b, opp / b, math.log1p(own), math.log1p(opp), inc / b, math.log1p(base), k / 100]


def tempo_feats(fracs_own: list[float], fracs_opp: list[float]) -> list[float]:
    out = []
    for fr in (fracs_own, fracs_opp):
        if fr:
            lg = _logit(np.array(fr))
            w = 0.7 ** np.arange(len(lg))[::-1]
            out += [float(np.median(lg)), float((w * lg).sum() / w.sum()), math.log1p(len(lg))]
        else:
            out += [0.0, 0.0, 0.0]
    return out


class Index:
    def __init__(self, path: Path):
        z = np.load(path)
        self.__dict__.update({k: z[k] for k in z.files})
        self.G = len(self.game_id)

    def plies(self, g: int) -> int:
        return int(self.off[g + 1] - self.off[g])


def build_sample(ix: Index, g: int, color: int, ctx_first: bool, K: int, rng: np.random.Generator):
    """color 0 = white, 1 = black. Returns a dict of numpy arrays, or None if there is nothing to predict."""
    a, n = int(ix.off[g]), ix.plies(g)
    mid = n // 2
    ctx = range(0, mid) if ctx_first else range(mid, n)
    tgt_lo, tgt_hi = (mid, n) if ctx_first else (0, mid)
    uci = [u.decode() for u in ix.uci[a:a + n]]
    clock = ix.clock[a:a + n]
    base, inc = float(ix.base[g]), float(ix.inc[g])
    elos = (int(ix.white_elo[g]), int(ix.black_elo[g]))
    resigned_on_move = ix.end[g] == END["resign"] and ix.resigner[g] == color + 1 and ix.on_move[g]
    final = n if (ctx_first and resigned_on_move and n % 2 == color) else None

    # Think fractions of every labelled move, for the encoder and the running tempo.
    frac = np.full(n, np.nan)
    for k in range(n):
        t, before = think_seconds(clock, k, base, inc)
        if np.isfinite(t):
            frac[k] = (t + 0.5) / (before + inc + 1.0)

    enc = []
    for k in ctx:
        own_ply = (k % 2) == color
        own, _ = remaining(clock, k, base)
        f = frac[k]
        enc.append([float(own_ply), float(ix.played_lp[a + k]), ix.rank[a + k] / 16.0, float(ix.entropy[a + k]),
                    float(_logit(f)) if np.isfinite(f) else 0.0, float(np.isfinite(f)), own / max(base, 1.0),
                    k / 100])
    if not enc or not any(e[0] for e in enc):
        return None

    cand = [k for k in range(tgt_lo, tgt_hi) if k % 2 == color]
    if not cand and final is None:
        return None
    pick = sorted(rng.choice(cand, size=min(K, len(cand)), replace=False).tolist()) if cand else []
    w_pick = len(cand) / max(len(pick), 1)
    targets = pick + ([final] if final is not None else [])
    weights = [w_pick] * len(pick) + ([1.0] if final is not None else [])

    # Replay to the last target, keeping board planes for the history window.
    last = max(targets)
    b = chess.Board()
    planes = [board_planes(b)]
    boards_at = {}
    for k in range(last + 1):
        if k in targets:
            boards_at[k] = b.copy(stack=False)
        if k < last:
            b.push_uci(uci[k])
            planes.append(board_planes(b))
    T = len(targets)
    tok = np.zeros((T, 64, 12 * HIST + 1), np.uint8)
    legal = np.full((T, MAX_LEGAL), -1, np.int16)
    played = np.full(T, -1, np.int64)
    think = np.full((T, 3), np.nan, np.float32)  # think s, remaining before, inc
    clk = np.zeros((T, CLK_FEATS), np.float32)
    tempo = np.zeros((T, TEMPO_FEATS), np.float32)
    resign = np.zeros(T, np.float32)
    for i, k in enumerate(targets):
        h = planes[max(0, k - HIST + 1):k + 1]
        h = [h[0]] * (HIST - len(h)) + h
        tok[i, :, :12 * HIST] = np.concatenate(h, axis=1)
        bk = boards_at[k]
        white = bk.turn == chess.WHITE
        idx = sorted(MOVE_INDEX[m.uci() if white else mirror_uci(m.uci())] for m in bk.legal_moves)
        legal[i, :len(idx)] = idx[:MAX_LEGAL]
        if k < n:
            played[i] = MOVE_INDEX[uci[k] if white else mirror_uci(uci[k])]
            t, before = think_seconds(clock, k, base, inc)
            think[i] = (t, before, inc)
        else:
            resign[i] = 1.0
        own, opp = remaining(clock, k, base)
        clk[i] = clock_feats(own, opp, base, inc, k)
        prev_own = [frac[j] for j in range(color, k, 2) if np.isfinite(frac[j])]
        prev_opp = [frac[j] for j in range(1 - color, k, 2) if np.isfinite(frac[j])]
        tempo[i] = tempo_feats(prev_own, prev_opp)
    es, eo = elos[color], elos[1 - color]
    return {"enc": np.array(enc, np.float32), "tok": tok, "legal": legal, "played": played, "think": think,
            "clk": clk, "tempo": tempo, "resign": resign, "weight": np.array(weights, np.float32),
            "elo_self": np.full(T, es, np.int64), "elo_opp": np.full(T, eo, np.int64),
            "ncand": np.array([max(len(cand), 1)], np.float32), "ply": np.array(targets, np.int64),
            "game": np.full(T, g, np.int64),
            "cond": np.array([(es - 1500) / 400, (eo - 1500) / 400, math.log1p(base), inc / max(base, 1.0)],
                             np.float32)}


def collate(samples: list[dict]) -> dict:
    """Pads encoder sequences; flattens target positions with a sample index."""
    import torch
    samples = [s for s in samples if s is not None]
    L = max(len(s["enc"]) for s in samples)
    enc = np.zeros((len(samples), L, ENC_FEATS), np.float32)
    mask = np.zeros((len(samples), L), bool)
    for i, s in enumerate(samples):
        enc[i, :len(s["enc"])] = s["enc"]
        mask[i, :len(s["enc"])] = True
    out = {"enc": torch.from_numpy(enc), "enc_mask": torch.from_numpy(mask),
           "cond": torch.from_numpy(np.stack([s["cond"] for s in samples])),
           "ncand": torch.from_numpy(np.concatenate([s["ncand"] for s in samples])),
           "sid": torch.from_numpy(np.concatenate([np.full(len(s["played"]), i) for i, s in enumerate(samples)]))}
    for k in ("tok", "legal", "played", "think", "clk", "tempo", "resign", "weight", "elo_self", "elo_opp", "ply", "game"):
        out[k] = torch.from_numpy(np.concatenate([s[k] for s in samples]))
    return out


class Samples:
    """torch-style dataset over (game, colour, context half); a fresh position draw each epoch via the seed."""

    def __init__(self, ix: Index, K: int = 8, seed: int = 0, games: np.ndarray | None = None):
        self.ix, self.K, self.seed = ix, K, seed
        self.games = np.arange(ix.G) if games is None else games
        self.epoch = 0

    def __len__(self):
        return len(self.games) * 4

    def __getitem__(self, i):
        g = int(self.games[i // 4])
        color, ctx_first = (i % 4) // 2, bool(i % 2)
        rng = np.random.default_rng((self.seed, self.epoch, i))
        return build_sample(self.ix, g, color, ctx_first, self.K, rng)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--split", choices=["train", "val"], default="val")
    p.add_argument("--band", type=int, default=1500)
    a = ap.parse_args()
    prep(a.split, a.band)

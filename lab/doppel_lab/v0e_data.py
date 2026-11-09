"""Doppel v0e data: a target game of player P plus a context set of P's earlier games (R2, SCIENCE.md §3).

    python -m doppel_lab.v0e_data select-valrich        # CPU: held-out players with >= 48 band games
    python -m doppel_lab.v0e_data prep --split valrich  # CPU: v0-format index + meta (order, players, surface)
    python -m doppel_lab.v0e_data prep --split train
    python -m doppel_lab.v0e_data norm                  # CPU: context-feature mean/std from train (model buffers)

Splits. train = data/v0/games_1500_train.parquet, the v0d training games (player-disjoint, split.py). valrich =
held-out 2026-10 band games of VAL_PLAYERS held-out players with >= VAL_PER band games each, their first VAL_PER
games in time order: the v0 val split (20k games, at most 22 per player) is too thin in players to measure z at
k = 10-40. Lichess 2026-10 only.

Time order. t = the game's row in the corpus shards (data/corpus/2026-10/{train,heldout}), which is dump order
(game start) up to the corpus writer's chunk jitter: chunks of 4,000 games finish out of order within a window
of a few minutes of play. Context games are P's games in the same split with a smaller t; the target game is
never in its own context.

Per context game and side (CTX_FEATS = 19 + the 1024-d Maia embedding, from v0e_vec):
  scal (5)    Maia-3 at 1500: played log-prob mean / std, entropy, top-1 agreement, mean log1p(rank)
  sweep (2)   ll(1000) - ll(1500), ll(2000) - ll(1500): the side's strength relative to the 1500 input
  surface (12, SURF): think-fraction logit mean / std, share of moves <= 1 s, log1p(own moves), final clock /
              base, score, resigned, lost on time, got mated, log1p(base), inc / base, played White
No ratings of the context games: the target game's ratings are the cond of q and p, and the players suite scores
z with a rating-blind cond, so z's strength information must come from the moves.

Sample = (target game g, colour c, k). k = 0 for P's first game in the split and with probability P_K0
otherwise; else k = min(log-uniform 1..K_MAX, games available). The context is P's k most recent earlier games.
Targets as v0d but over the whole target game: up to K own decision positions, plus the resign position.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np
import polars as pl

from doppel_lab.split import held_out
from doppel_lab.v0_data import END, V0, Index, _logit, resigned_on_move, targets_sample, think_seconds
from doppel_lab.v0e_vec import V0E, Vecs, games_path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data/corpus/2026-10"
SURF = ["tf_mean", "tf_std", "fast_share", "log_moves", "clock_end", "score", "resigned", "flag_loss", "mated",
        "log_base", "inc_frac", "white"]
CTX_FEATS = 5 + 2 + len(SURF)
EMB_DIM = 1024
K_MAX, P_K0 = 40, 0.05
VAL_PLAYERS, VAL_PER, VAL_TGT = 600, 48, 40       # targets: each player's games VAL_TGT..VAL_PER-1
KS_EVAL = (0, 1, 2, 5, 10, 20, 40)
c = pl.col


# ------------------------------------------------------------------ selection and order
def corpus_order(src: str) -> pl.DataFrame:
    """game_id -> t (row in the corpus shards of `src`, which is dump order)."""
    files = sorted((CORPUS / src).glob("part-*.parquet"))
    return pl.concat([pl.read_parquet(f, columns=["game_id"]) for f in files]).with_row_index("t")


def select_valrich(band: int = 1500, n_players: int = VAL_PLAYERS, per: int = VAL_PER, seed: int = 0):
    files = sorted((CORPUS / "heldout").glob("part-*.parquet"))
    g = pl.concat([pl.read_parquet(f) for f in files]).with_row_index("t")
    g = g.filter((c("band") == band) & (c("moves").list.len() >= 20))
    names = pl.concat([g["white"], g["black"]]).str.to_lowercase().value_counts().filter(c("count") >= per)
    names = sorted(n for n in names["white"].to_list() if held_out(n))
    rng = np.random.default_rng(seed)
    keep = sorted(rng.choice(names, min(n_players, len(names)), replace=False).tolist())
    sides = pl.concat([g.select("t", "game_id", player=c(col).str.to_lowercase()) for col in ("white", "black")])
    sides = sides.filter(c("player").is_in(keep)).sort("t").with_columns(j=pl.int_range(pl.len()).over("player"))
    sides = sides.filter(c("j") < per)
    out = g.filter(c("game_id").is_in(sides["game_id"].unique().implode())).sort("t")
    V0E.mkdir(parents=True, exist_ok=True)
    out.write_parquet(games_path("valrich", band))
    pl.DataFrame({"player": keep}).write_parquet(V0E / f"valrich_players_{band}.parquet")
    print(f"valrich: {len(keep)} of {len(names)} eligible held-out players, {out.height:,} games, "
          f"{out['moves'].list.len().sum():,} positions -> {games_path('valrich', band)}", flush=True)


# ------------------------------------------------------------------ surface stats
def _end_row(args):
    from doppel_lab.resign import label_game
    _, end, who, on_move, _ = label_game(args)
    return END[end], {"w": 1, "b": 2}.get(who, 0), bool(on_move)


def end_labels(g: pl.DataFrame, workers: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(end code, resigner 0/1 white/2 black, on_move) per game, as v0_data.prep / resign.label."""
    with mp.get_context("fork").Pool(workers) as pool:
        rows = pool.map(_end_row, g.select("game_id", "moves", "result", "termination").iter_rows(), chunksize=512)
    e, r, o = zip(*rows)
    return np.array(e, np.int8), np.array(r, np.int8), np.array(o, bool)


def side_surface(clock: np.ndarray, base: float, inc: float, side: int, result: str, end: int,
                 resigner: int) -> list[float]:
    n = len(clock)
    fr, fast = [], []
    for k in range(side, n, 2):
        t, before = think_seconds(clock, k, base, inc)
        if np.isfinite(t):
            fr.append(_logit((t + 0.5) / (before + inc + 1.0)))
            fast.append(t <= 1.0)
    own = clock[side::2]
    own = own[np.isfinite(own)]
    won = result == ("1-0" if side == 0 else "0-1")
    lost = result == ("0-1" if side == 0 else "1-0")
    score = 1.0 if won else 0.0 if lost else 0.5
    return [float(np.mean(fr)) if fr else 0.0, float(np.std(fr)) if len(fr) > 1 else 0.0,
            float(np.mean(fast)) if fast else 0.0, math.log1p(len(range(side, n, 2))),
            min(float(own[-1]) / max(base, 1.0), 3.0) if len(own) else 1.0, score,
            float(end == END["resign"] and resigner == side + 1), float(lost and end == END["flag"]),
            float(lost and end == END["mate"]), math.log1p(base), min(inc / max(base, 1.0), 1.0), float(side == 0)]


def surface(g: pl.DataFrame, end: np.ndarray, resigner: np.ndarray) -> np.ndarray:
    """(G, 2, len(SURF)) float32 from moves/clocks/base/inc/result and the end labels."""
    out = np.zeros((g.height, 2, len(SURF)), np.float32)
    for i, (clocks, base, inc, res) in enumerate(g.select("clocks", "base", "inc", "result").iter_rows()):
        cl = np.array([np.nan if x is None else x for x in clocks], np.float32)
        for s in (0, 1):
            out[i, s] = side_surface(cl, float(base), float(inc), s, res, int(end[i]), int(resigner[i]))
    return out


# ------------------------------------------------------------------ prep
def prep(split: str, band: int = 1500, workers: int = 16):
    g = pl.read_parquet(games_path(split, band))
    if split == "valrich":
        end, res, onm = end_labels(g, workers)
        n = g["moves"].list.len().to_numpy()
        off = np.r_[0, np.cumsum(n)].astype(np.int64)
        P = int(off[-1])
        # v0-format index. played_lp / rank / entropy are the v0d encoder's inputs and are unused by v0e: zeros.
        np.savez(V0E / f"index_{band}_{split}.npz", game_id=g["game_id"].to_numpy().astype("U"), off=off,
                 white_elo=g["white_elo"].to_numpy().astype(np.int32),
                 black_elo=g["black_elo"].to_numpy().astype(np.int32), base=g["base"].to_numpy().astype(np.int32),
                 inc=g["inc"].to_numpy().astype(np.int32), end=end, resigner=res, on_move=onm,
                 uci=np.array(g["moves"].explode().to_list(), dtype="S5"),
                 clock=g["clocks"].explode().cast(pl.Float32).fill_null(float("nan")).to_numpy(),
                 played_lp=np.zeros(P, np.float16), rank=np.zeros(P, np.uint8), entropy=np.zeros(P, np.float16))
        t = g["t"].to_numpy()
    else:
        ix = Index(V0 / f"index_{band}_{split}.npz")
        assert (ix.game_id == g["game_id"].to_numpy()).all()
        end, res = ix.end, ix.resigner
        o = corpus_order("train" if split == "train" else "heldout")
        t = g.select("game_id").join(o, on="game_id", how="left", maintain_order="left")["t"].to_numpy()
        assert not np.isnan(t.astype(float)).any()
    names = pl.concat([g["white"], g["black"]]).str.to_lowercase()
    uniq, inv = np.unique(names.to_numpy(), return_inverse=True)
    pid = inv.reshape(2, g.height).T.astype(np.int32)
    surf = surface(g, end, res)
    np.savez(V0E / f"meta_{band}_{split}.npz", game_id=g["game_id"].to_numpy().astype("U"), t=t.astype(np.int64), pid=pid,
             players=uniq, surf=surf)
    print(f"{split}: {g.height:,} games, {len(uniq):,} players -> {V0E / f'meta_{band}_{split}.npz'}", flush=True)


def ctx_feats(scal: np.ndarray, sweep: np.ndarray, surf: np.ndarray) -> np.ndarray:
    """(..., CTX_FEATS) from scal (..., 5), sweep (..., 3) = ll at 1000/1500/2000, surface (..., 12)."""
    d = np.stack([sweep[..., 0] - sweep[..., 1], sweep[..., 2] - sweep[..., 1]], -1)
    return np.nan_to_num(np.concatenate([scal, d, surf], -1).astype(np.float32))


def norm(band: int = 1500, n: int = 200_000, seed: int = 0):
    v, m = Vecs("train", band), np.load(V0E / f"meta_{band}_train.npz", allow_pickle=True)
    rng = np.random.default_rng(seed)
    gi = np.sort(rng.choice(len(v.game_ids), min(n, len(v.game_ids)), replace=False))
    si = rng.integers(0, 2, len(gi))
    e = v.emb[gi][np.arange(len(gi)), si].astype(np.float32)
    f = ctx_feats(v.scal[gi, si], v.sweep[gi, si], m["surf"][gi, si])
    out = V0E / f"norm_{band}.npz"
    np.savez(out, emb_mu=e.mean(0), emb_sd=e.std(0) + 1e-4, feat_mu=f.mean(0), feat_sd=f.std(0) + 1e-4)
    print(f"norm from {len(gi):,} sides -> {out}")


# ------------------------------------------------------------------ samples
class Split:
    """Index + vectors + meta of one split, and each side's place in its player's time-ordered history."""

    def __init__(self, split: str, band: int = 1500, vec=None, frac: float = 1.0, frac_seed: int = 0):
        self.ix = Index(V0 / f"index_{band}_{split}.npz" if split in ("train", "val")
                        else V0E / f"index_{band}_{split}.npz")
        m = np.load(V0E / f"meta_{band}_{split}.npz", allow_pickle=True)
        self.t, self.pid, self.players, self.surf = m["t"], m["pid"], m["players"], m["surf"]
        self.vec = vec if vec is not None else Vecs(split, band)
        assert (self.vec.game_ids == self.ix.game_id).all() and (m["game_id"] == self.ix.game_id).all()
        # data-scaling runs: keep a random share of games; dropped games vanish from targets and histories alike
        self.kept = (np.arange(len(self.t)) if frac >= 1.0 else
                     np.sort(np.random.default_rng(frac_seed).permutation(len(self.t))[:int(round(frac * len(self.t)))]))
        self._order()
        self.feat = ctx_feats(self.vec.scal, self.vec.sweep, self.surf)  # (G, 2, CTX_FEATS)

    def _order(self):
        G = len(self.t)
        g = np.r_[self.kept, self.kept]
        col = np.r_[np.zeros(len(self.kept), np.int64), np.ones(len(self.kept), np.int64)]
        p = self.pid[g, col]
        order = np.lexsort((col, self.t[g], p))                # by player, then time
        self.hg, self.hc, hp = g[order], col[order], p[order]
        start = np.r_[0, np.flatnonzero(np.diff(hp)) + 1]
        self.start_of = np.repeat(start, np.diff(np.r_[start, len(hp)]))
        self.pos = np.full((G, 2), -1, np.int64)               # (g, c) -> index in hg/hc (-1: dropped game)
        self.pos[self.hg, self.hc] = np.arange(len(self.hg))

    def history(self, g: int, col: int) -> tuple[np.ndarray, np.ndarray]:
        """P's earlier games in this split, oldest first: (games, colours)."""
        i = self.pos[g, col]
        s = self.start_of[i]
        return self.hg[s:i], self.hc[s:i]


def draw_k(n_hist: int, rng: np.random.Generator) -> int:
    if n_hist == 0 or rng.random() < P_K0:
        return 0
    return int(min(n_hist, math.floor(math.exp(rng.random() * math.log(K_MAX + 1)))))


def build(sp: Split, g: int, col: int, k: int, K: int, rng: np.random.Generator):
    ix = sp.ix
    n = ix.plies(g)
    cand = list(range(col, n, 2))
    final = n if resigned_on_move(ix, g, col) and n % 2 == col else None
    out = targets_sample(ix, g, col, cand, final, K, rng)
    if out is None:
        return None
    hg, hc = sp.history(g, col)
    hg, hc = hg[len(hg) - k:] if k else hg[:0], hc[len(hc) - k:] if k else hc[:0]
    out["ctx_emb"] = np.asarray(sp.vec.emb[hg, hc]) if k else np.zeros((0, EMB_DIM), np.float16)
    out["ctx_feat"] = sp.feat[hg, hc] if k else np.zeros((0, CTX_FEATS), np.float32)
    out["pid"] = np.array([sp.pid[g, col]], np.int64)
    out["nctx"] = np.array([k], np.int64)
    return out


TGT_KEYS = ("tok", "legal", "played", "think", "clk", "tempo", "resign", "weight", "elo_self", "elo_opp", "ply", "game")


def collate(samples: list[dict]) -> dict:
    import torch
    samples = [s for s in samples if s is not None]
    L = max(1, max(len(s["ctx_feat"]) for s in samples))
    emb = np.zeros((len(samples), L, EMB_DIM), np.float16)
    feat = np.zeros((len(samples), L, CTX_FEATS), np.float32)
    mask = np.zeros((len(samples), L), bool)
    for i, s in enumerate(samples):
        k = len(s["ctx_feat"])
        emb[i, :k], feat[i, :k], mask[i, :k] = s["ctx_emb"], s["ctx_feat"], True
    out = {"ctx_emb": torch.from_numpy(emb), "ctx_feat": torch.from_numpy(feat), "ctx_mask": torch.from_numpy(mask),
           "cond": torch.from_numpy(np.stack([s["cond"] for s in samples])),
           "sid": torch.from_numpy(np.concatenate([np.full(len(s["played"]), i) for i, s in enumerate(samples)]))}
    for k in ("ncand", "pid", "nctx", *TGT_KEYS):
        out[k] = torch.from_numpy(np.concatenate([s[k] for s in samples]))
    return out


class TrainSamples:
    """Every (game, colour) of the split; fresh k, context and target positions each epoch via the seed."""

    def __init__(self, sp: Split, K: int = 8, seed: int = 0):
        self.sp, self.K, self.seed, self.epoch = sp, K, seed, 0

    def __len__(self):
        return len(self.sp.kept) * 2

    def __getitem__(self, i):
        g, col = int(self.sp.kept[i // 2]), i % 2
        rng = np.random.default_rng((self.seed, self.epoch, i))
        hg, _ = self.sp.history(g, col)
        return build(self.sp, g, col, draw_k(len(hg), rng), self.K, rng)


class EvalSamples:
    """Fixed (target, k) grid: each side at history position VAL_TGT..VAL_PER-1 of a player with >= VAL_PER
    games, at every k in ks (most recent k earlier games). Target positions depend only on (g, colour), so the
    rows for one target are paired across k."""

    def __init__(self, sp: Split, ks=KS_EVAL, K: int = 8, max_players: int | None = None, seed: int = 12345):
        self.sp, self.K, self.seed = sp, K, seed
        hp = sp.pid[sp.hg, sp.hc]
        j = np.arange(len(hp)) - sp.start_of
        n = np.bincount(hp, minlength=len(sp.players))[hp]
        ok = (j >= VAL_TGT) & (j < VAL_PER) & (n >= VAL_PER)
        if max_players is not None:
            keep = np.unique(hp[ok])[:max_players]
            ok &= np.isin(hp, keep)
        self.items = [(int(sp.hg[i]), int(sp.hc[i]), k) for i in np.flatnonzero(ok) for k in ks]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        g, col, k = self.items[i]
        rng = np.random.default_rng((self.seed, g, col))
        return build(self.sp, g, col, k, self.K, rng)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("select-valrich")
    p = sub.add_parser("prep")
    p.add_argument("--split", default="valrich")
    p.add_argument("--workers", type=int, default=16)
    sub.add_parser("norm")
    a = ap.parse_args()
    if a.cmd == "select-valrich":
        select_valrich()
    elif a.cmd == "prep":
        prep(a.split, workers=a.workers)
    else:
        norm()

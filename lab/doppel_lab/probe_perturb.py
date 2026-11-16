"""One-piece perturbation probe: does Maia-3's board representation tell "a rook moved one file" from "a pawn moved
one square" the way its own policy does? (SCIENCE.md §8, repertoire memory: backoff to near positions.)

    python -m doppel_lab.probe_perturb --n 300                  # CPU smoke
    python -m doppel_lab.probe_perturb --n 4000 --gpu-ok        # through the warden -> lab/results/probe_perturb.json

Exact repeat visits are 98% in plies 0-9, so player-level ground truth later in the game needs backoff: reuse what
a player did at a position that is near the current one. One square can change everything (a rook leaving a file,
a queen stepping into a fork), so "near on the board" is not enough. This asks whether a network's representation
separates the two cases before we build retrieval on it.

Base positions: one per game from v0 val (1500 band), uniform over plies 16-60, the side to move's view.
Perturbations: one non-king piece of either side stepped to an empty king-neighbour square (8 directions; pawns
too). Kept if the board is valid (no pawn on a back rank, the side not to move not in check), the side to move has
a legal move, castling rights cleaned, no en passant on either board. Up to PER per base, so every pair has
exactly two changed squares: surface distance is the same for all of them by construction.

Per pair, Maia-3 79M at 1500 / 1500 with no move history (the current board repeated, as at ply 0, so the history
can't contradict the perturbation):
  policy    Jensen-Shannon divergence of the move distributions (moves illegal on one board have p = 0 there),
            and whether the top move changes. This is the target: does the difference matter for the decision?
  value     |change in expected score| from the W/D/L head
  rep       cosine distance of the final-layer features (mean over squares, and the max over squares of the
            per-square cosine distance), and of a middle transformer layer
Analysis: mean JS by piece type × owner (mover / opponent) and by attack change (how many squares the piece
attacks before xor after); Spearman of each rep distance with JS overall and within piece type; AUC of each rep
distance for "the decision changed" (JS > 0.1 or top move changed), overall and within piece type.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import chess
import numpy as np
import polars as pl
import torch
from scipy.stats import rankdata, spearmanr

from doppel_lab.v0_data import HIST
from doppel_model import MOVE_INDEX, board_planes, mirror_uci

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "lab/results/probe_perturb.json"
PER = 8
STEPS = (-9, -8, -7, -1, 1, 7, 8, 9)
NAMES = {1: "P", 2: "N", 3: "B", 4: "R", 5: "Q"}


def bases(n: int, seed: int) -> list[chess.Board]:
    g = pl.read_parquet(ROOT / "data/v0/games_1500_val.parquet", columns=["moves"])
    rng = np.random.default_rng(seed)
    out = []
    for i in rng.permutation(g.height):
        mv = g["moves"][int(i)].to_list()
        if len(mv) < 24:
            continue
        k = int(rng.integers(16, min(60, len(mv) - 1) + 1))
        b = chess.Board()
        for u in mv[:k]:
            b.push_uci(u)
        if b.is_game_over():
            continue
        b.ep_square = None
        out.append(b.copy(stack=False))
        if len(out) == n:
            break
    return out


def perturb(b: chess.Board, rng: np.random.Generator) -> list[tuple[chess.Board, dict]]:
    cands = []
    for sq, p in b.piece_map().items():
        if p.piece_type == chess.KING:
            continue
        for d in STEPS:
            to = sq + d
            if not 0 <= to < 64 or abs(chess.square_file(to) - chess.square_file(sq)) > 1 or b.piece_at(to):
                continue
            cands.append((sq, to, p))
    rng.shuffle(cands)
    out = []
    for sq, to, p in cands:
        c = b.copy(stack=False)
        c.remove_piece_at(sq)
        c.set_piece_at(to, p)
        c.castling_rights = c.clean_castling_rights()
        c.ep_square = None
        if not c.is_valid() or not any(True for _ in c.legal_moves):
            continue
        att0, att1 = b.attacks(sq), c.attacks(to)
        out.append((c, {"piece": NAMES[p.piece_type], "owner": "mover" if p.color == b.turn else "opp",
                        "d_attacks": len(att0 ^ att1), "check_changed": int(b.is_check() != c.is_check()),
                        "from": sq, "to": to}))
        if len(out) == PER:
            break
    return out


def tok(b: chess.Board) -> np.ndarray:
    x = np.zeros((64, 12 * HIST + 1), np.float32)
    x[:, :12 * HIST] = np.tile(board_planes(b), (1, HIST))
    return x


def legal_idx(b: chess.Board) -> list[int]:
    w = b.turn == chess.WHITE
    return [MOVE_INDEX[m.uci() if w else mirror_uci(m.uci())] for m in b.legal_moves]


@torch.no_grad()
def run_model(boards: list[chess.Board], device: str, batch: int = 512):
    from doppel_lab.engines import Maia3
    m = Maia3("79m", device).model
    feats = {}
    layers = next(mod for mod in m.transformer.modules() if isinstance(mod, torch.nn.ModuleList))
    mid = len(layers) // 2
    m.last_ln.register_forward_hook(lambda mod, i, o: feats.__setitem__("last", o))
    layers[mid].register_forward_hook(lambda mod, i, o: feats.__setitem__("mid", o[0] if isinstance(o, tuple) else o))
    out = {"logp": [], "ev": [], "last": [], "mid": []}
    for s in range(0, len(boards), batch):
        bb = boards[s:s + batch]
        x = torch.from_numpy(np.stack([tok(b) for b in bb])).to(device)
        e = torch.full((len(bb),), 1500.0, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            logits, value, _ = m(x, e, e)
        logits = logits.float()
        mask = torch.zeros_like(logits, dtype=torch.bool)
        for i, b in enumerate(bb):
            mask[i, legal_idx(b)] = True
        out["logp"].append(logits.masked_fill(~mask, -1e9).log_softmax(-1).cpu())
        wdl = value.float().softmax(-1)
        out["ev"].append((wdl[:, 0] + 0.5 * wdl[:, 1]).cpu())
        for k in ("last", "mid"):
            f = feats[k].float()
            out[k].append((f if f.dim() == 3 else f[:, None]).cpu().half())
    return {k: torch.cat(v) for k, v in out.items()}, mid, len(layers)


def js(lp, lq):
    p, q = lp.exp(), lq.exp()
    mm = 0.5 * (p + q)
    lm = (mm + 1e-30).log()
    kl = lambda a, la: torch.where(a > 0, a * (la - lm), 0.0).sum(-1)
    return 0.5 * kl(p, lp) + 0.5 * kl(q, lq)


def cos_d(a, b):
    a, b = a.float(), b.float()
    pooled = 1 - torch.nn.functional.cosine_similarity(a.mean(1), b.mean(1), dim=-1)
    per_sq = (1 - torch.nn.functional.cosine_similarity(a, b, dim=-1)).max(1).values
    return pooled, per_sq


def auc(score, y):
    y = y.astype(bool)
    if y.all() or not y.any():
        return None
    r = rankdata(score)
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def main(n: int, device: str, seed: int):
    rng = np.random.default_rng(seed)
    base = bases(n, seed)
    pairs = [(i, c, meta) for i, b in enumerate(base) for c, meta in perturb(b, rng)]
    print(f"{len(base)} bases, {len(pairs)} perturbations", flush=True)
    r, mid, nl = run_model(base + [c for _, c, _ in pairs], device)
    ia = torch.tensor([i for i, _, _ in pairs])
    ib = torch.arange(len(pairs)) + len(base)
    d = pl.DataFrame([meta for _, _, meta in pairs]).with_columns(
        js=pl.Series(js(r["logp"][ia], r["logp"][ib]).numpy()),
        top_changed=pl.Series((r["logp"][ia].argmax(-1) != r["logp"][ib].argmax(-1)).numpy().astype(np.int8)),
        d_ev=pl.Series((r["ev"][ia] - r["ev"][ib]).abs().numpy()))
    reps = {}
    for k in ("last", "mid"):
        pooled, per_sq = cos_d(r[k][ia], r[k][ib])
        reps[f"{k}_pooled"], reps[f"{k}_maxsq"] = pooled.numpy(), per_sq.numpy()
    d = d.with_columns(**{k: pl.Series(v) for k, v in reps.items()},
                       changed=((pl.col("js") > 0.1) | (pl.col("top_changed") == 1)).cast(pl.Int8))
    c = pl.col
    out = {"bases": len(base), "pairs": d.height, "mid_layer": f"{mid} of {nl}",
           "changed_share": float(d["changed"].mean()), "js_mean": float(d["js"].mean())}
    out["by_piece_owner"] = {f"{r_['piece']}_{r_['owner']}": {k: r_[k] for k in ("n", "js", "top_changed", "d_ev", "changed")}
                             for r_ in d.group_by("piece", "owner").agg(n=pl.len(), js=c("js").mean(),
                                 top_changed=c("top_changed").mean(), d_ev=c("d_ev").mean(),
                                 changed=c("changed").mean()).sort("piece", "owner").to_dicts()}
    out["by_d_attacks"] = {str(r_["b"]): {k: r_[k] for k in ("n", "js", "changed")} for r_ in
                           d.with_columns(b=c("d_attacks").cut([2, 5, 9], labels=["0-2", "3-5", "6-9", "10+"]))
                           .group_by("b").agg(n=pl.len(), js=c("js").mean(), changed=c("changed").mean()).sort("b").to_dicts()}
    y = d["changed"].to_numpy()
    out["rep"] = {}
    for k in (*reps, "d_attacks"):
        s = d[k].to_numpy().astype(np.float64)
        e = {"spearman_js": float(spearmanr(s, d["js"]).statistic), "auc_changed": auc(s, y), "by_piece": {}}
        for p in NAMES.values():
            m = (d["piece"] == p).to_numpy()
            if m.sum() > 30:
                e["by_piece"][p] = {"spearman_js": float(spearmanr(s[m], d["js"].to_numpy()[m]).statistic),
                                    "auc_changed": auc(s[m], y[m])}
        # within-piece AUC averaged (does the rep separate cases the piece type alone can't?)
        vals = [v["auc_changed"] for v in e["by_piece"].values() if v["auc_changed"] is not None]
        e["auc_within_piece_mean"] = float(np.mean(vals)) if vals else None
        out["rep"][k] = e
    d.write_parquet(ROOT / "data/probe/perturb_pairs.parquet")
    res = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    res["smoke" if n < 1000 else "main"] = out
    RESULTS.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-ok", action="store_true")
    a = ap.parse_args()
    main(a.n, "cuda" if a.gpu_ok else "cpu", a.seed)

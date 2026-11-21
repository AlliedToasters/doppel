"""Neuron atlas of Maia-3-79M's adapted MLPs (blocks 5-8, 4 x 2048 GELU neurons) on the fixed probe set, and each
player's adapter seen as a per-neuron modulation. Ties three layers together: chess concepts (python-chess
primitives), model internals (neurons) and players (doppel adapters).

    python -m doppel_lab.neuron_atlas base                 # CPU ok: base activations -> data/allie/neuron_atlas/base_b<l>.npy
    python -m doppel_lab.neuron_atlas concepts             # CPU: chess primitives per probe position and square
    python -m doppel_lab.neuron_atlas label                # CPU: per-neuron concept match -> lab/results/neuron_atlas.json
    python -m doppel_lab.neuron_atlas mod --gpu-ok [--tag] # GPU: exact per-neuron change of every adapter (resumable)
    python -m doppel_lab.neuron_atlas report               # CPU: players in neuron space -> lab/results/neuron_players.json

Tokens are the 64 squares of the board as Maia-3 sees it: mirrored when black is to move, so "us" (the side to move)
is always white. Activation = GELU(linear1(x)), the input of linear2. Concepts are computed on the same mirrored board.
Modulation: for side k, d = act(adapter k) - act(Maia-3) per (position, square, neuron), exact (adapters in earlier
adapted blocks change later blocks' inputs too), fp32. Stored per side: sum of d by probe band (5, 4, 2048) and sum
of d^2 (4, 2048); means are over probe tokens. Aggregate numbers only.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
BANDS = (700, 1100, 1500, 1900, 2300)
AT = ROOT / "data/allie/neuron_atlas"
LABEL_OUT = ROOT / "lab/results/neuron_atlas.json"
REPORT_OUT = ROOT / "lab/results/neuron_players.json"
NB, DFF = 4, 2048
PIECES = "PNBRQK"
TOK = (["empty"] + [f"our_{p}" for p in PIECES] + [f"their_{p}" for p in PIECES]
       + ["att_us", "att_them", "our_hanging", "their_hanging", "our_pinned", "legal_from", "legal_to", "capture_to",
          "our_kzone", "their_kzone", "our_passed", "their_passed",
          "top_from", "top_to", "played_from", "played_to"]       # Maia-3's top move / the move played
       + [f"rank{r + 1}" for r in range(8)] + [f"file{'abcdefgh'[f]}" for f in range(8)])
POS = ["in_check", "material", "phase", "n_legal", "ply", "castle_us", "castle_them", "n_captures", "n_checks",
       "maia_margin", "maia_entropy", "elo_self", "elo_opp"]
VAL = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}


def mod_dir(tag: str) -> Path:
    return AT / f"mod{tag}"


# ------------------------------------------------------------------ base activations
def capture_hooks(model, store: dict):
    """Forward hooks on linear1 of the adapted blocks; registered after any Lora hooks, so they see the adapted
    pre-activation. store[l] = GELU(pre) of the last forward."""
    import torch
    from doppel_lab.lora_probe import BLOCKS
    hs = []
    for l, bi in enumerate(BLOCKS):
        def f(mod, inp, out, l=l):
            store[l] = torch.nn.functional.gelu(out.float())
        hs.append(model.transformer.layers[bi].linear1.register_forward_hook(f))
    return hs


def base(device: str, workers: int):
    import torch

    from doppel_lab.adapter_space import probe_positions
    from doppel_lab.engines import Maia3
    AT.mkdir(parents=True, exist_ok=True)
    P, sel, meta = probe_positions(workers)
    model = Maia3("79m", device).model.eval().float()
    store = {}
    hs = capture_hooks(model, store)
    out = [np.zeros((len(sel), 64, DFF), np.float16) for _ in range(NB)]
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(sel), 256):
            x, es, eo, lg, _ = P.batch(sel[s:s + 256], device)
            model(x, es, eo)
            for l in range(NB):
                out[l][s:s + 256] = store[l].cpu().numpy().astype(np.float16)
    for h in hs:
        h.remove()
    for l in range(NB):
        np.save(AT / f"base_b{l}.npy", out[l])
    meta.with_columns(es=pl.Series(P.es[sel]), eo=pl.Series(P.eo[sel])).write_parquet(AT / "meta.parquet")
    print(f"base: {len(sel)} positions x 64 squares x {NB}x{DFF} neurons, {time.time() - t0:.0f}s -> {AT}", flush=True)


# ------------------------------------------------------------------ chess concepts
def _concepts(args):
    import chess
    moves, ply = args
    b = chess.Board()
    for u in moves[:ply]:
        b.push_uci(u)
    return concepts_board(b, ply)


def concepts_board(b, ply: int):
    """Token (64, len(TOK)) and position (first 9 of POS) concepts of one board, in Maia-3's mirrored frame.
    Move concepts (top_* / played_*) are left 0 here; concepts() fills them for the probe set."""
    import chess
    if b.turn == chess.BLACK:
        b = b.mirror()                                   # as Maia-3 sees it: side to move = white
    T = np.zeros((64, len(TOK)), np.float32)
    ix = {k: i for i, k in enumerate(TOK)}
    legal = list(b.legal_moves)
    ksq = {c: b.king(c) for c in (chess.WHITE, chess.BLACK)}
    for sq in chess.SQUARES:
        p = b.piece_at(sq)
        if p is None:
            T[sq, ix["empty"]] = 1
        else:
            side = "our" if p.color == chess.WHITE else "their"
            T[sq, ix[f"{side}_{PIECES[p.piece_type - 1]}"]] = 1
            att = b.is_attacked_by(not p.color, sq)
            dfd = b.is_attacked_by(p.color, sq)
            if att and not dfd and p.piece_type != chess.KING:
                T[sq, ix[f"{side}_hanging"]] = 1
            if p.color == chess.WHITE and b.is_pinned(chess.WHITE, sq):
                T[sq, ix["our_pinned"]] = 1
            if p.piece_type == chess.PAWN:
                f, r = chess.square_file(sq), chess.square_rank(sq)
                fwd = range(r + 1, 8) if p.color == chess.WHITE else range(0, r)
                opp = [b.piece_at(chess.square(ff, rr)) for ff in (f - 1, f, f + 1) if 0 <= ff < 8 for rr in fwd]
                if not any(q is not None and q.piece_type == chess.PAWN and q.color != p.color for q in opp):
                    T[sq, ix[f"{side}_passed"]] = 1
        T[sq, ix["att_us"]] = len(b.attackers(chess.WHITE, sq))
        T[sq, ix["att_them"]] = len(b.attackers(chess.BLACK, sq))
        for c, name in ((chess.WHITE, "our_kzone"), (chess.BLACK, "their_kzone")):
            if ksq[c] is not None and chess.square_distance(sq, ksq[c]) <= 1:
                T[sq, ix[name]] = 1
        T[sq, ix[f"rank{chess.square_rank(sq) + 1}"]] = 1
        T[sq, ix[f"file{'abcdefgh'[chess.square_file(sq)]}"]] = 1
    n_cap = n_chk = 0
    for m in legal:
        T[m.from_square, ix["legal_from"]] = 1
        T[m.to_square, ix["legal_to"]] = 1
        if b.is_capture(m):
            T[m.to_square, ix["capture_to"]] = 1
            n_cap += 1
        if b.gives_check(m):
            n_chk += 1
    mat = sum(VAL[p.piece_type] * (1 if p.color == chess.WHITE else -1) for p in b.piece_map().values())
    phase = sum(VAL[p.piece_type] for p in b.piece_map().values() if p.piece_type not in (chess.PAWN, chess.KING))
    pos = [b.is_check(), mat, phase, len(legal), ply, b.has_castling_rights(chess.WHITE),
           b.has_castling_rights(chess.BLACK), n_cap, n_chk]
    return T, np.array(pos, np.float32)


def concepts(workers: int):
    from doppel_lab.allie_v0e import test_games
    meta = pl.read_parquet(AT / "meta.parquet")
    mv = pl.concat([test_games(b).select("game_id", "moves") for b in BANDS]).unique("game_id")
    mv = dict(zip(mv["game_id"].to_list(), mv["moves"].to_list()))
    args = [(mv[g], int(p)) for g, p in zip(meta["game_id"].to_list(), meta["ply"].to_list())]
    with mp.get_context("fork").Pool(workers) as pool:
        res = pool.map(_concepts, args, chunksize=32)
    T = np.stack([r[0] for r in res])
    Pz = np.stack([r[1] for r in res])
    from doppel_lab.adapter_space import probe_dir                # Maia-3 margin / entropy from the probe base
    bz = np.load(probe_dir() / "base.npz")
    st = np.r_[0, np.cumsum(bz["lens"])]
    mar, ent = [], []
    for a, e in zip(st[:-1], st[1:]):
        p = np.sort(np.exp(bz["lp"][a:e]))[::-1]
        p = p / p.sum()
        mar.append(p[0] - (p[1] if len(p) > 1 else 0))
        ent.append(-(p * np.log(np.clip(p, 1e-12, None))).sum())
    assert (bz["lens"] == Pz[:, 3]).all(), "probe legal counts do not match the replayed boards"
    Pz = np.c_[Pz, mar, ent, meta["es"].to_numpy(), meta["eo"].to_numpy()].astype(np.float32)
    import chess
    from doppel_model.encoding import MOVES                         # move indices are in the mirrored frame
    from doppel_lab.adapter_space import probe_positions
    P, sel, _ = probe_positions(workers)
    ix = {k: i for i, k in enumerate(TOK)}
    for r, (i, a, e) in enumerate(zip(sel, st[:-1], st[1:])):
        lg = P.legal[i]
        for name, mi in (("top", int(lg[np.argmax(bz["lp"][a:e])])), ("played", int(P.played[i]))):
            u = chess.Move.from_uci(MOVES[mi])
            T[r, u.from_square, ix[f"{name}_from"]] = 1
            T[r, u.to_square, ix[f"{name}_to"]] = 1
    np.savez(AT / "concepts.npz", tok=T, pos=Pz, tok_names=np.array(TOK), pos_names=np.array(POS))
    print(f"concepts: tok {T.shape}, pos {Pz.shape}; legal counts match the probe -> {AT / 'concepts.npz'}")


FIRE = 0.1                                   # a neuron "fires" on a token when GELU(pre) > FIRE


def binary_concepts(T: np.ndarray, Pz: np.ndarray):
    """Binary token concepts for detector-style matching: one-hots, attack counts >= 1 / >= 2, and the binary
    position concepts broadcast to every square."""
    ix = {k: i for i, k in enumerate(TOK)}
    names, cols = [], []
    for k in TOK:
        if k in ("att_us", "att_them"):
            for t in (1, 2):
                names.append(f"{k}>={t}")
                cols.append(T[..., ix[k]] >= t)
        else:
            names.append(k)
            cols.append(T[..., ix[k]] > 0)
    for k in ("in_check", "castle_us", "castle_them"):
        names.append(k)
        cols.append(np.repeat(Pz[:, POS.index(k)][:, None] > 0, 64, 1))
    return names, np.stack(cols, -1)


def f1_table(F: np.ndarray, C: np.ndarray) -> np.ndarray:
    """F1 of 'concept true' as a detector of 'neuron fires', for every (neuron, concept): (n_neurons, n_concepts)."""
    tp = F.T.astype(np.float32) @ C                                   # (neurons, concepts)
    fp_, cp = F.sum(0)[:, None], C.sum(0)[None, :]
    return np.where(fp_ + cp > 0, 2 * tp / np.maximum(fp_ + cp, 1), 0)


def _z(X):
    X = X - X.mean(0)
    s = X.std(0)
    return X / np.where(s > 0, s, 1)


def label():
    """Per neuron: activation sparsity, square locality (share of variance explained by the square alone), best
    token concept (Pearson over all position x square tokens), best position concept (Pearson with the neuron's
    per-position mean over squares), and R^2 of a ridge fit of the activation on all concepts (token concepts plus
    position concepts broadcast to squares; fit on even positions, scored on odd)."""
    cz = np.load(AT / "concepts.npz")
    T, Pz = cz["tok"], cz["pos"]
    n = T.shape[0]
    Xt = _z(T.reshape(n * 64, -1))
    Xp = _z(Pz)
    Xall = np.c_[Xt, np.repeat(Xp, 64, 0), np.ones(n * 64)]
    tr = np.repeat(np.arange(n) % 2 == 0, 64)
    G = Xall[tr].T @ Xall[tr] + 1.0 * np.eye(Xall.shape[1])
    Bn, Bm = binary_concepts(T, Pz)
    Bm = Bm.reshape(n * 64, -1).astype(np.float32)
    pairs = [(a, b) for a in range(len(Bn)) for b in range(a + 1, len(Bn))]
    Pm = np.stack([Bm[:, a] * Bm[:, b] for a, b in pairs], 1)                # AND of two binary concepts
    pn = [f"{Bn[a]} & {Bn[b]}" for a, b in pairs]
    rep = {"blocks": "5-8 (index 4-7)", "positions": int(n), "tokens": int(n * 64), "fire_threshold": FIRE,
           "binary_concepts": Bn, "neurons": []}
    for l in range(NB):
        A = np.load(AT / f"base_b{l}.npy").astype(np.float32)            # (n, 64, 2048)
        At = A.reshape(n * 64, DFF)
        active = (At > 0.1).mean(0)
        sq_mean = A.mean(0)                                             # (64, 2048)
        tot = At.var(0)
        loc = np.where(tot > 0, sq_mean.var(0) / np.where(tot > 0, tot, 1), 0)
        Az = _z(At)
        rt = Az.T @ Xt / len(Xt)                                        # (2048, n_tok)
        Ap = _z(A.mean(1))
        rp = Ap.T @ Xp / n                                              # (2048, n_pos)
        W = np.linalg.solve(G, Xall[tr].T @ At[tr])
        res = At[~tr] - Xall[~tr] @ W
        r2 = 1 - res.var(0) / np.where(At[~tr].var(0) > 0, At[~tr].var(0), 1)
        f1s, f1p = f1_table(At > FIRE, Bm), f1_table(At > FIRE, Pm)
        for j in range(DFF):
            it, ip = int(np.abs(rt[j]).argmax()), int(np.abs(rp[j]).argmax())
            i1, i2 = int(f1s[j].argmax()), int(f1p[j].argmax())
            rep["neurons"].append({"b": l, "j": j, "active": round(float((At[:, j] > FIRE).mean()), 4),
                                   "f1_single": round(float(f1s[j, i1]), 4), "single": Bn[i1],
                                   "f1_pair": round(float(f1p[j, i2]), 4), "pair": pn[i2],
                                   "square_locality": round(float(loc[j]), 4),
                                   "tok": TOK[it], "tok_r": round(float(rt[j, it]), 4),
                                   "pos": POS[ip], "pos_r": round(float(rp[j, ip]), 4),
                                   "r2": round(float(r2[j]), 4), "std": round(float(np.sqrt(tot[j])), 4)})
        print(f"block {l}: median r2 {np.median(r2):.3f}, |best tok r| median {np.median(np.abs(rt).max(1)):.3f}, "
              f"square locality median {np.median(loc):.3f}, active median {np.median(active):.3f}", flush=True)
    N = rep["neurons"]
    r2 = np.array([x["r2"] for x in N])
    bt = np.abs([x["tok_r"] for x in N])
    f1a = np.array([x["f1_single"] for x in N])
    f1b = np.array([x["f1_pair"] for x in N])
    rep["summary"] = {"f1_single_quartiles": [round(float(q), 3) for q in np.quantile(f1a, [0.25, 0.5, 0.75])],
                      "f1_pair_quartiles": [round(float(q), 3) for q in np.quantile(f1b, [0.25, 0.5, 0.75])],
                      "share_f1_pair_gt_0.5": round(float((f1b > 0.5).mean()), 4),
                      "share_f1_pair_gt_0.7": round(float((f1b > 0.7).mean()), 4),
                      "active_quartiles": [round(float(q), 4) for q in
                                           np.quantile([x["active"] for x in N], [0.25, 0.5, 0.75])],
                      "best_pair_counts": dict(pl.Series([x["pair"] for x in N if x["f1_pair"] > 0.5])
                                               .value_counts(sort=True).head(15).iter_rows()),
                      "r2_quartiles": [round(float(q), 3) for q in np.quantile(r2, [0.25, 0.5, 0.75])],
                      "share_r2_gt_0.5": round(float((r2 > 0.5).mean()), 4),
                      "share_best_tok_r_gt_0.5": round(float((bt > 0.5).mean()), 4),
                      "share_best_tok_r_gt_0.7": round(float((bt > 0.7).mean()), 4),
                      "best_tok_counts": dict(pl.Series([x["tok"] for x in N]).value_counts(sort=True)
                                              .head(15).iter_rows())}
    LABEL_OUT.write_text(json.dumps(rep, indent=0))
    print(json.dumps(rep["summary"]), f"-> {LABEL_OUT}")


# ------------------------------------------------------------------ player modulation (GPU)
def mod(tag: str, device: str, workers: int, limit: int = 0, max_minutes: float = 0):
    import torch

    from doppel_lab.adapter_space import probe_positions
    from doppel_lab.engines import Maia3
    from doppel_lab.lora_probe import Lora
    out_dir = mod_dir(tag) / ("smoke" if limit else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    P, sel, meta = probe_positions(workers)
    band_ix = np.searchsorted(BANDS, meta["band"].to_numpy())
    model = Maia3("79m", device).model.eval().float()
    for p in model.parameters():
        p.requires_grad_(False)
    BS = 640
    batches = [P.batch(sel[s:s + BS], device)[:3] for s in range(0, len(sel), BS)]
    bix = [torch.from_numpy(band_ix[s:s + BS]).to(device) for s in range(0, len(sel), BS)]
    t0 = time.time()
    for b in BANDS:
        for f in sorted((ROOT / f"data/allie/adapters_{b}{tag}").glob("chunk_*.npz")):
            dst = out_dir / f"{b}_{f.stem.split('_')[1]}.npz"
            if dst.exists():
                continue
            if max_minutes and time.time() - t0 > 60 * max_minutes:
                print(f"stopping at the {max_minutes} min budget; rerun to resume", flush=True)
                return
            z = np.load(f)
            n = len(z["pid"])
            lora = Lora(model, n, device)
            with torch.no_grad():
                for i, p in enumerate(lora.params):
                    p.copy_(torch.from_numpy(z[f"p{i}"].astype(np.float32)).to(device))
            store = {}
            hs = capture_hooks(model, store)
            base_act = []
            with torch.no_grad():
                for x, es, eo in batches:
                    lora.cur = None
                    model(x, es, eo)
                    base_act.append([store[l].clone() for l in range(NB)])
            dsum = np.zeros((n, len(BANDS), NB, DFF), np.float32)
            dsq = np.zeros((n, NB, DFF), np.float32)
            todo = [k for k in range(n) if z["best_ep"][k] > 0][: limit or None]
            with torch.no_grad():
                for k in todo:
                    s_ = torch.zeros(len(BANDS), NB, DFF, device=device)
                    q_ = torch.zeros(NB, DFF, device=device)
                    for (x, es, eo), ba, bi in zip(batches, base_act, bix):
                        lora.cur = torch.full((x.shape[0],), k, device=device)
                        model(x, es, eo)
                        for l in range(NB):
                            d = store[l] - ba[l]                        # (B, 64, 2048)
                            s_[:, l].index_add_(0, bi, d.sum(1))
                            q_[l] += (d * d).sum((0, 1))
                    lora.cur = None
                    dsum[k], dsq[k] = s_.cpu().numpy(), q_.cpu().numpy()
            for h in hs + lora.hooks:
                h.remove()
            tmp = dst.with_suffix(".tmp")
            with open(tmp, "wb") as fh:
                np.savez(fh, dsum=dsum, dsq=dsq, pid=z["pid"], best_ep=z["best_ep"], test_game_id=z["test_game_id"])
            tmp.rename(dst)
            print(f"{b} {f.stem}: {n} sides, {time.time() - t0:.0f}s", flush=True)
            if limit:
                print("smoke done", flush=True)
                return
    print("mod done", flush=True)


def mod_load(band: int, tag: str = "_ctx200_restore") -> dict | None:
    fs = sorted(mod_dir(tag).glob(f"{band}_*.npz"))
    if not fs:
        return None
    zs = [np.load(f) for f in fs]
    cat = lambda k: np.concatenate([z[k] for z in zs])
    keep = cat("best_ep") > 0
    cnt = np.bincount(np.searchsorted(BANDS, pl.read_parquet(AT / "meta.parquet")["band"].to_numpy()),
                      minlength=len(BANDS)) * 64
    dsum = cat("dsum")[keep]
    return {"pid": cat("pid")[keep], "mean": dsum.sum(1).reshape(keep.sum(), -1) / cnt.sum(),
            "mean_by_band": dsum / cnt[None, :, None, None],
            "rms": np.sqrt(cat("dsq")[keep].reshape(keep.sum(), -1) / cnt.sum())}


def main():
    ap = argparse.ArgumentParser()
    s = ap.add_subparsers(dest="cmd", required=True)
    for name in ("base", "concepts", "label", "mod", "report"):
        p = s.add_parser(name)
        p.add_argument("--workers", type=int, default=12)
        p.add_argument("--gpu-ok", action="store_true")
        p.add_argument("--tag", default="_ctx200_restore")
        p.add_argument("--limit", type=int, default=0, help="mod smoke: first N sides of the first chunk")
        p.add_argument("--max-minutes", type=float, default=0, help="mod: stop cleanly after this many minutes")
    a = ap.parse_args()
    dev = "cuda" if a.gpu_ok else "cpu"
    if a.cmd == "base":
        base(dev, a.workers)
    elif a.cmd == "concepts":
        concepts(a.workers)
    elif a.cmd == "label":
        label()
    elif a.cmd == "mod":
        mod(a.tag, dev, a.workers, a.limit, a.max_minutes)
    elif a.cmd == "report":
        from doppel_lab.neuron_players import report
        report()


if __name__ == "__main__":
    main()

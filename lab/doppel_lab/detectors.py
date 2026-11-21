"""Executable neuron explanations: a subagent writes `detect(board, square, info) -> float` for one Maia-3 neuron,
scored against the neuron's real activations. CPU only (Maia-3 forward on CPU, Stockfish depth 12).

    python -m doppel_lab.detectors evalset --neurons b6.123,b7.42,...   # positions, activations, engine info
    python -m doppel_lab.detectors pack --neuron b6.123                   # evidence.md + detector template
    python -m doppel_lab.detectors score --neuron b6.123 --detector f.py [--split dev|test]
    python -m doppel_lab.detectors baseline --neuron b6.123               # best python-chess concept pair, same splits

Neuron names: b<block 5-8>.<index 0-2047>. Eval set: 240 Allie test games per band (none of them probe games),
10 own-move positions (ply >= 10) per game, 12,000 positions; split by game: dev (agents see it) / test (held out,
scored only by the orchestrator, threshold fixed on dev). Boards are given in real orientation; the harness maps
squares into Maia-3's mirrored frame (side to move = white) internally. Tokens = positions x 64 squares.
Metrics: Spearman(detector, activation) over tokens, AUROC of detector for "fires" (activation > FIRE), and F1 at
the threshold that maximises dev F1.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import time
import zlib
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
BANDS = (700, 1100, 1500, 1900, 2300)
DET = ROOT / "data/allie/neuron_atlas/detect"
EV = DET / "evalset"
FIRE = 0.1
GAMES_PER_BAND, POS_PER_GAME = 240, 10


def parse(name: str) -> tuple[int, int]:
    b, j = name[1:].split(".")
    return int(b) - 5, int(j)


# ------------------------------------------------------------------ eval set
def _sf(args):
    import chess
    import chess.engine
    fens = args
    eng = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")
    eng.configure({"Threads": 1, "Hash": 32})
    out = []
    for fen in fens:
        b = chess.Board(fen)
        info = eng.analyse(b, chess.engine.Limit(depth=12))
        sc = info["score"].pov(b.turn)
        pv = [m.uci() for m in info.get("pv", [])[:6]]
        out.append((sc.score(mate_score=10000), pv))
    eng.quit()
    return out


def evalset(neurons: list[str], workers: int = 24):
    import chess
    import torch

    from doppel_model.encoding import MOVES, mirror_uci
    from doppel_lab.adapter_space import PROBE_GAMES
    from doppel_lab.allie_v0e import test_games
    from doppel_lab.engines import Maia3
    from doppel_lab.lora_probe import Positions, encode_all
    from doppel_lab.neuron_atlas import capture_hooks
    EV.mkdir(parents=True, exist_ok=True)
    meta_path = EV / "meta.parquet"
    if not meta_path.exists():
        ds = []
        for b in BANDS:
            t = test_games(b).select("game_id", "moves", "white_elo", "black_elo").sort("game_id")
            probe = set(t.sample(PROBE_GAMES, seed=b)["game_id"].to_list())
            t = t.filter(~pl.col("game_id").is_in(list(probe))).filter(pl.col("moves").list.len() >= 30)
            t = t.sample(GAMES_PER_BAND, seed=b + 1)
            ds.append(pl.concat([t.with_columns(col=pl.lit(k, pl.Int64)) for k in (0, 1)]).with_columns(band=pl.lit(b)))
        d = pl.concat(ds).with_columns(pid=pl.lit(0, pl.Int64), role=pl.lit("eval"), tr_rank=pl.lit(-1, pl.Int64))
        P = Positions(d, encode_all(d, workers))
        gid = d["game_id"].to_numpy()[P.game]
        rng = np.random.default_rng(0)
        sel = []
        for g in np.unique(gid):
            cand = np.flatnonzero((gid == g) & (P.ply >= 10))
            sel.append(rng.choice(cand, min(POS_PER_GAME, len(cand)), replace=False))
        sel = np.sort(np.concatenate(sel))
        moves = d["moves"].to_list()
        fens, top, played = [], [], []
        for i in sel:
            bd = chess.Board()
            for u in moves[P.game[i]][:P.ply[i]]:
                bd.push_uci(u)
            fens.append(bd.fen())
            played.append(moves[P.game[i]][P.ply[i]])
        np.save(EV / "sel.npy", sel)
        meta = pl.DataFrame({"game_id": gid[sel], "ply": P.ply[sel], "band": d["band"].to_numpy()[P.game[sel]],
                             "fen": fens, "played": played,
                             "split": ["dev" if zlib.crc32(g.encode()) % 2 == 0 else "test" for g in gid[sel]]})
        model = Maia3("79m", "cpu").model.eval().float()
        tops, probs = [], []
        with torch.no_grad():
            for s in range(0, len(sel), 256):
                x, es, eo, lg, _ = P.batch(sel[s:s + 256], "cpu")
                o = model(x, es, eo)[0].float()
                z = o.gather(1, lg.clamp(min=0)).masked_fill(lg < 0, -1e9).softmax(1)
                pr, ix = z.topk(5, 1)
                for r in range(len(ix)):
                    white = chess.Board(fens[s + r]).turn == chess.WHITE
                    ms = [MOVES[int(lg[r, c])] for c in ix[r]]
                    tops.append([m if white else mirror_uci(m) for m in ms])
                    probs.append([round(float(p), 4) for p in pr[r]])
        meta = meta.with_columns(maia_top5=pl.Series(tops), maia_p5=pl.Series(probs))
        chunks = [fens[i::workers] for i in range(workers)]
        with mp.get_context("fork").Pool(workers) as pool:
            res = pool.map(_sf, chunks)
        cp, pv = [None] * len(fens), [None] * len(fens)
        for w, r in enumerate(res):
            for k, (c, p) in enumerate(r):
                cp[w + k * workers], pv[w + k * workers] = c, p
        meta = meta.with_columns(sf_cp=pl.Series(cp), sf_pv=pl.Series(pv))
        meta.write_parquet(meta_path)
        print(f"eval set: {meta.height} positions ({(meta['split'] == 'dev').sum()} dev) -> {meta_path}", flush=True)
        P_cache = (P, sel)
    else:
        P_cache = None
    todo = [n for n in neurons if not (EV / f"{n}.npy").exists()]
    if not todo:
        return
    if P_cache is None:                                         # rebuild Positions for the forward pass
        meta = pl.read_parquet(meta_path)
        ds = []
        for b in BANDS:
            t = test_games(b).select("game_id", "moves", "white_elo", "black_elo").sort("game_id")
            probe = set(t.sample(PROBE_GAMES, seed=b)["game_id"].to_list())
            t = t.filter(~pl.col("game_id").is_in(list(probe))).filter(pl.col("moves").list.len() >= 30)
            t = t.sample(GAMES_PER_BAND, seed=b + 1)
            ds.append(pl.concat([t.with_columns(col=pl.lit(k, pl.Int64)) for k in (0, 1)]).with_columns(band=pl.lit(b)))
        d = pl.concat(ds).with_columns(pid=pl.lit(0, pl.Int64), role=pl.lit("eval"), tr_rank=pl.lit(-1, pl.Int64))
        P_cache = (Positions(d, encode_all(d, workers)), np.load(EV / "sel.npy"))
    P, sel = P_cache
    model = Maia3("79m", "cpu").model.eval().float()
    store = {}
    hs = capture_hooks(model, store)
    want = [parse(n) for n in todo]
    acts = np.zeros((len(sel), 64, len(todo)), np.float16)
    with torch.no_grad():
        for s in range(0, len(sel), 256):
            x, es, eo, _, _ = P.batch(sel[s:s + 256], "cpu")
            model(x, es, eo)
            for k, (l, j) in enumerate(want):
                acts[s:s + 256, :, k] = store[l][:, :, j].numpy()
    for h in hs:
        h.remove()
    for k, n in enumerate(todo):
        np.save(EV / f"{n}.npy", acts[:, :, k])
    print(f"activations for {len(todo)} neurons", flush=True)


# ------------------------------------------------------------------ scoring
def _real_sq(fen: str, sq: int) -> int:
    """Maia-3 token square (mirrored frame) -> real board square."""
    import chess
    return chess.square_mirror(sq) if " b " in fen else sq


def _info(r: dict) -> dict:
    return {"maia_top5": r["maia_top5"], "maia_p5": r["maia_p5"], "played": r["played"], "sf_cp": r["sf_cp"],
            "sf_pv": r["sf_pv"], "ply": r["ply"], "band": r["band"]}


def _run(args):
    path, rows = args
    import chess
    spec = importlib.util.spec_from_file_location("det", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = np.zeros((len(rows), 64), np.float32)
    for i, r in enumerate(rows):
        b = chess.Board(r["fen"])
        inf = _info(r)
        for sq in range(64):
            out[i, sq] = float(mod.detect(b.copy(stack=False), _real_sq(r["fen"], sq), inf))
    return out


def run_detector(path: str, rows: list[dict], workers: int = 16) -> np.ndarray:
    chunks = [rows[i::workers] for i in range(workers)]
    with mp.get_context("fork").Pool(workers) as pool:
        res = pool.map(_run, [(path, c) for c in chunks])
    out = np.zeros((len(rows), 64), np.float32)
    for w, r in enumerate(res):
        out[w::workers] = r
    return out


def _auroc(s, y):
    from scipy.stats import rankdata
    r = rankdata(s)
    npos = y.sum()
    nneg = len(y) - npos
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg)) if npos and nneg else float("nan")


def _f1(s, y, thr):
    p = s > thr
    tp = (p & y).sum()
    return float(2 * tp / max(p.sum() + y.sum(), 1))


def metrics(det: np.ndarray, act: np.ndarray, thr: float | None = None) -> dict:
    from scipy.stats import spearmanr
    s, a = det.ravel(), act.ravel().astype(np.float32)
    y = a > FIRE
    if thr is None:
        qs = np.unique(np.quantile(s, np.linspace(0.5, 0.999, 120)))
        cands = np.r_[qs, s.min() - 1]
        thr = float(max(cands, key=lambda t: _f1(s, y, t)))
    rho = spearmanr(s, a).statistic if s.std() > 0 else 0.0
    ap_ = np.clip(a, 0, None)                    # GELU's negative dip carries no ordering; rank firing strength only
    rho_pos = spearmanr(s, ap_).statistic if s.std() > 0 and ap_.std() > 0 else 0.0
    return {"spearman": round(float(rho), 4), "spearman_pos": round(float(rho_pos), 4), "auroc": round(_auroc(s, y), 4), "f1": round(_f1(s, y, thr), 4),
            "threshold": thr, "fire_rate": round(float(y.mean()), 4), "det_rate": round(float((s > thr).mean()), 4)}


def score(neuron: str, detector: str, split: str = "dev", workers: int = 16) -> dict:
    meta = pl.read_parquet(EV / "meta.parquet").with_row_index("i")
    act = np.load(EV / f"{neuron}.npy")
    t0 = time.time()
    dev = meta.filter(pl.col("split") == "dev")
    ddev = run_detector(detector, dev.to_dicts(), workers)
    m = metrics(ddev, act[dev["i"].to_numpy()])
    m["seconds"] = round(time.time() - t0, 1)
    if split == "test":
        te = meta.filter(pl.col("split") == "test")
        dte = run_detector(detector, te.to_dicts(), workers)
        mt = metrics(dte, act[te["i"].to_numpy()], m["threshold"])
        return {"neuron": neuron, "dev": m, "test": mt}
    return {"neuron": neuron, "dev": m}


def baseline(neuron: str) -> dict:
    """Best AND-pair of binary python-chess concepts (neuron_atlas.binary_concepts), chosen on dev F1."""
    from doppel_lab.neuron_atlas import binary_concepts, concepts_board, f1_table
    meta = pl.read_parquet(EV / "meta.parquet").with_row_index("i")
    act = np.load(EV / f"{neuron}.npy").astype(np.float32)
    cache = EV / "concepts.npz"
    if cache.exists():
        z = np.load(cache)
        T, Pz = z["tok"], z["pos"]
    else:
        import chess
        res = [concepts_board(chess.Board(f), p) for f, p in zip(meta["fen"].to_list(), meta["ply"].to_list())]
        T = np.stack([x[0] for x in res])
        Pz = np.stack([x[1] for x in res])
        np.savez(cache, tok=T, pos=Pz)
    names, B = binary_concepts(T, Pz)
    pairs = [(a, b) for a in range(len(names)) for b in range(a + 1, len(names))]
    out = {}
    for split in ("dev", "test"):
        ix = meta.filter(pl.col("split") == split)["i"].to_numpy()
        Bm = B[ix].reshape(len(ix) * 64, -1).astype(np.float32)
        Pm = np.stack([Bm[:, a] * Bm[:, b] for a, b in pairs], 1)
        F = (act[ix].reshape(-1) > FIRE)[:, None]
        f1 = f1_table(F, np.c_[Bm, Pm])[0]
        if split == "dev":
            best = int(f1.argmax())
        allnames = names + [f"{names[a]} & {names[b]}" for a, b in pairs]
        out[split] = {"concept": allnames[best], "f1": round(float(f1[best]), 4)}
    return {"neuron": neuron, **out}


# ------------------------------------------------------------------ evidence packs
TEMPLATE = '''"""Detector for Maia-3 neuron {neuron}. Return a float that is high where the neuron fires.

board: chess.Board in real orientation (board.turn = side to move); square: a chess.Square (real coordinates);
info: maia_top5 / maia_p5 (Maia-3's top moves, UCI, and probabilities), played (UCI), sf_cp (Stockfish depth-12
score in centipawns for the side to move), sf_pv (Stockfish principal variation, UCI), ply, band (rating band).
Called once per (position, square). Keep it fast: no engine calls here; use python-chess and info only.
"""
import chess


def detect(board: chess.Board, square: chess.Square, info: dict) -> float:
    return 0.0
'''


def _ascii(fen: str, mark: int) -> str:
    import chess
    b = chess.Board(fen)
    rows = []
    for r in range(7, -1, -1):
        line = []
        for f in range(8):
            sq = chess.square(f, r)
            p = b.piece_at(sq)
            ch = p.symbol() if p else "."
            line.append(f"[{ch}]" if sq == mark else f" {ch} ")
        rows.append(f"{r + 1} " + "".join(line))
    rows.append("   " + "".join(f" {c} " for c in "abcdefgh"))
    return "\n".join(rows)


def pack(neuron: str, n_top: int = 30, n_mid: int = 15, n_zero: int = 15):
    import chess
    meta = pl.read_parquet(EV / "meta.parquet").with_row_index("i")
    act = np.load(EV / f"{neuron}.npy").astype(np.float32)
    dev = meta.filter(pl.col("split") == "dev")
    ix = dev["i"].to_numpy()
    A = act[ix]
    flat = A.ravel()
    order = np.argsort(-flat)
    fires = flat > FIRE
    rng = np.random.default_rng(0)
    pos_fire = (A > FIRE).any(1)
    mid = np.flatnonzero((flat > FIRE) & (flat < np.quantile(flat[fires], 0.5))) if fires.any() else np.array([], int)
    zero_same = np.flatnonzero(~fires & np.repeat(pos_fire, 64))       # silent squares on positions where it fires
    pick = [("top", order[:n_top]), ("weak firing (below the median firing level)",
                                     rng.choice(mid, min(n_mid, len(mid)), replace=False) if len(mid) else []),
            ("silent squares on positions where it fires elsewhere",
             rng.choice(zero_same, min(n_zero, len(zero_same)), replace=False) if len(zero_same) else [])]
    rows = dev.to_dicts()
    lab = json.loads((ROOT / "lab/results/neuron_atlas.json").read_text())["neurons"]
    l, j = parse(neuron)
    L = next(x for x in lab if x["b"] == l and x["j"] == j)
    out = [f"# Neuron {neuron}", "",
           f"Maia-3-79M, MLP block {l + 5} of 8, neuron {j}. Fires (GELU activation > {FIRE}) on "
           f"{fires.mean():.2%} of dev tokens; {pos_fire.mean():.1%} of positions have at least one firing square. "
           f"Max activation {flat.max():.3f}.",
           f"Best python-chess concept pair on the probe set: `{L['pair']}` (F1 {L['f1_pair']}). Most correlated "
           f"square concept: `{L['tok']}` (r {L['tok_r']}); position concept `{L['pos']}` (r {L['pos_r']}).",
           "", "Square = the token where the activation is measured (marked [x] on the board). Boards are real "
           "orientation; Maia-3 internally sees the board from the side to move.", ""]
    for title, sel in pick:
        out.append(f"## {title}")
        for k in sel:
            pi, sq = divmod(int(k), 64)
            r = rows[pi]
            real = _real_sq(r["fen"], sq)
            b = chess.Board(r["fen"])
            out += [f"### activation {flat[k]:.3f} on {chess.square_name(real)}  "
                    f"({'white' if b.turn else 'black'} to move, ply {r['ply']}, band {r['band']})",
                    "```", _ascii(r["fen"], real), "```",
                    f"FEN `{r['fen']}`  ",
                    f"Maia-3 top: {', '.join(f'{m} {p:.2f}' for m, p in zip(r['maia_top5'], r['maia_p5']))}; "
                    f"played {r['played']}; Stockfish {r['sf_cp']} cp, PV {' '.join(r['sf_pv'])}  ",
                    f"Other firing squares here: "
                    + ", ".join(f"{chess.square_name(_real_sq(r['fen'], s))} {A[pi, s]:.2f}"
                                for s in np.argsort(-A[pi])[:6] if A[pi, s] > FIRE and s != sq), ""]
    d = DET / neuron
    d.mkdir(parents=True, exist_ok=True)
    (d / "evidence.md").write_text("\n".join(out))
    if not (d / "detector.py").exists():
        (d / "detector.py").write_text(TEMPLATE.format(neuron=neuron))
    print(f"-> {d / 'evidence.md'} ({len(out)} lines)")


def main():
    ap = argparse.ArgumentParser()
    s = ap.add_subparsers(dest="cmd", required=True)
    e = s.add_parser("evalset")
    e.add_argument("--neurons", required=True)
    e.add_argument("--workers", type=int, default=24)
    p = s.add_parser("pack")
    p.add_argument("--neuron", required=True)
    sc = s.add_parser("score")
    sc.add_argument("--neuron", required=True)
    sc.add_argument("--detector", required=True)
    sc.add_argument("--split", default="dev", choices=("dev", "test"))
    sc.add_argument("--workers", type=int, default=16)
    bl = s.add_parser("baseline")
    bl.add_argument("--neuron", required=True)
    a = ap.parse_args()
    if a.cmd == "evalset":
        evalset(a.neurons.split(","), a.workers)
    elif a.cmd == "pack":
        pack(a.neuron)
    elif a.cmd == "score":
        print(json.dumps(score(a.neuron, a.detector, a.split, a.workers)))
    elif a.cmd == "baseline":
        print(json.dumps(baseline(a.neuron)))


if __name__ == "__main__":
    main()

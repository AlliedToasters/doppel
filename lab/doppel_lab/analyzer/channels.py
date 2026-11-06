"""Raw per-ply signals for one game, by cost tier (docs/SCIENCE.md §3).

Two tables, both plain dicts of numpy arrays:
  pos[i], i = 0..n     the position before ply i (i = n is the final position). White's point of view.
  mov[k], k = 0..n-1   ply k, from the mover's point of view.

Tier 0  clocks, think time, legal moves, check                     always
Tier 1  material                                                    always
Tier 2  Maia-3 79M: win/draw/loss, sharpness, policy entropy,       --no-maia skips; CPU, ~2 s per game
        top-move probability, played move's log-prob and rank
Tier 3  Stockfish cp at a fixed node budget, win chances, CPL,      --no-stockfish skips; ~50 ms per position
        Lichess move classes
"""

from __future__ import annotations

import math

import chess
import numpy as np

from doppel_lab.metrics import CLIP, wc
from doppel_lab.resign import VALUES

# Lichess move classes: drop in winning chances (range -1..1) on one move (lila Advice.scala).
CLASSES = (("blunder", 0.3), ("mistake", 0.2), ("inaccuracy", 0.1))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def white_material(b: chess.Board) -> int:
    return sum(v * (len(b.pieces(pt, chess.WHITE)) - len(b.pieces(pt, chess.BLACK))) for pt, v in VALUES.items())


def sharpness(win, loss):
    """jk_182's sharpness from win/draw/loss (after Lc0's WDL contempt): (2 / (logit(1-W) + logit(1-L)))^2.
    The denominator is log((1-W)(1-L) / WL) >= 0, zero only when the draw probability is zero, so the score grows
    without bound as draws vanish. Clipped at 50 for plotting."""
    w = np.clip(win, 1e-4, 1 - 1e-4)
    l_ = np.clip(loss, 1e-4, 1 - 1e-4)
    den = np.log((1 - w) * (1 - l_) / (w * l_))
    with np.errstate(divide="ignore"):
        s = (2.0 / np.maximum(den, 1e-9)) ** 2
    return np.minimum(s, 50.0)


def base_channels(rec: dict) -> tuple[dict, dict, list[chess.Board]]:
    moves, clocks = rec["moves"], rec["clocks"]
    n = len(moves)
    base = rec["base"] if rec["base"] else (max(clocks[:2]) if clocks else None)
    inc = rec["inc"] or 0.0
    b = chess.Board()
    boards = [b.copy()]
    sans, mat, legal, check = [], [white_material(b)], [b.legal_moves.count()], [b.is_check()]
    for u in moves:
        mv = b.parse_uci(u)
        sans.append(b.san(mv))
        b.push(mv)
        boards.append(b.copy())
        mat.append(white_material(b))
        legal.append(b.legal_moves.count())
        check.append(b.is_check())
    pos = {"ply": np.arange(n + 1), "material_w": np.array(mat, float), "n_legal": np.array(legal),
           "in_check": np.array(check), "fen": [x.fen() for x in boards]}
    mover = np.arange(n) % 2
    mov = {"ply": np.arange(n), "mover": mover, "uci": list(moves), "san": sans,
           "n_legal": pos["n_legal"][:n], "forced": pos["n_legal"][:n] == 1}
    # Material from the mover's view after the opponent's reply (what the mover faces at their next decision).
    sign = np.where(mover == 0, 1.0, -1.0)
    mov["material_own"] = sign * pos["material_w"][:n]
    if clocks is not None and base:
        c = np.array(clocks, float)
        before = np.array([c[k - 2] if k >= 2 else base for k in range(n)])
        opp = np.array([c[k - 1] if k >= 1 else base for k in range(n)])
        think = np.array([max(0.0, c[k - 2] - c[k] + inc) if k >= 2 else np.nan for k in range(n)])
        mov |= {"clock_after": c, "clock_before": before, "clock_opp": opp, "think_s": think,
                "think_frac": logit((think + 0.5) / (before + inc)), "clock_frac": before / base}
    rec["base_eff"] = base
    return pos, mov, boards


def maia_channels(rec: dict, pos: dict, mov: dict, boards: list[chess.Board], device: str = "cpu"):
    import torch

    from doppel_lab.engines import Maia3
    from doppel_model import MOVE_INDEX, board_planes, mirror_uci

    hist = 8
    n = len(rec["moves"])
    eng = Maia3("79m", device)
    planes = np.stack([board_planes(x) for x in boards])  # (n+1, 64, 12), each in its own mover's frame
    hidx = np.clip(np.arange(n + 1)[:, None] + np.arange(-hist + 1, 1)[None, :], 0, None)
    x = torch.from_numpy(planes[hidx]).float().permute(0, 2, 1, 3).reshape(n + 1, 64, hist * 12)
    x = torch.cat([x, torch.zeros(n + 1, 64, 1)], 2)
    white_to_move = np.array([x_.turn == chess.WHITE for x_ in boards])
    es = torch.tensor([rec["white_elo"] if w else rec["black_elo"] for w in white_to_move]).float()
    eo = torch.tensor([rec["black_elo"] if w else rec["white_elo"] for w in white_to_move]).float()
    with torch.no_grad():
        logits, value, _ = eng.model(x, es, eo)
    logits, value = logits.float().numpy(), torch.softmax(value.float(), 1).numpy()  # value: [loss, draw, win]
    win = np.where(white_to_move, value[:, 2], value[:, 0])
    loss = np.where(white_to_move, value[:, 0], value[:, 2])
    draw = value[:, 1]
    last = boards[-1]
    if last.is_checkmate():
        win[-1], draw[-1], loss[-1] = (0.0, 0.0, 1.0) if last.turn == chess.WHITE else (1.0, 0.0, 0.0)
    elif last.is_stalemate() or last.is_insufficient_material():
        win[-1], draw[-1], loss[-1] = 0.0, 1.0, 0.0
    pos |= {"maia_win_w": win, "maia_draw": draw, "maia_loss_w": loss, "maia_es_w": win + draw / 2,
            "sharpness": sharpness(win, loss)}
    ent, ptop, plp, prank = (np.full(n, np.nan) for _ in range(4))
    for k in range(n):
        b = boards[k]
        white = b.turn == chess.WHITE
        idx = np.array([MOVE_INDEX[m.uci() if white else mirror_uci(m.uci())] for m in b.legal_moves])
        lg = logits[k, idx]
        lp = lg - (lg.max() + math.log(np.exp(lg - lg.max()).sum()))
        u = rec["moves"][k]
        j = int(np.where(idx == MOVE_INDEX[u if white else mirror_uci(u)])[0][0])
        ent[k] = float(-(np.exp(lp) * lp).sum())
        ptop[k] = float(np.exp(lp.max()))
        plp[k] = float(lp[j])
        prank[k] = int((lp > lp[j]).sum()) + 1
    mov |= {"maia_entropy": ent, "maia_p_top": ptop, "maia_played_lp": plp, "maia_played_rank": prank}


def stockfish_channels(pos: dict, mov: dict, boards: list[chess.Board], nodes: int = 50_000,
                       path: str = "/usr/games/stockfish"):
    """Same settings as doppel_lab.analyze (1 thread, 16 MB hash, fixed nodes, mate = +/-10000), so a game scored
    here matches the eval sets."""
    import chess.engine

    from doppel_lab.analyze import MATE

    eng = chess.engine.SimpleEngine.popen_uci(path)
    eng.configure({"Threads": 1, "Hash": 16})
    cp = []
    try:
        for b in boards:
            if b.is_checkmate():
                cp.append(-MATE if b.turn == chess.WHITE else MATE)
            elif b.is_stalemate() or b.is_insufficient_material():
                cp.append(0)
            else:
                cp.append(eng.analyse(b, chess.engine.Limit(nodes=nodes))["score"].white().score(mate_score=MATE))
    finally:
        eng.quit()
    cp = np.array(cp, float)
    pos |= {"cp_w": cp, "wc_w": wc(cp)}
    n = len(mov["ply"])
    sign = np.where(mov["mover"] == 0, 1.0, -1.0)
    cpc = np.clip(cp, -CLIP, CLIP)
    mov["cpl"] = np.maximum(0.0, sign * (cpc[:n] - cpc[1:]))
    mov["wc_loss"] = np.maximum(0.0, sign * (pos["wc_w"][:n] - pos["wc_w"][1:]))
    cls = np.array([""] * n, dtype=object)
    for name, t in reversed(CLASSES):
        cls[mov["wc_loss"] >= t] = name
    mov["cls"] = cls

"""Implicit think-time probe: what does a head on frozen Maia-3 features learn about human think time?

Variants (each also gets the clock state: own/opp remaining, base, inc, ply, both Elos):
  A  Maia-3 79M pooled features of the current position only (history = the current board repeated)
  B  Maia-3 features with the real 8-ply history
  C  B + running tempo so far this game (own and opponent: median and decayed mean of logit think fraction, count)
Targets: y_log = log1p(think s) and y_frac = logit((think + 0.5) / (remaining_before + inc)).
Shock check: on own moves 1-5 after a realized queen loss (see lab/scripts/shock.py), the mean residual
(actual - predicted) per variant. Explicit test: humans speed up ~0.15 more than a stateless model on move 1.

    # GPU (queued through the warden): features -> data/probe/think_feats.npz
    python -m doppel_lab.think_probe extract --gpu-ok
    # CPU: ridge fits -> lab/results/think_probe_2026-10.json
    python -m doppel_lab.think_probe fit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import chess
import numpy as np
import polars as pl

from doppel_lab.split import held_out

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/probe/think_feats.npz"
VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def _lead(moves):
    b, lead, out = chess.Board(), 0, []
    for u in moves:
        m = chess.Move.from_uci(u)
        s = 1 if b.turn == chess.WHITE else -1
        if b.is_en_passant(m):
            lead += s
        elif (c := b.piece_at(m.to_square)) is not None:
            lead += s * VAL[c.piece_type]
        if m.promotion:
            lead += s * (VAL[m.promotion] - 1)
        b.push(m)
        out.append(lead)
    return np.array(out)


def _queen_shock_ply(lead_p, side, n):
    """First own ply after a realized loss of >= 9 (persists 2 own moves), else None."""
    at = lambda j: 0 if j < 0 else lead_p[j]  # noqa: E731
    for j in range(1 - side, n - 4, 2):
        b0 = at(j - 2)
        if b0 - lead_p[j] >= 9 and lead_p[j + 2] <= b0 - 9 and lead_p[j + 4] <= b0 - 9:
            return j + 1
    return None


def _logit(f):
    f = np.clip(f, 1e-3, 0.999)
    return np.log(f / (1 - f))


def build_rows(n_random=2000, n_shock=2000, seed=0):
    g = pl.concat([pl.read_parquet(ROOT / "data/human/eval_2026-10.parquet"),
                   pl.read_parquet(ROOT / "data/human/repeat_2026-10.parquet")], how="diagonal_relaxed").unique("game_id")
    g = g.filter(pl.col("clocks").list.len() == pl.col("moves").list.len()).filter(pl.col("moves").list.len() >= 30)
    rng = np.random.default_rng(seed)
    games = list(g.iter_rows(named=True))
    rng.shuffle(games)
    shocked, plain = [], []
    for r in games:
        lw = _lead(r["moves"])
        has = any(_queen_shock_ply(lw if s == 0 else -lw, s, len(r["moves"])) is not None for s in (0, 1))
        (shocked if has else plain).append(r)
    sel = plain[:n_random] + shocked[:n_shock]
    rows = []
    for r in sel:
        mv, ck, n = r["moves"], r["clocks"], len(r["moves"])
        base, inc = float(r["base"]), float(r["inc"] or 0)
        lw = _lead(mv)
        think = np.full(n, np.nan)
        for k in range(2, n):
            if ck[k] is not None and ck[k - 2] is not None:
                think[k] = max(0.0, ck[k - 2] - ck[k] + inc)
        for side in (0, 1):
            player = (r["white"] if side == 0 else r["black"]).lower()
            qs = _queen_shock_ply(lw if side == 0 else -lw, side, n)
            for k in range(side + 2, n, 2):
                if not np.isfinite(think[k]) or ck[k - 1] is None:
                    continue
                rem, opp_rem = ck[k - 2], ck[k - 1]
                own_prev = [think[j] / max(ck[j - 2] + inc, 1) for j in range(side + 2, k, 2) if np.isfinite(think[j])]
                opp_prev = [think[j] / max(ck[j - 2] + inc, 1) for j in range(3 - side, k, 2)
                            if j >= 2 and np.isfinite(think[j])]
                rows.append({
                    "game_id": r["game_id"], "ply": k, "side": side, "player": player,
                    "elo_self": r["white_elo"] if side == 0 else r["black_elo"],
                    "elo_opp": r["black_elo"] if side == 0 else r["white_elo"],
                    "rem": rem, "opp_rem": opp_rem, "base": base, "inc": inc, "think": think[k],
                    "own_tempo": own_prev, "opp_tempo": opp_prev,
                    "after_qshock": (k - qs) // 2 + 1 if qs is not None and k >= qs and (k - qs) // 2 < 5 else 0,
                })
    return rows


def _tempo(prev):
    if not prev:
        return [0.0, 0.0, 0.0]
    lg = _logit(np.array(prev))
    w = 0.7 ** np.arange(len(lg))[::-1]
    return [float(np.median(lg)), float((w * lg).sum() / w.sum()), float(np.log1p(len(lg)))]


def extract(device: str):
    import torch

    from doppel_lab.engines import Maia3
    rows = build_rows()
    eng = Maia3("79m", device)
    feats = {}
    eng.model.last_ln.register_forward_hook(lambda m, i, o: feats.__setitem__("x", o.detach()))
    by_game = {}
    for i, r in enumerate(rows):
        by_game.setdefault(r["game_id"], []).append(i)
    moves = {r["game_id"]: r for r in pl.concat([
        pl.read_parquet(ROOT / "data/human/eval_2026-10.parquet"),
        pl.read_parquet(ROOT / "data/human/repeat_2026-10.parquet")], how="diagonal_relaxed").unique("game_id")
        .filter(pl.col("game_id").is_in(list(by_game))).select("game_id", "moves").iter_rows(named=True)}
    XA, XB, E1, E2 = [], [], [], []
    for gid, idx in by_game.items():
        b, hist = chess.Board(), [Maia3.board_tokens(chess.Board())]
        tok_at = [hist[0]]
        for u in moves[gid]["moves"]:
            b.push_uci(u)
            tok_at.append(Maia3.board_tokens(b))
        for i in idx:
            k = rows[i]["ply"]  # position before ply k = tok_at[k]
            h = tok_at[max(0, k - 7):k + 1]
            h = [h[0]] * (8 - len(h)) + h
            XB.append(torch.cat(h + [torch.zeros(64, 1)], 1))
            XA.append(torch.cat([tok_at[k]] * 8 + [torch.zeros(64, 1)], 1))
            E1.append(rows[i]["elo_self"])
            E2.append(rows[i]["elo_opp"])
    out = {}
    for name, X in (("A", XA), ("B", XB)):
        embs = []
        with torch.no_grad(), torch.autocast("cuda", enabled=device.startswith("cuda")):
            for s in range(0, len(X), 512):
                eng.model(torch.stack(X[s:s + 512]).to(device), torch.tensor(E1[s:s + 512], device=device),
                          torch.tensor(E2[s:s + 512], device=device))
                embs.append(feats["x"].float().cpu().numpy().astype(np.float16))
        out[f"emb_{name}"] = np.concatenate(embs)
        print(name, out[f"emb_{name}"].shape, flush=True)
    meta = pl.DataFrame([{k: v for k, v in r.items() if k not in ("own_tempo", "opp_tempo")} for r in rows])
    tempo = np.array([_tempo(r["own_tempo"]) + _tempo(r["opp_tempo"]) for r in rows], np.float32)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, tempo=tempo, **out)
    meta.write_parquet(OUT.with_suffix(".meta.parquet"))
    print(f"wrote {OUT} ({len(rows)} rows)")


def fit():
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import StandardScaler
    d = np.load(OUT)
    m = pl.read_parquet(OUT.with_suffix(".meta.parquet"))
    rem, inc, base = m["rem"].to_numpy(), m["inc"].to_numpy(), m["base"].to_numpy()
    clock = np.c_[np.log1p(rem), np.log1p(m["opp_rem"].to_numpy()), rem / base, np.log(base), inc,
                  np.minimum(m["ply"].to_numpy(), 120) / 120, m["elo_self"].to_numpy() / 1000,
                  m["elo_opp"].to_numpy() / 1000]
    think = m["think"].to_numpy()
    ys = {"log_s": np.log1p(think), "logit_frac": _logit((think + 0.5) / (rem + inc + 1e-9))}
    test = np.array([held_out(p) for p in m["player"]])  # the project-wide player split
    shock = m["after_qshock"].to_numpy()
    variants = {
        "clock_only": clock,
        "A_position": np.c_[d["emb_A"].astype(np.float32), clock],
        "B_history": np.c_[d["emb_B"].astype(np.float32), clock],
        "C_history_tempo": np.c_[d["emb_B"].astype(np.float32), clock, d["tempo"]],
        "tempo_clock_only": np.c_[clock, d["tempo"]],
    }
    res = {"n_rows": int(len(think)), "n_test": int(test.sum()), "n_after_qshock_k1": int((shock == 1).sum())}
    for yname, y in ys.items():
        res[yname] = {}
        for vname, X in variants.items():
            sc = StandardScaler().fit(X[~test])
            mdl = RidgeCV(alphas=np.logspace(-1, 4, 11)).fit(sc.transform(X[~test]), y[~test])
            pred = mdl.predict(sc.transform(X))
            resid = y - pred
            res[yname][vname] = {
                "r2_test": round(float(r2_score(y[test], pred[test])), 4),
                "alpha": float(mdl.alpha_),
                "qshock_resid_k1": round(float(resid[shock == 1].mean()), 4),
                "qshock_resid_k1_5": round(float(resid[shock > 0].mean()), 4),
                "resid_all": round(float(resid.mean()), 4),
            }
            print(yname, vname, res[yname][vname], flush=True)
    (ROOT / "lab/results").mkdir(exist_ok=True)
    json.dump(res, open(ROOT / "lab/results/think_probe_2026-10.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "fit"])
    ap.add_argument("--gpu-ok", action="store_true", help="required to touch the GPU (run via the warden)")
    a = ap.parse_args()
    if a.cmd == "extract":
        extract("cuda" if a.gpu_ok else "cpu")
    else:
        fit()

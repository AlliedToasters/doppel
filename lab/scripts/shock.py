"""Do realized shocks change think time (and move error)? Phase 1 follow-up for the think-time head.

A blunder is not a shock: at low ratings many are never converted. Here a shock is a *realized material loss*,
oracle-free (position arithmetic, P1 N3 B3 R5 Q9):
  - the opponent's move leaves the player's material lead >= 3 below where it stood before the player's own
    previous move (so an even trade or immediate recapture does not count), and
  - the deficit persists: still >= 3 below that baseline after each of the player's next two moves (+ replies).
The event starts at the player's first move after the loss (the moment of realization).

Outcome: per own move, y = log1p(think seconds) with think = clock[k-2] - clock[k] + inc (Lichess clocks are
whole seconds), residualised on cells (band x time control x clock-fraction bin x game-phase bin) fitted within
each source. delta_k = r[t+k] - mean(r[:t]) minus the same in control player-games with no shock through t+K,
at the same t. Secondary outcome: move error log1p(CPL) from Stockfish evals.

Nulls (no internal state): ChessMimic self-play (its clock model samples think times; 180+2) for think time,
Maia-3 and ChessMimic for error. Also reports how often a Stockfish blunder is actually converted, by band.

Run from repo root (CPU): PYTHONPATH=. .venv/bin/python lab/scripts/shock.py
"""

import json
from collections import defaultdict
from pathlib import Path

import chess
import numpy as np
import polars as pl

from doppel_lab.metrics import BLUNDER, CLIP, wc

rng = np.random.default_rng(0)
K, T_MIN, SHOCK = 10, 4, 3
VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
CF_BINS = np.array([0.05, 0.1, 0.2, 0.35, 0.5, 0.75])
SIZES = {"3-4": (3, 4), "5-8": (5, 8), "9+": (9, 99)}
BANDS = (1100, 1300, 1500, 1700, 1900, 2100)


def load_human():
    g = pl.concat([pl.read_parquet("data/human/eval_2026-10.parquet"), pl.read_parquet("data/human/repeat_2026-10.parquet")],
                  how="diagonal_relaxed").unique("game_id")
    ev = pl.concat([pl.read_parquet(f) for f in ("data/evals/human.parquet", "data/evals/human_repeat.parquet")]).unique("game_id")
    return g.join(ev, on="game_id", how="left")


def load_synth(engine):
    out = []
    for b in BANDS:
        g = pl.read_parquet(f"data/synth/{engine}_{b}.parquet")
        if engine != "chessmimic":
            g = g.drop("clocks")
        out.append(g.join(pl.read_parquet(f"data/evals/{engine}_{b}.parquet"), on="game_id", how="left"))
    return pl.concat(out, how="diagonal_relaxed")


def white_lead(moves):
    """White's material lead after each ply."""
    b, lead, out = chess.Board(), 0, []
    for u in moves:
        m = chess.Move.from_uci(u)
        sign = 1 if b.turn == chess.WHITE else -1
        if b.is_en_passant(m):
            lead += sign
        else:
            cap = b.piece_at(m.to_square)
            if cap is not None:
                lead += sign * VAL[cap.piece_type]
        if m.promotion:
            lead += sign * (VAL[m.promotion] - 1)
        b.push(m)
        out.append(lead)
    return np.array(out)


def player_games(games, src):
    """One record per (game, side): own-move outcome arrays, cells, first realized shock."""
    recs, conv = [], defaultdict(lambda: [0, 0])
    for r in games.iter_rows(named=True):
        moves = r["moves"]
        n = len(moves)
        if n < 30:
            continue
        lw = white_lead(moves)
        clocks = r.get("clocks")
        ev = r.get("evals")
        ev = np.clip(np.asarray(ev, float), -CLIP, CLIP) if ev is not None and len(ev) == n + 1 else None
        base, inc = float(r["base"] or 180), float(r["inc"] or 0)
        tc = f"{int(base)}+{int(inc)}"
        for side in (0, 1):
            own = np.arange(side, n, 2)
            lead = lw if side == 0 else -lw
            lead_at = lambda j: 0 if j < 0 else lead[j]  # noqa: E731
            think = np.full(len(own), np.nan)
            cf = np.full(len(own), np.nan)
            if clocks is not None and len(clocks) == n:
                for i, k in enumerate(own):
                    if k >= 2 and clocks[k] is not None and clocks[k - 2] is not None:
                        think[i] = max(0.0, clocks[k - 2] - clocks[k] + inc)
                        cf[i] = clocks[k - 2] / base
            err = np.full(len(own), np.nan)
            if ev is not None:
                sg = 1.0 if side == 0 else -1.0
                before, after = ev[own] * sg, ev[own + 1] * sg
                err = np.log1p(np.maximum(0.0, before - after))
                # Blunder conversion: does the opponent's reply keep the loss (no blunder back)?
                for i, k in enumerate(own):
                    if k + 2 <= n and wc(before[i]) - wc(after[i]) >= BLUNDER:
                        kept = wc(before[i]) - wc(ev[k + 2] * sg) >= BLUNDER
                        c = conv[(src, r["band"])]
                        c[0] += 1
                        c[1] += int(kept)
            shock = size = None
            for j in range(1 - side, n - 4, 2):  # opponent plies
                b0 = lead_at(j - 2)
                drop = b0 - lead[j]
                if drop >= SHOCK and lead[j + 2] <= b0 - SHOCK and lead[j + 4] <= b0 - SHOCK:
                    shock, size = (j + 1 - side) // 2, int(drop)
                    break
            recs.append({"pg": f"{r['game_id']}:{side}", "band": r["band"], "tc": tc, "think": think, "cf": cf,
                         "err": err, "shock": shock, "size": size})
    return recs, {f"{k[0]}:{k[1]}": {"blunders": v[0], "converted_frac": round(v[1] / max(v[0], 1), 3)}
                  for k, v in sorted(conv.items())}


def residualise(recs, field):
    """Long table of own moves -> cell-mean residuals; returns {pg: residual array}."""
    rows = []
    for r in recs:
        y = r[field] if field == "err" else np.log1p(r["think"])
        for i, v in enumerate(y):
            if np.isfinite(v):
                cfb = int(np.digitize(r["cf"][i], CF_BINS)) if np.isfinite(r["cf"][i]) else -1
                rows.append((r["pg"], i, r["band"], r["tc"], cfb, min(i // 5, 8), float(v)))
    if not rows:
        return {}
    d = pl.DataFrame(rows, orient="row", schema=["pg", "i", "band", "tc", "cfb", "ph", "y"])
    fine = d.group_by(["band", "tc", "cfb", "ph"]).agg(pl.col("y").mean().alias("mf"), pl.len().alias("nf"))
    coarse = d.group_by(["cfb", "ph"]).agg(pl.col("y").mean().alias("mc"))
    d = d.join(fine, on=["band", "tc", "cfb", "ph"]).join(coarse, on=["cfb", "ph"])
    d = d.with_columns((pl.col("y") - pl.when(pl.col("nf") >= 30).then(pl.col("mf")).otherwise(pl.col("mc"))).alias("r"))
    out = {}
    lens = {r["pg"]: len(r[field] if field == "err" else r["think"]) for r in recs}
    for pg, g in d.group_by("pg"):
        a = np.full(lens[pg[0]], np.nan)
        a[g["i"].to_numpy()] = g["r"].to_numpy()
        out[pg[0]] = a
    return out


def drift(recs, res, size=None):
    ctrl = defaultdict(list)
    for r in recs:
        a = res.get(r["pg"])
        if a is None:
            continue
        last = len(a) - K
        for t in range(T_MIN, last + 1):
            if r["shock"] is not None and r["shock"] <= t + K:
                break
            pre = a[:t]
            if np.isfinite(pre).sum() >= 2:
                ctrl[t].append(a[t:t + K] - np.nanmean(pre))
    cm = {t: np.nanmean(np.array(v), axis=0) for t, v in ctrl.items() if len(v) >= 20}
    ev = []
    for r in recs:
        a, t = res.get(r["pg"]), r["shock"]
        if a is None or t is None or t < T_MIN or t + K > len(a) or t not in cm:
            continue
        if size is not None and not (SIZES[size][0] <= r["size"] <= SIZES[size][1]):
            continue
        pre = a[:t]
        if np.isfinite(pre).sum() < 2:
            continue
        ev.append(a[t:t + K] - np.nanmean(pre) - cm[t])
    if len(ev) < 30:
        return {"n_events": len(ev)}
    ev = np.array(ev)
    boots = np.array([np.nanmean(ev[rng.integers(0, len(ev), len(ev))], axis=0) for _ in range(300)])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    m15 = [np.nanmean(ev[rng.integers(0, len(ev), len(ev)), :5]) for _ in range(300)]
    return {"n_events": len(ev), "delta_by_k": np.round(np.nanmean(ev, axis=0), 4).tolist(),
            "ci_lo": np.round(lo, 4).tolist(), "ci_hi": np.round(hi, 4).tolist(),
            "mean_k1_5": round(float(np.nanmean(ev[:, :5])), 4),
            "mean_k1_5_ci": [round(float(x), 4) for x in np.percentile(m15, [2.5, 97.5])],
            "mean_k6_10": round(float(np.nanmean(ev[:, 5:])), 4)}


def analyse(recs, fields):
    out = {"n_player_games": len(recs), "n_shocked": sum(r["shock"] is not None for r in recs)}
    for f in fields:
        res = residualise(recs, f)
        out[f] = {"all": drift(recs, res)} | {s: drift(recs, res, s) for s in SIZES}
    return out


if __name__ == "__main__":
    out, conv = {}, {}
    H, c = player_games(load_human(), "human")
    conv |= c
    out["human"] = analyse(H, ["think", "err"])
    out["human_180+2"] = analyse([r for r in H if r["tc"] == "180+2"], ["think"])
    print("human", json.dumps(out["human"]["think"]["all"]))
    for eng, fields in (("chessmimic", ["think", "err"]), ("maia3", ["err"])):
        S, c = player_games(load_synth(eng), eng)
        conv |= c
        out[eng] = analyse(S, fields)
        print(eng, json.dumps({f: out[eng][f]["all"] for f in fields}))
    out["blunder_conversion"] = conv
    Path("lab/results").mkdir(exist_ok=True)
    json.dump(out, open("lab/results/shock_2026-10.json", "w"), indent=1)
    print("wrote lab/results/shock_2026-10.json")

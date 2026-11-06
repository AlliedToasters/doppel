"""Trajectory transforms, events and the per-game summary (docs/SCIENCE.md §3.4).

Every transform is causal: the value at a ply uses only that ply and earlier ones, so the same features can be fed
to a model that predicts the next move. Move-level channels are transformed per side, over that side's own moves.
"""

from __future__ import annotations

import chess
import numpy as np

from doppel_lab.resign import classify

SHOCK = 3            # realized shock: material loss of >= 3 that persists for the player's next two moves
TROUBLE = 0.10       # time trouble: own clock below 10% of base time
LEAD = 0.2           # lead change: winning chances (or Maia-3 expected score - 0.5, x2) beyond +/-0.2 on both sides
EWMA_ALPHA = 0.3     # matches the think-time probe's decayed tempo (weight 0.7 on the past)
ROLL = 5             # own moves in a rolling window

# Move-level channels transformed per side, when present.
SIDE_CHANNELS = ("think_s", "think_frac", "clock_frac", "maia_entropy", "maia_played_lp", "cpl", "wc_loss")


def ewma(x, alpha=EWMA_ALPHA):
    out, m = np.full(len(x), np.nan), np.nan
    for i, v in enumerate(x):
        if not np.isnan(v):
            m = v if np.isnan(m) else alpha * v + (1 - alpha) * m
        out[i] = m
    return out


def rolling(x, w, fn, min_periods=2):
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        win = x[max(0, i - w + 1):i + 1]
        win = win[~np.isnan(win)]
        if len(win) >= min_periods:
            out[i] = fn(win)
    return out


def running_median(x):
    return rolling(x, len(x), np.median, min_periods=1)


def velocity(x):
    return np.r_[np.nan, np.diff(x)]


def drawdown(x):
    """Distance below the running peak (<= 0)."""
    peak = np.fmax.accumulate(np.where(np.isnan(x), -np.inf, x))
    return np.where(np.isnan(x), np.nan, x - peak)


def since(mask):
    """Steps since the last True (NaN before the first)."""
    out, last = np.full(len(mask), np.nan), None
    for i, m in enumerate(mask):
        if m:
            last = i
        if last is not None:
            out[i] = i - last
    return out


def lichess_window(n_positions):
    """Window size of Lichess's volatility weights (lila AccuracyPercent): game length / 10, clamped to 2..8."""
    return int(min(8, max(2, n_positions // 10)))


def side_transforms(mov: dict) -> dict:
    """Per side, over own moves: EWMA, rolling mean and std, velocity; running median of think fraction (tempo)."""
    n = len(mov["ply"])
    out = {}
    for ch in SIDE_CHANNELS:
        if ch not in mov:
            continue
        for suffix in ("ewma", "roll_mean", "roll_std", "vel"):
            out[f"{ch}_{suffix}"] = np.full(n, np.nan)
        if ch == "think_frac":
            out["tempo_median"] = np.full(n, np.nan)
        for side in (0, 1):
            idx = np.where(mov["mover"] == side)[0]
            x = np.asarray(mov[ch], float)[idx]
            out[f"{ch}_ewma"][idx] = ewma(x)
            out[f"{ch}_roll_mean"][idx] = rolling(x, ROLL, np.mean)
            out[f"{ch}_roll_std"][idx] = rolling(x, ROLL, np.std)
            out[f"{ch}_vel"][idx] = velocity(x)
            if ch == "think_frac":
                # Tempo before this move: the player's pace so far, excluding the move being predicted.
                out["tempo_median"][idx] = np.r_[np.nan, running_median(x)[:-1]]
    return out


def position_transforms(pos: dict) -> dict:
    """White-POV position series: volatility of win% (Lichess window), velocity, and each side's drawdown."""
    out = {}
    w = lichess_window(len(pos["ply"]))
    for ch, scale in (("wc_w", 50.0), ("maia_es_w", 100.0)):
        if ch not in pos:
            continue
        # Win% on 0..100: Lichess uses 50 + 50 * wc; Maia-3 expected score is 0..1, so x100.
        pct = 50.0 + scale * pos[ch] if ch == "wc_w" else scale * pos[ch]
        out[f"{ch}_volatility"] = rolling(pct, w, np.std)
        out[f"{ch}_vel"] = velocity(pct)
        out[f"{ch}_drawdown_white"] = drawdown(pct)
        out[f"{ch}_drawdown_black"] = drawdown(100.0 - pct)
    out["material_vel"] = velocity(pos["material_w"])
    return out


def events(rec: dict, pos: dict, mov: dict) -> list[dict]:
    n = len(mov["ply"])
    ev = []
    if "cls" in mov:
        glyph = {"blunder": "??", "mistake": "?", "inaccuracy": "?!"}
        for k in range(n):
            if mov["cls"][k]:
                ev.append({"ply": k, "side": int(mov["mover"][k]), "kind": mov["cls"][k], "label": glyph[mov["cls"][k]]})
    # Realized shock, as in lab/scripts/shock.py: the opponent's ply j leaves the player's lead >= SHOCK below where
    # it stood before the player's previous move, and it stays that low after plies j+2 and j+4.
    lead_w = pos["material_w"][1:]  # lead after ply j
    for side in (0, 1):
        lead = lead_w if side == 0 else -lead_w
        j = 1 - side
        while j < n - 4:
            b0 = 0.0 if j - 2 < 0 else lead[j - 2]
            if b0 - lead[j] >= SHOCK and lead[j + 2] <= b0 - SHOCK and lead[j + 4] <= b0 - SHOCK:
                ev.append({"ply": j + 1, "side": side, "kind": "shock", "label": f"shock -{int(b0 - lead[j])}"})
                j += 6
            else:
                j += 2
    if "clock_frac" in mov:
        for side in (0, 1):
            idx = np.where((mov["mover"] == side) & (mov["clock_frac"] < TROUBLE))[0]
            if len(idx):
                ev.append({"ply": int(idx[0]), "side": side, "kind": "time_trouble", "label": "time trouble"})
    ch = pos.get("wc_w")
    if ch is None and "maia_es_w" in pos:
        ch = 2 * pos["maia_es_w"] - 1
    if ch is not None:
        sign = 0
        for i, v in enumerate(ch):
            s = 1 if v >= LEAD else -1 if v <= -LEAD else 0
            if s and sign and s != sign:
                ev.append({"ply": max(0, i - 1), "side": 0 if s > 0 else 1, "kind": "lead",
                           "label": "White leads" if s > 0 else "Black leads"})
            sign = s or sign
    b = chess.Board(pos["fen"][-1])
    end, resigner, _ = classify(rec["result"], rec["termination"], b)
    ev.append({"ply": n - 1, "side": -1, "kind": "end",
               "label": f"{end}" + (f" ({'White' if resigner == 'w' else 'Black'})" if resigner else "") +
                        f" {rec['result']}"})
    rec["end"] = end
    return sorted(ev, key=lambda e: e["ply"])


def summary(rec: dict, pos: dict, mov: dict, pos_tf: dict, evs: list[dict]) -> list[dict]:
    """One row per side: the per-game "dumb features"."""
    rows = []
    for side, name in ((0, "white"), (1, "black")):
        idx = mov["mover"] == side
        r = {"game_id": rec["game_id"], "side": name, "elo": rec[f"{name}_elo"],
             "opp_elo": rec["black_elo" if side == 0 else "white_elo"], "moves": int(idx.sum()),
             "end": rec.get("end"), "result": rec["result"]}

        def m(key, fn=np.nanmean, src=mov):
            return float(fn(np.asarray(src[key], float)[idx])) if key in src and idx.any() else None

        if "cpl" in mov:
            r |= {"acpl": m("cpl"), "blunders": int((mov["cls"][idx] == "blunder").sum()),
                  "mistakes": int((mov["cls"][idx] == "mistake").sum()),
                  "inaccuracies": int((mov["cls"][idx] == "inaccuracy").sum())}
        if "think_s" in mov:
            tf = np.asarray(mov["think_frac"], float)[idx]
            half = len(tf) // 2
            r |= {"think_s_median": m("think_s", np.nanmedian), "think_frac_mean": m("think_frac"),
                  "think_frac_std": m("think_frac", np.nanstd),
                  "tempo_drift": float(np.nanmean(tf[half:]) - np.nanmean(tf[:half])) if half >= 2 else None,
                  "final_clock_frac": float(np.asarray(mov["clock_frac"])[idx][-1]),
                  "moves_in_time_trouble": int((np.asarray(mov["clock_frac"])[idx] < TROUBLE).sum())}
        if "maia_entropy" in mov:
            r |= {"maia_entropy_mean": m("maia_entropy"), "maia_surprisal_mean": -(m("maia_played_lp") or 0.0),
                  "maia_top1_rate": float((np.asarray(mov["maia_played_rank"])[idx] == 1).mean())}
        for ch in ("wc_w", "maia_es_w"):
            if f"{ch}_volatility" in pos_tf:
                r[f"{ch}_volatility_mean"] = float(np.nanmean(pos_tf[f"{ch}_volatility"]))
                r[f"{ch}_worst_drawdown"] = float(np.nanmin(pos_tf[f"{ch}_drawdown_{name}"]))
        if "sharpness" in pos:
            r["sharpness_mean"] = float(np.nanmean(pos["sharpness"][:-1][idx]))
        r["shocks"] = sum(1 for e in evs if e["kind"] == "shock" and e["side"] == side)
        rows.append(r)
    return rows

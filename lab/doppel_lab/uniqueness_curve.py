"""Per-game uniqueness curves: how many other games reached this exact position, ply by ply (CPU, ~1 min).

    python -m doppel_lab.uniqueness_curve        # -> lab/results/uniqueness_curve.json

Reads posbook.py's probe (170k heldout 2026-10 games, stratified by band) and its exact per-ply position counts
against the 22.7 M-game train book at 9 nested sizes (N / 2^j). Count = train games containing the position in
their first 80 plies (R0 key, Polyglot Zobrist); heldout games are not in the book, so 0 means unique.

Questions (the user's picture: each game's count falls almost monotonically from N to ~1, the aggregate has an
expected uniqueness ply, and a bigger corpus pushes it out by about a ply per 2-10x):
  curve      quantiles of the count at each ply over games still running; share already unique
  decay      per-ply fall of the median count: count(k) ~ N b^-k implies exit ~ log N / log b and
             1 / log2 b plies per doubling; compare b with the measured exit shift
  monotone   share of ply steps where the count rises (a transposition joins), before exit
  rebound    after the first unique ply, does the game ever re-enter a known position (count >= 1, >= 10)?
             where, and with how much material left
  exit       distribution of the first unique ply at every corpus size; shift per doubling by quantile
"""

from __future__ import annotations

import json
from pathlib import Path

import chess
import numpy as np
import polars as pl

from doppel_lab.posbook import LEVELS, OUT, exit_ply, ply_index

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "lab/results/uniqueness_curve.json"
Q = (10, 25, 50, 75, 90)


def main():
    full = OUT / "full"
    st = json.loads((full / "stats.json").read_text())
    N, cap = st["games"], st["cap"]
    pc = np.load(full / "probe_counts.npy").astype(np.int64)        # (plies, LEVELS)
    lens = np.load(full / "probe_lens.npy").astype(np.int64)
    pb = pl.read_parquet(OUT / "probe.parquet", columns=["band", "moves"])
    band = pb["band"].to_numpy()
    off, gi, ply = ply_index(lens)
    c0 = pc[:, 0]
    out = {"train_games": N, "probe_games": int(len(lens)), "cap": cap}

    # curve: count quantiles by ply among games that reach the ply
    curve = []
    for k in range(cap):
        m = ply == k
        if m.sum() < 1000:
            break
        x = c0[m]
        curve.append({"ply": k, "games": int(m.sum()), "unique_share": float((x == 0).mean()),
                      **{f"q{q}": float(np.percentile(x, q)) for q in Q}})
    out["curve"] = curve

    # decay of the median count per ply, while the median is >= 2
    med = np.array([r["q50"] for r in curve])
    ok = med >= 2
    b = med[:-1][ok[1:]] / med[1:][ok[1:]]
    out["decay"] = {"median_drop_by_ply": {int(k): float(v) for k, v in zip(np.nonzero(ok[1:])[0], b)},
                    "b_geomean_plies_2_to_exit": float(np.exp(np.log(b[2:]).mean())) if len(b) > 2 else None}
    bb = out["decay"]["b_geomean_plies_2_to_exit"]
    out["decay"]["implied_plies_per_doubling"] = float(1 / np.log2(bb)) if bb else None

    # monotone: rises before the first unique ply
    e1 = exit_ply(c0, gi, ply, lens, 1)
    nxt = np.r_[c0[1:], 0]
    step = (ply < lens[gi] - 1) & (ply + 1 < e1[gi])                  # step k -> k+1, both in book
    rise = step & (nxt > c0)
    out["monotone"] = {"steps_in_book": int(step.sum()), "rise_share": float(rise[step].mean()),
                       "games_with_any_rise": float(np.bincount(gi, rise, len(lens)).astype(bool).mean()),
                       "rise_share_by_ply": {int(k): float(rise[step & (ply == k)].mean()) for k in range(1, 30)
                                             if (step & (ply == k)).sum() > 1000}}

    # rebound: back into a known position after the first unique ply
    after = ply > e1[gi]
    reb1, reb10 = after & (c0 >= 1), after & (c0 >= 10)
    g_reb = np.bincount(gi, reb1, len(lens)).astype(bool)
    g_reb10 = np.bincount(gi, reb10, len(lens)).astype(bool)
    left = (lens > e1 + 1)
    first_reb = exit_ply(np.where(reb1, 0, 1), gi, ply, lens, 1)       # first ply with a rebound (or game length)
    gap = (first_reb - e1)[g_reb]
    out["rebound"] = {"games_with_plies_after_exit": int(left.sum()),
                      "share_any_count_ge1": float(g_reb[left].mean()), "share_any_count_ge10": float(g_reb10[left].mean()),
                      "plies_after_exit_share_ge1": float(reb1[after].mean()),
                      "gap_plies_q": {q: float(np.percentile(gap, q)) for q in Q} if len(gap) else None,
                      "by_exit_ply": {}}
    for lo, hi in ((0, 10), (10, 15), (15, 20), (20, 30), (30, 80)):
        m = left & (e1 >= lo) & (e1 < hi)
        if m.sum() > 200:
            out["rebound"]["by_exit_ply"][f"{lo}-{hi - 1}"] = {"games": int(m.sum()), "share_ge1": float(g_reb[m].mean()),
                                                               "share_ge10": float(g_reb10[m].mean())}
    # what rebounds look like: replay a sample of first rebounds, count pieces
    rng = np.random.default_rng(0)
    gs = rng.choice(np.nonzero(g_reb)[0], min(3000, int(g_reb.sum())), replace=False)
    pieces, kinds = [], {"repetition_of_own_earlier": 0, "other": 0}
    for g in gs:
        k = int(first_reb[g])
        bd = chess.Board()
        seen = set()
        for u in pb["moves"][int(g)].to_list()[:k]:
            seen.add(bd._transposition_key())
            bd.push_uci(u)
        pieces.append(len(bd.piece_map()))
        kinds["repetition_of_own_earlier" if bd._transposition_key() in seen else "other"] += 1
    pieces = np.array(pieces)
    out["rebound"]["sample"] = {"n": int(len(gs)), "pieces_q": {q: float(np.percentile(pieces, q)) for q in Q},
                                "share_le10_pieces": float((pieces <= 10).mean()), **kinds,
                                "first_rebound_ply_q": {q: float(np.percentile(first_reb[gs], q)) for q in Q}}

    # exit distribution vs corpus size
    ex = []
    for j in range(LEVELS):
        e = exit_ply(pc[:, j], gi, ply, lens, 1)
        ex.append({"level": j, "games": N * 2.0 ** -j, "mean": float(e.mean()), "sd": float(e.std()),
                   **{f"q{q}": float(np.percentile(e, q)) for q in Q},
                   "by_band_median": {int(bd_): float(np.median(e[band == bd_])) for bd_ in sorted(set(band.tolist()))}})
    x = np.log2([r["games"] for r in ex])
    out["exit"] = ex
    out["exit_shift_per_doubling"] = {k: float(np.polyfit(x, [r[k] for r in ex], 1)[0])
                                      for k in ("mean", "sd", *(f"q{q}" for q in Q))}
    s = out["exit_shift_per_doubling"]["mean"]
    out["corpus_multiple_per_ply"] = float(2 ** (1 / s))
    RESULTS.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "curve"}, indent=1))
    for r in curve[:40:2]:
        print(r)


if __name__ == "__main__":
    main()

"""Does the uniqueness curve climb back late in the game? (CPU, ~3 min.)

    lab/poskey/target/release/poskey count --out data/v0/posbook/full_uncapped --threads 16 --cap 255 \
        --expected 2500000000 --probe data/v0/posbook/probe.parquet data/corpus/2026-10/train/*.parquet
    python -m doppel_lab.uniqueness_endgame          # -> lab/results/uniqueness_curve.json "endgame"

Same probe as uniqueness_curve.py (170k heldout games), but the book counts every ply up to 255 instead of the
first 80. After a game's first unique position, how often does a later position match some train game again,
by material left on the board (men incl. kings) and by corpus size (the 9 nested levels)?
"""

from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path

import chess
import numpy as np
import polars as pl

from doppel_lab.posbook import LEVELS, OUT, exit_ply, ply_index

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "lab/results/uniqueness_curve.json"
MEN = [(2, 3), (4, 6), (7, 10), (11, 16), (17, 24), (25, 32)]


def men(moves) -> list[int]:
    b, out = chess.Board(), []
    for u in moves[:255]:
        b.push_uci(u)
        out.append(chess.popcount(b.occupied))
    return out


def main():
    d = OUT / "full_uncapped"
    st = json.loads((d / "stats.json").read_text())
    N = st["games"]
    pc = np.load(d / "probe_counts.npy").astype(np.int64)
    lens = np.load(d / "probe_lens.npy").astype(np.int64)
    mv = pl.read_parquet(OUT / "probe.parquet", columns=["moves"])["moves"].to_list()
    with Pool(24) as p:
        m = np.concatenate([np.array(x[:n], np.int64) for x, n in zip(p.map(men, mv, chunksize=500), lens)])
    _, gi, ply = ply_index(lens)
    assert len(m) == len(gi) == len(pc)
    out = {"train_games": N, "cap": st["cap"], "probe_plies": int(len(gi)),
           "games_longer_than_cap": int((np.array([len(x) for x in mv]) > st["cap"]).sum())}
    levels = []
    for j in (0, 2, 4, 6, 8):
        c = pc[:, j]
        e1 = exit_ply(c, gi, ply, lens, 1)
        after = ply > e1[gi]
        known = after & (c >= 1)
        row = {"level": j, "games": N * 2.0 ** -j, "post_exit_known_share": float(known[after].mean()),
               "games_any_known_after_exit": float(np.bincount(gi, known, len(lens)).astype(bool)[lens > e1 + 1].mean()),
               "by_men": {}}
        for lo, hi in MEN:
            mm = after & (m >= lo) & (m <= hi)
            if mm.sum() < 500:
                continue
            g_any = np.bincount(gi[mm & known], minlength=len(lens)).astype(bool)
            g_has = np.bincount(gi[mm], minlength=len(lens)).astype(bool)
            row["by_men"][f"{lo}-{hi}"] = {
                "plies": int(mm.sum()), "known_share": float(known[mm].mean()),
                "known10_share": float((c[mm] >= 10).mean()), "median_count_if_known":
                    float(np.median(c[mm & known])) if (mm & known).any() else None,
                "games_reaching": int(g_has.sum()), "games_rejoin_share": float(g_any[g_has].mean())}
        levels.append(row)
    out["levels"] = levels
    # whole-game shape at the full book: last known ply, final position known
    c = pc[:, 0]
    e1 = exit_ply(c, gi, ply, lens, 1)
    last = np.full(len(lens), -1)
    k = np.nonzero(c >= 1)[0]
    np.maximum.at(last, gi[k], ply[k])
    off = np.r_[0, np.cumsum(lens)]
    fin = c[off[1:] - 1]
    fin_men = m[off[1:] - 1]
    out["full_book"] = {"exit_median": float(np.median(e1)),
                        "final_position_known_share": float((fin >= 1).mean()),
                        "final_known_by_men": {f"{lo}-{hi}": float((fin[(fin_men >= lo) & (fin_men <= hi)] >= 1).mean())
                                               for lo, hi in MEN if ((fin_men >= lo) & (fin_men <= hi)).sum() > 200},
                        "final_men_q": {q: float(np.percentile(fin_men, q)) for q in (10, 25, 50, 75, 90)},
                        "last_known_minus_exit_q": {q: float(np.percentile((last - e1)[last > e1], q)) for q in (50, 90, 99)},
                        "share_last_known_after_exit": float((last > e1).mean())}
    # examples: the most-matched positions with <= 6 men after exit
    sel = np.nonzero((ply > e1[gi]) & (m <= 6) & (c >= 1))[0]
    ex = []
    for i in sel[np.argsort(-c[sel])][:200]:
        g, k_ = int(gi[i]), int(ply[i])
        b = chess.Board()
        for u in mv[g][:k_ + 1]:
            b.push_uci(u)
        ex.append((b.board_fen(), int(c[i])))
    seen, top = set(), []
    for f, n in ex:
        if f not in seen:
            seen.add(f)
            top.append({"fen": f, "count": n})
        if len(top) == 10:
            break
    out["full_book"]["top_late_matches_le6_men"] = top
    res = json.loads(RESULTS.read_text())
    res["endgame"] = out
    RESULTS.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

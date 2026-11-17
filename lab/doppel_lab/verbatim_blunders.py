"""Verbatim blunders: on Allie test positions where the human actually blundered, does the Doppel predict that exact
move more often than Maia-3? (The rate-only question, how often a player blunders, is sim_fidelity.forecast.)

    python -m doppel_lab.verbatim_blunders [--bands 1100 1500 1900] [--workers 12]   # CPU (Stockfish 20k nodes)

Positions: data/allie/v0e_positions_v0g_<band>.parquet (v0g_b for 1500) at k = 40 context games (the headline
setting); top_t / ce_t = frozen Maia-3, top_q / ce_q = Doppel. Each scored position and the position after the
played move are evaluated with Stockfish (20k nodes, as sim_fidelity / revisit_model); wc_loss = mover's expected
score before minus after. Classes: blunder >= 0.2 (Maia's blunder2), blunder3 >= 0.3 (Lichess), mistake 0.1-0.2,
good < 0.05. Per class: top-1 hit and mean probability of the played move for both models, the paired difference,
and a game-cluster bootstrap CI. At 1500 the learned-lookup run (retr1_positions_1500: hit vs maia_hit) is added.
-> lab/results/verbatim_blunders.json
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import chess
import numpy as np
import polars as pl

from doppel_lab.revisit_model import _eval, _init, wc

ROOT = Path(__file__).resolve().parents[2]
ALLIE = ROOT / "data/allie"
c = pl.col
FILES = {1100: "v0e_positions_v0g_1100.parquet", 1500: "v0e_positions_v0g_b.parquet",
         1900: "v0e_positions_v0g_1900.parquet"}
KREQ = 40


def boot(gid: np.ndarray, f, reps: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    order = np.argsort(gid, kind="stable")
    cuts = np.flatnonzero(np.diff(gid[order])) + 1
    groups = np.split(order, cuts)
    out = [f(np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])) for _ in range(reps)]
    return [round(float(np.nanpercentile(out, 2.5)), 4), round(float(np.nanpercentile(out, 97.5)), 4)]


def label(pos: pl.DataFrame, workers: int) -> pl.DataFrame:
    games = pl.read_parquet(ALLIE / "test.parquet", columns=["game_id", "moves"]).filter(
        c("game_id").is_in(pos["game_id"].unique().implode()))
    moves = dict(zip(games["game_id"].to_list(), games["moves"].to_list()))
    want: dict[str, set] = {}
    for g, p in pos.select("game_id", "ply").iter_rows():
        want.setdefault(g, set()).update((p, p + 1))
    fen = {}
    for g, idx in want.items():
        b = chess.Board(); top = max(idx); mv = moves[g]
        for k, u in enumerate(mv[:top]):
            b.push_uci(u)
            if k + 1 in idx:
                fen[(g, k + 1)] = b.fen()
    fens = sorted(set(fen.values()))
    t0 = time.time()
    print(f"  {pos.height:,} positions, {len(fens):,} evals with {workers} workers", flush=True)
    ev = {}
    with Pool(workers, initializer=_init, initargs=(20000,)) as pool:
        for i, (f, v) in enumerate(zip(fens, pool.imap(_eval, fens, chunksize=64))):
            ev[f] = v
            if (i + 1) % 50000 == 0:
                print(f"    {i + 1:,}/{len(fens):,} evals, {time.time() - t0:.0f}s", flush=True)
    loss = []
    for g, p in pos.select("game_id", "ply").iter_rows():
        a, b = fen.get((g, p)), fen.get((g, p + 1))
        if a is None or b is None:
            loss.append(None); continue
        s = 1.0 if p % 2 == 0 else -1.0
        loss.append(float((1 + s * wc(ev[a])) / 2 - (1 + s * wc(ev[b])) / 2))
    return pos.with_columns(loss=pl.Series(loss, dtype=pl.Float64)).filter(c("loss").is_not_null())


CLASSES = {"good (<0.05)": lambda l: l < 0.05, "inaccuracy (0.05-0.1)": lambda l: (l >= 0.05) & (l < 0.1),
           "mistake (0.1-0.2)": lambda l: (l >= 0.1) & (l < 0.2), "blunder (>=0.2)": lambda l: l >= 0.2,
           "blunder3 (>=0.3)": lambda l: l >= 0.3}


def table(gid, loss, hit_a, hit_b, p_a=None, p_b=None, names=("maia", "doppel")):
    out = {}
    for name, sel in CLASSES.items():
        idx = np.flatnonzero(sel(loss))
        if len(idx) == 0:
            continue
        g = gid[idx]
        r = {"n": int(len(idx)), f"top1_{names[0]}": round(float(hit_a[idx].mean()), 4),
             f"top1_{names[1]}": round(float(hit_b[idx].mean()), 4)}
        d = (hit_b[idx].astype(float) - hit_a[idx].astype(float))
        r["top1_diff_pts"] = round(100 * float(d.mean()), 3)
        r["top1_diff_ci_pts"] = [round(100 * x, 3) for x in boot(g, lambda ii: float(d[ii].mean()))]
        r["only_" + names[1]] = int((hit_b[idx] & ~hit_a[idx]).sum())
        r["only_" + names[0]] = int((hit_a[idx] & ~hit_b[idx]).sum())
        if p_a is not None:
            r[f"p_{names[0]}"] = round(float(p_a[idx].mean()), 4)
            r[f"p_{names[1]}"] = round(float(p_b[idx].mean()), 4)
            lr = np.log(p_b[idx]) - np.log(p_a[idx])
            r["mean_log_ratio"] = round(float(lr.mean()), 4)
            r["mean_log_ratio_ci"] = boot(g, lambda ii: float(lr[ii].mean()))
        out[name] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", type=int, nargs="+", default=[1100, 1500, 1900])
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    res = {"kreq": KREQ, "bands": {}}
    for band in a.bands:
        pos = pl.read_parquet(ALLIE / FILES[band]).filter(c("kreq") == KREQ)
        print(f"band {band}", flush=True)
        d = label(pos, a.workers)
        d.select("game_id", "ply", "loss").write_parquet(ALLIE / f"verbatim_loss_{band}.parquet")
        gid = d["game_id"].cast(pl.Categorical).to_physical().to_numpy()
        loss = d["loss"].to_numpy()
        res["bands"][band] = {"positions": d.height, "blunder_rate": round(float((loss >= 0.2).mean()), 4),
                              "doppel_vs_maia": table(gid, loss, d["top_t"].to_numpy(), d["top_q"].to_numpy(),
                                                      np.exp(-d["ce_t"].to_numpy()), np.exp(-d["ce_q"].to_numpy()))}
        if band == 1500 and (ALLIE / "retr1_positions_1500.parquet").exists():
            r = pl.read_parquet(ALLIE / "retr1_positions_1500.parquet").join(d.select("game_id", "ply", "loss"),
                                                                            on=["game_id", "ply"])
            gid2 = r["game_id"].cast(pl.Categorical).to_physical().to_numpy()
            res["bands"][band]["lookup_vs_maia"] = table(gid2, r["loss"].to_numpy(), r["maia_hit"].to_numpy(),
                                                         r["hit"].to_numpy(), names=("maia", "lookup"))
            res["bands"][band]["lookup_positions"] = r.height
    res["note"] = ("Allie test protocol positions; loss = mover's expected-score drop on the played move (Stockfish "
                   "20k nodes). top1 = model's top move equals the played move; p = model probability of the played "
                   "move. Diffs are Doppel (or lookup) minus Maia-3; CIs: game-cluster bootstrap, 1000 reps.")
    out = ROOT / "lab/results/verbatim_blunders.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()

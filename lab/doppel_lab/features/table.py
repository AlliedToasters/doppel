"""Corpus-scale feature tables: one row per ply, every registered "mov" feature up to a tier (docs/SCIENCE.md §3).

Built from stored data only, with the same functions the analyzer uses:
  T0  data/v0/games_<band>_<split>.parquet (moves, clocks, ratings)          all games
  T1  data/v0/teacher_<band>_<split>/shard-*.npz (Maia-3 top-16 per ply)      all games
  T2  data/v0/evals_<band>_<split>.parquet (Stockfish 50k nodes)              only games with evals
Maia-3 win/draw/loss is not stored for v0 positions, so the T1 position features (maia_es_w volatility etc.) are
analyzer-only for now.

Output: data/v0/features/moves_<VERSION>_<tier>_<band>_<split>/part-XXXX.parquet (one part per teacher shard,
float32, resumable). Read it with load().

    python -m doppel_lab.features.table --split val --tier T2 --workers 8
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import polars as pl

from doppel_lab.blunder import THRESH, THRESH_MAIA
from doppel_lab.features import channels, registry, transforms

ROOT = Path(__file__).resolve().parents[3]
V0 = ROOT / "data/v0"
OUT = V0 / "features"
KEEP_META = ("game_id", "mover", "uci")


def out_dir(split: str, tier: str, band: int = 1500) -> Path:
    return OUT / f"moves_{registry.VERSION}_{tier}_{band}_{split}"


def game_rows(task) -> dict:
    """All mov features for one game, as equal-length numpy columns."""
    row, t1, evals = task
    rec = {"moves": row["moves"], "clocks": row["clocks"], "base": row["base"], "inc": row["inc"]}
    if rec["clocks"] is not None and any(c is None for c in rec["clocks"]):
        rec["clocks"] = None
    pos, mov, _ = channels.base_channels(rec)
    if t1 is not None:
        channels.teacher_moves(mov, *t1)
    if evals is not None:
        channels.eval_channels(pos, mov, evals)
        mov["blunder"] = mov["wc_loss"] >= THRESH
        mov["blunder2"] = mov["wc_loss"] >= THRESH_MAIA
    n = len(mov["ply"])
    white = mov["mover"] == 0
    mov["elo_self"] = np.where(white, row["white_elo"], row["black_elo"])
    mov["elo_opp"] = np.where(white, row["black_elo"], row["white_elo"])
    mov["base"] = np.full(n, row["base"] or np.nan)
    mov["inc"] = np.full(n, row["inc"] or 0)
    feats = mov | transforms.side_transforms(mov) | transforms.position_to_moves(transforms.position_transforms(pos), mov)
    out = {"game_id": np.full(n, row["game_id"]), "mover": mov["mover"].astype(np.int8),
           "uci": np.array(mov["uci"])}
    for k, v in feats.items():
        if (k, "mov") in registry.REGISTRY:
            a = np.asarray(v)
            out[k] = a.astype(np.float32) if a.dtype.kind == "f" else a
    return out


def _concat(parts: list[dict]) -> pl.DataFrame:
    cols = sorted(set().union(*parts), key=lambda c: (c not in KEEP_META, c))
    n = [len(p["game_id"]) for p in parts]
    data = {}
    for c in cols:
        chunks = [p[c] if c in p else np.full(k, np.nan, np.float32) for p, k in zip(parts, n)]
        data[c] = np.concatenate(chunks)
    return pl.DataFrame(data)


def build(split: str, tier: str = "T2", band: int = 1500, workers: int = 8, limit: int | None = None,
          evals: str = "", name: str = ""):
    """evals: an alternate evals glob (e.g. the eval net's data/evals/v0_train_evalnet_ft.part*.parquet);
    name: the output directory name under data/v0/features (default moves_<VERSION>_<tier>_<band>_<split>)."""
    games = pl.read_parquet(V0 / f"games_{band}_{split}.parquet",
                            columns=["game_id", "white_elo", "black_elo", "base", "inc", "moves", "clocks"])
    ev = None
    if registry.TIERS.index(tier) >= 2:
        src = sorted(ROOT.glob(evals)) if evals else [V0 / f"evals_{band}_{split}.parquet"]
        ev = dict(pl.concat([pl.read_parquet(f) for f in src]).iter_rows())
    shards = sorted((V0 / f"teacher_{band}_{split}").glob("shard-*.npz"))
    use_t1 = registry.TIERS.index(tier) >= 1
    d = OUT / name if name else out_dir(split, tier, band)
    d.mkdir(parents=True, exist_ok=True)
    by_id = {gid: i for i, gid in enumerate(games["game_id"].to_list())}
    rows_total = 0
    with Pool(workers) as pool:
        for s in shards:
            part = d / f"part-{s.stem.split('-')[1]}.parquet"
            if part.exists():
                continue
            z = np.load(s)
            gids, off = z["game_ids"], z["offsets"]
            arrays = {k: z[k] for k in ("top_lp", "played_idx", "top_idx", "played_lp", "entropy")} if use_t1 else None
            tasks = []
            for j, gid in enumerate(gids[:limit] if limit else gids):
                if ev is not None and gid not in ev:
                    continue
                row = games.row(by_id[gid], named=True)
                a, b = off[j], off[j + 1]
                assert b - a == len(row["moves"]), gid
                t1 = tuple(arrays[k][a:b] for k in ("top_lp", "played_idx", "top_idx", "played_lp", "entropy")) \
                    if use_t1 else None
                tasks.append((row, t1, ev[gid] if ev is not None else None))
            if not tasks:
                continue
            df = _concat(pool.map(game_rows, tasks, chunksize=64))
            df.write_parquet(part)
            rows_total += df.height
            print(f"{part.name}: {len(tasks):,} games, {df.height:,} moves, {df.width} columns", flush=True)
    print(f"{split} {tier}: {rows_total:,} new rows -> {d}")
    return d


def load(split: str, tier: str = "T2", band: int = 1500, columns: list[str] | None = None) -> pl.DataFrame:
    # base is Int64 in some parts and Float64 in others (game_rows writes NaN for a 0-second base)
    q = pl.scan_parquet(out_dir(split, tier, band) / "part-*.parquet",
                        cast_options=pl.ScanCastOptions(integer_cast="allow-float"))
    return (q.select(columns) if columns else q).collect()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val"], default="val")
    ap.add_argument("--tier", choices=registry.TIERS[:3], default="T2")
    ap.add_argument("--band", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="games per shard (smoke test)")
    ap.add_argument("--evals", default="", help="alternate evals glob, relative to the repo")
    ap.add_argument("--name", default="", help="output directory name under data/v0/features")
    a = ap.parse_args()
    build(a.split, a.tier, a.band, a.workers, a.limit, a.evals, a.name)

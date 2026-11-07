"""Position frequency ("book") from the Lichess corpus: how common is the line a game is on, ply by ply?

Key: a 64-bit hash of the move sequence so far (h_k = mix(h_{k-1} ^ move_k)), so two games share a key at ply k
iff they played the same first k moves. Transpositions (the same position by another move order) count apart;
`zobrist` measures how much merging them changes depth on a subsample. No board is replayed, so the full corpus
(22.7M train games, plies < CAP) hashes in numpy in minutes.

Counts come from data/corpus/2026-10/train. Every train game draws a level j ~ Geometric(1/2) (capped at LEVELS-1)
and belongs to the 2^-j subsample iff its level >= j, so one pass gives counts for N, N/2, ..., N/2^(LEVELS-1)
(the data-scaling curve). Lookups for evaluation games (heldout split and the human corpus) never see themselves:
heldout is disjoint from train, and human games found in train have their own contribution removed.

The lookup table itself (sorted uint64 keys + uint32 counts, count >= 2) is what a live client would ship:
np.searchsorted answers a batch of lookups in microseconds.

    python -m doppel_lab.book hash        # train keys -> data/v0/book/ (memmaps, ~12 GB, deleted after count)
    python -m doppel_lab.book count       # counts for eval games at every level + lookup table
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data/corpus/2026-10"
OUT = ROOT / "data/v0/book"
CAP = 80          # plies hashed per game
LEVELS = 9        # subsamples N, N/2, ..., N/256
BUCKETS = 8       # passes over the key memmap (memory bound)
SEED = 0
FILES = {c: i for i, c in enumerate("abcdefgh")}
PROMO = {"": 0, "q": 1, "r": 2, "b": 3, "n": 4}


def move_ids(moves: pl.Series) -> tuple[np.ndarray, np.ndarray]:
    """List[str] UCI per game -> (flat int32 ids, offsets). id = from*64 + to + 4096*promo."""
    lens = moves.list.len().fill_null(0).to_numpy()
    s = moves.explode().drop_nulls()
    sq = lambda f, r: (s.str.slice(f, 1).replace_strict(FILES, return_dtype=pl.Int32)
                       + 8 * (s.str.slice(r, 1).cast(pl.Int32) - 1))
    ids = (sq(0, 1) * 64 + sq(2, 3) + 4096 * s.str.slice(4, 1).replace_strict(PROMO, return_dtype=pl.Int32))
    off = np.zeros(len(lens) + 1, np.int64)
    np.cumsum(lens, out=off[1:])
    return ids.to_numpy().astype(np.int32), off


def _mix(x: np.ndarray) -> np.ndarray:
    """splitmix64 finaliser (uint64 arithmetic wraps)."""
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def prefix_hashes(ids: np.ndarray, off: np.ndarray, cap: int = CAP) -> tuple[np.ndarray, np.ndarray]:
    """Per game, the key after each of its first min(len, cap) plies. Returns (flat keys, per-game counts)."""
    lens = np.minimum(np.diff(off), cap)
    n = len(lens)
    out_off = np.zeros(n + 1, np.int64)
    np.cumsum(lens, out=out_off[1:])
    keys = np.empty(out_off[-1], np.uint64)
    h = np.full(n, 0x9E3779B97F4A7C15, np.uint64)
    with np.errstate(over="ignore"):
        for k in range(cap):
            g = np.nonzero(lens > k)[0]
            if not len(g):
                break
            h[g] = _mix(h[g] ^ (ids[off[g] + k].astype(np.uint64) + np.uint64(1)))
            keys[out_off[g] + k] = h[g]
    return keys, lens


def bucket_of(keys: np.ndarray) -> np.ndarray:
    return (keys >> np.uint64(61)).astype(np.int8)  # top 3 bits -> 8 buckets


# --- hash the train corpus ---------------------------------------------------------------------------------------

def hash_train():
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(CORPUS / "train/*.parquet")))
    rng = np.random.default_rng(SEED)
    t = time.time()
    kf, lf = open(OUT / "train_keys.u64", "wb"), open(OUT / "train_level.u8", "wb")
    ids_file = []
    n_games = n_keys = 0
    for i, f in enumerate(files):
        d = pl.read_parquet(f, columns=["game_id", "moves"])
        ids, off = move_ids(d["moves"])
        keys, lens = prefix_hashes(ids, off)
        level = np.minimum(rng.geometric(0.5, len(lens)) - 1, LEVELS - 1).astype(np.uint8)
        kf.write(keys.tobytes())
        lf.write(np.repeat(level, lens).tobytes())
        ids_file.append(d["game_id"])
        n_games += len(lens)
        n_keys += len(keys)
        print(f"{i + 1}/{len(files)} {n_games:,} games {n_keys:,} keys {time.time() - t:.0f}s", flush=True)
    kf.close(), lf.close()
    pl.DataFrame({"game_id": pl.concat(ids_file)}).write_parquet(OUT / "train_game_ids.parquet")
    (OUT / "train_meta.json").write_text(json.dumps({"games": n_games, "keys": n_keys, "cap": CAP,
                                                     "levels": LEVELS}) + "\n")


# --- evaluation games ----------------------------------------------------------------------------------------------

def eval_games() -> pl.DataFrame:
    """Heldout: up to 10k per band (bands 800..2400), every game with a rating gap >= 300; plus the human corpus."""
    cols = ["game_id", "white_elo", "black_elo", "band", "base", "inc", "result", "termination", "moves", "clocks"]
    ho = pl.scan_parquet(sorted(glob.glob(str(CORPUS / "heldout/*.parquet")))).select(cols)
    gap = (pl.col("white_elo") - pl.col("black_elo")).abs()
    strat = (ho.filter(pl.col("band").is_between(800, 2400)).with_columns(r=pl.col("game_id").hash(SEED))
             .filter(pl.col("r").rank("ordinal").over("band") <= 10_000).drop("r").collect())
    mism = ho.filter(gap >= 300).collect()
    hum = pl.read_parquet(ROOT / "data/human/games_2026-10.parquet").select(cols)
    out = pl.concat([strat.with_columns(src=pl.lit("heldout")), mism.with_columns(src=pl.lit("mismatch")),
                     hum.with_columns(src=pl.lit("human"))])
    return out.unique("game_id", keep="first", maintain_order=True)


def count():
    meta = json.loads((OUT / "train_meta.json").read_text())
    keys = np.memmap(OUT / "train_keys.u64", np.uint64, "r")
    level = np.memmap(OUT / "train_level.u8", np.uint8, "r")
    ev = eval_games()
    print(f"eval games: {ev.group_by('src').len().sort('src').rows()}", flush=True)
    ids, off = move_ids(ev["moves"])
    ekeys, elens = prefix_hashes(ids, off)
    # human games that are also train games: remove their own contribution (their keys, at their level)
    own = ev.select("game_id").with_row_index("i").join(pl.read_parquet(OUT / "train_game_ids.parquet")
                                                        .with_row_index("t"), on="game_id")
    print(f"eval games also in train: {own.height}", flush=True)
    eb = bucket_of(ekeys)
    counts = np.zeros((len(ekeys), LEVELS), np.uint32)
    table_k, table_c = [], []
    t = time.time()
    step = 200_000_000
    for b in range(BUCKETS):
        parts_k, parts_l = [], []
        for s in range(0, len(keys), step):
            k = np.asarray(keys[s:s + step])
            m = bucket_of(k) == b
            parts_k.append(k[m])
            parts_l.append(np.asarray(level[s:s + step])[m])
        k, lv = np.concatenate(parts_k), np.concatenate(parts_l)
        sel = np.nonzero(eb == b)[0]
        for j in range(LEVELS):
            kk = k if j == 0 else k[lv >= j]
            u, cnt = np.unique(kk, return_counts=True)
            pos = np.searchsorted(u, ekeys[sel]).clip(max=len(u) - 1)
            hit = u[pos] == ekeys[sel]
            counts[sel[hit], j] = cnt[pos[hit]]
            if j == 0:
                keep = cnt >= 2
                table_k.append(u[keep])
                table_c.append(cnt[keep].astype(np.uint32))
                n_distinct = len(u)
        print(f"bucket {b + 1}/{BUCKETS}: {len(k):,} keys, {n_distinct:,} distinct ({time.time() - t:.0f}s)", flush=True)
        del k, lv
    # self-exclusion for eval games that are in train: subtract 1 at levels the game belongs to
    if own.height:
        tl = np.zeros(meta["games"], np.uint8)  # per-train-game level, rebuilt from the per-key array
        tlen = None
        print("self-exclusion: recomputing train levels", flush=True)
        rng = np.random.default_rng(SEED)
        files = sorted(glob.glob(str(CORPUS / "train/*.parquet")))
        lv_all = []
        for f in files:
            n = pl.scan_parquet(f).select(pl.len()).collect().item()
            lv_all.append(np.minimum(rng.geometric(0.5, n) - 1, LEVELS - 1).astype(np.uint8))
        tl = np.concatenate(lv_all)
        eoff = np.zeros(len(elens) + 1, np.int64)
        np.cumsum(elens, out=eoff[1:])
        for i, ti in own.select("i", "t").iter_rows():
            a, z = eoff[i], eoff[i + 1]
            for j in range(int(tl[ti]) + 1):
                counts[a:z, j] -= 1
    np.save(OUT / "eval_counts.npy", counts)
    np.save(OUT / "eval_lens.npy", elens)
    ev.drop("moves", "clocks").write_parquet(OUT / "eval_games.parquet")
    ev.select("game_id", "moves", "clocks").write_parquet(OUT / "eval_moves.parquet")
    tk = np.concatenate(table_k)
    order = np.argsort(tk)
    np.save(OUT / "table_keys.npy", tk[order])
    np.save(OUT / "table_counts.npy", np.concatenate(table_c)[order])
    print(f"table: {len(tk):,} keys with count >= 2 ({len(tk) * 12 / 1e9:.1f} GB); done {time.time() - t:.0f}s")


# --- transpositions: move-order keys vs position keys on a subsample ------------------------------------------------

ZCAP = 40


def _zob(moves):
    import chess
    import chess.polyglot
    b = chess.Board()
    out = []
    for u in moves[:ZCAP]:
        b.push_uci(u)
        out.append(chess.polyglot.zobrist_hash(b))
    return out


def zobrist(n_train: int = 1_000_000, n_eval: int = 20_000, workers: int = 6):
    """Book exit (T = 1, 10) on n_eval heldout games against n_train train games, by both keys."""
    from multiprocessing import Pool
    files = sorted(glob.glob(str(CORPUS / "train/*.parquet")))
    tr = pl.concat([pl.read_parquet(f, columns=["moves"]) for f in files[: max(1, n_train // 500_000)]]).head(n_train)
    ev = pl.read_parquet(sorted(glob.glob(str(CORPUS / "heldout/*.parquet")))[0], columns=["moves"]).head(n_eval)
    t = time.time()
    res = {"train_games": tr.height, "eval_games": ev.height, "cap": ZCAP}
    with Pool(workers) as p:
        zt = p.map(_zob, tr["moves"].to_list(), chunksize=500)
        ze = p.map(_zob, ev["moves"].to_list(), chunksize=500)
    print(f"zobrist hashed ({time.time() - t:.0f}s)", flush=True)
    for name in ("prefix", "position"):
        if name == "prefix":
            ids, off = move_ids(tr["moves"])
            kt, _ = prefix_hashes(ids, off, ZCAP)
            ids, off = move_ids(ev["moves"])
            ke, le = prefix_hashes(ids, off, ZCAP)
        else:
            kt = np.array([h for z in zt for h in z], np.uint64)
            ke = np.array([h for z in ze for h in z], np.uint64)
            le = np.array([len(z) for z in ze])
        u, cnt = np.unique(kt, return_counts=True)
        pos = np.searchsorted(u, ke).clip(max=len(u) - 1)
        ce = np.where(u[pos] == ke, cnt[pos], 0)
        gi = np.repeat(np.arange(len(le)), le)
        ply = np.arange(len(gi)) - np.r_[0, np.cumsum(le)][:-1][gi]
        row = {"distinct_train_keys": int(len(u))}
        for T in (1, 10):
            ex = le.copy()
            below = np.nonzero(ce < T)[0]
            fg, fi = np.unique(gi[below], return_index=True)
            ex[fg] = ply[below[fi]]
            row[f"T{T}_mean_exit"] = float(ex.mean())
            row[f"T{T}_median_exit"] = float(np.median(ex))
        row["mean_log10_count_by_ply"] = [float(np.log10(ce[ply == k] + 1).mean()) for k in range(ZCAP)]
        res[name] = row
        print(name, {k: v for k, v in row.items() if k != "mean_log10_count_by_ply"}, flush=True)
    (ROOT / "lab/results").mkdir(exist_ok=True)
    (ROOT / "lab/results/book_transpositions.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["hash", "count", "zobrist"])
    a = ap.parse_args()
    {"hash": hash_train, "count": count, "zobrist": zobrist}[a.cmd]()

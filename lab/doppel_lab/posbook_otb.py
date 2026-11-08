"""Online vs over the board: the Lichess position book (R0) against Lumbra's Gigabase OTB. RESEARCH ONLY. (CPU)

Lumbra's Gigabase OTB is CC BY-NC 4.0 (data/raw/lumbras/README.local.md): research use only, nothing from it is
shipped, and every table built from it stays under data/raw/lumbras/. Only aggregate statistics leave this module.

    zstd -dc data/raw/lumbras/LumbrasGigaBase_OTB.pgn.zst | lab/poskey/target/release/poskey pgn \
        --out data/raw/lumbras/moves_cap80 --cap 80          # 13.25 M games, ~2 min
    python -m doppel_lab.posbook_otb prep       # 2% OTB probe + decade book files + probe_both.parquet
    python -m doppel_lab.posbook_otb count      # poskey count runs (OTB all / bands / decades, Lichess all / bands)
    python -m doppel_lab.posbook_otb analyze    # -> lab/results/posbook_explore.json, section "online_vs_otb"

Probe games: the Lichess heldout eval games (posbook.py, 10k per band) and a 2% hash sample of OTB games (not in
the OTB book). Every count run probes both, so each game gets exact counts in each book at 9 nested subsample
levels (N, N/2, ..., N/256), which puts books of different sizes at equal N by interpolating depth on log2 N.
Book exit = first ply whose position has count 0 (posbook.py's T1 rule). Ratings: OTB `band` is the mean FIDE
Elo of the two players floored to 100 (both present), Lichess `band` the mean Lichess blitz rating; the two
scales are not the same.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

import numpy as np
import polars as pl

from doppel_lab.posbook_common import (BIN, CORPUS, DATA, R0_DIR, V0, load_table, lookup, opening_keys,
                                       ply_index, save_section)

c = pl.col
LUM = DATA / "raw/lumbras"
SRC = LUM / "moves_cap80"
PB = LUM / "posbook"                       # every derived table stays under a path containing "lumbras"
DECADES = (1970, 1980, 1990, 2000, 2010, 2020)
BANDS = (1400, 1600, 1800, 2000, 2200, 2400, 2600)
LEVELS = 9
THREADS = 16
DEDUP_PLIES = 40


def prep():
    """Stream the converted parts: dedupe, then decade book files (98%) and a 2% hash-sampled OTB probe.

    Lumbra's merges sources, and the same game recurs under different headers: 11% of games of 60+ plies share
    their ply-60 position with another game (22-32% before 2000; Lichess: 0.01%). A game of >= DEDUP_PLIES plies
    whose first 80 plies equal another's is dropped (the copy with an Elo, else the first, is kept). Shorter
    identical games (quick draws, miniatures) are genuine repeats and are kept."""
    import pyarrow.parquet as pq
    (PB / "book").mkdir(parents=True, exist_ok=True)
    files = sorted(SRC.glob("part-*.parquet"))
    hs = []
    for fi, f in enumerate(files):
        h = pl.read_parquet(f, columns=["band", "moves"]).select(
            f=pl.lit(fi, pl.Int32), row=pl.int_range(pl.len(), dtype=pl.Int64), h=c("moves").list.join(" ").hash(11),
            long=c("moves").list.len() >= DEDUP_PLIES, has_band=c("band").is_not_null())
        hs.append(h)
    hs = pl.concat(hs)
    first = (hs.filter(c("long")).sort(["h", "has_band", "f", "row"], descending=[False, True, False, False])
             .unique("h", keep="first", maintain_order=True).select("f", "row", keep=pl.lit(True)))
    hs = hs.join(first, on=["f", "row"], how="left").with_columns(keep=c("keep").fill_null(~c("long")))
    n_dup = int((~hs["keep"]).sum())
    print(f"dedup: {hs.height:,} games, {n_dup:,} duplicates of >= {DEDUP_PLIES} plies dropped", flush=True)
    writers, probes, n_all = {}, [], 0
    for fi, f in enumerate(files):
        g = pl.read_parquet(f, columns=["game_id", "band", "year", "eco", "moves"])
        g = g.filter(hs.filter(c("f") == fi).sort("row")["keep"])
        n_all += g.height
        g = g.with_columns(probe=(c("game_id").hash(7) % 50) == 0,
                           dec=pl.when(c("year") < 1970).then(1960).otherwise(c("year") // 10 * 10).fill_null(0))
        probes.append(g.filter(c("probe")).select(src=pl.lit("otb"), game_id="game_id", band="band", year="year",
                                                  eco="eco", moves="moves"))
        for (d,), part in g.filter(~c("probe")).group_by("dec"):
            t = part.select("game_id", "band", "year", "moves").to_arrow()
            if d not in writers:
                writers[d] = pq.ParquetWriter(PB / f"book/dec{d}.parquet", t.schema, compression="zstd")
            writers[d].write_table(t)
        print(f.name, n_all, flush=True)
        del g
    for w in writers.values():
        w.close()
    otb = pl.concat(probes)
    li = pl.read_parquet(V0 / "posbook/probe.parquet", columns=["game_id", "band", "moves"])
    li = li.select(src=pl.lit("lichess"), game_id="game_id", band="band",
                   year=pl.lit(2026, pl.Int64), eco=pl.lit(None, pl.String), moves=c("moves").list.head(80))
    both = pl.concat([li, otb])
    both.write_parquet(PB / "probe_both.parquet")
    print(f"OTB: {n_all:,} games after dedup, probe {otb.height:,}, book {n_all - otb.height:,}; "
          f"probe_both {both.height:,}", flush=True)
    (PB / "prep.json").write_text(json.dumps({"games_converted": hs.height, "duplicates_dropped": n_dup,
                                              "dedup_min_plies": DEDUP_PLIES, "games_kept": n_all,
                                              "probe_games": otb.height}) + "\n")


def _run(name: str, files, extra=()):
    out = PB / "runs" / name
    if (out / "stats.json").exists():
        print(f"{name}: done", flush=True)
        return
    cmd = [str(BIN), "count", "--out", str(out), "--threads", str(THREADS), "--fpr", "0.001",
           "--probe", str(PB / "probe_both.parquet"), *extra, *map(str, files)]
    print(name, len(files), "files", " ".join(extra), flush=True)
    r = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
    if r.returncode:
        sys.stderr.write(r.stderr)
        r.check_returncode()
    s = json.loads((out / "stats.json").read_text())
    print(f"  {s['games']:,} games, {s['seconds_total']:.0f}s, {s['peak_rss_gb']:.1f} GB", flush=True)


def count():
    otb_files = sorted((PB / "book").glob("dec*.parquet"))
    li_files = sorted((CORPUS / "train").glob("*.parquet"))
    _run("otb_all", otb_files)
    _run("lichess_all", li_files)
    for d in DECADES:
        _run(f"otb_dec{d}", [PB / f"book/dec{d}.parquet"])
    for b in BANDS:
        rng = ["--band-min", str(b), "--band-max", str(b + 199)]
        _run(f"otb_b{b}", otb_files, rng)
        _run(f"lichess_b{b}", li_files, rng)
    # the Lichess tables duplicate data/v0/posbook; only their probe counts are needed here
    for p in (PB / "runs").glob("lichess_*/table_*.npy"):
        p.unlink()


# --- analysis -------------------------------------------------------------------------------------------------------

def _load(name: str):
    d = PB / "runs" / name
    s = json.loads((d / "stats.json").read_text())
    return np.load(d / "probe_counts.npy").astype(np.int64), s["games"]


_EXIT_CACHE: dict = {}


def _exit(cnt_col: np.ndarray, gi: np.ndarray, ply: np.ndarray, lens: np.ndarray) -> np.ndarray:
    """First ply whose position has count 0, per game (games that never leave book: their length)."""
    key = (id(cnt_col), cnt_col.ctypes.data)
    if key in _EXIT_CACHE:
        return _EXIT_CACHE[key]
    big = np.iinfo(np.int64).max
    v = np.where(cnt_col < 1, ply, big)
    off = np.r_[0, np.cumsum(lens)][:-1]
    out = lens.astype(np.int64).copy()
    ok = lens > 0
    f = np.minimum.reduceat(v, off[ok]) if ok.any() else np.zeros(0, np.int64)
    out[ok] = np.where(f == big, lens[ok], f)
    _EXIT_CACHE[key] = out
    return out


def depth_at(cnt, n_book, gi, ply, lens, games: np.ndarray, n_target: float) -> dict:
    """Mean and median book exit of the selected games, interpolated on log2 N to a book of n_target games."""
    xs, mean, med = [], [], []
    for j in range(LEVELS):
        e = _exit(_col(cnt, j), gi, ply, lens)[games]
        xs.append(np.log2(n_book * 2.0 ** -j))
        mean.append(e.mean())
        med.append(np.median(e))
    xs, mean, med = np.array(xs[::-1]), np.array(mean[::-1]), np.array(med[::-1])
    x = np.log2(n_target)
    extrap = bool(x > xs[-1] + 1e-9 or x < xs[0] - 1e-9)
    slope = float(np.polyfit(xs, mean, 1)[0])
    return {"mean": float(np.interp(x, xs, mean)), "median": float(np.interp(x, xs, med)), "n_book": n_book,
            "n_target": n_target, "extrapolated": extrap, "plies_per_doubling": slope,
            "mean_at_full_book": float(mean[-1])}


_COLS: dict = {}


def _clear():
    _COLS.clear()
    _EXIT_CACHE.clear()


def _col(cnt: np.ndarray, j: int) -> np.ndarray:
    """A contiguous copy of level j, kept alive so the exit cache key stays valid."""
    k = (id(cnt), j)
    if k not in _COLS:
        _COLS[k] = (cnt, np.ascontiguousarray(cnt[:, j]))
    return _COLS[k][1]


def analyze():
    pb = pl.read_parquet(PB / "probe_both.parquet", columns=["src", "game_id", "band", "year", "eco"])
    cnt_o, n_o = _load("otb_all")
    cnt_l, n_l = _load("lichess_all")
    lens = np.load(PB / "runs/otb_all/probe_lens.npy")
    keys = np.load(PB / "runs/otb_all/probe_keys.npy")
    gi, ply = ply_index(lens)
    src = pb["src"].to_numpy()
    band = pb["band"].fill_null(-1).to_numpy()
    year = pb["year"].fill_null(-1).to_numpy()
    is_l, is_o = src == "lichess", src == "otb"
    res = {"licence": "Lumbra's Gigabase OTB, CC BY-NC 4.0: research only; aggregate statistics only, no table shipped",
           "otb_book_games": n_o, "lichess_book_games": n_l, "otb_probe_games": int(is_o.sum()),
           "lichess_probe_games": int(is_l.sum()),
           "rating_note": "OTB band = mean FIDE Elo floored to 100; Lichess band = mean Lichess blitz rating. "
                          "Different scales."}
    # 1. overall, equal N (the OTB book's size)
    ov = {}
    for gname, gsel in (("lichess_games", is_l), ("otb_games", is_o)):
        idx = np.nonzero(gsel)[0]
        ov[gname] = {"in_lichess_book": depth_at(cnt_l, n_l, gi, ply, lens, idx, n_o),
                     "in_otb_book": depth_at(cnt_o, n_o, gi, ply, lens, idx, n_o)}
    res["depth_equal_N"] = ov
    _clear()
    # 2. overlap by ply: share of probe positions seen at least once in each book (full books)
    res["seen_share_by_ply"] = {}
    for gname, gsel in (("lichess_games", is_l), ("otb_games", is_o)):
        rows = {}
        for k in (4, 8, 12, 16, 20, 30):
            m = gsel[gi] & (ply == k - 1)
            rows[k] = {"in_lichess_book": float((cnt_l[m, 0] >= 1).mean()),
                       "in_otb_book": float((cnt_o[m, 0] >= 1).mean())}
        res["seen_share_by_ply"][gname] = rows
    # 3. bands: own-medium band books, equal N per band
    bands = {}
    for b in BANDS:
        co, no = _load(f"otb_b{b}")
        cl, nl = _load(f"lichess_b{b}")
        n_eq = min(no, nl)
        gl = np.nonzero(is_l & (band >= b) & (band < b + 200))[0]
        go = np.nonzero(is_o & (band >= b) & (band < b + 200))[0]
        row = {"otb_band_book_games": no, "lichess_band_book_games": nl, "equal_N": n_eq,
               "lichess_probe_games": len(gl), "otb_probe_games": len(go)}
        if len(gl) >= 300:
            row["lichess_in_lichess_band_book"] = depth_at(cl, nl, gi, ply, lens, gl, n_eq)
            row["lichess_in_otb_band_book"] = depth_at(co, no, gi, ply, lens, gl, n_eq)
        if len(go) >= 300:
            row["otb_in_otb_band_book"] = depth_at(co, no, gi, ply, lens, go, n_eq)
            row["otb_in_lichess_band_book"] = depth_at(cl, nl, gi, ply, lens, go, n_eq)
        bands[b] = row
        _clear()
        print(b, {k: (round(v["mean"], 2) if isinstance(v, dict) else v) for k, v in row.items()}, flush=True)
    res["bands"] = bands
    # 4. decades: OTB games in their own decade's book, Lichess games in each decade's book (equal N = 200k)
    dec = {}
    n_eq = 200_000
    for d in DECADES:
        cd, nd = _load(f"otb_dec{d}")
        go = np.nonzero(is_o & (year >= d) & (year < d + 10))[0]
        gl = np.nonzero(is_l)[0]
        own = depth_at(cd, nd, gi, ply, lens, go, n_eq)
        li = depth_at(cd, nd, gi, ply, lens, gl, n_eq)
        m16 = is_o[gi] & (year[gi] >= d) & (year[gi] < d + 10) & (ply == 15)
        dec[d] = {"otb_book_games": nd, "otb_probe_games": len(go), "otb_in_own_decade_book": own,
                  "lichess_in_decade_book": li,
                  "otb_ply16_seen_in_lichess_book": float((cnt_l[m16, 0] >= 1).mean()),
                  "otb_ply16_seen_in_own_decade_book": float((cd[m16, 0] >= 1).mean())}
        print(d, nd, round(own["mean"], 2), round(li["mean"], 2), flush=True)
        _clear()
    res["decades"] = dec
    # 5. positions common in one medium, rare in the other (full tables, frequency per game, plies <= 30)
    res["medium_specific_positions"] = medium_specific(keys, gi, ply, lens, pb, n_o, n_l)
    save_section("online_vs_otb", res)


def medium_specific(keys, gi, ply, lens, pb, n_o, n_l, top=15, min_share=0.002):
    ok, oc = load_table(PB / "runs/otb_all")
    lk, lc = load_table(R0_DIR)
    # candidate keys: positions in the first 30 plies of probe games (named below from those games)
    m = ply < 30
    k_u, first = np.unique(keys[m], return_index=True)
    fo, fl = lookup(ok, oc, k_u) / n_o, lookup(lk, lc, k_u) / n_l
    where_g, where_p = gi[m][first], ply[m][first]
    eps = 1e-6
    lr = np.log2((fl + eps) / (fo + eps))
    op = opening_keys()
    name = dict(zip(op["key"].to_numpy().tolist(), op["name"].to_list()))
    moves = pl.read_parquet(PB / "probe_both.parquet", columns=["moves"])["moves"]

    def describe(i):
        import chess
        b = chess.Board()
        san = []
        for u in moves[int(where_g[i])].to_list()[: int(where_p[i]) + 1]:
            mv = chess.Move.from_uci(u)
            san.append(b.san(mv))
            b.push(mv)
        line = " ".join((f"{j // 2 + 1}. " if j % 2 == 0 else "") + s for j, s in enumerate(san))
        return {"line": line, "ply": int(where_p[i]) + 1, "opening": name.get(int(k_u[i])),
                "lichess_share_of_games": float(fl[i]), "otb_share_of_games": float(fo[i]),
                "log2_ratio_online_over_otb": float(lr[i])}

    out = {"rule": f"positions in the first 30 plies of probe games, share of games reaching them >= {min_share} "
                   "in the medium where they are more common; ranked by log2(online share / OTB share)"}
    for nm, sel, order in (("online_not_otb", fl >= min_share, -lr), ("otb_not_online", fo >= min_share, lr)):
        idx = np.nonzero(sel)[0]
        idx = idx[np.argsort(order[idx])][:top]
        out[nm] = [describe(i) for i in idx]
    # overall agreement of position frequencies (positions common in either medium)
    both = (fl >= 1e-4) | (fo >= 1e-4)
    out["spearman_log_share_common_positions"] = float(
        pl.DataFrame({"a": np.log(fl[both] + eps), "b": np.log(fo[both] + eps)}).select(
            pl.corr("a", "b", method="spearman")).item())
    out["n_common_positions"] = int(both.sum())
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "count", "analyze"])
    a = ap.parse_args()
    {"prep": prep, "count": count, "analyze": analyze}[a.cmd]()

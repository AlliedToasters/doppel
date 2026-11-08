"""Position-keyed book (R0): Polyglot Zobrist counts from lab/poskey, against the move-order book (R1) of book.py.

The two book keys are information regimes (docs/SCIENCE.md §3). The position key reads only position k (R0):
a game that reaches a common position by an odd move order is still "in book". The move-order prefix key reads
the path (R1). The R1 residual is the share of plies whose position is common (>= 100 train games by position)
while the move-order prefix is rare (< 2): an unusual route to a common position.

Counting is Rust (lab/poskey: shakmaty replay, Bloom filter in front of an exact map, streaming, no sort files).
Its table (sorted uint64 keys + uint32 counts, count >= 2) has book.py's format; keys equal
chess.polyglot.zobrist_hash. Evaluation games (book.py's heldout stratified sample, 10k per band) are probed
for exact counts at 9 nested subsample levels in the same pass, by position and by (position, move) edge.

    cargo build --release --manifest-path lab/poskey/Cargo.toml
    python -m doppel_lab.posbook probe                  # heldout eval games -> data/v0/posbook/probe.parquet
    python -m doppel_lab.posbook run full               # 2026-10 train, Bloom + exact  -> data/v0/posbook/full
    python -m doppel_lab.posbook run full_p001          # same at 0.1% Bloom FPR (the table to ship)
    python -m doppel_lab.posbook run sub1m_bloom        # first 1M train games, Bloom   (exactness check)
    python -m doppel_lab.posbook run sub1m_exact        # first 1M train games, no Bloom
    python -m doppel_lab.posbook run quarter / half     # 12 / 23 train files (time and memory scaling)
    python -m doppel_lab.posbook analyze                # -> lab/results/position_book.json

DOPPEL_DATA overrides the data directory (default: <repo>/data), e.g. to run from a git worktree.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("DOPPEL_DATA", ROOT / "data"))
CORPUS = DATA / "corpus/2026-10"
BOOK = DATA / "v0/book"          # book.py outputs (move-order key)
OUT = DATA / "v0/posbook"
BIN = ROOT / "lab/poskey/target/release/poskey"
RESULTS = ROOT / "lab/results/position_book.json"
LEVELS = 9
BANDS = (800, 1500, 2400)
THREADS = 26                     # of 32 cores: leave some for other work
c = pl.col


def train_files() -> list[str]:
    return sorted(str(p) for p in (CORPUS / "train").glob("*.parquet"))


RUNS = {  # name -> (files, extra flags)
    "full": (lambda f: f, []),
    "full_p001": (lambda f: f, ["--fpr", "0.001"]),
    "half": (lambda f: f[:23], []),
    "quarter": (lambda f: f[:12], []),
    "sub1m_bloom": (lambda f: f[:2], []),
    "sub1m_exact": (lambda f: f[:2], ["--exact"]),
    "sub1m_bloom_p001": (lambda f: f[:2], ["--fpr", "0.001"]),
}


def heldout_mask(g: pl.DataFrame) -> np.ndarray:
    return (g["src"] == "heldout").to_numpy()


def probe():
    """book.py's heldout stratified eval games, in eval_games order, so rows line up with eval_counts.npy."""
    g = pl.read_parquet(BOOK / "eval_games.parquet")
    mv = pl.read_parquet(BOOK / "eval_moves.parquet", columns=["game_id", "moves"])
    ho = g.with_row_index("row").filter(c("src") == "heldout").select("row", "game_id", "band")
    p = ho.join(mv, on="game_id", how="left", maintain_order="left")
    OUT.mkdir(parents=True, exist_ok=True)
    p.write_parquet(OUT / "probe.parquet")
    print(f"probe: {p.height:,} heldout games -> {OUT / 'probe.parquet'}")


def run(name: str):
    pick, flags = RUNS[name]
    files = pick(train_files())
    out = OUT / name
    cmd = [str(BIN), "count", "--out", str(out), "--threads", str(THREADS), "--probe", str(OUT / "probe.parquet"),
           *flags, *files]
    print(" ".join(cmd[:8]), f"... ({len(files)} files)", flush=True)
    t = subprocess.run(["/usr/bin/time", "-v", *cmd], stderr=subprocess.PIPE, text=True)
    sys.stderr.write(t.stderr)
    (out / "time.txt").write_text(t.stderr)
    t.check_returncode()


# --- analysis ---------------------------------------------------------------------------------------------------

def ply_index(lens: np.ndarray):
    off = np.zeros(len(lens) + 1, np.int64)
    np.cumsum(lens, out=off[1:])
    gi = np.repeat(np.arange(len(lens)), lens)
    return off, gi, np.arange(len(gi)) - off[gi]


def exit_ply(cnt, gi, ply, lens, T):
    """First ply with count < T (book_science.exit_ply: games that never leave book get their length)."""
    out = lens.astype(np.int64).copy()
    below = np.nonzero(cnt < T)[0]
    g, i = np.unique(gi[below], return_index=True)
    out[g] = ply[below[i]]
    return out


def parent_of(cnt, ply, n_games):
    par = np.empty_like(cnt, dtype=np.float64)
    par[1:] = cnt[:-1]
    par[ply == 0] = n_games
    return par


def weird(child, par, band_p, bands):
    sup = par >= 100
    q = child[sup] / par[sup]
    return {int(b): float((q[band_p[sup] == b] < 0.01).mean()) for b in bands}


def table_compare(exact_dir: Path, bloom_dir: Path) -> dict:
    ek, ec = np.load(exact_dir / "table_keys.npy"), np.load(exact_dir / "table_counts.npy").astype(np.int64)
    bk, bc = np.load(bloom_dir / "table_keys.npy"), np.load(bloom_dir / "table_counts.npy").astype(np.int64)
    es, bs = json.loads((exact_dir / "stats.json").read_text()), json.loads((bloom_dir / "stats.json").read_text())
    i = np.searchsorted(bk, ek).clip(max=len(bk) - 1)
    hit = bk[i] == ek
    diff = np.where(hit, bc[i], 0) - ec
    j = np.searchsorted(ek, bk).clip(max=len(ek) - 1)
    false_entry = ek[j] != bk  # in the Bloom table but a true singleton
    distinct = es["first_sightings"]
    return {
        "games": es["games"], "true_distinct": distinct, "true_table_keys": len(ek),
        "true_singleton_share": 1 - len(ek) / distinct,
        "bloom_table_keys": len(bk), "bloom_fpr_target": bs["bloom_target_fpr"],
        "bloom_fpr_at_end": bs["bloom_fpr_at_end"], "bloom_bytes": bs["bloom_bytes"],
        "missing_from_bloom_table": int((~hit).sum()),
        "exact_count_share": float((diff == 0).mean()),
        "off_by_one_rate": float((diff == 1).mean()),     # among true count >= 2 keys
        "other_error_rate": float(((diff != 0) & (diff != 1)).mean()),
        "false_entries": int(false_entry.sum()),          # true singletons stored at count 2
        "false_entry_share_of_table": float(false_entry.mean()),
        "false_entry_rate_per_singleton": float(false_entry.sum() / (distinct - len(ek))),
        "singletons_est_bloom": bs["singletons_est"], "singletons_true": distinct - len(ek),
    }


def analyze():
    g = pl.read_parquet(BOOK / "eval_games.parquet")
    mo_counts = np.load(BOOK / "eval_counts.npy").astype(np.int64)
    mo_lens = np.load(BOOK / "eval_lens.npy")
    meta = json.loads((BOOK / "train_meta.json").read_text())
    mo_off, _, _ = ply_index(mo_lens)
    ho = np.nonzero(heldout_mask(g))[0]
    pb = pl.read_parquet(OUT / "probe.parquet", columns=["row", "band"])
    assert (pb["row"].to_numpy() == ho).all()
    full = OUT / "full"
    st = json.loads((full / "stats.json").read_text())
    pc = np.load(full / "probe_counts.npy").astype(np.int64)
    pe = np.load(full / "probe_edge_counts.npy").astype(np.int64)
    lens = np.load(full / "probe_lens.npy")
    n_mismatch = int((lens != mo_lens[ho]).sum())
    assert n_mismatch == 0, f"{n_mismatch} probe games replayed to a different length"
    # move-order counts for the same heldout plies
    mo = np.concatenate([mo_counts[mo_off[r]:mo_off[r + 1]] for r in ho])
    off, gi, ply = ply_index(lens)
    band = pb["band"].to_numpy()
    band_p = band[gi]
    N = st["games"]
    assert N == meta["games"], (N, meta["games"])
    bands = sorted(set(band.tolist()))

    def summary(cnt0, par_child, edge=None):
        e1 = exit_ply(cnt0, gi, ply, lens, 1)
        e10 = exit_ply(cnt0, gi, ply, lens, 10)
        par = parent_of(cnt0, ply, N)
        row = {"T1_median_exit": {int(b): float(np.median(e1[band == b])) for b in bands},
               "T1_mean_exit": {int(b): float(e1[band == b].mean()) for b in bands},
               "T10_median_exit": {int(b): float(np.median(e10[band == b])) for b in bands},
               "T1_median_exit_all": float(np.median(e1)), "T1_mean_exit_all": float(e1.mean()),
               "weird_rate": weird(edge if edge is not None else cnt0, par, band_p, bands)}
        if edge is not None:
            row["weird_rate_child_over_parent"] = weird(np.minimum(cnt0, par), par, band_p, bands)
        return row, e1

    r1, e1_mo = summary(mo[:, 0], None)
    r0, e1_pos = summary(pc[:, 0], None, edge=pe[:, 0])
    gain = e1_pos - e1_mo
    # R1 residual: position common (>= 100 by position key), move-order prefix rare (< 2)
    common = pc[:, 0] >= 100
    resid = common & (mo[:, 0] < 2)
    residual = {"share_of_plies": float(resid.mean()), "share_of_common_position_plies": float(resid[common].mean()),
                "games_with_any": float(np.bincount(gi, resid, len(lens)).astype(bool).mean()),
                "by_band_share_of_common": {int(b): float(resid[common & (band_p == b)].mean()) for b in bands}}
    # depth vs corpus size (nested levels; position levels use a game_id hash, move-order levels book.py's rng)
    scaling = []
    for j in range(LEVELS):
        row = {"level": j, "games": N * 2.0 ** -j}
        for name, cnt in (("R1_move_order", mo), ("R0_position", pc)):
            e = exit_ply(cnt[:, j], gi, ply, lens, 1)
            row[name] = {"T1_mean": float(e.mean()), "T1_median": float(np.median(e)),
                         "T1_median_by_band": {int(b): float(np.median(e[band == b])) for b in BANDS}}
        scaling.append(row)
    x = np.log2([r["games"] for r in scaling])
    slopes = {k: float(np.polyfit(x, [r[k]["T1_mean"] for r in scaling], 1)[0]) for k in ("R1_move_order", "R0_position")}

    runs = {}
    for name in ("quarter", "half", "full", "full_p001"):
        p = OUT / name / "stats.json"
        if p.exists():
            s = json.loads(p.read_text())
            runs[name] = {k: s[k] for k in ("files", "games", "keys", "first_sightings", "table_keys", "map_entries",
                                             "singletons_est", "bloom_bytes", "bloom_fpr_at_end", "map_bytes_est",
                                             "seconds_count", "games_per_s", "peak_rss_gb", "threads")}
            runs[name]["singleton_share_est"] = s["singletons_est"] / s["first_sightings"]
    fs = np.array(st["first_sightings_by_ply"], float)
    kp = np.array(st["keys_by_ply"], float)
    exactness = {}
    if (OUT / "sub1m_exact/stats.json").exists():
        for b in ("sub1m_bloom", "sub1m_bloom_p001"):
            if (OUT / b / "stats.json").exists():
                exactness[b] = table_compare(OUT / "sub1m_exact", OUT / b)
    res = {
        "date": "2026-11-07",
        "train_games": N, "eval_games": int(len(lens)), "eval_plies": int(len(gi)), "cap": st["cap"],
        "regimes": {"R0": "position key (Polyglot Zobrist): reads only position k",
                    "R1": "move-order prefix key (book.py): reads the path to k"},
        "book_exit_R1_move_order": r1, "book_exit_R0_position": r0,
        "R0_minus_R1_T1_exit": {"mean": float(gain.mean()), "median": float(np.median(gain)),
                                "by_band_mean": {int(b): float(gain[band == b].mean()) for b in bands}},
        "weird_rule": "R1: c(prefix k) / c(prefix k-1); R0: c(position k-1, move) / c(position k-1); "
                      "weird = < 1% where the parent count >= 100. weird_rate_child_over_parent uses "
                      "min(c(position k), c(position k-1)) / c(position k-1), which transpositions inflate",
        "R1_residual": residual,
        "scaling": scaling, "plies_per_doubling_T1_mean": slopes,
        "runs": runs,
        "position_table": {"distinct_est": st["first_sightings"], "table_keys": st["table_keys"],
                           "singleton_share_est": st["singletons_est"] / st["first_sightings"],
                           "new_key_share_by_ply": (fs / np.maximum(kp, 1)).round(4).tolist()},
        "exactness_1M": exactness,
        "outputs": "data/v0/posbook/<run>/{table_keys,table_counts}.npy (book.py table format), stats.json",
    }
    RESULTS.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k not in ("scaling", "position_table")}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["probe", "run", "analyze"])
    ap.add_argument("name", nargs="?", default="full", choices=list(RUNS))
    a = ap.parse_args()
    {"probe": probe, "run": lambda: run(a.name), "analyze": analyze}[a.cmd]()

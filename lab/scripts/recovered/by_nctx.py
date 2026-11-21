import numpy as np, polars as pl
from doppel_lab import elo_drift
c = pl.col
B = ((1, 10), (10, 20), (20, 30), (30, 40), (40, 41))
ARMS = ("lookup", "ttt40", "lookup_ttt40")
acc = {b: {"n": 0, "gid": [], **{a: [] for a in ARMS}} for b in B}
for band in (700, 1100, 1500, 1900, 2300):
    elo_drift.BAND = band
    s = elo_drift.sides()
    h = np.load(f"data/allie/combo/stack_hits_{band}.npz", allow_pickle=True)
    pos = pl.DataFrame({"game_id": h["game_id"].astype(str), "ply": h["ply"].astype(np.int64),
                        "i": np.arange(len(h["ply"]))}).with_columns(col=c("ply") % 2)
    pos = pos.join(s.select("game_id", "col", "n_ctx"), on=["game_id", "col"], how="left").sort("i")
    n = pos["n_ctx"].fill_null(0).to_numpy()
    gid = np.array([f"{band}:{g}" for g in pos["game_id"].to_list()])
    m = h["maia"].astype(float)
    for lo, hi in B:
        sel = (n >= lo) & (n < hi)
        acc[(lo, hi)]["n"] += int(sel.sum()); acc[(lo, hi)]["gid"].append(gid[sel])
        for a in ARMS:
            acc[(lo, hi)][a].append(h[a][sel].astype(float) - m[sel])
    print("done", band, flush=True)
rng = np.random.default_rng(0)
for b, v in acc.items():
    g = np.unique(np.concatenate(v["gid"]), return_inverse=True)[1]
    out = []
    for a in ARMS:
        d = np.concatenate(v[a]); gs, gn = np.bincount(g, d), np.bincount(g)
        bs = 100 * gs[rng.integers(0, len(gs), (1000, len(gs)))].sum(1) / gn[rng.integers(0, len(gs), (1000, len(gs)))].sum(1) if False else None
        idx = rng.integers(0, len(gs), (1000, len(gs)))
        bs = 100 * gs[idx].sum(1) / gn[idx].sum(1)
        out.append(f"{a} {100*d.mean():+.2f} [{np.percentile(bs,2.5):+.2f},{np.percentile(bs,97.5):+.2f}]")
    print(f"ctx {b[0]}-{b[1]-1}: {v['n']:,} pos  " + "  ".join(out), flush=True)

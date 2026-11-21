"""Residual weight alpha for the 200-game adapter, chosen OFF the test set, then scored once on test.

Fit data: allie_ttt --fit 5 --sample 200 (per band 200 random sides; their 5 most recent context games held out of
training and epoch choice) -> data/allie/ttt_<band>_ctx200_fit5_s200_fit.npz.
Choice: logits = Maia-3 + alpha * (adapter - Maia-3); alpha on a grid, criterion top-1 and (separately) CE; a
single alpha (pooled bands) and a per-phase pair (ply < 40, ply >= 40). 5-fold by side gives the fold spread.
Test: the full ctx200 run's adapters (ttt_<band>_ctx200.npz) with the chosen alpha; game bootstrap vs Maia-3 and vs
the gated arm (ttt200g40).
"""
import json
import sys
import zlib

import numpy as np

from doppel_lab.allie_knn import games, knn_dir
from doppel_lab.combo import align_any

ROOT = "/home/alliedtoasters/projects/chess/bot"
BANDS = (700, 1100, 1500, 1900, 2300)
GRID = np.round(np.arange(0, 1.501, 0.125), 3)


def lsm(x):
    x = x - x.max(1, keepdims=True)
    return x - np.log(np.exp(x).sum(1, keepdims=True))


def pad(flat, lens, fill):
    L = lens.max()
    out = np.full((len(lens), L), fill, np.float32)
    st = np.r_[0, np.cumsum(lens)[:-1]]
    for i, (s, n) in enumerate(zip(st, lens)):
        out[i, :n] = flat[s:s + n]
    return out


# ---------------------------------------------------------------- fit set
F = []
for b in BANDS:
    z = np.load(f"{ROOT}/data/allie/ttt_{b}_ctx200_fit5_s200_fit.npz")
    lm, la = pad(z["lm"], z["lens"], -1e9), pad(z["la"], z["lens"], -1e9)
    F.append(dict(band=b, side=z["pid"] + 1000 * b, ply=z["ply"], lm=lsm(lm), r=np.where(lm > -1e8, la - lm, 0),
                  t=z["truth"]))


def score(d, a_open, a_late):
    a = np.where(d["ply"] < 40, a_open, a_late)[:, None]
    x = lsm(d["lm"] + a * d["r"])
    ar = np.arange(len(x))
    return (x.argmax(1) == d["t"]).astype(float), -x[ar, d["t"]]


def table(sel_fn):
    """sum of hits and CE over selected positions, for every (a_open, a_late) on the grid."""
    H = np.zeros((len(GRID), len(GRID))); C = np.zeros_like(H); N = 0
    for d in F:
        s = sel_fn(d)
        if not s.any():
            continue
        dd = {k: (v[s] if isinstance(v, np.ndarray) else v) for k, v in d.items()}
        for i, ao in enumerate(GRID):
            for j, al in enumerate(GRID):
                h, c = score(dd, ao, al)
                H[i, j] += h.sum(); C[i, j] += c.sum()
        N += s.sum()
    return H / N, C / N, N


fold = lambda d: np.array([zlib.crc32(str(x).encode()) % 5 for x in d["side"]])
for d in F:
    d["fold"] = fold(d)
res = {"grid": GRID.tolist(), "fit_positions": {d["band"]: int(len(d["t"])) for d in F}}
H, C, N = table(lambda d: np.ones(len(d["t"]), bool))
diag = np.arange(len(GRID))
res["single"] = {"by_top1": float(GRID[np.argmax(H[diag, diag])]), "by_ce": float(GRID[np.argmin(C[diag, diag])]),
                 "top1_curve_pts": (100 * (H[diag, diag] - H[0, 0])).round(3).tolist(),
                 "ce_curve": (C[diag, diag] - C[0, 0]).round(4).tolist()}
io, jl = np.unravel_index(np.argmax(H), H.shape)
ic, jc = np.unravel_index(np.argmin(C), C.shape)
res["phase"] = {"by_top1": [float(GRID[io]), float(GRID[jl])], "by_ce": [float(GRID[ic]), float(GRID[jc])]}
res["folds"] = []
for k in range(5):
    Hk, Ck, _ = table(lambda d: d["fold"] != k)
    res["folds"].append({"single_top1": float(GRID[np.argmax(Hk[diag, diag])]),
                         "single_ce": float(GRID[np.argmin(Ck[diag, diag])])})
res["per_band_single_by_top1"] = {}
for d in F:
    Hb, Cb, _ = table(lambda e, b=d["band"]: np.full(len(e["t"]), e["band"] == b))
    res["per_band_single_by_top1"][d["band"]] = {"top1": float(GRID[np.argmax(Hb[diag, diag])]),
                                                 "ce": float(GRID[np.argmin(Cb[diag, diag])])}
print(json.dumps({k: res[k] for k in ("fit_positions", "single", "phase", "folds", "per_band_single_by_top1")}),
      flush=True)

# ---------------------------------------------------------------- test, scored once per policy
pol = {"ttt200": (1.0, 1.0), "ttt200g40": (1.0, 0.0),
       "alpha_single_top1": (res["single"]["by_top1"],) * 2, "alpha_single_ce": (res["single"]["by_ce"],) * 2,
       "alpha_phase_top1": tuple(res["phase"]["by_top1"]), "alpha_phase_ce": tuple(res["phase"]["by_ce"])}
hits = {k: [] for k in pol}; ces = {k: [] for k in pol}; hm_all, cm_all, gids, bands = [], [], [], []
for b in BANDS:
    t = np.load(f"{ROOT}/data/allie/ttt_{b}_ctx200.npz", allow_pickle=True)
    gid, ply = t["game_id"].astype(str), t["ply"]
    g = games(b)
    gmap = dict(zip(g["game_id"].to_list(), range(g.height)))
    kd = knn_dir(b)
    off, played = np.load(kd / "off.npy"), np.load(kd / "played.npy")
    tz = np.load(kd / "test.npz")
    trow = np.full(off[-1] + 1, -1, np.int64)
    trow[tz["pos"]] = np.arange(len(tz["pos"]))
    q = off[np.array([gmap[x] for x in gid])] + ply
    rows = trow[q]
    legal = tz["legal"][rows].astype(np.int64)
    ok = legal >= 0
    lm = lsm(np.where(ok, tz["logp"][rows].astype(np.float32), -1e9))
    la = lsm(np.where(ok, np.nan_to_num(align_any(t["legal"].astype(np.int64), t["logp"].astype(np.float32), legal),
                                        neginf=-1e9), -1e9))
    tr = np.argmax(legal == played[q].astype(np.int64)[:, None], 1)
    d = dict(lm=lm, r=np.where(ok, la - lm, 0), t=tr, ply=ply)
    hm, cm = score(d, 0.0, 0.0)
    hm_all.append(hm); cm_all.append(cm); gids.append(np.char.add(f"{b}:", gid)); bands.append(np.full(len(gid), b))
    for k, (ao, al) in pol.items():
        h, c_ = score(d, ao, al)
        hits[k].append(h); ces[k].append(c_)
hm, cm = np.concatenate(hm_all), np.concatenate(cm_all)
gi = np.unique(np.concatenate(gids), return_inverse=True)[1]
band = np.concatenate(bands)
rng = np.random.default_rng(0)
ix = rng.integers(0, gi.max() + 1, (1000, gi.max() + 1))
gn = np.bincount(gi)


def boot(v):
    gs = np.bincount(gi, v)
    bs = gs[ix].sum(1) / gn[ix].sum(1)
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


res["test"] = {"positions": int(len(hm)), "maia_top1": float(hm.mean()), "maia_ce": float(cm.mean()), "policies": {}}
H40 = np.concatenate(hits["ttt200g40"])
for k in pol:
    h, c_ = np.concatenate(hits[k]), np.concatenate(ces[k])
    res["test"]["policies"][k] = {
        "alphas": pol[k], "top1_pts": 100 * float((h - hm).mean()), "top1_ci": [100 * x for x in boot(h - hm)],
        "ce_delta": float((c_ - cm).mean()), "ce_ci": boot(c_ - cm),
        "vs_ttt200g40_pts": 100 * float((h - H40).mean()), "vs_ttt200g40_ci": [100 * x for x in boot(h - H40)],
        "by_band_pts": {str(b): round(100 * float((h - hm)[band == b].mean()), 3) for b in BANDS}}
    print(k, json.dumps(res["test"]["policies"][k]), flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1, default=float)

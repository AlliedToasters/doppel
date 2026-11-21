"""12f tier 1 + per-side late-gate proxy, CPU, from saved arrays only.

(a) Adapter gain by Maia-3's policy margin (p1 - p2) and entropy, per band and phase.
(b) Epoch-selection proxy: per side, decide late (ply >= 40) adapter on/off on alternate late positions, score on the
    other half (both directions averaged). Compares always-on (ttt200), gate (ttt200g40), and per-side choice.
"""
import json
import sys

import numpy as np

from doppel_lab.allie_knn import games, knn_dir

ROOT = "/home/alliedtoasters/projects/chess/bot"
MB = [0, .1, .25, .5, .75, 1.01]
out = {}
for band in [700, 1100, 1500, 1900, 2300]:
    h = np.load(f"{ROOT}/data/allie/combo/stack_hits_{band}.npz", allow_pickle=True)
    gid, ply = h["game_id"].astype(str), h["ply"]
    g = games(band)
    gmap = dict(zip(g["game_id"].to_list(), range(g.height)))
    off = np.load(knn_dir(band) / "off.npy")
    tz = np.load(knn_dir(band) / "test.npz")
    trow = np.full(off[-1] + 1, -1, np.int64)
    trow[tz["pos"]] = np.arange(len(tz["pos"]))
    q = off[np.array([gmap[x] for x in gid])] + ply
    rows = trow[q]
    assert (rows >= 0).all()
    lp = tz["logp"][rows].astype(np.float32)
    lp = np.where(tz["legal"][rows] >= 0, lp, -np.inf)
    p = np.exp(lp - lp.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    ps = -np.sort(-p, 1)
    margin = ps[:, 0] - ps[:, 1]
    ent = -(np.where(p > 0, p * np.log(np.where(p > 0, p, 1)), 0)).sum(1)
    m = h["maia"].astype(float)
    r = {"bins": {}, "late_gate": {}}
    for ph, sel in (("open<40", ply < 40), ("late>=40", ply >= 40)):
        for lo, hi in zip(MB[:-1], MB[1:]):
            s = sel & (margin >= lo) & (margin < hi)
            if s.sum() < 500:
                continue
            mt = m[s].mean()
            r["bins"][f"{ph} m{lo}-{hi}"] = {"n": int(s.sum()), "maia": round(100 * mt, 1),
                                             **{k: round(100 * (h[k][s].mean() - mt), 2) for k in ("ttt200", "ttt40")},
                                             "ttt200_headroom%": round(100 * (h["ttt200"][s].mean() - mt) / (1 - mt), 2)}
    # (b) per-side late choice, split-half within the side's late positions
    side = np.char.add(np.char.add(gid, ":"), (ply % 2).astype(str))
    late = np.where(ply >= 40)[0]
    d = h["ttt200"][late].astype(float) - m[late]
    su, si = np.unique(side[late], return_inverse=True)
    order = np.argsort(si, kind="stable")
    rank = np.empty(len(late), np.int64)
    first = np.r_[0, np.flatnonzero(np.diff(si[order])) + 1]
    cnt = np.diff(np.r_[first, len(order)])
    rank[order] = np.arange(len(order)) - np.repeat(first, cnt)
    half = rank % 2
    tot = 0.0
    for a in (0, 1):
        dec = np.bincount(si[half == a], d[half == a], minlength=len(su)) > 0
        tot += (d[half != a] * dec[si[half != a]]).sum()
    n_all = len(ply)
    r["late_gate"] = {"late_positions": int(len(late)), "late_sides": int(len(su)),
                      "always_on_pts_total": round(100 * d.sum() / n_all, 3),
                      "gate_off_pts_total": 0.0,
                      "per_side_choice_pts_total": round(100 * tot / n_all, 3),
                      "oracle_per_side_pts_total": round(100 * np.clip(np.bincount(si, d), 0, None).sum() / n_all, 3)}
    out[band] = r
    print(band, json.dumps(r["late_gate"]), flush=True)
    for k, v in r["bins"].items():
        print("  ", k, v)
json.dump(out, open(sys.argv[1], "w"), indent=1)

"""How strongly should the 200-game test-time adapter speak? Post-hoc analyses of saved outputs (CPU, no GPU).

    python -m doppel_lab.ttt_weight bins     # gain by Maia-3 margin x phase, per-side late gate proxy, policy
                                             #   search cross-fitted on test-game halves -> lab/results/ttt_weight_bins.json
    python -m doppel_lab.ttt_weight fit      # residual weight chosen on held-out pre-test games, test scored once
                                             #   -> lab/results/ttt_weight_fit.json

Inputs: data/allie/ttt_<band>_ctx200.npz (allie_ttt run --ctx 200: the adapter's log-probs on every test position),
the bucket's Maia-3 log-probs (allie_knn test.npz), and for `fit` data/allie/ttt_<band>_ctx200_fit5_s200_fit.npz
(allie_ttt run --ctx 200 --fit 5 --sample 200: 200 random sides per band, their 5 most recent context games held out
of training and epoch choice; Maia-3 and adapter log-probs on those games).
Weighted logits = Maia-3 + alpha * (adapter - Maia-3), renormalised over the legal moves. Margin = Maia-3's p1 - p2.
CE in nats per position. Game bootstrap (1,000) for intervals.
"""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BANDS = (700, 1100, 1500, 1900, 2300)
MB = (0, .1, .25, .5, .75, 1.01)


def lsm(x):
    x = x - x.max(1, keepdims=True)
    return x - np.log(np.exp(x).sum(1, keepdims=True))


def test_arrays(band: int) -> dict:
    """Per test position of the bucket: Maia-3 and adapter log-probs on the same legal rows, truth, ply, game."""
    from doppel_lab.allie_knn import games, knn_dir
    from doppel_lab.combo import align_any
    t = np.load(ROOT / f"data/allie/ttt_{band}_ctx200.npz", allow_pickle=True)
    gid, ply = t["game_id"].astype(str), t["ply"]
    g = games(band)
    gmap = dict(zip(g["game_id"].to_list(), range(g.height)))
    kd = knn_dir(band)
    off, played = np.load(kd / "off.npy"), np.load(kd / "played.npy")
    tz = np.load(kd / "test.npz")
    trow = np.full(off[-1] + 1, -1, np.int64)
    trow[tz["pos"]] = np.arange(len(tz["pos"]))
    q = off[np.array([gmap[x] for x in gid])] + ply
    rows = trow[q]
    assert (rows >= 0).all()
    legal = tz["legal"][rows].astype(np.int64)
    ok = legal >= 0
    lm = lsm(np.where(ok, tz["logp"][rows].astype(np.float32), -1e9))
    la = lsm(np.where(ok, np.nan_to_num(align_any(t["legal"].astype(np.int64), t["logp"].astype(np.float32), legal),
                                        neginf=-1e9), -1e9))
    truth = np.argmax(legal == played[q].astype(np.int64)[:, None], 1)
    assert (legal[np.arange(len(q)), truth] == played[q]).all()
    ps = -np.sort(-np.exp(lm), 1)
    return dict(lm=lm, r=np.where(ok, la - lm, 0.0), t=truth, ply=ply, gid=gid, margin=ps[:, 0] - ps[:, 1])


def score(d: dict, a_open: float, a_late: float, gate=None):
    """Top-1 hits and CE of Maia-3 + alpha * residual (alpha by phase; gate = bool mask where the residual applies)."""
    a = np.where(d["ply"] < 40, a_open, a_late)
    if gate is not None:
        a = a * gate
    x = lsm(d["lm"] + a[:, None] * d["r"])
    return (x.argmax(1) == d["t"]).astype(np.float32), -x[np.arange(len(x)), d["t"]]


def bins():
    alpha, tau = (0.25, 0.5, 0.75, 1.0, 1.25), (0.1, 0.25, 0.5, 1.01)
    out, pool = {}, []
    for band in BANDS:
        d = test_arrays(band)
        hm, cm = score(d, 0, 0)
        ha, ca = score(d, 1, 1)
        res = {"positions": len(hm), "maia_top1": float(hm.mean()), "by_phase_margin": {}, "policies": {}}
        for ph, sel in (("ply<40", d["ply"] < 40), ("ply>=40", d["ply"] >= 40)):
            for lo, hi in zip(MB[:-1], MB[1:]):
                s = sel & (d["margin"] >= lo) & (d["margin"] < hi)
                mt = float(hm[s].mean())
                res["by_phase_margin"][f"{ph} m{lo}-{hi}"] = {
                    "n": int(s.sum()), "maia_top1": mt, "top1_pts": 100 * float(ha[s].mean() - mt),
                    "headroom_pct": 100 * float(ha[s].mean() - mt) / (1 - mt),
                    "ce_maia": float(cm[s].mean()), "ce_delta": float(ca[s].mean() - cm[s].mean())}
        # per-side late on/off, decided on alternate late positions of the side, scored on the others
        side = np.char.add(np.char.add(d["gid"], ":"), (d["ply"] % 2).astype(str))
        late = np.flatnonzero(d["ply"] >= 40)
        dl = (ha - hm)[late]
        su, si = np.unique(side[late], return_inverse=True)
        o = np.argsort(si, kind="stable")
        first = np.r_[0, np.flatnonzero(np.diff(si[o])) + 1]
        rank = np.empty(len(late), np.int64)
        rank[o] = np.arange(len(o)) - np.repeat(first, np.diff(np.r_[first, len(o)]))
        tot = 0.0
        for a in (0, 1):
            on = np.bincount(si[rank % 2 == a], dl[rank % 2 == a], minlength=len(su)) > 0
            tot += float((dl[rank % 2 != a] * on[si[rank % 2 != a]]).sum())
        res["late_per_side_choice"] = {"always_on_pts": 100 * float(dl.sum()) / len(hm), "gate_off_pts": 0.0,
                                       "per_side_split_half_pts": 100 * tot / len(hm)}
        # policy grid, chosen on one half of the test games (crc32 parity), scored on the other
        half = np.array([zlib.crc32(x.encode()) % 2 for x in d["gid"]])
        hits, ces = {}, {}
        for pg in ("all", "lt40"):
            for t_ in tau:
                gate = (d["margin"] < t_) & ((d["ply"] < 40) if pg == "lt40" else True)
                for a in alpha:
                    k = f"a{a}_t{t_}_{pg}"
                    h, c = score(d, a, a, gate)
                    hits[k], ces[k] = h - hm, c - cm
                    res["policies"][k] = {"top1_pts": 100 * float(hits[k].mean()), "ce_delta": float(ces[k].mean())}
        held = [max(hits, key=lambda k: hits[k][half == A].mean()) for A in (0, 1)]
        res["crossfit"] = {"chosen_on_half": held,
                           "heldout_top1_pts": 100 * sum(float(hits[k][half != A].sum()) for A, k in zip((0, 1), held))
                           / len(hm),
                           "heldout_ce_delta": sum(float(ces[k][half != A].sum()) for A, k in zip((0, 1), held))
                           / len(hm)}
        out[band] = res
        pool.append((hits, ces, half))
        print(band, json.dumps({k: res[k] for k in ("late_per_side_choice", "crossfit")}), flush=True)
    n = sum(len(p[2]) for p in pool)
    keys = list(pool[0][0])
    held = [max(keys, key=lambda k: sum(float(p[0][k][p[2] == A].sum()) for p in pool)) for A in (0, 1)]
    out["pooled_crossfit"] = {
        "chosen_on_half": held,
        "heldout_top1_pts": 100 * sum(float(p[0][k][p[2] != A].sum()) for A, k in zip((0, 1), held) for p in pool) / n,
        "heldout_ce_delta": sum(float(p[1][k][p[2] != A].sum()) for A, k in zip((0, 1), held) for p in pool) / n}
    print("pooled", out["pooled_crossfit"])
    (ROOT / "lab/results/ttt_weight_bins.json").write_text(json.dumps(out, indent=1) + "\n")


def fit():
    grid = np.round(np.arange(0, 1.501, 0.125), 3)
    F = []
    for b in BANDS:
        z = np.load(ROOT / f"data/allie/ttt_{b}_ctx200_fit5_s200_fit.npz")
        st, L = np.r_[0, np.cumsum(z["lens"])[:-1]], z["lens"].max()
        lm = np.full((len(st), L), -1e9, np.float32)
        la = np.full((len(st), L), -1e9, np.float32)
        for i, (s, k) in enumerate(zip(st, z["lens"])):
            lm[i, :k], la[i, :k] = z["lm"][s:s + k], z["la"][s:s + k]
        side = z["pid"] + 1000 * b
        F.append(dict(band=b, lm=lsm(lm), r=np.where(lm > -1e8, la - lm, 0.0), t=z["truth"], ply=z["ply"],
                      fold=np.array([zlib.crc32(str(x).encode()) % 5 for x in side])))

    def table(sel):
        H = np.zeros((len(grid), len(grid))); C = np.zeros_like(H); N = 0
        for d in F:
            s = sel(d)
            if not s.any():
                continue
            dd = {k: v[s] for k, v in d.items() if isinstance(v, np.ndarray)}
            for i, ao in enumerate(grid):
                for j, al in enumerate(grid):
                    h, c = score(dd, ao, al)
                    H[i, j] += h.sum(); C[i, j] += c.sum()
            N += int(s.sum())
        return H / N, C / N
    dg = np.arange(len(grid))
    H, C = table(lambda d: np.ones(len(d["t"]), bool))
    res = {"grid": grid.tolist(), "fit_positions": {str(d["band"]): int(len(d["t"])) for d in F},
           "single": {"by_top1": float(grid[np.argmax(H[dg, dg])]), "by_ce": float(grid[np.argmin(C[dg, dg])]),
                      "top1_curve_pts": (100 * (H[dg, dg] - H[0, 0])).tolist(), "ce_curve": (C[dg, dg] - C[0, 0]).tolist()},
           "phase": {"by_top1": [float(grid[i]) for i in np.unravel_index(np.argmax(H), H.shape)],
                     "by_ce": [float(grid[i]) for i in np.unravel_index(np.argmin(C), C.shape)]},
           "folds": [], "per_band": {}}
    for k in range(5):
        Hk, Ck = table(lambda d: d["fold"] != k)
        res["folds"].append({"by_top1": float(grid[np.argmax(Hk[dg, dg])]), "by_ce": float(grid[np.argmin(Ck[dg, dg])])})
    for b in BANDS:
        Hb, Cb = table(lambda d: np.full(len(d["t"]), d["band"] == b))
        res["per_band"][str(b)] = {"by_top1": float(grid[np.argmax(Hb[dg, dg])]), "by_ce": float(grid[np.argmin(Cb[dg, dg])])}
    print(json.dumps({k: res[k] for k in ("fit_positions", "single", "phase", "folds", "per_band")}), flush=True)
    pol = {"ttt200": (1.0, 1.0), "ttt200g40": (1.0, 0.0), "alpha_top1": (res["single"]["by_top1"],) * 2,
           "alpha_ce": (res["single"]["by_ce"],) * 2, "alpha_phase_top1": tuple(res["phase"]["by_top1"]),
           "alpha_phase_ce": tuple(res["phase"]["by_ce"])}
    hits, ces, hm_, cm_, gids, bands = {k: [] for k in pol}, {k: [] for k in pol}, [], [], [], []
    for b in BANDS:
        d = test_arrays(b)
        hm, cm = score(d, 0, 0)
        hm_.append(hm); cm_.append(cm); gids.append(np.char.add(f"{b}:", d["gid"])); bands.append(np.full(len(hm), b))
        for k, (ao, al) in pol.items():
            h, c = score(d, ao, al)
            hits[k].append(h); ces[k].append(c)
    hm, cm, band = np.concatenate(hm_), np.concatenate(cm_), np.concatenate(bands)
    gi = np.unique(np.concatenate(gids), return_inverse=True)[1]
    ix = np.random.default_rng(0).integers(0, gi.max() + 1, (1000, gi.max() + 1))
    gn = np.bincount(gi)

    def boot(v):
        bs = np.bincount(gi, v)[ix].sum(1) / gn[ix].sum(1)
        return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    g40 = np.concatenate(hits["ttt200g40"])
    res["test"] = {"positions": int(len(hm)), "maia_top1": float(hm.mean()), "maia_ce": float(cm.mean()), "arms": {}}
    for k in pol:
        h, c = np.concatenate(hits[k]), np.concatenate(ces[k])
        res["test"]["arms"][k] = {
            "alphas": list(pol[k]), "top1_pts": 100 * float((h - hm).mean()), "top1_ci": [100 * x for x in boot(h - hm)],
            "ce_delta": float((c - cm).mean()), "ce_ci": boot(c - cm),
            "vs_ttt200g40_pts": 100 * float((h - g40).mean()), "vs_ttt200g40_ci": [100 * x for x in boot(h - g40)],
            "by_band_pts": {str(b): 100 * float((h - hm)[band == b].mean()) for b in BANDS},
            "by_band_ce": {str(b): float((c - cm)[band == b].mean()) for b in BANDS}}
        print(k, json.dumps(res["test"]["arms"][k]), flush=True)
    (ROOT / "lab/results/ttt_weight_fit.json").write_text(json.dumps(res, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bins", "fit"])
    a = ap.parse_args()
    bins() if a.cmd == "bins" else fit()


if __name__ == "__main__":
    main()

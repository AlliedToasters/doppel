"""CPU: how strongly should the 200-game adapter speak? From saved outputs only (no GPU).

logits = Maia-3 + alpha * (adapter - Maia-3) where gate(margin < tau, ply gate); else Maia-3.
Grid: alpha x tau x ply gate. Policies chosen on half A of the test games (crc32 parity), scored on half B, and the
reverse; reported as the average of the two held-out halves. Metric for choice: top-1 (also reported: CE in nats).
Also CE by phase x margin bin for the plain adapter (alpha 1, no gate).
"""
import json
import sys
import zlib

import numpy as np

from doppel_lab.allie_knn import games, knn_dir
from doppel_lab.combo import align_any

ROOT = "/home/alliedtoasters/projects/chess/bot"
ALPHA = (0.25, 0.5, 0.75, 1.0, 1.25)
TAU = (0.1, 0.25, 0.5, 1.01)
MB = (0, .1, .25, .5, .75, 1.01)


def lsm(x):
    x = x - x.max(1, keepdims=True)
    return x - np.log(np.exp(x).sum(1, keepdims=True))


out = {}
pool = []
for band in [700, 1100, 1500, 1900, 2300]:
    t = np.load(f"{ROOT}/data/allie/ttt_{band}_ctx200.npz", allow_pickle=True)
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
    legal = tz["legal"][rows].astype(np.int64)
    ok = legal >= 0
    lm = lsm(np.where(ok, tz["logp"][rows].astype(np.float32), -1e9))
    la = np.nan_to_num(align_any(t["legal"].astype(np.int64), t["logp"].astype(np.float32), legal), neginf=-1e9)
    la = lsm(np.where(ok, la, -1e9))
    truth = np.argmax(legal == played[q].astype(np.int64)[:, None], 1)
    assert (legal[np.arange(len(q)), truth] == played[q]).all()
    pm = np.exp(lm)
    ps = -np.sort(-pm, 1)
    margin = ps[:, 0] - ps[:, 1]
    r = np.where(ok, la - lm, 0.0)
    ar = np.arange(len(q))
    half = np.array([zlib.crc32(x.encode()) % 2 for x in gid])
    hit_m = lm.argmax(1) == truth
    ce_m = -lm[ar, truth]
    # sanity: plain adapter top-1 gain vs the saved run
    hit_a = la.argmax(1) == truth
    res = {"positions": len(q), "maia_top1": float(hit_m.mean()),
           "ttt200_gain_check": 100 * float(hit_a.mean() - hit_m.mean()),
           "ce_by_bin": {}, "policies": {}}
    for ph, sel in (("open<40", ply < 40), ("late>=40", ply >= 40)):
        for lo, hi in zip(MB[:-1], MB[1:]):
            s = sel & (margin >= lo) & (margin < hi)
            res["ce_by_bin"][f"{ph} m{lo}-{hi}"] = {
                "n": int(s.sum()), "top1_pts": round(100 * float(hit_a[s].mean() - hit_m[s].mean()), 2),
                "ce_maia": round(float(ce_m[s].mean()), 4),
                "ce_delta": round(float((-la[ar, truth][s]).mean() - ce_m[s].mean()), 4)}
    hits, ces = {}, {}
    for pg in ("all", "lt40"):
        for tau in TAU:
            gate = (margin < tau) & ((ply < 40) if pg == "lt40" else True)
            for a in ALPHA:
                x = lsm(lm + a * r * gate[:, None])
                k = f"a{a}_t{tau}_{pg}"
                hits[k] = (x.argmax(1) == truth).astype(np.float32) - hit_m
                ces[k] = -x[ar, truth] - ce_m
                res["policies"][k] = {"top1_pts": round(100 * float(hits[k].mean()), 3),
                                      "ce_delta": round(float(ces[k].mean()), 4)}
    held = []
    for A in (0, 1):
        kb = max(hits, key=lambda k: hits[k][half == A].mean())
        held.append((kb, hits[kb][half != A].sum(), ces[kb][half != A].sum()))
    ref = "a1.0_t1.01_lt40"     # = ttt200g40
    res["chosen_on_half"] = [h[0] for h in held]
    res["heldout_top1_pts"] = round(100 * sum(h[1] for h in held) / len(q), 3)
    res["heldout_ce_delta"] = round(sum(h[2] for h in held) / len(q), 4)
    res["ref_ttt200g40"] = res["policies"][ref]
    res["ref_ttt200"] = res["policies"]["a1.0_t1.01_all"]
    out[band] = res
    pool.append((hits, ces, half))
    print(band, {k: res[k] for k in ("ttt200_gain_check", "chosen_on_half", "heldout_top1_pts", "heldout_ce_delta",
                                     "ref_ttt200g40", "ref_ttt200")}, flush=True)
    for k, v in res["ce_by_bin"].items():
        print("  ", k, v)
# one policy for all bands, chosen on half A pooled, scored on B (and reverse)
keys = list(pool[0][0])
n = sum(len(p[2]) for p in pool)
tot = 0.0
totce = 0.0
ch = []
for A in (0, 1):
    kb = max(keys, key=lambda k: sum(p[0][k][p[2] == A].sum() for p in pool) / sum((p[2] == A).sum() for p in pool))
    ch.append(kb)
    tot += sum(p[0][kb][p[2] != A].sum() for p in pool)
    totce += sum(p[1][kb][p[2] != A].sum() for p in pool)
out["pooled"] = {"chosen_on_half": ch, "heldout_top1_pts": round(100 * tot / n, 3),
                 "heldout_ce_delta": round(totce / n, 4),
                 "ttt200g40_top1_pts": round(100 * sum(p[0]["a1.0_t1.01_lt40"].sum() for p in pool) / n, 3),
                 "ttt200g40_ce_delta": round(sum(p[1]["a1.0_t1.01_lt40"].sum() for p in pool) / n, 4)}
print("pooled", out["pooled"])
json.dump(out, open(sys.argv[1], "w"), indent=1)

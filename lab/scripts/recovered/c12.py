import json, numpy as np
from doppel_lab import allie_knn as K
out = {}
for b in (700, 1100, 1500, 1900, 2300):
    keys = np.load(K.knn_dir(b) / "keys.npy", mmap_mode="r")
    sides = [s for s in K.build_queries(b) if len(s[3]) > 0 and len(s[2]) > 0]
    Q = np.stack([keys[s[2]].astype(np.float32).mean(0) for s in sides])
    S = np.stack([keys[np.sort(s[3])].astype(np.float32).mean(0) for s in sides])
    r = {}
    for name, (q, st) in {"raw": (Q, S), "centred": (Q - Q.mean(0), S - S.mean(0))}.items():
        qn = q / np.linalg.norm(q, axis=1, keepdims=True); sn = st / np.linalg.norm(st, axis=1, keepdims=True)
        C = qn @ sn.T
        rank = (C > np.diag(C)[:, None]).sum(1)
        r[name] = {"id_top1": round(float((rank == 0).mean()), 4), "id_top10": round(float((rank < 10).mean()), 4),
                   "median_rank": int(np.median(rank))}
    r["n_sides"] = len(sides); r["chance_top1"] = round(1 / len(sides), 5)
    out[b] = r; print(b, r, flush=True)
json.dump(out, open("lab/results/maia_trunk_identity.json", "w"), indent=1)

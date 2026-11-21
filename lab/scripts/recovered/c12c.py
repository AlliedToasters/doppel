import json, numpy as np
from doppel_lab import allie_knn as K
out = json.load(open("lab/results/maia_trunk_identity.json"))
for b in (700, 1100, 1500, 1900, 2300):
    keys = np.load(K.knn_dir(b) / "keys.npy", mmap_mode="r"); off = np.load(K.knn_dir(b) / "off.npy")
    A, B = [], []
    for s in K.build_queries(b):
        st = np.sort(s[3])
        if len(st) == 0: continue
        g = np.searchsorted(off, st, side="right") - 1
        ug = np.unique(g)
        if len(ug) < 10: continue
        par = np.isin(g, ug[::2])
        A.append(keys[st[par]].astype(np.float32).mean(0)); B.append(keys[st[~par]].astype(np.float32).mean(0))
    A, B = np.stack(A), np.stack(B); A -= A.mean(0); B -= B.mean(0)
    A /= np.linalg.norm(A, axis=1, keepdims=True); B /= np.linalg.norm(B, axis=1, keepdims=True)
    rng = np.random.default_rng(b); acc = []
    for _ in range(200):
        ix = rng.choice(len(A), 150, replace=False)
        acc.append(((A[ix] @ B[ix].T).argmax(1) == np.arange(150)).mean())
    out[str(b)]["halves_pool150_top1"] = round(float(np.mean(acc)), 4); out[str(b)]["halves_n"] = len(A)
    print(b, len(A), out[str(b)]["halves_pool150_top1"], flush=True)
json.dump(out, open("lab/results/maia_trunk_identity.json", "w"), indent=1)

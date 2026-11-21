import json, numpy as np
from doppel_lab import allie_knn as K
out = json.load(open("lab/results/maia_trunk_identity.json"))
for b in (700, 1100, 1500, 1900, 2300):
    keys = np.load(K.knn_dir(b) / "keys.npy", mmap_mode="r"); off = np.load(K.knn_dir(b) / "off.npy")
    g_ = K.games(b).select("white_elo", "black_elo"); we, be = g_["white_elo"].to_numpy(), g_["black_elo"].to_numpy()
    A, B, EA, EB = [], [], [], []
    for s in K.build_queries(b):
        st = np.sort(s[3])
        if len(st) == 0: continue
        g = np.searchsorted(off, st, side="right") - 1
        late = (st - off[g]) >= 30
        st, g = st[late], g[late]
        if len(st) == 0: continue
        ug = np.unique(g)
        if len(ug) < 10: continue
        par = np.isin(g, ug[::2])
        A.append(keys[st[par]].astype(np.float32).mean(0)); B.append(keys[st[~par]].astype(np.float32).mean(0))
        el = np.array([we[gi] if (st[g == gi][0] - off[gi]) % 2 == 0 else be[gi] for gi in ug], float)
        EA.append(el[::2].mean()); EB.append(el[1::2].mean())
    A, B = np.stack(A), np.stack(B)
    def resid(X, e):
        Z = np.c_[np.ones(len(e)), (e - e.mean()) / e.std(), ((e - e.mean()) / e.std()) ** 2, ((e - e.mean()) / e.std()) ** 3]
        return X - Z @ np.linalg.lstsq(Z, X, rcond=None)[0]
    A, B = resid(A, np.array(EA)), resid(B, np.array(EB))
    A /= np.linalg.norm(A, axis=1, keepdims=True); B /= np.linalg.norm(B, axis=1, keepdims=True)
    rng = np.random.default_rng(b); acc = []
    for _ in range(200):
        ix = rng.choice(len(A), 150, replace=False)
        acc.append(((A[ix] @ B[ix].T).argmax(1) == np.arange(150)).mean())
    out[str(b)]["halves_elo_resid_ply30_pool150_top1"] = round(float(np.mean(acc)), 4); out[str(b)]["halves_n"] = len(A)
    print(b, len(A), out[str(b)]["halves_elo_resid_ply30_pool150_top1"], flush=True)
json.dump(out, open("lab/results/maia_trunk_identity.json", "w"), indent=1)

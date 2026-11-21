import json, numpy as np, polars as pl
from doppel_lab import allie_knn as K
out = json.load(open("lab/results/maia_trunk_identity.json"))
for b in (700, 1100, 1500, 1900, 2300):
    g_ = K.games(b).select("white_elo", "black_elo"); we, be = g_["white_elo"].to_numpy(), g_["black_elo"].to_numpy()
    off = np.load(K.knn_dir(b) / "off.npy")
    A, B = [], []
    for s in K.build_queries(b):
        st = np.sort(s[3])
        if len(st) == 0: continue
        g = np.searchsorted(off, st, side="right") - 1
        ug = np.unique(g)
        if len(ug) < 10: continue
        col = (st[0] - off[g[0]]) % 2   # the player's colour in each game varies; take elo per game by parity of ply
        elo = []
        for gi in ug:
            p0 = st[g == gi][0] - off[gi]
            elo.append(we[gi] if p0 % 2 == 0 else be[gi])
        elo = np.array(elo, float)
        A.append(elo[::2].mean()); B.append(elo[1::2].mean())
    A, B = np.array(A), np.array(B)
    rng = np.random.default_rng(b); acc = []
    for _ in range(200):
        ix = rng.choice(len(A), 150, replace=False)
        acc.append((np.abs(A[ix][:, None] - B[ix][None, :]).argmin(1) == np.arange(150)).mean())
    out[str(b)]["elo_only_halves_pool150_top1"] = round(float(np.mean(acc)), 4)
    print(b, out[str(b)]["elo_only_halves_pool150_top1"], flush=True)
json.dump(out, open("lab/results/maia_trunk_identity.json", "w"), indent=1)

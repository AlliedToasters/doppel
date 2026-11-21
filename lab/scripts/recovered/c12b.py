import json, numpy as np
from doppel_lab import allie_knn as K
out = json.load(open("lab/results/maia_trunk_identity.json"))
for b in (700, 1100, 1500, 1900, 2300):
    keys = np.load(K.knn_dir(b) / "keys.npy", mmap_mode="r")
    sides = [s for s in K.build_queries(b) if len(s[3]) > 0 and len(s[2]) > 0]
    Q = np.stack([keys[s[2]].astype(np.float32).mean(0) for s in sides]); S = np.stack([keys[np.sort(s[3])].astype(np.float32).mean(0) for s in sides])
    Q, S = Q - Q.mean(0), S - S.mean(0)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True); S /= np.linalg.norm(S, axis=1, keepdims=True)
    rng = np.random.default_rng(b); acc = []
    for _ in range(200):
        ix = rng.choice(len(sides), 150, replace=False)
        C = Q[ix] @ S[ix].T
        acc.append((C.argmax(1) == np.arange(150)).mean())
    out[str(b)]["pool150_top1"] = round(float(np.mean(acc)), 4)
    print(b, out[str(b)]["pool150_top1"], flush=True)
json.dump(out, open("lab/results/maia_trunk_identity.json", "w"), indent=1)

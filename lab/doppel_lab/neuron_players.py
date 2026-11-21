"""Players in neuron space (neuron_atlas report): each adapter as the signed mean change of the 8,192 adapted MLP
neurons' activations over the probe set (neuron_atlas mod). -> lab/results/neuron_players.json

  sparsity     per side: participation ratio of the per-neuron RMS change, share of it in the top 1% of neurons
  common       share of the per-neuron change that all sides of a band share (the band mean vector)
  style        per neuron, variance across sides after removing the band mean; concentration of that variance
  reliability  split-half bank: per-neuron correlation across players between the two halves' adapters, and top-1
               identification of the sister half in neuron space (compare adapter_split.json, function space)
  rsa          per band, Spearman between side-pair cosines in neuron space and in function space
  conformity   neurons whose change tracks function-space PC1 (the conformity axis), with their concept labels
Aggregate numbers only.
"""

from __future__ import annotations

import json

import numpy as np

from doppel_lab.adapter_space import SPLIT_TAG, fs_load, kpca, spearman
from doppel_lab.neuron_atlas import BANDS, DFF, LABEL_OUT, REPORT_OUT, mod_load


def _labels():
    if not LABEL_OUT.exists():
        return None
    return json.loads(LABEL_OUT.read_text())["neurons"]


def _desc(lab, i):
    if lab is None:
        return {"neuron": int(i)}
    x = lab[i]
    return {"neuron": f"b{x['b'] + 5}.{x['j']}", "pair": x["pair"], "f1_pair": x["f1_pair"], "tok": x["tok"],
            "tok_r": x["tok_r"], "pos": x["pos"], "pos_r": x["pos_r"], "active": x["active"]}


def _cos(X):
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return Xn @ Xn.T


def report(max_rsa: int = 1500):
    lab = _labels()
    rep = {"bands": {}}
    pooled_resid = []
    for b in BANDS:
        m = mod_load(b)
        if m is None:
            continue
        M, R = m["mean"].astype(np.float64), m["rms"].astype(np.float64)
        e = R ** 2
        pr = e.sum(1) ** 2 / (e ** 2).sum(1)
        top = np.sort(e, 1)[:, ::-1][:, : e.shape[1] // 100].sum(1) / e.sum(1)
        mu = M.mean(0)
        C = M - mu
        pooled_resid.append(C)
        var = C.var(0)
        vs = np.sort(var)[::-1]
        r = {"n_sides": int(len(M)),
             "sparsity": {"pr_quartiles": [round(float(q), 0) for q in np.quantile(pr, [0.25, 0.5, 0.75])],
                          "top1pct_share_quartiles": [round(float(q), 3) for q in np.quantile(top, [0.25, 0.5, 0.75])],
                          "by_block_share": [round(float(x), 3) for x in e.reshape(len(e), 4, DFF).sum(2).mean(0)
                                             / e.reshape(len(e), 4, DFF).sum((1, 2)).mean()]},
             "common_share": round(float((mu ** 2).sum() / (M ** 2).sum(1).mean()), 4),
             "style": {"pr": round(float(var.sum() ** 2 / (var ** 2).sum()), 1),
                       "top1pct_share": round(float(vs[: len(vs) // 100].sum() / vs.sum()), 4),
                       "top10pct_share": round(float(vs[: len(vs) // 10].sum() / vs.sum()), 4)},
             "top_common": [_desc(lab, i) | {"mean": round(float(mu[i]), 5)} for i in np.argsort(-np.abs(mu))[:10]]}
        f = fs_load(b)
        if f is not None:
            idx = {int(p): i for i, p in enumerate(m["pid"])}
            both = [(i, idx[int(p)]) for i, p in enumerate(f["pid"]) if int(p) in idx]
            fi, mi = map(np.array, zip(*both))
            sub = np.random.default_rng(b).permutation(len(fi))[:max_rsa]
            V = f["V"][fi[sub]].astype(np.float64)
            Cn = _cos(C[mi[sub]])
            Cf = _cos(V - V.mean(0))
            iu = np.triu_indices(len(sub), 1)
            r["rsa"] = {"n": int(len(sub)), "spearman": round(spearman(Cn[iu], Cf[iu]), 4)}
            G = V @ V.T
            _, _, sc, _ = kpca(G, 5)
            pc1 = sc[:, 0]
            Cs = C[mi[sub]]
            rho = np.array([np.corrcoef(pc1, Cs[:, j])[0, 1] if Cs[:, j].std() > 0 else 0 for j in range(Cs.shape[1])])
            r["conformity"] = {"max_abs_r": round(float(np.nanmax(np.abs(rho))), 4),
                               "share_abs_r_gt_0.3": round(float((np.abs(rho) > 0.3).mean()), 4),
                               "top": [_desc(lab, i) | {"r": round(float(rho[i]), 4)}
                                       for i in np.argsort(-np.nan_to_num(np.abs(rho)))[:10]]}
        rep["bands"][b] = r
        print(b, json.dumps({k: v for k, v in r.items() if k not in ("top_common", "conformity")}),
              "conformity max|r|", r.get("conformity", {}).get("max_abs_r"), flush=True)
    rep["split"] = split_reliability(lab)
    REPORT_OUT.write_text(json.dumps(rep, indent=1))
    print(f"-> {REPORT_OUT}")


def split_reliability(lab):
    """Pseudo-sides 2p / 2p+1 are the two halves of player p's training games (allie_ttt run --split)."""
    rs, A_all, B_all, ids = [], [], [], {}
    for b in BANDS:
        m = mod_load(b, SPLIT_TAG)
        if m is None:
            continue
        pid, M = m["pid"], m["mean"].astype(np.float64)
        M = M - M.mean(0)
        pair, half = pid // 2, pid % 2
        both = np.intersect1d(pair[half == 0], pair[half == 1])
        ia = np.array([np.flatnonzero((pair == q) & (half == 0))[0] for q in both])
        ib = np.array([np.flatnonzero((pair == q) & (half == 1))[0] for q in both])
        A, B = M[ia], M[ib]
        S = (A / np.linalg.norm(A, axis=1, keepdims=True)) @ (B / np.linalg.norm(B, axis=1, keepdims=True)).T
        ids[b] = {"pairs": int(len(both)), "id_top1": round(float((S.argmax(1) == np.arange(len(both))).mean()), 4),
                  "id_chance": round(1 / len(both), 4)}
        A_all.append(A)
        B_all.append(B)
    if not A_all:
        return None
    A, B = np.concatenate(A_all), np.concatenate(B_all)
    A, B = A - A.mean(0), B - B.mean(0)
    r = (A * B).mean(0) / np.maximum(A.std(0) * B.std(0), 1e-12)
    order = np.argsort(-r)
    return {"id_by_band": ids, "pairs": int(len(A)),
            "neuron_r_quartiles": [round(float(q), 3) for q in np.quantile(r, [0.25, 0.5, 0.75])],
            "share_r_gt_0.3": round(float((r > 0.3).mean()), 4), "share_r_gt_0.5": round(float((r > 0.5).mean()), 4),
            "by_block_median_r": [round(float(np.median(r[l * DFF:(l + 1) * DFF])), 3) for l in range(4)],
            "top_reliable": [_desc(lab, i) | {"r": round(float(r[i]), 4)} for i in order[:20]]}

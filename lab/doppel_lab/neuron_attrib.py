"""Which of the 8,192 adapted MLP neurons carry the differences between players (function-space distance, the metric
that kswap showed predicts transfer)?

Tier 1 (metric, CPU): learn a diagonal metric on neuron modulation (neuron_atlas mod) that predicts pairwise
  function-space squared distance:  ||V_i - V_j||^2 ~ c + sum_n w_n (M_in - M_jn)^2,  w >= 0 (positive lasso).
  Fit on pairs among half the players, tested on pairs among the other half. A neuron's share = w_n * mean dM_n^2
  over test pairs / mean prediction. Correlational, and blind to neurons whose change averages out over positions.
  -> lab/results/neuron_metric.json

Tier 2 (attrib, GPU): exact local decomposition of each adapter's effect, per neuron, plus attribution patching.
  In block l the adapter injects  Delta_l = sum_j  W2[:, j] (h'_j - hb_j)   [write: neuron fires differently]
                                              +  (B2 A2)[:, j] h'_j       [read: neuron is read differently]
  where h' = adapted activation, hb = GELU(W1 x' + b1) at the same input x' (no linear1 adapter). Projecting each
  term on g_l = dL/d(MLP output of block l) gives first-order credit per neuron and channel. The objective per
  player i is L_i = <logp_i - logp_0, w_i> with w_i chosen so that L_i = <V_i, V_i - Vbar> (function space, Fisher
  weighted, as fs_load); summed over players this is the band's total function-space variance, i.e. the mean
  pairwise squared distance / 2. So a_in sums (over n, to first order) to player i's share of the distance budget.
  -> data/allie/neuron_atlas/attrib<tag>/<band>_<chunk>.npz; report -> lab/results/neuron_attrib.json
Aggregate numbers only.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from doppel_lab.adapter_space import fs_load, probe_dir, spearman
from doppel_lab.neuron_atlas import AT, BANDS, DFF, LABEL_OUT, NB, ROOT, mod_load

METRIC_OUT = ROOT / "lab/results/neuron_metric.json"
ATTRIB_OUT = ROOT / "lab/results/neuron_attrib.json"
TAG = "_ctx200_restore"


def _labels():
    return json.loads(LABEL_OUT.read_text())["neurons"] if LABEL_OUT.exists() else None


def _name(i):
    return f"b{i // DFF + 5}.{i % DFF}"


def _desc(lab, i, **kw):
    d = {"neuron": _name(i)}
    if lab is not None:
        x = lab[i]
        d |= {"single": x["single"], "f1_single": x["f1_single"], "pair": x["pair"], "f1_pair": x["f1_pair"],
              "pos": x["pos"], "pos_r": x["pos_r"], "active": x["active"]}
    return d | {k: (round(float(v), 5) if isinstance(v, (float, np.floating)) else v) for k, v in kw.items()}


def _groups(lab, share):
    """Share by block and by best single python-chess concept (atlas label)."""
    out = {"by_block": [round(float(share[l * DFF:(l + 1) * DFF].sum()), 4) for l in range(NB)]}
    if lab is not None:
        g = {}
        for i, x in enumerate(lab):
            g.setdefault(x["single"], []).append(i)
        rows = [{"concept": k, "n": len(v), "share": round(float(share[v].sum()), 4),
                 "per_neuron_x_uniform": round(float(share[v].sum() / max(share.sum(), 1e-12) * len(share) / len(v)), 2)}
                for k, v in g.items()]
        out["by_concept"] = sorted(rows, key=lambda r: -r["share"])
    return out


def _cum(share):
    s = np.sort(np.clip(share, 0, None))[::-1]
    c = np.cumsum(s) / max(s.sum(), 1e-12)
    return {f"n_for_{int(q * 100)}pct": int(np.searchsorted(c, q) + 1) for q in (0.5, 0.8, 0.9)}


def _gram(V):
    V = V.astype(np.float32)
    return (V @ V.T).astype(np.float64)


# ------------------------------------------------------------------ Tier 1
def metric(n_fit: int = 30000, n_test: int = 20000, bands=BANDS):
    from sklearn.linear_model import Lasso
    lab = _labels()
    rep = json.loads(METRIC_OUT.read_text()) if METRIC_OUT.exists() else {"bands": {}}
    for b in bands:
        m, f = mod_load(b), fs_load(b)
        if m is None or f is None:
            print(b, "no data yet", flush=True)
            continue
        idx = {int(p): i for i, p in enumerate(m["pid"])}
        both = [(i, idx[int(p)]) for i, p in enumerate(f["pid"]) if int(p) in idx]
        fi, mi = map(np.array, zip(*both))
        G = _gram(f["V"][fi])
        sq = np.diag(G)
        feats = {"mean": m["mean"][mi].astype(np.float32),
                 "by_posband": m["mean_by_band"][mi].reshape(len(mi), len(BANDS), -1).astype(np.float32)}
        rng = np.random.default_rng(b)
        perm = rng.permutation(len(fi))
        half = {"fit": perm[: len(perm) // 2], "test": perm[len(perm) // 2:]}

        def pairs(pool, k):
            i, j = rng.choice(pool, k), rng.choice(pool, k)
            ok = i != j
            return i[ok], j[ok]

        nf = len(half["fit"]) // 4                                   # alpha on players held out of the fit
        P = {"fit": pairs(half["fit"][nf:], n_fit), "val": pairs(half["fit"][:nf], n_test // 2),
             "test": pairs(half["test"], n_test)}
        y = {s: sq[i] + sq[j] - 2 * G[i, j] for s, (i, j) in P.items()}
        r = {"n_sides": int(len(fi)), "pairs": {s: int(len(v)) for s, v in y.items()}}
        best = None
        for fname, F in feats.items():
            def X(s):                                                    # by_posband: sum the 5 squared diffs
                i, j = P[s]
                out = np.empty((len(i), F.shape[-1]), np.float32)
                for c in range(0, len(i), 2000):
                    d = F[i[c:c + 2000]] - F[j[c:c + 2000]]
                    out[c:c + 2000] = (d * d).sum(1) if d.ndim == 3 else d * d
                return out
            Xf, Xv, Xt = X("fit"), X("val"), X("test")
            cap = np.quantile(Xf, 0.995, axis=0) + 1e-12                  # winsorize: no single pair dominates
            Xf, Xv, Xt = np.minimum(Xf, cap), np.minimum(Xv, cap), np.minimum(Xt, cap)
            sd = Xf.std(0) + 1e-12
            uni = Xt.sum(1)
            fr = {"uniform_spearman": round(spearman(uni, y["test"]), 4),
                  "uniform_r2": round(float(np.corrcoef(uni, y["test"])[0, 1] ** 2), 4)}
            r2 = lambda pv, t: float(1 - ((pv - t) ** 2).mean() / t.var())
            fits = []
            for a in (1.0, 0.3, 1e-1, 3e-2, 1e-2):
                mdl = Lasso(alpha=a * y["fit"].std(), positive=True, max_iter=3000, tol=1e-3)
                mdl.fit(Xf / sd, y["fit"])
                fits.append((r2(mdl.predict(Xv / sd), y["val"]), a, mdl))
                print(b, fname, f"alpha {a}: held-out-player val r2 {fits[-1][0]:.3f}, nonzero "
                      f"{(mdl.coef_ > 0).sum()}", flush=True)
            v2, a, mdl = max(fits, key=lambda t: t[0])
            pt = mdl.predict(Xt / sd)
            w = mdl.coef_ / sd
            share = w * Xt.mean(0)
            share = share / max(share.sum(), 1e-12)
            fr |= {"alpha": a, "val_r2": round(float(v2), 4),
                   "test_r2": round(r2(pt, y["test"]), 4),
                   "test_spearman": round(spearman(pt, y["test"]), 4), "nonzero": int((w > 0).sum()),
                   "concentration": _cum(share), **_groups(lab, share),
                   "top": [_desc(lab, i, share=share[i]) for i in np.argsort(-share)[:25]]}
            r[fname] = fr
            if best is None or fr["test_r2"] > r[best]["test_r2"]:
                best = fname
            np.save(AT / f"metric_share_{b}_{fname}.npy", share.astype(np.float32))
        r["best_features"] = best
        rep["bands"][str(b)] = r
        METRIC_OUT.write_text(json.dumps(rep, indent=1))
        print(b, json.dumps({k: (v if not isinstance(v, dict) else {kk: v[kk] for kk in v if kk in
                             ("uniform_r2", "test_r2", "test_spearman", "nonzero", "concentration", "by_block")})
                             for k, v in r.items()}), flush=True)
    print(f"-> {METRIC_OUT}")


# ------------------------------------------------------------------ Tier 2 (GPU)
def attrib_dir(tag: str):
    return AT / f"attrib{tag}"


def _weights(band: int, tag: str):
    """Per side w_i over the flat legal-move vector, such that <D_i, w_i> = <V_i, V_i - Vbar> (fs_load's V)."""
    f = fs_load(band, tag=tag)
    base = np.load(probe_dir(tag) / "base.npz")
    lens = base["lens"]
    st = np.r_[0, np.cumsum(lens)[:-1]]
    p = np.exp(base["lp"]).astype(np.float32)
    sp = np.sqrt(p)
    V = f["V"]
    U = V - V.mean(0, keepdims=True)
    W = sp * U
    seg = np.repeat(np.arange(len(lens)), lens)
    W -= p * np.add.reduceat(W, st, axis=1)[:, seg]
    return f, W.astype(np.float32), (V * U).sum(1)


def attrib(tag: str, device: str, workers: int, bands, limit: int = 0, max_minutes: float = 0, bs: int = 256):
    import torch
    import torch.nn.functional as Fn

    from doppel_lab.adapter_space import probe_positions
    from doppel_lab.engines import Maia3
    from doppel_lab.lora_probe import BLOCKS, Lora
    out_dir = attrib_dir(tag) / ("smoke" if limit else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    P, sel, meta = probe_positions(workers)
    model = Maia3("79m", device).model.eval().float()
    for q in model.parameters():
        q.requires_grad_(False)
    batches = [P.batch(sel[s:s + bs], device)[:4] for s in range(0, len(sel), bs)]
    masks = [lg >= 0 for *_, lg in batches]
    offs = np.r_[0, np.cumsum([int(mk.sum()) for mk in masks])]
    layers = [model.transformer.layers[bi] for bi in BLOCKS]
    W1 = [(ly.linear1.weight, ly.linear1.bias) for ly in layers]
    W2 = [ly.linear2.weight for ly in layers]                                  # (1024, 2048)
    t0 = time.time()
    for b in bands:
        f, Wt, target = _weights(b, tag)
        row_of = {int(pid): r for r, pid in enumerate(f["pid"])}
        for fc in sorted((ROOT / f"data/allie/adapters_{b}{tag}").glob("chunk_*.npz")):
            dst = out_dir / f"{b}_{fc.stem.split('_')[1]}.npz"
            if dst.exists():
                continue
            if max_minutes and time.time() - t0 > 60 * max_minutes:
                print(f"stopping at the {max_minutes} min budget; rerun to resume", flush=True)
                return
            z = np.load(fc)
            n = len(z["pid"])
            lora = Lora(model, n, device)
            with torch.no_grad():
                for i, q in enumerate(lora.params):
                    q.copy_(torch.from_numpy(z[f"p{i}"].astype(np.float32)).to(device))
                    q.requires_grad_(False)
            st = {}

            def h1(l):
                def f_(mod, inp, out):
                    st[("x", l)] = inp[0].detach()
                return f_

            def h2(l):
                def f_(mod, inp, out):
                    st[("h", l)] = inp[0].detach()
                    if st.get("grad"):
                        zz = torch.zeros_like(out, requires_grad=True)
                        st[("z", l)] = zz
                        return out + zz
                    return out
                return f_

            hs = [ly.linear1.register_forward_hook(h1(l)) for l, ly in enumerate(layers)]
            hs += [ly.linear2.register_forward_hook(h2(l)) for l, ly in enumerate(layers)]

            def logp(x, es, eo, lg, mk):
                o = model(x, es, eo)[0]
                return o.float().gather(1, lg.clamp(min=0)).masked_fill(~mk, -1e9).log_softmax(1)[mk]

            st["grad"] = False
            with torch.no_grad():
                lora.cur = None
                lp0 = [logp(*bt, mk) for bt, mk in zip(batches, masks)]
            todo = [k for k in range(n) if z["best_ep"][k] > 0 and int(z["pid"][k]) in row_of][: limit or None]
            A = np.zeros((n, 2, NB, DFF), np.float32)                        # [write, read]
            tot = np.full((n, 2), np.nan, np.float32)                        # [sum of credit, actual L_i]
            for k in todo:
                w = torch.from_numpy(Wt[row_of[int(z["pid"][k])]]).to(device)
                acc = torch.zeros(2, NB, DFF, device=device)
                L_act = 0.0
                with torch.no_grad():                                           # B2 A2 (1024, 2048)
                    M = [lora.params[4 * l + 3][k] @ lora.params[4 * l + 2][k] for l in range(NB)]
                for (x, es, eo, lg), mk, l0, o0 in zip(batches, masks, lp0, offs):
                    st["grad"] = True
                    lora.cur = torch.full((x.shape[0],), k, device=device)
                    lp = logp(x, es, eo, lg, mk)
                    ww = w[o0:o0 + lp.shape[0]]
                    L = (lp * ww).sum()
                    gs = torch.autograd.grad(L, [st[("z", l)] for l in range(NB)])
                    st["grad"] = False
                    L_act += float(((lp.detach() - l0) * ww).sum())
                    with torch.no_grad():
                        for l in range(NB):
                            g, hp, xp = gs[l].float(), st[("h", l)], st[("x", l)]
                            hb = Fn.gelu(Fn.linear(xp, *W1[l]))
                            acc[0, l] += ((g @ W2[l]) * (hp - hb)).sum((0, 1))
                            acc[1, l] += ((g @ M[l]) * hp).sum((0, 1))
                lora.cur = None
                A[k] = acc.cpu().numpy()
                tot[k] = (A[k].sum(), L_act)
                if limit:
                    print(f"side {k}: credit {tot[k, 0]:.4f}  actual {tot[k, 1]:.4f}  probe-target "
                          f"{target[row_of[int(z['pid'][k])]]:.4f}  write/read {A[k, 0].sum():.4f}/{A[k, 1].sum():.4f}"
                          f"  {time.time() - t0:.0f}s", flush=True)
            for h in hs + lora.hooks:
                h.remove()
            tmp = dst.with_suffix(".tmp")
            with open(tmp, "wb") as fh:
                np.savez(fh, A=A, tot=tot, pid=z["pid"], best_ep=z["best_ep"])
            tmp.rename(dst)
            print(f"{b} {fc.stem}: {len(todo)} sides, {time.time() - t0:.0f}s", flush=True)
            if limit:
                print("smoke done", flush=True)
                return
    print("attrib done", flush=True)


def attrib_report(tag: str = TAG):
    lab = _labels()
    rep = {"bands": {}}
    for b in BANDS:
        fs = sorted(attrib_dir(tag).glob(f"{b}_*.npz"))
        if not fs:
            continue
        zs = [np.load(x) for x in fs]
        tot = np.concatenate([z["tot"] for z in zs])
        keep = ~np.isnan(tot[:, 0])
        A = np.concatenate([z["A"] for z in zs])[keep].astype(np.float64)    # (n, 2, 4, 2048)
        tot = tot[keep]
        budget = tot[:, 1].sum()
        S = A.sum(0) / budget                                                 # share of the distance budget
        share = S.sum(0).ravel()
        mag = np.abs(A).sum(0).sum(0).ravel()
        r = {"n_sides": int(keep.sum()),
             "completeness": {"credit_over_actual": round(float(tot[:, 0].sum() / budget), 4),
                              "per_side_r": round(float(np.corrcoef(tot[:, 0], tot[:, 1])[0, 1]), 4)},
             "channels": {"write": round(float(S[0].sum()), 4), "read": round(float(S[1].sum()), 4)},
             "by_block_channel": [[round(float(S[c, l].sum()), 4) for l in range(NB)] for c in range(2)],
             "negative_share": round(float(np.clip(share, None, 0).sum()), 4),
             "concentration": _cum(share), **_groups(lab, share),
             "top": [_desc(lab, i, share=share[i], write=S[0].ravel()[i], read=S[1].ravel()[i])
                     for i in np.argsort(-share)[:25]],
             "least": {"share_below_1e-5_abs": round(float((np.abs(share) < 1e-5).mean()), 4),
                       "bottom_half_abs_share": round(float(np.sort(np.abs(share))[: len(share) // 2].sum()), 5)}}
        mf = AT / f"metric_share_{b}_mean.npy"
        if mf.exists():
            r["spearman_vs_tier1"] = round(spearman(share, np.load(mf)), 4)
        rep["bands"][str(b)] = r
        np.save(AT / f"attrib_share_{b}.npy", np.stack([S[0].ravel(), S[1].ravel()]).astype(np.float32))
        print(b, json.dumps({k: r[k] for k in ("n_sides", "completeness", "channels", "by_block_channel",
                                               "concentration", "negative_share")}), flush=True)
    ATTRIB_OUT.write_text(json.dumps(rep, indent=1))
    print(f"-> {ATTRIB_OUT}")


def main():
    ap = argparse.ArgumentParser()
    s = ap.add_subparsers(dest="cmd", required=True)
    m = s.add_parser("metric")
    m.add_argument("--bands", type=int, nargs="+", default=list(BANDS))
    a = s.add_parser("attrib")
    a.add_argument("--tag", default=TAG)
    a.add_argument("--device", default="cuda")
    a.add_argument("--workers", type=int, default=12)
    a.add_argument("--bands", type=int, nargs="+", default=[700])
    a.add_argument("--limit", type=int, default=0)
    a.add_argument("--max-minutes", type=float, default=0)
    a.add_argument("--bs", type=int, default=256)
    a.add_argument("--gpu-ok", action="store_true")
    s.add_parser("report")
    x = ap.parse_args()
    if x.cmd == "metric":
        metric(bands=x.bands)
    elif x.cmd == "attrib":
        assert x.gpu_ok or not x.device.startswith("cuda"), "GPU work goes through the warden (--gpu-ok)"
        attrib(x.tag, x.device, x.workers, x.bands, x.limit, x.max_minutes, x.bs)
    else:
        attrib_report()


if __name__ == "__main__":
    main()

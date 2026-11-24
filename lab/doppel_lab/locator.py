"""Cold start with a locator: place a new player in function space from a few games, then borrow neighbours.

adapter_space coldstart showed that averaging the 16 nearest players' adapters gains about 60% of a player's own
gain, but finding those neighbours used the player's own 200-game adapter. Here each probed side gets a quick
adapter trained on only its most recent games (NVAL = 5 for epoch choice + `ntrain` for training, the same recipe as
allie_ttt otherwise). Its function-space vector (adapter_space.fs_load's weighting, on the same probe set) locates
the side among the other sides' full 200-game adapters (cosine on the centred vectors). The test game is then
scored with:
  - few{s}:       the quick adapter itself, B scaled by s;
  - loc{k}_x{s}:  the average of the k donors nearest to the quick adapter (the side itself excluded);
  - ora{k}_x{s}:  the k donors nearest to the side's full adapter (same metric; the upper bound);
  - mix{k}_x{s}:  the quick adapter averaged in with its k located donors (weight 1/(k+1) each).
Scale s chosen on one crc32 half of the sides, reported on the other (as adapter_space swap/coldstart).

  python -m doppel_lab.locator run --bands 700 --ntrain 5 --gpu-ok      # via the warden
  python -m doppel_lab.locator run --bands 700 --ntrain 5 --limit 40    # CPU smoke
-> data/allie/locator/<band>_n<ntrain>.npz (quick adapters, V, hits), lab/results/adapter_locator.json
"""
from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np
import polars as pl

from doppel_lab.adapter_space import NL, fs_load, load, probe_dir, probe_positions, test_positions

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "data/allie/locator"
OUT = ROOT / "lab/results/adapter_locator.json"
SCALES = (0.5, 1.0)
KS = (4, 16, 64)
c = pl.col


def unit(V: np.ndarray, mu: np.ndarray) -> np.ndarray:
    X = V.astype(np.float64) - mu
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def fs_vectors(delta: np.ndarray, lp0: np.ndarray, lens: np.ndarray) -> np.ndarray:
    """adapter_space.fs_load's weighting for raw (n, L) log-prob deltas."""
    seg = np.repeat(np.arange(len(lens)), lens)
    p = np.exp(lp0)
    st = np.r_[0, np.cumsum(lens)[:-1]]
    mean = np.add.reduceat(delta * p, st, axis=1)
    return (delta - mean[:, seg]) * np.sqrt(p)


def train_quick(model, rows: pl.DataFrame, n: int, device: str, workers: int, epochs=3, lr=1e-3, wd=0.01, bs=512,
                chunk=400):
    """allie_ttt's per-side recipe on `rows` (pid 0..n-1, roles val/train). Returns A, B lists (numpy, per layer)
    and best epochs."""
    import torch

    from doppel_lab.lora_probe import Lora, Positions, encode_all, logits, per_position
    As = [[] for _ in range(NL)]
    Bs = [[] for _ in range(NL)]
    eps = []
    for c0 in range(0, n, chunk):
        m = min(n, c0 + chunk) - c0
        d = rows.filter((c("pid") >= c0) & (c("pid") < c0 + m)).with_columns(pid=c("pid") - c0)
        P = Positions(d, encode_all(d, workers))
        lora = Lora(model, m, device)
        opt = torch.optim.AdamW(lora.params, lr=lr, weight_decay=wd)
        tr, va = np.flatnonzero(P.role == "train"), np.flatnonzero(P.role == "val")
        own = lambda idx: P.pid[idx]
        lp0, _ = per_position(model, lora, P, va, None, device, torch.float16)
        nv = np.bincount(P.pid[va], minlength=m)
        cnt = np.maximum(nv, 1)
        best_ce = np.bincount(P.pid[va], -lp0, m) / cnt
        best, best_ep = lora.snapshot(), np.zeros(m, np.int64)
        rng = np.random.default_rng(c0)
        for ep in range(1, epochs + 1):
            perm = rng.permutation(tr)
            for s in range(0, len(perm), bs):
                idx = perm[s:s + bs]
                b = P.batch(idx, device)
                lp = logits(model, b, lora, torch.from_numpy(P.pid[idx]).to(device), torch.bfloat16)
                loss = -lp.gather(1, b[4][:, None]).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            lpv, _ = per_position(model, lora, P, va, own, device, torch.float16)
            ce = np.bincount(P.pid[va], -lpv, m) / cnt
            better = (ce < best_ce) & (nv > 0)
            mk = torch.from_numpy(better).to(device)
            for b_, s_ in zip(best, lora.snapshot()):
                b_[mk] = s_[mk]
            best_ce, best_ep = np.where(better, ce, best_ce), np.where(better, ep, best_ep)
        lora.load_rows(best, np.ones(m, bool))
        for h in lora.hooks:
            h.remove()
        for l in range(NL):
            As[l].append(lora.params[2 * l].detach().float().cpu().numpy())
            Bs[l].append(lora.params[2 * l + 1].detach().float().cpu().numpy())
        eps.append(best_ep)
        del lora, opt, P
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"  quick adapters {c0 + m}/{n}, epochs {np.bincount(best_ep, minlength=epochs + 1).tolist()}", flush=True)
    return [np.concatenate(a) for a in As], [np.concatenate(b) for b in Bs], np.concatenate(eps)


def probe_deltas(model, A, B, device: str, workers: int):
    """Per side: log p(adapter) - log p(Maia-3) over the legal moves of every probe position (adapter_space.probe)."""
    import torch

    from doppel_lab.lora_probe import Lora
    P, sel, _ = probe_positions(workers)
    x, es, eo, lg, _ = P.batch(sel, device)
    mask = lg >= 0
    n = len(A[0])
    lora = Lora(model, n, device)
    with torch.no_grad():
        for l in range(NL):
            lora.params[2 * l].copy_(torch.from_numpy(A[l]).to(device))
            lora.params[2 * l + 1].copy_(torch.from_numpy(B[l]).to(device))

    def lp_of(pid):
        out = []
        with torch.no_grad():
            for s in range(0, len(sel), 1280):
                sl = slice(s, s + 1280)
                lora.cur = None if pid is None else torch.full((x[sl].shape[0],), pid, device=device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                    o = model(x[sl], es[sl], eo[sl])[0]
                lora.cur = None
                out.append(o.float().gather(1, lg[sl].clamp(min=0)).masked_fill(~mask[sl], -1e9).log_softmax(1))
        return torch.cat(out)[mask]

    base = lp_of(None)
    D = np.zeros((n, int(mask.sum())), np.float32)
    for k in range(n):
        D[k] = (lp_of(k) - base).cpu().numpy()
    for h in lora.hooks:
        h.remove()
    return D, base.cpu().numpy(), mask.sum(1).cpu().numpy()


def score_avg(model, P, te, tp, A_pool, B_pool, nbr, wts, device):
    """Hits per test position for each scale: side i scored with sum_j wts[i, j] * dW(pool[nbr[i, j]])
    (exact, as concatenated LoRAs). A_pool/B_pool: per-layer torch tensors (pool, r, in) / (pool, out, r)."""
    import torch

    from doppel_lab.lora_probe import Lora, per_position
    n, k = nbr.shape
    r0 = A_pool[0].shape[1]
    H = {s: np.zeros(len(te), bool) for s in SCALES}
    m = max(1, int(3e9 / (k * r0 * 24576 * 4)))
    for s0 in range(0, n, m):
        e = min(n, s0 + m)
        lo = Lora(model, e - s0, device, r=r0 * k)
        nb = torch.from_numpy(nbr[s0:e].copy())
        w = torch.from_numpy(wts[s0:e].astype(np.float32))[:, None, :, None]       # (m, 1, k, 1)
        with torch.no_grad():
            for l in range(NL):
                lo.params[2 * l].copy_(A_pool[l][nb].reshape(e - s0, r0 * k, -1).to(device))
                Bn = B_pool[l][nb].permute(0, 2, 1, 3) * w                            # (m, out, k, r)
                lo.params[2 * l + 1].copy_(Bn.reshape(e - s0, Bn.shape[1], r0 * k).to(device))
        sm = (tp >= s0) & (tp < e)
        prev = 1.0
        for s in SCALES:
            with torch.no_grad():
                for p in lo.params[1::2]:
                    p.mul_(s / prev)
            prev = s
            _, h = per_position(model, lo, P, te[sm], lambda ix: P.pid[ix] - s0, device, torch.float16,
                                bs=max(64, min(1024, 2 ** 16 // k)))
            H[s][sm] = h
        for hk in lo.hooks:
            hk.remove()
        del lo
    return H


def run(bands: list[int], ntrain: int, device: str, workers: int, limit: int = 0):
    import torch

    from doppel_lab import allie_ttt
    from doppel_lab.engines import Maia3
    from doppel_lab.lora_probe import per_position
    allie_ttt.use_store200()
    model = Maia3("79m", device).model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    DIR.mkdir(parents=True, exist_ok=True)
    out = OUT.with_name("adapter_locator_smoke.json") if limit else OUT
    rep = json.loads(out.read_text()) if out.exists() else {}
    t0 = time.time()
    for band in bands:
        dst = DIR / (f"{band}_n{ntrain}" + ("_smoke" if limit else "") + ".npz")
        if dst.exists():
            print(band, "done already", flush=True)
            continue
        f = fs_load(band)
        a = load(band)
        idx = {int(p): i for i, p in enumerate(a["pid"])}
        rows_ = np.array([idx[int(p)] for p in f["pid"]])
        if limit:
            rows_, f = rows_[:limit], {k: v[:limit] for k, v in f.items()}
        n = len(rows_)
        d = {k: v[rows_] for k, v in a.items() if k not in ("A", "B")}
        A_full = [torch.from_numpy(a["A"][l][rows_]) for l in range(NL)]
        B_full = [torch.from_numpy(a["B"][l][rows_]) for l in range(NL)]
        # quick adapters: the side's NVAL most recent context games for epoch choice + the ntrain before them
        sr, _ = allie_ttt.side_rows(band)
        mp = pl.DataFrame({"pid": d["pid"].astype(np.int64), "new": np.arange(n, dtype=np.int64)})
        q = (sr.filter((c("role") == "val") | ((c("role") == "train") & (c("tr_rank") < ntrain)))
             .join(mp, on="pid").with_columns(pid=c("new")).drop("new").sort("pid", "j"))
        ngames = q.group_by("pid").agg(pl.len()).sort("pid")
        print(band, f"{n} sides, {q.height} quick-context games (median per side "
                    f"{int(ngames['len'].median())}), {time.time() - t0:.0f}s", flush=True)
        Aq, Bq, ep = train_quick(model, q, n, device, workers)
        Dq, lp0, lens = probe_deltas(model, Aq, Bq, device, workers)
        Vq = fs_vectors(Dq, lp0, lens)
        Vq[ep == 0] = 0.0                                   # epoch 0 = Maia-3 exactly
        mu = f["V"].astype(np.float64).mean(0)
        Xf, Xq = unit(f["V"], mu), unit(Vq, mu)
        Sq, So = Xq @ Xf.T, Xf @ Xf.T
        np.fill_diagonal(Sq, -np.inf)
        np.fill_diagonal(So, -np.inf)
        nq, no = np.argsort(-Sq, 1), np.argsort(-So, 1)
        print(band, f"located, {time.time() - t0:.0f}s", flush=True)
        P, te = test_positions(d, band, workers)
        tp = P.pid[te]
        H = {}
        from doppel_lab.lora_probe import Lora
        m0 = Lora(model, 1, device)
        _, H["maia"] = per_position(model, m0, P, te, None, device, torch.float16)
        for hk in m0.hooks:
            hk.remove()
        Aq_t, Bq_t = [torch.from_numpy(x) for x in Aq], [torch.from_numpy(x) for x in Bq]
        one = np.arange(n)[:, None]
        for s, h in score_avg(model, P, te, tp, Aq_t, Bq_t, one, np.ones((n, 1)), device).items():
            H[f"few_x{s}"] = h
        for k in KS:
            k = min(k, n - 1)
            for name, nbr in (("loc", nq), ("ora", no)):
                for s, h in score_avg(model, P, te, tp, A_full, B_full, nbr[:, :k], np.full((n, k), 1 / k),
                                      device).items():
                    H[f"{name}{k}_x{s}"] = h
            # mix: quick adapter + its k located donors, equal weights; pool = full adapters then quick ones
            pool_A = [torch.cat([A_full[l], Aq_t[l]]) for l in range(NL)]
            pool_B = [torch.cat([B_full[l], Bq_t[l]]) for l in range(NL)]
            nb = np.concatenate([n + np.arange(n)[:, None], nq[:, :k]], 1)
            for s, h in score_avg(model, P, te, tp, pool_A, pool_B, nb, np.full((n, k + 1), 1 / (k + 1)),
                                  device).items():
                H[f"mix{k}_x{s}"] = h
            print(band, f"k={k} scored, {time.time() - t0:.0f}s", flush=True)
        ov = {k: float(np.mean([len(set(nq[i, :k]) & set(no[i, :k])) / k for i in range(n)])) for k in KS if k < n}
        half = np.array([zlib.crc32(g.encode()) & 1 for g in d["test_game_id"]])[tp]
        keys = list(H)
        tmp = dst.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            np.savez(fh, keys=np.array(keys), H=np.array([H[k] for k in keys]), pid=tp, half=half, best_ep=ep,
                     Vq=Vq.astype(np.float16), cos_self=np.einsum("ij,ij->i", Xq, Xf),
                     overlap=json.dumps(ov), ngames=ngames["len"].to_numpy(),
                     **{f"A{l}": Aq[l].astype(np.float16) for l in range(NL)},
                     **{f"B{l}": Bq[l].astype(np.float16) for l in range(NL)})
        tmp.rename(dst)
        print(band, f"saved {dst.name}, {time.time() - t0:.0f}s", flush=True)
    report(out, limit)


def report(out: Path = OUT, limit: int = 0):
    rep = {}
    for f in sorted(DIR.glob("*.npz")):
        if ("smoke" in f.stem) != bool(limit):
            continue
        z = np.load(f)
        keys, H, half = list(z["keys"]), z["H"].astype(np.float64), z["half"]
        i0 = keys.index("maia")
        base = H[i0]
        gain = {k: round(100 * (H[i].mean() - base.mean()), 3) for i, k in enumerate(keys) if k != "maia"}
        cv = {}
        for name in sorted({k.rsplit("_x", 1)[0] for k in keys if "_x" in k}):
            cand = [keys.index(f"{name}_x{s}") for s in SCALES if f"{name}_x{s}" in keys]
            g = []
            for tr in (0, 1):
                best = max(cand, key=lambda i: H[i][half == tr].mean())
                g.append(100 * (H[best][half != tr].mean() - base[half != tr].mean()))
            cv[name] = round(float(np.mean(g)), 3)
        rep[f.stem.replace("_smoke", "")] = {
            "n_pos": int(H.shape[1]), "maia_acc": round(100 * base.mean(), 2),
            "median_quick_games": int(np.median(z["ngames"])), "quick_epoch0_share": round(float((z["best_ep"] == 0).mean()), 3),
            "cos_quick_vs_full_median": round(float(np.median(z["cos_self"])), 3),
            "neighbour_overlap_loc_vs_oracle": {k: round(v, 3) for k, v in json.loads(str(z["overlap"])).items()},
            "gain_pts": gain, "cv_best_scale_pts": cv}
        print(f.stem, json.dumps(cv), flush=True)
    out.write_text(json.dumps(rep, indent=1))
    print(f"-> {out}")


def main():
    ap = argparse.ArgumentParser()
    s = ap.add_subparsers(dest="cmd", required=True)
    r = s.add_parser("run")
    r.add_argument("--bands", nargs="+", type=int, default=[700, 1100, 1500, 1900, 2300])
    r.add_argument("--ntrain", type=int, default=5)
    r.add_argument("--workers", type=int, default=12)
    r.add_argument("--gpu-ok", action="store_true")
    r.add_argument("--limit", type=int, default=0)
    s.add_parser("report")
    a = ap.parse_args()
    if a.cmd == "report":
        report()
        return
    if not a.gpu_ok and not a.limit:
        raise SystemExit("refusing to touch the GPU without --gpu-ok (run through the warden)")
    run(a.bands, a.ntrain, "cuda" if a.gpu_ok else "cpu", a.workers, a.limit)


if __name__ == "__main__":
    main()

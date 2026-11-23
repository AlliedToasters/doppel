"""Geometry of the per-player adapter space (backlog 11 geometric count, 11f step 1, 12a engine axis). CPU.

    python -m doppel_lab.adapter_space geom [--tag _ctx200_restore]   -> lab/results/adapter_space.json

Adapters: the restored ctx200 bank (allie_ttt restore; data/allie/adapters_<band><tag>/chunk_*.npz), rank-8 LoRA on
linear1/linear2 of Maia-3 blocks 5-8. A side's adapter is the weight change dW = B @ A per layer (8 layers); the
inner product of two sides = sum over layers of <dW_i, dW_j>_F = sum((B_i^T B_j) * (A_i A_j^T)), so the exact Gram
over all sides needs no dense dW. Epoch-0 sides (B = 0, the adapter is Maia-3) are excluded from the geometry.
Per band and pooled across bands: kernel PCA of the cosine Gram (centred), eigen-spectrum shares, participation ratio
(effective dimension), and Spearman correlations of the top PCs with the side's Elo, its own-adapter gain on its test
game, log norm, chosen epoch and val CE. 12a asks whether PC1 is a rating / engine-likeness axis. Aggregate only.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
BANDS = (700, 1100, 1500, 1900, 2300)
OUT = ROOT / "lab/results/adapter_space.json"
NL = 8
c = pl.col


def load(band: int, tag: str = "_ctx200_restore") -> dict | None:
    fs = sorted((ROOT / f"data/allie/adapters_{band}{tag}").glob("chunk_*.npz"))
    if not fs:
        return None
    zs = [np.load(f) for f in fs]
    cat = lambda k: np.concatenate([z[k] for z in zs])
    return {"A": [cat(f"p{2 * l}").astype(np.float32) for l in range(NL)],
            "B": [cat(f"p{2 * l + 1}").astype(np.float32) for l in range(NL)],
            **{k: cat(k) for k in ("pid", "test_game_id", "test_col", "best_ep", "val_ce")},
            "band": np.full(sum(len(z["pid"]) for z in zs), band)}


def merge(ds: list[dict]) -> dict:
    out = {k: [np.concatenate([d[k][l] for d in ds]) for l in range(NL)] for k in ("A", "B")}
    for k in ds[0]:
        if k not in ("A", "B"):
            out[k] = np.concatenate([d[k] for d in ds])
    return out


def gram(d: dict, blk: int = 256) -> np.ndarray:
    """G[i, j] = sum_l <B_i A_i, B_j A_j>_F."""
    import torch
    torch.set_num_threads(16)
    n = len(d["pid"])
    G = torch.zeros(n, n, dtype=torch.float64)
    for l in range(NL):
        A = torch.from_numpy(d["A"][l])                    # (n, r, in)
        B = torch.from_numpy(d["B"][l])                    # (n, out, r)
        r = A.shape[1]
        Af = A.reshape(n * r, -1)                          # rows (i, a)
        Bf = B.transpose(1, 2).reshape(n * r, -1)          # rows (i, a) = column a of B_i
        for s in range(0, n, blk):
            e = min(n, s + blk)
            aa = (Af[s * r:e * r] @ Af.T).reshape(e - s, r, n, r)
            bb = (Bf[s * r:e * r] @ Bf.T).reshape(e - s, r, n, r)
            G[s:e] += (aa * bb).sum((1, 3)).double()
    return G.numpy()


def side_meta(d: dict) -> pl.DataFrame:
    """Elo of the side and its own-adapter gain on its test game (allie_ttt ctx200 positions)."""
    from doppel_lab.allie_v0e import TEST
    t = pl.read_parquet(TEST, columns=["game_id", "white_elo", "black_elo"])
    s = pl.DataFrame({"i": np.arange(len(d["pid"])), "game_id": d["test_game_id"], "col": d["test_col"].astype(np.int64),
                      "band": d["band"]})
    s = s.join(t, on="game_id", how="left").with_columns(
        elo=pl.when(c("col") == 0).then(c("white_elo")).otherwise(c("black_elo"))).drop("white_elo", "black_elo")
    gs = []
    for b in np.unique(d["band"]):
        z = np.load(ROOT / f"data/allie/ttt_{b}_ctx200.npz")
        p = pl.DataFrame({"game_id": z["game_id"], "col": z["ply"] % 2, "own": z["own"], "maia": z["maia"],
                          "ctrl": z["ctrl"]})
        gs.append(p.group_by("game_id", "col").agg(n_pos=pl.len(), own_gain=(c("own").cast(pl.Float64) - c("maia")).mean(),
                                                   swap_gain=(c("ctrl").cast(pl.Float64) - c("maia")).mean()))
    g = pl.concat(gs).unique(["game_id", "col"])
    return s.join(g, on=["game_id", "col"], how="left").sort("i")


def spearman(x, y) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        return float("nan")
    rx, ry = np.argsort(np.argsort(x[ok])), np.argsort(np.argsort(y[ok]))
    return float(np.corrcoef(rx, ry)[0, 1])


def kpca(G: np.ndarray, k: int = 10):
    nrm = np.sqrt(np.diag(G))
    C = G / np.outer(nrm, nrm)
    n = len(C)
    H = np.eye(n) - 1.0 / n
    Cc = H @ C @ H
    w, v = np.linalg.eigh(Cc)
    w, v = w[::-1], v[:, ::-1]
    w = np.clip(w, 0, None)
    share = w / w.sum()
    pr = float(w.sum() ** 2 / (w ** 2).sum())
    scores = v[:, :k] * np.sqrt(w[:k])
    return share, pr, scores, C


def report(G: np.ndarray, meta: pl.DataFrame, d: dict, k: int = 5) -> dict:
    share, pr, sc, C = kpca(G, 10)
    nrm = np.sqrt(np.diag(G))
    off = C[~np.eye(len(C), dtype=bool)]
    cov = {"elo": meta["elo"].to_numpy().astype(float), "own_gain": meta["own_gain"].to_numpy().astype(float),
           "log_norm": np.log(nrm), "best_ep": d["best_ep"].astype(float), "val_ce": d["val_ce"].astype(float),
           "colour": d["test_col"].astype(float), "band": d["band"].astype(float)}
    return {"n": int(len(G)), "participation_ratio": round(pr, 2),
            "pc_share": [round(float(x), 4) for x in share[:10]],
            "cos_mean": round(float(off.mean()), 4), "cos_p90": round(float(np.percentile(off, 90)), 4),
            "pc_spearman": {f"pc{j + 1}": {kk: round(spearman(sc[:, j], vv), 3) for kk, vv in cov.items()}
                            for j in range(k)},
            "norm_spearman": {kk: round(spearman(np.log(nrm), vv), 3) for kk, vv in cov.items() if kk != "log_norm"}}


def sub(d: dict, m: np.ndarray) -> dict:
    return {k: ([x[m] for x in v] if k in ("A", "B") else v[m]) for k, v in d.items()}


def geom(tag: str):
    ds = [x for x in (load(b, tag) for b in BANDS) if x is not None]
    d = merge(ds)
    keep = d["best_ep"] > 0
    print(f"{len(keep):,} sides, {keep.sum():,} with an adapter (epoch > 0)", flush=True)
    d = sub(d, keep)
    G = gram(d)
    meta = side_meta(d)
    rep = {"tag": tag, "epoch0_share": round(float(1 - keep.mean()), 4), "pooled": report(G, meta, d), "bands": {}}
    print("pooled", json.dumps(rep["pooled"]), flush=True)
    for b in np.unique(d["band"]):
        m = d["band"] == b
        r = report(G[np.ix_(m, m)], meta.filter(pl.Series(m)), sub(d, m))
        rep["bands"][int(b)] = r
        print(b, json.dumps({k: r[k] for k in ("n", "participation_ratio", "pc_share", "cos_mean")}),
              {j: r["pc_spearman"][j] for j in ("pc1", "pc2")}, flush=True)
    OUT.write_text(json.dumps(rep, indent=1))
    np.savez(ROOT / f"data/allie/adapter_gram{tag}.npz", G=G, band=d["band"], pid=d["pid"])
    print(f"-> {OUT}")


PROBE = ROOT / "data/allie/adapter_probe"


def probe_dir(tag: str = "_ctx200_restore") -> Path:
    """The restore bank probes into adapter_probe/, any other bank into adapter_probe<tag>/ (same probe set)."""
    return PROBE if tag == "_ctx200_restore" else ROOT / f"data/allie/adapter_probe{tag}"
PROBE_GAMES, PROBE_PER_BAND = 96, 512


def probe_positions(workers: int = 12):
    """The fixed probe set: PROBE_PER_BAND own-move positions (ply >= 10) per band from PROBE_GAMES Allie test games
    per band (seeded), both colours. Returns (Positions, selected index, meta frame)."""
    from doppel_lab.allie_v0e import test_games
    from doppel_lab.lora_probe import Positions, encode_all
    ds, metas = [], []
    for b in BANDS:
        t = test_games(b).select("game_id", "moves", "white_elo", "black_elo").sort("game_id")
        t = t.sample(PROBE_GAMES, seed=b)
        ds.append(pl.concat([t.with_columns(col=pl.lit(k, pl.Int64)) for k in (0, 1)]).with_columns(band=pl.lit(b)))
    d = pl.concat(ds).with_columns(pid=pl.lit(0, pl.Int64), role=pl.lit("probe"), tr_rank=pl.lit(-1, pl.Int64))
    P = Positions(d, encode_all(d, workers))
    band_of = d["band"].to_numpy()[P.game]
    sel = []
    for b in BANDS:
        cand = np.flatnonzero((band_of == b) & (P.ply >= 10))
        sel.append(np.sort(np.random.default_rng(b).choice(cand, PROBE_PER_BAND, replace=False)))
    sel = np.concatenate(sel)
    meta = pl.DataFrame({"band": band_of[sel], "game_id": d["game_id"].to_numpy()[P.game[sel]], "ply": P.ply[sel]})
    return P, sel, meta


def probe(tag: str, device: str, workers: int, limit: int = 0):
    """GPU: per restored side, log p(adapter) - log p(Maia-3) over the legal moves of every probe position (raw
    adapter, residual weight 1). -> data/allie/adapter_probe/<band>_<c0>.npz (delta fp16 (n, L)); base.npz holds the
    probe meta, Maia-3 log-probs and legal counts. Chunks already probed are skipped."""
    import torch

    from doppel_lab.engines import Maia3
    from doppel_lab.lora_probe import Lora
    out_dir = PROBE / "smoke" if limit else probe_dir(tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    P, sel, meta = probe_positions(workers)
    model = Maia3("79m", device).model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    x, es, eo, lg, _ = P.batch(sel, device)
    mask = lg >= 0
    lens = mask.sum(1).cpu().numpy()

    def lp_of(lora, pid):
        out = []
        with torch.no_grad():
            for s in range(0, len(sel), 1280):
                sl = slice(s, s + 1280)
                lora.cur = None if pid is None else torch.full((x[sl].shape[0],), pid, device=device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                    o = model(x[sl], es[sl], eo[sl])[0]
                lora.cur = None
                z = o.float().gather(1, lg[sl].clamp(min=0)).masked_fill(~mask[sl], -1e9).log_softmax(1)
                out.append(z)
        return torch.cat(out)[mask]

    base = None
    t0 = time.time()
    for b in BANDS:
        for f in sorted((ROOT / f"data/allie/adapters_{b}{tag}").glob("chunk_*.npz")):
            dst = out_dir / f"{b}_{f.stem.split('_')[1]}.npz"
            if dst.exists():
                continue
            z = np.load(f)
            n = len(z["pid"])
            lora = Lora(model, n, device)
            with torch.no_grad():
                for i, p in enumerate(lora.params):
                    p.copy_(torch.from_numpy(z[f"p{i}"].astype(np.float32)).to(device))
            if base is None:
                base = lp_of(lora, None)
                np.savez(out_dir / "base.npz", lp=base.cpu().numpy().astype(np.float32), lens=lens,
                         band=meta["band"].to_numpy(), game_id=meta["game_id"].to_numpy().astype(str),
                         ply=meta["ply"].to_numpy())
            delta = np.zeros((n, int(mask.sum())), np.float16)
            for k in range(min(n, limit) if limit else n):
                if z["best_ep"][k] > 0:
                    delta[k] = (lp_of(lora, k) - base).cpu().numpy().astype(np.float16)
            for h in lora.hooks:
                h.remove()
            tmp = dst.with_suffix(".tmp")
            with open(tmp, "wb") as fh:
                np.savez(fh, delta=delta, pid=z["pid"], best_ep=z["best_ep"], test_game_id=z["test_game_id"],
                         test_col=z["test_col"], val_ce=z["val_ce"])
            tmp.rename(dst)
            print(f"{b} {f.stem}: {n} sides, {time.time() - t0:.0f}s", flush=True)
            if limit:
                print("smoke done", flush=True)
                return
    print("probe done", flush=True)

SWAP_OUT = ROOT / "lab/results/adapter_swap.json"
FS_OUT = ROOT / "lab/results/adapter_fs.json"
KS = (1, 2, 4, 8, 16, 32, 64)
RANKS = (1, 2, 5, 10, 50)


def fs_load(band: int, own_band_only: bool = False, tag: str = "_ctx200_restore", plies: tuple | None = None) -> dict | None:
    """Function-space vectors of a band's probed sides (epoch > 0): per probe position, the adapter's change in
    log-prob centred under Maia-3's policy and weighted by sqrt(p), so <v_i, v_j> ~ the Fisher inner product (and
    |v|^2 / 2 ~ KL(Maia-3 || adapter) for small changes)."""
    fs = sorted(probe_dir(tag).glob(f"{band}_*.npz"))
    if not fs:
        return None
    base = np.load(probe_dir(tag) / "base.npz")
    lp0, lens = base["lp"], base["lens"]
    seg = np.repeat(np.arange(len(lens)), lens)
    p = np.exp(lp0)
    zs = [np.load(f) for f in fs]
    cat = lambda k: np.concatenate([z[k] for z in zs])
    D = cat("delta").astype(np.float32)
    keep = cat("best_ep") > 0
    D = D[keep]
    mean = np.zeros((len(D), len(lens)), np.float32)
    for j0 in range(0, len(D), 256):                         # per-position p-weighted mean of the change
        blk = D[j0:j0 + 256] * p
        mean[j0:j0 + 256] = np.add.reduceat(blk, np.r_[0, np.cumsum(lens)[:-1]], axis=1)
    V = (D - mean[:, seg]) * np.sqrt(p)
    if own_band_only:
        V = V[:, base["band"][seg] == band]
    if plies is not None:                                   # only probe positions with lo <= ply < hi
        pl_ = base["ply"][seg]
        V = V[:, (pl_ >= plies[0]) & (pl_ < plies[1])]
    top = np.zeros(len(lens), np.int64)                      # Maia-3's argmax column per position (flat index)
    st = np.r_[0, np.cumsum(lens)[:-1]]
    for j, (a, n) in enumerate(zip(st, lens)):
        top[j] = a + np.argmax(lp0[a:a + n])
    return {"V": V, "pid": cat("pid")[keep], "best_ep": cat("best_ep")[keep], "val_ce": cat("val_ce")[keep],
            "test_game_id": cat("test_game_id")[keep], "test_col": cat("test_col")[keep],
            "band": np.full(int(keep.sum()), band), "sharpen": D[:, top].mean(1),
            "kl": 0.5 * (V ** 2).sum(1) / len(lens)}


def fs_geom(own_band_only: bool = False):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    ds = [x for x in (fs_load(b, own_band_only) for b in BANDS) if x is not None]
    rep = {"own_band_only": own_band_only, "bands": {}}
    for d in ds + ([{k: np.concatenate([x[k] for x in ds]) for k in ds[0]}] if len(ds) > 1 and not own_band_only else []):
        name = int(d["band"][0]) if len(np.unique(d["band"])) == 1 else "pooled"
        G = (d["V"].astype(np.float64) @ d["V"].T.astype(np.float64))
        meta = side_meta(d)
        r = report(G, meta, d)
        share, pr, sc, C = kpca(G, 64)
        r["pc_spearman_extra"] = {f"pc{j + 1}": {"sharpen": round(spearman(sc[:, j], d["sharpen"]), 3),
                                                 "kl": round(spearman(sc[:, j], d["kl"]), 3)} for j in range(5)}
        r["kl_spearman"] = {k: round(spearman(d["kl"], v), 3) for k, v in
                            (("elo", meta["elo"].to_numpy().astype(float)),
                             ("own_gain", meta["own_gain"].to_numpy().astype(float)),
                             ("best_ep", d["best_ep"].astype(float)))}
        r["sharpen_mean"] = round(float(d["sharpen"].mean()), 4)
        r["sharpen_spearman_elo"] = round(spearman(d["sharpen"], meta["elo"].to_numpy().astype(float)), 3)
        r["cumshare_64"] = [round(float(x), 4) for x in np.cumsum(share[:64])[[0, 4, 9, 19, 49, 63]]]
        if name != "pooled":
            X = sc / np.linalg.norm(sc, axis=1, keepdims=True)
            r["silhouette"] = {k: round(float(silhouette_score(X, KMeans(k, n_init=10, random_state=0).fit_predict(X))), 4)
                               for k in (2, 4, 8, 16) if len(X) > 4 * k}
        rep["bands"][name] = r
        print(name, json.dumps({k: r[k] for k in ("n", "participation_ratio", "cos_mean", "cumshare_64",
                                                  "sharpen_mean", "sharpen_spearman_elo", "kl_spearman")}),
              "pc1", r["pc_spearman"]["pc1"], r["pc_spearman_extra"]["pc1"], r.get("silhouette"), flush=True)
    out = FS_OUT.with_name(FS_OUT.stem + ("_ownband" if own_band_only else "") + ".json")
    out.write_text(json.dumps(rep, indent=1))
    print(f"-> {out}")


def test_positions(d: dict, band: int, workers: int):
    """Positions of each side's test game (pid = side row), scored on the Allie protocol (ply >= 10, before cut)."""
    from doppel_lab.allie_v0e import test_games
    from doppel_lab.lora_probe import Positions, encode_all
    t = test_games(band).select("game_id", "moves", "white_elo", "black_elo", "cut")
    s = pl.DataFrame({"pid": np.arange(len(d["pid"]), dtype=np.int64), "game_id": d["test_game_id"],
                      "col": d["test_col"].astype(np.int64)})
    rows = s.join(t, on="game_id", how="left").sort("pid").with_columns(role=pl.lit("test"),
                                                                       tr_rank=pl.lit(-1, pl.Int64))
    P = Positions(rows, encode_all(rows, workers))
    cut = rows["cut"].to_numpy()
    te = np.flatnonzero((P.ply >= 10) & (P.ply < cut[P.game]))
    return P, te


def swap(tag: str, device: str, workers: int, bands: list[int], limit: int = 0):
    """GPU: score each probed side's test game with (own, the r-th most similar side in function space for r in
    RANKS, a random side, the cluster 'soup' for k in KS at scales 1/2/4). Soup = mean of the members' dW, as a
    LoRA of rank <= 256 (SVD). Clusters: k-means on the band's function-space kernel PCs (64, unit-normalised).
    Cross-fit: sides are split in two by crc32 of the test game; clusters and soups come from one half and score
    the other half (each side assigned to the nearest centroid by its own function-space vector, which depends on its
    context games only), so no side's own adapter is in its soup. Scale is chosen on one half, reported on the other."""
    import zlib

    import torch
    from sklearn.cluster import KMeans

    from doppel_lab.engines import Maia3
    from doppel_lab.lora_probe import Lora, per_position
    model = Maia3("79m", device).model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    out = SWAP_OUT.with_name("adapter_swap_smoke.json") if limit else SWAP_OUT
    rep = json.loads(out.read_text()) if out.exists() else {}
    for band in bands:
        f = fs_load(band)
        a = load(band, tag)
        idx = {int(p): i for i, p in enumerate(a["pid"])}
        rows = np.array([idx[int(p)] for p in f["pid"]])
        if limit:
            rows, f = rows[:limit], {k: v[:limit] for k, v in f.items()}
        d = {k: v[rows] for k, v in a.items() if k not in ("A", "B")}
        A = [x[rows] for x in a["A"]]
        B = [x[rows] for x in a["B"]]
        n = len(rows)
        G = f["V"].astype(np.float64) @ f["V"].T.astype(np.float64)
        nrm = np.sqrt(np.diag(G))
        C = G / np.outer(nrm, nrm)
        np.fill_diagonal(C, -np.inf)
        order = np.argsort(-C, 1)
        _, _, sc, _ = kpca(G, 64)
        X = sc / np.linalg.norm(sc, axis=1, keepdims=True)
        rng = np.random.default_rng(band)
        rnd = (np.arange(n) + rng.integers(1, n, n)) % n
        P, te = test_positions(d, band, workers)
        half = np.array([zlib.crc32(g.encode()) & 1 for g in d["test_game_id"]])
        own_pid = P.pid
        res = {"n_sides": n, "n_pos": int(len(te))}

        def hits(lora, mapping):
            _, h = per_position(model, lora, P, te, None if mapping is None else (lambda ix: mapping[own_pid[ix]]),
                                device, torch.float16)
            return h.astype(np.float64)

        lora = Lora(model, n, device)
        with torch.no_grad():
            for l in range(NL):
                lora.params[2 * l].copy_(torch.from_numpy(A[l]).to(device))
                lora.params[2 * l + 1].copy_(torch.from_numpy(B[l]).to(device))
        h = {"maia": hits(lora, None), "own": hits(lora, np.arange(n)), "random": hits(lora, rnd)}
        for r in RANKS:
            if r < n:
                h[f"nn{r}"] = hits(lora, order[:, r - 1])
        h["far"] = hits(lora, order[:, n - 2])                       # least similar other side
        for hk in lora.hooks:
            hk.remove()
        ttt = np.load(ROOT / f"data/allie/ttt_{band}_ctx200.npz")
        orig = pl.DataFrame({"game_id": ttt["game_id"], "ply": ttt["ply"], "o": ttt["own"]})
        mine = pl.DataFrame({"game_id": d["test_game_id"][P.pid[te]], "ply": P.ply[te], "m": h["own"] > 0})
        j = mine.join(orig, on=["game_id", "ply"], how="inner")
        res["own_agrees_original"] = round(float((j["m"] == j["o"]).mean()), 4)
        res["own_matched"] = j.height
        for k in KS:                    # cross-fit: clusters + soups from one half of the sides, scored on the other
            if k > n // 8:
                continue
            mapping = np.zeros(n, np.int64)
            members = []                                   # soup row -> member side indices
            sizes = []
            for fo in (0, 1):
                src, dst = np.flatnonzero(half == fo), np.flatnonzero(half != fo)
                km = KMeans(k, n_init=10, random_state=0).fit(X[src]) if k > 1 else None
                ls = km.labels_ if km is not None else np.zeros(len(src), int)
                ld = km.predict(X[dst]) if km is not None else np.zeros(len(dst), int)
                mapping[dst] = fo * k + ld
                members += [src[ls == ci] for ci in range(k)]
                sizes += np.bincount(ls, minlength=k).tolist()
            soup = Lora(model, 2 * k, device, r=256)
            with torch.no_grad():
                for l in range(NL):
                    At = torch.from_numpy(A[l]).to(device)
                    Bt = torch.from_numpy(B[l]).to(device)
                    for ci, mem in enumerate(members):
                        soup.params[2 * l][ci].zero_()
                        soup.params[2 * l + 1][ci].zero_()
                        if len(mem) == 0:
                            continue
                        m_ = torch.from_numpy(mem).to(device)
                        W = torch.einsum("nor,nri->oi", Bt[m_], At[m_]) / len(mem)
                        U, S_, Vh = torch.linalg.svd(W, full_matrices=False)
                        rr = min(256, S_.shape[0])
                        soup.params[2 * l][ci, :rr] = Vh[:rr]
                        soup.params[2 * l + 1][ci, :, :rr] = U[:, :rr] * S_[:rr]
            for scale in (1, 2, 4):
                h[f"k{k}_x{scale}"] = hits(soup, mapping)
                with torch.no_grad():
                    for l in range(NL):
                        soup.params[2 * l + 1].mul_({1: 2, 2: 2, 4: 1}[scale])
            for hk in soup.hooks:
                hk.remove()
            res[f"k{k}_sizes"] = [int(min(sizes)), int(np.median(sizes)), int(max(sizes))]
        side_half = half[P.pid[te]]
        m0 = h["maia"].mean()
        res["maia_acc"] = round(100 * m0, 2)
        res["gain"] = {k: round(100 * (v.mean() - m0), 3) for k, v in h.items() if k != "maia"}
        for k in KS:
            ks = [f"k{k}_x{s}" for s in (1, 2, 4) if f"k{k}_x{s}" in h]
            if not ks:
                continue
            cv = []
            for tr in (0, 1):
                best = max(ks, key=lambda x: h[x][side_half == tr].mean())
                cv.append(100 * (h[best][side_half != tr].mean() - h["maia"][side_half != tr].mean()))
            res["gain"][f"k{k}_cv"] = round(float(np.mean(cv)), 3)
        g = res["gain"]
        res["nn1_cost_share_of_random"] = round((g["own"] - g["nn1"]) / (g["own"] - g["random"]), 3)
        rep[str(band)] = res
        print(band, json.dumps(res), flush=True)
        out.write_text(json.dumps(rep, indent=1))
    print(f"-> {out}")

KSWAP_OUT = ROOT / "lab/results/adapter_kswap.json"
KSWAP_KS = (2, 4, 8, 16, 32, 64, 128)


def kswap(tag: str, device: str, workers: int, bands: list[int], draws: int = 6, limit: int = 0):
    """GPU: choose k by member swaps. For each k, k-means on the band's function-space PCs (as in swap); each side's
    test game is scored with a random other member of its own cluster ('same') and a random member of another cluster
    ('other'), `draws` independent draws each. Clusters depend only on context games (probe positions), never on test
    games, and no side is ever scored with its own adapter, so no cross-fit is needed. Per k:
      gap     = same - other (top-1 pts): what the cluster label alone predicts about a player
      capture = (same - random) / (own - random): share of the own-vs-random personal signal a cluster mate keeps
    A k is 'good' where gap and capture stop growing with k (each doubling of k costs one more bit to store)."""
    import torch
    from sklearn.cluster import KMeans

    from doppel_lab.engines import Maia3
    from doppel_lab.lora_probe import Lora, per_position
    model = Maia3("79m", device).model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    out = KSWAP_OUT.with_name("adapter_kswap_smoke.json") if limit else KSWAP_OUT
    rep = json.loads(out.read_text()) if out.exists() else {}
    t0 = time.time()
    for band in bands:
        if str(band) in rep and not limit:                   # resumable: one band per checkpoint
            print(band, "done already", flush=True)
            continue
        f = fs_load(band)
        a = load(band, tag)
        idx = {int(p): i for i, p in enumerate(a["pid"])}
        rows = np.array([idx[int(p)] for p in f["pid"]])
        if limit:
            rows, f = rows[:limit], {k: v[:limit] for k, v in f.items()}
        d = {k: v[rows] for k, v in a.items() if k not in ("A", "B")}
        n = len(rows)
        G = f["V"].astype(np.float64) @ f["V"].T.astype(np.float64)
        _, _, sc, _ = kpca(G, 64)
        X = sc / np.linalg.norm(sc, axis=1, keepdims=True)
        rng = np.random.default_rng(band)
        P, te = test_positions(d, band, workers)
        own_pid = P.pid
        lora = Lora(model, n, device)
        with torch.no_grad():
            for l in range(NL):
                lora.params[2 * l].copy_(torch.from_numpy(a["A"][l][rows]).to(device))
                lora.params[2 * l + 1].copy_(torch.from_numpy(a["B"][l][rows]).to(device))

        def side_hits(mapping):                  # mean hit per side (sides with no scored position -> nan)
            _, h = per_position(model, lora, P, te, None if mapping is None else (lambda ix: mapping[own_pid[ix]]),
                                device, torch.float16)
            s = np.bincount(own_pid[te], h.astype(np.float64), n)
            return s, np.bincount(own_pid[te], minlength=n)

        maia, cnt = side_hits(None)
        own, _ = side_hits(np.arange(n))
        rnd = np.mean([side_hits((np.arange(n) + rng.integers(1, n, n)) % n)[0] for _ in range(draws)], 0)
        tot = cnt.sum()
        pts = lambda s: 100 * (s.sum() - maia.sum()) / tot
        res = {"n_sides": n, "n_pos": int(tot), "draws": draws, "maia_acc": round(100 * maia.sum() / tot, 2),
               "own": round(pts(own), 3), "random": round(pts(rnd), 3), "k": {}}

        def boot(x, y):                          # side-level bootstrap SE of pts(x) - pts(y)
            diff = 100 * (x - y)
            bs = [diff[ii].sum() / cnt[ii].sum() for ii in rng.integers(0, n, (400, n))]
            return round(float(np.std(bs)), 3)

        for k in KSWAP_KS:
            if k > n // 4:
                continue
            lab = KMeans(k, n_init=10, random_state=0).fit_predict(X)
            mem = [np.flatnonzero(lab == ci) for ci in range(k)]
            same, other = np.zeros(n), np.zeros(n)
            ok = np.array([len(mem[lab[i]]) > 1 for i in range(n)])
            for _ in range(draws):
                ms, mo = np.arange(n), np.arange(n)
                for i in range(n):
                    m = mem[lab[i]]
                    if len(m) > 1:
                        mm = m[m != i]
                        ms[i] = mm[rng.integers(len(mm))]
                    nm = np.flatnonzero(lab != lab[i])
                    mo[i] = nm[rng.integers(len(nm))]
                same += side_hits(ms)[0] / draws
                other += side_hits(mo)[0] / draws
            w = ok & (cnt > 0)
            sz = np.bincount(lab, minlength=k)
            g_same = 100 * (same[w].sum() - maia[w].sum()) / cnt[w].sum()
            g_other = 100 * (other[w].sum() - maia[w].sum()) / cnt[w].sum()
            g_own = 100 * (own[w].sum() - maia[w].sum()) / cnt[w].sum()
            g_rnd = 100 * (rnd[w].sum() - maia[w].sum()) / cnt[w].sum()
            res["k"][k] = {"sizes": [int(sz.min()), int(np.median(sz)), int(sz.max())], "sides_used": int(w.sum()),
                           "same": round(g_same, 3), "other": round(g_other, 3), "gap": round(g_same - g_other, 3),
                           "gap_se": boot(np.where(w, same, 0), np.where(w, other, 0)),
                           "capture": round((g_same - g_rnd) / (g_own - g_rnd), 4)}
            print(band, k, json.dumps(res["k"][k]), f"{time.time() - t0:.0f}s", flush=True)
        for hk in lora.hooks:
            hk.remove()
        rep[str(band)] = res
        print(band, json.dumps({kk: v for kk, v in res.items() if kk != "k"}), flush=True)
        out.write_text(json.dumps(rep, indent=1))
    print(f"-> {out}")


KPAIR_DIR = ROOT / "data/allie/kpair"
KPAIR_OUT = ROOT / "lab/results/adapter_kpair.json"


def kpair(tag: str, device: str, workers: int, bands: list[int], n_bins: int = 14, limit: int = 0, draws: int = 4):
    """GPU: transfer gain as a function of function-space distance, per donor-recipient pair. For each side, donors
    are drawn at log-spaced distance ranks (rank 1 = nearest neighbour, the k -> N ceiling of kswap), one donor per
    rank bin per side, in the same normalized kernel-PCA space kswap clusters in. Raw per-pass hits are saved to
    data/allie/kpair/<band>.npz; kpair_report (CPU) does the analysis. Resumable per band."""
    import torch

    from doppel_lab.engines import Maia3
    from doppel_lab.lora_probe import Lora, per_position
    model = Maia3("79m", device).model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    KPAIR_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for band in bands:
        dst = KPAIR_DIR / (f"{band}_smoke.npz" if limit else f"{band}.npz")
        if dst.exists():
            print(band, "done already", flush=True)
            continue
        f = fs_load(band)
        a = load(band, tag)
        idx = {int(p): i for i, p in enumerate(a["pid"])}
        rows = np.array([idx[int(p)] for p in f["pid"]])
        if limit:
            rows, f = rows[:limit], {k: v[:limit] for k, v in f.items()}
        d = {k: v[rows] for k, v in a.items() if k not in ("A", "B")}
        n = len(rows)
        G = f["V"].astype(np.float64) @ f["V"].T.astype(np.float64)
        sq = np.diag(G)
        _, _, sc, _ = kpca(G, 64)
        X = sc / np.linalg.norm(sc, axis=1, keepdims=True)
        D = np.sqrt(np.maximum(2 - 2 * X @ X.T, 0))              # distance in the clustering space
        np.fill_diagonal(D, np.inf)
        order = np.argsort(D, 1)                                 # order[i, r - 1] = donor at rank r
        rng = np.random.default_rng(band + 7)
        P, te = test_positions(d, band, workers)
        own_pid = P.pid
        lora = Lora(model, n, device)
        with torch.no_grad():
            for l in range(NL):
                lora.params[2 * l].copy_(torch.from_numpy(a["A"][l][rows]).to(device))
                lora.params[2 * l + 1].copy_(torch.from_numpy(a["B"][l][rows]).to(device))

        def side_hits(mapping):
            _, h = per_position(model, lora, P, te, None if mapping is None else (lambda ix: mapping[own_pid[ix]]),
                                device, torch.float16)
            return np.bincount(own_pid[te], h.astype(np.float64), n)

        cnt = np.bincount(own_pid[te], minlength=n)
        maia = side_hits(None)
        own = side_hits(np.arange(n))
        edges = np.unique(np.round(np.geomspace(2, n - 1, n_bins)).astype(int))
        donors, hits, pbin = [order[:, 0]], [side_hits(order[:, 0])], [-1]
        print(band, f"n {n}, nn pass done, {time.time() - t0:.0f}s", flush=True)
        for _ in range(draws):
            for bi, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
                r = rng.integers(lo, hi, n)                      # rank in [lo, hi)
                dn = order[np.arange(n), r - 1]
                donors.append(dn)
                hits.append(side_hits(dn))
                pbin.append(bi)
            print(band, f"draw done, {time.time() - t0:.0f}s", flush=True)
        for hk in lora.hooks:
            hk.remove()
        tmp = dst.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            np.savez(fh, donors=np.array(donors), hits=np.array(hits), maia=maia, own=own, cnt=cnt, X=X.astype(np.float32), pbin=np.array(pbin),
                     fisher_sq=sq, G=G.astype(np.float32), edges=edges)
        tmp.rename(dst)
    print("kpair done", flush=True)


def kpair_report(ks=(8, 32), boots: int = 400):
    """CPU: (1) gain vs distance rank, (2) nearest-neighbour capture ceiling, (3) per-pair Spearman of distance vs gain,
    (4) boundary test: at matched distance, does a donor in the recipient's k-means cluster transfer better than one
    across a boundary? Within each rank bin, same - cross gain, averaged over bins (weights = min count); bootstrap
    over recipients. A continuum predicts 0."""
    from sklearn.cluster import KMeans
    rep = {}
    for f in sorted(KPAIR_DIR.glob("*.npz")):
        if "smoke" in f.stem:
            continue
        z = np.load(f)
        dn, H, maia, own, cnt, X = z["donors"], z["hits"], z["maia"], z["own"], z["cnt"], z["X"].astype(np.float64)
        n = len(maia)
        w = cnt > 0
        G = z["G"].astype(np.float64)
        sq = np.diag(G)
        tot = cnt[w].sum()
        gain = lambda h, m=w: 100 * (h[m].sum() - maia[m].sum()) / cnt[m].sum()
        g_own = gain(own)
        rng = np.random.default_rng(0)
        # per pair: gain of donor on recipient, in pts, and distances
        pg = 100 * (H - maia[None]) / np.maximum(cnt[None], 1)        # (passes, n)
        rec = np.broadcast_to(np.arange(n), dn.shape)
        dcl = np.sqrt(np.maximum(2 - 2 * (X[rec] * X[dn]).sum(-1), 0))
        dfi = np.sqrt(np.maximum(sq[rec] + sq[dn] - 2 * G[rec, dn], 0))
        ww = np.broadcast_to(cnt, dn.shape)
        m = ww > 0
        r = {"n_sides": int(n), "own": round(g_own, 3), "nn": round(gain(H[0]), 3),
             "by_rank_bin": []}
        pbin = z["pbin"]
        bin_gain = lambda bi: 100 * (H[pbin == bi][:, w].sum() - (pbin == bi).sum() * maia[w].sum()) / ((pbin == bi).sum() * tot)
        for i, (a, b) in enumerate(zip(z["edges"][:-1], z["edges"][1:])):
            r["by_rank_bin"].append({"ranks": [int(a), int(b)], "gain": round(bin_gain(i), 3)})
        far = bin_gain(int(pbin.max()))
        r["nn_capture_vs_farthest_bin"] = round((r["nn"] - far) / (g_own - far), 4)

        def wspear(x, y, wt):                                    # Spearman, pairs with >= 10 positions
            k = wt >= 10
            return round(spearman(x[k], y[k]), 4)
        r["pair_spearman"] = {"cluster_space_dist": wspear(-dcl[m], pg[m], ww[m]),
                              "fisher_dist": wspear(-dfi[m], pg[m], ww[m])}
        # pooled gain vs distance: binned means (weighted by positions)
        r["boundary"] = {}
        for k in ks:
            lab = KMeans(k, n_init=10, random_state=0).fit_predict(X)
            same = lab[rec] == lab[dn]

            def effect(sel):
                diffs, wts = [], []
                for p_ in range(1, len(dn)):                     # rank bins only (nn is almost always same)
                    s_ = same[p_] & w & sel
                    c_ = ~same[p_] & w & sel
                    if s_.sum() < 20 or c_.sum() < 20:
                        continue
                    gs = (pg[p_][s_] * cnt[s_]).sum() / cnt[s_].sum()
                    gc = (pg[p_][c_] * cnt[c_]).sum() / cnt[c_].sum()
                    # matched distance within the bin: regress out the distance difference
                    diffs.append(gs - gc)
                    wts.append(min(s_.sum(), c_.sum()))
                return (float(np.average(diffs, weights=wts)) if diffs else float("nan")), len(diffs)
            est, nb = effect(np.ones(n, bool))
            bs = []
            for _ in range(boots):
                sel = np.zeros(n, bool)
                sel[np.unique(rng.integers(0, n, n))] = True
                bs.append(effect(sel)[0])
            # distance confound check: mean distance same vs cross within mixed bins
            dd = []
            for p_ in range(1, len(dn)):
                s_, c_ = same[p_] & w, ~same[p_] & w
                if s_.sum() >= 20 and c_.sum() >= 20:
                    dd.append(float(dcl[p_][s_].mean() - dcl[p_][c_].mean()))
            # regression: pair gain ~ bin fixed effects + distance + same
            rows_ = [(p_, i) for p_ in range(1, len(dn)) for i in np.flatnonzero(w)]
            pi = np.array([a for a, _ in rows_])
            ii = np.array([b for _, b in rows_])
            Xr = np.column_stack([np.eye(int(pbin.max()) + 1)[pbin[pi]], dcl[pi, ii], same[pi, ii].astype(float)])
            yr = pg[pi, ii]
            wr = cnt[ii].astype(float)
            beta = np.linalg.lstsq(Xr * np.sqrt(wr)[:, None], yr * np.sqrt(wr), rcond=None)[0]
            r["boundary"][k] = {"same_minus_cross_in_bin": round(est, 3),
                                "se": round(float(np.nanstd(bs)), 3), "bins_used": nb,
                                "dist_diff_same_minus_cross": [round(x, 4) for x in dd],
                                "regression_same_coef": round(float(beta[-1]), 3),
                                "regression_dist_coef": round(float(beta[-2]), 3)}
        rep[f.stem] = r
        print(f.stem, json.dumps(r), flush=True)
    KPAIR_OUT.write_text(json.dumps(rep, indent=1))
    print(f"-> {KPAIR_OUT}")


SPLIT_TAG = "_ctx200_fit0_s200_split"
SPLIT_OUT = ROOT / "lab/results/adapter_split.json"


def split_report(band: int, plies: tuple | None = None, tag: str = SPLIT_TAG) -> dict | None:
    """Split-half reliability of one band (allie_ttt run --sample 200 --split; probed with --tag SPLIT_TAG).
    Function space: sister-half vs other-player cosines (raw and minus each half's mean), top-1 identification of the
    sister among all other-half adapters, and cross-validated PCA (PCs from half a, variance shared with half b per
    PC; both directions averaged): the reliable spectrum, its participation ratio and the reliable share of variance.
    Test game: half-0 adapter (own) vs its sister half-1 adapter (ctrl of even pseudo-sides) vs another player."""
    f = fs_load(band, tag=tag, plies=plies)
    if f is None:
        return None
    pid = f["pid"]
    pair, half = pid // 2, pid % 2
    both = np.intersect1d(pair[half == 0], pair[half == 1])
    ia = np.array([np.flatnonzero((pair == q) & (half == 0))[0] for q in both])
    ib = np.array([np.flatnonzero((pair == q) & (half == 1))[0] for q in both])
    Va, Vb = f["V"][ia].astype(np.float64), f["V"][ib].astype(np.float64)
    n = len(both)

    def cos_stats(X, Y):
        Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
        Yn = Y / np.linalg.norm(Y, axis=1, keepdims=True)
        C = Xn @ Yn.T
        sis = np.diag(C)
        oth = C[~np.eye(n, dtype=bool)]
        ident = float((C.argmax(1) == np.arange(n)).mean())
        return {"sister": round(float(sis.mean()), 4), "other": round(float(oth.mean()), 4),
                "id_top1": round(ident, 4), "id_chance": round(1 / n, 4)}

    Ac, Bc = Va - Va.mean(0), Vb - Vb.mean(0)
    rel = []
    for X, Y in ((Ac, Bc), (Bc, Ac)):
        Gx = X @ X.T
        w, u = np.linalg.eigh(Gx)
        w, u = w[::-1], u[:, ::-1]
        k = int((w > 1e-9 * w[0]).sum())
        s_ = np.sqrt(w[:k])
        K = Y @ X.T
        pa = u[:, :k] * s_
        pb = (K @ u[:, :k]) / s_
        rel.append(((pa * pb).mean(0), w[:k] / n))
    kk = min(len(rel[0][0]), len(rel[1][0]))
    rv = (rel[0][0][:kk] + rel[1][0][:kk]) / 2
    tv = (rel[0][1][:kk] + rel[1][1][:kk]) / 2
    pos = np.clip(rv, 0, None)
    r = {"band": band, "pairs": n, "raw": cos_stats(Va, Vb), "centred": cos_stats(Ac, Bc),
         "reliable_share": round(float(rv.sum() / tv.sum()), 4),
         "reliable_pr": round(float(pos.sum() ** 2 / (pos ** 2).sum()), 2),
         "reliable_share_top": [round(float(x), 4) for x in (rv[:20] / tv.sum())],
         "n_pc_reliable_gt_1pct": int((rv / rv[rv > 0].sum() > 0.01).sum()),
         "total_pr_half": round(float(tv.sum() ** 2 / (tv ** 2).sum()), 2)}
    z = np.load(ROOT / f"data/allie/ttt_{band}{tag}.npz")
    t = pl.DataFrame({k: z[k] for k in ("game_id", "ply", "maia", "own", "ctrl")}).with_row_index("row")
    t = t.with_columns(h=pl.int_range(pl.len()).over("game_id", "ply"))
    a0, a1 = t.filter(c("h") == 0), t.filter(c("h") == 1)
    m = a0["maia"].mean()
    r["test"] = {"positions": a0.height, "maia_acc": round(100 * m, 2),
                 "half0_own": round(100 * (a0["own"].mean() - m), 3),
                 "half1_own": round(100 * (a1["own"].mean() - a1["maia"].mean()), 3),
                 "sister_swap": round(100 * (a0["ctrl"].mean() - m), 3),
                 "other_player_swap": round(100 * (a1["ctrl"].mean() - a1["maia"].mean()), 3)}
    return r


def split_all(tag: str = SPLIT_TAG):
    rep = {}
    for b in BANDS:
        r = split_report(b, tag=tag)
        if r is not None:
            rep[b] = r
            print(b, json.dumps(r), flush=True)
    out = SPLIT_OUT if tag == SPLIT_TAG else SPLIT_OUT.with_name(f"adapter_split{tag}.json")
    out.write_text(json.dumps(rep, indent=1))
    print(f"-> {out}")

TRAITS = ("capture", "check", "castle", "pawn", "knight", "bishop", "rook", "queen", "king", "promotion", "backward",
          "maia_top1", "maia_top3")
TRAITS_OUT = ROOT / "lab/results/adapter_traits.json"


def probe_move_features(tag: str = "_ctx200_restore") -> np.ndarray:
    """(L, len(TRAITS)) bool: features of every legal move of every probe position, in the delta column order
    (python-chess legal_moves order, as teacher.encode_game)."""
    import chess
    from doppel_lab.allie_v0e import TEST
    base = np.load(probe_dir(tag) / "base.npz")
    games = dict(pl.read_parquet(TEST, columns=["game_id", "moves"]).iter_rows())
    lp0, lens = base["lp"], base["lens"]
    rows, k = [], 0
    for gid, ply, n in zip(base["game_id"], base["ply"], lens):
        b = chess.Board()
        for u in games[str(gid)][:int(ply)]:
            b.push_uci(u)
        lm = list(b.legal_moves)
        assert len(lm) == n, (gid, ply, len(lm), n)
        rank = np.argsort(np.argsort(-lp0[k:k + n]))
        for j, m in enumerate(lm):
            pc = b.piece_at(m.from_square).piece_type
            fwd = chess.square_rank(m.to_square) - chess.square_rank(m.from_square)
            if b.turn == chess.BLACK:
                fwd = -fwd
            rows.append((b.is_capture(m), b.gives_check(m), b.is_castling(m), pc == chess.PAWN, pc == chess.KNIGHT,
                         pc == chess.BISHOP, pc == chess.ROOK, pc == chess.QUEEN, pc == chess.KING and not b.is_castling(m),
                         m.promotion is not None, fwd < 0, rank[j] == 0, rank[j] < 3))
        k += n
    return np.array(rows, bool)


def side_traits(D: np.ndarray, lp0: np.ndarray, lens: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Per side and trait: mean over probe positions of the change in probability mass on moves with the trait
    (adapter - Maia-3), in percentage points."""
    st = np.r_[0, np.cumsum(lens)[:-1]]
    seg = np.repeat(np.arange(len(lens)), lens)
    p0 = np.exp(lp0)
    out = np.zeros((len(D), F.shape[1]))
    for i0 in range(0, len(D), 128):
        lp = lp0 + D[i0:i0 + 128].astype(np.float32)
        p = np.exp(lp)
        p /= np.add.reduceat(p, st, axis=1)[:, seg]               # renormalise (fp16 deltas)
        dp = p - p0
        for t in range(F.shape[1]):
            out[i0:i0 + 128, t] = 100 * np.add.reduceat(dp * F[:, t], st, axis=1).mean(1)
    return out


def traits(tag: str = "_ctx200_restore"):
    """11f step 3: what the adapter axes mean. Per band: mean and spread of each trait over sides, the share of
    sides whose trait shift is beyond 2 se of the probe noise, and Spearman of traits with the top PCs and Elo."""
    F = probe_move_features(tag)
    base = np.load(probe_dir(tag) / "base.npz")
    rep = {"traits": list(TRAITS), "feature_rate_pct": {t: round(100 * float(F[:, i].mean()), 2) for i, t in
                                                       enumerate(TRAITS)}, "bands": {}}
    for b in BANDS:
        fs = sorted(probe_dir(tag).glob(f"{b}_*.npz"))
        if not fs:
            continue
        zs = [np.load(f) for f in fs]
        keep = np.concatenate([z["best_ep"] for z in zs]) > 0
        D = np.concatenate([z["delta"] for z in zs])[keep]
        T = side_traits(D, base["lp"], base["lens"], F)
        f = fs_load(b, tag=tag)
        meta = side_meta(f)
        G = f["V"].astype(np.float64) @ f["V"].T.astype(np.float64)
        _, _, sc, _ = kpca(G, 5)
        elo = meta["elo"].to_numpy().astype(float)
        r = {"n": int(len(T))}
        for i, t in enumerate(TRAITS):
            r[t] = {"mean": round(float(T[:, i].mean()), 3), "sd": round(float(T[:, i].std()), 3),
                    "elo": round(spearman(T[:, i], elo), 3),
                    **{f"pc{j + 1}": round(spearman(T[:, i], sc[:, j]), 3) for j in range(3)}}
        rep["bands"][b] = r
        print(b, r["n"], " ".join(f"{t}:{r[t]['mean']:+.2f}/{r[t]['sd']:.2f}(pc1 {r[t]['pc1']:+.2f})" for t in TRAITS),
              flush=True)
    out = TRAITS_OUT if tag == "_ctx200_restore" else TRAITS_OUT.with_name(f"adapter_traits{tag}.json")
    out.write_text(json.dumps(rep, indent=1))
    print(f"-> {out}")


COLD_DIR = ROOT / "data/allie/coldstart"
COLD_OUT = ROOT / "lab/results/adapter_coldstart.json"
COLD_SCALES = (0.25, 0.5, 1.0)


def coldstart(tag: str, device: str, workers: int, bands: list[int], limit: int = 0):
    """GPU: can a player with no adapter of their own beat Maia with other players' adapters? Each side's test game
    is scored with the average of k donor adapters (exact: the k rank-8 LoRAs concatenated with weight 1/k), shrunk
    by a scale s in COLD_SCALES. Donors:
      - oracle{k}: the k nearest sides in kpair's normalized kernel-PCA space. Locating the side needs its own
        adapter, so this is an upper bound for any cold-start locator (k=1 at s=1 reproduces kpair's NN pass);
      - elo{k}: the k sides of nearest rating (realistic: needs only the rating);
      - bandmean: mean dW of every side in the band (SVD, rank 256; the side's own 1/n share is left in).
    Scale is chosen on one crc32 half of the sides and reported on the other (as in swap). Raw hits ->
    data/allie/coldstart/<band>.npz; resumable per band."""
    import zlib

    import torch

    from doppel_lab.engines import Maia3
    from doppel_lab.lora_probe import Lora, per_position
    model = Maia3("79m", device).model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    COLD_DIR.mkdir(parents=True, exist_ok=True)
    out = COLD_OUT.with_name("adapter_coldstart_smoke.json") if limit else COLD_OUT
    rep = json.loads(out.read_text()) if out.exists() else {}
    t0 = time.time()
    for band in bands:
        dst = COLD_DIR / (f"{band}_smoke.npz" if limit else f"{band}.npz")
        if not dst.exists():
            f = fs_load(band)
            a = load(band, tag)
            idx = {int(p): i for i, p in enumerate(a["pid"])}
            rows = np.array([idx[int(p)] for p in f["pid"]])
            if limit:
                rows, f = rows[:limit], {k: v[:limit] for k, v in f.items()}
            d = {k: v[rows] for k, v in a.items() if k not in ("A", "B")}
            A = [torch.from_numpy(a["A"][l][rows]) for l in range(NL)]
            B = [torch.from_numpy(a["B"][l][rows]) for l in range(NL)]
            n = len(rows)
            G = f["V"].astype(np.float64) @ f["V"].T.astype(np.float64)
            _, _, sc, _ = kpca(G, 64)
            X = sc / np.linalg.norm(sc, axis=1, keepdims=True)
            D = np.sqrt(np.maximum(2 - 2 * X @ X.T, 0))
            np.fill_diagonal(D, np.inf)
            elo = side_meta(d)["elo"].to_numpy().astype(np.float64)
            rng = np.random.default_rng(band)
            E = np.abs(elo[:, None] - elo[None, :]) + rng.random((n, n)) * 1e-3     # random tie-break
            np.fill_diagonal(E, np.inf)
            nbr = {"oracle": np.argsort(D, 1), "elo": np.argsort(E, 1)}
            P, te = test_positions(d, band, workers)
            tp = P.pid[te]
            H, L = {}, {}

            def run(lora, rowmap, sel_mask, key, bs):
                lp, h = per_position(model, lora, P, te[sel_mask], lambda ix: rowmap(P.pid[ix]), device,
                                     torch.float16, bs=bs)
                H.setdefault(key, np.zeros(len(te), bool))[sel_mask] = h
                L.setdefault(key, np.zeros(len(te)))[sel_mask] = lp

            def scaled_passes(lora, rowmap, sel_mask, name, bs):
                with torch.no_grad():
                    for p in lora.params[1::2]:
                        p.mul_(COLD_SCALES[0])
                for si, s in enumerate(COLD_SCALES):
                    if si:
                        with torch.no_grad():
                            for p in lora.params[1::2]:
                                p.mul_(s / COLD_SCALES[si - 1])
                    run(lora, rowmap, sel_mask, f"{name}_x{s}", bs)
                for hk in lora.hooks:
                    hk.remove()

            m0 = Lora(model, 1, device)
            lp0, h0 = per_position(model, m0, P, te, None, device, torch.float16)
            for hk in m0.hooks:
                hk.remove()
            H["maia"], L["maia"] = h0, lp0
            own = Lora(model, n, device)
            with torch.no_grad():
                for l in range(NL):
                    own.params[2 * l].copy_(A[l].to(device))
                    own.params[2 * l + 1].copy_(B[l].to(device))
            scaled_passes(own, lambda p: p, np.ones(len(te), bool), "own", 1024)
            del own
            for src, ks in (("oracle", (1, 4, 16, 64)), ("elo", (16, 64))):
                for k in ks:
                    k = min(k, n - 1)
                    m = max(1, int(3e9 / (k * 8 * 24576 * 4)))          # sides per chunk, ~3 GB of factors
                    for s in range(0, n, m):
                        e = min(n, s + m)
                        lo = Lora(model, e - s, device, r=8 * k)
                        nb = torch.from_numpy(nbr[src][s:e, :k].copy())
                        with torch.no_grad():
                            for l in range(NL):
                                lo.params[2 * l].copy_(A[l][nb].reshape(e - s, 8 * k, -1).to(device))
                                lo.params[2 * l + 1].copy_(
                                    (B[l][nb].permute(0, 2, 1, 3).reshape(e - s, B[l].shape[1], 8 * k) / k).to(device))
                        scaled_passes(lo, lambda p, s=s: p - s, (tp >= s) & (tp < e), f"{src}{k}",
                                      max(64, min(1024, 2 ** 16 // k)))
                        del lo
                    print(band, f"{src}{k} done, {time.time() - t0:.0f}s", flush=True)
            bm = Lora(model, 1, device, r=256)
            with torch.no_grad():
                for l in range(NL):
                    W = torch.einsum("nor,nri->oi", B[l].to(device), A[l].to(device)) / n
                    U, S_, Vh = torch.linalg.svd(W, full_matrices=False)
                    rr = min(256, S_.shape[0])
                    bm.params[2 * l][0].zero_()
                    bm.params[2 * l + 1][0].zero_()
                    bm.params[2 * l][0, :rr] = Vh[:rr]
                    bm.params[2 * l + 1][0, :, :rr] = U[:, :rr] * S_[:rr]
            scaled_passes(bm, lambda p: np.zeros_like(p), np.ones(len(te), bool), "bandmean", 1024)
            half = np.array([zlib.crc32(g.encode()) & 1 for g in d["test_game_id"]])[tp]
            tmp = dst.with_suffix(".tmp")
            with open(tmp, "wb") as fh:
                np.savez(fh, keys=np.array(list(H)), H=np.array([H[k] for k in H]), L=np.array([L[k] for k in H]),
                         pid=tp, half=half)
            tmp.rename(dst)
        z = np.load(dst)
        keys, Hs, Ls, half = list(z["keys"]), z["H"].astype(np.float64), z["L"], z["half"]
        i0 = keys.index("maia")
        gain = {k: round(100 * (Hs[i].mean() - Hs[i0].mean()), 3) for i, k in enumerate(keys) if k != "maia"}
        lpg = {k: round(float(Ls[i].mean() - Ls[i0].mean()), 4) for i, k in enumerate(keys) if k != "maia"}
        cv = {}
        for name in sorted({k.rsplit("_x", 1)[0] for k in keys if "_x" in k}):
            cand = [keys.index(f"{name}_x{s}") for s in COLD_SCALES]
            g = []
            for tr in (0, 1):
                best = max(cand, key=lambda i: Hs[i][half == tr].mean())
                g.append(100 * (Hs[best][half != tr].mean() - Hs[i0][half != tr].mean()))
            cv[name] = round(float(np.mean(g)), 3)
        rep[str(band)] = {"n_pos": int(Hs.shape[1]), "maia_acc": round(100 * Hs[i0].mean(), 2), "gain_pts": gain,
                          "gain_lp": lpg, "cv_best_scale_pts": cv}
        print(band, json.dumps(rep[str(band)]["cv_best_scale_pts"]), flush=True)
        out.write_text(json.dumps(rep, indent=1))
    print(f"-> {out}")


def main():
    ap = argparse.ArgumentParser()
    s = ap.add_subparsers(dest="cmd", required=True)
    g = s.add_parser("geom")
    g.add_argument("--tag", default="_ctx200_restore")
    p = s.add_parser("probe")
    p.add_argument("--tag", default="_ctx200_restore")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--gpu-ok", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="CPU smoke: first N sides of the first chunk -> probe/smoke/")
    f = s.add_parser("fs")
    f.add_argument("--own-band", action="store_true")
    w = s.add_parser("swap")
    w.add_argument("--tag", default="_ctx200_restore")
    w.add_argument("--workers", type=int, default=12)
    w.add_argument("--bands", default=",".join(map(str, BANDS)))
    w.add_argument("--gpu-ok", action="store_true")
    w.add_argument("--limit", type=int, default=0, help="CPU smoke: first N probed sides per band")
    ks = s.add_parser("kswap")
    ks.add_argument("--tag", default="_ctx200_restore")
    ks.add_argument("--workers", type=int, default=12)
    ks.add_argument("--bands", default=",".join(map(str, BANDS)))
    ks.add_argument("--draws", type=int, default=6)
    ks.add_argument("--gpu-ok", action="store_true")
    ks.add_argument("--limit", type=int, default=0, help="CPU smoke: first N probed sides per band")
    kp = s.add_parser("kpair")
    kp.add_argument("--tag", default="_ctx200_restore")
    kp.add_argument("--workers", type=int, default=12)
    kp.add_argument("--bands", default=",".join(map(str, BANDS)))
    kp.add_argument("--gpu-ok", action="store_true")
    kp.add_argument("--limit", type=int, default=0)
    s.add_parser("kpair_report")
    cs = s.add_parser("coldstart")
    cs.add_argument("--tag", default="_ctx200_restore")
    cs.add_argument("--workers", type=int, default=12)
    cs.add_argument("--bands", default=",".join(map(str, BANDS)))
    cs.add_argument("--gpu-ok", action="store_true")
    cs.add_argument("--limit", type=int, default=0)
    sp = s.add_parser("split")
    sp.add_argument("--tag", default=SPLIT_TAG)
    tr = s.add_parser("traits")
    tr.add_argument("--tag", default="_ctx200_restore")
    a = ap.parse_args()
    if a.cmd == "split":
        split_all(a.tag)
    elif a.cmd == "traits":
        traits(a.tag)
    elif a.cmd == "fs":
        fs_geom(a.own_band)
    elif a.cmd == "swap":
        swap(a.tag, "cuda" if a.gpu_ok else "cpu", a.workers, [int(x) for x in a.bands.split(",")], a.limit)
    elif a.cmd == "kswap":
        kswap(a.tag, "cuda" if a.gpu_ok else "cpu", a.workers, [int(x) for x in a.bands.split(",")], a.draws, a.limit)
    elif a.cmd == "kpair":
        kpair(a.tag, "cuda" if a.gpu_ok else "cpu", a.workers, [int(x) for x in a.bands.split(",")], limit=a.limit,
              draws=1 if a.limit else 4)
    elif a.cmd == "coldstart":
        coldstart(a.tag, "cuda" if a.gpu_ok else "cpu", a.workers, [int(x) for x in a.bands.split(",")], a.limit)
    elif a.cmd == "kpair_report":
        kpair_report()
    elif a.cmd == "geom":
        geom(a.tag)
    elif a.cmd == "probe":
        probe(a.tag, "cuda" if a.gpu_ok else "cpu", a.workers, a.limit)


if __name__ == "__main__":
    main()

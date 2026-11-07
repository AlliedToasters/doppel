"""Encoder benchmark, Maia-3 side: per-game embeddings from the frozen Maia-3 79M trunk (GPU, via the warden).

    python -m doppel_lab.bench_embed run --gpu-ok         # data/bench/games.parquet -> data/bench/maia3.npz
    python -m doppel_lab.bench_embed probe                # CPU: same tasks and folds as bench.surface

Maia-3 takes both players' ratings as input. Feeding the real ones would hand the encoder the label, so every
position gets 1500 vs 1500: the trunk sees only the board history, like the surface features.
Per side (White, Black): the mean of the pooled trunk output (last_ln) over that side's positions, and policy
scalars of the move it played: mean and std of log-prob, mean entropy, top-1 agreement, mean log-prob rank.
Probes: ridge / logistic regression on the standardised embeddings (linear probe); gradient-boosted trees on
the policy scalars; and both on top of the surface features.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import polars as pl

from doppel_lab.teacher import HIST, encode_game

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "data" / "bench"
RESULTS = ROOT / "lab" / "results"
BLOCK = 4_000
SCAL = ["lp_mean", "lp_std", "ent_mean", "top1", "rank_mean"]


def run(device: str, workers: int, batch: int, limit: int = 0, out_name: str = "maia3.npz", suite: str = "lichess"):
    import torch

    from doppel_lab.engines import Maia3
    from doppel_lab.bench import suite_dir
    games = pl.read_parquet(suite_dir(suite) / "games.parquet").select("game_id", "moves")
    if limit:
        games = games.head(limit)
    out = suite_dir(suite) / out_name
    model = Maia3("79m", device).model.eval()
    pooled = {}
    model.last_ln.register_forward_hook(lambda m, i, o: pooled.__setitem__("x", o))
    dim = model.last_ln.normalized_shape[0]
    G = games.height
    emb = np.zeros((G, 2, dim), np.float16)
    scal = np.zeros((G, 2, len(SCAL)), np.float32)
    rows = list(games.iter_rows())
    t0, done = time.time(), 0
    pos_lp, pos_rank, pos_ent, pos_n = [], [], [], np.zeros(G, np.int64)  # per position, for the Doppel encoder
    with mp.get_context("fork").Pool(workers) as pool:
        for b0 in range(0, G, BLOCK):
            blk = rows[b0:b0 + BLOCK]
            enc = pool.map(encode_game, [(m, 1500, 1500) for _, m in blk], chunksize=32)
            n_per = np.array([len(e[3]) for e in enc])
            offsets = np.r_[0, np.cumsum(n_per)]
            planes = torch.from_numpy(np.concatenate([e[0] for e in enc])).to(device)
            hidx = np.concatenate([np.clip(np.arange(n)[:, None] + np.arange(-HIST + 1, 1)[None, :], 0, None) + off
                                   for n, off in zip(n_per, offsets[:-1])])
            legal = np.concatenate([e[1] for e in enc]).astype(np.int64)
            lcount = np.concatenate([e[2] for e in enc]).astype(np.int64)
            lrow = np.repeat(np.arange(len(lcount)), lcount)
            lstart = np.r_[0, np.cumsum(lcount)]
            played = torch.from_numpy(np.concatenate([e[3] for e in enc]).astype(np.int64)).to(device)
            P = len(hidx)
            feat = torch.empty(P, dim, device=device)
            lp_p, ent, top1, rank = (torch.empty(P, device=device) for _ in range(4))
            hidx_t = torch.from_numpy(hidx).to(device)
            e1500 = torch.full((batch,), 1500, device=device)
            for s in range(0, P, batch):
                e_ = min(P, s + batch)
                x = planes[hidx_t[s:e_]].float().permute(0, 2, 1, 3).reshape(e_ - s, 64, HIST * 12)
                x = torch.cat([x, torch.zeros(e_ - s, 64, 1, device=device)], 2)
                with torch.no_grad(), torch.autocast("cuda", enabled=device.startswith("cuda")):
                    logits = model(x, e1500[:e_ - s], e1500[:e_ - s])[0].float()
                px = pooled["x"].float()
                feat[s:e_] = px.mean(1) if px.dim() == 3 else px
                mask = torch.zeros_like(logits, dtype=torch.bool)
                a, b_ = lstart[s], lstart[e_]
                mask[torch.from_numpy(lrow[a:b_] - s).to(device), torch.from_numpy(legal[a:b_]).to(device)] = True
                lp = torch.log_softmax(logits.masked_fill(~mask, float("-inf")), 1)
                pl_ = lp.gather(1, played[s:e_, None])[:, 0]
                lp_p[s:e_] = pl_.clamp(min=-30)
                ent[s:e_] = -(lp.exp() * lp.masked_fill(~mask, 0.0)).sum(1)
                top1[s:e_] = (lp.argmax(1) == played[s:e_]).float()
                rank[s:e_] = (lp > pl_[:, None]).sum(1).float()
            feat, lp_p, ent, top1, rank = (t.cpu().numpy() for t in (feat, lp_p, ent, top1, rank))
            pos_lp.append(lp_p.astype(np.float16))
            pos_rank.append(np.minimum(rank, 16).astype(np.uint8))   # v0_data's rank: top-16 index, 16 = outside
            pos_ent.append(ent.astype(np.float16))
            pos_n[b0:b0 + len(blk)] = n_per
            for j in range(len(blk)):
                o, n = offsets[j], n_per[j]
                for side in (0, 1):
                    ix = np.arange(o + side, o + n, 2)
                    if len(ix) == 0:
                        continue
                    emb[b0 + j, side] = feat[ix].mean(0)
                    scal[b0 + j, side] = (lp_p[ix].mean(), lp_p[ix].std(), ent[ix].mean(), top1[ix].mean(),
                                          np.log1p(rank[ix]).mean())
            done += P
            print(f"{b0 + len(blk):,}/{G:,} games, {done:,} positions, {done / (time.time() - t0):,.0f}/s", flush=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, game_ids=games["game_id"].to_numpy(), emb=emb, scal=scal, scal_names=np.array(SCAL),
             pos_off=np.r_[0, np.cumsum(pos_n)], pos_lp=np.concatenate(pos_lp), pos_rank=np.concatenate(pos_rank),
             pos_ent=np.concatenate(pos_ent))
    tmp.rename(out)
    print(f"done -> {out}", flush=True)


def probe(suite: str = "lichess"):
    from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    from doppel_lab.bench import SEED, SUITES, suite_dir
    TASKS, D = SUITES[suite], suite_dir(suite)
    z = np.load(D / "maia3.npz", allow_pickle=True)
    gid = pl.DataFrame({"game_id": z["game_ids"], "i": np.arange(len(z["game_ids"]))})
    E = z["emb"].astype(np.float32).reshape(len(gid), -1)
    S = z["scal"].reshape(len(gid), -1)
    S = np.concatenate([S, S[:, :len(SCAL)] - S[:, len(SCAL):]], 1)        # + White minus Black
    tasks = pl.read_parquet(D / "tasks.parquet").join(gid, on="game_id")
    surf = pl.read_parquet(D / "surface.parquet")
    scols = [k for k in surf.columns if k != "game_id" and surf[k].null_count() < surf.height]
    d = tasks.join(surf, on="game_id")
    sname_ = "bench_surface.json" if suite == "lichess" else f"bench_surface_{suite}.json"
    surface = json.loads((RESULTS / sname_).read_text())["tasks"]
    res = {"tasks": {}}
    for name, (flt, lab, kind) in TASKS.items():
        x = d.filter(flt).with_columns(y=lab)
        y, fold, ii = x["y"].to_numpy().astype(float), x["fold"].to_numpy(), x["i"].to_numpy()
        Xs = x.select(scols).to_numpy().astype(float)
        sets = {"maia_linear": ("lin", E[ii]), "maia_policy": ("hgb", S[ii]),
                "surface+maia_policy": ("hgb", np.concatenate([Xs, S[ii]], 1))}
        r = {"surface": surface[name]}
        for sname, (how, X) in sets.items():
            pred = np.empty(len(y))
            for k in range(5):
                tr, te = fold != k, fold == k
                if how == "lin":
                    sc = StandardScaler().fit(X[tr])
                    pca = PCA(256, random_state=SEED).fit(sc.transform(X[tr]))
                    A, B = pca.transform(sc.transform(X[tr])), pca.transform(sc.transform(X[te]))
                    if kind == "reg":
                        pred[te] = Ridge(alpha=10.0).fit(A, y[tr]).predict(B)
                    else:
                        pred[te] = LogisticRegression(C=0.1, max_iter=2000).fit(A, y[tr]).predict_proba(B)[:, 1]
                else:
                    if kind == "reg":
                        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=SEED)
                        pred[te] = m.fit(X[tr], y[tr]).predict(X[te])
                    else:
                        m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, random_state=SEED)
                        pred[te] = m.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            r[sname] = ({"mae": float(mean_absolute_error(y, pred)), "r2": float(r2_score(y, pred))} if kind == "reg"
                        else {"auc": float(roc_auc_score(y, pred))})
        res["tasks"][name] = r
        key = "r2" if kind == "reg" else "auc"
        print(f"{name:13s} " + "  ".join(f"{k} {v[key]:.3f}" for k, v in r.items()), flush=True)
    (RESULTS / ("bench_maia3.json" if suite == "lichess" else f"bench_maia3_{suite}.json")).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--workers", type=int, default=12)
    r.add_argument("--batch", type=int, default=2048)
    r.add_argument("--gpu-ok", action="store_true")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--out", default="maia3.npz")
    r.add_argument("--suite", default="lichess")
    pr = sub.add_parser("probe")
    pr.add_argument("--suite", default="lichess")
    a = ap.parse_args()
    if a.cmd == "run":
        run("cuda" if a.gpu_ok else "cpu", a.workers, a.batch, a.limit, a.out, a.suite)
    else:
        probe(a.suite)

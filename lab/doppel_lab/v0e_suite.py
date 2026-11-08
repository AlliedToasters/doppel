"""Players suite (players_suite.py) for the v0e z: re-ID and rating R² at k = 1..40 games, CPU only.

    python -m doppel_lab.v0e_suite --run v0e            # -> lab/results/players_pool_v0e.json

The suite's per-game Maia vectors (data/bench/players/maia3.npz, maia3_sweep.npz) have the definitions v0e was
trained on, so z needs no GPU: z(k) = posterior mean of v0e's set encoder over the player's first k games;
gallery z = the same over games 40-79. The cond is rating-blind (1500 vs 1500, 3+0) so, like the bars, z's rating
information comes from the moves alone. Sweep features: ll(1000) - ll(1500) and ll(2000) - ll(1500) from the
9-rating sweep over all moves (v0e_vec used half of them; same expectation). Protocol as players_suite.score:
rating R² by trees, 5-fold by player (crc32); re-ID top-1 in pools of 500 against the gallery, standardised by the
gallery's mean and std.

Two scopes: all 2,000 players (the bars of SCIENCE.md §8), and "band": players whose mean rating is within
1400-1700, where the 1500-band model is in distribution (pools of 250; the bars recomputed on that subset).
"""

from __future__ import annotations

import argparse
import json
import zlib

import numpy as np
import polars as pl
import torch

from doppel_lab.bench import BENCH
from doppel_lab.persistence import CORPUS, FEATS, RESULTS, SEED
from doppel_lab.players_suite import GALLERY, KS, PER, POOL
from doppel_lab.v0e_data import V0E, ctx_feats, end_labels, surface

D = BENCH / "players"
BAND = (1400, 1700)
c = pl.col


def suite_surface() -> np.ndarray:
    """(G, 2, 12) v0e surface stats for the suite's games, in maia3.npz order (cached)."""
    out = V0E / "players_surf.npz"
    z = np.load(D / "maia3.npz", allow_pickle=True)
    if out.exists():
        s = np.load(out, allow_pickle=True)
        assert (s["game_ids"] == z["game_ids"]).all()
        return s["surf"]
    g = pl.read_parquet(D / "games.parquet")
    hdr = pl.concat([pl.read_parquet(f, columns=["game_id", "result", "termination"])
                     for f in sorted(CORPUS.glob("part-*.parquet"))])
    g = g.join(hdr, on="game_id", how="left", maintain_order="left")
    assert g["result"].null_count() == 0
    order = {gid: i for i, gid in enumerate(g["game_id"].to_list())}
    g = g[[order[x] for x in z["game_ids"]]]
    end, res, _ = end_labels(g)
    surf = surface(g, end, res)
    np.savez(out, game_ids=z["game_ids"], surf=surf)
    return surf


@torch.no_grad()
def z_sets(model, emb, feat, groups: list[np.ndarray], batch: int = 256) -> np.ndarray:
    """Posterior mean for each index set (rows of emb / feat), rating-blind 3+0 cond."""
    cond = torch.tensor([0.0, 0.0, float(np.log1p(180)), 0.0])
    out = []
    for b0 in range(0, len(groups), batch):
        gs = groups[b0:b0 + batch]
        L = max(len(x) for x in gs)
        E = torch.zeros(len(gs), L, emb.shape[1])
        F = torch.zeros(len(gs), L, feat.shape[1])
        M = torch.zeros(len(gs), L, dtype=torch.bool)
        for i, x in enumerate(gs):
            E[i, :len(x)] = torch.from_numpy(emb[x].astype(np.float32))
            F[i, :len(x)] = torch.from_numpy(feat[x])
            M[i, :len(x)] = True
        mu, _ = model.encoder(E, F, M, cond.expand(len(gs), -1))
        out.append(mu.numpy())
    return np.concatenate(out)


def metrics(probe: dict, gallery: np.ndarray, y: dict, fold: np.ndarray, pool: int, rng) -> tuple[dict, dict]:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import r2_score
    mu_, sd_ = np.nanmean(gallery, 0), np.nanstd(gallery, 0) + 1e-9
    Gs = np.nan_to_num((gallery - mu_) / sd_)
    r2, reid = {}, {}
    n = len(gallery)
    for k in KS:
        P = probe[k]
        Ps = np.nan_to_num((P - mu_) / sd_)
        Pf = P[:, [j for j in range(P.shape[1]) if len(np.unique(P[np.isfinite(P[:, j]), j])) >= 2]]
        pred = np.empty(n)
        for f in range(5):
            tr, te = fold != f, fold == f
            pred[te] = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05,
                                                     random_state=SEED).fit(Pf[tr], y[k][tr]).predict(Pf[te])
        t1 = m = 0
        for _ in range(max(1, n // pool)):
            ii = rng.choice(n, pool, replace=False)
            Dm = ((Ps[ii][:, None, :] - Gs[ii][None, :, :]) ** 2).sum(-1)
            t1 += (Dm.argmin(1) == np.arange(pool)).sum()
            m += pool
        r2[str(k)], reid[str(k)] = float(r2_score(y[k], pred)), t1 / m
    return r2, reid


def main(run: str):
    from sklearn.decomposition import PCA

    from doppel_lab.v0_data import V0
    from doppel_lab.v0e_train import build
    sides = pl.read_parquet(D / "sides.parquet")
    z = np.load(D / "maia3.npz", allow_pickle=True)
    w = np.load(D / "maia3_sweep.npz", allow_pickle=True)
    assert (w["game_ids"] == z["game_ids"]).all()
    gi = {g: i for i, g in enumerate(z["game_ids"])}
    idx = np.array([gi[g] for g in sides["game_id"].to_list()])
    col = sides["col"].to_numpy()
    elos = list(w["elos"])
    ll = w["ll"][idx, col].astype(np.float32)
    scal = z["scal"][idx, col].astype(np.float32)
    sw3 = np.stack([ll[:, elos.index(1000)], scal[:, 0], ll[:, elos.index(2000)]], 1)
    surf = suite_surface()[idx, col]
    feat = ctx_feats(scal, sw3, surf)
    emb = z["emb"][idx, col]                                    # (n sides, 1024) f16

    model = build(True, "cpu")                                  # the encoder only; the trunk is not used
    s = torch.load(V0 / "runs" / run / "ckpt.pt", map_location="cpu", weights_only=False)
    model.encoder.load_state_dict({k[len("encoder."):]: v for k, v in s["model"].items() if k.startswith("encoder.")})
    model.eval()

    surf_df = pl.read_parquet(D / "surface_sides.parquet")
    sides = sides.join(surf_df.select("t", "player", "elo", *FEATS), on=["t", "player"], how="left")
    players, k_ = sides["player"].to_numpy(), sides["k"].to_numpy()
    elo = sides["elo"].to_numpy().astype(float)
    up = np.unique(players)
    pi = {p: i for i, p in enumerate(up)}
    pidx = np.array([pi[p] for p in players])
    fold_all = np.array([zlib.crc32(p.encode()) % 5 for p in up])
    rows_of = [np.flatnonzero(pidx == i) for i in range(len(up))]
    rows_of = [r[np.argsort(k_[r])] for r in rows_of]

    def pool_mean(X, mask):
        out, cnt = np.zeros((len(up), X.shape[1])), np.zeros((len(up), X.shape[1]))
        ok = np.isfinite(X) & mask[:, None]
        np.add.at(out, pidx, np.where(ok, X, 0))
        np.add.at(cnt, pidx, ok)
        return np.where(cnt > 0, out / np.maximum(cnt, 1), np.nan)

    y = {k: pool_mean(elo[:, None], k_ < k)[:, 0] for k in KS}
    print(f"{len(up)} players; encoding z for k = {KS} and the gallery", flush=True)
    zk = {k: z_sets(model, emb, feat, [r[k_[r] < k] for r in rows_of]) for k in KS}
    zg = z_sets(model, emb, feat, [r[k_[r] >= PER - GALLERY] for r in rows_of])

    rng0 = np.random.default_rng(SEED)
    pca = PCA(64, random_state=SEED).fit(emb[rng0.choice(len(emb), min(50_000, len(emb)), replace=False)].astype(np.float32))
    em = pca.transform(emb.astype(np.float32))
    S = sides.select(FEATS).to_numpy().astype(float)
    sw9 = w["ll"][idx, col].astype(float)
    base = {"surface": S, "maia_emb": em, "maia_sweep": sw9, "surface+maia+sweep": np.c_[S, em, sw9]}
    gal_mask = k_ >= PER - GALLERY
    views = {name: ({k: pool_mean(X, k_ < k) for k in KS}, pool_mean(X, gal_mask)) for name, X in base.items()}
    views[f"{run}_z"] = (zk, zg)
    views[f"surface+{run}_z"] = ({k: np.c_[views["surface"][0][k], zk[k]] for k in KS},
                                 np.c_[views["surface"][1], zg])

    mean_elo = pool_mean(elo[:, None], np.ones(len(elo), bool))[:, 0]
    scopes = {"all": (np.ones(len(up), bool), POOL), "band": ((mean_elo >= BAND[0]) & (mean_elo < BAND[1]), 250)}
    res = {"run": run, "ks": list(KS), "cond": "rating-blind 1500 vs 1500, 3+0", "scopes": {}}
    for scope, (sel, pool) in scopes.items():
        rng = np.random.default_rng(SEED)
        r = {"players": int(sel.sum()), "pool": pool, "rating_r2": {}, "reid_top1": {}}
        for name, (probe, gal) in views.items():
            r2, reid = metrics({k: v[sel] for k, v in probe.items()}, gal[sel], {k: v[sel] for k, v in y.items()},
                               fold_all[sel], pool, rng)
            r["rating_r2"][name], r["reid_top1"][name] = r2, reid
            print(f"[{scope} n={sel.sum()}] {name:22s} R² " + " ".join(f"k{k}:{r2[str(k)]:.3f}" for k in KS)
                  + " | re-ID " + " ".join(f"k{k}:{reid[str(k)]:.3f}" for k in KS), flush=True)
        res["scopes"][scope] = r
    (RESULTS / f"players_pool_{run}.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v0e")
    main(ap.parse_args().run)

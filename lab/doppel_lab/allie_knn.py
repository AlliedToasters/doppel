"""Retrieval step 0 on the Allie test set: nearest neighbours among the player's own earlier positions, no training.

    python -m doppel_lab.allie_knn states --gpu-ok     # GPU (warden), ~20 min: Maia-3 keys + policy, every position
    python -m doppel_lab.allie_knn score               # CPU, ~10 min -> lab/results/allie_knn.json

Games: split allie1500 (data/v0e/games_1500_allie1500.parquet, from `allie_v0e select`): the 2,021 test games of
the 1400-1600 bucket plus each side's 40 most recent earlier non-test 2022 games. One Maia-3 79M pass (fp16, real
Elos, the same input as top1.py) over every position gives
  key     the pooled trunk output (last_ln, mean over squares), PCA to D dims (fit on the first block), fp16
  played  the played move (mover-frame vocabulary index)
  legal / logp (test games only)  Maia-3's log-probs over the legal moves, for mixing
Store for a test position = the test player's positions as the mover in their context games for that test game
(strictly earlier by selection; test games are never in a store). Query = the test position's key. Prediction:
the top-N neighbours by cosine, each voting for its played move with weight exp(cos / tau) if that move is legal in
the query, mixed into Maia-3 as p = (1 - lam) p_maia + lam p_knn when the best cosine is >= thr. (lam, thr, tau,
N) are tuned on one half of the test games (crc32 of game_id) and reported on the other half, both ways.
Reported: top-1 gain over Maia-3, by ply bucket, by best cosine, and split by exact-repeat history (the memory
table's `hist`), so "beyond exact repeats" is the hist = False row. Control: the same procedure with each side's
store replaced by another test side's store (shuffled players), which measures the generic (crowd) retrieval
effect; the personal effect is the difference.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
import zlib
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/allie/knn_1500"
RESULTS = ROOT / "lab/results/allie_knn.json"


def knn_dir(band: int = 1500) -> Path:
    """Per-bucket states (keys, played, off, test.npz). Every bucket other than 1500 projects with the 1500 PCA
    (knn_1500/pca.npz), so one learned head design and one key space serve all buckets."""
    return ROOT / f"data/allie/knn_{band}"
D, BLOCK, MAXL = 128, 4_000, 224
c = pl.col


def games(band: int = 1500) -> pl.DataFrame:
    from doppel_lab.allie_v0e import split_name
    from doppel_lab.v0e_vec import games_path
    return pl.read_parquet(games_path(split_name(band), band), columns=["game_id", "white", "black", "white_elo",
                                                                        "black_elo", "moves", "t"])


def states(device: str, workers: int, batch: int, limit: int = 0, band: int = 1500):
    import torch

    from doppel_lab.allie_v0e import test_games
    from doppel_lab.engines import Maia3
    from doppel_lab.teacher import HIST, encode_game
    base_dir = knn_dir(band)
    out = base_dir.with_name(base_dir.name + "_smoke") if limit else base_dir
    out.mkdir(parents=True, exist_ok=True)
    g = games(band)
    tid = test_games(band)["game_id"].implode()
    if limit:                                           # smoke: some context games and some test games
        g = pl.concat([g.filter(~c("game_id").is_in(tid)).head(limit), g.filter(c("game_id").is_in(tid)).head(20)])
    is_test = g["game_id"].is_in(tid).to_numpy()
    n_per = g["moves"].list.len().to_numpy()
    off = np.r_[0, np.cumsum(n_per)].astype(np.int64)
    P = int(off[-1])
    tpos = np.concatenate([np.zeros(0, np.int64)] + [np.arange(off[i], off[i + 1]) for i in np.flatnonzero(is_test)])
    trow = np.full(P, -1, np.int64)
    trow[tpos] = np.arange(len(tpos))
    keys = np.lib.format.open_memmap(out / "keys.npy", "w+", np.float16, (P, D))
    played = np.zeros(P, np.int16)
    t_legal = np.full((len(tpos), MAXL), -1, np.int16)
    t_logp = np.full((len(tpos), MAXL), -np.inf, np.float16)
    model = Maia3("79m", device).model.eval()
    pooled = {}
    model.last_ln.register_forward_hook(lambda m, i, o: pooled.__setitem__("x", o))
    mean = comp = None
    if band != 1500:                                    # shared key space: the 1500 PCA
        pca = np.load(OUT / "pca.npz")
        mean = torch.from_numpy(pca["mean"]).to(device)
        comp = torch.from_numpy(pca["comp"]).to(device)
    rows = list(g.select("moves", "white_elo", "black_elo").iter_rows())
    t0 = time.time()
    with mp.get_context("fork").Pool(workers) as pool:
        for b0 in range(0, len(rows), BLOCK):
            enc = pool.map(encode_game, rows[b0:b0 + BLOCK], chunksize=32)
            nb = np.array([len(e[3]) for e in enc])
            ob = np.r_[0, np.cumsum(nb)]
            base = int(off[b0])
            planes = torch.from_numpy(np.concatenate([e[0] for e in enc])).to(device)
            hidx = torch.from_numpy(np.concatenate([np.clip(np.arange(n)[:, None] + np.arange(-HIST + 1, 1)[None, :],
                                                            0, None) + o for n, o in zip(nb, ob[:-1])])).to(device)
            legal = np.concatenate([e[1] for e in enc]).astype(np.int64)
            lcount = np.concatenate([e[2] for e in enc]).astype(np.int64)
            lstart = np.r_[0, np.cumsum(lcount)]
            es = torch.from_numpy(np.concatenate([e[4] for e in enc]).astype(np.int64)).to(device)
            eo = torch.from_numpy(np.concatenate([e[5] for e in enc]).astype(np.int64)).to(device)
            played[base:base + ob[-1]] = np.concatenate([e[3] for e in enc])
            feats = []
            for s in range(0, int(ob[-1]), batch):
                e_ = min(int(ob[-1]), s + batch)
                x = planes[hidx[s:e_]].float().permute(0, 2, 1, 3).reshape(e_ - s, 64, HIST * 12)
                x = torch.cat([x, torch.zeros(e_ - s, 64, 1, device=device)], 2)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                    logits = model(x, es[s:e_], eo[s:e_])[0].float()
                px = pooled["x"].float()
                feats.append(px.mean(1) if px.dim() == 3 else px)
                tr = trow[base + s:base + e_]
                for j in np.flatnonzero(tr >= 0):          # test positions: log-probs over legal moves
                    li = legal[lstart[s + j]:lstart[s + j + 1]]
                    lp = torch.log_softmax(logits[j, torch.from_numpy(li).to(device)], 0).cpu().numpy()
                    t_legal[tr[j], :len(li)] = li[:MAXL]
                    t_logp[tr[j], :len(li)] = lp[:MAXL]
            f = torch.cat(feats)
            if mean is None:                                # PCA on the first block
                mean = f.mean(0)
                _, _, v = torch.pca_lowrank(f - mean, q=D, center=False, niter=4)
                comp = v[:, :D]
                np.savez(out / "pca.npz", mean=mean.cpu().numpy(), comp=comp.cpu().numpy())
            k = torch.nn.functional.normalize((f - mean) @ comp, dim=1)
            keys[base:base + ob[-1]] = k.half().cpu().numpy()
            print(f"{b0 + len(enc):,}/{len(rows):,} games, {base + ob[-1]:,}/{P:,} positions, "
                  f"{(base + ob[-1]) / (time.time() - t0):,.0f}/s", flush=True)
    keys.flush()
    np.save(out / "played.npy", played)
    np.save(out / "off.npy", off)
    np.savez(out / "test.npz", pos=tpos, legal=t_legal, logp=t_logp)
    print(f"done -> {out}", flush=True)


def build_queries(band: int = 1500):
    """Per test side: query position rows (global index), store rows, and per-query metadata."""
    from doppel_lab.allie_v0e import CTX, test_games
    g = games(band).with_row_index("gi").with_columns(wl=c("white").str.to_lowercase(), bl=c("black").str.to_lowercase())
    t = test_games(band)
    cut = dict(zip(t["game_id"].to_list(), t["cut"].to_list()))
    test_ids = set(t["game_id"].to_list())
    off = np.load(knn_dir(band) / "off.npy")
    hist = g.filter(~c("game_id").is_in(list(test_ids)))
    hs = pl.concat([hist.select("gi", "t", player=c(col), col=pl.lit(i)) for i, col in enumerate(("wl", "bl"))])
    sides = []
    for r in g.filter(c("game_id").is_in(list(test_ids))).iter_rows(named=True):
        for col, name in ((0, r["wl"]), (1, r["bl"])):
            plies = [p for p in range(col, len(r["moves"]), 2) if 10 <= p < cut[r["game_id"]]]
            if plies:
                sides.append((r["game_id"], r["gi"], col, name, r["t"], plies))
    sd = pl.DataFrame([(s[0], s[3], s[4]) for s in sides], schema=["game_id", "player", "tq"], orient="row")
    ctx = (sd.with_row_index("si").join(hs, on="player").filter(c("t") < c("tq"))
           .sort("t", descending=True).group_by("si").head(CTX))
    store = {si: (gis, cols) for si, gis, cols in ctx.group_by("si").agg("gi", "col").iter_rows()}
    out = []
    for si, (gid, gi, col, name, tq, plies) in enumerate(sides):
        q = off[gi] + np.array(plies)
        gis, cols = store.get(si, ([], []))
        st = (np.concatenate([np.arange(off[a] + cc, off[a + 1], 2) for a, cc in zip(gis, cols)])
              if gis else np.zeros(0, np.int64))
        out.append((gid, np.array(plies), q, st))
    return out


def score():
    keys = np.load(OUT / "keys.npy", mmap_mode="r")
    played = np.load(OUT / "played.npy")
    tz = np.load(OUT / "test.npz")
    trow = {int(p): i for i, p in enumerate(tz["pos"])}
    T_legal, T_logp = tz["legal"], tz["logp"].astype(np.float32)
    sides = build_queries()
    NMAX = 32
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(sides))
    rows = []
    for si, (gid, plies, q, st) in enumerate(sides):
        for ctrl, store in ((0, st), (1, sides[perm[si]][3])):     # ctrl 1: another side's store
            Q = keys[q].astype(np.float32)
            if len(store):
                S = keys[store].astype(np.float32)
                sim = Q @ S.T
                nn = np.argsort(-sim, 1)[:, :NMAX]
                nsim = np.take_along_axis(sim, nn, 1)
                nmv = played[store][nn]
                if nn.shape[1] < NMAX:  # store smaller than NMAX: pad with empty neighbours (sim -1, move -1)
                    pad = NMAX - nn.shape[1]
                    nsim = np.pad(nsim, ((0, 0), (0, pad)), constant_values=-1.0)
                    nmv = np.pad(nmv, ((0, 0), (0, pad)), constant_values=-1)
            else:
                nsim = np.full((len(q), NMAX), -1.0, np.float32)
                nmv = np.full((len(q), NMAX), -1, np.int16)
            for j, p in enumerate(q):
                tr = trow[int(p)]
                rows.append((gid, int(plies[j]), ctrl, tr, nsim[j], nmv[j]))
    print(f"{len(rows):,} query rows", flush=True)
    gidl = [r[0] for r in rows]
    ply = np.array([r[1] for r in rows])
    ctrl = np.array([r[2] for r in rows])
    tr = np.array([r[3] for r in rows])
    nsim = np.stack([r[4] for r in rows])
    nmv = np.stack([r[5] for r in rows]).astype(np.int64)
    legal, logp = T_legal[tr].astype(np.int64), T_logp[tr]
    # played move of the query = played[pos]
    qplayed = played[tz["pos"][tr]].astype(np.int64)
    pm = np.exp(logp)                                                    # (R, MAXL), 0 on padding
    vote_col = np.full(nmv.shape, -1)                                     # neighbour move -> legal column
    for k in range(NMAX):
        hit = legal == nmv[:, k:k + 1]
        vote_col[:, k] = np.where(hit.any(1), hit.argmax(1), -1)
    truth = np.argmax(legal == qplayed[:, None], 1)
    maia_hit = pm.argmax(1) == truth
    mem = pl.read_parquet(ROOT / "data/allie/memory_positions.parquet").select("game_id", ply="k", hist="hist")
    key = pl.DataFrame({"game_id": gidl, "ply": ply})
    hist = key.join(mem, on=["game_id", "ply"], how="left")["hist"].fill_null(False).to_numpy()
    half = np.array([zlib.crc32(x.encode()) % 2 for x in gidl])

    def predict(n, tau, lam, thr, idx):
        w = np.where(vote_col[idx, :n] >= 0, np.exp((nsim[idx, :n] - 1) / tau), 0.0)
        pk = np.zeros((len(idx), MAXL))
        np.add.at(pk, (np.repeat(np.arange(len(idx)), n), np.clip(vote_col[idx, :n], 0, None).ravel()), w.ravel())
        tot = pk.sum(1, keepdims=True)
        use = (nsim[idx, 0] >= thr) & (tot[:, 0] > 0)
        pk = pk / np.maximum(tot, 1e-30)
        p = np.where(use[:, None], (1 - lam) * pm[idx] + lam * pk, pm[idx])
        return p.argmax(1) == truth[idx]

    grid = [(n, tau, lam, thr) for n in (4, 16, 32) for tau in (0.01, 0.05, 0.2) for lam in (0.3, 0.6, 0.9)
            for thr in (0.8, 0.9, 0.95, 0.98)]
    res = {"store_positions_median": float(np.median([len(s[3]) for s in sides])), "query_positions": int((ctrl == 0).sum()),
           "maia3_top1": float(maia_hit[ctrl == 0].mean())}
    for cname, cv in (("personal", 0), ("shuffled", 1)):
        out_hit = np.zeros(len(rows), bool)
        picks = {}
        for h in (0, 1):
            tune = np.flatnonzero((ctrl == cv) & (half == h))
            evalx = np.flatnonzero((ctrl == cv) & (half == 1 - h))
            best = max(grid, key=lambda gp: predict(*gp, tune).sum() - maia_hit[tune].sum())
            picks[f"tuned_on_half{h}"] = dict(zip(("n", "tau", "lam", "thr"), best))
            out_hit[evalx] = predict(*best, evalx)
        m = ctrl == cv

        def gain(sel):
            sel = sel & m
            return {"positions": int(sel.sum()), "gain_pts": 100 * float(out_hit[sel].mean() - maia_hit[sel].mean()),
                    "maia": float(maia_hit[sel].mean())}
        r = {"params": picks, "overall": gain(np.ones(len(rows), bool)),
             "exact_repeat_hist": gain(hist), "no_exact_history": gain(~hist), "by_ply": {}, "by_best_cos": {}}
        for lo, hi in ((10, 19), (20, 29), (30, 39), (40, 9999)):
            r["by_ply"][f"{lo}-{hi}"] = gain((ply >= lo) & (ply <= hi))
        for lo, hi in ((-1, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 0.98), (0.98, 0.995), (0.995, 2)):
            r["by_best_cos"][f"{lo}-{hi}"] = gain((nsim[:, 0] >= lo) & (nsim[:, 0] < hi))
        # game bootstrap of the overall gain
        gi = np.unique(np.array(gidl)[m], return_inverse=True)[1]
        d = out_hit[m].astype(float) - maia_hit[m]
        gs, gn = np.bincount(gi, d), np.bincount(gi)
        b = rng.integers(0, len(gs), (2000, len(gs)))
        bs = 100 * gs[b].sum(1) / gn[b].sum(1)
        r["overall"]["ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        res[cname] = r
    RESULTS.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("states")
    s.add_argument("--gpu-ok", action="store_true")
    s.add_argument("--workers", type=int, default=12)
    s.add_argument("--batch", type=int, default=2048)
    s.add_argument("--limit", type=int, default=0, help="CPU smoke: first N games, to knn_<band>_smoke/")
    s.add_argument("--band", type=int, default=1500)
    sub.add_parser("score")
    a = ap.parse_args()
    if a.cmd == "states":
        if not a.gpu_ok and not a.limit:
            raise SystemExit("refusing to touch the GPU without --gpu-ok (run through the warden)")
        states("cuda" if a.gpu_ok else "cpu", a.workers, a.batch, a.limit, a.band)
    else:
        score()


if __name__ == "__main__":
    main()

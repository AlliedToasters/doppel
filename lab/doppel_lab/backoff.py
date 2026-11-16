"""Policy-filtered backoff: do a player's moves at NEAR positions from their earlier games predict them beyond Maia,
where exact repeats can't reach? (SCIENCE.md §8: repertoire memory, one-piece perturbations.)

    python -m doppel_lab.backoff neighbours --gpu-ok      # warden: -> data/v0e/backoff_<band>.npz
    python -m doppel_lab.backoff neighbours --limit 20    # CPU smoke: first 20 target players
    python -m doppel_lab.backoff score                    # CPU: -> lab/results/backoff.json[band]

Queries: every own decision position (plies 0-79, the repeat_v0f visit table, one per (player, game, position))
of the 600 valrich target players, in games after their first. Candidates (online): the same player's positions in
strictly earlier games. Control: the same retrieval over a partner target player's positions in their games with
a lower within-player game rank (a fixed derangement of the 600), so the pool size matches; whatever the control
gains is generic "a similar position's move", and own minus control is the player-specific part.

Per query, Maia-3 79M (real move history, real Elos):
  mid   layer 4 of 8, mean over squares, L2-normalised; neighbours = the K nearest by cosine distance
  pol   Maia's move distribution; per neighbour the Jensen-Shannon divergence to the query's
Stored per neighbour: cosine distance, JS, whether its played move is legal at the query, whether it equals the
query's played move, whether the position key is the same (an exact repeat). score fits on half the players, over
  keep    JS <= tau (tau grid), legal, and optionally not an exact repeat (backoff-only)
  mix     p(played) = (sum_kept w 1[match] + a p_maia(played)) / (sum_kept w + a), w = 1 (kernel by distance: later)
and reports on the other half: gain over Maia in nats per query, coverage (share of queries with a kept neighbour),
own vs control, by ply.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

from doppel_lab.v0_data import targets_sample
from doppel_lab.v0_model import NMOVES, legal_mask

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "lab/results/backoff.json"
K = 16
c = pl.col


def out_path(band: int) -> Path:
    return ROOT / f"data/v0e/backoff_{band}.npz"


def queries(band: int, limit: int):
    from doppel_lab.repeat_v0f import visits
    from doppel_lab.v0e_data import Split
    sp = Split("valrich", band)
    v = visits(sp, band)
    tp = set(pl.read_parquet(ROOT / f"data/v0e/valrich_players_{band}.parquet")["player"].to_list())
    tid = sorted(i for i, n in enumerate(np.asarray(sp.players)) if n in tp)
    if limit:
        tid = tid[:limit]
    v = (v.filter(c("player").is_in(tid))
         .with_columns(grank=c("t").rank("dense").over("player").cast(pl.Int32))
         .sort("player", "t", "g", "ply"))
    return sp, v, tid


torch.multiprocessing.set_sharing_strategy("file_system")


def collate(samples):
    return {k: torch.from_numpy(np.concatenate([s[k] for s in samples]))
            for k in ("tok", "legal", "played", "elo_self", "elo_opp", "ply", "game")}


class Positions:
    def __init__(self, sp, items):
        self.sp, self.items = sp, items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        g, col, plies = self.items[i]
        return targets_sample(self.sp.ix, g, col, plies, None, len(plies), np.random.default_rng(0))


@torch.no_grad()
def forward(sp, v, device, workers, batch):
    from doppel_lab.engines import Maia3
    from doppel_lab.v0_model import Trunk
    trunk = Trunk(Maia3("79m", device).model).to(device)
    layers = next(mod for mod in trunk.m.transformer.modules() if isinstance(mod, torch.nn.ModuleList))
    feats = {}
    layers[len(layers) // 2].register_forward_hook(
        lambda mod, i, o: feats.__setitem__("mid", o[0] if isinstance(o, tuple) else o))
    side = v.group_by("g", (c("ply") % 2).alias("col"), maintain_order=True).agg(c("ply").sort())
    items = [(int(g), int(col), [int(p) for p in pl_]) for g, col, pl_ in side.iter_rows()]
    ld = torch.utils.data.DataLoader(Positions(sp, items), batch_size=batch, shuffle=False, collate_fn=collate,
                                     num_workers=workers, prefetch_factor=4 if workers else None,
                                     multiprocessing_context="fork" if workers else None)
    acc = {k: [] for k in ("game", "ply", "emb", "lidx", "llp", "played")}
    t0 = time.time()
    for n, b in enumerate(ld):
        tok = b["tok"].to(device).float()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            logits, _, _ = trunk(tok, b["elo_self"].to(device), b["elo_opp"].to(device))
        legal = b["legal"].to(device)
        mask = legal_mask(legal)
        lp = logits.float().masked_fill(~mask, -1e9).log_softmax(-1)
        e = feats["mid"].float()
        e = e.mean(1) if e.dim() == 3 else e
        acc["emb"].append(torch.nn.functional.normalize(e, dim=-1).half().cpu())
        li = legal.long()
        acc["lidx"].append(legal.cpu().clone())
        acc["llp"].append(torch.where(li >= 0, lp.gather(1, li.clamp(min=0)), -1e4).half().cpu())
        acc["played"].append(b["played"].clone())       # clone: don't hold the workers' shared-memory fds
        acc["game"].append(b["game"].clone())
        acc["ply"].append(b["ply"].clone())
        if n % 200 == 0:
            print(f"  batch {n}: {sum(len(x) for x in acc['played']):,} positions ({time.time() - t0:.0f}s)", flush=True)
    return {k: torch.cat(x) for k, x in acc.items()}


def dense(lidx, llp, device):
    """(n, NMOVES) probabilities from the padded legal-move log-probs."""
    P = torch.zeros(len(lidx), NMOVES + 1, device=device)
    li = lidx.long().to(device)
    li = torch.where(li >= 0, li, NMOVES)
    P.scatter_(1, li, llp.float().to(device).exp())
    return P[:, :NMOVES]


def js_rows(Pq, Pc):
    """JS divergence between Pq (n, M) and each of Pc (n, k, M)."""
    Pq = Pq[:, None]
    M = 0.5 * (Pq + Pc)
    lm = (M + 1e-30).log()
    kl = lambda A: torch.where(A > 0, A * ((A + 1e-30).log() - lm), 0.0).sum(-1)
    return 0.5 * kl(Pq) + 0.5 * kl(Pc)


def neighbours(band: int, device: str, workers: int, batch: int, limit: int):
    t0 = time.time()
    sp, v, tid = queries(band, limit)
    print(f"{len(tid)} target players, {v.height:,} positions ({time.time() - t0:.0f}s)", flush=True)
    r = forward(sp, v, device, workers, batch)
    sc = pl.DataFrame({"g": r["game"].numpy().astype(np.int32), "ply": r["ply"].numpy().astype(np.int32),
                       "row": np.arange(len(r["game"]), dtype=np.int64)})
    x = v.join(sc, on=["g", "ply"], how="inner").sort("player", "t", "g", "ply")
    assert x.height == v.height == sc.height, (x.height, v.height, sc.height)
    row = x["row"].to_numpy()
    emb, lidx, llp, played = r["emb"][row], r["lidx"][row], r["llp"][row], r["played"][row]
    pid, grank, key = x["player"].to_numpy(), x["grank"].to_numpy(), x["key"].to_numpy()
    starts = {p: (s, e) for p, s, e in zip(*np.unique(pid, return_index=True),
                                           np.r_[np.unique(pid, return_index=True)[1][1:], len(pid)])}
    players = sorted(starts)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(players))
    while (perm == np.arange(len(players))).any():
        perm = rng.permutation(len(players))
    partner = {players[i]: players[perm[i]] for i in range(len(players))}
    nq = len(pid)
    out = {f"{pool}_{f}": np.zeros((nq, K), dt) for pool in ("own", "ctl")
           for f, dt in (("dist", np.float16), ("js", np.float16), ("legal", bool), ("match", bool),
                         ("same", bool), ("valid", bool))}
    # maia log-prob of the played move at each query
    pl_lp = torch.where(lidx.long() == played[:, None], llp.float(), -1e4).max(1).values.numpy()
    for n, p in enumerate(players):
        s, e = starts[p]
        Eq = emb[s:e].to(device).float()
        Pq = dense(lidx[s:e], llp[s:e], device)
        for pool, q in (("own", p), ("ctl", partner[p])):
            cs, ce = starts[q]
            Ec = emb[cs:ce].to(device).float()
            D = 1 - Eq @ Ec.T                                   # (nq_p, nc)
            allowed = torch.from_numpy(grank[cs:ce][None, :] < grank[s:e][:, None]).to(device)
            D = torch.where(allowed, D, 9.0)
            k = min(K, D.shape[1])
            dv, di = D.topk(k, dim=1, largest=False)
            valid = dv < 9.0
            Pc_all = dense(lidx[cs:ce], llp[cs:ce], device)
            jsv = torch.cat([js_rows(Pq[i:i + 256], Pc_all[di[i:i + 256]]) for i in range(0, len(Pq), 256)])
            cmove = played[cs:ce].to(device)[di]                # neighbour's played move (side-relative index)
            qleg = lidx[s:e].long().to(device)
            legal = (cmove[:, :, None] == qleg[:, None, :]).any(-1)
            match = cmove == played[s:e].to(device)[:, None]
            same = torch.from_numpy(key[cs:ce]).to(device)[di] == torch.from_numpy(key[s:e]).to(device)[:, None]
            for f, val in (("dist", dv), ("js", jsv), ("legal", legal), ("match", match), ("same", same),
                           ("valid", valid)):
                out[f"{pool}_{f}"][s:e, :k] = val.cpu().numpy()
        if n % 100 == 0:
            print(f"  player {n}/{len(players)} ({time.time() - t0:.0f}s)", flush=True)
    np.savez(out_path(band) if not limit else out_path(band).with_suffix(".smoke.npz"),
             player=pid, g=x["g"].to_numpy(), ply=x["ply"].to_numpy(), grank=grank,
             N_prev=x["N_prev"].to_numpy(), n_prev=x["n_prev"].to_numpy(), maia=pl_lp, **out)
    print(f"-> {out_path(band) if not limit else out_path(band).with_suffix('.smoke.npz')} ({time.time() - t0:.0f}s)", flush=True)


def ci(pid, gain, rng):
    u, inv = np.unique(pid, return_inverse=True)
    s, n = np.bincount(inv, gain), np.bincount(inv)
    b = [s[i].sum() / n[i].sum() for i in (rng.integers(0, len(u), len(u)) for _ in range(500))]
    return [round(float(v), 4) for v in np.percentile(b, [2.5, 97.5])]


def score(band: int, smoke: bool):
    f = np.load(out_path(band).with_suffix(".smoke.npz") if smoke else out_path(band))
    pid, ply, maia = f["player"], f["ply"], f["maia"].astype(np.float64)
    pm = np.exp(maia)
    q = f["grank"] > 1
    test = (np.unique(pid, return_inverse=True)[1] * 2654435761 % 1000) < 500
    rng = np.random.default_rng(0)
    TAUS, AS, KS = (0.01, 0.02, 0.05, 0.1, 0.2), (0.5, 1, 2, 4, 8, 16), (1, 4, 16)

    def gain(pool, tau, a, k, exact):
        keep = (f[f"{pool}_valid"][:, :k] & f[f"{pool}_legal"][:, :k] & (f[f"{pool}_js"][:, :k] <= tau))
        if not exact:
            keep &= ~f[f"{pool}_same"][:, :k]
        hit = (keep & f[f"{pool}_match"][:, :k]).sum(1)
        tot = keep.sum(1)
        g = np.log((hit + a * pm) / (tot + a)) - maia
        return g, tot > 0

    out = {"band": band, "queries": int(q.sum()), "players": int(len(np.unique(pid[q])))}
    plybins = (("0-9", ply < 10), ("10-19", (ply >= 10) & (ply < 20)), ("20-39", (ply >= 20) & (ply < 40)),
               ("40-79", ply >= 40))
    for exact in (False, True):
        tag = "with_exact" if exact else "backoff_only"
        # fit (tau, a, k) on the fit half for own; apply the same setting to control
        best = max(((t, a, k) for t in TAUS for a in AS for k in KS),
                   key=lambda s: gain("own", *s, exact)[0][q & ~test].mean())
        res = {"tau": best[0], "a": best[1], "k": best[2]}
        for pool in ("own", "ctl"):
            g, cov = gain(pool, *best, exact)
            m = q & test
            res[pool] = {"gain": round(float(g[m].mean()), 4), "ci": ci(pid[m], g[m], rng),
                         "coverage": round(float(cov[m].mean()), 4),
                         "gain_covered": round(float(g[m & cov].mean()), 4) if (m & cov).any() else None,
                         "by_ply": {b: {"n": int((m & bm).sum()), "gain": round(float(g[m & bm].mean()), 4),
                                        "coverage": round(float(cov[m & bm].mean()), 4)}
                                    for b, bm in plybins if (m & bm).any()}}
        go, _ = gain("own", *best, exact)
        gc, _ = gain("ctl", *best, exact)
        m = q & test
        res["own_minus_ctl"] = {"gain": round(float((go - gc)[m].mean()), 4), "ci": ci(pid[m], (go - gc)[m], rng),
                                "by_ply": {b: round(float((go - gc)[m & bm].mean()), 4) for b, bm in plybins
                                           if (m & bm).any()}}
        # the same, with the control's own best setting (so the control isn't handicapped by own's fit)
        bc = max(((t, a, k) for t in TAUS for a in AS for k in KS),
                 key=lambda s: gain("ctl", *s, exact)[0][q & ~test].mean())
        g2, _ = gain("ctl", *bc, exact)
        res["ctl_own_fit"] = {"tau": bc[0], "a": bc[1], "k": bc[2], "gain": round(float(g2[m].mean()), 4),
                              "by_ply": {b: round(float(g2[m & bm].mean()), 4) for b, bm in plybins if (m & bm).any()}}
        out[tag] = res
    # refit per ply range (the global fit is dominated by the opening): backoff only, own vs control
    out["refit_by_ply"] = {}
    for b, bm in (("10-19", (ply >= 10) & (ply < 20)), ("20+", ply >= 20), ("40+", ply >= 40)):
        sel = q & bm
        e = {}
        for pool in ("own", "ctl"):
            st = max(((t, a, k) for t in TAUS for a in AS for k in KS),
                     key=lambda s_: gain(pool, *s_, False)[0][sel & ~test].mean())
            g, cov = gain(pool, *st, False)
            m = sel & test
            e[pool] = {"tau": st[0], "a": st[1], "k": st[2], "gain": round(float(g[m].mean()), 5),
                       "ci": ci(pid[m], g[m], rng), "coverage": round(float(cov[m].mean()), 4),
                       "gain_covered": round(float(g[m & cov].mean()), 4) if (m & cov).any() else None}
        out["refit_by_ply"][b] = e
    # match rate of a kept-type neighbour's move with the query's played move, own vs control, by ply and JS
    out["match_by_ply_js"] = {}
    for b, bm in (("0-9", ply < 10), ("10-19", (ply >= 10) & (ply < 20)), ("20+", ply >= 20)):
        e = {}
        for pool in ("own", "ctl"):
            vv = f[f"{pool}_valid"] & f[f"{pool}_legal"] & ~f[f"{pool}_same"] & (q & bm)[:, None]
            jj, mm = f[f"{pool}_js"].astype(np.float32), f[f"{pool}_match"]
            e[pool] = {f"<={hi}": [round(float(mm[vv & (jj > lo) & (jj <= hi)].mean()), 4),
                                   int((vv & (jj > lo) & (jj <= hi)).sum())]
                       for lo, hi in ((-1, 0.01), (0.01, 0.05), (0.05, 0.1), (0.1, 0.2))
                       if (vv & (jj > lo) & (jj <= hi)).any()}
        out["match_by_ply_js"][b] = e
    # how the JS filter works on own neighbours: match rate of the neighbour's move by JS bin (k = 16, non-exact)
    v = f["own_valid"] & f["own_legal"] & ~f["own_same"] & q[:, None]
    js, mt = f["own_js"].astype(np.float32), f["own_match"]
    out["match_rate_by_js"] = {f"<={hi}": round(float(mt[v & (js > lo) & (js <= hi)].mean()), 4)
                               for lo, hi in ((-1, 0.01), (0.01, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.7))
                               if (v & (js > lo) & (js <= hi)).any()}
    vc = f["ctl_valid"] & f["ctl_legal"] & ~f["ctl_same"] & q[:, None]
    jc, mc = f["ctl_js"].astype(np.float32), f["ctl_match"]
    out["ctl_match_rate_by_js"] = {f"<={hi}": round(float(mc[vc & (jc > lo) & (jc <= hi)].mean()), 4)
                                   for lo, hi in ((-1, 0.01), (0.01, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.7))
                                   if (vc & (jc > lo) & (jc <= hi)).any()}
    res = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    res[f"{band}{'_smoke' if smoke else ''}"] = out
    RESULTS.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["neighbours", "score"])
    ap.add_argument("--band", type=int, default=1500)
    ap.add_argument("--gpu-ok", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.cmd == "neighbours":
        neighbours(a.band, "cuda" if a.gpu_ok else "cpu", a.workers, a.batch, a.limit)
    else:
        score(a.band, a.smoke)

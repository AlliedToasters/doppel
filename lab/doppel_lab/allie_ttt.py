"""Test-time per-player adapters on the Allie test set (the lora_probe recipe applied to real test sides).

    python -m doppel_lab.allie_ttt run --band 1500 --gpu-ok      # GPU (warden) -> data/allie/ttt_<band>.npz + results

Per test side: its context = the side's 40 most recent earlier 2022 games (allie_knn.build_queries' rule: strictly
before the test game, never a test game). Val = the 5 most recent of those, train = the other 35. A rank-8 LoRA on
the MLP of Maia-3 blocks 5-8 (lora_probe.Lora) is trained per side, epoch picked per side on val CE (epoch 0 = Maia-3).
Sides are trained CHUNK at a time (the probe's 199 players per batch, so the per-player optimiser dynamics match).
Scored: the Allie protocol positions of the side's test game (ply >= 10, before the first mover under 30 s).
Control: each side's positions through the next side's adapter in its chunk.
Saved per position: game_id, ply, hits (Maia-3 / own / control) and the own adapter's log-probs over the legal moves
in python-chess order (= allie_knn test.npz order), for stacking with the lookup and z.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
CHUNK = 200
NVAL = 5
c = pl.col


def side_rows(band: int) -> tuple[pl.DataFrame, dict]:
    """One row per (side, game it uses): the side's context games (role val / train) and its test game."""
    from doppel_lab.allie_knn import games
    from doppel_lab.allie_v0e import CTX, test_games
    g = games(band).with_row_index("gi").with_columns(wl=c("white").str.to_lowercase(), bl=c("black").str.to_lowercase())
    t = test_games(band)
    cut = dict(zip(t["game_id"].to_list(), t["cut"].to_list()))
    test_ids = set(t["game_id"].to_list())
    hist = g.filter(~c("game_id").is_in(list(test_ids)))
    hs = pl.concat([hist.select("gi", "t", player=c(col), col=pl.lit(i, pl.Int64)) for i, col in enumerate(("wl", "bl"))])
    sides = []
    for r in g.filter(c("game_id").is_in(list(test_ids))).iter_rows(named=True):
        for col, name in ((0, r["wl"]), (1, r["bl"])):
            if any(10 <= p < cut[r["game_id"]] for p in range(col, len(r["moves"]), 2)):
                sides.append((r["gi"], col, name, r["t"]))
    sd = pl.DataFrame(sides, schema=["tgi", "tcol", "player", "tq"], orient="row").with_row_index("pid")
    ctx = (sd.join(hs, on="player").filter(c("t") < c("tq")).sort(["pid", "t"], descending=[False, True])
           .group_by("pid", maintain_order=True).head(CTX)
           .with_columns(r=pl.int_range(pl.len()).over("pid")))
    ctx = ctx.select("pid", "gi", "col", role=pl.when(c("r") < NVAL).then(pl.lit("val")).otherwise(pl.lit("train")),
                     tr_rank=c("r") - NVAL, j=c("r"))
    tst = sd.select("pid", gi=c("tgi"), col=c("tcol"), role=pl.lit("test"), tr_rank=pl.lit(-1, pl.Int64),
                    j=pl.lit(-1, pl.Int64))
    rows = pl.concat([ctx.with_columns(c("pid").cast(pl.Int64), c("gi").cast(pl.Int64)),
                      tst.with_columns(c("pid").cast(pl.Int64), c("gi").cast(pl.Int64))], how="vertical_relaxed")
    rows = rows.join(g.select(c("gi").cast(pl.Int64), "game_id", "white_elo", "black_elo", "moves"), on="gi")
    return rows.sort("pid", "j"), cut


def use_store200():
    """Context = the 200 most recent earlier games (lookup_store's selection; run `lookup_store select --band` first)."""
    from doppel_lab import allie_knn, allie_v0e, lookup_store
    allie_knn.games, allie_v0e.CTX = lookup_store.games, lookup_store.CTX


def run(band: int, device: str, workers: int, epochs: int, lr: float, wd: float, bs: int, limit: int, tag: str = ""):
    import torch

    from doppel_lab.engines import Maia3
    from doppel_lab.lora_probe import Lora, Positions, encode_all, logits, per_position
    rows, cut = side_rows(band)
    n_sides = int(rows["pid"].max()) + 1
    if limit:
        n_sides = min(n_sides, limit)
    print(f"band {band}: {n_sides:,} sides, {rows.filter(c('role') != 'test').height:,} context rows", flush=True)
    model = Maia3("79m", device).model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    out = {k: [] for k in ("game_id", "ply", "maia", "own", "ctrl", "best_ep")}
    lps, legs = [], []
    t0 = time.time()
    for c0 in range(0, n_sides, CHUNK):
        d = rows.filter((c("pid") >= c0) & (c("pid") < min(n_sides, c0 + CHUNK))).with_columns(pid=c("pid") - c0)
        n = int(d["pid"].max()) + 1
        enc = encode_all(d, workers)
        P = Positions(d, enc)
        lora = Lora(model, n, device)
        opt = torch.optim.AdamW(lora.params, lr=lr, weight_decay=wd)
        tr, va = np.flatnonzero(P.role == "train"), np.flatnonzero(P.role == "val")
        own = lambda idx: P.pid[idx]
        lp0, _ = per_position(model, lora, P, va, None, device, torch.float16)
        cnt = np.maximum(np.bincount(P.pid[va], minlength=n), 1)
        best_ce = np.bincount(P.pid[va], -lp0, n) / cnt
        best, best_ep = lora.snapshot(), np.zeros(n, np.int64)
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
            ce = np.bincount(P.pid[va], -lpv, n) / cnt
            better = (ce < best_ce) & (np.bincount(P.pid[va], minlength=n) > 0)
            m = torch.from_numpy(better).to(device)
            for b_, s_ in zip(best, lora.snapshot()):
                b_[m] = s_[m]
            best_ce, best_ep = np.where(better, ce, best_ce), np.where(better, ep, best_ep)
        lora.load_rows(best, np.ones(n, bool))
        gid_of = d["game_id"].to_numpy().astype(str)
        te = np.flatnonzero((P.role == "test") & (P.ply >= 10))
        te = te[np.array([P.ply[i] < cut[gid_of[P.game[i]]] for i in te], bool)]
        _, h_m = per_position(model, lora, P, te, None, device, torch.float16)
        _, h_o = per_position(model, lora, P, te, own, device, torch.float16)
        _, h_c = per_position(model, lora, P, te, lambda idx: (P.pid[idx] + 1) % n, device, torch.float16)
        with torch.no_grad():                                   # full legal-row log-probs of the own adapter
            for s in range(0, len(te), 1024):
                idx = te[s:s + 1024]
                b = P.batch(idx, device)
                lp = logits(model, b, lora, torch.from_numpy(P.pid[idx]).to(device), torch.float16).cpu().numpy()
                for j, i in enumerate(idx):
                    lps.append(lp[j, :len(P.legal[i])].astype(np.float16)); legs.append(P.legal[i])
        for h in lora.hooks:
            h.remove()
        out["game_id"].append(gid_of[P.game[te]]); out["ply"].append(P.ply[te])
        out["maia"].append(h_m); out["own"].append(h_o); out["ctrl"].append(h_c)
        out["best_ep"].append(best_ep[P.pid[te]])
        done = min(n_sides, c0 + CHUNK)
        hm, ho = np.concatenate(out["maia"]), np.concatenate(out["own"])
        print(f"{done:,}/{n_sides:,} sides, {len(hm):,} positions, own {100 * (ho.mean() - hm.mean()):+.3f} pts, "
              f"ctrl {100 * (np.concatenate(out['ctrl']).mean() - hm.mean()):+.3f}, best epochs "
              f"{np.bincount(best_ep, minlength=epochs + 1).tolist()}, {time.time() - t0:.0f}s", flush=True)
    r = {k: np.concatenate(v) for k, v in out.items()}
    L = max(len(x) for x in legs)
    lp_arr = np.full((len(lps), L), -np.inf, np.float16)
    lg_arr = np.full((len(lps), L), -1, np.int16)
    for i, (a, b) in enumerate(zip(lps, legs)):
        lp_arr[i, :len(a)] = a; lg_arr[i, :len(b)] = b
    sfx = tag if not limit else tag + "_smoke"
    np.savez(ROOT / f"data/allie/ttt_{band}{sfx}.npz", logp=lp_arr, legal=lg_arr, **r)
    summary(band, sfx)


def summary(band: int, sfx: str = ""):
    z = np.load(ROOT / f"data/allie/ttt_{band}{sfx}.npz", allow_pickle=True)
    gid, ply, m, o, cc = z["game_id"], z["ply"], z["maia"], z["own"], z["ctrl"]
    gi = np.unique(gid, return_inverse=True)[1]
    rng = np.random.default_rng(0)

    def gain(h, sel):
        d = h[sel].astype(float) - m[sel]
        g = np.unique(gi[sel], return_inverse=True)[1]
        gs, gn = np.bincount(g, d), np.bincount(g)
        b = rng.integers(0, len(gs), (1000, len(gs)))
        bs = 100 * gs[b].sum(1) / gn[b].sum(1)
        return {"positions": int(sel.sum()), "top1_gain_pts": 100 * float(d.mean()),
                "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    res = {"band": band, "positions": len(m), "maia_top1": float(m.mean()),
           "best_epoch_hist": np.bincount(z["best_ep"]).tolist()}
    for name, h in (("own", o), ("control", cc)):
        res[name] = {"all": gain(h, np.ones(len(m), bool)), "by_ply": {}}
        for lo, hi in ((10, 19), (20, 29), (30, 39), (40, 9999)):
            res[name]["by_ply"][f"{lo}-{hi}"] = gain(h, (ply >= lo) & (ply <= hi))
    if not sfx.endswith("_smoke"):
        (ROOT / f"lab/results/allie_ttt_{band}{sfx}.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--band", type=int, default=1500)
    r.add_argument("--gpu-ok", action="store_true")
    r.add_argument("--workers", type=int, default=12)
    r.add_argument("--epochs", type=int, default=3)
    r.add_argument("--lr", type=float, default=1e-3)
    r.add_argument("--wd", type=float, default=0.01)
    r.add_argument("--bs", type=int, default=512)
    r.add_argument("--limit", type=int, default=0, help="smoke: first N sides")
    r.add_argument("--ctx", type=int, default=40, choices=(40, 200), help="200: lookup_store's context")
    s = sub.add_parser("summary")
    s.add_argument("--band", type=int, default=1500)
    s.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.cmd == "run":
        if not a.gpu_ok and not a.limit:
            raise SystemExit("refusing to touch the GPU without --gpu-ok (run through the warden)")
        tag = ""
        if a.ctx == 200:
            use_store200()
            tag = "_ctx200"
        run(a.band, "cuda" if a.gpu_ok else "cpu", a.workers, a.epochs, a.lr, a.wd, a.bs, a.limit, tag)
    else:
        summary(a.band, a.tag)


if __name__ == "__main__":
    main()

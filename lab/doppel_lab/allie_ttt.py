"""Test-time per-player adapters on the Allie test set (the lora_probe recipe applied to real test sides).

    python -m doppel_lab.allie_ttt run --band 1500 --gpu-ok      # GPU (warden) -> data/allie/ttt_<band>.npz + results

Per test side: its context = the side's 40 most recent earlier 2022 games (allie_knn.build_queries' rule: strictly
before the test game, never a test game). Val = the 5 most recent of those, train = the other 35. A rank-8 LoRA on
the MLP of Maia-3 blocks 5-8 (lora_probe.Lora) is trained per side, epoch picked per side on val CE (epoch 0 = Maia-3).
Sides are trained CHUNK at a time (the probe's 199 players per batch, so the per-player optimiser dynamics match).
Scored: the Allie protocol positions of the side's test game (ply >= 10, before the first mover under 30 s).
Control: each side's positions through the next side's adapter in its chunk.
Saved per side (default; --no-save-adapters to skip): the chosen adapter, everything a Doppel app needs on top of
Maia-3 -> data/allie/adapters_<band><tag>/chunk_*.npz + recipe.json (see save_adapters).
Saved per position: game_id, ply, hits (Maia-3 / own / control) and the own adapter's log-probs over the legal moves
in python-chess order (= allie_knn test.npz order), for stacking with the lookup and z.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
CHUNK = 200
NVAL = 5
c = pl.col


def side_rows(band: int, nfit: int = 0) -> tuple[pl.DataFrame, dict]:
    """One row per (side, game it uses): the side's context games (role val / train) and its test game.
    nfit > 0: the nfit most recent context games become role "fit" (held out from training and epoch choice, for
    fitting post-hoc knobs such as the residual weight off the test set); val = the NVAL before them."""
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
    ctx = ctx.select("pid", "gi", "col", role=pl.when(c("r") < nfit).then(pl.lit("fit"))
                     .when(c("r") < nfit + NVAL).then(pl.lit("val")).otherwise(pl.lit("train")),
                     tr_rank=c("r") - nfit - NVAL, j=c("r"))
    tst = sd.select("pid", gi=c("tgi"), col=c("tcol"), role=pl.lit("test"), tr_rank=pl.lit(-1, pl.Int64),
                    j=pl.lit(-1, pl.Int64))
    rows = pl.concat([ctx.with_columns(c("pid").cast(pl.Int64), c("gi").cast(pl.Int64)),
                      tst.with_columns(c("pid").cast(pl.Int64), c("gi").cast(pl.Int64))], how="vertical_relaxed")
    rows = rows.join(g.select(c("gi").cast(pl.Int64), "game_id", "white_elo", "black_elo", "moves"), on="gi")
    return rows.sort("pid", "j"), cut


def save_adapters(dirp: Path, c0: int, lora, d: pl.DataFrame, best_ep, best_ce, recipe: dict):
    """The deployable artifact per side: the chosen LoRA (fp16) + which side it is (test game id and colour; no
    usernames). p{2k} = A (n, r, in), p{2k+1} = B (n, out, r), for (block, layer) in recipe order; the hook adds
    scale * B @ (A @ x) to the layer's output. Epoch-0 sides have B = 0 (= Maia-3). Written before the chunk's part
    file, atomically, so a resumed chunk always has its adapters."""
    dirp.mkdir(parents=True, exist_ok=True)
    rj = dirp / "recipe.json"
    if not rj.exists():
        rj.write_text(json.dumps(recipe, indent=1) + "\n")
    t = d.filter(c("role") == "test").sort("pid")
    arr = {f"p{i}": p.detach().float().cpu().numpy().astype(np.float16) for i, p in enumerate(lora.params)}
    f = dirp / f"chunk_{c0:06d}_{CHUNK}.npz"
    tmp = f.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, **arr, pid=t["pid"].to_numpy() + c0, test_game_id=t["game_id"].to_numpy().astype(str),
                 test_col=t["col"].to_numpy(), best_ep=best_ep, val_ce=best_ce.astype(np.float32))
    tmp.rename(f)


def use_store200():
    """Context = the 200 most recent earlier games (lookup_store's selection; run `lookup_store select --band` first)."""
    from doppel_lab import allie_knn, allie_v0e, lookup_store
    allie_knn.games, allie_v0e.CTX = lookup_store.games, lookup_store.CTX


def run(band: int, device: str, workers: int, epochs: int, lr: float, wd: float, bs: int, limit: int, tag: str = "",
        nfit: int = 0, sample: int = 0, keep_adapters: bool = True):
    import torch

    from doppel_lab.engines import Maia3
    from doppel_lab import lora_probe
    from doppel_lab.lora_probe import Lora, Positions, encode_all, logits, per_position
    rows, cut = side_rows(band, nfit)
    if sample:                                              # a random subset of sides, pids renumbered 0..sample-1
        keep = np.sort(np.random.default_rng(band).choice(int(rows["pid"].max()) + 1, sample, replace=False))
        rows = rows.join(pl.DataFrame({"pid": keep.astype(np.int64), "new": np.arange(sample, dtype=np.int64)}),
                         on="pid").with_columns(pid=c("new")).drop("new").sort("pid", "j")
        tag += f"_fit{nfit}_s{sample}"
    n_sides = int(rows["pid"].max()) + 1
    if limit:
        n_sides = min(n_sides, limit)
    print(f"band {band}: {n_sides:,} sides, {rows.filter(c('role') != 'test').height:,} context rows", flush=True)
    model = Maia3("79m", device).model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    out = {k: [] for k in ("game_id", "ply", "maia", "own", "ctrl", "best_ep")}
    lps, legs = [], []
    fit = {k: [] for k in ("pid", "ply", "truth", "lens", "lm", "la", "lg")}   # per fit position, nfit > 0
    sfx = tag if not limit else tag + "_smoke"
    parts = ROOT / f"data/allie/ttt_{band}{sfx}_parts"      # one file per finished chunk: a killed run resumes
    parts.mkdir(parents=True, exist_ok=True)
    adir = ROOT / f"data/allie/adapters_{band}{sfx}"
    recipe = {"base": "maia3-79m (doppel_lab.engines.Maia3('79m'))", "blocks": list(lora_probe.BLOCKS),
              "layers": ["linear1", "linear2"], "rank": lora_probe.R, "scale": 1.0,
              "param_order": "for block in blocks: for layer in layers: A (n, rank, in), B (n, out, rank)",
              "forward": "layer_out + scale * B @ (A @ layer_in), Maia-3 Elo inputs unchanged; final log-probs = "
                         "Maia-3 + residual_weight * (adapted - Maia-3), renormalised over legal moves",
              "train": {"epochs": epochs, "lr": lr, "wd": wd, "bs": bs, "nval": NVAL, "nfit": nfit,
                        "context": "lookup_store 200 most recent" if tag.startswith("_ctx200") else "40 most recent",
                        "epoch_choice": "per side, val CE; epoch 0 = Maia-3"},
              "residual_weight": 0.625,      # deploy: Maia-3 + w * (adapter - Maia-3); ttt_weight fit (pre-test games)
              "band": band}
    t0 = time.time()
    for c0 in range(0, n_sides, CHUNK):
        part = parts / f"chunk_{c0:06d}_{CHUNK}.npz"
        if part.exists():
            z = np.load(part)
            for k in out:
                out[k].append(z[k])
            lps.extend(np.split(z["lp_flat"], np.cumsum(z["lens"])[:-1]))
            legs.extend(np.split(z["lg_flat"], np.cumsum(z["lens"])[:-1]))
            for k in fit:
                if f"fit_{k}" in z.files:
                    fit[k].append(z[f"fit_{k}"])
            print(f"chunk {c0:,}: resumed from {part.name}", flush=True)
            continue
        n_lp = len(lps)
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
        if keep_adapters:
            save_adapters(adir, c0, lora, d, best_ep, best_ce, recipe)
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
        if nfit:                                                # Maia-3 and own-adapter rows on the held-out fit games
            fi = np.flatnonzero((P.role == "fit") & (P.ply >= 10))
            fl = {k: [] for k in ("lm", "la", "lg")}
            tr_ = []
            with torch.no_grad():
                for s in range(0, len(fi), 1024):
                    idx = fi[s:s + 1024]
                    b = P.batch(idx, device)
                    l0 = logits(model, b, lora, None, torch.float16).cpu().numpy()
                    l1 = logits(model, b, lora, torch.from_numpy(P.pid[idx]).to(device), torch.float16).cpu().numpy()
                    tr_.append(b[4].cpu().numpy())
                    for j, i in enumerate(idx):
                        k_ = len(P.legal[i])
                        fl["lm"].append(l0[j, :k_].astype(np.float16)); fl["la"].append(l1[j, :k_].astype(np.float16))
                        fl["lg"].append(np.asarray(P.legal[i]))
            fit["pid"].append(P.pid[fi] + c0); fit["ply"].append(P.ply[fi])
            fit["truth"].append(np.concatenate(tr_) if tr_ else np.zeros(0, np.int64))
            fit["lens"].append(np.array([len(x) for x in fl["lm"]], np.int64))
            for k in ("lm", "la", "lg"):
                fit[k].append(np.concatenate(fl[k]) if fl[k] else np.zeros(0))
        for h in lora.hooks:
            h.remove()
        out["game_id"].append(gid_of[P.game[te]]); out["ply"].append(P.ply[te])
        out["maia"].append(h_m); out["own"].append(h_o); out["ctrl"].append(h_c)
        out["best_ep"].append(best_ep[P.pid[te]])
        new_lp, new_lg = lps[n_lp:], legs[n_lp:]
        tmp = part.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            np.savez(f, **{k: v[-1] for k, v in out.items()}, lens=np.array([len(x) for x in new_lp], np.int64),
                     lp_flat=np.concatenate(new_lp) if new_lp else np.zeros(0, np.float16),
                     lg_flat=np.concatenate(new_lg) if new_lg else np.zeros(0, np.int64),
                     **({f"fit_{k}": v[-1] for k, v in fit.items()} if nfit else {}))
        tmp.rename(part)
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
    np.savez(ROOT / f"data/allie/ttt_{band}{sfx}.npz", logp=lp_arr, legal=lg_arr, **r)
    if nfit:
        np.savez(ROOT / f"data/allie/ttt_{band}{sfx}_fit.npz", **{k: np.concatenate(v) for k, v in fit.items()})
    shutil.rmtree(parts)
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
    r.add_argument("--fit", type=int, default=0, help="hold out the N most recent context games as a fit split")
    r.add_argument("--sample", type=int, default=0, help="a random subset of N sides (seeded by band)")
    r.add_argument("--no-save-adapters", action="store_true", help="skip writing the per-side adapters")
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
        run(a.band, "cuda" if a.gpu_ok else "cpu", a.workers, a.epochs, a.lr, a.wd, a.bs, a.limit, tag, a.fit, a.sample,
            not a.no_save_adapters)
    else:
        summary(a.band, a.tag)


if __name__ == "__main__":
    main()

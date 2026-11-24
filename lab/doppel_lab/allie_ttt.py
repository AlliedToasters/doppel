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
Restore (--chunks spread:K or 0,2000,...): rerun only those chunks with the same per-chunk seeds, -> adapters_<band>
<tag>_restore/ and ttt_<band><tag>_restore_parts/ (never the original files); `verify` compares them to the full run.
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
CHUNK_EST_S = 720          # one 200-side ctx200 chunk on the 5090 (682 s measured 2026-11-19)
BANDS = (700, 1100, 1500, 1900, 2300)
INDEX = ROOT / "data/allie/restore_index_ctx200.json"
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
        nfit: int = 0, sample: int = 0, keep_adapters: bool = True, chunks: str = "", deadline: float = 0.0, pre=None, split: bool = False):
    import torch

    from doppel_lab.engines import Maia3
    from doppel_lab import lora_probe
    from doppel_lab.lora_probe import Lora, Positions, encode_all, logits, per_position
    rows, cut = pre if pre is not None else side_rows(band, nfit)
    if sample:                                              # a random subset of sides, pids renumbered 0..sample-1
        keep = np.sort(np.random.default_rng(band).choice(int(rows["pid"].max()) + 1, sample, replace=False))
        rows = rows.join(pl.DataFrame({"pid": keep.astype(np.int64), "new": np.arange(sample, dtype=np.int64)}),
                         on="pid").with_columns(pid=c("new")).drop("new").sort("pid", "j")
        tag += f"_fit{nfit}_s{sample}"
    if split:     # split-half reliability: side p -> pseudo-sides 2p (even-ranked train games) and 2p+1 (odd);
        tr_ = rows.filter(c("role") == "train")         # val and test games shared; control of 2p = its sister half
        rest = rows.filter(c("role") != "train")
        rows = pl.concat([tr_.with_columns(pid=2 * c("pid") + c("tr_rank") % 2), rest.with_columns(pid=2 * c("pid")),
                          rest.with_columns(pid=2 * c("pid") + 1)]).sort("pid", "j")
        tag += "_split"
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
    starts = list(range(0, n_sides, CHUNK))
    if chunks:                                  # restore mode: only these chunks, same seeds; nothing aggregated
        starts = pick_chunks(chunks, n_sides)
        sfx += "_restore"
        print(f"restore: chunks {starts}", flush=True)
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
    est = CHUNK_EST_S
    for c0 in starts:
        if deadline and time.time() + est > deadline:
            print(f"deadline: stopping before chunk {c0:,} (est {est:.0f}s)", flush=True)
            return False
        tc = time.time()
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
        est = time.time() - tc
        done = min(n_sides, c0 + CHUNK)
        hm, ho = np.concatenate(out["maia"]), np.concatenate(out["own"])
        print(f"{done:,}/{n_sides:,} sides, {len(hm):,} positions, own {100 * (ho.mean() - hm.mean()):+.3f} pts, "
              f"ctrl {100 * (np.concatenate(out['ctrl']).mean() - hm.mean()):+.3f}, best epochs "
              f"{np.bincount(best_ep, minlength=epochs + 1).tolist()}, {time.time() - t0:.0f}s", flush=True)
    if chunks:
        print(f"restore done: adapters in {adir}, parts kept in {parts} (allie_ttt verify --band {band})", flush=True)
        return True
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


def pick_chunks(spec: str, n_sides: int) -> list[int]:
    """'spread:K' = K chunk starts evenly over the band (the first and every ~n/K-th); else comma-separated starts."""
    allc = list(range(0, n_sides, CHUNK))
    if spec.startswith("spread:"):
        k = min(int(spec.split(":")[1]), len(allc))
        return [allc[i] for i in sorted({round(j * (len(allc) - 1) / max(1, k - 1)) for j in range(k)})] if k > 1 \
            else [allc[0]]
    got = [int(x) for x in spec.split(",") if x]
    bad = [x for x in got if x not in allc]
    if bad:
        raise SystemExit(f"not chunk starts: {bad} (multiples of {CHUNK} below {n_sides})")
    return got


def spread_order(n: int) -> list[int]:
    """0..n-1 such that every prefix is spread over the range: van der Corput (bit-reversal) points 0, 1/2, 1/4,
    3/4, 1/8 ... mapped to floor(v * n), duplicates skipped."""
    got, seen, k = [], set(), 0
    while len(got) < n:
        v, f, x = 0.0, 0.5, k
        while x:
            v, f, x = v + f * (x & 1), f / 2, x >> 1
        i = min(n - 1, int(v * n))
        if i not in seen:
            seen.add(i)
            got.append(i)
        k += 1
    return got


def build_index(force: bool = False):
    """The one sampling order for adapter restores (ctx200, no fit split): every (band, chunk start), bands in turn,
    each band's chunks in spread_order. Any prefix is a balanced sample; done = the adapter file exists. Written once;
    n_sides per band is recorded and checked at restore time so the index can't silently drift from the data."""
    if INDEX.exists() and not force:
        raise SystemExit(f"{INDEX} exists (it is the sampling order of record; --force to rebuild)")
    use_store200()
    per = {}
    for b in BANDS:
        rows, _ = side_rows(b)
        n = int(rows["pid"].max()) + 1
        per[b] = {"n_sides": n, "order": [k * CHUNK for k in spread_order(-(-n // CHUNK))]}
        print(f"band {b}: {n:,} sides, {len(per[b]['order'])} chunks", flush=True)
    units, k = [], 0
    while any(k < len(v["order"]) for v in per.values()):
        units += [{"band": b, "c0": per[b]["order"][k]} for b in BANDS if k < len(per[b]["order"])]
        k += 1
    idx = {"made": time.strftime("%Y-%m-%d"), "chunk": CHUNK, "tag": "_ctx200",
           "rule": "bands in turn, chunks per band in spread_order; resume = first unit without its adapter file",
           "bands": {str(b): {"n_sides": v["n_sides"]} for b, v in per.items()}, "units": units}
    INDEX.write_text(json.dumps(idx, indent=1) + "\n")
    print(f"{len(units)} units -> {INDEX}", flush=True)


def unit_done(band: int, c0: int) -> bool:
    return (ROOT / f"data/allie/adapters_{band}_ctx200_restore/chunk_{c0:06d}_{CHUNK}.npz").exists() and \
        (ROOT / f"data/allie/ttt_{band}_ctx200_restore_parts/chunk_{c0:06d}_{CHUNK}.npz").exists()


def restore(upto: int, max_minutes: float, workers: int):
    """Work down the index: the first `upto` units, skipping done ones, until max_minutes would be exceeded.
    Consecutive same-band units share one model load; a kill mid-chunk loses only that chunk."""
    idx = json.loads(INDEX.read_text())
    assert idx["chunk"] == CHUNK
    use_store200()
    deadline = time.time() + 60 * max_minutes
    todo = [u for u in idx["units"][:upto] if not unit_done(u["band"], u["c0"])]
    print(f"index: {len(idx['units'])} units; first {upto}: {upto - len(todo)} done, {len(todo)} to go", flush=True)
    cache = {}
    for u in todo:
        b = u["band"]
        if b not in cache:
            cache[b] = side_rows(b)
            if int(cache[b][0]["pid"].max()) + 1 != idx["bands"][str(b)]["n_sides"]:
                raise SystemExit(f"band {b}: side count changed since the index was built")
        if not run(b, "cuda", workers, 3, 1e-3, 0.01, 512, 0, "_ctx200", 0, 0, True, str(u["c0"]), deadline,
                   cache[b]):
            break
    left = sum(not unit_done(u["band"], u["c0"]) for u in idx["units"][:upto])
    print(f"restore: {upto - left}/{upto} of the target done", flush=True)


def verify(band: int, tag: str = "_ctx200"):
    """Restored chunks vs the original full run: own-adapter hits and log-probs on the same (game, ply)."""
    o = np.load(ROOT / f"data/allie/ttt_{band}{tag}.npz", allow_pickle=True)
    key = {(g, int(p)): i for i, (g, p) in enumerate(zip(o["game_id"].astype(str), o["ply"]))}
    parts = sorted((ROOT / f"data/allie/ttt_{band}{tag}_restore_parts").glob("chunk_*.npz"))
    if not parts:
        raise SystemExit("no restored chunks")
    res = []
    for f in parts:
        z = np.load(f)
        idx = np.array([key[(g, int(p))] for g, p in zip(z["game_id"].astype(str), z["ply"])])
        lps = np.split(z["lp_flat"].astype(np.float32), np.cumsum(z["lens"])[:-1])
        d = np.array([np.abs(a - o["logp"][i, :len(a)].astype(np.float32)).max() for a, i in zip(lps, idx)])
        r = {"chunk": f.name, "positions": len(idx), "own_hit_agree": float((z["own"] == o["own"][idx]).mean()),
             "maia_hit_agree": float((z["maia"] == o["maia"][idx]).mean()),
             "best_ep_agree": float((z["best_ep"] == o["best_ep"][idx]).mean()),
             "own_gain_restored": 100 * float(z["own"].mean() - z["maia"].mean()),
             "own_gain_original": 100 * float(o["own"][idx].mean() - o["maia"][idx].mean()),
             "dlogp_median": float(np.median(d)), "dlogp_p99": float(np.percentile(d, 99))}
        res.append(r)
        print(json.dumps(r), flush=True)
    return res


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
    r.add_argument("--split", action="store_true", help="two adapters per side on disjoint halves of its train games")
    r.add_argument("--chunks", default="", help="restore: 'spread:K' or '0,2000,...'; only those chunks, -> *_restore")
    r.add_argument("--blocks", default="", help="adapt only these Maia-3 blocks, 1-based (e.g. '5'); tag += _b<blocks>")
    s = sub.add_parser("summary")
    s.add_argument("--band", type=int, default=1500)
    s.add_argument("--tag", default="")
    ix = sub.add_parser("index", help="write the restore sampling order (CPU, once)")
    ix.add_argument("--force", action="store_true")
    rs = sub.add_parser("restore", help="work down the index (GPU, through the warden)")
    rs.add_argument("--upto", type=int, required=True, help="target = the first N units (5 per 1,000 sides/band)")
    rs.add_argument("--max-minutes", type=float, default=100)
    rs.add_argument("--workers", type=int, default=12)
    rs.add_argument("--gpu-ok", action="store_true")
    v = sub.add_parser("verify", help="restored chunks vs the original run (CPU)")
    v.add_argument("--band", type=int, default=1500)
    v.add_argument("--tag", default="_ctx200")
    a = ap.parse_args()
    if a.cmd == "index":
        build_index(a.force)
        return
    if a.cmd == "restore":
        if not a.gpu_ok:
            raise SystemExit("refusing to touch the GPU without --gpu-ok (run through the warden)")
        restore(a.upto, a.max_minutes, a.workers)
        return
    if a.cmd == "verify":
        verify(a.band, a.tag)
        return
    if a.cmd == "run":
        if not a.gpu_ok and not a.limit:
            raise SystemExit("refusing to touch the GPU without --gpu-ok (run through the warden)")
        tag = ""
        if a.ctx == 200:
            use_store200()
            tag = "_ctx200"
        if a.blocks:
            from doppel_lab import lora_probe
            lora_probe.BLOCKS = tuple(int(x) - 1 for x in a.blocks.split(","))
            tag += "_b" + a.blocks.replace(",", "")
        run(a.band, "cuda" if a.gpu_ok else "cpu", a.workers, a.epochs, a.lr, a.wd, a.bs, a.limit, tag, a.fit, a.sample,
            not a.no_save_adapters, a.chunks, split=a.split)
    else:
        summary(a.band, a.tag)


if __name__ == "__main__":
    main()

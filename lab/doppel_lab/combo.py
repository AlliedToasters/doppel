"""Combo: the Doppel's player vector z and the learned lookup on the same prediction (SCIENCE.md §8).

    python -m doppel_lab.combo zres --split valrich --band 1500 --gpu-ok   # GPU (warden): z residual per position
    python -m doppel_lab.combo zres --split allie --band 1500 --gpu-ok     # GPU (warden): same on the Allie bucket
    python -m doppel_lab.combo fit --band 1500                             # CPU: weights on valrich -> combo_fit_<band>.json
    python -m doppel_lab.combo eval --band 1500                            # CPU: Allie bucket -> combo_positions_<band>.parquet
    python -m doppel_lab.combo full                                        # CPU: all buckets -> lab/results/allie_combo_full.json
    python -m doppel_lab.combo stack --band 1500                           # CPU: + allie_ttt adapters -> allie_stack_<band>.json

Both signals are corrections to frozen Maia-3's log-probs over the legal moves:
  zres  = Doppel log-probs - Maia-3 log-probs (v0g model of the band, z from the side's 40 most recent earlier games)
  rlook = learned-lookup logits - Maia-3 log-probs (retr_train head of the band, store = 40 earlier games)
Combo logits = Maia-3 + a * zres + b * rlook, with (a, b) per ply bin chosen on valrich (600 held-out 2026 players
per band; queries = their own moves at ply >= 10 in games 40-47, the same targets retr_train and v0e use). Nothing is
chosen on the Allie test set. The < 1000 bucket borrows the 1100 band throughout, as both parts already do.
a = 0, b = 1 is the lookup alone; a = 1, b = 0 the Doppel alone. A two-fold player split of valrich gives the
held-out valrich gain of the chosen weights.

Move indices: both pipelines use teacher.MOVE_INDEX; v0e lists legal moves sorted, retr in python-chess order, so
zres is mapped row-wise by index.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/allie/combo"
RUN = {700: ("v0g_1100", 1100), 1100: ("v0g_1100", 1100), 1500: ("v0g_b", 1500), 1900: ("v0g_1900", 1900),
       2300: ("v0g_2300", 2300)}
FIT_BAND = {700: 1100}
PLY_BINS = ((10, 19), (20, 29), (30, 39), (40, 9999))
GRID = np.round(np.arange(0, 2.01, 0.125), 3)
MULT = 8192                      # > NMOVES (4352): row-wise searchsorted key
c = pl.col


def zres_path(split: str, band: int) -> Path:
    return OUT / f"zres_{split}_{band}.npz"


# ------------------------------------------------------------------ GPU: z residual
class ValrichSamples:
    """Each valrich side at history position 40..47 (k = 40 most recent earlier games), every own ply >= 10."""

    def __init__(self, sp, k: int = 40, lo: int = 40, hi: int = 48):
        hp = sp.pid[sp.hg, sp.hc]
        j = np.arange(len(hp)) - sp.start_of
        n = np.bincount(hp, minlength=len(sp.players))[hp]
        ok = (j >= lo) & (j < hi) & (n >= hi)
        self.sp, self.k = sp, k
        self.items = [(int(sp.hg[i]), int(sp.hc[i])) for i in np.flatnonzero(ok)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from doppel_lab.v0_data import targets_sample
        from doppel_lab.v0e_data import CTX_FEATS, EMB_DIM
        g, col = self.items[i]
        sp, ix = self.sp, self.sp.ix
        cand = [p for p in range(col, ix.plies(g), 2) if p >= 10]
        if not cand:
            return None
        out = targets_sample(ix, g, col, cand, None, len(cand), np.random.default_rng(0))
        hg, hc = sp.history(g, col)
        k = min(self.k, len(hg))
        hg, hc = hg[len(hg) - k:], hc[len(hc) - k:]
        out["ctx_emb"] = np.asarray(sp.vec.emb[hg, hc]) if k else np.zeros((0, EMB_DIM), np.float16)
        out["ctx_feat"] = sp.feat[hg, hc] if k else np.zeros((0, CTX_FEATS), np.float32)
        out["pid"] = np.array([sp.pid[g, col]], np.int64)
        out["nctx"] = np.array([k], np.int64)
        out["gidx"] = np.full(len(out["played"]), g, np.int64)
        return out


def zres(split: str, band: int, device: str, workers: int, batch: int, limit: int = 0):
    import torch

    from doppel_lab.allie_v0e import AllieSamples, split_name, test_games
    from doppel_lab.v0_model import legal_mask
    from doppel_lab.v0_train import to_dev
    from doppel_lab.v0e_data import Split, collate
    from doppel_lab.v0e_train import V0, build
    run, mb = RUN[band]
    if split == "allie":
        t = test_games(band)
        sp = Split(split_name(band), band)
        ds = AllieSamples(sp, set(t["game_id"].to_list()), dict(zip(t["game_id"].to_list(), t["cut"].to_list())),
                          ks=(40,))
    else:
        sp = Split("valrich", band)
        ds = ValrichSamples(sp)
    if limit:
        ds.items = ds.items[:limit]

    def coll(ss):
        ss = [s for s in ss if s is not None]
        b = collate(ss)
        b["gidx"] = torch.from_numpy(np.concatenate([s["gidx"] for s in ss]))
        return b

    ld = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False, collate_fn=coll, num_workers=workers)
    model = build(False, device, band=mb)
    s = torch.load(V0 / "runs" / run / "ckpt.pt", map_location=device, weights_only=False)
    model.load_state_dict(s["model"], strict=False)
    model.eval()
    rows = {k: [] for k in ("gidx", "ply", "legal", "res", "top_t", "top_q")}
    t0 = time.time()
    with torch.no_grad():
        for i, b in enumerate(ld):
            b = to_dev(b, device)
            tok = b["tok"].float()
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                t_logits, _, _ = model.trunk(tok, b["elo_self"], b["elo_opp"])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                mask = legal_mask(b["legal"])
                t_logp = t_logits.float().masked_fill(~mask, -1e9).log_softmax(-1)
                mu_q, _ = model.posterior(b)
                emb = model.residual.board(tok)
                sid, pl_idx = b["sid"], b["played"].clamp(min=0)
                lg = t_logp + model.residual(emb, mu_q[sid], b["elo_self"], b["elo_opp"], mask).float()
                lp = lg.masked_fill(~mask, -1e9).log_softmax(-1)
            legal = b["legal"].long()
            d = (lp - t_logp).float().gather(1, legal.clamp(min=0))
            d = torch.where(legal >= 0, d, torch.zeros_like(d))
            rows["gidx"].append(b["gidx"].cpu())
            rows["ply"].append(b["ply"].cpu())
            rows["legal"].append(b["legal"].cpu().to(torch.int16))
            rows["res"].append(d.half().cpu())
            rows["top_t"].append((t_logp.argmax(-1) == pl_idx).cpu())
            rows["top_q"].append((lp.argmax(-1) == pl_idx).cpu())
            if i % 50 == 0:
                print(f"batch {i}/{len(ld)}, {time.time() - t0:.0f}s", flush=True)
    r = {k: torch.cat(v).numpy() for k, v in rows.items()}
    gid = sp.ix.game_id[r.pop("gidx")].astype(str)
    OUT.mkdir(parents=True, exist_ok=True)
    out = zres_path(split, band) if not limit else OUT / f"zres_{split}_{band}_smoke.npz"
    np.savez(out, game_id=gid, **r)
    print(f"{len(gid):,} positions; teacher top-1 {r['top_t'].mean():.4f}, Doppel {r['top_q'].mean():.4f} -> {out}",
          flush=True)


# ------------------------------------------------------------------ CPU: align and combine
def load_zres(split: str, band: int, smoke: bool = False) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    z = np.load(OUT / f"zres_{split}_{band}{'_smoke' if smoke else ''}.npz")
    key = pl.DataFrame({"game_id": z["game_id"], "ply": z["ply"].astype(np.int64)}).with_row_index("zrow")
    return key, z["legal"], z["res"].astype(np.float32)


def align(zlegal: np.ndarray, zres_: np.ndarray, legal: np.ndarray) -> np.ndarray:
    """zres over retr's legal columns. zlegal/zres_ (N, L) sorted with -1 padding; legal (N, L) retr order."""
    n, L = zlegal.shape
    big = np.where(zlegal < 0, MULT - 1, zlegal).astype(np.int64) + MULT * np.arange(n)[:, None]
    flat = big.ravel()
    want = np.where(legal < 0, 0, legal).astype(np.int64) + MULT * np.arange(n)[:, None]
    j = np.clip(np.searchsorted(flat, want.ravel()), 0, flat.size - 1)
    ok = (flat[j] == want.ravel()) & (legal.ravel() >= 0)
    out = np.where(ok, zres_.ravel()[j], 0.0).reshape(legal.shape)
    miss = ((legal >= 0) & ~ok.reshape(legal.shape)).sum()
    assert miss == 0, f"{miss} legal moves without a z residual"
    return out.astype(np.float32)


def valrich_parts(band: int):
    """Per valrich query: game_id, ply, player, Maia log-probs, rlook, zres (retr legal order), truth."""
    import torch

    from doppel_lab.retr_train import Head, groups_for, load_split, out_dir, query_rows
    root = out_dir(band) / "valrich"
    sub, groups = groups_for("valrich", 10**9, 40, 8, 0, band=band)
    off = np.load(root / "off.npy")
    assert sub.height == len(off) - 1
    gids = sub["game_id"].to_numpy().astype(str)
    qz = dict(np.load(root / "query.npz"))
    qrow_of = np.full(int(off[-1]), -1, np.int64)
    qrow_of[qz["pos"]] = np.arange(len(qz["pos"]))
    ck = torch.load(out_dir(band) / "head.pt", weights_only=False)
    head = Head.make()
    head.load_state_dict(ck["state"])
    head.eval()
    batches = load_split(root)
    saved = np.load(root / "groups.npy", allow_pickle=True)
    meta = []
    for pi, (sg, sc, qg, qc) in enumerate(saved):
        qpos, qply = query_rows(off, qg, qc)
        keep = qrow_of[qpos] >= 0
        qpos, qply = qpos[keep], qply[keep]
        if len(qpos):
            gi = np.searchsorted(off, qpos, side="right") - 1
            meta.append((gids[gi], qply, np.full(len(qpos), pi), qrow_of[qpos]))
    assert len(meta) == len(batches)
    logp, rlook, truth = [], [], []
    with torch.no_grad():
        for kq, ks, col, ply, lp, tr in batches:
            out = head(kq, ks, col, ply, lp)
            logp.append(lp.numpy()); rlook.append((out - lp).numpy()); truth.append(tr.numpy())
    df = pl.DataFrame({"game_id": np.concatenate([m[0] for m in meta]), "ply": np.concatenate([m[1] for m in meta]),
                       "player": np.concatenate([m[2] for m in meta]), "qrow": np.concatenate([m[3] for m in meta])})
    logp, rlook, truth = np.concatenate(logp), np.concatenate(rlook), np.concatenate(truth)
    legal = qz["legal"][df["qrow"].to_numpy()].astype(np.int64)
    key, zl, zr = load_zres("valrich", band)
    # players with > 48 valrich sides (opponents of other valrich players) get different target games in the two
    # pipelines (retr: last 48 by time; v0e: history positions 40-47); keep the queries both scored
    j = df.select("game_id", "ply").join(key, on=["game_id", "ply"], how="left")["zrow"]
    ok = j.is_not_null().to_numpy()
    print(f"valrich queries with a z residual: {ok.sum():,} of {len(ok):,}", flush=True)
    jj = j.to_numpy()[ok].astype(np.int64)
    df, logp, rlook, truth, legal = df.filter(pl.Series(ok)), logp[ok], rlook[ok], truth[ok], legal[ok]
    zr = align(zl[jj], zr[jj], legal)
    return df, logp, rlook, zr, truth


def hits(logp, rlook, zr, truth, a, b):
    return (logp + a * zr + b * rlook).argmax(1) == truth


def choose(logp, rlook, zr, truth, ply):
    """Best (a, b) per ply bin by top-1 (ties: closest to (0, 1), the lookup alone)."""
    out = {}
    for lo, hi in PLY_BINS:
        s = (ply >= lo) & (ply <= hi)
        best = None
        for a in GRID:
            for b in GRID:
                h = hits(logp[s], rlook[s], zr[s], truth[s], a, b).mean() if s.any() else 0.0
                k = (h, -abs(a) - abs(b - 1))
                if best is None or k > best[0]:
                    best = (k, float(a), float(b))
        out[f"{lo}-{hi}"] = {"a": best[1], "b": best[2]}
    return out


def apply(w, logp, rlook, zr, truth, ply):
    h = np.zeros(len(truth), bool)
    for lo, hi in PLY_BINS:
        s = (ply >= lo) & (ply <= hi)
        ww = w[f"{lo}-{hi}"]
        h[s] = hits(logp[s], rlook[s], zr[s], truth[s], ww["a"], ww["b"])
    return h


def fit(band: int):
    df, logp, rlook, zr, truth = valrich_parts(band)
    ply = df["ply"].to_numpy()
    maia = logp.argmax(1) == truth
    res = {"band": band, "queries": len(truth), "players": int(df["player"].n_unique()),
           "maia_top1": float(maia.mean()), "fixed": {}}
    for name, a, b in (("lookup", 0, 1), ("doppel", 1, 0), ("sum", 1, 1)):
        res["fixed"][name] = 100 * float(hits(logp, rlook, zr, truth, a, b).mean() - maia.mean())
    w = choose(logp, rlook, zr, truth, ply)
    res["weights"] = w
    res["fit_gain_pts"] = 100 * float(apply(w, logp, rlook, zr, truth, ply).mean() - maia.mean())
    # honest valrich number: weights chosen on one half of the players, scored on the other
    half = df["player"].to_numpy() % 2
    cv = np.zeros(len(truth), bool)
    for f in (0, 1):
        tr, te = half != f, half == f
        wf = choose(logp[tr], rlook[tr], zr[tr], truth[tr], ply[tr])
        cv[te] = apply(wf, logp[te], rlook[te], zr[te], truth[te], ply[te])
    res["cv_gain_pts"] = 100 * float(cv.mean() - maia.mean())
    res["cv_lookup_pts"] = res["fixed"]["lookup"]
    p = OUT / f"combo_fit_{band}.json"
    p.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


def eval_allie(band: int):
    """Allie bucket: Maia-3, lookup, Doppel, fixed sum and the valrich-chosen combo per position (+ shuffled-store
    lookup inside the combo as the control)."""
    import torch

    from doppel_lab.allie_knn import build_queries, knn_dir
    from doppel_lab.retr_train import HEAD_BAND, Head, out_dir
    hb = HEAD_BAND.get(band, band)
    ck = torch.load(out_dir(hb) / "head.pt", weights_only=False)
    head = Head.make()
    head.load_state_dict(ck["state"])
    head.eval()
    w = json.loads((OUT / f"combo_fit_{FIT_BAND.get(band, band)}.json").read_text())["weights"]
    kd = knn_dir(band)
    keys = np.load(kd / "keys.npy", mmap_mode="r")
    played = np.load(kd / "played.npy")
    tz = dict(np.load(kd / "test.npz"))
    trow = np.full(len(played), -1, np.int64)
    trow[tz["pos"]] = np.arange(len(tz["pos"]))
    sides = build_queries(band)
    perm = np.random.default_rng(0).permutation(len(sides))
    key, zl, zr_all = load_zres("allie", band)
    kmap = {(g, p): r for g, p, r in zip(key["game_id"].to_list(), key["ply"].to_list(), key["zrow"].to_list())}
    rec = []
    with torch.no_grad():
        for si, (gid, plies, q, st) in enumerate(sides):
            rows = trow[q]
            legal = tz["legal"][rows].astype(np.int64)
            logp = np.nan_to_num(tz["logp"][rows].astype(np.float32), neginf=-1e9)
            truth = np.argmax(legal == played[q].astype(np.int64)[:, None], 1)
            kq = keys[q].astype(np.float32)
            zrow = np.array([kmap[(gid, int(p))] for p in plies])
            zr = align(zl[zrow], zr_all[zrow], legal)
            rl = []
            for store in (st, sides[perm[si]][3]):
                ks = keys[store].astype(np.float32)
                sm = played[store].astype(np.int64)
                hit = legal[:, None, :] == sm[None, :, None] if len(store) else np.zeros((len(q), 0, legal.shape[1]), bool)
                col = np.where(hit.any(2), hit.argmax(2), -1) if len(store) else np.zeros((len(q), 0), np.int64)
                out = head(torch.from_numpy(kq), torch.from_numpy(ks), torch.from_numpy(col),
                           torch.from_numpy(plies.astype(np.float32)), torch.from_numpy(logp)).numpy()
                rl.append(out - logp)
            rec.append({"game_id": np.full(len(q), gid, object), "ply": plies,
                        "maia_hit": logp.argmax(1) == truth,
                        "lookup_hit": hits(logp, rl[0], zr, truth, 0, 1),
                        "doppel_hit": hits(logp, rl[0], zr, truth, 1, 0),
                        "sum_hit": hits(logp, rl[0], zr, truth, 1, 1),
                        "combo_hit": apply(w, logp, rl[0], zr, truth, plies),
                        "combo_ctrl_hit": apply(w, logp, rl[1], zr, truth, plies)})
    d = pl.DataFrame({k: np.concatenate([r[k] for r in rec]) for k in rec[0]}).with_columns(
        c("game_id").cast(pl.String), c("ply").cast(pl.Int64))
    d.write_parquet(OUT / f"combo_positions_{band}.parquet")
    print(band, d.height, {k: round(100 * (d[k].mean() - d["maia_hit"].mean()), 3)
                           for k in d.columns if k.endswith("_hit") and k != "maia_hit"}, flush=True)


def align_any(srcl: np.ndarray, srcv: np.ndarray, legal: np.ndarray) -> np.ndarray:
    """align() for source rows in any order: sort each row by move index first."""
    o = np.argsort(np.where(srcl < 0, MULT - 1, srcl), axis=1, kind="stable")
    return align(np.take_along_axis(srcl, o, 1), np.take_along_axis(srcv, o, 1), legal)


def stack(band: int = 1500):
    """Maia-3 + z residual + lookup residual + test-time adapter residual (allie_ttt), equal weights (no fitting),
    and every subset, on the Allie bucket -> lab/results/allie_stack_<band>.json."""
    import itertools

    import torch

    from doppel_lab.allie_knn import build_queries, knn_dir
    from doppel_lab.retr_train import HEAD_BAND, Head, out_dir
    ck = torch.load(out_dir(HEAD_BAND.get(band, band)) / "head.pt", weights_only=False)
    head = Head.make()
    head.load_state_dict(ck["state"])
    head.eval()
    kd = knn_dir(band)
    keys = np.load(kd / "keys.npy", mmap_mode="r")
    played = np.load(kd / "played.npy")
    tz = dict(np.load(kd / "test.npz"))
    trow = np.full(len(played), -1, np.int64)
    trow[tz["pos"]] = np.arange(len(tz["pos"]))
    key, zl, zr_all = load_zres("allie", band)
    kmap = {(g, p): r for g, p, r in zip(key["game_id"].to_list(), key["ply"].to_list(), key["zrow"].to_list())}
    tt = np.load(ROOT / f"data/allie/ttt_{band}.npz", allow_pickle=True)
    tmap = {(str(g), int(p)): i for i, (g, p) in enumerate(zip(tt["game_id"], tt["ply"]))}
    parts = ("z", "lookup", "ttt")
    combos = [s for k in range(1, 4) for s in itertools.combinations(parts, k)]
    combos += [tuple("ttt40" if p == "ttt" else p for p in s) for s in combos if "ttt" in s]
    rec = {"+".join(s): [] for s in combos}
    rec["maia"], gids, plys = [], [], []
    with torch.no_grad():
        for gid, plies, q, st in build_queries(band):
            sel = np.array([(gid, int(p)) in tmap for p in plies])
            if not sel.any():
                continue
            plies, q = plies[sel], q[sel]
            rows = trow[q]
            legal = tz["legal"][rows].astype(np.int64)
            logp = np.nan_to_num(tz["logp"][rows].astype(np.float32), neginf=-1e9)
            truth = np.argmax(legal == played[q].astype(np.int64)[:, None], 1)
            zrow = np.array([kmap[(gid, int(p))] for p in plies])
            res = {"z": align(zl[zrow], zr_all[zrow], legal)}
            ks = keys[st].astype(np.float32)
            sm = played[st].astype(np.int64)
            hit = legal[:, None, :] == sm[None, :, None] if len(st) else np.zeros((len(q), 0, legal.shape[1]), bool)
            col = np.where(hit.any(2), hit.argmax(2), -1) if len(st) else np.zeros((len(q), 0), np.int64)
            out = head(torch.from_numpy(keys[q].astype(np.float32)), torch.from_numpy(ks), torch.from_numpy(col),
                       torch.from_numpy(plies.astype(np.float32)), torch.from_numpy(logp)).numpy()
            res["lookup"] = out - logp
            ti = np.array([tmap[(gid, int(p))] for p in plies])
            tl = np.nan_to_num(align_any(tt["legal"][ti].astype(np.int64), tt["logp"][ti].astype(np.float32), legal),
                               neginf=-1e9)
            res["ttt"] = np.where(legal >= 0, tl - logp, 0.0)
            # ply gate fixed from lora_probe (independent players): the adapter loses at ply >= 40 there
            res["ttt40"] = np.where((plies < 40)[:, None], res["ttt"], 0.0)
            rec["maia"].append(logp.argmax(1) == truth)
            for s in combos:
                rec["+".join(s)].append((logp + sum(res[p] for p in s)).argmax(1) == truth)
            gids.append(np.full(len(q), gid, object)); plys.append(plies)
    d = {k: np.concatenate(v) for k, v in rec.items()}
    gid, ply = np.concatenate(gids).astype(str), np.concatenate(plys)
    gi = np.unique(gid, return_inverse=True)[1]
    rng = np.random.default_rng(0)
    m = d.pop("maia")
    out = {"band": band, "positions": len(m), "maia_top1": float(m.mean()), "arms": {}}
    for k, h in d.items():
        dd = h.astype(float) - m
        gs, gn = np.bincount(gi, dd), np.bincount(gi)
        b = rng.integers(0, len(gs), (1000, len(gs)))
        bs = 100 * gs[b].sum(1) / gn[b].sum(1)
        out["arms"][k] = {"gain_pts": 100 * float(dd.mean()), "ci95": [float(np.percentile(bs, 2.5)),
                                                                         float(np.percentile(bs, 97.5))],
                          "by_ply": {f"{lo}-{hi}": 100 * float(dd[(ply >= lo) & (ply <= hi)].mean())
                                     for lo, hi in PLY_BINS}}
        print(k, round(out["arms"][k]["gain_pts"], 3), {a: round(v, 2) for a, v in out["arms"][k]["by_ply"].items()},
              flush=True)
    t40 = d["ttt40"].astype(float)
    for k, h in d.items():                                  # paired: arm minus the gated adapter alone
        dd = h.astype(float) - t40
        gs, gn = np.bincount(gi, dd), np.bincount(gi)
        b = rng.integers(0, len(gs), (1000, len(gs)))
        bs = 100 * gs[b].sum(1) / gn[b].sum(1)
        out["arms"][k]["vs_ttt40"] = {"pts": 100 * float(dd.mean()),
                                      "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    np.savez(OUT / f"stack_hits_{band}.npz", game_id=gid, ply=ply, maia=m, **{k.replace("+", "_"): v for k, v in d.items()})
    (ROOT / f"lab/results/allie_stack_{band}.json").write_text(json.dumps(out, indent=1) + "\n")


def full():
    import math

    from doppel_lab.allie_v0e import BANDS
    x = pl.concat([pl.read_parquet(OUT / f"combo_positions_{b}.parquet").with_columns(bucket=pl.lit(b))
                   for b in sorted(BANDS)])
    rng = np.random.default_rng(0)

    def paired(d, a, b):
        n01 = int((d[a] & ~d[b]).sum()); n10 = int((~d[a] & d[b]).sum())
        chi = (abs(n01 - n10) - 1) ** 2 / max(n01 + n10, 1)
        g = d.select("game_id", v=d[a].cast(pl.Float64) - d[b].cast(pl.Float64)).group_by("game_id").agg(
            c("v").sum(), pl.len().alias("n"))
        v, n = g["v"].to_numpy(), g["n"].to_numpy()
        idx = rng.integers(0, len(g), (2000, len(g)))
        bs = 100 * v[idx].sum(1) / n[idx].sum(1)
        return {"a_top1": float(d[a].mean()), "b_top1": float(d[b].mean()),
                "gain_pts": 100 * float(d[a].mean() - d[b].mean()), "discordant_win": n01, "discordant_loss": n10,
                "mcnemar_chi2": chi, "mcnemar_p": math.erfc(math.sqrt(chi / 2)),
                "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    res = {"positions": x.height, "games": x["game_id"].n_unique(),
           "duplicate_keys": x.height - x.unique(["game_id", "ply"]).height,
           "weights": {str(b): json.loads((OUT / f"combo_fit_{FIT_BAND.get(b, b)}.json").read_text())["weights"]
                       for b in sorted(BANDS)}, "overall": {}, "by_bucket": {}, "by_ply": {}}
    cols = ("lookup_hit", "doppel_hit", "sum_hit", "combo_hit", "combo_ctrl_hit")
    for k in cols:
        res["overall"][k.removesuffix("_hit") + "_vs_maia"] = paired(x, k, "maia_hit")
    res["overall"]["combo_vs_lookup"] = paired(x, "combo_hit", "lookup_hit")
    for b in sorted(BANDS):
        d = x.filter(c("bucket") == b)
        res["by_bucket"][str(b)] = {"positions": d.height, "maia": float(d["maia_hit"].mean()),
                                    **{k.removesuffix("_hit"): 100 * float(d[k].mean() - d["maia_hit"].mean())
                                       for k in cols},
                                    "combo_vs_lookup": paired(d, "combo_hit", "lookup_hit")}
    for lo, hi in PLY_BINS:
        d = x.filter((c("ply") >= lo) & (c("ply") <= hi))
        res["by_ply"][f"{lo}-{hi}"] = {"positions": d.height, **{k.removesuffix("_hit"): 100 * float(
            d[k].mean() - d["maia_hit"].mean()) for k in cols}}
    p = ROOT / "lab/results/allie_combo_full.json"
    p.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("positions", "overall")}, indent=1))
    for b, r in res["by_bucket"].items():
        print(b, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)})


def stack_full():
    """All five buckets' stack hits (combo.stack) pooled: the full Allie test set -> lab/results/allie_stack_full.json.
    Paired game bootstrap against Maia-3 and against z + lookup (equal weights, the same positions)."""
    from doppel_lab.allie_v0e import BANDS
    hs = {b: np.load(OUT / f"stack_hits_{b}.npz", allow_pickle=True) for b in sorted(BANDS)}
    arms = [k for k in hs[sorted(BANDS)[0]].files if k not in ("game_id", "ply")]
    cat = {k: np.concatenate([h[k] for h in hs.values()]) for k in arms + ["ply"]}
    gid = np.concatenate([np.char.add(f"{b}:", h["game_id"].astype(str)) for b, h in hs.items()])
    gi = np.unique(gid, return_inverse=True)[1]
    rng = np.random.default_rng(0)
    idx = rng.integers(0, gi.max() + 1, (1000, gi.max() + 1))

    def paired(a, b, sel=None):
        d = cat[a].astype(float) - cat[b].astype(float)
        g = gi
        if sel is not None:
            d, g = d[sel], np.unique(gi[sel], return_inverse=True)[1]
        gs, gn = np.bincount(g, d), np.bincount(g)
        ix = idx if sel is None else rng.integers(0, len(gs), (1000, len(gs)))
        bs = 100 * gs[ix].sum(1) / gn[ix].sum(1)
        return {"pts": 100 * float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    ply = cat["ply"]
    res = {"positions": len(gi), "games": int(gi.max() + 1), "maia_top1": float(cat["maia"].mean()),
           "by_bucket_positions": {str(b): len(h["ply"]) for b, h in hs.items()}, "arms": {}}
    for k in arms:
        if k == "maia":
            continue
        r = {"top1": float(cat[k].mean()), "vs_maia": paired(k, "maia"), "vs_z_lookup": paired(k, "z_lookup"),
             "by_ply_vs_maia": {f"{lo}-{hi}": 100 * float((cat[k][(ply >= lo) & (ply <= hi)].astype(float)
                                                           - cat["maia"][(ply >= lo) & (ply <= hi)]).mean())
                                for lo, hi in PLY_BINS},
             "by_bucket_vs_maia": {str(b): 100 * float((h[k].astype(float) - h["maia"]).mean()) for b, h in hs.items()}}
        res["arms"][k] = r
        print(f"{k:16s} {100 * r['top1']:.2f}%  vs maia {r['vs_maia']['pts']:+.3f} {[round(x, 2) for x in r['vs_maia']['ci95']]}"
              f"  vs z+lookup {r['vs_z_lookup']['pts']:+.3f} {[round(x, 2) for x in r['vs_z_lookup']['ci95']]}", flush=True)
    (ROOT / "lab/results/allie_stack_full.json").write_text(json.dumps(res, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    z = sub.add_parser("zres")
    z.add_argument("--split", choices=["valrich", "allie"], required=True)
    z.add_argument("--gpu-ok", action="store_true")
    z.add_argument("--workers", type=int, default=8)
    z.add_argument("--batch", type=int, default=64)
    z.add_argument("--limit", type=int, default=0, help="smoke: first N sides, CPU ok")
    for q in (z, sub.add_parser("fit"), sub.add_parser("eval")):
        q.add_argument("--band", type=int, default=1500, choices=sorted(RUN))
    sub.add_parser("full")
    sub.add_parser("stack-full")
    st = sub.add_parser("stack")
    st.add_argument("--band", type=int, default=1500)
    a = ap.parse_args()
    if a.cmd == "zres":
        if not a.gpu_ok and not a.limit:
            raise SystemExit("refusing to touch the GPU without --gpu-ok (run through the warden)")
        zres(a.split, a.band, "cuda" if a.gpu_ok else "cpu", a.workers, a.batch, a.limit)
    elif a.cmd == "fit":
        fit(a.band)
    elif a.cmd == "eval":
        eval_allie(a.band)
    elif a.cmd == "stack":
        stack(a.band)
    elif a.cmd == "stack-full":
        stack_full()
    else:
        full()


if __name__ == "__main__":
    main()

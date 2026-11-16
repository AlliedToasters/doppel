"""Retrieval step 1: a learned lookup over a player's own earlier positions (1500 band).

    python -m doppel_lab.retr_train states --gpu-ok           # GPU (warden), ~15 min: keys + Maia-3 policy, train + valrich
    python -m doppel_lab.retr_train states --limit 60          # CPU smoke -> data/allie/retr1_smoke/
    python -m doppel_lab.retr_train train                      # CPU, ~10-30 min -> data/allie/retr1/head.pt
    python -m doppel_lab.retr_train eval                       # CPU -> lab/results/allie_retr1.json

Step 0 (allie_knn.py) voted with frozen Maia-3 keys (pooled last_ln, PCA-128, cosine) and gained +0.11 top-1 on the
Allie 1400-1600 bucket, all of it at plies 10-19 and best cosine >= 0.9: near-transpositions only. This step learns
the lookup. Same key space (the step-0 PCA, data/allie/knn_1500/pca.npz), so the Allie test keys are reused as is.

Data (2026, never the test year):
  train    2,000 players sampled from the 1500-band train split with >= 50 games; per player, their 50 most recent
           games by time: store = the first 40, queries = the player's own moves at ply >= 10 in the last 10.
  valrich  the held-out valrich players (>= 48 games): store 40, queries from the last 8. Used to pick the epoch.
One Maia-3 79M pass (fp16, real Elos) over every position of those games: key (PCA-128, unit norm), played move;
for query positions also the legal moves and Maia-3's log-probs over them.

Head (a few thousand parameters; Maia-3 frozen and untouched):
  q = norm(Wq k_query), s = norm(Wk k_store)      Wq, Wk: 128 -> 64, both start at the same random orthogonal map
  w[q, s] = exp((q.s - 1) / tau)                   only store positions whose played move is legal in the query
  mass_j = sum of w over store positions that played legal move j;  pk = mass / (sum mass + 1e-6)
  r_j = g * (log(eps + pk_j) - log eps)             0 for moves no neighbour played
  g = softplus(MLP(max q.s, log total mass, ply / 100, [ply < 20], log store size))
  logits = Maia-3 log-probs + r;  loss = cross-entropy of the played move.
Eval on the Allie 1400-1600 positions: top-1 gain over Maia-3 with a game bootstrap, split by exact-repeat history,
ply bucket and best frozen cosine, plus the shuffled-store control (each side gets another test side's store, the
same permutation as step 0). No tuning on the test set: the epoch is chosen on valrich.
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
KNN = ROOT / "data/allie/knn_1500"          # the shared PCA key space (every band projects with it)
OUT = ROOT / "data/allie/retr1"
RESULTS = ROOT / "lab/results/allie_retr1.json"
# Allie buckets without their own 2026 band data borrow a head (out-of-band), like allie_v0e.MODEL_BAND
HEAD_BAND = {700: 1100}


def out_dir(band: int = 1500) -> Path:
    """Head-band states and head.pt; 1500 keeps the original paths."""
    return OUT if band == 1500 else ROOT / f"data/allie/retr1_{band}"


def results_path(band: int = 1500) -> Path:
    return RESULTS if band == 1500 else ROOT / f"lab/results/allie_retr1_{band}.json"


def positions_path(band: int) -> Path:
    """Per-position rows of an Allie bucket eval (game_id, ply, hit, maia_hit, ctrl_hit), for the full-set combine."""
    return ROOT / f"data/allie/retr1_positions_{band}.parquet"
D, H, BLOCK, MAXL = 128, 64, 4_000, 224
c = pl.col


# ------------------------------------------------------------------ groups
def groups_for(split: str, n_players: int, n_store: int, n_query: int, limit: int, seed: int = 0, band: int = 1500):
    """Games to encode and, per player, (game rows, colours) of the store and query games."""
    if split == "train":
        g = pl.read_parquet(ROOT / f"data/v0/games_{band}_train.parquet",
                            columns=["game_id", "white", "black", "white_elo", "black_elo", "moves"])
        m = np.load(ROOT / f"data/v0e/meta_{band}_train.npz", allow_pickle=True)
        assert (g["game_id"].to_numpy().astype("U") == m["game_id"]).all()
        g = g.with_columns(t=pl.Series(m["t"]))
    else:
        g = pl.read_parquet(ROOT / f"data/v0e/games_{band}_valrich.parquet",
                            columns=["game_id", "white", "black", "white_elo", "black_elo", "moves", "t"])
    g = g.with_row_index("gi")
    side = pl.concat([g.select("gi", "t", player=c(col).str.to_lowercase(), col=pl.lit(i, pl.Int8))
                      for i, col in enumerate(("white", "black"))])
    need = n_store + n_query
    cnt = side.group_by("player").len().filter(c("len") >= need)
    rng = np.random.default_rng(seed)
    pls = cnt["player"].sort().to_numpy()
    pls = pls[rng.permutation(len(pls))[:n_players]] if n_players < len(pls) else pls
    if limit:
        pls = pls[:max(2, limit // need)]
    sel = (side.filter(c("player").is_in(pls.tolist())).sort("t").group_by("player", maintain_order=True)
           .tail(need).with_columns(k=pl.int_range(pl.len()).over("player")))
    used = np.unique(sel["gi"].to_numpy())
    sub = g.filter(c("gi").is_in(used.tolist())).sort("gi")
    remap = np.full(g.height, -1, np.int64)
    remap[sub["gi"].to_numpy()] = np.arange(sub.height)
    groups = []
    for (pname,), d in sel.group_by("player", maintain_order=True):
        d = d.sort("k")
        gis, cols = remap[d["gi"].to_numpy()], d["col"].to_numpy()
        groups.append((gis[:n_store], cols[:n_store], gis[n_store:], cols[n_store:]))
    return sub.drop("gi"), groups


# ------------------------------------------------------------------ states
def encode(g: pl.DataFrame, qmask_fn, out: Path, device: str, workers: int, batch: int):
    """Keys (step-0 PCA), played moves, and legal / Maia-3 log-probs at query positions (qmask_fn(off) -> pos)."""
    import torch

    from doppel_lab.engines import Maia3
    from doppel_lab.teacher import HIST, encode_game
    out.mkdir(parents=True, exist_ok=True)
    n_per = g["moves"].list.len().to_numpy()
    off = np.r_[0, np.cumsum(n_per)].astype(np.int64)
    P = int(off[-1])
    qpos = np.unique(qmask_fn(off))
    qrow = np.full(P, -1, np.int64)
    qrow[qpos] = np.arange(len(qpos))
    pca = np.load(KNN / "pca.npz")
    mean = torch.from_numpy(pca["mean"]).to(device)
    comp = torch.from_numpy(pca["comp"]).to(device)
    keys = np.lib.format.open_memmap(out / "keys.npy", "w+", np.float16, (P, D))
    played = np.zeros(P, np.int16)
    q_legal = np.full((len(qpos), MAXL), -1, np.int16)
    q_logp = np.full((len(qpos), MAXL), -np.inf, np.float16)
    model = Maia3("79m", device).model.eval()
    pooled = {}
    model.last_ln.register_forward_hook(lambda m, i, o: pooled.__setitem__("x", o))
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
            lstart = np.r_[0, np.cumsum(np.concatenate([e[2] for e in enc]).astype(np.int64))]
            es = torch.from_numpy(np.concatenate([e[4] for e in enc]).astype(np.int64)).to(device)
            eo = torch.from_numpy(np.concatenate([e[5] for e in enc]).astype(np.int64)).to(device)
            played[base:base + ob[-1]] = np.concatenate([e[3] for e in enc])
            for s in range(0, int(ob[-1]), batch):
                e_ = min(int(ob[-1]), s + batch)
                x = planes[hidx[s:e_]].float().permute(0, 2, 1, 3).reshape(e_ - s, 64, HIST * 12)
                x = torch.cat([x, torch.zeros(e_ - s, 64, 1, device=device)], 2)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                    logits = model(x, es[s:e_], eo[s:e_])[0].float()
                px = pooled["x"].float()
                f = px.mean(1) if px.dim() == 3 else px
                keys[base + s:base + e_] = torch.nn.functional.normalize((f - mean) @ comp, dim=1).half().cpu().numpy()
                qr = qrow[base + s:base + e_]
                for j in np.flatnonzero(qr >= 0):
                    li = legal[lstart[s + j]:lstart[s + j + 1]]
                    lp = torch.log_softmax(logits[j, torch.from_numpy(li).to(device)], 0).cpu().numpy()
                    q_legal[qr[j], :len(li)] = li[:MAXL]
                    q_logp[qr[j], :len(li)] = lp[:MAXL]
            print(f"  {b0 + len(enc):,}/{len(rows):,} games, {base + ob[-1]:,}/{P:,} positions, "
                  f"{(base + ob[-1]) / (time.time() - t0):,.0f}/s", flush=True)
    keys.flush()
    np.save(out / "played.npy", played)
    np.save(out / "off.npy", off)
    np.savez(out / "query.npz", pos=qpos, legal=q_legal, logp=q_logp)
    return off


def states(device: str, workers: int, batch: int, limit: int = 0, band: int = 1500):
    o = out_dir(band)
    root = o.with_name(o.name + "_smoke") if limit else o
    for split, n_players, n_store, n_query in (("valrich", 10**9, 40, 8), ("train", 2000, 40, 10)):
        g, groups = groups_for(split, n_players, n_store, n_query, limit, band=band)
        print(f"{split}: {len(groups):,} players, {g.height:,} games", flush=True)

        def qpos(off, groups=groups):
            return np.concatenate([query_rows(off, qg, qc)[0] for _, _, qg, qc in groups])
        off = encode(g, qpos, root / split, device, workers, batch)
        np.save(root / split / "groups.npy", np.array(groups, dtype=object), allow_pickle=True)
        print(f"  -> {root / split}", flush=True)


# ------------------------------------------------------------------ head
def query_rows(off, gis, cols):
    """Global positions and plies of the player's own moves at ply >= 10 in their query games (colour cc moves
    at plies with ply % 2 == cc)."""
    pos = np.concatenate([np.arange(off[a] + 10 + cc, off[a + 1], 2) for a, cc in zip(gis, cols)]
                         + [np.zeros(0, np.int64)])
    ply = np.concatenate([np.arange(10 + cc, off[a + 1] - off[a], 2) for a, cc in zip(gis, cols)]
                         + [np.zeros(0, np.int64)])
    return pos, ply


def store_rows(off, gis, cols):
    return np.concatenate([np.arange(off[a] + cc, off[a + 1], 2) for a, cc in zip(gis, cols)] + [np.zeros(0, np.int64)])


class Head:
    """Built lazily so the module imports without torch."""

    @staticmethod
    def make():
        import torch
        import torch.nn as nn

        class _Head(nn.Module):
            def __init__(self):
                super().__init__()
                w = torch.linalg.qr(torch.randn(D, H, generator=torch.Generator().manual_seed(0)))[0].T.contiguous()
                self.wq = nn.Parameter(w.clone())
                self.wk = nn.Parameter(w.clone())
                self.log_tau = nn.Parameter(torch.tensor(float(np.log(0.05))))
                self.log_eps = nn.Parameter(torch.tensor(float(np.log(0.01))))
                self.gate = nn.Sequential(nn.Linear(5, 16), nn.Tanh(), nn.Linear(16, 1))
                nn.init.constant_(self.gate[2].bias, -4.0)

            def forward(self, kq, ks, col, ply, logp):
                """kq (Q, D), ks (S, D), col (Q, S) legal column of the store move or -1, ply (Q,), logp (Q, L)."""
                q = torch.nn.functional.normalize(kq @ self.wq.T, dim=1)
                s = torch.nn.functional.normalize(ks @ self.wk.T, dim=1)
                sim = q @ s.T
                col = col.long()
                ok = col >= 0
                w = torch.where(ok, torch.exp((sim - 1) / self.log_tau.exp()), torch.zeros_like(sim))
                mass = torch.zeros_like(logp).scatter_add_(1, col.clamp(min=0), w)
                tot = mass.sum(1, keepdim=True)
                pk = mass / (tot + 1e-6)
                eps = self.log_eps.exp()
                r = torch.log(eps + pk) - torch.log(eps)
                msim = torch.where(ok, sim, torch.full_like(sim, -1.0)).max(1).values if sim.shape[1] else \
                    torch.full((len(kq),), -1.0)
                feat = torch.stack([msim, torch.log(tot[:, 0] + 1e-6), ply / 100, (ply < 20).float(),
                                    torch.full_like(ply, float(np.log1p(ks.shape[0])))], 1)
                g = torch.nn.functional.softplus(self.gate(feat))
                return logp + g * r
        return _Head()


def tensors(keys, played, qz, qrow_of, off, store_g, store_c, qpos, qply):
    """Torch inputs for one player: query / store keys, store-move -> query legal column, ply, Maia log-probs."""
    import torch
    st = store_rows(off, store_g, store_c)
    rows = qrow_of[qpos]
    legal = qz["legal"][rows].astype(np.int64)
    logp = np.nan_to_num(qz["logp"][rows].astype(np.float32), neginf=-1e9)
    sm = played[st].astype(np.int64)
    col = np.full((len(qpos), len(st)), -1, np.int16)
    if len(st):
        hit = legal[:, None, :] == sm[None, :, None]                 # (Q, S, L)
        col = np.where(hit.any(2), hit.argmax(2), -1).astype(np.int16)    # int16: ~2 GB for train
    truth = np.argmax(legal == played[qpos].astype(np.int64)[:, None], 1)
    return (torch.from_numpy(keys[qpos].astype(np.float32)), torch.from_numpy(keys[st].astype(np.float32)),
            torch.from_numpy(col), torch.from_numpy(qply.astype(np.float32)), torch.from_numpy(logp),
            torch.from_numpy(truth), st)


def load_split(root: Path):
    keys = np.load(root / "keys.npy", mmap_mode="r")
    played = np.load(root / "played.npy")
    off = np.load(root / "off.npy")
    qz = dict(np.load(root / "query.npz"))
    qrow_of = np.full(int(off[-1]), -1, np.int64)
    qrow_of[qz["pos"]] = np.arange(len(qz["pos"]))
    groups = np.load(root / "groups.npy", allow_pickle=True)
    batches = []
    for sg, sc, qg, qc in groups:
        qpos, qply = query_rows(off, qg, qc)
        qpos, qply = qpos[qrow_of[qpos] >= 0], qply[qrow_of[qpos] >= 0]
        if len(qpos):
            batches.append(tensors(keys, played, qz, qrow_of, off, sg, sc, qpos, qply)[:6])
    return batches


def evaluate(head, batches):
    import torch
    hit = base = n = 0
    with torch.no_grad():
        for kq, ks, col, ply, logp, truth in batches:
            out = head(kq, ks, col, ply, logp)
            hit += (out.argmax(1) == truth).sum().item()
            base += (logp.argmax(1) == truth).sum().item()
            n += len(truth)
    return 100 * (hit - base) / max(n, 1), n


def train(epochs: int, lr: float, smoke: bool = False, band: int = 1500):
    import torch
    o = out_dir(band)
    root = o.with_name(o.name + "_smoke") if smoke else o
    torch.manual_seed(0)
    tr, va = load_split(root / "train"), load_split(root / "valrich")
    print(f"train {len(tr):,} players / {sum(len(b[5]) for b in tr):,} queries; "
          f"valrich {len(va):,} / {sum(len(b[5]) for b in va):,}", flush=True)
    head = Head.make()
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    best, log = (-1e9, None), []
    g0, _ = evaluate(head, va)
    print(f"epoch 0: valrich gain {g0:+.3f} pts", flush=True)
    rng = np.random.default_rng(0)
    for ep in range(1, epochs + 1):
        t0, tot = time.time(), 0.0
        for i in rng.permutation(len(tr)):
            kq, ks, col, ply, logp, truth = tr[i]
            loss = torch.nn.functional.cross_entropy(head(kq, ks, col, ply, logp), truth)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        gv, _ = evaluate(head, va)
        log.append({"epoch": ep, "train_ce": tot / len(tr), "valrich_gain_pts": gv, "tau": head.log_tau.exp().item(),
                    "eps": head.log_eps.exp().item()})
        print(json.dumps(log[-1]), f"{time.time() - t0:.0f}s", flush=True)
        if gv > best[0]:
            best = (gv, {k: v.clone() for k, v in head.state_dict().items()})
    torch.save({"state": best[1], "valrich_gain_pts": best[0], "log": log}, root / "head.pt")
    print(f"best valrich gain {best[0]:+.3f} -> {root / 'head.pt'}", flush=True)


# ------------------------------------------------------------------ eval on Allie 1400-1600
def eval_allie(limit: int = 0, band: int = 1500, write_results: bool = True, knn_smoke: bool = False):
    import torch

    from doppel_lab.allie_knn import build_queries, knn_dir
    hb = HEAD_BAND.get(band, band)
    o = out_dir(hb)
    root = o.with_name(o.name + "_smoke") if limit else o
    ck = torch.load(root / "head.pt", weights_only=False)
    head = Head.make()
    head.load_state_dict(ck["state"])
    head.eval()
    kd = knn_dir(band)
    if knn_smoke:
        kd = kd.with_name(kd.name + "_smoke")
    keys = np.load(kd / "keys.npy", mmap_mode="r")
    played = np.load(kd / "played.npy")
    tz = dict(np.load(kd / "test.npz"))
    trow = np.full(len(played), -1, np.int64)
    trow[tz["pos"]] = np.arange(len(tz["pos"]))
    sides = build_queries(band) if not knn_smoke else _smoke_sides(band, kd)
    perm = np.random.default_rng(0).permutation(len(sides))       # step 0's control permutation
    if limit:                                                      # smoke: first N sides, no results file
        sides, perm = sides[:limit], np.arange(1, limit + 1) % limit
    rec = {0: [], 1: []}
    with torch.no_grad():
        for si, (gid, plies, q, st) in enumerate(sides):
            rows = trow[q]
            legal = tz["legal"][rows].astype(np.int64)
            logp = np.nan_to_num(tz["logp"][rows].astype(np.float32), neginf=-1e9)
            truth = np.argmax(legal == played[q].astype(np.int64)[:, None], 1)
            kq = keys[q].astype(np.float32)
            for ctrl, store in ((0, st), (1, sides[perm[si]][3])):
                ks = keys[store].astype(np.float32)
                sm = played[store].astype(np.int64)
                hit = legal[:, None, :] == sm[None, :, None] if len(store) else np.zeros((len(q), 0, MAXL), bool)
                col = np.where(hit.any(2), hit.argmax(2), -1) if len(store) else np.zeros((len(q), 0), np.int64)
                out = head(torch.from_numpy(kq), torch.from_numpy(ks), torch.from_numpy(col),
                           torch.from_numpy(plies.astype(np.float32)), torch.from_numpy(logp))
                cos = (kq @ ks.T).max(1) if len(store) else np.full(len(q), -1.0)
                rec[ctrl].append((np.full(len(q), gid, object), plies, out.argmax(1).numpy() == truth,
                                  logp.argmax(1) == truth, cos))
    mem = pl.read_parquet(ROOT / "data/allie/memory_positions.parquet").select("game_id", ply="k", hist="hist")
    rng = np.random.default_rng(0)
    res = {"band": band, "head_band": hb, "head": {"valrich_gain_pts": ck["valrich_gain_pts"], "epochs": ck["log"]}}
    gid0, ply0, h0, mh0, _ = (np.concatenate(x) for x in zip(*rec[0]))
    h1 = np.concatenate([r[2] for r in rec[1]])
    if not limit:
        pl.DataFrame({"game_id": gid0.astype(str), "ply": ply0.astype(np.int64), "hit": h0, "maia_hit": mh0,
                      "ctrl_hit": h1}).write_parquet(positions_path(band))
    for cname, cv in (("personal", 0), ("shuffled", 1)):
        gid, ply, h, mh, cos = (np.concatenate(x) for x in zip(*rec[cv]))
        hist = pl.DataFrame({"game_id": gid.astype(str), "ply": ply}).join(mem, on=["game_id", "ply"], how="left")[
            "hist"].fill_null(False).to_numpy()
        half = np.array([zlib.crc32(x.encode()) % 2 for x in gid.astype(str)])

        def gain(sel):
            return {"positions": int(sel.sum()), "gain_pts": 100 * float(h[sel].mean() - mh[sel].mean()),
                    "maia": float(mh[sel].mean())}
        r = {"overall": gain(np.ones(len(h), bool)), "exact_repeat_hist": gain(hist), "no_exact_history": gain(~hist),
             "half0": gain(half == 0), "half1": gain(half == 1), "by_ply": {}, "by_best_cos": {}}
        for lo, hi in ((10, 19), (20, 29), (30, 39), (40, 9999)):
            r["by_ply"][f"{lo}-{hi}"] = gain((ply >= lo) & (ply <= hi))
        for lo, hi in ((-1, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 0.98), (0.98, 0.995), (0.995, 2)):
            r["by_best_cos"][f"{lo}-{hi}"] = gain((cos >= lo) & (cos < hi))
        gi = np.unique(gid.astype(str), return_inverse=True)[1]
        d = h.astype(float) - mh
        gs, gn = np.bincount(gi, d), np.bincount(gi)
        b = rng.integers(0, len(gs), (2000, len(gs)))
        bs = 100 * gs[b].sum(1) / gn[b].sum(1)
        r["overall"]["ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        res[cname] = r
    if not limit and write_results:
        results_path(band).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k]["overall"] for k in ("personal", "shuffled")}, indent=1))


def _smoke_sides(band: int, kd: Path):
    """Smoke only: test sides restricted to the games present in knn_<band>_smoke (same off layout)."""
    from doppel_lab.allie_knn import games
    from doppel_lab.allie_v0e import test_games
    off = np.load(kd / "off.npy")
    g = games(band)
    tid = test_games(band)["game_id"].implode()
    n = len(off) - 1
    lim = n - 20
    g = pl.concat([g.filter(~c("game_id").is_in(tid)).head(lim), g.filter(c("game_id").is_in(tid)).head(20)])
    cut = dict(zip(test_games(band)["game_id"].to_list(), test_games(band)["cut"].to_list()))
    out = []
    for gi, r in enumerate(g.iter_rows(named=True)):
        if r["game_id"] not in cut:
            continue
        for col in (0, 1):
            plies = np.array([p for p in range(col, len(r["moves"]), 2) if 10 <= p < cut[r["game_id"]]])
            if len(plies):
                st = np.concatenate([np.arange(off[a] + 0, off[a + 1], 2) for a in range(min(lim, 5))])
                out.append((r["game_id"], plies, off[gi] + plies, st))
    return out


# ------------------------------------------------------------------ full test set
def full():
    """All five Allie buckets: learned lookup vs Maia-3 on the same positions (McNemar + game bootstrap)."""
    import math

    from doppel_lab.allie_v0e import BANDS
    x = pl.concat([pl.read_parquet(positions_path(b)).with_columns(bucket=pl.lit(b)) for b in sorted(BANDS)])
    dup = x.height - x.unique(["game_id", "ply"]).height
    res = {"positions": x.height, "games": x["game_id"].n_unique(), "duplicate_keys": dup,
           "head_band": {str(b): HEAD_BAND.get(b, b) for b in sorted(BANDS)}, "by_bucket": {}}
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
    res["lookup_vs_maia"] = paired(x, "hit", "maia_hit")
    res["shuffled_vs_maia"] = paired(x, "ctrl_hit", "maia_hit")
    for b in sorted(BANDS):
        d = x.filter(c("bucket") == b)
        res["by_bucket"][str(b)] = {"positions": d.height, **paired(d, "hit", "maia_hit")}
    p = ROOT / "lab/results/allie_retr1_full.json"
    p.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "by_bucket"}, indent=1))
    for b, r in res["by_bucket"].items():
        print(b, r["positions"], f"{r['gain_pts']:+.3f}", [round(v, 3) for v in r["ci95"]])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("states")
    s.add_argument("--gpu-ok", action="store_true")
    s.add_argument("--workers", type=int, default=12)
    s.add_argument("--batch", type=int, default=2048)
    s.add_argument("--limit", type=int, default=0, help="CPU smoke: about N games per split, to retr1_smoke/")
    s.add_argument("--band", type=int, default=1500)
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=8)
    t.add_argument("--lr", type=float, default=3e-3)
    t.add_argument("--smoke", action="store_true")
    t.add_argument("--band", type=int, default=1500)
    e = sub.add_parser("eval")
    e.add_argument("--limit", type=int, default=0, help="smoke: first N test sides, smoke head")
    e.add_argument("--band", type=int, default=1500, help="Allie bucket (BANDS key in allie_v0e)")
    e.add_argument("--no-results", action="store_true", help="write only the per-position rows")
    e.add_argument("--knn-smoke", action="store_true", help="smoke: use knn_<band>_smoke states")
    f = sub.add_parser("full", help="combine all five buckets -> lab/results/allie_retr1_full.json")
    a = ap.parse_args()
    if a.cmd == "states":
        if not a.gpu_ok and not a.limit:
            raise SystemExit("refusing to touch the GPU without --gpu-ok (run through the warden)")
        states("cuda" if a.gpu_ok else "cpu", a.workers, a.batch, a.limit, a.band)
    elif a.cmd == "train":
        train(a.epochs, a.lr, a.smoke, a.band)
    elif a.cmd == "eval":
        eval_allie(a.limit, a.band, not a.no_results, a.knn_smoke)
    else:
        full()


if __name__ == "__main__":
    main()

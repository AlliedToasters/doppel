"""How many earlier games does the learned lookup need? Store = the side's k most recent earlier games (k in KS),
same head (trained with 40), same Allie positions. CPU.

    python -m doppel_lab.lookup_k [--bands 1500 1900]   -> lab/results/lookup_k.json

Mirrors retr_train.eval_allie (personal store only); the k = 40 row reproduces allie_retr1_<band>.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
KS = (1, 2, 5, 10, 20, 40)
c = pl.col


def sides_with_games(band: int):
    """build_queries, but the store as a most-recent-first list of (game row, colour)."""
    from doppel_lab.allie_knn import games, knn_dir
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
           .sort(["si", "t"], descending=[False, True]).group_by("si", maintain_order=True).head(CTX))
    store = {si: (gis, cols) for si, gis, cols in ctx.group_by("si", maintain_order=True).agg("gi", "col").iter_rows()}
    out = []
    for si, (gid, gi, col, name, tq, plies) in enumerate(sides):
        gis, cols = store.get(si, ([], []))
        out.append((gid, np.array(plies), off[gi] + np.array(plies), list(gis), list(cols)))
    return out, off


def run(band: int) -> dict:
    import torch

    from doppel_lab.allie_knn import knn_dir
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
    sides, off = sides_with_games(band)
    hit = {k: [] for k in KS}
    maia, gids, nhist = [], [], []
    with torch.no_grad():
        for gid, plies, q, gis, cols in sides:
            rows = trow[q]
            legal = tz["legal"][rows].astype(np.int64)
            logp = np.nan_to_num(tz["logp"][rows].astype(np.float32), neginf=-1e9)
            truth = np.argmax(legal == played[q].astype(np.int64)[:, None], 1)
            kq = torch.from_numpy(keys[q].astype(np.float32))
            maia.append(logp.argmax(1) == truth)
            gids.append(np.full(len(q), gid, object))
            nhist.append(np.full(len(q), len(gis)))
            for k in KS:
                st = (np.concatenate([np.arange(off[a] + cc, off[a + 1], 2) for a, cc in zip(gis[:k], cols[:k])])
                      if gis else np.zeros(0, np.int64))
                ks = keys[st].astype(np.float32)
                sm = played[st].astype(np.int64)
                h = legal[:, None, :] == sm[None, :, None] if len(st) else np.zeros((len(q), 0, legal.shape[1]), bool)
                col = np.where(h.any(2), h.argmax(2), -1) if len(st) else np.zeros((len(q), 0), np.int64)
                out = head(kq, torch.from_numpy(ks), torch.from_numpy(col), torch.from_numpy(plies.astype(np.float32)),
                           torch.from_numpy(logp))
                hit[k].append(out.argmax(1).numpy() == truth)
    maia, gid, nh = np.concatenate(maia), np.concatenate(gids).astype(str), np.concatenate(nhist)
    gi = np.unique(gid, return_inverse=True)[1]
    rng = np.random.default_rng(0)
    b = rng.integers(0, gi.max() + 1, (1000, gi.max() + 1))
    res = {"band": band, "positions": len(maia), "maia": float(maia.mean()),
           "sides_with_40": float((nh >= 40).mean()), "by_k": {}}
    for k in KS:
        h = np.concatenate(hit[k])
        d = h.astype(float) - maia
        gs, gn = np.bincount(gi, d), np.bincount(gi)
        bs = 100 * gs[b].sum(1) / gn[b].sum(1)
        res["by_k"][str(k)] = {"gain_pts": 100 * float(d.mean()),
                               "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                               "gain_pts_full40": 100 * float(d[nh >= 40].mean())}
        print(band, k, f"{100 * d.mean():+.3f}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", type=int, nargs="+", default=[1500, 1900])
    a = ap.parse_args()
    p = ROOT / "lab/results/lookup_k.json"
    res = json.loads(p.read_text()) if p.exists() else {}
    for band in a.bands:
        res[str(band)] = run(band)
        p.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

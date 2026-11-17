"""The learned lookup on the lora_probe players: same test positions, store = the same N training games, so the
per-player adapter and the lookup are compared on identical data.

    python -m doppel_lab.probe_lookup states --gpu-ok     # GPU (warden): keys + Maia-3 log-probs (retr_train.encode)
    python -m doppel_lab.probe_lookup eval                # CPU -> lab/results/probe_lookup.json

Head: retr_train's head of the player's band (1500 or 1900). Queries: the player's own moves at ply >= 10 in their
20 test games. Store: their own moves in the N most recent training games (tr_rank < N), N in lora_probe.NS.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from doppel_lab.lora_probe import BANDS, NS, OUT as PROBE, PLY_BINS

ROOT = Path(__file__).resolve().parents[2]
OUT = PROBE / "lookup"
c = pl.col


def rows() -> pl.DataFrame:
    return pl.read_parquet(PROBE / "games.parquet").sort("band", "player", "j").with_row_index("ri")


def qmask(d: pl.DataFrame):
    col, n, role = d["col"].to_numpy(), d["moves"].list.len().to_numpy(), d["role"].to_numpy()

    def f(off):
        return np.concatenate([off[i] + np.arange(10 + col[i], n[i], 2) for i in np.flatnonzero(role == "test")])
    return f


def states(workers: int, batch: int):
    from doppel_lab.retr_train import encode
    d = rows()
    encode(d, qmask(d), OUT, "cuda", workers, batch)


def evaluate():
    import torch

    from doppel_lab.retr_train import Head, out_dir
    d = rows()
    keys = np.load(OUT / "keys.npy", mmap_mode="r")
    played = np.load(OUT / "played.npy")
    off = np.load(OUT / "off.npy")
    qz = dict(np.load(OUT / "query.npz"))
    qrow = np.full(int(off[-1]), -1, np.int64)
    qrow[qz["pos"]] = np.arange(len(qz["pos"]))
    heads = {}
    for b in BANDS:
        ck = torch.load(out_dir(b) / "head.pt", weights_only=False)
        h = Head.make(); h.load_state_dict(ck["state"]); h.eval(); heads[b] = h
    rec = {N: [] for N in NS}
    base = []
    with torch.no_grad():
        for (player,), g in d.group_by("player", maintain_order=True):
            band = int(g["band"][0])
            ri, col, n = g["ri"].to_numpy(), g["col"].to_numpy(), g["moves"].list.len().to_numpy()
            role, tr = g["role"].to_numpy(), g["tr_rank"].to_numpy()
            te = np.flatnonzero(role == "test")
            q = np.concatenate([off[ri[i]] + np.arange(10 + col[i], n[i], 2) for i in te])
            ply = np.concatenate([np.arange(10 + col[i], n[i], 2) for i in te])
            game = np.concatenate([np.full(len(range(10 + col[i], n[i], 2)), ri[i]) for i in te])
            r = qrow[q]
            legal = qz["legal"][r].astype(np.int64)
            logp = np.nan_to_num(qz["logp"][r].astype(np.float32), neginf=-1e9)
            truth = np.argmax(legal == played[q].astype(np.int64)[:, None], 1)
            kq = torch.from_numpy(keys[q].astype(np.float32))
            base.append((game, ply, logp.argmax(1) == truth, np.full(len(q), band)))
            for N in NS:
                sel = np.flatnonzero((role == "train") & (tr < N))
                st = np.concatenate([np.arange(off[ri[i]] + col[i], off[ri[i] + 1], 2) for i in sel])
                ks = keys[st].astype(np.float32)
                sm = played[st].astype(np.int64)
                out = []
                for s in range(0, len(q), 256):          # chunk: (Q, S, L) at S ~ 20k
                    lq = legal[s:s + 256]
                    hit = lq[:, None, :] == sm[None, :, None]
                    cc = np.where(hit.any(2), hit.argmax(2), -1)
                    o = heads[band](kq[s:s + 256], torch.from_numpy(ks), torch.from_numpy(cc),
                                    torch.from_numpy(ply[s:s + 256].astype(np.float32)), torch.from_numpy(logp[s:s + 256]))
                    out.append(o.argmax(1).numpy())
                rec[N].append(np.concatenate(out) == truth)
            print(player[:0] + f"{band} done {len(rec[NS[0]])}", flush=True) if len(rec[NS[0]]) % 20 == 0 else None
    game, ply, maia, bandv = (np.concatenate(x) for x in zip(*base))
    rng = np.random.default_rng(0)
    gi = np.unique(game, return_inverse=True)[1]
    res = {"positions": len(maia), "maia_top1": float(maia.mean()), "runs": {}}

    def gain(h, sel):
        dd = h[sel].astype(float) - maia[sel]
        g = np.unique(gi[sel], return_inverse=True)[1]
        gs, gn = np.bincount(g, dd), np.bincount(g)
        b = rng.integers(0, len(gs), (1000, len(gs)))
        bs = 100 * gs[b].sum(1) / gn[b].sum(1)
        return {"positions": int(sel.sum()), "top1_gain_pts": 100 * float(dd.mean()),
                "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    for N in NS:
        h = np.concatenate(rec[N])
        r = {"all": gain(h, np.ones(len(h), bool)), "by_ply": {}, "by_band": {}}
        for lo, hi in PLY_BINS:
            r["by_ply"][f"{lo}-{hi}"] = gain(h, (ply >= lo) & (ply <= hi))
        for b in BANDS:
            r["by_band"][str(b)] = gain(h, bandv == b)
        res["runs"][str(N)] = r
        print(N, round(r["all"]["top1_gain_pts"], 3), {k: round(v["top1_gain_pts"], 3) for k, v in r["by_ply"].items()},
              flush=True)
    p = ROOT / "lab/results/probe_lookup.json"
    p.write_text(json.dumps(res, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("states")
    s.add_argument("--gpu-ok", action="store_true")
    s.add_argument("--workers", type=int, default=12)
    s.add_argument("--batch", type=int, default=2048)
    sub.add_parser("eval")
    a = ap.parse_args()
    if a.cmd == "states":
        if not a.gpu_ok:
            raise SystemExit("refusing to touch the GPU without --gpu-ok (run through the warden)")
        states(a.workers, a.batch)
    else:
        evaluate()


if __name__ == "__main__":
    main()

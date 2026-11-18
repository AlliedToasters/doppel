"""Bigger lookup store: does the learned lookup keep gaining past 40 earlier games? Allie 1400-1600 bucket, the same
head (retr_train, trained with stores of 40), stores of the side's k most recent earlier 2022 games, k up to CTX.

    python -m doppel_lab.lookup_store select                 # CPU: test games + each side's CTX earlier games
    python -m doppel_lab.lookup_store states --gpu-ok        # GPU (warden): keys in the step-0 key space (1500 PCA)
    python -m doppel_lab.lookup_store eval                   # CPU -> lab/results/lookup_store.json

Same rules as allie_v0e.select (context strictly before the test game, test games never context). The k <= 40 rows
reproduce lookup_k.json up to players whose 40 most recent games are the same either way (they are).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
BAND = 1500
CTX = 200
OUT = ROOT / f"data/allie/store{CTX}_{BAND}"


def out(band: int) -> Path:
    return ROOT / f"data/allie/store{CTX}_{band}"
KS = (10, 20, 40, 80, 120, 200)
c = pl.col


def select(band: int = BAND):
    from doppel_lab.allie_v0e import HIST, TEST, test_games
    t = test_games(band)
    all_test = set(pl.read_parquet(TEST, columns=["game_id"])["game_id"].to_list())
    sides = pl.concat([t.select("game_id", "created_at", player=c(col)) for col in ("white", "black")]).drop_nulls()
    players = sides["player"].str.to_lowercase().unique()
    sides = sides.with_columns(c("player").str.to_lowercase())
    hist = []
    for f in sorted(HIST.glob("*.parquet")):
        h = pl.read_parquet(f).with_columns(wl=c("white").str.to_lowercase(), bl=c("black").str.to_lowercase())
        hist.append(h.filter((c("wl").is_in(players.implode()) | c("bl").is_in(players.implode()))
                             & ~c("game_id").is_in(list(all_test))))
        print(f"  {f.name}: {hist[-1].height:,}", flush=True)
    h = pl.concat(hist)
    hs = pl.concat([h.select(hid="game_id", utc_ms="utc_ms", player=c(col)) for col in ("wl", "bl")])
    ctx = (sides.join(hs, on="player").filter(c("utc_ms") < c("created_at"))
           .sort("utc_ms", descending=True).group_by("game_id", "player").head(CTX))
    keep = ctx["hid"].unique()
    hg = h.filter(c("game_id").is_in(keep.implode())).unique("game_id").rename({"utc_ms": "t"}).drop("wl", "bl")
    tg = t.with_columns(t=c("created_at")).select(hg.columns)
    g = pl.concat([tg, hg]).sort("t")
    o = out(band)
    o.mkdir(parents=True, exist_ok=True)
    g.write_parquet(o / "games.parquet")
    per = ctx.group_by("game_id", "player").len()["len"]
    print(f"{t.height:,} test games, {sides.height:,} sides, context games {hg.height:,}; per side median "
          f"{per.median():.0f}, >= 80: {(per >= 80).mean():.2f}, >= 200: {(per >= 200).mean():.2f}; "
          f"{g['moves'].list.len().sum():,} positions -> {o / 'games.parquet'}", flush=True)


def games(band: int = BAND) -> pl.DataFrame:
    return pl.read_parquet(out(band) / "games.parquet", columns=["game_id", "white", "black", "white_elo", "black_elo",
                                                          "moves", "t"])


def knn_dir(band: int = BAND) -> Path:
    return OUT


def states(device: str, workers: int, batch: int):
    """allie_knn.states with the 1500 PCA loaded (never refit), on this store's games."""
    import torch

    from doppel_lab.allie_knn import BLOCK, D, MAXL
    from doppel_lab.allie_knn import knn_dir as kd0
    from doppel_lab.allie_v0e import test_games
    from doppel_lab.engines import Maia3
    from doppel_lab.teacher import HIST, encode_game
    g = games()
    tid = test_games(BAND)["game_id"].implode()
    is_test = g["game_id"].is_in(tid).to_numpy()
    n_per = g["moves"].list.len().to_numpy()
    off = np.r_[0, np.cumsum(n_per)].astype(np.int64)
    P = int(off[-1])
    tpos = np.concatenate([np.zeros(0, np.int64)] + [np.arange(off[i], off[i + 1]) for i in np.flatnonzero(is_test)])
    trow = np.full(P, -1, np.int64)
    trow[tpos] = np.arange(len(tpos))
    keys = np.lib.format.open_memmap(OUT / "keys.npy", "w+", np.float16, (P, D))
    played = np.zeros(P, np.int16)
    t_legal = np.full((len(tpos), MAXL), -1, np.int16)
    t_logp = np.full((len(tpos), MAXL), -np.inf, np.float16)
    model = Maia3("79m", device).model.eval()
    pooled = {}
    model.last_ln.register_forward_hook(lambda m, i, o: pooled.__setitem__("x", o))
    pca = np.load(kd0(1500) / "pca.npz")
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
                tr = trow[base + s:base + e_]
                for j in np.flatnonzero(tr >= 0):
                    li = legal[lstart[s + j]:lstart[s + j + 1]]
                    lp = torch.log_softmax(logits[j, torch.from_numpy(li).to(device)], 0).cpu().numpy()
                    t_legal[tr[j], :len(li)] = li[:MAXL]
                    t_logp[tr[j], :len(li)] = lp[:MAXL]
            print(f"{b0 + len(enc):,}/{len(rows):,} games, {base + ob[-1]:,}/{P:,} positions, "
                  f"{(base + ob[-1]) / (time.time() - t0):,.0f}/s", flush=True)
    keys.flush()
    np.save(OUT / "played.npy", played)
    np.save(OUT / "off.npy", off)
    np.savez(OUT / "test.npz", pos=tpos, legal=t_legal, logp=t_logp)
    print(f"done -> {OUT}", flush=True)


def evaluate():
    """lookup_k.run with this store (monkeypatched paths and CTX)."""
    from doppel_lab import allie_knn, allie_v0e, lookup_k
    allie_knn.games, allie_knn.knn_dir, allie_v0e.CTX = games, knn_dir, CTX
    lookup_k.KS = KS
    res = lookup_k.run(BAND)
    p = ROOT / "lab/results/lookup_store.json"
    p.write_text(json.dumps({"ctx": CTX, **res}, indent=1) + "\n")
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sel = sub.add_parser("select")
    sel.add_argument("--band", type=int, default=BAND, help="context selection only; states/eval stay 1500")
    s = sub.add_parser("states")
    s.add_argument("--gpu-ok", action="store_true")
    s.add_argument("--workers", type=int, default=12)
    s.add_argument("--batch", type=int, default=2048)
    sub.add_parser("eval")
    a = ap.parse_args()
    if a.cmd == "select":
        select(a.band)
    elif a.cmd == "states":
        if not a.gpu_ok:
            raise SystemExit("refusing to touch the GPU without --gpu-ok (run through the warden)")
        states("cuda", a.workers, a.batch)
    else:
        evaluate()


if __name__ == "__main__":
    main()

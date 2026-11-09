"""Eval approximator, fine-tuned: Maia-3 79M trunk + a new win-chance head trained end to end on our Stockfish labels.
The frozen-feature probe (evalnet.py) asks how much eval the trunk already knows; this asks how close a net of this
size gets when it is allowed to learn it.

    python -m doppel_lab.evalnet_ft train --games 80000 --gpu-ok     # GPU via the warden -> data/probe/evalnet_ft.pt,
                                                                     #   evalnet_ft_{val,human}.npz (predictions)
    python -m doppel_lab.evalnet_ft score                            # CPU -> lab/results/evalnet_ft.json,
                                                                     #   data/evals/human_evalnet_ft.parquet
    python -m doppel_lab.evalnet science --hat ft                    # the game_science swap test with these evals

Train: 1500-band train games with labels, excluding none of the probe's (it trains on the same pool). Val: exactly
the probe's 5,000 val games (same seed), so the two are scored on identical positions. Both Elos fixed at 1500.
Target: side-to-move win chance wc(cp) in [-1, 1]; smooth L1 (beta 0.1). One pass over the training positions,
shuffled within blocks of 2,000 games. AdamW: trunk 5e-5, head 1e-3, linear warmup, cosine decay, bf16 autocast.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import time

import numpy as np
import polars as pl

from doppel_lab import evalnet as en
from doppel_lab.features.channels import wc
from doppel_lab.teacher import HIST, encode_game

ELO = en.ELO
BLOCK = 2000


def games_for(split: str, n: int | None) -> list:
    if split == "human_all" or split.endswith(".parquet"):  # every game, evals or not (dummy zeros: labels unused)
        src = "data/human/games_2026-10.parquet" if split == "human_all" else split
        g = pl.read_parquet(en.ROOT / src, columns=["game_id", "moves"]).filter(pl.col("moves").list.len() > 0)
        return [(gid, m, [0] * (len(m) + 1)) for gid, m in g.iter_rows()]
    if split == "human":
        ev = pl.read_parquet(en.HUMAN_EVALS)
        g = pl.read_parquet(en.ROOT / "data/human/games_2026-10.parquet", columns=["game_id", "moves"]).join(ev, on="game_id")
    else:
        ev = pl.read_parquet(en.V0 / f"evals_1500_{split}.parquet")
        g = pl.read_parquet(en.V0 / f"games_1500_{split}.parquet", columns=["game_id", "moves"]).join(ev, on="game_id")
    g = g.filter(pl.col("evals").list.len() == pl.col("moves").list.len() + 1)
    if n is not None:
        g = g.sample(min(n, g.height), seed=0, shuffle=True)
    return list(g.iter_rows())


def block_tensors(pool, blk, device):
    import torch
    enc = pool.map(encode_game, [(m, ELO, ELO) for _, m, _ in blk], chunksize=32)
    n_per = np.array([len(e[3]) for e in enc])
    off = np.r_[0, np.cumsum(n_per)]
    planes = torch.from_numpy(np.concatenate([e[0] for e in enc])).to(device)
    hidx = torch.from_numpy(np.concatenate([
        np.clip(np.arange(n)[:, None] + np.arange(-HIST + 1, 1)[None, :], 0, None) + o
        for n, o in zip(n_per, off[:-1])])).to(device)
    cp = np.concatenate([np.array(evs[:n], np.float64) for (_, _, evs), n in zip(blk, n_per)])
    ply = np.concatenate([np.arange(n) for n in n_per])
    y = np.where(ply % 2 == 0, 1.0, -1.0) * wc(cp)
    return planes, hidx, torch.from_numpy(y.astype(np.float32)).to(device), n_per, cp, ply


def inputs(planes, idx):
    import torch
    x = planes[idx].float().permute(0, 2, 1, 3).reshape(len(idx), 64, HIST * 12)
    return torch.cat([x, torch.zeros(len(idx), 64, 1, device=x.device)], 2)


class Net:
    def __init__(self, device):
        import torch
        from doppel_lab.engines import Maia3
        self.m = Maia3("79m", device).model.train()
        self.feats = {}
        self.m.last_ln.register_forward_hook(lambda mod, i, o: self.feats.__setitem__("x", o))
        d = self.m.last_ln.normalized_shape[0]
        self.head = torch.nn.Sequential(torch.nn.Linear(d, 256), torch.nn.GELU(), torch.nn.Linear(256, 1)).to(device)
        self.device = device

    def __call__(self, x):
        import torch
        e = torch.full((len(x),), ELO, dtype=torch.long, device=self.device)
        self.m(x, e, e)
        return torch.tanh(self.head(self.feats["x"].float())[:, 0])


def predict(net, pool, rows, device, batch=1024):
    import torch
    net.m.eval()
    P, CP, PLY, GI = [], [], [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        for b0 in range(0, len(rows), BLOCK):
            blk = rows[b0:b0 + BLOCK]
            planes, hidx, _, n_per, cp, ply = block_tensors(pool, blk, device)
            for s in range(0, len(hidx), batch):
                P.append(net(inputs(planes, hidx[s:s + batch])).float().cpu().numpy())
            CP.append(cp.astype(np.float32))
            PLY.append(ply.astype(np.int16))
            GI += [np.full(n, b0 + i, np.int32) for i, n in enumerate(n_per)]
    net.m.train()
    return {"pred": np.concatenate(P), "cp": np.concatenate(CP), "ply": np.concatenate(PLY),
            "game": np.concatenate(GI), "game_ids": np.array([r[0] for r in rows])}


def train(n_games: int, device: str, batch: int = 512, workers: int = 20, smoke: bool = False, epochs: int = 1):
    import torch
    torch.manual_seed(0)
    rows = games_for("train", n_games)
    val, hum = games_for("val", 5000), games_for("human", None)
    if smoke:  # CPU check of the whole path: a few games everywhere, outputs under *_smoke
        val, hum = val[:2], hum[:2]
    net = Net(device)
    opt = torch.optim.AdamW([{"params": net.m.parameters(), "lr": 5e-5}, {"params": net.head.parameters(), "lr": 1e-3}],
                            weight_decay=0.01)
    est_steps = int(epochs * sum(len(r[1]) for r in rows) / batch) + 1
    warm = 2 if smoke else 500
    base = [g["lr"] for g in opt.param_groups]
    rng = np.random.default_rng(0)
    t0, step, seen, run_loss = time.time(), 0, 0, []
    with mp.get_context("fork").Pool(workers) as pool:
        for ep, b0 in ((e, b) for e in range(epochs) for b in range(0, len(rows), BLOCK)):
            if b0 == 0 and ep > 0:
                rows = [rows[i] for i in rng.permutation(len(rows))]
            planes, hidx, y, *_ = block_tensors(pool, rows[b0:b0 + BLOCK], device)
            perm = torch.from_numpy(rng.permutation(len(hidx))).to(device)
            for s in range(0, len(perm) - batch + 1, batch):
                i = perm[s:s + batch]
                f = min(1.0, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(step / est_steps, 1.0)))
                for g, b in zip(opt.param_groups, base):
                    g["lr"] = b * f
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    p = net(inputs(planes, hidx[i]))
                loss = torch.nn.functional.smooth_l1_loss(p.float(), y[i], beta=0.1)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(net.m.parameters()) + list(net.head.parameters()), 1.0)
                opt.step()
                step += 1
                seen += batch
                run_loss.append(loss.item())
            el = time.time() - t0
            print(f"ep {ep + 1}/{epochs} {b0 + BLOCK:,}/{len(rows):,} games, step {step}/{est_steps}, loss {np.mean(run_loss[-200:]):.4f}, "
                  f"{seen / el:,.0f} pos/s", flush=True)
        en.OUT.mkdir(parents=True, exist_ok=True)
        sfx = "_smoke" if smoke else ("" if n_games == 80_000 and epochs == 1 else f"_g{n_games // 1000}k")
        sfx += f"_e{epochs}" if epochs > 1 else ""
        torch.save({"head": net.head.state_dict(), "trunk": net.m.state_dict(), "steps": step},
                   en.OUT / f"evalnet_ft{sfx}.pt")
        for name, rs in (("val", val), ("human", hum)):
            t1 = time.time()
            d = predict(net, pool, rs, device)
            np.savez(en.OUT / f"evalnet_ft_{name}{sfx}.npz", **d)
            print(f"{name}: {len(d['pred']):,} positions predicted, {len(d['pred']) / (time.time() - t1):,.0f}/s",
                  flush=True)
    print(f"done ({time.time() - t0:.0f}s)")


def label(device: str, ckpt: str = "", workers: int = 20, games: str = "", out: str = ""):
    """Screening pass: the fine-tuned net over every game of a games parquet (default: the 36k corpus, which
    Stockfish labels 12k of) -> data/evals format. Large files go in chunks of 50k games, one output part each."""
    import torch
    t0 = time.time()
    net = Net(device)
    s = torch.load(en.OUT / f"evalnet_ft{ckpt}.pt", map_location=device, weights_only=False)
    net.m.load_state_dict(s["trunk"])
    net.head.load_state_dict(s["head"])
    rows = games_for(games or "human_all", None)
    dest = en.ROOT / (out or f"data/evals/human_evalnet_ft{ckpt}_all.parquet")
    CH = 50_000
    with mp.get_context("fork").Pool(workers) as pool:
        for c0 in range(0, len(rows), CH):
            path = dest if len(rows) <= CH else dest.with_name(f"{dest.stem}.part{c0 // CH:03d}.parquet")
            if path.exists():  # resumable after preemption
                continue
            d = predict(net, pool, rows[c0:c0 + CH], device)
            en.write_hat(d, d["pred"].astype(np.float64), path)
            print(f"{min(c0 + CH, len(rows)):,}/{len(rows):,} games ({time.time() - t0:.0f}s)", flush=True)


def score(tag: str = ""):
    t0 = time.time()
    res = {}
    for name in ("val", "human") if not tag else ("val",):
        d = dict(np.load(en.OUT / f"evalnet_ft_{name}{tag}.npz"))
        d["y"] = (np.where(d["ply"] % 2 == 0, 1.0, -1.0) * wc(d["cp"].astype(float))).astype(np.float32)
        p = d["pred"].astype(np.float64)
        r = en.pos_scores(d["y"], p, d["ply"]) | {"moves": en.move_scores(d, p)}
        res[name] = r
        print(f"ft {name:6s} MAE {r['mae_pct']:5.2f} win% | R2 {r['r2']:+.3f} | blunder AUC {r['moves']['blunder_auc']:.3f}"
              f" P/R@rate {r['moves']['rate_matched']['precision']:.2f}/{r['moves']['rate_matched']['recall']:.2f} | "
              f"game mean-loss rho {r['moves']['game_mean_loss_spearman']:.3f}", flush=True)
        if name == "human":
            en.write_hat(d, p, en.ROOT / "data/evals/human_evalnet_ft.parquet")
    rf = en.RESULTS / "evalnet_ft.json"
    if tag:  # data-scaling runs: val scores only, filed under the main result
        main = json.loads(rf.read_text())
        main.setdefault("scaling", {})[tag.lstrip("_")] = res["val"]
        rf.write_text(json.dumps(main, indent=1) + "\n")
    else:
        rf.write_text(json.dumps(res, indent=1) + "\n")
    print(f"-> evalnet_ft.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--games", type=int, default=80_000)
    t.add_argument("--batch", type=int, default=512)
    t.add_argument("--gpu-ok", action="store_true")
    t.add_argument("--smoke", action="store_true")
    t.add_argument("--epochs", type=int, default=1)
    lb = sub.add_parser("label")
    lb.add_argument("--ckpt", default="", help="checkpoint suffix, e.g. _g100k_e2")
    lb.add_argument("--gpu-ok", action="store_true")
    lb.add_argument("--games", default="", help="games parquet relative to the repo (default: the 36k corpus)")
    lb.add_argument("--out", default="", help="output parquet relative to the repo")
    sc = sub.add_parser("score")
    sc.add_argument("--tag", default="", help="e.g. _g20k for a data-scaling run")
    a = ap.parse_args()
    if a.cmd == "train":
        train(a.games, "cuda" if a.gpu_ok else "cpu", a.batch, smoke=a.smoke, epochs=a.epochs)
    elif a.cmd == "label":
        label("cuda" if a.gpu_ok else "cpu", a.ckpt, games=a.games, out=a.out)
    else:
        score(a.tag)

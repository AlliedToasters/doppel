"""Ceiling probe: how much can a per-player adapter inside frozen Maia-3 gain, by ply, at N = 40 / 200 / 500 games?

    python -m doppel_lab.lora_probe select                      # CPU: heavy held-out players -> data/lora_probe/games.parquet
    python -m doppel_lab.lora_probe run --gpu-ok [--smoke]      # GPU (warden): train + eval -> lab/results/lora_probe.json

Players: held-out (split.held_out) 2026-10 players with >= 545 games in one band (1500: 68, 1900: 101), the band
column of the corpus, games with >= 20 plies. Per player, their last 545 such games in time order (corpus row):
  train = the N games just before val (N in NS, nested: N = 40 is the most recent 40 of the 500)
  val   = 5 games (early stopping, per player), test = the last 20 games.
Adapter: rank-R LoRA on linear1 / linear2 (MLP) of blocks 5-8 (Maia-3's attention uses packed in-proj weights, so
LoRA there would need a fork; the MLP is half of each block). B starts at 0, so step 0 is Maia-3 exactly. One adapter
per player, all players trained together in each batch (per-sample gathered LoRA). AdamW with weight decay, epoch
picked per player on val CE. Positions: the player's own moves (all plies in training; ply >= 10 scored).
Readout on test: top-1 and CE gain over Maia-3 by ply bin, game bootstrap; control = each player's test positions
through another player's (same band, same N) adapter. Maia-3 and the adapters are evaluated in the same precision.
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
OUT = ROOT / "data/lora_probe"
RESULTS = ROOT / "lab/results/lora_probe.json"
BANDS = (1500, 1900)
NS = (40, 200, 500)
NVAL, NTEST = 5, 20
NEED = max(NS) + NVAL + NTEST
BLOCKS = (4, 5, 6, 7)
R = 8
PLY_BINS = ((10, 19), (20, 29), (30, 39), (40, 9999))
c = pl.col


def select():
    from doppel_lab.split import held_out
    from doppel_lab.teacher import band_expr
    files = sorted((ROOT / "data/corpus/2026-10/heldout").glob("part-*.parquet"))
    g = pl.concat([pl.read_parquet(f, columns=["game_id", "white", "black", "white_elo", "black_elo", "band", "moves"])
                   for f in files]).with_row_index("t").filter(c("moves").list.len() >= 20)
    parts = []
    for band in BANDS:
        x = g.filter(band_expr(band))
        sides = pl.concat([x.select("t", "game_id", player=c(col).str.to_lowercase(), col=pl.lit(i, pl.Int8))
                           for i, col in enumerate(("white", "black"))])
        cnt = sides.group_by("player").len().filter(c("len") >= NEED)
        names = sorted(n for n in cnt["player"].to_list() if held_out(n))
        s = (sides.filter(c("player").is_in(names)).sort("t").group_by("player", maintain_order=True).tail(NEED)
             .with_columns(j=pl.int_range(pl.len()).over("player"), band=pl.lit(band)))
        parts.append(s)
        print(f"band {band}: {len(names)} players", flush=True)
    s = pl.concat(parts)
    # role by position from the end: test = last NTEST, val = NVAL before, train index = games back from val
    s = s.with_columns(back=NEED - 1 - c("j"))
    s = s.with_columns(role=pl.when(c("back") < NTEST).then(pl.lit("test"))
                       .when(c("back") < NTEST + NVAL).then(pl.lit("val")).otherwise(pl.lit("train")),
                       tr_rank=c("back") - NTEST - NVAL)          # 0 = most recent train game
    games = g.filter(c("game_id").is_in(s["game_id"].unique().implode())).select(
        "game_id", "white_elo", "black_elo", "moves")
    OUT.mkdir(parents=True, exist_ok=True)
    s.join(games, on="game_id").write_parquet(OUT / "games.parquet")
    print(f"{s.height:,} player-games -> {OUT / 'games.parquet'}", flush=True)


# ------------------------------------------------------------------ data
def encode_all(d: pl.DataFrame, workers: int):
    """Per row (player-game): own-move positions. Returns planes per game and per-position index arrays."""
    from doppel_lab.teacher import encode_game
    rows = list(d.select("moves", "white_elo", "black_elo").iter_rows())
    with mp.get_context("fork").Pool(workers) as pool:
        enc = pool.map(encode_game, rows, chunksize=64)
    return enc


class Positions:
    """Flattened own-move positions with history windows into one planes array."""

    def __init__(self, d: pl.DataFrame, enc):
        from doppel_lab.teacher import HIST
        planes, legal, lstart, played, es, eo, hidx = [], [], [0], [], [], [], []
        meta = {"pid": [], "role": [], "tr_rank": [], "ply": [], "game": []}
        base, lbase = 0, 0
        for gi, (e, r) in enumerate(zip(enc, d.iter_rows(named=True))):
            pl_, lg, cnt, pm, s_, o_ = e
            n = len(pm)
            planes.append(pl_)
            ls = np.r_[0, np.cumsum(cnt)]
            own = np.arange(int(r["col"]), n, 2)
            for k in own:
                legal.append(lg[ls[k]:ls[k + 1]])
                h = np.clip(np.arange(k - HIST + 1, k + 1), 0, None) + base
                hidx.append(h)
            played.append(pm[own]); es.append(s_[own]); eo.append(o_[own])
            meta["pid"].append(np.full(len(own), r["pid"])); meta["role"].append(np.full(len(own), r["role"]))
            meta["tr_rank"].append(np.full(len(own), r["tr_rank"])); meta["ply"].append(own)
            meta["game"].append(np.full(len(own), gi))
            base += n
        self.planes = np.concatenate(planes)
        self.hidx = np.stack(hidx).astype(np.int64)
        self.legal = legal
        self.played = np.concatenate(played).astype(np.int64)
        self.es, self.eo = np.concatenate(es).astype(np.int64), np.concatenate(eo).astype(np.int64)
        for k, v in meta.items():
            setattr(self, k, np.concatenate(v))

    def batch(self, idx, device):
        import torch
        from doppel_lab.teacher import HIST
        x = torch.from_numpy(self.planes[self.hidx[idx]]).to(device).float()       # (B, HIST, 64, 12)
        x = x.permute(0, 2, 1, 3).reshape(len(idx), 64, HIST * 12)
        x = torch.cat([x, torch.zeros(len(idx), 64, 1, device=device)], 2)
        L = max(len(self.legal[i]) for i in idx)
        lg = np.full((len(idx), L), -1, np.int64)
        for j, i in enumerate(idx):
            lg[j, :len(self.legal[i])] = self.legal[i]
        lg = torch.from_numpy(lg).to(device)
        truth = torch.from_numpy(np.argmax(lg.cpu().numpy() == self.played[idx][:, None], 1)).to(device)
        return (x, torch.from_numpy(self.es[idx]).to(device), torch.from_numpy(self.eo[idx]).to(device), lg, truth)


# ------------------------------------------------------------------ LoRA
class Lora:
    def __init__(self, model, n_players: int, device: str, r: int = R, scale: float = 1.0):
        import torch
        self.cur = None            # (B,) player index of each sample, or None = Maia-3
        self.params, self.hooks = [], []
        g = torch.Generator().manual_seed(0)
        for bi in BLOCKS:
            blk = model.transformer.layers[bi]
            for lin in (blk.linear1, blk.linear2):
                di, do = lin.in_features, lin.out_features
                A = torch.nn.Parameter((torch.randn(n_players, r, di, generator=g) / np.sqrt(di)).to(device))
                B = torch.nn.Parameter(torch.zeros(n_players, do, r, device=device))
                self.params += [A, B]
                self.hooks.append(lin.register_forward_hook(self._hook(A, B, scale)))

    def _hook(self, A, B, scale):
        import torch

        def f(mod, inp, out):
            if self.cur is None:
                return out
            x = inp[0]
            h = torch.einsum("bsd,brd->bsr", x.float(), A[self.cur])
            return out + scale * torch.einsum("bsr,bor->bso", h, B[self.cur]).to(out.dtype)
        return f

    def snapshot(self):
        return [p.detach().clone() for p in self.params]

    def load_rows(self, snap, rows):
        """Copy players `rows` (bool mask) from snap into the live params."""
        import torch
        with torch.no_grad():
            for p, s in zip(self.params, snap):
                p[rows] = s[rows]

    def reset(self):
        import torch
        g = torch.Generator().manual_seed(0)
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if i % 2 == 0:
                    p.copy_((torch.randn(p.shape, generator=g) / np.sqrt(p.shape[2])).to(p.device))
                else:
                    p.zero_()


def logits(model, b, lora, pids, amp):
    import torch
    x, es, eo, lg, _ = b
    lora.cur = pids
    with torch.autocast("cuda", dtype=amp, enabled=x.is_cuda):
        out = model(x, es, eo)[0]
    lora.cur = None
    lgc = lg.clamp(min=0)
    z = out.float().gather(1, lgc)
    return z.masked_fill(lg < 0, -1e9).log_softmax(1)


def per_position(model, lora, P, sel, pids_of, device, amp, bs=1024):
    """log-prob of the played move and top-1 hit for positions sel, adapter of pids_of(sel) (None = Maia-3)."""
    import torch
    lp_out, hit_out = [], []
    with torch.no_grad():
        for s in range(0, len(sel), bs):
            idx = sel[s:s + bs]
            b = P.batch(idx, device)
            pid = None if pids_of is None else torch.from_numpy(pids_of(idx)).to(device)
            lp = logits(model, b, lora, pid, amp)
            truth = b[4]
            lp_out.append(lp.gather(1, truth[:, None])[:, 0].cpu().numpy())
            hit_out.append((lp.argmax(1) == truth).cpu().numpy())
    return np.concatenate(lp_out), np.concatenate(hit_out)


def train_eval(model, P, n_players, N, device, epochs, lr, wd, bs, log, smoke):
    import torch
    lora = Lora(model, n_players, device)
    opt = torch.optim.AdamW(lora.params, lr=lr, weight_decay=wd)
    tr = np.flatnonzero((P.role == "train") & (P.tr_rank < N))
    va = np.flatnonzero(P.role == "val")
    rng = np.random.default_rng(N)
    own = lambda idx: P.pid[idx]
    lp0, _ = per_position(model, lora, P, va, None, device, torch.float16)
    best_ce = np.full(n_players, np.inf)
    vpid = P.pid[va]
    ce0 = np.bincount(vpid, -lp0, n_players) / np.maximum(np.bincount(vpid, minlength=n_players), 1)
    best_ce[:] = ce0
    best = lora.snapshot()
    best_ep = np.zeros(n_players, np.int64)
    for ep in range(1, epochs + 1):
        t0 = time.time()
        perm = rng.permutation(tr)
        tot = 0.0
        for s in range(0, len(perm), bs):
            idx = perm[s:s + bs]
            b = P.batch(idx, device)
            lp = logits(model, b, lora, torch.from_numpy(P.pid[idx]).to(device), torch.bfloat16)
            loss = -lp.gather(1, b[4][:, None]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        lpv, _ = per_position(model, lora, P, va, own, device, torch.float16)
        ce = np.bincount(vpid, -lpv, n_players) / np.maximum(np.bincount(vpid, minlength=n_players), 1)
        better = ce < best_ce
        snap = lora.snapshot()
        for b_, s_ in zip(best, snap):
            b_[torch.from_numpy(better).to(device)] = s_[torch.from_numpy(better).to(device)]
        best_ce = np.where(better, ce, best_ce)
        best_ep = np.where(better, ep, best_ep)
        log.append({"N": N, "epoch": ep, "train_ce": tot / max(len(perm), 1), "val_ce_gain": float((ce0 - ce).mean()),
                    "improved": int(better.sum()), "s": round(time.time() - t0)})
        print(json.dumps(log[-1]), flush=True)
    lora.load_rows(best, np.ones(n_players, bool))
    te = np.flatnonzero((P.role == "test") & (P.ply >= 10))
    lp_m, h_m = per_position(model, lora, P, te, None, device, torch.float16)
    lp_p, h_p = per_position(model, lora, P, te, own, device, torch.float16)
    # control: another player of the same band (fixed derangement within band)
    other = np.arange(n_players)
    band_of = P.band_of
    for band in np.unique(band_of):
        ids = np.flatnonzero(band_of == band)
        other[ids] = np.roll(ids, 1)
    lp_c, h_c = per_position(model, lora, P, te, lambda idx: other[P.pid[idx]], device, torch.float16)
    for h in lora.hooks:
        h.remove()
    return te, {"maia": (lp_m, h_m), "own": (lp_p, h_p), "other": (lp_c, h_c)}, best_ep


def summarize(P, te, out, best_ep, N):
    rng = np.random.default_rng(0)
    game = P.game[te]
    gi = np.unique(game, return_inverse=True)[1]
    ply = P.ply[te]

    def gain(sel, arm):
        lp_m, h_m = out["maia"]
        lp_a, h_a = out[arm]
        d = h_a[sel].astype(float) - h_m[sel]
        dce = lp_a[sel] - lp_m[sel]                      # nats gained per position
        g = np.unique(gi[sel], return_inverse=True)[1]
        gs, gn = np.bincount(g, d), np.bincount(g)
        b = rng.integers(0, len(gs), (1000, len(gs)))
        bs = 100 * gs[b].sum(1) / gn[b].sum(1)
        return {"positions": int(sel.sum()), "maia_top1": float(h_m[sel].mean()), "top1_gain_pts": 100 * float(d.mean()),
                "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                "nats_gain": float(dce.mean())}
    res = {"N": N, "best_epoch_hist": np.bincount(best_ep).tolist()}
    for arm in ("own", "other"):
        r = {"all": gain(np.ones(len(te), bool), arm), "by_ply": {}, "by_band": {}}
        for lo, hi in PLY_BINS:
            r["by_ply"][f"{lo}-{hi}"] = gain((ply >= lo) & (ply <= hi), arm)
        for band in BANDS:
            r["by_band"][str(band)] = gain(P.band_of[P.pid[te]] == band, arm)
        res[arm] = r
    return res


def run(device: str, workers: int, smoke: bool, epochs: int, lr: float, wd: float, bs: int):
    import torch

    from doppel_lab.engines import Maia3
    d = pl.read_parquet(OUT / "games.parquet")
    players = d.select("player", "band").unique().sort("band", "player")
    if smoke:
        players = players.group_by("band", maintain_order=True).head(3)
        d = d.filter(c("player").is_in(players["player"].implode()) & ((c("role") != "train") | (c("tr_rank") < 40)))
    pid = {p: i for i, p in enumerate(players["player"].to_list())}
    d = d.with_columns(pid=c("player").replace_strict(pid, return_dtype=pl.Int64)).sort("pid", "j")
    t0 = time.time()
    enc = encode_all(d, workers)
    P = Positions(d, enc)
    P.band_of = players["band"].to_numpy()
    print(f"{len(pid)} players, {d.height:,} games, {len(P.played):,} own positions, encode {time.time() - t0:.0f}s",
          flush=True)
    model = Maia3("79m", device).model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    res = {"players": len(pid), "by_band_players": {str(b): int((P.band_of == b).sum()) for b in BANDS},
           "blocks": list(BLOCKS), "rank": R, "lr": lr, "wd": wd, "epochs": epochs, "runs": {}, "log": []}
    for N in (NS if not smoke else (40,)):
        te, out, best_ep = train_eval(model, P, len(pid), N, device, epochs if N < 500 else max(2, epochs - 2), lr, wd,
                                      bs, res["log"], smoke)
        res["runs"][str(N)] = summarize(P, te, out, best_ep, N)
        r = res["runs"][str(N)]
        print(N, json.dumps({a: {"all": r[a]["all"], "by_ply": {k: round(v["top1_gain_pts"], 3)
                                                                   for k, v in r[a]["by_ply"].items()}}
                             for a in ("own", "other")}), flush=True)
        if not smoke:
            RESULTS.write_text(json.dumps(res, indent=1) + "\n")
    if smoke:
        print("smoke ok")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("select")
    r = sub.add_parser("run")
    r.add_argument("--gpu-ok", action="store_true")
    r.add_argument("--smoke", action="store_true")
    r.add_argument("--workers", type=int, default=12)
    r.add_argument("--epochs", type=int, default=6)
    r.add_argument("--lr", type=float, default=1e-3)
    r.add_argument("--wd", type=float, default=0.01)
    r.add_argument("--bs", type=int, default=512)
    a = ap.parse_args()
    if a.cmd == "select":
        select()
    else:
        if not a.gpu_ok:
            raise SystemExit("refusing to touch the GPU without --gpu-ok (run through the warden)")
        run("cuda", a.workers, a.smoke, a.epochs, a.lr, a.wd, a.bs)


if __name__ == "__main__":
    main()

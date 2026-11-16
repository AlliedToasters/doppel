"""Score v0f at repeated positions on held-out players: does z capture what a player's own history shows?

    # GPU, through the warden (memory "warden-job-form"):
    python -m doppel_lab.repeat_v0f --run v0f --gpu-ok        # -> lab/results/repeat_v0f.json
    python -m doppel_lab.repeat_v0f --run v0f --limit 200     # CPU smoke: the first 200 target sides

player_markov (SCIENCE.md §8) found that a player's own earlier moves at a repeated position beat Maia by 0.93
nats per visit, on the train games. v0f trained on those games, so it can't be scored there. This uses the
valrich split instead: 600 held-out players, their first 48 band games in time order (v0e_data), which v0f never
saw as targets or contexts.

Visits as player_markov: position key = Polyglot Zobrist of the decision position (poskey, plies 0-79), one visit
per (player, game, key), ordered by game time. Scored: every visit with at least one earlier visit by the same
player. Per scored visit, the played move's log-prob under
  maia      the trunk alone (Maia-3 79M at the game's real Elos; the v0 teacher)
  prior     v0f with z = the prior mean p(z | cond): v0f without the player
  post      v0f with z = the posterior mean from the player's 40 most recent earlier games (all if fewer)
  hist_M    the player's earlier moves there, shrunk toward model M:  (n_prev(move) + a p_M(move)) / (N_prev + a)
The question: how much of hist_maia's gain does post recover, and does shrinking toward post instead of Maia
help (if z already explains the repertoire, hist_post gains little over post).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

from doppel_lab.posbook_common import annotate, ply_index
from doppel_lab.v0_data import targets_sample
from doppel_lab.v0_model import legal_mask
from doppel_lab.v0_train import to_dev
from doppel_lab.v0e_data import CTX_FEATS, EMB_DIM, K_MAX, Split
from doppel_lab.v0e_train import build, loader
from doppel_lab.v0e_vec import games_path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "lab/results/repeat_v0f.json"
CAP = 80
ALPHAS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
MODELS = ("maia", "prior", "post")
A_CEIL = 0.5                                  # ceiling shrinkage (repertoire_memory: a = 0.5 fits every band)
c = pl.col


def visits(sp: Split, band: int = 1500) -> pl.DataFrame:
    """One row per (player, game, decision position) of valrich: player, t, g, ply, key, mv; plus N_prev, n_prev."""
    name = "v0e_valrich_cap80" if band == 1500 else f"v0e_valrich_{band}_cap80"
    keys, lens = annotate(name, [games_path("valrich", band)], cap=CAP)
    keys = np.asarray(keys)
    n_all = np.diff(sp.ix.off)
    assert len(lens) == len(n_all) and (lens == np.minimum(n_all, CAP)).all()
    gi, ply = ply_index(lens)
    off = np.r_[0, np.cumsum(lens)]
    dec = np.empty_like(keys)                 # decision key at ply k = key after ply k-1; ply 0 = start (0)
    dec[1:] = keys[:-1]
    dec[off[:-1][lens > 0]] = 0
    uci = sp.ix.uci[sp.ix.off[gi] + ply]
    _, mv = np.unique(uci, return_inverse=True)
    side = (ply % 2).astype(np.int64)
    df = pl.DataFrame({"player": sp.pid[gi, side].astype(np.int64), "t": sp.t[gi].astype(np.int64),
                       "g": gi.astype(np.int32), "ply": ply.astype(np.int32),
                       "key": dec.view(np.int64) if dec.dtype == np.uint64 else dec.astype(np.int64),
                       "mv": mv.astype(np.int32)})
    df = df.unique(["player", "g", "key"], keep="first", maintain_order=True)
    return (df.sort("player", "key", "t", "g")
            .with_columns(N_prev=pl.int_range(pl.len()).over("player", "key"),
                          n_prev=pl.int_range(pl.len()).over("player", "key", "mv")))


class RepeatSamples:
    """One sample per (game, colour) that has scored visits: those plies as targets, context = the player's
    K_MAX most recent earlier games."""

    def __init__(self, sp: Split, items: list[tuple[int, int, list[int]]]):
        self.sp, self.items = sp, items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        sp = self.sp
        g, col, plies = self.items[i]
        out = targets_sample(sp.ix, g, col, plies, None, len(plies), np.random.default_rng(0))
        hg, hc = sp.history(g, col)
        hg, hc = hg[-K_MAX:], hc[-K_MAX:]
        k = len(hg)
        out["ctx_emb"] = np.asarray(sp.vec.emb[hg, hc]) if k else np.zeros((0, EMB_DIM), np.float16)
        out["ctx_feat"] = sp.feat[hg, hc] if k else np.zeros((0, CTX_FEATS), np.float32)
        out["pid"] = np.array([sp.pid[g, col]], np.int64)
        out["nctx"] = np.array([k], np.int64)
        return out


@torch.no_grad()
def forward(model, ld, device: str) -> dict:
    model.eval()
    acc = {k: [] for k in ("game", "ply", "nctx", "kl_post", *MODELS)}
    for b in ld:
        b = to_dev(b, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            tok = b["tok"].float()
            t_logits, _, _ = model.trunk(tok, b["elo_self"], b["elo_opp"])
            mask = legal_mask(b["legal"])
            t_logp = t_logits.float().masked_fill(~mask, -1e9).log_softmax(-1)
            mu_q, _ = model.posterior(b)
            mu_p, _ = model.prior(b["cond"])
            emb = model.residual.board(tok)
            sid, idx = b["sid"], b["played"].clamp(min=0)[:, None]
            acc["maia"].append(t_logp.gather(1, idx)[:, 0].cpu())
            for tag, z in (("prior", mu_p[sid]), ("post", mu_q[sid])):
                lg = t_logp + model.residual(emb, z, b["elo_self"], b["elo_opp"], mask).float()
                lp = lg.masked_fill(~mask, -1e9).log_softmax(-1)
                acc[tag].append(lp.gather(1, idx)[:, 0].cpu())
                if tag == "post":                 # policy KL(post || Maia): how far the model moved here
                    acc["kl_post"].append(torch.where(mask, lp.exp() * (lp - t_logp), 0.0).sum(-1).cpu())
        acc["game"].append(b["game"].cpu())
        acc["ply"].append(b["ply"].cpu())
        acc["nctx"].append(b["nctx"][b["sid"]].cpu())
    return {k: torch.cat(v).float().numpy() if k in (*MODELS, "kl_post") else torch.cat(v).numpy()
            for k, v in acc.items()}


def summarize(x: pl.DataFrame) -> dict:
    n, N = x["n_prev"].to_numpy().astype(np.float64), x["N_prev"].to_numpy().astype(np.float64)
    ll = {m: x[m].to_numpy().astype(np.float64) for m in MODELS}
    cols = {m: ll[m] for m in MODELS}
    out = {"visits_scored": x.height, "players": x["player"].n_unique(),
           "ll_per_visit": {m: float(v.mean()) for m, v in ll.items()},
           "seen_before_share": float((n > 0).mean()), "alpha_grid": {}}
    for base in ("maia", "post"):
        pm = np.exp(ll[base])
        grid = {a: np.log((n + a * pm) / (N + a) + 1e-12) for a in ALPHAS}
        best = max(ALPHAS, key=lambda a: grid[a].mean())
        out["alpha_grid"][f"hist_{base}"] = {str(a): float(v.mean()) for a, v in grid.items()}
        out[f"alpha_best_hist_{base}"] = best
        cols[f"hist_{base}"] = grid[best]
        out["ll_per_visit"][f"hist_{base}"] = float(grid[best].mean())
    lm = out["ll_per_visit"]["maia"]
    out["gain_over_maia"] = {m: v - lm for m, v in out["ll_per_visit"].items()}
    hg = out["gain_over_maia"]["hist_maia"]
    out["post_share_of_hist_gain"] = out["gain_over_maia"]["post"] / hg if hg else None
    # per-player bootstrap CI of the key gains (players are the unit)
    pidx = np.unique(x["player"].to_numpy(), return_inverse=True)[1]
    rng = np.random.default_rng(0)
    cnt = np.bincount(pidx)
    out["gain_ci"] = {}
    for m in ("prior", "post", "hist_maia", "hist_post"):
        s = np.bincount(pidx, cols[m] - ll["maia"])
        boots = [s[p].sum() / cnt[p].sum() for p in (rng.integers(0, len(s), len(s)) for _ in range(1000))]
        out["gain_ci"][m] = [float(v) for v in np.percentile(boots, [2.5, 97.5])]
    d = x.with_columns(**{f"g_{m}": pl.Series(cols[m] - ll["maia"]) for m in cols if m != "maia"})
    gcols = [f"g_{m}" for m in cols if m != "maia"]

    def by(col, edges, labels):
        b = d.with_columns(bin=c(col).cut(edges, labels=labels)).group_by("bin").agg(
            visits=pl.len(), **{g: c(g).mean() for g in gcols}).sort("bin")
        return {str(r["bin"]): {k: r[k] for k in ("visits", *gcols)} for r in b.to_dicts()}

    out["by_N_prev"] = by("N_prev", [1, 2, 4, 8, 16], ["1", "2", "3-4", "5-8", "9-16", "17+"])
    out["by_ply"] = by("ply", [0, 9, 19, 39], ["0", "1-9", "10-19", "20-39", "40-79"])
    out["by_nctx"] = by("nctx", [4, 9, 19, 39], ["1-4", "5-9", "10-19", "20-39", "40"])
    return out


def score(model, band: int, device: str, workers: int = 12, batch: int = 64, limit: int = 0) -> pl.DataFrame:
    """Every valrich repeat visit with the model's maia / prior / post log-probs and its policy KL to Maia."""
    t0 = time.time()
    sp = Split("valrich", band)
    v = visits(sp, band)
    rep = v.filter(c("N_prev") >= 1)
    print(f"{v.height:,} visits, {rep.height:,} repeat visits by {rep['player'].n_unique():,} players "
          f"({time.time() - t0:.0f}s)", flush=True)
    side = rep.group_by("g", (c("ply") % 2).alias("col")).agg(c("ply").sort()).sort("g", "col")
    items = [(int(g), int(col), [int(p) for p in pl_]) for g, col, pl_ in side.iter_rows()]
    if limit:
        items = items[:limit]
    r = forward(model, loader(RepeatSamples(sp, items), batch, workers, shuffle=False), device)
    print(f"forward: {len(r['maia']):,} positions ({time.time() - t0:.0f}s)", flush=True)
    sc = pl.DataFrame({"g": r["game"].astype(np.int32), "ply": r["ply"].astype(np.int32),
                       "nctx": r["nctx"].astype(np.int64), "kl_post": r["kl_post"], **{m: r[m] for m in MODELS}})
    x = rep.join(sc, on=["g", "ply"], how="inner")
    assert x.height == sc.height, (x.height, sc.height)
    return x


def fidelity(x: pl.DataFrame) -> dict:
    """Repeat-visit fidelity (SCIENCE.md §8, repertoire memory): the model's gain over Maia at held-out repeat
    visits as a share of the history ceiling (count table shrunk toward Maia, a = 0.5, fixed so runs compare),
    next to the policy KL to Maia spent there. By ply and N_prev, since the visits are opening-heavy."""
    ceil = np.log((x["n_prev"].to_numpy() + A_CEIL * np.exp(x["maia"].to_numpy())) / (x["N_prev"].to_numpy() + A_CEIL))
    d = x.with_columns(ceil=pl.Series(ceil) - c("maia"), gain=c("post") - c("maia"))

    def agg(f):
        if f.height == 0:
            return {"visits": 0}
        return {"visits": f.height, "ceiling": round(float(f["ceil"].mean()), 4), "gain": round(float(f["gain"].mean()), 5),
                "share": round(float(f["gain"].sum() / f["ceil"].sum()), 4),
                "kl_post_maia": round(float(f["kl_post"].mean()), 4),
                "gain_per_kl": round(float(f["gain"].mean() / f["kl_post"].mean()), 3) if f["kl_post"].mean() > 0 else None}

    out = agg(d)
    out["by_ply"] = {k: agg(d.filter(m)) for k, m in (("0-9", c("ply") < 10), ("10-19", c("ply").is_between(10, 19)),
                                                      ("20+", c("ply") >= 20))}
    out["by_N_prev"] = {k: agg(d.filter(m)) for k, m in (("1-2", c("N_prev") <= 2), ("3-8", c("N_prev").is_between(3, 8)),
                                                         ("9+", c("N_prev") >= 9))}
    return out


def main(run: str, device: str, workers: int, batch: int, limit: int, band: int = 1500, dump: bool = False):
    t0 = time.time()
    model = build(False, device, band=band)
    s = torch.load(ROOT / "data/v0/runs" / run / "ckpt.pt", map_location=device, weights_only=False)
    model.load_state_dict(s["model"], strict=False)
    x = score(model, band, device, workers, batch, limit)
    if dump:                                  # per-visit log-probs for the CPU repertoire-memory prototype
        dp = ROOT / f"data/v0e/repeat_{run}{'_smoke' if limit else ''}.parquet"
        x.write_parquet(dp)
        print(f"dump -> {dp}", flush=True)
    out = {"run": run, "step": s["step"], "split": "valrich", "band": band, "limit": limit, **summarize(x),
           "fidelity": fidelity(x)}
    res = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    res[run if not limit else f"{run}_smoke"] = out
    RESULTS.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    print(f"-> {RESULTS} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v0f")
    ap.add_argument("--gpu-ok", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64, help="target sides per batch")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--band", type=int, default=1500)
    ap.add_argument("--dump", action="store_true", help="also write the per-visit table to data/v0e/repeat_<run>.parquet")
    ap.add_argument("--runs", default="", help="several runs in one job: run@band,run@band,...")
    a = ap.parse_args()
    for r in (a.runs.split(",") if a.runs else [f"{a.run}@{a.band}"]):
        run, band = r.split("@")
        main(run, "cuda" if a.gpu_ok else "cpu", a.workers, a.batch, a.limit, int(band), a.dump)

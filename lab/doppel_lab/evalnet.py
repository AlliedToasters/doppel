"""Eval approximator: how well does a net on Maia-3 features estimate our Stockfish labels, and would it be good
enough to stand in for Stockfish in the science (T2 at T1 cost)?

    # GPU (via the warden): frozen Maia-3 79M pooled features + its value head, both Elos fixed at 1500
    python -m doppel_lab.evalnet extract --split train --games 25000 --gpu-ok   # -> data/probe/evalnet_train.npz
    python -m doppel_lab.evalnet extract --split val --games 5000 --gpu-ok
    # CPU: fit heads, score them             -> lab/results/evalnet.json
    python -m doppel_lab.evalnet extract --split human --gpu-ok     # the six-band corpus's 12k Stockfish games
    python -m doppel_lab.evalnet fit        # also writes data/evals/human_evalnet.parquet when the human npz exists
    python -m doppel_lab.evalnet science    # game_science rating models with net evals in place of Stockfish

Labels: data/v0/evals_1500_<split>.parquet (Stockfish 50k nodes, White-POV cp for positions 0..n), mapped to
win chances with the feature registry's wc(). Every position is scored from the side to move (Maia's frame).
Predictors of the side-to-move win chance:
  maia_value   Maia-3's own W/D/L head (trained on game results, not engines): p_win - p_loss, then a monotone
               calibration fitted on train
  ridge        ridge regression on the 1024-d pooled trunk features (frozen)
  mlp          2-layer MLP on the same features (CPU torch)
Scored per position (MAE in win% points, R2, Pearson; by game phase) and downstream, as the science uses it:
  move loss    wc_loss_hat from consecutive positions vs the label's wc_loss: blunder (>= 0.3) AUC, precision
               and recall at the same threshold, and at the threshold matching the label's blunder rate
  game level   Spearman of per side-game mean wc_loss and blunder count, hat vs label
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import polars as pl

from doppel_lab.features.channels import wc
from doppel_lab.teacher import HIST, encode_game

ROOT = Path(__file__).resolve().parents[2]
V0 = ROOT / "data/v0"
OUT = ROOT / "data/probe"
RESULTS = ROOT / "lab/results"
HUMAN_EVALS = ROOT / "data/evals/human.parquet"
HUMAN_HAT = ROOT / "data/evals/human_evalnet.parquet"  # the MLP's evals for the same games, same format
ELO = 1500


def extract(split: str, n_games: int, device: str, batch: int = 1024, workers: int = 20):
    import torch

    from doppel_lab.engines import Maia3
    if split == "human":  # the six-band corpus's 12k Stockfish games (game_science's T2 subset), all of them
        ev = pl.read_parquet(HUMAN_EVALS)
        g = pl.read_parquet(ROOT / "data/human/games_2026-10.parquet", columns=["game_id", "moves"]).join(ev, on="game_id")
    else:
        ev = pl.read_parquet(V0 / f"evals_1500_{split}.parquet")
        g = pl.read_parquet(V0 / f"games_1500_{split}.parquet", columns=["game_id", "moves"]).join(ev, on="game_id")
    g = g.filter(pl.col("evals").list.len() == pl.col("moves").list.len() + 1)
    if split != "human":
        g = g.sample(n_games, seed=0, shuffle=True)
    eng = Maia3("79m", device)
    model, feats = eng.model, {}
    model.last_ln.register_forward_hook(lambda m, i, o: feats.__setitem__("x", o.detach()))
    rows = list(g.iter_rows())
    t0 = time.time()
    E, V, CP, GI, PLY = [], [], [], [], []
    with mp.get_context("fork").Pool(workers) as pool:
        for b0 in range(0, len(rows), 2000):
            blk = rows[b0:b0 + 2000]
            enc = pool.map(encode_game, [(m, ELO, ELO) for _, m, _ in blk], chunksize=32)
            n_per = np.array([len(e[3]) for e in enc])
            off = np.r_[0, np.cumsum(n_per)]
            planes = torch.from_numpy(np.concatenate([e[0] for e in enc])).to(device)
            hidx = torch.from_numpy(np.concatenate([
                np.clip(np.arange(n)[:, None] + np.arange(-HIST + 1, 1)[None, :], 0, None) + o
                for n, o in zip(n_per, off[:-1])])).to(device)
            P = len(hidx)
            elo = torch.full((batch,), ELO, dtype=torch.long, device=device)
            for s in range(0, P, batch):
                e_ = min(P, s + batch)
                x = planes[hidx[s:e_]].float().permute(0, 2, 1, 3).reshape(e_ - s, 64, HIST * 12)
                x = torch.cat([x, torch.zeros(e_ - s, 64, 1, device=device)], 2)
                with torch.no_grad(), torch.autocast("cuda", enabled=device.startswith("cuda")):
                    _, lv, _ = model(x, elo[:e_ - s], elo[:e_ - s])
                E.append(feats["x"].float().cpu().numpy().astype(np.float16))
                V.append(torch.softmax(lv.float(), 1).cpu().numpy().astype(np.float32))
            for (gid, _, evs), n in zip(blk, n_per):
                CP.append(np.array(evs[:n], np.float32))
                PLY.append(np.arange(n, dtype=np.int16))
            GI += [np.full(n, b0 + i, np.int32) for i, n in enumerate(n_per)]
            print(f"{b0 + len(blk):,}/{len(rows):,} games, {sum(len(c) for c in CP):,} positions, "
                  f"{sum(len(c) for c in CP) / (time.time() - t0):,.0f}/s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    # final position of each game (after the last move) is labelled but not encoded; move loss uses pairs inside
    np.savez(OUT / f"evalnet_{split}.npz", emb=np.concatenate(E), value=np.concatenate(V), cp=np.concatenate(CP),
             game=np.concatenate(GI), ply=np.concatenate(PLY), game_ids=np.array([r[0] for r in rows]))
    print(f"wrote evalnet_{split}.npz ({time.time() - t0:.0f}s)")


def load(split):
    d = dict(np.load(OUT / f"evalnet_{split}.npz"))
    stm = np.where(d["ply"] % 2 == 0, 1.0, -1.0)  # +1 White to move
    d["y"] = (stm * wc(d["cp"].astype(float))).astype(np.float32)  # side-to-move win chance, label
    v = d["value"]
    d["vraw"] = v[:, 2] - v[:, 0]  # Maia-3 value labels are [loss, draw, win] for the side to move (maia3/uci.py)
    return d


def mlp_fit(Xtr, ytr, Xvas, epochs=6, seed=0):
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(24)
    net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], 512), torch.nn.GELU(), torch.nn.Linear(512, 256),
                              torch.nn.GELU(), torch.nn.Linear(256, 1), torch.nn.Tanh())
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    n, bs = len(Xt), 4096
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * (n // bs + 1))
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for s in range(0, n, bs):
            i = perm[s:s + bs]
            loss = torch.nn.functional.smooth_l1_loss(net(Xt[i].float())[:, 0], yt[i], beta=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(i)
        print(f"  mlp epoch {ep + 1}: train loss {tot / n:.4f}", flush=True)
    with torch.no_grad():
        return [np.concatenate([net(torch.from_numpy(X[s:s + 65536]).float())[:, 0].numpy()
                                for s in range(0, len(X), 65536)]) for X in Xvas]


def pos_scores(y, p, ply):
    e = np.abs(y - p) * 50  # win% points (wc is in [-1, 1])
    r = {"mae_pct": float(e.mean()), "median_ae_pct": float(np.median(e)), "p90_ae_pct": float(np.quantile(e, 0.9)),
         "r2": float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()), "pearson": float(np.corrcoef(y, p)[0, 1])}
    r["by_phase"] = {name: float(e[(ply >= a) & (ply < b)].mean()) for name, a, b in
                     (("opening<20", 0, 20), ("middle20-60", 20, 60), ("late60+", 60, 10_000))}
    return r


def move_scores(d, p):
    """Move loss from consecutive positions of the same game (p and y are side-to-move win chances)."""
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    same = d["game"][1:] == d["game"][:-1]
    loss = lambda w: np.maximum(0.0, w[:-1] + w[1:])[same]  # mover's wc before minus after (= -next stm wc)
    L, Lh = loss(d["y"]), loss(p)
    bl = L >= 0.3
    thr_match = np.quantile(Lh, 1 - bl.mean())
    out = {"blunder_rate_label": float(bl.mean()), "blunder_auc": float(roc_auc_score(bl, Lh)),
           "loss_pearson": float(np.corrcoef(L, Lh)[0, 1]), "loss_mae_pct": float(np.abs(L - Lh).mean() * 50)}
    for name, t in (("at_0.3", 0.3), ("rate_matched", float(thr_match))):
        hb = Lh >= t
        out[name] = {"threshold": t, "precision": float((hb & bl).sum() / max(hb.sum(), 1)),
                     "recall": float((hb & bl).sum() / bl.sum()), "rate_hat": float(hb.mean())}
    # per side-game aggregates: mean loss, blunder count
    gm = d["game"][:-1][same]
    side = (d["ply"][:-1][same] % 2).astype(np.int64)
    key = gm.astype(np.int64) * 2 + side
    _, k = np.unique(key, return_inverse=True)
    cnt = np.bincount(k)
    agg = lambda v: np.bincount(k, v) / cnt
    out["game_mean_loss_spearman"] = float(spearmanr(agg(L), agg(Lh)).statistic)
    out["game_blunders_spearman"] = float(spearmanr(np.bincount(k, bl), np.bincount(k, Lh >= thr_match)).statistic)
    return out


def fit():
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import Ridge
    t0 = time.time()
    tr, va = load("train"), load("val")
    print(f"train {len(tr['y']):,} positions, val {len(va['y']):,}", flush=True)
    res = {"n_train": int(len(tr["y"])), "n_val": int(len(va["y"])), "label": "Stockfish 50k nodes, wc() side to move",
           "label_sd_pct": float(va["y"].std() * 50), "models": {}}
    preds = {}
    iso = IsotonicRegression(out_of_bounds="clip").fit(tr["vraw"], tr["y"])
    preds["maia_value_raw"] = va["vraw"]
    preds["maia_value"] = iso.predict(va["vraw"])
    mu, sd = tr["emb"].astype(np.float32).mean(0), tr["emb"].astype(np.float32).std(0) + 1e-6
    Xtr = ((tr["emb"].astype(np.float32) - mu) / sd).astype(np.float32)
    Xva = ((va["emb"].astype(np.float32) - mu) / sd).astype(np.float32)
    rg = Ridge(alpha=10.0).fit(Xtr, tr["y"])
    preds["ridge"] = np.clip(rg.predict(Xva), -1, 1)
    Xtr2, Xva2 = np.c_[Xtr, tr["vraw"]], np.c_[Xva, va["vraw"]]
    hu = load("human") if (OUT / "evalnet_human.npz").exists() else None
    Xs = [Xva2.astype(np.float32)]
    if hu is not None:
        Xs.append(np.c_[((hu["emb"].astype(np.float32) - mu) / sd), hu["vraw"]].astype(np.float32))
    mp_ = mlp_fit(Xtr2.astype(np.float32), tr["y"], Xs)
    preds["mlp"] = mp_[0]
    for name, p in preds.items():
        r = pos_scores(va["y"], p, va["ply"]) | {"moves": move_scores(va, p)}
        res["models"][name] = r
        print(f"{name:15s} MAE {r['mae_pct']:5.2f} win% | R2 {r['r2']:+.3f} | blunder AUC {r['moves']['blunder_auc']:.3f}"
              f" P/R@rate {r['moves']['rate_matched']['precision']:.2f}/{r['moves']['rate_matched']['recall']:.2f} | "
              f"game mean-loss rho {r['moves']['game_mean_loss_spearman']:.3f}", flush=True)
    if hu is not None:  # out-of-distribution check (six bands, not the 1500 band) + evals for the science test
        ph = mp_[1]
        res["human_corpus"] = pos_scores(hu["y"], ph, hu["ply"]) | {"moves": move_scores(hu, ph)}
        r = res["human_corpus"]
        print(f"{'mlp on human':15s} MAE {r['mae_pct']:5.2f} win% | R2 {r['r2']:+.3f} | blunder AUC "
              f"{r['moves']['blunder_auc']:.3f} | game mean-loss rho {r['moves']['game_mean_loss_spearman']:.3f}")
        write_hat(hu, ph, HUMAN_HAT)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "evalnet.json").write_text(json.dumps(res, indent=1) + "\n")
    print(f"-> evalnet.json ({time.time() - t0:.0f}s)")


def write_hat(d: dict, p: np.ndarray, path: Path):
    """Side-to-move win-chance predictions -> White-POV cp lists in data/evals format (game_id, evals)."""
    stm = np.where(d["ply"] % 2 == 0, 1.0, -1.0)
    w = np.clip(stm * p, -0.995, 0.995)
    cp = np.clip(np.log((1 + w) / (1 - w)) / 0.00368208, -3000, 3000)  # inverse of metrics.wc
    hat = (pl.DataFrame({"game_id": d["game_ids"][d["game"]], "cp": np.round(cp).astype(np.int64)})
           .group_by("game_id", maintain_order=True).agg(pl.col("cp"))
           # the final position is not encoded: repeat the last estimate (the last move scores zero loss)
           .with_columns(evals=pl.concat_list(pl.col("cp"), pl.col("cp").list.last()))
           .select("game_id", "evals"))
    hat.write_parquet(path)
    print(f"wrote {path} ({hat.height:,} games)")


def science(hat: str = "mlp", workers: int = 24):
    """Does the science survive the swap? game_science's per-ply table rebuilt from the net's evals, then the same
    game_mean rating model: T0, T2 (Stockfish), T2-hat (net), and each plus rating-free Maia T1."""
    from doppel_lab import game_science as gs
    from doppel_lab import sharpness as sh
    t0 = time.time()
    tag = "" if hat == "mlp" else f"_{hat}"
    hat_ply = ROOT / f"data/v0/features/human_evalnet{tag}_T2.parquet"
    if not hat_ply.exists():
        gs.build(workers, evals=ROOT / f"data/evals/human_evalnet{tag}.parquet", out=hat_ply)
    out = {}
    for name, ply in (("stockfish", gs.PLY), ("evalnet", hat_ply)):
        sides, games = gs.tables(ply)
        games = sh.add_t1(games)
        ev = games.filter(pl.col("has_eval"))
        tc = ["base", "inc"]
        t2f = tc + ["plies", "decisive", "time_forfeit", "lead_changes"] + gs.tier_feats("T2", "w_") + \
            gs.tier_feats("T2", "b_")
        t1 = [pre + f for pre in ("w_", "b_") for f in sh.T1_FEATS]
        for fs_name, fs in (("T2", t2f), ("T2+T1", t2f + t1)):
            X = ev.select([pl.col(f).cast(pl.Float64) for f in fs]).to_numpy()
            y = ev["elo_mean"].cast(pl.Float64).to_numpy()
            r = gs.reg_scores(y, gs.cv_reg(X, y, ev["game_id"].to_numpy()), ev["band"].to_numpy())
            out[f"{name}/{fs_name}"] = {k: r[k] for k in ("n", "mae", "r2", "pearson", "band_exact")}
            print(f"{name}/{fs_name:6} mae {r['mae']:6.1f} r2 {r['r2']:+.3f}", flush=True)
        if name == "stockfish":
            se_sf = sides.filter(pl.col("has_eval")).select("game_id", "mover", "acpl")
        else:
            j = sides.filter(pl.col("has_eval")).select("game_id", "mover", pl.col("acpl").alias("acpl_hat")).join(
                se_sf.select("game_id", "mover", "acpl"), on=["game_id", "mover"])
            from scipy.stats import spearmanr
            out["side_acpl_spearman_hat_vs_sf"] = float(spearmanr(j["acpl"], j["acpl_hat"], nan_policy="omit").statistic)
            print(f"side acpl spearman (net vs Stockfish): {out['side_acpl_spearman_hat_vs_sf']:.3f}")
    rf = RESULTS / ("evalnet.json" if hat == "mlp" else f"evalnet_{hat}.json")
    res = json.loads(rf.read_text())
    res["science"] = out
    rf.write_text(json.dumps(res, indent=1) + "\n")
    print(f"-> evalnet.json science ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--split", choices=["train", "val", "human"], default="train")
    e.add_argument("--games", type=int, default=25_000)
    e.add_argument("--gpu-ok", action="store_true")
    sub.add_parser("fit")
    sc = sub.add_parser("science")
    sc.add_argument("--hat", choices=["mlp", "ft"], default="mlp")
    a = ap.parse_args()
    if a.cmd == "extract":
        extract(a.split, a.games, "cuda" if a.gpu_ok else "cpu")
    elif a.cmd == "fit":
        fit()
    else:
        science(a.hat)

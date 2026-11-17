"""Residual model of same-player revisit pairs: why do people deviate from their own earlier move? (CPU only)

    python -m doppel_lab.revisit_model power  [--src heldout]          # 1. power simulation from the real pairs
    python -m doppel_lab.revisit_model fit    [--src heldout]          # 2. mixed-effects logistic, Maia offset
    python -m doppel_lab.revisit_model momentum [--src heldout] [--n 50000] [--workers 8]   # 3. Stockfish momentum
    python -m doppel_lab.revisit_model all    [--src heldout] ...      # 1-3, then the report
    python -m doppel_lab.revisit_model selftest                        # recover known effects from simulated data

Inputs (from doppel_lab.revisits): data/revisits/<tag>/pairs.parquet (all pairs) and pairs_maia.parquet (pairs with
Maia-3 chance agreement). One row = one consecutive pair of visits by the same player (as mover) to the same board
(128-bit key), in different games, ply >= 10.

Model (step 2). Outcome: repeat (same move at both visits). Offset: logit(chance_cross), the probability that two
independent draws from Maia-3's policy at the two visits (each at that visit's real Elos) agree, so the intercept and
every coefficient describe the *personal* part of repetition beyond what the position makes likely for anyone.
Random intercept per player (how deterministic that person is), fitted by maximum likelihood with Gauss-Hermite
quadrature (one 1-D integral per player; exact up to quadrature, no PQL bias). Covariates are binned (reference
levels named in the output): clock difference between visits, the player's clock at visit 2, games between visits,
rating change, ply, time control, transposition vs identical path, k tier, population-rarity tier, first visit's
game result. Main fit: ply 10-29, different opponents; second fit adds same-opponent pairs with a flag.

Step 3 (Stockfish, 20k nodes, as sim_fidelity): on a random sample of main-fit pairs, eval swing over the 6 plies
before each visit (player's expected score), a blunder among the player's last 3 moves (wc_loss >= 0.2, the
registry's blunder2), and the eval level at the visit; visit-2-minus-visit-1 differences enter the model on that
subset. For switches: expected score after the new move minus after the old move, from the visit-2 board.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/revisits"
RES = ROOT / "lab/results"
SCRATCH_MD = None  # set by --md
c = pl.col
QN = 20  # Gauss-Hermite nodes
Z = 1.959964
ZP = 0.841621  # 80% power


# ------------------------------------------------------------------ GLMM (random intercept, logit, offset)
def _gh():
    x, w = np.polynomial.hermite.hermgauss(QN)
    return np.sqrt(2.0) * x, w / np.sqrt(np.pi)   # nodes/weights for E[f(u)], u ~ N(0, 1)


def _log1pexp(x):
    return np.where(x > 0, x + np.log1p(np.exp(-np.clip(x, 0, None))), np.log1p(np.exp(np.clip(x, None, 0))))


def glmm_negll(theta, y, X, off, gidx, ng, nodes, lw):
    """Negative marginal log-likelihood and gradient. theta = (beta..., log sigma)."""
    beta, ls = theta[:-1], theta[-1]
    s = np.exp(ls)
    eta0 = off + X @ beta                                       # (n,)
    E = eta0[:, None] + s * nodes[None, :]                      # (n, Q)
    ll = y[:, None] * E - _log1pexp(E)                          # per-row log p(y | u_q)
    G = np.zeros((ng, len(nodes)))
    for q in range(len(nodes)):
        G[:, q] = np.bincount(gidx, ll[:, q], minlength=ng)
    A = G + lw[None, :]
    m = A.max(1, keepdims=True)
    W = np.exp(A - m)
    Lg = m[:, 0] + np.log(W.sum(1))
    post = W / W.sum(1, keepdims=True)                          # (ng, Q) posterior node weights
    pr = post[gidx]                                             # (n, Q)
    P = 1.0 / (1.0 + np.exp(-E))
    R = (y[:, None] - P) * pr                                   # (n, Q)
    gb = X.T @ R.sum(1)
    gs = (R * nodes[None, :]).sum() * s
    return -Lg.sum(), -np.r_[gb, gs]


def fit_glmm(y, X, off, groups, names):
    """ML fit; SEs from a finite-difference Hessian of the analytic gradient."""
    from scipy.optimize import minimize
    y = np.asarray(y, np.float64); X = np.asarray(X, np.float64); off = np.asarray(off, np.float64)
    _, gidx = np.unique(np.asarray(groups), return_inverse=True)
    ng = int(gidx.max()) + 1
    nodes, w = _gh(); lw = np.log(w)
    th0 = np.r_[np.zeros(X.shape[1]), np.log(0.5)]
    f = lambda t: glmm_negll(t, y, X, off, gidx, ng, nodes, lw)
    t0 = time.time()
    r = minimize(f, th0, jac=True, method="L-BFGS-B", options={"maxiter": 500, "gtol": 1e-6})
    th = r.x
    k = len(th); H = np.zeros((k, k)); h = 1e-4
    for i in range(k):
        e = np.zeros(k); e[i] = h
        H[:, i] = (f(th + e)[1] - f(th - e)[1]) / (2 * h)
    H = (H + H.T) / 2
    try:
        cov = np.linalg.inv(H)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        cov, se = None, np.full(k, np.nan)
    return {"names": names + ["log_sigma"], "theta": th, "se": se, "cov": cov, "negll": float(r.fun),
            "converged": bool(r.success), "n": int(len(y)), "n_players": ng, "seconds": round(time.time() - t0, 1),
            "sigma": float(np.exp(th[-1])),
            "sigma_ci": [float(np.exp(th[-1] - Z * se[-1])), float(np.exp(th[-1] + Z * se[-1]))]}


def marginal_p(eta, sigma):
    nodes, w = _gh()
    return (w[None, :] / (1 + np.exp(-(eta[:, None] + sigma * nodes[None, :])))).sum(1)


# ------------------------------------------------------------------ design
def logit(p):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def binned(x: pl.Expr, edges: list, labels: list) -> pl.Expr:
    """labels[i] for edges[i-1] <= x < edges[i]; len(labels) == len(edges) + 1."""
    e = pl.lit(labels[-1])
    for edge, lab in reversed(list(zip(edges, labels[:-1]))):
        e = pl.when(x < edge).then(pl.lit(lab)).otherwise(e)
    return e


# feature: (expr builder, reference level, ordered levels)
def features(d: pl.DataFrame) -> pl.DataFrame:
    rq = d.filter(c("n_pop") > 0)["rarity"].quantile
    r_edges = [rq(0.25), rq(0.5), rq(0.75)] if d.filter(c("n_pop") > 0).height else [0, 0, 0]
    tc = (c("base").cast(pl.Utf8) + "+" + c("inc").cast(pl.Utf8))
    top_tc = d.select(tc.alias("t"))["t"].value_counts().sort("count", descending=True)["t"].head(4).to_list()
    return d.with_columns(
        f_clkdiff=binned(c("clk2") - c("clk1"), [-60, -20, -5, 5, 20, 60],
                         ["<=-60s", "-60..-20s", "-20..-5s", "-5..+5s", "+5..+20s", "+20..+60s", ">=+60s"]),
        f_clk2=binned(c("clk2"), [15, 30, 60, 120], ["<15s", "15-30s", "30-60s", "60-120s", ">=120s"]),
        f_gap=binned(c("games_between"), [1, 3, 10, 30, 100], ["0", "1-2", "3-9", "10-29", "30-99", "100+"]),
        f_drating=binned(c("elo2") - c("elo1"), [-50, -15, 15, 50], ["<=-50", "-50..-15", "-15..+15", "+15..+50", ">=+50"]),
        f_ply=binned(c("ply2"), [14, 18, 22, 26], ["10-13", "14-17", "18-21", "22-25", "26-29"]),
        f_tc=pl.when(tc.is_in(top_tc)).then(tc).otherwise(pl.lit("other")),
        f_path=pl.when(c("same_path")).then(pl.lit("identical")).otherwise(pl.lit("transposition")),
        f_k=binned(c("k"), [3, 6], ["2", "3-5", "6+"]),
        f_rarity=pl.when(c("n_pop") == 0).then(pl.lit("unseen elsewhere"))
        .otherwise(binned(c("rarity"), r_edges, ["common (q1)", "q2", "q3", "rare (q4)"])),
        f_result1=pl.when(c("score1") == 1.0).then(pl.lit("win")).when(c("score1") == 0.0).then(pl.lit("loss"))
        .otherwise(pl.lit("draw")),
    )


REF = {"f_clkdiff": "-5..+5s", "f_clk2": ">=120s", "f_gap": "1-2", "f_drating": "-15..+15", "f_ply": "10-13",
       "f_tc": None, "f_path": "identical", "f_k": "2", "f_rarity": "common (q1)", "f_result1": "win"}


def design(d: pl.DataFrame, feats: list[str], extra: list[str] = ()):
    cols, names = [], []
    for f in feats:
        v = d[f].to_numpy()
        levels = [x for x in d[f].value_counts().sort("count", descending=True)[f].to_list()]
        ref = REF.get(f) or levels[0]
        if ref not in levels:
            ref = levels[0]
        for lv in sorted(set(levels) - {ref}):
            if (v == lv).sum() < 30:
                continue
            cols.append((v == lv).astype(np.float64)); names.append(f"{f}={lv}")
    for e in extra:
        cols.append(d[e].cast(pl.Float64).to_numpy()); names.append(e)
    X = np.column_stack([np.ones(d.height)] + cols)
    return X, ["intercept"] + names


def effects(fit, X, off, names, d_levels=None):
    """Odds ratios with CIs and average marginal effects (points) vs the reference, integrating over players."""
    th, se, s = fit["theta"], fit["se"], fit["sigma"]
    beta = th[:-1]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(off), min(len(off), 40000), replace=False)
    Xs, os_ = X[idx], off[idx]
    out = {}
    for j, nm in enumerate(names):
        row = {"coef": round(float(beta[j]), 4), "se": round(float(se[j]), 4),
               "ci": [round(float(beta[j] - Z * se[j]), 4), round(float(beta[j] + Z * se[j]), 4)]}
        if nm != "intercept" and "=" in nm:
            f = nm.split("=")[0]
            grp = [k for k, n2 in enumerate(names) if n2.startswith(f + "=")]
            X0 = Xs.copy(); X0[:, grp] = 0.0
            X1 = X0.copy(); X1[:, j] = 1.0
            p0 = marginal_p(os_ + X0 @ beta, s); p1 = marginal_p(os_ + X1 @ beta, s)
            row["ame_pts"] = round(float(100 * (p1 - p0).mean()), 2)
            row["n_rows"] = int(X[:, j].sum())
        out[nm] = row
    return out


# ------------------------------------------------------------------ data
def load(src: str, maia: bool) -> pl.DataFrame:
    f = OUT / src / ("pairs_maia.parquet" if maia else "pairs.parquet")
    d = pl.read_parquet(f)
    if maia:
        d = d.filter(c("chance_cross").is_not_null())
    return d


def ply_bucket(x: pl.Expr) -> pl.Expr:
    return binned(x, [20, 30, 40], ["10-19", "20-29", "30-39", "40+"])


def deep_mask(d: pl.DataFrame) -> pl.Expr:
    q90 = d["rarity"].quantile(0.9)
    return (c("k") == 2) & (c("clen") >= 20) & (c("rarity") >= q90) & (c("ply2") >= 30) & (~c("same_opp"))


# ------------------------------------------------------------------ 1. power
def power(src: str, reps: int = 200) -> dict:
    """Detectable change in repeat rate per ply bucket x opponent type. Null simulation with the observed design:
    each player keeps their pair count; player intercepts u ~ N(0, sigma^2) with sigma fitted in that cell (or the
    pooled sigma for small cells); a binary covariate either varies within players (per pair, like time pressure)
    or between players (per player, like a trait). sd of the difference in repeat rate between covariate groups
    under the null gives the minimum detectable effect at the available n and the n needed for 1/2/5 points."""
    t0 = time.time()
    d = load(src, maia=False).with_columns(pb=ply_bucket(c("ply2")),
                                           opp=pl.when(c("same_opp")).then(pl.lit("same")).otherwise(pl.lit("different")))
    d = d.with_columns(deep=deep_mask(d))
    rng = np.random.default_rng(0)

    def sigma_of(sub):
        ss = sub.sample(min(sub.height, 300_000), seed=0) if sub.height > 300_000 else sub
        y = ss["repeat"].cast(pl.Float64).to_numpy()
        fit = fit_glmm(y, np.ones((len(y), 1)), np.zeros(len(y)), ss["pid"].to_numpy(), ["intercept"])
        return fit["sigma"], fit["sigma_ci"]

    pooled_sigma, pooled_ci = sigma_of(d.filter(~c("same_opp")))
    cells = []
    groups = [(pb, op) for pb in ("10-19", "20-29", "30-39", "40+") for op in ("different", "same")] + [("deep", "different")]
    for pb, op in groups:
        sub = d.filter(c("deep")) if pb == "deep" else d.filter((c("pb") == pb) & (c("opp") == op))
        n = sub.height
        row = {"ply": pb, "opponent": op, "pairs": n, "players": int(sub["pid"].n_unique()) if n else 0}
        if n < 20:
            row["note"] = "too few pairs to simulate"
            cells.append(row); continue
        rate = float(sub["repeat"].mean())
        sig, sig_ci = (sigma_of(sub) if n >= 2000 else (pooled_sigma, pooled_ci))
        sizes = sub.group_by("pid").len()["len"].to_numpy()
        # intercept giving the observed marginal rate
        lo, hi = -8.0, 8.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if marginal_p(np.array([mid]), sig)[0] < rate:
                lo = mid
            else:
                hi = mid
        b0 = (lo + hi) / 2
        gid = np.repeat(np.arange(len(sizes)), sizes)
        diffs = {"within": [], "between": []}
        for _ in range(reps):
            u = rng.normal(0, sig, len(sizes))
            p = 1 / (1 + np.exp(-(b0 + u[gid])))
            y = rng.random(len(gid)) < p
            cw = rng.random(len(gid)) < 0.5
            cb = (rng.random(len(sizes)) < 0.5)[gid]
            for nm, cv in (("within", cw), ("between", cb)):
                if cv.all() or (~cv).all():
                    continue
                diffs[nm].append(y[cv].mean() - y[~cv].mean())
        row |= {"repeat_rate": round(rate, 4), "sigma": round(sig, 3), "sigma_ci": [round(x, 3) for x in sig_ci],
                "sigma_source": "cell" if n >= 2000 else "pooled",
                "median_pairs_per_player": float(np.median(sizes)), "max_pairs_per_player": int(sizes.max())}
        for nm, v in diffs.items():
            sd = float(np.std(v))
            mde = (Z + ZP) * sd
            row[f"{nm}_mde_pts"] = round(100 * mde, 2)
            row[f"{nm}_pairs_needed"] = {f"{k}pt": int(np.ceil(n * (mde / (k / 100)) ** 2)) for k in (1, 2, 5)}
        cells.append(row)
        print(f"  power {pb}/{op}: n={n:,}, rate {rate:.3f}, sigma {sig:.2f}, MDE within {row['within_mde_pts']} / "
              f"between {row['between_mde_pts']} pts ({time.time() - t0:.0f}s)", flush=True)
    return {"cells": cells, "pooled_sigma_different_opp": round(pooled_sigma, 3),
            "reps": reps, "alpha": 0.05, "power": 0.8,
            "method": "null simulation with observed cluster sizes and fitted per-player sigma; MDE = 2.80 x sd of "
                      "the difference in repeat rate between covariate halves; pairs needed scale as 1/n"}


# ------------------------------------------------------------------ 2. fit
FEATS = ["f_clkdiff", "f_clk2", "f_gap", "f_drating", "f_ply", "f_tc", "f_path", "f_k", "f_rarity", "f_result1"]


def model_frame(src: str):
    d = load(src, maia=True)
    d = d.filter((c("ply2") >= 10) & (c("ply2") < 30))
    nforced = d.filter(c("chance_cross") > 0.99).height
    d = d.filter(c("chance_cross") <= 0.99)
    return features(d), nforced


def run_fit(d: pl.DataFrame, feats, extra=(), label=""):
    X, names = design(d, feats, extra)
    off = logit(d["chance_cross"].to_numpy())
    y = d["repeat"].cast(pl.Float64).to_numpy()
    fit = fit_glmm(y, X, off, d["pid"].to_numpy(), names)
    eff = effects(fit, X, off, names)
    p_chance = float(d["chance_cross"].mean()); p_obs = float(y.mean())
    p_model0 = float(marginal_p(off + fit["theta"][0], fit["sigma"]).mean())
    print(f"  fit {label}: n={fit['n']:,} players {fit['n_players']:,} sigma {fit['sigma']:.3f} "
          f"converged {fit['converged']} ({fit['seconds']}s)", flush=True)
    return {"label": label, "n_pairs": fit["n"], "n_players": fit["n_players"], "converged": fit["converged"],
            "seconds": fit["seconds"], "observed_repeat": round(p_obs, 4), "chance_repeat": round(p_chance, 4),
            "personal_repetition_pts": round(100 * (p_obs - p_chance), 2),
            "intercept_only_marginal_at_ref": round(p_model0, 4),
            "player_sigma_logit": round(fit["sigma"], 3), "player_sigma_ci": [round(x, 3) for x in fit["sigma_ci"]],
            "player_spread_note": "sd of a player's personal log-odds of repeating, beyond the Maia offset and covariates",
            "effects": eff, "negll": fit["negll"]}


def fit_all(src: str) -> dict:
    d, nforced = model_frame(src)
    main = d.filter(~c("same_opp"))
    out = {"excluded_near_forced": nforced,
           "offset": "logit(chance_cross): P(two draws from Maia-3's policies at visit 1 and visit 2 agree)",
           "main": run_fit(main, FEATS, label="ply 10-29, different opponents")}
    both = d.with_columns(same_opp_f=c("same_opp").cast(pl.Float64))
    out["with_same_opponent"] = run_fit(both, FEATS, extra=["same_opp_f"], label="ply 10-29, all opponents (same-opponent flag)")
    return out


# ------------------------------------------------------------------ 3. momentum
_eng = None
_nodes = 20000
MATE = 10000


def _init(nodes):
    global _eng, _nodes
    import chess.engine
    os.nice(15) if hasattr(os, "nice") else None
    _eng = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")
    _eng.configure({"Threads": 1, "Hash": 16})
    _nodes = nodes


def _eval(fen):
    import chess
    import chess.engine
    b = chess.Board(fen)
    if b.is_checkmate():
        return -MATE if b.turn == chess.WHITE else MATE
    if b.is_stalemate() or b.is_insufficient_material():
        return 0
    info = _eng.analyse(b, chess.engine.Limit(nodes=_nodes))
    return info["score"].white().score(mate_score=MATE)


def wc(cp):
    return 2.0 / (1.0 + np.exp(-0.00368208 * np.asarray(cp, np.float64))) - 1.0


WIN = 6


def momentum(src: str, n: int, workers: int, seed: int = 0) -> dict:
    import chess
    from multiprocessing import Pool
    from doppel_lab.revisits import src_files
    t0 = time.time()
    d, _ = model_frame(src)
    d = d.filter(~c("same_opp"))
    d = d.sample(min(n, d.height), seed=seed)
    gm = pl.read_parquet(OUT / src / "games.parquet", columns=["gi", "game_id"])
    need = pl.concat([d["g1"], d["g2"]]).unique()
    ids = gm.filter(c("gi").is_in(need.implode()))
    mv = (pl.scan_parquet(src_files(src.split("_smoke")[0])).select("game_id", "moves")
          .filter(c("game_id").is_in(ids["game_id"].implode())).unique("game_id").collect())
    moves = dict(zip(ids.join(mv, on="game_id")["gi"].to_list(), ids.join(mv, on="game_id")["moves"].to_list()))
    # positions to evaluate: per visit, indices ply-WIN .. ply; for switches, the visit-2 board after the old move
    want: dict[int, set] = {}
    for g1, p1, g2, p2 in d.select("g1", "ply1", "g2", "ply2").iter_rows():
        want.setdefault(g1, set()).update(range(max(0, p1 - WIN), p1 + 1))
        want.setdefault(g2, set()).update(range(max(0, p2 - WIN), p2 + 2))
    fen_of: dict[tuple, str] = {}
    for gi, idx in want.items():
        mvs = moves.get(gi)
        if mvs is None:
            continue
        b = chess.Board(); top = max(idx)
        if 0 in idx:
            fen_of[(gi, 0)] = b.fen()
        for k, u in enumerate(mvs[:top]):
            b.push_uci(u)
            if k + 1 in idx:
                fen_of[(gi, k + 1)] = b.fen()
    alt = {}
    for g1, p1, g2, p2, rep in d.select("g1", "ply1", "g2", "ply2", "repeat").iter_rows():
        if rep or g1 not in moves or g2 not in moves or (g2, p2) not in fen_of:
            continue
        b = chess.Board(fen_of[(g2, p2)])
        try:
            b.push_uci(moves[g1][p1])
            alt[(g1, p1, g2, p2)] = b.fen()
        except (ValueError, IndexError):
            pass
    fens = sorted(set(fen_of.values()) | set(alt.values()))
    print(f"  momentum: {d.height:,} pairs, {len(fens):,} positions to evaluate with {workers} workers", flush=True)
    ev = {}
    with Pool(workers, initializer=_init, initargs=(20000,)) as pool:
        for i, (f, v) in enumerate(zip(fens, pool.imap(_eval, fens, chunksize=64))):
            ev[f] = v
            if (i + 1) % 50000 == 0:
                print(f"    {i + 1:,}/{len(fens):,} evals, {time.time() - t0:.0f}s", flush=True)

    def visit_feats(gi, ply):
        """Player (mover at ply) POV: eval swing over WIN plies, blunder among own last 3 moves, level."""
        s = 1.0 if ply % 2 == 0 else -1.0
        w = {k: wc(ev[fen_of[(gi, k)]]) for k in range(max(0, ply - WIN), ply + 1) if (gi, k) in fen_of}
        if ply not in w or (ply - WIN) not in w:
            return None
        es = lambda k: (1 + s * w[k]) / 2
        bl = any(s * (w[m] - w[m + 1]) >= 0.2 for m in (ply - 2, ply - 4, ply - 6) if m in w and m + 1 in w)
        return es(ply) - es(ply - WIN), bl, es(ply)

    rows = []
    for r in d.iter_rows(named=True):
        a = visit_feats(r["g1"], r["ply1"]); b_ = visit_feats(r["g2"], r["ply2"])
        if a is None or b_ is None:
            continue
        dq = None
        if not r["repeat"]:
            k = (r["g1"], r["ply1"], r["g2"], r["ply2"])
            after_new = fen_of.get((r["g2"], r["ply2"] + 1))
            if k in alt and after_new is not None:
                s = 1.0 if r["ply2"] % 2 == 0 else -1.0
                dq = float((1 + s * wc(ev[after_new])) / 2 - (1 + s * wc(ev[alt[k]])) / 2)
        rows.append({**r, "swing1": a[0], "swing2": b_[0], "blund1": a[1], "blund2": b_[1], "lvl1": a[2], "lvl2": b_[2],
                     "switch_dq": dq})
    m = pl.DataFrame(rows, infer_schema_length=None)
    m = m.with_columns(d_swing=c("swing2") - c("swing1"), d_blund=c("blund2").cast(pl.Int8) - c("blund1").cast(pl.Int8),
                       d_lvl=c("lvl2") - c("lvl1"))
    m = m.with_columns(
        f_dswing=binned(c("d_swing"), [-0.15, -0.05, 0.05, 0.15], ["<=-0.15", "-0.15..-0.05", "-0.05..+0.05", "+0.05..+0.15", ">=+0.15"]),
        f_dblund=pl.when(c("d_blund") > 0).then(pl.lit("recent blunder only at visit 2"))
        .when(c("d_blund") < 0).then(pl.lit("recent blunder only at visit 1")).otherwise(pl.lit("same")),
        f_lvl2=binned(c("lvl2"), [0.2, 0.4, 0.6, 0.8], ["losing (<0.2)", "worse", "equal", "better", "winning (>=0.8)"]),
    )
    REF.update({"f_dswing": "-0.05..+0.05", "f_dblund": "same", "f_lvl2": "equal"})
    fit = run_fit(m, FEATS + ["f_dswing", "f_dblund", "f_lvl2"], label="momentum subset (Stockfish 20k nodes)")
    sw = m.filter(c("switch_dq").is_not_null())
    tol = 0.02

    def sw_summary(x):
        if x.height == 0:
            return {"n": 0}
        q = x["switch_dq"].to_numpy()
        return {"n": int(x.height), "better": round(float((q > tol).mean()), 4), "worse": round(float((q < -tol).mean()), 4),
                "same": round(float((np.abs(q) <= tol).mean()), 4), "mean_dq": round(float(q.mean()), 4)}

    switches = {"all": sw_summary(sw),
                "recent_blunder_at_visit2": sw_summary(sw.filter(c("blund2"))),
                "no_recent_blunder_at_visit2": sw_summary(sw.filter(~c("blund2"))),
                "clk2_lt_30s": sw_summary(sw.filter(c("clk2") < 30)),
                "after_loss_at_visit1": sw_summary(sw.filter(c("score1") == 0.0)),
                "after_win_at_visit1": sw_summary(sw.filter(c("score1") == 1.0)),
                "note": f"expected score after the new move minus after the old move, from the visit-2 board; "
                        f"better/worse = beyond +-{tol}"}
    m.select(pl.exclude("player", "opp1", "opp2")).write_parquet(OUT / src / "pairs_momentum.parquet")
    return {"pairs": int(m.height), "positions_evaluated": len(fens), "seconds": round(time.time() - t0),
            "fit": fit, "switch_quality": switches}


# ------------------------------------------------------------------ 4. learning from the move itself
def learn(src: str, workers: int) -> dict:
    """Was the visit-1 move a Stockfish blunder, and does that change whether it is repeated at visit 2?
    Uses the momentum sample (pairs_momentum.parquet; lvl1 = mover's expected score before the visit-1 move) and
    evaluates the board after the visit-1 move. Compares observed-minus-chance repetition for blunders vs non-blunders,
    and within bins of Maia's probability of the visit-1 move (blunders are unlikely moves, and unlikely moves are
    repeated less anyway: that bin control separates learning from regression to a player's usual move)."""
    import chess
    from multiprocessing import Pool
    from doppel_lab.revisits import src_files
    t0 = time.time()
    m = pl.read_parquet(OUT / src / "pairs_momentum.parquet")
    gm = pl.read_parquet(OUT / src / "games.parquet", columns=["gi", "game_id"])
    ids = gm.filter(c("gi").is_in(m["g1"].unique().implode()))
    mv = (pl.scan_parquet(src_files(src.split("_smoke")[0])).select("game_id", "moves")
          .filter(c("game_id").is_in(ids["game_id"].implode())).unique("game_id").collect())
    j = ids.join(mv, on="game_id")
    moves = dict(zip(j["gi"].to_list(), j["moves"].to_list()))
    fen_after = {}
    for g1, p1 in m.select("g1", "ply1").unique().iter_rows():
        mvs = moves.get(g1)
        if mvs is None or len(mvs) <= p1:
            continue
        b = chess.Board()
        for u in mvs[:p1 + 1]:
            b.push_uci(u)
        fen_after[(g1, p1)] = b.fen()
    fens = sorted(set(fen_after.values()))
    print(f"  learn: {m.height:,} pairs, {len(fens):,} positions to evaluate with {workers} workers", flush=True)
    ev = {}
    with Pool(workers, initializer=_init, initargs=(20000,)) as pool:
        for f, v in zip(fens, pool.imap(_eval, fens, chunksize=64)):
            ev[f] = v
    after = [ev.get(fen_after.get((g, p))) if (g, p) in fen_after else None for g, p in m.select("g1", "ply1").iter_rows()]
    s = np.where(m["ply1"].to_numpy() % 2 == 0, 1.0, -1.0)
    es_after = np.array([np.nan if a is None else 0.0 for a in after])
    ok = ~np.isnan(es_after)
    es_after[ok] = (1 + s[ok] * wc(np.array([a for a in after if a is not None]))) / 2
    loss1 = m["lvl1"].to_numpy() - es_after
    d = m.with_columns(loss1=pl.Series(loss1)).filter(c("loss1").is_not_nan())
    d = d.with_columns(blunder1=c("loss1") >= 0.2,
                       mistake1=(c("loss1") >= 0.1) & (c("loss1") < 0.2),
                       pbin=binned(c("p1_mv1"), [0.05, 0.15, 0.35, 0.6], ["<5%", "5-15%", "15-35%", "35-60%", ">=60%"]))
    y = d["repeat"].cast(pl.Float64).to_numpy(); ch = d["chance_cross"].to_numpy(); pid = d["pid"].to_numpy()

    def grp(mask):
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            return {"n": 0}
        f = lambda ii: float((y[ii] - ch[ii]).mean())
        return {"n": int(len(idx)), "players": int(len(np.unique(pid[idx]))), "observed": round(float(y[idx].mean()), 4),
                "chance": round(float(ch[idx].mean()), 4), "personal": round(f(idx), 4),
                "personal_ci": _boot_idx(pid, idx, f)}

    bl = d["blunder1"].to_numpy(); mi = d["mistake1"].to_numpy(); good = d["loss1"].to_numpy() < 0.05
    by_bin = {}
    for b_ in ["<5%", "5-15%", "15-35%", "35-60%", ">=60%"]:
        inb = (d["pbin"] == b_).to_numpy()
        gb, gg = grp(inb & bl), grp(inb & good)
        diff = None
        if gb["n"] >= 30 and gg["n"] >= 30:
            def fd(ii, inb=inb):
                bb = ii[bl[ii] & inb[ii]]; gg_ = ii[good[ii] & inb[ii]]
                return float((y[bb] - ch[bb]).mean() - (y[gg_] - ch[gg_]).mean()) if len(bb) and len(gg_) else np.nan
            allidx = np.flatnonzero(inb & (bl | good))
            diff = {"blunder_minus_good": round(fd(allidx), 4), "ci": _boot_idx(pid, allidx, fd)}
        by_bin[b_] = {"blunder": gb, "good": gg, "diff": diff}
    # pooled, bin-adjusted: GLMM with the blunder flag on top of the Maia offset and a p1_mv1 bin factor
    d2 = d.with_columns(blunder1_f=c("blunder1").cast(pl.Float64), mistake1_f=c("mistake1").cast(pl.Float64))
    REF.update({"pbin": "35-60%"})
    fit = run_fit(d2, FEATS + ["pbin"], extra=["blunder1_f", "mistake1_f"], label="learning: visit-1 move quality")
    # the post-blunder switch subgroup, with intervals (switch quality from the momentum run)
    sw = d.filter(c("switch_dq").is_not_null())
    q = sw["switch_dq"].to_numpy(); b2 = sw["blund2"].to_numpy(); spid = sw["pid"].to_numpy(); tol = 0.02

    def wmb(ii):
        return float((q[ii] < -tol).mean() - (q[ii] > tol).mean()) if len(ii) else np.nan
    sub = {}
    for name, mask in (("recent_blunder_at_visit2", b2), ("no_recent_blunder_at_visit2", ~b2)):
        idx = np.flatnonzero(mask)
        sub[name] = {"n": int(len(idx)), "worse_minus_better": round(wmb(idx), 4), "ci": _boot_idx(spid, idx, wmb)}
    alli = np.arange(len(q))

    def dd(ii):
        a = ii[b2[ii]]; b = ii[~b2[ii]]
        return wmb(a) - wmb(b) if len(a) and len(b) else np.nan
    sub["difference"] = {"value": round(dd(alli), 4), "ci": _boot_idx(spid, alli, dd)}
    return {"pairs": int(d.height), "seconds": round(time.time() - t0),
            "blunder_rate_visit1": round(float(bl.mean()), 4),
            "overall": {"blunder": grp(bl), "mistake": grp(mi), "good (<0.05)": grp(good)},
            "by_maia_prob_bin": by_bin, "fit": fit, "post_blunder_switches": sub,
            "note": "loss1 = mover's expected score before minus after the visit-1 move (Stockfish 20k nodes); "
                    "blunder >= 0.2, mistake 0.1-0.2; personal = observed repeat - Maia chance; "
                    "CIs: player-cluster bootstrap, 1000 reps."}


# ------------------------------------------------------------------ 5. triples: habitual vs one-off blunders
def triples(src: str, workers: int) -> dict:
    """Chains of three consecutive visits A -> B -> C to the same board by one player. Grade the move at B with
    Stockfish. 'Habitual' = the B move was also played at A; 'one-off' = it was not. If blunders vanish because they
    are slips, a habitual blunder should be kept at C about as often as a habitual good move; if players learn from
    them, a habitual blunder should be dropped more often, and more so when game B was lost."""
    import chess
    from multiprocessing import Pool
    from doppel_lab.revisits import src_files
    t0 = time.time()
    cols = ["pid", "key", "key2", "k", "g1", "ply1", "g2", "ply2", "repeat", "same_opp", "score1", "games_between", "clk2"]
    d = pl.read_parquet(OUT / src / "pairs.parquet", columns=cols)
    ab = d.select("pid", "key", "key2", "g1", "ply1", "g2", "ply2", rep_ab=c("repeat"), same_opp_ab=c("same_opp"))
    bc = d.rename({"g1": "gB", "ply1": "plyB", "g2": "gC", "ply2": "plyC", "repeat": "rep_bc", "score1": "scoreB",
                   "games_between": "gap_bc", "same_opp": "same_opp_bc", "clk2": "clkC"})
    ch = ab.join(bc, left_on=["pid", "key", "key2", "g2", "ply2"], right_on=["pid", "key", "key2", "gB", "plyB"])
    ch = ch.rename({"g2": "gB", "ply2": "plyB"})
    gm = pl.read_parquet(OUT / src / "games.parquet", columns=["gi", "game_id"])
    ids = gm.filter(c("gi").is_in(ch["gB"].unique().implode()))
    mv = (pl.scan_parquet(src_files(src.split("_smoke")[0])).select("game_id", "moves")
          .filter(c("game_id").is_in(ids["game_id"].implode())).unique("game_id").collect())
    j = ids.join(mv, on="game_id").select("gi", "moves")
    ch = ch.join(j, left_on="gB", right_on="gi").with_columns(mvB=c("moves").list.get(c("plyB")))
    # one FEN per board (any game B that reaches it), then the board after each distinct B move
    rep = ch.unique(["key", "key2"]).select("key", "key2", "moves", "plyB")
    fen_key = {}
    for k1, k2, mvs, ply in rep.iter_rows():
        b = chess.Board()
        for u in mvs[:ply]:
            b.push_uci(u)
        fen_key[(k1, k2)] = b.fen()
    ch = ch.drop("moves")
    fen_after = {}
    for k1, k2, u in ch.select("key", "key2", "mvB").unique().iter_rows():
        b = chess.Board(fen_key[(k1, k2)])
        try:
            b.push_uci(u)
            fen_after[(k1, k2, u)] = b.fen()
        except ValueError:
            pass
    fens = sorted(set(fen_key.values()) | set(fen_after.values()))
    print(f"  triples: {ch.height:,} chains, {len(fen_key):,} boards, {len(fen_after):,} board+move, "
          f"{len(fens):,} evals with {workers} workers ({time.time() - t0:.0f}s)", flush=True)
    ev = {}
    with Pool(workers, initializer=_init, initargs=(20000,)) as pool:
        for i, (f, v) in enumerate(zip(fens, pool.imap(_eval, fens, chunksize=64))):
            ev[f] = v
            if (i + 1) % 50000 == 0:
                print(f"    {i + 1:,}/{len(fens):,} evals, {time.time() - t0:.0f}s", flush=True)
    loss = []
    for k1, k2, u in ch.select("key", "key2", "mvB").iter_rows():
        fb = fen_key[(k1, k2)]; fa = fen_after.get((k1, k2, u))
        if fa is None:
            loss.append(None); continue
        sgn = 1.0 if " w " in fb else -1.0
        loss.append(float((1 + sgn * wc(ev[fb])) / 2 - (1 + sgn * wc(ev[fa])) / 2))
    ch = ch.with_columns(lossB=pl.Series(loss, dtype=pl.Float64)).filter(c("lossB").is_not_null())
    ch = ch.with_columns(
        quality=pl.when(c("lossB") >= 0.2).then(pl.lit("blunder")).when(c("lossB") >= 0.1).then(pl.lit("mistake"))
        .when(c("lossB") < 0.05).then(pl.lit("good")).otherwise(pl.lit("inaccuracy")),
        habit=pl.when(c("rep_ab")).then(pl.lit("habitual")).otherwise(pl.lit("one-off")),
        f_k=binned(c("k"), [3, 6, 12], ["2", "3-5", "6-11", "12+"]),
        f_gap=binned(c("gap_bc"), [3, 10, 30, 100], ["1-2", "3-9", "10-29", "30-99", "100+"]),
        f_resultB=pl.when(c("scoreB") == 0).then(pl.lit("loss")).when(c("scoreB") == 1).then(pl.lit("win")).otherwise(pl.lit("draw")))
    ch = ch.with_columns(cell=c("habit") + pl.lit(" ") + c("quality"))
    y = ch["rep_bc"].cast(pl.Float64).to_numpy(); pid = ch["pid"].to_numpy()
    mean = lambda ii: float(y[ii].mean()) if len(ii) else np.nan

    def cell(mask):
        idx = np.flatnonzero(mask)
        return {"n": int(len(idx)), "players": int(len(np.unique(pid[idx]))), "repeat_at_C": round(mean(idx), 4),
                "ci": _boot_idx(pid, idx, mean)} if len(idx) else {"n": 0}

    H = (ch["habit"] == "habitual").to_numpy(); Q = ch["quality"].to_numpy(); R = ch["f_resultB"].to_numpy()
    table = {f"{h} {q}": cell((H == (h == "habitual")) & (Q == q))
             for h in ("habitual", "one-off") for q in ("good", "inaccuracy", "mistake", "blunder")}
    by_result = {f"habitual blunder, game B {r}": cell(H & (Q == "blunder") & (R == r)) for r in ("win", "draw", "loss")}
    by_result.update({f"habitual good, game B {r}": cell(H & (Q == "good") & (R == r)) for r in ("win", "draw", "loss")})

    def contrast(a, b):
        idx = np.flatnonzero(a | b)
        f = lambda ii: mean(ii[a[ii]]) - mean(ii[b[ii]])
        return {"value": round(f(idx), 4), "ci": _boot_idx(pid, idx, f)}
    contrasts = {
        "habitual: blunder - good": contrast(H & (Q == "blunder"), H & (Q == "good")),
        "one-off: blunder - good": contrast(~H & (Q == "blunder"), ~H & (Q == "good")),
        "blunder: habitual - one-off": contrast(H & (Q == "blunder"), ~H & (Q == "blunder")),
        "habitual blunder: lost B - won B": contrast(H & (Q == "blunder") & (R == "loss"), H & (Q == "blunder") & (R == "win")),
        "habitual good: lost B - won B": contrast(H & (Q == "good") & (R == "loss"), H & (Q == "good") & (R == "win")),
    }
    # adjusted: logistic GLMM on repeat at C, cell indicators vs 'habitual good', with career count, gap, result of
    # game B, same opponent; no Maia offset (Maia was run on a 1.5M-pair sample, not on every chain)
    REF.update({"cell": "habitual good", "f_k": "3-5", "f_gap": "3-9", "f_resultB": "win"})
    d2 = ch.with_columns(same_opp_f=c("same_opp_bc").cast(pl.Float64))
    X, names = design(d2, ["cell", "f_k", "f_gap", "f_resultB"], ["same_opp_f"])
    fit = fit_glmm(y, X, np.zeros(len(y)), pid, names)
    eff = effects(fit, X, np.zeros(len(y)), names)
    print(f"  fit triples: n={fit['n']:,} players {fit['n_players']:,} sigma {fit['sigma']:.3f} converged {fit['converged']}", flush=True)
    ch.select(pl.exclude("mvB")).write_parquet(OUT / src / "triples.parquet")
    return {"chains": int(ch.height), "players": int(len(np.unique(pid))), "evals": len(fens), "seconds": round(time.time() - t0),
            "quality_mix": {q: int((Q == q).sum()) for q in ("good", "inaccuracy", "mistake", "blunder")},
            "cells": table, "by_result_of_B": by_result, "contrasts": contrasts,
            "fit": {"n": fit["n"], "n_players": fit["n_players"], "sigma": round(fit["sigma"], 3), "converged": fit["converged"],
                    "effects": eff},
            "note": "A->B->C = three consecutive visits by one player to one board (ply 10+ pairs). Quality of the B move: "
                    "Stockfish 20k nodes, expected-score loss; blunder >= 0.2, mistake 0.1-0.2, inaccuracy 0.05-0.1, "
                    "good < 0.05. Habitual = B's move also played at A. Outcome: B's move played again at C. "
                    "CIs: player-cluster bootstrap, 1000 reps. No Maia chance offset."}


def _boot_idx(pid, idx, f, reps=1000, seed=0):
    """Player-cluster bootstrap over the rows idx; f takes an index array into the full arrays."""
    rng = np.random.default_rng(seed)
    p = pid[idx]
    order = np.argsort(p, kind="stable"); ps = p[order]
    cuts = np.flatnonzero(np.diff(ps)) + 1
    groups = np.split(idx[order], cuts)
    out = []
    for _ in range(reps):
        pick = rng.integers(0, len(groups), len(groups))
        out.append(f(np.concatenate([groups[i] for i in pick])))
    return [round(float(np.nanpercentile(out, 2.5)), 4), round(float(np.nanpercentile(out, 97.5)), 4)]


# ------------------------------------------------------------------ report
def md_summary(res: dict) -> str:
    L = ["# Revisit residual model", "", f"Source: `{res['src']}`; written {res['written']}.", ""]
    if "power" in res:
        L += ["## 1. Power (alpha 0.05, 80%)", "",
              "| Ply | Opponent | Pairs | Players | Repeat | sigma | MDE within (pts) | MDE between (pts) | Pairs for 2 pt (within) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in res["power"]["cells"]:
            if "within_mde_pts" not in r:
                L.append(f"| {r['ply']} | {r['opponent']} | {r['pairs']:,} | {r['players']:,} | | | {r.get('note','')} | | |")
                continue
            L.append(f"| {r['ply']} | {r['opponent']} | {r['pairs']:,} | {r['players']:,} | {r['repeat_rate']:.3f} | "
                     f"{r['sigma']} | {r['within_mde_pts']} | {r['between_mde_pts']} | {r['within_pairs_needed']['2pt']:,} |")
        L.append("")
    for key, title in (("fit", "## 2. Mixed-effects fit"), ("momentum", "## 3. Momentum subset")):
        if key not in res:
            continue
        fits = [res["fit"]["main"], res["fit"]["with_same_opponent"]] if key == "fit" else [res["momentum"]["fit"]]
        L += [title, ""]
        for f in fits:
            L += [f"**{f['label']}**: {f['n_pairs']:,} pairs, {f['n_players']:,} players; observed repeat "
                  f"{f['observed_repeat']:.3f} vs chance {f['chance_repeat']:.3f} (personal +{f['personal_repetition_pts']} pts); "
                  f"player sigma {f['player_sigma_logit']} {f['player_sigma_ci']}; converged {f['converged']}.", "",
                  "| Term | Odds ratio [95% CI] | AME (pts) | Rows |", "|---|---|---|---|"]
            for nm, e in f["effects"].items():
                if nm == "intercept":
                    continue
                orr = np.exp(e["coef"]); lo, hi = np.exp(e["ci"][0]), np.exp(e["ci"][1])
                L.append(f"| {nm} | {orr:.2f} [{lo:.2f}, {hi:.2f}] | {e.get('ame_pts', '')} | {e.get('n_rows', '')} |")
            L.append("")
        if key == "momentum":
            L += ["Switch quality (new move vs old move, expected score from the visit-2 board):", "",
                  "| Subset | n | better | worse | same | mean dq |", "|---|---|---|---|---|---|"]
            for k, v in res["momentum"]["switch_quality"].items():
                if isinstance(v, dict) and v.get("n"):
                    L.append(f"| {k} | {v['n']:,} | {v['better']:.3f} | {v['worse']:.3f} | {v['same']:.3f} | {v['mean_dq']:+.4f} |")
            L.append("")
    return "\n".join(L)


def save(res: dict, src: str, md: str | None):
    RES.mkdir(parents=True, exist_ok=True)
    extra = set(res) - {"src", "written"}
    stem = {frozenset({"learn"}): "revisit_learn", frozenset({"triples"}): "revisit_triples"}.get(frozenset(extra), "revisit_model")
    out = RES / (f"{stem}.json" if src == "heldout" else f"{stem}_{src}.json")
    out.write_text(json.dumps(res, indent=1, default=str))
    if md:
        Path(md).write_text(md_summary(res))
    print(f"-> {out}" + (f", {md}" if md else ""), flush=True)


# ------------------------------------------------------------------ selftest
def selftest():
    """Simulate a GLMM with known beta and sigma; check recovery."""
    rng = np.random.default_rng(1)
    ng = 4000; sizes = rng.integers(1, 8, ng); gid = np.repeat(np.arange(ng), sizes); n = len(gid)
    X = np.column_stack([np.ones(n), rng.random(n) < 0.3, rng.normal(size=n)]).astype(np.float64)
    beta = np.array([0.8, -0.4, 0.25]); sig = 0.9
    off = rng.normal(0, 1, n)
    eta = off + X @ beta + rng.normal(0, sig, ng)[gid]
    y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(float)
    f = fit_glmm(y, X, off, gid, ["b0", "b1", "b2"])
    est = f["theta"]
    print("true beta", beta, "sigma", sig)
    print("est  beta", np.round(est[:-1], 3), "sigma", round(f["sigma"], 3), "se", np.round(f["se"], 3), f["converged"])
    ok = np.all(np.abs(est[:-1] - beta) < 3 * f["se"][:-1] + 0.02) and abs(f["sigma"] - sig) < 0.15
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["power", "fit", "momentum", "learn", "triples", "all", "selftest"])
    ap.add_argument("--src", default="heldout", help="data/revisits/<tag> (e.g. heldout, heldout_smoke200000)")
    ap.add_argument("--n", type=int, default=50000, help="momentum sample size (pairs)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--md", default=None, help="also write a markdown summary here")
    a = ap.parse_args()
    if a.cmd == "selftest":
        raise SystemExit(0 if selftest() else 1)
    res = {"src": a.src, "written": time.strftime("%Y-%m-%d %H:%M")}
    if a.cmd in ("power", "all"):
        res["power"] = power(a.src, a.reps)
    if a.cmd in ("fit", "all"):
        res["fit"] = fit_all(a.src)
    if a.cmd in ("momentum", "all"):
        res["momentum"] = momentum(a.src, a.n, a.workers)
    if a.cmd == "learn":
        res["learn"] = learn(a.src, a.workers)
    if a.cmd == "triples":
        res["triples"] = triples(a.src, a.workers)
    save(res, a.src, a.md)


if __name__ == "__main__":
    main()

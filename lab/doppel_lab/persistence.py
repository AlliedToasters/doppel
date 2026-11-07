"""Style persistence: do surface features identify a player across their games? (CPU, no engine)

    python -m doppel_lab.persistence

The bar a style latent (the Doppel z) has to clear. 2026-10 held-out corpus, 3+0 games only (so time-control choice can't pass for style),
held-out players with at least MIN_GAMES such games. Each player's games are split into odd and even (stream order) halves; per half,
the mean of each per-game side feature:
  think_mean, think_cv, fast_share (own moves <= 1 s), trouble_share (own clock < 10% of base), flag_loss,
  resign_loss (lost with Normal termination), plies, e4/d4 as White, c5/e5/e6/c6/d5 replies to 1.e4 as Black,
  book_t1 (2026-10 table), weird_rate (own moves, < 1% of >= 100 book games)
Reported: split-half Spearman per feature across players, raw and after regressing out the player's mean
rating; and re-identification: match each player's half A to the nearest half B (standardised, rating-
residualised features, Euclidean) among pools of POOL players. Top-1 and top-10 accuracy, against rating alone.
Output: lab/results/persistence.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from doppel_lab.book import move_ids, prefix_hashes
from doppel_lab.split import held_out

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "corpus" / "2026-10" / "heldout"
BOOK = ROOT / "data" / "v0" / "book"
RESULTS = ROOT / "lab" / "results"
MIN_GAMES, MAX_PLAYERS, POOL, SEED = 40, 20_000, 1_000, 0
c = pl.col


def side_feats(g: pl.DataFrame) -> pl.DataFrame:
    ref_k, ref_c = np.load(BOOK / "table_keys.npy"), np.load(BOOK / "table_counts.npy")
    ids, off = move_ids(g["moves"])
    keys, cnt = prefix_hashes(ids, off, cap=40)
    i = np.searchsorted(ref_k, keys).clip(max=len(ref_k) - 1)
    m = np.where(ref_k[i] == keys, ref_c[i], 0).astype(float)
    o = np.zeros(len(cnt) + 1, np.int64)
    np.cumsum(cnt, out=o[1:])
    start = o[:-1]
    pos = np.arange(len(keys)) - np.repeat(start, cnt)
    big = np.iinfo(np.int64).max
    f = np.minimum.reduceat(np.where(m < 2, pos, big), start)
    t1 = (np.where(f == big, cnt, f) + 1).astype(float)
    par = np.empty(len(keys))
    par[1:] = m[:-1]
    par[start] = 22_700_000.0
    ok = par >= 100
    wd = ok & (m < 0.01 * par)
    red = lambda x: np.add.reduceat(x.astype(np.int64), start).astype(float)
    white = pos % 2 == 0
    weird = {0: red(wd & white) / np.maximum(red(ok & white), 1), 1: red(wd & ~white) / np.maximum(red(ok & ~white), 1)}
    rows = []
    res = g["result"].to_numpy()
    term = g["termination"].to_numpy()
    mv = g["moves"]
    first, reply = mv.list.get(0, null_on_oob=True).to_numpy(), mv.list.get(1, null_on_oob=True).to_numpy()
    for side, col in ((0, "white"), (1, "black")):
        th = []
        for clocks, base, inc in zip(g["clocks"].to_list(), g["base"].to_list(), g["inc"].to_list()):
            a = np.asarray(clocks[side::2], float)
            sp = np.clip(a[:-1] - a[1:] + inc, 0, 300)[1:]
            if len(sp) < 3:
                th.append((np.nan, np.nan, np.nan, np.nan))
                continue
            mu = sp.mean()
            th.append((mu, sp.std() / mu if mu > 0 else np.nan, float((sp <= 1).mean()), float((a < 0.1 * base).mean())))
        th = np.array(th)
        lost = res == ("0-1" if side == 0 else "1-0")
        d = {"player": g[col].str.to_lowercase(), "elo": g[f"{col}_elo"].to_numpy(), "think_mean": th[:, 0],
             "think_cv": th[:, 1], "fast_share": th[:, 2], "trouble_share": th[:, 3],
             "flag_loss": (lost & (term == "Time forfeit")).astype(float),
             "resign_loss": (lost & (term == "Normal")).astype(float), "plies": cnt.astype(float) * 0 + g["moves"].list.len().to_numpy(),
             "book_t1": t1, "weird_rate": weird[side]}
        if side == 0:
            d |= {"w_e4": (first == "e2e4").astype(float), "w_d4": (first == "d2d4").astype(float)}
        else:
            e4 = first == "e2e4"
            for name, u in (("b_c5", "c7c5"), ("b_e5", "e7e5"), ("b_e6", "e7e6"), ("b_c6", "c7c6"), ("b_d5", "d7d5")):
                d[name] = np.where(e4, (reply == u).astype(float), np.nan)
        rows.append(pl.DataFrame(d))
    return pl.concat(rows, how="diagonal")


FEATS = ["think_mean", "think_cv", "fast_share", "trouble_share", "flag_loss", "resign_loss", "plies", "book_t1",
         "weird_rate", "w_e4", "w_d4", "b_c5", "b_e5", "b_e6", "b_c6", "b_d5"]


def main():
    rng = np.random.default_rng(SEED)
    g = pl.concat([pl.read_parquet(f) for f in sorted(CORPUS.glob("part-*.parquet"))]).with_row_index("t")
    # One time control only (3+0, 60% of games): a player's time-control choice would otherwise pass for style.
    g = g.filter((c("base") == 180) & (c("inc") == 0))
    names = pl.concat([g["white"], g["black"]]).str.to_lowercase().value_counts()
    names = names.filter(c("count") >= MIN_GAMES)
    names = names.filter(pl.Series([held_out(n) for n in names["white"].to_list()]))
    keep = set(names["white"].to_list())
    if len(keep) > MAX_PLAYERS:
        keep = set(rng.choice(sorted(keep), MAX_PLAYERS, replace=False).tolist())
    g = g.filter(c("white").str.to_lowercase().is_in(list(keep)) | c("black").str.to_lowercase().is_in(list(keep)))
    print(f"{len(keep):,} players with >= {MIN_GAMES} games; {g.height:,} of their games", flush=True)
    s = side_feats(g).filter(c("player").is_in(list(keep)))
    s = s.with_columns(k=pl.int_range(pl.len()).over("player")).with_columns(half=c("k") % 2)
    agg = s.group_by("player", "half").agg(c("elo").mean(), *[c(f).fill_nan(None).mean() for f in FEATS])
    A = agg.filter(c("half") == 0).sort("player")
    B = agg.filter(c("half") == 1).sort("player")
    A, B = A.join(B.select("player"), on="player"), B.join(A.select("player"), on="player")
    elo = ((A["elo"] + B["elo"]) / 2).to_numpy()
    res = {"players": A.height, "min_games": MIN_GAMES, "features": {}}

    def resid(x):  # regress out mean rating (linear + quadratic)
        ok = np.isfinite(x)
        Z = np.c_[np.ones(ok.sum()), elo[ok], elo[ok] ** 2]
        out = np.full_like(x, np.nan)
        out[ok] = x[ok] - Z @ np.linalg.lstsq(Z, x[ok], rcond=None)[0]
        return out

    XA, XB = [], []
    for f in FEATS:
        a, b = A[f].to_numpy().astype(float), B[f].to_numpy().astype(float)
        ok = np.isfinite(a) & np.isfinite(b)
        ra, rb = resid(np.where(ok, a, np.nan)), resid(np.where(ok, b, np.nan))
        r_raw = spearmanr(a[ok], b[ok]).statistic
        r_res = spearmanr(ra[ok], rb[ok]).statistic
        r_elo = spearmanr(a[ok], elo[ok]).statistic
        res["features"][f] = {"split_half_rho": float(r_raw), "split_half_rho_rating_resid": float(r_res),
                              "rho_with_rating": float(r_elo), "players": int(ok.sum())}
        print(f"{f:14s} split-half rho {r_raw:+.3f}  rating-residualised {r_res:+.3f}  (rho with rating {r_elo:+.3f})", flush=True)
        mu, sd = np.nanmean(np.r_[ra, rb]), np.nanstd(np.r_[ra, rb])
        XA.append(np.nan_to_num((ra - mu) / sd))
        XB.append(np.nan_to_num((rb - mu) / sd))
    XA, XB = np.array(XA).T, np.array(XB).T
    # re-identification in random pools
    n = len(elo)
    top1 = top10 = top1_elo = top10_elo = top1_both = top10_both = 0
    eall = np.r_[A["elo"].to_numpy(), B["elo"].to_numpy()]
    ez = lambda e: (e - eall.mean()) / eall.std()
    trials = 0
    for _ in range(max(1, n // POOL)):
        idx = rng.choice(n, min(POOL, n), replace=False)
        D = ((XA[idx][:, None, :] - XB[idx][None, :, :]) ** 2).sum(-1)
        rank = (D < D[np.arange(len(idx)), np.arange(len(idx))][:, None]).sum(1)
        eA, eB = A["elo"].to_numpy()[idx], B["elo"].to_numpy()[idx]
        De = np.abs(eA[:, None] - eB[None, :])
        rank_e = (De < De[np.arange(len(idx)), np.arange(len(idx))][:, None]).sum(1)
        Db = D + XA.shape[1] * (ez(eA)[:, None] - ez(eB)[None, :]) ** 2       # rating weighted like all style dims together
        rank_b = (Db < Db[np.arange(len(idx)), np.arange(len(idx))][:, None]).sum(1)
        top1_both += (rank_b == 0).sum(); top10_both += (rank_b < 10).sum()
        top1 += (rank == 0).sum(); top10 += (rank < 10).sum()
        top1_elo += (rank_e == 0).sum(); top10_elo += (rank_e < 10).sum()
        trials += len(idx)
    res["reid"] = {"pool": POOL, "trials": trials, "top1": top1 / trials, "top10": top10 / trials,
                   "top1_rating_only": top1_elo / trials, "top10_rating_only": top10_elo / trials,
                   "top1_style_and_rating": top1_both / trials, "top10_style_and_rating": top10_both / trials,
                   "chance_top1": 1 / POOL, "time_control": "180+0"}
    print(f"re-identification in pools of {POOL}: surface style top-1 {top1 / trials:.3f}, top-10 {top10 / trials:.3f}; "
          f"rating alone top-1 {top1_elo / trials:.3f}, top-10 {top10_elo / trials:.3f}; "
          f"style + rating top-1 {top1_both / trials:.3f}, top-10 {top10_both / trials:.3f}; chance {1 / POOL:.3f}", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "persistence.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()

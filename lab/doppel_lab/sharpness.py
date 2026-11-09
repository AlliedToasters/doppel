"""Sharpness as style: do players steer into sharp positions, and is that a stable, useful trait? (CPU only)

    python -m doppel_lab.sharpness style        # -> lab/results/sharpness_style.json
    python -m doppel_lab.sharpness rating       # -> lab/results/sharpness_rating.json (needs data/human/teacher_e1500)

rating: the 36k-game, six-band human corpus (game_science's tables; 12k games carry Stockfish evals). Maia-3 79M
run with BOTH Elos fixed at 1500 (teacher.py --fixed-elo), so its outputs carry no rating input. Asks:
  1. do sharpness-gated features (loss and blunders in sharp vs quiet positions, obvious misses, steering,
     time allocation) add rating signal beyond the T2 aggregates? (game_mean model, the 12k eval games)
  2. can rating-free T1 (Maia agreement, entropy, the gated T1 features) stand in for T2? (T0 vs T0+T1 vs T2 on
     the eval subset, and T0 vs T0+T1 on all 36k)

1500-band train table (moves_f1_T2_1500_train: Stockfish wc_loss and Maia-3 79M outputs per position; ratings
1460-1640, so this is a style question, not a rating one). Sharpness of a position = Maia-3 policy entropy (how
spread human move choice is there). Own moves at ply >= 10. Per side-game:
  faced_ent     mean entropy of the positions the player moved in (set by both players)
  steer_ent     mean entropy of the position the player hands the opponent, minus what the faced position and ply
                predict (pooled linear fit): does this player's move choice make the game sharper than usual?
  steer_vol     the same for Stockfish win% volatility after the move
  sharp_gap     mean wc_loss in sharp positions (faced entropy top third) minus quiet ones (bottom third)
  alloc_rho     Spearman of think fraction vs entropy over the side's moves with clock >= 10% (time allocation)
  obvious_miss  share of "obvious" positions (Maia top move >= 0.8) where the player chose another move
  maia_lp       mean Maia log-prob of the played move
Per player (>= MIN_GAMES featured games, odd/even split by game order): split-half Spearman, raw and with mean
rating regressed out; and predictive checks across halves: half-A trait vs half-B performance residual (score
minus the Elo expectation) and vs half-B opponents' blunder rate.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "data/v0/features/moves_f1_T2_1500_train"
GAMES = ROOT / "data/v0/games_1500_train.parquet"
RESULTS = ROOT / "lab/results"
MIN_GAMES = 16
c = pl.col
TRAITS = ["faced_ent", "steer_ent", "steer_vol", "sharp_gap", "alloc_rho", "obvious_miss", "maia_lp",
          "opp_blunder", "blunder"]


def side_table() -> pl.DataFrame:
    cols = ["game_id", "mover", "ply", "maia_entropy", "maia_p_top", "maia_played_rank", "maia_played_lp",
            "wc_loss", "blunder", "think_frac", "clock_frac", "wc_w_volatility"]
    t = (pl.scan_parquet(TABLE / "part-*.parquet").select(cols).collect()
         .with_columns([c(k).cast(pl.Float64).fill_nan(None) for k in cols[2:]]).sort("game_id", "ply")
         .with_columns(handed_ent=c("maia_entropy").shift(-1).over("game_id"),
                       handed_vol=c("wc_w_volatility").shift(-1).over("game_id"),
                       opp_blunder_next=c("blunder").shift(-1).over("game_id")))
    t = t.filter(c("ply") >= 10)
    # steering residuals: handed sharpness beyond what the faced position and ply predict
    for y, x in (("handed_ent", "maia_entropy"), ("handed_vol", "wc_w_volatility")):
        ok = t.select(c(y).is_not_null() & c(x).is_not_null()).to_series()
        d = t.filter(ok)
        Z = np.c_[np.ones(d.height), d[x].to_numpy(), d[x].to_numpy() ** 2, np.minimum(d["ply"].to_numpy(), 80)]
        beta = np.linalg.lstsq(Z, d[y].to_numpy(), rcond=None)[0]
        pred = beta[0] + beta[1] * c(x) + beta[2] * c(x) ** 2 + beta[3] * pl.min_horizontal(c("ply"), 80)
        t = t.with_columns((c(y) - pred).alias(f"{y}_res"))
    lo, hi = np.nanquantile(t["maia_entropy"].to_numpy(), [1 / 3, 2 / 3])
    print(f"own moves: {t.height:,}; entropy terciles {lo:.2f} / {hi:.2f}", flush=True)
    timed = c("clock_frac") >= 0.1
    s = t.group_by("game_id", "mover").agg(
        faced_ent=c("maia_entropy").mean(),
        steer_ent=c("handed_ent_res").mean(),
        steer_vol=c("handed_vol_res").mean(),
        sharp_gap=c("wc_loss").filter(c("maia_entropy") >= hi).mean() - c("wc_loss").filter(c("maia_entropy") <= lo).mean(),
        alloc_rho=pl.corr(c("think_frac").filter(timed), c("maia_entropy").filter(timed), method="spearman"),
        obvious_miss=(c("maia_played_rank") > 1).filter(c("maia_p_top") >= 0.8).mean(),
        maia_lp=c("maia_played_lp").mean(),
        opp_blunder=c("opp_blunder_next").mean(),
        blunder=c("blunder").mean(),
        n_own=pl.len())
    return s.with_columns([c(k).fill_nan(None) for k in TRAITS])


def teacher_frame(d: Path) -> pl.DataFrame:
    parts = []
    for f in sorted(d.glob("shard-*.npz")):
        z = np.load(f)
        n = np.diff(z["offsets"])
        parts.append(pl.DataFrame({
            "game_id": np.repeat(z["game_ids"], n), "ply": np.concatenate([np.arange(k) for k in n]),
            "t_ent": z["entropy"].astype(np.float64), "t_lp": z["played_lp"].astype(np.float64),
            "t_ptop": np.exp(z["top_lp"][:, 0].astype(np.float64)),
            "t_top1": (z["played_idx"] == z["top_idx"][:, 0]).astype(np.float64)}))
    return pl.concat(parts).with_columns(c("ply").cast(pl.Int64))


T1_FEATS = ["t1_lp", "t1_top1", "t1_ent", "t1_lp_sharp", "t1_lp_quiet", "t1_obvious_miss", "t1_steer"]
SHARP_FEATS = ["sh_loss_sharp", "sh_loss_quiet", "sh_blunder_sharp", "sh_blunder_quiet", "sh_obvious_loss",
               "sh_alloc_rho"]


def rating():
    from doppel_lab import game_science as gs
    t0 = time.time()
    sides, games = gs.tables()
    ply = pl.read_parquet(gs.PLY, columns=["game_id", "ply", "mover", "wc_loss", "blunder", "think_frac", "clock_frac"])
    ply = ply.with_columns([c(k).cast(pl.Float64).fill_nan(None) for k in ("wc_loss", "blunder", "think_frac",
                                                                             "clock_frac")])
    tf = teacher_frame(ROOT / "data/human/teacher_e1500")
    p = (ply.join(tf, on=["game_id", "ply"], how="inner").sort("game_id", "ply")
         .with_columns(handed=c("t_ent").shift(-1).over("game_id")).filter(c("ply") >= 10))
    print(f"{p.height:,} own moves with fixed-Elo Maia ({time.time() - t0:.0f}s)", flush=True)
    d = p.filter(c("handed").is_not_null())
    Z = np.c_[np.ones(d.height), d["t_ent"].to_numpy(), d["t_ent"].to_numpy() ** 2]
    beta = np.linalg.lstsq(Z, d["handed"].to_numpy(), rcond=None)[0]
    p = p.with_columns(steer=c("handed") - (beta[0] + beta[1] * c("t_ent") + beta[2] * c("t_ent") ** 2))
    lo, hi = np.quantile(p["t_ent"].to_numpy(), [1 / 3, 2 / 3])
    sharp, quiet, obvious = c("t_ent") >= hi, c("t_ent") <= lo, c("t_ptop") >= 0.8
    timed = c("clock_frac") >= 0.1
    agg = p.group_by("game_id", "mover").agg(
        t1_lp=c("t_lp").mean(), t1_top1=c("t_top1").mean(), t1_ent=c("t_ent").mean(),
        t1_lp_sharp=c("t_lp").filter(sharp).mean(), t1_lp_quiet=c("t_lp").filter(quiet).mean(),
        t1_obvious_miss=(1 - c("t_top1")).filter(obvious).mean(), t1_steer=c("steer").mean(),
        sh_loss_sharp=c("wc_loss").filter(sharp).mean(), sh_loss_quiet=c("wc_loss").filter(quiet).mean(),
        sh_blunder_sharp=c("blunder").filter(sharp).mean(), sh_blunder_quiet=c("blunder").filter(quiet).mean(),
        sh_obvious_loss=c("wc_loss").filter(obvious).mean(),
        sh_alloc_rho=pl.corr(c("think_frac").filter(timed), c("t_ent").filter(timed), method="spearman"))
    feats = T1_FEATS + SHARP_FEATS
    agg = agg.with_columns([c(k).cast(pl.Float64).fill_nan(None) for k in feats])
    for side, pre in ((0, "w_"), (1, "b_")):
        games = games.join(agg.filter(c("mover") == side).select("game_id", *[c(k).alias(pre + k) for k in feats]),
                           on="game_id", how="left")
    both = lambda fs: [pre + f for pre in ("w_", "b_") for f in fs]
    tc = ["base", "inc"]
    t0f = tc + ["plies", "decisive", "time_forfeit"] + gs.tier_feats("T0", "w_") + gs.tier_feats("T0", "b_")
    t2f = tc + ["plies", "decisive", "time_forfeit", "lead_changes"] + gs.tier_feats("T2", "w_") + \
        gs.tier_feats("T2", "b_")
    ev = games.filter(c("has_eval"))
    jobs = {
        "eval/T0": (ev, t0f), "eval/T0+T1": (ev, t0f + both(T1_FEATS)), "eval/T2": (ev, t2f),
        "eval/T2+sharp": (ev, t2f + both(SHARP_FEATS)), "eval/T2+T1": (ev, t2f + both(T1_FEATS)),
        "eval/T2+T1+sharp": (ev, t2f + both(T1_FEATS + SHARP_FEATS)),
        "all/T0": (games, t0f), "all/T0+T1": (games, t0f + both(T1_FEATS)),
    }
    out = {"entropy_terciles": [float(lo), float(hi)], "steer_fit": beta.tolist(), "rating": {}}
    for name, (df, fs) in jobs.items():
        X = df.select([c(f).cast(pl.Float64) for f in fs]).to_numpy()
        y = df["elo_mean"].cast(pl.Float64).to_numpy()
        r = gs.reg_scores(y, gs.cv_reg(X, y, df["game_id"].to_numpy()), df["band"].to_numpy())
        out["rating"][name] = {k: r[k] for k in ("n", "mae", "rmse", "r2", "pearson", "band_exact")} | {
            "n_features": len(fs)}
        print(f"{name:18} n {r['n']:6} mae {r['mae']:6.1f} r2 {r['r2']:+.3f} ({len(fs)} feats)", flush=True)
    se = sides.join(agg, on=["game_id", "mover"], how="inner")
    elo = se["elo"].cast(pl.Float64).to_numpy()
    out["per_feature_rho_rating"] = {}
    for f in feats:
        x = se[f].to_numpy()
        ok = np.isfinite(x)
        out["per_feature_rho_rating"][f] = float(spearmanr(x[ok], elo[ok]).statistic)
    print(json.dumps(out["per_feature_rho_rating"], indent=1))
    (RESULTS / "sharpness_rating.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"-> sharpness_rating.json ({time.time() - t0:.0f}s)")


def style():
    t0 = time.time()
    s = side_table()
    g = pl.read_parquet(GAMES, columns=["game_id", "white", "black", "white_elo", "black_elo", "result", "base", "inc"])
    s = s.join(g, on="game_id").with_columns(
        player=pl.when(c("mover") == 0).then(c("white")).otherwise(c("black")).str.to_lowercase(),
        elo=pl.when(c("mover") == 0).then(c("white_elo")).otherwise(c("black_elo")).cast(pl.Float64),
        opp_elo=pl.when(c("mover") == 0).then(c("black_elo")).otherwise(c("white_elo")).cast(pl.Float64),
        score=pl.when(c("result") == "1/2-1/2").then(0.5).when(
            (c("result") == "1-0") == (c("mover") == 0)).then(1.0).otherwise(0.0))
    s = s.with_columns(perf=c("score") - 1 / (1 + 10 ** ((c("opp_elo") - c("elo")) / 400)))
    # game-level descriptive: traits vs result residual (endogenous: sharp games are also decided games)
    out = {"n_side_games": s.height, "min_games": MIN_GAMES, "traits": TRAITS,
           "game_level_rho_with_perf": {k: float(spearmanr(*s.select(k, "perf").drop_nulls().to_numpy().T).statistic)
                                        for k in TRAITS}}
    keep = s.group_by("player").len().filter(c("len") >= MIN_GAMES)["player"]
    p = (s.filter(c("player").is_in(keep.implode())).sort("player", "game_id")  # game_id order is arbitrary but fixed
         .with_columns(half=pl.int_range(pl.len()).over("player") % 2))
    agg = p.group_by("player", "half").agg(c("elo").mean(), c("perf").mean(), *[c(k).mean() for k in TRAITS],
                                           games=pl.len())
    A = agg.filter(c("half") == 0).sort("player")
    B = agg.filter(c("half") == 1).sort("player")
    A, B = A.join(B.select("player"), on="player"), B.join(A.select("player"), on="player")
    elo = ((A["elo"] + B["elo"]) / 2).to_numpy()
    print(f"{A.height:,} players with >= {MIN_GAMES} games ({time.time() - t0:.0f}s)", flush=True)

    def resid(x):
        ok = np.isfinite(x)
        Z = np.c_[np.ones(ok.sum()), elo[ok], elo[ok] ** 2]
        r = np.full_like(x, np.nan)
        r[ok] = x[ok] - Z @ np.linalg.lstsq(Z, x[ok], rcond=None)[0]
        return r

    out["players"] = A.height
    out["per_trait"] = {}
    perf_b, perf_a = B["perf"].to_numpy(), A["perf"].to_numpy()
    for k in TRAITS:
        a, b = A[k].to_numpy().astype(float), B[k].to_numpy().astype(float)
        ok = np.isfinite(a) & np.isfinite(b)
        ra, rb = resid(a), resid(b)
        # cross-half predictive validity, symmetrised: trait in one half vs outcome in the other
        cross_perf = (spearmanr(ra[ok], perf_b[ok]).statistic + spearmanr(rb[ok], perf_a[ok]).statistic) / 2
        cross_oppbl = (spearmanr(ra[ok], B["opp_blunder"].to_numpy()[ok]).statistic +
                       spearmanr(rb[ok], A["opp_blunder"].to_numpy()[ok]).statistic) / 2
        r = {"split_half_rho": float(spearmanr(a[ok], b[ok]).statistic),
             "split_half_rho_rating_resid": float(spearmanr(ra[ok], rb[ok]).statistic),
             "rho_with_rating": float(spearmanr(a[ok], elo[ok]).statistic),
             "cross_half_rho_perf": float(cross_perf), "cross_half_rho_opp_blunder": float(cross_oppbl),
             "players": int(ok.sum()), "mean": float(np.nanmean(np.r_[a, b]))}
        out["per_trait"][k] = r
        print(f"{k:13s} split-half {r['split_half_rho']:+.3f} resid {r['split_half_rho_rating_resid']:+.3f} | "
              f"rho elo {r['rho_with_rating']:+.3f} | cross perf {cross_perf:+.3f} opp-blunder {cross_oppbl:+.3f}",
              flush=True)
    # trait inter-correlation (players, half A, rating-residualised)
    M = np.array([resid(A[k].to_numpy().astype(float)) for k in TRAITS])
    ok = np.all(np.isfinite(M), 0)
    out["trait_corr_half_a"] = {k: dict(zip(TRAITS, map(float, row)))
                                for k, row in zip(TRAITS, spearmanr(M[:, ok].T).statistic)}
    # perf reliability ceiling: split-half rho of perf itself
    out["perf_split_half_rho"] = float(spearmanr(perf_a, perf_b).statistic)
    print(f"perf split-half rho {out['perf_split_half_rho']:+.3f}", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / ("sharpness_style.json" if MIN_GAMES == 16 else f"sharpness_style_min{MIN_GAMES}.json")).write_text(json.dumps(out, indent=1) + "\n")
    print(f"-> sharpness_style.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["style", "rating"])
    ap.add_argument("--min-games", type=int, default=MIN_GAMES)
    a = ap.parse_args()
    MIN_GAMES = a.min_games
    {"style": style, "rating": rating}[a.cmd]()
    sys.exit(0)

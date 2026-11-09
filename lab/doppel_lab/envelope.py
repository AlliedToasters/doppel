"""Envelope features: do shape/timing/difficulty features of the eval trajectory add rating signal beyond
game_science's T2 aggregates? (CPU only; reads data/v0/features/human_f1_T2.parquet, built by game_science.)

Families (per side unless noted; all computed from Stockfish per-ply columns, so tier T2):
  quiet     difficulty-adjusted loss. Sharpness is not stored for this corpus, so "quiet" = the mover's local
            Lichess win% volatility below QUIET; loss there is the cheap mistake a strong player avoids.
  underwater  area below the side's running peak win% (Ulcer-index style): mean drawdown, RMS drawdown,
            share of own moves more than 10 win% points underwater.
  band      own move-quality band: Goh-Barabasi burstiness of wc_loss, spikes above the causal upper band
            (ewma_prev + 2 * roll_std_prev), and band width (mean roll_std of wc_loss) per half of the game.
  macd      per game: mean |fast EMA - slow EMA| of White's win% (alpha 0.5 vs 0.1), and its sign changes.

Scored as game_science's game_mean/T2/eval_subset: same games, same 5-fold CV and model, baseline features plus
each family. Also: Spearman with rating, within-band correlation with acpl (redundancy), and within-band
first-half vs second-half persistence of band width.

    python -m doppel_lab.envelope        # -> lab/results/envelope.json
"""

from __future__ import annotations

import json
import time

import polars as pl

from doppel_lab import game_science as gs

c = pl.col
QUIET = 2.0  # win% points of local volatility
FAMILIES = {
    "quiet": ["quiet_loss", "quiet_blunder_rate", "quiet_share"],
    "underwater": ["uw_mean", "uw_rms", "uw_share10"],
    "band": ["burstiness", "spike_rate", "band_w_h1", "band_w_h2"],
    "macd": ["macd_area", "macd_cross"],  # per game
    "time": ["loss_per_s", "uw_time_w", "vol_time_w"],  # per side: move-axis features re-indexed by clock time
    "time_game": ["swing_per_min", "vol_time_w_game"],  # per game
}
GAME_FAMS = ("macd", "time_game")
SIDE_FEATS = [f for k, v in FAMILIES.items() if k not in GAME_FAMS for f in v]


def side_envelope(ply: pl.DataFrame) -> pl.DataFrame:
    p = ply.sort("game_id", "ply").with_columns(
        own_i=pl.int_range(pl.len()).over("game_id", "mover"),
        own_n=pl.len().over("game_id", "mover"))
    quiet = c("wc_w_volatility") < QUIET
    upper = c("wc_loss_ewma_prev") + 2 * c("wc_loss_roll_std_prev")
    h1 = c("own_i") < c("own_n") // 2
    m, s = c("wc_loss").mean(), c("wc_loss").std()
    return p.group_by("game_id", "mover").agg(
        quiet_loss=c("wc_loss").filter(quiet).mean(),
        quiet_blunder_rate=c("blunder").filter(quiet).mean(),
        quiet_share=quiet.mean(),
        uw_mean=c("wc_w_drawdown_own").mean(),
        uw_rms=(c("wc_w_drawdown_own") ** 2).mean().sqrt(),
        uw_share10=(c("wc_w_drawdown_own") < -10).mean(),
        burstiness=(s - m) / (s + m),
        spike_rate=(c("wc_loss") > upper).filter(upper.is_not_null()).mean(),
        band_w_h1=c("wc_loss_roll_std").filter(h1).mean(),
        band_w_h2=c("wc_loss_roll_std").filter(~h1).mean(),
        # time axis: weight each own move by the seconds spent on it, not by 1
        loss_per_s=c("wc_loss").sum() / c("think_s").sum(),
        uw_time_w=(c("wc_w_drawdown_own") * c("think_s")).sum() / c("think_s").filter(
            c("wc_w_drawdown_own").is_not_null()).sum(),
        vol_time_w=(c("wc_w_volatility") * c("think_s")).sum() / c("think_s").filter(
            c("wc_w_volatility").is_not_null()).sum(),
    ).with_columns([c(f).cast(pl.Float64) for f in SIDE_FEATS])


def game_macd(ply: pl.DataFrame) -> pl.DataFrame:
    p = (ply.filter(c("wc_own").is_not_null()).sort("game_id", "ply")
         .with_columns(pct=50 + 50 * pl.when(c("mover") == 0).then(c("wc_own")).otherwise(-c("wc_own")))
         .with_columns(fast=c("pct").ewm_mean(alpha=0.5, adjust=False).over("game_id"),
                       slow=c("pct").ewm_mean(alpha=0.1, adjust=False).over("game_id"))
         .with_columns(gap=c("fast") - c("slow")))
    return p.group_by("game_id").agg(
        macd_area=c("gap").abs().mean(),
        macd_cross=(c("gap").sign().diff().abs() > 0).sum().cast(pl.Float64))


def game_time(ply: pl.DataFrame) -> pl.DataFrame:
    """Per game, on a clock-time axis: win% swing per minute of play, and win% std weighted by time in position."""
    p = (ply.filter(c("wc_own").is_not_null()).sort("game_id", "ply")
         .with_columns(pct=50 + 50 * pl.when(c("mover") == 0).then(c("wc_own")).otherwise(-c("wc_own")))
         .with_columns(dpct=c("pct").diff().abs().over("game_id"), w=c("think_s").clip(0.1)))
    mu = (c("pct") * c("w")).sum() / c("w").sum()
    return p.group_by("game_id").agg(
        swing_per_min=c("dpct").sum() / (c("think_s").sum() / 60),
        vol_time_w_game=(((c("pct") - mu) ** 2 * c("w")).sum() / c("w").sum()).sqrt())


def within_band_rho(df: pl.DataFrame, a: str, b: str) -> float:
    r = df.filter(c(a).is_not_null() & c(b).is_not_null()).with_columns(
        c(a).rank().over("band").alias("_a"), c(b).rank().over("band").alias("_b"))
    return gs.spearman(r["_a"].cast(pl.Float64).to_numpy(), r["_b"].cast(pl.Float64).to_numpy())


def run():
    t = time.time()
    sides, games = gs.tables()
    ply = pl.read_parquet(gs.PLY)
    ply = ply.with_columns([c(k).fill_nan(None) for k, ty in ply.schema.items() if ty.is_float()])
    env = side_envelope(ply)
    sides = sides.join(env, on=["game_id", "mover"], how="left")
    w = env.filter(c("mover") == 0).select("game_id", *[c(f).alias(f"w_{f}") for f in SIDE_FEATS])
    b = env.filter(c("mover") == 1).select("game_id", *[c(f).alias(f"b_{f}") for f in SIDE_FEATS])
    games = (games.join(w, on="game_id", how="left").join(b, on="game_id", how="left")
             .join(game_macd(ply), on="game_id", how="left").join(game_time(ply), on="game_id", how="left"))
    ev = games.filter(c("has_eval"))
    print(f"features built ({time.time() - t:.0f}s), {ev.height:,} games with evals", flush=True)

    base = ["base", "inc", "plies", "decisive", "time_forfeit", "lead_changes"] + gs.tier_feats("T2", "w_") + \
        gs.tier_feats("T2", "b_")

    def fam(k):
        return FAMILIES[k] if k in GAME_FAMS else [f"{s}{f}" for s in ("w_", "b_") for f in FAMILIES[k]]

    jobs = {"baseline": base} | {f"+{k}": base + fam(k) for k in FAMILIES} | {
        "+time_all": base + fam("time") + fam("time_game"),
        "+all": base + [f for k in FAMILIES for f in fam(k)],
        "envelope_only": ["base", "inc"] + [f for k in FAMILIES for f in fam(k)]}
    y = ev["elo_mean"].cast(pl.Float64).to_numpy()
    out = {"n_games": ev.height, "quiet_threshold": QUIET, "families": FAMILIES, "rating": {}}
    for name, feats in jobs.items():
        X = ev.select([c(f).cast(pl.Float64) for f in feats]).to_numpy()
        p = gs.cv_reg(X, y, ev["game_id"].to_numpy())
        r = gs.reg_scores(y, p, ev["band"].to_numpy())
        out["rating"][name] = {k: r[k] for k in ("mae", "rmse", "r2", "pearson", "band_exact")} | {
            "pred_sd_by_band": {k: v["sd"] for k, v in r["pred_by_band"].items()}, "n_features": len(feats)}
        print(f"{name:14} mae {r['mae']:6.1f}  r2 {r['r2']:+.3f}  ({len(feats)} feats)", flush=True)

    se = sides.filter(c("has_eval"))
    elo = se["elo"].cast(pl.Float64).to_numpy()
    out["per_feature"] = {f: {"rho_rating": gs.spearman(se[f].to_numpy(), elo),
                              "within_band_rho_acpl": within_band_rho(se, f, "acpl"),
                              "within_band_rho_elo": within_band_rho(se, f, "elo")} for f in SIDE_FEATS}
    gm = ev["elo_mean"].cast(pl.Float64).to_numpy()
    for f in FAMILIES["macd"] + FAMILIES["time_game"]:
        out["per_feature"][f] = {"rho_rating": gs.spearman(ev[f].to_numpy(), gm),
                                 "within_band_rho_volatility": within_band_rho(
                                     ev.with_columns(vol=(c("w_volatility") + c("b_volatility")) / 2), f, "vol")}
    # persistence: does first-half band width predict second-half band width, beyond the player's band?
    out["persistence"] = {"band_w_h1_vs_h2_within_band": within_band_rho(se, "band_w_h1", "band_w_h2")}
    path = gs.RESULTS / "envelope.json"
    path.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps({"per_feature": out["per_feature"], "persistence": out["persistence"]}, indent=1))
    print(f"-> {path} ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    run()

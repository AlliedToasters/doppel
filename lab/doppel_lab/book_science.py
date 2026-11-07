"""Book science and upsets, on top of doppel_lab.book counts (CPU only). -> lab/results/book_science.json

Book (per evaluation game, per ply k; c_k = train games that played the same first k+1 moves, c_-1 = N):
  curve        log10(c_k + 1) by ply: median and quartiles, per band
  exit         book-exit ply: first ply with c_k < T (T = 1: never seen; 10; 100), per band
  weird        conditional frequency q_k = c_k / c_{k-1} where c_{k-1} >= 100: the share of games on this line that
               chose this move. weird = q < 1%. Per band: rate, and mean log10 q ("book log-likelihood")
  scaling      book exit vs corpus size (subsamples N/2^j): plies gained per doubling, and the N for a target depth
Upsets (heldout games with a rating gap >= 300):
  dramatic     underdog wins ranked by gap and by surprise -log2(Elo expected score)
  signals      P(upset | gap, time control) vs + early-game features (first 30 plies, T0 and book), by
               gradient-boosted classifier with 5-fold CV; and upset-vs-expected feature contrasts in gap bins

    python -m doppel_lab.book_science
"""

from __future__ import annotations

import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import polars as pl

from doppel_lab.book import LEVELS, OUT
from doppel_lab.features.table import game_rows

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "lab/results"
EARLY = 30
SEED = 0
c = pl.col


def load():
    g = pl.read_parquet(OUT / "eval_games.parquet")
    counts = np.load(OUT / "eval_counts.npy").astype(np.int64)
    lens = np.load(OUT / "eval_lens.npy")
    meta = json.loads((OUT / "train_meta.json").read_text())
    off = np.zeros(len(lens) + 1, np.int64)
    np.cumsum(lens, out=off[1:])
    return g, counts, lens, off, meta


def per_ply(counts, lens, off, n_level):
    """Long arrays: game index, ply, count, parent count, conditional frequency."""
    gi = np.repeat(np.arange(len(lens)), lens)
    ply = np.arange(len(gi)) - off[gi]
    parent = np.empty_like(counts)
    parent[1:] = counts[:-1]
    first = ply == 0
    parent[first] = n_level[None, :]
    return gi, ply, parent


def exit_ply(counts_j, gi, ply, lens, T, cap):
    """First ply with count < T per game (cap if never within the hashed plies; censored)."""
    out = np.full(len(lens), cap, np.int64)
    below = np.nonzero(counts_j < T)[0]
    # first occurrence per game: below is sorted, so take the first index per gi
    g_b = gi[below]
    first = np.unique(g_b, return_index=True)
    out[first[0]] = ply[below[first[1]]]
    shorter = lens < cap  # games that ended inside book: exit = their length (also censored)
    never = np.ones(len(lens), bool)
    never[first[0]] = False
    out[never & shorter] = lens[never & shorter]
    return out, never


def q_stats(df: pl.DataFrame, by: str) -> list[dict]:
    return (df.group_by(by).agg(n=pl.len(), weird_rate=(c("q") < 0.01).mean(), log10_q=c("q").log10().mean(),
                                 top_like=(c("q") > 0.5).mean()).sort(by).to_dicts())


def book(g, counts, lens, off, meta) -> tuple[dict, pl.DataFrame]:
    N = meta["games"]
    # expected subsample sizes: P(level >= j) = 2^-j (the last level is capped, same formula holds)
    n_level = np.array([N * 2.0 ** -j for j in range(LEVELS)])
    gi, ply, parent = per_ply(counts, lens, off, n_level)
    band = g["band"].to_numpy()
    src = g["src"].to_numpy()
    cap = int(lens.max())
    out = {"train_games": N, "cap": cap}
    ho = src == "heldout"

    # curve (j = 0), heldout stratified sample
    lc = np.log10(counts[:, 0] + 1)
    d = pl.DataFrame({"band": band[gi], "ply": ply, "lc": lc, "ho": ho[gi]}).filter(c("ho"))
    out["curve"] = (d.group_by("band", "ply").agg(n=pl.len(), q25=c("lc").quantile(0.25), med=c("lc").median(),
                                                  q75=c("lc").quantile(0.75), mean=c("lc").mean(),
                                                  share_unique=(c("lc") == 0).mean())
                    .sort("band", "ply").to_dicts())
    # exit ply per band and threshold
    ex = {}
    for T in (1, 10, 100):
        e, never = exit_ply(counts[:, 0], gi, ply, lens, T, cap)
        rows = []
        for b in sorted(set(band[ho])):
            m = ho & (band == b)
            rows.append({"band": int(b), "n": int(m.sum()), "median": float(np.median(e[m])),
                         "mean": float(e[m].mean()), "q25": float(np.quantile(e[m], 0.25)),
                         "q75": float(np.quantile(e[m], 0.75)), "censored": float(never[m].mean())})
        ex[f"T{T}"] = rows
    out["exit"] = ex
    # weird moves (j = 0) where the parent line is well supported
    q = counts[:, 0] / np.maximum(parent[:, 0], 1)
    sup = parent[:, 0] >= 100
    w = pl.DataFrame({"band": band[gi][sup], "ply": ply[sup], "q": q[sup], "ho": ho[gi][sup]}).filter(c("ho"))
    w = w.with_columns(q=c("q").clip(1e-6, 1.0), phase=pl.when(c("ply") < 10).then(pl.lit("00-09"))
                       .when(c("ply") < 20).then(pl.lit("10-19")).otherwise(pl.lit("20+")))
    out["weird"] = {"by_band": q_stats(w, "band"),
                    "by_band_phase": (w.group_by("band", "phase").agg(n=pl.len(), weird_rate=(c("q") < 0.01).mean(),
                                                                      log10_q=c("q").log10().mean())
                                      .sort("band", "phase").to_dicts()),
                    "support_rule": "parent count >= 100"}
    # scaling: exit ply vs corpus size, all heldout bands pooled
    sc = []
    for j in range(LEVELS):
        row = {"level": j, "games": float(n_level[j])}
        for T in (1, 10):
            e, never = exit_ply(counts[:, j], gi, ply, lens, T, cap)
            row[f"T{T}_median"] = float(np.median(e[ho]))
            row[f"T{T}_mean"] = float(e[ho].mean())
            row[f"T{T}_by_band"] = {int(b): float(np.median(e[ho & (band == b)])) for b in sorted(set(band[ho]))}
        sc.append(row)
    for T in (1, 10):
        x = np.log2([r["games"] for r in sc])
        y = np.array([r[f"T{T}_mean"] for r in sc])
        slope, icpt = np.polyfit(x, y, 1)
        out[f"scaling_T{T}"] = {"plies_per_doubling": float(slope),
                                "games_for_mean_exit": {str(t): float(2 ** ((t - icpt) / slope)) for t in (16, 20, 24, 30)}}
    out["scaling"] = sc
    # per-game book features for the upset models
    e1, _ = exit_ply(counts[:, 0], gi, ply, lens, 1, cap)
    e10, _ = exit_ply(counts[:, 0], gi, ply, lens, 10, cap)
    mover = ply % 2
    early = ply < EARLY
    wf = pl.DataFrame({"gi": gi, "mover": mover, "weird": sup & (q < 0.01), "early": early,
                       "lq": np.where(sup, np.log10(np.clip(q, 1e-6, 1)), np.nan)})
    per_side = (wf.filter(c("early")).group_by("gi", "mover")
                .agg(weird_n=c("weird").sum(), book_lq=c("lq").filter(c("lq").is_not_nan()).mean()))
    gf = pl.DataFrame({"gi": np.arange(len(lens)), "exit1": e1, "exit10": e10,
                       "exit10_by": np.where(e10 < lens, e10 % 2, -1)})
    return out, (gf, per_side)


# --- upsets --------------------------------------------------------------------------------------------------------

def _rows(task):
    row = task
    r = game_rows((row, None, None))
    return {k: r[k] for k in ("game_id", "mover", "ply", "think_s", "think_frac", "clock_frac", "material_own",
                              "capture", "gives_check", "castle") if k in r}


def early_t0(g: pl.DataFrame) -> pl.DataFrame:
    mv = pl.read_parquet(OUT / "eval_moves.parquet")
    d = g.select("game_id", "white_elo", "black_elo", "base", "inc").join(mv, on="game_id")
    with Pool(6) as p:
        parts = p.map(_rows, list(d.iter_rows(named=True)), chunksize=64)
    ply = pl.concat([pl.DataFrame(x) for x in parts if len(x["game_id"])], how="diagonal_relaxed")
    ply = ply.with_columns([c(k).fill_nan(None) for k, t in ply.schema.items() if t.is_float()])  # mean() skips null only
    e = ply.filter(c("ply") < EARLY)
    return (e.sort("game_id", "ply").group_by("game_id", "mover")
            .agg(think_frac_mean=c("think_frac").mean(), think_s_median=c("think_s").median(),
                 premove_share=(c("think_s") < 0.5).mean(), clock_frac_30=c("clock_frac").last(),
                 material_30=c("material_own").last(), capture_share=c("capture").mean(),
                 check_share=c("gives_check").mean(), castled=c("castle").max()))


def upsets(g: pl.DataFrame, book_feats) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import log_loss, roc_auc_score
    gf, per_side = book_feats
    g = g.with_row_index("gi").with_columns(c("gi").cast(pl.Int64))
    m = g.filter((c("white_elo") - c("black_elo")).abs() >= 300)
    m = m.with_columns(gap=(c("white_elo") - c("black_elo")).abs(),
                       dog=pl.when(c("white_elo") < c("black_elo")).then(0).otherwise(1),  # underdog colour
                       score_w=pl.when(c("result") == "1-0").then(1.0).when(c("result") == "0-1").then(0.0)
                       .otherwise(0.5))
    m = m.with_columns(dog_score=pl.when(c("dog") == 0).then(c("score_w")).otherwise(1 - c("score_w")),
                       expected=1 / (1 + 10 ** (c("gap") / 400)))
    m = m.with_columns(upset=(c("dog_score") == 1.0), surprise_bits=-(c("expected").log(2)))
    t = time.time()
    t0 = early_t0(m)
    print(f"early T0 features for {m.height:,} mismatch games ({time.time() - t:.0f}s)", flush=True)
    feats_side = ["think_frac_mean", "think_s_median", "premove_share", "clock_frac_30", "material_30",
                  "capture_share", "check_share", "castled"]
    ps = per_side.join(m.select("gi", "game_id"), on="gi")
    both = t0.join(ps.select("game_id", c("mover").cast(t0["mover"].dtype), "weird_n", "book_lq"),
                   on=["game_id", "mover"], how="left")
    sf = feats_side + ["weird_n", "book_lq"]
    dog = m.select("game_id", "dog").join(both, on="game_id")
    d_side = dog.filter(c("mover") == c("dog")).select("game_id", *[c(k).alias(f"dog_{k}") for k in sf])
    f_side = dog.filter(c("mover") != c("dog")).select("game_id", *[c(k).alias(f"fav_{k}") for k in sf])
    m = m.join(d_side, on="game_id", how="left").join(f_side, on="game_id", how="left").join(gf, on="gi", how="left")
    m = m.with_columns(dog_left_book=(c("exit10_by") == c("dog")).cast(pl.Float64),
                       fav_left_book=(c("exit10_by") == (1 - c("dog"))).cast(pl.Float64),
                       dog_material_edge=c("dog_material_30"))
    res = {"games": m.height, "upset_rate": float(m["upset"].mean()),
           "by_gap": (m.with_columns(gb=(c("gap") // 100 * 100).clip(300, 800))
                      .group_by("gb").agg(n=pl.len(), upset=c("upset").mean(), dog_score=c("dog_score").mean(),
                                          elo_expected=c("expected").mean()).sort("gb").to_dicts())}
    cols_show = ["game_id", "white_elo", "black_elo", "base", "inc", "result", "termination", "gap", "surprise_bits"]
    up = m.filter(c("upset"))
    res["dramatic_by_gap"] = up.sort("gap", descending=True).head(15).select(cols_show).to_dicts()
    res["dramatic_by_surprise_normal_end"] = (up.filter(c("termination") == "Normal")
                                              .sort("surprise_bits", descending=True).head(15).select(cols_show)
                                              .to_dicts())
    res["termination_mix"] = (m.group_by("upset", "termination").len().sort("upset", "termination").to_dicts())
    # models
    y = m["upset"].to_numpy().astype(int)
    _, gidx = np.unique(m["game_id"].to_numpy(), return_inverse=True)
    fold = np.random.default_rng(SEED).integers(0, 5, gidx.max() + 1)[gidx]
    base = ["gap", "base", "inc", "dog"]
    early = [f"{p}_{k}" for p in ("dog", "fav") for k in feats_side]
    bookf = [f"{p}_{k}" for p in ("dog", "fav") for k in ("weird_n", "book_lq")] + ["exit1", "exit10",
                                                                                   "dog_left_book", "fav_left_book"]
    sets = {"gap+tc": base, "+book": base + bookf, "+early_T0": base + early, "+early_T0+book": base + early + bookf,
            "+early_T0_no_material": base + [f for f in early if "material" not in f] + bookf}
    res["models"] = {}
    prior = np.empty(len(y))
    for f in range(5):
        prior[fold == f] = y[fold != f].mean()
    ll0 = log_loss(y, prior)
    for name, feats in sets.items():
        X = m.select([c(k).cast(pl.Float64) for k in feats]).to_numpy()
        p = np.empty(len(y))
        for f in range(5):
            clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, min_samples_leaf=50,
                                                 l2_regularization=1.0, early_stopping=True, random_state=SEED)
            p[fold == f] = clf.fit(X[fold != f], y[fold != f]).predict_proba(X[fold == f])[:, 1]
        res["models"][name] = {"auc": float(roc_auc_score(y, p)), "logloss_gain": float(1 - log_loss(y, p) / ll0)}
        print(f"upset {name:24} auc {res['models'][name]['auc']:.3f} ll gain {res['models'][name]['logloss_gain']:+.3f}",
              flush=True)
    # contrasts: upset vs not, within gap bins (mean difference, pooled over bins weighted by upsets)
    con = {}
    for k in early + bookf:
        diffs, ws = [], []
        for lo in range(300, 800, 100):
            b = m.filter(c("gap").is_between(lo, lo + 99))
            u, n = b.filter(c("upset"))[k].cast(pl.Float64), b.filter(~c("upset"))[k].cast(pl.Float64)
            if u.len() >= 30:
                diffs.append(u.mean() - n.mean())
                ws.append(u.len())
        sd = m[k].cast(pl.Float64).std()
        if diffs and sd:
            con[k] = {"diff": float(np.average(diffs, weights=ws)), "diff_sd_units": float(np.average(diffs, weights=ws) / sd)}
    res["contrasts"] = dict(sorted(con.items(), key=lambda kv: -abs(kv[1]["diff_sd_units"])))
    return res


def main():
    t = time.time()
    g, counts, lens, off, meta = load()
    out, bf = book(g, counts, lens, off, meta)
    print(f"book done ({time.time() - t:.0f}s)", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "book_science.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    out["upsets"] = upsets(g, bf)
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / "book_science.json"
    p.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"-> {p} ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()

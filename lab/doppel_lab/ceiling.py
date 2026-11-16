"""How much top-1 could per-player memory buy on Allie-protocol positions? (CPU, ~5 min.)

    python -m doppel_lab.ceiling          # -> lab/results/ceiling.json

2026-10 held-out corpus (26 parts, file and row order = time order within the month dump), position keys from
`poskey annotate` (data/v0/posbook/annot/heldout_cap40: R0 key after each ply, first 40 plies; transpositions
merged). The position before ply k is the key after ply k-1 (k >= 1). Two position sets:
  full    every ply 1..39
  allie   the Allie / Maia-3 test protocol: ply >= 10, game cut at the first ply where the mover has < 30 s
          before moving (clock before = the same side's clock after its previous move, or the base)
For each scored visit (player, position, played move), strictly earlier visits in time order:
  own     the player's own earlier visits to the position: predict their modal earlier move (ties: most recent)
  crowd   every other player's earlier visits: predict the modal earlier move
Reported per set and ply bucket: share of visits with own history, own-mode top-1 on them, crowd-mode top-1 on the
same visits, and the implied top-1 gain of a perfect "own memory, else population" switch:
share x (own top-1 - crowd top-1). Plies past 40 are not keyed; they are counted as no-history (nearly every
position there is new: SCIENCE.md "Per-game uniqueness curves").
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data/corpus/2026-10/heldout"
ANNOT = ROOT / "data/v0/posbook/annot/heldout_cap40"
RESULTS = ROOT / "lab/results/ceiling.json"
CAP = 40
BUCKETS = ((1, 9), (10, 19), (20, 29), (30, 39))
c = pl.col


def load():
    files = sorted(CORPUS.glob("part-*.parquet"))
    g = pl.concat([pl.read_parquet(f, columns=["white", "black", "white_elo", "black_elo", "base", "inc", "moves", "clocks"]) for f in files])
    keys = np.load(ANNOT / "keys.npy")
    lens = np.load(ANNOT / "lens.npy").astype(np.int64)
    assert len(lens) == g.height, (len(lens), g.height)
    return g, keys, lens


def bucket(g: pl.DataFrame) -> np.ndarray:
    """200-Elo bucket of the two players' mean, clipped to 600..2600 (the test set's strata)."""
    m = (g["white_elo"].to_numpy().astype(np.int64) + g["black_elo"].to_numpy()) // 2
    return np.clip(m // 200 * 200, 600, 2600)


def first_under_30(g: pl.DataFrame) -> np.ndarray:
    """Per game, the first ply whose mover has < 30 s before moving (Allie cut); 0 if the base is under 30 s."""
    n = g.height
    ck = g.select("clocks").with_row_index("g").explode("clocks").drop_nulls().with_columns(
        p=pl.int_range(pl.len()).over("g") + 2).filter(c("clocks") < 30)
    fu_ = np.full(n, 10**6, np.int64)
    fu = ck.group_by("g").agg(c("p").min())
    fu_[fu["g"].to_numpy()] = fu["p"].to_numpy()
    return np.where(g["base"].to_numpy() < 30, 0, fu_)


def allie_positions(g: pl.DataFrame) -> np.ndarray:
    glen = g["moves"].list.len().to_numpy().astype(np.int64)
    return np.clip(np.minimum(glen, first_under_30(g)) - 10, 0, None)


def visits(g: pl.DataFrame, keys, lens) -> tuple[pl.DataFrame, int]:
    """One row per (game, ply k in 1..min(len, CAP)-1): order, player, key before ply k, move at k, allie flag.
    Also returns the number of Allie-protocol positions in the whole games (past CAP too)."""
    n = len(lens)
    off = np.r_[0, np.cumsum(lens)]
    gi = np.repeat(np.arange(n), lens)
    j = np.arange(len(gi)) - off[gi]               # key index j = position after ply j
    keep = j < lens[gi] - 1                         # the move at ply j + 1 exists within the cap
    gi, j = gi[keep], j[keep]
    k = j + 1
    key = keys[off[gi] + j]
    # moves and clocks of the first CAP plies as dense (n, CAP) arrays
    mv = g.select(c("moves").list.head(CAP)).with_row_index("g").explode("moves").drop_nulls().with_columns(
        k=pl.int_range(pl.len()).over("g"))
    M = np.full((n, CAP), -1, np.int64)
    M[mv["g"].to_numpy(), mv["k"].to_numpy()] = mv["moves"].cast(pl.Categorical).to_physical().to_numpy()
    move = M[gi, k]
    # Allie cut, on full games: mover's clock before ply p = own clock after ply p-2 (base for p < 2)
    first_under = first_under_30(g)
    glen = g["moves"].list.len().to_numpy().astype(np.int64)
    allie_total = int(np.clip(np.minimum(glen, first_under) - 10, 0, None).sum())
    allie = (k >= 10) & (k < first_under[gi])
    names = pl.concat([g["white"], g["black"]]).str.to_lowercase().cast(pl.Categorical).to_physical().to_numpy()
    white, black = names[:n].astype(np.int64), names[n:].astype(np.int64)
    player = np.where(k % 2 == 0, white[gi], black[gi])
    v = pl.DataFrame({"order": gi * CAP + k, "player": player, "key": key, "move": move, "k": k, "allie": allie,
                      "bucket": bucket(g)[gi]})
    return v, allie_total


def prior_mode_hits(gid: np.ndarray, mv: np.ndarray) -> np.ndarray:
    """Rows sorted by group id then time. For each row: 1 if the modal move among strictly earlier rows of its
    group (ties -> most recent) equals its move, 0 if it differs, -1 if there are no earlier rows."""
    out = np.full(len(mv), -1, np.int8)
    # only groups with >= 2 rows matter
    starts = np.r_[0, np.nonzero(np.diff(gid))[0] + 1]
    sizes = np.diff(np.r_[starts, len(mv)])
    for s, n in zip(starts[sizes >= 2], sizes[sizes >= 2]):
        cnt: dict[int, int] = {}
        best, bc = -1, 0
        for i in range(s, s + n):
            m = mv[i]
            if best >= 0:
                out[i] = 1 if m == best else 0
            x = cnt.get(m, 0) + 1
            cnt[m] = x
            if x >= bc:
                best, bc = m, x
    return out


def main():
    import sys
    g, keys, lens = load()
    if len(sys.argv) > 1:                         # smoke: first N games
        n = int(sys.argv[1])
        g, keys, lens = g.head(n), keys[:int(lens[:n].sum())], lens[:n]
    print(f"{g.height:,} games, {len(keys):,} keyed plies", flush=True)
    v, allie_total = visits(g, keys, lens)
    print(f"{v.height:,} keyed visits, {allie_total:,} Allie-protocol positions", flush=True)
    # own: causal prior mode, only (player, key) groups with >= 2 visits need the loop
    v = v.with_columns(n_own=pl.len().over("player", "key"))
    rep = v.filter(c("n_own") >= 2).sort("player", "key", "order")
    print(f"{rep.height:,} visits in repeated (player, position) groups; looping", flush=True)
    gid = rep.select(pl.struct("player", "key").rank("dense")).to_series().to_numpy()
    rep = rep.with_columns(own=pl.Series(prior_mode_hits(gid, rep["move"].to_numpy())))
    v = v.join(rep.select("order", "own"), on="order", how="left").with_columns(c("own").fill_null(-1))
    # crowd: leave-one-out mode over the whole month's visits to the key (non-causal; a population model that
    # has seen every other game, so it favours the crowd and makes the switch gain conservative)
    kc = v.group_by("key", "move").agg(n=pl.len())
    top = kc.sort("n", descending=True).group_by("key").agg(m1=c("move").first(), c1=c("n").first(),
                                                           c2=c("n").get(1, null_on_oob=True).fill_null(0), tot=c("n").sum())
    v = v.join(top, on="key", how="left").with_columns(
        crowd=pl.when(c("tot") < 2).then(-1).when((c("move") == c("m1")) & (c("c1") - 1 >= c("c2"))).then(1)
        .otherwise(0)).drop("m1", "c1", "c2", "tot", "n_own")
    total_plies = int(g["moves"].list.len().sum()) - g.height      # plies 1.. (ply 0 is never keyed)
    gb = bucket(g)
    tot_b = {"full": pl.DataFrame({"bucket": gb, "n": g["moves"].list.len().to_numpy() - 1}),
             "allie": pl.DataFrame({"bucket": gb, "n": allie_positions(g)})}
    tot_b = {k_: d.group_by("bucket").agg(c("n").sum()) for k_, d in tot_b.items()}
    t = pl.read_parquet(ROOT / "data/allie/test.parquet")
    test_w = (pl.DataFrame({"bucket": bucket(t), "n": allie_positions(t)}).group_by("bucket").agg(c("n").sum())
              .with_columns(w=c("n") / c("n").sum()))
    del g, rep, kc, top, keys
    v = v.select(c("player").cast(pl.Int32), c("k").cast(pl.Int8), "allie", c("own").cast(pl.Int8),
                 c("crowd").cast(pl.Int8), c("bucket").cast(pl.Int16))
    if len(sys.argv) == 1:
        v.write_parquet(ROOT / "data/v0/ceiling_visits.parquet")   # now with bucket
    out = {"games": int(len(lens)), "visits_keyed": v.height, "plies_total": total_plies, "allie_total": allie_total}
    bk = pl.lit(None, pl.Utf8)
    for lo, hi in BUCKETS:
        bk = pl.when((c("k") >= lo) & (c("k") <= hi)).then(pl.lit(f"{lo}-{hi}")).otherwise(bk)
    aggs = dict(visits=pl.len(), hist=(c("own") >= 0).sum(), own_hit=(c("own") == 1).sum(),
                crowd_hit_hist=((c("own") >= 0) & (c("crowd") == 1)).sum(), crowd_cov=(c("crowd") >= 0).sum(),
                crowd_hit=(c("crowd") == 1).sum())
    rng = np.random.default_rng(0)
    for name, sel, tot in (("full", pl.lit(True), total_plies), ("allie", c("allie"), allie_total)):
        s = v.filter(sel)
        by = s.group_by(bk.alias("b")).agg(**aggs).sort("b").to_dicts()
        allr = s.select(**aggs).to_dicts()[0]
        r = {"positions_total": tot, "visits_keyed": allr["visits"],
             "own_history_share_of_total": allr["hist"] / tot,
             "switch_gain_pts_of_total": 100 * (allr["own_hit"] - allr["crowd_hit_hist"]) / tot}
        for x in by + [dict(allr, b="1-39")]:
            h = max(x["hist"], 1)
            r[x["b"]] = {"visits": x["visits"], "own_history_share": x["hist"] / x["visits"],
                         "own_top1": x["own_hit"] / h, "crowd_top1_same_visits": x["crowd_hit_hist"] / h,
                         "crowd_top1_all": x["crowd_hit"] / max(x["crowd_cov"], 1),
                         "crowd_coverage": x["crowd_cov"] / x["visits"],
                         "switch_gain_pts": 100 * (x["own_hit"] - x["crowd_hit_hist"]) / x["visits"]}
        # player bootstrap of the gain over all positions: each player's positions_total is approximated by
        # their keyed visits scaled by tot / keyed (positions past ply 40 are spread pro rata)
        pp = s.group_by("player").agg(n=pl.len(), d=((c("own") == 1).cast(pl.Int32) -
                                                     ((c("own") >= 0) & (c("crowd") == 1)).cast(pl.Int32)).sum())
        n_, d_ = pp["n"].to_numpy().astype(np.float64), pp["d"].to_numpy().astype(np.float64)
        scale = tot / n_.sum()
        bs = []
        for _ in range(500):
            w = np.bincount(rng.integers(0, len(n_), len(n_)), minlength=len(n_))
            bs.append(100 * (w * d_).sum() / ((w * n_).sum() * scale))
        r["switch_gain_pts_of_total_ci95"] = [float(q) for q in np.percentile(bs, [2.5, 97.5])]
        r["players"] = int(len(n_))
        # per 200-Elo bucket, and reweighted to the Allie test set's position mix over buckets
        pb = (s.group_by("bucket").agg(**aggs).join(tot_b[name], on="bucket")
              .with_columns(share=c("hist") / c("n"), gain=100 * (c("own_hit") - c("crowd_hit_hist")) / c("n"),
                            own_top1=c("own_hit") / c("hist"), crowd_top1=c("crowd_hit_hist") / c("hist"),
                            corpus_w=c("n") / c("n").sum())
              .join(test_w.select("bucket", test_w="w"), on="bucket", how="left").sort("bucket"))
        r["by_bucket"] = {int(x["bucket"]): {k_: x[k_] for k_ in ("n", "share", "own_top1", "crowd_top1", "gain",
                                                                 "corpus_w", "test_w")} for x in pb.to_dicts()}
        r["gain_corpus_mix"] = float((pb["gain"] * pb["corpus_w"]).sum())
        r["gain_test_mix"] = float((pb["gain"] * pb["test_w"].fill_null(0)).sum() / pb["test_w"].fill_null(0).sum())
        out[name] = r
        print(name, json.dumps(r, indent=1), flush=True)
    if len(sys.argv) == 1:
        RESULTS.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()

"""Exact-repeat memory on the Allie test set itself, with each player's real earlier games (CPU, ~15 min, < 15 GB).

    python -m doppel_lab.allie_memory        # -> lab/results/allie_memory.json

Test positions: data/allie/test.parquet under the exact Allie / Maia-3 protocol (ply >= 10, game cut at the first ply
whose mover has < 30 s before moving; 884,049 positions), mover named by data/allie/players.parquet. Positions are
keyed by the move prefix (a 64-bit polynomial hash of the moves before ply k, with k), so transpositions are not
merged (a lower bound next to ceiling.py's R0 keys); plies >= CAP are not keyed and count as no history.
  own     the test player's own visits to the same position, as the mover, in their earlier games
          (data/allie/history/*.parquet, game start strictly before the test game's created_at, test games
          dropped by id): predict their modal earlier move (ties: most recent)
  crowd   the base corpus (data/allie/base: rating-stratified 2022 sample of non-test players), all its visits to
          the position: modal move (non-causal, which favours the crowd)
Reported overall, by ply bucket, by rating bucket and by the test game's month (history available grows with it):
share of positions with own history, own top-1 and crowd top-1 on them, crowd coverage, and the top-1 gain of an
"own memory, else population" switch, in points over all 884k positions. Months still downloading are simply absent:
the result is a lower bound until all 14 months are in (`months` in the output lists what was used).
Per-position own predictions are saved (data/allie/memory_positions.parquet) so the gain can later be measured
against Maia-3's actual predictions (top1 --save-positions) instead of the crowd.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from doppel_lab.ceiling import allie_positions, bucket, first_under_30

ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT / "data/allie/test.parquet"
PLAYERS = ROOT / "data/allie/players.parquet"
HIST = ROOT / "data/allie/history"
BASE = ROOT / "data/allie/base"
OUT_POS = ROOT / "data/allie/memory_positions.parquet"
RESULTS = ROOT / "lab/results/allie_memory.json"
CAP = 40
P = np.uint64(0x9E3779B97F4A7C15)
PW = np.ones(CAP, np.uint64)
with np.errstate(over="ignore"):
    for _i in range(1, CAP):
        PW[_i] = PW[_i - 1] * P
BUCKETS = ((10, 19), (20, 29), (30, 39), (40, 9999))
c = pl.col


def prefix_keys(moves: pl.Series) -> pl.DataFrame:
    """Rows (g, k, h, mv): game row g, ply k < min(len, CAP), h = hash of moves[:k], mv = hash of moves[k]."""
    e = (pl.DataFrame({"moves": moves.list.head(CAP)}).with_row_index("g").explode("moves").drop_nulls()
         .with_columns(k=pl.int_range(pl.len()).over("g").cast(pl.Int32), mv=c("moves").hash(seed=7)))
    g = e["g"].to_numpy()
    k = e["k"].to_numpy()
    code = e["mv"].to_numpy().astype(np.uint64)
    with np.errstate(over="ignore"):
        term = code * PW[k]
        cs = np.cumsum(term, dtype=np.uint64) - term          # sum of terms strictly before this row
        start = np.r_[0, np.nonzero(np.diff(g))[0] + 1]
        first = np.repeat(cs[start], np.diff(np.r_[start, len(g)]))
        h = cs - first
    return pl.DataFrame({"g": g, "k": k, "h": h, "mv": code})


def queries() -> tuple[pl.DataFrame, int, pl.DataFrame]:
    t = pl.read_parquet(TEST).drop("white", "black")
    pp = pl.read_parquet(PLAYERS, columns=["game_id", "white", "black", "created_at"])
    t = t.join(pp, on="game_id", how="left")
    assert t["created_at"].null_count() == 0
    npos = allie_positions(t)
    cut = first_under_30(t)
    per_game = pl.DataFrame({"game_id": t["game_id"], "bucket": bucket(t), "npos": npos,
                             "month": pl.from_epoch(t["created_at"], "ms").dt.strftime("%Y-%m")})
    pk = prefix_keys(t["moves"])
    gi = pk["g"].to_numpy()
    k = pk["k"].to_numpy()
    keep = (k >= 10) & (k < cut[gi])
    pk = pk.filter(pl.Series(keep))
    gi = pk["g"].to_numpy()
    white, black = t["white"].to_numpy(), t["black"].to_numpy()
    q = pk.with_columns(player=pl.Series(np.where(pk["k"].to_numpy() % 2 == 0, white[gi], black[gi])),
                        game_id=pl.Series(t["game_id"].to_numpy()[gi]),
                        created_at=pl.Series(t["created_at"].to_numpy()[gi]),
                        bucket=pl.Series(bucket(t)[gi]),
                        month=pl.Series(per_game["month"].to_numpy()[gi])).drop("g").with_row_index("qid")
    return q, int(npos.sum()), per_game


def scan(files, q_keys: pl.DataFrame, by_player: bool, test_ids: pl.Series):
    """Visits in the given parquet files to any query position. by_player: only visits where the mover is a query
    player at that position (joined on player, k, h), with the game's start time."""
    out = []
    for f in files:
        pf = pq.ParquetFile(f)
        for b in pf.iter_batches(batch_size=500_000, columns=["game_id", "white", "black", "moves", "utc_ms"]):
            d = pl.from_arrow(b)
            d = d.filter(~c("game_id").is_in(test_ids.implode()))
            pk = prefix_keys(d["moves"])
            gi = pk["g"].to_numpy()
            if by_player:
                w = d["white"].str.to_lowercase().to_numpy()
                bl = d["black"].str.to_lowercase().to_numpy()
                pk = pk.with_columns(player=pl.Series(np.where(pk["k"].to_numpy() % 2 == 0, w[gi], bl[gi])),
                                     utc_ms=pl.Series(d["utc_ms"].to_numpy()[gi]))
                out.append(pk.join(q_keys, on=["player", "k", "h"], how="semi").drop("g"))
            else:
                out.append(pk.join(q_keys, on=["k", "h"], how="semi").drop("g"))
        print(f"  {f.name}: {sum(o.height for o in out):,} matching visits so far", flush=True)
    return pl.concat(out) if out else None


def main():
    q, total, per_game = queries()
    months = sorted(p.stem for p in HIST.glob("*.parquet"))
    bmonths = sorted(p.stem for p in BASE.glob("*.parquet"))
    print(f"{q.height:,} keyed test positions (of {total:,}); history months {months}; base {bmonths}", flush=True)
    test_ids = pl.read_parquet(TEST, columns=["game_id"])["game_id"]
    own_v = scan(sorted(HIST.glob("*.parquet")), q.select("player", "k", "h").unique(), True, test_ids)
    crowd_v = scan(sorted(BASE.glob("*.parquet")), q.select("k", "h").unique(), False, test_ids)

    # own: modal earlier move per query, strictly before the test game, ties -> most recent
    j = (q.select("qid", "player", "k", "h", "created_at").join(own_v, on=["player", "k", "h"])
         .filter(c("utc_ms") < c("created_at")))
    om = (j.group_by("qid", "mv").agg(n=pl.len(), last=c("utc_ms").max())
          .sort(["qid", "n", "last"], descending=[False, True, True]).group_by("qid", maintain_order=True)
          .agg(own_mv=c("mv").first(), own_n=c("n").sum()))
    cm = (crowd_v.group_by("k", "h", "mv").agg(n=pl.len()).sort(["k", "h", "n"], descending=[False, False, True])
          .group_by("k", "h", maintain_order=True).agg(crowd_mv=c("mv").first(), crowd_n=c("n").sum()))
    r = (q.join(om, on="qid", how="left").join(cm, on=["k", "h"], how="left")
         .with_columns(hist=c("own_mv").is_not_null(), own_hit=c("own_mv") == c("mv"),
                       crowd_hit=(c("crowd_mv") == c("mv")).fill_null(False), crowd_cov=c("crowd_mv").is_not_null()))
    r.select("game_id", "k", "player", "bucket", "month", "hist", "own_n", "own_hit", "crowd_hit", "crowd_cov",
             "crowd_n").write_parquet(OUT_POS)

    def summ(s: pl.DataFrame, tot: int) -> dict:
        h = s.filter(c("hist"))
        nh = max(h.height, 1)
        return {"positions": tot, "keyed": s.height, "own_history_share": h.height / tot,
                "own_top1": h["own_hit"].sum() / nh, "crowd_top1_same": h["crowd_hit"].sum() / nh,
                "crowd_coverage_same": h["crowd_cov"].sum() / nh,
                "gain_pts": 100 * (int(h["own_hit"].sum()) - int(h["crowd_hit"].sum())) / tot}

    out = {"months_history": months, "months_base": bmonths, "cap": CAP, "overall": summ(r, total),
           "by_ply": {}, "by_bucket": {}, "by_month": {}}
    for lo, hi in BUCKETS:
        s = r.filter((c("k") >= lo) & (c("k") <= hi))
        # positions per ply bucket over whole games (keyed or not): count from the cut per game
        out["by_ply"][f"{lo}-{hi}"] = summ(s, max(s.height, 1)) if hi < 9999 else None
    for col, key in (("bucket", "by_bucket"), ("month", "by_month")):
        tots = per_game.group_by(col).agg(c("npos").sum())
        for v, n in sorted(tots.rows()):
            out[key][str(v)] = summ(r.filter(c(col) == v), int(n))
    # how much history each test game's player has: earlier own games in the months used
    out["note"] = "by_ply shares are over keyed positions in that range (all are keyed below CAP)"
    RESULTS.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["overall"], indent=1))
    for key in ("by_ply", "by_bucket", "by_month"):
        for v, x in out[key].items():
            if x:
                print(f"{key:9s} {v:8s} share {x['own_history_share']:.4f}  own {x['own_top1']:.3f}  "
                      f"crowd {x['crowd_top1_same']:.3f}  cov {x['crowd_coverage_same']:.2f}  gain {x['gain_pts']:+.3f}")


if __name__ == "__main__":
    main()

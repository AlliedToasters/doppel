"""Who leaves the book first, and does it pay? (2026-10 held-out corpus, blitz, CPU)

    python -m doppel_lab.novelty

The book is the 2026-10 train table (22.7 M games; held-out games are not in it). Per game, the first ply whose
move-order prefix has < 2 book games; the side that played it is the "leaver". Also whether the leaving move
was a sideline: its parent position has >= 100 book games (a known position, an off-book move), versus a
leave from a position already rare (< 100). Leaver's score minus the Elo expectation, by rating band and
exit-ply bin; 1500-1599 ratings excluded.
Output: lab/results/novelty.json
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl

from doppel_lab.book import move_ids, prefix_hashes
from doppel_lab.persistence import BOOK, CORPUS, RESULTS

c = pl.col


def main():
    g = pl.concat([pl.read_parquet(f, columns=["white_elo", "black_elo", "result", "moves"])
                   for f in sorted(CORPUS.glob("part-*.parquet"))])
    g = g.filter(~c("white_elo").is_between(1500, 1599) & ~c("black_elo").is_between(1500, 1599)
                 & (c("moves").list.len() >= 20))
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
    par = np.empty(len(keys))
    par[1:] = m[:-1]
    par[start] = 22_700_000.0
    has = f != big
    fi = np.where(has, start + np.where(has, f, 0), 0)
    g = g.with_columns(exit=pl.Series(np.where(has, f, -1)), parent=pl.Series(np.where(has, par[fi], np.nan)))
    g = g.filter(c("exit") >= 0)
    sw = pl.when(c("result") == "1-0").then(1.0).when(c("result") == "0-1").then(0.0).otherwise(0.5)
    g = g.with_columns(leaver_white=c("exit") % 2 == 0, sw=sw,
                       ew=1 / (1 + 10 ** ((c("black_elo") - c("white_elo")) / 400)))
    g = g.with_columns(r=pl.when(c("leaver_white")).then(c("sw") - c("ew")).otherwise(c("ew") - c("sw")),
                       elo=pl.when(c("leaver_white")).then(c("white_elo")).otherwise(c("black_elo")),
                       side=pl.when(c("leaver_white")).then(pl.lit("white")).otherwise(pl.lit("black")),
                       known=c("parent") >= 100)
    # White's general edge would show as a leaver effect; remove each colour's mean residual
    wres = float((g["sw"] - g["ew"]).mean())
    g = g.with_columns(r=c("r") - pl.when(c("leaver_white")).then(pl.lit(wres)).otherwise(pl.lit(-wres)))
    g = g.with_columns(band=(c("elo") // 400 * 400).clip(800, 2400),
                       eb=pl.when(c("exit") < 8).then(pl.lit("a <8")).when(c("exit") < 14).then(pl.lit("b 8-13"))
                       .when(c("exit") < 20).then(pl.lit("c 14-19")).otherwise(pl.lit("d 20+")))
    t = g.group_by("band", "known").agg(n=pl.len(), r=c("r").mean(), se=c("r").std() / pl.len().sqrt()).sort("band", "known")
    t2 = g.group_by("eb", "known").agg(n=pl.len(), r=c("r").mean(), se=c("r").std() / pl.len().sqrt()).sort("eb", "known")
    with pl.Config(float_precision=4, tbl_rows=30):
        print(f"{g.height:,} games; white edge {wres:+.4f} removed"); print(t); print(t2)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "novelty.json").write_text(json.dumps({"games": g.height, "by_band": t.to_dicts(), "by_exit": t2.to_dicts()},
                                                     indent=1) + "\n")


if __name__ == "__main__":
    main()

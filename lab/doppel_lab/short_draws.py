"""Short draws over the board: by rating, and in the last round (TWIC classical OTB, CPU).

    python -m doppel_lab.short_draws

Short draw = 1/2-1/2 in <= 40 plies (20 moves). Classical OTB games from twic_science.load(), both ratings known.
By mean rating (100-point bins) and by round position within the event (rounds parsed from the Round header's
leading integer; last = the event's highest round, events with >= 5 rounds, pairs within 200 points so Swiss round 1's
mismatches do not drive it): first, middle, last. Also the rating
gap: do evenly matched pairs agree to draws more?
Output: lab/results/short_draws.json
"""

from __future__ import annotations

import json

import polars as pl

from doppel_lab.twic_science import load, twic_files

c = pl.col


def main():
    g = load().filter((c("medium") == "otb") & (c("speed") == "classical") & c("white_elo").is_not_null()
                      & c("black_elo").is_not_null())
    r = pl.read_parquet(twic_files(), columns=["game_id", "round", "event", "site", "date"]).unique("game_id")
    g = g.join(r.select("game_id", "round", ev2="event", site2="site", d2="date"), on="game_id", how="left")
    g = g.with_columns(rd=c("round").str.extract(r"^(\d+)").cast(pl.Int32, strict=False),
                       ekey=c("ev2").fill_null("") + "|" + c("site2").fill_null("") + "|" + c("d2").fill_null("").str.slice(0, 4))
    g = g.with_columns(maxrd=c("rd").max().over("ekey"),
                       short=((c("result") == "1/2-1/2") & (c("plies") <= 40)).cast(pl.Float64),
                       draw=(c("result") == "1/2-1/2").cast(pl.Float64),
                       elo=(c("white_elo") + c("black_elo")) / 2, gap=(c("white_elo") - c("black_elo")).abs())
    by_elo = g.group_by(band=(c("elo") // 100 * 100).clip(1400, 2700)).agg(
        n=pl.len(), short=c("short").mean(), draw=c("draw").mean()).sort("band")
    x = g.filter((c("maxrd") >= 5) & c("rd").is_not_null() & (c("gap") < 200))   # gap < 200: Swiss round 1 pairs far apart
    x = x.with_columns(pos=pl.when(c("rd") == 1).then(pl.lit("a first")).when(c("rd") == c("maxrd")).then(pl.lit("c last"))
                       .otherwise(pl.lit("b middle")),
                       eb=pl.when(c("elo") >= 2500).then(pl.lit("2500+")).when(c("elo") >= 2200).then(pl.lit("2200-2499"))
                       .otherwise(pl.lit("<2200")))
    by_round = x.group_by("eb", "pos").agg(n=pl.len(), short=c("short").mean(), draw=c("draw").mean()).sort("eb", "pos")
    by_gap = g.group_by(gb=(c("gap") // 100 * 100).clip(0, 400)).agg(n=pl.len(), short=c("short").mean()).sort("gb")
    with pl.Config(tbl_rows=40, float_precision=4):
        print(by_elo); print(by_round); print(by_gap)
    (__import__("pathlib").Path(__file__).resolve().parents[2] / "lab" / "results" / "short_draws.json").write_text(
        json.dumps({"by_elo": by_elo.to_dicts(), "by_round": by_round.to_dicts(), "by_gap": by_gap.to_dicts()}, indent=1) + "\n")


if __name__ == "__main__":
    main()

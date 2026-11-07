"""Are big-gap upsets mis-rated accounts? Follow each side's rating through the rest of the month (CPU).

    python -m doppel_lab.misrating

Uses the 2026-10 heldout corpus shards in stream (≈ chronological) order. A held-out player's games are all in
heldout/, so only held-out sides are followed (the other side's games are split across train/ and heldout/).
For each mismatch game (gap >= 300) and each held-out side: games before / after in the month, rating change
from the side's first game to this one (past drift) and from this one to +10, +50 games and the month's last
game (future drift). Grouped by role (underdog / favourite) x outcome (won / drew / lost), plus a gap-matched
baseline of same-gap games.
Output: lab/results/misrating.json
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "corpus" / "2026-10" / "heldout"
BOOK = ROOT / "data" / "v0" / "book"
RESULTS = ROOT / "lab" / "results"


def main():
    files = sorted(CORPUS.glob("part-*.parquet"))
    g = pl.concat([pl.read_parquet(f, columns=["game_id", "white", "black", "white_elo", "black_elo", "result"])
                   for f in files]).with_row_index("t")
    from doppel_lab.split import held_out
    sides = pl.concat([
        g.select("t", "game_id", player=pl.col("white").str.to_lowercase(), elo="white_elo", opp="black_elo",
                 score=pl.col("result").replace_strict({"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}, default=None)),
        g.select("t", "game_id", player=pl.col("black").str.to_lowercase(), elo="black_elo", opp="white_elo",
                 score=pl.col("result").replace_strict({"1-0": 0.0, "0-1": 1.0, "1/2-1/2": 0.5}, default=None)),
    ])
    names = sides["player"].unique()
    ho = pl.DataFrame({"player": names, "ho": [held_out(n) for n in names.to_list()]})
    sides = sides.join(ho, on="player").filter(pl.col("ho")).drop("ho").sort("player", "t")
    sides = sides.with_columns(
        k=pl.int_range(pl.len()).over("player"), n=pl.len().over("player"),
        first_elo=pl.col("elo").first().over("player"), last_elo=pl.col("elo").last().over("player"),
        elo_p10=pl.col("elo").shift(-10).over("player"), elo_next=pl.col("elo").shift(-1).over("player"), elo_p50=pl.col("elo").shift(-50).over("player"))
    sides = sides.with_columns(past=pl.col("elo") - pl.col("first_elo"), fut_end=pl.col("last_elo") - pl.col("elo"),
                               fut10=pl.col("elo_p10") - pl.col("elo"), this_game=pl.col("elo_next") - pl.col("elo"),
                               fut_after=pl.col("last_elo") - pl.col("elo_next"), fut50=pl.col("elo_p50") - pl.col("elo"),
                               after=pl.col("n") - pl.col("k") - 1, gap=(pl.col("elo") - pl.col("opp")))
    mm = set(pl.read_parquet(BOOK / "eval_games.parquet").filter(pl.col("src") == "mismatch")["game_id"].to_list())
    s = sides.filter(pl.col("gap").abs() >= 300).with_columns(
        mismatch_set=pl.col("game_id").is_in(list(mm)),
        role=pl.when(pl.col("gap") < 0).then(pl.lit("underdog")).otherwise(pl.lit("favourite")),
        outcome=pl.when(pl.col("score") == 1).then(pl.lit("won")).when(pl.col("score") == 0).then(pl.lit("lost"))
        .otherwise(pl.lit("drew")), gb=(pl.col("gap").abs() // 100 * 100).clip(300, 800))
    agg = [pl.len().alias("sides"), pl.col("k").median().alias("games_before_med"),
           pl.col("after").median().alias("games_after_med"), pl.col("past").mean().alias("past_drift"),
           pl.col("fut10").mean().alias("fut10"), pl.col("fut50").mean().alias("fut50"),
           pl.col("fut_end").mean().alias("fut_end"), pl.col("this_game").mean().alias("this_game_delta"),
           pl.col("fut_after").mean().alias("drift_after_this_game"), pl.col("fut_end").median().alias("fut_end_med"),
           (pl.col("k") < 10).mean().alias("within_first10_games"),
           (pl.col("fut_end") >= 200).mean().alias("rises_200plus"),
           (pl.col("fut_end") <= -200).mean().alias("falls_200plus")]
    tab = s.group_by("role", "outcome").agg(agg).sort("role", "outcome")
    by_gap = s.group_by("role", "outcome", "gb").agg(agg).sort("role", "outcome", "gb")
    print(f"{s.height:,} held-out sides in gap>=300 games (of {sides.height:,} held-out sides)")
    with pl.Config(tbl_cols=20, tbl_width_chars=250, float_precision=2):
        print(tab)
        print(by_gap.filter(pl.col("gb").is_in([300, 500, 700])))
    # all-games baseline drift
    base = sides.select(pl.col("fut_end").mean().alias("fut_end_all"), pl.col("past").mean().alias("past_all"),
                        (pl.col("k") < 10).mean().alias("within_first10_all")).to_dicts()[0]
    print("all held-out sides:", base)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "misrating.json").write_text(json.dumps(
        {"sides_gap300": s.height, "baseline": base, "by_role_outcome": tab.to_dicts(),
         "by_role_outcome_gap": by_gap.to_dicts()}, indent=1) + "\n")



def uncertainty():
    """Does a pre-game proxy of rating uncertainty (|rating change| over the side's previous game) predict upsets?"""
    files = sorted(CORPUS.glob("part-*.parquet"))
    g = pl.concat([pl.read_parquet(f, columns=["white", "black", "white_elo", "black_elo", "result"]) for f in files]
                  ).with_row_index("t")
    from doppel_lab.split import held_out
    sides = pl.concat([
        g.select("t", player=pl.col("white").str.to_lowercase(), elo="white_elo", opp="black_elo",
                 score=pl.col("result").replace_strict({"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}, default=None)),
        g.select("t", player=pl.col("black").str.to_lowercase(), elo="black_elo", opp="white_elo",
                 score=pl.col("result").replace_strict({"1-0": 0.0, "0-1": 1.0, "1/2-1/2": 0.5}, default=None))])
    names = sides["player"].unique()
    ho = pl.DataFrame({"player": names, "ho": [held_out(n) for n in names.to_list()]})
    s = sides.join(ho, on="player").filter(pl.col("ho")).sort("player", "t").with_columns(
        prev_swing=(pl.col("elo") - pl.col("elo").shift(1).over("player")).abs(),
        k=pl.int_range(pl.len()).over("player"))
    fav = s.filter((pl.col("elo") - pl.col("opp") >= 300) & pl.col("prev_swing").is_not_null()).with_columns(
        upset=pl.col("score") == 0, exp_upset=1 / (1 + 10 ** ((pl.col("elo") - pl.col("opp")) / 400)),
        sw=pl.col("prev_swing").cut([10, 20, 40, 80], labels=["<10", "10-20", "20-40", "40-80", "80+"]))
    t = fav.group_by("sw").agg(n=pl.len(), upset_rate=pl.col("upset").mean(), elo_expected=pl.col("exp_upset").mean(),
                               gap_mean=(pl.col("elo") - pl.col("opp")).mean()).sort("sw")
    # Same question for all games: fraction of the population in each swing bin.
    pop = s.filter(pl.col("prev_swing").is_not_null()).with_columns(
        sw=pl.col("prev_swing").cut([10, 20, 40, 80], labels=["<10", "10-20", "20-40", "40-80", "80+"])
    ).group_by("sw").agg(share_all_sides=pl.len()).with_columns(pl.col("share_all_sides") / pl.col("share_all_sides").sum())
    t = t.join(pop, on="sw").sort("sw")
    with pl.Config(tbl_cols=10, float_precision=3):
        print("favourites (gap>=300) by |rating change| in their previous game:")
        print(t)
    out = json.loads((RESULTS / "misrating.json").read_text())
    out["favourite_upset_by_prev_swing"] = t.with_columns(pl.col("sw").cast(pl.String)).to_dicts()
    (RESULTS / "misrating.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    import sys
    main() if len(sys.argv) < 2 else {"uncertainty": uncertainty}[sys.argv[1]]()

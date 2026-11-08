"""Transposition map: which positions are reached by many move orders, from which openings, and who takes odd routes.

    python -m doppel_lab.posbook_transpose

Data: the 2026-10 heldout corpus (12.7 M games, first 40 plies, keys from poskey annotate), against the train
books: R0 = position key (data/v0/posbook/full_p001), R1 = move-order prefix (data/v0/book).

1. Routes. In the first 3 heldout files (~1.5 M games), every (position, prefix) pair at plies 1-30. For positions
   with >= 500 visits: distinct routes, the share of visits not by the modal route ("transposed share") and the
   top routes as SAN. Named by the Lichess opening table (CC0) where the position itself is named.
2. R1 residual by opening: a ply's decision position is common (R0 >= 100) but its prefix is rare (R1 < 2), as in
   posbook.py. Share of common-position plies, by the game's opening family (last named position by ply 40).
3. Players: per-player residual rate over all their heldout games (all time controls; plies < 40). Split-half
   reliability over alternate games (in stream order), Spearman-Brown corrected, raw and after regressing out the
   mean rating, for players with >= 40 games.
Output: lab/results/posbook_explore.json, section "transpositions".
"""

from __future__ import annotations

import chess
import numpy as np
import polars as pl

from doppel_lab.posbook_common import (CORPUS, R0_DIR, R1_DIR, annotate, decision_counts, load_table, lookup,
                                       name_games, offsets, opening_keys, ply_index, r1_keys, save_section)

c = pl.col
COMMON, RARE = 100, 2
MAX_PLY = 40


def san_line(uci: list[str]) -> str:
    b = chess.Board()
    out = []
    for i, u in enumerate(uci):
        m = chess.Move.from_uci(u)
        out.append((f"{i // 2 + 1}. " if i % 2 == 0 else "") + b.san(m))
        b.push(m)
    return " ".join(out)


def main():
    files = sorted((CORPUS / "heldout").glob("*.parquet"))
    keys_all, lens_all = annotate("heldout_cap40", files, cap=40)
    rows = [pl.scan_parquet(f).select(pl.len()).collect().item() for f in files]
    goff = np.r_[0, np.cumsum(rows)]
    koff = offsets(lens_all)
    t0k, t0c = load_table(R0_DIR)
    t1k, t1c = load_table(R1_DIR)
    op = opening_keys()
    per_game, route_parts = [], []
    for fi, f in enumerate(files):
        g = pl.read_parquet(f, columns=["game_id", "white", "black", "white_elo", "black_elo",
                                        "moves"]).with_columns(c("moves").list.head(MAX_PLY))
        lens = lens_all[goff[fi]:goff[fi + 1]]
        keys = np.asarray(keys_all[koff[goff[fi]]:koff[goff[fi + 1]]])
        pk, plen = r1_keys(g["moves"], cap=MAX_PLY)
        assert (plen == lens).all()
        gi, ply = ply_index(lens)
        d0 = decision_counts(lookup(t0k, t0c, keys), lens)
        d1 = decision_counts(lookup(t1k, t1c, pk), lens)
        common = (d0 >= COMMON) & (ply >= 1)
        resid = common & (d1 < RARE)
        nm = name_games(keys, lens, op, max_ply=MAX_PLY)
        per_game.append(pl.DataFrame({
            "t": np.arange(goff[fi], goff[fi + 1]), "white": g["white"].str.to_lowercase(),
            "black": g["black"].str.to_lowercase(), "white_elo": g["white_elo"], "black_elo": g["black_elo"],
            "family": nm["family"], "eco": nm["eco"],
            # per side (mover parity): common-position plies and residual plies
            "cw": np.bincount(gi, common & (ply % 2 == 0), len(lens)).astype(np.int32),
            "rw": np.bincount(gi, resid & (ply % 2 == 0), len(lens)).astype(np.int32),
            "cb": np.bincount(gi, common & (ply % 2 == 1), len(lens)).astype(np.int32),
            "rb": np.bincount(gi, resid & (ply % 2 == 1), len(lens)).astype(np.int32)}))
        if fi < 3:  # route sample: (position after ply k, prefix after ply k), k < 30
            m = ply < 30
            route_parts.append(pl.DataFrame({"pos": keys[m], "pre": pk[m], "ply": ply[m].astype(np.int16),
                                             "g": (gi[m] + goff[fi]).astype(np.int64)}))
        print(f"{f.name}: {len(lens):,} games, residual share of common plies {resid.sum() / max(common.sum(), 1):.4f}",
              flush=True)
    pg = pl.concat(per_game)
    res = {"heldout_games": pg.height, "rule": f"common: decision position R0 >= {COMMON}; residual: and R1 < {RARE}"}
    res["routes"] = routes(pl.concat(route_parts), op, files[:3])
    res["residual_by_opening"] = by_opening(pg)
    res["players"] = players(pg)
    save_section("transpositions", res)


def routes(r: pl.DataFrame, op: pl.DataFrame, files, min_visits=500, top=25):
    pr = r.group_by("pos", "pre").agg(n=pl.len(), ply=c("ply").first(), g=c("g").first())
    pos = pr.group_by("pos").agg(visits=c("n").sum(), routes=pl.len(), modal=c("n").max(),
                                 ply=c("ply").first(),
                                 top=pl.struct("n", "g").sort_by("n", descending=True).head(3))
    pos = pos.filter(c("visits") >= min_visits).with_columns(transposed=c("visits") - c("modal"),
                                                             tshare=1 - c("modal") / c("visits"))
    out = {"sample_games": int(r["g"].n_unique()), "positions_min_visits": min_visits, "n_positions": pos.height,
           "visit_weighted_transposed_share": float(pos["transposed"].sum() / pos["visits"].sum()),
           "transposed_share_by_ply": pos.group_by("ply").agg(
               s=c("transposed").sum() / c("visits").sum(), n=pl.len()).sort("ply").rows()}
    moves = pl.concat([pl.read_parquet(f, columns=["moves"]) for f in files])["moves"]
    name = dict(zip(op["key"].to_numpy().tolist(), op["name"].to_list()))

    def desc(row):
        rts = [{"share": t["n"] / row["visits"], "line": san_line(moves[int(t["g"])].to_list()[: row["ply"] + 1])}
               for t in row["top"]]
        return {"opening": name.get(int(row["pos"])), "ply": int(row["ply"]) + 1, "visits": int(row["visits"]),
                "routes": int(row["routes"]), "transposed_share": float(row["tshare"]), "top_routes": rts}
    out["most_transposed_visits"] = [desc(x) for x in pos.sort("transposed", descending=True).head(top).to_dicts()]
    out["highest_transposed_share_1k"] = [desc(x) for x in pos.filter(c("visits") >= 1000)
                                          .sort("tshare", descending=True).head(15).to_dicts()]
    return out


def by_opening(pg: pl.DataFrame, min_games=20_000):
    x = pg.with_columns(common=c("cw") + c("cb"), resid=c("rw") + c("rb"))
    fam = x.group_by("family").agg(games=pl.len(), common=c("common").sum(), resid=c("resid").sum(),
                                   any_resid=(c("resid") > 0).mean())
    fam = fam.with_columns(share=c("resid") / c("common")).filter(c("games") >= min_games).sort("share", descending=True)
    eco = x.with_columns(letter=c("eco").str.slice(0, 1)).group_by("letter").agg(
        games=pl.len(), share=c("resid").sum() / c("common").sum()).sort("letter")
    return {"overall_share_of_common_plies": float(x["resid"].sum() / x["common"].sum()),
            "games_with_any": float((x["resid"] > 0).mean()),
            "by_family": [{k: (float(v) if isinstance(v, float) else v) for k, v in r.items()} for r in fam.to_dicts()],
            "by_eco_letter": eco.to_dicts(), "min_games": min_games}


def players(pg: pl.DataFrame, min_games=40):
    side = pl.concat([
        pg.select("t", player="white", elo="white_elo", common="cw", resid="rw"),
        pg.select("t", player="black", elo="black_elo", common="cb", resid="rb")]).sort("t")
    side = side.with_columns(k=pl.int_range(pl.len()).over("player"), n=pl.len().over("player"))
    side = side.filter(c("n") >= min_games)
    agg = side.group_by("player").agg(
        elo=c("elo").mean(), games=pl.len(),
        a=c("resid").filter(c("k") % 2 == 0).sum() / c("common").filter(c("k") % 2 == 0).sum(),
        b=c("resid").filter(c("k") % 2 == 1).sum() / c("common").filter(c("k") % 2 == 1).sum(),
        rate=c("resid").sum() / c("common").sum()).drop_nulls().filter(c("a").is_finite() & c("b").is_finite())
    a, b, e = agg["a"].to_numpy(), agg["b"].to_numpy(), agg["elo"].to_numpy()

    def sb(x, y):
        r = float(np.corrcoef(x, y)[0, 1])
        return {"half_r": r, "spearman_brown": 2 * r / (1 + r)}

    def resid(x):
        Z = np.c_[np.ones(len(e)), e, e ** 2]
        return x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    rate = agg["rate"].to_numpy()
    q = np.quantile(rate, [0.1, 0.5, 0.9])
    elo_r = float(np.corrcoef(rate, e)[0, 1])
    return {"players": agg.height, "min_games": min_games, "raw": sb(a, b), "rating_residualised": sb(resid(a), resid(b)),
            "rate_p10_p50_p90": q.tolist(), "corr_rate_rating": elo_r,
            "by_rating": agg.with_columns(band=(c("elo") // 200 * 200).clip(800, 2400)).group_by("band")
            .agg(players=pl.len(), median_rate=c("rate").median()).sort("band").to_dicts()}


if __name__ == "__main__":
    main()

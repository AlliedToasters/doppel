"""Chart-ready series from the position-book explorations (lab/results/posbook_explore.json -> "charts").

    python -m doppel_lab.posbook_charts

Each chart: title, x (values + label + unit), series ({name: y values}), y label and unit, source section.
Nulls mark buckets with < 200 moves.
"""

from __future__ import annotations

import json

from doppel_lab.posbook_common import BUCKET_LABELS, BUCKET_X, RESULTS, save_section


def main():
    r = json.loads(RESULTS.read_text())
    charts = {}
    b = r.get("behaviour_1500", {}).get("train")
    if b:
        raw = b["raw_curves_plies_8_20"]
        keep = [i for i in range(len(BUCKET_LABELS))
                if raw["think_s"]["R0"][i] is not None and raw["think_s"]["R1"][i] is not None]
        x = [BUCKET_X[i] for i in keep]
        labels = [BUCKET_LABELS[i] for i in keep]
        charts["think_vs_popularity"] = {
            "title": "Think time by how common the position is (1500 band, plies 8-20)",
            "x": {"values": x, "labels": labels, "label": "train games reaching the position", "unit": "games",
                  "scale": "log"},
            "series": {"R0 position key": [raw["think_s"]["R0"][i] for i in keep],
                       "R1 move-order key": [raw["think_s"]["R1"][i] for i in keep]},
            "y": {"label": "mean think time", "unit": "s"}, "source": "behaviour_1500.train"}
        charts["blunders_vs_popularity"] = {
            "title": "Blunder rate by how common the position is (1500 band, plies 8-20)",
            "x": charts["think_vs_popularity"]["x"],
            "series": {"R0 position key": [100 * raw["blunder"]["R0"][i] for i in keep],
                       "R1 move-order key": [100 * raw["blunder"]["R1"][i] for i in keep]},
            "y": {"label": "blunders (win-chance loss >= 0.3)", "unit": "% of moves"}, "source": "behaviour_1500.train"}
        ev = b["event_study"]
        rel = ev["same_exit_rel_exit"]["rel"]
        sel = [i for i, v in enumerate(rel) if v >= -4]
        def pick(name):
            d = dict(zip(ev[name]["rel"], ev[name]["think_s"]))
            return [d.get(rel[i]) for i in sel]
        charts["think_around_book_exit"] = {
            "title": "Think time around the book exit (1500 band); -1 is the move into the first unseen position",
            "x": {"values": [rel[i] for i in sel], "label": "ply relative to the exit", "unit": "plies"},
            "series": {"keys agree": pick("same_exit_rel_exit"),
                       "transposed game, relative to the move-order (R1) exit": pick("transposed_rel_R1_exit"),
                       "transposed game, relative to the position (R0) exit": pick("transposed_rel_R0_exit")},
            "y": {"label": "mean think time", "unit": "s"}, "source": "behaviour_1500.train.event_study"}
    nf = r.get("noise_floor_by_popularity")
    if nf:
        rows = [x for x in nf["rows"] if x["pairs"] >= 1000]
        charts["noise_floor_vs_popularity"] = {
            "title": "Same player, same position: how often the move repeats (heldout 3+0, plies 6-29)",
            "x": {"values": [BUCKET_X[BUCKET_LABELS.index(x["bucket"])] for x in rows],
                  "labels": [x["bucket"] for x in rows], "label": "train games reaching the position (R0)",
                  "unit": "games", "scale": "log"},
            "series": {"same player": [100 * x["self_agreement"] for x in rows],
                       "two different players": [100 * (x["population_agreement"] or 0) for x in rows]},
            "y": {"label": "chance two visits get the same move", "unit": "%"}, "source": "noise_floor_by_popularity"}
    pb = r.get("personal_books")
    if pb:
        ks = sorted(pb["coverage"]["by_k"], key=int)
        cv = pb["coverage"]["by_k"]
        charts["personal_book_share_by_k"] = {
            "title": "Share of a later game's own decisions already in the player's own book (3+0, plies 1-39)",
            "x": {"values": [int(k) for k in ks], "label": "earlier games in the personal book", "unit": "games",
                  "scale": "log"},
            "series": {"personal book": [100 * cv[k]["personal_share"] for k in ks],
                       "band book": [100 * cv[k]["band_share"] for k in ks],
                       "personal only (not in band book)": [100 * cv[k]["personal_only_share"] for k in ks]},
            "y": {"label": "own decision positions", "unit": "%"}, "source": "personal_books.coverage"}
        rk = pb["reid"]["ks"]
        ks = sorted(rk, key=int)
        charts["reid_by_k"] = {
            "title": "Picking a player out of 1,000 from k games (gallery: their last 40)",
            "x": {"values": [int(k) for k in ks], "label": "probe games", "unit": "games", "scale": "log"},
            "series": {n: [100 * rk[k][n]["top1"] for k in ks] for n in ("surface", "rarity", "divergence", "surface+rarity")},
            "y": {"label": "top-1", "unit": "%"}, "source": "personal_books.reid"}
    ot = r.get("online_vs_otb")
    if ot:
        bands = [b for b in sorted(ot["bands"], key=int)
                 if "lichess_in_lichess_band_book" in ot["bands"][b] and "otb_in_otb_band_book" in ot["bands"][b]]
        charts["online_vs_otb_depth_by_rating"] = {
            "title": "Book depth in one's own medium and rating band, at equal book size (R0)",
            "x": {"values": [int(b) + 100 for b in bands], "label": "rating band (Lichess blitz / FIDE Elo)",
                  "unit": "rating"},
            "series": {"Lichess games, Lichess band book": [ot["bands"][b]["lichess_in_lichess_band_book"]["mean"]
                                                            for b in bands],
                       "OTB games, OTB band book": [ot["bands"][b]["otb_in_otb_band_book"]["mean"] for b in bands],
                       "Lichess games, OTB band book": [ot["bands"][b]["lichess_in_otb_band_book"]["mean"]
                                                        for b in bands]},
            "y": {"label": "mean book exit", "unit": "ply"}, "source": "online_vs_otb.bands",
            "note": "equal N per band = the smaller of the two band books; OTB data: Lumbra's Gigabase OTB "
                    "(CC BY-NC 4.0), aggregates only"}
        dec = sorted(ot["decades"], key=int)
        charts["otb_depth_by_decade"] = {
            "title": "OTB book depth by decade at 200k games, and Lichess 2026 games in each decade's OTB book",
            "x": {"values": [int(d) for d in dec], "label": "decade", "unit": "year"},
            "series": {"OTB games, own decade's book": [ot["decades"][d]["otb_in_own_decade_book"]["mean"]
                                                        for d in dec],
                       "Lichess 2026 games, that decade's OTB book": [ot["decades"][d]["lichess_in_decade_book"]["mean"]
                                                                      for d in dec]},
            "y": {"label": "mean book exit", "unit": "ply"}, "source": "online_vs_otb.decades",
            "note": "OTB data: Lumbra's Gigabase OTB (CC BY-NC 4.0), aggregates only; pre-1990 values are upper bounds "
                    "(residual near-duplicates)"}
    save_section("charts", charts)
    print(list(charts))


if __name__ == "__main__":
    main()

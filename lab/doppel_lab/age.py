"""Age at the board, at matched FIDE rating: TWIC classical OTB x the FIDE rating list's birth years (CPU).

    python -m doppel_lab.age          # needs data/raw/fide/players_list_foa.txt (ratings.fide.com/download)

Aggregates only: no per-player output, and the birth-year join stays in memory (data/ is not in git). Minors are
in the data; the study reports age-bin means and nothing that identifies a player.
Per side in classical OTB games (all event types): age = game year - birth year. Each age bin is compared with
players aged 25-34 of the same FIDE rating (25-point bins), opponent rating (50-point bins) and year, by
coarsened exact matching: score minus Elo expectation (a rating that lags true strength shows here), draw
rate, game length, book exit (T1, pooled TWIC book, self-excluded) and the side's own weird-move rate.
Output: lab/results/age.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from doppel_lab.twic_science import book_feats, load, twic_files

ROOT = Path(__file__).resolve().parents[2]
FIDE = ROOT / "data" / "raw" / "fide" / "players_list_foa.txt"
LAKE = ROOT / "data" / "lake" / "twic"
RESULTS = ROOT / "lab" / "results"
BINS = [(6, 12, "6-11"), (12, 15, "12-14"), (15, 18, "15-17"), (18, 25, "18-24"), (25, 35, "25-34"),
        (35, 50, "35-49"), (50, 65, "50-64"), (65, 100, "65+")]
REF = "25-34"
c = pl.col


def players() -> pl.DataFrame:
    with open(FIDE, encoding="latin-1") as f:
        head = f.readline()
        i_id, i_bd = head.index("ID Number"), head.index("B-day")
        ids, by = [], []
        for line in f:
            try:
                ids.append(int(line[i_id:i_id + 15].strip()))
                y = line[i_bd:i_bd + 5].strip()
                by.append(int(y) if y.isdigit() and int(y) > 1900 else None)
            except ValueError:
                continue
    return pl.DataFrame({"fide_id": ids, "birth": by}, schema={"fide_id": pl.Int64, "birth": pl.Int32})


def main():
    p = players().filter(c("birth").is_not_null())
    print(f"{p.height:,} FIDE players with a birth year", flush=True)
    ids = pl.read_parquet(twic_files(), columns=["game_id", "white_fide_id", "black_fide_id"]).unique("game_id")
    g = load().join(ids, on="game_id").filter((c("medium") == "otb") & (c("speed") == "classical")
                                              & c("white_elo").is_not_null() & c("black_elo").is_not_null())
    g = book_feats(g)
    # per-side weird moves need the side split; book_feats counts both sides, so recompute per side cheaply:
    sides = []
    for col, opp, sgn in (("white", "black", 1), ("black", "white", -1)):
        sides.append(g.select("year", "plies", "t1", "result", fide_id=c(f"{col}_fide_id"), me=c(f"{col}_elo"),
                              op=c(f"{opp}_elo"), score=c("score_w") if sgn == 1 else 1 - c("score_w")))
    s = pl.concat(sides).join(p, on="fide_id").with_columns(age=c("year") - c("birth"))
    lab = pl.lit(None, pl.String)
    for lo, hi, name in BINS:
        lab = pl.when(c("age").is_between(lo, hi - 1)).then(pl.lit(name)).otherwise(lab)
    s = s.with_columns(age_bin=lab).filter(c("age_bin").is_not_null())
    s = s.with_columns(exp=1 / (1 + 10 ** ((c("op") - c("me")) / 400)), mb=c("me") // 25, ob=c("op") // 50,
                       draw_f=(c("result") == "1/2-1/2").cast(pl.Float64), plies=c("plies").cast(pl.Float64))
    s = s.with_columns(score_minus_exp=c("score") - c("exp"))
    print(s.group_by("age_bin").agg(sides=pl.len(), elo=c("me").median()).sort("age_bin"), flush=True)
    res = {"sides_with_age": s.height, "reference": REF, "bins": {}}
    for _, _, name in BINS:
        if name == REF:
            continue
        d = s.filter(c("age_bin").is_in([name, REF])).with_columns(tr=c("age_bin") == name)
        r = {}
        for f in ["score_minus_exp", "draw_f", "plies", "t1"]:
            gg = d.group_by("year", "mb", "ob", "tr").agg(m=c(f).mean(), v=c(f).var(), n=pl.len())
            j = gg.filter(c("tr")).join(gg.filter(~c("tr")), on=["year", "mb", "ob"], suffix="_c").filter(c("n_c") >= 2)
            if j.height == 0:
                continue
            w = j["n"].to_numpy().astype(float)
            att = float((w * (j["m"] - j["m_c"]).to_numpy()).sum() / w.sum())
            se = float(np.sqrt((w ** 2 * (j["v"].fill_null(0) / j["n"] + j["v_c"].fill_null(0) / j["n_c"]).to_numpy()).sum()) / w.sum())
            r[f] = {"att": att, "se": se, "control_mean": float((w * j["m_c"].to_numpy()).sum() / w.sum()),
                    "treated_n": int(w.sum())}
        r["elo_median"] = float(d.filter(c("tr"))["me"].median())
        res["bins"][name] = r
        print(f"{name:6s} elo~{r['elo_median']:.0f} " + "  ".join(
            f"{f} {v['att']:+.3f} (z {v['att'] / v['se']:+.1f}, n {v['treated_n']})"
            for f, v in r.items() if isinstance(v, dict)), flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "age.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()

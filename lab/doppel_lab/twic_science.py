"""Over-the-board science on TWIC 2012-2026: era x strength, OTB vs online, junior vs open (CPU, no engine).

    python -m doppel_lab.twic_science

Classification from TWIC's own conventions:
  medium   online if Site ends in " INT" (TWIC's tag for internet play) or Site/Event names chess.com, lichess,
           "online", Titled Tuesday or the PRO League; otherwise over the board.
  speed    fast if the event says rapid / blitz / bullet / armageddon / speed, or is online Titled Tuesday;
           otherwise classical. TWIC has no TimeControl tags, so this is a name heuristic.
  age      junior (youth, junior, U8..U20, cadet, school, girls/boys), senior (senior, +50, +65, veteran), else open.
Ratings are FIDE (TWIC). Unlike Lichess, the FIDE scale is comparatively stable, so tiers here are raw bands.
Book features use move-order prefix keys (book.py): own-year books at equal N for the era table, and one
pooled TWIC book, self-excluded, for the matched contrasts.
Output: lab/results/twic_science.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from doppel_lab.book import move_ids, prefix_hashes

ROOT = Path(__file__).resolve().parents[2]
LAKE = ROOT / "data" / "lake" / "twic"
RESULTS = ROOT / "lab" / "results"
CAP, N_YEAR, SEED = 40, 25_000, 0
BANDS = [(0, 2200, "<2200"), (2200, 2400, "2200-2399"), (2400, 2600, "2400-2599"), (2600, 3000, "2600+")]
c = pl.col


def load() -> pl.DataFrame:
    df = pl.read_parquet(LAKE / "*" / "*.parquet", columns=["game_id", "event", "site", "date", "white_elo",
                                                               "black_elo", "result", "moves", "plies", "fen"])
    df = df.filter(c("fen").is_null() & c("result").is_in(["1-0", "0-1", "1/2-1/2"]) & (c("plies") >= 2)
                   & ~c("moves").list.contains("0000"))  # null moves ("--") in a few PGNs
    df = df.unique("game_id", keep="first")
    ev, site = c("event").fill_null(""), c("site").fill_null("")
    online = (site.str.contains(r" INT$") | site.str.contains(r"(?i)chess\.com|lichess|online")
              | ev.str.contains(r"(?i)online|titled tue|pro league|chess\.com|lichess|bullet"))
    fast = ev.str.contains(r"(?i)rapid|blitz|bullet|armageddon|speed|titled tue|blz|rpd")
    junior = ev.str.contains(r"(?i)youth|junior|\bu-?(8|10|12|14|16|18|20)\b|cadet|school|girls|boys|kids|juniors")
    senior = ev.str.contains(r"(?i)senior|\+50|\+65|veteran|seniors")
    df = df.with_columns(year=c("date").str.slice(0, 4).cast(pl.Int32, strict=False),
                         medium=pl.when(online).then(pl.lit("online")).otherwise(pl.lit("otb")),
                         speed=pl.when(fast).then(pl.lit("fast")).otherwise(pl.lit("classical")),
                         age=pl.when(junior).then(pl.lit("junior")).when(senior).then(pl.lit("senior"))
                         .otherwise(pl.lit("open")),
                         elo=pl.when(c("white_elo").is_not_null() & c("black_elo").is_not_null())
                         .then((c("white_elo") + c("black_elo")) / 2),
                         score_w=pl.when(c("result") == "1-0").then(1.0).when(c("result") == "0-1").then(0.0)
                         .otherwise(0.5))
    return df.filter(c("year").is_between(2012, 2026))


def book_feats(df: pl.DataFrame, self_exclude: bool = True) -> pl.DataFrame:
    """T1 exit, weird-move rate and 6-ply line key per game, against a book of these same games."""
    ids, off = move_ids(df["moves"])
    keys, cnt = prefix_hashes(ids, off, cap=CAP)
    _, inv, cc = np.unique(keys, return_inverse=True, return_counts=True)
    other = cc[inv] - (1 if self_exclude else 0)
    o = np.zeros(len(cnt) + 1, np.int64)
    np.cumsum(cnt, out=o[1:])
    start = o[:-1]
    pos = np.arange(len(keys)) - np.repeat(start, cnt)
    big = np.iinfo(np.int64).max
    f = np.minimum.reduceat(np.where(other < 1, pos, big), start)
    t1 = np.where(f == big, cnt, f) + 1
    par = np.empty(len(keys))
    par[1:] = other[:-1]
    par[start] = len(cnt) - 1
    ok = par >= 100
    red = lambda m: np.add.reduceat(m.astype(np.int64), start)
    weird, elig = red(ok & (other < 0.01 * par)), red(ok)
    line6 = np.where(cnt >= 6, keys[np.minimum(start + 5, len(keys) - 1)], np.uint64(0))
    return df.with_columns(t1=pl.Series(t1.astype(float)), weird=pl.Series(weird.astype(float)),
                           elig=pl.Series(elig.astype(float)), line6=pl.Series(line6),
                           first=c("moves").list.first())


def entropy_bits(x) -> float:
    _, n = np.unique(x, return_counts=True)
    p = n / n.sum()
    return float(-(p * np.log2(p)).sum())


def summ(t: pl.DataFrame) -> dict:
    if t.height == 0:
        return {"games": 0}
    return {"games": t.height, "elo_median": float(t["elo"].median()) if t["elo"].null_count() < t.height else None,
            "plies_median": float(t["plies"].median()), "draw": float((t["result"] == "1/2-1/2").mean()),
            "short_draw": float(((t["result"] == "1/2-1/2") & (t["plies"] <= 40)).mean()),
            "white_score": float(t["score_w"].mean()), "t1_mean": float(t["t1"].mean()),
            "weird_rate": float(t["weird"].sum() / max(1, t["elig"].sum())),
            "open6_entropy_bits": entropy_bits(t["line6"].to_numpy()[t["plies"].to_numpy() >= 6]),
            "e4": float((t["first"] == "e2e4").mean()), "d4": float((t["first"] == "d2d4").mean())}


def era(df: pl.DataFrame) -> dict:
    rng = np.random.default_rng(SEED)
    base = df.filter((c("medium") == "otb") & (c("speed") == "classical") & (c("age") == "open") & c("elo").is_not_null())
    out = {}
    for y in sorted(base["year"].unique().to_list()):
        d = base.filter(c("year") == y)
        if d.height < N_YEAR:
            continue
        d = book_feats(d[np.sort(rng.choice(d.height, N_YEAR, replace=False))])
        r = {"all": summ(d)}
        for lo, hi, name in BANDS:
            r[name] = summ(d.filter(c("elo").is_between(lo, hi - 1e-9)))
        out[str(y)] = r
        a = r["all"]
        print(f"{y} T1={a['t1_mean']:.2f} weird={a['weird_rate']:.3f} H6={a['open6_entropy_bits']:.2f} "
              f"e4={a['e4']:.2f} d4={a['d4']:.2f} draw={a['draw']:.3f} short={a['short_draw']:.3f} "
              f"plies={a['plies_median']:.0f} | " + " ".join(
                  f"{n}:T1={r[n]['t1_mean']:.1f},draw={r[n]['draw']:.2f}" for _, _, n in BANDS if r[n]["games"]),
              flush=True)
    return out


FEATS = ["plies", "draw_f", "short_draw_f", "t1", "weird_rate_g", "score_w"]


def cem(d: pl.DataFrame, treat: pl.Expr, name: str, keys=("year", "eb")) -> dict:
    d = d.with_columns(t=treat, eb=(c("elo") // 50), draw_f=(c("result") == "1/2-1/2").cast(pl.Float64),
                       short_draw_f=((c("result") == "1/2-1/2") & (c("plies") <= 40)).cast(pl.Float64),
                       weird_rate_g=pl.when(c("elig") > 0).then(c("weird") / c("elig")),
                       plies=c("plies").cast(pl.Float64))
    r = {"name": name}
    for f in FEATS:
        x = d.filter(c(f).is_not_null())
        g = x.group_by(list(keys) + ["t"]).agg(m=c(f).mean(), v=c(f).var(), n=pl.len())
        j = g.filter(c("t")).join(g.filter(~c("t")), on=list(keys), suffix="_c").filter(c("n_c") >= 2)
        if j.height == 0:
            r[f] = None
            continue
        w = j["n"].to_numpy().astype(float)
        att = float((w * (j["m"] - j["m_c"]).to_numpy()).sum() / w.sum())
        var_s = (j["v"].fill_null(0) / j["n"] + j["v_c"].fill_null(0) / j["n_c"]).to_numpy()
        se = float(np.sqrt((w ** 2 * var_s).sum()) / w.sum())
        r[f] = {"att": att, "se": se, "control_mean": float((w * j["m_c"].to_numpy()).sum() / w.sum()),
                "treated_n": int(w.sum())}
    print(name + "\n   " + "\n   ".join(
        f"{f} {v['att']:+.3f} (ctrl {v['control_mean']:.3f}, z {v['att'] / v['se'] if v['se'] else 0:+.1f}, n {v['treated_n']})"
        for f, v in r.items() if isinstance(v, dict)), flush=True)
    return r


def main():
    df = load()
    print(f"{df.height:,} TWIC games 2012-2026", flush=True)
    print(df.group_by("medium", "speed", "age").agg(n=pl.len(), elo=c("elo").median()).sort("n", descending=True))
    res = {"games": df.height, "era": era(df)}
    pooled = book_feats(df.filter(c("elo").is_not_null()))
    res["contrasts"] = [
        cem(pooled.filter((c("speed") == "fast") & (c("age") == "open")), c("medium") == "online",
            "online_vs_otb_fast"),
        cem(pooled.filter((c("medium") == "otb") & (c("speed") == "classical") & (c("age") != "senior")),
            c("age") == "junior", "junior_vs_open_classical_otb"),
        cem(pooled.filter((c("medium") == "otb") & (c("age") == "open")), c("speed") == "fast",
            "fast_vs_classical_otb"),
    ]
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "twic_science.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()

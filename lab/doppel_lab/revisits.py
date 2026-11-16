"""Same-player board revisits: how many, how deep, how serendipitous (CPU sizing; no models run).

    python -m doppel_lab.revisits hash  --src heldout [--limit N] [--workers 12]   # pass 1: per-position rows
    python -m doppel_lab.revisits pairs --src heldout [--limit N]                   # pass 2: pairs + rarity
    python -m doppel_lab.revisits report --src heldout [--limit N]                  # sizing tables -> lab/results

Sources: `heldout` / `train` = the 2026-10 rated-blitz corpus shards (data/corpus/2026-10/<src>/, dump order =
time order; no timestamps, so "days" between visits are dump-row distance scaled to the month), `allie` = the
2021-11..2022-12 career histories of the Allie test players (data/allie/history, real utc_ms).

Board key: polyglot Zobrist hash (pieces, side to move, castling, en passant when capturable) at plies >= 10, the
position *before* the mover plays ply k; the mover is the player. A board reached twice in one game counts once
(first visit). Rows are written in 64 key partitions so pass 2 never holds more than 1/64 of them.

Pass 2, per partition: leave-player-out population count of each board = games reaching it minus the player's
own games reaching it; rarity = -log((count + 1) / N_games). Pairs are consecutive visits in the player's career
order. Path: `prefix` = hash of the move list before the board; equal prefixes = the same line (identical path),
different = a transposition, with the divergence ply computed from the two move lists.

Hook (not run; needs GPU and the user's go): opponent surprisal along each path, -sum log P_Maia3(opponent move),
see `opponent_surprisal`.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import multiprocessing as mp
import time
import zlib
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/revisits"
RES = ROOT / "lab/results"
NPART = 64
MINPLY = 10
MONTH_DAYS = 31.0
c = pl.col
PIECE_VAL = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}


def src_files(src: str) -> list[str]:
    if src in ("heldout", "train"):
        return sorted(glob.glob(str(ROOT / f"data/corpus/2026-10/{src}/part-*.parquet")))
    if src == "allie":
        return sorted(glob.glob(str(ROOT / "data/allie/history/*.parquet")))
    raise SystemExit(f"unknown src {src}")


def tag(src: str, limit: int) -> str:
    return f"{src}_smoke{limit}" if limit else src


def load_games(src: str, limit: int) -> pl.DataFrame:
    cols = ["game_id", "white", "black", "white_elo", "black_elo", "base", "inc", "result", "moves", "clocks"]
    fs = src_files(src)
    lf = pl.scan_parquet(fs)
    if src == "allie":
        lf = lf.select(cols + ["utc_ms"]).unique("game_id", keep="first").sort("utc_ms")
    else:
        lf = lf.select(cols)
    if limit:
        lf = lf.head(limit)
    g = lf.collect().with_row_index("gi")
    if src == "allie":
        t0 = g["utc_ms"].min()
        g = g.with_columns(days=(c("utc_ms") - t0) / 86_400_000.0)
    else:
        g = g.with_columns(days=c("gi").cast(pl.Float64) * MONTH_DAYS / g.height)
    return g.with_columns(wl=c("white").str.to_lowercase(), bl=c("black").str.to_lowercase())


def _mvcode(u: str) -> int:
    f = (ord(u[0]) - 97) + 8 * (int(u[1]) - 1)
    t = (ord(u[2]) - 97) + 8 * (int(u[3]) - 1)
    p = " nbrq".find(u[4]) if len(u) > 4 else 0
    return f * 64 + t + 4096 * max(p, 0)


def _hash_chunk(args):
    """Rows for one chunk of games: (gi, ply, key, prefix, mat, mv, clk)."""
    import chess
    import chess.polyglot
    import random
    gis, movelists, clocklists, bases = args
    # second, independent 64-bit Zobrist (own random table): (key, key2) is a 128-bit board key
    h2 = chess.polyglot.ZobristHasher([random.Random(20261115).getrandbits(64) for _ in range(781)])
    out = {k: [] for k in ("gi", "ply", "key", "key2", "prefix", "mat", "mv", "clk")}
    M = (1 << 64) - 1
    for gi, moves, clocks, base in zip(gis, movelists, clocklists, bases):
        b = chess.Board()
        seen = set()
        pre = 1469598103934665603
        for k, u in enumerate(moves):
            if k >= MINPLY:
                key = chess.polyglot.zobrist_hash(b)
                if key not in seen:
                    seen.add(key)
                    mat = sum(PIECE_VAL[p.piece_type] for p in b.piece_map().values())
                    clk = clocks[k - 2] if clocks is not None and k >= 2 and k - 2 < len(clocks) else float(base)
                    out["gi"].append(gi); out["ply"].append(k); out["key"].append(key); out["key2"].append(h2(b))
                    out["prefix"].append(pre)
                    out["mat"].append(mat); out["mv"].append(_mvcode(u)); out["clk"].append(clk if clk is not None else -1.0)
            try:
                b.push_uci(u)
            except ValueError:
                break
            pre = ((pre ^ zlib.crc32(u.encode())) * 1099511628211) & M
    return out


def hash_pass(src: str, limit: int, workers: int):
    t0 = time.time()
    g = load_games(src, limit)
    d = OUT / tag(src, limit)
    d.mkdir(parents=True, exist_ok=True)
    g.drop("moves", "clocks").write_parquet(d / "games.parquet")
    print(f"{g.height:,} games loaded ({time.time() - t0:.0f}s)", flush=True)
    CH, BATCH = 5_000, 4 * workers                 # Python lists only for one batch of chunks at a time
    starts = list(range(0, g.height, CH))
    gsel = g.select("gi", "moves", "clocks", "base")
    del g

    def chunk(i):
        s = gsel.slice(i, CH)
        return (s["gi"].to_list(), s["moves"].to_list(), s["clocks"].to_list(), s["base"].to_list())
    writers: dict[int, list] = {p: [] for p in range(NPART)}
    nrow, flushed = 0, 0

    def flush(final=False):
        nonlocal flushed
        for p, parts in writers.items():
            if parts and (final or sum(x.height for x in parts) > 2_000_000):
                pl.concat(parts).write_parquet(d / f"rows_p{p:02d}_{flushed:04d}.parquet")
                writers[p] = []
        flushed += 1

    with mp.get_context("fork").Pool(workers) as pool:
        results = (r for j in range(0, len(starts), BATCH)
                   for r in pool.map(_hash_chunk, [chunk(i) for i in starts[j:j + BATCH]], chunksize=1))
        for i, r in enumerate(results):
            df = pl.DataFrame({"gi": np.array(r["gi"], np.int32), "ply": np.array(r["ply"], np.int16),
                               "key": np.array(r["key"], np.uint64), "key2": np.array(r["key2"], np.uint64),
                               "prefix": np.array(r["prefix"], np.uint64),
                               "mat": np.array(r["mat"], np.int8), "mv": np.array(r["mv"], np.int16),
                               "clk": np.array(r["clk"], np.float32)})
            nrow += df.height
            df = df.with_columns(part=(c("key") % NPART).cast(pl.Int8))
            for (p,), sub in df.partition_by("part", as_dict=True).items():
                writers[int(p)].append(sub.drop("part"))
            if i % 40 == 39:
                flush()
                print(f"  {i + 1}/{len(starts)} chunks, {nrow:,} rows, {time.time() - t0:.0f}s", flush=True)
    flush(final=True)
    print(f"hash done: {nrow:,} rows in {time.time() - t0:.0f}s -> {d}", flush=True)


def pairs_pass(src: str, limit: int):
    t0 = time.time()
    d = OUT / tag(src, limit)
    g = pl.read_parquet(d / "games.parquet")
    N = g.height
    # per-player career index (games in the sample, in time order)
    sides = pl.concat([g.select("gi", "days", player=c("wl"), opp=c("bl"), elo=c("white_elo"),
                                score=pl.when(c("result") == "1-0").then(1.0).when(c("result") == "0-1").then(0.0).otherwise(0.5),
                                col=pl.lit(0, pl.Int8)),
                       g.select("gi", "days", player=c("bl"), opp=c("wl"), elo=c("black_elo"),
                                score=pl.when(c("result") == "0-1").then(1.0).when(c("result") == "1-0").then(0.0).otherwise(0.5),
                                col=pl.lit(1, pl.Int8))])
    sides = sides.sort("gi").with_columns(cidx=pl.int_range(pl.len()).over("player"),
                                          clen=pl.len().over("player"))
    pids = sides["player"].unique().sort()
    pmap = pl.DataFrame({"player": pids, "pid": np.arange(len(pids), dtype=np.int32)})
    sides = sides.join(pmap, on="player")
    sides = sides.join(pmap.rename({"player": "opp", "pid": "opid"}), on="opp", how="left")
    side_tab = sides.select("gi", "col", "pid", "opid", "player", "opp", "elo", "score", "cidx", "clen", "days")
    out = []
    for p in range(NPART):
        fs = sorted(d.glob(f"rows_p{p:02d}_*.parquet"))
        if not fs:
            continue
        r = pl.concat([pl.read_parquet(f) for f in fs])
        # mover colour: ply even -> white to move
        r = r.with_columns(col=(c("ply") % 2).cast(pl.Int8)).join(side_tab, on=["gi", "col"], how="left")
        K = ["key", "key2"]
        pop = r.group_by(K).agg(n_all=pl.len())
        own = r.group_by(K + ["pid"]).agg(n_own=pl.len())                      # k: player's games, as mover
        as_opp = r.group_by(K + ["opid"]).agg(n_opp=pl.len()).rename({"opid": "pid"})  # player's games, opponent to move
        cand = own.filter(c("n_own") >= 2).select(K + ["pid"])
        if cand.height == 0:
            continue
        rr = r.join(cand, on=K + ["pid"]).sort("pid", "key", "key2", "gi")
        rr = rr.with_columns([c(x).shift(1).over("pid", "key", "key2").alias(f"{x}_1") for x in
                              ("gi", "ply", "prefix", "mv", "clk", "opp", "elo", "score", "cidx", "days", "mat")])
        pr = rr.filter(c("gi_1").is_not_null())
        pr = pr.join(pop, on=K).join(own, on=K + ["pid"]).join(as_opp, on=K + ["pid"], how="left")
        pr = pr.with_columns(n_pop=c("n_all") - c("n_own") - c("n_opp").fill_null(0))
        out.append(pr.select(
            "pid", "player", "key", "key2", "clen", k=c("n_own"),
            g1=c("gi_1"), g2=c("gi"), ply1=c("ply_1"), ply2=c("ply"), mat=c("mat"),
            same_path=(c("prefix_1") == c("prefix")), repeat=(c("mv_1") == c("mv")),
            clk1=c("clk_1"), clk2=c("clk"), opp1=c("opp_1"), opp2=c("opp"), same_opp=(c("opp_1") == c("opp")),
            elo1=c("elo_1"), elo2=c("elo"), score1=c("score_1"),
            games_between=(c("cidx") - c("cidx_1") - 1), days_between=(c("days") - c("days_1")),
            n_pop=c("n_pop"), rarity=-(((c("n_pop") + 1) / N).log()),
            career_freq=c("n_own") / c("clen"), corpus_p=(c("n_pop") + 1) / N,
            expected=c("clen") * (c("n_pop") + 1) / N,
            pmi=((c("n_own") / c("clen")) / ((c("n_pop") + 1) / N)).log()))
        if p % 8 == 7:
            print(f"  partition {p + 1}/{NPART}, {sum(x.height for x in out):,} pairs, {time.time() - t0:.0f}s", flush=True)
    pairs = pl.concat(out)
    # divergence ply for transpositions: first index where the two games' move lists differ
    tr = pairs.filter(~c("same_path")).select("g1", "g2", "ply2").unique()
    if tr.height:
        need = pl.concat([tr["g1"], tr["g2"]]).unique()
        ids = g.select("gi", "game_id").filter(c("gi").is_in(need.implode()))
        mv = (pl.scan_parquet(src_files(src)).select("game_id", "moves")
              .filter(c("game_id").is_in(ids["game_id"].implode())).unique("game_id").collect())
        gm = ids.join(mv, on="game_id")
        mvs = dict(zip(gm["gi"].to_list(), gm["moves"].to_list()))
        div = []
        for a, b in zip(tr["g1"].to_list(), tr["g2"].to_list()):
            ma, mb = mvs.get(a, []), mvs.get(b, [])
            k = 0
            while k < min(len(ma), len(mb)) and ma[k] == mb[k]:
                k += 1
            div.append(k)
        pairs = pairs.join(tr.select("g1", "g2").with_columns(div_ply=pl.Series(div, dtype=pl.Int32)).unique(["g1", "g2"]),
                           on=["g1", "g2"], how="left")
    else:
        pairs = pairs.with_columns(div_ply=pl.lit(None, pl.Int32))
    base = g.select(gi="gi", base="base", inc="inc")
    pairs = pairs.join(base, left_on="g2", right_on="gi", how="left")
    pairs.write_parquet(d / "pairs.parquet")
    meta = {"n_games": N, "n_players": int(len(pids)), "careers_ge2": int(sides.group_by("player").len().filter(c("len") >= 2).height)}
    (d / "meta.json").write_text(json.dumps(meta))
    print(f"pairs done: {pairs.height:,} pairs from {N:,} games in {time.time() - t0:.0f}s", flush=True)


SURP_WINDOW = 20


def _need(pairs: pl.DataFrame) -> pl.DataFrame:
    """Per (game, visit ply): the ply where its surprisal window starts (divergence ply, or ply - 20 if same path)."""
    v1 = pairs.select(gi=c("g1"), ply=c("ply1"), sp=c("same_path"), div=c("div_ply"))
    v2 = pairs.select(gi=c("g2"), ply=c("ply2"), sp=c("same_path"), div=c("div_ply"))
    v = pl.concat([v1, v2]).with_columns(
        start=pl.when(c("sp") | c("div").is_null()).then((c("ply") - SURP_WINDOW).clip(0, None))
        .otherwise(c("div").clip(0, None)).cast(pl.Int32))
    return v.group_by("gi", "ply").agg(start=c("start").min())


def maia_pass(src: str, limit: int, device: str, workers: int, batch: int, max_pairs: int):
    """Maia-3 (fp16, real Elos) at every pair board and along each visit's path: chance agreement and opponent
    surprisal. GPU via the warden; CPU for smoke."""
    import torch

    from doppel_lab.engines import Maia3
    from doppel_lab.teacher import HIST, encode_game
    t0 = time.time()
    d = OUT / tag(src, limit)
    pairs = pl.read_parquet(d / "pairs.parquet")
    if max_pairs and pairs.height > max_pairs:
        pairs = pairs.sample(max_pairs, seed=0)
    need = _need(pairs)
    games = pl.read_parquet(d / "games.parquet").select("gi", "game_id", "white_elo", "black_elo")
    gneed = need.group_by("gi").agg(maxply=c("ply").max())
    ids = games.join(gneed, on="gi")
    mv = (pl.scan_parquet(src_files(src)).select("game_id", "moves")
          .filter(c("game_id").is_in(ids["game_id"].implode())).unique("game_id").collect())
    ids = ids.join(mv, on="game_id").with_columns(moves=c("moves").list.head(c("maxply") + 1)).sort("gi")
    print(f"{pairs.height:,} pairs, {ids.height:,} games, {ids['moves'].list.len().sum():,} positions "
          f"({time.time() - t0:.0f}s)", flush=True)
    eng = Maia3("79m", device)
    model = eng.model.eval()
    visit = {}                                       # (gi, ply) -> (legal vocab idx, probs, played idx)
    surp = {}                                        # gi -> played log-probs per ply (float32)
    want = {}
    for gi, ply in need.select("gi", "ply").iter_rows():
        want.setdefault(gi, set()).add(ply)
    rows = ids.select("gi", "moves", "white_elo", "black_elo").rows()
    BLK = 4000
    with mp.get_context("fork").Pool(workers) as pool:
        for bi in range(0, len(rows), BLK):
            blk = rows[bi:bi + BLK]
            enc = pool.map(encode_game, [(m, we, be) for _, m, we, be in blk], chunksize=32)
            n_per = np.array([len(e[3]) for e in enc]); offsets = np.r_[0, np.cumsum(n_per)]
            planes = torch.from_numpy(np.concatenate([e[0] for e in enc])).to(device)
            hidx = np.concatenate([np.clip(np.arange(n)[:, None] + np.arange(-HIST + 1, 1)[None, :], 0, None) + off
                                   for n, off in zip(n_per, offsets[:-1])])
            legal = np.concatenate([e[1] for e in enc]).astype(np.int64)
            lcount = np.concatenate([e[2] for e in enc]).astype(np.int64)
            lstart = np.r_[0, np.cumsum(lcount)]
            lrow = np.repeat(np.arange(len(lcount)), lcount)
            played = np.concatenate([e[3] for e in enc]).astype(np.int64)
            es = torch.from_numpy(np.concatenate([e[4] for e in enc])).to(device)
            eo = torch.from_numpy(np.concatenate([e[5] for e in enc])).to(device)
            P = len(hidx); hidx_t = torch.from_numpy(hidx).to(device)
            plp = np.empty(P, np.float32)
            vis_rows = {}
            for (gi, *_), off, n in zip(blk, offsets[:-1], n_per):
                for ply in want.get(gi, ()):
                    if ply < n:
                        vis_rows[off + ply] = (gi, ply)
            for s0 in range(0, P, batch):
                e_ = min(P, s0 + batch)
                x = planes[hidx_t[s0:e_]].float().permute(0, 2, 1, 3).reshape(e_ - s0, 64, HIST * 12)
                x = torch.cat([x, torch.zeros(e_ - s0, 64, 1, device=device)], 2)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                    logits = model(x, es[s0:e_], eo[s0:e_])[0].float()
                mask = torch.zeros_like(logits, dtype=torch.bool)
                a_, b_ = lstart[s0], lstart[e_]
                mask[torch.from_numpy(lrow[a_:b_] - s0).to(device), torch.from_numpy(legal[a_:b_]).to(device)] = True
                lp = torch.log_softmax(logits.masked_fill(~mask, float("-inf")), 1)
                pt = torch.from_numpy(played[s0:e_]).to(device)
                plp[s0:e_] = lp.gather(1, pt[:, None])[:, 0].clamp(min=-60).cpu().numpy()
                rows_here = [r for r in range(s0, e_) if r in vis_rows]
                if rows_here:
                    lpc = lp[torch.tensor(rows_here, device=device) - s0].cpu().numpy()
                    for r, row in zip(rows_here, lpc):
                        li = legal[lstart[r]:lstart[r + 1]]
                        visit[vis_rows[r]] = (li.astype(np.int16), np.exp(row[li]).astype(np.float32), int(played[r]))
            for (gi, *_), off, n in zip(blk, offsets[:-1], n_per):
                surp[gi] = plp[off:off + n].copy()
            print(f"  games {min(bi + BLK, len(rows)):,}/{len(rows):,}, {time.time() - t0:.0f}s", flush=True)
    st = dict(zip(need["gi"].to_list(), need["ply"].to_list()))
    starts = {(g, p): s_ for g, p, s_ in need.select("gi", "ply", "start").iter_rows()}
    out = {k: [] for k in ("chance1", "chance_mean", "chance_cross", "p1_mv1", "p2_mv2", "p1_mv2",
                           "maia_hit1", "maia_hit2", "surp1", "surp2", "nopp1", "nopp2")}
    for g1, p1, g2, p2 in pairs.select("g1", "ply1", "g2", "ply2").iter_rows():
        v1, v2 = visit.get((g1, p1)), visit.get((g2, p2))
        if v1 is None or v2 is None or not np.array_equal(v1[0], v2[0]):
            for k in out:
                out[k].append(None)
            continue
        li, q1, m1 = v1; _, q2, m2 = v2
        qm = (q1 + q2) / 2
        pos = {int(x): i for i, x in enumerate(li)}
        out["chance1"].append(float((q1 ** 2).sum())); out["chance_mean"].append(float((qm ** 2).sum()))
        out["chance_cross"].append(float((q1 * q2).sum()))
        out["p1_mv1"].append(float(q1[pos[m1]])); out["p2_mv2"].append(float(q2[pos[m2]])); out["p1_mv2"].append(float(q1[pos[m2]]))
        out["maia_hit1"].append(bool(li[q1.argmax()] == m1)); out["maia_hit2"].append(bool(li[q2.argmax()] == m2))
        for tagv, g, p in (("1", g1, p1), ("2", g2, p2)):
            lpv = surp[g]; s_ = starts[(g, p)]
            opp = [j for j in range(s_, p) if (p - j) % 2 == 1]          # opponent to move at j
            out["surp" + tagv].append(float(-lpv[opp].sum()) if opp else 0.0); out["nopp" + tagv].append(len(opp))
    res = pairs.with_columns([pl.Series(k, v) for k, v in out.items()])
    res.write_parquet(d / "pairs_maia.parquet")
    print(f"maia done: {res.height:,} pairs, {res['chance1'].null_count():,} unmatched ({time.time() - t0:.0f}s)", flush=True)
    maia_report(src, limit)


def maia_report(src: str, limit: int):
    d = OUT / tag(src, limit)
    p = pl.read_parquet(d / "pairs_maia.parquet").filter(c("chance1").is_not_null())
    if p.height == 0:
        raise SystemExit("no matched pairs")
    seen = p.filter(c("n_pop") > 0)
    qs = {q: float(seen["rarity"].quantile(q)) for q in (0.5, 0.9, 0.99)} if seen.height else {0.5: 0, 0.9: 0, 0.99: 0}
    q90_all = float(p["rarity"].quantile(0.9))
    p = p.with_columns(
        rep=c("repeat").cast(pl.Float64),
        plyb=pl.when(c("ply2") < 20).then(pl.lit("10-19")).when(c("ply2") < 30).then(pl.lit("20-29"))
        .when(c("ply2") < 40).then(pl.lit("30-39")).otherwise(pl.lit("40+")),
        ktier=pl.when(c("k") == 2).then(pl.lit("k=2")).when(c("k") <= 5).then(pl.lit("k=3-5")).otherwise(pl.lit("k=6+")),
        rtier=pl.when(c("n_pop") == 0).then(pl.lit("unseen_elsewhere")).when(c("rarity") < qs[0.5]).then(pl.lit("r<q50"))
        .when(c("rarity") < qs[0.9]).then(pl.lit("r_q50-90")).when(c("rarity") < qs[0.99]).then(pl.lit("r_q90-99"))
        .otherwise(pl.lit("r>=q99")),
        gapb=pl.when(c("games_between") == 0).then(pl.lit("0")).when(c("games_between") <= 5).then(pl.lit("1-5"))
        .when(c("games_between") <= 50).then(pl.lit("6-50")).otherwise(pl.lit("51+")),
        corner=(c("k") == 2) & (c("clen") >= 20) & (c("rarity") >= q90_all) & (c("ply2") >= 30) & ~c("same_opp"))

    def agg(df):
        return df.agg(pairs=pl.len(), observed=c("rep").mean(), chance1=c("chance1").mean(), chance_mean=c("chance_mean").mean(),
                      chance_cross=c("chance_cross").mean(), personal=(c("rep") - c("chance1")).mean(),
                      personal_mean=(c("rep") - c("chance_mean")).mean(), maia_top1=c("maia_hit2").mean())

    def tab(*by):
        t = agg(p.group_by(*by)).sort(*by)
        return [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in t.iter_rows(named=True)]

    overall = agg(p.select(pl.all()).group_by(pl.lit(1).alias("all"))).drop("all").to_dicts()[0]
    # bootstrap over players for the overall personal repetition
    g = p.group_by("pid").agg(v=(c("rep") - c("chance1")).sum(), n=pl.len())
    v, n = g["v"].to_numpy(), g["n"].to_numpy(); idx = np.random.default_rng(0).integers(0, len(g), (1000, len(g)))
    bs = v[idx].sum(1) / n[idx].sum(1)
    cr = p.filter(c("corner"))
    if cr.height:
        cr = cr.with_columns(score=(c("rarity").rank() + c("surp1").rank() + c("surp2").rank() + c("ply2").rank()) / (4 * cr.height))
    res = {"src": src, "limit": limit, "pairs": p.height,
           "overall": {k: (round(v_, 4) if isinstance(v_, float) else v_) for k, v_ in overall.items()},
           "personal_ci95_player_bootstrap": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
           "by_ply": tab("plyb"), "k_x_rarity": tab("ktier", "rtier"), "by_same_opp": tab("same_opp"),
           "by_path": tab("same_path"), "by_gap_games": tab("gapb"),
           "surprisal": {"median_surp1": float(p["surp1"].median()), "median_surp2": float(p["surp2"].median()),
                         "median_per_opp_move": float((p["surp1"] / c("nopp1").clip(1, None)).median()) if False else
                         float((p["surp1"] / p["nopp1"].clip(1, None)).median())},
           "deep_corner": {"pairs": cr.height, **({k: (round(v_, 4) if isinstance(v_, float) else v_) for k, v_ in
                                                   agg(cr.group_by(pl.lit(1).alias("a"))).drop("a").to_dicts()[0].items()} if cr.height else {})},
           "deep_corner_top": cr.sort("score", descending=True).head(20).select(
               "player", "g1", "g2", "ply1", "ply2", "same_path", "div_ply", "k", "clen", "n_pop", "rarity", "surp1", "surp2",
               "repeat", "chance1", "p1_mv1", "games_between", "score").to_dicts() if cr.height else [],
           "notes": "chance1 = sum p^2 under Maia-3 at visit 1 (real Elos, fp16); chance_mean uses the mean of both visits' "
                    "distributions; personal = observed repeat - chance. Surprisal = -sum log p(opponent move) over the "
                    "opponent's moves from the divergence ply (transpositions) or the last 20 plies (same path)."}
    RES.mkdir(parents=True, exist_ok=True)
    out = RES / ("revisits_maia.json" if not limit else f"revisits_maia_{tag(src, limit)}.json")
    out.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v_ for k, v_ in res.items() if k not in ("deep_corner_top", "k_x_rarity")}, indent=1, default=str)[:4000])
    print(f"-> {out}")


def tc_class(base: pl.Expr, inc: pl.Expr) -> pl.Expr:
    est = base + 40 * inc
    return pl.when(est < 180).then(pl.lit("bullet")).when(est < 480).then(pl.lit("blitz")) \
        .when(est < 1500).then(pl.lit("rapid")).otherwise(pl.lit("classical"))


def report(src: str, limit: int):
    d = OUT / tag(src, limit)
    p = pl.read_parquet(d / "pairs.parquet")
    meta = json.loads((d / "meta.json").read_text())
    N, NP = meta["n_games"], meta["n_players"]
    q99 = float(p["rarity"].quantile(0.99)) if p.height else float("nan")
    seen = p.filter(c("n_pop") > 0)                  # quantiles over boards others reached too; n_pop = 0 is its own tier
    qs = {q: float(seen["rarity"].quantile(q)) for q in (0.5, 0.9, 0.99)} if seen.height else {}
    q90_all = float(p["rarity"].quantile(0.9)) if p.height else 0.0
    cfq = [float(p["career_freq"].quantile(q)) for q in (0.25, 0.5, 0.75)] if p.height else [0, 0, 0]
    p = p.with_columns(
        ktier=pl.when(c("k") == 2).then(pl.lit("k=2")).when(c("k") <= 5).then(pl.lit("k=3-5")).otherwise(pl.lit("k=6+")),
        cfq=pl.when(c("career_freq") <= cfq[0]).then(pl.lit("cf_q1")).when(c("career_freq") <= cfq[1]).then(pl.lit("cf_q2"))
        .when(c("career_freq") <= cfq[2]).then(pl.lit("cf_q3")).otherwise(pl.lit("cf_q4")),
        rtier=pl.when(c("n_pop") == 0).then(pl.lit("unseen_elsewhere")).when(c("rarity") < qs.get(0.5, 0)).then(pl.lit("r<q50")).when(c("rarity") < qs.get(0.9, 0)).then(pl.lit("r_q50-90"))
        .when(c("rarity") < qs.get(0.99, 0)).then(pl.lit("r_q90-99")).otherwise(pl.lit("r>=q99")),
        corner=(c("k") == 2) & (c("clen") >= 20) & (c("rarity") >= q90_all) & (c("ply2") >= 30) & ~c("same_opp"))
    p = p.with_columns(
        plyb=pl.when(c("ply2") < 20).then(pl.lit("10-19")).when(c("ply2") < 30).then(pl.lit("20-29"))
        .when(c("ply2") < 40).then(pl.lit("30-39")).otherwise(pl.lit("40+")),
        clb=pl.when(c("clen") < 10).then(pl.lit("<10")).when(c("clen") < 50).then(pl.lit("10-49"))
        .when(c("clen") < 200).then(pl.lit("50-199")).when(c("clen") < 1000).then(pl.lit("200-999")).otherwise(pl.lit("1000+")),
        tc=tc_class(c("base"), c("inc")),
        deep=(c("ply2") >= 30) & ~c("same_opp") & (~c("same_path") | (c("rarity") >= q99)),
        deep_strict=(c("ply2") >= 30) & ~c("same_opp") & ~c("same_path") & (c("rarity") >= q99))
    pl_clen = pl.read_parquet(d / "games.parquet").select(pl.concat_list("wl", "bl").alias("p")).explode("p") \
        .group_by("p").len()
    p99 = float(pl_clen["len"].quantile(0.99))

    def tab(by):
        t = p.group_by(by).agg(pairs=pl.len(), repeat_rate=c("repeat").mean(), players=c("pid").n_unique(),
                                median_rarity=c("rarity").median()).sort(by)
        return [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()} for row in t.iter_rows(named=True)]

    def tab2(a_, b_):
        t = p.group_by(a_, b_).agg(pairs=pl.len(), repeat_rate=c("repeat").mean(), players=c("pid").n_unique()).sort(a_, b_)
        return [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()} for row in t.iter_rows(named=True)]

    def tier(mask):
        s = p.filter(mask)
        return {"pairs": s.height, "players": s["pid"].n_unique(), "repeat_rate": round(float(s["repeat"].mean()), 4) if s.height else None,
                "pairs_per_1000_careers": round(1000 * s.height / NP, 3),
                "share_from_top1pct_careers": round(float((s["clen"] > p99).mean()), 4) if s.height else None}

    res = {"src": src, "limit": limit, "n_games": N, "n_players": NP, "careers_ge2": meta["careers_ge2"],
           "career_len_p99": p99, "rarity_q99": round(q99, 3),
           "all": tier(pl.lit(True)), "different_opponent": tier(~c("same_opp")),
           "middlegame_ply30plus_diffopp": tier((c("ply2") >= 30) & ~c("same_opp")),
           "deep": tier(c("deep")), "deep_strict": tier(c("deep_strict")),
           "by_ply": tab("plyb"), "by_career_len": tab("clb"), "by_tc": tab("tc"), "by_same_opp": tab("same_opp"),
           "by_path": tab("same_path"),
           "deep_by_career_len": [r for r in p.filter(c("deep")).group_by("clb").agg(pairs=pl.len(), repeat_rate=c("repeat").mean()).sort("clb").iter_rows(named=True)],
           "deep_examples": p.filter(c("deep_strict")).sort("rarity", descending=True).head(10)
           .select("player", "g1", "g2", "ply1", "ply2", "div_ply", "rarity", "repeat", "games_between", "days_between").to_dicts(),
           "rarity_quantiles_seen_elsewhere": {str(k): round(v, 3) for k, v in qs.items()}, "rarity_q90_all": round(q90_all, 3),
           "share_unseen_elsewhere": round(float((p["n_pop"] == 0).mean()), 4) if p.height else None, "career_freq_quartiles": [round(x, 4) for x in cfq],
           "k_x_rarity": tab2("ktier", "rtier"), "careerfreq_x_rarity": tab2("cfq", "rtier"),
           "pmi_quantiles": {str(q): round(float(p["pmi"].quantile(q)), 3) for q in (0.1, 0.5, 0.9, 0.99)} if p.height else {},
           "deep_corner_k2_N20_rarity_top10_ply30_diffopp": tier(c("corner")),
           "deep_corner_examples": p.filter(c("corner")).sort("pmi", descending=True).head(10)
           .select("player", "g1", "g2", "ply1", "ply2", "same_path", "div_ply", "k", "clen", "n_pop", "rarity", "pmi",
                   "repeat", "games_between", "days_between").to_dicts(),
           "notes": "rarity/corpus_p use (n_pop+1)/N_games; n_pop excludes all of the player's games (either colour). "
                    "Keys: two independent 64-bit Zobrist hashes (128 bits). Within-game repeats dropped (first visit kept)."}
    RES.mkdir(parents=True, exist_ok=True)
    out = RES / f"revisits_size_{tag(src, limit)}.json"
    out.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k not in ("deep_examples",)}, indent=1, default=str)[:6000])
    print(f"-> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("hash", "pairs", "report", "maia", "maia-report"):
        q = sub.add_parser(name)
        q.add_argument("--src", default="heldout", choices=["heldout", "train", "allie"])
        q.add_argument("--limit", type=int, default=0)
        if name in ("hash", "maia"):
            q.add_argument("--workers", type=int, default=12)
        if name == "maia":
            q.add_argument("--gpu-ok", action="store_true", help="run on cuda (via the warden only)")
            q.add_argument("--batch", type=int, default=2048)
            q.add_argument("--max-pairs", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "maia":
        maia_pass(a.src, a.limit, "cuda" if a.gpu_ok else "cpu", a.workers, a.batch, a.max_pairs)
        raise SystemExit(0)
    if a.cmd == "maia-report":
        maia_report(a.src, a.limit)
        raise SystemExit(0)
    if a.cmd == "hash":
        hash_pass(a.src, a.limit, a.workers)
    elif a.cmd == "pairs":
        pairs_pass(a.src, a.limit)
    else:
        report(a.src, a.limit)

"""Personal position books (R0, R2 history): how much of a later game is in a player's own book, and does the
book identify the player? (CPU)

    python -m doppel_lab.posbook_personal bandbooks   # Lichess train R0 tables per 200-point band (poskey)
    python -m doppel_lab.posbook_personal run

Players as reid_k.py (2026-10 heldout, 3+0, >= 80 games, the same 8,000 sampled with seed 0; stream order is time).
Target games: each player's last 40 (reid_k's gallery). History: their first k games (k = 1 .. 40, disjoint).
A player's personal book after k games = the positions they faced as the mover in those games (plies < 40).
The band book = 2026-10 train games of the player's 200-point band (mean rating over their games), position
count >= 2 (data/v0/posbook/band/b<band>). All counts are R0 (position key).

1. Coverage by k: share of own decision positions (plies 1-39) of the target games that are in the personal
   book, in the band book, in the personal book only, and the personal / band book exit (first own decision
   position not in it). Also how often the player repeats their own earlier move there (modal move of history).
2. Re-ID by personal book: tokens = positions the player creates with their own move (plies < 30). Two weightings:
   rarity (idf = log((N + 1) / (band count + 1)): rare positions weigh most) and divergence (only moves chosen by
   < 1% of band games from a position with >= 100: the weird moves). Cosine between the first-k-games probe and
   the last-40 gallery, pools of 1,000 as reid_k. Fusion with reid_k's surface features: sum of per-row z-scores.
Output: lab/results/posbook_explore.json, section "personal_books".
"""

from __future__ import annotations

import argparse
import json
import subprocess

import numpy as np
import polars as pl
import scipy.sparse as sp

from doppel_lab.posbook_common import (BIN, CORPUS, N_TRAIN, R1_DIR, V0, annotate, load_table, lookup, offsets,
                                       save_section)

c = pl.col
BAND_DIR = V0 / "posbook/band"
BANDS = tuple(range(600, 2800, 200))
KS = (1, 2, 5, 10, 20, 40)
GALLERY, MIN_GAMES, N_PLAYERS, POOL, SEED = 40, 80, 8_000, 1_000, 0
MAX_PLY, TOKEN_PLY = 40, 30


def bandbooks():
    files = sorted(str(p) for p in (CORPUS / "train").glob("*.parquet"))
    for b in BANDS:
        out = BAND_DIR / f"b{b}"
        if (out / "stats.json").exists():
            continue
        lo = 0 if b == BANDS[0] else b
        hi = 4000 if b == BANDS[-1] else b + 199
        subprocess.run([str(BIN), "count", "--out", str(out), "--threads", "16", "--fpr", "0.001",
                        "--band-min", str(lo), "--band-max", str(hi), *files], check=True, capture_output=True)
        s = json.loads((out / "stats.json").read_text())
        print(f"band {b}: {s['games']:,} games, {s['table_keys']:,} keys, {s['seconds_total']:.0f}s", flush=True)


def select_players():
    from doppel_lab.split import held_out
    files = sorted((CORPUS / "heldout").glob("*.parquet"))
    meta = pl.concat([pl.read_parquet(f, columns=["white", "black", "base", "inc"]) for f in files]).with_row_index("t")
    g3 = meta.filter((c("base") == 180) & (c("inc") == 0))
    names = pl.concat([g3["white"], g3["black"]]).str.to_lowercase().value_counts().filter(c("count") >= MIN_GAMES)
    names = [n for n in names["white"].to_list() if held_out(n)]
    rng = np.random.default_rng(SEED)
    keep = sorted(rng.choice(sorted(names), min(N_PLAYERS, len(names)), replace=False).tolist())
    return files, keep, rng


def load_games(files, keep):
    ks = set(keep)
    parts, t0 = [], 0
    cols = ["white", "black", "white_elo", "black_elo", "base", "inc", "result", "termination", "moves", "clocks"]
    for f in files:
        g = pl.read_parquet(f, columns=cols).with_row_index("t", offset=t0)
        t0 += g.height
        g = g.filter((c("base") == 180) & (c("inc") == 0) & (c("white").str.to_lowercase().is_in(list(ks)) |
                                                               c("black").str.to_lowercase().is_in(list(ks))))
        parts.append(g)
    return pl.concat(parts)


def build():
    """The players' games and one row per own move (plies < 40) with band-book counts."""
    files, keep, rng = select_players()
    g = load_games(files, keep)
    print(f"{len(keep):,} players, {g.height:,} games", flush=True)
    keys_all, lens_all = annotate("heldout_cap40", files, cap=40)
    koff = offsets(lens_all)
    t = g["t"].to_numpy()
    lens = lens_all[t]
    idx = np.repeat(koff[t], lens) + (np.arange(lens.sum()) - np.repeat(offsets(lens)[:-1], lens))
    keys = np.asarray(keys_all)[idx]
    goff = offsets(lens)
    pid = {n: i for i, n in enumerate(keep)}
    # per own move: player, game stream index, ply, decision key (start = 0 sentinel), child key
    rows = []
    for side, col in ((0, "white"), (1, "black")):
        names = g[col].str.to_lowercase().to_list()
        gsel = np.array([i for i, n in enumerate(names) if n in pid])
        p_of = np.array([pid[names[i]] for i in gsel])
        L = lens[gsel]
        # own plies side, side+2, ... < min(L, MAX_PLY)
        n_own = np.maximum((np.minimum(L, MAX_PLY) - side + 1) // 2, 0)
        gi = np.repeat(np.arange(len(gsel)), n_own)
        j = np.arange(n_own.sum()) - np.repeat(offsets(n_own)[:-1], n_own)
        ply = side + 2 * j
        base = goff[gsel][gi]
        child = keys[base + ply]
        dec = np.where(ply == 0, np.uint64(0), keys[np.maximum(base + ply - 1, 0)])
        rows.append(pl.DataFrame({"p": p_of[gi].astype(np.int32), "t": t[gsel][gi], "ply": ply.astype(np.int16),
                                  "dec": dec, "child": child}))
    m = pl.concat(rows)
    m = m.with_columns(k=c("t").rank("dense").over("p").cast(pl.Int32) - 1,
                       n=c("t").n_unique().over("p").cast(pl.Int32))
    elo = pl.concat([g.select(p=c("white").str.to_lowercase(), e=c("white_elo")),
                     g.select(p=c("black").str.to_lowercase(), e=c("black_elo"))]).filter(c("p").is_in(keep))
    elo = elo.group_by("p").agg(c("e").mean()).with_columns(p=c("p").replace_strict(pid, return_dtype=pl.Int32))
    elo = elo.with_columns(band=(c("e") // 200 * 200).clip(BANDS[0], BANDS[-1]).cast(pl.Int64))
    m = m.join(elo.select("p", "band"), on="p")
    # band-book counts of decision and child positions
    cd, cc = np.zeros(m.height, np.int64), np.zeros(m.height, np.int64)
    bnd = m["band"].to_numpy()
    dec, child = m["dec"].to_numpy(), m["child"].to_numpy()
    nband = {}
    for b in BANDS:
        sel = np.nonzero(bnd == b)[0]
        if not len(sel):
            continue
        tk, tc = load_table(BAND_DIR / f"b{b}")
        nband[b] = json.loads((BAND_DIR / f"b{b}/stats.json").read_text())["games"]
        cd[sel], cc[sel] = lookup(tk, tc, dec[sel]), lookup(tk, tc, child[sel])
        cd[sel[dec[sel] == 0]] = nband[b]
    m = m.with_columns(cd=pl.Series(cd), cc=pl.Series(cc))
    return g, keep, rng, m, nband


def run():
    g, keep, rng, m, nband = build()
    res = {"players": len(keep), "games": g.height, "own_moves": m.height, "band_games": nband}
    res["coverage"] = coverage(m)
    res["reid"] = reid(m, g, keep, rng, nband)
    save_section("personal_books", res)


def reid_only():
    """Recompute only the re-ID part and update it in place."""
    import json as _j
    from doppel_lab.posbook_common import RESULTS
    g, keep, rng, m, nband = build()
    res = _j.loads(RESULTS.read_text()).get("personal_books", {})
    res["reid"] = reid(m, g, keep, rng, nband)
    save_section("personal_books", res)


def modal(k: int = 40, sample: int = 150_000):
    """Same positions, two predictors: the player's own modal move from their first k games vs the band book's
    most common move (argmax over legal moves of the band count of the resulting position, R0). Target: the
    last 40 games' own decisions at plies 1-39 that are in the personal book; a random sample of them."""
    import chess
    import chess.polyglot
    g, keep, rng, m, nband = build()
    hist = m.filter(c("k") < k)
    mode = hist.group_by("p", "dec", "child").len().sort("len", descending=True).unique(["p", "dec"], keep="first")
    tgt = m.filter((c("k") >= c("n") - GALLERY) & (c("ply") >= 1)).join(
        mode.select("p", "dec", mchild="child", mlen="len"), on=["p", "dec"], how="inner")
    tgt = tgt.sample(min(sample, tgt.height), seed=SEED)
    moves = dict(zip(g["t"].to_list(), g["moves"].to_list()))
    tables = {b: load_table(BAND_DIR / f"b{b}") for b in set(tgt["band"].to_list())}
    band_hit, own_hit, agree, n_legal = [], [], [], []
    for r in tgt.iter_rows(named=True):
        b = chess.Board()
        for u in moves[r["t"]][: r["ply"]]:
            b.push_uci(u)
        tk, tc = tables[r["band"]]
        best, best_c = None, -1
        legal = list(b.legal_moves)
        for mv in legal:
            b.push(mv)
            kk = np.uint64(chess.polyglot.zobrist_hash(b))
            b.pop()
            cnt = int(lookup(tk, tc, np.array([kk], np.uint64))[0])
            if cnt > best_c:
                best, best_c = kk, cnt
        band_hit.append(best == np.uint64(r["child"]))
        own_hit.append(r["mchild"] == r["child"])
        agree.append(best == np.uint64(r["mchild"]))
        n_legal.append(len(legal))
    t = tgt.with_columns(band_hit=pl.Series(band_hit), own_hit=pl.Series(own_hit), agree=pl.Series(agree),
                         common=c("cd") >= 100)
    def summ(x):
        return {"n": x.height, "own_modal_top1": float(x["own_hit"].mean()), "band_modal_top1": float(x["band_hit"].mean()),
                "own_and_band_modal_agree": float(x["agree"].mean()),
                "own_right_band_wrong": float((x["own_hit"] & ~x["band_hit"]).mean()),
                "band_right_own_wrong": float((x["band_hit"] & ~x["own_hit"]).mean())}
    out = {"k": k, "all": summ(t), "common_position": summ(t.filter(c("common"))),
           "rare_position": summ(t.filter(~c("common"))),
           "own_mode_seen_once": summ(t.filter(c("mlen") == 1)), "own_mode_seen_3plus": summ(t.filter(c("mlen") >= 3))}
    print(json.dumps(out, indent=1), flush=True)
    save_section("personal_vs_band_modal_move", out)


def coverage(m: pl.DataFrame) -> dict:
    tgt = m.filter((c("k") >= c("n") - GALLERY) & (c("ply") >= 1))
    out = {"target_own_moves": tgt.height, "by_k": {}}
    in_band = tgt["cd"].to_numpy() >= 2
    out["band_share"] = float(in_band.mean())
    for k in KS:
        hist = m.filter(c("k") < k).select("p", "dec", "child")
        seen = hist.select("p", "dec").unique().with_columns(seen=pl.lit(True))
        x = tgt.join(seen, on=["p", "dec"], how="left").with_columns(c("seen").fill_null(False))
        # the player's own earlier move here: modal child of the history (ties: any)
        mode = hist.group_by("p", "dec", "child").len().sort("len", descending=True).unique(["p", "dec"], keep="first")
        x = x.join(mode.select("p", "dec", mchild="child"), on=["p", "dec"], how="left")
        pers = x["seen"].to_numpy()
        band = x["cd"].to_numpy() >= 2
        rep = (x["mchild"] == x["child"]).fill_null(False).to_numpy()
        common = x["cd"].to_numpy() >= 100
        # exits: first own decision ply not in the book, per target game (games that never leave: 40)
        ex = x.with_columns(pe=pl.when(~c("seen")).then(c("ply")).otherwise(MAX_PLY),
                            be=pl.when(c("cd") < 2).then(c("ply")).otherwise(MAX_PLY)).group_by("p", "t").agg(
            pe=c("pe").min(), be=c("be").min())
        ply = x["ply"].to_numpy()
        row = {"personal_share": float(pers.mean()), "band_share": float(band.mean()),
               "personal_only_share": float((pers & ~band).mean()), "union_share": float((pers | band).mean()),
               "personal_share_by_plies": {f"{a}-{b}": float(pers[(ply >= a) & (ply <= b)].mean())
                                           for a, b in ((1, 10), (11, 20), (21, 39))},
               "band_share_by_plies": {f"{a}-{b}": float(band[(ply >= a) & (ply <= b)].mean())
                                       for a, b in ((1, 10), (11, 20), (21, 39))},
               "repeat_own_move_where_personal": float(rep[pers].mean()) if pers.any() else None,
               "repeat_own_move_where_personal_common": float(rep[pers & common].mean()),
               "repeat_own_move_where_personal_rare": float(rep[pers & ~common].mean()),
               "personal_exit_mean": float(ex["pe"].mean()), "band_exit_mean": float(ex["be"].mean()),
               "personal_exit_beyond_band_share": float((ex["pe"] > ex["be"]).mean())}
        out["by_k"][k] = row
        print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in row.items() if not isinstance(b, dict)},
              flush=True)
    return out


def _tokens(m: pl.DataFrame, cond, nband: dict, mode: str):
    x = m.filter(cond & (c("ply") < TOKEN_PLY))
    if mode == "divergence":
        x = x.filter((c("cd") >= 100) & (c("cc") < 0.01 * c("cd")))
    x = x.select("p", "child", "band", "cc").unique(["p", "child"])
    nb = x["band"].replace_strict(nband, return_dtype=pl.Float64).to_numpy()
    w = np.log((nb + 1) / (x["cc"].to_numpy() + 1))
    return x["p"].to_numpy(), x["child"].to_numpy(), w


def reid(m: pl.DataFrame, g: pl.DataFrame, keep, rng, nband) -> dict:
    surf_probe, surf_gal = surface(g, keep)
    P = len(keep)
    out = {"pool": POOL, "gallery_games": GALLERY, "ks": {}}
    gal_cond = c("k") >= c("n") - GALLERY
    for k in KS:
        row = {}
        S = {}
        for mode in ("rarity", "divergence"):
            pp, pt, pw = _tokens(m, c("k") < k, nband, mode)
            gp, gt, gw = _tokens(m, gal_cond, nband, mode)
            vocab, inv = np.unique(np.r_[pt, gt], return_inverse=True)
            A = sp.csr_matrix((pw, (pp, inv[:len(pt)])), shape=(P, len(vocab)))
            B = sp.csr_matrix((gw, (gp, inv[len(pt):])), shape=(P, len(vocab)))
            na = np.sqrt(np.asarray(A.multiply(A).sum(1)).ravel()) + 1e-12
            nb_ = np.sqrt(np.asarray(B.multiply(B).sum(1)).ravel()) + 1e-12
            S[mode] = (A, B, na, nb_)
        row_empty = {mode: float((np.diff(S[mode][0].indptr) == 0).mean()) for mode in S}
        Dsurf = surf_probe[k]
        t1 = {n: 0 for n in ("surface", "rarity", "divergence", "surface+rarity", "surface+rarity+divergence")}
        t10 = dict.fromkeys(t1, 0)
        n = 0
        prng = np.random.default_rng(SEED + k)
        for _ in range(max(1, P // POOL)):
            idx = prng.choice(P, POOL, replace=False)
            sc = {}
            for mode, (A, B, na, nb_) in S.items():
                sc[mode] = (A[idx] @ B[idx].T).toarray() / na[idx][:, None] / nb_[idx][None, :]
            sc["surface"] = -((Dsurf[idx][:, None, :] - surf_gal[idx][None, :, :]) ** 2).sum(-1)
            z = {kk: (v - v.mean(1, keepdims=True)) / (v.std(1, keepdims=True) + 1e-12) for kk, v in sc.items()}
            sc["surface+rarity"] = z["surface"] + z["rarity"]
            sc["surface+rarity+divergence"] = z["surface"] + z["rarity"] + z["divergence"]
            for name in t1:
                v = sc[name]
                true = v[np.arange(POOL), np.arange(POOL)][:, None]
                gt, eq = (v > true).sum(1), (v == true).sum(1)       # eq counts the true match itself
                # random tie-breaking: an all-zero probe ties with everyone and scores chance, not a hit
                t1[name] += float(np.where(gt == 0, 1.0 / eq, 0.0).sum())
                t10[name] += float(np.clip((10 - gt) / eq, 0, 1).sum())
            n += POOL
        row = {name: {"top1": t1[name] / n, "top10": t10[name] / n} for name in t1}
        row["probe_without_tokens"] = row_empty
        out["ks"][k] = row
        print(f"k={k}: " + ", ".join(f"{a} {b['top1']:.3f}" for a, b in row.items() if "top1" in b), row_empty,
              flush=True)
    return out


def surface(g: pl.DataFrame, keep):
    """reid_k's surface features: probe = mean of the first k games, gallery = last 40, rating-residualised."""
    import doppel_lab.persistence as per
    per.BOOK = R1_DIR                       # persistence reads <repo>/data; point it at DOPPEL_DATA
    FEATS = per.FEATS
    s = per.side_feats(g).filter(c("player").is_in(keep)).sort("t")
    s = s.with_columns(k=pl.int_range(pl.len()).over("player"), n=pl.len().over("player"))
    order = pl.DataFrame({"player": keep})
    gal = order.join(s.filter(c("k") >= c("n") - GALLERY).group_by("player").agg(
        c("elo").mean(), *[c(f).fill_nan(None).mean() for f in FEATS]), on="player", how="left")
    G = gal.select(FEATS).to_numpy().astype(float)
    elo_g = gal["elo"].to_numpy().astype(float)
    sd = np.nanstd(G, 0)

    def resid(X, elo):
        out = np.empty_like(X)
        Z = np.c_[np.ones(len(elo)), elo, elo ** 2]
        for j in range(X.shape[1]):
            x = X[:, j]
            ok = np.isfinite(x) & np.isfinite(elo)
            b = np.linalg.lstsq(Z[ok], x[ok], rcond=None)[0] if ok.sum() > 10 else np.zeros(3)
            out[:, j] = np.where(ok, (x - Z @ b) / sd[j], 0.0)
        return np.nan_to_num(out)
    probes = {}
    for k in KS:
        pr = order.join(s.filter(c("k") < k).group_by("player").agg(
            c("elo").mean(), *[c(f).fill_nan(None).mean() for f in FEATS]), on="player", how="left")
        probes[k] = resid(pr.select(FEATS).to_numpy().astype(float), pr["elo"].to_numpy().astype(float))
    return probes, resid(G, elo_g)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bandbooks", "run", "modal", "reid"])
    a = ap.parse_args()
    {"bandbooks": bandbooks, "run": run, "modal": modal, "reid": reid_only}[a.cmd]()

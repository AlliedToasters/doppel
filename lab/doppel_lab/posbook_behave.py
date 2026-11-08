"""Popularity vs behaviour: does how often a position occurs predict think time, accuracy and blunders? (CPU)

    python -m doppel_lab.posbook_behave moves      # 1500 per-move tables (T1/T2) + R0/R1 counts
    python -m doppel_lab.posbook_behave bands      # heldout eval games, every band: think time only
    python -m doppel_lab.posbook_behave noise      # self-agreement by position popularity (heldout 3+0, R0)

For each own move, the decision position's count in the 2026-10 train book, by position key (R0) and by move-order
prefix (R1), in log buckets (posbook_common.bucket). Targets: think share of the remaining clock
(think / (clock before + inc)), Maia-3 top-1 and played-move log-prob, blunder (wc loss >= 0.3).
Controls: a fixed-effect cell per (ply, mover rating / 100, clock fraction bin, time control). "Partial R²" is the
share of within-cell variance a key's buckets explain (two-way fixed effects by alternating projections).
The two keys disagree only after a transposition: R1 <= R0 always. The transposition test compares plies where
the position is common but the route is rare (R0 >= 16, R1 <= 1) with plies in book by both keys and plies out
of book by both, in the same cells; position index 0 = they behave as in book (the position is what matters),
1 = as out of book (the route is what matters).

Data: data/v0/features/moves_f1_T2_1500_{train,val} (100k train games, self-excluded from both books; 20k
heldout games); book.py's heldout eval games (10k per band, exact probe counts) for the band view.
Output: lab/results/posbook_explore.json, section "behaviour".
"""

from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from doppel_lab.posbook_common import (BUCKET_LABELS, BUCKET_X, CORPUS, N_TRAIN, R0_DIR, V0, annotate, bucket,
                                       decision_counts, load_table, lookup, offsets, ply_index, r0_counts, r1_counts,
                                       save_section)

c = pl.col
MAX_PLY = 60
CLOCK_EDGES = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
TARGETS = {"think_logit": "logit((think + 0.5) / (clock before + inc)), the registry's think_frac",
           "think_share": "think share of the remaining clock", "top1": "Maia-3 top-1 (played = Maia's top move)",
           "lp": "Maia-3 log-prob of the played move (nats)", "blunder": "blunder rate (wc loss >= 0.3)"}


# --- fixed effects -------------------------------------------------------------------------------------------------

def fe_fit(y: np.ndarray, groups: list[np.ndarray], iters: int = 100, tol: float = 1e-10):
    """Additive fixed effects by alternating projections. Returns (residual, [effect per group level])."""
    r = y - y.mean()
    n = [np.bincount(g).astype(float) for g in groups]
    eff = [np.zeros(len(k)) for k in n]
    prev = np.inf
    for _ in range(iters):
        for j, g in enumerate(groups):
            m = np.bincount(g, r, len(n[j])) / np.maximum(n[j], 1)
            r = r - m[g]
            eff[j] += m
        ss = float(r @ r)
        if prev - ss < tol * max(ss, 1e-12):
            break
        prev = ss
    return r, eff


def codes(*cols) -> np.ndarray:
    """Dense integer codes of the joint value of several integer columns."""
    a = np.stack([np.asarray(x, np.int64) for x in cols], 1)
    _, inv = np.unique(a, axis=0, return_inverse=True)
    return inv.ravel()


def horse_race(y: np.ndarray, cell: np.ndarray, b0: np.ndarray, b1: np.ndarray, w_label: str) -> dict:
    """Partial R² of R0 buckets, R1 buckets, both and their interaction, beyond the control cells."""
    ok = np.isfinite(y)
    y, cell, b0, b1 = y[ok], cell[ok], b0[ok], b1[ok]
    cell = np.unique(cell, return_inverse=True)[1]
    r_c, _ = fe_fit(y, [cell])
    ss_c = float(r_c @ r_c)
    out = {"n": int(len(y)), "target": w_label, "ss_within_cells": ss_c}
    fits = {}
    for name, gs in (("R0", [cell, b0]), ("R1", [cell, b1]), ("R0+R1", [cell, b0, b1]),
                     ("R0xR1", [cell, codes(b0, b1)])):
        r, eff = fe_fit(y, gs)
        fits[name] = eff
        out[f"partial_r2_{name}"] = 1 - float(r @ r) / ss_c
    # bucket effects (relative to the top bucket present) in the joint additive model, and separately
    for name, j, b in (("R0", 1, b0), ("R1", 2, b1)):
        e = fits["R0+R1"][j]
        nb = np.bincount(b, minlength=12)
        ref = np.nonzero(nb)[0].max()
        out[f"joint_effect_{name}"] = {BUCKET_LABELS[k]: float(e[k] - e[ref]) for k in range(len(e)) if nb[k] >= 200}
        e1 = fits[name][1]
        out[f"alone_effect_{name}"] = {BUCKET_LABELS[k]: float(e1[k] - e1[ref]) for k in range(len(e1)) if nb[k] >= 200}
    return out


def adjusted_means(y, cell, groupvar, n_groups) -> np.ndarray:
    """Cell-adjusted mean of y per group: mean of (y - cell mean) + grand mean (direct standardisation)."""
    ok = np.isfinite(y)
    y, cell, g = y[ok], cell[ok], groupvar[ok]
    cell = np.unique(cell, return_inverse=True)[1]
    cm = np.bincount(cell, y) / np.bincount(cell)
    r = y - cm[cell]
    cnt = np.bincount(g, minlength=n_groups)
    return np.bincount(g, r, n_groups) / np.maximum(cnt, 1) + y.mean(), cnt


# --- the 1500 per-move tables ---------------------------------------------------------------------------------------

def move_table(split: str) -> pl.DataFrame:
    """Per own move (ply < MAX_PLY): T2 features + decision-position counts by R0 and R1."""
    gfile = V0 / ("blunder_games_1500_train.parquet" if split == "train" else "games_1500_val.parquet")
    name = "blunder_games_1500_train" if split == "train" else "games_1500_val"
    g = pl.read_parquet(gfile, columns=["game_id", "moves"])
    keys, lens = annotate(name, [gfile], cap=80)
    excl = split == "train"
    a0 = r0_counts(np.asarray(keys), lens, exclude_self=excl)
    a1, l1 = r1_counts(g["moves"], cap=80, exclude_self=excl)
    assert (l1 == lens).all()
    n_book = N_TRAIN - 1 if excl else N_TRAIN
    d0, d1 = decision_counts(a0, lens, n_book), decision_counts(a1, lens, n_book)
    gi, ply = ply_index(lens)
    cnt = pl.DataFrame({"game_id": g["game_id"].to_numpy()[gi], "ply": ply.astype(np.int64), "r0": d0, "r1": d1})
    cols = ["game_id", "ply", "think_s", "think_frac", "clock_before", "clock_frac", "base", "inc", "elo_self", "maia_played_lp",
            "maia_played_rank", "maia_entropy", "blunder"]
    parts = sorted((V0 / f"features/moves_f1_T2_1500_{split}").glob("*.parquet"))
    num = [x for x in cols if x != "game_id"]
    t = pl.concat([pl.read_parquet(f, columns=cols).with_columns(c(num).cast(pl.Float64)) for f in parts])
    t = t.with_columns(c("ply").cast(pl.Int64)).filter(c("ply") < MAX_PLY)
    t = t.join(cnt, on=["game_id", "ply"], how="inner")
    return t.with_columns(
        think_share=(c("think_s") / (c("clock_before") + c("inc"))).clip(0, 1),
        think_logit=c("think_frac").cast(pl.Float64),
        top1=(c("maia_played_rank") == 1).cast(pl.Float64), lp=c("maia_played_lp").cast(pl.Float64),
        blunder=c("blunder").cast(pl.Float64))


def cells_of(t: pl.DataFrame) -> np.ndarray:
    cf = np.digitize(t["clock_frac"].fill_nan(1.0).fill_null(1.0).to_numpy(), CLOCK_EDGES)
    elo = (t["elo_self"].to_numpy() // 100).astype(np.int64)
    tc = (t["base"].fill_nan(0).to_numpy() * 100 + t["inc"].fill_nan(0).to_numpy()).astype(np.int64)
    return codes(t["ply"].to_numpy(), elo, cf, tc)


def event_study(t: pl.DataFrame, T: int = 1) -> dict:
    """Mean think share and top-1 by ply relative to the book exit (first decision position with count < T)."""
    big = 10_000
    ex = t.group_by("game_id").agg(
        e0=pl.when(c("r0") < T).then(c("ply")).otherwise(big).min(),
        e1=pl.when(c("r1") < T).then(c("ply")).otherwise(big).min())
    ex = ex.filter(c("e0").is_between(6, 40) & c("e1").is_between(6, 40))
    tt = t.join(ex, on="game_id")
    out = {}
    for name, sel, ref in (("same_exit_rel_exit", c("e0") == c("e1"), "e0"),
                           ("transposed_rel_R1_exit", c("e0") >= c("e1") + 4, "e1"),
                           ("transposed_rel_R0_exit", c("e0") >= c("e1") + 4, "e0")):
        x = tt.filter(sel).with_columns(rel=c("ply") - c(ref)).filter(c("rel").is_between(-6, 8))
        r = x.group_by("rel").agg(n=pl.len(), think_share=c("think_share").mean(), top1=c("top1").mean(),
                                  think_s=c("think_s").mean()).sort("rel")
        out[name] = {"games": int(x["game_id"].n_unique()), **{k: r[k].to_list() for k in r.columns}}
    return out


def transposition_test(t: pl.DataFrame, cell: np.ndarray, lo: int = 16) -> dict:
    r0, r1 = t["r0"].to_numpy(), t["r1"].to_numpy()
    state = np.full(len(r0), 3)
    state[(r0 >= lo) & (r1 >= lo)] = 0      # in book by both keys
    state[(r0 >= lo) & (r1 <= 1)] = 1       # transposed in: common position, rare route
    state[r0 <= 1] = 2                      # out of book by both
    keep = state < 3
    out = {"rule": f"in: R0 >= {lo} and R1 >= {lo}; transposed: R0 >= {lo}, R1 <= 1; out: R0 <= 1",
           "n": {k: int((state == i).sum()) for i, k in enumerate(("in", "transposed", "out"))}}
    # transposed plies sit at a narrow ply range: restrict the comparison to cells that hold transposed plies
    has_t = np.zeros(cell.max() + 1, bool)
    has_t[cell[state == 1]] = True
    keep &= has_t[cell]
    for y in TARGETS:
        v = t[y].to_numpy().astype(float)
        m, n = adjusted_means(v[keep], cell[keep], state[keep], 3)
        idx = (m[1] - m[0]) / (m[2] - m[0]) if abs(m[2] - m[0]) > 1e-12 else float("nan")
        out[y] = {"in": float(m[0]), "transposed": float(m[1]), "out": float(m[2]), "position_index": float(idx)}
    return out


def moves_cmd():
    res = {}
    for split in ("val", "train"):
        t = move_table(split)
        t = t.filter(c("ply") >= 1)  # ply 0 is the start position for everyone
        cell = cells_of(t)
        b0, b1 = bucket(t["r0"].to_numpy()), bucket(t["r1"].to_numpy())
        if split == "val":   # heldout: counts 0 and 1 are not told apart in the table (Bloom / count >= 2)
            b0, b1 = np.maximum(b0, 1), np.maximum(b1, 1)
        r = {"games": int(t["game_id"].n_unique()), "moves": t.height,
             "self_excluded": split == "train",
             "share_R0_bucket_ne_R1": float((b0 != b1).mean()),
             "share_R0_gt_R1_by_ply": t.with_columns(d=(pl.Series(b0) != pl.Series(b1))).group_by("ply")
             .agg(c("d").mean()).sort("ply")["d"].round(4).to_list()}
        r["horse_race"] = {y: horse_race(t[y].to_numpy().astype(float), cell, b0, b1, d) for y, d in TARGETS.items()}
        tb = t.with_columns(b0=pl.Series(b0), b1=pl.Series(b1))
        r["raw_curves_plies_8_20"] = {"bucket": BUCKET_LABELS, "bucket_x": BUCKET_X,
                                      **{y: raw_curve(tb, y) for y in (*TARGETS, "think_s")}}
        r["transposition_test"] = transposition_test(t, cell)
        r["event_study"] = event_study(t)
        res[split] = r
        print(split, {k: v for k, v in r.items() if k not in ("raw_curves_plies_8_20", "event_study", "share_R0_gt_R1_by_ply")},
              flush=True)
    return res


def raw_curve(x: pl.DataFrame, y: str, lo: int = 8, hi: int = 20) -> dict:
    """Unadjusted mean of y by R0 and R1 bucket at plies lo..hi (chart-friendly)."""
    out = {}
    for key, b in (("R0", "b0"), ("R1", "b1")):
        r = x.filter(c("ply").is_between(lo, hi)).group_by(b).agg(n=pl.len(), m=c(y).mean()).sort(b)
        d = {int(k): (float(m), int(n)) for k, n, m in r.select(b, "n", "m").rows()}
        out[key] = [d[k][0] if k in d and d[k][1] >= 200 else None for k in range(12)]
    return out


# --- every band: think time on the heldout eval games ---------------------------------------------------------------

def bands_cmd(debug: bool = False):
    """book.py's heldout stratified eval games (10k per band): exact R0 probe counts and R1 eval counts."""
    book, pb = V0 / "book", V0 / "posbook"
    g = pl.read_parquet(book / "eval_games.parquet").with_row_index("row")
    mo_counts = np.load(book / "eval_counts.npy")[:, 0].astype(np.int64)
    mo_off = offsets(np.load(book / "eval_lens.npy"))
    probe = pl.read_parquet(pb / "probe.parquet", columns=["row", "game_id", "band"])
    rows = probe["row"].to_numpy()
    pc = np.load(pb / "full/probe_counts.npy")[:, 0].astype(np.int64)
    lens = np.load(pb / "full/probe_lens.npy")
    a1 = np.concatenate([mo_counts[mo_off[r]:mo_off[r + 1]] for r in rows])
    d0, d1 = decision_counts(pc, lens), decision_counts(a1, lens)
    gi, ply = ply_index(lens)
    mv = pl.read_parquet(book / "eval_moves.parquet", columns=["game_id", "clocks"])
    meta = g.filter(c("src") == "heldout").select("game_id", "band", "base", "inc", "white_elo", "black_elo")
    p = probe.select("game_id").join(meta, on="game_id", how="left", maintain_order="left").join(
        mv, on="game_id", how="left", maintain_order="left")
    # think per ply: mover's clock two plies back (or base) - clock after + inc
    clk = p["clocks"].list.head(80)
    cl_len = clk.list.len().fill_null(0).to_numpy()
    ok_game = (cl_len == lens) & (lens > 0) & (clk.list.eval(pl.element().is_null().any()).list.first()
                                                .fill_null(True).to_numpy() == False)  # noqa: E712
    base = p["base"].to_numpy().astype(float)[gi]
    inc = p["inc"].to_numpy().astype(float)[gi]
    good = ok_game[gi]
    flat_full = np.full(len(gi), np.nan)
    # flat clocks aligned to probe plies (games whose clock list matches the probe length)
    flat_ok = clk.filter(pl.Series(ok_game)).explode().cast(pl.Float64).to_numpy()
    flat_full[good] = flat_ok
    before = np.where(ply >= 2, np.r_[np.nan, np.nan, flat_full[:-2]], base)
    think = before - flat_full + inc
    share = np.clip(think / (before + inc), 0, 1)
    q = np.clip((think + 0.5) / (before + inc), 1e-4, 1 - 1e-4)
    logit = np.log(q / (1 - q))
    elo = np.where(ply % 2 == 0, p["white_elo"].to_numpy()[gi], p["black_elo"].to_numpy()[gi])
    band = p["band"].to_numpy()[gi]
    sel = (ply >= 1) & (ply < MAX_PLY) & np.isfinite(share) & (think >= 0)
    t = pl.DataFrame({"ply": ply[sel], "band": band[sel], "elo": elo[sel], "base": base[sel], "inc": inc[sel],
                      "clock_frac": (before / base)[sel], "think_share": share[sel], "think_logit": logit[sel], "r0": d0[sel], "r1": d1[sel]})
    t = t.with_columns(b0=pl.Series(np.maximum(bucket(t["r0"].to_numpy()), 1)),
                       b1=pl.Series(np.maximum(bucket(t["r1"].to_numpy()), 1)))
    if debug:
        return t
    out = {"games": int(ok_game.sum()), "plies": t.height, "by_band": {}}
    for bnd in (800, 1200, 1500, 1800, 2100, 2400):
        x = t.filter(c("band").is_between(bnd - 100, bnd + 100))
        cell = codes(x["ply"].to_numpy(), np.digitize(x["clock_frac"].to_numpy(), CLOCK_EDGES),
                     (x["base"].to_numpy() * 100 + x["inc"].to_numpy()).astype(np.int64), x["elo"].to_numpy() // 100)
        y = x["think_logit"].to_numpy()
        hr = horse_race(y, cell, x["b0"].to_numpy(), x["b1"].to_numpy(), "think logit")
        m0, n0 = adjusted_means(y, cell, x["b0"].to_numpy(), 12)
        m1, n1 = adjusted_means(y, cell, x["b1"].to_numpy(), 12)
        out["by_band"][bnd] = {"moves": x.height,
                               "partial_r2": {k[11:]: v for k, v in hr.items() if k.startswith("partial_r2_")},
                               "curve_R0": [float(m0[k]) if n0[k] >= 200 else None for k in range(12)],
                               "curve_R1": [float(m1[k]) if n1[k] >= 200 else None for k in range(12)],
                               "raw_think_share_plies_8_20": raw_curve(x, "think_share")}
        print(bnd, out["by_band"][bnd]["partial_r2"], flush=True)
    out["bucket"], out["bucket_x"] = BUCKET_LABELS, BUCKET_X
    out["note"] = "bands are band +-100 (the corpus band column); counts 0 and 1 merged (bucket '1' = '<2')"
    return out


# --- noise floor by popularity ----------------------------------------------------------------------------------------

def noise_cmd(plies=(6, 30)):
    """Self-agreement at repeated (player, position) visits, by the position's R0 train count (heldout 3+0)."""
    from doppel_lab.book import move_ids
    files = sorted((CORPUS / "heldout").glob("*.parquet"))
    keys_all, lens_all = annotate("heldout_cap40", files, cap=40)
    koff = offsets(lens_all)
    tk, tc = load_table(R0_DIR)
    lo, hi = plies
    parts, t0, n3 = [], 0, 0
    for f in files:
        g = pl.read_parquet(f, columns=["white", "black", "base", "inc", "moves"]).with_row_index("t", offset=t0)
        t0 += g.height
        g = g.filter((c("base") == 180) & (c("inc") == 0)).with_columns(c("moves").list.head(40))
        n3 += g.height
        t = g["t"].to_numpy()
        lens = lens_all[t]
        ids, moff = move_ids(g["moves"])
        assert (np.diff(moff) == lens).all()
        gi, ply = ply_index(lens)
        # decision position of the move at ply k (lo <= k < hi) = key after ply k-1
        sel = np.nonzero((ply >= lo) & (ply < hi))[0]
        dec = np.asarray(keys_all[koff[t[gi[sel]]] + ply[sel] - 1])
        side = ply[sel] % 2
        names = np.where(side == 0, g["white"].str.to_lowercase().to_numpy()[gi[sel]],
                         g["black"].str.to_lowercase().to_numpy()[gi[sel]])
        parts.append(pl.DataFrame({"player": names, "key": dec, "mv": ids[sel]}).with_columns(
            c("player").cast(pl.Categorical)))
        print(f.name, n3, flush=True)
    x = pl.concat(parts).with_columns(c("player").to_physical().alias("player"))
    cnt = lookup(tk, tc, x["key"].to_numpy())
    x = x.with_columns(b=pl.Series(np.maximum(bucket(cnt), 1).astype(np.int8)))
    per = (x.group_by("player", "key", "mv").agg(n=pl.len(), b=c("b").first())
           .with_columns(c("n").cast(pl.Float64)))
    del x
    pk = per.group_by("player", "key").agg(N=c("n").sum(), s2=(c("n") * (c("n") - 1)).sum(), b=c("b").first())
    pk = pk.filter(c("N") >= 2)
    # population agreement at the same positions, between different players
    rep_keys = pk.select("key").unique()
    per = per.join(rep_keys, on="key")
    pop = per.group_by("key", "mv").agg(n=c("n").sum(), own2=(c("n") ** 2).sum())
    pt = pop.group_by("key").agg(N=c("n").sum(), sq=(c("n") ** 2).sum(), own=c("own2").sum())
    pn = per.group_by("key", "player").agg(Np=c("n").sum()).group_by("key").agg(sqp=(c("Np") ** 2).sum())
    pt = pt.join(pn, on="key").join(per.select("key", "b").unique("key"), on="key")
    selfb = pk.group_by("b").agg(pairs=pl.len(), visits=c("N").sum(),
                                 self_agreement=c("s2").sum() / (c("N") * (c("N") - 1)).sum()).sort("b")
    popb = pt.filter(c("N") ** 2 > c("sqp")).group_by("b").agg(
        population_agreement=(c("sq") - c("own")).sum() / (c("N") ** 2 - c("sqp")).sum()).sort("b")
    tab = selfb.join(popb, on="b", how="left").sort("b")
    out = {"plies": list(plies), "games_3p0": int(n3), "bucket": BUCKET_LABELS, "bucket_x": BUCKET_X,
           "note": "R0 key; bucket '1' = train count < 2; heldout players only (disjoint from train)",
           "rows": [{"bucket": BUCKET_LABELS[int(r["b"])], **{k: r[k] for k in ("pairs", "visits", "self_agreement",
                                                                                "population_agreement")}}
                    for r in tab.to_dicts()]}
    for r in out["rows"]:
        print(r, flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["moves", "bands", "noise"])
    a = ap.parse_args()
    if a.cmd == "moves":
        save_section("behaviour_1500", moves_cmd())
    elif a.cmd == "bands":
        save_section("behaviour_bands", bands_cmd())
    else:
        save_section("noise_floor_by_popularity", noise_cmd())

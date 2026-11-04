"""Game-level realism metrics: sampled games vs held-out human games, per rating band.

Every source is treated identically:
- Adjudication: a game "ends" at the first ply where |eval| >= 600 cp holds for 6 consecutive plies
  (neither baseline resigns; humans do). Moves after that point are ignored.
- Per move (mover's view, evals clipped to +/-1000 cp): CPL = max(0, before - after);
  blunder = drop in Lichess winning chances >= 0.3, wc(cp) = 2 / (1 + exp(-0.00368208 cp)) - 1.
- Human games are split in two halves (A, B); A is the reference, B gives the noise floor.

Per player-game stats: acpl, blunders, burst (max blunders in any 10 own moves), and
acpl in each half of the player's moves (their correlation is the "per-game tendency" signal).
Per game: length (adjudicated plies), ending class, result.

Usage: python -m doppel_lab.metrics --games a.parquet ... --evals a.parquet ... --out results.json
"""

import argparse
import hashlib
import json
from collections import Counter

import chess
import numpy as np
import polars as pl
from scipy import stats

ADJ_CP, ADJ_RUN, CLIP = 600, 6, 1000
BLUNDER = 0.3
OPEN_PLIES = (6, 10)
SCALARS = ("acpl", "blunders", "burst", "length")


def wc(cp):
    return 2.0 / (1.0 + np.exp(-0.00368208 * cp)) - 1.0


def adjudicate(ev: np.ndarray) -> int:
    """Index of the last ply that counts (evals[k] is after ply k; evals[0] is the start)."""
    big = np.abs(ev) >= ADJ_CP
    run = 0
    for k in range(1, len(ev)):
        run = run + 1 if big[k] else 0
        if run == ADJ_RUN:
            return k - ADJ_RUN + 1
    return len(ev) - 1


def game_stats(row, ev):
    ev = np.clip(np.asarray(ev, dtype=float), -CLIP, CLIP)
    end = adjudicate(ev)
    sign = np.where(np.arange(end) % 2 == 0, 1.0, -1.0)  # ply k+1 is moved by White iff k even
    before, after = ev[:end] * sign, ev[1:end + 1] * sign
    cpl = np.maximum(0.0, before - after)
    drop = wc(before) - wc(after)
    bl = drop >= BLUNDER
    players = []
    for side in (0, 1):
        c, b = cpl[side::2], bl[side::2]
        if len(c) < 6:
            continue
        h = len(c) // 2
        burst = max(int(b[i:i + 10].sum()) for i in range(max(1, len(b) - 9)))
        players.append({"acpl": float(c.mean()), "blunders": int(b.sum()), "burst": burst,
                        "acpl_h1": float(c[:h].mean()), "acpl_h2": float(c[h:].mean()),
                        "bl_h1": float(b[:h].mean()), "bl_h2": float(b[h:].mean()),
                        "bl_seq": b.astype(int).tolist(),
                        "mistakes": int((drop[side::2] >= 0.2).sum()),
                        "inaccuracies": int((drop[side::2] >= 0.1).sum())})
    n = len(row["moves"])
    mated, mv_feats = _replay(row["moves"], end)
    if end < n:
        ending = "decided"  # adjudicated before the game actually stopped
    elif mated:
        ending = "mate"
    elif row["termination"] == "Time forfeit":
        ending = "time"
    elif row["result"] == "1/2-1/2":
        ending = "draw"
    elif row["result"] == "*":
        ending = "cap"
    else:
        ending = "resign"  # stopped (resigned) before the eval crossed the adjudication line
    final = ev[end]
    result = row["result"] if end == n else ("1-0" if final > 0 else "0-1")
    return {"length": end, "ending": ending, "result": result,
            "open6": " ".join(row["moves"][:6]), "open10": " ".join(row["moves"][:10]),
            "players": players, "mv_feats": mv_feats}


def _replay(moves, end):
    """(final position is mate, move-style features over the counted plies)."""
    b = chess.Board()
    checks = caps = queen = pawn20 = 0
    castle = [0, 0]
    for k, m in enumerate(moves):
        mv = chess.Move.from_uci(m)
        if k < end:
            side = k % 2
            pt = b.piece_type_at(mv.from_square)
            caps += b.is_capture(mv)
            queen += pt == chess.QUEEN
            pawn20 += k < 20 and pt == chess.PAWN
            if b.is_castling(mv):
                castle[side] = k + 1
        b.push(mv)
        if k < end:
            checks += b.is_check()
    n = max(1, min(end, len(moves)))
    return b.is_checkmate(), [checks / n, caps / n, queen / n, pawn20, castle[0], castle[1]]


ENDINGS = ("decided", "mate", "time", "draw", "cap", "resign")


def features(r):
    """Fixed-length per-game vector for the discriminator (no clock features: Maia-3 has none)."""
    keys = ("acpl", "blunders", "mistakes", "inaccuracies", "burst", "acpl_h1", "acpl_h2")
    pv = []
    for i in range(2):
        p = r["players"][i] if i < len(r["players"]) else None
        pv += [p[k] if p else -1.0 for k in keys]
    return pv + [r["length"]] + [float(r["ending"] == e) for e in ENDINGS] + \
        [float(r["result"] == x) for x in ("1-0", "0-1", "1/2-1/2")] + r["mv_feats"]


N_PLAYER_FEATS = 14  # features() starts with 2 x 7 per-player move-quality stats


def discriminator(ref, other, rng, folds=5, moves_only=False):
    """5-fold CV accuracy of gradient boosting on per-game features, classes balanced by subsampling.
    moves_only keeps the per-player error stats and move-style features, dropping length, ending and
    result (which give away bots that never resign and have no clock)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    n = min(len(ref), len(other))
    a = [ref[i] for i in rng.choice(len(ref), n, replace=False)]
    b = [other[i] for i in rng.choice(len(other), n, replace=False)]
    X = np.array([features(r) for r in a + b], dtype=float)
    if moves_only:
        X = np.concatenate([X[:, :N_PLAYER_FEATS], X[:, -6:]], axis=1)
    y = np.array([0] * n + [1] * n)
    acc = cross_val_score(HistGradientBoostingClassifier(max_iter=200, random_state=0), X, y, cv=folds)
    return {"n_per_class": n, "accuracy": float(acc.mean()), "accuracy_std": float(acc.std())}


def entropy_bits(items, n, rng):
    sample = rng.choice(np.asarray(items, dtype=object), size=min(n, len(items)), replace=False)
    p = np.array(list(Counter(sample).values()), dtype=float)
    p /= p.sum()
    return float(-(p * np.log2(p)).sum())


def bl_lift(seqs, k=2):
    """P(blunder within the next k own moves | blunder now) / P(blunder)."""
    hits = trials = 0
    allb = sum(sum(s) for s in seqs)
    alln = sum(len(s) for s in seqs)
    for s in seqs:
        for i, x in enumerate(s[:-k] if k else s):
            if x:
                trials += 1
                hits += any(s[i + 1:i + 1 + k])
    if not trials or not allb:
        return float("nan")
    base = 1 - (1 - allb / alln) ** k
    return (hits / trials) / base


def summarize(recs, rng):
    pl_rows = [p for r in recs for p in r["players"]]
    out = {"games": len(recs), "player_games": len(pl_rows)}
    for k in ("acpl", "blunders", "burst"):
        v = np.array([p[k] for p in pl_rows], dtype=float)
        out[k] = {"mean": float(v.mean()), "std": float(v.std()), "cv": float(v.std() / v.mean()),
                  "dispersion": float(v.var() / v.mean())}
    v = np.array([r["length"] for r in recs], dtype=float)
    out["length"] = {"mean": float(v.mean()), "std": float(v.std())}
    h1, h2 = np.array([p["acpl_h1"] for p in pl_rows]), np.array([p["acpl_h2"] for p in pl_rows])
    out["acpl_half_corr"] = float(stats.spearmanr(h1, h2).statistic)
    b1, b2 = np.array([p["bl_h1"] for p in pl_rows]), np.array([p["bl_h2"] for p in pl_rows])
    out["blunder_half_corr"] = float(stats.spearmanr(b1, b2).statistic)
    out["blunder_lift"] = bl_lift([p["bl_seq"] for p in pl_rows])
    for n in OPEN_PLIES:
        out[f"open{n}_entropy_bits"] = entropy_bits([r[f"open{n}"] for r in recs], 1000, rng)
    out["endings"] = {k: v / len(recs) for k, v in Counter(r["ending"] for r in recs).items()}
    out["results"] = {k: v / len(recs) for k, v in Counter(r["result"] for r in recs).items()}
    return out


def distances(ref, other):
    d = {}
    for k in SCALARS:
        if k == "length":
            a, b = [r[k] for r in ref], [r[k] for r in other]
        else:
            a = [p[k] for r in ref for p in r["players"]]
            b = [p[k] for r in other for p in r["players"]]
        d[k] = {"w1": float(stats.wasserstein_distance(a, b)), "ks": float(stats.ks_2samp(a, b).statistic)}
    ea, eb = Counter(r["ending"] for r in ref), Counter(r["ending"] for r in other)
    cats = set(ea) | set(eb)
    d["endings_tv"] = 0.5 * sum(abs(ea[c] / len(ref) - eb[c] / len(other)) for c in cats)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="+", required=True)
    ap.add_argument("--evals", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    cols = ["game_id", "band", "moves", "result", "termination", "source"]
    games = pl.concat([pl.read_parquet(f).with_columns(
        pl.col("source") if "source" in pl.read_parquet_schema(f) else pl.lit("human").alias("source")
    ).select(cols) for f in args.games])
    evals = pl.concat([pl.read_parquet(f) for f in args.evals])
    df = games.join(evals, on="game_id", how="inner")

    recs = {}
    for row in df.iter_rows(named=True):
        src = row["source"]
        if src == "human":  # deterministic A/B halves
            src = "human_" + "AB"[hashlib.sha1(row["game_id"].encode()).digest()[0] % 2]
        recs.setdefault((src, row["band"]), []).append(game_stats(row, row["evals"]))

    sources = sorted({s for s, _ in recs})
    bands = sorted({b for _, b in recs})
    result = {"config": {"adjudicate_cp": ADJ_CP, "adjudicate_run": ADJ_RUN, "clip_cp": CLIP,
                         "blunder_wc_drop": BLUNDER}, "bands": {}}
    for band in bands:
        ref = recs.get(("human_A", band))
        entry = {}
        for src in sources:
            r = recs.get((src, band))
            if not r:
                continue
            entry[src] = {"summary": summarize(r, rng)}
            if ref is not None and src != "human_A":
                entry[src]["vs_human_A"] = distances(ref, r)
                entry[src]["vs_human_A"]["discriminator"] = discriminator(ref, r, rng)
                entry[src]["vs_human_A"]["discriminator_moves_only"] = discriminator(ref, r, rng, moves_only=True)
        result["bands"][band] = entry
    json.dump(result, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}: sources {sources}, bands {bands}")


if __name__ == "__main__":
    main()

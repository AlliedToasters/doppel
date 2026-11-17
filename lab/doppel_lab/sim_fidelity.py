"""Simulation fidelity check (coaching plan step 1): do a player's Doppel's simulated games match the player's
real games on blunder rate, game length and score? Baselines: plain Maia-3 at the player's rating in the same
pool, the rating average, and the player's own first 40 games.

    python -m doppel_lab.sim_fidelity select                 # CPU: players, z inputs, real reference games
    python -m doppel_lab.sim_fidelity sim --gpu-ok           # GPU (warden): z for the pool, then self-play
    python -m doppel_lab.sim_fidelity sim --limit 3          # CPU smoke: 3 focal players x 4 games per condition
    nice -n 15 python -m doppel_lab.analyze data/sim/sim_games.parquet data/sim/sim_evals.parquet --nodes 20000 --workers 16
    nice -n 15 python -m doppel_lab.analyze data/sim/real_games.parquet data/sim/real_evals.parquet --nodes 20000 --workers 16
    python -m doppel_lab.sim_fidelity report                 # CPU -> lab/results/sim_fidelity.json

Players: the 1500-band valrich players (held-out 2026-10, never trained on). Focal: 200 of those with >= 80 band
games (>= 20 plies); z from their first 40 (v0g_b, as on Allie); real reference = their games 41..100. Pool: every
valrich player with >= 41 games, each with their own z. Opponents are drawn from the pool within +-100 Elo
(widened in steps of 100 if empty), never the focal player.

Conditions, each player x GAMES games, half with each colour:
  doppel_t1   focal = Doppel (Maia-3 + residual with the focal z), T = 1.0
  doppel_t08  same at T = 0.8 (both sides)
  maia_t1     focal = plain Maia-3 at the focal rating, T = 1.0
Opponents are always pool Doppels at the condition's temperature. Real Elo inputs on both sides (each player's
Elo in their 41st game). No clock: the time-trouble share of real blunders and losses on time cannot appear.

Endings: mate, stalemate, insufficient material, threefold repetition, 50-move rule, or MAX_PLIES. Resignation is
applied after annotation: the first ply from which |eval| >= 500 cp for 4 consecutive positions ends the game,
lost for the side behind; at MAX_PLIES without that, |eval| >= 200 cp decides, else a draw. Real games keep their
recorded result and length (time forfeits included).

Blunder: wc_loss >= 0.2 (blunder2, Maia's unit) and >= 0.3 (Lichess), same Stockfish budget for real and simulated
games (20k nodes; §7 uses 50k), own moves at ply >= 10 up to the game's end (or resignation point).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import chess
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/sim"
CORPUS = ROOT / "data/corpus/2026-10/heldout"
RUN = "v0g_b"
BAND = 1500
N_FOCAL, MIN_GAMES, CTX, REF = 200, 80, 40, 60
GAMES = 40
MAX_PLIES = 200
CONDS = {"doppel_t1": (True, 1.0), "doppel_t08": (True, 0.8), "maia_t1": (False, 1.0)}
c = pl.col


# ------------------------------------------------------------------ select (CPU)
def select(seed: int = 0):
    from doppel_lab.v0e_data import Split
    sp = Split("valrich", BAND)
    players = sp.players
    # each pid's sides, time-ordered (Split._order sorts by player then t)
    first = {}
    for i in range(len(sp.hg)):
        p = int(sp.pid[sp.hg[i], sp.hc[i]])
        first.setdefault(p, []).append(i)
    pool = [p for p, idx in first.items() if len(idx) > CTX]
    g = (pl.scan_parquet(str(CORPUS / "part-*.parquet")).filter((c("band") == BAND) & (c("moves").list.len() >= 20))
         .select("game_id", "white", "black", "white_elo", "black_elo", "result", "termination", "moves").collect()
         .with_row_index("t"))
    names = {str(players[p]): p for p in pool}
    sides = pl.concat([g.select("t", "game_id", player=c(col).str.to_lowercase(), col=pl.lit(i))
                       for i, col in enumerate(("white", "black"))]).filter(c("player").is_in(list(names)))
    sides = sides.sort("t").with_columns(j=pl.int_range(pl.len()).over("player"))
    n_by = sides.group_by("player").len()
    elig = sorted(n_by.filter(c("len") >= MIN_GAMES)["player"].to_list())
    rng = np.random.default_rng(seed)
    focal = sorted(rng.choice(elig, min(N_FOCAL, len(elig)), replace=False).tolist())
    # real games of focal players: first 40 (j < 40) and reference (40 <= j < 40 + REF)
    fs = sides.filter(c("player").is_in(focal) & (c("j") < CTX + REF)).with_columns(
        part=pl.when(c("j") < CTX).then(pl.lit("first40")).otherwise(pl.lit("ref")))
    real = fs.join(g.drop("t"), on="game_id").select("game_id", "player", "col", "j", "part", "white_elo", "black_elo",
                                                     "result", "termination", "moves")
    OUT.mkdir(parents=True, exist_ok=True)
    real.write_parquet(OUT / "real_sides.parquet")
    real.unique("game_id").select("game_id", "moves").write_parquet(OUT / "real_games.parquet")
    # z input: the pool player's 41st side in valrich (history = their first 40)
    rows = []
    for p in pool:
        i = first[p][CTX]
        rows.append({"player": str(players[p]), "pid": p, "g": int(sp.hg[i]), "col": int(sp.hc[i]),
                     "focal": str(players[p]) in focal})
    pl.DataFrame(rows).write_parquet(OUT / "pool.parquet")
    print(f"pool {len(pool)} players; eligible focal {len(elig)}, chose {len(focal)}; real sides {real.height:,} "
          f"({real['game_id'].n_unique():,} games) -> {OUT}", flush=True)


# ------------------------------------------------------------------ sim (GPU)
def pool_z(model, device: str):
    import torch

    from doppel_lab.v0_train import to_dev
    from doppel_lab.v0e_data import Split, build, collate
    sp = Split("valrich", BAND)
    pool = pl.read_parquet(OUT / "pool.parquet")
    zs, elos = [], []
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for r in pool.iter_rows(named=True):
            s = build(sp, r["g"], r["col"], CTX, 1, rng)
            b = to_dev(collate([s]), device)
            mu, _ = model.posterior(b)
            zs.append(mu[0].float().cpu())
            elos.append(int(b["elo_self"][0]))
    return pool.with_columns(elo=pl.Series(elos)), torch.stack(zs)


def opponents(elo: np.ndarray, focal_i: int, n: int, rng) -> np.ndarray:
    for w in (100, 200, 300, 500, 10_000):
        cand = np.flatnonzero((np.abs(elo - elo[focal_i]) <= w) & (np.arange(len(elo)) != focal_i))
        if len(cand):
            return rng.choice(cand, n, replace=True)
    raise ValueError("empty pool")


def play(model, z, elo, games, device: str, batch: int = 2048):
    """games: list of dicts {w, b, res_w, res_b, T} (pool indices; residual on/off per side). Returns moves, end."""
    import torch

    from doppel_model.encoding import MOVE_INDEX, MOVES, board_planes, mirror_uci
    from doppel_lab.v0_model import HIST, NMOVES
    out = [None] * len(games)
    for s in range(0, len(games), batch):
        part = games[s:s + batch]
        boards = [chess.Board() for _ in part]
        planes = [[board_planes(b)] for b in boards]
        moves = [[] for _ in part]
        end = [None] * len(part)
        for ply in range(MAX_PLIES):
            act = [i for i in range(len(part)) if end[i] is None]
            if not act:
                break
            tok = np.zeros((len(act), 64, 12 * HIST + 1), np.float32)
            mask = np.zeros((len(act), NMOVES), bool)
            es, eo, zi, use, T = (np.zeros(len(act), np.int64), np.zeros(len(act), np.int64),
                                  np.zeros(len(act), np.int64), np.zeros(len(act), np.float32),
                                  np.zeros(len(act), np.float32))
            for r, i in enumerate(act):
                b, gm = boards[i], part[i]
                h = planes[i][-HIST:]
                h = [h[0]] * (HIST - len(h)) + h
                tok[r, :, :12 * HIST] = np.concatenate(h, 1)
                white = b.turn == chess.WHITE
                for m in b.legal_moves:
                    u = m.uci()
                    mask[r, MOVE_INDEX[u if white else mirror_uci(u)]] = True
                me, op = (gm["w"], gm["b"]) if white else (gm["b"], gm["w"])
                es[r], eo[r], zi[r] = elo[me], elo[op], me
                use[r] = gm["res_w"] if white else gm["res_b"]
                T[r] = gm["T"]
            with torch.no_grad():
                tk = torch.from_numpy(tok).to(device)
                se, oe = torch.from_numpy(es).to(device), torch.from_numpy(eo).to(device)
                mk = torch.from_numpy(mask).to(device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                    t_logits, _, _ = model.trunk(tk, se, oe)
                t_logp = t_logits.float().masked_fill(~mk, -1e9).log_softmax(-1)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                    res = model.residual(model.residual.board(tk), z[torch.from_numpy(zi).to(device)], se, oe, mk).float()
                lg = t_logp + torch.from_numpy(use).to(device)[:, None] * res
                lg = (lg / torch.from_numpy(T).to(device)[:, None]).masked_fill(~mk, -1e9)
                pick = torch.multinomial(torch.softmax(lg, -1), 1)[:, 0].cpu().numpy()
            for r, i in enumerate(act):
                b = boards[i]
                u = MOVES[pick[r]]
                u = u if b.turn == chess.WHITE else mirror_uci(u)
                mv = chess.Move.from_uci(u)
                if mv not in b.legal_moves:                      # promotion spelled without a piece, etc.
                    mv = next(m for m in b.legal_moves if m.uci()[:4] == u[:4])
                b.push(mv)
                moves[i].append(mv.uci())
                planes[i].append(board_planes(b))
                if b.is_checkmate():
                    end[i] = "mate"
                elif b.is_stalemate() or b.is_insufficient_material():
                    end[i] = "draw_rule"
                elif b.is_fifty_moves() or b.is_repetition(3):
                    end[i] = "draw_claim"
        for i in range(len(part)):
            out[s + i] = (moves[i], end[i] or "max_plies")
        print(f"games {min(s + batch, len(games))}/{len(games)}", flush=True)
    return out


def sim(device: str, limit: int = 0, seed: int = 0):
    import torch

    from doppel_lab.v0e_train import V0, build
    model = build(False, device, band=BAND)
    st = torch.load(V0 / "runs" / RUN / "ckpt.pt", map_location=device, weights_only=False)
    model.load_state_dict(st["model"], strict=False)
    model.eval()
    t0 = time.time()
    pool, z = pool_z(model, device)
    z = z.to(device)
    elo = pool["elo"].to_numpy()
    print(f"z for {len(pool)} pool players in {time.time() - t0:.0f}s", flush=True)
    fi = np.flatnonzero(pool["focal"].to_numpy())
    n_games = GAMES
    if limit:
        fi, n_games = fi[:limit], 4
    rng = np.random.default_rng(seed)
    games = []
    for cond, (res_on, T) in CONDS.items():
        for f in fi:
            opp = opponents(elo, f, n_games, rng)
            for k, o in enumerate(opp):
                fw = k % 2 == 0
                w, b = (f, o) if fw else (o, f)
                games.append({"cond": cond, "focal": int(f), "focal_col": 0 if fw else 1, "opp": int(o), "w": int(w),
                              "b": int(b), "T": T, "res_w": 1.0 if (res_on or not fw) else 0.0,
                              "res_b": 1.0 if (res_on or fw) else 0.0})
    t0 = time.time()
    res = play(model, z, elo, games, device)
    names = pool["player"].to_list()
    df = pl.DataFrame({
        "game_id": [f"sim{seed}_{i}" for i in range(len(games))],
        "cond": [g["cond"] for g in games], "player": [names[g["focal"]] for g in games],
        "col": [g["focal_col"] for g in games], "opp": [names[g["opp"]] for g in games],
        "focal_elo": [int(elo[g["focal"]]) for g in games], "opp_elo": [int(elo[g["opp"]]) for g in games],
        "moves": [r[0] for r in res], "end": [r[1] for r in res]})
    out = OUT / ("sim_games_smoke.parquet" if limit else "sim_games.parquet")
    df.write_parquet(out)
    print(f"{len(games):,} games, {df['moves'].list.len().sum():,} plies in {time.time() - t0:.0f}s -> {out}; "
          f"ends {df['end'].value_counts().sort('end').rows()}", flush=True)


# ------------------------------------------------------------------ report (CPU)
def side_metrics(moves: list, evals: list, col: int, recorded: tuple | None):
    """Per side: (blunders2, blunders3, own moves scored, length, score). recorded = (result_str, plies) for real
    games (kept as is); None for simulated games (resignation / adjudication from the evals)."""
    from doppel_lab.blunder import labels
    ev = np.clip(np.array(evals, dtype=np.float64), -10000, 10000)
    n = len(moves)
    if recorded is None:
        end, score_w = n, None
        big = np.abs(ev) >= 500
        run = 0
        for k in range(len(ev)):
            run = run + 1 if big[k] else 0
            if run == 4 and k < n:
                end, score_w = k, 1.0 if ev[k] > 0 else 0.0
                break
        if score_w is None:
            b = chess.Board()
            for u in moves:
                b.push_uci(u)
            if b.is_checkmate():
                score_w = 0.0 if b.turn == chess.WHITE else 1.0
            elif n < MAX_PLIES:
                score_w = 0.5
            else:
                score_w = 1.0 if ev[-1] >= 200 else 0.0 if ev[-1] <= -200 else 0.5
    else:
        res, end = recorded
        score_w = {"1-0": 1.0, "0-1": 0.0}.get(res, 0.5)
    lab = labels(list(ev))
    own = np.arange(col, min(end, n), 2)
    own = own[own >= 10]
    return (int(lab["blunder2"][own].sum()), int(lab["blunder"][own].sum()), len(own), int(end),
            score_w if col == 0 else 1.0 - score_w)


def side_rows(tag: str = "") -> pl.DataFrame:
    """One row per (player, source, game side): blunder counts, own moves scored, length, score."""
    sims = pl.read_parquet(OUT / f"sim_games{tag}.parquet").join(
        pl.read_parquet(OUT / f"sim_evals{tag}.parquet"), on="game_id")
    real = pl.read_parquet(OUT / "real_sides.parquet").join(pl.read_parquet(OUT / "real_evals.parquet"), on="game_id")
    rows = []
    for r in sims.iter_rows(named=True):
        b2, b3, nm, L, s = side_metrics(r["moves"], r["evals"], r["col"], None)
        rows.append((r["player"], r["cond"], b2, b3, nm, L, s))
    for r in real.iter_rows(named=True):
        b2, b3, nm, L, s = side_metrics(r["moves"], r["evals"], r["col"], (r["result"], len(r["moves"])))
        rows.append((r["player"], r["part"], b2, b3, nm, L, s))
    return pl.DataFrame(rows, schema=["player", "src", "b2", "b3", "nmov", "len", "score"], orient="row")


def report(tag: str = ""):
    d = side_rows(tag)
    per = d.group_by("player", "src").agg(
        games=pl.len(), b2_per_move=c("b2").sum() / c("nmov").sum(), b3_per_move=c("b3").sum() / c("nmov").sum(),
        b2_per_game=c("b2").mean(), length=c("len").mean(), score=c("score").mean())
    elo = pl.read_parquet(OUT / "pool.parquet").select("player")  # rating for the rating-average baseline
    rel = pl.read_parquet(OUT / "real_sides.parquet").filter(c("part") == "first40").with_columns(
        e=pl.when(c("col") == 0).then(c("white_elo")).otherwise(c("black_elo"))).group_by("player").agg(
        elo=c("e").median())
    ref = per.filter(c("src") == "ref").join(rel, on="player")
    metrics = ["b2_per_move", "b3_per_move", "b2_per_game", "length", "score"]
    res = {"n_players": ref.height, "ref_games_median": float(ref["games"].median()), "sources": {}}
    E = ref["elo"].to_numpy()
    for src in ["first40", "rating_avg", *CONDS]:
        out = {}
        for m in metrics:
            y = ref.sort("player")[m].to_numpy()
            if src == "rating_avg":
                rs = ref.sort("player")
                e, yy = rs["elo"].to_numpy(), rs[m].to_numpy()
                x = np.array([yy[(np.abs(e - e[i]) <= 100) & (np.arange(len(e)) != i)].mean() for i in range(len(e))])
            else:
                p = per.filter(c("src") == src).join(ref.select("player"), on="player").sort("player")
                if p.height != ref.height:
                    continue
                x = p[m].to_numpy()
            ok = np.isfinite(x) & np.isfinite(y)
            x, yv = x[ok], y[ok]
            pear = float(np.corrcoef(x, yv)[0, 1]) if len(x) > 2 and x.std() > 0 else None
            rk = lambda a: np.argsort(np.argsort(a))
            spear = float(np.corrcoef(rk(x), rk(yv))[0, 1]) if len(x) > 2 and x.std() > 0 else None
            slope = float(np.polyfit(x, yv, 1)[0]) if len(x) > 2 and x.std() > 0 else None
            out[m] = {"pearson": pear, "spearman": spear, "bias_mean": float((x - yv).mean()),
                      "sim_mean": float(x.mean()), "real_mean": float(yv.mean()), "calib_slope_real_on_pred": slope}
        res["sources"][src] = out
    res["elo_range"] = [float(E.min()), float(E.max())] if len(E) else None
    path = ROOT / "lab/results" / f"sim_fidelity{tag}.json"
    path.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def forecast(tag: str = "", folds: int = 10, reps: int = 2000):
    """Coaching plan step 2 on blunder rates: predict each focal player's real per-move blunder rate in games 41-100
    from (a) their first 40 games alone, (b) the pooled mean, (c) the first 40 shrunk toward the pooled mean
    (empirical-Bayes beta-binomial), (d) the first 40 shrunk toward a simulation-based prior (prior mean = a + b * sim
    rate, fit out of fold), (e) a linear blend of first-40 and sim rates (out of fold). Same for plain Maia-3 sims as
    the control: if Maia adds as much as the Doppel, the gain is not personal. Score: out-of-fold mean squared error
    weighted by moves in games 41-100, with a paired player bootstrap for differences."""
    d = side_rows(tag)
    agg = d.group_by("player", "src").agg(k2=c("b2").sum(), k3=c("b3").sum(), n=c("nmov").sum())
    wide = None
    for src in ["first40", "ref", *CONDS]:
        t = agg.filter(c("src") == src).select("player", *(pl.col(x).alias(f"{x}_{src}") for x in ("k2", "k3", "n")))
        wide = t if wide is None else wide.join(t, on="player", how="inner")
    wide = wide.sort("player")
    rng = np.random.default_rng(0)
    fold = rng.permutation(np.arange(wide.height) % folds)
    res = {"n_players": wide.height, "folds": folds, "metrics": {}}
    for kk in ("k2", "k3"):
        kf, nf = wide[f"{kk}_first40"].to_numpy().astype(float), wide["n_first40"].to_numpy().astype(float)
        kr, nr = wide[f"{kk}_ref"].to_numpy().astype(float), wide["n_ref"].to_numpy().astype(float)
        yr, xf = kr / nr, kf / nf
        sim = {cnd: wide[f"{kk}_{cnd}"].to_numpy() / wide[f"n_{cnd}"].to_numpy() for cnd in CONDS}
        preds = {name: np.full(len(yr), np.nan) for name in
                 ["first40", "pooled_mean", "shrink_to_mean", *(f"sim_{c_}" for c_ in CONDS),
                  *(f"shrink_to_sim_{c_}" for c_ in CONDS), *(f"blend_{c_}" for c_ in CONDS)]}
        shrink_info = {}

        def tau_fit(k, n, mu):
            """Beta prior strength (pseudo-moves) by method of moments around prior means mu."""
            r = k / n
            v = np.average((r - mu) ** 2, weights=n) - np.average(mu * (1 - mu) / n, weights=n)
            return float(np.clip(np.mean(mu * (1 - mu)) / max(v, 1e-9) - 1, 1, 1e6))

        for f in range(folds):
            tr, te = fold != f, fold == f
            # targets for fitting priors use the first-40 data of training players (never games 41+ of test players);
            # the sim -> rate calibration is fit on training players' games 41+.
            preds["first40"][te] = xf[te]
            mu0 = np.average(xf[tr], weights=nf[tr])
            preds["pooled_mean"][te] = mu0
            tau0 = tau_fit(kf[tr], nf[tr], np.full(tr.sum(), mu0))
            preds["shrink_to_mean"][te] = (kf[te] + tau0 * mu0) / (nf[te] + tau0)
            shrink_info.setdefault("tau_mean", []).append(tau0)
            for cnd, xs in sim.items():
                b, a = np.polyfit(xs[tr], yr[tr], 1, w=np.sqrt(nr[tr]))
                preds[f"sim_{cnd}"][te] = a + b * xs[te]
                mu_tr = np.clip(a + b * xs[tr], 1e-4, 0.999)
                tau = tau_fit(kf[tr], nf[tr], mu_tr)
                mu_te = np.clip(a + b * xs[te], 1e-4, 0.999)
                preds[f"shrink_to_sim_{cnd}"][te] = (kf[te] + tau * mu_te) / (nf[te] + tau)
                shrink_info.setdefault(f"tau_{cnd}", []).append(tau)
                A = np.column_stack([np.ones(tr.sum()), xf[tr], xs[tr]])
                coef = np.linalg.lstsq(A * np.sqrt(nr[tr])[:, None], yr[tr] * np.sqrt(nr[tr]), rcond=None)[0]
                preds[f"blend_{cnd}"][te] = coef[0] + coef[1] * xf[te] + coef[2] * xs[te]
                shrink_info.setdefault(f"blend_{cnd}", []).append([round(float(x), 4) for x in coef])
        w = nr / nr.sum()
        err = {k: (p - yr) ** 2 for k, p in preds.items()}
        mse = {k: float((w * e).sum()) for k, e in err.items()}
        # noise floor: binomial sampling variance of games 41+ (no forecast can beat this)
        floor = float((w * (yr * (1 - yr) / nr)).sum())
        base = "shrink_to_mean"
        B = rng.integers(0, len(yr), (reps, len(yr)))
        boot = {}
        for k, e in err.items():
            diff = np.array([(w[b] * (e[b] - err[base][b])).sum() / w[b].sum() for b in B])
            boot[k] = [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]
        tot = float((w * (yr - np.average(yr, weights=nr)) ** 2).sum())
        res["metrics"]["blunder2_per_move" if kk == "k2" else "blunder3_per_move"] = {
            "mse_x1e4": {k: round(1e4 * v, 4) for k, v in mse.items()},
            "noise_floor_x1e4": round(1e4 * floor, 4), "total_var_x1e4": round(1e4 * tot, 4),
            "explainable_share_captured": {k: round((tot - v) / (tot - floor), 3) for k, v in mse.items()},
            "diff_vs_shrink_to_mean_x1e4_ci": {k: [round(1e4 * x, 4) for x in v] for k, v in boot.items()},
            "pearson": {k: round(float(np.corrcoef(p, yr)[0, 1]), 3) if np.std(p) > 0 else None for k, p in preds.items()},
            "prior_strength_and_blend": {k: (np.round(np.mean(v, axis=0), 4).tolist() if isinstance(v[0], list)
                                             else round(float(np.mean(v)), 1)) for k, v in shrink_info.items()},
        }
    res["note"] = ("Out-of-fold (10 folds). MSE weighted by moves in games 41-100; x1e4. explainable share = "
                   "(total variance - MSE) / (total variance - binomial noise floor). Differences vs shrink_to_mean: "
                   "negative = better; paired player bootstrap 95%. blend coefficients = [intercept, first40, sim].")
    path = ROOT / "lab/results" / f"sim_forecast{tag}.json"
    path.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["metrics"], indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("select")
    s = sub.add_parser("sim")
    s.add_argument("--gpu-ok", action="store_true")
    s.add_argument("--limit", type=int, default=0)
    r = sub.add_parser("report")
    r.add_argument("--tag", default="")
    fcp = sub.add_parser("forecast")
    fcp.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.cmd == "select":
        select()
    elif a.cmd == "sim":
        sim("cuda" if a.gpu_ok else "cpu", a.limit)
    elif a.cmd == "forecast":
        forecast(a.tag)
    else:
        report(a.tag)


if __name__ == "__main__":
    main()

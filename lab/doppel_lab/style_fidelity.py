"""Middlegame style fidelity: does a model's z reproduce each player's style traits at held-out positions? (GPU)

    python -m doppel_lab.style_fidelity --run v0f --gpu-ok          # warden -> lab/results/style_fidelity.json[run]
    python -m doppel_lab.style_fidelity --run v0f --limit 4         # CPU smoke: 4 target players

Move recall can't measure middlegame fidelity (SCIENCE.md §8, policy-filtered backoff: past ply 10 a near
position's move is the position's, not the player's). What players keep in the middlegame is style: steering,
time allocation, conformity to Maia (§8 style traits, split-half ρ ≈ 0.5). So score the model on those, per player.

Players: the 600 valrich target players of the band (48 games each, held out from training). Half A = their games
1-24, half B = 25-48. Positions: own decision plies >= 10 (and the >= 20 subset; "middlegame"). For each half-B
position, up to PER per game side, the model runs with z = the posterior mean from the player's 40 most recent
earlier games ("post") and with z = the prior mean ("prior": the model without the player). Children: every legal
move's resulting position, through the trunk (Maia-3 79M, real Elos, the opponent to move), so traits that depend
on the move are computed exactly as expectations under the model's move distribution:

  steer     entropy handed to the opponent (trunk entropy of the position after the move) minus the pooled fit of
            it on the faced entropy (quadratic) and ply: actual = the played move's child; model = E_p[child]
  conform   trunk log-prob of the move: actual = played; model = E_p[log p_trunk]
  alloc     per game side, Spearman of think fraction (think / (remaining + inc)) with faced entropy, positions with
            >= 10% of the base clock left; actual = real think times; model = the think head's median
Trait per player = mean over their game sides. Reported per trait, across players, Spearman with the actual
half-B trait for: post, prior, the trunk alone (Maia; steer and conform only), and actual half A (the player's
own earlier games: the history ceiling). post − prior is what z adds. Also the noise ceiling sqrt(ρ_AB), the best
a noiseless predictor of the half-B trait could reach given half B's own sampling noise.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import chess
import numpy as np
import polars as pl
import torch
from scipy.stats import spearmanr

from doppel_lab.v0_data import HIST, MAX_LEGAL, targets_sample
from doppel_lab.v0_model import NMOVES, legal_mask
from doppel_lab.v0_train import to_dev
from doppel_lab.v0e_data import CTX_FEATS, EMB_DIM, K_MAX, Split
from doppel_lab.v0e_data import collate as v0e_collate
from doppel_model import MOVE_INDEX, board_planes, mirror_uci

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "lab/results/style_fidelity.json"
PER = 12            # half-B positions per game side that get children
MIN_PLY = 10
HALF = 24
c = pl.col
torch.multiprocessing.set_sharing_strategy("file_system")


def targets(band: int, limit: int):
    sp = Split("valrich", band)
    tp = set(pl.read_parquet(ROOT / f"data/v0e/valrich_players_{band}.parquet")["player"].to_list())
    tid = sorted(i for i, n in enumerate(np.asarray(sp.players)) if n in tp)
    if limit:
        tid = tid[:limit]
    rows = []
    for p in tid:
        gs = sorted(((sp.t[g], g, col) for col in (0, 1) for g in np.flatnonzero(sp.pid[:, col] == p)))
        for r, (_, g, col) in enumerate(gs):
            rows.append((p, int(g), int(col), r + 1))
    return sp, pl.DataFrame(rows, schema=["player", "g", "col", "grank"], orient="row")


def legal_idx(b: chess.Board) -> list[int]:
    w = b.turn == chess.WHITE
    return sorted(MOVE_INDEX[m.uci() if w else mirror_uci(m.uci())] for m in b.legal_moves)


class ActualPositions:
    """Pass A, trunk only: every ply >= MIN_PLY - 1 of a target side's games, both movers (the opponent's
    entropy at ply k+1 is what the player handed over at ply k)."""

    def __init__(self, sp, items):
        self.sp, self.items = sp, items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        g, col = self.items[i]
        n = self.sp.ix.plies(g)
        plies = [k for k in range(MIN_PLY, n + 1) if k % 2 == col and k < n]
        if not plies:
            return None
        return targets_sample(self.sp.ix, g, col, plies, None, len(plies), np.random.default_rng(0))


class ModelPositions:
    """Pass B: sampled own plies of a half-B game side, with context, plus every child position."""

    def __init__(self, sp, items):
        self.sp, self.items = sp, items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        sp = self.sp
        g, col = self.items[i]
        n = sp.ix.plies(g)
        cand = [k for k in range(MIN_PLY, n) if k % 2 == col]
        if not cand:
            return None
        rng = np.random.default_rng((g, col))
        plies = sorted(rng.choice(cand, min(PER, len(cand)), replace=False).tolist())
        out = targets_sample(sp.ix, g, col, plies, None, len(plies), rng)
        hg, hc = sp.history(g, col)
        hg, hc = hg[-K_MAX:], hc[-K_MAX:]
        k = len(hg)
        out["ctx_emb"] = np.asarray(sp.vec.emb[hg, hc]) if k else np.zeros((0, EMB_DIM), np.float16)
        out["ctx_feat"] = sp.feat[hg, hc] if k else np.zeros((0, CTX_FEATS), np.float32)
        out["pid"] = np.array([sp.pid[g, col]], np.int64)
        out["nctx"] = np.array([k], np.int64)
        # children: replay, keep the planes window, expand every legal move at each sampled ply
        uci = sp.ix.uci[sp.ix.off[g]:sp.ix.off[g] + n]
        b = chess.Board()
        planes = [board_planes(b)]
        ch_tok, ch_legal, ch_par, ch_mv = [], [], [], []
        want = set(plies)
        for kk in range(max(plies) + 1):
            if kk in want:
                h = planes[max(0, kk - HIST + 1):kk + 1]
                h = [h[0]] * (HIST - len(h)) + h
                white = b.turn == chess.WHITE
                for m in b.legal_moves:
                    cb = b.copy(stack=False)
                    cb.push(m)
                    t = np.zeros((64, 12 * HIST + 1), np.uint8)
                    t[:, :12 * HIST] = np.concatenate(h[1:] + [board_planes(cb)], axis=1)
                    li = legal_idx(cb)[:MAX_LEGAL]
                    lg = np.full(MAX_LEGAL, -1, np.int16)
                    lg[:len(li)] = li
                    ch_tok.append(t)
                    ch_legal.append(lg)
                    ch_par.append(plies.index(kk))
                    ch_mv.append(MOVE_INDEX[m.uci() if white else mirror_uci(m.uci())])
            if kk < max(plies):
                b.push_uci(uci[kk].decode() if isinstance(uci[kk], bytes) else str(uci[kk]))
                planes.append(board_planes(b))
        out["ch_tok"] = np.stack(ch_tok)
        out["ch_legal"] = np.stack(ch_legal)
        out["ch_par"] = np.array(ch_par, np.int64)
        out["ch_mv"] = np.array(ch_mv, np.int64)
        return out


def collate_b(samples):
    samples = [s for s in samples if s is not None]
    ch = {k: [s.pop(k) for s in samples] for k in ("ch_tok", "ch_legal", "ch_par", "ch_mv")}
    off = np.cumsum([0] + [len(s["played"]) for s in samples[:-1]])
    out = v0e_collate(samples)
    out["ch_tok"] = torch.from_numpy(np.concatenate(ch["ch_tok"]))
    out["ch_legal"] = torch.from_numpy(np.concatenate(ch["ch_legal"]))
    out["ch_par"] = torch.from_numpy(np.concatenate([p + o for p, o in zip(ch["ch_par"], off)]))
    out["ch_mv"] = torch.from_numpy(np.concatenate(ch["ch_mv"]))
    return out


def collate_a(samples):
    samples = [s for s in samples if s is not None]
    return {k: torch.from_numpy(np.concatenate([s[k] for s in samples]))
            for k in ("tok", "legal", "played", "think", "elo_self", "elo_opp", "ply", "game")}


def entropy(logits, legal):
    mask = legal_mask(legal)
    lp = logits.float().masked_fill(~mask, -1e9).log_softmax(-1)
    H = -torch.where(mask, lp.exp() * lp, 0.0).sum(-1)
    return lp, torch.where(mask.any(-1), H, 0.0)


def mk_loader(ds, batch, workers, coll):
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False, collate_fn=coll, num_workers=workers,
                                       prefetch_factor=2 if workers else None,
                                       multiprocessing_context="fork" if workers else None)


@torch.no_grad()
def pass_a(trunk, sp, items, device, workers):
    acc = {k: [] for k in ("game", "ply", "H", "lp", "think_true", "before", "inc")}
    for b in mk_loader(ActualPositions(sp, items), 32, workers, collate_a):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            lg, _, _ = trunk(b["tok"].to(device).float(), b["elo_self"].to(device), b["elo_opp"].to(device))
        lp, H = entropy(lg, b["legal"].to(device))
        acc["H"].append(H.cpu())
        acc["lp"].append(lp.gather(1, b["played"].to(device).clamp(min=0)[:, None])[:, 0].cpu())
        for k, j in (("think_true", 0), ("before", 1), ("inc", 2)):
            acc[k].append(b["think"][:, j].clone())
        acc["game"].append(b["game"].clone())
        acc["ply"].append(b["ply"].clone())
    return pl.DataFrame({k: torch.cat(v).numpy() for k, v in acc.items()})


@torch.no_grad()
def pass_b(model, sp, items, device, workers, chunk=4096):
    acc = {k: [] for k in ("game", "ply", "H", "lp_played", "think_true", "before", "inc",
                           "steer_post", "steer_prior", "steer_maia", "conf_post", "conf_prior", "conf_maia",
                           "think_post", "think_prior", "child_played_H")}
    t0 = time.time()
    for n, b in enumerate(mk_loader(ModelPositions(sp, items), 8, workers, collate_b)):
        ch = {k: b.pop(k) for k in ("ch_tok", "ch_legal", "ch_par", "ch_mv")}
        b = to_dev(b, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            tok = b["tok"].float()
            t_logits, _, pooled = model.trunk(tok, b["elo_self"], b["elo_opp"])
            mask = legal_mask(b["legal"])
            t_lp, H = entropy(t_logits, b["legal"])
            mu_q, _ = model.posterior(b)
            mu_p, _ = model.prior(b["cond"])
            emb = model.residual.board(tok)
            sid = b["sid"]
            P = {"maia": t_lp.exp()}
            for tag, z in (("post", mu_q[sid]), ("prior", mu_p[sid])):
                lg = t_lp + model.residual(emb, z, b["elo_self"], b["elo_opp"], mask).float()
                P[tag] = lg.masked_fill(~mask, -1e9).softmax(-1)
                mu_t, _ = model.think(pooled, b["clk"], b["tempo"], z)
                acc[f"think_{tag}"].append(torch.sigmoid(mu_t.float()).cpu())
            # children through the trunk, the opponent to move (Elos swapped)
            par, mv = ch["ch_par"].to(device), ch["ch_mv"].to(device)
            Hc = torch.empty(len(par), device=device)
            for s in range(0, len(par), chunk):
                pp = par[s:s + chunk]
                lgc, _, _ = model.trunk(ch["ch_tok"][s:s + chunk].to(device).float(), b["elo_opp"][pp], b["elo_self"][pp])
                Hc[s:s + chunk] = entropy(lgc, ch["ch_legal"][s:s + chunk].to(device))[1]
        npar = len(b["played"])
        for tag, Pm in P.items():
            w = Pm.float()[par, mv]
            acc[f"steer_{tag}"].append(torch.zeros(npar, device=device).index_add_(0, par, w * Hc).cpu())
            acc[f"conf_{tag}"].append(torch.where(mask, Pm.float() * t_lp, 0.0).sum(-1).cpu())
        played = b["played"]
        is_pl = mv == played[par]
        acc["child_played_H"].append(torch.zeros(npar, device=device).index_add_(0, par, is_pl.float() * Hc).cpu())
        acc["H"].append(H.float().cpu())
        acc["lp_played"].append(t_lp.gather(1, played.clamp(min=0)[:, None])[:, 0].float().cpu())
        for k, j in (("think_true", 0), ("before", 1), ("inc", 2)):
            acc[k].append(b["think"][:, j].float().cpu())
        acc["game"].append(b["game"].cpu())
        acc["ply"].append(b["ply"].cpu())
        if n % 200 == 0:
            print(f"  pass B batch {n}: {sum(len(x) for x in acc['ply']):,} positions, {len(par):,} children in "
                  f"this batch ({time.time() - t0:.0f}s)", flush=True)
    return pl.DataFrame({k: torch.cat(v).numpy() for k, v in acc.items()})


def side_traits(d: pl.DataFrame, steer_cols: list[str], conf_cols: list[str], think_cols: list[str],
                beta: np.ndarray, base: np.ndarray) -> pl.DataFrame:
    """Per (game, mover) means / Spearman, given per-position columns."""
    d = d.with_columns(base=pl.Series(base[d["game"].to_numpy()]))
    pred = beta[0] + beta[1] * c("H") + beta[2] * c("H") ** 2 + beta[3] * pl.min_horizontal(c("ply"), 80)
    timed = (c("before") >= 0.1 * c("base")) & c("think_true").is_not_nan()
    aggs = [*[(c(s) - pred).mean().alias(s) for s in steer_cols], *[c(s).mean().alias(s) for s in conf_cols],
            *[pl.corr(c(s).filter(timed), c("H").filter(timed), method="spearman").alias(s) for s in think_cols]]
    out = d.group_by("game", (c("ply") % 2).alias("col")).agg(*aggs, n=pl.len())
    return out.with_columns([c(s).cast(pl.Float64).fill_nan(None) for s in think_cols])   # constant think -> NaN


def player_rho(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    return float(spearmanr(x[ok], y[ok]).statistic), int(ok.sum())


def choice_level(Bm: pl.DataFrame, T: pl.DataFrame) -> dict:
    """Move-choice traits, net of the position: at each half-B position, the played move's value minus Maia's
    expectation there (actual), and the model's expectation minus Maia's (post, prior). Scored at the player's own
    positions, a trait mean is mostly the positions they reach (the trunk alone matches the actual steering at
    rho 0.72); this keeps only what the move choice adds. Reliability: odd vs even half-B games (the history
    ceiling is within half B here, since the Maia expectation exists only for pass-B positions)."""
    key = T.select(c("g").alias("game"), "col", "player", "grank")
    d = (Bm.with_columns(col=(c("ply") % 2).cast(pl.Int64)).join(key, on=["game", "col"])
         .with_columns(steer_actual=c("child_played_H") - c("steer_maia"),
                       steer_post_c=c("steer_post") - c("steer_maia"), steer_prior_c=c("steer_prior") - c("steer_maia"),
                       conf_actual=c("lp_played") - c("conf_maia"),
                       conf_post_c=c("conf_post") - c("conf_maia"), conf_prior_c=c("conf_prior") - c("conf_maia")))
    out = {}
    for mp, tag in ((MIN_PLY, f"ply{MIN_PLY}+"), (20, "ply20+")):
        x = d.filter(c("ply") >= mp)
        cols = ["steer_actual", "steer_post_c", "steer_prior_c", "conf_actual", "conf_post_c", "conf_prior_c"]
        P = x.group_by("player").agg(*[c(k).mean() for k in cols]).sort("player")
        H = x.group_by("player", (c("grank") % 2).alias("par")).agg(c("steer_actual", "conf_actual").mean())
        H0, H1 = H.filter(c("par") == 0).sort("player"), H.filter(c("par") == 1).sort("player")
        H0, H1 = H0.join(H1.select("player"), on="player"), H1.join(H0.select("player"), on="player")
        e = {}
        for trait, a_ in (("steer", "steer"), ("conform", "conf")):
            y = P[f"{a_}_actual"].to_numpy()
            r_sh = player_rho(H0[f"{a_}_actual"].to_numpy(), H1[f"{a_}_actual"].to_numpy())[0]
            sb = 2 * r_sh / (1 + r_sh) if r_sh > -1 else float("nan")      # Spearman-Brown: full half B
            e[trait] = {"split_half_B": round(r_sh, 3), "reliability_B": round(sb, 3),
                        "noise_ceiling": round(float(np.sqrt(max(sb, 0))), 3),
                        "post": round(player_rho(P[f"{a_}_post_c"].to_numpy(), y)[0], 3),
                        "prior": round(player_rho(P[f"{a_}_prior_c"].to_numpy(), y)[0], 3),
                        "model_sd_post": round(float(P[f"{a_}_post_c"].std()), 5),
                        "actual_sd": round(float(P[f"{a_}_actual"].std()), 5), "players": P.height}
            e[trait]["z_adds"] = round(e[trait]["post"] - e[trait]["prior"], 3)
        out[tag] = e
    return out


def main(run: str, band: int, device: str, workers: int, limit: int, analyze_only: bool = False):
    from doppel_lab.v0e_train import build
    t0 = time.time()
    sp, T = targets(band, limit)
    T = T.with_columns(half=pl.when(c("grank") <= HALF).then(pl.lit("A")).otherwise(pl.lit("B")))
    items_a = [(g, col) for g, col in T.select("g", "col").iter_rows()]
    items_b = [(g, col) for g, col in T.filter(c("half") == "B").select("g", "col").iter_rows()]
    print(f"{T['player'].n_unique()} players, {len(items_a):,} sides, {len(items_b):,} half-B ({time.time() - t0:.0f}s)",
          flush=True)
    cache = ROOT / f"data/v0e/style_{run}{'_smoke' if limit else ''}"
    s = torch.load(ROOT / "data/v0/runs" / run / "ckpt.pt", map_location="cpu", weights_only=False)
    if analyze_only:
        A, Bm = pl.read_parquet(f"{cache}_A.parquet"), pl.read_parquet(f"{cache}_B.parquet")
    else:
        model = build(False, device, band=band)
        model.load_state_dict(s["model"], strict=False)
        model.eval()
        # pass A covers both movers' plies of every target game side (handed entropy = next ply's entropy)
        A = pass_a(model.trunk, sp, [(g, col) for g, col0 in items_a for col in (col0, 1 - col0)], device, workers)
        A = A.unique(["game", "ply"])
        A.write_parquet(f"{cache}_A.parquet")
        print(f"pass A: {A.height:,} positions ({time.time() - t0:.0f}s)", flush=True)
        Bm = pass_b(model, sp, items_b, device, workers)
        Bm.write_parquet(f"{cache}_B.parquet")
        print(f"pass B: {Bm.height:,} positions ({time.time() - t0:.0f}s)", flush=True)
    A = A.join(A.select("game", (c("ply") - 1).alias("ply"), handed=c("H")), on=["game", "ply"], how="left")
    own = A.join(T.select(c("g").alias("game"), "col", "player", "half"), on="game", how="inner").filter(
        (c("ply") % 2) == c("col")).filter(c("handed").is_not_null())
    Z = np.c_[np.ones(own.height), own["H"].to_numpy(), own["H"].to_numpy() ** 2, np.minimum(own["ply"].to_numpy(), 80)]
    beta = np.linalg.lstsq(Z, own["handed"].to_numpy(), rcond=None)[0]
    base = sp.ix.base.astype(np.float64)
    tf = lambda d: d.with_columns(think_true=c("think_true") / (c("before") + c("inc")).clip(1.0))
    out = {"run": run, "band": band, "step": int(s["step"]), "players": int(T["player"].n_unique()),
           "pass_a_positions": A.height, "pass_b_positions": Bm.height, "steer_fit": [float(v) for v in beta]}
    for mp, tag in ((MIN_PLY, f"ply{MIN_PLY}+"), (20, "ply20+")):
        a_own = tf(own.filter(c("ply") >= mp).rename({"handed": "steer_actual", "lp": "conf_actual"}))
        sa = side_traits(a_own.with_columns(think_actual=c("think_true")), ["steer_actual"], ["conf_actual"],
                         ["think_actual"], beta, base)
        b = tf(Bm.filter(c("ply") >= mp)).with_columns(
            **{f"steer_{t}": c(f"steer_{t}") for t in ("post", "prior", "maia")})
        sb = side_traits(b.with_columns(think_actual=c("think_true")),
                         ["steer_post", "steer_prior", "steer_maia"], ["conf_post", "conf_prior", "conf_maia"],
                         ["think_post", "think_prior"], beta, base)
        key = T.select(c("g").alias("game"), "col", "player", "half")
        pa = sa.join(key, on=["game", "col"]).group_by("player", "half").agg(
            c("steer_actual", "conf_actual", "think_actual").mean())
        pb = sb.join(key, on=["game", "col"]).group_by("player").agg(
            c("steer_post", "steer_prior", "steer_maia", "conf_post", "conf_prior", "conf_maia",
              "think_post", "think_prior").mean())
        P = (pa.filter(c("half") == "B").drop("half")
             .join(pa.filter(c("half") == "A").drop("half"), on="player", suffix="_A")
             .join(pb, on="player"))
        res = {}
        for trait, actual, preds in (("steer", "steer_actual", ("post", "prior", "maia")),
                                     ("conform", "conf_actual", ("post", "prior", "maia")),
                                     ("alloc", "think_actual", ("post", "prior"))):
            yb = P[actual].to_numpy().astype(float)
            col = {"steer": "steer", "conform": "conf", "alloc": "think"}[trait]
            e = {}
            rab, n_ab = player_rho(P[f"{actual}_A"].to_numpy().astype(float), yb)
            e["history_A_vs_B"] = round(rab, 3)
            e["noise_ceiling"] = round(float(np.sqrt(max(rab, 0))), 3)
            for pr in preds:
                e[pr] = round(player_rho(P[f"{col}_{pr}"].to_numpy().astype(float), yb)[0], 3)
            e["z_adds"] = round(e["post"] - e["prior"], 3)
            # does z carry the player's half-A trait into the model? model post vs half A (no shared noise)
            e["post_vs_A"] = round(player_rho(P[f"{col}_post"].to_numpy().astype(float),
                                              P[f"{actual}_A"].to_numpy().astype(float))[0], 3)
            e["prior_vs_A"] = round(player_rho(P[f"{col}_prior"].to_numpy().astype(float),
                                               P[f"{actual}_A"].to_numpy().astype(float))[0], 3)
            e["players"] = n_ab
            res[trait] = e
        out[tag] = res
    out["choice"] = choice_level(Bm, T)
    allres = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    allres[run + ("_smoke" if limit else "")] = out
    RESULTS.write_text(json.dumps(allres, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    print(f"-> {RESULTS} ({time.time() - t0:.0f}s)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v0f")
    ap.add_argument("--band", type=int, default=1500)
    ap.add_argument("--gpu-ok", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--analyze-only", action="store_true", help="reuse the cached passes")
    ap.add_argument("--runs", default="", help="run@band,... (several models in one job; overrides --run/--band)")
    a = ap.parse_args()
    todo = [(r.split("@")[0], int(r.split("@")[1])) for r in a.runs.split(",")] if a.runs else [(a.run, a.band)]
    for run, band in todo:
        main(run, band, "cuda" if a.gpu_ok else "cpu", a.workers, a.limit, a.analyze_only)

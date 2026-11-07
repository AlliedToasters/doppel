"""Phase 2 gate for a Doppel v0 run (brief, "Phase 2 gate"), on held-out players in the band.

  check                  source                                    pass
  1 z is used            v0 eval (val players)                     active units >= 4 of 8; KL per half-game >= free-bits
                                                                   floor + 1 nat; z-swap gain > 0 with its CI above 0
  2 persistence          self-play games, Stockfish 20k nodes      half-game ACPL Spearman above Maia-3's, CI excludes it
  3 variance             same games                                per-game ACPL KS distance to humans <= Maia-3's
  4 accuracy guardrail   v0 eval                                   top-1 with z from the prior >= Maia-3 79M - 0.3 points
  5 think time           v0 eval                                   log-seconds r with z from the prior > 0.56 (ChessMimic)

The z-swap ablation (check 1) stands in for the brief's "history-vs-z": the student sees no move history, so z is
the only channel for what earlier play says about this player. Gain = CE on the target half with z encoded from
another game minus CE with z encoded from this player's other half (nats per move, paired, bootstrap over games).

Self-play mirrors doppel_lab.selfplay for Maia-3: both sides at band + 50, time controls drawn from the band's human
games, no clocks, no resignation, the same terminations. Each side draws z once per game from the prior. Human and
Maia-3 references are data/human + data/synth/maia3_<band>, scored the same way (doppel_lab.metrics, 20k nodes).

    # 1. eval (GPU, warden) -> lab/results/v0_<run>.json
    python -m doppel_lab.v0_train eval --run v0b --gpu-ok
    # 2. self-play (GPU, warden) -> data/synth/doppel_<run>_<band>.parquet
    python -m doppel_lab.v0_gate selfplay --run v0b --gpu-ok
    # 3. Stockfish (CPU) -> data/evals/doppel_<run>_<band>.parquet
    python -m doppel_lab.analyze ../data/synth/doppel_v0b_1500.parquet ../data/evals/doppel_v0b_1500.parquet --nodes 20000
    # 4. score (CPU) -> lab/results/v0_gate_<run>.json; checks whose inputs are missing are "pending"
    python -m doppel_lab.v0_gate score --run v0b
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import chess
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import stats

from doppel_lab import metrics as M
from doppel_lab.selfplay import ended
from doppel_lab.v0_data import HIST, V0

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "lab/results"
SYNTH, EVALS, HUMAN = ROOT / "data/synth", ROOT / "data/evals", ROOT / "data/human"
CM_THINK_R = 0.56
TOP1_SLACK = 0.003
MIN_ACTIVE = 4
KL_MARGIN = 1.0


def synth_path(run: str, band: int) -> Path:
    return SYNTH / f"doppel_{run}_{band}.parquet"


def evals_path(run: str, band: int) -> Path:
    return EVALS / f"doppel_{run}_{band}.parquet"


# ---------------------------------------------------------------- self-play
def cond_vec(es: float, eo: float, base: float, inc: float) -> list[float]:
    """Same as v0_data.build_sample's cond."""
    return [(es - 1500) / 400, (eo - 1500) / 400, math.log1p(base), inc / max(base, 1.0)]


class Doppel:
    """Batched move sampler: teacher log-probs plus the residual at a fixed per-game, per-side z."""

    def __init__(self, model, device: str):
        self.m, self.device = model.eval(), device

    def sample_z(self, cond: np.ndarray, gen) -> np.ndarray:
        import torch
        with torch.no_grad():
            mu, lv = self.m.prior(torch.as_tensor(cond, dtype=torch.float32, device=self.device))
            z = mu + torch.randn(mu.shape, generator=gen, device=self.device) * (0.5 * lv).exp()
        return z.float().cpu().numpy()

    def sample(self, boards: list[chess.Board], hist: list[list[np.ndarray]], z: np.ndarray, elo_self, elo_opp,
               gen) -> list[str]:
        import torch
        from doppel_model import MOVES, encode, legal_mask, mirror_uci
        tok = torch.as_tensor(np.stack([encode(h, HIST) for h in hist]), device=self.device)
        masks, ucis = zip(*(legal_mask(b) for b in boards))
        mask = torch.as_tensor(np.stack(masks), device=self.device)
        es = torch.as_tensor(elo_self, dtype=torch.long, device=self.device)
        eo = torch.as_tensor(elo_opp, dtype=torch.long, device=self.device)
        zt = torch.as_tensor(z, dtype=torch.float32, device=self.device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")):
            t_logits, _, _ = self.m.trunk(tok, es, eo)
            t_logp = t_logits.float().masked_fill(~mask, -1e9).log_softmax(-1)
            res = self.m.residual(self.m.residual.board(tok), zt, es, eo, mask).float()
        logp = (t_logp + res).masked_fill(~mask, float("-inf")).log_softmax(-1)
        picks = torch.multinomial(logp.exp(), 1, generator=gen)[:, 0].cpu().numpy()
        out = []
        for b, p, legal in zip(boards, picks, ucis):
            u = MOVES[p]
            u = u if b.turn == chess.WHITE else mirror_uci(u)
            assert u in legal, (u, b.fen())
            out.append(u)
        return out


def selfplay(args):
    import torch
    from doppel_lab.v0_train import build
    device = "cuda" if args.gpu_ok else "cpu"
    if args.fake_trunk and args.gpu_ok:
        sys.exit("--fake-trunk is a CPU smoke test")
    model = build(args.fake_trunk, device)
    ck = V0 / "runs" / args.run / "ckpt.pt"
    if ck.exists():
        model.load_state_dict(torch.load(ck, map_location=device, weights_only=False)["model"], strict=False)
    elif not args.fake_trunk:
        sys.exit(f"no checkpoint at {ck}")
    eng = Doppel(model, device)
    from doppel_model import board_planes

    rng = np.random.default_rng(args.seed + args.band)
    gen = torch.Generator(device=device).manual_seed(args.seed + args.band)
    tcs = pl.read_parquet(HUMAN / "games_2026-10.parquet", columns=["band", "base", "inc"]).filter(
        pl.col("band") == args.band)
    tc = tcs[rng.integers(0, len(tcs), args.games).tolist()]
    base, inc = tc["base"].to_numpy().astype(float), tc["inc"].to_numpy().astype(float)
    elo = args.band + 50
    cond = np.array([cond_vec(elo, elo, b, i) for b, i in zip(base, inc)], np.float32)
    zs = np.stack([eng.sample_z(cond, gen), eng.sample_z(cond, gen)], 1)  # (G, 2 sides, Z)

    boards = [chess.Board() for _ in range(args.games)]
    hist = [[board_planes(b)] for b in boards]
    moves = [[] for _ in range(args.games)]
    results = [None] * args.games
    active = list(range(args.games))
    t0, step = time.time(), 0
    while active:
        picks = []
        for lo in range(0, len(active), args.batch):
            idx = active[lo:lo + args.batch]
            side = [0 if boards[i].turn == chess.WHITE else 1 for i in idx]
            picks += eng.sample([boards[i] for i in idx], [hist[i] for i in idx], zs[idx, side],
                                [elo] * len(idx), [elo] * len(idx), gen)
        still = []
        for i, u in zip(active, picks):
            boards[i].push_uci(u)
            moves[i].append(u)
            hist[i] = (hist[i] + [board_planes(boards[i])])[-HIST:]
            r = ended(boards[i])
            if r:
                results[i] = r
            else:
                still.append(i)
        active = still
        step += 1
        if step % 20 == 0:
            print(f"ply {step}: {len(active)} games running, {time.time() - t0:.0f}s", file=sys.stderr, flush=True)

    src = f"doppel-{args.run}"
    tbl = pa.table({
        "game_id": [f"{src}-{args.band}-{i}" for i in range(args.games)],
        "band": [args.band] * args.games,
        "base": base.astype(int).tolist(), "inc": inc.astype(int).tolist(),
        "result": [r[0] for r in results], "termination": [r[1] for r in results],
        "moves": moves, "clocks": pa.nulls(args.games),
        "source": [src] * args.games,
        "z_white": zs[:, 0].tolist(), "z_black": zs[:, 1].tolist(),
    })
    out = Path(args.out) if args.out else synth_path(args.run, args.band)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    pq.write_table(tbl, tmp, compression="zstd")
    tmp.rename(out)
    print(f"wrote {args.games} games to {out} in {time.time() - t0:.0f}s", file=sys.stderr)


# ---------------------------------------------------------------- scoring
def player_stats(games: pl.DataFrame, ev: pl.DataFrame) -> np.ndarray:
    """(n player-games, 3): acpl, acpl first half of own moves, acpl second half."""
    df = games.select("game_id", "moves", "result", "termination").join(ev, on="game_id")
    rows = []
    for row in df.iter_rows(named=True):
        for p in M.game_stats(row, row["evals"])["players"]:
            rows.append((p["acpl"], p["acpl_h1"], p["acpl_h2"]))
    return np.array(rows, float).reshape(-1, 3)


def half_corr(a: np.ndarray, rng, boots: int = 1000) -> dict:
    pt = stats.spearmanr(a[:, 1], a[:, 2]).statistic
    bs = [stats.spearmanr(*a[rng.integers(0, len(a), len(a))][:, 1:].T).statistic for _ in range(boots)]
    return {"n": len(a), "r": float(pt), "ci": [float(x) for x in np.percentile(bs, [2.5, 97.5])]}


def sources(run: str, band: int) -> dict[str, tuple[Path, Path]]:
    return {"human": (HUMAN / "games_2026-10.parquet", EVALS / "human.parquet"),
            "maia3": (SYNTH / f"maia3_{band}.parquet", EVALS / f"maia3_{band}.parquet"),
            "doppel": (synth_path(run, band), evals_path(run, band))}


def game_checks(run: str, band: int, rng) -> dict:
    src = sources(run, band)
    missing = [str(p) for g, e in src.values() for p in (g, e) if not p.exists()]
    d_games, d_ev = src["doppel"]
    if d_games.exists() and d_ev.exists():
        n_ev, n_g = pl.scan_parquet(d_ev).select(pl.len()).collect().item(), pq.read_metadata(d_games).num_rows
        if n_ev < n_g:
            missing.append(f"{d_ev} ({n_ev} of {n_g} games evaluated)")
    if missing:
        return {"persistence": {"status": "pending", "missing": missing},
                "variance": {"status": "pending", "missing": missing}}
    st = {}
    for name, (g, e) in src.items():
        games = pl.read_parquet(g)
        if "band" in games.columns:
            games = games.filter(pl.col("band") == band)
        st[name] = player_stats(games, pl.read_parquet(e))
    hc = {k: half_corr(v, rng) for k, v in st.items()}
    pers = {"pass": bool(hc["doppel"]["r"] > hc["maia3"]["r"] and hc["doppel"]["ci"][0] > hc["maia3"]["r"]),
            "rule": "doppel half-corr > maia3's, and doppel's CI lower bound > maia3's point", **hc}
    ks = {k: float(stats.ks_2samp(st[k][:, 0], st["human"][:, 0]).statistic) for k in ("maia3", "doppel")}
    var = {"pass": bool(ks["doppel"] <= ks["maia3"]), "rule": "per-game ACPL KS to humans: doppel <= maia3",
           "ks_to_human": ks, "mean_acpl": {k: float(v[:, 0].mean()) for k, v in st.items()}}
    return {"persistence": pers, "variance": var}


def eval_checks(ev: dict | None, free_bits: float) -> dict:
    if ev is None:
        pend = {"status": "pending", "missing": ["v0 eval json"]}
        return {"z_used": pend, "accuracy": pend, "think": pend}
    zu = {"active_units": ev["active_units"], "kl_nats_per_half": ev["kl_nats_per_half"], "free_bits": free_bits}
    parts = {"active_units": ev["active_units"] >= MIN_ACTIVE,
             "kl": ev["kl_nats_per_half"] >= free_bits + KL_MARGIN}
    if "z_swap_gain_nats" in ev:
        zu |= {"ce_post": ev["ce_post"], "ce_swap": ev["ce_swap"], "z_swap_gain_nats": ev["z_swap_gain_nats"],
               "z_swap_gain_ci": ev["z_swap_gain_ci"]}
        parts["z_swap"] = ev["z_swap_gain_ci"][0] > 0
    else:
        zu["z_swap"] = "eval predates the z-swap ablation; re-run v0_train eval"
        parts["z_swap"] = False
    zu |= {"parts": parts, "pass": all(parts.values()),
           "rule": f"active units >= {MIN_ACTIVE}; KL >= free bits + {KL_MARGIN}; z-swap gain CI > 0"}
    acc = {"top1_prior": ev["top1_prior"], "top1_post": ev["top1_post"], "top1_teacher": ev["top1_teacher"],
           "pass": ev["top1_prior"] >= ev["top1_teacher"] - TOP1_SLACK,
           "rule": "top-1 with prior z >= Maia-3 79M top-1 - 0.3 points"}
    th = {"r_log_prior": ev["think_r_log_prior"], "r_log_post": ev["think_r_log_post"], "chessmimic": CM_THINK_R,
          "pass": ev["think_r_log_prior"] > CM_THINK_R, "rule": "log-seconds r with prior z > ChessMimic's 0.56"}
    return {"z_used": zu, "accuracy": acc, "think": th}


def score(args):
    ev_path = RESULTS / f"v0_{args.run}.json"
    ev = json.loads(ev_path.read_text()) if ev_path.exists() else None
    free_bits = 3.0
    ck = V0 / "runs" / args.run / "ckpt.pt"
    if ck.exists():
        import torch
        free_bits = torch.load(ck, map_location="cpu", weights_only=False).get("args", {}).get("free_bits", 3.0)
    checks = eval_checks(ev, free_bits) | game_checks(args.run, args.band, np.random.default_rng(0))
    status = {k: ("pending" if v.get("status") == "pending" else "pass" if v["pass"] else "fail")
              for k, v in checks.items()}
    verdict = "pending" if "pending" in status.values() else "pass" if all(
        s == "pass" for s in status.values()) else "fail"
    r = {"run": args.run, "band": args.band, "step": ev and ev.get("step"), "verdict": verdict, "status": status,
         "checks": checks}
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"v0_gate_{args.run}.json"
    out.write_text(json.dumps(r, indent=2) + "\n")
    for k, s in status.items():
        print(f"{s:8} {k}")
    print(f"gate {verdict} -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("selfplay", "score"):
        p = sub.add_parser(name)
        p.add_argument("--run", required=True)
        p.add_argument("--band", type=int, default=1500)
    sp = sub.choices["selfplay"]
    sp.add_argument("--games", type=int, default=1000)
    sp.add_argument("--batch", type=int, default=500)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--out", default=None)
    sp.add_argument("--gpu-ok", action="store_true")
    sp.add_argument("--fake-trunk", action="store_true")
    a = ap.parse_args()
    if a.cmd == "selfplay":
        selfplay(a)
    else:
        score(a)

"""Train one player's doppel adapter from a trainer job (api trainer.py), then prove the file reproduces it.

    python -m doppel_lab.doppel_train --job <dir>/job.json --out <dir> [--device cuda]   # GPU: through the warden

Input (job.json, from POST /api/trainer/claim): package name, games oldest first, each {moves (SAN), color,
rating, opp_rating}. No user or game ids. Every game is replayed with python-chess; a game that doesn't replay is
dropped (it is untrusted input), as is one shorter than 2 plies; moves past MAX_PLIES are cut.
Recipe: the package's lora.train (allie_ttt: rank-8 LoRA, 3 epochs, AdamW, the NVAL most recent games as
validation, the epoch with the lowest validation CE; epoch 0 = Maia-3, i.e. B = 0). Own moves only.
Output in --out: <id>.npz (doppel_model.adapter) and result.json {adapter, best_epoch, val CE, games, positions,
parity}. Before writing result.json the adapter file is loaded back and run through the ONNX package against the
lab's hooked torch model (export_package.parity_lora, CPU, fp32) on validation positions: max |dlogp| <= TOL.
Exit 1 with a message on stderr when anything fails; the worker reports it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import chess
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
MAX_PLIES = 400
MIN_GAMES = 15
TOL = 1e-3
N_PARITY = 40


def package_dir(name: str) -> Path:
    ds = sorted((p for p in (ROOT / "models").glob(f"{name}-*") if (p / "manifest.json").exists()),
                key=lambda p: p.stat().st_mtime)
    for p in reversed(ds):
        if json.loads((p / "manifest.json").read_text()).get("name") == name:
            return p
    raise SystemExit(f"no package {name} under models/")


def to_uci(san: list) -> list[str] | None:
    """SAN -> UCI by replaying from the start; None if any move is illegal or unparsable."""
    b, out = chess.Board(), []
    try:
        for m in san[:MAX_PLIES]:
            mv = b.parse_san(str(m))
            out.append(mv.uci())
            b.push(mv)
    except (ValueError, AssertionError):
        return None
    return out


def elo(x) -> int:
    try:
        return int(min(2800, max(600, int(x))))
    except (TypeError, ValueError):
        return 1500


def rows(job: dict, nval: int) -> pl.DataFrame:
    gs = []
    for g in job["games"]:
        u = to_uci(g.get("moves") or [])
        col = {"white": 0, "black": 1}.get(g.get("color"))
        if u is None or col is None or len(u) < 2:
            continue
        me, opp = elo(g.get("rating")), elo(g.get("opp_rating"))
        gs.append({"moves": u, "col": col, "white_elo": me if col == 0 else opp, "black_elo": opp if col == 0 else me})
    if len(gs) < MIN_GAMES:
        raise SystemExit(f"only {len(gs)} usable games (need {MIN_GAMES})")
    n = len(gs)
    for i, g in enumerate(gs):                      # oldest first: the last nval are validation
        back = n - 1 - i
        g.update(pid=0, role="val" if back < nval else "train", tr_rank=back - nval, game_id=str(i))
    return pl.DataFrame(gs)


def train(d: pl.DataFrame, spec: dict, device: str, workers: int):
    import torch

    from doppel_lab.engines import Maia3
    from doppel_lab.lora_probe import Lora, Positions, encode_all, logits, per_position
    cfg = spec["train"]
    model = Maia3("79m", device).model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    P = Positions(d, encode_all(d, workers))
    lora = Lora(model, 1, device, r=spec["rank"], scale=spec["scale"])
    torch.manual_seed(0)
    opt = torch.optim.AdamW(lora.params, lr=cfg["lr"], weight_decay=cfg["wd"])
    tr, va = np.flatnonzero(P.role == "train"), np.flatnonzero(P.role == "val")
    amp_eval = torch.float16 if device.startswith("cuda") else torch.float32
    own = lambda idx: np.zeros(len(idx), np.int64)
    lp0, _ = per_position(model, lora, P, va, None, device, amp_eval)
    best_ce, ces = float(-lp0.mean()), [float(-lp0.mean())]
    best, best_ep = lora.snapshot(), 0
    rng = np.random.default_rng(0)
    for ep in range(1, cfg["epochs"] + 1):
        perm = rng.permutation(tr)
        for s in range(0, len(perm), cfg["bs"]):
            idx = perm[s:s + cfg["bs"]]
            b = P.batch(idx, device)
            lp = logits(model, b, lora, torch.zeros(len(idx), dtype=torch.long, device=device), torch.bfloat16)
            loss = -lp.gather(1, b[4][:, None]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        lpv, _ = per_position(model, lora, P, va, own, device, amp_eval)
        ce = float(-lpv.mean())
        ces.append(ce)
        if ce < best_ce:
            best_ce, best, best_ep = ce, lora.snapshot(), ep
    for h in lora.hooks:
        h.remove()
    arrays = [p[0].float().cpu().numpy() for p in best]
    if best_ep == 0:
        arrays = [a if i % 2 == 0 else np.zeros_like(a) for i, a in enumerate(arrays)]   # B = 0: Maia-3 exactly
    return arrays, {"best_epoch": best_ep, "val_ce": ces, "train_positions": int(len(tr)),
                    "val_positions": int(len(va))}


def verify(pkg_dir: Path, spec: dict, d: pl.DataFrame, arrays: list[np.ndarray]) -> dict:
    """ONNX package + reloaded adapter vs the lab's hooked torch model, CPU fp32, on validation positions."""
    from doppel_lab.engines import Maia3
    from doppel_lab.export_package import parity_lora
    from doppel_model import ModelPackage
    pos = []
    for r in d.filter(pl.col("role") == "val").iter_rows(named=True):
        for k in range(r["col"], len(r["moves"]), 2):
            pos.append((r["moves"][:k], r["white_elo"], r["black_elo"]))
    pos = pos[:: max(1, len(pos) // N_PARITY)][:N_PARITY]
    eng = Maia3("79m", "cpu")
    pkg = ModelPackage.load(pkg_dir, threads=4)
    res = parity_lora(eng, pkg, pos, spec, arrays)
    return res | {"tol": TOL, "pass": bool(res["max_abs_logp_diff"] <= TOL)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    t0 = time.time()
    from doppel_model.adapter import load_adapter, save_adapter
    job = json.loads(a.job.read_text())
    pkg_dir = package_dir(job["package"])
    man = json.loads((pkg_dir / "manifest.json").read_text())
    spec = man["lora"]
    d = rows(job, spec["train"]["nval"])
    arrays, info = train(d, spec, a.device, a.workers)
    aid = save_adapter(a.out, spec, arrays, {"package": man["name"], "rank": spec["rank"], "blocks": spec["blocks"],
                                             "layers": spec["layers"], "made": time.strftime("%Y-%m-%d"),
                                             "best_epoch": info["best_epoch"], "games": d.height})
    back, _ = load_adapter(a.out, aid, spec)
    par = verify(pkg_dir, spec, d, back)
    res = {"adapter": aid, "package": man["name"], "games": d.height, **info, "parity": par,
           "seconds": round(time.time() - t0, 1)}
    print(json.dumps(res), flush=True)
    if not par["pass"]:
        sys.exit(f"parity failed: {par}")
    (a.out / "result.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

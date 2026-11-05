"""Export a lab checkpoint to a model package (contracts/model_package.schema.json) and run the parity test.

    model.onnx      inputs  tokens (1, 64, 12*history+1) f32, elo_self (1,) i64, elo_opp (1,) i64
                    outputs logits_move (B, 4352), logits_value (B, 3)
    manifest.json   id, sha256, input spec, Elo range, lab commit, parity result

Parity: the lab torch model fed by maia3's own tokenizer vs ONNX Runtime fed by doppel_model's encoding, on
positions sampled from held-out human games. Log-probs over legal moves must agree within --tol.

CPU only (no GPU needed). Run from repo root:
    .venv/bin/python -m doppel_lab.export_package --model 79m --out models
"""

from __future__ import annotations

import argparse
import hashlib
import copy
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import chess  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402

from doppel_lab.engines import GameState, Maia3  # noqa: E402
from doppel_lab.top1 import logprobs  # noqa: E402
from doppel_model import ModelPackage  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


class _Policy(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, tokens, elo_self, elo_opp):
        # Elo as float: maia3 computes elos / 5000, which ONNX would export as integer division on int64.
        out = self.m(tokens, elo_self.float(), elo_opp.float())
        return out[0], out[1]


class _ExportMHA(torch.nn.Module):
    """Drop-in for a batch_first nn.MultiheadAttention (self-attention, eval, need_weights=False), written as
    plain matmuls: torch 2.15's ONNX decomposition of F.multi_head_attention_forward fails on a strided view."""

    def __init__(self, mha: torch.nn.MultiheadAttention):
        super().__init__()
        assert mha.batch_first and mha._qkv_same_embed_dim
        self.h, self.d = mha.num_heads, mha.head_dim
        self.w, self.b = mha.in_proj_weight, mha.in_proj_bias
        self.out = mha.out_proj

    def forward(self, query, key=None, value=None, need_weights=False, attn_mask=None, **_):
        B, L, E = query.shape
        qkv = torch.nn.functional.linear(query, self.w, self.b).reshape(B, L, 3, self.h, self.d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, L, d)
        s = q @ k.transpose(-1, -2) / math.sqrt(self.d)
        if attn_mask is not None:
            s = s + attn_mask.reshape(B, self.h, L, L)
        o = torch.softmax(s, -1) @ v
        return self.out(o.transpose(1, 2).reshape(B, L, E)), None


def _swap_mha(model: torch.nn.Module) -> torch.nn.Module:
    for mod in list(model.modules()):
        for name, child in list(mod.named_children()):
            if isinstance(child, torch.nn.MultiheadAttention):
                setattr(mod, name, _ExportMHA(child))
    return model


def sample_positions(n: int, seed: int = 0):
    g = pl.read_parquet(ROOT / "data/human/eval_2026-10.parquet").select("moves", "white_elo", "black_elo")
    rng = np.random.default_rng(seed)
    rows = g.sample(n, seed=seed).iter_rows()
    out = []
    for moves, we, be in rows:
        k = int(rng.integers(0, max(1, len(moves))))
        board = chess.Board()
        for u in moves[:k]:
            board.push_uci(u)
        if board.is_game_over():
            continue
        out.append((moves[:k], int(we), int(be)))
    return out


def parity(eng: Maia3, pkg: ModelPackage, positions) -> dict:
    worst, top1_agree = 0.0, 0
    for moves, we, be in positions:
        board = chess.Board()
        hist = [Maia3.board_tokens(board)]
        for u in moves:
            board.push_uci(u)
            hist.append(Maia3.board_tokens(board))
        st = GameState(board, we, be, moves=list(moves), maia_hist=hist[-8:])
        ref = logprobs(eng, [st])[0]
        mover, opp = (we, be) if board.turn == chess.WHITE else (be, we)
        _, legal, lp = pkg.policy(list(moves), mover, opp)
        got = dict(zip(legal, lp))
        assert set(got) == set(ref)
        worst = max(worst, float(max(abs(got[u] - ref[u]) for u in ref)))
        top1_agree += int(max(ref, key=ref.__getitem__) == max(got, key=got.__getitem__))
    return {"n": len(positions), "max_abs_logp_diff": worst, "top1_agree": top1_agree / len(positions)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="79m", choices=["5m", "23m", "79m"])
    ap.add_argument("--out", default="models")
    ap.add_argument("--n-parity", type=int, default=1000)
    ap.add_argument("--tol", type=float, default=1e-4)
    a = ap.parse_args()

    torch.set_num_threads(8)
    eng = Maia3(a.model, "cpu")
    hist = eng.cfg.history
    tmp = ROOT / a.out / f".tmp-maia3-{a.model}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    # Fixed batch 1: dynamic batch breaks the MultiheadAttention decomposition (torch 2.15), and infer
    # serves one position per request anyway.
    x = torch.zeros((1, 64, 12 * hist + 1))
    e1, e2 = torch.tensor([1500], dtype=torch.long), torch.tensor([1700], dtype=torch.long)  # distinct tensors, or the exporter ties the two inputs
    torch.onnx.export(_Policy(_swap_mha(copy.deepcopy(eng.model))).eval(), (x, e1, e2), str(tmp / "model.onnx"),
                      input_names=["tokens", "elo_self", "elo_opp"], output_names=["logits_move", "logits_value"],
                      dynamo=True, external_data=False, opset_version=18)
    sha = hashlib.sha256((tmp / "model.onnx").read_bytes()).hexdigest()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    name = f"maia3-{a.model}-{sha[:8]}"
    manifest = {
        "id": name, "name": f"maia3-{a.model}", "sha256": sha, "format": 1,
        "input": {"encoding": "maia3", "history": hist, "clk_ponder": 0.0},
        "outputs": ["logits_move", "logits_value"],
        "elo_range": [600, 2800], "z_prior": None, "clock_head": False,
        "base_model": {"family": "maia3", "size": a.model, "license": "AGPL-3.0"},
        "lab_commit": commit,
    }
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=1))
    pkg = ModelPackage.load(tmp, threads=8)
    res = parity(eng, pkg, sample_positions(a.n_parity))
    manifest["parity"] = res | {"tol": a.tol, "pass": bool(res["max_abs_logp_diff"] <= a.tol)}
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest["parity"]))
    if not manifest["parity"]["pass"]:
        raise SystemExit(f"parity FAILED; package left at {tmp}")
    dest = ROOT / a.out / name
    shutil.rmtree(dest, ignore_errors=True)
    tmp.rename(dest)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()

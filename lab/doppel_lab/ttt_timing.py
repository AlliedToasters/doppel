"""Wall time to build ONE player's test-time adapter (allie_ttt's recipe, one side at a time, no batching).

    python -m doppel_lab.ttt_timing --device cpu --sides 3          # CPU
    python -m doppel_lab.ttt_timing --device cuda --gpu-ok          # GPU (warden)

Sides: the first --sides Allie 1400-1600 test sides that have a full 40-game context. Phases timed separately:
encode (board tensors, CPU), train (3 epochs of the 35 train games + val CE each epoch), and the Maia-3 model load.
Each side starts from a fresh adapter, as a new user would.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
c = pl.col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--gpu-ok", action="store_true")
    ap.add_argument("--sides", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.device == "cuda" and not a.gpu_ok:
        raise SystemExit("refusing to touch the GPU without --gpu-ok (run through the warden)")
    import torch

    from doppel_lab.allie_ttt import side_rows
    from doppel_lab.engines import Maia3
    from doppel_lab.lora_probe import Lora, Positions, encode_all, logits, per_position
    if a.threads:
        torch.set_num_threads(a.threads)
    dev = a.device

    def sync():
        if dev == "cuda":
            torch.cuda.synchronize()
    rows, _ = side_rows(1500)
    full = rows.filter(c("role") != "test").group_by("pid").len().filter(c("len") == 40)["pid"].sort()
    t = time.time()
    model = Maia3("79m", dev).model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    sync()
    load_s = time.time() - t
    amp = torch.bfloat16 if dev == "cuda" else torch.float32
    out = []
    for k, pid in enumerate(full[:a.sides + 1].to_list()):   # side 0 = warm-up, not reported
        d = rows.filter(c("pid") == pid).with_columns(pid=pl.lit(0, pl.Int64))
        t0 = time.time()
        enc = encode_all(d, 1)
        P = Positions(d, enc)
        t1 = time.time()
        lora = Lora(model, 1, dev)
        opt = torch.optim.AdamW(lora.params, lr=1e-3, weight_decay=0.01)
        tr, va = np.flatnonzero(P.role == "train"), np.flatnonzero(P.role == "val")
        rng = np.random.default_rng(0)
        per_position(model, lora, P, va, None, dev, amp)              # epoch-0 val CE (Maia-3)
        for ep in range(a.epochs):
            perm = rng.permutation(tr)
            for s in range(0, len(perm), a.bs):
                idx = perm[s:s + a.bs]
                b = P.batch(idx, dev)
                lp = logits(model, b, lora, torch.zeros(len(idx), dtype=torch.long, device=dev), amp)
                loss = -lp.gather(1, b[4][:, None]).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            per_position(model, lora, P, va, lambda i: P.pid[i], dev, amp)
        sync()
        t2 = time.time()
        for h in lora.hooks:
            h.remove()
        r = {"pid": int(pid), "train_positions": int(len(tr)), "val_positions": int(len(va)),
             "encode_s": round(t1 - t0, 2), "train_s": round(t2 - t1, 2)}
        print(("warm-up " if k == 0 else "") + json.dumps(r), flush=True)
        if k:
            out.append(r)
    res = {"device": dev, "threads": torch.get_num_threads(), "epochs": a.epochs, "model_load_s": round(load_s, 2),
           "sides": out, "median_encode_s": float(np.median([r["encode_s"] for r in out])),
           "median_train_s": float(np.median([r["train_s"] for r in out]))}
    if dev == "cuda":
        res["gpu"] = torch.cuda.get_device_name(0)
        res["peak_vram_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
    p = ROOT / f"lab/results/ttt_timing_{dev}.json"
    p.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "sides"}))


if __name__ == "__main__":
    main()

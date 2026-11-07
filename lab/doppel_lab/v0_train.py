"""Train Doppel v0 on one rating band. Resumable and preemptible: a warden preemption loses at most --ckpt-every
steps, and --max-minutes stops cleanly before the warden's runtime cap (re-queue the same command to continue).

    # GPU, through the warden (memory "warden-job-form"):
    warden run --vram 16000 --max-runtime 14400 --label doppel:v0-train --cwd $PWD -- \
        uv run --project $PWD python -m doppel_lab.v0_train train --run v0a --gpu-ok
    # CPU smoke run on a fake trunk (no Maia weights, seconds):
    python -m doppel_lab.v0_train train --run smoke --fake-trunk --max-steps 20 --batch 16 --val-games 64
    # Eval a checkpoint on val (GPU via the warden, or --fake-trunk on CPU):
    python -m doppel_lab.v0_train eval --run v0a --gpu-ok

Outputs: data/v0/runs/<run>/{ckpt.pt, log.jsonl, eval.json}; eval summaries also go to lab/results/v0_<run>.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from doppel_lab.v0_data import V0, Index, Samples, collate
from doppel_lab.v0_model import DoppelV0, FakeTrunk, Trunk, legal_mask, losses

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "lab/results"


def build(fake: bool, device: str) -> DoppelV0:
    if fake:
        trunk = FakeTrunk()
    else:
        from doppel_lab.engines import Maia3
        trunk = Trunk(Maia3("79m", device).model)
    return DoppelV0(trunk).to(device)


def to_dev(b: dict, device: str) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in b.items()}


def loader(ds, batch, workers, shuffle, seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=shuffle, collate_fn=collate, num_workers=workers,
                                       pin_memory=torch.cuda.is_available(), drop_last=shuffle, generator=g,
                                       prefetch_factor=4 if workers else None,
                                       multiprocessing_context="fork" if workers else None)


@torch.no_grad()
def evaluate(model: DoppelV0, ld, device: str, max_batches: int | None = None) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    model.eval()
    acc = {k: [] for k in ("ce_t", "ce_q", "ce_p", "ce_s", "swap_ok", "game", "top_t", "top_q", "top_p", "kl",
                           "mu_q", "think_true", "think_q", "think_p", "res_y", "res_w", "res_q", "res_p")}
    for i, b in enumerate(ld):
        if max_batches and i >= max_batches:
            break
        b = to_dev(b, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            tok = b["tok"].float()
            t_logits, value, pooled = model.trunk(tok, b["elo_self"], b["elo_opp"])
            mask = legal_mask(b["legal"])
            t_logp = t_logits.float().masked_fill(~mask, -1e9).log_softmax(-1)
            mu_q, lv_q = model.encoder(b["enc"], b["enc_mask"], b["cond"])
            mu_p, lv_p = model.prior(b["cond"])
            z_p = mu_p + torch.randn_like(mu_p) * (0.5 * lv_p).exp()
            emb = model.residual.board(tok)
            sid, has = b["sid"], b["played"] >= 0
            pl_idx = b["played"].clamp(min=0)
            # z-swap control: each sample gets the posterior mean of a sample from another game in the batch.
            sg = torch.zeros(len(mu_q), dtype=b["game"].dtype, device=sid.device).index_put_((sid,), b["game"])
            swap = torch.roll(torch.arange(len(mu_q), device=sid.device), len(mu_q) // 2)
            out = {}
            for tag, z in (("q", mu_q[sid]), ("p", z_p[sid]), ("s", mu_q[swap][sid])):
                lg = (t_logp + model.residual(emb, z, b["elo_self"], b["elo_opp"]).float())
                lp = lg.masked_fill(~mask, -1e9).log_softmax(-1)
                acc[f"ce_{tag}"].append(-lp.gather(1, pl_idx[:, None])[:, 0][has].cpu())
                if tag == "s":
                    continue
                acc[f"top_{tag}"].append((lp.argmax(-1) == pl_idx)[has].cpu())
                mu_t, _ = model.think(pooled, b["clk"], b["tempo"], z)
                out[f"think_{tag}"] = model.think.seconds(mu_t.float(), b["think"][:, 1], b["think"][:, 2])
                out[f"res_{tag}"] = torch.sigmoid(model.resign(pooled, value, b["clk"], z).float())
        acc["ce_t"].append(-t_logp.gather(1, pl_idx[:, None])[:, 0][has].cpu())
        acc["top_t"].append((t_logp.argmax(-1) == pl_idx)[has].cpu())
        acc["swap_ok"].append((sg[swap][sid] != b["game"])[has].cpu())
        acc["game"].append(b["game"][has].cpu())
        kl_d = 0.5 * (lv_p - lv_q + ((lv_q.exp() + (mu_q - mu_p) ** 2) / lv_p.exp()) - 1)
        acc["kl"].append(kl_d.float().sum(-1).cpu())
        acc["mu_q"].append(mu_q.float().cpu())
        ok = has & torch.isfinite(b["think"][:, 0])
        acc["think_true"].append(b["think"][ok, 0].cpu())
        acc["think_q"].append(out["think_q"][ok].cpu())
        acc["think_p"].append(out["think_p"][ok].cpu())
        acc["res_y"].append(b["resign"].cpu())
        acc["res_w"].append(b["weight"].cpu())
        acc["res_q"].append(out["res_q"].cpu())
        acc["res_p"].append(out["res_p"].cpu())
    model.train()
    c = {k: torch.cat(v).float().numpy() for k, v in acc.items() if v}
    var = c["mu_q"].var(0)
    lt = np.log1p(c["think_true"])
    r = {"positions": int(len(c["ce_t"])), "samples": int(len(c["kl"])),
         "ce_teacher": float(c["ce_t"].mean()), "ce_post": float(c["ce_q"].mean()), "ce_prior": float(c["ce_p"].mean()),
         "top1_teacher": float(c["top_t"].mean()), "top1_post": float(c["top_q"].mean()),
         "top1_prior": float(c["top_p"].mean()),
         "kl_nats_per_half": float(c["kl"].mean()), "active_units": int((var > 0.01).sum()),
         "mu_var_by_dim": [round(float(v), 4) for v in var],
         "think_r_log_post": float(np.corrcoef(lt, np.log1p(c["think_q"]))[0, 1]),
         "think_r_log_prior": float(np.corrcoef(lt, np.log1p(c["think_p"]))[0, 1])}
    # Paired per-position gain of the player's own z over a swapped z, CI by bootstrap over games.
    ok = c["swap_ok"] > 0.5
    d, g = (c["ce_s"] - c["ce_q"])[ok], c["game"][ok].astype(np.int64)
    _, gi = np.unique(g, return_inverse=True)
    sums, cnt = np.bincount(gi, d), np.bincount(gi)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(1000):
        pick = rng.integers(0, len(sums), len(sums))
        boots.append(sums[pick].sum() / cnt[pick].sum())
    r["ce_swap"] = float(c["ce_s"][ok].mean())
    r["z_swap_gain_nats"] = float(d.mean())
    r["z_swap_gain_ci"] = [float(x) for x in np.percentile(boots, [2.5, 97.5])]
    y, w = c["res_y"], c["res_w"]
    if 0 < y.sum() < len(y):
        for tag, key in (("post", "res_q"), ("prior", "res_p")):
            p = c[key]
            r[f"resign_auc_{tag}"] = float(roc_auc_score(y, p, sample_weight=w))
            r[f"resign_ap_{tag}"] = float(average_precision_score(y, p, sample_weight=w))
        r["resign_positive_rate"] = float((y * w).sum() / w.sum())
    return r


def val_loader(args, workers):
    ix = Index(V0 / f"index_{args.band}_val.npz")
    games = np.random.default_rng(1).permutation(ix.G)[:args.val_games]
    ds = Samples(ix, K=args.K, seed=12345, games=np.sort(games))
    return loader(ds, args.batch, workers, shuffle=False)


def train(args):
    device = "cuda" if args.gpu_ok else "cpu"
    run = V0 / "runs" / args.run
    run.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    model = build(args.fake_trunk, device)
    opt = torch.optim.AdamW(model.heads_parameters(), lr=args.lr, weight_decay=0.01)
    step, epoch = 0, 0
    ck = run / "ckpt.pt"
    if ck.exists():
        s = torch.load(ck, map_location=device, weights_only=False)
        model.load_state_dict(s["model"], strict=False)
        opt.load_state_dict(s["opt"])
        step, epoch = s["step"], s["epoch"]
        print(f"resumed {ck} at step {step}, epoch {epoch}", flush=True)
    ix = Index(V0 / f"index_{args.band}_train.npz")
    ds = Samples(ix, K=args.K, seed=args.seed)
    steps_per_epoch = len(ds) // args.batch
    total = args.max_steps or steps_per_epoch * args.epochs
    vl = val_loader(args, min(args.workers, 8))
    log = (run / "log.jsonl").open("a")
    t0, deadline = time.time(), time.time() + 60 * args.max_minutes

    def save():
        tmp = ck.with_suffix(".tmp")
        state = {k: v for k, v in model.state_dict().items() if not k.startswith("trunk.")}
        torch.save({"model": state, "opt": opt.state_dict(), "step": step, "epoch": epoch, "args": vars(args)}, tmp)
        tmp.rename(ck)

    def lr_at(s):
        warm = min(1.0, s / args.warmup)
        return args.lr * warm * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))

    model.train()
    done = False
    while not done and step < total:
        ds.epoch = epoch
        skip = step - epoch * steps_per_epoch  # resume mid-epoch: skip batches already seen (same permutation)
        ld = loader(ds, args.batch, args.workers, shuffle=True, seed=args.seed * 1000 + epoch)
        for i, b in enumerate(ld):
            if i < skip:
                continue
            b = to_dev(b, device)
            beta = min(1.0, step / args.kl_warmup)
            for gp in opt.param_groups:
                gp["lr"] = lr_at(step)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                loss, parts = losses(model, b, beta=beta, free_bits=args.free_bits, alpha=args.alpha)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.heads_parameters(), 1.0)
            opt.step()
            step += 1
            if step % args.log_every == 0:
                el = time.time() - t0
                rec = {"step": step, "epoch": epoch, "beta": round(beta, 3), "lr": lr_at(step), "grad_norm": gn.item(),
                       "positions": int(b["tok"].shape[0]), "elapsed_s": round(el), **parts}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(json.dumps(rec), flush=True)
            if step % args.ckpt_every == 0:
                save()
            if step % args.val_every == 0 or step == total:
                r = evaluate(model, vl, device, max_batches=args.val_batches)
                log.write(json.dumps({"step": step, "val": r}) + "\n")
                log.flush()
                print("VAL", json.dumps(r), flush=True)
            if step >= total:
                done = True
                break
            if time.time() > deadline:
                print(f"time budget reached at step {step}; checkpointing, re-queue to continue", flush=True)
                save()
                return
        else:
            epoch += 1
    save()
    print(f"training done at step {step}", flush=True)


def eval_cmd(args):
    device = "cuda" if args.gpu_ok else "cpu"
    run = V0 / "runs" / args.run
    model = build(args.fake_trunk, device)
    s = torch.load(run / "ckpt.pt", map_location=device, weights_only=False)
    model.load_state_dict(s["model"], strict=False)
    r = {"run": args.run, "step": s["step"], **evaluate(model, val_loader(args, args.workers), device)}
    (run / "eval.json").write_text(json.dumps(r, indent=2) + "\n")
    if not args.fake_trunk:
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / f"v0_{args.run}.json").write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("train", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--run", required=True)
        p.add_argument("--band", type=int, default=1500)
        p.add_argument("--gpu-ok", action="store_true")
        p.add_argument("--fake-trunk", action="store_true")
        p.add_argument("--batch", type=int, default=256, help="samples (game, colour, half) per step")
        p.add_argument("--K", type=int, default=8, help="target positions per sample")
        p.add_argument("--workers", type=int, default=int(os.environ.get("V0_WORKERS", 20)))
        p.add_argument("--val-games", type=int, default=4000)
        p.add_argument("--seed", type=int, default=0)
    t = sub.choices["train"]
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--max-steps", type=int, default=0)
    t.add_argument("--max-minutes", type=float, default=225)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--warmup", type=int, default=500)
    t.add_argument("--kl-warmup", type=int, default=5000)
    t.add_argument("--free-bits", type=float, default=3.0)
    t.add_argument("--alpha", type=float, default=0.5)
    t.add_argument("--log-every", type=int, default=50)
    t.add_argument("--ckpt-every", type=int, default=1000)
    t.add_argument("--val-every", type=int, default=2000)
    t.add_argument("--val-batches", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "train":
        a.val_batches = a.val_batches or None
        train(a)
    else:
        eval_cmd(a)

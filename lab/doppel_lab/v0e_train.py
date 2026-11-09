"""Train / evaluate Doppel v0e (v0e_model): v0d plus a set encoder over the player's earlier games.

    # GPU, through the warden (memory "warden-job-form"), after v0e_vec run and v0e_data prep / norm:
    warden run --vram 20000 --max-runtime 14400 --label doppel:v0e-train --cwd $PWD -- \
        uv run --project $PWD python -m doppel_lab.v0e_train train --run v0e --gpu-ok
    python -m doppel_lab.v0e_train eval --run v0e --gpu-ok        # -> lab/results/v0_v0e.json (v0_gate input)
    # CPU smoke on a fake trunk and random context vectors (seconds):
    python -m doppel_lab.v0e_train train --run smoke-v0e --fake-trunk --fake-vecs --max-steps 5 --batch 8

Same loop, tripwires and resume rules as v0_train. Additions: init from the v0d checkpoint, beta capped at
--beta-max, and the KL logged per context-size bucket (it should grow with k). Validation is v0e_data's
EvalSamples on the valrich split: fixed targets (history positions 40-47 of 600 held-out players) at fixed k,
so every number in the eval is also reported by k.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from doppel_lab.v0_data import V0
from doppel_lab.v0_model import FakeTrunk, Trunk, legal_mask, losses
from doppel_lab.v0_train import to_dev
from doppel_lab.v0e_data import EMB_DIM, KS_EVAL, V0E, EvalSamples, Split, TrainSamples, collate
from doppel_lab.v0e_model import DoppelV0e, contrastive, init_from_v0d

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "lab/results"
BUCKETS = ((0, 0), (1, 1), (2, 4), (5, 9), (10, 19), (20, 40))


def bucket_name(lo, hi):
    return str(lo) if lo == hi else f"{lo}-{hi}"


class FakeVecs:
    """Random context vectors for CPU smoke runs (no GPU precompute needed)."""

    def __init__(self, game_ids, seed: int = 0):
        rng = np.random.default_rng(seed)
        G = len(game_ids)
        self.game_ids = game_ids
        self.emb = rng.standard_normal((G, 2, EMB_DIM)).astype(np.float16)
        self.scal = rng.standard_normal((G, 2, 5)).astype(np.float32)
        self.sweep = rng.standard_normal((G, 2, 3)).astype(np.float32)


def load_split(name: str, fake_vecs: bool, frac: float = 1.0) -> Split:
    if fake_vecs:
        from doppel_lab.v0_data import Index
        ix = Index(V0 / f"index_1500_{name}.npz" if name in ("train", "val") else V0E / f"index_1500_{name}.npz")
        return Split(name, vec=FakeVecs(ix.game_id), frac=frac)
    return Split(name, frac=frac)


def build(fake: bool, device: str, norm: bool = True) -> DoppelV0e:
    if fake:
        trunk = FakeTrunk()
    else:
        from doppel_lab.engines import Maia3
        trunk = Trunk(Maia3("79m", device).model)
    m = DoppelV0e(trunk)
    nf = V0E / "norm_1500.npz"
    if norm and nf.exists():
        m.encoder.set_norm(nf)
    return m.to(device)


def loader(ds, batch, workers, shuffle, seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=shuffle, collate_fn=collate, num_workers=workers,
                                       pin_memory=torch.cuda.is_available(), drop_last=shuffle, generator=g,
                                       prefetch_factor=4 if workers else None,
                                       multiprocessing_context="fork" if workers else None)


def kl_by_k(kl: np.ndarray, k: np.ndarray) -> dict:
    out = {}
    for lo, hi in BUCKETS:
        m = (k >= lo) & (k <= hi)
        if m.any():
            out[bucket_name(lo, hi)] = round(float(kl[m].mean()), 4)
    return out


def same_k_swap(nctx: torch.Tensor) -> torch.Tensor:
    """For each sample, a partner with the same k half a k-group away (a different target, usually player)."""
    idx = torch.arange(len(nctx), device=nctx.device)
    out = idx.clone()
    for k in nctx.unique():
        g = idx[nctx == k]
        out[g] = torch.roll(g, max(1, len(g) // 2))
    return out


@torch.no_grad()
def evaluate(model: DoppelV0e, ld, device: str, max_batches: int | None = None) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    model.eval()
    acc = {k: [] for k in ("ce_t", "ce_q", "ce_p", "ce_s", "swap_ok", "tgt", "pos_k", "top_t", "top_q", "top_p",
                           "kl", "k", "mu_q", "think_true", "think_k", "think_q", "think_p", "res_y", "res_w",
                           "res_q", "res_p")}
    for i, b in enumerate(ld):
        if max_batches and i >= max_batches:
            break
        b = to_dev(b, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            tok = b["tok"].float()
            t_logits, value, pooled = model.trunk(tok, b["elo_self"], b["elo_opp"])
            mask = legal_mask(b["legal"])
            t_logp = t_logits.float().masked_fill(~mask, -1e9).log_softmax(-1)
            mu_q, lv_q = model.posterior(b)
            mu_p, lv_p = model.prior(b["cond"])
            z_p = mu_p + torch.randn_like(mu_p) * (0.5 * lv_p).exp()
            emb = model.residual.board(tok)
            sid, has = b["sid"], b["played"] >= 0
            pl_idx = b["played"].clamp(min=0)
            swap = same_k_swap(b["nctx"])
            out = {}
            for tag, z in (("q", mu_q[sid]), ("p", z_p[sid]), ("s", mu_q[swap][sid])):
                lg = (t_logp + model.residual(emb, z, b["elo_self"], b["elo_opp"], mask).float())
                lp = lg.masked_fill(~mask, -1e9).log_softmax(-1)
                acc[f"ce_{tag}"].append(-lp.gather(1, pl_idx[:, None])[:, 0][has].cpu())
                if tag == "s":
                    continue
                acc[f"top_{tag}"].append((lp.argmax(-1) == pl_idx)[has].cpu())
                mu_t, _ = model.think(pooled, b["clk"], b["tempo"], z)
                out[f"think_{tag}"] = model.think.seconds(mu_t.float(), b["think"][:, 1], b["think"][:, 2])
                out[f"res_{tag}"] = torch.sigmoid(model.resign(pooled, value, b["clk"], z).float())
        k = b["nctx"]
        acc["ce_t"].append(-t_logp.gather(1, pl_idx[:, None])[:, 0][has].cpu())
        acc["top_t"].append((t_logp.argmax(-1) == pl_idx)[has].cpu())
        acc["swap_ok"].append((b["pid"][swap] != b["pid"])[sid][has].cpu())
        acc["tgt"].append((b["game"] * 2 + (b["ply"] % 2))[has].cpu())
        acc["pos_k"].append(k[sid][has].cpu())
        kl_d = 0.5 * (lv_p - lv_q + ((lv_q.exp() + (mu_q - mu_p) ** 2) / lv_p.exp()) - 1)
        acc["kl"].append(kl_d.float().sum(-1).cpu())
        acc["k"].append(k.cpu())
        acc["mu_q"].append(torch.where((k > 0)[:, None], mu_q, torch.nan).float().cpu())
        ok = has & torch.isfinite(b["think"][:, 0])
        acc["think_true"].append(b["think"][ok, 0].cpu())
        acc["think_k"].append(k[sid][ok].cpu())
        acc["think_q"].append(out["think_q"][ok].cpu())
        acc["think_p"].append(out["think_p"][ok].cpu())
        acc["res_y"].append(b["resign"].cpu())
        acc["res_w"].append(b["weight"].cpu())
        acc["res_q"].append(out["res_q"].cpu())
        acc["res_p"].append(out["res_p"].cpu())
    model.train()
    c = {k: torch.cat(v).float().numpy() for k, v in acc.items() if v}
    mu = c["mu_q"][np.isfinite(c["mu_q"][:, 0])]
    var = mu.var(0) if len(mu) else np.zeros(model.z_dim)
    lt = np.log1p(c["think_true"])
    pos = c["pos_k"] > 0
    ksample = c["k"]
    r = {"positions": int(len(c["ce_t"])), "samples": int(len(c["kl"])),
         "ce_teacher": float(c["ce_t"].mean()), "ce_post": float(c["ce_q"][pos].mean()),
         "ce_prior": float(c["ce_p"].mean()), "top1_teacher": float(c["top_t"].mean()),
         "top1_post": float(c["top_q"][pos].mean()), "top1_prior": float(c["top_p"].mean()),
         # gate's KL check: mean KL per target game over samples with k > 0 (v0d's was per half-game)
         "kl_nats_per_half": float(c["kl"][ksample > 0].mean()), "kl_nats_by_k": kl_by_k(c["kl"], ksample),
         "active_units": int((var > 0.01).sum()), "mu_var_by_dim": [round(float(v), 4) for v in var],
         "think_r_log_post": float(np.corrcoef(lt[c["think_k"] > 0], np.log1p(c["think_q"][c["think_k"] > 0]))[0, 1]),
         "think_r_log_prior": float(np.corrcoef(lt, np.log1p(c["think_p"]))[0, 1])}
    rng = np.random.default_rng(0)

    def gain(sel):
        if not sel.any():
            return None, None
        d, g = (c["ce_s"] - c["ce_q"])[sel], c["tgt"][sel].astype(np.int64)
        _, gi = np.unique(g, return_inverse=True)
        sums, cnt = np.bincount(gi, d), np.bincount(gi)
        boots = [sums[p].sum() / cnt[p].sum() for p in (rng.integers(0, len(sums), len(sums)) for _ in range(1000))]
        return float(d.mean()), [float(x) for x in np.percentile(boots, [2.5, 97.5])]

    ok = (c["swap_ok"] > 0.5) & pos
    r["ce_swap"] = float(c["ce_s"][ok].mean()) if ok.any() else None
    r["z_swap_gain_nats"], r["z_swap_gain_ci"] = gain(ok)
    by = {}
    for kk in np.unique(c["pos_k"]).astype(int):
        m = c["pos_k"] == kk
        tm = c["think_k"] == kk
        e = {"positions": int(m.sum()), "kl": float(c["kl"][ksample == kk].mean()),
             "ce_post_minus_teacher": float((c["ce_q"] - c["ce_t"])[m].mean()),
             "top1_post": float(c["top_q"][m].mean()),
             "think_r_log_post": float(np.corrcoef(lt[tm], np.log1p(c["think_q"][tm]))[0, 1])}
        if kk > 0:
            e["z_swap_gain_nats"], e["z_swap_gain_ci"] = gain(m & (c["swap_ok"] > 0.5))
        by[str(kk)] = e
    r["by_k"] = by
    y, w = c["res_y"], c["res_w"]
    if 0 < y.sum() < len(y):
        for tag, key in (("post", "res_q"), ("prior", "res_p")):
            r[f"resign_auc_{tag}"] = float(roc_auc_score(y, c[key], sample_weight=w))
            r[f"resign_ap_{tag}"] = float(average_precision_score(y, c[key], sample_weight=w))
        r["resign_positive_rate"] = float((y * w).sum() / w.sum())
    return r


def val_loader(args, workers, full: bool = False):
    sp = load_split("valrich", args.fake_vecs)
    ds = EvalSamples(sp, ks=KS_EVAL if full else (0, 2, 10, 40), K=args.K,
                     max_players=None if full else args.val_players)
    return loader(ds, args.batch, workers, shuffle=False)


def train(args):
    device = "cuda" if args.gpu_ok else "cpu"
    run = V0 / "runs" / args.run
    run.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    if not args.fake_vecs and not (V0E / "norm_1500.npz").exists():
        from doppel_lab.v0e_data import norm
        norm()                               # context-feature standardisation from the train vectors
    model = build(args.fake_trunk, device)
    named = [(n, p) for n, p in model.named_parameters() if not n.startswith("trunk.") and p.requires_grad]
    res_p = [p for n, p in named if n.startswith("residual.")]
    oth_p = [p for n, p in named if not n.startswith("residual.")]
    opt = torch.optim.AdamW([{"params": res_p, "scale": args.res_lr_mult, "name": "residual"},
                             {"params": oth_p, "scale": 1.0, "name": "other"}], lr=args.lr, weight_decay=0.01)
    step, epoch = 0, 0
    ck = run / "ckpt.pt"
    log = (run / "log.jsonl").open("a")
    if ck.exists():
        s = torch.load(ck, map_location=device, weights_only=False)
        model.load_state_dict(s["model"], strict=False)
        opt.load_state_dict(s["opt"])
        step, epoch = s["step"], s["epoch"]
        print(f"resumed {ck} at step {step}, epoch {epoch}", flush=True)
    elif args.init:
        rep = init_from_v0d(model, V0 / "runs" / args.init / "ckpt.pt")
        msg = {"init": args.init, "copied": len(rep["copied"]), "partial": rep["partial"], "skipped": rep["skipped"]}
        log.write(json.dumps(msg) + "\n")
        print(json.dumps(msg), flush=True)
    ds = TrainSamples(load_split(args.train_split, args.fake_vecs, args.train_frac), K=args.K, seed=args.seed)
    steps_per_epoch = len(ds) // args.batch
    total = args.max_steps or steps_per_epoch * args.epochs
    vl = val_loader(args, min(args.workers, 8))
    t0, deadline = time.time(), time.time() + 60 * args.max_minutes

    def save():
        tmp = ck.with_suffix(".tmp")
        state = {k: v for k, v in model.state_dict().items() if not k.startswith("trunk.")}
        torch.save({"model": state, "opt": opt.state_dict(), "step": step, "epoch": epoch,
                    "args": vars(args) | {"arch": "v0e"}}, tmp)
        tmp.rename(ck)

    def lr_at(s):
        warm = min(1.0, s / args.warmup)
        return args.lr * warm * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))

    model.train()
    done, sat_run, worse_run = False, 0, 0
    kl_acc, k_acc = [], []
    while not done and step < total:
        ds.epoch = epoch
        skip = step - epoch * steps_per_epoch
        ld = loader(ds, args.batch, args.workers, shuffle=True, seed=args.seed * 1000 + epoch)
        for i, b in enumerate(ld):
            if i < skip:
                continue
            b = to_dev(b, device)
            beta = args.beta_max * min(1.0, step / args.kl_warmup)
            for gp in opt.param_groups:
                gp["lr"] = lr_at(step) * gp["scale"]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                loss, parts = losses(model, b, beta=beta, free_bits=args.free_bits, alpha=args.alpha)
                if args.w_con > 0:                   # v0f: identity loss on the set encoder (v0e_model.contrastive)
                    L_con, con_acc, n_con = contrastive(model, b, tau=args.tau)
                    loss = loss + args.w_con * L_con
                    parts.update(con=L_con.item(), con_acc=con_acc.item(), con_n=n_con, loss=loss.item())
            kl_acc.append(parts.pop("_kl").float().cpu())
            k_acc.append(b["nctx"].cpu())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn_res = torch.nn.utils.clip_grad_norm_(res_p, 1.0)
            gn = torch.nn.utils.clip_grad_norm_(oth_p, 1.0)
            opt.step()
            step += 1
            if step % args.log_every == 0:
                el = time.time() - t0
                kk, kl = torch.cat(k_acc).numpy(), torch.cat(kl_acc).numpy()
                kl_acc, k_acc = [], []
                rec = {"step": step, "epoch": epoch, "beta": round(beta, 4), "lr": lr_at(step), "grad_norm": gn.item(),
                       "grad_norm_res": gn_res.item(), "positions": int(b["tok"].shape[0]), "elapsed_s": round(el),
                       "mean_k": round(float(kk.mean()), 2), **parts, "kl_by_k": kl_by_k(kl, kk)}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(json.dumps(rec), flush=True)
                sat_run = sat_run + 1 if parts["res_sat"] > args.sat_abort else 0
                if sat_run >= 5:
                    save()
                    msg = f"ABORT: residual saturated (res_sat {parts['res_sat']:.2f} > {args.sat_abort}) for 5 logs at step {step}"
                    log.write(json.dumps({"step": step, "abort": msg}) + "\n")
                    log.flush()
                    print(msg, flush=True)
                    sys.exit(3)
                if not math.isfinite(parts["loss"]):
                    save()
                    print(f"ABORT: non-finite loss at step {step}", flush=True)
                    sys.exit(5)
            if step % args.ckpt_every == 0:
                save()
            if step % args.val_every == 0 or step == total:
                r = evaluate(model, vl, device, max_batches=args.val_batches)
                log.write(json.dumps({"step": step, "val": r}) + "\n")
                log.flush()
                print("VAL", json.dumps(r), flush=True)
                gap = r["ce_post"] - r["ce_teacher"]
                worse_run = worse_run + 1 if step >= args.teacher_abort_after and gap > args.teacher_abort else 0
                if worse_run >= 2:
                    save()
                    msg = f"ABORT: val ce_post exceeds teacher by {gap:.4f} > {args.teacher_abort} for 2 vals at step {step}"
                    log.write(json.dumps({"step": step, "abort": msg}) + "\n")
                    log.flush()
                    print(msg, flush=True)
                    sys.exit(4)
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
    r = {"run": args.run, "step": s["step"], **evaluate(model, val_loader(args, args.workers, full=True), device)}
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
        p.add_argument("--gpu-ok", action="store_true")
        p.add_argument("--fake-trunk", action="store_true")
        p.add_argument("--fake-vecs", action="store_true", help="random context vectors (CPU smoke)")
        p.add_argument("--batch", type=int, default=256, help="samples (target game, colour) per step")
        p.add_argument("--K", type=int, default=8, help="target positions per sample")
        p.add_argument("--workers", type=int, default=int(os.environ.get("V0_WORKERS", 20)))
        p.add_argument("--val-players", type=int, default=150)
        p.add_argument("--seed", type=int, default=0)
    t = sub.choices["train"]
    t.add_argument("--train-split", default="train")
    t.add_argument("--train-frac", type=float, default=1.0,
                   help="data scaling: train on this random share of the split's games (histories shrink too)")
    t.add_argument("--init", default="v0d", help="v0d run to initialise residual/think/resign from ('' = none)")
    t.add_argument("--epochs", type=int, default=6)
    t.add_argument("--max-steps", type=int, default=0)
    t.add_argument("--max-minutes", type=float, default=225)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--res-lr-mult", type=float, default=0.1)
    t.add_argument("--teacher-abort", type=float, default=0.005)
    t.add_argument("--teacher-abort-after", type=int, default=6000)
    t.add_argument("--warmup", type=int, default=500)
    t.add_argument("--kl-warmup", type=int, default=5000)
    t.add_argument("--beta-max", type=float, default=0.25)
    t.add_argument("--free-bits", type=float, default=0.5)
    t.add_argument("--alpha", type=float, default=0.5)
    t.add_argument("--sat-abort", type=float, default=0.25)
    t.add_argument("--log-every", type=int, default=50)
    t.add_argument("--ckpt-every", type=int, default=1000)
    t.add_argument("--val-every", type=int, default=2000)
    t.add_argument("--val-batches", type=int, default=0)
    t.add_argument("--w-con", type=float, default=0.0, help="v0f: weight of the contrastive identity loss (0 = v0e)")
    t.add_argument("--tau", type=float, default=0.1, help="v0f: InfoNCE temperature")
    a = ap.parse_args()
    if a.cmd == "train":
        a.val_batches = a.val_batches or None
        train(a)
    else:
        eval_cmd(a)

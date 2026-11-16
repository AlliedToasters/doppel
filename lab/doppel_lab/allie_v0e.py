"""A v0e Doppel on the Allie test set, z built from each player's real earlier 2022 games (SCIENCE.md §8).

    python -m doppel_lab.allie_v0e select                    # CPU: test games of one rating bucket + context games
    python -m doppel_lab.v0e_vec run --split allie1500 --gpu-ok   # GPU (add --band for 1100/1900) (warden): Maia-3 vectors of every game
    python -m doppel_lab.v0e_data prep --split allie1500     # CPU: v0-format index + meta
    python -m doppel_lab.allie_v0e eval --run v0g_a --gpu-ok # GPU (warden): top-1 on every protocol position

Bucket: test games whose two players' mean Elo is in [1400, 1600) (the 200-Elo bucket around v0g's 1500 training
band; ~11% of the test set). Split allie1500 = those test games plus, for each side, the player's CTX most recent
earlier games in data/allie/history (game start strictly before the test game's created_at; test games are never
context, also not for a player's later test game). t = game start (ms), so v0e_data's history order is time order.
Scored positions: the exact Allie / Maia-3 protocol (ply >= 10, game cut at the first ply whose mover has < 30 s
before moving), every one of them (no K subsampling). Per position: teacher (frozen Maia-3 through the same trunk)
and the Doppel with posterior z from the k most recent context games, k in KS (capped at what is available; k = 0
is the prior). Per-position rows go to data/allie/v0e_positions_<run>.parquet; the summary joins Maia-3's
`top1` positions (same protocol) and the exact-repeat memory table for the memory switch on top.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from doppel_lab.ceiling import bucket, first_under_30

ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT / "data/allie/test.parquet"
PLAYERS = ROOT / "data/allie/players.parquet"
HIST = ROOT / "data/allie/history"
# model band -> test buckets [lo, hi) it is scored on (mean Elo of the two players)
BANDS = {700: (0, 1000), 1100: (1000, 1400), 1500: (1400, 1600), 1900: (1600, 2000), 2300: (2000, 99999)}
# buckets with no model trained in-band borrow the nearest band's model (out-of-band; norm comes with the model)
MODEL_BAND = {700: 1100, 2300: 1900}


def positions_path(run: str, band: int, model_band: int | None = None) -> Path:
    """Per-position rows; out-of-band scoring gets its own file so it never overwrites the model's own bucket."""
    mb = MODEL_BAND.get(band, band) if model_band is None else model_band
    return ROOT / f"data/allie/v0e_positions_{run}{f'_at{band}' if mb != band else ''}.parquet"


def split_name(band: int) -> str:
    return f"allie{band}"
CTX = 40
KS = (0, 1, 5, 10, 40)
c = pl.col


def test_games(band: int = 1500) -> pl.DataFrame:
    lo, hi = BANDS[band]
    t = pl.read_parquet(TEST).drop("white", "black")
    pp = pl.read_parquet(PLAYERS, columns=["game_id", "white", "black", "created_at"])
    t = t.join(pp, on="game_id", how="left")
    t = t.with_columns(bucket=pl.Series(bucket(t)), cut=pl.Series(first_under_30(t)))
    return t.filter((c("bucket") >= lo) & (c("bucket") < hi))


def select(band: int = 1500):
    from doppel_lab.v0e_vec import games_path
    t = test_games(band)
    all_test = set(pl.read_parquet(TEST, columns=["game_id"])["game_id"].to_list())
    sides = pl.concat([t.select("game_id", "created_at", player=c(col)) for col in ("white", "black")]).drop_nulls()
    players = sides["player"].unique()
    hist = []
    for f in sorted(HIST.glob("*.parquet")):
        h = pl.read_parquet(f).with_columns(wl=c("white").str.to_lowercase(), bl=c("black").str.to_lowercase())
        h = h.filter((c("wl").is_in(players.implode()) | c("bl").is_in(players.implode()))
                     & ~c("game_id").is_in(list(all_test)))
        hist.append(h)
    h = pl.concat(hist)
    hs = pl.concat([h.select(hid="game_id", utc_ms="utc_ms", player=c(col)) for col in ("wl", "bl")])
    ctx = (sides.join(hs, on="player").filter(c("utc_ms") < c("created_at"))
           .sort("utc_ms", descending=True).group_by("game_id", "player").head(CTX))
    keep = ctx["hid"].unique()
    hg = h.filter(c("game_id").is_in(keep.implode())).unique("game_id").rename({"utc_ms": "t"}).drop("wl", "bl")
    tg = t.with_columns(t=c("created_at")).select(hg.columns)
    g = pl.concat([tg, hg]).sort("t")
    out = games_path(split_name(band), band)
    out.parent.mkdir(parents=True, exist_ok=True)
    g.write_parquet(out)
    per = ctx.group_by("game_id", "player").len()["len"]
    print(f"{t.height:,} test games in {BANDS[band]}, {sides.height:,} sides; context games {hg.height:,}; "
          f"per side median {per.median():.0f}, sides with 0: {sides.height - per.len():,}, "
          f"{g['moves'].list.len().sum():,} positions -> {out}", flush=True)


class AllieSamples:
    def __init__(self, sp, test_ids: set, cut: dict, ks=KS):
        self.sp = sp
        gid = sp.ix.game_id
        self.is_test = np.array([x in test_ids for x in gid])
        self.items = [(g, col, k) for g in np.flatnonzero(self.is_test) for col in (0, 1) for k in ks]
        self.cut = cut

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from doppel_lab.v0_data import targets_sample
        from doppel_lab.v0e_data import CTX_FEATS, EMB_DIM
        g, col, k = self.items[i]
        sp, ix = self.sp, self.sp.ix
        n = ix.plies(g)
        cut = self.cut[str(ix.game_id[g])]
        cand = [p for p in range(col, n, 2) if 10 <= p < cut]
        if not cand:
            return None
        out = targets_sample(ix, g, col, cand, None, len(cand), np.random.default_rng(0))
        hg, hc = sp.history(g, col)
        ok = ~self.is_test[hg]
        hg, hc = hg[ok], hc[ok]
        k = min(k, len(hg))
        hg, hc = hg[len(hg) - k:], hc[len(hc) - k:]
        out["ctx_emb"] = np.asarray(sp.vec.emb[hg, hc]) if k else np.zeros((0, EMB_DIM), np.float16)
        out["ctx_feat"] = sp.feat[hg, hc] if k else np.zeros((0, CTX_FEATS), np.float32)
        out["pid"] = np.array([sp.pid[g, col]], np.int64)
        out["nctx"] = np.array([k], np.int64)
        out["kreq"] = np.full(len(out["played"]), self.items[i][2], np.int64)
        out["gidx"] = np.full(len(out["played"]), g, np.int64)
        return out


def evaluate_allie(run: str, band: int, device: str, workers: int, batch: int, model_band: int | None = None):
    import torch

    from doppel_lab.v0_model import legal_mask
    from doppel_lab.v0e_data import Split, collate
    from doppel_lab.v0_train import to_dev
    from doppel_lab.v0e_train import V0, build
    t = test_games(band)
    cut = dict(zip(t["game_id"].to_list(), t["cut"].to_list()))
    sp = Split(split_name(band), band)
    ds = AllieSamples(sp, set(t["game_id"].to_list()), cut)

    def coll(ss):
        ss = [s for s in ss if s is not None]
        b = collate(ss)
        for k in ("kreq", "gidx"):
            b[k] = torch.from_numpy(np.concatenate([s[k] for s in ss]))
        return b

    ld = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False, collate_fn=coll, num_workers=workers)
    model = build(False, device, band=MODEL_BAND.get(band, band) if model_band is None else model_band)
    s = torch.load(V0 / "runs" / run / "ckpt.pt", map_location=device, weights_only=False)
    model.load_state_dict(s["model"], strict=False)
    model.eval()
    rows = {k: [] for k in ("gidx", "ply", "kreq", "nctx", "top_t", "top_q", "ce_t", "ce_q")}
    with torch.no_grad():
        for i, b in enumerate(ld):
            b = to_dev(b, device)
            tok = b["tok"].float()
            # trunk in fp16 as in top1.py: bf16 flips ~2% of Maia-3's argmaxes vs fp32 (fp16: 0.3%; SCIENCE.md §8)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                t_logits, _, _ = model.trunk(tok, b["elo_self"], b["elo_opp"])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                mask = legal_mask(b["legal"])
                t_logp = t_logits.float().masked_fill(~mask, -1e9).log_softmax(-1)
                mu_q, _ = model.posterior(b)
                emb = model.residual.board(tok)
                sid, pl_idx = b["sid"], b["played"].clamp(min=0)
                lg = t_logp + model.residual(emb, mu_q[sid], b["elo_self"], b["elo_opp"], mask).float()
                lp = lg.masked_fill(~mask, -1e9).log_softmax(-1)
            rows["gidx"].append(b["gidx"].cpu())
            rows["ply"].append(b["ply"].cpu())
            rows["kreq"].append(b["kreq"].cpu())
            rows["nctx"].append(b["nctx"][sid].cpu())
            rows["top_t"].append((t_logp.argmax(-1) == pl_idx).cpu())
            rows["top_q"].append((lp.argmax(-1) == pl_idx).cpu())
            rows["ce_t"].append(-t_logp.gather(1, pl_idx[:, None])[:, 0].float().cpu())
            rows["ce_q"].append(-lp.gather(1, pl_idx[:, None])[:, 0].float().cpu())
            if i % 20 == 0:
                print(f"batch {i}/{len(ld)}", flush=True)
    r = pl.DataFrame({k: torch.cat(v).numpy() for k, v in rows.items()})
    r = r.with_columns(game_id=pl.Series(sp.ix.game_id[r["gidx"].to_numpy()].astype(str))).drop("gidx")
    out = positions_path(run, band, model_band)
    r.write_parquet(out)
    print(json.dumps(summary(run, band, model_band), indent=1))


def summary(run: str, band: int = 1500, model_band: int | None = None) -> dict:
    r = pl.read_parquet(positions_path(run, band, model_band))
    res = {"run": run, "band": band, "bucket": list(BANDS[band]), "by_k": {}}
    for k in KS:
        x = r.filter(c("kreq") == k)
        res["by_k"][k] = {"positions": x.height, "mean_nctx": float(x["nctx"].mean()),
                          "top1_teacher": float(x["top_t"].mean()), "top1_doppel": float(x["top_q"].mean()),
                          "ce_teacher": float(x["ce_t"].mean()), "ce_doppel": float(x["ce_q"].mean())}
    # against top1.py's Maia-3 on the same positions, and with the exact-repeat memory switch on top
    m = pl.read_parquet(ROOT / "data/allie/top1_positions_maia3-79m.parquet").select("game_id", "ply", maia_hit="hit")
    mem = pl.read_parquet(ROOT / "data/allie/memory_positions.parquet").select(
        "game_id", ply="k", hist="hist", own_hit="own_hit")
    x = (r.filter(c("kreq") == max(KS)).join(m, on=["game_id", "ply"], how="left")
         .join(mem, on=["game_id", "ply"], how="left").with_columns(c("hist").fill_null(False)))
    sw = pl.when(c("hist")).then(c("own_hit")).otherwise(c("top_q"))
    res["joined"] = {"positions": x.height, "maia3_top1py": float(x["maia_hit"].mean()),
                     "teacher_here": float(x["top_t"].mean()), "doppel": float(x["top_q"].mean()),
                     "doppel_plus_memory": float(x.select(sw.mean()).item()),
                     "maia3_plus_memory": float(x.select(pl.when(c("hist")).then(c("own_hit"))
                                                         .otherwise(c("maia_hit")).mean()).item())}
    return res


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    e = sub.add_parser("eval")
    e.add_argument("--run", default="v0g_a")
    e.add_argument("--gpu-ok", action="store_true")
    e.add_argument("--workers", type=int, default=8)
    e.add_argument("--batch", type=int, default=64)
    sm = sub.add_parser("summary")
    sm.add_argument("--run", default="v0g_a")
    for q in (s, e, sm):
        q.add_argument("--band", type=int, default=1500, choices=sorted(BANDS))
    for q in (e, sm):
        q.add_argument("--model-band", type=int, default=None,
                       help="band of the run's model (norm); default MODEL_BAND.get(band, band). In-band 2300: 2300")
    a = ap.parse_args()
    if a.cmd == "select":
        select(a.band)
    elif a.cmd == "eval":
        if not a.gpu_ok:
            raise SystemExit("refusing to touch the GPU without --gpu-ok (run through the warden)")
        evaluate_allie(a.run, a.band, "cuda", a.workers, a.batch, a.model_band)
    else:
        print(json.dumps(summary(a.run, a.band, a.model_band), indent=1))


if __name__ == "__main__":
    main()

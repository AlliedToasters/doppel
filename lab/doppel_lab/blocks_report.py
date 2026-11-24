"""Block-subset adapters vs the 4-block ctx200 adapters, on the same test positions (CPU).

Inputs: data/allie/ttt_<band>_ctx200_b<blocks>_fit0_s400.npz (allie_ttt run --ctx 200 --sample 400 --blocks ...)
and the original data/allie/ttt_<band>_ctx200.npz (all sides, 4 blocks). Joined on (game_id, ply); top-1 gain over
Maia-3 for both arms and the paired difference, game bootstrap.
  python -m doppel_lab.blocks_report [--blocks 5]
-> lab/results/adapter_blocks.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "lab/results/adapter_blocks.json"


def compare(band: int, blocks: str, boots: int = 1000) -> dict | None:
    f = ROOT / f"data/allie/ttt_{band}_ctx200_b{blocks}_fit0_s400.npz"
    if not f.exists():
        return None
    z, o = np.load(f, allow_pickle=True), np.load(ROOT / f"data/allie/ttt_{band}_ctx200.npz", allow_pickle=True)
    a = pl.DataFrame({"game_id": z["game_id"].astype(str), "ply": z["ply"], "maia": z["maia"].astype(np.float64),
                      "sub": z["own"].astype(np.float64), "sub_ctrl": z["ctrl"].astype(np.float64)})
    b = pl.DataFrame({"game_id": o["game_id"].astype(str), "ply": o["ply"], "full": o["own"].astype(np.float64)})
    j = a.join(b, on=["game_id", "ply"], how="inner")
    gi = np.unique(j["game_id"].to_numpy(), return_inverse=True)[1]
    rng = np.random.default_rng(band)
    bi = rng.integers(0, gi.max() + 1, (boots, gi.max() + 1))
    gn = np.bincount(gi)

    def stat(x):
        gs = np.bincount(gi, x)
        bs = 100 * gs[bi].sum(1) / gn[bi].sum(1)
        return {"pts": round(100 * float(x.mean()), 3),
                "ci95": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]}
    m = j["maia"].to_numpy()
    return {"positions": j.height, "unmatched": a.height - j.height, "maia_top1": round(100 * float(m.mean()), 2),
            "full_4_blocks": stat(j["full"].to_numpy() - m), f"blocks_{blocks}": stat(j["sub"].to_numpy() - m),
            f"blocks_{blocks}_control": stat(j["sub_ctrl"].to_numpy() - m),
            f"blocks_{blocks}_minus_full": stat(j["sub"].to_numpy() - j["full"].to_numpy())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", nargs="+", default=["5"])
    a = ap.parse_args()
    rep = json.loads(OUT.read_text()) if OUT.exists() else {}
    for bl in a.blocks:
        for band in (700, 1100, 1500, 1900, 2300):
            r = compare(band, bl)
            if r is not None:
                rep.setdefault(f"b{bl}", {})[str(band)] = r
                print(bl, band, json.dumps(r), flush=True)
    OUT.write_text(json.dumps(rep, indent=1))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()

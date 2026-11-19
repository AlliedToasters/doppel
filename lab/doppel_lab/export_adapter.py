"""Write deployable adapter files (doppel_model.adapter) from a lab adapter dir, for a format-2 package.

    .venv/bin/python -m doppel_lab.export_adapter --from data/allie/adapters_1500_ctx200_fit5_s200 \
        --package models/maia3-79m-lora8-b6bee68c --rows 0 1 2 --out adapters

Lab adapter dirs (allie_ttt save_adapters) hold sides by chunk, A (n, r, in) and B (n, out, r) in recipe param
order; row = side index across chunks in file order. The recipe must match the package's lora spec. Meta keeps the
source dir and row only (lab files carry no usernames).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from doppel_model.adapter import save_adapter


def rows_of(adir: Path):
    """Yield (row, arrays) over every side in a lab adapter dir."""
    row = 0
    for f in sorted(adir.glob("chunk_*.npz")):
        with np.load(f) as z:
            keys = sorted((k for k in z.files if k[0] == "p" and k[1:].isdigit()), key=lambda k: int(k[1:]))
            ps = [z[k] for k in keys]
        for i in range(len(ps[0])):
            yield row, [p[i] for p in ps]
            row += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", type=Path, required=True)
    ap.add_argument("--package", type=Path, required=True)
    ap.add_argument("--rows", type=int, nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path("adapters"))
    a = ap.parse_args()
    man = json.loads((a.package / "manifest.json").read_text())
    spec = man["lora"]
    recipe = json.loads((a.src / "recipe.json").read_text())
    for k in ("blocks", "layers", "rank", "scale"):
        if recipe[k] != spec[k]:
            raise SystemExit(f"recipe {k}={recipe[k]} but package has {spec[k]}")
    want = set(a.rows)
    for row, arrs in rows_of(a.src):
        if row in want:
            aid = save_adapter(a.out, spec, arrs, {"package": man["name"], "rank": spec["rank"],
                                                   "blocks": spec["blocks"], "layers": spec["layers"],
                                                   "made": time.strftime("%Y-%m-%d"),
                                                   "source": f"{a.src.name} row {row}"})
            print(f"row {row} -> {a.out / (aid + '.npz')}")
            want.discard(row)
    if want:
        raise SystemExit(f"rows not found: {sorted(want)}")


if __name__ == "__main__":
    main()

import time, numpy as np
from pathlib import Path
from doppel_model import ModelPackage
from doppel_lab.export_package import adapter_row, sample_positions
pkg = ModelPackage.load(Path("models/maia3-79m-lora8-b6bee68c"), threads=4)
ab = adapter_row(Path("data/allie/adapters_1500_ctx200_fit5_s200"))
pos = sample_positions(200, seed=1)
d, flips, tb, td = [], 0, [], []
for moves, we, be in pos:
    t = time.perf_counter(); _, legal, base = pkg.policy(list(moves), we, be); tb.append(time.perf_counter() - t)
    t = time.perf_counter(); _, _, dop = pkg.doppel_policy(list(moves), we, be, ab); td.append(time.perf_counter() - t)
    _, _, own = pkg.policy(list(moves), we, be, adapter=ab)
    d.append(np.abs(own - base).max()); flips += int(base.argmax() != own.argmax())
print(f"adapter vs base: median max|dlogp| {np.median(d):.3f}, top-1 changed on {flips}/{len(pos)}")
print(f"ms per move, 4 threads: base {1000*np.median(tb):.1f}, doppel (2 runs + blend) {1000*np.median(td):.1f}")

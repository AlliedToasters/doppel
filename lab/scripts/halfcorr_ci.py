"""Bootstrap CIs for the within-game persistence signal (Spearman corr of a player's ACPL in the two halves
of their moves), averaged over bands, and for the blunder dispersion index. Usage: games... -- evals..."""
import sys
import numpy as np
import polars as pl
from scipy import stats
from doppel_lab import metrics as M

i = sys.argv.index("--")
gfiles, efiles = sys.argv[1:i], sys.argv[i + 1:]
cols = ["game_id", "band", "moves", "result", "termination", "source"]
games = pl.concat([pl.read_parquet(f).with_columns(
    pl.col("source") if "source" in pl.read_parquet_schema(f) else pl.lit("human").alias("source")).select(cols)
    for f in gfiles])
df = games.join(pl.concat([pl.read_parquet(f) for f in efiles]), on="game_id")
per = {}
for row in df.iter_rows(named=True):
    for p in M.game_stats(row, row["evals"])["players"]:
        per.setdefault((row["source"], row["band"]), []).append((p["acpl_h1"], p["acpl_h2"], p["blunders"]))
rng = np.random.default_rng(0)
for src in sorted({s for s, _ in per}):
    bands = sorted(b for s, b in per if s == src)
    boots_c, boots_d, pt_c, pt_d = [], [], [], []
    for b in bands:
        a = np.array(per[(src, b)])
        pt_c.append(stats.spearmanr(a[:, 0], a[:, 1]).statistic)
        pt_d.append(a[:, 2].var() / a[:, 2].mean())
    for _ in range(300):
        cs, ds = [], []
        for b in bands:
            a = np.array(per[(src, b)])
            s = a[rng.integers(0, len(a), len(a))]
            cs.append(stats.spearmanr(s[:, 0], s[:, 1]).statistic)
            ds.append(s[:, 2].var() / s[:, 2].mean())
        boots_c.append(np.mean(cs)); boots_d.append(np.mean(ds))
    lo, hi = np.percentile(boots_c, [2.5, 97.5]); dlo, dhi = np.percentile(boots_d, [2.5, 97.5])
    print(f"{src:12} half-corr {np.mean(pt_c):+.3f} [{lo:+.3f}, {hi:+.3f}]   blunder dispersion {np.mean(pt_d):.2f} [{dlo:.2f}, {dhi:.2f}]")

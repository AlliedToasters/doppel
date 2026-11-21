"""Full Allie test set: paired tests of Doppel (+ memory) vs Maia-3 run in the same pass (McNemar + game bootstrap)."""
import polars as pl, numpy as np, math
c = pl.col
mem = pl.read_parquet("data/allie/memory_positions.parquet").select("game_id", ply="k", hist="hist", own="own_hit")
runs = ["v0g_1100_at700", "v0g_1100", "v0g_b", "v0g_1900", "v0g_1900_at2300"]
x = pl.concat([pl.read_parquet(f"data/allie/v0e_positions_{r}.parquet").filter(c("kreq") == 40)
               .select("game_id", "ply", "top_t", "top_q") for r in runs])
x = x.join(mem, on=["game_id", "ply"], how="left").with_columns(c("hist").fill_null(False))
x = x.with_columns(m=c("top_t").cast(pl.Int8), d=c("top_q").cast(pl.Int8),
                   dm=pl.when(c("hist")).then(c("own")).otherwise(c("top_q")).cast(pl.Int8),
                   mm=pl.when(c("hist")).then(c("own")).otherwise(c("top_t")).cast(pl.Int8))
print(f"{x.height:,} positions, {x['game_id'].n_unique():,} games; dup keys: {x.height - x.unique(['game_id','ply']).height}")
for a, b in (("d", "m"), ("dm", "m"), ("dm", "mm")):
    n01 = x.filter((c(a) == 1) & (c(b) == 0)).height; n10 = x.filter((c(a) == 0) & (c(b) == 1)).height
    chi = (abs(n01 - n10) - 1) ** 2 / (n01 + n10); p = math.erfc(math.sqrt(chi / 2))
    g = x.group_by("game_id").agg((c(a).cast(pl.Float64) - c(b)).sum().alias("v"), pl.len().alias("n"))
    v, n = g["v"].to_numpy(), g["n"].to_numpy(); idx = np.random.default_rng(0).integers(0, len(g), (2000, len(g)))
    bs = v[idx].sum(1) / n[idx].sum(1) * 100
    print(f"{a} vs {b}: {100*x[a].mean():.3f} vs {100*x[b].mean():.3f} = {100*(x[a].mean()-x[b].mean()):+.3f} pts | "
          f"discordant {n01:,} vs {n10:,}, McNemar chi2 {chi:.1f}, p {p:.1e} | game bootstrap 95% "
          f"[{np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}]")

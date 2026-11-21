import polars as pl, numpy as np, sys
c = pl.col
mem = pl.read_parquet("data/allie/memory_positions.parquet").select("game_id", ply="k", hist="hist", own="own_hit")
for run in sys.argv[1:]:
    r = pl.read_parquet(f"data/allie/v0e_positions_{run}.parquet").filter(c("kreq") == 40)
    x = r.join(mem, on=["game_id", "ply"], how="left").with_columns(c("hist").fill_null(False))
    x = x.with_columns(dm=pl.when(c("hist")).then(c("own")).otherwise(c("top_q")).cast(pl.Float64),
                       mm=pl.when(c("hist")).then(c("own")).otherwise(c("top_t")).cast(pl.Float64))
    cols = {"doppel_vs_maia": ("top_q", "top_t"), "doppel+mem_vs_maia": ("dm", "top_t"), "doppel+mem_vs_maia+mem": ("dm", "mm"),
            "nats(maia-doppel)": ("ce_t", "ce_q")}
    g = x.with_columns([(c(p).cast(pl.Float64) - c(q).cast(pl.Float64)).alias(k) for k, (p, q) in cols.items()]) \
         .group_by("game_id").agg([c(k).sum() for k in cols] + [pl.len().alias("n")])
    n = g["n"].to_numpy(); rng = np.random.default_rng(0); idx = rng.integers(0, len(g), (2000, len(g))); N = n[idx].sum(1)
    print(f"== {run}: {x.height:,} positions, {len(g):,} games, hist share {x['hist'].mean():.3f}")
    for k in cols:
        s = 1 if k.startswith("nats") else 100
        v = g[k].to_numpy(); est = v.sum() / n.sum() * s; bs = v[idx].sum(1) / N * s
        print(f"  {k:24s} {est:+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]")

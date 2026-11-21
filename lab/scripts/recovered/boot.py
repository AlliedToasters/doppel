import polars as pl, numpy as np
c=pl.col
def load(run):
    r=pl.read_parquet(f"data/allie/v0e_positions_{run}.parquet").filter(c("kreq")==40)
    m=pl.read_parquet("data/allie/top1_positions_maia3-79m.parquet").select("game_id","ply",maia="hit")
    mem=pl.read_parquet("data/allie/memory_positions.parquet").select("game_id",ply="k",hist="hist",own="own_hit")
    x=r.join(m,on=["game_id","ply"],how="left").join(mem,on=["game_id","ply"],how="left").with_columns(c("hist").fill_null(False))
    return x.with_columns(dm=pl.when(c("hist")).then(c("own")).otherwise(c("top_q")).cast(pl.Float64),
        mm=pl.when(c("hist")).then(c("own")).otherwise(c("maia")).cast(pl.Float64)).sort("game_id","ply","ce_t")
a,b=load("v0g_a"),load("v0g_b")
x=a.select("game_id","ply","maia","mm",qa="top_q",da="dm").with_columns(qb=b["top_q"],db=b["dm"])
assert (a["game_id"]==b["game_id"]).all()
cols={"b_vs_maia":("qb","maia"),"b+mem_vs_maia":("db","maia"),"b+mem_vs_maia+mem":("db","mm"),"b_vs_a":("qb","qa")}
g=x.with_columns([(c(p).cast(pl.Float64)-c(q).cast(pl.Float64)).alias(k) for k,(p,q) in cols.items()]).group_by("game_id").agg([c(k).sum() for k in cols]+[pl.len().alias("n")])
G=g.to_numpy(); n=g["n"].to_numpy(); rng=np.random.default_rng(0)
idx=rng.integers(0,len(g),(2000,len(g))); N=n[idx].sum(1)
for k in cols:
    v=g[k].to_numpy(); est=v.sum()/n.sum()*100; bs=v[idx].sum(1)/N*100
    print(f"{k:20s} {est:+.3f} [{np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}]")

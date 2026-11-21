"""Split v0g gains (1500, fp16) by Maia-3 difficulty: entropy and top-move prob of Maia-3's policy (no engine)."""
import numpy as np, polars as pl, json
from doppel_lab.allie_knn import games, OUT
c = pl.col
g = games().with_row_index("gi").select("gi", "game_id")
off = np.load(OUT / "off.npy"); tz = np.load(OUT / "test.npz")
pos = tz["pos"]; gi = np.searchsorted(off, pos, side="right") - 1; ply = pos - off[gi]
lp = tz["logp"].astype(np.float32); legal = tz["legal"]
p = np.where(legal >= 0, np.exp(lp), 0.0); p /= p.sum(1, keepdims=True)
ent = -(np.where(p > 0, p * np.log(np.where(p > 0, p, 1)), 0)).sum(1); top = p.max(1)
d = pl.DataFrame({"gi": gi.astype(np.uint32), "ply": ply.astype(np.int64), "ent": ent, "top": top}).join(g.with_columns(c("gi").cast(pl.UInt32)), on="gi")
r = pl.read_parquet("data/allie/v0e_positions_v0g_b.parquet").filter(c("kreq") == 40).with_columns(c("ply").cast(pl.Int64))
x = r.join(d, on=["game_id", "ply"], how="inner")
print(f"joined {x.height:,} of {r.height:,}")
x = x.with_columns(dec=(c("ent").rank("ordinal") * 5 / (pl.len() + 1)).floor().cast(pl.Int32),
                   diff=(c("top_q").cast(pl.Float64) - c("top_t").cast(pl.Float64)), dce=c("ce_t") - c("ce_q"))
rng = np.random.default_rng(0); out = {}
for q in range(5):
    s = x.filter(c("dec") == q); gg = s.group_by("game_id").agg(c("diff").sum(), pl.len().alias("n"))
    v, n = gg["diff"].to_numpy(), gg["n"].to_numpy(); idx = rng.integers(0, len(gg), (2000, len(gg)))
    bs = v[idx].sum(1) / n[idx].sum(1) * 100
    row = dict(positions=s.height, ent_range=[round(s["ent"].min(), 2), round(s["ent"].max(), 2)], maia_top1=round(s["top_t"].mean(), 3),
               gain_pts=round(v.sum() / n.sum() * 100, 3), ci95=[round(np.percentile(bs, 2.5), 3), round(np.percentile(bs, 97.5), 3)],
               nats_saved=round(s["dce"].mean(), 4))
    out[f"q{q + 1}"] = row; print(q + 1, row)
json.dump({"run": "v0g_b", "band": 1500, "split": "Maia-3 policy entropy quintile (q1 = most obvious)", "quintiles": out},
          open("lab/results/allie_difficulty.json", "w"), indent=1)

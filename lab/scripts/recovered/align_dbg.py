import numpy as np, polars as pl
from doppel_lab.combo import load_zres
from doppel_lab.retr_train import groups_for, out_dir
root=out_dir(1500)/"valrich"
sub,_=groups_for("valrich",10**9,40,8,0,band=1500)
off=np.load(root/"off.npy"); qz=dict(np.load(root/"query.npz"))
gids=sub["game_id"].to_numpy().astype(str)
gi=np.searchsorted(off,qz["pos"],side="right")-1
q=pl.DataFrame({"game_id":gids[gi],"ply":(qz["pos"]-off[gi]).astype(np.int64)})
key,_,_=load_zres("valrich",1500,smoke=True)
print(key.head(12))
for g in key["game_id"].unique().to_list():
    print(g, "in retr games:", g in set(gids), "retr plies:", sorted(q.filter(pl.col("game_id")==g)["ply"].to_list())[:12])
    print("   zres plies:", sorted(key.filter(pl.col("game_id")==g)["ply"].to_list())[:12])

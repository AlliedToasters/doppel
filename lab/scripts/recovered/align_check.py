import numpy as np, polars as pl
from doppel_lab.combo import load_zres, align
from doppel_lab.retr_train import groups_for, out_dir
band=1500
root=out_dir(band)/"valrich"
sub,_=groups_for("valrich",10**9,40,8,0,band=band)
off=np.load(root/"off.npy"); assert sub.height==len(off)-1
qz=dict(np.load(root/"query.npz")); played=np.load(root/"played.npy")
gids=sub["game_id"].to_numpy().astype(str)
gi=np.searchsorted(off,qz["pos"],side="right")-1
q=pl.DataFrame({"game_id":gids[gi],"ply":(qz["pos"]-off[gi]).astype(np.int64)}).with_row_index("qrow")
key,zl,zr=load_zres("valrich",band,smoke=True)
j=key.join(q,on=["game_id","ply"],how="left")
print("smoke rows",key.height,"found in retr",j["qrow"].null_count()==0)
qr=j["qrow"].to_numpy(); zrow=j["zrow"].to_numpy()
legal=qz["legal"][qr].astype(np.int64); logp=np.nan_to_num(qz["logp"][qr].astype(np.float32),neginf=-1e9)
truth=np.argmax(legal==played[qz["pos"][qr]].astype(np.int64)[:,None],1)
za=align(zl[zrow],zr[zrow],legal)
z=np.load("data/allie/combo/zres_valrich_1500_smoke.npz")
print("maia top1 retr",(logp.argmax(1)==truth).mean(),"v0e teacher",z["top_t"].mean())
print("doppel via align",((logp+za).argmax(1)==truth).mean(),"v0e doppel",z["top_q"].mean())
print("argmax agree with v0e doppel per row",(((logp+za).argmax(1)==truth)==z["top_q"][zrow]).mean())

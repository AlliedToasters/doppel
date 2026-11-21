import zlib, numpy as np, polars as pl
from doppel_lab.persona_pilots import positions
TAUS=[0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.75,1.01]
for b in (700,1100,1500,1900,2300):
    df=positions(b,"_ctx200")
    h=np.array([zlib.crc32(g.encode())&1 for g in df["game_id"].to_list()])
    m,o,mg=df["maia"].to_numpy(),df["own"].to_numpy(),df["margin"].to_numpy()
    def gain(t,sel): return 100*(np.where(mg[sel]<t,o[sel],m[sel]).mean()-m[sel].mean())
    res=[]
    for tr in (0,1):
        best=max(TAUS,key=lambda t:gain(t,h==tr)); res.append((best,gain(best,h!=tr)))
    curve=" ".join(f"{t}:{gain(t,np.ones_like(h,bool)):+.2f}" for t in TAUS)
    print(b,"heldout",[(t,round(g,3)) for t,g in res],"mean %+.3f"%np.mean([g for _,g in res]),"ungated %+.3f"%gain(1.01,np.ones_like(h,bool)),"|",curve,flush=True)

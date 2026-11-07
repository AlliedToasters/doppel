"""Held-out likelihood gain from a rating offset to Maia-3 (global, per time control, per time control x year).

    python -m doppel_lab.maia_offset_gain     (from lab/; Lichess bench sweep, sides rated 900-2300)

Offsets fit on even rows on a -200..+200 grid, scored on odd rows, vs the nominal rating (curves interpolated).
"""
import numpy as np, polars as pl
c=pl.col
z=np.load('../data/bench/maia3_sweep.npz',allow_pickle=True); ll=z['ll']; E=np.array(z['elos'],float)
gi={g:i for i,g in enumerate(z['game_ids'])}
t=pl.read_parquet('../data/bench/tasks.parquet').filter(c('game_id').is_in(list(gi))).with_columns(i=c('game_id').replace_strict(gi,return_dtype=pl.Int64))
R=[]
for col,k in (('white_elo',0),('black_elo',1)):
    x=t.select('i','year','tc','base','inc',elo=c(col).cast(pl.Float64)); L=ll[x['i'].to_numpy(),k]; ok=np.isfinite(L).all(1)&(x['elo'].to_numpy()>=900)&(x['elo'].to_numpy()<=2300)
    R.append((x.filter(pl.Series(ok)),L[ok]))
x=pl.concat([r[0] for r in R]); L=np.concatenate([r[1] for r in R])
def at(L,e): return np.array([np.interp(ei,E,li) for ei,li in zip(e,L)])
e=x['elo'].to_numpy(); base=at(L,e)
# fit per-(tc) offset on even rows, eval on odd rows
idx=np.arange(len(e)); tr=idx%2==0
grid=np.arange(-200,201,10)
for key in [['base','inc'],['base','inc','year']]:
    gain=np.zeros(len(e)); 
    keys=x.select(key).to_numpy() if isinstance(key,list) else x[key].to_numpy()[:,None]
    ks=[tuple(r) for r in keys]
    import collections; groups=collections.defaultdict(list)
    for j,kk in enumerate(ks): groups[kk].append(j)
    tot=0; n=0
    for kk,js in groups.items():
        js=np.array(js); a=js[tr[js]]; b=js[~tr[js]]
        if len(a)<200: continue
        sc=[at(L[a],e[a]+o).mean() for o in grid]; o=grid[int(np.argmax(sc))]
        tot+=(at(L[b],e[b]+o)-base[b]).sum(); n+=len(b)
    print(key, f"held-out gain {tot/n*1000:.2f} mnats/move over nominal rating; n={n}")
# global
sc=[at(L[tr],e[tr]+o).mean() for o in grid]; o=grid[int(np.argmax(sc))]
print('global offset',o, f"gain {(at(L[~tr],e[~tr]+o)-base[~tr]).mean()*1000:.2f} mnats")

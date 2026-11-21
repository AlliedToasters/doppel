import polars as pl, time
from doppel_lab.revisits import src_files
c=pl.col; t=time.time()
d=pl.read_parquet('../data/revisits/heldout/pairs.parquet', columns=['pid','key','key2','g1','ply1','g2','ply2','repeat','same_opp'])
ch=d.join(d, left_on=['pid','key','key2','g2','ply2'], right_on=['pid','key','key2','g1','ply1'], suffix='_c')
print('chains', ch.height)
gm=pl.read_parquet('../data/revisits/heldout/games.parquet', columns=['gi','game_id'])
ids=gm.filter(c('gi').is_in(ch['g2'].unique().implode()))
mv=(pl.scan_parquet(src_files('heldout')).select('game_id','moves').filter(c('game_id').is_in(ids['game_id'].implode())).unique('game_id').collect())
j=ids.join(mv,on='game_id').select('gi',c('moves'))
ch=ch.join(j, left_on='g2', right_on='gi').with_columns(mvB=c('moves').list.get(c('ply2'))).drop('moves')
print('unique keys', ch.select('key','key2').n_unique(), 'unique key+move', ch.select('key','key2','mvB').n_unique(), f'{time.time()-t:.0f}s')
print(ch.group_by('repeat').len())

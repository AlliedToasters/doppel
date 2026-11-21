import polars as pl, numpy as np, torch, chess
from doppel_lab.allie_v0e import test_games
from doppel_lab.v0_data import Index, targets_sample
from doppel_lab.v0e_data import V0E
from doppel_lab.top1 import positions
from doppel_lab.engines import Maia3
c = pl.col
torch.set_num_threads(16)
r = pl.read_parquet('data/allie/v0e_positions_v0g_b.parquet').filter(c('kreq') == 40)
m = pl.read_parquet('data/allie/top1_positions_maia3-79m.parquet').select('game_id', 'ply', 'hit')
x = r.join(m, on=['game_id', 'ply'])
games = x.filter(c('top_t') != c('hit'))['game_id'].unique().sort().head(80).to_list()
saved = {(g, p): (tt, h) for g, p, tt, h in x.filter(c('game_id').is_in(games)).select('game_id', 'ply', 'top_t', 'hit').iter_rows()}
t = test_games(1500).filter(c('game_id').is_in(games))
ix = Index(V0E / 'index_1500_allie1500.npz')
gpos = {str(g): i for i, g in enumerate(ix.game_id)}
maia = Maia3('79m', 'cpu')
rng = np.random.default_rng(0)
XT, XV, ET, EV, PL, LEG, KEY = [], [], [], [], [], [], []
for row in t.iter_rows(named=True):
    g = gpos[row['game_id']]
    for s, mv, (gid, ply) in positions(row, 10, True, 30):
        if (gid, ply) not in saved:
            continue
        o = targets_sample(ix, g, ply % 2, [ply], None, 1, rng)
        XT.append(maia._input(s)); XV.append(torch.from_numpy(o['tok'][0].astype(np.float32)))
        ET.append((s.elo_to_move, s.elo_opp)); EV.append((int(o['elo_self'][0]), int(o['elo_opp'][0])))
        PL.append(int(o['played'][0])); LEG.append(o['legal'][0]); KEY.append((gid, ply))
XT, XV = torch.stack(XT), torch.stack(XV)
print(len(KEY), 'positions; tok equal:', (XT == XV).all(dim=(1, 2)).float().mean().item(),
      'elo equal:', np.mean([a == b for a, b in zip(ET, EV)]))
bad = [i for i, (a, b) in enumerate(zip(ET, EV)) if a != b][:5]
print('elo mismatches e.g.', [(KEY[i], ET[i], EV[i]) for i in bad])
diffch = (XT != XV).any(dim=1).any(dim=0).nonzero().flatten().tolist(); print('differing channels', diffch[:20])

def run(X, E, dtype):
    out = []
    for i in range(0, len(X), 256):
        se = torch.tensor([e[0] for e in E[i:i+256]]); oe = torch.tensor([e[1] for e in E[i:i+256]])
        with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16, enabled=dtype == 'bf16'):
            out.append(maia.model(X[i:i+256], se.float() if dtype != 'long' else se, oe.float() if dtype != 'long' else oe)[0].float())
    L = torch.cat(out)
    hits = []
    for j in range(len(L)):
        leg = LEG[j][LEG[j] >= 0]
        hits.append(int(leg[L[j, leg].argmax()]) == PL[j])
    return np.array(hits)
tt = np.array([saved[k][0] for k in KEY]); th = np.array([saved[k][1] for k in KEY])
for name, X, E, dt in [('top1 input fp32 long-elo', XT, ET, 'long'), ('v0 input fp32 float-elo', XV, EV, 'fp32'),
                       ('v0 input bf16', XV, EV, 'bf16'), ('top1 input bf16', XT, ET, 'bf16')]:
    h = run(X, E, dt)
    print(f'{name:28s} top1 {h.mean():.4f}  agree w/ saved top1.py {np.mean(h == th):.4f}  w/ saved teacher {np.mean(h == tt):.4f}')
print('saved: top1.py', th.mean(), 'teacher', tt.mean())

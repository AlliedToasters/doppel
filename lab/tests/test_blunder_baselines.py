import numpy as np

from doppel_lab import blunder_baselines as B
from doppel_lab.features import registry


def test_feature_sets_are_pre_move_except_think():
    for name, feats in B.feature_sets().items():
        allowed = {"pre", "act"} if name == "T1+think" else {"pre"}
        assert {registry.get(f).timing for f in feats} <= allowed, name
    assert set(B.feature_sets()["T1+think"]) - set(B.feature_sets()["T1"]) == set(B.THINK)


def test_ece_and_folds():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 0.3, 200_000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    e, curve = B.ece(y, p)
    assert e < 0.005 and len(curve) == 10
    assert B.ece(y, np.clip(2 * p, 0, 1))[0] > 0.05
    gids = np.repeat(np.arange(100), 7)
    f = B.folds(gids, 5)
    assert all(len(set(f[gids == g])) == 1 for g in range(100)) and set(f) == set(range(5))

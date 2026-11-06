import numpy as np

from doppel_lab.analyzer import source
from doppel_lab.blunder import labels
from doppel_lab.features import channels, registry, table, transforms
from test_analyzer import PGN


def _game():
    rec = source.validate(source.from_pgn(PGN))
    pos, mov, _ = channels.base_channels(rec)
    channels.eval_channels(pos, mov, [20, 30, 25, 60, 500, 480, 450, -900, -950])
    return rec, pos, mov


def test_every_computed_name_is_registered():
    rec, pos, mov = _game()
    side = transforms.side_transforms(mov)
    ptf = transforms.position_transforms(pos)
    assert registry.unregistered(mov | side | transforms.position_to_moves(ptf, mov), "mov") == []
    assert registry.unregistered(pos | ptf, "pos") == []


def test_registry_labels():
    for f in registry.REGISTRY.values():
        assert f.tier in registry.TIERS and f.regime in registry.REGIMES and f.timing in registry.TIMINGS
    # Derived features inherit the channel's tier; "_prev" lags are known before the move.
    assert registry.get("cpl_ewma").tier == "T2" and registry.get("cpl_ewma").timing == "post"
    assert registry.get("cpl_ewma_prev").timing == "pre"
    assert registry.get("wc_w_volatility").tier == "T2"
    pre_t0 = registry.select(tier="T0", regime="R0")
    assert "material_own" in pre_t0 and "think_s" not in pre_t0 and "maia_entropy" not in pre_t0
    assert not set(registry.select(tier="T3", regime="R1")) & {"cpl", "wc_loss", "blunder", "cpl_ewma"}


def test_prev_is_lagged_by_own_move():
    _, _, mov = _game()
    side = transforms.side_transforms(mov)
    white = np.where(mov["mover"] == 0)[0]
    assert np.isnan(side["cpl_ewma_prev"][white[0]])
    assert np.allclose(side["cpl_ewma_prev"][white[1:]], side["cpl_ewma"][white[:-1]])


def test_blunder_labels_match_channels():
    _, _, mov = _game()
    lab = labels([20, 30, 25, 60, 500, 480, 450, -900, -950])
    assert np.allclose(lab["wc_loss"], mov["wc_loss"]) and np.allclose(lab["cpl"], mov["cpl"])


def test_game_rows_from_stored_shapes():
    rec, _, _ = _game()
    n = len(rec["moves"])
    rng = np.random.default_rng(0)
    top_lp = np.sort(np.log(rng.dirichlet(np.ones(16), n)), 1)[:, ::-1].astype(np.float16)
    top_idx = np.tile(np.arange(16, dtype=np.int16), (n, 1))
    played = np.array([0, 3, 20, 1, 0, 0, 2, 5], np.int16)
    t1 = (top_lp, played, top_idx, top_lp[np.arange(n), np.minimum(played, 15)], np.ones(n, np.float16))
    row = {"game_id": "abcdefgh", "white_elo": 1500, "black_elo": 1520, "base": 180, "inc": 2,
           "moves": rec["moves"], "clocks": rec["clocks"]}
    out = table.game_rows((row, t1, [20, 30, 25, 60, 500, 480, 450, -900, -950]))
    assert all(len(v) == n for v in out.values())
    assert list(out["maia_played_rank"]) == [1, 4, 17, 2, 1, 1, 3, 6]
    assert out["elo_self"][1] == 1520 and out["blunder"].sum() == 2
    assert registry.unregistered(out, "mov") == []

import chess
import numpy as np
import torch

from doppel_lab import v0_gate as G
from doppel_lab.v0_model import DoppelV0, FakeTrunk

EV = {"active_units": 5, "kl_nats_per_half": 4.5, "ce_post": 1.30, "ce_swap": 1.34, "z_swap_gain_nats": 0.04,
      "z_swap_gain_ci": [0.02, 0.06], "top1_prior": 0.528, "top1_post": 0.54, "top1_teacher": 0.530,
      "think_r_log_prior": 0.60, "think_r_log_post": 0.63}


def test_eval_checks_pass_and_fail():
    c = G.eval_checks(EV, free_bits=3.0)
    assert c["z_used"]["pass"] and c["accuracy"]["pass"] and c["think"]["pass"]
    bad = EV | {"active_units": 3, "z_swap_gain_ci": [-0.01, 0.03], "top1_prior": 0.52, "think_r_log_prior": 0.5}
    c = G.eval_checks(bad, free_bits=3.0)
    assert not (c["z_used"]["pass"] or c["accuracy"]["pass"] or c["think"]["pass"])
    assert c["z_used"]["parts"] == {"active_units": False, "kl": True, "z_swap": False}
    old = {k: v for k, v in EV.items() if not k.startswith(("z_swap", "ce_swap"))}
    assert not G.eval_checks(old, 3.0)["z_used"]["pass"]
    assert G.eval_checks(None, 3.0)["z_used"]["status"] == "pending"


def test_doppel_sampler_plays_legal_moves():
    torch.manual_seed(0)
    from doppel_model import board_planes
    eng = G.Doppel(DoppelV0(FakeTrunk()), "cpu")
    gen = torch.Generator().manual_seed(0)
    z = eng.sample_z(np.array([G.cond_vec(1550, 1550, 180, 2)] * 2, np.float32), gen)
    assert z.shape == (2, 8)
    boards = [chess.Board(), chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")]
    boards[1].push_san("Bc4")  # black to move: exercises the mirrored move frame
    hist = [[board_planes(b)] for b in boards]
    for u, b in zip(eng.sample(boards, hist, z, [1550, 1550], [1550, 1550], gen), boards):
        assert chess.Move.from_uci(u) in b.legal_moves

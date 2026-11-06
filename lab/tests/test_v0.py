"""Doppel v0 on CPU with a fake trunk: sample building, losses, gradient routing, think likelihood."""

import math

import numpy as np
import pytest
import torch

from doppel_lab.v0_data import V0, Index, Samples, build_sample, collate, think_seconds
from doppel_lab.v0_model import DoppelV0, FakeTrunk, ThinkHead, losses

VAL = V0 / "index_1500_val.npz"
needs_index = pytest.mark.skipif(not VAL.exists(), reason="run `python -m doppel_lab.v0_data prep --split val`")


def test_think_seconds_uses_mover_clock_two_plies_back():
    clock = np.array([300, 300, 295, 290, 280], np.float32)
    assert math.isnan(think_seconds(clock, 1, 300, 2)[0])  # first moves are free on Lichess
    assert think_seconds(clock, 2, 300, 2) == (7.0, 300.0)  # 300 - 295 + 2
    assert think_seconds(clock, 4, 300, 2) == (17.0, 295.0)


def test_censored_think_likelihood_is_a_probability():
    h = ThinkHead(8)
    mu, ls = torch.tensor([-3.0, -1.0]), torch.tensor([0.0, 0.5])
    nll = h.nll(mu, ls, torch.tensor([4.0, 0.0]), torch.tensor([100.0, 50.0]), torch.tensor([0.0, 2.0]))
    assert (nll > 0).all() and torch.isfinite(nll).all()


@needs_index
def test_samples_targets_are_own_moves_in_the_other_half():
    ix = Index(VAL)
    rng = np.random.default_rng(0)
    for g in range(20):
        n = ix.plies(g)
        for color in (0, 1):
            s = build_sample(ix, g, color, True, 8, rng)
            if s is None:
                continue
            assert all(k % 2 == color and k >= n // 2 for k in s["ply"])
            if s["resign"].any():
                assert s["ply"][-1] == n and s["played"][-1] == -1 and s["weight"][-1] == 1.0


@needs_index
def test_step0_is_teacher_and_heads_learn():
    torch.manual_seed(0)
    b = collate([Samples(Index(VAL))[i] for i in range(24)])
    m = DoppelV0(FakeTrunk())
    _, p = losses(m, b, beta=1.0)
    assert abs(p["ce"] - p["ce_teacher"]) < 1e-4  # residual gain starts at 0
    opt = torch.optim.AdamW(m.heads_parameters(), 1e-3)
    first = None
    for _ in range(15):
        opt.zero_grad()
        loss, p = losses(m, b, beta=1.0)
        first = first or p["loss"]
        loss.backward()
        opt.step()
    assert p["loss"] < first
    assert all(not q.requires_grad for q in m.trunk.parameters())  # the teacher never trains

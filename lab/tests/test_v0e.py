"""Doppel v0e on CPU with a fake trunk: context sets (R2, no leakage), set encoder, v0d init, gradient routing."""

import numpy as np
import pytest
import torch

from doppel_lab.v0_model import DoppelV0, FakeTrunk, losses
from doppel_lab.v0e_data import K_MAX, V0E, EvalSamples, Split, TrainSamples, collate, draw_k
from doppel_lab.v0e_model import DoppelV0e, SetEncoder, init_from_v0d

VALRICH = V0E / "meta_1500_valrich.npz"
needs_valrich = pytest.mark.skipif(not VALRICH.exists(), reason="run `python -m doppel_lab.v0e_data prep`")


def toy_split(t, pid):
    sp = Split.__new__(Split)
    sp.t, sp.pid = np.asarray(t), np.asarray(pid)
    sp._order()
    return sp


def test_history_is_the_players_earlier_games_only():
    # games 0..4; player 7 plays White in 0, 2 and Black in 3, 4; time order is 3, 0, 4, 2, 1
    t = [10, 50, 40, 0, 20]
    pid = [[7, 1], [2, 3], [7, 4], [5, 7], [6, 7]]
    sp = toy_split(t, pid)
    g, c = sp.history(2, 0)                       # game 2 at t=40: player 7's earlier games are 3, 0, 4
    assert g.tolist() == [3, 0, 4] and c.tolist() == [1, 0, 1]
    assert len(sp.history(3, 1)[0]) == 0          # first game: no history
    assert sp.history(1, 0)[0].tolist() == []     # player 2 has one game
    for gg in range(5):
        for cc in (0, 1):
            hg, hc = sp.history(gg, cc)
            assert gg not in hg.tolist()
            assert all(sp.t[h] < sp.t[gg] for h in hg)
            assert all(sp.pid[h, x] == sp.pid[gg, cc] for h, x in zip(hg, hc))


def test_k_draw_is_capped_and_skewed_small():
    rng = np.random.default_rng(0)
    assert all(draw_k(0, rng) == 0 for _ in range(50))
    ks = np.array([draw_k(100, rng) for _ in range(20_000)])
    assert ks.max() <= K_MAX and ks.min() >= 0
    assert 0.02 < (ks == 0).mean() < 0.08
    assert (ks <= 5).mean() > 0.4 > (ks >= 20).mean()
    assert all(draw_k(3, rng) <= 3 for _ in range(200))


def test_set_encoder_is_permutation_invariant_and_ignores_padding():
    torch.manual_seed(0)
    enc = SetEncoder()
    torch.nn.init.normal_(enc.out.weight)
    B, L = 3, 6
    emb, feat = torch.randn(B, L, 1024), torch.randn(B, L, enc.feat_mu.numel())
    mask = torch.ones(B, L, dtype=torch.bool)
    mask[0, 4:] = False
    cond = torch.randn(B, 4)
    mu, lv = enc(emb, feat, mask, cond)
    perm = torch.tensor([3, 1, 0, 2, 4, 5])        # permute the valid items of row 0
    emb2, feat2 = emb.clone(), feat.clone()
    emb2[0], feat2[0] = emb[0, perm], feat[0, perm]
    emb2[0, 4:], feat2[0, 4:] = 99.0, -99.0        # padding content must not matter
    mu2, lv2 = enc(emb2, feat2, mask, cond)
    assert torch.allclose(mu[0], mu2[0], atol=1e-5) and torch.allclose(lv[0], lv2[0], atol=1e-5)


def test_k0_posterior_is_the_prior():
    torch.manual_seed(0)
    m = DoppelV0e(FakeTrunk())
    torch.nn.init.normal_(m.prior.net[-1].weight)
    torch.nn.init.normal_(m.encoder.out.weight)
    B = 4
    b = {"ctx_emb": torch.randn(B, 2, 1024), "ctx_feat": torch.randn(B, 2, m.encoder.feat_mu.numel()),
         "ctx_mask": torch.tensor([[False, False], [True, False], [True, True], [False, False]]),
         "cond": torch.randn(B, 4), "nctx": torch.tensor([0, 1, 2, 0])}
    mu_q, lv_q = m.posterior(b)
    mu_p, lv_p = m.prior(b["cond"])
    assert torch.equal(mu_q[[0, 3]], mu_p[[0, 3]]) and torch.equal(lv_q[[0, 3]], lv_p[[0, 3]])
    assert not torch.allclose(mu_q[1], mu_p[1])


def test_init_from_v0d_is_v0d_with_z_ignored(tmp_path):
    torch.manual_seed(0)
    trunk = FakeTrunk()
    old = DoppelV0(trunk)
    for p in old.parameters():
        if p.requires_grad:
            torch.nn.init.normal_(p, std=0.05)
    path = tmp_path / "ckpt.pt"
    torch.save({"model": {k: v for k, v in old.state_dict().items() if not k.startswith("trunk.")}}, path)
    new = DoppelV0e(trunk)
    rep = init_from_v0d(new, path)
    assert set(rep["partial"]) == {"residual.ctx.weight", "think.net.0.weight", "resign.net.0.weight"}
    assert not any(k.startswith(("prior.", "encoder.")) for k in rep["copied"])
    from doppel_lab.v0_model import NMOVES
    B = 3
    tok = torch.randint(0, 2, (B, 64, 97)).float()
    emb_o, emb_n = old.residual.board(tok), new.residual.board(tok)
    mask = torch.zeros(B, NMOVES, dtype=torch.bool)
    mask[:, :40] = True
    es, eo = torch.full((B,), 1450), torch.full((B,), 1620)
    r_old = old.residual(emb_o, torch.zeros(B, 8), es, eo, mask)
    r_new = new.residual(emb_n, torch.randn(B, new.z_dim), es, eo, mask)
    assert torch.allclose(r_old, r_new, atol=1e-5)
    _, value, pooled = trunk(tok, es, eo)
    clk, tempo = torch.randn(B, 7), torch.randn(B, 6)
    assert torch.allclose(old.think(pooled, clk, tempo, torch.zeros(B, 8))[0],
                          new.think(pooled, clk, tempo, torch.randn(B, new.z_dim))[0], atol=1e-5)


def fake_split():
    from doppel_lab.v0e_train import FakeVecs, load_split
    return load_split("valrich", fake_vecs=True)


@needs_valrich
def test_samples_context_precedes_target_and_targets_are_own_moves():
    sp = fake_split()
    ds = TrainSamples(sp, K=8)
    rng = np.random.default_rng(0)
    for i in rng.choice(len(ds), 60, replace=False):
        s = ds[int(i)]
        if s is None:
            continue
        g, col = int(i) // 2, int(i) % 2
        assert all(p % 2 == col for p in s["ply"]) and (s["game"] == g).all()
        k = int(s["nctx"][0])
        hg, _ = sp.history(g, col)
        assert k <= len(hg) and len(s["ctx_feat"]) == k
        if k:
            np.testing.assert_array_equal(s["ctx_feat"], sp.feat[hg[-k:], sp.history(g, col)[1][-k:]])


@needs_valrich
def test_eval_grid_is_paired_across_k_and_losses_route_gradients():
    sp = fake_split()
    ev = EvalSamples(sp, ks=(0, 5, 40), max_players=2)
    rows = [ev[i] for i in range(6)]
    assert [int(r["nctx"][0]) for r in rows[:3]] == [0, 5, 40]
    assert all(np.array_equal(rows[0]["ply"], r["ply"]) for r in rows[:3])   # same targets at every k
    torch.manual_seed(0)
    b = collate(rows)
    m = DoppelV0e(FakeTrunk())
    loss, p = losses(m, b, beta=0.25, free_bits=0.5)
    assert abs(p["ce"] - p["ce_teacher"]) < 1e-4          # residual starts at 0
    assert float(p["_kl"][b["nctx"] == 0].abs().max()) == 0.0
    loss.backward()
    assert all(q.grad is None for q in m.trunk.parameters())
    # Gradient rule: with think / resign losses off, nothing reaches the think head.
    m.zero_grad()
    loss, _ = losses(m, b, beta=0.25, free_bits=0.5, w_think=0.0, w_resign=0.0)
    loss.backward()
    assert all(q.grad is None or q.grad.abs().max() == 0 for q in m.think.parameters())


def test_contrastive_views_are_disjoint_and_train_only_the_encoder():
    from doppel_lab.v0e_model import contrastive
    from doppel_lab.v0e_data import CTX_FEATS, EMB_DIM
    torch.manual_seed(0)
    m = DoppelV0e(FakeTrunk())
    B, L = 6, 5
    mask = torch.zeros(B, L, dtype=torch.bool)
    for i, k in enumerate([5, 4, 3, 2, 1, 0]):
        mask[i, :k] = True
    b = {"ctx_emb": torch.randn(B, L, EMB_DIM), "ctx_feat": torch.randn(B, L, CTX_FEATS), "ctx_mask": mask,
         "cond": torch.randn(B, 4), "pid": torch.tensor([0, 1, 2, 0, 3, 4]), "nctx": mask.sum(1)}
    loss, acc, n = contrastive(m, b)
    assert n == 4 and torch.isfinite(loss) and 0 <= acc <= 1
    loss.backward()
    grads = {k for k, p in m.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    assert grads and all(k.startswith("encoder.") for k in grads)

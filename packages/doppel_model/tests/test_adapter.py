"""Format-2 packages: adapter inputs are validated and zeros stand in for the base model (no ONNX needed)."""

import numpy as np
import pytest

from doppel_model import ModelPackage

INPUTS = [{"name": "lora_b4_linear1_A", "shape": [2, 3]}, {"name": "lora_b4_linear1_B", "shape": [4, 2]}]


class FakeSession:
    def __init__(self):
        self.feeds = []

    def run(self, _out, feed):
        self.feeds.append(feed)
        bias = float(feed.get("lora_b4_linear1_A", np.zeros(1)).sum())
        return [np.full((1, 4352), 0.0, np.float32) + bias * np.arange(4352, dtype=np.float32)[None] / 4352]


def pkg(lora=True):
    m = {"input": {"history": 8}, "lora": {"inputs": INPUTS, "residual_weight": 0.5}} if lora else {"input": {"history": 8}}
    return ModelPackage("t-00000000", m, FakeSession())


def test_zero_adapter_when_none():
    p = pkg()
    p.policy([], 1500, 1500)
    f = p.session.feeds[0]
    assert f["lora_b4_linear1_A"].shape == (2, 3) and not f["lora_b4_linear1_A"].any()


def test_rejects_bad_shape_or_nonfinite():
    p = pkg()
    with pytest.raises(ValueError):
        p.policy([], 1500, 1500, adapter=[np.zeros((3, 2)), np.zeros((4, 2))])
    with pytest.raises(ValueError):
        p.policy([], 1500, 1500, adapter=[np.full((2, 3), np.nan), np.zeros((4, 2))])
    with pytest.raises(ValueError):
        pkg(lora=False).policy([], 1500, 1500, adapter=[np.zeros((2, 3))])


def test_doppel_blend_is_weighted_between_base_and_adapter():
    p = pkg()
    ab = [np.ones((2, 3)), np.zeros((4, 2))]
    _, _, base = p.policy([], 1500, 1500)
    _, _, own = p.policy([], 1500, 1500, adapter=ab)
    _, _, dop = p.doppel_policy([], 1500, 1500, ab)
    want = base + 0.5 * (own - base)
    want -= np.log(np.exp(want - want.max()).sum()) + want.max()
    assert np.allclose(dop, want) and np.isclose(np.exp(dop).sum(), 1)

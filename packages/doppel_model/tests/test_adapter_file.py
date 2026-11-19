"""Adapter files (doppel_model.adapter): round trip, content id, and strict loading."""

import io
import json

import numpy as np
import pytest

from doppel_model.adapter import adapter_id, load_adapter, save_adapter

SPEC = {"inputs": [{"name": "a", "shape": [2, 3]}, {"name": "b", "shape": [4, 2]}]}


def arrays(seed=0):
    r = np.random.default_rng(seed)
    return [r.standard_normal((2, 3)), r.standard_normal((4, 2))]


def test_round_trip_and_stable_id(tmp_path):
    aid = save_adapter(tmp_path, SPEC, arrays(), {"package": "p"})
    assert aid == save_adapter(tmp_path / "x", SPEC, arrays(), {"package": "p", "made": "other day"})
    arrs, meta = load_adapter(tmp_path, aid, SPEC)
    assert meta["package"] == "p" and np.allclose(arrs[0], arrays()[0], atol=1e-3)


def test_rejects_paths_and_tampering(tmp_path):
    aid = save_adapter(tmp_path, SPEC, arrays(), {"package": "p"})
    with pytest.raises(ValueError):
        load_adapter(tmp_path, "../" + aid[:13], SPEC)
    other = arrays(1)
    buf = io.BytesIO()
    np.savez(buf, meta=np.array(json.dumps({"package": "p"})), a=other[0].astype(np.float16), b=other[1].astype(np.float16))
    (tmp_path / f"{aid}.npz").write_bytes(buf.getvalue())   # right name, wrong weights
    with pytest.raises(ValueError, match="content hash"):
        load_adapter(tmp_path, aid, SPEC)


def test_rejects_wrong_shape_dtype_nonfinite_and_pickle(tmp_path):
    with pytest.raises(ValueError):
        save_adapter(tmp_path, SPEC, [np.zeros((3, 2)), np.zeros((4, 2))], {})
    with pytest.raises(ValueError):
        save_adapter(tmp_path, SPEC, [np.full((2, 3), np.inf), np.zeros((4, 2))], {})
    bad = [np.zeros((2, 3), np.float32), np.zeros((4, 2), np.float32)]   # float32 on disk is refused
    aid = adapter_id(bad)
    buf = io.BytesIO()
    np.savez(buf, meta=np.array("{}"), a=bad[0], b=bad[1])
    (tmp_path / f"{aid}.npz").write_bytes(buf.getvalue())
    with pytest.raises(ValueError):
        load_adapter(tmp_path, aid, SPEC)
    buf = io.BytesIO()
    np.savez(buf, meta=np.array([{"x": 1}], dtype=object), a=bad[0].astype(np.float16), b=bad[1].astype(np.float16))
    (tmp_path / f"{aid}.npz").write_bytes(buf.getvalue())
    with pytest.raises(ValueError):                                        # object array needs pickle: refused
        load_adapter(tmp_path, aid, SPEC)

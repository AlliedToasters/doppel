"""infer serves doppels: a format-2 package + an adapter id. Needs a local format-2 package and lab adapters
(models/maia3-79m-lora8-*, data/allie/adapters_1500_ctx200_fit5_s200); skipped where they are absent."""

import importlib
import shutil
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[3]
LORA = sorted((ROOT / "models").glob("maia3-79m-lora8-*"))
BASE = sorted((ROOT / "models").glob("maia3-79m-[0-9a-f]*"))
LAB = ROOT / "data/allie/adapters_1500_ctx200_fit5_s200"
pytestmark = pytest.mark.skipif(not (LORA and BASE and LAB.exists()), reason="needs local packages and adapters")


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    from doppel_model.adapter import save_adapter
    import json
    t = tmp_path_factory.mktemp("infer")
    (t / "models").mkdir()
    for d in (BASE[-1], LORA[-1]):
        (t / "models" / d.name).symlink_to(d)
    man = json.loads((LORA[-1] / "manifest.json").read_text())
    with np.load(sorted(LAB.glob("chunk_*.npz"))[0]) as z:
        keys = sorted((k for k in z.files if k[0] == "p" and k[1:].isdigit()), key=lambda k: int(k[1:]))
        rows = [[z[k][i] for k in keys] for i in (0, 1)]
    ids = [save_adapter(t / "adapters", man["lora"], r, {"package": man["name"]}) for r in rows]
    wrong = save_adapter(t / "adapters", man["lora"], [a * 0.5 for a in rows[0]], {"package": "maia3-23m-lora8"})
    return t, man, ids, wrong


@pytest.fixture(scope="module")
def client(env):
    t, *_ = env
    mp = pytest.MonkeyPatch()
    mp.setenv("DOPPEL_MODELS", str(t / "models"))
    mp.setenv("DOPPEL_ADAPTERS", str(t / "adapters"))
    mp.setenv("DOPPEL_ADAPTER_CACHE", "1")
    import doppel_infer.app as m
    m = importlib.reload(m)
    with TestClient(m.app) as c:
        c.mod = m
        yield c
    mp.undo()


def mv(client, **kw):
    return client.post("/api/move", json={"elo": 1500, "moves": ["e2e4", "e7e5"], "temperature": 0} | kw)


def test_bots_list_unchanged(client, env):
    ids = [b["id"] for b in client.get("/api/bots").json()["bots"]]
    assert ids == ["maia3-79m"]
    assert client.get("/api/health").json()["doppel_bases"] == {"maia3-79m-lora8": LORA[-1].name}


def test_doppel_move_matches_package_blend(client, env):
    t, man, ids, _ = env
    r = mv(client, bot=man["name"], adapter=ids[0])
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["adapter"] == ids[0] and d["package"] == LORA[-1].name
    pkg = client.mod.DOPPEL_BASES[man["name"]]
    from doppel_model.adapter import load_adapter
    ab, _ = load_adapter(t / "adapters", ids[0], man["lora"])
    _, legal, lp = pkg.doppel_policy(["e2e4", "e7e5"], 1500, 1500, ab)
    assert d["move"] == legal[int(np.argmax(lp))]
    assert abs(d["top"][0]["p"] - float(np.exp(lp.max()))) < 1e-3


def test_two_players_differ_and_cache_is_bounded(client, env):
    _, man, ids, _ = env
    a = mv(client, bot=man["name"], adapter=ids[0]).json()["top"]
    b = mv(client, bot=man["name"], adapter=ids[1]).json()["top"]
    assert a != b
    assert client.get("/api/health").json()["adapters_cached"] == 1   # DOPPEL_ADAPTER_CACHE=1


def test_refusals(client, env):
    _, man, ids, wrong = env
    assert mv(client, bot=man["name"], adapter="0" * 16).status_code == 404
    assert mv(client, bot=man["name"], adapter="../../etc/passwd").status_code == 422   # pattern
    assert mv(client, bot=man["name"], adapter=wrong).status_code == 422                # made for another package
    assert mv(client, bot="maia3-79m", adapter=ids[0]).status_code == 400               # not a doppel base
    assert mv(client, bot=man["name"]).status_code == 400                               # base alone is not a bot
    assert mv(client, bot="maia3-79m").status_code == 200

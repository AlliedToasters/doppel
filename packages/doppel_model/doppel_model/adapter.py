"""Per-player adapter files: what a doppel is, beside its format-2 model package.

    <dir>/<id>.npz   one array per LoRA input of the package (manifest lora.inputs names), float16
                     + "meta": JSON string {package, rank, blocks, layers, made, source}
    id               sha256[:16] over the float16 arrays' bytes in manifest order (not the zip: zip entries
                     carry timestamps), so one set of weights has one id wherever it is written.

Loading is strict: the id must be 16 hex chars (no paths), arrays are read without pickle, names and shapes must
match the package, every value must be finite, and the recomputed hash must equal the id.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np

ID = re.compile(r"^[0-9a-f]{16}$")
MAX_BYTES = 1 << 20  # a rank-8 79M adapter is ~390 KB


def adapter_id(arrays: list[np.ndarray]) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a, np.float16).tobytes())
    return h.hexdigest()[:16]


def save_adapter(dirp: Path, spec: dict, arrays: list[np.ndarray], meta: dict) -> str:
    """Write one player's adapter for a format-2 package's lora spec; returns its id."""
    names = [i["name"] for i in spec["inputs"]]
    if len(arrays) != len(names):
        raise ValueError(f"{len(arrays)} arrays, spec has {len(names)}")
    arrs = [np.asarray(a, np.float16) for a in arrays]
    for i, a in zip(spec["inputs"], arrs):
        if list(a.shape) != i["shape"] or not np.isfinite(a).all():
            raise ValueError(f"{i['name']}: shape {list(a.shape)}, want {i['shape']}, or non-finite")
    aid = adapter_id(arrs)
    dirp.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    np.savez(buf, meta=np.array(json.dumps(meta)), **dict(zip(names, arrs)))
    tmp = dirp / f".{aid}.npz.tmp"
    tmp.write_bytes(buf.getvalue())
    tmp.replace(dirp / f"{aid}.npz")
    return aid


def read_adapter(data: bytes, spec: dict) -> tuple[str, list[np.ndarray], dict]:
    """Parse and check adapter bytes against a lora spec; returns (id, float32 arrays in order, meta)."""
    if len(data) > MAX_BYTES:
        raise ValueError(f"adapter is {len(data)} bytes, limit {MAX_BYTES}")
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        names = [i["name"] for i in spec["inputs"]]
        if set(z.files) != set(names) | {"meta"}:
            raise ValueError("adapter arrays do not match the package's lora inputs")
        arrs = [z[n] for n in names]
        meta = json.loads(str(z["meta"]))
    for i, a in zip(spec["inputs"], arrs):
        if a.dtype != np.float16 or list(a.shape) != i["shape"] or not np.isfinite(a).all():
            raise ValueError(f"{i['name']}: {a.dtype} {list(a.shape)}, want float16 {i['shape']}, finite")
    return adapter_id(arrs), [a.astype(np.float32) for a in arrs], meta


def load_adapter(dirp: Path, aid: str, spec: dict) -> tuple[list[np.ndarray], dict]:
    if not ID.match(aid):
        raise ValueError("adapter id must be 16 hex characters")
    got, arrs, meta = read_adapter((dirp / f"{aid}.npz").read_bytes(), spec)
    if got != aid:
        raise ValueError(f"adapter {aid}: content hash is {got}")
    return arrs, meta

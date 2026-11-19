"""Model package loader (contracts/model_package.schema.json): <name>-<sha8>/{model.onnx, manifest.json, LICENSE, NOTICE}."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import chess
import numpy as np

from .encoding import board_planes, encode, legal_mask


@dataclass
class ModelPackage:
    id: str
    manifest: dict
    session: object  # onnxruntime.InferenceSession

    @classmethod
    def load(cls, path: Path, threads: int = 4, verify: bool = True) -> "ModelPackage":
        import onnxruntime as ort
        manifest = json.loads((path / "manifest.json").read_text())
        onnx_path = path / "model.onnx"
        if verify:
            h = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
            if h != manifest["sha256"]:
                raise ValueError(f"{path.name}: sha256 mismatch")
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
        return cls(manifest["id"], manifest, sess)

    def adapter_feed(self, adapter: list[np.ndarray] | None) -> dict[str, np.ndarray]:
        """Format-2 packages: the LoRA inputs, zeros (= the base model) when adapter is None."""
        spec = self.manifest.get("lora")
        if spec is None:
            if adapter is not None:
                raise ValueError(f"{self.id} takes no adapter")
            return {}
        if adapter is None:
            return {i["name"]: np.zeros(i["shape"], np.float32) for i in spec["inputs"]}
        if len(adapter) != len(spec["inputs"]):
            raise ValueError(f"adapter has {len(adapter)} arrays, {self.id} needs {len(spec['inputs'])}")
        feed = {}
        for i, w in zip(spec["inputs"], adapter):
            w = np.asarray(w, np.float32)
            if list(w.shape) != i["shape"] or not np.isfinite(w).all():
                raise ValueError(f"adapter {i['name']}: shape {list(w.shape)}, want {i['shape']}, or non-finite")
            feed[i["name"]] = w
        return feed

    def doppel_policy(self, moves: list[str], elo_self: int, elo_opp: int, adapter: list[np.ndarray],
                      weight: float | None = None) -> tuple[chess.Board, list[str], np.ndarray]:
        """A player's doppel: base + weight * (adapted - base) over legal log-probs, renormalised (two runs).
        weight defaults to the manifest's residual_weight."""
        w = self.manifest["lora"]["residual_weight"] if weight is None else weight
        board, legal, base = self.policy(moves, elo_self, elo_opp)
        _, _, own = self.policy(moves, elo_self, elo_opp, adapter=adapter)
        z = base + w * (own - base)
        z -= z.max()
        return board, legal, z - np.log(np.exp(z).sum())

    def policy(self, moves: list[str], elo_self: int, elo_opp: int,
               adapter: list[np.ndarray] | None = None) -> tuple[chess.Board, list[str], np.ndarray]:
        """Replay `moves` from the start; return the board, its legal moves and their log-probabilities.
        adapter: a format-2 package's LoRA arrays in manifest order (None = the base model)."""
        board = chess.Board()
        planes = [board_planes(board)]
        for u in moves:
            board.push_uci(u)
            planes.append(board_planes(board))
        x = encode(planes, self.manifest["input"]["history"])[None]
        logits = self.session.run(["logits_move"], {
            "tokens": x,
            "elo_self": np.array([elo_self], np.int64),
            "elo_opp": np.array([elo_opp], np.int64),
            **self.adapter_feed(adapter),
        })[0][0]
        mask, legal = legal_mask(board)
        z = logits[mask].astype(np.float64)
        z -= z.max()
        return board, legal, z - np.log(np.exp(z).sum())


def load_packages(root: Path, **kw) -> dict[str, ModelPackage]:
    return {p.name: ModelPackage.load(p, **kw) for p in sorted(root.iterdir()) if (p / "manifest.json").exists()}

"""Model package loader (contracts/model_package.schema.json): <name>-<sha8>/{model.onnx, manifest.json}."""

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

    def policy(self, moves: list[str], elo_self: int, elo_opp: int) -> tuple[chess.Board, list[str], np.ndarray]:
        """Replay `moves` from the start; return the board, its legal moves and their log-probabilities."""
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
        })[0][0]
        mask, legal = legal_mask(board)
        z = logits[mask].astype(np.float64)
        z -= z.max()
        return board, legal, z - np.log(np.exp(z).sum())


def load_packages(root: Path, **kw) -> dict[str, ModelPackage]:
    return {p.name: ModelPackage.load(p, **kw) for p in sorted(root.iterdir()) if (p / "manifest.json").exists()}

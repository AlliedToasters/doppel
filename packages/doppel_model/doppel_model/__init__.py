"""Shared encoding for Doppel model packages. numpy + python-chess only: no torch, so the serving image stays small.

The encoding reproduces Maia-3's (vendor/maia3/maia3/dataset.py) exactly; the parity test in
lab/doppel_lab/export_package.py checks it against the torch model end to end.
"""

from .encoding import MOVES, MOVE_INDEX, encode, legal_mask, mirror_uci, board_planes
from .package import ModelPackage, load_packages

__all__ = ["MOVES", "MOVE_INDEX", "encode", "legal_mask", "mirror_uci", "board_planes", "ModelPackage", "load_packages"]

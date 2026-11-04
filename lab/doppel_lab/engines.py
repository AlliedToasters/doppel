"""Batched move samplers for the two baselines.

Both expose `sample(states) -> list[(uci_move, think_seconds | None)]` over a list of GameState.
ChessMimic follows its backend's /move flow: per-band common-moves lookup first, else the move model;
think time always comes from the clock model. Maia-3 uses real 8-ply history (the UCI wrapper's
default replicates the current board instead) and has no clock.
"""

from __future__ import annotations

import gzip
import json
import math
import pickle
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import chess
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]  # repo root
CM_DIR = ROOT / "vendor/chessmimic"
CM_MODELS = CM_DIR / "backend/models"


@dataclass
class GameState:
    board: chess.Board
    elo_white: int
    elo_black: int
    base: float = 180.0
    inc: float = 0.0
    clock_white: float = 180.0
    clock_black: float = 180.0
    moves: list = field(default_factory=list)
    maia_hist: list = field(default_factory=list)  # Maia-3 board tokens, oldest first

    @property
    def elo_to_move(self):
        return self.elo_white if self.board.turn == chess.WHITE else self.elo_black

    @property
    def elo_opp(self):
        return self.elo_black if self.board.turn == chess.WHITE else self.elo_white

    @property
    def clocks(self):  # (to-move, opponent)
        if self.board.turn == chess.WHITE:
            return self.clock_white, self.clock_black
        return self.clock_black, self.clock_white


# ---------------------------------------------------------------- ChessMimic

def _install_chessmimic_core_stub():
    """ChessMimic's trainers import a C++ module; its Python twin (tokenizer.py) has the same tables."""
    if "chessmimic_core" in sys.modules:
        return
    sys.path.insert(0, str(CM_DIR / "Training"))
    import tokenizer as tk  # vendor/chessmimic/Training/tokenizer.py

    m2a, a2m = tk._compute_all_possible_actions()
    mod = types.ModuleType("chessmimic_core")
    mod.tokenize = tk.tokenize
    mod.MOVE_TO_ACTION, mod.ACTION_TO_MOVE, mod.NUM_ACTIONS = m2a, a2m, len(m2a)
    mod.INPUT_VOCAB_SIZE, mod.SEQUENCE_LENGTH = tk.INPUT_VOCAB_SIZE, tk.SEQUENCE_LENGTH
    mod.CLASS_TOKEN, mod.PAD_TOKEN = tk.CLASS_TOKEN, tk.PAD_TOKEN
    sys.modules["chessmimic_core"] = mod
    # Trainer modules import their dataset readers at module level; inference never uses them.
    bagz = types.ModuleType("bagz")
    bagz.BagReader = bagz.BagFileReader = object
    sys.modules.setdefault("bagz", bagz)


def _load_state(model, path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck.get("state_dict", ck)
    sd = {k.removeprefix("model.").removeprefix("_orig_mod."): v for k, v in sd.items()}
    model.load_state_dict(sd)
    return model.to(device).eval()


class ChessMimic:
    def __init__(self, band_lo: int, device: str = "cpu"):
        _install_chessmimic_core_stub()
        import chessmimic_core as cc
        from ClockTrainer import ClockPatzerModel
        from MoveTrainer import PatzerModel
        from clock_bucket_utils import load_bucket_info

        self.cc, self.device = cc, torch.device(device)
        tag = f"{band_lo}_{band_lo + 100}_brier"
        mdir, cdir = CM_MODELS / "move_model" / tag, CM_MODELS / "clock_model" / tag

        s = pickle.load(open(mdir / "scalers.pkl", "rb"))
        self.m_rating = (float(s["rating"]["mean"]), float(s["rating"]["std"]))
        self.m_time = (float(s["log_time"]["mean"]), float(s["log_time"]["std"]))
        common = dict(recent_moves_sequence_length=12, recent_moves_vocab_size=cc.NUM_ACTIONS,
                      board_sequence_length=cc.SEQUENCE_LENGTH, board_input_vocab_size=cc.INPUT_VOCAB_SIZE,
                      embedding_dim=256, widening_factor=4, num_layers=8, num_heads=8)
        self.move_model = _load_state(PatzerModel(**common, output_vocab_size=cc.NUM_ACTIONS),
                                      mdir / "model.ckpt", self.device)

        s = pickle.load(open(cdir / "scalers.pkl", "rb"))
        self.c_scal = {k: (float(s[k]["mean"]), float(s[k]["std"]))
                       for k in ("rating", "log_player_clock", "log_opponent_clock", "log_increment")}
        self.buckets = load_bucket_info(str(cdir / "clock_buckets.json"))
        self.clock_model = _load_state(ClockPatzerModel(**common, n_buckets=self.buckets.n_buckets),
                                       cdir / "model.ckpt", self.device)

        self.common = {}
        for f in list(mdir.glob("common_moves.*.gz")):
            with gzip.open(f, "rt") as fh:
                for line in fh:
                    if line.strip():
                        d = json.loads(line)
                        self.common[d["fen"]] = d["moves"]

    def _recent(self, moves):
        toks = [self.cc.MOVE_TO_ACTION[m] for m in moves[-12:]]
        return [self.cc.PAD_TOKEN] * (12 - len(toks)) + toks

    @torch.no_grad()
    def sample(self, states: list[GameState], rng: np.random.Generator):
        cc, dev = self.cc, self.device
        fens = [s.board.fen() for s in states]
        recent = torch.tensor([self._recent(s.moves) for s in states], dtype=torch.long, device=dev)
        board = torch.tensor(np.stack([cc.tokenize(f) for f in fens]), dtype=torch.long, device=dev)
        clocks = [s.clocks for s in states]

        # clock model -> think time
        def z(x, ms):
            return (x - ms[0]) / ms[1]
        rat_c = torch.tensor([z(s.elo_to_move, self.c_scal["rating"]) for s in states], device=dev)
        feats = torch.tensor([[z(math.log(c[0] + 1), self.c_scal["log_player_clock"]),
                               z(math.log(c[1] + 1), self.c_scal["log_opponent_clock"]),
                               z(math.log(s.inc + 1), self.c_scal["log_increment"])]
                              for s, c in zip(states, clocks)], device=dev)
        ids = torch.cat([recent, board], dim=1)
        p_bucket = torch.softmax(self.clock_model(ids, torch.ones_like(ids), rat_c, feats).float(), 1).cpu().numpy()
        thinks = []
        for p in p_bucket:
            b = rng.choice(len(p), p=p / p.sum())
            thinks.append(float(self.buckets.sample_from_bucket_empirical(b)))

        # move model (skipped for positions the lookup table answers)
        out = [None] * len(states)
        need = []
        for i, f in enumerate(fens):
            hit = self.common.get(f)
            if hit:
                mv, w = zip(*hit.items())
                w = np.asarray(w, dtype=float)
                cand = mv[rng.choice(len(mv), p=w / w.sum())]
                if chess.Move.from_uci(cand) in states[i].board.legal_moves:
                    out[i] = cand
                    continue
            need.append(i)
        if need:
            rat = torch.tensor([z(states[i].elo_to_move, self.m_rating) for i in need], device=dev)
            lt = torch.tensor([z(math.log(1 + clocks[i][0]), self.m_time) for i in need], device=dev)
            logits = self.move_model(recent[need], board[need], rat, lt).float()
            mask = torch.zeros_like(logits, dtype=torch.bool)
            for r, i in enumerate(need):
                for m in states[i].board.legal_moves:
                    mask[r, cc.MOVE_TO_ACTION[m.uci()]] = True
            probs = torch.softmax(logits.masked_fill(~mask, float("-inf")), 1)
            picks = torch.multinomial(probs, 1).squeeze(1).tolist()
            for r, i in enumerate(need):
                out[i] = cc.ACTION_TO_MOVE[picks[r]]
        return list(zip(out, thinks))


# ---------------------------------------------------------------- Maia-3

class Maia3:
    def __init__(self, size: str = "79m", device: str = "cuda"):
        from argparse import Namespace

        from maia3.dataset import get_legal_moves_mask, tokenize_board  # noqa: F401
        from maia3.model_registry import resolve_checkpoint_path, resolve_model_spec
        from maia3.models import MAIA3Model
        from maia3.utils import get_all_possible_moves

        from maia3.model_registry import apply_model_config
        from maia3.uci import parse_args

        spec = resolve_model_spec(size)
        cfg = parse_args(["--model", size, "--device", device])  # UCI defaults, then the preset's architecture
        apply_model_config(cfg, spec)
        path = resolve_checkpoint_path(spec, checkpoint_filename=None, cache_dir=None, revision=None,
                                       local_files_only=False, force_download=False, token=None)
        model = MAIA3Model(cfg).to(device)
        ck = torch.load(path, map_location=device, weights_only=True)
        sd = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
        missing, unexpected = model.load_state_dict({k.replace("smolgen", "gab"): v for k, v in sd.items()},
                                                    strict=False)
        assert not missing and not unexpected, (missing[:5], unexpected[:5])
        self.model, self.cfg, self.device = model.eval(), cfg, device
        self.all_moves = get_all_possible_moves()
        self.move_idx = {m: i for i, m in enumerate(self.all_moves)}

    @staticmethod
    def board_tokens(board: chess.Board) -> torch.Tensor:
        from maia3.dataset import tokenize_board
        return tokenize_board(board)

    def _input(self, s: GameState) -> torch.Tensor:
        h = s.maia_hist[-self.cfg.history:]
        if len(h) < self.cfg.history:
            h = [h[0]] * (self.cfg.history - len(h)) + h
        x = torch.cat(h, dim=1)
        return torch.cat([x, torch.zeros((64, 1))], dim=1)  # clk_ponder = 0, as in the UCI wrapper

    @torch.no_grad()
    def sample(self, states: list[GameState], rng: np.random.Generator):
        from maia3.utils import mirror_move
        x = torch.stack([self._input(s) for s in states]).to(self.device)
        se = torch.tensor([s.elo_to_move for s in states], dtype=torch.long, device=self.device)
        oe = torch.tensor([s.elo_opp for s in states], dtype=torch.long, device=self.device)
        with torch.autocast("cuda", enabled=self.device.startswith("cuda")):
            logits, _, _ = self.model(x, se, oe)
        logits = logits.float()
        mask = torch.zeros_like(logits, dtype=torch.bool)
        for r, s in enumerate(states):
            white = s.board.turn == chess.WHITE
            for m in s.board.legal_moves:
                u = m.uci() if white else mirror_move(m.uci())
                mask[r, self.move_idx[u]] = True
        probs = torch.softmax(logits.masked_fill(~mask, float("-inf")), 1)
        picks = torch.multinomial(probs, 1).squeeze(1).tolist()
        out = []
        for s, p in zip(states, picks):
            u = self.all_moves[p]
            out.append((u if s.board.turn == chess.WHITE else mirror_move(u), None))
        return out

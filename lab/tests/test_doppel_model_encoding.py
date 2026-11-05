"""doppel_model's numpy encoding must equal maia3's torch tokenizer and move vocabulary exactly."""

import chess
import numpy as np
from maia3.dataset import get_legal_moves_mask, tokenize_board
from maia3.utils import get_all_possible_moves, mirror_move

from doppel_model import MOVES, board_planes, legal_mask, mirror_uci


def _boards():
    rng = np.random.default_rng(0)
    for _ in range(200):
        b = chess.Board()
        for _ in range(int(rng.integers(0, 120))):
            legal = list(b.legal_moves)
            if not legal:
                break
            b.push(legal[int(rng.integers(len(legal)))])
        yield b


def test_vocab():
    assert MOVES == get_all_possible_moves()


def test_planes_and_mask():
    idx = {m: i for i, m in enumerate(MOVES)}
    for b in _boards():
        assert np.array_equal(board_planes(b), tokenize_board(b).numpy())
        if b.is_game_over():
            continue
        mask, legal = legal_mask(b)
        assert np.array_equal(mask, get_legal_moves_mask(b, idx).numpy())
        assert sorted(legal) == sorted(m.uci() for m in b.legal_moves)
        for u in legal:
            assert mirror_uci(u) == mirror_move(u)

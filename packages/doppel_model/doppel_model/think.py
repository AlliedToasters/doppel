"""Heuristic think time for a bot move (placeholder until a model has a clock head). Shared by api and lichess_bot."""

from __future__ import annotations

import numpy as np


def heuristic_think_ms(left_ms: float, inc_ms: float, plies: int, n_legal: int, p_top: float,
                       rng: np.random.Generator) -> int:
    """Quick for the first few moves and for forced moves; otherwise the remaining time spread over the moves the
    bot expects to play, scaled up when the policy is unsure (low top-move probability), with lognormal noise.
    Never spends more than a quarter of what is left, and leaves a margin so it cannot flag by thinking."""
    own = plies // 2
    if n_legal == 1:
        t = rng.uniform(300, 900)
    elif own < 3:
        t = rng.uniform(400, 1500)
    else:
        budget = left_ms / max(50 - own, 25) + 0.6 * inc_ms
        t = budget * (0.3 + 1.3 * (1.0 - p_top)) * rng.lognormal(0.0, 0.55)
    cap = max(150.0, min(0.25 * left_ms, left_ms - 1500.0))
    return int(np.clip(t, 150.0, cap))

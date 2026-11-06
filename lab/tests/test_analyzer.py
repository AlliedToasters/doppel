import numpy as np
import pytest

from doppel_lab.analyzer import channels, render, source, transforms
from doppel_lab.analyzer.__main__ import analyze

PGN = """[Event "Rated blitz game"]
[Site "https://lichess.org/abcdefgh"]
[White "w"]
[Black "b"]
[Result "0-1"]
[WhiteElo "1500"]
[BlackElo "1520"]
[TimeControl "180+2"]
[Termination "Normal"]

1. e4 { [%clk 0:03:00] } e5 { [%clk 0:03:00] } 2. Qh5 { [%clk 0:02:58] } Nc6 { [%clk 0:02:57] }
3. Bb5 { [%clk 0:02:50] } Nf6 { [%clk 0:02:40] } 4. Qxf7+ { [%clk 0:02:49] } Kxf7 { [%clk 0:02:41] } 0-1
"""


def test_pgn_source_and_base_channels():
    rec = source.from_pgn(PGN)
    assert rec["game_id"] == "abcdefgh" and rec["base"] == 180 and rec["inc"] == 2
    assert rec["moves"][:2] == ["e2e4", "e7e5"] and rec["clocks"][2] == 178
    pos, mov, boards = channels.base_channels(source.validate(rec))
    assert len(boards) == 9 and len(mov["ply"]) == 8
    # White's 2nd move: clock 180 -> 178 with a 2 s increment = 4 s of thought
    assert mov["think_s"][2] == 4 and np.isnan(mov["think_s"][0])
    # 4. Qxf7+ Kxf7: White wins a pawn, then loses the queen
    assert pos["material_w"][7] == 1 and pos["material_w"][8] == -8


def test_illegal_pgn_raises():
    with pytest.raises(ValueError):
        source.from_pgn(PGN.replace("3. Bb5", "3. Bc4"))  # then 4. Qxf7# is mate and Kxf7 is illegal


def test_live_game_refused():
    with pytest.raises(source.LiveGame):
        source.from_pgn(PGN.replace('[Result "0-1"]', '[Result "*"]').replace(" 0-1\n", " *\n"))


def test_transforms_are_causal():
    rng = np.random.default_rng(0)
    x = rng.normal(size=30)
    y = x.copy()
    y[20:] += 100  # change the future only
    for f in (transforms.ewma, transforms.drawdown, transforms.velocity,
              lambda a: transforms.rolling(a, 5, np.std)):
        assert np.allclose(f(x)[:20], f(y)[:20], equal_nan=True)


def test_transform_values():
    assert np.allclose(transforms.drawdown(np.array([1.0, 3.0, 2.0, 5.0])), [0, 0, -1, 0])
    assert np.array_equal(transforms.since(np.array([0, 1, 0, 0, 1]).astype(bool))[1:], [0, 1, 2, 0])
    assert transforms.lichess_window(15) == 2 and transforms.lichess_window(55) == 5
    assert transforms.lichess_window(200) == 8
    # Sharpness: symmetric in W and L, larger when draws are less likely
    assert channels.sharpness(0.3, 0.2) == pytest.approx(channels.sharpness(0.2, 0.3))
    assert channels.sharpness(0.45, 0.45) > channels.sharpness(0.3, 0.3)


def test_end_to_end_without_engines(tmp_path):
    p = tmp_path / "g.pgn"
    p.write_text(PGN)
    a = analyze(str(p), maia=False, stockfish=False)
    assert a["events"][-1]["kind"] == "end" and a["rec"]["end"] == "resign"
    assert [s["side"] for s in a["summary"]] == ["white", "black"]
    page = render.render(a["rec"], a["pos"], a["mov"], a["side_tf"], a["pos_tf"], a["events"], a["summary"])
    assert page.startswith("<!doctype html>") and "Qxf7+" in page and "https://" not in page.split("<script")[0]

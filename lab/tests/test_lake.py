import io

from doppel_lab import lake


def test_main_line_skips_comments_variations_nags():
    t = "1. e4 {best} (1. d4 d5 (1... Nf6)) 1... e5!? $1 2. Nf3 ; eol comment\n Nc6 3. Bb5 a6 1-0"
    moves, clocks = lake.parse_movetext(t)
    assert moves == ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"]
    assert clocks is None


def test_clocks_kept_only_when_every_ply_has_one():
    full = "1. e4 { [%clk 0:03:00] } 1... e5 { [%clk 0:02:59.5] } 1-0"
    assert lake.parse_movetext(full)[1] == [180.0, 179.5]
    part = "1. e4 { [%clk 0:03:00] } 1... e5 2. Nf3 { [%clk 0:02:58] } 1-0"
    assert lake.parse_movetext(part)[1] is None


def test_illegal_or_empty_dropped():
    assert lake.parse_movetext("1. e5 1-0") is None
    assert lake.parse_movetext("1-0") is None


def test_multiline_games_and_ids():
    pgn = ('[Event "X"]\n[Site "Y"]\n[White "A"]\n[Black "B"]\n[WhiteElo "2500"]\n[Result "1-0"]\n\n'
           '1. e4 e5 2. Nf3\nNc6 1-0\n\n[Event "X"]\n[White "C"]\n[Black "D"]\n[Result "*"]\n\n1. d4 *\n')
    g = list(lake.games(io.StringIO(pgn)))
    assert len(g) == 2 and g[0][1].endswith("Nc6 1-0")
    r = lake.row("twic", *g[0])
    assert r["plies"] == 4 and r["white_elo"] == 2500 and r["black_elo"] is None and len(r["game_id"]) == 16
    assert lake.row("twic", *g[0])["game_id"] == r["game_id"]

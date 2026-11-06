import chess

from doppel_lab.resign import classify, game_features, material


def board(moves):
    b = chess.Board()
    for m in moves:
        b.push_uci(m)
    return b


def test_classify():
    mate = board(["f2f3", "e7e5", "g2g4", "d8h4"])
    assert classify("0-1", "Normal", mate) == ("mate", None, False)
    b = board(["e2e4", "e7e5"])  # white to move
    assert classify("0-1", "Normal", b) == ("resign", "w", True)
    assert classify("1-0", "Normal", b) == ("resign", "b", False)  # black resigned on white's turn
    assert classify("1/2-1/2", "Normal", b)[0] == "draw"
    assert classify("1-0", "Time forfeit", b)[0] == "flag"


def test_material_is_side_to_move():
    b = chess.Board("4k3/8/8/8/8/8/8/3QK3 b - - 0 1")
    assert material(b) == -9


def test_target_only_on_final_resigner_position():
    moves = ["e2e4", "e7e5", "g1f3"]  # black to move at the end, black resigned
    rows, ys = game_features((moves, [300.0, 300.0, 299.0], 1500, 1500, 300, 0, "resign", "b"))
    assert len(rows) == 4 and ys == [0, 0, 0, 1]
    rows, ys = game_features((moves, [300.0, 300.0, 299.0], 1500, 1500, 300, 0, "flag", None))
    assert len(rows) == 3 and sum(ys) == 0  # final position dropped when nobody resigned


def test_blunder_labels_mover_pov():
    from doppel_lab.blunder import labels
    # start +20; White blunders to -500; Black then blunders back to +600; White holds at +590
    lab = labels([20, -500, 600, 590])
    assert lab["blunder"].tolist() == [1, 1, 0]
    assert lab["cpl"].tolist() == [520, 1100, 10]
    assert (lab["wc_loss"] >= 0).all() and lab["blunder2"][2] == 0

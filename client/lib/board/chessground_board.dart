import 'package:chessground/chessground.dart';
import 'package:dartchess/dartchess.dart';
import 'package:flutter/material.dart';

import 'board_surface.dart';

/// [BoardView] -> chessground adapter shared by skins that render with chessground.
class ChessgroundBoard extends StatefulWidget {
  const ChessgroundBoard({
    super.key,
    required this.view,
    required this.size,
    required this.onMove,
    required this.settings,
  });

  final BoardView view;
  final double size;
  final ValueChanged<Move> onMove;
  final ChessboardSettings settings;

  @override
  State<ChessgroundBoard> createState() => _ChessgroundBoardState();
}

class _ChessgroundBoardState extends State<ChessgroundBoard> {
  late final ChessboardController _controller = ChessboardController(game: _game(widget.view));

  static GameData _game(BoardView v) {
    final p = v.position;
    return GameData(
      fen: p.fen,
      lastMove: v.lastMove,
      playerSide: switch (v.interactive) {
        Side.white => PlayerSide.white,
        Side.black => PlayerSide.black,
        null => PlayerSide.none,
      },
      sideToMove: p.turn,
      kingSquareInCheck: p.isCheck ? p.board.kingOf(p.turn) : null,
      validMoves: makeLegalMoves(p),
    );
  }

  @override
  void didUpdateWidget(ChessgroundBoard old) {
    super.didUpdateWidget(old);
    final v = widget.view;
    if (old.view.position.fen != v.position.fen || old.view.interactive != v.interactive) {
      final newGame = old.view.position.fen != v.position.fen && v.lastMove == null;
      _controller.updatePosition(_game(v), animate: !newGame, resetPremove: newGame);
      // Execute a pending premove once it's the human's turn and it's still legal.
      final pre = _controller.premove;
      if (pre != null && v.interactive == v.position.turn) {
        _controller.premove = null;
        if (v.position.isLegal(pre)) {
          WidgetsBinding.instance.addPostFrameCallback((_) => widget.onMove(pre));
        }
      }
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => Chessboard(
        controller: _controller,
        size: widget.size,
        orientation: widget.view.orientation,
        settings: widget.settings,
        onMove: (move, {viaDragAndDrop}) => widget.onMove(move),
      );
}

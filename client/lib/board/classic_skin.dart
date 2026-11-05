import 'package:chessground/chessground.dart';
import 'package:flutter/material.dart';

import 'board_surface.dart';
import 'chessground_board.dart';

/// The familiar board: chessground (GPL-3.0) with its default look, for players used to it.
class ClassicSkin extends BoardSkin {
  const ClassicSkin();

  @override
  String get id => 'classic';
  @override
  String get name => 'Classic';

  @override
  Widget buildBoard({required BoardView view, required double size, required MoveHandler onMove}) =>
      ChessgroundBoard(
          view: view, size: size, onMove: onMove, settings: const ChessboardSettings(enablePremoves: true));
}

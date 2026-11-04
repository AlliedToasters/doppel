import 'package:dartchess/dartchess.dart';
import 'package:flutter/widgets.dart';

/// Everything a skin needs to draw the board. Game logic lives above this line and never forks per skin.
@immutable
class BoardView {
  const BoardView({
    required this.position,
    required this.orientation,
    required this.interactive,
    this.lastMove,
  });

  final Position position;
  final Side orientation;

  /// Side the human may move (and premove) with; null when the board is read-only.
  final Side? interactive;
  final Move? lastMove;
}

/// A skin: one board widget implementation plus its theme.
///
/// Contract: render [BoardView], call `onMove` with a legal move (for the human side, including executed
/// premoves), never mutate game state itself.
abstract class BoardSkin {
  const BoardSkin();

  String get id;
  String get name;

  Widget buildBoard({
    required BoardView view,
    required double size,
    required ValueChanged<Move> onMove,
  });
}

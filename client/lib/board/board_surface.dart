import 'package:dartchess/dartchess.dart';
import 'package:flutter/material.dart';

import '../game/material.dart';

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
/// premoves), never mutate game state itself. Skins may add feedback (sound, haptics, effects) derived from
/// consecutive views.
abstract class BoardSkin {
  const BoardSkin();

  String get id;
  String get name;

  /// Screen background behind the board.
  Color get background => const Color(0xFF161512);

  Widget buildBoard({
    required BoardView view,
    required double size,
    required ValueChanged<Move> onMove,
  });

  /// Material display for one player bar. Material only: never anything model- or engine-derived.
  Widget buildMaterial(BuildContext context, MaterialCount material, Side side, {required bool isHuman}) {
    const glyph = {Role.queen: '♛', Role.rook: '♜', Role.bishop: '♝', Role.knight: '♞', Role.pawn: '♟'};
    final lead = material.lead(side);
    return Row(mainAxisSize: MainAxisSize.min, children: [
      Text(material.captured[side]!.map((r) => glyph[r]).join(), style: const TextStyle(fontSize: 18)),
      if (lead > 0) Text(' +$lead', style: const TextStyle(fontWeight: FontWeight.bold)),
    ]);
  }
}

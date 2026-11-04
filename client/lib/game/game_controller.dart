import 'package:dartchess/dartchess.dart';
import 'package:flutter/foundation.dart';

import 'opponent.dart';

/// One game against an [Opponent]. Skin-independent: every skin renders this same state.
class GameController extends ChangeNotifier {
  GameController({required this.opponent, required this.humanSide}) {
    _maybeOpponentMove();
  }

  final Opponent opponent;
  final Side humanSide;

  Position position = Chess.initial;
  Move? lastMove;
  final List<String> ucis = [];
  final List<String> sans = [];
  bool thinking = false;
  String? error;

  bool get over => position.isGameOver;

  String? get resultText {
    final o = position.outcome;
    if (o == null) return null;
    if (o.winner == null) return position.isStalemate ? 'Draw · stalemate' : 'Draw';
    return o.winner == humanSide ? 'You win!' : '${opponent.name} wins';
  }

  void play(Move move) {
    if (over || !position.isLegal(move)) return;
    final (next, san) = position.makeSan(move);
    position = next;
    lastMove = move;
    ucis.add(move.uci);
    sans.add(san);
    notifyListeners();
    _maybeOpponentMove();
  }

  Future<void> _maybeOpponentMove() async {
    if (over || position.turn == humanSide) return;
    thinking = true;
    error = null;
    notifyListeners();
    final at = ucis.length;
    try {
      final m = await opponent.move(position, List.of(ucis));
      if (ucis.length != at) return; // game moved on (new game / takeback)
      thinking = false;
      play(m);
    } catch (e) {
      thinking = false;
      error = '$e';
      notifyListeners();
    }
  }

  void retry() => _maybeOpponentMove();
}

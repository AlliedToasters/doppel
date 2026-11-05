import 'package:dartchess/dartchess.dart';

import 'opponent.dart';
import 'session.dart';

export 'session.dart';

/// A local game against an [Opponent] (offline Random, or a stateless backend bot). No clocks.
class GameController extends GameSession {
  GameController({required this.opponent, required this.humanSide}) {
    _maybeOpponentMove();
  }

  final Opponent opponent;
  @override
  final Side humanSide;

  @override
  Position position = Chess.initial;
  @override
  Move? lastMove;
  final List<String> ucis = [];
  @override
  final List<String> sans = [];
  @override
  bool thinking = false;
  @override
  String? error;

  @override
  String get opponentName => opponent.name;

  @override
  bool get over => position.isGameOver;

  @override
  String? get resultText {
    final o = position.outcome;
    if (o == null) return null;
    if (o.winner == null) return position.isStalemate ? 'Draw · stalemate' : 'Draw';
    return o.winner == humanSide ? 'You win!' : '${opponent.name} wins';
  }

  @override
  void play(Move move, {bool premove = false}) {
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

  @override
  void retry() => _maybeOpponentMove();
}

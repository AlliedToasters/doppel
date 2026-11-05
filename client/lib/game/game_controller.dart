import 'package:dartchess/dartchess.dart';
import 'package:flutter/foundation.dart';

import 'opponent.dart';

/// Server-held clock snapshot; [remaining] runs it forward locally between server updates.
class ClockState {
  ClockState({required this.whiteMs, required this.blackMs, required this.running, required this.turn})
      : at = DateTime.now();

  final int whiteMs, blackMs;
  final bool running;
  final Side turn;
  final DateTime at;

  Duration remaining(Side side) {
    var ms = side == Side.white ? whiteMs : blackMs;
    if (running && side == turn) ms -= DateTime.now().difference(at).inMilliseconds;
    return Duration(milliseconds: ms < 0 ? 0 : ms);
  }
}

/// What the play screen renders. Skin-independent: every skin renders this same state.
abstract class GameSession extends ChangeNotifier {
  Side get humanSide;
  String get opponentName;
  Position get position;
  Move? get lastMove;
  List<String> get sans;
  bool get thinking;
  String? get error;
  bool get over;
  String? get resultText;
  ClockState? get clock => null;
  bool get canResign => false;

  void play(Move move);
  void retry();
  void resign() {}
}

/// Replays UCI moves from the start; returns the final position, SANs and last move.
(Position, List<String>, Move?) replay(List<String> ucis) {
  Position pos = Chess.initial;
  final sans = <String>[];
  Move? last;
  for (final u in ucis) {
    final m = Move.parse(u)!;
    final (next, san) = pos.makeSan(m);
    pos = next;
    sans.add(san);
    last = m;
  }
  return (pos, sans, last);
}

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

  @override
  void retry() => _maybeOpponentMove();
}

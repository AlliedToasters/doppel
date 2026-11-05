import 'package:dartchess/dartchess.dart';
import 'package:flutter/foundation.dart';

/// Skin- and transport-independent game session types. No backend imports here: Lichess games use these too.

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

  void play(Move move, {bool premove = false});
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

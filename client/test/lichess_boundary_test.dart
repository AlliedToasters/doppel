import 'dart:io' as io;

import 'package:dartchess/dartchess.dart';
import 'package:doppel/features/lichess/lichess_game.dart';
import 'package:doppel/game/session.dart';
import 'package:flutter_test/flutter_test.dart';

/// The Lichess hard boundary (ARCHITECTURE.md): nothing lib/features/lichess imports, directly or through
/// other files, may reach the Doppel backend connection or model code.
const forbidden = {'lib/config.dart', 'lib/game/server_game.dart', 'lib/game/opponent.dart', 'lib/game/account.dart',
    'lib/game/game_controller.dart'};

Set<String> closure(String start) {
  final seen = <String>{}, todo = [start];
  final imp = RegExp(r'''^(?:import|export)\s+['"]([^'"]+)['"]''', multiLine: true);
  while (todo.isNotEmpty) {
    final f = todo.removeLast();
    if (!seen.add(f)) continue;
    for (final m in imp.allMatches(io.File(f).readAsStringSync())) {
      final u = m.group(1)!;
      if (u.startsWith('package:doppel/')) {
        todo.add('lib/${u.substring(15)}');
      } else if (!u.contains(':')) {
        todo.add(io.File(f).parent.uri.resolve(u).toFilePath());
      }
    }
  }
  return seen;
}

void main() {
  test('features/lichess never reaches the Doppel backend', () {
    for (final f in io.Directory('lib/features/lichess').listSync(recursive: true).whereType<io.File>()) {
      if (!f.path.endsWith('.dart')) continue;
      final reached = closure(f.path).map((p) => p.replaceFirst(RegExp(r'^.*?(?=lib/)'), ''));
      expect(reached.toSet().intersection(forbidden), isEmpty, reason: f.path);
    }
  });

  test('castling goes to Lichess as standard UCI, and Lichess castling replays', () {
    final (pos, _, _) = replay(['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1c4', 'g8f6']);
    expect(standardUci(pos, Move.parse('e1h1')!), 'e1g1');
    expect(standardUci(pos, Move.parse('e1g1')!), 'e1g1');
    expect(standardUci(pos, Move.parse('d2d3')!), 'd2d3');
    final (after, sans, _) = replay(['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1c4', 'g8f6', 'e1g1']);
    expect(sans.last, 'O-O');
    expect(after.board.pieceAt(Square.fromName('g1'))?.role, Role.king);
  });
}

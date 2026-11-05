import 'dart:io';

import 'package:flutter_test/flutter_test.dart';

/// The Lichess hard boundary (ARCHITECTURE.md): lib/features/lichess may not import the Doppel backend
/// connection or any model/inference code, so nothing in a live Lichess game can reach a doppel.
void main() {
  test('features/lichess imports nothing from the Doppel backend', () {
    final forbidden = RegExp(r'''import\s+['"](?:package:doppel/)?(?:\.\./)*(config\.dart|game/|board/doppel)''');
    final bad = <String>[];
    for (final f in Directory('lib/features/lichess').listSync(recursive: true).whereType<File>()) {
      if (!f.path.endsWith('.dart')) continue;
      for (final line in f.readAsLinesSync()) {
        if (forbidden.hasMatch(line)) bad.add('${f.path}: $line');
      }
    }
    expect(bad, isEmpty);
  });
}

import 'package:doppel/board/doppel_skin.dart';
import 'package:doppel/main.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  DoppelSfx.enabled = false;
  testWidgets('play screen renders a board and a random opponent game', (tester) async {
    tester.view.physicalSize = const Size(1080, 2400);
    tester.view.devicePixelRatio = 2.6;
    await tester.runAsync(() async {
      await preloadPieces();
      await tester.pumpWidget(const DoppelApp());
      await Future.delayed(const Duration(milliseconds: 500));
    });
    await tester.pumpAndSettle();
    await tester.runAsync(() async {
      for (final e in find.byType(Image).evaluate()) {
        await precacheImage((e.widget as Image).image, e);
      }
    });
    await tester.pumpAndSettle();
    expect(find.text('You'), findsOneWidget);
    await expectLater(find.byType(DoppelApp), matchesGoldenFile('goldens/play_screen.png'));
  });
}

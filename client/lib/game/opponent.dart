import 'dart:convert';
import 'dart:math';

import 'package:dartchess/dartchess.dart';
import 'package:http/http.dart' as http;

import '../config.dart';

abstract class Opponent {
  String get name;
  Future<Move> move(Position position, List<String> ucis);
}

class RandomOpponent implements Opponent {
  final _rng = Random();
  @override
  String get name => 'Random';

  @override
  Future<Move> move(Position position, List<String> ucis) async {
    await Future.delayed(const Duration(milliseconds: 350));
    final moves = [
      for (final MapEntry(key: from, value: tos) in position.legalMoves.entries)
        for (final to in tos.squares) NormalMove(from: from, to: to),
    ];
    var m = moves[_rng.nextInt(moves.length)];
    if (!position.isLegal(m)) m = NormalMove(from: m.from, to: m.to, promotion: Role.queen);
    return m;
  }
}

/// Bots the backend serves: (id, display name). Hosted games themselves live in server_game.dart.
Future<List<(String, String)>> listBots() async {
  final r = await http.get(Uri.parse('$backendUrl/api/bots')).timeout(const Duration(seconds: 5));
  final list = jsonDecode(r.body)['bots'] as List;
  return [for (final b in list) (b['id'] as String, b['name'] as String)];
}

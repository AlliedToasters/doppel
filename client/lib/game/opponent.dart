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

/// A bot served by the bigbox backend (today: demo/server.py's POST /api/move).
class BackendOpponent implements Opponent {
  BackendOpponent({required this.bot, required this.label, required this.elo});

  final String bot;
  final String label;
  final int elo;

  @override
  String get name => '$label · $elo';

  @override
  Future<Move> move(Position position, List<String> ucis) async {
    final r = await http
        .post(Uri.parse('$backendUrl/api/move'),
            headers: {'content-type': 'application/json'},
            body: jsonEncode({'bot': bot, 'elo': elo, 'moves': ucis}))
        .timeout(const Duration(seconds: 30));
    if (r.statusCode != 200) throw Exception('backend ${r.statusCode}: ${r.body}');
    return Move.parse(jsonDecode(r.body)['move'] as String)!;
  }

  static Future<List<(String, String)>> listBots() async {
    final r = await http.get(Uri.parse('$backendUrl/api/bots')).timeout(const Duration(seconds: 5));
    final list = jsonDecode(r.body)['bots'] as List;
    return [for (final b in list) (b['id'] as String, b['name'] as String)];
  }
}

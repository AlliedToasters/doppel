import 'dart:async';
import 'dart:convert';

import 'package:dartchess/dartchess.dart';
import 'package:http/http.dart' as http;

import '../../game/session.dart';
import 'lichess_auth.dart';

/// Streams newline-delimited JSON from a long-lived Lichess endpoint. Closing [client] ends it.
Stream<Map<String, dynamic>> ndjson(
  http.Client client,
  String method,
  String path, {
  Map<String, String>? form,
}) async* {
  final req = http.Request(method, Uri.parse('${LichessAuth.host}$path'))..headers.addAll(LichessAuth.headers);
  if (form != null) req.bodyFields = form;
  final res = await client.send(req);
  if (res.statusCode != 200) {
    final body = await res.stream.bytesToString();
    throw Exception('lichess ${res.statusCode}: $body');
  }
  await for (final line in res.stream.transform(utf8.decoder).transform(const LineSplitter())) {
    if (line.trim().isEmpty) continue; // keep-alive
    yield jsonDecode(line) as Map<String, dynamic>;
  }
}

/// Lichess sends standard UCI (e1g1); dartchess may produce king-takes-rook (e1h1). Convert for sending.
String standardUci(Position pos, Move move) {
  if (move is NormalMove && pos.board.pieceAt(move.from)?.role == Role.king) {
    final rook = pos.board.pieceAt(move.to);
    if (rook != null && rook.role == Role.rook && rook.color == pos.turn) {
      final f = move.to.file > move.from.file ? 6 : 2;
      return NormalMove(from: move.from, to: Square.fromCoords(File(f), move.from.rank)).uci;
    }
  }
  return move.uci;
}

/// A pending public seek. [game] completes when Lichess pairs us; [cancel] withdraws the seek.
class LichessSeek {
  LichessSeek._(this.game, this._clients);
  final Future<LichessGame> game;
  final List<http.Client> _clients;

  void cancel() {
    for (final c in _clients) {
      c.close();
    }
  }

  /// Real-time seek ([minutes]+[inc]). Board API games need rapid or slower time controls.
  static LichessSeek start({required int minutes, required int inc, bool rated = false}) {
    final events = http.Client(), seek = http.Client();
    final done = Completer<LichessGame>();
    // The event stream announces the paired game; the seek request stays open until then.
    () async {
      try {
        await for (final e in ndjson(events, 'GET', '/api/stream/event')) {
          if (e['type'] == 'gameStart') {
            final id = (e['game'] as Map<String, dynamic>)['gameId'] as String;
            if (!done.isCompleted) done.complete(LichessGame(id));
            break;
          }
        }
      } catch (err) {
        if (!done.isCompleted) done.completeError(err);
      } finally {
        events.close();
        seek.close();
      }
    }();
    () async {
      try {
        await for (final _ in ndjson(
          seek,
          'POST',
          '/api/board/seek',
          form: {'time': '$minutes', 'increment': '$inc', 'rated': '$rated'},
        )) {}
      } catch (err) {
        if (!done.isCompleted) done.completeError(err);
        events.close();
      }
    }();
    return LichessSeek._(done.future, [events, seek]);
  }
}

/// Ongoing Lichess games of the signed-in account (to resume one started here or on lichess.org).
Future<List<String>> lichessPlaying() async {
  final r = await http
      .get(Uri.parse('${LichessAuth.host}/api/account/playing'), headers: LichessAuth.headers)
      .timeout(const Duration(seconds: 10));
  if (r.statusCode != 200) return [];
  final list = (jsonDecode(r.body) as Map<String, dynamic>)['nowPlaying'] as List<dynamic>;
  return [for (final g in list) (g as Map<String, dynamic>)['gameId'] as String];
}

/// A game hosted by Lichess, played through the Board API. Lichess is the authority on moves and clocks;
/// nothing here touches the Doppel backend.
class LichessGame extends GameSession {
  LichessGame(this.id) {
    _connect();
  }

  final String id;
  http.Client? _client;
  bool _disposed = false;

  @override
  Side humanSide = Side.white;
  @override
  String opponentName = 'Lichess';
  @override
  Position position = Chess.initial;
  @override
  Move? lastMove;
  @override
  List<String> sans = [];
  List<String> _ucis = [];
  @override
  String? error;
  @override
  ClockState? clock;
  String status = 'created';
  String? winner;
  bool opponentDrawOffer = false;
  bool connected = false;

  @override
  bool get thinking => false;
  @override
  bool get over => !(status == 'created' || status == 'started');
  @override
  bool get canResign => !over;
  bool get canAbort => !over && _ucis.length < 2;

  @override
  String? get resultText {
    if (!over) return null;
    if (status == 'aborted') return 'Game aborted';
    if (winner == null) return status == 'stalemate' ? 'Draw · stalemate' : 'Draw';
    final why = const {
      'mate': 'checkmate',
      'resign': 'resignation',
      'outoftime': 'time',
      'timeout': 'opponent left',
    }[status];
    final who = winner == humanSide.name ? 'You win' : '$opponentName wins';
    return why == null ? who : '$who · $why';
  }

  Future<void> _connect() async {
    while (!_disposed && !over) {
      _client = http.Client();
      try {
        await for (final e in ndjson(_client!, 'GET', '/api/board/game/stream/$id')) {
          connected = true;
          error = null;
          if (e['type'] == 'gameFull') {
            _full(e);
          } else if (e['type'] == 'gameState') {
            _state(e);
          }
        }
      } catch (err) {
        if (_disposed) return;
        error = '$err';
        notifyListeners();
      } finally {
        _client?.close();
      }
      connected = false;
      if (!_disposed && !over) await Future<void>.delayed(const Duration(seconds: 2));
    }
  }

  void _full(Map<String, dynamic> e) {
    final w = e['white'] as Map<String, dynamic>, b = e['black'] as Map<String, dynamic>;
    final me = LichessAuth.username?.toLowerCase();
    humanSide = (w['id'] as String?) == me ? Side.white : Side.black;
    final opp = humanSide == Side.white ? b : w;
    final name =
        (opp['name'] ?? opp['id'] ?? (opp['aiLevel'] != null ? 'Stockfish ${opp['aiLevel']}' : 'Anonymous')) as String;
    opponentName = opp['rating'] != null ? '$name (${opp['rating']})' : name;
    _state(e['state'] as Map<String, dynamic>);
  }

  void _state(Map<String, dynamic> s) {
    final moves = (s['moves'] as String? ?? '').split(' ').where((m) => m.isNotEmpty).toList();
    if (moves.length != _ucis.length || moves.join() != _ucis.join()) {
      final (p, sn, last) = replay(moves);
      position = p;
      sans = sn;
      lastMove = last;
      _ucis = moves;
    }
    status = s['status'] as String? ?? status;
    winner = s['winner'] as String?;
    opponentDrawOffer = (humanSide == Side.white ? s['bdraw'] : s['wdraw']) == true;
    clock = ClockState(
      whiteMs: (s['wtime'] as num? ?? 0).toInt(),
      blackMs: (s['btime'] as num? ?? 0).toInt(),
      running: status == 'started' && _ucis.length >= 2, // Lichess clocks start after both first moves
      turn: position.turn,
    );
    notifyListeners();
  }

  @override
  void play(Move move, {bool premove = false}) {
    if (over || position.turn != humanSide || !position.isLegal(move)) return;
    final uci = standardUci(position, move);
    // Optimistic: show the move now; the next gameState confirms it or puts the board back.
    final (next, san) = position.makeSan(move);
    final before = (position, lastMove, sans, _ucis);
    position = next;
    lastMove = move;
    sans = [...sans, san];
    _ucis = [..._ucis, uci];
    notifyListeners();
    _post('/move/$uci').then((ok) {
      if (!ok && _ucis.length == before.$4.length + 1) {
        position = before.$1;
        lastMove = before.$2;
        sans = before.$3;
        _ucis = before.$4;
        notifyListeners();
      }
    });
  }

  Future<bool> _post(String action) async {
    try {
      final r = await http
          .post(Uri.parse('${LichessAuth.host}/api/board/game/$id$action'), headers: LichessAuth.headers)
          .timeout(const Duration(seconds: 10));
      if (r.statusCode == 200) return true;
      error = 'lichess ${r.statusCode}: ${r.body}';
    } catch (e) {
      error = '$e';
    }
    notifyListeners();
    return false;
  }

  @override
  void resign() => _post(canAbort ? '/abort' : '/resign');

  /// Offer a draw, or accept the opponent's offer.
  void offerDraw() => _post('/draw/yes');
  void declineDraw() => _post('/draw/no');

  @override
  void retry() {
    _client?.close(); // the loop reconnects
  }

  @override
  void dispose() {
    _disposed = true;
    _client?.close();
    super.dispose();
  }
}

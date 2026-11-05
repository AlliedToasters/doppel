import 'dart:async';
import 'dart:convert';

import 'package:dartchess/dartchess.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

import '../config.dart';
import 'game_controller.dart';

/// A Doppel-hosted game: the backend (backend/api) holds the moves, the clocks and the doppel's z.
/// This class sends intents over a WebSocket and renders whatever state the server returns. Moves are shown
/// optimistically and replaced by the server's list on every update. The game id is saved, so a restarted
/// app resumes the game (see [resumeSaved]).
class ServerGame extends GameSession {
  ServerGame._(this.id, this.opponentLabel, Map<String, dynamic> state) {
    _apply(state);
    _connect();
  }

  static const _prefKey = 'server_game_id';

  final String id;
  final String opponentLabel;
  late Side _human;
  late int _botElo;
  List<String> _ucis = [];
  Position _position = Chess.initial;
  List<String> _sans = [];
  Move? _last;
  ClockState? _clock;
  String _status = 'active';
  String? _result, _termination;
  String _pace = 'realtime';
  bool _pending = false; // our move sent, server state not back yet
  String? _error;
  WebSocketChannel? _ws;
  StreamSubscription? _sub;
  Timer? _reconnect;
  int _backoff = 1;
  bool _disposed = false;

  static Future<ServerGame> create({
    required String bot,
    required String label,
    required int botElo,
    required Side humanSide,
    required int baseS,
    required int incS,
    required String pace,
  }) async {
    final r = await http
        .post(Uri.parse('$backendUrl/api/games'),
            headers: {'content-type': 'application/json'},
            body: jsonEncode({
              'bot': bot,
              'bot_elo': botElo,
              'color': humanSide == Side.white ? 'white' : 'black',
              'base_s': baseS,
              'inc_s': incS,
              'pace': pace,
            }))
        .timeout(const Duration(seconds: 30));
    if (r.statusCode != 200) throw Exception('backend ${r.statusCode}: ${r.body}');
    final state = jsonDecode(r.body) as Map<String, dynamic>;
    (await SharedPreferences.getInstance()).setString(_prefKey, state['id'] as String);
    return ServerGame._(state['id'] as String, label, state);
  }

  /// The saved game if it is still in progress, else null.
  static Future<ServerGame?> resumeSaved(String Function(String botId) labelFor) async {
    final prefs = await SharedPreferences.getInstance();
    final id = prefs.getString(_prefKey);
    if (id == null) return null;
    final r = await http.get(Uri.parse('$backendUrl/api/games/$id')).timeout(const Duration(seconds: 5));
    if (r.statusCode != 200) {
      prefs.remove(_prefKey);
      return null;
    }
    final state = jsonDecode(r.body) as Map<String, dynamic>;
    if (state['status'] != 'active') {
      prefs.remove(_prefKey);
      return null;
    }
    return ServerGame._(id, labelFor(state['bot'] as String), state);
  }

  // ---------- GameSession ----------
  @override
  Side get humanSide => _human;
  @override
  String get opponentName => '$opponentLabel · $_botElo';
  @override
  Position get position => _position;
  @override
  Move? get lastMove => _last;
  @override
  List<String> get sans => _sans;
  @override
  ClockState? get clock => _clock;
  @override
  bool get over => _status != 'active';
  @override
  bool get thinking => !over && (_pending || _position.turn != _human);
  @override
  String? get error => _error;
  @override
  bool get canResign => !over;

  /// "realtime": the bot takes its think time on the wall clock. "fast": it replies at once, and the
  /// same think time is still charged to its clock.
  String get pace => _pace;

  void setPace(String pace) {
    _pace = pace;
    notifyListeners();
    _send({'type': 'pace', 'pace': pace});
  }

  @override
  String? get resultText {
    if (_result == null) return null;
    final why = switch (_termination) {
      'checkmate' => 'checkmate',
      'timeout' => 'on time',
      'resign' => 'resignation',
      'stalemate' => 'stalemate',
      'timeout_vs_insufficient' => 'timeout vs insufficient material',
      'insufficient_material' => 'insufficient material',
      'threefold_repetition' || 'fivefold_repetition' => 'repetition',
      'fifty_moves' || 'seventyfive_moves' => '50-move rule',
      final t => t ?? '',
    };
    if (_result == '1/2-1/2') return 'Draw · $why';
    final whiteWon = _result == '1-0';
    final won = (whiteWon && _human == Side.white) || (!whiteWon && _human == Side.black);
    return won ? 'You win · $why' : '$opponentLabel wins · $why';
  }

  @override
  void play(Move move) {
    if (over || _pending || _position.turn != _human || !_position.isLegal(move)) return;
    final (next, san) = _position.makeSan(move);
    final ply = _ucis.length;
    _position = next;
    _sans = [..._sans, san];
    _ucis = [..._ucis, move.uci];
    _last = move;
    _pending = true;
    notifyListeners();
    _send({'type': 'move', 'uci': move.uci, 'ply': ply});
  }

  @override
  void resign() => _send({'type': 'resign'});

  @override
  void retry() {
    _error = null;
    _backoff = 1;
    _connect();
    notifyListeners();
  }

  // ---------- wire ----------
  void _apply(Map<String, dynamic> s) {
    _human = s['human_color'] == 'white' ? Side.white : Side.black;
    _botElo = s['bot_elo'] as int;
    final moves = (s['moves'] as List).cast<String>();
    if (!listEquals(moves, _ucis)) {
      final (pos, sans, last) = replay(moves);
      _position = pos;
      _sans = sans;
      _last = last;
      _ucis = moves;
    }
    _pending = false;
    final c = s['clock'] as Map<String, dynamic>;
    _clock = ClockState(
      whiteMs: c['white_ms'] as int,
      blackMs: c['black_ms'] as int,
      running: c['running'] as bool,
      turn: s['turn'] == 'white' ? Side.white : Side.black,
    );
    _status = s['status'] as String;
    _result = s['result'] as String?;
    _termination = s['termination'] as String?;
    _pace = (s['pace'] as String?) ?? _pace;
    if (over) SharedPreferences.getInstance().then((p) => p.remove(_prefKey));
  }

  void _connect() {
    if (_disposed) return;
    _reconnect?.cancel();
    _sub?.cancel();
    _ws?.sink.close();
    final url = Uri.parse(backendUrl).replace(
        scheme: backendUrl.startsWith('https') ? 'wss' : 'ws', path: '/api/games/$id/ws');
    final ws = WebSocketChannel.connect(url);
    _ws = ws;
    // Events from a socket we already replaced are ignored: closing the old one fires its onDone, which
    // must not schedule yet another reconnect (that loop opened 3 sockets per resume).
    _sub = ws.stream.listen((raw) {
      if (!identical(_ws, ws)) return;
      _backoff = 1;
      final msg = jsonDecode(raw as String) as Map<String, dynamic>;
      if (msg['type'] == 'state') {
        _error = null;
        _apply(msg['game'] as Map<String, dynamic>);
      } else if (msg['type'] == 'error' && msg['status'] == 503) {
        _error = 'Opponent unavailable: ${msg['detail']}';
      } else if (msg['type'] == 'error') {
        // Refused intent (stale ply, illegal): drop the optimistic move and take the server's state.
        _error = null;
        _pending = false;
        _ucis = [];
        _send({'type': 'sync'});
      }
      notifyListeners();
    }, onError: (Object e) {
      if (identical(_ws, ws)) _lost('$e');
    }, onDone: () {
      if (identical(_ws, ws)) _lost('connection closed');
    });
  }

  void _lost(String why) {
    if (_disposed || over) return;
    _reconnect?.cancel();
    _pending = false;
    _error = 'Reconnecting… ($why)';
    notifyListeners();
    _reconnect = Timer(Duration(seconds: _backoff), _connect);
    _backoff = (_backoff * 2).clamp(1, 16);
  }

  void _send(Map<String, dynamic> msg) {
    try {
      _ws?.sink.add(jsonEncode(msg));
    } catch (e) {
      _lost('$e');
    }
  }

  @override
  void dispose() {
    _disposed = true;
    _reconnect?.cancel();
    _sub?.cancel();
    _ws?.sink.close();
    super.dispose();
  }
}

bool listEquals(List<String> a, List<String> b) {
  if (a.length != b.length) return false;
  for (var i = 0; i < a.length; i++) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

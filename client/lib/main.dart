import 'dart:async';

import 'package:chessground/chessground.dart';
import 'package:dartchess/dartchess.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'board/board_surface.dart';
import 'board/classic_skin.dart';
import 'board/doppel_skin.dart';
import 'features/lichess/lichess_auth.dart';
import 'features/lichess/lichess_game.dart';
import 'game/account.dart';
import 'game/game_controller.dart';
import 'game/material.dart';
import 'game/opponent.dart';
import 'game/server_game.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await preloadPieces();
  await LichessAuth.load();
  runApp(const DoppelApp());
}

/// Decode piece images before the first frame so the board never appears without pieces.
Future<void> preloadPieces() async {
  final dpr = WidgetsBinding.instance.platformDispatcher.implicitView?.devicePixelRatio ?? 1.0;
  await ChessgroundImages.instance.loadAll(const ChessboardSettings().pieceAssets, devicePixelRatio: dpr);
  await ChessgroundImages.instance.loadAll(doppelPieceAssets, devicePixelRatio: dpr);
}

const skins = <BoardSkin>[DoppelSkin(), ClassicSkin()];

class DoppelApp extends StatelessWidget {
  const DoppelApp({super.key});

  @override
  Widget build(BuildContext context) => MaterialApp(
    title: 'Doppel',
    debugShowCheckedModeBanner: false,
    theme: ThemeData(colorSchemeSeed: const Color(0xFF58CC02), brightness: Brightness.dark, useMaterial3: true),
    home: const PlayScreen(),
  );
}

class PlayScreen extends StatefulWidget {
  const PlayScreen({super.key});

  @override
  State<PlayScreen> createState() => _PlayScreenState();
}

class _PlayScreenState extends State<PlayScreen> {
  BoardSkin skin = skins.first;
  List<(String, String)> bots = [];
  String botId = 'random';
  int elo = 1500;
  Side humanSide = Side.white;
  (int, int) tc = (300, 3);
  String pace = 'realtime';
  GameSession? game;
  String? startError;
  LichessSeek? seeking;
  (int, int) lichessTc = (10, 0);
  bool lichessRated = false;

  /// Board API games must be rapid or slower.
  static const lichessTimeControls = [(10, 0), (10, 5), (15, 10), (30, 0)];

  static const timeControls = [(180, 2), (300, 3), (600, 0), (900, 10)];

  @override
  void initState() {
    super.initState();
    _boot();
  }

  Future<void> _boot() async {
    // A live Lichess game (started here or on lichess.org) takes over the screen: its clock is running.
    if (LichessAuth.signedIn) {
      try {
        final ids = await lichessPlaying();
        if (ids.isNotEmpty) {
          setState(() => game = LichessGame(ids.first));
          return;
        }
      } catch (_) {}
    }
    try {
      final saved = (await SharedPreferences.getInstance()).getString('bot_pace');
      if (saved != null) pace = saved;
      bots = await listBots();
      if (bots.isNotEmpty) botId = bots.first.$1;
      await Account.ensure();
      final resumed = await ServerGame.resumeSaved(_label);
      if (resumed != null) {
        if (resumed.pace != pace) resumed.setPace(pace);
        setState(() => game = resumed);
        return;
      }
    } catch (_) {
      // backend unreachable: fall through to an offline game
    }
    _newGame();
  }

  String _label(String id) => bots.firstWhere((b) => b.$1 == id, orElse: () => (id, id)).$2;

  Future<void> _newGame() async {
    final old = game;
    seeking?.cancel();
    setState(() {
      game = null;
      startError = null;
      seeking = null;
    });
    old?.dispose();
    if (botId == 'random' || bots.isEmpty) {
      setState(() => game = GameController(opponent: RandomOpponent(), humanSide: humanSide));
      return;
    }
    try {
      final g = await ServerGame.create(
        bot: botId,
        label: _label(botId),
        botElo: elo,
        humanSide: humanSide,
        baseS: tc.$1,
        incS: tc.$2,
        pace: pace,
      );
      if (!mounted) return g.dispose();
      setState(() => game = g);
    } catch (e) {
      setState(() => startError = '$e');
    }
  }

  Future<void> _seekLichess() async {
    final old = game;
    seeking?.cancel();
    final s = LichessSeek.start(minutes: lichessTc.$1, inc: lichessTc.$2, rated: lichessRated);
    setState(() {
      game = null;
      startError = null;
      seeking = s;
    });
    old?.dispose();
    try {
      final g = await s.game;
      if (!mounted || seeking != s) return g.dispose();
      setState(() {
        seeking = null;
        game = g;
      });
    } catch (e) {
      if (mounted && seeking == s) {
        setState(() {
          seeking = null;
          startError = '$e';
        });
      }
    }
  }

  /// Applies to the next game and, mid-game, to the current one from the bot's next move.
  void _setPace(String p) {
    setState(() => pace = p);
    SharedPreferences.getInstance().then((prefs) => prefs.setString('bot_pace', p));
    final g = game;
    if (g is ServerGame && !g.over) g.setPace(p);
  }

  Future<void> _confirmResign(GameSession g) async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (c) => AlertDialog(
        title: const Text('Resign?'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(c, false), child: const Text('Keep playing')),
          FilledButton(onPressed: () => Navigator.pop(c, true), child: const Text('Resign')),
        ],
      ),
    );
    if (ok == true) g.resign();
  }

  @override
  Widget build(BuildContext context) {
    final g = game;
    return Scaffold(
      backgroundColor: skin.background,
      appBar: AppBar(
        backgroundColor: skin.background,
        title: const Text('Doppel'),
        actions: [
          if (g != null)
            ListenableBuilder(
              listenable: g,
              builder: (context, _) => g.canResign
                  ? IconButton(icon: const Icon(Icons.flag), tooltip: 'Resign', onPressed: () => _confirmResign(g))
                  : const SizedBox.shrink(),
            ),
          if (g is LichessGame)
            ListenableBuilder(
              listenable: g,
              builder: (context, _) => g.over || g.canAbort
                  ? const SizedBox.shrink()
                  : IconButton(icon: const Icon(Icons.handshake), tooltip: 'Offer draw', onPressed: g.offerDraw),
            ),
          IconButton(icon: const Icon(Icons.tune), onPressed: _settings),
          IconButton(icon: const Icon(Icons.add), tooltip: 'New game', onPressed: _newGame),
        ],
      ),
      body: g == null && seeking != null
          ? Center(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const CircularProgressIndicator(),
                  const SizedBox(height: 16),
                  Text(
                    'Looking for a ${lichessTc.$1}+${lichessTc.$2} ${lichessRated ? 'rated' : 'casual'} game on Lichess',
                  ),
                  const SizedBox(height: 12),
                  OutlinedButton(onPressed: _newGame, child: const Text('Cancel')),
                ],
              ),
            )
          : g == null
          ? Center(
              child: startError == null
                  ? const CircularProgressIndicator()
                  : Padding(
                      padding: const EdgeInsets.all(24),
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          const Icon(Icons.wifi_off, size: 40),
                          const SizedBox(height: 8),
                          Text('Could not start a game\n$startError', textAlign: TextAlign.center),
                          const SizedBox(height: 12),
                          FilledButton(onPressed: _newGame, child: const Text('Retry')),
                        ],
                      ),
                    ),
            )
          : ListenableBuilder(listenable: g, builder: (context, _) => _game(context, g)),
    );
  }

  Widget _game(BuildContext context, GameSession g) {
    final width = MediaQuery.sizeOf(context).width;
    final mat = MaterialCount(g.position.board);
    return Column(
      children: [
        _PlayerBar(
          name: g.opponentName,
          thinking: g.thinking,
          clock: g.clock,
          side: g.humanSide.opposite,
          material: skin.buildMaterial(context, mat, g.humanSide.opposite, isHuman: false),
        ),
        skin.buildBoard(
          view: BoardView(
            position: g.position,
            orientation: g.humanSide,
            interactive: g.over ? null : g.humanSide,
            lastMove: g.lastMove,
          ),
          size: width,
          onMove: (m, {premove = false}) => g.play(m, premove: premove),
        ),
        _PlayerBar(
          name: 'You',
          clock: g.clock,
          side: g.humanSide,
          material: skin.buildMaterial(context, mat, g.humanSide, isHuman: true),
        ),
        if (g is LichessGame && g.opponentDrawOffer && !g.over)
          ListTile(
            leading: const Icon(Icons.handshake),
            title: const Text('Draw offered'),
            trailing: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                TextButton(onPressed: g.declineDraw, child: const Text('Decline')),
                FilledButton(onPressed: g.offerDraw, child: const Text('Accept')),
              ],
            ),
          ),
        if (g.error != null)
          ListTile(
            leading: const Icon(Icons.wifi_off),
            title: Text(g is ServerGame || g is LichessGame ? 'Connection lost' : 'Opponent unreachable'),
            subtitle: Text(g.error!, maxLines: 2, overflow: TextOverflow.ellipsis),
            trailing: TextButton(onPressed: g.retry, child: const Text('Retry')),
          ),
        if (g.resultText != null)
          Padding(
            padding: const EdgeInsets.all(16),
            child: FilledButton.icon(
              onPressed: g is LichessGame ? _seekLichess : _newGame,
              icon: const Icon(Icons.replay),
              label: Text('${g.resultText} · Play again'),
            ),
          ),
        Expanded(
          child: SingleChildScrollView(
            reverse: true,
            padding: const EdgeInsets.all(12),
            child: Text(_moveList(g.sans), style: const TextStyle(fontFamily: 'monospace')),
          ),
        ),
      ],
    );
  }

  static String _moveList(List<String> sans) =>
      [for (var i = 0; i < sans.length; i += 2) '${i ~/ 2 + 1}. ${sans[i]} ${i + 1 < sans.length ? sans[i + 1] : ''}']
          .join('  ');

  Future<void> _settings() async {
    // Account state for the sheet: rating and training consent (null when the backend is unreachable).
    Map<String, dynamic>? me;
    (String, String)? consent;
    try {
      if (await Account.ensure() != null) {
        me = await Account.me();
        consent = await Account.consentText();
      }
    } catch (_) {}
    if (!mounted) return;
    await showModalBottomSheet(
      isScrollControlled: true,
      context: context,
      builder: (context) => StatefulBuilder(
        builder: (context, set) => SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              DropdownButton<String>(
                value: botId,
                isExpanded: true,
                items: [
                  for (final (id, name) in bots) DropdownMenuItem(value: id, child: Text(name)),
                  const DropdownMenuItem(value: 'random', child: Text('Random (offline)')),
                ],
                onChanged: (v) => set(() => botId = v!),
              ),
              Text('Opponent rating: $elo'),
              Slider(
                value: elo.toDouble(),
                min: 600,
                max: 2600,
                divisions: 20,
                onChanged: (v) => set(() => elo = v.round()),
              ),
              SegmentedButton<Side>(
                segments: const [
                  ButtonSegment(value: Side.white, label: Text('White')),
                  ButtonSegment(value: Side.black, label: Text('Black')),
                ],
                selected: {humanSide},
                onSelectionChanged: (s) => set(() => humanSide = s.first),
              ),
              const SizedBox(height: 8),
              Wrap(
                spacing: 8,
                children: [
                  for (final t in timeControls)
                    ChoiceChip(
                      label: Text('${t.$1 ~/ 60}+${t.$2}'),
                      selected: tc == t,
                      onSelected: (_) => set(() => tc = t),
                    ),
                ],
              ),
              const SizedBox(height: 12),
              SegmentedButton<String>(
                segments: const [
                  ButtonSegment(value: 'realtime', label: Text('Realtime'), icon: Icon(Icons.schedule)),
                  ButtonSegment(value: 'fast', label: Text('Fast'), icon: Icon(Icons.bolt)),
                ],
                selected: {pace},
                onSelectionChanged: (s) => set(() => _setPace(s.first)),
              ),
              Padding(
                padding: const EdgeInsets.only(top: 4),
                child: Text(
                  pace == 'realtime'
                      ? 'The bot takes its think time for real; watch its clock run.'
                      : 'The bot replies instantly; its think time still comes off its clock.',
                  style: Theme.of(context).textTheme.bodySmall,
                ),
              ),
              const SizedBox(height: 8),
              DropdownButton<String>(
                value: skin.id,
                isExpanded: true,
                items: [for (final s in skins) DropdownMenuItem(value: s.id, child: Text('Skin: ${s.name}'))],
                onChanged: (v) => set(() => skin = skins.firstWhere((s) => s.id == v)),
              ),
              if (me != null && consent != null) ...[
                const Divider(height: 24),
                if (me!['rating'] != null)
                  Text(
                    'Your rating: ${(me!['rating']['rating'] as num).round()} '
                    '± ${(me!['rating']['rd'] as num).round()} (${me!['rating']['games']} rated games)',
                  ),
                SwitchListTile(
                  contentPadding: EdgeInsets.zero,
                  title: const Text('Help train doppels'),
                  subtitle: Text(consent.$2, style: Theme.of(context).textTheme.bodySmall),
                  value: me!['consent']['training'] as bool,
                  onChanged: (v) async {
                    try {
                      final m = await Account.setTrainingConsent(v, consent!.$1);
                      set(() => me = m);
                    } catch (_) {}
                  },
                ),
              ],
              const Divider(height: 24),
              ListTile(
                contentPadding: EdgeInsets.zero,
                leading: const Icon(Icons.public),
                title: Text(LichessAuth.signedIn ? 'Lichess: ${LichessAuth.username}' : 'Sign in with Lichess'),
                subtitle: Text(
                  LichessAuth.signedIn
                      ? 'Stays on this phone. Doppel never sees your Lichess games.'
                      : 'Play your Lichess games here, with this board.',
                  style: Theme.of(context).textTheme.bodySmall,
                ),
                trailing: LichessAuth.signedIn
                    ? TextButton(
                        onPressed: () async {
                          await LichessAuth.signOut();
                          set(() {});
                        },
                        child: const Text('Sign out'),
                      )
                    : null,
                onTap: LichessAuth.signedIn
                    ? null
                    : () async {
                        try {
                          await LichessAuth.signIn();
                        } catch (e) {
                          if (context.mounted) {
                            ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Lichess sign-in: $e')));
                          }
                        }
                        set(() {});
                      },
              ),
              if (LichessAuth.signedIn) ...[
                Wrap(
                  spacing: 8,
                  children: [
                    for (final t in lichessTimeControls)
                      ChoiceChip(
                        label: Text('${t.$1}+${t.$2}'),
                        selected: lichessTc == t,
                        onSelected: (_) => set(() => lichessTc = t),
                      ),
                  ],
                ),
                SwitchListTile(
                  contentPadding: EdgeInsets.zero,
                  title: const Text('Rated'),
                  value: lichessRated,
                  onChanged: (v) => set(() => lichessRated = v),
                ),
                OutlinedButton.icon(
                  onPressed: () {
                    Navigator.pop(context);
                    _seekLichess();
                  },
                  icon: const Icon(Icons.public),
                  label: const Text('Find a Lichess game'),
                ),
              ],
              const SizedBox(height: 8),
              FilledButton(
                onPressed: () {
                  Navigator.pop(context);
                  _newGame();
                },
                child: const Text('Start new game'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _PlayerBar extends StatelessWidget {
  const _PlayerBar({required this.name, required this.material, this.thinking = false, this.clock, this.side});

  final String name;
  final Widget material;
  final bool thinking;
  final ClockState? clock;
  final Side? side;

  @override
  Widget build(BuildContext context) => Padding(
    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
    child: Row(
      children: [
        Text(name, style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w800)),
        const SizedBox(width: 8),
        if (thinking) const SizedBox(width: 14, height: 14, child: CircularProgressIndicator(strokeWidth: 2)),
        const Spacer(),
        material,
        if (clock != null && side != null) ...[
          const SizedBox(width: 10),
          _Clock(clock: clock!, side: side!, warn: name == 'You'),
        ],
      ],
    ),
  );
}

/// Counts down locally from the last server snapshot; the server stays the authority.
class _Clock extends StatefulWidget {
  const _Clock({required this.clock, required this.side, this.warn = false});
  final ClockState clock;
  final Side side;

  /// Low-time cues for the human's own clock: an alert + buzz on dropping under 20 s, a tick each second under 10 s.
  final bool warn;

  @override
  State<_Clock> createState() => _ClockState();
}

class _ClockState extends State<_Clock> {
  Timer? _tick;
  bool _warned = false;
  int _lastTickSec = -1;

  @override
  void initState() {
    super.initState();
    _tick = Timer.periodic(const Duration(milliseconds: 100), (_) {
      if (!(widget.clock.running && widget.clock.turn == widget.side)) return;
      if (widget.warn) _cues(widget.clock.remaining(widget.side));
      setState(() {});
    });
  }

  void _cues(Duration left) {
    if (left > const Duration(seconds: 25)) _warned = false; // re-arm once increments lift it clear
    if (!_warned && left <= const Duration(seconds: 20) && left > Duration.zero) {
      _warned = true;
      HapticFeedback.heavyImpact();
      DoppelSfx.instance.play('lowtime');
    }
    final sec = left.inSeconds;
    if (left < const Duration(seconds: 10) && left > Duration.zero && sec != _lastTickSec) {
      _lastTickSec = sec;
      HapticFeedback.selectionClick();
      DoppelSfx.instance.play('tick');
    }
  }

  @override
  void dispose() {
    _tick?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final left = widget.clock.remaining(widget.side);
    final active = widget.clock.running && widget.clock.turn == widget.side;
    final low = left < const Duration(seconds: 20);
    final m = left.inMinutes, s = left.inSeconds % 60;
    final text = left < const Duration(seconds: 10)
        ? '$s.${(left.inMilliseconds % 1000) ~/ 100}'
        : '$m:${s.toString().padLeft(2, '0')}';
    return AnimatedContainer(
      duration: const Duration(milliseconds: 150),
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: active ? (low ? const Color(0xFFF43F5E) : const Color(0xFFF3F1EA)) : const Color(0x22FFFFFF),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Text(
        text,
        style: TextStyle(
          fontFeatures: const [FontFeature.tabularFigures()],
          fontWeight: FontWeight.w800,
          fontSize: 18,
          color: active ? const Color(0xFF0F1418) : const Color(0xFFF3F1EA),
        ),
      ),
    );
  }
}

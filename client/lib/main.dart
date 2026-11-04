import 'package:chessground/chessground.dart';
import 'package:dartchess/dartchess.dart';
import 'package:flutter/material.dart';

import 'board/board_surface.dart';
import 'board/classic_skin.dart';
import 'game/game_controller.dart';
import 'game/material.dart';
import 'game/opponent.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await preloadPieces();
  runApp(const DoppelApp());
}

/// Decode piece images before the first frame so the board never appears without pieces.
Future<void> preloadPieces() => ChessgroundImages.instance.loadAll(
      const ChessboardSettings().pieceAssets,
      devicePixelRatio: WidgetsBinding.instance.platformDispatcher.implicitView?.devicePixelRatio ?? 1.0,
    );

const skins = <BoardSkin>[ClassicSkin()];

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
  GameController? game;

  @override
  void initState() {
    super.initState();
    BackendOpponent.listBots().then((b) => setState(() {
          bots = b;
          if (b.isNotEmpty) botId = b.first.$1;
          _newGame();
        }), onError: (_) => _newGame());
  }

  Opponent _opponent() {
    for (final (id, name) in bots) {
      if (id == botId) return BackendOpponent(bot: id, label: name, elo: elo);
    }
    return RandomOpponent();
  }

  void _newGame() {
    game?.dispose();
    setState(() => game = GameController(opponent: _opponent(), humanSide: humanSide));
  }

  @override
  Widget build(BuildContext context) {
    final g = game;
    return Scaffold(
      appBar: AppBar(title: const Text('Doppel'), actions: [
        IconButton(icon: const Icon(Icons.tune), onPressed: _settings),
        IconButton(icon: const Icon(Icons.add), tooltip: 'New game', onPressed: _newGame),
      ]),
      body: g == null
          ? const Center(child: CircularProgressIndicator())
          : ListenableBuilder(listenable: g, builder: (context, _) => _game(context, g)),
    );
  }

  Widget _game(BuildContext context, GameController g) {
    final width = MediaQuery.sizeOf(context).width;
    final mat = MaterialCount(g.position.board);
    return Column(children: [
      _PlayerBar(name: g.opponent.name, side: g.humanSide.opposite, material: mat, thinking: g.thinking),
      skin.buildBoard(
        view: BoardView(
          position: g.position,
          orientation: g.humanSide,
          interactive: g.over ? null : g.humanSide,
          lastMove: g.lastMove,
        ),
        size: width,
        onMove: g.play,
      ),
      _PlayerBar(name: 'You', side: g.humanSide, material: mat),
      if (g.error != null)
        ListTile(
          leading: const Icon(Icons.wifi_off),
          title: const Text('Opponent unreachable'),
          subtitle: Text(g.error!, maxLines: 2, overflow: TextOverflow.ellipsis),
          trailing: TextButton(onPressed: g.retry, child: const Text('Retry')),
        ),
      if (g.resultText != null)
        Padding(
          padding: const EdgeInsets.all(16),
          child: FilledButton.icon(
              onPressed: _newGame, icon: const Icon(Icons.replay), label: Text('${g.resultText} · Play again')),
        ),
      Expanded(
        child: SingleChildScrollView(
          reverse: true,
          padding: const EdgeInsets.all(12),
          child: Text(_moveList(g.sans), style: const TextStyle(fontFamily: 'monospace')),
        ),
      ),
    ]);
  }

  static String _moveList(List<String> sans) => [
        for (var i = 0; i < sans.length; i += 2) '${i ~/ 2 + 1}. ${sans[i]} ${i + 1 < sans.length ? sans[i + 1] : ''}'
      ].join('  ');

  Future<void> _settings() async {
    await showModalBottomSheet(
      context: context,
      builder: (context) => StatefulBuilder(
        builder: (context, set) => Padding(
          padding: const EdgeInsets.all(16),
          child: Column(mainAxisSize: MainAxisSize.min, crossAxisAlignment: CrossAxisAlignment.start, children: [
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
                value: elo.toDouble(), min: 600, max: 2600, divisions: 20, onChanged: (v) => set(() => elo = v.round())),
            SegmentedButton<Side>(
              segments: const [
                ButtonSegment(value: Side.white, label: Text('White')),
                ButtonSegment(value: Side.black, label: Text('Black')),
              ],
              selected: {humanSide},
              onSelectionChanged: (s) => set(() => humanSide = s.first),
            ),
            const SizedBox(height: 8),
            DropdownButton<String>(
              value: skin.id,
              isExpanded: true,
              items: [for (final s in skins) DropdownMenuItem(value: s.id, child: Text('Skin: ${s.name}'))],
              onChanged: (v) => set(() => skin = skins.firstWhere((s) => s.id == v)),
            ),
            const SizedBox(height: 8),
            FilledButton(
                onPressed: () {
                  Navigator.pop(context);
                  _newGame();
                },
                child: const Text('Start new game')),
          ]),
        ),
      ),
    );
  }
}

class _PlayerBar extends StatelessWidget {
  const _PlayerBar({required this.name, required this.side, required this.material, this.thinking = false});

  final String name;
  final Side side;
  final MaterialCount material;
  final bool thinking;

  static const _glyph = {Role.queen: '♛', Role.rook: '♜', Role.bishop: '♝', Role.knight: '♞', Role.pawn: '♟'};

  @override
  Widget build(BuildContext context) {
    final lead = material.lead(side);
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      child: Row(children: [
        Text(name, style: Theme.of(context).textTheme.titleMedium),
        const SizedBox(width: 8),
        if (thinking) const SizedBox(width: 14, height: 14, child: CircularProgressIndicator(strokeWidth: 2)),
        const Spacer(),
        Text(material.captured[side]!.map((r) => _glyph[r]).join(), style: const TextStyle(fontSize: 18)),
        if (lead > 0) Text(' +$lead', style: const TextStyle(fontWeight: FontWeight.bold)),
      ]),
    );
  }
}

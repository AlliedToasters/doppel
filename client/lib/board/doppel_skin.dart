import 'dart:math' as math;

import 'package:audioplayers/audioplayers.dart';
import 'package:chessground/chessground.dart';
import 'package:dartchess/dartchess.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../game/material.dart';
import 'board_surface.dart';
import 'chessground_board.dart';

/// Palette: dark squares are the screen background; light squares a lighter slate. See docs/doppel_skin.md.
abstract final class DoppelColors {
  static const bg = Color(0xFF1A2329);
  static const light = Color(0xFF2C3840);
  static const ink = Color(0xFFF3F1EA);
  static const ahead = Color(0xFF34D399);
  static const behind = Color(0xFFF43F5E);
  static const even = Color(0xFF5B6B76);
  static const accent = Color(0xFFFFC53D);
}

const _pieces = 'assets/pieces/doppel';
final PieceAssets doppelPieceAssets = {
  for (final kind in PieceKind.values)
    kind: AssetImage('$_pieces/${kind.side == Side.white ? 'w' : 'b'}${kind.role.letter.toUpperCase()}.png'),
};

const _scheme = ChessboardColorScheme(
  lightSquare: DoppelColors.light,
  darkSquare: DoppelColors.bg,
  background: SolidColorChessboardBackground(lightSquare: DoppelColors.light, darkSquare: DoppelColors.bg),
  whiteCoordBackground: SolidColorChessboardBackground(
      lightSquare: DoppelColors.light, darkSquare: DoppelColors.bg, coordinates: true),
  blackCoordBackground: SolidColorChessboardBackground(
      lightSquare: DoppelColors.light, darkSquare: DoppelColors.bg, coordinates: true, orientation: Side.black),
  lastMove: HighlightDetails(solidColor: Color(0x55FFC53D)),
  selected: HighlightDetails(solidColor: Color(0x8034D399)),
  validMoves: Color(0x6634D399),
  validPremoves: Color(0x664F8CFF),
);

class DoppelSkin extends BoardSkin {
  const DoppelSkin();

  @override
  String get id => 'doppel';
  @override
  String get name => 'Doppel';
  @override
  Color get background => DoppelColors.bg;

  @override
  Widget buildBoard({required BoardView view, required double size, required ValueChanged<Move> onMove}) =>
      _DoppelBoard(view: view, size: size, onMove: onMove);

  @override
  Widget buildMaterial(BuildContext context, MaterialCount material, Side side, {required bool isHuman}) {
    final caps = material.captured[side]!;
    return Row(mainAxisSize: MainAxisSize.min, children: [
      for (final r in caps)
        Padding(
          padding: const EdgeInsets.only(left: 1),
          child: Image(
              image: doppelPieceAssets[PieceKind.values.firstWhere((k) => k.side == side.opposite && k.role == r)]!,
              width: 20,
              height: 20),
        ),
      if (isHuman) ...[const SizedBox(width: 8), _MaterialPill(lead: material.lead(side))],
    ]);
  }
}

/// Points-of-material badge: green when ahead, red when behind; pops on every change.
class _MaterialPill extends StatelessWidget {
  const _MaterialPill({required this.lead});
  final int lead;

  @override
  Widget build(BuildContext context) {
    final color = lead > 0 ? DoppelColors.ahead : (lead < 0 ? DoppelColors.behind : DoppelColors.even);
    return AnimatedSwitcher(
      duration: const Duration(milliseconds: 260),
      switchInCurve: Curves.elasticOut,
      transitionBuilder: (child, a) => ScaleTransition(scale: a, child: child),
      child: AnimatedContainer(
        key: ValueKey(lead),
        duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
        decoration: BoxDecoration(
          color: color,
          borderRadius: BorderRadius.circular(14),
          boxShadow: [BoxShadow(color: color.withValues(alpha: 0.45), blurRadius: 10)],
        ),
        child: Text(lead > 0 ? '+$lead' : '$lead',
            style: const TextStyle(color: Color(0xFF0F1418), fontWeight: FontWeight.w900, fontSize: 16)),
      ),
    );
  }
}

class DoppelSfx {
  DoppelSfx._();
  static final instance = DoppelSfx._();

  /// Master switch (a mute setting later; off in widget tests, which have no audio plugin).
  static bool enabled = true;
  final _pools = <String, Future<AudioPool?>>{};

  Future<AudioPool?> _pool(String name) => _pools.putIfAbsent(
      name,
      () => AudioPool.create(source: AssetSource('sounds/$name.wav'), maxPlayers: 3)
          .then<AudioPool?>((p) => p)
          .catchError((Object _) => null)); // no audio (tests, missing plugin): stay silent

  Future<void> play(String name) async {
    if (!enabled) return;
    try {
      await (await _pool(name))?.start();
    } catch (_) {}
  }

  void warm() {
    if (!enabled) return;
    for (final n in ['move', 'capture', 'check', 'thunder', 'win']) {
      _pool(n);
    }
  }
}

class _DoppelBoard extends StatefulWidget {
  const _DoppelBoard({required this.view, required this.size, required this.onMove});
  final BoardView view;
  final double size;
  final ValueChanged<Move> onMove;

  @override
  State<_DoppelBoard> createState() => _DoppelBoardState();
}

class _DoppelBoardState extends State<_DoppelBoard> with SingleTickerProviderStateMixin {
  late final AnimationController _bolt =
      AnimationController(vsync: this, duration: const Duration(milliseconds: 900));
  Square? _boltSquare;
  int _boltSeed = 0;

  static final _settings = ChessboardSettings(
    colorScheme: _scheme,
    pieceAssets: doppelPieceAssets,
    enablePremoves: true,
    animationDuration: Duration(milliseconds: 120),
    dragFeedbackScale: 1.8,
    borderRadius: BorderRadius.all(Radius.circular(10)),
    pieceShiftMethod: PieceShiftMethod.either,
  );

  @override
  void initState() {
    super.initState();
    DoppelSfx.instance.warm();
  }

  @override
  void didUpdateWidget(_DoppelBoard old) {
    super.didUpdateWidget(old);
    final before = old.view.position, after = widget.view.position, move = widget.view.lastMove;
    if (before.fen == after.fen || move == null) return;
    _feedback(before, after, move);
  }

  /// Derive events from consecutive positions (docs/doppel_skin.md § Event hooks).
  void _feedback(Position before, Position after, Move move) {
    final victim = before.turn.opposite;
    final lost = <Role>[
      for (final r in Role.values)
        if ((after.board.materialCount(victim)[r] ?? 0) < (before.board.materialCount(victim)[r] ?? 0)) r,
    ];
    if (after.isCheckmate) {
      HapticFeedback.heavyImpact();
      DoppelSfx.instance.play('win');
    } else if (lost.contains(Role.queen)) {
      HapticFeedback.heavyImpact();
      DoppelSfx.instance.play('thunder');
      setState(() {
        _boltSquare = (move as NormalMove).to;
        _boltSeed++;
      });
      _bolt.forward(from: 0);
    } else if (after.isCheck) {
      HapticFeedback.mediumImpact();
      DoppelSfx.instance.play('check');
    } else if (lost.isNotEmpty) {
      HapticFeedback.mediumImpact();
      DoppelSfx.instance.play('capture');
    } else {
      HapticFeedback.selectionClick();
      DoppelSfx.instance.play('move');
    }
  }

  @override
  void dispose() {
    _bolt.dispose();
    super.dispose();
  }

  Offset _center(Square sq) {
    final s = widget.size / 8;
    final white = widget.view.orientation == Side.white;
    final f = white ? sq.file : 7 - sq.file, r = white ? 7 - sq.rank : sq.rank;
    return Offset((f + 0.5) * s, (r + 0.5) * s);
  }

  @override
  Widget build(BuildContext context) {
    final board = ChessgroundBoard(view: widget.view, size: widget.size, onMove: widget.onMove, settings: _settings);
    return AnimatedBuilder(
      animation: _bolt,
      child: board,
      builder: (context, child) {
        final t = _bolt.value;
        final active = _bolt.isAnimating && _boltSquare != null;
        final shake = active && t < 0.45 ? math.sin(t * 90) * 7 * (1 - t / 0.45) : 0.0;
        return Transform.translate(
          offset: Offset(shake, -shake * 0.4),
          child: Stack(clipBehavior: Clip.none, children: [
            child!,
            if (active)
              IgnorePointer(
                child: CustomPaint(
                  size: Size.square(widget.size),
                  painter: _LightningPainter(t: t, target: _center(_boltSquare!), seed: _boltSeed),
                ),
              ),
          ]),
        );
      },
    );
  }
}

/// A jagged bolt from above the board onto the capture square, a white flash, and a fading scorch ring.
class _LightningPainter extends CustomPainter {
  _LightningPainter({required this.t, required this.target, required this.seed});
  final double t;
  final Offset target;
  final int seed;

  @override
  void paint(Canvas canvas, Size size) {
    final flash = (1 - t / 0.25).clamp(0.0, 1.0);
    if (flash > 0) {
      canvas.drawRect(Offset.zero & size, Paint()..color = Colors.white.withValues(alpha: 0.55 * flash));
    }
    // Bolt flickers twice during the first ~40%.
    final visible = t < 0.14 || (t > 0.2 && t < 0.4);
    if (visible) {
      final rng = math.Random(seed * 7 + (t < 0.14 ? 0 : 1));
      final path = Path()..moveTo(target.dx + (rng.nextDouble() - 0.5) * size.width * 0.3, -size.height * 0.1);
      const steps = 9;
      final start = path.getBounds().topLeft;
      for (var i = 1; i <= steps; i++) {
        final k = i / steps;
        final x = start.dx + (target.dx - start.dx) * k + (i < steps ? (rng.nextDouble() - 0.5) * 38 : 0);
        final y = start.dy + (target.dy - start.dy) * k;
        path.lineTo(x, y);
      }
      final glow = Paint()
        ..color = DoppelColors.accent.withValues(alpha: 0.8)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 16
        ..strokeJoin = StrokeJoin.round
        ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 10);
      final core = Paint()
        ..color = Colors.white
        ..style = PaintingStyle.stroke
        ..strokeWidth = 5
        ..strokeJoin = StrokeJoin.round;
      canvas.drawPath(path, glow);
      canvas.drawPath(path, core);
    }
    // Impact ring expanding from the square.
    if (t > 0.05) {
      final k = ((t - 0.05) / 0.95).clamp(0.0, 1.0);
      canvas.drawCircle(
          target,
          size.width / 8 * (0.4 + 1.6 * Curves.easeOut.transform(k)),
          Paint()
            ..color = DoppelColors.accent.withValues(alpha: (1 - k) * 0.9)
            ..style = PaintingStyle.stroke
            ..strokeWidth = 6 * (1 - k) + 1);
    }
  }

  @override
  bool shouldRepaint(_LightningPainter old) => old.t != t || old.target != target;
}

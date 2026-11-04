import 'package:dartchess/dartchess.dart';

/// Material points from piece counts on the board. Pure position arithmetic, no model or engine,
/// so it is allowed in every mode, including Lichess-hosted games.
const pieceValue = {Role.pawn: 1, Role.knight: 3, Role.bishop: 3, Role.rook: 5, Role.queen: 9, Role.king: 0};
const _start = {Role.pawn: 8, Role.knight: 2, Role.bishop: 2, Role.rook: 2, Role.queen: 1, Role.king: 1};

class MaterialCount {
  MaterialCount(Board board)
      : points = {for (final s in Side.values) s: _points(board.materialCount(s))},
        captured = {for (final s in Side.values) s: _captured(board.materialCount(s.opposite))};

  /// Points on the board per side.
  final Map<Side, int> points;

  /// Opponent pieces each side has captured (promotions can make this undercount).
  final Map<Side, List<Role>> captured;

  /// Positive when [side] is ahead.
  int lead(Side side) => points[side]! - points[side.opposite]!;

  static int _points(Map<Role, int> c) => c.entries.fold(0, (a, e) => a + pieceValue[e.key]! * e.value);

  static List<Role> _captured(Map<Role, int> remaining) => [
        for (final r in [Role.queen, Role.rook, Role.bishop, Role.knight, Role.pawn])
          for (var i = 0; i < (_start[r]! - (remaining[r] ?? 0)).clamp(0, 8); i++) r,
      ];
}

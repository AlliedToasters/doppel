import 'dart:convert';

import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter_web_auth_2/flutter_web_auth_2.dart';
import 'package:http/http.dart' as http;
import 'package:url_launcher/url_launcher.dart';

import '../config.dart';
import 'server_game.dart' show installId;
import 'token_store/token_store.dart';

/// The device's anonymous Doppel account: a bearer token from POST /api/users, kept by [readToken].
/// Signing up passes the install id, so games this device played before accounts move to the account.
class Account {
  static String? token;

  static Map<String, String> get headers => {
        'content-type': 'application/json',
        if (token != null) 'authorization': 'Bearer $token',
      };

  /// Load the saved token, or create an account. Returns null if the backend is unreachable.
  static Future<String?> ensure() async {
    token ??= await readToken();
    if (token != null) return token;
    final id = await installId();
    var r = await _signup({'install_id': id});
    if (r.statusCode == 409) r = await _signup({}); // install id already claimed and its token lost
    if (r.statusCode != 200) return null;
    token = (jsonDecode(r.body) as Map<String, dynamic>)['token'] as String;
    await writeToken(token!);
    return token;
  }

  /// Link this account to a Lichess login (identity only; the backend keeps no Lichess token). The backend
  /// runs the OAuth flow and hands back a one-time login code. On Android this waits for the redirect and
  /// returns the account; on web the page navigates away and [finishWebLink] completes it after the reload.
  /// This is a different login from features/lichess (board:play, token on the device): that one never
  /// touches the backend.
  static Future<Map<String, dynamic>?> linkLichess() async {
    await ensure();
    if (kIsWeb) {
      await launchUrl(Uri.parse('$backendUrl/api/auth/lichess/start?client=web'), webOnlyWindowName: '_self');
      return null;
    }
    final result = await FlutterWebAuth2.authenticate(
      url: '$backendUrl/api/auth/lichess/start?client=app',
      callbackUrlScheme: 'app.doppelchess.doppel',
    );
    return _finish(Uri.parse(result).queryParameters);
  }

  /// Web only: complete a link when the page loads with ?login_code= (or ?login_error=). Null when absent.
  static Future<Map<String, dynamic>?> finishWebLink() async {
    if (!kIsWeb) return null;
    final q = Uri.base.queryParameters;
    if (!q.containsKey('login_code') && !q.containsKey('login_error')) return null;
    return _finish(q);
  }

  static Future<Map<String, dynamic>> _finish(Map<String, String> q) async {
    if (q['login_error'] != null) throw Exception('Lichess sign-in failed (${q['login_error']})');
    await ensure();
    final r = await http
        .post(Uri.parse('$backendUrl/api/auth/lichess/exchange'),
            headers: headers, body: jsonEncode({'code': q['login_code']}))
        .timeout(const Duration(seconds: 10));
    if (r.statusCode == 409) throw Exception('This device is linked to another Lichess account. Sign out first.');
    if (r.statusCode != 200) throw Exception('backend ${r.statusCode}');
    final d = jsonDecode(r.body) as Map<String, dynamic>;
    token = d['token'] as String;
    await writeToken(token!);
    return d['user'] as Map<String, dynamic>;
  }

  /// Forget this device's token and start a fresh anonymous account. A Lichess-linked account stays on the
  /// server and comes back by linking again.
  static Future<void> signOut() async {
    token = null;
    await clearToken();
    await ensure();
  }

  static Future<http.Response> _signup(Map<String, dynamic> body) => http
      .post(Uri.parse('$backendUrl/api/users'), headers: headers, body: jsonEncode(body))
      .timeout(const Duration(seconds: 5));

  static Future<Map<String, dynamic>> me() async {
    final r = await http.get(Uri.parse('$backendUrl/api/me'), headers: headers).timeout(const Duration(seconds: 5));
    if (r.statusCode != 200) throw Exception('backend ${r.statusCode}');
    return jsonDecode(r.body) as Map<String, dynamic>;
  }

  /// (version, text) of the current training consent.
  static Future<(String, String)> consentText() async {
    final r = await http.get(Uri.parse('$backendUrl/api/consent')).timeout(const Duration(seconds: 5));
    final d = jsonDecode(r.body) as Map<String, dynamic>;
    return (d['version'] as String, d['training'] as String);
  }

  /// Turning it on declares 18+: callers confirm that with the user first.
  static Future<Map<String, dynamic>> setTrainingConsent(bool on, String version) async {
    final r = await http
        .put(Uri.parse('$backendUrl/api/me/consent'),
            headers: headers, body: jsonEncode({'training': on, 'version': version, 'adult': on}))
        .timeout(const Duration(seconds: 5));
    if (r.statusCode != 200) throw Exception('backend ${r.statusCode}: ${r.body}');
    return jsonDecode(r.body) as Map<String, dynamic>;
  }
}

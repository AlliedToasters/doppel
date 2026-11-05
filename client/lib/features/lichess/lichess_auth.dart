import 'dart:convert';
import 'dart:math';

import 'package:crypto/crypto.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:flutter_web_auth_2/flutter_web_auth_2.dart';
import 'package:http/http.dart' as http;

/// Lichess sign-in: OAuth2 authorization code + PKCE, run entirely on the device.
///
/// Hard boundary: this feature talks to lichess.org only. It must not import the Doppel backend or
/// inference modules (test/lichess_boundary_test.dart fails if it does), and the token never leaves the
/// device. Lichess needs no client registration: client_id is any stable identifier.
class LichessAuth {
  static const host = 'https://lichess.org';
  static const clientId = 'app.doppelchess.doppel';
  static const callbackScheme = 'app.doppelchess.doppel';
  static const redirectUri = '$callbackScheme://lichess-callback';
  static const scopes = ['board:play'];

  static const _storage = FlutterSecureStorage();
  static const _tokenKey = 'lichess_token';
  static const _userKey = 'lichess_user';

  static String? token;
  static String? username;

  /// Load a saved session (call once at boot).
  static Future<void> load() async {
    token = await _storage.read(key: _tokenKey);
    username = await _storage.read(key: _userKey);
  }

  static bool get signedIn => token != null;

  static Map<String, String> get headers => {if (token != null) 'authorization': 'Bearer $token'};

  /// Opens lichess.org in a browser tab, waits for the redirect, exchanges the code. Returns the username.
  /// Throws if the user cancels or Lichess refuses.
  static Future<String> signIn() async {
    final verifier = _random(64);
    final state = _random(16);
    final challenge = base64Url.encode(sha256.convert(ascii.encode(verifier)).bytes).replaceAll('=', '');
    final url = Uri.parse('$host/oauth').replace(
      queryParameters: {
        'response_type': 'code',
        'client_id': clientId,
        'redirect_uri': redirectUri,
        'scope': scopes.join(' '),
        'code_challenge_method': 'S256',
        'code_challenge': challenge,
        'state': state,
      },
    );
    final result = await FlutterWebAuth2.authenticate(url: url.toString(), callbackUrlScheme: callbackScheme);
    final q = Uri.parse(result).queryParameters;
    if (q['state'] != state) throw Exception('state mismatch');
    if (q['error'] != null) throw Exception('lichess: ${q['error']}');
    final r = await http
        .post(
          Uri.parse('$host/api/token'),
          body: {
            'grant_type': 'authorization_code',
            'code': q['code'] ?? '',
            'code_verifier': verifier,
            'redirect_uri': redirectUri,
            'client_id': clientId,
          },
        )
        .timeout(const Duration(seconds: 10));
    if (r.statusCode != 200) throw Exception('token exchange ${r.statusCode}: ${r.body}');
    token = (jsonDecode(r.body) as Map<String, dynamic>)['access_token'] as String;
    final acct = await account();
    username = acct['username'] as String;
    await _storage.write(key: _tokenKey, value: token);
    await _storage.write(key: _userKey, value: username);
    return username!;
  }

  /// GET /api/account: username, perfs (ratings per speed), etc.
  static Future<Map<String, dynamic>> account() async {
    final r = await http.get(Uri.parse('$host/api/account'), headers: headers).timeout(const Duration(seconds: 10));
    if (r.statusCode == 401) {
      await _clear();
      throw Exception('lichess token expired or revoked');
    }
    if (r.statusCode != 200) throw Exception('lichess ${r.statusCode}');
    return jsonDecode(r.body) as Map<String, dynamic>;
  }

  /// Revoke the token at Lichess and forget it locally.
  static Future<void> signOut() async {
    try {
      await http.delete(Uri.parse('$host/api/token'), headers: headers).timeout(const Duration(seconds: 5));
    } catch (_) {}
    await _clear();
  }

  static Future<void> _clear() async {
    token = null;
    username = null;
    await _storage.delete(key: _tokenKey);
    await _storage.delete(key: _userKey);
  }

  static String _random(int n) {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~';
    final rnd = Random.secure();
    return List.generate(n, (_) => chars[rnd.nextInt(chars.length)]).join();
  }
}

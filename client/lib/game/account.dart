import 'dart:convert';

import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

import '../config.dart';
import 'server_game.dart' show installId;

/// The device's anonymous Doppel account: a bearer token from POST /api/users, kept in prefs.
/// Signing up passes the install id, so games this device played before accounts move to the account.
class Account {
  static String? token;
  static const _key = 'auth_token';

  static Map<String, String> get headers => {
        'content-type': 'application/json',
        if (token != null) 'authorization': 'Bearer $token',
      };

  /// Load the saved token, or create an account. Returns null if the backend is unreachable.
  static Future<String?> ensure() async {
    final prefs = await SharedPreferences.getInstance();
    token ??= prefs.getString(_key);
    if (token != null) return token;
    final id = await installId();
    var r = await _signup({'install_id': id});
    if (r.statusCode == 409) r = await _signup({}); // install id already claimed and its token lost
    if (r.statusCode != 200) return null;
    token = (jsonDecode(r.body) as Map<String, dynamic>)['token'] as String;
    await prefs.setString(_key, token!);
    return token;
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

  static Future<Map<String, dynamic>> setTrainingConsent(bool on, String version) async {
    final r = await http
        .put(Uri.parse('$backendUrl/api/me/consent'),
            headers: headers, body: jsonEncode({'training': on, 'version': version}))
        .timeout(const Duration(seconds: 5));
    if (r.statusCode != 200) throw Exception('backend ${r.statusCode}: ${r.body}');
    return jsonDecode(r.body) as Map<String, dynamic>;
  }
}

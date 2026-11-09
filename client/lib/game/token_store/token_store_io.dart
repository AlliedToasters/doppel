import 'package:shared_preferences/shared_preferences.dart';

const _key = 'auth_token';

Future<String?> readToken() async => (await SharedPreferences.getInstance()).getString(_key);

Future<void> writeToken(String token) async => (await SharedPreferences.getInstance()).setString(_key, token);

Future<void> clearToken() async => (await SharedPreferences.getInstance()).remove(_key);

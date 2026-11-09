import 'package:web/web.dart' as web;

/// Same key as site/src/pages/account.astro. Storage can be blocked (private mode): then the token lasts
/// for this page load only.
const _key = 'doppel_token';

Future<String?> readToken() async {
  try {
    return web.window.localStorage.getItem(_key);
  } catch (_) {
    return null;
  }
}

Future<void> writeToken(String token) async {
  try {
    web.window.localStorage.setItem(_key, token);
  } catch (_) {}
}

Future<void> clearToken() async {
  try {
    web.window.localStorage.removeItem(_key);
  } catch (_) {}
}

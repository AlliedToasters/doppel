// Where the account token lives. On web it is the site's own localStorage key, so /play and /account/ are
// one account (signing in with Lichess on /account/ carries over to /play). Elsewhere it is app prefs.
export 'token_store_io.dart' if (dart.library.js_interop) 'token_store_web.dart';

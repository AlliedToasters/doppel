import 'package:flutter/foundation.dart' show kIsWeb;

/// Build-time configuration: `--dart-define=BACKEND_URL=...` (tool/publish_apk.sh reads it from the untracked
/// client/backend_url.local). Without it, a web build talks to the origin that served it (doppelbot.org/play
/// -> doppelbot.org/api), and a desktop build to a local api.
const _defined = String.fromEnvironment('BACKEND_URL');

final String backendUrl = _defined.isNotEmpty ? _defined : (kIsWeb ? Uri.base.origin : 'http://127.0.0.1:8612');

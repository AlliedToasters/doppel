// Web entry point (flutter build web --wasm -t lib/web_entry.dart). dartchess keeps 64-bit bitboards, which
// dart2js can't represent, so only the WebAssembly build runs the app. Flutter always compiles a JS fallback
// too; that one gets a screen asking for a newer browser. dart:html exists under dart2js only, which makes it
// the switch. Native builds use lib/main.dart directly.
import 'main.dart' if (dart.library.html) 'no_wasm.dart' as app;

Future<void> main() async => app.main();

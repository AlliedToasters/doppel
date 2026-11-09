{{flutter_js}}
{{flutter_build_config}}

// The app only runs as WebAssembly (see lib/web_entry.dart), so let every engine with WasmGC have it rather than
// Flutter's default of Chromium only; the JS fallback is just an "update your browser" screen.
_flutter.loader.load({
  config: {
    wasmAllowList: { blink: true, gecko: true, webkit: true },
    verboseBuildSelection: true,
  },
  onEntrypointLoaded: async (engineInitializer) => {
    const app = await engineInitializer.initializeEngine();
    await app.runApp();
    document.getElementById('loading')?.remove();
  },
});

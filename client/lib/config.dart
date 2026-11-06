/// Build-time configuration: `--dart-define=BACKEND_URL=...` (tool/publish_apk.sh reads it from the untracked
/// client/backend_url.local). The default only works for a desktop/web build next to a local api.
const backendUrl = String.fromEnvironment(
  'BACKEND_URL',
  defaultValue: 'http://127.0.0.1:8612',
);

/// Build-time configuration. Override with `--dart-define=BACKEND_URL=...`.
const backendUrl = String.fromEnvironment(
  'BACKEND_URL',
  defaultValue: 'https://bigbox.<tailnet>.ts.net:8460',
);

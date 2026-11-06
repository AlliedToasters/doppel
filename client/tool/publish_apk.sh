#!/usr/bin/env bash
# Build the arm64 release APK and publish it as "latest" at $BACKEND_URL/doppel.apk
# (backend/api serves demo/static/ via DOPPEL_STATIC). Updates install over the old app because every build is signed
# with this box's ~/.android/debug.keystore; a different key would force an uninstall first.
set -euo pipefail
cd "$(dirname "$0")/.."
# The backend URL is deployment-specific and kept out of git: client/backend_url.local (one line) or $BACKEND_URL.
BACKEND_URL=${BACKEND_URL:-$(cat backend_url.local 2>/dev/null || true)}
[ -n "$BACKEND_URL" ] || { echo "set BACKEND_URL or write it to client/backend_url.local" >&2; exit 1; }
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 ANDROID_HOME=$HOME/opt/android-sdk PATH=$HOME/opt/flutter/bin:$PATH
flutter build apk --release --split-per-abi --build-number "$(date +%s)" --dart-define=BACKEND_URL="$BACKEND_URL" | grep -E "arm64|rror"
cp build/app/outputs/flutter-apk/app-arm64-v8a-release.apk ../demo/static/doppel.apk.tmp
mv ../demo/static/doppel.apk.tmp ../demo/static/doppel.apk
echo "published: $BACKEND_URL/doppel.apk"

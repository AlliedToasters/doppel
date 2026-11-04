#!/usr/bin/env bash
# Build the arm64 release APK and publish it as "latest" at https://bigbox.<tailnet>.ts.net:8460/doppel.apk
# (the demo server serves demo/static/). Updates install over the old app because every build is signed
# with this box's ~/.android/debug.keystore; a different key would force an uninstall first.
set -euo pipefail
cd "$(dirname "$0")/.."
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 ANDROID_HOME=$HOME/opt/android-sdk PATH=$HOME/opt/flutter/bin:$PATH
flutter build apk --release --split-per-abi --build-number "$(date +%s)" | grep -E "arm64|rror"
cp build/app/outputs/flutter-apk/app-arm64-v8a-release.apk ../demo/static/doppel.apk.tmp
mv ../demo/static/doppel.apk.tmp ../demo/static/doppel.apk
echo "published: https://bigbox.<tailnet>.ts.net:8460/doppel.apk"

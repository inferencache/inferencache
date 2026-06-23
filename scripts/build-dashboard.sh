#!/usr/bin/env bash
# Build the Next.js dashboard and embed static assets into the Python package.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROMPTCACHE_REPO="${PROMPTCACHE_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
DASHBOARD_REPO="${DASHBOARD_REPO:-$(cd "$PROMPTCACHE_REPO/../promptcache-dashboard" && pwd)}"
FRONTEND_DIR="$DASHBOARD_REPO/frontend-next"
OUT_DIR="$PROMPTCACHE_REPO/src/promptcache/proxy/dashboard"

if [ ! -d "$FRONTEND_DIR" ]; then
  echo "ERROR: frontend not found at $FRONTEND_DIR"
  echo "Set DASHBOARD_REPO to your promptcache-ui / promptcache-dashboard clone."
  exit 1
fi

echo "Building dashboard from $FRONTEND_DIR ..."
cd "$FRONTEND_DIR"
npm ci
npm run build

echo "Copying static export to $OUT_DIR ..."
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"
cp -r out/* "$OUT_DIR/"

echo "Done. Dashboard embedded at src/promptcache/proxy/dashboard/"

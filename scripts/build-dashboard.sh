#!/usr/bin/env bash
# Build the Next.js site and embed static assets into the Python package.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INFERENCACHE_REPO="${INFERENCACHE_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
DASHBOARD_REPO="${DASHBOARD_REPO:-$(cd "$INFERENCACHE_REPO/../inferencache-dashboard" && pwd)}"
FRONTEND_DIR="$DASHBOARD_REPO/frontend-next"
OUT_DIR="$INFERENCACHE_REPO/src/inferencache/proxy/site"

if [ ! -d "$FRONTEND_DIR" ]; then
  echo "ERROR: frontend not found at $FRONTEND_DIR"
  echo "Set DASHBOARD_REPO to your inferencache-dashboard clone."
  exit 1
fi

echo "Building site from $FRONTEND_DIR ..."
cd "$FRONTEND_DIR"
npm ci
npm run build

echo "Copying static export to $OUT_DIR ..."
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"
cp -r out/* "$OUT_DIR/"

echo "Done. Site embedded at src/inferencache/proxy/site/"

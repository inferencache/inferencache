#!/usr/bin/env bash
# Create the embedded dashboard output dir (gitignored; may be empty until build-dashboard.sh runs).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INFERENCACHE_REPO="${INFERENCACHE_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
OUT_DIR="$INFERENCACHE_REPO/src/inferencache/proxy/site"

mkdir -p "$OUT_DIR"

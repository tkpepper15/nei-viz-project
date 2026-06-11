#!/usr/bin/env bash
# Start the Flask API and Next.js dev server together.
# Usage: ./dev.sh
# Ctrl-C kills both.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'

cleanup() {
  echo ""
  echo "Shutting down..."
  kill "$API_PID" "$NEXT_PID" 2>/dev/null || true
  wait "$API_PID" "$NEXT_PID" 2>/dev/null || true
}
trap cleanup INT TERM

# Flask API — prefix each line with [api]
(
  cd "$ROOT/pipeline"
  exec python backend_api.py 2>&1
) | sed "s/^/${YELLOW}[api]${NC} /" &
API_PID=$!

# Next.js dev server — prefix each line with [next]
(
  cd "$ROOT"
  exec env NODE_OPTIONS='--max-old-space-size=8192' npx next dev 2>&1
) | sed "s/^/${CYAN}[next]${NC} /" &
NEXT_PID=$!

echo "api  PID=$API_PID"
echo "next PID=$NEXT_PID"
echo "Press Ctrl-C to stop both."

wait "$API_PID" "$NEXT_PID"

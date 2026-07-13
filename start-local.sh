#!/usr/bin/env bash
# start-local.sh — quick-start script for macOS / Linux.
#
# What this does:
#   1. Builds + starts the FastConformer backend in Docker
#   2. Waits for it to be healthy
#   3. Opens the frontend page in your default browser
#
# Requirements:
#   - Docker installed and running
#   - This script in the repo root
#
# Usage:
#   ./start-local.sh
#   ./start-local.sh --no-open   # build + start but don't open browser

set -euo pipefail

OPEN_BROWSER=1
for arg in "$@"; do
  case $arg in
    --no-open) OPEN_BROWSER=0 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo
echo "============================================================"
echo "  Hifzapp - Local Quick Start"
echo "============================================================"
echo

# Check Docker is running
if ! docker info >/dev/null 2>&1; then
  echo "[ERROR] Docker is not running. Start Docker Desktop and try again."
  exit 1
fi

echo "[1/4] Building and starting the backend..."
echo "  (First time will take 5-10 minutes to download deps + model)"
echo

cd backend

# Check for HF token (used to download the gated Muno459 model)
if [ -f "hf_token.txt" ]; then
  echo "  Found hf_token.txt — will use the gated Muno459 model (best quality)"
else
  echo "  No hf_token.txt found — will fall back to the public mohammed model"
  echo "  (only works for Alafasy; other reciters won't match well)"
  echo "  To use the best model: see backend/hf_token.txt.example"
fi
echo

docker compose up -d --build
if [ $? -ne 0 ]; then
  echo "[ERROR] Docker compose failed. Check the output above."
  exit 1
fi
cd ..

echo
echo "[2/4] Waiting for the backend to be healthy (up to 90s)..."
echo

for i in {1..30}; do
  if curl -sf http://127.0.0.1:8080/healthz >/dev/null 2>&1; then
    echo
    echo "[3/4] Backend is healthy!"
    curl -s http://127.0.0.1:8080/healthz
    echo
    break
  fi
  if [ "$i" -eq 30 ]; then
    echo
    echo "[ERROR] Backend did not become healthy in 90s."
    echo "Check logs with:  cd backend && docker compose logs -f fastconformer"
    exit 1
  fi
  printf "  waiting... (%d/30)\n" "$i"
  sleep 3
done

if [ "$OPEN_BROWSER" -eq 1 ]; then
  echo
  echo "[4/4] Opening the frontend in your browser..."
  URL="http://localhost/index.html"
  case "$(uname -s)" in
    Darwin)  open "$SCRIPT_DIR/index.html" ;;
    Linux)   xdg-open "$SCRIPT_DIR/index.html" 2>/dev/null || echo "  Open this file manually: $SCRIPT_DIR/index.html" ;;
    *)       echo "  Open this file manually: $SCRIPT_DIR/index.html" ;;
  esac
else
  echo
  echo "[4/4] Skipping browser open (--no-open)"
fi

echo
echo "============================================================"
echo "  Ready!"
echo "============================================================"
echo
echo "Frontend:    $SCRIPT_DIR/index.html"
echo "Backend:     http://localhost:8080/healthz"
echo "WebSocket:   ws://localhost:8080/ws"
echo
echo "To see live logs:  cd backend && docker compose logs -f fastconformer"
echo "To stop:           cd backend && docker compose down"
echo

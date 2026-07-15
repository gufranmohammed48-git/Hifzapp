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
echo "  (First time will take 5-10 minutes for Python deps; model is mounted from your PC)"
echo

# Check that the model file exists at the expected location
MODEL_FILE="$HOME/Downloads/model/nemo/fastconformer-quran.nemo"
if [ ! -f "$MODEL_FILE" ]; then
  echo "[ERROR] Model file not found at: $MODEL_FILE"
  echo
  echo "To fix: download the model first with this command:"
  echo "  python -c \"from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Muno459/fastconformer-quran', filename='nemo/fastconformer-quran.nemo', local_dir='$HOME/Downloads/model')\""
  echo
  echo "Or download the public mohammed model:"
  echo "  python -c \"from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='mohammed/fastconformer-quran-ar', filename='phase3_full_finetune/phase3_full_finetune_wer0.1432.nemo', local_dir='$HOME/Downloads/model/nemo')\""
  echo
  echo "Then re-run this script."
  exit 1
fi
echo "  Found model at: $MODEL_FILE"
echo

cd backend

# Check for HF token (optional, not used since we're mounting the model)
if [ -f "hf_token.txt" ]; then
  echo "  Found hf_token.txt (not needed - using mounted model)"
else
  echo "  No hf_token.txt - using mounted model from host"
fi
echo

docker compose up -d --build
if [ $? -ne 0 ]; then
  echo "[ERROR] Docker compose failed. Check the output above."
  exit 1
fi
cd ..

# Verify the model file is accessible inside the container
echo
echo "Verifying model mount inside container..."
docker exec fastconformer ls -la /data/fastconformer-quran.nemo 2>&1 || true

echo
echo "[2/4] Waiting for the backend to be healthy (up to 240s)..."
echo

for i in {1..80}; do
  if curl -sf http://127.0.0.1:8080/healthz >/dev/null 2>&1; then
    echo
    echo "[3/4] Backend is healthy!"
    curl -s http://127.0.0.1:8080/healthz
    echo
    break
  fi
  if [ "$i" -eq 80 ]; then
    echo
    echo "[ERROR] Backend did not become healthy in 180s."
    echo "Common causes:"
    echo "  1. Model is still downloading (459MB, can take a few min)"
    echo "  2. Model is loading on slow CPU (60-90s on Intel UHD)"
    echo "  3. HF_TOKEN invalid - check hf_token.txt"
    echo
    echo "Check logs with:  cd backend && docker compose logs -f fastconformer"
    exit 1
  fi
  printf "  waiting... (%d/80)\n" "$i"
  sleep 3
done

if [ "$OPEN_BROWSER" -eq 1 ]; then
  echo
  echo "[4/4] Opening the frontend in your browser..."
  URL="http://localhost:8080/"
  case "$(uname -s)" in
    Darwin)  open "$URL" ;;
    Linux)   xdg-open "$URL" 2>/dev/null || echo "  Open this URL in your browser: $URL" ;;
    *)       echo "  Open this URL in your browser: $URL" ;;
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
echo "Frontend:    http://localhost:8080/"
echo "Backend:     http://localhost:8080/healthz"
echo "WebSocket:   ws://localhost:8080/ws"
echo
echo "To see live logs:  cd backend && docker compose logs -f fastconformer"
echo "To stop:           cd backend && docker compose down"
echo

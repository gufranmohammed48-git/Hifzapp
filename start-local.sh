#!/usr/bin/env bash
# ============================================================================
# start-local.sh — One-click local dev start for Hifzapp (macOS / Linux)
# ============================================================================

set -e

# Load .env if it exists
if [ -f backend/.env ]; then
    set -a
    # shellcheck disable=SC1091
    source backend/.env
    set +a
fi

MODEL_HOST_PATH="${MODEL_HOST_PATH:-$HOME/Documents/model}"

echo
echo "=== Hifzapp local dev start (Zipformer streaming ASR) ==="
echo

# Check model files
echo "Checking model files in: $MODEL_HOST_PATH"
if [ ! -f "$MODEL_HOST_PATH/quran_phoneme_zipformer.int8.onnx" ]; then
    echo
    echo "ERROR: model file not found"
    echo "  Expected: $MODEL_HOST_PATH/quran_phoneme_zipformer.int8.onnx"
    echo
    echo "Download from https://huggingface.co/Muno459/zipformer_p-quran"
    echo "and update MODEL_HOST_PATH in backend/.env"
    exit 1
fi
if [ ! -f "$MODEL_HOST_PATH/tokens.txt" ]; then
    echo
    echo "ERROR: tokens.txt not found"
    echo "  Expected: $MODEL_HOST_PATH/tokens.txt"
    exit 1
fi
echo "  ✓ quran_phoneme_zipformer.int8.onnx OK"
echo "  ✓ tokens.txt OK"

# Clean up old container
echo
echo "Cleaning up old container..."
docker rm -f zipformer-quran 2>/dev/null || true

# Build and start
echo
echo "Building and starting container (this may take 2-5 minutes first time)..."
cd backend
docker compose --env-file .env up -d --build
cd ..

# Wait for healthy
echo
echo "Waiting for backend to be ready..."
attempt=0
max_attempts=60
while [ $attempt -lt $max_attempts ]; do
    attempt=$((attempt + 1))
    status=$(docker inspect --format="{{.State.Health.Status}}" zipformer-quran 2>/dev/null || echo "starting")
    if [ "$status" = "healthy" ]; then
        break
    fi
    if [ $attempt -eq 1 ]; then
        echo "  Loading model... (this can take 30-90s on first run)"
    fi
    echo "  Still loading... attempt $attempt/$max_attempts"
    sleep 5
done

if [ "$status" != "healthy" ]; then
    echo
    echo "ERROR: backend did not become healthy in 5 minutes"
    echo
    echo "Last 30 lines of logs:"
    docker logs zipformer-quran --tail 30
    exit 1
fi

echo
echo "=== Backend is healthy ==="
echo
echo "  Local:   http://localhost:8080"
echo "  Health:  http://localhost:8080/healthz"
echo
echo "To expose via HTTPS tunnel (access from phone/anywhere):"
echo "  cloudflared tunnel --url http://localhost:8080"
echo

# Open browser
if command -v open >/dev/null 2>&1; then
    open "http://localhost:8080"
elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "http://localhost:8080"
fi

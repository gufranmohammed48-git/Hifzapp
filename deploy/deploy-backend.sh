#!/usr/bin/env bash
# deploy-backend.sh — rebuild + restart the FastConformer Docker container on the DO droplet.
#
# Use this when:
#   - You change app.py
#   - You change requirements.txt
#   - You change the model
#   - You change Dockerfile
#
# For frontend-only changes, use deploy-frontend.sh instead (much faster).
#
# Usage:
#   ./deploy/deploy-backend.sh [tag]
#   ./deploy/deploy-backend.sh v1.5.0
#
# Env vars:
#   DROPLET_IP  (default: 167.99.116.58)
#   SSH_KEY     (default: /workspace/.ssh/quran-deploy)
#   REPO_DIR    (default: /opt/quran/Hifzapp)
#   IMAGE_TAG   (default: latest, or first arg)

set -euo pipefail

DROPLET_IP="${DROPLET_IP:-167.99.116.58}"
SSH_KEY="${SSH_KEY:-/workspace/.ssh/quran-deploy}"
REPO_DIR="${REPO_DIR:-/opt/quran/Hifzapp}"
IMAGE_TAG="${1:-latest}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Deploying backend to $DROPLET_IP ==="
echo "  Image tag: $IMAGE_TAG"
echo "  Repo dir:  $REPO_DIR"
echo ""

# Step 1: Sync the repo to the droplet (only the backend/ folder + root files needed)
echo "[1/5] Syncing repo to droplet..."
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  "root@$DROPLET_IP" "mkdir -p $REPO_DIR"

rsync -avz --delete \
  -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='.venv' \
  --exclude='*.pyc' \
  --exclude='debug_chunks' \
  --exclude='*.wav' \
  --exclude='*.ogg' \
  --exclude='*.mp3' \
  "$REPO_ROOT/" "root@$DROPLET_IP:$REPO_DIR/"

# Step 2: Build the Docker image
echo "[2/5] Building Docker image..."
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  "root@$DROPLET_IP" "cd $REPO_DIR/backend && docker build -t fastconformer-quran:$IMAGE_TAG . && docker tag fastconformer-quran:$IMAGE_TAG fastconformer-quran:latest"

# Step 3: Stop the old container
echo "[3/5] Stopping old container..."
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  "root@$DROPLET_IP" "docker stop fastconformer 2>/dev/null || true && docker rm fastconformer 2>/dev/null || true"

# Step 4: Start the new container
echo "[4/5] Starting new container..."
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  "root@$DROPLET_IP" "cd $REPO_DIR/backend && docker run -d \
    --name fastconformer \
    --restart unless-stopped \
    -p 127.0.0.1:8080:8080 \
    -v fastconformer_model:/data \
    -v /opt/quran/web:/opt/quran/web:ro \
    -e MODEL_PATH=/data/fastconformer-quran.nemo \
    -e PYTHONUNBUFFERED=1 \
    -e PORT=8080 \
    --memory=3g \
    --cpus=1.5 \
    fastconformer-quran:$IMAGE_TAG"

# Step 5: Wait for healthcheck
echo "[5/5] Waiting for /healthz..."
for i in {1..30}; do
  if ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    "root@$DROPLET_IP" "curl -sf http://127.0.0.1:8080/healthz" >/dev/null 2>&1; then
    echo "  ✓ Backend is healthy after $((i * 2))s"
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      "root@$DROPLET_IP" "curl -s http://127.0.0.1:8080/healthz"
    echo ""
    echo "✓ Backend deployed"
    echo "  Health: http://$DROPLET_IP/healthz"
    echo "  WebSocket: ws://$DROPLET_IP/ws"
    exit 0
  fi
  sleep 2
done

echo "✗ Backend didn't become healthy in 60s"
echo "Check logs:"
echo "  ssh root@$DROPLET_IP 'docker logs fastconformer'"
exit 1

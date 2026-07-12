#!/usr/bin/env bash
# deploy-frontend.sh — copy frontend HTML + phonemes-data.js to the DO droplet.
#
# This is what runs after every change to the frontend in Hifzapp.
# It ONLY ships the static files (no backend, no Python, no rebuild).
#
# Usage:
#   ./deploy/deploy-frontend.sh                  # uses default IP + key
#   DROPLET_IP=1.2.3.4 ./deploy/deploy-frontend.sh
#   SSH_KEY=/path/to/key ./deploy/deploy-frontend.sh
#
# Env vars:
#   DROPLET_IP  (default: 167.99.116.58)
#   SSH_KEY     (default: /workspace/.ssh/quran-deploy)
#   REMOTE_DIR  (default: /opt/quran/web)

set -euo pipefail

DROPLET_IP="${DROPLET_IP:-167.99.116.58}"
SSH_KEY="${SSH_KEY:-/workspace/.ssh/quran-deploy}"
REMOTE_DIR="${REMOTE_DIR:-/opt/quran/web}"
CONTAINER="${CONTAINER:-fastconformer}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Files to ship (relative to repo root)
FILES=(
  "hafizAssist_streaming.html"
  "HafizAssist.html"
  "index.html"
  "phonemes-data.js"
)

echo "=== Deploying frontend to $DROPLET_IP:$REMOTE_DIR ==="
echo "Files: ${FILES[*]}"
echo ""

# Step 1: scp to host
echo "[1/3] Copying to host..."
for f in "${FILES[@]}"; do
  echo "  - $f"
  scp -i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    "$REPO_ROOT/$f" "root@$DROPLET_IP:$REMOTE_DIR/"
done

# Step 2: docker cp into container
echo "[2/3] Copying into $CONTAINER container..."
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  "root@$DROPLET_IP" "for f in ${FILES[*]}; do
    docker cp $REMOTE_DIR/\$f $CONTAINER:$REMOTE_DIR/\$f
  done"

# Step 3: Verify
echo "[3/3] Verifying..."
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  "root@$DROPLET_IP" "ls -la $REMOTE_DIR/ && echo '' && echo '--- version badge ---' && grep -o '1\.[0-9]\.[0-9][^<]*' $REMOTE_DIR/hafizAssist_streaming.html | head -1"

echo ""
echo "✓ Frontend deployed"
echo "  HTTP:  http://$DROPLET_IP/"
echo "  HTTPS: https://$DROPLET_IP/ (self-signed) or via cloudflared tunnel"

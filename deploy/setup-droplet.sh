#!/usr/bin/env bash
# setup-droplet.sh — first-time setup for a fresh DO droplet.
#
# Run this ONCE on a new Ubuntu 22.04 droplet to install Docker,
# nginx, copy the nginx config, and start the cloudflared tunnel.
#
# Usage:
#   ./deploy/setup-droplet.sh
#
# This is a one-time bootstrap. After it runs, use:
#   deploy-frontend.sh  for frontend-only changes
#   deploy-backend.sh   for backend changes (rebuilds Docker)

set -euo pipefail

if [ "$EUID" -ne 0 ]; then
  echo "Must run as root. Try: sudo $0"
  exit 1
fi

echo "=== Setting up Hifzapp droplet ==="

# Step 1: Install Docker
echo "[1/5] Installing Docker..."
if ! command -v docker &> /dev/null; then
  apt-get update
  apt-get install -y ca-certificates curl gnupg
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update
  apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  systemctl enable --now docker
fi
echo "  ✓ Docker $(docker --version)"

# Step 2: Install nginx
echo "[2/5] Installing nginx..."
if ! command -v nginx &> /dev/null; then
  apt-get install -y nginx
  systemctl enable --now nginx
fi
echo "  ✓ nginx $(nginx -v 2>&1)"

# Step 3: Generate self-signed cert (for HTTPS in dev)
echo "[3/5] Generating self-signed SSL cert..."
mkdir -p /etc/nginx/certs
if [ ! -f /etc/nginx/certs/server.crt ]; then
  openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /etc/nginx/certs/server.key \
    -out /etc/nginx/certs/server.crt \
    -subj "/CN=localhost" \
    -addext "subjectAltName=DNS:localhost,IP:$(curl -s ifconfig.me)" 2>/dev/null || \
  openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /etc/nginx/certs/server.key \
    -out /etc/nginx/certs/server.crt \
    -subj "/CN=localhost"
fi
echo "  ✓ Cert at /etc/nginx/certs/server.crt"

# Step 4: Install nginx config
echo "[4/5] Installing nginx config..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cp "$SCRIPT_DIR/nginx.conf" /etc/nginx/sites-available/hifzapp
ln -sf /etc/nginx/sites-available/hifzapp /etc/nginx/sites-enabled/hifzapp
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl reload nginx
echo "  ✓ nginx reloaded"

# Step 5: Create web dir for static files
mkdir -p /opt/quran/web
echo "  ✓ /opt/quran/web ready (frontend files will go here)"

# Step 6: Install cloudflared (optional but recommended for real HTTPS)
echo "[5/5] Installing cloudflared..."
if ! command -v cloudflared &> /dev/null; then
  curl -L --output /tmp/cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
  dpkg -i /tmp/cloudflared.deb
  echo "  ✓ cloudflared installed. Start with: cloudflared tunnel --url http://localhost:80"
else
  echo "  ✓ cloudflared already installed"
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. From your local machine, clone the Hifzapp repo"
echo "  2. Run deploy/deploy-backend.sh to build the FastConformer container"
echo "  3. Run deploy/deploy-frontend.sh to ship the static files"
echo "  4. Optional: run 'cloudflared tunnel --url http://localhost:80' for HTTPS"

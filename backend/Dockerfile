FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (cached separately from app code for faster rebuilds)
# Cython + build tools MUST come BEFORE nemo_toolkit (youtokentome needs Cython) because youtokentome
# (NeMo dep) needs to compile from source on Python 3.10.
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir cython setuptools wheel && \
    pip install --no-cache-dir --no-build-isolation -r requirements.txt && \
    pip uninstall -y pyarrow && \
    pip install --no-cache-dir 'pyarrow==13.0.0'

# Copy app code
COPY backend/app.py ./
COPY backend/download_model.py ./

# Copy the static frontend files (HTML, JS) so the FastAPI app can
# serve them at /. This is the standard pattern for local dev:
# one container, one port (8080) serves both the API and the UI.
# The DO production deployment uses nginx to serve these from
# /opt/quran/web instead (see deploy/nginx.conf).
COPY hafizAssist_streaming.html /app/static/
COPY HafizAssist.html /app/static/
COPY index.html /app/static/
COPY phonemes-data.js /app/static/

# Download the FastConformer model at build time (cached in image).
#
# Production model: Muno459/fastconformer-quran (4.13% WER overall,
# 0.93% WER on held-out unseen reciters, supports full Arabic voice
# diversity with diacritics). Was originally at
# shahabazkc10/fastconformer-quran-bucket, now hosted under Muno459.
# See https://huggingface.co/Muno459/fastconformer-quran
#
# NOTE: Muno459/fastconformer-quran is GATED. To download it you must:
#   1. Visit https://huggingface.co/Muno459/fastconformer-quran
#   2. Click "Agree and access repository" (accepts the license terms)
#   3. Create a read token at https://huggingface.co/settings/tokens
#   4. Put it in backend/hf_token.txt before running start-local.bat
#
# If no HF_TOKEN is set, the build falls back to the public (but
# Alafasy-biased) mohammed/fastconformer-quran-ar model.
#
# Override at runtime with MODEL_PATH env var.
ARG MODEL_REPO=Muno459/fastconformer-quran
ARG MODEL_FILENAME=nemo/fastconformer-quran.nemo
ARG FALLBACK_REPO=mohammed/fastconformer-quran-ar
ARG FALLBACK_FILENAME=phase3_full_finetune/phase3_full_finetune_wer0.1432.nemo
# Set SKIP_DOWNLOAD=1 to skip the HF download (use when mounting the
# model from the host instead — recommended for local dev).
ARG SKIP_DOWNLOAD=0

# Download at build time. Uses HF_TOKEN if present (for gated models).
# Falls back to the public mohammed model if the gated one is inaccessible.
# Set SKIP_DOWNLOAD=1 to skip this entirely (when the model is mounted
# from the host via a volume — see docker-compose.yml).
# Note: --mount=type=secret must be on the RUN itself (Dockerfile syntax).
RUN --mount=type=secret,id=hf_token \
    if [ "$SKIP_DOWNLOAD" = "1" ]; then \
      echo "SKIP_DOWNLOAD=1: skipping model download. Mount model at /data via docker-compose."; \
    else \
      if [ -f /run/secrets/hf_token ]; then export HF_TOKEN="$(cat /run/secrets/hf_token)"; fi \
      && python /app/download_model.py; \
    fi
ENV MODEL_REPO=${MODEL_REPO} \
    MODEL_FILENAME=${MODEL_FILENAME} \
    FALLBACK_REPO=${FALLBACK_REPO} \
    FALLBACK_FILENAME=${FALLBACK_FILENAME}

# Default model path (can be overridden by MODEL_PATH env var at runtime)
ENV MODEL_PATH=/data/fastconformer-quran.nemo \
    PYTHONUNBUFFERED=1 \
    PORT=8080

EXPOSE 8080

# Use shell form to expand $PORT (works for both Railway and docker-compose)
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1 --log-level info"]

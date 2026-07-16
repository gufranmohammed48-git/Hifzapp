# Dockerfile for Zipformer Quran ASR
#
# Image is much smaller than the previous NeMo-based one (~500MB vs ~3GB):
# - sherpa-onnx is ~50MB (vs nemo_toolkit at ~2GB)
# - No Cython compilation, no CUDA deps
# - CPU-only by default (Intel UHD friendly)

FROM python:3.10-slim

WORKDIR /app

# System deps (kept minimal — sherpa-onnx bundles most of what it needs)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy backend code
COPY backend/app.py /app/app.py
COPY backend/download_model.py /app/download_model.py

# Copy frontend files (static-served by the middleware)
COPY hafizAssist_streaming.html /app/static/hafizAssist_streaming.html
COPY HafizAssist.html            /app/static/HafizAssist.html
COPY index.html                  /app/static/index.html
COPY phonemes-data.js            /app/static/phonemes-data.js

# Default config — overridden by docker-compose.yml env block
ENV MODEL_PATH=/data/quran_phoneme_zipformer.int8.onnx \
    TOKENS_PATH=/data/tokens.txt \
    NUM_THREADS=2 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

EXPOSE 8080

# Health check — confirm uvicorn is up
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/healthz', timeout=3).read()" \
    || exit 1

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1 --log-level info"]

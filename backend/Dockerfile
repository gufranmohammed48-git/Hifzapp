# Dockerfile for the int8 ONNX FastConformer Quran ASR.
#
# Image is much smaller than the previous NeMo-based one (~700MB vs ~3GB):
# - No nemo_toolkit (saves ~2GB)
# - No torch / torchaudio (saves ~1GB)
# - No Cython compilation, no CUDA deps
# - CPU-only by default (Intel UHD friendly)

FROM python:3.10-slim

WORKDIR /app

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

# Copy frontend files (static-served by the middleware)
COPY hafizAssist_streaming.html /app/static/
COPY HafizAssist.html            /app/static/
COPY index.html                  /app/static/
COPY phonemes-data.js            /app/static/

ENV MODEL_PATH=/data/model_with_encoder.q8.onnx \
    TOKENIZER_PATH=/data/tokenizer.model \
    NUM_THREADS=2 \
    WINDOW_SEC=1.0 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/healthz', timeout=3).read()" \
    || exit 1

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1 --log-level info"]

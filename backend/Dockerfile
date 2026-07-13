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
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir cython setuptools wheel && \
    pip install --no-cache-dir --no-build-isolation -r requirements.txt && \
    pip uninstall -y pyarrow && \
    pip install --no-cache-dir 'pyarrow==13.0.0'

# Copy app code
COPY app.py ./

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
#   4. Pass it via --build-arg HF_TOKEN=hf_xxx or set HF_TOKEN env var
#
# If no HF_TOKEN is set, the build falls back to the public (but
# Alafasy-biased) mohammed/fastconformer-quran-ar model.
#
# Override at runtime with MODEL_PATH env var.
ARG MODEL_REPO=Muno459/fastconformer-quran
ARG MODEL_FILENAME=nemo/fastconformer-quran.nemo
ARG FALLBACK_REPO=mohammed/fastconformer-quran-ar
ARG FALLBACK_FILENAME=phase3_full_finetune/phase3_full_finetune_wer0.1432.nemo

# Download at build time. Uses HF_TOKEN if present (for gated models).
# Falls back to the public mohammed model if the gated one is inaccessible.
RUN --mount=type=secret,id=hf_token,env=HF_TOKEN \
    python -c "
import os, sys
from huggingface_hub import hf_hub_download

repo = os.environ.get('MODEL_REPO', 'Muno459/fastconformer-quran')
filename = os.environ.get('MODEL_FILENAME', 'nemo/fastconformer-quran.nemo')
fallback_repo = os.environ.get('FALLBACK_REPO', 'mohammed/fastconformer-quran-ar')
fallback_filename = os.environ.get('FALLBACK_FILENAME', 'phase3_full_finetune/phase3_full_finetune_wer0.1432.nemo')
local_dir = '/data'

# Map filename -> final /data path
def dest_path(filename):
    return '/data/' + filename.split('/')[-1]

try:
    print(f'Downloading {repo}/{filename}...')
    path = hf_hub_download(repo_id=repo, filename=filename, local_dir=local_dir)
    final = dest_path(filename)
    if path != final:
        import shutil; shutil.move(path, final)
    print(f'  -> {final}')
except Exception as e:
    print(f'  Primary failed: {e}')
    print(f'Falling back to {fallback_repo}/{fallback_filename}...')
    path = hf_hub_download(repo_id=fallback_repo, filename=fallback_filename, local_dir=local_dir)
    final = dest_path(fallback_filename)
    if path != final:
        import shutil; shutil.move(path, final)
    print(f'  -> {final}')
" \
    && ls -la /data/*.nemo
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

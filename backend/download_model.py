#!/usr/bin/env python3
"""download_model.py — downloads the FastConformer model at Docker build time.

Used by the Dockerfile RUN step. Tries the primary (gated) model first,
falls back to the public one if the gated one is inaccessible.

Reads configuration from environment variables:
  MODEL_REPO         — primary HF repo (e.g. Muno459/fastconformer-quran)
  MODEL_FILENAME     — file in primary repo (e.g. nemo/fastconformer-quran.nemo)
  FALLBACK_REPO      — fallback HF repo (e.g. mohammed/fastconformer-quran-ar)
  FALLBACK_FILENAME  — file in fallback repo
  HF_TOKEN           — optional HF read token (needed for gated models)
"""
import os
import shutil
import sys

from huggingface_hub import hf_hub_download


REPO = os.environ.get('MODEL_REPO', 'Muno459/fastconformer-quran')
FILENAME = os.environ.get('MODEL_FILENAME', 'nemo/fastconformer-quran.nemo')
FALLBACK_REPO = os.environ.get('FALLBACK_REPO', 'mohammed/fastconformer-quran-ar')
FALLBACK_FILENAME = os.environ.get(
    'FALLBACK_FILENAME',
    'phase3_full_finetune/phase3_full_finetune_wer0.1432.nemo',
)
LOCAL_DIR = '/data'


def dest_path(filename: str) -> str:
    """Map a (possibly subpath) filename to its final /data destination."""
    return '/data/' + filename.split('/')[-1]


def try_download(repo: str, filename: str) -> str:
    """Download a file from HF, move it to its final /data path. Returns path."""
    print(f'Downloading {repo}/{filename}...')
    cached_path = hf_hub_download(
        repo_id=repo,
        filename=filename,
        local_dir=LOCAL_DIR,
    )
    final = dest_path(filename)
    if cached_path != final and os.path.exists(cached_path):
        os.makedirs(os.path.dirname(final) or '.', exist_ok=True)
        shutil.move(cached_path, final)
    return final


def main() -> int:
    try:
        path = try_download(REPO, FILENAME)
        print(f'  -> {path}')
        return 0
    except Exception as e:
        print(f'  Primary failed: {type(e).__name__}: {e}', file=sys.stderr)
        print(f'Falling back to {FALLBACK_REPO}/{FALLBACK_FILENAME}...', file=sys.stderr)
        try:
            path = try_download(FALLBACK_REPO, FALLBACK_FILENAME)
            print(f'  -> {path}')
            return 0
        except Exception as e2:
            print(f'  Fallback also failed: {type(e2).__name__}: {e2}', file=sys.stderr)
            return 1


if __name__ == '__main__':
    sys.exit(main())

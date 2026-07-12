# FastConformer Quran Arabic ASR — Backend

The Python backend for the [Hifzapp live Quran recitation tracker](../).
Streams audio in over WebSocket, runs it through the FastConformer model,
returns partial transcriptions.

## Quick start (local)

```bash
cd backend
docker compose up --build

# In another terminal:
curl http://127.0.0.1:8080/healthz
# → {"status":"ok","model":"fastconformer-quran-ar"}
```

The model (~459MB) is downloaded into a named volume on first build, so
subsequent restarts don't re-download.

## Production deploy

See [`../deploy/`](../deploy/) for the deploy scripts. The one-liner
from your local machine is:

```bash
./deploy/deploy-backend.sh
```

This rsyncs the repo to the droplet, rebuilds the Docker image, and
restarts the container with the right volume mounts + healthcheck.

## Model

[shahabazkc10/fastconformer-quran-bucket](https://huggingface.co/shahabazkc10/fastconformer-quran-bucket)

| Metric | Value |
|--------|-------|
| Overall WER | 4.13% |
| **WER on held-out unseen reciters** | **0.93%** |
| Includes diacritics in output | ✅ |
| Supported voices | All Arabic (not biased to one reciter) |

Override the model path at runtime with the `MODEL_PATH` env var.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/healthz` | Health check |
| WS | `/ws` | Streaming audio + partial transcripts |
| GET | `/api/debug-audio` | Download saved audio chunks as zip (debug) |

## WebSocket protocol

**Send audio**: raw 16-bit PCM, mono, 16kHz, little-endian binary frames.

**Send control** (JSON text frames):
- `{"type": "commit"}` — finalize current utterance
- `{"type": "finalize"}` — transcribe full buffer
- `{"type": "reset"}` — clear audio buffer

**Receive** (JSON text frames):
- `{"type": "partial", "text": "...", "full_text": "..."}` — incremental words
- `{"type": "committed", "text": "..."}` — committed text
- `{"type": "final", "text": "...", "words": [...]}` — final with word timestamps
- `{"type": "error", "message": "..."}` — error

## Environment variables

| Var | Default | Purpose |
|-----|---------|---------|
| `MODEL_PATH` | `/data/fastconformer-quran.nemo` | Override the model file |
| `PORT` | `8080` | Listening port |
| `PYTHONUNBUFFERED` | `1` | Force flush stdout for log streaming |

## File layout (in the monorepo)

```
backend/
├── app.py                  # FastAPI + WebSocket + ASR
├── Dockerfile              # Python 3.10-slim, downloads model
├── docker-compose.yml      # Local dev (named volume, healthcheck, limits)
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md               # This file
```

The frontend lives at the repo root (one level up). The full monorepo
layout is in [`../README.md`](../README.md).

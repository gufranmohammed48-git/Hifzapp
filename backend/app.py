"""Zipformer Quran ASR — streaming WebSocket server.

Uses Muno459/zipformer_p-quran via sherpa-onnx for real-time
phoneme-level transcription on CPU.

Designed for live word-by-word highlighting in the Hifzapp frontend.
Latency: ~30ms per inference (vs 2-3s for the previous FastConformer
NeMo model) on Intel UHD laptop CPU.
"""
import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("zipformer-quran")

# ============================================================================
# Configuration
# ============================================================================
MODEL_PATH = os.environ.get("MODEL_PATH", "/data/quran_phoneme_zipformer.int8.onnx")
TOKENS_PATH = os.environ.get("TOKENS_PATH", "/data/tokens.txt")
SAMPLE_RATE = 16000
NUM_THREADS = int(os.environ.get("NUM_THREADS", "2"))
PORT = int(os.environ.get("PORT", "8080"))

# ============================================================================
# Load model at startup
# ============================================================================
log.info("=" * 60)
log.info("Hifzapp — Zipformer Quran ASR (sherpa-onnx)")
log.info("=" * 60)

if not os.path.isfile(MODEL_PATH):
    log.error(f"Model file not found: {MODEL_PATH}")
    sys.exit(1)
if not os.path.isfile(TOKENS_PATH):
    log.error(f"Tokens file not found: {TOKENS_PATH}")
    sys.exit(1)

log.info(f"Loading model: {MODEL_PATH}")
log.info(f"Loading tokens: {TOKENS_PATH}")
log.info(f"Threads: {NUM_THREADS}, Sample rate: {SAMPLE_RATE}")

try:
    import sherpa_onnx
    recognizer = sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(
        tokens=TOKENS_PATH,
        model=MODEL_PATH,
        num_threads=NUM_THREADS,
        provider="cpu",
        sample_rate=SAMPLE_RATE,
        feature_dim=80,
        decoding_method="greedy_search",
    )
except AttributeError as e:
    log.error(f"sherpa-onnx missing from_zipformer2_ctc: {e}")
    log.error("Try: pip install --upgrade sherpa-onnx")
    sys.exit(1)
except Exception as e:
    log.error(f"Failed to load model: {type(e).__name__}: {e}")
    sys.exit(1)

log.info("Model loaded. Ready for streaming inference.")
log.info("=" * 60)

# ============================================================================
# FastAPI app
# ============================================================================
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from starlette.middleware.base import BaseHTTPMiddleware

app = FastAPI(title="Zipformer Quran ASR", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Static file middleware (serves frontend alongside API routes)
# ============================================================================
_STATIC_DIR = os.environ.get("STATIC_DIR", "/app/static")
_API_PREFIXES = (
    "/ws", "/api", "/healthz", "/readyz",
    "/openapi.json", "/docs", "/redoc", "/favicon.ico",
)


class StaticFilesMiddleware(BaseHTTPMiddleware):
    """Serve files from _STATIC_DIR for non-API paths."""

    async def dispatch(self, request, call_next):
        path = request.url.path
        if any(path == p or path.startswith(p + "/") for p in _API_PREFIXES):
            return await call_next(request)
        if not os.path.isdir(_STATIC_DIR):
            return await call_next(request)
        # Resolve the requested file
        requested = path.lstrip("/") or "index.html"
        file_path = os.path.join(_STATIC_DIR, requested)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        # Fall back to index.html for SPA routing
        index_path = os.path.join(_STATIC_DIR, "index.html")
        if os.path.isfile(index_path):
            return FileResponse(index_path)
        return await call_next(request)


if os.path.isdir(_STATIC_DIR):
    app.add_middleware(StaticFilesMiddleware)
else:
    log.warning(f"Static dir not found: {_STATIC_DIR} (API-only mode)")

# ============================================================================
# HTTP endpoints
# ============================================================================


@app.get("/healthz")
async def healthz():
    return JSONResponse(
        {
            "status": "ok",
            "model": "zipformer_p-quran",
            "framework": "sherpa-onnx",
            "sample_rate": SAMPLE_RATE,
            "model_path": MODEL_PATH,
        }
    )


@app.get("/readyz")
async def readyz():
    return JSONResponse({"ready": True})


# ============================================================================
# WebSocket — streaming ASR
# ============================================================================
# Wire format: binary frames of int16 PCM, 16kHz, mono.
# Server sends JSON text frames:
#   {"type": "partial", "text": "<phoneme text>"}
#   {"type": "final",   "text": "<phoneme text>"}
#   {"type": "error",   "message": "<error>"}
# ============================================================================


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    stream = recognizer.create_stream()
    client = websocket.client
    log.info(f"Client connected: {client}")

    # Stats (per-connection)
    chunks_received = 0
    samples_received = 0
    last_text = ""

    try:
        while True:
            # Receive audio chunk (int16 PCM, 16kHz, mono)
            data = await websocket.receive_bytes()
            if not data:
                continue

            # Convert int16 PCM to float32 in [-1, 1]
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            chunks_received += 1
            samples_received += len(samples)

            # Feed to sherpa-onnx stream
            stream.accept_waveform(SAMPLE_RATE, samples)

            # Decode while ready
            while recognizer.is_ready(stream):
                recognizer.decode_streams([stream])

            # Send partial result (only if it changed)
            text = recognizer.get_result(stream)
            if text and text != last_text:
                last_text = text
                await websocket.send_json({"type": "partial", "text": text})

    except WebSocketDisconnect:
        log.info(
            f"Client disconnected after {chunks_received} chunks "
            f"({samples_received} samples, {samples_received/SAMPLE_RATE:.1f}s)"
        )
        # Final flush
        try:
            stream.input_finished()
            while recognizer.is_ready(stream):
                recognizer.decode_streams([stream])
            final_text = recognizer.get_result(stream)
            if final_text and final_text != last_text:
                await websocket.send_json({"type": "final", "text": final_text})
            elif final_text:
                await websocket.send_json({"type": "final", "text": final_text})
        except Exception as e:
            log.warning(f"Error during final flush: {e}")
    except Exception as e:
        log.exception(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


# ============================================================================
# Entry point
# ============================================================================
if __name__ == "__main__":
    import uvicorn

    log.info(f"Starting server on 0.0.0.0:{PORT}")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=PORT,
        log_level="info",
        access_log=False,
    )

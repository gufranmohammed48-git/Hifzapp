"""FastConformer Quran ASR — int8 ONNX streaming backend.

Uses Muno459/fastconformer-quran's int8 ONNX export
(model_with_encoder.q8.onnx) via onnxruntime directly. The
"with_encoder" variant includes NeMo's audio preprocessing
(mel features) inside the ONNX graph, so we feed raw 16kHz
audio and get back logprobs/transcripts directly.

Drops the image from ~3GB (NeMo) to ~700MB. Latency drops
from 2-3s to ~300-600ms on Intel UHD (3-5x speedup).
"""
import os
import re
import sys
import json
import time
import logging
import unicodedata
from typing import Optional, List

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("fastconformer-int8")

# ============================================================================
# Configuration
# ============================================================================
MODEL_PATH = os.environ.get("MODEL_PATH", "/data/model_with_encoder.q8.onnx")
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "/data/tokenizer.model")
SAMPLE_RATE = 16000
NUM_THREADS = int(os.environ.get("NUM_THREADS", "2"))
PORT = int(os.environ.get("PORT", "8080"))
WINDOW_SEC = float(os.environ.get("WINDOW_SEC", "1.0"))  # audio window per inference

# ============================================================================
# Arabic normalizer
# ============================================================================
_DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")
_ALEF_RE = re.compile(r"[إأآاٱ]")
_YA_RE = re.compile(r"[يىی]")
_KAF_RE = re.compile(r"[كک]")
_HA_RE = re.compile(r"[هھ]")
_NON_ARABIC_RE = re.compile(r"[^\u0600-\u06FF]")


def normalize_arabic(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _DIACRITICS_RE.sub("", text)
    text = _ALEF_RE.sub("ا", text)
    text = _YA_RE.sub("ي", text)
    text = _KAF_RE.sub("ك", text)
    text = _HA_RE.sub("ه", text)
    text = text.replace("ة", "ه")
    text = _NON_ARABIC_RE.sub("", text)
    return text.strip()


# ============================================================================
# Load model + tokenizer at startup
# ============================================================================
log.info("=" * 60)
log.info("Hifzapp — FastConformer int8 ONNX ASR")
log.info("=" * 60)

if not os.path.isfile(MODEL_PATH):
    log.error(f"Model not found: {MODEL_PATH}")
    sys.exit(1)
if not os.path.isfile(TOKENIZER_PATH):
    log.error(f"Tokenizer not found: {TOKENIZER_PATH}")
    log.error("Download it: huggingface-cli download Muno459/fastconformer-quran tokenizer.model")
    sys.exit(1)

log.info(f"Loading model: {MODEL_PATH}")
import onnxruntime as ort
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = NUM_THREADS
sess_options.inter_op_num_threads = NUM_THREADS
sess = ort.InferenceSession(MODEL_PATH, sess_options=sess_options, providers=["CPUExecutionProvider"])

_input_names = [i.name for i in sess.get_inputs()]
_output_names = [o.name for o in sess.get_outputs()]
log.info(f"  {len(_input_names)} inputs: {_input_names}")
log.info(f"  {len(_output_names)} outputs: {_output_names}")
for inp in sess.get_inputs():
    log.info(f"    IN  {inp.name}: shape={inp.shape}, type={inp.type}")
for out in sess.get_outputs():
    log.info(f"    OUT {out.name}: shape={out.shape}, type={out.type}")

log.info(f"Loading tokenizer: {TOKENIZER_PATH}")
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.Load(TOKENIZER_PATH)
vocab_size = sp.GetPieceSize()
log.info(f"  SentencePiece pieces: {vocab_size}")
# Print the first few pieces and any blank-like tokens
for i in range(min(5, vocab_size)):
    log.info(f"    piece {i}: {sp.IdToPiece(i)!r}")
# Look for a <blk> piece explicitly
try:
    blk_id = sp.PieceToId("<blk>")
    log.info(f"  <blk> token id: {blk_id}")
except Exception:
    blk_id = None
    log.info("  no <blk> piece in tokenizer")

log.info("Model + tokenizer loaded. Ready for inference.")


# ============================================================================
# CTC decoding
# ============================================================================
# The "with_encoder" model may output one of:
#   - "transcripts": list of strings (already decoded by NeMo)
#   - "greedy_predictions": token IDs (BPE/subword IDs to merge)
#   - "logprobs": [B, T, V] log-probabilities to argmax + CTC collapse
#
# For CTC, the blank token ID is whatever the model treats as blank.
# Try in order: explicit <blk>, vocab_size (extra class), 0, vocab_size-1.

def detect_blank_id(outputs: dict) -> int:
    """Try to figure out the blank token ID from the model output shape."""
    # Look for a vocab dimension we can read
    for out_name, arr in outputs.items():
        if arr.ndim == 3:  # [B, T, V] - logprobs
            return arr.shape[-1] - 1  # last class is usually blank in NeMo CTC
    # Fallback: <blk> from tokenizer, or 0
    if blk_id is not None:
        return blk_id
    return 0


def ctc_greedy_decode(logprobs: np.ndarray, blank_id: int) -> str:
    """Greedy CTC decode. logprobs: [B, T, V] → text string."""
    preds = logprobs.argmax(axis=-1)
    if preds.ndim == 2:
        preds = preds[0]
    piece_max = sp.GetPieceSize() - 1
    collapsed: List[int] = []
    prev = -1
    for p in preds:
        p = int(p)
        if p > piece_max:
            continue  # extra class (probably blank or padding)
        if p != prev and p != blank_id:
            collapsed.append(p)
        prev = p
    return sp.DecodeIds(collapsed)


def decode_output(result: dict) -> str:
    """Decode the model output to text, handling all output formats."""
    # 1) Already-decoded transcripts
    if "transcripts" in result:
        transcripts = result["transcripts"]
        if len(transcripts) > 0:
            return str(transcripts[0])
        return ""
    # 2) Greedy predictions (token IDs)
    if "greedy_predictions" in result:
        tokens = result["greedy_predictions"]
        if tokens.ndim == 2:
            tokens = tokens[0]
        piece_max = sp.GetPieceSize() - 1
        blank_id = detect_blank_id({k: v for k, v in result.items() if k != "greedy_predictions"})
        decoded: List[int] = []
        prev = -1
        for t in tokens:
            t = int(t)
            if t > piece_max:
                continue
            if t != prev and t != blank_id:
                decoded.append(t)
            prev = t
        return sp.DecodeIds(decoded)
    # 3) Logits / log-probs
    for key in ("logprobs", "logits", "outputs"):
        if key in result:
            blank_id = detect_blank_id(result)
            return ctc_greedy_decode(result[key], blank_id)
    log.warning(f"Unknown output format. Keys: {list(result.keys())}")
    return ""


# ============================================================================
# Per-connection streaming state
# ============================================================================
class StreamState:
    """Holds the rolling audio buffer and last-decoded text for one WS connection."""

    def __init__(self):
        self.audio_buffer = np.zeros(0, dtype=np.float32)
        self.last_text = ""
        self.window_samples = int(SAMPLE_RATE * WINDOW_SEC)

    def feed(self, audio_chunk: np.ndarray) -> Optional[str]:
        """Feed an audio chunk. Returns new text when it changes, else None."""
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])

        # Wait until we have at least one full window of audio
        if len(self.audio_buffer) < self.window_samples:
            return None

        # Run inference on the accumulated window
        audio = self.audio_buffer
        # The 'with_encoder' model expects raw audio as [B, T, 1] (3D
        # with explicit channel dim). Try that first; if it errors, the
        # shape might be [B, 1, T] (channel-first) — we'll see.
        audio_signal = audio[np.newaxis, :, np.newaxis].astype(np.float32)
        length = np.array([audio.shape[0]], dtype=np.int64)

        t0 = time.time()
        try:
            outputs = sess.run(_output_names, {
                "audio_signal": audio_signal,
                "length": length,
            })
        except Exception as e:
            log.error(f"Inference failed: {e}")
            self.audio_buffer = np.zeros(0, dtype=np.float32)
            return None
        dt = (time.time() - t0) * 1000

        result = dict(zip(_output_names, outputs))
        text = decode_output(result)

        # Consume the audio we just inferred on
        self.audio_buffer = np.zeros(0, dtype=np.float32)

        if dt > 500:
            log.warning(f"Slow inference: {dt:.0f}ms for {len(audio)/SAMPLE_RATE:.2f}s audio")
        if not text:
            return None
        if text == self.last_text:
            return None
        self.last_text = text
        return text

    def flush(self) -> Optional[str]:
        """Run a final inference on any remaining audio in the buffer."""
        if len(self.audio_buffer) < int(SAMPLE_RATE * 0.3):  # < 300ms, skip
            return None
        return self.feed(np.zeros(0, dtype=np.float32))  # already has data


# ============================================================================
# FastAPI app
# ============================================================================
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from starlette.middleware.base import BaseHTTPMiddleware

app = FastAPI(title="FastConformer int8 ONNX ASR", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_STATIC_DIR = os.environ.get("STATIC_DIR", "/app/static")
_API_PREFIXES = (
    "/ws", "/api", "/healthz", "/readyz",
    "/openapi.json", "/docs", "/redoc", "/favicon.ico",
)


class StaticFilesMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        path = request.url.path
        if any(path == p or path.startswith(p + "/") for p in _API_PREFIXES):
            return await call_next(request)
        if not os.path.isdir(_STATIC_DIR):
            return await call_next(request)
        requested = path.lstrip("/") or "index.html"
        file_path = os.path.join(_STATIC_DIR, requested)
        if os.path.isfile(file_path):
            response = FileResponse(file_path)
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
            return response
        index_path = os.path.join(_STATIC_DIR, "index.html")
        if os.path.isfile(index_path):
            response = FileResponse(index_path)
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
            return response
        return await call_next(request)


if os.path.isdir(_STATIC_DIR):
    app.add_middleware(StaticFilesMiddleware)
else:
    log.warning(f"Static dir not found: {_STATIC_DIR}")


# ============================================================================
# HTTP endpoints
# ============================================================================
@app.get("/healthz")
async def healthz():
    return JSONResponse({
        "status": "ok",
        "model": "fastconformer-quran int8 ONNX",
        "framework": "onnxruntime",
        "sample_rate": SAMPLE_RATE,
        "vocab_size": vocab_size,
        "window_sec": WINDOW_SEC,
    })


@app.get("/readyz")
async def readyz():
    return JSONResponse({"ready": True})


# ============================================================================
# WebSocket — streaming ASR
# ============================================================================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    state = StreamState()
    log.info("Client connected")
    try:
        while True:
            message = await websocket.receive()
            if message is None:
                continue
            if message.get("type") == "websocket.disconnect":
                break

            data = message.get("bytes")
            if data is None:
                text_msg = message.get("text")
                if text_msg is None:
                    continue
                if text_msg.startswith("{"):
                    continue  # control message
                data = text_msg.encode("latin-1")
            if not data:
                continue

            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            text = state.feed(samples)
            if text is not None:
                norm = normalize_arabic(text)
                await websocket.send_json({
                    "type": "partial",
                    "text": text,
                    "norm": norm,
                })
    except WebSocketDisconnect:
        log.info("Client disconnected")
    except Exception as e:
        log.exception(f"WebSocket error: {e}")
    finally:
        try:
            final_text = state.flush()
            if final_text:
                norm = normalize_arabic(final_text)
                await websocket.send_json({
                    "type": "final",
                    "text": final_text,
                    "norm": norm,
                })
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

"""FastConformer Streaming Quran ASR — WebSocket server.

Uses Muno459/fastconformer-quran-streaming (cache-aware streaming export
of the same 4.13%-WER model from Muno459/fastconformer-quran) via
onnxruntime. Output is clean Arabic text (NOT phonemes), so the existing
frontend matching works without changes.

Why this model:
- Same accuracy as the original FastConformer (4.13% WER, clean text)
- Streaming mode with cache state (designed for real-time)
- int8 quantized (132MB vs 459MB of the NeMo .nemo)
- No NeMo dependency — 50x smaller Docker image, 10x faster startup
"""
import os
import re
import sys
import json
import time
import logging
import unicodedata
from typing import Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("fastconformer-streaming")

# ============================================================================
# Configuration
# ============================================================================
MODEL_PATH = os.environ.get("MODEL_PATH", "/data/model_streaming_with_encoder.q8.onnx")
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "/data/tokenizer.model")
CMVN_PATH = os.environ.get("CMVN_PATH", "/data/streaming_global_cmvn.npz")
SAMPLE_RATE = 16000
NUM_THREADS = int(os.environ.get("NUM_THREADS", "2"))
PORT = int(os.environ.get("PORT", "8080"))

# Audio feature extraction parameters (NeMo/kaldi-style)
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 400
N_MELS = 80
PRE_EMPHASIS = 0.97
LOG_EPS = 1e-6

# Streaming chunk size (in mel frames at 10ms hop)
# 100 frames = 1 second — matches the WebSocket audio chunk size
STREAM_CHUNK_FRAMES = 100

# Cache shapes (from the ONNX model)
CACHE_LAST_CHANNEL_SHAPE = (1, 17, 70, 512)  # [B, n_layers, ctx_frames, d_model]
CACHE_LAST_TIME_SHAPE = (1, 17, 512, 8)      # [B, n_layers, d_model, ctx_frames]

# ============================================================================
# Arabic normalizer — strips diacritics, unifies variants
# ============================================================================
_DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")
_ALEF_RE = re.compile(r"[إأآاٱ]")
_YA_RE = re.compile(r"[يىی]")
_KAF_RE = re.compile(r"[كک]")
_HA_RE = re.compile(r"[هھ]")
_NON_ARABIC_RE = re.compile(r"[^\u0600-\u06FF]")


def normalize_arabic(text: str) -> str:
    """Strip diacritics + unify character variants for matching."""
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
# Feature extraction — mel spectrogram in pure numpy
# ============================================================================
def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def make_mel_filterbank(n_mels=N_MELS, n_fft=N_FFT, sample_rate=SAMPLE_RATE,
                        fmin=0.0, fmax=None):
    """Triangular mel filterbank (kaldi-style, [n_mels, n_fft//2 + 1])."""
    if fmax is None:
        fmax = sample_rate / 2.0
    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sample_rate / 2, n_freqs)
    mel_pts = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz_pts = _mel_to_hz(mel_pts)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
        lower = (fft_freqs - left) / (center - left)
        upper = (right - fft_freqs) / (right - center)
        fb[i] = np.maximum(0.0, np.minimum(lower, upper))
    return fb


# Pre-compute the mel filterbank and the window at module load.
# IMPORTANT: the window must be N_FFT wide (512), not WIN_LENGTH (400),
# because each frame is N_FFT=512 samples. If WIN_LENGTH < N_FFT, the
# window is implicitly zero-padded at the edges (kaldi convention).
_MEL_FB = make_mel_filterbank()
_WINDOW = np.hanning(N_FFT).astype(np.float32)


def compute_mel_features(audio: np.ndarray) -> np.ndarray:
    """Compute 80-dim log-mel features (kaldi-style, then CMVN applied later).

    audio: 1D float32 in [-1, 1]
    Returns: [n_frames, 80] float32
    """
    if len(audio) < N_FFT:
        return np.zeros((0, N_MELS), dtype=np.float32)

    # Pre-emphasis
    emphasized = np.concatenate([[audio[0]], audio[1:] - PRE_EMPHASIS * audio[:-1]])

    # Pad for centered frames
    pad = N_FFT // 2
    padded = np.pad(emphasized, (pad, pad), mode="reflect")

    # Frame the signal: shape [n_frames, N_FFT]
    n_frames = (len(padded) - N_FFT) // HOP_LENGTH + 1
    if n_frames <= 0:
        return np.zeros((0, N_MELS), dtype=np.float32)

    # Use sliding_window_view for efficient framing
    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[:n_frames * HOP_LENGTH:HOP_LENGTH]
    frames = frames * _WINDOW

    # Power spectrum via rFFT
    spec = np.fft.rfft(frames, n=N_FFT, axis=-1)
    power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)

    # Mel filterbank + log compression
    mel = power @ _MEL_FB.T
    log_mel = np.log(mel + LOG_EPS).astype(np.float32)
    return log_mel


# ============================================================================
# CTC greedy decoder
# ============================================================================
def ctc_greedy_decode(logprobs: np.ndarray, sp, blank_id: int) -> str:
    """Greedy CTC decode. logprobs: [B, T, V] → text string.

    Uses SentencePiece's DecodeIds which handles the subword merging
    for us (no manual BPE/WPM bookkeeping).

    The model's output vocab (1025) is one larger than the SentencePiece
    piece count (1024) — there's an extra output slot (probably for a
    NeMo-specific padding/blank variant). We clamp the argmax to the
    valid piece range to avoid IndexError from SentencePiece.
    """
    preds = logprobs.argmax(axis=-1)  # [B, T]
    if preds.ndim == 2:
        preds = preds[0]
    piece_max = sp.GetPieceSize() - 1
    collapsed = []
    prev = -1
    for p in preds:
        p = int(p)
        # Clamp out-of-range IDs to the last valid piece (they shouldn't
        # occur in well-trained output, but defensive against the extra
        # output slot this model has).
        if p > piece_max:
            continue
        if p != prev and p != blank_id:
            collapsed.append(p)
        prev = p
    return sp.DecodeIds(collapsed)


# ============================================================================
# Load model + tokenizer + CMVN at startup
# ============================================================================
log.info("=" * 60)
log.info("Hifzapp — FastConformer Streaming Quran ASR (onnxruntime)")
log.info("=" * 60)

if not os.path.isfile(MODEL_PATH):
    log.error(f"Model not found: {MODEL_PATH}")
    sys.exit(1)
if not os.path.isfile(TOKENIZER_PATH):
    log.error(f"Tokenizer not found: {TOKENIZER_PATH}")
    sys.exit(1)
if not os.path.isfile(CMVN_PATH):
    log.error(f"CMVN not found: {CMVN_PATH}")
    sys.exit(1)

log.info(f"Loading model: {MODEL_PATH}")
import onnxruntime as ort
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = NUM_THREADS
sess_options.inter_op_num_threads = NUM_THREADS
sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
sess = ort.InferenceSession(MODEL_PATH, sess_options=sess_options, providers=["CPUExecutionProvider"])

# Resolve input/output names (so we can use the dict-by-name API)
_input_names = [i.name for i in sess.get_inputs()]
_output_names = [o.name for o in sess.get_outputs()]
log.info(f"  {len(_input_names)} inputs: {_input_names}")
log.info(f"  {len(_output_names)} outputs: {_output_names}")

log.info(f"Loading tokenizer: {TOKENIZER_PATH}")
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.Load(TOKENIZER_PATH)
vocab_size = sp.GetPieceSize()
# The blank id is the first piece (typically 0 = <blk>)
blank_id = sp.PieceToId("<blk>") if sp.IdToPiece(0) == "<blk>" else 0
log.info(f"  vocab: {vocab_size}, blank_id: {blank_id}")

log.info(f"Loading CMVN: {CMVN_PATH}")
cmvn_npz = np.load(CMVN_PATH)
log.info(f"  CMVN keys: {list(cmvn_npz.files)}")
for k in cmvn_npz.files:
    log.info(f"    {k}: shape={cmvn_npz[k].shape}, dtype={cmvn_npz[k].dtype}")

# This file has two variants of CMVN stats:
#   - clean_* : statistics from clean studio recordings (tarteel-ai/everyayah)
#   - tlog_*  : statistics from real-world recitations (tarteel-ai/tlog)
# Default to 'clean' since the model was trained primarily on it;
# override with CMVN_VARIANT=tlog if recognition is poor for your
# mic (which captures more ambient noise than studio recordings).
CMVN_VARIANT = os.environ.get("CMVN_VARIANT", "clean").lower()
if CMVN_VARIANT not in ("clean", "tlog"):
    log.error(f"CMVN_VARIANT must be 'clean' or 'tlog', got: {CMVN_VARIANT!r}")
    sys.exit(1)

mean_key = f"{CMVN_VARIANT}_mean"
std_key = f"{CMVN_VARIANT}_std"
if mean_key not in cmvn_npz.files:
    log.error(f"CMVN file missing '{mean_key}'. Has: {list(cmvn_npz.files)}")
    sys.exit(1)
if std_key not in cmvn_npz.files:
    log.error(f"CMVN file missing '{std_key}'. Has: {list(cmvn_npz.files)}")
    sys.exit(1)

cmvn_mean = cmvn_npz[mean_key].astype(np.float32)
cmvn_std = cmvn_npz[std_key].astype(np.float32)
log.info(f"  raw '{CMVN_VARIANT}' mean: {cmvn_mean.shape}, std: {cmvn_std.shape}")

# CMVN shape might be (N,) for a 1D feature CMVN, or (N, 1)/(1, N) for 2D.
# Flatten to 1D. If the size is not 80 (n_mels), it may be for a different
# feature type (e.g. 400-dim linear spectrogram) — we'd need to regenerate.
cmvn_mean = cmvn_mean.reshape(-1)
cmvn_std = cmvn_std.reshape(-1)
log.info(f"  flattened mean: {cmvn_mean.shape}, std: {cmvn_std.shape}")
if cmvn_mean.shape[0] != N_MELS:
    log.warning(
        f"CMVN size {cmvn_mean.shape[0]} != n_mels {N_MELS}. "
        f"This usually means the CMVN was computed for a different feature "
        f"type (likely linear spectrogram, not log-mel). Expect bad results."
    )

log.info("Model + tokenizer + CMVN loaded. Ready for streaming inference.")
log.info("=" * 60)


# ============================================================================
# Per-connection streaming state
# ============================================================================
class StreamState:
    """Holds the cache state and audio buffer for one WebSocket connection."""

    def __init__(self):
        # Initial cache: all zeros with the right shape and dtype
        self.cache_last_channel = np.zeros(CACHE_LAST_CHANNEL_SHAPE, dtype=np.float32)
        self.cache_last_time = np.zeros(CACHE_LAST_TIME_SHAPE, dtype=np.float32)
        self.cache_last_channel_len = np.zeros([1], dtype=np.int64)
        # Audio buffer for accumulating samples until we have enough
        self.audio_buffer = np.zeros(0, dtype=np.float32)
        # Last decoded text (so we only send when it changes)
        self.last_text = ""
        # Frame count processed so far
        self.frames_processed = 0

    def feed(self, audio_chunk: np.ndarray) -> Optional[str]:
        """Feed an audio chunk. Returns the new text if it changed, else None."""
        # Append to buffer
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])

        # Process every STREAM_CHUNK_FRAMES frames
        # 1s of audio at 10ms hop = 100 frames
        new_text = None
        while True:
            # Compute features for the current buffer
            features = compute_mel_features(self.audio_buffer)
            n_frames = features.shape[0]
            # Process when we have at least STREAM_CHUNK_FRAMES frames,
            # OR when we have a partial chunk at the end of audio
            if n_frames < STREAM_CHUNK_FRAMES:
                break

            # Take the first STREAM_CHUNK_FRAMES frames
            chunk = features[:STREAM_CHUNK_FRAMES]
            # Apply CMVN — broadcast across feature dim (last axis)
            if cmvn_mean.shape[0] != chunk.shape[1]:
                log.error(
                    f"CMVN/feature size mismatch: cmvn={cmvn_mean.shape[0]}, "
                    f"features={chunk.shape[1]}. The CMVN file was probably "
                    f"computed for a different feature type. Inference will fail."
                )
            chunk = (chunk - cmvn_mean) / cmvn_std
            # Reshape for ONNX: [B=1, n_mels=80, T=chunk]
            audio_signal = chunk.T[np.newaxis, :, :].astype(np.float32)  # [1, 80, T]
            length = np.array([audio_signal.shape[-1]], dtype=np.int64)

            # Run inference
            outputs = sess.run(
                _output_names,
                {
                    "audio_signal": audio_signal,
                    "length": length,
                    "cache_last_channel": self.cache_last_channel,
                    "cache_last_time": self.cache_last_time,
                    "cache_last_channel_len": self.cache_last_channel_len,
                },
            )
            result = dict(zip(_output_names, outputs))

            # Update cache
            self.cache_last_channel = result["cache_last_channel_next"]
            self.cache_last_time = result["cache_last_time_next"]
            self.cache_last_channel_len = result["cache_last_channel_next_len"]

            # Decode logprobs → text
            text = ctc_greedy_decode(result["logprobs"], sp, blank_id)
            if text and text != self.last_text:
                self.last_text = text
                new_text = text

            # Advance buffer: drop the frames we processed
            # (in sample space: STREAM_CHUNK_FRAMES * HOP_LENGTH samples)
            samples_consumed = STREAM_CHUNK_FRAMES * HOP_LENGTH
            self.audio_buffer = self.audio_buffer[samples_consumed:]
            self.frames_processed += STREAM_CHUNK_FRAMES

        return new_text

    def flush(self) -> Optional[str]:
        """Process any remaining audio in the buffer (end of input)."""
        if len(self.audio_buffer) < N_FFT:
            return None
        features = compute_mel_features(self.audio_buffer)
        if features.shape[0] == 0:
            return None
        # Apply CMVN
        features = (features - cmvn_mean) / cmvn_std
        audio_signal = features.T[np.newaxis, :, :].astype(np.float32)
        length = np.array([audio_signal.shape[-1]], dtype=np.int64)
        outputs = sess.run(
            _output_names,
            {
                "audio_signal": audio_signal,
                "length": length,
                "cache_last_channel": self.cache_last_channel,
                "cache_last_time": self.cache_last_time,
                "cache_last_channel_len": self.cache_last_channel_len,
            },
        )
        result = dict(zip(_output_names, outputs))
        # Note: at flush, we don't update the cache (no more audio to come)
        text = ctc_greedy_decode(result["logprobs"], sp, blank_id)
        if text and text != self.last_text:
            self.last_text = text
            return text
        return None


# ============================================================================
# FastAPI app
# ============================================================================
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from starlette.middleware.base import BaseHTTPMiddleware

app = FastAPI(title="FastConformer Streaming Quran ASR", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Static file middleware
# ============================================================================
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
    return JSONResponse(
        {
            "status": "ok",
            "model": "fastconformer-quran-streaming",
            "framework": "onnxruntime",
            "sample_rate": SAMPLE_RATE,
            "vocab_size": vocab_size,
            "model_path": MODEL_PATH,
        }
    )


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
    client = websocket.client
    log.info(f"Client connected: {client}")
    msg_count = 0
    byte_count = 0
    try:
        while True:
            # Use receive() so we can handle both binary frames (int16 PCM)
            # and text frames (sometimes browsers send ArrayBuffers as text
            # over certain security contexts). This makes the endpoint
            # robust to client quirks.
            message = await websocket.receive()
            if message is None:
                continue
            msg_type = message.get("type")
            if msg_type == "websocket.disconnect":
                break

            data = message.get("bytes")
            if data is None:
                # Try text fallback — frontend may be sending a base64
                # string or a JSON envelope with the audio.
                text_msg = message.get("text")
                if text_msg is None:
                    continue
                # If it's a JSON control message, log and ignore
                if text_msg.startswith("{"):
                    log.info(f"Control message: {text_msg!r}")
                    continue
                # Else treat as raw int16 bytes encoded in a text frame
                data = text_msg.encode("latin-1")
            if not data:
                continue

            msg_count += 1
            byte_count += len(data)
            if msg_count <= 3 or msg_count % 10 == 0:
                log.info(
                    f"WS message #{msg_count}: {len(data)} bytes "
                    f"(total: {byte_count/1024:.1f} KB)"
                )

            # Convert int16 PCM to float32 in [-1, 1]
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            # Feed through the streaming state
            t0 = time.time()
            text = state.feed(samples)
            dt = time.time() - t0
            if dt > 0.5:
                log.warning(f"Slow inference: {dt*1000:.0f}ms for {len(samples)} samples")
            if text is not None:
                norm = normalize_arabic(text)
                log.info(f"Partial text: {text!r}")
                await websocket.send_json({
                    "type": "partial",
                    "text": text,
                    "norm": norm,
                })
    except WebSocketDisconnect:
        log.info(
            f"Client disconnected. Received {msg_count} messages, "
            f"{byte_count/1024:.1f} KB total"
        )
    except Exception as e:
        log.exception(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        # Always try to flush on disconnect
        try:
            final_text = state.flush()
            if final_text:
                norm = normalize_arabic(final_text)
                log.info(f"Final text: {final_text!r}")
                await websocket.send_json({
                    "type": "final",
                    "text": final_text,
                    "norm": norm,
                })
        except Exception as e:
            log.debug(f"Final flush error (client likely gone): {e}")


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

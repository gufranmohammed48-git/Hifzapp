#!/usr/bin/env python3
"""Test the Zipformer Quran model."""
import os
import sys
import time

MODEL_DIR = r"C:\Users\Gufran\Documents\model" if os.name == "nt" else os.path.expanduser("~/Documents/model")
MODEL_FILE = "quran_phoneme_zipformer.int8.onnx"
TOKENS_FILE = "tokens.txt"

model_path = os.path.join(MODEL_DIR, MODEL_FILE)
tokens_path = os.path.join(MODEL_DIR, TOKENS_FILE)

print(f"Model: {model_path}  exists={os.path.isfile(model_path)}")
print(f"Tokens: {tokens_path}  exists={os.path.isfile(tokens_path)}")
if not os.path.isfile(model_path):
    sys.exit(f"ERROR: model file not found: {model_path}")
if not os.path.isfile(tokens_path):
    sys.exit(f"ERROR: tokens file not found: {tokens_path}")

print()
print("=" * 60)
print("STEP 1: load tokens.txt")
print("=" * 60)
# FIX: open as utf-8 (Windows default cp1252 can't read Arabic)
with open(tokens_path, encoding="utf-8") as f:
    token_lines = [l.strip().split() for l in f if l.strip()]
id_to_token = {int(t[1]): t[0] for t in token_lines if len(t) == 2}
vocab_size = max(id_to_token.keys()) + 1
print(f"  Loaded {len(id_to_token)} tokens, vocab size = {vocab_size}")
blank_id = next((i for i, t in id_to_token.items() if t == "<blank>"), None)
print(f"  blank_id = {blank_id}")
print(f"  Sample: 0={id_to_token.get(0)!r}, 1={id_to_token.get(1)!r}, 5={id_to_token.get(5)!r}")
print(f"  Letters 'ا ب ت': {id_to_token.get(2)!r} {id_to_token.get(3)!r} {id_to_token.get(4)!r}")

print()
print("=" * 60)
print("STEP 2: try sherpa-onnx")
print("=" * 60)
try:
    import sherpa_onnx
    print(f"  sherpa-onnx already installed")
except ImportError:
    print("  sherpa-onnx not installed, trying pip install...")
    import subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "sherpa-onnx"], timeout=300)
        import sherpa_onnx
        print(f"  Installed")
    except Exception as e:
        print(f"  pip install FAILED: {type(e).__name__}: {e}")
        print("  sherpa-onnx likely doesn't have wheels for Python 3.14.")
        sherpa_onnx = None

if sherpa_onnx is not None:
    print()
    print("  Trying OnlineRecognizer loader variants...")
    loaders = [
        ("from_zipformer2_ctc",
            lambda: sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(
                tokens=tokens_path, model=model_path, num_threads=2, provider="cpu",
                sample_rate=16000, feature_dim=80, decoding_method="greedy_search")),
        ("from_zipformer_ctc",
            lambda: sherpa_onnx.OnlineRecognizer.from_zipformer_ctc(
                tokens=tokens_path, model=model_path, num_threads=2, provider="cpu",
                sample_rate=16000, feature_dim=80, decoding_method="greedy_search")),
        ("from_icefall_ctc",
            lambda: sherpa_onnx.OnlineRecognizer.from_icefall_ctc(
                tokens=tokens_path, model=model_path, num_threads=2, provider="cpu",
                sample_rate=16000, feature_dim=80, decoding_method="greedy_search")),
    ]
    recognizer = None
    used = None
    for name, fn in loaders:
        try:
            print(f"    {name}...", end=" ", flush=True)
            t0 = time.time()
            recognizer = fn()
            dt = time.time() - t0
            print(f"OK ({dt:.1f}s)")
            used = name
            break
        except (AttributeError, TypeError) as e:
            print(f"no signature ({e})")
        except Exception as e:
            print(f"failed: {type(e).__name__}: {e}")
            recognizer = None

    if recognizer is not None:
        print()
        print(f"  LOADED via {used}!")
        import numpy as np
        sample_rate = 16000
        silence = np.zeros(sample_rate, dtype=np.float32)
        stream = recognizer.create_stream()
        t0 = time.time()
        stream.accept_waveform(sample_rate, silence)
        while recognizer.is_ready(stream):
            recognizer.decode_streams([stream])
        dt = time.time() - t0
        text = recognizer.get_result(stream)
        print(f"    Silence inference: {dt*1000:.0f}ms, result: '{text}'")
        print()
        print("=" * 60)
        print("SHERPA-ONNX WORKS")
        print("=" * 60)
        sys.exit(0)

print()
print("=" * 60)
print("STEP 3: direct onnxruntime fallback")
print("=" * 60)
import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "onnxruntime"], timeout=180)
import onnxruntime as ort
import numpy as np

print(f"  onnxruntime: {ort.__version__}")
sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

feed = {}
for inp in sess.get_inputs():
    if inp.name == "x":
        feed[inp.name] = np.zeros([1, 61, 80], dtype=np.float32)
    elif inp.name == "processed_lens":
        feed[inp.name] = np.array([61], dtype=np.int64)
    else:
        shape = [1 if isinstance(s, str) and s == "N" else s for s in inp.shape]
        feed[inp.name] = np.zeros(shape, dtype=np.float32)

output_names = [o.name for o in sess.get_outputs()]
t0 = time.time()
outputs = sess.run(output_names, feed)
dt = time.time() - t0
print(f"  Inference time: {dt*1000:.0f}ms")
print(f"  log_probs shape: {outputs[0].shape} (expected [1, 61, 251])")
print()
print("=" * 60)
print("MODEL WORKS WITH ONNXRUNTIME")
print(f"Per-chunk latency: {dt*1000:.0f}ms")
print("=" * 60)

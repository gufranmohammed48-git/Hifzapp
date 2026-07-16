#!/usr/bin/env python3
"""test_streaming_fc.py — smoke test for the streaming FastConformer Quran model.

Tries sherpa-onnx first (cleanest path). If sherpa-onnx doesn't have a
direct factory for this model, falls back to onnxruntime direct test.

The streaming FastConformer is a cache-aware Transducer (RNN-T) with
encoder+decoder+joiner baked into a single ONNX file. It outputs
clean Arabic text (same as the original FastConformer), so the
existing frontend matching should work.
"""
import os
import sys
import time

MODEL_DIR = r"C:\Users\Gufran\Documents\model" if os.name == "nt" else os.path.expanduser("~/Documents/model")
MODEL_FILE = "model_streaming_with_encoder.q8.onnx"
TOKENIZER_FILE = "tokenizer.model"
CMVN_FILE = "streaming_global_cmvn.npz"

model_path = os.path.join(MODEL_DIR, MODEL_FILE)
tokenizer_path = os.path.join(MODEL_DIR, TOKENIZER_FILE)
cmvn_path = os.path.join(MODEL_DIR, CMVN_FILE)

print(f"Model:    {model_path}  exists={os.path.isfile(model_path)}")
print(f"Tokenizer:{tokenizer_path}  exists={os.path.isfile(tokenizer_path)}")
print(f"CMVN:     {cmvn_path}  exists={os.path.isfile(cmvn_path)}")
if not os.path.isfile(model_path):
    sys.exit(f"ERROR: model file not found: {model_path}")
if not os.path.isfile(tokenizer_path):
    sys.exit(f"ERROR: tokenizer not found: {tokenizer_path}")
if not os.path.isfile(cmvn_path):
    sys.exit(f"ERROR: CMVN not found: {cmvn_path}")

# Show file sizes
for f in [model_path, tokenizer_path, cmvn_path]:
    sz = os.path.getsize(f) / 1024 / 1024
    print(f"  {os.path.basename(f)}: {sz:.2f} MB")

print()
print("=" * 60)
print("STEP 1: try sherpa-onnx")
print("=" * 60)
try:
    import sherpa_onnx
    print(f"  sherpa-onnx installed")
except ImportError:
    print("  sherpa-onnx not installed, trying pip install...")
    import subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "sherpa-onnx"], timeout=300)
        import sherpa_onnx
    except Exception as e:
        print(f"  pip install FAILED: {e}")
        sherpa_onnx = None

if sherpa_onnx is not None:
    # Try different factory methods that might work for a Transducer with
    # cache state. The model is single-file with encoder baked in.
    loaders = [
        ("from_transducer (single file as encoder)",
            lambda: sherpa_onnx.OnlineRecognizer.from_transducer(
                tokens=tokenizer_path,
                encoder=model_path,
                decoder=model_path,
                joiner=model_path,
                num_threads=2, provider="cpu",
                sample_rate=16000, feature_dim=80,
                decoding_method="greedy_search")),
        ("from_transducer (just encoder, auto-find)",
            lambda: sherpa_onnx.OnlineRecognizer.from_transducer(
                tokens=tokenizer_path,
                encoder=model_path,
                num_threads=2, provider="cpu",
                sample_rate=16000, feature_dim=80,
                decoding_method="greedy_search")),
        ("from_nemo_encdec (if exists)",
            lambda: getattr(sherpa_onnx.OnlineRecognizer, "from_nemo_encdec")(
                tokens=tokenizer_path, model=model_path,
                num_threads=2, provider="cpu",
                sample_rate=16000, feature_dim=80,
                decoding_method="greedy_search")),
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
            print(f"no ({type(e).__name__})")
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
        print("SHERPA-ONNX WORKS — ready to integrate into app.py")
        print("=" * 60)
        sys.exit(0)
    else:
        print("  No loader worked. Will inspect model directly.")

print()
print("=" * 60)
print("STEP 2: direct onnxruntime inspection")
print("=" * 60)
import onnxruntime as ort
import numpy as np

print(f"  onnxruntime: {ort.__version__}")
sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

print(f"  {len(sess.get_inputs())} inputs, {len(sess.get_outputs())} outputs")
print()
print("  INPUTS (top 15):")
for i, inp in enumerate(sess.get_inputs()[:15]):
    print(f"    {inp.name:35} shape={inp.shape}  type={inp.type}")
if len(sess.get_inputs()) > 15:
    print(f"    ... and {len(sess.get_inputs()) - 15} more")

print()
print("  OUTPUTS:")
for out in sess.get_outputs():
    print(f"    {out.name:35} shape={out.shape}  type={out.type}")

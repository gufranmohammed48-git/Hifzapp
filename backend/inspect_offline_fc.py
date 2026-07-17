"""inspect_offline_fc.py — inspect the offline int8 FastConformer ONNX.

Run:  python inspect_offline_fc.py

Prints all inputs/outputs with shapes, plus runs a quick smoke test
with a short synthetic audio so we know what the model produces.
"""
import os
import sys
import numpy as np

MODEL = r"C:\Users\Gufran\Documents\model\model_with_encoder.q8.onnx"

if not os.path.isfile(MODEL):
    sys.exit(f"ERROR: model not found at {MODEL}")

import onnxruntime as ort
print(f"Loading {MODEL} ...")
sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])

print()
print("=== INPUTS ===")
for i in sess.get_inputs():
    print(f"  Name: {i.name}")
    print(f"    Shape: {i.shape}")
    print(f"    Type:  {i.type}")
    print()

print("=== OUTPUTS ===")
for o in sess.get_outputs():
    print(f"  Name: {o.name}")
    print(f"    Shape: {o.shape}")
    print(f"    Type:  {o.type}")
    print()

# Smoke test with 1 second of silence at 16kHz
print("=== SMOKE TEST (1s silence) ===")
sample_rate = 16000
audio = np.zeros(sample_rate, dtype=np.float32)
audio_signal = audio[np.newaxis, :].astype(np.float32)  # [1, T]
length = np.array([audio.shape[0]], dtype=np.int64)

feed = {
    "audio_signal": audio_signal,
    "length": length,
}

# Catch any other input names the model might expect
for inp in sess.get_inputs():
    if inp.name not in feed:
        # If there's a third input we don't know about, init it as zeros
        shape = [1 if s == "N" else (s if isinstance(s, int) else 1) for s in inp.shape]
        feed[inp.name] = np.zeros(shape, dtype=np.int64 if "length" in inp.name or "lens" in inp.name else np.float32)
        print(f"  [init] extra input {inp.name} with shape {shape}")

import time
t0 = time.time()
outputs = sess.run([o.name for o in sess.get_outputs()], feed)
dt = (time.time() - t0) * 1000
result = dict(zip([o.name for o in sess.get_outputs()], outputs))

print(f"  Inference: {dt:.0f}ms")
for name, arr in result.items():
    print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}")
    if hasattr(arr, 'flat') and arr.size > 0:
        print(f"    sample: {arr.flat[0]}")

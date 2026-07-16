"""inspect_streaming_fc.py — inspect the streaming FastConformer Quran ONNX.

Prints all inputs/outputs with their shapes and types. Same as what
we did for the Zipformer model, so we know what to feed the model.
"""
import onnxruntime as ort

MODEL = r"C:\Users\Gufran\Documents\model\model_streaming_with_encoder.q8.onnx"

print(f"Loading {MODEL}...")
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

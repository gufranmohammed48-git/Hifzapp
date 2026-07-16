"""inspect_model.py — dump the ONNX model's inputs and outputs."""
import onnxruntime as ort

MODEL = r"C:\Users\Gufran\Documents\model\quran_phoneme_zipformer.int8.onnx"

print(f"Loading {MODEL}...")
sess = ort.InferenceSession(MODEL)

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

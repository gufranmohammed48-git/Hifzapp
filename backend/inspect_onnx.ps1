# inspect_onnx.ps1 — dump the ONNX model's inputs and outputs
# Run: powershell -ExecutionPolicy Bypass -File inspect_onnx.ps1

$model = "C:\Users\Gufran\Documents\model\quran_phoneme_zipformer.int8.onnx"

Write-Host "Installing onnxruntime (first run only)..." -ForegroundColor Cyan
python -m pip install --quiet onnxruntime 2>&1 | Out-Null

Write-Host ""
Write-Host "Model: $model" -ForegroundColor Cyan
Write-Host ""

python @"
import onnxruntime as ort
sess = ort.InferenceSession(r'$model')
print('=== INPUTS ===')
for i in sess.get_inputs():
    print(f'  Name: {i.name}')
    print(f'    Shape: {i.shape}')
    print(f'    Type:  {i.type}')
    print()
print('=== OUTPUTS ===')
for o in sess.get_outputs():
    print(f'  Name: {o.name}')
    print(f'    Shape: {o.shape}')
    print(f'    Type:  {o.type}')
    print()
"@

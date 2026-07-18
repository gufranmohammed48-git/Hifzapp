# ===========================================================================
# start-direct.ps1 - Run Hifzapp ASR backend directly (no Docker)
# ===========================================================================
# PowerShell version with cleaner output. Same effect as start-direct.bat.
#
# Usage:  .\start-direct.ps1
# ===========================================================================

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=== Hifzapp direct mode (no Docker) ===" -ForegroundColor Cyan
Write-Host ""

# ----- Detect Python -----
$python = (Get-Command python -ErrorAction SilentlyContinue)
if (-not $python) {
    Write-Host "[ERROR] Python not found in PATH." -ForegroundColor Red
    Write-Host "Install Python 3.10+ from https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}
Write-Host "Using $(python --version)"

# ----- Verify model files -----
$MODEL_DIR = "C:\Users\Gufran\Documents\model"
if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match "^\s*MODEL_HOST_PATH\s*=\s*(.+)\s*$") {
            $MODEL_DIR = $Matches[1].Trim()
        }
    }
}
if ($env:MODEL_HOST_PATH) { $MODEL_DIR = $env:MODEL_HOST_PATH }

Write-Host "Checking model files in: $MODEL_DIR"
$modelFile = Join-Path $MODEL_DIR "model_with_encoder.q8.onnx"
$tokFile   = Join-Path $MODEL_DIR "tokenizer.model"
if (-not (Test-Path $modelFile)) {
    Write-Host "[ERROR] $modelFile not found" -ForegroundColor Red
    Write-Host "Download it:" -ForegroundColor Yellow
    Write-Host "    huggingface-cli download Muno459/fastconformer-quran model_with_encoder.q8.onnx --local-dir `"$MODEL_DIR`"" -ForegroundColor Yellow
    exit 1
}
if (-not (Test-Path $tokFile)) {
    Write-Host "[ERROR] $tokFile not found" -ForegroundColor Red
    Write-Host "Download it:" -ForegroundColor Yellow
    Write-Host "    huggingface-cli download Muno459/fastconformer-quran tokenizer.model --local-dir `"$MODEL_DIR`"" -ForegroundColor Yellow
    exit 1
}
Write-Host "  [OK] model_with_encoder.q8.onnx" -ForegroundColor Green
Write-Host "  [OK] tokenizer.model" -ForegroundColor Green
Write-Host ""

# ----- Setup venv -----
$VENV_DIR = ".venv"
$VENV_PY = Join-Path $VENV_DIR "Scripts\python.exe"
if (-not (Test-Path $VENV_PY)) {
    Write-Host "Creating Python venv at $VENV_DIR..." -ForegroundColor Cyan
    python -m venv $VENV_DIR
    Write-Host "Installing dependencies (2-3 min first time)..." -ForegroundColor Cyan
    & "$VENV_DIR\Scripts\activate.ps1"
    python -m pip install --upgrade pip | Out-Null
    # --only-binary :all: prevents pip from trying to build packages
    # from source (which needs Visual Studio Build Tools on Windows).
    # If a package has no wheel for your Python, this surfaces a
    # clear error instead of a confusing meson/MSVC trace.
    pip install --only-binary :all: -r backend\requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "" -ForegroundColor Red
        Write-Host "[ERROR] Failed to install dependencies" -ForegroundColor Red
        Write-Host "Common cause: your Python version (likely 3.13) has no prebuilt wheels for some package." -ForegroundColor Yellow
        Write-Host "Fix: install Python 3.12 from https://www.python.org/downloads/ and re-run this script." -ForegroundColor Yellow
        exit 1
    }
} else {
    Write-Host "Using existing venv at $VENV_DIR" -ForegroundColor Cyan
    & "$VENV_DIR\Scripts\activate.ps1"
}

# ----- Set env vars -----
$env:MODEL_PATH     = $modelFile
$env:TOKENIZER_PATH = $tokFile
$env:STATIC_DIR     = (Get-Location).Path
$env:NUM_THREADS    = "2"
$env:WINDOW_SEC     = "1.0"
$env:PORT           = "8080"

Write-Host ""
Write-Host "Environment:" -ForegroundColor Cyan
Write-Host "  MODEL_PATH     = $env:MODEL_PATH"
Write-Host "  TOKENIZER_PATH = $env:TOKENIZER_PATH"
Write-Host "  STATIC_DIR     = $env:STATIC_DIR"
Write-Host "  PORT           = $env:PORT"
Write-Host "  NUM_THREADS    = $env:NUM_THREADS"
Write-Host ""
Write-Host "Starting backend (Ctrl+C to stop)..." -ForegroundColor Green
Write-Host "  Frontend:  http://localhost:$env:PORT/" -ForegroundColor White
Write-Host "  Health:    http://localhost:$env:PORT/healthz" -ForegroundColor White
Write-Host "  WebSocket: ws://localhost:$env:PORT/ws" -ForegroundColor White
Write-Host ""
Write-Host "For HTTPS (mic access), run in another terminal:" -ForegroundColor Yellow
Write-Host "  cloudflared tunnel --url http://localhost:$env:PORT" -ForegroundColor White
Write-Host ""

# ----- Run -----
python backend\app.py

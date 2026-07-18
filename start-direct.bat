@echo off
REM ===========================================================================
REM start-direct.bat - Run Hifzapp ASR backend directly (no Docker)
REM ===========================================================================
REM Sets up a Python venv, installs requirements, and runs backend/app.py.
REM No Docker. No image builds. No stuck "Starting" containers.
REM
REM Usage:  start-direct.bat
REM
REM After it starts, run cloudflared in a separate terminal for HTTPS mic:
REM     cloudflared tunnel --url http://localhost:8080
REM ===========================================================================

setlocal enabledelayedexpansion

echo.
echo === Hifzapp direct mode (no Docker) ===
echo.

REM ----- Detect Python -----
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH.
    echo Install Python 3.10+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during install.
    goto :end
)
for /f "tokens=*" %%v in ('python --version') do echo Using %%v

REM ----- Verify model files -----
set "MODEL_DIR=C:\Users\Gufran\Documents\model"
if exist ".env" (
    for /f "usebackq tokens=1,2 delims==" %%a in (".env") do (
        if /i "%%a"=="MODEL_HOST_PATH" set "MODEL_DIR=%%b"
    )
)
if defined MODEL_HOST_PATH set "MODEL_DIR=%MODEL_HOST_PATH%"

echo Checking model files in: %MODEL_DIR%
if not exist "%MODEL_DIR%\model_with_encoder.q8.onnx" (
    echo [ERROR] model_with_encoder.q8.onnx not found
    echo Download it:
    echo     huggingface-cli download Muno459/fastconformer-quran model_with_encoder.q8.onnx --local-dir "%MODEL_DIR%"
    goto :end
)
if not exist "%MODEL_DIR%\tokenizer.model" (
    echo [ERROR] tokenizer.model not found
    echo Download it:
    echo     huggingface-cli download Muno459/fastconformer-quran tokenizer.model --local-dir "%MODEL_DIR%"
    goto :end
)
echo   [OK] model_with_encoder.q8.onnx
echo   [OK] tokenizer.model
echo.

REM ----- Setup venv -----
set "VENV_DIR=.venv"
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Creating Python venv at %VENV_DIR%...
    python -m venv %VENV_DIR%
    if errorlevel 1 (
        echo [ERROR] Failed to create venv
        goto :end
    )
    echo Installing dependencies (this takes 2-3 min the first time)...
    call %VENV_DIR%\Scripts\activate.bat
    python -m pip install --upgrade pip
    REM --only-binary :all: prevents pip from building from source
    REM (which needs Visual Studio Build Tools on Windows).
    pip install --only-binary :all: -r backend\requirements.txt
    if errorlevel 1 (
        echo.
        echo [ERROR] Failed to install dependencies
        echo Common cause: your Python version (likely 3.13) has no
        echo prebuilt wheels for some package. Install Python 3.12 from
        echo https://www.python.org/downloads/ and re-run this script.
        goto :end
    )
) else (
    echo Using existing venv at %VENV_DIR%
    call %VENV_DIR%\Scripts\activate.bat
)

REM ----- Set env vars -----
set "MODEL_PATH=%MODEL_DIR%\model_with_encoder.q8.onnx"
set "TOKENIZER_PATH=%MODEL_DIR%\tokenizer.model"
set "STATIC_DIR=%CD%"
set "NUM_THREADS=2"
set "WINDOW_SEC=2.0"
set "PORT=8080"

REM Point at the .nemo file if present (used to extract CMVN stats).
if exist "%MODEL_DIR%\fastconformer-quran.nemo" (
    set "NEMO_PATH=%MODEL_DIR%\fastconformer-quran.nemo"
) else (
    echo   (no fastconformer-quran.nemo found - CMVN will be skipped)
)

echo.
echo Environment:
echo   MODEL_PATH    = %MODEL_PATH%
echo   TOKENIZER_PATH= %TOKENIZER_PATH%
echo   NEMO_PATH     = %NEMO_PATH%
echo   STATIC_DIR    = %STATIC_DIR%
echo   PORT          = %PORT%
echo   NUM_THREADS   = %NUM_THREADS%
echo.
echo Starting backend (Ctrl+C to stop)...
echo WebSocket: ws://localhost:%PORT%/ws
echo Health:    http://localhost:%PORT%/healthz
echo Frontend:  http://localhost:%PORT%/
echo.

REM ----- Run -----
python backend\app.py

:end
endlocal

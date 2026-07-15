@echo off
REM start-local.bat — quick-start script for Windows.
REM
REM What this does:
REM   1. Builds + starts the FastConformer backend in Docker
REM   2. Waits for it to be healthy
REM   3. Opens the frontend page in your default browser
REM
REM Requirements:
REM   - Docker Desktop for Windows installed and running
REM   - This script in the repo root
REM
REM Usage: double-click start-local.bat, OR from PowerShell:
REM   .\start-local.bat

setlocal

echo.
echo ============================================================
echo   Hifzapp - Local Quick Start
echo ============================================================
echo.

REM Check Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not running. Start Docker Desktop and try again.
    pause
    exit /b 1
)

echo [1/4] Building and starting the backend...
echo (First time will take 5-10 minutes for Python deps; model is mounted from your PC)
echo.

REM Check that the model file exists at the expected location
set MODEL_PATH=C:\Users\Gufran\Downloads\model\nemo\fastconformer-quran.nemo
if not exist "%MODEL_PATH%" (
    echo [ERROR] Model file not found at: %MODEL_PATH%
    echo.
    echo To fix: download the model first with this PowerShell one-liner:
    echo   python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Muno459/fastconformer-quran', filename='nemo/fastconformer-quran.nemo', local_dir='C:/Users/Gufran/Downloads/model')"
    echo.
    echo Or download the public mohammed model:
    echo   python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='mohammed/fastconformer-quran-ar', filename='phase3_full_finetune/phase3_full_finetune_wer0.1432.nemo', local_dir='C:/Users/Gufran/Downloads/model/nemo')"
    echo.
    echo Then re-run this script.
    pause
    exit /b 1
)
echo   Found model at: %MODEL_PATH%
echo.

REM Clean up any stuck/failed containers from previous runs.
REM The container name is fixed (fastconformer) so a second start always
REM needs to stop the old one first. Sometimes it gets stuck.
echo Cleaning up any previous containers...
docker rm -f fastconformer 2>nul
echo.

cd backend

REM Set MODEL_HOST_PATH for docker-compose (path to the dir containing the .nemo file).
REM Tries Windows path first, then WSL path. The container will see this as /data.
set MODEL_HOST_PATH=C:/Users/Gufran/Downloads/model/nemo
echo   Using MODEL_HOST_PATH=%MODEL_HOST_PATH% (host path mounted as /data in container)
echo.

REM Check for HF token (optional, not used since we're mounting the model)
if exist hf_token.txt (
    echo   Found hf_token.txt ^(not needed - using mounted model^)
) else (
    echo   No hf_token.txt - using mounted model from host
)
echo.

docker compose up -d --build

REM Verify the mount worked - the file should be visible inside the container
echo.
echo Verifying model mount inside container...
docker exec fastconformer ls -la /data/fastconformer-quran.nemo 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker compose failed. Check the output above.
    pause
    exit /b 1
)
cd ..

echo.
echo [2/4] Waiting for the backend to be healthy (up to 240s)...
echo.

set /a attempts=0
:wait_loop
set /a attempts+=1
curl -sf http://127.0.0.1:8080/healthz >nul 2>&1
if %errorlevel% equ 0 goto ready

if %attempts% geq 60 (
    echo.
    echo [ERROR] Backend did not become healthy in 180s.
    echo Common causes:
    echo   1. Model is still downloading ^(459MB, can take a few min^)
    echo   2. Model is loading on slow CPU ^(60-90s on Intel UHD^)
    echo   3. HF_TOKEN invalid - check hf_token.txt
    echo.
    echo Check logs with:  cd backend ^&^& docker compose logs -f fastconformer
    pause
    exit /b 1
)

echo   waiting... (%attempts%/80)
timeout /t 3 /nobreak >nul
goto wait_loop

:ready
echo.
echo [3/4] Backend is healthy!
curl -s http://127.0.0.1:8080/healthz
echo.

echo.
echo [4/4] Opening the frontend in your browser...
start "" "http://localhost:8080/"

echo.
echo ============================================================
echo   Ready!
echo ============================================================
echo.
echo Frontend:    http://localhost:8080/  (or the browser tab that just opened)
echo Backend:     http://localhost:8080/healthz
echo WebSocket:   ws://localhost:8080/ws
echo.
echo To see live logs:  cd backend ^&^& docker compose logs -f fastconformer
echo To stop:           cd backend ^&^& docker compose down
echo.
pause

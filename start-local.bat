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
echo (First time will take 5-10 minutes to download deps + model)
echo.

cd backend

REM Check for HF token (used to download the gated Muno459 model)
if exist hf_token.txt (
    echo   Found hf_token.txt - will use the gated Muno459 model ^(best quality^)
) else (
    echo   No hf_token.txt found - will fall back to the public mohammed model
    echo   ^(only works for Alafasy; other reciters won't match well^)
    echo   To use the best model: see backend\hf_token.txt.example
)
echo.

docker compose up -d --build
if %errorlevel% neq 0 (
    echo [ERROR] Docker compose failed. Check the output above.
    pause
    exit /b 1
)
cd ..

echo.
echo [2/4] Waiting for the backend to be healthy (up to 90s)...
echo.

set /a attempts=0
:wait_loop
set /a attempts+=1
curl -sf http://127.0.0.1:8080/healthz >nul 2>&1
if %errorlevel% equ 0 goto ready

if %attempts% geq 30 (
    echo.
    echo [ERROR] Backend did not become healthy in 90s.
    echo Check logs with:  cd backend ^&^& docker compose logs -f fastconformer
    pause
    exit /b 1
)

echo   waiting... (%attempts%/30)
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

@echo off
REM ===========================================================================
REM start-local.bat — One-click local dev start for Hifzapp
REM ===========================================================================
REM Builds and starts the Zipformer Quran ASR container, waits for it to
REM be healthy, then opens the browser.

setlocal enabledelayedexpansion

echo.
echo === Hifzapp local dev start (Zipformer streaming ASR) ===
echo.

REM ----- Check model files -----
if not defined MODEL_HOST_PATH (
    if exist backend\.env (
        for /f "usebackq tokens=1,2 delims==" %%a in ("backend\.env") do (
            if /i "%%a"=="MODEL_HOST_PATH" set MODEL_HOST_PATH=%%b
        )
    )
)
if not defined MODEL_HOST_PATH set MODEL_HOST_PATH=C:\Users\Gufran\Documents\model

echo Checking model files in: %MODEL_HOST_PATH%
if not exist "%MODEL_HOST_PATH%\quran_phoneme_zipformer.int8.onnx" (
    echo.
    echo ERROR: model file not found
    echo   Expected: %MODEL_HOST_PATH%\quran_phoneme_zipformer.int8.onnx
    echo.
    echo Download from https://huggingface.co/Muno459/zipformer_p-quran
    echo and update MODEL_HOST_PATH in backend\.env
    exit /b 1
)
if not exist "%MODEL_HOST_PATH%\tokens.txt" (
    echo.
    echo ERROR: tokens.txt not found
    echo   Expected: %MODEL_HOST_PATH%\tokens.txt
    exit /b 1
)
echo   ^> quran_phoneme_zipformer.int8.onnx OK
echo   ^> tokens.txt OK

REM ----- Clean up old container -----
echo.
echo Cleaning up old container...
docker rm -f zipformer-quran 2^>nul

REM ----- Build and start -----
echo.
echo Building and starting container (this may take 2-5 minutes first time)...
cd backend
docker compose --env-file .env up -d --build
if errorlevel 1 (
    echo.
    echo ERROR: docker compose failed
    cd ..
    exit /b 1
)
cd ..

REM ----- Wait for healthy -----
echo.
echo Waiting for backend to be ready...
set /a attempt=0
:wait_loop
set /a attempt+=1
docker inspect --format="{{.State.Health.Status}}" zipformer-quran 2^>nul | findstr /c:"healthy" ^>nul
if errorlevel 1 (
    if !attempt! lss 60 (
        if !attempt! == 1 (
            echo   Loading model... (this can take 30-90s on first run)
        )
        echo   Still loading... attempt !attempt!/60
        timeout /t 5 /nobreak ^>nul
        goto wait_loop
    )
    echo.
    echo ERROR: backend did not become healthy in 5 minutes
    echo.
    echo Last 30 lines of logs:
    docker logs zipformer-quran --tail 30
    exit /b 1
)

echo.
echo === Backend is healthy ===
echo.
echo   Local:   http://localhost:8080
echo   Health:  http://localhost:8080/healthz
echo.
echo To expose via HTTPS tunnel (access from phone/anywhere):
echo   cloudflared tunnel --url http://localhost:8080
echo.
echo Opening browser...

start http://localhost:8080

endlocal

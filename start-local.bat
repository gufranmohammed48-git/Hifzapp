@echo off
REM ===========================================================================
REM start-local.bat - One-click local dev start for Hifzapp
REM ===========================================================================
REM Builds and starts the Zipformer Quran ASR container, waits for it to
REM be healthy, then opens the browser.
REM
REM Usage:  start-local.bat
REM ===========================================================================

setlocal

echo.
echo === Hifzapp local dev start (Zipformer streaming ASR) ===
echo.

REM ----- Set MODEL_HOST_PATH (allow override via .env or env var) -----
set "MODEL_DIR=C:\Users\Gufran\Documents\model"
if exist "backend\.env" (
    for /f "usebackq tokens=1,2 delims==" %%a in ("backend\.env") do (
        if /i "%%a"=="MODEL_HOST_PATH" set "MODEL_DIR=%%b"
    )
)
if defined MODEL_HOST_PATH set "MODEL_DIR=%MODEL_HOST_PATH%"

echo Checking model file in: %MODEL_DIR%
if not exist "%MODEL_DIR%\fastconformer-quran.nemo" goto :no_model
echo   [OK] fastconformer-quran.nemo

REM ----- Clean up any old container -----
echo.
echo Cleaning up old container...
docker rm -f fastconformer-quran 2>nul
docker rm -f fastconformer 2>nul
docker rm -f zipformer-quran 2>nul

REM ----- Build and start -----
echo.
echo Building and starting container (this may take 2-5 minutes first time)...
cd backend
docker compose --env-file .env up -d --build
set "BUILD_ERR=%errorlevel%"
cd ..

if not "%BUILD_ERR%"=="0" (
    echo.
    echo ERROR: docker compose failed with code %BUILD_ERR%
    exit /b 1
)

REM ----- Wait for healthy -----
echo.
echo Waiting for backend to be ready (max 5 minutes)...
set "MAX_ATTEMPTS=60"
set "ATTEMPT=0"
set "STATUS_FILE=%TEMP%\hifzapp_health.txt"
goto :check_health

:wait_loop
set /a ATTEMPT+=1
if "%ATTEMPT%"=="1" echo   Loading model... (this can take 30-90s on first run)
echo   Still loading... attempt %ATTEMPT%/%MAX_ATTEMPTS%
timeout /t 5 /nobreak >nul

:check_health
REM Capture health status to a temp file (avoids pipe-parsing issues)
docker inspect --format={{.State.Health.Status}} fastconformer > "%STATUS_FILE%" 2>nul
if errorlevel 1 (
    REM Container might be starting, not necessarily an error
    set "STATUS="
) else (
    set /p STATUS=<"%STATUS_FILE%"
)
del "%STATUS_FILE%" 2>nul

if /i "%STATUS%"=="healthy" goto :ready
if %ATTEMPT% LSS %MAX_ATTEMPTS% goto :wait_loop

REM Timeout
echo.
echo ERROR: backend did not become healthy in 5 minutes
echo.
echo Last 40 lines of logs:
docker logs fastconformer --tail 40
exit /b 1

:ready
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
exit /b 0

:no_model
echo.
echo ERROR: model file not found
echo   Expected: %MODEL_DIR%\fastconformer-quran.nemo
echo.
echo If you have the model at a different path, set MODEL_HOST_PATH
echo in backend\.env to point to the directory containing
echo fastconformer-quran.nemo.
exit /b 1

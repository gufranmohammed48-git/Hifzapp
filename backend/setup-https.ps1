# setup-https.ps1 — start the HTTPS-enabled stack and export the self-signed cert.
#
# Run from PowerShell in the Hifzapp repo root:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\backend\setup-https.ps1
#
# This will:
#   1. Start the fastconformer + caddy containers (with the "https" profile)
#   2. Wait for caddy to generate its self-signed cert
#   3. Copy the cert to backend/hifzapp-ca.crt on the host
#   4. Print instructions for installing the cert on other devices

$ErrorActionPreference = 'Stop'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RepoRoot = Split-Path -Parent $ScriptDir
Set-Location $ScriptDir

Write-Host "=== Setting up Hifzapp HTTPS with self-signed cert ===" -ForegroundColor Cyan
Write-Host ""

# 1. Start the HTTPS profile
Write-Host "[1/4] Starting fastconformer + caddy containers..." -ForegroundColor Cyan
docker compose --profile https up -d --build
if ($LASTEXITCODE -ne 0) {
    Write-Host "  [ERROR] docker compose failed" -ForegroundColor Red
    exit 1
}
Write-Host "  Started." -ForegroundColor Green
Write-Host ""

# 2. Wait for caddy to generate the cert
Write-Host "[2/4] Waiting for caddy to generate the self-signed cert..." -ForegroundColor Cyan
$cert = ""
for ($i = 1; $i -le 30; $i++) {
    $cert = docker exec hifzapp-caddy cat /data/caddy/pki/authorities/local/root.crt 2>$null
    if ($cert) {
        Write-Host "  Cert generated (after $i second(s))" -ForegroundColor Green
        break
    }
    Start-Sleep -Seconds 1
}
if (-not $cert) {
    Write-Host "  [ERROR] Cert not generated after 30s" -ForegroundColor Red
    Write-Host "  Check: docker logs hifzapp-caddy" -ForegroundColor Yellow
    exit 1
}
Write-Host ""

# 3. Copy the cert to the host
Write-Host "[3/4] Exporting cert to backend/hifzapp-ca.crt..." -ForegroundColor Cyan
$cert | Out-File -FilePath "$ScriptDir\hifzapp-ca.crt" -Encoding ASCII
Write-Host "  Saved to: $ScriptDir\hifzapp-ca.crt" -ForegroundColor Green
Write-Host ""

# 4. Print instructions
Write-Host "[4/4] === Setup complete! ===" -ForegroundColor Green
Write-Host ""
Write-Host "Your Hifzapp is now available at:" -ForegroundColor Cyan
Write-Host "  https://localhost/       (this PC, mic will work)" -ForegroundColor White
Write-Host ""

# Get IP for LAN access
$ip = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.InterfaceAlias -notlike "*Loopback*" -and $_.IPAddress -notlike "169.254.*" } | Select-Object -First 1).IPAddress
if ($ip) {
    Write-Host "  https://$ip/   (other PCs on this WiFi - see install steps below)" -ForegroundColor White
    Write-Host ""
}

Write-Host "To make other PCs trust this server (so the mic works):" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Windows:" -ForegroundColor Yellow
Write-Host "    1. Copy backend\hifzapp-ca.crt to the other PC" -ForegroundColor White
Write-Host "    2. Double-click it on the other PC" -ForegroundColor White
Write-Host "    3. Click 'Install Certificate...'" -ForegroundColor White
Write-Host "    4. Choose 'Local Machine' (admin) or 'Current User'" -ForegroundColor White
Write-Host "    5. Select 'Place all certificates in the following store'" -ForegroundColor White
Write-Host "    6. Browse -> 'Trusted Root Certification Authorities'" -ForegroundColor White
Write-Host "    7. Finish. Restart Chrome." -ForegroundColor White
Write-Host ""
Write-Host "  Mac:" -ForegroundColor Yellow
Write-Host "    1. Copy hifzapp-ca.crt to the Mac" -ForegroundColor White
Write-Host "    2. Open Keychain Access -> System keychain" -ForegroundColor White
Write-Host "    3. Drag the .crt file in" -ForegroundColor White
Write-Host "    4. Double-click the cert, expand 'Trust', set 'When using this cert' to 'Always Trust'" -ForegroundColor White
Write-Host ""
Write-Host "  iPhone / iPad:" -ForegroundColor Yellow
Write-Host "    1. Email or AirDrop the .crt file to the device" -ForegroundColor White
Write-Host "    2. Open it, go to Settings -> General -> VPN & Device Management" -ForegroundColor White
Write-Host "    3. Tap the cert profile, Install" -ForegroundColor White
Write-Host "    4. Then: Settings -> General -> About -> Certificate Trust Settings" -ForegroundColor White
Write-Host "    5. Enable the toggle for the Hifzapp cert" -ForegroundColor White
Write-Host ""
Write-Host "  Android:" -ForegroundColor Yellow
Write-Host "    1. Copy the .crt to the device" -ForegroundColor White
Write-Host "    2. Settings -> Security -> Install from storage" -ForegroundColor White
Write-Host "    3. Pick the .crt file, name it 'hifzapp'" -ForegroundColor White
Write-Host ""
Write-Host "To stop the HTTPS stack:" -ForegroundColor Cyan
Write-Host "  cd backend; docker compose --profile https down" -ForegroundColor White
Write-Host ""

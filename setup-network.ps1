# One-time setup: allow inbound connections to port 8080 (the Hifzapp backend).
# Run this in PowerShell as Administrator.

Write-Host "=== Opening Windows Firewall for Hifzapp backend (port 8080) ===" -ForegroundColor Cyan

$existing = Get-NetFirewallRule -DisplayName "Hifzapp Backend" -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "  Rule 'Hifzapp Backend' already exists, skipping" -ForegroundColor Yellow
} else {
    New-NetFirewallRule -DisplayName "Hifzapp Backend" -Direction Inbound -LocalPort 8443 -Protocol TCP -Action Allow
    Write-Host "  Created firewall rule" -ForegroundColor Green
}

Write-Host ""
Write-Host "=== Your PC's IP addresses ===" -ForegroundColor Cyan
ipconfig | Select-String "IPv4"
Write-Host ""
Write-Host "Other PCs on your WiFi can now reach your backend at https://<your-ip>:8443/" -ForegroundColor Green
Write-Host "But mic will be blocked by Chrome on non-localhost HTTP. Use cloudflared for HTTPS:" -ForegroundColor Yellow
Write-Host "  cloudflared tunnel --url http://localhost:8080" -ForegroundColor White

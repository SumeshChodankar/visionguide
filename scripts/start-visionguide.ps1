<#
.SYNOPSIS
  Starts everything VisionGuide needs on the laptop, in one go.

.DESCRIPTION
  Tunnel mode (default) - works on UCL Wi-Fi, eduroam or mobile data:
    1. backend (FastAPI, port 8001)               new window "VisionGuide backend"
    2. Cloudflare tunnel to the backend           new window "VisionGuide backend tunnel"
       -> reads its new https://....trycloudflare.com address
       -> writes it into mobile/.env (EXPO_PUBLIC_API_URL)
    3. Expo with its ngrok tunnel                  new window "VisionGuide Expo" (scan its QR code)
  Local mode (-Local) - phone and laptop on the same hotspot:
    backend + Expo on the local network, no tunnels; EXPO_PUBLIC_API_URL is removed from mobile/.env.

  The access key is checked every time: created if missing, and copied so backend/.env and mobile/.env match.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\start-visionguide.ps1
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\start-visionguide.ps1 -Local
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\start-visionguide.ps1 -Stop
#>
param(
    [switch]$Local,        # hotspot / same network: no tunnels
    [switch]$Stop,         # stop the backend, Expo and tunnels, then exit
    [string]$Cloudflared   # path to cloudflared.exe (default: ..\tools\cloudflared\cloudflared.exe next to the repo)
)

$ErrorActionPreference = 'Stop'
# Windows PowerShell 5.1 may default to TLS 1.0/1.1, which Cloudflare refuses.
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
$Repo     = Split-Path -Parent $PSScriptRoot
$Backend  = Join-Path $Repo 'backend'
$Mobile   = Join-Path $Repo 'mobile'
$Python   = Join-Path $Backend '.venv\Scripts\python.exe'
if (-not $Cloudflared) { $Cloudflared = Join-Path (Split-Path -Parent $Repo) 'tools\cloudflared\cloudflared.exe' }
$TunnelLog = Join-Path (Split-Path -Parent $Cloudflared) 'tunnel.log'

function Say([string]$msg, [string]$color = 'Cyan') { Write-Host "==> $msg" -ForegroundColor $color }
function Fail([string]$msg) { Write-Host "XX  $msg" -ForegroundColor Red; exit 1 }

function Stop-Port([int]$port) {
    Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique |
        ForEach-Object { try { Stop-Process -Id $_ -Force -ErrorAction Stop; Say "stopped process $_ on port $port" 'DarkGray' } catch {} }
}

function Stop-All {
    Stop-Port 8001
    Stop-Port 8081
    foreach ($name in 'cloudflared', 'ngrok') {
        Get-Process $name -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.Id -Force; Say "stopped $name ($($_.Id))" 'DarkGray' }
    }
}

# Read / write KEY=value lines in a .env file without touching the other lines.
function Get-EnvValue([string]$file, [string]$name) {
    if (-not (Test-Path $file)) { return $null }
    $line = Get-Content $file | Where-Object { $_ -match "^$name=" } | Select-Object -First 1
    if ($line) { return $line.Substring($name.Length + 1).Trim() } else { return $null }
}
function Set-EnvValue([string]$file, [string]$name, [string]$value) {
    $lines = @(); if (Test-Path $file) { $lines = @(Get-Content $file) }
    $found = $false
    $out = foreach ($l in $lines) {
        if ($l -match "^$name=") { $found = $true; if ($null -ne $value) { "$name=$value" } } else { $l }
    }
    if (-not $found -and $null -ne $value) { $out = @($out) + "$name=$value" }
    # UTF-8 without BOM: Expo's dotenv and Python both read it cleanly.
    [System.IO.File]::WriteAllLines($file, [string[]]@($out), (New-Object System.Text.UTF8Encoding($false)))
}

function Wait-Http([string]$url, [int]$seconds, [hashtable]$headers = @{}) {
    $deadline = (Get-Date).AddSeconds($seconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $r = Invoke-WebRequest $url -UseBasicParsing -TimeoutSec 5 -Headers $headers
            if ($r.StatusCode -eq 200) { return $true }
        } catch {}
        Start-Sleep -Seconds 1
    }
    return $false
}

function Start-Window([string]$title, [string]$dir, [string]$command) {
    $cmd = "`$Host.UI.RawUI.WindowTitle = '$title'; Set-Location -LiteralPath '$dir'; $command"
    Start-Process powershell -ArgumentList @('-NoExit', '-ExecutionPolicy', 'Bypass', '-Command', $cmd) | Out-Null
}

# ---------------------------------------------------------------- stop
if ($Stop) { Say 'Stopping VisionGuide...'; Stop-All; Say 'Stopped.' 'Green'; exit 0 }

# ---------------------------------------------------------------- checks
if (-not (Test-Path $Python)) { Fail "Backend virtual environment not found at $Python. See README 'Quick start'." }
if (-not (Test-Path (Join-Path $Backend '.env'))) { Fail "backend\.env is missing. Copy backend\.env.example to backend\.env and add your keys." }
if (-not (Test-Path (Join-Path $Mobile 'node_modules'))) { Fail "mobile\node_modules missing. Run 'npm install' in the mobile folder first." }
if (-not $Local -and -not (Test-Path $Cloudflared)) { Fail "cloudflared.exe not found at $Cloudflared (use -Cloudflared <path>, or -Local for hotspot mode)." }

$BackendEnv = Join-Path $Backend '.env'
$MobileEnv  = Join-Path $Mobile '.env'

# Access key: same value on both sides (created once if missing).
$key = Get-EnvValue $BackendEnv 'ACCESS_KEY'
if (-not $key) {
    $bytes = New-Object byte[] 24; [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    $key = [Convert]::ToBase64String($bytes).TrimEnd('=').Replace('+', '-').Replace('/', '_')
    Set-EnvValue $BackendEnv 'ACCESS_KEY' $key
    Say 'Created a new access key in backend\.env'
}
if ((Get-EnvValue $MobileEnv 'EXPO_PUBLIC_ACCESS_KEY') -ne $key) {
    Set-EnvValue $MobileEnv 'EXPO_PUBLIC_ACCESS_KEY' $key
    Say 'Copied the access key into mobile\.env'
}

Say 'Stopping any earlier VisionGuide processes...'
Stop-All
Start-Sleep -Seconds 1

# ---------------------------------------------------------------- 1. backend
Say 'Starting the backend (loads YOLO and OCR, ~15 s)...'
Start-Window 'VisionGuide backend' $Backend "& '$Python' -m uvicorn app.main:app --host 0.0.0.0 --port 8001"
if (-not (Wait-Http 'http://127.0.0.1:8001/health' 90)) { Fail 'Backend did not start. Look at the "VisionGuide backend" window for the error.' }
Say 'Backend is up on port 8001.' 'Green'

# ---------------------------------------------------------------- 2. backend tunnel / local address
if ($Local) {
    Set-EnvValue $MobileEnv 'EXPO_PUBLIC_API_URL' $null   # app finds the laptop it was loaded from
    $ip = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.PrefixOrigin -in 'Dhcp', 'Manual' -and $_.IPAddress -notlike '169.*' } | Select-Object -First 1).IPAddress
    $apiUrl = "http://${ip}:8001"
    Say "Local mode: the phone must be on the same network as this laptop ($ip)." 'Yellow'
} else {
    Say 'Starting the Cloudflare tunnel to the backend...'
    if (Test-Path $TunnelLog) { Remove-Item $TunnelLog -Force }
    Start-Window 'VisionGuide backend tunnel' (Split-Path -Parent $Cloudflared) "& '$Cloudflared' tunnel --no-autoupdate --url http://localhost:8001 --logfile '$TunnelLog'"
    $apiUrl = $null
    $deadline = (Get-Date).AddSeconds(60)
    while (-not $apiUrl -and (Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 1
        if (Test-Path $TunnelLog) {
            $m = Select-String -Path $TunnelLog -Pattern 'https://[a-z0-9-]+\.trycloudflare\.com' -AllMatches | Select-Object -Last 1
            if ($m) { $apiUrl = $m.Matches[-1].Value }
        }
    }
    if (-not $apiUrl) { Fail 'No tunnel address after 60 s. Look at the "VisionGuide backend tunnel" window.' }
    # A brand-new trycloudflare address can take 1-2 minutes to resolve.
    Say "Waiting for $apiUrl to become reachable (can take up to 2 minutes)..."
    if (-not (Wait-Http "$apiUrl/health" 180)) { Fail "Tunnel $apiUrl is still not reachable after 3 minutes. Run the script again." }
    if (-not (Wait-Http "$apiUrl/v1/buildings" 20 @{ 'X-VisionGuide-Key' = $key })) { Fail 'The backend refused the access key through the tunnel.' }
    Set-EnvValue $MobileEnv 'EXPO_PUBLIC_API_URL' $apiUrl
    Say "Backend tunnel: $apiUrl (written to mobile\.env)" 'Green'
}

# ---------------------------------------------------------------- 3. Expo
$expoArgs = if ($Local) { 'start --clear' } else { 'start --tunnel ngrok --clear' }
Say 'Starting Expo (first build ~30-60 s)...'
Start-Window 'VisionGuide Expo' $Mobile "npx expo $expoArgs"
if (-not (Wait-Http 'http://127.0.0.1:8081/status' 120)) { Fail 'Expo did not start. Look at the "VisionGuide Expo" window.' }

Write-Host ''
Say 'VisionGuide is running.' 'Green'
Write-Host "    Backend:  $apiUrl"
Write-Host "    Version:  $((Select-String -Path (Join-Path $Mobile 'src\config.ts') -Pattern "BUILD_TAG = '([^']+)'").Matches[0].Groups[1].Value)"
Write-Host ''
Write-Host '    On the phone:'
Write-Host '      1. Settings > Apps > Expo Go > Force stop   (the old app has the old backend address)'
Write-Host '      2. Open Expo Go and scan the QR code in the "VisionGuide Expo" window'
Write-Host '      3. Check the bottom line of the camera view shows the version above'
Write-Host ''
Write-Host '    Stop everything:  powershell -ExecutionPolicy Bypass -File scripts\start-visionguide.ps1 -Stop'

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $projectRoot "model_training\.venv\Scripts\python.exe"
$apiPort = 8765
$apiBase = "http://127.0.0.1:$apiPort"
$apiProcess = $null

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Python environment not found: $pythonExe"
}

$listener = Get-NetTCPConnection -LocalPort $apiPort -State Listen -ErrorAction SilentlyContinue |
    Select-Object -First 1

if (-not $listener) {
    $apiProcess = Start-Process `
        -FilePath $pythonExe `
        -ArgumentList "-m", "diagnostic_platform.api", "--host", "127.0.0.1", "--port", "$apiPort" `
        -WorkingDirectory $projectRoot `
        -WindowStyle Hidden `
        -PassThru
}

$ready = $false
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    Start-Sleep -Milliseconds 300
    try {
        $health = Invoke-RestMethod -Uri "$apiBase/api/health" -TimeoutSec 2
        if ($health.live_api_version -eq 3) {
            $ready = $true
            break
        }
    } catch {
        if ($apiProcess -and $apiProcess.HasExited) { break }
    }
}

if (-not $ready) {
    if ($apiProcess -and -not $apiProcess.HasExited) {
        Stop-Process -Id $apiProcess.Id -Force
    }
    throw "The diagnostic API did not start on $apiBase."
}

try {
    Write-Host "Starting the combined CARLA camera and ABS dashboard..."
    Write-Host "Select the sensor state in the Pygame window, then press Enter."
    Write-Host "CARLA must already be running on port 2000."

    $body = @{ fault_wheel = "none" } | ConvertTo-Json
    Invoke-RestMethod `
        -Uri "$apiBase/api/live/start" `
        -Method Post `
        -ContentType "application/json" `
        -Body $body `
        -TimeoutSec 30 | Out-Null

    while ($true) {
        Start-Sleep -Seconds 1
        $state = Invoke-RestMethod -Uri "$apiBase/api/live/hud" -TimeoutSec 5
        if ($state.status -in @("stopped", "error", "idle")) {
            Write-Host $state.message
            if ($state.status -eq "error") { exit 1 }
            break
        }
    }
} finally {
    if ($apiProcess -and -not $apiProcess.HasExited) {
        Stop-Process -Id $apiProcess.Id -Force
    }
}

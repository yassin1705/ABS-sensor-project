$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $projectRoot "model_training\.venv\Scripts\python.exe"
$siteDirectory = Join-Path $projectRoot "architecture-presentation"
$bundledPnpm = "C:\Users\yasmo\.cache\codex-runtimes\codex-primary-runtime\dependencies\bin\fallback\pnpm.cmd"
$apiPort = 8765
$apiBase = "http://127.0.0.1:$apiPort"

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Python environment not found: $pythonExe"
}

$pnpmExe = if (Test-Path -LiteralPath $bundledPnpm) {
    $bundledPnpm
} else {
    (Get-Command pnpm.cmd -ErrorAction Stop).Source
}

if (-not (Test-Path -LiteralPath (Join-Path $siteDirectory "node_modules"))) {
    Push-Location $siteDirectory
    try {
        & $pnpmExe install --no-frozen-lockfile
        if ($LASTEXITCODE -ne 0) { throw "Dashboard dependency installation failed." }
    } finally {
        Pop-Location
    }
}

$apiProcess = $null
$existingListener = Get-NetTCPConnection `
    -LocalPort $apiPort `
    -State Listen `
    -ErrorAction SilentlyContinue | Select-Object -First 1

if ($existingListener) {
    try {
        $existingHealth = Invoke-RestMethod -Uri "$apiBase/api/health" -TimeoutSec 3
        if ($existingHealth.live_api_version -ne 3) {
            throw "Incompatible API version"
        }
        Invoke-RestMethod -Uri "$apiBase/api/live/state" -TimeoutSec 3 | Out-Null
        Write-Host "Using compatible API already running on port $apiPort."
    } catch {
        $existingPid = $existingListener.OwningProcess
        throw "Port $apiPort is occupied by an outdated diagnostic API (PID $existingPid). Run: Stop-Process -Id $existingPid -Force"
    }
} else {
    $apiProcess = Start-Process `
        -FilePath $pythonExe `
        -ArgumentList "-m", "diagnostic_platform.api", "--host", "127.0.0.1", "--port", "$apiPort" `
        -WorkingDirectory $projectRoot `
        -WindowStyle Hidden `
        -PassThru

    $apiReady = $false
    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        Start-Sleep -Milliseconds 250
        try {
            Invoke-RestMethod -Uri "$apiBase/api/live/state" -TimeoutSec 2 | Out-Null
            $apiReady = $true
            break
        } catch {
            if ($apiProcess.HasExited) { break }
        }
    }
    if (-not $apiReady) {
        if ($apiProcess -and -not $apiProcess.HasExited) {
            Stop-Process -Id $apiProcess.Id -Force
        }
        throw "The live diagnostic API did not start correctly on $apiBase."
    }
}

try {
    Write-Host ""
    Write-Host "ABS diagnostic platform"
    Write-Host "Dashboard : http://localhost:3000"
    Write-Host "Model API : $apiBase"
    Write-Host "CARLA live: start Town04 with -dx11 first, then use Start live drive in the dashboard."
    Write-Host "Use Ctrl+C to stop both services."
    Write-Host ""

    Start-Process powershell.exe `
        -ArgumentList "-NoProfile", "-WindowStyle", "Hidden", "-Command", "Start-Sleep -Seconds 3; Start-Process 'http://localhost:3000'" `
        -WindowStyle Hidden

    Push-Location $siteDirectory
    try {
        & $pnpmExe run dev
    } finally {
        Pop-Location
    }
} finally {
    if ($apiProcess -and -not $apiProcess.HasExited) {
        Stop-Process -Id $apiProcess.Id -Force
    }
}

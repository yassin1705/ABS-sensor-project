param(
    [ValidateSet("combined", "separate")]
    [string]$ViewMode = "combined"
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = if ($env:PFA_PYTHON_EXE) {
    $env:PFA_PYTHON_EXE
} else {
    Join-Path $projectRoot "model_training\.venv\Scripts\python.exe"
}
if (-not $env:CARLA_ROOT) {
    throw "CARLA_ROOT is not set. Point it to the CARLA 0.9.16 installation directory."
}
$carlaRoot = $env:CARLA_ROOT
$carlaExe = Join-Path $carlaRoot "CarlaUE4.exe"
$apiPort = if ($env:PFA_API_PORT) { [int]$env:PFA_API_PORT } else { 8765 }
$apiBase = "http://127.0.0.1:$apiPort"
$apiProcess = $null
$carlaProcess = $null
$apiReady = $false

function Stop-ProjectPythonProcesses {
    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name -match '^python(?:\.exe)?$' -and
            $_.CommandLine -match 'diagnostic_platform\.api|carla_live_driver\.py'
        } |
        ForEach-Object {
            Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        }
}

function Stop-CarlaProcesses {
    Get-Process -Name "CarlaUE4", "CarlaUE4-Win64-Shipping" -ErrorAction SilentlyContinue |
        Stop-Process -Force -ErrorAction SilentlyContinue
}

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Python environment not found: $pythonExe"
}
if (-not (Test-Path -LiteralPath $carlaExe)) {
    throw "CARLA executable not found: $carlaExe"
}

try {
    Write-Host "Cleaning up stale CARLA and diagnostic processes..."
    Stop-ProjectPythonProcesses
    Stop-CarlaProcesses
    Start-Sleep -Seconds 1

    $carlaPortOwner = Get-NetTCPConnection -LocalPort 2000 -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($carlaPortOwner) {
        throw "Port 2000 is still occupied by PID $($carlaPortOwner.OwningProcess)."
    }

    $apiPortOwner = Get-NetTCPConnection -LocalPort $apiPort -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($apiPortOwner) {
        throw "Port $apiPort is still occupied by PID $($apiPortOwner.OwningProcess)."
    }

    Write-Host "Starting one CARLA server on Town04 ($ViewMode view)..."
    $carlaArguments = @(
        "/Game/Carla/Maps/Town04",
        "-quality-level=Low",
        "-dx11"
    )
    if ($ViewMode -eq "combined") {
        $carlaArguments += "-RenderOffScreen"
    } else {
        $carlaArguments += @("-windowed", "-ResX=960", "-ResY=540")
    }
    $carlaProcess = Start-Process `
        -FilePath $carlaExe `
        -ArgumentList $carlaArguments `
        -WorkingDirectory $carlaRoot `
        -PassThru

    $carlaReady = $false
    for ($attempt = 0; $attempt -lt 120; $attempt++) {
        Start-Sleep -Milliseconds 500
        $listener = Get-NetTCPConnection -LocalPort 2000 -State Listen -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($listener) {
            $carlaReady = $true
            break
        }
        if ($carlaProcess.HasExited) { break }
    }
    if (-not $carlaReady) {
        throw "CARLA did not become ready on 127.0.0.1:2000 within 60 seconds."
    }

    Write-Host "CARLA is listening on port 2000. Starting the diagnostic API..."
    $apiProcess = Start-Process `
        -FilePath $pythonExe `
        -ArgumentList "-m", "diagnostic_platform.api", "--host", "127.0.0.1", "--port", "$apiPort" `
        -WorkingDirectory $projectRoot `
        -WindowStyle Hidden `
        -PassThru

    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        Start-Sleep -Milliseconds 300
        try {
            $health = Invoke-RestMethod -Uri "$apiBase/api/health" -TimeoutSec 2
            if ($health.live_api_version -eq 3) {
                $apiReady = $true
                break
            }
        } catch {
            if ($apiProcess.HasExited) { break }
        }
    }
    if (-not $apiReady) {
        throw "The diagnostic API did not start on $apiBase."
    }

    Write-Host "Starting the CARLA Pygame ABS dashboard in $ViewMode mode..."

    $body = @{ fault_wheel = "none"; view_mode = $ViewMode } | ConvertTo-Json
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
            if ($state.status -eq "error") { throw $state.message }
            break
        }

        $carlaShipping = Get-Process -Name "CarlaUE4-Win64-Shipping" -ErrorAction SilentlyContinue
        if (-not $carlaShipping) {
            throw "The CARLA server closed unexpectedly."
        }
    }
} finally {
    Write-Host "Stopping the Pygame driver, diagnostic API, and CARLA..."
    if ($apiReady) {
        try {
            Invoke-RestMethod -Uri "$apiBase/api/live/stop" -Method Post -ContentType "application/json" -Body "{}" -TimeoutSec 3 | Out-Null
        } catch {
            # Cleanup continues even if the local API is already gone.
        }
        Start-Sleep -Seconds 2
    }

    Stop-ProjectPythonProcesses
    if ($apiProcess -and -not $apiProcess.HasExited) {
        Stop-Process -Id $apiProcess.Id -Force -ErrorAction SilentlyContinue
    }
    Stop-CarlaProcesses
    Write-Host "Session cleanup complete."
}

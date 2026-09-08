param(
    [ValidateSet("combined", "separate")]
    [string]$ViewMode = "combined"
)

$ErrorActionPreference = "Stop"
$launcher = Join-Path $PSScriptRoot "start_carla_dashboard.ps1"
& $launcher -ViewMode $ViewMode
exit $LASTEXITCODE

@echo off
setlocal
cd /d "%~dp0"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_carla_dashboard.ps1"

if errorlevel 1 (
    echo.
    echo The CARLA diagnostic window stopped with an error.
    pause
)

endlocal

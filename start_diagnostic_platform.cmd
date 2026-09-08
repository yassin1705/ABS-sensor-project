@echo off
setlocal

rem Default: one combined CARLA camera and diagnostic dashboard window.
rem Fallback: run this file with "separate" to restore the previous two-window setup.
if /I "%~1"=="separate" (
    call "%~dp0start_carla_dashboard.cmd" -ViewMode separate
) else (
    call "%~dp0start_carla_dashboard.cmd" -ViewMode combined
)

endlocal

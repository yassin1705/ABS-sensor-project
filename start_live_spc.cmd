@echo off
setlocal
cd /d "%~dp0"

if not exist "model_training\.venv\Scripts\python.exe" (
    echo Python environment not found: model_training\.venv
    pause
    exit /b 1
)

"model_training\.venv\Scripts\python.exe" ^
    -m model_training.realtime_spc_replay ^
    --playback-speed 1 ^
    %*

if errorlevel 1 pause
endlocal

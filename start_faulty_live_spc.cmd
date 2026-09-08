@echo off
setlocal
cd /d "%~dp0"

if not exist "model_training\.venv\Scripts\python.exe" (
    echo Python environment not found: model_training\.venv
    pause
    exit /b 1
)

set "FAULTY_CSV="
for /f "delims=" %%F in ('dir /b /a-d /o-d "ABS_SoH_Simulator\simulation_results\abs_faulty_braking_dataset_*.csv" 2^>nul') do (
    if not defined FAULTY_CSV set "FAULTY_CSV=ABS_SoH_Simulator\simulation_results\%%F"
)

set "FAULTY_MANIFEST=%FAULTY_CSV:dataset=manifest%"

if not defined FAULTY_CSV (
    echo No faulty braking dataset was found in ABS_SoH_Simulator\simulation_results.
    pause
    exit /b 1
)

if not exist "%FAULTY_MANIFEST%" (
    echo Matching faulty manifest not found: %FAULTY_MANIFEST%
    pause
    exit /b 1
)

echo Replaying faulty dataset: %FAULTY_CSV%
"model_training\.venv\Scripts\python.exe" ^
    -m model_training.realtime_spc_replay ^
    --dataset-csv "%FAULTY_CSV%" ^
    --fault-manifest "%FAULTY_MANIFEST%" ^
    --split any ^
    --fault-aware ^
    --playback-speed 1 ^
    %*

if errorlevel 1 pause
endlocal

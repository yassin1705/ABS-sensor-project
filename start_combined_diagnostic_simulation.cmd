@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=model_training\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    echo Python environment not found: %PYTHON_EXE%
    pause
    exit /b 1
)

"%PYTHON_EXE%" -m fault_parameter_training.combined_simulation %*
if errorlevel 1 pause
endlocal

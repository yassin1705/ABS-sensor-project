@echo off
setlocal
cd /d "%~dp0"

set "JUPYTER_EXE=model_training\.venv\Scripts\jupyter-lab.exe"
if not exist "%JUPYTER_EXE%" (
    echo Jupyter Lab not found: %JUPYTER_EXE%
    exit /b 1
)

"%JUPYTER_EXE%" fault_parameter_training\notebooks\train_models.ipynb
endlocal

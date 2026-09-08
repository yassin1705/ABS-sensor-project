@echo off
setlocal
cd /d "%~dp0"

echo The mixed healthy/faulty benchmark is managed by the notebook.
call start_fault_parameter_notebook.cmd
endlocal

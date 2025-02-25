@echo off
cd /d %~dp0

set PYTHON_PATH=%~dp0python
set SCRIPT_PATH=%~dp0predict.py

set PATH=%PYTHON_PATH%;%PATH%

set TCL_LIBRARY=%~dp0python\tcl\tcl8.6
set TK_LIBRARY=%~dp0python\tcl\tk8.6

if not exist "%PYTHON_PATH%\python.exe" (
    echo [ERROR] Python executable not found at %PYTHON_PATH%\python.exe
    echo [INFO] Please ensure the embeddable Python package is extracted to 'python' in this folder
    pause
    exit /b 1
)

if not exist "%~dp0site-packages" (
    echo [ERROR] Libraries folder not found at %~dp0site-packages
    echo [INFO] Please ensure the 'site-packages' folder is in this directory
    pause
    exit /b 1
)

if not exist "%SCRIPT_PATH%" (
    echo [ERROR] predict.py not found at %SCRIPT_PATH%
    echo [INFO] Please ensure 'predict.py' is in this directory
    pause
    exit /b 1
)

echo [INFO] Running predict.py...
"%PYTHON_PATH%\python.exe" "%SCRIPT_PATH%"

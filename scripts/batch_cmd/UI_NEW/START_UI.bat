@echo off
setlocal
set "PROJECT_ROOT=%~dp0..\..\.."
for %%I in ("%PROJECT_ROOT%") do set "PROJECT_ROOT=%%~fI"
cd /d "%PROJECT_ROOT%"

set "PYTHON_EXE="
if exist ".venv312\Scripts\python.exe" (
  set "PYTHON_EXE=.venv312\Scripts\python.exe"
) else if exist ".venv\Scripts\python.exe" (
  set "PYTHON_EXE=.venv\Scripts\python.exe"
)

if "%PYTHON_EXE%"=="" (
  echo [ERROR] Python virtual environment was not found. Please run SETUP_ENV.bat first.
  pause
  exit /b 1
)

echo [UI] Starting Satellite_UI...
"%PYTHON_EXE%" "%PROJECT_ROOT%\source\python\UI_NEW\main.py"
pause

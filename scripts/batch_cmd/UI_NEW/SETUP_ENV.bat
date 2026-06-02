@echo off
setlocal
set "PROJECT_ROOT=%~dp0..\..\.."
for %%I in ("%PROJECT_ROOT%") do set "PROJECT_ROOT=%%~fI"
cd /d "%PROJECT_ROOT%"

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python was not found. Please install Python 3.10-3.12 and add it to PATH.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [SETUP] Creating virtual environment...
  python -m venv .venv
)

echo [SETUP] Upgrading pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip

echo [SETUP] Installing UI dependencies...
".venv\Scripts\python.exe" -m pip install -r requirements.txt

echo.
echo [OK] Environment is ready. Run START_UI.bat to launch the platform.
pause

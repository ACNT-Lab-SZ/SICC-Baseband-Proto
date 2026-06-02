@echo off
setlocal
set "PROJECT_ROOT=%~dp0..\..\.."
for %%I in ("%PROJECT_ROOT%") do set "PROJECT_ROOT=%%~fI"
cd /d "%PROJECT_ROOT%"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] .venv was not found. Please run SETUP_ENV.bat first.
  pause
  exit /b 1
)

echo [UI] Starting Satellite_UI...
".venv\Scripts\python.exe" "%PROJECT_ROOT%\source\python\Satellite_UI\main.py"
pause

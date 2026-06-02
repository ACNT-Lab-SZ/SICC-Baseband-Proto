@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] .venv was not found. Please run SETUP_ENV.bat first.
  pause
  exit /b 1
)

echo [DEMO] Sending demo NTN/channel metrics to UDP 65436...
".venv\Scripts\python.exe" "%~dp0examples\send_metric_demo.py"
pause

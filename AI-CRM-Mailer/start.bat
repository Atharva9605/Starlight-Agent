@echo off
REM ============================================================
REM  Starlight AI-CRM Mailer — Local launcher (Windows)
REM ============================================================
cd /d "%~dp0"

echo.
echo ╔══════════════════════════════════════════════╗
echo ║     Starlight AI-CRM Mailer  — Startup      ║
echo ╚══════════════════════════════════════════════╝
echo.

REM 1. Check .env
if not exist ".env" (
  echo ERROR: .env not found. Copy .env.example to .env and fill in your keys.
  pause
  exit /b 1
)

REM 2. Create static image directory
if not exist "static\page_images" mkdir "static\page_images"

REM 3. Start static file server in a new window (for product images in emails)
echo [START] Static image server on http://localhost:8000 ...
start "Starlight-Static" cmd /k "cd /d %~dp0 && python -m http.server 8000 --directory static"

REM Wait 1 second
timeout /t 1 /nobreak >nul

REM 4. Start Streamlit in a new window
echo [START] Streamlit app on http://localhost:8501 ...
start "Starlight-App" cmd /k "cd /d %~dp0 && streamlit run app.py --server.port 8501"

echo.
echo  Both processes started in separate windows.
echo  Streamlit UI:  http://localhost:8501
echo  Static images: http://localhost:8000
echo.
pause

@echo off
echo Starting Frontend Development Server...
echo.
echo Frontend will be available at: http://localhost:5173
echo Make sure backend is running on: http://127.0.0.1:8001
echo.

cd /d "%~dp0\frontend"
npm run dev
pause
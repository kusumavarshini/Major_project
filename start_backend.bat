@echo off
echo Starting Fingerprint Recognition Backend...
echo.
echo SecuGen WebAPI: http://localhost:8000
echo FastAPI Backend: http://127.0.0.1:8001  
echo Frontend Dev: http://localhost:5173
echo.

cd /d "%~dp0"
call .venv\Scripts\activate.bat
python backend_api.py
pause
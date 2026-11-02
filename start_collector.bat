@echo off
title FingerSure Fingerprint Dataset Collector
cd /d "%~dp0"
echo ============================================================
echo   Starting FingerSure Terminal Fingerprint Dataset Collector
echo ============================================================
echo.

if exist ".venv\Scripts\activate.bat" (
    echo Activating project virtual environment...
    call .venv\Scripts\activate.bat
)

python collect_fingerprints.py
pause


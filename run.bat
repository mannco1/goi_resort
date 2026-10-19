@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (echo Сначала запустите setup.bat & pause & exit /b 1)
.venv\Scripts\python.exe -m interview_helper --config config.toml %*
if errorlevel 1 pause

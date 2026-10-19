@echo off
chcp 65001 >nul
cd /d "%~dp0"
where py >nul 2>nul || (echo Установите Python 3.11+ с python.org и отметьте "Add to PATH" & pause & exit /b 1)
if not exist .venv py -3 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if not exist config.toml copy config.example.toml config.toml >nul
echo.
echo Готово. Заполните profile\resume.md и profile\vacancy.md, задайте ключ:
echo    setx ANTHROPIC_API_KEY "sk-ant-..."
echo и запустите run.bat
pause

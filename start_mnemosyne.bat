@echo off
chcp 65001 > nul
echo ========================================
echo  Mnemosyne - Digital Life Archival System
echo ========================================
echo.

REM --- CHECK PYTHON 3.11 ---
for /f "tokens=2" %%v in ('py -3.11 --version 2^>nul') do set PY311=%%v

if "%PY311%"=="" (
    echo ERROR: Python 3.11 is not installed.
    echo Please install Python 3.11 from python.org and try again.
    pause
    exit /b 1
)

REM --- CREATE VENV IF NOT EXISTS ---
if not exist "venv\Scripts\activate.bat" (
    echo Creating virtual environment with Python 3.11...
    py -3.11 -m venv venv
)

REM --- ACTIVATE VENV ---
call venv\Scripts\activate.bat

REM --- INSTALL DEPENDENCIES ONLY IF NOT INSTALLED ---
if not exist "venv\installed.flag" (
    echo Installing base dependencies...
    pip install --upgrade pip

    echo Installing requirements.txt...
    pip install -r requirements.txt

    echo Installing PyTorch manually...
    pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

    echo Dependencies successfully installed!
    echo DO NOT DELETE THIS FILE > venv\installed.flag
)

REM --- CHECK OLLAMA ---
curl http://localhost:11434/api/tags > nul 2>&1
if errorlevel 1 (
    echo Ollama is not running. Starting Ollama...
    start "" "C:\Program Files\Ollama\ollama app.exe"
    timeout /t 10 /nobreak > nul
)

REM --- CREATE NEEDED DIRECTORIES ---
if not exist "data" mkdir data
if not exist "logs" mkdir logs
if not exist "thumbnails" mkdir thumbnails

REM --- START MNEMOSYNE ---
echo Starting Mnemosyne...
python main.py

pause

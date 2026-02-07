@echo off
setlocal enabledelayedexpansion

echo ============================================
echo   Lore Compendium - Easy Setup Script
echo ============================================
echo.

REM Check if Python is installed
echo [1/6] Checking for Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH!
    echo.
    echo Please install Python 3.13 from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation!
    echo.
    pause
    exit /b 1
)
echo [OK] Python is installed.
python --version
echo.

REM Check if Ollama is installed
echo [2/6] Checking for Ollama installation...
ollama --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Ollama is not installed or not in PATH!
    echo.
    echo Please install Ollama from: https://ollama.com/download/windows
    echo After installation, restart your computer and run this setup again.
    echo.
    pause
    exit /b 1
)
echo [OK] Ollama is installed.
ollama --version
echo.

REM Create virtual environment if it doesn't exist
echo [3/6] Setting up Python virtual environment...
if not exist ".venv" (
    echo Creating new virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
) else (
    echo [OK] Virtual environment already exists.
)
echo.

REM Activate virtual environment and install dependencies
echo [4/6] Installing Python dependencies...
echo This may take a few minutes...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install Python dependencies!
    pause
    exit /b 1
)
echo [OK] Python dependencies installed.
echo.

REM Pull Ollama models
echo [5/6] Downloading AI models (this may take 10-20 minutes)...
echo.
echo Downloading and creating custom models...
cd modelfiles

echo   - Creating gpt-oss model...
ollama create -f gpt-oss-20b-modelfile.txt gpt-oss
if errorlevel 1 (
    echo [WARNING] Failed to create gpt-oss model!
)

echo   - Creating llama3.2 model...
ollama create -f llama3.2-modelfile.txt llama3.2
if errorlevel 1 (
    echo [WARNING] Failed to create llama3.2 model!
)

echo   - Initializing gpt-oss model...
echo /bye | ollama run gpt-oss >nul 2>&1

echo   - Initializing llama3.2 model...
echo /bye | ollama run llama3.2 >nul 2>&1

echo   - Downloading embedding model...
ollama pull mxbai-embed-large
if errorlevel 1 (
    echo [WARNING] Failed to download embedding model!
)

cd ..
echo [OK] AI models downloaded.
echo.

REM Create input folder if it doesn't exist
if not exist "input" (
    mkdir input
    echo Sample documents folder created: input\
)

REM Run config wizard
echo [6/6] Setting up configuration...
echo.
if not exist "config.json" (
    echo No config.json found. Running configuration wizard...
    python config_wizard.py
    if errorlevel 1 (
        echo [WARNING] Configuration wizard failed or was cancelled.
        echo You'll need to create config.json manually before running the bot.
    )
) else (
    echo [OK] config.json already exists.
    set /p RECONFIG="Do you want to reconfigure? (y/n): "
    if /i "!RECONFIG!"=="y" (
        python config_wizard.py
    )
)
echo.

echo ============================================
echo   Setup Complete!
echo ============================================
echo.
echo Next steps:
echo   1. Add your documents to the 'input' folder
echo   2. Double-click 'start.bat' to run the bot
echo.
echo For help, see BEGINNER_GUIDE.md
echo.
pause

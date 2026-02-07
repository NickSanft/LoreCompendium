@echo off
setlocal enabledelayedexpansion

echo ============================================
echo   Starting Lore Compendium Discord Bot
echo ============================================
echo.

REM Check if virtual environment exists
if not exist ".venv" (
    echo [ERROR] Virtual environment not found!
    echo.
    echo Please run setup.bat first to install the bot.
    echo.
    pause
    exit /b 1
)

REM Check if config.json exists
if not exist "config.json" (
    echo [ERROR] config.json not found!
    echo.
    echo You need to configure the bot before running it.
    echo Running configuration wizard...
    echo.
    call .venv\Scripts\activate.bat
    python config_wizard.py
    if errorlevel 1 (
        echo.
        echo [ERROR] Configuration failed!
        echo Please run config_wizard.py manually or create config.json.
        echo.
        pause
        exit /b 1
    )
)

REM Check if Ollama is running
echo Checking Ollama service...
ollama list >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Ollama is not running!
    echo.
    echo Please make sure Ollama is installed and running.
    echo Ollama usually runs automatically on Windows.
    echo If not, you may need to restart your computer.
    echo.
    pause
    exit /b 1
)
echo [OK] Ollama is running.
echo.

REM Create input folder if it doesn't exist
if not exist "input" (
    mkdir input
    echo Created 'input' folder for your documents.
    echo.
)

REM Activate virtual environment
echo Activating Python environment...
call .venv\Scripts\activate.bat

REM Start the bot
echo.
echo ============================================
echo   Bot is starting...
echo ============================================
echo.
echo The bot will process documents in the 'input' folder.
echo Press Ctrl+C to stop the bot at any time.
echo.
echo --- Bot Output ---
echo.

python discord_main.py

REM If the bot exits with an error
if errorlevel 1 (
    echo.
    echo ============================================
    echo   Bot stopped with an error!
    echo ============================================
    echo.
    echo Common issues:
    echo   - Invalid Discord bot token in config.json
    echo   - Bot doesn't have proper permissions
    echo   - Network connectivity issues
    echo   - Ollama models not properly installed
    echo.
    echo Check BEGINNER_GUIDE.md for troubleshooting help.
    echo.
    pause
    exit /b 1
)

echo.
echo Bot stopped normally.
pause

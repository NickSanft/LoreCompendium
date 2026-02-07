#!/bin/bash

echo "============================================"
echo "  Starting Lore Compendium Discord Bot"
echo "============================================"
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "[ERROR] Virtual environment not found!"
    echo ""
    echo "Please run ./setup.sh first to install the bot."
    echo ""
    read -p "Press Enter to exit..."
    exit 1
fi

# Check if config.json exists
if [ ! -f "config.json" ]; then
    echo "[ERROR] config.json not found!"
    echo ""
    echo "You need to configure the bot before running it."
    echo "Running configuration wizard..."
    echo ""
    source .venv/bin/activate
    python3 config_wizard.py
    if [ $? -ne 0 ]; then
        echo ""
        echo "[ERROR] Configuration failed!"
        echo "Please run python3 config_wizard.py manually or create config.json."
        echo ""
        read -p "Press Enter to exit..."
        exit 1
    fi
fi

# Check if Ollama is running
echo "Checking Ollama service..."
if ! pgrep -x "ollama" > /dev/null; then
    echo "[WARNING] Ollama service is not running!"
    echo "Attempting to start Ollama..."
    nohup ollama serve > /dev/null 2>&1 &
    sleep 3
fi

# Verify Ollama is responding
ollama list > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "[ERROR] Ollama is not responding!"
    echo ""
    echo "Please make sure Ollama is installed and running."
    echo "Try running: ollama serve"
    echo ""
    read -p "Press Enter to exit..."
    exit 1
fi
echo "[OK] Ollama is running."
echo ""

# Create input folder if it doesn't exist
if [ ! -d "input" ]; then
    mkdir input
    echo "Created 'input' folder for your documents."
    echo ""
fi

# Activate virtual environment
echo "Activating Python environment..."
source .venv/bin/activate

# Start the bot
echo ""
echo "============================================"
echo "  Bot is starting..."
echo "============================================"
echo ""
echo "The bot will process documents in the 'input' folder."
echo "Press Ctrl+C to stop the bot at any time."
echo ""
echo "--- Bot Output ---"
echo ""

python3 discord_main.py

# Check exit code
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "============================================"
    echo "  Bot stopped with an error!"
    echo "============================================"
    echo ""
    echo "Common issues:"
    echo "  - Invalid Discord bot token in config.json"
    echo "  - Bot doesn't have proper permissions"
    echo "  - Network connectivity issues"
    echo "  - Ollama models not properly installed"
    echo ""
    echo "Check BEGINNER_GUIDE.md for troubleshooting help."
    echo ""
    read -p "Press Enter to exit..."
    exit $EXIT_CODE
fi

echo ""
echo "Bot stopped normally."
read -p "Press Enter to exit..."

#!/bin/bash

echo "============================================"
echo "  Lore Compendium - Easy Setup Script"
echo "============================================"
echo ""

# Check if Python is installed
echo "[1/6] Checking for Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed!"
    echo ""
    echo "Please install Python 3.13 from:"
    echo "  macOS: brew install python@3.13"
    echo "  Linux: sudo apt install python3.13 (or your distro's package manager)"
    echo ""
    read -p "Press Enter to exit..."
    exit 1
fi
echo "[OK] Python is installed."
python3 --version
echo ""

# Check if Ollama is installed
echo "[2/6] Checking for Ollama installation..."
if ! command -v ollama &> /dev/null; then
    echo "[ERROR] Ollama is not installed!"
    echo ""
    echo "Please install Ollama from: https://ollama.com"
    echo "  macOS: brew install ollama"
    echo "  Linux: curl -fsSL https://ollama.com/install.sh | sh"
    echo ""
    read -p "Press Enter to exit..."
    exit 1
fi
echo "[OK] Ollama is installed."
ollama --version
echo ""

# Create virtual environment if it doesn't exist
echo "[3/6] Setting up Python virtual environment..."
if [ ! -d ".venv" ]; then
    echo "Creating new virtual environment..."
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to create virtual environment!"
        read -p "Press Enter to exit..."
        exit 1
    fi
    echo "[OK] Virtual environment created."
else
    echo "[OK] Virtual environment already exists."
fi
echo ""

# Activate virtual environment and install dependencies
echo "[4/6] Installing Python dependencies..."
echo "This may take a few minutes..."
source .venv/bin/activate
python -m pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install Python dependencies!"
    read -p "Press Enter to exit..."
    exit 1
fi
echo "[OK] Python dependencies installed."
echo ""

# Start Ollama service if not running
echo "[5/6] Checking Ollama service..."
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama service in background..."
    nohup ollama serve > /dev/null 2>&1 &
    sleep 3
fi
echo "[OK] Ollama service is running."
echo ""

# Pull Ollama models
echo "[5/6] Downloading AI models (this may take 10-20 minutes)..."
echo ""
echo "Downloading and creating custom models..."
cd modelfiles

echo "  - Creating gpt-oss model..."
ollama create -f gpt-oss-20b-modelfile.txt gpt-oss
if [ $? -ne 0 ]; then
    echo "[WARNING] Failed to create gpt-oss model!"
fi

echo "  - Creating llama3.2 model..."
ollama create -f llama3.2-modelfile.txt llama3.2
if [ $? -ne 0 ]; then
    echo "[WARNING] Failed to create llama3.2 model!"
fi

echo "  - Initializing gpt-oss model..."
echo "/bye" | ollama run gpt-oss > /dev/null 2>&1

echo "  - Initializing llama3.2 model..."
echo "/bye" | ollama run llama3.2 > /dev/null 2>&1

echo "  - Downloading embedding model..."
ollama pull mxbai-embed-large
if [ $? -ne 0 ]; then
    echo "[WARNING] Failed to download embedding model!"
fi

cd ..
echo "[OK] AI models downloaded."
echo ""

# Create input folder if it doesn't exist
if [ ! -d "input" ]; then
    mkdir input
    echo "Sample documents folder created: input/"
fi

# Run config wizard
echo "[6/6] Setting up configuration..."
echo ""
if [ ! -f "config.json" ]; then
    echo "No config.json found. Running configuration wizard..."
    python3 config_wizard.py
    if [ $? -ne 0 ]; then
        echo "[WARNING] Configuration wizard failed or was cancelled."
        echo "You'll need to create config.json manually before running the bot."
    fi
else
    echo "[OK] config.json already exists."
    read -p "Do you want to reconfigure? (y/n): " RECONFIG
    if [[ "$RECONFIG" =~ ^[Yy]$ ]]; then
        python3 config_wizard.py
    fi
fi
echo ""

echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Add your documents to the 'input' folder"
echo "  2. Run './start.sh' to start the bot"
echo ""
echo "For help, see BEGINNER_GUIDE.md"
echo ""
read -p "Press Enter to exit..."

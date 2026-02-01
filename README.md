# Lore Compendium - Discord AI Chatbot

Lore Compendium is an AI-powered Discord bot that can provide conversational responses with document retrieval.
The primary purpose is to provide anyone with their own documents (lore, books, etc) and be able to semantically seacrch
them.

## Features

- **Conversational AI**: Witty, butler-like responses powered by local LLM models via Ollama
- **Document Search**: RAG (Retrieval-Augmented Generation) system for querying local documents (Word docs & PDFs)
- **Discord Commands**:
    - `$lore <query>` - Search local documents
- Direct messages or mentions trigger conversational responses that can use any of the below tools. You can also upload
  a voice message using Discord mobile as well.
- **Tools**: Dice rolling, current time lookup, web search, document search, memory retrieval, image generation, and
  image recognition (limited to one attachment per message)

## Architecture

The bot uses a LangGraph-based agent system with:

- **Conversation Node**: Handles user queries with tool access
- **ChromaDB**: Stores user memories and document embeddings

Document Engine Diagram:

![Document Engine Diagram](document_engine_diagram.png)

## Prerequisites

### 1. Install Python

- Python 3.13 is recommended
- Download from [python.org](https://www.python.org/downloads/)

### 2. Install Ollama

**Windows:**

1. Download the Ollama installer from [ollama.com](https://ollama.com/download/windows)
2. Run the installer and follow the prompts
3. Ollama will run as a background service

**macOS:**

```bash
# Using Homebrew
brew install ollama

# Or download from ollama.com/download/mac
```

**Linux:**

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 3. Pull Required Ollama Models

After installing Ollama, pull the models used by Mister Fritz:

Windows:

Run ./modelfiles/run.bat

Linux/macOS:

```bash
ollama create -f .\gpt-oss-20b-modelfile.txt gpt-oss
ollama create -f .\llama3.2-modelfile.txt llama3.2

ollama run gpt-oss /bye
ollama run llama3.2 /bye

ollama pull mxbai-embed-large
```

## Installation

1. **Clone the repository:**

```bash
git clone <repository-url>
cd LoreCompendium
```

2. **Create a virtual environment:**

```bash
python -m venv .venv
```

3. **Activate the virtual environment OR install with pip:**

Virtual environment:

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Pip installation:

```bash
pip install -r requirements.txt
```

4. **Configure the bot:**

Create a `config.json` file in the project root with the following structure:

```json
{
  "discord_bot_token": "YOUR_DISCORD_BOT_TOKEN",
  "role_description": "How you want the conversational agent to behave (by default, an english butler)"
}
```

To get a Discord bot token:

1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Create a new application
3. Go to the "Bot" section
4. Click "Reset Token" to generate a token
5. Enable "Message Content Intent" under Privileged Gateway Intents
6. Invite the bot to your server using the OAuth2 URL generator

6. **Set up document folder (optional):**

Add your own documents in the `input/` folder. By default, there is a sample file that you can use to test.

## Running the Bot

1. **Start Ollama** (if not already running):

```bash
ollama serve
```

2. **Run the Discord bot:**

```bash
python main_discord.py
```

The bot will:

- Connect to Discord
- Initialize the vector store for document search
- Begin responding to messages and commands

## License

See repository for license information.

## Contributing

Contributions are welcome! Please open an issue or pull request.

import json
import os
import urllib.request
from enum import Enum

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


class MessageSource(Enum):
    DISCORD_TEXT = 0,
    DISCORD_TEXT_AND_IMAGE = 1,
    DISCORD_VOICE = 2,
    LOCAL = 3


def get_key_from_json_config_file(key_name: str, default: str) -> str | None:
    try:
        with open(_CONFIG_PATH, 'r') as file:
            data = json.load(file).get(key_name)
            if not data:
                return default
            return data  # Get the key value by key name
    except FileNotFoundError:
        print(f"Error: config.json not found at {_CONFIG_PATH}")
    except json.JSONDecodeError:
        print(f"Error: config.json is not valid JSON.")
    except Exception as e:
        print(f"Error reading config: {e}")
    return default


SYSTEM_DESCRIPTION = get_key_from_json_config_file("role_description",
                                                   "You are an AI conversationalist named Lore Compendium, you respond to the user's messages with sophisticated, sardonic, and witty remarks like an English butler.")
THINKING_OLLAMA_MODEL = get_key_from_json_config_file("thinking_ollama_model", "gpt-oss")
FAST_OLLAMA_MODEL = get_key_from_json_config_file("fast_ollama_model", "llama3.2")
EMBEDDING_MODEL = get_key_from_json_config_file("embedding_model", "mxbai-embed-large")
SUPPORTED_EXTENSIONS = ('.docx', '.pdf', '.xlsx', '.csv', '.txt', '.md')
DOC_FOLDER = "./input"
CHROMA_DB_PATH = "./chroma_store"
CHROMA_COLLECTION_NAME = "word_docs_rag"


def check_ollama_health() -> list[str]:
    """
    Checks that Ollama is reachable and that all configured models are installed.
    Returns a list of human-readable error strings; empty means everything is OK.
    """
    errors = []
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5) as resp:
            data = json.loads(resp.read())
            available = {m["name"] for m in data.get("models", [])}
            for model in [THINKING_OLLAMA_MODEL, FAST_OLLAMA_MODEL, EMBEDDING_MODEL]:
                if not any(a == model or a.startswith(model + ":") for a in available):
                    errors.append(f"  Model '{model}' is not installed. Run: ollama pull {model}")
    except OSError:
        errors.append("  Cannot connect to Ollama at http://localhost:11434")
        errors.append("  Make sure Ollama is installed and running.")
    return errors
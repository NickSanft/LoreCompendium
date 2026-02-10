import json
from enum import Enum


class MessageSource(Enum):
    DISCORD_TEXT = 0,
    DISCORD_TEXT_AND_IMAGE = 1,
    DISCORD_VOICE = 2,
    LOCAL = 3


def get_key_from_json_config_file(key_name: str, default: str) -> str | None:
    file_path = "config.json"
    try:
        with open(file_path, 'r') as file:
            data = json.load(file).get(key_name)
            if not data:
                return default
            return data  # Get the key value by key name
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file at {file_path} is not a valid JSON file.")
    except Exception as e:
        print(f"Error reading file: {e}")
    return default


SYSTEM_DESCRIPTION = get_key_from_json_config_file("role_description",
                                                   "You are an AI conversationalist named Lore Compendium, you respond to the user's messages with sophisticated, sardonic, and witty remarks like an English butler.")
THINKING_OLLAMA_MODEL = get_key_from_json_config_file("thinking_ollama_model", "gpt-oss")
FAST_OLLAMA_MODEL = get_key_from_json_config_file("fast_ollama_model", "llama3.2")
EMBEDDING_MODEL = get_key_from_json_config_file("embedding_model", "mxbai-embed-large")

#!/usr/bin/env python3
"""
Configuration Wizard for Lore Compendium
Interactive setup to generate config.json for non-programmers
"""

import json
import os
import sys


def print_header():
    """Print a friendly header."""
    print("\n" + "=" * 50)
    print("  Lore Compendium - Configuration Wizard")
    print("=" * 50)
    print()


def print_section(title):
    """Print a section header."""
    print(f"\n--- {title} ---")


def get_discord_token():
    """Guide user through getting Discord bot token."""
    print_section("Discord Bot Token Setup")
    print("To get a Discord bot token:")
    print("  1. Go to: https://discord.com/developers/applications")
    print("  2. Click 'New Application' and give it a name")
    print("  3. Go to the 'Bot' section on the left")
    print("  4. Click 'Reset Token' to generate a new token")
    print("  5. Copy the token (you won't be able to see it again!)")
    print()
    print("IMPORTANT: Also enable 'Message Content Intent' in the Bot settings!")
    print()

    while True:
        token = input("Paste your Discord bot token here: ").strip()
        if not token:
            print("[ERROR] Token cannot be empty!")
            continue

        # Basic validation - Discord tokens are usually long strings
        if len(token) < 50:
            print("[WARNING] This token looks too short. Discord tokens are usually 70+ characters.")
            confirm = input("Use this token anyway? (y/n): ").strip().lower()
            if confirm != 'y':
                continue

        return token


def get_role_description():
    """Get the bot's personality description."""
    print_section("Bot Personality")
    print("How should your bot behave? Examples:")
    print("  - A helpful assistant")
    print("  - A sarcastic British butler")
    print("  - A wise wizard who speaks in riddles")
    print("  - A friendly librarian")
    print()

    default = "You are an AI conversationalist named Lore Compendium, you respond to the user's messages with sophisticated, sardonic, and witty remarks like an English butler."
    role = input(f"Enter bot personality (or press Enter for default): ").strip()

    if not role:
        print(f"[OK] Using default: {default}")
        return default

    return role


def get_model_settings():
    """Get Ollama model settings."""
    print_section("AI Model Settings")
    print("The bot uses two AI models:")
    print("  1. Thinking model: For complex reasoning (default: gpt-oss)")
    print("  2. Fast model: For quick responses (default: llama3.2)")
    print()

    use_defaults = input("Use default models? (recommended for beginners) (y/n): ").strip().lower()

    if use_defaults == 'y' or use_defaults == '':
        return {
            "thinking_model": "gpt-oss",
            "fast_model": "llama3.2",
            "embedding_model": "mxbai-embed-large"
        }

    thinking = input("Thinking model name (default: gpt-oss): ").strip() or "gpt-oss"
    fast = input("Fast model name (default: llama3.2): ").strip() or "llama3.2"
    embedding = input("Embedding model name (default: mxbai-embed-large): ").strip() or "mxbai-embed-large"

    return {
        "thinking_model": thinking,
        "fast_model": fast,
        "embedding_model": embedding
    }


def create_config():
    """Main configuration creation flow."""
    print_header()
    print("This wizard will help you set up your Lore Compendium bot.")
    print("You'll need a Discord bot token to continue.")
    print()

    proceed = input("Ready to start? (y/n): ").strip().lower()
    if proceed != 'y':
        print("\nSetup cancelled.")
        return False

    # Collect information
    token = get_discord_token()
    role = get_role_description()
    models = get_model_settings()

    # Build config
    config = {
        "discord_bot_token": token,
        "role_description": role,
        "thinking_ollama_model": models["thinking_model"],
        "fast_ollama_model": models["fast_model"],
        "embedding_model": models["embedding_model"]
    }

    # Show summary
    print_section("Configuration Summary")
    print(f"Bot Token: {'*' * 20}...{token[-10:]}")
    print(f"Personality: {role[:80]}{'...' if len(role) > 80 else ''}")
    print(f"Thinking Model: {config['thinking_ollama_model']}")
    print(f"Fast Model: {config['fast_ollama_model']}")
    print(f"Embedding Model: {config['embedding_model']}")
    print()

    confirm = input("Save this configuration? (y/n): ").strip().lower()
    if confirm != 'y':
        print("\nConfiguration not saved.")
        return False

    # Save to file
    try:
        with open('config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        print("\n[OK] Configuration saved to config.json!")
        return True
    except Exception as e:
        print(f"\n[ERROR] Failed to save config.json: {e}")
        return False


def main():
    """Entry point."""
    try:
        # Check if config already exists
        if os.path.exists('config.json'):
            print("\n[WARNING] config.json already exists!")
            overwrite = input("Do you want to overwrite it? (y/n): ").strip().lower()
            if overwrite != 'y':
                print("\nSetup cancelled. Existing config.json preserved.")
                return 0

        success = create_config()

        if success:
            print("\n" + "=" * 50)
            print("  Configuration Complete!")
            print("=" * 50)
            print("\nNext steps:")
            print("  1. Add documents to the 'input' folder")
            print("  2. Run the bot using start.bat (Windows) or ./start.sh (Mac/Linux)")
            print()
            return 0
        else:
            return 1

    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        return 1
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

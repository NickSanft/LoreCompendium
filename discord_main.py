import asyncio
import logging
import queue
import sys
import time
from datetime import datetime
import discord
import os
from discord import app_commands
from discord.ext import commands

from conversation import ask_stuff
from document_engine import (
    query_documents, query_documents_scoped, initialize_vectorstore,
    get_indexed_files, get_chunk_counts, get_duplicate_source, trigger_reindex,
)
from lore_utils import (
    get_key_from_json_config_file, MessageSource, DOC_FOLDER,
    SUPPORTED_EXTENSIONS, check_ollama_health, setup_logging,
)

logger = logging.getLogger(__name__)

_RATE_LIMIT_SECONDS = 10
_MAX_QUERY_LENGTH = 500
_user_last_query: dict[str, float] = {}


def _check_rate_limit(user_id: str) -> float:
    """Returns 0.0 if the user is allowed through, or remaining cooldown seconds if not."""
    now = time.time()
    remaining = _RATE_LIMIT_SECONDS - (now - _user_last_query.get(user_id, 0.0))
    if remaining > 0:
        return remaining
    _user_last_query[user_id] = now
    return 0.0


def _validate_query(query: str) -> str | None:
    """Returns an error message if the query is invalid, or None if it is fine."""
    if not query.strip():
        return "Query cannot be empty."
    if len(query) > _MAX_QUERY_LENGTH:
        return f"Query is too long ({len(query)} chars). Please keep it under {_MAX_QUERY_LENGTH} characters."
    return None


_STREAM_EDIT_INTERVAL = 0.8  # minimum seconds between Discord message edits while streaming


def _fmt_size(n_bytes: int) -> str:
    """Format a byte count as a human-readable string (e.g. 1.4 MB)."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n_bytes) < 1024:
            return f"{n_bytes:.0f} {unit}"
        n_bytes /= 1024  # type: ignore[assignment]
    return f"{n_bytes:.1f} TB"


async def _stream_to_interaction(
    interaction: discord.Interaction,
    stream_q: queue.Queue,
    query_task: asyncio.Task,
) -> None:
    """Drain stream_q and progressively edit the deferred interaction response.

    Edits are throttled to _STREAM_EDIT_INTERVAL seconds to stay within Discord
    rate limits. A blinking cursor (▌) is appended during generation and removed
    on the final edit. Returns once the generator sentinel (None) is received or
    the backing task finishes unexpectedly.
    """
    accumulated = ""
    last_edit = 0.0

    while True:
        # Drain all currently available chunks without blocking
        got_chunk = False
        while True:
            try:
                chunk = stream_q.get_nowait()
            except queue.Empty:
                break
            if chunk is None:  # sentinel — generation finished
                if accumulated:
                    try:
                        await interaction.edit_original_response(content=accumulated[-1950:])
                    except Exception:
                        pass
                return
            accumulated += chunk
            got_chunk = True

        # If the task died before sending the sentinel, stop waiting
        if query_task.done() and stream_q.empty():
            return

        # Throttled intermediate edit with a cursor so the user sees progress
        if got_chunk and accumulated:
            now = time.monotonic()
            if now - last_edit >= _STREAM_EDIT_INTERVAL:
                try:
                    preview = accumulated[-1950:] if len(accumulated) > 1950 else accumulated
                    await interaction.edit_original_response(content=preview + " ▌")
                    last_edit = now
                except Exception:
                    pass

        await asyncio.sleep(0.05)


def _classify_error(e: Exception) -> str:
    """Convert a raw exception into a user-facing error message."""
    msg = str(e).lower()
    if "connection" in msg or "refused" in msg or "timeout" in msg:
        return "⚠️ Cannot reach the Ollama service. Make sure it is running (`ollama serve`)."
    return "⚠️ Something went wrong while processing your request. Please try again."


command_prefix = "$"
intents = discord.Intents.default()
intents.message_content = True
client = commands.Bot(command_prefix=command_prefix, intents=intents)


async def _startup_init():
    """Initialize the vectorstore in the background so the bot is ready for queries."""
    try:
        await asyncio.to_thread(initialize_vectorstore)
        logger.info("Vectorstore initialized and ready.")
    except Exception as e:
        logger.error(f"Failed to initialize vectorstore on startup: {e}")


@client.event
async def on_ready():
    logger.info(f"Logged in as {client.user}")
    if not os.path.exists(DOC_FOLDER):
        os.makedirs("input")
        logger.info("Created 'input' directory.")

    asyncio.create_task(_startup_init())

    try:
        synced = await client.tree.sync()
        logger.info(f"Synced {len(synced)} command(s)")
    except Exception as e:
        logger.error(f"Failed to sync commands: {e}")


@client.tree.command(name="lore", description="Search your documents for an answer")
@app_commands.describe(query="The question you want to ask about your lore")
async def lore_slash(interaction: discord.Interaction, query: str):
    error = _validate_query(query)
    if error:
        await interaction.response.send_message(error, ephemeral=True)
        return

    remaining = _check_rate_limit(str(interaction.user.id))
    if remaining > 0:
        await interaction.response.send_message(
            f"Please wait {remaining:.1f}s before sending another query.", ephemeral=True)
        return

    logger.info(f"Slash lore request from {interaction.user}: {query}")
    await interaction.response.defer(thinking=True)

    stream_q: queue.Queue = queue.Queue()
    query_task = asyncio.ensure_future(
        asyncio.to_thread(query_documents, query, stream_queue=stream_q)
    )
    await _stream_to_interaction(interaction, stream_q, query_task)

    try:
        response = await query_task
    except Exception as e:
        logger.error(f"Lore query error: {e}")
        response = _classify_error(e)
    await chunk_and_send(ctx=None, original_message=None, original_response=response, interaction=interaction)


@client.tree.command(name="ask", description="Search within a specific document")
@app_commands.describe(filename="The document to search in", query="Your question")
async def ask_slash(interaction: discord.Interaction, filename: str, query: str):
    error = _validate_query(query)
    if error:
        await interaction.response.send_message(error, ephemeral=True)
        return

    remaining = _check_rate_limit(str(interaction.user.id))
    if remaining > 0:
        await interaction.response.send_message(
            f"Please wait {remaining:.1f}s before sending another query.", ephemeral=True)
        return

    logger.info(f"Scoped query from {interaction.user}: file='{filename}', query='{query}'")
    await interaction.response.defer(thinking=True)

    stream_q: queue.Queue = queue.Queue()
    query_task = asyncio.ensure_future(
        asyncio.to_thread(query_documents_scoped, query, filename, stream_queue=stream_q)
    )
    await _stream_to_interaction(interaction, stream_q, query_task)

    try:
        response = await query_task
    except Exception as e:
        logger.error(f"Scoped query error: {e}")
        response = _classify_error(e)
    await chunk_and_send(ctx=None, original_message=None, original_response=response, interaction=interaction)


@ask_slash.autocomplete("filename")
async def ask_filename_autocomplete(
    interaction: discord.Interaction, current: str
) -> list[app_commands.Choice[str]]:
    """Populate filename choices from the live index manifest as the user types."""
    indexed = get_indexed_files()
    filenames = sorted({os.path.basename(p) for p in indexed})
    matches = [f for f in filenames if current.lower() in f.lower()]
    return [app_commands.Choice(name=f, value=f) for f in matches[:25]]


@client.tree.command(name="help", description="Show available bot commands")
async def help_slash(interaction: discord.Interaction):
    supported = ", ".join(SUPPORTED_EXTENSIONS)
    help_text = (
        "**Lore Compendium Commands**\n\n"
        "`/lore <query>` — Search all indexed documents for an answer\n"
        "`/ask <filename> <query>` — Search within a specific document (filename autocompletes)\n"
        "`/status` — Show which documents are currently indexed\n"
        "`/reindex` — Force a re-scan of the input folder\n"
        "`/help` — Show this message\n\n"
        "**Conversational Mode**\n"
        "Mention me or send a DM to have a conversation. I can search your documents automatically.\n\n"
        "**Adding Documents**\n"
        f"Drag and drop a file into any channel. Supported formats: {supported}"
    )
    await interaction.response.send_message(help_text, ephemeral=True)


@client.tree.command(name="status", description="Show which documents are currently indexed")
async def status_slash(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    indexed = get_indexed_files()
    if not indexed:
        msg = "No documents are currently indexed. Add files to the `input` folder or drag and drop them here."
    else:
        chunk_counts = await asyncio.to_thread(get_chunk_counts)
        lines = [f"**{len(indexed)} document(s) indexed:**\n"]
        for path, sig in sorted(indexed.items(), key=lambda x: os.path.basename(x[0]).lower()):
            name = os.path.basename(path)
            size_str = _fmt_size(int(sig["size"])) if isinstance(sig.get("size"), (int, float)) else "?"
            mtime = sig.get("mtime")
            ts = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M") if mtime else "unknown"
            chunks = chunk_counts.get(path, 0)
            chunk_str = f"{chunks} chunk{'s' if chunks != 1 else ''}"
            lines.append(f"• `{name}` — {size_str} · {chunk_str} · indexed {ts}")
        msg = "\n".join(lines)
    await interaction.followup.send(msg, ephemeral=True)


@client.tree.command(name="reindex", description="Force a re-scan of the input folder for new or changed documents")
async def reindex_slash(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    try:
        count = await asyncio.to_thread(trigger_reindex)
        if count == 0:
            msg = "No documents found in the `input` folder. Add some files and try again."
        else:
            msg = f"Queued {count} document(s) for re-indexing. Changes will be available shortly."
    except Exception as e:
        msg = f"Error starting re-index: {e}"
    await interaction.followup.send(msg, ephemeral=True)


@client.event
async def on_message(message: discord.Message):
    if message.author == client.user:
        return

    if message.attachments:
        saved_count = 0
        notices: list[str] = []
        for attachment in message.attachments:
            ext = os.path.splitext(attachment.filename)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                save_path = os.path.join(DOC_FOLDER, attachment.filename)
                is_update = os.path.exists(save_path)
                try:
                    await attachment.save(save_path)
                except Exception as e:
                    logger.error(f"Failed to save {attachment.filename}: {e}")
                    notices.append(f"⚠️ Failed to save `{attachment.filename}`: {e}")
                    continue

                # Detect content-identical files already indexed under a different name
                duplicate = get_duplicate_source(save_path)
                if duplicate and duplicate != attachment.filename:
                    notices.append(
                        f"⚠️ `{attachment.filename}` appears to be a duplicate of already-indexed `{duplicate}`."
                    )
                elif is_update:
                    notices.append(f"🔄 Updated existing file `{attachment.filename}`.")

                logger.info(f"Saved {'updated' if is_update else 'new'} file: {save_path}")
                saved_count += 1

        if saved_count > 0 or notices:
            lines = [f"📥 **Received!** Saved {saved_count} document(s) to the `input` folder."]
            if notices:
                lines.extend(notices)
            else:
                lines.append("*These will be indexed shortly.*")
            await message.channel.send("\n".join(lines))
            return

    # Handle Conversational/DM Logic
    channel_type = message.channel
    is_dm = isinstance(channel_type, discord.DMChannel)
    is_mention = client.user.mentioned_in(message)

    if not is_dm and not is_mention:
        return

    message_clean = message.clean_content
    author = message.author.name

    error = _validate_query(message_clean)
    if error:
        await message.channel.send(error)
        return

    remaining = _check_rate_limit(str(message.author.id))
    if remaining > 0:
        await message.channel.send(f"Please wait {remaining:.1f}s before sending another message.")
        return

    logger.info(f"Conversational request from {author}: {message_clean}")

    original_message = await message.channel.send(
        "This may take a few seconds, please wait. This message will be updated with the result!")

    try:
        async with message.channel.typing():
            original_response = await asyncio.to_thread(ask_stuff, message_clean, author, MessageSource.DISCORD_TEXT)
    except Exception as e:
        logger.error(f"Conversation error: {e}")
        original_response = _classify_error(e)

    await chunk_and_send(message.channel, original_message, original_response)


@client.event
async def on_close():
    from document_engine import shutdown
    await asyncio.to_thread(shutdown)


async def chunk_and_send(ctx, original_message, original_response, interaction: discord.Interaction = None):
    resp_len = len(original_response)

    # Determine author name based on interaction or context
    if interaction:
        author = interaction.user.name
    elif hasattr(ctx, 'author'):
        author = ctx.author.name
    else:
        author = "User"

    if resp_len > 2000:
        header = f"Result too long ({resp_len} chars). Sending multiple messages to {author}:\n"
        responses = split_into_chunks(header + original_response)

        for i, chunk in enumerate(responses):
            if i == 0:
                if interaction:
                    await interaction.edit_original_response(content=chunk)
                else:
                    await original_message.edit(content=chunk)
            else:
                if interaction:
                    await interaction.followup.send(content=chunk)
                else:
                    await ctx.send(chunk)
    else:
        if interaction:
            await interaction.edit_original_response(content=original_response)
        else:
            await original_message.edit(content=original_response)


def split_into_chunks(s: str, chunk_size: int = 1990) -> list[str]:
    """Split s into chunks of at most chunk_size characters.

    Prefers to break at paragraph boundaries (double newline), then single
    newlines, then sentence-ending punctuation, before falling back to a hard
    character cut. This prevents words and sentences being split mid-flow.
    """
    if not s:
        return []
    if len(s) <= chunk_size:
        return [s]
    chunks: list[str] = []
    while s:
        if len(s) <= chunk_size:
            chunks.append(s)
            break
        window = s[:chunk_size]
        pos = -1
        for sep in ("\n\n", "\n", ". ", "! ", "? "):
            idx = window.rfind(sep)
            # Accept a break at or past the halfway mark
            if idx >= chunk_size // 2:
                pos = idx + len(sep)
                break
        if pos <= 0:
            pos = chunk_size
        chunks.append(s[:pos].rstrip())
        s = s[pos:].lstrip("\n")
    return [c for c in chunks if c]


if __name__ == '__main__':
    setup_logging(level=logging.DEBUG)
    logger.info("Running pre-flight checks...")
    errors = check_ollama_health()
    if errors:
        logger.error("Pre-flight checks failed:")
        for err in errors:
            logger.error(err)
        logger.error("Fix the above issues and restart the bot.")
        sys.exit(1)
    logger.info("Ollama is running and all models are available.")

    discord_secret = get_key_from_json_config_file("discord_bot_token", "")
    client.run(discord_secret)

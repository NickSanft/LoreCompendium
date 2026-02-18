import discord
import os
from discord import app_commands
from discord.ext import commands

# Ensure you have these modules or that they handle the specific logic
from conversation import ask_stuff
from document_engine import query_documents
from lore_utils import get_key_from_json_config_file, MessageSource, DOC_FOLDER, SUPPORTED_EXTENSIONS

command_prefix = "$"
intents = discord.Intents.default()
intents.message_content = True
client = commands.Bot(command_prefix=command_prefix, intents=intents)


@client.event
async def on_ready():
    print("Logged in as ", client.user)
    if not os.path.exists(DOC_FOLDER):
        os.makedirs("input")
        print("Created 'input' directory.")

    try:
        # This registers the slash commands with Discord
        synced = await client.tree.sync()
        print(f"Synced {len(synced)} command(s)")
    except Exception as e:
        print(f"Failed to sync commands: {e}")


@client.tree.command(name="lore", description="Search your documents for an answer")
@app_commands.describe(query="The question you want to ask about your lore")
async def lore_slash(interaction: discord.Interaction, query: str):
    print(f"Slash Lore request: {query}")
    await interaction.response.defer(thinking=True)
    response = query_documents(query)
    await chunk_and_send(ctx=None, original_message=None, original_response=response, interaction=interaction)


@client.event
async def on_message(message: discord.Message):
    if message.author == client.user:
        return

    if message.attachments:
        saved_count = 0
        for attachment in message.attachments:
            ext = os.path.splitext(attachment.filename)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                save_path = os.path.join(DOC_FOLDER, attachment.filename)
                await attachment.save(save_path)
                print(f"Saved file: {save_path}")
                saved_count += 1

        # If we saved files, let the user know and stop processing (don't treat it as a question)
        if saved_count > 0:
            await message.channel.send(
                f"📥 **Received!** Saved {saved_count} document(s) to the `input` folder.\n*If you have live-sync enabled, these will be indexed shortly.*")
            return

    # 3. Handle Conversational/DM Logic
    channel_type = message.channel
    is_dm = isinstance(channel_type, discord.DMChannel)
    is_mention = client.user.mentioned_in(message)

    if not is_dm and not is_mention:
        return

    message_clean = message.clean_content
    author = message.author.name
    print("Lore request: ", message_clean)

    original_message = await message.channel.send(
        "This may take a few seconds, please wait. This message will be updated with the result!")

    # Pass the clean message to your AI logic
    original_response = ask_stuff(message_clean, author, MessageSource.DISCORD_TEXT)

    # We pass message.channel as 'ctx' because chunk_and_send expects an object with .send()
    await chunk_and_send(message.channel, original_message, original_response)


async def chunk_and_send(ctx, original_message, original_response, interaction: discord.Interaction = None):
    resp_len = len(original_response)

    # Determine author name based on interaction or context
    if interaction:
        author = interaction.user.name
    elif hasattr(ctx, 'author'):
        author = ctx.author.name
    else:
        # Fallback if ctx is just a channel object (from on_message)
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


def split_into_chunks(s, chunk_size=2000):
    return [s[i:i + chunk_size] for i in range(0, len(s), chunk_size)]


if __name__ == '__main__':
    discord_secret = get_key_from_json_config_file("discord_bot_token", "")
    client.run(discord_secret)
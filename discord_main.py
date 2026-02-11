import discord
from discord import app_commands
from discord.ext import commands

from conversation import ask_stuff
from document_engine import query_documents
from lore_utils import get_key_from_json_config_file, MessageSource

command_prefix = "$"
intents = discord.Intents.default()
intents.message_content = True
client = commands.Bot(command_prefix=command_prefix, intents=intents)


@client.event
async def on_ready():
    print("Logged in as ", client.user)
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

    # "Defer" tells Discord we are working on it (shows "Bot is thinking...")
    # This prevents the command from timing out if the LLM is slow.
    await interaction.response.defer(thinking=True)

    # Run the actual query
    response = query_documents(query)

    # Reuse the chunking logic, passing the interaction object
    await chunk_and_send(ctx=None, original_message=None, original_response=response, interaction=interaction)

@client.command()
async def lore(ctx, *, message):
    print("Lore request: %s", message)
    original_message = await ctx.send("This may take a bit, please wait. This message will be updated with the result!")
    original_response = query_documents(message)
    await chunk_and_send(ctx, original_message, original_response)


@client.event
async def on_message(ctx):
    author = ctx.author.name
    channel_type = ctx.channel
    message_clean = ctx.clean_content
    if ctx.author == client.user:
        return
    elif ctx.content.startswith(command_prefix):
        await client.process_commands(ctx)
        return
    elif not isinstance(channel_type, discord.DMChannel) and not client.user.mentioned_in(ctx):
        return
    print("Lore request: ", message_clean)
    original_message = await ctx.channel.send(
        "This may take a few seconds, please wait. This message will be updated with the result!")
    original_response = ask_stuff(message_clean, author, MessageSource.DISCORD_TEXT)
    await chunk_and_send(ctx, original_message, original_response)


async def chunk_and_send(ctx, original_message, original_response, interaction: discord.Interaction = None):
    resp_len = len(original_response)
    author = interaction.user.name if interaction else ctx.author.name

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

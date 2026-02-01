import json

import discord
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
    print("Logged in as %s", client.user)


@client.command()
async def lore(ctx, *, message):
    print("Lore request: %s", message)
    original_message = await ctx.send(
        "This may take a few seconds, please wait. This message will be updated with the result!")
    original_response = query_documents(message)
    resp_len = len(original_response)
    author = ctx.author.name

    if resp_len > 2000:
        response = "The answer was over 2000 ({}), so you're getting multiple messages {} \r\n".format(resp_len,
                                                                                                       author) + original_response
        responses = split_into_chunks(response)
        for i, response in enumerate(responses):
            await ctx.send(response)
    else:
        await original_message.edit(content=original_response)


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
    print("Lore request: %s", message_clean)
    original_message = await ctx.channel.send(
        "This may take a few seconds, please wait. This message will be updated with the result!")
    original_response = ask_stuff(message_clean, author, MessageSource.DISCORD_TEXT)
    resp_len = len(original_response)
    author = ctx.author.name

    if resp_len > 2000:
        response = "The answer was over 2000 ({}), so you're getting multiple messages {} \r\n".format(resp_len,
                                                                                                       author) + original_response
        responses = split_into_chunks(response)
        for i, response in enumerate(responses):
            await ctx.send(response)
    else:
        await original_message.edit(content=original_response)


def split_into_chunks(s, chunk_size=2000):
    return [s[i:i + chunk_size] for i in range(0, len(s), chunk_size)]


if __name__ == '__main__':
    discord_secret = get_key_from_json_config_file("discord_bot_token", "")
    client.run(discord_secret)

import os
import discord
import asyncio
from dotenv import load_dotenv
from discord.channel import DMChannel
from dclite.data import DiscordDataset, DiscordDataSource
from dclite.chatbot import StupidAhhChatBot
from chatterbot import ChatBot

from chatterbot.trainers import ListTrainer

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
DATA_TOKEN = os.getenv("DATA_TOKEN")
DATA_CHANNEL = int(os.getenv("DATA_CHANNEL"))
DM_CHANNEL = int(os.getenv("DM_CHANNEL"))

MESSAGE_SEPARATOR = "[NEWLINE]"

chatterbot = StupidAhhChatBot("chatterbot")

class MyClient(discord.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def on_ready(self):
        print(f'Logged on as {self.user}!')

    async def on_message(self, message: discord.Message):
        if message.author.id == self.user.id:
            return

        if message.channel.id == DM_CHANNEL:
            response = str(chatterbot.respond(message.content)).replace(MESSAGE_SEPARATOR, "\n")
            print(f"replying {response}")
            message = await message.reply(content=response)
            print(f"sent {message}");

def test_chatbot():
    while True:
        user_msg = input("> ")
        response = str(chatterbot.respond(user_msg))
        print(f"bot: {response.replace(MESSAGE_SEPARATOR, "\n")}")

client = MyClient(chunk_guilds_at_startup=False, request_guilds=False)

async def main():

    await client.start(BOT_TOKEN)

    # messages_file = f"data/{DATA_CHANNEL}.txt"

    dataset = DiscordDataset.load("data/parri.json")

    msgs_coalesced = dataset.messages_txt_raw_coalesced(separator=MESSAGE_SEPARATOR)
    chatterbot.train(msgs_coalesced)

    test_chatbot()

asyncio.run(main())

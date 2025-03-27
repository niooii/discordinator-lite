import os
import discord
import asyncio
from dotenv import load_dotenv
from discord.channel import DMChannel
from dclite.data import DiscordDataset, DiscordDataSource

from chatterbot import ChatBot

from chatterbot.trainers import ListTrainer

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
DATA_TOKEN = os.getenv("DATA_TOKEN")
DATA_CHANNEL = int(os.getenv("DATA_CHANNEL"))
DM_CHANNEL = int(os.getenv("DM_CHANNEL"))

chatbot = ChatBot("someguy")

class MyClient(discord.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def on_ready(self):
        print(f'Logged on as {self.user}!')

    async def on_message(self, message: discord.Message):
        if message.author.id == self.user.id:
            return

        if message.channel.id == DM_CHANNEL:
            response = chatbot.get_response(message.content)
            print(f"replying {response}")
            message = await message.reply(content=response)
            print(f"sent {message}");

def test_chatbot():
    while True:
        user_msg = input("> ")
        response = chatbot.get_response(user_msg)
        print(f"bot: {response}")

client = MyClient(chunk_guilds_at_startup=False, request_guilds=False)

async def main():

    await client.login(BOT_TOKEN)

    trainer = ListTrainer(chatbot)

    messages_file = f"data/{DATA_CHANNEL}.txt"

    source = DiscordDataSource(DATA_TOKEN)

    dataset = DiscordDataset.load(messages_file)

    if dataset is None:
        dataset = await source.fetch(
            DATA_CHANNEL, 
            8000, 
            True
        )
        dataset.save(messages_file)

    trainer.train(dataset.messages)

    # await client.connect()

    test_chatbot()

asyncio.run(main())

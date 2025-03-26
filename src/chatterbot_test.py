import os
import discord
import time
import asyncio
import random
from dotenv import load_dotenv
from discord.channel import DMChannel
import ssl 

from chatterbot import ChatBot

from chatterbot.trainers import ListTrainer

load_dotenv()
TOKEN = os.getenv("TOKEN")
DATA_CHANNEL = int(os.getenv("DATA_CHANNEL"))
DM_CHANNEL = int(os.getenv("DM_CHANNEL"))

chatbot = ChatBot("someguy")

class MyClient(discord.Client):
    async def on_message(self, message: discord.Message):
        if message.author.id == self.user.id:
            return

        if message.channel.id == DM_CHANNEL:
            response = chatbot.get_response(message.content)
            print(f"replying {response}")
            message = await message.reply(content=response)
            print(f"sent {message}");

async def main():
    client = MyClient()

    await client.login(TOKEN)

    channel: DMChannel = await client.fetch_channel(DATA_CHANNEL)

    trainer = ListTrainer(chatbot)

    messages: list[str] = []

    messages_file = f"data/{DATA_CHANNEL}.txt"

    if not os.path.isfile(messages_file):
        print("scraping channel")
        async for message in channel.history(limit=8000, oldest_first=True):
            if len(message.content) != 0:
                messages.append(message.content)

        with open(messages_file, "a", encoding="utf-8") as file:
            for message in messages:
                file.write(message + "\n")
                
    else:
        with open(messages_file, "r", encoding="utf-8") as file:
            messages = file.readlines()
        

    trainer.train(messages)

    await client.connect()

asyncio.run(main())

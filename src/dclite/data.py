import json
from typing import Any, Self
import discord
from discord.message import Message
from discord.channel import DMChannel
from collections import deque
import torch
from tqdm.asyncio import tqdm
from discord import Client

class DiscordDataset:
    messages: list[str]

    def __init__(self, messages: list[str]):
        self.messages = messages

    @staticmethod 
    async def scrape(client: discord.Client, channel_id: int, limit: int, oldest_first: bool = True)  -> Self:
        messages = deque([])

        channel: DMChannel = await client.fetch_channel(channel_id)

        print("Fetching messages...");

        last_message: Message = None

        async for message in tqdm(channel.history(limit=limit, oldest_first=oldest_first)):
            should_append_to_last = last_message is not None and last_message.author == message.author
            if len(message.content) != 0:
                if should_append_to_last:
                    existing: str = messages[-1 if oldest_first else 0]
                    new: str = f"{existing}\n{message.content}" if oldest_first else f"{message.content}\n{existing}"
                    messages[-1 if oldest_first else 0] = new
                    continue
                
                if oldest_first:
                    messages.append(message.content)
                else:
                    messages.appendleft(message.content)  
                last_message = message

        return DiscordDataset(list(messages))  
    
    @staticmethod
    def load(path: str) -> Self | None:
        try:
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)
            
            return DiscordDataset(data["messages"])
        except FileNotFoundError:
            return None

    def save(self, path: str):
        data = {}
        data["messages"] = self.messages
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

class DiscordDataSource:
    auth: str
    _logged_in_: bool
    _client_: Client

    def __init__(self, auth: str):
        self.auth = auth
        self._logged_in_ = False
        self._client_ = Client(request_guilds=False, chunk_guilds_at_startup=False)

    async def fetch(self, channel_id: int, limit: int, oldest_first: bool = True) -> DiscordDataset:
        if not self._logged_in_:
            await self._client_.login(self.auth)
            self._logged_in_ = True

        return await DiscordDataset.scrape(
            self._client_, 
            channel_id=channel_id, 
            limit=limit, 
            oldest_first=oldest_first
        )

class Gpt2Dataset(torch.utils.data.Dataset):
    def __init__(self, messages: list[str], tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.inputs = []
        self.labels = []
        
        for msg in messages:
            # "<|user|> user_text <|assistant|> assistant_text"
            # should do this instead
            # formatted_text = f"<|user|> {msg['user']} <|assistant|> {msg['assistant']}"
            formatted_text = msg

            encodings = tokenizer(
                formatted_text, 
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            self.inputs.append(encodings["input_ids"].squeeze())
            
            # for modeling causal language, labels are the same as inputs apparently
            self.labels.append(encodings["input_ids"].squeeze())
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.inputs[idx],
            "labels": self.labels[idx]
        }
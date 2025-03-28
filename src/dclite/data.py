import json
from typing import Any, Self
from collections import deque
import torch
from datetime import datetime
from tqdm.asyncio import tqdm

class DiscordMessage:
    # unix timestamp ms
    sent_at: int
    content: str
    author_name: str
    author_id: int
    id: int
    references_id: int | None

    def __init__(self, message):
        self.sent_at = int(message.created_at.timestamp() * 1000)
        self.content = message.content
        self.author_id = message.author.id
        self.author_name = message.author.name
        self.id = message.id
        self.replied_to = message.reference.message_id if message.reference else None

    def __str__(self) -> str:
        return f"{self.author_name} at {datetime.fromtimestamp(self.sent_at/1000)}\n{self.content}"

class DiscordDataset:
    messages: list[DiscordMessage]
    oldest_first: bool
    length: int

    def __init__(self, messages: list[DiscordMessage], oldest_first: bool, length: int):
        self.messages = messages
        self.oldest_first = oldest_first
        self.length = length

    @staticmethod 
    async def scrape(client, channel_id: int, limit: int, oldest_first: bool = True)  -> Self:
        from discord.message import Message
        from discord.channel import DMChannel
        from discord import Client

        client: Client = client
        
        messages = deque([])

        channel: DMChannel = await client.fetch_channel(channel_id)

        print("Fetching messages...")

        last_message: Message = None

        length = 0
        async for message in tqdm(channel.history(limit=limit, oldest_first=oldest_first)):
            import time
            import random
            # random wait for extremely large dataset downloads
            # if (length - 1) % 100 == 0:
            #     time.sleep(random.uniform(1, 2))

            if oldest_first:
                messages.append(DiscordMessage(message))
            else:
                messages.appendleft(DiscordMessage(message))  
            length += 1

        return DiscordDataset(list(messages), oldest_first, length)  
    
    @staticmethod
    def load(path: str) -> Self | None:
        try:
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)

            messages: list[DiscordMessage] = []
            
            for dict in data["messages"]:
                msg = object.__new__(DiscordMessage)
                msg.__dict__ = dict
                messages.append(msg)
            
            return DiscordDataset(messages, data["oldest_first"], data["length"])
        except FileNotFoundError:
            return None
        
    def messages_as_map(self) -> dict[int, DiscordMessage]:
        return {msg.id: msg for msg in self.messages}
        
    # returns a list of messages that contain contain text (no links, or only media)
    def messages_txt(self) -> list[DiscordMessage]:
        return [msg for msg in self.messages if not "https://" in msg.content and len(msg.content) != 0]
        
    # returns a list of the content of every message without additional data
    def messages_txt_raw(self) -> list[str]:
        return [msg.content for msg in self.messages_txt()]
        
    # returns a list of the content of every message without additional data,
    # with consecutive messages from the same user joined together with a separator
    def messages_txt_raw_coalesced(self, separator: str ="\n") -> list[str]:
        messages = []
        last_message: DiscordMessage = None

        for msg in self.messages:
            if last_message and last_message.author_id == msg.author_id:
                messages[-1] += f"{separator}{msg.content}"
            else:
                messages.append(msg.content)
            
            last_message = msg
            
        return messages

    def save(self, path: str):
        data = {}
        data["length"] = self.length
        data["oldest_first"] = self.oldest_first
        data["messages"] = [msg.__dict__ for msg in self.messages]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

class DiscordDataSource:
    auth: str
    _logged_in_: bool

    def __init__(self, auth: str):
        from discord import Client

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

class DiscordTrainingSet(torch.utils.data.Dataset):
    def __init__(self, dataset: DiscordDataset, tokenizer, max_msg_length=512, max_length=250000):
        self.tokenizer = tokenizer
        self.inputs = []
        self.labels = []

        messages = dataset.messages_txt_raw()
        
        for i, msg in enumerate(messages):
            if i >= max_length:
                break
            
            # "<|user|> user_text <|assistant|> assistant_text"
            # should do this instead
            # formatted_text = f"<|user|> {msg["user"]} <|assistant|> {msg["assistant"]}"
            formatted_text = msg

            encodings = tokenizer(
                formatted_text, 
                truncation=True,
                max_length=max_msg_length,
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
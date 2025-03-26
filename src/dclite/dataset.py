import json

class Dataset:
    messages: list[str]

    def __init__(self, channel_id: int, limit: int | None = None):
        self.channel_id = channel_id
        self.messages = []

    @classmethod
    def load(cls, path: str) -> "Dataset" | None:
        try:
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)
            
            dataset = object.__new__(cls)
            dataset.channel_id = data.get("channel_id", 0)
            dataset.messages = data["messages"]
            return dataset
        except FileNotFoundError:
            return None

    def save(self):
        data = {}
        data["messages"] = self.messages
        data["channels"] = self.messages
        with open('data.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
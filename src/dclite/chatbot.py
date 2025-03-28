from abc import ABC, abstractmethod

class Chatbot(ABC):
    @abstractmethod
    def __init__(self, dir: str):
        pass

    @abstractmethod
    def respond(self, user_input: str) -> str:
        pass

class LLMChatBot(Chatbot):
    pass

class StupidAhhChatBot(Chatbot):
    def __init__(self, db_file: str):
        import chatterbot 
        self.chatbot = chatterbot.ChatBot(
            "stupidahhchatbot", 
            database_uri=f"sqlite:///{db_file}.sqlite"
        )

    def respond(self, user_input: str) -> str:
        return self.chatbot.get_response(user_input)
    
    def train(self, text: list[str]):
        import chatterbot
        trainer = chatterbot.trainers.ListTrainer(self.chatbot)
        trainer.train(text)
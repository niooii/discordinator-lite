import time
import traceback
from dclite.model import ChatModel, ModelType
from dclite.data import Gpt2Dataset, DiscordDataSource, DiscordDataset
import os
from dotenv import load_dotenv
import argparse
import shlex
import cmd
import asyncio

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
DATA_TOKEN = os.getenv("DATA_TOKEN")
DATA_CHANNEL = int(os.getenv("DATA_CHANNEL"))
DM_CHANNEL = int(os.getenv("DM_CHANNEL"))

# override the default exiting behavior of argparse
class ArgParser(argparse.ArgumentParser):
    def exit(self, status=0, message=None):
        raise argparse.ArgumentError(None, message or "")
    
    def error(self, message):
        raise argparse.ArgumentError(None, f"{message}\n{self.usage or ""}")

class DcRepl(cmd.Cmd):
    intro = "ghehehehaw..."
    prompt = "> "
    loop: asyncio.AbstractEventLoop
    data_source: DiscordDataSource
    recent_chatmodel: ChatModel

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loop = asyncio.get_event_loop()
        self.data_source = DiscordDataSource(DATA_TOKEN)
        self.recent_chatmodel = None


    # tweak the cmdloop function to catch exceptions
    def cmdloop(self, intro=None):
        print(self.intro)
        while True:
            try:
                super().cmdloop(intro='')
                break
            except KeyboardInterrupt:
                exit(1)
            except Exception as e:
                print(f"Error: {e}")

    def do_data(self, arg):
        """Download a new DiscordDataset. Usage: data [-c channel_id] [-l limit] [-old/--oldest-first] [-o output_file]"""
        parser = ArgParser(prog="data", description="Download data from a discord channel")
        parser.add_argument("-c", "--channel", type=int, help="Discord channel id")
        parser.add_argument("-l", "--limit", type=int, default=1000, help="Limit the downloaded messages")
        parser.add_argument("-old", "--oldest-first", action="store_true", help="Start downloading from the oldest messages first")
        parser.add_argument("-o", "--output", type=str, help="The output file to store the dataset in")

        args = parser.parse_args(shlex.split(arg))

        dataset = self.loop.run_until_complete(
            self.data_source.fetch(
                args.channel, 
                args.limit, 
                args.oldest_first
            )
        )
        output = args.output or f"data/{args.channel}.json"
        dataset.save(output)

        print(f"Saved data to {output}")

    # doesnt really work rn
    def do_train(self, arg):
        """Train a ChatModel on a DiscordDataset. Usage: train [-f dataset_file] [-e epochs] [-lr learning_rate] [-b batch_size] [-o output_file]"""
        parser = ArgParser(prog="train", description="Finetune a conversational model with discord data")
        parser.add_argument("-f", "--file", type=str, help="The dataset file to use")
        parser.add_argument("-o", "--output", type=str, help="The output file to save the model in")
        parser.add_argument("-e", "--epochs", type=int, default=3, help="Number of training epochs")
        parser.add_argument("-lr", "--learning-rate", type=float, default=5e-5, help="Learning rate for optimization")
        parser.add_argument("-b", "--batch-size", type=int, default=8, help="Training batch size")
        args = parser.parse_args(arg.split())

        self.recent_chatmodel = ChatModel(
            model_type=ModelType.GPT2_MEDIUM,
            device="cuda"
        )

        if args.file:
            dataset = DiscordDataset(args.file)
        else:
            raise ValueError("Dataset file must be specified with -f flag")

        output = args.output or f"models/{time.time()}"

        training_dataset = Gpt2Dataset(
            messages=dataset.messages,
            tokenizer=self.recent_chatmodel.tokenizer,
            max_length=2048
        )
        self.recent_chatmodel.finetune(
            dataset=training_dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            output_dir=output
        )
        print(f"Model saved to {output}")


    def do_chat(self, arg):
        """Enter a chat with a ChatModel. If no model path is specified, defaults to the last model trained. Usage: chat [-f model_file]"""
        parser = ArgParser(prog="chat", description="Enter a chat with a conversational model")
        parser.add_argument("-f", "--file", type=str, help="Path to the model file to load")
        args = parser.parse_args(arg.split())

    def do_quit(self, arg):
        """Exit the application"""
        print("Goodbye!")
        return True
    
    do_exit = do_quit

if __name__ == "__main__":
    DcRepl().cmdloop()

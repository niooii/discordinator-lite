import time
import os
from dotenv import load_dotenv
import argparse
import asyncio

load_dotenv(dotenv_path=".env")
BOT_TOKEN = os.getenv("BOT_TOKEN")
DATA_TOKEN = os.getenv("DATA_TOKEN")
DM_CHANNEL = int(os.getenv("DM_CHANNEL"))

# override the default exiting behavior of argparse
# converted to a cli so keep it
class ArgParser(argparse.ArgumentParser):
    pass
    # def exit(self, status=0, message=None):
    #     raise argparse.ArgumentError(None, message or "")
    
    # def error(self, message):
    #     raise argparse.ArgumentError(None, f"{message}\n{self.usage or ""}")

def data(args):
    """Download a new DiscordDataset"""
    from dclite.data import DiscordDataSource, DiscordDataset
    data_source = DiscordDataSource(DATA_TOKEN)

    if not args.channel:
        raise ValueError("Channel ID must be specified with the -c flag")

    dataset = asyncio.get_event_loop().run_until_complete(
        data_source.fetch(
            args.channel,
            args.limit,
            args.oldest_first
        ) 
    )
    output = args.output or f"data/{args.channel}.json"
    dataset.save(output)
    print(f"Saved data to {output}")

def train(args):
    """Train a ChatModel on a DiscordDataset"""
    from dclite.model import ChatModel, ModelType
    from dclite.data import DiscordTrainingSet, DiscordDataset
    model = ChatModel(
        model_type=ModelType.GPT2,
        device="cuda"
    )
    if args.file:
        dataset = DiscordDataset.load(args.file)
    else:
        raise ValueError("Dataset file must be specified with the -f flag")
    output = args.output or f"models/{int(time.time())}"

    print("Loading dataset...")
    training_dataset = DiscordTrainingSet(
        dataset,
        tokenizer=model.tokenizer,
        max_msg_length=1024,
        max_length=500
    )

    model.finetune(
        dataset=training_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=output
    )
    print(f"Model saved to {output}")
    return output

def chat(args):
    """Enter a chat with a ChatModel"""
    from dclite.model import ChatModel, ModelType
    if args.dir:
        model = ChatModel.from_pretrained(args.dir, device="cuda")
    else:
        raise ValueError("Model directory must be specified with the -d flag")

    while True:
        user_input = input("> ")
        print(model.gen_text(user_input))

def main():
    parser = argparse.ArgumentParser(description="Discordinator CLI")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # data command
    data_parser = subparsers.add_parser("data", help="Download data from a discord channel")
    data_parser.add_argument("-c", "--channel", type=int, required=True, help="Discord channel id")
    data_parser.add_argument("-l", "--limit", type=int, default=1000, help="Limit the downloaded messages")
    data_parser.add_argument("-old", "--oldest-first", action="store_true", help="Start downloading from the oldest messages first")
    data_parser.add_argument("-o", "--output", type=str, help="The output file to store the dataset in")
    data_parser.set_defaults(func=data)
    
    # train command
    train_parser = subparsers.add_parser("train", help="Finetune a conversational model with discord data")
    train_parser.add_argument("-f", "--file", type=str, required=True, help="The dataset file to use")
    train_parser.add_argument("-o", "--output", type=str, help="The output file to save the model in")
    train_parser.add_argument("-e", "--epochs", type=int, default=3, help="Number of training epochs")
    train_parser.add_argument("-lr", "--learning-rate", type=float, default=1e-5, help="Learning rate for optimization")
    train_parser.add_argument("-b", "--batch-size", type=int, default=8, help="Training batch size")
    train_parser.set_defaults(func=train)
    
    # chat command
    chat_parser = subparsers.add_parser("chat", help="Enter a chat with a conversational model")
    chat_parser.add_argument("-d", "--dir", type=str, help="The directory containig the model file to load")
    chat_parser.set_defaults(func=chat)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    try:
        result = args.func(args)
        return result
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    main()
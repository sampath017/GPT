import tiktoken
import settings as s
import pickle
import torch


class DataLoaderLite:
    def __init__(self):
        self.get_tokens()

    def get_tokens(self):
        self.enc = tiktoken.get_encoding("gpt2")
        if not s.tokens_path.exists():
            text = s.data_path.read_text(encoding="utf-8")
            tokens = self.enc.encode(text)
            print(f"Number of tokens: {len(tokens)}")
            with s.tokens_path.open("wb") as f:
                pickle.dump(tokens, f)

            print(f"Tokens saved to: {s.tokens_path.resolve()}")
        else:
            print("Tokens already saved!")
            with s.tokens_path.open("rb") as f:
                tokens = pickle.load(f)
                print(f"Loaded {len(tokens)} tokens from disk")

        tokens = torch.tensor(tokens, dtype=torch.long)
        train_split = int(len(tokens)*s.config["dataset"]["train_split"])
        self.train_data = tokens[:train_split]
        self.val_data = tokens[train_split:]

    def get_batch_optimized(self, split):
        data = self.train_data if split == "train" else self.val_data
        idx = torch.randint(0, len(
            data) - s.config["dataset"]["block_size"], (s.config["dataset"]["batch_size"],))

        offsets = torch.arange(s.config["dataset"]["block_size"]).unsqueeze(0)
        positions = idx.unsqueeze(1) + offsets

        x = data[positions].to(s.config["device"])
        y = data[positions + 1].to(s.config["device"])

        return x, y

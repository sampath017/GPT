import tiktoken
import settings as s
import pickle
import torch


class DatasetLite:
    def __init__(self):
        self.get_tokens()

    def get_tokens(self):
        B, T = s.config["dataset"]["batch_size"], s.config["dataset"]["block_size"]
        self.enc = tiktoken.get_encoding("gpt2")
        if not s.tokens_path.exists():
            text = s.shakespear_data_path.read_text(encoding="utf-8")
            tokens = self.enc.encode(text)
            with s.tokens_path.open("wb") as f:
                pickle.dump(tokens, f)

            if s.ddp_master_process:
                print(f"Number of tokens: {len(tokens)}")
                print(f"Tokens saved to: {s.tokens_path.resolve()}")
        else:
            if s.ddp_master_process:
                print("Tokens already saved!")
            with s.tokens_path.open("rb") as f:
                tokens = pickle.load(f)
                if s.ddp_master_process:
                    print(f"Loaded {len(tokens)} tokens from disk")

        self.tokens = torch.tensor(tokens, dtype=torch.long, device=s.device)
        if s.ddp_master_process:
            print(
                f"1 Epoch = {(len(tokens) - 1) // (B*T)} batches.")


class DataLoaderLite:
    def __init__(self, dataset, split="train"):
        self.dataset = dataset
        self.split = split
        self.B, self.T = s.config["dataset"]["batch_size"], s.config["dataset"]["block_size"]

        self.current_position = self.B * self.T * s.ddp_global_rank
        train_split = int(len(self.dataset.tokens) *
                          s.config["dataset"]["train_split"])
        train_data = self.dataset.tokens[:train_split]
        val_data = self.dataset.tokens[train_split:]

        self.tokens = train_data if self.split == "train" else val_data

    def next_batch(self):
        buf = self.tokens[self.current_position: self.current_position +
                          self.B*self.T + 1]
        x = (buf[:-1]).reshape(self.B, self.T)
        y = (buf[1:]).reshape(self.B, self.T)

        # advance the position in the tensor
        self.current_position += self.B * self.T * s.ddp_world_size
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (self.B * self.T * s.ddp_world_size + 1) > len(self.tokens):
            self.current_position = self.B * self.T * s.ddp_global_rank

        return x, y

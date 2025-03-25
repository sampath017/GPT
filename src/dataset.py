import numpy as np
import torch
from torch.utils.data import Dataset
import settings as s
import tiktoken
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path


def get_dataloader(data_path, split="train", ddp=False):
    dataset = ShakespearDataset(data_path/"shakespear.txt")

    train_dataset, val_dataset = random_split(
        dataset, [s.dataset["train_split"], s.dataset["val_split"]]
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=s.dataset["batch_size"],
        shuffle=not ddp,
        sampler=DistributedSampler(train_dataset) if ddp else None
    )

    val_dataloader = DataLoader(
        val_dataset, batch_size=s.dataset["batch_size"])

    dataloader = train_dataloader if split == "train" else val_dataloader

    return dataloader


class ShakespearDataset(Dataset):
    # ~3M tokens
    def __init__(self, path):
        self.path = path
        self.tokens_file = path.parent / "tokens.pt"

        self.setup()
        self.save_tokens()
        self.load_tokens()

    def setup(self):
        self.text = self.path.read_text()
        self.encoder = tiktoken.encoding_for_model("gpt-2")

    def _encode(self, string):
        return self.encoder.encode(string)

    def _decode(self, l):
        return self.encoder.decode(l)

    def save_tokens(self):
        if not self.tokens_file.exists():
            tokens = torch.tensor(self._encode(self.text))
            torch.serialization.add_safe_globals(tokens)
            torch.save(tokens, self.tokens_file)

    def load_tokens(self):
        self.tokens = torch.load(self.tokens_file, weights_only=True)

    def __len__(self):
        return len(self.tokens) - (s.dataset["context_size"] + 1)

    def __getitem__(self, idx):
        if idx < 0:
            idx = len(self) + (idx + 1)

        if idx <= len(self):
            x = self.tokens[idx:idx + s.dataset["context_size"]]
            y = self.tokens[idx + 1:idx + s.dataset["context_size"] + 1]

            return x, y
        else:
            raise IndexError(
                f"Index {idx} out of range for data length {len(self.tokens)} with block size {s.dataset['context_size']}")

    def __repr__(self): return f"Dataset({s.dataset['context_size']=})"


def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)

    return ptt


class DataLoaderLite:
    def __init__(self, data_path, process_rank, num_processes, split, master_process):
        self.data_path = data_path
        self.B, self.T = s.dataset["batch_size"], s.dataset["context_size"]
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        shards = [s for s in self.data_path.iterdir() if split in s.name]
        shards = sorted(shards)
        self.shards = shards

        assert len(shards) > 0, f"no shards found for split {split}"

        if master_process:
            print(f"found {len(shards)} shards for split {split}")

        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes

        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank

        return x, y

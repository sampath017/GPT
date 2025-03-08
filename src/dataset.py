import torch
from torch.utils.data import Dataset
import settings as s
import tiktoken
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler


def get_dataloader(data_path, type="train", ddp=False):
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

    return train_dataloader


class ShakespearDataset(Dataset):
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
                f"Index {idx} out of range for data length {len(self.tokens)} with block size {s.dataset["context_size"]}")

    def __repr__(self): return f"Dataset({s.dataset["context_size"]=})"

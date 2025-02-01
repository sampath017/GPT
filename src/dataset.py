import torch
from torch.utils.data import Dataset, DataLoader, random_split


class ShakespearDataset(Dataset):
    def __init__(self, path, context_size=8):
        self.path = path
        self.context_size = context_size
        self.tokens_file = path.parent / "tokens.pt"

        self.setup()
        self.save_tokens()
        self.load_tokens()

    def setup(self):
        self.text = self.path.read_text()
        chars = sorted(set(self.text))

        self.num_chars = len(chars)
        self.ctoi = {c: i for i, c in enumerate(chars)}
        self.itoc = {i: c for i, c in enumerate(chars)}
        self.encode = lambda s: [self.ctoi[c] for c in s]
        self.decode = lambda l: "".join([self.itoc[i] for i in l])

    def save_tokens(self):
        if not self.tokens_file.exists():
            data = torch.tensor(self.encode(self.text))
            torch.serialization.add_safe_globals(data)

            torch.save(data, self.tokens_file)

    def load_tokens(self):
        self.data = torch.load(self.tokens_file, weights_only=True)

    def __len__(self):
        return len(self.data) - (self.context_size + 1)

    def __getitem__(self, idx):
        if idx < 0:
            idx = len(self) + (idx + 1)

        if idx <= len(self):
            x = self.data[idx:idx + self.context_size]
            y = self.data[idx + 1:idx + self.context_size + 1]

            return x, y
        else:
            raise IndexError(
                f"Index {idx} out of range for data length {len(self.data)} with block size {self.context_size}")

    def __repr__(self): return f"Dataset({self.context_size=})"

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import lightning as L


class ShakespearDataset(Dataset):
    def __init__(self, path, block_size=8):
        self.path = path
        self.block_size = block_size
        self.tokens_file = path.parent / "tokens.pt"

        self.setup()
        self.save_tokens()
        self.load_tokens()

    def setup(self):
        self.text = self.path.read_text()
        chars = "".join(sorted(set(self.text)))

        self.num_chars = len(chars)
        self.ctoi = {c: i for i, c in enumerate(chars)}
        self.itoc = {i: c for c, i in self.ctoi.items()}
        self.encode = lambda s: [self.ctoi[c] for c in s]
        self.decode = lambda l: "".join([self.itoc[i] for i in l])

    def save_tokens(self):
        if not self.tokens_file.exists():
            data = torch.tensor(self.encode(self.text))

            torch.save(data, self.tokens_file)

    def load_tokens(self):
        self.data = torch.load(self.tokens_file)

    def __len__(self):
        return len(self.data) - (self.block_size + 1)

    def __getitem__(self, idx):
        if idx < 0:
            idx = len(self) + (idx + 1)

        if idx <= len(self):
            x = self.data[idx:idx + self.block_size]
            y = self.data[idx + 1:idx + self.block_size + 1]

            return x, y
        else:
            raise IndexError(
                f"Index {idx} out of range for data length {len(self.data)} with block size {self.block_size}")

    def __repr__(self): return f"Dataset({self.block_size=})"


class ShakespearDataModule(L.LightningDataModule):
    def __init__(self, path, block_size=8, batch_size=64):
        super().__init__()
        self.path = path
        self.block_size = block_size
        self.batch_size = batch_size

    def prepare_data(self):
        ShakespearDataset(self.path, self.block_size)

    def setup(self, stage):
        if stage == "fit":
            self.dataset = ShakespearDataset(self.path, self.block_size)
            train_size = int(len(self.dataset) * 0.7)
            val_size = len(self.dataset) - train_size

            self.train_dataset, self.val_dataset = random_split(
                self.dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

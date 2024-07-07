import torch
from torch.utils.data import Dataset, DataLoader, random_split
import lightning as L


class NamesDataset(Dataset):
    def __init__(self, path, block_size):
        self.path = path
        self.block_size = block_size
        self.tokens_file = path.parent / "tokens.pt"

        self.setup()
        self.save_tokens()
        self.load_tokens()

    def setup(self):
        self.names = self.path.read_text().splitlines()
        self.chars = sorted(set("".join(self.names)))

        self.start_char = "<S>"
        self.end_char = "<E>"
        self.ctoi = {self.start_char: 0}
        self.ctoi.update({c: i for i, c in enumerate(self.chars, 1)})
        self.ctoi[self.end_char] = len(self.ctoi)

        self.itoc = {i: c for c, i in self.ctoi.items()}

        self.num_chars = len(self.itoc)

    def save_tokens(self):
        if not self.tokens_file.exists():
            x = []
            y = []
            for name in self.names:
                context = [0] * self.block_size
                for ch in list(name) + [self.end_char]:
                    ix = self.ctoi[ch]
                    x.append(context)
                    y.append(ix)
                    context = context[1:] + [ix]  # crop and append

            torch.save([torch.tensor(x), torch.tensor(y)],
                       self.tokens_file)

    def load_tokens(self):
        self.x, self.y = torch.load(self.tokens_file)

    def __len__(self): return len(self.names)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __repr__(self):
        return f"Dataset({self.num_chars=})"


class NamesDataModule(L.LightningDataModule):
    def __init__(self, path, block_size, batch_size):
        super().__init__()
        self.path = path
        self.block_size = block_size
        self.batch_size = batch_size

    def prepare_data(self):
        NamesDataset(self.path, self.block_size)

    def setup(self, stage):
        if stage == "fit":
            self.dataset = NamesDataset(self.path, self.block_size)
            train_size = int(len(self.dataset) * 0.7)
            val_size = len(self.dataset) - train_size

            self.train_dataset, self.val_dataset = random_split(
                self.dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

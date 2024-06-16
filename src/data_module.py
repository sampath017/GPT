import torch
from torch.utils.data import Dataset, DataLoader, random_split
import lightning as L


class NamesDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.names = self.path.read_text().splitlines()

        self.chars = set("".join(self.names))

        self.start_char = "<S>"
        self.end_char = "<E>"
        ctoi = {self.start_char: 0}
        ctoi.update({c: i for i, c in enumerate(self.chars, 1)})
        ctoi[self.end_char] = len(ctoi)

        itoc = {i: c for c, i in ctoi.items()}

        self.num_chars = len(itoc)

        self.encode = lambda str: [ctoi[char] for char in str]
        self.decode = lambda num_list: "".join([itoc[num] for num in num_list])

        x = []
        y = []
        for name in self.names:
            for ch1, ch2 in zip(name, name[1:]):
                ix1 = ctoi[ch1]
                ix2 = ctoi[ch2]

                x.append(ix1)
                y.append(ix2)

        self.x = torch.tensor(x)
        self.y = torch.tensor(y)

    def __len__(self): return len(self.names)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __repr__(self):
        return f"Dataset({self.num_chars=})"


class NamesDataModule(L.LightningDataModule):
    def __init__(self, path, batch_size):
        super().__init__()
        self.path = path
        self.batch_size = batch_size

    def setup(self, stage):
        if stage == "fit":
            self.dataset = NamesDataset(self.path)
            train_size = int(len(self.dataset) * 0.7)
            val_size = len(self.dataset) - train_size

            self.train_dataset, self.val_dataset = random_split(
                self.dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

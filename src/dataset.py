import torch
from torch.utils.data import Dataset


class NamesDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.names = self.path.read_text().splitlines()

        self.chars = set("".join(self.names))
        self.num_chars = len(self.chars)
        ctoi = {c: i for i, c in enumerate(self.chars)}
        itoc = {i: c for c, i in ctoi.items()}

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

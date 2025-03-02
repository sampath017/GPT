from models import GPT
from quickai.module import QuickModule, ToyNet
import torch.nn.functional as F
from quickai.utils import accuracy
import settings as s


class GPTModule(QuickModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.device = self.device

    def forward(self, batch):
        self.model = self.model.to(self.device)
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        logits = self.model(x)
        B, T, C = logits.shape
        logits = logits.reshape(B*T, C)
        y = y.reshape(B*T)

        loss = F.cross_entropy(logits, y)
        acc = accuracy(logits, y)

        return loss, acc

    def training_step(self, batch):
        loss, acc = self.forward(batch)

        return loss, acc

    def validation_step(self, batch):
        loss, acc = self.forward(batch)

        return loss, acc

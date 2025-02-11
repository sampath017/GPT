from models import BigramLanguageModel
from quickai.module import QuickModule, ToyNet
import torch.nn.functional as F
from quickai.utils import accuracy
import settings as s


class BigramLanguageModule(QuickModule):
    def __init__(self, vocab_size=None, num_embds=None, num_heads=None, head_size=None, model=None, toy_model=False):
        super().__init__()
        if model:
            self.model = model
        elif toy_model:
            self.model = ToyNet()
        else:
            self.model = BigramLanguageModel(
                vocab_size=vocab_size,
                num_embds=num_embds,
                head_size=head_size,
                num_heads=num_heads,
                context_size=s.dataset["context_size"]
            )
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

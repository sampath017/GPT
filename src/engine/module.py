import torch
from torch import nn
import torch.nn.functional as F

import lightning as L


class NamesModule(L.LightningModule):
    def __init__(self, num_chars, block_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_chars, 10)
        self.model = nn.Sequential(
            nn.Linear(block_size*10, 200),
            nn.ReLU(),

            nn.Linear(200, num_chars)
        )

    def forward(self, batch, batch_idx):
        x, y = batch
        embs = self.embedding_table(x)
        embs = embs.reshape(embs.shape[0], -1)
        logits = self.model(embs)
        loss = F.cross_entropy(logits, y)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self(batch, batch_idx)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self(batch, batch_idx)

        self.log("val_loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())

        return optimizer

import torch
from torch import nn
import torch.nn.functional as F

import lightning as L


class Head(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size
        tril = torch.tril(torch.ones(self.block_size, self.block_size))
        weights = torch.zeros_like(tril)
        weights = weights.masked_fill(tril == 0, -torch.inf)
        self.register_buffer('weights', F.softmax(weights, dim=-1))

    def forward(self, x):
        result = self.weights.matmul(x)

        return result


class ShakespeareModule(L.LightningModule):
    def __init__(self, num_chars, block_size):
        super().__init__()
        self.num_chars = num_chars
        self.model = nn.Sequential(
            nn.Embedding(num_chars, num_chars),
            Head(block_size)
        )

    def forward(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, self.num_chars), y.reshape(-1))

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

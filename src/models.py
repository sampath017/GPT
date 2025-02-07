import torch
import torch.nn.functional as F
from torch import nn


class Head(nn.Module):
    def __init__(self, num_embds, head_size, context_size):
        super().__init__()
        self.key = nn.Linear(num_embds, head_size, bias=False)
        self.query = nn.Linear(num_embds, head_size, bias=False)
        self.value = nn.Linear(num_embds, head_size, bias=False)

        self.register_buffer('tril', torch.tril(
            torch.ones(context_size, context_size)))

    def forward(self, x):
        _, T, E = x.shape
        k = self.key(x)
        q = self.query(x)

        # compute attention scores ("affinities")
        wei = q.matmul(k.permute(0, 2, 1)) * E**-0.5
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        # perform the weighted aggregation of the values
        v = self.value(x)
        out = wei.matmul(v)

        return out


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, num_embds, head_size, context_size):
        super().__init__()
        self.context_size = context_size
        self.token_embedding_table = nn.Embedding(vocab_size, num_embds)
        self.position_embedding_table = nn.Embedding(context_size, num_embds)
        self.sa_head = Head(num_embds, head_size, context_size)

    def forward(self, x):
        _, T = x.shape
        token_embds = self.token_embedding_table(x)
        position_embds = self.position_embedding_table(torch.arange(T))

        x = token_embds + position_embds
        logits = self.sa_head(x)

        return logits

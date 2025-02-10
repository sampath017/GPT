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


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, num_embds, context_size):
        super().__init__()
        self.heads = [Head(num_embds, head_size, context_size)
                      for _ in range(num_heads)]
        self.proj = nn.Linear(num_heads*head_size, num_embds)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)

        return self.proj(x)


class FeedForward(nn.Module):
    def __init__(self, num_embds):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(num_embds, num_embds*4),
            nn.ReLU(),
            nn.Linear(num_embds*4, num_embds),
            nn.ReLU()
        )

    def forward(self, x):
        return self.ff(x)


class Block(nn.Module):
    def __init__(self, vocab_size, num_heads=None, num_embds=None, head_size=None, context_size=None):
        super().__init__()
        self.sa_heads = MultiHeadAttention(
            num_heads, head_size, num_embds, context_size)
        self.ff = FeedForward(num_embds)

    def forward(self, x):
        x = self.sa_heads(x)
        logits = self.ff(x)

        return logits


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, num_heads=None, num_embds=None, head_size=None, context_size=None):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, num_embds)
        self.position_embedding_table = nn.Embedding(context_size, num_embds)
        self.blocks = nn.Sequential(
            *[Block(vocab_size, num_heads, num_embds, head_size, context_size) for _ in range(3)]
        )
        self.lm_head = nn.Linear(num_embds, vocab_size)

    def forward(self, x):
        _, T = x.shape
        token_embds = self.token_embedding_table(x)
        position_embds = self.position_embedding_table(torch.arange(T, device=self.device))

        x = token_embds + position_embds

        x = self.blocks(x)
        logits = self.lm_head(x)

        return logits

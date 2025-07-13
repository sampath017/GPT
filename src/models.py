import torch
import torch.nn as nn
import torch.nn.functional as F
import settings as s

vocab_size = s.config["dataset"]["vocab_size"]
block_size = s.config["dataset"]["block_size"]
num_embds = s.config["model"]["num_embds"]
head_size = s.config["model"]["head_size"]
num_heads = s.config["model"]["num_heads"]
num_blocks = s.config["model"]["num_blocks"]


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.keys = nn.Linear(num_embds, head_size, bias=False)
        self.queries = nn.Linear(num_embds, head_size, bias=False)
        self.values = nn.Linear(num_embds, head_size, bias=False)

        tril = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("tril", tril)

    def forward(self, embds):
        B, T, C = embds.shape
        q = self.queries(embds)
        k = self.keys(embds)
        v = self.values(embds)

        weights = q @ k.transpose(-2, -1) * (C ** -0.5)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)

        out = weights @ v

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Head() for _ in range(num_heads)])

    def forward(self, embds):
        return torch.cat([head(embds) for head in self.heads], dim=-1)


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()

        self.ff = nn.Sequential(
            nn.Linear(num_embds, num_embds*4),
            nn.ReLU(),
            nn.Linear(num_embds*4, num_embds)
        )

    def forward(self, affinities):
        return self.ff(affinities)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = MultiHeadAttention()
        self.ff = FeedForward()
        self.ln1 = nn.LayerNorm(num_embds)
        self.ln2 = nn.LayerNorm(num_embds)

    def forward(self, embds):
        embds = embds + self.sa(self.ln1(embds))
        affinities = embds + self.ff(self.ln2(embds))

        return affinities


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, num_embds)
        self.position_embedding_table = nn.Embedding(block_size, num_embds)
        self.blocks = nn.Sequential(*[Block() for _ in range(num_blocks)])
        self.lm_head = nn.Linear(head_size*num_heads, vocab_size)

    def forward(self, x, y=None):
        B, T = x.shape

        token_embds = self.token_embedding_table(x)  # (B, T, num_embds)
        pos_embds = self.position_embedding_table(
            torch.arange(T))  # (T, num_embds)
        embds = token_embds + pos_embds  # (B, T, num_embds)

        affinities = self.blocks(embds)
        logits = self.lm_head(affinities)  # (B, T, vocab_size)

        loss = None
        if y is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), y.view(B * T))

        return logits, loss

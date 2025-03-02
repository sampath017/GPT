import torch
import torch.nn.functional as F
from torch import nn
import settings as s


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.key = nn.Linear(
            s.model["num_embds"],
            s.model["head_size"],
            bias=False
        )
        self.query = nn.Linear(
            s.model["num_embds"],
            s.model["head_size"],
            bias=False
        )
        self.value = nn.Linear(
            s.model["num_embds"],
            s.model["head_size"],
            bias=False
        )

        self.register_buffer('tril', torch.tril(
            torch.ones(s.dataset["context_size"], s.dataset["context_size"])))

        self.dropout = nn.Dropout(s.model["dropout"])

    def forward(self, x):
        _, T, E = x.shape
        k = self.key(x)
        q = self.query(x)

        # compute attention scores ("affinities")
        wei = q.matmul(k.permute(0, 2, 1)) * E**-0.5
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        v = self.value(x)
        out = wei.matmul(v)

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Head()
                                   for _ in range(s.model["num_heads"])])

        self.proj = nn.Linear(
            s.model["num_heads"]*s.model["head_size"], s.model["num_embds"])
        self.dropout = nn.Dropout(s.model["dropout"])

    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)

        return self.proj(x)


class FeedForward(nn.Module):
    def __init__(self, num_embds):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(num_embds, num_embds*4),
            nn.GELU(),
            nn.Linear(num_embds*4, num_embds),
            nn.Dropout(s.model["dropout"]),
        )

    def forward(self, x):
        return self.ff(x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(s.model["num_embds"])
        self.sa_heads = MultiHeadAttention()
        self.ln2 = nn.LayerNorm(s.model["num_embds"])
        self.ff = FeedForward(s.model["num_embds"])

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))
        logits = x + self.ff(self.ln2(x))

        return logits


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            token_embedding_table=nn.Embedding(
                s.dataset["vocab_size"], s.model["num_embds"]),
            position_embedding_table=nn.Embedding(
                s.dataset["context_size"], s.model["num_embds"]),
            blocks=nn.ModuleList([Block()
                                 for _ in range(s.model["num_blocks"])]),
            ln=nn.LayerNorm(s.model["num_embds"])
        ))

        self.lm_head = nn.Linear(s.model["num_embds"], s.dataset["vocab_size"])

    def forward(self, x):
        _, T = x.shape
        token_embds = self.transformer.token_embedding_table(x)
        position_embds = self.transformer.position_embedding_table(
            torch.arange(T, device=self.device))

        x = token_embds + position_embds
        for block in self.transformer.blocks:
            x = block(x)
        x = self.transformer.ln(x)
        x = self.lm_head(x)

        return x

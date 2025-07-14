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
dropout = s.config["model"]["dropout"]
batch_size = s.config["dataset"]["batch_size"]


class CasualSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(num_embds, num_embds * 3, bias=False)
        self.proj = nn.Linear(num_embds, num_embds)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embds):
        k, q, v = self.qkv(embds).chunk(3, dim=-1)

        k = k.reshape(batch_size, block_size, num_heads,
                      head_size).transpose(1, 2)
        q = q.reshape(batch_size, block_size, num_heads,
                      head_size).transpose(1, 2)
        v = v.reshape(batch_size, block_size, num_heads,
                      head_size).transpose(1, 2)

        tril = torch.tril(torch.ones(block_size, block_size, device=s.device))
        weights = q @ k.transpose(-2, -1) * (num_embds ** -0.5)
        weights = weights.masked_fill(
            tril[:block_size, :block_size] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        affinities = weights @ v                      # [B, nh, block_size, hs]
        affinities = affinities.transpose(1, 2).reshape(
            batch_size, block_size, num_embds)

        return self.dropout(self.proj(affinities))


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()

        self.ff = nn.Sequential(
            nn.Linear(num_embds, num_embds*4),
            nn.GELU(),
            nn.Linear(num_embds*4, num_embds),
            nn.Dropout(dropout),
        )

    def forward(self, affinities):
        return self.ff(affinities)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = CasualSelfAttention()
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
        self.transformer = nn.ModuleDict(dict(
            token_embedding_table=nn.Embedding(vocab_size, num_embds),
            position_embedding_table=nn.Embedding(block_size, num_embds),
            blocks=nn.Sequential(*[Block() for _ in range(num_blocks)]),
            ln_f=nn.LayerNorm(num_embds)
        ))
        self.lm_head = nn.Linear(head_size*num_heads, vocab_size)

    def forward(self, x, y=None):
        B, T = x.shape

        token_embds = self.transformer.token_embedding_table(
            x)  # type: ignore # (B, T, num_embds)
        pos_embds = self.transformer.position_embedding_table(
            # type: ignore # (T, num_embds)
            torch.arange(T, device=s.config["device"]))
        embds = token_embds + pos_embds  # (B, T, num_embds)

        affinities = self.transformer.ln_f(
            self.transformer.blocks(embds))  # type: ignore
        logits = self.lm_head(affinities)  # (B, T, vocab_size)

        loss = None
        if y is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), y.view(B * T))

        return logits, loss

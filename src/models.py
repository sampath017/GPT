import torch
import torch.nn.functional as F
from torch import nn
import settings as s
import math


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        assert s.model["num_embds"] % s.model["num_heads"] == 0
        self.c_attn = nn.Linear(s.model["num_embds"], 3 * s.model["num_embds"])
        self.c_proj = nn.Linear(s.model["num_embds"], s.model["num_embds"])
        self.register_buffer("bias", torch.tril(torch.ones(s.dataset["context_size"], s.dataset["context_size"])).reshape(
            1, 1, s.dataset["context_size"], s.dataset["context_size"]))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(s.model["num_embds"], dim=2)
        k = k.view(B, T, s.model["num_heads"], C //
                   s.model["num_heads"]).transpose(1, 2)
        q = q.view(B, T, s.model["num_heads"], C //
                   s.model["num_heads"]).transpose(1, 2)
        v = v.view(B, T, s.model["num_heads"], C //
                   s.model["num_heads"]).transpose(1, 2)

        # General Attention
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v

        # Flash Attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)

        return y


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(s.model["num_embds"], s.model["num_embds"]*4),
            nn.GELU(),
            nn.Linear(s.model["num_embds"]*4, s.model["num_embds"]),
        )

    def forward(self, x):
        return self.ff(x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(s.model["num_embds"])
        self.attention_block = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(s.model["num_embds"])
        self.ff = FeedForward()

    def forward(self, x):
        x = x + self.attention_block(self.ln1(x))
        logits = x + self.ff(self.ln2(x))

        return logits


class GPT(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
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

        # weight sharing scheme
        self.transformer.token_embedding_table.weight = self.lm_head.weight

    def forward(self, x):
        _, T = x.shape
        token_embds = self.transformer.token_embedding_table(x)
        position_embds = self.transformer.position_embedding_table(
            torch.arange(T, device=self.device))

        x = token_embds + position_embds
        for block in self.transformer.blocks:
            x = block(x)
        x = self.transformer.ln(x)
        logits = self.lm_head(x)

        return logits
